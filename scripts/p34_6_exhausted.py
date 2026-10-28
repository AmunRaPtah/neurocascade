#!/usr/bin/env python3
"""P34.6: is there any approved drug left that could widen the fresh set?

Feasibility check after P34.4 and P34.4b returned EMPTY and the human asked to
widen the fresh pool. The frozen P34.3 candidate scores a drug for a liability
only if it has a measured Ki at that liability's receptors
(``candidate.LIABILITY_RECEPTORS``: DRD2, HRH1, CHRM1-5, ADRA1A/B/D, all P33.1
anchors). Approved drugs with an anchor Ki/Kd pChEMBL >= 6 are already P33.1
candidates (v2), and those at 5-6 are already the P34.4 fresh pool. This counts
what is left, from inputs only:

1. approved parents (ChEMBL_37 snapshot, as P33.1) in neither set;
2. of those, with a single-ingredient FDA Rx label (P33.1 ``label_index``,
   unchanged); how many also have a placebo adverse-reaction table is recorded;
3. of those and the 37 fresh drugs, which have a measured ``ki_nM`` at each
   liability's receptors in the unchanged ``scripts/v2_binding.py`` build
   (ChEMBL Ki, GtoPdb pKi, PDSP Ki), the exact field the candidate reads.

The best case counts every P34.4 labelled positive plus every newly scorable
drug as a positive; a liability can reach the floor only with >= 8 positives
and >= 16 scorable drugs. No liability label is computed for any new drug, so
no outcome is seen.

    python scripts/p34_6_exhausted.py

Network: the openFDA label cache (local) and the v2_binding sources (ChEMBL,
GtoPdb, PDSP), built once into $NC_V2_CACHE/fresh/p34_6_binding.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_universe as U  # noqa: E402  (P33.1 helpers, unchanged)

sys.path.insert(0, str(ROOT / "src"))
from neurocascade.benchmark.v2.candidate import LIABILITY_RECEPTORS  # noqa: E402

V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
CACHE = U.CACHE / "fresh"


def binding_ki(drugs: list[dict]) -> dict[str, set[str]]:
    """inn -> panel genes with a measured ki_nM, from the frozen v2_binding build.

    Runs ``scripts/v2_binding.py`` unchanged (ChEMBL Ki, GtoPdb pKi and PDSP Ki,
    merged by its own rules) on ``drugs``, writing to the cache instead of
    v2/inputs; the candidate reads exactly this ``ki_nM``."""
    import v2_binding as B
    CACHE.mkdir(parents=True, exist_ok=True)
    universe = json.loads((V2 / "drugs.json").read_text())
    B.DRUGS_FILE = CACHE / "p34_6_drugs.json"
    B.OUT = CACHE / "p34_6_binding.json"
    B.DRUGS_FILE.write_text(json.dumps({"anchor_targets": universe["anchor_targets"], "drugs": drugs}))
    if not B.OUT.exists():
        assert B.main([]) == 0
    built = json.loads(B.OUT.read_text())["drugs"]
    return {inn: {g for g, t in row.get("targets", {}).items()
                  if t.get("ki_nM") is not None and float(t["ki_nM"]) > 0}
            for inn, row in built.items()}


def main() -> int:
    snap = json.loads((U.CACHE / "chembl-universe-ChEMBL_37.json").read_text())
    mols, hits = snap["mols"], snap["hits"]
    parent_of = {c: (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or c
                 for c, m in mols.items()}
    names: dict[str, str] = {}
    atc: dict[str, set[str]] = defaultdict(set)
    pref: dict[str, str] = {}
    for c, m in mols.items():
        p = parent_of[c]
        if m.get("pref_name"):
            names.setdefault(m["pref_name"].upper(), p)
            if c == p:
                pref[p] = m["pref_name"].lower()
        atc[p].update(m.get("atc_classifications") or [])
    parents = set(parent_of.values())
    p331 = {p for p in parents if any(a.startswith("N") for a in atc[p])} | \
           {p for p in hits if p in parents}
    pool = {m["chembl_id"] for m in json.loads((V2 / "truth_fresh" / "pool.json").read_text())["members"]}
    rest = parents - p331 - pool

    labels = U.label_index(names, rest)
    with_label = {p for p in rest if labels.get(p)}
    placebo = {p for p in with_label
               if any(lab["has_ar_table"] and lab["mentions_placebo"] for lab in labels[p])}

    fresh = json.loads((V2 / "truth_fresh" / "drugs.json").read_text())["drugs"]
    new = [{"inn": pref.get(p) or p, "chembl_id": p} for p in sorted(with_label)]
    ki = binding_ki(new + [{"inn": d["inn"], "chembl_id": d["chembl_id"]} for d in fresh])
    inn_of = {d["chembl_id"]: d["inn"] for d in new}
    p344 = json.loads((ROOT / "results" / "P34.4.json").read_text())
    per_liability = {}
    for L, genes in LIABILITY_RECEPTORS.items():
        new_ki = sorted(p for p in with_label if ki.get(inn_of[p], set()) & set(genes))
        fresh_ki = sorted(d["inn"] for d in fresh if ki.get(d["inn"], set()) & set(genes))
        best_pos = p344["labelled_counts_upper_bound_on_common_set"][L]["labelled_pos"] + len(new_ki)
        per_liability[L] = {
            "receptors": list(genes),
            "new_with_ki": len(new_ki),
            "new_with_ki_and_placebo_table": len(set(new_ki) & placebo),
            "fresh_with_ki": len(fresh_ki),
            "max_positives": best_pos,
            "max_scorable": len(new_ki) + len(fresh_ki),
            "could_reach_floor": best_pos >= 8 and len(new_ki) + len(fresh_ki) >= 16,
            "new_drugs": [{"chembl_id": p, "inn": inn_of[p], "placebo_table": p in placebo,
                           "genes": sorted(ki[inn_of[p]] & set(genes))} for p in new_ki],
        }
    feasible = sorted(L for L, v in per_liability.items() if v["could_reach_floor"])

    out = {
        "task": "P34.6",
        "question": "can the fresh set be widened to the 8/8 power floor with the frozen candidate?",
        "inputs_only": True,
        "n_approved_parents": len(parents),
        "n_p33_1_candidates": len(p331),
        "n_p34_4_pool": len(pool),
        "n_remaining": len(rest),
        "n_remaining_with_rx_label": len(with_label),
        "n_remaining_with_placebo_ar_table": len(placebo),
        "placebo_table_by_atc2": dict(Counter(
            (sorted(a[:3] for a in atc[p]) or ["none"])[0] for p in placebo).most_common()),
        "ki_source": "scripts/v2_binding.py unchanged (ChEMBL + GtoPdb + PDSP)",
        "per_liability": per_liability,
        "power_floor": {"positives": 8, "negatives": 8},
        "feasible_liabilities": feasible,
        "result": "WIDENING_FEASIBLE" if feasible else "EMPTY",
        "gap": None if feasible else (
            "approved-drug universe exhausted for the frozen candidate: even if every newly "
            "scorable drug were positive, no liability reaches the floor"),
    }
    (ROOT / "results" / "P34.6.json").write_text(json.dumps(out, indent=1) + "\n")
    print("TASK_RESULT:", out["result"], json.dumps({L: [v["max_positives"], v["max_scorable"]] for L, v in per_liability.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
