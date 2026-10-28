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
   unchanged) and an adverse-reaction table that mentions placebo;
3. of those, with a measured human Ki (``=``, nM, pChEMBL < 5) at any anchor.

The best case adds every newly scorable drug, as a positive, to the P34.4
labelled positives; P34 closes if no liability reaches the 8-positive floor.
No liability label is computed for any drug, so no outcome is seen.

    python scripts/p34_6_exhausted.py

Network: the openFDA label cache (local) and ChEMBL activity.json for the
pChEMBL < 5 Ki band at the 11 anchors, cached under $NC_V2_CACHE/fresh/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_universe as U  # noqa: E402  (P33.1 helpers, unchanged)

V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
CACHE = U.CACHE / "fresh"


def weak_ki(targets: dict[str, str]) -> dict[str, list[str]]:
    """anchor -> parents with a measured Ki pChEMBL < 5, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "anchor_ki_below5.json"
    if f.exists():
        return json.loads(f.read_text())
    out: dict[str, set[str]] = defaultdict(set)
    for sym, tid in targets.items():
        q = (f"activity.json?target_chembl_id={tid}&pchembl_value__lt=5&standard_type=Ki"
             f"&standard_relation==&standard_units=nM&only=parent_molecule_chembl_id&limit=1000")
        for a in U._pages(q, "activities"):
            if a.get("parent_molecule_chembl_id"):
                out[sym].add(a["parent_molecule_chembl_id"])
    res = {s: sorted(out[s]) for s in targets}
    f.write_text(json.dumps(res))
    return res


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

    weak = weak_ki(snap["targets"])
    per_anchor = {s: {"with_label": len(set(v) & with_label), "placebo_table": len(set(v) & placebo)}
                  for s, v in weak.items()}
    any_ki = set().union(*map(set, weak.values()))
    scorable = sorted(any_ki & placebo)
    liab_receptors = {"eps": ["DRD2"], "sedation": ["HRH1"],
                      "anticholinergic": ["CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5"],
                      "orthostatic_hypotension": ["ADRA1A", "ADRA1B", "ADRA1D"]}
    per_liability = {L: len(set().union(*(set(weak[g]) for g in gs)) & placebo)
                     for L, gs in liab_receptors.items()}
    p344 = json.loads((ROOT / "results" / "P34.4.json").read_text())
    max_pos = {L: p344["labelled_counts_upper_bound_on_common_set"][L]["labelled_pos"] + n
               for L, n in per_liability.items()}

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
        "anchor_ki_below_pki5": per_anchor,
        "max_scorable_per_liability": per_liability,
        "scorable_drugs": [{"chembl_id": p, "inn": pref.get(p),
                            "anchors": sorted(s for s, v in weak.items() if p in v)} for p in scorable],
        "power_floor": {"positives": 8, "negatives": 8},
        # Best case: every P34.4 labelled positive survives the Ki filter and
        # every newly scorable drug is positive.
        "max_positives_if_widened": max_pos,
        "result": "EMPTY",
        "gap": ("approved-drug universe exhausted for the frozen candidate: even if every "
                "remaining scorable drug were positive, no liability reaches 8 positives "
                f"({json.dumps(max_pos)})"),
    }
    assert all(v < 8 for v in max_pos.values()), "a liability could reach the floor: do not close P34"
    (ROOT / "results" / "P34.6.json").write_text(json.dumps(out, indent=1) + "\n")
    print("TASK_RESULT: EMPTY", json.dumps(per_liability))
    return 0


if __name__ == "__main__":
    sys.exit(main())
