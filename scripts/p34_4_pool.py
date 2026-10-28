#!/usr/bin/env python3
"""P34.4 stage 1: the fresh held-out pool and its drugs.json.

Applies the inclusion rule of docs/prereg-p34-go.md mechanically:

- approved small molecules (ChEMBL max_phase 4, ChEMBL_37 as used by P33.1);
- NOT among the 465 P33.1 candidates (recomputed from the P33.1 snapshot with
  the P33.1 rule: any ATC code under N, or an anchor Ki/Kd pChEMBL >= 6);
- with an anchor-target Ki/Kd pChEMBL >= 5.0, or an ATC code under
  N05, N06, N03, N07, A03, A04, G04BD, C02CA, R06.

Every pool member is written to truth_fresh/pool.json, including the ones
later excluded. Pool members with a single-ingredient FDA prescription label
(matched by the P33.1 ``v2_universe.label_index``, unchanged) go into
truth_fresh/drugs.json in the v2 drugs.json schema; the others are excluded
for a missing input (no FDA label).

Imports nothing from the candidate. Network: ChEMBL activity.json for the
pChEMBL 5-6 band at the 11 anchor targets, cached under
$NC_V2_CACHE/fresh/.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_universe as U  # noqa: E402  (P33.1 helpers, unchanged)

V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
FRESH = V2 / "truth_fresh"
CACHE = U.CACHE / "fresh"
ATC_PREFIXES = ("N05", "N06", "N03", "N07", "A03", "A04", "G04BD", "C02CA", "R06")
PCHEMBL_MIN_FRESH = 5.0


def band_hits(targets: dict[str, str]) -> dict[str, dict[str, float]]:
    """parent id -> {anchor: best pChEMBL} for Ki/Kd in [5.0, 6.0), cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "anchor_hits_5to6.json"
    if f.exists():
        return json.loads(f.read_text())
    hits: dict[str, dict[str, float]] = defaultdict(dict)
    for sym, tid in targets.items():
        q = (f"activity.json?target_chembl_id={tid}&pchembl_value__gte={PCHEMBL_MIN_FRESH}"
             f"&pchembl_value__lt={U.PCHEMBL_MIN}&standard_type__in=Ki,Kd"
             f"&only=parent_molecule_chembl_id,pchembl_value&limit=1000")
        n = 0
        for a in U._pages(q, "activities"):
            n += 1
            p, v = a.get("parent_molecule_chembl_id"), float(a["pchembl_value"])
            if p and v > hits[p].get(sym, 0):
                hits[p][sym] = v
        print(f"  {sym} {tid}: {n} activities in [5, 6)", flush=True)
    f.write_text(json.dumps(hits))
    return hits


def main() -> int:
    snap = json.loads((U.CACHE / "chembl-universe-ChEMBL_37.json").read_text())
    mols, targets, hits6 = snap["mols"], snap["targets"], snap["hits"]
    parent_of = {c: (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or c
                 for c, m in mols.items()}
    names: dict[str, str] = {}
    atc: dict[str, set[str]] = defaultdict(set)
    for c, m in mols.items():
        if m.get("pref_name"):
            names.setdefault(m["pref_name"].upper(), parent_of[c])
        atc[parent_of[c]].update(m.get("atc_classifications") or [])
    parents = set(parent_of.values())
    p331 = {p for p in parents if any(a.startswith("N") for a in atc[p])} | \
           {p for p in hits6 if p in parents}
    assert len(p331) == 465, len(p331)

    hits5 = band_hits(targets)
    by_atc = {p for p in parents - p331 if any(a.startswith(ATC_PREFIXES) for a in atc[p])}
    by_pki = {p for p in parents - p331 if p in hits5}
    pool = by_atc | by_pki
    labels = U.label_index(names, pool)

    v2_inns = {d["inn"] for d in json.loads((V2 / "drugs.json").read_text())["drugs"]}
    members, drugs = [], []
    for p in sorted(pool):
        m = mols.get(p) or {}
        inn = (m.get("pref_name") or "").lower()
        labs = labels.get(p, [])
        sel = sorted(({"atc"} if p in by_atc else set()) | ({"anchor_pki_ge5"} if p in by_pki else set()))
        excluded = None
        if not inn:
            excluded = "missing input: no ChEMBL preferred name"
        elif inn in v2_inns:
            excluded = "missing input: name collides with a v2 universe drug"
        elif not labs:
            excluded = "missing input: no single-ingredient FDA prescription label"
        members.append({"chembl_id": p, "inn": inn or None, "atc": sorted(atc[p]),
                        "selected_by": sel, "anchor_pki": hits5.get(p, {}),
                        "n_labels": len(labs), "excluded": excluded})
        if excluded:
            continue
        nda = [x for x in labs if (x["application_number"] or "").startswith("NDA")]
        drugs.append({
            "inn": inn, "chembl_id": p, "atc": sorted(atc[p]),
            "first_approval": m.get("first_approval"), "selected_by": sel,
            "anchor_pki": hits5.get(p, {}),
            "n_labels": len(labs), "n_nda_labels": len(nda),
            "any_label_mentions_placebo": any(x["mentions_placebo"] for x in labs),
            "labels": sorted(labs, key=lambda x: (not (x["application_number"] or "").startswith("NDA"),
                                                  x["application_number"] is None,
                                                  -(int(x["effective_time"] or 0))))[:8],
            "v1": None,
        })

    meta = json.loads((U.CACHE / "labels.meta.json").read_text())
    FRESH.mkdir(parents=True, exist_ok=True)
    rule = ("approved small molecules (ChEMBL_37 max_phase 4) not among the 465 P33.1 candidates, "
            "with anchor Ki/Kd pChEMBL >= 5.0 or ATC in " + ", ".join(ATC_PREFIXES))
    (FRESH / "pool.json").write_text(json.dumps({
        "description": "P34.4 fresh held-out pool, every member listed (docs/prereg-p34-go.md). "
                       "Written by scripts/p34_4_pool.py.",
        "rule": rule, "chembl_release": "ChEMBL_37", "n_pool": len(members),
        "n_included": len(drugs), "members": members}, indent=1) + "\n")
    (FRESH / "drugs.json").write_text(json.dumps({
        "description": "P34.4 fresh held-out universe, v2 drugs.json schema. Every drug is "
                       "fresh held-out; none is in benchmark/v2/drugs.json.",
        "chembl_release": "ChEMBL_37",
        "anchor_targets": {s: {"uniprot": U.ANCHORS[s], "chembl": t} for s, t in targets.items()},
        "pchembl_min": PCHEMBL_MIN_FRESH, "openfda_label_export": meta["export_date"],
        "n_candidates": len(pool), "drugs": drugs}, indent=1) + "\n")
    print(len(members), "pool members;", len(by_atc), "by ATC;", len(by_pki), "by pKi>=5;",
          len(drugs), "with an FDA Rx label;", sum(d["any_label_mentions_placebo"] for d in drugs),
          "mention placebo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
