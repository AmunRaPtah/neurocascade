#!/usr/bin/env python3
"""Assemble the v2 benchmark drug universe (P33.1, stage 1).

Why a new universe
------------------
The P28.1 list is every ATC N05A/N05B/N06A entry. About two thirds of those
drugs have no FDA label and so no measured human liability outcome, which is
why P30.1/P30.2 scored only ~40 drugs. The liabilities the occupancy model
predicts are anchored on receptors, not indications (H1 sedation, muscarinic
anticholinergic burden, alpha1 orthostasis, D2 EPS, 5-HT2C weight gain), so
v2 selects drugs by what they bind and by whether a human outcome can exist:

  candidates = approved small molecules (ChEMBL max_phase 4, parent form) that
               are EITHER in ATC chapter N (nervous system) OR have a human
               Ki/Kd with pChEMBL >= 6 at one of the anchor receptors;
  kept       = candidates with at least one single-ingredient FDA prescription
               label in the openFDA bulk index (scripts/v2_label_index.py).

The anchor-receptor criterion uses model INPUTS only (binding), never outcomes,
so it cannot leak liability truth into the selection. Drugs whose first label
mentions placebo are flagged, since only those can give placebo-subtracted
labels (P33.2); the rest stay in for PK and binding stages.

Held-out membership is NOT written anywhere: it is guards.v2_heldout(inn).
Each record carries its P28.1 membership and split so that drugs already seen
in P19-P30 design can be reported separately (design exposure, see the
2026-10-12 audit).

    scripts/v2_universe.py     # network (ChEMBL REST) + local label index
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "src" / "neurocascade" / "benchmark"
OUT = BENCH / "v2" / "drugs.json"
CACHE = Path(os.environ.get("NC_V2_CACHE", "/root/nc-v2-cache"))
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

# Anchor receptors by UniProt accession; resolved to ChEMBL target ids at run
# time so the id is never typed from memory.
ANCHORS = {
    "DRD2": "P14416", "HRH1": "P35367",
    "CHRM1": "P11229", "CHRM2": "P08172", "CHRM3": "P20309", "CHRM4": "P08173",
    "CHRM5": "P08912",
    "ADRA1A": "P35348", "ADRA1B": "P35368", "ADRA1D": "P25100",
    "HTR2C": "P28335",
}
PCHEMBL_MIN = 6.0


def _get(url: str):
    err = None
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # noqa: BLE001 - transient network errors
            err = e
        time.sleep(3 * (i + 1))
    raise err


def _pages(path: str, key: str):
    url = f"{CHEMBL}/{path}"
    while url:
        page = _get(url)
        yield from page[key]
        nxt = page["page_meta"]["next"]
        url = f"https://www.ebi.ac.uk{nxt}" if nxt else None


def approved_molecules() -> dict[str, dict]:
    """Every max_phase-4 small molecule (salts included), keyed by ChEMBL id."""
    only = "molecule_chembl_id,pref_name,atc_classifications,molecule_hierarchy,first_approval"
    q = f"molecule.json?max_phase=4&molecule_type=Small%20molecule&only={only}&limit=1000"
    return {m["molecule_chembl_id"]: m for m in _pages(q, "molecules")}


def resolve_targets() -> dict[str, str]:
    out = {}
    for sym, acc in ANCHORS.items():
        ts = _get(f"{CHEMBL}/target.json?target_components__accession={acc}"
                  f"&target_type=SINGLE%20PROTEIN&organism=Homo%20sapiens")["targets"]
        if len(ts) != 1:
            raise RuntimeError(f"{sym} {acc}: expected one target, got {[t['target_chembl_id'] for t in ts]}")
        out[sym] = ts[0]["target_chembl_id"]
    return out


def anchor_hits(targets: dict[str, str]) -> dict[str, dict[str, float]]:
    """parent ChEMBL id -> {anchor symbol: best pChEMBL (Ki/Kd)}."""
    hits: dict[str, dict[str, float]] = defaultdict(dict)
    for sym, tid in targets.items():
        q = (f"activity.json?target_chembl_id={tid}&pchembl_value__gte={PCHEMBL_MIN}"
             f"&standard_type__in=Ki,Kd&only=parent_molecule_chembl_id,pchembl_value&limit=1000")
        n = 0
        for a in _pages(q, "activities"):
            n += 1
            p = a.get("parent_molecule_chembl_id")
            v = float(a["pchembl_value"])
            if p and v > hits[p].get(sym, 0):
                hits[p][sym] = v
        print(f"  {sym} {tid}: {n} activities", flush=True)
    return hits


SALTS = {"HYDROCHLORIDE", "DIHYDROCHLORIDE", "HYDROBROMIDE", "MALEATE", "MESYLATE",
         "MESILATE", "SULFATE", "TARTRATE", "BITARTRATE", "ACETATE", "SODIUM", "POTASSIUM",
         "CALCIUM", "CITRATE", "FUMARATE", "HEMIFUMARATE", "SUCCINATE", "PHOSPHATE",
         "BESYLATE", "BROMIDE", "CHLORIDE", "OXALATE", "LACTATE", "PAMOATE", "DECANOATE",
         "ENANTHATE", "TOSYLATE", "NAPSYLATE", "MALATE", "MONOHYDRATE", "HEMIHYDRATE",
         "DIHYDRATE", "TRIHYDRATE", "SESQUIHYDRATE", "ANHYDROUS", "HYDRATE", "LAURYL",
         "LAUROXIL", "DIHYDROGEN", "HYDROGEN"}


def _strip_salt(name: str) -> str:
    words = name.split()
    while len(words) > 1 and words[-1] in SALTS:
        words.pop()
    return " ".join(words)


def _name_segment(spl: str) -> str:
    """Brand and generic name at the start of spl_product_data_elements.

    The field reads "Brand Generic INGREDIENT MOIETY EXCIPIENT ...": names in
    mixed case, then upper-case ingredient and excipient lists. Matching the
    whole field hits excipients (phenol in Acthar, chloroform in a carisoprodol
    label), so only the words before the first upper-case word after the brand
    are searched.
    """
    words = spl.split()
    for i, w in enumerate(words[1:], 1):
        if len(w) >= 3 and w.isupper():
            words = words[:i]
            break
    # Brand plus the first words of the generic name. Mixed-case excipients
    # can follow (phenol in an Anzemet label), and a kit's second active can
    # trail ("Diclofenac Sodium Topical with Capsaicin").
    return " ".join(words[:4])


def label_index(names: dict[str, str], candidates: set[str]) -> dict[str, list[dict]]:
    """parent id -> single-ingredient Rx labels for it.

    Harmonised labels (openfda block present) match on substance_name, exactly
    or with salt words stripped. Unharmonised labels (no openfda block) match
    when the ingredient names in spl_product_data_elements resolve to exactly
    one candidate parent; two or more means a combination product, skipped.
    Only candidate names in the brand/generic segment are searched, because the
    field also lists excipients that are approved molecules too.
    """
    from v2_label_index import iter_labels
    by_name = {**{_strip_salt(n): p for n, p in names.items()}, **names}
    alts = sorted((n for n, p in by_name.items() if len(n) >= 5 and p in candidates),
                  key=len, reverse=True)
    word = re.compile(r"\b(" + "|".join(re.escape(n) for n in alts) + r")\b")
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for lab in iter_labels():
        of = lab["openfda"]
        if of:
            subs = of.get("substance_name") or []
            if len(subs) != 1:
                continue
            s = subs[0].strip().upper()
            parent, how = by_name.get(s) or by_name.get(_strip_salt(s)), "substance_name"
        else:
            head = _name_segment(" ".join(lab.get("spl_product_data_elements") or []))
            found = {by_name[m] for m in word.findall(head.upper())}
            combo = re.search(r"\b(and|with|plus)\b|/|,|\w-\w", head, re.I)
            parent = found.pop() if len(found) == 1 and not combo else None
            how = "spl_product_data_elements"
        if not parent:
            continue
        text = " ".join((lab.get("adverse_reactions_table") or [])
                        + (lab.get("adverse_reactions") or []))
        by_parent[parent].append({
            "set_id": lab["set_id"], "version": lab.get("version"),
            "effective_time": lab.get("effective_time"),
            "application_number": (of.get("application_number") or [None])[0],
            "route": of.get("route"), "matched_on": how,
            "has_ar_table": bool(lab.get("adverse_reactions_table")),
            "mentions_placebo": "placebo" in text.lower(),
        })
    return by_parent


def main() -> int:
    status = _get(f"{CHEMBL}/status.json")
    release = f"{status['chembl_db_version']} ({status['chembl_release_date']})"
    print("ChEMBL", release, flush=True)
    snap = CACHE / f"chembl-universe-{status['chembl_db_version']}.json"
    if snap.exists():
        cached = json.loads(snap.read_text())
        mols, targets, hits = cached["mols"], cached["targets"], cached["hits"]
    else:
        mols = approved_molecules()
        targets = resolve_targets()
        hits = anchor_hits(targets)
        snap.write_text(json.dumps({"mols": mols, "targets": targets, "hits": hits}))
    print(len(mols), "approved small molecules (incl. salts)", flush=True)

    parent_of = {cid: (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or cid
                 for cid, m in mols.items()}
    names: dict[str, str] = {}
    atc: dict[str, set[str]] = defaultdict(set)
    for cid, m in mols.items():
        p = parent_of[cid]
        if m.get("pref_name"):
            names.setdefault(m["pref_name"].upper(), p)
        atc[p].update(m.get("atc_classifications") or [])
    parents = set(parent_of.values())

    candidates = {p for p in parents if any(c.startswith("N") for c in atc[p])} | \
                 {p for p in hits if p in parents}
    print(len(candidates), "candidates (ATC N or anchor pKi>=6)", flush=True)

    labels = label_index(names, candidates)
    v1 = {d["chembl_id"]: d for d in json.loads((BENCH / "drugs.json").read_text())["drugs"]}
    v1_split = json.loads((BENCH / "split.json").read_text())
    v1_held = set(v1_split["heldout"])

    drugs = []
    for p in sorted(candidates):
        labs = labels.get(p, [])
        if not labs:
            continue
        m = mols.get(p) or {}
        inn = (m.get("pref_name") or "").lower()
        if not inn:
            continue
        nda = [x for x in labs if (x["application_number"] or "").startswith("NDA")]
        v1d = v1.get(p)
        drugs.append({
            "inn": inn, "chembl_id": p, "atc": sorted(atc[p]),
            "first_approval": m.get("first_approval"),
            "selected_by": sorted(({"atc_N"} if any(c.startswith("N") for c in atc[p]) else set())
                                  | ({"anchor_pki"} if p in hits else set())),
            "anchor_pki": hits.get(p, {}),
            "n_labels": len(labs), "n_nda_labels": len(nda),
            "any_label_mentions_placebo": any(x["mentions_placebo"] for x in labs),
            "labels": sorted(labs, key=lambda x: (not (x["application_number"] or "").startswith("NDA"),
                                                  x["application_number"] is None,
                                                  -(int(x["effective_time"] or 0))))[:8],
            "v1": None if not v1d else {"inn": v1d["inn"],
                                        "split": "heldout" if v1d["inn"] in v1_held else "train"},
        })
    meta = json.loads((CACHE / "labels.meta.json").read_text())
    out = {
        "description": "v2 benchmark universe (P33.1 stage 1). Inclusion rule in the "
                       "docstring of scripts/v2_universe.py. Held-out membership is the "
                       "rule guards.v2_heldout(inn) and is deliberately not stored.",
        "chembl_release": release,
        "anchor_targets": {s: {"uniprot": ANCHORS[s], "chembl": t} for s, t in targets.items()},
        "pchembl_min": PCHEMBL_MIN,
        "openfda_label_export": meta["export_date"],
        "n_candidates": len(candidates),
        "drugs": drugs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(len(drugs), "drugs with an FDA prescription label;",
          sum(d["any_label_mentions_placebo"] for d in drugs), "mention placebo;",
          sum(1 for d in drugs if d["v1"]), "were in P28.1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
