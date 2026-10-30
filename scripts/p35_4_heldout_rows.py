#!/usr/bin/env python3
"""P35.4: freeze the SIDER rows of the P35 held-out drugs for the host dump.

``python scripts/p35_4_heldout_rows.py fetch [--cache DIR]``
    Network. Downloads the two SIDER 4.1 files (sha256-checked against
    ``sider_labels.SIDER_FILES``) and, for every SIDER stereo id under a flat id
    that equals or shares connectivity with a P35 held-out drug's PubChem CID
    (``sider/cid_map.json``, P35.2), its PubChem title, parent (or component)
    CIDs and synonyms, exactly as ``scripts/p35_2_build.py fetch`` did for train.

``python scripts/p35_4_heldout_rows.py build [--cache DIR]``
    Offline. Matches each P35 held-out drug to its SIDER ids with
    ``sider_labels.match_stereo`` (the P35.2 rule, unchanged) and writes
    ``v2/truth/p35_sider_heldout_rows.json``: per held-out drug the match, the
    record flags and every mapped-PT ``meddra_freq`` row. **No label is computed
    or printed here.** Labels are built from this file only by
    ``go_dump_p35.build_dump`` under ``NC_HOST_EVAL``.

The output file has no ``drugs`` table, so ``pipeline/guards.hide_heldout``
deletes it from every later session tree (its unknown-shape rule), and once on
main ``guards.frozen_paths`` freezes it.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neurocascade.benchmark.v2 import load_v2_drugs  # noqa: E402
from neurocascade.benchmark.v2 import sider_labels as sl  # noqa: E402
from neurocascade.benchmark.v2.go_dump_p35 import HELDOUT_ROWS, ROWS_KEY  # noqa: E402
from p35_2_build import PUG, _cache, _cids, _flat_to_stereo, _get, _sider  # noqa: E402


def _heldout_flats(flat: dict[int, set[int]]) -> dict[str, list[int]]:
    cmap = sl.load_cid_map()
    out = {}
    for d in load_v2_drugs():
        if sl.p35_heldout(d["inn"]):
            r = cmap[d["inn"]]
            own = set(r["pubchem_cids"]) | set(r["pubchem_same_connectivity_cids"])
            out[d["inn"]] = sorted(f for f in flat if f in own)
    return out


def fetch(argv: list[str]) -> None:
    cache = _cache(argv)
    cache.mkdir(parents=True, exist_ok=True)
    for name in sl.SIDER_FILES:
        if not (cache / name).exists():
            (cache / name).write_bytes(_get(sl.SIDER_MIRROR + name))
    flat = _flat_to_stereo(*_sider(cache))
    pf = cache / "stereo_heldout.json"
    st = json.loads(pf.read_text()) if pf.exists() else {}
    for inn, flats in _heldout_flats(flat).items():
        for fl in flats:
            for s in sorted(flat[fl]):
                if str(s) in st:
                    continue
                t = json.loads(_get(f"{PUG}cid/{s}/property/Title/JSON"))["PropertyTable"]["Properties"][0]
                par, kind = _cids(_get(f"{PUG}cid/{s}/cids/JSON?cids_type=parent")), "parent"
                if not par:
                    par, kind = _cids(_get(f"{PUG}cid/{s}/cids/JSON?cids_type=component")), "component"
                b = _get(f"{PUG}cid/{s}/synonyms/JSON")
                syn = json.loads(b)["InformationList"]["Information"][0].get("Synonym", []) if b else []
                st[str(s)] = {"title": t.get("Title"), "parents": par, "parent_kind": kind,
                              "synonyms": syn}
                time.sleep(0.25)
        pf.write_text(json.dumps(st, indent=0, sort_keys=True))
    print(f"stereo records cached: {len(st)}")


def build(argv: list[str]) -> None:
    cache = _cache(argv)
    freq, se = _sider(cache)
    flat = _flat_to_stereo(freq, se)
    stereo = json.loads((cache / "stereo_heldout.json").read_text())
    parents = {int(k): v["parents"] for k, v in stereo.items()}
    synonyms = {int(k): v["synonyms"] for k, v in stereo.items()}
    all_inns = [d["inn"] for d in load_v2_drugs()]
    cmap = sl.load_cid_map()
    match = {}
    for inn in sorted(_heldout_flats(flat)):
        r = cmap[inn]
        m = sl.match_stereo(inn, r["pubchem_cids"], r["pubchem_same_connectivity_cids"], flat,
                            parents, synonyms, all_inns)
        m["stereo_rejected"] = [{"cid": s, "title": stereo[str(s)]["title"],
                                 "parents": stereo[str(s)]["parents"]} for s in m["stereo_rejected"]]
        m["stereo_titles"] = {str(s): stereo[str(s)]["title"] for s in m["stereo"]}
        match[inn] = m
    # A SIDER stereo id belongs to one drug only, across train and held-out.
    owner = {s: inn for inn, row in cmap.items() if "sider" in row for s in row["sider"]["stereo"]}
    for inn, m in match.items():
        for s in m["stereo"]:
            if owner.setdefault(s, inn) != inn:
                raise ValueError(f"SIDER stereo id {s} matches both {owner[s]} and {inn}")
    rows = {inn: {"sider_match": m, "has_record": False, "n_freq_rows": 0, "n_all_se_rows": 0,
                  "mapped_rows": [], "listed_pts": set()} for inn, m in match.items()}
    seen = set()
    for r in freq:
        inn = owner.get(r["stereo"])
        if inn not in rows or r["flat"] not in match[inn]["flat"]:
            continue
        e = rows[inn]
        e["has_record"], e["n_freq_rows"] = True, e["n_freq_rows"] + 1
        if sl.liability_of(r["side_effect_name"]) and (inn, tuple(r.values())) not in seen:
            seen.add((inn, tuple(r.values())))  # LLT/PT duplicates of one row
            e["mapped_rows"].append(r)
    for r in se:
        inn = owner.get(r["stereo"])
        if inn not in rows or r["flat"] not in match[inn]["flat"]:
            continue
        rows[inn]["has_record"], rows[inn]["n_all_se_rows"] = True, rows[inn]["n_all_se_rows"] + 1
        if sl.liability_of(r["side_effect_name"]):
            rows[inn]["listed_pts"].add(r["side_effect_name"])
    for e in rows.values():
        e["listed_pts"] = sorted(e["listed_pts"])
    retrieved = json.loads(sl.CID_MAP.read_text())["provenance"]["sider"]
    out = {
        "description": "SIDER 4.1 meddra_freq rows with a mapped PT (LLT/PT duplicates collapsed), "
                       "record flags and the match_stereo result for every P35 held-out drug "
                       "(sha256(inn) % 10 in {0,1}). Built by scripts/p35_4_heldout_rows.py. No label "
                       "is stored: go_dump_p35 builds labels from these rows only under NC_HOST_EVAL.",
        "session_trees": "this file has no 'drugs' table, so pipeline/guards.hide_heldout deletes it "
                         "from every session tree; frozen on main by guards.frozen_paths",
        "sider": {**retrieved, "retrieved": time.strftime("%Y-%m-%d"), "n_freq_rows": len(freq), "n_all_se_rows": len(se),
                  "stereo_lookup": "PubChem PUG-REST title, parent/component CIDs and synonyms, "
                                   f"retrieved {time.strftime('%Y-%m-%d')}"},
        ROWS_KEY: dict(sorted(rows.items())),
    }
    HELDOUT_ROWS.write_text(json.dumps(out, indent=0, sort_keys=False, ensure_ascii=False) + "\n")
    print(f"wrote {HELDOUT_ROWS.relative_to(ROOT)}: {len(rows)} P35 held-out drugs (labels not computed)")


if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]](sys.argv[2:])
