#!/usr/bin/env python3
"""P35.2: PubChem CID map, SIDER train labels and the train universe.

``python scripts/p35_2_build.py fetch [--cache DIR]``
    Network. Downloads the two SIDER 4.1 files (sha256-checked against
    ``sider_labels.SIDER_FILES``), and for every v2 drug resolves its ChEMBL
    parent to PubChem: (1) ChEMBL ``standard_inchi_key`` -> PubChem CIDs; (2)
    when PubChem has no record under that key, an exact-structure search
    (``fastidentity`` ``same_stereo_isotope``) on the ChEMBL canonical SMILES;
    (3) the PubChem same-connectivity CIDs of the result. Then title,
    parent (or, if none, component) CIDs and synonyms of every SIDER stereo
    id under a candidate flat id. Everything is cached as fetched under DIR
    (default ``$NC_V2_CACHE/sider``).

``python scripts/p35_2_build.py build [--cache DIR]``
    Offline. Writes, under ``src/neurocascade/benchmark/v2/sider/``:
    ``cid_map.json`` (all v2 drugs; SIDER match only for P35-train drugs),
    ``sider_train_rows.json.gz`` (frozen extract: every mapped-PT frequency
    row and record flag for train drugs, enough to rebuild the labels without
    SIDER), ``labels_train.json``, ``pt_listing.json`` and
    ``universe_train.json``. P35 held-out drugs never get a SIDER match,
    row, label or universe row here; P35.4 builds those host-side with the
    same functions.
"""
from __future__ import annotations

import collections
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurocascade.benchmark.v2 import (  # noqa: E402
    load_v2_binding, load_v2_drugs, load_v2_human_cmax, v2_heldout)
from neurocascade.benchmark.v2 import sider_labels as sl  # noqa: E402

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"


def _get(url: str, data: bytes | None = None) -> bytes | None:
    for i in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            time.sleep(2 * (i + 1))
        except OSError:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"giving up on {url}")


def _cids(b: bytes | None) -> list[int]:
    return json.loads(b)["IdentifierList"]["CID"] if b else []


def _cache(argv: list[str]) -> Path:
    if "--cache" in argv:
        return Path(argv[argv.index("--cache") + 1])
    return Path(os.environ.get("NC_V2_CACHE", ROOT / ".cache" / "v2")) / "sider"


def _sider(cache: Path) -> tuple[list[dict], list[dict]]:
    return (sl.read_freq(sl.read_gz_lines(cache / "meddra_freq.tsv.gz")),
            sl.read_all_se(sl.read_gz_lines(cache / "meddra_all_se.tsv.gz")))


def _flat_to_stereo(freq: list[dict], se: list[dict]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = collections.defaultdict(set)
    for r in freq + se:
        out[r["flat"]].add(r["stereo"])
    return out


def fetch(argv: list[str]) -> None:
    cache = _cache(argv)
    (cache / "pubchem").mkdir(parents=True, exist_ok=True)
    for name in sl.SIDER_FILES:
        if not (cache / name).exists():
            (cache / name).write_bytes(_get(sl.SIDER_MIRROR + name))
    (cache / "retrieved.json").exists() or (cache / "retrieved.json").write_text(
        json.dumps({"date": dt.date.today().isoformat()}))
    for d in load_v2_drugs():
        f = cache / "pubchem" / f"{d['inn']}.json"
        if f.exists():
            continue
        m = json.loads(_get(f"{CHEMBL}molecule/{d['chembl_id']}.json"))
        s = m.get("molecule_structures") or {}
        r = {"chembl_id": d["chembl_id"], "inchikey": s.get("standard_inchi_key"),
             "smiles": s.get("canonical_smiles")}
        r["cids"], r["resolution"] = _cids(_get(f"{PUG}inchikey/{r['inchikey']}/cids/JSON")), "inchikey"
        if not r["cids"]:
            body = urllib.parse.urlencode({"smiles": r["smiles"]}).encode()
            r["cids"] = _cids(_get(f"{PUG}fastidentity/smiles/cids/JSON?identity_type=same_stereo_isotope", body))
            r["resolution"] = "smiles_same_stereo_isotope"
        r["same_connectivity"] = _cids(_get(
            f"{PUG}fastidentity/cid/{r['cids'][0]}/cids/JSON?identity_type=same_connectivity")) if r["cids"] else []
        f.write_text(json.dumps(r))
        time.sleep(0.25)
    flat = _flat_to_stereo(*_sider(cache))
    pf = cache / "stereo.json"
    st = json.loads(pf.read_text()) if pf.exists() else {}
    for d in load_v2_drugs():
        r = json.loads((cache / "pubchem" / f"{d['inn']}.json").read_text())
        for fl in set(r["cids"]) | set(r["same_connectivity"]):
            for s in flat.get(fl, ()):
                if str(s) in st:
                    continue
                t = json.loads(_get(f"{PUG}cid/{s}/property/Title/JSON"))["PropertyTable"]["Properties"][0]
                par = _cids(_get(f"{PUG}cid/{s}/cids/JSON?cids_type=parent"))
                kind = "parent"
                if not par:
                    par, kind = _cids(_get(f"{PUG}cid/{s}/cids/JSON?cids_type=component")), "component"
                st[str(s)] = {"title": t.get("Title"), "parents": par, "parent_kind": kind}
                time.sleep(0.25)
            for s in flat.get(fl, ()):
                if "synonyms" not in st[str(s)]:
                    b = _get(f"{PUG}cid/{s}/synonyms/JSON")
                    st[str(s)]["synonyms"] = (json.loads(b)["InformationList"]["Information"][0]
                                              .get("Synonym", []) if b else [])
                    time.sleep(0.25)
        pf.write_text(json.dumps(st, indent=0, sort_keys=True))


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n")


def build(argv: list[str]) -> None:
    cache = _cache(argv)
    freq, se = _sider(cache)
    flat = _flat_to_stereo(freq, se)
    stereo = json.loads((cache / "stereo.json").read_text())
    parents = {int(k): v["parents"] for k, v in stereo.items()}
    synonyms = {int(k): v["synonyms"] for k, v in stereo.items()}
    all_inns = [d["inn"] for d in load_v2_drugs()]
    retrieved = json.loads((cache / "retrieved.json").read_text())["date"]
    drugs = load_v2_drugs()
    train = [d for d in drugs if not sl.p35_heldout(d["inn"])]
    sl.SIDER_DIR.mkdir(exist_ok=True)

    # (1) CID map: PubChem resolution for every drug, SIDER match for train only.
    cid_map, match = {}, {}
    for d in drugs:
        r = json.loads((cache / "pubchem" / f"{d['inn']}.json").read_text())
        row = {"chembl_id": d["chembl_id"], "chembl_standard_inchikey": r["inchikey"],
               "pubchem_cids": r["cids"], "resolution": r["resolution"],
               "pubchem_same_connectivity_cids": r["same_connectivity"]}
        if not sl.p35_heldout(d["inn"]):
            m = sl.match_stereo(d["inn"], r["cids"], r["same_connectivity"], flat, parents,
                                synonyms, all_inns)
            m["stereo_rejected"] = [{"cid": s, "title": stereo[str(s)]["title"],
                                     "parents": stereo[str(s)]["parents"]} for s in m["stereo_rejected"]]
            m["stereo_titles"] = {str(s): stereo[str(s)]["title"] for s in m["stereo"]}
            row["sider"] = match[d["inn"]] = m
        cid_map[d["inn"]] = row
    sider_prov = {"release": "SIDER 4.1 (Kuhn et al. 2016, Nucleic Acids Res 44:D1075, PMID 26481350)",
                  "upstream": sl.SIDER_UPSTREAM, "fetched_from": sl.SIDER_MIRROR, "retrieved": retrieved,
                  "sha256": sl.SIDER_FILES, "n_freq_rows": len(freq), "n_all_se_rows": len(se),
                  "n_flat_ids": len(flat)}
    _dump(sl.CID_MAP, {
        "description": "v2 drug -> PubChem CID (from the ChEMBL parent), and for P35-train drugs the "
                       "SIDER flat/stereo ids matched by sider_labels.match_stereo. P35 held-out drugs "
                       "carry no SIDER field; P35.4 recomputes it host-side.",
        "provenance": {"chembl_release": json.loads((sl.SIDER_DIR.parent / "drugs.json").read_text())["chembl_release"],
                       "pubchem_pug_rest_retrieved": retrieved, "sider": sider_prov,
                       "resolution": "ChEMBL standard_inchi_key -> PubChem CID; if none, PubChem "
                                     "fastidentity same_stereo_isotope on ChEMBL canonical_smiles; "
                                     "connectivity CIDs by fastidentity same_connectivity"},
        "drugs": cid_map})

    # (2) Frozen extract: mapped-PT rows and record flags, train drugs only.
    stereo_owner: dict[int, str] = {}
    for inn, m in match.items():
        for s in m["stereo"]:
            if stereo_owner.setdefault(s, inn) != inn:
                raise ValueError(f"SIDER stereo id {s} matches both {stereo_owner[s]} and {inn}")
    rows = {inn: {"has_record": False, "n_freq_rows": 0, "n_all_se_rows": 0, "mapped_rows": [],
                  "listed_pts": set()} for inn in match}
    unmatched: dict[str, set[str]] = collections.defaultdict(set)
    map_hits: dict[str, set[str]] = collections.defaultdict(set)
    seen_mapped = set()
    for r in freq:
        inn = stereo_owner.get(r["stereo"])
        if inn is None or r["flat"] not in match[inn]["flat"]:
            continue
        e = rows[inn]
        e["has_record"], e["n_freq_rows"] = True, e["n_freq_rows"] + 1
        if sl.liability_of(r["side_effect_name"]):
            key = tuple(r.values())
            if (inn, key) not in seen_mapped:  # LLT/PT duplicates of one row
                seen_mapped.add((inn, key))
                e["mapped_rows"].append(r)
            map_hits[r["side_effect_name"].casefold()].add(inn)
        else:
            unmatched[r["side_effect_name"]].add(inn)
    for r in se:
        inn = stereo_owner.get(r["stereo"])
        if inn is None or r["flat"] not in match[inn]["flat"]:
            continue
        rows[inn]["has_record"], rows[inn]["n_all_se_rows"] = True, rows[inn]["n_all_se_rows"] + 1
        if sl.liability_of(r["side_effect_name"]):
            rows[inn]["listed_pts"].add(r["side_effect_name"])
    for e in rows.values():
        e["listed_pts"] = sorted(e["listed_pts"])
    extract = {"description": "Every SIDER 4.1 meddra_freq row with a mapped PT for each P35-train "
                              "drug (LLT/PT duplicate rows collapsed), plus record flags and row counts. "
                              "labels_train.json is rebuilt from this file alone.",
               "sider": sider_prov, "drugs": dict(sorted(rows.items()))}
    sl.TRAIN_ROWS.write_bytes(gzip.compress(json.dumps(extract, indent=0, sort_keys=True).encode(), mtime=0))

    # (3) Labels.
    labels = sl.build_train_labels(rows)

    # (4) PT listing.
    vocab = {r["side_effect_name"].casefold() for r in freq} | {r["side_effect_name"].casefold() for r in se}
    _dump(sl.PT_LISTING, {
        "description": "Matched and unmatched SIDER side_effect_name strings over P35-train drugs' "
                       "frequency rows. Map PTs absent from the SIDER vocabulary match nothing; they "
                       "are not replaced (the map is sealed).",
        "map": {liab: [{"pt": pt, "in_sider_vocabulary": pt.casefold() in vocab,
                        "n_train_drugs_with_frequency_row": len(map_hits.get(pt.casefold(), ()))}
                       for pt in pts] for liab, pts in sl.LIABILITY_PTS.items()},
        "unmatched": {n: len(v) for n, v in sorted(unmatched.items())}})

    # (5) Universe.
    anchors = list(json.loads((sl.SIDER_DIR.parent / "drugs.json").read_text())["anchor_targets"])
    binding, cmax = load_v2_binding(), load_v2_human_cmax()
    uni = [sl.universe_row(d["inn"], anchors, binding.get(d["inn"]), rows[d["inn"]]["has_record"],
                           cmax.get(d["inn"]), v2_heldout(d["inn"])) for d in train]
    members = [u["inn"] for u in uni if u["in_universe"] is True]
    counts = {"universe": sl.label_counts(labels, members),
              "all_train_with_sider_record": sl.label_counts(labels, [i for i in rows if rows[i]["has_record"]])}
    _dump(sl.TRAIN_LABELS, {
        "description": "P35-train SIDER liability labels per docs/prereg-p35-go.md, built by "
                       "scripts/p35_2_build.py with sider_labels.build_train_labels.",
        "rule": {"positive": "a mapped PT with placebo-adjusted frequency_upper >= 0.01",
                 "negative": "SIDER record, no mapped PT >= 0.01", "excluded": "no SIDER record",
                 "aggregation": "per drug x PT: max non-placebo upper minus max placebo upper"},
        "split_rule": "held-out iff int(sha256(inn.strip().lower()), 16) % 10 in {0, 1}; not stored",
        "n_train_drugs": len(train), "counts": counts, "drugs": labels})
    _dump(sl.UNIVERSE, {
        "description": "P35-train universe: measured Ki at >= 1 anchor target (inputs/binding.json, "
                       "scripts/v2_binding.py unchanged), a SIDER record, and a Cmax value "
                       "(truth/human_cmax.json, scripts/v2_cmax.py unchanged). pending_host: the Cmax "
                       "row is a v2 held-out row stripped from session trees.",
        "frozen_inputs": {p: hashlib.sha256((sl.SIDER_DIR.parent / p).read_bytes()).hexdigest()
                          for p in ("inputs/binding.json", "truth/human_cmax.json", "drugs.json")},
        "frozen_inputs_note": "sha256 of the files as seen in the session tree (human_cmax.json "
                              "with v2 held-out rows stripped)",
        "n_train_drugs": len(train),
        "n_in_universe": len(members),
        "n_pending_host": sum(u["in_universe"] == "pending_host" for u in uni),
        "drugs": uni})
    # (6) Concordance with the v2 FDA placebo-table labels (train drugs only),
    # and the results file the host reads.
    ref = {r["inn"]: r for r in json.loads((sl.SIDER_DIR.parent / "truth" / "liabilities.json")
                                         .read_text())["drugs"]
           if not sl.p35_heldout(r["inn"]) and not v2_heldout(r["inn"])}
    shared = [x for x in sl.LIABILITY_PTS if x != "seizure"]
    conc = sl.concordance(labels, ref, shared)
    conc_any = sl.concordance(labels, ref, shared, ("placebo_subtracted", "absolute_fallback"))
    n_scored = len({inn for inn, r in ref.items() if inn in labels and labels[inn]["sedation"]["label"] != "excluded"
                    and any(isinstance(r.get(x), dict) and r[x].get("basis") == "placebo_subtracted"
                            and r[x].get("label") in (0, 1) for x in shared)})
    _dump(ROOT / "results" / "P35.2.json", {
        "task": "P35.2", "prereg": "docs/prereg-p35-go.md",
        "n_scored": n_scored,
        "n_scored_means": "P35-train drugs with a SIDER 0/1 label and a v2 FDA placebo-subtracted label "
                          "for at least one shared liability (concordance of two external label sources); "
                          "v2 held-out drugs are left out so the number is the same in session and host trees",
        "split": "P35 held-out (sha256(inn) % 10 in {0,1}) labels NOT built; P35.4 builds them host-side",
        "n_train": len(train), "n_train_with_sider_record": sum(r["has_record"] for r in rows.values()),
        "n_universe": len(members), "n_universe_pending_host": sum(u["in_universe"] == "pending_host" for u in uni),
        "train_counts": counts,
        "concordance_vs_v2_fda_placebo_subtracted": conc,
        "concordance_vs_v2_fda_any_basis": conc_any,
        "sider_sha256": sl.SIDER_FILES})
    print(json.dumps(counts, indent=1))
    print(json.dumps(conc, indent=1), n_scored)


if __name__ == "__main__":
    {"fetch": fetch, "build": build}[sys.argv[1]](sys.argv[2:])
