#!/usr/bin/env python3
"""Build a local index of every FDA prescription label from the openFDA bulk export.

Why bulk and not the API
------------------------
The v2 benchmark (P33) needs label tables for several hundred drugs. openFDA
without a key allows 1,000 requests a day; the bulk export is the same
Structured Product Labels in 14 zip files (~1.8 GB). Each part is streamed,
filtered to HUMAN PRESCRIPTION DRUG labels, reduced to the fields the v2
stages read, and deleted, so peak disk is one part.

Output: one gzipped JSON-lines file per part in ``$NC_V2_CACHE/label-parts/``
(default /root/nc-v2-cache; read them with ``iter_labels()``), plus ``labels.meta.json`` recording the export
date and part checksums, so every value derived from a label can cite the
label's set_id, version and effective_time.

    scripts/v2_label_index.py            # build (resumes per part)
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

CACHE = Path(os.environ.get("NC_V2_CACHE", "/root/nc-v2-cache"))
MANIFEST = "https://api.fda.gov/download.json"
KEEP = ("adverse_reactions", "adverse_reactions_table", "clinical_studies",
        "clinical_pharmacology", "pharmacokinetics", "dosage_and_administration",
        "warnings_and_cautions", "warnings", "spl_product_data_elements",
        "description")
OPENFDA_KEEP = ("generic_name", "brand_name", "substance_name", "application_number",
                "manufacturer_name", "route", "rxcui", "unii")


def _slim(rec: dict) -> dict | None:
    of = rec.get("openfda") or {}
    # About a third of labels carry no openfda block at all (Spravato, Ubrelvy),
    # so a label is also kept when it is unharmonised but has an ADVERSE
    # REACTIONS section, which OTC Drug Facts labels do not have.
    rx = "HUMAN PRESCRIPTION DRUG" in (of.get("product_type") or [])
    if not rx and (of or "adverse_reactions" not in rec):
        return None
    out = {"set_id": rec.get("set_id"), "id": rec.get("id"), "version": rec.get("version"),
           "effective_time": rec.get("effective_time"),
           "openfda": {k: of[k] for k in OPENFDA_KEEP if k in of}}
    for k in KEEP:
        if k in rec:
            out[k] = rec[k]
    return out


def iter_labels():
    """Every indexed label, part by part."""
    for p in sorted((CACHE / "label-parts").glob("*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                yield json.loads(line)


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MANIFEST, timeout=120) as r:
        lab = json.load(r)["results"]["drug"]["label"]
    parts_dir = CACHE / "label-parts"
    parts_dir.mkdir(exist_ok=True)
    meta = {"source": "openFDA drug/label bulk export", "export_date": lab["export_date"],
            "total_records": lab["total_records"], "parts": []}
    for p in lab["partitions"]:
        name = p["file"].rsplit("/", 1)[1]
        done = parts_dir / (name + ".jsonl.gz")
        if done.exists():
            meta["parts"].append(json.loads((parts_dir / (name + ".meta")).read_text()))
            continue
        print("fetching", name, p["size_mb"], "MB", flush=True)
        tmp = parts_dir / name
        urllib.request.urlretrieve(p["file"], tmp)
        sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        n_in = n_out = 0
        with zipfile.ZipFile(tmp) as z, gzip.open(str(done) + ".tmp", "wt") as out:
            for member in z.namelist():
                with z.open(member) as f:
                    results = json.load(io.TextIOWrapper(f, encoding="utf-8"))["results"]
                for rec in results:
                    n_in += 1
                    s = _slim(rec)
                    if s:
                        n_out += 1
                        out.write(json.dumps(s) + "\n")
                del results
        os.replace(str(done) + ".tmp", done)
        tmp.unlink()
        pm = {"file": p["file"], "sha256": sha, "records": n_in, "rx_kept": n_out}
        (parts_dir / (name + ".meta")).write_text(json.dumps(pm))
        meta["parts"].append(pm)
        print(f"  {n_in} labels, {n_out} prescription kept", flush=True)
    (CACHE / "labels.meta.json").write_text(json.dumps(meta, indent=1))
    print("done", sum(x["rx_kept"] for x in meta["parts"]), "prescription labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
