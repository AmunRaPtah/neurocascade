#!/usr/bin/env python3
"""Build the v2 binding INPUT table for every drug in the v2 universe.

``python scripts/v2_binding.py [--cache DIR]``

Writes ``src/neurocascade/benchmark/v2/inputs/binding.json``: for every drug
in ``src/neurocascade/benchmark/v2/drugs.json`` (331 drugs), a human Ki panel
across drugs.json's 11 anchor receptors plus the other CNS targets v1's
``scripts/build_benchmark_binding.py`` panels used in at least 15 of their
drugs (the "core" monoamine / cholinergic / adrenergic / histaminergic /
transporter / hERG set; v1's low-n and multi-subunit-complex targets are
left out; this is a bulk per-drug-x-target input table, not a curated panel).

Why this is an INPUT, not an outcome
-------------------------------------
Binding (receptor occupancy) is a model input, not a human liability label.
``pipeline/guards.v2_heldout`` governs which drugs' *truth* (outcome) rows are
hidden from architect/reviewer sessions; held-out membership is deliberately
never written to any file (see v2_universe.py's docstring and guards.py). This
script writes none either: ``binding.json`` carries no "heldout"/"split"
field, and every drug, held-out or not, gets a full binding row.

Source and inclusion rules
---------------------------
ChEMBL REST (``https://www.ebi.ac.uk/chembl/api/data``), ``activity`` records
for each drug's ``parent_molecule_chembl_id`` (salts folded into the parent,
matching drugs.json and v1), restricted to:

* ``target_organism`` = ``Homo sapiens``, target itself a ``SINGLE PROTEIN``
  resolved to a ChEMBL target id from its UniProt accession via the API (see
  ``resolve_targets``); no target id is ever typed from memory;
* ``standard_type`` in (``Ki``, ``Kd``), ``pchembl_value`` present;
* no ``data_validity_comment``, not ``potential_duplicate``;
* ``standard_relation`` == ``"="`` (censored ``>``/``>=`` rows are kept out of
  the measured summary and recorded as lower bounds only; see (a) below);
* ``standard_units`` == ``"nM"``.

**Ki and Kd are kept separate.** Only Ki rows feed the per-target summary
(``pki_median``, ``ki_nM``, min/max, document list); this mirrors v1's
Ki-only panels and keeps Kd (a different assay family, mostly radioligand
displacement at low-throughput targets) from silently pulling the Ki estimate
toward a different measurement type. A target with Kd rows but no Ki rows
still lands in ``searched_and_empty`` for Ki purposes; its Kd rows are listed,
unaggregated, under ``kd_only``.

**Aggregation**: pChEMBL values are grouped by ``document_chembl_id``, each
document's values averaged, then the per-target ``pki_median`` is the median
of those per-document means (same two-step summary as
``build_benchmark_binding.py._summarise``, which avoids one prolific document
dominating the estimate). ``n_documents``/``n_measurements`` count distinct
documents and raw rows; ``ki_min_nM``/``ki_max_nM`` are the extremes of the
individual (not per-document-averaged) Ki values. Up to 10 documents are
listed per target: ``document_chembl_id``, ``pubmed_id`` (fetched from the
ChEMBL ``/document`` endpoint, never typed), and publication year.

Binding coverage extension (2026-10-28): censored rows and GtoPdb
-------------------------------------------------------------------
The rules below were written down before the extended build was run or any
coverage number was looked at.

**(a) Censored ChEMBL rows -> lower bounds, never values.** A second ChEMBL
query per drug fetches human ``standard_type=Ki`` rows with
``standard_relation`` in (``>``, ``>=``) (pChEMBL is null for these, so the
main query never saw them), same validity / duplicate / ``nM`` / panel-target
filters. They are summarised per gene under the per-drug key
``ki_lower_bounds`` (NOT inside ``targets``): ``ki_lower_bound_nM`` is the
SMALLEST censored threshold (the bound every censored record supports),
``ki_lower_bound_max_nM`` the largest, plus ``n_measurements``,
``n_documents`` and up to 10 ChEMBL document records. A gene can carry both
a measured ``targets`` entry and a ``ki_lower_bounds`` entry; the two are
never combined. ``searched_and_empty`` keeps its meaning ("no measured Ki
from any source"), so a bound-only gene is in ``searched_and_empty`` AND in
``ki_lower_bounds``. ``<``/``<=``/``~`` rows are not used.

**(b) Second measured source: IUPHAR/BPS Guide to Pharmacology (GtoPdb).**
The bulk download ``https://www.guidetopharmacology.org/DATA/interactions.csv``
and ``.../DATA/ligands.csv`` are fetched once into ``--cache``/gtopdb (with
the retrieval date and sha256 recorded in provenance; the GtoPdb release is
read from the file's own header line). Rows used: ``Target Species`` =
``Human``; ``Target UniProt ID`` equal to the panel accession (the same
UniProt mapping the ChEMBL side uses); ``Affinity Units`` = ``pKi``;
``Original Affinity Relation`` either ``=`` or blank (GtoPdb's "no
qualifier"; ``>``/``<``/``~`` rows are excluded, and GtoPdb censored rows are
NOT used as lower bounds because the file does not state unambiguously
whether the qualifier applies to the pKi or the nM value); ``Affinity
Median`` numeric. Ligand -> v2 drug mapping is structural/identifier only: a
GtoPdb ligand maps to a drug when its ``ChEMBL ID`` equals the drug's ChEMBL
parent id, or its ``InChIKey`` equals the full standard InChIKey of that
ChEMBL parent molecule (fetched from the ChEMBL ``/molecule`` endpoint). No
name matching. A ligand matching two different drugs is dropped.

The PDSP Ki database was checked and not used: its bulk download
(``https://pdsp.unc.edu/databases/kiDownload/``) returned HTTP 403 to
programmatic access on 2026-10-28, and its records carry no InChIKey or
ChEMBL id, so it could not be mapped without name matching.

**(c) Merge rule (all ``=`` measurements, all sources, no source
preference).** Per drug x gene, every qualifying measurement is a pKi
(ChEMBL ``pchembl_value``; GtoPdb ``Affinity Median``). Measurements are
grouped by publication: ChEMBL rows by ``document_chembl_id`` (unchanged),
GtoPdb rows by their PubMed id (first listed), or by a per-row GtoPdb key
when no PMID is given. A GtoPdb row whose PMID equals the PMID of a ChEMBL
document already present for the same drug x gene is dropped as the same
published measurement already counted (de-duplication, not preference).
``pki_median`` = median over publication groups of the within-group mean
pKi; ``ki_nM`` = 10**(9 - pki_median). ``ki_min_nM``/``ki_max_nM`` are the
extremes over individual measurements (GtoPdb pKi converted to nM).
``n_documents`` counts publication groups from both sources,
``n_measurements`` rows from both. ``documents`` keeps the ChEMBL document
list as before; GtoPdb publications are listed under
``gtopdb_references`` (``gtopdb_ligand_id``, ``pubmed_id`` as given in the
GtoPdb row, ``pki``), and ``sources`` gives the per-source group counts.
For a drug x gene with only ChEMBL data, every field is identical to the
ChEMBL-only build.

Caching: every raw ChEMBL page is cached by URL hash under
``--cache`` (default ``/root/nc-v2-cache/binding``), so a rerun with an
intact cache makes zero network calls. Target-activity fetches run with
<=4 concurrent workers; each request retries with exponential backoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "src" / "neurocascade" / "benchmark"
V2 = BENCH / "v2"
DRUGS_FILE = V2 / "drugs.json"
OUT = V2 / "inputs" / "binding.json"
DEFAULT_CACHE = Path(os.environ.get("NC_V2_BINDING_CACHE", "/root/nc-v2-cache/binding"))
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

# Non-anchor CNS targets added to drugs.json's 11 anchors: gene symbol ->
# UniProt accession. Chosen as the genes v1's build_benchmark_binding.py
# panels (src/neurocascade/benchmark/binding/*.json) found an exact Ki for in
# >=15 of its ~80 drugs, restricted to single UniProt accessions (v1's
# multi-subunit GABA-A/nicotinic complexes are left out; this script
# queries one ChEMBL SINGLE PROTEIN target per gene, not a receptor complex).
# Accessions are taken from v1's already-published per-drug files (so the
# same protein is meant), but every ChEMBL *target id* below is re-resolved
# from the accession via the API at run time (resolve_targets), never copied
# from v1's json.
EXTRA_UNIPROT = {
    "HTR1A": "P08908", "SLC6A3": "Q01959", "HTR2A": "P28223", "SLC6A4": "P31645",
    "DRD3": "P35462", "DRD4": "P21917", "SLC6A2": "P23975", "HTR6": "P50406",
    "HTR7": "P34969", "DRD1": "P21728", "ADRA2A": "P08913", "HRH2": "P25021",
    "ADRA2C": "P18825", "HTR1D": "P28221", "HTR3A": "P46098", "ADRA2B": "P18089",
    "HTR1B": "P28222", "HTR2B": "P41595", "HRH4": "Q9H3N8", "HTR5A": "P47898",
    "HRH3": "Q9Y5N1", "DRD5": "P21918", "HTR1E": "P28566", "ADRB1": "P08588",
    "ADRB2": "P07550", "KCNH2": "Q12809",
}
MAX_DOCS = 10


# ---------------------------------------------------------------- transport

def _cache_file(cache: Path, url: str) -> Path:
    return cache / "raw" / (hashlib.sha1(url.encode()).hexdigest() + ".json")


def _get_json(cache: Path, url: str, timeout: int = 180):
    """GET url as JSON, cached by url hash. 404 -> None. Retries w/ backoff."""
    fn = _cache_file(cache, url)
    if fn.exists():
        return json.loads(fn.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-v2-binding"})
    err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            fn.parent.mkdir(parents=True, exist_ok=True)
            fn.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # noqa: BLE001 - retry any transport error
            err = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {err!r}")


def _pages(cache: Path, path_and_query: str) -> list[dict]:
    out: list[dict] = []
    url = f"{CHEMBL}/{path_and_query}"
    while url:
        page = _get_json(cache, url)
        if page is None:
            break
        key = next(k for k in page if k != "page_meta")
        out.extend(page[key])
        nxt = page["page_meta"].get("next")
        url = f"https://www.ebi.ac.uk{nxt}" if nxt else None
    return out


# ---------------------------------------------------------------- targets

def resolve_targets(cache: Path, panel: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """gene -> chembl target id, resolved from its UniProt accession; and
    gene -> failure reason for any that did not resolve to exactly one
    human SINGLE PROTEIN target."""
    ids, failures = {}, {}
    for gene, acc in panel.items():
        q = (f"target.json?target_components__accession={acc}"
             f"&target_type=SINGLE%20PROTEIN&organism=Homo%20sapiens")
        ts = (_get_json(cache, f"{CHEMBL}/{q}") or {}).get("targets", [])
        if len(ts) != 1:
            failures[gene] = f"{acc}: expected exactly one target, got {[t['target_chembl_id'] for t in ts]}"
            continue
        ids[gene] = ts[0]["target_chembl_id"]
    return ids, failures


# ---------------------------------------------------------------- activities

ACT_ONLY = (
    "activity_id,parent_molecule_chembl_id,target_chembl_id,target_organism,"
    "standard_type,standard_relation,standard_value,standard_units,pchembl_value,"
    "data_validity_comment,potential_duplicate,document_chembl_id"
)


def fetch_drug_activities(cache: Path, chembl_id: str) -> list[dict]:
    q = (f"activity.json?parent_molecule_chembl_id={chembl_id}"
         f"&target_organism=Homo%20sapiens&standard_type__in=Ki,Kd"
         f"&pchembl_value__isnull=false&only={ACT_ONLY}&limit=1000")
    return _pages(cache, q)


def fetch_documents(cache: Path, doc_ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ids = sorted(doc_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        q = (f"document.json?document_chembl_id__in={','.join(chunk)}"
             f"&only=document_chembl_id,pubmed_id,year&limit=1000")
        for d in _pages(cache, q):
            out[d["document_chembl_id"]] = d
    return out


# ---------------------------------------------------------------- aggregation

def _usable(a: dict) -> bool:
    return (not a.get("data_validity_comment")
            and not a.get("potential_duplicate")
            and a.get("standard_relation") == "="
            and a.get("standard_units") == "nM"
            and a.get("standard_value") not in (None, "")
            and a.get("pchembl_value") not in (None, ""))


def _doc_rows(doc_ids: list[str], docs: dict[str, dict]) -> list[dict]:
    rows = []
    for did in doc_ids[:MAX_DOCS]:
        d = docs.get(did) or {}
        rows.append({"document_chembl_id": did, "pubmed_id": d.get("pubmed_id"),
                     "year": d.get("year")})
    return rows


def summarise_target(acts: list[dict], docs: dict[str, dict]) -> dict:
    by_doc: dict[str, list[float]] = defaultdict(list)
    vals_nM: list[float] = []
    for a in acts:
        by_doc[a["document_chembl_id"]].append(float(a["pchembl_value"]))
        vals_nM.append(float(a["standard_value"]))
    pki_median = statistics.median(statistics.fmean(v) for v in by_doc.values())
    doc_ids = sorted(by_doc, key=lambda d: -((docs.get(d) or {}).get("year") or 0))
    return {
        "status": "data",
        "pki_median": round(pki_median, 4),
        "ki_nM": round(10 ** (9 - pki_median), 6),
        "ki_min_nM": min(vals_nM),
        "ki_max_nM": max(vals_nM),
        "n_documents": len(by_doc),
        "n_measurements": len(acts),
        "documents": _doc_rows(doc_ids, docs),
    }


def build_drug(drug: dict, target_ids: dict[str, str], acts: list[dict],
                docs: dict[str, dict]) -> dict:
    by_target_id = defaultdict(list)   # Ki only
    kd_by_target_id = defaultdict(list)  # Kd only
    for a in acts:
        if not _usable(a):
            continue
        tid = a["target_chembl_id"]
        if tid not in target_ids.values():
            continue
        (by_target_id if a["standard_type"] == "Ki" else kd_by_target_id)[tid].append(a)

    gene_of = {tid: g for g, tid in target_ids.items()}
    targets: dict[str, dict] = {}
    searched_and_empty: list[str] = []
    kd_only: dict[str, dict] = {}
    for gene, tid in sorted(target_ids.items()):
        acts_t = by_target_id.get(tid, [])
        if acts_t:
            targets[gene] = summarise_target(acts_t, docs)
        else:
            searched_and_empty.append(gene)
            kd_acts = kd_by_target_id.get(tid, [])
            if kd_acts:
                doc_ids = sorted({a["document_chembl_id"] for a in kd_acts})
                kd_only[gene] = {
                    "n_measurements": len(kd_acts),
                    "documents": _doc_rows(doc_ids, docs),
                }
    return {
        "inn": drug["inn"],
        "chembl_id": drug["chembl_id"],
        "targets": targets,
        "searched_and_empty": sorted(searched_and_empty),
        "kd_only": kd_only,
    }


# ---------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args(argv)
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)

    universe = json.loads(DRUGS_FILE.read_text())
    drugs = universe["drugs"]
    anchor_panel = {g: v["uniprot"] for g, v in universe["anchor_targets"].items()}
    panel_uniprot = {**anchor_panel, **EXTRA_UNIPROT}

    status = _get_json(cache, f"{CHEMBL}/status.json")
    release = f"{status['chembl_db_version']} ({status['chembl_release_date']})"
    print("ChEMBL", release, file=sys.stderr)

    target_ids, failures = resolve_targets(cache, panel_uniprot)
    for g, why in sorted(failures.items()):
        print("TARGET RESOLUTION FAILED:", g, why, file=sys.stderr)
    print(len(target_ids), "of", len(panel_uniprot), "panel targets resolved", file=sys.stderr)

    acts_by_drug: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(fetch_drug_activities, cache, d["chembl_id"]): d for d in drugs}
        done = 0
        for fut in as_completed(futs):
            d = futs[fut]
            acts_by_drug[d["inn"]] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  fetched activities for {done}/{len(drugs)} drugs", file=sys.stderr)

    all_doc_ids = {a["document_chembl_id"] for acts in acts_by_drug.values() for a in acts
                   if _usable(a) and a["target_chembl_id"] in target_ids.values()}
    docs = fetch_documents(cache, all_doc_ids)
    print(len(docs), "documents fetched", file=sys.stderr)

    out_drugs = {}
    for d in drugs:
        out_drugs[d["inn"]] = build_drug(d, target_ids, acts_by_drug[d["inn"]], docs)

    provenance = {
        "retrieved": time.strftime("%Y-%m-%d"),
        "chembl": {"api": CHEMBL, "release": release},
        "panel": {g: {"uniprot": panel_uniprot[g], "chembl_target_id": target_ids.get(g)}
                  for g in sorted(panel_uniprot)},
        "panel_resolution_failures": failures,
        "aggregation_rule": (
            "Per drug x target: ChEMBL activities with standard_type=Ki, "
            "target_organism=Homo sapiens, target a ChEMBL SINGLE PROTEIN "
            "resolved from the panel's UniProt accession, standard_relation='=', "
            "standard_units='nM', pchembl_value present, no data_validity_comment, "
            "not potential_duplicate. pChEMBL values are grouped by "
            "document_chembl_id, averaged within each document, and pki_median "
            "is the median of those per-document means. ki_nM = 10**(9-pki_median). "
            "ki_min_nM/ki_max_nM are the extremes of the individual measurements."),
        "kd_handling": (
            "Kd rows (same filters, standard_type=Kd) are kept separate under "
            "kd_only for drug x target pairs with no qualifying Ki, and are never "
            "merged into the Ki aggregate."),
        "no_data_handling": (
            "A drug x target pair with zero qualifying Ki rows is listed by gene "
            "in that drug's searched_and_empty array; it is never imputed or "
            "omitted silently."),
        "held_out_note": (
            "Binding is a model input, not a held-out outcome (pipeline/guards.py "
            "V2_TRUTH / v2_heldout); no held-out membership is recorded in this file."),
        "gtopdb": "not queried in this build; ChEMBL only (see module docstring).",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"provenance": provenance, "drugs": out_drugs},
                              indent=1, sort_keys=True) + "\n")
    print("wrote", OUT, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
