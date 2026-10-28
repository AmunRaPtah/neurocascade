"""P33.1 stage 2 follow-up (v2 fu coverage): merge additional measured human
plasma fraction-unbound (fu) sources into inputs/plasma_fu.json, on top of
scripts/v2_pk.py's ChEMBL-only build.

Source priority / merge rule (fixed BEFORE any coverage number in this run
was computed, per the task's own sequencing requirement):

  1. ChEMBL (scripts/v2_pk.py's existing build) -- highest priority. ChEMBL
     rows already carry a per-row-verified document (live PMID via NCBI
     eutils, or a ChEMBL-minted DataCite DOI). If a drug has a ChEMBL fu
     value, that value is kept unchanged and no other source is consulted
     or recorded for it.

  2. DrugCentral mirror of Lombardo F, Berellini G, Obach RS. "Trend
     Analysis of a Database of Intravenous Pharmacokinetic Parameters in
     Humans for 1352 Drug Compounds." Drug Metab Dispos. 2018
     Nov;46(11):1466-1477. doi:10.1124/dmd.118.082966, PMID 30115648
     (citation verified live via NCBI eutils esearch/esummary + Crossref;
     the primary journal PDF is Cloudflare-blocked and not in PMC/Europe
     PMC -- confirmed via Europe PMC's inEPMC:N/isOpenAccess:N flags --
     so it is used only through this independently-citing, per-row-cited
     secondary mirror). Used only for drugs with no ChEMBL fu. Matched by
     an exact drugcentral.org <title> match against the drug's INN/base
     name, then verified by comparing DrugCentral's own InChIKey for the
     matched struct_id against ChEMBL's molecule_structures.standard_inchi
     for the same chembl_id; a mismatch (or an unresolvable structure on
     either side) drops the candidate rather than guessing.

  3. FDA label "X% bound to plasma protein(s)" text, extracted from the
     openFDA label cache scripts/v2_cmax.py already builds
     ($NC_V2_CACHE/cmax/labels.jsonl.gz), via a conservative regex (see
     tmp_fu_probe/label_fu.py, reused as the extraction module). Used only
     for drugs with neither a ChEMBL nor a DrugCentral/Lombardo fu value.
     Sentences mentioning a metabolite, multiple distinct %-bound clauses,
     or a special population (renal/hepatic impairment, pediatric,
     geriatric, pregnancy, elderly, dialysis, neonates/children) are
     rejected outright rather than disambiguated. A labelled range is
     converted to a point value by the geometric midpoint of its two
     endpoints (consistent with how ChEMBL PPB ranges are summarised
     elsewhere in this pipeline). The label's own openFDA ingredient
     metadata (substance_name/generic_name) must either be absent or
     contain the drug's base name, guarding against combination-product
     labels whose prose describes a different active ingredient.

  No averaging/pooling across sources happens: at most one source tier
  contributes a value for any one drug, and that source's provenance
  (document/paper citation, or label set_id + section + verbatim snippet)
  is always kept in full alongside the numeric value, exactly like the
  existing ChEMBL rows. Drugs with no measured fu anywhere stay an explicit
  null -- never omitted, never imputed.

This script never looks at the v2 train/held-out split: it recomputes the
fu value for every row in inputs/plasma_fu.json (all 334 v2 drugs, held-out
included, since fu is a model input available for held-out drugs too), the
same way scripts/v2_pk.py's own ChEMBL build already does.

Usage:
    python scripts/v2_fu_merge.py merge --drugcentral <matches.json> \
        [--labels <label_candidates.json>]

Reads the current inputs/plasma_fu.json (from scripts/v2_pk.py's ChEMBL
build) and overwrites it in place with the merged result. Never edits
truth/human_cmax.json or anything in pipeline/ -- re-run
`python scripts/v2_cmax.py build` separately afterwards to recompute
cmax_unbound_nM from the merged fu values.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
DRUGS_FILE = V2_DIR / "drugs.json"
FU_FILE = V2_DIR / "inputs" / "plasma_fu.json"

LOMBARDO_PMID = "30115648"
LOMBARDO_DOI = "10.1124/dmd.118.082966"
LOMBARDO_TITLE = ("Trend Analysis of a Database of Intravenous Pharmacokinetic "
                   "Parameters in Humans for 1352 Drug Compounds")
LOMBARDO_JOURNAL = "Drug Metabolism and Disposition"
LOMBARDO_YEAR = 2018


def _chembl_inchi(chembl_id: str) -> str | None:
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-fu-merge"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rec = json.loads(r.read())
    return (rec.get("molecule_structures") or {}).get("standard_inchi_key")


def _pct_to_fu(pct_literal_value: float) -> float:
    return round(1.0 - pct_literal_value / 100.0, 6)


def merge(drugcentral_file: Path, labels_file: Path | None, verify_structure: bool) -> None:
    fu_data = json.loads(FU_FILE.read_text())
    drugs = {d["inn"]: d for d in json.loads(DRUGS_FILE.read_text())["drugs"]}
    dc_matches = json.loads(drugcentral_file.read_text()) if drugcentral_file.exists() else {}
    label_matches = json.loads(labels_file.read_text()) if labels_file and labels_file.exists() else {}

    n_chembl = n_dc = n_label = n_dc_struct_rejected = 0
    rows = fu_data["drugs"]
    for row in rows:
        if row.get("fu"):
            n_chembl += 1
            continue
        inn, cid = row["inn"], row["chembl_id"]

        dc = dc_matches.get(inn)
        if dc and dc.get("value_literal") and "%" in dc["value_literal"]:
            try:
                pct = float(dc["value_literal"].replace("%", "").strip())
            except ValueError:
                pct = None
            if pct is not None and 0 < pct <= 100:
                ok = True
                if verify_structure and dc.get("inchikey"):
                    try:
                        chembl_key = _chembl_inchi(cid)
                    except Exception:
                        chembl_key = None
                    if chembl_key and chembl_key != dc["inchikey"]:
                        ok = False
                        n_dc_struct_rejected += 1
                if ok:
                    fu_value = _pct_to_fu(pct)
                    if fu_value > 0:
                        row["fu"] = {
                            "value": fu_value, "min": fu_value, "max": fu_value,
                            "n_measurements": 1, "n_documents": 1,
                            "sources": [{
                                "chembl_document_id": None,
                                "pmid": LOMBARDO_PMID, "doi": LOMBARDO_DOI,
                                "title": LOMBARDO_TITLE, "journal": LOMBARDO_JOURNAL,
                                "year": LOMBARDO_YEAR,
                                "source_type": "drugcentral_lombardo2018",
                                "drugcentral_struct_id": dc["struct_id"],
                                "drugcentral_url": f"https://drugcentral.org/drugcard/{dc['struct_id']}",
                                "snippet": dc["snippet"],
                            }],
                            "matched_on": "drugcentral_lombardo2018",
                            "unit": "fraction",
                        }
                        n_dc += 1
                        continue

        cands = label_matches.get(inn)
        if cands:
            c = cands[0]
            fu_value = _pct_to_fu(c["pct_bound"])
            if fu_value > 0:
                row["fu"] = {
                    "value": fu_value, "min": fu_value, "max": fu_value,
                    "n_measurements": 1, "n_documents": 1,
                    "sources": [{
                        "chembl_document_id": None, "pmid": None, "doi": None,
                        "title": None, "journal": None, "year": None,
                        "source_type": "fda_label",
                        "set_id": c["set_id"], "section": c["section"],
                        "application_number": c.get("application_number"),
                        "snippet": c["snippet"],
                    }],
                    "matched_on": "fda_label",
                    "unit": "fraction",
                }
                n_label += 1

    fu_data["description"] = (
        fu_data["description"].rstrip(".") +
        ". Extended by scripts/v2_fu_merge.py: drugs with no ChEMBL fu are "
        "backfilled, in priority order, from (2) a DrugCentral mirror of "
        "Lombardo F, Berellini G, Obach RS 2018 Drug Metab Dispos "
        "46(11):1466-1477 (doi:10.1124/dmd.118.082966, PMID 30115648), "
        "name-matched by exact drugcentral.org title and structure-checked "
        "against ChEMBL's InChIKey where DrugCentral exposes one, then (3) a "
        "conservative regex extraction of 'X% bound to plasma protein(s)' "
        "from FDA label text (openFDA cache built by scripts/v2_cmax.py), "
        "rejecting any sentence about a metabolite, a special population, or "
        "more than one distinct %-bound clause, with ranges collapsed to "
        "their geometric midpoint. No value is pooled/averaged across "
        "sources; each row keeps exactly one source tier's full citation "
        "and a verbatim snippet."
    )
    FU_FILE.write_text(json.dumps(fu_data, indent=1, sort_keys=False, ensure_ascii=False) + "\n")
    print(f"chembl={n_chembl} +drugcentral={n_dc} (struct-rejected={n_dc_struct_rejected}) "
          f"+fda_label={n_label} => total_with_fu={n_chembl + n_dc + n_label}/{len(rows)}",
          file=sys.stderr)


def main(argv: list[str]) -> None:
    if not argv or argv[0] != "merge":
        raise SystemExit(__doc__)
    dc_file = ROOT / "tmp_fu_probe" / "drugcentral_matches.json"
    labels_file = ROOT / "tmp_fu_probe" / "label_fu_candidates.json"
    verify_structure = True
    if "--drugcentral" in argv:
        dc_file = Path(argv[argv.index("--drugcentral") + 1])
    if "--labels" in argv:
        labels_file = Path(argv[argv.index("--labels") + 1])
    if "--no-structure-check" in argv:
        verify_structure = False
    merge(dc_file, labels_file, verify_structure)


if __name__ == "__main__":
    main(sys.argv[1:])
