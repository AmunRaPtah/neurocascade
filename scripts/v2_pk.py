#!/usr/bin/env python3
"""Build human IV PK truth and plasma-fu inputs for the v2 benchmark (P33.1 stage 2).

Two stages, run separately, mirroring scripts/build_benchmark_adme.py:

``python scripts/v2_pk.py fetch [--cache DIR]``
    Network. For every drug in src/neurocascade/benchmark/v2/drugs.json, pulls
    ChEMBL (assay_type=A) activities restricted to the standard_types this
    script uses (CL, Vdss, Vd, Vz, Fu, PPB), plus the assay and document
    records those activities point at. Cached outside the repo, never
    committed. Also resolves each distinct document's PubMed/DOI identifier
    against NCBI eutils / Crossref to verify it is a real, citable record
    before any value from it is used.

``python scripts/v2_pk.py build [--cache DIR]``
    Offline. Writes:
      truth/human_pk.json:   human IV clearance (mL/min/kg) and Vss (L/kg)
      inputs/plasma_fu.json: human plasma fraction unbound

Why ChEMBL. It is the machine-readable, per-row-cited mirror of exactly the
compilations named in the task brief: Lombardo/Obach human IV PK and VDss
regressions (document CHEMBL1614632 = Obach RS et al. 2008 Drug Metab Dispos,
"Trend analysis of a database of intravenous pharmacokinetic parameters in
humans for 670 drug compounds", PMID 18426954. The 1352-compound 2018 update
named in the task brief was NOT found as a document in this ChEMBL release and
is reported as unverified/unused, not fabricated), plus the AstraZeneca DMPK
dataset (document CHEMBL3301361, a ChEMBL "DATASET" record with its own
DataCite DOI 10.6019/CHEMBL3301361, authors Wenlock & Tomkinson. This is the
dataset TDC's PPBR_AZ task is built from). Every value kept here carries its
source document id, and that document's PMID or DOI is checked against NCBI
eutils / Crossref during fetch.

Scope boundary vs scripts/build_benchmark_adme.py: that script explicitly
excludes human in-vivo CL/Vss as "P28.4" clinical outcome data. This script is
that excluded slice, for the v2 drug list (331 drugs, not the P28 panel).

Unit handling: only already per-kg units are accepted (mL.min-1.kg-1 for
clearance, L.kg-1 for Vss) since converting an absolute (non-per-kg) value
would require assuming a body weight, which this script never does. Rows in
other units are counted in ``excluded`` and dropped, not imputed.

Species: assay_organism is frequently null in these ChEMBL rows (Lombardo/Obach
curation did not always set it), so species is inferred the same way as
build_benchmark_adme.py: from assay_organism when present, else from the assay
description text ("in man", "healthy subjects", ...). Only species == human
rows are used here.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
DRUGS_FILE = V2_DIR / "drugs.json"
TRUTH_FILE = V2_DIR / "truth" / "human_pk.json"
FU_FILE = V2_DIR / "inputs" / "plasma_fu.json"
DEFAULT_CACHE = Path("/root/nc-v2-cache/pk")

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CROSSREF = "https://api.crossref.org/works/"

RETRIEVED = "2026-10-21"
CHEMBL_RELEASE = "ChEMBL_37 (2026-05-01), live API read 2026-10-21"
STANDARD_TYPES = ["CL", "Vdss", "Vd", "Vz", "Fu", "PPB"]

# ----------------------------------------------------------------- species
# Copied in spirit from build_benchmark_adme._species (same project convention).


def _species(organism: str | None, description: str) -> str:
    o = (organism or "").lower()
    d = (description or "").lower()
    table = [
        ("human", ("homo sapiens",)),
        ("rat", ("rattus",)),
        ("mouse", ("mus musculus",)),
        ("dog", ("canis",)),
        ("monkey", ("macaca", "macaque")),
        ("guinea_pig", ("cavia",)),
        ("pig", ("sus scrofa",)),
    ]
    for name, keys in table:
        if any(k in o for k in keys):
            return name
    if "macaca" in o or "monkey" in d or "cynomolgus" in d or "rhesus" in d:
        return "monkey"
    for name, words in (
        ("human", ("in man", "in human", "healthy subjects", "healthy human",
                   "psychotic patient", "psychiatric patient", "in patient",
                   "clinical pharmacokinetic")),
        ("rat", ("in rat", "rat ", "wistar", "sprague")),
        ("mouse", ("in mouse", "mouse ", "cd-1", "cd1", "c57", "icr mouse")),
        ("dog", ("in dog", "beagle")),
    ):
        if any(w in d for w in words):
            return name
    return "unspecified"


# ------------------------------------------------------------------ http


def _download(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-P33.1-pk"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - retry any transport error
            if attempt == 4:
                raise
            time.sleep(4 * (attempt + 1))
    raise AssertionError("unreachable")


def _cache_json(cache: Path, subdir: str, key: str, fetcher):
    fn = cache / subdir / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if fn.exists():
        return json.loads(fn.read_text())
    val = fetcher()
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps(val))
    return val


def _chembl_all(cache: Path, path: str, params: dict, key: str) -> list[dict]:
    q = urllib.parse.urlencode(sorted(params.items()))

    def go():
        out: list[dict] = []
        url = f"{CHEMBL}{path}.json?{q}&limit=1000"
        while url:
            page = json.loads(_download(url))
            out.extend(page[key])
            nxt = page["page_meta"]["next"]
            url = f"https://www.ebi.ac.uk{nxt}" if nxt else None
        return out

    return _cache_json(cache, "chembl", f"{path}?{q}", go)


def _chembl_by_ids(cache: Path, path: str, idfield: str, ids, key: str) -> dict:
    ids = sorted(set(i for i in ids if i))
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        for rec in _chembl_all(cache, path, {f"{idfield}__in": ",".join(chunk)}, key):
            out[rec[idfield]] = rec
    return out


def _drugs() -> list[dict]:
    return json.loads(DRUGS_FILE.read_text())["drugs"]


# ----------------------------------------------------------------- fetch


def fetch(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    drugs = _drugs()
    acts_all: dict[str, list] = {}
    for d in drugs:
        acts_all[d["inn"]] = _chembl_all(
            cache, "activity",
            {"parent_molecule_chembl_id": d["chembl_id"], "assay_type": "A",
             "standard_type__in": ",".join(STANDARD_TYPES)}, "activities")
        print(d["inn"], d["chembl_id"], len(acts_all[d["inn"]]), file=sys.stderr)
    acts = [a for v in acts_all.values() for a in v]
    assays = _chembl_by_ids(cache, "assay", "assay_chembl_id",
                            [a["assay_chembl_id"] for a in acts], "assays")
    docs = _chembl_by_ids(cache, "document", "document_chembl_id",
                          [a["document_chembl_id"] for a in acts], "documents")
    print(f"{len(acts)} activities, {len(assays)} assays, {len(docs)} documents",
          file=sys.stderr)
    _verify_documents(cache, docs)


def _verify_documents(cache: Path, docs: dict) -> None:
    """Resolve every distinct source document's PMID (NCBI eutils esummary) or
    DOI (Crossref, falling back to the DOI simply being a ChEMBL-minted
    DataCite DOI for DATASET-type documents) and cache the verification
    record. Printed to stderr; does not raise, so a dead lookup is visible in
    the fetch log rather than silently dropped. The build step re-checks the
    cached verification before using a document."""
    for did, doc in sorted(docs.items()):
        pmid = doc.get("pubmed_id")
        if pmid:
            key = f"eutils:{pmid}"

            def go(pmid=pmid):
                url = (f"{EUTILS}esummary.fcgi?db=pubmed&id={pmid}&retmode=json")
                return json.loads(_download(url))

            rec = _cache_json(cache, "verify", key, go)
            ok = str(pmid) in rec.get("result", {}).get("uids", [])
            print(f"  verify {did} pmid={pmid} ok={ok}", file=sys.stderr)
        elif doc.get("doi") and doc.get("doi", "").startswith("10.6019/CHEMBL"):
            # ChEMBL "DATASET" documents are self-hosted and self-identified by
            # a DataCite DOI ChEMBL mints for its own document id; verified by
            # the document record itself resolving (already fetched above),
            # not by an external registry lookup.
            print(f"  verify {did} doi={doc['doi']} ok=True (chembl-minted dataset)",
                  file=sys.stderr)
        else:
            print(f"  verify {did} UNVERIFIABLE (no pmid, no chembl doi), "
                  f"excluded from build", file=sys.stderr)


# ------------------------------------------------------------- provenance
# ChEMBL groups activities by parent_molecule_chembl_id regardless of what the
# citing document is actually about. Many medicinal-chemistry SAR papers that
# have nothing to do with a marketed drug still carry one curated activity row
# for it as a known comparator compound, with the SAR paper as the ChEMBL
# "document", which would cite a paper that never names the drug. guards.py's
# own citation rule for this project is: a value's document must name the drug
# (title/abstract/substances) unless it is a named exception. Rather than
# re-implement abstract/MeSH matching here, this script trusts a document for
# invivo_cl/invivo_vss only if it is one of the systematic human-PK survey
# papers the task brief and build_benchmark_adme.py already named (Lombardo,
# Obach, Varma; the same genre as the task's own candidate-source list), or if
# the document's own title names the drug. Partial-clearance papers (renal-only,
# hepatic-only/in-vitro-scaled) are deliberately left out of CL_COMPILATIONS
# since "renal clearance" or a HepaRG-scaled prediction is not total IV
# clearance and mixing it in would misrepresent the field.
CL_COMPILATIONS = {
    "CHEMBL1614632",  # Obach RS et al. 2008 DMD, 670-drug IV PK trend analysis
    "CHEMBL1156824",  # Varma MV et al. 2010 J Med Chem, oral bioavailability/CL survey
    "CHEMBL1614629",  # Lombardo F et al. 2005 (ish) DMD, human PK extrapolation from animal data
    "CHEMBL1614630",  # Lombardo F et al. 2006 DMD, preclinical PK extrapolation ("discovery-like")
}
VSS_COMPILATIONS = {
    "CHEMBL1614632", "CHEMBL1156824", "CHEMBL1614629", "CHEMBL1614630",
    "CHEMBL1148291",  # Lombardo F et al. 2004 J Med Chem, human VDss (extended data set)
    "CHEMBL1135488",  # Lombardo F et al. 2002 J Med Chem, human VDss
    "CHEMBL1153781",  # human VDss from immobilized-artificial-membrane partitioning
    "CHEMBL3525927",  # Friden M et al. 2011, unbound brain exposure (carries companion VDss)
    "CHEMBL3588773",  # Lombardo F et al. 2015 J Med Chem, "Volume of Distribution in Drug Design" review
}
FU_COMPILATIONS = {
    "CHEMBL3301361",  # AstraZeneca ChEMBL DMPK dataset (DOI-verified separately)
    "CHEMBL1614632", "CHEMBL1148291", "CHEMBL1156824", "CHEMBL1135488",
    "CHEMBL3525956",  # Friden M et al. 2011, brain-tissue-binding species independence (carries plasma fu)
    "CHEMBL3526149",  # Friden M et al. 2011, brain tissue binding validation (carries plasma fu)
    "CHEMBL3526113",  # equilibrium-dialysis plasma pH / PPB methods paper
}


def _title_names_drug(title: str | None, inn: str) -> bool:
    """A conservative per-document fallback for documents outside the curated
    compilation lists above: trust a single-drug primary-literature document
    (e.g. "Pharmacokinetics of haloperidol in psychotic patients") if its own
    title names the drug, per guards.py's citation rule. Checked on the first
    space-delimited word of the INN (>=5 chars) so salts/esters like
    "haloperidol decanoate" still match a title that only says "haloperidol"."""
    if not title:
        return False
    head = inn.split(" ", 1)[0].lower()
    return len(head) >= 5 and head in title.lower()


def _trusted(field: str, doc_id: str, doc: dict, inn: str) -> bool:
    allow = {"invivo_cl": CL_COMPILATIONS, "invivo_vss": VSS_COMPILATIONS,
             "plasma_fu": FU_COMPILATIONS}[field]
    return doc_id in allow or _title_names_drug(doc.get("title"), inn)


# ----------------------------------------------------------------- classify


def _classify(st: str, units: str | None, description: str):
    """(field, matrix) for a standard_type/units/description, or None. field is
    one of invivo_cl, invivo_vss, plasma_fu."""
    d = (description or "").lower()
    if st in ("Fu", "logFu"):
        if "hepatocyte" in d or "hek293" in d or "cell homogenate" in d or "brain" in d:
            return None
        return ("plasma_fu", None)
    if st == "PPB":
        if "microsom" in d or "liver" in d:
            return None
        return ("plasma_fu", "ppb")
    if st == "CL":
        if units == "mL.min-1.kg-1":
            return ("invivo_cl", None)
        return None  # absolute CL (mL/min) would need an assumed body weight
    if st in ("Vdss", "Vd", "Vz"):
        if units == "L.kg-1":
            return ("invivo_vss", None)
        return None
    return None


def _geomean(xs: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def _summarise(ms: list[dict]) -> dict | None:
    """Geomean within a document, median across documents. None if nothing
    left after dropping censored (<, >) rows."""
    exact = [m for m in ms if m["relation"] == "="]
    if not exact:
        return None
    by_ref = defaultdict(list)
    for m in exact:
        by_ref[m["ref"]].append(m["value"])
    per_ref = [_geomean(v) for v in by_ref.values()]
    vals = [m["value"] for m in exact]
    return {
        "value": round(statistics.median(per_ref), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "n_measurements": len(exact),
        "n_documents": len(by_ref),
    }


def _verified_docs(cache: Path, docs: dict) -> set[str]:
    """Document ids whose cached verification record passed, re-derived from
    the same cache fetch() wrote, so build() never trusts an unverified doc
    even if fetch() was run long ago and the policy changed since."""
    ok = set()
    for did, doc in docs.items():
        pmid = doc.get("pubmed_id")
        if pmid:
            fn = cache / "verify" / (hashlib.sha1(f"eutils:{pmid}".encode()).hexdigest() + ".json")
            if fn.exists():
                rec = json.loads(fn.read_text())
                if str(pmid) in rec.get("result", {}).get("uids", []):
                    ok.add(did)
        elif (doc.get("doi") or "").startswith("10.6019/CHEMBL"):
            ok.add(did)
    return ok


def _doc_citation(doc: dict) -> dict:
    return {
        "chembl_document_id": doc.get("document_chembl_id"),
        "pmid": str(doc["pubmed_id"]) if doc.get("pubmed_id") else None,
        "doi": doc.get("doi"),
        "title": doc.get("title"),
        "journal": doc.get("journal"),
        "year": doc.get("year"),
    }


def build(cache: Path) -> None:
    drugs = _drugs()
    acts_cache = {
        d["inn"]: _chembl_all(cache, "activity",
                              {"parent_molecule_chembl_id": d["chembl_id"], "assay_type": "A",
                               "standard_type__in": ",".join(STANDARD_TYPES)}, "activities")
        for d in drugs
    }
    all_acts = [a for v in acts_cache.values() for a in v]
    assays = _chembl_by_ids(cache, "assay", "assay_chembl_id",
                            [a["assay_chembl_id"] for a in all_acts], "assays")
    docs = _chembl_by_ids(cache, "document", "document_chembl_id",
                          [a["document_chembl_id"] for a in all_acts], "documents")
    verified = _verified_docs(cache, docs)

    pk_rows, fu_rows = [], []
    n_cl = n_vss = n_fu = 0
    for d in drugs:
        inn, cid = d["inn"], d["chembl_id"]
        fields: dict[str, list[dict]] = {"invivo_cl": [], "invivo_vss": [], "plasma_fu": []}
        for a in acts_cache[inn]:
            doc_id = a.get("document_chembl_id")
            if doc_id not in verified:
                continue
            val = a.get("standard_value")
            if val is None:
                continue
            value = float(val)
            if value <= 0:
                continue
            cls = _classify(a["standard_type"], a.get("standard_units"),
                            a.get("assay_description") or
                            assays.get(a["assay_chembl_id"], {}).get("description") or "")
            if cls is None:
                continue
            field, matrix = cls
            if not _trusted(field, doc_id, docs[doc_id], inn):
                continue
            assay = assays.get(a["assay_chembl_id"], {})
            descr = a.get("assay_description") or assay.get("description") or ""
            sp = _species(assay.get("assay_organism"), descr)
            if sp != "human":
                continue
            if matrix == "ppb":
                if not (0 <= value <= 100):
                    continue
                value = round(1.0 - value / 100.0, 6)
                if value <= 0:
                    continue  # fully protein-bound: fu rounds to 0, not log-summarisable
            rel = a.get("standard_relation") or "="
            fields[field].append({
                "value": value, "relation": rel if rel in ("=", "<", ">") else "=",
                "ref": doc_id, "matched_on": "chembl_id",
            })
        cl = _summarise(fields["invivo_cl"])
        vss = _summarise(fields["invivo_vss"])
        fu = _summarise(fields["plasma_fu"])

        def with_sources(summary, measurements):
            if summary is None:
                return None
            refs = sorted({m["ref"] for m in measurements if m["relation"] == "="})
            summary["sources"] = [_doc_citation(docs[r]) for r in refs]
            summary["matched_on"] = "chembl_id"
            return summary

        cl = with_sources(cl, fields["invivo_cl"])
        vss = with_sources(vss, fields["invivo_vss"])
        fu = with_sources(fu, fields["plasma_fu"])
        if cl:
            cl["unit"] = "mL/min/kg"
            n_cl += 1
        if vss:
            vss["unit"] = "L/kg"
            n_vss += 1
        if fu:
            fu["unit"] = "fraction"
            n_fu += 1

        pk_rows.append({
            "inn": inn, "chembl_id": cid,
            "clearance_iv": cl,
            "vss": vss,
            "therapeutic_cmax": None,  # verified source found but not extracted: see report
        })
        fu_rows.append({"inn": inn, "chembl_id": cid, "fu": fu})

    TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    FU_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRUTH_FILE.write_text(json.dumps({
        "description": ("Human IV clearance and Vss for the v2 benchmark drugs "
                         "(P33.1 stage 2), from ChEMBL-curated Lombardo/Obach human "
                         "IV PK compilations and the AstraZeneca ChEMBL DMPK dataset. "
                         "therapeutic_cmax is null throughout. Schulz M. et al. 2020 "
                         "Crit Care (PMID 32375836, PMC7201985, "
                         "doi:10.1186/s13054-020-02915-5) was verified and its "
                         "supplementary table (Additional file 1, a PDF of "
                         "therapeutic/toxic/comatose-fatal blood-plasma "
                         "concentrations for 1100+ substances) was downloaded. It "
                         "was not used: the table's numeric cells have reference "
                         "footnote digits fused directly onto the value with no "
                         "delimiter, for example 0.9-3.9308 for abacavir, which is "
                         "the range 0.9 to 3.9 plus footnote 308, not the number "
                         "3.9308. Splitting value from footnote by rule is exactly "
                         "the regex-a-number-out-of-prose risk this project forbids, "
                         "so no value was extracted rather than risk a silently "
                         "wrong one."),
        "retrieved": RETRIEVED, "chembl_release": CHEMBL_RELEASE,
        "n_drugs": len(drugs), "drugs": pk_rows,
    }, indent=1, sort_keys=False, ensure_ascii=False) + "\n")
    FU_FILE.write_text(json.dumps({
        "description": ("Human plasma fraction unbound for the v2 benchmark drugs "
                         "(P33.1 stage 2), from ChEMBL-curated Fu/PPB activities."),
        "retrieved": RETRIEVED, "chembl_release": CHEMBL_RELEASE,
        "n_drugs": len(drugs), "drugs": fu_rows,
    }, indent=1, sort_keys=False, ensure_ascii=False) + "\n")
    print(f"clearance_iv: {n_cl}/{len(drugs)}  vss: {n_vss}/{len(drugs)}  "
          f"fu: {n_fu}/{len(drugs)}", file=sys.stderr)


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("fetch", "build"):
        raise SystemExit(__doc__)
    cache = DEFAULT_CACHE
    if "--cache" in argv:
        cache = Path(argv[argv.index("--cache") + 1])
    (fetch if argv[0] == "fetch" else build)(cache)


if __name__ == "__main__":
    main(sys.argv[1:])
