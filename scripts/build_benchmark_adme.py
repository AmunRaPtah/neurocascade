"""Build the Phase 28 preclinical ADME/PK inputs per benchmark drug (P28.3).

Two stages, run separately:

``python scripts/build_benchmark_adme.py fetch [--cache DIR]``
    Network. Pulls, for each benchmark drug's parent molecule (salts included
    through ``parent_molecule_chembl_id``), every ChEMBL ADME activity
    (``assay_type=A``) and the assay and document records those activities point
    at. Cached outside the repository (default ``<tmp>/neurocascade_p28_3``),
    never committed.

``python scripts/build_benchmark_adme.py build [--cache DIR]``
    Offline. Writes ``src/neurocascade/benchmark/adme/<inn>.json`` for every
    benchmark drug and ``coverage.json`` beside them.

Why ChEMBL is the aggregator. The brief names Lombardo 2018 companion data, the
Friden/Loryan Kp,uu compilations, ChEMBL ADME assays and primary rat PK papers.
ChEMBL_37 already carries those compilations as curated activities, each row
keeping its own species, tissue, units and source document (which resolves to a
PMID): Lombardo 2002/2008/2016 and Obach 2008 IV-PK (CHEMBL1135488,
CHEMBL1148291, CHEMBL1614630/1614632, CHEMBL3588773), Friden 2011 brain tissue
binding (CHEMBL3526149, CHEMBL3525956) and Loryan 2011 unbound brain exposure
(CHEMBL3525927). Using ChEMBL as the single machine-readable, per-row-cited
mirror keeps every value traceable to its primary paper without scraping
paywalled supplements. See docs/architecture.md section "P28.3".

The four preclinical fields (brief): plasma fraction unbound, brain fraction
unbound or Kp,uu, in-vitro intrinsic clearance (microsomal/hepatocyte), and
rodent in-vivo CL/Vss. The in-vivo CL/Vss field is NON-HUMAN only: human
in-vivo CL/Vss is clinical outcome data and belongs to P28.4, not to the
preclinical package. Human in-vitro plasma fu and CLint are preclinical and are
kept here. The boundary is enforced in ``_classify`` and recorded per drug under
``excluded``.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "src" / "neurocascade" / "benchmark"
OUT_DIR = BENCH_DIR / "adme"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "neurocascade_p28_3"

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"

RETRIEVED = "2026-10-02"
CHEMBL_RELEASE = "ChEMBL_37 (2026-05-01)"

# The ChEMBL molecule queried when it is not drugs.json's chembl_id. P28.2
# settled flupentixol's identity on the active cis(Z) isomer; this task uses the
# same query molecule so the ADME package is for the same entity as the panel.
CHEMBL_QUERY_OVERRIDE = {"flupentixol": "CHEMBL54661"}

# ----------------------------------------------------------------- species


def _species(organism: str | None, description: str) -> str:
    """Resolve the measurement species. Falls back to the assay description when
    ChEMBL leaves ``assay_organism`` null (Lombardo rows say 'in man')."""
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
        ("chimp", ("pan trog",)),
    ]
    for name, keys in table:
        if any(k in o for k in keys):
            return name
    # organism null or non-specific ("Not specified", "Macaca"): read the text
    if "macaca" in o or "monkey" in d or "cynomolgus" in d or "rhesus" in d:
        return "monkey"
    for name, words in (
        ("human", ("in man", "in human", "healthy subjects", "healthy human",
                   "psychotic patient", "psychiatric patient", "in patient")),
        ("rat", ("in rat", "rat ", "wistar", "sprague")),
        ("mouse", ("in mouse", "mouse ", "cd-1", "cd1", "c57", "icr mouse")),
        ("dog", ("in dog", "beagle")),
    ):
        if any(w in d for w in words):
            return name
    return "unspecified"


RODENT = ("rat", "mouse")
NONHUMAN = ("rat", "mouse", "dog", "monkey", "guinea_pig", "pig", "chimp")

# ----------------------------------------------------------------- classify


def _classify(st: str, units: str | None, description: str, tissue: str | None):
    """Map a ChEMBL ADME activity to (field, matrix) or None.

    Fields: plasma_fu, brain_fu, kp_uu, kp_total, clint,
    invivo_cl_preclinical, invivo_vss_preclinical. ``matrix`` names the in-vitro
    system for clint (microsomes/hepatocytes/s9/intestinal), else None.
    """
    d = (description or "").lower()
    t = (tissue or "").lower()
    is_brain = "brain" in d or "cortex" in t or "cortex" in d or "cerebr" in d

    if st in ("Fu", "logFu"):
        # logFu rows in this corpus carry the de-logged fraction as the standard
        # value (e.g. haloperidol 0.08), so both are read as a fraction unbound.
        if is_brain:
            return ("brain_fu", None)
        if "hepatocyte" in d or "hek293" in d or "cell homogenate" in d:
            return None  # a cell-system fu, not plasma or brain
        if t in ("plasma", "serum", "blood") or "plasma" in d or "serum" in d:
            return ("plasma_fu", None)
        return None
    if st == "PPB":
        if "microsom" in d or "liver" in d:
            return None  # covalent binding in microsomes, not plasma protein binding
        if t in ("plasma", "serum") or "plasma" in d or "serum" in d or "albumin" in d:
            return ("plasma_fu", "ppb")
        return None
    if st == "K(p,uu,brain)":
        return ("kp_uu", None)
    if st == "Kp":
        if units and "cm/s" in units:
            return None  # PAMPA permeability, not a partition coefficient
        return ("kp_total", None)
    if st == "BPR":
        return ("kp_total", None)
    if st == "CL":
        matrix = None
        if "microsom" in d:
            matrix = "microsomes"
        elif "hepatocyte" in d:
            matrix = "hepatocytes"
        elif "s9" in d:
            matrix = "s9"
        elif "cytosol" in d or "intestin" in d:
            matrix = "intestinal_or_cytosol"
        elif units == "uL.min-1.(10^6cells)-1":
            matrix = "hepatocytes"
        elif units == "mL.min-1.g-1" or "intrinsic" in d:
            matrix = "microsomes"
        if matrix is not None:
            return ("clint", matrix)
        if units in ("mL.min-1.kg-1", "ml/min"):
            return ("invivo_cl", None)
        return None
    if st in ("Vdss", "Vd"):
        if units in ("L.kg-1", "L", "mL/min/kg"):
            return ("invivo_vss", None)
        return None
    return None


# ----------------------------------------------------------------- fetching


def _download(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-P28.3"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:  # noqa: BLE001 - retry any transport error
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _chembl_all(cache: Path, path: str, params: dict, key: str) -> list[dict]:
    q = urllib.parse.urlencode(sorted(params.items()))
    fn = cache / "chembl" / (hashlib.sha1(f"{path}?{q}".encode()).hexdigest() + ".json")
    if fn.exists():
        return json.loads(fn.read_text())
    out: list[dict] = []
    url = f"{CHEMBL}{path}.json?{q}&limit=1000"
    while url:
        page = json.loads(_download(url))
        out.extend(page[key])
        nxt = page["page_meta"]["next"]
        url = f"https://www.ebi.ac.uk{nxt}" if nxt else None
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps(out))
    return out


def _chembl_by_ids(cache: Path, path: str, idfield: str, ids, key: str) -> dict:
    ids = sorted(set(i for i in ids if i))
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        for rec in _chembl_all(cache, path, {f"{idfield}__in": ",".join(chunk)}, key):
            out[rec[idfield]] = rec
    return out


def chembl_query_id(drug: dict) -> str:
    return CHEMBL_QUERY_OVERRIDE.get(drug["inn"], drug["chembl_id"])


def _drugs() -> list[dict]:
    return json.loads((BENCH_DIR / "drugs.json").read_text())["drugs"]


def fetch(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    acts_all: dict[str, list] = {}
    for d in _drugs():
        cid = chembl_query_id(d)
        acts_all[d["inn"]] = _chembl_all(
            cache, "activity",
            {"parent_molecule_chembl_id": cid, "assay_type": "A"}, "activities")
        print(d["inn"], cid, len(acts_all[d["inn"]]), file=sys.stderr)
    acts = [a for v in acts_all.values() for a in v]
    _chembl_by_ids(cache, "assay", "assay_chembl_id",
                   [a["assay_chembl_id"] for a in acts], "assays")
    _chembl_by_ids(cache, "document", "document_chembl_id",
                   [a["document_chembl_id"] for a in acts], "documents")


# ----------------------------------------------------------------- building


def _geomean(xs: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def _summarise(measurements: list[dict]) -> list[dict]:
    """One summary row per (species, units, matrix) group. Values are only ever
    pooled within a group, so incompatible units (mL/min/g vs uL/min/10^6cells,
    L/kg vs L) and incompatible in-vitro systems (microsomes vs S9) are never
    averaged into one fabricated number. Within a group: exact (=) values only,
    geometric mean within a document, then median across documents, with spread
    kept visible."""
    groups = defaultdict(list)
    for m in measurements:
        groups[(m["species"], m.get("units"), m.get("matrix"))].append(m)
    out = []
    for (sp, units, matrix), ms in sorted(
            groups.items(), key=lambda kv: tuple(x or "" for x in kv[0])):
        exact = [m for m in ms if m["relation"] == "="]
        row = {"species": sp, "units": units,
               "n_measurements": len(ms), "n_documents": len({m["ref"] for m in ms}),
               "n_censored": len(ms) - len(exact)}
        if matrix is not None:
            row["matrix"] = matrix
        if exact:
            by_ref = defaultdict(list)
            for m in exact:
                by_ref[m["ref"]].append(m["value"])
            per_ref = [_geomean(v) for v in by_ref.values()]
            vals = [m["value"] for m in exact]
            row["value"] = round(statistics.median(per_ref), 6)
            row["min"] = round(min(vals), 6)
            row["max"] = round(max(vals), 6)
        else:
            row["value"] = None
            row["min"] = row["max"] = None
        out.append(row)
    return out


def _build_one(d, acts, assays, docs):
    cid = chembl_query_id(d)
    fields = {k: [] for k in ("plasma_fu", "brain_fu", "kp_uu", "kp_total",
                              "clint", "invivo_cl_preclinical",
                              "invivo_vss_preclinical")}
    excluded = defaultdict(int)
    refs = {}
    used_refs = set()

    def add_ref(doc_id):
        if doc_id in refs:
            return doc_id
        r = docs.get(doc_id, {})
        refs[doc_id] = {
            "pmid": str(r["pubmed_id"]) if r.get("pubmed_id") else None,
            "year": r.get("year"),
            "journal": r.get("journal"),
            "title": r.get("title"),
            "chembl_document_id": doc_id,
        }
        return doc_id

    for a in acts:
        st = a["standard_type"]
        val = a.get("standard_value")
        units = a.get("standard_units")
        rel = a.get("standard_relation") or "="
        assay = assays.get(a["assay_chembl_id"], {})
        descr = a.get("assay_description") or assay.get("description") or ""
        tissue = assay.get("assay_tissue")
        cls = _classify(st, units, descr, tissue)
        if cls is None or val is None:
            continue
        field, matrix = cls
        sp = _species(assay.get("assay_organism"), descr)

        # fraction unbound from PPB%: fu = 1 - PPB/100
        value = float(val)
        ppb_percent = None
        if matrix == "ppb":
            if not (0 <= value <= 100):
                excluded["ppb_out_of_range"] += 1
                continue
            ppb_percent = value
            value = round(1.0 - value / 100.0, 6)
            matrix = None
            units = None  # the stored value is now a fraction unbound, not a percent
        if value <= 0:  # fu/kp/cl/vss must be positive (and log-summarisable)
            excluded["nonpositive_value"] += 1
            continue

        # the human/non-human boundary for in-vivo PK (human -> P28.4)
        if field in ("invivo_cl", "invivo_vss"):
            if sp == "human":
                excluded[f"human_{field}"] += 1
                continue
            if sp not in NONHUMAN:
                excluded[f"species_unresolved_{field}"] += 1
                continue
            field = field + "_preclinical"

        doc_id = a.get("document_chembl_id")
        ref = add_ref(doc_id)
        used_refs.add(ref)
        m = {
            "species": sp,
            "value": value,
            "relation": rel if rel in ("=", "<", ">") else "=",
            "units": units,
            "standard_type": st,
            "assay_chembl_id": a["assay_chembl_id"],
            "ref": ref,
            "description": descr,
        }
        if matrix:
            m["matrix"] = matrix
        if ppb_percent is not None:
            m["ppb_percent"] = ppb_percent
        fields[field].append(m)

    out_fields = {}
    for name, ms in fields.items():
        if not ms:
            continue
        ms.sort(key=lambda m: (m["species"], m.get("matrix", ""), m["ref"]))
        out_fields[name] = {
            "measurements": ms,
            "summary": _summarise(ms),
        }

    has_brain = "brain_fu" in out_fields or "kp_uu" in out_fields
    status = "data" if out_fields else "searched_and_empty"
    return {
        "inn": d["inn"],
        "stratum": d["stratum"],
        "chembl_query_id": cid,
        "retrieved": RETRIEVED,
        "chembl_release": CHEMBL_RELEASE,
        "status": status,
        "fields": out_fields,
        "field_coverage": {
            "plasma_fu": "plasma_fu" in out_fields,
            "brain_fu_or_kp_uu": has_brain,
            "clint": "clint" in out_fields,
            "rodent_in_vivo_cl": any(
                m["species"] in RODENT
                for m in out_fields.get("invivo_cl_preclinical", {}).get("measurements", [])),
            "rodent_in_vivo_vss": any(
                m["species"] in RODENT
                for m in out_fields.get("invivo_vss_preclinical", {}).get("measurements", [])),
        },
        "excluded": dict(sorted(excluded.items())),
        "searched": {"chembl_adme_rows_returned": len(acts)},
        "references": {k: refs[k] for k in sorted(used_refs)},
    }


def build(cache: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drugs = _drugs()
    acts_cache = {
        d["inn"]: _chembl_all(cache, "activity",
                              {"parent_molecule_chembl_id": chembl_query_id(d),
                               "assay_type": "A"}, "activities")
        for d in drugs
    }
    all_acts = [a for v in acts_cache.values() for a in v]
    assays = _chembl_by_ids(cache, "assay", "assay_chembl_id",
                            [a["assay_chembl_id"] for a in all_acts], "assays")
    docs = _chembl_by_ids(cache, "document", "document_chembl_id",
                          [a["document_chembl_id"] for a in all_acts], "documents")

    rows = []
    for d in drugs:
        panel = _build_one(d, acts_cache[d["inn"]], assays, docs)
        (OUT_DIR / f"{d['inn'].replace(' ', '_')}.json").write_text(
            json.dumps(panel, indent=1, sort_keys=True) + "\n")
        fc = panel["field_coverage"]
        rows.append({
            "inn": d["inn"], "stratum": d["stratum"], "status": panel["status"],
            **fc,
            "n_fields": sum(fc.values()),
        })
    _write_coverage(rows, {"retrieved": RETRIEVED, "chembl_release": CHEMBL_RELEASE,
                           "n_drugs": len(drugs)})
    print(f"wrote {len(rows)} ADME records", file=sys.stderr)


def _write_coverage(rows, provenance):
    fields = ["plasma_fu", "brain_fu_or_kp_uu", "clint",
              "rodent_in_vivo_cl", "rodent_in_vivo_vss"]
    summary = {
        "n_drugs": len(rows),
        "n_with_any_field": sum(1 for r in rows if r["status"] == "data"),
        "per_field": {f: sum(1 for r in rows if r[f]) for f in fields},
        "by_stratum": {},
    }
    for s in sorted({r["stratum"] for r in rows}):
        rs = [r for r in rows if r["stratum"] == s]
        summary["by_stratum"][s] = {
            "n": len(rs),
            "with_any_field": sum(1 for r in rs if r["status"] == "data"),
            "per_field": {f: sum(1 for r in rs if r[f]) for f in fields},
        }
    summary["searched_and_empty"] = sorted(r["inn"] for r in rows if r["status"] != "data")
    (OUT_DIR / "coverage.json").write_text(json.dumps(
        {"summary": summary, "drugs": sorted(rows, key=lambda r: r["inn"]),
         "provenance": provenance}, indent=1, sort_keys=True) + "\n")


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("fetch", "build"):
        raise SystemExit(__doc__)
    cache = DEFAULT_CACHE
    if "--cache" in argv:
        cache = Path(argv[argv.index("--cache") + 1])
    (fetch if argv[0] == "fetch" else build)(cache)


if __name__ == "__main__":
    main(sys.argv[1:])
