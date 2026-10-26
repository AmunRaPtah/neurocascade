#!/usr/bin/env python3
"""Human therapeutic total plasma Cmax for the v2 benchmark drugs (P33.3 data gap).

P33.3 reports free plasma exposure, multi-target occupancy and the
Cmax,u/Ki occupancy baseline as "no data" because v2 had no human Cmax
(``truth/human_pk.json`` carries ``therapeutic_cmax: null`` and is frozen, so
this is a new truth file rather than an edit of that one).

Stages, run separately (mirrors scripts/v2_pk.py):

``python scripts/v2_cmax.py fetch``
    Network/offline-cache. (1) Copies the clinical pharmacology,
    pharmacokinetics, dosage and description sections of every v2 label out of
    the openFDA bulk index built by scripts/v2_label_index.py (commit d661221)
    into ``$NC_V2_CACHE/cmax/labels.jsonl.gz``. (2) Pulls each drug's parent
    molecular weight (``full_mwt`` of the ChEMBL parent, i.e. the free base or
    free acid, which is the moiety label and paper concentrations are reported
    for) into ``$NC_V2_CACHE/cmax/mw.json``. (3) Writes, per drug, every label
    sentence that mentions a peak/maximum concentration next to a
    concentration unit into ``$NC_V2_CACHE/cmax/candidates/<inn>.json`` for
    curation.

``python scripts/v2_cmax.py pubmed <inn> [query]``
    Network. Fallback when no usable label sentence exists: NCBI eutils
    esearch + efetch of abstracts, cached as
    ``$NC_V2_CACHE/cmax/pubmed/<pmid>.json``. Abstract text is stored as
    fetched so a curated snippet can be checked against it offline.

``python scripts/v2_cmax.py build``
    Offline. Reads curation records from ``$NC_V2_CACHE/cmax/curation/*.json``
    (or ``$NC_CMAX_CURATION/*.json``; one record per drug: either a value with its verbatim source snippet or
    ``searched_and_empty`` with what was searched), REJECTS any record whose
    snippet is not a verbatim substring of the cached source text, or whose
    value/unit literal is not in the snippet, converts to ng/mL and nM, joins
    plasma fu from ``inputs/plasma_fu.json`` for Cmax,u, and writes
    ``truth/human_cmax.json``.

Why truth/ and not inputs/. Therapeutic Cmax is a human clinical measurement
that later stages are scored against (free plasma exposure), so it lives with
the other human truth and its held-out rows are stripped from session trees by
``pipeline/guards.hide_heldout``, which strips every ``v2/truth/*.json`` with a
``{"drugs": [{"inn": ...}]}`` shape. No registration is needed for that.

Never from memory: every value must be found in text this script fetched and
cached; the build step enforces that mechanically.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
DRUGS = V2 / "drugs.json"
FU_FILE = V2 / "inputs" / "plasma_fu.json"
OUT = V2 / "truth" / "human_cmax.json"
CACHE = Path(os.environ.get("NC_V2_CACHE", "/root/nc-v2-cache")) / "cmax"
LABELS = CACHE / "labels.jsonl.gz"
MW_FILE = CACHE / "mw.json"
CAND_DIR = CACHE / "candidates"
PUBMED_DIR = CACHE / "pubmed"
CUR_DIR = Path(os.environ.get("NC_CMAX_CURATION", str(CACHE / "curation")))

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SECTIONS = ("clinical_pharmacology", "pharmacokinetics", "dosage_and_administration",
            "description")
# Same local-route set as scripts/v2_liabilities.py (amendment A7): a Cmax from
# an eye drop or a skin cream is not a systemic therapeutic exposure.
LOCAL_ROUTES = {"OPHTHALMIC", "TOPICAL", "OTIC", "CUTANEOUS", "VAGINAL",
                "DENTAL", "INTRAVITREAL", "AURICULAR (OTIC)", "TRANSMUCOSAL",
                "INFILTRATION", "EPIDURAL", "PERINEURAL", "SUBMUCOSAL", "PERCUTANEOUS",
                "INTRADERMAL", "INTRACAVERNOUS", "INTRA-ARTICULAR"}

# Concentration units, as written in labels and abstracts, to ng/mL.
# Mass units only; molar units are converted with the parent MW.
MASS_TO_NG_ML = {
    "pg/mL": 1e-3, "ng/mL": 1.0, "ng/L": 1e-3, "mcg/L": 1.0, "µg/L": 1.0, "ug/L": 1.0,
    "mcg/mL": 1e3, "µg/mL": 1e3, "ug/mL": 1e3, "mg/L": 1e3, "mg/mL": 1e6,
}
MOLAR_TO_NM = {"pmol/mL": 1.0, "nmol/L": 1.0, "nM": 1.0, "pmol/L": 1e-3, "pM": 1e-3,
               "µmol/L": 1e3, "umol/L": 1e3, "µM": 1e3, "uM": 1e3, "nmol/mL": 1e3}
UNITS = tuple(MASS_TO_NG_ML) + tuple(MOLAR_TO_NM)

CMAX_RE = re.compile(r"C\s?max|C\s?ss,?\s?max|peak (?:plasma |serum |blood )?(?:concentration|level)|"
                     r"maximum (?:plasma |serum )?(?:concentration|level)|peak (?:mean )?plasma",
                     re.I)
UNIT_RE = re.compile(r"(?:pg|ng|mcg|µg|μg|ug|mg|nmol|pmol|µmol|μmol|umol)\s?/\s?(?:mL|ml|L|dL)|"
                     r"\b(?:nM|µM|μM|pM)\b")


def clean(s: str) -> str:
    """Whitespace-normalised text; the form snippets are matched against."""
    s = s.replace(" ", " ").replace(" ", " ").replace("μ", "µ")
    return re.sub(r"\s+", " ", s).strip()


def _eutils_slot() -> None:
    """Keep every process on this box under NCBI's 3 requests/s keyless limit."""
    import fcntl
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / ".eutils.lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        last = float(f.read() or 0)
        wait = last + 0.4 - time.time()
        if wait > 0:
            time.sleep(wait)
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))


def _get(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        if url.startswith(EUTILS):
            _eutils_slot()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-v2-cmax"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError(url)


# ------------------------------------------------------------------ fetch

def single_ingredient(label: dict) -> bool:
    of = label.get("openfda") or {}
    subs = of.get("substance_name") or []
    gen = " ".join(of.get("generic_name") or []).lower()
    return len(subs) <= 1 and " and " not in gen and "/" not in gen


def systemic(meta: dict) -> bool:
    routes = {r.upper() for r in meta.get("route") or []}
    return not routes or bool(routes - LOCAL_ROUTES)


def build_label_cache(drugs: list[dict]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from v2_label_index import iter_labels
    want = {m["set_id"] for d in drugs for m in d["labels"]}
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS.with_suffix(".tmp")
    n = 0
    with gzip.open(tmp, "wt") as f:
        for lab in iter_labels():
            if lab.get("set_id") in want:
                f.write(json.dumps({k: lab.get(k) for k in (
                    "set_id", "id", "version", "effective_time", "openfda", *SECTIONS)}) + "\n")
                n += 1
    os.replace(tmp, LABELS)
    print(f"cached {n} of {len(want)} labels -> {LABELS}", file=sys.stderr)


def load_labels() -> dict[str, dict]:
    out = {}
    with gzip.open(LABELS, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["set_id"]] = r
    return out


def label_text(lab: dict, section: str) -> str:
    return clean(" ".join(lab.get(section) or []))


def fetch_mw(drugs: list[dict]) -> None:
    mw = json.loads(MW_FILE.read_text()) if MW_FILE.exists() else {}
    for d in drugs:
        cid = d["chembl_id"]
        if cid in mw:
            continue
        m = json.loads(_get(f"{CHEMBL}molecule/{cid}.json"))
        parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or cid
        if parent != cid:
            m = json.loads(_get(f"{CHEMBL}molecule/{parent}.json"))
        props = m.get("molecule_properties") or {}
        mw[cid] = {"parent_chembl_id": parent,
                   "full_mwt": float(props["full_mwt"]) if props.get("full_mwt") else None,
                   "full_molformula": props.get("full_molformula")}
        MW_FILE.write_text(json.dumps(mw, indent=1))
    print(f"mw: {sum(1 for v in mw.values() if v['full_mwt'])}/{len(drugs)}", file=sys.stderr)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text) if s]


def write_candidates(drugs: list[dict]) -> None:
    labels = load_labels()
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    for d in drugs:
        metas = [m for m in d["labels"] if m["set_id"] in labels
                 and single_ingredient(labels[m["set_id"]]) and systemic(m)]
        # NDA labels first (the innovator's own clinical pharmacology text),
        # oral before other routes, then newest.
        metas.sort(key=lambda m: (not str(m.get("application_number", "")).startswith("NDA"),
                                  "ORAL" not in (m.get("route") or []),
                                  -int(m.get("effective_time") or 0)))
        seen: dict[str, dict] = {}
        for m in metas:
            lab = labels[m["set_id"]]
            for sec in ("clinical_pharmacology", "pharmacokinetics"):
                for s in sentences(label_text(lab, sec)):
                    # Every sentence with a concentration unit is shown to the
                    # curator; cmax_term marks those that name a peak.
                    if UNIT_RE.search(s):
                        key = s.lower()
                        if key not in seen:
                            seen[key] = {"text": s, "cmax_term": bool(CMAX_RE.search(s)),
                                         "set_id": m["set_id"], "section": sec,
                                         "application_number": m.get("application_number"),
                                         "route": m.get("route"), "n_labels_with_text": 0}
                        seen[key]["n_labels_with_text"] += 1
        dosing = []
        for m in metas[:1]:
            lab = labels[m["set_id"]]
            dosing.append({"set_id": m["set_id"],
                           "text": label_text(lab, "dosage_and_administration")[:3000]})
        (CAND_DIR / f"{d['inn']}.json").write_text(json.dumps({
            "inn": d["inn"], "n_systemic_single_ingredient_labels": len(metas),
            "candidates": sorted(seen.values(), key=lambda c: not c["cmax_term"])[:60],
            "dosing": dosing}, indent=1,
            ensure_ascii=False))


def fetch(_argv: list[str]) -> None:
    drugs = json.loads(DRUGS.read_text())["drugs"]
    if not LABELS.exists():
        build_label_cache(drugs)
    if "--candidates-only" not in _argv:
        fetch_mw(drugs)
    write_candidates(drugs)


# ------------------------------------------------------------------ pubmed

def pubmed(argv: list[str]) -> None:
    """``pubmed <inn> [term]`` searches; ``pubmed <inn> --pmids 1,2`` fetches given PMIDs."""
    inn = argv[0]
    if "--pmids" in argv:
        ids = argv[argv.index("--pmids") + 1].split(",")
        term = None
    else:
        term = argv[1] if len(argv) > 1 else (
            f'"{inn}"[Title/Abstract] AND (Cmax OR "peak plasma concentration" OR '
            f'"peak plasma concentrations" OR "maximum plasma concentration") AND humans[MeSH]')
        q = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": 25,
                                    "retmode": "json", "sort": "relevance"})
        ids = json.loads(_get(EUTILS + "esearch.fcgi?" + q))["esearchresult"]["idlist"]
    PUBMED_DIR.mkdir(parents=True, exist_ok=True)
    log = CACHE / "pubmed_searches.jsonl"
    with log.open("a") as f:
        f.write(json.dumps({"inn": inn, "term": term, "pmids": ids}) + "\n")
    todo = [i for i in ids if not (PUBMED_DIR / f"{i}.json").exists()]
    if todo:
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(todo), "retmode": "xml"})
        xml = _get(EUTILS + "efetch.fcgi?" + q).decode()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            abstract = " ".join("".join(a.itertext()) for a in art.iter("AbstractText"))
            journal = art.findtext(".//Journal/Title")
            year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate")
            (PUBMED_DIR / f"{pmid}.json").write_text(json.dumps({
                "pmid": pmid, "title": clean(title), "abstract": clean(abstract),
                "journal": journal, "year": year}, ensure_ascii=False))
        time.sleep(0.4)
    print(f"term: {term}  pmids: {len(ids)}")
    for i in ids:
        p = PUBMED_DIR / f"{i}.json"
        if p.exists():
            r = json.loads(p.read_text())
            hit = [s for s in sentences(r["abstract"]) if UNIT_RE.search(s)
                   or UNIT_RE.search(_unit_norm_text(s))]
            if hit:
                print(json.dumps({"pmid": i, "title": r["title"], "year": r["year"],
                                  "hits": hit}, ensure_ascii=False))


def _unit_norm_text(s: str) -> str:
    """Rewrite ng ml-1 / ng·mL−1 style units as ng/mL, for display matching only."""
    s = s.replace("μ", "µ").replace("−", "-").replace("⁻¹", "-1")
    return re.sub(r"(pg|ng|mcg|µg|ug|mg|nmol|pmol|µmol|umol)\s*[·.\s]\s*(m?[lL])\s*-1\b", r"\1/\2", s)


def check(argv: list[str]) -> None:
    """``check <curation.json>``: verify every record in one curation file."""
    labels = load_labels()
    recs = json.loads(Path(argv[0]).read_text())
    bad = 0
    for r in recs:
        try:
            if r.get("status") == "searched_and_empty":
                if not r.get("searched"):
                    raise ValueError(f"{r['inn']}: searched_and_empty needs 'searched'")
                continue
            for k in ("value", "value_literal", "unit", "dose", "regimen", "state",
                      "population", "source", "snippet"):
                if r.get(k) in (None, ""):
                    raise ValueError(f"{r['inn']}: missing {k}")
            if r["unit"] not in UNITS:
                raise ValueError(f"{r['inn']}: unit {r['unit']!r} not one of {UNITS}")
            if r["state"] not in ("steady_state", "single_dose"):
                raise ValueError(f"{r['inn']}: state must be steady_state or single_dose")
            verify(r, labels)
        except (ValueError, KeyError) as e:
            bad += 1
            print("FAIL", e)
    print(f"{len(recs) - bad}/{len(recs)} records pass")


# ------------------------------------------------------------------ build

def to_ng_ml(value: float, unit: str, mw: float | None) -> float | None:
    if unit in MASS_TO_NG_ML:
        return value * MASS_TO_NG_ML[unit]
    if unit in MOLAR_TO_NM:
        return None if mw is None else value * MOLAR_TO_NM[unit] * mw / 1e3
    raise ValueError(f"unknown unit {unit!r}")


def to_nm(value: float, unit: str, mw: float | None) -> float | None:
    if unit in MOLAR_TO_NM:
        return value * MOLAR_TO_NM[unit]
    if unit in MASS_TO_NG_ML:
        # ng/mL = µg/L; / (g/mol) = µmol/L; x1000 = nmol/L
        return None if mw is None else value * MASS_TO_NG_ML[unit] / mw * 1e3
    raise ValueError(f"unknown unit {unit!r}")


def _unit_key(s: str) -> str:
    s = _unit_norm_text(s).replace(" ", "").lower()
    return s.replace("mcg", "µg").replace("ug/", "µg/").replace("umol", "µmol")


def unit_in(unit: str, snippet: str) -> bool:
    """The canonical unit, or a typographic variant of it, occurs in the snippet."""
    keys = {_unit_key(unit)}
    if unit in ("nmol/L", "nM"):
        keys |= {"nmol/l", "nm"}
    if unit in ("µmol/L", "µM", "umol/L", "uM"):
        keys |= {"µmol/l", "µm"}
    if unit in ("mcg/mL", "µg/mL", "ug/mL"):
        keys |= {"µg/ml"}
    if unit in ("mcg/L", "µg/L", "ug/L"):
        keys |= {"µg/l"}
    s = _unit_key(snippet)
    if any(k in ("nm", "µm", "pm") for k in keys):
        # bare molar symbols need a word boundary (nm also ends "fasting nm..." rarely)
        if re.search(r"\d(?:nm|µm|pm)\b", s):
            return True
    return any(k in s for k in keys if k not in ("nm", "µm", "pm"))


def verify(rec: dict, labels: dict[str, dict]) -> str:
    """Source text the record's snippet was checked against; raises on mismatch."""
    src = rec["source"]
    if src["type"] == "fda_label":
        lab = labels.get(src["set_id"])
        if lab is None:
            raise ValueError(f"{rec['inn']}: set_id {src['set_id']} not in label cache")
        text = label_text(lab, src["section"])
    elif src["type"] == "pubmed":
        p = PUBMED_DIR / f"{src['pmid']}.json"
        if not p.exists():
            raise ValueError(f"{rec['inn']}: PMID {src['pmid']} not in pubmed cache")
        r = json.loads(p.read_text())
        text = clean(r["title"] + " " + r["abstract"])
    else:
        raise ValueError(f"{rec['inn']}: unknown source type {src['type']}")
    for key in ("snippet", "dose_snippet"):
        snip = rec.get(key)
        if snip and clean(snip) not in text:
            raise ValueError(f"{rec['inn']}: {key} is not verbatim in its source")
    if not unit_in(rec["unit"], rec["snippet"]):
        raise ValueError(f"{rec['inn']}: unit {rec['unit']!r} not in snippet")
    if rec.get("statistic") == "range_geomean":
        # The source gives only a range: value is the geometric mean of the
        # two literal ends, both of which must be in the snippet.
        lo, hi = rec["value_low_literal"], rec["value_high_literal"]
        for lit in (lo, hi):
            if lit not in rec["snippet"]:
                raise ValueError(f"{rec['inn']}: range literal {lit!r} not in snippet")
        expect = math.sqrt(_num(lo) * _num(hi))
        if not math.isclose(float(rec["value"]), expect, rel_tol=1e-3):
            raise ValueError(f"{rec['inn']}: value != geomean of range literals")
        if rec.get("value_literal") != f"{lo}-{hi}":
            raise ValueError(f"{rec['inn']}: value_literal for a range must be 'low-high'")
        return text
    if not _literal_in(rec["value_literal"], rec["snippet"]):
        raise ValueError(f"{rec['inn']}: value literal {rec['value_literal']!r} not in snippet")
    if _num(rec["value_literal"]) != float(rec["value"]):
        raise ValueError(f"{rec['inn']}: value != value_literal")
    return text


def _num(lit: str) -> float:
    return float(lit.replace(",", ""))


def _literal_in(lit: str, snippet: str) -> bool:
    """The number occurs in the snippet as a whole number token, so '5' does
    not match inside '50' or '0.5'."""
    return re.search(r"(?<![\d.,])" + re.escape(lit) + r"(?![\d]|[.,]\d)", snippet) is not None


def fu_status(fu_row: dict | None) -> tuple[float | None, list[str]]:
    fu = (fu_row or {}).get("fu")
    if not fu:
        return None, ["fu_missing"]
    flags = []
    if fu.get("n_documents", 0) == 1 and fu.get("n_measurements", 0) == 1:
        flags.append("fu_single_measurement")
    if fu.get("min") and fu.get("max") and fu["max"] / fu["min"] > 3:
        flags.append("fu_sources_disagree_gt3x")
    return float(fu["value"]), flags


def _sig(x: float) -> float:
    """Round to 6 significant figures (decimal rounding loses pg-level values)."""
    return float(f"{x:.6g}")


def _heldout(inn: str) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from neurocascade.benchmark.v2 import v2_heldout
    return v2_heldout(inn)


def build(_argv: list[str]) -> None:
    drugs = json.loads(DRUGS.read_text())["drugs"]
    fu_rows = {r["inn"]: r for r in json.loads(FU_FILE.read_text())["drugs"]}
    mw = json.loads(MW_FILE.read_text())
    labels = load_labels()
    lab_meta = json.loads((CACHE.parent / "labels.meta.json").read_text())
    cur = {}
    for p in sorted(CUR_DIR.glob("*.json")):
        for r in json.loads(p.read_text()):
            if r["inn"] in cur:
                raise ValueError(f"duplicate curation record for {r['inn']} in {p.name}")
            cur[r["inn"]] = r
    rows, problems = [], []
    for d in drugs:
        inn = d["inn"]
        rec = cur.get(inn)
        m = mw.get(d["chembl_id"], {})
        base = {"inn": inn, "chembl_id": d["chembl_id"],
                "parent_chembl_id": m.get("parent_chembl_id"),
                "mw_parent": m.get("full_mwt")}
        if rec is None:
            problems.append(f"{inn}: no curation record")
            continue
        if rec.get("status") == "searched_and_empty":
            if not rec.get("searched"):
                problems.append(f"{inn}: searched_and_empty without a search record")
                continue
            rows.append({**base, "status": "searched_and_empty", "cmax": None,
                         "searched": rec["searched"], "note": rec.get("note")})
            continue
        try:
            verify(rec, labels)
        except ValueError as e:
            problems.append(str(e))
            continue
        value, unit = float(rec["value"]), rec["unit"]
        ng = to_ng_ml(value, unit, base["mw_parent"])
        nm = to_nm(value, unit, base["mw_parent"])
        fu, flags = fu_status(fu_rows.get(inn))
        src = dict(rec["source"])
        if src["type"] == "fda_label":
            lab = labels[src["set_id"]]
            src.update({"version": lab.get("version"), "effective_time": lab.get("effective_time"),
                        "openfda_export_date": lab_meta.get("export_date")})
        else:
            r = json.loads((PUBMED_DIR / f"{src['pmid']}.json").read_text())
            src.update({"title": r["title"], "journal": r["journal"], "year": r["year"]})
        rows.append({**base, "status": "value", "cmax": {
            "value": value, "unit": unit, "value_literal": rec["value_literal"],
            "value_low_literal": rec.get("value_low_literal"),
            "value_high_literal": rec.get("value_high_literal"),
            "statistic": rec.get("statistic"),
            "ng_per_ml": None if ng is None else _sig(ng),
            "nM": None if nm is None else _sig(nm),
            "dose": rec["dose"], "regimen": rec["regimen"],
            "state": rec["state"], "population": rec["population"],
            "route": rec.get("route"),
            "snippet": rec["snippet"], "dose_snippet": rec.get("dose_snippet"),
            "usual_dose_basis": rec.get("usual_dose_basis"),
            # False: the source gives the dose but no label/guideline quote confirms it is the usual one
            "dose_basis_confirmed": bool(rec.get("usual_dose_basis")),
            "note": rec.get("note"), "source": src},
            "fu": fu, "fu_flags": flags,
            "cmax_unbound_nM": None if (fu is None or nm is None) else _sig(nm * fu),
            "cmax_unbound_ng_per_ml": None if (fu is None or ng is None) else _sig(ng * fu)})
    if problems:
        for p in problems:
            print("REJECT", p, file=sys.stderr)
        raise SystemExit(f"{len(problems)} curation problems; nothing written")
    OUT.write_text(json.dumps({
        "description": (
            "Human therapeutic total plasma Cmax at the usual approved adult dose for the "
            "v2 benchmark drugs, with unbound Cmax = Cmax x fu from inputs/plasma_fu.json. "
            "Each value carries the verbatim label or abstract sentence it was read from "
            "(snippet), checked mechanically by scripts/v2_cmax.py build against the cached "
            "source text. ng/mL and nM conversions use the ChEMBL parent full_mwt. "
            "searched_and_empty rows record what was searched. Steady-state is preferred; "
            "state says which was used."),
        "label_export": lab_meta.get("export_date"),
        "mw_source": "ChEMBL API molecule.molecule_properties.full_mwt of the parent compound",
        "fu_source": "inputs/plasma_fu.json",
        "n_drugs": len(drugs),
        "n_with_value": sum(1 for r in rows if r["status"] == "value"),
        "n_train_with_value": sum(1 for r in rows
                                  if r["status"] == "value" and not _heldout(r["inn"])),
        "drugs": rows,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {sum(1 for r in rows if r['status'] == 'value')}/{len(rows)} with value",
          file=sys.stderr)


def main(argv: list[str]) -> None:
    stages = {"fetch": fetch, "pubmed": pubmed, "check": check, "build": build}
    if not argv or argv[0] not in stages:
        raise SystemExit(__doc__)
    stages[argv[0]](argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
