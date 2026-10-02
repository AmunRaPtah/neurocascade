"""Build the Phase 28 human clinical truth per benchmark drug (P28.4).

Two stages, run separately:

``python scripts/build_benchmark_truth.py fetch [--cache DIR]``
    Network. Caches, outside the repository (default
    ``<tmp>/neurocascade_p28_4``): every openFDA label whose
    ``openfda.generic_name`` matches the drug; SIDER 4.1 (frequency, side-effect,
    name and ATC tables); the WHO ATC/DDD index page of every ATC code; ChEMBL_37
    ADME activities (the same pull as P28.3); and the Europe PMC abstract of every
    PET source in ``PET_OBSERVATIONS``.

``python scripts/build_benchmark_truth.py build [--cache DIR]``
    Offline. Writes ``src/neurocascade/benchmark/truth/<inn>.json`` for every
    benchmark drug, plus ``coverage.json`` and ``effect_labels.json`` beside them.

What each section is, and what it is not, is in docs/architecture.md under
"P28.4". In short:

* ``pk.iv`` - human in-vivo CL, Vss and t1/2 from ChEMBL_37, which carries the
  Obach 2008 and Lombardo 2002-2016 IV compilations row by row. The Lombardo 2018
  1352-drug supplement is not reachable (publisher HTTP 403) and is not in ChEMBL.
* ``pk.label`` and ``dose.label`` - transcriptions from one selected openFDA label
  (``LABEL_TRANSCRIPTIONS``), each carrying the verbatim label sentence. The build
  refuses any transcription whose sentence is not in the cached label text or
  whose numbers are not in its sentence.
* ``dose.ddd`` - the WHO DDD for every ATC code of the drug, parsed from the
  index page as printed (value, unit, route, note).
* ``pet`` - abstract-quoted group occupancies (``PET_OBSERVATIONS``); the build
  refuses any quote that is not a verbatim substring of the Europe PMC abstract.
* ``effects`` - the seven-category clinical-effect label vector under the rule in
  ``EFFECT_RULE``, from SIDER 4.1 frequencies and the selected label's own
  adverse-reaction tables.
"""

from __future__ import annotations

import gzip
import html
import importlib.util
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "src" / "neurocascade" / "benchmark"
OUT_DIR = BENCH_DIR / "truth"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "neurocascade_p28_4"
RETRIEVED = "2026-10-02"

# Hand-curated inputs, each row checked against its cached source at build time.
INPUT_DIR = Path(__file__).resolve().parent / "benchmark_truth_inputs"
PET_OBSERVATIONS = json.loads((INPUT_DIR / "pet_observations.json").read_text())
PET_SEARCHED_EMPTY = json.loads((INPUT_DIR / "pet_searched_empty.json").read_text())
LABEL_TRANSCRIPTIONS = json.loads((INPUT_DIR / "label_transcriptions.json").read_text())

_spec = importlib.util.spec_from_file_location(
    "build_benchmark_adme", Path(__file__).resolve().parent / "build_benchmark_adme.py")
adme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adme)

SIDER = "http://sideeffects.embl.de/media/download/"
SIDER_FILES = ("meddra_freq.tsv.gz", "meddra_all_se.tsv.gz", "drug_names.tsv",
               "drug_atc.tsv")
OPENFDA = "https://api.fda.gov/drug/label.json"
DDD_URL = "https://atcddd.fhi.no/atc_ddd_index/?code={}&showdescription=no"
EPMC = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
        "EXT_ID:{}%20AND%20SRC:MED&format=json&resultType=core")

# US adopted / label names for the same entity (openFDA is keyed by USAN).
US_NAME = {
    "tiotixene": "thiothixene",
    "potassium clorazepate": "clorazepate",
}

# SIDER drug_names.tsv spells some INNs differently; same entity only.
SIDER_NAME = {
    "dosulepin": "dothiepin",
    "tiotixene": "thiothixene",
    "periciazine": "propericiazine",
}

# SIDER's flat compound id merges stereoisomers. Where one flat id holds two
# benchmark entities, the stereo id (column 2) decides which rows are whose.
# CID000146570 is escitalopram, CID006917778 levomilnacipran hydrochloride
# (PubChem titles, read 2026-10-02).
SIDER_STEREO_SPLIT = {
    "citalopram": ("CID100002771", {"CID000002771"}),
    "escitalopram": ("CID100002771", {"CID000146570"}),
    "milnacipran": ("CID100055480", {"CID000055480"}),
    "levomilnacipran": ("CID100055480", {"CID006917778"}),
}

SALT_WORDS = {
    "hydrochloride", "hcl", "hci", "hydrobromide", "maleate", "fumarate",
    "mesylate", "succinate", "sulfate", "tartrate", "oxalate", "dipotassium",
    "edisylate", "lactate", "citrate", "besylate", "hemihydrate", "er", "xl",
    "sr", "oral", "odt", "c-iv", "intensol", "d/r", "capsules", "tablets",
    "extended-release", "(xl)", "(sr)", "er(sr)",
}

# ----------------------------------------------------------------- fetching


def _download(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "neurocascade-P28.4"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if attempt == 4:
                raise
        except Exception:  # noqa: BLE001 - retry any transport error
            if attempt == 4:
                raise
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _drugs() -> list[dict]:
    return json.loads((BENCH_DIR / "drugs.json").read_text())["drugs"]


def _fname(inn: str) -> str:
    return inn.replace(" ", "_")


def _fetch_openfda(cache: Path, d: dict) -> None:
    fn = cache / "openfda" / f"{_fname(d['inn'])}.json"
    if fn.exists():
        return
    name = US_NAME.get(d["inn"], d["inn"])
    q = urllib.parse.quote(f'openfda.generic_name:"{name}"')
    out, skip = [], 0
    while True:
        try:
            page = json.loads(_download(f"{OPENFDA}?search={q}&limit=100&skip={skip}"))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            break
        out.extend(page["results"])
        skip += 100
        if skip >= page["meta"]["results"]["total"]:
            break
        time.sleep(0.3)
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps({"query": name, "results": out}))


def fetch(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for f in SIDER_FILES:
        p = cache / "sider" / f
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_download(SIDER + f))
    drugs = _drugs()
    for d in drugs:
        _fetch_openfda(cache, d)
        for code in d["atc_codes_all"]:
            p = cache / "ddd" / f"{code}.html"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(_download(DDD_URL.format(code)))
                time.sleep(0.2)
    for pmid in sorted({o["pmid"] for o in PET_OBSERVATIONS}):
        p = cache / "epmc" / f"{pmid}.json"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_download(EPMC.format(pmid)))
    adme.fetch(cache / "chembl_adme")


# ----------------------------------------------------------------- openFDA label


def _generic_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s,]+", name.lower()) if t and t not in SALT_WORDS]


def label_candidates(d: dict, results: list[dict]) -> list[dict]:
    """Labels for the drug's own entity: generic name equal to the INN (or its
    USAN) once salt and formulation words are removed. That drops combinations
    (olanzapine and fluoxetine), esters (haloperidol decanoate, paliperidone
    palmitate, aripiprazole lauroxil) and depot salts (olanzapine pamoate)."""
    want = _generic_tokens(US_NAME.get(d["inn"], d["inn"]))
    out = []
    for r in results:
        gn = r.get("openfda", {}).get("generic_name", [])
        if any(_generic_tokens(g) == want for g in gn):
            out.append(r)
    return out


def select_label(d: dict, results: list[dict]) -> dict | None:
    """One label per drug, by a fixed order. Only labels for an oral or
    sublingual product approved under an NDA or ANDA are eligible (a dietary
    supplement "tryptophan" label has no application; amisulpride's only US label is an IV
    antiemetic, a different dose and population). Then: a psychiatric
    indication first (paroxetine's BRISDELLE is for hot flushes, doxepin's
    SILENOR for insomnia at 3-6 mg); an NDA over an ANDA copy; among NDAs the
    lowest application number, i.e. the innovator rather than a 505(b)(2)
    reformulation whose dosing is product-specific (sertraline 150/200 mg
    capsules, a venlafaxine tablet only for patients switching from another
    product); then the latest effective_time; set_id for determinism."""
    cands = [r for r in label_candidates(d, results)
             if {"ORAL", "SUBLINGUAL"} & set(r.get("openfda", {}).get("route") or [])
             and any(a.startswith(("NDA", "ANDA"))
                     for a in r.get("openfda", {}).get("application_number", []))]
    if not cands:
        return None

    def key(r):
        o = r.get("openfda", {})
        psych = is_psychiatric(r)
        ndas = [int(a[3:]) for a in o.get("application_number", [])
                if a.startswith("NDA") and a[3:].isdigit()]
        # NDA numbers are assigned in submission order, so the lowest is the
        # original application: the innovator, not a later reformulation.
        return (psych, bool(ndas), -min(ndas) if ndas else 0,
                r.get("effective_time", ""), r.get("set_id", ""))

    return max(cands, key=key)


def is_psychiatric(r: dict) -> bool:
    text = label_text(r, ("indications_and_usage",))
    return bool(PSYCH_INDICATION.search(text)) and not NOT_PSYCH.search(text)


def label_meta(r: dict) -> dict:
    o = r.get("openfda", {})
    return {
        "set_id": r.get("set_id"),
        "id": r.get("id"),
        "effective_time": r.get("effective_time"),
        "application_number": sorted(o.get("application_number", [])),
        "brand_name": sorted(o.get("brand_name", [])),
        "generic_name": sorted(o.get("generic_name", [])),
        "route": sorted(o.get("route") or []),
        "source": "openFDA drug/label.json (Structured Product Label), retrieved " + RETRIEVED,
    }


def label_text(r: dict, sections) -> str:
    return " ".join(t for s in sections for t in r.get(s, []))


# ----------------------------------------------------------------- AR tables


class _TableParser(HTMLParser):
    """Rows of an SPL table as lists of (text, colspan, rowspan); header rows flagged."""

    def __init__(self):
        super().__init__()
        self.rows: list[tuple[bool, list[tuple[str, int]]]] = []
        self.caption = ""
        self._in_head = False
        self._row = None
        self._cell = None
        self._span = 1
        self._in_caption = False
        self._skip = 0  # inside <footnote>

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._span = int(a.get("colspan") or 1)
            self._rspan = int(a.get("rowspan") or 1)
        elif tag == "caption":
            self._in_caption = True
        elif tag in ("footnote", "sup", "footnoteref"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_head = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append((" ".join("".join(self._cell).split()), self._span, self._rspan))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append((self._in_head, self._row))
            self._row = None
        elif tag == "caption":
            self._in_caption = False
        elif tag in ("footnote", "sup", "footnoteref"):
            self._skip -= 1

    def handle_startendtag(self, tag, attrs):
        if tag == "footnoteref":
            return
        if tag in ("td", "th") and self._row is not None:
            a = dict(attrs)
            self._row.append(("", int(a.get("colspan") or 1), int(a.get("rowspan") or 1)))

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_caption:
            self.caption += data
        elif self._cell is not None:
            self._cell.append(data)


def _grid(rows) -> list[list[str]]:
    """Lay the parsed rows out on a column grid, honouring colspan and rowspan,
    so that a header cell spanning two rows does not shift the columns of the
    row below it (ABILIFY's tables)."""
    grid, carry = [], {}  # carry: column -> (text, rows still to fill)
    for row in rows:
        out, col, cells = [], 0, list(row)
        while cells or any(c >= col for c in carry):
            if col in carry:
                text, left = carry[col]
                out.append(text)
                carry[col] = (text, left - 1)
                if left - 1 == 0:
                    del carry[col]
                col += 1
                continue
            if not cells:
                out.append("")
                col += 1
                continue
            text, span, rspan = cells.pop(0)
            for _ in range(span):
                out.append(text)
                if rspan > 1:
                    carry[col] = (text, rspan - 1)
                col += 1
        grid.append(out)
    return grid


_NUM = re.compile(r"^\s*(<|≤|&lt;)?\s*(\d+(?:\.\d+)?)\s*%?")
DISCONTINUATION = re.compile(r"discontinu|dropout|withdraw", re.I)
PEDIATRIC = re.compile(r"pediatric|paediatric|adolescent|child|aged 1[0-7]|ages? 1[0-7]", re.I)


def normalise_term(raw: str) -> list[str]:
    """A table row label or SIDER term as candidate MedDRA-style terms."""
    t = html.unescape(raw).lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[*\u2020\u2021\u00a7\u00b6#\d]+", " ", t)
    parts = re.split(r"/|;|,| or ", t)
    return [" ".join(p.split()).strip(" .:-") for p in parts if p.strip(" .:-")]


def parse_ar_tables(r: dict, drug_names: list[str]) -> dict:
    """Drug-arm incidences (%) per row term from the label's adverse-reaction
    tables. A column counts as a drug arm only when its stacked header text names
    the drug (INN, USAN or brand) and does not say placebo; comparator columns
    (haloperidol in an olanzapine label) and unlabelled columns are dropped.
    Tables whose caption names a pediatric or adolescent population are skipped,
    so the vector is the adult label. Returns {"rows": [...], "n_tables": ...,
    "n_tables_used": ...}."""
    names = [n.lower() for n in drug_names if n]
    rows_out, used = [], 0
    tables = r.get("adverse_reactions_table", [])
    for ti, t in enumerate(tables):
        p = _TableParser()
        p.feed(t)
        if PEDIATRIC.search(p.caption) or DISCONTINUATION.search(p.caption):
            continue
        rows = _grid([row for _, row in p.rows])
        # Many SPL tables carry no <thead>: the header is every leading row up to
        # the first row with a number in a data cell. A single cell spanning the
        # whole table is a title or footnote, not a column header.
        first = next((i for i, row in enumerate(rows)
                      if any(_NUM.match(c) for c in row[1:])), None)
        if first is None or first == 0:
            continue
        width = max(len(row) for row in rows[first:])
        head = [row for i, row in enumerate(rows[:first]) if len(p.rows[i][1]) > 1]
        body = rows[first:]
        if not head:
            continue
        cols = [" ".join(h[i] for h in head if i < len(h)).lower() for i in range(width)]
        cap = p.caption.lower()
        drug_cols = []
        for i, c in enumerate(cols):
            if i == 0 or "placebo" in c:
                continue
            if any(n in c for n in names) or (
                    any(n in cap for n in names) and "placebo" not in c
                    and re.search(r"\d\s*mg|n\s*=", c) and not _names_other_drug(c, names)):
                drug_cols.append(i)
        if not drug_cols:
            continue
        used += 1
        first_drug = min(drug_cols)
        for b in body:
            if len(b) != width:
                continue
            # the term is the rightmost text cell left of the drug columns
            # ("Body System | Preferred Term | drug | placebo" in older labels)
            term = next((c for c in reversed(b[:first_drug]) if c and not _NUM.match(c)), "")
            if not term:
                continue
            vals = []
            for i in drug_cols:
                m = _NUM.match(b[i])
                if m and not m.group(1):
                    vals.append(float(m.group(2)))
            if vals:
                rows_out.append({"term": term, "max_drug_arm_percent": max(vals),
                                 "table": ti, "caption": " ".join(p.caption.split())})
    return {"rows": rows_out, "n_tables": len(tables), "n_tables_used": used}


_OTHER_DRUG = re.compile(r"haloperidol|risperidone|olanzapine|lithium|valproate|divalproex|"
                         r"placebo|comparator|active control|amitriptyline|imipramine|"
                         r"fluoxetine|paroxetine|lorazepam|diazepam|clozapine|"
                         r"chlorpromazine|quetiapine|aripiprazole|ziprasidone")


def _names_other_drug(col: str, names: list[str]) -> bool:
    return any(m.group(0) not in names for m in _OTHER_DRUG.finditer(col))


def label_drug_names(d: dict, meta: dict) -> list[str]:
    """Names that identify the drug's own arm in a table header: INN, USAN, and
    each brand name and its first word ("Pristiq" of "Pristiq Extended-Release")."""
    names = {d["inn"], US_NAME.get(d["inn"], d["inn"])}
    for b in meta["brand_name"]:
        names.add(b)
        first = b.split()[0]
        if len(first) >= 4:
            names.add(first)
    return sorted(n.lower() for n in names)


NOT_PSYCH = re.compile(r"not (?:approved|indicated) for (?:the treatment of )?any psychiatric", re.I)
PSYCH_INDICATION = re.compile(
    r"schizophren|depress|anxiety|psychos|psychotic|bipolar|mania|manic|panic|"
    r"obsessive|post-?traumatic|agitation|tourette", re.I)


# ----------------------------------------------------------------- effects

# The seven effect categories of the brief, each as the MedDRA preferred/lowest-
# level terms (and their label-table spellings) that count for it. Fixed before
# any prevalence was computed. Tremor is deliberately not EPS: it is a labelled
# effect of SSRIs, TCAs and lithium through non-dopaminergic mechanisms, and
# counting it would make "EPS" mean "tremor" for antidepressants. Plain
# "hypotension" and "dizziness" are not orthostasis for the same reason.
EFFECT_TERMS = {
    "eps": (
        "extrapyramidal disorder", "extrapyramidal symptoms", "extrapyramidal syndrome",
        "extrapyramidal reaction", "extrapyramidal reactions", "extrapyramidal events",
        "extrapyramidal signs", "extrapyramidal symptom", "eps", "akathisia",
        "dystonia", "acute dystonia", "parkinsonism", "drug-induced parkinsonism",
        "parkinsonian syndrome", "dyskinesia", "tardive dyskinesia", "oculogyric crisis",
        "oculogyration", "muscle rigidity", "cogwheel rigidity", "hypokinesia",
        "bradykinesia", "torticollis", "opisthotonus",
    ),
    "sedation": ("sedation", "somnolence", "drowsiness", "hypersomnia", "sleepiness"),
    "weight_gain": ("weight increased", "weight gain", "increased weight",
                    "abnormal weight gain", "weight increase"),
    "anticholinergic": ("dry mouth", "constipation", "urinary retention",
                        "vision blurred", "blurred vision"),
    "orthostasis": ("orthostatic hypotension", "postural hypotension",
                    "hypotension orthostatic", "dizziness postural", "postural dizziness",
                    "orthostatic intolerance"),
    "sexual_dysfunction": (
        "sexual dysfunction", "ejaculation disorder", "ejaculation delayed",
        "ejaculation failure", "abnormal ejaculation", "ejaculatory dysfunction",
        "ejaculation disturbance", "retrograde ejaculation", "erectile dysfunction",
        "impotence", "libido decreased", "decreased libido", "loss of libido",
        "anorgasmia", "orgasm abnormal", "abnormal orgasm", "orgasmic disorder",
        "female orgasmic disorder", "male sexual dysfunction", "female sexual dysfunction",
        "orgasm delayed", "delayed orgasm", "sexual function abnormal",
    ),
    "seizure": ("seizure", "seizures", "convulsion", "convulsions", "grand mal convulsion",
                "epilepsy", "status epilepticus", "generalised tonic-clonic seizure",
                "generalized tonic-clonic seizure", "tonic-clonic seizure"),
}
EFFECTS = tuple(EFFECT_TERMS)
_TERM_TO_EFFECT = {t: e for e, ts in EFFECT_TERMS.items() for t in ts}

# Spellings of one concept, so that a frequency printed for "weight increased"
# covers a label that says "weight gain". Every other term is its own concept.
_SYNONYM_GROUPS = (
    ("extrapyramidal disorder", "extrapyramidal symptoms", "extrapyramidal syndrome",
     "extrapyramidal reaction", "extrapyramidal reactions", "extrapyramidal events",
     "extrapyramidal signs", "extrapyramidal symptom", "eps"),
    ("dystonia", "acute dystonia"),
    ("parkinsonism", "drug-induced parkinsonism", "parkinsonian syndrome"),
    ("oculogyric crisis", "oculogyration"),
    ("muscle rigidity", "cogwheel rigidity"),
    ("somnolence", "drowsiness", "sleepiness"),
    ("weight increased", "weight gain", "increased weight", "abnormal weight gain",
     "weight increase"),
    ("vision blurred", "blurred vision"),
    ("orthostatic hypotension", "postural hypotension", "hypotension orthostatic"),
    ("dizziness postural", "postural dizziness"),
    ("erectile dysfunction", "impotence"),
    ("libido decreased", "decreased libido", "loss of libido"),
    ("ejaculation disorder", "abnormal ejaculation", "ejaculatory dysfunction",
     "ejaculation disturbance"),
    ("orgasm abnormal", "abnormal orgasm", "orgasmic disorder", "female orgasmic disorder"),
    ("orgasm delayed", "delayed orgasm"),
    ("sexual dysfunction", "male sexual dysfunction", "female sexual dysfunction",
     "sexual function abnormal"),
    ("seizure", "seizures", "convulsion", "convulsions", "grand mal convulsion",
     "generalised tonic-clonic seizure", "generalized tonic-clonic seizure",
     "tonic-clonic seizure", "epilepsy", "status epilepticus"),
)
CONCEPT = {t: g[0] for g in _SYNONYM_GROUPS for t in g}

EFFECT_RULE = {
    "threshold_percent": 5.0,
    "positive": "a category term has a drug-arm incidence >= 5% in the drug's own "
                "adult adverse-reaction table (selected openFDA label), or a SIDER 4.1 "
                "non-placebo frequency row whose LOWER bound is >= 0.05",
    "negative": "no category term reaches 5% and EITHER the drug's own adult "
                "adverse-reaction table was parsed (it lists every reaction at or above "
                "its inclusion threshold, 2% or 5% in these labels, so a term absent from "
                "it is below that) OR, with SIDER frequencies only, every category concept "
                "that SIDER or the label names has a printed frequency below 5%",
    "null": "otherwise: no frequency-bearing source, or a category concept is named "
            "(SIDER side effect, label prose) with no frequency anywhere. Unknown is never "
            "defaulted to negative: old labels name haloperidol's EPS and amitriptyline's "
            "dry mouth without a number. Synonyms ('weight gain' / 'weight increased') "
            "are one concept",
    "why_5_percent": "the incidence most US labels use for their Highlights 'most "
                     "common adverse reactions'; the CIOMS 'common' boundary (1%) admits "
                     "almost every term in a psychiatric label table. Fixed before any "
                     "prevalence was computed; every per-term maximum is stored so another "
                     "threshold can be applied without refetching",
    "why_lower_bound": "SIDER encodes a frequency word as a range ('frequent' = 0.01-1); "
                       "only the lower bound is a frequency the label actually asserts",
    "postmarketing": "SIDER rows described 'postmarketing' (stored as 0-0.001) are not "
                     "frequency evidence: a spontaneous report has no denominator. They "
                     "count toward the listed vector only",
    "not_applied": "the 'twice placebo' half of the Highlights convention: SIDER placebo "
                   "rows are not paired to drug rows by trial, so it cannot be applied "
                   "uniformly",
    "listed_vector": "1 if any category term appears in SIDER 4.1 side effects for the "
                     "drug or in the selected label's adverse-reactions, warnings, "
                     "precautions or boxed-warning text, at any or no frequency; 0 if the "
                     "drug has a source and no mention; null if neither source exists",
}


def effect_term(raw: str) -> tuple[str, str] | None:
    """(effect, concept) for a SIDER term or table row label, else None."""
    for t in normalise_term(raw):
        m = re.match(r"^.*\bdisorders\s+(.+)$", t)  # merged "SOC  Preferred Term" cells
        for cand in (t, m.group(1) if m else None):
            if cand in _TERM_TO_EFFECT:
                return _TERM_TO_EFFECT[cand], CONCEPT.get(cand, cand)
    return None


def effect_of(raw: str) -> str | None:
    hit = effect_term(raw)
    return hit[0] if hit else None


def _text_mentions(text: str) -> set[tuple[str, str]]:
    """(effect, concept) pairs named anywhere in label prose."""
    low = " ".join(html.unescape(text).lower().split())
    found = set()
    for e, terms in EFFECT_TERMS.items():
        for t in terms:
            if t == "eps":
                continue
            if re.search(r"\b" + re.escape(t) + r"\b", low):
                found.add((e, CONCEPT.get(t, t)))
    return found


# ----------------------------------------------------------------- SIDER


def load_sider(cache: Path) -> dict:
    sd = cache / "sider"
    names = {}
    for line in (sd / "drug_names.tsv").read_text().splitlines():
        cid, name = line.split("\t")
        names[cid] = name
    atc = defaultdict(set)
    for line in (sd / "drug_atc.tsv").read_text().splitlines():
        cid, code = line.split("\t")
        atc[cid].add(code)
    freq = defaultdict(list)
    with gzip.open(sd / "meddra_freq.tsv.gz", "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            freq[c[0]].append({"stereo": c[1], "placebo": c[3] == "placebo",
                               "description": c[4], "lower": float(c[5]),
                               "upper": float(c[6]), "meddra_type": c[7],
                               "term": c[9]})
    se = defaultdict(list)
    with gzip.open(sd / "meddra_all_se.tsv.gz", "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            se[c[0]].append({"stereo": c[1], "meddra_type": c[3], "term": c[5]})
    return {"names": names, "atc": atc, "freq": freq, "se": se}


def sider_match(d: dict, sider: dict) -> dict | None:
    """The SIDER flat id (and allowed stereo ids) for the drug's own entity.

    Matched by SIDER's drug name equal to the INN (or a same-entity spelling in
    SIDER_NAME). When a name matches several flat ids, only those whose SIDER ATC
    codes include one of the drug's own are kept (fluphenazine: the hydrochloride
    record, not the decanoate's). An ATC-only match is never used: SIDER maps
    clopenthixol's code to zuclopenthixol and esketamine's to racemic ketamine."""
    if d["inn"] in SIDER_STEREO_SPLIT:
        flat, stereo = SIDER_STEREO_SPLIT[d["inn"]]
        return {"flat": [flat], "stereo": sorted(stereo), "match": "stereo_split"}
    want = SIDER_NAME.get(d["inn"], d["inn"]).lower()
    flats = sorted(c for c, n in sider["names"].items() if n.lower() == want)
    if len(flats) > 1:
        own = set(d["atc_codes_all"])
        flats = [c for c in flats if sider["atc"].get(c, set()) & own]
    if not flats:
        return None
    return {"flat": flats, "stereo": None,
            "match": "name" if want == d["inn"] else "name_alias"}


def _sider_rows(rows, m):
    return [r for r in rows if m["stereo"] is None or r["stereo"] in m["stereo"]]



# ----------------------------------------------------------------- WHO DDD


def parse_ddd(page: str, code: str) -> dict:
    """The DDD rows printed for one ATC code: value, unit, route, note. An ATC
    code with no assigned DDD has a name row and empty DDD cells; that is
    returned as ``ddd: []`` with the name, never filled in."""
    m = re.search(r"<table[^>]*>(.*?)</table>", page, re.S)
    rows = re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S) if m else []
    out, name = [], None
    current = None
    for r in rows[1:]:
        cells = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).replace("\xa0", " ").split())
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)]
        if len(cells) < 6:
            continue
        if cells[0]:
            current = cells[0]
            if current == code:
                name = cells[1]
        if current != code or not cells[2]:
            continue
        out.append({"ddd": float(cells[2]), "unit": cells[3], "route": cells[4],
                    "note": cells[5] or None})
    upd = re.search(r"Last updated:\s*([\d-]+)", page)
    return {"atc_code": code, "who_name": name, "ddd": out,
            "index_last_updated": upd.group(1) if upd else None}


# ----------------------------------------------------------------- ChEMBL PK

_INVITRO = re.compile(r"microsom|hepatocyte|\bs9\b|intrinsic|stability|cytosol|"
                      r"\bcyp|ugt|heparg|liver model", re.I)


def classify_human_pk(st: str, units: str | None, descr: str) -> tuple[str, str] | None:
    """(field, route) for a human in-vivo CL, Vss or terminal half-life row, else
    None. Only total systemic quantities: hepatic, renal, unbound and intrinsic
    clearances, absorption half-lives, metabolite or enantiomer half-lives, DDI
    arms and in-vitro half-lives are refused. Oral CL is CL/F, not CL, and is
    refused here (oral apparent parameters come from the label)."""
    d = descr.lower()
    if _INVITRO.search(d):
        return None
    route = ("iv" if re.search(r"\biv\b|intravenous", d) else
             "oral" if re.search(r"\bpo\b|oral", d) else "unspecified")
    if re.search(r"\biv\b|intravenous", d) and re.search(r"\bpo\b|oral", d):
        return None  # mixed-route study row: cannot tell which the value is
    if st == "CL" and units == "mL.min-1.kg-1":
        if re.search(r"hepatic|renal|unbound|impairment|apparent|hepatobiliary", d):
            return None
        if not re.search(r"clearance", d) or route == "oral":
            return None
        return ("cl_mL_min_kg", route)
    if st == "Vdss" and units == "L.kg-1":
        if "unbound" in d or route == "oral":
            return None
        return ("vss_L_kg", route)
    if st == "T1/2" and units == "hr":
        if re.search(r"absorption|co-treated|hydroxy|metabolite|radioactivity|"
                     r"[rs]-bupropion|\(r|\(s", d):
            return None
        if not re.search(r"half.?life", d):
            return None
        return ("t_half_h", route)
    return None


def human_pk(d: dict, acts: list, assays: dict, docs: dict) -> dict:
    fields = defaultdict(list)
    refs = {}
    for a in acts:
        val = a.get("standard_value")
        if val is None:
            continue
        assay = assays.get(a["assay_chembl_id"], {})
        descr = a.get("assay_description") or assay.get("description") or ""
        if adme._species(assay.get("assay_organism"), descr) != "human":
            continue
        cls = classify_human_pk(a["standard_type"], a.get("standard_units"), descr)
        if cls is None:
            continue
        value = float(val)
        if value <= 0:
            continue
        field, route = cls
        doc_id = a.get("document_chembl_id")
        r = docs.get(doc_id, {})
        refs[doc_id] = {"pmid": str(r["pubmed_id"]) if r.get("pubmed_id") else None,
                        "year": r.get("year"), "journal": r.get("journal"),
                        "title": r.get("title"), "chembl_document_id": doc_id}
        rel = a.get("standard_relation") or "="
        fields[field].append({
            "species": "human", "value": value,
            "relation": rel if rel in ("=", "<", ">") else "=",
            "units": a.get("standard_units"), "route": route,
            "standard_type": a["standard_type"],
            "assay_chembl_id": a["assay_chembl_id"], "ref": doc_id,
            "description": descr,
        })
    out = {}
    for f, ms in sorted(fields.items()):
        ms.sort(key=lambda m: (m["ref"] or "", m["assay_chembl_id"]))
        # summary over every route-compatible row: route is a grouping key, so an
        # unspecified-route compilation value is never pooled with an explicit-IV one
        summ = adme._summarise([{**m, "matrix": m["route"]} for m in ms])
        for row in summ:
            row["route"] = row.pop("matrix")
        out[f] = {"measurements": ms, "summary": summ}
    return {"fields": out, "references": dict(sorted(refs.items()))}


# ----------------------------------------------------------------- PET


def _abstract(cache: Path, pmid: str) -> str:
    res = json.loads((cache / "epmc" / f"{pmid}.json").read_text())["resultList"]["result"]
    if not res:
        raise ValueError(f"Europe PMC returned nothing for PMID {pmid}")
    return res[0].get("abstractText", "")


def _numbers_in(text: str) -> set[float]:
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def verify_pet(cache: Path) -> list[dict]:
    """Every PET observation, refused unless its quote is a verbatim substring of
    the cached Europe PMC abstract and every occupancy number is printed in it.
    A daily dose that is arithmetic on a printed dose (300 mg b.i.d.) must say so
    in ``dose_derivation``."""
    out = []
    for o in PET_OBSERVATIONS:
        abstract = _abstract(cache, o["pmid"])
        if o["quote"] not in abstract:
            raise ValueError(f"PET quote not in abstract of PMID {o['pmid']}: {o['quote'][:80]}")
        printed = _numbers_in(o["quote"])
        occ = o["occupancy_percent"]
        for v in occ if isinstance(occ, list) else [occ]:
            if float(v) not in printed:
                raise ValueError(f"occupancy {v} not printed in quote, PMID {o['pmid']}")
        dose = o["dose_mg_per_day"]
        for v in dose if isinstance(dose, list) else [dose]:
            if float(v) not in printed and not o.get("dose_derivation"):
                raise ValueError(f"dose {v} not printed and not derived, PMID {o['pmid']}")
        out.append(o)
    return out


# ----------------------------------------------------------------- label transcriptions


def verify_transcription(inn: str, t: dict, label: dict) -> dict:
    """A hand transcription from the selected label. Its set_id must be the
    selected label's, each quoted sentence must be in that label's text (white
    space normalised), and every number it reports must be printed in its
    quotes."""
    if t["set_id"] != label.get("set_id"):
        raise ValueError(f"{inn}: transcription is from set_id {t['set_id']}, "
                         f"selected label is {label.get('set_id')}")
    text = " ".join(" ".join(label.get(s, [])) for s in label if isinstance(label.get(s), list)
                    and label.get(s) and isinstance(label[s][0], str))
    norm = " ".join(text.split())
    for item in t.get("items", []):
        for q in item["quotes"]:
            if " ".join(q.split()) not in norm:
                raise ValueError(f"{inn}: quote not in label: {q[:80]}")
        printed = set().union(*(_numbers_in(q) for q in item["quotes"]))
        for k in ("value", "low", "high"):
            v = item.get(k)
            if v is not None and float(v) not in printed and not item.get("derivation"):
                raise ValueError(f"{inn}: {item['quantity']} {k}={v} not printed in its quotes")
    return t



# ----------------------------------------------------------------- build


def effects_for(d: dict, sider: dict, label: dict | None, meta: dict | None) -> dict:
    """The effect label vector and its evidence for one drug (rule: EFFECT_RULE)."""
    evidence = {e: [] for e in EFFECTS}
    named = {e: set() for e in EFFECTS}  # concepts the sources name, at any frequency
    with_freq = {e: set() for e in EFFECTS}  # concepts with a frequency
    sider_freq = table_freq = False
    sources = {}

    m = sider_match(d, sider)
    if m:
        freq = [r for flat in m["flat"] for r in _sider_rows(sider["freq"].get(flat, []), m)]
        se = [r for flat in m["flat"] for r in _sider_rows(sider["se"].get(flat, []), m)]
        # SIDER encodes a postmarketing (spontaneous) report as frequency 0-0.001.
        # A spontaneous report has no denominator, so that range is SIDER's
        # placeholder, not a frequency the label asserts: such a row names the
        # term and is never frequency evidence.
        drug_freq = [r for r in freq if not r["placebo"] and r["description"] != "postmarketing"]
        sider_freq = bool(drug_freq)
        sources["sider"] = {
            "flat_ids": m["flat"], "stereo_ids": m["stereo"], "match": m["match"],
            "n_side_effect_rows": len(se), "n_frequency_rows_drug": len(drug_freq),
            "n_frequency_rows_postmarketing": sum(
                1 for r in freq if r["description"] == "postmarketing"),
            "n_frequency_rows_placebo": sum(1 for r in freq if r["placebo"])}
        for r in se + freq:
            hit = effect_term(r["term"])
            if hit:
                named[hit[0]].add(hit[1])
        best = {}
        for r in drug_freq:
            hit = effect_term(r["term"])
            if not hit:
                continue
            with_freq[hit[0]].add(hit[1])
            k = (hit[0], r["term"].lower())
            if k not in best or r["lower"] > best[k]["lower"]:
                best[k] = r
        for (e, _), r in sorted(best.items()):
            evidence[e].append({"source": "SIDER 4.1", "term": r["term"],
                                "frequency_lower": r["lower"], "frequency_upper": r["upper"],
                                "description": r["description"]})
    if label is not None:
        tab = parse_ar_tables(label, label_drug_names(d, meta))
        table_freq = bool(tab["rows"])
        sources["label"] = {"set_id": meta["set_id"], "n_ar_tables": tab["n_tables"],
                            "n_ar_tables_used": tab["n_tables_used"],
                            "n_rows_with_drug_arm_value": len(tab["rows"])}
        best = {}
        for r in tab["rows"]:
            hit = effect_term(r["term"])
            if not hit:
                continue
            named[hit[0]].add(hit[1])
            with_freq[hit[0]].add(hit[1])
            k = (hit[0], " ".join(r["term"].lower().split()))
            if k not in best or r["max_drug_arm_percent"] > best[k]["max_drug_arm_percent"]:
                best[k] = r
        for (e, _), r in sorted(best.items()):
            evidence[e].append({"source": "openFDA label table", "term": r["term"],
                                "drug_arm_percent": r["max_drug_arm_percent"],
                                "table_caption": r["caption"]})
        text = label_text(label, ("adverse_reactions", "warnings_and_cautions", "warnings",
                                  "precautions", "boxed_warning"))
        for e, concept in _text_mentions(text):
            named[e].add(concept)

    def pct(ev):
        return ev["frequency_lower"] * 100 if "frequency_lower" in ev else ev["drug_arm_percent"]

    labels, listed_out, max_pct, unquantified = {}, {}, {}, {}
    any_source = bool(sources)
    for e in EFFECTS:
        mx = max((pct(ev) for ev in evidence[e]), default=None)
        max_pct[e] = round(mx, 4) if mx is not None else None
        unquantified[e] = sorted(named[e] - with_freq[e])
        if mx is not None and mx >= EFFECT_RULE["threshold_percent"]:
            labels[e] = 1
        elif table_freq:
            # the drug's own adult table lists every reaction at or above its
            # inclusion threshold (2% or 5% in these labels): absent means below
            labels[e] = 0
        elif sider_freq and not unquantified[e]:
            labels[e] = 0
        else:
            labels[e] = None
        listed_out[e] = int(bool(named[e])) if any_source else None
    return {"label": labels, "listed": listed_out, "max_percent": max_pct,
            "named_without_frequency": unquantified,
            "has_frequency_source": sider_freq or table_freq,
            "has_label_table": table_freq, "evidence": evidence, "sources": sources}


def _pk_label_and_dose(inn: str, label: dict | None) -> tuple[dict | None, list]:
    t = LABEL_TRANSCRIPTIONS.get(inn)
    if t is None or label is None:
        return None, []
    t = verify_transcription(inn, t, label)
    return t, t.get("items", [])


def build(cache: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drugs = _drugs()
    split = json.loads((BENCH_DIR / "split.json").read_text())
    side = {i: "train" for i in split["train"]} | {i: "heldout" for i in split["heldout"]}
    sider = load_sider(cache)
    pet = verify_pet(cache)
    acache = cache / "chembl_adme"
    acts_by = {d["inn"]: adme._chembl_all(acache, "activity",
                                          {"parent_molecule_chembl_id": adme.chembl_query_id(d),
                                           "assay_type": "A"}, "activities")
               for d in drugs}
    all_acts = [a for v in acts_by.values() for a in v]
    assays = adme._chembl_by_ids(acache, "assay", "assay_chembl_id",
                                 [a["assay_chembl_id"] for a in all_acts], "assays")
    docs = adme._chembl_by_ids(acache, "document", "document_chembl_id",
                               [a["document_chembl_id"] for a in all_acts], "documents")
    unknown = set(LABEL_TRANSCRIPTIONS) - {d["inn"] for d in drugs}
    if unknown:
        raise ValueError(f"label transcriptions for non-benchmark drugs: {sorted(unknown)}")

    rows, matrix = [], {}
    for d in drugs:
        inn = d["inn"]
        fda = json.loads((cache / "openfda" / f"{_fname(inn)}.json").read_text())
        label = select_label(d, fda["results"])
        meta = label_meta(label) if label else None
        if label is not None:
            meta["indication_psychiatric"] = is_psychiatric(label)
        transcription, items = _pk_label_and_dose(inn, label)

        ddd = []
        for code in d["atc_codes_all"]:
            page = (cache / "ddd" / f"{code}.html").read_text(encoding="latin-1")
            ddd.append(parse_ddd(page, code))

        pk_iv = human_pk(d, acts_by[inn], assays, docs)
        obs = [o for o in pet if o["drug"] == inn]
        eff = effects_for(d, sider, label, meta)

        def items_of(kind):
            return [i for i in items if i["kind"] == kind]

        rec = {
            "inn": inn, "stratum": d["stratum"], "split": side[inn],
            "retrieved": RETRIEVED,
            "pk": {
                "iv_compilations": pk_iv["fields"],
                "label": items_of("pk"),
            },
            "dose": {"ddd": ddd, "label": items_of("dose")},
            "pet": obs,
            "effects": eff,
            "label": meta,
            "label_transcription": (None if transcription is None else
                                    {k: v for k, v in transcription.items() if k != "items"}),
            "searched": {
                "openfda_query": f'openfda.generic_name:"{fda["query"]}"',
                "openfda_labels_returned": len(fda["results"]),
                "openfda_labels_same_entity": len(label_candidates(d, fda["results"])),
                "chembl_adme_rows_returned": len(acts_by[inn]),
                "pet_searched_empty": [e for e in PET_SEARCHED_EMPTY if e["drug"] == inn],
            },
            "references": pk_iv["references"],
        }
        (OUT_DIR / f"{_fname(inn)}.json").write_text(
            json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        pk = rec["pk"]
        cov = {
            "inn": inn, "stratum": d["stratum"], "split": side[inn],
            "iv_cl": "cl_mL_min_kg" in pk["iv_compilations"],
            "iv_vss": "vss_L_kg" in pk["iv_compilations"],
            "iv_t_half": "t_half_h" in pk["iv_compilations"],
            "label_t_half": any(i["quantity"] == "t_half_h" for i in pk["label"]),
            "ddd_oral": any(r["route"] == "O" for x in ddd for r in x["ddd"]),
            "ddd_any": any(x["ddd"] for x in ddd),
            "label_dose_range": bool(rec["dose"]["label"]),
            "pet_d2": any(o["target"] == "D2" for o in obs),
            "pet_5ht2a": any(o["target"] == "5-HT2A" for o in obs),
            "pet_sert": any(o["target"] == "SERT" for o in obs),
            "openfda_label": label is not None,
            "sider": "sider" in eff["sources"],
            "effect_vector_known": eff["has_frequency_source"],
        }
        rows.append(cov)
        matrix[inn] = {"split": side[inn], "stratum": d["stratum"],
                       "label": eff["label"], "listed": eff["listed"]}
    _write_coverage(rows)
    _write_effect_matrix(matrix)
    print(f"wrote {len(rows)} truth records", file=sys.stderr)


COVERAGE_FIELDS = ("iv_cl", "iv_vss", "iv_t_half", "label_t_half", "ddd_oral", "ddd_any",
                   "label_dose_range", "pet_d2", "pet_5ht2a", "pet_sert", "openfda_label",
                   "sider", "effect_vector_known")


def _count(rows):
    return {f: sum(1 for r in rows if r[f]) for f in COVERAGE_FIELDS}


def _write_coverage(rows):
    summary = {"n_drugs": len(rows), "per_field": _count(rows), "by_stratum": {},
               "by_split": {}}
    for s in sorted({r["stratum"] for r in rows}):
        rs = [r for r in rows if r["stratum"] == s]
        summary["by_stratum"][s] = {"n": len(rs), **_count(rs)}
    for s in ("train", "heldout"):
        rs = [r for r in rows if r["split"] == s]
        summary["by_split"][s] = {"n": len(rs), **_count(rs)}
    (OUT_DIR / "coverage.json").write_text(json.dumps(
        {"summary": summary, "drugs": sorted(rows, key=lambda r: r["inn"]),
         "provenance": {"retrieved": RETRIEVED,
                        "sources": ["openFDA drug/label.json", "SIDER 4.1 (2015-10-21)",
                                    "WHO ATC/DDD index (atcddd.fhi.no)",
                                    "ChEMBL_37 (2026-05-01) ADME activities",
                                    "Europe PMC abstracts"]}},
        indent=1, sort_keys=True) + "\n")


def _write_effect_matrix(matrix):
    def prevalence(vectors, which):
        out = {}
        for e in EFFECTS:
            known = [v[which][e] for v in vectors if v[which][e] is not None]
            out[e] = {"n_known": len(known), "n_positive": sum(known),
                      "prevalence": round(sum(known) / len(known), 4) if known else None}
        return out

    vecs = list(matrix.values())
    prev = {"all": {"label": prevalence(vecs, "label"), "listed": prevalence(vecs, "listed")}}
    for s in ("train", "heldout"):
        vs = [v for v in vecs if v["split"] == s]
        prev[s] = {"label": prevalence(vs, "label"), "listed": prevalence(vs, "listed")}
    for st in sorted({v["stratum"] for v in vecs}):
        vs = [v for v in vecs if v["stratum"] == st]
        prev[st] = {"label": prevalence(vs, "label")}
    (OUT_DIR / "effect_labels.json").write_text(json.dumps(
        {"effects": list(EFFECTS), "terms": {e: list(t) for e, t in EFFECT_TERMS.items()},
         "rule": EFFECT_RULE, "prevalence": prev, "drugs": matrix},
        indent=1, sort_keys=True) + "\n")


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("fetch", "build"):
        raise SystemExit(__doc__)
    cache = DEFAULT_CACHE
    if "--cache" in argv:
        cache = Path(argv[argv.index("--cache") + 1])
    (fetch if argv[0] == "fetch" else build)(cache)


if __name__ == "__main__":
    main(sys.argv[1:])
