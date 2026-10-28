"""Deterministic, conservative extraction of human plasma fu from openFDA
label text already cached by scripts/v2_cmax.py fetch
($NC_V2_CACHE/cmax/labels.jsonl.gz). Looks for sentences of the form
"X% bound to (human) plasma protein(s)" (or "plasma protein binding is X%",
"X% of <drug> is bound to plasma proteins") in clinical_pharmacology /
pharmacokinetics / description sections, converts %bound -> fu = 1 - %/100
(done downstream, in scripts/v2_fu_merge.py -- this module only extracts the
literal %-bound value and never itself converts to fu).
Ranges -> geometric midpoint of the two endpoints. Rejects any sentence
mentioning a metabolite or a special population (renal/hepatic impairment,
pediatric, geriatric, pregnancy, elderly) since those are not the drug's own
normal-adult plasma protein binding.

Two extra guards, added after an independent review of PR #52 found two
wrong-drug extractions that slipped past the original openfda-metadata-only
name check (both offending labels had empty openfda metadata, which made
_name_matches() pass unconditionally):

  - A sentence-level subject check: if a sentence has the shape "<N>% of
    <subject> is/are bound", the captured <subject> must itself name the
    target drug (or a salt form of it) -- independent of openfda metadata.
    This caught procaine's fu being read from a sentence about "penicillin
    G" and ethyl chloride's fu being read from a sentence about "unesterified
    EPA", both via labels with empty openfda metadata.

  - A censored-value guard: sentences phrased as an inequality ("<X% bound",
    "greater than X% bound", "less than X% bound", etc.) are flagged
    "censored" rather than treated as an exact point value; the merge script
    (scripts/v2_fu_merge.py) skips any candidate so flagged. This caught
    cevimeline's "<20% bound" and oxybate's "less than 1% is bound" being
    stored as if they were exact measurements.
"""
import gzip, json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v2_cmax as cmax_mod  # reuse LABELS path, load_labels, clean, LOCAL_ROUTES, single_ingredient

DEFAULT_OUT = Path("/tmp/v2_fu_label_candidates.json")

RANGE_BOUND_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)"
    r"[^.;]{0,40}?\bbound\b", re.IGNORECASE)
SINGLE_BOUND_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)[^.;]{0,40}?\bbound\b", re.IGNORECASE)
ANY_BOUND_PCT_RE = re.compile(r"\d{1,3}(?:\.\d+)?\s*(?:%|percent)[^.;]{0,40}?\bbound\b", re.IGNORECASE)

REJECT_RE = re.compile(
    r"metabolite|respectively|renal impair|hepatic impair|pediatric|geriatric|elderly|"
    r"pregnan|end-stage renal|dialysis patient|neonat|\bchildren\b|desmethyl|"
    r"glucuronide|N-oxide|hydroxy[- ]?(?:derivative|metabolite)|\bMHD\b", re.IGNORECASE)

# "<N>% of <subject> is/are bound" -- subject must name the target drug.
OF_SUBJECT_RE = re.compile(
    r"of\s+([A-Za-z][\w'\-]*(?:\s+[A-Za-z][\w'\-]*){0,3})\s+(?:is|are)\s+bound",
    re.IGNORECASE)

# Inequality / censored phrasing near the %-bound clause.
CENSORED_RE = re.compile(
    r"(?:<|>|≤|≥|\bless than\b|\bgreater than\b|\bmore than\b|\bat least\b|\bup to\b)\s*\d",
    re.IGNORECASE)


def sentences(text):
    return [s for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text) if s]


def _name_matches(inn, lab):
    """Guard against label-ingredient mismatch (e.g. a combination product's
    label text describing a different active ingredient's PPB than the one
    being looked up)."""
    of = lab.get("openfda") or {}
    subs = of.get("substance_name") or []
    gens = of.get("generic_name") or []
    if not subs and not gens:
        return True  # no openfda ingredient metadata to cross-check against
    names = " ".join(subs + gens).lower()
    base = inn.split(" ", 1)[0].lower()
    return base in names


def _subject_matches(inn, sentence):
    """Sentence-level backstop for _name_matches(): if the sentence has the
    "<N>% of <subject> is/are bound" shape, the captured subject must name
    the target drug (or a salt form of it), regardless of what the label's
    openfda metadata says (which may be empty and thus uninformative)."""
    m = OF_SUBJECT_RE.search(sentence)
    if not m:
        return True  # sentence doesn't have this shape; nothing to check
    subject = m.group(1).lower().strip()
    base = inn.split(" ", 1)[0].lower()
    if base in subject or subject in base:
        return True
    return any(tok == base for tok in subject.split())


def extract_for_drug(inn, labels, drug_labels):
    candidates = []
    for m in drug_labels:
        lab = labels.get(m["set_id"])
        if not lab:
            continue
        if not cmax_mod.single_ingredient(lab):
            continue
        if not _name_matches(inn, lab):
            continue
        for sec in ("clinical_pharmacology", "pharmacokinetics", "description"):
            text = cmax_mod.label_text(lab, sec)
            for s in sentences(text):
                if REJECT_RE.search(s):
                    continue
                if "protein" not in s.lower():
                    continue
                if not _subject_matches(inn, s):
                    continue
                # reject sentences with more than one distinct %-bound clause
                # (ambiguous: likely parent + metabolite, or multiple ranges)
                if len(ANY_BOUND_PCT_RE.findall(s)) != 1:
                    continue
                mt = RANGE_BOUND_RE.search(s)
                if mt:
                    lo, hi = float(mt.group(1)), float(mt.group(2))
                    pct = math.sqrt(lo * hi)
                    literal = (mt.group(1), mt.group(2))
                else:
                    mt = SINGLE_BOUND_RE.search(s)
                    if not mt:
                        continue
                    pct = float(mt.group(1))
                    literal = (mt.group(1),)
                if not (0 < pct <= 100):
                    continue
                censored = bool(CENSORED_RE.search(s))
                candidates.append({
                    "set_id": m["set_id"], "section": sec, "snippet": s.strip(),
                    "pct_bound": pct, "literal": literal,
                    "application_number": m.get("application_number"),
                    "censored": censored,
                })
    return candidates


def main(argv):
    out_path = DEFAULT_OUT
    if "--out" in argv:
        out_path = Path(argv[argv.index("--out") + 1])

    drugs = json.loads((ROOT / "src/neurocascade/benchmark/v2/drugs.json").read_text())["drugs"]
    fu_file = json.loads((ROOT / "src/neurocascade/benchmark/v2/inputs/plasma_fu.json").read_text())
    have = {r["inn"] for r in fu_file["drugs"] if r.get("fu")}
    labels = cmax_mod.load_labels()
    out = {}
    n_cand = 0
    for d in drugs:
        if d["inn"] in have:
            continue
        cands = extract_for_drug(d["inn"], labels, d["labels"])
        # drop censored-only candidate lists at the point-value level, but
        # keep them visible in the output (flagged) rather than silently
        # discarding the evidence -- scripts/v2_fu_merge.py skips any
        # candidate with censored=True rather than storing it as fu.
        if cands:
            out[d["inn"]] = cands
            n_cand += 1
    print(f"{n_cand} of {sum(1 for d in drugs if d['inn'] not in have)} missing-fu drugs have a label fu candidate",
          file=sys.stderr)
    out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
