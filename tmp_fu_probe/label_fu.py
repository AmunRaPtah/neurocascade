"""Deterministic, conservative extraction of human plasma fu from openFDA
label text already cached by scripts/v2_cmax.py fetch
($NC_V2_CACHE/cmax/labels.jsonl.gz). Looks for sentences of the form
"X% bound to (human) plasma protein(s)" (or "plasma protein binding is X%",
"X% of <drug> is bound to plasma proteins") in clinical_pharmacology /
pharmacokinetics / description sections, converts %bound -> fu = 1 - %/100.
Ranges -> geometric midpoint of the two endpoints. Rejects any sentence
mentioning a metabolite or a special population (renal/hepatic impairment,
pediatric, geriatric, pregnancy, elderly) since those are not the drug's own
normal-adult plasma protein binding.
"""
import gzip, json, math, re, sys
from pathlib import Path

ROOT = Path("/root/projects/neurocascade/.claude/worktrees/agent-aa546ac89132d41e2")
sys.path.insert(0, str(ROOT / "scripts"))
import v2_cmax as cmax_mod  # reuse LABELS path, load_labels, clean, LOCAL_ROUTES, single_ingredient

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
                candidates.append({
                    "set_id": m["set_id"], "section": sec, "snippet": s.strip(),
                    "pct_bound": pct, "literal": literal,
                    "application_number": m.get("application_number"),
                })
    return candidates


def main():
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
        if cands:
            out[d["inn"]] = cands
            n_cand += 1
    print(f"{n_cand} of {sum(1 for d in drugs if d['inn'] not in have)} missing-fu drugs have a label fu candidate",
          file=sys.stderr)
    (ROOT / "tmp_fu_probe" / "label_fu_candidates.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
