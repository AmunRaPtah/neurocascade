"""P32.3: blind demo of ``predict_compound`` on P28.1 held-out benchmark drugs.

Each drug is run from its preclinical package only (``package_from_benchmark``:
binding panel and ADME record, no truth, no human exposure), and every
returned quantity is put beside what was observed in humans:

* **Target-engagement dose and panel occupancy.** ``predict_compound`` returns
  them as ``pending``, because P29.3 and P29.4 are searched-and-empty. The
  actual label dose range and PET occupancy come from the drug's P28.4 truth
  record. They are shown but cannot be scored, because there is no prediction.
* **Human CL and Vss.** Compared with P28.4 IV truth, the truth P32.1 scored CL
  against. When the P28.4 record is absent, which is the case for every
  held-out drug in a session checkout, the v2 human PK truth for the same drug
  is used, and ``truth_source`` says which source was used.
* **Sedation and anticholinergic liabilities.** These are scored against the v2
  label, which is the label rule the P30.4 bands were calibrated on. The P28.4
  label uses a different rule (any listed frequency), so it is shown but not
  scored.

A comparison is ``in_sample`` when the drug's own truth went into the point
it is compared to: a v2 train drug's CL or Vss inside the train mean, or a
P30.4 calibration drug under the train-prevalence liability. In-sample
comparisons are reported, but they do not count toward ``n_scored``.

Everything is retrospective. The P28.1 held-out drugs were in the repository
during P19-P25 development, and diazepam and ziprasidone have human values
that were read or fitted there. P32.4 is the clean test. Every row and every
number carries ``PROVENANCE_LABEL``.
"""

from __future__ import annotations

import html
import math
from typing import Any, Iterable, Mapping

from .benchmark import load_split, load_truth_record
from .benchmark import human_pk_translation as _pk
from .benchmark.v2 import load_v2_human_cmax, load_v2_human_pk, load_v2_liabilities, v2_partition
from .benchmark.v2 import candidate_bands
from .compound import (
    PROVENANCE_LABEL,
    CoreInputMissing,
    package_from_benchmark,
    predict_compound,
)

N_DEMO = 5
LEVELS = ("0.80", "0.95")
SCORED_LIABILITIES = candidate_bands.BANDED
ALL_LIABILITIES = candidate_bands.BANDED + candidate_bands.UNDERPOWERED
#: P28.4 truth names orthostatic hypotension "orthostasis".
_P28_4_EFFECT_KEY = {"orthostatic_hypotension": "orthostasis"}

#: Disclosed in P28.1 (docs/architecture.md, "Disclosed after the draw"):
#: held-out drugs whose values were already read or fitted in Phases 19-25.
NOT_BLIND = {
    "amisulpride": "human PET/PK/label values read or fitted in Phases 19-25",
    "quetiapine": "human PET/PK/label values read or fitted in Phases 19-25",
    "ziprasidone": "human PET/PK/label values read or fitted in Phases 19-25",
    "diazepam": "human PET/PK/label values read or fitted in Phases 19-25",
    "alprazolam": "rodent elevated-plus-maze series only (retired mouse calibration); "
                  "no human value was used",
}

#: The P28.1 ledger rule: every held-out score is also reported without these.
HUMAN_TOUCHED = ("amisulpride", "diazepam", "quetiapine", "ziprasidone")

SELECTION_RULE = (
    "the first five, in alphabetical order, of the P28.1 held-out drugs that "
    "predict_compound does not refuse from their preclinical package. The rule "
    "reads no truth, so held-out outcomes cannot steer which drugs are shown"
)


# ---- selection --------------------------------------------------------------
def runnable_heldout() -> list[str]:
    """P28.1 held-out drugs that ``predict_compound`` does not refuse, sorted."""
    out = []
    for inn in sorted(load_split()["heldout"]):
        try:
            predict_compound(package_from_benchmark(inn))
        except CoreInputMissing:
            continue
        out.append(inn)
    return out


def demo_drugs() -> list[str]:
    """The five demo drugs given by ``SELECTION_RULE``."""
    return runnable_heldout()[:N_DEMO]


# ---- truth ------------------------------------------------------------------
def _p28_4(inn: str) -> dict[str, Any] | None:
    try:
        return load_truth_record(inn)
    except KeyError:
        return None


def _actual_dose(record: Mapping[str, Any] | None, v2_cmax_row: Mapping[str, Any] | None
                 ) -> dict[str, Any]:
    if record is None:
        basis = ((v2_cmax_row or {}).get("cmax") or {}).get("usual_dose_basis")
        if basis:
            src = (v2_cmax_row["cmax"].get("source") or {})
            return {"status": "observed", "source": "v2 human Cmax truth, label usual-dose text "
                    f"(FDA SPL set_id {src.get('set_id')}); P28.4 record absent",
                    "label_text": basis}
        return {"status": "host-pending", "reason": "P28.4 truth record absent from this checkout"}
    rows = [{"indication": r.get("indication"), "low": r.get("low"), "high": r.get("high"),
             "units": r.get("units"), "route": r.get("route")}
            for r in (record.get("dose") or {}).get("label") or []]
    if not rows:
        return {"status": "searched-and-empty", "reason": "P28.4 record holds no label dose"}
    return {"status": "observed", "source": "P28.4 label dose (FDA SPL)", "label_doses": rows}


def _actual_occupancy(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"status": "host-pending", "reason": "P28.4 truth record absent from this checkout"}
    rows = [{k: r.get(k) for k in ("target", "region", "dose_mg_per_day", "occupancy_percent",
                                    "statistic", "first_author", "year", "pmid")}
            for r in record.get("pet") or []]
    if not rows:
        return {"status": "searched-and-empty", "reason": "P28.4 PET search found no human "
                                                          "occupancy with a dose for this drug"}
    return {"status": "observed", "source": "P28.4 human PET", "pet": rows}


def _actual_pk(inn: str, parameter: str, record: Mapping[str, Any] | None,
               v2_pk: Mapping[str, Any]) -> dict[str, Any] | None:
    if record is not None:
        truth = _pk._human_truth(inn, parameter)
        if truth is not None:
            return {"value": truth[0], "unit": "mL/min/kg" if parameter == "cl" else "L/kg",
                    "truth_source": "P28.4 IV compilation", "refs": list(truth[1])}
    field = "clearance_iv" if parameter == "cl" else "vss"
    blob = (v2_pk.get(inn) or {}).get(field) or {}
    if blob.get("value") is None:
        return None
    return {"value": float(blob["value"]), "unit": "mL/min/kg" if parameter == "cl" else "L/kg",
            "truth_source": "v2 human PK truth (P28.4 record absent)",
            "refs": sorted({s.get("pmid") or s.get("chembl_document_id")
                            for s in blob.get("sources") or []} - {None})}


def _v2_has(v2_pk: Mapping[str, Any], inn: str, field: str) -> bool:
    return ((v2_pk.get(inn) or {}).get(field) or {}).get("value") is not None


# ---- one drug -----------------------------------------------------------------
def _pk_comparison(rec: Mapping[str, Any], actual: Mapping[str, Any] | None,
                   in_sample: bool) -> dict[str, Any]:
    if actual is None:
        return {"scored": False, "reason": "no human truth in this checkout"}
    fold = max(rec["point"] / actual["value"], actual["value"] / rec["point"])
    return {"scored": True, "in_sample": in_sample, "fold_error": fold,
            "covered": {lv: bool(rec["bands"][lv]["lower"] <= actual["value"]
                                 <= rec["bands"][lv]["upper"]) for lv in LEVELS}}


def _liability_entry(effect: str, rec: Mapping[str, Any], inn: str,
                     record: Mapping[str, Any] | None, v2_labels: Mapping[str, Any],
                     calibration: Mapping[str, list[str]]) -> dict[str, Any]:
    v2_label = ((v2_labels.get(inn) or {}).get(effect) or {}).get("label")
    p28_label = None
    if record is not None:
        labels = (record.get("effects") or {}).get("label") or {}
        p28_label = labels.get(_P28_4_EFFECT_KEY.get(effect, effect))
    actual = {"v2_label": v2_label, "p28_4_label": p28_label,
              "scored_against": "v2 label (the rule P30.4 was calibrated on)",
              "p28_4_note": "P28.4 label uses a different rule (any listed frequency); shown, "
                            "not scored"}
    if rec.get("status") != "predicted":
        comparison = {"scored": False, "reason": f"no prediction ({rec.get('status')}): "
                                                 f"{rec.get('reason')}"}
    elif v2_label is None:
        comparison = {"scored": False, "reason": "no v2 label for this drug in this checkout"}
    else:
        in_sample = (not rec["provenance"].get("compound_specific")
                     and inn in calibration.get(effect, []))
        sets = {lv: rec["bands"][lv]["labels"] for lv in LEVELS}
        comparison = {"scored": True, "in_sample": in_sample,
                      "covered": {lv: int(v2_label) in sets[lv] for lv in LEVELS},
                      "set_size": {lv: len(sets[lv]) for lv in LEVELS},
                      "informative": {lv: len(sets[lv]) == 1 for lv in LEVELS}}
    return {"predicted": rec, "actual": actual, "comparison": comparison,
            "label": PROVENANCE_LABEL}


def demo_row(inn: str, *, v2_pk: Mapping[str, Any] | None = None,
             v2_labels: Mapping[str, Any] | None = None,
             v2_cmax: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Predicted (blind) vs actual for one held-out drug."""
    v2_cmax = load_v2_human_cmax() if v2_cmax is None else v2_cmax
    v2_pk = load_v2_human_pk() if v2_pk is None else v2_pk
    v2_labels = load_v2_liabilities() if v2_labels is None else v2_labels
    v2_train = set(v2_partition()[0])
    bands = candidate_bands.load_bands()["quantities"]
    calibration = {e: list(bands[f"liability:{e}"].get("calibration_drugs") or [])
                   for e in SCORED_LIABILITIES}
    package = package_from_benchmark(inn)
    result = predict_compound(package)
    record = _p28_4(inn)

    quantities: dict[str, Any] = {
        "target_engagement_dose": {
            "predicted": result["target_engagement_dose"],
            "actual": _actual_dose(record, v2_cmax.get(inn)),
            "comparison": {"scored": False, "reason": "no prediction: "
                           + result["target_engagement_dose"]["reason"]},
            "label": PROVENANCE_LABEL},
        "occupancy": {
            "predicted": result["panel_occupancy_time"],
            "actual": _actual_occupancy(record),
            "comparison": {"scored": False, "reason": "no prediction: "
                           + result["panel_occupancy_time"]["reason"]},
            "label": PROVENANCE_LABEL},
    }
    for key, parameter, field in (("clearance", "cl", "clearance_iv"), ("vss", "vss", "vss")):
        rec = result["human_pk"][key]
        actual = _actual_pk(inn, parameter, record, v2_pk)
        # The train-mean point averages v2 train truth; IVIVE fits nothing per drug.
        in_sample = (not rec["provenance"].get("compound_specific", True)
                     and inn in v2_train and _v2_has(v2_pk, inn, field))
        quantities[key] = {"predicted": rec, "actual": actual,
                           "comparison": _pk_comparison(rec, actual, in_sample),
                           "label": PROVENANCE_LABEL}
    for effect in ALL_LIABILITIES:
        quantities[f"liability:{effect}"] = _liability_entry(
            effect, result["liabilities"][effect], inn, record, v2_labels, calibration)
    return {
        "inn": inn,
        "p28_1_split": "heldout",
        "v2_partition": ("train" if inn in v2_train
                         else "heldout" if inn in set(v2_partition()[1]) else "absent"),
        "not_blind": NOT_BLIND.get(inn),
        "p28_4_truth_present": record is not None,
        "inputs": "package_from_benchmark: binding panel + ADME only; no human exposure",
        "quantities": quantities,
        "label": PROVENANCE_LABEL,
    }


# ---- summary ----------------------------------------------------------------
def _gmfe(xs: list[float]) -> float | None:
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else None


def summarise(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Out-of-sample accuracy per quantity, plus what is host-pending."""
    rows = list(rows)
    per: dict[str, Any] = {}
    scored_drugs: set[str] = set()
    specific_drugs: set[str] = set()
    for row in rows:
        for name, q in row["quantities"].items():
            c = q["comparison"]
            entry = per.setdefault(name, {"n": 0, "n_in_sample": 0, "drugs": [],
                                          "covered": {lv: 0 for lv in LEVELS},
                                          "fold_errors": [], "n_informative": {lv: 0 for lv in LEVELS},
                                          "not_scored": {}})
            if not c["scored"]:
                entry["not_scored"][row["inn"]] = c["reason"]
                continue
            if c["in_sample"]:
                entry["n_in_sample"] += 1
                continue
            entry["n"] += 1
            entry["drugs"].append(row["inn"])
            scored_drugs.add(row["inn"])
            if q["predicted"].get("provenance", {}).get("compound_specific"):
                specific_drugs.add(row["inn"])
            for lv in LEVELS:
                entry["covered"][lv] += int(c["covered"][lv])
                if "informative" in c:
                    entry["n_informative"][lv] += int(c["informative"][lv])
            if "fold_error" in c:
                entry["fold_errors"].append(c["fold_error"])
    for entry in per.values():
        entry["gmfe"] = _gmfe(entry.pop("fold_errors"))
    host_pending = sorted(r["inn"] for r in rows if not r["p28_4_truth_present"])
    return {
        "n_drugs": len(rows),
        "n_scored": len(scored_drugs),
        "n_scored_definition": "drugs with at least one out-of-sample predicted-vs-actual "
                               "comparison (in-sample train-mean/prevalence comparisons excluded)",
        "scored_drugs": sorted(scored_drugs),
        "n_scored_compound_specific": len(specific_drugs),
        "compound_specific_drugs": sorted(specific_drugs),
        "per_quantity": per,
        "host_pending": host_pending,
        "host_pending_note": ("P28.4 truth (label dose, PET occupancy, IV PK) is absent from "
                              "this checkout for these drugs; the host scores them with the "
                              "same function" if host_pending else None),
    }


def run_blind_demo(drugs: list[str] | None = None) -> dict[str, Any]:
    """Demo table (five drugs by ``SELECTION_RULE``) plus all runnable held-out drugs."""
    v2_pk, v2_labels, v2_cmax = load_v2_human_pk(), load_v2_liabilities(), load_v2_human_cmax()
    runnable = runnable_heldout()
    demo = list(drugs) if drugs is not None else runnable[:N_DEMO]
    rows = {inn: demo_row(inn, v2_pk=v2_pk, v2_labels=v2_labels, v2_cmax=v2_cmax)
            for inn in dict.fromkeys(demo + runnable)}
    demo_summary = summarise(rows[i] for i in demo)
    all_summary = summarise(rows[i] for i in runnable)
    return {
        "task": "P32.3",
        "label": PROVENANCE_LABEL,
        "retrospective": True,
        "retrospective_note": "the P28.1 held-out drugs were present during P19-P25 "
                              "development; P32.4 is the clean test",
        "selection_rule": SELECTION_RULE,
        "demo_drugs": demo,
        "runnable_heldout": runnable,
        "refused_heldout": sorted(set(load_split()["heldout"]) - set(runnable)),
        "n_scored": all_summary["n_scored"],
        "host_pending": all_summary["host_pending"],
        "demo": {"rows": [rows[i] for i in demo], "summary": demo_summary},
        "all_runnable": {"summary": all_summary},
        "untouched_runnable": {
            "excluded": [i for i in runnable if i in HUMAN_TOUCHED],
            "summary": summarise(rows[i] for i in runnable if i not in HUMAN_TOUCHED)},
    }


# ---- rendering ----------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.3g}"


def _band_text(band: Mapping[str, Any]) -> str:
    if "labels" in band:
        return "{" + ",".join(str(v) for v in band["labels"]) + "}"
    if not band.get("bounded") or not math.isfinite(band["upper"]):
        return "unbounded"
    return f"{_fmt(band['lower'])}-{_fmt(band['upper'])}"


def cell_texts(row: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    """(quantity, predicted, actual, comparison) strings for one demo row."""
    out = []
    for name, q in row["quantities"].items():
        p, a, c = q["predicted"], q["actual"], q["comparison"]
        if p.get("status") != "predicted":
            pred = f"{p.get('status')}"
        else:
            pred = (f"{_fmt(p['point'])} (80% {_band_text(p['bands']['0.80'])}; "
                    f"95% {_band_text(p['bands']['0.95'])})")
        if name == "target_engagement_dose":
            if "label_doses" in a:
                act = ", ".join(f"{d['low']}-{d['high']} {d['units']} ({d['indication']})"
                                for d in a["label_doses"])
            else:
                act = f"label: {a['label_text']}" if "label_text" in a else a["status"]
        elif name == "occupancy":
            act = (", ".join(f"{r['target']} {r['occupancy_percent']}% at {r['dose_mg_per_day']} mg/d"
                             for r in a["pet"]) if a["status"] == "observed" else a["status"])
        elif name.startswith("liability:"):
            act = (f"v2 {a['v2_label'] if a['v2_label'] is not None else 'n/a'}; "
                   f"P28.4 {a['p28_4_label'] if a['p28_4_label'] is not None else 'n/a'}")
        else:
            act = (f"{_fmt(a['value'])} {a['unit']} ({a['truth_source']})" if a else "no truth")
        if not c["scored"]:
            comp = "not scored"
        else:
            parts = [f"{'covered' if c['covered'][lv] else 'missed'} at {lv}" for lv in LEVELS]
            if "fold_error" in c:
                parts.insert(0, f"fold {_fmt(c['fold_error'])}")
            if "informative" in c and not any(c["informative"].values()):
                parts.append("set holds both labels (uninformative)")
            if c["in_sample"]:
                parts.append("in-sample")
            comp = "; ".join(parts)
        out.append((name, pred, act, comp))
    return out


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [f"# P32.3 blind demo: predict_compound on held-out drugs ({PROVENANCE_LABEL})", "",
             f"Selection: {result['selection_rule']}.", "",
             f"Retrospective: {result['retrospective_note']}.", "",
             f"| drug | quantity | predicted (blind) | actual | comparison | status |",
             "|---|---|---|---|---|---|"]
    for row in result["demo"]["rows"]:
        drug = row["inn"] + (" (not blind)" if row["not_blind"] else "")
        for name, pred, act, comp in cell_texts(row):
            cells = (drug, name, pred, act, comp, PROVENANCE_LABEL)
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    s = result["all_runnable"]["summary"]
    lines += ["", f"All {s['n_drugs']} runnable held-out drugs: {s['n_scored']} scored out of "
                  f"sample ({', '.join(s['scored_drugs']) or 'none'}); host-pending P28.4 "
                  f"truth: {', '.join(s['host_pending']) or 'none'}. {PROVENANCE_LABEL}.", ""]
    return "\n".join(lines)


def render_html(result: Mapping[str, Any]) -> str:
    """Demo table in the report's conventions, so ``report.audit_report`` applies.

    Every cell with a number is a link to the drug's source entry, and every
    row carries ``PROVENANCE_LABEL``.
    """
    esc = lambda t: html.escape(str(t), quote=True)  # noqa: E731
    nv = f'<td><span class="nv">{esc(PROVENANCE_LABEL)}</span></td>'
    body, prov = [], []
    for row in result["demo"]["rows"]:
        target = f"prov-demo-{row['inn']}"
        drug = f'<span class="ident">{esc(row["inn"])}</span>'
        for name, pred, act, comp in cell_texts(row):
            cells = "".join(f'<td><a class="num" href="#{target}">{esc(t)}</a></td>'
                            for t in (pred, act, comp))
            body.append(f"<tr><td>{drug}</td><td>{esc(name)}</td>{cells}{nv}</tr>")
        prov.append(f'<li id="{target}"><b>{esc(row["inn"])}</b>: predicted blind from '
                    f'{esc(row["inputs"])}; actual from P28.4 truth (present: '
                    f'{esc(row["p28_4_truth_present"])}) and v2 truth; not blind: '
                    f'{esc(row["not_blind"] or "no")}. '
                    f'<span class="nv">{esc(PROVENANCE_LABEL)}</span></li>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>neurocascade P32.3 blind demo</title>
<style>body{{font:13px system-ui,sans-serif;max-width:1200px;margin:auto}}
td,th{{border-bottom:1px solid #ddd;padding:3px 5px;text-align:left;vertical-align:top}}
table{{border-collapse:collapse}} .nv{{color:#a40000;font-weight:600}}
a.num{{color:#0645ad;text-decoration:none}}</style></head><body>
<h1>Blind demo on held-out benchmark drugs</h1>
<p><span class="nv">{esc(PROVENANCE_LABEL)}</span>. Selection: {esc(result['selection_rule'])}.
Retrospective: {esc(result['retrospective_note'])}.</p>
<section id="results"><table class="prediction">
<tr><th>drug</th><th>quantity</th><th>predicted (blind)</th><th>actual</th><th>comparison</th>
<th>status</th></tr>{''.join(body)}</table></section>
<section id="provenance"><ol>{''.join(prov)}</ol></section></body></html>
"""


__all__ = [
    "HUMAN_TOUCHED",
    "N_DEMO",
    "NOT_BLIND",
    "SELECTION_RULE",
    "cell_texts",
    "demo_drugs",
    "demo_row",
    "render_html",
    "render_markdown",
    "run_blind_demo",
    "runnable_heldout",
    "summarise",
]
