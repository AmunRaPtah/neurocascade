"""P32.4: prospective blind test of ``predict_compound`` on recently disclosed
CNS compounds that are absent from the benchmark and were never used in any fit.

This is the first clean test of the P34.3 frozen candidate (P34 returned no
verdict; the requires-GO P34.5 gate was lifted by the human decision of
2026-10-28, so no GO marker exists and none is written).  Unlike P32.3's held-out
demo, these compounds were in the repository during NO development phase: their
preclinical packages (``benchmark/prospective/packages/*.json``) were built from
preclinical sources only and committed, with a SHA-256 of the prediction file
recorded in the ledger, BEFORE any human-truth record for them existed in git.

What the product actually delivers, and what can be scored:

* **Human clearance** is the one compound-specific number ``predict_compound``
  returns (P29.2 well-stirred IVIVE when the package has human liver CLint and
  human plasma fu, else fixed-exponent rat allometry).  It is scored here
  against the published observed human clearance, with the v2 train-mean CL as
  the mean baseline beside it.  The product predicts *systemic* clearance, so
  the comparison is only apples-to-apples when the observed value is a true IV
  systemic CL (``cl_basis == "iv_systemic"``).  For oral CL/F values (F
  unmeasured) the fold error embeds the F <= 1 assumption and is flagged.
* **Target-engagement dose, panel occupancy-time and effect liabilities** have
  no scored compound-specific predictor (P29.3/P29.4 searched-and-empty; the
  candidate needs a human exposure a preclinical package lacks).  The observed
  clinical dose, PET occupancy and effect fallback are tabulated beside the
  ``pending`` prediction and marked not scored, exactly as P32.3 does.

Every record carries ``PROVENANCE_LABEL`` ("not validated: P34 returned no
verdict").
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .benchmark import conformal
from .compound import (
    PROVENANCE_LABEL,
    CoreInputMissing,
    load_package,
    predict_compound,
)

_ROOT = Path(__file__).resolve().parent / "benchmark" / "prospective"
PACKAGE_DIR = _ROOT / "packages"
OBSERVED_DIR = _ROOT / "observed"
PREDICTIONS_FILE = (Path(__file__).resolve().parents[2] / "results" / "P32.4_predictions.json")

#: Compounds whose preclinical package could NOT be built to a prediction,
#: with the single missing core field.  Listed, never silently dropped
#: (acceptance criterion).  Clearance is the field that forced each refusal.
UNBUILDABLE = {
    "xanomeline": "clearance: no human liver CLint and no rat IV CL in public preclinical "
                  "sources (FDA NDA 216158 tabulates oral rat TK only, called 'unreliable to "
                  "use quantitatively')",
    "emraclidine": "clearance: no human liver CLint and no rat IV CL; the sole primary source "
                   "(Butler 2024, J Med Chem, PMID 38888621) is closed-access and its ChEMBL "
                   "curation holds potency only, no ADME",
    "tavapadon": "clearance: no human liver CLint and no rat IV CL; all disclosed rat data are "
                 "oral toxicokinetics (NCT02687542 protocol / FDA Juvmo review not yet posted)",
    "centanafadine": "clearance: no total human liver CLint (only enzyme-specific MAO-A/CYP "
                     "CLint) and no rat IV CL in FDA NDA 218145 (oral TK only)",
    "deutetrabenazine": "clearance: no human liver CLint in uL/min/mg (FDA NDA 208082 reports "
                        "microsomal t1/2 only) and the only rat IV CL (58.9 mL/min/kg) is for "
                        "tetrabenazine PARENT, not the deuterated drug - rule 2 forbids "
                        "substituting a different drug's measured value",
}

_TRAIN_MEAN_BAND = "human_cl_v2_train_mean"


# ---- the compounds ----------------------------------------------------------
def package_names() -> list[str]:
    return sorted(p.stem for p in PACKAGE_DIR.glob("*.json"))


def load_observed(name: str) -> dict[str, Any] | None:
    path = OBSERVED_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _train_mean_point() -> float:
    return conformal.load_bands()["quantities"][_TRAIN_MEAN_BAND]["frozen_point"]


# ---- predictions (blind; committed before any observed truth) ---------------
def run_predictions() -> dict[str, Any]:
    """Blind ``predict_compound`` output for every buildable package.

    Reads only the packages (no observed file), so the returned structure is
    exactly what was committed before any human truth existed.
    """
    predictions = {}
    for name in package_names():
        predictions[name] = predict_compound(load_package(PACKAGE_DIR / f"{name}.json"))
    return {
        "task": "P32.4",
        "label": PROVENANCE_LABEL,
        "note": "blind predictions from preclinical packages only; committed before any "
                "human-truth record for these compounds existed in git",
        "mean_baseline": {"quantity": "human_cl", "band": _TRAIN_MEAN_BAND,
                          "point": _train_mean_point(), "unit": "mL/min/kg",
                          "method": "P33.3 v2 train arithmetic-mean clearance (the mean baseline "
                                    "placed beside every compound)"},
        "compounds": sorted(predictions),
        "unbuildable": UNBUILDABLE,
        "predictions": predictions,
    }


def write_predictions() -> str:
    """Write the committed prediction file and return its SHA-256 (hex)."""
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(run_predictions(), indent=1, sort_keys=True) + "\n"
    PREDICTIONS_FILE.write_text(blob)
    return hashlib.sha256(blob.encode()).hexdigest()


def predictions_sha256() -> str:
    return hashlib.sha256(PREDICTIONS_FILE.read_bytes()).hexdigest()


# ---- scoring (added with the observed truth, in a later commit) -------------
def _fold(a: float, b: float) -> float:
    return max(a / b, b / a)


def _score_clearance(name: str, prediction: Mapping[str, Any],
                     observed: Mapping[str, Any] | None, mean_point: float) -> dict[str, Any]:
    rec = prediction["human_pk"]["clearance"]
    predicted = rec["point"]
    out = {"predicted_mL_min_kg": predicted,
           "band": rec["provenance"]["band"],
           "predictor": rec["provenance"]["predictor"],
           "compound_specific": rec["provenance"].get("compound_specific"),
           "mean_baseline_mL_min_kg": mean_point}
    obs = (observed or {}).get("clearance") if observed else None
    if not obs or obs.get("cl_mL_min_kg") is None:
        out["scored"] = False
        out["reason"] = (obs or {}).get("note", "no observed human clearance in this checkout "
                                                "(host scores with full data)")
        out["cl_basis"] = (obs or {}).get("cl_basis", "missing")
        return out
    observed_cl = float(obs["cl_mL_min_kg"])
    out.update({
        "scored": True,
        "observed_mL_min_kg": observed_cl,
        "cl_basis": obs["cl_basis"],
        "fold_error": _fold(predicted, observed_cl),
        "mean_baseline_fold_error": _fold(mean_point, observed_cl),
        "beats_mean_baseline": _fold(predicted, observed_cl) < _fold(mean_point, observed_cl),
        "is_systemic": obs["cl_basis"] in ("iv_systemic", "oral_clf_F_bounded"),
        "observed_source": obs["source"],
        "observed_note": obs["note"],
    })
    for lv, band in rec["bands"].items():
        out.setdefault("covered", {})[lv] = bool(band["lower"] <= observed_cl <= band["upper"])
    return out


def _tabulate_pending(prediction: Mapping[str, Any], key: str,
                      observed: Mapping[str, Any] | None, obs_key: str) -> dict[str, Any]:
    pred = prediction.get(key) if key in prediction else prediction["human_pk"].get(key)
    obs = (observed or {}).get(obs_key)
    return {"predicted_status": pred.get("status"), "predicted_reason": pred.get("reason"),
            "observed": obs, "scored": False,
            "reason": "no compound-specific predictor (pending); observed shown for context"}


def score() -> dict[str, Any]:
    """Join blind predictions with observed truth; fold errors + mean baseline.

    ``n_scored`` counts compounds with a published observed human clearance
    compared against the prediction.  ``n_scored_systemic`` counts the subset
    whose observed value is a true systemic clearance (``iv_systemic`` or an
    F-bounded oral CL/F), the only apples-to-apples comparisons for a model
    that predicts systemic CL.
    """
    mean_point = _train_mean_point()
    rows, scored, systemic = [], 0, 0
    sum_log_fold = sum_log_base = 0.0
    for name in package_names():
        prediction = predict_compound(load_package(PACKAGE_DIR / f"{name}.json"))
        observed = load_observed(name)
        cl = _score_clearance(name, prediction, observed, mean_point)
        if cl["scored"]:
            scored += 1
            sum_log_fold += math.log(cl["fold_error"])
            sum_log_base += math.log(cl["mean_baseline_fold_error"])
            if cl["is_systemic"]:
                systemic += 1
        rows.append({
            "compound": name,
            "disclosed": (observed or {}).get("disclosed"),
            "clearance": cl,
            "dose": _tabulate_pending(prediction, "target_engagement_dose", observed, "dose_mg_per_day"),
            "occupancy": _tabulate_pending(prediction, "panel_occupancy_time", observed, "occupancy"),
            "effects": {"predicted_fallback": {e: {"status": v.get("status"),
                        "point_kind": v.get("point_kind"), "bands": v.get("bands")}
                        for e, v in prediction["liabilities"].items()},
                        "observed": None, "scored": False,
                        "reason": "liabilities fall back to train prevalence without a human "
                                  "exposure (uninformative {0,1} sets); no effect-label truth "
                                  "sourced for these out-of-benchmark compounds"},
            "label": PROVENANCE_LABEL,
        })
    scored_rows = [r for r in rows if r["clearance"]["scored"]]
    gmfe = math.exp(sum_log_fold / scored) if scored else None
    gmfe_base = math.exp(sum_log_base / scored) if scored else None
    return {
        "task": "P32.4",
        "label": PROVENANCE_LABEL,
        "prediction_file": str(PREDICTIONS_FILE.relative_to(PREDICTIONS_FILE.parents[1])),
        "prediction_sha256": predictions_sha256() if PREDICTIONS_FILE.exists() else None,
        "mean_baseline": {"band": _TRAIN_MEAN_BAND, "point_mL_min_kg": mean_point,
                          "method": "P33.3 v2 train arithmetic-mean clearance"},
        "n_compounds_built": len(rows),
        "n_scored": scored,
        "n_scored_systemic": systemic,
        "n_scored_definition": "compounds with a published observed human clearance scored "
                               "against the blind prediction; n_scored_systemic is the subset "
                               "whose observed value is a true systemic CL (iv_systemic or "
                               "F-bounded oral CL/F)",
        "clearance_gmfe": gmfe,
        "mean_baseline_gmfe": gmfe_base,
        "beats_mean_baseline": (gmfe is not None and gmfe < gmfe_base),
        "unbuildable": UNBUILDABLE,
        "systemic_compounds": sorted(r["compound"] for r in scored_rows
                                     if r["clearance"]["is_systemic"]),
        "oral_clf_compounds": sorted(r["compound"] for r in scored_rows
                                     if not r["clearance"]["is_systemic"]),
        "rows": rows,
    }


# ---- rendering --------------------------------------------------------------
def _fmt(x: Any) -> str:
    return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)


def render_markdown(result: Mapping[str, Any]) -> str:
    mb = result["mean_baseline"]
    lines = [
        f"# P32.4 prospective blind test ({PROVENANCE_LABEL})", "",
        f"Prediction file `{result['prediction_file']}` SHA-256 `{result['prediction_sha256']}`, "
        "committed before any human-truth record for these compounds existed in git.", "",
        f"Mean baseline: v2 train-mean clearance {mb['point_mL_min_kg']:.3g} mL/min/kg "
        f"({mb['method']}).", "",
        f"Scored: {result['n_scored']} of {result['n_compounds_built']} built compounds have a "
        f"published observed human clearance; {result['n_scored_systemic']} are true-systemic "
        f"comparisons. Clearance GMFE {_fmt(result['clearance_gmfe'])} vs mean-baseline GMFE "
        f"{_fmt(result['mean_baseline_gmfe'])}.", "",
        "| compound | predicted CL | observed CL | basis | fold-error | mean-baseline fold | "
        "dose (obs) | occupancy (obs) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        cl = r["clearance"]
        obs = f"{_fmt(cl.get('observed_mL_min_kg'))}" if cl["scored"] else "none"
        fe = _fmt(cl["fold_error"]) if cl["scored"] else "not scored"
        mb_fe = _fmt(cl["mean_baseline_fold_error"]) if cl["scored"] else "-"
        dose = (r["dose"]["observed"] or {}).get("value", "-")
        occ = (r["occupancy"]["observed"] or {})
        occ_s = (f"{occ.get('target')} {occ.get('occupancy_percent')}%"
                 if occ.get("status") == "observed" else occ.get("status", "-"))
        lines.append(f"| {r['compound']} | {_fmt(cl['predicted_mL_min_kg'])} ({cl['predictor'].split(';')[0]}) "
                     f"| {obs} | {cl.get('cl_basis')} | {fe} | {mb_fe} | {dose} | {occ_s} |")
    lines += ["", "Compounds whose package could not be built (missing core field):", ""]
    for name, why in result["unbuildable"].items():
        lines.append(f"- **{name}** - {why}")
    lines += ["", f"Dose and occupancy are `pending` from the product (no scored compound-specific "
                  f"predictor) and shown for context only; effects fall back to uninformative "
                  f"train-prevalence sets. {PROVENANCE_LABEL}."]
    return "\n".join(lines) + "\n"


__all__ = [
    "OBSERVED_DIR",
    "PACKAGE_DIR",
    "PREDICTIONS_FILE",
    "PROVENANCE_LABEL",
    "UNBUILDABLE",
    "load_observed",
    "package_names",
    "predictions_sha256",
    "render_markdown",
    "run_predictions",
    "score",
    "write_predictions",
]
