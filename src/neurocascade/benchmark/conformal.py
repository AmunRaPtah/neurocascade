"""P30.3: distribution-free uncertainty bands for every human prediction.

Every band here is split-conformal (Vovk, Gammerman & Shafer 2005; the
finite-sample quantile rule as stated by Lei et al. 2018, J Am Stat Assoc
113:1094-1111): a nonconformity score is computed for each calibration drug
from a prediction that drug did not train, and the band at level ``1 - a`` is
the ``ceil((n + 1)(1 - a))``-th smallest score.  When that rank exceeds ``n``
the band is unbounded; with n=15 that is what happens at 95%, and the band
file says so instead of shrinking the level.

Calibration uses TRAIN drugs only, and only out-of-sample predictions:

* ``human_cl_ivive`` - P29.2's well-stirred IVIVE / fixed-exponent allometry,
  which fits nothing per drug, so its train predictions are already
  leave-one-drug-out.  Score: ``|log10(predicted / observed)|``; the band is
  multiplicative, ``point / 10**q`` to ``point * 10**q``.
* ``human_cl_v2_train_mean`` / ``human_vss_v2_train_mean`` - P33.3's
  train-arithmetic-mean fallback over the v2 universe, scored leave-one-out
  exactly as P33.3 scores it.  It is the only CL/Vss a compound without a
  preclinical ADME package can be given, and the only Vss predictor that has
  any scored row.
* ``effect_probability:<effect>`` - P30.1's L2 logistic, leave-one-drug-out
  probabilities.  Score: ``1 - p(observed label)``; the band is a conformal
  label set, ``{y : 1 - p(y) <= q}``, i.e. ``{1}``, ``{0}``, ``{0, 1}`` or
  empty.

Quantities without a scored upstream prediction (P29.2 Vss, n=0; P29.3
target-engagement dose and P29.4 occupancy, both searched-and-empty) carry no
band and ``predict_with_bands`` refuses them, so no caller can attach an
interval to an unscored estimate.

Empirical LODO coverage is a nested jackknife: each calibration drug's score
is compared against the quantile of the *other* drugs' scores.  Held-out
coverage (``heldout_coverage``) reads the frozen ``conformal_bands.json``
and scores whichever held-out truth the checkout holds; a session checkout
holds none and gets ``searched-and-empty`` with every held-out drug named.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from . import effects as _effects
from . import human_pk_translation as _pk
from . import load_split, load_truth_record
from .v2 import load_v2_human_pk, v2_partition


LEVELS = (0.80, 0.95)
BANDS_PATH = Path(__file__).resolve().with_name("conformal_bands.json")

#: Quantities the product names but cannot band, with the reason.  A pending
#: quantity is never given an interval and never given a point estimate here.
UNBANDED = {
    "human_vss_ivive": (
        "searched-and-empty",
        "P29.2's Oie-Tozer Vss arm has n=0 scoreable train drugs: no benchmark "
        "drug pairs a clean rat plasma Vss with human Vss truth",
    ),
    "target_engagement_dose": (
        "pending",
        "P29.3 is searched-and-empty (n=0); it waits on P29.5, which is blocked on data",
    ),
    "occupancy": (
        "pending",
        "P29.4 is searched-and-empty (n=0): no held-out drug has the oral exposure, "
        "Kp,uu and P29.2-compatible clearance the occupancy-time chain needs",
    ),
}


class UnbandedQuantityError(ValueError):
    """Raised when a band is requested for a quantity that has no calibration."""


# ---- the conformal arithmetic ---------------------------------------------
def _key(level: float) -> str:
    return f"{level:.2f}"


def conformal_quantile(scores: Iterable[float], level: float) -> float | None:
    """Split-conformal quantile; ``None`` means the band is unbounded."""
    ordered = sorted(float(s) for s in scores)
    n = len(ordered)
    # The 1e-9 keeps e.g. 20 * 0.95 == 19.000000000000004 at rank 19.
    rank = math.ceil((n + 1) * level - 1e-9)
    if n == 0 or rank > n:
        return None
    return ordered[rank - 1]


def _covers(score: float, q: float | None) -> bool:
    return q is None or score <= q


def coverage(scores: Iterable[float], q: float | None) -> dict[str, Any]:
    """Fraction of ``scores`` inside a frozen band ``q``."""
    scores = list(scores)
    covered = sum(_covers(s, q) for s in scores)
    return {"covered": covered, "n": len(scores),
            "rate": covered / len(scores) if scores else None}


def lodo_coverage(scores: Iterable[float], level: float) -> dict[str, Any]:
    """Nested jackknife: each score against the quantile of all the others."""
    scores = list(scores)
    covered = 0
    unbounded = 0
    for i, score in enumerate(scores):
        q = conformal_quantile(scores[:i] + scores[i + 1:], level)
        unbounded += q is None
        covered += _covers(score, q)
    return {"covered": covered, "n": len(scores),
            "rate": covered / len(scores) if scores else None,
            "n_unbounded": unbounded}


def label_set(probability: float, q: float | None) -> tuple[int, ...]:
    """Conformal label set for a binary probability ``p(label == 1)``."""
    return tuple(y for y, p in ((0, 1.0 - probability), (1, probability))
                 if _covers(1.0 - p, q))


def _fold_score(predicted: float, observed: float) -> float:
    if predicted <= 0 or observed <= 0:
        raise ValueError("fold-error bands need strictly positive values")
    return abs(math.log10(predicted / observed))


# ---- calibration rows (train only) ----------------------------------------
def _ivive_cl_rows() -> list[tuple[str, float, float]]:
    metric = _pk.score_train().get("cl")
    if metric is None:
        return []
    return [(p.drug, p.predicted, p.observed) for p in metric.predictions]


def _v2_value(row: Mapping[str, Any] | None, field: str) -> float | None:
    item = (row or {}).get(field)
    if not isinstance(item, Mapping) or item.get("value") is None:
        return None
    return float(item["value"])


def _v2_train_values(field: str) -> dict[str, float]:
    train, _ = v2_partition()
    # The frozen conformal_bands.json was calibrated on human_pk.json as
    # published, before the 2026-10-26 provenance corrections. Calibration
    # reads that same input so the frozen bands stay exactly reproducible;
    # recalibrating on corrected truth is a separate, deliberate re-freeze.
    truth = load_v2_human_pk(corrected=False)
    values = {inn: _v2_value(truth.get(inn), field) for inn in train}
    return {inn: v for inn, v in sorted(values.items()) if v is not None}


def _v2_mean_rows(field: str) -> list[tuple[str, float, float]]:
    values = _v2_train_values(field)
    total = sum(values.values())
    n = len(values)
    return [(inn, (total - v) / (n - 1), v) for inn, v in values.items()] if n > 1 else []


def _fold_quantity(rows, *, source_task, method, unit, frozen_point=None) -> dict[str, Any]:
    scores = [_fold_score(p, o) for _, p, o in rows]
    entry: dict[str, Any] = {
        "status": "calibrated" if scores else "searched-and-empty",
        "kind": "fold",
        "source_task": source_task,
        "method": method,
        "unit": unit,
        "nonconformity": "|log10(predicted / observed)|",
        "n_calibration": len(scores),
        "calibration_drugs": [inn for inn, _, _ in rows],
        "quantiles": {_key(a): conformal_quantile(scores, a) for a in LEVELS},
        "lodo_coverage": {_key(a): lodo_coverage(scores, a) for a in LEVELS},
    }
    entry["fold_bands"] = {
        k: (None if q is None else 10 ** q) for k, q in entry["quantiles"].items()
    }
    if frozen_point is not None:
        entry["frozen_point"] = frozen_point
    return entry


def _effect_quantities() -> dict[str, dict[str, Any]]:
    out = {}
    for effect, result in _effects.leave_one_drug_out().items():
        rows = result["predictions"]
        scores = [1.0 - (r["predicted_probability"] if r["observed"] == 1
                         else 1.0 - r["predicted_probability"]) for r in rows]
        set_sizes = {}
        for a in LEVELS:
            sizes = []
            for i, r in enumerate(rows):
                q = conformal_quantile(scores[:i] + scores[i + 1:], a)
                sizes.append(len(label_set(r["predicted_probability"], q)))
            set_sizes[_key(a)] = {str(k): sizes.count(k) for k in (0, 1, 2)}
        out[f"effect_probability:{effect}"] = {
            "status": "calibrated" if scores else "searched-and-empty",
            "kind": "label_set",
            "source_task": "P30.1",
            "method": "L2 logistic on the occupancy index, leave-one-drug-out over train",
            "unit": "probability of a positive clinical-effect label",
            "nonconformity": "1 - p(observed label)",
            "n_calibration": len(scores),
            "n_positive": result["n_positive"],
            "n_negative": result["n_negative"],
            "underpowered": result["underpowered"],
            "calibration_drugs": [r["drug"] for r in rows],
            "quantiles": {_key(a): conformal_quantile(scores, a) for a in LEVELS},
            "lodo_coverage": {_key(a): lodo_coverage(scores, a) for a in LEVELS},
            "lodo_set_sizes": set_sizes,
        }
    return out


def calibrate() -> dict[str, Any]:
    """Recompute every band from train / LODO residuals.  Reads no held-out row."""
    cl_values = _v2_train_values("clearance_iv")
    vss_values = _v2_train_values("vss")
    quantities: dict[str, Any] = {
        "human_cl_ivive": _fold_quantity(
            _ivive_cl_rows(), source_task="P29.2", unit="mL/min/kg",
            method="well-stirred IVIVE (rat allometry fallback), no per-drug fit, raw (no bias factor)",
        ),
        "human_cl_v2_train_mean": _fold_quantity(
            _v2_mean_rows("clearance_iv"), source_task="P33.3", unit="mL/min/kg",
            method="v2 train arithmetic mean (fallback when no ADME package)",
            frozen_point=mean(cl_values.values()),
        ),
        "human_vss_v2_train_mean": _fold_quantity(
            _v2_mean_rows("vss"), source_task="P33.3", unit="L/kg",
            method="v2 train arithmetic mean (fallback when no ADME package)",
            frozen_point=mean(vss_values.values()),
        ),
    }
    quantities.update(_effect_quantities())
    for name, (status, reason) in UNBANDED.items():
        quantities[name] = {"status": status, "reason": reason}
    return {
        "task": "P30.3",
        "method": "split-conformal on train leave-one-drug-out residuals",
        "levels": [_key(a) for a in LEVELS],
        "quantities": quantities,
    }


def freeze(path: Path = BANDS_PATH) -> dict[str, Any]:
    """Write the band file.  Run once; held-out coverage reads only this file."""
    bands = calibrate()
    path.write_text(json.dumps(bands, indent=2) + "\n")
    return bands


def load_bands(path: Path = BANDS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


# ---- the band API -----------------------------------------------------------
def predict_with_bands(quantity: str, point: float,
                       bands: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Attach the frozen 80% and 95% bands to one point estimate.

    ``fold`` quantities return ``[point / f, point * f]`` per level, with
    ``upper = inf`` and ``bounded = False`` where the calibration set is too
    small for that level.  ``label_set`` quantities take ``p(label == 1)`` and
    return the conformal label set per level.  A quantity with no calibration
    raises ``UnbandedQuantityError``; the caller must list it as pending.
    """
    bands = bands or load_bands()
    entry = bands["quantities"].get(quantity)
    if entry is None:
        raise KeyError(f"{quantity!r} is not a P30.3 quantity")
    if entry["status"] != "calibrated":
        raise UnbandedQuantityError(f"{quantity}: {entry['status']}: {entry.get('reason', '')}")
    point = float(point)
    out_bands = {}
    for level, q in entry["quantiles"].items():
        if entry["kind"] == "fold":
            if point <= 0:
                raise ValueError(f"{quantity}: a fold band needs a positive point estimate")
            if q is None:
                out_bands[level] = {"lower": 0.0, "upper": math.inf, "bounded": False}
            else:
                f = 10 ** q
                out_bands[level] = {"lower": point / f, "upper": point * f, "bounded": True}
        else:
            if not 0.0 <= point <= 1.0:
                raise ValueError(f"{quantity}: a probability must lie in [0, 1]")
            out_bands[level] = {"labels": list(label_set(point, q)), "bounded": q is not None}
    return {
        "quantity": quantity,
        "point": point,
        "kind": entry["kind"],
        "unit": entry["unit"],
        "bands": out_bands,
        "provenance": {
            "source_task": entry["source_task"],
            "method": entry["method"],
            "calibration": "P30.3 split-conformal, train LODO",
            "n_calibration": entry["n_calibration"],
            "underpowered": bool(entry.get("underpowered", False)),
        },
    }


def pending_quantities(bands: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Every product quantity that has no band, with its reason."""
    bands = bands or load_bands()
    return {name: f"{e['status']}: {e['reason']}"
            for name, e in bands["quantities"].items() if e["status"] != "calibrated"}


# ---- held-out coverage (host) --------------------------------------------
def _heldout_block(entry: Mapping[str, Any], scores: list[float], heldout: Iterable[str],
                   missing_truth: list[str], not_scoreable: list[str]) -> dict[str, Any]:
    return {
        "status": "scored" if scores else "searched-and-empty",
        "n_heldout": len(list(heldout)),
        "coverage": {level: coverage(scores, q) for level, q in entry["quantiles"].items()},
        "missing_truth": sorted(missing_truth),
        "not_scoreable": sorted(not_scoreable),
    }


def _heldout_ivive_cl(entry: Mapping[str, Any]) -> dict[str, Any]:
    heldout = load_split()["heldout"]
    model = _pk.fit_translation_models()
    scores, missing_truth, not_scoreable = [], [], []
    for inn in heldout:
        try:
            # Truth first: a stripped checkout must name the drug as a truth
            # gap even when its preclinical input is also missing.
            load_truth_record(inn)
            metric = _pk.score([inn], model=model).get("cl")
        except KeyError:
            missing_truth.append(inn)
            continue
        if metric is None:
            not_scoreable.append(inn)
            continue
        p = metric.predictions[0]
        scores.append(_fold_score(p.predicted, p.observed))
    return _heldout_block(entry, scores, heldout, missing_truth, not_scoreable)


def _heldout_v2_mean(entry: Mapping[str, Any], field: str) -> dict[str, Any]:
    _, heldout = v2_partition()
    truth = load_v2_human_pk()
    scores, missing_truth, not_scoreable = [], [], []
    for inn in heldout:
        if inn not in truth:
            missing_truth.append(inn)
            continue
        observed = _v2_value(truth[inn], field)
        if observed is None or observed <= 0:
            not_scoreable.append(inn)  # truth row present, no usable measurement of this field
            continue
        scores.append(_fold_score(entry["frozen_point"], observed))
    return _heldout_block(entry, scores, heldout, missing_truth, not_scoreable)


def _heldout_effects(bands: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    heldout = load_split()["heldout"]
    models, _ = _effects.fit_effect_models()
    rows, missing_truth, not_scoreable = [], [], []
    for drug in heldout:
        try:
            labels = {row.effect: row.label for row in _effects.effect_labels(drug)}
        except (KeyError, ValueError):
            missing_truth.append(drug)
            continue
        try:
            profile = _effects.occupancy_profile(drug)
        except (KeyError, ValueError):
            not_scoreable.append(drug)
            continue
        rows.append((profile, labels))
    out = {}
    for effect, model in models.items():
        name = f"effect_probability:{effect}"
        entry = bands["quantities"].get(name)
        if entry is None or entry["status"] != "calibrated":
            continue
        scores = []
        for profile, labels in rows:
            if labels.get(effect) in (0, 1):
                p = model.predict_proba(profile.features)
                scores.append(1.0 - (p if labels[effect] == 1 else 1.0 - p))
        out[name] = _heldout_block(entry, scores, heldout, missing_truth, not_scoreable)
    return out


def heldout_coverage(bands: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Score the frozen bands on whatever held-out truth this checkout holds."""
    bands = bands or load_bands()
    q = bands["quantities"]
    result = {
        "human_cl_ivive": _heldout_ivive_cl(q["human_cl_ivive"]),
        "human_cl_v2_train_mean": _heldout_v2_mean(q["human_cl_v2_train_mean"], "clearance_iv"),
        "human_vss_v2_train_mean": _heldout_v2_mean(q["human_vss_v2_train_mean"], "vss"),
    }
    result.update(_heldout_effects(bands))
    n = max((r["coverage"][_key(LEVELS[0])]["n"] for r in result.values()), default=0)
    return {
        "task": "P30.3",
        "status": "scored" if n else "searched-and-empty",
        "max_n_scored": n,
        "quantities": result,
        "pending": pending_quantities(bands),
    }


def heldout_summary_lines(report: Mapping[str, Any]) -> list[str]:
    lines = []
    for name, r in report["quantities"].items():
        cells = []
        for level, c in r["coverage"].items():
            rate = "n/a" if c["rate"] is None else f"{c['rate']:.3f}"
            cells.append(f"{level}: {c['covered']}/{c['n']} ({rate})")
        lines.append(f"{name}: {r['status']}; " + "; ".join(cells))
    for name, reason in report["pending"].items():
        lines.append(f"{name}: {reason}")
    return lines


__all__ = [
    "BANDS_PATH",
    "LEVELS",
    "UNBANDED",
    "UnbandedQuantityError",
    "calibrate",
    "conformal_quantile",
    "coverage",
    "freeze",
    "heldout_coverage",
    "heldout_summary_lines",
    "label_set",
    "load_bands",
    "lodo_coverage",
    "pending_quantities",
    "predict_with_bands",
]
