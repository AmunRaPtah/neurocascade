"""P30.3 distribution-free calibrated uncertainty for product predictions.

Every number the product returns to a discovery team needs an interval whose
width is measured, not asserted.  This module fits split-/leave-one-out
conformal intervals on TRAIN residuals only and measures their empirical
leave-one-drug-out (LODO) coverage, so the band attached to a prediction is
backed by the fraction of held-back drugs it actually contained.

Two prediction families need two conformal objects:

* Continuous human PK (CL, Vss).  The product's point predictor is the
  leave-one-drug-out train geometric mean, the P33.3 baseline; no per-drug
  model beats it yet, so it is the thing whose uncertainty we calibrate.  The
  nonconformity score is the absolute log-fold error, so the interval is
  multiplicative: ``point * exp(+/- radius)``.  A multiplicative interval is
  the right shape for a strictly positive, roughly log-normal quantity.

* Per-effect clinical probability (P30.1 ``EffectModel``).  A probability does
  not have a distribution-free *interval* against a 0/1 outcome, so the
  calibrated object is the standard conformal label *set*: the subset of
  ``{0, 1}`` whose nonconformity ``1 - p(label)`` is within the calibrated
  quantile.  An uninformative ``{0, 1}`` set is an honest statement that the
  model cannot, at that confidence, exclude either outcome -- which is what a
  near-chance model (P30.1 is at chance on train) should say.

Nothing here is fitted on a held-out drug.  The conductor checkout has no
held-out truth at all (v2 strips 67 PK rows; the P28.1 split strips 55 effect
rows), so the available-data number reported here is the TRAIN LODO coverage,
and the held-out coverage is scored once on the host after the radii below are
frozen.  The frozen radius per quantity is the one a new compound's interval
uses; it is computed on every available train residual.

Reference for the method: Vovk, Gammerman & Shafer, *Algorithmic Learning in a
Random World* (2005); Lei et al. 2018 (J Am Stat Assoc 113:1094) for the
split-conformal regression form.  These are methodological citations, not a
source for any neurocascade biological parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence


# Nominal coverages the product reports.  Fixed before any coverage is
# measured; these are reporting choices, not biological constants.
NOMINAL_COVERAGES = (0.80, 0.95)


@dataclass(frozen=True)
class Interval:
    """A multiplicative prediction interval around a positive point estimate."""

    point: float
    lower: float
    upper: float
    nominal_coverage: float
    radius_log: float
    kind: str = "multiplicative-log-fold"

    @property
    def uninformative(self) -> bool:
        return not math.isfinite(self.radius_log)


@dataclass(frozen=True)
class LabelSet:
    """A conformal prediction set for a binary clinical-effect label."""

    labels: tuple[int, ...]
    point_probability: float
    nominal_coverage: float

    @property
    def uninformative(self) -> bool:
        return set(self.labels) == {0, 1}


def conformal_log_radius(residuals: Sequence[float], nominal_coverage: float) -> float:
    """The split-conformal radius: the k-th smallest absolute residual.

    ``k = ceil((n + 1) * nominal_coverage)``.  When that index exceeds ``n``
    the finite-sample guarantee cannot be met with any finite radius and the
    radius is ``+inf`` (the interval is the whole positive line -- flagged as
    uninformative rather than silently clipped to the largest residual).
    """
    values = sorted(float(r) for r in residuals)
    n = len(values)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * nominal_coverage)
    if k > n:
        return math.inf
    return values[k - 1]


def multiplicative_interval(point: float, radius_log: float, nominal_coverage: float) -> Interval:
    if point <= 0:
        raise ValueError("multiplicative interval requires a positive point estimate")
    if math.isfinite(radius_log):
        factor = math.exp(radius_log)
        lower, upper = point / factor, point * factor
    else:
        lower, upper = 0.0, math.inf
    return Interval(point, lower, upper, nominal_coverage, radius_log)


# ---- continuous quantities ------------------------------------------------
def lodo_geometric_mean(values: Mapping[str, float]) -> dict[str, float]:
    """Leave-one-drug-out geometric-mean predictor, one prediction per drug."""
    items = sorted(values.items())
    logs = {inn: math.log(float(v)) for inn, v in items}
    out: dict[str, float] = {}
    for inn, _ in items:
        rest = [logv for other, logv in logs.items() if other != inn]
        out[inn] = math.exp(mean(rest))
    return out


def _abs_log_fold(predicted: float, observed: float) -> float:
    return abs(math.log(predicted / observed))


def continuous_coverage(values: Mapping[str, float]) -> dict:
    """Frozen radii and jackknife LODO coverage for a positive PK quantity.

    ``values`` maps drug -> observed value on the fitting split.  The point
    predictor is the LODO geometric mean; the calibration residuals are the
    LODO absolute log-fold errors.  Empirical coverage is a *second*,
    non-circular leave-one-out over the residual set: each drug's membership is
    decided by a radius computed from the other drugs' residuals only, so no
    drug helps set the radius that judges it.
    """
    values = {inn: float(v) for inn, v in values.items() if v is not None and float(v) > 0}
    n = len(values)
    predictor = lodo_geometric_mean(values) if n >= 2 else {}
    residuals = {inn: _abs_log_fold(predictor[inn], values[inn]) for inn in predictor}
    residual_list = list(residuals.values())

    per_coverage = {}
    for nominal in NOMINAL_COVERAGES:
        frozen_radius = conformal_log_radius(residual_list, nominal)
        covered = 0
        scored = 0
        for inn, r in residuals.items():
            others = [residuals[o] for o in residuals if o != inn]
            radius = conformal_log_radius(others, nominal)
            scored += 1
            if r <= radius:
                covered += 1
        per_coverage[nominal] = {
            "nominal": nominal,
            "frozen_log_radius": frozen_radius,
            "frozen_fold_width": math.exp(frozen_radius) if math.isfinite(frozen_radius) else math.inf,
            "empirical_lodo_coverage": (covered / scored) if scored else None,
            "n": scored,
            "uninformative": not math.isfinite(frozen_radius),
        }
    return {"n": n, "coverage": per_coverage}


def interval_for(point: float, frozen_log_radius: float, nominal_coverage: float) -> Interval:
    """Attach a frozen conformal radius to a fresh point estimate."""
    return multiplicative_interval(point, frozen_log_radius, nominal_coverage)


# ---- per-effect probabilities ---------------------------------------------
def _true_label_nonconformity(label: int, p_positive: float) -> float:
    p_true = p_positive if label == 1 else (1.0 - p_positive)
    return 1.0 - p_true


def classification_coverage(predictions: Sequence[tuple[int, float]]) -> dict:
    """Conformal label-set coverage for one effect.

    ``predictions`` is a sequence of ``(observed_label, predicted_p_positive)``
    LODO rows.  For each nominal level the calibration quantile of the
    true-label nonconformity ``1 - p(label)`` is taken, and empirical coverage
    is a second leave-one-out: drug i's set is built from the quantile of the
    other drugs' scores, and counts as covered iff its observed label is in the
    set.  Mean set size reports how often the set collapses to the informative
    singletons versus the uninformative ``{0, 1}``.
    """
    rows = [(int(y), float(p)) for y, p in predictions]
    scores = [_true_label_nonconformity(y, p) for y, p in rows]
    n = len(rows)
    per_coverage = {}
    for nominal in NOMINAL_COVERAGES:
        frozen_q = _score_quantile(scores, nominal)
        covered = 0
        set_sizes = []
        for i, (y, p) in enumerate(rows):
            others = scores[:i] + scores[i + 1:]
            q = _score_quantile(others, nominal)
            labels = tuple(l for l in (0, 1)
                           if _true_label_nonconformity(l, p) <= q)
            set_sizes.append(len(labels))
            if y in labels:
                covered += 1
        per_coverage[nominal] = {
            "nominal": nominal,
            "frozen_score_quantile": frozen_q,
            "empirical_lodo_coverage": (covered / n) if n else None,
            "mean_set_size": mean(set_sizes) if set_sizes else None,
            "n": n,
        }
    return {"n": n, "coverage": per_coverage}


def _score_quantile(scores: Sequence[float], nominal_coverage: float) -> float:
    values = sorted(float(s) for s in scores)
    n = len(values)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * nominal_coverage)
    if k > n:
        return math.inf  # every label admitted: the uninformative {0, 1} set
    return values[k - 1]


def label_set_for(p_positive: float, frozen_score_quantile: float, nominal_coverage: float) -> LabelSet:
    labels = tuple(l for l in (0, 1)
                   if _true_label_nonconformity(l, p_positive) <= frozen_score_quantile)
    return LabelSet(labels, float(p_positive), nominal_coverage)


# ---- the product-facing report --------------------------------------------
def _v2_pk_values(parameter: str) -> dict[str, float]:
    from .v2 import load_v2_human_pk, v2_partition

    pk = load_v2_human_pk()
    train, _ = v2_partition()
    out: dict[str, float] = {}
    for inn in train:
        row = pk.get(inn, {})
        field = row.get(parameter) if isinstance(row, Mapping) else None
        if isinstance(field, Mapping) and field.get("value") is not None and float(field["value"]) > 0:
            out[inn] = float(field["value"])
    return out


def _effect_predictions() -> dict[str, list[tuple[int, float]]]:
    """P30.1 leave-one-drug-out per-effect (label, predicted probability)."""
    from .effects import leave_one_drug_out

    lodo = leave_one_drug_out()
    out: dict[str, list[tuple[int, float]]] = {}
    for effect, result in lodo.items():
        out[effect] = [
            (int(row["observed"]), float(row["predicted_probability"]))
            for row in result["predictions"]
        ]
    return out


# Quantities whose upstream prediction task is not done: they are declared
# pending, never filled with an unscored point estimate or a made-up band.
PENDING_QUANTITIES = {
    "target_engagement_dose": "P29.3 blocked on P29.5 oral-exposure/PET data",
    "panel_occupancy": "needs therapeutic Cmax and Kp,uu; P33.3 reports both as no-data",
}


def calibration_report(*, include_effects: bool = True) -> dict:
    """Freeze every radius and measure train LODO coverage per quantity."""
    continuous = {
        "human_cl": {
            "source": "v2 human IV CL truth; LODO train geometric-mean predictor",
            **continuous_coverage(_v2_pk_values("clearance_iv")),
        },
        "vss": {
            "source": "v2 human Vss truth; LODO train geometric-mean predictor",
            **continuous_coverage(_v2_pk_values("vss")),
        },
    }
    effects = {}
    if include_effects:
        for effect, rows in _effect_predictions().items():
            n_pos = sum(y == 1 for y, _ in rows)
            n_neg = sum(y == 0 for y, _ in rows)
            report = classification_coverage(rows)
            report["n_positive"] = n_pos
            report["n_negative"] = n_neg
            report["underpowered"] = min(n_pos, n_neg) < 5
            report["source"] = "P30.1 EffectModel LODO probabilities; conformal label set"
            effects[effect] = report
    n_scored = max(
        [continuous["human_cl"]["n"], continuous["vss"]["n"]]
        + [e["n"] for e in effects.values()],
        default=0,
    )
    return {
        "task": "P30.3",
        "nominal_coverages": list(NOMINAL_COVERAGES),
        "continuous": continuous,
        "effects": effects,
        "pending": PENDING_QUANTITIES,
        "n_scored": n_scored,
        "held_out_status": "host-pending: no held-out truth in this checkout "
        "(v2 strips 67 PK rows; P28.1 strips 55 effect rows)",
        "notes": [
            "Radii/quantiles here are frozen on train residuals; the host scores "
            "held-out coverage once with these same radii.",
            "A probability has no distribution-free interval against a 0/1 label; "
            "its calibrated object is a conformal label set, and a {0,1} set is an "
            "honest 'cannot exclude either outcome', not a missing number.",
        ],
    }


def _frozen_from_report(report: Mapping) -> dict:
    """Extract the frozen radii/quantiles a fresh prediction needs."""
    cont = {}
    for quantity, block in report["continuous"].items():
        cont[quantity] = {
            float(nominal): float(cov["frozen_log_radius"])
            for nominal, cov in block["coverage"].items()
        }
    eff = {}
    for effect, block in report.get("effects", {}).items():
        eff[effect] = {
            float(nominal): float(cov["frozen_score_quantile"])
            for nominal, cov in block["coverage"].items()
        }
    return {"continuous": cont, "effects": eff}


def predict_with_bands(drug: str, *, report: Mapping | None = None) -> dict:
    """Every product number for ``drug`` beside its calibrated band.

    This is the band-attachment layer P32.1's ``predict_compound`` surfaces:
    each point estimate carries an interval/label-set at both nominal
    coverages, pending quantities are declared (never filled with an unscored
    estimate), and the point estimate for a fresh compound is the TRAIN-only
    predictor, so no held-out drug informs the number or the band.

    ``report`` can be injected to avoid recomputing the slow P30.1 LODO; by
    default it is computed with frozen radii from the train split.
    """
    from .effects import fit_effect_models, occupancy_profile
    from .v2 import load_v2_human_pk, v2_partition

    report = report or calibration_report()
    frozen = _frozen_from_report(report)

    pk = load_v2_human_pk()
    train, _ = v2_partition()

    predictions: dict[str, dict] = {}
    for quantity, field in (("human_cl", "clearance_iv"), ("vss", "vss")):
        values = {
            inn: float(pk[inn][field]["value"])
            for inn in train
            if inn in pk and isinstance(pk[inn].get(field), Mapping)
            and pk[inn][field].get("value") is not None and float(pk[inn][field]["value"]) > 0
        }
        point = math.exp(mean(math.log(v) for v in values.values()))
        intervals = {
            nominal: interval_for(point, frozen["continuous"][quantity][nominal], nominal)
            for nominal in NOMINAL_COVERAGES
        }
        predictions[quantity] = {
            "point": point,
            "predictor": "TRAIN geometric mean (P33.3 baseline; no per-drug model beats it)",
            "intervals": intervals,
        }

    models, _ = fit_effect_models()
    try:
        profile = occupancy_profile(drug)
    except (KeyError, ValueError):
        profile = None
    effects: dict[str, dict] = {}
    for effect, model in models.items():
        if effect not in frozen["effects"]:
            continue
        if profile is None:
            effects[effect] = {"status": "no exposure/binding input for this compound"}
            continue
        p = model.predict_proba(profile.features)
        effects[effect] = {
            "point_probability": p,
            "label_sets": {
                nominal: label_set_for(p, frozen["effects"][effect][nominal], nominal)
                for nominal in NOMINAL_COVERAGES
            },
        }
    predictions["effects"] = effects

    pending = {name: reason for name, reason in PENDING_QUANTITIES.items()}
    return {
        "drug": drug,
        "nominal_coverages": list(NOMINAL_COVERAGES),
        "predictions": predictions,
        "pending": pending,
    }


__all__ = [
    "NOMINAL_COVERAGES",
    "Interval",
    "LabelSet",
    "calibration_report",
    "predict_with_bands",
    "classification_coverage",
    "conformal_log_radius",
    "continuous_coverage",
    "interval_for",
    "label_set_for",
    "lodo_geometric_mean",
    "multiplicative_interval",
    "PENDING_QUANTITIES",
]
