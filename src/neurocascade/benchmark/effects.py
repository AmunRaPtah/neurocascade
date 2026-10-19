"""P30.1 occupancy-profile to clinical-effect model.

This module deliberately keeps the exposure and label seams small.  The
clinical exposure function is the only place that turns a benchmark record
into an exposure value, and the label adapter keeps the placebo provenance
visible instead of treating a label table's ``greater than placebo`` caption
as a numerical placebo arm.

The current benchmark does not contain molecular weights or paired placebo
percentages.  Consequently the feature called ``occupancy`` is a transparent
occupancy *index*: ``x / (x + Ki)`` where ``x`` is the observed-exposure
quantity or the pre-registered dose-proportional proxy.  It is not presented
as a molar brain concentration.  A future provider can supply a molar,
unbound concentration without changing the model interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import minimize

from . import load_binding_panel, load_split, load_truth_record


EFFECTS = (
    "eps",
    "sedation",
    "weight_gain",
    "anticholinergic",
    "orthostasis",
    "sexual_dysfunction",
    "seizure",
)

# This is a classification-analysis setting, not a biological parameter.  It
# is reported as an illustrative analysis hyperparameter and is not silently
# promoted to a calibrated scientific constant.
ILLUSTRATIVE_L2_STRENGTH = 1.0
UNDERPOWERED_MIN_CLASS = 5


@dataclass(frozen=True)
class Exposure:
    drug: str
    tier: int
    dose_mg_day: float
    exposure_value: float
    exposure_kind: str
    source: str
    source_quotes: tuple[str, ...]
    observed_cl_over_f_l_h: float | None
    unbound_fraction: float | None
    kp_uu: float | None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tier not in (1, 2):
            raise ValueError("exposure tier must be 1 or 2")
        if self.dose_mg_day <= 0 or self.exposure_value <= 0:
            raise ValueError("dose and exposure must be positive")
        if self.tier == 1 and self.observed_cl_over_f_l_h is None:
            raise ValueError("tier-1 exposure requires observed human CL/F")


@dataclass(frozen=True)
class LabelObservation:
    drug: str
    effect: str
    label: int | None
    drug_percent: float | None
    placebo_percent: float | None
    source: str
    rule_applied: str
    placebo_arm_published: bool
    placebo_pair_available: bool
    evidence_terms: tuple[str, ...]


@dataclass(frozen=True)
class OccupancyProfile:
    drug: str
    features: Mapping[str, float]
    exposure: Exposure
    feature_kind: str
    panel_source: str


@dataclass(frozen=True)
class EffectModel:
    effect: str
    targets: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    n_positive: int
    n_negative: int
    regularization_strength: float
    status: str
    # Per-target mean and standard deviation of the exposure features the model
    # was fitted on. The features are occupancy indices spanning about 1e-4 to
    # 4e-1, so an L2 penalty applied to raw values shrinks every coefficient to
    # roughly zero and the model becomes an intercept whose value tracks the
    # fold's class balance (P30.1, 2026-10-18: that produced a leave-one-out
    # AUROC of exactly 0.000 for every effect while the in-sample AUROC was
    # 0.755, the signature of an intercept-only fit, not of a real inversion).
    feature_mean: tuple[float, ...] = ()
    feature_scale: tuple[float, ...] = ()

    def standardised(self, features: Mapping[str, float]) -> tuple[float, ...]:
        if not self.feature_mean:
            return tuple(float(features.get(target, 0.0)) for target in self.targets)
        return tuple(
            (float(features.get(target, 0.0)) - mean) / scale
            for target, mean, scale in zip(self.targets, self.feature_mean, self.feature_scale)
        )

    def predict_proba(self, features: Mapping[str, float]) -> float:
        standardised = self.standardised(features)
        z = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardised)
        )
        # Stable enough for the small benchmark and avoids a dependency on a
        # black-box classifier.
        if z >= 0:
            q = math.exp(-z)
            return 1.0 / (1.0 + q)
        q = math.exp(z)
        return q / (1.0 + q)

    def coefficient_table(self) -> list[dict]:
        return [
            {"target": target, "coefficient": coefficient,
             "per": "one standard deviation of the train-split occupancy index"}
            for target, coefficient in zip(self.targets, self.coefficients)
        ]


def _dose_mg_day(record: dict) -> tuple[float | None, str, tuple[str, ...]]:
    """Return an exact WHO oral DDD, or an exact single label dose.

    A label range is not collapsed to a midpoint.  This prevents an apparently
    precise exposure from being manufactured from a dose interval.
    """
    oral = []
    for atc in record.get("dose", {}).get("ddd", []):
        for row in atc.get("ddd", []):
            if row.get("route") != "O":
                continue
            value = float(row["ddd"])
            unit = row.get("unit")
            if unit == "g":
                value *= 1000.0
            elif unit == "mcg":
                value *= 0.001
            elif unit != "mg":
                continue
            oral.append((value, "WHO ATC/DDD index"))
    if oral and len({value for value, _ in oral}) == 1:
        return oral[0][0], oral[0][1], ()
    if oral:
        return None, "WHO ATC/DDD index", ("multiple conflicting oral DDD values",)

    exact = []
    ranges = []
    for row in record.get("dose", {}).get("label", []):
        if row.get("quantity") != "daily_dose_mg" or row.get("route") not in ("oral", "sublingual"):
            continue
        if row.get("low") is not None and row.get("high") is not None:
            if float(row["low"]) == float(row["high"]):
                exact.append((float(row["low"]), tuple(row.get("quotes", ()))))
            else:
                ranges.append((float(row["low"]), tuple(row.get("quotes", ()))))
        elif row.get("low") is not None:
            # A printed fixed recommended dose has no upper endpoint.
            exact.append((float(row["low"]), tuple(row.get("quotes", ()))))
    if len(exact) == 1:
        return exact[0][0], "selected label exact daily dose", exact[0][1]
    if len(exact) > 1 and len({value for value, _ in exact}) == 1:
        return exact[0][0], "selected label exact daily dose", tuple(
            quote for _, quotes in exact for quote in quotes
        )
    if len(ranges) == 1:
        # The lower endpoint is printed by the source.  It is a conservative
        # boundary observation, not an invented midpoint; the provenance is
        # carried in the source string so it cannot be mistaken for a typical
        # prescribed dose.
        return ranges[0][0], "selected label lower dose bound", ranges[0][1]
    return None, "P28.4 dose record", ("no usable oral clinical dose",)


def _observed_cl_over_f(record: dict) -> tuple[float, tuple[str, ...]] | None:
    rows = [
        row for row in record.get("pk", {}).get("label", [])
        if row.get("quantity") == "cl_over_f"
        and row.get("value") is not None
        and row.get("units") == "L/h"
        and row.get("route") == "oral"
    ]
    if not rows:
        return None
    values = {float(row["value"]) for row in rows}
    if len(values) != 1:
        # The label gives multiple incompatible estimates; no unregistered
        # averaging rule is allowed to turn them into one exposure.
        return None
    return next(iter(values)), tuple(
        quote for row in rows for quote in row.get("quotes", ())
    )


def train_observed_cl_over_f(train_drugs: Iterable[str] | None = None) -> float:
    """Median absolute oral CL/F available on the training side.

    Per-kg CL/F is not mixed with absolute CL/F because the benchmark has no
    source-defined subject weight for that conversion.
    """
    drugs = tuple(train_drugs or load_split()["train"])
    values = []
    for drug in drugs:
        try:
            observed = _observed_cl_over_f(load_truth_record(drug))
        except KeyError:
            continue
        if observed is not None:
            values.append(observed[0])
    if not values:
        raise ValueError("no absolute observed human oral CL/F on the training split")
    return float(median(values))


def _kp_uu(drug: str) -> tuple[float | None, str]:
    """Return a P29.1 Kp,uu only when this repository has one for the drug."""
    # P29.1's four human CSF records are the reachable human Kp,uu values.
    # Import lazily: the benchmark effect model must remain usable without the
    # broader human model package being imported at module import time.
    from ..human.kp_uu import HUMAN_KP_UU_CSF, kp_uu_records

    direct = kp_uu_records(drug)
    if direct and direct[0].kp_uu is not None:
        return direct[0].kp_uu, direct[0].citation
    csf = [row for row in HUMAN_KP_UU_CSF if row.drug == drug]
    if csf:
        return csf[0].kp_uu_csf, csf[0].citation + "; P29.1 human CSF surrogate"
    return None, "P29.1: no drug-specific human Kp,uu available"


def clinical_exposure(drug: str, *, train_drugs: Iterable[str] | None = None) -> Exposure:
    """Return the pluggable observed-human or pre-registered proxy exposure."""
    record = load_truth_record(drug)
    dose, dose_source, dose_quotes = _dose_mg_day(record)
    if dose is None:
        raise ValueError(f"{drug}: no unique exact oral clinical dose")
    observed = _observed_cl_over_f(record)
    kp, kp_source = _kp_uu(drug)
    if observed is not None:
        clf, clf_quotes = observed
        value = dose / 24.0 / clf
        notes = ("observed total oral Css index; no human fu was recorded",)
        if kp is not None:
            value *= kp
            notes += ("multiplied by drug-specific P29.1 Kp,uu",)
        return Exposure(
            drug=drug,
            tier=1,
            dose_mg_day=dose,
            exposure_value=value,
            exposure_kind="observed-human-CL/F-index",
            source=f"P28.4 {dose_source}; observed human oral CL/F; {kp_source}",
            source_quotes=tuple(dose_quotes) + tuple(clf_quotes),
            observed_cl_over_f_l_h=clf,
            unbound_fraction=None,
            kp_uu=kp,
            notes=notes,
        )
    median_clf = train_observed_cl_over_f(train_drugs)
    value = dose / median_clf
    if kp is not None:
        value *= kp
    return Exposure(
        drug=drug,
        tier=2,
        dose_mg_day=dose,
        exposure_value=value,
        exposure_kind="dose-proportional-CL/F-proxy-index",
        source=(
            f"P28.4 {dose_source}; train median observed absolute human CL/F "
            f"({median_clf:g} L/h); {kp_source}"
        ),
        source_quotes=tuple(dose_quotes),
        observed_cl_over_f_l_h=None,
        unbound_fraction=None,
        kp_uu=kp,
        notes=(
            "tier-2 fallback is pre-registered dose/Ki ranking, not predicted PK",
            "no human unbound fraction was recorded",
        ),
    )


def effect_labels(drug: str) -> tuple[LabelObservation, ...]:
    """Adapt P28.4 labels while preserving placebo evidence and fallback rules."""
    record = load_truth_record(drug)
    output = []
    for effect in EFFECTS:
        evidence = record["effects"]["evidence"].get(effect, [])
        percents = []
        terms = []
        sources = set()
        placebo_published = False
        for row in evidence:
            if "drug_arm_percent" in row:
                percent = float(row["drug_arm_percent"])
            elif "frequency_lower" in row:
                percent = float(row["frequency_lower"]) * 100.0
            else:
                continue
            percents.append(percent)
            terms.append(row.get("term", ""))
            sources.add(row.get("source", "unknown"))
            caption = row.get("table_caption", "").lower()
            placebo_published |= "placebo" in caption
        sider = record["effects"].get("sources", {}).get("sider", {})
        placebo_published |= bool(sider.get("n_frequency_rows_placebo", 0))
        if not evidence:
            rule = "unknown-no-frequency-evidence"
        elif placebo_published:
            # The committed P28.4 rows preserve drug incidence but not the
            # paired placebo percentage.  Refuse subtraction rather than
            # borrowing a placebo value from another study.
            rule = "absolute-fallback-unpaired-placebo-percentage-not-printed"
        else:
            rule = "absolute-fallback-no-placebo-arm-published"
        source = "; ".join(sorted(sources)) or "P28.4 effect record"
        output.append(LabelObservation(
            drug=drug,
            effect=effect,
            label=record["effects"]["label"].get(effect),
            drug_percent=max(percents) if percents else None,
            placebo_percent=None,
            source=source,
            rule_applied=rule,
            placebo_arm_published=placebo_published,
            placebo_pair_available=False,
            evidence_terms=tuple(sorted(set(terms))),
        ))
    return tuple(output)


def occupancy_profile(drug: str, *, train_drugs: Iterable[str] | None = None) -> OccupancyProfile:
    exposure = clinical_exposure(drug, train_drugs=train_drugs)
    panel = load_binding_panel(drug)
    features = {}
    for target, record in panel.get("targets", {}).items():
        ki = record.get("ki_nM")
        if ki is None or float(ki) <= 0:
            continue
        ratio = exposure.exposure_value / float(ki)
        features[target] = ratio / (1.0 + ratio)
    return OccupancyProfile(
        drug=drug,
        features=features,
        exposure=exposure,
        feature_kind="clinical-dose occupancy index; Ki in nM, exposure index in source units",
        panel_source="P28.2 human Ki panel",
    )


def _fit_binary(effect: str, targets: tuple[str, ...], x: np.ndarray, y: np.ndarray,
                regularization_strength: float) -> EffectModel:
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        return EffectModel(effect, targets, float(math.log((positive + 0.5) / (negative + 0.5))),
                           tuple(0.0 for _ in targets), positive, negative,
                           regularization_strength, "single-class")

    # Standardise with this fit's own rows (the LOO fold's training drugs, never
    # the left-out drug), then penalise the standardised coefficients. Raw
    # occupancy indices are three to four orders of magnitude below 1, so a
    # fixed L2 on raw values dominates the likelihood and the fit collapses to
    # the fold intercept; standardising makes the penalty comparable to the
    # per-SD signal each target carries.
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardised = (x - mean) / scale

    def objective(theta):
        z = theta[0] + standardised @ theta[1:]
        loss = np.logaddexp(0.0, z) - y * z
        value = float(np.mean(loss) + regularization_strength * np.dot(theta[1:], theta[1:]) / 2.0)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
        gradient = np.empty_like(theta)
        gradient[0] = np.mean(p - y)
        gradient[1:] = ((standardised.T @ (p - y)) / len(y)) + regularization_strength * theta[1:]
        return value, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        np.zeros(x.shape[1] + 1),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"{effect}: regularized logistic fit failed: {result.message}")
    return EffectModel(effect, targets, float(result.x[0]), tuple(float(v) for v in result.x[1:]),
                       positive, negative, regularization_strength, "fitted",
                       feature_mean=tuple(float(v) for v in mean),
                       feature_scale=tuple(float(v) for v in scale))


def _training_rows(drugs: Iterable[str]) -> tuple[tuple[str, OccupancyProfile, dict[str, int | None]], ...]:
    split = set(load_split()["train"])
    rows = []
    for drug in drugs:
        if drug not in split:
            raise ValueError(f"fit attempted to use non-training drug {drug!r}")
        try:
            profile = occupancy_profile(drug, train_drugs=split)
            labels = {row.effect: row.label for row in effect_labels(drug)}
        except (KeyError, ValueError):
            continue
        if not any(label in (0, 1) for label in labels.values()):
            continue
        rows.append((drug, profile, labels))
    return tuple(rows)


def fit_effect_models(*, drugs: Iterable[str] | None = None,
                      regularization_strength: float = ILLUSTRATIVE_L2_STRENGTH
                      ) -> tuple[dict[str, EffectModel], tuple[str, ...]]:
    """Fit every effect on train only and return models plus eligible drugs."""
    rows = _training_rows(tuple(drugs or load_split()["train"]))
    targets = tuple(sorted({target for _, profile, _ in rows for target in profile.features}))
    models = {}
    for effect in EFFECTS:
        usable = [(profile, labels[effect]) for _, profile, labels in rows
                  if labels.get(effect) in (0, 1)]
        x = np.asarray([[profile.features.get(target, 0.0) for target in targets]
                        for profile, _ in usable], dtype=float)
        y = np.asarray([label for _, label in usable], dtype=float)
        if len(y) == 0:
            continue
        models[effect] = _fit_binary(effect, targets, x, y, regularization_strength)
    return models, tuple(drug for drug, _, _ in rows)


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positive = sum(label == 1 for label in labels)
    negative = sum(label == 0 for label in labels)
    if not positive or not negative:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(labels[order[i]] == 1 for i in range(index, end))
        index = end
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def leave_one_drug_out(*, regularization_strength: float = ILLUSTRATIVE_L2_STRENGTH,
                       tiers: tuple[int, ...] | None = None) -> dict:
    """Primary internal estimate, with one held-out train drug per prediction."""
    split = load_split()["train"]
    rows = _training_rows(split)
    if tiers is not None:
        rows = tuple(row for row in rows if row[1].exposure.tier in tiers)
    results = {}
    for effect in EFFECTS:
        predictions = []
        for left_out, profile, labels in rows:
            if labels.get(effect) not in (0, 1):
                continue
            fit_rows = [row for row in rows if row[0] != left_out and row[2].get(effect) in (0, 1)]
            targets = tuple(sorted({target for _, p, _ in rows for target in p.features}))
            x = np.asarray([[p.features.get(target, 0.0) for target in targets]
                            for _, p, _ in fit_rows], dtype=float)
            y = np.asarray([lab[effect] for _, _, lab in fit_rows], dtype=float)
            model = _fit_binary(effect, targets, x, y, regularization_strength)
            predictions.append({
                "drug": left_out,
                "observed": int(labels[effect]),
                "predicted_probability": model.predict_proba(profile.features),
            })
        labs = [row["observed"] for row in predictions]
        scores = [row["predicted_probability"] for row in predictions]
        positive, negative = sum(v == 1 for v in labs), sum(v == 0 for v in labs)
        results[effect] = {
            "n_scored": len(predictions),
            "n_positive": positive,
            "n_negative": negative,
            "underpowered": min(positive, negative) < UNDERPOWERED_MIN_CLASS,
            "auroc": _auc(labs, scores),
            "predictions": predictions,
        }
    return results


def heldout_evaluation(*, regularization_strength: float = ILLUSTRATIVE_L2_STRENGTH) -> dict:
    """Score the frozen held-out split wherever the caller supplies its truth.

    The conductor's checkout removes every held-out truth file, so there the
    label lookup fails for all 55 drugs and this returns ``searched-and-empty``
    with every drug listed: a missing label is never read as a negative. A host
    holding the full tree scores the drugs it can and reports the remainder
    under the reason that actually applies, so an input gap (no unique exact
    oral clinical dose) is never reported as a truth gap. Scoring is the only
    thing that touches held-out data; nothing is fitted on it.
    """
    heldout = load_split()["heldout"]
    models, _ = fit_effect_models(regularization_strength=regularization_strength)
    rows: list[tuple[str, OccupancyProfile, dict[str, int | None]]] = []
    missing_truth: list[str] = []
    missing_input: list[str] = []
    unlabelled: list[str] = []
    for drug in heldout:
        try:
            labels = {row.effect: row.label for row in effect_labels(drug)}
        except (KeyError, ValueError):
            missing_truth.append(drug)
            continue
        try:
            profile = occupancy_profile(drug)
        except (KeyError, ValueError):
            missing_input.append(drug)
            continue
        # Two independent reasons a drug cannot be scored, reported separately
        # and allowed to overlap: no known binary label for any effect (P28.4
        # carries SIDER/label evidence without always resolving an incidence
        # threshold), and no unique exact oral clinical dose. Of the 55
        # held-out drugs, 37 have no known label, 16 have no dose, 15 have
        # neither, and 17 have both -- those 17 are the scoring set.
        if not any(label in (0, 1) for label in labels.values()):
            unlabelled.append(drug)
            continue
        rows.append((drug, profile, labels))
    if not rows:
        return {
            "status": "searched-and-empty",
            "n_scored": 0,
            "n_heldout": len(heldout),
            "missing_truth": missing_truth,
            "missing_input": missing_input,
            "unlabelled": unlabelled,
        }
    result = {}
    for effect, model in models.items():
        scored = [(model.predict_proba(profile.features), labels[effect])
                  for _, profile, labels in rows if labels.get(effect) in (0, 1)]
        labs = [label for _, label in scored]
        scores = [score for score, _ in scored]
        positive, negative = sum(v == 1 for v in labs), sum(v == 0 for v in labs)
        result[effect] = {
            "n_scored": len(scored),
            "n_positive": positive,
            "n_negative": negative,
            "underpowered": min(positive, negative) < UNDERPOWERED_MIN_CLASS,
            "auroc": _auc(labs, scores),
        }
    return {
        "status": "scored",
        "n_scored": len(rows),
        "n_heldout": len(heldout),
        "missing_truth": missing_truth,
        "missing_input": missing_input,
        "unlabelled": unlabelled,
        "effects": result,
    }


def report(*, regularization_strength: float = ILLUSTRATIVE_L2_STRENGTH) -> dict:
    """Build a serialisable report without fitting on held-out drugs."""
    models, fitted_drugs = fit_effect_models(regularization_strength=regularization_strength)
    return {
        "task": "P30.1",
        "fit_split": "train",
        "fitted_drugs": list(fitted_drugs),
        "regularization": {
            "kind": "L2 logistic coefficient penalty",
            "strength": regularization_strength,
            "status": "illustrative analysis hyperparameter, not a biological parameter",
        },
        "models": {
            effect: {
                "n_positive": model.n_positive,
                "n_negative": model.n_negative,
                "status": model.status,
                "coefficients": model.coefficient_table(),
            }
            for effect, model in models.items()
        },
        "leave_one_drug_out": leave_one_drug_out(regularization_strength=regularization_strength),
        "tier1_only_sensitivity": leave_one_drug_out(
            regularization_strength=regularization_strength, tiers=(1,)
        ),
        "heldout": heldout_evaluation(regularization_strength=regularization_strength),
    }
