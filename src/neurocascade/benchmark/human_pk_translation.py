"""P29.2: train-only translation of preclinical inputs to human PK.

This module deliberately scores only quantities that the frozen benchmark
actually contains.  The preclinical records do not contain animal body
weights, so the animal arm is a single-species log scaling of the published
per-kg value, with its intercept and exponent fitted on the training drugs.
It is named ``allometric`` in the result because it is the empirical
single-species scaling form, but it must not be confused with a mechanistic
body-weight allometry: no body weight is available in P28.3.

The IVIVE arm is restricted to the one common, compatible group present in
the benchmark: human microsomal CLint in mL/min/g and human plasma fu.  No
incompatible matrices or units are pooled.  All coefficients are fitted from
the frozen training split only.  Human truth is read only while scoring the
held-out split.

Method citations:

* Boxenbaum H. (1982), PMID 7120049, describes interspecies PK scaling.
* Ring et al. (2011), PMID 21541938, compares allometry and IVIVE on blinded
  preclinical data and reports the importance of binding corrections.
* Obach RS. (1999), PMID 10534321, describes the microsomal intrinsic-
  clearance approach.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from . import load_adme_record, load_split, load_truth_record


CL_TRUTH_FIELD = "cl_mL_min_kg"
VSS_TRUTH_FIELD = "vss_L_kg"
RAT_CL_UNITS = "mL.min-1.kg-1"
RAT_VSS_UNITS = "L.kg-1"
MICROSOME_CLINT_GROUP = ("human", "mL.min-1.g-1", "microsomes")

ALLOMETRY_CITATION = "Boxenbaum 1982, PMID 7120049; Ring et al. 2011, PMID 21541938"
IVIVE_CITATION = "Obach 1999, PMID 10534321; Ring et al. 2011, PMID 21541938"


@dataclass(frozen=True)
class _Feature:
    value: float
    references: tuple[str, ...]


@dataclass(frozen=True)
class ScalingModel:
    """A positive log-log model fitted without any held-out drug."""

    parameter: str
    method: str
    intercept: float
    exponent: float
    fit_drugs: tuple[str, ...]
    citation: str

    @property
    def coefficient(self) -> float:
        return math.exp(self.intercept)

    def predict(self, feature: float) -> float:
        if feature <= 0:
            raise ValueError("scaling features must be positive")
        return self.coefficient * feature ** self.exponent


@dataclass(frozen=True)
class Prediction:
    drug: str
    parameter: str
    method: str
    predicted: float
    observed: float
    fold_error: float
    input_references: tuple[str, ...]


@dataclass(frozen=True)
class Metric:
    parameter: str
    n: int
    geometric_mean_fold_error: float
    within_2_fold: float
    within_3_fold: float
    predictions: tuple[Prediction, ...]


def _summary_feature(record: dict, field: str, **match) -> _Feature | None:
    """Return the one summary row matching an explicit compatible group."""
    blob = record.get("fields", {}).get(field)
    if not blob:
        return None
    rows = [
        row for row in blob.get("summary", [])
        if row.get("value") is not None
        and all(row.get(key) == value for key, value in match.items())
    ]
    if len(rows) != 1:
        return None
    row = rows[0]
    references = sorted({
        measurement["ref"]
        for measurement in blob.get("measurements", [])
        if measurement.get("relation") == "="
        and measurement.get("value") is not None
        and all(measurement.get(key) == value for key, value in match.items())
    })
    return _Feature(float(row["value"]), tuple(references))


def _animal_feature(record: dict, parameter: str) -> _Feature | None:
    if parameter == "cl":
        return _summary_feature(
            record,
            "invivo_cl_preclinical",
            species="rat",
            units=RAT_CL_UNITS,
        )
    if parameter == "vss":
        return _summary_feature(
            record,
            "invivo_vss_preclinical",
            species="rat",
            units=RAT_VSS_UNITS,
        )
    raise ValueError(f"unknown parameter: {parameter}")


def _ivive_feature(record: dict) -> _Feature | None:
    clint = _summary_feature(
        record,
        "clint",
        species=MICROSOME_CLINT_GROUP[0],
        units=MICROSOME_CLINT_GROUP[1],
        matrix=MICROSOME_CLINT_GROUP[2],
    )
    fu = _summary_feature(record, "plasma_fu", species="human", units=None)
    if clint is None or fu is None:
        return None
    return _Feature(
        clint.value * fu.value,
        tuple(sorted(set(clint.references + fu.references))),
    )


def _human_truth(inn: str, parameter: str) -> tuple[float, tuple[str, ...]] | None:
    """Select IV truth, falling back to unspecified-route truth.

    This route preference is fixed before scoring and is the same for train
    and held-out drugs.  It prevents an IV prediction from being compared to
    a route-specific oral or depot quantity.
    """
    field = CL_TRUTH_FIELD if parameter == "cl" else VSS_TRUTH_FIELD
    blob = load_truth_record(inn).get("pk", {}).get("iv_compilations", {}).get(field)
    if not blob:
        return None
    for route in ("iv", "unspecified"):
        rows = [row for row in blob.get("summary", []) if row.get("route") == route]
        if rows and rows[0].get("value") is not None:
            refs = sorted({
                measurement["ref"]
                for measurement in blob.get("measurements", [])
                if measurement.get("relation") == "="
                and measurement.get("route") == route
            })
            return float(rows[0]["value"]), tuple(refs)
    return None


def _fit(parameter: str, drugs: Iterable[str], feature_getter) -> ScalingModel:
    rows = []
    for inn in drugs:
        feature = feature_getter(load_adme_record(inn))
        truth = _human_truth(inn, parameter)
        if feature is not None and truth is not None:
            rows.append((inn, feature.value, truth[0]))
    if len(rows) < 2:
        raise ValueError(f"{parameter} {len(rows)} train pairs; at least two are required")
    x = np.log([row[1] for row in rows])
    y = np.log([row[2] for row in rows])
    exponent, intercept = np.polyfit(x, y, 1)
    return ScalingModel(
        parameter=parameter,
        method="rat single-species allometric scaling",
        intercept=float(intercept),
        exponent=float(exponent),
        fit_drugs=tuple(row[0] for row in rows),
        citation=ALLOMETRY_CITATION,
    )


def fit_translation_models(train_drugs: Iterable[str] | None = None) -> dict[str, ScalingModel]:
    """Fit P29.2's models using only ``train_drugs``.

    The default is the frozen P28.1 training split.  Passing drug names is
    intentional: tests and later callers can prove exactly which records were
    allowed into a fit.
    """
    drugs = tuple(load_split()["train"] if train_drugs is None else train_drugs)
    models = {
        "cl_allometric": _fit("cl", drugs, lambda record: _animal_feature(record, "cl")),
        "vss_allometric": _fit("vss", drugs, lambda record: _animal_feature(record, "vss")),
    }
    models["cl_ivive"] = _fit("cl", drugs, _ivive_feature)
    models["cl_ivive"] = ScalingModel(
        parameter="cl",
        method="human microsome IVIVE with human plasma fu",
        intercept=models["cl_ivive"].intercept,
        exponent=models["cl_ivive"].exponent,
        fit_drugs=models["cl_ivive"].fit_drugs,
        citation=IVIVE_CITATION,
    )
    return models


def _prediction_for(inn: str, parameter: str, models: dict[str, ScalingModel]) -> Prediction | None:
    record = load_adme_record(inn)
    if parameter == "cl":
        feature = _animal_feature(record, "cl")
        model = models["cl_allometric"] if feature is not None else None
        if model is None:
            feature = _ivive_feature(record)
            model = models["cl_ivive"] if feature is not None else None
    elif parameter == "vss":
        feature = _animal_feature(record, "vss")
        model = models["vss_allometric"] if feature is not None else None
    else:
        raise ValueError(f"unknown parameter: {parameter}")
    truth = _human_truth(inn, parameter)
    if feature is None or model is None or truth is None:
        return None
    predicted = model.predict(feature.value)
    observed, truth_references = truth
    fold_error = max(predicted / observed, observed / predicted)
    return Prediction(
        drug=inn,
        parameter=parameter,
        method=model.method,
        predicted=predicted,
        observed=observed,
        fold_error=fold_error,
        input_references=tuple(sorted(set(feature.references + truth_references))),
    )


def _metric(parameter: str, predictions: list[Prediction]) -> Metric:
    if not predictions:
        raise ValueError(f"no held-out {parameter} predictions")
    folds = [prediction.fold_error for prediction in predictions]
    return Metric(
        parameter=parameter,
        n=len(predictions),
        geometric_mean_fold_error=math.exp(sum(math.log(x) for x in folds) / len(folds)),
        within_2_fold=sum(x <= 2.0 for x in folds) / len(folds),
        within_3_fold=sum(x <= 3.0 for x in folds) / len(folds),
        predictions=tuple(predictions),
    )


def score_heldout() -> dict[str, Metric]:
    """Fit on P28.1 train, then score only the frozen held-out drugs."""
    models = fit_translation_models()
    heldout = load_split()["heldout"]
    return {
        parameter: _metric(
            parameter,
            [
                prediction
                for inn in heldout
                for prediction in (_prediction_for(inn, parameter, models),)
                if prediction is not None
            ],
        )
        for parameter in ("cl", "vss")
    }


__all__ = [
    "Metric",
    "Prediction",
    "ScalingModel",
    "fit_translation_models",
    "score_heldout",
]
