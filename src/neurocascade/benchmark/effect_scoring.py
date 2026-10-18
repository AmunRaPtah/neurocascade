"""P30.2 effect scoring and baseline comparisons.

The P30.1 model is frozen on the training split.  This module only scores its
predictions and two deliberately simple baselines:

* ``strongest_target_ki`` is the negative log10 of the smallest reported
  human Ki in a drug's panel.  Missing targets are not treated as inactive.
* ``nearest_neighbor`` compares only targets measured for both drugs, using
  Euclidean distance in log10(Ki), and returns the nearest eligible training
  drug's label.

All confidence intervals are drug-level percentile bootstrap intervals.  The
replicate count and seed are analysis settings, not biological parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable, Mapping

import numpy as np

from . import load_binding_panel, load_split
from .effects import (
    EFFECTS,
    ILLUSTRATIVE_L2_STRENGTH,
    UNDERPOWERED_MIN_CLASS,
    _auc,
    _fit_binary,
    effect_labels,
    fit_effect_models,
    occupancy_profile,
)


BASELINES = ("strongest_target_ki", "nearest_neighbor")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261018


@dataclass(frozen=True)
class BaselineScore:
    score: float
    neighbor: str | None = None
    shared_targets: int | None = None
    distance: float | None = None


def _auprc(labels: list[int], scores: list[float]) -> float | None:
    """Average precision, with scores ordered from high to low."""
    positives = sum(label == 1 for label in labels)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    found = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            found += 1
            total += found / rank
    return total / positives


def _bootstrap_interval(
    labels: list[int],
    scores: list[float],
    statistic,
    *,
    seed: int,
    replicates: int,
) -> list[float] | None:
    if len(labels) < 2:
        return None
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        value = statistic(
            labels_array[indices].tolist(), scores_array[indices].tolist()
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    # Percentile intervals are the pre-declared reporting rule.  The 2.5 and
    # 97.5 percentiles are analysis settings, not model parameters.
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def metric_summary(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Return AUROC, AUPRC, prevalence and drug-level bootstrap CIs."""
    labels = [int(value) for value in labels]
    scores = [float(value) for value in scores]
    positive = sum(value == 1 for value in labels)
    negative = sum(value == 0 for value in labels)
    auroc = _auc(labels, scores)
    auprc = _auprc(labels, scores)
    return {
        "n_scored": len(labels),
        "n_positive": positive,
        "n_negative": negative,
        "underpowered": min(positive, negative) < UNDERPOWERED_MIN_CLASS,
        "headline": min(positive, negative) >= UNDERPOWERED_MIN_CLASS,
        "auroc": auroc,
        "auroc_ci": _bootstrap_interval(
            labels, scores, _auc, seed=seed, replicates=replicates
        ) if auroc is not None else None,
        "auprc": auprc,
        "auprc_ci": _bootstrap_interval(
            labels, scores, _auprc, seed=seed + 1, replicates=replicates
        ) if auprc is not None else None,
        "prevalence_baseline": (
            positive / len(labels) if labels else None
        ),
        "bootstrap": {
            "kind": "drug-level percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "status": "illustrative analysis settings, not biological parameters",
        },
    }


@lru_cache(maxsize=None)
def _fingerprint(drug: str) -> dict[str, float]:
    panel = load_binding_panel(drug)
    result = {}
    for target, record in panel.get("targets", {}).items():
        ki = record.get("ki_nM")
        if ki is not None and float(ki) > 0:
            result[target] = math.log10(float(ki))
    return result


def strongest_target_ki(drug: str) -> BaselineScore | None:
    """Return the strongest reported target as a monotone Ki score."""
    fingerprint = _fingerprint(drug)
    if not fingerprint:
        return None
    target, log_ki = min(fingerprint.items(), key=lambda item: (item[1], item[0]))
    return BaselineScore(score=-log_ki, neighbor=target)


def nearest_neighbor(
    drug: str,
    candidates: Iterable[str],
) -> BaselineScore | None:
    """Find a nearest binding fingerprint without inventing missing Ki values."""
    query = _fingerprint(drug)
    if not query:
        return None
    best = None
    for candidate in sorted(set(candidates)):
        if candidate == drug:
            continue
        fingerprint = _fingerprint(candidate)
        shared = sorted(set(query) & set(fingerprint))
        if not shared:
            continue
        distance = math.sqrt(
            sum((query[target] - fingerprint[target]) ** 2 for target in shared)
            / len(shared)
        )
        key = (distance, -len(shared), candidate)
        if best is None or key < best[0]:
            best = (key, candidate, len(shared), distance)
    if best is None:
        return None
    _, candidate, shared_count, distance = best
    return BaselineScore(
        score=0.0, neighbor=candidate,
        shared_targets=shared_count, distance=distance,
    )


@lru_cache(maxsize=None)
def _available_rows(drugs: tuple[str, ...]) -> tuple[tuple[str, object, dict[str, int | None]], ...]:
    """Load profiles and labels, refusing unavailable truth or inputs."""
    train = tuple(load_split()["train"])
    rows = []
    for drug in drugs:
        try:
            profile = occupancy_profile(drug, train_drugs=train)
            labels = {row.effect: row.label for row in effect_labels(drug)}
        except (KeyError, ValueError):
            continue
        if any(label in (0, 1) for label in labels.values()):
            rows.append((drug, profile, labels))
    return tuple(rows)


def _profiles_for_split(drugs: Iterable[str]) -> tuple[tuple[str, object, dict[str, int | None]], ...]:
    """Use P30.1's exact train-only exposure seam for a scoring split."""
    return _available_rows(tuple(drugs))


def _comparison(
    effect: str,
    predictions: Mapping[str, list[tuple[int, float]]],
    *,
    seed: int,
    replicates: int,
) -> dict:
    summaries = {
        method: metric_summary(
            [label for label, _ in values],
            [score for _, score in values],
            seed=seed + index * 10,
            replicates=replicates,
        )
        for index, (method, values) in enumerate(predictions.items())
    }
    model = summaries["occupancy_model"]
    comparisons = {}
    for baseline in BASELINES:
        baseline_summary = summaries[baseline]
        comparisons[baseline] = {
            "auroc_delta_model_minus_baseline": (
                model["auroc"] - baseline_summary["auroc"]
                if model["auroc"] is not None and baseline_summary["auroc"] is not None
                else None
            ),
            "auprc_delta_model_minus_baseline": (
                model["auprc"] - baseline_summary["auprc"]
                if model["auprc"] is not None and baseline_summary["auprc"] is not None
                else None
            ),
        }
        comparisons[baseline]["model_beats_auroc"] = (
            comparisons[baseline]["auroc_delta_model_minus_baseline"] > 0
            if comparisons[baseline]["auroc_delta_model_minus_baseline"] is not None
            else None
        )
        comparisons[baseline]["model_beats_auprc"] = (
            comparisons[baseline]["auprc_delta_model_minus_baseline"] > 0
            if comparisons[baseline]["auprc_delta_model_minus_baseline"] is not None
            else None
        )
    return {"effect": effect, "methods": summaries, "comparisons": comparisons}


def _strongest_scores(rows):
    return {drug: strongest_target_ki(drug) for drug, _, _ in rows}


def _lodo_all_drugs(
    rows,
    *,
    regularization_strength: float,
    seed: int,
    replicates: int,
) -> dict:
    results = {}
    strongest = _strongest_scores(rows)
    for effect in EFFECTS:
        usable = [row for row in rows if row[2].get(effect) in (0, 1)]
        predictions = {"occupancy_model": [], "strongest_target_ki": [], "nearest_neighbor": []}
        for left_out, profile, labels in usable:
            fit_rows = [row for row in usable if row[0] != left_out]
            targets = tuple(sorted({target for _, p, _ in rows for target in p.features}))
            x = np.asarray(
                [[p.features.get(target, 0.0) for target in targets] for _, p, _ in fit_rows],
                dtype=float,
            )
            y = np.asarray([lab[effect] for _, _, lab in fit_rows], dtype=float)
            model = _fit_binary(effect, targets, x, y, regularization_strength)
            label = int(labels[effect])
            ki_score = strongest.get(left_out)
            candidates = [drug for drug, _, _ in fit_rows]
            neighbor = nearest_neighbor(left_out, candidates)
            # Compare all three methods on the same drug set.  A missing Ki
            # or a non-overlapping fingerprint is an input gap, not a value
            # to be filled with an assumed inactive target.
            if ki_score is None or neighbor is None:
                continue
            neighbor_label = next(row[2][effect] for row in fit_rows if row[0] == neighbor.neighbor)
            predictions["occupancy_model"].append((label, model.predict_proba(profile.features)))
            predictions["strongest_target_ki"].append((label, ki_score.score))
            predictions["nearest_neighbor"].append((label, float(neighbor_label)))
        results[effect] = _comparison(
            effect, predictions, seed=seed, replicates=replicates
        )
        results[effect]["n_lodo_drugs"] = len(predictions["occupancy_model"])
    return results


def _fixed_train_heldout(
    heldout_rows,
    train_rows,
    *,
    regularization_strength: float,
    seed: int,
    replicates: int,
) -> dict:
    models, _ = fit_effect_models(regularization_strength=regularization_strength)
    strongest = _strongest_scores(heldout_rows)
    results = {}
    for effect in EFFECTS:
        predictions = {"occupancy_model": [], "strongest_target_ki": [], "nearest_neighbor": []}
        for drug, profile, labels in heldout_rows:
            label = labels.get(effect)
            if label not in (0, 1) or effect not in models:
                continue
            label = int(label)
            ki_score = strongest.get(drug)
            neighbor = nearest_neighbor(drug, [candidate for candidate, _, candidate_labels in train_rows
                                                if candidate_labels.get(effect) in (0, 1)])
            if ki_score is None or neighbor is None:
                continue
            neighbor_label = next(row[2][effect] for row in train_rows if row[0] == neighbor.neighbor)
            predictions["occupancy_model"].append((label, models[effect].predict_proba(profile.features)))
            predictions["strongest_target_ki"].append((label, ki_score.score))
            predictions["nearest_neighbor"].append((label, float(neighbor_label)))
        results[effect] = _comparison(
            effect, predictions, seed=seed, replicates=replicates
        )
    return results


def _failure_analysis(section: Mapping[str, dict]) -> dict:
    excluded = []
    cannot_predict = []
    does_not_beat_both = []
    for effect, result in section.get("effects", {}).items():
        model = result["methods"]["occupancy_model"]
        if model["underpowered"]:
            excluded.append(effect)
        if model["auroc"] is None:
            cannot_predict.append(effect)
        if not any(
            result["comparisons"][baseline]["model_beats_auroc"] is True
            for baseline in BASELINES
        ):
            does_not_beat_both.append(effect)
    return {
        "underpowered_excluded_from_headline": excluded,
        "effects_not_estimable": cannot_predict,
        "occupancy_model_does_not_beat_both_baselines": does_not_beat_both,
    }


def score_p30_2(
    *,
    regularization_strength: float = ILLUSTRATIVE_L2_STRENGTH,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Score P30.1, both baselines, held-out truth, and all-available LODO."""
    split = load_split()
    train_rows = _profiles_for_split(split["train"])
    heldout_rows = _profiles_for_split(split["heldout"])
    lodo_rows = _profiles_for_split(tuple(split["train"]) + tuple(split["heldout"]))
    missing_truth = []
    for drug in split["heldout"]:
        try:
            effect_labels(drug)
        except (KeyError, ValueError):
            missing_truth.append(drug)
    available_heldout = {row[0] for row in heldout_rows}
    missing_input = [
        drug for drug in split["heldout"]
        if drug not in missing_truth and drug not in available_heldout
    ]
    lodo = _lodo_all_drugs(
        lodo_rows,
        regularization_strength=regularization_strength,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    if not heldout_rows:
        heldout = {
            "status": "searched-and-empty",
            "n_scored": 0,
            "n_heldout": len(split["heldout"]),
            "missing_truth": missing_truth,
            "missing_input": missing_input,
            "effects": {},
        }
    else:
        heldout = {
            "status": "scored",
            "n_scored": len(heldout_rows),
            "n_heldout": len(split["heldout"]),
            "missing_truth": missing_truth,
            "missing_input": missing_input,
            "effects": _fixed_train_heldout(
                heldout_rows, train_rows,
                regularization_strength=regularization_strength,
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            ),
        }
    heldout["failure_analysis"] = _failure_analysis(heldout)
    if not heldout_rows:
        heldout["failure_analysis"]["effects_blocked_by_missing_heldout_truth"] = list(EFFECTS)
    return {
        "task": "P30.2",
        "status": heldout["status"],
        "n_scored": heldout["n_scored"],
        "n_heldout": len(split["heldout"]),
        "scored_unit": "held-out drugs with P30.1 truth and a unique exact oral dose",
        "fit_split": "train",
        "methods": {
            "occupancy_model": "frozen P30.1 train-only L2 logistic model",
            "strongest_target_ki": "negative log10 of the smallest reported human Ki",
            "nearest_neighbor": "1-nearest binding fingerprint on shared log10(Ki) targets",
        },
        "bootstrap": {
            "kind": "drug-level percentile bootstrap",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "citation": "Efron 1979, doi:10.1214/aos/1176344552",
            "status": "illustrative analysis settings, not biological parameters",
        },
        "heldout": heldout,
        "lodo_all_available_drugs": {
            "n_drugs": len(lodo_rows),
            "effects": lodo,
            "failure_analysis": _failure_analysis({"effects": lodo}),
        },
    }
