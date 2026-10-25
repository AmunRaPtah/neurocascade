"""Per-stage v2 benchmark and baseline scorer (P33.3).

This module does not manufacture a chain prediction.  It records the honest
baseline performance and the data state needed by ``predict_compound``.  A
host checkout containing held-out truth scores it with exactly the same
functions; a conductor checkout reports every missing held-out drug instead
of treating it as a negative or as a missing-zero measurement.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from . import (
    load_v2_binding,
    load_v2_drugs,
    load_v2_human_pk,
    load_v2_liabilities,
    load_v2_plasma_fu,
    v2_partition,
)


LIABILITIES = ("eps", "sedation", "weight_gain", "anticholinergic", "orthostatic_hypotension")
SCORED_LIABILITIES = ("sedation", "anticholinergic")
UNDERPOWERED_LIABILITIES = {
    "eps": "six positives in the v2 universe",
    "orthostatic_hypotension": "three positives in the v2 universe",
}
WITHHELD_LIABILITIES = {"weight_gain": "P33.2 withheld this label; no binary truth exists"}

# This is an analysis reproducibility setting, not a biological parameter.
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_REPLICATES = 2000

# Affinity transform and the inactivity floor used by the 1-NN fingerprint
# baseline.  A target a drug was searched for but has no usable Ki at is not
# a missing value: it is weak binding, so it is imputed to the weakest
# affinity present anywhere in the v2 panel (Ki = 100 uM, a standard receptor-
# screening inactivity concentration).  Imputing 0 on the -log10(nM) scale
# would instead assert Ki = 1 nM (potent) at every unmeasured target, which is
# the opposite of what the data say.  These are analysis conventions, not
# calibrated neurocascade parameters.
INACTIVE_KI_NM = 100_000.0
FINGERPRINT_FLOOR = -math.log10(INACTIVE_KI_NM)  # -5.0


def _all_targets(binding: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """The full receptor panel: every target named by any v2 binding row.

    Both measured targets and searched-and-empty targets count, so the panel
    is identical for every drug and does not depend on which rows happen to
    carry a measurement.
    """
    seen: set[str] = set()
    for panel in binding.values():
        seen.update(panel.get("targets", {}))
        seen.update(panel.get("searched_and_empty", []) or [])
        seen.update(panel.get("kd_only", {}))
    return tuple(sorted(seen))


def gmfe(predicted: Iterable[float], observed: Iterable[float]) -> float:
    """Geometric mean absolute fold error for positive quantities."""
    pairs = list(zip(predicted, observed))
    if not pairs:
        return math.nan
    if any(p <= 0 or o <= 0 for p, o in pairs):
        raise ValueError("GMFE requires strictly positive predictions and observations")
    return math.exp(mean(abs(math.log(p / o)) for p, o in pairs))


def fold_errors(predicted: Iterable[float], observed: Iterable[float]) -> list[float]:
    """Absolute multiplicative fold errors, one per scored row."""
    pairs = list(zip(predicted, observed))
    if any(p <= 0 or o <= 0 for p, o in pairs):
        raise ValueError("fold error requires strictly positive values")
    return [max(p / o, o / p) for p, o in pairs]


def _rank_average(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def auroc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    """Rank-based AUROC, returning ``None`` when one class is absent."""
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if len(y) != len(s):
        raise ValueError("labels and scores must have equal length")
    positives = y == 1
    negatives = y == 0
    if not positives.any() or not negatives.any():
        return None
    ranks = _rank_average(s)
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    """Average precision with tied scores receiving their block precision."""
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if len(y) != len(s):
        raise ValueError("labels and scores must have equal length")
    n_positive = int((y == 1).sum())
    if not n_positive:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    total = 0.0
    cumulative = 0
    i = 0
    while i < len(y_sorted):
        j = i + 1
        while j < len(y_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        block_positive = int(y_sorted[i:j].sum())
        cumulative += block_positive
        if block_positive:
            total += block_positive * float(cumulative) / float(j)
        i = j
    return total / n_positive


def bootstrap_ci(
    labels: Iterable[int],
    scores: Iterable[float],
    metric: Callable[[Iterable[int], Iterable[float]], float | None],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float] | None:
    """Percentile 95% bootstrap CI, omitting resamples without both classes."""
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if len(y) != len(s) or len(y) == 0 or len(set(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        indices = rng.integers(0, len(y), len(y))
        value = metric(y[indices], s[indices])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def _metric_row(labels: list[int], scores: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": len(labels),
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
    }
    result["auroc_ci95"] = bootstrap_ci(labels, scores, auroc)
    result["auprc_ci95"] = bootstrap_ci(labels, scores, auprc, seed=BOOTSTRAP_SEED + 1)
    return result


def _value(row: Mapping[str, Any], key: str) -> float | None:
    item = row.get(key)
    if not isinstance(item, Mapping) or item.get("value") is None:
        return None
    return float(item["value"])


def _continuous_stage(
    field: str,
    label: str,
    train: set[str],
    heldout: set[str],
    truth: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    train_values = {inn: _value(truth[inn], field) for inn in train if inn in truth}
    train_values = {inn: value for inn, value in train_values.items() if value is not None}
    prediction = mean(train_values.values()) if train_values else None
    heldout_values = {inn: _value(truth[inn], field) for inn in heldout if inn in truth}
    heldout_values = {inn: value for inn, value in heldout_values.items() if value is not None}
    missing = sorted(heldout - set(heldout_values))

    # A leave-one-out train result is the largest honest local result.  It is
    # also a useful regression test for the metric implementation.
    loo_pred, loo_obs = [], []
    for inn, observed in sorted(train_values.items()):
        peers = [value for other, value in train_values.items() if other != inn]
        if peers:
            loo_pred.append(mean(peers))
            loo_obs.append(observed)
    available = {
        "status": "scored" if loo_obs else "searched-and-empty",
        "n": len(loo_obs),
        "gmfe": gmfe(loo_pred, loo_obs) if loo_obs else None,
        "fold_error_median": float(np.median(fold_errors(loo_pred, loo_obs))) if loo_obs else None,
    }
    heldout_result = {
        "status": "scored" if heldout_values else "searched-and-empty",
        "n": len(heldout_values),
        "gmfe": gmfe([prediction] * len(heldout_values), list(heldout_values.values())) if heldout_values else None,
        "fold_error_median": (
            float(np.median(fold_errors([prediction] * len(heldout_values), list(heldout_values.values()))))
            if heldout_values else None
        ),
        "missing_truth": missing,
    }
    return {
        "stage": label,
        "prediction": "train arithmetic mean",
        "baseline": {"name": "train_mean", "value": prediction, "unit": field},
        "available_data": available,
        "heldout": heldout_result,
    }


def _binding_score(panel: Mapping[str, Any], targets: tuple[str, ...]) -> float | None:
    values = []
    for target in targets:
        item = panel.get("targets", {}).get(target)
        if item and item.get("ki_nM") is not None and float(item["ki_nM"]) > 0:
            values.append(-math.log10(float(item["ki_nM"])))
    return max(values) if values else None


def _fingerprint(panel: Mapping[str, Any], targets: tuple[str, ...]) -> np.ndarray | None:
    """Full-panel receptor-affinity vector for the 1-NN baseline.

    Each coordinate is ``-log10(Ki_nM)`` where a measurement exists and the
    inactivity floor otherwise.  A drug with no measured affinity anywhere is
    refused (returns ``None``) rather than imputed to an all-floor vector,
    because its nearest neighbour would be an artefact of the floor.
    """
    values = []
    have = False
    for target in targets:
        item = panel.get("targets", {}).get(target)
        if item and item.get("ki_nM") is not None and float(item["ki_nM"]) > 0:
            values.append(-math.log10(float(item["ki_nM"])))
            have = True
        else:
            values.append(FINGERPRINT_FLOOR)
    return np.asarray(values, dtype=float) if have else None


def _atc_class_prevalence(
    inn: str,
    labels: Mapping[str, int],
    drugs: Mapping[str, Mapping[str, Any]],
    train: set[str],
) -> float | None:
    classes = {code[:3] for code in drugs[inn].get("atc", [])}
    prevalences = []
    for code in sorted(classes):
        members = [
            other for other in train
            if code in {x[:3] for x in drugs[other].get("atc", [])}
            and other in labels
        ]
        if members:
            prevalences.append(mean(labels[other] for other in members))
    return mean(prevalences) if prevalences else None


def _liability_stage(
    effect: str,
    train: set[str],
    heldout: set[str],
    truth: Mapping[str, Mapping[str, Any]],
    binding: Mapping[str, Mapping[str, Any]],
    drugs: Mapping[str, Mapping[str, Any]],
    fp_targets: tuple[str, ...],
) -> dict[str, Any]:
    if effect in WITHHELD_LIABILITIES:
        return {"stage": effect, "status": "unavailable", "reason": WITHHELD_LIABILITIES[effect]}
    if effect in UNDERPOWERED_LIABILITIES:
        return {"stage": effect, "status": "underpowered", "reason": UNDERPOWERED_LIABILITIES[effect]}

    targets = ("HRH1",) if effect == "sedation" else ("CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5")
    # Sorted so the row order fed to the bootstrap is deterministic: a set
    # union iterates in a hash-randomised order, which would make the CIs
    # differ between the session and the host even with a fixed seed.
    labels = {
        inn: int(truth[inn][effect]["label"])
        for inn in sorted(train | heldout)
        if inn in truth and truth[inn].get(effect, {}).get("label") is not None
    }
    train_labels = {inn: label for inn, label in labels.items() if inn in train}
    held_labels = {inn: label for inn, label in labels.items() if inn in heldout}

    def scores_for(inns: Iterable[str], leave_out: str | None = None) -> dict[str, dict[str, float]]:
        fit = {inn: label for inn, label in train_labels.items() if inn != leave_out}
        fit_inns = sorted(fit)
        out: dict[str, dict[str, float]] = {}
        vectors = {inn: _fingerprint(binding.get(inn, {}), fp_targets) for inn in fit_inns}
        for inn in inns:
            row: dict[str, float] = {}
            strongest = _binding_score(binding.get(inn, {}), targets)
            if strongest is not None:
                row["strongest_target_ki"] = strongest
            vector = _fingerprint(binding.get(inn, {}), fp_targets)
            if vector is not None and vectors:
                nearest = min(
                    (
                        (float(np.linalg.norm(vector - vectors[other])), other)
                        for other in fit_inns
                        if vectors[other] is not None
                    ),
                    key=lambda pair: (pair[0], pair[1]),
                )[1]
                row["one_nn_fingerprint"] = float(fit[nearest])
            prevalence = _atc_class_prevalence(inn, fit, drugs, set(fit))
            if prevalence is not None:
                row["atc_class_prevalence"] = prevalence
            if fit:
                row["train_prevalence"] = mean(fit.values())
            out[inn] = row
        return out

    def score_rows(
        rows: Mapping[str, int],
        prediction_rows: Mapping[str, Mapping[str, float]],
    ) -> dict[str, Any]:
        baselines = {}
        for name in ("strongest_target_ki", "one_nn_fingerprint", "atc_class_prevalence", "train_prevalence"):
            pairs = [
                (label, prediction_rows[inn][name])
                for inn, label in rows.items()
                if name in prediction_rows.get(inn, {})
            ]
            baselines[name] = (
                _metric_row([label for label, _ in pairs], [score for _, score in pairs])
                if pairs else {"n": 0, "status": "no data"}
            )
        return baselines

    loo_rows = {}
    for inn in sorted(train_labels):
        scored = scores_for([inn], leave_out=inn)
        loo_rows[inn] = scored[inn]
    held_rows = scores_for(sorted(held_labels))
    available = score_rows(train_labels, loo_rows)
    held = score_rows(held_labels, held_rows)
    missing = sorted(heldout - set(held_labels))
    return {
        "stage": effect,
        "status": "scored",
        "target_set": list(targets),
        "available_data": {"status": "scored", "baselines": available},
        "heldout": {
            "status": "scored" if held_labels else "searched-and-empty",
            "baselines": held,
            "missing_truth": missing,
            "n_truth": len(held_labels),
        },
    }


def run_chain_benchmark() -> dict[str, Any]:
    """Score every P33.3 stage without opening a held-out split file."""
    drugs = {row["inn"]: row for row in load_v2_drugs()}
    train, heldout = v2_partition()
    train_set, heldout_set = set(train), set(heldout)
    pk = load_v2_human_pk()
    fu = load_v2_plasma_fu()
    binding = load_v2_binding()
    liabilities = load_v2_liabilities()
    fp_targets = _all_targets(binding)

    stages = {
        "human_cl": _continuous_stage("clearance_iv", "human CL", train_set, heldout_set, pk),
        "vss": _continuous_stage("vss", "Vss", train_set, heldout_set, pk),
        "f": {
            "stage": "oral F",
            "status": "no data",
            "reason": "v2 has no oral absolute-bioavailability truth",
        },
        "free_plasma_exposure": {
            "stage": "free plasma exposure",
            "status": "no data",
            "reason": "therapeutic Cmax is absent; fu alone is not an exposure",
            "fu_input_rows": sum(row.get("fu") is not None for row in fu.values()),
        },
        "kp_uu": {
            "stage": "Kp,uu",
            "status": "no data",
            "reason": "v2 has no human Kp,uu truth",
        },
        "multi_target_occupancy": {
            "stage": "multi-target occupancy",
            "status": "no data",
            "reason": "therapeutic Cmax and Kp,uu are both absent",
        },
    }
    for effect in LIABILITIES:
        stages[effect] = _liability_stage(
            effect, train_set, heldout_set, liabilities, binding, drugs, fp_targets
        )

    train_counts = {
        "human_cl": stages["human_cl"]["available_data"]["n"],
        "vss": stages["vss"]["available_data"]["n"],
    }
    train_counts.update({
        effect: stages[effect].get("available_data", {}).get("baselines", {})
        .get("train_prevalence", {}).get("n", 0)
        for effect in SCORED_LIABILITIES
    })
    n_scored = max(train_counts.values(), default=0)
    return {
        "task": "P33.3",
        "status": "complete-with-host-pending-heldout" if n_scored else "searched-and-empty",
        "partition": {"n_train": len(train), "n_heldout": len(heldout)},
        "stages": stages,
        "n_scored": n_scored,
        "available_data_n_scored": n_scored,
        "heldout_truth_missing": sorted(heldout_set - set(pk) - set(liabilities)),
        "notes": [
            "No predict_compound implementation exists yet; this report freezes "
            "the stage baselines and scoring seam for P32.1.",
            "Held-out rows are scored when the host restores them; missing rows "
            "are searched-and-empty, never negatives.",
            "F, free plasma exposure, Kp,uu and multi-target occupancy are no data, not zero-error stages.",
        ],
    }


def heldout_summary_lines(report: Mapping[str, Any]) -> list[str]:
    """One compact line per stage for the host_eval pytest tail."""
    lines = []
    for name, stage in report["stages"].items():
        held = stage.get("heldout")
        if held is None:
            lines.append(f"{name}: {stage['status']}")
        elif "gmfe" in held:
            gmfe_text = "n/a" if held["gmfe"] is None else f"{held['gmfe']:.3f}"
            lines.append(f"{name}: {held['status']} n={held['n']} train-mean GMFE={gmfe_text}")
        else:
            parts = []
            for baseline, row in held["baselines"].items():
                if row.get("auroc") is None:
                    parts.append(f"{baseline} n={row.get('n', 0)} no data")
                else:
                    lo, hi = row["auroc_ci95"]
                    parts.append(
                        f"{baseline} n={row['n']} AUROC={row['auroc']:.3f} [{lo:.3f},{hi:.3f}] "
                        f"AUPRC={row['auprc']:.3f}"
                    )
            lines.append(f"{name}: {held['status']} n_truth={held['n_truth']}; " + "; ".join(parts))
    return lines


__all__ = [
    "LIABILITIES",
    "SCORED_LIABILITIES",
    "auprc",
    "auroc",
    "bootstrap_ci",
    "fold_errors",
    "gmfe",
    "heldout_summary_lines",
    "run_chain_benchmark",
]
