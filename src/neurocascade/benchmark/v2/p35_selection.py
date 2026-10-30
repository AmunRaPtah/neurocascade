"""P35.3: select the Phase 35 candidate on SIDER train leave-one-drug-out.

Under ``docs/prereg-p35-go.md``.  Reads the P35.2 **train** labels
(``sider_labels.load_train_labels``), the P35.2 train universe, the v2 binding
input and the v2 Cmax truth, and scores every candidate family evaluated in
P35.3 beside the four prereg baselines (strongest-target Ki, 1-NN fingerprint,
ATC-class prevalence, occupancy_max) on the identical common set per liability.
Every INN it touches is asserted P35-train (``not p35_heldout``); no P35
held-out label exists in the tree, and nothing here could build one.

Scored set: P35-train universe members (``in_universe is True``) with a 0/1
label and a non-null score for every baseline, the go_verdict common-set rule.
Fit pool for the fitted methods (1-NN, ATC, kNN, logistic): every P35-train
drug with a SIDER record (and, for fingerprint methods, a measured Ki
somewhere in the panel), minus the drug being scored.

The selected family is frozen by ``scripts/p35_3_freeze.py`` into
``frozen/p35_candidate.json`` and scored through ``p35_candidate.py``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
from scipy.stats import rankdata

from . import load_v2_binding, load_v2_drugs, load_v2_human_cmax
from .candidate import LIABILITY_RECEPTORS, liability_predict, occupancy_baseline
from .candidate_benchmark import _cu, _ki_by_gene
from .chain_benchmark import (
    FINGERPRINT_FLOOR,
    INACTIVE_KI_NM,
    _all_targets,
    _atc_class_prevalence,
    _binding_score,
    auprc,
    auroc,
)
from .frozen_candidate import predict_cohort as p34_predict_cohort
from .p35_candidate import fingerprint, knn_fraction, predict_cohort
from .sider_labels import LIABILITY_PTS, load_train_labels, load_universe, p35_heldout

LIABILITIES = tuple(LIABILITY_PTS)
BASELINES = ("occupancy_max", "strongest_target_ki", "one_nn_fingerprint", "atc_class_prevalence")
# The prereg's power floor: a liability is GO-eligible with >= 8 positives and
# >= 8 negatives in the common set.  Selection counts only such liabilities.
MIN_CLASS = 8
# kNN sizes and logistic L2 evaluated.  Analysis settings, not biological
# parameters; every value tried is listed here and reported.
KNN_K = (5, 10)
LOGIT_L2 = 1.0
SELECTED = "rank_occupancy_knn10"
SELECTED_K = 10

# Every family evaluated in P35.3, in the order it was tried.  None is a
# baseline; "rank_*" are within-cohort rank sums of the named views.
FAMILIES: dict[str, tuple[str, ...]] = {
    "knn5": ("knn5",),
    "knn10": ("knn10",),
    "logit_fingerprint": ("logit",),
    "occupancy_sum": ("occupancy_sum",),
    "rank_occupancy_affinity_p34": ("occupancy_max", "strongest_target_ki"),
    "rank_occupancy_knn5": ("occupancy_max", "knn5"),
    "rank_occupancy_affinity_knn5": ("occupancy_max", "strongest_target_ki", "knn5"),
    "rank_occupancy_knn10": ("occupancy_max", "knn10"),
    "rank_occupancy_sum_knn10": ("occupancy_sum", "knn10"),
    "rank_occupancy_sum_affinity": ("occupancy_sum", "strongest_target_ki"),
    "occupancy_sum_then_affinity": ("occupancy_sum", "strongest_target_ki"),  # lexicographic
}


def load_inputs() -> dict[str, Any]:
    """Train-only inputs.  Raises if any P35 held-out INN carries a label."""
    labels = load_train_labels()
    bad = sorted(inn for inn in labels if p35_heldout(inn))
    if bad:
        raise ValueError(f"P35 held-out drugs in the train labels: {bad}")
    universe = [r["inn"] for r in load_universe() if r["in_universe"] is True]
    assert not any(p35_heldout(inn) for inn in universe)
    return {
        "labels": labels,
        "universe": sorted(universe),
        "binding": load_v2_binding(),
        "cmax": load_v2_human_cmax(),
        "drugs": {row["inn"]: row for row in load_v2_drugs()},
    }


def build_params(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen candidate's params: the selected k, receptors and kNN pool."""
    binding, labels = inputs["binding"], inputs["labels"]
    panel = list(_all_targets(binding))
    pool = {}
    for inn in sorted(labels):
        lab = {e: labels[inn][e]["label"] for e in LIABILITIES}
        if "excluded" in lab.values():
            continue
        vec = fingerprint(_ki_by_gene(binding, inn), panel, FINGERPRINT_FLOOR)
        if vec is not None:
            pool[inn] = {"fingerprint": [round(x, 12) for x in vec], "labels": lab}
    return {
        "task": "P35.3",
        "family": SELECTED,
        "description": (
            "Within-cohort rank sum of occupancy_max = max_t Cu/(Cu+Ki_t) over each liability's "
            "pre-specified receptors and the fraction of SIDER-positive drugs among the k nearest "
            "P35-train drugs in full-panel -log10(Ki) space (Euclidean, ties by INN, a drug never "
            "its own neighbour). Equal weights."),
        "liability_receptors": {e: list(r) for e, r in LIABILITY_RECEPTORS.items()},
        "k": SELECTED_K,
        "view_weights": [1.0, 1.0],
        "panel": panel,
        "fingerprint_floor": FINGERPRINT_FLOOR,
        "fingerprint_floor_note": (
            f"-log10({INACTIVE_KI_NM:g} nM): chain_benchmark's inactivity convention for the 1-NN "
            "baseline, reused so candidate and baseline share one fingerprint"),
        "missing_view_rule": "no Cmax,u or no fingerprint: no score; never imputed",
        "fitted_on": ("P35-train drugs (sha256(inn) % 10 not in {0,1}) with a SIDER record and a "
                      "measured Ki in the panel; labels from sider/labels_train.json (P35.2)"),
        "n_pool": len(pool),
        "pool": pool,
    }


def _views(inn: str, effect: str, inputs: Mapping[str, Any], pool_params: Mapping[str, Any],
           record: list[str], fp: Mapping[str, Any]) -> dict[str, float | None]:
    binding, labels = inputs["binding"], inputs["labels"]
    rc = LIABILITY_RECEPTORS.get(effect)
    cu, ki = _cu(inputs["cmax"], inn), _ki_by_gene(binding, inn)
    y = {o: labels[o][effect]["label"] for o in record if o != inn}
    v: dict[str, float | None] = {}
    v["occupancy_max"] = occupancy_baseline(cu, ki, rc) if rc else None
    v["occupancy_sum"] = (sum(occupancy_baseline(cu, ki, (g,)) for g in rc)
                          if rc and cu is not None else None)
    v["strongest_target_ki"] = _binding_score(binding.get(inn, {}), rc) if rc else None
    v["atc_class_prevalence"] = _atc_class_prevalence(inn, y, inputs["drugs"], set(y))
    for k in (1, *KNN_K):
        v[f"knn{k}"] = knn_fraction(inn, fp[inn], effect, {**pool_params, "k": k})
    v["one_nn_fingerprint"] = v.pop("knn1")
    if fp[inn] is None:
        v["logit"] = None
    else:
        fit = [o for o in sorted(pool_params["pool"]) if o != inn]
        v["logit"] = liability_predict([pool_params["pool"][o]["fingerprint"] for o in fit],
                                       [y[o] for o in fit], [fp[inn]], l2=LOGIT_L2)[0]
    return v


def _family_scores(views: Mapping[str, Mapping[str, float | None]], common: list[str]
                   ) -> dict[str, list[float]]:
    out = {}
    for fam, parts in FAMILIES.items():
        cols = [[views[i][p] for i in common] for p in parts]
        if fam == "occupancy_sum_then_affinity":
            keys = list(zip(*cols))
            order = {key: n for n, key in enumerate(sorted(set(keys)))}
            out[fam] = [float(order[key]) for key in keys]
        elif len(parts) == 1:
            out[fam] = [float(x) for x in cols[0]]
        else:
            out[fam] = [float(x) for x in sum(rankdata(c) for c in cols)]
    return out


def score_liability(effect: str, inputs: Mapping[str, Any], pool_params: Mapping[str, Any],
                    frozen_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    labels = inputs["labels"]
    record = [i for i in sorted(labels) if labels[i][effect]["label"] in (0, 1)]
    fp = {i: fingerprint(_ki_by_gene(inputs["binding"], i), pool_params["panel"],
                         pool_params["fingerprint_floor"]) for i in inputs["universe"]}
    scored = [i for i in inputs["universe"] if labels[i][effect]["label"] in (0, 1)]
    views = {i: _views(i, effect, inputs, pool_params, record, fp) for i in scored}
    has_receptors = effect in LIABILITY_RECEPTORS
    base = BASELINES if has_receptors else ("one_nn_fingerprint", "atc_class_prevalence")
    common = [i for i in scored if all(views[i][b] is not None for b in base)]
    y = [int(labels[i][effect]["label"]) for i in common]
    row: dict[str, Any] = {"common_n": len(common), "n_positive": sum(y),
                           "n_negative": len(y) - sum(y), "common_set": common,
                           "go_eligible": min(sum(y), len(y) - sum(y)) >= MIN_CLASS}

    def metric(s: list[float]) -> dict[str, Any]:
        return {"auroc": auroc(y, s), "auprc": auprc(y, s)}

    row["baselines"] = {b: metric([views[i][b] for i in common]) for b in base}
    best = max(base, key=lambda b: row["baselines"][b]["auroc"] or -1.0)
    row["best_baseline"] = best
    if not has_receptors:
        row["families"] = {}
        row["candidate_status"] = "not scoreable: no pre-specified receptor for this liability"
        return row
    fams = _family_scores(views, common)
    row["families"] = {f: metric(s) for f, s in fams.items()}
    for f in row["families"].values():
        f["margin_vs_best_baseline"] = f["auroc"] - row["baselines"][best]["auroc"]
        f["beats_every_baseline"] = f["margin_vs_best_baseline"] > 0
    if frozen_params is not None:
        cand = predict_cohort(effect, common, [_cu(inputs["cmax"], i) for i in common],
                              [_ki_by_gene(inputs["binding"], i) for i in common], frozen_params)
        row["frozen_candidate"] = metric(cand)
        p34 = p34_predict_cohort(effect, [_cu(inputs["cmax"], i) for i in common],
                                 [_ki_by_gene(inputs["binding"], i) for i in common])
        row["p34_3_secondary"] = metric(p34)
        row["frozen_candidate_scores"] = dict(zip(common, cand))
        row["y"] = y
        row["baseline_scores"] = {b: [views[i][b] for i in common] for b in base}
    return row


def select(per_liability: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Selection rule, fixed before the freeze: the family that beats the best
    baseline on the most GO-eligible liabilities; ties go to the larger mean
    margin over those liabilities."""
    elig = [e for e, r in per_liability.items() if r["go_eligible"] and r["families"]]
    table = {}
    for fam in FAMILIES:
        margins = [per_liability[e]["families"][fam]["margin_vs_best_baseline"] for e in elig]
        table[fam] = {"wins": sum(m > 0 for m in margins),
                      "mean_margin": float(np.mean(margins)) if margins else math.nan}
    best = max(FAMILIES, key=lambda f: (table[f]["wins"], table[f]["mean_margin"]))
    return {"eligible_liabilities": elig, "table": table, "selected": best}


def run_p35_3(frozen_params: Mapping[str, Any] | None = None,
              liabilities: tuple[str, ...] = LIABILITIES,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    inputs = load_inputs()
    pool_params = build_params(inputs)
    per = {}
    for e in liabilities:
        if progress:
            progress(e)
        per[e] = score_liability(e, inputs, pool_params, frozen_params)
    return {"inputs": inputs, "pool_params": pool_params, "liabilities": per,
            "selection": select(per)}


__all__ = ["BASELINES", "FAMILIES", "LIABILITIES", "MIN_CLASS", "SELECTED", "build_params",
           "load_inputs", "run_p35_3", "score_liability", "select"]
