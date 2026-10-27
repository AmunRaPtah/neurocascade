"""The P34.1 candidate: a parameter-free rank ensemble of two mechanistic views.

Phase 33's candidate (``candidate.py``) was a multi-target occupancy -> L2
logistic.  P33.4 showed it loses to the unfitted ``occupancy_max`` baseline on
train leave-one-drug-out (sedation 0.571/0.446, anticholinergic 0.343/0.357 in
the committed ``results/P33.4.json``), and that the near-zero anticholinergic
LODO AUROC is a leave-one-out *intercept* artefact: with ~15 drugs and a
zero-inflated occupancy feature, each fold's refitted intercept re-orders the
many occupancy-0 drugs, manufacturing a spurious ranking (P34.1 diagnosis,
docs/architecture.md).

This module is the P34.1 candidate's *predict path*.  Like ``candidate.py`` it
imports no truth module and reads no file: it takes already-assembled numeric
mechanistic scores and returns a combined score.  The feature assembly (which
reads binding + Cmax truth, train rows only) lives in
``candidate_v2_benchmark.py``.

The candidate is the rank average of two mechanistic views of receptor
engagement at a liability's pre-specified receptors:

  * exposure-adjusted occupancy, ``occ_max = max_t Cu/(Cu+Ki_t)`` (the P33.3
    ``occupancy_max`` baseline), and
  * intrinsic affinity, ``aff = max_t -log10(Ki_t)`` (the P33.3
    ``strongest_target_ki`` baseline).

Each drug's two views are converted to average ranks *within the scored
cohort* and summed.  There is no free parameter: no intercept, no L2 strength,
no tuned weight, so there is nothing to overfit under LODO and no intercept to
jitter.  The mechanistic motivation is that occupancy *saturates*: when
Cu >> Ki every strong binder sits at occ ~ 1, so occupancy cannot rank the
strong binders against each other, while affinity still can.  Conversely
affinity ignores exposure entirely.  Combining the two by rank is robust to
each one's failure mode.  Equal weight is the parameter-free choice; a tuned
weight would be a fitted number and is deliberately not used.

The score uses no labels and is unsupervised, so (exactly like ``occupancy_max``
and ``strongest_target_ki``) it is scored on the whole cohort rather than by a
fold loop; there is no estimator to leave a drug out of.
"""

from __future__ import annotations

from typing import Sequence

from scipy.stats import rankdata


def rank_combined_score(
    occupancy_max: Sequence[float | None],
    affinity: Sequence[float | None],
) -> list[float | None]:
    """Parameter-free rank ensemble of exposure-adjusted occupancy and affinity.

    ``occupancy_max[i]`` and ``affinity[i]`` are the two mechanistic views for
    drug ``i`` over the liability's pre-specified receptors (higher = stronger
    engagement for both).  ``None`` in either view means the drug is
    unscoreable by the candidate (no exposure, or no measured affinity), and
    its combined score is ``None`` -- it is never imputed.

    For the drugs where both views are present, each view is converted to an
    average rank within that cohort (ties share the mean rank) and the two
    ranks are summed.  The sum is a monotone, label-free score; its absolute
    scale is irrelevant to AUROC/AUPRC, only its ordering.  No fit, no free
    parameter.
    """
    if len(occupancy_max) != len(affinity):
        raise ValueError("occupancy_max and affinity must be the same length")
    idx = [
        i
        for i in range(len(occupancy_max))
        if occupancy_max[i] is not None and affinity[i] is not None
    ]
    out: list[float | None] = [None] * len(occupancy_max)
    if not idx:
        return out
    occ_rank = rankdata([float(occupancy_max[i]) for i in idx])
    aff_rank = rankdata([float(affinity[i]) for i in idx])
    for k, i in enumerate(idx):
        out[i] = float(occ_rank[k] + aff_rank[k])
    return out


__all__ = ["rank_combined_score"]
