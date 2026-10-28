"""Train leave-one-drug-out scoring of the P34.1 candidate vs every baseline.

This is the harness for ``candidate_v2.rank_combined_score``.  It reads the v2
binding input and the Cmax / liability truth for the **train** INNs only
(``v2_partition()[0]``), assembles the two mechanistic views, and scores the
candidate beside the four frozen P33.3 baselines (occupancy_max,
strongest_target_ki, one_nn_fingerprint, atc_class_prevalence) on the identical
common set per liability -- the same intersection ``pipeline/go_verdict.py``
uses on held-out.

It never touches a held-out or fresh drug: every INN it scores comes from the
train side of ``v2_partition`` and is asserted not ``v2_heldout``.  The held-out
and fresh evaluations belong to later Phase 34 tasks (P34.3 freezes a candidate,
P34.4 builds a fresh set, P34.5 scores on the host); nothing here reads them.

The candidate's predict path lives in ``candidate_v2.py`` and reads no truth;
this module does the (allowed, train-only) truth reads and hands it numbers.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import (
    load_v2_binding,
    load_v2_drugs,
    load_v2_human_cmax,
    load_v2_liabilities,
    v2_heldout,
    v2_partition,
)
from .candidate import LIABILITY_RECEPTORS, occupancy_baseline
from .candidate_benchmark import _cu, _ki_by_gene
from .candidate_v2 import rank_combined_score
from .chain_benchmark import (
    _all_targets,
    _atc_class_prevalence,
    _binding_score,
    _fingerprint,
    auprc,
    auroc,
)

# The candidate scores the same liabilities P33.4 did; sedation and
# anticholinergic are the GO-registered scored set (the others are underpowered
# in the v2 universe).  No receptor set or label is introduced here.
CANDIDATE_LIABILITIES = ("sedation", "anticholinergic", "eps", "orthostatic_hypotension")
REGISTERED_SCORED = ("sedation", "anticholinergic")


def _metric(labels: list[int], scores: list[float]) -> dict[str, Any]:
    return {
        "n": len(labels),
        "n_positive": int(sum(labels)),
        "n_negative": len(labels) - int(sum(labels)),
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
    }


def score_liability(
    effect: str,
    train: list[str],
    binding: Mapping[str, Mapping[str, Any]],
    cmax: Mapping[str, Mapping[str, Any]],
    liabilities: Mapping[str, Mapping[str, Any]],
    drugs: Mapping[str, Mapping[str, Any]],
    fp_targets: tuple[str, ...],
) -> dict[str, Any]:
    """Candidate and baseline metrics for one liability, train drugs only.

    The common set is the train drugs with a label and a non-null score for
    every method, so every method is scored on identical rows.  The candidate
    and both unfitted baselines (occupancy_max, strongest_target_ki) are
    label-free and scored directly on the common set; the two fitted baselines
    (1-NN fingerprint, ATC-class prevalence) are leave-one-drug-out, exactly as
    in the P33.3 / P33.4 harness.
    """
    receptors = LIABILITY_RECEPTORS[effect]
    labelled = [
        inn
        for inn in sorted(train)
        if liabilities.get(inn, {}).get(effect, {}).get("label") is not None
    ]
    # Train-only invariant: nothing held-out ever enters the scored universe.
    assert all(not v2_heldout(inn) for inn in labelled)

    y_all = {inn: int(liabilities[inn][effect]["label"]) for inn in labelled}
    cu = {inn: _cu(cmax, inn) for inn in labelled}
    kis = {inn: _ki_by_gene(binding, inn) for inn in labelled}
    vectors = {inn: _fingerprint(binding.get(inn, {}), fp_targets) for inn in labelled}

    occ = {inn: occupancy_baseline(cu[inn], kis[inn], receptors) for inn in labelled}
    aff = {inn: _binding_score(binding.get(inn, {}), receptors) for inn in labelled}

    # Leave-one-drug-out baselines (fit-dependent).
    nn_score: dict[str, float | None] = {}
    atc_score: dict[str, float | None] = {}
    for inn in labelled:
        fit = [o for o in labelled if o != inn]
        vec = vectors[inn]
        nbrs = [
            (float(np.linalg.norm(vec - vectors[o])), o)
            for o in fit
            if vec is not None and vectors[o] is not None
        ]
        nn_score[inn] = (
            float(y_all[min(nbrs, key=lambda p: (p[0], p[1]))[1]]) if nbrs else None
        )
        atc_score[inn] = _atc_class_prevalence(
            inn, {o: y_all[o] for o in fit}, drugs, set(fit)
        )

    methods: dict[str, dict[str, float | None]] = {
        "candidate_rank_combined": {},  # filled on the common set below
        "occupancy_max": occ,
        "strongest_target_ki": aff,
        "one_nn_fingerprint": nn_score,
        "atc_class_prevalence": atc_score,
    }
    # Common set: labelled drugs with a non-null score for every non-candidate
    # method AND both candidate views (occupancy + affinity).  (occupancy_max
    # and strongest_target_ki already require those two views, so this is the
    # same intersection the candidate needs.)
    other = ("occupancy_max", "strongest_target_ki", "one_nn_fingerprint", "atc_class_prevalence")
    common = [
        inn for inn in labelled if all(methods[m][inn] is not None for m in other)
    ]
    # The candidate's rank ensemble is computed within the common cohort.
    cand = rank_combined_score(
        [occ[inn] for inn in common], [aff[inn] for inn in common]
    )
    methods["candidate_rank_combined"] = {inn: cand[k] for k, inn in enumerate(common)}

    y = [y_all[inn] for inn in common]
    scored = {m: _metric(y, [methods[m][inn] for inn in common]) for m in methods}

    beats_occ_max = (
        scored["candidate_rank_combined"]["auroc"] is not None
        and scored["occupancy_max"]["auroc"] is not None
        and scored["candidate_rank_combined"]["auroc"] > scored["occupancy_max"]["auroc"]
    )
    return {
        "effect": effect,
        "receptors": list(receptors),
        "common_n": len(common),
        "n_positive": int(sum(y)),
        "n_negative": len(y) - int(sum(y)),
        "lodo": scored,
        "candidate_beats_occupancy_max": beats_occ_max,
    }


def run_candidate_v2_benchmark() -> dict[str, Any]:
    """Train-only report: the P34.1 candidate beside every baseline, per liability."""
    drugs = {row["inn"]: row for row in load_v2_drugs()}
    train, heldout = v2_partition()
    binding = load_v2_binding()
    cmax = load_v2_human_cmax()
    liabilities = load_v2_liabilities()
    fp_targets = _all_targets(binding)

    per_liability = {
        effect: score_liability(
            effect, list(train), binding, cmax, liabilities, drugs, fp_targets
        )
        for effect in CANDIDATE_LIABILITIES
    }

    n_scored = max(
        (per_liability[e]["common_n"] for e in REGISTERED_SCORED), default=0
    )
    beaten = [
        e for e in REGISTERED_SCORED if per_liability[e]["candidate_beats_occupancy_max"]
    ]
    return {
        "task": "P34.1",
        "candidate": (
            "parameter-free rank ensemble of exposure-adjusted occupancy "
            "(occupancy_max) and intrinsic affinity (strongest_target_ki) over "
            "the pre-specified receptors"
        ),
        "selection": "train leave-one-drug-out only; no held-out or fresh drug is read",
        "partition": {"n_train": len(train), "n_heldout": len(heldout)},
        "liabilities": per_liability,
        "registered_scored_liabilities": list(REGISTERED_SCORED),
        "candidate_beats_occupancy_max_liabilities": beaten,
        "n_scored": n_scored,
        "notes": [
            "Candidate and every baseline are scored on the identical common set "
            "per liability (label + occupancy + affinity + both fitted baselines "
            "defined), matching the common-set rule in pipeline/go_verdict.py.",
            "The candidate is label-free and parameter-free, so it has no "
            "intercept to jitter under LODO -- the P33.4 failure mode -- and is "
            "scored directly, like occupancy_max and strongest_target_ki.",
            "Beating occupancy_max on sedation does not make this a GO candidate: "
            "it does not beat occupancy_max on anticholinergic. GO requires "
            "beating every baseline and is a host-side held-out decision, not "
            "this train-LODO selection.",
        ],
    }


def lodo_summary_lines(report: Mapping[str, Any]) -> list[str]:
    """One line per liability for a human-readable tail."""
    lines = []
    for effect, row in report["liabilities"].items():
        parts = []
        for method, m in row["lodo"].items():
            a = m["auroc"]
            parts.append(f"{method}={'n/a' if a is None else f'{a:.3f}'}")
        lines.append(
            f"{effect}: common_n={row['common_n']} (pos {row['n_positive']}); "
            + "; ".join(parts)
        )
    return lines


__all__ = [
    "CANDIDATE_LIABILITIES",
    "REGISTERED_SCORED",
    "lodo_summary_lines",
    "run_candidate_v2_benchmark",
    "score_liability",
]
