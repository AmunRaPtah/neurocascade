"""Train leave-one-drug-out scoring of the P33.4 candidate vs every baseline.

This is the harness: it reads the v2 binding input and the Cmax / liability
truth, assembles occupancy features, and scores the candidate
(``candidate.liability_predict``) and the P33.3 baselines plus the new
Cmax,u/Ki occupancy baseline, all on the *same* common set of train drugs by
leave-one-drug-out.  The held-out comparison that actually decides GO / NO-GO
is run once on the host through ``go_dump.py`` + ``pipeline/go_verdict.py``;
nothing here touches held-out truth.

The candidate's predict path lives in ``candidate.py`` and reads no truth; this
module does the (allowed) truth reads and feeds it numbers, exactly as the host
dump will.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Mapping

import numpy as np

from . import (
    load_v2_binding,
    load_v2_drugs,
    load_v2_human_cmax,
    load_v2_liabilities,
    v2_partition,
)
from .candidate import (
    L2_LAMBDA,
    LIABILITY_RECEPTORS,
    feature_vector,
    liability_predict,
    occupancy_baseline,
)
from .chain_benchmark import (
    _all_targets,
    _atc_class_prevalence,
    _binding_score,
    _fingerprint,
    auprc,
    auroc,
)

# Liabilities the candidate can score: a receptor set is pre-specified and the
# v2 universe has some positives.  weight_gain is withheld (no labels) and so
# is absent.  eps and orthostatic_hypotension are underpowered in the v2
# universe (6 and 3 positives); they are scored here for completeness and the
# host gate decides on held-out power, but the GO-registered scored set is
# sedation and anticholinergic.
CANDIDATE_LIABILITIES = ("sedation", "anticholinergic", "eps", "orthostatic_hypotension")


def _cu(cmax: Mapping[str, Mapping[str, Any]], inn: str) -> float | None:
    row = cmax.get(inn)
    if not row or row.get("status") != "value":
        return None
    return row.get("cmax_unbound_nM")


def _ki_by_gene(binding: Mapping[str, Mapping[str, Any]], inn: str) -> dict[str, float | None]:
    panel = binding.get(inn, {}).get("targets", {})
    out: dict[str, float | None] = {}
    for gene, item in panel.items():
        ki = item.get("ki_nM")
        out[gene] = float(ki) if ki is not None and float(ki) > 0 else None
    return out


def _metric(labels: list[int], scores: list[float]) -> dict[str, Any]:
    return {
        "n": len(labels),
        "n_positive": int(sum(labels)),
        "n_negative": len(labels) - int(sum(labels)),
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
    }


def score_liability_lodo(
    effect: str,
    train: list[str],
    binding: Mapping[str, Mapping[str, Any]],
    cmax: Mapping[str, Mapping[str, Any]],
    liabilities: Mapping[str, Mapping[str, Any]],
    drugs: Mapping[str, Mapping[str, Any]],
    fp_targets: tuple[str, ...],
) -> dict[str, Any]:
    """Candidate and baseline metrics for one liability, train LODO.

    The common set is the train drugs that have a label, a Cmax (so the
    candidate and occupancy baseline are defined) and a usable score for every
    baseline, so every method is scored on identical rows -- the same
    intersection ``pipeline/go_verdict.py`` applies on held-out."""
    receptors = LIABILITY_RECEPTORS[effect]
    labelled = [
        inn for inn in sorted(train)
        if liabilities.get(inn, {}).get(effect, {}).get("label") is not None
    ]
    y_all = {inn: int(liabilities[inn][effect]["label"]) for inn in labelled}
    cu = {inn: _cu(cmax, inn) for inn in labelled}
    kis = {inn: _ki_by_gene(binding, inn) for inn in labelled}
    feats = {inn: feature_vector(cu[inn], kis[inn], receptors) for inn in labelled}
    vectors = {inn: _fingerprint(binding.get(inn, {}), fp_targets) for inn in labelled}

    # Intrinsic (fit-independent) baselines.
    strongest = {inn: _binding_score(binding.get(inn, {}), receptors) for inn in labelled}
    occ = {inn: occupancy_baseline(cu[inn], kis[inn], receptors) for inn in labelled}

    # Fit-dependent baselines and candidate, by leave-one-drug-out.
    cand_score: dict[str, float | None] = {}
    nn_score: dict[str, float | None] = {}
    atc_score: dict[str, float | None] = {}
    for inn in labelled:
        fit = [o for o in labelled if o != inn]
        # candidate: needs this drug's features and >=1 other labelled+featured drug
        if feats[inn] is not None:
            fit_feat = [(feats[o], y_all[o]) for o in fit if feats[o] is not None]
            if fit_feat:
                tr_X = [f for f, _ in fit_feat]
                tr_y = [v for _, v in fit_feat]
                cand_score[inn] = liability_predict(tr_X, tr_y, [feats[inn]], l2=L2_LAMBDA)[0]
            else:
                cand_score[inn] = None
        else:
            cand_score[inn] = None
        # 1-NN fingerprint
        vec = vectors[inn]
        nbrs = [(float(np.linalg.norm(vec - vectors[o])), o)
                for o in fit if vec is not None and vectors[o] is not None]
        nn_score[inn] = float(y_all[min(nbrs, key=lambda p: (p[0], p[1]))[1]]) if nbrs else None
        # ATC-class prevalence
        atc_score[inn] = _atc_class_prevalence(inn, {o: y_all[o] for o in fit}, drugs, set(fit))

    # Full featured set (every drug with a Cmax, occupancy 0 where no Ki): the
    # candidate and occupancy baseline are both defined here, so this shows the
    # candidate's own quality at full n, independent of the strongest-Ki common
    # set.  Single-receptor liabilities expose the LODO intercept jitter: the
    # per-fold intercept re-orders the many occupancy-0 drugs, so the LODO AUROC
    # falls below the single-model (held-out) value; the held-out dump fits one
    # model and does not have this.
    featured = [inn for inn in labelled if feats[inn] is not None]
    y_feat = [y_all[inn] for inn in featured]
    occ_feat = [occ[inn] for inn in featured]
    cand_feat = [cand_score[inn] for inn in featured]
    full = {
        "n": len(featured),
        "n_positive": int(sum(y_feat)),
        "candidate_lodo_auroc": auroc(y_feat, cand_feat) if featured else None,
        "candidate_single_model_auroc": (
            auroc(y_feat, liability_predict(
                [feats[inn] for inn in featured], y_feat, [feats[inn] for inn in featured]))
            if len(set(y_feat)) > 1 else None
        ),
        "occupancy_max_auroc": auroc(y_feat, occ_feat) if featured else None,
    }

    methods = {
        "candidate_occupancy_logistic": cand_score,
        "occupancy_max": occ,
        "strongest_target_ki": strongest,
        "one_nn_fingerprint": nn_score,
        "atc_class_prevalence": atc_score,
    }
    # Common set: labelled drugs with a non-null score for every method.
    common = [inn for inn in labelled if all(methods[m][inn] is not None for m in methods)]
    y = [y_all[inn] for inn in common]
    scored = {
        m: _metric(y, [methods[m][inn] for inn in common])
        for m in methods
    }
    return {
        "effect": effect,
        "receptors": list(receptors),
        "common_n": len(common),
        "n_positive": int(sum(y)),
        "n_negative": len(y) - int(sum(y)),
        "lodo": scored,
        "full_featured": full,
    }


def run_candidate_benchmark() -> dict[str, Any]:
    """Train-only LODO report: candidate beside every baseline, per liability."""
    drugs = {row["inn"]: row for row in load_v2_drugs()}
    train, heldout = v2_partition()
    binding = load_v2_binding()
    cmax = load_v2_human_cmax()
    liabilities = load_v2_liabilities()
    fp_targets = _all_targets(binding)

    per_liability = {}
    for effect in CANDIDATE_LIABILITIES:
        per_liability[effect] = score_liability_lodo(
            effect, list(train), binding, cmax, liabilities, drugs, fp_targets
        )

    scored_registered = ("sedation", "anticholinergic")
    n_scored = max(
        (per_liability[e]["common_n"] for e in scored_registered), default=0
    )
    return {
        "task": "P33.4",
        "candidate": "multi-target occupancy -> L2 logistic on pre-specified receptors",
        "selection": "train leave-one-drug-out only; held-out decided once by host go_verdict",
        "l2_lambda": L2_LAMBDA,
        "partition": {"n_train": len(train), "n_heldout": len(heldout)},
        "liabilities": per_liability,
        "registered_scored_liabilities": list(scored_registered),
        "secondary_pk": {
            "status": "searched-and-empty",
            "what": "human CL by well-stirred IVIVE on corrected v2 PK (candidate 2)",
            "why": (
                "Well-stirred IVIVE needs an in-vitro intrinsic-clearance input "
                "(microsomal or hepatocyte CLint) per drug; v2 has none. "
                "human_pk.json/human_pk_corrections.json (P33.5) hold observed in-vivo "
                "CL and Vss only, and the correction rule explicitly removes microsomal "
                "and intrinsic-clearance activities. No CLint source was assembled, so "
                "IVIVE cannot be computed here; the train-mean CL baseline (P33.3, GMFE "
                "2.93 LODO) stands. PK is secondary and non-gating per "
                "docs/prereg-p33-4-go.md."
            ),
            "train_mean_cl_lodo_gmfe": 2.93,
        },
        "n_scored": n_scored,
        "notes": [
            "Candidate and every baseline are scored on the identical common set "
            "per liability (label + Cmax + every baseline defined), matching the "
            "common-set rule in pipeline/go_verdict.py.",
            "Single-receptor liabilities (sedation=H1) give the candidate the same "
            "ranking as the occupancy_max baseline, so their LODO AUROCs coincide; "
            "the candidate's degree of freedom is multi-receptor (anticholinergic).",
            "This is train LODO only. The GO/NO-GO verdict is host-side and binds once.",
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
            f"{effect}: common_n={row['common_n']} (pos {row['n_positive']}); " + "; ".join(parts)
        )
    return lines


__all__ = [
    "CANDIDATE_LIABILITIES",
    "lodo_summary_lines",
    "run_candidate_benchmark",
    "score_liability_lodo",
]
