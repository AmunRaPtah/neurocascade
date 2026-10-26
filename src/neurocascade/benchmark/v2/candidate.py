"""The v2 chain candidate: an interpretable liability model (P33.4).

This module is the candidate's *predict path*.  It deliberately imports no
truth module and reads no file: it takes already-assembled numeric features
(receptor occupancies) and labels, and returns scores.  The feature assembly
-- which does read the v2 binding input and the Cmax truth -- lives in the
harness (``candidate_benchmark.py`` and ``go_dump.py``), never here.  The GO
pre-registration (docs/prereg-p33-4-go.md) requires that the same predict
function be used for the train leave-one-drug-out selection and for the single
held-out dump, fitted on train rows only; ``liability_predict`` is that
function.

The candidate is multi-target receptor occupancy at therapeutic exposure,

    occ_t = Cu / (Cu + Ki_t),

for the receptors pre-specified for each liability in ``liability_rules.json``
(the same receptor sets the P33.3 baselines use), fed to an L2-regularised
logistic regression.  It is kept deliberately small: one feature for sedation
(H1), five for anticholinergic (M1-M5).  Interpretability, not raw AUROC, is
the point -- a single mechanistic feature per receptor, no fingerprints, no
learned embeddings.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


# Receptor sets per liability, taken from liability_rules.json's "receptor"
# field and resolved to the v2 binding panel's gene symbols.  These are the
# identical targets the P33.3 baselines score (chain_benchmark.py), so the
# candidate and the strongest-Ki / occupancy baselines see the same panel.
# This is a pre-specification, not a fitted or invented quantity.
LIABILITY_RECEPTORS: dict[str, tuple[str, ...]] = {
    "eps": ("DRD2",),
    "sedation": ("HRH1",),
    "anticholinergic": ("CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5"),
    "orthostatic_hypotension": ("ADRA1A", "ADRA1B", "ADRA1D"),
}

# L2 strength on the standardised weights (intercept unpenalised).  This is an
# analysis-reproducibility hyperparameter, not a biological parameter: it is
# fixed, not tuned on held-out data, and chosen moderate so a single-feature
# liability is not driven to a near-step function on 68 train rows.
L2_LAMBDA = 1.0


def occupancy(cu_nM: float | None, ki_nM: float | None) -> float | None:
    """Fractional receptor occupancy ``Cu / (Cu + Ki)`` at unbound exposure.

    ``cu_nM`` is unbound therapeutic plasma Cmax in nM.  ``ki_nM is None`` means
    the drug was searched and has no usable human Ki at this receptor: that is
    weak binding, so occupancy is 0, never missing.  ``cu_nM is None`` means no
    exposure is known, so occupancy is undefined (``None``) and the drug cannot
    be scored by this candidate.
    """
    if cu_nM is None:
        return None
    if ki_nM is None:
        return 0.0
    cu = float(cu_nM)
    ki = float(ki_nM)
    if cu < 0 or ki < 0:
        raise ValueError("Cu and Ki must be non-negative")
    if cu == 0.0:
        return 0.0
    return cu / (cu + ki)


def feature_vector(
    cu_nM: float | None,
    ki_by_gene: Mapping[str, float | None],
    receptors: Sequence[str],
) -> list[float] | None:
    """Occupancy feature row for one drug, one per receptor, or ``None``.

    Returns ``None`` when ``cu_nM`` is unknown (the drug is unscoreable).  A
    receptor absent from ``ki_by_gene`` is treated as no usable Ki (occupancy
    0), matching ``occupancy``.
    """
    if cu_nM is None:
        return None
    return [occupancy(cu_nM, ki_by_gene.get(gene)) for gene in receptors]


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return mu, sd


def _fit(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """L2 logistic regression on standardised features (intercept unpenalised).

    Returns ``(w, b, mu, sd)``.  Standardisation statistics come from the fit
    rows only, so the whole fit (including scaling) depends on train data
    alone; held-out rows never touch it.
    """
    mu, sd = _standardise(X)
    Xs = (X - mu) / sd
    n, d = Xs.shape

    def neg_log_lik(theta: np.ndarray) -> float:
        w = theta[:d]
        b = theta[d]
        z = Xs @ w + b
        # stable log(1+exp(z)): logaddexp(0, z)
        ll = np.sum(y * z - np.logaddexp(0.0, z))
        return -ll / n + 0.5 * l2 * np.dot(w, w) / n

    def grad(theta: np.ndarray) -> np.ndarray:
        w = theta[:d]
        b = theta[d]
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        resid = p - y
        gw = Xs.T @ resid / n + l2 * w / n
        gb = np.sum(resid) / n
        return np.concatenate([gw, [gb]])

    theta0 = np.zeros(d + 1)
    res = minimize(neg_log_lik, theta0, jac=grad, method="L-BFGS-B")
    theta = res.x
    return theta[:d], float(theta[d]), mu, sd


def _predict(model: tuple[np.ndarray, float, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    w, b, mu, sd = model
    Xs = (X - mu) / sd
    z = Xs @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def liability_predict(
    train_X: Sequence[Sequence[float]],
    train_y: Sequence[int],
    test_X: Sequence[Sequence[float]],
    *,
    l2: float = L2_LAMBDA,
) -> list[float]:
    """Fit the candidate on train rows and score test rows (THE predict path).

    ``train_X``/``test_X`` are occupancy feature rows (see ``feature_vector``);
    ``train_y`` is the 0/1 liability labels.  The model is fitted on the train
    rows only and applied to the test rows; this one function is used both for
    train leave-one-drug-out and for the held-out dump, so the two are the same
    estimator by construction.  Degenerate train sets (one class, or empty)
    fall back to the train positive rate, which is a constant score and has an
    AUROC of 0.5 -- an honest "no signal", not a crash.
    """
    Xtr = np.asarray(train_X, dtype=float)
    ytr = np.asarray(train_y, dtype=float)
    Xte = np.asarray(test_X, dtype=float)
    if Xte.ndim == 1:
        Xte = Xte.reshape(1, -1)
    if len(ytr) == 0 or len(np.unique(ytr)) < 2:
        rate = float(ytr.mean()) if len(ytr) else 0.0
        return [rate] * len(Xte)
    model = _fit(Xtr, ytr, l2)
    return [float(p) for p in _predict(model, Xte)]


def occupancy_baseline(
    cu_nM: float | None,
    ki_by_gene: Mapping[str, float | None],
    receptors: Sequence[str],
) -> float | None:
    """The unfitted Cmax,u/Ki occupancy baseline: max occupancy over the
    liability's receptors.  ``None`` when exposure is unknown.  This is the
    extra baseline the GO pre-registration adds once v2 Cmax exists; it is a
    baseline, not the candidate (no fit, no multi-receptor combination)."""
    row = feature_vector(cu_nM, ki_by_gene, receptors)
    if row is None:
        return None
    return max(row) if row else None


__all__ = [
    "L2_LAMBDA",
    "LIABILITY_RECEPTORS",
    "feature_vector",
    "liability_predict",
    "occupancy",
    "occupancy_baseline",
]
