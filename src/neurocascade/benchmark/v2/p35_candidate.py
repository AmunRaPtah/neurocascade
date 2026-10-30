"""The frozen P35 candidate's predict path (P35.3).

Selected on P35-train leave-one-drug-out under ``docs/prereg-p35-go.md``
(``p35_selection.py`` is the harness) and frozen in ``frozen/p35_candidate.json``;
``results/P35.3.json`` pins the sha256 of this file and the params file before
any P35 held-out label exists.  After the freeze nothing edits them.

The candidate is the within-cohort rank sum of two views of a drug:

  * ``occupancy_max = max_t Cu/(Cu+Ki_t)`` over the liability's pre-specified
    receptors (the exposure-adjusted primary mechanism), and
  * ``knn_fraction``: the fraction of SIDER-positive drugs among the ``k``
    nearest P35-train drugs in full-panel receptor-affinity space (the
    off-target pharmacology the pre-specified receptor list leaves out).

The kNN "fit" is the frozen pool in the params file: train INNs, their
fingerprints and their train labels.  A drug is never its own neighbour (it is
excluded by INN), which is what makes the train scores leave-one-drug-out.

This module imports no truth loader and reads no file except the params file.
Callers hand it per-drug INN, unbound Cmax and Ki-by-gene numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .candidate import occupancy_baseline

PARAMS = Path(__file__).resolve().parent / "frozen" / "p35_candidate.json"


def load_params() -> dict[str, Any]:
    return json.loads(PARAMS.read_text())


def fingerprint(ki_by_gene: Mapping[str, float | None], panel: Sequence[str],
                floor: float) -> list[float] | None:
    """Full-panel -log10(Ki_nM) vector, ``floor`` where no Ki; None if no Ki at all.

    Identical to ``chain_benchmark._fingerprint`` (the 1-NN baseline's vector)."""
    vals = [-math.log10(ki_by_gene[g]) if ki_by_gene.get(g) else floor for g in panel]
    return vals if any(ki_by_gene.get(g) for g in panel) else None


def knn_fraction(inn: str, vector: Sequence[float] | None, effect: str,
                 params: Mapping[str, Any]) -> float | None:
    """Mean train label of the k nearest pool drugs (Euclidean; ties by INN).

    ``inn`` itself is never a neighbour.  None when the drug has no fingerprint."""
    if vector is None:
        return None
    pool = params["pool"]
    v = np.asarray(vector, dtype=float)
    dists = sorted(
        (float(np.linalg.norm(v - np.asarray(row["fingerprint"], dtype=float))), other)
        for other, row in pool.items()
        if other != inn and row["labels"].get(effect) in (0, 1)
    )
    nearest = dists[: int(params["k"])]
    if not nearest:
        return None
    return float(np.mean([pool[o]["labels"][effect] for _, o in nearest]))


def predict_cohort(
    effect: str,
    inns: Sequence[str],
    cu_nM: Sequence[float | None],
    ki_by_gene: Sequence[Mapping[str, float | None]],
    params: Mapping[str, Any] | None = None,
) -> list[float | None]:
    """Frozen-candidate scores for one liability over one cohort of drugs.

    The score is a within-cohort rank sum over the drugs where both views are
    defined (others get None, never imputed), so it is defined relative to the
    cohort passed in; AUROC depends only on its ordering."""
    params = load_params() if params is None else params
    if effect not in params["liability_receptors"]:
        raise KeyError(f"the P35 candidate has no pre-specified receptors for {effect!r}")
    receptors = tuple(params["liability_receptors"][effect])
    panel, floor = params["panel"], float(params["fingerprint_floor"])
    occ = [occupancy_baseline(c, k, receptors) for c, k in zip(cu_nM, ki_by_gene)]
    knn = [knn_fraction(i, fingerprint(k, panel, floor), effect, params)
           for i, k in zip(inns, ki_by_gene)]
    idx = [j for j in range(len(inns)) if occ[j] is not None and knn[j] is not None]
    out: list[float | None] = [None] * len(inns)
    if idx:
        r = rankdata([occ[j] for j in idx]) + rankdata([knn[j] for j in idx])
        for n, j in enumerate(idx):
            out[j] = float(r[n])
    return out


__all__ = ["PARAMS", "fingerprint", "knn_fraction", "load_params", "predict_cohort"]
