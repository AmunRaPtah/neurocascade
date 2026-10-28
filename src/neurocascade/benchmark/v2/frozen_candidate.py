"""The frozen P34 candidate's predict path (P34.3).

``frozen/p34_candidate.json`` holds the frozen candidate: the P34.1 family
(``candidate_v2.rank_combined_score``), its liability->receptor map and its
fitted parameters, of which there are none.  ``results/P34.3.json`` pins the
sha256 of this file, the params file and every module the predict path runs,
before any fresh held-out truth exists (docs/prereg-p34-go.md).  After the
freeze nothing edits them; tests/test_p34_3_frozen_candidate.py enforces it.

This module imports no truth loader and reads no truth file.  Callers hand it
per-drug unbound Cmax and Ki-by-gene numbers and get scores back.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate import occupancy_baseline
from .candidate_v2 import rank_combined_score

PARAMS = Path(__file__).resolve().parent / "frozen" / "p34_candidate.json"


def load_params() -> dict[str, Any]:
    return json.loads(PARAMS.read_text())


def affinity(ki_by_gene: Mapping[str, float | None], receptors: Sequence[str]) -> float | None:
    """max -log10(Ki_nM) over the receptors with a measured Ki; None if none."""
    vals = [-math.log10(ki_by_gene[g]) for g in receptors if ki_by_gene.get(g)]
    return max(vals) if vals else None


def predict_cohort(
    effect: str,
    cu_nM: Sequence[float | None],
    ki_by_gene: Sequence[Mapping[str, float | None]],
    params: Mapping[str, Any] | None = None,
) -> list[float | None]:
    """Frozen-candidate scores for one liability over one cohort of drugs.

    The score is a within-cohort rank sum, so it is defined relative to the
    cohort passed in; AUROC depends only on its ordering."""
    params = load_params() if params is None else params
    receptors = tuple(params["liability_receptors"][effect])
    occ = [occupancy_baseline(c, k, receptors) for c, k in zip(cu_nM, ki_by_gene)]
    aff = [affinity(k, receptors) for k in ki_by_gene]
    return rank_combined_score(occ, aff)


__all__ = ["PARAMS", "affinity", "load_params", "predict_cohort"]
