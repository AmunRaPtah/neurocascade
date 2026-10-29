"""P30.4: conformal bands refit around the P34.3 frozen candidate.

P30.3 (``neurocascade.benchmark.conformal``) banded the train-mean CL/Vss
fallback and P30.1's logistic.  This module refits the same split-conformal
arithmetic on the residuals of the candidate P34.3 froze
(``frozen_candidate``, the P34.1 rank ensemble of occupancy_max and
affinity_max_pki), and reports every band beside a train-mean comparator.

What the candidate actually predicts decides what can be banded:

* It scores liabilities only.  It has no CL or Vss arm: P33.4's IVIVE arm is
  searched-and-empty because v2 holds no in-vitro CLint.  So CL and Vss carry
  no candidate band, and the P30.3 train-mean band stays the only one.
* Its output is a rank sum within a cohort, not a probability.  A new drug is
  scored by ranking it among the frozen train reference cohort plus itself.
  The rank sum is rescaled to ``u = (R - 2) / (2 m)`` in [0, 1], where ``m``
  is the reference size; that is the drug's engagement percentile.  ``u`` is
  label-free and is not a calibrated probability.  Only the conformal label
  set built on it has checkable coverage.
* Nonconformity is ``1 - u`` for a positive label and ``u`` for a negative,
  the P30.3 ``1 - p(observed)`` form with ``u`` in place of ``p``.  Because
  ``u`` uses no labels, a calibration drug's LODO residual is simply its
  percentile against the other n - 1 train drugs.
* Comparator: the train-prevalence predictor, the binary form of the train
  mean, on the identical calibration drugs (leave-one-out prevalence).

Every output carries ``PROVENANCE_LABEL`` (human decision 2026-10-28: P34
closed with no verdict possible and the work proceeds without a GO).  The v2
held-out split was already scored in P33.4.  Its coverage here is
retrospective and does not certify the bands; P32.4's prospective compounds
are the clean test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..conformal import (
    LEVELS,
    UnbandedQuantityError,
    _heldout_v2_mean,
    _key,
    conformal_quantile,
    coverage,
    label_set,
    load_bands as load_p30_3_bands,
    lodo_coverage,
)
from . import load_v2_binding, load_v2_human_cmax, load_v2_liabilities, v2_heldout, v2_partition
from .candidate import occupancy_baseline
from .candidate_benchmark import _cu, _ki_by_gene
from .candidate_v2 import rank_combined_score
from .frozen_candidate import PARAMS, affinity, load_params

PROVENANCE_LABEL = "not validated: P34 returned no verdict"
BANDS_PATH = Path(__file__).resolve().parent / "frozen" / "p30_4_bands.json"
CANDIDATE_CODE = Path(__file__).resolve().with_name("frozen_candidate.py")

#: P34.1's registered scored liabilities.  EPS and orthostatic hypotension are
#: refused rather than banded: with 2 and 1 train positives a label set would
#: just repeat the majority class (P30.3 recorded this artefact for its
#: anticholinergic and orthostasis bands).
BANDED = ("sedation", "anticholinergic")
UNDERPOWERED = ("eps", "orthostatic_hypotension")

RETROSPECTIVE_NOTE = (
    "retrospective: the v2 held-out split was already scored in P33.4, so this "
    "coverage cannot certify the bands; P32.4 (prospective, post-freeze compounds) "
    "is the clean test"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the candidate score ----------------------------------------------------
def views(effect: str, cu_nM: float | None, ki_by_gene: Mapping[str, float | None],
          params: Mapping[str, Any] | None = None) -> tuple[float | None, float | None]:
    """The frozen candidate's two views (occupancy_max, affinity_max_pki)."""
    params = load_params() if params is None else params
    receptors = tuple(params["liability_receptors"][effect])
    return occupancy_baseline(cu_nM, ki_by_gene, receptors), affinity(ki_by_gene, receptors)


def engagement_percentile(occ: float, aff: float, ref_occ: list[float],
                          ref_aff: list[float]) -> float:
    """The candidate rank sum of one drug among ``ref`` plus itself, scaled to [0, 1]."""
    m = len(ref_occ)
    if m == 0 or m != len(ref_aff):
        raise ValueError("a non-empty reference cohort with both views is required")
    rank_sum = rank_combined_score(list(ref_occ) + [occ], list(ref_aff) + [aff])[-1]
    return (rank_sum - 2.0) / (2.0 * m)


def _score(p_positive: float, label: int) -> float:
    return 1.0 - p_positive if label == 1 else p_positive


# ---- calibration (train only) -----------------------------------------------
def calibration_rows(effect: str, *, train: Iterable[str] | None = None,
                     liabilities=None, cmax=None, binding=None,
                     params=None) -> list[dict[str, Any]]:
    """Train drugs with a label and both candidate views, sorted by INN."""
    train = v2_partition()[0] if train is None else train
    liabilities = load_v2_liabilities() if liabilities is None else liabilities
    cmax = load_v2_human_cmax() if cmax is None else cmax
    binding = load_v2_binding() if binding is None else binding
    rows = []
    for inn in sorted(train):
        label = liabilities.get(inn, {}).get(effect, {}).get("label")
        if label is None:
            continue
        occ, aff = views(effect, _cu(cmax, inn), _ki_by_gene(binding, inn), params)
        if occ is None or aff is None:
            continue
        rows.append({"inn": inn, "label": int(label), "occupancy_max": occ,
                     "affinity_max_pki": aff})
    return rows


def _set_sizes(points: list[float], scores: list[float], level: float) -> dict[str, int]:
    sizes = []
    for i, p in enumerate(points):
        q = conformal_quantile(scores[:i] + scores[i + 1:], level)
        sizes.append(len(label_set(p, q)))
    return {str(k): sizes.count(k) for k in (0, 1, 2)}


def _band_block(points: list[float], scores: list[float]) -> dict[str, Any]:
    return {
        "quantiles": {_key(a): conformal_quantile(scores, a) for a in LEVELS},
        "lodo_coverage": {_key(a): lodo_coverage(scores, a) for a in LEVELS},
        "lodo_set_sizes": {_key(a): _set_sizes(points, scores, a) for a in LEVELS},
    }


def calibrate_liability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Candidate and train-prevalence label-set bands on identical rows."""
    n = len(rows)
    labels = [r["label"] for r in rows]
    occ = [r["occupancy_max"] for r in rows]
    aff = [r["affinity_max_pki"] for r in rows]
    u = [engagement_percentile(occ[i], aff[i], occ[:i] + occ[i + 1:], aff[:i] + aff[i + 1:])
         for i in range(n)]
    cand_scores = [_score(p, y) for p, y in zip(u, labels)]
    total = sum(labels)
    prev = [(total - y) / (n - 1) for y in labels]
    prev_scores = [_score(p, y) for p, y in zip(prev, labels)]
    return {
        "status": "calibrated",
        "kind": "label_set",
        "n_calibration": n,
        "n_positive": total,
        "n_negative": n - total,
        "calibration_drugs": [r["inn"] for r in rows],
        "reference_cohort": {"occupancy_max": occ, "affinity_max_pki": aff},
        "candidate": {
            "method": "P34.3 frozen rank ensemble, engagement percentile u = (R - 2) / (2m)",
            "nonconformity": "1 - u if label 1, u if label 0",
            **_band_block(u, cand_scores),
        },
        "train_mean": {
            "method": "train prevalence (leave-one-out on the calibration drugs)",
            "nonconformity": "1 - p if label 1, p if label 0",
            "frozen_point": total / n,
            **_band_block(prev, prev_scores),
        },
        "provenance": PROVENANCE_LABEL,
    }


def _pk_entry(name: str, p30_3: Mapping[str, Any]) -> dict[str, Any]:
    e = p30_3["quantities"][name]
    return {
        "status": "no-candidate-prediction",
        "reason": "the P34.3 candidate scores liabilities only and has no PK arm; P33.4's "
                  "IVIVE CL arm is searched-and-empty (v2 holds no in-vitro CLint) and no "
                  "Vss predictor was ever scored, so the P30.3 train-mean band is the only band",
        "train_mean": {
            "source": "P30.3 conformal_bands.json",
            "n_calibration": e["n_calibration"],
            "fold_bands": e["fold_bands"],
            "lodo_coverage": e["lodo_coverage"],
        },
        "provenance": PROVENANCE_LABEL,
    }


def calibrate(**sources) -> dict[str, Any]:
    """Recompute every P30.4 band from train rows.  Reads no held-out row."""
    params = load_params()
    quantities: dict[str, Any] = {}
    for effect in BANDED:
        quantities[f"liability:{effect}"] = calibrate_liability(
            calibration_rows(effect, params=params, **sources))
    for effect in UNDERPOWERED:
        rows = calibration_rows(effect, params=params, **sources)
        pos = sum(r["label"] for r in rows)
        quantities[f"liability:{effect}"] = {
            "status": "underpowered",
            "reason": f"{pos} positive(s) among {len(rows)} train drugs with both candidate "
                      "views; a label set would repeat the majority class",
            "provenance": PROVENANCE_LABEL,
        }
    p30_3 = load_p30_3_bands()
    quantities["human_cl"] = _pk_entry("human_cl_v2_train_mean", p30_3)
    quantities["human_vss"] = _pk_entry("human_vss_v2_train_mean", p30_3)
    return {
        "task": "P30.4",
        "method": "split-conformal on the P34.3 frozen candidate's train LODO residuals",
        "provenance": PROVENANCE_LABEL,
        "candidate": {
            "params_path": "src/neurocascade/benchmark/v2/frozen/p34_candidate.json",
            "params_sha256": _sha256(PARAMS),
            "code_path": "src/neurocascade/benchmark/v2/frozen_candidate.py",
            "code_sha256": _sha256(CANDIDATE_CODE),
        },
        "levels": [_key(a) for a in LEVELS],
        "heldout_read_at_freeze": False,
        "quantities": quantities,
    }


def freeze(path: Path = BANDS_PATH) -> dict[str, Any]:
    """Write the band file.  Held-out coverage reads only this file."""
    bands = calibrate()
    path.write_text(json.dumps(bands, indent=1) + "\n")
    return bands


def load_bands(path: Path = BANDS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


# ---- the band API -----------------------------------------------------------
def _entry(bands: Mapping[str, Any], effect: str) -> Mapping[str, Any]:
    entry = bands["quantities"].get(f"liability:{effect}")
    if entry is None:
        raise KeyError(f"{effect!r} is not a P30.4 liability")
    if entry["status"] != "calibrated":
        raise UnbandedQuantityError(f"{effect}: {entry['status']}: {entry.get('reason', '')}")
    return entry


def predict_with_bands(effect: str, cu_nM: float | None, ki_by_gene: Mapping[str, float | None],
                       bands: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Frozen-candidate engagement percentile and 80%/95% label sets for one drug.

    Raises ``ValueError`` when the drug lacks a view (no Cmax,u or no Ki at
    the liability's receptors), since the candidate never imputes one, and
    ``UnbandedQuantityError`` for an underpowered liability.
    """
    bands = bands or load_bands()
    entry = _entry(bands, effect)
    occ, aff = views(effect, cu_nM, ki_by_gene)
    if occ is None or aff is None:
        raise ValueError(f"{effect}: the candidate needs Cmax,u and a Ki at a listed receptor")
    ref = entry["reference_cohort"]
    u = engagement_percentile(occ, aff, ref["occupancy_max"], ref["affinity_max_pki"])
    return {
        "liability": effect,
        "engagement_percentile": u,
        "is_probability": False,
        "bands": {lv: {"labels": list(label_set(u, q)), "bounded": q is not None}
                  for lv, q in entry["candidate"]["quantiles"].items()},
        "provenance": {
            "label": PROVENANCE_LABEL,
            "candidate": "P34.3 frozen rank ensemble",
            "calibration": "P30.4 split-conformal, train LODO",
            "n_calibration": entry["n_calibration"],
        },
    }


# ---- held-out coverage (host) -----------------------------------------------
def _heldout_liability(effect: str, entry: Mapping[str, Any], heldout: list[str],
                       liabilities, cmax, binding) -> dict[str, Any]:
    ref = entry["reference_cohort"]
    cand, prev, cand_u, prev_p = [], [], [], []
    missing_truth, not_scoreable = [], []
    p_prev = entry["train_mean"]["frozen_point"]
    for inn in heldout:
        if inn not in liabilities:
            missing_truth.append(inn)
            continue
        label = liabilities[inn].get(effect, {}).get("label")
        occ, aff = views(effect, _cu(cmax, inn), _ki_by_gene(binding, inn))
        if label is None or occ is None or aff is None:
            not_scoreable.append(inn)
            continue
        u = engagement_percentile(occ, aff, ref["occupancy_max"], ref["affinity_max_pki"])
        cand.append(_score(u, int(label)))
        cand_u.append(u)
        prev.append(_score(p_prev, int(label)))
        prev_p.append(p_prev)

    def block(scores, points, method):
        qs = entry[method]["quantiles"]
        return {
            "coverage": {lv: coverage(scores, q) for lv, q in qs.items()},
            "set_sizes": {lv: {str(k): [len(label_set(p, q)) for p in points].count(k)
                               for k in (0, 1, 2)} for lv, q in qs.items()},
        }

    return {
        "status": "scored" if cand else "searched-and-empty",
        "n_heldout": len(heldout),
        "candidate": block(cand, cand_u, "candidate"),
        "train_mean": block(prev, prev_p, "train_mean"),
        "missing_truth": sorted(missing_truth),
        "not_scoreable": sorted(not_scoreable),
    }


def heldout_coverage(bands: Mapping[str, Any] | None = None, *, heldout: list[str] | None = None,
                     liabilities=None, cmax=None, binding=None,
                     include_pk: bool = True) -> dict[str, Any]:
    """Score the frozen bands on whatever v2 held-out truth this checkout holds."""
    bands = bands or load_bands()
    heldout = list(v2_partition()[1]) if heldout is None else list(heldout)
    liabilities = load_v2_liabilities() if liabilities is None else liabilities
    cmax = load_v2_human_cmax() if cmax is None else cmax
    binding = load_v2_binding() if binding is None else binding
    result: dict[str, Any] = {}
    for name, entry in bands["quantities"].items():
        if name.startswith("liability:") and entry["status"] == "calibrated":
            result[name] = _heldout_liability(name.split(":", 1)[1], entry, heldout,
                                              liabilities, cmax, binding)
    if include_pk:
        p30_3 = load_p30_3_bands()["quantities"]
        for name, field, src in (("human_cl", "clearance_iv", "human_cl_v2_train_mean"),
                                 ("human_vss", "vss", "human_vss_v2_train_mean")):
            r = _heldout_v2_mean(p30_3[src], field)
            result[name] = {"status": r["status"], "n_heldout": r["n_heldout"],
                            "train_mean": {"coverage": r["coverage"]},
                            "missing_truth": r["missing_truth"],
                            "not_scoreable": r["not_scoreable"]}
    n = max((r["train_mean"]["coverage"][_key(LEVELS[0])]["n"] for r in result.values()),
            default=0)
    return {
        "task": "P30.4",
        "status": "scored" if n else "searched-and-empty",
        "retrospective": True,
        "certifies_bands": False,
        "note": RETROSPECTIVE_NOTE,
        "provenance": PROVENANCE_LABEL,
        "max_n_scored": n,
        "quantities": result,
    }


def heldout_summary_lines(report: Mapping[str, Any]) -> list[str]:
    lines = [f"P30.4 held-out coverage [{report['note']}] [{report['provenance']}]"]
    for name, r in report["quantities"].items():
        for method in ("candidate", "train_mean"):
            if method not in r:
                continue
            cells = []
            for lv, c in r[method]["coverage"].items():
                rate = "n/a" if c["rate"] is None else f"{c['rate']:.3f}"
                sizes = r[method].get("set_sizes", {}).get(lv)
                extra = f" sets {sizes}" if sizes else ""
                cells.append(f"{lv}: {c['covered']}/{c['n']} ({rate}){extra}")
            lines.append(f"{name} [{method}]: {r['status']}; " + "; ".join(cells))
    return lines


def train_only_invariant(bands: Mapping[str, Any]) -> bool:
    """No calibration drug is a v2 held-out drug."""
    return not any(v2_heldout(inn)
                   for e in bands["quantities"].values()
                   for inn in e.get("calibration_drugs", []))


__all__ = [
    "BANDED",
    "BANDS_PATH",
    "PROVENANCE_LABEL",
    "RETROSPECTIVE_NOTE",
    "UNDERPOWERED",
    "calibrate",
    "calibrate_liability",
    "calibration_rows",
    "engagement_percentile",
    "freeze",
    "heldout_coverage",
    "heldout_summary_lines",
    "load_bands",
    "predict_with_bands",
    "train_only_invariant",
    "views",
]
