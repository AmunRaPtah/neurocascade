"""P32.1: the compound preclinical-package schema and ``predict_compound()``.

A discovery team hands over a compound's preclinical package and gets back
the predicted human consequences.  Each returned number comes from a scored
upstream predictor and carries that predictor's frozen conformal band:

* **Human clearance:** P29.2's well-stirred IVIVE, falling back to fixed-exponent
  rat allometry, banded by P30.3 ``human_cl_ivive``.  If the package holds
  clearance data that the P29.2 arm cannot use (for example CLint without a
  human fu_p), the v2 train-mean point is used with its own P30.3 band.  This
  is the "widen the band for a missing optional input" route, and the
  provenance says so.
* **Human Vss:** only the v2 train-mean point and band.  P29.2's Oie-Tozer arm
  has never been scored (n=0), so no compound-specific Vss is returned.
* **Liabilities:** P30.4 bands on the P34.3 frozen candidate (rank ensemble of
  occupancy and affinity).  The candidate needs a human unbound Cmax, and a
  preclinical package has none.  When the caller supplies a projected one
  (``human_exposure``), the candidate's engagement percentile and label sets
  are returned.  Without it, the result falls back to P30.4's train-prevalence
  label sets, which never exclude either label.
* **Target-engagement dose, panel occupancy-time, half-life and oral exposure**
  have no scored predictor (P29.3, P29.4 and P29.2's oral arm are
  searched-and-empty).  They come back as ``pending`` with the reason and no
  number.

The result can only be refused when a core input is missing: no measured Ki
anywhere in the binding panel, or no clearance measurement of any kind
(neither human liver CLint nor rat in-vivo CL).  Human decision 2026-10-28:
P34 closed without a verdict, and this work goes ahead without a GO.  Every
returned record therefore carries ``PROVENANCE_LABEL``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .benchmark import load_adme_record, load_binding_panel, load_truth_record
from .benchmark import conformal
from .benchmark import human_pk_translation as _pk
from .benchmark.v2 import load_v2_binding, v2_partition
from .benchmark.v2 import candidate_bands
from .benchmark.v2.chain_benchmark import _all_targets, _fingerprint

SCHEMA = "neurocascade.compound_package/1"
PROVENANCE_LABEL = candidate_bands.PROVENANCE_LABEL
N_NEAREST = 3

_KNOWN_KEYS = {"schema", "name", "binding", "adme", "human_exposure",
               "rodent_occupancy", "rodent_behaviour"}


class CoreInputMissing(ValueError):
    """The package lacks a core input; no prediction is made."""

    def __init__(self, name: str, missing: list[str]):
        self.missing = missing
        super().__init__(f"{name}: refused, missing core input(s): {'; '.join(missing)}")


# ---- the package --------------------------------------------------------------
def validate_package(package: Mapping[str, Any]) -> None:
    """Schema check: field shapes only.  Core-input refusal happens in predict."""
    unknown = set(package) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown package field(s): {sorted(unknown)}")
    if package.get("schema") != SCHEMA:
        raise ValueError(f"package schema must be {SCHEMA!r}")
    if not isinstance(package.get("name"), str) or not package["name"]:
        raise ValueError("package needs a non-empty 'name'")
    for gene, entry in (package.get("binding") or {}).items():
        ki = entry.get("ki_nM")
        if ki is not None and not (isinstance(ki, (int, float)) and ki > 0):
            raise ValueError(f"binding[{gene!r}].ki_nM must be a positive number or null")
    adme = package.get("adme")
    if adme is not None and not isinstance(adme.get("fields", {}), Mapping):
        raise ValueError("adme must hold a 'fields' mapping in the benchmark ADME format")
    exposure = package.get("human_exposure")
    if exposure is not None:
        cu = exposure.get("cmax_unbound_nM")
        if not (isinstance(cu, (int, float)) and cu > 0) or not exposure.get("source"):
            raise ValueError("human_exposure needs a positive cmax_unbound_nM and a 'source'")


def load_package(path: str | Path) -> dict[str, Any]:
    package = json.loads(Path(path).read_text())
    validate_package(package)
    return package


def package_from_benchmark(inn: str) -> dict[str, Any]:
    """A benchmark drug's preclinical package, from preclinical files only.

    It reads the P28.2 binding panel and the P28.3 ADME record and nothing
    else.  No truth file, label, human Cmax or v2 truth is read, so a held-out
    drug's package is exactly what a discovery team would hold.
    """
    panel = load_binding_panel(inn)
    binding = {}
    for gene, item in panel.get("targets", {}).items():
        ki = item.get("ki_nM")
        binding[gene] = {
            "ki_nM": float(ki) if ki is not None and float(ki) > 0 else None,
            "efficacy": sorted({a.get("action") for a in item.get("actions", [])
                                if a.get("action")}) or None,
            "refs": sorted({m["ref"] for m in item.get("measurements", []) if m.get("ref")}),
        }
    adme = load_adme_record(inn)
    return {"schema": SCHEMA, "name": inn, "binding": binding,
            "adme": {"fields": adme.get("fields", {})}}


# ---- core-input check -----------------------------------------------------------
def _ki_by_gene(package: Mapping[str, Any]) -> dict[str, float | None]:
    return {g: e.get("ki_nM") for g, e in (package.get("binding") or {}).items()}


def _has_clearance_data(fields: Mapping[str, Any]) -> bool:
    clint = any(m.get("species") == "human" and m.get("value") is not None
                for m in (fields.get("clint") or {}).get("measurements", []))
    rat_cl = any(m.get("species") == "rat" and m.get("value") is not None
                 for m in (fields.get("invivo_cl_preclinical") or {}).get("measurements", []))
    return clint or rat_cl


def missing_core_inputs(package: Mapping[str, Any]) -> list[str]:
    missing = []
    if not any(_ki_by_gene(package).values()):
        missing.append("binding panel: no measured Ki at any target")
    if not _has_clearance_data((package.get("adme") or {}).get("fields", {})):
        missing.append("clearance: no human liver CLint and no rat in-vivo CL measurement")
    return missing


# ---- records ----------------------------------------------------------------
def _pending(quantity: str, reason: str) -> dict[str, Any]:
    return {"quantity": quantity, "status": "pending", "reason": reason,
            "provenance": {"label": PROVENANCE_LABEL}}


def _fold_record(quantity: str, point: float, *, method: str, inputs: list[str],
                 fallback_reason: str | None = None) -> dict[str, Any]:
    banded = conformal.predict_with_bands(quantity, point)
    return {
        "status": "predicted",
        "point": banded["point"],
        "unit": banded["unit"],
        "bands": banded["bands"],
        "provenance": {
            **banded["provenance"],
            "label": PROVENANCE_LABEL,
            "predictor": method,
            "band": quantity,
            "input_references": inputs,
            "compound_specific": fallback_reason is None,
            "fallback_reason": fallback_reason,
        },
    }


def _clearance(package: Mapping[str, Any]) -> dict[str, Any]:
    record = {"fields": (package.get("adme") or {}).get("fields", {})}
    k = _pk.CONSTANTS
    feat = _pk._ivive_cl(record, package["name"], k)
    method = "well-stirred IVIVE (human liver CLint, human fu)"
    if feat is None:
        feat = _pk._allometric_cl(record, k)
        method = "rat single-species allometry (fixed exponent 0.75), IVIVE-missing fallback"
    if feat is not None:
        return _fold_record("human_cl_ivive", feat.value,
                            method=f"P29.2 {method}; {_pk.WELL_STIRRED_CITATION}",
                            inputs=list(feat.references))
    gaps = []
    if _pk._liver_clint(record) is None:
        gaps.append("no usable human liver-microsome or hepatocyte CLint")
    if _pk._plasma_fu(record, "human") is None:
        gaps.append("no single human plasma fu")
    if _pk._rat_cl(record) is None:
        gaps.append("no rat plasma systemic CL")
    if package["name"] in _pk.NON_MICROSOMAL_CLEARANCE:
        gaps.append(_pk.NON_MICROSOMAL_CLEARANCE[package["name"]])
    entry = conformal.load_bands()["quantities"]["human_cl_v2_train_mean"]
    return _fold_record(
        "human_cl_v2_train_mean", entry["frozen_point"],
        method="P33.3 v2 train arithmetic mean", inputs=[],
        fallback_reason="the package's clearance data cannot feed the P29.2 arm: "
                        + "; ".join(gaps))


def _vss() -> dict[str, Any]:
    entry = conformal.load_bands()["quantities"]["human_vss_v2_train_mean"]
    return _fold_record(
        "human_vss_v2_train_mean", entry["frozen_point"],
        method="P33.3 v2 train arithmetic mean", inputs=[],
        fallback_reason="no compound-specific Vss: " + conformal.UNBANDED["human_vss_ivive"][1])


def _liability(effect: str, package: Mapping[str, Any], bands: Mapping[str, Any]) -> dict[str, Any]:
    entry = bands["quantities"][f"liability:{effect}"]
    if entry["status"] != "calibrated":
        return {"liability": effect, "status": entry["status"], "reason": entry["reason"],
                "provenance": {"label": PROVENANCE_LABEL}}
    exposure = package.get("human_exposure")
    ki = _ki_by_gene(package)
    gap = None
    if exposure is None:
        gap = "no human unbound Cmax in the package (the candidate's exposure view)"
    else:
        occ, aff = candidate_bands.views(effect, exposure["cmax_unbound_nM"], ki)
        if occ is None or aff is None:
            receptors = candidate_bands.load_params()["liability_receptors"][effect]
            gap = f"no measured Ki at any of {', '.join(receptors)}"
    if gap is None:
        out = candidate_bands.predict_with_bands(effect, exposure["cmax_unbound_nM"], ki, bands)
        return {
            "liability": effect,
            "status": "predicted",
            "point": out["engagement_percentile"],
            "point_kind": "engagement percentile among the frozen train cohort",
            "is_probability": False,
            "bands": out["bands"],
            "provenance": {
                **out["provenance"],
                "exposure_source": exposure["source"],
                "exposure_note": "the P30.4 band was calibrated on observed label therapeutic "
                                 "Cmax,u; its coverage under a projected exposure is unmeasured",
                "compound_specific": True,
                "fallback_reason": None,
            },
        }
    p = entry["train_mean"]["frozen_point"]
    return {
        "liability": effect,
        "status": "predicted",
        "point": p,
        "point_kind": "train prevalence",
        "is_probability": True,
        "bands": {lv: {"labels": list(conformal.label_set(p, q)), "bounded": q is not None}
                  for lv, q in entry["train_mean"]["quantiles"].items()},
        "provenance": {
            "label": PROVENANCE_LABEL,
            "method": entry["train_mean"]["method"],
            "calibration": "P30.4 split-conformal, train LODO (train-mean comparator)",
            "n_calibration": entry["n_calibration"],
            "compound_specific": False,
            "fallback_reason": gap,
        },
    }


def _nearest(package: Mapping[str, Any]) -> dict[str, Any]:
    binding = load_v2_binding()
    targets = _all_targets(binding)
    query = {"targets": {g: {"ki_nM": v} for g, v in _ki_by_gene(package).items()}}
    vec = _fingerprint(query, targets)
    method = ("1-NN-style Euclidean distance on the full v2 panel of -log10 Ki "
              "(unmeasured = 100 uM floor), as in the P33.3 1-NN baseline; v2 train drugs only")
    if vec is None:
        return {"status": "pending", "drugs": [], "reason": "no measured Ki on the v2 panel",
                "provenance": {"label": PROVENANCE_LABEL, "method": method}}
    measured = {g for g, v in _ki_by_gene(package).items() if v}
    ranked = []
    for inn in v2_partition()[0]:
        if inn == package["name"]:
            continue
        other = _fingerprint(binding.get(inn, {}), targets)
        if other is not None:
            ranked.append((float(np.linalg.norm(vec - other)), inn))
    ranked.sort()
    drugs = []
    for _, inn in ranked[:N_NEAREST]:
        theirs = {g for g, item in binding[inn].get("targets", {}).items() if item.get("ki_nM")}
        drugs.append({"inn": inn, "shared_measured_targets": sorted(measured & theirs & set(targets))})
    return {"status": "listed", "drugs": drugs,
            "provenance": {"label": PROVENANCE_LABEL, "method": method}}


# ---- the entry point --------------------------------------------------------------
def predict_compound(package: Mapping[str, Any]) -> dict[str, Any]:
    """Predicted human consequences of one compound's preclinical package.

    Raises ``CoreInputMissing`` only when a core input is absent.  Each number
    in the result is inside a record that holds ``bands`` and ``provenance``.
    """
    validate_package(package)
    missing = missing_core_inputs(package)
    if missing:
        raise CoreInputMissing(package["name"], missing)
    p30_4 = candidate_bands.load_bands()
    fields = (package.get("adme") or {}).get("fields", {})
    optional_missing = [name for name, present in (
        ("human_exposure (projected human unbound Cmax)", package.get("human_exposure") is not None),
        ("adme.plasma_fu", bool(fields.get("plasma_fu"))),
        ("adme.invivo_vss_preclinical", bool(fields.get("invivo_vss_preclinical"))),
        ("adme.kp_uu or adme.brain_fu", bool(fields.get("kp_uu") or fields.get("brain_fu"))),
        ("binding efficacy", any((package.get("binding") or {}).get(g, {}).get("efficacy")
                                 for g in package.get("binding") or {})),
        ("rodent_occupancy", package.get("rodent_occupancy") is not None),
        ("rodent_behaviour", package.get("rodent_behaviour") is not None),
    ) if not present]
    pk_pending = conformal.UNBANDED
    return {
        "compound": package["name"],
        "schema": SCHEMA,
        "provenance": {"label": PROVENANCE_LABEL},
        "human_pk": {
            "clearance": _clearance(package),
            "vss": _vss(),
            "half_life": _pending("half_life", "derived from CL and Vss; the two fold bands were "
                                  "calibrated separately and no joint band exists"),
            "oral_exposure": _pending("oral_exposure", "P29.2 oral exposure is searched-and-empty: "
                                      "no same-entity oral F, CL/F or AUC truth to score against"),
        },
        "target_engagement_dose": _pending("target_engagement_dose",
                                           pk_pending["target_engagement_dose"][1]),
        "panel_occupancy_time": _pending("occupancy", pk_pending["occupancy"][1]),
        "liabilities": {effect: _liability(effect, package, p30_4)
                        for effect in candidate_bands.BANDED + candidate_bands.UNDERPOWERED},
        "nearest_benchmark_drugs": _nearest(package),
        "optional_inputs_missing": optional_missing,
        "inputs_not_used": {
            "binding efficacy": "no scored predictor consumes efficacy; the P34.3 candidate is "
                                "occupancy and affinity only",
            "adme.kp_uu / adme.brain_fu": "consumed only by the occupancy chain, which is pending",
            "rodent_occupancy / rodent_behaviour": "no scored rodent-to-human effect translation",
        },
    }


# ---- scoring the product path end to end (train and, on the host, held out) ------
def score_clearance(drugs: list[str]) -> dict[str, Any]:
    """Run ``predict_compound`` from each drug's package; score CL against P28.4 truth.

    Drugs without a truth record go to ``missing_truth``, which is what happens
    to every held-out drug in a session checkout.
    """
    by_band: dict[str, dict[str, Any]] = {}
    missing_truth, refused, no_cl_truth = [], [], []
    for inn in drugs:
        try:
            package = package_from_benchmark(inn)
            result = predict_compound(package)
        except CoreInputMissing:
            refused.append(inn)
            continue
        try:
            load_truth_record(inn)
        except KeyError:
            missing_truth.append(inn)
            continue
        truth = _pk._human_truth(inn, "cl")
        if truth is None:
            no_cl_truth.append(inn)
            continue
        rec = result["human_pk"]["clearance"]
        band = by_band.setdefault(rec["provenance"]["band"], {"drugs": [], "fold_errors": [],
                                                             "covered": {}})
        band["drugs"].append(inn)
        band["fold_errors"].append(max(rec["point"] / truth[0], truth[0] / rec["point"]))
        for lv, b in rec["bands"].items():
            band["covered"].setdefault(lv, 0)
            band["covered"][lv] += b["lower"] <= truth[0] <= b["upper"]
    v2_train = set(v2_partition()[0])
    for name, band in by_band.items():
        fe = band["fold_errors"]
        band["n"] = len(fe)
        band["gmfe"] = float(math.exp(sum(math.log(x) for x in fe) / len(fe)))
        # The train-mean point averages v2 train truth, so a v2 train drug's
        # own CL is inside its prediction; IVIVE fits nothing per drug.
        band["in_sample"] = (sorted(d for d in band["drugs"] if d in v2_train)
                             if name == "human_cl_v2_train_mean" else [])
    n = sum(b["n"] for b in by_band.values())
    n_oos = n - sum(len(b["in_sample"]) for b in by_band.values())
    return {"status": "scored" if n else "searched-and-empty", "n_scored": n,
            "n_out_of_sample": n_oos,
            "by_band": by_band, "missing_truth": sorted(missing_truth),
            "refused": sorted(refused), "no_cl_truth": sorted(no_cl_truth)}


__all__ = [
    "CoreInputMissing",
    "N_NEAREST",
    "PROVENANCE_LABEL",
    "SCHEMA",
    "load_package",
    "missing_core_inputs",
    "package_from_benchmark",
    "predict_compound",
    "score_clearance",
    "validate_package",
]
