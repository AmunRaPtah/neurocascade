"""P32.1: compound package schema and predict_compound()."""

import ast
import inspect
import json
import math
from pathlib import Path

import pytest

from neurocascade import compound
from neurocascade.benchmark import conformal, load_split
from neurocascade.benchmark import human_pk_translation as pk
from neurocascade.benchmark.v2 import candidate_bands, load_v2_human_cmax, v2_heldout
from neurocascade.compound import (
    PROVENANCE_LABEL,
    SCHEMA,
    CoreInputMissing,
    load_package,
    package_from_benchmark,
    predict_compound,
    score_clearance,
)

LABEL = "not validated: P34 returned no verdict"

# Synthetic test inputs, not model parameters: they only exercise the code paths.
RAT_CL_ROW = {"species": "rat", "relation": "=", "value": 30.0, "units": "mL.min-1.kg-1",
              "description": "Clearance in rat after iv dosing", "ref": "SYNTHETIC_TEST"}
HUMAN_CLINT_ROW = {"species": "human", "relation": "=", "value": 20.0,
                   "units": "mL.min-1.g-1", "matrix": "microsomes",
                   "description": "Intrinsic clearance in human liver microsomes",
                   "ref": "SYNTHETIC_TEST"}


def _minimal(**over):
    package = {
        "schema": SCHEMA,
        "name": "synthetic_minimal",
        "binding": {"HRH1": {"ki_nM": 5.0}},
        "adme": {"fields": {"invivo_cl_preclinical": {"measurements": [RAT_CL_ROW]}}},
    }
    package.update(over)
    return package


def _walk_numbers_are_banded(node, banded=False, path="$"):
    """Every number sits under a record with bands and the P34 label; every
    record with bands, or with a status, carries the label itself."""
    if isinstance(node, dict):
        if "bands" in node or "status" in node:
            assert node.get("provenance", {}).get("label") == LABEL, path
        here = banded or ("bands" in node and node["provenance"]["label"] == LABEL)
        for k, v in node.items():
            _walk_numbers_are_banded(v, here, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_numbers_are_banded(v, banded, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        assert banded, f"unbanded number at {path}"


def test_provenance_label_is_the_p30_4_label():
    assert PROVENANCE_LABEL == LABEL == candidate_bands.PROVENANCE_LABEL


# ---- held-out drug, end to end from a package file ---------------------------------
def test_a_heldout_drug_runs_end_to_end_from_its_package_file(tmp_path):
    inn = "venlafaxine"
    assert inn in load_split()["heldout"] and v2_heldout(inn)
    path = tmp_path / "venlafaxine.json"
    path.write_text(json.dumps(package_from_benchmark(inn)))
    result = predict_compound(load_package(path))
    cl = result["human_pk"]["clearance"]
    assert cl["provenance"]["band"] == "human_cl_ivive"
    assert cl["provenance"]["compound_specific"] is True
    assert cl["point"] == pytest.approx(13.256792046681554)
    f80 = conformal.load_bands()["quantities"]["human_cl_ivive"]["fold_bands"]["0.80"]
    assert cl["bands"]["0.80"]["upper"] == pytest.approx(cl["point"] * f80)
    assert cl["bands"]["0.95"] == {"lower": 0.0, "upper": math.inf, "bounded": False}
    assert [d["inn"] for d in result["nearest_benchmark_drugs"]["drugs"]] == [
        "desvenlafaxine", "cocaine", "duloxetine"]
    _walk_numbers_are_banded(result)


def test_package_and_predict_paths_read_no_truth():
    src = inspect.getsource(compound.package_from_benchmark) + inspect.getsource(
        compound.predict_compound)
    for name in ("load_truth_record", "load_v2_human_cmax", "load_v2_liabilities",
                 "load_v2_human_pk", "_human_truth", "effect_labels"):
        assert name not in src
    tree = ast.parse(inspect.getsource(compound))
    funcs = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for helper in ("_clearance", "_vss", "_liability", "_nearest", "_ki_by_gene"):
        assert "truth" not in funcs[helper] and "load_v2_human" not in funcs[helper]


# ---- full package ---------------------------------------------------------------
def test_a_full_package_reaches_the_frozen_candidate():
    # clozapine is a P28 and v2 train drug, so its observed label Cmax,u is
    # available here; it stands in for a team's projected exposure.
    package = package_from_benchmark("clozapine")
    cu = load_v2_human_cmax()["clozapine"]["cmax_unbound_nM"]
    package["human_exposure"] = {"cmax_unbound_nM": cu,
                                 "source": "v2 truth/human_cmax.json (train drug)"}
    package["rodent_occupancy"] = {"note": "accepted, not used"}
    package["rodent_behaviour"] = {"note": "accepted, not used"}
    result = predict_compound(package)
    ki = {g: e["ki_nM"] for g, e in package["binding"].items()}
    for effect in ("sedation", "anticholinergic"):
        rec = result["liabilities"][effect]
        direct = candidate_bands.predict_with_bands(effect, cu, ki)
        assert rec["point"] == direct["engagement_percentile"]
        assert rec["bands"] == direct["bands"]
        assert rec["is_probability"] is False
        assert rec["provenance"]["compound_specific"] is True
        assert "unmeasured" in rec["provenance"]["exposure_note"]
    for effect in ("eps", "orthostatic_hypotension"):
        assert result["liabilities"][effect]["status"] == "underpowered"
        assert "point" not in result["liabilities"][effect]
    assert result["human_pk"]["clearance"]["provenance"]["band"] == "human_cl_ivive"
    assert "human_exposure (projected human unbound Cmax)" not in result["optional_inputs_missing"]
    _walk_numbers_are_banded(result)


# ---- minimal package -------------------------------------------------------------
def test_a_minimal_package_widens_to_measured_fallback_bands():
    result = predict_compound(_minimal())
    cl = result["human_pk"]["clearance"]
    k = pk.CONSTANTS
    expected = 30.0 * k.rat_bw_kg * (k.human_bw_kg / k.rat_bw_kg) ** 0.75 / k.human_bw_kg
    assert cl["point"] == pytest.approx(expected)
    assert cl["provenance"]["band"] == "human_cl_ivive"
    assert "allometry" in cl["provenance"]["predictor"]
    vss = result["human_pk"]["vss"]
    assert vss["provenance"]["band"] == "human_vss_v2_train_mean"
    assert vss["provenance"]["compound_specific"] is False
    for effect in ("sedation", "anticholinergic"):
        rec = result["liabilities"][effect]
        assert rec["point_kind"] == "train prevalence"
        assert rec["bands"]["0.80"]["labels"] == [0, 1]
        assert rec["bands"]["0.95"]["labels"] == [0, 1]
        assert "no human unbound Cmax" in rec["provenance"]["fallback_reason"]
    for name in ("target_engagement_dose", "panel_occupancy_time"):
        assert result[name]["status"] == "pending" and "point" not in result[name]
    assert len(result["optional_inputs_missing"]) == 7
    _walk_numbers_are_banded(result)


def test_unusable_clearance_data_falls_back_to_the_train_mean_band():
    package = _minimal(adme={"fields": {"clint": {"measurements": [HUMAN_CLINT_ROW]}}})
    cl = predict_compound(package)["human_pk"]["clearance"]
    entry = conformal.load_bands()["quantities"]["human_cl_v2_train_mean"]
    assert cl["point"] == entry["frozen_point"]
    assert cl["provenance"]["band"] == "human_cl_v2_train_mean"
    assert "no single human plasma fu" in cl["provenance"]["fallback_reason"]


def test_exposure_without_a_listed_receptor_ki_falls_back():
    package = _minimal(human_exposure={"cmax_unbound_nM": 10.0, "source": "synthetic"})
    result = predict_compound(package)
    assert result["liabilities"]["sedation"]["point_kind"].startswith("engagement")
    anti = result["liabilities"]["anticholinergic"]
    assert anti["point_kind"] == "train prevalence"
    assert "CHRM1" in anti["provenance"]["fallback_reason"]
    _walk_numbers_are_banded(result)


# ---- refusal ---------------------------------------------------------------------
def test_a_package_missing_core_inputs_is_refused():
    with pytest.raises(CoreInputMissing) as err:
        predict_compound(_minimal(binding={"HRH1": {"ki_nM": None}}))
    assert err.value.missing == ["binding panel: no measured Ki at any target"]
    with pytest.raises(CoreInputMissing) as err:
        predict_compound(_minimal(adme={"fields": {}}))
    assert err.value.missing == [
        "clearance: no human liver CLint and no rat in-vivo CL measurement"]
    with pytest.raises(CoreInputMissing) as err:
        predict_compound(_minimal(binding={}, adme=None))
    assert len(err.value.missing) == 2


def test_schema_violations_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        predict_compound(_minimal(human_cmax=1.0))
    with pytest.raises(ValueError, match="schema"):
        predict_compound(_minimal(schema="other"))
    with pytest.raises(ValueError, match="source"):
        predict_compound(_minimal(human_exposure={"cmax_unbound_nM": 1.0}))
    with pytest.raises(ValueError, match="positive"):
        predict_compound(_minimal(binding={"HRH1": {"ki_nM": -1.0}}))


# ---- scored product path -------------------------------------------------------------
def test_train_clearance_scored_end_to_end_matches_results_file():
    report = score_clearance(list(load_split()["train"]))
    ivive = report["by_band"]["human_cl_ivive"]
    mean = report["by_band"]["human_cl_v2_train_mean"]
    assert ivive["n"] == 13 and ivive["covered"] == {"0.80": 11, "0.95": 13}
    assert ivive["gmfe"] == pytest.approx(3.0874815703237726)
    assert mean["drugs"] == ["aripiprazole", "benperidol", "prochlorperazine", "remoxipride"]
    assert mean["covered"] == {"0.80": 2, "0.95": 4}
    assert mean["in_sample"] == ["aripiprazole", "prochlorperazine"]
    assert report["n_out_of_sample"] == 15
    # The IVIVE-path points are exactly P29.2's no-fit train predictions.
    p29_2 = {p.drug: p.predicted for p in pk.score_train()["cl"].predictions}
    assert set(ivive["drugs"]) <= set(p29_2)
    stored = json.loads((Path(__file__).resolve().parents[1]
                         / "results" / "P32.1.json").read_text())
    assert stored["n_scored"] == report["n_out_of_sample"]
    assert stored["train"]["by_band"]["human_cl_ivive"]["covered"] == ivive["covered"]


def test_heldout_scoring_puts_every_drug_in_one_bucket():
    heldout = list(load_split()["heldout"])
    report = score_clearance(heldout)
    scored = [d for b in report["by_band"].values() for d in b["drugs"]]
    buckets = report["missing_truth"] + report["refused"] + report["no_cl_truth"] + scored
    assert sorted(buckets) == sorted(heldout)
    assert (report["status"] == "scored") == (report["n_scored"] > 0)
