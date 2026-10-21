"""P33.1 stage 2: human IV PK truth (clearance_iv, Vss) and the plasma_fu
input, for the v2 benchmark's 331 drugs."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
DRUGS_FILE = V2_DIR / "drugs.json"
TRUTH_FILE = V2_DIR / "truth" / "human_pk.json"
FU_FILE = V2_DIR / "inputs" / "plasma_fu.json"

CL_RANGE = (0.01, 200.0)
VSS_RANGE = (0.01, 100.0)
FU_RANGE = (0.0, 1.0)


@pytest.fixture(scope="module")
def drug_inns():
    return [d["inn"] for d in json.loads(DRUGS_FILE.read_text())["drugs"]]


@pytest.fixture(scope="module")
def truth():
    return json.loads(TRUTH_FILE.read_text())


@pytest.fixture(scope="module")
def fu_inputs():
    return json.loads(FU_FILE.read_text())


def _has_source(entry) -> bool:
    sources = entry.get("sources")
    if not sources:
        return False
    for s in sources:
        if s.get("pmid") or (s.get("doi") or "").startswith("10.6019/CHEMBL"):
            return True
    return False


def test_truth_file_is_a_v2_drugs_table(truth, drug_inns):
    """guards.py's v2 convention: {"drugs": [{"inn": ...}, ...]}."""
    assert isinstance(truth, dict)
    rows = truth["drugs"]
    assert isinstance(rows, list)
    assert len(rows) == len(drug_inns) == truth["n_drugs"]
    seen = {r["inn"] for r in rows}
    assert seen == set(drug_inns)
    for r in rows:
        assert set(r) >= {"inn", "chembl_id", "clearance_iv", "vss", "therapeutic_cmax"}


def test_fu_input_file_is_a_v2_drugs_table(fu_inputs, drug_inns):
    assert isinstance(fu_inputs, dict)
    rows = fu_inputs["drugs"]
    assert len(rows) == len(drug_inns) == fu_inputs["n_drugs"]
    seen = {r["inn"] for r in rows}
    assert seen == set(drug_inns)
    for r in rows:
        assert set(r) >= {"inn", "chembl_id", "fu"}


def test_no_heldout_flag_is_stored(truth, fu_inputs):
    """The v2 held-out rule is guards.v2_heldout(inn): a hash rule, never a
    stored flag. These truth/input files must not carry one."""
    for blob in (truth, fu_inputs):
        text = json.dumps(blob).lower()
        assert "heldout" not in text
        assert "held_out" not in text
        assert "held-out" not in text


def test_clearance_values_are_plausible_and_cited(truth):
    n = 0
    for r in truth["drugs"]:
        cl = r["clearance_iv"]
        if cl is None:
            continue
        n += 1
        assert cl["unit"] == "mL/min/kg"
        assert CL_RANGE[0] <= cl["value"] <= CL_RANGE[1], (r["inn"], cl["value"])
        assert cl["min"] <= cl["value"] <= cl["max"]
        assert cl["n_measurements"] >= 1
        assert cl["matched_on"] == "chembl_id"
        assert _has_source(cl), (r["inn"], "clearance_iv has no verified source")
    assert n >= 100, f"expected at least 100/331 drugs with human IV clearance, got {n}"


def test_vss_values_are_plausible_and_cited(truth):
    n = 0
    for r in truth["drugs"]:
        vss = r["vss"]
        if vss is None:
            continue
        n += 1
        assert vss["unit"] == "L/kg"
        assert VSS_RANGE[0] <= vss["value"] <= VSS_RANGE[1], (r["inn"], vss["value"])
        assert vss["min"] <= vss["value"] <= vss["max"]
        assert vss["n_measurements"] >= 1
        assert vss["matched_on"] == "chembl_id"
        assert _has_source(vss), (r["inn"], "vss has no verified source")
    assert n >= 100, f"expected at least 100/331 drugs with human Vss, got {n}"


def test_therapeutic_cmax_is_explicitly_absent_not_fabricated(truth):
    """No row claims a therapeutic plasma concentration range in this build:
    the only candidate source (Schulz et al. 2020 Crit Care) could not be
    downloaded/verified, so the field is null throughout rather than
    containing an invented or scraped-from-prose number."""
    for r in truth["drugs"]:
        assert r["therapeutic_cmax"] is None


def test_fu_values_are_plausible_and_cited(fu_inputs):
    n = 0
    for r in fu_inputs["drugs"]:
        fu = r["fu"]
        if fu is None:
            continue
        n += 1
        assert fu["unit"] == "fraction"
        assert FU_RANGE[0] <= fu["value"] <= FU_RANGE[1], (r["inn"], fu["value"])
        assert fu["min"] <= fu["value"] <= fu["max"]
        assert fu["n_measurements"] >= 1
        assert fu["matched_on"] == "chembl_id"
        assert _has_source(fu), (r["inn"], "fu has no verified source")
    assert n >= 100, f"expected at least 100/331 drugs with human plasma fu, got {n}"


def test_missing_rows_are_explicit_null_not_omitted(truth, fu_inputs):
    """Missing = explicit absent, never imputed: every drug has a row, and a
    row with no data carries null fields rather than being left out."""
    for r in truth["drugs"]:
        for field in ("clearance_iv", "vss"):
            assert field in r
            assert r[field] is None or isinstance(r[field]["value"], float)
    for r in fu_inputs["drugs"]:
        assert "fu" in r
        assert r["fu"] is None or isinstance(r["fu"]["value"], float)
