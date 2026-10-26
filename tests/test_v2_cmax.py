"""P33.3 follow-up: human therapeutic plasma Cmax truth for the v2 drugs.

The file may have its held-out rows stripped (pipeline guards hide_heldout),
so row-level checks only require rows to be a subset of the universe and
every train drug to be present.
"""

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import load_v2_human_cmax, v2_heldout

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
DRUGS_FILE = V2_DIR / "drugs.json"
CMAX_FILE = V2_DIR / "truth" / "human_cmax.json"
FU_FILE = V2_DIR / "inputs" / "plasma_fu.json"

STATUSES = {"value", "searched_and_empty"}
STATES = {"steady_state", "single_dose"}
FU_FLAGS = {"fu_missing", "fu_single_measurement", "fu_sources_disagree_gt3x"}


def _load_script():
    spec = importlib.util.spec_from_file_location("v2_cmax", ROOT / "scripts" / "v2_cmax.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_script()


@pytest.fixture(scope="module")
def doc():
    return json.loads(CMAX_FILE.read_text())


@pytest.fixture(scope="module")
def universe():
    return {d["inn"] for d in json.loads(DRUGS_FILE.read_text())["drugs"]}


@pytest.fixture(scope="module")
def fu_rows():
    return {r["inn"]: r for r in json.loads(FU_FILE.read_text())["drugs"]}


def test_schema_and_universe(doc, universe):
    rows = doc["drugs"]
    inns = [r["inn"] for r in rows]
    assert len(inns) == len(set(inns))
    assert set(inns) <= universe
    assert {i for i in universe if not v2_heldout(i)} <= set(inns)
    assert doc["n_drugs"] == len(universe)
    for r in rows:
        assert r["status"] in STATUSES, r["inn"]
        if r["status"] == "searched_and_empty":
            assert r["cmax"] is None
            assert r["searched"], r["inn"]
            assert r.get("cmax_unbound_nM") is None


def test_loader_keys_by_inn(doc):
    loaded = load_v2_human_cmax()
    assert set(loaded) == {r["inn"] for r in doc["drugs"]}


def test_no_value_without_source_snippet(doc):
    for r in doc["drugs"]:
        if r["status"] != "value":
            continue
        c = r["cmax"]
        assert c["snippet"] and c["snippet"].strip(), r["inn"]
        src = c["source"]
        assert src["type"] in {"fda_label", "pubmed"}, r["inn"]
        if src["type"] == "fda_label":
            assert src["set_id"] and src["section"], r["inn"]
        else:
            assert re.fullmatch(r"\d+", str(src["pmid"])), r["inn"]
        assert c["state"] in STATES, r["inn"]
        assert c["dose"] and c["regimen"] and c["population"], r["inn"]
        if c["statistic"] == "range_geomean":
            lo, hi = c["value_low_literal"], c["value_high_literal"]
            assert lo in c["snippet"] and hi in c["snippet"], r["inn"]
            assert math.isclose(
                c["value"], math.sqrt(float(lo.replace(",", "")) * float(hi.replace(",", ""))),
                rel_tol=1e-3)
        else:
            assert c["value_literal"] in c["snippet"], r["inn"]
            assert float(c["value_literal"].replace(",", "")) == c["value"], r["inn"]


def test_snippets_verbatim_in_cached_source(doc, mod):
    """Re-run the build verifier when the source cache is on this machine."""
    if not mod.LABELS.exists() or not mod.PUBMED_DIR.exists():
        pytest.skip("source cache not present")
    labels = mod.load_labels()
    for r in doc["drugs"]:
        if r["status"] != "value":
            continue
        c = r["cmax"]
        rec = {"inn": r["inn"], **{k: c[k] for k in (
            "value", "unit", "value_literal", "statistic", "snippet", "dose_snippet",
            "value_low_literal", "value_high_literal")}, "source": c["source"]}
        mod.verify(rec, labels)


@pytest.mark.parametrize("value,unit,mw,ng,nm", [
    (1.0, "µg/mL", 300.0, 1000.0, 1000.0 / 300.0 * 1e3),
    (1.0, "mcg/mL", 300.0, 1000.0, 1000.0 / 300.0 * 1e3),
    (1.0, "mg/L", 300.0, 1000.0, 1000.0 / 300.0 * 1e3),
    (500.0, "pg/mL", 250.0, 0.5, 2.0),
    (10.0, "ng/mL", 200.0, 10.0, 50.0),
    (10.0, "nmol/L", 200.0, 2.0, 10.0),
    (2.0, "µmol/L", 150.0, 300.0, 2000.0),
    (100.0, "pmol/L", 400.0, 0.04, 0.1),
])
def test_unit_conversions(mod, value, unit, mw, ng, nm):
    assert math.isclose(mod.to_ng_ml(value, unit, mw), ng, rel_tol=1e-9)
    assert math.isclose(mod.to_nm(value, unit, mw), nm, rel_tol=1e-9)


def test_unknown_unit_rejected(mod):
    with pytest.raises(ValueError):
        mod.to_nm(1.0, "mg", 100.0)


def test_unit_in_variants(mod):
    assert mod.unit_in("µg/mL", "peak of 2.1 mcg/mL")
    assert mod.unit_in("ng/mL", "Cmax 35 ng ml-1")
    assert mod.unit_in("nmol/L", "Cmax 12 nM")
    assert not mod.unit_in("ng/mL", "Cmax 12 nM")


def test_literal_whole_token(mod):
    assert mod._literal_in("5", "Cmax was 5 ng/mL")
    assert not mod._literal_in("5", "Cmax was 50 ng/mL")
    assert not mod._literal_in("5", "Cmax was 0.5 ng/mL")


def test_stored_conversions_consistent(doc, mod):
    for r in doc["drugs"]:
        if r["status"] != "value":
            continue
        c, mw = r["cmax"], r["mw_parent"]
        assert mw and mw > 0, r["inn"]
        assert math.isclose(c["ng_per_ml"], mod.to_ng_ml(c["value"], c["unit"], mw), rel_tol=1e-5)
        assert math.isclose(c["nM"], mod.to_nm(c["value"], c["unit"], mw), rel_tol=1e-5)
        # round trip
        assert math.isclose(c["nM"], c["ng_per_ml"] / mw * 1e3, rel_tol=1e-5), r["inn"]


def test_unbound_cmax_and_fu_flags(doc, fu_rows, mod):
    for r in doc["drugs"]:
        if r["status"] != "value":
            continue
        assert set(r["fu_flags"]) <= FU_FLAGS
        fu, flags = mod.fu_status(fu_rows.get(r["inn"]))
        assert r["fu"] == fu and r["fu_flags"] == flags, r["inn"]
        if fu is None:
            assert "fu_missing" in r["fu_flags"]
            assert r["cmax_unbound_nM"] is None
        else:
            assert 0 < fu <= 1
            assert math.isclose(r["cmax_unbound_nM"], r["cmax"]["nM"] * fu, rel_tol=1e-5)
            assert math.isclose(r["cmax_unbound_ng_per_ml"], r["cmax"]["ng_per_ml"] * fu,
                                rel_tol=1e-5)


def test_coverage_count(doc):
    rows = doc["drugs"]
    train_values = sum(1 for r in rows if r["status"] == "value" and not v2_heldout(r["inn"]))
    assert train_values == doc["n_train_with_value"]
    values = sum(1 for r in rows if r["status"] == "value")
    assert values <= doc["n_with_value"]
    if len(rows) == doc["n_drugs"]:
        assert values == doc["n_with_value"]
    assert doc["n_train_with_value"] >= 100
