"""P30.3: split-conformal bands, their LODO coverage, the band API, and the
held-out coverage path the host runs against restored truth."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurocascade.benchmark import conformal, load_split
from neurocascade.benchmark.v2 import v2_partition


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen():
    return conformal.load_bands()


@pytest.fixture(scope="module")
def recomputed():
    return conformal.calibrate()


@pytest.fixture(scope="module")
def heldout(request, frozen):
    request.config._nc_p30_3_ran = True
    return conformal.heldout_coverage(frozen)


# ---- the arithmetic -------------------------------------------------------
def test_conformal_quantile_uses_the_finite_sample_rank():
    scores = [float(i) for i in range(1, 11)]
    assert conformal.conformal_quantile(scores, 0.80) == 9.0  # ceil(11 * 0.8) = 9
    assert conformal.conformal_quantile(scores, 0.95) is None  # rank 11 > n: unbounded
    # n = 19 at 95% is rank 19 exactly, not 20 from float round-off.
    assert conformal.conformal_quantile(range(1, 20), 0.95) == 19.0
    assert conformal.conformal_quantile([], 0.80) is None


def test_lodo_coverage_compares_each_score_with_the_others():
    result = conformal.lodo_coverage([1.0, 2.0, 3.0, 4.0, 100.0], 0.80)
    # n-1 = 4 others: rank ceil(5 * 0.8) = 4, the largest of the four.
    assert result == {"covered": 4, "n": 5, "rate": 0.8, "n_unbounded": 0}


def test_label_set_includes_every_conforming_label():
    assert conformal.label_set(0.9, 0.2) == (1,)
    assert conformal.label_set(0.1, 0.2) == (0,)
    assert conformal.label_set(0.5, 0.6) == (0, 1)
    assert conformal.label_set(0.5, 0.4) == ()
    assert conformal.label_set(0.5, None) == (0, 1)


# ---- the frozen bands -----------------------------------------------------
def test_frozen_bands_are_exactly_what_train_calibration_produces(frozen, recomputed):
    assert set(frozen["quantities"]) == set(recomputed["quantities"])
    for name, entry in recomputed["quantities"].items():
        old = frozen["quantities"][name]
        assert old["status"] == entry["status"], name
        if entry["status"] != "calibrated":
            continue
        assert old["calibration_drugs"] == entry["calibration_drugs"], name
        assert old["lodo_coverage"] == entry["lodo_coverage"], name
        for level, q in entry["quantiles"].items():
            assert old["quantiles"][level] == (None if q is None else pytest.approx(q)), name


def test_calibration_never_reads_a_heldout_drug(frozen):
    p28_train = set(load_split()["train"])
    v2_train = set(v2_partition()[0])
    for name, entry in frozen["quantities"].items():
        if entry["status"] != "calibrated":
            continue
        train = v2_train if entry["source_task"] == "P33.3" else p28_train
        assert set(entry["calibration_drugs"]) <= train, name


def test_every_product_quantity_is_banded_or_listed_pending(frozen):
    q = frozen["quantities"]
    for name in ("human_cl_ivive", "human_cl_v2_train_mean", "human_vss_v2_train_mean"):
        assert q[name]["status"] == "calibrated"
    for effect in ("eps", "sedation", "weight_gain", "anticholinergic",
                   "orthostasis", "sexual_dysfunction", "seizure"):
        assert q[f"effect_probability:{effect}"]["status"] == "calibrated"
    assert q["target_engagement_dose"]["status"] == "pending"
    assert q["occupancy"]["status"] == "pending"
    assert q["human_vss_ivive"]["status"] == "searched-and-empty"


def test_frozen_band_values(frozen):
    q = frozen["quantities"]
    ivive = q["human_cl_ivive"]
    assert ivive["n_calibration"] == 15
    assert ivive["fold_bands"]["0.80"] == pytest.approx(2.28, abs=0.01)  # P29.6 fu_mic tightened from 7.55
    assert ivive["fold_bands"]["0.95"] is None  # n=15 cannot support a 95% band
    assert ivive["lodo_coverage"]["0.80"]["covered"] == 12
    cl = q["human_cl_v2_train_mean"]
    assert (cl["n_calibration"], round(cl["fold_bands"]["0.80"], 2), round(cl["fold_bands"]["0.95"], 2)) \
        == (109, 5.89, 23.85)
    vss = q["human_vss_v2_train_mean"]
    assert (vss["n_calibration"], round(vss["fold_bands"]["0.80"], 2), round(vss["fold_bands"]["0.95"], 2)) \
        == (107, 7.48, 12.32)
    assert q["effect_probability:eps"]["lodo_set_sizes"]["0.95"] == {"0": 0, "1": 1, "2": 29}


# ---- the band API ---------------------------------------------------------
def test_predict_with_bands_attaches_the_frozen_fold_band(frozen):
    out = conformal.predict_with_bands("human_cl_v2_train_mean", 4.0, frozen)
    f80 = frozen["quantities"]["human_cl_v2_train_mean"]["fold_bands"]["0.80"]
    assert out["bands"]["0.80"]["lower"] == pytest.approx(4.0 / f80)
    assert out["bands"]["0.80"]["upper"] == pytest.approx(4.0 * f80)
    assert out["provenance"]["source_task"] == "P33.3"
    assert out["unit"] == "mL/min/kg"


def test_predict_with_bands_reports_an_unbounded_band_rather_than_shrinking_it(frozen):
    out = conformal.predict_with_bands("human_cl_ivive", 3.0, frozen)
    assert out["bands"]["0.95"] == {"lower": 0.0, "upper": math.inf, "bounded": False}
    assert out["bands"]["0.80"]["bounded"] is True


def test_predict_with_bands_returns_label_sets_for_probabilities():
    bands = {"quantities": {"effect_probability:x": {
        "status": "calibrated", "kind": "label_set", "unit": "probability",
        "source_task": "P30.1", "method": "m", "n_calibration": 20, "underpowered": True,
        "quantiles": {"0.80": 0.3, "0.95": 0.8},
    }}}
    out = conformal.predict_with_bands("effect_probability:x", 0.75, bands)
    assert out["bands"]["0.80"]["labels"] == [1]
    assert out["bands"]["0.95"]["labels"] == [0, 1]
    assert out["provenance"]["underpowered"] is True
    with pytest.raises(ValueError):
        conformal.predict_with_bands("effect_probability:x", 1.5, bands)


def test_predict_with_bands_refuses_pending_quantities(frozen):
    for name in ("target_engagement_dose", "occupancy", "human_vss_ivive"):
        with pytest.raises(conformal.UnbandedQuantityError):
            conformal.predict_with_bands(name, 1.0, frozen)
    assert set(conformal.pending_quantities(frozen)) == {
        "target_engagement_dose", "occupancy", "human_vss_ivive"}
    with pytest.raises(KeyError):
        conformal.predict_with_bands("not_a_quantity", 1.0, frozen)


# ---- held-out coverage: synthetic rows drive every host branch -------------
def test_heldout_v2_branch_scores_restored_rows(monkeypatch):
    monkeypatch.setattr(conformal, "v2_partition", lambda: (("a",), ("w", "x", "y", "z")))
    monkeypatch.setattr(conformal, "load_v2_human_pk", lambda: {
        "x": {"vss": {"value": 2.0}}, "y": {"vss": None}, "w": {"vss": {"value": 0.0}}})
    entry = {"quantiles": {"0.80": math.log10(2.0) + 1e-12, "0.95": 0.1}, "frozen_point": 1.0}
    block = conformal._heldout_v2_mean(entry, "vss")
    assert block["status"] == "scored"
    assert block["coverage"]["0.80"] == {"covered": 1, "n": 1, "rate": 1.0}
    assert block["coverage"]["0.95"] == {"covered": 0, "n": 1, "rate": 0.0}
    assert block["missing_truth"] == ["z"] and block["not_scoreable"] == ["w", "y"]


def test_heldout_ivive_branch_uses_the_p28_heldout_split(monkeypatch):
    monkeypatch.setattr(conformal, "load_split", lambda: {"train": ["a"], "heldout": ["x", "y", "z"]})
    monkeypatch.setattr(conformal._pk, "fit_translation_models", lambda: None)

    def fake_truth(inn):
        if inn == "z":
            raise KeyError(inn)  # a stripped held-out truth file
        return {}

    def fake_score(drugs, model=None):
        (inn,) = drugs
        if inn == "y":
            return {}
        return {"cl": SimpleNamespace(predictions=[SimpleNamespace(predicted=10.0, observed=1.0)])}

    monkeypatch.setattr(conformal, "load_truth_record", fake_truth)
    monkeypatch.setattr(conformal._pk, "score", fake_score)
    block = conformal._heldout_ivive_cl({"quantiles": {"0.80": 0.5, "0.95": None}})
    assert block["coverage"]["0.80"]["covered"] == 0  # 10-fold miss, band 10**0.5
    assert block["coverage"]["0.95"]["covered"] == 1  # unbounded
    assert block["missing_truth"] == ["z"] and block["not_scoreable"] == ["y"]


def test_heldout_effect_branch_scores_labels_from_the_frozen_model(monkeypatch):
    monkeypatch.setattr(conformal, "load_split", lambda: {"train": ["a"], "heldout": ["x", "y"]})
    model = SimpleNamespace(predict_proba=lambda features: features["p"])
    monkeypatch.setattr(conformal._effects, "fit_effect_models", lambda: ({"eps": model}, ("a",)))
    labels = {"x": 1, "y": 0}
    monkeypatch.setattr(conformal._effects, "effect_labels",
                        lambda d: (SimpleNamespace(effect="eps", label=labels[d]),))
    monkeypatch.setattr(conformal._effects, "occupancy_profile",
                        lambda d: SimpleNamespace(features={"p": 0.9}))
    bands = {"quantities": {"effect_probability:eps": {
        "status": "calibrated", "quantiles": {"0.80": 0.2, "0.95": 0.95}}}}
    block = conformal._heldout_effects(bands)["effect_probability:eps"]
    # x: 1 - 0.9 = 0.1 covered at both; y: 1 - 0.1 = 0.9 covered only at 95%.
    assert block["coverage"]["0.80"] == {"covered": 1, "n": 2, "rate": 0.5}
    assert block["coverage"]["0.95"] == {"covered": 2, "n": 2, "rate": 1.0}


# ---- held-out coverage on this checkout's real data ------------------------
def test_heldout_coverage_accounts_for_every_heldout_drug(heldout):
    p28 = len(load_split()["heldout"])
    v2 = len(v2_partition()[1])
    for name, block in heldout["quantities"].items():
        n = block["coverage"]["0.80"]["n"]
        expected = v2 if "v2" in name else p28
        assert block["n_heldout"] == expected, name
        if name.startswith("effect_probability") or "v2" in name:
            # a drug can be present yet unlabelled for one effect / one field
            assert len(block["missing_truth"]) + len(block["not_scoreable"]) <= expected
        else:
            assert n + len(block["missing_truth"]) + len(block["not_scoreable"]) == expected, name
        assert (block["status"] == "scored") == (n > 0), name
    if heldout["status"] == "searched-and-empty":
        # session checkout: truth stripped, every held-out drug named, none scored
        assert heldout["max_n_scored"] == 0
        assert len(heldout["quantities"]["human_cl_ivive"]["missing_truth"]) == p28
        assert len(heldout["quantities"]["human_vss_v2_train_mean"]["missing_truth"]) == v2
    assert set(heldout["pending"]) == {"target_engagement_dose", "occupancy", "human_vss_ivive"}
    assert conformal.heldout_summary_lines(heldout)


# ---- the ledger and the results file read from the frozen bands ------------
def test_ledger_row_quotes_the_frozen_lodo_coverage(frozen):
    ledger = (ROOT / "docs" / "validation-ledger.md").read_text()
    row = next(line for line in ledger.splitlines() if line.startswith("| V20 "))
    for name, entry in frozen["quantities"].items():
        if entry["status"] != "calibrated":
            assert name in row, name
            continue
        for level in ("0.80", "0.95"):
            c = entry["lodo_coverage"][level]
            assert f"{c['covered']}/{c['n']}" in row, (name, level)


def test_results_file_reports_the_largest_scored_calibration_set(frozen):
    result = json.loads((ROOT / "results" / "P30.3.json").read_text())
    n = max(e["n_calibration"] for e in frozen["quantities"].values() if e["status"] == "calibrated")
    assert result["n_scored"] == n == 109
    assert result["heldout"]["status"] == "host-pending"
