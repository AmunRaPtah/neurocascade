"""P29.2: mechanistic human PK translation must be train-only and data-backed.

The method is textbook PK with no fitted log-log regressions: well-stirred
IVIVE for CL, fixed-exponent (0.75) single-species rat allometry as the CL
fallback, and Oie-Tozer fu-ratio Vss.  The only empirical quantity is one
optional TRAIN geometric-mean bias factor, reported separately.
"""

import math

import pytest

from neurocascade.benchmark import load_split
import neurocascade.benchmark.human_pk_translation as translation
from neurocascade.benchmark.human_pk_translation import (
    CONSTANTS,
    fit_translation_models,
    score,
    score_heldout,
    score_train,
    train_geomean_baseline,
)


# --- train / held-out isolation -------------------------------------------
def test_fitted_model_names_only_training_drugs():
    split = load_split()
    model = fit_translation_models()
    heldout = set(split["heldout"])
    assert model.fit_drugs
    assert not heldout.intersection(model.fit_drugs)


def test_fit_is_explicitly_independent_of_heldout_names():
    split = load_split()
    baseline = fit_translation_models(split["train"])
    # Feeding held-out names in must visibly change the fitted drug set, so a
    # leak cannot hide: the bias factor is estimated only on drugs it names.
    leaked = fit_translation_models(split["train"] + split["heldout"])
    assert set(leaked.fit_drugs) != set(baseline.fit_drugs)
    assert set(leaked.fit_drugs) - set(baseline.fit_drugs)  # held-out drugs entered


def test_default_fit_refuses_to_read_heldout_records(monkeypatch):
    heldout = set(load_split()["heldout"])
    original_adme = translation.load_adme_record
    original_truth = translation.load_truth_record

    def guarded_adme(inn):
        assert inn not in heldout
        return original_adme(inn)

    def guarded_truth(inn):
        assert inn not in heldout
        return original_truth(inn)

    monkeypatch.setattr(translation, "load_adme_record", guarded_adme)
    monkeypatch.setattr(translation, "load_truth_record", guarded_truth)
    fit_translation_models()


# --- metric definitions -----------------------------------------------------
def test_heldout_metrics_have_fixed_fold_definitions():
    report = score_heldout()
    assert "cl" in report  # CL is scoreable on held-out
    for metric in report.values():
        assert metric.n == len(metric.predictions) > 0
        assert metric.geometric_mean_fold_error == pytest.approx(
            math.exp(
                sum(math.log(p.fold_error) for p in metric.predictions) / metric.n
            )
        )
        assert metric.within_2_fold == pytest.approx(
            sum(p.fold_error <= 2 for p in metric.predictions) / metric.n
        )
        assert metric.within_3_fold == pytest.approx(
            sum(p.fold_error <= 3 for p in metric.predictions) / metric.n
        )


# --- pinned results ---------------------------------------------------------
def test_heldout_cl_is_the_pinned_mechanistic_result():
    report = score_heldout()
    assert report["cl"].n == 5
    assert report["cl"].geometric_mean_fold_error == pytest.approx(3.420223149363284)
    assert report["cl"].within_2_fold == pytest.approx(0.2)
    assert report["cl"].within_3_fold == pytest.approx(0.6)


def test_train_cl_is_the_pinned_no_fit_result():
    report = score_train()
    assert report["cl"].n == 15
    assert report["cl"].geometric_mean_fold_error == pytest.approx(2.854830092083889)
    assert report["cl"].within_2_fold == pytest.approx(0.4666666666666667)
    assert report["cl"].within_3_fold == pytest.approx(0.6)


def test_train_geomean_baselines_are_pinned():
    cl_gmfe, cl_n = train_geomean_baseline("cl")
    vss_gmfe, vss_n = train_geomean_baseline("vss")
    assert (cl_n, vss_n) == (32, 32)
    assert cl_gmfe == pytest.approx(3.0824491254020803)
    assert vss_gmfe == pytest.approx(3.8885263983028184)
    # the mechanistic CL arm must beat the geomean floor on train
    assert score_train()["cl"].geometric_mean_fold_error < cl_gmfe


def test_optional_bias_factor_is_train_only_and_reported_separately():
    model = fit_translation_models()
    assert model.cl_bias_train == pytest.approx(0.3512706425906185)
    assert model.cl_bias_correction == pytest.approx(2.8468077850884637)
    corrected = score_heldout(corrected=True)
    assert corrected["cl"].geometric_mean_fold_error == pytest.approx(2.7329221252413696)


def test_vss_arm_has_no_compatible_data():
    # Honest limitation: no benchmark drug pairs clean rat plasma Vss with
    # human Vss truth, so the Oie-Tozer arm yields zero scoreable predictions.
    assert "vss" not in score_train()
    assert "vss" not in score_heldout()


# --- mechanistic guarantees -------------------------------------------------
def test_well_stirred_cl_never_exceeds_hepatic_blood_flow():
    # The well-stirred model caps CL_H strictly below Q_H; every IVIVE
    # prediction must honour that.  Allometric-fallback rows are excluded.
    for drugs in (load_split()["train"], load_split()["heldout"]):
        report = score(drugs)
        for p in report.get("cl", []).predictions if "cl" in report else []:
            if p.method.startswith("well-stirred IVIVE"):
                assert p.predicted < CONSTANTS.q_h_ml_min_kg


def test_microsome_and_hepatocyte_unit_conversions():
    k = CONSTANTS
    # microsomal CLint 1 mL/min/g -> 1 * 40 mg/g / 1000 * 21.4 g/kg mL/min/kg
    rec_mic = {"fields": {
        "clint": {"measurements": [{
            "species": "human", "relation": "=", "value": 1.0,
            "units": translation.MICROSOME_CLINT_UNITS, "matrix": "microsomes",
            "description": "liver microsomes", "ref": "X"}]},
        "plasma_fu": {"summary": [{"species": "human", "value": 1.0}],
                      "measurements": []},
    }}
    feat = translation._liver_clint(rec_mic)
    assert feat.detail.startswith("microsomes")
    # hepatocyte CLint 1 uL/min/1e6cells -> 1 *1e-3 *99e6/1e6 *21.4 mL/min/kg
    rec_hep = {"fields": {
        "clint": {"measurements": [{
            "species": "human", "relation": "=", "value": 1.0,
            "units": translation.HEPATOCYTE_CLINT_UNITS, "matrix": "hepatocytes",
            "description": "hepatocytes", "ref": "Y"}]}}}
    feat_h = translation._liver_clint(rec_hep)
    assert feat_h.detail.startswith("hepatocytes")
    # per-kg "CLint" rows (already a CL output) and S9 are rejected
    rec_bad = {"fields": {"clint": {"measurements": [{
        "species": "human", "relation": "=", "value": 5.0,
        "units": "mL.min-1.kg-1", "matrix": "liver", "description": "clint", "ref": "Z"}]}}}
    assert translation._liver_clint(rec_bad) is None


def test_clint_grouping_separates_organ_and_bound_unbound():
    # Liver vs intestinal microsomes, and total vs unbound CLint, must not be
    # pooled.  Here liver-total must be chosen and the intestinal/unbound rows
    # dropped, so the result is the median of the two liver-total references.
    rec = {"fields": {"clint": {"measurements": [
        {"species": "human", "relation": "=", "value": 10.0,
         "units": translation.MICROSOME_CLINT_UNITS, "matrix": "microsomes",
         "description": "liver microsomes", "ref": "a"},
        {"species": "human", "relation": "=", "value": 40.0,
         "units": translation.MICROSOME_CLINT_UNITS, "matrix": "microsomes",
         "description": "liver microsomes", "ref": "b"},
        {"species": "human", "relation": "=", "value": 999.0,
         "units": translation.MICROSOME_CLINT_UNITS, "matrix": "microsomes",
         "description": "intestinal microsomes", "ref": "c"},
        {"species": "human", "relation": "=", "value": 999.0,
         "units": translation.MICROSOME_CLINT_UNITS, "matrix": "microsomes",
         "description": "liver microsomes, unbound", "ref": "d"},
    ]}}}
    feat = translation._liver_clint(rec)
    assert feat.detail == "microsomes/total"
    assert feat.value == pytest.approx(25.0)


def test_clint_drops_a_unit_suspect_reference():
    # imipramine's ChEMBL rows include one paper at ~0.045 next to 7.8-79.5:
    # mL/min/mg stored under the uL/min/mg-equivalent unit, 1000-fold low.
    rows = [("a", 13.49), ("b", 0.0511), ("b", 0.0396), ("c", 79.5), ("d", 7.8)]
    rec = {"fields": {"clint": {"measurements": [
        {"species": "human", "relation": "=", "value": v,
         "units": translation.MICROSOME_CLINT_UNITS, "matrix": "microsomes",
         "description": "human liver microsomes", "ref": ref}
        for ref, v in rows]}}}
    feat = translation._liver_clint(rec)
    assert feat.references == ("a", "c", "d")
    assert feat.value == pytest.approx(13.49)


def test_ao_cleared_drug_is_excluded_from_ivive():
    assert "ziprasidone" in translation.NON_MICROSOMAL_CLEARANCE
    rec = translation.load_adme_record("ziprasidone")
    assert translation._ivive_cl(rec, "ziprasidone", CONSTANTS) is None


def test_predictions_carry_citations_and_only_use_their_own_drug():
    split = load_split()
    report = score_heldout()
    heldout = set(split["heldout"])
    for metric in report.values():
        for p in metric.predictions:
            assert p.drug in heldout
            assert p.input_references
            assert p.predicted > 0
            assert p.observed > 0
