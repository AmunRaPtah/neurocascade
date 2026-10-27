"""P29.6: microsomal-binding (fu_mic) and blood:plasma-ratio (Rb) corrections
for the P29.2 well-stirred IVIVE clearance arm.

The arm that omitted fu_mic under-predicted lipophilic bases (haloperidol
18-fold, desipramine 8.7, chlorpromazine 7.5).  P29.6 adds a sourced logP,
pKa/ionisation class and (measured-where-public) Rb per drug, computes fu_mic
with the Hallifax-Houston 2006 quadratic at a declared incubation protein
concentration, and rescores.  Development is train-only; the host scores
held-out.  These tests pin the TRAIN result (computable here), the sourcing,
and that the arm now beats the train-mean floor on the same drugs.
"""

import math

import pytest

from neurocascade.benchmark import load_split
import neurocascade.benchmark.human_pk_translation as T


# --- fu_mic equation --------------------------------------------------------
def test_fu_mic_equation_matches_hallifax_houston():
    # fu_mic = 1 / (1 + C * 10**(0.072*x^2 + 0.067*x - 1.126))
    for logp, C in [(4.43, 0.5), (0.0, 1.0), (3.0, 0.5), (5.0, 0.1)]:
        expected = 1.0 / (1.0 + C * 10 ** (0.072 * logp * logp + 0.067 * logp - 1.126))
        assert T.fu_mic(logp, C) == pytest.approx(expected)
    # monotone: more lipophilic -> more bound -> smaller fu_mic; result in (0,1]
    assert T.fu_mic(5.0, 0.5) < T.fu_mic(2.0, 0.5) < T.fu_mic(0.0, 0.5) <= 1.0
    assert 0.0 < T.fu_mic(5.0, 0.5)


def test_fu_mic_coefficients_and_protein_are_sourced():
    cfg = T.load_physchem()["fu_mic"]
    assert cfg["coefficients"] == {"a2": 0.072, "a1": 0.067, "a0": -1.126}
    assert cfg["equation_pmids"] == ["16552024", "12433825"]
    assert cfg["protein_mg_mL"] == 0.5
    assert "31259741" in cfg["protein_pmid"]  # Sloczynska 2019, HLM 0.5 mg/mL


# --- per-drug sourcing ------------------------------------------------------
TRAIN_IVIVE = [p.drug for p in T.score_train()["cl"].predictions]
MEASURED_RB = {"haloperidol": 0.79, "clozapine": 0.88, "risperidone": 0.67}


def test_every_train_cl_drug_has_a_sourced_physchem_record():
    drugs = T.load_physchem()["drugs"]
    for inn in TRAIN_IVIVE:
        rec = drugs[inn]
        assert rec["logp"] and rec["logp_source"]
        assert rec["pka_source"] and rec["ionization_class"] in {"base", "neutral", "acid"}
        assert rec["rb"] > 0 and rec["rb_basis"] in {"measured", "default"}
        assert rec["rb_source"]


def test_measured_rb_records_cite_a_pmid_and_defaults_say_searched_and_empty():
    drugs = T.load_physchem()["drugs"]
    for inn, rb in MEASURED_RB.items():
        rec = drugs[inn]
        assert rec["rb_basis"] == "measured"
        assert rec["rb"] == pytest.approx(rb)
        assert rec["rb_pmid"] and rec["rb_pmid"].isdigit()
    for inn in TRAIN_IVIVE:
        rec = drugs[inn]
        if rec["rb_basis"] == "default":
            assert rec["rb"] == 1.0  # base/neutral default
            assert "searched-and-empty" in rec["rb_source"]
    # no train IVIVE drug is an acid, so no acid Rb default (0.55) is ever used
    assert all(drugs[inn]["ionization_class"] != "acid" for inn in TRAIN_IVIVE)


def test_binding_inputs_apply_measured_rb_and_default_otherwise():
    for inn, rb in MEASURED_RB.items():
        fm, got_rb, refs = T._binding_inputs(inn)
        assert got_rb == pytest.approx(rb)
        assert 0 < fm < 1 and any("fu_mic" in r for r in refs)
    fm, got_rb, _ = T._binding_inputs("trimipramine")  # defaulted base
    assert got_rb == 1.0


# --- the result: beats the train-mean floor on the same drugs --------------
def test_binding_corrected_train_cl_beats_same_drug_mean_baseline():
    metric = T.score_train_binding_corrected()["cl"]
    baseline = T.same_drugs_mean_baseline(metric.predictions)
    assert metric.n == 15
    # acceptance gate: train LOO CL GMFE below the train-mean baseline
    assert metric.geometric_mean_fold_error < baseline
    assert baseline == pytest.approx(2.8168, abs=5e-3)


def test_binding_corrected_train_cl_is_pinned():
    metric = T.score_train_binding_corrected()["cl"]
    assert metric.geometric_mean_fold_error == pytest.approx(1.9113, abs=5e-3)
    assert metric.within_2_fold == pytest.approx(11 / 15)
    assert metric.within_3_fold == pytest.approx(12 / 15)
    # GMFE is the geometric mean of the per-drug fold errors
    assert metric.geometric_mean_fold_error == pytest.approx(
        math.exp(sum(math.log(p.fold_error) for p in metric.predictions) / metric.n)
    )


def test_lipophilic_bases_improve_before_vs_after():
    before = {p.drug: p.fold_error for p in T.score_train()["cl"].predictions}
    after = {p.drug: p.fold_error for p in T.score_train_binding_corrected()["cl"].predictions}
    for inn, lo, hi in [("haloperidol", 3.0, 7.0), ("chlorpromazine", 1.0, 2.5),
                        ("desipramine", 4.0, 7.0)]:
        assert after[inn] < before[inn]          # each named base improves
        assert lo <= after[inn] <= hi            # to the pinned post-fix fold error
    # the headline 18-fold haloperidol miss is gone
    assert before["haloperidol"] > 18 and after["haloperidol"] < 6


def test_binding_corrected_predictions_stay_below_hepatic_blood_flow():
    for p in T.score_train_binding_corrected()["cl"].predictions:
        assert p.predicted < T.CONSTANTS.q_h_ml_min_kg
        assert p.input_references  # fu_mic / Rb / logP citations attached


# --- frozen P29.2 arm must be untouched ------------------------------------
def test_frozen_p29_2_cl_arm_is_unchanged():
    # P29.6 adds a binding-corrected arm; it must NOT move the frozen result
    # that conformal_bands.json and the pinned P29.2 tests depend on.
    assert T.score_train()["cl"].geometric_mean_fold_error == pytest.approx(
        2.854830092083889)
    assert T.fit_translation_models().cl_bias_train == pytest.approx(
        0.3512706425906185)


# --- held-out: host scores; sandbox reports searched-and-empty -------------
def test_heldout_binding_corrected_handles_absent_truth():
    status = T.heldout_binding_corrected_status()
    # inputs exist for the held-out drugs with an IVIVE path, so the host can
    # apply the frozen P29.6 method
    assert set(status["heldout_drugs_with_physchem_inputs"]) == {
        "alprazolam", "diazepam", "fluphenazine", "imipramine",
        "nomifensine", "venlafaxine"}
    heldout = set(load_split()["heldout"])
    report = T.score_heldout_binding_corrected()
    if status["scored_here"]:            # host run, truth present
        assert report["cl"].n == len(status["scored_here"])
        for p in report["cl"].predictions:
            assert p.drug in heldout and p.predicted > 0
    else:                                # this sandbox, truth removed
        assert "cl" not in report
        assert status["searched_and_empty_truth_absent"] == \
            status["heldout_drugs_with_physchem_inputs"]
