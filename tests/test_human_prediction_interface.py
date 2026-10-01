"""P24.1: the public prediction interface, and its refusals.

Refusal is a feature of this interface, so it is tested at least as hard as the
success path: every drug in the P22.2 denominator is pinned to the exact set of
layers it is missing, each layer is removed in turn from a drug that has all of
them to show the refusal names that drug and that layer (and never borrows a
neighbour's record), and a refusal is shown to happen before anything is
integrated.
"""
import dataclasses
import inspect
import math

import numpy as np
import pytest

from neurocascade.human import d2_ec50, dose_to_occupancy, kp_uu
from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.occupancy_validation import (
    CLINICAL_OCCUPANCY_THRESHOLDS,
    CLINICAL_WINDOW_PROVENANCE,
    OCCUPANCY_IS_NOT_THE_OUTCOME,
    ClinicalOccupancyThreshold,
    steady_state_occupancy,
)
import neurocascade.human.prediction_interface as api
from neurocascade.human.prediction_interface import (
    LAYERS,
    REGIMENS,
    PredictionRefused,
    predict,
    refusal_layers,
    supported_drugs,
)

# ---------------------------------------------------------------------------
# success paths
# ---------------------------------------------------------------------------


def test_exactly_the_two_drugs_with_every_layer_are_supported():
    assert supported_drugs() == ("haloperidol", "olanzapine")
    assert refusal_layers("haloperidol") == ()
    assert refusal_layers("olanzapine") == ()


def test_haloperidol_steady_state_is_the_v4_chain_bit_for_bit():
    prediction = predict("haloperidol", 2.0, "steady-state-daily")
    reference = steady_state_occupancy(2.0, drug="haloperidol")
    np.testing.assert_array_equal(prediction.occupancy, reference.occupancy)
    np.testing.assert_array_equal(prediction.plasma_nM, reference.plasma_nM)
    np.testing.assert_array_equal(prediction.times_min, reference.times_min)
    # Ledger row V4: regimen-matched steady-state mean 0.5635 at 2 mg/day.
    assert prediction.mean_occupancy == pytest.approx(0.5635, abs=5e-5)
    assert prediction.statistic == "mean"
    assert prediction.statistic_occupancy == prediction.mean_occupancy
    # One 24 h interval at 5 min steps.
    assert prediction.times_min[0] == 0.0
    assert len(prediction.times_min) == 288


def test_haloperidol_single_dose_is_the_chain_peak():
    prediction = predict("haloperidol", 2.0, "single-dose")
    reference = predict_occupancy(2.0, drug="haloperidol", times_min=prediction.times_min)
    np.testing.assert_array_equal(prediction.occupancy, reference.occupancy)
    # Ledger row V4: single-dose peak 0.5253 at 2 mg.
    assert prediction.peak_occupancy == pytest.approx(0.5253, abs=5e-5)
    assert prediction.statistic == "peak"
    assert prediction.statistic_occupancy == prediction.peak_occupancy
    assert prediction.times_min[-1] == 72.0 * 60.0
    assert prediction.occupancy[0] == 0.0


def test_olanzapine_uses_its_own_constants_and_not_haloperidols():
    prediction = predict("olanzapine", 10.0, "steady-state-daily")
    row = d2_ec50.d2_plasma_ec50("olanzapine")
    assert prediction.emax == row.emax == 0.875
    assert prediction.ec50_nM == pytest.approx(16.93125, rel=1e-5)
    assert prediction.ec50_nM != predict("haloperidol", 10.0, "steady-state-daily").ec50_nM
    assert prediction.peak_occupancy < 0.875
    for layer in prediction.layers:
        for haloperidol_source in ("Pilla Reddy", "Nyberg", "Regenthal", "haloperidol"):
            assert haloperidol_source not in layer.citation, (layer.name, layer.citation)


def test_layers_are_the_consumed_ones_and_carry_provenance():
    for drug in supported_drugs():
        prediction = predict(drug, 5.0, "single-dose")
        assert [layer.name for layer in prediction.layers] == [
            "pk",
            "weight-scaling",
            "exposure",
            "occupancy",
        ]
        assert prediction.provenance == "measured"
        assert prediction.route == "PO"
        assert prediction.body_weight_kg == 70.0
        assert prediction.caveats == (OCCUPANCY_IS_NOT_THE_OUTCOME, CLINICAL_WINDOW_PROVENANCE)


def test_occupancy_rises_with_dose():
    for regimen in REGIMENS:
        low = predict("olanzapine", 5.0, regimen).statistic_occupancy
        high = predict("olanzapine", 20.0, regimen).statistic_occupancy
        assert 0.0 < low < high < 0.875


def test_every_sourced_threshold_is_placed():
    prediction = predict("haloperidol", 5.0, "steady-state-daily")
    assert [p.threshold for p in prediction.thresholds] == [
        t.occupancy for t in CLINICAL_OCCUPANCY_THRESHOLDS
    ] == [0.65, 0.72, 0.78, 0.775]
    for position in prediction.thresholds:
        assert position.statistic == "mean"
        assert position.occupancy == prediction.mean_occupancy
        assert position.margin == pytest.approx(position.occupancy - position.threshold)
        assert position.at_or_above == (position.occupancy >= position.threshold)
        expected = float(np.mean(prediction.occupancy >= position.threshold))
        assert position.fraction_of_time_at_or_above == expected
        assert not position.unreachable_at_any_dose


def test_haloperidol_window_edges_from_v8_land_on_the_thresholds():
    # V8: haloperidol W = 2.89-5.56 mg/day for 0.65 and 0.78 steady-state mean.
    low = predict("haloperidol", 2.89, "steady-state-daily").thresholds[0]
    high = predict("haloperidol", 5.56, "steady-state-daily").thresholds[2]
    assert low.outcome == "clinical response"
    assert high.outcome == "extrapyramidal side effects"
    assert abs(low.margin) < 0.002
    assert abs(high.margin) < 0.002


def test_kapur_thresholds_are_haloperidol_only_and_say_so():
    kapur = [t for t in CLINICAL_OCCUPANCY_THRESHOLDS if "10739409" in t.source]
    assert len(kapur) == 3
    for threshold in kapur:
        assert threshold.measured_in_drugs == ("haloperidol",)
        assert threshold.measured_regimen == "steady-state-daily"
    pooled = [t for t in CLINICAL_OCCUPANCY_THRESHOLDS if "21694629" in t.source]
    assert len(pooled) == 1
    assert pooled[0].measured_in_drugs is None
    assert pooled[0].measured_regimen is None


def test_haloperidol_against_kapur_is_same_drug_same_regimen():
    positions = predict("haloperidol", 3.0, "steady-state-daily").thresholds
    for position in positions[:3]:
        assert position.same_drug and position.same_regimen
        assert position.note == "measured in this drug under this regimen"
    pooled = positions[3]
    assert not pooled.same_drug and not pooled.same_regimen
    assert "composition was not read" in pooled.note


def test_olanzapine_against_kapur_is_labelled_a_cross_drug_transfer():
    for position in predict("olanzapine", 10.0, "steady-state-daily").thresholds[:3]:
        assert not position.same_drug
        assert position.same_regimen
        assert "measured in haloperidol-treated patients, not olanzapine" in position.note
        assert "cross-drug transfer" in position.note


def test_single_dose_is_labelled_a_regimen_mismatch_against_kapur():
    for position in predict("haloperidol", 2.0, "single-dose").thresholds[:3]:
        assert position.same_drug
        assert not position.same_regimen
        assert position.statistic == "peak"
        assert "measured under steady-state-daily, predicted under single-dose" in position.note


def test_a_threshold_above_the_drugs_own_emax_is_flagged_unreachable(monkeypatch):
    above_olanzapine_emax = ClinicalOccupancyThreshold(
        outcome="test", occupancy=0.9, as_printed="90%", source="s",
        source_locator="l", tracer=None, region="striatum", region_origin="r",
        statistical_basis="b",
    )
    monkeypatch.setattr(api, "CLINICAL_OCCUPANCY_THRESHOLDS", (above_olanzapine_emax,))
    (olanzapine,) = predict("olanzapine", 500.0, "steady-state-daily").thresholds
    (haloperidol,) = predict("haloperidol", 500.0, "steady-state-daily").thresholds
    assert olanzapine.unreachable_at_any_dose and not olanzapine.at_or_above
    assert not haloperidol.unreachable_at_any_dose and haloperidol.at_or_above


def test_summary_names_drug_regimen_and_every_threshold():
    text = predict("olanzapine", 10.0, "steady-state-daily").summary()
    assert "olanzapine 10 mg PO, steady-state-daily" in text
    for threshold in CLINICAL_OCCUPANCY_THRESHOLDS:
        assert threshold.outcome in text
    assert "cross-drug transfer" in text


# ---------------------------------------------------------------------------
# refusal paths
# ---------------------------------------------------------------------------

#: The P22.2 denominator plus amisulpride, each pinned to the exact layers it is
#: missing today. A change here is a change in what the model can predict and
#: must be made deliberately, with the data that justifies it.
EXPECTED_REFUSALS = {
    "risperidone": ("active-metabolite", "pk", "occupancy", "exposure"),
    # P25.2: Knights 2015 (literature popPK) clears the pk layer; the drug is
    # still refused at every other layer it was refused at before.
    "aripiprazole": ("active-metabolite", "occupancy", "exposure"),
    "paliperidone": ("pk", "occupancy", "exposure"),
    # P25.2: Zheng 2024 (literature popPK) clears the pk layer; no D2 EC50.
    "quetiapine": ("occupancy", "exposure"),
    "ziprasidone": ("pk", "occupancy", "exposure"),
    "clozapine": ("pk", "occupancy", "exposure"),
    "amisulpride": ("pk", "occupancy", "exposure"),
    # Has a Kp,uu record but no oral PK and no D2 EC50.
    "fentanyl": ("pk", "occupancy"),
    # A recorded Kp,uu value is itself a refusal: the no-partition design would
    # count brain penetration twice.
    "diazepam": ("pk", "occupancy", "exposure"),
    "not-a-drug": ("pk", "occupancy", "exposure"),
    # No case-folding: a guessed lookup is the first step to a wrong one.
    "Haloperidol": ("pk", "occupancy", "exposure"),
    " haloperidol": ("pk", "occupancy", "exposure"),
    "": ("pk", "occupancy", "exposure"),
}


@pytest.mark.parametrize("drug, layers", sorted(EXPECTED_REFUSALS.items()))
@pytest.mark.parametrize("regimen", REGIMENS)
def test_each_unsupported_drug_is_refused_naming_every_missing_layer(drug, layers, regimen):
    with pytest.raises(PredictionRefused) as caught:
        predict(drug, 5.0, regimen)
    error = caught.value
    assert error.drug == drug
    assert error.layers == layers
    assert tuple(layer for layer, _ in refusal_layers(drug)) == layers
    message = str(error)
    assert repr(drug) in message
    for layer in layers:
        assert "  %s: " % layer in message
    assert "Drugs with every layer: haloperidol, olanzapine." in message


def test_refusal_layers_are_reported_in_the_declared_order():
    assert LAYERS == ("active-metabolite", "pk", "occupancy", "exposure")
    for layers in EXPECTED_REFUSALS.values():
        assert list(layers) == [layer for layer in LAYERS if layer in layers]


def test_refusal_reasons_name_the_requested_drug_not_another():
    for drug in ("clozapine", "quetiapine", "paliperidone"):
        for layer, reason in refusal_layers(drug):
            assert drug in reason, (drug, layer, reason)


def test_active_metabolite_refusal_names_the_metabolite():
    reasons = dict(refusal_layers("risperidone"))
    assert "9-hydroxyrisperidone" in reasons["active-metabolite"]
    reasons = dict(refusal_layers("aripiprazole"))
    assert "dehydro-aripiprazole" in reasons["active-metabolite"]


def test_refusal_is_a_value_error_and_happens_before_any_integration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the chain ran for a refused drug")

    monkeypatch.setattr(api, "predict_occupancy", forbidden)
    monkeypatch.setattr(api, "steady_state_occupancy", forbidden)
    for drug in EXPECTED_REFUSALS:
        for regimen in REGIMENS:
            with pytest.raises(ValueError) as caught:
                predict(drug, 5.0, regimen)
            assert isinstance(caught.value, PredictionRefused)


def test_there_is_no_default_drug():
    parameters = inspect.signature(predict).parameters
    assert list(parameters) == ["drug", "dose_mg", "regimen"]
    assert all(p.default is inspect.Parameter.empty for p in parameters.values())
    with pytest.raises(TypeError):
        predict(dose_mg=2.0, regimen="single-dose")


@pytest.mark.parametrize("drug", [None, 1, ("haloperidol",), b"haloperidol"])
def test_a_drug_that_is_not_a_string_is_a_type_error(drug):
    with pytest.raises(TypeError, match="registry key"):
        predict(drug, 2.0, "single-dose")


@pytest.mark.parametrize("regimen", ["depot", "steady-state", "Single-dose", None, "IV"])
def test_an_unparameterised_regimen_is_refused(regimen):
    with pytest.raises(ValueError, match="regimen must be one of single-dose, steady-state-daily"):
        predict("haloperidol", 2.0, regimen)


@pytest.mark.parametrize("dose", [0.0, -1.0, math.nan, math.inf])
def test_a_non_positive_or_non_finite_dose_is_refused(dose):
    with pytest.raises(ValueError, match="positive and finite"):
        predict("haloperidol", dose, "single-dose")


@pytest.mark.parametrize("dose", [True, "2", None])
def test_a_dose_that_is_not_a_number_is_a_type_error(dose):
    with pytest.raises(TypeError, match="milligrams"):
        predict("haloperidol", dose, "single-dose")


# ---------------------------------------------------------------------------
# no cross-drug fallback: remove one layer from olanzapine at a time
# ---------------------------------------------------------------------------


def _forbid_chain(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the chain ran after a layer was removed")

    monkeypatch.setattr(api, "predict_occupancy", forbidden)
    monkeypatch.setattr(api, "steady_state_occupancy", forbidden)


def test_olanzapine_without_its_ec50_is_refused_at_occupancy_only(monkeypatch):
    without = {k: v for k, v in d2_ec50.D2_PLASMA_EC50.items() if k != "olanzapine"}
    monkeypatch.setattr(d2_ec50, "D2_PLASMA_EC50", without)
    _forbid_chain(monkeypatch)
    with pytest.raises(PredictionRefused) as caught:
        predict("olanzapine", 10.0, "steady-state-daily")
    assert caught.value.layers == ("occupancy",)
    assert "'olanzapine'" in dict(caught.value.missing)["occupancy"]
    assert supported_drugs() == ("haloperidol",)


def test_olanzapine_without_its_pk_set_is_refused_at_pk_only(monkeypatch):
    original = dose_to_occupancy.human_pk_records

    def without_olanzapine(drug=None, route=None, **kwargs):
        return tuple(
            r for r in original(drug, route, **kwargs) if r.drug != "olanzapine"
        )

    monkeypatch.setattr(dose_to_occupancy, "human_pk_records", without_olanzapine)
    monkeypatch.setattr(api, "human_pk_records", without_olanzapine)
    _forbid_chain(monkeypatch)
    with pytest.raises(PredictionRefused) as caught:
        predict("olanzapine", 10.0, "single-dose")
    assert caught.value.layers == ("pk",)
    assert "no human olanzapine PK record" in dict(caught.value.missing)["pk"]


def test_olanzapine_without_its_kp_uu_record_is_refused_at_exposure_only(monkeypatch):
    without = tuple(r for r in kp_uu.HUMAN_KP_UU_RECORDS if r.drug != "olanzapine")
    monkeypatch.setattr(kp_uu, "HUMAN_KP_UU_RECORDS", without)
    _forbid_chain(monkeypatch)
    with pytest.raises(PredictionRefused) as caught:
        predict("olanzapine", 10.0, "single-dose")
    assert caught.value.layers == ("exposure",)
    assert "no human olanzapine Kp,uu record" in dict(caught.value.missing)["exposure"]


def test_a_recorded_haloperidol_kp_uu_refuses_rather_than_double_counting(monkeypatch):
    records = tuple(
        dataclasses.replace(r, kp_uu=0.5) if r.drug == "haloperidol" else r
        for r in kp_uu.HUMAN_KP_UU_RECORDS
    )
    monkeypatch.setattr(kp_uu, "HUMAN_KP_UU_RECORDS", records)
    with pytest.raises(PredictionRefused) as caught:
        predict("haloperidol", 2.0, "single-dose")
    assert caught.value.layers == ("exposure",)
    assert "count it twice" in str(caught.value)
    # Olanzapine is untouched by haloperidol's change.
    assert predict("olanzapine", 10.0, "single-dose").drug == "olanzapine"


def test_removing_haloperidol_layers_never_routes_olanzapine_through_them(monkeypatch):
    without = {k: v for k, v in d2_ec50.D2_PLASMA_EC50.items() if k != "haloperidol"}
    monkeypatch.setattr(d2_ec50, "D2_PLASMA_EC50", without)
    before = predict("olanzapine", 10.0, "steady-state-daily")
    assert before.ec50_nM == pytest.approx(16.93125, rel=1e-5)
    with pytest.raises(PredictionRefused) as caught:
        predict("haloperidol", 2.0, "single-dose")
    assert caught.value.layers == ("occupancy",)
