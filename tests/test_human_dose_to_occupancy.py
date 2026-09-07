from dataclasses import replace

import numpy as np
import pytest

from neurocascade.human import kp_uu as kp_uu_module
from neurocascade.human.d2_ec50 import HALOPERIDOL_D2_PLASMA_EC50_nM, fit_plasma_ec50
from neurocascade.human.dose_to_occupancy import (
    ChainLayer,
    OccupancyPrediction,
    WeightScaling,
    _gate_illustrative,
    default_times_min,
    predict_occupancy,
)
from neurocascade.human.pk_parameters import human_pk_records
from neurocascade.human.receptor_parameters import require_human_parameter

REFERENCE_WEIGHT_KG = 70.0
# 5 mg delivered instantaneously into Vc/F = 669 L, in nM. Every bioavailable
# milligram has to pass through the central compartment, and absorption spreads
# it out, so no predicted concentration may reach this.
DOSE_5MG_CEILING_nM = 5.0 / 669.0 * 1e6 / 375.86


def test_the_chain_runs_on_the_registry_alone():
    prediction = predict_occupancy(5.0)
    assert isinstance(prediction, OccupancyPrediction)
    assert prediction.drug == "haloperidol"
    assert prediction.target == "D2"
    assert prediction.route == "PO"
    assert prediction.body_weight_kg == REFERENCE_WEIGHT_KG
    assert prediction.times_min.shape == prediction.occupancy.shape
    assert prediction.plasma_nM.shape == prediction.occupancy.shape


def test_the_predicted_curve_has_the_shape_an_oral_dose_must_have():
    prediction = predict_occupancy(5.0)
    occupancy = prediction.occupancy
    assert occupancy[0] == 0.0, "no drug has been absorbed at the moment of dosing"
    assert np.all((occupancy >= 0.0) & (occupancy <= 1.0))
    peak_index = int(np.argmax(occupancy))
    assert 0 < peak_index < occupancy.size - 1, "the peak is inside the window"
    # Absorption then elimination: monotone up to the peak, monotone down after.
    assert np.all(np.diff(occupancy[: peak_index + 1]) >= 0.0)
    assert np.all(np.diff(occupancy[peak_index:]) <= 0.0)
    assert prediction.peak_plasma_nM < DOSE_5MG_CEILING_nM


def test_occupancy_uses_the_p19_4_ec50_and_nothing_else():
    prediction = predict_occupancy(5.0)
    assert prediction.ec50_nM == HALOPERIDOL_D2_PLASMA_EC50_nM
    assert prediction.ec50_nM == require_human_parameter("D2").occupancy_constant_nM()
    assert prediction.concentration_basis == "total plasma"
    # The chain's own hyperbola is the fitted model's, not a second one that
    # happens to look similar. The registry stores the fitted EC50 rounded to
    # four decimals, so the fit is re-pointed at the stored constant rather than
    # the comparison being given a tolerance wide enough to hide a real gap.
    fit = fit_plasma_ec50()
    assert HALOPERIDOL_D2_PLASMA_EC50_nM == round(fit.ec50_nM, 4)
    registry_fit = replace(fit, ec50_nM=HALOPERIDOL_D2_PLASMA_EC50_nM)
    for concentration, occupancy in zip(prediction.plasma_nM, prediction.occupancy):
        assert occupancy == pytest.approx(registry_fit.occupancy_at(float(concentration)))


def test_a_bigger_dose_raises_occupancy_but_saturates():
    low = predict_occupancy(1.0).peak_occupancy
    mid = predict_occupancy(5.0).peak_occupancy
    high = predict_occupancy(20.0).peak_occupancy
    assert low < mid < high < 1.0
    # A rectangular hyperbola: twentyfold the dose is far less than twentyfold
    # the occupancy, which is the whole reason occupancy is modelled rather than
    # concentration.
    assert high / low < 20.0


def test_no_dose_is_no_occupancy():
    prediction = predict_occupancy(0.0)
    assert np.all(prediction.plasma_nM == 0.0)
    assert np.all(prediction.occupancy == 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        predict_occupancy(-1.0)


def test_every_layer_carries_its_provenance_and_none_is_illustrative():
    prediction = predict_occupancy(5.0)
    names = [layer.name for layer in prediction.layers]
    assert names == ["pk", "weight-scaling", "exposure", "occupancy", "transduction"]
    for layer in prediction.layers:
        assert layer.detail and layer.citation and layer.status
        assert layer.provenance != "illustrative"
    assert prediction.provenance == "measured"
    assert prediction.layer("pk").citation.startswith("Pilla Reddy")
    assert prediction.layer("occupancy").provenance == "fitted-in-repo"
    with pytest.raises(KeyError):
        prediction.layer("brain-partition")


def test_the_pk_layer_says_that_f_is_folded_in_rather_than_known():
    detail = predict_occupancy(5.0).layer("pk").detail
    assert "Vc/F" in detail and "CL/F" in detail
    assert "F is unidentified by the source" in detail


def test_there_is_no_brain_partition_step_and_the_object_says_so():
    prediction = predict_occupancy(5.0)
    assert "no brain partition" in prediction.receptor_compartment
    exposure = prediction.layer("exposure")
    assert exposure.status == "not-required"
    assert "no brain-partition step" in exposure.detail


def test_the_chain_refuses_to_run_if_a_kp_uu_appears_later(monkeypatch):
    """The no-partition design is checked, not assumed.

    Adding a measured Kp,uu would mean brain penetration is represented twice:
    once in the partition and once inside an EC50 fitted against plasma in
    living people. That is a decision for whoever adds the number.
    """
    patched = tuple(
        replace(record, kp_uu=0.6, measurement_kind="Kp,uu", status="sourced")
        if record.drug == "haloperidol"
        else record
        for record in kp_uu_module.HUMAN_KP_UU_RECORDS
    )
    monkeypatch.setattr(kp_uu_module, "HUMAN_KP_UU_RECORDS", patched)
    with pytest.raises(ValueError, match="count it twice"):
        predict_occupancy(5.0)


def test_the_chain_refuses_when_a_required_pk_parameter_is_removed():
    record = human_pk_records("haloperidol", "PO")[0]
    for field in ("ka_per_min", "V1_L", "V2_L", "Q12_L_per_h", "CL_L_per_h"):
        stripped = replace(record, **{field: None})
        with pytest.raises(ValueError, match="cannot drive a model") as excinfo:
            predict_occupancy(5.0, pk_record=stripped)
        assert field.replace("_per_h", "").replace("_per_min", "") in str(excinfo.value)


def test_the_chain_refuses_when_the_ec50_is_removed():
    d2 = require_human_parameter("D2")
    with pytest.raises(ValueError, match="cannot drive the occupancy layer"):
        predict_occupancy(5.0, receptor_record=replace(d2, ec50_in_vivo_nM=None))


def test_a_free_basis_ec50_is_not_silently_fed_total_plasma():
    d2 = require_human_parameter("D2")
    free_basis = replace(d2, ec50_plasma_basis="free")
    with pytest.raises(ValueError, match="free-fraction step"):
        predict_occupancy(5.0, receptor_record=free_basis)


def test_records_for_another_drug_or_species_are_refused():
    record = human_pk_records("haloperidol", "PO")[0]
    with pytest.raises(ValueError, match="human haloperidol only"):
        predict_occupancy(5.0, pk_record=replace(record, drug="risperidone"))
    d2 = require_human_parameter("D2")
    # Olanzapine rather than risperidone. P21.4 put the active-metabolite gate
    # ahead of the ligand check, so a risperidone record is now refused for the
    # stronger reason that its parent concentration does not determine its
    # occupancy at all. Keeping risperidone here would have tested the ordering
    # rather than the ligand check, so this case names a drug with no registered
    # active metabolite and the risperidone case is asserted separately below.
    with pytest.raises(ValueError, match="not 'haloperidol'"):
        predict_occupancy(5.0, receptor_record=replace(d2, ligand="olanzapine"))
    with pytest.raises(ValueError, match="9-hydroxyrisperidone"):
        predict_occupancy(5.0, receptor_record=replace(d2, ligand="risperidone"))


def test_the_intravenous_route_has_no_sourced_set_so_it_raises():
    """P19.2 recorded the haloperidol IV search as empty. The chain inherits that."""
    with pytest.raises(ValueError, match="cannot drive a model"):
        predict_occupancy(5.0, route="IV")
    with pytest.raises(ValueError, match="no human haloperidol PK record"):
        predict_occupancy(5.0, route="IM")


def test_an_off_reference_weight_needs_a_stated_scaling_rule():
    with pytest.raises(ValueError, match="needs a stated rule") as excinfo:
        predict_occupancy(5.0, body_weight_kg=90.0)
    assert "clearance only" in str(excinfo.value)


def test_a_supplied_scaling_rule_is_applied_and_attributed():
    scaling = WeightScaling(
        volume_exponent=1.0,
        q_exponent=0.75,
        source="illustrative textbook allometry, not from the PK source",
    )
    heavier = predict_occupancy(5.0, body_weight_kg=90.0, weight_scaling=scaling)
    reference = predict_occupancy(5.0)
    # A larger central volume for the same absolute dose is a lower exposure.
    assert heavier.peak_plasma_nM < reference.peak_plasma_nM
    layer = heavier.layer("weight-scaling")
    assert layer.provenance == "caller-supplied"
    assert "illustrative textbook allometry" in layer.detail
    assert "the exponent the source printed" in layer.detail
    assert layer.citation.endswith(scaling.source)


def test_a_scaling_rule_must_name_where_its_exponents_came_from():
    with pytest.raises(ValueError, match="must name where"):
        WeightScaling(volume_exponent=1.0, q_exponent=0.75, source="")
    with pytest.raises(ValueError, match="must be finite"):
        WeightScaling(volume_exponent=float("nan"), q_exponent=0.75, source="somewhere")


def test_at_the_reference_weight_no_scaling_is_claimed():
    layer = predict_occupancy(5.0).layer("weight-scaling")
    assert layer.provenance == "structural"
    assert "no scaling applied" in layer.detail


def test_the_illustrative_gate_is_the_p19_1_gate():
    """No layer is illustrative today, so the gate is exercised directly."""
    layers = (
        ChainLayer("pk", "sourced", "d", "c", "measured"),
        ChainLayer("ke0", "illustrative", "d", "c", "illustrative"),
    )
    with pytest.raises(ValueError, match="ILLUSTRATIVE"):
        _gate_illustrative(layers, allow_illustrative=False)
    _gate_illustrative(layers, allow_illustrative=True)
    assert (
        OccupancyPrediction(
            drug="haloperidol",
            target="D2",
            dose_mg=1.0,
            route="PO",
            body_weight_kg=70.0,
            times_min=np.array([0.0, 1.0]),
            plasma_nM=np.zeros(2),
            occupancy=np.zeros(2),
            concentration_basis="total plasma",
            receptor_compartment="total plasma (no brain partition attached)",
            ec50_nM=1.0,
            layers=layers,
        ).provenance
        == "illustrative"
    )


def test_a_layer_cannot_claim_a_provenance_the_vocabulary_does_not_have():
    with pytest.raises(ValueError, match="is not one of"):
        ChainLayer("pk", "sourced", "d", "c", "plausible")


def test_the_transducer_is_a_re_expression_of_occupancy_not_a_behavioural_claim():
    prediction = predict_occupancy(5.0, times_min=np.linspace(0.0, 600.0, 25))
    effects = prediction.effects()
    assert len(effects) == prediction.occupancy.size
    for effect, occupancy in zip(effects, prediction.occupancy):
        assert effect.emax == 1.0
        assert effect.baseline_occupancy == 0.0
        assert abs(effect.effect) == pytest.approx(occupancy)
    assert "not a behavioural claim" in prediction.layer("transduction").detail


def test_the_reporting_grid_is_a_display_choice_not_a_model_parameter():
    coarse = predict_occupancy(5.0, times_min=np.linspace(0.0, 4320.0, 73))
    fine = predict_occupancy(5.0, times_min=np.linspace(0.0, 4320.0, 721))
    for time_min in (60.0, 240.0, 1440.0, 4320.0):
        assert coarse.occupancy_at(time_min) == pytest.approx(
            fine.occupancy_at(time_min), rel=1e-3
        )


def test_the_default_grid_spans_more_than_one_terminal_half_life():
    times = default_times_min()
    assert times[0] == 0.0
    assert times[-1] == 72.0 * 60.0
    assert np.all(np.diff(times) > 0.0)


def test_a_reporting_grid_that_cannot_be_read_is_refused():
    with pytest.raises(ValueError, match="at least 2 points"):
        predict_occupancy(5.0, times_min=[0.0])
    with pytest.raises(ValueError, match="before the dose"):
        predict_occupancy(5.0, times_min=[-10.0, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        predict_occupancy(5.0, times_min=[0.0, 60.0, 30.0])


def test_reading_a_time_off_the_grid_is_refused_rather_than_extrapolated():
    prediction = predict_occupancy(5.0, times_min=np.linspace(0.0, 600.0, 21))
    assert prediction.occupancy_at(600.0) == pytest.approx(prediction.occupancy[-1])
    with pytest.raises(ValueError, match="outside the predicted window"):
        prediction.occupancy_at(601.0)


def test_the_summary_states_the_limits_of_what_was_predicted():
    summary = predict_occupancy(5.0).summary()
    assert "haloperidol 5 mg PO" in summary
    assert "provenance: measured" in summary
    assert "no brain partition" in summary
    for name in ("pk", "weight-scaling", "exposure", "occupancy", "transduction"):
        assert name in summary
