"""P25.3: oral dose to D2 occupancy end to end, scored against held-out PET."""
import numpy as np
import pytest

from neurocascade.human.active_metabolites import require_parent_only_path
from neurocascade.human.d2_ec50 import D2_PLASMA_EC50
from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.heldout_dose_occupancy import (
    EC50_ROWS,
    KP_UU_NOT_REQUIRED,
    OBSERVATIONS,
    QUETIAPINE_D2_PLASMA_EC50,
    QUETIAPINE_MW_G_PER_MOL,
    REFUSED,
    SCORED_DRUGS,
    SINGLE_DOSE,
    haloperidol_v4,
    score_all,
    summary_rows,
    unscored_candidates,
)
from neurocascade.human.kp_uu import KpUURecord, kp_uu_records
from neurocascade.human.leave_one_drug_out import (
    ARIPIPRAZOLE_D2_PLASMA_EC50,
    DENOMINATOR,
    PASS_MAE_SAME_TRACER,
)
from neurocascade.human.occupancy_validation import steady_state_occupancy
from neurocascade.human.receptor_parameters import d2_occupancy_parameter_from_ec50


@pytest.fixture(scope="module")
def scores():
    return score_all()


def _aripiprazole_kwargs():
    return dict(
        drug="aripiprazole",
        receptor_record=d2_occupancy_parameter_from_ec50(ARIPIPRAZOLE_D2_PLASMA_EC50),
        kp_uu_record=KP_UU_NOT_REQUIRED["aripiprazole"],
    )


# --- the constants and records this task adds --------------------------------


def test_quetiapine_constant_is_nords_ir_value_on_the_label_base_mass():
    assert QUETIAPINE_D2_PLASMA_EC50.ec50_nM == 1373.0
    assert QUETIAPINE_MW_G_PER_MOL == pytest.approx(383.52)
    row = QUETIAPINE_D2_PLASMA_EC50
    assert (row.emax, row.emax_convention) == (1.0, "fixed-at-1")
    assert (row.plasma_basis, row.analyte_is_summed) == ("total", False)
    assert row.tracer == "[11C]raclopride"
    assert "PMC3198174" in row.citation and "1373" in row.citation
    assert row.ec50_ng_per_mL == pytest.approx(526.57, abs=0.01)


def test_aripiprazole_constant_is_p22_3s_kim_2012_row():
    assert EC50_ROWS["aripiprazole"] is ARIPIPRAZOLE_D2_PLASMA_EC50
    assert EC50_ROWS["quetiapine"] is QUETIAPINE_D2_PLASMA_EC50


def test_exposure_records_are_not_required_with_no_value():
    for drug, record in KP_UU_NOT_REQUIRED.items():
        assert record.drug == drug and record.species == "human"
        assert record.kp_uu is None
        assert record.status == "not-required"


def test_live_registries_are_untouched_so_p22_results_stay_frozen():
    assert set(D2_PLASMA_EC50) == {"haloperidol", "olanzapine"}
    assert kp_uu_records("quetiapine") == ()
    assert kp_uu_records("aripiprazole") == ()


# --- the gates this task opens, and how narrowly ------------------------------


def test_aripiprazole_still_refused_without_the_single_dose_regimen():
    with pytest.raises(ValueError, match="active metabolite"):
        predict_occupancy(3.0, **_aripiprazole_kwargs())
    with pytest.raises(ValueError, match="active metabolite"):
        require_parent_only_path("aripiprazole")


def test_single_dose_regimen_clears_aripiprazole_and_nothing_else():
    assert require_parent_only_path("aripiprazole", regimen=SINGLE_DOSE) == ()
    with pytest.raises(ValueError, match="active metabolite"):
        require_parent_only_path("aripiprazole", regimen="oral, steady state")
    with pytest.raises(ValueError, match="active metabolite"):
        require_parent_only_path("risperidone", regimen=SINGLE_DOSE)


def test_single_dose_aripiprazole_runs_and_carries_no_refused_caveat():
    times = np.array([0.0, 210.0, 270.0])
    curve = predict_occupancy(3.0, times_min=times, regimen=SINGLE_DOSE, **_aripiprazole_kwargs())
    assert curve.metabolite_caveats == ()
    assert "single oral dose" in curve.layer("occupancy").detail
    assert curve.body_weight_kg == 74.19
    assert curve.occupancy[1] == pytest.approx(0.4977, abs=1e-4)


def test_steady_state_refuses_a_single_dose_regimen():
    with pytest.raises(ValueError, match="single dose"):
        steady_state_occupancy(3.0, regimen=SINGLE_DOSE, **_aripiprazole_kwargs())


def test_supplied_kp_uu_record_is_checked_like_the_registrys():
    kwargs = _aripiprazole_kwargs()
    kwargs["kp_uu_record"] = KpUURecord(
        "aripiprazole", "human", 0.5, "Kp,uu", "test", "measured-human", "test"
    )
    with pytest.raises(ValueError, match="count it twice"):
        predict_occupancy(3.0, regimen=SINGLE_DOSE, **kwargs)
    kwargs["kp_uu_record"] = KP_UU_NOT_REQUIRED["quetiapine"]
    with pytest.raises(ValueError, match="not human aripiprazole"):
        predict_occupancy(3.0, regimen=SINGLE_DOSE, **kwargs)


# --- the scored comparison -----------------------------------------------------


def test_observations_are_all_same_tracer_and_name_their_disjointness():
    assert {o.drug for o in OBSERVATIONS} == set(SCORED_DRUGS)
    for o in OBSERVATIONS:
        assert o.tracer == "[11C]raclopride"
        assert o.disjointness


PINNED = {
    # (scan start h, region): (predicted occupancy, predicted plasma ng/mL)
    "aripiprazole": [
        (0.5005, 11.123), (0.5005, 11.123), (0.6671, 22.247),
        (0.6671, 22.247), (0.7504, 33.370), (0.7504, 33.370),
    ],
    "quetiapine": [(0.5503, 644.377), (0.0903, 52.242)],
    "olanzapine": [(0.7494, 31.553)],
}


@pytest.mark.parametrize("drug", SCORED_DRUGS)
def test_per_observation_predictions_are_pinned(scores, drug):
    got = [(s.predicted, s.predicted_plasma_ng_mL) for s in scores[drug].scored]
    assert len(got) == len(PINNED[drug])
    for (occ, plasma), (want_occ, want_plasma) in zip(got, PINNED[drug]):
        assert occ == pytest.approx(want_occ, abs=1e-4)
        assert plasma == pytest.approx(want_plasma, abs=1e-3)


def test_per_drug_scores_and_verdicts(scores):
    assert PASS_MAE_SAME_TRACER == 0.10
    expected = {
        "aripiprazole": (0.0406, -0.0406, 3),
        "quetiapine": (0.0597, -0.0597, 1),
        "olanzapine": (0.0584, +0.0584, 1),
    }
    for drug, (mae, bias, groups) in expected.items():
        score = scores[drug]
        assert score.mae == pytest.approx(mae, abs=1e-4)
        assert score.bias == pytest.approx(bias, abs=1e-4)
        assert score.n_independent_groups == groups
        assert score.passed


def test_aripiprazole_errors_are_one_signed_and_the_plasma_arm_is_low(scores):
    score = scores["aripiprazole"]
    assert all(s.error < 0 for s in score.scored)
    assert [round(f, 3) for f in score.plasma_folds] == [0.927, 0.927, 0.767, 0.767, 0.834, 0.834]
    assert scores["olanzapine"].plasma_folds == pytest.approx((0.8437,), abs=1e-4)
    assert scores["quetiapine"].plasma_folds == ()


def test_haloperidol_is_v4_recomputed():
    v4 = haloperidol_v4()
    assert v4.steady_state_mean_occupancy == pytest.approx(0.5635, abs=1e-4)
    assert v4.steady_state_mean_contained and not v4.was_blind


def test_summary_has_four_drugs_end_to_end():
    rows = summary_rows()
    assert [r.split(" | ")[0] for r in rows] == [
        "haloperidol", "aripiprazole", "quetiapine", "olanzapine"
    ]


# --- what was not scored, and why ---------------------------------------------


def test_schedule_ambiguous_sources_show_both_readings():
    gefvert, kapur = unscored_candidates()[:2]
    assert gefvert.readings == "once daily 0.563, twice daily 0.435, observed 0.44"
    assert kapur.readings == (
        "at 12 h, 150 mg/d: once daily 0.099, twice daily 0.061; 600 mg/d: once "
        "daily 0.305, twice daily 0.206"
    )
    assert all(c.reason for c in unscored_candidates())


def test_every_other_census_drug_is_refused_with_a_named_layer():
    covered = set(SCORED_DRUGS) | {"haloperidol"}
    refused = {r.drug: r for r in REFUSED}
    assert set(refused) == set(DENOMINATOR) - covered
    assert {r.missing_layer for r in REFUSED} <= {"pk", "active-metabolite"}
    assert refused["risperidone"].missing_layer == "active-metabolite"
    assert refused["clozapine"].missing_layer == "pk"
