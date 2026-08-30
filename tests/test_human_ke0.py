import pytest

from neurocascade.human.ke0 import GatedFloat, HUMAN_KE0_RECORDS, HumanKe0, human_ke0


def test_effectors_have_explicit_positive_ke0():
    assert {r.effector for r in HUMAN_KE0_RECORDS} == {
        "cognitive_performance", "motor_activity", "subjective_sedation", "anxiety_state"
    }
    assert all(r.ke0_per_min > 0 for r in HUMAN_KE0_RECORDS)


def test_effect_site_moves_toward_brain_concentration():
    record = human_ke0("subjective_sedation", allow_illustrative=True)
    assert record.effect_site_rhs(0.0, 10.0, allow_illustrative=True) > 0
    assert record.effect_site_rhs(10.0, 0.0, allow_illustrative=True) < 0
    assert record.half_life_min == pytest.approx(0.69314718056 / record.ke0_per_min)


def test_unknown_and_invalid_records_are_rejected():
    with pytest.raises(ValueError):
        human_ke0("missing")
    with pytest.raises(ValueError):
        HumanKe0("", 0.1, "test")
    with pytest.raises(ValueError):
        HumanKe0("x", 0.0, "test")


# ---------------------------------------------------------------------------
# P19.1 gate: illustrative ke0 records require the explicit opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effector", [r.effector for r in HUMAN_KE0_RECORDS])
def test_every_illustrative_record_refuses_without_opt_in(effector):
    with pytest.raises(ValueError, match="ILLUSTRATIVE"):
        human_ke0(effector)


def test_opt_in_returns_record_and_rhs_carries_provenance():
    record = human_ke0("motor_activity", allow_illustrative=True)
    assert record.provenance == "illustrative"
    rhs = record.effect_site_rhs(0.0, 10.0, allow_illustrative=True)
    assert rhs.provenance == "illustrative"
    assert rhs == pytest.approx(0.08 * 10.0)  # value unchanged by the gate


def test_measured_record_is_ungated():
    record = HumanKe0("fitted_effector", 0.07, "fitted against human data", "measured")
    assert record.provenance == "measured"
    assert record.effect_site_rhs(0.0, 1.0).provenance == "measured"


def test_gated_float_behaves_like_float():
    g = GatedFloat(1.5, "illustrative")
    assert g == 1.5
    assert g + 0.5 == 2.0
    assert g > 1.0
    assert g.provenance == "illustrative"
