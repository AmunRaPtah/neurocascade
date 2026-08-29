import pytest

from neurocascade.human.ke0 import HUMAN_KE0_RECORDS, HumanKe0, human_ke0


def test_effectors_have_explicit_positive_ke0():
    assert {r.effector for r in HUMAN_KE0_RECORDS} == {
        "cognitive_performance", "motor_activity", "subjective_sedation", "anxiety_state"
    }
    assert all(r.ke0_per_min > 0 for r in HUMAN_KE0_RECORDS)


def test_effect_site_moves_toward_brain_concentration():
    record = human_ke0("subjective_sedation")
    assert record.effect_site_rhs(0.0, 10.0) > 0
    assert record.effect_site_rhs(10.0, 0.0) < 0
    assert record.half_life_min == pytest.approx(0.69314718056 / record.ke0_per_min)


def test_unknown_and_invalid_records_are_rejected():
    with pytest.raises(ValueError):
        human_ke0("missing")
    with pytest.raises(ValueError):
        HumanKe0("", 0.1, "test")
    with pytest.raises(ValueError):
        HumanKe0("x", 0.0, "test")
