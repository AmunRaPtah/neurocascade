import pytest

from neurocascade.human.d2_effect import d2_effect_from_occupancy


def test_d2_effect_is_occupancy_limited_and_inhibitory():
    result = d2_effect_from_occupancy(0.8, baseline_occupancy=0.2, emax=0.5)
    assert result.effect == pytest.approx(-0.3)
    assert result.available_reserve == pytest.approx(0.8)


def test_baseline_occupancy_does_not_create_drug_effect():
    assert d2_effect_from_occupancy(0.2, baseline_occupancy=0.2).effect == 0.0


def test_invalid_occupancy_is_rejected():
    with pytest.raises(ValueError, match="occupancy"):
        d2_effect_from_occupancy(1.1)
