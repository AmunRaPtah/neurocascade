import pytest

from neurocascade.human.gabaa_effect import (
    GABAAOperationalModel,
    HUMAN_GABAA_OPERATIONAL_PRIOR,
)


def test_full_agonist_reaches_emax():
    assert HUMAN_GABAA_OPERATIONAL_PRIOR.effect(1.0) == pytest.approx(1.0)


def test_partial_agonist_is_bounded():
    model = GABAAOperationalModel(reserve=2.0, emax=0.8)
    value = model.effect(1.0, efficacy=0.5)
    assert 0.0 < value < model.emax


def test_baseline_is_preserved():
    model = GABAAOperationalModel(baseline=0.2, emax=1.0)
    assert model.effect(0.0) == pytest.approx(0.2)


@pytest.mark.parametrize("kwargs", [{"reserve": 0}, {"emax": -1}, {"baseline": 2}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GABAAOperationalModel(**kwargs)
