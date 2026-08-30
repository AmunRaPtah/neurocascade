import pytest

from neurocascade.human.gabaa_effect import (
    GABAAOperationalModel,
    HUMAN_GABAA_OPERATIONAL_PRIOR,
)


def test_full_agonist_reaches_emax():
    assert HUMAN_GABAA_OPERATIONAL_PRIOR.effect(1.0, allow_illustrative=True) == pytest.approx(1.0)


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


# ---------------------------------------------------------------------------
# P19.1 gate: the illustrative prior cannot be consumed by accident
# ---------------------------------------------------------------------------


def test_illustrative_prior_refuses_without_opt_in():
    with pytest.raises(ValueError, match="ILLUSTRATIVE"):
        HUMAN_GABAA_OPERATIONAL_PRIOR.effect(0.5)


def test_illustrative_prior_opt_in_carries_provenance():
    value = HUMAN_GABAA_OPERATIONAL_PRIOR.effect(0.5, allow_illustrative=True)
    assert value.provenance == "illustrative"
    assert 0.0 < float(value) < 1.0


def test_explicit_sourced_model_is_not_gated():
    model = GABAAOperationalModel(reserve=1.0, status="measured", source="assay X")
    value = model.effect(0.5)  # no opt-in needed
    assert value.provenance == "measured"
    assert model.effect(0.0) == pytest.approx(0.0)
