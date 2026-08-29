import pytest

from neurocascade.human.mor_effect import HUMAN_MOR_RESERVE_PRIOR, MORReserveModel


def test_reserve_amplifies_submaximal_response():
    plain = MORReserveModel(reserve=1.0).response(0.25)
    reserved = MORReserveModel(reserve=2.0).response(0.25)
    assert reserved > plain


def test_response_is_bounded_and_monotonic():
    model = HUMAN_MOR_RESERVE_PRIOR
    values = [model.response(x) for x in (0.0, 0.25, 0.75, 1.0)]
    assert values == sorted(values)
    assert values[-1] <= model.emax


def test_partial_efficacy_reduces_response():
    model = HUMAN_MOR_RESERVE_PRIOR
    assert model.response(1.0, efficacy=0.5) < model.response(1.0)


@pytest.mark.parametrize("kwargs", [{"reserve": 0}, {"emax": -1}, {"baseline": 2}])
def test_invalid_model_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MORReserveModel(**kwargs)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        HUMAN_MOR_RESERVE_PRIOR.response(1.1)
    with pytest.raises(ValueError):
        HUMAN_MOR_RESERVE_PRIOR.response(0.5, efficacy=-0.1)
