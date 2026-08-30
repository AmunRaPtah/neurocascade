import pytest

from neurocascade.human.mor_effect import HUMAN_MOR_RESERVE_PRIOR, MORReserveModel


def test_reserve_amplifies_submaximal_response():
    plain = MORReserveModel(reserve=1.0).response(0.25, allow_illustrative=True)
    reserved = MORReserveModel(reserve=2.0).response(0.25, allow_illustrative=True)
    assert reserved > plain


def test_response_is_bounded_and_monotonic():
    model = HUMAN_MOR_RESERVE_PRIOR
    values = [model.response(x, allow_illustrative=True) for x in (0.0, 0.25, 0.75, 1.0)]
    assert values == sorted(values)
    assert values[-1] <= model.emax


def test_partial_efficacy_reduces_response():
    model = HUMAN_MOR_RESERVE_PRIOR
    assert model.response(1.0, efficacy=0.5, allow_illustrative=True) < model.response(
        1.0, allow_illustrative=True
    )


@pytest.mark.parametrize("kwargs", [{"reserve": 0}, {"emax": -1}, {"baseline": 2}])
def test_invalid_model_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MORReserveModel(**kwargs)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        HUMAN_MOR_RESERVE_PRIOR.response(1.1, allow_illustrative=True)
    with pytest.raises(ValueError):
        HUMAN_MOR_RESERVE_PRIOR.response(0.5, efficacy=-0.1, allow_illustrative=True)


# ---------------------------------------------------------------------------
# P19.1 gate: the illustrative reserve prior cannot be consumed by accident
# ---------------------------------------------------------------------------


def test_illustrative_prior_refuses_without_opt_in():
    with pytest.raises(ValueError, match="ILLUSTRATIVE"):
        HUMAN_MOR_RESERVE_PRIOR.response(0.5)


def test_illustrative_prior_opt_in_carries_provenance():
    value = HUMAN_MOR_RESERVE_PRIOR.response(0.5, allow_illustrative=True)
    assert value.provenance == "illustrative"
    assert 0.0 < float(value) < 1.0


def test_explicit_sourced_model_is_not_gated():
    model = MORReserveModel(reserve=1.0, status="measured", source="human assay Y")
    value = model.response(0.5)  # no opt-in needed
    assert value.provenance == "measured"
