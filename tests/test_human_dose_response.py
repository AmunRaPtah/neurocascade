import pytest

from neurocascade.human.dose_response import (
    HumanDoseResponseObservation,
    fit_human_log_linear,
)


def observations():
    return [
        HumanDoseResponseObservation(1, 0.0, "subjective_sedation", "human-trial"),
        HumanDoseResponseObservation(2, 0.5, "subjective_sedation", "human-trial"),
        HumanDoseResponseObservation(4, 1.0, "subjective_sedation", "human-trial"),
    ]


def test_fit_requires_and_uses_human_observations():
    fit = fit_human_log_linear(observations())
    assert fit.status == "fitted-human"
    assert fit.n_observations == 3
    assert fit.predict(2) == pytest.approx(0.5)


@pytest.mark.parametrize("items", [[], observations()[:2]])
def test_insufficient_data_is_rejected(items):
    with pytest.raises(ValueError, match="insufficient"):
        fit_human_log_linear(items)


def test_mixed_endpoints_and_sources_are_rejected():
    mixed_endpoint = observations()[:2] + [HumanDoseResponseObservation(4, 1, "anxiety_state", "human-trial")]
    with pytest.raises(ValueError, match="one endpoint"):
        fit_human_log_linear(mixed_endpoint)
    mixed_source = observations()[:2] + [HumanDoseResponseObservation(4, 1, "subjective_sedation", "other")]
    with pytest.raises(ValueError, match="provenance"):
        fit_human_log_linear(mixed_source)
