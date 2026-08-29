import pytest

from neurocascade.human.behavioral_endpoints import (
    HUMAN_ENDPOINTS,
    endpoint_score,
    human_endpoints,
)


def test_registry_contains_human_observable_endpoints():
    assert len(HUMAN_ENDPOINTS) >= 4
    assert all(item.population == "adult human" for item in HUMAN_ENDPOINTS)


def test_endpoint_lookup_and_transparent_score():
    assert human_endpoints("cognitive_performance")[0].measure
    assert endpoint_score("cognitive_performance", {"PFC": 2}, {"PFC": 0.5}) == pytest.approx(1.0)


def test_unknown_or_unweighted_endpoint_is_rejected():
    with pytest.raises(ValueError):
        endpoint_score("missing", {}, {})
    with pytest.raises(ValueError):
        endpoint_score("motor_activity", {"PFC": 1}, {})
