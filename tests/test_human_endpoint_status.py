import pytest

from neurocascade.human.endpoint_status import (
    ENDPOINT_STATUS,
    endpoint_status,
    require_human_endpoint,
)


def test_human_endpoints_are_retained():
    retained = [x for x in ENDPOINT_STATUS if x.status == "retained"]
    assert {x.name for x in retained} >= {
        "cognitive_performance", "motor_activity", "subjective_sedation", "anxiety_state"
    }
    assert all(require_human_endpoint(x.name).human_analogue for x in retained)


def test_rodent_specific_endpoints_are_retired():
    for name in ("elevated_plus_maze", "open_field_locomotion"):
        assert endpoint_status(name).status == "retired"
        with pytest.raises(ValueError, match="retired"):
            require_human_endpoint(name)


def test_unknown_endpoint_is_rejected():
    with pytest.raises(ValueError):
        endpoint_status("unknown")
