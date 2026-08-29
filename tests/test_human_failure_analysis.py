import pytest

from neurocascade.human.failure_analysis import (
    HUMAN_LAYER_FINDINGS,
    blocking_findings,
    human_failure_analysis,
)


def test_analysis_covers_required_layers():
    layers = {item.layer for item in human_failure_analysis()}
    assert {"PK", "BBB partition", "Receptor binding", "Connectivity", "Effect transduction", "Behavioral validation"} <= layers
    assert all(item.required_repair for item in HUMAN_LAYER_FINDINGS)


def test_behavioral_validation_remains_the_blocker():
    blockers = blocking_findings()
    assert len(blockers) == 1
    assert blockers[0].layer == "Behavioral validation"


def test_invalid_finding_is_rejected():
    from neurocascade.human.failure_analysis import LayerFinding
    with pytest.raises(ValueError):
        LayerFinding("x", "unknown", "limitation", "repair")
