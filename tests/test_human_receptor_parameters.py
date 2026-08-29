import pytest

from neurocascade.human.receptor_parameters import (
    HUMAN_RECEPTOR_PARAMETERS,
    human_receptor_parameters,
    require_human_parameter,
)


def test_registry_is_human_only_and_explicitly_incomplete():
    assert HUMAN_RECEPTOR_PARAMETERS
    assert all(r.species == "human" for r in HUMAN_RECEPTOR_PARAMETERS)
    assert all(not r.usable for r in HUMAN_RECEPTOR_PARAMETERS)


def test_unvalidated_target_refuses_rodent_fallback():
    with pytest.raises(ValueError, match="rodent fallback is disabled"):
        require_human_parameter("D2")


def test_unknown_target_is_not_scored_as_zero():
    assert human_receptor_parameters("unknown") == ()
    with pytest.raises(ValueError, match="no validated human"):
        require_human_parameter("unknown")


def test_in_vivo_and_in_vitro_contexts_remain_distinct():
    assert {r.target for r in human_receptor_parameters(evidence_context="in-vivo PET")} == {"D2", "MOR", "GABA_A_BZD"}
    assert all(r.evidence_context == "in-vitro" for r in human_receptor_parameters(evidence_context="in-vitro"))
