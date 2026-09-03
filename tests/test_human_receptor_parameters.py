import pytest

from neurocascade.human.d2_ec50 import HALOPERIDOL_D2_PLASMA_EC50_nM
from neurocascade.human.receptor_parameters import (
    HUMAN_RECEPTOR_PARAMETERS,
    human_receptor_parameters,
    require_human_parameter,
)


def test_registry_is_human_only_and_still_explicitly_incomplete():
    """P19.4 made exactly one record usable. The other five stay closed.

    This test previously asserted that no record was usable at all. D2 now has a
    fitted in-vivo EC50 and a measured intrinsic efficacy, so the assertion moved
    to the five records that still have neither, rather than being relaxed.
    """
    assert HUMAN_RECEPTOR_PARAMETERS
    assert all(r.species == "human" for r in HUMAN_RECEPTOR_PARAMETERS)
    usable = {r.target for r in HUMAN_RECEPTOR_PARAMETERS if r.usable}
    assert usable == {"D2"}
    for record in HUMAN_RECEPTOR_PARAMETERS:
        if record.target == "D2":
            continue
        assert record.ec50_in_vivo_nM is None
        assert record.efficacy is None
        assert record.ligand is None, "an unsourced record must not name a ligand"


def test_targets_without_a_human_source_still_refuse_rodent_fallback():
    for target in ("MOR", "GABA_A_BZD", "D1", "5HT2A", "H1"):
        with pytest.raises(ValueError, match="rodent fallback is disabled"):
            require_human_parameter(target)


def test_d2_is_usable_and_serves_the_in_vivo_constant():
    d2 = require_human_parameter("D2")
    assert d2.ligand == "haloperidol"
    assert d2.occupancy_constant_nM() == HALOPERIDOL_D2_PLASMA_EC50_nM
    assert d2.ec50_plasma_basis == "total"
    assert d2.efficacy < 0, "haloperidol is an inverse agonist, not a neutral antagonist"


def test_the_in_vitro_ki_cannot_reach_the_occupancy_path_unannounced():
    d2 = require_human_parameter("D2")
    assert d2.ki_nM is not None
    # The default path returns the EC50 and never the Ki.
    assert d2.occupancy_constant_nM() != d2.ki_nM
    # Promoting the Ki requires both a factor and a named origin for it.
    with pytest.raises(ValueError, match="correction_source"):
        d2.occupancy_constant_nM(in_vitro_correction=0.213)
    with pytest.raises(ValueError, match="must be positive"):
        d2.occupancy_constant_nM(in_vitro_correction=-1.0, correction_source="nowhere")
    corrected = d2.occupancy_constant_nM(
        in_vitro_correction=0.5, correction_source="illustrative only, not a measurement"
    )
    assert corrected == pytest.approx(d2.ki_nM * 0.5)


def test_a_record_holding_only_a_ki_stays_unusable():
    """Guards the rule directly rather than relying on no such record existing."""
    from dataclasses import replace

    d2 = require_human_parameter("D2")
    ki_only = replace(d2, ec50_in_vivo_nM=None)
    assert not ki_only.usable
    assert ki_only.ki_nM is not None
    with pytest.raises(ValueError, match="will not be substituted"):
        ki_only.occupancy_constant_nM()


def test_the_ratio_is_reported_and_not_folded_into_either_number():
    d2 = require_human_parameter("D2")
    assert d2.in_vitro_in_vivo_ratio == pytest.approx(d2.ec50_in_vivo_nM / d2.ki_nM)
    # Far from 1, which is the finding, not an error to be corrected away.
    assert d2.in_vitro_in_vivo_ratio == pytest.approx(0.213, abs=0.002)
    assert "straddles unity" in d2.notes


def test_the_d2_record_states_its_own_uncertainty_and_provenance():
    d2 = require_human_parameter("D2")
    assert d2.ec50_origin == "fitted-in-repo"
    assert "not independent" in d2.ec50_uncertainty
    assert "1.382 to 2.525 nM" in d2.ec50_uncertainty
    # Depot data driving an oral chain is carried forward, not quietly dropped.
    assert "oral" in d2.ec50_uncertainty
    # The Ki fields name the assay, since a Ki without them is uninterpretable.
    assert "[3H]spiperone" in d2.ki_radioligand
    assert "human D2S" in d2.ki_preparation
    assert "120 mM NaCl" in d2.ki_assay_conditions
    assert "not printed" in d2.ki_assay_conditions
    assert d2.ki_citation.startswith("Bartuzi")
    assert "butaclamol" in d2.efficacy_origin


def test_unknown_target_is_not_scored_as_zero():
    assert human_receptor_parameters("unknown") == ()
    with pytest.raises(ValueError, match="no validated human"):
        require_human_parameter("unknown")


def test_in_vivo_and_in_vitro_contexts_remain_distinct():
    assert {r.target for r in human_receptor_parameters(evidence_context="in-vivo PET")} == {"D2", "MOR", "GABA_A_BZD"}
    assert all(r.evidence_context == "in-vitro" for r in human_receptor_parameters(evidence_context="in-vitro"))
