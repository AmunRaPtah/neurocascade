import pytest

from neurocascade.human.d2_effect import (
    PartialAgonistTransductionError,
    d2_effect_for_drug,
    d2_effect_from_occupancy,
    d2_functional_class,
    registered_d2_functional_classes,
)


def test_d2_effect_is_occupancy_limited_and_inhibitory():
    result = d2_effect_from_occupancy(0.8, baseline_occupancy=0.2, emax=0.5)
    assert result.effect == pytest.approx(-0.3)
    assert result.available_reserve == pytest.approx(0.8)


def test_baseline_occupancy_does_not_create_drug_effect():
    assert d2_effect_from_occupancy(0.2, baseline_occupancy=0.2).effect == 0.0


def test_invalid_occupancy_is_rejected():
    with pytest.raises(ValueError, match="occupancy"):
        d2_effect_from_occupancy(1.1)


def test_silent_antagonist_transduces_on_the_occupancy_identity():
    # Haloperidol is a silent D2 antagonist, so occupancy is the fractional loss
    # of D2 tone and the transducer is the identity on the sign convention.
    result = d2_effect_for_drug(0.9, "haloperidol")
    assert result.emax == 1.0
    assert result.baseline_occupancy == 0.0
    assert abs(result.effect) == pytest.approx(0.9)


def test_partial_agonist_is_refused_by_the_transducer():
    # The occupancy-equals-effect identity must not hold silently for a partial
    # agonist: aripiprazole's occupancy overstates its effect by an intrinsic
    # activity with no single sourced value, so the transducer refuses it.
    with pytest.raises(PartialAgonistTransductionError, match="partial_agonist"):
        d2_effect_for_drug(0.9, "aripiprazole")


def test_partial_agonist_refusal_names_the_drug_and_a_reason():
    with pytest.raises(PartialAgonistTransductionError) as excinfo:
        d2_effect_for_drug(0.9, "Aripiprazole")
    message = str(excinfo.value)
    assert "aripiprazole" in message
    # The refusal carries the sourced reason, not a bare rejection.
    assert "intrinsic activity" in message
    assert "12065741" in message  # Burris 2002 PMID


def test_a_partial_agonist_and_a_silent_antagonist_diverge_at_the_same_occupancy():
    # The regression this task exists to prevent: identical occupancy must not
    # produce an identical (silent) effect for both functional classes.
    antagonist = d2_effect_for_drug(0.9, "haloperidol")
    assert abs(antagonist.effect) == pytest.approx(0.9)
    with pytest.raises(PartialAgonistTransductionError):
        d2_effect_for_drug(0.9, "aripiprazole")


def test_unclassified_drug_is_refused_by_name_not_defaulted():
    with pytest.raises(KeyError, match="risperidone"):
        d2_effect_for_drug(0.9, "risperidone")


def test_the_functional_registry_is_sourced_and_non_empty():
    classified = registered_d2_functional_classes()
    assert "haloperidol" in classified
    assert "aripiprazole" in classified
    for drug in classified:
        record = d2_functional_class(drug)
        assert record.reason.strip()
        assert record.source.strip()
    assert d2_functional_class("aripiprazole").kind == "partial_agonist"
    assert d2_functional_class("haloperidol").kind == "silent_antagonist"
