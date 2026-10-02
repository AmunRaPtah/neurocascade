import pytest

from neurocascade.human.kp_uu import kp_uu_records, transfer_kp_uu
from neurocascade.species_gate import TranslationTag, UnsourcedParameterError


def test_human_records_are_explicitly_unavailable_until_sourced():
    records = kp_uu_records()
    assert [r.drug for r in records[:3]] == ["fentanyl", "haloperidol", "diazepam"]
    # Rows added later (olanzapine, P22.2) must not invent a human Kp,uu.
    for record in records[3:]:
        assert record.kp_uu is None and record.status == "not-required"
    assert records[0].kp_uu is None and records[0].status == "least-supported"
    assert records[1].kp_uu is None and records[1].status == "not-required"
    assert records[2].kp_uu == pytest.approx(0.79)
    assert records[2].measurement_kind == "Kp,uu"


def test_measured_human_value_is_not_marked_as_transfer():
    result = transfer_kp_uu(human_kp_uu=0.42)
    assert result == {"kp_uu": 0.42, "provenance": "measured-human", "source_species": "human"}


def test_missing_value_requires_explicit_factor():
    with pytest.raises(ValueError, match="transfer_factor"):
        transfer_kp_uu(source_species="rat", source_kp_uu=0.5)


def test_species_transfer_is_explicit_and_inferred():
    # P29.1: a rat Kp,uu now needs a translation tag to cross the species gate.
    tag = TranslationTag("kp-uu-transfer", "rat", "test measurement")
    result = transfer_kp_uu(source_species="rat", source_kp_uu=0.5, transfer_factor=0.8, translation=tag)
    assert result["kp_uu"] == pytest.approx(0.4)
    assert result["provenance"] == "inferred-species-transfer"
    assert result["translation"].method == "kp-uu-transfer"
    assert result["translation"].species == "rat"


def test_untagged_species_transfer_is_refused_by_the_species_gate():
    with pytest.raises(UnsourcedParameterError, match="rat"):
        transfer_kp_uu(source_species="rat", source_kp_uu=0.5, transfer_factor=0.8)


def test_species_transfer_refuses_a_tag_for_another_method():
    tag = TranslationTag("allometric-scaling", "rat", "test measurement")
    with pytest.raises(ValueError, match="kp-uu-transfer"):
        transfer_kp_uu(source_species="rat", source_kp_uu=0.5, transfer_factor=1.0, translation=tag)


def test_human_is_not_a_species_transfer_source():
    with pytest.raises(ValueError, match="non-human"):
        transfer_kp_uu(source_species="human", source_kp_uu=0.5, transfer_factor=1.0)
