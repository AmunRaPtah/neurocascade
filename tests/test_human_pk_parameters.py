"""P14.2: human PK provenance records."""

from neurocascade.human.pk_parameters import HUMAN_PK_RECORDS, human_pk_records


def test_records_cover_the_three_required_human_drugs():
    assert {record.drug for record in HUMAN_PK_RECORDS} == {"fentanyl", "haloperidol", "diazepam"}
    assert all(record.species == "human" for record in HUMAN_PK_RECORDS)
    assert all(record.route == "IV" for record in HUMAN_PK_RECORDS)


def test_incomplete_human_sets_are_not_presented_as_complete_models():
    assert all(not record.complete_for_model for record in HUMAN_PK_RECORDS)
    assert human_pk_records("fentanyl")[0].status == "parameter-set-required"
    assert human_pk_records("diazepam")[0].status == "searched-and-empty"
