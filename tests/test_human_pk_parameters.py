"""P14.2: human PK provenance records. P19.2: the haloperidol transcription."""

import pytest

from neurocascade.human.pk_parameters import (
    HUMAN_PK_RECORDS, HumanPKRecord, human_pk_records,
)


def _blank_record(**overrides):
    """A record with every parameter absent, for exercising the schema rules."""
    kwargs = dict(
        drug="testdrug", route="IV", species="human",
        V1_L_per_kg=None, V2_L_per_kg=None, V3_L_per_kg=None,
        Q12_mL_per_min_per_kg=None, Q13_mL_per_min_per_kg=None,
        CL_mL_per_min_per_kg=None, F=None, ka_per_min=None,
        citation="test", status="test", notes="test",
    )
    kwargs.update(overrides)
    return HumanPKRecord(**kwargs)


def test_records_cover_the_three_required_human_drugs():
    assert {record.drug for record in HUMAN_PK_RECORDS} == {"fentanyl", "haloperidol", "diazepam"}
    assert all(record.species == "human" for record in HUMAN_PK_RECORDS)


def test_every_record_names_its_route_and_non_iv_routes_carry_absorption():
    # P19.2 added the first non-IV record, so route is no longer a constant.
    # A non-IV record that cannot say how fast the drug is absorbed is not a
    # usable record, whatever else it carries.
    assert all(record.route in {"IV", "PO", "IM"} for record in HUMAN_PK_RECORDS)
    for record in HUMAN_PK_RECORDS:
        if record.route != "IV" and record.complete_for_model:
            assert record.ka_per_min is not None


def test_unsourced_human_sets_are_not_presented_as_complete_models():
    for record in human_pk_records("fentanyl") + human_pk_records("diazepam"):
        assert not record.complete_for_model
        assert record.missing_fields
    assert human_pk_records("fentanyl")[0].status == "parameter-set-required"
    assert human_pk_records("diazepam")[0].status == "searched-and-empty"


def test_haloperidol_intravenous_remains_a_recorded_gap():
    # P19.2's search did not reach a readable human IV micro-parameter set.
    # That outcome is the deliverable; it must not quietly acquire values.
    record = human_pk_records("haloperidol", route="IV")[0]
    assert record.status == "searched-and-empty"
    assert not record.complete_for_model
    assert record.V1_L_per_kg is None and record.V1_L is None
    assert "6831826" in record.citation  # Holley 1983, the unreached full text


def test_haloperidol_oral_carries_the_pilla_reddy_set_as_printed():
    record = human_pk_records("haloperidol", route="PO")[0]
    assert record.units == "absolute"
    assert record.n_compartments == 2
    # Table 2, original data set column, transcribed without rescaling.
    assert record.CL_L_per_h == 88.0
    assert record.Q12_L_per_h == 233.0
    assert record.V1_L == 669.0
    assert record.V2_L == 2500.0
    assert record.ka_per_min == pytest.approx(0.236 / 60.0)
    assert record.reference_weight_kg == 70.0
    assert "24113674" in record.citation


def test_haloperidol_oral_is_complete_without_ever_identifying_F():
    # The point of an apparent parameter set: F cancels out of an oral
    # prediction, so its absence is not a gap for this route.
    record = human_pk_records("haloperidol", route="PO")[0]
    assert record.apparent_over_F
    assert record.F is None
    assert record.complete_for_model
    assert record.missing_fields == ()


def test_two_compartment_record_does_not_report_a_third_compartment_as_missing():
    record = human_pk_records("haloperidol", route="PO")[0]
    assert "V3_L" not in record.required_fields
    assert "Q13_L_per_h" not in record.required_fields
    assert record.V3_L is None and record.Q13_L_per_h is None


def test_a_record_without_a_compartment_count_is_judged_strictly():
    # An unread source gets the three-compartment demand, so a record cannot
    # look complete merely because nobody recorded how many compartments the
    # literature supports.
    record = _blank_record()
    assert record.n_compartments is None
    assert "V3_L_per_kg" in record.required_fields
    assert "Q13_mL_per_min_per_kg" in record.required_fields


def test_apparent_values_cannot_be_paired_with_a_borrowed_bioavailability():
    with pytest.raises(ValueError, match="composite"):
        _blank_record(apparent_over_F=True, F=0.65)


def test_derived_per_kg_values_must_name_the_weight_they_came_from():
    with pytest.raises(ValueError, match="body weight used to normalise"):
        _blank_record(CL_mL_per_min_per_kg=12.0, source_reports_per_kg=False)
    # Reporting the weight makes the same record legal.
    ok = _blank_record(CL_mL_per_min_per_kg=12.0, source_reports_per_kg=False,
                       reference_weight_kg=70.5)
    assert ok.CL_mL_per_min_per_kg == 12.0


def test_per_kg_values_must_say_which_way_the_source_reported():
    with pytest.raises(ValueError, match="whether the source reported"):
        _blank_record(CL_mL_per_min_per_kg=12.0)


def test_absolute_values_must_name_their_reference_subject():
    with pytest.raises(ValueError, match="reference subject"):
        _blank_record(units="absolute", CL_L_per_h=88.0)


def test_unit_families_are_never_mixed():
    with pytest.raises(ValueError, match="never both"):
        _blank_record(units="absolute", reference_weight_kg=70.0,
                      CL_L_per_h=88.0, V1_L_per_kg=9.6)


def test_a_two_compartment_source_cannot_carry_a_third_compartment():
    with pytest.raises(ValueError, match="cannot support"):
        _blank_record(units="absolute", reference_weight_kg=70.0, n_compartments=2,
                      V1_L=669.0, V2_L=2500.0, V3_L=1000.0)


def test_clearance_scales_by_the_exponent_the_source_states():
    record = human_pk_records("haloperidol", route="PO")[0]
    assert record.CL_L_per_h_at_weight(70.0) == pytest.approx(88.0)
    # CL/F = TVCL * [weight / 70] ** 0.75, verbatim from the paper's methods.
    assert record.CL_L_per_h_at_weight(90.0) == pytest.approx(88.0 * (90.0 / 70.0) ** 0.75)
    # The naive per-kg reading would give 88 * 90/70 = 113.1; the allometric
    # value is meaningfully lower, which is the whole reason the exponent is
    # stored rather than assumed.
    assert record.CL_L_per_h_at_weight(90.0) < 88.0 * (90.0 / 70.0)


def test_clearance_cannot_be_moved_off_a_reference_weight_without_a_rule():
    record = _blank_record(units="absolute", reference_weight_kg=70.0, CL_L_per_h=50.0)
    with pytest.raises(ValueError, match="no weight-scaling rule"):
        record.CL_L_per_h_at_weight(90.0)


def test_records_can_be_selected_by_route():
    assert len(human_pk_records("haloperidol")) == 2
    assert len(human_pk_records("haloperidol", route="PO")) == 1

    # Until P22.2 the registry held exactly one PO record, so this test could
    # assert that selecting by route and selecting by drug-and-route returned
    # the same tuple. The FDA population-PK bridge added a second PO drug, so
    # that identity is now false for a legitimate reason. What the filter
    # actually promises is asserted instead: route selection returns every
    # record on that route and nothing else, and adding the drug narrows it.
    po = human_pk_records(route="PO")
    assert {r.route for r in po} == {"PO"}
    assert {r.drug for r in po} == {"haloperidol", "olanzapine"}
    assert set(human_pk_records("haloperidol", route="PO")) < set(po)

    # Provenance survives the merge: the hand-transcribed rows and the bridged
    # ones stay distinguishable in the data, not only in the import.
    bridged = [r for r in po if r.status == "fda-population-pk-bridged"]
    assert [r.drug for r in bridged] == ["olanzapine"]
