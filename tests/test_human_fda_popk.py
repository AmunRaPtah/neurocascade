"""P21.3: population PK transcribed from FDA review documents.

Three groups of tests. The schema tests defend the invariants that reading
twenty-six reviews established, and every one of them names the application
that would have broken the rule it enforces. The algebra tests check the
half-life and area calculations against closed forms derived independently.
The comparison tests pin the actual result, including the failures, so that a
later change that quietly improves the agreement by dropping a row is caught.
"""

import math

import pytest

from neurocascade.human.fda_popk import (
    EXTRACTION_BARRIERS,
    FDA_POPPK_RECORDS,
    LABEL_HALF_LIVES,
    POPPK_ANALYTES,
    POPPK_STATUSES,
    FDAPopPK,
    PopPKParameter,
    PopPKVariability,
    half_life_comparisons,
    label_half_life,
    record,
    records_for,
    registered_drugs,
    searched_and_empty_records,
    terminal_half_life_h,
    terminal_phase_auc_fraction,
    transcribed_records,
)


def _blank_record(**overrides):
    kwargs = dict(
        drug="testdrug",
        formulation="test tablet",
        route="PO",
        analyte="parent",
        application_number="NDA 000000",
        review_date="2020-01-01",
        review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
        structural_model="one-compartment with first-order absorption",
        apparent_parameters=True,
        apparent_evidence="printed as CL/F",
        population="test",
        status="transcribed",
    )
    kwargs.update(overrides)
    return FDAPopPK(**kwargs)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_every_row_names_its_application_review_date_and_structural_model():
    # The acceptance criterion for this task, asserted directly. A row without
    # one of these is a number with no provenance, which this project treats
    # the same as an invented one.
    for row in FDA_POPPK_RECORDS:
        assert row.application_number.startswith(("NDA ", "ANDA "))
        assert re_date(row.review_date), row.application_number
        assert row.structural_model.strip()
        assert row.review_document.strip()
        assert row.apparent_evidence.strip()


def re_date(value):
    parts = value.split("-")
    return (
        len(parts) == 3
        and len(parts[0]) == 4
        and all(part.isdigit() for part in parts)
    )


def test_row_keys_are_unique():
    keys = [row.key for row in FDA_POPPK_RECORDS]
    assert len(keys) == len(set(keys))


def test_key_separates_the_four_applications_that_hold_more_than_one_model():
    # Each of these would collide under a key of application number alone.
    for application, expected in [
        ("NDA 210655", 2),  # oral and depot sub-models, same analyte and date
        ("NDA 213586", 2),  # parent and metabolite
        ("NDA 209510", 2),  # the applicant's model and the FDA reviewer's refit
    ]:
        rows = [r for r in FDA_POPPK_RECORDS if r.application_number == application]
        assert len(rows) == expected
        assert len({r.key for r in rows}) == expected


def test_an_explicit_slash_F_symbol_cannot_sit_in_an_absolute_record():
    # The one direction of the apparent rule that holds. NDA 207533 prints
    # CL/F; no reading of any sentence makes that an absolute clearance.
    with pytest.raises(ValueError, match="CL/F"):
        _blank_record(
            apparent_parameters=False,
            apparent_evidence="an intravenous arm identified F",
            parameters=(PopPKParameter("CL/F", 2.0, "L/h", None, "test"),),
        )


def test_a_bare_symbol_may_belong_to_an_apparent_record():
    # The converse is deliberately not enforced. NDA 213586 prints bare CL and
    # V with a legend reading "CL=apparent clearance"; NDA 207946 prints bare
    # CL and V with an abbreviation list defining them as apparent. An
    # invariant tying the flag to the symbol would have mislabelled both.
    row = _blank_record(
        apparent_parameters=True,
        apparent_evidence="the table legend reads 'CL=apparent clearance'",
        parameters=(PopPKParameter("CL", 14.3, "L/h", None, "test"),),
    )
    assert row.apparent_parameters


def test_the_registry_contains_both_kinds_of_evidence_for_apparent():
    # Not a tautology: it asserts the reviews really do disagree in notation,
    # which is why the flag is a transcribed field and not a derived one.
    apparent = [r for r in FDA_POPPK_RECORDS if r.apparent_parameters and r.parameters]
    with_slash = [
        r for r in apparent if any("/F" in p.name.upper() for p in r.parameters)
    ]
    bare = [
        r
        for r in apparent
        if any(p.name in ("CL", "V", "V1") for p in r.parameters)
        and not any("/F" in p.name.upper() for p in r.parameters)
    ]
    assert with_slash and bare


def test_the_same_molecule_appears_as_both_apparent_and_absolute():
    # Paliperidone. NDA 022264 states that the palmitate ester "offers complete
    # bioavailability" and prints plain CL; NDA 216352 prints CL/F for the same
    # molecule in the same class of formulation. Route does not decide this.
    rows = records_for("paliperidone")
    flags = {r.apparent_parameters for r in rows}
    assert flags == {True, False}
    assert record("NDA 022264", "parent").apparent_parameters is False
    assert record("NDA 216352", "parent").apparent_parameters is True


def test_status_barrier_and_analyte_vocabularies_are_closed():
    for row in FDA_POPPK_RECORDS:
        assert row.status in POPPK_STATUSES
        assert row.analyte in POPPK_ANALYTES
        assert row.extraction_barrier in (None,) + EXTRACTION_BARRIERS


def test_searched_and_empty_rows_carry_a_barrier_and_no_parameters():
    empty = searched_and_empty_records()
    assert empty
    for row in empty:
        assert row.extraction_barrier is not None
        assert row.parameters == ()


def test_all_five_extraction_barriers_are_exercised_by_a_real_row():
    # The taxonomy is only worth having if the reviews actually produced all of
    # it. Four of the five are used by searched-and-empty rows; image-table is
    # used by a transcribed row, because an image can be read.
    used = {row.extraction_barrier for row in FDA_POPPK_RECORDS} - {None}
    assert "redacted" in used  # NDA 021999, 13 pages withheld as trade secret
    assert "not-fitted" in used  # NDA 022047, non-compartmental throughout
    assert "estimates-not-printed" in used  # NDA 214835, model described, no table
    assert "image-table" in used  # NDA 212849


def test_the_invega_oral_row_is_recorded_as_redacted_not_as_unreachable():
    # Corrects the P21.2 ledger line claiming this review 404s. It was fetched.
    # Its parameter table is inside thirteen withheld pages.
    row = record("NDA 021999", "parent")
    assert row.status == "searched-and-empty"
    assert row.extraction_barrier == "redacted"
    assert "does not 404" in row.notes
    assert "Trade Secret" in row.notes


def test_records_for_an_unregistered_drug_names_what_is_registered():
    with pytest.raises(ValueError, match="ziprasidone"):
        records_for("ziprasidone")


def test_lookup_refuses_an_ambiguous_application_rather_than_guessing():
    with pytest.raises(ValueError, match="ambiguous"):
        record("NDA 210655")
    assert record(
        "NDA 210655",
        "parent-and-metabolite",
        "PERSERIS subcutaneous ATRIGEL depot",
    ).route == "SC"


def test_variability_scale_is_validated():
    with pytest.raises(ValueError):
        PopPKVariability("CL", 40.0, "percent", None, False, "test")


def test_require_units_refuses_the_table_that_printed_none():
    # NDA 209510 Table 97 has a parameter column and a value column and no
    # units anywhere. The same document's Table 92 prints L/h for the same
    # symbols, and copying them across would be an inference.
    fda_refit = record(
        "NDA 209510", "parent", "BARHEMSYS intravenous injection, FDA reviewer's refit"
    )
    with pytest.raises(ValueError, match="unit"):
        fda_refit.parameter("CL").require_units()
    sponsor = record(
        "NDA 209510", "parent", "BARHEMSYS intravenous injection, Applicant's model"
    )
    assert sponsor.parameter("CL healthy").require_units() == "L/h"
    assert sponsor.parameter("CL healthy").value == 24.1


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------


def test_one_compartment_half_life_is_the_textbook_ratio():
    row = record("NDA 213586", "parent")
    expected = math.log(2.0) * 66.3 / 14.3
    assert terminal_half_life_h(row) == pytest.approx(expected)
    assert terminal_phase_auc_fraction(row) == 1.0
    assert row.n_compartments == 1


def test_two_compartment_half_life_matches_the_closed_form_beta_root():
    row = record("NDA 213378", "parent")
    cl, v1, q, v2 = 15.5, 656.0, 6.15, 225.0
    k10, k12, k21 = cl / v1, q / v1, q / v2
    total = k10 + k12 + k21
    beta = 0.5 * (total - math.sqrt(total**2 - 4.0 * k10 * k21))
    assert terminal_half_life_h(row) == pytest.approx(math.log(2.0) / beta)


def test_two_compartment_terminal_auc_fraction_matches_the_closed_form():
    # For a bolus, A_beta = (k21 - beta)/(alpha - beta) and the fraction of
    # total AUC in the terminal phase is (A_beta/beta) / (A_alpha/alpha +
    # A_beta/beta). Derived here independently of the general n-compartment
    # implementation, which builds the coefficients from the eigenvalues.
    row = record("NDA 213378", "parent")
    cl, v1, q, v2 = 15.5, 656.0, 6.15, 225.0
    k10, k12, k21 = cl / v1, q / v1, q / v2
    total = k10 + k12 + k21
    disc = math.sqrt(total**2 - 4.0 * k10 * k21)
    alpha, beta = 0.5 * (total + disc), 0.5 * (total - disc)
    a_beta = (k21 - beta) / (alpha - beta)
    a_alpha = (alpha - k21) / (alpha - beta)
    expected = (a_beta / beta) / (a_alpha / alpha + a_beta / beta)
    assert terminal_phase_auc_fraction(row) == pytest.approx(expected)


def test_three_compartment_auc_fractions_sum_to_one_and_terminal_is_smallest_rate():
    row = record(
        "NDA 209510", "parent", "BARHEMSYS intravenous injection, Applicant's model"
    )
    assert row.n_compartments == 3
    fraction = terminal_phase_auc_fraction(row)
    assert 0.0 < fraction < 1.0
    # The terminal phase must be the slowest, so its half-life exceeds the
    # naive CL/V one-compartment estimate on the central volume alone.
    assert terminal_half_life_h(row) > math.log(2.0) * 23.9 / 24.1


def test_half_life_refuses_a_row_with_no_mamillary_mapping():
    # NDA 210655 is parameterised in rate constants with a scaling factor
    # linking two entities' volumes. There is no clearance in it to use.
    row = record(
        "NDA 210655", "parent-and-metabolite", "PERSERIS subcutaneous ATRIGEL depot"
    )
    assert row.n_compartments is None
    with pytest.raises(ValueError):
        terminal_half_life_h(row)
    with pytest.raises(ValueError):
        terminal_phase_auc_fraction(row)


def test_apparent_parameters_cancel_in_the_ratio():
    # (V/F)/(CL/F) = V/CL, so an apparent row's half-life is not itself
    # apparent. This is what makes the comparison legitimate at all.
    apparent = record("NDA 213378", "parent")
    assert apparent.apparent_parameters
    assert terminal_half_life_h(apparent) > 0.0


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def test_every_comparison_names_a_registered_row_and_a_quoted_label():
    comparisons = half_life_comparisons()
    assert len(comparisons) == 13
    for item in comparisons:
        assert record(item.application_number, item.analyte, item.formulation)
        assert item.label.quote.strip()
        assert item.label.source.strip()
        assert item.comment.strip()


def test_the_comparison_reports_failures_as_well_as_agreements():
    # Rule 3 of the validation ledger. If this ever becomes all-pass, either
    # something was fixed or something was dropped, and the test forces the
    # question to be asked.
    comparisons = half_life_comparisons()
    passes = [c for c in comparisons if c.agrees_within_twofold]
    failures = [c for c in comparisons if not c.agrees_within_twofold]
    assert len(passes) == 5
    assert len(failures) == 8


def test_the_headline_agreements_are_the_expected_ones():
    by_key = {(c.application_number, c.analyte): c for c in half_life_comparisons()}
    olanzapine = by_key[("NDA 213378", "parent")]
    assert olanzapine.predicted_h == pytest.approx(49.83, abs=0.05)
    assert olanzapine.label.contains(olanzapine.predicted_h)
    assert olanzapine.label.signed_error_h(olanzapine.predicted_h) == 0.0

    risperidone = by_key[("NDA 213586", "parent")]
    assert risperidone.predicted_h == pytest.approx(3.21, abs=0.05)
    assert risperidone.fold_error == pytest.approx(1.07, abs=0.01)


def test_the_depot_failures_are_the_rows_with_a_thin_terminal_phase():
    # The diagnostic that makes the failures interpretable rather than merely
    # embarrassing: the two worst are exactly the two whose terminal
    # exponential carries the least of the area, so their slowest eigenvalue is
    # a depot release rate parked in a peripheral compartment.
    comparisons = half_life_comparisons()
    worst = sorted(comparisons, key=lambda c: -c.fold_error)[:2]
    assert {c.application_number for c in worst} == {"NDA 209830", "NDA 207533"}
    for item in worst:
        assert item.terminal_auc_fraction < 0.20
        assert item.fold_error > 50.0


def test_the_two_aripiprazole_depot_rows_are_the_same_number_twice():
    # NDA 217006 carried NDA 202971's fixed effects over rather than
    # re-estimating them. Reported as two rows so the duplication is visible,
    # not hidden by dropping one.
    by_key = {c.application_number: c for c in half_life_comparisons()}
    assert by_key["NDA 217006"].predicted_h == pytest.approx(
        by_key["NDA 202971"].predicted_h
    )


def test_fold_error_is_one_inside_the_interval_and_grows_outside_it():
    label = label_half_life("olanzapine", "olanzapine")
    assert label.contains(30.0)
    assert label.signed_error_h(60.0) == pytest.approx(6.0)
    assert label.signed_error_h(11.0) == pytest.approx(-10.0)


def test_the_label_rows_declare_whether_they_are_innovator_labels():
    # P20.6 established that most of these molecules have no current innovator
    # SPL, so the comparator is a generic's copy. Carried per row rather than
    # stated once in prose, because four of the five are generics and one is
    # not.
    innovators = [item for item in LABEL_HALF_LIVES if item.innovator]
    generics = [item for item in LABEL_HALF_LIVES if not item.innovator]
    assert len(innovators) == 1
    assert innovators[0].drug == "amisulpride"
    assert len(generics) >= 4


def test_drugs_sought_by_this_phase_are_all_accounted_for():
    # Every drug P21.2 found reachable is either transcribed or recorded as
    # searched-and-empty. Silence is not an outcome this module permits.
    for drug in ("risperidone", "paliperidone", "olanzapine", "aripiprazole",
                 "quetiapine", "clozapine", "amisulpride"):
        assert records_for(drug)


def test_quetiapine_and_clozapine_are_present_only_as_negatives():
    for drug in ("quetiapine", "clozapine"):
        assert transcribed_records(drug) == ()
        assert searched_and_empty_records(drug)


def test_registered_drugs_is_sorted_and_matches_the_rows():
    assert registered_drugs() == tuple(sorted(registered_drugs()))
    assert set(registered_drugs()) == {row.drug for row in FDA_POPPK_RECORDS}
