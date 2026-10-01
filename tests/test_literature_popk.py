"""P25.1: adult oral population PK for six antipsychotics, from the journal literature.

Schema tests defend the rules the search established (fixed parameters carry an
origin, a summary is not a transcription, every drug has a recorded search).
Algebra tests check the half-life arithmetic against closed forms and against
the one half-life a source printed itself. Result tests pin the comparison
table, failures included, and check that the document carries it verbatim.
"""

import math
import pathlib

import pytest

from neurocascade.human.fda_popk import PopPKParameter
from neurocascade.human.literature_popk import (
    CANDIDATE_BARRIERS,
    DRUG_SEARCHES,
    LITERATURE_POPPK_RECORDS,
    SEARCHED_DRUGS,
    Candidate,
    DrugSearch,
    LiteraturePopPK,
    SearchQuery,
    Variability,
    comparison_table,
    drug_search,
    half_life_comparisons,
    record,
    records_for,
    searched_and_empty_drugs,
    terminal_half_life_h,
    terminal_phase_auc_fraction,
)

DOC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "p25-1-literature-popk.md"


def _record(**overrides):
    kwargs = dict(
        drug="quetiapine",
        formulation="tablet",
        analyte="parent",
        citation="Test A. 2020. Title. J 1:1",
        pmid="1",
        doi="10/x",
        read_from="raw",
        population="adults",
        structural_model="one compartment",
        apparent_parameters=True,
        apparent_evidence="printed as CL/F",
        software="NONMEM",
        n_subjects=10,
        n_observations=20,
        parameters=(
            PopPKParameter("CL/F", 10.0, "L/h"),
            PopPKParameter("V/F", 100.0, "L"),
            PopPKParameter("Ka", 1.0, "1/h"),
        ),
        disposition_slots=(("parent", (("CL", "CL/F"), ("V1", "V/F"))),),
    )
    kwargs.update(overrides)
    return LiteraturePopPK(**kwargs)


# ---------------------------------------------------------------------------
# Search coverage
# ---------------------------------------------------------------------------


def test_every_drug_in_the_brief_has_exactly_one_recorded_search():
    assert tuple(s.drug for s in DRUG_SEARCHES) == SEARCHED_DRUGS
    assert SEARCHED_DRUGS == (
        "clozapine", "quetiapine", "ziprasidone", "aripiprazole", "risperidone", "paliperidone",
    )


def test_every_search_records_the_five_standard_queries_with_hit_counts():
    for search in DRUG_SEARCHES:
        standard = search.queries[:5]
        assert [q.database for q in standard] == ["Europe PMC"] * 4 + ["PubMed"]
        assert all(search.drug in q.query for q in standard)
        assert all(isinstance(q.hits, int) and q.hits >= 0 for q in search.queries)
        assert all(q.run_date == "2026-10-01" for q in search.queries)


def test_hit_counts_as_run():
    counts = {s.drug: tuple(q.hits for q in s.queries[:5]) for s in DRUG_SEARCHES}
    assert counts == {
        "clozapine": (169, 74, 24, 12, 33),
        "quetiapine": (141, 70, 11, 3, 10),
        "ziprasidone": (53, 30, 8, 1, 2),
        "aripiprazole": (157, 77, 17, 8, 25),
        "risperidone": (217, 129, 22, 16, 32),
        "paliperidone": (92, 56, 9, 6, 13),
    }


def test_ziprasidone_is_the_one_searched_and_empty_drug():
    assert searched_and_empty_drugs() == ("ziprasidone",)
    assert records_for("ziprasidone") == ()
    search = drug_search("ziprasidone")
    by_pmid = {c.pmid: c for c in search.candidates}
    assert by_pmid["21209243"].disposition == "paywalled"
    assert by_pmid["38918237"].disposition == "abstract-only"
    assert "18.74 L/h" in by_pmid["38918237"].reason
    assert len(search.queries) == 9


def test_every_other_drug_has_a_transcribed_record_matching_its_search():
    for search in DRUG_SEARCHES:
        transcribed = {c.citation for c in search.candidates if c.disposition == "transcribed"}
        assert transcribed == {r.citation for r in records_for(search.drug)}
        if search.drug != "ziprasidone":
            assert search.outcome == "transcribed"
            assert transcribed


def test_every_refusal_names_a_known_barrier_and_a_reason():
    for search in DRUG_SEARCHES:
        for c in search.candidates:
            assert c.disposition == "transcribed" or c.disposition in CANDIDATE_BARRIERS
            assert c.reason


def test_summarised_full_texts_are_refused_not_transcribed():
    kim = {c.pmid: c for c in drug_search("aripiprazole").candidates}["19032724"]
    feng = {c.pmid: c for c in drug_search("risperidone").candidates}["18771484"]
    assert kim.disposition == feng.disposition == "summariser-only"
    assert "192" in kim.reason and "193" in kim.reason


def test_search_outcome_must_agree_with_its_candidates():
    q = (SearchQuery("PubMed", "x", 0, "2026-10-01"),)
    with pytest.raises(ValueError):
        DrugSearch("x", q, (Candidate("A", "1", "transcribed", "r"),), "searched-and-empty", "s")
    with pytest.raises(ValueError):
        DrugSearch("x", q, (Candidate("A", "1", "paywalled", "r"),), "transcribed", "s")
    with pytest.raises(ValueError):
        DrugSearch("x", (), (), "searched-and-empty", "s")
    with pytest.raises(ValueError):
        Candidate("A", "1", "lost", "r")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_registry_holds_eight_models_across_five_drugs():
    assert [r.short_citation for r in LITERATURE_POPPK_RECORDS] == [
        "Li 2012", "Reeves 2024", "Chen 2024", "Zheng 2024",
        "Koue 2007", "Knights 2015", "Størset 2024", "Korell 2017",
    ]
    assert {r.pmid for r in LITERATURE_POPPK_RECORDS} == {
        "22820910", "36793249", "39635430", "39698209",
        "17978491", "25821065", "38963454", "28614899",
    }


def test_every_record_is_cited_read_from_a_raw_copy_and_located_per_parameter():
    for r in LITERATURE_POPPK_RECORDS:
        assert r.pmid and r.doi and r.read_from
        assert "summar" not in r.read_from.lower()
        assert r.route == "PO"
        assert r.apparent_parameters
        for p in r.parameters:
            assert p.source, (r.short_citation, p.name)


def test_every_record_carries_variability_and_covariates():
    for r in LITERATURE_POPPK_RECORDS:
        assert r.variability, r.short_citation
        assert r.covariates, r.short_citation
        assert r.residual_error, r.short_citation


@pytest.mark.parametrize(
    "drug, author, name, value, units",
    [
        ("clozapine", "Li", "CL/F", 21.9, "L/h"),
        ("clozapine", "Li", "V/F", 526.0, "L"),
        ("clozapine", "Reeves", "CL/Fclozapine", 20.2, "L/h"),
        ("clozapine", "Reeves", "V/Fclozapine", 1460.0, "L"),
        ("clozapine", "Reeves", "Km", 0.005, "1/h"),
        ("quetiapine", "Chen", "CL/F", 118.0, "L/h"),
        ("quetiapine", "Chen", "V/F", 2460.0, "L"),
        ("quetiapine", "Zheng", "CL/F", 76.1, "L/h"),
        ("quetiapine", "Zheng", "V/F", 530.0, "L"),
        ("aripiprazole", "Koue", "θ4 CL/F", 0.0645, "L/h/kg"),
        ("aripiprazole", "Koue", "θ3 V1/F", 3.84, "L/kg"),
        ("aripiprazole", "Koue", "θ8 V2/F", 1.54, "L/kg"),
        ("aripiprazole", "Koue", "θ9 Q/F", 0.168, "L/h/kg"),
        ("aripiprazole", "Knights", "CL", 3.88, "L/h"),
        ("aripiprazole", "Knights", "VC", 192.0, "L"),
        ("aripiprazole", "Knights", "Q", 12.2, "L/h"),
        ("aripiprazole", "Knights", "VP", 151.0, "L"),
        ("risperidone", "Størset", "V/F", 333.0, "L"),
        ("risperidone", "Størset", "CL/F typical normal metabolizer (CYP2D6*1/*1)", 50.6, "L/h"),
        ("risperidone", "Størset", "Vmet/F × fmet", 96.0, "L"),
        ("risperidone", "Størset", "CLmet/F × fmet", 8.0, "L/h"),
        ("paliperidone", "Korell", "CL/F", 10.9, "L/h"),
        ("paliperidone", "Korell", "Vc/F", 198.0, "L"),
        ("paliperidone", "Korell", "CLp/F", 22.0, "L/h"),
        ("paliperidone", "Korell", "Vp/F", 244.0, "L"),
        ("paliperidone", "Korell", "Effect of CrCL on CL/F", 0.0240, None),
    ],
)
def test_transcribed_disposition_values(drug, author, name, value, units):
    p = record(drug, author).parameter(name)
    assert p.value == value
    assert p.units == units


def test_borrowed_absorption_constants_are_flagged_with_their_origin():
    fixed = {r.short_citation: dict(r.fixed_parameters) for r in LITERATURE_POPPK_RECORDS}
    assert "25975812" in fixed["Chen 2024"]["Ka"]
    assert "25975812" in fixed["Zheng 2024"]["Ka"]
    assert "28614899" in fixed["Størset 2024"]["ka"]
    assert fixed["Koue 2007"] == {}
    assert fixed["Knights 2015"] == {}
    assert record("clozapine", "Reeves").is_fixed("Ka")


def test_variability_scales_are_as_printed():
    scales = {
        r.short_citation: {v.parameter: v.scale for v in r.variability}
        for r in LITERATURE_POPPK_RECORDS
    }
    assert scales["Reeves 2024"]["CL/Fclozapine"] == "standard-deviation"
    assert scales["Knights 2015"]["CL"] == "variance"
    assert scales["Li 2012"]["CL/F"] == "percent-cv"
    assert scales["Chen 2024"]["CL/F"] == "unstated"
    assert scales["Zheng 2024"]["CL/F"] == "unstated"
    korell = record("paliperidone", "Korell")
    assert [v.parameter for v in korell.variability if v.inter_occasion] == ["F1"]
    with pytest.raises(ValueError):
        Variability("CL/F", 0.3, "guess")


def test_reference_weights_are_only_the_printed_ones():
    weights = {r.short_citation: r.reference_weight_kg for r in LITERATURE_POPPK_RECORDS}
    assert weights == {
        "Li 2012": None, "Reeves 2024": 70.0, "Chen 2024": 70.0, "Zheng 2024": 70.0,
        "Koue 2007": None, "Knights 2015": 74.19, "Størset 2024": None, "Korell 2017": 74.4,
    }


def test_schema_refuses_bad_records():
    with pytest.raises(ValueError):
        _record(apparent_parameters=False)
    with pytest.raises(ValueError):
        _record(fixed_parameters=(("Ka", ""),))
    with pytest.raises(ValueError):
        _record(fixed_parameters=(("KA", "elsewhere"),))
    with pytest.raises(ValueError):
        _record(disposition_slots=(("parent", (("CL", "CL/F"),)),))
    with pytest.raises(ValueError):
        _record(variability=(Variability("Q/F", 0.1, "variance"),))
    with pytest.raises(ValueError):
        _record(drug="haloperidol")
    with pytest.raises(ValueError):
        _record(apparent_evidence="")


def test_mixed_unit_families_are_refused():
    bad = _record(parameters=(
        PopPKParameter("CL/F", 0.1, "L/h/kg"),
        PopPKParameter("V/F", 100.0, "L"),
    ))
    with pytest.raises(ValueError):
        terminal_half_life_h(bad, "parent")


def test_unitless_disposition_parameter_is_refused():
    bad = _record(parameters=(PopPKParameter("CL/F", 10.0, None), PopPKParameter("V/F", 100.0, "L")))
    with pytest.raises(ValueError):
        terminal_half_life_h(bad, "parent")


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------


def test_one_compartment_half_life_is_ln2_v_over_cl():
    assert terminal_half_life_h(record("clozapine", "Li"), "parent") == pytest.approx(
        math.log(2) * 526.0 / 21.9
    )


def test_parallel_metabolic_rate_reproduces_the_source_own_half_life():
    """Reeves 2024 prints t½ = 36 h for its reference subject; with km it is 36.8."""
    reeves = record("clozapine", "Reeves")
    t = terminal_half_life_h(reeves, "parent")
    assert t == pytest.approx(math.log(2) / (20.2 / 1460.0 + 0.005))
    assert round(t) in (36, 37)
    assert math.log(2) * 1460.0 / 20.2 > 49.0


def test_two_compartment_half_life_against_closed_form():
    cl, v1, q, v2 = 3.88, 192.0, 12.2, 151.0
    k10, k12, k21 = cl / v1, q / v1, q / v2
    b = k10 + k12 + k21
    beta = (b - math.sqrt(b * b - 4 * k10 * k21)) / 2
    assert terminal_half_life_h(record("aripiprazole", "Knights"), "parent") == pytest.approx(
        math.log(2) / beta
    )


def test_two_compartment_auc_fraction_against_closed_form():
    cl, v1, q, v2 = 3.88, 192.0, 12.2, 151.0
    k10, k12, k21 = cl / v1, q / v1, q / v2
    b = k10 + k12 + k21
    disc = math.sqrt(b * b - 4 * k10 * k21)
    alpha, beta = (b + disc) / 2, (b - disc) / 2
    a = (alpha - k21) / (alpha - beta)
    bb = (k21 - beta) / (alpha - beta)
    expected = (bb / beta) / (a / alpha + bb / beta)
    got = terminal_phase_auc_fraction(record("aripiprazole", "Knights"), "parent")
    assert got == pytest.approx(expected)


def test_per_kg_set_gives_a_weight_free_half_life():
    koue = record("aripiprazole", "Koue")
    expected = terminal_half_life_h(_record(
        drug="aripiprazole",
        parameters=(
            PopPKParameter("CL/F", 0.0645 * 60.7, "L/h"),
            PopPKParameter("V1/F", 3.84 * 60.7, "L"),
            PopPKParameter("Q/F", 0.168 * 60.7, "L/h"),
            PopPKParameter("V2/F", 1.54 * 60.7, "L"),
        ),
        disposition_slots=(("parent", (("CL", "CL/F"), ("V1", "V1/F"), ("Q2", "Q/F"), ("V2", "V2/F"))),),
    ), "parent")
    assert terminal_half_life_h(koue, "parent") == pytest.approx(expected)


def test_creatinine_clearance_bound_uses_only_the_printed_cap():
    korell = record("paliperidone", "Korell")
    slow = terminal_half_life_h(korell, "parent")
    fast = terminal_half_life_h(korell, "parent", 0.0240 * 150.0)
    c = [c for c in half_life_comparisons() if c.drug == "paliperidone"][0]
    assert (c.predicted_low_h, c.predicted_high_h) == pytest.approx((fast, slow))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

EXPECTED = [
    ("clozapine", "Li 2012", "parent", 16.65, 16.65, 1.39, True),
    ("clozapine", "Reeves 2024", "parent", 36.80, 36.80, 3.07, False),
    ("quetiapine", "Chen 2024", "parent", 14.45, 14.45, 2.41, False),
    ("quetiapine", "Zheng 2024", "parent", 4.83, 4.83, 1.24, True),
    ("aripiprazole", "Koue 2007", "parent", 59.78, 59.78, 1.25, True),
    ("aripiprazole", "Knights 2015", "parent", 65.35, 65.35, 1.15, True),
    ("risperidone", "Størset 2024", "parent", 4.56, 4.56, 1.52, True),
    ("risperidone", "Størset 2024", "metabolite", 8.32, 8.32, 2.52, False),
    ("paliperidone", "Korell 2017", "parent", 26.02, 32.85, 1.43, True),
]


def test_half_life_comparisons_are_pinned_including_failures():
    got = [
        (c.drug, c.source, c.entity, round(c.predicted_low_h, 2), round(c.predicted_high_h, 2),
         round(c.fold_error, 2), c.agrees_within_twofold)
        for c in half_life_comparisons()
    ]
    assert got == EXPECTED


def test_six_of_nine_pass_and_the_threshold_is_twofold():
    results = half_life_comparisons()
    assert sum(c.agrees_within_twofold for c in results) == 6
    assert len(results) == 9
    assert all(c.agrees_within_twofold == (c.fold_error <= 2.0) for c in results)


def test_label_sources_for_the_new_drugs_are_innovator_labels():
    labels = {c.drug: c.label for c in half_life_comparisons()}
    assert labels["clozapine"].innovator and "NDA 019758" in labels["clozapine"].source
    assert labels["quetiapine"].innovator and "NDA 020639" in labels["quetiapine"].source
    assert labels["clozapine"].low_h == labels["clozapine"].high_h == 12.0
    assert labels["quetiapine"].low_h == 6.0


def test_document_carries_the_table_verbatim():
    assert comparison_table() in DOC.read_text(encoding="utf-8")
