"""The P22.1 pre-registration exists and agrees with the records it points at.

The document's authority comes from being committed before any P22 result, and
git history is the proof of that ordering, not these tests. What tests can
guard is drift: the pre-registered denominators are defined by reference to
committed code (`fda_sources.TESTABLE` for P22.2, the P21.2 census for P22.3),
so if either side moves without the other, a P22 task would score against a
denominator the pre-registration never fixed. These assertions make that
divergence loud.
"""

from pathlib import Path

import pytest

from neurocascade.human import fda_sources

DOC = Path(__file__).resolve().parent.parent / "docs" / "p22-preregistration.md"

#: The P21.2 reachability census, which the pre-registration fixes as the
#: P22.3 denominator. Spelled out here rather than imported because no module
#: exports it: P21.2 was a documentation task, and the point of this constant
#: is to notice if a later task changes the census in the doc without a
#: deliberate edit here.
P22_3_DENOMINATOR = {
    "haloperidol",
    "risperidone",
    "olanzapine",
    "clozapine",
    "ziprasidone",
    "amisulpride",
    "aripiprazole",
    "paliperidone",
    "quetiapine",
}


@pytest.fixture(scope="module")
def text():
    assert DOC.is_file(), "the P22.1 pre-registration document is missing"
    return DOC.read_text()


def test_p22_2_denominator_matches_fda_sources(text):
    """The eight-drug P22.2 denominator is fda_sources.TESTABLE, verbatim."""
    assert len(fda_sources.TESTABLE) == 8
    for drug in fda_sources.TESTABLE:
        assert drug in text, "pre-registered P22.2 denominator omits %s" % drug


def test_amisulpride_split_is_recorded(text):
    """Amisulpride: out of P22.2 (no US oral label), in the P22.3 census."""
    assert fda_sources.LABELS_BY_DRUG["amisulpride"]["status"] == "searched-and-empty"
    assert "Amisulpride is excluded" in text
    assert "It remains in the P22.3 denominator" in text


def test_p22_3_denominator_named_in_full(text):
    for drug in P22_3_DENOMINATOR:
        assert drug in text.lower(), (
            "pre-registered P22.3 denominator omits %s" % drug
        )


def test_pass_criteria_are_the_preregistered_ones(text):
    """The thresholds this document exists to freeze are actually in it."""
    for token in (
        "log10(2)",          # P22.2 twofold midpoint criterion
        "MAE ≤ 0.10",        # P22.3 same-tracer pass
        "≤ 0.15",            # P22.3 cross-tracer pass
        "0.65",              # clinical-window lower threshold
        "0.78",              # clinical-window upper threshold
        "steady-state-daily",  # the regimen statistic the doses are defined on
    ):
        assert token in text, "pre-registration lost its %r criterion" % token


def test_negative_result_commitment_is_explicit(text):
    """Acceptance criterion of P22.1: the document must state this in terms."""
    assert "A negative result will be reported and is publishable" in text


def test_outcome_map_for_clozapine_is_fixed(text):
    """P22.4's reading may not bend to the number: both outcomes are mapped."""
    assert "Outcome A" in text and "Outcome B" in text
    assert "nothing" in text and "tuned" in text
