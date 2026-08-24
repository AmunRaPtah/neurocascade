"""The five retired rodent drive-response series, tested where they now live.

P11.3 moved the whole graded-series library out of `decoder.SERIES` into
`legacy/rodent_series.py`: four mouse optogenetic series and one rat
microinjection series, all rodent, so none may inform a human readout under
the v2 roadmap. Quarantine-not-deletion means they stay importable AND
tested - these are the data pins that used to sit in the data section of
`tests/test_decoder.py`, re-pointed at `neurocascade.legacy.rodent_series`
and keeping the same job: a number transcribed wrongly fails here rather
than changing a verdict.

The rule tests stay in `test_decoder.py` and run against synthetic series.
This module adds the pairing assertions that keep the two libraries from
drifting: production stays empty, the five verdicts still reproduce through
the production machinery from their new home, and retiring them adopted
nothing - which is also what they did for production when they were there.
"""

import pytest

from neurocascade import decoder
from neurocascade import endpoints
from neurocascade.legacy import rodent_series as rs


def _series(key):
    for series in rs.SERIES:
        if series.key == key:
            return series
    raise AssertionError("no retired series %r" % (key,))


# ---------------------------------------------------------------------------
# The pairing: production empty, legacy intact.
# ---------------------------------------------------------------------------

def test_the_production_library_is_empty_and_legacy_holds_all_five():
    """Both halves of the move, pinned together so neither can drift.

    If `decoder.SERIES` grows again, this fails and the growth has to be a
    HUMAN series added deliberately; if a series disappears from legacy,
    the historical record is being rewritten.
    """
    assert decoder.SERIES == ()
    assert len(rs.SERIES) == 5
    assert {s.key for s in rs.SERIES} == {
        "herent2023_cnf_locomotion",
        "caggiano2018_cnf_locomotion",
        "caggiano2018_ppn_locomotion",
        "deluca2022_vlpo_sedation",
        "diaz2011_bla_anxiety",
    }


def test_every_retired_series_is_rodent():
    """The reason the move happened, checked rather than assumed."""
    assert {s.species for s in rs.SERIES} == {"mouse", "rat"}
    assert not any(s.species == "human" for s in rs.SERIES)


def test_every_retired_series_names_a_real_endpoint():
    for series in rs.SERIES:
        assert series.endpoint in endpoints.ENDPOINTS, series.key


def test_every_retired_series_targets_a_structure_the_endpoint_actually_reads():
    """A verdict about a region the readout does not contain changes nothing.

    Kept from production because it is why [Guo2023]'s parasubthalamic
    series and [Liang2023]'s supramammillary-to-septum series are in
    `SEARCHED_AND_EMPTY` and never were in `SERIES`.
    """
    for series in rs.SERIES:
        endpoint = endpoints.ENDPOINTS[series.endpoint]
        regions = set(endpoint.regions())
        assert set(series.structures) <= regions, series.key


def test_every_retired_citation_resolves_in_decoder():
    """One copy of each reference: the keys resolve in `decoder.CITATIONS`,
    which keeps all fifteen entries after the retirement."""
    for series in rs.SERIES:
        assert series.citation in decoder.CITATIONS, series.citation


# ---------------------------------------------------------------------------
# The verdicts, reproduced through the production machinery.
# ---------------------------------------------------------------------------

def test_the_five_series_and_the_endpoints_they_speak_to():
    """Which endpoints P7.2 could say anything about at all.

    Three of seven, all via rodent work. The other four have no graded
    single-region series with per-level numbers outside a figure, which was
    that task's main result and is recorded in `decoder.SEARCHED_AND_EMPTY`.
    """
    by_endpoint = {}
    for series in rs.SERIES:
        by_endpoint.setdefault(series.endpoint, []).append(series.key)
    assert set(by_endpoint) == {"locomotion", "sedation", "anxiety"}
    assert len(by_endpoint["locomotion"]) == 3
    assert len(by_endpoint["sedation"]) == 1
    assert len(by_endpoint["anxiety"]) == 1


def test_each_series_has_the_verdict_its_shape_implies():
    """The five verdicts, pinned, reproduced from quarantine.

    This is a REGRESSION PIN and not an independent derivation - the table
    was written after the comparison was first run, and it would agree with
    a wrong rule as readily as with a right one. It is worth having anyway,
    for the same reason `test_nothing_was_adopted` is: it makes any change to
    a verdict a visible, deliberate edit. The tests that can actually catch a
    wrong rule run on synthetic series in `tests/test_decoder.py`; the test
    that can catch a wrong number is
    `test_the_published_numbers_are_what_the_papers_print`, which restates
    the values independently.
    """
    expected = {
        "herent2023_cnf_locomotion": "weighting-dependent",
        "caggiano2018_cnf_locomotion": "no-signal",
        "caggiano2018_ppn_locomotion": "linear-wins",
        "deluca2022_vlpo_sedation": "saturating-wins-not-identifiable",
        "diaz2011_bla_anxiety": "too-few-levels",
    }
    assert set(expected) == {s.key for s in rs.SERIES}
    for series in rs.SERIES:
        verdict = decoder.assess_series(series)
        assert verdict.reason == expected[series.key], series.key


def test_nothing_was_adopted_then_and_nothing_is_adopted_now():
    """Retiring the library changed no prediction, because none of the five
    had licensed a nonlinearity when they lived in production. Both halves
    asserted: no adoption through the retired series directly, and an empty
    adoption list in production where they no longer sit."""
    for series in rs.SERIES:
        assert not decoder.assess_series(series).adopted, series.key
    assert decoder.adopted_nonlinearities() == {}


def test_the_saturating_case_that_the_brief_worried_about_is_a_real_one():
    """[DeLuca2022] is the brief's own concern, met by real data.

    "Behavioral endpoints are frequently saturating" - and this one is: the
    saturating family beats the straight line four-fold out of sample, under
    both weightings. It is still rejected, because a linear decoder reading a
    region whose rate saturates in its own drive produces exactly that curve.
    """
    verdict = decoder.assess_series(_series("deluca2022_vlpo_sedation"))
    for weighting in decoder.WEIGHTINGS:
        assessment = verdict.by_weighting[weighting]
        satexp = assessment.fits_by_name["satexp"]
        linear = assessment.fits_by_name["linear"]
        assert satexp.loo_rmse < linear.loo_rmse, weighting
        assert assessment.best.name == "satexp", weighting
    assert verdict.verdict == "keep-linear"
    assert verdict.reason == "saturating-wins-not-identifiable"


def test_the_weighting_dependent_case_still_runs_under_both_weightings():
    """Rule 6's real case keeps exercising the combination logic even though
    the series itself is quarantined: `tests/test_decoder.py` runs the rule
    half of this on the retired object, and this pin holds the two
    weightings' disagreement itself."""
    series = _series("herent2023_cnf_locomotion")
    uniform = decoder._assess_under(series, "uniform")
    dispersion = decoder._assess_under(series, "dispersion")
    assert uniform.adopted
    assert not dispersion.adopted


# ---------------------------------------------------------------------------
# The transcription pins.
# ---------------------------------------------------------------------------

def test_the_published_numbers_are_what_the_papers_print():
    """The transcription, checked against the values read off each source.

    Written out a second time here rather than imported, so that an edit to
    `rs.SERIES` has to be made twice to go unnoticed.
    """
    herent = _series("herent2023_cnf_locomotion")
    assert herent.levels == (5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    assert herent.values == pytest.approx(
        (0.187214, 0.128781, 54.3845, 67.4137, 84.8552, 85.9764)
    )
    assert herent.dispersion_kind == "SD"
    assert herent.n_per_level == (5, 5, 5, 5, 5, 5)

    cnf = _series("caggiano2018_cnf_locomotion")
    assert cnf.levels == (5.0, 10.0, 20.0, 50.0)
    assert cnf.n_per_level == (14, 41, 37, 39)

    ppn = _series("caggiano2018_ppn_locomotion")
    assert ppn.levels == (5.0, 10.0, 20.0, 50.0)
    assert ppn.n_per_level == (4, 21, 18, 24)

    vlpo = _series("deluca2022_vlpo_sedation")
    assert vlpo.levels == (0.0, 1.0, 5.0, 10.0, 20.0)
    assert vlpo.values == pytest.approx(
        (32.242, 33.701, 48.206, 53.854, 59.234)
    )
    assert vlpo.score_sign == -1

    bla = _series("diaz2011_bla_anxiety")
    assert bla.levels == (0.0, 1.5, 5.0)
    assert bla.values == pytest.approx((7.5, 41.7, 104.8))
    assert bla.score_sign == -1


def test_the_diaz_table_sums_to_the_trial_duration():
    """The one internal check [Diaz2011]'s Table 1 supports.

    Open, closed and junction times are the three parts of a 300 s trial, so
    they must sum to it. They do, to 0.1 s. This catches a transposed digit
    in a way no restatement of the same number can.
    """
    open_arm = (7.5, 41.7, 104.8)
    closed_arm = (208.3, 173.9, 106.6)
    junction = (84.3, 84.5, 88.5)
    assert _series("diaz2011_bla_anxiety").values == pytest.approx(open_arm)
    for o, c, j in zip(open_arm, closed_arm, junction):
        assert o + c + j == pytest.approx(300.0, abs=0.15)
