"""P7.2: the decoder model comparison, and what it is allowed to adopt.

The tests split into three groups, and the split matters:

  * the DECISION RULE, checked against synthetic series whose true shape is
    known by construction. These are the tests that can fail if the rule is
    wrong, and they include the one that matters most - a saturating series
    must NOT be adopted, because saturation is what the circuit layer's own
    sigmoid already supplies;
  * the NUMERICS, checked against properties that hold for any data
    (monotonicity of the `satexp` basis, the leave-one-out arithmetic, the
    conditioning bound that stops an unidentifiable fit from handing the
    comparison to `quadratic`);
  * the RECORD, which since P11.3 is an EMPTY library: every graded series
    this module ever held was rodent and moved to
    `neurocascade.legacy.rodent_series`, so these tests pin the emptiness,
    the all-endpoints `no-data` verdicts it implies, and the fact that the
    decision rule above still runs - including on the one real case that
    motivated rule 6, reached through the quarantined copy. The published
    numbers themselves are pinned in `tests/test_legacy_rodent_series.py`.
"""

import math

import numpy as np
import pytest

from neurocascade import behavior
from neurocascade.legacy import connectome
from neurocascade.legacy import rodent_series
from neurocascade import decoder
from neurocascade import endpoints


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def synthetic(levels, values, **kwargs):
    """A series with a known shape, for testing the rule and not the data."""
    kwargs.setdefault("key", "synthetic")
    kwargs.setdefault("endpoint", "analgesia")
    kwargs.setdefault("structures", ("RM",))
    kwargs.setdefault("manipulation", "electrical_stimulation")
    kwargs.setdefault("drive_units", "uA")
    kwargs.setdefault("value_units", "s")
    kwargs.setdefault("score_sign", 1)
    kwargs.setdefault("citation", "synthetic")
    kwargs.setdefault("species", "rat")
    kwargs.setdefault("finding", "constructed for a test")
    kwargs.setdefault("provenance", "table")
    return decoder.DriveResponseSeries(levels=levels, values=values, **kwargs)


SIX = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


# ---------------------------------------------------------------------------
# the decision rule
# ---------------------------------------------------------------------------

def test_a_clean_reversal_is_adopted():
    v = decoder.assess_series(synthetic(SIX, [0.0, 1.0, 1.6, 1.6, 1.0, 0.0]))
    assert v.verdict == "adopt-nonlinear"
    assert v.best.name == "quadratic"
    assert v.shape == "inverted_U"
    assert 0.0 < v.turning_point_z < 1.0


def test_a_reversal_in_the_other_direction_is_adopted_as_a_U():
    v = decoder.assess_series(synthetic(SIX, [0.0, -1.0, -1.6, -1.6, -1.0,
                                              0.0]))
    assert v.verdict == "adopt-nonlinear"
    assert v.shape == "U"


def test_a_saturating_series_is_not_adopted():
    """The central claim of the module.

    A behaviour that bends over as one region's drive is raised is exactly
    what a linear decoder over a SATURATING drive-to-rate map produces. If
    this test ever passes an adoption, the module is attributing the
    Wilson-Cowan sigmoid to the readout and double-counting it.
    """
    v = decoder.assess_series(
        synthetic(SIX, [0.0, 0.60, 0.85, 0.95, 0.98, 0.99])
    )
    assert v.verdict == "keep-linear"
    assert v.best.name == "satexp"
    assert v.reason == "saturating-wins-not-identifiable"


def test_a_linear_series_is_not_adopted():
    v = decoder.assess_series(synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert v.verdict == "keep-linear"
    assert v.best.name == "linear"
    assert v.reason == "linear-wins"


def test_a_flat_series_reports_no_signal_rather_than_a_shape():
    v = decoder.assess_series(
        synthetic(SIX, [1.00, 0.99, 1.01, 1.00, 0.98, 1.02])
    )
    assert v.verdict == "keep-linear"
    assert v.reason == "no-signal"


def test_a_saturating_series_whose_vertex_falls_just_inside_is_still_rejected():
    """The false positive that rule 5 was written to stop, kept as data.

    `0, 3, 5.2, 6.6, 7.4, 7.7` is strictly increasing and decelerating - the
    canonical saturating shape, and the shape the identifiability argument
    says carries NO information about the decoder. `quadratic` nonetheless
    beats `satexp` out of sample on it and puts its vertex at z = 0.945,
    inside the tested range, so rules 2 and 4 both pass and the series was
    adopted before rule 5 existed. Deleting the `observed_shape` check in
    `assess_series` turns this test red, which is the point of it.
    """
    v = decoder.assess_series(synthetic(SIX, [0.0, 3.0, 5.2, 6.6, 7.4, 7.7]))
    assert v.series.observed_shape() == "increasing"
    assert v.best.name == "quadratic"
    assert 0.0 < v.turning_point_z < 1.0
    assert v.verdict == "keep-linear"
    assert v.reason == "no-measured-reversal"


def test_the_rejection_reason_is_turning_point_outside_range_when_rule_4_bites():
    """Rule 4 must actually be reachable, or the branch is dead code.

    Constructed rather than found: a series that is quadratic BY
    CONSTRUCTION with its vertex past the last level, so `quadratic` fits it
    exactly and wins rule 2, and only rule 4 can reject it.
    """
    z = np.linspace(0.0, 1.0, 6)
    y = 2.0 * z - 0.6 * z ** 2          # vertex at z = 5/3, well outside
    v = decoder.assess_series(synthetic(SIX, list(y)))
    assert v.best.name == "quadratic"
    assert v.turning_point_z > 1.0
    assert v.verdict == "keep-linear"
    assert v.reason == "turning-point-outside-range"


def test_both_weightings_are_run_and_disagreement_is_a_rejection():
    """Rule 6's combination logic, checked on a series where they disagree.

    Not constructed: [Herent2023] is the real case and the reason the rule
    exists, so it is what the test uses - reached through its quarantined
    copy in `legacy/rodent_series`, since P11.3 retired it from production
    without retiring the rule that exists because of it. Under uniform
    weights the quadratic wins, its vertex is inside the range and the means
    are ordinally non-monotone, so all of rules 2-5 pass and the series is
    ADOPTED. Under `1/SEM^2` the straight line wins. The dip that makes the
    means non-monotone is 0.187 -> 0.129 cm/s between two frequencies that
    evoke no locomotion in any animal, and it is smaller than either level's
    own SD - so rules 2-5 were satisfied by a floor-effect wiggle, and rule
    6 is what caught it.
    """
    series = next(s for s in rodent_series.SERIES
                  if s.key == "herent2023_cnf_locomotion")
    uniform = decoder._assess_under(series, "uniform")
    dispersion = decoder._assess_under(series, "dispersion")
    assert uniform.adopted
    assert not dispersion.adopted
    verdict = decoder.assess_series(series)
    assert verdict.verdict == "keep-linear"
    assert verdict.reason == "weighting-dependent"
    assert set(verdict.by_weighting) == set(decoder.WEIGHTINGS)


def test_every_rejection_reason_is_reachable():
    """No branch of the decision rule is unexercised prose.

    `no-data` is reached by every endpoint since P11.3 emptied the library,
    and `weighting-dependent` by [Herent2023] through the quarantined series
    in the test above (its numbers pinned in
    `tests/test_legacy_rodent_series.py`). Everything else is constructed
    here.
    """
    z = np.linspace(0.0, 1.0, 6)
    cases = {
        "too-few-levels": synthetic([0.0, 1.0, 2.0], [0.0, 1.0, 0.1]),
        "no-signal": synthetic(SIX, [1.0, 0.99, 1.01, 1.0, 0.98, 1.02]),
        "linear-wins": synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "saturating-wins-not-identifiable":
            synthetic(SIX, [0.0, 0.6, 0.85, 0.95, 0.98, 0.99]),
        "turning-point-outside-range":
            synthetic(SIX, list(2.0 * z - 0.6 * z ** 2)),
        "no-measured-reversal":
            synthetic(SIX, [0.0, 3.0, 5.2, 6.6, 7.4, 7.7]),
    }
    for expected, series in cases.items():
        assert decoder.assess_series(series).reason == expected, expected
    reached = set(cases) | {"no-data", "weighting-dependent"}
    assert reached == set(decoder.REJECTION_REASONS)


def test_three_levels_is_too_few_to_cross_validate_a_three_parameter_family():
    v = decoder.assess_series(synthetic([0.0, 1.0, 2.0], [0.0, 1.0, 0.1]))
    assert v.verdict == "keep-linear"
    assert v.reason == "too-few-levels"
    assert decoder.FAMILIES_BY_NAME["quadratic"].n_params == 3
    quad = v.fits_by_name["quadratic"]
    assert quad.loo_rmse is None, (
        "with 3 levels a 3-parameter family interpolates its 2-point "
        "training set, so a leave-one-out number here would be an artefact"
    )


def test_the_head_to_head_families_have_the_same_parameter_count():
    """`satexp` versus `quadratic` is the comparison the verdict turns on.

    If they ever differ in parameter count, a verdict stops being about
    monotone-versus-not and starts being about complexity.
    """
    a, b = decoder.DISCRIMINATING_PAIR
    assert (decoder.FAMILIES_BY_NAME[a].n_params
            == decoder.FAMILIES_BY_NAME[b].n_params)
    assert decoder.FAMILIES_BY_NAME[a].monotone
    assert not decoder.FAMILIES_BY_NAME[b].monotone


def test_only_one_family_is_capable_of_a_reversal():
    assert [f.name for f in decoder.FAMILIES if not f.monotone] == \
        ["quadratic"]


# ---------------------------------------------------------------------------
# the numerics
# ---------------------------------------------------------------------------

def test_the_satexp_basis_is_monotone_and_spans_the_unit_interval():
    """Monotone is the property the identifiability argument needs.

    Non-DECREASING and not strictly increasing, because at large `lam` the
    basis reaches 1.0 in double precision and stays there. That flat tail is
    the regime `_resolvable_lam_max` excludes from fitting; here it only has
    to not run backwards.
    """
    z = np.linspace(0.0, 1.0, 101)
    for lam in (0.01, 0.1, 1.0, 5.0, 50.0):
        b = decoder._satexp_basis(lam, z)
        assert b[0] == pytest.approx(0.0, abs=1e-12)
        assert b[-1] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(b) >= 0.0), lam
    for lam in (0.01, 0.1, 1.0, 5.0):
        assert np.all(np.diff(decoder._satexp_basis(lam, z)) > 0.0), lam


def test_the_satexp_basis_reduces_to_the_linear_one_as_lam_vanishes():
    z = np.linspace(0.0, 1.0, 11)
    assert decoder._satexp_basis(1e-6, z) == pytest.approx(z, abs=1e-5)


def test_satexp_stays_finite_on_a_flat_series():
    """A regression test with a number in it, because the number was 2e6.

    Without `_resolvable_lam_max`, leave-one-out on a flat six-level series
    picks a `lam` whose basis is a step at the held-out level, the two
    columns go collinear, and the solver's tie-break decides the prediction.
    The failure is not cosmetic: an inflated `satexp` error can only ever
    hand the comparison to `quadratic`, which is a false adoption.
    """
    series = synthetic(SIX, [1.00, 0.99, 1.01, 1.00, 0.98, 1.02])
    fit = decoder.fit_family(series,
                             decoder.FAMILIES_BY_NAME["satexp"], "uniform")
    spread = max(series.values) - min(series.values)
    assert fit.loo_rmse < 100.0 * spread


def test_the_resolvable_lam_bound_is_read_off_the_design():
    z = np.linspace(0.0, 1.0, 6)
    assert decoder._resolvable_lam_max(z) == pytest.approx(
        math.log(100.0) / 0.2
    )
    finer = np.linspace(0.0, 1.0, 11)
    assert decoder._resolvable_lam_max(finer) > decoder._resolvable_lam_max(z)


def test_leave_one_out_of_the_constant_model_is_the_jackknife_mean():
    """One family's LOO computed a second, independent way."""
    values = [1.0, 2.0, 4.0, 8.0]
    series = synthetic([0.0, 1.0, 2.0, 3.0], values)
    fit = decoder.fit_family(series,
                             decoder.FAMILIES_BY_NAME["constant"])
    expected = []
    for i in range(4):
        rest = [v for j, v in enumerate(values) if j != i]
        expected.append(values[i] - sum(rest) / 3.0)
    assert fit.loo_rmse == pytest.approx(
        math.sqrt(sum(e ** 2 for e in expected) / 4.0)
    )


def test_leave_one_out_error_exceeds_in_sample_error_for_the_big_families():
    series = synthetic(SIX, [0.0, 1.0, 1.6, 1.6, 1.0, 0.0])
    for name in ("linear", "satexp", "quadratic"):
        fit = decoder.fit_family(series, decoder.FAMILIES_BY_NAME[name])
        assert fit.loo_rmse >= fit.in_sample_rmse - 1e-9, name


def test_a_series_with_sems_is_weighted_by_them():
    series = synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                       dispersion=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       dispersion_kind="SEM")
    w, how = series.weights("dispersion")
    assert how == "1/SEM^2"
    assert np.allclose(w, 1.0)
    assert series.weights("uniform")[1] == "uniform"


def test_an_sd_without_a_group_size_falls_back_to_uniform_weights():
    series = synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                       dispersion=[1.0] * 6, dispersion_kind="SD")
    w, how = series.weights("dispersion")
    assert np.allclose(w, 1.0)
    assert "no group size" in how


def test_an_sd_with_a_group_size_becomes_a_sem():
    series = synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                       dispersion=[2.0] * 6, dispersion_kind="SD",
                       n_per_level=[4] * 6)
    w, how = series.weights("dispersion")
    assert how == "1/SEM^2"
    assert np.allclose(w, 1.0)


# ---------------------------------------------------------------------------
# what a series is allowed to be
# ---------------------------------------------------------------------------

def test_a_two_level_comparison_is_not_a_graded_series():
    with pytest.raises(ValueError):
        synthetic([0.0, 1.0], [0.0, 1.0])


def test_levels_must_increase():
    with pytest.raises(ValueError):
        synthetic([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])


def test_a_log_drive_transform_cannot_hold_a_vehicle_level():
    with pytest.raises(ValueError):
        synthetic([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0],
                  drive_transform="log10", transform_reason="doses")


def test_a_non_default_drive_transform_needs_its_reason_recorded():
    with pytest.raises(ValueError):
        synthetic([1.0, 2.0, 4.0, 8.0], [0.0, 1.0, 2.0, 3.0],
                  drive_transform="log10")


def test_the_polarity_conversion_has_to_be_written_down():
    with pytest.raises(ValueError):
        synthetic(SIX, [0.0] * 6, score_sign=0)


def test_a_figure_is_not_a_source_this_module_can_read():
    with pytest.raises(ValueError):
        synthetic(SIX, [0.0] * 6, provenance="figure")


def test_score_sign_flips_the_series_into_the_endpoints_own_direction():
    up = synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], score_sign=1)
    down = synthetic(SIX, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], score_sign=-1)
    assert up.observed_shape() == "increasing"
    assert down.observed_shape() == "decreasing"


# ---------------------------------------------------------------------------
# the basis and the readout
# ---------------------------------------------------------------------------

def test_the_inverted_u_basis_peaks_at_its_peak_rate():
    term = decoder.NonlinearTerm("inverted_U", peak_rate=0.4)
    rates = np.linspace(0.0, 1.0, 201)
    values = np.array([term.basis(r) for r in rates])
    assert term.basis(0.4) == pytest.approx(1.0)
    assert values.max() == pytest.approx(1.0)
    assert rates[values.argmax()] == pytest.approx(0.4, abs=0.01)


def test_the_u_basis_troughs_at_its_peak_rate():
    term = decoder.NonlinearTerm("U", peak_rate=0.6)
    assert term.basis(0.6) == pytest.approx(0.0)
    assert term.basis(0.0) > 0.0
    assert term.basis(1.0) > 0.0


def test_a_basis_maps_the_unit_interval_into_the_unit_interval():
    for shape in ("U", "inverted_U"):
        for peak in (0.1, 0.5, 0.9):
            term = decoder.NonlinearTerm(shape, peak_rate=peak)
            values = [term.basis(r) for r in np.linspace(0.0, 1.0, 101)]
            assert min(values) >= -1e-12
            assert max(values) <= 1.0 + 1e-12


def test_a_peak_rate_is_illustrative_and_says_so():
    term = decoder.NonlinearTerm("U")
    assert "ILLUSTRATIVE" in term.provenance
    assert term.peak_rate == 0.5


def test_a_nonlinear_readout_with_no_nonlinear_terms_is_the_linear_one():
    net = _toy_network(["A", "B"])
    rates = np.array([0.3, 0.7])
    weights = {"A": 0.6, "B": -0.4}
    linear = behavior.BehaviorReadout("x", weights)
    same = decoder.NonlinearReadout("x", weights)
    assert same.score(net, rates) == pytest.approx(linear.score(net, rates))


def test_a_nonlinear_readout_rejects_a_term_on_a_node_with_no_weight():
    with pytest.raises(ValueError):
        decoder.NonlinearReadout(
            "x", {"A": 1.0}, {"B": decoder.NonlinearTerm("U")}
        )


def test_a_nonlinear_node_scores_through_its_basis():
    net = _toy_network(["A", "B"])
    term = decoder.NonlinearTerm("inverted_U", peak_rate=0.5)
    readout = decoder.NonlinearReadout("x", {"A": 1.0, "B": 1.0},
                                       {"B": term})
    rates = np.array([0.2, 0.5])
    assert readout.score(net, rates) == pytest.approx(0.2 + 1.0)


class _ToyNode(object):
    def __init__(self, name):
        self.name = name


class _ToyNetwork(object):
    def __init__(self, names):
        self.nodes = [_ToyNode(n) for n in names]
        self.index = {n: i for i, n in enumerate(names)}


def _toy_network(names):
    return _ToyNetwork(names)


# ---------------------------------------------------------------------------
# the record, and the wiring into P7.1
# ---------------------------------------------------------------------------

def test_every_series_names_a_real_endpoint():
    """Both libraries: production's is empty since P11.3, so the invariant
    is carried by the retired five."""
    for series in list(decoder.SERIES) + list(rodent_series.SERIES):
        assert series.endpoint in endpoints.ENDPOINTS, series.key


def test_every_series_key_is_unique():
    keys = [s.key for s in decoder.SERIES] + [s.key for s in rodent_series.SERIES]
    assert len(keys) == len(set(keys))


def test_every_series_citation_resolves():
    for series in list(decoder.SERIES) + list(rodent_series.SERIES):
        assert series.citation in decoder.CITATIONS, series.citation


def test_no_citation_is_an_orphan_in_either_direction():
    """Both directions, because P7.1's orphan test only checked one.

    Forward: every key in `CITATIONS` is used by a series or named in the
    prose - since P11.3 retired the five series, prose carries all fifteen,
    including the four whose series now live in `legacy/rodent_series`.
    Backward: every key listed in `CITED_IN_PROSE` really does appear
    in the module's prose, so the list cannot become a place to park a
    citation that no longer says anything.
    """
    used = {s.citation for s in decoder.SERIES}
    used |= set(decoder.CITED_IN_PROSE)
    assert not set(decoder.CITATIONS) - used, \
        sorted(set(decoder.CITATIONS) - used)

    prose = "\n".join(
        [decoder.__doc__ or "", decoder.P71_TARGETS]
        + list(decoder.SEARCHED_AND_EMPTY.values())
        + [decoder.CITATIONS[k] for k in decoder.CITATIONS]
    )
    for key in decoder.CITED_IN_PROSE:
        assert key in prose, key
    assert set(decoder.CITED_IN_PROSE) <= set(decoder.CITATIONS)


def test_every_bracketed_citation_in_the_prose_resolves():
    """A `[Name2000]` that resolves to nothing is a broken reference.

    P7.1 shipped five of these because its own check ran in one direction
    only; this is the two-directional version, and it reads the same prose
    the forward check does.
    """
    import re
    prose = "\n".join(
        [decoder.__doc__ or "", decoder.P71_TARGETS]
        + list(decoder.SEARCHED_AND_EMPTY.values())
    )
    known = set(decoder.CITATIONS) | set(endpoints.CITATIONS)
    for name in set(re.findall(r"\[([A-Z][A-Za-z]+\d{4})\]", prose)):
        assert name in known, name


def test_every_endpoint_has_a_verdict():
    verdicts = {v.endpoint_key for v in decoder.all_endpoint_verdicts()}
    assert verdicts == set(endpoints.ENDPOINTS)


def test_every_rejection_carries_one_of_the_recorded_reasons():
    known = set(decoder.REJECTION_REASONS)
    for v in decoder.assess_all():
        if not v.adopted:
            assert v.reason in known, (v.series.key, v.reason)


def test_the_adoption_list_is_derived_from_the_comparison_not_written_down():
    """No second copy of the answer.

    `adopted_nonlinearities()` must agree with re-running `assess_series`
    from scratch on every series - production's and, since P11.3, the
    quarantined five as well - so a transcription change in the data moves
    the adoption list rather than leaving it stale.
    """
    fresh = set()
    for series in list(decoder.SERIES) + list(rodent_series.SERIES):
        v = decoder.assess_series(series)
        if v.adopted:
            for acronym in series.structures:
                fresh.add((series.endpoint, acronym))
    assert set(decoder.adopted_nonlinearities()) == fresh


def test_a_contest_is_only_resolved_where_a_reason_is_written_down():
    adopted = decoder.adopted_nonlinearities()
    for (endpoint_key, acronym) in decoder.CONTEST_RESOLUTIONS:
        assert (endpoint_key, acronym) in adopted, (
            "a contest may only be declared resolved by a nonlinearity the "
            "held-out comparison actually adopted"
        )


def test_the_searched_and_empty_record_covers_every_endpoint_without_data():
    for verdict in decoder.all_endpoint_verdicts():
        if verdict.series_verdicts:
            continue
        assert verdict.endpoint_key in decoder.SEARCHED_AND_EMPTY, (
            "an endpoint with no graded series needs its search recorded"
        )


# ---------------------------------------------------------------------------
# the record since P11.3 emptied it
# ---------------------------------------------------------------------------

def test_the_production_library_is_empty():
    """P11.3's retirement, pinned at the site it happened.

    Every series this module ever held was rodent - four mouse optogenetic,
    one rat microinjection - so under the human-primary roadmap the library
    stays empty until a HUMAN graded series exists. The five retired objects,
    their published numbers and their preserved verdicts are tested in
    `tests/test_legacy_rodent_series.py`.
    """
    assert decoder.SERIES == ()
    assert len(rodent_series.SERIES) == 5


def test_nothing_was_adopted():
    """The result of P7.2, pinned - and unchanged by P11.3.

    Every series was rejected even when the library was full, each for its
    own reason, so retiring all of them changed no readout anywhere. If a
    future data addition changes this, this test is where it announces
    itself - which is the point: an adoption should be a deliberate, visible
    event, not a quiet consequence of adding a row.
    """
    assert decoder.adopted_nonlinearities() == {}
    for verdict in decoder.all_endpoint_verdicts():
        assert not verdict.adopted


def test_every_endpoint_now_verdicts_to_no_data():
    """P7.2 left four endpoints without a graded series; P11.3 retired the
    rodent series that covered the other three, so the empty set is now the
    WHOLE endpoint list until human data lands."""
    empty = {v.endpoint_key for v in decoder.all_endpoint_verdicts()
             if not v.series_verdicts}
    assert empty == set(endpoints.ENDPOINTS)
    assert len(empty) == 7
    for key in empty:
        assert decoder.endpoint_verdict(key).reason == \
            decoder.REJECTION_REASONS["no-data"]


def test_the_analgesia_target_p71_handed_over_is_recorded_as_unusable():
    """P7.1's flagship case, and its fate.

    [ZhuoGebhart1997] is the one within-paper, within-assay, within-site
    reversal in this literature, and its per-intensity values are in a
    figure. The record of that has to survive, because "P7.2 found nothing
    for analgesia" without it reads as not having looked.
    """
    assert decoder.endpoint_verdict("analgesia").series_verdicts == ()
    record = decoder.SEARCHED_AND_EMPTY["analgesia"]
    assert "ZhuoGebhart1997" in decoder.P71_TARGETS
    assert "MendoncaDosSantos2024" in record
    assert "Thomas & Hammond 1995" in record


def test_a_startle_magnitude_table_was_not_substituted_for_percent_ppi():
    """Rule 2 of the codebase, at the one place it bit in this task.

    [Alsene2011] is the only tabulated graded intracerebral series with SEM
    that exists for the gating literature - 4 doses in 9 regions - and its
    readout is startle magnitude, which is one of the two terms in the PPI
    ratio and not the ratio. Using it would be substituting a different
    measured effect for the one being modelled.
    """
    assert "Alsene2011" not in {s.citation for s in decoder.SERIES}
    assert "STARTLE MAGNITUDE" in \
        decoder.SEARCHED_AND_EMPTY["sensorimotor_gating"]


def test_resolve_nonlinear_gives_back_p71s_own_object_when_nothing_is_adopted():
    """No adoption anywhere, so every endpoint must resolve exactly as P7.1.

    Checked by `type(...) is` rather than by score equality, because a
    `NonlinearReadout` with an empty `nonlinear` dict would pass an equality
    check while quietly replacing the object P7.1 promised downstream.
    """
    names = ["CUN", "PPN", "VLPO", "BLA", "SNr", "VTA", "LC"]

    class _Node(object):
        def __init__(self, name):
            self.name = name

    class _Net(object):
        def __init__(self, ns):
            self.nodes = [_Node(n) for n in ns]
            self.index = {n: i for i, n in enumerate(ns)}

    ontology = connectome.load_fixture_ontology()
    for key in endpoints.ENDPOINT_ORDER:
        resolution = decoder.resolve_nonlinear(key, _Net(names), ontology)
        assert resolution.applied == {}
        assert type(resolution.readout) is behavior.BehaviorReadout, key
