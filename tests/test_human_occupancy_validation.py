from dataclasses import replace

import pytest

from neurocascade.human import occupancy_validation as ov
from neurocascade.human.d2_ec50 import (
    HALOPERIDOL_PLASMA_FREE_FRACTION,
    fit_plasma_ec50,
)
from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.heldout_validation import summarize_heldout
from neurocascade.human.occupancy_validation import (
    CLINICAL_OCCUPANCY_THRESHOLDS,
    CLINICAL_WINDOW_PROVENANCE,
    OCCUPANCY_IS_NOT_THE_OUTCOME,
    PRE_REGISTERED_SPLITS,
    ClinicalOccupancyThreshold,
    HeldOutFold,
    SplitRule,
    as_heldout_results,
    basis_observations,
    build_folds,
    chain_testable_observations,
    clinical_window_check,
    dose_reaching_occupancy,
    kapur1996_containment,
    run_all_splits,
    run_split,
    split_rule,
    steady_state_occupancy,
    summary,
)
from neurocascade.human.pet_observations import (
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    haloperidol_d2_observations,
)

SPLIT_NAMES = (
    "A-leave-one-out",
    "B-leave-one-study-out",
    "C-free-basis-set",
    "D-external-parameter",
)

# The aggregate errors these splits actually produced, recorded so that a change
# in any of them shows up as a failing test rather than as a quietly different
# validation. They are results, not targets: splits D and C are the two bad ones
# and are asserted at the size they came out.
RECORDED_MAE = {
    "A-leave-one-out": 0.0459,
    "B-leave-one-study-out": 0.0351,
    "C-free-basis-set": 0.0620,
    "D-external-parameter": 0.0939,
}


# ---------------------------------------------------------------------------
# the splits were declared before anything was fitted
# ---------------------------------------------------------------------------


def test_the_four_split_rules_are_declared_and_each_one_says_why_it_is_forced():
    assert tuple(rule.name for rule in PRE_REGISTERED_SPLITS) == SPLIT_NAMES
    for rule in PRE_REGISTERED_SPLITS:
        assert isinstance(rule, SplitRule)
        assert rule.question and rule.parameter_origin and rule.forced_by
        assert rule.plasma_basis in {"total", "free"}
        assert split_rule(rule.name) is rule
    # forced_by is the field that separates a pre-registered split from a split
    # chosen after seeing a result, so each rule has to name what fixed it.
    assert "exhaustive" in split_rule("A-leave-one-out").forced_by
    assert "exhaustive over studies" in split_rule("B-leave-one-study-out").forced_by
    assert split_rule("C-free-basis-set").forced_by.startswith("forced by P19.4")
    assert split_rule("D-external-parameter").forced_by.startswith("forced by P19.4")


def test_a_fifth_split_invented_after_the_fact_is_not_available():
    with pytest.raises(KeyError, match="would not be a held-out test"):
        split_rule("E-whatever-makes-this-look-better")
    with pytest.raises(KeyError, match="no pre-registered split named"):
        build_folds("E-whatever-makes-this-look-better")


def test_a_split_rule_cannot_claim_a_plasma_basis_that_does_not_exist():
    with pytest.raises(ValueError, match="plasma_basis must be total or free"):
        SplitRule(
            name="X",
            question="q",
            plasma_basis="bound",
            parameter_origin="p",
            forced_by="f",
            refits_in_repo=False,
        )


def test_the_folds_are_built_without_any_fit_running(monkeypatch):
    """Held-out points are identified before fitting, not during it.

    This is asserted by removing the ability to fit: with ``fit_plasma_ec50``
    replaced by something that raises, every fold of every split still builds,
    and every attempt to score a refitting split fails. So the partition cannot
    be a function of the fit.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("a fold was built by fitting something")

    monkeypatch.setattr(ov, "fit_plasma_ec50", refuse)
    for name in SPLIT_NAMES:
        folds = build_folds(name)
        assert folds
        for fold in folds:
            assert fold.test
    for name in ("A-leave-one-out", "B-leave-one-study-out", "C-free-basis-set"):
        with pytest.raises(AssertionError, match="built by fitting"):
            run_split(name)


def test_no_held_out_point_appears_in_its_own_folds_basis_set():
    for name in SPLIT_NAMES:
        for fold in build_folds(name):
            for held in fold.test:
                assert held not in fold.train
                assert all(held is not item for item in fold.train)


def test_a_fold_that_leaks_its_test_point_refuses_to_be_built():
    points = basis_observations("total")
    with pytest.raises(ValueError, match="in-sample error wearing a held-out label"):
        HeldOutFold(
            split_name="A-leave-one-out",
            label="leaky",
            train=points,
            test=(points[0],),
        )


def test_a_fold_with_nothing_to_test_is_refused():
    with pytest.raises(ValueError, match="tests nothing"):
        HeldOutFold(
            split_name="A-leave-one-out",
            label="empty",
            train=basis_observations("total"),
            test=(),
        )


def test_each_rule_partitions_the_data_the_way_it_says_it_does():
    points = basis_observations("total")
    assert len(points) == 4

    a_folds = build_folds("A-leave-one-out")
    assert len(a_folds) == len(points)
    held = [fold.test[0] for fold in a_folds]
    for point in points:
        assert sum(1 for item in held if item == point) == 1, "held out exactly once"
    assert all(len(fold.train) == 3 for fold in a_folds)

    b_folds = build_folds("B-leave-one-study-out")
    assert len(b_folds) == 2
    for fold in b_folds:
        assert len(fold.test) == 2 and len(fold.train) == 2
        # Patient-disjoint: the two points inside one depot study are the same
        # eight patients twice, which is exactly what split A leaks.
        test_sources = {item.source for item in fold.test}
        train_sources = {item.source for item in fold.train}
        assert len(test_sources) == 1
        assert not (test_sources & train_sources)

    (c_fold,) = build_folds("C-free-basis-set")
    assert c_fold.train == ()
    assert c_fold.test == basis_observations("free")
    assert "entered no fit" in c_fold.parameter_note

    d_folds = build_folds("D-external-parameter")
    assert len(d_folds) == len(HALOPERIDOL_D2_REPORTED_RELATIONS)
    for fold, relation in zip(d_folds, HALOPERIDOL_D2_REPORTED_RELATIONS):
        assert fold.train == ()
        assert fold.test == points
        assert relation.label in fold.label
        assert "nothing in this fold was fitted here" in fold.parameter_note


def test_the_refit_must_consume_exactly_the_points_the_fold_handed_it():
    """The leakage guard is arithmetic, not a promise, so it is exercised.

    A training point that is not a registered observation is silently dropped by
    ``fit_plasma_ec50``, which would mean the reported held-out error came from a
    different training set than the fold declared. The count check catches it.
    """
    points = basis_observations("total")
    fabricated = replace(
        points[0],
        source_locator="invented (week-9)",
        plasma_concentration_nM=3.0,
        observed_occupancy=0.6,
    )
    fold = HeldOutFold(
        split_name="A-leave-one-out",
        label="fabricated training point",
        train=(points[0], points[1], fabricated),
        test=(points[2],),
    )
    with pytest.raises(AssertionError, match="handed 3 training points and the refit consumed 2"):
        ov._run_refitting_fold(split_rule("A-leave-one-out"), fold)


def test_total_and_free_observations_are_never_pooled():
    total = basis_observations("total")
    free = basis_observations("free")
    assert {item.source.split()[0] for item in total} == {"Nyberg", "Regenthal"}
    assert {item.source.split()[0] for item in free} == {"Videbaek"}
    assert all(item.plasma_concentration_basis == "total" for item in total)
    assert all(item.plasma_concentration_basis == "free" for item in free)
    assert not set(total) & set(free)


# ---------------------------------------------------------------------------
# the error that comes out, per point and in aggregate
# ---------------------------------------------------------------------------


def test_every_split_reports_a_per_point_and_an_aggregate_error():
    for report in run_all_splits():
        assert report.n == len(report.predictions)
        for prediction in report.predictions:
            assert prediction.observation_label
            assert 0.0 < prediction.observed < 1.0
            assert 0.0 < prediction.predicted < 1.0
            assert prediction.error == pytest.approx(
                prediction.predicted - prediction.observed
            )
            assert prediction.absolute_error == abs(prediction.error)
            assert prediction.ec50_nM > 0.0
        assert report.mean_absolute_error == pytest.approx(
            sum(p.absolute_error for p in report.predictions) / report.n
        )
        assert report.bias == pytest.approx(
            sum(p.error for p in report.predictions) / report.n
        )
        assert report.max_absolute_error == max(
            p.absolute_error for p in report.predictions
        )
        assert report.rmse >= report.mean_absolute_error


def test_the_aggregate_errors_are_the_ones_that_actually_came_out():
    reports = {report.rule.name: report for report in run_all_splits()}
    assert set(reports) == set(SPLIT_NAMES)
    for name, mae in RECORDED_MAE.items():
        assert reports[name].mean_absolute_error == pytest.approx(mae, abs=5e-5)
    # The two bad splits are the two that use a parameter this repository did not
    # fit on the points being predicted, and they are worse than either in-repo
    # refit. That ordering is the finding, so it is asserted rather than noted.
    assert (
        reports["D-external-parameter"].mean_absolute_error
        > reports["C-free-basis-set"].mean_absolute_error
        > reports["A-leave-one-out"].mean_absolute_error
        > reports["B-leave-one-study-out"].mean_absolute_error
    )
    assert reports["D-external-parameter"].max_absolute_error == pytest.approx(
        0.1637, abs=5e-5
    )
    assert reports["C-free-basis-set"].max_absolute_error == pytest.approx(
        0.1409, abs=5e-5
    )


def test_split_d_over_predicts_every_single_point():
    """A signed bias equal to the mean absolute error is a one-sided miss."""
    report = run_split("D-external-parameter")
    assert report.n == 8
    assert all(p.error > 0.0 for p in report.predictions)
    assert report.bias == pytest.approx(report.mean_absolute_error)
    by_fold = dict(report.by_fold())
    assert len(by_fold) == 2
    kapur = by_fold["the depot points, against kapur1997's published EC50"]
    fitzgerald = by_fold["the depot points, against fitzgerald2000's published EC50"]
    assert kapur.mean_absolute_error == pytest.approx(0.0712, abs=5e-5)
    assert fitzgerald.mean_absolute_error == pytest.approx(0.1167, abs=5e-5)
    # The pooled figure averages over a choice of published constant rather than
    # over held-out points, so the per-fold rows are the ones that mean anything.
    assert kapur.mean_absolute_error < report.mean_absolute_error < fitzgerald.mean_absolute_error


def test_split_c_converts_the_total_basis_constant_and_does_not_refit_it():
    report = run_split("C-free-basis-set")
    assert report.rule.refits_in_repo is False
    expected = fit_plasma_ec50().ec50_nM * HALOPERIDOL_PLASMA_FREE_FRACTION
    assert {p.ec50_nM for p in report.predictions} == {expected}
    assert all(p.n_train == 0 for p in report.predictions)
    # The constant it uses is not the free-basis fit. Fitting the same four
    # points directly gives 0.2298 nM, a third larger than the converted 0.1696,
    # and that gap is why the split misses high on three of the four subjects.
    direct = fit_plasma_ec50(plasma_basis="free")
    assert direct.ec50_nM == pytest.approx(0.2298, abs=5e-5)
    assert direct.ec50_nM / expected == pytest.approx(1.355, abs=5e-4)
    assert report.bias > 0.0


def test_split_b_is_the_patient_disjoint_version_of_split_a():
    a = run_split("A-leave-one-out")
    b = run_split("B-leave-one-study-out")
    assert a.n == b.n == 4
    # Both are honest held-out errors; B is the one whose folds do not share
    # patients. It happens to be the smaller of the two, which is worth stating
    # because the leak in A would be expected to run the other way.
    assert b.mean_absolute_error < a.mean_absolute_error
    for report in (a, b):
        assert all(p.n_train >= 2 for p in report.predictions)
    assert {p.n_train for p in a.predictions} == {3}
    assert {p.n_train for p in b.predictions} == {2}


def test_standardised_error_is_reported_as_not_computable_rather_than_faked():
    """No source in the P19.3 set printed an SD, so nothing standardises."""
    for report in run_all_splits():
        assert report.standardized_errors_available is False
        assert "not computable, no source printed an SD" in report.summary()
        for prediction in report.predictions:
            assert prediction.standardized_error is None
            assert prediction.comparison.observation.observed_sd is None


def test_the_training_rmse_ratio_is_not_dressed_up_as_a_standardised_error():
    for report in run_all_splits():
        for prediction in report.predictions:
            ratio = prediction.error_in_training_rmse_units
            if report.rule.refits_in_repo:
                assert prediction.training_rmse > 0.0
                assert ratio == pytest.approx(prediction.error / prediction.training_rmse)
            else:
                # No in-repo fit means no residual to divide by, and the module
                # returns None rather than borrowing one from another split.
                assert prediction.training_rmse is None
                assert ratio is None
                assert "no in-repo training residual" in report.summary()
    worst = max(
        (p for p in run_split("A-leave-one-out").predictions),
        key=lambda p: abs(p.error_in_training_rmse_units),
    )
    assert abs(worst.error_in_training_rmse_units) == pytest.approx(2.13, abs=5e-3)


def test_containment_is_a_weaker_test_than_the_signed_error():
    """Every depot point is inside its own reported range, including the bad ones."""
    reports = {report.rule.name: report for report in run_all_splits()}
    for name in ("A-leave-one-out", "B-leave-one-study-out"):
        assert reports[name].n_with_range == 4
        assert reports[name].n_contained == 4
    # Split D is contained on all eight points and still has the worst error in
    # the task. The depot sources print ranges up to 0.77 wide, so containment
    # there discriminates almost nothing and the MAE is the number to read.
    assert reports["D-external-parameter"].n_with_range == 8
    assert reports["D-external-parameter"].n_contained == 8
    assert reports["D-external-parameter"].mean_absolute_error > 0.09
    # The Videbaek subjects print no range at all, so containment is unavailable
    # rather than passing.
    assert reports["C-free-basis-set"].n_with_range == 0
    assert reports["C-free-basis-set"].n_contained == 0
    assert all(p.contained is None for p in reports["C-free-basis-set"].predictions)


def test_the_split_summaries_print_the_errors_including_the_bad_ones():
    text = summary()
    assert "split D-external-parameter (n=8, total basis)" in text
    assert "MAE 0.0939" in text
    assert "split C-free-basis-set (n=4, free basis)" in text
    assert "MAE 0.0620" in text
    assert "MAE 0.0459" in text and "MAE 0.0351" in text
    # Per-point rows as well as aggregates.
    assert "Videbaek Table 2 (V)" in text
    assert "Nyberg abstract, Results (week-1)" in text


def test_the_splits_agree_with_the_p18_2_summariser():
    """A cross-check by code that knows nothing about this drug."""
    results = as_heldout_results()
    assert len(results) == 20
    assert {r.status for r in results} == {"scored"}
    assert {r.compound for r in results} == {"haloperidol"}
    assert {r.endpoint for r in results} == {"striatal D2 occupancy"}
    aggregate = summarize_heldout(results)
    assert aggregate["total"] == aggregate["scored"] == 20
    pooled = [p for report in run_all_splits() for p in report.predictions]
    assert aggregate["mean_absolute_error"] == pytest.approx(
        sum(p.absolute_error for p in pooled) / len(pooled)
    )
    assert aggregate["mean_absolute_error"] == pytest.approx(0.0662, abs=5e-5)
    for result, prediction in zip(results, pooled):
        assert result.note.startswith(prediction.split_name)
        assert result.error == pytest.approx(prediction.error)


# ---------------------------------------------------------------------------
# the arm of the chain nothing tests
# ---------------------------------------------------------------------------


def test_no_registered_observation_pairs_an_oral_dose_with_a_usable_occupancy():
    assert chain_testable_observations() == ()
    # Computed, not asserted in prose. Each dose-bearing observation is excluded
    # for a reason the record itself carries.
    dosed = [item for item in haloperidol_d2_observations() if item.dose is not None]
    assert dosed, "there are dose-bearing observations; they just are not usable"
    for item in dosed:
        interval_only = item.observed_occupancy is None
        not_an_occupancy = item.quantity != "occupancy"
        wrong_route = item.dose_regimen not in ov._CHAIN_COMPATIBLE_REGIMENS
        assert interval_only or not_an_occupancy or wrong_route
    assert {item.dose_regimen for item in dosed if item.observed_occupancy is None} == {
        "steady-state-daily"
    }
    assert {item.quantity for item in dosed if item.quantity != "occupancy"} == {
        "binding-index"
    }
    assert "intravenous-infusion" in {item.dose_regimen for item in dosed}


def test_the_untested_arm_is_stated_at_the_top_of_the_summary_not_buried():
    text = summary()
    headline = (
        "observations pairing an oral dose with a differenceable occupancy: 0"
    )
    assert headline in text
    assert "the dose-to-plasma arm of the chain is untested" in text
    assert text.index(headline) < text.index("split A-leave-one-out")


# ---------------------------------------------------------------------------
# the one whole-chain comparison, which is a miss
# ---------------------------------------------------------------------------


def test_the_kapur_single_dose_containment_is_recorded_as_a_miss():
    result = kapur1996_containment()
    assert result.dose_mg == 2.0
    assert result.dose_regimen == "steady-state-daily"
    assert (result.range_low, result.range_high) == (0.53, 0.74)
    assert result.single_dose_contained is False
    assert result.single_dose_peak_occupancy == pytest.approx(0.5253, abs=5e-5)
    assert result.single_dose_peak_occupancy < result.range_low
    assert result.distance_below_lower_bound == pytest.approx(0.00475, abs=5e-6)
    assert result.distance_below_lower_bound > 0.0


def test_the_steady_state_figure_is_a_diagnosis_of_that_miss_not_a_second_test():
    result = kapur1996_containment()
    assert result.steady_state_mean_contained is True
    assert result.steady_state_trough_occupancy == pytest.approx(0.4718, abs=5e-5)
    assert result.steady_state_mean_occupancy == pytest.approx(0.5635, abs=5e-5)
    assert result.steady_state_peak_occupancy == pytest.approx(0.6592, abs=5e-5)
    # The passing figure never claims to be blind, and the reason it is not is
    # carried on the result rather than left to a reader's memory of the docs.
    assert result.was_blind is False
    caveats = " ".join(result.caveats)
    assert "not blind" in caveats
    assert "docs/p19-5-dose-to-occupancy-chain.md" in caveats
    assert "computed after the single-dose miss was seen" in caveats
    assert "is not a second, passing test" in caveats
    assert "a range across seven patients, not a mean with a spread" in caveats


def test_the_miss_and_the_diagnosis_both_reach_the_summary():
    text = summary()
    assert "single-dose peak 0.5253, contained: False" in text
    assert "0.0047 below the lower bound" in text
    assert "mean contained: True" in text
    assert text.count("caveat:") == 3


# ---------------------------------------------------------------------------
# steady state, which is a construction on the chain and not a second model
# ---------------------------------------------------------------------------


def test_steady_state_superposes_the_chains_own_single_dose_curve():
    repeated = steady_state_occupancy(2.0)
    single = predict_occupancy(2.0)
    assert repeated.ec50_nM == single.ec50_nM
    assert repeated.times_min.shape == repeated.plasma_nM.shape == repeated.occupancy.shape
    assert repeated.times_min[0] == 0.0
    assert repeated.times_min[-1] < 1440.0
    assert repeated.trough_occupancy < repeated.mean_occupancy < repeated.peak_occupancy
    # Accumulation: every point of the steady-state interval sits above the
    # single dose's own trough, and the peak sits above the single-dose peak.
    assert repeated.peak_occupancy > single.peak_occupancy


def test_the_default_dose_count_is_past_convergence():
    forty = steady_state_occupancy(2.0, n_doses=40)
    seventy = steady_state_occupancy(2.0, n_doses=70)
    assert forty.n_doses == 40
    assert forty.mean_occupancy == pytest.approx(seventy.mean_occupancy, abs=1e-8)


def test_steady_state_refuses_a_dose_a_count_or_a_step_it_cannot_use():
    with pytest.raises(ValueError, match="dose_mg_per_day must be positive"):
        steady_state_occupancy(0.0)
    with pytest.raises(ValueError, match="n_doses must be at least 1"):
        steady_state_occupancy(2.0, n_doses=0)
    with pytest.raises(ValueError, match="too coarse to resolve a dosing interval"):
        steady_state_occupancy(2.0, step_min=1000.0)


# ---------------------------------------------------------------------------
# the clinical occupancy window
# ---------------------------------------------------------------------------


def test_the_clinical_window_check_returns_numbers_for_every_sourced_threshold():
    rows = clinical_window_check()
    assert len(rows) == len(CLINICAL_OCCUPANCY_THRESHOLDS) == 4
    for row, threshold in zip(rows, CLINICAL_OCCUPANCY_THRESHOLDS):
        assert row["outcome"] == threshold.outcome
        assert row["occupancy"] == threshold.occupancy
        assert row["as_printed"] == threshold.as_printed
        assert row["source"] == threshold.source
        assert 0.0 < row["steady_state_dose_mg_per_day"] < row["single_dose_mg"]
    # A higher occupancy always needs a higher dose, so ordering the rows by
    # their threshold orders them by dose too.
    by_occupancy = sorted(rows, key=lambda row: row["occupancy"])
    doses = [row["steady_state_dose_mg_per_day"] for row in by_occupancy]
    assert doses == sorted(doses)
    # The recorded figures, which are what the window question is answered with.
    assert rows[0]["steady_state_dose_mg_per_day"] == pytest.approx(2.89, abs=5e-3)
    assert rows[0]["single_dose_mg"] == pytest.approx(3.36, abs=5e-3)
    assert rows[1]["steady_state_dose_mg_per_day"] == pytest.approx(4.02, abs=5e-3)
    assert rows[1]["single_dose_mg"] == pytest.approx(4.65, abs=5e-3)
    assert rows[2]["steady_state_dose_mg_per_day"] == pytest.approx(5.56, abs=5e-3)
    assert rows[2]["single_dose_mg"] == pytest.approx(6.41, abs=5e-3)
    assert rows[3]["steady_state_dose_mg_per_day"] == pytest.approx(5.40, abs=5e-3)
    assert rows[3]["single_dose_mg"] == pytest.approx(6.23, abs=5e-3)


def test_the_window_lands_inside_clinically_used_daily_doses():
    """65 to 78 percent falls between about 2.9 and 5.6 mg/day on this chain."""
    rows = {row["occupancy"]: row for row in clinical_window_check()}
    low = rows[0.65]["steady_state_dose_mg_per_day"]
    high = rows[0.78]["steady_state_dose_mg_per_day"]
    assert 1.0 < low < high < 10.0
    assert high / low == pytest.approx(1.924, abs=5e-3)


def test_the_two_regimens_are_different_questions_with_different_answers():
    for target in (0.65, 0.72, 0.78):
        steady = dose_reaching_occupancy(target, regimen="steady-state-daily")
        single = dose_reaching_occupancy(target, regimen="single-dose")
        assert single > steady
        assert steady_state_occupancy(steady).mean_occupancy == pytest.approx(
            target, abs=1e-4
        )
    assert dose_reaching_occupancy(0.65) < dose_reaching_occupancy(0.78)


def test_dose_reaching_occupancy_refuses_what_the_chain_has_no_parameters_for():
    with pytest.raises(ValueError, match="strictly inside"):
        dose_reaching_occupancy(0.0)
    with pytest.raises(ValueError, match="strictly inside"):
        dose_reaching_occupancy(1.0)
    with pytest.raises(ValueError, match="no parameters for anything else"):
        dose_reaching_occupancy(0.65, regimen="depot-injection")


def test_the_65_to_80_window_is_not_claimed_as_a_sourced_fact():
    assert "is a synthesis, not a primary result" in CLINICAL_WINDOW_PROVENANCE
    assert "This module uses 78, not 80" in CLINICAL_WINDOW_PROVENANCE
    assert "appears to be a rounding of Kapur 2000's 78" in CLINICAL_WINDOW_PROVENANCE
    assert "65% to 85%" in CLINICAL_WINDOW_PROVENANCE  # Farde 1988, as printed
    assert "70% to 89%" in CLINICAL_WINDOW_PROVENANCE  # Farde 1992, as printed
    assert "60 to 78 percent" in CLINICAL_WINDOW_PROVENANCE  # Uchida 2011
    occupancies = {t.occupancy for t in CLINICAL_OCCUPANCY_THRESHOLDS}
    assert occupancies == {0.65, 0.72, 0.78, 0.775}
    assert 0.80 not in occupancies
    assert CLINICAL_WINDOW_PROVENANCE in summary()


def test_each_threshold_carries_what_a_reader_needs_to_discount_it():
    for threshold in CLINICAL_OCCUPANCY_THRESHOLDS:
        assert threshold.region == "striatum"
        assert threshold.region_origin and threshold.statistical_basis
        assert "PMID" in threshold.source
    kapur = [t for t in CLINICAL_OCCUPANCY_THRESHOLDS if "10739409" in t.source]
    assert len(kapur) == 3
    for threshold in kapur:
        assert threshold.tracer == "[11C]raclopride"
        # The region is not the authors' printed word, and the record says so.
        assert "inferred" in threshold.region_origin
        assert "significantly" in threshold.statistical_basis
    (uchida,) = [t for t in CLINICAL_OCCUPANCY_THRESHOLDS if "21694629" in t.source]
    assert uchida.tracer is None
    assert "12 studies and 82 subjects" in uchida.statistical_basis
    assert "paywalled" in uchida.statistical_basis


def test_a_threshold_must_be_a_fraction_strictly_inside_zero_and_one():
    for bad in (0.0, 1.0, 78.0):
        with pytest.raises(ValueError, match="strictly inside"):
            ClinicalOccupancyThreshold(
                outcome="o",
                occupancy=bad,
                as_printed="p",
                source="s",
                source_locator="l",
                tracer=None,
                region="striatum",
                region_origin="ro",
                statistical_basis="sb",
            )


def test_reaching_an_occupancy_is_not_predicting_the_outcome_attached_to_it():
    assert "Siafis 2023" in OCCUPANCY_IS_NOT_THE_OUTCOME
    assert "does not determine the side effect" in OCCUPANCY_IS_NOT_THE_OUTCOME
    assert "no effect compartment and no ke0" in OCCUPANCY_IS_NOT_THE_OUTCOME
    assert (
        "It does not predict response, hyperprolactinemia or extrapyramidal side "
        "effects" in OCCUPANCY_IS_NOT_THE_OUTCOME
    )
    assert OCCUPANCY_IS_NOT_THE_OUTCOME in summary()
