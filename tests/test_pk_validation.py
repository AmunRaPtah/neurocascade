"""P1.6: tests for the concentration-time curve validation.

Two things these tests are shaped to do that ordinary tolerance tests do not:

1. **They pin the error, not the threshold.** Every independent comparison
   asserts its own relative error to five significant figures. If someone
   later widens a criterion to turn a FAIL into a PASS, these tests still
   pass and the widening is visible only in the verdict assertions - which is
   why `test_the_failing_comparisons_are_exactly_these` asserts the failing
   SET by key. To make a failure disappear you have to change a number that
   is written down here, not a tolerance that is not.

2. **They assert that the round-trip tier cannot be mistaken for
   validation.** `test_round_trip_tier_is_graded_only_against_the_model_
   itself` checks that no round-trip comparison quotes an external source.
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade import drug_pk
from neurocascade import pk_validation as pv
from neurocascade.pk import LN2, DosedOneCompartmentPK


ALL = pv.all_results()
BY_KEY = {r.key: r for r in ALL}


def err(key):
    return BY_KEY[key].relative_error


# ---------------------------------------------------------------------------
# The machinery: does the simulated curve and its NCA actually work
# ---------------------------------------------------------------------------


def test_nca_recovers_the_parameters_of_a_known_monoexponential():
    """A synthetic C(t) = C0 exp(-k t) with no model attached: Cmax, Tmax,
    half-life and AUC must all come back to the analytic values, so that a
    later failure can be attributed to a drug rather than to this code."""
    c0, k = 250.0, 0.03
    times = np.linspace(0.0, 12.0 * LN2 / k, 40001)
    curve = c0 * np.exp(-k * times)
    m = pv.CurveMetrics(times, curve)

    assert m.cmax == pytest.approx(c0, rel=1e-12)
    assert m.tmax_min == 0.0
    assert m.terminal_half_life_min == pytest.approx(LN2 / k, rel=1e-9)
    assert m.auc_0_inf == pytest.approx(c0 / k, rel=1e-6)
    assert m.terminal_fit_r_squared > 1.0 - 1e-12
    # Truncated AUC against the closed form, at a boundary that is NOT a grid
    # point, so the interpolation path is exercised.
    for t_upper in (15.0, 90.0, 137.5):
        assert m.auc_to(t_upper) == pytest.approx(
            c0 / k * (1.0 - math.exp(-k * t_upper)), rel=1e-6
        )


def test_nca_recovers_the_parameters_of_a_known_bateman_curve():
    """Same, for first-order absorption: this is the only curve shape in the
    project with a non-zero Tmax, so it is the only one where the Tmax branch
    of `CurveMetrics` does anything."""
    model = DosedOneCompartmentPK(
        route="IP",
        Vd_L_per_kg=10.0,
        CL_mL_per_min_per_kg=100.0,
        body_weight_kg=0.25,
        ka_per_min=0.2,
        F=0.5,
    )
    m = pv.metrics_for(model, 5.0)
    assert m.tmax_min == pytest.approx(model.tmax_min(), rel=1e-3)
    assert m.terminal_half_life_min == pytest.approx(
        model.terminal_half_life_min, rel=1e-6
    )
    assert m.auc_0_inf == pytest.approx(model.auc_0_inf(5.0), rel=1e-6)
    # The peak of a Bateman curve is strictly below the dose-equivalent
    # concentration, always.
    assert m.cmax < model.dose_equivalent_concentration(5.0)


def test_truncated_auc_refuses_to_extrapolate_past_the_simulated_window():
    times = np.linspace(0.0, 10.0, 101)
    curve = np.exp(-times)
    with pytest.raises(ValueError):
        pv.linear_trapezoid_auc(times, curve, t_upper=20.0)


def test_simulated_curve_matches_the_one_the_coupled_engine_produces():
    """The metrics must describe the curve `engine.Simulation` actually
    integrates, not a separate standalone calculation that happens to use the
    same constants."""
    from neurocascade import CircuitNetwork, CircuitNode, Ligand, ReceptorSite
    from neurocascade import Simulation

    model = drug_pk.diazepam_rat_ip_pk(
        0.25, molar_mass_g_per_mol=None, brain_partition=None
    )
    site = ReceptorSite(
        name="site",
        ligands=[Ligand(name="diazepam", kon=1e-3, koff=1e-2, efficacy=1.0)],
    )
    network = CircuitNetwork([CircuitNode("A", tau=10.0, baseline_input=0.5)], [[0.0]])
    sim = Simulation(
        pk_models=[model],
        receptor_site=site,
        network=network,
        coupling={"A": 1.0},
    )
    t_eval = np.linspace(0.0, 300.0, 3001)
    result = sim.run(doses=[5.0], t_span=(0.0, 300.0), t_eval=t_eval)

    times, standalone = pv.simulate_curve(model, 5.0, t_end_min=300.0, dt_min=0.1)
    assert np.allclose(
        result["concentrations"][0], np.interp(t_eval, times, standalone), rtol=1e-6
    )


def test_metrics_do_not_depend_on_the_body_weight_passed():
    """Every parameter set here is per-kg, so a validation run against a
    study's mean weight and one against a different weight must give the same
    concentrations, and therefore the same verdicts."""
    light = drug_pk.diazepam_rat_iv_pk(
        0.2, molar_mass_g_per_mol=None, brain_partition=None
    )
    heavy = drug_pk.diazepam_rat_iv_pk(
        0.4, molar_mass_g_per_mol=None, brain_partition=None
    )
    a, b = pv.metrics_for(light, 1.0), pv.metrics_for(heavy, 1.0)
    assert a.cmax == pytest.approx(b.cmax, rel=1e-12)
    assert a.auc_0_inf == pytest.approx(b.auc_0_inf, rel=1e-12)
    assert a.terminal_half_life_min == pytest.approx(b.terminal_half_life_min, rel=1e-12)
    # ...and the absolute quantities do track it, so nothing is being ignored.
    assert heavy.dose_mg(1.0) == pytest.approx(2.0 * light.dose_mg(1.0))


# ---------------------------------------------------------------------------
# Round-trip tier: must pass, and must be honest about what it is
# ---------------------------------------------------------------------------


def test_round_trip_covers_every_model_drug_pk_can_build():
    routes = {(r.drug, r.route) for r in pv.round_trip_results()}
    assert routes == {
        ("haloperidol", "IV"),
        ("fentanyl", "IV"),
        ("diazepam", "IV"),
        ("diazepam", "IP"),
    }
    # All five builders, including both one-compartment collapses.
    prefixes = {r.key.split(":")[0] for r in pv.round_trip_results()}
    assert prefixes == {
        "haloperidol/rat/IV/2C",
        "haloperidol/rat/IV/1C",
        "fentanyl/rat/IV/2C",
        "fentanyl/rat/IV/1C",
        "diazepam/rat/IV/1C",
        "diazepam/rat/IP",
    }


def test_round_trip_errors_are_numerical_not_structural():
    """Every round-trip error must be small enough that it is obviously the
    integrator and the trapezoid rule, not a modelling difference. The one
    comparison above 1e-5 is the Bateman Tmax, whose resolution is the grid
    spacing (0.011 min on a 17.1 min peak = 6.4e-4 of it), and it is
    identified by name rather than covered by a blanket tolerance."""
    for r in pv.round_trip_results():
        assert r.passed, r
        limit = 5e-4 if r.key.endswith(":tmax") else 1e-5
        assert abs(r.relative_error) < limit, r


def test_round_trip_tier_is_graded_only_against_the_model_itself():
    """The tier exists to be discounted, so it must not be possible to read a
    literature citation off one of its rows."""
    for r in pv.round_trip_results():
        assert r.source == "model's own closed-form value"
        assert r.criterion_kind == pv.NUMERICAL


def test_the_only_tmax_in_the_round_trip_tier_is_the_one_absorption_model():
    tmax_rows = [r for r in pv.round_trip_results() if r.quantity == "Tmax"]
    assert [r.key for r in tmax_rows] == ["diazepam/rat/IP:tmax"]
    assert err("diazepam/rat/IP:tmax") == pytest.approx(2.2799e-4, rel=1e-3)


# ---------------------------------------------------------------------------
# Source-consistency tier
# ---------------------------------------------------------------------------


def test_haloperidol_simulated_half_life_sits_inside_the_sources_rounding():
    r = BY_KEY["haloperidol/rat/IV/2C vs ChengPaalzow1992:printed_t_half"]
    # 92.698 min against a printed "1.5 h": +3.00%, and inside the 87-93 min
    # that two significant figures leave open.
    assert r.relative_error == pytest.approx(0.029978, rel=1e-4)
    assert r.interval == (87.0, 93.0)
    assert r.passed


def test_diazepam_ip_half_life_misses_its_sources_printed_value_and_why():
    """The one source-consistency failure, and it is the source's arithmetic
    rather than the model's: `drug_pk` already pins this same -0.64%."""
    r = BY_KEY["diazepam/rat/IP vs Friedman1986:printed_t_half"]
    assert r.relative_error == pytest.approx(-0.0064067, rel=1e-4)
    assert not r.passed
    implied, reported, rel = drug_pk.diazepam_rat_ip_reported_half_life_consistency()
    assert r.predicted == pytest.approx(implied, rel=1e-6)
    assert r.relative_error == pytest.approx(rel, rel=1e-6)
    # The interval the model's own printed INPUTS leave open excludes the
    # printed 52.8 min, which is where the inconsistency actually lives.
    lo = LN2 / (255.5 / (19.25 * 1000.0))
    hi = LN2 / (254.5 / (19.35 * 1000.0))
    assert lo < r.predicted < hi
    assert reported > hi


def test_fentanyl_bootstrap_and_point_estimates_give_the_same_terminal_phase():
    r = BY_KEY["fentanyl/rat/IV vs Yassen2008:bootstrap_t_half"]
    assert r.relative_error == pytest.approx(0.021428, rel=1e-4)
    lo, hi = drug_pk.fentanyl_rat_iv_terminal_half_life_interval_min()
    assert lo < r.predicted < hi
    assert r.passed


# ---------------------------------------------------------------------------
# Independent tier: haloperidol vs [Lei2018]
# ---------------------------------------------------------------------------


def test_haloperidol_curve_against_lei_2018_misses_auc_and_clearance():
    auc = BY_KEY["haloperidol/rat/IV vs Lei2018:auc"]
    cl = BY_KEY["haloperidol/rat/IV vs Lei2018:clearance"]

    assert auc.predicted == pytest.approx(140.56, rel=1e-4)
    assert auc.relative_error == pytest.approx(-0.253916, rel=1e-5)
    assert auc.z_score == pytest.approx(-2.3336, rel=1e-4)
    assert not auc.passed
    # It does clear the looser two-fold criterion, and saying so is part of
    # reporting the size of the miss rather than only its verdict.
    assert auc.meets_two_fold_criterion

    assert cl.relative_error == pytest.approx(0.310526, rel=1e-5)
    assert cl.z_score == pytest.approx(2.95, rel=1e-4)
    assert not cl.passed

    # AUC and CL are one finding measured twice: Dose/AUC must reproduce the
    # clearance comparison up to that paper's own 2.2% internal rounding.
    implied_cl = pv.LEI2018_IV_DOSE_MG_PER_KG / (
        pv.LEI2018_IV_AUC_NG_H_PER_ML / 1000.0
    )
    assert implied_cl == pytest.approx(3.715, rel=1e-3)
    assert abs(implied_cl / pv.LEI2018_IV_CL_L_PER_H_PER_KG - 1.0) < 0.023


def test_haloperidol_half_life_passes_only_because_lei_2018_is_imprecise():
    """A pass that would be misreported as agreement: 92.7 min against
    174 +/- 42 min is a 1.88-fold under-prediction that clears the dispersion
    criterion at z = -1.94 purely because the observed SD is 24% of the
    observed mean."""
    r = BY_KEY["haloperidol/rat/IV vs Lei2018:t_half"]
    assert r.relative_error == pytest.approx(-0.467253, rel=1e-5)
    assert r.z_score == pytest.approx(-1.9358, rel=1e-4)
    assert r.passed
    assert 1.0 / r.fold_error == pytest.approx(1.877, rel=1e-3)
    # It scrapes inside the two-fold criterion too (0.53x), which is the
    # other reason the verdict alone is not the finding here.
    assert r.meets_two_fold_criterion
    assert r.fold_error == pytest.approx(0.5328, rel=1e-3)


def test_the_haloperidol_matrix_gap_is_recorded_not_used_to_explain_away():
    """An Rb of 0.76 would reconcile the two clearances exactly, and no rat
    blood:plasma ratio for haloperidol exists, so the reconciliation is not
    applied. Both halves of that have to stay on the record."""
    cl = BY_KEY["haloperidol/rat/IV vs Lei2018:clearance"]
    assert "blood:plasma" in cl.note
    assert "0.76" in cl.note
    implied_rb = pv.LEI2018_IV_CL_L_PER_H_PER_KG / cl.predicted
    assert implied_rb == pytest.approx(0.763, rel=1e-2)
    assert ("haloperidol", "rat", "blood:plasma") in drug_pk.SEARCHED_AND_EMPTY


# ---------------------------------------------------------------------------
# Independent tier: diazepam vs [Rautiola2019]
# ---------------------------------------------------------------------------


def test_diazepam_iv_peak_is_3_4_fold_low_and_that_sizes_a_known_unknown():
    """`diazepam_rat_iv_pk` says its post-bolus peak is understated by an
    unknown factor, bounded below by 1 and not computable because [Klotz1976]
    publishes no central volume. [Rautiola2019]'s extrapolated C0 finally
    sizes it."""
    r = BY_KEY["diazepam/rat/IV vs Rautiola2019:c0"]
    assert r.relative_error == pytest.approx(-0.708155, rel=1e-5)
    assert not r.passed
    assert not r.meets_two_fold_criterion
    assert 1.0 / r.fold_error == pytest.approx(3.426, rel=1e-3)

    # Dose/C0 is an empirical central volume, against the model's V_beta.
    empirical_vc = pv.RAUTIOLA2019_IV_DOSE_MG_PER_KG / (
        pv.RAUTIOLA2019_IV_C0_NG_PER_ML / 1000.0
    )
    assert empirical_vc == pytest.approx(2.268, rel=1e-3)
    assert empirical_vc < drug_pk.DIAZEPAM_RAT_IV_VBETA_L_PER_KG
    # ...and it stays an observation: no diazepam Vc constant was introduced.
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_IV_VC_L_PER_KG")


def test_diazepam_early_auc_passes_and_late_auc_fails_for_opposite_reasons():
    early = BY_KEY["diazepam/rat/IV vs Rautiola2019:auc_0_15"]
    late = BY_KEY["diazepam/rat/IV vs Rautiola2019:auc_0_90"]

    assert early.relative_error == pytest.approx(-0.0648304, rel=1e-5)
    assert early.passed
    assert late.relative_error == pytest.approx(0.379858, rel=1e-5)
    assert not late.passed
    assert late.z_score == pytest.approx(10.856, rel=1e-4)
    # The sign flips between the two windows: too low early (no distribution
    # phase), too high late (half-life twice as long as observed).
    assert early.relative_error < 0.0 < late.relative_error


def test_diazepam_terminal_half_life_is_twice_the_independent_observation():
    r = BY_KEY["diazepam/rat/IV vs Rautiola2019:t_half"]
    assert r.relative_error == pytest.approx(1.006079, rel=1e-5)
    assert r.fold_error == pytest.approx(2.0061, rel=1e-4)
    # Fails the two-fold criterion by 0.3%, which is close enough that the
    # verdict alone would be misleading; the fold error is what is reported.
    assert not r.passed
    assert not r.meets_two_fold_criterion
    # The observed value comes from a 90-minute window, under three of its own
    # half-lives, which is the standing caveat on it.
    assert "90 min" in r.note


def test_both_diazepam_auc_comparisons_use_the_sources_own_windows():
    """Comparing an AUC(0-inf) against a study that stopped sampling at 90
    minutes would manufacture a discrepancy out of the study design, so the
    model is integrated over 0-15 and 0-90 min instead."""
    for key in (
        "diazepam/rat/IV vs Rautiola2019:auc_0_15",
        "diazepam/rat/IV vs Rautiola2019:auc_0_90",
    ):
        assert "0-" in BY_KEY[key].quantity
    model = drug_pk.diazepam_rat_iv_pk(
        pv.RAUTIOLA2019_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    auc_inf = model.auc_0_inf(pv.RAUTIOLA2019_IV_DOSE_MG_PER_KG) * 1000.0
    # AUC(0-inf) is 2.3x the observed 90-minute AUC; using it would have
    # reported a 126% error where the honest one is 38%.
    assert auc_inf / pv.RAUTIOLA2019_IV_AUC_0_90_NG_MIN_PER_ML == pytest.approx(
        2.257, rel=1e-3
    )


# ---------------------------------------------------------------------------
# Independent tier: fentanyl, cross-route only
# ---------------------------------------------------------------------------


def test_fentanyl_iv_clearance_implies_an_impossible_subcutaneous_f():
    r = BY_KEY["fentanyl/rat/IV vs CanfieldSprague2025:implied_F"]
    assert r.criterion_kind == pv.UPPER_BOUND
    assert r.predicted == pytest.approx(1.5301, rel=1e-4)
    assert not r.passed
    # An implied F above 1 is not a near miss: it says the two clearances
    # cannot both describe rat fentanyl.
    assert r.predicted > 1.0


def test_fentanyl_intraperitoneal_check_is_a_lower_bound_on_the_same_failure():
    r = BY_KEY["fentanyl/rat/IV vs Almousa2011:implied_F"]
    assert r.predicted == pytest.approx(1.13456, rel=1e-4)
    assert not r.passed
    assert "lower bound" in r.note.lower()
    # The source it leans on is independently known to contradict itself.
    implied, reported, ratio = drug_pk.fentanyl_rat_ip_auc_consistency()
    assert ratio > 1.9


def test_only_fentanyl_fails_the_structural_bioavailability_test():
    """The same CL_iv/(CL/F) <= 1 test applied to all three drugs. Diazepam
    and haloperidol pass it from cross-study data assembled exactly the same
    way, which is what makes fentanyl's failure evidence about fentanyl."""
    structural = [r for r in ALL if r.criterion_kind == pv.UPPER_BOUND]
    assert {r.drug for r in structural} == {"fentanyl", "diazepam", "haloperidol"}
    failing = {r.drug for r in structural if not r.passed}
    assert failing == {"fentanyl"}
    assert BY_KEY["diazepam/rat: implied F(IP) <= 1"].predicted == pytest.approx(
        0.32, rel=1e-6
    )
    assert BY_KEY["diazepam/rat: implied F(PO) <= 1"].predicted == pytest.approx(
        0.08190, rel=1e-4
    )
    assert BY_KEY["haloperidol/rat: implied F(PO) <= 1"].predicted == pytest.approx(
        0.29123, rel=1e-4
    )


def test_fentanyl_terminal_half_life_misses_both_extravascular_observations():
    sc = BY_KEY["fentanyl/rat/IV vs CanfieldSprague2025:t_half"]
    ip = BY_KEY["fentanyl/rat/IV vs Almousa2011:t_half"]
    assert sc.relative_error == pytest.approx(-0.187948, rel=1e-5)
    assert ip.relative_error == pytest.approx(-0.729726, rel=1e-5)
    assert not sc.passed and not ip.passed
    # The SC miss clears the two-fold criterion; the IP one does not.
    assert sc.meets_two_fold_criterion
    assert not ip.meets_two_fold_criterion


def test_fentanyl_auc_upper_bound_still_lands_below_the_observed_mean():
    """The IV model has no SC route, so the most exposure it can predict for
    0.3 mg/kg SC is the F = 1 value. That upper bound is 38% BELOW what was
    measured, which is the implied-F-above-1 finding restated."""
    r = BY_KEY["fentanyl/rat/IV vs CanfieldSprague2025:auc"]
    assert r.relative_error == pytest.approx(-0.380901, rel=1e-5)
    # It passes both criteria, because that study's between-animal spread is
    # wide - which is exactly why the structural row is kept alongside it.
    assert r.passed
    assert r.meets_two_fold_criterion
    assert not BY_KEY["fentanyl/rat/IV vs CanfieldSprague2025:implied_F"].passed


def test_fentanyl_has_no_independent_same_route_validation_at_all():
    for r in pv.independent_results():
        if r.drug == "fentanyl":
            assert " vs " in r.route  # every one of them is cross-route
    record = pv.SEARCHED_AND_EMPTY_VALIDATION[("fentanyl", "rat", "IV")]
    for token in ("Ohtsuka", "17598696", "Toxicol Sci", "PMC12575352", "Searched"):
        assert token in record


# ---------------------------------------------------------------------------
# The overall verdict set, and the honesty invariants
# ---------------------------------------------------------------------------


def test_the_failing_comparisons_are_exactly_these():
    """The regression that makes the criteria non-negotiable: to move a
    comparison out of this set you have to change the model or the target,
    and either way this test says so."""
    assert {r.key for r in pv.failing_results()} == {
        "diazepam/rat/IP vs Friedman1986:printed_t_half",
        "haloperidol/rat/IV vs Lei2018:auc",
        "haloperidol/rat/IV vs Lei2018:clearance",
        "diazepam/rat/IV vs Rautiola2019:c0",
        "diazepam/rat/IV vs Rautiola2019:auc_0_90",
        "diazepam/rat/IV vs Rautiola2019:t_half",
        "fentanyl/rat/IV vs CanfieldSprague2025:implied_F",
        "fentanyl/rat/IV vs Almousa2011:implied_F",
        "fentanyl/rat/IV vs CanfieldSprague2025:t_half",
        "fentanyl/rat/IV vs Almousa2011:t_half",
    }
    independent = pv.independent_results()
    assert len(independent) == 15
    assert sum(1 for r in independent if not r.passed) == 9


def test_no_independent_comparison_is_graded_against_a_model_input():
    """The whole point of the tier. [ChengPaalzow1992], [Yassen2008],
    [Klotz1976], [Friedman1986], [Hironaka1984] and [Hua2025] parameterise
    models here, so none of them may appear alone as an independent target."""
    parameterising = {
        "[ChengPaalzow1992]",
        "[Yassen2008]",
        "[Klotz1976]",
        "[Friedman1986]",
        "[Hironaka1984]",
    }
    for r in pv.independent_results():
        if r.criterion_kind == pv.UPPER_BOUND:
            # Structural rows deliberately pair a model source with an
            # independent one; the claim they make is about the pair.
            assert "/" in r.source
            continue
        assert r.source not in parameterising, r


def test_every_result_carries_its_provenance():
    for r in ALL:
        assert r.tier in (pv.ROUND_TRIP, pv.SOURCE_CONSISTENCY, pv.INDEPENDENT)
        assert r.source
        assert r.units
        assert r.criterion
        assert isinstance(r.passed, bool)
        assert math.isfinite(r.relative_error)


def test_dispersion_understatement_is_flagged_where_it_happens():
    """[Rautiola2019]'s sparse-sampling SEs cannot be converted to SDs, so
    they are used unconverted - a stricter test than the data justify, and
    the record has to say so rather than quietly benefit from it."""
    understated = [r.key for r in ALL if r.dispersion_is_understated]
    assert understated == [
        "diazepam/rat/IV vs Rautiola2019:auc_0_15",
        "diazepam/rat/IV vs Rautiola2019:auc_0_90",
    ]
    # Where n IS known, the SEM is inflated to an SD before use.
    sc = BY_KEY["fentanyl/rat/IV vs CanfieldSprague2025:auc"]
    assert sc.dispersion_kind == "SEM"
    assert sc.observed_n == 6
    assert sc.sd_equivalent == pytest.approx(
        pv.CANFIELD2025_SC_AUC_SEM_NG_H_PER_ML * math.sqrt(6), rel=1e-9
    )


def test_tmax_is_recorded_as_unvalidatable_with_the_reason():
    record = pv.TMAX_IS_NOT_VALIDATABLE
    for token in (
        "17.1 min",
        "Friedman1986",
        "PMC11296087",
        "Almousa2011",
        "Hua2025",
        "Searched",
    ):
        assert token in record
    # And it is true: no comparison in any tier validates a Tmax against an
    # external observation.
    external_tmax = [
        r for r in ALL if r.quantity == "Tmax" and r.tier != pv.ROUND_TRIP
    ]
    assert external_tmax == []


def test_searched_and_empty_validation_records_name_candidates_and_why():
    assert set(pv.SEARCHED_AND_EMPTY_VALIDATION) == {
        ("fentanyl", "rat", "IV"),
        ("haloperidol", "rat", "Cmax"),
        ("*", "mouse", "*"),
    }
    for key, record in pv.SEARCHED_AND_EMPTY_VALIDATION.items():
        assert len(record) > 200, key


def test_validation_report_shows_every_comparison_and_its_verdict():
    text = pv.validation_report()
    for r in ALL:
        assert r.key in text
    assert text.count("FAIL") == len(pv.failing_results())
    for tier in (pv.ROUND_TRIP, pv.SOURCE_CONSISTENCY, pv.INDEPENDENT):
        assert tier in text
