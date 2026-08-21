"""Tests for the diazepam rat PK parameter set (P1.4).

Its own file, for the same reason `test_fentanyl_pk.py` is: diazepam sits in
a third distinct evidential position, and the tests have to be shaped by it.

* Haloperidol (P1.3): one source, half-life reported, Q derived.
* Fentanyl (P1.2): one source, full parameter set reported, half-life derived.
* Diazepam (P1.4): **no single source covers a route**. The intravenous
  disposition is one paper's, the intraperitoneal disposition is a second
  paper's, the intraperitoneal absorption rate is a third paper's, and the
  oral data are a fourth's. So the tests here have to check something the
  other two files never needed: that the *composite* is internally coherent,
  that its cross-study seams are visible, and that the numbers that were
  derived across papers (both bioavailabilities) are still derived and not
  quietly promoted to measurements.

And one thing no other drug in this codebase has: an active metabolite that
demonstrably matters on the simulated timescale. `test_active_metabolite_*`
below pins that record, because "we forgot about nordiazepam" and "we
recorded that nordiazepam matters and chose not to model it" are different
states and only the second one is acceptable.

Four kinds of assertion, kept apart:

1. **Transcription.** Constants equal what the cited papers print.
2. **Behaviour.** Curves integrated through the same ODE protocol
   `engine.Simulation` uses reproduce the sources' half-lives, and the
   apparent parameterisation's AUC identity holds.
3. **Refusal.** Routes and species with no usable parameter set raise, and
   carry the search record; no bioavailability or absorption constant exists
   for any route that did not measure one.
4. **Honesty about the seams.** The real, non-zero disagreements between
   sources are asserted at their real values, not hidden inside tolerances.
"""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade.legacy import drug_pk
from neurocascade.legacy.drug_pk import (
    DIAZEPAM_ACTIVE_METABOLITE_LIMITATION,
    DIAZEPAM_ALPHA_HALF_LIFE_RANGE_ALL_SPECIES_H,
    DIAZEPAM_MOLAR_MASS_G_PER_MOL,
    DIAZEPAM_RAT_IP_BRAIN_HALF_LIFE_MIN,
    DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO,
    DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG,
    DIAZEPAM_RAT_IP_FREE_FRACTION,
    DIAZEPAM_RAT_IP_KA_PER_MIN,
    DIAZEPAM_RAT_IP_KA_SOURCE_AGE_DAYS,
    DIAZEPAM_RAT_IP_KA_SOURCE_DOSE_MG_PER_KG,
    DIAZEPAM_RAT_IP_KA_SOURCE_K12_PER_MIN,
    DIAZEPAM_RAT_IP_KA_SOURCE_K21_PER_MIN,
    DIAZEPAM_RAT_IP_KA_SOURCE_KEL_PER_MIN,
    DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG,
    DIAZEPAM_RAT_IP_SOURCE_N_PER_TIMEPOINT,
    DIAZEPAM_RAT_IP_SOURCE_SAMPLING_TIMES_MIN,
    DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN,
    DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG,
    DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG,
    DIAZEPAM_RAT_IV_F,
    DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN,
    DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_MIN,
    DIAZEPAM_RAT_IV_VBETA_L_PER_KG,
    DIAZEPAM_RAT_PLASMA_BOUND_FRACTION,
    DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L,
    DIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L,
    DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG,
    DIAZEPAM_RAT_PO_CMAX_SD_UG_PER_L,
    DIAZEPAM_RAT_PO_CMAX_UG_PER_L,
    DIAZEPAM_RAT_PO_MRT_0_INF_H,
    DIAZEPAM_RAT_PO_SOURCE_BODY_WEIGHT_KG,
    DIAZEPAM_RAT_PO_SOURCE_DOSE_MG_PER_KG,
    DIAZEPAM_RAT_PO_SOURCE_N,
    DIAZEPAM_RAT_PO_SOURCE_SAMPLING_TIMES_H,
    DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H,
    DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_SD_H,
    DIAZEPAM_RAT_PO_TMAX_H,
    DIAZEPAM_RAT_PO_TMAX_SD_H,
    DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG,
    DIAZEPAM_RAT_PO_VZ_OVER_F_SD_L_PER_KG,
    NORDIAZEPAM_MOLAR_MASS_G_PER_MOL,
    NORDIAZEPAM_RAT_PLASMA_BOUND_FRACTION,
    NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L,
    NORDIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L,
    NORDIAZEPAM_RAT_PO_CMAX_UG_PER_L,
    NORDIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H,
    NORDIAZEPAM_RAT_PO_TMAX_H,
    UnsourcedParameterError,
    diazepam_mouse_pk,
    diazepam_rat_ip_pk,
    diazepam_rat_iv_pk,
    diazepam_rat_pk,
)
from neurocascade.pk import LN2, concentration_mg_per_L_to_nM

#: A rat weight. Nothing in this file's half-life or Tmax assertions depends
#: on it (`test_derived_quantities_do_not_depend_on_the_body_weight_passed`
#: proves that), and none of the four sources reports a weight for the two
#: routes that are modelled, so no weight is "the source's".
BODY_WEIGHT_KG = 0.25
#: [Friedman1986]'s own IP dose.
IP_DOSE_MG_PER_KG = 5.0


def integrate(model, dose, t_eval):
    """Integrate a PK model standalone, through the engine's own protocol."""
    sol = solve_ivp(
        lambda t, s: model.concentration_rhs(s),
        (t_eval[0], t_eval[-1]),
        model.initial_state(dose),
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-16,
        method="RK45",
    )
    assert sol.success
    return model.concentration(sol.y)


def simulated_terminal_half_life_min(model, dose, window_min):
    """Half-life read off a *simulated* curve the way it is read off a real
    one: log-linear regression over a terminal window."""
    t = np.linspace(0.0, window_min[1], 2000)
    c = integrate(model, dose, t)
    terminal = t >= window_min[0]
    # Only the fitted window has to be positive: an extravascular model
    # starts at C(0) = 0 by construction (all of the dose is in the depot).
    assert np.all(c[terminal] > 0.0)
    slope = np.polyfit(t[terminal], np.log(c[terminal]), 1)[0]
    return LN2 / (-slope)


# ---------------------------------------------------------------------------
# Transcription: the constants are what the papers print
# ---------------------------------------------------------------------------


def test_reported_rat_iv_parameters_match_the_cited_abstract():
    # Klotz U, Antonin KH, Bieck PR. J Pharmacol Exp Ther 1976;199(1):67-73,
    # PMID 789856: "from 1.1 hours and 81.6 ml/min/kg in the rat"; rat plasma
    # binding of diazepam 86.3% and of desmethyldiazepam 90.5%; T1/2(alpha)
    # "varied only approximately 3-fold (0.3-1.0 hours) in the different
    # species".
    assert DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG == 81.6
    assert DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_MIN == pytest.approx(66.0)
    assert DIAZEPAM_RAT_PLASMA_BOUND_FRACTION == 0.863
    assert NORDIAZEPAM_RAT_PLASMA_BOUND_FRACTION == 0.905
    # Named for what it is: a four-species range, not a rat value. If a
    # future edit renames it into a rat alpha half-life this fails.
    assert DIAZEPAM_ALPHA_HALF_LIFE_RANGE_ALL_SPECIES_H == (0.3, 1.0)
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_IV_DISTRIBUTION_HALF_LIFE_MIN")
    # F for an intravenous dose is 1 by definition, not by measurement.
    assert DIAZEPAM_RAT_IV_F == 1.0
    # Chemical facts (PubChem CID 3016 and 2997).
    assert DIAZEPAM_MOLAR_MASS_G_PER_MOL == 284.74
    assert NORDIAZEPAM_MOLAR_MASS_G_PER_MOL == 270.71


def test_reported_rat_ip_parameters_match_the_cited_abstract():
    # Friedman H, Abernethy DR, Greenblatt DJ, Shader RI. Psychopharmacology
    # (Berl) 1986;88(3):267-70, PMID 3083446: 5 mg/kg IP; overall half-lives
    # 0.88 h (plasma) and 0.89 h (brain); apparent Vd 19.3 l/kg; apparent
    # total clearance 255 ml/kg/min; free fraction 19.6%; brain:plasma 4.5.
    assert DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG == 19.3
    assert DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG == 255.0
    assert DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN == pytest.approx(52.8)
    assert DIAZEPAM_RAT_IP_BRAIN_HALF_LIFE_MIN == pytest.approx(53.4)
    assert DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO == 4.5
    assert DIAZEPAM_RAT_IP_FREE_FRACTION == 0.196
    assert DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG == 5.0
    assert DIAZEPAM_RAT_IP_SOURCE_N_PER_TIMEPOINT == 4
    assert DIAZEPAM_RAT_IP_SOURCE_SAMPLING_TIMES_MIN[0] == 5.0
    assert DIAZEPAM_RAT_IP_SOURCE_SAMPLING_TIMES_MIN[-1] == 360.0


def test_reported_rat_ip_absorption_constants_match_the_cited_abstract():
    # Hironaka T, Fuchino K, Fujii T. J Pharmacol Exp Ther 1984;229(3):809-15,
    # PMID 6726659: "The specific rate constants for absorption, distribution
    # (k12 and k21) and elimination were 0.158, 0.015 and 0.136 and 0.013/min,
    # respectively", after 0.9 mg/kg i.p. "to the rat of 21 days of age".
    assert DIAZEPAM_RAT_IP_KA_PER_MIN == 0.158
    assert DIAZEPAM_RAT_IP_KA_SOURCE_K12_PER_MIN == 0.015
    assert DIAZEPAM_RAT_IP_KA_SOURCE_K21_PER_MIN == 0.136
    assert DIAZEPAM_RAT_IP_KA_SOURCE_KEL_PER_MIN == 0.013
    # The two facts that make this a cross-study composite rather than a
    # parameter set, asserted so they cannot quietly leave the record.
    assert DIAZEPAM_RAT_IP_KA_SOURCE_AGE_DAYS == 21
    assert DIAZEPAM_RAT_IP_KA_SOURCE_DOSE_MG_PER_KG == 0.9
    assert (
        DIAZEPAM_RAT_IP_KA_SOURCE_DOSE_MG_PER_KG
        != DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG
    )


def test_reported_rat_po_parameters_match_the_cited_tables():
    # Hua A et al. Iran J Pharm Res 2025;24(1):e164825, PMID 41477118,
    # control arm, Table 1 (diazepam) and Table 2 (nordiazepam), diazepam
    # 10 mg/kg by gavage in six male Sprague-Dawley rats of 220 +/- 20 g.
    assert DIAZEPAM_RAT_PO_CMAX_UG_PER_L == 57.75
    assert DIAZEPAM_RAT_PO_CMAX_SD_UG_PER_L == 24.81
    assert DIAZEPAM_RAT_PO_TMAX_H == 1.33
    assert DIAZEPAM_RAT_PO_TMAX_SD_H == 0.52
    assert DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H == 1.81
    assert DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_SD_H == 1.26
    assert DIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L == 172.25
    assert DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L == 182.02
    assert DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG == 59.78
    assert DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG == 164.39
    assert DIAZEPAM_RAT_PO_VZ_OVER_F_SD_L_PER_KG == 141.97
    assert DIAZEPAM_RAT_PO_MRT_0_INF_H == 3.11
    assert DIAZEPAM_RAT_PO_SOURCE_DOSE_MG_PER_KG == 10.0
    assert DIAZEPAM_RAT_PO_SOURCE_BODY_WEIGHT_KG == 0.220
    assert DIAZEPAM_RAT_PO_SOURCE_N == 6
    assert DIAZEPAM_RAT_PO_SOURCE_SAMPLING_TIMES_H[0] == 0.083
    assert DIAZEPAM_RAT_PO_SOURCE_SAMPLING_TIMES_H[-1] == 8.0
    assert NORDIAZEPAM_RAT_PO_CMAX_UG_PER_L == 47.75
    assert NORDIAZEPAM_RAT_PO_TMAX_H == 1.25
    assert NORDIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H == 1.97
    assert NORDIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L == 127.95
    assert NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L == 139.73
    # The SDs on the terminal-phase estimates are what disqualify this
    # dataset's late phase, so the relation is asserted, not just recorded.
    assert DIAZEPAM_RAT_PO_VZ_OVER_F_SD_L_PER_KG > 0.85 * (
        DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG
    )
    # Nordiazepam's peak occurs at essentially the same time as the parent's:
    # this is the metabolite fact that makes it a same-timescale problem.
    assert abs(NORDIAZEPAM_RAT_PO_TMAX_H - DIAZEPAM_RAT_PO_TMAX_H) < 0.1


# ---------------------------------------------------------------------------
# Behaviour: the simulated curves reproduce the sources
# ---------------------------------------------------------------------------


def test_simulated_iv_half_life_matches_the_cited_1_1_hours():
    """P1.4's "simulated half-life matches the cited value" criterion, IV.

    The tolerance is the interval that "1.1 hours", printed to two
    significant figures, actually constrains (1.05-1.15 h), not a chosen
    percentage. The exact simulated value is asserted too, so the real
    agreement is on the record rather than hidden behind a bound.
    """
    model = diazepam_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    simulated = simulated_terminal_half_life_min(
        model, 1.0, window_min=(120.0, 400.0)
    )
    low, high = DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN
    assert low <= simulated <= high
    assert simulated == pytest.approx(66.0, abs=0.01)
    assert model.terminal_half_life_min == pytest.approx(66.0, abs=1e-9)
    # Vd = CL/beta is what makes that exact, so assert the volume too.
    assert DIAZEPAM_RAT_IV_VBETA_L_PER_KG == pytest.approx(7.7698, abs=1e-4)


def test_simulated_ip_half_life_matches_the_cited_0_88_hours():
    """Same criterion, IP - and the real 0.64% disagreement is reported
    rather than absorbed into a tolerance.

    [Friedman1986] prints V/F = 19.3 L/kg, CL/F = 255 mL/min/kg and a
    half-life of 0.88 h. The first two imply 52.46 min and the third is
    52.8 min. The model is built from the first two (they are what set the
    concentration scale), so it lands on 52.46, and this test asserts that
    value exactly *and* asserts that the gap to the printed half-life is
    under 1%. Widening a tolerance to 0.88 h would have hidden a real, if
    small, internal inconsistency in the source.
    """
    model = diazepam_rat_ip_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    # Start the terminal window well past the absorption phase: the composite
    # ka is 0.158/min, so ten absorption half-lives is ~44 min.
    simulated = simulated_terminal_half_life_min(
        model, IP_DOSE_MG_PER_KG, window_min=(150.0, 400.0)
    )
    assert simulated == pytest.approx(52.4617, abs=0.01)

    implied, reported, relative_error = (
        drug_pk.diazepam_rat_ip_reported_half_life_consistency()
    )
    assert implied == pytest.approx(52.4617, abs=1e-3)
    assert reported == pytest.approx(52.8)
    assert relative_error == pytest.approx(-0.006407, abs=1e-5)
    assert abs(relative_error) < 0.01


def test_the_three_rat_studies_agree_on_the_disposition_envelope():
    """The one check that licenses the cross-study IP composite.

    59.7 min ([Hironaka1984]'s own constants, IP, weanlings), 52.8 min
    ([Friedman1986], IP, adults) and 66.0 min ([Klotz1976], IV). A 25% spread
    across two routes, three assays and two decades. Asserted at its real
    width: this is deliberately not called good agreement.
    """
    hironaka, friedman, klotz = (
        drug_pk.diazepam_rat_ip_composite_disposition_cross_check()
    )
    assert hironaka == pytest.approx(59.748, abs=0.01)
    assert friedman == pytest.approx(52.8)
    assert klotz == pytest.approx(66.0)
    spread = max(hironaka, friedman, klotz) / min(hironaka, friedman, klotz)
    assert spread == pytest.approx(1.25, abs=0.01)


def test_composite_ip_model_peaks_later_than_the_disposition_source_sampled():
    """The composite's most likely error, pinned so it cannot be forgotten.

    ka = 0.158/min puts Tmax at 17.1 min, whereas [Friedman1986]'s first
    post-dose sample - the one that would have located the true adult peak -
    is at 5 min. The direction matters: if the adult peak really is at or
    before 5 min, the composite absorbs too slowly and understates early
    concentrations.
    """
    tmax = drug_pk.diazepam_rat_ip_predicted_tmax_min()
    assert tmax == pytest.approx(17.138, abs=0.01)
    assert tmax > DIAZEPAM_RAT_IP_SOURCE_SAMPLING_TIMES_MIN[0]
    # And the model agrees with the closed-form Bateman stationary point.
    model = diazepam_rat_ip_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    t = np.linspace(0.0, 60.0, 6001)
    c = integrate(model, IP_DOSE_MG_PER_KG, t)
    assert t[int(np.argmax(c))] == pytest.approx(tmax, abs=0.05)


def test_apparent_parameterisation_is_used_and_declared_for_ip():
    """The IP model must never claim a measured bioavailability.

    Its V/F and CL/F already contain F, so `F == 1` is bookkeeping. The two
    flags `pk.py` exposes are what distinguish that from a measurement, and
    both are asserted, along with the constants' `_OVER_F_` naming - the
    single most likely way this block decays is someone tidying `OVER_F` out
    of a name and turning an apparent clearance into a clearance.
    """
    model = diazepam_rat_ip_pk(BODY_WEIGHT_KG)
    assert model.route == "IP"
    assert model.F == 1.0
    assert model.apparent_parameters is True
    assert model.F_is_measured is False
    assert "apparent parameters" in repr(model)
    # The IV model is the opposite case and must not be flagged apparent.
    iv = diazepam_rat_iv_pk(BODY_WEIGHT_KG)
    assert iv.apparent_parameters is False

    assert hasattr(drug_pk, "DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG")
    assert hasattr(drug_pk, "DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG")
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_IP_CL_ML_PER_MIN_PER_KG")
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_IP_VD_L_PER_KG")
    assert hasattr(drug_pk, "DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG")
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_PO_CL_L_PER_H_PER_KG")


def test_auc_identity_holds_for_both_modelled_routes():
    """AUC = Dose/CL for IV and Dose/(CL/F) for the apparent IP model, both
    exactly and both against a numerically integrated curve."""
    iv = diazepam_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    dose = 2.0
    analytic = iv.auc_0_inf(dose)
    assert analytic == pytest.approx(
        dose * BODY_WEIGHT_KG / (DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG
                                 * BODY_WEIGHT_KG / 1000.0)
    )
    t = np.linspace(0.0, 4000.0, 200001)
    numeric = np.trapezoid(integrate(iv, dose, t), t)
    assert numeric == pytest.approx(analytic, rel=1e-5)

    ip = diazepam_rat_ip_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    analytic_ip = ip.auc_0_inf(IP_DOSE_MG_PER_KG)
    numeric_ip = np.trapezoid(integrate(ip, IP_DOSE_MG_PER_KG, t), t)
    assert numeric_ip == pytest.approx(analytic_ip, rel=1e-5)
    # Absorption cannot create or destroy exposure: the same dose by the two
    # routes differs only by the ratio of clearance to apparent clearance.
    assert analytic_ip / iv.auc_0_inf(IP_DOSE_MG_PER_KG) == pytest.approx(
        DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG
        / DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG
    )


def test_derived_quantities_do_not_depend_on_the_body_weight_passed():
    """Half-lives and Tmax are weight-invariant; absolute amounts are not."""
    for weight in (0.15, 0.25, 0.40):
        iv = diazepam_rat_iv_pk(weight, molar_mass_g_per_mol=None)
        ip = diazepam_rat_ip_pk(weight, molar_mass_g_per_mol=None)
        assert iv.terminal_half_life_min == pytest.approx(66.0, abs=1e-9)
        assert ip.terminal_half_life_min == pytest.approx(52.4617, abs=1e-3)
        assert ip.tmax_min() == pytest.approx(17.138, abs=1e-3)
        # ... while the absolute quantities track the weight exactly.
        assert iv.V_central_L == pytest.approx(
            DIAZEPAM_RAT_IV_VBETA_L_PER_KG * weight
        )
        assert iv.CL_L_per_min == pytest.approx(
            DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG * weight / 1000.0
        )


def test_nanomolar_conversion_uses_the_right_molar_mass():
    """5 mg/kg IP in a 250 g rat, in nM, is the concentration scale the
    receptor layer would actually see, so it is checked end to end."""
    model_nm = diazepam_rat_ip_pk(BODY_WEIGHT_KG)
    model_mg = diazepam_rat_ip_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    assert model_nm.concentration_unit == "nM"
    assert model_mg.concentration_unit == "mg/L"
    t = np.linspace(0.0, 240.0, 481)
    nm = integrate(model_nm, IP_DOSE_MG_PER_KG, t)
    mg = integrate(model_mg, IP_DOSE_MG_PER_KG, t)
    np.testing.assert_allclose(
        nm,
        concentration_mg_per_L_to_nM(mg, DIAZEPAM_MOLAR_MASS_G_PER_MOL),
        rtol=1e-9,
    )
    # Sanity on the magnitude: the peak after 5 mg/kg IP should sit in the
    # high-hundreds of nM, i.e. two orders above the 3.6 nM benzodiazepine-
    # site Ki the calibrated GABA-A scenario uses. A unit slip would show up
    # here as orders of magnitude.
    assert 100.0 < nm.max() < 3000.0


# ---------------------------------------------------------------------------
# Honesty about the seams: derived numbers stay derived
# ---------------------------------------------------------------------------


def test_both_bioavailabilities_are_derived_across_studies_not_measured():
    cl_iv, cl_over_f_ip, f_ip = drug_pk.diazepam_rat_ip_bioavailability()
    assert cl_iv == DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG
    assert cl_over_f_ip == DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG
    assert f_ip == pytest.approx(0.320, abs=1e-3)

    f_po_from_cl, f_po_from_auc = drug_pk.diazepam_rat_po_bioavailability()
    assert f_po_from_cl == pytest.approx(0.0819, abs=1e-4)
    assert f_po_from_auc == pytest.approx(0.0891, abs=1e-4)
    # The oral route is severalfold less available than the intraperitoneal
    # one, which is the qualitative shape a high-extraction drug given by a
    # partly-portal route should have.
    assert f_ip > 3.0 * f_po_from_cl

    # These are the ONLY bioavailability values in the block, and neither is
    # a module constant, because neither was measured. A future pass that
    # promotes one to a `DIAZEPAM_RAT_IP_F = 0.32` constant fails here.
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_IP_F")
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_PO_F")
    assert not hasattr(drug_pk, "DIAZEPAM_RAT_SC_F")
    assert not hasattr(drug_pk, "DIAZEPAM_MOUSE_F")


def test_no_absorption_constant_exists_for_a_route_that_did_not_measure_one():
    assert hasattr(drug_pk, "DIAZEPAM_RAT_IP_KA_PER_MIN")
    for name in (
        "DIAZEPAM_RAT_PO_KA_PER_MIN",
        "DIAZEPAM_RAT_SC_KA_PER_MIN",
        "DIAZEPAM_MOUSE_IP_KA_PER_MIN",
    ):
        assert not hasattr(drug_pk, name)


def test_the_oral_dataset_does_not_identify_an_absorption_rate():
    """Why the PO route is recorded and not modelled, as arithmetic.

    Two absorption rate constants come out of the same paper depending on
    which of its numbers you read: 0.0217/min from the reported Tmax,
    0.0996/min from the reported Cmax-to-AUC ratio. 4.6x apart.
    """
    ka_tmax, ka_cmax, ratio = drug_pk.diazepam_rat_po_ka_is_not_identified()
    assert ka_tmax == pytest.approx(0.021742, abs=1e-6)
    assert ka_cmax == pytest.approx(0.099580, abs=1e-6)
    assert ratio == pytest.approx(4.580, abs=0.01)
    assert ratio > 4.0

    # Unlike the fentanyl SC case, the reported Cmax does NOT breach the
    # structural one-compartment ceiling AUC*ke - so what rules this dataset
    # out is the ka ambiguity, not model-structure incompatibility, and the
    # two failure modes are asserted separately.
    ceiling = DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L * (
        LN2 / DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H
    )
    assert DIAZEPAM_RAT_PO_CMAX_UG_PER_L < ceiling
    assert DIAZEPAM_RAT_PO_CMAX_UG_PER_L / ceiling == pytest.approx(
        0.8285, abs=1e-3
    )


def test_the_oral_datasets_two_internal_inconsistencies_are_pinned():
    reported_vz, implied_vz = drug_pk.diazepam_rat_po_apparent_volume_l_per_kg()
    assert reported_vz == DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG
    assert implied_vz == pytest.approx(156.10, abs=0.01)
    assert abs(implied_vz - reported_vz) / reported_vz == pytest.approx(
        0.0504, abs=1e-3
    )

    dose_over_auc, reported_cl, relative_error = (
        drug_pk.diazepam_rat_po_clearance_auc_consistency()
    )
    assert dose_over_auc == pytest.approx(54.939, abs=1e-3)
    assert reported_cl == DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG
    assert relative_error == pytest.approx(-0.0810, abs=1e-3)


def test_clearance_exceeds_hepatic_blood_flow_so_no_f_may_be_assumed_near_one():
    clearance, hepatic_flow = drug_pk.diazepam_hepatic_extraction_note()
    assert clearance == 81.6  # [Klotz1976], plasma
    assert hepatic_flow == pytest.approx(55.2)  # [DaviesMorris1993]
    assert clearance > hepatic_flow


# ---------------------------------------------------------------------------
# The active-metabolite question, which P1.4 was required to answer
# ---------------------------------------------------------------------------


def test_active_metabolite_exposure_is_quantified_in_molar_terms():
    """Nordiazepam carries 81% of the parent's molar exposure in the rat.

    The molar conversion is the point: the mass-based ratio (0.768) is not
    the number that matters for receptor occupancy, and the two differ by 5%.
    """
    dz, ndz, ratio = drug_pk.diazepam_nordiazepam_molar_exposure_ratio()
    assert dz == pytest.approx(
        DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L / DIAZEPAM_MOLAR_MASS_G_PER_MOL
    )
    assert ndz == pytest.approx(
        NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L / NORDIAZEPAM_MOLAR_MASS_G_PER_MOL
    )
    assert ratio == pytest.approx(0.8074, abs=1e-3)
    mass_ratio = (
        NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L
        / DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L
    )
    assert ratio > mass_ratio
    assert ratio / mass_ratio == pytest.approx(
        DIAZEPAM_MOLAR_MASS_G_PER_MOL / NORDIAZEPAM_MOLAR_MASS_G_PER_MOL
    )


def test_active_metabolite_limitation_is_recorded_and_not_modelled():
    """P1.4's "active-metabolite question addressed explicitly" criterion.

    The metabolite is *not* modelled - doing so honestly needs a rat
    formation fraction and a numeric nordiazepam Ki/efficacy, neither of
    which this project has. What is required instead is that the decision be
    on the record with its evidence, so that a later reader finds a
    documented limitation rather than an omission. This test guards exactly
    that, in both directions.
    """
    record = DIAZEPAM_ACTIVE_METABOLITE_LIMITATION
    # The three legs of the argument, and the conclusion, must all be there.
    for token in (
        "EXPOSURE",
        "POTENCY",
        "TIMESCALE",
        "UNDERSTATES",
        "Gobbi1987",
        "Hua2025",
        "Friedman1986",
        "partial agonist",
        "TEMAZEPAM AND OXAZEPAM",
    ):
        assert token in record, token
    # And it must carry the numbers, not just the claim.
    assert "139.73" in record and "182.02" in record

    # Nothing anywhere in the codebase silently models nordiazepam: no PK
    # model, no ligand, no Ki. If a later pass adds one it must also update
    # this record, which is the behaviour this assertion is buying.
    assert not hasattr(drug_pk, "nordiazepam_rat_pk")
    assert not hasattr(drug_pk, "NORDIAZEPAM_RAT_KI_NM")
    from neurocascade import scenarios

    ligand_names = {
        lig.name for lig in scenarios.gaba_a_benzodiazepine_ligands_calibrated()
    }
    assert ligand_names == {"diazepam", "flumazenil"}


# ---------------------------------------------------------------------------
# Refusals: the routes and species with no usable parameter set
# ---------------------------------------------------------------------------


def test_modelled_routes_dispatch_and_unmodelled_ones_raise_the_record():
    iv = diazepam_rat_pk("IV", BODY_WEIGHT_KG)
    assert iv.route == "IV"
    ip = diazepam_rat_pk("ip", BODY_WEIGHT_KG)  # case-insensitive
    assert ip.route == "IP"

    with pytest.raises(UnsourcedParameterError) as po:
        diazepam_rat_pk("PO", BODY_WEIGHT_KG)
    # PO is "present and unusable", and its record has to say which.
    assert "DISPOSITION is sourced" in str(po.value)
    assert "4.6-fold" in str(po.value)
    assert "PMID 41477118" in str(po.value)

    with pytest.raises(UnsourcedParameterError) as sc:
        diazepam_rat_pk("SC", BODY_WEIGHT_KG)
    # SC is "absent", a different state, and its record says so.
    assert "No pharmacokinetic dataset of any kind" in str(sc.value)
    assert "not set to 1" in str(sc.value)


def test_mouse_raises_for_every_route_and_is_not_scaled_from_the_rat():
    for route in ("IV", "IP", "SC", "PO"):
        with pytest.raises(UnsourcedParameterError) as excinfo:
            diazepam_mouse_pk(route, 0.025)
        message = str(excinfo.value)
        assert "No pharmacokinetic parameter set for diazepam in the MOUSE" in message
        # The single most useful thing in the record: which paper to read
        # next if someone gets journal access.
        assert "PMID 1979181" in message
        assert "FIRST PAPER A FUTURE PASS" in message
        # And the drug-specific reason allometric scaling is refused.
        assert "which diazepam metabolite dominates" in message

    assert not hasattr(drug_pk, "DIAZEPAM_MOUSE_CL_ML_PER_MIN_PER_KG")
    assert not hasattr(drug_pk, "DIAZEPAM_MOUSE_VD_L_PER_KG")


def test_every_searched_and_empty_diazepam_record_names_what_was_searched():
    keys = [k for k in drug_pk.SEARCHED_AND_EMPTY if k[0] == "diazepam"]
    assert set(keys) == {
        ("diazepam", "rat", "PO"),
        ("diazepam", "rat", "SC"),
        ("diazepam", "mouse", "*"),
        ("diazepam", "rat", "blood:plasma"),
    }
    for key in keys:
        record = drug_pk.SEARCHED_AND_EMPTY[key]
        assert len(record) > 400
        if key[2] != "blood:plasma":
            assert "Searched" in record or "searched" in record


# ---------------------------------------------------------------------------
# One cross-check against the rest of the codebase
# ---------------------------------------------------------------------------


def test_ip_dose_reaches_concentrations_above_the_calibrated_ki():
    """A dose-to-occupancy sanity check, not a calibration.

    The calibrated GABA-A scenario binds diazepam with Ki = 3.6 nM
    ([Mohler1977], see `scenarios.py`). The lowest elevated-plus-maze dose
    Track E recorded is 1 mg/kg IP. If that dose did not produce plasma
    concentrations above 3.6 nM, either the PK block or the binding block
    would have a unit error, and it would be invisible in every other test in
    this file.

    The margin is worth stating exactly rather than waving at, because it is
    smaller than one might expect and it is the reason P1.5 matters: peak
    total plasma is 145 nM, about 40x Ki, but only 19.6% of that is unbound
    ([Friedman1986]), so unbound plasma peaks near 28 nM, about 8x Ki. Brain
    total is another 4.5-fold higher and brain binding is not known here.
    Whether the receptor sees 8x Ki or 40x Ki is exactly the question P1.5
    exists to answer, and this test records that the answer is not obvious.
    """
    model = diazepam_rat_ip_pk(BODY_WEIGHT_KG)
    t = np.linspace(0.0, 120.0, 1201)
    peak_nm = integrate(model, 1.0, t).max()
    assert peak_nm == pytest.approx(145.09, abs=0.05)
    assert peak_nm > 10.0 * 3.6
    # Unbound, using [Friedman1986]'s own free fraction, it is still above Ki
    # but by less than an order of magnitude.
    unbound_nm = peak_nm * DIAZEPAM_RAT_IP_FREE_FRACTION
    assert 3.6 < unbound_nm < 36.0
    # And the arithmetic behind it is the plain apparent-volume one.
    expected_scale = concentration_mg_per_L_to_nM(
        1.0 / DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG, DIAZEPAM_MOLAR_MASS_G_PER_MOL
    )
    assert peak_nm < expected_scale  # absorption never delivers it all at once
    assert peak_nm > 0.5 * expected_scale
    assert math.isfinite(peak_nm)
