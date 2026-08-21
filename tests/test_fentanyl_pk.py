"""Tests for the fentanyl rat PK parameter set (P1.2).

Kept in its own file rather than appended to `test_drug_pk.py` because the
two drugs are in a genuinely different evidential position and the tests
have to look different as a result:

* Haloperidol's source reports a terminal half-life ("1.5 h") and no
  inter-compartmental clearance, so P1.3's central test compares a simulated
  half-life against a printed one and its central caveat is a derived Q.
* Fentanyl's IV source ([Yassen2008]) reports the complete parameter set,
  including Q, and no half-life at all. So the half-life here is the derived
  quantity, and the test that corresponds to "simulated half-life matches the
  cited half-life" has to be stated precisely: the simulated curve must
  reproduce the half-life that the *source's own four parameters* imply, to
  numerical precision, and the uncertainty that the source's 95% CIs leave on
  that half-life is asserted separately so its width is on the record.

Three kinds of assertion live here, deliberately kept apart:

1. **Transcription.** Constants equal what the cited papers print, so a later
   edit that rounds 917.60 mL/h into 918 fails loudly.
2. **Behaviour.** The simulated concentration-time course, integrated through
   the same ODE protocol `engine.Simulation` uses, reproduces the source's
   parameters (half-lives, Vss, Vz, AUC = Dose/CL, dose-linearity).
3. **Refusal.** Every route with no usable parameter set raises, and the
   exception carries the search record; and no bioavailability or absorption
   constant exists for any route except the two that were actually measured
   (IV F = 1 by definition, PO F = 1.5% by Schneider & Brune 1985).
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade.legacy import drug_pk
from neurocascade.circuit import CircuitNode
from neurocascade.legacy.drug_pk import (
    FENTANYL_MOLAR_MASS_G_PER_MOL,
    FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML,
    FENTANYL_RAT_IP_CL_OVER_F_ML_PER_MIN,
    FENTANYL_RAT_IP_CMAX_NG_PER_ML,
    FENTANYL_RAT_IP_MRT_0_180_MIN,
    FENTANYL_RAT_IP_SOURCE_N,
    FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN,
    FENTANYL_RAT_IP_TMAX_MIN,
    FENTANYL_RAT_IV_BOOTSTRAP,
    FENTANYL_RAT_IV_CL_CI_ML_PER_MIN,
    FENTANYL_RAT_IV_CL_ML_PER_MIN,
    FENTANYL_RAT_IV_DISTRIBUTION_HALF_LIFE_MIN,
    FENTANYL_RAT_IV_F,
    FENTANYL_RAT_IV_Q_CI_ML_PER_MIN,
    FENTANYL_RAT_IV_Q_ML_PER_MIN,
    FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG,
    FENTANYL_RAT_IV_SOURCE_DOSE_RANGE_MG_PER_KG,
    FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN,
    FENTANYL_RAT_IV_V1_CI_ML,
    FENTANYL_RAT_IV_V1_ML,
    FENTANYL_RAT_IV_V2_CI_ML,
    FENTANYL_RAT_IV_V2_ML,
    FENTANYL_RAT_IV_VSS_ML,
    FENTANYL_RAT_IV_VZ_ML,
    FENTANYL_RAT_PO_F,
    FENTANYL_RAT_SC_AUC_NG_H_PER_ML,
    FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H,
    FENTANYL_RAT_SC_CMAX_NG_PER_ML,
    FENTANYL_RAT_SC_SOURCE_BODY_WEIGHT_KG,
    FENTANYL_RAT_SC_SOURCE_DOSE_MG_PER_KG,
    FENTANYL_RAT_SC_SOURCE_SAMPLING_TIMES_MIN,
    FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN,
    FENTANYL_RAT_SC_TMAX_H,
    FENTANYL_RAT_SC_VD_OVER_F_ML,
    UnsourcedParameterError,
    fentanyl_mouse_pk,
    fentanyl_rat_iv_one_compartment_pk,
    fentanyl_rat_iv_pk,
    fentanyl_rat_pk,
)
from neurocascade.engine import Simulation
from neurocascade.network import CircuitNetwork
from neurocascade.pk import LN2, concentration_mg_per_L_to_nM
from neurocascade.receptor import Ligand, ReceptorSite

#: Midpoint of [Yassen2008]'s "225-250 g" band. Only the concentration scale
#: depends on it (the disposition rate constants do not), and
#: `test_derived_half_lives_do_not_depend_on_the_body_weight_passed` proves
#: exactly that rather than leaving it as a comment.
BODY_WEIGHT_KG = 0.2375
#: Inside [Yassen2008]'s 0.015-0.085 mg/kg IV range.
DOSE_MG_PER_KG = 0.05


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


def simulated_terminal_half_life_min(model, dose, window_min=(180.0, 360.0)):
    """Half-life read off a *simulated* curve the way it is read off a real
    one: log-linear regression on the terminal portion.

    The window starts at 180 min, thirty distribution half-lives (5.98 min)
    after the dose, so what is fitted is the terminal slope and not a mixture
    of the two phases.
    """
    t = np.linspace(0.0, window_min[1], 1500)
    c = integrate(model, dose, t)
    assert np.all(c > 0.0)
    terminal = t >= window_min[0]
    slope = np.polyfit(t[terminal], np.log(c[terminal]), 1)[0]
    return LN2 / (-slope)


# ---------------------------------------------------------------------------
# Transcription: the constants are what the papers print
# ---------------------------------------------------------------------------


def test_reported_rat_iv_parameters_match_the_cited_table():
    # Yassen A, Olofsen E, Kan J, Dahan A, Danhof M. Pharm Res 2008;25:183-93,
    # PMID 17914664, Table II ("Original Data" column, 95% CI in brackets):
    # Cl 17.0 [16.9-18.1] ml/min, V1 250 [218-282] ml, V2 440 [404-476] ml,
    # Q 9.7 [8.3-11.1] ml/min.
    assert FENTANYL_RAT_IV_CL_ML_PER_MIN == 17.0
    assert FENTANYL_RAT_IV_V1_ML == 250.0
    assert FENTANYL_RAT_IV_V2_ML == 440.0
    assert FENTANYL_RAT_IV_Q_ML_PER_MIN == 9.7
    assert FENTANYL_RAT_IV_CL_CI_ML_PER_MIN == (16.9, 18.1)
    assert FENTANYL_RAT_IV_V1_CI_ML == (218.0, 282.0)
    assert FENTANYL_RAT_IV_V2_CI_ML == (404.0, 476.0)
    assert FENTANYL_RAT_IV_Q_CI_ML_PER_MIN == (8.3, 11.1)
    # Same table, "1,000 Bootstrap Replicates" column.
    assert FENTANYL_RAT_IV_BOOTSTRAP == (16.6, 255.0, 444.0, 9.7)
    # Male Wistar rats "weighting initially 225-250 g"; fentanyl 0.015-0.085
    # mg/kg by zero-order IV infusion.
    assert FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG == (0.225, 0.250)
    assert FENTANYL_RAT_IV_SOURCE_DOSE_RANGE_MG_PER_KG == (0.015, 0.085)
    # F for an intravenous dose is 1 by definition, not by measurement.
    assert FENTANYL_RAT_IV_F == 1.0
    # Molar mass is a chemical fact (PubChem CID 3345) and must stay
    # identical to the one calibration/evidence.py already uses, or the PK
    # layer and the occupancy evidence will silently disagree about how many
    # nM a ng/mL is.
    assert FENTANYL_MOLAR_MASS_G_PER_MOL == 336.47
    from neurocascade.calibration import evidence

    assert FENTANYL_MOLAR_MASS_G_PER_MOL == evidence.FENTANYL_MW


def test_reported_rat_sc_parameters_match_the_cited_table():
    # Canfield JR, Sprague JE. Arch Toxicol 2025;99:287-97, PMID 39419833,
    # Table 1, fentanyl column: t1/2 1.10 h, AUC0-inf 112.83 ng/mL x h,
    # Cmax 82.12 ng/mL, Tmax 0.5 h, Clp/F 917.60 mL/h, VD/F 1524.90 ml;
    # 300 ug/kg s.c. in 326.92 g male Sprague-Dawley rats.
    assert FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN == pytest.approx(66.0)
    assert FENTANYL_RAT_SC_AUC_NG_H_PER_ML == 112.83
    assert FENTANYL_RAT_SC_CMAX_NG_PER_ML == 82.12
    assert FENTANYL_RAT_SC_TMAX_H == 0.5
    assert FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H == 917.60
    assert FENTANYL_RAT_SC_VD_OVER_F_ML == 1524.90
    assert FENTANYL_RAT_SC_SOURCE_DOSE_MG_PER_KG == 0.300
    assert FENTANYL_RAT_SC_SOURCE_BODY_WEIGHT_KG == 0.32692
    # The reported Tmax coincides with the first post-dose sample. This is
    # the single fact that makes ka unavailable from this study, so it is
    # asserted rather than only written down.
    assert FENTANYL_RAT_SC_SOURCE_SAMPLING_TIMES_MIN == (
        0.0,
        30.0,
        60.0,
        120.0,
        240.0,
        480.0,
    )
    assert FENTANYL_RAT_SC_TMAX_H * 60.0 == FENTANYL_RAT_SC_SOURCE_SAMPLING_TIMES_MIN[1]


def test_reported_rat_ip_parameters_match_the_cited_table():
    # Almousa AA et al. J Chromatogr B 2011;879:2941-6, PMID 21917532,
    # Table 3, 0.1 mg/kg i.p., n = 3 male Wistar rats.
    assert FENTANYL_RAT_IP_TMAX_MIN == 28.3
    assert FENTANYL_RAT_IP_CMAX_NG_PER_ML == 21.3
    assert FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN == 198.3
    assert FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML == 1560.0
    assert FENTANYL_RAT_IP_MRT_0_180_MIN == 75.3
    assert FENTANYL_RAT_IP_CL_OVER_F_ML_PER_MIN == 19.4
    assert FENTANYL_RAT_IP_SOURCE_N == 3


def test_the_only_measured_rat_bioavailability_is_the_oral_one():
    # Schneider E, Brune K. Naunyn Schmiedebergs Arch Pharmacol
    # 1985;331:359-63, PMID 4094625: "Its oral bioavailability was found to
    # be only 1.5%."
    assert FENTANYL_RAT_PO_F == 0.015


def test_derived_quantities_follow_from_the_reported_ones():
    # Vss = V1 + V2 by definition for a two-compartment model.
    assert FENTANYL_RAT_IV_VSS_ML == pytest.approx(690.0)
    # Vz = CL/beta, the terminal-phase volume, derived from the model's own
    # hybrid rate constants rather than restated.
    assert FENTANYL_RAT_IV_VZ_ML == pytest.approx(1314.5, abs=0.5)
    # The half-lives [Yassen2008] does not report, implied by the four
    # parameters it does. Asserted to the digit so that a change to any
    # constant surfaces here.
    assert FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN == pytest.approx(53.60, abs=0.01)
    assert FENTANYL_RAT_IV_DISTRIBUTION_HALF_LIFE_MIN == pytest.approx(5.98, abs=0.01)
    # The bootstrap parameter set gives essentially the same terminal phase,
    # which is what makes the point estimate worth building on.
    cl, v1, v2, q = FENTANYL_RAT_IV_BOOTSTRAP
    from neurocascade.pk import DosedTwoCompartmentPK

    bootstrap_model = DosedTwoCompartmentPK(
        route="IV",
        V1_L_per_kg=v1 / 1000.0,
        V2_L_per_kg=v2 / 1000.0,
        Q_mL_per_min_per_kg=q,
        CL_mL_per_min_per_kg=cl,
        body_weight_kg=1.0,
    )
    assert bootstrap_model.terminal_half_life_min == pytest.approx(54.7, abs=0.2)


def test_the_derived_half_life_carries_its_sources_uncertainty():
    low, high = drug_pk.fentanyl_rat_iv_terminal_half_life_interval_min()
    # Corner enumeration over the source's four 95% CIs: 44.3-63.6 min. This
    # is an outer bound (the estimates are correlated and the covariance is
    # not published), and it is recorded so that any later use of this
    # half-life inherits the width knowingly instead of discovering it.
    assert low == pytest.approx(44.29, abs=0.05)
    assert high == pytest.approx(63.61, abs=0.05)
    assert low < FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN < high


# ---------------------------------------------------------------------------
# Behaviour: the simulated curve reproduces the source
# ---------------------------------------------------------------------------


def test_simulated_terminal_half_life_matches_the_half_life_the_source_implies():
    """P1.2's "simulated half-life matches the cited half-life", in the only
    form this source admits.

    [Yassen2008] prints a parameter set and no half-life, so the honest
    version of the check is: integrate the model through the engine's ODE
    protocol, read the terminal slope off the simulated curve exactly as it
    would be read off real plasma data, and require it to agree with the
    closed-form half-life those four published parameters imply.
    """
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG)
    simulated = simulated_terminal_half_life_min(model, DOSE_MG_PER_KG)
    assert simulated == pytest.approx(FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN, rel=1e-4)
    # The actual value, stated rather than merely bounded: 53.60 min.
    assert simulated == pytest.approx(53.60, abs=0.05)


def test_derived_half_lives_do_not_depend_on_the_body_weight_passed():
    """[Yassen2008]'s estimates are absolute (mL, mL/min) and are divided by
    the caller's body weight at construction, so this is the property that
    makes that division safe: the disposition is weight-invariant and only
    the concentration scale moves."""
    light = fentanyl_rat_iv_pk(FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG[0])
    heavy = fentanyl_rat_iv_pk(FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG[1])
    for attribute in ("terminal_half_life_min", "distribution_half_life_min"):
        assert getattr(light, attribute) == pytest.approx(getattr(heavy, attribute))
    # ...and the absolute volumes and clearance are the paper's, whichever
    # weight was passed, which is the point of storing them absolute.
    for model in (light, heavy):
        assert model.V_central_L * 1000.0 == pytest.approx(FENTANYL_RAT_IV_V1_ML)
        assert model.V_peripheral_L * 1000.0 == pytest.approx(FENTANYL_RAT_IV_V2_ML)
        assert model.CL_L_per_min * 1000.0 == pytest.approx(
            FENTANYL_RAT_IV_CL_ML_PER_MIN
        )
    # The concentration a mg/kg dose achieves does scale with weight, since
    # the same per-kg dose is a larger absolute dose into the same volume.
    ratio = (
        light.initial_state(DOSE_MG_PER_KG)[0]
        / heavy.initial_state(DOSE_MG_PER_KG)[0]
    )
    assert ratio == pytest.approx(
        FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG[0]
        / FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG[1]
    )


def test_model_reproduces_reported_vss_and_derived_vz():
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG)
    assert model.Vd_ss_L_per_kg * BODY_WEIGHT_KG * 1000.0 == pytest.approx(
        FENTANYL_RAT_IV_VSS_ML
    )
    # Vz = CL/beta recovered from the model's own hybrid rate constants: a
    # round-trip through `hybrid_rate_constants`, not a restatement.
    _alpha, beta = model.hybrid_rate_constants()
    assert FENTANYL_RAT_IV_CL_ML_PER_MIN / beta == pytest.approx(
        FENTANYL_RAT_IV_VZ_ML, rel=1e-9
    )


def test_two_compartment_model_is_genuinely_biphasic():
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG)
    # [Yassen2008] selected a two-compartment model over one- and
    # three-compartment alternatives; a model whose phases had collapsed
    # together would still pass the terminal half-life test while
    # misrepresenting the first half hour, which for fentanyl is when the
    # drug is doing almost everything it does.
    assert model.distribution_half_life_min < 0.15 * model.terminal_half_life_min
    assert model.distribution_half_life_min == pytest.approx(5.98, abs=0.02)


def test_one_compartment_collapse_trades_the_peak_for_the_terminal_phase():
    two = fentanyl_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    one = fentanyl_rat_iv_one_compartment_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    # Using Vd = Vz (not V1) is what makes the collapse preserve the terminal
    # phase; that is the whole reason for the choice.
    assert simulated_terminal_half_life_min(
        one, DOSE_MG_PER_KG
    ) == pytest.approx(FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN, rel=1e-4)
    # ...and the documented cost: the bolus peak is understated ~5-fold,
    # because the real drug starts in V1 = 250 mL and the collapse starts it
    # in Vz = 1314 mL.
    ratio = two.initial_state(DOSE_MG_PER_KG)[0] / one.initial_state(DOSE_MG_PER_KG)[0]
    assert ratio == pytest.approx(FENTANYL_RAT_IV_VZ_ML / FENTANYL_RAT_IV_V1_ML)
    assert 5.0 < ratio < 5.5
    # Total exposure is identical, since AUC depends only on clearance.
    assert one.auc_0_inf(DOSE_MG_PER_KG) == pytest.approx(
        two.auc_0_inf(DOSE_MG_PER_KG)
    )


def test_initial_concentration_uses_the_central_volume():
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    # An IV bolus starts at Dose/V1, not Dose/Vss or Dose/Vz.
    assert model.initial_state(DOSE_MG_PER_KG)[0] == pytest.approx(
        DOSE_MG_PER_KG * BODY_WEIGHT_KG * 1000.0 / FENTANYL_RAT_IV_V1_ML
    )


def test_auc_is_dose_over_clearance():
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    expected = DOSE_MG_PER_KG * BODY_WEIGHT_KG / model.CL_L_per_min
    assert model.auc_0_inf(DOSE_MG_PER_KG) == pytest.approx(expected)


def test_disposition_is_linear_over_the_dose_range_the_source_studied():
    # [Yassen2008] fitted one linear parameter set across 0.015-0.085 mg/kg.
    # A linear model cannot fail this; the point is that this codebase
    # represents the source's structure rather than contradicting it, and
    # this is the check that would fail first if a saturable-elimination term
    # were ever added without revisiting this parameter set.
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG)
    t = np.linspace(0.0, 360.0, 1500)
    low, high = FENTANYL_RAT_IV_SOURCE_DOSE_RANGE_MG_PER_KG
    reference = integrate(model, low, t)
    scaled = integrate(model, high, t)
    assert np.allclose(scaled, (high / low) * reference, rtol=1e-6, atol=1e-12)


def test_concentrations_are_reported_in_nM_by_default():
    nM_model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG)
    mgL_model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    assert nM_model.concentration_unit == "nM"
    assert mgL_model.concentration_unit == "mg/L"
    c0_mg_per_L = mgL_model.initial_state(DOSE_MG_PER_KG)[0]
    assert nM_model.initial_state(DOSE_MG_PER_KG)[0] == pytest.approx(
        concentration_mg_per_L_to_nM(c0_mg_per_L, FENTANYL_MOLAR_MASS_G_PER_MOL)
    )
    # Magnitude sanity, and a cross-check worth having: 0.05 mg/kg IV into
    # V1 = 250 mL for a 237.5 g rat is 0.0475 mg/L, i.e. 141 nM, the same
    # order as the 148.6 nM (50 ng/mL) human overdose exposure the calibrated
    # mu-opioid evidence uses. Two independent routes to a similar number is
    # weak evidence, but a three-order-of-magnitude disagreement would have
    # meant a unit error, which is the failure this asserts against.
    assert nM_model.initial_state(DOSE_MG_PER_KG)[0] == pytest.approx(141.2, abs=1.0)


def test_high_extraction_claim_is_checkable_not_asserted():
    cl, hepatic_blood_flow = drug_pk.fentanyl_hepatic_extraction_note()
    # 17.0 mL/min in a 237.5 g rat is 71.6 mL/min/kg of plasma clearance
    # against ~55 mL/min/kg of rat hepatic blood flow ([DaviesMorris1993]).
    # This is the documented reason no extravascular F for fentanyl is
    # assumed near 1.
    assert cl > hepatic_blood_flow
    assert cl == pytest.approx(71.6, abs=0.1)


# ---------------------------------------------------------------------------
# The two extravascular datasets that were found and could not be used
# ---------------------------------------------------------------------------


def test_sc_data_exceed_the_one_compartment_ceiling():
    cmax, ceiling, ratio = drug_pk.fentanyl_rat_sc_one_compartment_ceiling()
    # For any one-compartment model with first-order absorption,
    # Cmax <= AUC*ke holds at every ka. [CanfieldSprague2025]'s reported
    # means break it by 15.5%, which is why the SC route is not modelled from
    # that source: no ka exists that would reproduce them.
    assert cmax == FENTANYL_RAT_SC_CMAX_NG_PER_ML
    assert ceiling == pytest.approx(71.10, abs=0.05)
    assert ratio == pytest.approx(1.155, abs=0.002)
    assert ratio > 1.0


def test_sc_apparent_volume_has_two_readings_and_both_are_recorded():
    reported, implied = drug_pk.fentanyl_rat_sc_apparent_volume_ml()
    assert reported == FENTANYL_RAT_SC_VD_OVER_F_ML
    assert implied == pytest.approx(1456.2, abs=0.5)
    # ~4.5% apart, because a mean of per-animal CL/lambda_z ratios is not the
    # ratio of the reported means. Pinned so that the gap stays visible.
    assert 0.03 < abs(reported - implied) / implied < 0.06


def test_ip_dataset_contradicts_itself_by_a_factor_of_two():
    implied, reported, ratio = drug_pk.fentanyl_rat_ip_auc_consistency()
    assert reported == FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML
    assert implied == pytest.approx(3044.0, abs=5.0)
    assert ratio == pytest.approx(1.95, abs=0.02)


def test_ip_apparent_volume_is_the_part_that_does_corroborate():
    volume_ml = drug_pk.fentanyl_rat_ip_apparent_volume_ml()
    body_weight_kg = sum(drug_pk.FENTANYL_RAT_IP_SOURCE_BODY_WEIGHT_RANGE_KG) / 2.0
    per_kg = volume_ml / 1000.0 / body_weight_kg
    assert per_kg == pytest.approx(4.25, abs=0.02)
    # [CanfieldSprague2025]'s SC VD/F, by both of its readings, is 4.45-4.66
    # L/kg. The IP study's apparent volume lands just below that band: two
    # different labs, strains, routes and assays agreeing to within ~5% on
    # the one quantity they can both estimate.
    sc_reported, sc_implied = drug_pk.fentanyl_rat_sc_apparent_volume_ml()
    sc_band = sorted(
        v / 1000.0 / FENTANYL_RAT_SC_SOURCE_BODY_WEIGHT_KG
        for v in (sc_reported, sc_implied)
    )
    assert sc_band[0] == pytest.approx(4.45, abs=0.02)
    assert sc_band[1] == pytest.approx(4.66, abs=0.02)
    assert abs(per_kg - sc_band[0]) / sc_band[0] < 0.06


# ---------------------------------------------------------------------------
# Inside the coupled system
# ---------------------------------------------------------------------------


def test_parameter_set_runs_inside_the_coupled_simulation():
    """The model must work as a PK block of `Simulation`'s state vector, not
    only standalone: that is the whole reason P1.1 kept PK inside the ODE."""
    node_a = CircuitNode(name="VTA", tau=1.0, baseline_input=0.5, gain=4.0)
    node_b = CircuitNode(name="NAc", tau=2.0, baseline_input=0.5, gain=4.0)
    network = CircuitNetwork(
        nodes=[node_a, node_b], connectivity=[[0.0, 0.0], [1.0, 0.0]]
    )
    # kon/koff are the calibrated fentanyl mu-opioid constants from
    # scenarios.mu_opioid_ligands_calibrated; reused, not re-derived.
    site = ReceptorSite(
        name="mu",
        ligands=[Ligand(name="fentanyl", kon=0.1931, koff=0.2599, efficacy=1.0)],
    )
    model = fentanyl_rat_iv_pk(BODY_WEIGHT_KG)
    sim = Simulation(
        pk_models=[model], receptor_site=site, network=network, coupling={"VTA": 2.0}
    )
    t_eval = np.linspace(0.0, 360.0, 1500)
    result = sim.run(doses=[DOSE_MG_PER_KG], t_span=(0.0, 360.0), t_eval=t_eval)

    concentrations = result["concentrations"][0]
    # Two PK states (central + peripheral) for one ligand, and the reported
    # concentration is the central one.
    assert result["pk_states"].shape[0] == 2
    assert concentrations[0] == pytest.approx(model.initial_state(DOSE_MG_PER_KG)[0])
    # The coupled integration must give the same trace as the standalone one.
    standalone = integrate(model, DOSE_MG_PER_KG, t_eval)
    assert np.allclose(concentrations, standalone, rtol=1e-5, atol=1e-8)
    # An agonist at ~141 nM against a ligand whose Kd is 1.35 nM occupies
    # most of the pool early and lets go as the drug clears; a flat occupancy
    # trace would mean the PK is not reaching the receptor layer at all.
    occupancy = result["occupancy"][0]
    assert occupancy.max() > 0.5
    assert occupancy[-1] < occupancy.max()


# ---------------------------------------------------------------------------
# The routes and the species that have no numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["SC", "IP", "PO"])
def test_unsourced_rat_routes_raise_with_their_search_record(route):
    with pytest.raises(UnsourcedParameterError) as excinfo:
        fentanyl_rat_pk(route, BODY_WEIGHT_KG)
    message = str(excinfo.value)
    assert route in message
    assert "Searched" in message or "searched" in message
    assert drug_pk.SEARCHED_AND_EMPTY[("fentanyl", "rat", route)] in message


@pytest.mark.parametrize("route", ["IV", "SC", "IP", "PO"])
def test_no_mouse_route_is_available_and_the_rat_set_is_not_scaled(route):
    with pytest.raises(UnsourcedParameterError) as excinfo:
        fentanyl_mouse_pk(route, 0.025)
    message = str(excinfo.value)
    assert "mouse" in message
    assert "Kalvass" in message, "the search record must name the closed source to read"
    assert drug_pk.SEARCHED_AND_EMPTY[("fentanyl", "mouse", "*")] in message


def test_unknown_rat_route_also_refuses_rather_than_defaulting():
    with pytest.raises(UnsourcedParameterError):
        fentanyl_rat_pk("IM", BODY_WEIGHT_KG)


def test_no_unmeasured_bioavailability_or_absorption_constant_exists():
    """Guard against the most likely future mistake in this block: adding
    `FENTANYL_RAT_SC_F = 1.0` because subcutaneous dosing bypasses first
    pass, or a `KA` derived from the SC study's unresolved Tmax.

    Exactly two bioavailabilities are allowed to exist for this drug: the IV
    one, which is 1 by definition, and the oral one, which Schneider & Brune
    1985 measured. Anything else needs a citation and a
    docs/architecture.md update in the same commit.
    """
    allowed = {"FENTANYL_RAT_IV_F", "FENTANYL_RAT_PO_F"}
    for name in dir(drug_pk):
        if not name.startswith("FENTANYL_"):
            continue
        if name.endswith("_F") or "KA" in name:
            assert name in allowed, (
                "%s defines an unmeasured bioavailability or absorption rate; "
                "P1.2 recorded ka as searched-and-empty for every fentanyl "
                "route and F as available only for IV (by definition) and PO "
                "(Schneider & Brune 1985)" % name
            )


def test_apparent_parameters_are_named_as_apparent():
    """The SC and IP disposition constants are all divided by an unknown F.
    Naming them `CL_OVER_F` / `VD_OVER_F` rather than `CL` / `VD` is what
    stops a later reader treating them as absolute, so the naming is pinned
    here rather than left to review."""
    for name in ("FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H", "FENTANYL_RAT_SC_VD_OVER_F_ML"):
        assert hasattr(drug_pk, name)
    assert not hasattr(drug_pk, "FENTANYL_RAT_SC_CL_ML_PER_H")
    assert not hasattr(drug_pk, "FENTANYL_RAT_SC_VD_ML")
    assert hasattr(drug_pk, "FENTANYL_RAT_IP_CL_OVER_F_ML_PER_MIN")
    assert not hasattr(drug_pk, "FENTANYL_RAT_IP_CL_ML_PER_MIN")


def test_blood_plasma_gap_is_recorded():
    record = drug_pk.SEARCHED_AND_EMPTY[("fentanyl", "rat", "blood:plasma")]
    assert "blood" in record and "plasma" in record
