"""Tests for the literature PK parameter sets (P1.3: haloperidol, rat).

Two different kinds of assertion live here and they are deliberately kept
apart:

* **Transcription checks.** Constants must equal what the cited paper says,
  so that a later edit that "tidies" 83 mL/min/kg into something rounder
  fails loudly. These are checks against `docs/architecture.md`'s quoted
  abstract, not against the model.
* **Behavioural checks.** The *simulated* concentration-time course, produced
  by integrating the model through the same ODE protocol the engine uses,
  must reproduce the source's reported terminal half-life, its reported
  V_beta and Vss, and its reported dose-linearity. These are the ones that
  would catch a mis-scaled unit conversion, which is the failure mode
  `pk.py`'s module docstring calls out as indistinguishable from an invented
  number.

The half-life tolerance is not a chosen percentage: [ChengPaalzow1992]
reports "1.5 h" to two significant figures, so the interval it constrains
the true value to is 1.45-1.55 h, and that interval is the tolerance
(`HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN`). The model lands at
92.7 min, i.e. 1.545 h, inside it; the test also prints the value so the
real error is visible rather than hidden by a pass.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade.legacy import drug_pk
from neurocascade.circuit import CircuitNode
from neurocascade.legacy.drug_pk import (
    HALOPERIDOL_MOLAR_MASS_G_PER_MOL,
    HALOPERIDOL_RAT_IV_BETA_PER_MIN,
    HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG,
    HALOPERIDOL_RAT_IV_F,
    HALOPERIDOL_RAT_IV_Q_ML_PER_MIN_PER_KG,
    HALOPERIDOL_RAT_IV_Q_SENSITIVITY_ML_PER_MIN_PER_KG,
    HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN,
    HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_MIN,
    HALOPERIDOL_RAT_IV_V2_L_PER_KG,
    HALOPERIDOL_RAT_IV_VBETA_L_PER_KG,
    HALOPERIDOL_RAT_IV_VC_L_PER_KG,
    HALOPERIDOL_RAT_IV_VSS_L_PER_KG,
    UnsourcedParameterError,
    haloperidol_rat_iv_one_compartment_pk,
    haloperidol_rat_iv_pk,
    haloperidol_rat_pk,
)
from neurocascade.engine import Simulation
from neurocascade.network import CircuitNetwork
from neurocascade.pk import LN2, concentration_mg_per_L_to_nM
from neurocascade.receptor import Ligand, ReceptorSite

BODY_WEIGHT_KG = 0.25  # a rat; per-kg parameters make the value immaterial
DOSE_MG_PER_KG = 1.0  # inside [ChengPaalzow1992]'s 0.5-2.5 mg/kg IV range


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


def simulated_terminal_half_life_min(model, dose, window_min=(300.0, 600.0)):
    """Half-life read off a *simulated* curve, the way it is read off a real
    one: log-linear regression on the terminal portion.

    The window starts well after the distribution phase (whose half-life is
    ~12 min here), so what is fitted is the terminal slope beta and not a
    mixture of both phases.
    """
    t = np.linspace(0.0, window_min[1], 1200)
    c = integrate(model, dose, t)
    assert np.all(c > 0.0)
    terminal = t >= window_min[0]
    slope = np.polyfit(t[terminal], np.log(c[terminal]), 1)[0]
    return LN2 / (-slope)


# ---------------------------------------------------------------------------
# Transcription: the constants are what the paper says
# ---------------------------------------------------------------------------


def test_reported_rat_iv_parameters_match_the_cited_abstract():
    # Cheng YF, Paalzow LK, Biopharm Drug Dispos 1992;13(1):69-76,
    # PMID 1554878: "total blood clearance averaging 83 ml min-1 kg-1",
    # "5.5 l kg-1 (Vc), 11.1 l kg-1 (V beta), and 9.6 l kg-1 (Vss)",
    # "The terminal half-life was 1.5 h."
    assert HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG == 83.0
    assert HALOPERIDOL_RAT_IV_VC_L_PER_KG == 5.5
    assert HALOPERIDOL_RAT_IV_VBETA_L_PER_KG == 11.1
    assert HALOPERIDOL_RAT_IV_VSS_L_PER_KG == 9.6
    assert HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_MIN == 90.0
    # F for an intravenous bolus is 1 by definition, not by measurement.
    assert HALOPERIDOL_RAT_IV_F == 1.0
    # Molar mass is a chemical fact (PubChem CID 3559) and must stay
    # identical to the one scenarios.py already uses for the Kapur 1997
    # occupancy conversion, or the two will silently disagree.
    assert HALOPERIDOL_MOLAR_MASS_G_PER_MOL == 375.86


def test_derived_quantities_follow_from_the_reported_ones():
    # V2 = Vss - Vc, by the definition of Vss for a two-compartment model.
    assert HALOPERIDOL_RAT_IV_V2_L_PER_KG == pytest.approx(9.6 - 5.5)
    # beta = CL / V_beta, by the definition of V_beta.
    assert HALOPERIDOL_RAT_IV_BETA_PER_MIN == pytest.approx(83.0 / 11100.0)
    # Q is derived, not reported, and the derivation is ill-conditioned: the
    # two self-consistent ways of reading the same paper differ by ~24%.
    q_from_vbeta, q_from_half_life = HALOPERIDOL_RAT_IV_Q_SENSITIVITY_ML_PER_MIN_PER_KG
    assert q_from_vbeta == HALOPERIDOL_RAT_IV_Q_ML_PER_MIN_PER_KG
    assert q_from_vbeta == pytest.approx(114.5, abs=0.5)
    assert q_from_half_life == pytest.approx(141.6, abs=0.5)
    spread = abs(q_from_half_life - q_from_vbeta) / q_from_vbeta
    assert 0.20 < spread < 0.30, (
        "the documented sensitivity of the derived Q is ~24%%; got %.1f%%"
        % (100 * spread)
    )


def test_high_extraction_claim_is_checkable_not_asserted():
    cl, hepatic_blood_flow = drug_pk.hepatic_extraction_is_high_note()
    # [ChengPaalzow1992] calls haloperidol "a high extraction ratio drug";
    # the measured blood clearance exceeds rat hepatic blood flow
    # (~55 mL/min/kg, [DaviesMorris1993]), which is the documented reason no
    # extravascular F is assumed here and the reason elimination cannot be
    # purely hepatic.
    assert cl > hepatic_blood_flow


# ---------------------------------------------------------------------------
# Behaviour: the simulated curve reproduces the source
# ---------------------------------------------------------------------------


def test_simulated_terminal_half_life_matches_the_cited_half_life():
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG)
    simulated = simulated_terminal_half_life_min(model, DOSE_MG_PER_KG)
    low, high = HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN
    assert low <= simulated <= high, (
        "simulated terminal half-life %.2f min (%.3f h) is outside the "
        "1.45-1.55 h interval that Cheng & Paalzow 1992's reported '1.5 h' "
        "constrains" % (simulated, simulated / 60.0)
    )
    # The actual value, stated rather than merely bounded: 92.70 min.
    assert simulated == pytest.approx(92.70, abs=0.05)
    # And the closed-form property the model exposes agrees with the curve.
    assert model.terminal_half_life_min == pytest.approx(simulated, rel=1e-4)


def test_one_compartment_collapse_has_the_same_terminal_half_life():
    model = haloperidol_rat_iv_one_compartment_pk(BODY_WEIGHT_KG)
    simulated = simulated_terminal_half_life_min(model, DOSE_MG_PER_KG)
    low, high = HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN
    assert low <= simulated <= high
    # Using Vd = V_beta (not Vc) is what makes the collapse preserve the
    # terminal phase; this is the whole reason for that choice.
    assert simulated == pytest.approx(
        haloperidol_rat_iv_pk(BODY_WEIGHT_KG).terminal_half_life_min, rel=1e-3
    )


def test_model_reproduces_reported_vss_and_vbeta():
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG)
    assert model.Vd_ss_L_per_kg == pytest.approx(HALOPERIDOL_RAT_IV_VSS_L_PER_KG)
    # V_beta = CL / beta is a derived property of the fitted disposition, so
    # recovering 11.1 L/kg from the model's own hybrid rate constants is a
    # real round-trip through `hybrid_rate_constants`, not a restatement.
    _alpha, beta = model.hybrid_rate_constants()
    v_beta = HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG / (beta * 1000.0)
    assert v_beta == pytest.approx(HALOPERIDOL_RAT_IV_VBETA_L_PER_KG, rel=1e-6)


def test_two_compartment_model_is_genuinely_biphasic():
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG)
    # [ChengPaalzow1992] describes biexponential disposition; a model whose
    # two phases had collapsed together would still pass the terminal
    # half-life test above while misrepresenting the first hour entirely.
    assert model.distribution_half_life_min < 0.25 * model.terminal_half_life_min
    assert model.distribution_half_life_min == pytest.approx(12.3, abs=0.2)


def test_initial_concentration_uses_the_central_volume():
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    # An IV bolus starts at Dose/Vc, not Dose/Vss or Dose/V_beta.
    assert model.initial_state(DOSE_MG_PER_KG)[0] == pytest.approx(
        DOSE_MG_PER_KG / HALOPERIDOL_RAT_IV_VC_L_PER_KG
    )
    collapsed = haloperidol_rat_iv_one_compartment_pk(
        BODY_WEIGHT_KG, molar_mass_g_per_mol=None
    )
    # ...and the one-compartment collapse understates that peak ~2-fold,
    # exactly as its docstring says. Documented cost, asserted so it stays
    # documented.
    ratio = model.initial_state(DOSE_MG_PER_KG)[0] / collapsed.initial_state(
        DOSE_MG_PER_KG
    )[0]
    assert ratio == pytest.approx(
        HALOPERIDOL_RAT_IV_VBETA_L_PER_KG / HALOPERIDOL_RAT_IV_VC_L_PER_KG
    )
    assert 1.9 < ratio < 2.1


def test_disposition_is_linear_over_the_dose_range_the_source_studied():
    # [ChengPaalzow1992]: "The disposition kinetics of haloperidol was found
    # to be linear over the dose range studied" (0.5, 1.0, 2.5 mg/kg IV).
    # A linear model cannot fail this, which is the point: it confirms this
    # codebase represents the source's finding rather than contradicting it,
    # and it is the check that would fail first if a future saturable-
    # elimination term were added without revisiting this parameter set.
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG)
    t = np.linspace(0.0, 600.0, 1200)
    reference = integrate(model, 0.5, t)
    for dose in (1.0, 2.5):
        scaled = integrate(model, dose, t)
        assert np.allclose(scaled, (dose / 0.5) * reference, rtol=1e-6, atol=1e-9)
        assert simulated_terminal_half_life_min(model, dose) == pytest.approx(
            simulated_terminal_half_life_min(model, 0.5), rel=1e-6
        )


def test_auc_is_dose_over_clearance_and_route_independent():
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    expected = DOSE_MG_PER_KG * BODY_WEIGHT_KG / model.CL_L_per_min
    assert model.auc_0_inf(DOSE_MG_PER_KG) == pytest.approx(expected)
    # The one-compartment collapse keeps total exposure identical (AUC
    # depends only on CL), which is the other half of the trade it makes.
    collapsed = haloperidol_rat_iv_one_compartment_pk(
        BODY_WEIGHT_KG, molar_mass_g_per_mol=None
    )
    assert collapsed.auc_0_inf(DOSE_MG_PER_KG) == pytest.approx(expected)


def test_concentrations_are_reported_in_nM_by_default():
    nM_model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG)
    mgL_model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG, molar_mass_g_per_mol=None)
    assert nM_model.concentration_unit == "nM"
    assert mgL_model.concentration_unit == "mg/L"
    c0_mg_per_L = mgL_model.initial_state(DOSE_MG_PER_KG)[0]
    assert nM_model.initial_state(DOSE_MG_PER_KG)[0] == pytest.approx(
        concentration_mg_per_L_to_nM(c0_mg_per_L, HALOPERIDOL_MOLAR_MASS_G_PER_MOL)
    )
    # Sanity on magnitude: 1 mg/kg IV into 5.5 L/kg is 0.18 mg/L, ~484 nM.
    assert nM_model.initial_state(DOSE_MG_PER_KG)[0] == pytest.approx(483.7, abs=1.0)


# ---------------------------------------------------------------------------
# Inside the coupled system
# ---------------------------------------------------------------------------


def test_parameter_set_runs_inside_the_coupled_simulation():
    """The model must work as a PK block of `Simulation`'s state vector, not
    only standalone: that is the whole reason P1.1 kept PK inside the ODE."""
    node_a = CircuitNode(name="A", tau=1.0, baseline_input=0.5, gain=4.0)
    node_b = CircuitNode(name="B", tau=2.0, baseline_input=0.5, gain=4.0)
    network = CircuitNetwork(
        nodes=[node_a, node_b], connectivity=[[0.0, -1.0], [0.5, 0.0]]
    )
    # kon/koff are the calibrated haloperidol D2 constants from
    # scenarios.d2_dopamine_ligands_calibrated ([Sykes2017]); efficacy 0 for
    # an antagonist. Reused, not re-derived.
    site = ReceptorSite(
        name="D2",
        ligands=[Ligand(name="haloperidol", kon=2.13, koff=0.65, efficacy=0.0)],
    )
    model = haloperidol_rat_iv_pk(BODY_WEIGHT_KG)
    sim = Simulation(
        pk_models=[model], receptor_site=site, network=network, coupling={"A": 2.0}
    )
    t_eval = np.linspace(0.0, 600.0, 1200)
    result = sim.run(doses=[DOSE_MG_PER_KG], t_span=(0.0, 600.0), t_eval=t_eval)

    concentrations = result["concentrations"][0]
    # Two PK states (central + peripheral) for one ligand, and the reported
    # concentration is the central one.
    assert result["pk_states"].shape[0] == 2
    assert concentrations[0] == pytest.approx(model.initial_state(DOSE_MG_PER_KG)[0])
    # The coupled integration must give the same trace as the standalone one.
    standalone = integrate(model, DOSE_MG_PER_KG, t_eval)
    assert np.allclose(concentrations, standalone, rtol=1e-5, atol=1e-8)
    # An antagonist at ~480 nM against a 2.13 nM^-1 min^-1 / 0.65 min^-1
    # ligand occupies essentially the whole pool early and lets go as the
    # drug clears; a flat occupancy trace would mean the PK is not reaching
    # the receptor layer at all.
    occupancy = result["occupancy"][0]
    assert occupancy.max() > 0.5
    assert occupancy[-1] < occupancy.max()


# ---------------------------------------------------------------------------
# The routes that have no numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["IP", "SC"])
def test_unsourced_routes_raise_with_their_search_record(route):
    with pytest.raises(UnsourcedParameterError) as excinfo:
        haloperidol_rat_pk(route, BODY_WEIGHT_KG)
    message = str(excinfo.value)
    # The error IS the searched-and-empty record: it has to say what was
    # searched and why the candidates failed, so that a caller who hits it
    # can either supply a citation or stop.
    assert route in message
    assert "Searched" in message
    assert "Candidates found and why each is unusable" in message
    assert drug_pk.SEARCHED_AND_EMPTY[("haloperidol", "rat", route)] in message


def test_unknown_route_also_refuses_rather_than_defaulting():
    with pytest.raises(UnsourcedParameterError):
        haloperidol_rat_pk("PO", BODY_WEIGHT_KG)


def test_no_bioavailability_or_absorption_constant_is_defined_for_any_route():
    """Guard against the single most likely way this file could go wrong
    later: someone adding `HALOPERIDOL_RAT_SC_F = 1.0` because SC bypasses
    first pass. It does bypass first pass; that still is not a measurement.
    """
    for name in dir(drug_pk):
        if name.startswith("HALOPERIDOL_RAT_") and ("_F" == name[-2:] or "KA" in name):
            assert name == "HALOPERIDOL_RAT_IV_F", (
                "%s defines an extravascular F or ka; P1.3 recorded both as "
                "searched-and-empty for rat IP and SC, so adding one needs a "
                "citation and a docs/architecture.md update in the same "
                "commit" % name
            )


def test_blood_plasma_gap_is_recorded():
    record = drug_pk.SEARCHED_AND_EMPTY[("haloperidol", "rat", "blood:plasma")]
    # The parameter set is blood-based; the codebase's existing occupancy
    # anchor (Kapur 1997) is plasma-based. That mismatch is unresolved and
    # must stay written down.
    assert "blood" in record and "plasma" in record
