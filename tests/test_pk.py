"""Tests for the dose-administered PK layer (P1.1).

The load-bearing test in this file is
`test_iv_bolus_reduces_exactly_to_one_compartment_pk`: the whole point of
the extension is that it ADDS a way in (an administered mg/kg dose) without
moving any existing result, and an IV bolus with F = 1 is the case where the
old and new models are the same differential equation, so they must agree to
machine precision, not merely closely.

Analytic references used below (all textbook closed forms for linear
compartmental models, so the tests check the implementation, not a fitted
number):

  one-compartment IV bolus      C(t) = C0 exp(-ke t)
  one-compartment extravascular C(t) = D0 ka/(ka-ke) (exp(-ke t) - exp(-ka t))
  two-compartment IV bolus      C1(t) = C0[(a-k21)/(a-b) exp(-a t)
                                          + (k21-b)/(a-b) exp(-b t)]
  any of the above              AUC(0->inf) = F Dose / CL
"""

import math

import numpy as np
import pytest
from scipy.integrate import simpson, solve_ivp

from neurocascade.circuit import CircuitNode
from neurocascade.engine import Simulation
from neurocascade.network import CircuitNetwork
from neurocascade.pk import (
    DosedOneCompartmentPK,
    DosedTwoCompartmentPK,
    DosedThreeCompartmentPK,
    OneCompartmentPK,
    clearance_L_per_h_per_kg_to_mL_per_min_per_kg,
    concentration_mg_per_L_to_nM,
    half_life_to_rate_constant,
    intercompartmental_clearance_from_terminal_rate,
    per_hour_to_per_min,
)
from neurocascade.receptor import Ligand, ReceptorSite

# A parameter set chosen so the dose -> concentration arithmetic is exact in
# binary floating point (every operation below is a division by a power of
# two or an exactly representable ratio), which is what lets the reduction
# test assert bit-for-bit equality rather than a tolerance:
#   ke = CL / Vd = 25 / (0.5 * 1000)      = 0.05 min^-1  (== the ke the
#                                            existing scenarios use)
#   C0 = F * dose * BW / (Vd * BW) = 1 * 1 * 0.025 / (0.5 * 0.025) = 2 mg/L
KE_PER_MIN = 0.05
VD_L_PER_KG = 0.5
CL_ML_PER_MIN_PER_KG = 25.0
BODY_WEIGHT_KG = 0.025  # a mouse
DOSE_MG_PER_KG = 1.0
EXPECTED_C0 = 2.0  # mg/L


def integrate_pk(model, dose, t_eval):
    """Integrate one PK model on its own, tightly, straight through the ODE
    protocol the engine uses (`initial_state` + `concentration_rhs`)."""
    y0 = model.initial_state(dose)
    sol = solve_ivp(
        lambda t, s: model.concentration_rhs(s),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-14,
        method="RK45",
    )
    assert sol.success
    return model.concentration(sol.y)


def build_sim(pk_model):
    """Two coupled nodes driven by one agonist, so a PK difference shows up
    in occupancy and firing rate, not only in the concentration trace."""
    a = CircuitNode(name="A", tau=1.0, baseline_input=0.5, gain=4.0)
    b = CircuitNode(name="B", tau=2.0, baseline_input=0.5, gain=4.0)
    network = CircuitNetwork(nodes=[a, b], connectivity=[[0.0, -1.0], [0.5, 0.0]])
    ligand = Ligand(name="agonist", kon=0.1, koff=0.2, efficacy=1.0)
    site = ReceptorSite(name="site", ligands=[ligand])
    return Simulation(
        pk_models=[pk_model],
        receptor_site=site,
        network=network,
        coupling={"A": 2.0},
    )


# ---------------------------------------------------------------------------
# The reduction that must hold exactly
# ---------------------------------------------------------------------------


def test_iv_bolus_reduces_exactly_to_one_compartment_pk():
    legacy = build_sim(OneCompartmentPK(ke=KE_PER_MIN))
    dosed = build_sim(
        DosedOneCompartmentPK(
            route="IV",
            Vd_L_per_kg=VD_L_PER_KG,
            CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
            body_weight_kg=BODY_WEIGHT_KG,
        )
    )

    # The dose-driven model must derive exactly the initial concentration the
    # bolus model is handed directly.
    assert dosed.pk_models[0].dose_equivalent_concentration(DOSE_MG_PER_KG) == EXPECTED_C0
    assert dosed.pk_models[0].ke_per_min == KE_PER_MIN
    # Same number of ODE states, so the two Simulations are the same system.
    assert dosed.d == legacy.d == 1

    t_eval = np.linspace(0, 60, 300)
    legacy_result = legacy.run(
        dose_concentrations=[EXPECTED_C0], t_span=(0, 60), t_eval=t_eval
    )
    dosed_result = dosed.run(doses=[DOSE_MG_PER_KG], t_span=(0, 60), t_eval=t_eval)

    for key in ("concentrations", "occupancy", "rates"):
        assert np.array_equal(legacy_result[key], dosed_result[key]), (
            "%s differs by up to %g"
            % (key, np.max(np.abs(legacy_result[key] - dosed_result[key])))
        )
    # Not a trivial pass on an all-zero trajectory.
    assert dosed_result["concentrations"][0, 0] == EXPECTED_C0
    assert dosed_result["occupancy"].max() > 0.1


def test_iv_bolus_matches_analytic_exponential():
    model = DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    t = np.linspace(0, 120, 200)
    c = integrate_pk(model, DOSE_MG_PER_KG, t)
    analytic = EXPECTED_C0 * np.exp(-KE_PER_MIN * t)
    assert np.max(np.abs(c - analytic)) < 1e-9
    assert model.terminal_half_life_min == pytest.approx(math.log(2) / KE_PER_MIN)
    assert model.tmax_min() == 0.0


# ---------------------------------------------------------------------------
# Dose, bioavailability, body weight
# ---------------------------------------------------------------------------


def test_dose_scales_concentration_linearly_and_zero_dose_means_no_drug():
    model = DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    assert model.dose_equivalent_concentration(0.0) == 0.0
    assert model.dose_equivalent_concentration(2.0) == 2 * model.dose_equivalent_concentration(1.0)
    assert np.array_equal(model.initial_state(0.0), np.zeros(1))


def test_body_weight_cancels_for_per_kg_parameters_but_absolute_amounts_track_it():
    kwargs = dict(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
    )
    mouse = DosedOneCompartmentPK(body_weight_kg=0.025, **kwargs)
    rat = DosedOneCompartmentPK(body_weight_kg=0.30, **kwargs)

    # Same mg/kg dose, same per-kg disposition -> same concentration-time
    # course regardless of body size. This is a property of the allometric
    # parameterization, documented in pk.py, not an accident.
    assert mouse.dose_equivalent_concentration(1.0) == pytest.approx(
        rat.dose_equivalent_concentration(1.0), rel=1e-12
    )
    assert mouse.ke_per_min == rat.ke_per_min
    # Body weight is not ignored: the absolute quantities scale with it.
    assert rat.dose_mg(1.0) == pytest.approx(12.0 * mouse.dose_mg(1.0))
    assert rat.V_central_L == pytest.approx(12.0 * mouse.V_central_L)
    assert rat.CL_L_per_min == pytest.approx(12.0 * mouse.CL_L_per_min)


def test_bioavailability_scales_the_whole_curve():
    def model(F):
        return DosedOneCompartmentPK(
            route="PO",
            Vd_L_per_kg=VD_L_PER_KG,
            CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
            body_weight_kg=BODY_WEIGHT_KG,
            F=F,
            ka_per_min=0.2,
        )

    t = np.linspace(0, 120, 200)
    full = integrate_pk(model(1.0), DOSE_MG_PER_KG, t)
    half = integrate_pk(model(0.5), DOSE_MG_PER_KG, t)
    assert np.allclose(half, 0.5 * full, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# Absorption
# ---------------------------------------------------------------------------


def test_extravascular_absorption_matches_bateman_function():
    ka = 0.2
    model = DosedOneCompartmentPK(
        route="SC",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
        F=0.8,
        ka_per_min=ka,
    )
    assert model.n_states == 2  # central concentration + depot

    t = np.linspace(0, 240, 600)
    c = integrate_pk(model, DOSE_MG_PER_KG, t)

    d0 = model.dose_equivalent_concentration(DOSE_MG_PER_KG)
    ke = model.ke_per_min
    analytic = d0 * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))
    assert np.max(np.abs(c - analytic)) < 1e-8

    # Absorption takes time, so the peak is later than t=0 and lower than the
    # same bioavailable dose given IV.
    assert c[0] == 0.0
    tmax = model.tmax_min()
    assert tmax == pytest.approx(math.log(ka / ke) / (ka - ke))
    assert t[np.argmax(c)] == pytest.approx(tmax, abs=t[1] - t[0])
    assert c.max() < d0


def test_absorption_and_iv_deliver_the_same_auc():
    common = dict(
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    iv = DosedOneCompartmentPK(route="IV", **common)
    po = DosedOneCompartmentPK(route="PO", F=1.0, ka_per_min=0.2, **common)

    # Analytic identity AUC = F*Dose/CL, route-independent.
    assert iv.auc_0_inf(DOSE_MG_PER_KG) == pytest.approx(po.auc_0_inf(DOSE_MG_PER_KG))

    # >> 20 half-lives, so AUC(0..T) ~ AUC(0..inf); Simpson rather than
    # trapezoid because on this grid the trapezoid rule's own O(h^2)
    # discretization error (~2e-6 relative on a decaying exponential) is
    # larger than the agreement being tested for, and that error belongs to
    # the test's reference quadrature, not to the model.
    t = np.linspace(0, 2000, 20001)
    for model in (iv, po):
        auc_numeric = simpson(integrate_pk(model, DOSE_MG_PER_KG, t), x=t)
        assert auc_numeric == pytest.approx(model.auc_0_inf(DOSE_MG_PER_KG), rel=1e-6)


# ---------------------------------------------------------------------------
# Two compartments
# ---------------------------------------------------------------------------


def two_compartment_iv(Q):
    return DosedTwoCompartmentPK(
        route="IV",
        V1_L_per_kg=VD_L_PER_KG,
        V2_L_per_kg=1.5,
        Q_mL_per_min_per_kg=Q,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )


def test_two_compartment_with_no_distribution_reduces_to_one_compartment():
    two = two_compartment_iv(Q=0.0)
    one = DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    t = np.linspace(0, 120, 200)
    assert np.max(np.abs(integrate_pk(two, DOSE_MG_PER_KG, t)
                         - integrate_pk(one, DOSE_MG_PER_KG, t))) < 1e-10
    assert two.terminal_half_life_min == pytest.approx(one.terminal_half_life_min)


def test_two_compartment_iv_matches_analytic_biexponential():
    model = two_compartment_iv(Q=30.0)
    alpha, beta = model.hybrid_rate_constants()
    k21 = model.k21_per_min
    c0 = model.dose_equivalent_concentration(DOSE_MG_PER_KG)

    t = np.linspace(0, 300, 600)
    c = integrate_pk(model, DOSE_MG_PER_KG, t)
    analytic = c0 * (
        (alpha - k21) / (alpha - beta) * np.exp(-alpha * t)
        + (k21 - beta) / (alpha - beta) * np.exp(-beta * t)
    )
    assert np.max(np.abs(c - analytic)) < 1e-9

    # A real distribution phase: the early decline is faster than the
    # terminal one, which is the behavior a one-compartment collapse cannot
    # represent (see pk.py's class docstring).
    assert alpha > beta
    assert model.distribution_half_life_min < model.terminal_half_life_min


def test_hybrid_rate_constants_satisfy_their_defining_identities():
    model = two_compartment_iv(Q=30.0)
    alpha, beta = model.hybrid_rate_constants()
    k10, k12, k21 = model.k10_per_min, model.k12_per_min, model.k21_per_min
    assert alpha + beta == pytest.approx(k10 + k12 + k21)
    assert alpha * beta == pytest.approx(k10 * k21)
    assert model.Vd_ss_L_per_kg == pytest.approx(VD_L_PER_KG + 1.5)


def test_two_compartment_conserves_mass_into_elimination():
    """Drug that has left the system must equal drug eliminated: integrating
    CL*C1 over time recovers the whole bioavailable dose."""
    model = two_compartment_iv(Q=30.0)
    t = np.linspace(0, 5000, 50001)
    c1 = integrate_pk(model, DOSE_MG_PER_KG, t)
    eliminated_mg = model.CL_L_per_min * simpson(c1, x=t)
    assert eliminated_mg == pytest.approx(model.dose_mg(DOSE_MG_PER_KG), rel=1e-6)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_molar_mass_switches_output_to_nanomolar():
    mg_per_L = DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    nM = DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
        molar_mass_g_per_mol=336.47,  # fentanyl free base, PubChem CID 3345
    )
    assert mg_per_L.concentration_unit == "mg/L"
    assert nM.concentration_unit == "nM"
    assert nM.dose_equivalent_concentration(DOSE_MG_PER_KG) == pytest.approx(
        EXPECTED_C0 * 1e6 / 336.47
    )
    # Unit choice must not touch the kinetics.
    assert nM.ke_per_min == mg_per_L.ke_per_min


def test_unit_helpers():
    assert per_hour_to_per_min(60.0) == 1.0
    assert half_life_to_rate_constant(10.0) == pytest.approx(math.log(2) / 10.0)
    # 1 L/h/kg = 1000 mL / 60 min / kg
    assert clearance_L_per_h_per_kg_to_mL_per_min_per_kg(1.0) == pytest.approx(1000.0 / 60.0)
    # 1 mg/L of a 1000 g/mol compound is 1 uM = 1000 nM.
    assert concentration_mg_per_L_to_nM(1.0, 1000.0) == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Refusals: the parameters this model must not invent for you
# ---------------------------------------------------------------------------


def test_extravascular_route_requires_explicit_bioavailability_and_ka():
    common = dict(
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    with pytest.raises(ValueError, match="requires an explicit bioavailability"):
        DosedOneCompartmentPK(route="PO", ka_per_min=0.2, **common)
    with pytest.raises(ValueError, match="ka_per_min"):
        DosedOneCompartmentPK(route="PO", F=0.5, **common)
    with pytest.raises(ValueError, match="F must be in"):
        DosedOneCompartmentPK(route="IP", F=1.5, ka_per_min=0.2, **common)


def test_iv_route_rejects_absorption_parameters():
    common = dict(
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    with pytest.raises(ValueError, match="no absorption phase"):
        DosedOneCompartmentPK(route="IV", ka_per_min=0.2, **common)
    with pytest.raises(ValueError, match="implies F = 1"):
        DosedOneCompartmentPK(route="IV", F=0.4, **common)
    with pytest.raises(ValueError, match="unknown route"):
        DosedOneCompartmentPK(route="intrathecal", **common)


def test_apparent_parameterisation_is_flagged_and_cannot_double_apply_f():
    """`apparent_parameters=True` (added by P1.4 for diazepam's rat IP and PO
    data) is the only way to build an extravascular model without a measured
    F, and it must stay distinguishable from a measured F = 1.

    The distinction is not cosmetic: with apparent parameters the model still
    predicts the right *concentration* (F is inside V/F and CL/F, and cancels
    out of the concentration equation), but its `Vd_L_per_kg` is not a volume
    of distribution and its `CL_mL_per_min_per_kg` is not a clearance, so
    anything reading those as true values is wrong. `F_is_measured` is the
    flag that says which situation you are in.
    """
    common = dict(
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
        ka_per_min=0.2,
    )
    apparent = DosedOneCompartmentPK(
        route="IP", apparent_parameters=True, **common
    )
    assert apparent.apparent_parameters is True
    assert apparent.F == 1.0
    assert apparent.F_is_measured is False
    assert "apparent parameters" in repr(apparent)

    measured = DosedOneCompartmentPK(route="IP", F=1.0, **common)
    assert measured.apparent_parameters is False
    assert measured.F_is_measured is True
    assert "apparent" not in repr(measured)

    # Identical curves: the two differ only in what they claim, which is
    # exactly why the flag has to exist.
    t = np.linspace(0.0, 300.0, 601)
    np.testing.assert_allclose(
        integrate_pk(apparent, 1.0, t), integrate_pk(measured, 1.0, t), rtol=1e-12
    )

    # F must not be supplied on top of parameters that already contain it.
    with pytest.raises(ValueError, match="must not be given again"):
        DosedOneCompartmentPK(
            route="IP", F=0.5, apparent_parameters=True, **common
        )
    # ka is still required: absorption is a separate, unrelated gap.
    with pytest.raises(ValueError, match="ka_per_min"):
        DosedOneCompartmentPK(
            route="IP",
            apparent_parameters=True,
            Vd_L_per_kg=VD_L_PER_KG,
            CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
            body_weight_kg=BODY_WEIGHT_KG,
        )
    # And it is meaningless intravenously, where F is 1 by definition.
    with pytest.raises(ValueError, match="meaningless for route 'IV'"):
        DosedOneCompartmentPK(
            route="IV",
            apparent_parameters=True,
            Vd_L_per_kg=VD_L_PER_KG,
            CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
            body_weight_kg=BODY_WEIGHT_KG,
        )


def test_negative_dose_is_rejected():
    model = DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=VD_L_PER_KG,
        CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        body_weight_kg=BODY_WEIGHT_KG,
    )
    with pytest.raises(ValueError, match="non-negative"):
        model.initial_state(-1.0)


# ---------------------------------------------------------------------------
# Engine integration: the PK layer stays inside the coupled state vector
# ---------------------------------------------------------------------------


def mixed_pk_simulation():
    """One receptor site, two competing ligands whose PK models own
    different numbers of states: an SC one-compartment agonist (central +
    depot) and an IV two-compartment antagonist (central + peripheral)."""
    a = CircuitNode(name="A", tau=1.0, baseline_input=0.5, gain=4.0)
    b = CircuitNode(name="B", tau=2.0, baseline_input=0.5, gain=4.0)
    network = CircuitNetwork(nodes=[a, b], connectivity=[[0.0, -1.0], [0.5, 0.0]])
    site = ReceptorSite(
        name="site",
        ligands=[
            Ligand(name="agonist", kon=0.1, koff=0.2, efficacy=1.0),
            Ligand(name="antagonist", kon=0.5, koff=1.0, efficacy=0.0),
        ],
    )
    pk_models = [
        DosedOneCompartmentPK(
            route="SC",
            Vd_L_per_kg=VD_L_PER_KG,
            CL_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
            body_weight_kg=BODY_WEIGHT_KG,
            F=0.9,
            ka_per_min=0.15,
        ),
        two_compartment_iv(Q=30.0),
    ]
    return Simulation(
        pk_models=pk_models,
        receptor_site=site,
        network=network,
        coupling={"A": 2.0},
    )


def test_multi_state_pk_models_are_integrated_in_the_shared_state_vector():
    sim = mixed_pk_simulation()
    # 2 states (central + depot) + 2 states (central + peripheral) = 4 PK
    # states for 2 ligands: the PK region is no longer one row per ligand.
    assert sim.d == 4
    assert sim.r == 2
    assert list(sim.concentration_rows) == [0, 2]

    t_eval = np.linspace(0, 120, 200)
    result = sim.run(doses=sim.dose_vector(agonist=1.0), t_span=(0, 120), t_eval=t_eval)

    assert result["concentrations"].shape == (2, 200)
    assert result["pk_states"].shape == (4, 200)
    assert result["occupancy"].shape == (2, 200)
    assert result["rates"].shape == (2, 200)

    # rhs() must return a derivative for every state in the vector.
    y0 = np.concatenate(
        [sim.initial_pk_state([1.0, 0.0]), np.zeros(sim.r), np.zeros(sim.c)]
    )
    assert sim.rhs(0.0, y0).shape == y0.shape

    # The agonist's depot (PK state row 1) starts full and empties; its
    # plasma concentration starts empty and peaks later.
    depot = result["pk_states"][1]
    assert depot[0] > 0.0
    assert np.all(np.diff(depot) < 0.0)
    assert depot[-1] < 0.01 * depot[0]
    agonist_plasma = result["concentrations"][0]
    assert agonist_plasma[0] == 0.0
    assert np.argmax(agonist_plasma) > 0

    # The undosed antagonist stays at exactly zero in both of its states, and
    # the dosed agonist actually drives the circuit.
    assert np.all(result["pk_states"][2:] == 0.0)
    assert result["occupancy"][0].max() > 0.05
    assert result["occupancy"][1].max() == 0.0


def test_simulated_concentration_matches_the_standalone_pk_solution():
    """The engine must be integrating the PK model itself, not applying some
    separately-computed trace: the coupled run's concentration column has to
    agree with the same model integrated alone."""
    sim = mixed_pk_simulation()
    t_eval = np.linspace(0, 120, 200)
    result = sim.run(doses=[1.0, 0.5], t_span=(0, 120), t_eval=t_eval)

    for i, dose in enumerate([1.0, 0.5]):
        standalone = integrate_pk(sim.pk_models[i], dose, t_eval)
        assert np.allclose(result["concentrations"][i], standalone, rtol=1e-5, atol=1e-9)


def test_add_dose_puts_a_later_extravascular_dose_in_the_depot():
    sim = mixed_pk_simulation()
    y = np.zeros(sim.d + sim.r + sim.c)
    y2 = sim.add_dose(y, sim.dose_vector(agonist=1.0, antagonist=2.0))

    agonist_pk = sim.pk_models[0]
    # Agonist is SC: the new dose lands in the depot (row 1), not plasma.
    assert y2[0] == 0.0
    assert y2[1] == agonist_pk.dose_equivalent_concentration(1.0)
    # Antagonist is IV: straight into the central compartment (row 2).
    assert y2[2] == sim.pk_models[1].dose_equivalent_concentration(2.0)
    assert y2[3] == 0.0
    # A second administration superposes on whatever is already on board.
    y3 = sim.add_dose(y2, sim.dose_vector(agonist=1.0))
    assert y3[1] == pytest.approx(2 * y2[1])
    assert y is not y2  # not modified in place


def test_zero_dose_on_a_dosed_model_is_identical_to_no_drug():
    sim = mixed_pk_simulation()
    t_eval = np.linspace(0, 60, 120)
    baseline = sim.run(doses=[0.0, 0.0], t_span=(0, 60), t_eval=t_eval)
    assert np.all(baseline["pk_states"] == 0.0)
    assert np.all(baseline["occupancy"] == 0.0)


def test_run_rejects_ambiguous_or_missing_dose_arguments():
    sim = build_sim(OneCompartmentPK(ke=KE_PER_MIN))
    with pytest.raises(TypeError, match="not both"):
        sim.run(dose_concentrations=[1.0], doses=[1.0], t_span=(0, 1))
    with pytest.raises(TypeError, match="requires doses"):
        sim.run(t_span=(0, 1))
    with pytest.raises(ValueError, match="length 1"):
        sim.run(doses=[1.0, 2.0], t_span=(0, 1))


def test_legacy_rhs_only_pk_model_still_works_inside_the_engine():
    """A duck-typed PK object written against the pre-P1.1 interface (just
    `rhs(C)`) must keep integrating, since `Simulation` accepts any object
    with the documented PK attributes."""

    class LegacyDuck(object):
        def __init__(self, ke):
            self.ke = ke

        def rhs(self, C):
            return -self.ke * C

    duck = build_sim(LegacyDuck(KE_PER_MIN))
    real = build_sim(OneCompartmentPK(KE_PER_MIN))
    t_eval = np.linspace(0, 60, 100)
    a = duck.run(dose_concentrations=[EXPECTED_C0], t_span=(0, 60), t_eval=t_eval)
    b = real.run(dose_concentrations=[EXPECTED_C0], t_span=(0, 60), t_eval=t_eval)
    assert np.array_equal(a["rates"], b["rates"])


# ---------------------------------------------------------------------------
# Solving for Q from a published (V1, V2, CL, terminal half-life) table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "V1,V2,CL,Q",
    [
        (5.5, 4.1, 83.0, 114.0),   # haloperidol-in-rat scale (P1.3)
        (0.5, 1.5, 25.0, 10.0),    # a small-volume, slow-exchange case
        (2.0, 2.0, 40.0, 200.0),   # exchange much faster than elimination
    ],
)
def test_intercompartmental_clearance_solves_back_to_the_model_it_came_from(
    V1, V2, CL, Q
):
    """Round-trip: build a two-compartment model with a known Q, read off its
    terminal rate constant, and check the inversion recovers Q.

    This is the arithmetic that lets P1.2-P1.4 build two-compartment models
    from published tables, which report V1/Vss/CL/half-life and essentially
    never report Q, so it has to be exactly right rather than approximately.
    """
    model = DosedTwoCompartmentPK(
        route="IV",
        V1_L_per_kg=V1,
        V2_L_per_kg=V2,
        Q_mL_per_min_per_kg=Q,
        CL_mL_per_min_per_kg=CL,
        body_weight_kg=0.25,
    )
    _alpha, beta = model.hybrid_rate_constants()
    recovered = intercompartmental_clearance_from_terminal_rate(
        V1_L_per_kg=V1, V2_L_per_kg=V2, CL_mL_per_min_per_kg=CL, beta_per_min=beta
    )
    assert recovered == pytest.approx(Q, rel=1e-9)


def test_intercompartmental_clearance_refuses_impossible_terminal_rates():
    kwargs = dict(V1_L_per_kg=5.5, V2_L_per_kg=4.1, CL_mL_per_min_per_kg=83.0)
    k10 = 83.0 / 5500.0
    # A terminal phase faster than elimination from the central compartment
    # is not reachable by any Q: the model cannot have been two-compartment.
    with pytest.raises(ValueError, match="must be smaller than"):
        intercompartmental_clearance_from_terminal_rate(
            beta_per_min=k10 * 1.5, **kwargs
        )
    with pytest.raises(ValueError, match="must be positive"):
        intercompartmental_clearance_from_terminal_rate(beta_per_min=0.0, **kwargs)
    # A beta just barely below k10 needs an unbounded Q; past the point where
    # the denominator turns over there is no positive solution at all.
    with pytest.raises(ValueError, match="not consistent"):
        intercompartmental_clearance_from_terminal_rate(
            beta_per_min=k10 * 0.999, **kwargs
        )
