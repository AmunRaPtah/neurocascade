"""Tests for the administered-dose PK layer (roadmap P1.1).

Every numeric parameter used here is a synthetic, deliberately round
placeholder chosen to make an analytic check exact or a reduction visible.
None of them is offered as a literature value for any real drug: the
literature parameter sets for fentanyl, haloperidol and diazepam are
roadmap P1.2-P1.4 and belong in `scenarios.py` with their citations.
"""

import numpy as np
import pytest

from neurocascade.circuit import CircuitNode
from neurocascade.engine import Simulation
from neurocascade.network import CircuitNetwork
from neurocascade.pk import (
    OneCompartmentDosePK,
    OneCompartmentPK,
    TwoCompartmentDosePK,
    _dose_nmol,
)
from neurocascade.receptor import Ligand, ReceptorSite


# --- synthetic, illustrative-only parameter block -------------------------
# Round numbers, not any drug's measured PK.
WEIGHT_KG = 0.25
MOLAR_MASS = 300.0  # g/mol
VD_L_PER_KG = 5.0
CL_ML_PER_MIN_PER_KG = 250.0  # -> ke = CL/Vd = 0.05 min^-1
DOSE_MG_PER_KG = 0.1


def _dose_model(**overrides):
    kwargs = dict(
        dose_mg_per_kg=DOSE_MG_PER_KG,
        body_weight_kg=WEIGHT_KG,
        molar_mass_g_per_mol=MOLAR_MASS,
        volume_of_distribution_L_per_kg=VD_L_PER_KG,
        clearance_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        route="iv",
    )
    kwargs.update(overrides)
    return OneCompartmentDosePK(**kwargs)


def _single_ligand_sim(pk_model):
    """A 2-node circuit driven by one agonist, used to check that the PK
    layer really is integrated inside `Simulation.rhs` rather than
    precomputed: the circuit rates can only be right if the concentration
    trace advanced in lockstep with occupancy.
    """
    a = CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0)
    b = CircuitNode(name="B", tau=2.0, baseline_input=0.5, gain=4.0)
    network = CircuitNetwork(nodes=[a, b], connectivity=[[0.0, -1.0], [0.5, 0.0]])
    site = ReceptorSite(
        name="site", ligands=[Ligand(name="agonist", kon=0.1, koff=0.05, efficacy=1.0)]
    )
    return Simulation(
        pk_models=[pk_model],
        receptor_site=site,
        network=network,
        coupling={"A": 1.5},
    )


# --- unit conversion ------------------------------------------------------


def test_dose_conversion_to_nmol():
    # 0.1 mg/kg * 0.25 kg = 0.025 mg; 0.025 mg / 300 mg/mmol = 8.3333e-5 mmol
    # = 83.333 nmol.
    assert _dose_nmol(0.1, 0.25, 300.0) == pytest.approx(0.025 / 300.0 * 1e6)
    assert _dose_nmol(0.1, 0.25, 300.0) == pytest.approx(83.3333333, rel=1e-6)


def test_iv_initial_concentration_is_dose_over_volume():
    pk = _dose_model()
    C0, aux0 = pk.initial_state()
    assert aux0.shape == (0,)
    assert C0 == pytest.approx(_dose_nmol(DOSE_MG_PER_KG, WEIGHT_KG, MOLAR_MASS) / 1.25)
    assert pk.V == pytest.approx(1.25)  # 5 L/kg * 0.25 kg
    assert pk.ke == pytest.approx(0.05)  # (250 mL/min/kg / 1000) * 0.25 / 1.25 L


def test_bioavailability_scales_the_amount_that_ever_enters_the_body():
    full = _dose_model(route="sc", bioavailability=1.0, ka_per_min=0.2)
    half = _dose_model(route="sc", bioavailability=0.5, ka_per_min=0.2)
    assert half.bioavailable_nmol == pytest.approx(0.5 * full.bioavailable_nmol)
    assert half.initial_state()[1][0] == pytest.approx(0.5 * full.initial_state()[1][0])


# --- the headline reduction: IV bolus == OneCompartmentPK -----------------


def test_iv_bolus_rhs_is_identical_to_one_compartment_pk():
    dose_pk = _dose_model()
    legacy = OneCompartmentPK(ke=dose_pk.ke)
    for C in (0.0, 1.0, 37.5, 1234.0):
        assert dose_pk.concentration_rhs(C)[0] == legacy.rhs(C)
        assert dose_pk.concentration_rhs(C)[1].shape == (0,)
    assert dose_pk.n_aux == 0


def test_iv_bolus_simulation_reduces_exactly_to_one_compartment_pk():
    """The acceptance criterion for P1.1: routing a dose through the new
    model must not move any existing result. The reference simulation is
    given the *derived* initial concentration and elimination constant, so
    the two integrations see the identical y0 and the identical RHS and
    must agree to machine precision, not merely closely.
    """
    dose_pk = _dose_model()
    C0 = dose_pk.initial_state()[0]
    legacy_pk = OneCompartmentPK(ke=dose_pk.ke)

    dose_sim = _single_ligand_sim(dose_pk)
    legacy_sim = _single_ligand_sim(legacy_pk)

    t_eval = np.linspace(0, 60, 200)
    dosed = dose_sim.run(t_span=(0, 60), t_eval=t_eval)
    reference = legacy_sim.run(
        dose_concentrations=[C0], t_span=(0, 60), t_eval=t_eval
    )

    assert dose_sim.a == 0  # no extra state at all on the IV route
    for key in ("concentrations", "occupancy", "rates"):
        np.testing.assert_allclose(
            dosed[key], reference[key], rtol=1e-13, atol=1e-13, err_msg=key
        )
    # Same equations, same initial condition, same solver: bit-for-bit.
    assert np.array_equal(dosed["rates"], reference["rates"])


def test_iv_bolus_matches_analytic_monoexponential_decay():
    pk = _dose_model()
    sim = _single_ligand_sim(pk)
    t_eval = np.linspace(0, 60, 61)
    result = sim.run(t_span=(0, 60), t_eval=t_eval)
    expected = pk.initial_state()[0] * np.exp(-pk.ke * t_eval)
    np.testing.assert_allclose(result["concentrations"][0], expected, rtol=1e-6)


def test_two_compartment_iv_with_zero_q_reduces_to_one_compartment():
    one = _dose_model()
    two = TwoCompartmentDosePK(
        dose_mg_per_kg=DOSE_MG_PER_KG,
        body_weight_kg=WEIGHT_KG,
        molar_mass_g_per_mol=MOLAR_MASS,
        central_volume_L_per_kg=VD_L_PER_KG,
        peripheral_volume_L_per_kg=2.0,  # irrelevant when Q = 0
        clearance_mL_per_min_per_kg=CL_ML_PER_MIN_PER_KG,
        intercompartmental_clearance_mL_per_min_per_kg=0.0,
        route="iv",
    )
    t_eval = np.linspace(0, 60, 200)
    one_result = _single_ligand_sim(one).run(t_span=(0, 60), t_eval=t_eval)
    two_result = _single_ligand_sim(two).run(t_span=(0, 60), t_eval=t_eval)
    np.testing.assert_allclose(
        two_result["concentrations"], one_result["concentrations"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        two_result["rates"], one_result["rates"], rtol=1e-10, atol=1e-12
    )
    # The peripheral compartment exists as a state but never fills.
    np.testing.assert_allclose(two_result["pk_aux"][0], 0.0, atol=1e-12)


# --- absorption phase -----------------------------------------------------


@pytest.mark.parametrize("route", ["ip", "sc", "po"])
def test_extravascular_routes_start_empty_and_peak_later(route):
    pk = _dose_model(route=route, ka_per_min=0.2, bioavailability=0.8)
    assert pk.n_aux == 1
    sim = _single_ligand_sim(pk)
    t_eval = np.linspace(0, 120, 601)
    result = sim.run(t_span=(0, 120), t_eval=t_eval)
    C = result["concentrations"][0]

    assert C[0] == 0.0  # nothing in plasma at the instant of injection
    tmax = t_eval[int(np.argmax(C))]
    assert tmax > 0.0
    # Flip-flop-free case (ka > ke): the analytic Tmax of a one-compartment
    # first-order-absorption model is ln(ka/ke)/(ka - ke).
    expected_tmax = np.log(pk.ka / pk.ke) / (pk.ka - pk.ke)
    assert tmax == pytest.approx(expected_tmax, rel=1e-2)
    # Depot empties monotonically.
    depot = result["pk_aux"][0]
    assert depot[0] == pytest.approx(pk.bioavailable_nmol)
    assert np.all(np.diff(depot) <= 1e-12)


def test_extravascular_concentration_matches_bateman_function():
    """Analytic check on the absorption model itself: with a one-
    compartment body and first-order absorption, C(t) is the Bateman
    function F*D*ka/(V*(ka-ke)) * (exp(-ke t) - exp(-ka t)).
    """
    pk = _dose_model(route="ip", ka_per_min=0.2, bioavailability=0.6)
    sim = _single_ligand_sim(pk)
    t_eval = np.linspace(0, 120, 241)
    C = sim.run(t_span=(0, 120), t_eval=t_eval)["concentrations"][0]
    prefactor = pk.bioavailable_nmol * pk.ka / (pk.V * (pk.ka - pk.ke))
    expected = prefactor * (np.exp(-pk.ke * t_eval) - np.exp(-pk.ka * t_eval))
    np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-7)


def test_lower_bioavailability_lowers_exposure_proportionally():
    full = _dose_model(route="po", ka_per_min=0.2, bioavailability=1.0)
    partial = _dose_model(route="po", ka_per_min=0.2, bioavailability=0.25)
    t_eval = np.linspace(0, 200, 401)
    C_full = _single_ligand_sim(full).run(t_span=(0, 200), t_eval=t_eval)[
        "concentrations"
    ][0]
    C_partial = _single_ligand_sim(partial).run(t_span=(0, 200), t_eval=t_eval)[
        "concentrations"
    ][0]
    # A linear model: a quarter of the bioavailable amount gives a quarter
    # of the concentration at every time point, not merely a lower peak.
    np.testing.assert_allclose(C_partial, 0.25 * C_full, rtol=1e-6, atol=1e-9)


def test_dose_scales_concentration_linearly():
    low = _dose_model(dose_mg_per_kg=0.1)
    high = _dose_model(dose_mg_per_kg=1.0)
    assert high.initial_state()[0] == pytest.approx(10.0 * low.initial_state()[0])


# --- two-compartment distribution behaviour -------------------------------


def _two_compartment(**overrides):
    kwargs = dict(
        dose_mg_per_kg=DOSE_MG_PER_KG,
        body_weight_kg=WEIGHT_KG,
        molar_mass_g_per_mol=MOLAR_MASS,
        central_volume_L_per_kg=1.0,
        peripheral_volume_L_per_kg=4.0,
        clearance_mL_per_min_per_kg=50.0,
        intercompartmental_clearance_mL_per_min_per_kg=200.0,
        route="iv",
    )
    kwargs.update(overrides)
    return TwoCompartmentDosePK(**kwargs)


def test_two_compartment_iv_shows_a_distribution_phase():
    pk = _two_compartment()
    sim = _single_ligand_sim(pk)
    t_eval = np.linspace(0, 240, 961)
    result = sim.run(t_span=(0, 240), t_eval=t_eval)
    C = result["concentrations"][0]
    peripheral = result["pk_aux"][0]

    assert C[0] == pytest.approx(pk.bioavailable_nmol / pk.V1)
    assert peripheral[0] == 0.0
    # Peripheral amount rises, peaks, then falls back as the whole system
    # is cleared: the signature of a real distribution phase.
    peak = int(np.argmax(peripheral))
    assert 0 < peak < len(t_eval) - 1
    assert peripheral[-1] < peripheral[peak]
    # Total drug in the body only ever decreases.
    total = C * pk.V1 + peripheral
    assert np.all(np.diff(total) <= 1e-9)
    # Early decline is faster than late decline: two distinguishable slopes
    # on a log scale is exactly what one compartment cannot produce.
    log_C = np.log(C)
    early_slope = (log_C[8] - log_C[0]) / (t_eval[8] - t_eval[0])
    late_slope = (log_C[-1] - log_C[-9]) / (t_eval[-1] - t_eval[-9])
    assert early_slope < late_slope < 0


def test_two_compartment_conserves_drug_when_clearance_is_zero():
    pk = _two_compartment(clearance_mL_per_min_per_kg=0.0)
    sim = _single_ligand_sim(pk)
    t_eval = np.linspace(0, 300, 301)
    result = sim.run(t_span=(0, 300), t_eval=t_eval)
    total = result["concentrations"][0] * pk.V1 + result["pk_aux"][0]
    np.testing.assert_allclose(total, pk.bioavailable_nmol, rtol=1e-6)
    # At equilibrium the two compartments share one concentration.
    C_central = result["concentrations"][0][-1]
    C_peripheral = result["pk_aux"][0][-1] / pk.V2
    assert C_central == pytest.approx(C_peripheral, rel=1e-3)


def test_two_compartment_extravascular_has_depot_then_peripheral():
    pk = _two_compartment(route="sc", ka_per_min=0.1, bioavailability=0.7)
    assert pk.n_aux == 2
    sim = _single_ligand_sim(pk)
    t_eval = np.linspace(0, 300, 601)
    result = sim.run(t_span=(0, 300), t_eval=t_eval)
    depot, peripheral = result["pk_aux"]
    assert depot[0] == pytest.approx(pk.bioavailable_nmol)
    assert peripheral[0] == 0.0
    assert result["concentrations"][0][0] == 0.0
    assert depot[-1] < 1e-6 * pk.bioavailable_nmol  # fully absorbed by then
    assert peripheral.max() > 0.0


# --- integration with the rest of the coupled system ----------------------


def test_dose_driven_pk_state_lives_in_the_coupled_state_vector():
    """`Simulation.rhs` must be the thing advancing the PK states; nothing
    may precompute a concentration trace. Evaluating `rhs` directly at a
    hand-built y and checking the depot/plasma derivatives is the direct
    proof of that.
    """
    pk = _dose_model(route="ip", ka_per_min=0.2, bioavailability=0.8)
    sim = _single_ligand_sim(pk)
    assert sim.a == 1
    C0, aux0 = sim.initial_pk_state()
    y0 = np.concatenate([C0, np.zeros(sim.r), np.zeros(sim.c), aux0])
    assert y0.shape == (sim.d + sim.r + sim.c + sim.a,)

    dy = sim.rhs(0.0, y0)
    d_depot = dy[sim.d + sim.r + sim.c]
    dC = dy[0]
    assert d_depot == pytest.approx(-pk.ka * pk.bioavailable_nmol)
    assert dC == pytest.approx(pk.ka * pk.bioavailable_nmol / pk.V)


def test_dose_driven_and_bolus_ligands_coexist_in_one_receptor_site():
    """An agonist given as a real mg/kg dose competing with an antagonist
    specified as an achieved concentration: the mixed case a reversal
    experiment needs, and a check that the aux block indexes correctly when
    only some ligands own aux states.
    """
    a = CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0)
    network = CircuitNetwork(nodes=[a], connectivity=[[0.0]])
    site = ReceptorSite(
        name="site",
        ligands=[
            Ligand(name="agonist", kon=0.1, koff=0.05, efficacy=1.0),
            Ligand(name="antagonist", kon=0.5, koff=0.05, efficacy=0.0),
        ],
    )
    agonist_pk = _dose_model(route="sc", ka_per_min=0.2, bioavailability=0.9)
    sim = Simulation(
        pk_models=[agonist_pk, OneCompartmentPK(ke=0.1)],
        receptor_site=site,
        network=network,
        coupling={"A": 1.0},
    )
    assert sim.a == 1  # only the dose-driven ligand owns an aux state

    t_eval = np.linspace(0, 120, 241)
    agonist_only = sim.run(
        dose_concentrations=[0.0, 0.0], t_span=(0, 120), t_eval=t_eval
    )
    blocked = sim.run(
        dose_concentrations=[0.0, 500.0], t_span=(0, 120), t_eval=t_eval
    )
    # The antagonist's own concentration decays from its bolus; the
    # agonist's rises from its depot, unaffected by the antagonist.
    np.testing.assert_allclose(
        blocked["concentrations"][0], agonist_only["concentrations"][0], rtol=1e-8
    )
    assert blocked["occupancy"][0].max() < agonist_only["occupancy"][0].max()
    assert blocked["rates"][0].max() < agonist_only["rates"][0].max()


def test_bolus_and_administered_dose_for_the_same_ligand_is_an_error():
    sim = _single_ligand_sim(_dose_model())
    with pytest.raises(ValueError, match="dose it twice"):
        sim.run(dose_concentrations=[1.0], t_span=(0, 10))


# --- input validation -----------------------------------------------------


def test_unknown_route_rejected():
    with pytest.raises(ValueError, match="route must be one of"):
        _dose_model(route="inhaled")


def test_iv_rejects_a_bioavailability_below_one():
    with pytest.raises(ValueError, match="bioavailability=1.0"):
        _dose_model(route="iv", bioavailability=0.5)


def test_iv_rejects_an_absorption_rate():
    with pytest.raises(ValueError, match="no absorption phase"):
        _dose_model(route="iv", ka_per_min=0.3)


def test_extravascular_route_requires_ka():
    with pytest.raises(ValueError, match="ka_per_min"):
        _dose_model(route="po")


def test_bioavailability_outside_zero_to_one_rejected():
    with pytest.raises(ValueError, match="bioavailability must lie"):
        _dose_model(route="po", ka_per_min=0.2, bioavailability=1.4)


@pytest.mark.parametrize(
    "bad",
    [
        {"dose_mg_per_kg": -1.0},
        {"body_weight_kg": 0.0},
        {"molar_mass_g_per_mol": 0.0},
        {"volume_of_distribution_L_per_kg": 0.0},
        {"clearance_mL_per_min_per_kg": -1.0},
    ],
)
def test_nonphysical_parameters_rejected(bad):
    with pytest.raises(ValueError):
        _dose_model(**bad)
