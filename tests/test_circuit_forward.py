"""Tests for `neurocascade.calibration.circuit_forward` (P2.2).

This module is the piece that lets a CIRCUIT-layer gain be fitted at all, so
the things worth testing are the claims its docstring makes rather than its
arithmetic:

1. The steady state it returns really is one (checked against a far longer,
   far tighter integration, not against a hard-coded number).
2. The connection-modulation sign convention matches `engine.Simulation`'s -
   a positive gain on an inhibitory weight strengthens the inhibition.
3. The drug/vehicle ratio is exactly 1 with no drug on board, whatever the
   gain, which is the property that makes the passed-in baseline a pure
   cache rather than an approximation.
4. The clamped-activation approximation - holding receptor activation at its
   equilibrium value while the circuit relaxes, instead of co-integrating a
   decaying concentration - agrees with a real `engine.Simulation` run of
   the same scenario. This is the one approximation in the P2.2 forward
   model, and it is checked rather than argued for.
"""

import numpy as np
import pytest
from scipy import sparse

from neurocascade import scenarios
from neurocascade.calibration import circuit_forward as cfwd
from neurocascade.engine import Simulation
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import Ligand, ReceptorSite


MODULATED_CONNECTION = ("BLA", "GABA_Interneuron")


def _network():
    return scenarios.amygdala_anxiety_circuit_network()


def test_steady_state_matches_a_much_longer_and_tighter_integration():
    network = _network()
    modulation = {MODULATED_CONNECTION: 0.3}
    fast = cfwd.steady_state_rates(network, modulation, activation=0.8)
    reference = cfwd.steady_state_rates(
        network, modulation, activation=0.8,
        rtol=1e-11, atol=1e-14, tau_multiples=300.0,
    )
    assert np.allclose(fast, reference, atol=1e-8)


def test_vectorized_rhs_matches_original_nodewise_rhs():
    network = _network()
    modulation = {MODULATED_CONNECTION: 0.3}
    W = cfwd.modulated_connectivity(network, modulation, 0.8)
    rates = np.linspace(0.2, 0.8, network.n)
    drive = network.baseline_drive()
    assert np.array_equal(
        network.rhs(rates, drive, connectivity=W),
        network.rhs_reference(rates, drive, connectivity=W),
    )


def test_direct_root_matches_integrated_steady_state():
    network = _network()
    modulation = {MODULATED_CONNECTION: 0.3}
    integrated = cfwd.steady_state_rates(
        network, modulation, activation=0.8, tau_multiples=100.0,
    )
    direct = cfwd.steady_state_rates(
        network, modulation, activation=0.8, solver="root",
    )
    assert np.allclose(direct, integrated, atol=1e-8)
    W = cfwd.modulated_connectivity(network, modulation, 0.8)
    assert np.max(np.abs(network.rhs(
        direct, network.baseline_drive(), connectivity=W
    ))) < 1e-10


def test_steady_state_is_a_fixed_point_of_the_networks_own_rhs():
    """Independent of the integrator: at the returned rates, the derivative
    the circuit layer itself reports must be zero."""
    network = _network()
    modulation = {MODULATED_CONNECTION: 0.3}
    rates = cfwd.steady_state_rates(network, modulation, activation=0.8)
    W = cfwd.modulated_connectivity(network, modulation, 0.8)
    derivative = network.rhs(rates, network.baseline_drive(), connectivity=W)
    assert np.max(np.abs(derivative)) < 1e-7


def test_circuit_jacobian_matches_finite_difference_and_is_sparse():
    network = _network()
    modulation = {MODULATED_CONNECTION: 0.3}
    activation = 0.8
    W = cfwd.modulated_connectivity(network, modulation, activation)
    drive = network.baseline_drive()
    rates = np.linspace(0.2, 0.8, network.n)
    analytic = cfwd.circuit_jacobian(network, rates, drive=drive, connectivity=W)

    epsilon = 1e-7
    numeric = np.column_stack([
        (
            network.rhs(rates + epsilon * np.eye(network.n)[j], drive, connectivity=W)
            - network.rhs(rates - epsilon * np.eye(network.n)[j], drive, connectivity=W)
        ) / (2.0 * epsilon)
        for j in range(network.n)
    ])
    assert sparse.isspmatrix_csr(analytic)
    assert np.allclose(analytic.toarray(), numeric, rtol=1e-6, atol=1e-8)


def test_stiff_solver_matches_rk45_with_sparse_jacobian():
    network = _network()
    modulation = {MODULATED_CONNECTION: 0.3}
    reference = cfwd.steady_state_rates(
        network, modulation, activation=0.8, method="RK45"
    )
    for method in ("BDF", "Radau"):
        rates = cfwd.steady_state_rates(
            network, modulation, activation=0.8, method=method,
            use_sparse_jacobian=True,
        )
        assert np.allclose(rates, reference, rtol=1e-6, atol=1e-8)


def test_positive_gain_strengthens_an_inhibitory_connection():
    network = _network()
    i = network.index[MODULATED_CONNECTION[0]]
    j = network.index[MODULATED_CONNECTION[1]]
    base_weight = network.connectivity[i, j]
    assert base_weight < 0.0  # the GABAergic synapse benzodiazepines potentiate

    W = cfwd.modulated_connectivity(network, {MODULATED_CONNECTION: 0.5}, 0.6)
    assert W[i, j] == pytest.approx(base_weight * (1.0 + 0.5 * 0.6))
    assert W[i, j] < base_weight  # more negative = stronger inhibition
    # Nothing else moved.
    untouched = np.array(network.connectivity, dtype=float)
    untouched[i, j] = W[i, j]
    assert np.allclose(W, untouched)


def test_modulated_connectivity_does_not_mutate_the_network():
    network = _network()
    before = np.array(network.connectivity, dtype=float, copy=True)
    cfwd.modulated_connectivity(network, {MODULATED_CONNECTION: 2.0}, 1.0)
    assert np.allclose(network.connectivity, before)


def test_ratio_is_one_without_drug_whatever_the_gain():
    network = _network()
    readout = scenarios.anxiety_index_readout()
    for gain in (0.0, 0.07, 1.0, 5.0):
        ratio = cfwd.readout_ratio(
            network, readout, {MODULATED_CONNECTION: gain}, activation=0.0
        )
        assert ratio == pytest.approx(1.0, abs=1e-9)


def test_passing_the_baseline_in_changes_nothing():
    """`readout_ratio`'s `baseline_rates` argument exists only to avoid
    re-integrating a parameter-independent leg inside an MCMC chain. If it
    ever changed an answer it would be an approximation, not a cache."""
    network = _network()
    readout = scenarios.anxiety_index_readout()
    modulation = {MODULATED_CONNECTION: 0.07}
    baseline = cfwd.steady_state_rates(network, modulation, activation=0.0)
    with_cache = cfwd.readout_ratio(
        network, readout, modulation, activation=0.9, baseline_rates=baseline
    )
    without_cache = cfwd.readout_ratio(network, readout, modulation, activation=0.9)
    assert with_cache == pytest.approx(without_cache, rel=1e-9)


def test_anxiolysis_is_monotone_in_the_gain_over_the_whole_prior_range():
    """The property that makes the gain identifiable from a single published
    drug/vehicle ratio: one gain, one ratio, no folding."""
    network = _network()
    readout = scenarios.anxiety_index_readout()
    gains = [0.0, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    ratios = [
        cfwd.readout_ratio(network, readout, {MODULATED_CONNECTION: g}, activation=0.9)
        for g in gains
    ]
    assert ratios[0] == pytest.approx(1.0, abs=1e-9)
    assert all(later <= earlier + 1e-9 for earlier, later in zip(ratios, ratios[1:]))
    assert ratios[-1] < 0.1  # the range brackets the published effect comfortably


def test_direct_coupling_path_is_covered_too():
    """The GABA-A scenario uses `connection_modulation` only, but the other
    two receptor systems in this codebase act through `coupling`, so the
    helper has to handle a drive coupling as well; a Gi-coupled negative
    gain must lower its target node's rate."""
    network = _network()
    with_drive = cfwd.steady_state_rates(
        network, activation=1.0, coupling={"CeA": -2.0}
    )
    without = cfwd.steady_state_rates(network, activation=1.0)
    assert with_drive[network.index["CeA"]] < without[network.index["CeA"]]


def test_clamped_activation_agrees_with_a_full_engine_simulation():
    """The one approximation in the P2.2 forward model.

    `circuit_forward` holds receptor activation at its equilibrium value
    while the circuit relaxes. `engine.Simulation` instead co-integrates a
    decaying plasma concentration, the receptor ODE and the circuit ODE in
    one state vector. Build the same scenario both ways, with a PK model
    slow enough that the concentration is effectively constant over the
    circuit's relaxation (which is the regime the [Graziani2022] fit sits
    in), and require the two anxiety-index ratios to agree.
    """
    gain = 0.07
    kd = 3.6  # nM, diazepam at the benzodiazepine site [Mohler1977]
    kon = 0.2
    concentration = 32.0  # nM, the order the P2.2 exposure reaches
    modulation = {MODULATED_CONNECTION: gain}

    network = _network()
    readout = scenarios.anxiety_index_readout()

    # Clamped-activation prediction.
    activation = concentration / (concentration + kd)
    clamped = cfwd.readout_ratio(network, readout, modulation, activation=activation)

    # Full coupled simulation, with a very slow ke so the concentration the
    # receptor sees barely moves over the run (the assumption under test is
    # about a flat exposure, not about a fast one).
    site = ReceptorSite(
        name="benzodiazepine_site",
        ligands=[Ligand(name="diazepam", kon=kon, koff=kon * kd, efficacy=1.0)],
    )
    sim = Simulation(
        pk_models=[OneCompartmentPK(ke=1e-6)],
        receptor_site=site,
        network=_network(),
        coupling={},
        connection_modulation=modulation,
    )
    t_eval = np.linspace(0.0, 240.0, 400)
    baseline = sim.run(dose_concentrations=[0.0], t_span=(0.0, 240.0), t_eval=t_eval)
    drugged = sim.run(
        dose_concentrations=[concentration], t_span=(0.0, 240.0), t_eval=t_eval
    )
    simulated = readout.score(sim.network, drugged["rates"][:, -1]) / readout.score(
        sim.network, baseline["rates"][:, -1]
    )

    assert clamped == pytest.approx(simulated, rel=0.02)
