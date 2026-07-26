import numpy as np

from neurocascade import scenarios
from neurocascade.engine import Simulation
from neurocascade.circuit import CircuitNode
from neurocascade.network import CircuitNetwork
from neurocascade.receptor import Ligand, ReceptorSite
from neurocascade.pk import OneCompartmentPK


def build_sim():
    return Simulation(
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        network=scenarios.reward_circuit_network(),
        coupling=scenarios.mu_opioid_coupling(),
    )


def test_agonist_raises_incentive_salience_above_baseline():
    sim = build_sim()
    readout = scenarios.incentive_salience_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_scores = np.array(
        [readout.score(sim.network, baseline["rates"][:, i]) for i in range(len(t_eval))]
    )
    agonist_scores = np.array(
        [readout.score(sim.network, agonist["rates"][:, i]) for i in range(len(t_eval))]
    )
    assert agonist_scores.max() > baseline_scores.max()


def test_antagonist_alone_stays_near_baseline():
    sim = build_sim()
    readout = scenarios.incentive_salience_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    antagonist = sim.run(dose_concentrations=[0.0, 1.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    antagonist_final = readout.score(sim.network, antagonist["rates"][:, -1])
    assert abs(antagonist_final - baseline_final) < 1e-2


def _minimal_allosteric_setup():
    """A synthetic 2-node network with one inhibitory connection, used to
    unit test `connection_modulation` in isolation from any scenario's
    tuned parameters: a full-efficacy "modulator" ligand should strengthen
    the B -> A inhibitory connection, and coupling is intentionally empty
    since the modulator has no direct drive of its own.
    """
    a = CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0)
    b = CircuitNode(name="B", tau=1.0, baseline_input=1.0, gain=4.0)
    W = [[0.0, -1.0], [0.0, 0.0]]  # B inhibits A
    network = CircuitNetwork(nodes=[a, b], connectivity=W)

    modulator = Ligand(name="modulator", kon=1.0, koff=0.1, efficacy=1.0)
    site = ReceptorSite(name="allosteric_site", ligands=[modulator])
    pk_models = [OneCompartmentPK(ke=0.05)]

    sim = Simulation(
        pk_models=pk_models,
        receptor_site=site,
        network=network,
        coupling={},
        connection_modulation={("A", "B"): 2.0},
    )
    return sim


def test_connection_modulation_strengthens_targeted_connection_only():
    sim = _minimal_allosteric_setup()
    t_eval = np.linspace(0, 20, 100)

    baseline = sim.run(dose_concentrations=[0.0], t_span=(0, 20), t_eval=t_eval)
    modulated = sim.run(dose_concentrations=[1.0], t_span=(0, 20), t_eval=t_eval)

    # Occupancy builds up, so the B -> A inhibition strengthens, which
    # should suppress node A's rate relative to the unmodulated baseline.
    assert modulated["rates"][0, -1] < baseline["rates"][0, -1]
    # Node B receives no coupling and isn't the modulation target, so it
    # should be essentially unaffected.
    assert abs(modulated["rates"][1, -1] - baseline["rates"][1, -1]) < 1e-6
    # The network's own connectivity must be untouched by any of this.
    assert np.array_equal(sim.network.connectivity, np.array([[0.0, -1.0], [0.0, 0.0]]))


def test_connection_modulation_defaults_to_no_effect():
    a = CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0)
    b = CircuitNode(name="B", tau=1.0, baseline_input=1.0, gain=4.0)
    network = CircuitNetwork(nodes=[a, b], connectivity=[[0.0, -1.0], [0.0, 0.0]])
    modulator = Ligand(name="modulator", kon=1.0, koff=0.1, efficacy=1.0)
    site = ReceptorSite(name="allosteric_site", ligands=[modulator])

    sim_no_modulation = Simulation(
        pk_models=[OneCompartmentPK(ke=0.05)],
        receptor_site=site,
        network=network,
        coupling={},
    )
    assert sim_no_modulation.connection_modulation == {}
    t_eval = np.linspace(0, 20, 50)
    result = sim_no_modulation.run(
        dose_concentrations=[1.0], t_span=(0, 20), t_eval=t_eval
    )
    # With no coupling and no connection_modulation, dosing the ligand
    # should have no effect on either node's dynamics at all.
    still_baseline = sim_no_modulation.run(
        dose_concentrations=[0.0], t_span=(0, 20), t_eval=t_eval
    )
    assert np.allclose(result["rates"], still_baseline["rates"])


def test_state_arrays_have_expected_shapes():
    sim = build_sim()
    t_eval = np.linspace(0, 10, 50)
    result = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    assert result["concentrations"].shape == (2, 50)
    assert result["occupancy"].shape == (2, 50)
    assert result["rates"].shape == (4, 50)
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)
