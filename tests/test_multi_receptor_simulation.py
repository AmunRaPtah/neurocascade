"""Tests for the multi-receptor-system extension to `Simulation`
(`engine.ReceptorSystem` + `Simulation(network, receptor_systems=[...])`),
which lets several receptor sites, each with its own PK models, ligands,
coupling, and connection_modulation, drive one shared `CircuitNetwork`
simultaneously. See docs/architecture.md and scenarios.py's "Combined
scenario" section for the full design rationale.
"""

import numpy as np

from neurocascade import scenarios
from neurocascade.circuit import CircuitNode
from neurocascade.engine import ReceptorSystem, Simulation
from neurocascade.network import CircuitNetwork
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import Ligand, ReceptorSite


def build_combined_sim():
    network = scenarios.combined_mu_opioid_d2_network()
    mu_system = scenarios.mu_opioid_receptor_system()
    d2_system = scenarios.d2_receptor_system()
    return Simulation(network=network, receptor_systems=[mu_system, d2_system])


# ---------------------------------------------------------------------------
# (a) runs and integrates without error
# ---------------------------------------------------------------------------


def test_combined_simulation_integrates_without_error():
    sim = build_combined_sim()
    assert sim.d == 4  # 2 mu-opioid ligands + 2 D2 ligands
    assert sim.r == 4
    assert sim.c == 9  # 4 reward-circuit nodes + 5 basal-ganglia nodes

    doses = sim.dose_vector(fentanyl=1.0, dopamine=1.0)
    t_eval = np.linspace(0, 40, 100)
    result = sim.run(dose_concentrations=doses, t_span=(0, 40), t_eval=t_eval)

    assert result["concentrations"].shape == (4, 100)
    assert result["occupancy"].shape == (4, 100)
    assert result["rates"].shape == (9, 100)
    assert np.all(np.isfinite(result["rates"]))
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)


def test_dose_vector_rejects_unknown_ligand_name():
    sim = build_combined_sim()
    try:
        sim.dose_vector(not_a_real_ligand=1.0)
        assert False, "expected ValueError for unknown ligand name"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# (b) each receptor system's dynamics in isolation match single-receptor-
# system behavior: zero out one system's ligands and compare the other
# system's block of the combined network against a standalone Simulation
# on that system's own circuit alone.
# ---------------------------------------------------------------------------


def test_mu_opioid_block_matches_standalone_when_d2_ligands_are_zero():
    combined = build_combined_sim()
    standalone = Simulation(
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        network=scenarios.reward_circuit_network(),
        coupling=scenarios.mu_opioid_coupling(),
    )

    t_eval = np.linspace(0, 40, 100)
    combined_doses = combined.dose_vector(fentanyl=1.0)  # dopamine/haloperidol = 0
    combined_result = combined.run(
        dose_concentrations=combined_doses, t_span=(0, 40), t_eval=t_eval
    )
    standalone_result = standalone.run(
        dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval
    )

    reward_nodes = ["VTA_GABA", "VTA_DA", "NAc", "PFC"]
    reward_idx = [combined.network.index[name] for name in reward_nodes]
    combined_reward_rates = combined_result["rates"][reward_idx, :]

    # Same block-diagonal network, same PK/receptor parameters, same dosing:
    # this must reproduce the standalone reward-circuit run to numerical
    # integration tolerance, not just qualitatively.
    assert np.allclose(
        combined_reward_rates, standalone_result["rates"], atol=1e-5
    )
    assert np.allclose(
        combined_result["concentrations"][0:2],
        standalone_result["concentrations"],
        atol=1e-8,
    )
    assert np.allclose(
        combined_result["occupancy"][0:2], standalone_result["occupancy"], atol=1e-8
    )


def test_d2_block_matches_standalone_when_mu_opioid_ligands_are_zero():
    combined = build_combined_sim()
    standalone = Simulation(
        pk_models=scenarios.d2_pk_models(),
        receptor_site=scenarios.d2_receptor_site(),
        network=scenarios.basal_ganglia_indirect_pathway_network(),
        coupling=scenarios.d2_coupling(),
    )

    t_eval = np.linspace(0, 40, 100)
    combined_doses = combined.dose_vector(dopamine=1.0)  # fentanyl/naloxone = 0
    combined_result = combined.run(
        dose_concentrations=combined_doses, t_span=(0, 40), t_eval=t_eval
    )
    standalone_result = standalone.run(
        dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval
    )

    bg_nodes = ["Striatum_D2_MSN", "GPe", "STN", "GPi", "Thalamus"]
    bg_idx = [combined.network.index[name] for name in bg_nodes]
    combined_bg_rates = combined_result["rates"][bg_idx, :]

    assert np.allclose(combined_bg_rates, standalone_result["rates"], atol=1e-5)
    assert np.allclose(
        combined_result["concentrations"][2:4],
        standalone_result["concentrations"],
        atol=1e-8,
    )
    assert np.allclose(
        combined_result["occupancy"][2:4], standalone_result["occupancy"], atol=1e-8
    )


def test_both_readouts_show_correct_effect_under_simultaneous_dosing():
    """Both drugs given at once: each readout should show the same
    qualitative effect it shows alone (fentanyl raises incentive_salience,
    dopamine raises locomotor_activity), confirming the two receptor
    systems' effects on their own parts of the shared network don't corrupt
    each other under genuinely simultaneous integration (not just when one
    side is zeroed, as in the two tests above).
    """
    combined = build_combined_sim()
    incentive_readout = scenarios.incentive_salience_readout()
    locomotor_readout = scenarios.locomotor_activity_readout()

    t_eval = np.linspace(0, 40, 100)
    baseline_doses = combined.dose_vector()
    both_doses = combined.dose_vector(fentanyl=1.0, dopamine=1.0)

    baseline = combined.run(
        dose_concentrations=baseline_doses, t_span=(0, 40), t_eval=t_eval
    )
    both = combined.run(dose_concentrations=both_doses, t_span=(0, 40), t_eval=t_eval)

    baseline_incentive = np.array(
        [
            incentive_readout.score(combined.network, baseline["rates"][:, i])
            for i in range(len(t_eval))
        ]
    )
    both_incentive = np.array(
        [
            incentive_readout.score(combined.network, both["rates"][:, i])
            for i in range(len(t_eval))
        ]
    )
    baseline_locomotor = np.array(
        [
            locomotor_readout.score(combined.network, baseline["rates"][:, i])
            for i in range(len(t_eval))
        ]
    )
    both_locomotor = np.array(
        [
            locomotor_readout.score(combined.network, both["rates"][:, i])
            for i in range(len(t_eval))
        ]
    )

    assert both_incentive.max() > baseline_incentive.max()
    assert both_locomotor.max() > baseline_locomotor.max()


# ---------------------------------------------------------------------------
# (c) a plausible interaction case: the engine-level multiplicative
# composition rule for connection_modulation when two receptor systems
# target the exact same (post_node, pre_node) connection. This is the one
# case in the combined mu-opioid/D2 scenario's block-diagonal network where
# "interaction" would actually be possible (docs/architecture.md documents
# why that scenario deliberately doesn't wire one), so it's tested directly
# at the engine level with a minimal synthetic network, the same pattern
# tests/test_engine.py already uses for the single-system
# connection_modulation tests.
# ---------------------------------------------------------------------------


def _two_pam_systems_on_one_connection(gain_1, gain_2):
    """A synthetic 2-node network (B inhibits A) with two independent
    receptor systems, each an allosteric modulator of the SAME B -> A
    connection with its own gain. Mirrors
    tests/test_engine.py::_minimal_allosteric_setup, extended to two
    systems targeting one connection instead of one.
    """
    a = CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0)
    b = CircuitNode(name="B", tau=1.0, baseline_input=1.0, gain=4.0)
    network = CircuitNetwork(nodes=[a, b], connectivity=[[0.0, -1.0], [0.0, 0.0]])

    modulator_1 = Ligand(name="modulator_1", kon=1.0, koff=0.1, efficacy=1.0)
    site_1 = ReceptorSite(name="allosteric_site_1", ligands=[modulator_1])
    system_1 = ReceptorSystem(
        pk_models=[OneCompartmentPK(ke=0.05)],
        receptor_site=site_1,
        coupling={},
        connection_modulation={("A", "B"): gain_1},
    )

    modulator_2 = Ligand(name="modulator_2", kon=1.0, koff=0.1, efficacy=1.0)
    site_2 = ReceptorSite(name="allosteric_site_2", ligands=[modulator_2])
    system_2 = ReceptorSystem(
        pk_models=[OneCompartmentPK(ke=0.05)],
        receptor_site=site_2,
        coupling={},
        connection_modulation={("A", "B"): gain_2},
    )

    return Simulation(network=network, receptor_systems=[system_1, system_2])


def test_two_systems_modulating_the_same_connection_compose_multiplicatively():
    # Force near-saturating occupancy directly (rather than waiting out
    # PK/binding kinetics) to isolate the composition rule from binding
    # dynamics. With both modulators at occ ~= 1 and gain 1.0 each, the
    # documented rule scales the B -> A connection by
    # (1 + 1*occ) * (1 + 1*occ) = 4, taking the base weight of -1.0 to
    # -4.0. The rejected additive alternative (`1 + sum(gain_i * occ_i)`)
    # would instead give 1 + (1 + 1) = 3, i.e. a weight of -3.0. This test
    # checks the actual derivative matches the multiplicative case and
    # differs measurably from the additive one.
    sim = _two_pam_systems_on_one_connection(gain_1=1.0, gain_2=1.0)
    occ = 0.999
    rate_b = 1.0  # nonzero, so the B -> A weight actually affects A's input
    y = np.array([1000.0, 1000.0, occ, occ, 0.0, rate_b])
    dy = sim.rhs(0.0, y)
    d_rate_a = dy[sim.d + sim.r + sim.network.index["A"]]

    from neurocascade.circuit import sigmoid

    node_a = sim.network.nodes[sim.network.index["A"]]
    baseline_input_a = node_a.baseline_input

    def d_rate_a_under(weight_ba):
        target = sigmoid(weight_ba * rate_b + baseline_input_a, gain=node_a.gain, threshold=node_a.threshold)
        return (-0.0 + target) / node_a.tau

    multiplicative_weight = -1.0 * (1.0 + 1.0 * occ) * (1.0 + 1.0 * occ)
    additive_weight = -1.0 * (1.0 + (1.0 * occ + 1.0 * occ))
    assert abs(multiplicative_weight - additive_weight) > 0.5  # sanity: they differ

    assert abs(d_rate_a - d_rate_a_under(multiplicative_weight)) < 1e-10
    assert abs(d_rate_a - d_rate_a_under(additive_weight)) > 1e-6


def test_single_system_connection_modulation_unaffected_by_composition_change():
    # With only one system touching a connection, multiplicative and
    # additive composition are identical (nothing to compose): confirms the
    # multi-system composition rule didn't change single-system behavior,
    # which the full existing test_engine.py suite also already covers, but
    # is worth asserting explicitly here since it's the precondition that
    # makes the composition choice "safe" for every pre-existing scenario.
    a = CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0)
    b = CircuitNode(name="B", tau=1.0, baseline_input=1.0, gain=4.0)
    network = CircuitNetwork(nodes=[a, b], connectivity=[[0.0, -1.0], [0.0, 0.0]])
    modulator = Ligand(name="modulator", kon=1.0, koff=0.1, efficacy=1.0)
    site = ReceptorSite(name="allosteric_site", ligands=[modulator])
    system = ReceptorSystem(
        pk_models=[OneCompartmentPK(ke=0.05)],
        receptor_site=site,
        coupling={},
        connection_modulation={("A", "B"): 2.0},
    )
    sim_new_api = Simulation(network=network, receptor_systems=[system])

    sim_legacy_api = Simulation(
        pk_models=[OneCompartmentPK(ke=0.05)],
        receptor_site=ReceptorSite(
            name="allosteric_site", ligands=[Ligand(name="modulator", kon=1.0, koff=0.1, efficacy=1.0)]
        ),
        network=CircuitNetwork(
            nodes=[
                CircuitNode(name="A", tau=1.0, baseline_input=1.0, gain=4.0),
                CircuitNode(name="B", tau=1.0, baseline_input=1.0, gain=4.0),
            ],
            connectivity=[[0.0, -1.0], [0.0, 0.0]],
        ),
        coupling={},
        connection_modulation={("A", "B"): 2.0},
    )

    t_eval = np.linspace(0, 20, 50)
    result_new = sim_new_api.run(dose_concentrations=[1.0], t_span=(0, 20), t_eval=t_eval)
    result_legacy = sim_legacy_api.run(dose_concentrations=[1.0], t_span=(0, 20), t_eval=t_eval)

    assert np.allclose(result_new["rates"], result_legacy["rates"])
