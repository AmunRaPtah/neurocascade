import numpy as np

from neurocascade import scenarios
from neurocascade.engine import Simulation


def build_sim():
    return Simulation(
        pk_models=scenarios.d2_pk_models(),
        receptor_site=scenarios.d2_receptor_site(),
        network=scenarios.basal_ganglia_indirect_pathway_network(),
        coupling=scenarios.d2_coupling(),
    )


def test_agonist_raises_locomotor_activity_above_baseline():
    sim = build_sim()
    readout = scenarios.locomotor_activity_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_scores = np.array(
        [readout.score(sim.network, baseline["rates"][:, i]) for i in range(len(t_eval))]
    )
    agonist_scores = np.array(
        [readout.score(sim.network, agonist["rates"][:, i]) for i in range(len(t_eval))]
    )
    # Dopamine at D2 disinhibits the indirect pathway (Striatum_D2_MSN
    # suppressed -> GPe up -> STN down -> GPi down -> Thalamus up), so
    # locomotor_activity (weighted on Thalamus) should rise above baseline.
    assert agonist_scores.max() > baseline_scores.max()


def test_antagonist_alone_stays_near_baseline():
    sim = build_sim()
    readout = scenarios.locomotor_activity_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    antagonist = sim.run(dose_concentrations=[0.0, 1.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    antagonist_final = readout.score(sim.network, antagonist["rates"][:, -1])
    # Haloperidol has nothing to block without endogenous dopamine present in
    # this bolus-only model, so it should track baseline closely.
    assert abs(antagonist_final - baseline_final) < 1e-2


def test_state_arrays_have_expected_shapes():
    sim = build_sim()
    t_eval = np.linspace(0, 10, 50)
    result = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    assert result["concentrations"].shape == (2, 50)
    assert result["occupancy"].shape == (2, 50)
    assert result["rates"].shape == (5, 50)  # 5 basal-ganglia nodes
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)
