import numpy as np
from scipy.integrate import solve_ivp

from neurocascade import scenarios
from neurocascade.engine import Simulation


def build_sim():
    return Simulation(
        pk_models=scenarios.gaba_a_pk_models(),
        receptor_site=scenarios.gaba_a_receptor_site(),
        network=scenarios.amygdala_anxiety_circuit_network(),
        coupling=scenarios.gaba_a_coupling(),
        connection_modulation=scenarios.gaba_a_connection_modulation(),
    )


def test_diazepam_lowers_anxiety_index_below_baseline():
    sim = build_sim()
    readout = scenarios.anxiety_index_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    diazepam = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    diazepam_final = readout.score(sim.network, diazepam["rates"][:, -1])
    # Diazepam potentiates the GABAergic interneuron -> BLA synapse
    # (connection_modulation, not coupling), suppressing BLA and, downstream,
    # CeA output: anxiety_index should drop well below baseline.
    assert diazepam_final < baseline_final - 0.1


def test_flumazenil_alone_stays_near_baseline():
    sim = build_sim()
    readout = scenarios.anxiety_index_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    flumazenil = sim.run(dose_concentrations=[0.0, 1.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    flumazenil_final = readout.score(sim.network, flumazenil["rates"][:, -1])
    # Flumazenil has efficacy=0 (no potentiation of its own) and there's no
    # diazepam present to displace, so it should do ~nothing on its own.
    assert abs(flumazenil_final - baseline_final) < 1e-2


def test_flumazenil_partially_reverses_diazepam():
    """Two-phase dosing (diazepam first, flumazenil injected mid-course)
    mirrors examples/gaba_a_benzodiazepine_circuit.py and
    examples/fentanyl_reward_circuit.py's reversal pattern: flumazenil
    competes with diazepam at the shared benzodiazepine-site pool
    (ReceptorSite's ordinary competitive occupancy), displacing some of it
    and partially restoring anxiety_index toward baseline.
    """
    sim = build_sim()
    readout = scenarios.anxiety_index_readout()
    t_span = (0, 40)
    t_split = 20.0

    t_eval_1 = np.linspace(t_span[0], t_split, 200)
    phase1 = sim.run(
        dose_concentrations=[1.0, 0.0], t_span=(t_span[0], t_split), t_eval=t_eval_1
    )
    score_at_split = readout.score(sim.network, phase1["rates"][:, -1])

    C0 = phase1["concentrations"][:, -1].copy()
    C0[1] += 3.0  # flumazenil bolus injected at t_split
    occ0 = phase1["occupancy"][:, -1]
    rates0 = phase1["rates"][:, -1]
    y0 = np.concatenate([C0, occ0, rates0])

    t_eval_2 = np.linspace(t_split, t_span[1], 200)
    sol = solve_ivp(
        sim.rhs, (t_split, t_span[1]), y0, method="RK45", t_eval=t_eval_2, max_step=0.1
    )
    d, r = sim.d, sim.r
    rates2 = sol.y[d + r : d + r + sim.c]
    scores_after = np.array(
        [readout.score(sim.network, rates2[:, i]) for i in range(len(t_eval_2))]
    )

    # Flumazenil should push the score up (toward baseline) at some point
    # after injection, relative to where diazepam alone had it.
    assert scores_after.max() > score_at_split


def test_state_arrays_have_expected_shapes():
    sim = build_sim()
    t_eval = np.linspace(0, 10, 50)
    result = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    assert result["concentrations"].shape == (2, 50)
    assert result["occupancy"].shape == (2, 50)
    assert result["rates"].shape == (3, 50)  # GABA_Interneuron, BLA, CeA
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)
