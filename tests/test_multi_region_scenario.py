import numpy as np
from scipy.integrate import solve_ivp

from neurocascade import scenarios
from neurocascade.engine import Simulation


def build_sim():
    return Simulation(
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        network=scenarios.multi_region_mu_opioid_network(),
        coupling=scenarios.mu_opioid_multi_region_coupling(),
    )


def test_multi_region_coupling_targets_at_least_three_nodes():
    coupling = scenarios.mu_opioid_multi_region_coupling()
    assert len(coupling) >= 3
    assert set(coupling) == {"VTA_GABA", "PAG_Interneuron", "LC"}
    # all three gains are nonzero and inhibitory (Gi-coupled receptor)
    assert all(gain < 0 for gain in coupling.values())


def test_state_arrays_have_expected_shapes():
    sim = build_sim()
    t_eval = np.linspace(0, 10, 50)
    result = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    assert result["concentrations"].shape == (2, 50)
    assert result["occupancy"].shape == (2, 50)
    # VTA_GABA, VTA_DA, NAc, PFC, PAG_Interneuron, PAG_Output, LC
    assert result["rates"].shape == (7, 50)
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)


def test_agonist_raises_incentive_salience_above_baseline():
    """Sanity check that expanding the reward circuit's network with two
    additional, unconnected sub-systems (PAG, LC) doesn't perturb the
    original reward-circuit dynamics: this should behave the same as
    test_engine.py's identically-named test on the smaller 4-node network.
    """
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


def test_agonist_raises_analgesia_index_above_baseline():
    sim = build_sim()
    readout = scenarios.analgesia_index_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    agonist_final = readout.score(sim.network, agonist["rates"][:, -1])
    # Agonist disinhibits PAG_Output (via PAG_Interneuron suppression), so
    # descending-analgesia output, and therefore analgesia_index, should
    # rise well above baseline.
    assert agonist_final > baseline_final + 0.1


def test_agonist_lowers_arousal_index_below_baseline():
    sim = build_sim()
    readout = scenarios.arousal_index_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    agonist_final = readout.score(sim.network, agonist["rates"][:, -1])
    # LC is coupled directly (not via disinhibition): agonist activation
    # suppresses LC firing directly, so arousal_index should fall below
    # baseline (illustrative opioid-sedation effect).
    assert agonist_final < baseline_final - 0.05


def test_antagonist_alone_stays_near_baseline_on_all_three_readouts():
    sim = build_sim()
    readouts = {
        "incentive_salience": scenarios.incentive_salience_readout(),
        "analgesia_index": scenarios.analgesia_index_readout(),
        "arousal_index": scenarios.arousal_index_readout(),
    }

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    antagonist = sim.run(dose_concentrations=[0.0, 1.0], t_span=(0, 40), t_eval=t_eval)

    for readout in readouts.values():
        baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
        antagonist_final = readout.score(sim.network, antagonist["rates"][:, -1])
        assert abs(antagonist_final - baseline_final) < 1e-2


def test_naloxone_reverses_all_three_simultaneous_effects():
    """The key demonstration: one antagonist, competing at ONE shared
    receptor pool, reverses reward, analgesia, AND sedation at once, because
    all three come from the same occupancy trace, not three independent
    drug actions on three independent receptors.
    """
    sim = build_sim()
    readouts = {
        "incentive_salience": scenarios.incentive_salience_readout(),
        "analgesia_index": scenarios.analgesia_index_readout(),
        "arousal_index": scenarios.arousal_index_readout(),
    }
    t_span = (0, 40)
    t_split = 20.0

    t_eval_1 = np.linspace(t_span[0], t_split, 200)
    phase1 = sim.run(
        dose_concentrations=[1.0, 0.0], t_span=(t_span[0], t_split), t_eval=t_eval_1
    )
    scores_at_split = {
        name: readout.score(sim.network, phase1["rates"][:, -1])
        for name, readout in readouts.items()
    }

    C0 = phase1["concentrations"][:, -1].copy()
    C0[1] += 3.0  # naloxone bolus injected at t_split
    occ0 = phase1["occupancy"][:, -1]
    rates0 = phase1["rates"][:, -1]
    y0 = np.concatenate([C0, occ0, rates0])

    t_eval_2 = np.linspace(t_split, t_span[1], 200)
    sol = solve_ivp(
        sim.rhs, (t_split, t_span[1]), y0, method="RK45", t_eval=t_eval_2, max_step=0.1
    )
    d, r = sim.d, sim.r
    rates2 = sol.y[d + r : d + r + sim.c]

    # incentive_salience and analgesia_index were elevated by fentanyl, so
    # naloxone should pull them back DOWN toward baseline; arousal_index was
    # suppressed by fentanyl, so naloxone should pull it back UP.
    for name in ("incentive_salience", "analgesia_index"):
        readout = readouts[name]
        scores_after = np.array(
            [readout.score(sim.network, rates2[:, i]) for i in range(len(t_eval_2))]
        )
        assert scores_after.min() < scores_at_split[name]

    readout = readouts["arousal_index"]
    scores_after = np.array(
        [readout.score(sim.network, rates2[:, i]) for i in range(len(t_eval_2))]
    )
    assert scores_after.max() > scores_at_split["arousal_index"]
