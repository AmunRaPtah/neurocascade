"""v0 worked example: mu-opioid agonism/antagonism in a VTA-NAc-PFC reward circuit.

Runs three scenarios end to end (PK -> receptor occupancy -> circuit rates ->
behavior score) and plots them together:
  1. fentanyl alone: agonist, should raise incentive salience
  2. naloxone alone: antagonist, should do ~nothing on its own
  3. fentanyl + naloxone: antagonist given after the agonist, should reverse
     the salience spike (opioid-overdose reversal, the mechanism naloxone
     rescue relies on)

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/fentanyl_reward_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, scenarios  # noqa: E402


def run_scenario(dose_fentanyl, dose_naloxone_delayed=None, t_span=(0, 40)):
    receptor_site = scenarios.mu_opioid_receptor_site()
    pk_models = scenarios.mu_opioid_pk_models()
    network = scenarios.reward_circuit_network()
    coupling = scenarios.mu_opioid_coupling()
    readout = scenarios.incentive_salience_readout()

    sim = Simulation(
        pk_models=pk_models, receptor_site=receptor_site, network=network,
        coupling=coupling,
    )

    if dose_naloxone_delayed is None:
        t_eval = np.linspace(t_span[0], t_span[1], 400)
        result = sim.run(
            dose_concentrations=[dose_fentanyl, 0.0], t_span=t_span, t_eval=t_eval
        )
    else:
        # Two-phase run: fentanyl bolus first, naloxone bolus injected at
        # t_split by restarting integration with naloxone added to state.
        t_split = t_span[1] / 2
        t_eval_1 = np.linspace(t_span[0], t_split, 200)
        phase1 = sim.run(
            dose_concentrations=[dose_fentanyl, 0.0],
            t_span=(t_span[0], t_split),
            t_eval=t_eval_1,
        )
        C0 = phase1["concentrations"][:, -1].copy()
        C0[1] += dose_naloxone_delayed
        occ0 = phase1["occupancy"][:, -1]
        rates0 = phase1["rates"][:, -1]
        y0 = np.concatenate([C0, occ0, rates0])
        t_eval_2 = np.linspace(t_split, t_span[1], 200)
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            sim.rhs, (t_split, t_span[1]), y0, method="RK45", t_eval=t_eval_2,
            max_step=0.1,
        )
        d, r = sim.d, sim.r
        phase2 = {
            "t": sol.t,
            "concentrations": sol.y[0:d],
            "occupancy": sol.y[d : d + r],
            "rates": sol.y[d + r : d + r + sim.c],
        }
        result = {
            "t": np.concatenate([phase1["t"], phase2["t"]]),
            "concentrations": np.concatenate(
                [phase1["concentrations"], phase2["concentrations"]], axis=1
            ),
            "occupancy": np.concatenate(
                [phase1["occupancy"], phase2["occupancy"]], axis=1
            ),
            "rates": np.concatenate([phase1["rates"], phase2["rates"]], axis=1),
        }

    behavior = np.array(
        [readout.score(network, result["rates"][:, i]) for i in range(len(result["t"]))]
    )
    return result, behavior, network


def main():
    results = {}
    results["fentanyl alone"] = run_scenario(dose_fentanyl=1.0)
    results["naloxone alone"] = run_scenario(dose_fentanyl=0.0)
    results["fentanyl, naloxone reversal"] = run_scenario(
        dose_fentanyl=1.0, dose_naloxone_delayed=3.0
    )

    for label, (result, behavior, _network) in results.items():
        peak = behavior.max()
        final = behavior[-1]
        print(f"{label:35s} peak incentive_salience={peak:.3f}  final={final:.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, (result, behavior, _network) in results.items():
            ax.plot(result["t"], behavior, label=label)
        ax.set_xlabel("time")
        ax.set_ylabel("incentive_salience (behavior readout)")
        ax.set_title("neurocascade v0: mu-opioid reward circuit")
        ax.legend()
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "fentanyl_reward_circuit.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nplot saved to {os.path.abspath(out_path)}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
