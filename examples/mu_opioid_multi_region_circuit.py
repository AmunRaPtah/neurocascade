"""Multi-region mu-opioid receptor density map (Track C1): one receptor,
three simultaneous behavioral effects.

Runs the SAME fentanyl/naloxone receptor pair from
examples/fentanyl_reward_circuit.py, but coupled into three nodes at three
different, individually-justified gains instead of one:
  - VTA_GABA (reward): unchanged from the original reward-circuit scenario
  - PAG_Interneuron (analgesia): PAG descending-analgesia local circuit
  - LC (sedation/arousal): direct receptor action on LC noradrenergic tone

Runs three scenarios end to end (PK -> receptor occupancy -> circuit rates ->
behavior scores) and plots all three readouts together:
  1. fentanyl alone: agonist, should raise incentive_salience AND
     analgesia_index, and lower arousal_index, all at once, from one shared
     receptor occupancy trace
  2. naloxone alone: antagonist, should do ~nothing on its own
  3. fentanyl, then naloxone: antagonist given after the agonist reverses
     ALL THREE effects simultaneously, because they all come from the same
     receptor pool naloxone competes at, not three independent drug actions

See scenarios.py for the full mechanistic writeup and literature citations
on relative mu-opioid receptor density across these regions, and
docs/architecture.md for why this needed no engine.py/network.py extension:
the existing `Simulation(coupling={...})` dict already generalizes to
multiple targets from one receptor.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/mu_opioid_multi_region_circuit.py
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, scenarios  # noqa: E402


def build_sim():
    return Simulation(
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        network=scenarios.multi_region_mu_opioid_network(),
        coupling=scenarios.mu_opioid_multi_region_coupling(),
    )


def run_scenario(dose_fentanyl, dose_naloxone_delayed=None, t_span=(0, 40)):
    sim = build_sim()
    network = sim.network
    readouts = {
        "incentive_salience": scenarios.incentive_salience_readout(),
        "analgesia_index": scenarios.analgesia_index_readout(),
        "arousal_index": scenarios.arousal_index_readout(),
    }

    if dose_naloxone_delayed is None:
        t_eval = np.linspace(t_span[0], t_span[1], 400)
        result = sim.run(
            dose_concentrations=[dose_fentanyl, 0.0], t_span=t_span, t_eval=t_eval
        )
    else:
        # Two-phase run: fentanyl bolus first, naloxone bolus injected at
        # t_split by restarting integration with naloxone added to state
        # (mirrors examples/fentanyl_reward_circuit.py's reversal pattern).
        t_split = t_span[1] / 2
        t_eval_1 = np.linspace(t_span[0], t_split, 200)
        phase1 = sim.run(
            dose_concentrations=[dose_fentanyl, 0.0],
            t_span=(t_span[0], t_split),
            t_eval=t_eval_1,
        )
        # A second administration is added to the state vector through the
        # PK layer itself (`Simulation.add_dose`), not by bumping a plasma
        # concentration by hand: for a dose-administered extravascular PK
        # model the new dose enters a depot compartment, and only the model
        # knows that (see src/neurocascade/pk.py).
        pk_state = phase1["pk_states"][:, -1]
        occ0 = phase1["occupancy"][:, -1]
        rates0 = phase1["rates"][:, -1]
        y0 = sim.add_dose(
            np.concatenate([pk_state, occ0, rates0]), [0.0, dose_naloxone_delayed]
        )
        t_eval_2 = np.linspace(t_split, t_span[1], 200)

        sol = solve_ivp(
            sim.rhs, (t_split, t_span[1]), y0, method="RK45", t_eval=t_eval_2,
            max_step=0.1,
        )
        phase2 = sim.unpack_solution(sol)
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

    behaviors = {
        name: np.array(
            [readout.score(network, result["rates"][:, i]) for i in range(len(result["t"]))]
        )
        for name, readout in readouts.items()
    }
    return result, behaviors, network


def main():
    results = {}
    results["fentanyl alone"] = run_scenario(dose_fentanyl=1.0)
    results["naloxone alone"] = run_scenario(dose_fentanyl=0.0)
    results["fentanyl, naloxone reversal"] = run_scenario(
        dose_fentanyl=1.0, dose_naloxone_delayed=3.0
    )

    for label, (result, behaviors, _network) in results.items():
        print(f"{label}:")
        for name, series in behaviors.items():
            print(
                f"    {name:20s} peak={series.max():.3f}  "
                f"trough={series.min():.3f}  final={series[-1]:.3f}"
            )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
        readout_names = ["incentive_salience", "analgesia_index", "arousal_index"]
        for ax, name in zip(axes, readout_names):
            for label, (result, behaviors, _network) in results.items():
                ax.plot(result["t"], behaviors[name], label=label)
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("time")
        fig.suptitle(
            "neurocascade: multi-region mu-opioid receptor density map\n"
            "(one receptor, three simultaneous behavioral effects)"
        )
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            "mu_opioid_multi_region_circuit.png",
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nplot saved to {os.path.abspath(out_path)}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
