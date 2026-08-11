"""Benzodiazepine allosteric modulation of GABA-A in an amygdala anxiety circuit.

Runs three scenarios end to end (PK -> receptor occupancy -> circuit rates ->
behavior score) and plots them together:
  1. diazepam alone: positive allosteric modulator, should lower the
     anxiety_index (potentiates the GABAergic interneuron -> BLA synapse,
     suppressing amygdala output)
  2. flumazenil alone: competes with diazepam at the benzodiazepine site but
     has no potentiation of its own, so it should do ~nothing on its own
  3. diazepam, then flumazenil given later: flumazenil displaces diazepam
     from the shared benzodiazepine-site pool, partially reversing the
     anxiolytic effect. This is the same reversal mechanism real flumazenil
     dosing relies on for benzodiazepine overdose, and the same reason it
     needs repeat dosing: flumazenil's own clearance is fast relative to
     diazepam's, so the reversal fades and diazepam's effect creeps back.

See scenarios.py for the full mechanistic writeup of why this uses
`Simulation(..., connection_modulation=...)` -- an allosteric modulator
potentiates an existing GABAergic synapse rather than driving a node
directly the way mu-opioid/D2 do, so it doesn't fit the plain `coupling`
mechanism those scenarios use. All parameters are illustrative placeholders,
not literature-calibrated values.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/gaba_a_benzodiazepine_circuit.py
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, scenarios  # noqa: E402


def build_sim():
    return Simulation(
        pk_models=scenarios.gaba_a_pk_models(),
        receptor_site=scenarios.gaba_a_receptor_site(),
        network=scenarios.amygdala_anxiety_circuit_network(),
        coupling=scenarios.gaba_a_coupling(),
        connection_modulation=scenarios.gaba_a_connection_modulation(),
    )


def run_scenario(dose_diazepam, dose_flumazenil_delayed=None, t_span=(0, 40)):
    sim = build_sim()
    network = sim.network
    readout = scenarios.anxiety_index_readout()

    if dose_flumazenil_delayed is None:
        t_eval = np.linspace(t_span[0], t_span[1], 400)
        result = sim.run(
            dose_concentrations=[dose_diazepam, 0.0], t_span=t_span, t_eval=t_eval
        )
    else:
        # Two-phase run: diazepam bolus first, flumazenil bolus injected at
        # t_split by restarting integration with flumazenil added to state
        # (mirrors examples/fentanyl_reward_circuit.py's reversal pattern).
        t_split = t_span[1] / 2
        t_eval_1 = np.linspace(t_span[0], t_split, 200)
        phase1 = sim.run(
            dose_concentrations=[dose_diazepam, 0.0],
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
            np.concatenate([pk_state, occ0, rates0]), [0.0, dose_flumazenil_delayed]
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

    behavior = np.array(
        [readout.score(network, result["rates"][:, i]) for i in range(len(result["t"]))]
    )
    return result, behavior, network


def main():
    results = {}
    results["diazepam alone"] = run_scenario(dose_diazepam=1.0)
    results["flumazenil alone"] = run_scenario(dose_diazepam=0.0)
    results["diazepam, flumazenil reversal"] = run_scenario(
        dose_diazepam=1.0, dose_flumazenil_delayed=3.0
    )

    for label, (result, behavior, _network) in results.items():
        # Skip the startup transient (rates0 is always 0, and this circuit's
        # two-stage relaxation briefly undershoots on the way to steady
        # state) when reporting the trough; it's shared across every
        # scenario and not informative about the drug's actual effect.
        settled = behavior[result["t"] > 10.0]
        trough = settled.min()
        final = behavior[-1]
        print(f"{label:35s} trough anxiety_index={trough:.3f}  final={final:.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, (result, behavior, _network) in results.items():
            ax.plot(result["t"], behavior, label=label)
        ax.set_xlabel("time")
        ax.set_ylabel("anxiety_index (behavior readout)")
        ax.set_title("neurocascade: GABA-A benzodiazepine amygdala circuit")
        ax.legend()
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "gaba_a_benzodiazepine_circuit.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nplot saved to {os.path.abspath(out_path)}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
