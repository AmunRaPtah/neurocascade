"""D2 dopamine agonism/antagonism in a basal ganglia indirect-pathway circuit.

Runs two scenarios end to end (PK -> receptor occupancy -> circuit rates ->
behavior score) and plots them together:
  1. dopamine alone: agonist at D2, should raise locomotor activity
     (disinhibition of the indirect "no-go" pathway facilitates
     thalamocortical/motor output)
  2. haloperidol alone: antagonist, should do ~nothing on its own (there's no
     endogenous dopamine tone in this bolus-only model for it to block)

See scenarios.py for the full mechanistic rationale (D2 is Gi-coupled on
striatal indirect-pathway MSNs, so this reuses the same disinhibition motif
as the mu-opioid reward-circuit example, just relayed through one more
basal-ganglia synapse). All parameters are illustrative placeholders, not
literature-calibrated values.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/d2_basal_ganglia_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, scenarios  # noqa: E402


def run_scenario(dose_dopamine, dose_haloperidol=0.0, t_span=(0, 40)):
    receptor_site = scenarios.d2_receptor_site()
    pk_models = scenarios.d2_pk_models()
    network = scenarios.basal_ganglia_indirect_pathway_network()
    coupling = scenarios.d2_coupling()
    readout = scenarios.locomotor_activity_readout()

    sim = Simulation(
        pk_models=pk_models, receptor_site=receptor_site, network=network,
        coupling=coupling,
    )

    t_eval = np.linspace(t_span[0], t_span[1], 400)
    result = sim.run(
        dose_concentrations=[dose_dopamine, dose_haloperidol],
        t_span=t_span,
        t_eval=t_eval,
    )
    behavior = np.array(
        [readout.score(network, result["rates"][:, i]) for i in range(len(result["t"]))]
    )
    return result, behavior, network


def main():
    results = {}
    results["dopamine alone"] = run_scenario(dose_dopamine=1.0)
    results["haloperidol alone"] = run_scenario(dose_dopamine=0.0, dose_haloperidol=1.0)

    for label, (result, behavior, _network) in results.items():
        peak = behavior.max()
        final = behavior[-1]
        print(f"{label:25s} peak locomotor_activity={peak:.3f}  final={final:.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for label, (result, behavior, _network) in results.items():
            ax.plot(result["t"], behavior, label=label)
        ax.set_xlabel("time")
        ax.set_ylabel("locomotor_activity (behavior readout)")
        ax.set_title("neurocascade: D2 basal ganglia indirect-pathway circuit")
        ax.legend()
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "d2_basal_ganglia_circuit.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nplot saved to {os.path.abspath(out_path)}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
