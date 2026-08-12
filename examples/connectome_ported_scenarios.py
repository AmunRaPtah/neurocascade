"""P3.3: every hand-built scenario, re-run with published connectome weights.

For each of the four scenarios this prints the port's provenance report
(which weights are published, which are still illustrative and why, which
hand-drawn arrows the tracer data contradicts) and plots the hand-built and
ported behavioural readouts on the same axes.

Three of the four reproduce their qualitative result. The D2 basal-ganglia
indirect pathway does not, and the bottom-right panel is what that failure
looks like: the locomotor readout does not move at all. `docs/architecture.md`,
"Phase 3: porting the hand-built scenarios onto connectome regions (P3.3)",
has the diagnosis.

Runs off the committed fixtures by default, so it needs no download; the
fifteen fixture regions carry exactly the same numbers the 291-region files
do for every pair a scenario uses. Pass --cache to read the full files.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/connectome_ported_scenarios.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import connectome_scenarios as cs  # noqa: E402

T_SPAN = (0.0, 40.0)
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 400)


def behaviour(port, network, readout, doses):
    sim = port.simulation(network)
    result = sim.run(dose_concentrations=doses, t_span=T_SPAN, t_eval=T_EVAL)
    return np.array(
        [readout.score(network, result["rates"][:, i]) for i in range(len(T_EVAL))]
    )


def main():
    source = "cache" if "--cache" in sys.argv else "fixture"
    circuit = cs.scenario_connectome(source=source)
    print("connectome: %s, %d regions\n" % (circuit.report.dataset_key, circuit.n))

    panels = []
    for key in sorted(cs.SCENARIO_PORTS):
        port = cs.SCENARIO_PORTS[key]
        ported = cs.port_scenario(port, circuit)
        print(ported.report.summary())
        name = sorted(port.readouts)[0]
        readout = port.readouts[name][0]()
        rows = []
        for label, network in (
            ("hand-built", port.hand_built()),
            ("ported", ported.network),
        ):
            baseline = behaviour(port, network, readout, port.doses())
            agonist = behaviour(port, network, readout, port.doses(agonist=1.0))
            rows.append((label, baseline, agonist))
            print(
                "  %-10s %-20s baseline final=%.3f  agonist final=%.3f  peak=%.3f"
                % (label, name, baseline[-1], agonist[-1], agonist.max())
            )
        print()
        panels.append((port, name, rows))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (port, name, rows) in zip(axes.ravel(), panels):
        for label, baseline, agonist in rows:
            style = "-" if label == "ported" else "--"
            ax.plot(T_EVAL, baseline, style, color="0.6", label="%s, no drug" % label)
            ax.plot(T_EVAL, agonist, style, label="%s, agonist" % label)
        ax.set_title("%s\n%s" % (port.key, name), fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("time")
    fig.suptitle(
        "neurocascade P3.3: hand-built (dashed) vs connectome-ported (solid) weights"
    )
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "connectome_ported_scenarios.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("plot saved to %s" % os.path.abspath(out_path))


if __name__ == "__main__":
    main()
