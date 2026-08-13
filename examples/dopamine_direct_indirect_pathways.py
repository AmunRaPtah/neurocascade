"""D1 and D2 on one cortico-basal-ganglia-thalamic circuit (P5.2).

Runs six conditions end to end (PK -> receptor occupancy -> circuit rates
-> readout) on ONE seven-node network carrying BOTH the direct and the
indirect pathway, with two `ReceptorSystem`s (D1 on the striatonigral MSN,
D2 on the striatopallidal one) integrated in one `solve_ivp` call:

  1. no drug: every node rests at 0.5, the maximum-sensitivity point.
  2. dopamine alone: engages D2 almost completely (Kd 2.03 nM) and D1
     partially (Kd 1079 nM), and both arms release the thalamus.
  3. dopamine + dihydrexidine: a D1 agonist adds direct-pathway drive.
     Locomotor readout UP relative to dopamine alone.
  4. dopamine + haloperidol: a D2 antagonist takes the indirect pathway's
     disinhibition away. Locomotor readout DOWN relative to dopamine alone.
  5. dopamine + both: the two opposing drugs partly cancel, which is what
     one shared output nucleus buys and two separate models could not
     produce.
  6. dihydrexidine, then SCH-23390 later: competitive reversal at the
     shared D1 pool. SCH-23390 wins by residence time (derived koff
     0.00389 min^-1, t1/2 178 min) rather than by molar excess, the same
     shape as P5.1's WAY-100635 and the opposite of naloxone/fentanyl.

Every D1 and D2 binding constant and efficacy is literature-sourced; the
circuit and the tool compounds' pharmacokinetics are illustrative. See
`dopamine.py` for the sources, the derivations, and the three rules that
fix the illustrative values.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/dopamine_direct_indirect_pathways.py
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, dopamine  # noqa: E402

T_END = 40.0
T_SPLIT = 20.0


def build_sim():
    return Simulation(
        network=dopamine.cortico_basal_ganglia_network(),
        receptor_systems=dopamine.dopamine_receptor_systems(),
    )


def run_condition(first_dose, second_dose=None):
    """One condition, optionally with a second drug given at T_SPLIT.

    Every run starts at `rest_rates()`, the fixed point the network's
    baselines were built to sit at, so the drug-free trace is exactly flat
    and every deviation below is the drug rather than a start-up transient.
    """
    sim = build_sim()
    rates0 = dopamine.rest_rates(sim.network)
    if second_dose is None:
        return sim, sim.run(
            doses=sim.dose_vector(**first_dose),
            t_span=(0.0, T_END),
            t_eval=np.linspace(0.0, T_END, 500),
            rates0=rates0,
        )

    phase1 = sim.run(
        doses=sim.dose_vector(**first_dose),
        t_span=(0.0, T_SPLIT),
        t_eval=np.linspace(0.0, T_SPLIT, 250),
        rates0=rates0,
    )
    y = np.concatenate(
        [
            phase1["pk_states"][:, -1],
            phase1["occupancy"][:, -1],
            phase1["rates"][:, -1],
        ]
    )
    y = sim.add_dose(y, sim.dose_vector(**second_dose))
    solution = solve_ivp(
        sim.rhs,
        (T_SPLIT, T_END),
        y,
        method="RK45",
        t_eval=np.linspace(T_SPLIT, T_END, 250),
        max_step=0.1,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    phase2 = sim.unpack_solution(solution)
    joined = {
        key: np.concatenate([phase1[key], phase2[key]], axis=-1)
        for key in ("t", "concentrations", "occupancy", "rates")
    }
    return sim, joined


def scores(sim, result, readout):
    return np.array(
        [
            readout.score(sim.network, result["rates"][:, i])
            for i in range(result["rates"].shape[1])
        ]
    )


def main():
    conditions = [
        ("no drug", {}, None),
        ("dopamine 1000 nM", {"dopamine": 1000.0}, None),
        (
            "dopamine + dihydrexidine (D1 agonist)",
            {"dopamine": 1000.0, "dihydrexidine": 1000.0},
            None,
        ),
        (
            "dopamine + haloperidol 200 nM (D2 antagonist)",
            {"dopamine": 1000.0, "haloperidol": 200.0},
            None,
        ),
        (
            "dopamine + both",
            {"dopamine": 1000.0, "dihydrexidine": 1000.0, "haloperidol": 200.0},
            None,
        ),
        (
            "dihydrexidine, then SCH-23390 at t=20",
            {"dihydrexidine": 300.0},
            {"SCH-23390": 1000.0},
        ),
    ]
    locomotion = dopamine.locomotor_activity_readout()
    direct = dopamine.direct_pathway_readout()
    indirect = dopamine.indirect_pathway_readout()
    output = dopamine.basal_ganglia_output_readout()

    traces = []
    for label, first, second in conditions:
        sim, result = run_condition(first, second)
        locomotor_trace = scores(sim, result, locomotion)
        traces.append(
            (
                label,
                result["t"],
                locomotor_trace,
                scores(sim, result, direct),
                scores(sim, result, indirect),
            )
        )
        print(
            "%-40s locomotion %.4f (peak %.4f)  direct %.4f  indirect %.4f  output %.4f"
            % (
                label,
                locomotor_trace[-1],
                locomotor_trace.max(),
                scores(sim, result, direct).max(),
                scores(sim, result, indirect).min(),
                scores(sim, result, output).min(),
            )
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; numbers only)")
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for label, t, locomotor_trace, direct_trace, indirect_trace in traces:
        axes[0].plot(t, locomotor_trace, label=label)
        axes[1].plot(t, direct_trace, label=label)
        axes[2].plot(t, indirect_trace, label=label)
    axes[0].set_ylabel("locomotor_activity (Thalamus rate)")
    axes[1].set_ylabel("direct pathway (D1 MSN rate)")
    axes[2].set_ylabel("indirect pathway (D2 MSN rate)")
    axes[2].set_xlabel("time (min)")
    axes[0].legend(fontsize=8)
    fig.suptitle("D1 direct and D2 indirect pathways on one network (P5.2)")
    out = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "dopamine_direct_indirect_pathways.png",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
