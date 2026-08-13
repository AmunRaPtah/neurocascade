"""5-HT1A and 5-HT2A on one raphe/cortex circuit (P5.1).

Runs five conditions end to end (PK -> receptor occupancy -> circuit rates
-> readout) and plots them:

  1. 8-OH-DPAT alone: 5-HT1A agonist. Suppresses the raphe node through the
     somatodendritic autoreceptor, and the cortical node through the
     postsynaptic 5-HT1A receptor on the same pyramidal cells.
  2. WAY-100635 alone: silent 5-HT1A antagonist, nothing to displace in a
     model with no endogenous serotonin tone, so it does nothing. Real
     WAY-100635 is not inert in vivo; supplying the tone it would act on is
     Phase 6 (P6.2).
  3. 8-OH-DPAT, then WAY-100635 later: displacement at the shared 5-HT1A
     pool. WAY-100635 wins by residence time rather than by molar excess:
     its measured koff is 0.0078 min^-1 against 8-OH-DPAT's 0.986, so once
     it is on it stays on. That is the opposite of the naloxone/fentanyl
     reversal in `fentanyl_reward_circuit.py`, where the antagonist is the
     faster-dissociating ligand and needs a large excess to work.
  4. DOI alone: 5-HT2A agonist, Gq-coupled and excitatory on the cortical
     pyramidal node.
  5. DOI, then ketanserin later: the same reversal at the other receptor.

Both receptor systems drive ONE shared network in ONE `solve_ivp` call.
Every binding constant and efficacy is literature-sourced; the circuit and
the pharmacokinetics are illustrative. See `serotonin.py` for the sources,
the derivations, and the autoreceptor's Phase 6 dependency.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/serotonin_raphe_cortex_circuit.py
"""

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import Simulation, serotonin  # noqa: E402

T_END = 60.0
T_SPLIT = 30.0


def build_sim():
    return Simulation(
        network=serotonin.raphe_cortical_network(),
        receptor_systems=serotonin.serotonin_receptor_systems(),
    )


def run_condition(first_dose, second_dose=None):
    """One condition, optionally with a second drug given at T_SPLIT."""
    sim = build_sim()
    if second_dose is None:
        t_eval = np.linspace(0.0, T_END, 600)
        return sim, sim.run(
            doses=sim.dose_vector(**first_dose), t_span=(0.0, T_END), t_eval=t_eval
        )

    phase1 = sim.run(
        doses=sim.dose_vector(**first_dose),
        t_span=(0.0, T_SPLIT),
        t_eval=np.linspace(0.0, T_SPLIT, 300),
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
        t_eval=np.linspace(T_SPLIT, T_END, 300),
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
        ("8-OH-DPAT 100 nM", {"8-OH-DPAT": 100.0}, None),
        ("WAY-100635 100 nM alone", {"WAY-100635": 100.0}, None),
        (
            "8-OH-DPAT, then WAY-100635 at t=30",
            {"8-OH-DPAT": 100.0},
            {"WAY-100635": 1000.0},
        ),
        ("DOI 100 nM", {"DOI": 100.0}, None),
        ("DOI, then ketanserin at t=30", {"DOI": 100.0}, {"ketanserin": 1000.0}),
        ("no drug", {}, None),
    ]
    tone = serotonin.serotonergic_tone_readout()
    cortex = serotonin.cortical_output_readout()

    traces = []
    for label, first, second in conditions:
        sim, result = run_condition(first, second)
        traces.append((label, result["t"], scores(sim, result, tone),
                       scores(sim, result, cortex)))
        print(
            "%-38s serotonergic_tone %.4f -> %.4f   cortical_output %.4f -> %.4f"
            % (
                label,
                traces[-1][2][0],
                traces[-1][2][-1],
                traces[-1][3][0],
                traces[-1][3][-1],
            )
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; numbers only)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, t, tone_trace, cortex_trace in traces:
        axes[0].plot(t, tone_trace, label=label)
        axes[1].plot(t, cortex_trace, label=label)
    axes[0].set_ylabel("serotonergic_tone (DRN_5HT rate)")
    axes[1].set_ylabel("cortical_output (PFC_Pyramidal rate)")
    axes[1].set_xlabel("time (min)")
    axes[0].legend(fontsize=8)
    fig.suptitle("5-HT1A and 5-HT2A on one raphe/cortex circuit (P5.1)")
    out = os.path.join(
        os.path.dirname(__file__), "..", "results", "serotonin_raphe_cortex.png"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote %s" % os.path.normpath(out))


if __name__ == "__main__":
    main()
