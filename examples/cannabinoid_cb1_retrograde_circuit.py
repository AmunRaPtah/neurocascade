"""Presynaptic CB1 as connection modulation, on a cortical E-I loop (P5.6b).

Prints five conditions end to end (PK -> receptor occupancy -> EFFECTIVE
CONNECTIVITY -> circuit rates -> readouts) for the same two-node loop P5.5
uses, with a CB1 system on the GABAergic terminal.

What this example exists to show is why CB1 could not be a `coupling` entry.
CB1 is presynaptic: it sits on the axon terminal of the interneuron
([Katona1999]), not on the neuron whose rate changes, and the signal that
activates it travels BACKWARDS across the synapse ([Wilson2001]). A
`coupling` entry has one index, the postsynaptic node, and no way to name
which input it acts on. `connection_modulation` has a (post, pre) PAIR and a
multiplicative effect on that projection's weight, which is what presynaptic
inhibition of release IS in a rate model.

  0. no drug: the effective weight is exactly the base weight and both nodes
     sit exactly at 0.5. This module changed no earlier result.
  1. 2-AG: the pyramidal node rises. Inhibition is weakened, so the same
     interneuron firing delivers less of it. The interneuron then rises too,
     but LATER and only through the loop, because presynaptic inhibition of
     release does not change the presynaptic cell's own firing.
  2. anandamide: the same direction, smaller, and the difference is
     efficacy rather than affinity. The two endocannabinoids' compiled
     affinities are within 1.1-fold; their measured efficacies differ
     1.6-fold.
  3. THC: smaller again, because THC is a weak partial agonist (Emax 20% of
     a full agonist, [Cheng2012]).
  4. rimonabant alone: nothing. For once that is not only the standing
     no-endogenous-tone limitation: [HoffmanLupica2000] applied SR141716A
     alone to the same slices and measured 115 +/- 17% of control, because
     an acute slice has no ongoing endocannabinoid release to block.

The gain is a MEASUREMENT, which makes this the second Phase 5 module with
no illustrative coupling gain (after P5.5, and by a different route).
[HoffmanLupica2000] measured a saturating CB1 agonist to remove 47 +/- 4% of
an evoked GABA-A IPSC in rat CA1, so the ceiling printed in the last block
is not a modelling choice: a saturating agonist here removes 47% of the
inhibitory weight and no more.

Every affinity, efficacy and the gain are literature-sourced; the circuit,
the association rate constant (no CB1 binding kinetics exist in reach) and
the pharmacokinetics are illustrative and labelled. See `cannabinoid.py` for
the sources, for the fact that the published affinity ranges do NOT resolve
most of this module's ligand orderings, and for the CB1 population on
glutamatergic terminals that is deliberately not built.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/cannabinoid_cb1_retrograde_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import cannabinoid as cb  # noqa: E402
from neurocascade import stability  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402


T_END = 60.0
SATURATING = 10000000.0  # nM: >20000x the weakest published Kd in this module

CONDITIONS = (
    ("no drug", {}),
    ("2-AG (full agonist)", {"2ag": SATURATING}),
    ("anandamide (partial)", {"anandamide": SATURATING}),
    ("THC (weak partial)", {"thc": SATURATING}),
    ("rimonabant alone", {"rimonabant": SATURATING}),
)


def run(sim, network, **doses):
    return sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=np.linspace(0.0, T_END, 241),
        rates0=cb.rest_rates(network),
        method="LSODA",
    )


def main():
    network = cb.cortical_network()
    sim = Simulation(
        network=network, receptor_systems=cb.cannabinoid_receptor_systems()
    )
    site = sim.receptor_systems[0].receptor_site

    print("CB1 as connection modulation on a cortical E-I loop (P5.6b)")
    print("=" * 78)
    print("affinities (nM), default 'midpoint' of the published ranges:")
    for lig in site.ligands:
        low, high = cb.CB1_KD_BOUNDS[lig.name]
        print(
            "  %-12s Kd %8.2f  (published %6.2f - %6.1f)  efficacy %.2f"
            % (lig.name, lig.koff / lig.kon, low, high, lig.efficacy)
        )
    print()
    print(
        "modulated connection %s, gain %.2f  [HoffmanLupica2000], MEASURED"
        % (cb.GABAERGIC_CONNECTION, cb.CB1_CONNECTION_GAIN)
    )
    print(
        "the OTHER connection of this loop, %s, is glutamatergic and is P5.5's"
        % (cb.GLUTAMATERGIC_CONNECTION,)
    )
    print()
    print(
        "%-24s %9s %11s %10s %10s"
        % ("condition", "activation", "W_eff(inh)", "PFC_Pyr", "PFC_GABA")
    )
    print("-" * 78)

    for label, doses in CONDITIONS:
        result = run(sim, network, **doses)
        occupancy = result["occupancy"]
        activation = np.array(
            [site.activation(occupancy[:, k]) for k in range(occupancy.shape[1])]
        )
        peak_activation = activation.max()
        print(
            "%-24s %9.4f %11.4f %10.4f %10.4f"
            % (
                label,
                peak_activation,
                cb.modulated_weight(peak_activation),
                result["rates"][0].max(),
                result["rates"][1].max(),
            )
        )

    print()
    print("Presynaptic means the presynaptic node's own rate is not what moves")
    print("-" * 78)
    result = run(sim, network, **{"2ag": SATURATING})
    times = result["t"]
    pyramidal = result["rates"][0] - 0.5
    interneuron = result["rates"][1] - 0.5
    print(
        "  PFC_Pyramidal peaks at t = %5.2f min (+%.4f)  <- the disinhibited node"
        % (times[np.argmax(pyramidal)], pyramidal.max())
    )
    print(
        "  PFC_GABA      peaks at t = %5.2f min (+%.4f)  <- follows, through the "
        "loop only" % (times[np.argmax(interneuron)], interneuron.max())
    )
    print(
        "  The receptor is on PFC_GABA's TERMINAL. Its cell body's firing changes\n"
        "  only because the pyramidal node that excites it has risen, which is\n"
        "  exactly why this receptor needed connection_modulation and not coupling."
    )

    print()
    print("The measured ceiling")
    print("-" * 78)
    W = network.connectivity.copy()
    W[0, 1] = cb.modulated_weight(1.0)
    bounded = type(network)(nodes=network.nodes, connectivity=W)
    ceiling, residual = stability.steady_state(bounded)
    print(
        "  base inhibitory weight                       %+.4f"
        % cb.CORTICAL_INHIBITION_WEIGHT
    )
    print(
        "  at full activation by a full agonist          %+.4f  (53%% retained)"
        % cb.modulated_weight(1.0)
    )
    print(
        "  pyramidal rate at that ceiling                 %.4f  (residual %.1e)"
        % (ceiling[0], residual)
    )
    print(
        "  A saturating CB1 agonist in this model cannot do better than that,\n"
        "  because 47 +/- 4% is what was measured (rat CA1, 5 uM WIN 55,212-2,\n"
        "  n = 12, at 36x its own EC50 of 138 nM)."
    )

    print()
    print("What the published affinities do NOT settle")
    print("-" * 78)
    for tighter, weaker in (
        ("rimonabant", "2ag"),
        ("rimonabant", "anandamide"),
        ("rimonabant", "thc"),
        ("thc", "anandamide"),
    ):
        resolved = cb.affinity_ordering_is_resolved(tighter, weaker)
        print(
            "  %-12s tighter than %-12s ? %-14s (midpoints %7.2f vs %7.2f nM)"
            % (
                tighter,
                weaker,
                "yes" if resolved else "NOT RESOLVED",
                cb.cb1_kd(tighter),
                cb.cb1_kd(weaker),
            )
        )
    print(
        "  The midpoints look decisive and the ranges overlap. Only rimonabant's\n"
        "  two orderings survive every combination of published values, so the\n"
        "  model states only those."
    )


if __name__ == "__main__":
    main()
