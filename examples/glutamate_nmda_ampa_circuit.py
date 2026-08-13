"""NMDA and AMPA antagonism on a cortical E-I loop (P5.5).

Prints five conditions end to end (PK -> receptor state -> effective
connectivity -> circuit rates -> readouts) for the same two-node cortical
loop P5.1, P5.3 and P5.4 use, now carrying an NMDA and an AMPA system.

What this example exists to show is the thing that makes glutamate
different from every receptor system before it: its receptors do not ADD
drive to a node, they take a share of a connectivity weight that is already
there. Fast glutamatergic transmission has been in every simulation this
codebase has ever run, as the positive entries of W. So the glutamate
systems here have EMPTY `coupling` and act only through
`connection_modulation`, and the numbers printed under "effective weight"
are the whole story:

  0. no drug: the effective weight is exactly the base weight and both
     nodes sit exactly at 0.5. Adding this module changed no earlier result,
     and this line is what says so.
  1. ketamine: the pyramidal node's firing goes UP and the interneuron's
     goes DOWN, which is [Homayoun2007]'s direction in awake rats, and the
     interneuron moves first. Nothing is fitted to that result. It follows
     from the fact that the only glutamatergic connection in this loop is
     the one ONTO the interneuron; the return connection is GABAergic and
     the drug does not touch it.
  2. MK-801: the drug [Homayoun2007] actually used, same direction, and the
     only ligand in this codebase with BOTH rate constants measured.
  3. NBQX: the same circuit through the other receptor, and an order of
     magnitude larger, because at any physiological membrane potential the
     Mg2+ block leaves AMPA receptors carrying ~90% of the weight.
  4. ketamine + NBQX: the two systems' modulation factors compose
     multiplicatively, which is `engine.Simulation`'s documented rule and
     the one place it is load-bearing here.

The final block sweeps the operating membrane potential. It is the module's
most load-bearing prediction and its most honest one: this model has no
membrane potential, so the voltage at which the Mg2+ block is evaluated is
the module's one illustrative constant, and ketamine's leverage on the
circuit is a 27-fold function of it. At the resting potential
[ChiuCarter2022] measured in the same cells, NMDA carries under 1% of the
weight and ketamine does almost nothing.

Every affinity here is literature-sourced, as is the Mg2+ block, the
NMDA:AMPA ratio and MK-801's rate pair; the circuit, the operating voltage,
the orthosteric association rate constant and the pharmacokinetics are
illustrative and labelled. See `glutamate.py` for the sources, the
double-counting argument in full, and the 650-fold error this module
carries in MK-801's unblocking rate.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/glutamate_nmda_ampa_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import glutamate as glu  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402

# Illustrative bolus concentrations, in nM, like every tool compound in this
# codebase that has no ingested PK parameter set.
KETAMINE_nM = 50000.0
MK801_nM = 1000.0
NBQX_nM = 1000.0

T_END = 60.0


def run(sim, network, **doses):
    return sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=np.linspace(0.0, T_END, 241),
        rates0=glu.rest_rates(network),
        # A measured association rate constant makes this system stiff; see
        # `engine.Simulation.run`. Same answer as RK45, ~1000x faster.
        method="LSODA",
    )


def describe(sim, network, label, result):
    print("\n%s" % label)
    i, j = (network.index[name] for name in glu.GLUTAMATERGIC_CONNECTION)
    base_weight = network.connectivity[i, j]

    factor = 1.0
    for index, system in enumerate(sim.receptor_systems):
        site = system.receptor_site
        state = result["occupancy"][sim.state_rows(index)][:, -1]
        removed = site.removed_weight_fraction(state)
        factor *= 1.0 - removed
        print(
            "  %-5s share of weight %.3f   blocked %.3f   weight removed %.3f"
            % (
                system.name,
                site.synaptic_share,
                site.blocked_transmission_fraction(state),
                removed,
            )
        )
    print(
        "  effective weight %s -> %.4f  (base %.4f)"
        % (glu.GLUTAMATERGIC_CONNECTION, base_weight * factor, base_weight)
    )

    pyramidal = result["rates"][network.index["PFC_Pyramidal"]]
    interneuron = result["rates"][network.index["PFC_GABA"]]
    print(
        "  PFC_Pyramidal %.4f -> %.4f (peak %+.4f)   "
        "PFC_GABA %.4f -> %.4f (peak %+.4f)"
        % (
            pyramidal[0],
            pyramidal[-1],
            (pyramidal - 0.5)[np.argmax(np.abs(pyramidal - 0.5))],
            interneuron[0],
            interneuron[-1],
            (interneuron - 0.5)[np.argmax(np.abs(interneuron - 0.5))],
        )
    )


def main():
    network = glu.cortical_network()
    sim = Simulation(
        network=network, receptor_systems=glu.glutamatergic_receptor_systems()
    )

    print(
        "operating potential %.1f mV, ionic condition %r"
        % (glu.ILLUSTRATIVE_OPERATING_VOLTAGE, glu.DEFAULT_IONIC_CONDITION)
    )
    print(
        "NMDA carries %.1f%% of an excitatory weight, AMPA %.1f%%"
        % (100.0 * glu.nmda_share(), 100.0 * glu.ampa_share())
    )

    describe(sim, network, "0. no drug", run(sim, network))
    describe(sim, network, "1. ketamine", run(sim, network, ketamine=KETAMINE_nM))
    describe(sim, network, "2. MK-801", run(sim, network, mk801=MK801_nM))
    describe(sim, network, "3. NBQX", run(sim, network, nbqx=NBQX_nM))
    describe(
        sim,
        network,
        "4. ketamine + NBQX",
        run(sim, network, ketamine=KETAMINE_nM, nbqx=NBQX_nM),
    )

    print("\nketamine's effect against the operating membrane potential:")
    print("  V (mV)   unblocked   NMDA share   peak change in pyramidal rate")
    for voltage in (-82.7, -70.0, -60.0, -50.0, -40.0, -29.5, -10.0, 0.0):
        systems = [glu.receptor_system_nmda(v_mV=voltage)]
        one = Simulation(network=network, receptor_systems=systems)
        result = run(one, network, ketamine=KETAMINE_nM)
        deviation = result["rates"][network.index["PFC_Pyramidal"]] - 0.5
        print(
            "  %6.1f   %9.4f   %10.4f   %+.5f"
            % (
                voltage,
                float(glu.mg_unblocked_fraction(voltage)),
                glu.nmda_share(voltage),
                deviation[np.argmax(np.abs(deviation))],
            )
        )
    print(
        "\n  (the top row is the resting potential [ChiuCarter2022] measured\n"
        "   in the same cells: at rest an excitatory weight is essentially\n"
        "   pure AMPA and an NMDA antagonist has almost no circuit effect.)"
    )


if __name__ == "__main__":
    main()
