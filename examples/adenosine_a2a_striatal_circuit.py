"""Adenosine A2A on the indirect pathway, and the caffeine problem (P5.6c).

Prints five conditions end to end (PK -> receptor occupancy -> node drive ->
circuit rates -> readouts) for the basal-ganglia indirect-pathway network the
D2 scenario has used since Track A, unchanged, now carrying an A2A system on
the node that already carries D2.

This module builds no circuit. [Schiffmann1991] found the striatal A2
receptor exclusively on enkephalinergic medium spiny neurons, i.e. on the
indirect pathway; [Ferre1993] found A2 and D2 co-localized on that same
striatopallidal neuron and acting on it in opposition. Of the five nodes in
`scenarios.basal_ganglia_indirect_pathway_network()`, A2A therefore belongs
on exactly one, and it is the one `scenarios.d2_coupling()` already names.

  0. no drug: exactly flat at the network's own fixed point.
  1. adenosine: the striatopallidal node rises (Gs/Golf), and the effect
     travels the four-synapse chain Striatum -| GPe -| STN -> GPi -|
     Thalamus, arriving at the thalamic readout INVERTED.
  2. caffeine alone: nothing at all. Caffeine's entire mechanism is the
     removal of an ongoing adenosine action, and this model has no ongoing
     action to remove (P6.2). For caffeine that is the difference between a
     model of the drug and a model of the receptor, and it is the largest
     honest gap in this module's scenario.
  3. adenosine + caffeine: reversed, at a dose 300-fold larger, because that
     is the published affinity ratio and it is not tunable.
  4. adenosine + dopamine, with the D2 system added: [Ferre1993]'s
     functional opposition, reproduced without asserting any molecular
     mechanism for it. No heteromer is modelled; see `adenosine.py`.

The last block is the honest headline. CAFFEINE IS NOT AN A2A DRUG: its
affinity is 41 uM at A1 and 45 uM at A2A in rat (1.1-fold apart), and in
human it PREFERS A1 by 2.2-fold. This model has no A1 receptor, so a caffeine
dose here produces the A2A-mediated half of caffeine's action and none of the
other. The direction is still the right one to test against, because
[Ledent1997] showed in A2A-knockout mice that caffeine does not merely lose
its stimulant effect but becomes a depressant.

Adenosine's own number carries a caveat of a different kind: it is a
FUNCTIONAL POTENCY, not a binding affinity, because adenosine's binding
affinity at its own receptors cannot be measured -- it is metabolized and
regenerated inside any preparation used to measure it, and [Fredholm2011]
says so in as many words.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/adenosine_a2a_striatal_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import adenosine as ad  # noqa: E402
from neurocascade import scenarios  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402


T_END = 60.0
ADENOSINE_BOLUS = 10000.0  # nM, illustrative
CAFFEINE_BOLUS = 1000000.0  # nM: caffeine's Kd here is 45 uM, so ~22x
DOPAMINE_BOLUS = 10000.0  # nM, illustrative

NODES = ("Striatum_D2_MSN", "GPe", "STN", "GPi", "Thalamus")


def run(sim, network, **doses):
    return sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=np.linspace(0.0, T_END, 241),
        rates0=ad.rest_rates(network),
        method="LSODA",
    )


def main():
    network = ad.striatal_network()
    a2a_only = Simulation(
        network=network, receptor_systems=ad.adenosine_receptor_systems()
    )
    with_d2 = Simulation(
        network=network,
        receptor_systems=ad.adenosine_and_dopamine_receptor_systems(),
    )
    site = a2a_only.receptor_systems[0].receptor_site

    print("A2A on the indirect pathway (P5.6c)")
    print("=" * 78)
    print("affinities (nM), default 'rat' set:")
    for lig in site.ligands:
        print(
            "  %-10s Kd %9.1f  kon %9.6f  koff %8.3f  efficacy %.2f"
            % (lig.name, lig.koff / lig.kon, lig.kon, lig.koff, lig.efficacy)
        )
    print(
        "  adenosine's Kd is a FUNCTIONAL POTENCY [Fredholm2011 Table 3, footnote a];"
        "\n  its binding affinity cannot be measured. See adenosine.py."
    )
    print()
    print("A2A node: %s (also carries D2)  [Schiffmann1991]" % ad.A2A_NODE)
    print(
        "gains on that node: A2A %+0.1f (Gs/Golf), D2 %+0.1f (Gi) -- opposite signs\n"
        "are [Ferre1993]; the unequal magnitudes are two illustrative choices and\n"
        "are NOT a potency comparison."
        % (ad.A2A_MECHANISM_GAIN, scenarios.d2_coupling()[ad.A2A_NODE])
    )
    print()

    header = "%-26s" % "condition" + "".join("%11s" % n[:10] for n in NODES)
    print(header)
    print("-" * len(header))

    conditions = (
        ("no drug", a2a_only, {}),
        ("adenosine", a2a_only, {"adenosine": ADENOSINE_BOLUS}),
        ("caffeine alone", a2a_only, {"caffeine": CAFFEINE_BOLUS}),
        (
            "adenosine + caffeine",
            a2a_only,
            {"adenosine": ADENOSINE_BOLUS, "caffeine": CAFFEINE_BOLUS},
        ),
        ("adenosine (+D2 system)", with_d2, {"adenosine": ADENOSINE_BOLUS}),
        (
            "adenosine + dopamine",
            with_d2,
            {"adenosine": ADENOSINE_BOLUS, "dopamine": DOPAMINE_BOLUS},
        ),
    )
    for label, sim, doses in conditions:
        result = run(sim, network, **doses)
        row = "%-26s" % label
        for node in NODES:
            row += "%11.4f" % result["rates"][network.index[node]][-1]
        print(row)

    print()
    print("The sign chain, node by node")
    print("-" * 78)
    print(
        "  Striatum -| GPe -| STN -> GPi -| Thalamus: three inhibitory links and\n"
        "  one excitatory, so an INCREASE at the striatum arrives at the thalamic\n"
        "  readout as a DECREASE. An A2A agonist therefore reduces the locomotor\n"
        "  readout and an A2A antagonist restores it, which is the direction\n"
        "  [Ledent1997]'s knockout result requires."
    )

    print()
    print("Caffeine is not an A2A-selective drug")
    print("-" * 78)
    for species in ad.SPECIES_SETS:
        print(
            "  %-6s  A1 %8.0f nM   A2A %8.0f nM   Kd(A1)/Kd(A2A) = %.2f"
            % (
                species,
                ad.A2A_KD_CAFFEINE_A1_RAT
                if species == "rat"
                else ad.A2A_KD_CAFFEINE_A1_HUMAN,
                ad.caffeine_kd(species),
                ad.caffeine_a1_a2a_selectivity(species),
            )
        )
    print(
        "  Non-selective in rat and A1-preferring in human. A1 is not modelled, so\n"
        "  a caffeine dose here produces one half of caffeine's action and none of\n"
        "  the other, and no potency statement from this module is a statement\n"
        "  about caffeine rather than about a hypothetical A2A-selective xanthine."
    )

    print()
    print("What the species set changes")
    print("-" * 78)
    for species in ad.SPECIES_SETS:
        sim = Simulation(
            network=network,
            receptor_systems=ad.adenosine_receptor_systems(species=species),
        )
        result = run(sim, network, adenosine=ADENOSINE_BOLUS)
        thalamus = result["rates"][network.index["Thalamus"]][-1]
        print(
            "  %-6s  adenosine Kd %6.0f nM, caffeine %6.0f nM, ratio %5.1f-fold"
            "  -> thalamus %.4f"
            % (
                species,
                ad.adenosine_kd(species),
                ad.caffeine_kd(species),
                ad.caffeine_over_adenosine_affinity_ratio(species),
                thalamus,
            )
        )
    print(
        "  Both affinities move ~2-fold between species IN THE SAME DIRECTION, so\n"
        "  the individual values shift 2x and the caffeine-to-adenosine RATIO\n"
        "  shifts 4x. The default is rat, because every mechanism paper this\n"
        "  module depends on is rodent."
    )


if __name__ == "__main__":
    main()
