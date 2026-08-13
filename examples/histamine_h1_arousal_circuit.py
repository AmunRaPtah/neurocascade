"""Histamine H1 agonism and antagonism on a cortical E-I loop (P5.6a).

Prints five conditions end to end (PK -> receptor occupancy -> node drive ->
circuit rates -> readouts) for the same two-node cortical loop P5.1, P5.3,
P5.4 and P5.5 use, now with a tuberomammillary source node and an H1 system
on the pyramidal node.

What this example exists to show:

  0. no drug: exactly flat, so every deviation below is the drug.
  1. histamine: cortical output rises. H1 is Gq-coupled and histamine closes
     a resting K+ conductance ([McCormick1991], in thalamus, where the block
     by diphenhydramine itself was demonstrated).
  2. diphenhydramine alone: nothing at all. That is the standing
     no-endogenous-tone limitation (P6.2), and here it is at its most
     conspicuous, because the entire clinical point of diphenhydramine is
     what it does given alone.
  3. histamine + diphenhydramine: the response is cut down. The drug's
     margin over the transmitter is only 190-fold, which is small by this
     codebase's standards, and the block is correspondingly incomplete.
  4. histamine + mepyramine: more complete, because mepyramine is 9-fold
     tighter at the same pool and is dosed identically.

The final block is the arithmetic that carries this module. Diphenhydramine
has no measured association rate anywhere used here: its kon is a residence
time from [Bosma2018] divided into an affinity from [Booth2002], a different
laboratory and a different decade. It lands within 3% of mepyramine's
MEASURED kon. Nothing in the construction could have produced that
agreement, and it is a second, independent confirmation at a second receptor
of the chemotype refinement P5.3 took from [Sykes2022]'s beta1 panel.

Every affinity, rate constant and efficacy here is literature-sourced; the
circuit, the mechanism gain and the pharmacokinetics are illustrative and
labelled. See `histamine.py` for the sources, for the 33-fold spread in
histamine's own published affinity, and for what this module deliberately
does not build.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/histamine_h1_arousal_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import histamine as ha  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402


T_END = 60.0
BOLUS = 100000.0  # nM, illustrative, as for every tool compound in Phase 5

CONDITIONS = (
    ("no drug", {}),
    ("histamine", {"histamine": BOLUS}),
    ("diphenhydramine alone", {"diphenhydramine": BOLUS}),
    ("histamine + diphenhydramine", {"histamine": BOLUS, "diphenhydramine": BOLUS}),
    ("histamine + mepyramine", {"histamine": BOLUS, "mepyramine": BOLUS}),
)


def run(sim, network, **doses):
    return sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=np.linspace(0.0, T_END, 241),
        rates0=ha.rest_rates(network),
        method="LSODA",
    )


def main():
    network = ha.tuberomammillary_cortical_network()
    sim = Simulation(
        network=network, receptor_systems=ha.histaminergic_receptor_systems()
    )
    site = sim.receptor_systems[0].receptor_site
    cortical = ha.cortical_output_readout()
    tone = ha.histaminergic_tone_readout()

    print("H1 receptor on a cortical E-I loop (P5.6a)")
    print("=" * 78)
    print("affinities (nM), default 'sf9' set:")
    for lig in site.ligands:
        print(
            "  %-16s Kd %10.2f  kon %8.4f  koff %10.3f  efficacy %.2f"
            % (lig.name, lig.koff / lig.kon, lig.kon, lig.koff, lig.efficacy)
        )
    print()
    print(
        "%-30s %10s %10s %10s %10s"
        % ("condition", "peak occ", "PFC_Pyr", "PFC_GABA", "cortical")
    )
    print("-" * 78)

    for label, doses in CONDITIONS:
        result = run(sim, network, **doses)
        occupancy = result["occupancy"]
        activation = np.array(
            [site.activation(occupancy[:, k]) for k in range(occupancy.shape[1])]
        )
        pyramidal = result["rates"][1]
        interneuron = result["rates"][2]
        readout = np.array(
            [
                cortical.score(network, result["rates"][:, k])
                for k in range(result["rates"].shape[1])
            ]
        )
        peak = readout[np.argmax(np.abs(readout - 0.5))]
        print(
            "%-30s %10.4f %10.4f %10.4f %10.4f"
            % (label, activation.max(), pyramidal.max(), interneuron.max(), peak)
        )

    result = run(sim, network, histamine=BOLUS)
    print()
    print(
        "histaminergic tone (TMN_HA) stays at %.4f in every condition: it carries "
        "no\nreceptor and has no edges. P6.1 is where a projection system gets a "
        "source." % tone.score(network, result["rates"][:, -1])
    )

    print()
    print("The arithmetic this module rests on")
    print("-" * 78)
    print(
        "  diphenhydramine koff  = 1 / (0.41 min residence time)  = %.4f min^-1"
        "   [Bosma2018]" % ha.H1_KOFF_DIPHENHYDRAMINE
    )
    print(
        "  diphenhydramine Kd    = mean(11.7, 13.5)               = %.2f nM"
        "        [Booth2002]" % ha.H1_KD_DIPHENHYDRAMINE
    )
    print(
        "  => derived kon        = koff / Kd                      = %.4f "
        "nM^-1 min^-1" % ha.H1_KON_DIPHENHYDRAMINE
    )
    print(
        "  mepyramine kon        = MEASURED, a third experiment   = %.4f "
        "nM^-1 min^-1   [Bosma2017]" % ha.H1_KON_MEPYRAMINE
    )
    print(
        "  ratio                 = %.3f, i.e. %.1f%% apart, nothing tuned"
        % (
            ha.H1_KON_DIPHENHYDRAMINE / ha.H1_KON_MEPYRAMINE,
            100.0 * abs(1.0 - ha.H1_KON_DIPHENHYDRAMINE / ha.H1_KON_MEPYRAMINE),
        )
    )
    kons = list(ha.H1_KON_PANEL.values())
    print(
        "  and [Bosma2017]'s own ten-antihistamine panel spans %.0f-fold in kon,"
        "\n  so the agreement is a chemotype result and not a property of the site."
        % (max(kons) / min(kons))
    )

    print()
    print("What the affinity-state choice costs")
    print("-" * 78)
    for state in ha.AFFINITY_STATES:
        state_sim = Simulation(
            network=network,
            receptor_systems=ha.histaminergic_receptor_systems(affinity_state=state),
        )
        state_result = run(state_sim, network, histamine=BOLUS)
        peak = state_result["rates"][1].max() - 0.5
        print(
            "  histamine Kd %9.1f nM (%s): peak cortical deviation %+0.4f"
            % (ha.histamine_kd(state), state, peak)
        )
    print(
        "  Both published, %0.0f-fold apart, and the direction is the same under "
        "both.\n  The magnitude is not, and that is the honest error bar."
        % (ha.H1_KD_HISTAMINE_COS7 / ha.H1_KD_HISTAMINE_SF9)
    )


if __name__ == "__main__":
    main()
