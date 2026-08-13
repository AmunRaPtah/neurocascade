"""Monoamine transporters DAT, SERT and NET (P5.8).

Prints the transporter layer end to end (PK -> extracellular transmitter
concentration -> receptor occupancy -> circuit rates -> readout) for the five
drugs the P5.8 brief names, on the five-node basal-ganglia indirect pathway
the D2 scenario has used since v0.

A transporter is not a receptor. It produces no activation for a node to
read; what it produces is a CONCENTRATION, and the concentration is what
receptors then bind. So the first half of this script has no circuit in it at
all, and the second half is the wiring that makes the first half matter.

  Part 1, the transmitter pools. Cocaine, methylphenidate, amphetamine,
  fluoxetine and atomoxetine, each at 1 uM, against all three transporters.
  The steady-state fold-rise a pure blocker produces is exactly `1 + C/Ki`
  with Km, Vmax and the basal level cancelling out of the mass balance, so
  these numbers depend on ONE measured constant per drug and on nothing
  illustrative.

  Part 2, substrate versus blocker. Amphetamine is a transporter SUBSTRATE:
  it is carried across the membrane and drives transmitter back out by
  reverse transport, which no blocker does. [Rothman2012] measured its
  release EC50 at 6.4 nM at DAT against an uptake-inhibition Ki of 560 nM
  ([HanGu2006]), so its releasing action is 88-fold the more potent of the
  two. The script runs amphetamine with the release mechanism on and off and
  prints what leaving it out would cost.

  Part 3, the Phase 6 interface. The calibrated D2 site, unchanged, with its
  `dopamine` ligand driven by DAT's extracellular pool instead of by a dose
  injected by hand. Three things follow, and the third is new to this
  codebase: a psychostimulant raises the readout, an antipsychotic lowers it,
  and an ANTAGONIST GIVEN ALONE IS NO LONGER INERT, because for the first
  time there is measured endogenous transmitter for it to displace.

Species: mouse throughout for the uptake kinetics, the basal levels and the
three psychostimulants' constants. Fluoxetine and atomoxetine use human
transporters, and `monoamine_transporters.species_ratio()` prints what that
substitution costs in [HanGu2006]'s own two-species measurement.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python \\
        examples/monoamine_transporter_psychostimulants.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import monoamine_transporters as mt  # noqa: E402
from neurocascade import scenarios  # noqa: E402


DOSE_NM = 1000.0
T_END = 30.0


def _wrap(text, width):
    """Minimal greedy wrap; the repository has no formatting dependency."""
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = word if not current else current + " " + word
    if current:
        lines.append(current)
    return lines


def part_one():
    print("=" * 74)
    print("1. The transmitter pools: what one micromolar of each drug does")
    print("=" * 74)
    print(
        "\n   Extracellular transmitter at steady state, as a multiple of its own\n"
        "   drug-free level. Exactly `1 + C/Ki`: Km, Vmax and the basal level\n"
        "   cancel, so no illustrative constant appears in this table.\n"
    )
    header = "   %-17s %-26s %-26s %-26s" % (
        "drug",
        "dopamine (DAT)",
        "serotonin (SERT)",
        "noradrenaline (NET)",
    )
    print(header)
    for drug in mt.DRUGS:
        cells = []
        for transporter in mt.TRANSPORTERS:
            if mt.is_substrate(drug):
                cells.append("substrate, see part 2")
                continue
            fold = mt.fold_rise(drug, DOSE_NM, transporter)
            if fold is None:
                cells.append("no constant (gap)")
            else:
                cells.append(
                    "%8.2fx (Ki %7.0f nM)" % (fold, mt.ki(drug, transporter))
                )
        print("   %-17s %-26s %-26s %-26s" % (drug, cells[0], cells[1], cells[2]))

    print("\n   The two gaps are recorded, not filled:")
    for (drug, transporter), why in sorted(mt.MISSING_KI.items()):
        print("     %s at %s:" % (drug, transporter))
        for line in _wrap(why, 66):
            print("       " + line)

    print(
        "\n   The two largest numbers in that table are over-predictions, and this\n"
        "   module says so rather than presenting them as results. Fluoxetine at\n"
        "   334x and atomoxetine at 201x are what the open-loop mass balance gives\n"
        "   when nothing lets a rising transmitter cut its own release. Real 5-HT1A\n"
        "   and alpha2 autoreceptors do exactly that, and they are P6.1/P6.2's."
    )

    print("\n   Simulated, to check the algebra against the integrated ODE:")
    network, sim, result = mt.run(cocaine=DOSE_NM, t_end=T_END)
    for transporter in mt.TRANSPORTERS:
        trace = mt.transmitter_trace(sim, result, transporter)
        fold = mt.transmitter_fold(sim, result, transporter)
        print(
            "     cocaine 1 uM at %-5s peak %8.3f nM = %5.2fx basal"
            "   (predicted %5.2fx)"
            % (
                transporter,
                trace.max(),
                fold.max(),
                mt.fold_rise("cocaine", DOSE_NM, transporter),
            )
        )
    print(
        "\n   The peak is at t = 0 and the trace decays with the drug: every drug\n"
        "   here carries the shared illustrative TOOL_COMPOUND_KE (t1/2 30 min),\n"
        "   so read the peaks and not the endpoints."
    )


def part_two():
    print()
    print("=" * 74)
    print("2. Substrate versus blocker, which is a mechanism and not an adjective")
    print("=" * 74)
    ratios = mt.release_over_block_ratio()
    print(
        "\n   Amphetamine, the one substrate among the five. Its two measured\n"
        "   potencies, from two laboratories and two assays:\n"
    )
    print("   %-8s %-22s %-22s %s" % ("", "release EC50", "uptake-block Ki", "ratio"))
    for transporter in mt.TRANSPORTERS:
        print(
            "   %-8s %-22s %-22s %.1fx"
            % (
                transporter,
                "%7.1f nM  [Rothman2012]" % mt.release_ec50("amphetamine", transporter),
                "%7.0f nM  [HanGu2006]" % mt.ki("amphetamine", transporter),
                ratios[transporter],
            )
        )

    print(
        "\n   What representing only the blockade would cost, at 100 nM "
        "amphetamine:\n"
    )
    for capacity, label in ((mt.RELEASE_CAPACITY, "substrate + blocker"),
                            (0.0, "blocker only (wrong)")):
        systems = mt.monoamine_transporter_systems(release_capacity=capacity)
        _, sim, result = mt.run(transporters=systems, amphetamine=100.0, t_end=T_END)
        folds = [
            mt.transmitter_fold(sim, result, name).max() for name in mt.TRANSPORTERS
        ]
        print(
            "     %-22s DA %5.2fx   5-HT %5.2fx   NA %5.2fx"
            % (label, folds[0], folds[1], folds[2])
        )
    print(
        "\n   The blocker-only row is what the brief says must not happen. Note the\n"
        "   serotonin column barely moves either way: amphetamine's release EC50 at\n"
        "   SERT is 1960 nM, 300-fold its DAT value, which is [Rothman2012]'s own\n"
        "   headline and is why amphetamine is a catecholamine drug.\n"
        "\n   RELEASE_CAPACITY is the module's one illustrative mechanism constant\n"
        "   (how much efflux a saturating substrate drives, in absolute terms). The\n"
        "   ORDERING above holds for any positive value of it, and a test says so."
    )


def part_three():
    print()
    print("=" * 74)
    print("3. The Phase 6 interface: a receptor fed by a transporter")
    print("=" * 74)
    network = mt.tonic_compensated_network()
    readout = scenarios.locomotor_activity_readout()
    occupancy = mt.resting_d2_occupancy()
    print(
        "\n   Endogenous dopamine, measured rather than dosed: [Bosse2011]'s basal\n"
        "   5.0 nM against [Agren2021]'s D2 Kd of 2.03 nM puts the receptor at\n"
        "   %.1f%% occupancy before any drug is given.\n" % (100.0 * occupancy)
    )
    print(
        "   %-20s %-12s %-12s %-12s %s"
        % ("condition", "DA (nM)", "D2 occ", "haloperidol", "locomotor")
    )
    baseline = None
    for label, doses in (
        ("no drug", {}),
        ("cocaine 1 uM", {"cocaine": DOSE_NM}),
        ("methylphenidate 1 uM", {"methylphenidate": DOSE_NM}),
        ("amphetamine 100 nM", {"amphetamine": 100.0}),
        ("atomoxetine 1 uM", {"atomoxetine": DOSE_NM}),
        ("haloperidol 100 nM", {"haloperidol": 100.0}),
    ):
        _, sim, result = mt.run(
            receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
            network=network,
            t_end=T_END,
            **doses
        )
        score = readout.score(network, result["rates"][:, -1])
        if baseline is None:
            baseline = score
        dopamine = result["transmitter"][sim.transmitter_index["DAT"]]
        print(
            "   %-20s %-12s %-12s %-12s %.4f  (%+.4f)"
            % (
                label,
                "%.2f" % dopamine[-1],
                "%.3f" % result["occupancy"][0, -1],
                "%.3f" % result["occupancy"][1, -1],
                score,
                score - baseline,
            )
        )
    print(
        "\n   Three things in that table, and the third is new to this codebase:\n"
        "     * the psychostimulants raise the readout, by raising extracellular\n"
        "       dopamine at a receptor that was given no dose of dopamine;\n"
        "     * haloperidol lowers it, which is the direction the D2 scenario has\n"
        "       always claimed and could never demonstrate from an antagonist\n"
        "       given alone;\n"
        "     * haloperidol given ALONE does something at all. Every module from\n"
        "       P5.1 to P5.7 had to write 'an antagonist alone is inert here,\n"
        "       because there is no endogenous transmitter to displace'. There is\n"
        "       now, for dopamine, and it is a measured 5.0 nM.\n"
        "\n   What is still missing is the loop back: nothing here lets the circuit's\n"
        "   own firing rate set the release rate. That is P6.1, and the release rate\n"
        "   is a plain settable attribute precisely so that it can be replaced."
    )


def main():
    part_one()
    part_two()
    part_three()


if __name__ == "__main__":
    main()
