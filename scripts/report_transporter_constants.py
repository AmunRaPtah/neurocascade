"""P5.8: the monoamine transporter constants, and what they predict.

Writes `results/transporter_constants.txt`. Five blocks:

  1. The uptake kinetics and basal levels the mechanism runs on, with the
     species and preparation of each, and the one that is a unit rather than
     a measurement flagged as such.
  2. The drug table: every cited constant for the five drugs the brief names,
     the gaps recorded as gaps, and the exact steady-state fold-rise each
     blocker produces.
  3. Substrate versus blocker, quantified: amphetamine's release potency
     against its blocking potency, and what a blocker-only model would get
     wrong.
  4. The provenance checks: two mouse preparations against each other, two
     species from one paper, two laboratories on atomoxetine.
  5. The Phase 6 interface, run: the calibrated D2 site driven by DAT's
     extracellular dopamine pool, with a psychostimulant, an antipsychotic,
     and the antagonist-alone result that was impossible before this task.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python scripts/report_transporter_constants.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from neurocascade import monoamine_transporters as mt  # noqa: E402
from neurocascade import scenarios  # noqa: E402


OUT = os.path.join(HERE, "..", "results", "transporter_constants.txt")

DOSE_NM = 1000.0
T_END = 30.0


def block_one(w):
    w("=" * 72 + "\n")
    w("1. Uptake kinetics and basal levels\n")
    w("=" * 72 + "\n\n")
    w("  quantity                       value            source\n")
    w("  " + "-" * 68 + "\n")
    rows = [
        ("Km, dopamine uptake", "%.0f nM" % mt.KM_DOPAMINE,
         "[JohnJones2007] mouse CPu slice"),
        ("Vmax, dopamine uptake", "%.0f nM/min" % mt.VMAX_DOPAMINE,
         "[JohnJones2007] 3.1 uM/s"),
        ("  same, second laboratory", "%.0f nM/min" % mt.VMAX_DOPAMINE_SECOND_LAB,
         "[Bosse2011] 4.1 uM/s, mouse"),
        ("basal extracellular dopamine", "%.1f nM" % mt.BASAL_DOPAMINE,
         "[Bosse2011] zero net flux, mouse"),
        ("Km, serotonin uptake", "%.0f nM" % mt.KM_SEROTONIN,
         "[JohnJones2007] mouse SNr slice"),
        ("Vmax, serotonin uptake", "%.0f nM/min" % mt.VMAX_SEROTONIN,
         "[JohnJones2007] 0.080 uM/s"),
        ("basal extracellular serotonin", "%.1f nM" % mt.BASAL_SEROTONIN,
         "[Mathews2004] zero net flux, mouse"),
        ("noradrenaline clearance t1/2", "%.2f s" % mt.NORADRENALINE_CLEARANCE_HALF_LIFE_S,
         "[Park2012] rat vBNST, in vivo"),
        ("  as a rate constant", "%.2f /min" % mt.NORADRENALINE_CLEARANCE_RATE,
         "derived: ln2 / t1/2"),
        ("basal extracellular noradrenaline", "%.2f nM" % mt.BASAL_NORADRENALINE,
         "[PanLai1995] zero net flux, conscious rat mPFC (P6.2)"),
    ]
    for name, value, source in rows:
        w("  %-30s %-16s %s\n" % (name, value, source))
    w(
        "\n  NET is the only one of the three built in the first-order form, and\n"
        "  that is a gap rather than a preference: what is published for it is a\n"
        "  clearance half-life, which fixes Vmax/Km and neither of them. Splitting\n"
        "  a measured ratio into two numbers would be inventing one.\n"
        "\n  The dopamine pool relaxes with a time constant of Km/Vmax = %.2f ms,\n"
        "  more than three orders of magnitude below the circuit's. That is why\n"
        "  every run in this module defaults to LSODA, and a test asserts that\n"
        "  RK45 agrees.\n"
        % (1000.0 * 60.0 * mt.KM_DOPAMINE / mt.VMAX_DOPAMINE)
    )


def block_two(w):
    w("\n" + "=" * 72 + "\n")
    w("2. The five drugs, and what one micromolar of each does\n")
    w("=" * 72 + "\n\n")
    w(
        "  Extracellular transmitter at steady state as a multiple of its own\n"
        "  drug-free level. For a pure blocker this is EXACTLY 1 + C/Ki: the mass\n"
        "  balance cancels Km, Vmax and the basal level, so this column depends on\n"
        "  one measured constant per drug and on nothing illustrative.\n\n"
    )
    w("  %-17s %-6s %10s %12s   %s\n"
      % ("drug", "site", "Ki (nM)", "1 uM ->", "source"))
    w("  " + "-" * 84 + "\n")
    sources = {
        "cocaine": "[HanGu2006] mouse",
        "methylphenidate": "[HanGu2006] mouse",
        "amphetamine": "[HanGu2006] mouse",
        "fluoxetine": "[Tatsumi1997] human",
        "atomoxetine": "[Bymaster2002] human",
    }
    for drug in mt.DRUGS:
        for transporter in mt.TRANSPORTERS:
            constant = mt.ki(drug, transporter)
            if constant is None:
                w("  %-17s %-6s %10s %12s   %s\n"
                  % (drug, transporter, "none", "-", "GAP, see below"))
                continue
            if mt.is_substrate(drug):
                predicted = "see block 3"
            else:
                predicted = "%.2fx" % mt.fold_rise(drug, DOSE_NM, transporter)
            w("  %-17s %-6s %10.1f %12s   %s\n"
              % (drug, transporter, constant, predicted, sources[drug]))
    w("\n  The gaps, recorded rather than filled:\n")
    for (drug, transporter), why in sorted(mt.MISSING_KI.items()):
        w("    %s at %s\n" % (drug, transporter))
        for line in _wrap(why, 66):
            w("      %s\n" % line)
    w(
        "\n  Two numbers in that table are OVER-PREDICTIONS and are reported as\n"
        "  such: fluoxetine at 334x and atomoxetine at 201x. They are what an\n"
        "  open-loop mass balance gives when nothing lets a rising transmitter\n"
        "  cut its own release. Real 5-HT1A and alpha2 autoreceptors do exactly\n"
        "  that. This is the sharpest statement in this codebase so far of what\n"
        "  the projection-system layer (P6.1, P6.2) is for.\n"
    )


def block_three(w):
    w("\n" + "=" * 72 + "\n")
    w("3. Substrate versus blocker\n")
    w("=" * 72 + "\n\n")
    w(
        "  Cocaine, methylphenidate, fluoxetine and atomoxetine bind the\n"
        "  transporter and are not carried across it. Amphetamine is carried\n"
        "  across, and drives transmitter back out by reverse transport, which no\n"
        "  blocker does ([Rothman2012]'s substrate-reversal criterion, [Zhu2008]).\n"
        "  In this model that is two constants rather than an adjective.\n\n"
    )
    ratios = mt.release_over_block_ratio()
    w("  %-6s %18s %18s %10s %10s\n"
      % ("site", "release EC50 (nM)", "block Ki (nM)", "ratio", "Emax"))
    w("  " + "-" * 66 + "\n")
    for transporter in mt.TRANSPORTERS:
        w("  %-6s %18.1f %18.0f %9.1fx %10.2f\n"
          % (
              transporter,
              mt.release_ec50("amphetamine", transporter),
              mt.ki("amphetamine", transporter),
              ratios[transporter],
              mt.release_emax("amphetamine", transporter),
          ))
    w(
        "\n  Amphetamine's releasing action at DAT is %.0f-fold more potent than its\n"
        "  blocking action, measured in two different laboratories' assays. A\n"
        "  model representing it as a blocker alone would put its dopaminergic\n"
        "  potency nearly two orders of magnitude too low.\n\n"
        % ratios["DAT"]
    )
    w("  Simulated, 100 nM amphetamine, with the release mechanism on and off:\n\n")
    w("  %-24s %10s %10s %10s\n" % ("", "DA", "5-HT", "NA"))
    for capacity, label in ((mt.RELEASE_CAPACITY, "substrate + blocker"),
                            (0.0, "blocker only (wrong)")):
        systems = mt.monoamine_transporter_systems(release_capacity=capacity)
        _, sim, result = mt.run(transporters=systems, amphetamine=100.0, t_end=T_END)
        folds = [mt.transmitter_fold(sim, result, n).max() for n in mt.TRANSPORTERS]
        w("  %-24s %9.2fx %9.2fx %9.2fx\n" % (label, folds[0], folds[1], folds[2]))
    w(
        "\n  RELEASE_CAPACITY, the absolute efflux a saturating substrate drives, is\n"
        "  this module's one illustrative mechanism constant; the ORDERING above\n"
        "  holds for every positive value of it, and a test says so.\n"
    )


def block_four(w):
    w("\n" + "=" * 72 + "\n")
    w("4. Provenance checks\n")
    w("=" * 72 + "\n\n")
    w("  a. Two MOUSE preparations, cloned transporters against brain slices\n")
    w("     ([HanGu2006] against [JohnJones2007]), fold disagreement:\n\n")
    for (drug, transporter), value in sorted(mt.preparation_agreement().items()):
        w("       %-17s %-5s %12.2fx\n" % (drug, transporter, value))
    w(
        "\n     The three DAT cells agree within 1.4-fold: a cloned mouse\n"
        "     transporter in a gut-derived cell line and a mouse striatal terminal\n"
        "     in a slice give the same number. The SERT cells do not, and\n"
        "     [JohnJones2007] states the caveat itself. Fluoxetine is a different\n"
        "     world entirely, and the module docstring argues which value to use.\n\n"
    )
    w("  b. Two SPECIES from one paper's own two columns ([HanGu2006]):\n\n")
    for (drug, transporter), value in sorted(mt.species_ratio().items()):
        w("       %-17s %-5s %12.2fx\n" % (drug, transporter, value))
    w(
        "\n     Eight of nine under 2.4-fold. This is what using human constants\n"
        "     for fluoxetine and atomoxetine costs, measured rather than hoped.\n\n"
    )
    w("  c. Two LABORATORIES on atomoxetine ([Bymaster2002], [Carlier1998]):\n\n")
    for transporter, value in sorted(mt.atomoxetine_lab_disagreement().items()):
        w("       %-5s %12.2fx   (%6.1f nM against %6.2f nM)\n"
          % (transporter, value, mt.ki("atomoxetine", transporter),
             mt.KI_SECOND_LAB["atomoxetine"][transporter]))
    used = mt.selectivity("atomoxetine", "NET", "SERT")
    other = (mt.KI_SECOND_LAB["atomoxetine"]["SERT"]
             / mt.KI_SECOND_LAB["atomoxetine"]["NET"])
    w(
        "\n     The SERT cell is the one that matters: NET-over-SERT selectivity is\n"
        "     %.1f-fold on the numbers used here and %.1f-fold on the other paper's.\n"
        "     Between-laboratory spread is LARGER here than the species spread in\n"
        "     (b), which is the useful ordering.\n" % (used, other)
    )


def block_five(w):
    w("\n" + "=" * 72 + "\n")
    w("5. The Phase 6 interface, run\n")
    w("=" * 72 + "\n\n")
    network = mt.tonic_compensated_network()
    readout = scenarios.locomotor_activity_readout()
    w(
        "  The calibrated D2 site, unchanged, with its `dopamine` ligand driven by\n"
        "  DAT's extracellular pool instead of by a dose injected by hand. The\n"
        "  interface is one dict entry: transmitter_sources={'dopamine': 'DAT'}.\n\n"
        "  Endogenous tone, from two measurements and no fitting: [Bosse2011]'s\n"
        "  basal 5.0 nM against [Agren2021]'s D2 Kd of 2.03 nM puts the receptor\n"
        "  at %.1f%% occupancy before any drug.\n\n" % (100.0 * mt.resting_d2_occupancy())
    )
    w("  %-22s %9s %9s %9s %11s\n"
      % ("condition", "DA (nM)", "D2 occ", "halo occ", "locomotor"))
    w("  " + "-" * 64 + "\n")
    baseline = None
    for label, doses in (
        ("no drug", {}),
        ("cocaine 1 uM", {"cocaine": DOSE_NM}),
        ("methylphenidate 1 uM", {"methylphenidate": DOSE_NM}),
        ("amphetamine 100 nM", {"amphetamine": 100.0}),
        ("atomoxetine 1 uM", {"atomoxetine": DOSE_NM}),
        ("fluoxetine 1 uM", {"fluoxetine": DOSE_NM}),
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
        w("  %-22s %9.2f %9.3f %9.3f %8.4f (%+.4f)\n"
          % (label, dopamine[-1], result["occupancy"][0, -1],
             result["occupancy"][1, -1], score, score - baseline))
    w(
        "\n  Three readings, and the third is new to this codebase:\n"
        "    * the psychostimulants raise the readout, by raising extracellular\n"
        "      dopamine at a receptor that was given no dose of dopamine;\n"
        "    * fluoxetine does nothing here, correctly: it has no measured\n"
        "      constant at DAT and this circuit has no serotonin receptor;\n"
        "    * haloperidol given ALONE moves the circuit. Every module from P5.1\n"
        "      to P5.7 had to write 'an antagonist alone is inert here, because\n"
        "      there is no endogenous transmitter to displace'. There is now, for\n"
        "      dopamine, and it is a measured 5.0 nM.\n"
        "\n  What is still missing is the loop back: nothing lets the circuit's own\n"
        "  firing rate set the release rate. That is P6.1, and\n"
        "  MonoamineTransporter.release_rate is a plain settable attribute exactly\n"
        "  so that P6.1 can replace it without touching this mechanism.\n"
    )


def _wrap(text, width):
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


def main():
    with open(OUT, "w") as handle:
        w = handle.write
        w("P5.8: monoamine transporters DAT, SERT and NET\n")
        w("Generated by scripts/report_transporter_constants.py\n\n")
        block_one(w)
        block_two(w)
        block_three(w)
        block_four(w)
        block_five(w)
    print("wrote %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
