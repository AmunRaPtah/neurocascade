"""Kappa and delta opioid receptors, and the end of mu selectivity (P5.7).

Prints seven conditions end to end (PK -> receptor occupancy -> node drive or
modulated weight -> circuit rates -> readouts) for the seven-node reward /
analgesia / arousal network the mu-opioid scenario has used since Track C1,
now carrying three opioid receptors instead of one.

Every affinity and every efficacy below is [Toll1998] (NIDA Res Monogr
178:440-466), human cloned mu/delta/kappa receptors on CHO cells: one
laboratory, one panel, so the selectivity ratios are ratios of numbers
measured in the same experiment rather than of two papers.

  0. no drug: exactly flat at the network's own fixed point, and identical
     with and without the two new receptor systems.
  1. fentanyl 10 nM: the three established mu results, unchanged. Reward up,
     analgesia up, arousal down.
  2. U50488 100 nM: the readout the mu agonist RAISES, a kappa agonist
     LOWERS. That is [DiChiara1988], who gave freely-moving rats fentanyl
     and U-50,488 and measured accumbal dopamine going opposite ways.
  3. dynorphin A: the endogenous peptide, same direction as the tool drug,
     despite being a full agonist at all three receptors. Read its 'd peak'
     column and not its 'd final' one: `PEPTIDE_KE` clears it in seconds and
     its derivation-3 `koff` unbinds it in minutes, so by t = 40 min there
     is nothing left. That is the illustrative pharmacokinetics talking, not
     the pharmacology.
  4. Leu-enkephalin: delta, and therefore the SAME side as mu, so condition
     2's result is specific to kappa rather than generic to adding a
     receptor.
  5. nor-BNI alone: nothing. [Britt2008] measured that too.
  6. fentanyl 100 uM: the limit. Four orders of magnitude above a clinical
     exposure, fentanyl's own 30% partial agonism at kappa suppresses the
     projection its mu action is disinhibiting, and the reward effect comes
     out ~29% below what the mu-only model predicts.

The network is Track C1's, put through P3.4's `stabilize()`: one global
weight scalar and re-solved baselines, relative weights and topology
untouched. That is not cosmetic and `opioid.stabilized_reward_network()`
explains why -- on Track C1's own baselines VTA_DA settles saturated OFF, the
mesolimbic projection transmits nothing, and a PRESYNAPTIC modulator of a
silent projection has, exactly, no effect.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/opioid_kappa_delta_circuit.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import opioid as op  # noqa: E402
from neurocascade import scenarios  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402


T_END = 40.0
N_POINTS = 201

READOUTS = [
    ("incentive_salience", op.incentive_salience_readout()),
    ("accumbal_drive", op.accumbal_drive_readout()),
    ("analgesia_index", scenarios.analgesia_index_readout()),
    ("arousal_index", scenarios.arousal_index_readout()),
]

CONDITIONS = [
    ("1. fentanyl 10 nM", {"fentanyl": 10.0}),
    ("2. U50488 100 nM", {"U50488": 100.0}),
    ("3. dynorphin A 1000 nM", {"dynorphin_A": 1000.0}),
    ("4. Leu-enkephalin 100 nM", {"leu_enkephalin": 100.0}),
    ("5. nor-BNI 1000 nM", {"nor_BNI": 1000.0}),
    ("6. fentanyl 100 uM", {"fentanyl": 1.0e5}),
]


def main():
    network = op.stabilized_reward_network()
    rest = op.rest_rates(network)
    systems = op.opioid_receptor_systems()
    mu_only = op.mu_only_receptor_systems()

    def go(receptor_systems, **doses):
        sim = Simulation(network=network, receptor_systems=receptor_systems)
        result = sim.run(
            doses=sim.dose_vector(**doses),
            t_span=(0.0, T_END),
            t_eval=np.linspace(0.0, T_END, N_POINTS),
            rates0=rest,
            method="LSODA",
        )
        return sim, result

    print(__doc__.split("Run with")[0].rstrip())
    print("=" * 78)

    print("\nThe two claims the brief turns on, from [Toll1998]:")
    print(
        "  naloxone   Ki  mu %.1f nM   kappa %.1f nM   -> %.2f-fold selective"
        % (
            op.ki("naloxone", "mu"),
            op.ki("naloxone", "kappa"),
            op.naloxone_mu_over_kappa_selectivity(),
        )
    )
    kappa_efficacy, mu_efficacy = op.fentanyl_is_a_partial_kappa_agonist()
    print(
        "  fentanyl   efficacy  mu %.2f   kappa %.2f   (partial agonist, "
        "not an inert occupier)" % (mu_efficacy, kappa_efficacy)
    )

    _, base_full = go(systems)
    _, base_mu = go(mu_only)
    print(
        "\n0. no drug: max |rate - 0.5| = %.2e with three receptors, %.2e with"
        % (
            np.abs(base_full["rates"] - 0.5).max(),
            np.abs(base_mu["rates"] - 0.5).max(),
        )
    )
    print("   one; the two runs differ by %.2e." % np.abs(
        base_full["rates"] - base_mu["rates"]
    ).max())

    for label, doses in CONDITIONS:
        sim, result = go(systems, **doses)
        print("\n" + label)
        occupancy_note = []
        for index, name in ((0, "mu"), (1, "kappa"), (2, "delta")):
            block = result["occupancy"][sim.state_rows(index)]
            occupancy_note.append("%s %.3f" % (name, block.sum(axis=0).max()))
        print("  peak total occupancy: " + ",  ".join(occupancy_note))
        kappa_activation = sum(
            result["occupancy"][sim.state_rows(1)][i] * lig.efficacy
            for i, lig in enumerate(sim.receptor_systems[1].receptor_site.ligands)
        )
        print(
            "  W[NAc, VTA_DA] at peak kappa activation: %.4f of a base 2.0000"
            % op.modulated_mesolimbic_weight(kappa_activation.max())
        )
        print(
            "  %-22s %10s %12s %12s" % ("readout", "final", "d final", "d peak")
        )
        for name, readout in READOUTS:
            trace = op.readout_trace(readout, network, result)
            free = op.readout_trace(readout, network, base_full)
            delta = trace - free
            peak = delta[np.argmax(np.abs(delta))]
            print(
                "  %-22s %10.5f %12.5f %12.5f"
                % (name, trace[-1], delta[-1], peak)
            )

    print("\n" + "=" * 78)
    print("The re-run, on the readout the two receptors compete for:")
    readout = op.incentive_salience_readout()
    for label, dose in (("10 nM (clinical)", 10.0), ("100 uM (the limit)", 1.0e5)):
        _, full = go(systems, fentanyl=dose)
        _, only = go(mu_only, fentanyl=dose)
        a = readout.score(network, only["rates"][:, -1]) - readout.score(
            network, base_mu["rates"][:, -1]
        )
        b = readout.score(network, full["rates"][:, -1]) - readout.score(
            network, base_full["rates"][:, -1]
        )
        print(
            "  fentanyl %-20s mu only %+.5f   mu+kappa+delta %+.5f   (%+.1f%%)"
            % (label, a, b, 100.0 * (b - a) / abs(a))
        )
    print(
        "\n  The second line is what 'not mu-selective in the limit' means as a\n"
        "  number: fentanyl's own kappa partial agonism working against its mu\n"
        "  effect. The full table, both networks, and the naloxone reversal\n"
        "  comparison are in results/opioid_cross_affinity.txt."
    )


if __name__ == "__main__":
    main()
