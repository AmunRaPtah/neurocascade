"""P5.7: what adding kappa and delta does to the existing mu results.

Writes `results/opioid_cross_affinity.txt`. Three blocks:

  1. The constants, as a selectivity table, so the two claims the brief
     turns on -- naloxone is 1.8-fold mu-over-kappa selective, fentanyl is
     a 30% partial agonist at kappa -- are printed rather than asserted.
  2. The RE-RUN on Track C1's original hand-baselined network, which is the
     strict regression against every mu result this codebase already has.
  3. The RE-RUN on the P3.4-stabilized network, which is the only one of
     the three candidate operating points where a presynaptic modulator of
     the mesolimbic projection can do anything at all (see
     `opioid.stabilized_reward_network`), and therefore the only one where
     the kappa result exists.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python scripts/report_opioid_cross_affinity.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from neurocascade import opioid as op  # noqa: E402
from neurocascade import scenarios, stability  # noqa: E402
from neurocascade.engine import Simulation  # noqa: E402


OUT = os.path.join(HERE, "..", "results", "opioid_cross_affinity.txt")

T_END = 40.0
N_POINTS = 201

#: nM. A fentanyl exposure near the top of what `pk_validation` treats as
#: the clinical/overdose range, and a second one four orders of magnitude
#: above it. The second is NOT a plausible exposure: it exists to make the
#: limit in "not mu-selective in the limit" a number rather than a word.
FENTANYL_CLINICAL = 10.0
FENTANYL_LIMIT = 1.0e5

#: nM. ~500x U50488's kappa Ki of 0.2 nM.
U50488_DOSE = 100.0
DYNORPHIN_DOSE = 1000.0
NALOXONE_DOSE = 100.0


READOUTS = [
    ("incentive_salience", op.incentive_salience_readout()),
    ("accumbal_drive", op.accumbal_drive_readout()),
    ("analgesia_index", scenarios.analgesia_index_readout()),
    ("arousal_index", scenarios.arousal_index_readout()),
]


def traces(network, systems, rates0, **doses):
    sim = Simulation(network=network, receptor_systems=systems)
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=np.linspace(0.0, T_END, N_POINTS),
        rates0=rates0,
        method="LSODA",
    )
    return {
        name: op.readout_trace(readout, network, result) for name, readout in READOUTS
    }


def _extreme(trace, baseline):
    """The point of the trace furthest from the drug-free trace, signed."""
    delta = trace - baseline
    return delta[np.argmax(np.abs(delta))]


#: Below this, an effect is start-up-transient noise rather than drug, and
#: quoting a percentage change of it would be quoting a percentage of zero.
#: Track C1's network is not started at a fixed point (it has none that a
#: drug-free run reaches within the window), so its drug-free trace moves.
NOISE_FLOOR = 1e-3


def half_reversal_dose(network, rates0, readout, agonist_doses, ladder):
    """The naloxone concentration that halves an established circuit effect.

    Returns `(dose, effect_without_naloxone)`, by a scan up a log ladder of
    naloxone concentrations.
    """
    systems = op.opioid_receptor_systems()
    base = traces(network, systems, rates0)[readout]
    alone = traces(network, systems, rates0, **agonist_doses)[readout]
    effect = _extreme(alone, base)
    if abs(effect) < NOISE_FLOOR:
        return None, effect
    for dose in ladder:
        combined = dict(agonist_doses)
        combined["naloxone"] = dose
        with_nx = traces(network, systems, rates0, **combined)[readout]
        if abs(_extreme(with_nx, base)) <= 0.5 * abs(effect):
            return dose, effect
    return None, effect


def peak_occupancy(network, rates0, system_index, ligand_index, **doses):
    """Peak fractional occupancy of one ligand at one receptor over a run."""
    sim = Simulation(network=network, receptor_systems=op.opioid_receptor_systems())
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, T_END),
        t_eval=np.linspace(0.0, T_END, N_POINTS),
        rates0=rates0,
        method="LSODA",
    )
    rows = sim.state_rows(system_index)
    return result["occupancy"][rows][ligand_index].max()


def half_occupancy_dose(network, rates0, system_index, ligand_index, agonist_doses, ladder):
    """The naloxone concentration that halves an agonist's peak occupancy.

    The pharmacological form of the reversal question, and the one that
    isolates naloxone's own selectivity: no circuit, no readout shape, just
    competition at one pool. Compared at MATCHED agonist occupancy (each
    agonist dosed at the same multiple of its own Kd), the ratio between
    the mu answer and the kappa answer is a statement about naloxone.
    """
    alone = peak_occupancy(network, rates0, system_index, ligand_index, **agonist_doses)
    for dose in ladder:
        combined = dict(agonist_doses)
        combined["naloxone"] = dose
        if (
            peak_occupancy(network, rates0, system_index, ligand_index, **combined)
            <= 0.5 * alone
        ):
            return dose, alone
    return None, alone


def block(handle, title, network, rates0, conditions):
    handle.write("\n%s\n%s\n" % (title, "-" * len(title)))
    mu_only = op.mu_only_receptor_systems()
    full = op.opioid_receptor_systems()
    base_mu = traces(network, mu_only, rates0)
    base_full = traces(network, full, rates0)

    header = "%-28s %-20s %12s %12s %12s" % (
        "condition",
        "readout",
        "mu only",
        "mu+kappa+d",
        "change",
    )
    handle.write(header + "\n")
    for label, doses, mu_runnable in conditions:
        run_full = traces(network, full, rates0, **doses)
        run_mu = (
            traces(network, mu_only, rates0, **doses)
            if mu_runnable
            else None
        )
        for name, _ in READOUTS:
            d_full = _extreme(run_full[name], base_full[name])
            if run_mu is None:
                handle.write(
                    "%-28s %-20s %12s %12.5f %12s\n"
                    % (label, name, "n/a", d_full, "n/a")
                )
                continue
            d_mu = _extreme(run_mu[name], base_mu[name])
            if abs(d_mu) > NOISE_FLOOR:
                change = "%+11.1f%%" % (100.0 * (d_full - d_mu) / abs(d_mu))
            else:
                # A percentage of an effect smaller than the drug-free
                # trace's own drift would be a percentage of noise.
                change = "%+11.2e" % (d_full - d_mu)
            handle.write(
                "%-28s %-20s %12.5f %12.5f %12s\n"
                % (label, name, d_mu, d_full, change)
            )
        handle.write("\n")


def main():
    hand = op.reward_network()
    stabilized_report = stability.stabilize(
        scenarios.multi_region_mu_opioid_network(), operating_gain=op.OPERATING_GAIN
    )
    stabilized = stabilized_report.network

    conditions = [
        ("fentanyl %g nM" % FENTANYL_CLINICAL, {"fentanyl": FENTANYL_CLINICAL}, True),
        ("fentanyl %g nM" % FENTANYL_LIMIT, {"fentanyl": FENTANYL_LIMIT}, True),
        ("naloxone %g nM" % NALOXONE_DOSE, {"naloxone": NALOXONE_DOSE}, True),
        (
            "fentanyl %g + naloxone %g" % (FENTANYL_CLINICAL, NALOXONE_DOSE),
            {"fentanyl": FENTANYL_CLINICAL, "naloxone": NALOXONE_DOSE},
            True,
        ),
        ("U50488 %g nM" % U50488_DOSE, {"U50488": U50488_DOSE}, False),
        ("dynorphin A %g nM" % DYNORPHIN_DOSE, {"dynorphin_A": DYNORPHIN_DOSE}, False),
        ("nor-BNI %g nM" % U50488_DOSE, {"nor_BNI": U50488_DOSE}, False),
        (
            "U50488 %g + nor-BNI %g" % (U50488_DOSE, U50488_DOSE * 10),
            {"U50488": U50488_DOSE, "nor_BNI": U50488_DOSE * 10},
            False,
        ),
        ("Leu-enkephalin 100 nM", {"leu_enkephalin": 100.0}, False),
        ("naltrindole 100 nM", {"naltrindole": 100.0}, False),
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as handle:
        handle.write("P5.7 opioid kappa/delta: cross-affinity re-run\n")
        handle.write("=" * 46 + "\n")
        handle.write(
            "\nEvery affinity and efficacy below is [Toll1998] (NIDA Res Monogr\n"
            "178:440-466, PMID 9686407), human cloned mu/delta/kappa receptors on\n"
            "CHO cells, one laboratory and one panel. See src/neurocascade/opioid.py\n"
            "for the full sourcing, the derivations and what is not built.\n"
        )

        handle.write("\nKi (nM), human CHO\n------------------\n")
        handle.write("%-16s %10s %10s %10s\n" % ("ligand", "mu", "delta", "kappa"))
        for name in op.LIGANDS:
            row = []
            for receptor in op.RECEPTORS:
                value = op.ki(name, receptor)
                row.append(
                    (">%9.0f" % value) if op.is_censored(name, receptor) else "%10.1f" % value
                )
            handle.write("%-16s %s\n" % (name, " ".join(row)))

        handle.write(
            "\nEfficacy (fraction of DAMGO / DPDPE / U69,593 max [35S]GTPyS)\n"
            "------------------------------------------------------------\n"
        )
        handle.write("%-16s %10s %10s %10s\n" % ("ligand", "mu", "delta", "kappa"))
        for name in op.LIGANDS:
            handle.write(
                "%-16s %10.2f %10.2f %10.2f\n"
                % tuple([name] + [op.efficacy(name, r) for r in op.RECEPTORS])
            )
        handle.write(
            "\n  'flat' in the source table (measured, no stimulation at any\n"
            "  concentration) is written 0.00 above and listed in\n"
            "  opioid.MEASURED_FLAT: naloxone, nor-BNI and naltrindole at all\n"
            "  three receptors, U50488 at mu and delta.\n"
        )

        handle.write("\nSelectivity, from the same table\n")
        handle.write("--------------------------------\n")
        handle.write(
            "  naloxone   mu over kappa : %6.2f x   <-- the number the brief is about\n"
            % op.naloxone_mu_over_kappa_selectivity()
        )
        handle.write(
            "  naloxone   mu over delta : %6.2f x\n"
            % op.selectivity("naloxone", "mu", "delta")
        )
        fen = op.fentanyl_mu_selectivity()
        handle.write("  fentanyl   mu over kappa : %6.2f x\n" % fen["kappa"])
        handle.write("  fentanyl   mu over delta : %6.2f x\n" % fen["delta"])
        handle.write(
            "  dynorphin  kappa over mu : %6.2f x\n"
            % op.selectivity("dynorphin_A", "kappa", "mu")
        )
        handle.write(
            "  nor-BNI    kappa over mu : %6.2f x\n"
            % op.selectivity("nor_BNI", "kappa", "mu")
        )
        handle.write(
            "  nor-BNI kappa over delta : %6.2f x\n"
            % op.selectivity("nor_BNI", "kappa", "delta")
        )
        handle.write(
            "  naltrind. delta over mu  : %6.2f x\n"
            % op.selectivity("naltrindole", "delta", "mu")
        )
        agreement = op.mu_assay_agreement()
        handle.write(
            "\n  cross-assay check, Toll mu Ki / Volpe mu Ki:"
            " fentanyl %.2f, naloxone %.2f\n" % (agreement["fentanyl"], agreement["naloxone"])
        )

        handle.write("\n\nThe two networks\n----------------\n")
        handle.write(
            "  Track C1 (hand baselines): drug-free fixed point has VTA_DA at\n"
            "  %.2e, i.e. saturated OFF, so the VTA_DA -> NAc projection carries\n"
            "  nothing at rest and a PRESYNAPTIC modulator of it cannot act.\n"
            % stability.steady_state(hand)[0][hand.index["VTA_DA"]]
        )
        handle.write("\n  " + stabilized_report.summary().replace("\n", "\n  ") + "\n")

        handle.write(
            "\n\nEach cell is the signed extremum of (dosed - drug free) over the\n"
            "%g-minute run. 'change' is (mu+kappa+delta) against (mu only), as a\n"
            "percentage of the mu-only effect. n/a where the ligand does not\n"
            "exist in the mu-only configuration.\n" % T_END
        )

        block(
            handle,
            "RE-RUN 1: Track C1 network, hand baselines (the strict regression)",
            hand,
            None,
            conditions,
        )
        block(
            handle,
            "RE-RUN 2: P3.4-stabilized network, started at its own fixed point",
            stabilized,
            op.rest_rates(stabilized),
            conditions,
        )

        handle.write(
            "\nNALOXONE REVERSAL POTENCY, mu against kappa\n"
            "-------------------------------------------\n"
            "The consequence of a 1.79-fold selectivity, made into a number.\n"
            "Naloxone concentration (nM) needed to halve an established\n"
            "effect, by search over a log ladder, on the stabilized network.\n"
            "The mu effect is read on analgesia_index (PAG, which only the mu\n"
            "system touches) and the kappa effect on accumbal_drive.\n\n"
        )
        ladder = [1.0 * (10 ** (k / 4.0)) for k in range(0, 25)]
        rows = [
            (
                "fentanyl %g nM (mu)" % FENTANYL_CLINICAL,
                "analgesia_index",
                {"fentanyl": FENTANYL_CLINICAL},
            ),
            (
                "U50488 %g nM (kappa)" % U50488_DOSE,
                "accumbal_drive",
                {"U50488": U50488_DOSE},
            ),
            (
                "dynorphin A %g nM (kappa)" % DYNORPHIN_DOSE,
                "accumbal_drive",
                {"dynorphin_A": DYNORPHIN_DOSE},
            ),
        ]
        handle.write(
            "%-28s %-18s %14s %12s\n"
            % ("agonist", "readout", "effect alone", "naloxone_50")
        )
        for label, readout_name, doses in rows:
            dose, effect = half_reversal_dose(
                stabilized, op.rest_rates(stabilized), readout_name, doses, ladder
            )
            handle.write(
                "%-28s %-18s %14.5f %12s\n"
                % (
                    label,
                    readout_name,
                    effect,
                    ("%.1f" % dose) if dose is not None else "> %.0f" % ladder[-1],
                )
            )
        handle.write(
            "\n  These are circuit-level numbers and they mix three things:\n"
            "  naloxone's own selectivity, the two agonists' different\n"
            "  affinities for their own receptors, and the different shapes of\n"
            "  the two readouts. The pharmacological version below removes the\n"
            "  last two.\n"
        )

        handle.write(
            "\nThe same question at the receptor, with the circuit removed\n"
            "-----------------------------------------------------------\n"
            "Naloxone concentration halving the AGONIST's peak occupancy, with\n"
            "each agonist dosed at 7.5x its own Kd so the two competitions\n"
            "start from the same place. This isolates naloxone.\n\n"
        )
        fine_ladder = [10 ** (k / 8.0) for k in range(0, 33)]
        kd_fentanyl_mu = 0.2599 / 0.1931  # scenarios.mu_opioid_ligands_calibrated()
        kd_u50488_kappa = op.ki("U50488", "kappa")
        matched = [
            (
                "fentanyl at mu",
                0,
                0,
                {"fentanyl": 7.5 * kd_fentanyl_mu},
                kd_fentanyl_mu,
            ),
            (
                "U50488 at kappa",
                1,
                op.LIGANDS.index("U50488"),
                {"U50488": 7.5 * kd_u50488_kappa},
                kd_u50488_kappa,
            ),
        ]
        handle.write(
            "%-22s %10s %10s %12s %12s\n"
            % ("agonist / receptor", "Kd (nM)", "dose (nM)", "peak occ", "naloxone_50")
        )
        answers = {}
        for label, system_index, ligand_index, doses, kd in matched:
            dose, alone = half_occupancy_dose(
                stabilized,
                op.rest_rates(stabilized),
                system_index,
                ligand_index,
                doses,
                fine_ladder,
            )
            answers[label] = dose
            handle.write(
                "%-22s %10.3f %10.3f %12.4f %12s\n"
                % (
                    label,
                    kd,
                    list(doses.values())[0],
                    alone,
                    ("%.2f" % dose) if dose is not None else "> %.0f" % fine_ladder[-1],
                )
            )
        if all(answers.values()):
            handle.write(
                "\n  ratio kappa / mu = %.2f, against a naloxone Ki ratio of %.2f.\n"
                % (
                    answers["U50488 at kappa"] / answers["fentanyl at mu"],
                    op.naloxone_mu_over_kappa_selectivity(),
                )
            )
        handle.write(
            "\n  That is the brief's sentence as a number. A model that treated\n"
            "  naloxone as a mu-selective antagonist would put these two\n"
            "  concentrations orders of magnitude apart; measured, they are\n"
            "  within a small factor, because 1.4 nM and 2.5 nM are within a\n"
            "  small factor. What this model still cannot show is the\n"
            "  clinically consequential half of it -- that a reversal dose also\n"
            "  strips ENDOGENOUS dynorphin off kappa -- because P6.2 has\n"
            "  supplied no tone, so the kappa agonist is injected by hand here.\n"
        )

    print("wrote %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
