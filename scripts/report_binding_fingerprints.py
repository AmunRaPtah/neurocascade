"""P8.1: dump the binding-fingerprint target table and one worked prediction.

    python scripts/report_binding_fingerprints.py > results/binding_fingerprints.txt

Two things a reader needs and cannot get from the code by skimming:

1. the `kon` every fingerprint-derived ligand at every target inherits, and
   what that number's standing is, because `koff = kon * Ki` puts the entire
   binding timescale on it while leaving the affinity exact;
2. the round trip, run rather than asserted: three hand-built, calibrated
   ligand sets rebuilt from the Ki values their own sources published, with
   the agreement (and the disagreement) printed.

Needs no network and no data cache.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                            # noqa: E402

from neurocascade import fingerprint as fp                    # noqa: E402
from neurocascade import opioid, scenarios           # noqa: E402
from neurocascade.legacy import drug_pk


def one_ligand(compound, receptor_alias, ki_nM, efficacy, target):
    """A one-row fingerprint, which is what a round trip needs."""
    built = fp.BindingFingerprint.from_pdsp_rows(
        [fp.PDSPRow(receptor=receptor_alias, test_ligand=compound, ki_nM=ki_nM)],
        efficacy={target: efficacy},
    )
    return fp.ligand_set(built)[target]


def round_trip_table():
    lines = [
        "The round trip: hand-built ligands rebuilt from their published Ki",
        "=" * 78,
        "",
        "%-14s %-12s %10s %10s %10s %10s"
        % ("site", "ligand", "kon(hand)", "kon(fp)", "koff(hand)", "koff(fp)"),
        "-" * 78,
    ]

    exact, inexact = [], []

    # GABA-A benzodiazepine site: derivation 3 with a shared kon, which is
    # what this API does, so it round-trips.
    published_ki = {"diazepam": 3.6, "flumazenil": 1.0}
    efficacies = {"diazepam": 1.0, "flumazenil": 0.0}
    for hand in scenarios.gaba_a_benzodiazepine_ligands_calibrated():
        got = one_ligand(
            hand.name, "BZP Rat Brain Site", published_ki[hand.name],
            efficacies[hand.name], "GABA_A_BZD",
        )
        lines.append("%-14s %-12s %10.5g %10.5g %10.5g %10.5g"
                     % ("GABA_A_BZD", hand.name, hand.kon, got.kon,
                        hand.koff, got.koff))
        (exact if np.isclose(got.koff, hand.koff, rtol=1e-12) else inexact
         ).append(("GABA_A_BZD", hand.name))

    # Kappa and delta opioid: derivation 3 over the module's own Ki table.
    for target, receptor, hand_built in (
        ("KOR", "kappa", opioid.ligands_kappa()),
        ("DOR", "delta", opioid.ligands_delta()),
    ):
        for hand in hand_built:
            got = one_ligand(
                hand.name, receptor, opioid.ki(hand.name, receptor),
                opioid.efficacy(hand.name, receptor), target,
            )
            lines.append("%-14s %-12s %10.5g %10.5g %10.5g %10.5g"
                         % (target, hand.name, hand.kon, got.kon,
                            hand.koff, got.koff))
            (exact if np.isclose(got.koff, hand.koff, rtol=1e-12) else inexact
             ).append((target, hand.name))

    # Mu opioid: derivation 2, so it does not round-trip and the gap is the
    # point.
    published_ki = {"fentanyl": 1.346, "naloxone": 1.518}
    efficacies = {"fentanyl": 1.0, "naloxone": 0.0}
    for hand in scenarios.mu_opioid_ligands_calibrated():
        got = one_ligand(
            hand.name, "MU", published_ki[hand.name], efficacies[hand.name],
            "MOR",
        )
        lines.append("%-14s %-12s %10.5g %10.5g %10.5g %10.5g"
                     % ("MOR", hand.name, hand.kon, got.kon,
                        hand.koff, got.koff))
        (exact if np.isclose(got.koff, hand.koff, rtol=1e-12) else inexact
         ).append(("MOR", hand.name))

    lines += [
        "",
        "exact (%d): %s" % (len(exact), ", ".join("%s/%s" % p for p in exact)),
        "inexact (%d): %s"
        % (len(inexact), ", ".join("%s/%s" % p for p in inexact)),
        "",
        "Every row above reproduces its ligand's AFFINITY exactly; the two",
        "inexact rows are mu-opioid, where scenarios.py derived each kon from",
        "that ligand's own MEASURED koff [Mann2022] and a fingerprint has only",
        "the Ki. What that costs, in the quantity the measurement was made",
        "for:",
        "",
    ]
    for name, ki in (("fentanyl", 1.346), ("naloxone", 1.518)):
        hand = {l.name: l for l in scenarios.mu_opioid_ligands_calibrated()}[name]
        got = one_ligand(name, "MU", ki, 1.0 if name == "fentanyl" else 0.0,
                         "MOR")
        lines.append(
            "  %-9s receptor residence time 1/koff: %.2f min measured, "
            "%.2f min from the fingerprint, %.1fx too long"
            % (name, 1.0 / hand.koff, 1.0 / got.koff, hand.koff / got.koff)
        )
    lines += [
        "",
        "Naloxone is the one that matters: scenarios.py's own note says it is",
        "naloxone's SHORT receptor residence time, not its clearance, that",
        "makes reversal wear off before a large fentanyl exposure does. A",
        "fingerprint-derived naloxone holds the receptor almost eleven times",
        "too long, so a reversal timing predicted from binding data alone",
        "would be wrong in exactly the direction that matters clinically.",
        "",
    ]
    return lines


def worked_prediction():
    """Fentanyl-affinity analogue at MOR: the P8.1 worked example, re-run on
    the human side of the P11.2 species gate.

    The pre-P11.2 example was diazepam 2 mg/kg IP in the rat through the
    rat PK bridge onto the benzodiazepine site. That call is exactly what
    the species gate now refuses -- twice over, since both its PK set and
    the site's Kd are rodent-sourced -- so the worked example runs on MOR
    (human Ki) with a caller-supplied species-agnostic bolus model, and the
    refusal itself is printed below it.
    """
    from neurocascade.pk import OneCompartmentPK
    from neurocascade import species_gate as sg

    compound = fp.BindingFingerprint.from_pdsp_rows(
        [fp.PDSPRow(
            receptor="MU", test_ligand="novelmuagonist", ki_nM=1.346,
            source="recombinant human MOR assay standing in for [Volpe2011]",
            reference="Ki value as scenarios.mu_opioid_ligands_calibrated",
        )],
        efficacy={"MOR": 1.0},
    )
    prediction = fp.predict(
        compound, dose=148.6, route="IV", species="human", body_weight_kg=70.0,
        pk_model=OneCompartmentPK(ke=0.05332), t_span=(0.0, 120.0),
        n_times=241,
    )
    at_site = prediction["MOR"]
    lines = [
        "A worked prediction: fentanyl-affinity ligand at MOR, IV bolus, human",
        "=" * 78,
        "",
        "Fingerprint: one row, Ki 1.346 nM at mu ([Volpe2011]'s recombinant-",
        "             human-MOR value), efficacy 1.0 supplied separately.",
        "PK:          %s" % prediction.pk_model,
        "Partition:   %s" % prediction.brain_partition_status[0],
        "Compartment: %s" % at_site.compartment,
        "",
        "  peak receptor concentration      %.2f nM"
        % at_site.peak_receptor_concentration_nM,
        "  peak mu-site occupancy           %.3f at %.0f min"
        % (at_site.peak_occupancy, at_site.time_of_peak_occupancy_min),
        "  site activation at peak          %.3f"
        % float(at_site.activation.max()),
        "",
        "This is where P8.1 stops. Turning that activation into a behavioural",
        "score needs a fitted mechanism gain and a region-wise density, which",
        "is P8.2's composition and not a binding fingerprint's to supply.",
        "",
        "And the call this replaced, which the P11.2 species gate now refuses:",
        "",
    ]
    try:
        diazepam = fp.BindingFingerprint.from_pdsp_rows(
            [fp.PDSPRow(
                receptor="BZP Rat Brain Site", test_ligand="diazepam",
                ki_nM=3.6, species="RAT", source="CEREBRAL CORTEX",
                reference="Mohler & Okada, Life Sci 1977;20:2101",
            )],
            efficacy={"GABA_A_BZD":
                      (1.0, "potentiation strength; see scenarios.py")},
        )
        fp.predict(diazepam, dose=2.0, route="IP", species="rat",
                   body_weight_kg=0.25, on_duplicate="replace")
    except sg.UnsourcedParameterError as error:
        lines += ["  " + line for line in str(error).splitlines()]
    lines += [
        "",
        "The rodent parameters still exist under neurocascade.legacy and stay",
        "tested; what changed is that they cannot drive this API. See",
        "docs/roadmap-human.md's quarantine policy and",
        "docs/architecture.md's P11.2 section.",
        "",
    ]
    return lines


def main():
    from neurocascade import species_gate

    print("Binding-fingerprint input API (P8.1)")
    print("=" * 78)
    print()
    print(__doc__.strip().splitlines()[0])
    print()
    print(fp.target_report())
    print()
    print(species_gate.anchors_report())
    for line in round_trip_table():
        print(line)
    for line in worked_prediction():
        print(line)
    print("PK coverage, which is the other half a fingerprint does not carry")
    print("=" * 78)
    print()
    for pair in sorted(drug_pk.PK_BUILDERS):
        routes = drug_pk.MODELLED_ROUTES.get(pair, ())
        print("  %-22s %s" % (
            "%s / %s" % pair,
            ", ".join(routes) if routes else "no route has a sourced model",
        ))
    print()
    print("Three drugs, four (drug, species, route) triples. Every other")
    print("compound needs its PK model passed in explicitly, and predict()")
    print("raises UnsourcedParameterError rather than substituting one.")


if __name__ == "__main__":
    main()
