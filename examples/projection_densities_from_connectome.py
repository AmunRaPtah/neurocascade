"""Projection densities derived from the connectome, and the loops they close (P6.3).

The brief: derive per-system projection densities from the Phase 3
connectome by taking the outgoing connection strengths of each source
nucleus, rather than assigning them by hand; include autoreceptor feedback
(D2 on VTA/SNc, 5-HT1A on raphe, alpha2 on LC); and verify stability
explicitly with the P3.4 check after the feedback is closed.

Five parts.

  Part 1, the derivation. One formula over the Knox et al. 2019 matrices
  P3.1 already ingested, and the two normalization choices in it, both
  forced by what a projection density is FOR.

  Part 2, what the map predicts. Every region's basal extracellular
  concentration per transmitter, with the ONE measured region in each marked
  as measured and everything else marked as the prediction it is.

  Part 3, the validation. Five predicted ratios against held-out
  measurements from four papers, and the one whole-brain, same-atlas,
  cell-type-specific check that the derivation method itself passes.

  Part 4, the one density this task did NOT derive, and the seven
  measurements that bound it.

  Part 5, the third autoreceptor loop closed, what it costs cocaine, and
  P3.4's check run where it can see the loop.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/projection_densities_from_connectome.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from neurocascade import monoamine_transporters as mt  # noqa: E402
from neurocascade.legacy import projection_density as pdens  # noqa: E402
from neurocascade.legacy import projection_scenarios as pscen  # noqa: E402


def rule(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    table = pdens.load_table()

    rule("1. The derivation")
    print(
        "rho(nucleus -> region) = sum strength / sum target voxels, i.e. Knox\n"
        "et al. 2019's connection_density at this grouping. Divide by the\n"
        "TARGET's size because a concentration is an amount per volume; do NOT\n"
        "divide by the source's, because the release law consumes a\n"
        "dimensionless firing-rate ratio and the number of releasing neurons\n"
        "has to stay inside rho. Only ratios are used: the reference region's\n"
        "measured basal level fixes the scale.\n"
    )
    print(table.report())

    rule("2. What the map predicts")
    print(pdens.concentration_report(table))

    rule("3. The validation")
    print(table.validation_report())
    print()
    try:
        print(pdens.fibre_density_report())
    except Exception as error:
        print(
            "[Lin2023] comparison skipped: it is over regions this model has\n"
            "no node for, so it needs the gitignored data/connectome cache.\n"
            "%s: %s" % (type(error).__name__, error)
        )

    rule("4. The one density that is not derived")
    print(
        "A source nucleus's own pool is somatodendritic release, which no\n"
        "anterograde tracer measures, and the tracer's own diagonal would put\n"
        "2.6 uM of serotonin in the dorsal raphe -- past SERT's measured Vmax,\n"
        "so the mass balance has no finite steady state at all.\n"
    )
    for key in pscen.PORT_ORDER:
        scenario = pscen.port(key)
        for region, _, relative, provenance in scenario.projection.density_report():
            if provenance == pdens.SOMATODENDRITIC_PROVENANCE:
                print(
                    "  %-14s %-5s %-30s %.4g nM"
                    % (
                        key,
                        region,
                        provenance,
                        scenario.projection.basal_concentration(region),
                    )
                )
    print()
    print(pdens.SEARCHED_AND_EMPTY.strip())

    rule("5. The loops, closed, and P3.4's check where it can see them")
    print(
        "P6.1 placed 5-HT1A in the raphe and alpha2 at the locus coeruleus and\n"
        "deliberately left the D2 autoreceptor out. P6.3 adds it, reusing\n"
        "scenarios.d2_coupling()'s own -3.0 rather than inventing a gain.\n"
    )
    closed = pscen.port("dopamine")
    opened = pscen.port("dopamine", autoreceptor=False)
    for label, scenario in (("loop open", opened), ("loop closed", closed)):
        _, sim, dosed = scenario.run(cocaine=1000.0)
        net, _, _ = scenario.run(cocaine=1000.0)
        ligands = [lig.name for lig in scenario.projection.ligands]
        row = sim.n_ligands - scenario.projection.n + ligands.index("cocaine")
        present = dosed["concentrations"][row][-1]
        print(
            "  %-12s cocaine raises striatal dopamine %.4f-fold "
            "(P5.8's law: %.4f), SNc rate %.4f"
            % (
                label,
                scenario.field_fold(sim, dosed, "CP")[-1],
                1.0 + present / mt.ki("cocaine", "DAT"),
                scenario.rate_trace(net, dosed, "SNc_DA")[-1],
            )
        )
    print(
        "\nThe gap is the autoreceptor: P5.8's law holds RELEASE fixed and the\n"
        "loop does not. Blocking uptake raises dopamine at the VTA and SNc,\n"
        "which lowers their firing, which lowers release.\n"
    )
    print(pscen.stability_report())
    print(
        "\nAll eight self-weights are negative, because all three receptors are\n"
        "Gi-coupled on the cell that releases the transmitter. A negative\n"
        "self-weight moves a logistic node's Jacobian entry further left, and\n"
        "the pool it acts through relaxes 40-2000x faster than the circuit, so\n"
        "there is no lag for it to oscillate around."
    )

    rule("What is a measurement here and what is not")
    print(
        "MEASURED: every terminal-field projection density (derived from\n"
        "  [Knox2019]); every reference region's basal concentration, each in\n"
        "  the region it was measured in; the five validation ratios.\n"
        "ILLUSTRATIVE, labelled: the somatodendritic density at each source\n"
        "  nucleus, and every coupling gain, unchanged from P5.x.\n"
        "NOT KNOWN: cell-type-specific projections for four of the five\n"
        "  transmitters, and a region-wise uptake Vmax for any of them."
    )


if __name__ == "__main__":
    main()
