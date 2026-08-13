"""P6.3: the derived projection densities, reported.

Writes `results/projection_densities.txt`. Five blocks:

  1. The derived table itself: every (source nucleus, target region) density
     from the committed `projection_densities.csv`, terminal rows first and
     the recorded-but-refused nuclear rows after them.
  2. Every region's predicted basal extracellular concentration per
     transmitter, with the one MEASURED region in each marked as such and
     every other one marked as the prediction it is.
  3. The validation: five predicted concentration ratios against held-out
     measurements from four papers. One is in range, one is 4.2x out
     against a second measurement of the same ratio that disagrees with the
     first, and two are in the wrong direction. Nothing here is fitted.
  4. The [Lin2023] comparison, which needs the gitignored cache: the derived
     tuberomammillary density against a measured, cell-type-specific,
     whole-brain histaminergic fibre map in the same atlas. Skipped with a
     note when the cache is absent.
  5. P6.3's stability block: P3.4's check on the whole coupled system with
     each autoreceptor loop open and closed, and the effective self-weight
     each closed loop adds to its nucleus.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python scripts/report_projection_densities.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from neurocascade import projection_density as pdens  # noqa: E402
from neurocascade import projection_scenarios as pscen  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "projection_densities.txt")


def main():
    table = pdens.load_table()
    blocks = [table.report(), pdens.concentration_report(table)]
    blocks.append(table.validation_report())

    try:
        blocks.append(pdens.fibre_density_report())
    except Exception as error:  # pragma: no cover - depends on the cache
        blocks.append(
            "The [Lin2023] fibre-density comparison needs the gitignored\n"
            "data/connectome cache (run scripts/fetch_connectome.py); it is\n"
            "over regions this model has no node for, so the committed\n"
            "table does not carry them.\n\n%s: %s"
            % (type(error).__name__, error)
        )

    blocks.append(pscen.stability_report())

    citations = ["", "Citations", "-" * 72]
    for key in sorted(pdens.CITATIONS):
        citations.append("[%s] %s" % (key, pdens.CITATIONS[key]))
        citations.append("")
    citations.append("Searched and empty")
    citations.append("-" * 72)
    citations.append(pdens.SEARCHED_AND_EMPTY.strip())
    blocks.append("\n".join(citations))

    text = ("\n\n" + "=" * 72 + "\n\n").join(blocks) + "\n"
    with open(OUT, "w") as handle:
        handle.write(text)
    print(text)
    print("written to %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
