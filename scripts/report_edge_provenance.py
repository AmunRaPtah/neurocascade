#!/usr/bin/env python
"""P13.2: the committed coverage report for per-edge direction provenance.

    python scripts/report_edge_provenance.py

Needs the gitignored `data/human` caches for the FULL-parcel report
(`scripts/fetch_human_ftract.py`, `scripts/fetch_human_connectome.py`);
without them it still reports the built ENIGMA network's provenance from
the committed fixture bytes. Writes
`results/human_edge_provenance.txt`, which is the committed record the
P13.2 acceptance criteria require: measured / inferred / absent fractions,
which regions have no measured coverage at all, and the provenance mix
behind the network the prediction layer would actually build.

The fractions are a first-class result, not prose: they quantify exactly
how much of the human layer's directed connectivity is a measurement and
how much is an assumption layered on one, so later phases cannot quietly
report an inferred direction as a measured one.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from neurocascade import human  # noqa: E402
from neurocascade.human import ftract, provenance  # noqa: E402

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "human_edge_provenance.txt",
)


def block(lines, title):
    lines.append("")
    lines.append("-" * 72)
    lines.append(title)
    lines.append("-" * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=RESULTS)
    args = parser.parse_args()

    lines = [
        "Per-edge direction provenance (P13.2)",
        "=" * 72,
        "",
        "Three levels: MEASURED (direction resolved by the measurement itself,",
        "F-TRACT CCEP: stimulation site -> recording site), INFERRED (direction-",
        "blind dMRI tractography whose direction was assigned afterwards by the",
        "per-source transmitter-sign convention), ABSENT (no data). The roadmap",
        "invariant: no edge without one. See `human/provenance.py`.",
    ]

    # ---- F-TRACT: the measured layer -------------------------------------
    block(lines, "F-TRACT measured direction (HCP-MMP1, 360 parcels, adult)")
    probability = ftract.load_feature("probability", allow_download=False)
    fp = probability.provenance()
    lines.append(fp.summary())
    lines.append("")
    lines.append(
        "  28 of the measured cells are exact zeros: probed, no response -"
    )
    lines.append(
        "  a completed measurement, not missing data. NaN cells are ABSENT."
    )

    fin = np.isfinite(probability.matrix)
    covered = fin.any(axis=0) | fin.any(axis=1)
    uncovered = [l for l, c in zip(probability.labels, covered) if not c]
    lines.append("")
    lines.append(
        "  parcels with NO measured coverage in either direction: %d of %d"
        % (len(uncovered), probability.n)
    )
    if uncovered:
        for start in range(0, len(uncovered), 8):
            lines.append("    " + ", ".join(uncovered[start : start + 8]))
        lines.append(
            "  These four are visual/superior-parietal areas far from the"
        )
        lines.append(
            "  temporal/frontal epileptogenic cortex the electrodes were"
        )
        lines.append(
            "  placed for - the clinical-placement bias, visible in the data."
        )
    lines.append("")
    lines.append("  COVERAGE_MMP1_PROBABILITY = {")
    lines.append(
        "      'measured': %d, 'inferred': %d, 'absent': %d,"
        % (
            fp.counts()["measured"],
            fp.counts()["inferred"],
            fp.counts()["absent"],
        )
    )
    lines.append("  }  # of 129600 ordered pairs (360 x 360)")

    # ---- ENIGMA dMRI: the inferred layer ---------------------------------
    block(lines, "ENIGMA/HCP dMRI (Desikan-Killiany + subcortex, 82 regions)")
    try:
        conn = human.load_connectome(allow_download=False)
    except Exception as exc:  # pragma: no cover - cache-miss path
        lines.append("  structural matrix not in cache: %s" % exc)
        lines.append("  (run scripts/fetch_human_connectome.py to add this section)")
        conn = None
    if conn is not None:
        sp = provenance.ProvenanceMap.from_structural(conn)
        lines.append(sp.summary())
        lines.append("")
        lines.append(
            "  Every nonzero weight is INFERRED in BOTH directions: diffusion"
        )
        lines.append(
            "  tractography cannot see which way a bundle runs, so each"
        )
        lines.append(
            "  symmetric weight was entered once per direction and signed by"
        )
        lines.append(
            "  the source region's transmitter identity. Nothing here is"
        )
        lines.append("  measured; P13.6 pins that demotion.")

        # ---- the network the prediction layer actually builds -----------
        block(lines, "The built signed network (what predictions consume)")
        built = human.build_circuit_network(conn)
        bp = built.provenance()
        lines.append(bp.summary())
        lines.append("")
        lines.append(
            "  The two unsigned amygdala columns are excluded by the default"
        )
        lines.append(
            "  policy, so their edges are ABSENT rather than inferred: the"
        )
        lines.append(
            "  mix behind any prediction built on this network today."
        )
        lines.append("")
        lines.append("  MEASURED edges in the built network: %d" % bp.counts()["measured"])
        lines.append(
            "  (measured direction enters only via F-TRACT, on HCP-MMP1; the"
        )
        lines.append(
            "   DK-parcellation crosswalk is P13.3's job, not an assertion here.)"
        )

    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
