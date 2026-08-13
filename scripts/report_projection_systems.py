"""P6.1: the projection-system abstraction, reported.

Writes `results/projection_systems.txt`. Four blocks:

  1. The volume-transmission timescale argument, computed rather than
     asserted: the release-site spacing the well-mixed-region assumption
     rests on, the diffusion time over it against the diffusion time across
     a region, and each transmitter's extracellular clearance time constant
     against the circuit's.
  2. The nuclei, their CCF acronyms and ids, and the firing-rate context for
     what each source node's dimensionless resting rate stands for, with the
     one gap recorded as a gap.
  3. Each ported scenario: its nuclei, its target regions with their
     predicted drug-free concentrations, where each receptor sits, and the
     standing tonic drive the compensation folds into the baselines.
  4. The two results: the autoreceptor loop closed end to end, and an
     antagonist given alone that is no longer inert.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python scripts/report_projection_systems.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from neurocascade import projection_scenarios as pscen  # noqa: E402
from neurocascade import projection_systems as ps  # noqa: E402


OUT = os.path.join(HERE, "..", "results", "projection_systems.txt")


def main():
    blocks = [ps.timescale_report()]

    citations = ["", "Citations", "-" * 72]
    for key in sorted(ps.CITATIONS):
        citations.append("[%s] %s" % (key, ps.CITATIONS[key]))
        citations.append("")
    blocks.append("\n".join(citations))

    for key in pscen.PORT_ORDER:
        blocks.append(pscen.port(key).report())

    blocks.append(pscen.summary())

    text = ("\n\n" + "=" * 72 + "\n\n").join(blocks) + "\n"
    with open(OUT, "w") as handle:
        handle.write(text)
    print(text)
    print("written to %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
