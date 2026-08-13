"""P6.2: endogenous transmitter tone, reported.

Writes `results/endogenous_tone.txt`. Four blocks:

  1. The five projection systems' cited basal extracellular concentrations,
     with the species, region and method of each, its caveats, and the
     independent cross-check that disagrees with it where one exists.
  2. Tonic occupancy before any drug is given, at every receptor in this
     codebase that one of those five transmitters binds, with the Gaddum
     dose ratio each tone imposes on every drug at that receptor.
  3. The same table re-run at each cross-check value, so the range the
     literature supports is visible rather than one number.
  4. What P6.2 changed in P6.1's results, computed by re-running each port
     at the basal concentration P6.1 used and at P6.2's cited one, plus the
     empty-pool counterfactual the brief names.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python scripts/report_endogenous_tone.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from neurocascade import endogenous_tone as tone  # noqa: E402


OUT = os.path.join(HERE, "..", "results", "endogenous_tone.txt")


def main():
    citations = ["Citations", "-" * 72]
    for key in sorted(tone.CITATIONS):
        citations.append("[%s] %s" % (key, tone.CITATIONS[key]))
        citations.append("")

    text = ("\n\n" + "=" * 72 + "\n\n").join(
        [tone.summary(), "\n".join(citations)]
    ) + "\n"
    with open(OUT, "w") as handle:
        handle.write(text)
    print(text)
    print("written to %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
