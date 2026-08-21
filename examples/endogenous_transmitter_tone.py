"""Endogenous transmitter tone at every projection system (P6.2).

The brief: give every projection system a baseline endogenous transmitter
tone with cited basal concentrations, so that an administered drug competes
at a receptor pool that is already partly occupied by real transmitter rather
than an empty one, and report by how much that changes existing results.

Five parts.

  Part 1, the five cited basal concentrations, with the species, region and
  method of each and its caveats. Two of them are new in P6.2 and are the
  reason the task is not bookkeeping: noradrenaline ([PanLai1995], conscious
  rat medial prefrontal cortex, zero net flux, 1.59 nM) and acetylcholine
  ([Vinson1997], rat striatum, no-net-flux, 18.4 nM). Both stood as labelled
  unit scales through P5.8 and P6.1, which searched for them and recorded
  them empty.

  Part 2, the pool an administered drug actually arrives at. Tonic occupancy
  at every receptor in this codebase that one of the five transmitters binds,
  and the Gaddum dose ratio `1 + C/Kd` that tone imposes on EVERY drug at
  that receptor.

  Part 3, the range the literature supports. Both new constants have an
  independent measurement that disagrees with them, by 8.8-fold and 4.6-fold
  respectively; the table is re-run at each so a reader sees the spread
  rather than one number.

  Part 4, what changed. Each of the five ported scenarios re-run at the basal
  concentration P6.1 used and at P6.2's cited one. Three ports do not move at
  all, which is asserted by re-running them rather than by not looking.

  Part 5, the counterfactual the brief names. The same ports with the tone
  driven to nothing: every lone antagonist becomes inert again, which is the
  sentence every module from P5.1 to P5.7 had to write.

Species: mouse for dopamine and serotonin, rat for the other three, and
`endogenous_tone.BASAL_TONE` states every substitution against the number
rather than in a footnote.

Run with the shared Pardalos venv:
    /root/projects/pardalos/.venv/bin/python examples/endogenous_transmitter_tone.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.legacy import endogenous_tone as tone  # noqa: E402


def main():
    print(__doc__.split("Run with")[0].strip())
    print()
    for block in (
        tone.basal_report(),
        tone.tone_report(),
        tone.sensitivity_report(),
        tone.change_report(),
    ):
        print()
        print("=" * 72)
        print()
        print(block)

    print()
    print("=" * 72)
    print()
    print("What to take from this:")
    print()
    print("  * All five projection systems now carry a cited basal")
    print("    concentration. In P6.1 three did, and the other two sat at a")
    print("    unit scale of 1.0 nM whose only property was that no result")
    print("    depended on it.")
    print("  * The dose ratio column is what tone costs an administered drug,")
    print("    and it is the same number for every drug at a given receptor.")
    print("    At the striatal D2 receptor it is 3.46: 71% of the pool is")
    print("    already held by dopamine, so a drug needs 3.46 times the")
    print("    concentration it would need against an empty receptor.")
    print("  * An antagonist has nothing to do until there is tone. That is")
    print("    the last block, and it is why P6.2 is a modelling change and")
    print("    not a table of constants.")


if __name__ == "__main__":
    main()
