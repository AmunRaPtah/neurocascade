#!/usr/bin/env python3
"""Correct the PMID in P21.2's brief, and say what the brief got wrong about it.

P21.2's brief was written on the assumption that "Uchida 2011 (PMID 21694629) is
a pooled systematic review publishing exactly this table". Running P21.2 found
two things wrong with that sentence.

There are two Uchida 2011 papers, same six authors, same journal, same year,
confusable titles. PMID 21694629, 31(4):497-502, "Dopamine D2 receptor occupancy
and clinical effects", is the one this repository already uses for the 77-78%
extrapyramidal threshold. It carries no per-drug plasma ED50 table. The table
Phase 21 wants is in the sibling, PMID 21508857, 31(3):318-25, "Predicting
dopamine D2 receptor occupancy from plasma levels of antipsychotic drugs".

And the sibling is paywalled. Five independent checks agree (Unpaywall, OpenAlex,
Semantic Scholar, Europe PMC, PMC), there is no repository deposit, and the
abstract prints prediction-error statistics with no ED50 value and no
concentration unit anywhere in it. So the brief's "attempt it first" is not a
step that can succeed, and a reader following the brief would spend the task
re-discovering that.

The brief is corrected in place rather than annotated, because a brief is an
instruction to whoever runs the task next, and an instruction that sends them to
the wrong paper is a defect in the instruction. What the brief was wrong about is
recorded here, and the citation fix it triggered in
`src/neurocascade/human/occupancy_validation.py` is pinned by
`UCHIDA_2011_IS_TWO_PAPERS` and a test.

Kept in-tree so the backlog change is reviewable as code, matching
_add_phases_20_24.py, _add_p20_6.py and _reorder_p21_reachability_first.py.
"""

from __future__ import annotations

import json
from pathlib import Path

BACKLOG = Path(__file__).resolve().parent / "backlog.json"

WRONG = "Uchida 2011 (PMID 21694629) is a pooled systematic review publishing exactly this table and is currently cited in this repository only in prose. Attempt it first, then the primary PET papers behind it."

RIGHT = (
    "The per-drug plasma ED50 table is in Uchida 2011 PMID 21508857, "
    "31(3):318-25, not in its sibling PMID 21694629, 31(4):497-502, which is "
    "the clinical-effects paper this repository already cites for the 77-78% "
    "extrapyramidal threshold. 21508857 was checked and is paywalled at every "
    "route tried, and its abstract prints no ED50 and no concentration unit, "
    "so go to the primary PET papers per drug and treat the pooled table as "
    "unreachable."
)


def main() -> int:
    data = json.loads(BACKLOG.read_text())
    (task,) = [t for t in data["tasks"] if t["id"] == "P21.2"]

    if RIGHT in task["brief"]:
        print("already corrected, nothing to do")
        return 0
    if WRONG not in task["brief"]:
        print("P21.2 brief does not contain the sentence this script replaces")
        return 1

    task["brief"] = task["brief"].replace(WRONG, RIGHT)
    BACKLOG.write_text(json.dumps(data, indent=2) + "\n")
    print("P21.2 brief now reads:\n")
    print(task["brief"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
