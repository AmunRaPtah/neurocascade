#!/usr/bin/env python3
"""Add P20.6, the FDA data-route spike, to tasks/backlog.json.

The spike was run before it was written down, which is the wrong order and is
recorded here as such. It was run first because Phase 22's whole design rests on
one assumption: that an approved oral daily dose range for schizophrenia is
retrievable per drug, free, for enough drugs to make a test. If that assumption
had been wrong, P22.2 would have been written, scaffolded and marked done
against a dose range someone here had chosen, which is the failure this phase
exists to prevent. Checking reachability before committing to the design is
Rule 12; doing it without a task record is the part to not repeat.

What it found is worth more than the code: two filters that were not in the
original design, both discovered by reading labels rather than by reasoning
about them. Recorded as acceptance criteria so they cannot be quietly dropped.

Kept in-tree so the backlog change is reviewable as code, matching
_add_phases_20_24.py.
"""

from __future__ import annotations

import json
from pathlib import Path

BACKLOG = Path(__file__).resolve().parent / "backlog.json"

TASK = {
    "id": "P20.6",
    "phase": "Phase 20: process repair, so the build can hold a validated result",
    "title": "Spike: is the FDA approved dose range actually reachable per drug?",
    "brief": (
        "Phase 22 tests the chain's predicted 65-to-78-percent D2 occupancy "
        "window against the FDA-approved oral daily dose range for "
        "schizophrenia, which is held-out ground truth because it was set by "
        "the registration trials and has never entered any fit here. Before "
        "designing against that, establish it is reachable, for how many drugs, "
        "and with what traps. Fetch and snapshot the current label per drug "
        "from openFDA, pin the bytes by SHA-256 in a committed provenance "
        "module because data/ is gitignored, and record every drug that comes "
        "back empty as a searched-and-empty result rather than a gap. Do not "
        "parse a dose out of the label: the section is prose that separates "
        "initial from maintenance from maximum and adult from adolescent, so a "
        "regex over it would produce a number that looks sourced and is not. "
        "Transcription is a separate attributed step in P22.2."
    ),
    "acceptance": [
        "scripts/fetch_fda_labels.py snapshots labels and writes provenance; "
        "it parses no dose",
        "src/neurocascade/human/fda_sources.py pins application number, "
        "revision date and SHA-256 per drug, and is committed",
        "Label selection filters on oral route before preferring an NDA, "
        "because the most recent NDA is a depot injection for 3 of 9 drugs",
        "Label selection filters on indication, accepting the pre-modern "
        "wording 'management of manifestations of psychotic disorders' with "
        "which phrase matched recorded per drug",
        "Drugs with no usable US oral label are recorded as "
        "searched-and-empty with the reason, not dropped",
        "The reachable drug count is stated, and P22.2 is scoped to it",
    ],
    "validation": {
        "kind": "searched-and-empty",
        "evidence": (
            "8 of 9 drugs have a usable US oral label. Amisulpride has none: "
            "its only US application is Barhemsys, intravenous, for "
            "postoperative nausea and vomiting, so it is excluded from the "
            "label-dose test in P22.2 and retained for the PET test in P22.3. "
            "Haloperidol has no current innovator oral label either, only 68 "
            "ANDA copies of a withdrawn one, which is recorded on the row "
            "rather than smoothed over because haloperidol is the drug the "
            "chain is validated on."
        ),
    },
    "deps": [],
    "status": "done",
    "attempts": 0,
}


def main() -> int:
    data = json.loads(BACKLOG.read_text())
    tasks = data["tasks"]
    if any(t["id"] == TASK["id"] for t in tasks):
        print(f"{TASK['id']} already present, nothing to do")
        return 0

    # After the last Phase 20 task, so the phase stays contiguous.
    last_p20 = max(i for i, t in enumerate(tasks) if t["id"].startswith("P20."))
    tasks.insert(last_p20 + 1, TASK)
    BACKLOG.write_text(json.dumps(data, indent=2) + "\n")
    print(f"inserted {TASK['id']} at index {last_p20 + 1}, {len(tasks)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
