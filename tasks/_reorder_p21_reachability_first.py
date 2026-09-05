#!/usr/bin/env python3
"""Put the P21 reachability check before the registry refactor it justifies.

As written, P21.2 (can per-drug D2 EC50 values actually be obtained?) depends on
P21.1 (turn the haloperidol constant into a drug-keyed registry). That is the
wrong way round, for the reason P20.6 had just finished demonstrating: a registry
keyed by drug is a container shaped by what its sources print, and building it
before knowing whether four drugs are reachable risks two failures at once. If
the values do not exist, the refactor is scaffolding for data that never
arrives, which is how 23 tasks in Phases 13-19 came to be. And if they do exist
but print a basis, an interval or a binding fraction the schema did not
anticipate, the schema gets bent to fit rather than the other way round.

P21.2's own acceptance criteria already say "no ingestion code written in this
task" and "fewer than four drugs reachable means BLOCKED_ON_DATA, not a partial
build". Those only bind if the check runs first. Rule 12 is not a step in a
sequence, it is the gate on the sequence.

So the dependency inverts: P21.2 now depends on P20.5, and P21.1 on P21.2. The
task order in the file follows, so a reader sees the gate before the work it
gates. IDs are left alone: renumbering would break every reference in the
commit log, the validation ledger and the roadmap, to buy nothing.

Kept in-tree so the backlog change is reviewable as code, matching
_add_phases_20_24.py and _add_p20_6.py.
"""

from __future__ import annotations

import json
from pathlib import Path

BACKLOG = Path(__file__).resolve().parent / "backlog.json"

#: Why the inversion, recorded on P21.1 itself so it is visible where the
#: surprising dependency is rather than only in this script.
NOTE_ON_P21_1 = (
    " Sequenced after P21.2, not before it: the registry's schema is decided by "
    "what the sources actually print, and if fewer than four drugs are "
    "reachable this refactor should not happen at all."
)


def main() -> int:
    data = json.loads(BACKLOG.read_text())
    tasks = data["tasks"]
    by_id = {t["id"]: t for t in tasks}

    for task_id in ("P21.1", "P21.2"):
        if task_id not in by_id:
            print(f"{task_id} missing from the backlog; nothing to do")
            return 1

    if by_id["P21.2"]["deps"] == ["P20.5"]:
        print("already reordered, nothing to do")
        return 0

    by_id["P21.2"]["deps"] = ["P20.5"]
    by_id["P21.1"]["deps"] = ["P21.2"]
    if NOTE_ON_P21_1.strip() not in by_id["P21.1"]["brief"]:
        by_id["P21.1"]["brief"] += NOTE_ON_P21_1

    # Move P21.2 ahead of P21.1 in file order so the gate reads before the work.
    tasks.remove(by_id["P21.2"])
    tasks.insert(tasks.index(by_id["P21.1"]), by_id["P21.2"])

    BACKLOG.write_text(json.dumps(data, indent=2) + "\n")
    order = [t["id"] for t in tasks if t["id"].startswith("P21.")]
    print(f"P21 order is now {order}")
    print(f"  P21.2 deps -> {by_id['P21.2']['deps']}")
    print(f"  P21.1 deps -> {by_id['P21.1']['deps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
