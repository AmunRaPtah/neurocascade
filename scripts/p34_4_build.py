#!/usr/bin/env python3
"""P34.4: run the P34.2-frozen build scripts, unchanged, on the fresh pool.

Each frozen script is imported as a module and its path constants are pointed
at truth_fresh/ and a separate cache under $NC_V2_CACHE/fresh/, so the v2
truth, inputs and caches are never touched. Before a stage runs, the script's
sha256 is checked against results/P34.2.json: a stage refuses to run a file
that changed after the freeze.

    python scripts/p34_4_build.py labels    # v2_liabilities.py (offline)
    python scripts/p34_4_build.py summary   # results/P34.4.json, power floor

The common set needs a label, Cmax,u and a receptor Ki, so it is a subset of
the labelled drugs: labelled positives/negatives bound it from above. If no
liability reaches the floor on labels alone, Cmax and Ki cannot change the
outcome and the task ends EMPTY without building them.

Imports nothing from the candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
FRESH = V2 / "truth_fresh"
CACHE = Path(os.environ.get("NC_V2_CACHE", "/root/nc-v2-cache")) / "fresh"
DRUGS = FRESH / "drugs.json"


def check_frozen(rel: str) -> None:
    pinned = {f["path"]: f["sha256"] for f in
              json.loads((ROOT / "results" / "P34.2.json").read_text())["frozen_files"]}
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    if pinned.get(rel) != got:
        raise SystemExit(f"{rel} is not the P34.2-frozen file (sha {got[:12]})")


def stage_labels() -> int:
    check_frozen("scripts/v2_liabilities.py")
    check_frozen("src/neurocascade/benchmark/v2/liability_rules.json")
    import v2_liabilities as L
    L.DRUGS = DRUGS
    L.OUT = FRESH / "liabilities.json"
    L.CACHE = CACHE / "liab" / "labels.jsonl.gz"
    for argv in (["--build-cache"], []):
        sys.argv = ["v2_liabilities.py", *argv]
        if L.main():
            return 1
    return 0


FLOOR = 8  # docs/prereg-p34-go.md power floor, per class


def counts(liab: dict) -> dict[str, dict[str, int]]:
    out = {}
    for L in liab["liabilities"]:
        ys = [d[L].get("label") for d in liab["drugs"]]
        out[L] = {"labelled_pos": ys.count(1), "labelled_neg": ys.count(0),
                  "unlabelled": sum(y is None for y in ys)}
    return out


def stage_summary() -> int:
    pool = json.loads((FRESH / "pool.json").read_text())
    liab = json.loads((FRESH / "liabilities.json").read_text())
    c = counts(liab)
    reach = [L for L, v in c.items() if v["labelled_pos"] >= FLOOR and v["labelled_neg"] >= FLOOR]
    best = max(c, key=lambda L: min(c[L]["labelled_pos"], c[L]["labelled_neg"]))
    excl: dict[str, int] = {}
    for m in pool["members"]:
        if m["excluded"]:
            excl[m["excluded"]] = excl.get(m["excluded"], 0) + 1
    result = "COMPLETE" if reach else "EMPTY"
    out = {
        "task": "P34.4", "prereg": "docs/prereg-p34-go.md", "result": result,
        "pool": {"n": pool["n_pool"], "n_with_fda_label": pool["n_included"], "exclusions": excl},
        "labels_rules_sha256": liab["rules_sha256"],
        "power_floor": {"positives": FLOOR, "negatives": FLOOR},
        "labelled_counts_upper_bound_on_common_set": c,
        "liabilities_reaching_floor": reach,
        "closest_liability": best,
        "cmax_and_ki_built": bool(reach),
        "gap": None if reach else (
            f"no liability reaches {FLOOR}/{FLOOR} even before the Cmax and Ki filters: best is "
            f"{best} with {c[best]['labelled_pos']} positives / {c[best]['labelled_neg']} negatives "
            f"among {pool['n_included']} fresh drugs with an FDA label (of {pool['n_pool']} in the pool)"),
    }
    (ROOT / "results" / "P34.4.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"TASK_RESULT: {result}", out["gap"] or reach)
    return 0


STAGES = {"labels": stage_labels, "summary": stage_summary}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: p34_4_build.py {{{'|'.join(STAGES)}}}")
    sys.exit(STAGES[sys.argv.pop(1)]())
