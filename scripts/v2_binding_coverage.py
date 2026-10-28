#!/usr/bin/env python3
"""Train-split coverage of the v2 binding input, per liability.

``python scripts/v2_binding_coverage.py [--before OLD_binding.json] [--after NEW]``

For each liability in ``candidate.LIABILITY_RECEPTORS`` prints, over the v2
TRAIN drugs only (``v2_partition()[0]``; no held-out drug is listed or
counted, and nothing is written to disk):

* ``measured``: drugs with a measured Ki (``_ki_by_gene`` non-None) at >= 1 of
  the liability's receptors;
* ``bound_only``: drugs with no measured Ki at any of those receptors but a
  censored lower bound (``ki_lower_bounds``) at >= 1 of them;
* ``either``: the union.

Output is a markdown table, one row per liability, before/after when
``--before`` is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurocascade.benchmark.v2 import v2_partition
from neurocascade.benchmark.v2.candidate import LIABILITY_RECEPTORS
from neurocascade.benchmark.v2.candidate_benchmark import _ki_by_gene

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "src" / "neurocascade" / "benchmark" / "v2" / "inputs" / "binding.json"


def coverage(binding: dict, train: tuple[str, ...]) -> dict[str, dict[str, int]]:
    out = {}
    for liab, receptors in LIABILITY_RECEPTORS.items():
        measured = bound_only = 0
        for inn in train:
            kis = _ki_by_gene(binding, inn)
            if any(kis.get(g) is not None for g in receptors):
                measured += 1
            elif any(g in binding.get(inn, {}).get("ki_lower_bounds", {}) for g in receptors):
                bound_only += 1
        out[liab] = {"measured": measured, "bound_only": bound_only,
                     "either": measured + bound_only}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path)
    ap.add_argument("--after", type=Path, default=DEFAULT)
    args = ap.parse_args(argv)
    train, _ = v2_partition()
    after = coverage(json.loads(args.after.read_text())["drugs"], train)
    before = (coverage(json.loads(args.before.read_text())["drugs"], train)
              if args.before else None)
    print(f"Train drugs: {len(train)}\n")
    if before:
        print("| liability | receptors | measured before | measured after "
              "| bound-only after | measured or bound after |")
        print("|---|---|---|---|---|---|")
        for liab, rec in LIABILITY_RECEPTORS.items():
            b, a = before[liab], after[liab]
            print(f"| {liab} | {', '.join(rec)} | {b['measured']} | {a['measured']} "
                  f"| {a['bound_only']} | {a['either']} |")
    else:
        print("| liability | receptors | measured | bound-only | measured or bound |")
        print("|---|---|---|---|---|")
        for liab, rec in LIABILITY_RECEPTORS.items():
            a = after[liab]
            print(f"| {liab} | {', '.join(rec)} | {a['measured']} | {a['bound_only']} "
                  f"| {a['either']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
