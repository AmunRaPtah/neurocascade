"""Host-side GO / NO-GO verdict for the v2 chain (P33.4 gate on P32.x).

Human-owned: lives under pipeline/, which architect sessions cannot edit, and
runs only from a task's host_eval on the host clone (full v2 truth). The rule is
the one pre-registered in P33.3's brief on 2026-10-21 and operationalised in
docs/prereg-p33-4-go.md (sealed 2026-10-26, before any candidate existed):

  GO iff, for at least one scored liability, the candidate's held-out AUROC
  beats the BEST baseline with the 95% paired-bootstrap CI of
  (AUROC_candidate - AUROC_best_baseline) entirely above 0.

"Best baseline" is the baseline with the highest held-out AUROC on the same
drugs. A liability is scored only with >= MIN_CLASS positives and negatives
among drugs where the candidate and every baseline have a score.

Input: JSON written by the task's held-out dump, never committed:
  {"liabilities": {name: {"drugs": [inn...], "y": [0/1...],
                          "candidate": [score...],
                          "baselines": {name: [score or null...]}}}}
Prints a table and exactly one line `NC_GO_VERDICT: GO` or
`NC_GO_VERDICT: NO-GO`, which the conductor turns into the go marker.
"""
from __future__ import annotations

import json
import sys

import numpy as np

MIN_CLASS = 5
N_BOOT = 2000
SEED = 20261026


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def judge(name: str, d: dict, rng: np.random.Generator) -> tuple[bool, str]:
    y = np.asarray(d["y"], dtype=int)
    cand = np.asarray([np.nan if v is None else v for v in d["candidate"]], dtype=float)
    base = {k: np.asarray([np.nan if v is None else v for v in vs], dtype=float)
            for k, vs in d["baselines"].items()}
    if not base:
        return False, f"{name}: no baselines, not scored"
    ok = ~np.isnan(cand)
    for b in base.values():
        ok &= ~np.isnan(b)
    y, cand = y[ok], cand[ok]
    base = {k: b[ok] for k, b in base.items()}
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos < MIN_CLASS or nneg < MIN_CLASS:
        return False, f"{name}: underpowered (pos {npos}, neg {nneg}, common n {len(y)})"
    a_c = auroc(y, cand)
    a_b = {k: auroc(y, b) for k, b in base.items()}
    best = max(a_b, key=a_b.get)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            diffs.append(auroc(y[i], cand[i]) - auroc(y[i], base[best][i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    go = lo > 0
    bl = ", ".join(f"{k} {v:.3f}" for k, v in sorted(a_b.items()))
    return go, (f"{name}: n {len(y)} (pos {npos}); candidate {a_c:.3f}; baselines {bl}; "
                f"best {best}; diff 95% CI [{lo:+.3f}, {hi:+.3f}] -> {'beats' if go else 'does not beat'}")


def main(path: str) -> int:
    data = json.load(open(path))
    rng = np.random.default_rng(SEED)
    any_go = False
    print("==== P33.4 GO verdict (pipeline/go_verdict.py) ====")
    for name in sorted(data.get("liabilities", {})):
        go, line = judge(name, data["liabilities"][name], rng)
        any_go |= go
        print(line)
    print(f"NC_GO_VERDICT: {'GO' if any_go else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
