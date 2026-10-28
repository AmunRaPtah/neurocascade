#!/usr/bin/env python3
"""Freeze the P34.1 candidate (P34.3) into results/P34.3.json.

Records the sha256 of the frozen candidate's code, of its params file and of
every module its predict path and the fresh-set feature assembly run, before
any fresh held-out truth exists (docs/prereg-p34-go.md, order step 2). The
candidate is parameter-free, so "fitted on all train drugs" leaves the params
file with zero fitted parameters. tests/test_p34_3_frozen_candidate.py checks
that the frozen path reproduces the P34.1 selection on train drugs.

    PYTHONPATH=src python scripts/p34_3_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "P34.3.json"
V2 = "src/neurocascade/benchmark/v2/"

CODE = V2 + "frozen_candidate.py"
PARAMS = V2 + "frozen/p34_candidate.json"
# Modules the predict path imports, and the harness helpers whose definitions
# fix the candidate's two views and the baselines it is compared with.
DEPENDS = [
    V2 + "candidate_v2.py",
    V2 + "candidate.py",
    V2 + "candidate_benchmark.py",
    V2 + "candidate_v2_benchmark.py",
    V2 + "chain_benchmark.py",
]


def sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from neurocascade.benchmark.v2.frozen_candidate import load_params

    params = load_params()
    p341 = json.loads((ROOT / "results" / "P34.1.json").read_text())
    out = {
        "task": "P34.3",
        "family": params["family"],
        "selected_in": "P34.1",
        "p34_1_candidate": p341["candidate"],
        "code": {"path": CODE, "sha256": sha256(CODE)},
        "params": {"path": PARAMS, "sha256": sha256(PARAMS),
                   "n_fitted_parameters": params["n_fitted_parameters"]},
        "depends": [{"path": p, "sha256": sha256(p)} for p in DEPENDS],
        "liability_receptors": params["liability_receptors"],
        "l2": params["l2"],
        "fitted_on": params["fitted_on"],
        "fresh_truth_existed_at_freeze": (ROOT / V2 / "truth_fresh").exists(),
        "note": ("After this commit nothing edits the frozen files; P34.5 scores this candidate once on the "
                 "P34.4 fresh set. The score is a within-cohort rank sum, so P34.5 ranks within the fresh "
                 "common set."),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"code": out["code"]["sha256"][:12], "params": out["params"]["sha256"][:12],
                      "fresh_truth_existed_at_freeze": out["fresh_truth_existed_at_freeze"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
