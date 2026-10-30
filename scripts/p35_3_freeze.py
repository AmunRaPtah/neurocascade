#!/usr/bin/env python3
"""Select and freeze the P35 candidate (P35.3) into results/P35.3.json.

Under docs/prereg-p35-go.md, order step 2. Scores every family in
``p35_selection.FAMILIES`` beside the four prereg baselines on P35-train
leave-one-drug-out, checks the selection rule picks ``p35_selection.SELECTED``,
writes the frozen params (the kNN pool of train fingerprints and train labels)
to ``frozen/p35_candidate.json``, scores the frozen predict path, and records
the sha256 of the code, the params and every module they run. No P35 held-out
label exists when this runs (asserted).

    PYTHONPATH=src python scripts/p35_3_freeze.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "P35.3.json"
V2 = "src/neurocascade/benchmark/v2/"

CODE = V2 + "p35_candidate.py"
PARAMS = V2 + "frozen/p35_candidate.json"
DEPENDS = [
    V2 + "candidate.py",
    V2 + "candidate_benchmark.py",
    V2 + "chain_benchmark.py",
    V2 + "p35_selection.py",
    V2 + "sider_labels.py",
    V2 + "sider/labels_train.json",
    V2 + "sider/universe_train.json",
    V2 + "inputs/binding.json",
]


def sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def r3(x):
    return None if x is None else round(float(x), 3)


def go_verdict_train_lines(per: dict) -> list[str]:
    """pipeline/go_verdict.py's unchanged judge, run on the TRAIN common sets.

    Informational only: the binding run is P35.4's, on held-out drugs."""
    spec = importlib.util.spec_from_file_location("go_verdict", ROOT / "pipeline" / "go_verdict.py")
    gv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gv)
    rng = np.random.default_rng(gv.SEED)
    lines = []
    for name in sorted(per):
        row = per[name]
        if "frozen_candidate_scores" not in row:
            continue
        d = {"y": row["y"], "candidate": [row["frozen_candidate_scores"][i] for i in row["common_set"]],
             "baselines": row["baseline_scores"]}
        lines.append(gv.judge(name, d, rng)[1])
    return lines


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from neurocascade.benchmark.v2.p35_selection import BASELINES, FAMILIES, SELECTED, run_p35_3
    from neurocascade.benchmark.v2.sider_labels import p35_heldout

    params_path = ROOT / PARAMS
    run = run_p35_3(frozen_params=None)
    params = run["pool_params"]
    assert run["selection"]["selected"] == SELECTED, run["selection"]
    assert not any(p35_heldout(i) for i in params["pool"])
    params_path.parent.mkdir(exist_ok=True)
    params_path.write_text(json.dumps(params, indent=1) + "\n")
    frozen = json.loads(params_path.read_text())
    run = run_p35_3(frozen_params=frozen)
    per = run["liabilities"]

    table = {}
    for e, row in per.items():
        t = {"common_n": row["common_n"], "n_positive": row["n_positive"],
             "n_negative": row["n_negative"], "go_eligible_on_train": row["go_eligible"],
             "baselines": {b: r3(m["auroc"]) for b, m in row["baselines"].items()},
             "best_baseline": row["best_baseline"]}
        if "frozen_candidate" in row:
            cand = row["frozen_candidate"]["auroc"]
            assert abs(cand - row["families"][SELECTED]["auroc"]) < 1e-12, e
            t["candidate"] = r3(cand)
            t["candidate_auprc"] = r3(row["frozen_candidate"]["auprc"])
            t["candidate_beats"] = sorted(b for b in row["baselines"]
                                          if cand > row["baselines"][b]["auroc"])
            t["p34_3_secondary"] = r3(row["p34_3_secondary"]["auroc"])
            t["all_families"] = {f: r3(m["auroc"]) for f, m in row["families"].items()}
        else:
            t["candidate"] = None
            t["candidate_status"] = row["candidate_status"]
        table[e] = t

    scored = sorted({i for row in per.values() if "frozen_candidate" in row for i in row["common_set"]})
    beats_one = [e for e, t in table.items() if t.get("candidate_beats")]
    out = {
        "task": "P35.3",
        "prereg": "docs/prereg-p35-go.md",
        "family": SELECTED,
        "description": frozen["description"],
        "code": {"path": CODE, "sha256": sha256(CODE)},
        "params": {"path": PARAMS, "sha256": sha256(PARAMS), "k": frozen["k"],
                   "n_pool": frozen["n_pool"],
                   "fitted_parameters": "the kNN pool: train INNs, fingerprints, train labels"},
        "depends": [{"path": p, "sha256": sha256(p)} for p in DEPENDS],
        "heldout_labels_existed_at_freeze": any((ROOT / V2 / "sider").glob("*heldout*")),
        "baselines": list(BASELINES),
        "families_evaluated": {f: list(v) for f, v in FAMILIES.items()},
        "selection_rule": ("most GO-eligible train liabilities (>= 8 pos and >= 8 neg) where the family "
                           "beats the best baseline; ties by mean margin"),
        "selection": run["selection"],
        "train_lodo_auroc": table,
        "candidate_beats_at_least_one_baseline": beats_one,
        "go_verdict_judge_on_train": go_verdict_train_lines(per),
        "secondary": {"task": "P34.3", "path": "results/P34.3.json",
                      "code_sha256": json.loads((ROOT / "results/P34.3.json").read_text())["code"]["sha256"],
                      "note": "pre-committed secondary for P35.4; scored here on the same train common sets"},
        "n_scored": len(scored),
        "n_scored_means": ("distinct P35-train universe drugs in at least one liability's train-LODO "
                           "common set scored by the frozen candidate"),
        "heldout": {"status": "host-pending",
                    "note": "P35.4 builds the P35 held-out labels host-side and scores this frozen "
                            "candidate once under NC_HOST_EVAL; no held-out label was read here"},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"code": out["code"]["sha256"][:12], "params": out["params"]["sha256"][:12],
                      "n_scored": out["n_scored"]}))
    for line in out["go_verdict_judge_on_train"]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
