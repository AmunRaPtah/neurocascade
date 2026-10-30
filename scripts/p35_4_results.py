#!/usr/bin/env python3
"""Write results/P35.4.json from the session tree (no held-out label read).

    PYTHONPATH=src python scripts/p35_4_results.py

Records (1) the session run of the P35.4 host_eval chain, verbatim: the dump
without NC_HOST_EVAL (empty liabilities) and pipeline/go_verdict.py on it; (2)
the frozen P35.3 candidate re-scored on P35-train leave-one-drug-out, the
largest scoring this tree supports, checked against results/P35.3.json; (3) a
label-free upper bound on each liability's held-out common set; (4) the
host-pending held-out split. The binding verdict is the host's.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "P35.4.json"

from neurocascade.benchmark.v2 import (  # noqa: E402
    load_v2_binding, load_v2_drugs, load_v2_human_cmax, v2_heldout)
from neurocascade.benchmark.v2 import go_dump_p35 as gd  # noqa: E402
from neurocascade.benchmark.v2 import sider_labels as sl  # noqa: E402
from neurocascade.benchmark.v2.candidate_benchmark import _cu  # noqa: E402
from neurocascade.benchmark.v2.p35_candidate import load_params  # noqa: E402
from neurocascade.benchmark.v2.p35_selection import run_p35_3  # noqa: E402


def session_chain() -> str:
    """The host_eval command without the flag; stdout verbatim."""
    env = {k: v for k, v in os.environ.items() if k != "NC_HOST_EVAL"}
    env["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "p35_4_go.json"
        a = subprocess.run([sys.executable, "-m", "neurocascade.benchmark.v2.go_dump_p35", str(out)],
                           env=env, cwd=ROOT, capture_output=True, text=True, check=True)
        b = subprocess.run([sys.executable, "pipeline/go_verdict.py", str(out)], env=env, cwd=ROOT,
                           capture_output=True, text=True, check=True)
    return (a.stdout.replace(str(out), "/tmp/p35_4_go.json") + b.stdout).strip()


def power_upper_bound() -> dict:
    """Per liability, P35 held-out drugs with a Ki at a pre-specified receptor
    and either a Cmax,u here or a Cmax row hidden from this tree (v2 held-out).
    Needs no SIDER record or label, so it bounds the held-out common set."""
    binding, cmax = load_v2_binding(), load_v2_human_cmax()
    held = [d["inn"] for d in load_v2_drugs() if sl.p35_heldout(d["inn"])]
    out = {}
    for effect, rc in load_params()["liability_receptors"].items():
        ki = [i for i in held if any((binding.get(i, {}).get("targets", {}).get(g) or {}).get("ki_nM")
                                     for g in rc)]
        have = sorted(i for i in ki if _cu(cmax, i) is not None)
        hidden = sorted(i for i in ki if i not in cmax and v2_heldout(i))
        out[effect] = {"ki_at_receptor": len(ki), "cu_in_session": have,
                       "cmax_hidden_v2_heldout": hidden, "upper_bound": len(have) + len(hidden),
                       "needed_for_prereg_floor": 2 * gd.PREREG_MIN_CLASS}
    return out


def train_lodo() -> tuple[dict, int]:
    rec = json.loads((ROOT / "results" / "P35.3.json").read_text())["train_lodo_auroc"]
    run = run_p35_3(frozen_params=load_params(), liabilities=tuple(load_params()["liability_receptors"]))
    out, inns = {}, set()
    for e, row in run["liabilities"].items():
        cand = round(row["frozen_candidate"]["auroc"], 3)
        assert cand == rec[e]["candidate"], (e, cand, rec[e]["candidate"])
        inns |= set(row["common_set"])
        out[e] = {"common_n": row["common_n"], "n_positive": row["n_positive"],
                  "candidate": cand, "p34_3_secondary": round(row["p34_3_secondary"]["auroc"], 3),
                  "baselines": {b: round(m["auroc"], 3) for b, m in row["baselines"].items()},
                  "best_baseline": row["best_baseline"]}
    return out, len(inns)


def main() -> None:
    lodo, n = train_lodo()
    held = gd.heldout_inns()
    res = {
        "task": "P35.4",
        "prereg": "docs/prereg-p35-go.md",
        "host_eval": ("NC_HOST_EVAL=1 PYTHONPATH=src python -m neurocascade.benchmark.v2.go_dump_p35 "
                      "/tmp/p35_4_go.json && python pipeline/go_verdict.py /tmp/p35_4_go.json"),
        "frozen_sha256": gd.check_freeze(),
        "n_scored": n,
        "n_scored_means": ("distinct P35-train universe drugs scored by the frozen P35.3 candidate in "
                           "train leave-one-drug-out (re-run here; equals results/P35.3.json). The "
                           "held-out score is the host's and binds."),
        "heldout": {"status": "host-pending",
                    "verdict_record": "HOME/go/P35.4.json (the conductor's first NC_GO_VERDICT binds)",
                    "n_heldout_drugs": len(held), "searched_and_empty_in_session": held},
        "session_chain_output_verbatim": session_chain(),
        "heldout_power_upper_bound_session": power_upper_bound(),
        "train_lodo_frozen_candidate": lodo,
        "heldout_rows_file": str(gd.HELDOUT_ROWS.relative_to(ROOT)),
        "prereg_floor_enforced_in_dump": (
            f">= {gd.PREREG_MIN_CLASS} positives and >= {gd.PREREG_MIN_CLASS} negatives (prereg 'Rule'); "
            "go_verdict.MIN_CLASS is 5, so a liability below 8/8 is not passed to go_verdict"),
    }
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("n_scored", "session_chain_output_verbatim")}, indent=1))


if __name__ == "__main__":
    main()
