"""P34.4: the fresh held-out truth set.

Checks the fresh drugs are new (absent from the v2 universe), every pool member
is listed with exclusions for missing input only, the labels were built with
the P34.2-frozen rules, the build scripts import no candidate, and the power
floor verdict in results/P34.4.json follows from the labels.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
FRESH = V2 / "truth_fresh"
BUILD = ("scripts/p34_4_pool.py", "scripts/p34_4_build.py")

pytestmark = pytest.mark.skipif(not (FRESH / "liabilities.json").exists(),
                                reason="fresh truth is stripped from session trees")


def load(name: str) -> dict:
    return json.loads((FRESH / name).read_text())


def test_fresh_drugs_are_absent_from_the_v2_universe():
    v2 = json.loads((V2 / "drugs.json").read_text())["drugs"]
    inns, ids = {d["inn"] for d in v2}, {d["chembl_id"] for d in v2}
    fresh = load("drugs.json")["drugs"]
    assert fresh
    for d in fresh:
        assert d["inn"] not in inns and d["chembl_id"] not in ids


def test_every_pool_member_is_listed_and_exclusions_are_missing_input_only():
    pool = load("pool.json")
    included = [m["inn"] for m in pool["members"] if not m["excluded"]]
    assert len(pool["members"]) == pool["n_pool"]
    assert included == [d["inn"] for d in load("drugs.json")["drugs"]]
    for m in pool["members"]:
        assert m["selected_by"], m
        assert m["excluded"] is None or m["excluded"].startswith("missing input:"), m


def test_labels_built_with_the_p34_2_frozen_rules():
    pinned = {f["path"]: f["sha256"] for f in
              json.loads((ROOT / "results" / "P34.2.json").read_text())["frozen_files"]}
    liab = load("liabilities.json")
    assert liab["rules_sha256"] == pinned["src/neurocascade/benchmark/v2/liability_rules.json"]
    assert [d["inn"] for d in liab["drugs"]] == [d["inn"] for d in load("drugs.json")["drugs"]]


@pytest.mark.parametrize("rel", BUILD)
def test_build_scripts_import_no_candidate(rel):
    tree = ast.parse((ROOT / rel).read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
    for n in names:
        assert "candidate" not in n and "go_dump" not in n, f"{rel} imports {n}"


def test_power_floor_verdict_follows_from_the_labels():
    sys.path.insert(0, str(ROOT / "scripts"))
    import p34_4_build
    res = json.loads((ROOT / "results" / "P34.4.json").read_text())
    c = p34_4_build.counts(load("liabilities.json"))
    assert res["labelled_counts_upper_bound_on_common_set"] == c
    reach = [L for L, v in c.items()
             if v["labelled_pos"] >= p34_4_build.FLOOR and v["labelled_neg"] >= p34_4_build.FLOOR]
    assert res["liabilities_reaching_floor"] == reach
    assert res["result"] == ("COMPLETE" if reach else "EMPTY")
    assert res["power_floor"] == {"positives": 8, "negatives": 8}


def test_fresh_files_are_seen_by_the_guard():
    sys.path.insert(0, str(ROOT / "pipeline"))
    import guards
    files = guards.v2_fresh_files(ROOT)
    for name in ("pool.json", "drugs.json", "liabilities.json"):
        assert guards.V2_FRESH + name in files
