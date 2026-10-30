"""P35.3: the P35 candidate, selected and frozen on SIDER train LODO.

Pins the sha256 of the frozen predict path and params (results/P35.3.json),
checks the candidate path imports no held-out or fresh truth and reads only
P35-train INNs, re-runs the train-LODO selection and compares it with the
committed table, and unit-tests the kNN view on synthetic rows.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from neurocascade.benchmark.v2.p35_candidate import (
    fingerprint,
    knn_fraction,
    load_params,
    predict_cohort,
)
from neurocascade.benchmark.v2.p35_selection import (
    BASELINES,
    FAMILIES,
    SELECTED,
    build_params,
    load_inputs,
    run_p35_3,
)
from neurocascade.benchmark.v2.sider_labels import load_train_labels, p35_heldout

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "P35.3.json"
V2 = ROOT / "src/neurocascade/benchmark/v2"

# Frozen 2026-10-29 by scripts/p35_3_freeze.py, before any P35 held-out label
# existed. Changing one breaks the P35 pre-registration: do not update a sha to pass.
PINNED = {
    "src/neurocascade/benchmark/v2/p35_candidate.py": "62d4bb46d25b",
    "src/neurocascade/benchmark/v2/frozen/p35_candidate.json": "430f8e9af81b",
}
# Modules a candidate path may import from the package: none reads truth.
ALLOWED_PACKAGE_IMPORTS = {".candidate"}
FORBIDDEN_TOKENS = ("truth_fresh", "truth/", "heldout", "load_v2_liabilities",
                    "load_v2_human_cmax", "labels_heldout")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def rerun():
    try:  # many tiny BLAS calls; threads only add contention
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return run_p35_3(frozen_params=load_params())
    except ImportError:
        return run_p35_3(frozen_params=load_params())


# ------------------------------------------------------------------ freeze

@pytest.mark.parametrize("rel", sorted(PINNED))
def test_frozen_file_sha_is_pinned(rel, result):
    assert sha(rel).startswith(PINNED[rel]), f"{rel} changed after the P35.3 freeze"
    recorded = {result["code"]["path"]: result["code"]["sha256"],
                result["params"]["path"]: result["params"]["sha256"]}
    assert recorded[rel] == sha(rel)


def test_dependency_shas_match_the_freeze(result):
    for dep in result["depends"]:
        assert sha(dep["path"]) == dep["sha256"], f"{dep['path']} changed after the P35.3 freeze"


def test_no_heldout_label_existed_at_freeze(result):
    assert result["heldout_labels_existed_at_freeze"] is False
    assert not list((V2 / "sider").glob("*heldout*"))


def test_result_is_generated_by_the_script(result):
    sys.path.insert(0, str(ROOT / "scripts"))
    import p35_3_freeze
    assert result["code"]["path"] == p35_3_freeze.CODE
    assert result["params"]["path"] == p35_3_freeze.PARAMS
    assert [d["path"] for d in result["depends"]] == p35_3_freeze.DEPENDS


def test_params_file_is_rebuilt_from_train_inputs():
    assert load_params() == json.loads(json.dumps(build_params(load_inputs())))


# ------------------------------------------------------------------ train only, no truth

def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_candidate_predict_path_imports_no_truth():
    src = (V2 / "p35_candidate.py").read_text()
    package = {m for m in _imports(V2 / "p35_candidate.py") if m.startswith(".")}
    assert package <= ALLOWED_PACKAGE_IMPORTS
    assert not any(m.startswith(".") for m in _imports(V2 / "candidate.py"))
    for token in FORBIDDEN_TOKENS:
        assert token not in src


def test_selection_harness_reads_only_train_labels():
    src = (V2 / "p35_selection.py").read_text()
    assert "truth_fresh" not in src and "labels_heldout" not in src
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module == "sider_labels":
            imported.update(a.name for a in node.names)
    assert imported == {"LIABILITY_PTS", "load_train_labels", "load_universe", "p35_heldout"}


def test_frozen_pool_holds_only_train_inns():
    pool = load_params()["pool"]
    labels = load_train_labels()
    assert len(pool) == load_params()["n_pool"] == 126
    for inn, row in pool.items():
        assert not p35_heldout(inn)
        assert row["labels"] == {e: labels[inn][e]["label"] for e in row["labels"]}


def test_scored_drugs_are_train_universe_members(rerun):
    for row in rerun["liabilities"].values():
        assert all(not p35_heldout(i) for i in row["common_set"])
        assert set(row["common_set"]) <= set(rerun["inputs"]["universe"])


# ------------------------------------------------------------------ selection

def test_candidate_is_not_a_baseline(result):
    assert SELECTED not in BASELINES and result["family"] == SELECTED
    assert set(FAMILIES[SELECTED]) - set(BASELINES) == {"knn10"}


def test_rerun_reproduces_selection_and_table(result, rerun):
    assert rerun["selection"]["selected"] == SELECTED == result["selection"]["selected"]
    for e, t in result["train_lodo_auroc"].items():
        row = rerun["liabilities"][e]
        assert row["common_n"] == t["common_n"] and row["n_positive"] == t["n_positive"]
        for b, a in t["baselines"].items():
            assert round(row["baselines"][b]["auroc"], 3) == a
        if t["candidate"] is not None:
            assert round(row["frozen_candidate"]["auroc"], 3) == t["candidate"]
            assert row["frozen_candidate"]["auroc"] == pytest.approx(
                row["families"][SELECTED]["auroc"], abs=1e-12)
            assert round(row["p34_3_secondary"]["auroc"], 3) == t["p34_3_secondary"]


def test_pinned_train_lodo_values(result):
    t = result["train_lodo_auroc"]
    assert {e: (t[e]["common_n"], t[e]["n_positive"]) for e in t} == {
        "sedation": (21, 8), "anticholinergic": (21, 12), "eps": (25, 11),
        "orthostatic_hypotension": (25, 9), "seizure": (49, 6)}
    assert {e: t[e]["candidate"] for e in t} == {
        "sedation": 0.76, "anticholinergic": 0.542, "eps": 0.74,
        "orthostatic_hypotension": 0.642, "seizure": None}
    assert t["sedation"]["baselines"]["occupancy_max"] == 0.673
    assert t["eps"]["baselines"]["occupancy_max"] == 0.675
    assert result["n_scored"] == 41


def test_acceptance_beats_a_baseline(result):
    beats = result["candidate_beats_at_least_one_baseline"]
    assert beats
    # On sedation and EPS it beats all four baselines on train.
    for e in ("sedation", "eps"):
        assert sorted(result["train_lodo_auroc"][e]["candidate_beats"]) == sorted(BASELINES)


def test_secondary_is_the_p34_3_freeze(result):
    p343 = json.loads((ROOT / "results" / "P34.3.json").read_text())
    assert result["secondary"]["code_sha256"] == p343["code"]["sha256"]


# ------------------------------------------------------------------ synthetic

PARAMS = {
    "k": 2, "panel": ["A", "B"], "fingerprint_floor": -5.0,
    "liability_receptors": {"sedation": ["A"]},
    "pool": {
        "p1": {"fingerprint": [-1.0, -5.0], "labels": {"sedation": 1}},
        "p2": {"fingerprint": [-2.0, -5.0], "labels": {"sedation": 0}},
        "p3": {"fingerprint": [-4.0, -5.0], "labels": {"sedation": 0}},
        "p4": {"fingerprint": [-1.0, -5.0], "labels": {"sedation": "excluded"}},
    },
}


def test_fingerprint_floor_and_refusal():
    assert fingerprint({"A": 10.0}, ["A", "B"], -5.0) == [-1.0, -5.0]
    assert fingerprint({"A": None}, ["A", "B"], -5.0) is None


def test_knn_excludes_self_and_unlabelled():
    assert knn_fraction("q", [-1.0, -5.0], "sedation", PARAMS) == 0.5  # p1, p2
    assert knn_fraction("p1", [-1.0, -5.0], "sedation", PARAMS) == 0.0  # p2, p3
    assert knn_fraction("q", None, "sedation", PARAMS) is None


def test_predict_cohort_ranks_and_refuses():
    scores = predict_cohort("sedation", ["a", "b", "c"], [10.0, 1.0, None],
                            [{"A": 10.0}, {"A": 1000.0}, {"A": 10.0}], PARAMS)
    assert scores[2] is None
    assert scores[0] > scores[1]
    with pytest.raises(KeyError):
        predict_cohort("seizure", ["a"], [1.0], [{"A": 1.0}], PARAMS)
