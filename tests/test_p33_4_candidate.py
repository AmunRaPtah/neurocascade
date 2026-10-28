"""P33.4 candidate: pure predict path and the train-LODO benchmark.

These pin the occupancy feature, the logistic estimator, and the fact that the
candidate's predict path (``neurocascade.benchmark.v2.candidate``) imports no
truth module and reads no file -- the property the GO reviewer checks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import candidate
from neurocascade.benchmark.v2.candidate import (
    LIABILITY_RECEPTORS,
    feature_vector,
    liability_predict,
    occupancy,
    occupancy_baseline,
)
from neurocascade.benchmark.v2.candidate_benchmark import run_candidate_benchmark

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "src" / "neurocascade" / "benchmark" / "v2" / "candidate.py"


# P33.4 was scored on the v2 inputs as of commit 70ea056; its NO-GO verdict is
# the first verdict and binds. PRs #51 (binding coverage) and #52 (plasma fu
# coverage, plus truth/human_cmax_corrections.json, which load_v2_human_cmax
# applies by default) later extended these inputs, so a re-run scores more drugs.
# The committed record is not regenerated; the reproduction tests only apply
# while the inputs are byte-identical to the ones P33.4 was scored on.
P33_4_INPUTS = {
    "inputs/binding.json": "63fdbc904c983102",
    "inputs/plasma_fu.json": "d5120f95495161ec",
}


def _inputs_as_scored() -> bool:
    import hashlib
    v2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
    if (v2 / "truth" / "human_cmax_corrections.json").exists():  # absent at 70ea056
        return False
    return all(hashlib.sha256((v2 / rel).read_bytes()).hexdigest().startswith(want)
               for rel, want in P33_4_INPUTS.items())


as_scored = pytest.mark.skipif(
    not _inputs_as_scored(),
    reason="v2 inputs (binding, plasma fu, Cmax corrections) extended after P33.4 by PRs #51/#52; "
           "the binding P33.4 record is not re-run")


@pytest.fixture(scope="module")
def report():
    # The train LODO benchmark is ~1 min; run it once and share it.
    return run_candidate_benchmark()


def test_occupancy_is_cu_over_cu_plus_ki():
    assert occupancy(10.0, 10.0) == pytest.approx(0.5)
    assert occupancy(90.0, 10.0) == pytest.approx(0.9)
    # No usable Ki means weak binding -> occupancy 0, never missing.
    assert occupancy(10.0, None) == 0.0
    # No exposure means undefined, not zero.
    assert occupancy(None, 10.0) is None
    assert occupancy(0.0, 10.0) == 0.0


def test_feature_vector_follows_receptor_order_and_missing_ki():
    recs = ("CHRM1", "CHRM2", "CHRM3")
    row = feature_vector(10.0, {"CHRM1": 10.0, "CHRM3": 90.0}, recs)
    assert row == pytest.approx([0.5, 0.0, 10.0 / 100.0])
    assert feature_vector(None, {"CHRM1": 10.0}, recs) is None


def test_liability_predict_is_monotone_for_a_single_feature():
    X = [[0.0], [0.1], [0.3], [0.7], [0.9], [1.0]]
    y = [0, 0, 0, 1, 1, 1]
    p = liability_predict(X, y, X)
    assert p == sorted(p)  # higher occupancy -> higher score
    assert p[0] < 0.5 < p[-1]


def test_liability_predict_degenerate_train_returns_constant_rate():
    # One class only -> fall back to the positive rate, a constant (AUROC 0.5),
    # never a crash.
    p = liability_predict([[0.1], [0.2], [0.3]], [1, 1, 1], [[0.5], [0.9]])
    assert p == [1.0, 1.0]
    p0 = liability_predict([], [], [[0.5]])
    assert p0 == [0.0]


def test_occupancy_baseline_is_max_occupancy_over_receptors():
    recs = ("CHRM1", "CHRM2")
    assert occupancy_baseline(10.0, {"CHRM1": 10.0, "CHRM2": 90.0}, recs) == pytest.approx(0.5)
    assert occupancy_baseline(None, {"CHRM1": 10.0}, recs) is None


def test_receptor_sets_match_the_prespecified_liability_rules():
    rules = json.loads(
        (ROOT / "src" / "neurocascade" / "benchmark" / "v2" / "liability_rules.json").read_text()
    )["liabilities"]
    # Sedation is H1, anticholinergic is M1-M5: the same receptor field the
    # pre-registered rules name (and the same sets the P33.3 baselines score).
    assert LIABILITY_RECEPTORS["sedation"] == ("HRH1",)
    assert LIABILITY_RECEPTORS["anticholinergic"] == ("CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5")
    assert rules["sedation"]["receptor"] == "H1"
    assert rules["anticholinergic"]["receptor"] == "M1-M5"


def test_candidate_predict_path_imports_no_truth():
    """The candidate module must not import a truth loader or read any file;
    feature assembly (which does read truth) lives in the harness."""
    src = CAND.read_text()
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    joined = " ".join(imported)
    for forbidden in ("load_v2_", "truth", "chain_benchmark", "candidate_benchmark"):
        assert forbidden not in joined, f"candidate imports {forbidden!r}"
    assert "open(" not in src and ".read_text" not in src and "json.load" not in src
    # Only numpy / scipy are pulled in.
    assert not {m for m in imported if m.startswith("neurocascade")}


def test_benchmark_scores_candidate_beside_every_baseline(report):
    assert report["n_scored"] > 0
    for effect in ("sedation", "anticholinergic"):
        lodo = report["liabilities"][effect]["lodo"]
        for method in (
            "candidate_occupancy_logistic",
            "occupancy_max",
            "strongest_target_ki",
            "one_nn_fingerprint",
            "atc_class_prevalence",
        ):
            assert method in lodo
            assert lodo[method]["n"] == report["liabilities"][effect]["common_n"]
    # Secondary PK (IVIVE) has no CLint source: recorded searched-and-empty.
    assert report["secondary_pk"]["status"] == "searched-and-empty"


@as_scored
def test_committed_results_file_matches_a_fresh_run(report):
    committed = json.loads((ROOT / "results" / "P33.4.json").read_text())
    fresh = report
    assert committed["n_scored"] == fresh["n_scored"] > 0
    for effect in ("sedation", "anticholinergic"):
        for method, m in fresh["liabilities"][effect]["lodo"].items():
            assert committed["liabilities"][effect]["lodo"][method]["auroc"] == pytest.approx(
                m["auroc"]
            )


@as_scored
def test_docs_quote_the_programmatic_auroc_numbers(report):
    """The architecture table and the ledger row must carry the same AUROCs the
    benchmark produces (acceptance: numbers generated programmatically)."""
    arch = (ROOT / "docs" / "architecture.md").read_text()
    ledger = (ROOT / "docs" / "validation-ledger.md").read_text()
    for effect in ("sedation", "anticholinergic"):
        for method in ("candidate_occupancy_logistic", "occupancy_max", "strongest_target_ki"):
            value = f"{report['liabilities'][effect]['lodo'][method]['auroc']:.3f}"
            assert value in arch, f"{effect}/{method}={value} missing from architecture.md"
            assert value in ledger, f"{effect}/{method}={value} missing from ledger"
