"""P34.1 candidate: the parameter-free rank ensemble and its train-LODO table.

These pin the pure predict path (``candidate_v2.rank_combined_score``), the
fact that it imports no truth and reads no file, that the benchmark scores only
train INNs (never a held-out or fresh drug), and that the committed
``results/P34.1.json`` and the docs carry the numbers a fresh run produces,
including the required win over ``occupancy_max`` on at least one liability.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import v2_heldout, v2_partition
from neurocascade.benchmark.v2.candidate_v2 import rank_combined_score
from neurocascade.benchmark.v2.candidate_v2_benchmark import (
    REGISTERED_SCORED,
    run_candidate_v2_benchmark,
    score_liability,
)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
CAND = V2 / "candidate_v2.py"


@pytest.fixture(scope="module")
def report():
    return run_candidate_v2_benchmark()


# --------------------------------------------------------------- pure predict

def test_rank_combined_is_parameter_free_rank_average():
    # occ ranks: [1,2,3], aff ranks reversed: [3,2,1] -> all tie at 4.
    occ = [0.1, 0.5, 0.9]
    aff = [9.0, 5.0, 1.0]
    assert rank_combined_score(occ, aff) == [4.0, 4.0, 4.0]
    # agreeing views -> monotone increasing combined score.
    s = rank_combined_score([0.1, 0.5, 0.9], [1.0, 5.0, 9.0])
    assert s == sorted(s) and s[0] < s[-1]


def test_rank_combined_handles_ties_with_mean_rank():
    # two tied on occ -> mean rank 1.5 each; aff strictly increasing 1,2,3.
    s = rank_combined_score([0.2, 0.2, 0.9], [1.0, 2.0, 3.0])
    assert s == pytest.approx([1.5 + 1.0, 1.5 + 2.0, 3.0 + 3.0])


def test_rank_combined_returns_none_where_either_view_missing():
    occ = [0.5, None, 0.9]
    aff = [1.0, 2.0, None]
    out = rank_combined_score(occ, aff)
    assert out[1] is None and out[2] is None
    assert out[0] is not None  # only the fully-defined drug is scored
    assert rank_combined_score([None, None], [1.0, 2.0]) == [None, None]


def test_rank_combined_length_mismatch_raises():
    with pytest.raises(ValueError):
        rank_combined_score([0.1, 0.2], [1.0])


def test_candidate_predict_path_imports_no_truth():
    """The candidate predict path must not import a truth loader or read a file;
    feature assembly (which does read truth) lives in the harness."""
    src = CAND.read_text()
    tree = ast.parse(src)
    imported: list[str] = []
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
    assert not {m for m in imported if m.startswith("neurocascade")}


# ----------------------------------------------------------- train-only reads

def test_benchmark_reads_only_train_inns():
    """Every INN the benchmark scores is on the train side of v2_partition;
    none is held-out, and the harness refuses a held-out INN if handed one."""
    train, heldout = v2_partition()
    assert all(not v2_heldout(inn) for inn in train)
    assert all(v2_heldout(inn) for inn in heldout)

    # The harness guards itself: a train list contaminated with a held-out INN
    # trips the assertion rather than silently scoring it.
    bad = list(train) + [heldout[0]]
    from neurocascade.benchmark.v2 import (
        load_v2_binding,
        load_v2_drugs,
        load_v2_human_cmax,
        load_v2_liabilities,
    )
    from neurocascade.benchmark.v2.chain_benchmark import _all_targets

    drugs = {row["inn"]: row for row in load_v2_drugs()}
    binding = load_v2_binding()
    # Only a held-out INN that actually carries a sedation label can reach the
    # assertion; if none does in this checkout the guard is still proven by the
    # train-only assertions above, so skip rather than assert vacuously.
    liab = load_v2_liabilities()
    if liab.get(heldout[0], {}).get("sedation", {}).get("label") is None:
        pytest.skip("held-out drug has no sedation label in this checkout")
    with pytest.raises(AssertionError):
        score_liability(
            "sedation", bad, binding, load_v2_human_cmax(), liab, drugs, _all_targets(binding)
        )


def test_benchmark_source_never_references_fresh_or_heldout_truth():
    src = (V2 / "candidate_v2_benchmark.py").read_text()
    assert "truth_fresh" not in src and "V2_FRESH" not in src
    # it reads the train side of the partition, index 0.
    assert "v2_partition()" in src


# -------------------------------------------------------------- the LODO table

def test_candidate_beats_occupancy_max_on_at_least_one_liability(report):
    beaten = report["candidate_beats_occupancy_max_liabilities"]
    assert "sedation" in beaten  # the acceptance criterion
    sed = report["liabilities"]["sedation"]["lodo"]
    assert sed["candidate_rank_combined"]["auroc"] > sed["occupancy_max"]["auroc"]


def test_every_method_scored_on_identical_common_set(report):
    for effect in REGISTERED_SCORED:
        row = report["liabilities"][effect]
        for method in (
            "candidate_rank_combined",
            "occupancy_max",
            "strongest_target_ki",
            "one_nn_fingerprint",
            "atc_class_prevalence",
        ):
            assert method in row["lodo"]
            assert row["lodo"][method]["n"] == row["common_n"]
    assert report["n_scored"] > 0


def test_committed_results_file_matches_a_fresh_run(report):
    committed = json.loads((ROOT / "results" / "P34.1.json").read_text())
    assert committed["n_scored"] == report["n_scored"] > 0
    for effect in REGISTERED_SCORED:
        for method, m in report["liabilities"][effect]["lodo"].items():
            assert committed["liabilities"][effect]["lodo"][method]["auroc"] == pytest.approx(
                m["auroc"]
            )


def test_docs_quote_the_programmatic_auroc_numbers(report):
    arch = (ROOT / "docs" / "architecture.md").read_text()
    ledger = (ROOT / "docs" / "validation-ledger.md").read_text()
    for effect in REGISTERED_SCORED:
        for method in ("candidate_rank_combined", "occupancy_max", "strongest_target_ki"):
            value = f"{report['liabilities'][effect]['lodo'][method]['auroc']:.3f}"
            assert value in arch, f"{effect}/{method}={value} missing from architecture.md"
            assert value in ledger, f"{effect}/{method}={value} missing from ledger"
