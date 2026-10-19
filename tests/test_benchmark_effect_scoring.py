"""P30.2: baseline comparisons, metrics and explicit held-out absence."""

import json
from itertools import permutations
from pathlib import Path

import pytest

from neurocascade.benchmark import load_split
from neurocascade.benchmark.effect_scoring import (
    _auprc,
    _brier,
    metric_summary,
    nearest_neighbor,
    score_p30_2,
    strongest_target_ki,
)


def test_auprc_and_prevalence_are_distinct_and_checkable():
    assert _auprc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    result = metric_summary([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], replicates=20)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["prevalence_baseline"] == pytest.approx(0.5)
    assert result["auroc_ci"] == pytest.approx([1.0, 1.0])
    assert result["auprc_ci"] == pytest.approx([1.0, 1.0])


def test_strongest_target_baseline_uses_the_smallest_reported_ki():
    score = strongest_target_ki("aripiprazole")
    assert score is not None
    assert score.neighbor == "HTR2B"
    assert score.score == pytest.approx(0.1718358944505682)


def test_nearest_neighbor_uses_shared_targets_and_is_deterministic():
    score = nearest_neighbor("aripiprazole", ["olanzapine", "haloperidol"])
    assert score is not None
    assert score.neighbor == "haloperidol"
    assert score.shared_targets == 30
    assert score.distance == pytest.approx(1.1581860935570485)


def test_p30_2_never_reads_missing_heldout_truth_as_a_negative():
    """One function, both environments: the conductor checkout strips held-out
    truth, a host holds it. Either way no missing label is scored as a negative
    and no drug is both scored and missing.
    """
    result = score_p30_2(bootstrap_replicates=3)
    assert result["task"] == "P30.2"
    heldout = result["heldout"]
    missing_truth, missing_input = heldout["missing_truth"], heldout["missing_input"]
    assert not set(missing_truth) & set(missing_input)
    assert set(missing_truth) | set(missing_input) <= set(load_split()["heldout"])
    if heldout["status"] == "searched-and-empty":
        assert result["status"] == "searched-and-empty"
        assert result["n_scored"] == 0
        assert len(missing_truth) == result["n_heldout"]
        assert heldout["effects"] == {}
    else:
        assert heldout["status"] == "scored"
        assert result["status"] == "scored"
        assert result["n_scored"] > 0
        assert missing_truth == []
        unlabelled = heldout["unlabelled"]
        assert result["n_scored"] + len(missing_input) + len(unlabelled) == result["n_heldout"]
        assert heldout["effects"]
        for effect in heldout["effects"].values():
            assert set(effect["methods"]) == {
                "occupancy_model", "strongest_target_ki", "nearest_neighbor",
            }
            for method in effect["methods"].values():
                for key in ("auroc", "auprc", "prevalence_baseline", "auroc_ci", "auprc_ci"):
                    assert key in method
    assert set(result["lodo_all_available_drugs"]["effects"]) == {
        "eps", "sedation", "weight_gain", "anticholinergic", "orthostasis",
        "sexual_dysfunction", "seizure",
    }
    for effect in result["lodo_all_available_drugs"]["effects"].values():
        assert set(effect["methods"]) == {
            "occupancy_model", "strongest_target_ki", "nearest_neighbor",
        }
        for method in effect["methods"].values():
            assert "auroc" in method
            assert "auprc" in method
            assert "prevalence_baseline" in method


def _brute_force_ap(labels, scores):
    """Independent definition: average AP over every ordering of tied blocks."""
    positives = sum(labels)
    orderings = []
    for order in permutations(range(len(labels))):
        ranked = sorted(order, key=lambda i: -scores[i])
        found = 0
        total = 0.0
        for rank, index in enumerate(ranked, start=1):
            if labels[index] == 1:
                found += 1
                total += found / rank
        orderings.append(total / positives)
    return sum(orderings) / len(orderings)


def test_auprc_equals_the_brute_force_average_over_tied_orderings():
    """Tied scores must be averaged over every ordering of the tied block.

    A simple permutation check cannot catch a formula that is invariant but
    wrong: the first "fix" was order-invariant yet disagreed with brute force
    on `_auprc([1,0],[0,0])` (0.5 where the average over the two orderings is
    0.75), which the acceptance review of PR #33 found. This test compares
    against the definition, not against the implementation's own output.
    """
    cases = [
        ([1, 0], [0.0, 0.0]),
        ([1, 0, 1], [1.0, 1.0, 0.5]),
        ([0, 1, 0, 1, 1], [0.5, 0.5, 1.0, 0.0, 0.5]),
        ([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]),
        ([0, 0, 1], [1.0, 0.0, 1.0]),
    ]
    for labels, scores in cases:
        assert _auprc(labels, scores) == pytest.approx(_brute_force_ap(labels, scores), abs=1e-9)


def test_auprc_is_invariant_under_input_order():
    labels = [1, 0, 0, 1, 0]
    scores = [1.0, 1.0, 0.0, 0.5, 1.0]
    base = _auprc(labels, scores)
    for order in permutations(range(len(labels))):
        assert _auprc([labels[i] for i in order], [scores[i] for i in order]) == pytest.approx(base)


def test_metric_summary_reports_calibration_with_an_interval():
    result = metric_summary([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], replicates=20)
    assert result["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)
    assert result["brier_ci"] is not None and len(result["brier_ci"]) == 2
    assert _brier([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_committed_results_file_is_the_function_schema():
    """results/P30.2.json is generated by scripts/score_p30_2.py, not edited."""
    path = Path(__file__).resolve().parents[1] / "results" / "P30.2.json"
    data = json.loads(path.read_text())
    for key in ("task", "status", "n_scored", "n_heldout", "heldout",
                "lodo_all_available_drugs", "bootstrap", "fit_split"):
        assert key in data
    heldout = data["heldout"]
    for key in ("status", "n_scored", "missing_truth", "missing_input",
                "effects", "failure_analysis"):
        assert key in heldout
    for effect in heldout["effects"].values():
        for metrics in effect["methods"].values():
            for key in ("auroc", "auprc", "brier", "brier_note",
                        "prevalence_baseline", "auroc_ci", "auprc_ci", "brier_ci"):
                assert key in metrics


def _fmt(value):
    return "NA" if value is None else f"{value:.2f}"


def _ci(values):
    return "" if not values else f" [{_fmt(values[0])}, {_fmt(values[1])}]"


def test_docs_tables_match_the_generated_results():
    """The published tables are transcribed from results/P30.2.json by
    scripts/score_p30_2.py's formatter; this test fails if they drift.

    The acceptance review of PR #34 found the docs LODO table still carrying
    AUPRC values from before the tie fix, so the check is mechanical now.
    """
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "results" / "P30.2.json").read_text())
    docs = (root / "docs" / "architecture.md").read_text()
    for effect, entry in results["heldout"]["effects"].items():
        methods = entry["methods"]
        model = methods["occupancy_model"]
        strongest = methods["strongest_target_ki"]
        neighbour = methods["nearest_neighbor"]
        label = "EPS" if effect == "eps" else effect.replace("_", " ")
        row = (f"| {label} "
               f"| {model['n_scored']} | {entry['n_excluded_for_baseline']} "
               f"| {model['n_positive']} / {model['n_negative']} "
               f"| {_fmt(model['auroc'])}{_ci(model['auroc_ci'])} "
               f"| {_fmt(strongest['auroc'])} | {_fmt(neighbour['auroc'])} "
               f"| {_fmt(model['auprc'])}{_ci(model['auprc_ci'])} "
               f"| {_fmt(model['prevalence_baseline'])} | {_fmt(neighbour['auprc'])} "
               f"| {_fmt(model['brier'])}{_ci(model['brier_ci'])} |")
        assert row in docs, f"held-out row for {effect} drifted from results/P30.2.json"
    for effect, entry in results["lodo_all_available_drugs"]["effects"].items():
        methods = entry["methods"]
        model = methods["occupancy_model"]
        strongest = methods["strongest_target_ki"]
        neighbour = methods["nearest_neighbor"]
        label = "EPS" if effect == "eps" else effect.replace("_", " ")
        row = (f"| {label} | {model['n_scored']} "
               f"| {_fmt(model['auroc'])} / {_fmt(model['auprc'])} / {_fmt(model['brier'])} "
               f"| {_fmt(strongest['auroc'])} / {_fmt(strongest['auprc'])} "
               f"| {_fmt(neighbour['auroc'])} / {_fmt(neighbour['auprc'])} |")
        assert row in docs, f"LODO row for {effect} drifted from results/P30.2.json"


def test_ledger_v19_quotes_the_generated_metrics():
    """The ledger row restates the metrics (its acceptance requires it), so it
    is checked against results/P30.2.json rather than trusted."""
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "results" / "P30.2.json").read_text())
    ledger = (root / "docs" / "validation-ledger.md").read_text()
    row = next(line for line in ledger.splitlines() if line.startswith("| V19 |"))
    for effect, entry in results["heldout"]["effects"].items():
        if effect == "seizure":
            continue
        metrics = entry["methods"]["occupancy_model"]
        auroc = metrics["auroc"]
        auroc_ci = metrics["auroc_ci"]
        auprc = metrics["auprc"]
        auprc_ci = metrics["auprc_ci"]
        brier = metrics["brier"]
        brier_ci = metrics["brier_ci"]
        assert (f"{auroc:.2f} [{auroc_ci[0]:.2f}, {auroc_ci[1]:.2f}]" in row), effect
        assert (f"{auprc:.2f} [{auprc_ci[0]:.2f}, {auprc_ci[1]:.2f}]" in row), effect
        assert (f"{brier:.2f} [{brier_ci[0]:.2f}, {brier_ci[1]:.2f}]" in row), effect
        neighbour = entry["methods"]["nearest_neighbor"]
        assert f"{neighbour['auprc']:.2f}" in row, effect
