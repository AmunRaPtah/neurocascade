"""P30.2: baseline comparisons, metrics and explicit held-out absence."""

import pytest

from neurocascade.benchmark import load_split
from neurocascade.benchmark.effect_scoring import (
    _auprc,
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
        assert result["n_scored"] + len(missing_input) == result["n_heldout"]
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
