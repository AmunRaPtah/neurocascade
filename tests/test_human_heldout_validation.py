import pytest

from neurocascade.human.heldout_validation import HeldOutHumanDrugResult, summarize_heldout


def test_scored_result_and_summary():
    result = HeldOutHumanDrugResult("diazepam", "subjective_sedation", "scored", "trial", 0.8, 0.6)
    summary = summarize_heldout([result])
    assert result.error == pytest.approx(0.2)
    assert summary["scored"] == 1
    assert summary["mean_absolute_error"] == pytest.approx(0.2)


def test_unavailable_result_is_explicit():
    result = HeldOutHumanDrugResult("fentanyl", "motor_activity", "unavailable", note="no held-out human endpoint")
    assert result.error is None
    assert summarize_heldout([result])["unavailable"] == 1


def test_invalid_scored_result_is_rejected():
    with pytest.raises(ValueError):
        HeldOutHumanDrugResult("x", "y", "scored")
    with pytest.raises(ValueError):
        summarize_heldout([])
