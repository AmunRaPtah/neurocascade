"""P8.3: the standing failure analysis names causes and repairs."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "heldout-failure-analysis.md"
REPORT = ROOT / "results" / "heldout" / "heldout_validation.txt"


def test_failure_analysis_is_committed_and_has_each_required_layer():
    text = DOC.read_text()
    for phrase in (
        "Dose-to-concentration PK",
        "Plasma-to-brain partition",
        "Occupancy tier",
        "Binding provenance",
        "Transporter/projection layer",
        "Endogenous transmitter tone",
        "Behavioural evidence",
        "Behaviour readout shape",
        "Circuit gains and endpoint magnitudes",
        "Whole-brain numerical reachability",
    ):
        assert phrase in text, phrase


def test_each_failure_attribution_has_a_concrete_repair():
    text = DOC.read_text()
    assert text.count("| Upstream") >= 1
    assert text.count("| P6.2 coverage failure") == 1
    assert "required data" in text
    assert "Repair order" in text
    assert "Never substitute fentanyl" in text


def test_analysis_is_tied_to_the_committed_validation_snapshot():
    report = REPORT.read_text()
    assert "Predictions for 7 compounds" in report
    assert "Compounds with a complete chain: 0 of 7" in report
    assert "5 occupancy observations were collected and NONE could be scored" in report
    assert "Direction agreement: 4 of 15" in report
    assert "at 1.00 sd: REFUTES" in report
    assert "at 1.96 sd: not resolved" in report
