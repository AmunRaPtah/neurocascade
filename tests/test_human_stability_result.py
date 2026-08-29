"""P13.5: human stability result is reproducible and honest."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "human_whole_brain_stability.txt"


def test_human_stability_result_records_all_three_stages():
    text = RESULT.read_text()
    assert "as built (baseline_input=0, weight_scale=1)" in text
    assert "baselines balanced, weights still unscaled" in text
    assert "P3.4 procedure applied" in text
    assert text.count("VERDICT: NOT ok") == 2
    assert text.count("VERDICT: ok") == 2


def test_human_stability_result_keeps_operating_gain_illustrative():
    text = RESULT.read_text()
    assert "Operating gain rho = 0.9 (ILLUSTRATIVE" in text
    assert "kappa = 236.932" in text
    assert "contraction number after scaling: 0.9 < 1" in text
    assert "no node below the floor" in text
