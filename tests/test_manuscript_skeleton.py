"""P10.3: manuscript skeleton is anchored to committed evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "manuscript-skeleton.md"


def test_manuscript_skeleton_has_research_sections_and_limitations_source():
    text = MANUSCRIPT.read_text()
    for heading in (
        "## 1. Introduction",
        "## 2. Materials and methods",
        "## 3. Results",
        "## 4. Discussion",
        "## 5. Limitations",
        "## 6. Reproducibility and data availability",
        "## 7. References",
    ):
        assert heading in text
    assert "P8.3" in text
    assert "heldout-failure-analysis.md" in text
    assert "arbitrary-compound behavioral accuracy estimate is reported" in text


def test_quantitative_claims_point_to_committed_artifacts():
    text = MANUSCRIPT.read_text()
    for artifact in (
        "results/benchmarks/p9_1_benchmark.json",
        "results/heldout/heldout_validation.txt",
        "results/identifiability/summary.txt",
        "results/heldout/heldout_errors.csv",
    ):
        assert artifact in text
    assert "Quantitative claim ledger" in text
    assert "291 regional nodes and 43,527 non-zero edges" in text
