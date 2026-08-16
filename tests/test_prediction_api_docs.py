"""P10.2: public prediction documentation and minimal API contract."""

from pathlib import Path

from neurocascade import OneCompartmentPK
from neurocascade.fingerprint import BindingFingerprint, FingerprintEntry, predict


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOC = ROOT / "docs" / "prediction-api.md"


def test_prediction_api_guide_contains_both_prediction_levels():
    text = DOC.read_text()
    assert "fingerprint.predict()" in text
    assert "build_whole_brain_model" in text
    assert "build_multi_target_model" in text
    assert "does not produce a behavioral endpoint ratio" in text


def test_minimal_fingerprint_example_runs_through_public_api():
    fingerprint = BindingFingerprint(
        "documentation_example",
        entries={
            "D2": FingerprintEntry(
                target="D2", ki_nM=10.0, efficacy=0.0,
                efficacy_source="test antagonist",
            )
        },
    )
    prediction = predict(
        fingerprint,
        dose=25.0,
        route="IV",
        species="rat",
        body_weight_kg=0.25,
        pk_model=OneCompartmentPK(ke=0.02),
        n_times=21,
    )
    assert prediction["D2"].peak_occupancy > 0.0
    assert prediction.occupancy_ranking()[0][0] == "D2"


def test_readme_has_prominent_limits_and_parameter_inventory():
    text = README.read_text()
    assert "## Status: current model coverage" in text
    assert "### Parameter calibration status" in text
    for parameter_class in (
        "Administered-dose PK",
        "Plasma-to-brain partition",
        "Receptor affinity",
        "Circuit node parameters",
        "Behavioral endpoint weight magnitudes",
    ):
        assert parameter_class in text
    assert "not yet a validated arbitrary-compound-to-behavior predictor" in text
