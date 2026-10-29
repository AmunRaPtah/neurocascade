"""P32.4 commit 2: the scored half -- observed truth present, compared to blind predictions."""

import json
from pathlib import Path

import pytest

from neurocascade.benchmark import prospective as P

REPO = Path(__file__).resolve().parents[1]


def test_prediction_file_unchanged_by_the_truth_commit():
    """The scorer must read the byte-identical file that was hashed when blind."""
    committed = (REPO / "results" / "P32.4_predictions.json").read_bytes()
    assert committed == P.predictions_bytes()
    assert P.sha256_predictions() == \
        "82aeba8d981b1d8206aafc755585429e5133ca08e5a7be5873a0f1d099ca0eb7"


def test_every_buildable_compound_has_observed_truth_now():
    for name in P.PACKAGES:
        obs = P.load_observed(name)
        assert obs is not None, name
        assert obs["clearance"]["value_mL_min_kg"] > 0
        assert obs["clearance"]["refs"]


def test_scoring_reports_per_compound_fold_errors_and_baseline():
    s = P.score_prospective()
    assert s["n_scored"] == len(P.PACKAGES) >= 3
    for name, rec in s["per_compound"].items():
        assert rec["status"] == "scored"
        assert rec["fold_error_prediction"] >= 1.0
        assert rec["fold_error_mean_baseline"] >= 1.0
        # the mean baseline is shown beside every prediction
        assert rec["mean_baseline_cl_mL_min_kg"] == pytest.approx(P.MEAN_BASELINE_CL)
        assert rec["dose"]["predicted"] == "pending"
        assert rec["occupancy"]["predicted"] == "pending"


def test_liabilities_are_covered_but_uninformative():
    s = P.score_prospective()
    assert s["liabilities"]["n_informative"] == 0
    for effect, frac in s["liabilities"]["coverage"].items():
        n, d = frac.split("/")
        assert n == d  # {0,1} always covers


def test_results_file_reports_n_scored():
    data = json.loads((REPO / "results" / "P32.4.json").read_text())
    assert data["task"] == "P32.4"
    assert data["n_scored"] == len(P.PACKAGES)
    assert data["n_scored"] >= 1  # not EMPTY


def test_truth_absent_path_is_searched_and_empty(tmp_path, monkeypatch):
    """With no observed files (the blind state / host-pending), scoring degrades gracefully."""
    monkeypatch.setattr(P, "_OBSERVED_DIR", tmp_path)
    s = P.score_prospective()
    assert s["status"] == "searched-and-empty"
    assert s["n_scored"] == 0
    assert sorted(s["missing_truth"]) == sorted(P.PACKAGES)
    for rec in s["per_compound"].values():
        assert rec["status"] == "missing_truth"
        # the prediction itself survives without truth
        assert rec["predicted_cl_mL_min_kg"] > 0


def test_pmids_in_observed_truth_name_the_drug_or_are_noted():
    # light check that refs are present and cite a source per scored quantity
    for name in P.PACKAGES:
        obs = P.load_observed(name)
        assert obs["clearance"]["refs"]
        assert obs["dose"]["refs"]
