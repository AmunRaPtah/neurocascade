"""P32.4 commit 1: the blind half -- predictions exist and carry no human truth.

These tests pin the prediction side of the prospective test.  They must pass in
a checkout that has NO observed-truth files (the state of the first commit), so
they never import or read ``prospective.observed``.
"""

import json
from pathlib import Path

import pytest

from neurocascade.benchmark import prospective as P
from neurocascade.compound import CoreInputMissing, predict_compound, validate_package

REPO = Path(__file__).resolve().parents[1]
PRED_FILE = REPO / "results" / "P32.4_predictions.json"


def test_at_least_three_compounds_build():
    assert len(P.PACKAGES) >= 3
    for name, pkg in P.PACKAGES.items():
        validate_package(pkg)  # a valid compound_package/1, no unknown fields
        predict_compound(pkg)  # does not refuse


def test_packages_carry_no_human_truth():
    """A preclinical package must not smuggle in the quantities it is scored on."""
    blob = json.dumps(P.PACKAGES)
    assert "human_exposure" not in blob          # no projected human Cmax
    assert "cl_mL_min_kg" not in blob             # no human clearance truth
    for pkg in P.PACKAGES.values():
        assert set(pkg) <= {"schema", "name", "binding", "adme"}


def test_every_package_has_affinity_and_clearance_input():
    for name, pkg in P.PACKAGES.items():
        kis = [e.get("ki_nM") for e in pkg["binding"].values()]
        assert any(k for k in kis), name
        fields = pkg["adme"]["fields"]
        has_rat_cl = bool(fields.get("invivo_cl_preclinical"))
        has_clint = bool(fields.get("clint"))
        assert has_rat_cl or has_clint, name


def test_each_prediction_is_banded_and_labelled():
    blob = P.predictions_blob()
    for name, pred in blob["predictions"].items():
        cl = pred["clearance"]
        assert cl["predicted_cl_mL_min_kg"] > 0
        assert cl["compound_specific"] is True
        assert set(cl["bands"]) == set(P.LEVELS)
        assert cl["mean_baseline_cl_mL_min_kg"] == pytest.approx(P.MEAN_BASELINE_CL)
        assert pred["provenance"]["label"].startswith("not validated")
        # dose and occupancy are pending, not numbers
        assert pred["target_engagement_dose"]["status"] == "pending"
        assert pred["panel_occupancy"]["status"] == "pending"


def test_committed_prediction_file_matches_recomputation_and_hash():
    assert PRED_FILE.exists()
    committed = PRED_FILE.read_bytes()
    assert committed == P.predictions_bytes()          # deterministic, byte-identical
    import hashlib
    digest = hashlib.sha256(committed).hexdigest()
    assert digest == P.sha256_predictions()
    # the hash recorded in the ledger must match the committed file
    ledger = (REPO / "docs" / "validation-ledger.md").read_text()
    assert digest in ledger


def test_could_not_build_names_the_missing_field():
    assert len(P.COULD_NOT_BUILD) >= 1
    for rec in P.COULD_NOT_BUILD:
        assert rec["name"] and rec["missing"] and rec["detail"]
        assert rec["name"] not in P.PACKAGES  # not both built and unbuilt


def test_observed_truth_not_required_for_prediction():
    """The blind commit has no observed/ directory; prediction must not need it."""
    # predict_one for every compound without any truth lookup
    for name in P.PACKAGES:
        pred = P.predict_one(name)
        assert pred["clearance"]["predicted_cl_mL_min_kg"] > 0


def test_refusal_still_raises_for_empty_package():
    with pytest.raises(CoreInputMissing):
        predict_compound({"schema": P.PACKAGES["ulotaront"]["schema"],
                          "name": "empty", "binding": {}, "adme": {"fields": {}}})
