"""Tests for P32.4: prospective blind test on repository-absent CNS compounds."""

from __future__ import annotations

import glob
import json
import os
import subprocess

import pytest

from neurocascade import prospective as P
from neurocascade.compound import PROVENANCE_LABEL

_BENCH = os.path.join(os.path.dirname(P.__file__), "benchmark")
_REPO = os.path.abspath(os.path.join(os.path.dirname(P.__file__), "..", ".."))

_ALL_NAMES = [e["name"] for e in P.SURVEY]


def _surveyed():
    return list(P.SURVEY)


def test_at_least_six_compounds_surveyed():
    # Acceptance: pick at least 6 CNS compounds.
    assert len(P.SURVEY) >= 6


@pytest.mark.parametrize("entry", _surveyed(), ids=_ALL_NAMES)
def test_every_surveyed_compound_is_absent_from_the_repository(entry):
    """The navacaprant failure mode: a surveyed compound must NOT already have a
    benchmark input or truth record, and must not be in the frozen split."""
    name = entry["name"]
    split = json.load(open(os.path.join(_BENCH, "split.json")))
    assert name not in set(split["train"]) | set(split["heldout"]), f"{name} is in split.json"
    for sub in ("binding", "adme", "truth"):
        files = {os.path.basename(f)[:-5] for f in glob.glob(os.path.join(_BENCH, sub, "*.json"))}
        assert name not in files, f"{name} has a benchmark {sub}/ record"
    # content scan of the v2 corpus (binding panel, PK/liability/cmax truth)
    hit = subprocess.run(["grep", "-rliw", name, os.path.join(_BENCH, "v2")],
                         capture_output=True, text=True).stdout.strip()
    assert not hit, f"{name} appears in v2 corpus: {hit}"


def test_navacaprant_is_not_surveyed():
    # It is a P28.1 held-out drug with a full binding+ADME panel in this repo.
    assert "navacaprant" not in _ALL_NAMES


def test_unbuildable_compounds_are_listed_with_a_missing_field():
    # Acceptance: compounds whose package could not be built are listed with the
    # missing field, not silently dropped.
    preds = P.build_predictions()
    for c in preds["compounds"]:
        if not c["buildable"]:
            assert c.get("missing_field"), f"{c['name']} unbuildable but no missing_field"
            assert c["prediction"] is None


def test_ulotaront_is_the_one_buildable_compound_and_predicts_systemic_cl():
    preds = P.build_predictions()
    buildable = [c for c in preds["compounds"] if c["buildable"]]
    assert [c["name"] for c in buildable] == ["ulotaront"]
    cl = buildable[0]["prediction"]["human_systemic_cl"]
    assert cl["point_ml_min_kg"] == pytest.approx(6.698245407358145, rel=1e-9)
    assert cl["compound_specific"] is True
    assert "allometry" in cl["predictor"]
    assert set(cl["bands"]) == {"0.80", "0.95"}


def test_n_scored_is_zero_no_systemic_cl_truth_exists():
    """No surveyed compound has a published human systemic CL (only apparent oral
    CL/F), so the comparison the P30.3 band is calibrated for is empty."""
    score = P.score()
    assert score["n_scored"] == 0
    assert score["status"] == "searched-and-empty"
    assert score["scored"] == []
    # every buildable compound appears in the systemic-CL searched-and-empty list
    empty_names = {r["name"] for r in score["systemic_cl_searched_and_empty"]}
    assert "ulotaront" in empty_names


def test_mean_baseline_is_beside_the_apparent_context():
    score = P.score()
    ctx = {c["name"]: c for c in score["apparent_cl_context"]}
    assert "ulotaront" in ctx
    row = ctx["ulotaront"]
    assert row["mean_baseline_apparent_fold_error"] == pytest.approx(1.4360623220091007, rel=1e-9)
    # the compound-specific prediction beats the mean baseline against apparent CL/F
    assert row["apparent_fold_error_upper_bound"] < row["mean_baseline_apparent_fold_error"]
    assert "uncounted" in row["caveat"].lower() or "upper bound" in row["caveat"].lower()


def test_predictions_are_deterministic_and_hash_stable():
    assert P.predictions_sha256() == P.predictions_sha256()
    reloaded = json.load(open(os.path.join(_REPO, "results", "P32.4_predictions.json")))
    assert P.predictions_sha256(reloaded) == P.predictions_sha256()


def test_committed_predictions_hash_matches_results_file():
    score_file = json.load(open(os.path.join(_REPO, "results", "P32.4.json")))
    assert score_file["predictions_sha256"] == P.predictions_sha256()
    assert score_file["n_scored"] == 0


def test_every_number_carries_the_provenance_label():
    preds = P.build_predictions()
    assert preds["provenance"]["label"] == PROVENANCE_LABEL
    for c in preds["compounds"]:
        assert c["provenance"]["label"] == PROVENANCE_LABEL
        if c["buildable"]:
            # the full predict_compound result carries the label on its records
            full = c["prediction"]["full_result"]
            assert full["provenance"]["label"] == PROVENANCE_LABEL
            assert full["human_pk"]["clearance"]["provenance"]["label"] == PROVENANCE_LABEL
