"""P32.4 prospective blind test: packages, blind predictions and scoring."""

import json
import math

import pytest

from neurocascade import prospective as P
from neurocascade.compound import CoreInputMissing, load_package, predict_compound

BUILT = ["aticaprant", "basmisanil", "darigabat", "fezolinetant",
         "seltorexant", "ulotaront", "vornorexant"]
UNBUILDABLE = ["centanafadine", "deutetrabenazine", "emraclidine", "tavapadon", "xanomeline"]
SYSTEMIC = ["basmisanil", "fezolinetant"]


def test_seven_packages_built_and_run():
    assert P.package_names() == BUILT
    for name in BUILT:
        pkg = load_package(P.PACKAGE_DIR / f"{name}.json")
        result = predict_compound(pkg)  # must not refuse
        assert result["human_pk"]["clearance"]["status"] == "predicted"
        assert result["human_pk"]["clearance"]["provenance"]["compound_specific"] is True


def test_unbuildable_listed_with_missing_clearance_field():
    assert sorted(P.UNBUILDABLE) == UNBUILDABLE
    for name, why in P.UNBUILDABLE.items():
        assert why.startswith("clearance:")


def test_predictions_are_blind_no_observed_read():
    """run_predictions reads only packages; no observed-truth value appears."""
    blob = json.dumps(P.run_predictions(), sort_keys=True)
    # the observed IV clearance of basmisanil (0.97) and dose strings must not leak
    assert "0.97" not in blob
    assert "menopausal" not in blob and "Galluppi" not in blob


def test_prediction_file_hash_is_stable():
    first = P.write_predictions()
    second = P.write_predictions()
    assert first == second == P.predictions_sha256()
    assert len(first) == 64


def test_score_counts_and_systemic_split():
    r = P.score()
    assert r["n_compounds_built"] == 7
    assert r["n_scored"] == 6          # seltorexant has no published CL -> unscored
    assert r["n_scored_systemic"] == 2
    assert r["systemic_compounds"] == SYSTEMIC
    assert r["oral_clf_compounds"] == ["aticaprant", "darigabat", "ulotaront", "vornorexant"]


def test_seltorexant_unscored_reason():
    r = P.score()
    row = next(x for x in r["rows"] if x["compound"] == "seltorexant")
    cl = row["clearance"]
    assert cl["scored"] is False
    assert cl["cl_basis"] == "searched-and-empty"


def test_basmisanil_is_the_clean_systemic_comparison():
    r = P.score()
    row = next(x for x in r["rows"] if x["compound"] == "basmisanil")
    cl = row["clearance"]
    assert cl["cl_basis"] == "iv_systemic"
    assert cl["observed_mL_min_kg"] == 0.97
    assert cl["predicted_mL_min_kg"] == pytest.approx(0.347, abs=0.01)
    assert cl["fold_error"] == pytest.approx(2.79, abs=0.02)
    # IVIVE beats the v2 train-mean baseline on this low-clearance drug
    assert cl["beats_mean_baseline"] is True


def test_overall_gmfe_and_mean_baseline():
    r = P.score()
    assert r["clearance_gmfe"] == pytest.approx(2.88, abs=0.03)
    assert r["mean_baseline_gmfe"] == pytest.approx(3.05, abs=0.03)
    assert r["beats_mean_baseline"] is True
    # every scored fold error is finite and positive
    for row in r["rows"]:
        if row["clearance"]["scored"]:
            assert math.isfinite(row["clearance"]["fold_error"]) and row["clearance"]["fold_error"] > 0


def test_dose_and_occupancy_are_pending_not_scored():
    r = P.score()
    for row in r["rows"]:
        assert row["dose"]["scored"] is False
        assert row["dose"]["predicted_status"] == "pending"
        assert row["occupancy"]["scored"] is False
        assert row["effects"]["scored"] is False


def test_every_row_carries_provenance_label():
    r = P.score()
    assert r["label"] == P.PROVENANCE_LABEL
    for row in r["rows"]:
        assert row["label"] == P.PROVENANCE_LABEL


def test_markdown_renders_hash_and_table():
    r = P.score()
    md = P.render_markdown(r)
    assert P.predictions_sha256() in md
    for name in BUILT + UNBUILDABLE:
        assert name in md
    assert P.PROVENANCE_LABEL in md
