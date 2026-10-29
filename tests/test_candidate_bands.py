"""P30.4: conformal bands refit on the P34.3 frozen candidate's train residuals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurocascade.benchmark.conformal import UnbandedQuantityError
from neurocascade.benchmark.v2 import candidate_bands as cb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen():
    return cb.load_bands()


def test_frozen_band_file_reproduces_from_train_data(frozen):
    assert json.loads(json.dumps(cb.calibrate())) == frozen
    assert frozen["heldout_read_at_freeze"] is False
    assert cb.train_only_invariant(frozen)


def test_bands_are_tied_to_the_p34_3_frozen_candidate(frozen):
    p34_3 = json.loads((ROOT / "results" / "P34.3.json").read_text())
    assert frozen["candidate"]["code_sha256"] == p34_3["code"]["sha256"]
    assert frozen["candidate"]["params_sha256"] == p34_3["params"]["sha256"]


def test_every_output_carries_the_not_validated_label(frozen):
    assert cb.PROVENANCE_LABEL == "not validated: P34 returned no verdict"
    assert frozen["provenance"] == cb.PROVENANCE_LABEL
    for entry in frozen["quantities"].values():
        assert entry["provenance"] == cb.PROVENANCE_LABEL


def test_engagement_percentile_spans_zero_to_one():
    ref_occ, ref_aff = [0.1, 0.2, 0.3], [5.0, 6.0, 7.0]
    assert cb.engagement_percentile(0.4, 8.0, ref_occ, ref_aff) == 1.0
    assert cb.engagement_percentile(0.0, 4.0, ref_occ, ref_aff) == 0.0
    # Middle of both views: ranks 2.5 + 2.5 = 5 among four drugs -> (5 - 2) / 6.
    assert cb.engagement_percentile(0.2, 6.0, ref_occ, ref_aff) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cb.engagement_percentile(0.2, 6.0, [], [])


def test_sedation_bands_pinned(frozen):
    e = frozen["quantities"]["liability:sedation"]
    assert (e["n_calibration"], e["n_positive"]) == (19, 8)
    c, t = e["candidate"], e["train_mean"]
    assert c["quantiles"]["0.80"] == pytest.approx(25 / 36)
    assert c["quantiles"]["0.95"] == 1.0
    assert c["lodo_coverage"]["0.80"] == {"covered": 16, "n": 19, "rate": 16 / 19, "n_unbounded": 0}
    # 18 others at 95%: rank ceil(19 * 0.95) = 19 > 18, so every LODO band is unbounded.
    assert c["lodo_coverage"]["0.95"]["n_unbounded"] == 19
    assert c["lodo_set_sizes"]["0.80"] == {"0": 0, "1": 11, "2": 8}
    assert c["lodo_set_sizes"]["0.95"] == {"0": 0, "1": 0, "2": 19}
    assert t["frozen_point"] == pytest.approx(8 / 19)
    assert t["lodo_coverage"]["0.80"]["covered"] == 19
    assert t["lodo_set_sizes"]["0.80"] == {"0": 0, "1": 0, "2": 19}


def test_anticholinergic_bands_pinned(frozen):
    e = frozen["quantities"]["liability:anticholinergic"]
    assert (e["n_calibration"], e["n_positive"]) == (25, 9)
    c, t = e["candidate"], e["train_mean"]
    assert c["quantiles"]["0.80"] == pytest.approx(53 / 96)
    assert c["quantiles"]["0.95"] == pytest.approx(0.875)
    assert (c["lodo_coverage"]["0.80"]["covered"], c["lodo_coverage"]["0.95"]["covered"]) == (20, 24)
    assert c["lodo_set_sizes"]["0.80"] == {"0": 0, "1": 24, "2": 1}
    assert c["lodo_set_sizes"]["0.95"] == {"0": 0, "1": 3, "2": 22}
    assert t["lodo_set_sizes"]["0.95"] == {"0": 0, "1": 0, "2": 25}


def test_underpowered_and_pk_quantities_are_not_candidate_banded(frozen):
    q = frozen["quantities"]
    assert q["liability:eps"]["status"] == "underpowered"
    assert q["liability:orthostatic_hypotension"]["status"] == "underpowered"
    assert q["human_cl"]["status"] == "no-candidate-prediction"
    assert q["human_cl"]["train_mean"]["fold_bands"]["0.95"] == pytest.approx(23.852743833727345)
    assert q["human_vss"]["status"] == "no-candidate-prediction"
    with pytest.raises(UnbandedQuantityError):
        cb.predict_with_bands("eps", 10.0, {"DRD2": 1.0}, frozen)
    with pytest.raises(KeyError):
        cb.predict_with_bands("seizure", 10.0, {}, frozen)


def test_predict_with_bands_returns_label_sets_not_a_probability(frozen):
    out = cb.predict_with_bands("sedation", 1e6, {"HRH1": 0.01}, frozen)
    assert out["engagement_percentile"] == 1.0 and out["is_probability"] is False
    assert out["bands"]["0.80"] == {"labels": [1], "bounded": True}
    assert out["bands"]["0.95"] == {"labels": [0, 1], "bounded": True}
    assert out["provenance"]["label"] == cb.PROVENANCE_LABEL
    with pytest.raises(ValueError):
        cb.predict_with_bands("sedation", None, {"HRH1": 1.0}, frozen)


def test_heldout_branch_scores_synthetic_rows(frozen):
    """Dummy in-memory rows only, never real held-out truth."""
    cmax = {"X": {"status": "value", "cmax_unbound_nM": 1e6}}
    binding = {"X": {"targets": {"HRH1": {"ki_nM": 0.01}, "CHRM1": {"ki_nM": 0.01}}}}
    liabilities = {"X": {"sedation": {"label": 1}, "anticholinergic": {"label": 0}},
                   "Z": {"sedation": {"label": None}, "anticholinergic": {"label": 1}}}
    r = cb.heldout_coverage(frozen, heldout=["X", "Y", "Z"], liabilities=liabilities,
                            cmax=cmax, binding=binding, include_pk=False)
    assert r["status"] == "scored" and r["retrospective"] and not r["certifies_bands"]
    sed = r["quantities"]["liability:sedation"]
    assert sed["missing_truth"] == ["Y"] and sed["not_scoreable"] == ["Z"]
    assert sed["candidate"]["coverage"]["0.80"] == {"covered": 1, "n": 1, "rate": 1.0}
    assert sed["candidate"]["set_sizes"]["0.80"] == {"0": 0, "1": 1, "2": 0}
    assert sed["train_mean"]["set_sizes"]["0.80"] == {"0": 0, "1": 0, "2": 1}
    ach = r["quantities"]["liability:anticholinergic"]
    # u = 1 with label 0 scores 1.0 > every frozen quantile below 1: a miss.
    assert ach["candidate"]["coverage"]["0.95"]["covered"] == 0
    assert any("retrospective" in line for line in cb.heldout_summary_lines(r))


def test_results_file_matches_the_band_file(frozen):
    res = json.loads((ROOT / "results" / "P30.4.json").read_text())
    drugs = set()
    for name in ("liability:sedation", "liability:anticholinergic"):
        drugs |= set(frozen["quantities"][name]["calibration_drugs"])
    assert res["n_scored"] == len(drugs)
    assert res["provenance"] == cb.PROVENANCE_LABEL
    assert res["heldout"]["retrospective"] is True
