"""P22.2 scoring rules, as fixed in docs/p22-preregistration.md."""

import math

import pytest

from neurocascade.human.label_dose_scoring import (
    Score, aggregate, refused, round_sig, score,
)

LOG2 = math.log10(2)


def test_round_sig():
    assert round_sig(2.8924) == 2.89
    assert round_sig(5.5649) == 5.56
    assert round_sig(14.749) == 14.7
    assert round_sig(1234.5) == 1230
    assert round_sig(0.012345) == 0.0123


def test_touching_endpoint_counts_as_overlap():
    s = score("x", 2, 4, 4, 8)
    assert s.overlap is True
    assert s.refusal is None


def test_disjoint_windows_do_not_overlap():
    s = score("x", 1, 3.9, 4, 8)
    assert s.overlap is False
    assert s.passed is False


def test_exact_twofold_boundary_passes():
    s = score("x", 1, 4, 2, 8)
    assert s.e == pytest.approx(-LOG2, abs=1e-12)
    assert s.overlap and s.passed


def test_beyond_twofold_fails_even_with_overlap():
    s = score("x", 1, 3, 3, 30)
    assert s.overlap is True
    assert s.e == pytest.approx(math.log10(math.sqrt(3) / math.sqrt(90)))
    assert s.passed is False


def test_sign_of_e_is_prediction_over_label():
    assert score("x", 10, 20, 2, 4).e > 0
    assert score("x", 2, 4, 10, 20).e < 0


def test_rounding_happens_before_scoring():
    # 3.9996 prints as 4.00, so it touches a label starting at 4.
    assert score("x", 2, 3.9996, 4, 8).overlap is True


def test_unreachable_is_scored_failure_not_refusal():
    s = score("x", 3.0, None, 2, 8)
    assert (s.overlap, s.e, s.passed, s.refusal) == (False, None, False, None)


@pytest.mark.parametrize("args", [
    (0, 4, 2, 8), (1, 4, -2, 8), (5, 4, 2, 8), (1, 4, 9, 8),
])
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        score("x", *args)


def test_refused():
    s = refused("quetiapine", "receptor: no D2 plasma EC50")
    assert s.refusal == "receptor: no D2 plasma EC50"
    assert (s.overlap, s.e, s.passed) == (False, None, False)
    with pytest.raises(ValueError):
        refused("x", "")


def test_aggregate():
    scores = [
        score("a", 1, 4, 2, 8),        # pass, e = -log2
        score("b", 10, 20, 2, 4),      # no overlap, fail
        score("c", 3.0, None, 2, 8),   # unreachable, scored
        refused("d", "pk"),
    ]
    agg = aggregate(scores)
    assert agg["n_pass"] == 1
    assert agg["n_scored"] == 3
    assert agg["denominator"] == 8
    e_b = math.log10(math.sqrt(200) / math.sqrt(8))
    assert agg["max_abs_e"] == pytest.approx(e_b)
    assert agg["median_abs_e"] == pytest.approx((LOG2 + e_b) / 2)
    assert agg["per_drug"]["c"] is None and agg["per_drug"]["d"] is None
    assert agg["per_drug"]["a"] == pytest.approx(-LOG2)


def test_aggregate_all_refused():
    agg = aggregate([refused("a", "pk")])
    assert agg["median_abs_e"] is None and agg["max_abs_e"] is None


def test_aggregate_rejects_duplicates_and_overflow():
    with pytest.raises(ValueError):
        aggregate([refused("a", "pk"), refused("a", "pk")])
    with pytest.raises(ValueError):
        aggregate([refused(str(i), "pk") for i in range(3)], denominator=2)


def test_score_is_frozen():
    with pytest.raises(Exception):
        score("x", 1, 4, 2, 8).passed = False  # type: ignore[misc]
    assert isinstance(score("x", 1, 4, 2, 8), Score)
