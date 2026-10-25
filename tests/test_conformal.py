"""P30.3: distribution-free calibrated intervals and their LODO coverage."""

import math

import pytest

from neurocascade.benchmark import conformal as c


# ---- conformal machinery, pinned on synthetic inputs ----------------------
def test_conformal_log_radius_picks_the_finite_sample_order_statistic():
    # n=4, nominal 0.80 -> k = ceil(5*0.8) = 4 -> 4th smallest.
    residuals = [0.4, 0.1, 0.3, 0.2]
    assert c.conformal_log_radius(residuals, 0.80) == pytest.approx(0.4)
    # n=4, nominal 0.95 -> k = ceil(5*0.95) = 5 > 4 -> no finite radius.
    assert math.isinf(c.conformal_log_radius(residuals, 0.95))


def test_multiplicative_interval_is_symmetric_in_log_space():
    interval = c.multiplicative_interval(10.0, math.log(2.0), 0.95)
    assert interval.lower == pytest.approx(5.0)
    assert interval.upper == pytest.approx(20.0)
    assert not interval.uninformative


def test_an_infinite_radius_is_flagged_uninformative_not_clipped():
    interval = c.multiplicative_interval(10.0, math.inf, 0.80)
    assert interval.uninformative
    assert interval.lower == 0.0 and math.isinf(interval.upper)


def test_lodo_geometric_mean_leaves_each_drug_out():
    values = {"a": 1.0, "b": 10.0, "c": 100.0}
    out = c.lodo_geometric_mean(values)
    # prediction for b excludes b: geomean(1, 100) = 10.
    assert out["b"] == pytest.approx(10.0)
    assert out["a"] == pytest.approx(math.exp((math.log(10.0) + math.log(100.0)) / 2))


def test_continuous_coverage_is_near_nominal_on_a_log_normal_sample():
    rng = __import__("random").Random(0)
    values = {f"d{i}": math.exp(rng.gauss(0.0, 1.0)) for i in range(200)}
    report = c.continuous_coverage(values)
    cov80 = report["coverage"][0.80]["empirical_lodo_coverage"]
    cov95 = report["coverage"][0.95]["empirical_lodo_coverage"]
    # Split-conformal guarantees coverage >= nominal in expectation; allow a
    # finite-sample band rather than widening a tolerance to force a pass.
    assert 0.74 <= cov80 <= 0.88
    assert 0.90 <= cov95 <= 0.99
    assert report["coverage"][0.95]["frozen_log_radius"] > report["coverage"][0.80]["frozen_log_radius"]


# ---- classification label sets --------------------------------------------
def test_label_set_admits_both_labels_when_the_quantile_is_one():
    # q = 1.0 admits every label: the honest 'cannot exclude either outcome'.
    both = c.label_set_for(0.3, 1.0, 0.95)
    assert set(both.labels) == {0, 1}
    assert both.uninformative
    # A tight quantile keeps only the confident label.
    one = c.label_set_for(0.05, 0.1, 0.80)
    assert one.labels == (0,)
    assert not one.uninformative


def test_classification_coverage_is_near_nominal_and_reports_set_size():
    rng = __import__("random").Random(1)
    rows = []
    for _ in range(300):
        p = rng.random()
        y = 1 if rng.random() < p else 0  # well-specified probabilities
        rows.append((y, p))
    report = c.classification_coverage(rows)
    cov80 = report["coverage"][0.80]["empirical_lodo_coverage"]
    cov95 = report["coverage"][0.95]["empirical_lodo_coverage"]
    assert 0.74 <= cov80 <= 0.86
    assert 0.90 <= cov95 <= 0.99
    assert report["coverage"][0.95]["mean_set_size"] >= report["coverage"][0.80]["mean_set_size"]


# ---- the real continuous quantities (fast: no logistic refits) ------------
def test_v2_pk_coverage_is_calibrated_on_train_clearance_and_vss():
    for parameter in ("clearance_iv", "vss"):
        report = c.continuous_coverage(c._v2_pk_values(parameter))
        assert report["n"] >= 100
        for nominal in (0.80, 0.95):
            cov = report["coverage"][nominal]["empirical_lodo_coverage"]
            # Within a few points of nominal in both directions; no tolerance
            # is widened to hide a miss, the real number is asserted.
            assert abs(cov - nominal) <= 0.06


# ---- band attachment the product surface will call ------------------------
def test_predict_with_bands_attaches_a_band_and_declares_pending_quantities():
    injected = {
        "continuous": {
            "human_cl": {"coverage": {0.80: {"frozen_log_radius": math.log(5.0)},
                                      0.95: {"frozen_log_radius": math.log(13.0)}}},
            "vss": {"coverage": {0.80: {"frozen_log_radius": math.log(4.7)},
                                 0.95: {"frozen_log_radius": math.log(13.7)}}},
        },
        "effects": {},
    }
    result = c.predict_with_bands("haloperidol", report=injected)
    cl = result["predictions"]["human_cl"]
    assert cl["point"] > 0
    band = cl["intervals"][0.80]
    assert band.lower < cl["point"] < band.upper
    assert band.upper / band.point == pytest.approx(5.0, rel=1e-6)
    # Quantities whose upstream task is not done are pending, not invented.
    assert set(result["pending"]) == {"target_engagement_dose", "panel_occupancy"}


def test_calibration_report_without_effects_is_the_fast_continuous_surface():
    report = c.calibration_report(include_effects=False)
    assert report["task"] == "P30.3"
    assert report["n_scored"] >= 100
    assert set(report["continuous"]) == {"human_cl", "vss"}
    assert set(report["pending"]) == {"target_engagement_dose", "panel_occupancy"}
    assert "host-pending" in report["held_out_status"]
