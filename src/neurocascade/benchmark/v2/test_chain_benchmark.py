"""P33.3 host-evaluation tests and report entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import load_v2_binding, load_v2_drugs, v2_partition
from .chain_benchmark import (
    FINGERPRINT_FLOOR,
    LIABILITIES,
    SCORED_LIABILITIES,
    _all_targets,
    _continuous_stage,
    _fingerprint,
    _liability_stage,
    auprc,
    auroc,
    heldout_summary_lines,
    run_chain_benchmark,
)


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def report():
    return run_chain_benchmark()


def test_metrics_are_defined_on_small_known_vectors():
    assert auroc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(1.0)
    assert auprc([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(5 / 6)
    assert auroc([1, 1], [0.2, 0.8]) is None


def test_partition_is_from_the_v2_api_not_p28_split():
    train, heldout = v2_partition()
    assert len(train) + len(heldout) == len(load_v2_drugs())
    assert not set(train) & set(heldout)
    assert 0.15 < len(heldout) / len(load_v2_drugs()) < 0.25


def test_report_has_every_required_stage_and_scope(report):
    for stage in ("human_cl", "vss", "f", "free_plasma_exposure", "kp_uu", "multi_target_occupancy"):
        assert stage in report["stages"]
    for effect in LIABILITIES:
        assert effect in report["stages"]
    assert set(SCORED_LIABILITIES) == {"sedation", "anticholinergic"}
    assert report["stages"]["f"]["status"] == "no data"
    assert report["stages"]["kp_uu"]["status"] == "no data"
    assert report["available_data_n_scored"] > 0


def test_fingerprint_uses_full_panel_and_inactivity_floor():
    # The floor is the weak-binding end of the scale, never 0 (which would be
    # Ki = 1 nM, i.e. potent binding asserted at an unmeasured target).
    assert FINGERPRINT_FLOOR == pytest.approx(-5.0)
    panel = {
        "targets": {"HRH1": {"ki_nM": 1.0}},
        "searched_and_empty": ["CHRM1"],
    }
    targets = ("CHRM1", "HRH1")
    vector = _fingerprint(panel, targets)
    assert list(vector) == pytest.approx([FINGERPRINT_FLOOR, 0.0])  # -log10(1) == 0
    # A drug with no measured affinity anywhere is refused, not all-floor.
    assert _fingerprint({"searched_and_empty": ["HRH1"]}, ("HRH1",)) is None


def test_fingerprint_panel_spans_every_v2_target():
    panel = _all_targets(load_v2_binding())
    assert len(panel) == 37
    assert {"HRH1", "CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5"} <= set(panel)


def test_one_nn_fingerprint_is_independent_of_the_ki_baseline(report):
    # Over the full panel the 1-NN baseline covers far more drugs than the
    # handful carrying the effect's own measurement, so it is a genuine
    # comparator rather than a copy of strongest-target Ki.
    sed = report["stages"]["sedation"]["available_data"]["baselines"]
    assert sed["one_nn_fingerprint"]["n"] > 2 * sed["strongest_target_ki"]["n"]


def test_underpowered_and_withheld_liabilities_are_not_scored(report):
    stages = report["stages"]
    assert stages["eps"]["status"] == "underpowered"
    assert stages["orthostatic_hypotension"]["status"] == "underpowered"
    assert stages["weight_gain"]["status"] == "unavailable"
    assert all("baselines" not in stages[e] for e in ("eps", "orthostatic_hypotension", "weight_gain"))


def test_heldout_continuous_branch_scores_values_not_inn_keys():
    # Regression: the held-out branch once passed the {inn: value} dict to
    # gmfe(), which iterates INN strings and raises TypeError on the host.
    # Dummy values only; no held-out truth is involved.
    truth = {inn: {"vss": {"value": v}} for inn, v in {"a": 1.0, "b": 4.0, "x": 2.0, "y": 10.0}.items()}
    stage = _continuous_stage("vss", "Vss", {"a", "b"}, {"x", "y", "z"}, truth)
    held = stage["heldout"]
    assert held["status"] == "scored"
    assert held["n"] == 2
    assert held["missing_truth"] == ["z"]
    # Prediction 2.5: fold errors 1.25 and 4, GMFE = sqrt(5).
    assert held["gmfe"] == pytest.approx(5 ** 0.5)
    assert held["fold_error_median"] == pytest.approx((1.25 + 4.0) / 2)


def test_heldout_liability_branch_scores_dummy_rows():
    truth = {
        inn: {"sedation": {"label": label}}
        for inn, label in {"a": 1, "b": 0, "c": 1, "d": 0, "x": 1, "y": 0}.items()
    }
    binding = {
        inn: {"targets": {"HRH1": {"ki_nM": ki}}}
        for inn, ki in {"a": 1.0, "b": 1000.0, "c": 2.0, "d": 500.0, "x": 3.0, "y": 800.0}.items()
    }
    drugs = {inn: {"atc": ["R06AA02"] if inn in "acx" else ["N02BE01"]} for inn in "abcdxy"}
    stage = _liability_stage(
        "sedation", {"a", "b", "c", "d"}, {"x", "y"}, truth, binding, drugs, ("HRH1",)
    )
    held = stage["heldout"]
    assert held["status"] == "scored"
    assert held["n_truth"] == 2
    assert held["baselines"]["strongest_target_ki"]["auroc"] == pytest.approx(1.0)
    assert held["baselines"]["one_nn_fingerprint"]["auroc"] == pytest.approx(1.0)
    assert held["baselines"]["atc_class_prevalence"]["auroc"] == pytest.approx(1.0)
    assert held["missing_truth"] == []


def test_heldout_summary_lines_name_every_scored_stage(report):
    text = "\n".join(heldout_summary_lines(report))
    for stage in ("human_cl", "vss", "sedation", "anticholinergic", "kp_uu", "eps"):
        assert stage in text


if __name__ == "__main__":
    result = run_chain_benchmark()
    output = ROOT / "results" / "P33.3.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
