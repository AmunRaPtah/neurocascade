"""P29.2: human PK translation must be train-only and data-backed."""

import math

import pytest

from neurocascade.benchmark import load_split
import neurocascade.benchmark.human_pk_translation as translation
from neurocascade.benchmark.human_pk_translation import (
    fit_translation_models,
    score_heldout,
)


def test_every_fitted_model_names_only_training_drugs():
    split = load_split()
    models = fit_translation_models()
    heldout = set(split["heldout"])
    assert models
    for model in models.values():
        assert model.fit_drugs
        assert not heldout.intersection(model.fit_drugs)


def test_fit_is_explicitly_independent_of_heldout_names():
    split = load_split()
    baseline = fit_translation_models(split["train"])
    # Supplying held-out names as an additional iterable is not an allowed
    # hidden path: the fitted drug set must make the leak observable.
    leaked = fit_translation_models(split["train"] + split["heldout"])
    assert set(leaked["cl_allometric"].fit_drugs) != set(
        baseline["cl_allometric"].fit_drugs
    )
    assert set(leaked["vss_allometric"].fit_drugs) != set(
        baseline["vss_allometric"].fit_drugs
    )


def test_default_fit_refuses_to_read_heldout_records(monkeypatch):
    heldout = set(load_split()["heldout"])
    original_adme = translation.load_adme_record
    original_truth = translation.load_truth_record

    def guarded_adme(inn):
        assert inn not in heldout
        return original_adme(inn)

    def guarded_truth(inn):
        assert inn not in heldout
        return original_truth(inn)

    monkeypatch.setattr(translation, "load_adme_record", guarded_adme)
    monkeypatch.setattr(translation, "load_truth_record", guarded_truth)
    translation.fit_translation_models()


def test_heldout_metrics_are_scored_and_have_fixed_fold_definitions():
    report = score_heldout()
    assert set(report) == {"cl", "vss"}
    for metric in report.values():
        assert metric.n == len(metric.predictions) > 0
        assert metric.geometric_mean_fold_error == pytest.approx(
            math.exp(
                sum(math.log(prediction.fold_error) for prediction in metric.predictions)
                / metric.n
            )
        )
        assert metric.within_2_fold == pytest.approx(
            sum(prediction.fold_error <= 2 for prediction in metric.predictions)
            / metric.n
        )
        assert metric.within_3_fold == pytest.approx(
            sum(prediction.fold_error <= 3 for prediction in metric.predictions)
            / metric.n
        )


def test_heldout_scores_are_the_pinned_current_result():
    report = score_heldout()
    assert report["cl"].n == 5
    assert report["cl"].geometric_mean_fold_error == pytest.approx(6.774746573244732)
    assert report["cl"].within_2_fold == pytest.approx(0.2)
    assert report["cl"].within_3_fold == pytest.approx(0.2)
    assert report["vss"].n == 3
    assert report["vss"].geometric_mean_fold_error == pytest.approx(1.2845743806996)
    assert report["vss"].within_2_fold == pytest.approx(1.0)
    assert report["vss"].within_3_fold == pytest.approx(1.0)


def test_predictions_carry_citations_and_never_use_heldout_fit_rows():
    split = load_split()
    report = score_heldout()
    heldout = set(split["heldout"])
    for metric in report.values():
        for prediction in metric.predictions:
            assert prediction.drug in heldout
            assert prediction.input_references
            assert prediction.predicted > 0
            assert prediction.observed > 0
