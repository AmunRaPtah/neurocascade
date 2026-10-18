"""P30.1: train-only effect model and explicit provenance fallbacks."""

import pytest

from neurocascade.benchmark.effects import (
    EFFECTS,
    effect_labels,
    fit_effect_models,
    heldout_evaluation,
    leave_one_drug_out,
    clinical_exposure,
)
from neurocascade.benchmark import load_split


def test_fit_rejects_heldout_drug_before_any_model_is_built():
    with pytest.raises(ValueError, match="non-training"):
        fit_effect_models(drugs=[load_split()["heldout"][0]])


def test_observed_and_proxy_exposure_tiers_are_distinct():
    observed = clinical_exposure("olanzapine")
    proxy = clinical_exposure("aripiprazole")
    assert observed.tier == 1
    assert observed.observed_cl_over_f_l_h == pytest.approx(25.0)
    assert proxy.tier == 2
    assert proxy.observed_cl_over_f_l_h is None
    assert "dose/Ki" in " ".join(proxy.notes)


def test_effect_label_records_never_fabricate_a_placebo_percentage():
    rows = {row.effect: row for row in effect_labels("aripiprazole")}
    assert set(rows) == set(EFFECTS)
    assert rows["eps"].drug_percent == pytest.approx(25.0)
    assert rows["eps"].placebo_percent is None
    assert rows["eps"].placebo_pair_available is False
    assert "fallback" in rows["eps"].rule_applied


def test_loo_reports_counts_and_underpowered_effects():
    result = leave_one_drug_out()
    assert set(result) == set(EFFECTS)
    assert result["eps"]["n_positive"] == 10
    assert result["eps"]["n_negative"] == 20
    assert result["anticholinergic"]["underpowered"] is True
    assert result["seizure"]["underpowered"] is True


def test_each_effect_exposes_receptor_coefficients():
    models, fitted = fit_effect_models()
    assert len(fitted) == 35
    assert set(models) == set(EFFECTS)
    for effect, model in models.items():
        assert model.n_positive + model.n_negative > 0
        assert len(model.coefficient_table()) == len(model.targets)
        assert all(set(row) == {"target", "coefficient", "per"}
                   for row in model.coefficient_table())


def test_loo_predictions_vary_beyond_the_fold_intercept():
    """A fit that has collapsed to an intercept cannot rank drugs.

    Every fold leaves out one drug, so an intercept-only model predicts the
    left-out drug below a positive's fold mean and above a negative's, giving
    an AUROC of exactly 0.000; that is what the first P30.1 fit produced. A fit
    with real per-drug variation spreads its predictions.
    """
    result = leave_one_drug_out()
    for effect, rows in result.items():
        if rows["auroc"] is None:
            continue
        probabilities = [row["predicted_probability"] for row in rows["predictions"]]
        spread = max(probabilities) - min(probabilities)
        assert spread > 0.01, f"{effect}: leave-one-out predictions collapsed (spread {spread})"


def test_heldout_truth_is_never_read_as_a_negative():
    """Truth-stripped checkout: searched-and-empty, every drug accounted for.

    Full-truth host: score what is scorable and name the reason for the rest.
    Either way a missing label is never read as a negative and no drug is both
    scored and missing.
    """
    result = heldout_evaluation()
    heldout = load_split()["heldout"]
    missing_truth, missing_input = result["missing_truth"], result["missing_input"]
    assert not set(missing_truth) & set(missing_input)
    assert set(missing_truth) | set(missing_input) <= set(heldout)
    if result["status"] == "searched-and-empty":
        assert result["n_scored"] == 0
        assert len(missing_truth) == len(heldout)
    else:
        assert result["status"] == "scored"
        assert result["n_scored"] > 0
        assert missing_truth == []
        unlabelled = result["unlabelled"]
        assert not set(unlabelled) & (set(missing_truth) | set(missing_input))
        assert (result["n_scored"] + len(missing_input) + len(unlabelled)
                == len(heldout))
        for drug in unlabelled:
            assert all(row.label not in (0, 1) for row in effect_labels(drug))
