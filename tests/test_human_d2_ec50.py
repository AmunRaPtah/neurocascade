import math

import pytest

from neurocascade.human.d2_ec50 import (
    HALOPERIDOL_D2_PLASMA_EC50_nM,
    HALOPERIDOL_PLASMA_FREE_FRACTION,
    compare_with_reported_relations,
    ec50_landscape,
    fit_plasma_ec50,
)
from neurocascade.human.pet_observations import (
    HALOPERIDOL_D2_DATASET_IDS,
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    haloperidol_d2_manifests,
    haloperidol_d2_observations,
    occupancy_observations,
)
from neurocascade.human.pet_validation import HALOPERIDOL_MW_G_PER_MOL


def test_recorded_constant_still_equals_the_recomputed_fit():
    """The guard that makes the recorded constant safe to hardcode.

    If the fixtures or the fit change, this fails rather than letting the model
    quietly run on a stale number.
    """
    fit = fit_plasma_ec50()
    assert fit.ec50_nM == pytest.approx(HALOPERIDOL_D2_PLASMA_EC50_nM, abs=1e-4)


def test_fit_uses_only_the_total_plasma_depot_points():
    fit = fit_plasma_ec50(plasma_basis="total")
    assert fit.n_points == 4
    assert fit.n_independent_groups == 2
    assert set(fit.sources) == {"nyberg1995", "regenthal1997"}
    assert fit.plasma_basis == "total"


def test_the_two_depot_studies_agree_without_being_pooled_to_get_there():
    """Independent replication is the reason a four-point fit is usable at all."""
    fit = fit_plasma_ec50()
    per_study = dict(fit.per_study_ec50_nM)
    assert set(per_study) == {"nyberg1995", "regenthal1997"}
    spread = abs(per_study["nyberg1995"] - per_study["regenthal1997"])
    assert spread / fit.ec50_nM < 0.10
    # Each study's own fit brackets the pooled one, so the pooled value is not
    # being carried by one of them.
    assert min(per_study.values()) < fit.ec50_nM < max(per_study.values())


def test_fit_reproduces_each_observed_occupancy_closely():
    fit = fit_plasma_ec50()
    points = [
        item
        for item in occupancy_observations()
        if item.plasma_concentration_basis == "total"
        and item.plasma_concentration_nM is not None
    ]
    assert len(points) == 4
    for item in points:
        predicted = fit.occupancy_at(item.plasma_concentration_nM)
        assert abs(predicted - item.observed_occupancy) < 0.06
    assert fit.residual_rmse < 0.05


def test_unit_conversion_of_the_fitted_constant():
    fit = fit_plasma_ec50()
    assert fit.ec50_ng_per_mL == pytest.approx(
        fit.ec50_nM * HALOPERIDOL_MW_G_PER_MOL / 1000.0
    )
    # The published relations are printed in ng/ml, so this is the axis on which
    # the comparison in the docs is made.
    assert fit.ec50_ng_per_mL == pytest.approx(0.708, abs=0.005)


def test_interval_is_reported_with_its_dependence_caveat():
    fit = fit_plasma_ec50()
    assert fit.interval_low_nM < fit.ec50_nM < fit.interval_high_nM
    assert fit.standard_error_nM is not None and fit.standard_error_nM > 0
    # The interval must not be presented as if the four points were independent.
    assert "not independent" in fit.interval_caveat
    assert "week 1 and week 4" in fit.interval_caveat


def test_total_and_free_bases_are_never_pooled():
    """Pooling them would fit the protein binding rather than the receptor."""
    total = fit_plasma_ec50(plasma_basis="total")
    free = fit_plasma_ec50(plasma_basis="free")
    assert set(total.sources).isdisjoint(free.sources)
    assert free.sources == ("videbaek2001",)
    # About an order of magnitude apart, which is the free fraction, not biology.
    assert total.ec50_nM / free.ec50_nM > 5
    with pytest.raises(ValueError):
        fit_plasma_ec50(plasma_basis="unbound")


def test_free_basis_fit_matches_the_papers_own_kd_values():
    """Cross-check against a number Videbaek 2001 printed itself."""
    free = fit_plasma_ec50(plasma_basis="free")
    printed_kd = (0.16, 0.24, 0.19, 0.39)
    assert free.ec50_nM == pytest.approx(
        sum(printed_kd) / len(printed_kd), abs=0.06
    )


def test_comparison_against_published_relations_is_a_ratio_not_a_merge():
    fit = fit_plasma_ec50()
    comparisons = compare_with_reported_relations(fit)
    assert {c.label for c in comparisons} == {
        r.label for r in HALOPERIDOL_D2_REPORTED_RELATIONS
    }
    for comparison in comparisons:
        assert comparison.fitted_ec50_nM == fit.ec50_nM
        assert comparison.ratio_fitted_over_reported == pytest.approx(
            fit.ec50_nM / comparison.reported_ec50_nM
        )
    by_label = {c.label: c for c in comparisons}
    assert by_label["kapur1997"].ratio_fitted_over_reported == pytest.approx(1.389, abs=0.01)
    assert by_label["fitzgerald2000"].ratio_fitted_over_reported == pytest.approx(1.771, abs=0.01)


def test_neither_published_constant_falls_inside_our_interval():
    """A disagreement the downstream tasks have to carry, recorded as a test.

    Both published constants sit below the fitted interval, so no single
    hyperbola in this repository reproduces the oral raclopride literature and
    the depot means at once.
    """
    comparisons = compare_with_reported_relations()
    assert all(c.inside_our_interval is False for c in comparisons)


def test_landscape_puts_every_constant_on_one_basis_and_names_the_conversion():
    rows = ec50_landscape()
    labels = [row[0] for row in rows]
    assert any("this repository" in label for label in labels)
    assert any("videbaek2001" in label for label in labels)
    values = {row[0]: row[1] for row in rows}
    videbaek = next(v for k, v in values.items() if "videbaek" in k)
    free = fit_plasma_ec50(plasma_basis="free")
    assert videbaek == pytest.approx(free.ec50_nM / HALOPERIDOL_PLASMA_FREE_FRACTION)
    # The converted row says so in its own note rather than blending in.
    note = next(row[2] for row in rows if "videbaek" in row[0])
    assert "converted" in note
    # The tracer spread that P19.3 recorded survives into this table.
    assert max(values.values()) / min(values.values()) > 2


def test_degenerate_occupancies_are_refused_rather_than_fitted():
    """0 and 1 imply an infinite or zero EC50 and would dominate any fit."""
    from neurocascade.human.d2_ec50 import _fit_one

    with pytest.raises(ValueError):
        _fit_one([1.0, 2.0], [0.0, 0.5])
    with pytest.raises(ValueError):
        _fit_one([1.0, 2.0], [1.0, 0.5])
    with pytest.raises(ValueError):
        _fit_one([0.0, 2.0], [0.3, 0.5])


def test_fit_is_the_least_squares_optimum():
    """Guards the bisection against silently returning a non-optimal point."""
    fit = fit_plasma_ec50()
    points = [
        item
        for item in occupancy_observations()
        if item.plasma_concentration_basis == "total"
        and item.plasma_concentration_nM is not None
    ]

    def rss(ec50):
        return sum(
            (
                item.observed_occupancy
                - item.plasma_concentration_nM / (item.plasma_concentration_nM + ec50)
            )
            ** 2
            for item in points
        )

    best = rss(fit.ec50_nM)
    for offset in (-0.2, -0.05, -0.01, 0.01, 0.05, 0.2):
        assert rss(fit.ec50_nM + offset) >= best


def test_dataset_ids_cover_every_registered_fixture():
    assert HALOPERIDOL_D2_DATASET_IDS == tuple(sorted(HALOPERIDOL_D2_DATASET_IDS))
    manifests = haloperidol_d2_manifests()
    assert len(manifests) == len(HALOPERIDOL_D2_DATASET_IDS)
    for manifest, dataset_id in zip(manifests, HALOPERIDOL_D2_DATASET_IDS):
        assert manifest.dataset_id.endswith(dataset_id)
    # Every id resolves to at least one registered observation, so grouping by
    # id cannot silently drop a study.
    for dataset_id in HALOPERIDOL_D2_DATASET_IDS:
        assert haloperidol_d2_observations(dataset_id)


def test_occupancy_at_rejects_negative_concentrations():
    fit = fit_plasma_ec50()
    assert fit.occupancy_at(0.0) == 0.0
    assert math.isclose(fit.occupancy_at(fit.ec50_nM), 0.5)
    with pytest.raises(ValueError):
        fit.occupancy_at(-1.0)
