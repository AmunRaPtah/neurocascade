import pytest

from neurocascade.human.pet_validation import (
    HALOPERIDOL_MW_G_PER_MOL,
    PETOccupancyObservation,
    compare_pet_occupancy,
    ng_per_mL_to_nM,
    observations_by_tracer,
    require_single_tracer,
)


def _observation(**overrides):
    """A minimal well-formed observation, overridable field by field."""
    values = dict(
        compound="haloperidol",
        target="D2",
        dose=2.0,
        observed_occupancy=0.6,
        source="source",
        dose_regimen="steady-state-daily",
    )
    values.update(overrides)
    return PETOccupancyObservation(**values)


def test_pet_comparison_reports_raw_and_standardized_error():
    observation = _observation(observed_occupancy=0.6, observed_sd=0.1)
    comparison = compare_pet_occupancy(observation, 0.8)
    assert comparison.error == pytest.approx(0.2)
    assert comparison.absolute_error == pytest.approx(0.2)
    assert comparison.standardized_error == pytest.approx(2.0)


def test_pet_observations_and_predictions_are_bounded():
    with pytest.raises(ValueError):
        _observation(observed_occupancy=1.1)
    with pytest.raises(ValueError):
        compare_pet_occupancy(_observation(observed_occupancy=0.2), -0.1)


def test_a_dose_without_a_regimen_is_not_an_exposure():
    with pytest.raises(ValueError, match="dose_regimen"):
        PETOccupancyObservation("haloperidol", "D2", 2.0, 0.6, "source")
    with pytest.raises(ValueError, match="unrecognised dose_regimen"):
        _observation(dose_regimen="whenever")


def test_a_digitised_value_must_name_its_figure():
    with pytest.raises(ValueError, match="figure"):
        _observation(value_origin="digitised")
    assert _observation(value_origin="digitised", source_locator="Figure 3").value_origin


def test_a_plasma_concentration_carries_its_printed_form_and_its_basis():
    with pytest.raises(ValueError, match="printed form"):
        _observation(plasma_concentration_nM=1.0)
    with pytest.raises(ValueError, match="total or free"):
        _observation(plasma_concentration_nM=1.0, plasma_concentration_as_printed="1 nM")
    point = _observation(
        plasma_concentration_nM=1.0,
        plasma_concentration_as_printed="1 nM",
        plasma_concentration_basis="free",
    )
    assert point.plasma_concentration_basis == "free"
    with pytest.raises(ValueError, match="total or free"):
        _observation(
            plasma_concentration_nM=1.0,
            plasma_concentration_as_printed="1 nM",
            plasma_concentration_basis="unbound-ish",
        )


def test_a_range_only_point_carries_no_invented_central_value():
    point = _observation(
        observed_occupancy=None, observed_range_low=0.53, observed_range_high=0.74
    )
    assert point.is_interval_only
    assert point.observed_occupancy_percent is None
    assert point.contains(0.6)
    assert not point.contains(0.4)
    # The interval test is the only one such a point supports. Differencing it
    # would require a mean the source never printed.
    with pytest.raises(ValueError, match="range"):
        compare_pet_occupancy(point, 0.6)


def test_a_range_must_be_complete_ordered_and_consistent_with_its_mean():
    with pytest.raises(ValueError, match="both bounds"):
        _observation(observed_occupancy=None, observed_range_low=0.5)
    with pytest.raises(ValueError, match="must not exceed"):
        _observation(observed_range_low=0.8, observed_range_high=0.5)
    with pytest.raises(ValueError, match="outside its own reported range"):
        _observation(observed_occupancy=0.9, observed_range_low=0.5, observed_range_high=0.8)
    with pytest.raises(ValueError, match="value or a range"):
        _observation(observed_occupancy=None)


def test_a_binding_index_is_not_an_occupancy():
    binding_index = _observation(quantity="binding-index")
    with pytest.raises(ValueError, match="did not report an occupancy"):
        compare_pet_occupancy(binding_index, 0.7)
    availability = _observation(quantity="availability-ratio")
    with pytest.raises(ValueError, match="did not report an occupancy"):
        compare_pet_occupancy(availability, 0.7)
    with pytest.raises(ValueError, match="quantity must be one of"):
        _observation(quantity="something-else")


def test_occupancy_from_different_tracers_is_never_pooled_silently():
    raclopride = _observation(tracer="[11C]raclopride")
    ibzm = _observation(tracer="[123I]IBZM")
    grouped = observations_by_tracer([raclopride, ibzm])
    assert sorted(grouped) == ["[11C]raclopride", "[123I]IBZM"]
    with pytest.raises(ValueError, match="must not be pooled"):
        require_single_tracer([raclopride, ibzm])
    assert require_single_tracer([raclopride]) == "[11C]raclopride"
    with pytest.raises(ValueError, match="do not name a tracer"):
        observations_by_tracer([raclopride, _observation()])


def test_ng_per_mL_converts_to_nM_at_the_haloperidol_molar_mass():
    assert HALOPERIDOL_MW_G_PER_MOL == pytest.approx(375.86)
    # Kapur 1997's reported EC50 of 0.51 ng/ml.
    assert ng_per_mL_to_nM(0.51) == pytest.approx(1000.0 * 0.51 / 375.86)
    assert ng_per_mL_to_nM(0.51) == pytest.approx(1.357, abs=1e-3)


def test_is_measured_exposure_needs_a_dose_or_a_concentration():
    assert _observation().is_measured_exposure
    assert not _observation(dose=None).is_measured_exposure
    assert _observation(
        dose=None,
        plasma_concentration_nM=2.0,
        plasma_concentration_as_printed="2 nmol/L",
        plasma_concentration_basis="total",
    ).is_measured_exposure
