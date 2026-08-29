import pytest

from neurocascade.human.pet_validation import (
    PETOccupancyObservation,
    compare_pet_occupancy,
)


def test_pet_comparison_reports_raw_and_standardized_error():
    observation = PETOccupancyObservation("fentanyl", "MOR", 10, 0.6, "human-PET", 0.1)
    comparison = compare_pet_occupancy(observation, 0.8)
    assert comparison.error == pytest.approx(0.2)
    assert comparison.absolute_error == pytest.approx(0.2)
    assert comparison.standardized_error == pytest.approx(2.0)


def test_pet_observations_and_predictions_are_bounded():
    with pytest.raises(ValueError):
        PETOccupancyObservation("x", "MOR", 1, 1.1, "source")
    with pytest.raises(ValueError):
        compare_pet_occupancy(PETOccupancyObservation("x", "MOR", 1, 0.2, "source"), -0.1)
