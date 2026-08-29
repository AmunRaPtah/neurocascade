"""Human PET occupancy validation contracts (P18.1)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PETOccupancyObservation:
    compound: str
    target: str
    dose: float
    observed_occupancy: float
    source: str
    observed_sd: float | None = None

    def __post_init__(self):
        if self.dose <= 0:
            raise ValueError("dose must be positive")
        if not 0 <= self.observed_occupancy <= 1:
            raise ValueError("observed_occupancy must be in [0, 1]")
        if not self.compound or not self.target or not self.source:
            raise ValueError("compound, target, and source are required")
        if self.observed_sd is not None and self.observed_sd <= 0:
            raise ValueError("observed_sd must be positive")


@dataclass(frozen=True)
class PETOccupancyComparison:
    observation: PETOccupancyObservation
    predicted_occupancy: float

    @property
    def error(self):
        return self.predicted_occupancy - self.observation.observed_occupancy

    @property
    def absolute_error(self):
        return abs(self.error)

    @property
    def standardized_error(self):
        sd = self.observation.observed_sd
        if sd is None:
            return None
        return self.error / sd


def compare_pet_occupancy(observation, predicted_occupancy):
    if not 0 <= predicted_occupancy <= 1:
        raise ValueError("predicted_occupancy must be in [0, 1]")
    return PETOccupancyComparison(observation, float(predicted_occupancy))
