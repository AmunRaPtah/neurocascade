"""Human MOR reserve/amplification model (P16.2).

This is an effect-transducer layer, not a claim that receptor reserve is a
single universal human constant.  It converts occupancy into response using
an explicit reserve factor and Emax, with validation and provenance-friendly
metadata.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MORReserveModel:
    """Operational MOR response model.

    ``reserve`` is dimensionless and must be positive. Values above one allow
    receptor reserve/amplification; response is still bounded by ``emax``.
    ``baseline`` is the response at zero drug effect.
    """

    reserve: float = 1.0
    emax: float = 1.0
    baseline: float = 0.0
    source: str = "human operational reserve prior; not yet drug-specific"

    def __post_init__(self):
        if self.reserve <= 0:
            raise ValueError("reserve must be positive")
        if self.emax < 0:
            raise ValueError("emax must be non-negative")
        if not 0 <= self.baseline <= self.emax:
            raise ValueError("baseline must lie between zero and emax")

    def response(self, occupancy: float, efficacy: float = 1.0) -> float:
        """Return bounded response for fractional occupancy and efficacy."""
        if not 0 <= occupancy <= 1:
            raise ValueError("occupancy must be between 0 and 1")
        if not 0 <= efficacy <= 1:
            raise ValueError("efficacy must be between 0 and 1")
        drive = occupancy * efficacy
        amplified = self.reserve * drive / (1.0 + (self.reserve - 1.0) * drive)
        return self.baseline + (self.emax - self.baseline) * amplified

    def gain_over_baseline(self, occupancy: float, efficacy: float = 1.0) -> float:
        return self.response(occupancy, efficacy) - self.baseline


HUMAN_MOR_RESERVE_PRIOR = MORReserveModel(
    reserve=1.5,
    emax=1.0,
    source="illustrative human MOR reserve prior; replace with matched human assay",
)
