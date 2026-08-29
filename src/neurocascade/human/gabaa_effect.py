"""Human GABA-A operational agonism model (P16.3)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GABAAOperationalModel:
    """Map occupancy to fractional GABA-A inhibitory effect.

    ``efficacy`` is intrinsic agonist efficacy; ``reserve`` is an explicit
    operational amplification factor. The result is bounded in [0, emax].
    """

    reserve: float = 1.0
    emax: float = 1.0
    baseline: float = 0.0
    source: str = "human GABA-A operational prior; not drug-specific"

    def __post_init__(self):
        if self.reserve <= 0:
            raise ValueError("reserve must be positive")
        if self.emax < 0 or not 0 <= self.baseline <= self.emax:
            raise ValueError("baseline must lie between zero and emax")

    def effect(self, occupancy: float, efficacy: float = 1.0) -> float:
        if not 0 <= occupancy <= 1:
            raise ValueError("occupancy must be between 0 and 1")
        if not 0 <= efficacy <= 1:
            raise ValueError("efficacy must be between 0 and 1")
        drive = occupancy * efficacy
        operational = self.reserve * drive / (1 + (self.reserve - 1) * drive)
        return self.baseline + (self.emax - self.baseline) * operational


HUMAN_GABAA_OPERATIONAL_PRIOR = GABAAOperationalModel(
    reserve=1.0,
    source="illustrative human GABA-A operational prior; matched human efficacy pending",
)
