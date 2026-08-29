"""D2 occupancy-limited effect model (P16.1)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class D2Effect:
    occupancy: float
    baseline_occupancy: float = 0.0
    emax: float = 1.0
    direction: float = -1.0

    def __post_init__(self):
        for name, value in (("occupancy", self.occupancy), ("baseline_occupancy", self.baseline_occupancy)):
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)
        if self.emax < 0.0:
            raise ValueError("emax must be non-negative")
        if self.direction not in (-1.0, 1.0):
            raise ValueError("direction must be -1 or 1")

    @property
    def effect(self):
        """Drug-induced effect relative to baseline, bounded by Emax."""
        return self.direction * self.emax * max(0.0, self.occupancy - self.baseline_occupancy)

    @property
    def available_reserve(self):
        return max(0.0, 1.0 - self.baseline_occupancy)


def d2_effect_from_occupancy(occupancy, baseline_occupancy=0.0, emax=1.0):
    return D2Effect(occupancy=float(occupancy), baseline_occupancy=float(baseline_occupancy), emax=float(emax))
