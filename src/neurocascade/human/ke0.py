"""Human effector-specific effect-site equilibration (P17.2)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanKe0:
    """Effect-site equilibration rate in inverse minutes.

    ``ke0`` is kept separate from systemic elimination: it describes delay
    between brain exposure and an observable effector, not drug clearance.
    """

    effector: str
    ke0_per_min: float
    source: str
    status: str = "illustrative"

    def __post_init__(self):
        if not self.effector:
            raise ValueError("effector is required")
        if self.ke0_per_min <= 0:
            raise ValueError("ke0_per_min must be positive")

    @property
    def half_life_min(self):
        from math import log
        return log(2.0) / self.ke0_per_min

    def effect_site_rhs(self, effect_site: float, brain_concentration: float) -> float:
        return self.ke0_per_min * (float(brain_concentration) - float(effect_site))


HUMAN_KE0_RECORDS = (
    HumanKe0("cognitive_performance", 0.10, "human effector prior; fit required", "illustrative"),
    HumanKe0("motor_activity", 0.08, "human effector prior; fit required", "illustrative"),
    HumanKe0("subjective_sedation", 0.05, "human effector prior; fit required", "illustrative"),
    HumanKe0("anxiety_state", 0.04, "human effector prior; fit required", "illustrative"),
)


def human_ke0(effector: str) -> HumanKe0:
    for record in HUMAN_KE0_RECORDS:
        if record.effector == effector:
            return record
    raise ValueError("unknown human effector %r" % effector)
