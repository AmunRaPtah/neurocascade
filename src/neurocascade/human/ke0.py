"""Human effector-specific effect-site equilibration (P17.2, gated P19.1).

``ke0`` is kept separate from systemic elimination: it describes delay
between brain exposure and an observable effector, not drug clearance.

P19.1 gate: every record in ``HUMAN_KE0_RECORDS`` is ILLUSTRATIVE (a human
effector prior, not a fitted measurement).  Consuming one from a prediction
path requires the explicit ``allow_illustrative=True`` opt-in; otherwise
``human_ke0`` raises.  Results computed with an illustrative record carry
``provenance="illustrative"`` on the returned value.
"""
from dataclasses import dataclass


class GatedFloat(float):
    """A float that carries the provenance of the constants behind it.

    Behaves exactly like ``float`` (existing comparisons and arithmetic keep
    working) but exposes ``.provenance`` so a caller can never lose track of
    the fact that an illustrative constant fed the computation.
    """

    provenance: str = "measured"

    def __new__(cls, value, provenance="measured"):
        obj = super().__new__(cls, value)
        obj.provenance = provenance
        return obj


@dataclass(frozen=True)
class HumanKe0:
    """Effect-site equilibration rate in inverse minutes.

    ``status`` is ``"illustrative"`` until the record is fitted against real
    human effector data; ``human_ke0`` enforces the opt-in for such records.
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

    @property
    def provenance(self) -> str:
        return "illustrative" if self.status == "illustrative" else "measured"

    def effect_site_rhs(
        self,
        effect_site: float,
        brain_concentration: float,
        allow_illustrative: bool = False,
    ) -> GatedFloat:
        """d(effect_site)/dt toward brain concentration.

        Raises for an illustrative record unless ``allow_illustrative=True``;
        the returned value always carries ``.provenance``.
        """
        if self.status == "illustrative" and not allow_illustrative:
            raise ValueError(
                "ke0 record for %r is ILLUSTRATIVE (not fitted); consuming it "
                "requires allow_illustrative=True and the result carries "
                "provenance='illustrative'" % self.effector
            )
        value = self.ke0_per_min * (float(brain_concentration) - float(effect_site))
        return GatedFloat(value, self.provenance)


HUMAN_KE0_RECORDS = (
    HumanKe0("cognitive_performance", 0.10, "human effector prior; fit required", "illustrative"),
    HumanKe0("motor_activity", 0.08, "human effector prior; fit required", "illustrative"),
    HumanKe0("subjective_sedation", 0.05, "human effector prior; fit required", "illustrative"),
    HumanKe0("anxiety_state", 0.04, "human effector prior; fit required", "illustrative"),
)


def human_ke0(effector: str, allow_illustrative: bool = False) -> HumanKe0:
    """Return the ke0 record for ``effector``.

    Illustrative records (all of them today) raise unless the caller passes
    ``allow_illustrative=True`` — the explicit P19.1 opt-in that makes it
    impossible to consume an invented number by accident.
    """
    for record in HUMAN_KE0_RECORDS:
        if record.effector == effector:
            if record.status == "illustrative" and not allow_illustrative:
                raise ValueError(
                    "ke0 record for %r is ILLUSTRATIVE (not fitted); pass "
                    "allow_illustrative=True to consume it knowingly" % effector
                )
            return record
    raise ValueError("unknown human effector %r" % effector)
