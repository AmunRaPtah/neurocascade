"""Human GABA-A operational agonism model (P16.3, gated P19.1).

P19.1 gate: ``HUMAN_GABAA_OPERATIONAL_PRIOR`` is ILLUSTRATIVE.  ``effect()``
raises unless the caller passes ``allow_illustrative=True``; results computed
with it carry ``provenance="illustrative"``.  A model constructed with an
explicit, sourced ``reserve`` (status ``"measured"``) is not gated.
"""
from dataclasses import dataclass

from .ke0 import GatedFloat


@dataclass(frozen=True)
class GABAAOperationalModel:
    """Map occupancy to fractional GABA-A inhibitory effect.

    ``efficacy`` is intrinsic agonist efficacy; ``reserve`` is an explicit
    operational amplification factor. The result is bounded in [0, emax].
    ``status``: ``"illustrative"`` (gated) or ``"measured"`` (ungated).
    """

    reserve: float = 1.0
    emax: float = 1.0
    baseline: float = 0.0
    source: str = "human GABA-A operational prior; not drug-specific"
    status: str = ""

    def __post_init__(self):
        if self.reserve <= 0:
            raise ValueError("reserve must be positive")
        if self.emax < 0 or not 0 <= self.baseline <= self.emax:
            raise ValueError("baseline must lie between zero and emax")
        # An explicitly-constructed model (caller chose the parameters, e.g. a
        # test or a spec) is ungated; only the SHARED prior constant is
        # illustrative. Empty status means "derive from the source string".
        if not self.status:
            object.__setattr__(
                self,
                "status",
                "illustrative"
                if self.source.startswith("illustrative")
                else "measured",
            )

    @property
    def provenance(self) -> str:
        return "illustrative" if self.status == "illustrative" else "measured"

    def effect(
        self,
        occupancy: float,
        efficacy: float = 1.0,
        allow_illustrative: bool = False,
    ) -> GatedFloat:
        if self.status == "illustrative" and not allow_illustrative:
            raise ValueError(
                "GABA-A operational model is ILLUSTRATIVE (reserve=%.2f is "
                "not a measured human value); consuming it requires "
                "allow_illustrative=True" % self.reserve
            )
        if not 0 <= occupancy <= 1:
            raise ValueError("occupancy must be between 0 and 1")
        if not 0 <= efficacy <= 1:
            raise ValueError("efficacy must be between 0 and 1")
        drive = occupancy * efficacy
        operational = self.reserve * drive / (1 + (self.reserve - 1) * drive)
        value = self.baseline + (self.emax - self.baseline) * operational
        return GatedFloat(value, self.provenance)


HUMAN_GABAA_OPERATIONAL_PRIOR = GABAAOperationalModel(
    reserve=1.0,
    source="illustrative human GABA-A operational prior; matched human efficacy pending",
    status="illustrative",
)
