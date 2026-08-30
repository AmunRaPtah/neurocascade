"""Human MOR reserve/amplification model (P16.2, gated P19.1).

This is an effect-transducer layer, not a claim that receptor reserve is a
single universal human constant.  It converts occupancy into response using
an explicit reserve factor and Emax, with validation and provenance-friendly
metadata.

P19.1 gate: ``HUMAN_MOR_RESERVE_PRIOR`` is ILLUSTRATIVE.  ``response()`` and
``gain_over_baseline()`` raise unless the caller passes
``allow_illustrative=True``; results computed with it carry
``provenance="illustrative"``.  A model constructed with an explicit,
sourced ``reserve`` (status ``"measured"``) is not gated.
"""
from dataclasses import dataclass

from .ke0 import GatedFloat


@dataclass(frozen=True)
class MORReserveModel:
    """Operational MOR response model.

    ``reserve`` is dimensionless and must be positive. Values above one allow
    receptor reserve/amplification; response is still bounded by ``emax``.
    ``baseline`` is the response at zero drug effect.
    ``status``: ``"illustrative"`` (gated) or ``"measured"`` (ungated).
    """

    reserve: float = 1.0
    emax: float = 1.0
    baseline: float = 0.0
    source: str = "human operational reserve prior; not yet drug-specific"
    status: str = ""

    def __post_init__(self):
        if self.reserve <= 0:
            raise ValueError("reserve must be positive")
        if self.emax < 0:
            raise ValueError("emax must be non-negative")
        if not 0 <= self.baseline <= self.emax:
            raise ValueError("baseline must lie between zero and emax")
        # Only the SHARED prior constant is illustrative; an explicitly
        # constructed model (caller chose the parameters) is ungated.
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

    def _gate(self, allow_illustrative: bool) -> None:
        if self.status == "illustrative" and not allow_illustrative:
            raise ValueError(
                "MOR reserve model is ILLUSTRATIVE (reserve=%.2f is not a "
                "measured human value); consuming it requires "
                "allow_illustrative=True" % self.reserve
            )

    def response(
        self,
        occupancy: float,
        efficacy: float = 1.0,
        allow_illustrative: bool = False,
    ) -> GatedFloat:
        """Return bounded response for fractional occupancy and efficacy."""
        self._gate(allow_illustrative)
        if not 0 <= occupancy <= 1:
            raise ValueError("occupancy must be between 0 and 1")
        if not 0 <= efficacy <= 1:
            raise ValueError("efficacy must be between 0 and 1")
        drive = occupancy * efficacy
        amplified = self.reserve * drive / (1.0 + (self.reserve - 1.0) * drive)
        value = self.baseline + (self.emax - self.baseline) * amplified
        return GatedFloat(value, self.provenance)

    def gain_over_baseline(
        self,
        occupancy: float,
        efficacy: float = 1.0,
        allow_illustrative: bool = False,
    ) -> GatedFloat:
        return self.response(occupancy, efficacy, allow_illustrative) - self.baseline


HUMAN_MOR_RESERVE_PRIOR = MORReserveModel(
    reserve=1.5,
    emax=1.0,
    source="illustrative human MOR reserve prior; replace with matched human assay",
    status="illustrative",
)
