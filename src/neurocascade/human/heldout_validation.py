"""Held-out human-drug validation contracts (P18.2)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HeldOutHumanDrugResult:
    compound: str
    endpoint: str
    status: str
    source: str | None = None
    predicted: float | None = None
    observed: float | None = None
    note: str = ""

    def __post_init__(self):
        if not self.compound or not self.endpoint:
            raise ValueError("compound and endpoint are required")
        if self.status not in ("scored", "unavailable", "failed"):
            raise ValueError("invalid validation status")
        if self.status == "scored" and (self.predicted is None or self.observed is None):
            raise ValueError("scored results require predicted and observed values")

    @property
    def error(self):
        if self.predicted is None or self.observed is None:
            return None
        return self.predicted - self.observed


def summarize_heldout(results):
    results = tuple(results)
    if not results:
        raise ValueError("at least one held-out result is required")
    scored = tuple(r for r in results if r.status == "scored")
    return {
        "total": len(results),
        "scored": len(scored),
        "unavailable": sum(r.status == "unavailable" for r in results),
        "failed": sum(r.status == "failed" for r in results),
        "mean_absolute_error": (
            sum(abs(r.error) for r in scored) / len(scored) if scored else None
        ),
    }
