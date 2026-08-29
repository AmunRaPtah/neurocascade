"""Human E/I synaptic time constants from F-TRACT (P13.4).

The release supplies parcel maps, not a complete parameterization. This module
keeps measured finite values separate from the illustrative fallback and never
silently fills missing human observations.
"""

from dataclasses import dataclass
import math

from .provenance import MEASURED

EXCITATORY_KEY = "synaptic_excitatory_delay"
INHIBITORY_KEY = "synaptic_inhibitory_delay"


@dataclass(frozen=True)
class HumanEITimeConstants:
    labels: tuple
    excitatory: dict
    inhibitory: dict
    citation: str
    units: str
    fallback_tau: float = 1.0

    def __post_init__(self):
        labels = tuple(self.labels)
        if len(set(labels)) != len(labels):
            raise ValueError("labels contain duplicates")
        if self.fallback_tau <= 0:
            raise ValueError("fallback_tau must be positive")
        for name, values in (("excitatory", self.excitatory), ("inhibitory", self.inhibitory)):
            unknown = set(values) - set(labels)
            if unknown:
                raise ValueError("%s contains labels outside the atlas: %s" % (name, sorted(unknown)))

    def value(self, label, kind, *, allow_fallback=False):
        if label not in self.labels:
            raise KeyError(label)
        if kind not in ("excitatory", "inhibitory"):
            raise ValueError("kind must be 'excitatory' or 'inhibitory'")
        value = getattr(self, kind).get(label, float("nan"))
        if math.isfinite(value):
            return value
        if allow_fallback:
            return self.fallback_tau
        raise ValueError("no measured human %s time constant for %r" % (kind, label))

    def coverage(self, kind):
        values = getattr(self, kind)
        finite = sum(math.isfinite(values.get(label, float("nan"))) for label in self.labels)
        return {"n_labels": len(self.labels), "n_measured": finite, "fraction": finite / max(len(self.labels), 1), "provenance": MEASURED}

    def measured(self, label, kind):
        return math.isfinite(getattr(self, kind).get(label, float("nan")))


def from_delay_maps(labels, excitatory, inhibitory, *, citation, units, fallback_tau=1.0):
    """Construct a validated adapter from parsed F-TRACT parcel maps."""
    return HumanEITimeConstants(tuple(labels), dict(excitatory), dict(inhibitory), citation, units, fallback_tau)
