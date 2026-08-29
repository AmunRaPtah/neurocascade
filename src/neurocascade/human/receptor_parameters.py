"""Human-first receptor parameter registry for P15.1-P15.4."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanReceptorParameter:
    target: str
    species: str
    ki_nM: float | None
    efficacy: float | None
    evidence_context: str
    citation: str
    status: str
    notes: str

    @property
    def usable(self):
        return self.species == "human" and self.ki_nM is not None and self.efficacy is not None


HUMAN_RECEPTOR_PARAMETERS = (
    HumanReceptorParameter("D2", "human", None, None, "in-vivo PET", "Human PET occupancy calibration reserved for P16.1", "pending-human-occupancy", "No rodent affinity is promoted."),
    HumanReceptorParameter("MOR", "human", None, None, "in-vivo PET", "Human PET occupancy calibration pending", "pending-human-occupancy", "No rodent affinity is promoted."),
    HumanReceptorParameter("GABA_A_BZD", "human", None, None, "in-vivo PET", "Human benzodiazepine occupancy calibration pending", "pending-human-occupancy", "No rodent affinity is promoted."),
    HumanReceptorParameter("D1", "human", None, None, "in-vitro", "Human functional affinity source not yet validated", "pending-human-source", "In-vitro values must not be treated as in-vivo occupancy parameters without an explicit correction."),
    HumanReceptorParameter("5HT2A", "human", None, None, "in-vitro", "Human functional affinity source not yet validated", "pending-human-source", "No isolated rodent rate constant is promoted."),
    HumanReceptorParameter("H1", "human", None, None, "in-vitro", "Human functional affinity source not yet validated", "pending-human-source", "Context correction remains explicit and optional."),
)


def human_receptor_parameters(target=None, evidence_context=None):
    records = HUMAN_RECEPTOR_PARAMETERS
    if target is not None:
        records = tuple(r for r in records if r.target == target)
    if evidence_context is not None:
        records = tuple(r for r in records if r.evidence_context == evidence_context)
    return records


def require_human_parameter(target):
    records = human_receptor_parameters(target)
    if not records or not records[0].usable:
        raise ValueError("no validated human receptor parameter for %s; rodent fallback is disabled" % target)
    return records[0]
