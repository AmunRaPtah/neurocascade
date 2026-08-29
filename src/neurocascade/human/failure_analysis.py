"""Layer-by-layer human prediction failure analysis (P18.3)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerFinding:
    layer: str
    status: str
    limitation: str
    required_repair: str

    def __post_init__(self):
        if self.status not in ("ready", "partial", "blocked"):
            raise ValueError("status must be ready, partial, or blocked")
        if not self.layer or not self.limitation or not self.required_repair:
            raise ValueError("layer, limitation, and required_repair are required")


HUMAN_LAYER_FINDINGS = (
    LayerFinding("PK", "partial", "Some human drug parameter sets remain incomplete.", "Add matched human compartmental PK and validate curves."),
    LayerFinding("BBB partition", "partial", "Human Kp,uu coverage is compound-specific.", "Measure or validate unbound brain partition for each compound."),
    LayerFinding("Receptor binding", "partial", "Several human receptor parameters remain unavailable.", "Add matched human affinity and efficacy measurements."),
    LayerFinding("Connectivity", "partial", "Most usable directed edges are inferred rather than measured.", "Expand human directed connectivity and preserve provenance."),
    LayerFinding("Effect transduction", "partial", "Operational priors are not fitted to human endpoints.", "Fit receptor-to-effect models on human functional data."),
    LayerFinding("Behavioral validation", "blocked", "No complete held-out arbitrary-drug human behavioral chain is established.", "Collect independent human dose-response and held-out validation data."),
)


def human_failure_analysis():
    return HUMAN_LAYER_FINDINGS


def blocking_findings():
    return tuple(f for f in HUMAN_LAYER_FINDINGS if f.status == "blocked")
