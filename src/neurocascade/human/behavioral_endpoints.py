"""Human behavioral endpoint contracts (P17.1).

Endpoints describe human-observable outcomes and their direction. They do not
claim fitted magnitudes; weights must be supplied by a later calibration step.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanEndpoint:
    name: str
    measure: str
    population: str
    status: str
    notes: str


HUMAN_ENDPOINTS = (
    HumanEndpoint("cognitive_performance", "reaction time / accuracy", "adult human", "available-readout", "Use task-specific normative data."),
    HumanEndpoint("motor_activity", "actigraphy or psychomotor task", "adult human", "available-readout", "Do not equate with rodent locomotion."),
    HumanEndpoint("subjective_sedation", "validated sedation scale", "adult human", "available-readout", "Requires human dose-response calibration."),
    HumanEndpoint("anxiety_state", "validated self-report and physiological scale", "adult human", "available-readout", "Not interchangeable with rodent elevated-plus-maze behavior."),
)


def human_endpoints(name=None):
    if name is None:
        return HUMAN_ENDPOINTS
    return tuple(endpoint for endpoint in HUMAN_ENDPOINTS if endpoint.name == name)


def endpoint_score(endpoint_name, node_values, weights):
    """Compute a transparent score; ``weights`` are caller-supplied."""
    endpoint = human_endpoints(endpoint_name)
    if not endpoint:
        raise ValueError("unknown human endpoint %r" % endpoint_name)
    if not weights:
        raise ValueError("weights are required; no human endpoint magnitudes are fitted")
    return sum(float(weights[node]) * float(node_values[node]) for node in weights)
