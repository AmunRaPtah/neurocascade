"""Human endpoint readiness and retirement rules (P17.4)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointStatus:
    name: str
    status: str
    human_analogue: str | None
    reason: str


ENDPOINT_STATUS = (
    EndpointStatus("cognitive_performance", "retained", "reaction time / accuracy", "direct human behavioral analogue"),
    EndpointStatus("motor_activity", "retained", "actigraphy / psychomotor task", "human measure is available but requires endpoint-specific fitting"),
    EndpointStatus("subjective_sedation", "retained", "validated sedation scale", "direct human clinical readout"),
    EndpointStatus("anxiety_state", "retained", "validated anxiety scale", "human self-report and physiological measures are available"),
    EndpointStatus("elevated_plus_maze", "retired", None, "rodent-specific assay with no defensible direct human analogue"),
    EndpointStatus("open_field_locomotion", "retired", None, "rodent-specific assay; do not relabel as human motor activity"),
)


def endpoint_status(name: str) -> EndpointStatus:
    for item in ENDPOINT_STATUS:
        if item.name == name:
            return item
    raise ValueError("unknown endpoint %r" % name)


def require_human_endpoint(name: str) -> EndpointStatus:
    status = endpoint_status(name)
    if status.status == "retired":
        raise ValueError("endpoint %s is retired: %s" % (name, status.reason))
    return status
