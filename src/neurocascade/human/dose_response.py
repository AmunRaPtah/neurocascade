"""Evidence-gated human dose-response fitting (P17.3)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HumanDoseResponseObservation:
    dose: float
    response: float
    endpoint: str
    source: str
    response_sd: float | None = None

    def __post_init__(self):
        if self.dose <= 0 or not math.isfinite(self.dose):
            raise ValueError("dose must be positive and finite")
        if not math.isfinite(self.response):
            raise ValueError("response must be finite")
        if not self.endpoint or not self.source:
            raise ValueError("endpoint and source are required")
        if self.response_sd is not None and self.response_sd <= 0:
            raise ValueError("response_sd must be positive")


@dataclass(frozen=True)
class HumanDoseResponseFit:
    endpoint: str
    baseline: float
    slope_per_log_dose: float
    n_observations: int
    source: str
    status: str = "fitted-human"

    def predict(self, dose: float) -> float:
        if dose <= 0:
            raise ValueError("dose must be positive")
        return self.baseline + self.slope_per_log_dose * math.log(dose)


def fit_human_log_linear(observations, *, min_observations=3):
    """Fit a transparent log-dose response, requiring human observations.

    This intentionally does not provide a default fit or accept species labels
    as a substitute for provenance. It is a small exploratory fitter; final
    nonlinear endpoint models remain a later validation task.
    """
    observations = tuple(observations)
    if len(observations) < min_observations:
        raise ValueError("insufficient human observations for fitting")
    endpoints = {o.endpoint for o in observations}
    sources = {o.source for o in observations}
    if len(endpoints) != 1:
        raise ValueError("all observations must describe one endpoint")
    if len(sources) != 1:
        raise ValueError("all observations must share one provenance source")
    xs = [math.log(o.dose) for o in observations]
    ys = [o.response for o in observations]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("dose values must vary to identify a dose response")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    baseline = mean_y - slope * mean_x
    return HumanDoseResponseFit(observations[0].endpoint, baseline, slope, len(observations), observations[0].source)
