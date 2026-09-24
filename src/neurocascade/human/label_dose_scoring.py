"""P22.2 label-dose scoring functions.

Pure functions that compare predicted occupancy target dose ranges (d65, d78)
against FDA label ranges (l_low, l_high). No PK, no drug data, no I/O: this
module only evaluates overlap and log-midpoint error.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Any

# Metric constants from the pre-registration (C5: doses are reported to 3 sig figs)
LOG2 = math.log10(2)  # Twofold boundary for passing
SIG_FIGS = 3  # Number of significant figures for dose rounding


def round_sig(x: float, sig: int = SIG_FIGS) -> float:
    """Round to `sig` significant figures."""
    if x == 0:
        return 0.0
    # round() to a decimal place, not digits * 10**order: the multiplication
    # leaves artifacts like 4.000000000000001, and C5 evaluates the
    # touching-endpoint boundary at exactly the printed value.
    return round(x, sig - 1 - math.floor(math.log10(abs(x))))


@dataclass(frozen=True)
class Score:
    """Result of scoring a drug against its label dose range.

    Attributes:
        drug: Name of the drug being scored.
        overlap: Whether the predicted dose window intersects (including touching)
            with the label dose range.
        e: Log midpoint error, or None if undefined (unreachable windows).
        passed: Whether the drug meets both overlap and twofold criteria.
        refusal: If non-None, indicates why scoring was refused (layer name).
    """

    drug: str
    overlap: bool
    e: float | None
    passed: bool
    refusal: str | None


def score(
    drug: str,
    d65: float | None,
    d78: float | None,
    l_low: float,
    l_high: float,
) -> Score:
    """Score a drug's predicted occupancy window against its label range.

    Args:
        drug: Name of the drug.
        d65: Predicted daily dose at 65% occupancy (or None if unreachable).
        d78: Predicted daily dose at 78% occupancy (or None if unreachable).
        l_low: Label's lower dose bound (non-negative).
        l_high: Label's upper dose bound (positive).

    Returns:
        Score object with overlap, e, passed, and refusal fields.

    Raises:
        ValueError: If any dose <= 0, if d65 > d78, or if l_low > l_high.
    """
    # Validate inputs
    if d65 is not None and d65 <= 0:
        raise ValueError(f"D65 must be > 0, got {d65}")
    if d78 is not None and d78 <= 0:
        raise ValueError(f"D78 must be > 0, got {d78}")
    if d65 is not None and d78 is not None and d65 > d78:
        raise ValueError(f"D65 ({d65}) cannot exceed D78 ({d78})")
    if l_low <= 0:
        raise ValueError(f"L_low must be > 0, got {l_low}")
    if l_high < l_low:
        raise ValueError(f"L_low ({l_low}) cannot exceed L_high ({l_high})")

    # Round doses to 3 sig figs (C5)
    dr65 = round_sig(d65) if d65 is not None else None
    dr78 = round_sig(d78) if d78 is not None else None
    l_lowr = round_sig(l_low)
    l_highr = round_sig(l_high)

    # Check for unreachable occupancy window
    if d65 is None or d78 is None:
        # Unreachable: scored failure, not a refusal
        return Score(
            drug=drug,
            overlap=False,
            e=None,
            passed=False,
            refusal=None,
        )

    # Check for closed-interval overlap (touching at endpoint counts)
    overlap = not (dr78 < l_lowr or dr65 > l_highr)

    # Compute log midpoint error
    # e = log10( sqrt(d65*d78) / sqrt(l_low*l_high) )
    geom_pred = math.sqrt(dr65 * dr78)
    geom_label = math.sqrt(l_lowr * l_highr)
    e = math.log10(geom_pred / geom_label)

    # Twofold pass criteria with 1e-12 tolerance
    passed = overlap and abs(e) <= LOG2 + 1e-12

    return Score(
        drug=drug,
        overlap=overlap,
        e=e,
        passed=passed,
        refusal=None,
    )


def refused(drug: str, layer: str) -> Score:
    """Create a refusal score for a drug.

    Args:
        drug: Name of the drug.
        layer: Non-empty reason layer (e.g., "pk", "receptor: no ...").

    Returns:
        Refusal Score with overlap=False, e=None, passed=False.

    Raises:
        ValueError: If `layer` is empty.
    """
    if not layer:
        raise ValueError("Refusal layer must be non-empty")
    return Score(
        drug=drug,
        overlap=False,
        e=None,
        passed=False,
        refusal=layer,
    )


def aggregate(scores: list[Score], denominator: int = 8) -> dict[str, Any]:
    """Aggregate per-drug scores into summary statistics.

    Args:
        scores: List of Score objects.
        denominator: Fixed denominator (8) of Phase 22.

    Returns:
        Dictionary with keys:
            - n_pass: Number of drugs that passed (overlap=True and |e|<=log10(2))
            - n_scored: Number of scored drugs (refusal is None)
            - denominator: The fixed denominator
            - median_abs_e: Median of |e| values (or None if no e values)
            - max_abs_e: Max of |e| values (or None if no e values)
            - per_drug: Dictionary mapping every drug name to its e (None when
              refused or unreachable).

    Raises:
        ValueError: If drug names repeat or total drugs exceed the denominator.
    """
    names = [s.drug for s in scores]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate drug names in scores: {names}")
    if len(scores) > denominator:
        raise ValueError(f"{len(scores)} scores for a denominator of {denominator}")

    abs_e = [abs(s.e) for s in scores if s.e is not None]
    return {
        "n_pass": sum(s.passed for s in scores),
        # Unreachable windows are scored failures, so they count here.
        "n_scored": sum(s.refusal is None for s in scores),
        "denominator": denominator,
        "median_abs_e": statistics.median(abs_e) if abs_e else None,
        "max_abs_e": max(abs_e) if abs_e else None,
        "per_drug": {s.drug: s.e for s in scores},
    }
