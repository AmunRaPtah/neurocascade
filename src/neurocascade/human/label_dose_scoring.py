"""P22.2 label-dose scoring functions.

Pure functions that compare predicted occupancy target dose ranges (d65, d78)
against FDA label ranges (l_low, l_high). No PK, no drug data, no I/O: this
module only evaluates overlap and log-midpoint error.
"""

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

# Metric constants from the pre-registration (C5: doses are reported to 3 sig figs)
LOG2 = math.log10(2)  # Twofold boundary for passing
SIG_FIGS = 3  # Number of significant figures for dose rounding


def round_sig(x: float, sig: int = SIG_FIGS) -> float:
    """Round to `sig` significant figures."""
    if x == 0:
        return 0.0  # Zero rounds to zero
    order = int(math.floor(math.log10(abs(x))))
    digits = x / 10 ** order
    rounded = round(digits, sig - 1)
    return rounded * (10 ** order)


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
            - per_drug: Dictionary mapping drug name to e values (None for refused)
                     or None if no e values.

    Raises:
        ValueError: If drug names repeat or total drugs exceed the denominator.
    """
    # Check for duplicate drug names first
    seen_drugs = set()
    for i, s in enumerate(scores):
        if s.drug in seen_drugs:
            raise ValueError(f"Duplicate drug name in scores: {s.drug}")
        seen_drugs.add(s.drug)

    # Check denominator constraint: total number of drugs cannot exceed denominator
    if len(scores) > denominator:
        raise ValueError(
            f"Cannot aggregate {len(scores)} drugs for denominator {denominator}"
        )

    # Count scored drugs (only those with refusal is None)
    n_scored = sum(1 for s in scores if s.refusal is None)

    # SUMMARY STATISTICS
    n_pass = sum(1 for s in scores if s.passed)
    n_scored = sum(1 for s in scores if s.refusal is None)

    # Build per_drug dict for all drugs:
    # - Scored drugs with non-None e: store the e value
    # - Scored drugs with e=None (unreachable): store None
    # - Refused drugs: store None
    # per_drug is None only when there are no scored/drugs (including refused) that could have e=None
    # Actually, based on test: if any e_values exist, per_drug should include all scored/refused with e=None
    per_drug = {}
    e_values = [s.e for s in scores if s.refusal is None and s.e is not None]

    if e_values:
        for s in scores:
            per_drug[s.drug] = s.e if s.refusal is None else None
    else:
        # No scored drugs have non-None e - per_drug should be None per spec
        per_drug = None

    # median_abs_e and max_abs_e computed from e_values
    if e_values:
        # Implement median manually (math.median not available in stdlib)
        sorted_abs_e = sorted(abs(e) for e in e_values)
        n = len(sorted_abs_e)
        if n % 2 == 1:
            median_abs_e = sorted_abs_e[n // 2]
        else:
            median_abs_e = (sorted_abs_e[n // 2 - 1] + sorted_abs_e[n // 2]) / 2
        max_abs_e = max(sorted_abs_e)
    else:
        median_abs_e = None
        max_abs_e = None

    return {
        "n_pass": n_pass,
        "n_scored": n_scored,
        "denominator": denominator,
        "median_abs_e": median_abs_e,
        "max_abs_e": max_abs_e,
        "per_drug": per_drug,
    }
