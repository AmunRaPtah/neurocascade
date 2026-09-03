"""The human in-vivo plasma EC50 for haloperidol D2 occupancy (P19.4).

This is the constant the occupancy model actually consumes. It is **not** an
in-vitro Ki, and the difference is the point of this module.

An in-vitro Ki is measured against a chosen radioligand in a chosen buffer on a
chosen receptor preparation, and Seeman & Van Tol 1995 showed that the apparent
Ki of a neuroleptic at the human D2 receptor moves systematically with that
choice, by more than tenfold across [3H]nemonapride, [3H]spiperone and
[3H]raclopride. The quantity that predicts an occupancy in a living person from
a plasma concentration is an in-vivo EC50, which additionally absorbs plasma
protein binding, brain penetration, the endogenous dopamine the drug is
competing against, and the tracer the occupancy itself was measured with. The
two are different quantities with the same units, so this module keeps them in
separate fields and reports their ratio rather than treating one as the other.

The EC50 here is **fitted in this repository** from the observations registered
in :mod:`neurocascade.human.pet_observations`, not transcribed from a paper.
That choice is deliberate:

* It is reproducible. :func:`fit_plasma_ec50` recomputes it from the committed
  fixtures, and a test asserts the recorded constant still equals the recompute.
* It leaves the published relations held out. Kapur 1997 and
  Fitzgerald 2000 each report their own fitted plasma-to-occupancy hyperbola.
  Consuming one of those as our parameter and then "validating" against the same
  authors' occupancy figures would be a model checked against itself.

What the fit rests on is thin and is stated rather than hidden. Only four
registered occupancy points carry a **total** plasma concentration, and all four
come from the two haloperidol-decanoate depot studies: Nyberg 1995 and Regenthal
1997, each contributing a week-1 and a week-4 group mean for the same eight
patients. So the fit has four points but only two independent patient groups,
and both are depot intramuscular steady state in patients rather than the oral
case the chain is being built for. The reasons to take it seriously anyway are
that the two studies were done by different groups with different methods and
land within 7% of each other, and that the residuals are small.
"""
import math
from dataclasses import dataclass

from .pet_observations import (
    HALOPERIDOL_D2_DATASET_IDS,
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    haloperidol_d2_observations,
    occupancy_observations,
)
from .pet_validation import HALOPERIDOL_MW_G_PER_MOL

#: The fitted human in-vivo plasma EC50 for haloperidol striatal D2 occupancy,
#: on a **total** plasma basis. Recorded rather than computed at import so that
#: a change in the underlying observations shows up as a failing test instead of
#: a silently different model. ``test_human_d2_ec50`` asserts the recompute.
HALOPERIDOL_D2_PLASMA_EC50_nM = 1.8849

#: Free fraction of haloperidol in plasma, as printed by Videbaek 2001: "the
#: mean nonprotein bound fraction is 9.0% +/- 0.8%". Used only to put that
#: paper's free-plasma constant on the same axis as the total-plasma ones, in
#: :func:`ec50_landscape`. Nothing in the fit itself uses it.
HALOPERIDOL_PLASMA_FREE_FRACTION = 0.090


def _hyperbola(concentration, ec50):
    return concentration / (concentration + ec50)


def _fit_one(concentrations, occupancies):
    """Least-squares EC50 of occupancy = C / (C + EC50).

    One parameter, so this is done by bisection on the derivative of the
    residual sum of squares rather than by pulling in an optimiser. The
    objective is smooth and unimodal in EC50 over the bracket, and doing it this
    way keeps the fit exactly reproducible across scipy versions.
    """
    if len(concentrations) < 1:
        raise ValueError("need at least one point to fit an EC50")
    for value, occupancy in zip(concentrations, occupancies):
        if value <= 0:
            raise ValueError("plasma concentrations must be positive")
        if not 0.0 < occupancy < 1.0:
            # An occupancy of exactly 0 or 1 puts the implied EC50 at infinity
            # or zero and would dominate any fit it entered.
            raise ValueError("occupancies must lie strictly inside (0, 1) to fit")

    def rss(ec50):
        return sum(
            (occupancy - _hyperbola(value, ec50)) ** 2
            for value, occupancy in zip(concentrations, occupancies)
        )

    def d_rss(ec50):
        total = 0.0
        for value, occupancy in zip(concentrations, occupancies):
            predicted = _hyperbola(value, ec50)
            total += 2.0 * (occupancy - predicted) * value / (value + ec50) ** 2
        return total

    # Every point's own implied EC50 is C(1-y)/y, and the least-squares
    # solution cannot lie outside the range of those.
    implied = [
        value * (1.0 - occupancy) / occupancy
        for value, occupancy in zip(concentrations, occupancies)
    ]
    low, high = min(implied) * 0.5, max(implied) * 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if d_rss(low) * d_rss(mid) <= 0:
            high = mid
        else:
            low = mid
    ec50 = 0.5 * (low + high)
    return ec50, rss(ec50)


def _profile_interval(concentrations, occupancies, ec50, rss_min, level=0.95):
    """95% interval from the residual-sum-of-squares profile.

    The bound is the classical one for a single nonlinear parameter,
    ``RSS(theta) <= RSS_min * (1 + F(1, n-1, level) / (n-1))``. With four points
    this is an asymptotic approximation being asked to work at n=4, which is
    why :attr:`EC50Fit.interval_caveat` says so in words next to the number.
    """
    n = len(concentrations)
    if n < 2:
        return None, None
    from scipy.stats import f as f_distribution

    threshold = rss_min * (1.0 + float(f_distribution.ppf(level, 1, n - 1)) / (n - 1))

    def excess(value):
        return (
            sum(
                (occupancy - _hyperbola(concentration, value)) ** 2
                for concentration, occupancy in zip(concentrations, occupancies)
            )
            - threshold
        )

    def bisect(low, high):
        for _ in range(200):
            mid = 0.5 * (low + high)
            if excess(low) * excess(mid) <= 0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    lower = bisect(1e-9, ec50) if excess(1e-9) > 0 else None
    upper_bracket = ec50 * 1000.0
    upper = bisect(ec50, upper_bracket) if excess(upper_bracket) > 0 else None
    return lower, upper


@dataclass(frozen=True)
class EC50Fit:
    """A plasma-concentration to occupancy fit and everything needed to judge it."""

    ec50_nM: float
    plasma_basis: str
    n_points: int
    n_independent_groups: int
    per_study_ec50_nM: tuple
    residual_rmse: float
    standard_error_nM: float | None
    interval_low_nM: float | None
    interval_high_nM: float | None
    interval_method: str
    interval_caveat: str
    model: str
    sources: tuple
    notes: str = ""

    @property
    def ec50_ng_per_mL(self):
        return self.ec50_nM * HALOPERIDOL_MW_G_PER_MOL / 1000.0

    def occupancy_at(self, plasma_nM):
        if plasma_nM < 0:
            raise ValueError("plasma concentration must not be negative")
        return _hyperbola(plasma_nM, self.ec50_nM)


def fit_plasma_ec50(observations=None, plasma_basis="total"):
    """Fit occupancy = C / (C + EC50) to the registered observations.

    Only points that are occupancies with a value, that carry a plasma
    concentration, and that are on the requested basis are used. Total and free
    are never pooled: they differ here by about elevenfold, so a pooled fit
    would be a fit to the protein binding rather than to the receptor.
    """
    if plasma_basis not in {"total", "free"}:
        raise ValueError("plasma_basis must be total or free")
    if observations is None:
        observations = haloperidol_d2_observations()

    grouped = []
    for dataset_id in HALOPERIDOL_D2_DATASET_IDS:
        points = [
            item
            for item in occupancy_observations(haloperidol_d2_observations(dataset_id))
            if item.plasma_concentration_nM is not None
            and item.plasma_concentration_basis == plasma_basis
            and item in observations
        ]
        if points:
            grouped.append((dataset_id, points))
    if not grouped:
        raise ValueError("no registered occupancy points on a %s plasma basis" % plasma_basis)

    concentrations = [p.plasma_concentration_nM for group in grouped for p in group[1]]
    occupancies = [p.observed_occupancy for group in grouped for p in group[1]]
    ec50, rss_min = _fit_one(concentrations, occupancies)

    per_study = []
    for dataset_id, points in grouped:
        if len(points) >= 1:
            study_ec50, _ = _fit_one(
                [p.plasma_concentration_nM for p in points],
                [p.observed_occupancy for p in points],
            )
            per_study.append((dataset_id, study_ec50))

    n = len(concentrations)
    rmse = math.sqrt(rss_min / n)
    standard_error = None
    if n > 1:
        # Asymptotic standard error for the single fitted parameter, from the
        # Gauss-Newton approximation to the Hessian.
        gradient_sq = sum(
            (value / (value + ec50) ** 2) ** 2 for value in concentrations
        )
        if gradient_sq > 0:
            standard_error = math.sqrt((rss_min / (n - 1)) / gradient_sq)
    low, high = _profile_interval(concentrations, occupancies, ec50, rss_min)

    return EC50Fit(
        ec50_nM=ec50,
        plasma_basis=plasma_basis,
        n_points=n,
        n_independent_groups=len(grouped),
        per_study_ec50_nM=tuple(per_study),
        residual_rmse=rmse,
        standard_error_nM=standard_error,
        interval_low_nM=low,
        interval_high_nM=high,
        interval_method="profile residual sum of squares, F(1, n-1) at 95%",
        interval_caveat=(
            "%d points from %d independent patient groups; the two points within "
            "a depot study are the same patients measured at week 1 and week 4, "
            "so the points are not independent and this interval is optimistic."
            % (n, len(grouped))
        ),
        model="rectangular hyperbola, occupancy = C / (C + EC50), no baseline term",
        sources=tuple(dataset_id for dataset_id, _ in grouped),
        notes=(
            "Fitted on unweighted occupancy residuals. The sources print group "
            "means without standard errors, so there is nothing to weight by."
        ),
    )


@dataclass(frozen=True)
class EC50Comparison:
    """Our fitted constant set against one the literature already reported."""

    label: str
    reported_ec50_nM: float
    fitted_ec50_nM: float
    ratio_fitted_over_reported: float
    reported_basis: str
    tracer: str | None
    inside_our_interval: bool | None
    comment: str


def compare_with_reported_relations(fit=None):
    """Compare the in-repo fit against the two published relations.

    These are compared, never merged. Both published constants are themselves
    fitted models, so agreement here is a consistency check between two models
    and not a validation of either against data.
    """
    if fit is None:
        fit = fit_plasma_ec50()
    comparisons = []
    for relation in HALOPERIDOL_D2_REPORTED_RELATIONS:
        inside = None
        if fit.interval_low_nM is not None and fit.interval_high_nM is not None:
            inside = fit.interval_low_nM <= relation.ec50_nM <= fit.interval_high_nM
        comparisons.append(
            EC50Comparison(
                label=relation.label,
                reported_ec50_nM=relation.ec50_nM,
                fitted_ec50_nM=fit.ec50_nM,
                ratio_fitted_over_reported=fit.ec50_nM / relation.ec50_nM,
                reported_basis=relation.plasma_concentration_basis,
                tracer=relation.tracer,
                inside_our_interval=inside,
                comment=(
                    "Oral dosing measured with %s; ours is depot intramuscular "
                    "steady state with the tracer unnamed in both sources."
                    % relation.tracer
                ),
            )
        )
    return tuple(comparisons)


def ec50_landscape():
    """Every human D2 constant this repository can put on one axis.

    All values are nM on a **total** plasma basis. Videbaek 2001 is the one
    entry that had to be converted to get there, and the conversion is named in
    its own row rather than applied silently.
    """
    fit = fit_plasma_ec50(plasma_basis="total")
    free_fit = fit_plasma_ec50(plasma_basis="free")
    rows = [
        (
            "fitted, depot, this repository",
            fit.ec50_nM,
            "Nyberg 1995 and Regenthal 1997 group means, tracer unnamed",
        ),
    ]
    for relation in HALOPERIDOL_D2_REPORTED_RELATIONS:
        rows.append(
            (
                "reported, oral, %s" % relation.label,
                relation.ec50_nM,
                "%s, as the authors fitted it" % relation.tracer,
            )
        )
    rows.append(
        (
            "fitted, intravenous, videbaek2001",
            free_fit.ec50_nM / HALOPERIDOL_PLASMA_FREE_FRACTION,
            "free-plasma fit of %.3f nM divided by the paper's own free fraction "
            "of %.1f%%; the only converted row here"
            % (free_fit.ec50_nM, 100.0 * HALOPERIDOL_PLASMA_FREE_FRACTION),
        )
    )
    return tuple(rows)
