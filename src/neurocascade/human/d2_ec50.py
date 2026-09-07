"""Human in-vivo plasma EC50 values for D2 occupancy (P19.4, registry in P21.1).

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

Why this became a registry (P21.1)
----------------------------------

Until P21.1 the constant above was a bare module-level float,
``HALOPERIDOL_D2_PLASMA_EC50_nM``, and every consumer reached for it by name.
That was honest while haloperidol was the only drug, because the one thing a
bare float cannot carry, the conditions the number was fitted under, was written
out in prose in this module and nowhere else was it needed.

P21.2 read the sources for eight further antipsychotics and established that a
bare float stops being honest the moment a second drug arrives. Three of its
findings are schema requirements rather than commentary, and they are what
:class:`D2PlasmaEC50` exists to enforce:

1. **An EC50 without its Emax is an ambiguous number.** Every published value is
   conditional on whether Emax was fitted freely or pinned at 100%, and the
   conditioning is not a refinement: the same clozapine dataset yields 105 ng/mL
   with Emax free at 60% and 483 ng/mL with Emax pinned, a factor of 4.6.
   Haloperidol moves 0.32 to 0.70 on the same choice. So `emax` and
   `emax_convention` are required fields, not optional annotations.
2. **The concentration a drug's EC50 is expressed in differs by drug.** Every
   reachable risperidone value is on the *active moiety*, parent plus
   9-hydroxyrisperidone summed; every aripiprazole primary is parent-only.
   Feeding a summed concentration to a parent-basis constant is a silent
   two-fold error, so `analyte` is required alongside `plasma_basis`.
3. **"Striatal" is a label, not a measurement.** Vernaleken 2004 prints
   amisulpride at 143.6 ng/mL in putamen against 59.9 in caudate, in the same
   nine subjects with the same tracer; Suzuki 2013 has ziprasidone running the
   other way. `region` is therefore stored as the source printed it, including
   ``None`` when the source printed nothing, rather than normalised to
   "striatum".

P21.1 is a pure refactor and adds **no drug**. The registry has exactly one
entry, the haloperidol constant that was already here, and it predicts exactly
what it predicted before; ``tests/test_human_d2_registry.py`` holds that to
bit-identity against a fixture recorded from the pre-refactor code. Ingesting
the drugs P21.2 found reachable is a later task, and until it happens this
registry is a one-row table with a schema, not a capability. That is stated
plainly here so the schema is not mistaken for coverage.
"""
import math
from dataclasses import dataclass
from types import MappingProxyType

from .pet_observations import (
    HALOPERIDOL_D2_DATASET_IDS,
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    haloperidol_d2_observations,
    occupancy_observations,
)
from .pet_validation import HALOPERIDOL_MW_G_PER_MOL

#: How a registered EC50 was obtained, weakest last. The distinction that
#: matters most is between ``primary`` and ``second-hand``: P21.2 found a review
#: pairing Uchida's risperidone ``Emax = 88%`` with ``EC50 = 8.2 ng/mL``, a pair
#: that does not exist in the source, so a value read only in a quoting source
#: has to say so and carry the Emax from the same sentence it was read in.
EC50_PROVENANCE_CLASSES = (
    "fitted-in-repo",
    "primary",
    "second-hand",
    "sponsor",
)

#: Which Emax the EC50 was fitted against. These are not two parameterisations
#: of one curve, they are two different fitted models, and their constants are
#: not interchangeable. See the module docstring for the sizes involved.
EMAX_CONVENTIONS = (
    "fixed-at-1",
    "fitted-free",
)

#: Whether the concentration axis is total plasma drug or the unbound fraction.
PLASMA_BASES = ("total", "free")


@dataclass(frozen=True)
class D2PlasmaEC50:
    """One drug's plasma concentration to D2 occupancy constant, with the
    conditions that make it mean something.

    Every field below either came off a page or is ``None`` because no page
    printed it. Nothing here is inferred from another drug or another species.
    """

    drug: str
    #: The half-occupancy plasma concentration, nM, on `plasma_basis`.
    ec50_nM: float
    #: Needed to move between nM and the ng/mL the literature prints. Belongs to
    #: the drug, so it lives in the drug's row rather than in a module constant.
    molar_mass_g_per_mol: float
    #: "total" or "free". Not the same axis as `analyte`.
    plasma_basis: str
    #: What was measured in the plasma: parent drug alone, or a summed active
    #: moiety. A summed concentration fed to a parent-basis EC50 is a two-fold
    #: error that raises no exception anywhere else in this codebase.
    analyte: str
    #: True when `analyte` is a sum over parent and one or more metabolites.
    #: Kept as a flag as well as prose because the check a consumer needs to make
    #: is a machine one: a PK model that integrates the parent alone must refuse
    #: a summed-moiety EC50 rather than match on the wording of `analyte`.
    analyte_is_summed: bool
    #: The maximum occupancy the fit was allowed to approach, as a fraction.
    emax: float
    #: One of :data:`EMAX_CONVENTIONS`, saying whether `emax` was fitted or
    #: pinned. The EC50 is only comparable across drugs that share it.
    emax_convention: str
    #: The brain region the occupancy was measured in, exactly as the source
    #: printed it, or ``None`` when the source printed none.
    region: str | None
    #: The PET/SPECT tracer, or ``None`` when the source did not name one.
    tracer: str | None
    #: Route, formulation and dosing state the constant was fitted on. Kept
    #: because the fitted regimen and the driven regimen differ for haloperidol
    #: and P21.2 found that to be the rule rather than the exception.
    regimen: str
    #: One of :data:`EC50_PROVENANCE_CLASSES`.
    provenance: str
    citation: str
    #: Where the value was actually read, when that is not the citation.
    #: Required for ``second-hand`` and ``sponsor`` provenance.
    read_in: str | None = None
    #: Plasma free fraction for this drug, and where it came from. Optional: it
    #: is what allows :meth:`ec50_on_basis` to move between total and free, and
    #: without it that conversion is refused rather than guessed.
    plasma_free_fraction: float | None = None
    free_fraction_citation: str | None = None
    interval_low_nM: float | None = None
    interval_high_nM: float | None = None
    interval_method: str | None = None
    interval_caveat: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.drug:
            raise ValueError("a registry row must name its drug")
        if not (self.ec50_nM > 0.0 and math.isfinite(self.ec50_nM)):
            raise ValueError(
                "%s: ec50_nM must be a positive finite number" % self.drug
            )
        if not (self.molar_mass_g_per_mol > 0.0):
            raise ValueError("%s: molar_mass_g_per_mol must be positive" % self.drug)
        if self.plasma_basis not in PLASMA_BASES:
            raise ValueError(
                "%s: plasma_basis %r is not one of %s"
                % (self.drug, self.plasma_basis, ", ".join(PLASMA_BASES))
            )
        if not self.analyte:
            raise ValueError(
                "%s: analyte must say what was measured in the plasma; parent "
                "and active moiety differ by about twofold for the drugs P21.2 "
                "found, and the difference is silent" % self.drug
            )
        if self.emax_convention not in EMAX_CONVENTIONS:
            raise ValueError(
                "%s: emax_convention %r is not one of %s"
                % (self.drug, self.emax_convention, ", ".join(EMAX_CONVENTIONS))
            )
        if not (0.0 < self.emax <= 1.0):
            raise ValueError(
                "%s: emax must be a fraction in (0, 1]; got %r"
                % (self.drug, self.emax)
            )
        if self.emax_convention == "fixed-at-1" and self.emax != 1.0:
            raise ValueError(
                "%s: emax_convention is 'fixed-at-1' but emax is %r"
                % (self.drug, self.emax)
            )
        if self.provenance not in EC50_PROVENANCE_CLASSES:
            raise ValueError(
                "%s: provenance %r is not one of %s"
                % (self.drug, self.provenance, ", ".join(EC50_PROVENANCE_CLASSES))
            )
        if not self.citation:
            raise ValueError("%s: citation must name the source" % self.drug)
        if not self.regimen:
            raise ValueError(
                "%s: regimen must state the route and dosing state the constant "
                "was fitted on" % self.drug
            )
        if self.provenance in ("second-hand", "sponsor") and not self.read_in:
            raise ValueError(
                "%s: provenance %r requires read_in to name where the value was "
                "actually read, because the primary was not"
                % (self.drug, self.provenance)
            )
        if self.plasma_free_fraction is not None:
            if not 0.0 < self.plasma_free_fraction <= 1.0:
                raise ValueError(
                    "%s: plasma_free_fraction must lie in (0, 1]" % self.drug
                )
            if not self.free_fraction_citation:
                raise ValueError(
                    "%s: a free fraction needs the source that printed it"
                    % self.drug
                )
        bounds = (self.interval_low_nM, self.interval_high_nM)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError(
                "%s: an interval needs both bounds or neither" % self.drug
            )
        if bounds[0] is not None:
            if not bounds[0] < self.ec50_nM < bounds[1]:
                raise ValueError(
                    "%s: the interval %r to %r does not contain the EC50 %r"
                    % (self.drug, bounds[0], bounds[1], self.ec50_nM)
                )
            if not self.interval_method:
                raise ValueError(
                    "%s: an interval must name how it was constructed" % self.drug
                )

    @property
    def ec50_ng_per_mL(self):
        return self.ec50_nM * self.molar_mass_g_per_mol / 1000.0

    def occupancy_at(self, plasma_nM):
        """Occupancy at a plasma concentration, on this row's own model.

        ``Emax * C / (C + EC50)``. For a row whose Emax was pinned at 1 this is
        the plain rectangular hyperbola the chain has always used, and the
        multiplication by exactly 1.0 is exact in IEEE-754, so nothing moves.
        """
        if plasma_nM < 0:
            raise ValueError("plasma concentration must not be negative")
        return self.emax * _hyperbola(plasma_nM, self.ec50_nM)

    def ec50_on_basis(self, plasma_basis):
        """This EC50 restated on the other plasma basis, using the free fraction.

        Refused, not approximated, when no free fraction was recorded for the
        drug. The conversion is arithmetic on a measured fraction and does not
        make the converted value a measurement: V2 in
        ``docs/validation-ledger.md`` is the record of this exact conversion
        failing against Videbaek's free-plasma subjects with a one-signed bias.
        """
        if plasma_basis not in PLASMA_BASES:
            raise ValueError(
                "plasma_basis %r is not one of %s"
                % (plasma_basis, ", ".join(PLASMA_BASES))
            )
        if plasma_basis == self.plasma_basis:
            return self.ec50_nM
        if self.plasma_free_fraction is None:
            raise ValueError(
                "no plasma free fraction is recorded for %s, so its %s-basis "
                "EC50 cannot be restated on a %s basis"
                % (self.drug, self.plasma_basis, plasma_basis)
            )
        if plasma_basis == "free":
            return self.ec50_nM * self.plasma_free_fraction
        return self.ec50_nM / self.plasma_free_fraction

    def comparability_conflicts(self, other):
        """Every axis on which this row and `other` are not the same quantity.

        Empty means the two constants sit on one axis and may be compared. A
        non-empty result is the reason a cross-drug comparison would be wrong,
        stated rather than silently absorbed: P21.2 measured these differences
        at 4.6-fold (Emax convention), about twofold (analyte) and 2.4-fold
        (region), all larger than any held-out error in the ledger.
        """
        conflicts = []
        if self.plasma_basis != other.plasma_basis:
            conflicts.append(
                "plasma basis: %s vs %s" % (self.plasma_basis, other.plasma_basis)
            )
        if (
            self.analyte != other.analyte
            or self.analyte_is_summed != other.analyte_is_summed
        ):
            conflicts.append("analyte: %r vs %r" % (self.analyte, other.analyte))
        if self.emax_convention != other.emax_convention:
            conflicts.append(
                "Emax convention: %s (Emax %g) vs %s (Emax %g)"
                % (
                    self.emax_convention,
                    self.emax,
                    other.emax_convention,
                    other.emax,
                )
            )
        if self.region != other.region:
            conflicts.append("region: %r vs %r" % (self.region, other.region))
        if self.tracer != other.tracer:
            conflicts.append("tracer: %r vs %r" % (self.tracer, other.tracer))
        return tuple(conflicts)

    def require_comparable_with(self, other):
        """Raise unless the two rows are on one axis. See P21.2's schema traps."""
        conflicts = self.comparability_conflicts(other)
        if conflicts:
            raise ValueError(
                "%s and %s are not the same measured quantity and must not be "
                "compared without a stated correction: %s"
                % (self.drug, other.drug, "; ".join(conflicts))
            )

    def summary(self):
        return (
            "%s D2: EC50 %.4f nM (%.3f ng/mL), %s plasma, analyte %s, "
            "Emax %g (%s), region %s, tracer %s, %s [%s]"
            % (
                self.drug,
                self.ec50_nM,
                self.ec50_ng_per_mL,
                self.plasma_basis,
                self.analyte,
                self.emax,
                self.emax_convention,
                self.region if self.region is not None else "not printed",
                self.tracer if self.tracer is not None else "not printed",
                self.regimen,
                self.provenance,
            )
        )


#: The one registered row. Fitted in this repository in P19.4 rather than
#: transcribed, for the reasons in the module docstring. The EC50 is recorded to
#: four decimals rather than computed at import so that a change in the
#: underlying observations shows up as a failing test instead of a silently
#: different model; ``test_human_d2_ec50`` asserts the recompute.
HALOPERIDOL_D2_PLASMA_EC50 = D2PlasmaEC50(
    drug="haloperidol",
    ec50_nM=1.8849,
    molar_mass_g_per_mol=HALOPERIDOL_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent haloperidol",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fixed-at-1",
    region=None,
    tracer=None,
    regimen=(
        "haloperidol decanoate, intramuscular depot, steady state at weeks 1 "
        "and 4, in patients"
    ),
    provenance="fitted-in-repo",
    citation="Nyberg 1995 and Regenthal 1997 group means, fitted here in P19.4",
    plasma_free_fraction=0.090,
    free_fraction_citation=(
        "Videbaek 2001: 'the mean nonprotein bound fraction is 9.0% +/- 0.8%'"
    ),
    interval_low_nM=1.382,
    interval_high_nM=2.525,
    interval_method="profile residual sum of squares, F(1, n-1) at 95%",
    interval_caveat=(
        "Four group means from two depot studies, and the four are not "
        "independent: each study measured the same eight patients at week 1 and "
        "week 4, so the interval is narrower than two independent groups would "
        "justify. Both published oral relations, Kapur 1997 and Fitzgerald 2000, "
        "fall below the lower bound."
    ),
    notes=(
        "region and tracer are None because neither Nyberg 1995 nor Regenthal "
        "1997 printed either for the points fitted on; the 'striatal' label the "
        "rest of the repository uses is inherited from context and P19.7 says so. "
        "The Emax convention is 'fixed-at-1' because the P19.4 model form is "
        "occupancy = C / (C + EC50) with no Emax term, which is a pin at 1 rather "
        "than a free fit. That is also the convention Uchida 2011's pooled "
        "haloperidol ED50 of 0.70 ng/mL uses, which is why it is the comparable "
        "one of his two; it is not recorded as a replication because the study "
        "list behind his pool is paywalled and its overlap with Nyberg and "
        "Regenthal cannot be checked."
    ),
)

#: The drug-keyed registry. Read-only: adding a drug is a task that has to
#: source it, not an assignment at runtime.
D2_PLASMA_EC50 = MappingProxyType(
    {HALOPERIDOL_D2_PLASMA_EC50.drug: HALOPERIDOL_D2_PLASMA_EC50}
)

#: The fitted human in-vivo plasma EC50 for haloperidol striatal D2 occupancy,
#: on a **total** plasma basis. Retained as the name every P19-era caller
#: imports, and derived from the registry row so the two can never diverge.
HALOPERIDOL_D2_PLASMA_EC50_nM = HALOPERIDOL_D2_PLASMA_EC50.ec50_nM

#: Free fraction of haloperidol in plasma, as printed by Videbaek 2001. Used
#: only to put that paper's free-plasma constant on the same axis as the
#: total-plasma ones, in :func:`ec50_landscape`. Nothing in the fit uses it.
HALOPERIDOL_PLASMA_FREE_FRACTION = HALOPERIDOL_D2_PLASMA_EC50.plasma_free_fraction


def registered_d2_drugs():
    """Every drug the registry can supply a D2 plasma EC50 for.

    One, today. P21.2 established that six more are reachable in a primary
    human PET paper and a seventh second-hand, but ingesting them is a later
    task and this function reports what is here rather than what is reachable.
    """
    return tuple(sorted(D2_PLASMA_EC50))


def d2_plasma_ec50(drug):
    """The registry row for `drug`, or a ValueError naming what is registered."""
    try:
        return D2_PLASMA_EC50[drug]
    except KeyError:
        raise ValueError(
            "no human D2 plasma EC50 is registered for %r; registered drugs are "
            "%s. P21.2 (docs/p21-2-multi-drug-ec50-reachability.md) records which "
            "further drugs are reachable in the literature and which are not."
            % (drug, ", ".join(registered_d2_drugs()))
        ) from None


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
