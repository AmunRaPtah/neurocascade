"""Human PET occupancy validation contracts (P18.1, extended in P19.3).

P18.1 defined the comparison arithmetic. It had nothing real to compare
against: every observation in its tests was built inline from invented
numbers, so the module could not tell a measured point from a plausible one.

P19.3 registers the first measured human observations in this repository, and
that required the observation type to carry what a PET occupancy number
actually depends on. Four things an occupancy value cannot be read without:

* **The tracer.** [11C]raclopride, [123I]IBZM, [18F]fallypride and
  [11C]FLB 457 have affinities spanning orders of magnitude and different
  susceptibility to endogenous dopamine. Occupancies measured with different
  tracers are different quantities, so ``tracer`` is required and
  :func:`observations_by_tracer` is the only supported way to take a set
  apart. Nothing here silently pools them.
* **The region.** Striatal and extrastriatal D2 occupancy differ in the same
  subject at the same dose, so "the striatum" is not a synonym for "the brain".
* **The exposure the occupancy belongs to.** A dose with no timing is not an
  exposure: 2 mg once and 2 mg/day at steady state are different states of the
  same patient. ``dose_regimen`` is required whenever a dose is given, and a
  point may instead (or additionally) carry a measured plasma concentration,
  which is the quantity a PK chain can actually predict.
* **Where the number came from.** A value read off a published scatter plot is
  not the same evidence as a value printed in a table. ``value_origin`` marks
  it, and a digitised point must name the figure it was digitised from.

Two further distinctions came out of the P19.3 search rather than out of
theory, because the literature contains points that look like occupancies and
are not:

* **What the number measures.** Xiberas 2001 prints striatal binding indices
  and says in the text that they "should not be considered as effective D2
  receptor occupancy figures". Vallabhajosula 1997 prints basal-ganglia to
  cerebellum ratios, which run the other way: a higher ratio means *less* drug
  bound. ``quantity`` separates the three, and :func:`compare_pet_occupancy`
  refuses anything that is not an occupancy.
* **Whether a plasma concentration is total or free.** Videbaek 2001 reports
  free haloperidol, everyone else reports total, and the free fraction is
  about 9 percent. Comparing the two without saying so is an order-of-magnitude
  error dressed as agreement, so ``plasma_concentration_basis`` is required
  whenever a concentration is present.

Some sources print a group range with no mean. Kapur 1996 gives "53%-74%" for
seven patients and no central value anywhere in the abstract. Inventing the
midpoint would put a number in this repository that nobody measured, so such a
point carries ``observed_range_low``/``observed_range_high`` and leaves
``observed_occupancy`` as ``None``. It still supports a real test, just an
interval one: see :meth:`PETOccupancyObservation.contains`.
"""
from dataclasses import dataclass


#: Haloperidol, C21H23ClFNO2. Used only to make printed mass concentrations
#: and molar concentrations comparable; the printed value is kept verbatim in
#: ``plasma_concentration_as_printed`` so the conversion stays inspectable.
HALOPERIDOL_MW_G_PER_MOL = 375.86

_VALUE_ORIGINS = {"table", "text", "digitised"}

#: What the printed number measures. Only ``occupancy`` is the fraction of
#: receptors bound by the drug. ``binding-index`` is a tracer-binding reduction
#: the source itself declines to call an occupancy. ``availability-ratio`` is a
#: region-to-reference ratio, which moves in the opposite direction to
#: occupancy and so is the easiest of the three to misread.
_QUANTITIES = {"occupancy", "binding-index", "availability-ratio"}

_PLASMA_BASES = {"total", "free"}

_DOSE_REGIMENS = {
    "single-dose",
    "steady-state-daily",
    "single-dose-then-followed",
    "unspecified-clinical",
    # A depot injection is neither: the exposure decays over weeks from one
    # administration, so the same patient sits at a different concentration in
    # week 1 and week 4 with no dose in between.
    "depot-injection",
    "intravenous-infusion",
}


@dataclass(frozen=True)
class PETOccupancyObservation:
    """One measured receptor-occupancy observation.

    ``dose`` is in mg. ``observed_occupancy``, ``observed_sd`` and the range
    bounds are fractions in [0, 1], not percentages, whatever the source
    printed. ``observed_occupancy`` is ``None`` only when the source printed a
    range and no central value.
    """

    compound: str
    target: str
    dose: float | None
    observed_occupancy: float | None
    source: str
    observed_sd: float | None = None
    tracer: str | None = None
    region: str | None = None
    n: int | None = None
    quantification_method: str | None = None
    population: str | None = None
    dose_regimen: str | None = None
    plasma_concentration_nM: float | None = None
    plasma_concentration_as_printed: str | None = None
    plasma_concentration_basis: str | None = None
    source_locator: str | None = None
    value_origin: str | None = None
    quantity: str = "occupancy"
    observed_range_low: float | None = None
    observed_range_high: float | None = None
    notes: str = ""

    def __post_init__(self):
        if self.dose is not None and self.dose <= 0:
            raise ValueError("dose must be positive")
        if self.quantity not in _QUANTITIES:
            raise ValueError("quantity must be one of %s" % sorted(_QUANTITIES))
        for name in ("observed_occupancy", "observed_range_low", "observed_range_high"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError("%s must be in [0, 1]" % name)
        has_range = self.observed_range_low is not None or self.observed_range_high is not None
        if has_range and (self.observed_range_low is None or self.observed_range_high is None):
            raise ValueError("a range needs both bounds")
        if has_range and self.observed_range_low > self.observed_range_high:
            raise ValueError("observed_range_low must not exceed observed_range_high")
        if self.observed_occupancy is None and not has_range:
            # Refusing to default here is the point: a source that printed
            # neither a value nor a range has not reported an observation.
            raise ValueError("an observation needs a value or a range")
        if self.observed_occupancy is not None and has_range:
            if not self.observed_range_low <= self.observed_occupancy <= self.observed_range_high:
                raise ValueError("observed_occupancy lies outside its own reported range")
        if not self.compound or not self.target or not self.source:
            raise ValueError("compound, target, and source are required")
        if self.observed_sd is not None and self.observed_sd <= 0:
            raise ValueError("observed_sd must be positive")
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1")
        if self.plasma_concentration_nM is not None and self.plasma_concentration_nM <= 0:
            raise ValueError("plasma_concentration_nM must be positive")
        if self.dose is not None and self.dose_regimen is None:
            # A dose without timing is not an exposure. 2 mg once and 2 mg/day
            # at steady state are different states of the same patient.
            raise ValueError("a dose requires a dose_regimen")
        if self.dose_regimen is not None and self.dose_regimen not in _DOSE_REGIMENS:
            raise ValueError("unrecognised dose_regimen: %s" % self.dose_regimen)
        if self.value_origin is not None and self.value_origin not in _VALUE_ORIGINS:
            raise ValueError("value_origin must be one of %s" % sorted(_VALUE_ORIGINS))
        if self.value_origin == "digitised" and not self.source_locator:
            # A digitised value has to name the figure it was read off, or the
            # reader cannot check it and cannot re-digitise it.
            raise ValueError("a digitised value must name the figure it came from")
        if self.plasma_concentration_nM is not None and not self.plasma_concentration_as_printed:
            # Keeping the printed value makes any unit conversion inspectable
            # rather than something a reader has to trust.
            raise ValueError("a plasma concentration must record its printed form")
        if self.plasma_concentration_nM is not None and self.plasma_concentration_basis is None:
            # Total and free differ here by about elevenfold. A concentration
            # that does not say which it is cannot be compared to anything.
            raise ValueError("a plasma concentration must say whether it is total or free")
        if (
            self.plasma_concentration_basis is not None
            and self.plasma_concentration_basis not in _PLASMA_BASES
        ):
            raise ValueError("plasma_concentration_basis must be total or free")

    @property
    def is_measured_exposure(self):
        """True if this point can be placed on a PK prediction at all."""
        return self.dose is not None or self.plasma_concentration_nM is not None

    @property
    def is_interval_only(self):
        """True if the source printed a range and no central value."""
        return self.observed_occupancy is None

    @property
    def observed_occupancy_percent(self):
        if self.observed_occupancy is None:
            return None
        return 100.0 * self.observed_occupancy

    def contains(self, predicted_occupancy):
        """Test a prediction against a range-only point.

        This is the whole test such a point supports. It is weaker than a
        signed error against a mean, and it is what the source actually
        reported.
        """
        if self.observed_range_low is None:
            raise ValueError("this observation reports no range")
        return self.observed_range_low <= predicted_occupancy <= self.observed_range_high


@dataclass(frozen=True)
class PETOccupancyComparison:
    observation: PETOccupancyObservation
    predicted_occupancy: float

    @property
    def error(self):
        return self.predicted_occupancy - self.observation.observed_occupancy

    @property
    def absolute_error(self):
        return abs(self.error)

    @property
    def standardized_error(self):
        sd = self.observation.observed_sd
        if sd is None:
            return None
        return self.error / sd


def compare_pet_occupancy(observation, predicted_occupancy):
    if not 0 <= predicted_occupancy <= 1:
        raise ValueError("predicted_occupancy must be in [0, 1]")
    if observation.quantity != "occupancy":
        raise ValueError(
            "cannot difference a prediction against a %s; the source did not "
            "report an occupancy" % observation.quantity
        )
    if observation.observed_occupancy is None:
        raise ValueError(
            "this observation reports only a range; use contains() instead of "
            "inventing a central value for it"
        )
    return PETOccupancyComparison(observation, float(predicted_occupancy))


def observations_by_tracer(observations):
    """Group observations by tracer.

    This is the only supported way to split a set. Occupancy measured with
    different tracers is not the same quantity, so a caller that wants one
    number out of several tracers has to say which tracer it means.
    """
    grouped = {}
    for item in observations:
        if item.tracer is None:
            raise ValueError("cannot group observations that do not name a tracer")
        grouped.setdefault(item.tracer, []).append(item)
    return {tracer: tuple(items) for tracer, items in grouped.items()}


def require_single_tracer(observations):
    """Return the one tracer behind a set, or raise naming the ones found."""
    tracers = sorted(observations_by_tracer(observations))
    if len(tracers) != 1:
        raise ValueError(
            "observations span %d tracers (%s); occupancy from different "
            "tracers is not the same quantity and must not be pooled"
            % (len(tracers), ", ".join(tracers))
        )
    return tracers[0]


def ng_per_mL_to_nM(value_ng_per_mL, mw_g_per_mol=HALOPERIDOL_MW_G_PER_MOL):
    """ng/mL to nM. 1 ng/mL = (1000 / MW) nmol/L."""
    return 1000.0 * value_ng_per_mL / mw_g_per_mol
