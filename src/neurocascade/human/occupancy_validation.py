"""P19.6: held-out validation of the human dose-to-occupancy chain.

P19.4 fitted a single constant, ``HALOPERIDOL_D2_PLASMA_EC50_nM``, on the P19.3
observations. P19.5 built a chain that uses it. Neither has been checked against
a number it did not already see. This module does that check.

The discipline it is written under
---------------------------------

Every split rule in :data:`PRE_REGISTERED_SPLITS` was written down before any
refit was run, and every one of them is exhaustive or forced. In split A each
point is held out exactly once; in split B each study is held out exactly once;
split C is fixed by which plasma basis P19.4 chose to fit on; split D is fixed
by which relations P19.4's docstring already declared it was leaving out. No
split has a free parameter, so none of them could have been chosen after the
fact to flatter a result. The leakage check in :class:`HeldOutFold` is
arithmetic rather than a promise: a fold whose test point appears in its own
training set raises, and :func:`run_split` asserts that the refit consumed
exactly as many points as the fold handed it.

Nothing here was refitted after a held-out number was seen, no tolerance was
widened, and the two comparisons that came out badly are reported below at the
size they came out.

What is testable, and what is not
--------------------------------

The chain has two arms: dose to plasma, and plasma to occupancy. Only the
second one has validating data in this repository.

:func:`chain_testable_observations` computes the set of registered observations
that pair an oral dose with a differenceable occupancy, and that set is empty.
Every dose-bearing observation is either interval-only (Kapur 1996), a
binding-index the source declines to call an occupancy (Xiberas 2001), or given
by intravenous infusion, which the chain refuses because P19.2 recorded the
haloperidol IV search as empty. So the dose-to-plasma arm is untested here, and
splits A through D all test the occupancy hyperbola against a plasma
concentration somebody measured, not against a dose.

The one whole-chain comparison available is interval containment against Kapur
1996, and :func:`kapur1996_containment` reports it with two disclosures
attached: the 2 mg single-dose peak it turns on was already published in
``docs/p19-5-dose-to-occupancy-chain.md``, so the comparison was not blind, and
the regimen-matched steady-state figure was computed after the single-dose miss
was seen, which makes it a diagnosis of that miss and not a second test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .d2_ec50 import (
    HALOPERIDOL_PLASMA_FREE_FRACTION,
    fit_plasma_ec50,
)
from .dose_to_occupancy import predict_occupancy
from .heldout_validation import HeldOutHumanDrugResult
from .pet_observations import (
    HALOPERIDOL_D2_DATASET_IDS,
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    haloperidol_d2_observations,
    occupancy_observations,
)
from .pet_validation import compare_pet_occupancy

#: Dosing regimens the PO chain could in principle be run against. Depot
#: injection and intravenous infusion are excluded because the chain refuses
#: both. ``unspecified-clinical`` is excluded because a route the source never
#: stated cannot be matched to a PK model that is oral-only.
_CHAIN_COMPATIBLE_REGIMENS = (
    "single-dose",
    "steady-state-daily",
    "single-dose-then-followed",
)

#: One dosing interval for daily administration, in minutes.
_DAILY_INTERVAL_MIN = 1440.0


# ---------------------------------------------------------------------------
# the split rules, declared before any of them was run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitRule:
    """A held-out split rule, stated in full before it is executed.

    ``forced_by`` is the reason the rule has no free parameter left in it. That
    field is the whole point of the class: a split that could have been chosen
    after seeing a result is not a held-out test, and writing down why this one
    could not be is the only thing that distinguishes the two.
    """

    name: str
    question: str
    plasma_basis: str
    parameter_origin: str
    forced_by: str
    refits_in_repo: bool

    def __post_init__(self):
        if self.plasma_basis not in {"total", "free"}:
            raise ValueError("plasma_basis must be total or free")


PRE_REGISTERED_SPLITS = (
    SplitRule(
        name="A-leave-one-out",
        question=(
            "refitting on three of the four total-basis points, how far off is "
            "the fourth?"
        ),
        plasma_basis="total",
        parameter_origin="refitted in repo on the other points",
        forced_by=(
            "exhaustive: every point is held out exactly once, so there is no "
            "subset left to choose"
        ),
        refits_in_repo=True,
    ),
    SplitRule(
        name="B-leave-one-study-out",
        question=(
            "refitting on one depot study, how far off is the other one? this "
            "is the patient-disjoint version of split A"
        ),
        plasma_basis="total",
        parameter_origin="refitted in repo on the other study",
        forced_by=(
            "exhaustive over studies: the two points inside one depot study "
            "are the same eight patients at week 1 and week 4, so split A "
            "leaks at the patient level and this split is the correction for "
            "it, not an alternative to it"
        ),
        refits_in_repo=True,
    ),
    SplitRule(
        name="C-free-basis-set",
        question=(
            "converted to a free-plasma basis, does the total-basis constant "
            "predict the four Videbaek subjects it was never fitted on?"
        ),
        plasma_basis="free",
        parameter_origin=(
            "the full total-basis fit, converted by the free fraction"
        ),
        forced_by=(
            "forced by P19.4: it fitted on the total basis only, so the "
            "free-basis points were already out of the training set before "
            "P19.6 existed"
        ),
        refits_in_repo=False,
    ),
    SplitRule(
        name="D-external-parameter",
        question=(
            "using an EC50 published by other authors, not fitted here at all, "
            "how far off are the four depot points?"
        ),
        plasma_basis="total",
        parameter_origin="published by the source authors, no in-repo fit",
        forced_by=(
            "forced by P19.4: its docstring already declared both published "
            "relations held out, on the grounds that consuming one as our "
            "parameter and validating against the same authors' occupancy "
            "figures would be a model checked against itself"
        ),
        refits_in_repo=False,
    ),
)


def split_rule(name):
    for rule in PRE_REGISTERED_SPLITS:
        if rule.name == name:
            return rule
    raise KeyError(
        "no pre-registered split named %r; the four rules were fixed before "
        "any of them was run and adding a fifth after seeing these results "
        "would not be a held-out test" % (name,)
    )


# ---------------------------------------------------------------------------
# folds, predictions, reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOutFold:
    """One training/test partition, with leakage refused rather than promised."""

    split_name: str
    label: str
    train: tuple
    test: tuple
    parameter_note: str = ""

    def __post_init__(self):
        if not self.test:
            raise ValueError("a fold with no test points tests nothing")
        for item in self.test:
            if any(item == other for other in self.train):
                raise ValueError(
                    "fold %r puts a test observation in its own training set; "
                    "the error it would report is an in-sample error wearing a "
                    "held-out label" % (self.label,)
                )


@dataclass(frozen=True)
class HeldOutPrediction:
    """One held-out point, differenced through ``pet_validation``."""

    split_name: str
    fold_label: str
    observation_label: str
    comparison: object
    ec50_nM: float
    n_train: int
    training_rmse: float | None

    @property
    def observed(self):
        return self.comparison.observation.observed_occupancy

    @property
    def predicted(self):
        return self.comparison.predicted_occupancy

    @property
    def error(self):
        return self.comparison.error

    @property
    def absolute_error(self):
        return self.comparison.absolute_error

    @property
    def standardized_error(self):
        """The error in units of the observation's own reported SD.

        This is ``None`` for every point in the P19.3 set, because not one of
        the fifteen registered observations printed an SD. That is a property
        of the sources, not an oversight here, and it is reported as a result
        rather than patched by assuming a spread nobody measured.
        """
        return self.comparison.standardized_error

    @property
    def error_in_training_rmse_units(self):
        """The error divided by the residual RMSE of its own training fold.

        Deliberately not called a standardised error. It says how the held-out
        miss compares to the scatter the fit already had in sample; it says
        nothing about the measurement precision of the held-out point, which is
        what a standardised error would claim and what this data cannot support.
        Returns ``None`` where there was no in-repo fit to take a residual from.
        """
        if self.training_rmse is None or self.training_rmse <= 0:
            return None
        return self.error / self.training_rmse

    @property
    def contained(self):
        """Whether the prediction falls in the observation's reported range.

        ``None`` when the source printed no range.
        """
        observation = self.comparison.observation
        if observation.observed_range_low is None:
            return None
        return observation.contains(min(self.predicted, 1.0))


@dataclass(frozen=True)
class HeldOutReport:
    """Aggregate error for one split."""

    rule: SplitRule
    predictions: tuple

    @property
    def n(self):
        return len(self.predictions)

    @property
    def mean_absolute_error(self):
        return sum(p.absolute_error for p in self.predictions) / self.n

    @property
    def rmse(self):
        return math.sqrt(sum(p.error ** 2 for p in self.predictions) / self.n)

    @property
    def bias(self):
        """Mean signed error. Positive means the model over-predicts occupancy."""
        return sum(p.error for p in self.predictions) / self.n

    @property
    def max_absolute_error(self):
        return max(p.absolute_error for p in self.predictions)

    @property
    def n_contained(self):
        return sum(1 for p in self.predictions if p.contained is True)

    @property
    def n_with_range(self):
        return sum(1 for p in self.predictions if p.contained is not None)

    @property
    def standardized_errors_available(self):
        """False for every split, because no source printed an SD."""
        return any(p.standardized_error is not None for p in self.predictions)

    def by_fold(self):
        """Per-fold sub-reports, in fold order.

        Split D runs two folds against two different published constants, so
        its pooled aggregate averages over a parameter choice rather than over
        held-out points. The per-fold figures are the ones to read there.
        """
        order = []
        grouped = {}
        for prediction in self.predictions:
            if prediction.fold_label not in grouped:
                order.append(prediction.fold_label)
                grouped[prediction.fold_label] = []
            grouped[prediction.fold_label].append(prediction)
        return tuple(
            (label, HeldOutReport(rule=self.rule, predictions=tuple(grouped[label])))
            for label in order
        )

    def summary(self):
        lines = [
            "split %s (n=%d, %s basis)" % (self.rule.name, self.n, self.rule.plasma_basis),
            "  %s" % self.rule.question,
            "  parameter: %s" % self.rule.parameter_origin,
            "  MAE %.4f  RMSE %.4f  bias %+.4f  max|error| %.4f"
            % (self.mean_absolute_error, self.rmse, self.bias, self.max_absolute_error),
            "  inside the source's own reported range: %d of %d points that have one"
            % (self.n_contained, self.n_with_range),
            "  standardised errors: %s"
            % (
                "available"
                if self.standardized_errors_available
                else "not computable, no source printed an SD"
            ),
        ]
        folds = self.by_fold()
        if len(folds) > 1 and self.n > len(folds):
            for label, sub in folds:
                lines.append(
                    "  fold %s: MAE %.4f  bias %+.4f  max|error| %.4f"
                    % (label, sub.mean_absolute_error, sub.bias, sub.max_absolute_error)
                )
        for p in self.predictions:
            in_rmse = p.error_in_training_rmse_units
            lines.append(
                "    %-36s EC50 %6.4f  observed %.3f  predicted %.4f  error %+.4f  %s"
                % (
                    p.observation_label,
                    p.ec50_nM,
                    p.observed,
                    p.predicted,
                    p.error,
                    (
                        "%+.2f training RMSE" % in_rmse
                        if in_rmse is not None
                        else "no in-repo training residual"
                    ),
                )
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# building the folds
# ---------------------------------------------------------------------------


def _observation_label(observation):
    return "%s %s" % (observation.source.split()[0], observation.source_locator)


def basis_observations(plasma_basis):
    """The differenceable occupancy observations on one plasma basis.

    Total and free are never pooled. They sit about elevenfold apart, which is
    plasma protein binding and not biology, and averaging across them would
    report a spread that no biological process produced.
    """
    return tuple(
        item
        for item in occupancy_observations(haloperidol_d2_observations())
        if item.plasma_concentration_nM is not None
        and item.plasma_concentration_basis == plasma_basis
    )


def _observations_by_dataset(plasma_basis):
    grouped = []
    for dataset_id in HALOPERIDOL_D2_DATASET_IDS:
        points = tuple(
            item
            for item in occupancy_observations(haloperidol_d2_observations(dataset_id))
            if item.plasma_concentration_nM is not None
            and item.plasma_concentration_basis == plasma_basis
        )
        if points:
            grouped.append((dataset_id, points))
    return tuple(grouped)


def build_folds(split_name):
    """The folds a split rule produces, before any of them is run."""
    rule = split_rule(split_name)
    if split_name == "A-leave-one-out":
        points = basis_observations("total")
        return tuple(
            HeldOutFold(
                split_name=split_name,
                label="hold out %s" % _observation_label(held),
                train=tuple(item for item in points if item is not held),
                test=(held,),
            )
            for held in points
        )
    if split_name == "B-leave-one-study-out":
        grouped = _observations_by_dataset("total")
        return tuple(
            HeldOutFold(
                split_name=split_name,
                label="hold out study %s" % dataset_id,
                train=tuple(
                    item
                    for other_id, other_points in grouped
                    if other_id != dataset_id
                    for item in other_points
                ),
                test=points,
            )
            for dataset_id, points in grouped
        )
    if split_name == "C-free-basis-set":
        return (
            HeldOutFold(
                split_name=split_name,
                label="the free-basis set, against the total-basis fit",
                train=(),
                test=basis_observations("free"),
                parameter_note=(
                    "the EC50 is the full total-basis fit converted by the free "
                    "fraction; the free-basis points entered no fit"
                ),
            ),
        )
    if split_name == "D-external-parameter":
        return tuple(
            HeldOutFold(
                split_name=split_name,
                label="the depot points, against %s's published EC50" % relation.label,
                train=(),
                test=basis_observations("total"),
                parameter_note=(
                    "the EC50 is %s's own published constant; nothing in this "
                    "fold was fitted here" % relation.label
                ),
            )
            for relation in HALOPERIDOL_D2_REPORTED_RELATIONS
        )
    raise KeyError(rule.name)


# ---------------------------------------------------------------------------
# running them
# ---------------------------------------------------------------------------


def _predict(ec50_nM, plasma_nM):
    return plasma_nM / (plasma_nM + ec50_nM)


def _run_refitting_fold(rule, fold):
    fit = fit_plasma_ec50(observations=fold.train, plasma_basis=rule.plasma_basis)
    if fit.n_points != len(fold.train):
        raise AssertionError(
            "fold %r handed %d training points and the refit consumed %d; a "
            "held-out error is only held out if those two agree"
            % (fold.label, len(fold.train), fit.n_points)
        )
    predictions = []
    for held in fold.test:
        predicted = _predict(fit.ec50_nM, held.plasma_concentration_nM)
        predictions.append(
            HeldOutPrediction(
                split_name=rule.name,
                fold_label=fold.label,
                observation_label=_observation_label(held),
                comparison=compare_pet_occupancy(held, predicted),
                ec50_nM=fit.ec50_nM,
                n_train=fit.n_points,
                training_rmse=fit.residual_rmse,
            )
        )
    return predictions


def _run_fixed_parameter_fold(rule, fold, ec50_nM):
    predictions = []
    for held in fold.test:
        predicted = _predict(ec50_nM, held.plasma_concentration_nM)
        predictions.append(
            HeldOutPrediction(
                split_name=rule.name,
                fold_label=fold.label,
                observation_label=_observation_label(held),
                comparison=compare_pet_occupancy(held, predicted),
                ec50_nM=ec50_nM,
                n_train=0,
                training_rmse=None,
            )
        )
    return predictions


def run_split(split_name):
    """Execute one pre-registered split and report the error it actually gives."""
    rule = split_rule(split_name)
    folds = build_folds(split_name)
    predictions = []
    if rule.refits_in_repo:
        for fold in folds:
            predictions.extend(_run_refitting_fold(rule, fold))
    elif split_name == "C-free-basis-set":
        ec50_free = fit_plasma_ec50().ec50_nM * HALOPERIDOL_PLASMA_FREE_FRACTION
        for fold in folds:
            predictions.extend(_run_fixed_parameter_fold(rule, fold, ec50_free))
    elif split_name == "D-external-parameter":
        for fold, relation in zip(folds, HALOPERIDOL_D2_REPORTED_RELATIONS):
            predictions.extend(
                _run_fixed_parameter_fold(rule, fold, relation.ec50_nM)
            )
    else:  # pragma: no cover - split_rule already refused anything else
        raise KeyError(split_name)
    return HeldOutReport(rule=rule, predictions=tuple(predictions))


def run_all_splits():
    return tuple(run_split(rule.name) for rule in PRE_REGISTERED_SPLITS)


# ---------------------------------------------------------------------------
# what the chain as a whole can be tested against, which is almost nothing
# ---------------------------------------------------------------------------


def _drug_observations(drug):
    """The registered human D2 PET observation set for ``drug``.

    P21.6: the observation side of the chain is drug-keyed too. Only haloperidol
    has a transcribed set today, so any other drug is refused here rather than
    silently returning an empty tuple, because an empty tuple already carries the
    P19.6 meaning "observations exist but none pairs a dose with a differenceable
    occupancy", and "no observations are registered at all" is a different claim.
    """
    if str(drug).lower() == "haloperidol":
        return haloperidol_d2_observations()
    raise ValueError(
        "no registered human D2 PET observations for %r; only haloperidol has a "
        "transcribed observation set (see neurocascade.human.pet_observations), "
        "so the chain cannot be paired with data for this drug" % drug
    )


def chain_testable_observations(drug="haloperidol"):
    """Observations pairing an oral dose of ``drug`` with a differenceable occupancy.

    For haloperidol this returns an empty tuple, and that emptiness is the single
    most important result in P19.6. It is computed rather than asserted in prose
    so that it becomes false the moment somebody registers an observation that
    would make it false. Any drug without a registered observation set is refused
    by :func:`_drug_observations` rather than returning the same empty tuple,
    which would mean something different.
    """
    return tuple(
        item
        for item in occupancy_observations(_drug_observations(drug))
        if item.dose is not None
        and item.dose_regimen in _CHAIN_COMPATIBLE_REGIMENS
    )


@dataclass(frozen=True, eq=False)
class SteadyStateOccupancy:
    """Occupancy over one dosing interval at steady state under daily dosing.

    Built by superposing the chain's own single-dose curve over successive
    daily intervals, which is exact rather than approximate because the PK
    model is linear. It is a construction on top of the chain, not a second
    model: the same PK parameters and the same EC50 produce it.
    """

    dose_mg_per_day: float
    n_doses: int
    times_min: object
    plasma_nM: object
    occupancy: object
    ec50_nM: float
    #: The occupancy ceiling of the EC50's own fit, already applied to
    #: :attr:`occupancy`. Defaults to 1, which is what haloperidol's row pins.
    emax: float = 1.0

    @property
    def trough_occupancy(self):
        return float(self.occupancy.min())

    @property
    def peak_occupancy(self):
        return float(self.occupancy.max())

    @property
    def mean_occupancy(self):
        return float(self.occupancy.mean())


def steady_state_occupancy(
    dose_mg_per_day, n_doses=40, step_min=5.0, drug="haloperidol", **kwargs
):
    """Steady-state occupancy for once-daily oral dosing of ``drug``.

    ``n_doses`` defaults to 40 because the mean occupancy at 2 mg/day is
    converged to eight decimal places by 30 doses; 40 is past that with margin
    and cheaper than running the tail out further. ``drug`` is the P21.6 key,
    forwarded to :func:`predict_occupancy`, which resolves the PK set and EC50
    from the registries and refuses a drug that lacks either.
    """
    if dose_mg_per_day <= 0:
        raise ValueError("dose_mg_per_day must be positive")
    if n_doses < 1:
        raise ValueError("n_doses must be at least 1")
    per_interval = int(round(_DAILY_INTERVAL_MIN / step_min))
    if per_interval < 2:
        raise ValueError("step_min is too coarse to resolve a dosing interval")
    times = np.arange(0.0, n_doses * _DAILY_INTERVAL_MIN + step_min, step_min)
    prediction = predict_occupancy(
        dose_mg_per_day, drug=drug, times_min=times, **kwargs
    )
    concentration = prediction.plasma_nM
    n_full = len(concentration) // per_interval
    blocks = concentration[: n_full * per_interval].reshape(n_full, per_interval)
    accumulated = blocks.sum(axis=0)
    # The Emax of the EC50's own fit, not 1. Before P22.2 this line divided
    # without it, which was harmless while haloperidol was the only drug in the
    # registry (its row pins Emax at exactly 1) and wrong for the first drug
    # whose source fitted the ceiling free. The pre-registration fixes that the
    # registered constant is used under its own Emax convention, so the ceiling
    # has to travel with it through the superposition as well as through the
    # single-dose curve.
    return SteadyStateOccupancy(
        dose_mg_per_day=float(dose_mg_per_day),
        n_doses=int(n_full),
        times_min=times[:per_interval],
        plasma_nM=accumulated,
        occupancy=prediction.emax * accumulated / (accumulated + prediction.ec50_nM),
        ec50_nM=prediction.ec50_nM,
        emax=prediction.emax,
    )


@dataclass(frozen=True)
class ContainmentResult:
    """An interval-containment comparison, with its own caveats attached."""

    observation_label: str
    dose_mg: float
    dose_regimen: str
    range_low: float
    range_high: float
    single_dose_peak_occupancy: float
    single_dose_contained: bool
    steady_state_trough_occupancy: float
    steady_state_mean_occupancy: float
    steady_state_peak_occupancy: float
    steady_state_mean_contained: bool
    was_blind: bool
    caveats: tuple = field(default_factory=tuple)

    @property
    def distance_below_lower_bound(self):
        """How far the single-dose peak falls short of the reported range."""
        return self.range_low - self.single_dose_peak_occupancy


def kapur1996_containment():
    """The one whole-chain comparison the registered data supports.

    Kapur 1996 reports 53 to 74 percent striatal D2 occupancy in seven patients
    on 2 mg/day of haloperidol, as a range with no central value, so
    ``compare_pet_occupancy`` cannot difference against it and containment is
    the only test available.

    The chain's single-dose peak at 2 mg is not contained. It misses the lower
    bound of the range, and the miss is reported here at the size it came out.
    The regimen-matched steady-state figure is contained, but it was computed
    after the single-dose miss was seen, so it is recorded as a diagnosis of
    that miss rather than as a test the chain passed.
    """
    observation = haloperidol_d2_observations("kapur1996")[0]
    times = np.arange(0.0, 72.0 * 60.0 + 5.0, 5.0)
    single = predict_occupancy(observation.dose, times_min=times)
    repeated = steady_state_occupancy(observation.dose)
    return ContainmentResult(
        observation_label=_observation_label(observation),
        dose_mg=observation.dose,
        dose_regimen=observation.dose_regimen,
        range_low=observation.observed_range_low,
        range_high=observation.observed_range_high,
        single_dose_peak_occupancy=single.peak_occupancy,
        single_dose_contained=observation.contains(single.peak_occupancy),
        steady_state_trough_occupancy=repeated.trough_occupancy,
        steady_state_mean_occupancy=repeated.mean_occupancy,
        steady_state_peak_occupancy=repeated.peak_occupancy,
        steady_state_mean_contained=observation.contains(repeated.mean_occupancy),
        was_blind=False,
        caveats=(
            "not blind: the 2 mg single-dose peak this turns on was already "
            "published in docs/p19-5-dose-to-occupancy-chain.md before P19.6 "
            "ran, so the comparison was made against a number already in view",
            "the steady-state figure was computed after the single-dose miss "
            "was seen; it explains the miss and is not a second, passing test",
            "the source reports a range across seven patients, not a mean with "
            "a spread, so containment says the prediction is somewhere among "
            "them and nothing about where",
        ),
    )


# ---------------------------------------------------------------------------
# the clinical occupancy window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClinicalOccupancyThreshold:
    """An occupancy figure a source associated with a clinical outcome."""

    outcome: str
    occupancy: float
    as_printed: str
    source: str
    source_locator: str
    tracer: str | None
    region: str
    region_origin: str
    statistical_basis: str
    #: A hazard specific to this source, where one exists. Not every threshold
    #: has one, so it defaults to absent rather than to an empty string.
    note: str | None = None
    #: The drugs whose patients the threshold was measured in, or ``None`` when
    #: the source pooled drugs and its composition was not read (P24.1). A
    #: threshold is an occupancy-outcome relation observed under one drug;
    #: placing another drug's occupancy against it is a cross-drug transfer,
    #: and :func:`neurocascade.human.prediction_interface.predict` labels it as one.
    measured_in_drugs: tuple | None = None
    #: The dosing regimen the threshold was observed under, in the regimen
    #: vocabulary of :func:`dose_reaching_occupancy`, or ``None`` if unread.
    measured_regimen: str | None = None

    def __post_init__(self):
        if not 0 < self.occupancy < 1:
            raise ValueError("occupancy must be a fraction strictly inside (0, 1)")


#: Uchida and colleagues published two pooled analyses in J Clin Psychopharmacol
#: volume 31 in 2011, with the same six authors in the same order, and this
#: module cited one of them under the other's title until P21.2 caught it. The
#: number was right and the identifier was right; only the title was borrowed
#: from the sibling paper, which is the kind of error that survives review
#: because every part of it looks correct on its own.
#:
#: They are different papers answering different questions, and the distinction
#: is load-bearing here:
#:
#:   PMID 21694629, 31(4):497-502, "Dopamine D2 receptor occupancy and clinical
#:   effects". Twelve studies, 82 subjects. Cutoffs against EPS and symptom
#:   reduction; the 77-to-78-percent figure this module uses. No plasma
#:   concentration modelling at all. Explicitly excludes partial agonists, which
#:   is why it says nothing about aripiprazole.
#:
#:   PMID 21508857, 31(3):318-25, "Predicting dopamine D2 receptor occupancy
#:   from plasma levels of antipsychotic drugs". Thirty-four treatment arms, 23
#:   studies, 281 subjects. Fits a one-site binding model per drug to estimate
#:   the total plasma concentration giving 50 percent occupancy, for
#:   risperidone, clozapine, olanzapine, haloperidol and ziprasidone. This is
#:   the multi-drug table Phase 21 needs, and it is fully paywalled: closed on
#:   Unpaywall, OpenAlex, Semantic Scholar, Europe PMC and PMC, with no
#:   repository deposit anywhere.
UCHIDA_2011_IS_TWO_PAPERS = (
    "Cited here for the clinical-effects paper, PMID 21694629, 31(4):497-502. "
    "Its sibling PMID 21508857, 31(3):318-25, has the same six authors and the "
    "confusable title 'Predicting dopamine D2 receptor occupancy from plasma "
    "levels of antipsychotic drugs', and is the one carrying per-drug plasma "
    "ED50 values. Neither substitutes for the other."
)


#: ``measured_in_drugs`` and ``measured_regimen`` on the Kapur 2000 rows
#: (P24.1) are read from its PubMed abstract (fetched 2026-10-01): "22 patients
#: with first-episode schizophrenia were randomly assigned to 1.0 or 2.5 mg/day
#: of haloperidol. After 2 weeks of treatment, D(2) receptor occupancy was
#: determined", and in the conclusions "While limited to haloperidol". Uchida
#: 2011 pooled 12 studies whose drug list sits in the paywalled full text, so
#: its row leaves both fields unread rather than guessing.
CLINICAL_OCCUPANCY_THRESHOLDS = (
    ClinicalOccupancyThreshold(
        outcome="clinical response",
        occupancy=0.65,
        as_printed="65%",
        source=(
            "Kapur S, Zipursky R, Jones C, Remington G, Houle S. Relationship "
            "between dopamine D(2) occupancy, clinical response, and side "
            "effects: a double-blind PET study of first-episode schizophrenia. "
            "Am J Psychiatry. 2000;157(4):514-20. PMID 10739409."
        ),
        source_locator="abstract, Results",
        tracer="[11C]raclopride",
        region="striatum",
        region_origin=(
            "inferred: the abstract names the tracer but not the region; "
            "Corpus Striatum is MEDLINE indexer-assigned metadata, not the "
            "authors' printed words"
        ),
        statistical_basis=(
            "the abstract prints only the word significantly: no test, no "
            "p-value, no interval, and no statement of how the change-point "
            "was derived. The full text is paywalled and was not obtained, so "
            "no statistical method is attributed to this paper."
        ),
        measured_in_drugs=("haloperidol",),
        measured_regimen="steady-state-daily",
    ),
    ClinicalOccupancyThreshold(
        outcome="hyperprolactinemia",
        occupancy=0.72,
        as_printed="72%",
        source=(
            "Kapur S, Zipursky R, Jones C, Remington G, Houle S. Relationship "
            "between dopamine D(2) occupancy, clinical response, and side "
            "effects: a double-blind PET study of first-episode schizophrenia. "
            "Am J Psychiatry. 2000;157(4):514-20. PMID 10739409."
        ),
        source_locator="abstract, Results",
        tracer="[11C]raclopride",
        region="striatum",
        region_origin="inferred, as above",
        statistical_basis="as above: significantly, with nothing behind it in the abstract",
        measured_in_drugs=("haloperidol",),
        measured_regimen="steady-state-daily",
    ),
    ClinicalOccupancyThreshold(
        outcome="extrapyramidal side effects",
        occupancy=0.78,
        as_printed="78%",
        source=(
            "Kapur S, Zipursky R, Jones C, Remington G, Houle S. Relationship "
            "between dopamine D(2) occupancy, clinical response, and side "
            "effects: a double-blind PET study of first-episode schizophrenia. "
            "Am J Psychiatry. 2000;157(4):514-20. PMID 10739409."
        ),
        source_locator="abstract, Results",
        tracer="[11C]raclopride",
        region="striatum",
        region_origin="inferred, as above",
        statistical_basis="as above: significantly, with nothing behind it in the abstract",
        measured_in_drugs=("haloperidol",),
        measured_regimen="steady-state-daily",
    ),
    ClinicalOccupancyThreshold(
        outcome="extrapyramidal side effects, pooled reanalysis",
        occupancy=0.775,
        as_printed="77% to 78%",
        source=(
            "Uchida H, Takeuchi H, Graff-Guerrero A, Suzuki T, Watanabe K, "
            "Mamo DC. Dopamine D2 receptor occupancy and clinical effects: a "
            "systematic review and pooled analysis. J Clin Psychopharmacol. "
            "2011;31(4):497-502. PMID 21694629."
        ),
        source_locator="abstract, Results",
        tracer=None,
        region="striatum",
        region_origin="stated by the source as a pooled striatal analysis",
        statistical_basis=(
            "cutoff maximising accuracy at sensitivity and specificity both at "
            "or above 0.5, pooled over 12 studies and 82 subjects. The full "
            "text is paywalled and was not obtained."
        ),
        note=UCHIDA_2011_IS_TWO_PAPERS,
    ),
)

#: Why this module never states a 65-to-80-percent window as a sourced fact.
CLINICAL_WINDOW_PROVENANCE = (
    "The 65 to 80 percent window is a synthesis, not a primary result, and no "
    "primary PET paper checked for P19.6 prints it. Farde 1988 (PMID 2892477) "
    "prints 65% to 85% as the range observed across eleven antipsychotics. "
    "Farde 1992 (PMID 1352677) prints 70% to 89% for classical neuroleptics "
    "and, for extrapyramidal side effects, prints no number at all: its claim "
    "is the qualitative one that patients with acute extrapyramidal syndromes "
    "had a higher D2 occupancy than those without. Kapur 2000 prints 65, 72 "
    "and 78. The lower bound of 65 is therefore traceable, but to two papers "
    "making different kinds of claim: a range floor in Farde 1988 and a "
    "response change-point in Kapur 2000. The upper bound of 80 was not found "
    "in any primary paper and appears to be a rounding of Kapur 2000's 78. "
    "Uchida 2011's pooled reanalysis of 12 studies and 82 subjects revises the "
    "window to 60 to 78 percent and describes 65 to 80 as something PET "
    "studies proposed from samples under twenty. This module uses 78, not 80, "
    "and cites Kapur 2000 for it."
)

#: Why reproducing an occupancy is not predicting the outcome attached to it.
OCCUPANCY_IS_NOT_THE_OUTCOME = (
    "Reaching a given occupancy is not the same as predicting the clinical "
    "state associated with it, and this repository already holds the "
    "falsification. Siafis 2023 (PMC10618092, N=37,193) reports haloperidol at "
    "11 mg/day carrying an extrapyramidal odds ratio of 4.7 against "
    "risperidone at 16 mg/day carrying 3.3, at approximately the same 87 "
    "percent occupancy, so occupancy alone does not determine the side effect. "
    "Kurose 2020 (PMID 32726002) reports that central occupancy decays more "
    "slowly than plasma concentration, which this chain cannot represent at "
    "all: it maps plasma to occupancy instantaneously, with no effect "
    "compartment and no ke0. The chain predicts an occupancy. It does not "
    "predict response, hyperprolactinemia or extrapyramidal side effects, and "
    "no arithmetic in this module should be read as if it did."
)


#: The dose bracket :func:`dose_reaching_occupancy` searches, mg/day. Named
#: rather than inlined so that a refusal can quote the bracket it failed in.
DOSE_SEARCH_MIN_MG_PER_DAY = 0.01
DOSE_SEARCH_MAX_MG_PER_DAY = 500.0


class OccupancyUnreachableError(ValueError):
    """A target occupancy the chain cannot reach for a drug, with the reason.

    Raised rather than returned as a sentinel because a dose that does not exist
    must not be able to flow into an arithmetic mean by accident. P22.2 scores a
    drug whose window is unreachable as an overlap failure with an undefined
    midpoint error, which requires catching this rather than reading a NaN.
    """


def dose_reaching_occupancy(
    target_occupancy, regimen="steady-state-daily", drug="haloperidol", **kwargs
):
    """The daily dose at which the chain reaches a target occupancy for ``drug``.

    ``steady-state-daily`` uses the mean occupancy over a dosing interval at
    steady state; ``single-dose`` uses the peak of a single administration.
    Which one is meant matters more than the number: the same target is reached
    at very different doses under the two, and quoting one where the other was
    intended is the easiest error to make with these figures. ``drug`` is the
    P21.6 key, forwarded to the chain, which refuses a drug that lacks a PK set
    or an EC50.
    """
    from scipy.optimize import brentq

    if not 0 < target_occupancy < 1:
        raise ValueError("target_occupancy must be a fraction strictly inside (0, 1)")
    # The Emax the drug's registered EC50 was fitted under, captured from
    # whichever object the regimen builds rather than looked up a second way, so
    # a refusal quotes the ceiling the search actually ran against.
    seen = {}
    if regimen == "steady-state-daily":
        def miss(dose):
            result = steady_state_occupancy(dose, drug=drug, **kwargs)
            seen["emax"] = result.emax
            return result.mean_occupancy - target_occupancy
    elif regimen == "single-dose":
        times = np.arange(0.0, 72.0 * 60.0 + 5.0, 5.0)

        def miss(dose):
            result = predict_occupancy(dose, drug=drug, times_min=times, **kwargs)
            seen["emax"] = result.emax
            return result.peak_occupancy - target_occupancy
    else:
        raise ValueError(
            "regimen must be steady-state-daily or single-dose; the chain has "
            "no parameters for anything else"
        )
    # Two ways the target can be out of reach, and they are different findings.
    # A target above the EC50 fit's own Emax is unreachable at *any* dose: the
    # source's hyperbola asymptotes below it. A target below Emax but not met at
    # the top of the search bracket is reachable in principle and merely needs
    # more drug than this bracket allows. The pre-registration requires the
    # first to be recorded as unreachable rather than imputed, so both are
    # raised as a named exception instead of surfacing as brentq's
    # "f(a) and f(b) must have different signs".
    top = miss(DOSE_SEARCH_MAX_MG_PER_DAY)
    if top < 0.0:
        emax = seen["emax"]
        if target_occupancy >= emax:
            raise OccupancyUnreachableError(
                "%s cannot reach %.3f occupancy at any dose: the registered "
                "EC50's own fit has Emax %.3f, so its hyperbola asymptotes "
                "below the target. The bound is unreachable, not merely large."
                % (drug, target_occupancy, emax)
            )
        raise OccupancyUnreachableError(
            "%s does not reach %.3f occupancy by %g mg/day under regimen %r "
            "(it reaches %.3f there); the EC50 fit's Emax is %.3f, so the "
            "target is reachable in principle but not inside the dose bracket "
            "this search uses."
            % (
                drug,
                target_occupancy,
                DOSE_SEARCH_MAX_MG_PER_DAY,
                regimen,
                top + target_occupancy,
                emax,
            )
        )
    return float(brentq(miss, DOSE_SEARCH_MIN_MG_PER_DAY, DOSE_SEARCH_MAX_MG_PER_DAY, xtol=1e-6))


def clinical_window_check():
    """Does the chain place the sourced thresholds at clinically used doses?

    Returns one row per threshold with the daily dose the chain says reaches
    it. The question this answers is narrow and worth stating precisely: it
    asks whether the chain's dose-occupancy curve lands the sourced occupancy
    figures inside the range of doses haloperidol is actually given at. It does
    not ask, and cannot ask, whether the outcome attached to each threshold
    follows. See :data:`OCCUPANCY_IS_NOT_THE_OUTCOME`.
    """
    rows = []
    for threshold in CLINICAL_OCCUPANCY_THRESHOLDS:
        rows.append(
            {
                "outcome": threshold.outcome,
                "occupancy": threshold.occupancy,
                "as_printed": threshold.as_printed,
                "source": threshold.source,
                "steady_state_dose_mg_per_day": dose_reaching_occupancy(
                    threshold.occupancy, regimen="steady-state-daily"
                ),
                "single_dose_mg": dose_reaching_occupancy(
                    threshold.occupancy, regimen="single-dose"
                ),
            }
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# reporting into the existing held-out vocabulary
# ---------------------------------------------------------------------------


def as_heldout_results():
    """The P19.6 splits expressed in the P18.2 held-out vocabulary.

    This exists so the aggregate can be cross-checked by ``summarize_heldout``,
    which was written for a different set of drugs and knows nothing about this
    one. An error the two disagree about is an error in one of them.
    """
    results = []
    for report in run_all_splits():
        for prediction in report.predictions:
            results.append(
                HeldOutHumanDrugResult(
                    compound="haloperidol",
                    endpoint="striatal D2 occupancy",
                    status="scored",
                    source=prediction.observation_label,
                    predicted=prediction.predicted,
                    observed=prediction.observed,
                    note="%s / %s" % (report.rule.name, prediction.fold_label),
                )
            )
    return tuple(results)


def summary():
    """Everything P19.6 found, in the order it was found."""
    lines = ["P19.6 held-out validation of the haloperidol dose-to-occupancy chain", ""]
    testable = chain_testable_observations()
    lines.append(
        "observations pairing an oral dose with a differenceable occupancy: %d"
        % len(testable)
    )
    lines.append(
        "  so the dose-to-plasma arm of the chain is untested; every split "
        "below tests the plasma-to-occupancy hyperbola only"
    )
    lines.append("")
    for report in run_all_splits():
        lines.append(report.summary())
        lines.append("")
    containment = kapur1996_containment()
    lines.append(
        "whole-chain containment against %s (%.1f mg %s, range %.2f-%.2f):"
        % (
            containment.observation_label,
            containment.dose_mg,
            containment.dose_regimen,
            containment.range_low,
            containment.range_high,
        )
    )
    lines.append(
        "  single-dose peak %.4f, contained: %s (%.4f below the lower bound)"
        % (
            containment.single_dose_peak_occupancy,
            containment.single_dose_contained,
            containment.distance_below_lower_bound,
        )
    )
    lines.append(
        "  regimen-matched steady state: trough %.4f, mean %.4f, peak %.4f; "
        "mean contained: %s"
        % (
            containment.steady_state_trough_occupancy,
            containment.steady_state_mean_occupancy,
            containment.steady_state_peak_occupancy,
            containment.steady_state_mean_contained,
        )
    )
    for caveat in containment.caveats:
        lines.append("  caveat: %s" % caveat)
    lines.append("")
    lines.append("sourced occupancy thresholds and the dose the chain reaches them at:")
    for row in clinical_window_check():
        lines.append(
            "  %-46s %s  %.2f mg/day at steady state, %.2f mg single dose"
            % (
                row["outcome"],
                row["as_printed"],
                row["steady_state_dose_mg_per_day"],
                row["single_dose_mg"],
            )
        )
    lines.append("")
    lines.append(CLINICAL_WINDOW_PROVENANCE)
    lines.append("")
    lines.append(OCCUPANCY_IS_NOT_THE_OUTCOME)
    return "\n".join(lines)
