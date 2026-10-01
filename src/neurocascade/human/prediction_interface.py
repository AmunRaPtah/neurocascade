"""The public prediction interface (P24.1).

One entry point, :func:`predict`, takes a drug, a dose and a regimen and
returns the striatal D2 occupancy-time course the chain predicts, together with
where that regimen sits against each sourced clinical occupancy threshold. It
adds no model: the curve is :func:`~neurocascade.human.dose_to_occupancy.
predict_occupancy` for a single dose and :func:`~neurocascade.human.
occupancy_validation.steady_state_occupancy` for once-daily dosing, which are
the computations ledger rows V4 and V8 scored.

What this module adds is the refusal discipline at the boundary.

**Every missing layer is named before anything is computed.** The chain's own
gates stop at the first missing layer. A caller asking about a drug needs the
whole list, because a drug missing its PK set *and* its EC50 is not one data
item away from a prediction. :func:`refusal_layers` checks every layer
independently and :func:`predict` raises :class:`PredictionRefused` carrying all
of them, before any integration runs.

**There is no default drug and no name normalisation.** The chain functions
underneath default to haloperidol, for backward compatibility with P19.5; this
interface does not, so a caller who forgets the drug gets a ``TypeError``
rather than haloperidol's curve. ``"Haloperidol"`` is refused rather than
case-folded, because a lookup that guesses which drug was meant is the first
step towards one that guesses wrong.

**A threshold measured in another drug is labelled as a transfer.** Kapur 2000
(PMID 10739409) measured its 65/72/78% thresholds in haloperidol-treated
patients only, and says so ("While limited to haloperidol"). Placing
olanzapine's occupancy against them is arithmetic on a haloperidol relation, so
each :class:`ThresholdPosition` carries ``same_drug`` and a note saying which
drug the threshold came from. The comparison is reported, not hidden, because
the P22.2 pre-registration already applied these thresholds across drugs; it is
never presented as if it had been measured in the drug being predicted.
"""
from dataclasses import dataclass

import numpy as np

from .active_metabolites import require_parent_only_path
from .d2_ec50 import d2_plasma_ec50, registered_d2_drugs
from .dose_to_occupancy import (
    _require_no_brain_partition_step,
    _require_pk_record,
    predict_occupancy,
)
from .occupancy_validation import (
    CLINICAL_OCCUPANCY_THRESHOLDS,
    CLINICAL_WINDOW_PROVENANCE,
    OCCUPANCY_IS_NOT_THE_OUTCOME,
    steady_state_occupancy,
)
from .pk_parameters import human_pk_records

#: The regimens the chain has parameters for, and nothing else. A single oral
#: dose is reported from 0 to 72 h; once-daily dosing is reported over one
#: 24 h interval at steady state.
REGIMENS = ("single-dose", "steady-state-daily")

#: The occupancy statistic each regimen is placed against the thresholds by.
#: Steady-state mean is the P22.1 pre-registered statistic (V8); a single dose
#: has no interval to average over, so its peak is used.
REGIMEN_STATISTIC = {"single-dose": "peak", "steady-state-daily": "mean"}

#: Every layer :func:`refusal_layers` checks, in the order it reports them.
LAYERS = ("active-metabolite", "pk", "occupancy", "exposure")

#: The only route any drug has a sourced PK record for.
_ROUTE = "PO"
_TARGET = "D2"
#: Single-dose report grid: 72 h at 5 min, the grid
#: :func:`dose_reaching_occupancy` uses for its single-dose search.
_SINGLE_DOSE_TIMES_MIN = np.arange(0.0, 72.0 * 60.0 + 5.0, 5.0)


class PredictionRefused(ValueError):
    """The chain cannot predict this drug; every missing layer is named.

    A feature, not a failure: the refusal is the interface telling the caller
    which data item does not exist, rather than producing a number from
    parameters that belong to another drug.
    """

    def __init__(self, drug, missing):
        self.drug = drug
        self.missing = tuple(missing)
        super().__init__(
            "predict refuses %r: missing layer%s %s. Drugs with every layer: %s.\n%s"
            % (
                drug,
                "" if len(self.missing) == 1 else "s",
                ", ".join(layer for layer, _ in self.missing),
                ", ".join(supported_drugs()) or "none",
                "\n".join("  %s: %s" % pair for pair in self.missing),
            )
        )

    @property
    def layers(self):
        return tuple(layer for layer, _ in self.missing)


def refusal_layers(drug):
    """``((layer, reason), ...)`` for every layer missing for ``drug``.

    Each layer is checked on its own, so the result lists all of them rather
    than the first. Empty means the chain can drive the drug. Every check is a
    lookup keyed by ``drug``; none reads another drug's record.
    """
    found = []
    try:
        require_parent_only_path(drug, target=_TARGET)
    except ValueError as error:
        found.append(("active-metabolite", str(error)))
    try:
        _require_pk_record(drug, _ROUTE, None)
    except ValueError as error:
        found.append(("pk", str(error)))
    try:
        d2_plasma_ec50(drug)
    except ValueError as error:
        found.append(("occupancy", str(error)))
    try:
        _require_no_brain_partition_step(drug)
    except ValueError as error:
        found.append(("exposure", str(error)))
    return tuple(found)


def supported_drugs():
    """Drugs any registry names that :func:`refusal_layers` clears, sorted."""
    candidates = set(registered_d2_drugs())
    candidates.update(record.drug for record in human_pk_records())
    return tuple(sorted(d for d in candidates if not refusal_layers(d)))


@dataclass(frozen=True)
class ThresholdPosition:
    """Where a regimen's occupancy statistic sits against one sourced threshold."""

    outcome: str
    threshold: float
    source: str
    #: The regimen statistic placed against the threshold (see
    #: :data:`REGIMEN_STATISTIC`).
    statistic: str
    occupancy: float
    #: Fraction of the reported time course at or above the threshold.
    fraction_of_time_at_or_above: float
    #: The occupancy ceiling of the drug's own EC50 fit is below the threshold,
    #: so no dose reaches it.
    unreachable_at_any_dose: bool
    #: The threshold was measured in patients taking the drug being predicted.
    #: ``False`` both for a different drug and for an unread pooled source.
    same_drug: bool
    #: The threshold was measured under the regimen being predicted.
    same_regimen: bool
    note: str

    @property
    def at_or_above(self):
        return self.occupancy >= self.threshold

    @property
    def margin(self):
        return self.occupancy - self.threshold


def _position(threshold, drug, regimen, statistic_value, occupancy, emax):
    measured_in = threshold.measured_in_drugs
    same_drug = measured_in is not None and drug in measured_in
    same_regimen = threshold.measured_regimen == regimen
    notes = []
    if measured_in is None:
        notes.append(
            "pooled across drugs whose composition was not read, so it is not "
            "specific to %s" % drug
        )
    elif not same_drug:
        notes.append(
            "measured in %s-treated patients, not %s: a cross-drug transfer of "
            "the occupancy-outcome relation" % (", ".join(measured_in), drug)
        )
    if threshold.measured_regimen is None:
        notes.append("the regimen it was measured under was not read")
    elif not same_regimen:
        notes.append(
            "measured under %s, predicted under %s"
            % (threshold.measured_regimen, regimen)
        )
    return ThresholdPosition(
        outcome=threshold.outcome,
        threshold=threshold.occupancy,
        source=threshold.source,
        statistic=REGIMEN_STATISTIC[regimen],
        occupancy=statistic_value,
        fraction_of_time_at_or_above=float(np.mean(occupancy >= threshold.occupancy)),
        unreachable_at_any_dose=threshold.occupancy >= emax,
        same_drug=same_drug,
        same_regimen=same_regimen,
        note="; ".join(notes) if notes else "measured in this drug under this regimen",
    )


@dataclass(frozen=True, eq=False)
class Prediction:
    """An occupancy-time course, its provenance, and its threshold positions."""

    drug: str
    dose_mg: float
    regimen: str
    route: str
    body_weight_kg: float
    #: Minutes from the dose (single dose) or from the start of a steady-state
    #: dosing interval (steady-state-daily).
    times_min: np.ndarray
    plasma_nM: np.ndarray
    occupancy: np.ndarray
    ec50_nM: float
    emax: float
    layers: tuple
    metabolite_caveats: tuple
    thresholds: tuple
    #: Read before using any threshold position: reaching an occupancy is not
    #: predicting the outcome a source attached to it.
    caveats: tuple = (OCCUPANCY_IS_NOT_THE_OUTCOME, CLINICAL_WINDOW_PROVENANCE)

    @property
    def peak_occupancy(self):
        return float(np.max(self.occupancy))

    @property
    def trough_occupancy(self):
        return float(np.min(self.occupancy))

    @property
    def mean_occupancy(self):
        return float(np.mean(self.occupancy))

    @property
    def statistic(self):
        return REGIMEN_STATISTIC[self.regimen]

    @property
    def statistic_occupancy(self):
        return self.peak_occupancy if self.statistic == "peak" else self.mean_occupancy

    @property
    def provenance(self):
        if any(layer.provenance == "illustrative" for layer in self.layers):
            return "illustrative"
        return "measured"

    def summary(self):
        lines = [
            "%s %g mg %s, %s: %s striatal D2 occupancy %.3f (trough %.3f, peak %.3f)"
            % (
                self.drug,
                self.dose_mg,
                self.route,
                self.regimen,
                self.statistic,
                self.statistic_occupancy,
                self.trough_occupancy,
                self.peak_occupancy,
            ),
            "occupancy = %g * C / (C + %.4f nM), provenance %s"
            % (self.emax, self.ec50_nM, self.provenance),
        ]
        for position in self.thresholds:
            lines.append(
                "  %s %.3f (%s): %s by %+.3f [%s]"
                % (
                    position.outcome,
                    position.threshold,
                    position.statistic,
                    "at or above" if position.at_or_above else "below",
                    position.margin,
                    position.note,
                )
            )
        return "\n".join(lines)


def _validated_dose(dose_mg):
    if isinstance(dose_mg, bool) or not isinstance(dose_mg, (int, float, np.number)):
        raise TypeError("dose_mg must be a number of milligrams; got %r" % (dose_mg,))
    dose = float(dose_mg)
    if not np.isfinite(dose) or dose <= 0.0:
        raise ValueError("dose_mg must be positive and finite; got %r" % (dose_mg,))
    return dose


def predict(drug, dose_mg, regimen):
    """Predict striatal D2 occupancy for an oral ``drug`` regimen.

    Parameters
    ----------
    drug : a registry key, exactly as registered (``supported_drugs()`` lists
        them). No default and no case-folding.
    dose_mg : mg per administration, positive. Under ``steady-state-daily`` it is
        the once-daily dose.
    regimen : one of :data:`REGIMENS`.

    The subject is the reference subject of the drug's PK source (70 kg for
    haloperidol and olanzapine; since P25.2 the chain takes the weight from the
    record rather than assuming 70 kg). The chain refuses any other weight
    without a sourced scaling rule, which no drug has, so this interface does
    not offer one.

    Raises
    ------
    PredictionRefused : naming every missing layer, before anything is computed.
    """
    if not isinstance(drug, str):
        raise TypeError("drug must be a registry key string; got %r" % (drug,))
    dose = _validated_dose(dose_mg)
    if regimen not in REGIMENS:
        raise ValueError(
            "regimen must be one of %s; got %r. The chain has parameters for no "
            "other regimen." % (", ".join(REGIMENS), regimen)
        )
    missing = refusal_layers(drug)
    if missing:
        raise PredictionRefused(drug, missing)

    if regimen == "single-dose":
        chain = predict_occupancy(dose, drug=drug, times_min=_SINGLE_DOSE_TIMES_MIN)
        times, plasma, occupancy = chain.times_min, chain.plasma_nM, chain.occupancy
    else:
        steady = steady_state_occupancy(dose, drug=drug)
        # The provenance of the steady-state curve is that of the single-dose
        # chain it superposes; the short call recovers it without a second model.
        chain = predict_occupancy(dose, drug=drug, times_min=steady.times_min)
        if steady.ec50_nM != chain.ec50_nM or steady.emax != chain.emax:
            raise RuntimeError("steady-state and single-dose chains resolved different constants")
        times, plasma, occupancy = steady.times_min, steady.plasma_nM, steady.occupancy
    if chain.drug != drug:
        raise RuntimeError("the chain returned %r for %r" % (chain.drug, drug))

    statistic_value = float(
        np.max(occupancy) if REGIMEN_STATISTIC[regimen] == "peak" else np.mean(occupancy)
    )
    thresholds = tuple(
        _position(threshold, drug, regimen, statistic_value, occupancy, chain.emax)
        for threshold in CLINICAL_OCCUPANCY_THRESHOLDS
    )
    return Prediction(
        drug=drug,
        dose_mg=dose,
        regimen=regimen,
        route=chain.route,
        body_weight_kg=chain.body_weight_kg,
        times_min=np.asarray(times),
        plasma_nM=np.asarray(plasma),
        occupancy=np.asarray(occupancy),
        ec50_nM=chain.ec50_nM,
        emax=chain.emax,
        # The chain's transduction layer describes d2_effect, which this
        # interface does not apply, and its text is written for haloperidol.
        # Returning it for another drug would hand that drug a haloperidol
        # statement as provenance, so only the layers consumed are kept.
        layers=tuple(layer for layer in chain.layers if layer.name != "transduction"),
        metabolite_caveats=chain.metabolite_caveats,
        thresholds=thresholds,
    )
