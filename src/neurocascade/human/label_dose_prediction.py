"""Predicted steady-state D2 dose windows for the P22.2 label-dose test.

For each drug in the pre-registered denominator
(``docs/p22-preregistration.md``, P22.2) this module either predicts the window
W = [D65, D78] of total once-daily oral doses whose steady-state mean striatal
D2 occupancy is 0.65 and 0.78, or names every chain layer that refuses the
drug. It reads no label. The label ranges are transcribed separately, after the
table this module produces has been committed.
"""
from dataclasses import dataclass

from .active_metabolites import require_parent_only_path
from .d2_ec50 import d2_plasma_ec50
from .dose_to_occupancy import _require_pk_record
from .fda_pk_bridge import refusals
from .fda_popk import FDA_POPPK_RECORDS
from .occupancy_validation import OccupancyUnreachableError, dose_reaching_occupancy

#: The eight drugs with ``status="ok"`` in ``fda_sources.LABELS``, in the order
#: the pre-registration lists them. Fixed in P22.1.
DENOMINATOR = (
    "haloperidol",
    "risperidone",
    "paliperidone",
    "olanzapine",
    "quetiapine",
    "ziprasidone",
    "aripiprazole",
    "clozapine",
)

#: The two occupancy targets, from ``CLINICAL_WINDOW_PROVENANCE`` (Kapur 2000;
#: 0.78 rather than the synthesized 0.80), fixed by the pre-registration.
TARGET_LOW = 0.65
TARGET_HIGH = 0.78

#: The one drug whose window was published before P22.2 (P19.5/P19.6).
NOT_BLIND = ("haloperidol",)


@dataclass(frozen=True)
class PredictedWindow:
    drug: str
    d65_mg_per_day: float | None
    d78_mg_per_day: float | None
    refused_layers: tuple
    reasons: tuple
    blind: bool

    @property
    def scored(self):
        return not self.refused_layers


def _pk_reason(drug, error):
    rows = [
        record
        for record in FDA_POPPK_RECORDS
        if record.drug == drug and record.route == "PO"
    ]
    if not rows:
        return "%s; no FDA population-PK row exists for oral %s" % (error, drug)
    return "%s; %s" % (
        error,
        "; ".join(
            "FDA %s (%s): %s"
            % (
                record.application_number,
                record.formulation,
                (refusals(record) or ("bridged",))[0],
            )
            for record in rows
        ),
    )


def refusal_layers(drug):
    """``((layer, reason), ...)`` for every layer that refuses ``drug``.

    Checked in the order active-metabolite, pk, occupancy. Empty means the
    chain can drive the drug.
    """
    found = []
    try:
        require_parent_only_path(drug, target="D2")
    except ValueError as error:
        found.append(("active-metabolite", str(error)))
    try:
        # Frozen at the P22 sources: this result was scored against the PK
        # registry as it stood then, before P25.2 added literature records.
        _require_pk_record(drug, "PO", None, include_literature=False)
    except ValueError as error:
        found.append(("pk", _pk_reason(drug, error)))
    try:
        d2_plasma_ec50(drug)
    except ValueError as error:
        found.append(("occupancy", str(error)))
    return tuple(found)


def _dose_or_none(target, drug):
    try:
        return dose_reaching_occupancy(target, regimen="steady-state-daily", drug=drug)
    except OccupancyUnreachableError:
        return None


def predict_window(drug):
    if drug not in DENOMINATOR:
        raise ValueError(
            "%r is not in the pre-registered P22.2 denominator: %s"
            % (drug, ", ".join(DENOMINATOR))
        )
    blind = drug not in NOT_BLIND
    found = refusal_layers(drug)
    if found:
        return PredictedWindow(
            drug,
            None,
            None,
            tuple(layer for layer, _ in found),
            tuple(reason for _, reason in found),
            blind,
        )
    return PredictedWindow(
        drug,
        _dose_or_none(TARGET_LOW, drug),
        _dose_or_none(TARGET_HIGH, drug),
        (),
        (),
        blind,
    )


def prediction_table():
    return tuple(predict_window(drug) for drug in DENOMINATOR)
