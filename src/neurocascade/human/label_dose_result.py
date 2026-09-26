"""P22.2: predicted D2 dose windows scored against FDA label dose ranges.

This is the comparison step of the label-dose test pre-registered in
``docs/p22-preregistration.md``. The predicted windows come unchanged from
:mod:`.label_dose_prediction`, whose values were committed in P22.2a before any
label range was transcribed as a number. The scoring rules come unchanged from
:mod:`.label_dose_scoring` (P22.2f). This module adds the one thing neither of
those could hold without breaking the ordering rule: the label ranges
themselves, each with the verbatim sentences it was read from.

Only the two drugs the chain can drive have their labels transcribed. The six
refused drugs are reported as named-layer refusals (commitment C2) and their
labels are deliberately left untranscribed. Transcribing them now would put
their ground truth into the repository before any prediction for them exists,
so a later task that wires one of them into the chain could no longer satisfy
the pre-registration's prediction-before-transcription ordering. The reasoning
is recorded as deviation D2 in ``docs/p22-2-label-dose-test.md``.
"""
from dataclasses import dataclass

from .label_dose_prediction import (
    DENOMINATOR,
    TARGET_HIGH,
    TARGET_LOW,
    predict_window,
)
from .label_dose_scoring import aggregate, refused, score
from .occupancy_validation import steady_state_occupancy


@dataclass(frozen=True)
class LabelReading:
    """One defensible reading of a label's adult oral dose range, in mg/day."""

    name: str
    l_low: float
    l_high: float
    derivation: str


@dataclass(frozen=True)
class LabelDoseRange:
    """A drug's transcribed label range, pinned like a PMID is pinned.

    ``readings`` holds every defensible reading. The pre-registration scores the
    narrower one and reports the rest. ``sha256_read`` is the digest of the
    openFDA bytes the quotes were checked against; ``matches_p20_6_pin`` says
    whether that equals the digest ``fda_sources`` pinned on 2026-09-05.
    """

    drug: str
    application_number: str
    effective_time: str
    spl_set_id: str
    sha256_read: str
    matches_p20_6_pin: bool
    quotes: tuple
    readings: tuple
    notes: str = ""

    def __post_init__(self):
        if not self.readings:
            raise ValueError("a transcription needs at least one reading")
        for reading in self.readings:
            if not 0 < reading.l_low <= reading.l_high:
                raise ValueError("%s: need 0 < l_low <= l_high" % reading.name)

    @property
    def scoring_reading(self):
        """The narrowest reading, by fold width, as the pre-registration fixes."""
        return min(self.readings, key=lambda r: r.l_high / r.l_low)


HALOPERIDOL_LABEL = LabelDoseRange(
    drug="haloperidol",
    application_number="ANDA216918",
    effective_time="20260831",
    spl_set_id="a0751f2e-05ab-4ba9-9bae-7b791bf1a621",
    sha256_read="16046778d0b3978b4a6c50f4605c76cbb03c6518708129c1cb08313dc44a90ef",
    matches_p20_6_pin=False,
    quotes=(
        "Oral Administration Initial Dosage Range Adults Moderate "
        "Symptomatology - 0.5 mg to 2 mg b.i.d. or t.i.d. Severe Symptomatology "
        "- 3 mg to 5 mg b.i.d. or t.i.d.",
        "Chronic or Resistant Patients - 3 mg to 5 mg b.i.d. or t.i.d.",
        "Daily dosages up to 100 mg may be necessary in some cases to achieve "
        "an optimal response.",
        "Dosage adjustments, either upward or downward, should be carried out "
        "as rapidly as practicable to achieve optimum therapeutic control.",
        "Upon achieving a satisfactory therapeutic response, dosage should then "
        "be gradually reduced to the lowest effective maintenance level.",
    ),
    readings=(
        LabelReading(
            "printed adult ranges",
            1.0,
            15.0,
            "lowest printed adult dose 0.5 mg b.i.d. = 1.0 mg/day; highest "
            "printed adult dose 5 mg t.i.d. = 15.0 mg/day (Severe, and Chronic "
            "or Resistant)",
        ),
        LabelReading(
            "up to 100 mg",
            1.0,
            100.0,
            "same lower end; upper end read as the label's 'Daily dosages up to "
            "100 mg may be necessary in some cases'",
        ),
    ),
    notes=(
        "A generic's copy of the withdrawn Haldol label; no NDA oral label is "
        "current (P20.6). The indication is 'psychotic disorders', not "
        "schizophrenia, and the adult oral dosing is not split by indication. "
        "The adult doses are printed only as an 'Initial Dosage Range', which "
        "the label does not instruct to increase from (adjustment is 'either "
        "upward or downward'), so under the pre-registration's rule they are "
        "the recommended range and not an excluded titration step. The "
        "sentence 'little evidence that behavior improvement is further "
        "enhanced in dosages beyond 6 mg per day' sits in the Children "
        "section and is not used. The on-disk snapshot does not match the P20.6 "
        "pin (deviation D1); a live openFDA fetch on 2026-09-26 returned "
        "exactly the on-disk bytes, same set_id, version 3, same "
        "effective_time."
    ),
)

OLANZAPINE_LABEL = LabelDoseRange(
    drug="olanzapine",
    application_number="NDA020592",
    effective_time="20260211",
    spl_set_id="b418946a-1ab4-4d89-a012-d94fc361a3c4",
    sha256_read="397beb24e6aaa67921e4b1baab8ef08533af081d86728498cc60dd7cc273d36b",
    matches_p20_6_pin=True,
    quotes=(
        "generally beginning with 5 to 10 mg initially, with a target dose of "
        "10 mg/day within several days.",
        "Efficacy in schizophrenia was demonstrated in a dose range of 10 to 15 "
        "mg/day in clinical trials.",
        "An increase to a dose greater than the target dose of 10 mg/day (i.e., "
        "to a dose of 15 mg/day or greater) is recommended only after clinical "
        "assessment.",
        "Olanzapine is not indicated for use in doses above 20 mg/day.",
        "Maintenance Treatment — The effectiveness of oral olanzapine, 10 "
        "mg/day to 20 mg/day, in maintaining treatment response in "
        "schizophrenic patients",
    ),
    readings=(
        LabelReading(
            "target to cap, union with maintenance",
            10.0,
            20.0,
            "acute: target 10 mg/day (the 5 mg start is a titration step the "
            "label instructs to increase from, so excluded), cap 20 mg/day; "
            "maintenance 10 to 20 mg/day; union 10 to 20 mg/day",
        ),
    ),
    notes=(
        "ZYPREXA, section 2.1, Schizophrenia, Adults. The label record also "
        "carries NDA021086 (Zydis, oral) and NDA021253 (intramuscular); the "
        "oral tablet application is cited. The acute efficacy range 10 to 15 "
        "mg/day does not produce a second reading: the maintenance range 10 to "
        "20 mg/day is in the union either way."
    ),
)

#: Label transcriptions for the drugs the chain can drive, and only those.
LABEL_TRANSCRIPTIONS = {
    "haloperidol": HALOPERIDOL_LABEL,
    "olanzapine": OLANZAPINE_LABEL,
}


@dataclass(frozen=True)
class LabelDoseRow:
    """One drug's line in the P22.2 result table."""

    drug: str
    window: object
    label: LabelDoseRange | None
    score: object
    other_readings: tuple
    #: ((target, dose, trough, mean, peak), ...) at D65 and D78, reported
    #: alongside the mean statistic as the pre-registration requires; they do
    #: not enter the metric.
    occupancy_profile: tuple


def _profile(drug, window):
    rows = []
    for target, dose in ((TARGET_LOW, window.d65_mg_per_day), (TARGET_HIGH, window.d78_mg_per_day)):
        if dose is None:
            continue
        state = steady_state_occupancy(dose, drug=drug)
        rows.append(
            (target, dose, state.trough_occupancy, state.mean_occupancy, state.peak_occupancy)
        )
    return tuple(rows)


def label_dose_row(drug):
    window = predict_window(drug)
    if not window.scored:
        # The first refusing layer is the attribution; every refusing layer
        # and its reason stay on ``window``.
        return LabelDoseRow(drug, window, None, refused(drug, window.refused_layers[0]), (), ())
    label = LABEL_TRANSCRIPTIONS[drug]
    primary = label.scoring_reading
    others = tuple(
        (reading.name, score(drug, window.d65_mg_per_day, window.d78_mg_per_day, reading.l_low, reading.l_high))
        for reading in label.readings
        if reading is not primary
    )
    return LabelDoseRow(
        drug,
        window,
        label,
        score(drug, window.d65_mg_per_day, window.d78_mg_per_day, primary.l_low, primary.l_high),
        others,
        _profile(drug, window),
    )


def label_dose_table():
    return tuple(label_dose_row(drug) for drug in DENOMINATOR)


def label_dose_summary(table=None):
    """The pre-registered aggregate: n_pass / n_scored / 8, median and max |e|."""
    table = label_dose_table() if table is None else table
    return aggregate([row.score for row in table], denominator=len(DENOMINATOR))
