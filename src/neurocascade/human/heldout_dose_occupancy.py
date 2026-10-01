"""Oral dose to striatal D2 occupancy, end to end, scored against held-out PET (P25.3).

P25.2 left four drugs with an oral PK record the chain accepts: haloperidol,
olanzapine, quetiapine (Zheng 2024) and aripiprazole (Knights 2015). This module
completes the chain for the two new ones and scores all four against PET
occupancy measured in people who were given a known oral dose, by groups that
are not the source of the constant the prediction uses.

**What the brief called "Kp,uu and D2 affinity" is one layer here, not two.**
The chain identifies brain exposure from an in-vivo plasma EC50 fitted in living
people, so brain penetration, protein binding and competition with dopamine are
already inside the constant (P19.5). Each new drug therefore gets:

* a plasma EC50 from a human [11C]raclopride study, read first-hand; and
* a Kp,uu record of status ``not-required`` that names that EC50, which the
  chain's double-counting guard checks. No Kp,uu value is entered, and an
  in-vitro Ki is never substituted for the in-vivo constant.

Both live in this module rather than in the live registries, as P22.3's rows
did. :data:`~neurocascade.human.d2_ec50.D2_PLASMA_EC50` and
:data:`~neurocascade.human.kp_uu.HUMAN_KP_UU_RECORDS` drive the frozen P22.2 and
P22.5 results and the public interface's refusals. They are handed to
:func:`~neurocascade.human.dose_to_occupancy.predict_occupancy` explicitly.

**The pass threshold is reused, not chosen**: P22.3's pre-registered
:data:`~neurocascade.human.leave_one_drug_out.PASS_MAE_SAME_TRACER`, 0.10
occupancy units of mean absolute error per drug. Every observation and every
constant here is [11C]raclopride. Plasma, where the PET source printed it, is
compared at the V5/V13 twofold threshold. That comparison is reported and does
not decide anything.

The design, the disjointness argument per drug and the observations refused are
in ``docs/p25-3-heldout-dose-occupancy.md``.
"""
from dataclasses import dataclass

import numpy as np

from .d2_ec50 import D2PlasmaEC50, d2_plasma_ec50
from .dose_to_occupancy import predict_occupancy
from .kp_uu import KpUURecord
from .leave_one_drug_out import (
    ARIPIPRAZOLE_D2_PLASMA_EC50,
    OBSERVATIONS as P22_3_OBSERVATIONS,
    PASS_MAE_SAME_TRACER,
)
from .occupancy_validation import kapur1996_containment, steady_state_occupancy
from .receptor_parameters import d2_occupancy_parameter_from_ec50

#: Twofold, the V5/V13/V14 threshold, reused for the plasma arm.
PLASMA_FOLD_THRESHOLD = 2.0

#: Quetiapine free base. The SEROQUEL label (NDA 020639, data/fda snapshot,
#: effective_time 20260410) prints the 2:1 fumarate C42H50N6O4S2.C4H4O4 at
#: 883.11 g/mol and states that "all doses and tablet strengths are expressed as
#: milligrams of base". One base is half of (883.11 - 116.07), where 116.07 is
#: fumaric acid C4H4O4 from standard atomic weights.
QUETIAPINE_MW_G_PER_MOL = (883.11 - 116.07) / 2.0

#: Nord 2011's IR-only constant. The source prints three Emax-fixed values:
#: 1369 nM for IR and XR pooled, 1373 for IR alone and 1365 for XR alone. The
#: IR value is used because the PK record (Zheng 2024) and the held-out
#: observation (Tauscher-Wisniewski 2002) are both immediate release. The
#: choice moves the constant by 0.3%.
QUETIAPINE_D2_PLASMA_EC50 = D2PlasmaEC50(
    drug="quetiapine",
    ec50_nM=1373.0,
    molar_mass_g_per_mol=QUETIAPINE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent quetiapine",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fixed-at-1",
    region="putamen",
    tracer="[11C]raclopride",
    regimen=(
        "oral IR 300 mg once daily in the morning for 4 d after 8 d of XR, "
        "healthy men, steady state; scans at 2 h (Cmax) and next-morning trough"
    ),
    provenance="primary",
    citation=(
        "Nord M, Nyberg S, Brogren J, et al. Int J Neuropsychopharmacol "
        "2011;14(10):1357-66. PMID 21477416, PMC3198174: 'K i,app calculated for "
        "quetiapine IR only was 1373 nmol/l'; 'Occ max was fixed to 100%'"
    ),
    notes=(
        "An Emax-fixed plasma-to-occupancy constant fitted across peak and trough "
        "scans, so it absorbs whatever plasma-to-occupancy hysteresis exists "
        "between those two times (P21.2, P22.3). It is used only for a test in "
        "the same once-daily peak-and-trough design. Karolinska healthy "
        "volunteers recruited from an AstraZeneca database."
    ),
)

_NOT_REQUIRED = (
    "No Kp,uu is consumed; the in-vivo plasma EC50 was fitted against plasma in "
    "living people, so it already contains brain penetration."
)

#: Exposure-layer records for the two drugs this task completes. Same status
#: and wording as olanzapine's registry record, each naming its own EC50.
KP_UU_NOT_REQUIRED = {
    "aripiprazole": KpUURecord(
        "aripiprazole", "human", None, "absent",
        "Kim 2012 plasma EC50 (PMID 22186667, PMC3318151), fitted against plasma "
        "in living people", "not-required", _NOT_REQUIRED,
    ),
    "quetiapine": KpUURecord(
        "quetiapine", "human", None, "absent",
        "Nord 2011 plasma K_i,app (PMID 21477416, PMC3198174), fitted against "
        "plasma in living people", "not-required", _NOT_REQUIRED,
    ),
}

#: The EC50 row each new drug is driven with.
EC50_ROWS = {
    "aripiprazole": ARIPIPRAZOLE_D2_PLASMA_EC50,
    "quetiapine": QUETIAPINE_D2_PLASMA_EC50,
}

SINGLE_DOSE = "single oral dose"
ONCE_DAILY = "once daily, steady state"


@dataclass(frozen=True)
class DoseOccupancyObservation:
    """A group-mean PET occupancy after a known oral dose regimen.

    ``scan_h`` is the time after the last dose, as an interval when the source
    averaged over a scan. ``observed_plasma_ng_mL`` is the source's own plasma
    at the scan, or ``None``. ``independent_group`` names the subjects, so two
    regions measured in the same people are not counted as two samples.
    """

    drug: str
    source: str
    dose_mg: float
    regimen: str
    scan_h: tuple
    region: str
    tracer: str
    observed: float
    observed_sd: float | None
    n: int | None
    independent_group: str
    observed_plasma_ng_mL: float | None
    read_from: str
    disjointness: str
    notes: str = ""

    def __post_init__(self):
        if self.regimen not in (SINGLE_DOSE, ONCE_DAILY):
            raise ValueError("unknown regimen %r" % self.regimen)
        if len(self.scan_h) != 2 or self.scan_h[0] > self.scan_h[1]:
            raise ValueError("scan_h must be (start, end) hours after the dose")
        if not 0.0 <= self.observed <= 1.0:
            raise ValueError("occupancy must be a fraction")


_ITO = (
    "Ito H, Takano H, Arakawa R, et al. PLoS One 2012;7(9):e46488, PMC3460902, "
    "Table 1 and Results"
)
_ITO_DISJOINT = (
    "Chiba (NIRS) healthy Japanese men; Kim 2012's EC50 subjects were Seoul "
    "National University Hospital volunteers, and Knights 2015 is the ABILIFY "
    "programme's pooled trial data"
)
_ITO_NOTE = (
    "aripiprazole given 3.5 h before the start of a 60 min [11C]raclopride scan; "
    "plasma 'averaged between the start and end of each scanning'"
)
_TAUSCHER = (
    "Tauscher-Wisniewski S, Kapur S, Tauscher J, et al. J Clin Psychiatry "
    "2002;63(11):992-7. PMID 12444812, abstract"
)
_TAUSCHER_DISJOINT = (
    "Toronto first-episode patients; Nord 2011's constant came from Karolinska "
    "healthy volunteers"
)
_TAUSCHER_NOTE = (
    "'single daily dose (mean +/- SD dose at the time of the PET scan = 427 +/- "
    "69 mg)' for 12 weeks; IR, the only formulation in 2002. Predicted at the "
    "mean dose. The abstract prints n = 14 enrolled and not the number scanned"
)


def _from_p22_3(record):
    """Ito 2012's group means, taken from P22.3's transcription rather than
    retyped, so the two tests cannot drift apart."""
    return DoseOccupancyObservation(
        record.drug, _ITO, record.dose_mg, SINGLE_DOSE, (3.5, 4.5), record.region,
        record.tracer, record.observed_occupancy, record.observed_sd, record.n,
        "Ito 2012, %g mg, n=%d" % (record.dose_mg, record.n),
        record.plasma_ng_per_mL, "full text", _ITO_DISJOINT,
        _ITO_NOTE + ("; " + record.notes if record.notes else ""),
    )


OBSERVATIONS = tuple(
    _from_p22_3(record) for record in P22_3_OBSERVATIONS if record.drug == "aripiprazole"
) + (
    DoseOccupancyObservation(
        "quetiapine", _TAUSCHER, 427.0, ONCE_DAILY, (2.0, 2.0), "striatum",
        "[11C]raclopride", 0.62, 0.10, None,
        "Tauscher-Wisniewski 2002", None, "abstract", _TAUSCHER_DISJOINT,
        _TAUSCHER_NOTE + "; 'approximately 2 hours postdose'",
    ),
    DoseOccupancyObservation(
        "quetiapine", _TAUSCHER, 427.0, ONCE_DAILY, (20.0, 20.0), "striatum",
        "[11C]raclopride", 0.14, 0.08, None,
        "Tauscher-Wisniewski 2002", None, "abstract", _TAUSCHER_DISJOINT,
        _TAUSCHER_NOTE + "; 'approximately 20 hours postdose'",
    ),
    DoseOccupancyObservation(
        "olanzapine",
        "Mamo D, Kapur S, Keshavan M, et al. Neuropsychopharmacology "
        "2008;33(2):298-304. PMID 17443131, bronze OA PDF via the University of "
        "Malta repository, Results and Table 1, week 0",
        15.2, ONCE_DAILY, (24.0, 24.0), "striatum", "[11C]raclopride",
        0.691, 0.152, 14, "Mamo 2008 oral baseline, n=14", 37.4, "full text",
        "Lilly study F1D-EW-HGJW, patients in Toronto, Pittsburgh and New York "
        "in the 2000s; de Greef 2011's olanzapine pool is Kapur 1999 (17 points) "
        "and Nyberg 1997 (3 points), 1990s Toronto and Stockholm series",
        "patients stable for at least 4 weeks on oral olanzapine 'dose ... "
        "15.2 +/- 4.8 mg/day', scanned 'proximal to trough plasma "
        "concentrations' before the first depot injection. Once daily is the "
        "ZYPREXA label schedule (NDA 020592, data/fda), not printed by Mamo. "
        "Predicted at the mean dose and at 24 h; plasma 37.4 +/- 31.2 ng/mL, "
        "n=13",
    ),
)


@dataclass(frozen=True)
class ScoredObservation:
    observation: DoseOccupancyObservation
    predicted: float
    predicted_plasma_ng_mL: float

    @property
    def error(self):
        return self.predicted - self.observation.observed

    @property
    def error_over_sd(self):
        sd = self.observation.observed_sd
        return None if sd is None else self.error / sd

    @property
    def plasma_fold(self):
        """Predicted over observed plasma, or ``None`` if none was printed."""
        observed = self.observation.observed_plasma_ng_mL
        return None if observed is None else self.predicted_plasma_ng_mL / observed


def _receptor_kwargs(drug):
    if drug not in EC50_ROWS:
        return {}
    return dict(
        receptor_record=d2_occupancy_parameter_from_ec50(EC50_ROWS[drug]),
        kp_uu_record=KP_UU_NOT_REQUIRED[drug],
    )


def ec50_row(drug):
    """The EC50 row the chain is driven with for ``drug``."""
    return EC50_ROWS[drug] if drug in EC50_ROWS else d2_plasma_ec50(drug)


def _plasma_at(times_min, plasma_nM, hours, mw):
    """Plasma in ng/mL averaged over the endpoints of ``hours``, as the PET
    sources average start-of-scan and end-of-scan samples."""
    values = [np.interp(h * 60.0, times_min, plasma_nM) for h in hours]
    return float(np.mean(values)) * mw / 1000.0


def predict_observation(observation):
    """The chain's prediction for one observation."""
    drug = observation.drug
    row = ec50_row(drug)
    kwargs = _receptor_kwargs(drug)
    if observation.regimen == SINGLE_DOSE:
        times = np.arange(0.0, 24.0 * 60.0 + 5.0, 5.0)
        curve = predict_occupancy(
            observation.dose_mg, drug=drug, times_min=times,
            regimen=SINGLE_DOSE, **kwargs
        )
        times, plasma_nM = curve.times_min, curve.plasma_nM
        emax, ec50_nM = curve.emax, curve.ec50_nM
    else:
        state = steady_state_occupancy(observation.dose_mg, drug=drug, **kwargs)
        # One interval, with the next dose's trough appended at 24 h: the
        # superposed curve at t = 0 is the concentration just before a dose.
        times = np.append(state.times_min, 24.0 * 60.0)
        plasma_nM = np.append(state.plasma_nM, state.plasma_nM[0])
        emax, ec50_nM = state.emax, state.ec50_nM
    plasma_ng_mL = _plasma_at(times, plasma_nM, observation.scan_h, row.molar_mass_g_per_mol)
    concentration_nM = plasma_ng_mL * 1000.0 / row.molar_mass_g_per_mol
    predicted = emax * concentration_nM / (concentration_nM + ec50_nM)
    return ScoredObservation(observation, float(predicted), plasma_ng_mL)


@dataclass(frozen=True)
class DrugScore:
    drug: str
    scored: tuple
    n_independent_groups: int

    @property
    def mae(self):
        return float(np.mean([abs(s.error) for s in self.scored]))

    @property
    def bias(self):
        return float(np.mean([s.error for s in self.scored]))

    @property
    def passed(self):
        return self.mae <= PASS_MAE_SAME_TRACER

    @property
    def plasma_folds(self):
        return tuple(s.plasma_fold for s in self.scored if s.plasma_fold is not None)


def score_drug(drug):
    scored = tuple(predict_observation(o) for o in OBSERVATIONS if o.drug == drug)
    if not scored:
        raise ValueError("no held-out dose-occupancy observation for %r" % drug)
    groups = {s.observation.independent_group for s in scored}
    return DrugScore(drug, scored, len(groups))


SCORED_DRUGS = ("aripiprazole", "quetiapine", "olanzapine")


def score_all():
    return {drug: score_drug(drug) for drug in SCORED_DRUGS}


def haloperidol_v4():
    """Haloperidol's whole-chain comparison, V4, recomputed and not re-scored.

    No further readable dose-occupancy observation was found that is disjoint
    from haloperidol's EC50 sources (P22.3's searched-and-empty list). Kapur
    1996 is a range, so the comparison is containment, and it is not blind.
    """
    return kapur1996_containment()


@dataclass(frozen=True)
class UnscoredCandidate:
    """A PET source that looked usable and was not scored, with the reason."""

    drug: str
    source: str
    reason: str
    readings: str = ""


def _ambiguous_schedule_readings(dose_mg_per_day, hours):
    """Quetiapine occupancy at ``hours`` after the last dose under once-daily
    and twice-daily dosing. Shown for sources that do not print the schedule,
    so that leaving them unscored hides nothing."""
    row = QUETIAPINE_D2_PLASMA_EC50
    out = {}
    once = steady_state_occupancy(dose_mg_per_day, drug="quetiapine", **_receptor_kwargs("quetiapine"))
    half = steady_state_occupancy(dose_mg_per_day / 2.0, drug="quetiapine", **_receptor_kwargs("quetiapine"))
    # Twice daily is the once-daily curve of half the dose superposed with
    # itself shifted 12 h, exact because the PK is linear.
    shift = int(round(12.0 * 60.0 / 5.0))
    twice = half.plasma_nM + np.roll(half.plasma_nM, shift)
    for label, plasma in (("once daily", once.plasma_nM), ("twice daily", twice)):
        index = int(round(hours * 60.0 / 5.0)) % len(plasma)
        c = plasma[index]
        out[label] = float(c / (c + row.ec50_nM))
    return out


def unscored_candidates():
    gefvert = _ambiguous_schedule_readings(450.0, 2.0)
    kapur = {d: _ambiguous_schedule_readings(d, 12.0) for d in (150.0, 600.0)}
    return (
        UnscoredCandidate(
            "quetiapine",
            "Gefvert 1998, Psychopharmacology 135:119, PMID 9497016",
            "closed (Unpaywall and OpenAlex: closed); the abstract prints 450 mg "
            "and 44% at 2 h after the last dose but not the dosing schedule, and "
            "the two schedules predict differently",
            "once daily %.3f, twice daily %.3f, observed 0.44"
            % (gefvert["once daily"], gefvert["twice daily"]),
        ),
        UnscoredCandidate(
            "quetiapine",
            "Kapur 2000, Arch Gen Psychiatry 57:553, PMID 10839333",
            "closed (JAMA 403, Unpaywall closed); the abstract prints only a range "
            "of 0-27% at 12-14 h across 150-600 mg/d and no schedule",
            "at 12 h, 150 mg/d: once daily %.3f, twice daily %.3f; 600 mg/d: once "
            "daily %.3f, twice daily %.3f"
            % (kapur[150.0]["once daily"], kapur[150.0]["twice daily"],
               kapur[600.0]["once daily"], kapur[600.0]["twice daily"]),
        ),
        UnscoredCandidate(
            "quetiapine",
            "Gefvert 2001, Eur Neuropsychopharmacol 11:105, PMID 11313155",
            "closed; the abstract prints 41% at 750 and 30% at 450 mg/day with "
            "neither scan time nor schedule",
        ),
        UnscoredCandidate(
            "quetiapine",
            "Mamo 2008, J Clin Psychiatry 69:81, PMID 18312041",
            "closed; the abstract prints no occupancy value",
        ),
        UnscoredCandidate(
            "quetiapine",
            "Nord 2011, PMC3198174",
            "the source of the constant, so not held out",
        ),
        UnscoredCandidate(
            "quetiapine",
            "Vernaleken 2010 (PMID 20392299), Kessler 2006 (PMID 16738543)",
            "[18F]fallypride, a different tracer from the constant; doses and "
            "scan times not in the abstracts",
        ),
        UnscoredCandidate(
            "olanzapine",
            "Kapur 1998, Am J Psychiatry 155:921, PMID 9659858",
            "not disjoint: its patients are inside Kapur 1999, which is in de "
            "Greef 2011's olanzapine pool; the abstract prints only a 43-80% "
            "range over 5-20 mg/day",
        ),
        UnscoredCandidate(
            "olanzapine",
            "Kessler 2005, PMID 16123775",
            "[18F]fallypride, a different tracer; no dose in the abstract",
        ),
        UnscoredCandidate(
            "olanzapine",
            "Raedler 1999 (PMID 10482380), Lavalaye 1999 (PMID 10688158), "
            "Bernardo 2001 (PMID 11530275), Attarbaschi 2007 (PMID 16945507)",
            "[123I]IBZM SPECT, a different tracer and modality from de Greef's "
            "[11C]raclopride constant",
        ),
        UnscoredCandidate(
            "aripiprazole",
            "Kim 2012, PMC3318151",
            "the source of the constant, so not held out",
        ),
        UnscoredCandidate(
            "aripiprazole",
            "Yokoi 2002 (PMID 12093598), Mamo 2007 (PMID 17728427)",
            "repeated dosing: the single-dose clearance of the active-metabolite "
            "gate does not apply, and no steady-state metabolite constant exists",
        ),
    )


@dataclass(frozen=True)
class RefusedDrug:
    """A census drug that cannot run end to end, and the layer that stops it."""

    drug: str
    missing_layer: str
    reason: str


REFUSED = (
    RefusedDrug(
        "clozapine", "pk",
        "Li 2012 co-fits norclozapine and prints no reference weight; Reeves 2024 "
        "co-fits, needs a parallel-elimination slot and fails V13 (3.07x). The "
        "EC50 (Gruender 2006) and the occupancy-outcome test (V11) exist",
    ),
    RefusedDrug(
        "risperidone", "active-metabolite",
        "every reachable EC50 is on the active moiety and every adult oral popPK "
        "co-fits 9-OH-risperidone (P21.4, P25.1); the parent axis does not exist",
    ),
    RefusedDrug(
        "paliperidone", "pk",
        "Korell 2017 has a zero-order release phase and no reference creatinine "
        "clearance, both refused by the P25.2 bridge",
    ),
    RefusedDrug(
        "ziprasidone", "pk",
        "no readable adult oral popPK in either corpus (P21.3, P25.1)",
    ),
    RefusedDrug(
        "amisulpride", "pk",
        "no oral popPK record in the registry; it was outside P25.1's search",
    ),
)


def summary_rows():
    """One row per drug, as pasted into the design document and the ledger."""
    rows = []
    v4 = haloperidol_v4()
    rows.append(
        "haloperidol | Kapur 1996 2 mg/day range %.2f-%.2f | steady-state mean "
        "%.4f, contained | V4, not blind"
        % (v4.range_low, v4.range_high, v4.steady_state_mean_occupancy)
    )
    for drug, score in score_all().items():
        rows.append(
            "%s | n=%d (%d groups) | MAE %.4f bias %+.4f | %s"
            % (drug, len(score.scored), score.n_independent_groups, score.mae,
               score.bias, "pass" if score.passed else "fail")
        )
    return tuple(rows)
