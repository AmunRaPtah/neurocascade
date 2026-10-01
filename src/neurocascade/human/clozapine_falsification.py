"""P22.4: the clozapine anomaly as a falsification test of occupancy-as-sole-mediator.

Clozapine is the most effective antipsychotic available and is clinically
effective at striatal D2 occupancy well below the 65 percent response threshold
this repository sources from Kapur 2000 (``CLINICAL_OCCUPANCY_THRESHOLDS`` in
:mod:`neurocascade.human.occupancy_validation`). The pre-registration
(``docs/p22-preregistration.md``, section P22.4) fixes, before any number here
was computed, the mapping from outcome to reading: if the untuned chain places
clozapine's 65 percent dose above the doses at which clozapine is clinically
effective, that is **not a bug** -- it is the model reproducing a real anomaly
that falsifies occupancy-threshold-as-the-sole-mediator of response.

Why this module is self-contained, and does not touch the live registry
-----------------------------------------------------------------------

The pre-registration's literal procedure computes ``D65`` as a *dose*, "by the
identical procedure as P22.2", and the predicted occupancy at the label dose
range. That procedure needs a dose->plasma PK bridge for clozapine, and the
chain does not have one: P21.3 recorded clozapine population-PK as
searched-and-empty (VERSACLOZ NDA203479 performed no popPK analysis; the older
applications predate electronic review posting), no clozapine PK set is
hand-transcribed in :mod:`neurocascade.human.pk_parameters`, and
:func:`neurocascade.human.fda_pk_bridge.refusals` has nothing to bridge. So the
PK layer refuses clozapine, exactly as P22.2 (ledger row V8) recorded. That
refusal is reported here as the pre-registered dose-domain outcome
(:func:`dose_domain_refusal`), under commitment C2 (no silent drops).

The anomaly, though, is a statement about **plasma concentration and occupancy**,
not about dose. Nordstrom 1995, Gruender 2006 and Kapur 2000 all read it at the
concentration level, and clozapine is dosed clinically to a *plasma* target
(therapeutic drug monitoring) rather than to a fixed mg/day precisely because
dose->plasma varies more than 45-fold between patients (Potkin 1994). So the
falsification is run in the concentration domain, which needs no PK bridge and
is strictly the sharper form of the test. This is recorded as a deviation from
the pre-registration (its "Deviations" clause under C1), with the dose-domain
layer refusal reported alongside rather than hidden.

Registering clozapine's EC50 into the live ``D2_PLASMA_EC50`` dict would also
retroactively change the P22.2 (V8) and P22.3 clozapine records, which committed
clozapine as refused / searched-and-empty before this task ran; C1 forbids that.
clozapine is additionally the suite's canonical "unknown drug" fixture for the
refusal-layer tests. So the EC50 records live here, built with the *same*
schema-validated :class:`D2PlasmaEC50` dataclass the registry uses ("ingested
under P22.2/P22.3's rules"), without being inserted into the chain.

Nothing in this module is tuned to make clozapine fit, in either direction (C1).
"""
from dataclasses import dataclass

from .d2_ec50 import D2PlasmaEC50

#: Clozapine molar mass, from the Clozapine USP Orally Disintegrating Tablets
#: label DESCRIPTION (ANDA090308): "C 18 H 19 ClN 4 M.W. 326.82". The VERSACLOZ
#: NDA203479 snapshot this task's label ground truth comes from does not itself
#: print the molecular weight, so a cited generic label supplies it. Used only
#: to express the ng/mL EC50 on the nM axis the schema stores; every comparison
#: below is computed in ng/mL, the unit the sources and the thresholds print.
CLOZAPINE_MW_G_PER_MOL = 326.82


def _ng_per_mL_to_nM(ng_per_mL):
    return ng_per_mL / CLOZAPINE_MW_G_PER_MOL * 1000.0


#: The two occupancy targets, identical to P22.2's: the sourced clinical-window
#: thresholds in ``CLINICAL_WINDOW_PROVENANCE`` (response 0.65, Kapur 2000; EPS
#: 0.78, Kapur 2000, with 78 used rather than the synthesized 80). These are
#: striatal measurements, which is why the striatal (putamen) EC50 is the
#: like-for-like primary and the extrastriatal cortical row is sensitivity only.
TARGET_LOW = 0.65
TARGET_HIGH = 0.78


# ---------------------------------------------------------------------------
# clozapine's registered EC50 records, schema-validated, held outside the chain
# ---------------------------------------------------------------------------

#: Primary. Gruender 2006 putamen -- the striatal, like-for-like constant, since
#: the 0.65/0.78 thresholds are striatal. Emax "approached 100%" (fitted free,
#: not pinned); the abstract prints no precise fraction, so 1.0 is recorded as
#: that stated approach, and it is the choice *least* favourable to the anomaly:
#: a lower Emax raises the concentration needed for 65% occupancy, or puts it out
#: of reach entirely (see the Uchida free-Emax sensitivity row). The conclusion
#: therefore holds a fortiori at the registered Emax.
CLOZAPINE_D2_EC50_PUTAMEN = D2PlasmaEC50(
    drug="clozapine",
    ec50_nM=_ng_per_mL_to_nM(950.0),
    molar_mass_g_per_mol=CLOZAPINE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent clozapine",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fitted-free",
    region="putamen",
    tracer="[18F]fallypride",
    regimen="oral, steady state, inpatients with schizophrenia",
    provenance="primary",
    citation=(
        "Gruender G, Landvogt C, Vernaleken I, et al. The striatal and "
        "extrastriatal D2/D3 receptor-binding profile of clozapine in patients "
        "with schizophrenia. Neuropsychopharmacology 2006;31(5):1027-35. PMID "
        "16237387. Abstract: 'the maximum attainable receptor occupancy Emax "
        "approached 100% both in the striatum and cortex, the plasma "
        "concentration at 50% of Emax (ED50) was much higher in the putamen "
        "(950 ng/ml) than in the inferior temporal cortex (333 ng/ml)'."
    ),
    notes=(
        "15 clozapine-treated inpatients, [18F]fallypride, occupancy as percent "
        "reduction in binding potential against seven normal volunteers. The "
        "fallypride family is D2/D3 and does not separate them, so this is a "
        "D2/D3 composite. Emax recorded 1.0 for 'approached 100%'; the abstract "
        "prints no precise fitted fraction, and 1.0 is conservative against the "
        "anomaly (a lower ceiling only strengthens Outcome A)."
    ),
)

#: Sensitivity -- Gruender 2006 caudate, the same dataset's other striatal
#: region. Printed as 582 ng/mL in the Spangemacher 2025 restatement (the
#: abstract prints only putamen and inferior temporal cortex).
CLOZAPINE_D2_EC50_CAUDATE = D2PlasmaEC50(
    drug="clozapine",
    ec50_nM=_ng_per_mL_to_nM(582.0),
    molar_mass_g_per_mol=CLOZAPINE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent clozapine",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fitted-free",
    region="caudate",
    tracer="[18F]fallypride",
    regimen="oral, steady state, inpatients with schizophrenia",
    provenance="second-hand",
    citation="Gruender 2006, PMID 16237387, caudate ED50 582 ng/mL",
    read_in=(
        "Spangemacher M, et al. 2025, PMC12488982: 'EC50 values of 950 ng/mL "
        "for the putamen and 582 ng/mL for the caudate [38]', [38] being "
        "Gruender 2006"
    ),
    notes="Caudate value quoted second-hand; the Gruender abstract prints only putamen and cortex.",
)

#: Sensitivity -- Gruender 2006 inferior temporal cortex, EXTRASTRIATAL. Not
#: like-for-like with the striatal 0.65/0.78 thresholds, so it never decides the
#: headline; it is the quantitative form of Gruender's own thesis that clozapine
#: binds extrastriatal D2/D3 preferentially, and it is the one candidate that
#: lands near the threshold rather than far above it.
CLOZAPINE_D2_EC50_CORTEX = D2PlasmaEC50(
    drug="clozapine",
    ec50_nM=_ng_per_mL_to_nM(333.0),
    molar_mass_g_per_mol=CLOZAPINE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent clozapine",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fitted-free",
    region="inferior temporal cortex",
    tracer="[18F]fallypride",
    regimen="oral, steady state, inpatients with schizophrenia",
    provenance="primary",
    citation="Gruender 2006, PMID 16237387, inferior temporal cortex ED50 333 ng/mL (in the abstract)",
    notes=(
        "Extrastriatal: reported as sensitivity only. The 0.65/0.78 thresholds "
        "are striatal, so a cortical EC50 is a different measured quantity. "
        "Gruender: cortical occupancy 'approaches 60% with plasma clozapine in "
        "the range 350-400 ng/ml, which corresponds to the threshold for "
        "antipsychotic efficacy'."
    ),
)

#: Sensitivity -- Uchida 2011's pooled plasma fit, free-Emax parameterisation:
#: EC50 105 ng/mL with Emax 60%. Second-hand (the pooled paper is paywalled; the
#: pair is quoted in open secondaries). Under this fit 65% occupancy is
#: unreachable at ANY plasma concentration, because the fitted ceiling is 0.60.
CLOZAPINE_D2_EC50_UCHIDA_FREE = D2PlasmaEC50(
    drug="clozapine",
    ec50_nM=_ng_per_mL_to_nM(105.0),
    molar_mass_g_per_mol=CLOZAPINE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent clozapine",
    analyte_is_summed=False,
    emax=0.60,
    emax_convention="fitted-free",
    region="striatum",
    tracer="[11C]raclopride",
    regimen="pooled multi-study oral fit",
    provenance="second-hand",
    citation=(
        "Uchida H, et al. Predicting dopamine D2 receptor occupancy from plasma "
        "levels of antipsychotic drugs. J Clin Psychopharmacol 2011;31(3):318-25. "
        "PMID 21508857. Clozapine ED50 105 ng/mL, Emax 60% (unconstrained)."
    ),
    read_in="Spangemacher 2025, PMC12488982 (the primary pooled paper is paywalled)",
    notes="Raclopride-based striatal pool; Emax fitted free to 60%, so 65% is above the ceiling.",
)

#: Sensitivity -- Uchida 2011's Emax-fixed-at-100% parameterisation of the same
#: data: EC50 483 ng/mL. The 4.6-fold span between this and the free-Emax row,
#: on the Emax convention alone, is the span the pre-registration flagged.
CLOZAPINE_D2_EC50_UCHIDA_FIXED = D2PlasmaEC50(
    drug="clozapine",
    ec50_nM=_ng_per_mL_to_nM(483.0),
    molar_mass_g_per_mol=CLOZAPINE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent clozapine",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fixed-at-1",
    region="striatum",
    tracer="[11C]raclopride",
    regimen="pooled multi-study oral fit",
    provenance="second-hand",
    citation="Uchida 2011, PMID 21508857. Clozapine ED50 483 ng/mL, Emax fixed 100%.",
    read_in="Spangemacher 2025, PMC12488982 (the primary pooled paper is paywalled)",
    notes="Same pool as the free-Emax row, re-parameterised with the ceiling pinned at 1.",
)

CLOZAPINE_D2_EC50_PRIMARY = CLOZAPINE_D2_EC50_PUTAMEN

#: Every candidate reported as sensitivity, none chosen after the fact.
CLOZAPINE_D2_EC50_SENSITIVITY = (
    CLOZAPINE_D2_EC50_CAUDATE,
    CLOZAPINE_D2_EC50_CORTEX,
    CLOZAPINE_D2_EC50_UCHIDA_FREE,
    CLOZAPINE_D2_EC50_UCHIDA_FIXED,
)


# ---------------------------------------------------------------------------
# the ground truth: plasma concentrations at which clozapine is effective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClinicalPlasmaConcentration:
    """A clozapine plasma concentration (range) a source tied to clinical effect.

    This is the ground-truth side of the falsification, and none of it was
    chosen in this repository: these are external response thresholds and the
    community therapeutic-drug-monitoring reference range.
    """

    label: str
    low_ng_per_mL: float
    high_ng_per_mL: float
    measure: str
    source: str

    def __post_init__(self):
        if not 0 < self.low_ng_per_mL <= self.high_ng_per_mL:
            raise ValueError("%s: need 0 < low <= high" % self.label)


#: The primary ground-truth range: the AGNP consensus therapeutic reference
#: range, the plasma window clozapine is dosed to clinically. Sourced
#: independently of Gruender 2006 (the EC50 source) to avoid circularity.
THERAPEUTIC_REFERENCE_RANGE = (350.0, 600.0)

CLOZAPINE_EFFECTIVE_PLASMA = (
    ClinicalPlasmaConcentration(
        label="AGNP therapeutic reference range",
        low_ng_per_mL=350.0,
        high_ng_per_mL=600.0,
        measure="recommended therapeutic plasma range for schizophrenia",
        source=(
            "Hiemke C, et al. Consensus Guidelines for Therapeutic Drug "
            "Monitoring in Neuropsychopharmacology: Update 2017. "
            "Pharmacopsychiatry 2018;51(1-02):9-62. PMID 28910830. Clozapine "
            "350-600 ng/mL."
        ),
    ),
    ClinicalPlasmaConcentration(
        label="Potkin 1994 response threshold",
        low_ng_per_mL=420.0,
        high_ng_per_mL=420.0,
        measure=(
            "blood concentration optimally distinguishing responders from "
            "non-responders (8% response below, 60% above), fixed 400 mg/day"
        ),
        source=(
            "Potkin SG, et al. Plasma clozapine concentrations predict clinical "
            "response in treatment-resistant schizophrenia. J Clin Psychiatry "
            "1994;55 Suppl B:133-6. PMID 7961557."
        ),
    ),
    ClinicalPlasmaConcentration(
        label="VanderZwaag 1996 high serum range",
        low_ng_per_mL=350.0,
        high_ng_per_mL=450.0,
        measure=(
            "the highest of three randomised serum ranges (low 50-150, medium "
            "200-300, high 350-450 ng/mL); response best in the high range"
        ),
        source=(
            "VanderZwaag C, et al. Response of patients with "
            "treatment-refractory schizophrenia to clozapine within three serum "
            "level ranges. Am J Psychiatry 1996;153(12):1579-84. PMID 8942454."
        ),
    ),
    ClinicalPlasmaConcentration(
        label="Oyewumi 1995 putative therapeutic threshold",
        low_ng_per_mL=350.0,
        high_ng_per_mL=350.0,
        measure="putative threshold therapeutic concentration",
        source=(
            "Oyewumi LK, et al. Can low-dose clozapine pharmacokinetics predict "
            "steady-state plasma concentration? Ther Drug Monit 1995;17(2):137-41. "
            "PMID 7624901. 'putative threshold therapeutic concentration of 350 "
            "micrograms/L'."
        ),
    ),
)


@dataclass(frozen=True)
class VersaclozSteadyStatePK:
    """VERSACLOZ's own label-stated steady-state plasma concentrations.

    A second, in-repo-pinned corroboration that needs no PK model: the FDA
    snapshot states what plasma clozapine the label dose range actually reaches.
    """

    dose_text: str
    peak_mean_ng_per_mL: float
    peak_low_ng_per_mL: float
    peak_high_ng_per_mL: float
    trough_mean_ng_per_mL: float
    trough_low_ng_per_mL: float
    trough_high_ng_per_mL: float
    source: str


#: Transcribed from the pinned VERSACLOZ NDA203479 DOSAGE/CLINICAL PHARMACOLOGY
#: snapshot (data/fda/, SHA-256 in fda_sources). Pooled across the label dose
#: range, so the highest *observed* peak (723 ng/mL) is the relevant extreme.
VERSACLOZ_STEADY_STATE_PK = VersaclozSteadyStatePK(
    dose_text="100 mg to 800 mg VERSACLOZ once daily",
    peak_mean_ng_per_mL=275.0,
    peak_low_ng_per_mL=105.0,
    peak_high_ng_per_mL=723.0,
    trough_mean_ng_per_mL=75.0,
    trough_low_ng_per_mL=11.0,
    trough_high_ng_per_mL=198.0,
    source=(
        "VERSACLOZ label, NDA203479, pinned snapshot: 'Following oral "
        "administration of 100 mg to 800 mg VERSACLOZ, once daily, the average "
        "steady-state peak plasma concentration was 275 ng/mL (range: 105 - 723 "
        "ng/mL)... The average minimum concentration at steady-state was 75 "
        "ng/mL (range: 11 - 198 ng/mL).'"
    ),
)

#: The VERSACLOZ label dose range, under the P22.2 extraction rule, for the
#: dose-domain refusal record only. Target 150-225 mg twice per day = 300-450
#: mg/day at the low end; maximum 450 mg twice daily = 900 mg/day.
VERSACLOZ_LABEL_DOSE_MG_PER_DAY = (300.0, 900.0)


# ---------------------------------------------------------------------------
# the falsification
# ---------------------------------------------------------------------------


def plasma_for_occupancy(record, target_occupancy):
    """Plasma concentration (ng/mL) reaching ``target_occupancy`` under ``record``.

    Inverts the record's own hyperbola ``occ = Emax * C / (C + EC50)`` for C.
    Returns ``None`` when the target is at or above the fit's Emax, which means
    it is unreachable at any concentration -- the strongest form of the anomaly,
    not an error.
    """
    if not 0 < target_occupancy < 1:
        raise ValueError("target_occupancy must be a fraction strictly inside (0, 1)")
    if target_occupancy >= record.emax:
        return None
    return record.ec50_ng_per_mL * target_occupancy / (record.emax - target_occupancy)


@dataclass(frozen=True)
class FalsificationResult:
    """The P22.4 outcome for one EC50 record, with its fixed reading."""

    region: str
    ec50_ng_per_mL: float
    emax: float
    emax_convention: str
    c65_ng_per_mL: float | None
    c78_ng_per_mL: float | None
    effective_low_ng_per_mL: float
    effective_high_ng_per_mL: float
    fold_c65_over_effective_high: float | None
    occupancy_at_effective_low: float
    occupancy_at_effective_high: float
    outcome: str
    reading: str
    is_extrastriatal: bool


def _occupancy_at(record, ng_per_mL):
    return record.emax * ng_per_mL / (ng_per_mL + record.ec50_ng_per_mL)


def clozapine_falsification(record=None, effective_range=None):
    """Run the fixed-mapping falsification for one EC50 record.

    ``effective_range`` is the clinically-effective plasma window to compare
    against; it defaults to the AGNP therapeutic reference range. Outcome A
    (anomaly reproduced, hypothesis falsified) when the 65% concentration does
    not exist or exceeds the top of the effective range; Outcome B otherwise. A
    tie (C65 exactly equal to the top) resolves to B, the reading against the
    more interesting result, per the pre-registration.
    """
    if record is None:
        record = CLOZAPINE_D2_EC50_PRIMARY
    low, high = effective_range or THERAPEUTIC_REFERENCE_RANGE
    c65 = plasma_for_occupancy(record, TARGET_LOW)
    c78 = plasma_for_occupancy(record, TARGET_HIGH)
    is_extrastriatal = record.region not in (None, "putamen", "caudate", "striatum")

    if c65 is None:
        outcome = "A"
        fold = None
        reading = (
            "Outcome A: %.0f%% occupancy is unreachable at any plasma "
            "concentration under this fit (Emax %.2f). The hypothesis that a "
            "0.65-0.78 striatal occupancy window mediates clozapine's response "
            "is falsified: clozapine is clinically effective at %g-%g ng/mL, a "
            "range it reaches, while the window it is supposed to need does not "
            "exist for this drug at all."
            % (100 * TARGET_LOW, record.emax, low, high)
        )
    else:
        fold = c65 / high
        if c65 > high:
            outcome = "A"
            reading = (
                "Outcome A: reaching %.0f%% striatal occupancy needs %.0f ng/mL, "
                "%.2fx above the top of the clinically-effective plasma range "
                "(%g-%g ng/mL). The untuned chain places clozapine's 65%% "
                "concentration above the concentrations at which clozapine "
                "works, so it reproduces the real anomaly and falsifies "
                "occupancy-threshold-as-sole-mediator for clozapine."
                % (100 * TARGET_LOW, c65, fold, low, high)
            )
        else:
            outcome = "B"
            reading = (
                "Outcome B: %.0f%% occupancy is reached at %.0f ng/mL, within or "
                "below the clinically-effective plasma range (%g-%g ng/mL). The "
                "anomaly does not appear through this fit; this contradicts the "
                "received PET literature and is an unresolved disagreement, not a "
                "vindication of the threshold." % (100 * TARGET_LOW, c65, low, high)
            )
    if is_extrastriatal:
        reading += (
            " [EXTRASTRIATAL: this is a cortical EC50, not the striatal quantity "
            "the 0.65/0.78 thresholds were measured as; sensitivity only.]"
        )
    return FalsificationResult(
        region=record.region,
        ec50_ng_per_mL=record.ec50_ng_per_mL,
        emax=record.emax,
        emax_convention=record.emax_convention,
        c65_ng_per_mL=c65,
        c78_ng_per_mL=c78,
        effective_low_ng_per_mL=low,
        effective_high_ng_per_mL=high,
        fold_c65_over_effective_high=fold,
        occupancy_at_effective_low=_occupancy_at(record, low),
        occupancy_at_effective_high=_occupancy_at(record, high),
        outcome=outcome,
        reading=reading,
        is_extrastriatal=is_extrastriatal,
    )


def sensitivity_results():
    """The primary result followed by every sensitivity record's result."""
    return tuple(
        clozapine_falsification(record)
        for record in (CLOZAPINE_D2_EC50_PRIMARY,) + CLOZAPINE_D2_EC50_SENSITIVITY
    )


def dose_domain_refusal():
    """Why the pre-registered dose-domain procedure cannot be run (C2).

    The literal pre-registration computes D65 as a dose and the occupancy at the
    label dose range, both of which need a clozapine dose->plasma PK bridge the
    chain does not have. This states that refusal rather than letting the
    dose-domain test silently disappear from the P22.4 denominator.
    """
    return (
        "pk: D65 as a dose, and the occupancy at the VERSACLOZ label dose range "
        "(%g-%g mg/day), are not computable: clozapine has no accepted oral PK "
        "record. P21.3 recorded clozapine population-PK as searched-and-empty "
        "(VERSACLOZ NDA203479 performed no popPK analysis), no clozapine set is "
        "hand-transcribed, and the FDA bridge has nothing to convert. P22.2 "
        "(ledger V8) recorded the same pk refusal. The falsification is run in "
        "the concentration domain instead (clozapine_falsification), which needs "
        "no PK bridge and is the sharper form of the test."
        % VERSACLOZ_LABEL_DOSE_MG_PER_DAY
    )
