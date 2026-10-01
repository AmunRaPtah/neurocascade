"""P22.3: leave-one-drug-out against published PET occupancy.

The test pre-registered in ``docs/p22-preregistration.md`` (section P22.3). For
each drug X in the nine-drug P21.2 census, X's own sourced plasma EC50 and Emax
predict ``occ = Emax * C / (C + EC50)`` at each concentration a *different*
study measured alongside an occupancy, and nothing is refitted on X. The
result, and every source that was searched and rejected, is in
``docs/p22-3-leave-one-drug-out.md``.

What the search found shapes this module more than the design does:

* **Two drugs are scored.** Aripiprazole (Ito 2012, three dose groups, caudate
  and putamen) and ziprasidone (one subject, linked across Bench 1993 and
  Bench 1996). Those are the only eligible observations, under the
  pre-registered rules, that a full text this host could open actually prints.
* **Seven drugs are searched-and-empty on the observation side.** In every case
  the paired numbers exist in a paper that is paywalled or bot-gated, or the
  readable paper prints occupancy and concentration without pairing them. The
  record of what was opened and why it failed is :data:`SEARCHED_AND_EMPTY`.

The EC50 rows for aripiprazole and ziprasidone are held here, not added to
:data:`~neurocascade.human.d2_ec50.D2_PLASMA_EC50`. That registry drives the
dose-to-occupancy chain, and P22.2's pinned result attributes refusals by
asking it which drugs have a constant. Adding a row there would silently
rewrite a published result. Neither drug has chain PK, so the whole-chain
comparison the pre-registration allows is refused for both, and the reason is
recorded per drug.
"""
from dataclasses import dataclass

from .d2_ec50 import D2PlasmaEC50

#: The nine-drug P21.2 census, fixed by the pre-registration.
DENOMINATOR = (
    "haloperidol",
    "risperidone",
    "olanzapine",
    "clozapine",
    "ziprasidone",
    "amisulpride",
    "aripiprazole",
    "paliperidone",
    "quetiapine",
)

#: Pre-registered pass thresholds, occupancy units. A tie passes.
PASS_MAE_SAME_TRACER = 0.10
PASS_MAE_CROSS_TRACER = 0.15
#: Below this n an aggregate is flagged anecdotal.
ANECDOTAL_BELOW_N = 3

#: Molar masses, read from the FDA label snapshots in ``data/fda``.
ARIPIPRAZOLE_MW_G_PER_MOL = 448.38
ZIPRASIDONE_MW_G_PER_MOL = 412.94

ARIPIPRAZOLE_D2_PLASMA_EC50 = D2PlasmaEC50(
    drug="aripiprazole",
    ec50_nM=11.1 / ARIPIPRAZOLE_MW_G_PER_MOL * 1000.0,
    molar_mass_g_per_mol=ARIPIPRAZOLE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent aripiprazole",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fixed-at-1",
    region="striatum (putamen and caudate nucleus)",
    tracer="[11C]raclopride",
    regimen="single oral dose 2, 5, 10 or 30 mg, healthy men, scans at 3, 45 and 120 h",
    provenance="primary",
    citation=(
        "Kim E, Howes OD, Kim BH, et al. J Cereb Blood Flow Metab "
        "2012;32(4):759-68. PMID 22186667, PMC3318151: 'EC 50 derived from PD "
        "modeling alone=11.1 ng/mL (95% confidence interval (CI)=10.1 to 12.1)'"
    ),
    notes=(
        "The plasma (PD-alone) EC50, not the 8.63 ng/mL effect-site value, "
        "because the comparison is against plasma concentrations. Emax is the "
        "paper's 'maximum occupancy (100% of receptors occupied by drug)'. "
        "Molar mass from the OPIPZA label (NDA216655, effective_time 20250130). "
        "Seoul National University Hospital subjects."
    ),
)

ZIPRASIDONE_D2_PLASMA_EC50 = D2PlasmaEC50(
    drug="ziprasidone",
    ec50_nM=204.0,
    molar_mass_g_per_mol=ZIPRASIDONE_MW_G_PER_MOL,
    plasma_basis="total",
    analyte="parent ziprasidone (serum)",
    analyte_is_summed=False,
    emax=1.0,
    emax_convention="fixed-at-1",
    region="striatum",
    tracer="[11C]raclopride",
    regimen="oral 60 mg twice daily with food, steady state, patients; scans 5, 13 and 23 h post-dose",
    provenance="primary",
    citation=(
        "Suzuki T, Graff-Guerrero A, Uchida H, et al. Psychopharmacology "
        "2013;228(1):43-51. PMID 23417515, PMC3679209: '204 nmol/L (84 ng/ml; "
        "95% confidence interval: 177-231 nmol/L; R2=0.68)'"
    ),
    interval_low_nM=177.0,
    interval_high_nM=231.0,
    interval_method="95% confidence interval as printed by the source",
    notes=(
        "The headline striatal value; the source also prints putamen 174, "
        "caudate 205 and ventral striatum 190 nmol/L. 'X was fixed to 100'. "
        "Molar mass is the free base, from the ziprasidone label (ANDA077561, "
        "effective_time 20260811), so 204 nM is 84.24 ng/mL against the "
        "source's rounded 84. CAMH Toronto patients, NCT00818298."
    ),
)

#: The EC50 row each scored drug is predicted with.
HELD_OUT_EC50 = {
    ARIPIPRAZOLE_D2_PLASMA_EC50.drug: ARIPIPRAZOLE_D2_PLASMA_EC50,
    ZIPRASIDONE_D2_PLASMA_EC50.drug: ZIPRASIDONE_D2_PLASMA_EC50,
}


@dataclass(frozen=True)
class HeldOutObservation:
    """One measured occupancy with the concentration from the same visit.

    ``tracer`` and ``region`` are as the source printed them, ``None`` when it
    printed none. ``independent_group`` names the subjects behind the value, so
    two regions measured in the same people are not mistaken for two samples.
    """

    drug: str
    source: str
    locator: str
    quote: str
    group: str
    independent_group: str
    n: int
    dose_mg: float
    regimen: str
    population: str
    tracer: str | None
    region: str | None
    observed_occupancy: float
    plasma_ng_per_mL: float
    plasma_basis: str
    analyte: str
    disjoint_because: str
    observed_range: tuple | None = None
    observed_sd: float | None = None
    regimen_matches_ec50_source: bool = True
    notes: str = ""

    def __post_init__(self):
        if not 0.0 <= self.observed_occupancy <= 1.0:
            raise ValueError("observed occupancy must be a fraction")
        if not self.plasma_ng_per_mL > 0.0:
            raise ValueError("plasma concentration must be positive")
        if self.n < 1:
            raise ValueError("n must be at least 1")
        if not self.disjoint_because:
            raise ValueError(
                "an observation must state why its subjects are disjoint from "
                "the EC50 source; undeterminable independence is dependence"
            )


_ITO_2012 = (
    "Ito H, Takano H, Arakawa R, et al. Effects of dopamine D2 receptor partial "
    "agonist antipsychotic aripiprazole on dopamine synthesis in human brain "
    "measured by PET with L-[beta-11C]DOPA. PLoS One 2012;7(9):e46488. "
    "PMID 23029533, PMC3460902"
)
_ITO_PLASMA_QUOTE = (
    "The plasma concentrations of aripiprazole during [11C]raclopride and "
    "L-[beta-11C]DOPA PET studies, averaged between the start and end of each "
    "scanning, were 12.0+/-2.1 ng/mL (mean +/- SD) and 10.4+/-1.5 ng/mL for 3 mg "
    "[...], 29.0+/-2.1 ng/mL and 25.6+/-2.1 ng/mL for 6 mg, and 39.6-40.4 ng/mL "
    "and 38.2-39.7 ng/mL for 9 mg, respectively."
)
_ITO_COMMON = dict(
    drug="aripiprazole",
    source=_ITO_2012,
    regimen="single oral dose, [11C]raclopride scan started 3.5 h post-dose",
    population="healthy men, 23-34 years, National Institute of Radiological Sciences, Chiba",
    tracer="[11C]raclopride",
    plasma_basis="total",
    analyte="parent aripiprazole (dehydroaripiprazole reported separately, 0.4-1.6 ng/mL)",
    disjoint_because=(
        "Chiba (NIRS) healthy volunteers; Kim 2012 studied Seoul National "
        "University Hospital volunteers"
    ),
)
_ITO_9MG_NOTE = (
    "n=2: the printed range endpoints are the two subjects' values, so the mean "
    "of the endpoints is the group mean exactly. Which plasma value goes with "
    "which occupancy is not printed."
)


def _ito(dose, region, occ, rng, sd, plasma, n, note=""):
    return HeldOutObservation(
        locator="Table 1 (occupancy); Results text (plasma)",
        quote=_ITO_PLASMA_QUOTE,
        group="%g mg" % dose,
        independent_group="ito2012-%gmg" % dose,
        n=n,
        dose_mg=dose,
        region=region,
        observed_occupancy=occ,
        observed_range=rng,
        observed_sd=sd,
        plasma_ng_per_mL=plasma,
        notes=note,
        **_ITO_COMMON,
    )


_BENCH = (
    "Bench CJ, Lammertsma AA, Dolan RJ, et al. Dose dependent occupancy of "
    "central dopamine D2 receptors by the novel neuroleptic CP-88,059-01: a "
    "study using positron emission tomography and 11C-raclopride. "
    "Psychopharmacology 1993;112(2-3):308-14. PMID 7871035; and Bench CJ, "
    "Lammertsma AA, Grasby PM, et al. The time course of binding to striatal "
    "dopamine D2 receptors by the neuroleptic ziprasidone (CP-88,059-01) "
    "determined by positron emission tomography. Psychopharmacology "
    "1996;124(1-2):141-7. PMID 8935809. Both read in full at pure.mpg.de"
)

#: Every eligible held-out observation, in source order.
OBSERVATIONS = (
    _ito(3, "caudate", 0.57, (0.53, 0.61), 0.04, 12.0, 5),
    _ito(3, "putamen", 0.55, (0.51, 0.58), 0.02, 12.0, 5),
    _ito(6, "caudate", 0.73, (0.70, 0.77), 0.03, 29.0, 5),
    _ito(6, "putamen", 0.69, (0.66, 0.72), 0.03, 29.0, 5),
    _ito(9, "caudate", 0.78, (0.77, 0.79), None, 40.0, 2, _ITO_9MG_NOTE),
    _ito(9, "putamen", 0.76, (0.75, 0.77), None, 40.0, 2, _ITO_9MG_NOTE),
    HeldOutObservation(
        drug="ziprasidone",
        source=_BENCH,
        locator="Bench 1993 Table 1 subject 7; Bench 1996 Table 1 subject (8)",
        quote=(
            "Bench 1993: '7 34 66 40 518.2 0.51 (+/- 0.03) 77% 77%' (age, kg, "
            "dose mg, AUC, BP, occupancy vs placebo, vs 8 normals). Bench 1996: "
            "'(8) 34 66 40 6 44.6 511 767 432 0.51(+/-0.03)', footnote 'Additional "
            "subject from previous dose-finding study (Bench et al. 1993) who "
            "received 40 mg ziprasidone, 5.75 h post-dose'"
        ),
        group="Bench 1993 subject 7 = Bench 1996 subject (8)",
        independent_group="bench1993-s7",
        n=1,
        dose_mg=40.0,
        regimen="single oral dose, scan about 5-6 h post-dose",
        population="healthy man, 34 years, MRC Cyclotron Unit, London",
        tracer="[11C]raclopride",
        region="striatum",
        observed_occupancy=0.77,
        plasma_ng_per_mL=44.6,
        plasma_basis="total",
        analyte="parent ziprasidone (serum, 'at time of PET')",
        disjoint_because=(
            "London healthy volunteers, 1992-1995; Suzuki 2013 studied CAMH "
            "Toronto patients under NCT00818298"
        ),
        regimen_matches_ec50_source=False,
        notes=(
            "Occupancy is printed in Bench 1993 and is 77% against both of its "
            "references; the serum concentration at the scan is printed only in "
            "Bench 1996. The link is the 1996 footnote plus identical age, weight, "
            "dose and BP. The other Bench 1996 subjects print BP only, and turning "
            "BP into occupancy needs a reference the paper leaves open (placebo "
            "subject 2.67, normals 2.26, age-corrected 2.57), so they are not used."
        ),
    ),
)


@dataclass(frozen=True)
class SearchedAndEmpty:
    """Why a drug has no eligible held-out observation. Commitment C2."""

    drug: str
    ec50_source: str
    searched: tuple
    summary: str


SEARCHED_AND_EMPTY = {
    "haloperidol": SearchedAndEmpty(
        drug="haloperidol",
        ec50_source="fitted in repo on Nyberg 1995 and Regenthal 1997 (P19.4)",
        searched=(
            ("Wolkin 1989, PMID 2785373", "paywalled; the PubMed record carries no abstract"),
            ("Nordstrom 1992, PMID 1533719", "paywalled; abstract prints no plasma value"),
            ("Farde 1988, PMID 2892477", "paywalled; serum data only in figures"),
            ("Lim 2013, PMID 23138261", "paywalled; abstract prints only a fitted EC50 (putamen 0.791 ng/ml)"),
            ("Kessler 2005, PMID 16123775", "abstract prints occupancy, no plasma"),
            ("Reimold 2007, PMID 17111172", "paywalled; abstract gives haloperidol D1 only"),
            ("Arakawa 2010, PMID 20361897", "paywalled; extrastriatal FLB 457; no plasma value"),
            ("Schlosser 1997, PMID 9351492", "binding ratio, plasma only as a group range"),
            ("Kapur 1996, Kapur 2000", "Toronto series, overlap with Kapur 1997 undeterminable; no paired plasma"),
            ("Videbaek 2001", "free plasma; basis conversion barred (V2)"),
            ("Xiberas 2001", "binding index, not occupancy"),
        ),
        summary="no readable source pairs total parent plasma with occupancy outside the excluded studies",
    ),
    "risperidone": SearchedAndEmpty(
        drug="risperidone",
        ec50_source="Remington 2006, PMID 16513859, active moiety, ED50 11.06 ng/ml, Emax fixed 100%",
        searched=(
            ("Ito 2009, PMC6665007", "occupancy by dose group, active-moiety plasma only pooled over all subjects: not paired"),
            ("Nakajima 2016, PMC4681559", "observed occupancy only in Figure 3"),
            ("Ismail 2022, PMC9197923", "Table 1 mean concentration 34.23 ng/ml matches neither visit in Nakajima; same-visit pairing not established"),
            ("Gefvert 2005, PMID 15710053", "abstract prints per-dose ranges of occupancy and plasma, not pairs"),
            ("Uchida 2009, PMID 19225277", "abstract prints a fitted ED50 (7.3 ng/ml) only"),
            ("Nyberg 1999, Kapur 1999", "no plasma in abstract; closed"),
        ),
        summary="readable sources print occupancy and active-moiety concentration without pairing them",
    ),
    "olanzapine": SearchedAndEmpty(
        drug="olanzapine",
        ec50_source="de Greef 2011 Table III, pooled from Kapur 1999 and Nyberg 1997 (its Table II)",
        searched=(
            ("Ismail 2022, PMC9197923", "Table 1 mean 36.01 ng/ml matches neither visit mean in Nakajima 2016 (61.0, 42.3): concentration and occupancy means are not shown to be from the same visits"),
            ("Graff-Guerrero 2015, PMID 26131622", "occupancy printed for olanzapine and risperidone patients pooled"),
            ("Nakajima 2016, PMC4681559", "observed occupancy only in Figure 3"),
            ("Kapur 1998, PMID 9659858", "closed; probably the Kapur 1999 cohort, so not disjoint"),
            ("Mamo 2008, Attarbaschi 2007, Tauscher 1999/2002/2004, Kessler 2005, Lavalaye 1999", "paired plasma not in any abstract; full texts closed"),
        ),
        summary="no readable source pairs olanzapine plasma with olanzapine-only occupancy from the same visit",
    ),
    "clozapine": SearchedAndEmpty(
        drug="clozapine",
        ec50_source="Grunder 2006, PMID 16237387, [18F]fallypride (not ingested)",
        searched=(
            ("Nordstrom 1995, PMID 7573582", "abstract prints serum 105-2121 ng/ml and occupancy 20-67% as ranges; table closed"),
            ("Kapur 1999, PMID 9989565", "no plasma in abstract; closed"),
            ("Kessler 2006, PMID 16738543", "no plasma in abstract; closed"),
            ("Talvik 2001, Kufferle 1997, Pilowsky 1992, Xiberas 2001", "no paired plasma, or binding index"),
        ),
        summary="per-subject serum and occupancy exist only in closed tables",
    ),
    "amisulpride": SearchedAndEmpty(
        drug="amisulpride",
        ec50_source="Vernaleken 2004, PMID 15683553, [18F]desmethoxyfallypride (not ingested)",
        searched=(
            ("la Fougere 2005, PMID 15937316", "abstract prints correlations only; J Nucl Med full text 403"),
            ("Meisenzahl 2008, PMID 18763218", "abstract prints no pairs; Munich cohort shared with la Fougere"),
            ("Xiberas 2001, PMID 11270918", "binding index, not occupancy"),
            ("Stone 2005, PMID 15719226", "no plasma in abstract"),
            ("Martinot 1996, PMID 8935811", "dose only, no plasma"),
        ),
        summary="no readable paired amisulpride plasma and occupancy",
    ),
    "paliperidone": SearchedAndEmpty(
        drug="paliperidone",
        ec50_source="Arakawa 2008 via sponsor page (second-hand; not ingested)",
        searched=(
            ("NCT00934635 posted results", "terminated; one subject, occupancy posted as 0, plasma 'No measures available'"),
            ("Muly 2012", "macaque (rule 2)"),
            ("Ikai 2012, Nakajima 2016, Uchida 2009", "risperidone active moiety, not parent paliperidone"),
        ),
        summary="no parent-paliperidone PET study disjoint from Arakawa 2008 was found",
    ),
    "quetiapine": SearchedAndEmpty(
        drug="quetiapine",
        ec50_source=(
            "Nord 2011, PMC3198174: K_i,app 1369 nmol/l, Occmax fixed 100%, putamen, "
            "[11C]raclopride (found here; P21.2 recorded none; not ingested)"
        ),
        searched=(
            ("Vernaleken 2010, PMID 20392299", "Table 1 prints per-subject serum and occupancy; the HTML omits the table and the PDF is behind a bot check, so it was not read"),
            ("Kapur 2000, PMID 10839333", "no plasma in abstract; closed"),
            ("Gefvert 2001, PMID 11313155", "no plasma values in abstract; closed"),
        ),
        summary="paired observations exist in a table this host could not read",
    ),
}


@dataclass(frozen=True)
class ScoredObservation:
    observation: HeldOutObservation
    predicted: float
    same_tracer: bool

    @property
    def error(self):
        return self.predicted - self.observation.observed_occupancy

    @property
    def contained(self):
        rng = self.observation.observed_range
        if rng is None:
            return None
        return rng[0] <= self.predicted <= rng[1]

    @property
    def standardized_error(self):
        if self.observation.observed_sd is None:
            return None
        return self.error / self.observation.observed_sd


@dataclass(frozen=True)
class DrugResult:
    drug: str
    status: str
    scored: tuple = ()
    empty: SearchedAndEmpty | None = None
    whole_chain: str = ""

    def _metric_set(self):
        same = tuple(s for s in self.scored if s.same_tracer)
        return (same, False) if same else (self.scored, True)

    @property
    def cross_tracer(self):
        return self._metric_set()[1] if self.scored else None

    @property
    def n(self):
        return len(self._metric_set()[0])

    @property
    def n_independent_groups(self):
        return len({s.observation.independent_group for s in self._metric_set()[0]})

    @property
    def mae(self):
        rows = self._metric_set()[0]
        return sum(abs(s.error) for s in rows) / len(rows) if rows else None

    @property
    def bias(self):
        rows = self._metric_set()[0]
        return sum(s.error for s in rows) / len(rows) if rows else None

    @property
    def anecdotal(self):
        return self.n_independent_groups < ANECDOTAL_BELOW_N if self.scored else None

    @property
    def passed(self):
        if not self.scored:
            return None
        limit = PASS_MAE_CROSS_TRACER if self.cross_tracer else PASS_MAE_SAME_TRACER
        return round(self.mae, 3) <= limit

    def mae_by_region(self):
        regions = {}
        for s in self.scored:
            regions.setdefault(s.observation.region, []).append(abs(s.error))
        return {r: sum(v) / len(v) for r, v in regions.items()}


def predict(observation):
    """Occupancy from the drug's own EC50 row at the observed concentration."""
    row = HELD_OUT_EC50[observation.drug]
    if observation.plasma_basis != row.plasma_basis:
        raise ValueError("%s: plasma basis mismatch is a refusal, not a conversion" % observation.drug)
    plasma_nM = observation.plasma_ng_per_mL / row.molar_mass_g_per_mol * 1000.0
    return row.occupancy_at(plasma_nM)


def score_observation(observation):
    row = HELD_OUT_EC50[observation.drug]
    return ScoredObservation(
        observation=observation,
        predicted=predict(observation),
        same_tracer=observation.tracer is not None and observation.tracer == row.tracer,
    )


_NO_CHAIN_PK = (
    "refused: no oral PK record the chain accepts (P22.2 refuses %s at the pk layer)"
)


def drug_result(drug):
    if drug not in DENOMINATOR:
        raise ValueError("%r is not in the P22.3 denominator" % drug)
    if drug in SEARCHED_AND_EMPTY:
        return DrugResult(drug=drug, status="searched-and-empty", empty=SEARCHED_AND_EMPTY[drug])
    scored = tuple(score_observation(o) for o in OBSERVATIONS if o.drug == drug)
    return DrugResult(drug=drug, status="scored", scored=scored, whole_chain=_NO_CHAIN_PK % drug)


def leave_one_drug_out_table():
    """One result per denominator drug, in denominator order."""
    return tuple(drug_result(drug) for drug in DENOMINATOR)


def unrecorded_counts():
    """How many eligible observations lack a tracer or a region.

    The acceptance criterion asks for these to be counted rather than pooled.
    Every eligible observation printed both, so both counts are zero; the EC50
    rows are a different matter and are reported by the result document.
    """
    return {
        "tracer": sum(1 for o in OBSERVATIONS if o.tracer is None),
        "region": sum(1 for o in OBSERVATIONS if o.region is None),
        "total": len(OBSERVATIONS),
    }
