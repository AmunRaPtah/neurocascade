"""Adult oral population PK for six antipsychotics, from the journal literature (P25.1).

Why this module exists
----------------------

P22.5 found that seven of the nine census drugs stop at the PK layer, and
recorded the reason as a ledger row: population PK for those drugs had only
ever been searched for in FDA review documents (P21.3,
:mod:`neurocascade.human.fda_popk`). The searched-and-empty rows there describe
that corpus, not the literature. This module is the literature search: Europe
PMC and PubMed for published adult population-PK models of oral clozapine,
quetiapine, ziprasidone, aripiprazole, risperidone and paliperidone (extended
release), each candidate either transcribed or refused with a reason, and every
query recorded with its hit count so that an empty drug is evidence rather than
silence.

What counts as read
-------------------

A value is transcribed here only if it was read from a raw copy of the source:
Europe PMC full-text XML, the publisher's or a repository's PDF, the PMC HTML
page, or a supplementary file (control stream, DOCX, PDF). Two candidates that
would otherwise have been accepted were reachable only through a tool that
returns a model-written summary of a page instead of the page, and on the one
where it was asked twice for the same table it returned two different central
volumes (Kim 2008, V/F 192 vs 193 L). A summary is not a transcription, so those
rows are refused with ``barrier="summariser-only"`` rather than entered with a
caveat.

Shape of the records
--------------------

The record reuses the two ideas from :mod:`neurocascade.human.fda_popk` that
reading the FDA reviews proved necessary: parameters are keyed by the symbol
the source printed, and ``apparent_parameters`` is decided by a quoted sentence
rather than by the route. It adds three things the literature needed:

* **Fixed parameters are named, with where they were borrowed from.** Five of the
  eight transcribed models fix the absorption rate constant to another study's
  value (Li 2012, Reeves 2024, Chen 2024, Zheng 2024, Størset 2024 all do), and
  quetiapine's two models both borrow Zhou 2015's ka from a paywalled paper. A
  fixed value is not an estimate from this data set, and a consumer must be able
  to tell.
* **Covariates are transcribed with their printed functional form.** The
  sources disagree on form (proportional, additive, power, piecewise-linear,
  allele-additive), and Reeves 2024 prints the age effect two incompatible ways.
  The relation is stored verbatim; nothing here evaluates it.
* **Variability scale may be "unstated".** The three Xuzhou papers (Chen 2024,
  Zheng 2024, and the refused clozapine trio) print an omega for CL/F without
  saying whether it is a variance, an SD or a CV. Assigning one would be a guess
  that changes the number by up to a square root.

Disposition slots are mapped per entity, by hand, because four of the eight
models co-fit a metabolite and the half-life of each entity is a different
quantity with a different label value.

What is here and what is not
----------------------------

The transcription is not a result. The result, as for P21.3, is
:func:`half_life_comparisons`: the terminal half-life implied by each
transcribed parameter set against the half-life printed in the drug's approved
US label, a number nobody in this repository chose. None of these records is
wired into the dose-to-occupancy chain.
"""
import math
from dataclasses import dataclass

from .fda_popk import (
    LabelHalfLife,
    PopPKParameter,
    _hybrid_rate_constants,
    _is_apparent_symbol,
    label_half_life,
)

#: The six drugs P25.1 searched. Order follows the brief.
SEARCHED_DRUGS = (
    "clozapine",
    "quetiapine",
    "ziprasidone",
    "aripiprazole",
    "risperidone",
    "paliperidone",
)

#: Why a candidate model was not transcribed.
#:
#: ``paywalled``
#:     No open copy: Unpaywall reports it closed and the publisher refused the
#:     fetch. Recoverable with a subscription.
#: ``abstract-only``
#:     Paywalled, but the abstract prints some typical values. Those values are
#:     noted on the candidate, never transcribed as a record: an abstract does
#:     not print the structural model, the variability or the covariate form.
#: ``conference-abstract-only``
#:     Published only as a meeting abstract; no parameter table exists.
#: ``summariser-only``
#:     Open access, but every route this host has to the full text returns a
#:     model-written summary rather than the page. See the module docstring.
#: ``out-of-scope``
#:     Read or identified and refused on content: paediatric-only, overdose,
#:     PBPK, a regression with no structural model, a fit to digitised means, or
#:     a reuse of an earlier model with no new estimation.
#: ``not-transcribed``
#:     Read in full and usable, but not transcribed because a better-defined
#:     model of the same drug from a raw source was; the reason is on the row.
CANDIDATE_BARRIERS = (
    "paywalled",
    "abstract-only",
    "conference-abstract-only",
    "summariser-only",
    "out-of-scope",
    "not-transcribed",
)

#: As in :data:`neurocascade.human.fda_popk.VARIABILITY_SCALES`, plus
#: ``unstated`` for the papers that print an omega without its scale.
VARIABILITY_SCALES = (
    "percent-cv",
    "variance",
    "standard-deviation",
    "unstated",
)

#: Unit pairs a disposition mapping may use. A per-kg set is as good as an
#: absolute one for a half-life, because the weight cancels in CL/V, but the
#: two may not be mixed within one entity.
_UNIT_FAMILIES = (("L", "L/h"), ("L/kg", "L/h/kg"))


@dataclass(frozen=True)
class SearchQuery:
    """One query as run, with the hit count it returned."""

    database: str
    query: str
    hits: int
    run_date: str


@dataclass(frozen=True)
class Candidate:
    """A model the search turned up, and what was done with it."""

    citation: str
    pmid: str
    #: ``"transcribed"`` or one of :data:`CANDIDATE_BARRIERS`.
    disposition: str
    reason: str

    def __post_init__(self):
        if self.disposition != "transcribed" and self.disposition not in CANDIDATE_BARRIERS:
            raise ValueError(
                "%s: unknown disposition %r" % (self.citation, self.disposition)
            )
        if not self.reason:
            raise ValueError("%s: a candidate needs a reason" % self.citation)


@dataclass(frozen=True)
class Variability:
    """An inter-individual or inter-occasion variability estimate, as printed."""

    parameter: str
    value: float
    scale: str
    rse_percent: float = None
    shrinkage_percent: float = None
    inter_occasion: bool = False
    source: str = ""

    def __post_init__(self):
        if self.scale not in VARIABILITY_SCALES:
            raise ValueError(
                "unknown variability scale %r for %s" % (self.scale, self.parameter)
            )


@dataclass(frozen=True)
class Covariate:
    """One covariate effect: the printed estimate and its printed form."""

    parameter: str
    covariate: str
    #: The functional form, quoted from the source.
    relation: str
    value: float = None
    rse_percent: float = None
    source: str = ""


@dataclass(frozen=True)
class LiteraturePopPK:
    """One published population-PK model, transcribed from a raw copy."""

    drug: str
    formulation: str
    #: ``parent`` or ``parent-and-metabolite``: whether a metabolite was co-fitted.
    analyte: str
    #: Authors, year, title, journal, volume(issue):pages.
    citation: str
    pmid: str
    doi: str
    #: The raw copy the numbers were read from.
    read_from: str
    population: str
    structural_model: str
    apparent_parameters: bool
    apparent_evidence: str
    software: str
    n_subjects: int
    n_observations: int
    parameters: tuple
    #: ``((printed name, where the fixed value came from), ...)``.
    fixed_parameters: tuple = ()
    variability: tuple = ()
    covariates: tuple = ()
    #: The residual error model and estimates, quoted.
    residual_error: str = ""
    #: ``((entity, ((slot, printed name), ...)), ...)`` with slots ``CL``, ``V1``
    #: and optionally ``Q2``, ``V2``. Hand-mapped per entity.
    disposition_slots: tuple = ()
    #: ``((entity, printed name), ...)`` for a first-order rate constant that
    #: removes the entity from its compartment in addition to ``CL/V1``. Only
    #: Reeves 2024 needs it: its parent leaves by ``CL/F`` and by a metabolic
    #: rate constant ``km`` in parallel.
    extra_elimination_slots: tuple = ()
    reference_weight_kg: float = None
    reference_subject: str = ""
    route: str = "PO"
    notes: str = ""

    def __post_init__(self):
        if self.drug not in SEARCHED_DRUGS:
            raise ValueError("%s is not one of the P25.1 drugs" % self.drug)
        if self.analyte not in ("parent", "parent-and-metabolite"):
            raise ValueError("%s: unknown analyte %r" % (self.key, self.analyte))
        if not self.parameters:
            raise ValueError("%s: a transcribed model needs parameters" % (self.key,))
        if not self.apparent_evidence:
            raise ValueError("%s: the deciding sentence is required" % (self.key,))
        names = {p.name for p in self.parameters}
        if len(names) != len(self.parameters):
            raise ValueError("%s: duplicate parameter names" % (self.key,))
        for p in self.parameters:
            if _is_apparent_symbol(p.name) and not self.apparent_parameters:
                raise ValueError(
                    "%s: %r carries /F but the record claims absolute values"
                    % (self.key, p.name)
                )
        for name, origin in self.fixed_parameters:
            if name not in names:
                raise ValueError("%s: fixed %r is not a parameter" % (self.key, name))
            if not origin:
                raise ValueError("%s: fixed %r needs an origin" % (self.key, name))
        for item in self.variability + self.covariates:
            if item.parameter not in names:
                raise ValueError(
                    "%s: %r refers to %r, which is not a parameter"
                    % (self.key, type(item).__name__, item.parameter)
                )
        for entity, slots in self.disposition_slots:
            mapping = dict(slots)
            if "CL" not in mapping or "V1" not in mapping:
                raise ValueError("%s: %s needs CL and V1 slots" % (self.key, entity))
            for printed in mapping.values():
                if printed not in names:
                    raise ValueError(
                        "%s: slot maps to %r, not a parameter" % (self.key, printed)
                    )
        for entity, printed in self.extra_elimination_slots:
            if printed not in names:
                raise ValueError("%s: %r is not a parameter" % (self.key, printed))
            if entity not in self.entities:
                raise ValueError("%s: no disposition for %s" % (self.key, entity))

    @property
    def key(self):
        """``(drug, first author and year)``: unique within this registry."""
        return (self.drug, self.citation.split(".")[0])

    @property
    def short_citation(self):
        """First author's surname and year, e.g. ``"Koue 2007"``."""
        parts = self.citation.split(".")
        return "%s %s" % (parts[0].split()[0], parts[1].strip())

    @property
    def entities(self):
        return tuple(entity for entity, _ in self.disposition_slots)

    def parameter(self, name):
        for item in self.parameters:
            if item.name == name:
                return item
        raise KeyError("%s has no parameter %r" % (self.key, name))

    def is_fixed(self, name):
        return name in dict(self.fixed_parameters)


@dataclass(frozen=True)
class DrugSearch:
    """Everything the search did for one drug, whether or not it found a model."""

    drug: str
    queries: tuple
    candidates: tuple
    #: ``"transcribed"`` or ``"searched-and-empty"``.
    outcome: str
    summary: str

    def __post_init__(self):
        if self.outcome not in ("transcribed", "searched-and-empty"):
            raise ValueError("%s: unknown outcome %r" % (self.drug, self.outcome))
        if not self.queries:
            raise ValueError("%s: a search with no recorded query" % self.drug)
        transcribed = [c for c in self.candidates if c.disposition == "transcribed"]
        if self.outcome == "searched-and-empty" and transcribed:
            raise ValueError("%s: empty search lists a transcribed model" % self.drug)
        if self.outcome == "transcribed" and not transcribed:
            raise ValueError("%s: no transcribed candidate" % self.drug)


def _p(name, value, units, source, rse=None):
    return PopPKParameter(name=name, value=value, units=units, rse_percent=rse, source=source)


def _q(database, query, hits):
    return SearchQuery(database, query, hits, "2026-10-01")


def _standard_queries(drug, epmc, pubmed):
    """The five queries run identically for every drug, with their hit counts."""
    templates = (
        '"population pharmacokinetic*" AND %s',
        "%s AND NONMEM",
        '%s AND "nonlinear mixed effect*"',
        '(TITLE:%s) AND (TITLE:"population pharmacokinetic*")',
    )
    queries = tuple(_q("Europe PMC", t % drug, n) for t, n in zip(templates, epmc))
    return queries + (
        _q("PubMed", "%s[ti] AND population pharmacokinetic*[tiab]" % drug, pubmed),
    )


# ---------------------------------------------------------------------------
# Clozapine
# ---------------------------------------------------------------------------

_LI_2012 = "Li LJ, Shang DW, Li WB, et al. 2012. Population pharmacokinetics of clozapine and its primary metabolite norclozapine in Chinese patients with schizophrenia. Acta Pharmacol Sin 33(11):1409-1416"

_CLOZAPINE_LI_2012 = LiteraturePopPK(
    drug="clozapine",
    formulation="oral tablets (formulation not stated further)",
    analyte="parent-and-metabolite",
    citation=_LI_2012,
    pmid="22820910",
    doi="10.1038/aps.2012.71",
    read_from="PMC4011352, PMC HTML full text (Table 1, Table 2, Methods equations)",
    population=(
        "162 Chinese inpatients with refractory schizophrenia (74 male, 88 female), "
        "age 35.5 +/- 10.6 (18-59) years, multiple mental health sites; smokers: "
        "female 2/74, male 48/88; dose range not printed; HPLC-UV"
    ),
    structural_model=(
        "A one-compartment pharmacokinetic model with mixture error best described "
        "the concentration of clozapine and norclozapine. (ADVAN5; 'a proportion of "
        "clozapine was converted into norclozapine in the central compartment of "
        "clozapine through a first-order conversion mechanism and the rest was "
        "converted into other metabolites')"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the apparent clearance and the apparent distribution volume of clozapine "
        "(CL/F and V/F), where F is the unknown oral bioavailability of clozapine"
    ),
    software="NONMEM 7 level 1.0, FOCE (FOCE-I tested and rejected)",
    n_subjects=162,
    n_observations=1617,
    parameters=(
        _p("CL/F", 21.9, "L/h", "Table 2", rse=6),
        _p("V/F", 526.0, "L", "Table 2", rse=10),
        _p("Ka", 1.3, "1/h", "Table 2, 'Ka (Fixed)'"),
        _p("KF", 0.66, None, "Table 2, 'KF (Fixed)'"),
        _p("CLM", 32.7, "L/h", "Table 2", rse=5.6),
        _p("VM", 624.0, "L", "Table 2", rse=5.5),
        _p("θSmoking (CL/F)", 0.45, None, "Table 2", rse=34.9),
        _p("θGender (CL/F)", 0.208, None, "Table 2", rse=44.6),
        _p("θSmoking (CLM)", 0.543, None, "Table 2", rse=35.7),
        _p("θGender (CLM)", 0.242, None, "Table 2", rse=49.2),
    ),
    fixed_parameters=(
        ("Ka", "fixed at 1.3 h-1 'based on several pharmacokinetic studies that obtained rich data' (ref 12)"),
        ("KF", "fixed at 0.66 'in published papers' (refs 22, 23)"),
    ),
    variability=(
        Variability("CL/F", 42.9, "percent-cv", source="Table 2, 'Interindividual variability%'"),
        Variability("V/F", 65.7, "percent-cv", source="Table 2"),
        Variability("CLM", 42.1, "percent-cv", source="Table 2"),
        Variability("VM", 75.6, "percent-cv", source="Table 2"),
    ),
    covariates=(
        Covariate("CL/F", "male sex", "TVCL=θCLF×(1+SEX×θSEX)", 0.208, 44.6, "Methods equation; Table 2"),
        Covariate("CL/F", "smoking (>=5 cigarettes/day)", "same proportional form; 'clozapine by 45%'", 0.45, 34.9, "Table 2; Abstract"),
        Covariate("CLM", "male sex", "same proportional form; 'norclozapine by 24.2%'", 0.242, 49.2, "Table 2; Abstract"),
        Covariate("CLM", "smoking", "same proportional form; 'norclozapine by 54.3%'", 0.543, 35.7, "Table 2; Abstract"),
    ),
    residual_error=(
        "mixture error; σ1 (μmol/L) SD=0.162 additive; σ2 % CV=26.6 proportional "
        "(clozapine); σ3 (μmol/L) SD=0.117; σ4 % CV=16.9 (norclozapine). Table 2"
    ),
    disposition_slots=(("parent", (("CL", "CL/F"), ("V1", "V/F"))),),
    reference_subject=(
        "female nonsmoker: 'The population-predicted clearance of clozapine and "
        "norclozapine in female nonsmokers were 21.9 and 32.7 L/h'. Age and weight "
        "were tested and not retained, so no reference weight exists"
    ),
    notes=(
        "CL/F is read as the total apparent clearance of clozapine, of which the "
        "fixed fraction KF forms norclozapine: Figure 1 splits it into K20 and K23 "
        "and the Methods say 'the rest was converted into other metabolites'. "
        "Table 2's footnote labels θGender 'effect of smoking status on clearance', "
        "a typo the Methods equation resolves. Covariate rows print two numbers, "
        "read as estimate and RSE% (they have no IIV column)."
    ),
)

_REEVES_2024 = "Reeves S, Bertrand J, Obee SJ, et al. 2024. A population pharmacokinetic model to guide clozapine dose selection, based on age, sex, ethnicity, body weight and smoking status. Br J Clin Pharmacol 90(1):135-145"

_CLOZAPINE_REEVES_2024 = LiteraturePopPK(
    drug="clozapine",
    formulation="oral tablets",
    analyte="parent-and-metabolite",
    citation=_REEVES_2024,
    pmid="36793249",
    doi="10.1111/bcp.15691",
    read_from=(
        "UCL Discovery open-access accepted-online PDF (eprint 10165013), pp. 1-11 "
        "of the 2023 online pagination; Table 2 'Ethnic CL/Fclozapine' (final) column"
    ),
    population=(
        "UK/Ireland clozapine TDM service 1993-2017: 5960 patients (4315 male, "
        "1645 female) aged 18-86; inducers, inhibitors, valproate and omeprazole "
        "excluded; 50% of the daily dose assumed every 12 h"
    ),
    structural_model=(
        "A pharmacokinetic model that accounted for the metabolism of plasma "
        "clozapine to norclozapine ... (i) clozapine first-order absorption rate "
        "constant (ka) fixed at 0.69 h-1; (ii) apparent volume of distribution of "
        "clozapine (V/Fclozapine); (iii) apparent plasma clearance of clozapine "
        "(CL/Fclozapine); (iv) metabolic rate constant (km); (v) apparent volume of "
        "distribution of norclozapine (V/Fnorclozapine), which was assumed to be "
        "equal to V/Fclozapine; and (vi) apparent plasma clearance of norclozapine "
        "(CL/Fnorclozapine)."
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "The term F represents, here, the bioavailability of the drug that cannot "
        "be estimated from oral data alone."
    ),
    software="Monolix 2020R1, SAEM",
    n_subjects=5960,
    n_observations=17787,
    parameters=(
        _p("Ka", 0.69, "1/h", "Table 2, 'Ka (h-1; fixed)'"),
        _p("V/Fclozapine", 1460.0, "L", "Table 2", rse=0.4),
        _p("CL/Fclozapine", 20.2, "L/h", "Table 2", rse=1.3),
        _p("Km", 0.005, "1/h", "Table 2", rse=0.5),
        _p("CL/Fnorclozapine", 12.0, "L/h", "Table 2", rse=0.3),
        _p("β Weight_V/Fclozapine", 1.0, None, "Table 2, 'ne'"),
        _p("β Weight_CL/Fclozapine", 0.75, None, "Table 2, 'ne'"),
        _p("β Smoking_CL/Fclozapine", 0.37, None, "Table 2", rse=3.1),
        _p("β Female_CL/Fclozapine", -0.24, None, "Table 2", rse=8.8),
        _p("β Age_CL/Fclozapine (exponential)", -0.013, None, "Table 2", rse=4.4),
        _p("β Afro-Caribbean_CL/Fclozapine", 0.12, None, "Table 2", rse=30.5),
        _p("β Asian_CL/Fclozapine", -0.19, None, "Table 2", rse=21.4),
    ),
    fixed_parameters=(
        ("Ka", "fixed at 0.69 h-1, 'ne' (not estimated) in Table 2"),
        ("β Weight_V/Fclozapine", "allometric exponent fixed, 'ne'"),
        ("β Weight_CL/Fclozapine", "allometric exponent fixed, 'ne'"),
    ),
    variability=(
        Variability("V/Fclozapine", 0.11, "standard-deviation", 3.4, 87, source="Table 2 ω; shrinkage from Results"),
        Variability("CL/Fclozapine", 0.72, "standard-deviation", 1.1, 15, source="Table 2 ω; shrinkage from Results"),
        Variability("Km", 0.12, "standard-deviation", 3.1, 80, source="Table 2 ω; shrinkage from Results"),
        Variability("CL/Fnorclozapine", 0.06, "standard-deviation", 3.9, 95, source="Table 2 ω; shrinkage from Results"),
    ),
    covariates=(
        Covariate(
            "CL/Fclozapine",
            "weight, age, smoking, sex, ethnicity",
            "CL/Fclozapinei = μCL/Fclozapine *(weight/70)^0.75 * (agei-40)^0.013 * "
            "(1 + 0.37 if Smoker) * (1 –0.24 if Female) * (1 + 0.12 if "
            "Afro-Caribbean) * (1 –0.19 if Asian).",
            source="Results, p. 5",
        ),
        Covariate("V/Fclozapine", "weight", "exponent 1.0 on weight/70, fixed", 1.0, source="Table 2"),
    ),
    residual_error=(
        "combined, (yij = ŷij + (σinter + σslope.ŷij)εij: clozapine σ_intercept "
        "0.02 mg/L (RSE 15.9), σ_slope 0.39 (0.7); norclozapine 0.02 mg/L (7.1), "
        "0.31 (0.8). Table 2"
    ),
    disposition_slots=(
        ("parent", (("CL", "CL/Fclozapine"), ("V1", "V/Fclozapine"))),
        ("metabolite", (("CL", "CL/Fnorclozapine"), ("V1", "V/Fclozapine"))),
    ),
    extra_elimination_slots=(("parent", "Km"),),
    reference_weight_kg=70.0,
    reference_subject=(
        "'For the reference group (male, age 40 years, 70 kg weight, White, "
        "nonsmoker), model estimates of CL/Fclozapine, km, V/F clozapine and t½ "
        "were 20.2 L h-1, 0.005 h-1, 1460 L and 36 h'"
    ),
    notes=(
        "The age effect is printed two incompatible ways: Table 2 labels it "
        "'exponential' with β = -0.013, the Results formula is a power of "
        "(age-40) with exponent +0.013. Not resolved here; nothing evaluates it. "
        "Km is a parallel first-order elimination of the parent, which is how the "
        "paper's own reference t½ of 36 h arises: ln2 / (20.2/1460 + 0.005) is "
        "36.8 h, while ln2 * 1460 / 20.2 alone would be 50 h. The metabolite mapping "
        "uses V/Fclozapine because the paper assumed the two volumes equal."
    ),
)

_CLOZAPINE_SEARCH = DrugSearch(
    drug="clozapine",
    queries=_standard_queries("clozapine", (169, 74, 24, 12), 33),
    candidates=(
        Candidate(_LI_2012, "22820910", "transcribed", "adult, raw PMC HTML, full Table 2"),
        Candidate(_REEVES_2024, "36793249", "transcribed", "adult 18-86, largest data set, OA PDF"),
        Candidate("Olmos I, et al. 2019. Biomed Res Int 2019:3163502", "30956977", "not-transcribed",
                  "OA, adult, but V/F fixed to 750 L from another study and ka fixed, trough-only; its half-life would test borrowed constants"),
        Candidate("Nomura N, et al. 2021. Ther Adv Psychopharmacol 11:20451253211016189", "34046160", "not-transcribed",
                  "OA, adult, but estimates an F of 0.817 from oral data alone while printing bare CL and V, so apparent vs absolute is undecidable; no RSE"),
        Candidate("Geers LM, et al. 2022. Ther Adv Psychopharmacol 12:20451253211065857", "35518123", "out-of-scope",
                  "15 patients, iterative two-stage Bayesian (MwPharm), V per kg lean body mass; not a population NLME fit of comparable standing"),
        Candidate("Han HH / Zhang Y / Jiang L, et al. 2025 (Front Psychiatry; Drug Des Devel Ther x2)", "41058642; 41262335; 41209151", "not-transcribed",
                  "OA, trough TDM, ka fixed 1.3, omega scale unstated, interaction-focused; the Jiang cohort includes a 14.6-year-old"),
        Candidate("Ng W, et al. 2009. Ther Drug Monit 31(3):360-6", "19349931", "abstract-only",
                  "paywalled; abstract: V fixed 7 L/kg, CL/F 18.0 L/h"),
        Candidate("Ismail Z, et al. 2012. Am J Geriatr Psychiatry 20(1):53-60", "21422906", "paywalled", "paywalled; ages 11-79"),
        Candidate("Shang DW, et al. 2014. Ther Drug Monit 36(3):378-86", "24342896", "paywalled", "two-compartment PK/PD; no values in abstract"),
        Candidate("Park SI, et al. 2023. Ther Drug Monit 45(3):417-424", "36097334", "paywalled", "two-compartment with lag; no values in abstract"),
        Candidate("Berneri M, et al. 2024. Ther Drug Monit 46(2):217-226", "38446630", "paywalled", "no values in abstract"),
        Candidate("Panić B, et al. 2026. Prog Neuropsychopharmacol Biol Psychiatry 144:111583", "41380932", "abstract-only",
                  "paywalled; abstract: CL/F 26.4 L/h (female non-smoker)"),
        Candidate("Jerling M, et al. 1997. Br J Clin Pharmacol 44(5):447-53", "9384461", "out-of-scope",
                  "nonparametric maximum likelihood; scanned PMC PDF behind a browser check"),
        Candidate("Qiu XW, et al. 2009. Yao Xue Xue Bao 44(7):785-92", "19806921", "abstract-only", "no full text; abstract prints CL/F 28.5 L/h, V/F 1290 L"),
        Candidate("Mostafa S, et al. 2024; Wills KH, et al. 2021", "PMC10941576; PMC8779032", "out-of-scope", "PBPK"),
    ),
    outcome="transcribed",
    summary="two adult oral models transcribed from raw copies; twelve further candidates refused with reasons",
)


# ---------------------------------------------------------------------------
# Quetiapine
# ---------------------------------------------------------------------------

_CHEN_2024 = "Chen X, Zhang Y, Yin D, et al. 2024. Optimization of initial dosage of quetiapine in schizophrenic patients: effects of fluvoxamine or duloxetine coadministration. Front Pharmacol 15:1496043"

_QUETIAPINE_CHEN_2024 = LiteraturePopPK(
    drug="quetiapine",
    formulation="immediate-release tablets",
    analyte="parent",
    citation=_CHEN_2024,
    pmid="39635430",
    doi="10.3389/fphar.2024.1496043",
    read_from="PMC11614649, Europe PMC full-text XML (Section 2.2, Eqs 1-7, Tables 1 and 3)",
    population=(
        "96 Chinese schizophrenia inpatients, single centre (Xuzhou), retrospective "
        "TDM, 52 men / 44 women, age 43.53 +/- 14.17, weight 70.88 +/- 16.84 kg; "
        "sparse TDM samples; dose not printed"
    ),
    structural_model=(
        "We constructed a PPK model using the non-linear mixed-effect modeling "
        "(NONMEM) approach using the apparent oral clearance (CL/F), apparent volume "
        "of distribution (V/F), and absorption rate constant (Ka) fixed at 1.46/h "
        "(Zhou et al., 2015) as the assessment parameters. (No compartment count is "
        "printed; CL/F, V/F and Ka are the only structural parameters, which is a "
        "one-compartment model with first-order absorption.)"
    ),
    apparent_parameters=True,
    apparent_evidence="CL/F, apparent oral clearance (L/h); V/F, apparent volume of distribution (L)",
    software="NONMEM (version and estimation method not printed)",
    n_subjects=96,
    n_observations=154,
    parameters=(
        _p("CL/F", 118.0, "L/h", "Table 3, SE (%)", rse=6.9),
        _p("V/F", 2460.0, "L", "Table 3, SE (%)", rse=33.5),
        _p("Ka", 1.46, "1/h", "Table 3, '1.46 (fixed)'"),
        _p("θFLU", -0.536, None, "Table 3", rse=4.7),
        _p("θDUL", -0.537, None, "Table 3", rse=12.0),
    ),
    fixed_parameters=(("Ka", "Zhou D et al. 2015, J Clin Pharmacol 55(11):1248-55, PMID 25975812"),),
    variability=(Variability("CL/F", 0.333, "unstated", 12.9, source="Table 3, 'ω CL/F'"),),
    covariates=(
        Covariate("CL/F", "weight, fluvoxamine, duloxetine",
                  "CL / F = 118 × weight / 70 0.75 × 1 − 0.536 × FLU × 1 − 0.537 × DUL",
                  source="Eq. 6; allometric exponent fixed per Eq. 3"),
        Covariate("V/F", "weight", "V / F = 2460 × weight / 70", source="Eq. 7"),
    ),
    residual_error="D i = F i + F i * ε 1 + ε 2; σ1 (proportional) 0.267 (SE 15.3%), σ2 (additive) 29.917 (34.7%); scale unstated. Table 3",
    disposition_slots=(("parent", (("CL", "CL/F"), ("V1", "V/F"))),),
    reference_weight_kg=70.0,
    reference_subject="70 kg ('L std is the standard weight of 70 kg'), no fluvoxamine, no duloxetine",
    notes="V/F carries 33.5% SE and a bootstrap 90% interval of 1222-5163 L, the mark of a volume barely informed by sparse TDM samples.",
)

_ZHENG_2024 = "Zheng ZQ, Jin YW, Yin D, et al. 2024. Model-informed precision dosing of quetiapine in bipolar affective disorder patients: initial dose recommendation. Front Psychiatry 15:1497119"

_QUETIAPINE_ZHENG_2024 = LiteraturePopPK(
    drug="quetiapine",
    formulation="oral (IR or XR not stated)",
    analyte="parent",
    citation=_ZHENG_2024,
    pmid="39698209",
    doi="10.3389/fpsyt.2024.1497119",
    read_from="PMC11652601, Europe PMC full-text XML (Section 2.2, Eqs 1-7, Tables 1 and 3)",
    population=(
        "99 Chinese patients with bipolar affective disorder (66 men, 33 women), "
        "age 17.05-69.47 (median 36.66), weight 43-119 kg, retrospective TDM, "
        "single centre (Xuzhou); number of samples and doses not printed"
    ),
    structural_model=(
        "A quetiapine PPK model for patients with BAD was built using non-linear "
        "mixed effect modeling (NONMEM), including apparent oral clearance (CL/F), "
        "apparent volume of distribution (V/F), and the absorption rate constant "
        "(K a , fixed at 1.46/h) (23)."
    ),
    apparent_parameters=True,
    apparent_evidence="CL/F and V/F represent the apparent oral clearance and the apparent volume of distribution, respectively.",
    software="NONMEM (version and estimation method not printed)",
    n_subjects=99,
    n_observations=None,
    parameters=(
        _p("CL/F", 76.1, "L/h", "Table 3, SE (%)", rse=26.9),
        _p("V/F", 530.0, "L", "Table 3, SE (%)", rse=59.6),
        _p("Ka", 1.46, "1/h", "Table 3, '1.46 (fixed)'"),
    ),
    fixed_parameters=(("Ka", "ref 23 = Zhou D et al. 2015, PMID 25975812"),),
    variability=(Variability("CL/F", 0.285, "unstated", 24.0, source="Table 3, 'ω CL/F'"),),
    covariates=(
        Covariate("CL/F", "weight", "CL / F = 76.1 × ( weight/70 ) 0.75", source="Eq. 6"),
        Covariate("V/F", "weight", "V / F = 530 × ( weight / 70 )", source="Eq. 7"),
    ),
    residual_error="E i = G i + G i * ϵ 1 + ϵ 2; σ1 (proportional) 0.312 (SE 7.9%), σ2 (additive) 23.896 (23.9%); scale unstated. Table 3",
    disposition_slots=(("parent", (("CL", "CL/F"), ("V1", "V/F"))),),
    reference_weight_kg=70.0,
    reference_subject="70 kg ('N std is the standard weight of 70 kg'); no other covariate retained",
    notes=(
        "Youngest patient 17.05 years, so the cohort is adult with at least one "
        "adolescent. V/F SE 59.6%, bootstrap 90% interval 210-2088 L."
    ),
)

_QUETIAPINE_SEARCH = DrugSearch(
    drug="quetiapine",
    queries=_standard_queries("quetiapine", (141, 70, 11, 3), 10),
    candidates=(
        Candidate(_CHEN_2024, "39635430", "transcribed", "adult, oral IR, OA full text"),
        Candidate(_ZHENG_2024, "39698209", "transcribed", "adult (min 17.05 y), OA full text"),
        Candidate("Zhou D, Bui KH, Li J, Al-Huniti N. 2015. J Clin Pharmacol 55(11):1248-55", "25975812", "abstract-only",
                  "paywalled; abstract: one-compartment, V/F 574 L, k 0.12 h-1, ka 1.46 (IR) and 0.10 h-1 (XR); the source of the ka both transcribed models fix"),
        Candidate("Fukushi R, et al. 2020. Clin Ther 42(6):1067-1076.e2", "32518042", "abstract-only",
                  "paywalled; abstract: XR, Japanese, CL/F 87.7 L/h, V/F 277 L"),
        Candidate("Kimko HC, et al. 2000. Clin Pharmacol Ther 68(5):568-77", "11103759", "paywalled", "one-compartment; no values in abstract"),
        Candidate("Shilbayeh SA, et al. 2015. Clin Pharmacol Drug Dev 4(5):387-94", "27137148", "paywalled", "single 25 mg dose, transit compartments; no typical values in abstract"),
        Candidate("Lin M, et al. 2024. J Affect Disord 351:309-313", "38262522", "out-of-scope", "multivariable linear regression, no structural model"),
        Candidate("Isbister GK, et al. 2007. Clin Pharmacol Ther 81(6):821-7", "17410121", "out-of-scope", "overdose population"),
        Candidate("Chen X, et al. 2025. Drug Des Devel Ther 19:8051-8067", "40951689", "out-of-scope", "depression cohort with median age 17.05 (12.05-65.29): about half minors"),
        Candidate("Li X, et al. 2026. J Pharmacokinet Pharmacodyn 53(4):26", "42143189", "out-of-scope",
                  "pyDarwin methods paper on CATIE data; the CATIE quetiapine model it reproduces is a book chapter, not journal literature"),
    ),
    outcome="transcribed",
    summary="two adult oral models transcribed; both fix ka to Zhou 2015, whose own paper is paywalled",
)


# ---------------------------------------------------------------------------
# Ziprasidone
# ---------------------------------------------------------------------------

_ZIPRASIDONE_SEARCH = DrugSearch(
    drug="ziprasidone",
    queries=_standard_queries("ziprasidone", (53, 30, 8, 1), 2) + (
        _q("Europe PMC", 'ziprasidone AND "mixed effect*" AND pharmacokinetic*', 57),
        _q("Europe PMC", "ziprasidone AND CATIE AND pharmacokinetic*", 136),
        _q("Europe PMC", 'ziprasidone AND "Phoenix NLME"', 1),
        _q("PubMed", "ziprasidone AND NONMEM", 0),
    ),
    candidates=(
        Candidate("Wessels AM, Bies RR, Pollock BG, et al. 2011. Population pharmacokinetic modeling of ziprasidone in patients with schizophrenia from the CATIE study. J Clin Pharmacol 51(11):1587-91",
                  "21209243", "paywalled",
                  "the one adult oral model of the CATIE cohort (233 subjects, 568 concentrations per PMC3960457). "
                  "Unpaywall: is_oa false, closed, no OA location; the publisher PDF returns HTTP 403; PubMed and "
                  "Europe PMC carry no abstract. The only parameter reachable second-hand is ka fixed at 0.5 h-1 "
                  "(Li X et al. 2026, PMC13179877, Discussion), which is not a transcription of the paper"),
        Candidate("Bao S, Yang S, Hua Z, et al. 2024. Ziprasidone population pharmacokinetics and co-medication effects in Chinese patients. Naunyn Schmiedebergs Arch Pharmacol 397(12):9811-21",
                  "38918237", "abstract-only",
                  "Unpaywall closed; the Springer PDF URL returns an HTML page. The abstract prints 'Ka was fixed at "
                  "0.5 h-1 based on literature, and the population typical values for ziprasidone clearance (CL) and "
                  "volume of distribution (V) were 18.74 L/h and 110.24 L', 185 patients, 547 concentrations, Phoenix "
                  "NLME 8.1. It does not print the compartment count, the variability, the covariate form, or "
                  "whether CL and V are apparent"),
        Candidate("Perera V, et al. 2014. Br J Clin Pharmacol 78(4):800-14", "24773369", "out-of-scope",
                  "optimal-sampling simulation reusing earlier models; no new estimation"),
        Candidate("CATIE ziprasidone D2-occupancy papers", "24491908; 22290266; 24028745; 23899638", "out-of-scope",
                  "apply Wessels 2011 post-hoc estimates; no model of their own"),
    ),
    outcome="searched-and-empty",
    summary=(
        "searched and empty. Exactly two adult oral population-PK models of ziprasidone exist in the "
        "indexed literature and both are closed-access with no open copy. No sponsor (Pfizer) model was "
        "published in a journal; P21.3 found the FDA route empty for the same drug"
    ),
)


# ---------------------------------------------------------------------------
# Aripiprazole
# ---------------------------------------------------------------------------

_KOUE_2007 = "Koue T, Kubo M, Funaki T, et al. 2007. Nonlinear mixed effects model analysis of the pharmacokinetics of aripiprazole in healthy Japanese males. Biol Pharm Bull 30(11):2154-2158"

_ARIPIPRAZOLE_KOUE_2007 = LiteraturePopPK(
    drug="aripiprazole",
    formulation="oral tablets (ABILIFY)",
    analyte="parent",
    citation=_KOUE_2007,
    pmid="17978491",
    doi="10.1248/bpb.30.2154",
    read_from="J-STAGE PDF, pp. 2154-2157: Methods Eqs 3-7, Table 1, Table 2 (p. 2157)",
    population=(
        "68 healthy Japanese males aged 20-32 (mean 23.1), 48.2-75.2 kg (mean "
        "60.7); single 6 mg fasted (26), 3 mg daily for 14 days (15), 3 mg alone and "
        "with itraconazole (27); dense sampling to 366 h; CYP2D6 groups 1-3, no poor "
        "metabolisers"
    ),
    structural_model=(
        "The two-compartment model with first-order absorption was parameterized in "
        "terms of absorption lag time (ALAG), absorption rate constant (KA), the "
        "apparent volume of central compartment (V1/F), oral clearance (CL/F), the "
        "apparent volume of peripheral compartment (V2/F), and the apparent "
        "inter-compartment clearance (Q/F) with ADVAN4 and TRANS4."
    ),
    apparent_parameters=True,
    apparent_evidence="printed V1/F, CL/F, V2/F, Q/F ('the apparent volume of central compartment (V1/F)')",
    software="NONMEM, FOCE",
    n_subjects=68,
    n_observations=None,
    parameters=(
        _p("θ1 ALAG", 0.805, "h", "Table 2, 95% CI 0.738-0.872"),
        _p("θ2 KA", 2.65, "1/h", "Table 2, 95% CI 1.92-3.38"),
        _p("θ3 V1/F", 3.84, "L/kg", "Table 2, 95% CI 3.32-4.36"),
        _p("θ4 CL/F", 0.0645, "L/h/kg", "Table 2, 95% CI 0.0494-0.0796 (CYP2D6 Group 1)"),
        _p("θ5 (CYP2D6 Group 2 decrease in CL/F)", 0.0135, "L/h/kg", "Table 2, 95% CI -0.0067-0.0337"),
        _p("θ6 (CYP2D6 Group 3 decrease in CL/F)", 0.0293, "L/h/kg", "Table 2, 95% CI 0.0070-0.0516"),
        _p("θ7 (itraconazole decrease in CL/F)", 0.0181, "L/h/kg", "Table 2, 95% CI 0.0010-0.0352"),
        _p("θ8 V2/F", 1.54, "L/kg", "Table 2, 95% CI 1.32-1.76"),
        _p("θ9 Q/F", 0.168, "L/h/kg", "Table 2, 95% CI 0.125-0.211"),
    ),
    variability=(
        Variability("θ1 ALAG", 0.110, "standard-deviation", source="Table 2, ωALAG"),
        Variability("θ2 KA", 0.910, "standard-deviation", source="Table 2, ωKA"),
        Variability("θ3 V1/F", 0.381, "standard-deviation", source="Table 2, ωV1/F"),
        Variability("θ4 CL/F", 0.397, "standard-deviation", source="Table 2, ωCL/F"),
        Variability("θ8 V2/F", 0.365, "standard-deviation", source="Table 2, ωV2/F"),
        Variability("θ9 Q/F", 0.262, "standard-deviation", source="Table 2, ωQ/F"),
    ),
    covariates=(
        Covariate("θ4 CL/F", "CYP2D6 group, itraconazole, weight",
                  "CL/Fi = (θ4 − θ5·G2 − θ6·G3 − θ7·ITZ)·WTi·exp(ηCL/Fi)", source="Eq. 4"),
        Covariate("θ3 V1/F", "weight", "V1/Fi = θ3·WTi·exp(ηV1/Fi)", source="Eq. 3"),
        Covariate("θ8 V2/F", "weight", "V2/Fi = θ8·WTi·exp(ηV2/Fi)", source="Eq. 5"),
        Covariate("θ9 Q/F", "weight", "Q/Fi = θ9·WTi·exp(ηQ/Fi)", source="Eq. 6"),
    ),
    residual_error="exponential, Cij = Cij*·exp(εij), σ 0.166 (95% CI 0.124-0.200). Eq. 7, Table 2",
    disposition_slots=(
        ("parent", (("CL", "θ4 CL/F"), ("V1", "θ3 V1/F"), ("Q2", "θ9 Q/F"), ("V2", "θ8 V2/F"))),
    ),
    reference_subject=(
        "CYP2D6 Group 1 (*1/*1, *1/*2, *2/*2), no itraconazole; every disposition "
        "parameter is linear in weight, so the set is per kg and needs no reference weight"
    ),
    notes=(
        "Table 2 prints 95% CIs, not RSEs. The ω are standard deviations: the "
        "Methods give each η a 'variance of ω²'. The paper's own non-compartmental "
        "terminal half-lives after 6 mg are 44.2, 47.7 and 76.3 h for Groups 1-3."
    ),
)

_KNIGHTS_2015 = "Knights J, Rohatagi S. 2015. Development and application of an aggregate adherence metric derived from population pharmacokinetics to inform clinical trial enrichment. J Pharmacokinet Pharmacodyn 42(3):263-273"

_ARIPIPRAZOLE_KNIGHTS_2015 = LiteraturePopPK(
    drug="aripiprazole",
    formulation="oral",
    analyte="parent",
    citation=_KNIGHTS_2015,
    pmid="25821065",
    doi="10.1007/s10928-015-9414-4",
    read_from=(
        "PMC4432109 Europe PMC full-text XML (Methods, Eq. 2), Fig. 1b image, and "
        "Electronic Supplementary Material 1 (DOCX) NONMEM control stream $THETA/$OMEGA/$SIGMA"
    ),
    population=(
        "'24 clinical studies comprising 448 individuals with over 13,500 "
        "observations' (the abstract says '448 patients'); healthy/patient mix, age "
        "and dose ranges not printed; covariates centred at 74.19 kg and 32 years"
    ),
    structural_model="A 2-compartment model with first-order absorption was found to best describe the data.",
    apparent_parameters=True,
    apparent_evidence=(
        "'the apparent clearance of the drug from the central compartment (CL)'; "
        "control stream 'CLEARANCE (apparent) OF ARI'. The volumes are called "
        "'theoretical', and no intravenous data are mentioned"
    ),
    software="NONMEM 7.2",
    n_subjects=448,
    n_observations=13500,
    parameters=(
        _p("KA", 0.515, "1/h", "Fig. 1b; control stream TH(1) KA(1/hr)"),
        _p("VC", 192.0, "L", "Fig. 1b; TH(2) VC(L)"),
        _p("Q", 12.2, "L/h", "control stream TH(3) Q(L/hr); Fig. 1b header prints '1/hr'"),
        _p("VP", 151.0, "L", "Fig. 1b; TH(4) VP(L)"),
        _p("CL", 3.88, "L/h", "Fig. 1b; TH(5) CL(L/hr)"),
        _p("WT_VP", 4.07, None, "TH(6) Linear relationship slope WT on VP"),
        _p("AGE_VP", 0.915, None, "TH(7) Linear AGE on VP"),
        _p("WT_CL", 0.0251, None, "TH(8); Fig. 1b prints 0.025"),
        _p("WT_Q", 0.425, None, "TH(9) Linear WT(+)<115 on Q"),
        _p("AGE_CL", -0.0167, None, "TH(10); Fig. 1b prints -0.017"),
        _p("2D6PM_CL", -0.478, None, "TH(11) Proportional shift in CL from CYP2D6 PM status"),
        _p("SEXF_Q", 0.543, None, "TH(12) PROPORTIONAL(+) EFFECT OF SEX0 ON Q"),
        _p("SEXF_VP", 0.341, None, "TH(13) PROPORTIONAL(+) EFFECT OF SEX0 ON VP"),
    ),
    variability=(
        Variability("KA", 0.398, "variance", shrinkage_percent=15.08, source="$OMEGA; Fig. 1b"),
        Variability("VC", 0.0493, "variance", shrinkage_percent=11.83, source="$OMEGA BLOCK(3)"),
        Variability("Q", 0.0553, "variance", shrinkage_percent=35.71, source="$OMEGA BLOCK(3)"),
        Variability("VP", 0.0767, "variance", shrinkage_percent=14.45, source="$OMEGA BLOCK(3)"),
        Variability("CL", 0.153, "variance", shrinkage_percent=2.05, source="$OMEGA; Fig. 1b"),
    ),
    covariates=(
        Covariate("CL", "weight, age, CYP2D6 poor metaboliser",
                  "CLexp(L/hr) = TVCL = [3.88 + WTA∗0.0251∗(WT−74.19) − 0.0167∗(AGE−32)]∗(1 − 0.478∗2D6PM)",
                  source="Eq. 2"),
        Covariate("VP", "weight, age, sex",
                  "TVVP=(THETA(4) + THETA(6)*(WT-74.19) + THETA(7)*(AGE-32))*(1+SEXO*THETA(13))",
                  source="control stream"),
        Covariate("Q", "weight, sex", "TVQA=THETA(3) + WTOKQ*THETA(9)*(WT-74.19); TVQ=(TVQA)*(1+SEXO*THETA(12))",
                  source="control stream"),
    ),
    residual_error="proportional, $SIGMA 0.0538, ε-shrinkage 5.91%. Fig. 1b; control stream",
    disposition_slots=(("parent", (("CL", "CL"), ("V1", "VC"), ("Q2", "Q"), ("V2", "VP"))),),
    reference_weight_kg=74.19,
    reference_subject="WT 74.19 kg, AGE 32 years, not a CYP2D6 poor metaboliser, SEXO = 0",
    notes=(
        "No %RSE is printed; the 'SE' row of Fig. 1b is the SE of ETABAR. Q's unit "
        "is printed as 1/hr in the figure and L/hr in the control stream; the "
        "control stream is followed because the model code uses Q as a flow between "
        "two volumes. The VC-Q-VP block also carries covariances 0.0333, 0.0459 and 0.0121."
    ),
)

_ARIPIPRAZOLE_SEARCH = DrugSearch(
    drug="aripiprazole",
    queries=_standard_queries("aripiprazole", (157, 77, 17, 8), 25),
    candidates=(
        Candidate(_KOUE_2007, "17978491", "transcribed", "healthy adults, dense sampling, every disposition parameter estimated"),
        Candidate(_KNIGHTS_2015, "25821065", "transcribed", "the sponsor's 24-study oral model; full control stream published"),
        Candidate("Kim JR, Seo HB, Cho JY, et al. 2008. Br J Clin Pharmacol 66(6):802-10", "19032724", "summariser-only",
                  "PMC2675764 is open, but Europe PMC full text returns HTTP 500, the PMC page serves a browser check to scripts, "
                  "Wiley returns 403, and the only route that answered was a summarising fetch, which gave the final V/F as 192 L "
                  "on one request and 193 L on another"),
        Candidate("Zhang C, Jiang L, Hu K, et al. 2024. Front Psychiatry 15:1377268", "38957736", "not-transcribed",
                  "OA adult TDM, ka fixed to Kim 2008, one compartment from troughs, omega scale unstated; Koue and Knights are richer"),
        Candidate("Jeon JY, Chae SW, Kim MG. 2016. Int J Clin Pharmacol Ther 54(4):293-304", "26902506", "paywalled", "two-compartment, healthy Korean; no values in abstract"),
        Candidate("long-acting injectable models (Hard 2017/2018, Wang 2022, Bandín-Vilar 2024 and others)", "29943125; 28597226; 34979059; 38245977", "out-of-scope", "depot formulation"),
        Candidate("paediatric models (Xin 2022; Hermans 2023)", "36532742; 37222228", "out-of-scope", "paediatric only"),
    ),
    outcome="transcribed",
    summary="two adult oral models transcribed, one healthy-volunteer and one sponsor; Kim 2008 refused as summariser-only",
)


# ---------------------------------------------------------------------------
# Risperidone
# ---------------------------------------------------------------------------

_STORSET_2024 = "Størset E, Bråten LS, Ingelman-Sundberg M, et al. 2024. Impact of CYP2D6*2, CYP2D6*35, rs5758550, and related haplotypes on risperidone clearance in vivo. Eur J Clin Pharmacol 80(10):1531-1541"

_RISPERIDONE_STORSET_2024 = LiteraturePopPK(
    drug="risperidone",
    formulation="oral",
    analyte="parent-and-metabolite",
    citation=_STORSET_2024,
    pmid="38963454",
    doi="10.1007/s00228-024-03721-6",
    read_from="PMC11393095 Europe PMC full-text XML (Methods, Table 1, Table 2, Discussion)",
    population=(
        "512 Norwegian TDM patients (Oslo), 275 male / 237 female, age 37 (18-89), "
        "daily dose 3 (0.5-12) mg, serum sampled 13 (9-30) h after dose, 1-41 "
        "samples per subject; CYP2D6 genotyped; no body-size data"
    ),
    structural_model=(
        "The structural model consisted of one compartment for risperidone (parent) "
        "and one compartment for 9-hydroxyrisperidone (metabolite), both with linear "
        "elimination. ... a first-order absorption rate constant fixed to a "
        "previously reported value of 2.01 h-1 [31]."
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the oral bioavailability (F) of risperidone and the fraction of risperidone "
        "converted into 9-hydroxyrisperidone (f met) were not identifiable, and the "
        "reported disposition parameters of the parent and metabolite are therefore "
        "reported as apparent values (e.g. CL/F, CL met /[F × f met ])"
    ),
    software="NONMEM 7.5.1, FOCE-I",
    n_subjects=512,
    n_observations=1565,
    parameters=(
        _p("ka", 2.01, "1/h", "Table 2, '2.01 (fix)'"),
        _p("V/F", 333.0, "L", "Table 2, bootstrap 95% CI 177-586"),
        _p("CL/F base", 4.2, "L/h", "Table 2, bootstrap 95% CI 3.6-4.9"),
        _p("CL/F CYP2D6*1 (per allele)", 23.2, "L/h", "Table 2, bootstrap 95% CI 15.4-31.6"),
        _p("CL/F typical normal metabolizer (CYP2D6*1/*1)", 50.6, "L/h",
           "Discussion: 'In a typical normal metabolizer (e.g., CYP2D6 *1/*1 ), risperidone clearance is predicted to be 50.6 L/h'"),
        _p("NFIB-C (fraction of NFIB-T)", 1.41, None, "Table 2"),
        _p("Age breaking point (risperidone)", 34.0, "years", "Table 2"),
        _p("Reduction per year (risperidone)", -0.9, "%/year", "Table 2"),
        _p("Vmet/F × fmet", 96.0, "L", "Table 2, bootstrap 95% CI 77-127"),
        _p("CLmet/F × fmet", 8.0, "L/h", "Table 2, bootstrap 95% CI 7.5-8.6"),
        _p("Age breaking point (9-hydroxyrisperidone)", 39.0, "years", "Table 2"),
        _p("Reduction per year (9-hydroxyrisperidone)", -1.3, "%/year", "Table 2"),
    ),
    fixed_parameters=(("ka", "ref 31 = Korell J et al. 2017, PMID 28614899"),),
    variability=(
        Variability("CL/F base", 86.0, "percent-cv", source="Table 2, 'BSV CL/F (% CV)'"),
        Variability("CLmet/F × fmet", 49.0, "percent-cv", source="Table 2; correlation with BSV CL/F 0.43"),
    ),
    covariates=(
        Covariate("CL/F base", "CYP2D6 diplotype, NFIB, age",
                  "CL risperidone = (CL base + [CL CYP2D6, allele1 + CL CYP2D6, allele2 ] × CL NFIB [if CC or CT ]) "
                  "× (1–CL age × [Age–Age BP ]) [if Age ≥ Age BP ]",
                  source="Table 2 footnote"),
        Covariate("CLmet/F × fmet", "age",
                  "CL 9-hydroxyrisperidone = CL × (1–CL age × [Age– Age BP ]) [if Age ≥ Age BP ]",
                  source="Table 2 footnote"),
        Covariate("CL/F CYP2D6*1 (per allele)", "CYP2D6 allele",
                  "allele activity as fraction of *1: *2-rs5758550G 0.66, *35 0.57, *2-rs5758550A 0.30, *9 0.39, "
                  "*10 0.32, *41 0.15, *3-*6 0 (fix)",
                  source="Table 2"),
    ),
    residual_error="risperidone proportional 58% + additive 0.06 ng/mL; 9-hydroxyrisperidone proportional 38%. Table 2",
    disposition_slots=(
        ("parent", (("CL", "CL/F typical normal metabolizer (CYP2D6*1/*1)"), ("V1", "V/F"))),
        ("metabolite", (("CL", "CLmet/F × fmet"), ("V1", "Vmet/F × fmet"))),
    ),
    reference_subject="CYP2D6 *1/*1, NFIB-T/T, age below both breaking points (34 and 39 years); body size not modelled",
    notes=(
        "Samples were drawn 9-30 h after a dose, in a drug whose parent half-life "
        "in normal metabolizers is about 3 h, so the parent's V/F rests on late, "
        "low concentrations (bootstrap 95% CI 177-586 L)."
    ),
)

_RISPERIDONE_SEARCH = DrugSearch(
    drug="risperidone",
    queries=_standard_queries("risperidone", (217, 129, 22, 16), 32),
    candidates=(
        Candidate(_STORSET_2024, "38963454", "transcribed", "adult, genotyped, CC-BY full text, full Table 2"),
        Candidate("Feng Y, Pollock BG, Coley K, et al. 2008. Br J Clin Pharmacol 66(5):629-39", "18771484", "summariser-only",
                  "PMC2661978 is open but every raw route failed (Europe PMC full text HTTP 500, PMC browser check, Wiley 403); "
                  "only a summarising fetch answered. The age-on-CLM functional form could not be read at all"),
        Candidate("Korell J, Green B, Remmerie B, Vermeulen A. 2017. CPT Pharmacometrics Syst Pharmacol 6(9):589-95", "28614899", "not-transcribed",
                  "Table S2.3 read in full, but the model is a two-plus-two-compartment parent-metabolite fit with first-pass "
                  "conversion and a CYP2D6 mixture, refitted by FO, whose build data include paediatric oral studies; "
                  "its paliperidone ER sub-model is transcribed under paliperidone"),
        Candidate("Vandenberghe F, et al. 2015. Clin Pharmacokinet 54(12):1259-72", "26129906", "not-transcribed",
                  "only the authors' submitted manuscript is open; the published values may differ"),
        Candidate("Reeves S, et al. 2021. Br J Psychiatry 218(5):268-75; Roughley M, et al. 2023", "33176899; 36791792", "out-of-scope",
                  "elderly Alzheimer's disease cohorts (CATIE-AD), mean age 78; not the adult schizophrenia population"),
        Candidate("Vermeulen A, et al. 2007. J Pharmacokinet Pharmacodyn 34(2):183-206", "17136449", "paywalled", "closed; values exist only second-hand in a 2025 review"),
        Candidate("Yoo HD, et al. 2012. J Pharmacokinet Pharmacodyn 39(4):329-41", "22623266", "paywalled", "healthy Korean, 2 mg single dose"),
        Candidate("Locatelli I, et al. 2010. Eur J Pharm Sci 41(2):289-98", "20599499", "paywalled", "the listed repository copy returns 410 Gone"),
        Candidate("Ji S, et al. 2016. Int J Clin Pharmacol Ther 54(5):378-89", "27007997", "paywalled", "closed"),
        Candidate("Thyssen A, et al. 2010. Clin Pharmacokinet 49(7):465-78", "20528007", "paywalled", "closed; pooled paediatric and adult"),
        Candidate("long-acting injectable risperidone models", "various", "out-of-scope", "depot formulation"),
    ),
    outcome="transcribed",
    summary="one adult oral model transcribed; every adult oral model found co-fits 9-hydroxyrisperidone",
)


# ---------------------------------------------------------------------------
# Paliperidone (oral extended release)
# ---------------------------------------------------------------------------

_KORELL_2017 = "Korell J, Green B, Remmerie B, Vermeulen A. 2017. Determination of Plasma Concentration Reference Ranges for Risperidone and Paliperidone. CPT Pharmacometrics Syst Pharmacol 6(9):589-595"

_PALIPERIDONE_KORELL_2017 = LiteraturePopPK(
    drug="paliperidone",
    formulation="oral extended release (OROS, INVEGA)",
    analyte="parent",
    citation=_KORELL_2017,
    pmid="28614899",
    doi="10.1002/psp4.12217",
    read_from=(
        "PMC5613178 Europe PMC full-text XML and supplementary files: Online "
        "Supplement 2 Figure S2.1 and Table S2.1 (p. 2), Online Supplement 1 Table "
        "S1.1, NONMEM control stream 'POPPK pali OROS sim 3mg'"
    ),
    population=(
        "'developed in 327 individuals across five clinical studies' (Table S1.1): "
        "three adult studies (healthy elderly and young, healthy males 3-15 mg, renal "
        "impairment) and two adolescent studies (10-17 and 12-17 years); per-study n, "
        "observations and ages not printed"
    ),
    structural_model=(
        "Figure S2.1: depot dose with zero-order input of duration D1 after Tlag, "
        "first-order Ka into Vc/F, peripheral Vp/F linked by CLp/F, elimination "
        "CL/F; F1 = 'apparent bioavailability (changes with occasion)'"
    ),
    apparent_parameters=True,
    apparent_evidence="CL/F = apparent clearance; ... V c/F = apparent central volume of distribution",
    software="NONMEM 7.2 for the simulations; the original estimation method is not stated",
    n_subjects=327,
    n_observations=None,
    parameters=(
        _p("CL/F", 10.9, "L/h", "Table S2.1, SE%", rse=11.5),
        _p("Effect of weight on CL/F", 0.727, None, "Table S2.1", rse=21.7),
        _p("Effect of CrCL on CL/F", 0.0240, None, "Table S2.1", rse=46.3),
        _p("CLp/F", 22.0, "L/h", "Table S2.1", rse=10.0),
        _p("Vc/F", 198.0, "L", "Table S2.1", rse=6.7),
        _p("Vp/F", 244.0, "L", "Table S2.1", rse=6.8),
        _p("F1", 1.0, None, "Table S2.1, '1 FIX'"),
        _p("Tlag", 0.761, "h", "Table S2.1", rse=8.7),
        _p("D1", 25.0, "h", "Table S2.1 prints '25.' (SE 0.7%); control stream THETA5 = 25.4", rse=0.7),
        _p("Ka", 0.630, "1/h", "Table S2.1", rse=2.0),
    ),
    fixed_parameters=(("F1", "fixed to 1 as the reference occasion; between-occasion variability estimated"),),
    variability=(
        Variability("CL/F", 44.4, "percent-cv", 18.1, source="Table S2.1 BSV CV% (SE%)"),
        Variability("Vc/F", 34.5, "percent-cv", 31.3, source="Table S2.1"),
        Variability("Vp/F", 28.6, "percent-cv", 24.5, source="Table S2.1"),
        Variability("Ka", 59.9, "percent-cv", 20.3, source="Table S2.1"),
        Variability("F1", 17.4, "percent-cv", 14.1, inter_occasion=True, source="Table S2.1, BOV"),
    ),
    covariates=(
        Covariate("CL/F", "weight, creatinine clearance",
                  "CL = θ1 × (weight (kg)/74.4)θ8 + CrCL (mL/min) ×θ9, where CrCL is capped at 150 mL/min.",
                  source="Table S2.1 footnote"),
    ),
    residual_error="RUV 20.5 CV% (SE 4.6%), on log-transformed data. Table S2.1",
    disposition_slots=(("parent", (("CL", "CL/F"), ("V1", "Vc/F"), ("Q2", "CLp/F"), ("V2", "Vp/F"))),),
    reference_weight_kg=74.4,
    reference_subject=(
        "74.4 kg ('adult model 74.4' in the control stream). No reference creatinine "
        "clearance: CrCL enters additively, so the typical clearance at 74.4 kg is "
        "10.9 + 0.0240 × CrCL L/h for CrCL in [0, 150] mL/min"
    ),
    notes=(
        "Not a de-novo journal model: the main text calls it 'A previously developed "
        "population PK model' and its ref 6 is a J&J internal report on children and "
        "adolescents. The only original adult oral ER popPK, Cirincione et al. 2007, "
        "was published as an ASCPT abstract (Clin Pharmacol Ther 81:S19) with no "
        "parameter table. Table S2.1 prints no unit for θ9; the footnote equation "
        "makes θ9 × CrCL a clearance in the units of θ1."
    ),
)

_PALIPERIDONE_SEARCH = DrugSearch(
    drug="paliperidone",
    queries=_standard_queries("paliperidone", (92, 56, 9, 6), 13) + (
        _q("Europe PMC", 'paliperidone AND "extended release" AND "population pharmacokinetic*"', 34),
        _q("Europe PMC", '(AUTH:"Cirincione B") AND paliperidone', 0),
    ),
    candidates=(
        Candidate(_KORELL_2017, "28614899", "transcribed", "only oral ER parameter table in the journal literature; mixed adult/adolescent build"),
        Candidate("Cirincione B, Redman M, Fiedler-Kelly J, Ludwig E, Vermeulen A. 2007. Clin Pharmacol Ther 81:S19 (ASCPT abstract PI-21)",
                  "none", "conference-abstract-only",
                  "abstract only: two-compartment, zero-order input 23.9 h, lag 0.668 h, CL/F 13.8 L/h, Vss/F 487 L; no table, IIV or RSE"),
        Candidate("Samtani MN, et al. 2012. Clin Pharmacol 4:25-40", "22888277", "out-of-scope",
                  "simulation paper reusing Cirincione 2007's estimates; no new estimation"),
        Candidate("Gomeni R, Bressolle-Gomeni F. 2023. J Pharmacokinet Pharmacodyn 50(2):89-96", "36484885", "out-of-scope",
                  "fit to digitised mean curves from an EMA document, not individual data"),
        Candidate("Thyssen A, et al. 2010. Clin Pharmacokinet 49(7):465-78", "20528007", "out-of-scope",
                  "an oral risperidone model; paliperidone appears only as the metabolite"),
        Candidate("paliperidone palmitate models (Samtani 2009/2011, Magnusson 2017, Shimizu 2020 and others)", "19725593; 21936586; 27743205; 31876117", "out-of-scope", "depot formulation"),
    ),
    outcome="transcribed",
    summary="one oral ER model transcribed, from a supplement; its build set includes adolescents",
)


LITERATURE_POPPK_RECORDS = (
    _CLOZAPINE_LI_2012,
    _CLOZAPINE_REEVES_2024,
    _QUETIAPINE_CHEN_2024,
    _QUETIAPINE_ZHENG_2024,
    _ARIPIPRAZOLE_KOUE_2007,
    _ARIPIPRAZOLE_KNIGHTS_2015,
    _RISPERIDONE_STORSET_2024,
    _PALIPERIDONE_KORELL_2017,
)

DRUG_SEARCHES = (
    _CLOZAPINE_SEARCH,
    _QUETIAPINE_SEARCH,
    _ZIPRASIDONE_SEARCH,
    _ARIPIPRAZOLE_SEARCH,
    _RISPERIDONE_SEARCH,
    _PALIPERIDONE_SEARCH,
)


def records_for(drug):
    return tuple(r for r in LITERATURE_POPPK_RECORDS if r.drug == drug)


def drug_search(drug):
    for search in DRUG_SEARCHES:
        if search.drug == drug:
            return search
    raise KeyError("P25.1 did not search %r" % drug)


def searched_and_empty_drugs():
    return tuple(s.drug for s in DRUG_SEARCHES if s.outcome == "searched-and-empty")


def record(drug, first_author_year):
    """The record whose citation starts ``first_author_year``, e.g. ``"Koue"``."""
    matches = [r for r in records_for(drug) if r.citation.startswith(first_author_year)]
    if len(matches) != 1:
        raise KeyError("%s / %s matches %d records" % (drug, first_author_year, len(matches)))
    return matches[0]


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _disposition_values(rec, entity, clearance_increment=0.0):
    slots = dict(dict(rec.disposition_slots)[entity])
    units = {}
    values = {}
    for slot, printed in slots.items():
        parameter = rec.parameter(printed)
        units[slot] = parameter.require_units()
        values[slot] = parameter.value
    volume_unit = {units[s] for s in units if s.startswith("V")}
    flow_unit = {units[s] for s in units if not s.startswith("V")}
    if len(volume_unit) != 1 or len(flow_unit) != 1 or (
        volume_unit.pop(), flow_unit.pop()) not in _UNIT_FAMILIES:
        raise ValueError("%s %s: inconsistent disposition units %r" % (rec.key, entity, units))
    values["CL"] += clearance_increment
    return values


def disposition_rate_constants(rec, entity, clearance_increment=0.0):
    """Hybrid rate constants for one entity, most negative first, in 1/h.

    A parallel elimination rate constant (Reeves 2024's ``km``) is added to the
    one-compartment rate; it is refused on a multi-compartment mapping, where it
    would need a micro-constant this module does not invent.
    """
    values = _disposition_values(rec, entity, clearance_increment)
    extra = dict(rec.extra_elimination_slots).get(entity)
    if extra is not None:
        if "Q2" in values:
            raise ValueError("%s: parallel elimination on a two-compartment entity" % (rec.key,))
        parameter = rec.parameter(extra)
        if parameter.require_units() != "1/h":
            raise ValueError("%s: %s is not in 1/h" % (rec.key, extra))
        return (-(values["CL"] / values["V1"] + parameter.value),)
    return _hybrid_rate_constants(values["CL"], values["V1"], values.get("Q2"), values.get("V2"))


def terminal_half_life_h(rec, entity, clearance_increment=0.0):
    """ln 2 over the slowest disposition eigenvalue of ``entity``."""
    return math.log(2.0) / abs(disposition_rate_constants(rec, entity, clearance_increment)[-1])


def terminal_phase_auc_fraction(rec, entity, clearance_increment=0.0):
    """Share of the bolus AUC carried by the slowest phase; 1.0 for one compartment."""
    lambdas = disposition_rate_constants(rec, entity, clearance_increment)
    if len(lambdas) == 1:
        return 1.0
    values = _disposition_values(rec, entity, clearance_increment)
    k21 = values["Q2"] / values["V2"]
    # For a two-compartment bolus, A_i is proportional to (k21 + lambda_i) /
    # (lambda_i - lambda_j), and the area of phase i is A_i / -lambda_i.
    fast, slow = lambdas
    area_fast = ((k21 + fast) / (fast - slow)) / -fast
    area_slow = ((k21 + slow) / (slow - fast)) / -slow
    return area_slow / (area_fast + area_slow)


#: Label half-lives for the two drugs P21.3 had no row for. Same convention as
#: :data:`neurocascade.human.fda_popk.LABEL_HALF_LIVES`, read from openFDA
#: ``drug/label.json`` on 2026-10-01 without a browser User-Agent.
LITERATURE_LABEL_HALF_LIVES = (
    LabelHalfLife(
        drug="clozapine",
        entity="clozapine, steady state",
        low_h=12.0,
        high_h=12.0,
        quote=(
            "The mean elimination half-life of clozapine after a single 75 mg dose "
            "was 8 hours (range: 4 to 12 hours), compared to a mean elimination "
            "half-life of 12 hours (range: 4 to 66 hours), after achieving steady "
            "state with 100 mg twice daily dosing."
        ),
        source="NDA 019758 (CLOZARIL), SPL effective 2025-06-06, set 90876802-0e3a-44c9-9ff7-1754dfbe736a",
        innovator=True,
    ),
    LabelHalfLife(
        drug="quetiapine",
        entity="quetiapine",
        low_h=6.0,
        high_h=6.0,
        quote=(
            "Elimination of quetiapine is mainly via hepatic metabolism with a mean "
            "terminal half-life of about 6 hours within the proposed clinical dose range."
        ),
        source="NDA 020639 (SEROQUEL), SPL effective 2025-01-22, set 0584dda8-bc3c-48fe-1a90-79608f78e8a0",
        innovator=True,
    ),
)


def literature_label_half_life(drug, entity):
    for item in LITERATURE_LABEL_HALF_LIVES:
        if item.drug == drug and item.entity == entity:
            return item
    return label_half_life(drug, entity)


@dataclass(frozen=True)
class HalfLifeComparison:
    """A literature-model half-life, or a covariate-bounded interval of them,
    against a label half-life."""

    drug: str
    source: str
    entity: str
    predicted_low_h: float
    predicted_high_h: float
    label: LabelHalfLife
    terminal_auc_fraction: float
    n_compartments: int
    comment: str

    def _fold(self, hours):
        if self.label.contains(hours):
            return 1.0
        if hours > self.label.high_h:
            return hours / self.label.high_h
        return self.label.low_h / hours

    @property
    def fold_error(self):
        """Worst fold over the predicted interval; a point prediction has one."""
        return max(self._fold(self.predicted_low_h), self._fold(self.predicted_high_h))

    @property
    def agrees_within_twofold(self):
        return self.fold_error <= 2.0


#: ``(drug, first author, entity, label entity, clearance bound, comment)``.
#: Paired by hand. The clearance bound, used only for Korell 2017, is
#: ``(printed slope, low, high, why)``: the clearance increment ``slope × x``
#: over a covariate range the source itself prints, because the source has no
#: reference value for it and choosing one would be inventing a subject.
_PAIRINGS = (
    ("clozapine", "Li", "parent", "clozapine, steady state", None,
     "Compared against the label's steady-state mean, 12 h. The label also prints the "
     "range 4-66 h, which would contain every prediction here and so tests nothing; "
     "the mean was chosen before computing. The model is fitted to sparse steady-state "
     "TDM data, the same regimen class as the label statement."),
    ("clozapine", "Reeves", "parent", "clozapine, steady state", None,
     "Fails at 3.07-fold. The paper's own reference half-life is 36 h and this "
     "reproduces it (36.8 h), so the disagreement is between the model and the label, "
     "not an arithmetic slip. Its V/F of 1460 L is 2.8 times Li 2012's 526 L at a similar "
     "clearance, and V/F is the parameter sparse TDM troughs identify least (87% shrinkage)."),
    ("quetiapine", "Chen", "parent", "quetiapine", None,
     "Fails at 2.41-fold, driven by V/F = 2460 L (SE 33.5%, bootstrap 1222-5163 L) with "
     "ka fixed and only 154 sparse TDM samples. The volume, not the clearance, is out of line."),
    ("quetiapine", "Zheng", "parent", "quetiapine", None,
     "Passes at 1.24-fold, from the same group, method and borrowed ka as Chen 2024, "
     "with a V/F of 530 L that is itself poorly identified (SE 59.6%)."),
    ("aripiprazole", "Koue", "parent", "aripiprazole, non poor metabolisers", None,
     "CYP2D6 Group 1, normal metabolisers. Group 2 gives 75.1 h and Group 3 107.8 h; the "
     "label's 75 h describes non-poor-metabolisers as a whole, so Group 1 is the "
     "conservative choice and was fixed before computing."),
    ("aripiprazole", "Knights", "parent", "aripiprazole, non poor metabolisers", None,
     "The sponsor's 24-study oral model at its centring subject (74.19 kg, 32 years, "
     "not a poor metaboliser). Same clinical programme as the label, so not independent."),
    ("risperidone", "Størset", "parent", "risperidone, extensive metabolisers", None,
     "CYP2D6 *1/*1 typical clearance as printed in the Discussion. Passes at 1.52-fold "
     "with a V/F informed by samples 9-30 h post-dose."),
    ("risperidone", "Størset", "metabolite", "9-hydroxyrisperidone, extensive metabolisers", None,
     "Fails at 2.52-fold low. The metabolite parameters are Vmet/(F·fmet) and "
     "CLmet/(F·fmet); the unidentified F·fmet cancels in the ratio, so this is a real "
     "8.3 h against the label's 21 h, from a one-compartment metabolite with no BSV "
     "on its volume fitted to sparse data."),
    ("paliperidone", "Korell", "parent", "paliperidone, extended-release oral",
     ("Effect of CrCL on CL/F", 0.0, 150.0,
      "CrCL from 0 to the 150 mL/min cap printed in the Table S2.1 footnote"),
     "No reference creatinine clearance exists, so the half-life is reported over the "
     "whole printed CrCL range at the 74.4 kg reference weight: 32.8 h at CrCL 0 to "
     "26.0 h at the 150 mL/min cap. Both ends pass. The OROS input is zero-order over "
     "about 25 h; the label's 23 h is a terminal value and the disposition half-life "
     "here is not absorption-limited because input has ended by then."),
)


def half_life_comparisons():
    """Every literature-model half-life against its label, in pairing order."""
    out = []
    for drug, author, entity, label_entity, bound, comment in _PAIRINGS:
        rec = record(drug, author)
        if bound is None:
            low = high = terminal_half_life_h(rec, entity)
            fraction = terminal_phase_auc_fraction(rec, entity)
        else:
            slope_name, x_low, x_high, _why = bound
            slope = rec.parameter(slope_name).value
            ends = [terminal_half_life_h(rec, entity, slope * x) for x in (x_low, x_high)]
            low, high = min(ends), max(ends)
            fraction = min(
                terminal_phase_auc_fraction(rec, entity, slope * x) for x in (x_low, x_high)
            )
        out.append(
            HalfLifeComparison(
                drug=drug,
                source=rec.short_citation,
                entity=entity,
                predicted_low_h=low,
                predicted_high_h=high,
                label=literature_label_half_life(drug, label_entity),
                terminal_auc_fraction=fraction,
                n_compartments=1 + ("Q2" in dict(dict(rec.disposition_slots)[entity])),
                comment=comment,
            )
        )
    return tuple(out)


def comparison_table():
    """The result rendered as a markdown table, one row per comparison."""
    lines = [
        "| drug | source | entity | predicted t½ (h) | label t½ (h) | fold | terminal AUC frac | cmt | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in half_life_comparisons():
        if c.predicted_low_h == c.predicted_high_h:
            predicted = "%.2f" % c.predicted_low_h
        else:
            predicted = "%.2f–%.2f" % (c.predicted_low_h, c.predicted_high_h)
        label = "%g" % c.label.low_h if c.label.low_h == c.label.high_h else "%g–%g" % (
            c.label.low_h, c.label.high_h)
        lines.append(
            "| %s | %s | %s | %s | %s | %.2f | %.3f | %d | %s |"
            % (c.drug, c.source, c.entity, predicted, label, c.fold_error,
               c.terminal_auc_fraction, c.n_compartments,
               "PASS" if c.agrees_within_twofold else "FAIL")
        )
    return "\n".join(lines)
