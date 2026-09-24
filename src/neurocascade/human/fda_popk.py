"""Human population PK transcribed from US FDA review documents (P21.3).

Where the numbers come from, and why that source
------------------------------------------------

P19.2 sourced one human population-PK set, oral haloperidol, out of the
published literature, and the search half of that task was longer than the
transcription half: the two studies that would have identified an absolute
bioavailability were paywalled, and the one open-access set that could be read
printed apparent parameters only. That is the normal condition of this
literature. A population-PK analysis is a regulatory deliverable long before it
is a paper, and the paper, when it exists at all, prints the model but not
always the estimates, and sits behind a paywall.

The US FDA publishes the regulatory deliverable itself. Every approved NDA has
an approval package at ``accessdata.fda.gov/scripts/cder/daf/`` containing a
Clinical Pharmacology and Biopharmaceutics Review (older applications) or a
Multi-Disciplinary Review (newer ones), and inside it a pharmacometrics section
that prints the structural model, the final parameter estimates, the covariate
model and the inter-individual variability, written by an FDA reviewer who in
several of the documents read here **re-ran the sponsor's model and disagreed
with it**. It is free, unpaywalled and in the public domain.

So each row below carries an application number and a review date the way
:mod:`neurocascade.human.pk_parameters` rows carry a PMID. That is the same
commitment: the number is traceable to a document a reader can fetch.

What reading twenty-six of these documents actually established
--------------------------------------------------------------

Four findings shaped this schema. Each is a distinction the module is built to
express, not a remark about the search.

1. **"Apparent" is decided by a sentence, not by the route and not by the
   symbol.** The obvious rule, oral is apparent and intravenous is absolute, is
   wrong on this evidence in both directions. NDA 022264 (Invega Sustenna, an
   intramuscular depot) prints a plain ``CL`` and says why: "The estimates of CL
   (which is equal to intravenous paliperidone CL reported previously) and KA
   substantiated that paliperidone palmitate offers complete bioavailability".
   NDA 216352 (Erzofri), a later application for the *same molecule in the same
   kind of formulation*, prints ``CL/F`` and ``V/F`` throughout. Two FDA reviews
   of one drug disagree about whether its bioavailability was identified.

   The fallback rule, read the symbol, is also wrong, and the counter-example
   matters because it would have been silently mis-recorded: NDA 213586 (Uzedy)
   prints bare ``CL`` and ``V`` in every table and then defines them in the
   table legend as "CL=apparent clearance; V=apparent volume of distribution",
   and elsewhere equates its own ``V`` with the oral ``V/F`` of risperidone.
   NDA 210655 (Perseris) does the same, writing ``V_OR`` and describing it as
   "apparent central volume of distribution".

   So ``apparent_parameters`` is set from the sentence, that sentence is stored
   in ``apparent_evidence`` as a required field, and the only implication
   :meth:`FDAPopPK.__post_init__` enforces is the one that holds in every
   document read: a symbol written with an explicit ``/F`` cannot belong to a
   record claiming absolute parameters. The converse is not enforced, because a
   bare symbol carries no information here.

2. **A missing number has three different causes and only one of them is
   permanent.** The obvious reading of a caption followed by white space is that
   the sponsor's estimates were withheld. Sometimes they were: ``(b) (4)``
   marks material redacted as confidential commercial information, and it is
   unrecoverable. But most of the blank tables here are not redactions at all,
   they are **raster images with no text layer**, which ``pdftotext`` skips in
   silence and which optical character recognition reads back. And a third
   group is neither: the review contains no population-PK analysis because none
   was ever fitted. NDA 022047 (Seroquel XR) is 4,420 lines of exclusively
   non-compartmental analysis with no NONMEM run anywhere in it. Collapsing
   these three into "missing" would have thrown away recoverable numbers and
   would have misrepresented quetiapine, so ``extraction_barrier`` names which
   one applies.

3. **A parameter table can print an estimate without printing its unit.** The
   FDA reviewer's own final model for amisulpride, Table 97 of NDA 209510, is a
   two-column list of parameter names and values with no unit column. The units
   are inferable from the narrative, but inferring is not transcribing, so
   ``units`` is ``None`` on those parameters and
   :meth:`PopPKParameter.require_units` raises rather than guessing.

4. **The structural model is not predictable from the drug or the brief.** The
   task that commissioned this work said not to assume a compartment count, and
   that was correct: the same molecule is fitted one-compartment in one
   application and two-compartment in another, and the depot formulations use
   input models with no analogue in :mod:`neurocascade.human.pk_parameters` at
   all, including dual zero-order-plus-first-order absorption with an estimated
   fraction ``F2``, and, for the three-month paliperidone formulation, two
   parallel saturable absorption processes with a Hill coefficient. So
   ``structural_model`` stores the sentence as printed and the parameters are
   keyed by the symbol the source used, rather than being mapped onto fixed
   slots.

Why this is a separate module from ``pk_parameters``
----------------------------------------------------

:class:`~neurocascade.human.pk_parameters.HumanPKRecord` has fixed slots for a
one-, two- or three-compartment mamillary model with first-order absorption and
a single ``ka``. Most of what the FDA reviews actually fitted does not fit in
those slots, and the way to make it fit would be to drop the parameters that
have no slot, which silently converts a dual-input depot model into a
first-order one that no source fitted. Rule: never substitute a different model
for the one that was measured. So the transcription lives here, keyed by
printed name. The one bridge to the rest of the project is
``mamillary_slots``, filled by hand per row and only for the rows whose
disposition genuinely is a mamillary model; every other row leaves it empty and
:func:`terminal_half_life_h` refuses them by name rather than guessing.

What is here and what is not
----------------------------

This module is a transcription, and by the project's own rule a transcription
is not a validated claim. The claim P21.3 puts on the board is in
:func:`half_life_comparisons`: for the sets that are complete enough, the
terminal half-life implied by the population-PK parameters is computed and
compared against the half-life printed in the drug's approved US label, a
document written from non-compartmental analysis by a different process. That
is a prediction against a number nobody in this repository chose. It is not a
strong test of the model, since both documents describe the same clinical
programme, and the ledger row says so.

Drugs whose population PK could not be sourced are recorded here as rows with
``status="searched-and-empty"`` naming the documents read, not omitted and not
defaulted.
"""
import math
from dataclasses import dataclass
from types import MappingProxyType

#: Why a parameter table could not be transcribed. The three are not
#: interchangeable and the distinction was worth several recovered tables: see
#: finding 2 in the module docstring.
#:
#: ``redacted``
#:     The document prints ``(b) (4)`` in place of the values. Permanent.
#: ``image-table``
#:     The table is a raster image with no text layer. Recoverable by OCR, and
#:     P21.3 recovered several this way; a row still carrying this barrier is
#:     one where OCR was attempted and did not produce trustworthy digits.
#: ``not-fitted``
#:     No population-PK analysis exists in the review. The analysis is
#:     non-compartmental throughout. Nothing to recover.
#: ``estimates-not-printed``
#:     A population-PK analysis was performed and the review describes it,
#:     assesses it and agrees with it, but prints no parameter table anywhere.
#:     NDA 214835 is the case: the model sentence runs to four lines and the
#:     count of ``%RSE``, ``CL/F`` and "Parameter Estimates" occurrences in the
#:     whole document is zero. Not a redaction, not an image, not unfitted.
#: ``no-review-document``
#:     No approval package with a clinical pharmacology review is published for
#:     the application at all.
EXTRACTION_BARRIERS = (
    "redacted",
    "image-table",
    "not-fitted",
    "estimates-not-printed",
    "no-review-document",
)

#: Which chemical entity the parameter set describes. This is not bookkeeping:
#: for risperidone the reviews split three ways, with NDA 213586 fitting parent
#: and metabolite as separate sequential models, NDA 212849 and NDA 214835
#: fitting only the summed active moiety, and NDA 210655 fitting both parent and
#: metabolite with a volume scaling factor between them. A clearance of the
#: summed moiety and a clearance of the parent are different quantities, and
#: :mod:`neurocascade.human.d2_ec50` already records that risperidone's
#: occupancy EC50 is on the summed moiety, so a PK set fitted to parent alone
#: cannot be driven into it.
#:
#: ``parent-and-metabolite`` is separate from a pair of rows because some models
#: cannot be split into two: NDA 210655 estimates a single scaling factor ``SF``
#: relating the parent volume to the metabolite volume, so neither entity has a
#: volume of its own, and NDA 207533 estimates the metabolite's clearance only
#: as ``CLM/F/fm``, divided by the unidentified formation fraction as well as by
#: F. Splitting those into two rows would imply an independence the fit does not
#: have.
POPPK_ANALYTES = (
    "parent",
    "metabolite",
    "parent-and-metabolite",
    "active-moiety-sum",
)

#: How a variability estimate is scaled. The reviews print all four and do not
#: always say which; a %CV of 43.6 and a variance of 0.0476 are the same kind of
#: quantity written two ways, and reading one as the other is a factor of ten.
VARIABILITY_SCALES = (
    "percent-cv",
    "variance",
    "standard-deviation",
    "standard-deviation-logit",
)

#: A transcribed row either carries estimates or records a search that found
#: none. There is no third state, and in particular there is no "pending".
POPPK_STATUSES = ("transcribed", "searched-and-empty")


def _is_apparent_symbol(name):
    """True if a printed parameter symbol carries an ``/F``.

    Matches ``CL/F``, ``V/F``, ``Vc/F``, ``Q/F``, ``V3/F``. Deliberately does
    not match ``F``, ``F2`` or ``FINJS``, which are bioavailability parameters
    rather than disposition parameters divided by one.
    """
    return "/F" in name.upper()


def _is_disposition_symbol(name):
    """True if a symbol names a clearance or a volume.

    Only these carry the apparent/absolute distinction. An absorption rate
    constant, a lag time or a fraction does not.
    """
    head = name.upper().split("/")[0].strip()
    if head.startswith("CL") or head.startswith("Q"):
        return True
    return head.startswith("V") and not head.startswith("VAR")


@dataclass(frozen=True)
class PopPKParameter:
    """One estimate, named as the source named it."""

    #: The printed symbol, verbatim: ``CL/F``, ``KA``, ``D2``, ``F2``, ``ALAG1``.
    #: Not normalised, because the normalisation would have to decide questions
    #: the sources answer differently.
    name: str
    #: The point estimate as printed.
    value: float
    #: The unit as printed, or ``None`` when the table printed no unit column.
    #: ``None`` is a real state here, not a gap in the transcription: see
    #: finding 3 in the module docstring.
    units: str = None
    #: Relative standard error in percent, printed as %RSE or %SEE. ``None``
    #: when the table has no precision column at all, which is common.
    rse_percent: float = None
    #: Where in the document it was read: table number and, where the document
    #: paginates usefully, the page.
    source: str = ""

    def require_units(self):
        """The unit, or a refusal.

        A consumer that needs to convert has to know the unit. Where the
        document did not print one, this raises rather than assuming the
        conventional choice, because "conventional" is exactly the kind of
        assumption this project has been wrong about before.
        """
        if self.units is None:
            raise ValueError(
                "parameter %r was printed without a unit (%s); the unit is not "
                "inferred here" % (self.name, self.source or "no source note")
            )
        return self.units


@dataclass(frozen=True)
class PopPKVariability:
    """One inter-individual (or inter-occasion) variability estimate."""

    #: The parameter the variability is on, by the same printed symbol.
    parameter: str
    #: The estimate as printed.
    value: float
    #: Which of :data:`VARIABILITY_SCALES` ``value`` is on.
    scale: str
    #: Eta shrinkage in percent where the review printed it. High shrinkage
    #: means the individual estimates are pulled to the mean and the variability
    #: is barely identified; NDA 209510 prints 98.7% on one inter-compartmental
    #: clearance, whose variability estimate is therefore zero and meaningless.
    shrinkage_percent: float = None
    #: ``True`` for inter-occasion rather than inter-individual variability.
    inter_occasion: bool = False
    source: str = ""

    def __post_init__(self):
        if self.scale not in VARIABILITY_SCALES:
            raise ValueError(
                "unknown variability scale %r for %s; expected one of %s"
                % (self.scale, self.parameter, ", ".join(VARIABILITY_SCALES))
            )


@dataclass(frozen=True)
class FDAPopPK:
    """A population-PK analysis as one FDA review document printed it."""

    #: Generic drug name, lowercase.
    drug: str
    #: The formulation the analysis is of, in the document's own terms. Kept
    #: separate from ``route`` because "oral extended-release (OROS)" and "oral
    #: immediate-release" are both ``PO`` and are not the same PK.
    formulation: str
    #: ``PO``, ``IM``, ``IV`` or ``SC``.
    route: str
    #: One of :data:`POPPK_ANALYTES`. Where one review fits parent and
    #: metabolite in one simultaneous model, each entity is a separate row, so
    #: that neither can be picked up without its analyte label.
    analyte: str
    #: ``"NDA 213378"``. The analogue of a PMID for this source class.
    application_number: str
    #: The review's own completion or signature date, ISO, as printed. Where a
    #: file concatenates two reviews with different dates, each is a separate
    #: row.
    review_date: str
    #: Which document within the approval package: ``"Clinical Pharmacology and
    #: Biopharmaceutics Review(s)"``, ``"Multi-Disciplinary Review"``, and so on.
    review_document: str
    #: The structural model, quoted from the document rather than summarised,
    #: because the summary is where the compartment count gets invented.
    structural_model: str
    #: ``True`` when the disposition parameters are divided by an unidentified
    #: bioavailability. Enforced against the parameter symbols below.
    apparent_parameters: bool
    #: The sentence or symbol that decided ``apparent_parameters``. Required,
    #: because on this evidence the flag cannot be derived from the route.
    apparent_evidence: str
    #: Who was in the analysis dataset: age band, healthy or patient, and
    #: anything that limits transfer, such as an adolescents-only cohort.
    population: str
    #: :data:`POPPK_STATUSES`.
    status: str
    #: Estimating software and method as printed, e.g. ``"NONMEM 7.4.4, FOCEI"``.
    software: str = None
    n_subjects: int = None
    n_observations: int = None
    parameters: tuple = ()
    variability: tuple = ()
    #: One of :data:`EXTRACTION_BARRIERS`, or ``None``. Set on a
    #: ``searched-and-empty`` row to say which of the three reasons applies, and
    #: also on a ``transcribed`` row that is partial, e.g. one where the FDA
    #: reviewer's table was readable but the sponsor's was an image.
    extraction_barrier: str = None
    #: Which printed symbols fill the mamillary slots, as
    #: ``(("CL", "CL/F"), ("V1", "V/F"))``, or ``()`` when the structural model
    #: is not a mamillary model and no such mapping exists. This is the only
    #: place a printed name is mapped onto a modelling slot, and it is done by
    #: hand per row rather than by pattern-matching names.
    mamillary_slots: tuple = ()
    #: Which printed symbols fill the absorption slots, as
    #: ``(("ka", "Ka"), ("lag", "ALAG"))``. Kept separate from
    #: :attr:`mamillary_slots` because absorption is not part of the disposition
    #: model and a row can have one without the other: an intravenous fit has no
    #: absorption slots, and a rate-constant parameterisation has no mamillary
    #: ones. Hand-mapped per row, same as the disposition slots.
    absorption_slots: tuple = ()
    #: Which printed symbols carry the fixed allometric exponents, as
    #: ``(("cl_weight_exponent", "WT on CL/F"),)``. An exponent without the
    #: weight it is normalised to is useless, so a row carrying these must also
    #: carry :attr:`reference_weight_kg`.
    allometric_slots: tuple = ()
    #: The body weight of the reference subject the typical values belong to, in
    #: kg. ``None`` when the review never printed one, which is the common case:
    #: an allometric exponent fixed at 0.75 says nothing about the weight it is
    #: normalised to, and inferring 70 kg because 70 kg is conventional is
    #: exactly the assumption this registry exists to avoid. Absolute typical
    #: values mean nothing without it, so :mod:`neurocascade.human.fda_pk_bridge`
    #: refuses to convert a row that lacks one.
    reference_weight_kg: float = None
    #: The reference subject as the review described them, verbatim where
    #: possible. The covariate forest plots are usually the only place a review
    #: pins the reference subject down, and they pin all of it at once: age,
    #: weight, sex, race, smoking status, organ function, dose and fed state.
    reference_subject: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.status not in POPPK_STATUSES:
            raise ValueError(
                "%s: unknown status %r" % (self.application_number, self.status)
            )
        if self.analyte not in POPPK_ANALYTES:
            raise ValueError(
                "%s: unknown analyte %r; expected one of %s"
                % (self.application_number, self.analyte, ", ".join(POPPK_ANALYTES))
            )
        if self.extraction_barrier is not None:
            if self.extraction_barrier not in EXTRACTION_BARRIERS:
                raise ValueError(
                    "%s: unknown extraction barrier %r"
                    % (self.application_number, self.extraction_barrier)
                )
        if self.status == "searched-and-empty":
            if self.parameters:
                raise ValueError(
                    "%s: a searched-and-empty row cannot carry parameters"
                    % self.application_number
                )
            if self.extraction_barrier is None:
                raise ValueError(
                    "%s: a searched-and-empty row must say which barrier "
                    "applies; 'missing' is not a finding"
                    % self.application_number
                )
        elif not self.parameters:
            raise ValueError(
                "%s: a transcribed row with no parameters should be recorded as "
                "searched-and-empty" % self.application_number
            )
        if not self.apparent_evidence:
            raise ValueError(
                "%s: apparent_parameters is not derivable from the route on this "
                "evidence, so the deciding sentence is required"
                % self.application_number
            )
        for parameter in self.parameters:
            if not _is_disposition_symbol(parameter.name):
                continue
            if _is_apparent_symbol(parameter.name) and not self.apparent_parameters:
                raise ValueError(
                    "%s: parameter %r is printed with an explicit /F but the "
                    "record claims absolute parameters"
                    % (self.application_number, parameter.name)
                )
        slots = dict(self.mamillary_slots)
        names = {parameter.name for parameter in self.parameters}
        for slot, printed in slots.items():
            if printed not in names:
                raise ValueError(
                    "%s: mamillary slot %r maps to %r, which is not a parameter "
                    "of this record" % (self.application_number, slot, printed)
                )
        if slots and "CL" not in slots:
            raise ValueError(
                "%s: a mamillary mapping needs a clearance slot"
                % self.application_number
            )
        if slots and "V1" not in slots:
            raise ValueError(
                "%s: a mamillary mapping needs a central volume slot"
                % self.application_number
            )
        for label, mapping in (
            ("absorption", self.absorption_slots),
            ("allometric", self.allometric_slots),
        ):
            for slot, printed in mapping:
                if printed not in names:
                    raise ValueError(
                        "%s: %s slot %r maps to %r, which is not a parameter of "
                        "this record"
                        % (self.application_number, label, slot, printed)
                    )
        if self.allometric_slots and self.reference_weight_kg is None:
            raise ValueError(
                "%s: allometric exponents are recorded without the weight they "
                "are normalised to, which makes them unusable"
                % self.application_number
            )

    @property
    def key(self):
        """Stable identifier for a row.

        Every component earned its place by a collision. The review date,
        because one PDF in this set contains two reviews of the same
        application completed a year apart. The analyte, because NDA 213586
        fits risperidone and 9-hydroxyrisperidone as two separate models in one
        review. The formulation, because NDA 210655 fits an oral sub-model and
        a subcutaneous depot sub-model in one review, with the same analyte and
        the same date.
        """
        return (
            self.drug,
            self.application_number,
            self.review_date,
            self.formulation,
            self.analyte,
        )

    def parameter(self, name):
        for item in self.parameters:
            if item.name == name:
                return item
        raise KeyError(
            "%s has no parameter %r; it has %s"
            % (self.application_number, name, ", ".join(sorted(
                item.name for item in self.parameters)) or "none")
        )

    @property
    def is_mamillary(self):
        return bool(self.mamillary_slots)

    @property
    def n_compartments(self):
        """Compartment count implied by the mamillary mapping, or ``None``.

        ``None`` is not "unknown": it means the fitted model is not a mamillary
        model, so the question does not apply.
        """
        if not self.mamillary_slots:
            return None
        slots = dict(self.mamillary_slots)
        return 1 + ("V2" in slots) + ("V3" in slots)

    def summary(self):
        head = "%s %s (%s, %s), %s %s" % (
            self.drug,
            self.formulation,
            self.route,
            self.analyte,
            self.application_number,
            self.review_date,
        )
        if self.status == "searched-and-empty":
            return "%s: searched and empty (%s). %s" % (
                head,
                self.extraction_barrier,
                self.notes,
            )
        basis = "apparent (X/F)" if self.apparent_parameters else "absolute"
        body = ", ".join(
            "%s=%g%s" % (p.name, p.value, " " + p.units if p.units else "")
            for p in self.parameters
        )
        return "%s: %s, %s\n  %s" % (head, basis, body, body and self.structural_model)


def _hybrid_rate_constants(cl, v1, q2=None, v2=None, q3=None, v3=None):
    """Eigenvalues of the mamillary model, most negative first.

    Written out rather than pulled from a library so the algebra is visible: the
    terminal half-life this feeds is the one number P21.3 compares against an
    external source, and it must be reproducible by hand.
    """
    k10 = cl / v1
    if q2 is None:
        return (-k10,)
    k12 = q2 / v1
    k21 = q2 / v2
    if q3 is None:
        b = k10 + k12 + k21
        c = k10 * k21
        disc = math.sqrt(max(b * b - 4.0 * c, 0.0))
        return (-(b + disc) / 2.0, -(b - disc) / 2.0)
    k13 = q3 / v1
    k31 = q3 / v3
    a1 = k10 + k12 + k21 + k13 + k31
    a2 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21
    a3 = k10 * k21 * k31
    roots = np_roots([1.0, a1, a2, a3])
    return tuple(sorted(-abs(root) for root in roots))


def np_roots(coefficients):
    """Real roots of a cubic with all-positive coefficients.

    The mamillary three-compartment characteristic polynomial has three real
    negative roots, so a closed-form trigonometric solution applies and no
    dependency on a root finder is needed.
    """
    a, b, c, d = coefficients
    b /= a
    c /= a
    d /= a
    p = c - b * b / 3.0
    q = 2.0 * b ** 3 / 27.0 - b * c / 3.0 + d
    if p > 0.0:
        raise ValueError("expected three real roots; got a positive depressed p")
    m = 2.0 * math.sqrt(-p / 3.0)
    argument = 3.0 * q / (p * m)
    argument = max(-1.0, min(1.0, argument))
    theta = math.acos(argument) / 3.0
    return [
        m * math.cos(theta - 2.0 * math.pi * k / 3.0) - b / 3.0
        for k in range(3)
    ]


def terminal_half_life_h(record):
    """Terminal half-life in hours implied by a mamillary parameter set.

    Only defined for rows with a mamillary mapping whose disposition parameters
    all print a unit. An apparent set gives an apparent half-life, which is the
    same number: the ``/F`` cancels between clearance and volume. That
    cancellation is the reason this is the comparison P21.3 can actually make,
    since almost every set here is apparent.
    """
    if not record.is_mamillary:
        raise ValueError(
            "%s fitted %r, which is not a mamillary model; no terminal "
            "half-life is defined for it without inventing one"
            % (record.application_number, record.structural_model[:60])
        )
    slots = dict(record.mamillary_slots)
    values = {}
    for slot, printed in slots.items():
        parameter = record.parameter(printed)
        unit = parameter.require_units()
        if slot.startswith("V"):
            if unit != "L":
                raise ValueError(
                    "%s: %s is in %r, not L" % (record.application_number, printed, unit)
                )
        elif unit != "L/h":
            raise ValueError(
                "%s: %s is in %r, not L/h" % (record.application_number, printed, unit)
            )
        values[slot] = parameter.value
    lambdas = _hybrid_rate_constants(
        values["CL"],
        values["V1"],
        values.get("Q2"),
        values.get("V2"),
        values.get("Q3"),
        values.get("V3"),
    )
    return math.log(2.0) / abs(lambdas[-1])


def terminal_phase_auc_fraction(record):
    """Fraction of the bolus AUC carried by the slowest exponential phase.

    The diagnostic that separates a terminal half-life worth comparing from one
    that is an artefact. A three-compartment fit can put a deep, low-flow
    compartment behind a very slow eigenvalue that contributes almost nothing to
    the area, in which case the "terminal half-life" is a curve feature the
    label's non-compartmental analysis never sampled and the two numbers are not
    measuring the same thing. Reporting the fold error without this would
    misattribute a sampling-window difference to a model disagreement.
    """
    slots = dict(record.mamillary_slots)
    if not slots:
        raise ValueError(
            "%s: no mamillary mapping, so there are no exponential phases"
            % record.application_number
        )
    value = {
        slot: record.parameter(printed).value for slot, printed in slots.items()
    }
    lambdas = _hybrid_rate_constants(
        value["CL"],
        value["V1"],
        value.get("Q2"),
        value.get("V2"),
        value.get("Q3"),
        value.get("V3"),
    )
    if len(lambdas) == 1:
        return 1.0
    peripheral_k = []
    if "Q2" in value:
        peripheral_k.append(value["Q2"] / value["V2"])
    if "Q3" in value:
        peripheral_k.append(value["Q3"] / value["V3"])
    areas = []
    for i, lam in enumerate(lambdas):
        numerator = 1.0
        for k_p1 in peripheral_k:
            numerator *= lam + k_p1
        denominator = 1.0
        for j, other in enumerate(lambdas):
            if j != i:
                denominator *= lam - other
        areas.append((numerator / denominator) / -lam)
    return areas[-1] / sum(areas)


# ---------------------------------------------------------------------------
# The transcription.
#
# Read from the PDFs at accessdata.fda.gov/drugsatfda_docs/nda/. Where a table
# is a raster image its digits were read off a rendering of the page rather than
# from a text layer, and the ``source`` note says which table and which page of
# the PDF, so that a reader can go to the same pixels.
# ---------------------------------------------------------------------------

_RISPERIDONE_PERSERIS_ORAL = FDAPopPK(
    drug="risperidone",
    formulation="oral tablet (reference arm of the PERSERIS analysis)",
    route="PO",
    analyte="parent-and-metabolite",
    application_number="NDA 210655",
    review_date="2018-07-16",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "Risperidone disposition was described using a one compartment model "
        "with first order absorption and first order elimination. A one "
        "compartment model was also used to describe the disposition of "
        "9-hydroxyrisperidone... The apparent volume of distribution of "
        "risperidone and 9-hydroxyrisperidone were related by a constant "
        "scaling factor (SF) that was estimated "
        "(V/Frisperidone = SF x V/F9-hydroxyrisperidone)"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the model description writes V/F for both entities, and Table 6 "
        "describes V_OR as the '9-hydroxyrisperidone apparent central volume "
        "of distribution'"
    ),
    population=(
        "315 adults with schizophrenia across 3 studies, subcutaneous 60-120 mg "
        "and oral 2-4 mg"
    ),
    status="transcribed",
    n_subjects=315,
    n_observations=10671,
    parameters=(
        PopPKParameter("ka_OR", 4.07, "1/h", 24.6, "Table 6, PDF p.19"),
        PopPKParameter("V_OR", 71.0, "L", 9.4, "Table 6, PDF p.19"),
        PopPKParameter("kr_el,OR", 0.0298, "1/h", 23.9, "Table 6, PDF p.19"),
        PopPKParameter("kr9_OR", 0.108, "1/h", 13.1, "Table 6, PDF p.19"),
        PopPKParameter("k9_el,OR", 0.0758, "1/h", 4.4, "Table 6, PDF p.19"),
        PopPKParameter("SF", 2.83, None, 10.0, "Table 6, PDF p.19"),
    ),
    variability=(
        PopPKVariability("V_OR", 36.7, "percent-cv", None, False, "Table 6"),
        PopPKVariability("kr_el,OR", 69.3, "percent-cv", None, False, "Table 6"),
        PopPKVariability("kr9_OR", 86.9, "percent-cv", None, False, "Table 6"),
        PopPKVariability("k9_el,OR", 15.8, "percent-cv", None, False, "Table 6"),
    ),
    notes=(
        "Parameterised in rate constants, not in clearance and volume, so no "
        "mamillary mapping is possible without dividing V_OR by an SF that "
        "belongs to the other entity. The review's own caveat is carried "
        "forward: 'Parameter estimates of the model should be interpreted with "
        "caution.' The two (b) (4) redactions in this document are on the "
        "polymer composition, not on the parameter tables."
    ),
)

_RISPERIDONE_PERSERIS_DEPOT = FDAPopPK(
    drug="risperidone",
    formulation="PERSERIS subcutaneous ATRIGEL depot",
    route="SC",
    analyte="parent-and-metabolite",
    application_number="NDA 210655",
    review_date="2018-07-16",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "dual absorption (two first-order inputs) into a two-compartment "
        "risperidone disposition model with two first-order elimination "
        "processes, one of which forms 9-hydroxyrisperidone, whose disposition "
        "is one-compartment"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "same analysis and same V/F notation as the oral sub-model above; no "
        "intravenous arm is in the dataset"
    ),
    population=(
        "315 adults with schizophrenia across 3 studies, subcutaneous 60-120 mg "
        "and oral 2-4 mg"
    ),
    status="transcribed",
    n_subjects=315,
    n_observations=10671,
    parameters=(
        PopPKParameter("ka1", 0.00599, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("ka2", 0.0185, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("kr_el", 0.0225, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("V", 139.0, "L", None, "Table 7 final, PDF p.22"),
        PopPKParameter("k_tr", 0.0237, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("kr_rp", 0.759, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("kr_pr", 0.00781, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("kr9", 0.187, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("k9_el", 0.0652, "1/h", None, "Table 7 final, PDF p.22"),
        PopPKParameter("SF", 1.11, None, None, "Table 7 final, PDF p.22"),
        PopPKParameter("BMI on ka1", 0.0241, None, None, "Table 7 final"),
        PopPKParameter("CYP2D6 IM on kr9", 0.468, None, None, "Table 7 final"),
        PopPKParameter("CYP2D6 PM on kr9", 0.929, None, None, "Table 7 final"),
    ),
    notes=(
        "Rate-constant parameterisation again, so no mamillary mapping. The "
        "depot release is carried by two first-order inputs plus a transit rate "
        "k_tr, which has no slot in pk_parameters at all."
    ),
)

_RISPERIDONE_RYKINDO = FDAPopPK(
    drug="risperidone",
    formulation="RYKINDO (LY03004) intramuscular microspheres",
    route="IM",
    analyte="active-moiety-sum",
    application_number="NDA 212849",
    review_date="2020-01-28",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "one-compartment PK model with a linear elimination and three-phase "
        "absorption i.e. zero-order immediate release, zero-order "
        "middle-release and first-order main-release"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the programme is a relative-bioavailability comparison against "
        "RISPERDAL CONSTA; F1, F2 and F3 are release fractions constrained to "
        "sum to one, so no absolute bioavailability is identified"
    ),
    population=(
        "adults with schizophrenia, studies CT-1S01/104 and CT-USA-102, "
        "intramuscular 12.5-50 mg every two weeks"
    ),
    status="transcribed",
    software="NONMEM",
    parameters=(
        PopPKParameter("KA (CT-1S01/104)", 0.288, "1/d", None, "Table 45, PDF p.234"),
        PopPKParameter("KA (CT-USA-102)", 0.380, "1/d", None, "Table 45, PDF p.234"),
        PopPKParameter("CL male", 186.7, "L/d", 6.30, "Table 45, PDF p.234"),
        PopPKParameter("CL female", 153.4, "L/d", 6.96, "Table 45, PDF p.234"),
        PopPKParameter("D2", 0.764, "d", None, "Table 45, PDF p.234"),
        PopPKParameter("F2", 0.148, None, None, "Table 45, PDF p.234"),
        PopPKParameter("ALAG3", 3.47, "d", None, "Table 45, PDF p.234"),
        PopPKParameter("K32", 0.118, "1/d", None, "Table 45, PDF p.234"),
        PopPKParameter("D3", 2.18, "d", None, "Table 45, PDF p.234"),
        PopPKParameter("F3", 0.421, None, None, "Table 45, PDF p.234"),
        PopPKParameter("ALAG1", 13.3, "d", None, "Table 45, PDF p.234"),
        PopPKParameter("F1", 0.430, None, None, "Table 45, PDF p.234"),
    ),
    variability=(
        PopPKVariability("KA", 27.0, "percent-cv", 7.84, False, "Table 45"),
        PopPKVariability("CL", 35.0, "percent-cv", 5.38, False, "Table 45"),
    ),
    extraction_barrier="image-table",
    notes=(
        "Table 45 prints no central volume, so the one-compartment disposition "
        "cannot be closed and no half-life follows from it; the row is "
        "transcribed but carries the image-table barrier because the volume "
        "row, if it exists, was not recoverable from the rendering. Units are "
        "per day, not per hour, which is the kind of thing that silently "
        "becomes a factor of 24. The companion Table 46 fits RISPERDAL CONSTA "
        "in the same run: KA 0.271 and 0.179 1/d, CL 108.0 L/d (%RSE 4.15), "
        "IIV CL 34%, D2 0.0467 d, F2 0.119, ALAG3 0.0417 d, K32 0.0830 1/d, "
        "D3 23.9 d, F3 0.126, ALAG1 27.0 d, F1 0.755."
    ),
)

_RISPERIDONE_UZEDY_PARENT = FDAPopPK(
    drug="risperidone",
    formulation="UZEDY (TV-46000) subcutaneous",
    route="SC",
    analyte="parent",
    application_number="NDA 213586",
    review_date="2022-04-15",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "1-compartmental model with double first-order absorption route and "
        "first-order elimination... a fast component (KA1)... and an indirect "
        "absorption process, with a slow absorption route (KA2) including 5 "
        "transit compartments"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the table legend reads 'CL=apparent clearance; RSE=relative standard "
        "error; V=apparent volume of distribution parent', and the text "
        "elsewhere equates this V with the oral V/F of risperidone. The "
        "symbols themselves are bare CL and V"
    ),
    population=(
        "694 subjects, healthy volunteers and adults with schizophrenia, 5 "
        "studies, subcutaneous 12.5-250 mg"
    ),
    status="transcribed",
    software="NONMEM 7.4.3, FOCEI",
    n_subjects=694,
    n_observations=25638,
    parameters=(
        PopPKParameter("CL", 14.3, "L/h", 2.9, "Table 52, PDF p.249"),
        PopPKParameter("V", 66.3, "L", 4.4, "Table 52, PDF p.249"),
        PopPKParameter("KA1", 0.000632, "1/h", 3.6, "Table 52, PDF p.249"),
        PopPKParameter("KA2", 0.000408, "1/h", 9.9, "Table 52, PDF p.249"),
        PopPKParameter("KTR", 0.0252, "1/h", 4.8, "Table 52, PDF p.249"),
        PopPKParameter("KA1ADMSITE1", 0.331, None, None, "Table 52, PDF p.249"),
        PopPKParameter("KA1BMI1", -1.1, None, None, "Table 52, PDF p.249"),
        PopPKParameter("KA1INJV1", -0.384, None, None, "Table 52, PDF p.249"),
        PopPKParameter("KA2BMI1", 1.7, None, None, "Table 52, PDF p.249"),
    ),
    variability=(
        PopPKVariability("CL", 82.3, "percent-cv", 6.6, False, "Table 52"),
        PopPKVariability("KA1", 51.0, "percent-cv", 32.7, False, "Table 52"),
        PopPKVariability("KA2", 254.2, "percent-cv", 25.8, False, "Table 52"),
    ),
    mamillary_slots=(("CL", "CL"), ("V1", "V")),
    notes=(
        "The narrative of this review misreports its own table: the prose "
        "attributes 42.8% to inter-individual variability on KA1, where Table "
        "52 gives 42.8% as the KA2-KA1 correlation and 51.0% as the IIV on "
        "KA1. Transcribed from the table. The 2023 cycle of the same "
        "application (Reference ID 5165749, Table 27) restates the same fit "
        "unrounded as CL 14.26 L/h and V 66.26 L; not a separate row because "
        "the estimates are the same fit, not a refit."
    ),
)

_RISPERIDONE_UZEDY_METABOLITE = FDAPopPK(
    drug="risperidone",
    formulation="UZEDY (TV-46000) subcutaneous",
    route="SC",
    analyte="metabolite",
    application_number="NDA 213586",
    review_date="2022-04-15",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "1-compartmental model with a first order conversion from risperidone "
        "and a first order elimination"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "fitted sequentially on the parent model above, whose CL and V the "
        "legend calls apparent; the metabolite parameters inherit that basis "
        "and also the unidentified formation fraction"
    ),
    population=(
        "694 subjects, healthy volunteers and adults with schizophrenia, 5 "
        "studies, subcutaneous 12.5-250 mg"
    ),
    status="transcribed",
    software="NONMEM 7.4.3, FOCEI",
    n_subjects=694,
    n_observations=25638,
    parameters=(
        PopPKParameter("CLMO", 5.78, "L/h", 2.322, "Table 53, PDF p.250"),
        PopPKParameter("VMO", 95.7, "L", 3.822, "Table 53, PDF p.250"),
    ),
    variability=(
        PopPKVariability("CLMO", 65.1, "percent-cv", 5.5, False, "Table 53"),
    ),
    mamillary_slots=(("CL", "CLMO"), ("V1", "VMO")),
    notes=(
        "A metabolite's CL/V ratio is its own elimination half-life only when "
        "the metabolite is not formation-rate-limited. For 9-hydroxyrisperidone "
        "formed from a slow-release subcutaneous depot it plainly is, so the "
        "half-life this row implies is a disposition property and the label's "
        "21 hours after oral dosing is not measuring the same thing."
    ),
)

_RISPERIDONE_RISVAN = FDAPopPK(
    drug="risperidone",
    formulation="RISVAN in-situ intramuscular microimplant",
    route="IM",
    analyte="active-moiety-sum",
    application_number="NDA 214835",
    review_date="2021-09-24",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "one-compartment disposition model with a complex absorption process, "
        "combining a small amount immediately entering the central active "
        "moiety compartment, two first-order absorption processes, and a "
        "combined zero-order and first order absorption process, and a "
        "first-order elimination"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "no intravenous arm and no absolute bioavailability anywhere in the "
        "document; flagged apparent on the same basis as every other depot "
        "analysis here, though with no printed symbol to check it against"
    ),
    population=(
        "447 adults, 5 trials, 6288 concentrations, intramuscular 25-100 mg"
    ),
    status="searched-and-empty",
    software="NONMEM 7.3",
    n_subjects=447,
    n_observations=6288,
    extraction_barrier="estimates-not-printed",
    notes=(
        "The review describes the model, states that 'The PopPK model was "
        "successfully reproduced. We agree with the Applicant's analyses', and "
        "never prints a number. Counts of 'CL/F', '%RSE' and 'Parameter "
        "Estimates' in the whole document are zero. Nothing here is redacted "
        "and nothing is an image: the table does not exist in the released "
        "document, which is why this row is searched-and-empty rather than a "
        "partial transcription."
    ),
)

_PALIPERIDONE_SUSTENNA = FDAPopPK(
    drug="paliperidone",
    formulation="INVEGA SUSTENNA paliperidone palmitate 1-month intramuscular",
    route="IM",
    analyte="parent",
    application_number="NDA 022264",
    review_date="2009-07-22",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "A one-compartment model with 1st order elimination best described the "
        "PK of paliperidone following i.m. administration of its palmitate "
        "ester... a dual input model was specified... a fraction of the dose to "
        "enter relatively quickly into the central compartment via a zero order "
        "process. After a certain lag-time the remaining fraction then entered "
        "the systemic circulation via a 1st order process"
    ),
    apparent_parameters=False,
    apparent_evidence=(
        "'The estimates of CL (which is equal to intravenous paliperidone CL "
        "reported previously) and KA substantiated that paliperidone palmitate "
        "offers complete bioavailability'. The symbols are printed as plain CL "
        "and V, and the sentence says why"
    ),
    population=(
        "pooled across 10 studies of paliperidone palmitate depot, paliperidone "
        "ER oral and paliperidone IR oral, adults"
    ),
    status="transcribed",
    software="NONMEM 6.1.0 double precision",
    parameters=(
        PopPKParameter("CL", 4.91, "L/h", None, "Table 13, model with IOV"),
        PopPKParameter("V", 346.0, "L", None, "Table 13, model with IOV"),
        PopPKParameter("KA", 0.00052, "1/h", None, "Table 13, printed as KA*10^3 = 0.52"),
        PopPKParameter("ALAG1 or D2", 375.0, "h", None, "Table 13, model with IOV"),
        PopPKParameter("F2", 0.197, None, None, "Table 13, model with IOV"),
        PopPKParameter("CL-CRCL power", 0.39, None, None, "Table 13"),
        PopPKParameter("V BMI power", 0.693, None, None, "Table 13"),
        PopPKParameter("V female shift", 0.83, None, None, "Table 13"),
    ),
    variability=(
        PopPKVariability("CL", 43.58898944, "percent-cv", None, False, "Table 13"),
        PopPKVariability("V", 75.29940239, "percent-cv", None, False, "Table 13"),
        PopPKVariability("KA", 56.83308895, "percent-cv", None, False, "Table 13"),
        PopPKVariability("F2", 0.360555128, "standard-deviation-logit", None, False, "Table 13"),
        PopPKVariability("CL", 26.0, "percent-cv", None, True, "narrative, IOV"),
        PopPKVariability("V", 14.0, "percent-cv", None, True, "narrative, IOV"),
    ),
    mamillary_slots=(("CL", "CL"), ("V1", "V")),
    notes=(
        "The only absolute set in this module. Table 13 is the FDA reviewer's "
        "own comparison of the sponsor's final model with and without "
        "inter-occasion variability; the without-IOV variant, which the review "
        "used for its simulations, gives CL 5.03 L/h and V 345 L. The review "
        "states that the slow first-order input 'governed the apparent "
        "half-life of this drug due to flip-flop kinetics', which is the "
        "reason the label's 25-49 day half-life cannot be compared against the "
        "disposition half-life this row implies."
    ),
)

_PALIPERIDONE_ERZOFRI = FDAPopPK(
    drug="paliperidone",
    formulation="ERZOFRI (LY03010) paliperidone palmitate intramuscular",
    route="IM",
    analyte="parent",
    application_number="NDA 216352",
    review_date="2024-07-26",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "one-compartment disposition with parallel zero-order (fraction F2, "
        "duration D2) and first-order (KA) absorption from the injection site, "
        "with separate ratios for the loading dose"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "'the estimated apparent clearance (CL) was 6.7 L/hr, the apparent "
        "volume of distribution (V/F) was 2840 L'. The same molecule in the "
        "same kind of formulation as NDA 022264, and this review says apparent "
        "where that one said complete bioavailability"
    ),
    population=(
        "adults with schizophrenia; typical subject is male, 51 years, BMI "
        "29.4 kg/m2"
    ),
    status="transcribed",
    parameters=(
        PopPKParameter("CL/F", 6.7, "L/h", None, "narrative of Table 39"),
        PopPKParameter("V/F", 2840.0, "L", None, "narrative of Table 39"),
        PopPKParameter("KA", 0.00165, "1/h", None, "narrative of Table 39"),
        PopPKParameter("F2", 0.179, None, None, "narrative of Table 39"),
        PopPKParameter("D2", 97.8, "h", None, "narrative of Table 39"),
        PopPKParameter("FINJS", 1.47, None, None, "narrative of Table 39"),
        PopPKParameter("R_F2", 0.684, None, None, "narrative of Table 39"),
        PopPKParameter("R_FINJS", 2.07, None, None, "narrative of Table 39"),
        PopPKParameter("R_D2", 1.05, None, None, "narrative of Table 39"),
    ),
    variability=(
        PopPKVariability("CL/F", 33.4, "percent-cv", None, False, "narrative, LY03010"),
        PopPKVariability("V/F", 50.6, "percent-cv", None, False, "narrative, LY03010"),
        PopPKVariability("KA", 63.0, "percent-cv", None, False, "narrative, LY03010"),
    ),
    mamillary_slots=(("CL", "CL/F"), ("V1", "V/F")),
    notes=(
        "Transcribed from the review's narrative restatement of Table 39, "
        "which is an image. This is the counter-example that killed the "
        "route-based rule for the apparent flag: same molecule, same depot "
        "class, opposite basis from NDA 022264, and a V/F eight times that "
        "review's V."
    ),
)

_PALIPERIDONE_INVEGA_ORAL = FDAPopPK(
    drug="paliperidone",
    formulation="INVEGA extended-release OROS oral tablet",
    route="PO",
    analyte="parent",
    application_number="NDA 021999",
    review_date="2006-09-12",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "not printed; the review states only that the sponsor submitted "
        "'Population Pharmacokinetic Analysis of ER OROS Paliperidone', report "
        "psdb-4647364, and that 'NONMEM data sets were not provided by the "
        "sponsor'"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the only oral clearance notation anywhere in the released text is "
        "Cl/F; no plain CL or V appears for an oral treatment"
    ),
    population=(
        "not stated for the population-PK analysis; the pharmacodynamic pooling "
        "used 1304 adults with schizophrenia or schizoaffective disorder from "
        "three phase 3 studies"
    ),
    status="searched-and-empty",
    extraction_barrier="redacted",
    notes=(
        "The single most important negative in this task, and the one that "
        "corrects an earlier record. The review is not missing and does not "
        "404: it is three scanned PDF parts, and it was fetched, rendered and "
        "read. Its Table 12, 'Paliperidone Oral Solution NONMEM Parameter "
        "Estimates - Study Alza-034', is inside a pull sheet reading '13 "
        "Page(s) Withheld / Trade Secret / Confidential / Withheld Track "
        "Number: Clin Pharm/Bio-8', which swallows Tables 11 through 21. "
        "Permanently unrecoverable; optical character recognition cannot read "
        "a page that was removed. The review does print, in clean legible "
        "text, the absolute oral bioavailability of the OROS formulation: "
        "27.7% with a 90% confidence interval of 23.4-32.8 from study P01-0007, "
        "n=20, against a 1 mg intravenous reference, which is the provenance of "
        "the label's 28%. That number cannot be applied to anything here, "
        "because this review prints no apparent parameters to convert."
    ),
)

_ARIPIPRAZOLE_MAINTENA = FDAPopPK(
    drug="aripiprazole",
    formulation="ABILIFY MAINTENA intramuscular depot",
    route="IM",
    analyte="parent",
    application_number="NDA 202971",
    review_date="2012-06-04",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "a linear 3-compartment model with sigmoid absorption for orally "
        "administered and first-order absorption for the IM Depot formulation"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "only a relative bioavailability F2 = 1.48 is estimated, the review's "
        "'Absolute bioavailability' checklist row is empty, and the successor "
        "application NDA 217006, which retains these same fixed effects, "
        "defines them in its own table as 'Apparent clearance' and 'Apparent "
        "central volume of distribution'"
    ),
    population=(
        "663 subjects, 6153 concentrations, 5 trials: 2 phase 1 in healthy "
        "volunteers and 3 in adults with schizophrenia"
    ),
    status="transcribed",
    software="NONMEM 6.1.0",
    n_subjects=663,
    n_observations=6153,
    parameters=(
        PopPKParameter("Ka", 0.540, "1/h", None, "Table 6, PDF p.48, fixed"),
        PopPKParameter("CL EM", 3.71, "L/h", 4.0, "Table 6, PDF p.48"),
        PopPKParameter("CL PM", 1.88, "L/h", 6.9, "Table 6, PDF p.48"),
        PopPKParameter("Vc", 93.4, "L", 8.8, "Table 6, PDF p.48"),
        PopPKParameter("Q1", 0.591, "L/h", None, "Table 6, PDF p.48, fixed"),
        PopPKParameter("Vp1", 118.0, "L", None, "Table 6, PDF p.48, fixed"),
        PopPKParameter("Q2", 28.8, "L/h", None, "Table 6, PDF p.48, fixed"),
        PopPKParameter("Vp2", 134.0, "L", None, "Table 6, PDF p.48, fixed"),
        PopPKParameter("R1", 9.33, "mg/h", None, "Table 6, PDF p.48, fixed"),
        PopPKParameter("IM Ka", 0.000904, "1/h", 5.3, "Table 6, PDF p.48"),
        PopPKParameter("F2", 1.48, None, 4.9, "Table 6, PDF p.48"),
        PopPKParameter("2D6 inhibitor on CL", -0.511, None, None, "Table 6, fixed"),
        PopPKParameter("3A4 inhibitor on CL", -0.237, None, None, "Table 6, fixed"),
    ),
    variability=(
        PopPKVariability("CL EM", 38.34, "percent-cv", None, False, "Table 6"),
        PopPKVariability("Vc", 124.50, "percent-cv", None, False, "Table 6"),
        PopPKVariability("Ka", 65.88, "percent-cv", None, False, "Table 6"),
        PopPKVariability("IM Ka", 55.59, "percent-cv", None, False, "Table 6"),
    ),
    mamillary_slots=(
        ("CL", "CL EM"),
        ("V1", "Vc"),
        ("Q2", "Q1"),
        ("V2", "Vp1"),
        ("Q3", "Q2"),
        ("V3", "Vp2"),
    ),
    notes=(
        "The slot mapping is deliberately confusing to read and was done by "
        "hand for that reason: this review's Q2 and Vp2 are the *second* "
        "peripheral compartment, so they fill the Q3/V3 slots, and its Q1/Vp1 "
        "fill Q2/V2. Pattern-matching the names would have swapped a 0.591 L/h "
        "inter-compartmental clearance with a 28.8 L/h one. Four of the six "
        "disposition parameters are fixed rather than estimated, carried in "
        "from an earlier oral analysis."
    ),
)

_ARIPIPRAZOLE_ARISTADA = FDAPopPK(
    drug="aripiprazole",
    formulation="ARISTADA aripiprazole lauroxil intramuscular",
    route="IM",
    analyte="parent-and-metabolite",
    application_number="NDA 207533",
    review_date="2015-10-01",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "depot compartments per route feeding central and peripheral "
        "compartments for aripiprazole and for dehydro-aripiprazole "
        "simultaneously, with a zero-order conversion of duration D1 and a "
        "1/D1 first-order absorption, and a lag time ALAG"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "'the bioavailability of aripiprazole following IM aripiprazole "
        "lauroxil relative to oral aripiprazole was estimated from the "
        "population PK model by fixing the bioavailability of oral "
        "aripiprazole to 1'. Every disposition symbol is printed with an "
        "explicit /F, and the metabolite's with /F/fm"
    ),
    population=(
        "616 adults with schizophrenia, 5 trials, intramuscular 441-882 mg "
        "plus oral aripiprazole 10 mg"
    ),
    status="transcribed",
    software="NONMEM 7.2",
    n_subjects=616,
    n_observations=10817,
    parameters=(
        PopPKParameter("Ka", 0.574, "1/h", 24.3, "Table 6, PDF p.120"),
        PopPKParameter("FPO", 1.00, None, None, "Table 6, fixed"),
        PopPKParameter("FIM FORM2", 0.581, None, 6.30, "Table 6, PDF p.120"),
        PopPKParameter("FIM FORM1", 1.57, None, 21.6, "Table 6, PDF p.120"),
        PopPKParameter("D1", 854.0, "h", 0.387, "Table 6, PDF p.120"),
        PopPKParameter("ALAG FORM2", 129.0, "h", 1.17, "Table 6, PDF p.120"),
        PopPKParameter("ALAG FORM1", 0.247, "h", 12.3, "Table 6, PDF p.120"),
        PopPKParameter("CL/F Non-PM", 2.02, "L/h", 4.02, "Table 6, PDF p.120"),
        PopPKParameter("CL/F PM", 0.767, "L/h", 22.7, "Table 6, PDF p.120"),
        PopPKParameter("VC/F", 268.0, "L", 0.537, "Table 6, PDF p.120"),
        PopPKParameter("Q/F", 0.423, "L/h", 7.37, "Table 6, PDF p.120"),
        PopPKParameter("VP/F", 2122.0, "L", 1.21, "Table 6, PDF p.120"),
        PopPKParameter("CLM/F/fm", 5.16, "L/h", 1.84, "Table 6, PDF p.120"),
        PopPKParameter("VCM/F/fm", 354.0, "L", 0.756, "Table 6, PDF p.120"),
        PopPKParameter("QM/F/fm", 0.868, "L/h", 74.6, "Table 6, PDF p.120"),
        PopPKParameter("VPM/F/fm", 351.0, "L", 1.86, "Table 6, PDF p.120"),
    ),
    variability=(
        PopPKVariability("CL/F Non-PM", 57.4, "percent-cv", None, False, "Table 6"),
        PopPKVariability("VC/F", 62.9, "percent-cv", None, False, "Table 6"),
        PopPKVariability("Q/F", 108.0, "percent-cv", None, False, "Table 6"),
        PopPKVariability("VP/F", 132.0, "percent-cv", None, False, "Table 6"),
        PopPKVariability("Ka", 209.0, "percent-cv", None, False, "Table 6"),
    ),
    mamillary_slots=(
        ("CL", "CL/F Non-PM"),
        ("V1", "VC/F"),
        ("Q2", "Q/F"),
        ("V2", "VP/F"),
    ),
    notes=(
        "The metabolite half of this fit cannot be separated out: CLM/F/fm is "
        "divided by the formation fraction as well as by F, so it is not a "
        "clearance of anything on its own. Hence one row with analyte "
        "parent-and-metabolite and a mamillary mapping covering only the "
        "parent."
    ),
)

_ARIPIPRAZOLE_ARISTADA_INITIO = FDAPopPK(
    drug="aripiprazole",
    formulation="ARISTADA INITIO aripiprazole lauroxil nanocrystal intramuscular",
    route="IM",
    analyte="parent",
    application_number="NDA 209830",
    review_date="2018-06-18",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "two-compartment disposition with double Weibull absorption (MDT1/GAM1 "
        "and MDT2/GAM2 with a mixing fraction FRAC) for the nanocrystal "
        "formulation, first-order for oral and zero-order of duration D with a "
        "lag for the original lauroxil formulation"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "printed as CL/F, VC/F, Q/F and VP/F throughout, with FPO ARI fixed to "
        "1.00 as in NDA 207533"
    ),
    population=(
        "249 adults with schizophrenia or schizoaffective disorder, 3 phase 1 "
        "studies"
    ),
    status="transcribed",
    n_subjects=249,
    n_observations=8944,
    parameters=(
        PopPKParameter("CL/F", 1.98, "L/h", 2.54, "Table 2, PDF p.25"),
        PopPKParameter("VC/F", 327.0, "L", 4.51, "Table 2, PDF p.25"),
        PopPKParameter("Q/F", 0.102, "L/h", 10.9, "Table 2, PDF p.25"),
        PopPKParameter("VP/F", 1720.0, "L", 13.9, "Table 2, PDF p.25"),
        PopPKParameter("MDT1", 596.0, "h", 3.29, "Table 2, PDF p.25"),
        PopPKParameter("GAM1", 2.20, None, None, "Table 2, PDF p.25"),
        PopPKParameter("MDT2", 76.7, "h", 4.47, "Table 2, PDF p.25"),
        PopPKParameter("GAM2", 2.09, None, None, "Table 2, PDF p.25"),
        PopPKParameter("FRAC", 2.02, None, None, "Table 2, PDF p.25"),
        PopPKParameter("Ka PO", 0.47, "1/h", 14.2, "Table 2, PDF p.25"),
        PopPKParameter("D AL", 934.0, "h", None, "Table 2, PDF p.25"),
        PopPKParameter("ALAG AL", 106.0, "h", None, "Table 2, PDF p.25"),
        PopPKParameter("FIM AL", 0.571, None, None, "Table 2, fixed"),
        PopPKParameter("FPO ARI", 1.00, None, None, "Table 2, fixed"),
        PopPKParameter("FIM AL-NCD", 1.12, None, None, "Table 2, PDF p.25"),
    ),
    variability=(
        PopPKVariability("CL/F", 84.5, "percent-cv", None, False, "Table 2"),
        PopPKVariability("VC/F", 52.0, "percent-cv", None, False, "Table 2"),
        PopPKVariability("Q/F", 353.0, "percent-cv", None, False, "Table 2"),
        PopPKVariability("VP/F", 99.3, "percent-cv", None, False, "Table 2"),
    ),
    mamillary_slots=(
        ("CL", "CL/F"),
        ("V1", "VC/F"),
        ("Q2", "Q/F"),
        ("V2", "VP/F"),
    ),
    notes=(
        "Every value in this row was read off a rendering of the page. The "
        "text layer of this PDF is mangled in two different font encodings and "
        "silently drops all digits, so pdftotext returns parameter names with "
        "no numbers beside them and no error. A transcription driven by the "
        "text layer would have produced an empty table and called it redacted. "
        "The 353% inter-individual variability on Q/F is as printed and says "
        "the peripheral compartment is barely identified."
    ),
)

_ARIPIPRAZOLE_ASIMTUFII = FDAPopPK(
    drug="aripiprazole",
    formulation="ABILIFY ASIMTUFII ready-to-use intramuscular depot",
    route="IM",
    analyte="parent",
    application_number="NDA 217006",
    review_date="2023-04-27",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "the NDA 202971 three-compartment model extended with a separate "
        "central volume, absorption rate and lag for the ready-to-use "
        "formulation, retaining the earlier fixed effects"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the table's own Definition column reads 'Apparent clearance...' and "
        "'Apparent central volume of distribution...', while the symbols are "
        "printed bare as CL and VC"
    ),
    population="1191 adults, 17113 observations, 10 studies",
    status="transcribed",
    n_subjects=1191,
    n_observations=17113,
    parameters=(
        PopPKParameter("CL_EM", 3.71, "L/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("CL_PM", 1.88, "L/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("VC", 93.4, "L", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("VCRTU", 2035.0, "L", 0.02, "Tables 42/43, PDF p.133"),
        PopPKParameter("Q1", 0.591, "L/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("VP1", 118.0, "L", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("Q2", 28.8, "L/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("VP2", 134.0, "L", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("R1", 9.33, "mg/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("KAPO", 0.54, "1/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("KAGLU", 0.000904, "1/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("KADEL", 0.000776, "1/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("KARTU", 0.00127, "1/h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("DUR0", 162.0, "h", 6.5, "Tables 42/43, PDF p.133"),
        PopPKParameter("FRAC1", 3.82, None, None, "Tables 42/43, PDF p.133"),
        PopPKParameter("ALAG1", 419.0, "h", None, "Tables 42/43, PDF p.133"),
        PopPKParameter("FREL_MAINTENA", 1.48, None, None, "Tables 42/43, PDF p.133"),
        PopPKParameter("FREL_RTU", 1.58, None, 1.9, "Tables 42/43, PDF p.133"),
    ),
    variability=(
        PopPKVariability("CL", 0.155, "variance", None, False, "Table 43, CV 39.3%"),
        PopPKVariability("VC", 1.87, "variance", None, False, "Table 43, CV 136.7%"),
        PopPKVariability("KAPO", 0.434, "variance", None, False, "Table 43, CV 65.9%"),
        PopPKVariability("KAGLU", 0.359, "variance", None, False, "Table 43, CV 59.9%"),
        PopPKVariability("KARTU", 0.307, "variance", None, False, "Table 43, CV 55.4%"),
    ),
    mamillary_slots=(
        ("CL", "CL_EM"),
        ("V1", "VC"),
        ("Q2", "Q1"),
        ("V2", "VP1"),
        ("Q3", "Q2"),
        ("V3", "VP2"),
    ),
    notes=(
        "The disposition fixed effects are numerically identical to NDA "
        "202971's, because they were carried over rather than re-estimated, so "
        "this row's implied half-life is the same number and is not an "
        "independent replication of it. Kept as a separate row because it is a "
        "separate application with a separate review date, and because it is "
        "the row whose Definition column settled the apparent question for the "
        "whole aripiprazole family. This review prints its variability as "
        "variances with the %CV alongside, which is the reason "
        "PopPKVariability carries a scale at all."
    ),
)

_ARIPIPRAZOLE_ABILIFY_ORAL = FDAPopPK(
    drug="aripiprazole",
    formulation="ABILIFY oral tablet, supplement s018",
    route="PO",
    analyte="parent",
    application_number="NDA 021436",
    review_date="2007-10-19",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model="none; the review is non-compartmental throughout",
    apparent_parameters=True,
    apparent_evidence=(
        "not applicable, no parameters exist; the review's 'Absolute "
        "bioavailability' checklist row is empty"
    ),
    population="not applicable",
    status="searched-and-empty",
    extraction_barrier="not-fitted",
    notes=(
        "No population-PK analysis exists. The review records that the FDA "
        "asked for one and that the sponsor's rationale for not performing it "
        "'is not valid'. This is the reason the whole family's parameters come "
        "from depot applications: the oral innovator package never fitted a "
        "population model."
    ),
)

_ARIPIPRAZOLE_OPIPZA = FDAPopPK(
    drug="aripiprazole",
    formulation="OPIPZA oral soluble film",
    route="PO",
    analyte="parent",
    application_number="NDA 216655",
    review_date="2024-07-19",
    review_document="Multi-Disciplinary Review",
    structural_model="none; a bioequivalence package only",
    apparent_parameters=True,
    apparent_evidence="not applicable, no parameters exist",
    population="not applicable",
    status="searched-and-empty",
    extraction_barrier="not-fitted",
    notes=(
        "Zero occurrences of 'compartment', 'NONMEM', 'population PK', 'CL/F' "
        "or '%RSE' in the entire document."
    ),
)

_OLANZAPINE_ZYPREXA_ADOLESCENT = FDAPopPK(
    drug="olanzapine",
    formulation="ZYPREXA oral tablet, supplements s040 and s041",
    route="PO",
    analyte="parent",
    application_number="NDA 020592",
    review_date="2007-03-27",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "a one compartment pharmacokinetic model with parameters such as "
        "absorption rate constant (Ka), oral clearance (CL/F), and oral volume "
        "of distribution (V/F)... The available data in adolescent patients did "
        "not allow reliable estimation of Ka"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "printed as 'Oral clearance, CL/F L/hr' and 'Oral Volume of "
        "Distribution, V/F L' in the table itself"
    ),
    population=(
        "adolescents aged 13 to 17 with schizophrenia or bipolar I disorder, "
        "pooled from studies HGCS, HGCR, HGGC and HGMF; HGMF alone contributed "
        "105 patients"
    ),
    status="transcribed",
    software="NONMEM",
    parameters=(
        PopPKParameter("Ka", 0.543, "1/h", None, "FDA base model table, fixed"),
        PopPKParameter("CL/F", 16.3, "L/h", 4.61, "FDA base model table"),
        PopPKParameter("V/F", 879.0, "L", 17.1, "FDA base model table"),
    ),
    variability=(
        PopPKVariability("CL/F", 43.7, "percent-cv", None, False, "FDA base model table"),
        PopPKVariability("V/F", 62.2, "percent-cv", None, False, "FDA base model table"),
    ),
    mamillary_slots=(("CL", "CL/F"), ("V1", "V/F")),
    notes=(
        "This is the FDA reviewer's own base model, not the sponsor's, and the "
        "review notes 'FDA Values all agree with sponsor for the Base model'. "
        "The base model is transcribed rather than the final covariate model "
        "on purpose: the final model prints CL/F 13.66 L/h and V 899 L with "
        "%SEE of 456 and 993 on those two estimates and 1291 on the "
        "variability, which is not an identified fit. Transcribing the final "
        "model because it is labelled final would have carried a number with "
        "an eleven-fold standard error into the registry as though it were the "
        "better one. Adolescents only, which is the population caveat on any "
        "comparison against an adult label."
    ),
)

_OLANZAPINE_LYBALVI = FDAPopPK(
    drug="olanzapine",
    formulation="LYBALVI olanzapine/samidorphan bilayer oral tablet",
    route="PO",
    analyte="parent",
    application_number="NDA 213378",
    review_date="2020-11-13",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "two-compartment disposition with first-order absorption and an "
        "absorption lag time, allometric weight exponents fixed at 0.75 on "
        "clearance and 1.0 on central volume"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "printed as CL/F, Vc/F, Vp/F and Q/F in the table, with the figure "
        "legend spelling it out: 'Vc/F and Vp/F: apparent central and "
        "peripheral volumes of distribution respectively; CL/F: apparent "
        "clearance'"
    ),
    population=(
        "healthy adults and adults with schizophrenia, pooled across the "
        "ALK3831 programme including dedicated hepatic-impairment, "
        "renal-impairment and rifampin-induction studies"
    ),
    status="transcribed",
    software="NONMEM",
    parameters=(
        PopPKParameter("CL/F", 15.5, "L/h", 2.85, "Table 6, PDF p.285"),
        PopPKParameter("Vc/F", 656.0, "L", 2.23, "Table 6, PDF p.285"),
        PopPKParameter("Vp/F", 225.0, "L", 9.42, "Table 6, PDF p.285"),
        PopPKParameter("Q/F", 6.15, "L/h", 18.4, "Table 6, PDF p.285"),
        PopPKParameter("Ka", 0.861, "1/h", 5.70, "Table 6, PDF p.285"),
        PopPKParameter("ALAG", 0.782, "h", None, "Table 6, PDF p.285, fixed"),
        PopPKParameter(
            "WT on CL/F", 0.75, None, None,
            "Table 6, PDF p.285, printed as 'WT ON CL/F 0.75 Fixed' with the "
            "footnote 'Fixed at allometric exponent'",
        ),
        PopPKParameter(
            "WT on V/F", 1.0, None, None,
            "Table 6, PDF p.285, printed as 'WT ON V/F 1.0 Fixed' with the "
            "footnote 'Fixed at allometric exponent'",
        ),
        PopPKParameter("smoking on CL/F", 1.30, None, 3.64, "Table 6, PDF p.285"),
        PopPKParameter("rifampin on CL/F", 1.80, None, 4.45, "Table 6, PDF p.285"),
        PopPKParameter("female on CL/F", 0.862, None, 3.56, "Table 6, PDF p.285"),
        PopPKParameter("black race on CL/F", 1.10, None, 3.22, "Table 6, PDF p.285"),
        PopPKParameter("age on Vc/F", 0.356, None, 11.8, "Table 6, PDF p.285"),
        PopPKParameter("food on F", 0.943, None, 2.26, "Table 6, PDF p.285"),
    ),
    variability=(
        PopPKVariability("CL/F", 43.2, "percent-cv", None, False, "Table 6, from variance 0.171"),
        PopPKVariability("Vc/F", 35.6, "percent-cv", None, False, "Table 6, from variance 0.127"),
        PopPKVariability("Ka", 48.2, "percent-cv", None, False, "Table 6, from variance 0.209"),
        PopPKVariability("ALAG", 61.3, "percent-cv", None, False, "Table 6, from variance 0.319"),
        PopPKVariability("Vp/F", 50.0, "percent-cv", None, False, "Table 6, variance fixed at 0.223"),
        PopPKVariability("Q/F", 50.0, "percent-cv", None, False, "Table 6, variance fixed at 0.223"),
        PopPKVariability("Ka", 93.7, "percent-cv", None, True, "Table 6, inter-occasion"),
    ),
    mamillary_slots=(
        ("CL", "CL/F"),
        ("V1", "Vc/F"),
        ("Q2", "Q/F"),
        ("V2", "Vp/F"),
    ),
    absorption_slots=(
        ("ka", "Ka"),
        ("lag", "ALAG"),
    ),
    allometric_slots=(
        ("cl_weight_exponent", "WT on CL/F"),
        ("volume_weight_exponent", "WT on V/F"),
    ),
    reference_weight_kg=70.0,
    reference_subject=(
        "read off the covariate forest plots, Figures 15 and 16, PDF pp.286-287: "
        "'the reference subject set as a healthy 36 years old, 70 kg, non-black "
        "male non-smoker, with normal hepatic and renal function receiving a once "
        "daily oral doses of ALKS 3831 10/10 (10 mg olanzapine/10 mg samidorphan) "
        "in a fasted condition'. The same two figures print the reference "
        "exposures that subject reaches: Cmax,ss = 31.7 ng/mL and AUCtau = 635 "
        "ng.h/mL. Table 6 itself prints only 'WT ON CL/F 0.75 Fixed' and 'WT ON "
        "V/F 1.0 Fixed' and never names the normalisation weight, so the forest "
        "plots are the only place in the review that pins it."
    ),
    notes=(
        "The best olanzapine row in this registry and the reason the "
        "adolescent NDA 020592 row is not the only one: this is an adult oral "
        "fit against an adult oral label, two-compartment rather than one, "
        "with %RSE under 20 on every estimated disposition parameter. Found "
        "only because the caption sits at the foot of one page and the raster "
        "table at the head of the next, so a caption-anchored page extraction "
        "returns an empty table and looks like a redaction. The same review "
        "fits samidorphan simultaneously (CL/F 35.4 L/h, Vc/F 297 L, Vp/F 124 "
        "L, Q/F 12.1 L/h); samidorphan is a mu-opioid antagonist with no D2 "
        "activity and is not registered here, because this project has no use "
        "for a parameter set it cannot connect to a receptor it models."
    ),
)

_OLANZAPINE_RELPREVV = FDAPopPK(
    drug="olanzapine",
    formulation="ZYPREXA RELPREVV olanzapine pamoate intramuscular depot",
    route="IM",
    analyte="parent",
    application_number="NDA 022173",
    review_date="2009-08-17",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "one-compartment disposition with a mixed depot input: a fraction "
        "Frac absorbed by a first-order rate K13 and the remainder by a "
        "zero-order process of duration D3, plus first-order oral absorption "
        "K23 with an estimated oral bioavailability F2"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "printed as CL/F and V/F in the table's parameter column. F2 is an "
        "oral bioavailability estimated at 0.748 relative to the depot, not "
        "an absolute one, so the /F is not removable"
    ),
    population="patients receiving intramuscular olanzapine pamoate and oral olanzapine",
    status="transcribed",
    software="NONMEM",
    parameters=(
        PopPKParameter("CL/F", 12.8, "L/h", 3.37, "Table 4, PDF p.16, non-smokers"),
        PopPKParameter("V/F", 634.0, "L", 7.90, "Table 4, PDF p.16, non-smokers"),
        PopPKParameter("Frac", 0.939, None, 0.433, "Table 4, PDF p.16"),
        PopPKParameter("K13", 0.000529, "1/h", 4.20, "Table 4, PDF p.16, female"),
        PopPKParameter("K23", 0.758, "1/h", 8.51, "Table 4, PDF p.16"),
        PopPKParameter("D3", 65.9, "h", 1.35, "Table 4, PDF p.16"),
        PopPKParameter("F2", 0.748, None, 3.18, "Table 4, PDF p.16"),
        PopPKParameter("smoking on CL/F", 0.468, None, 13.5, "Table 4, PDF p.16"),
        PopPKParameter("smoking on V/F", 0.537, None, 27.0, "Table 4, PDF p.16"),
        PopPKParameter("male on K13", 0.331, None, 23.1, "Table 4, PDF p.16"),
    ),
    variability=(
        PopPKVariability("CL/F", 39.5, "percent-cv", None, False, "Table 4, %SEE 12.4"),
        PopPKVariability("F2", 41.6, "percent-cv", None, False, "Table 4, %SEE 16.9"),
        PopPKVariability("CL/F", 25.4, "percent-cv", None, True, "Table 4, %SEE 17.0"),
    ),
    mamillary_slots=(("CL", "CL/F"), ("V1", "V/F")),
    notes=(
        "The covariate effects are additive shifts, not multipliers: the "
        "review's own footnotes give CL for smokers as 12.8 x (1 + 0.468) = "
        "18.8 L/h and V for smokers as 634 x (1 + 0.537) = 974.5 L. "
        "Transcribing 0.468 as a multiplicative ratio, which is how most other "
        "reviews in this set write covariates, would have halved the smoker "
        "clearance instead of raising it by half. The non-smoker values are "
        "the ones registered. No variability was estimated on V/F, which is "
        "why the row has fewer variability entries than parameters."
    ),
)

_PALIPERIDONE_TRINZA = FDAPopPK(
    drug="paliperidone",
    formulation="INVEGA TRINZA paliperidone palmitate 3-month intramuscular",
    route="IM",
    analyte="parent",
    application_number="NDA 207946",
    review_date="2015-05-05",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model=(
        "one-compartment disposition with saturable, dose-dependent absorption: "
        "the depot input rate is an Emax function of the amount remaining at "
        "the injection site (ka_max, k_amt50, Hill coefficient gamma), with a "
        "second such process for the fraction F3 present as unhydrolysed "
        "palmitate"
    ),
    apparent_parameters=True,
    apparent_evidence=(
        "the symbols are printed bare as CL and V, and the abbreviation legend "
        "defines them as 'CL: apparent clearance' and 'V: apparent volume of "
        "distribution'. The clearest case in the registry of notation and "
        "meaning disagreeing, and the reason the schema requires a quoted "
        "sentence rather than inferring the flag from the symbol"
    ),
    population="adults with schizophrenia receiving 3-monthly intramuscular paliperidone palmitate",
    status="transcribed",
    software="NONMEM",
    parameters=(
        PopPKParameter("CL", 3.84, "L/h", 2.16, "Table 12, PDF p.52"),
        PopPKParameter("V", 1960.0, "L", None, "Table 12, PDF p.52, fixed"),
        PopPKParameter("ka1_max", 90.4, "ug/h", 6.96, "Table 12, PDF p.52"),
        PopPKParameter("k_amt1_50", 120.0, "mg", 3.83, "Table 12, PDF p.52"),
        PopPKParameter("gamma", 1.44, None, 1.65, "Table 12, PDF p.52"),
        PopPKParameter("V palmitate", 156.0, "L", None, "Table 12, PDF p.52, fixed"),
        PopPKParameter("F3", 0.209, None, None, "Table 12, PDF p.52, fixed"),
        PopPKParameter("ka3_max", 164.0, "ug/h", 4.65, "Table 12, PDF p.52"),
        PopPKParameter("k_amt3_50", 21.4, "mg", 9.52, "Table 12, PDF p.52"),
        PopPKParameter("CRCL on CL", 0.316, None, 21.5, "Table 12, PDF p.52"),
        PopPKParameter("BMI on V", 1.18, None, None, "Table 12, PDF p.52, fixed"),
    ),
    variability=(
        PopPKVariability("CL", 35.7, "percent-cv", 31.6, False, "Table 12, omega 0.357"),
        PopPKVariability("V", 62.8, "percent-cv", 35.6, False, "Table 12, omega fixed 0.628"),
        PopPKVariability("ka1_max", 82.7, "percent-cv", 31.3, False, "Table 12, omega 0.827"),
        PopPKVariability("F3", 108.0, "percent-cv", 29.5, False, "Table 12, omega fixed 1.08"),
    ),
    mamillary_slots=(("CL", "CL"), ("V1", "V")),
    notes=(
        "The central volume is fixed, not estimated, so the half-life this row "
        "implies is a fixed constant divided by a fitted clearance and is not "
        "an independent estimate of a half-life at all. Registered and compared "
        "anyway, with that stated, because dropping it would remove the "
        "clearest instance of the pattern the comparison exists to expose."
    ),
)

_QUETIAPINE_SEROQUEL_XR = FDAPopPK(
    drug="quetiapine",
    formulation="SEROQUEL XR extended-release oral tablet",
    route="PO",
    analyte="parent",
    application_number="NDA 022047",
    review_date="2007-05-10",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model="none; 4420 lines of non-compartmental analysis",
    apparent_parameters=True,
    apparent_evidence="not applicable, no parameters exist",
    population="not applicable",
    status="searched-and-empty",
    extraction_barrier="not-fitted",
    notes=(
        "Quetiapine now has two independent reasons to be absent from this "
        "project's chain. P21.2 established that no plasma EC50 exists for it "
        "because the occupancy time course does not admit one. P21.3 adds that "
        "the extended-release innovator application contains no population-PK "
        "model either. Neither is a failure of searching."
    ),
)

_CLOZAPINE_VERSACLOZ = FDAPopPK(
    drug="clozapine",
    formulation="VERSACLOZ oral suspension",
    route="PO",
    analyte="parent",
    application_number="NDA 203479",
    review_date="2012-09-04",
    review_document="Clinical Pharmacology and Biopharmaceutics Review(s)",
    structural_model="none; a bioequivalence and food-effect package",
    apparent_parameters=True,
    apparent_evidence="not applicable, no parameters exist",
    population="not applicable",
    status="searched-and-empty",
    extraction_barrier="not-fitted",
    notes=(
        "700 lines, entirely non-compartmental. The label that came out of "
        "this application likewise prints no clearance and no volume of "
        "distribution for clozapine at all, only half-lives."
    ),
)

_AMISULPRIDE_BARHEMSYS_SPONSOR = FDAPopPK(
    drug="amisulpride",
    formulation="BARHEMSYS intravenous injection, Applicant's model",
    route="IV",
    analyte="parent",
    application_number="NDA 209510",
    review_date="2020-02-25",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "three-compartment disposition with allometric weight scaling fixed at "
        "0.75 on the clearances and 1.0 on the volumes, and a standardised "
        "renal function term on clearance"
    ),
    apparent_parameters=False,
    apparent_evidence=(
        "the product is an intravenous infusion, every symbol is printed "
        "without /F, and the approved label restates this model's clearance as "
        "'the plasma clearance of amisulpride is 20.6 L/h in surgical patients "
        "and 24.1 L/h in healthy subjects' with no apparent qualifier"
    ),
    population=(
        "143 subjects, 1273 concentrations, 4 studies: 39 healthy subjects in a "
        "thorough-QT crossover and 104 surgical patients under general "
        "anaesthesia in three phase 3 trials, intravenous 5-40 mg"
    ),
    status="transcribed",
    n_subjects=143,
    n_observations=1273,
    parameters=(
        PopPKParameter("CL healthy", 24.1, "L/h", 3.0, "Table 92, PDF p.190"),
        PopPKParameter("CL surgical", 20.6, "L/h", 4.0, "Table 92, PDF p.190"),
        PopPKParameter("V1", 23.9, "L", 4.0, "Table 92, PDF p.190"),
        PopPKParameter("Q", 73.1, "L/h", 3.0, "Table 92, PDF p.190"),
        PopPKParameter("V2", 69.3, "L", 3.0, "Table 92, PDF p.190"),
        PopPKParameter("Q2", 5.6, "L/h", 13.0, "Table 92, PDF p.190"),
        PopPKParameter("V3", 84.9, "L", 5.0, "Table 92, PDF p.190"),
    ),
    variability=(
        PopPKVariability("CL healthy", 17.0, "percent-cv", None, False, "Table 92, %RSE 12"),
        PopPKVariability("CL surgical", 32.0, "percent-cv", None, False, "Table 92, %RSE 9"),
        PopPKVariability("V1", 37.0, "percent-cv", None, False, "Table 92, %RSE 9"),
        PopPKVariability("Q healthy", 10.0, "percent-cv", None, False, "Table 92, %RSE 38"),
        PopPKVariability("Q surgical", 75.0, "percent-cv", None, False, "Table 92, %RSE 11"),
        PopPKVariability("V2", 16.0, "percent-cv", None, False, "Table 92, %RSE 14"),
        PopPKVariability("Q2", 87.0, "percent-cv", None, False, "Table 92, %RSE 14"),
    ),
    mamillary_slots=(
        ("CL", "CL healthy"),
        ("V1", "V1"),
        ("Q2", "Q"),
        ("V2", "V2"),
        ("Q3", "Q2"),
        ("V3", "V3"),
    ),
    notes=(
        "The Applicant's model, which is the one the label quotes. Registered "
        "separately from the FDA reviewer's refit in the same document because "
        "they are different fits with different estimates: 24.1 against 16 for "
        "clearance and 84.9 against 991 for the deep peripheral volume. Only "
        "this one prints units. The slot mapping is by hand again: this "
        "review's Q is the shallow inter-compartmental clearance and its Q2 is "
        "the deep one, the opposite of the naming in NDA 202971. The healthy "
        "clearance is the one mapped, because the label's half-life is a "
        "healthy-subject number; the surgical value differs only in clearance "
        "and its terminal half-life is 14.21 hours against 13.58."
    ),
)

_AMISULPRIDE_BARHEMSYS_FDA = FDAPopPK(
    drug="amisulpride",
    formulation="BARHEMSYS intravenous injection, FDA reviewer's refit",
    route="IV",
    analyte="parent",
    application_number="NDA 209510",
    review_date="2020-02-25",
    review_document="Multi-Disciplinary Review",
    structural_model=(
        "three-compartment disposition; the reviewer's own refit, distinct "
        "from the sponsor's model"
    ),
    apparent_parameters=False,
    apparent_evidence=(
        "the product is intravenous, so F is 1 by construction and the "
        "clearance is a plasma clearance; the label states 'Population "
        "pharmacokinetic analysis estimated that the plasma clearance of "
        "amisulpride is 20.6 L/h in surgical patients and 24.1 L/h in healthy "
        "subjects', with no apparent qualifier"
    ),
    population=(
        "surgical patients and healthy subjects; amisulpride is not approved "
        "in the United States as an antipsychotic, only intravenously for "
        "postoperative nausea and vomiting at 5-10 mg"
    ),
    status="transcribed",
    parameters=(
        PopPKParameter("CL", 16.0, None, None, "Table 97, FDA reviewer's final model"),
        PopPKParameter("V1", 30.9, None, None, "Table 97, FDA reviewer's final model"),
        PopPKParameter("Q2", 73.0, None, None, "Table 97, FDA reviewer's final model"),
        PopPKParameter("V2", 85.9, None, None, "Table 97, FDA reviewer's final model"),
        PopPKParameter("Q3", 15.9, None, None, "Table 97, FDA reviewer's final model"),
        PopPKParameter("V3", 991.0, None, None, "Table 97, FDA reviewer's final model"),
    ),
    mamillary_slots=(
        ("CL", "CL"),
        ("V1", "V1"),
        ("Q2", "Q2"),
        ("V2", "V2"),
        ("Q3", "Q3"),
        ("V3", "V3"),
    ),
    notes=(
        "Table 97 has two columns, parameter and value, and no unit column "
        "anywhere. This is the row that finding 3 is about, and the reason "
        "require_units() refuses rather than inferring: the temptation is to "
        "copy the units from Table 92 in the same document, which prints L/h "
        "and L for the same six symbols, and that would be an inference, not a "
        "transcription. It would also be a doubtful one, because the two fits "
        "disagree substantially where they can be compared: CL 16 against "
        "24.1, and a deep peripheral volume of 991 against 84.9. Excluded "
        "from half_life_comparisons by its own missing units, which is the "
        "correct outcome and not a gap. Had the units been assumed, this row "
        "would have implied a terminal half-life of 88.7 hours against a label "
        "value of 4 to 5."
    ),
)

FDA_POPPK_RECORDS = (
    _RISPERIDONE_PERSERIS_ORAL,
    _RISPERIDONE_PERSERIS_DEPOT,
    _RISPERIDONE_RYKINDO,
    _RISPERIDONE_UZEDY_PARENT,
    _RISPERIDONE_UZEDY_METABOLITE,
    _RISPERIDONE_RISVAN,
    _PALIPERIDONE_SUSTENNA,
    _PALIPERIDONE_TRINZA,
    _PALIPERIDONE_ERZOFRI,
    _PALIPERIDONE_INVEGA_ORAL,
    _ARIPIPRAZOLE_MAINTENA,
    _ARIPIPRAZOLE_ARISTADA,
    _ARIPIPRAZOLE_ARISTADA_INITIO,
    _ARIPIPRAZOLE_ASIMTUFII,
    _ARIPIPRAZOLE_ABILIFY_ORAL,
    _ARIPIPRAZOLE_OPIPZA,
    _OLANZAPINE_ZYPREXA_ADOLESCENT,
    _OLANZAPINE_LYBALVI,
    _OLANZAPINE_RELPREVV,
    _QUETIAPINE_SEROQUEL_XR,
    _CLOZAPINE_VERSACLOZ,
    _AMISULPRIDE_BARHEMSYS_SPONSOR,
    _AMISULPRIDE_BARHEMSYS_FDA,
)


def _index(records):
    index = {}
    for item in records:
        if item.key in index:
            raise ValueError("duplicate record key %r" % (item.key,))
        index[item.key] = item
    return MappingProxyType(index)


_BY_APPLICATION = _index(FDA_POPPK_RECORDS)


def registered_drugs():
    """Drugs with at least one row, transcribed or searched-and-empty."""
    return tuple(sorted({record.drug for record in FDA_POPPK_RECORDS}))


def records_for(drug):
    """Every row for a drug, in registration order.

    Raises rather than returning an empty tuple, because an empty tuple would
    be indistinguishable from "searched and found nothing", which is a finding
    this module records explicitly and never by omission.
    """
    matches = tuple(r for r in FDA_POPPK_RECORDS if r.drug == drug)
    if not matches:
        raise ValueError(
            "no FDA population-PK row for %r; registered drugs are %s"
            % (drug, ", ".join(registered_drugs()))
        )
    return matches


def transcribed_records(drug=None):
    records = records_for(drug) if drug is not None else FDA_POPPK_RECORDS
    return tuple(r for r in records if r.status == "transcribed")


def searched_and_empty_records(drug=None):
    records = records_for(drug) if drug is not None else FDA_POPPK_RECORDS
    return tuple(r for r in records if r.status == "searched-and-empty")


def record(application_number, analyte=None, formulation=None, review_date=None):
    """One row, narrowed until exactly one matches.

    Raises on ambiguity rather than returning the first hit. An application
    number alone is not a key here: four of the applications in this registry
    contain more than one fitted model, and silently returning one of them
    would let a caller compare a depot sub-model against an oral label without
    ever seeing that it had a choice.
    """
    candidates = [
        r for r in FDA_POPPK_RECORDS if r.application_number == application_number
    ]
    if analyte is not None:
        candidates = [r for r in candidates if r.analyte == analyte]
    if formulation is not None:
        candidates = [r for r in candidates if r.formulation == formulation]
    if review_date is not None:
        candidates = [r for r in candidates if r.review_date == review_date]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "no row for %s (analyte=%r, formulation=%r, review_date=%r); "
            "registered rows are %s"
            % (
                application_number,
                analyte,
                formulation,
                review_date,
                ", ".join(
                    "%s/%s/%s"
                    % (r.application_number, r.analyte, r.formulation)
                    for r in FDA_POPPK_RECORDS
                ),
            )
        )
    raise ValueError(
        "%s is ambiguous; %d rows match: %s"
        % (
            application_number,
            len(candidates),
            "; ".join("%r" % (r.key,) for r in candidates),
        )
    )


# ---------------------------------------------------------------------------
# The comparison. This is the part of P21.3 that is a result rather than a
# transcription.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelHalfLife:
    """A half-life as an approved US label prints it.

    Separate from the review rows on purpose. The label is written from
    non-compartmental analysis of the same clinical programme, by a different
    part of the same process, and it is the only externally-chosen number
    available for these parameter sets. It is a weak external standard and the
    ledger row says so.
    """

    drug: str
    entity: str
    low_h: float
    high_h: float
    quote: str
    source: str
    innovator: bool

    def contains(self, hours):
        return self.low_h <= hours <= self.high_h

    def signed_error_h(self, hours):
        """Distance outside the printed interval, signed. Zero if inside."""
        if hours < self.low_h:
            return hours - self.low_h
        if hours > self.high_h:
            return hours - self.high_h
        return 0.0


LABEL_HALF_LIVES = (
    LabelHalfLife(
        drug="risperidone",
        entity="risperidone, extensive metabolisers",
        low_h=3.0,
        high_h=3.0,
        quote=(
            "The apparent half-life of risperidone was 3 hours (CV=30%) in "
            "extensive metabolizers and 20 hours (CV=40%) in poor metabolizers."
        ),
        source="ANDA 201003, SPL effective 2026-06-25",
        innovator=False,
    ),
    LabelHalfLife(
        drug="risperidone",
        entity="9-hydroxyrisperidone, extensive metabolisers",
        low_h=21.0,
        high_h=21.0,
        quote=(
            "The apparent half-life of 9-hydroxyrisperidone was about 21 hours "
            "(CV=20%) in extensive metabolizers and 30 hours (CV=25%) in poor "
            "metabolizers."
        ),
        source="ANDA 201003, SPL effective 2026-06-25",
        innovator=False,
    ),
    LabelHalfLife(
        drug="olanzapine",
        entity="olanzapine",
        low_h=21.0,
        high_h=54.0,
        quote=(
            "Its half-life ranges from 21 to 54 hours (5th to 95th percentile; "
            "mean of 30 hr)"
        ),
        source="ANDA 208146, SPL effective 2023-07-10",
        innovator=False,
    ),
    LabelHalfLife(
        drug="aripiprazole",
        entity="aripiprazole, non poor metabolisers",
        low_h=75.0,
        high_h=75.0,
        quote=(
            "The mean elimination half-lives are about 75 hours and 94 hours "
            "for aripiprazole and dehydro-aripiprazole, respectively."
        ),
        source="ANDA 202101, SPL effective 2026-03-05",
        innovator=False,
    ),
    LabelHalfLife(
        drug="amisulpride",
        entity="amisulpride, intravenous",
        low_h=4.0,
        high_h=5.0,
        quote="The terminal elimination half-life of amisulpride is about 4 to 5 hours.",
        source="NDA 209510, BARHEMSYS SPL",
        innovator=True,
    ),
    LabelHalfLife(
        drug="paliperidone",
        entity="paliperidone, extended-release oral",
        low_h=23.0,
        high_h=23.0,
        quote=(
            "The terminal elimination half-life of paliperidone is "
            "approximately 23 hours."
        ),
        source="ANDA 204707, SPL effective 2026-01-19",
        innovator=False,
    ),
)


def label_half_life(drug, entity):
    for item in LABEL_HALF_LIVES:
        if item.drug == drug and item.entity == entity:
            return item
    raise ValueError("no label half-life for %s / %s" % (drug, entity))


@dataclass(frozen=True)
class HalfLifeComparison:
    """One population-PK-implied half-life against one label-printed one."""

    application_number: str
    analyte: str
    formulation: str
    predicted_h: float
    label: LabelHalfLife
    terminal_auc_fraction: float
    n_compartments: int
    agrees_within_twofold: bool
    comment: str

    @property
    def fold_error(self):
        """Predicted over the nearest edge of the label interval, >= 1."""
        if self.label.contains(self.predicted_h):
            return 1.0
        if self.predicted_h > self.label.high_h:
            return self.predicted_h / self.label.high_h
        return self.label.low_h / self.predicted_h


#: Which transcribed row is compared against which label statement, and why
#: that pairing is the honest one. Paired by hand: pattern-matching on drug
#: name would have compared the depot rows against an oral label without
#: noticing that the label's number is absorption-rate-limited.
_HALF_LIFE_PAIRINGS = (
    (
        "NDA 213586",
        "parent",
        None,
        "risperidone",
        "risperidone, extensive metabolisers",
        "Closest thing to a clean test in this set. Both numbers are "
        "disposition half-lives of the parent, the analysis pooled oral and "
        "subcutaneous arms, and the model is one-compartment so the whole area "
        "is in the terminal phase.",
    ),
    (
        "NDA 213586",
        "metabolite",
        None,
        "risperidone",
        "9-hydroxyrisperidone, extensive metabolisers",
        "Under-predicts by roughly half. The metabolite in this analysis is "
        "formed from a slow-release subcutaneous depot, so its observed decline "
        "is formation-rate-limited and the fitted CLMO/VMO is not what the "
        "label's oral 21 hours measures.",
    ),
    (
        "NDA 020592",
        "parent",
        None,
        "olanzapine",
        "olanzapine",
        "Inside the label's 21-54 hour interval, but the parameters are fitted "
        "on adolescents aged 13-17 and the label interval is adult. The "
        "agreement is real and the population mismatch is real with it.",
    ),
    (
        "NDA 202971",
        "parent",
        None,
        "aripiprazole",
        "aripiprazole, non poor metabolisers",
        "Over-predicts by a factor of about two. Four of the six disposition "
        "parameters were fixed rather than estimated, so this is a test of "
        "carried-over constants as much as of a fit.",
    ),
    (
        "NDA 217006",
        "parent",
        None,
        "aripiprazole",
        "aripiprazole, non poor metabolisers",
        "Numerically identical to NDA 202971 because the fixed effects were "
        "carried over. Reported rather than dropped, because dropping it would "
        "hide that two of the five aripiprazole comparisons are the same "
        "number counted twice.",
    ),
    (
        "NDA 207533",
        "parent-and-metabolite",
        None,
        "aripiprazole",
        "aripiprazole, non poor metabolisers",
        "Fails by a factor of 56. The dataset is a depot programme with a 10 mg "
        "oral arm, and the fit has put the lauroxil release into the peripheral "
        "compartment: VP/F is 2122 L against a VC/F of 268 L with a Q/F of "
        "0.423 L/h. The slowest eigenvalue is a formulation property, not an "
        "elimination rate.",
    ),
    (
        "NDA 209830",
        "parent",
        None,
        "aripiprazole",
        "aripiprazole, non poor metabolisers",
        "Fails by a factor of 164, the same mechanism as NDA 207533 and worse: "
        "Q/F is 0.102 L/h with 353% inter-individual variability, so the "
        "peripheral compartment is close to unidentified and its eigenvalue is "
        "close to meaningless.",
    ),
    (
        "NDA 022264",
        "parent",
        None,
        "paliperidone",
        "paliperidone, extended-release oral",
        "Over-predicts by a factor of about two, and both sides of this "
        "comparison are compromised: the review states the depot's observed "
        "half-life is governed by flip-flop kinetics, and the label's 23 hours "
        "is for an OROS tablet whose release is also rate-limiting. There is no "
        "immediate-release paliperidone half-life in any current US label to "
        "compare against instead.",
    ),
    (
        "NDA 216352",
        "parent",
        None,
        "paliperidone",
        "paliperidone, extended-release oral",
        "Fails by a factor of 13. Same molecule and same depot class as NDA "
        "022264, which is within twofold; the difference is that this analysis "
        "estimated V/F as 2840 L where the other estimated V as 346 L, having "
        "absorbed the depot into the volume rather than into the absorption "
        "model.",
    ),
    (
        "NDA 207946",
        "parent",
        None,
        "paliperidone",
        "paliperidone, extended-release oral",
        "Fails by a factor of 15, and the failure is partly built in: the "
        "central volume was fixed at 1960 L rather than estimated, so this "
        "number is a constant chosen by the modeller divided by a fitted "
        "clearance. A 3-month depot with saturable release has nothing in its "
        "data to identify a disposition volume with, which is presumably why "
        "it was fixed.",
    ),
    (
        "NDA 213378",
        "parent",
        None,
        "olanzapine",
        "olanzapine",
        "The strongest agreement in the set. Adults, oral, two-compartment, "
        "against an adult oral label: 49.8 hours inside a printed 21-54 hour "
        "interval, with 70% of the area in the terminal phase so the two "
        "quantities are measuring the same thing.",
    ),
    (
        "NDA 022173",
        "parent",
        None,
        "olanzapine",
        "olanzapine",
        "Also inside the interval, at 34.3 hours, and this one is a depot. The "
        "counter-example to a tidy story that depots always fail: this "
        "analysis kept the depot release in an explicit mixed zero- and "
        "first-order input rather than letting a peripheral compartment absorb "
        "it, and the disposition survived.",
    ),
    (
        "NDA 209510",
        "parent",
        "BARHEMSYS intravenous injection, Applicant's model",
        "amisulpride",
        "amisulpride, intravenous",
        "Over-predicts by a factor of 2.7, and it is the one comparison in the "
        "set where neither flip-flop kinetics nor an unidentified peripheral "
        "compartment can be blamed: the drug is intravenous, the parameters "
        "are absolute, and the terminal phase carries 27% of the area. The "
        "likely explanation is the sampling window. Concentrations were drawn "
        "to 24 hours at most, and a non-compartmental half-life read off that "
        "window returns the dominant phase, which this model puts at 2.6 "
        "hours, rather than the true terminal one at 13.6. Partial dependence "
        "must be declared: the label's clearance figures are quoted directly "
        "from this model, so the label's PK section is not fully independent "
        "of it. The half-life statement itself is not a model output.",
    ),
)


def half_life_comparisons():
    """Every population-PK-implied terminal half-life against its label value.

    Returns all of them, including the four that fail by more than an order of
    magnitude. The failures are the informative part: they are not scattered,
    they are exactly the depot analyses that had no immediate-release or
    intravenous reference to pin the disposition, and the two-compartment ones
    among them carry the slow phase in a peripheral compartment that the
    label's non-compartmental sampling window never reached.
    """
    out = []
    for application, analyte, formulation, drug, entity, comment in _HALF_LIFE_PAIRINGS:
        row = record(application, analyte, formulation)
        fields = dict(
            application_number=application,
            analyte=analyte,
            formulation=row.formulation,
            predicted_h=terminal_half_life_h(row),
            label=label_half_life(drug, entity),
            terminal_auc_fraction=terminal_phase_auc_fraction(row),
            n_compartments=row.n_compartments,
            comment=comment,
        )
        # fold_error is a property of the comparison, so the pass/fail flag can
        # only be filled once the rest of it exists. Built twice rather than
        # made mutable.
        probe = HalfLifeComparison(agrees_within_twofold=False, **fields)
        out.append(
            HalfLifeComparison(
                agrees_within_twofold=probe.fold_error <= 2.0, **fields
            )
        )
    return tuple(out)
