"""P2.1: the real dose-to-behaviour datasets Track E found, re-examined now
that the Phase 1 dose-to-brain-concentration bridge exists.

What this module is, and what `evidence.py` next to it is not
--------------------------------------------------------------

`evidence.py` holds RECEPTOR-OCCUPANCY validation targets: a concentration
in, a bound fraction out, every one of them measured at a directly reported
plasma concentration. That is why those three were usable from the start -
concentration is the unit this codebase's state vector already spoke.

This module holds BEHAVIOURAL targets: a *dose* in (mg/kg, by a named route,
in a named species), a behavioural or neurochemical readout out. Track E
(`docs/architecture.md`, "Track E: circuit-layer calibration attempt")
established that no such dataset could be used, and diagnosed the reason
precisely: real behavioural pharmacology reports administered dose, and
`pk.py` at the time started from an already-achieved bolus concentration, so
there was no honest route from one to the other. Phase 1 built that route
(`pk.DosedOneCompartmentPK`/`DosedTwoCompartmentPK`, `drug_pk.py`'s
literature parameter sets, `brain_partition.py`'s plasma-to-brain step). P2.1
is the task that spends it.

The answer, up front, so it is not buried
-----------------------------------------

**The bridge did not, on its own, unblock any of the three papers Track E
named.** It moved the blocker, per system, to a different and more specific
place, and the point of this module is to record exactly where:

  fentanyl / NAc dopamine  [Yoshida1999]
      The bridge covers this one completely: fentanyl, rat, IV is a fully
      sourced parameter set (`drug_pk.fentanyl_rat_iv_pk`, [Yassen2008]).
      What is missing is now the *paper*, not the model: the article is
      closed access, its abstract states doses only as units ("microg/kg"),
      and neither the dose values nor the percent-of-baseline dopamine
      figures could be obtained from any source reachable here. Track E
      asserts the doses were "5 and 10 ug/kg IV"; that could NOT be
      re-verified in this pass and is recorded as such below rather than
      copied forward.

  haloperidol / locomotor suppression  [Gao2021]
      The opposite failure. The effect size IS published and was read from
      the open-access full text (ED50 0.11 mg/kg for reversing MK-801-induced
      hyperactivity), but it is in the MOUSE by the ORAL route, and this
      codebase has no mouse haloperidol PK by any route and no haloperidol
      oral parameter set in any species - both already recorded, before this
      task, in `drug_pk.SEARCHED_AND_EMPTY`. So the bridge cannot be applied,
      and Track E's second objection survives untouched besides: an ED50
      alone is not a Gaussian-likelihood target, because it carries neither
      the concentration nor the maximal effect it is 50% of.

  diazepam / elevated plus maze  [Pellow1985]
      Neither. The founding EPM paper is closed access; its PubMed record
      gives species, sex and the direction of the diazepam effect but no
      dose, no route, no timepoint and no magnitude. A bridge cannot convert
      a dose that was never obtained.

To keep this from being a module of three absences, it also carries one
dataset that DOES pass end to end - [Stickling2024], diazepam 1.5 mg/kg IP in
adult male Sprague-Dawley rats, 60 min before a 5 min EPM trial, with means
and SEMs printed in the paper's own table and a stated group-size range.
`drug_pk` has a rat IP diazepam model and a receptor-drivable rat diazepam
brain partition, so that entry converts all the way from mg/kg to an unbound
brain concentration in nM. It is the existence proof that the bridge works on
real behavioural
dosing; it is NOT yet a calibration target, for a reason recorded on the
entry itself (the paper's numeric EPM measure is one whose definition it
never states, and its open-arm times are figure-only).

Design rules, and why each exists
---------------------------------

1. **A missing field is `None` AND is listed in `provenance_gaps`.** The
   constructor enforces the pairing. This makes "we do not know this rat's
   body weight" a recorded fact with a reason attached, and makes it
   impossible to leave a field out quietly.
2. **No number is carried forward on the authority of this repository's own
   prose.** Track E's "5 and 10 ug/kg" is the test case: it is quoted below
   as a claim of `docs/architecture.md`, next to the record of the sources
   that were checked and did not confirm it, and it is not stored as a dose.
3. **Bridge coverage is computed, never asserted.** `pk_route_status()` asks
   `drug_pk` itself, so if a later phase sources haloperidol PO kinetics, the
   verdicts in this module change with it instead of going stale.
4. **Nothing here is a likelihood target yet.** P2.2 is the task that fits
   `mu_opioid_coupling`/`d2_coupling`/`gaba_a_connection_modulation`. This
   module deliberately exposes no `sd`, because inventing an allowance for a
   target nobody can compute yet would be the same error
   `evidence.OCCUPANCY_LIKELIHOOD_SD` at least declares itself to be.
"""

import numpy as np

from .. import drug_pk
from ..pk_validation import simulate_curve

# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------

#: The kinds of published effect this module can hold. `group_means` is the
#: ordinary "vehicle vs drug, mean +/- error, n per group" shape; `ed50` is
#: the dose producing half of that drug's own maximal effect, which is a
#: different kind of number and is deliberately not stored in the same slot.
EFFECT_KINDS = ("group_means", "ed50")

#: Error-bar kinds this module will accept. Recorded, never converted: SEM
#: and SD differ by sqrt(n), and silently turning one into the other is the
#: kind of arithmetic that should be visible at the point of use, not baked
#: into storage (`pk_validation.DISPERSION` does the same thing for PK).
DISPERSION_KINDS = ("SEM", "SD", "95% CI half-width")


class PublishedEffect(object):
    """One effect magnitude exactly as the source printed it.

    Parameters
    ----------
    kind : one of `EFFECT_KINDS`
    measure : str
        What was measured, in the source's own words where possible
        ("time spent in the open arms", "extracellular dopamine").
    units : str
        Units of `values` ("s", "% of baseline", "mg/kg").
    values : dict
        Group label -> value. For `ed50` this is a single entry.
    dispersion : dict or None
        Same keys as `values`. None means the source printed no error bar,
        which is itself worth recording and is not the same as zero.
    dispersion_kind : one of `DISPERSION_KINDS`, required iff `dispersion`.
    group_sizes : dict or None
        Same keys as `values`; n per group as printed.
    source_location : str or None
        Where in the paper the numbers are ("Table 1", "Results, para 2"), so
        a reader can check them without re-deriving the search.
    note : str or None
    """

    def __init__(
        self,
        kind,
        measure,
        units,
        values,
        dispersion=None,
        dispersion_kind=None,
        group_sizes=None,
        source_location=None,
        note=None,
    ):
        if kind not in EFFECT_KINDS:
            raise ValueError(
                "kind must be one of %s; got %r" % (", ".join(EFFECT_KINDS), kind)
            )
        if not measure or not units:
            raise ValueError("measure and units are required")
        if not values:
            raise ValueError("values must be a non-empty mapping")
        if dispersion is not None:
            if set(dispersion) != set(values):
                raise ValueError(
                    "dispersion keys %r do not match values keys %r"
                    % (sorted(dispersion), sorted(values))
                )
            if dispersion_kind not in DISPERSION_KINDS:
                raise ValueError(
                    "dispersion_kind must be one of %s when a dispersion is "
                    "given; got %r" % (", ".join(DISPERSION_KINDS), dispersion_kind)
                )
        elif dispersion_kind is not None:
            raise ValueError("dispersion_kind given without a dispersion")
        if group_sizes is not None and not set(group_sizes) <= set(values):
            raise ValueError("group_sizes keys must be a subset of values keys")

        self.kind = kind
        self.measure = str(measure)
        self.units = str(units)
        self.values = dict(values)
        self.dispersion = None if dispersion is None else dict(dispersion)
        self.dispersion_kind = dispersion_kind
        self.group_sizes = None if group_sizes is None else dict(group_sizes)
        self.source_location = source_location
        self.note = note

    @property
    def has_error_bars(self):
        return self.dispersion is not None

    def __repr__(self):
        return "PublishedEffect(%s, %r, %r)" % (self.kind, self.measure, self.values)


class UnavailableEffect(object):
    """Placeholder for an effect magnitude that was searched for and not
    obtained, carrying the reason.

    A distinct type rather than `None` for the same reason
    `drug_pk.UnsourcedParameterError` is a distinct exception: "we looked and
    it is not reachable" is a result in this codebase, and a caller that
    treats it as a missing dict key will get a type error instead of a
    silently skipped datum.
    """

    def __init__(self, measure, reason):
        if not measure or not reason:
            raise ValueError("measure and reason are required")
        self.measure = str(measure)
        self.reason = str(reason)

    has_error_bars = False

    def __repr__(self):
        return "UnavailableEffect(%r)" % (self.measure,)


# ---------------------------------------------------------------------------
# The entry type
# ---------------------------------------------------------------------------

#: Fields whose absence must be justified. A `None` in any of these without a
#: matching `provenance_gaps` entry is a construction error.
GAPPABLE_FIELDS = (
    "strain",
    "sex",
    "body_weight_kg",
    "doses_mg_per_kg",
    "timepoint_min",
    "route",
)


class BehavioralEvidenceEntry(object):
    """One published dose-to-behaviour dataset, with its provenance and its
    Phase 1 bridge status.

    `doses_mg_per_kg` is a tuple because a dose-response study reports
    several, and because a single-dose study reporting one is the degenerate
    case rather than a different shape.

    `timepoint_min` is minutes from dosing to the measurement the effect
    refers to. For a microdialysis time course with no single reported peak
    time it is a gap, not a guess.
    """

    def __init__(
        self,
        key,
        drug,
        species,
        strain,
        sex,
        route,
        doses_mg_per_kg,
        timepoint_min,
        endpoint,
        effect,
        citation,
        source_access,
        body_weight_kg=None,
        provenance_gaps=(),
        note=None,
    ):
        if not key or not drug or not species or not endpoint or not citation:
            raise ValueError(
                "key, drug, species, endpoint and citation are required: an "
                "uncited behavioural datum is the invented number this "
                "codebase forbids"
            )
        if not isinstance(effect, (PublishedEffect, UnavailableEffect)):
            raise TypeError(
                "effect must be a PublishedEffect or an UnavailableEffect, "
                "got %r" % (effect,)
            )
        if route is not None and str(route).upper() not in ("IV", "IP", "SC", "PO"):
            raise ValueError("unknown route %r" % (route,))
        if doses_mg_per_kg is not None:
            doses_mg_per_kg = tuple(float(d) for d in doses_mg_per_kg)
            if not doses_mg_per_kg or any(d <= 0.0 for d in doses_mg_per_kg):
                raise ValueError("doses must be positive")

        self.key = str(key)
        self.drug = str(drug).lower()
        self.species = str(species).lower()
        self.strain = strain
        self.sex = sex
        self.route = None if route is None else str(route).upper()
        self.doses_mg_per_kg = doses_mg_per_kg
        self.timepoint_min = None if timepoint_min is None else float(timepoint_min)
        self.endpoint = str(endpoint)
        self.effect = effect
        self.citation = str(citation)
        self.source_access = str(source_access)
        self.body_weight_kg = (
            None if body_weight_kg is None else float(body_weight_kg)
        )
        self.provenance_gaps = dict(provenance_gaps)
        self.note = note

        unexplained = [
            f
            for f in GAPPABLE_FIELDS
            if getattr(self, f) is None and f not in self.provenance_gaps
        ]
        if unexplained:
            raise ValueError(
                "%s: fields %s are absent with no recorded reason. Every gap "
                "in this module is a searched-and-empty result and has to say "
                "so." % (self.key, ", ".join(sorted(unexplained)))
            )
        spurious = [
            f for f in self.provenance_gaps if getattr(self, f, None) is not None
        ]
        if spurious:
            raise ValueError(
                "%s: %s are recorded as provenance gaps but have values"
                % (self.key, ", ".join(sorted(spurious)))
            )

    # --- Phase 1 bridge ----------------------------------------------------

    def pk_route_status(self):
        """`(status, explanation)`: can the Phase 1 bridge turn this entry's
        dosing into a concentration?

        Status is one of:

        ``"available"``
            `drug_pk` builds a model for this drug/species/route.
        ``"no-dosing-recorded"``
            The source did not give a dose or a route, so there is nothing to
            convert. The blocker is the paper, not the model.
        ``"unsourced"``
            `drug_pk` has a builder for the species but raises for this
            route; the explanation is that route's own searched-and-empty
            record.
        ``"no-species"``
            `drug_pk` has no builder for this drug in this species at all.

        Computed by asking `drug_pk`, never hard-coded, so that sourcing a
        new route later changes these verdicts automatically.
        """
        if self.route is None or self.doses_mg_per_kg is None:
            missing = [
                name
                for name, value in (
                    ("route", self.route),
                    ("dose", self.doses_mg_per_kg),
                )
                if value is None
            ]
            return (
                "no-dosing-recorded",
                "%s not obtainable from the source; see provenance_gaps"
                % " and ".join(missing),
            )
        builder = _PK_BUILDERS.get((self.drug, self.species))
        if builder is None:
            return (
                "no-species",
                "drug_pk has no %s PK builder for the %s by any route."
                % (self.drug, self.species),
            )
        try:
            builder(self.route, 0.25)
        except drug_pk.UnsourcedParameterError as exc:
            return ("unsourced", str(exc))
        return ("available", "drug_pk builds %s/%s/%s." % (
            self.drug, self.species, self.route))

    @property
    def is_bridgeable(self):
        return self.pk_route_status()[0] == "available"

    def pk_model(self, body_weight_kg):
        """The Phase 1 PK model for this entry's drug/species/route.

        `body_weight_kg` is required and never defaulted, following
        `drug_pk`'s own rule. Where the behavioural source did not report a
        body weight (the common case - behavioural papers report strain and
        age, not grams), the honest substitute is not a typical rat but an
        explicit choice by the caller, recorded next to the number it
        produces; `PK_SOURCE_BODY_WEIGHT_BANDS_KG` gives the band the PK
        parameters themselves were measured in, which is the defensible one.
        """
        status, explanation = self.pk_route_status()
        if status != "available":
            raise drug_pk.UnsourcedParameterError(
                "%s: %s (%s)" % (self.key, explanation, status)
            )
        return _PK_BUILDERS[(self.drug, self.species)](self.route, body_weight_kg)

    def brain_exposure(
        self,
        body_weight_kg,
        dose_mg_per_kg=None,
        timepoint_min=None,
        t_end_min=None,
    ):
        """Convert one of this entry's doses into a brain concentration.

        Returns a `BrainExposure`. `dose_mg_per_kg` defaults to the entry's
        highest recorded dose, `timepoint_min` to the entry's recorded
        timepoint (and raises if the entry has neither, rather than picking
        one).

        The concentration is read off an integrated trajectory, using
        `pk_validation.simulate_curve`, i.e. the same ODE path
        `engine.Simulation` runs - not a closed-form solution - so what this
        reports is what a simulation would actually see.
        """
        if dose_mg_per_kg is None:
            if self.doses_mg_per_kg is None:
                raise ValueError("%s records no dose to convert" % self.key)
            dose_mg_per_kg = max(self.doses_mg_per_kg)
        if timepoint_min is None:
            timepoint_min = self.timepoint_min
        if timepoint_min is None:
            raise ValueError(
                "%s records no timepoint; pass timepoint_min explicitly and "
                "say in the caller where it came from" % self.key
            )
        model = self.pk_model(body_weight_kg)
        if t_end_min is None:
            t_end_min = max(12.0 * model.terminal_half_life_min, 1.5 * timepoint_min)
        times, plasma = simulate_curve(model, dose_mg_per_kg, t_end_min=t_end_min)
        plasma_nM = float(np.interp(timepoint_min, times, plasma))

        partition = drug_pk.brain_partition_for(
            self.drug, self.species, require_receptor_drivable=False
        )
        status, _ = drug_pk.brain_partition_status(self.drug, self.species)
        if partition is None:
            brain_nM, basis = None, "none (no sourced brain partition)"
        elif partition.can_drive_receptor:
            brain_nM = partition.unbound_brain_concentration(plasma_nM)
            basis = "unbound brain"
        else:
            brain_nM = partition.total_brain_concentration(plasma_nM)
            basis = "total brain"
        return BrainExposure(
            entry=self,
            dose_mg_per_kg=dose_mg_per_kg,
            body_weight_kg=body_weight_kg,
            timepoint_min=timepoint_min,
            plasma_nM=plasma_nM,
            cmax_nM=float(np.max(plasma)),
            brain_nM=brain_nM,
            brain_basis=basis,
            partition_status=status,
            receptor_compartment=model.receptor_compartment,
            model_name=getattr(model, "name", repr(model)),
        )

    def __repr__(self):
        return "BehavioralEvidenceEntry(%r, %s/%s/%s)" % (
            self.key,
            self.drug,
            self.species,
            self.route,
        )


class BrainExposure(object):
    """The result of pushing one published dose through the Phase 1 bridge.

    `brain_nM` is None when no brain partition has been sourced for the
    drug/species, and `brain_basis` says which of the three P1.5 cases
    applies. Reporting a total-brain concentration as if it were the
    receptor-driving one is the mistake `brain_partition.py` exists to
    prevent, so the basis travels with the number.
    """

    def __init__(
        self,
        entry,
        dose_mg_per_kg,
        body_weight_kg,
        timepoint_min,
        plasma_nM,
        cmax_nM,
        brain_nM,
        brain_basis,
        partition_status,
        receptor_compartment,
        model_name,
    ):
        self.entry = entry
        self.dose_mg_per_kg = float(dose_mg_per_kg)
        self.body_weight_kg = float(body_weight_kg)
        self.timepoint_min = float(timepoint_min)
        self.plasma_nM = float(plasma_nM)
        self.cmax_nM = float(cmax_nM)
        self.brain_nM = None if brain_nM is None else float(brain_nM)
        self.brain_basis = brain_basis
        self.partition_status = partition_status
        self.receptor_compartment = receptor_compartment
        self.model_name = model_name

    @property
    def drives_receptor(self):
        """True when the number this carries is the one a `ReceptorSite` may
        legitimately bind."""
        return self.brain_basis == "unbound brain"

    def __repr__(self):
        return (
            "BrainExposure(%s, %g mg/kg, t=%g min, plasma=%.4g nM, "
            "%s=%s nM)"
            % (
                self.entry.key,
                self.dose_mg_per_kg,
                self.timepoint_min,
                self.plasma_nM,
                self.brain_basis,
                "None" if self.brain_nM is None else "%.4g" % self.brain_nM,
            )
        )


#: drug/species -> the `drug_pk` builder that takes `(route, body_weight_kg)`.
#: Only pairs `drug_pk` actually has a function for; a missing key is the
#: "no-species" verdict above, and is itself informative (there is no
#: haloperidol mouse builder anywhere in this codebase).
_PK_BUILDERS = {
    ("fentanyl", "rat"): drug_pk.fentanyl_rat_pk,
    ("fentanyl", "mouse"): drug_pk.fentanyl_mouse_pk,
    ("haloperidol", "rat"): drug_pk.haloperidol_rat_pk,
    ("diazepam", "rat"): drug_pk.diazepam_rat_pk,
    ("diazepam", "mouse"): drug_pk.diazepam_mouse_pk,
}

#: The body-weight band each PK parameter set was measured in, kg. Behavioural
#: papers overwhelmingly report strain and age rather than grams, and
#: `drug_pk` refuses to default a weight, so this is the defensible thing for
#: a caller to use: dose a rat of the size the KINETICS came from, and say so.
#: Weight cancels out of a per-kg-parameterised model (diazepam) and does not
#: cancel out of an absolute-parameter one (fentanyl, [Yassen2008]).
PK_SOURCE_BODY_WEIGHT_BANDS_KG = {
    ("fentanyl", "rat", "IV"): drug_pk.FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG,
    ("haloperidol", "rat", "IV"): None,  # [ChengPaalzow1992] is closed access
    ("diazepam", "rat", "IP"): None,  # [Friedman1986] abstract states no weight
}


# ---------------------------------------------------------------------------
# 1. Mu-opioid: fentanyl -> nucleus accumbens dopamine
# ---------------------------------------------------------------------------
#
# [Yoshida1999] Yoshida Y, Koide S, Hirose N, Takada K, Tomiyama K,
#               Koshikawa N, Cools AR. "Fentanyl increases dopamine release in
#               rat nucleus accumbens: involvement of mesolimbic mu- and
#               delta-2-opioid receptors." Neuroscience. 1999;92(4):1357-65.
#               doi:10.1016/S0306-4522(99)00046-9, PMID 10426490.
#               Closed access (Semantic Scholar reports openAccessPdf status
#               CLOSED and records that the publisher elided the abstract;
#               OpenAlex holds no abstract; the publisher's own article page
#               returns HTTP 403 to this environment). Track E already
#               recorded Europe PMC's isOpenAccess: N and the absence of a PMC
#               full text.
#               What the PubMed record DOES establish, and is used below:
#               species and strain (MeSH "Rats, Sprague-Dawley"), sex (MeSH
#               "Male"), route (MeSH "Injections, Intravenous"), the endpoint
#               (in vivo microdialysis of nucleus accumbens dopamine in awake
#               animals) and the direction ("Fentanyl dose-dependently
#               increased the levels of dopamine when given intravenously
#               (microg/kg)").
#               What it does not: the abstract as PubMed carries it states
#               every dose as a bare unit - "(microg/kg)", "(nmol)" - so it
#               contains no dose value at all, and no percentage, no n, no
#               body weight and no sampling time.
#
# A correction to this repository's own record, made here rather than left to
# be inherited: docs/architecture.md's Track E section states "Its abstract
# confirms fentanyl dose-dependently increased NAc dopamine at 5 and 10 ug/kg
# IV." The abstract reachable in this pass does not contain those values. They
# may well be right - they are exactly the kind of dose this assay uses - but
# nothing this project can reach confirms them, so they are recorded below as
# an unverified claim of an internal document and NOT stored as a dose. Rule 1
# of this codebase applies to numbers copied from our own prose as much as to
# numbers copied from a paper.

TRACK_E_UNVERIFIED_FENTANYL_DOSE_CLAIM = (
    "docs/architecture.md's Track E section states that [Yoshida1999]'s "
    "abstract reports fentanyl at 5 and 10 ug/kg IV. Re-checked 2026-08 "
    "against: PubMed EFetch (the abstract states doses only as units, "
    "'(microg/kg)'), Semantic Scholar (abstract elided by the publisher), "
    "OpenAlex (no abstract stored), and the publisher article page (HTTP "
    "403). None confirms the two values, so they are not stored as doses "
    "here. If a later pass obtains the full text, this record is the place "
    "to close."
)

FENTANYL_NAC_DOPAMINE_YOSHIDA1999 = BehavioralEvidenceEntry(
    key="fentanyl/NAc-dopamine/Yoshida1999",
    drug="fentanyl",
    species="rat",
    strain="Sprague-Dawley",
    sex="male",
    route="IV",
    doses_mg_per_kg=None,
    timepoint_min=None,
    endpoint=(
        "extracellular dopamine in the nucleus accumbens, in vivo "
        "microdialysis in awake freely-moving rats"
    ),
    effect=UnavailableEffect(
        measure="nucleus accumbens dopamine, percent of baseline",
        reason=(
            "Closed access; the percent-of-baseline figures are in the "
            "paper's figures. The abstract reports only the direction "
            "('dose-dependently increased'). Track E reached the same "
            "conclusion; this pass adds that the abstract does not carry the "
            "doses either."
        ),
    ),
    citation=(
        "Yoshida Y, Koide S, Hirose N, Takada K, Tomiyama K, Koshikawa N, "
        "Cools AR. Neuroscience. 1999;92(4):1357-65. "
        "doi:10.1016/S0306-4522(99)00046-9, PMID 10426490"
    ),
    source_access="closed; PubMed abstract + MeSH only",
    provenance_gaps={
        "doses_mg_per_kg": TRACK_E_UNVERIFIED_FENTANYL_DOSE_CLAIM,
        "timepoint_min": (
            "Microdialysis time course; the abstract reports no sampling "
            "interval and no time of maximal effect."
        ),
        "body_weight_kg": (
            "Not in the abstract or the MeSH record. [Yassen2008], whose "
            "kinetics would be used, measured 225-250 g rats; see "
            "PK_SOURCE_BODY_WEIGHT_BANDS_KG."
        ),
    },
    note=(
        "This is the entry where Phase 1 did its job and it still was not "
        "enough. fentanyl/rat/IV is fully sourced, so the moment a dose and "
        "a percent-of-baseline figure are obtained from this paper, "
        "`brain_exposure` runs on it unchanged. The remaining blocker is "
        "access to a 1999 closed-access article, not a missing submodel - a "
        "materially different, and cheaper, problem than the one Track E "
        "described."
    ),
)


# ---------------------------------------------------------------------------
# 2. D2: haloperidol -> locomotor suppression
# ---------------------------------------------------------------------------
#
# [Gao2021]     Gao L, Hao C, Ma R, Chen J, Zhang G, Chen Y. "Synthesis and
#               biological evaluation of a new class of multi-target
#               heterocycle piperazine derivatives as potential
#               antipsychotics." RSC Adv. 2021;11:16931-16941.
#               doi:10.1039/d1ra02426d, PMID 35479681, open access
#               (PMC9031908, read in full here).
#               Animals, Section 2.2.1 verbatim: "Chinese Kun Ming (KM) mice
#               (20 +/- 2.0 g) and Sprague-Dawley (SD) rats (250 +/- 5.0 g)
#               were used as experimental animals in this study." The
#               behavioural models are the mouse ones; the rats appear
#               elsewhere in the paper.
#               Haloperidol's own numbers, all ED50 in mg/kg by mouth
#               (Table 3, and confirmed in the prose): MK-801-induced
#               hyperactivity 0.11; apomorphine-induced climbing 0.09;
#               catalepsy 0.51 ("haloperidol (ED50: 0.51 mg kg-1) and
#               risperidone (ED50: 1.15 mg kg-1) induce significant cataleptic
#               effects"); acute toxicity LD50 21.0.
#               Not reported for haloperidol: sex, group sizes, the
#               pretreatment interval, any dispersion on the ED50s, and the
#               underlying per-dose response points from which they were
#               fitted. The flattened Table 3 as served by Europe PMC is also
#               column-misaligned against the prose (it puts 0.12 where the
#               text says catalepsy 0.51, and its printed CAT/APO and
#               CAT/MK-801 ratios only reconcile with 0.51), so every number
#               quoted here is one the running text states in words, not one
#               read out of the machine-readable table.

HALOPERIDOL_HYPERLOCOMOTION_GAO2021 = BehavioralEvidenceEntry(
    key="haloperidol/MK801-hyperlocomotion/Gao2021",
    drug="haloperidol",
    species="mouse",
    strain="Kun Ming (KM)",
    sex=None,
    route="PO",
    doses_mg_per_kg=(0.11,),
    timepoint_min=None,
    endpoint=(
        "reversal of MK-801-induced hyperlocomotion (locomotor activity), "
        "mouse"
    ),
    effect=PublishedEffect(
        kind="ed50",
        measure="ED50 for attenuating MK-801-induced hyperactivity",
        units="mg/kg",
        values={"haloperidol": 0.11},
        source_location="Table 3 and Section 2.7.2",
        note=(
            "An ED50 is 50% of THAT drug's own maximal effect, so on its own "
            "it fixes neither the effect size nor the concentration at which "
            "it occurs. Track E made this objection and it is untouched by "
            "the Phase 1 bridge; the bridge could only ever have supplied the "
            "concentration half. Same table, same units, for context: "
            "apomorphine-climbing ED50 0.09, catalepsy ED50 0.51, LD50 21.0."
        ),
    ),
    body_weight_kg=0.020,
    citation=(
        "Gao L, Hao C, Ma R, Chen J, Zhang G, Chen Y. RSC Adv. "
        "2021;11:16931-16941. doi:10.1039/d1ra02426d, PMID 35479681"
    ),
    source_access="open access (PMC9031908), read in full",
    provenance_gaps={
        "sex": "Section 2.2.1 gives strain and weight but not sex.",
        "timepoint_min": (
            "The paper states no interval between oral dosing and the "
            "locomotor measurement."
        ),
    },
    note=(
        "The mirror image of the fentanyl entry: the effect size exists and "
        "the bridge does not reach it. `drug_pk` has no mouse haloperidol PK "
        "by any route (there is no builder at all), and no haloperidol oral "
        "parameter set in any species; `pk_route_status()` reports "
        "'no-species' for exactly that reason. Note also that the dose stored "
        "here IS the ED50 - it is the only haloperidol dose the paper prints "
        "- so even a hypothetical mouse model would be converting a fitted "
        "summary statistic, not an administered dose."
    ),
)


# ---------------------------------------------------------------------------
# 3. GABA-A: diazepam -> elevated plus maze
# ---------------------------------------------------------------------------
#
# [Pellow1985]  Pellow S, Chopin P, File SE, Briley M. "Validation of
#               open:closed arm entries in an elevated plus-maze as a measure
#               of anxiety in the rat." J Neurosci Methods. 1985;14(3):149-67.
#               doi:10.1016/0165-0270(85)90031-7, PMID 2864480. Closed access.
#               The founding validation paper for the assay, and the one Track
#               E named.
#               From the PubMed record: rat (MeSH "Rats, Inbred Strains" - the
#               strain itself is not named), male (MeSH "Male"), endpoints
#               "the percentage of time spent on the open arms and the number
#               of entries into the open arms", and the finding that a
#               significant increase in both "was observed only within
#               clinically effective anxiolytics (chlordiazepoxide, diazepam
#               and, less effectively, phenobarbitone)".
#               Absent from everything reachable: the diazepam doses, the
#               route, the pretreatment interval, the group sizes, and every
#               numeric value.
#
# [Stickling2024] Stickling CP, Rosenkranz JA. "Effects of repeated social
#               stress on risk assessment behaviors and response to diazepam
#               in the elevated plus maze in adult male rats." Behav Neurosci.
#               2024 Dec 2;139(2):60-73. doi:10.1037/bne0000612, PMC11966652,
#               read in full here. Male Sprague Dawley rats (Envigo, n = 50
#               total), arrived at postnatal day 64-69, injected at PND 90-95;
#               "a single intraperitoneal injection of low dose diazepam in
#               vehicle (1.5 mg/kg body weight) or vehicle (sterile saline,
#               0.9% NaCl) administered at 2 mL/kg"; "each rat was placed on
#               the elevated plus maze (EPM) 60 min post injection"; 5 min
#               trial under 25-30 lux; n = 12-13 per group.
#               This paper is used here because it is the one diazepam EPM
#               dataset found in this pass that is (a) in the rat, (b) by a
#               route `drug_pk` can model, (c) at a stated timepoint, and (d)
#               numeric with error bars in the text rather than only in
#               figures.
#               Two caveats are stored on the entry, not buried: its
#               introduction says "diazepam (1 mg/kg)" while its Methods say
#               1.5 mg/kg, and the one EPM measure it prints numerically,
#               "closed arm preference", is never defined in the paper. Its
#               open-arm times are in Figure 4A only.

DIAZEPAM_EPM_PELLOW1985 = BehavioralEvidenceEntry(
    key="diazepam/elevated-plus-maze/Pellow1985",
    drug="diazepam",
    species="rat",
    strain=None,
    sex="male",
    route=None,
    doses_mg_per_kg=None,
    timepoint_min=None,
    endpoint=(
        "percentage of time spent on the open arms, and percentage of "
        "open-arm entries, elevated plus maze"
    ),
    effect=UnavailableEffect(
        measure="percentage of time spent on the open arms",
        reason=(
            "Closed access. The PubMed abstract reports the direction of the "
            "diazepam effect and nothing quantitative; Track E reached the "
            "same conclusion from the same source."
        ),
    ),
    citation=(
        "Pellow S, Chopin P, File SE, Briley M. J Neurosci Methods. "
        "1985;14(3):149-67. doi:10.1016/0165-0270(85)90031-7, PMID 2864480"
    ),
    source_access="closed; PubMed abstract + MeSH only",
    provenance_gaps={
        "strain": (
            'MeSH records only "Rats, Inbred Strains"; no strain name is '
            "given in the record."
        ),
        "route": "Not stated in the abstract or the MeSH record.",
        "doses_mg_per_kg": "Not stated in the abstract.",
        "timepoint_min": "Not stated in the abstract.",
        "body_weight_kg": "Not stated in the abstract.",
    },
    note=(
        "The bridge cannot be blamed or credited here: with neither a dose "
        "nor a route obtainable, there is nothing for it to convert. This is "
        "a dataset-access gap, and it is the reason [Stickling2024] below "
        "was sought out - not as a substitute for this paper's numbers, but "
        "as a diazepam-in-the-rat dataset that is complete enough to exercise "
        "the bridge at all."
    ),
)

DIAZEPAM_EPM_STICKLING2024 = BehavioralEvidenceEntry(
    key="diazepam/elevated-plus-maze/Stickling2024",
    drug="diazepam",
    species="rat",
    strain="Sprague Dawley (Envigo)",
    sex="male",
    route="IP",
    doses_mg_per_kg=(1.5,),
    timepoint_min=60.0,
    endpoint=(
        "elevated plus maze, 5 min trial at 25-30 lux, 60 min after a single "
        "IP injection; non-stressed (home-cage control) arm"
    ),
    effect=PublishedEffect(
        kind="group_means",
        measure="closed arm preference (control, i.e. non-stressed, rats)",
        units="s",
        values={"saline": 77.5, "diazepam 1.5 mg/kg": 76.7},
        dispersion={"saline": 12.1, "diazepam 1.5 mg/kg": 24.4},
        dispersion_kind="SEM",
        group_sizes={"saline": 12, "diazepam 1.5 mg/kg": 12},
        source_location="Table 1, rows 'Closed arm preference'",
        note=(
            'The paper never defines "closed arm preference", so this number '
            "is recorded with its units and its provenance but is NOT usable "
            "as a likelihood target: a target whose operational definition is "
            "unknown cannot be compared with a model readout. The group sizes "
            'are the low end of the paper\'s own "n = 12-13/group"; it does '
            "not print a per-cell n. The paper's open-arm times, which would "
            "be the usable measure, appear only in Figure 4A. Its own Table 1 "
            "also prints a saline-control EPM distance (3.69 m) that "
            "contradicts the difference and CI printed beside it, which is "
            "one more reason to treat its table as read-with-care."
        ),
    ),
    citation=(
        "Stickling CP, Rosenkranz JA. Behav Neurosci. 2024;139(2):60-73. "
        "doi:10.1037/bne0000612, PMC11966652"
    ),
    source_access="open access (PMC11966652), read in full",
    provenance_gaps={
        "body_weight_kg": (
            "Rats were weighed for dosing but no weight is printed; the paper "
            "reports age (PND 90-95) instead. Weight does not affect this "
            "entry's concentrations anyway: the rat IP diazepam model is "
            "parameterised per kg (V/F 19.3 L/kg, CL/F 255 mL/min/kg), so it "
            "cancels - `test_behavioral_evidence.py` asserts that rather "
            "than asserting it here."
        )
    },
    note=(
        "The one entry in this module that goes end to end. diazepam/rat/IP "
        "is a sourced (composite) parameter set and diazepam/rat is the one "
        "drug/species pair in this codebase with a receptor-drivable "
        "partition (Kp,uu 0.918), so `brain_exposure` returns an UNBOUND "
        "BRAIN concentration - the quantity a `ReceptorSite` may bind - "
        "rather than a plasma one. Dose discrepancy inside the source: its "
        "introduction says 'diazepam (1 mg/kg)' while its Methods and its "
        "Figure 1 legend both say 1.5 mg/kg. The Methods value is used, and "
        "the discrepancy is recorded rather than silently resolved."
    ),
)


#: Every entry, in the order the Track E section discusses them.
BEHAVIORAL_EVIDENCE = (
    FENTANYL_NAC_DOPAMINE_YOSHIDA1999,
    HALOPERIDOL_HYPERLOCOMOTION_GAO2021,
    DIAZEPAM_EPM_PELLOW1985,
    DIAZEPAM_EPM_STICKLING2024,
)

#: Keyed access, for tests and for P2.2.
BEHAVIORAL_EVIDENCE_BY_KEY = {e.key: e for e in BEHAVIORAL_EVIDENCE}


def bridgeable_entries():
    """Entries the Phase 1 bridge can convert from dose to concentration."""
    return tuple(e for e in BEHAVIORAL_EVIDENCE if e.is_bridgeable)


def unbridgeable_entries():
    """Entries it cannot, each still carrying its own reason via
    `pk_route_status()`."""
    return tuple(e for e in BEHAVIORAL_EVIDENCE if not e.is_bridgeable)


def calibration_ready_entries():
    """Entries that could serve as a P2.2 likelihood target as they stand.

    Currently empty, and that is the finding, not an oversight: an entry is
    calibration-ready only if the bridge reaches it AND it carries a published
    effect magnitude with an error bar AND that magnitude's operational
    definition is known. No entry in this module satisfies all three yet, for
    three different reasons - recorded per entry above and summarised in
    `BRIDGE_VERDICT`.
    """
    ready = []
    for entry in BEHAVIORAL_EVIDENCE:
        if not entry.is_bridgeable:
            continue
        effect = entry.effect
        if not isinstance(effect, PublishedEffect) or not effect.has_error_bars:
            continue
        if effect.kind != "group_means":
            continue
        if entry.key in _EFFECT_DEFINITION_UNKNOWN:
            continue
        ready.append(entry)
    return tuple(ready)


#: Entries whose stored effect magnitude has no stated operational
#: definition in its source, and therefore cannot be compared with any model
#: readout even though it has a mean, an error bar and an n.
_EFFECT_DEFINITION_UNKNOWN = frozenset(
    ["diazepam/elevated-plus-maze/Stickling2024"]
)


BRIDGE_VERDICT = {
    "fentanyl/NAc-dopamine/Yoshida1999": (
        "Bridge available (fentanyl/rat/IV, [Yassen2008]); dataset not "
        "obtainable. Blocker moved from 'this codebase has no dose-to-"
        "concentration submodel' to 'this 1999 article is closed access and "
        "its abstract carries neither its doses nor its effect sizes'."
    ),
    "haloperidol/MK801-hyperlocomotion/Gao2021": (
        "Bridge unavailable (no mouse haloperidol PK by any route; no "
        "haloperidol oral parameter set in any species - both already in "
        "drug_pk.SEARCHED_AND_EMPTY before this task); dataset obtainable but "
        "an ED50, which is not a likelihood target on its own. Both of Track "
        "E's objections survive."
    ),
    "diazepam/elevated-plus-maze/Pellow1985": (
        "Bridge inapplicable: the source supplies neither a dose nor a route, "
        "so there is nothing to convert. Dataset not obtainable."
    ),
    "diazepam/elevated-plus-maze/Stickling2024": (
        "Bridge available and exercised end to end: 1.5 mg/kg IP -> plasma -> "
        "unbound brain nM. Not yet a likelihood target because the numeric "
        "EPM measure it prints is undefined in the source and its open-arm "
        "times are figure-only."
    ),
}


# ---------------------------------------------------------------------------
# Searched-and-empty record (the Track E template, applied to behaviour)
# ---------------------------------------------------------------------------

SEARCHED_AND_EMPTY_BEHAVIORAL = {
    ("fentanyl", "rat", "IV", "nucleus accumbens dopamine"): (
        "No fentanyl percent-of-baseline accumbens dopamine figure at a "
        "stated IV dose in the rat could be obtained.\n"
        "Searched (2026-08): the source paper itself ([Yoshida1999]) via "
        "PubMed EFetch, Semantic Scholar, OpenAlex, Europe PMC and the "
        "publisher page; Europe PMC full-text search for open-access rat "
        "fentanyl accumbens microdialysis studies (77 hits screened).\n"
        "Candidates found and why each is unusable: [Yoshida1999] itself is "
        "closed access and its abstract states doses as units only. "
        "Sustkova-Fiserova et al., Int J Mol Sci 2017;18(11):2486, "
        "doi:10.3390/ijms18112486, PMID 29165386 (PMC5713452, read in "
        "full) is the right drug, right species, right region and "
        "reports real percent-of-baseline numbers - but its systemic "
        "fentanyl is 30 ug/kg SUBCUTANEOUS, and drug_pk.SEARCHED_AND_EMPTY"
        "[('fentanyl','rat','SC')] records that no ka exists for that route, "
        "so no SC model can be built; its measured analytes are anandamide, "
        "2-AG and GABA rather than dopamine besides. The same group's "
        "Addiction Biology 2020 fentanyl paper (doi:10.1111/adb.12845), which "
        "does measure accumbens dopamine, is closed access.\n"
        "What would close it: full text of [Yoshida1999], or any rat study "
        "reporting accumbens dopamine as percent of baseline after an "
        "INTRAVENOUS fentanyl dose."
    ),
    ("haloperidol", "rat", "IV", "locomotor activity"): (
        "No haloperidol locomotor-activity dose-response by the INTRAVENOUS "
        "route in the rat was found, and IV is the only route drug_pk has "
        "haloperidol kinetics for.\n"
        "Searched (2026-08): Europe PMC (haloperidol AND 'locomotor "
        "activity' AND rats AND intravenous, open access; 219 hits, of which "
        "the first 25 by relevance were screened and none is an IV "
        "behavioural dose-response - they are reviews and nanoparticle/"
        "intranasal delivery papers) and web search for the same.\n"
        "Why the literature looks like this: haloperidol behavioural "
        "pharmacology doses by IP, SC or PO essentially without exception "
        "(0.025-1.0 mg/kg IP is the standard range in the papers screened). "
        "drug_pk.SEARCHED_AND_EMPTY already records that no F and no ka "
        "exists for haloperidol IP or SC in the rat. So the gap for this "
        "system is not one missing paper: it is that the routes the PK "
        "literature covers and the routes the behavioural literature uses do "
        "not intersect for this drug.\n"
        "What would close it: an absolute bioavailability and absorption rate "
        "constant for haloperidol IP or SC in the rat (which would make the "
        "large IP behavioural literature reachable), or mouse haloperidol PK "
        "plus an oral ka (which would make [Gao2021] reachable)."
    ),
    ("diazepam", "rat", "IP", "elevated plus maze open-arm time"): (
        "No diazepam-specific percent-open-arm-time figure with an error bar, "
        "for the rat by the IP route, was found in text or table form.\n"
        "Searched (2026-08): Europe PMC full-text search over the open-access "
        "rat EPM literature. The widest sweep took all 600 open-access "
        "articles matching (diazepam AND 'elevated plus' AND rats), fetched "
        "each full text, and kept those that (a) never mention mice, (b) "
        "contain a diazepam mg/kg IP dosing statement, and (c) contain a "
        "table with an open-arm measure carrying a '+/-'. Zero of 600 "
        "matched. Two narrower query formulations plus targeted reads of the "
        "most promising hits were run before it.\n"
        "Candidates found and why each is unusable: the overwhelming majority "
        "of matches are mouse studies (no mouse diazepam PK exists here; "
        "drug_pk.SEARCHED_AND_EMPTY[('diazepam','mouse','*')]); Nachiappan "
        "KR, Sadasivam B, Najmi A, K C. Cureus 2023;15:e38717, "
        "doi:10.7759/cureus.38717 (PMC10246733) is a rat study printing "
        "diazepam 0.5 and 1 mg/kg open-arm times with SEMs in a table, but "
        "dosed ORALLY, and drug_pk has no diazepam PO model because [Hua2025] "
        "does not identify a ka; Saitoh et al., Front Pharmacol 2022;13:826783 "
        "(PMC8940232) prints diazepam EPM numbers but in mice by the SC "
        "route; the rat "
        "papers that are IP (e.g. PMC13213177, PMC10514533) put their "
        "open-arm values in bar charts only. [Stickling2024] was kept because "
        "it is the one rat/IP/stated-timepoint study whose EPM numbers are "
        "printed rather than plotted, even though the measure it prints is "
        "undefined.\n"
        "What this does NOT license: using the chlordiazepoxide figures Track "
        "E found (9.4% vehicle vs 39.1% at 5 mg/kg, PMC12564412) as if they "
        "were diazepam's. That substitution was refused in Track E and is "
        "refused again here."
    ),
}


__all__ = [
    "EFFECT_KINDS",
    "DISPERSION_KINDS",
    "GAPPABLE_FIELDS",
    "PublishedEffect",
    "UnavailableEffect",
    "BehavioralEvidenceEntry",
    "BrainExposure",
    "PK_SOURCE_BODY_WEIGHT_BANDS_KG",
    "TRACK_E_UNVERIFIED_FENTANYL_DOSE_CLAIM",
    "FENTANYL_NAC_DOPAMINE_YOSHIDA1999",
    "HALOPERIDOL_HYPERLOCOMOTION_GAO2021",
    "DIAZEPAM_EPM_PELLOW1985",
    "DIAZEPAM_EPM_STICKLING2024",
    "BEHAVIORAL_EVIDENCE",
    "BEHAVIORAL_EVIDENCE_BY_KEY",
    "BRIDGE_VERDICT",
    "SEARCHED_AND_EMPTY_BEHAVIORAL",
    "bridgeable_entries",
    "unbridgeable_entries",
    "calibration_ready_entries",
]
