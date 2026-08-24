"""The rodent dose-to-behaviour bridge, retired here by P11.3.

This was `calibration/behavioral_evidence.py` from P2.1 until P11.3. It holds
all eleven rodent behavioural datasets the project ever gathered - ten rat,
one mouse, zero human - plus the value types, ratio math and coverage
registries that were built around them. P11.3 moved it here whole because the
v2 roadmap (`docs/roadmap-human.md`) makes the project human-primary: every
dataset below describes a rat or a mouse, so none of them may calibrate a
human model, and the honest place for them is a quarantine that keeps them
importable and tested rather than a deletion that loses the provenance.

What still lives in production is the species-agnostic machinery that was
built to consume this data and now waits for human observations:
`calibration/bayesian_calibration.py`, `calibration/identifiability.py`,
`calibration/endpoint_calibration.py`, `calibration/circuit_forward.py`.
They run against synthetic or human targets; nothing reads this module from
production except `heldout.training_compounds()`, which reports which drugs
Phase 1's PK layer was validated against.

The one fitted parameter this module fed - `gaba_a_connection_modulation`'s
gain 0.0742 against [Graziani2022] - was retired with it. The derivation of
its likelihood target survives below as
`GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE`, kept so the historical posterior in
`results/calibration/gaba_a_summary.csv` stays reproducible from quarantined
data. See `docs/heldout-failure-analysis.md`.

Earlier history (P2.1 / P2.2)
-----------------------------

What this module is, and what `evidence.py` next to it was not
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
4. **Every uncertainty here is the source's own.** P2.1 exposed no `sd` at
   all, because no target was computable and an allowance for nothing would
   have been the same error `evidence.OCCUPANCY_LIKELIHOOD_SD` at least
   declares itself to be. P2.2 changed the first half of that: one entry now
   IS a likelihood target, and the width it carries
   (`anxiolysis_ratio`) is propagated from the source's own printed SEMs by
   the delta method, not chosen. The second half stands: nothing in this
   module invents a dispersion.

What P2.2 added
---------------

[Graziani2022], below: diazepam 2 mg/kg IP in drug-naive male Wistar rats,
5 min elevated plus maze, vehicle and drug open-arm times printed with SEMs
and group sizes. It is the first entry in this module that is
calibration-ready, and it is the target `bayesian_calibration.py` fits
`gaba_a_connection_modulation`'s gain against. It also corrects a P2.1
result: P2.1's widest sweep reported "Zero of 600" for exactly this query
and criteria, and [Graziani2022] meets all three of them. The correction is
recorded in `SEARCHED_AND_EMPTY_BEHAVIORAL` next to the claim it supersedes.

The same P2.2 pass re-ran the fentanyl and haloperidol sweeps at the same
scale and they are still empty, for reasons now stated more precisely in
those two records. So P2.2 fits one of the three coupling gains, not three.
"""

import numpy as np

from . import drug_pk
from .pk_validation import simulate_curve

# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------

#: The kinds of published effect this module can hold. `group_means` is the
#: ordinary "vehicle vs drug, mean +/- error, n per group" shape; `ed50` is
#: the dose producing half of that drug's own maximal effect, which is a
#: different kind of number and is deliberately not stored in the same slot.
#: `logistic_regression` is P7.3's addition: a source that published its
#: dose- or concentration-response as a FITTED CURVE rather than as per-dose
#: means. [Yassen2008]'s antinociception model is the case - its 486
#: observations in 40 rats are summarised by an intercept and a slope with
#: 95% confidence intervals, and the per-observation data are not printed.
#: It is a distinct kind rather than a `group_means` entry with two rows
#: because the numbers are parameters of a model, not measurements of a
#: group, and a caller that treats them as the latter would be comparing
#: the wrong things.
EFFECT_KINDS = ("group_means", "ed50", "logistic_regression")

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

    def receptor_drive_status(self):
        """`(status, explanation)`: can this entry's dose reach a RECEPTOR?

        `pk_route_status` answers "can the dose become a plasma
        concentration". This answers the next question, which P7.3 needed
        and P2.1/P2.2 never had to ask because both of their bridgeable
        entries were diazepam: is there a plasma-to-brain step that yields
        an UNBOUND BRAIN concentration, the only kind `ReceptorSite` may
        legitimately bind?

        Status is `"receptor-drivable"`, `"total-brain-only"`, `"none"` (no
        partition sourced at all) or the PK status when the PK step already
        fails. Asked of `drug_pk`, never hard-coded, so sourcing a rat
        brain unbound fraction later changes this verdict automatically.

        This is why the fentanyl entries are `available` above and still
        cannot be fitted: `drug_pk.SEARCHED_AND_EMPTY` records that no rat
        f(u,brain) for fentanyl was found, so its Kp of ~16.5 is a
        total-brain ratio that "must not drive receptor occupancy".
        """
        pk_status, pk_explanation = self.pk_route_status()
        if pk_status != "available":
            return pk_status, pk_explanation
        status, explanation = drug_pk.brain_partition_status(
            self.drug, self.species
        )
        return status, explanation

    @property
    def can_drive_receptor(self):
        return self.receptor_drive_status()[0] == "receptor-drivable"

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


# [Graziani2022]  Graziani D, Ribeiro JVV, Cruz VS, Gomes RM, Araújo EG,
#               Santos ACM Jr, Ferreira PM, Xavier CH, Rebelo ACS, Gomes RM,
#               Pedrino GR, et al. "Behavioral effects of a low molecular
#               weight peptide fraction from Phaseolus vulgaris in rats."
#               Braz J Med Biol Res. 2022;55:e12314.
#               doi:10.1590/1414-431X2022e12314, PMID 36477952, PMC9728632,
#               open access, read in full here.
#               Found in P2.2 by re-running P2.1's own sweep with the
#               screening step that had excluded it (see below). Male Wistar
#               rats, 300-380 g, drug-naive, no disease model. Methods,
#               verbatim: "the benzodiazepine diazepam (Sigma, USA) at
#               2 mg/kg, used as positive control for anxiety and locomotion
#               tests"; "All drugs and vehicle were injected by the ip
#               route"; "In the acute trials, animals received ip injections
#               (100 uL) of PV3 doses 1 h before trials"; EPM trial "5 min".
#               Table 2 ("Data are reported as means+/-SE"), the two rows this
#               module uses, in seconds and counts:
#                   Vehicle      n=11  open-arm time 30.7+/-6.4
#                                      open-arm entries 1.8+/-0.4
#                                      total entries 7.5+/-0.9
#                   DZP 2 mg/kg  n=8   open-arm time 89.0+/-13.0
#                                      open-arm entries 6.4+/-0.7
#                                      total entries 12.8+/-0.6
#               The table is internally consistent, which is why its units
#               (seconds, and a 300 s trial) can be asserted rather than
#               assumed: open + closed + centre time is 302.7 s (vehicle) and
#               299.9 s (diazepam) against a stated 5 min trial, and open +
#               closed entries reproduce the printed total in both rows.
#               `test_behavioral_evidence.py` checks both sums instead of
#               trusting this comment.
#
# Why P2.1's sweep missed it, stated plainly rather than quietly fixed:
# P2.1 screened "all 600 open-access Europe PMC articles matching (diazepam
# AND 'elevated plus' AND rats)" and kept only articles that "(a) never
# mention mice, (b) contain a diazepam mg/kg IP dosing statement, and (c)
# contain a table with an open-arm measure carrying a '+/-'", reporting
# "Zero of 600 matched". Re-run here, the same query returns 724 records and
# [Graziani2022] satisfies all three criteria, so criterion (c)'s
# table-detection step, not the literature, is what returned zero. The
# reproducible finding: the rat EPM literature does tabulate diazepam
# open-arm times; it is scarce, not absent. `SEARCHED_AND_EMPTY_BEHAVIORAL`
# below is corrected accordingly rather than left standing.

#: Trial duration of [Graziani2022]'s EPM, s. "The rat was placed in the
#: centre of the maze facing one of the closed arms and allowed to explore
#: the maze for 5 min." Needed because the fraction of the trial spent on
#: the open arms, not the raw seconds, is the bounded quantity P2.2 fits
#: against (see `open_arm_avoidance_fraction`).
GRAZIANI2022_EPM_TRIAL_DURATION_S = 300.0

#: Body-weight band [Graziani2022] reports, kg. Recorded rather than
#: averaged into a single number at the point of use; the rat IP diazepam
#: model is per-kg, so the band's width does not propagate into any
#: concentration (proved by
#: `test_bridged_concentration_is_weight_invariant_for_the_per_kg_model`).
GRAZIANI2022_BODY_WEIGHT_BAND_KG = (0.300, 0.380)

DIAZEPAM_EPM_GRAZIANI2022 = BehavioralEvidenceEntry(
    key="diazepam/elevated-plus-maze/Graziani2022",
    drug="diazepam",
    species="rat",
    strain="Wistar",
    sex="male",
    route="IP",
    doses_mg_per_kg=(2.0,),
    timepoint_min=None,
    endpoint=(
        "elevated plus maze, 5 min trial, drug-naive rats, no disease model; "
        "time spent on the open arms and entries, vehicle vs diazepam"
    ),
    effect=PublishedEffect(
        kind="group_means",
        measure=(
            "time spent in the open arms (s), entries into the open arms, "
            "and total arm entries, elevated plus maze"
        ),
        units="s (time rows); count (entry rows)",
        values={
            "vehicle open_arm_time_s": 30.7,
            "diazepam 2 mg/kg open_arm_time_s": 89.0,
            "vehicle open_arm_entries": 1.8,
            "diazepam 2 mg/kg open_arm_entries": 6.4,
            "vehicle total_entries": 7.5,
            "diazepam 2 mg/kg total_entries": 12.8,
        },
        dispersion={
            "vehicle open_arm_time_s": 6.4,
            "diazepam 2 mg/kg open_arm_time_s": 13.0,
            "vehicle open_arm_entries": 0.4,
            "diazepam 2 mg/kg open_arm_entries": 0.7,
            "vehicle total_entries": 0.9,
            "diazepam 2 mg/kg total_entries": 0.6,
        },
        dispersion_kind="SEM",
        group_sizes={
            "vehicle open_arm_time_s": 11,
            "diazepam 2 mg/kg open_arm_time_s": 8,
            "vehicle open_arm_entries": 11,
            "diazepam 2 mg/kg open_arm_entries": 8,
            "vehicle total_entries": 11,
            "diazepam 2 mg/kg total_entries": 8,
        },
        source_location="Table 2, rows 'Vehicle' and 'DZP 2 mg/kg'",
        note=(
            "The measure this entry is fitted through is the paper's own "
            "primary EPM variable ('Six variables were measured: i) time "
            "spent in the open arms; ...'), not a composite the source does "
            "not define - the objection that keeps [Stickling2024]'s 'closed "
            "arm preference' out of the fit. The entry rows are carried "
            "because they support the alternative anxiety index used in the "
            "P2.2 sensitivity analysis, not because the primary fit needs "
            "them."
        ),
    ),
    citation=(
        "Graziani D, et al. Braz J Med Biol Res. 2022;55:e12314. "
        "doi:10.1590/1414-431X2022e12314, PMID 36477952, PMC9728632"
    ),
    source_access="open access (PMC9728632), read in full",
    provenance_gaps={
        "body_weight_kg": (
            "The paper gives a band (300-380 g), not a single weight, so no "
            "single value is stored; see GRAZIANI2022_BODY_WEIGHT_BAND_KG. "
            "The rat IP diazepam model is parameterised per kg, so the band "
            "does not propagate into a concentration."
        ),
        "timepoint_min": (
            "Not stated for diazepam specifically. The paper's acute-trial "
            "protocol is 'In the acute trials, animals received ip "
            "injections (100 uL) of PV3 doses 1 h before trials', and Table "
            "1 lists 'DZP 2 mg/kg' among the acute treatments, but the "
            "sentence that carries the interval names PV3 and not diazepam. "
            "One hour is therefore the paper's stated acute-trial interval "
            "and NOT an explicitly stated diazepam interval, so it is not "
            "stored on this entry: a caller must pass a timepoint and say "
            "where it came from. P2.2 passes 60 min and reports the "
            "posterior at 30 and 45 min as well, because at this dose the "
            "benzodiazepine site is near saturation across that whole "
            "window and the choice barely moves the fit."
        ),
    },
    note=(
        "The one entry in this module that is calibration-ready. Drug-naive "
        "rats, no stress or disease model the amygdala circuit does not "
        "represent; the route (IP) and species (rat) are the pair drug_pk "
        "models and brain_partition.py can drive a receptor with; the effect "
        "is the assay's own primary measure with an error bar, an n, and a "
        "vehicle group measured in the same session. It is the target P2.2 "
        "fits gaba_a_connection_modulation against."
    ),
)


# ---------------------------------------------------------------------------
# 4. GABA-A: diazepam -> open field (P7.3: locomotion and sedation)
# ---------------------------------------------------------------------------
#
# The same paper as entry 3, a different table, and two different P7.1
# endpoints. [Graziani2022] ran an open field on the same protocol as its
# elevated plus maze ("the benzodiazepine diazepam (Sigma, USA) at 2 mg/kg,
# used as positive control for anxiety and locomotion tests"; "All drugs and
# vehicle were injected by the ip route"; "Each rat was placed in the center
# of the field and allowed to explore the apparatus for 5 min"), and its
# Table 3 prints the result with means, SEs and group sizes:
#
#   Treatment      n    periphery(s)  centre(s)  immobility(s)  "Hearing"  grooming  crossings
#   Vehicle        11   291.5+/-1.6   8.4+/-1.6  57.4+/-12.2    9.4+/-1.1  3.4+/-0.9 31.1+/-2.6
#   DZP 2 mg/kg     4   284.3+/-5.6  10.1+/-4.3  43.6+/-10.5   11.5+/-2.4  3.3+/-0.6 43.0+/-6.1
#
# Two defects in that table are recorded on the entry rather than repaired:
#
# 1. **The group size disagrees with the paper's own Table 1**, which lists
#    the diazepam OF group as n = 8 while Table 3's own "Sample(n)" column
#    says 4. The smaller of the two is stored, because it is the one printed
#    next to the numbers being used, and both are named in the note. This is
#    the same discipline P7.2 applied to [Caggiano2018]'s deposit, which
#    disagreed with its own figure legend about n.
# 2. **The rearing column is headed "Hearing episodes"**, which is a
#    typographical error for "Rearing episodes": rearing is one of the six
#    standard open-field measures, hearing episodes are not a measure at
#    all, and the paper's Results discuss rearing. That column is NOT used
#    here - neither endpoint needs it - so the reading is recorded and never
#    relied on.
#
# Which measure belongs to which endpoint is fixed by P7.1's declared assay
# strings and not chosen here:
#
#   locomotion  assay "open-field / photobeam ambulation" -> CROSSING EPISODES
#   sedation    assay "reduced spontaneous locomotion, loss of righting
#               reflex, EEG-defined NREM, sleep latency" -> IMMOBILITY TIME
#
# Those two are the same animals measured in the same 5 min, read in
# opposite directions, and that is a property of the assay rather than a
# convenience: a benzodiazepine's locomotor effect and its sedative effect
# are one behaviour at open-field resolution. It is stated on the entry so
# that a reader does not mistake two endpoints for two experiments.

#: Trial duration of [Graziani2022]'s open field, s. Same 5 min as its EPM,
#: stated separately because the two are separate sentences in the source.
GRAZIANI2022_OPEN_FIELD_DURATION_S = 300.0

DIAZEPAM_OPEN_FIELD_GRAZIANI2022 = BehavioralEvidenceEntry(
    key="diazepam/open-field/Graziani2022",
    drug="diazepam",
    species="rat",
    strain="Wistar",
    sex="male",
    route="IP",
    doses_mg_per_kg=(2.0,),
    timepoint_min=None,
    endpoint=(
        "open field, 5 min trial, drug-naive rats, no disease model; "
        "crossing episodes (P7.1 locomotion) and immobility time (P7.1 "
        "sedation), vehicle vs diazepam"
    ),
    effect=PublishedEffect(
        kind="group_means",
        measure=(
            "crossing episodes, immobility time (s), time spent in the "
            "centre (s) and in the periphery (s), open field"
        ),
        units="count (crossings); s (times)",
        values={
            "vehicle crossings": 31.1,
            "diazepam 2 mg/kg crossings": 43.0,
            "vehicle immobility_s": 57.4,
            "diazepam 2 mg/kg immobility_s": 43.6,
            "vehicle centre_time_s": 8.4,
            "diazepam 2 mg/kg centre_time_s": 10.1,
            "vehicle periphery_time_s": 291.5,
            "diazepam 2 mg/kg periphery_time_s": 284.3,
        },
        dispersion={
            "vehicle crossings": 2.6,
            "diazepam 2 mg/kg crossings": 6.1,
            "vehicle immobility_s": 12.2,
            "diazepam 2 mg/kg immobility_s": 10.5,
            "vehicle centre_time_s": 1.6,
            "diazepam 2 mg/kg centre_time_s": 4.3,
            "vehicle periphery_time_s": 1.6,
            "diazepam 2 mg/kg periphery_time_s": 5.6,
        },
        dispersion_kind="SEM",
        group_sizes={
            "vehicle crossings": 11,
            "diazepam 2 mg/kg crossings": 4,
            "vehicle immobility_s": 11,
            "diazepam 2 mg/kg immobility_s": 4,
            "vehicle centre_time_s": 11,
            "diazepam 2 mg/kg centre_time_s": 4,
            "vehicle periphery_time_s": 11,
            "diazepam 2 mg/kg periphery_time_s": 4,
        },
        source_location="Table 3, rows 'Vehicle' and 'DZP 2 mg/kg'",
        note=(
            "Centre and periphery times are carried because they are the "
            "internal-consistency check on the table's units (they sum to "
            "the stated 5 min trial), not because either endpoint reads "
            "them. The rearing column is headed 'Hearing episodes' in the "
            "source and is not used. Table 1 of the same paper gives n = 8 "
            "for this group where Table 3 gives 4; the printed-next-to-the-"
            "numbers value is the one stored."
        ),
    ),
    citation=(
        "Graziani D, et al. Braz J Med Biol Res. 2022;55:e12314. "
        "doi:10.1590/1414-431X2022e12314, PMID 36477952, PMC9728632"
    ),
    source_access="open access (PMC9728632), read in full",
    provenance_gaps={
        "body_weight_kg": (
            "The paper gives a band (300-380 g), not a single weight; see "
            "GRAZIANI2022_BODY_WEIGHT_BAND_KG. The rat IP diazepam model is "
            "per kg, so the band does not propagate into a concentration."
        ),
        "timepoint_min": (
            "Not stated for diazepam specifically, exactly as for this "
            "paper's EPM arm: the sentence carrying the 1 h acute-trial "
            "interval names the test peptide PV3, not diazepam. A caller "
            "must pass a timepoint and say where it came from."
        ),
    },
    note=(
        "P7.3's locomotion and sedation entry, and the smallest usable "
        "dataset in this module: n = 4 in the drug group, one dose, one "
        "vehicle group. It is a two-point contrast and NOT a dose-response "
        "curve, which is the distinction P7.3 turns on - see "
        "`ENDPOINT_COVERAGE` and docs/architecture.md's P7.3 section for "
        "why a two-point contrast cannot separate a circuit gain from an "
        "unknown behavioural scale, and why this entry is therefore "
        "recorded, bridged and reported without a gain being fitted from it."
    ),
)


# ---------------------------------------------------------------------------
# 5. Mu-opioid: fentanyl -> tail flick (P7.3: analgesia)
# ---------------------------------------------------------------------------
#
# [Yassen2008] is already this codebase's fentanyl rat IV PK source
# (`drug_pk.fentanyl_rat_iv_pk`, Table II), and P2.2's record noted in
# passing that it is also a PK/PD paper whose pharmacodynamic half "needs no
# dose bridge at all". P7.3 is the task that spends that observation, and
# doing so required reading the pharmacodynamic half properly rather than
# from the abstract.
#
#   Yassen A, Olofsen E, Kan J, Dahan A, Danhof M. "Pharmacokinetic-
#   pharmacodynamic modeling of the effectiveness and safety of
#   buprenorphine and fentanyl in rats." Pharm Res. 2008;25(1):183-93.
#   doi:10.1007/s11095-007-9440-z, PMID 17914664, PMC2190336, open access,
#   read in full here.
#
# Verbatim, Materials and Methods: "Male Wistar rats (Charles River BV,
# Zeist, The Netherlands), weighting initially 225-250 g, were used in the
# studies"; "indwelling cannulae were implanted in the right femoral artery
# for the serial collection of blood samples and in the right jugular vein
# for drug administration"; "Different doses of buprenorphine (0.025 to
# 0.1 mg/kg) and fentanyl (0.015-0.04 mg/kg) were tested to characterize the
# concentration-antinociceptive effect relationships"; "Buprenorphine and
# fentanyl were administered via a zero-order intravenous infusion (infusion
# duration 20-40 min)"; the tail-flick beam was "set at a level producing
# basal latency times between 2.5 and 3.5 s. To prevent tissue injury, the
# cut-off time was fixed at 10 s"; and the endpoint is BINARY: "If
# antinociceptive effect >= 10 s, then observation score is 1 (a yes
# response; complete antinociception) else 0 (a no response; no
# antinociception)".
#
# Results, verbatim: "For fentanyl, 486 (40 animals) and 216 (32 animals)
# observations were included for pharmacodynamic analysis for the
# antinociceptive and respiratory depressant effect, respectively"; "For
# fentanyl, the values of the parameters alpha and beta were -8.77 (95% CI,
# -11.49 to -6.05) and 1.11 (95% CI, 0.726-1.494), respectively. The odds
# ratio for the antinociceptive effect was 3.03 (95% CI, 1.87-4.21)".
# Table III gives T(1/2,ke0) for fentanyl antinociception as 6.08 min
# (4.43-7.73).
#
# The model is a LINEAR LOGIT in the biophase concentration:
#
#     P(complete antinociception) = 1 / (1 + exp(-(alpha + beta * Ce)))
#
# with Ce in ng/mL, which is what makes it usable without a dose bridge and
# what makes it falsifiable against an occupancy-driven model - see
# `YASSEN2008_STEEPNESS` below and docs/architecture.md's P7.3 section.
#
# One arithmetic inconsistency in the source, checked rather than inherited.
# Its Discussion says "the estimated risk of severe respiratory depression
# increased from 10.5 to 22.9% when the concentration was increased from 4
# to 5 ng/ml. In addition, the probability of achieving complete
# antinociception increased from 0.4 to 1.2%." The respiratory half
# reproduces from the printed alpha = -5.86, beta = 0.930 (10.5% and 23.0%).
# The antinociceptive half does not: alpha = -8.77, beta = 1.11 gives 1.30%
# and 3.84% at 4 and 5 ng/mL, while 0.43% and 1.30% are the values at 3 and
# 4 ng/mL. So that sentence's concentrations are off by one step. The
# Results' parameters are used and the Discussion's sentence is not;
# `tests/test_behavioral_evidence.py` pins both halves of this arithmetic so
# the discrepancy cannot be quietly re-inherited.
#
# What this entry is NOT: the per-dose, per-timepoint tail-flick latencies.
# Those were published in Yassen A, Olofsen E, Dahan A, Danhof M, J
# Pharmacol Exp Ther 2005;313(3):1136-49, doi:10.1124/jpet.104.082560,
# PMID 15701707, which [Yassen2008] cites as the source of its own
# antinociception dataset and which is hard-closed (Unpaywall is_oa false,
# no oa_locations, no PMC, no repository copy; ASPET serves a bot challenge
# to every automated route tried). Its abstract adds one fentanyl number
# this entry does not use, k_eo = 0.123/min (95% CI 0.095-0.151), and one
# structural finding worth recording: "Fentanyl binds instantaneously to the
# OP3 receptor because no reasonable values for k(on) and k(off) were
# obtained with the dynamical receptor model."

#: [Yassen2008]'s antinociception logit, as published. `alpha` is
#: dimensionless, `beta` is per ng/mL of biophase concentration.
YASSEN2008_ALPHA = -8.77
YASSEN2008_BETA_PER_NG_PER_ML = 1.11


def yassen2008_antinociception_probability(concentration_ng_per_ml):
    """[Yassen2008]'s published curve, evaluated. Nothing is fitted here."""
    z = YASSEN2008_ALPHA + YASSEN2008_BETA_PER_NG_PER_ML * np.asarray(
        concentration_ng_per_ml, dtype=float
    )
    return 1.0 / (1.0 + np.exp(-z))


def yassen2008_concentration_at(probability):
    """The biophase concentration (ng/mL) at which the published curve
    reaches `probability`. Used to state the curve's STEEPNESS in the same
    units a receptor-occupancy curve is steep in."""
    p = float(probability)
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return (np.log(p / (1.0 - p)) - YASSEN2008_ALPHA) / YASSEN2008_BETA_PER_NG_PER_ML


#: The ratio C90/C10 of the published curve. A one-site (Hill-1) occupancy
#: curve has C90/C10 = 81 exactly, for every drug and every Kd, because
#: occupancy is C/(C+Kd). [Yassen2008]'s curve is far steeper than that, and
#: the number below is the whole reason the analgesia endpoint comes out
#: UNFIT in P7.3 rather than fitted; see docs/architecture.md.
YASSEN2008_STEEPNESS = (
    yassen2008_concentration_at(0.9) / yassen2008_concentration_at(0.1)
)

#: The C90/C10 of any one-site occupancy curve, for comparison. Exact.
HILL_ONE_STEEPNESS = 81.0

FENTANYL_TAIL_FLICK_YASSEN2008 = BehavioralEvidenceEntry(
    key="fentanyl/tail-flick/Yassen2008",
    drug="fentanyl",
    species="rat",
    strain="Wistar",
    sex="male",
    route="IV",
    doses_mg_per_kg=(0.015, 0.04),
    timepoint_min=None,
    endpoint=(
        "tail-flick latency, scored as complete antinociception (latency at "
        "or above the 10 s cut-off) versus no antinociception, in "
        "conscious rats"
    ),
    effect=PublishedEffect(
        kind="logistic_regression",
        measure=(
            "logistic regression of the probability of complete "
            "antinociception on the biophase concentration Ce: "
            "P = 1/(1+exp(-(alpha + beta*Ce)))"
        ),
        units="alpha dimensionless; beta per ng/mL",
        values={
            "alpha": YASSEN2008_ALPHA,
            "beta_per_ng_per_ml": YASSEN2008_BETA_PER_NG_PER_ML,
            "odds_ratio_per_ng_per_ml": 3.03,
            "t_half_ke0_min": 6.08,
        },
        dispersion={
            # Every interval printed in the source is symmetric about its
            # point estimate to the digits given, so a half-width loses
            # nothing; both endpoints are in the comment above.
            "alpha": 2.72,
            "beta_per_ng_per_ml": 0.384,
            "odds_ratio_per_ng_per_ml": 1.17,
            "t_half_ke0_min": 1.65,
        },
        dispersion_kind="95% CI half-width",
        group_sizes={"alpha": 40, "beta_per_ng_per_ml": 40},
        source_location="Results (Pharmacodynamics) and Table III",
        note=(
            "486 observations in 40 animals. The 95% CIs are bootstrap "
            "(1000 replicates). The odds ratio and T(1/2,ke0) are carried "
            "because they are the two numbers P2.2's record already quoted "
            "from the abstract; the intercept alpha is the piece that "
            "abstract does not contain and that fixes the curve's position "
            "on the concentration axis at all."
        ),
    ),
    citation=(
        "Yassen A, Olofsen E, Kan J, Dahan A, Danhof M. Pharm Res. "
        "2008;25(1):183-93. doi:10.1007/s11095-007-9440-z, PMID 17914664, "
        "PMC2190336"
    ),
    source_access="open access (PMC2190336), read in full",
    body_weight_kg=None,
    provenance_gaps={
        "body_weight_kg": (
            "A band, 225-250 g, not a single weight; it is the same band "
            "drug_pk.FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG already "
            "records, because this is the same paper and the same animals."
        ),
        "timepoint_min": (
            "'Antinociception or respiratory depression was measured at "
            "pre-defined time points' - the times themselves are never "
            "stated. This costs nothing here: the published relationship is "
            "anchored to a CONCENTRATION rather than to a time after a "
            "dose, so no timepoint is needed to use it, which is exactly "
            "why this entry does not need the Phase 1 bridge."
        ),
    },
    note=(
        "The only entry in this module whose target is a concentration-"
        "response CURVE rather than a per-dose contrast, and the only one "
        "for which the Phase 1 bridge is unnecessary rather than "
        "unavailable. Two things nonetheless stop it from being fitted, and "
        "both are measured rather than argued in P7.3: (a) "
        "`drug_pk.brain_partition_for('fentanyl', 'rat')` is total-brain "
        "only - no rat brain unbound fraction exists for fentanyl - so no "
        "receptor-driving concentration can be produced from a plasma one; "
        "and (b) the published curve rises from 10% to 90% over a 1.67-fold "
        "concentration range (`YASSEN2008_STEEPNESS`) where a one-site "
        "occupancy curve needs 81-fold, so no monotone function of "
        "fractional occupancy can reproduce it. The dosing is also a "
        "20-40 min zero-order infusion, which `drug_pk`'s IV model does not "
        "represent; that matters for a dose-anchored target and not for "
        "this one."
    ),
)


# ---------------------------------------------------------------------------
# 6. Mu-opioid: fentanyl -> righting reflex (P7.3: sedation)
# ---------------------------------------------------------------------------
#
# Sedation is the ONE endpoint of the seven for which a real intravenous
# dose-response exists in the rat, and there are two of them. Both are
# recorded; neither is fitted, for a reason that is structural rather than
# about data quality and is worked through in docs/architecture.md: their
# measure is the TIME AT WHICH A THRESHOLD IS CROSSED, and a
# threshold-crossing time is invariant under every strictly monotone map of
# the quantity being thresholded. Concentration -> occupancy -> regional
# rates -> endpoint index is such a map, so "when does the rat right
# itself" is determined by the pharmacokinetics and by ONE number - the
# concentration at which the threshold sits - with the circuit layer and
# the endpoint's regional weights cancelling out entirely. These datasets
# calibrate `drug_pk`, not `endpoints`.
#
# [Obert2025]   Obert DP, Park GH, Strong K, Schreier DR, Korn E, Troyas C,
#               Vincent KF, Solt K. "Repeated High-dose Fentanyl
#               Administration in Rats Reveals Minimal Tolerance to
#               Unconsciousness, Bradycardia, Muscle Rigidity, and
#               Respiratory Depression." Anesthesiology. 2025;142(3):465-75.
#               doi:10.1097/ALN.0000000000005324, PMID 39705671, PMC11813677
#               (NIH author manuscript), read in full here.
#               Verbatim: "Male (n=20) and female (n=20) adult
#               Sprague-Dawley rats weighing 250-350g at the start of the
#               experiment were included in this study"; "All rats were
#               briefly anesthetized with isoflurane for 5-10min to place a
#               24-gauge intravenous catheter in the lateral tail vein to
#               allow for intravenous (IV) injection of fentanyl. The
#               animals had at least 45min to fully recover from isoflurane
#               anesthesia before fentanyl was administered"; "We
#               administered 12.5, 25, 50, and 100 microg/kg body weight of
#               fentanyl in random order and evaluated the time to return of
#               righting reflex (RORR) using video recordings"; "The median
#               latency to RORR was 13[10-14]min for 12.5microg/kg,
#               15[12-17]min for 25microg/kg, 26[24-29]min for 50microg/kg,
#               and 50[45-53]min for 100microg/kg"; "Data is presented as
#               median [1st quartile-3rd quartile]"; the dose-finding
#               subgroup is "2 male and 2 female rats".
#               Also verbatim, and the reason the top dose is the top dose:
#               "Thirty-eight rats received 100microg/kg of fentanyl IV as
#               the first dose and three of them died. Therefore, in a naive
#               rat, the LD10 was approximately 100microg/kg."
#
# The dispersion is an interquartile range on n = 4, which `DISPERSION_KINDS`
# does not contain, and it is not converted into an SEM or an SD: doing so
# needs a distributional assumption the source does not license, and the
# module's standing rule is that dispersions are recorded, never converted.
# The quartiles are therefore carried in `note` and the effect is stored
# with `dispersion=None`, which is the honest reading of "this source
# reports a spread this module cannot express".

#: The four IV doses [Obert2025] gave, mg/kg, and the median time to return
#: of the righting reflex at each, minutes. Stored as a mapping so that the
#: pairing cannot be broken by re-ordering one of the two lists.
OBERT2025_RORR_MIN_BY_DOSE_MG_PER_KG = {
    0.0125: 13.0,
    0.025: 15.0,
    0.050: 26.0,
    0.100: 50.0,
}

#: The published interquartile ranges, same keys, minutes. Recorded, not used
#: as a likelihood width; see the section comment.
OBERT2025_RORR_IQR_MIN_BY_DOSE_MG_PER_KG = {
    0.0125: (10.0, 14.0),
    0.025: (12.0, 17.0),
    0.050: (24.0, 29.0),
    0.100: (45.0, 53.0),
}

FENTANYL_RIGHTING_REFLEX_OBERT2025 = BehavioralEvidenceEntry(
    key="fentanyl/righting-reflex/Obert2025",
    drug="fentanyl",
    species="rat",
    strain="Sprague-Dawley",
    sex="male and female",
    route="IV",
    doses_mg_per_kg=tuple(sorted(OBERT2025_RORR_MIN_BY_DOSE_MG_PER_KG)),
    timepoint_min=None,
    endpoint=(
        "time to return of the righting reflex after an IV fentanyl bolus, "
        "i.e. the duration of fentanyl-induced unconsciousness"
    ),
    effect=PublishedEffect(
        kind="group_means",
        measure="median latency to return of righting reflex",
        units="min",
        values={
            "%g mg/kg" % dose: value
            for dose, value in OBERT2025_RORR_MIN_BY_DOSE_MG_PER_KG.items()
        },
        group_sizes={
            "%g mg/kg" % dose: 4 for dose in OBERT2025_RORR_MIN_BY_DOSE_MG_PER_KG
        },
        source_location="Results, 'Dose-response Relationship for Fentanyl-induced LORR'",
        note=(
            "Medians, not means, and the published spread is an "
            "interquartile range (see "
            "OBERT2025_RORR_IQR_MIN_BY_DOSE_MG_PER_KG) which "
            "DISPERSION_KINDS deliberately cannot express, so no dispersion "
            "is stored rather than an IQR being converted into an SD. The "
            "paper has no data tables; these four numbers are in its "
            "Results prose."
        ),
    ),
    citation=(
        "Obert DP, Park GH, Strong K, Schreier DR, Korn E, Troyas C, "
        "Vincent KF, Solt K. Anesthesiology. 2025;142(3):465-75. "
        "doi:10.1097/ALN.0000000000005324, PMID 39705671, PMC11813677"
    ),
    source_access="subscription; NIH author manuscript in PMC11813677, read in full",
    provenance_gaps={
        "body_weight_kg": (
            "A band, 250-350 g at the start of the study, not a per-animal "
            "weight. The rat IV fentanyl parameter set is in ABSOLUTE "
            "volumes and clearances ([Yassen2008]), so unlike the diazepam "
            "entries the weight does NOT cancel here, and a caller has to "
            "choose one and say so."
        ),
        "timepoint_min": (
            "There is no single measurement time: the measurement IS a "
            "time. This entry's readout is a duration, which is what makes "
            "it circuit-independent - see the section comment."
        ),
    },
    note=(
        "The only four-level intravenous behavioural dose-response found "
        "for any of the seven endpoints, in any drug this codebase models. "
        "It is recorded and not fitted, and the reason is worth being "
        "precise about because it is not 'the data are too weak': a "
        "return-of-righting-reflex LATENCY is the time at which the "
        "sedation index falls back through a threshold, and every monotone "
        "stage between plasma concentration and that index - occupancy, "
        "the Wilson-Cowan steady state, the linear readout - preserves the "
        "crossing time exactly. Four latencies therefore constrain one "
        "number (the threshold concentration) and say nothing at all about "
        "the endpoint's regional weights. Two further caveats: the "
        "dose-finding subgroup is 4 animals, and the top dose is its own "
        "LD10."
    ),
)


# ---------------------------------------------------------------------------
# 7. Mu-opioid: fentanyl -> loss of the righting reflex, quantal (sedation)
# ---------------------------------------------------------------------------
#
# [Kissin1983]  Kissin I, Kerr CR, Smith LR. "Assessment of anaesthetic
#               action of morphine and fentanyl in rats." Can Anaesth Soc J.
#               1983;30(6):623-8. doi:10.1007/BF03015233, PMID 6640399.
#               Unpaywall reports a free published-version PDF; it was
#               obtained and read by OCR, and THAT IS RECORDED ON THE ENTRY
#               rather than glossed, because an OCR misread is a wrong
#               number and this module's whole discipline is about where a
#               number came from. The OCR of the Methods is legible enough
#               to quote: "The experiments were done on 150 male
#               Sprague-Dawley rats weighing 225-250 g"; "Morphine sulphate
#               (Merck) and fentanyl citrate (Janssen) were injected into
#               the saphenous vein over 10 sec"; "In each series of
#               experiments, five doses were administered, each in a group
#               of five rats"; "Loss of righting reflex. The test was
#               regarded as positive if the rat failed to right itself (with
#               all four feet on the table) within 15 s after being placed
#               in a side position"; "The end points were determined five
#               minutes after injection of fentanyl".
#               Table I, fentanyl, righting-reflex row, as OCR'd:
#               ED50 0.023 mg/kg IV, 95% fiducial limits 0.020-0.026, N=25.
#
# The per-dose quantal incidences are figure-only ("Each dot represents the
# incidence of effect in a group of five animals at the indicated dosage"),
# so what is storable is the ED50 - and, unlike [Gao2021]'s graded ED50, a
# QUANTAL ED50 is a complete statement: at 0.023 mg/kg, half the rats lose
# the righting reflex, five minutes after an IV bolus. That is a
# (dose, probability) pair with a published interval, which is a usable
# target shape, and it is still not fitted here for the same
# monotone-invariance reason as [Obert2025].
#
# One internal inconsistency, recorded rather than resolved: the paper's own
# Results say the righting-reflex-to-purposeful-movement ED50 ratio for
# fentanyl is 2.6, its Table II says 2.4, and its Table I's own two ED50s
# give 0.023/0.009 = 2.56. The tabulated ED50s and the Results text agree
# with each other and Table II does not.

KISSIN1983_FENTANYL_RR_ED50_MG_PER_KG = 0.023
KISSIN1983_FENTANYL_RR_ED50_FIDUCIAL_LIMITS_MG_PER_KG = (0.020, 0.026)

FENTANYL_RIGHTING_REFLEX_KISSIN1983 = BehavioralEvidenceEntry(
    key="fentanyl/righting-reflex-ED50/Kissin1983",
    drug="fentanyl",
    species="rat",
    strain="Sprague-Dawley",
    sex="male",
    route="IV",
    doses_mg_per_kg=(KISSIN1983_FENTANYL_RR_ED50_MG_PER_KG,),
    timepoint_min=5.0,
    endpoint=(
        "loss of the righting reflex, scored 5 min after an IV fentanyl "
        "bolus; quantal (all-or-none) endpoint"
    ),
    effect=PublishedEffect(
        kind="ed50",
        measure=(
            "ED50 for loss of the righting reflex, probit fit over five "
            "doses of five rats each"
        ),
        units="mg/kg",
        values={"fentanyl": KISSIN1983_FENTANYL_RR_ED50_MG_PER_KG},
        dispersion={"fentanyl": 0.003},
        dispersion_kind="95% CI half-width",
        group_sizes={"fentanyl": 25},
        source_location="Table I, fentanyl row, RR endpoint",
        note=(
            "The published interval is 95% FIDUCIAL limits (0.020-0.026), "
            "which is what a probit fit produces and is not the same object "
            "as a confidence interval; it is stored under the nearest "
            "available kind and named here so the distinction is not lost. "
            "The limits are symmetric about the estimate to the digits "
            "given, so the half-width loses nothing. Unlike [Gao2021]'s "
            "graded ED50, this one is quantal, so it fixes a probability "
            "(0.5) at a dose rather than a fraction of an unknown maximum."
        ),
    ),
    body_weight_kg=None,
    citation=(
        "Kissin I, Kerr CR, Smith LR. Can Anaesth Soc J. 1983;30(6):623-8. "
        "doi:10.1007/BF03015233, PMID 6640399"
    ),
    source_access=(
        "free publisher PDF per Unpaywall; obtained and read by OCR, whose "
        "output mangles the units column and several decimal points. The "
        "two numbers used here (ED50 and its limits) are the legible ones; "
        "no per-dose value is taken from this source."
    ),
    provenance_gaps={
        "body_weight_kg": (
            "A band, 225-250 g - the same band [Yassen2008] used, which is "
            "a coincidence worth noticing and not a shared source."
        ),
    },
    note=(
        "The second of the two intravenous sedation dose-responses, and the "
        "older one. Recorded, not fitted: see [Obert2025]'s note for the "
        "monotone-invariance argument, which applies to a quantal threshold "
        "even more directly than to a latency."
    ),
)


# ---------------------------------------------------------------------------
# 8. GABA-A: diazepam -> locomotor activity, THREE DOSES (P7.3: locomotion)
# ---------------------------------------------------------------------------
#
# The dataset P7.3 was looking for and P2.1/P2.2 never had: a graded
# dose-response, in the rat, by a route `drug_pk` models, printed in a table
# with a dispersion and a group size. It is a NULL result, and that is what
# makes it usable rather than what makes it weak - see the entry's note.
#
# [Sagara2008]  Sagara H, Kitamura Y, Sendo T, Araki H, Gomita Y. "Effect of
#               diazepam on the runway method using priming stimulation of
#               intracranial self stimulation behavior." J Pharmacol Sci.
#               2008;107(3):355-60. doi:10.1254/jphs.08008SC, PMID 18603833,
#               open access (J-STAGE), read in full here.
#               Verbatim: "Male Wistar rats (Charles River, Yokohama),
#               weighing 250-300 g at the time of surgery"; "Groups of rats
#               received 0.2, 0.5, or 1 mg/kg diazepam or vehicle (40%
#               propylene glycol)"; "Diazepam, flumazenil, or vehicle was
#               administered intraperitoneally (i.p.) at 0.1 ml per 100 g of
#               body weight"; "Rats were administered diazepam (0.2, 0.5,
#               and 1.0 mg/kg, i.p.) 15 min before the test"; "Each data
#               represents the mean +/- S.E.M locomotor activity for 10 min
#               (n = 10)".
#               Table 1, ANIMEX locomotor counts over 10 min:
#                   vehicle          153.6 +/- 16.6
#                   diazepam 0.2     156.5 +/- 22.9
#                   diazepam 0.5     158.3 +/- 24.6
#                   diazepam 1.0     161.7 +/- 19.1
#
# Two things about these animals are on the entry rather than in a footnote:
# they carry chronically implanted intracranial self-stimulation electrodes
# (the locomotor measurement is a control experiment inside an ICSS study),
# and the vehicle is 40% propylene glycol rather than saline.

#: Duration of [Sagara2008]'s locomotor measurement, minutes. The counts are
#: per this window, so a ratio between groups needs no duration at all; it
#: is recorded because a count without its window is not a measurement.
SAGARA2008_ACTIVITY_WINDOW_MIN = 10.0

#: Interval between injection and test, minutes, stated for diazepam by name.
SAGARA2008_PRETREATMENT_MIN = 15.0

#: Body-weight band, kg. The rat IP diazepam model is per kg, so this does
#: not propagate into a concentration.
SAGARA2008_BODY_WEIGHT_BAND_KG = (0.250, 0.300)

DIAZEPAM_LOCOMOTION_SAGARA2008 = BehavioralEvidenceEntry(
    key="diazepam/locomotor-activity/Sagara2008",
    drug="diazepam",
    species="rat",
    strain="Wistar",
    sex="male",
    route="IP",
    doses_mg_per_kg=(0.2, 0.5, 1.0),
    timepoint_min=SAGARA2008_PRETREATMENT_MIN,
    endpoint=(
        "spontaneous locomotor activity, ANIMEX counts over a 10 min "
        "session, 15 min after IP dosing; vehicle and three diazepam doses"
    ),
    effect=PublishedEffect(
        kind="group_means",
        measure="locomotor activity counts in 10 min (ANIMEX)",
        units="counts",
        values={
            "vehicle": 153.6,
            "diazepam 0.2 mg/kg": 156.5,
            "diazepam 0.5 mg/kg": 158.3,
            "diazepam 1.0 mg/kg": 161.7,
        },
        dispersion={
            "vehicle": 16.6,
            "diazepam 0.2 mg/kg": 22.9,
            "diazepam 0.5 mg/kg": 24.6,
            "diazepam 1.0 mg/kg": 19.1,
        },
        dispersion_kind="SEM",
        group_sizes={
            "vehicle": 10,
            "diazepam 0.2 mg/kg": 10,
            "diazepam 0.5 mg/kg": 10,
            "diazepam 1.0 mg/kg": 10,
        },
        source_location="Table 1",
        note=(
            "A null: no dose differs from vehicle, and all four means lie "
            "within about 5% of each other. The paper's own conclusion for "
            "this table is that diazepam's effect on its ICSS runway measure "
            "is not a locomotor artefact."
        ),
    ),
    citation=(
        "Sagara H, Kitamura Y, Sendo T, Araki H, Gomita Y. J Pharmacol Sci. "
        "2008;107(3):355-60. doi:10.1254/jphs.08008SC, PMID 18603833"
    ),
    source_access="open access (J-STAGE), read in full",
    provenance_gaps={
        "body_weight_kg": (
            "A band, 250-300 g at surgery; see "
            "SAGARA2008_BODY_WEIGHT_BAND_KG. The rat IP diazepam model is "
            "per kg, so the band does not propagate into a concentration."
        ),
    },
    note=(
        "The only graded, tabulated, rat, bridgeable-route behavioural "
        "dose-response found for any of the seven endpoints in a drug this "
        "codebase models. It is a NULL, and a null with error bars is the "
        "kind of datum that CAN bound a model predicting a monotone "
        "decrease: potentiate inhibition brain-wide and the locomotion "
        "index falls, so three doses that move nothing would bound the gain "
        "from above.\n"
        "It does not, and the fit says so rather than this note assuming "
        "it. `results/calibration/endpoint_fits.txt`: at the largest gain "
        "the prior allows, the whole-brain locomotion index falls by 4.7 to "
        "6.4% across the three doses, against published SEM-derived widths "
        "of 17 to 20% on the same ratios. The posterior is 2.37 +/- 1.43 on "
        "a Uniform(0, 5) prior whose own mean is 2.5 and sd 1.44 - the "
        "prior, to two digits - and `bound_verdict` returns wrong-direction "
        "(the counts rose slightly). What defeats it is not the dataset but "
        "the operating point: P3.4's illustrative rho = 0.9 makes every "
        "inhibitory weight ~50x smaller than the published density, so the "
        "index the drug is being asked to move barely moves at all. The "
        "same cause as P7.3's refutation 2, measured on the one graded "
        "series in the module.\n"
        "Two caveats travel with it. The animals carry chronically "
        "implanted ICSS electrodes and had received priming stimulation in "
        "the same study, so they are not naive; and the vehicle is 40% "
        "propylene glycol, which is not inert. Neither is a reason to "
        "discard the table, and both are reasons not to read a small "
        "difference in it as a drug effect."
    ),
)


# ---------------------------------------------------------------------------
# 9. GABA-A: diazepam -> hot plate, TWO DOSES (P7.3: analgesia)
# ---------------------------------------------------------------------------
#
# [Lustyk2023]  Lustyk K, Salaciak K, Jakubczyk M, Jastrzebska-Wiesek M,
#               Partyka A, Wesolowska A, Marona H, Pytka K. "HBK-15, a
#               Multimodal Compound, Showed an Anxiolytic-Like Effect in
#               Rats." Neurochem Res. 2023;48(3):839-51.
#               doi:10.1007/s11064-022-03802-x, PMID 36350432, PMC9644393,
#               open access, read in full here.
#               Verbatim: "In all experiments, we used male Wistar rats
#               (200-220 g, in total 140 animals)"; "The studied compound or
#               diazepam (Sigma, Germany) was dissolved in saline and
#               administered intraperitoneally (ip) in a 1 ml/kg volume";
#               "HBK-15, diazepam, or vehicle (saline) were injected
#               intraperitoneally (ip) 30 min before the test. The results
#               are presented as means +/- SD ... n = 6-9 rats per group";
#               hot plate "(52.5 +/- 0.5 C, 19-cm diameter)".
#               Table 2, hot plate, time of reaction [s]:
#                   vehicle           8.20 +/- 2.38
#                   diazepam 5 mg/kg 11.40 +/- 4.76
#                   diazepam 10      10.06 +/- 3.70
#               and the paper's own test on those three: "F (2,20) = 1.409,
#               ns".
#
# The dispersion is an SD, not an SEM, and this module does not convert
# between them. What a ratio's likelihood width needs is the sd OF THE MEAN,
# so the caller divides by sqrt(n) at the point of use and says which n it
# used - the source gives a band (6-9) and not a per-group number, so the
# defensible choice is the SMALLEST, which gives the WIDEST width and
# therefore the weakest constraint. `LUSTYK2023_MIN_GROUP_SIZE` is that
# number and P7.3 uses it.

LUSTYK2023_MIN_GROUP_SIZE = 6
LUSTYK2023_PRETREATMENT_MIN = 30.0
LUSTYK2023_BODY_WEIGHT_BAND_KG = (0.200, 0.220)

DIAZEPAM_HOT_PLATE_LUSTYK2023 = BehavioralEvidenceEntry(
    key="diazepam/hot-plate/Lustyk2023",
    drug="diazepam",
    species="rat",
    strain="Wistar",
    sex="male",
    route="IP",
    doses_mg_per_kg=(5.0, 10.0),
    timepoint_min=LUSTYK2023_PRETREATMENT_MIN,
    endpoint=(
        "hot-plate latency at 52.5 C, 30 min after IP dosing; vehicle and "
        "two diazepam doses"
    ),
    effect=PublishedEffect(
        kind="group_means",
        measure="hot-plate time of reaction",
        units="s",
        values={
            "vehicle": 8.20,
            "diazepam 5 mg/kg": 11.40,
            "diazepam 10 mg/kg": 10.06,
        },
        dispersion={
            "vehicle": 2.38,
            "diazepam 5 mg/kg": 4.76,
            "diazepam 10 mg/kg": 3.70,
        },
        dispersion_kind="SD",
        source_location="Table 2, hot plate",
        note=(
            "SD, not SEM - the only entry in this module reported that way, "
            "and the reason `DISPERSION_KINDS` records the kind instead of "
            "normalising it. The source gives n as a band (6-9) rather than "
            "per group, so no group_sizes are stored; see "
            "LUSTYK2023_MIN_GROUP_SIZE for what a caller should use and "
            "why. The paper's own analysis of these three groups is ns."
        ),
    ),
    citation=(
        "Lustyk K, Salaciak K, Jakubczyk M, Jastrzebska-Wiesek M, Partyka "
        "A, Wesolowska A, Marona H, Pytka K. Neurochem Res. "
        "2023;48(3):839-51. doi:10.1007/s11064-022-03802-x, PMID 36350432, "
        "PMC9644393"
    ),
    source_access="open access (PMC9644393), read in full",
    provenance_gaps={
        "body_weight_kg": (
            "A band, 200-220 g; the rat IP diazepam model is per kg, so it "
            "does not propagate into a concentration."
        ),
    },
    note=(
        "P7.3's analgesia target, and the endpoint's escape from the "
        "fentanyl brain-partition blocker: diazepam/rat/IP is the one "
        "drug/species/route triple in this codebase with a "
        "RECEPTOR-DRIVABLE brain partition, so a hot-plate dose-response in "
        "it can be pushed all the way to an occupancy where a fentanyl one "
        "cannot. Two doses and a vehicle, so it is a graded series rather "
        "than a contrast, and like [Sagara2008] it is a null - which again "
        "bounds the gain from above rather than leaving it unconstrained.\n"
        "It is the HOT PLATE and not the tail flick, so it belongs to "
        "`endpoints.ASSAY_VARIANTS['analgesia_hot_plate']` as much as to "
        "the union endpoint; P7.1 recorded two papers that dissociate the "
        "two assays, and P7.3 reports the fit under both readouts rather "
        "than choosing one."
    ),
)


#: Every entry, in the order the Track E section discusses them.
BEHAVIORAL_EVIDENCE = (
    FENTANYL_NAC_DOPAMINE_YOSHIDA1999,
    HALOPERIDOL_HYPERLOCOMOTION_GAO2021,
    DIAZEPAM_EPM_PELLOW1985,
    DIAZEPAM_EPM_STICKLING2024,
    DIAZEPAM_EPM_GRAZIANI2022,
    DIAZEPAM_OPEN_FIELD_GRAZIANI2022,
    FENTANYL_TAIL_FLICK_YASSEN2008,
    FENTANYL_RIGHTING_REFLEX_OBERT2025,
    FENTANYL_RIGHTING_REFLEX_KISSIN1983,
    DIAZEPAM_LOCOMOTION_SAGARA2008,
    DIAZEPAM_HOT_PLATE_LUSTYK2023,
)

#: Keyed access, for tests and for P2.2.
BEHAVIORAL_EVIDENCE_BY_KEY = {e.key: e for e in BEHAVIORAL_EVIDENCE}


#: P7.3's acceptance criterion, as a computable object: every one of the
#: seven P7.1 endpoints, with the evidence found for it and the verdict on
#: whether that evidence can calibrate it. `endpoint_coverage()` below
#: checks this against `endpoints.ENDPOINTS` so an endpoint cannot be
#: silently omitted, and against `BEHAVIORAL_EVIDENCE_BY_KEY` so a record
#: cannot name an entry that does not exist.
#:
#: Verdicts:
#:   "fitted"          a gain was identified and its label removed.
#:   "fit-attempted"   a dataset exists, the fit runs and converges, and
#:                     the posterior does not license a number. The reason
#:                     is on the record and the diagnostics are in
#:                     results/calibration/.
#:   "recorded-not-fitted"  a dataset exists and is structurally unable to
#:                     calibrate this layer, for a reason that is not about
#:                     data quality.
#:   "no-data"         nothing admissible was found.
ENDPOINT_COVERAGE_VERDICTS = (
    "fitted",
    "fit-attempted",
    "recorded-not-fitted",
    "no-data",
)

ENDPOINT_COVERAGE = {
    "locomotion": {
        "verdict": "fit-attempted",
        "entries": ("diazepam/locomotor-activity/Sagara2008",
                    "diazepam/open-field/Graziani2022"),
        "record": (
            "The endpoint with the only GRADED bridgeable series in the "
            "task. [Sagara2008] gives three diazepam IP doses (0.2, 0.5, "
            "1.0 mg/kg) and a vehicle, with SEMs and n = 10 per group, in a "
            "table, 15 min after dosing - and finds nothing: 153.6+/-16.6 "
            "counts on vehicle against 156.5, 158.3 and 161.7. A null with "
            "error bars is the kind of datum that CAN bound a model "
            "predicting a monotone decrease, and this one does not: the "
            "whole-brain index moves 4.7-6.4% at the largest gain the prior "
            "allows against widths of 17-20% on the observed ratios, and "
            "the posterior comes back as its own prior (2.37 +/- 1.43 on "
            "Uniform(0, 5)). No target in P7.3 identifies anything; the "
            "graded series fails for a different reason from the contrasts, "
            "and the reason is the illustrative operating point rather than "
            "the data.\n"
            "DIAZEPAM, rat, IP, second dataset. [Graziani2022]'s Table 3 prints open-field "
            "crossing episodes for vehicle (31.1+/-2.6, n=11) and diazepam "
            "2 mg/kg (43.0+/-6.1, n=4). It is a two-point contrast, not a "
            "dose-response curve, so it cannot separate a circuit gain from "
            "the unknown behavioural scale; under P2.2's ratio formulation "
            "the fit runs and comes back WRONG-DIRECTION - the rats crossed "
            "38% MORE lines under diazepam, and a whole-brain potentiation "
            "of inhibition can only lower the locomotion index. The "
            "posterior therefore sits on the mechanistic null at gain 0.\n"
            "HALOPERIDOL, the drug whose locomotor pharmacology this "
            "codebase's D2 scenario is built on, remains unreachable, and "
            "P7.3 re-ran that search with two corrections to this "
            "repository's own record; see "
            "SEARCHED_AND_EMPTY_BEHAVIORAL[('haloperidol', 'rat', 'IV', "
            "'locomotor activity')].\n"
            "FENTANYL, rat, IV: locomotion is measured in the Kiyatkin "
            "lab's IV fentanyl series (Choi S, Noya MR, Kiyatkin EA, J "
            "Neurophysiol 2024;132(2):322-334, doi:10.1152/jn.00177.2024, "
            "PMID 38863429, PMC11427041, is a five-dose IV series - 1, 3, "
            "10, 30, 90 ug/kg - in 15 male Long-Evans rats) and its "
            "locomotor result is stated only in words: 'Fentanyl at each "
            "dose decreased locomotor activity; hypoactivity was dose "
            "dependent'. The values are in Fig. 4D and the paper's only "
            "tables are statistical. Every rat fentanyl study that "
            "TABULATES locomotion doses subcutaneously (e.g. Gaulden AD, et "
            "al., Drug Alcohol Depend 2021;229:109101, PMID 34628096: 'All "
            "injections were given subcutaneously')."
        ),
    },
    "anxiety": {
        "verdict": "fit-attempted",
        "entries": ("diazepam/elevated-plus-maze/Graziani2022",
                    "diazepam/elevated-plus-maze/Stickling2024",
                    "diazepam/elevated-plus-maze/Pellow1985"),
        "record": (
            "The best-supplied endpoint of the seven, and the one P2.2 "
            "already spent: [Graziani2022] is rat, IP, drug-naive, with the "
            "assay's own primary measure, SEMs and group sizes. P7.3 re-runs "
            "it through the WHOLE-BRAIN endpoint rather than the three-node "
            "amygdala scenario, and the two do not agree: the scenario "
            "reproduces the published 22% anxiolysis at a gain of 0.074, "
            "while the whole-brain readout moves 4% at the largest gain the "
            "mechanism allows, so its posterior piles against the upper "
            "prior bound. That is a statement about the whole-brain "
            "operating point (P3.4's illustrative rho = 0.9 scales every "
            "weight by 0.020, which is what makes a synaptic-modulation drug "
            "nearly inert), not about the data."
        ),
    },
    "analgesia": {
        "verdict": "fit-attempted",
        "entries": ("diazepam/hot-plate/Lustyk2023",
                    "fentanyl/tail-flick/Yassen2008"),
        "record": (
            "The fit that IS attempted is [Lustyk2023]: two diazepam IP "
            "doses (5 and 10 mg/kg) and a vehicle against 52.5 C hot-plate "
            "latency in male Wistar rats, 30 min after dosing, means and "
            "SDs in a table. Diazepam is the one drug whose rat brain "
            "partition can drive a receptor, so this series reaches an "
            "occupancy where every fentanyl series below cannot. The "
            "paper's own ANOVA is ns (F(2,20) = 1.409), which is the kind "
            "of result that could bound the gain from above, and the fit "
            "reports something weaker than that: the ratios of the group "
            "means are 1.3902 +/- 0.2886 and 1.2268 +/- 0.2347, i.e. ABOVE "
            "one, while a whole-brain potentiation of inhibition can only "
            "put the analgesia index below one (0.9718 and 0.9714, a 2.8% "
            "move, at the largest gain the prior allows). `bound_verdict` "
            "is therefore wrong-direction, at a worst residual of 1.35 sd, "
            "and the posterior (2.30 +/- 1.45 on Uniform(0, 5), whose mean "
            "is 2.5 and sd 1.44) is its own prior. Neither graded series "
            "identifies anything, and what defeats both is the model's "
            "reach at P3.4's illustrative operating point rather than the "
            "datasets; see results/calibration/endpoint_fits.txt.\n"
            "The rest of this record is about FENTANYL, which is where the "
            "published data are and where the structural blocker is. "
            "[Yassen2008] - already this codebase's "
            "fentanyl rat IV PK source - publishes the concentration-"
            "response as a logistic regression, so it needs no dose bridge "
            "at all. Two measured facts stop it being fitted:\n"
            "(1) `drug_pk.brain_partition_status('fentanyl', 'rat')` is "
            "'total-brain-only'. No rat brain unbound fraction for fentanyl "
            "was found (P1.5's searched-and-empty), so the Kp of ~16.5 "
            "cannot become a receptor-driving concentration, and "
            "`receptor_drive_status()` says so for all three fentanyl "
            "entries here.\n"
            "(2) Even taking plasma as the driving concentration, the "
            "published curve rises from 10% to 90% of its range over a "
            "1.67-fold concentration span (`YASSEN2008_STEEPNESS`), where a "
            "one-site occupancy curve needs 81-fold "
            "(`HILL_ONE_STEEPNESS`). No monotone function of fractional "
            "occupancy is that steep, so a linear readout over occupancy-"
            "driven rates cannot reproduce this dataset at any gain. The "
            "steepness is partly an artefact of a QUANTAL endpoint - a "
            "probability of crossing a 10 s cut-off is steepened by "
            "between-animal variance - and modelling that needs a threshold "
            "and a spread, i.e. two more free parameters against a curve "
            "that publishes two.\n"
            "Four further rat IV fentanyl antinociception datasets were "
            "found, all ED50s with 95% intervals in tables, and they are "
            "recorded here rather than entered separately because each "
            "faces the same two blockers and would add no independent "
            "constraint: Crowley ML, et al., J Pharmacol Toxicol Methods "
            "2021;111:107101, doi:10.1016/j.vascn.2021.107101, PMID "
            "34242797 (PMC8494486), 52 C hot plate, cumulative IV dosing, "
            "'The ED50 (95% CI) value of fentanyl to produce antinociception "
            "was 0.0196 (0.0162-0.0229) mg/kg', six rats per point; "
            "Townsend EA, et al., ACS Chem Neurosci 2020;11(9):1300-1310, "
            "doi:10.1021/acschemneuro.0c00064, PMID 32271538 (PMC7531604), "
            "50 C tail withdrawal 120 s after IV, ED50 5.3 ug/kg (4.1-6.5); "
            "Schwienteck KL, et al., Drug Alcohol Depend 2019;204:107501, "
            "doi:10.1016/j.drugalcdep.2019.06.006, PMID 31479865 "
            "(PMC6878171), same assay 1 min after IV, ED50 0.014 mg/kg "
            "(0.011-0.017); Zamarripa CA, et al., Psychopharmacology "
            "2024;241(12):2453-2469, doi:10.1007/s00213-024-06690-x, PMID "
            "39333403 (PMC13273602), 52.5 C hot plate 20 min after IV, ED50 "
            "0.0073 mg/kg (0.004-0.01). They agree within about five-fold "
            "and differ mainly in post-injection sampling time, which is "
            "the biophase equilibration [Yassen2008] models with its "
            "T(1/2,ke0) of 6.08 min.\n"
            "The one dataset that would be a genuine multi-dose IV bolus "
            "tail-flick curve is Getsy PM, et al., Front Pharmacol "
            "2022;13:892307, doi:10.3389/fphar.2022.892307, PMID 35721204 "
            "(PMC9199495), whose fentanyl-alone arm received 10, 25 and 50 "
            "ug/kg IV 30 min apart in six rats - and whose per-dose "
            "tail-flick latencies are in Supplementary Tables S6 and S7, "
            "which neither Frontiers nor NCBI would serve to any automated "
            "route tried. Ossipov MH, et al., J Pharmacol Exp Ther "
            "1990;255(3):1107-1116, PMID 2262895 and Anesthesiology "
            "1990;73(6):1227-1235, PMID 1978990 are the two other "
            "high-prior IV-fentanyl tail-flick-and-hot-plate papers, and "
            "both are Cloudflare-blocked."
        ),
    },
    "reward": {
        "verdict": "no-data",
        "entries": (),
        "record": (
            "No rat conditioned place preference and no rat intracranial "
            "self-stimulation study administers fentanyl INTRAVENOUSLY, "
            "which is the only fentanyl route `drug_pk` models. Every rat "
            "fentanyl CPP found is subcutaneous (e.g. Psychopharmacology "
            "1988, PMID 3149777, 'fentanyl (0.04 mg/kg SC)'; [Gaulden2021] "
            "at 4 and 16 ug/kg SC), and every rat fentanyl ICSS dose-effect "
            "is subcutaneous too - Moerke MJ, Negus SS, Neuropharmacology "
            "2019;151:180-188, doi:10.1016/j.neuropharm.2019.03.020, PMID "
            "30880123 (PMC6500752) tabulates a full fentanyl ICSS "
            "dose-effect with 95% CIs and states 'All drugs were injected "
            "subcutaneously in a volume of 1 ml/kg'. PubMed returns ZERO "
            "records for fentanyl AND 'conditioned place preference' AND "
            "rat AND intravenous.\n"
            "IV fentanyl SELF-ADMINISTRATION is abundant and is a different "
            "measurement: it reports intake, not preference, and P7.1's "
            "reward endpoint is a place-preference readout. The one "
            "tabulated IV dose-effect, Ren M, Lotfipour S, Microorganisms "
            "2022;10(6):1127, doi:10.3390/microorganisms10061127, PMID "
            "35744645 (PMC9229572), Table 1, prints infusion counts at 0, "
            "1.25 and 2.5 ug/kg/infusion with NO error bar of any kind and "
            "pools the sexes.\n"
            "Diazepam and haloperidol add nothing: neither has a rat CPP "
            "dose-response by a bridgeable route that prints numbers "
            "outside a figure.\n"
            "What would close it: a rat IV fentanyl CPP or ICSS "
            "dose-effect, which does not currently exist and would need a "
            "de novo study; or the per-dose means behind [Ren2022]'s Table "
            "1 with a dispersion."
        ),
    },
    "sedation": {
        "verdict": "fit-attempted",
        "entries": ("diazepam/open-field/Graziani2022",
                    "fentanyl/righting-reflex/Obert2025",
                    "fentanyl/righting-reflex-ED50/Kissin1983"),
        "record": (
            "The one endpoint with a real intravenous dose-response curve, "
            "twice over, and the one where having it changes nothing. "
            "[Obert2025] gives four IV fentanyl doses against the time to "
            "return of the righting reflex; [Kissin1983] gives a quantal "
            "ED50 for losing it. Both measure a THRESHOLD-CROSSING TIME or "
            "DOSE, and every stage between plasma concentration and the "
            "endpoint index is monotone, so the crossing is preserved "
            "exactly and the data constrain one number - where the "
            "threshold sits on the concentration axis - while saying "
            "nothing about the endpoint's regional weights. They are "
            "recorded as evidence about `drug_pk` and not fitted as "
            "evidence about `endpoints`. Both also inherit the fentanyl "
            "brain-partition blocker in (1) of the analgesia record.\n"
            "The fit that IS attempted uses [Graziani2022]'s open-field "
            "immobility time (vehicle 57.4+/-12.2 s, n=11; diazepam 2 mg/kg "
            "43.6+/-10.5 s, n=4), which is inside P7.1's declared sedation "
            "assay list ('reduced spontaneous locomotion'). It is the same "
            "four animals as the locomotion target read in the opposite "
            "direction, and it comes back WRONG-DIRECTION for the same "
            "reason: diazepam at 2 mg/kg made the rats less immobile, not "
            "more."
        ),
    },
    "working_memory": {
        "verdict": "no-data",
        "entries": (),
        "record": (
            "One paper would qualify and it is closed: La Marca S, Lozito "
            "RJ, Dunn RW, 'Cognitive and EEG recovery following bolus "
            "intravenous administration of anesthetic agents', "
            "Psychopharmacology (Berl) 1995;120(4):426-432, "
            "doi:10.1007/BF02245814, PMID 8539323 - an eight-arm radial "
            "maze after a BOLUS IV fentanyl dose at the ED150 for loss of "
            "righting (0.06 mg/kg), tested immediately, 15 min and 30 min "
            "after the righting reflex returned. Unpaywall reports is_oa "
            "false, Springer redirects to an authentication wall, and "
            "Semantic Scholar reports CLOSED, so neither the group sizes "
            "nor whether any value is tabulated could be established.\n"
            "PubMed returns ONE record for fentanyl AND rats AND ('working "
            "memory' OR 'delayed match' OR 'radial arm' OR '5-choice' OR "
            "'attentional set'). The nearest verified substitute, Obert DP, "
            "et al., Br J Anaesth 2025;135(4):953-964, "
            "doi:10.1016/j.bja.2025.05.058, PMID 40841222 (PMC12495418), "
            "gives IV fentanyl 50 ug/kg against a paired-associate learning "
            "task in 16 rats - but paired-associate learning is long-term "
            "associative memory, not working memory, and P7.1's endpoint "
            "names delayed alternation, delayed non-match-to-sample and "
            "spontaneous alternation. Substituting it would be the assay "
            "substitution rule 2 forbids.\n"
            "P7.1 already recorded that this endpoint is structurally "
            "unable to hold an inverted U (all six terms positive), and "
            "P7.2 recorded that [Zahrt1997]'s prefrontal dose-response is "
            "figure-only. Nothing found here changes either.\n"
            "What would close it: the full text of [LaMarca1995]."
        ),
    },
    "sensorimotor_gating": {
        "verdict": "no-data",
        "entries": (),
        "record": (
            "A complete void, and the strongest of the four negatives "
            "because it is not about access. NO rat study, by ANY route, "
            "has measured prepulse inhibition or acoustic startle after "
            "fentanyl: PubMed returns 1 record for fentanyl[tiab] AND "
            "'prepulse inhibition'[tiab], 6 for the wider startle "
            "formulation, and 0 for opioid AND 'prepulse inhibition' AND "
            "rats AND intravenous. All eight candidates fail on identity "
            "rather than on access - PMID 11432686 delivers fentanyl by "
            "osmotic minipump and studies WITHDRAWAL; PMID 31856925's PPI "
            "arm contains no fentanyl at all; PMID 33145940 and PMID "
            "36397617 are mouse; PMID 2569282 is alfentanil.\n"
            "Haloperidol has the design and not the route: Danna CL, Elmer "
            "GI, Pharmacol Biochem Behav 2010;96(1):40-47, "
            "doi:10.1016/j.pbb.2010.04.004, PMID 20416333 (PMC3752986) runs "
            "a four-dose (3, 10, 30, 100 ug/kg) haloperidol dose-response "
            "against amphetamine-disrupted PPI in male Sprague-Dawley rats "
            "at a 15 min pretreatment - INTRAPERITONEALLY, and "
            "`drug_pk.SEARCHED_AND_EMPTY[('haloperidol', 'rat', 'IP')]` "
            "records that no F and no ka exists for that route. This is the "
            "single closest miss for this endpoint in the whole search, and "
            "what it needs is a PK number, not a behavioural one.\n"
            "P7.2 independently reported that the entire open-access "
            "intracerebral PPI literature - 170 papers - prints no %PPI "
            "value with a mean and SEM in a table. The two searches agree "
            "and they were looking at different literatures.\n"
            "What would close it: haloperidol rat IP bioavailability and "
            "absorption rate, which would make [Danna2010] reachable."
        ),
    },
}


def endpoint_coverage():
    """`ENDPOINT_COVERAGE`, checked against the two things it can drift
    from: the seven P7.1 endpoints and this module's own entry keys.

    Computed rather than trusted for the same reason `pk_route_status`
    asks `drug_pk` instead of storing a verdict: a coverage table that can
    disagree with the library it covers is worse than no table.
    """
    from .. import endpoints as _endpoints

    missing = set(_endpoints.ENDPOINTS) - set(ENDPOINT_COVERAGE)
    extra = set(ENDPOINT_COVERAGE) - set(_endpoints.ENDPOINTS)
    if missing or extra:
        raise RuntimeError(
            "ENDPOINT_COVERAGE does not cover exactly the seven endpoints: "
            "missing %s, unknown %s" % (sorted(missing), sorted(extra))
        )
    for key, record in ENDPOINT_COVERAGE.items():
        if record["verdict"] not in ENDPOINT_COVERAGE_VERDICTS:
            raise RuntimeError("%s: unknown verdict %r" % (key, record["verdict"]))
        for entry_key in record["entries"]:
            if entry_key not in BEHAVIORAL_EVIDENCE_BY_KEY:
                raise RuntimeError(
                    "%s names an entry that does not exist: %r" % (key, entry_key)
                )
        if record["verdict"] == "no-data" and record["entries"]:
            raise RuntimeError("%s: no-data with entries attached" % key)
        if record["verdict"] != "no-data" and not record["entries"]:
            raise RuntimeError("%s: %s with no entries" % (key, record["verdict"]))
    return ENDPOINT_COVERAGE


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
    "diazepam/open-field/Graziani2022": (
        "Bridge available and exercised end to end, same dose and route as "
        "the EPM entry from the same paper. P7.3 fits the locomotion and "
        "sedation endpoints against it and reports both as "
        "wrong-direction: a whole-brain potentiation of inhibition lowers "
        "the locomotion index, and these rats crossed more lines."
    ),
    "diazepam/locomotor-activity/Sagara2008": (
        "Bridge available and exercised end to end at THREE doses, which "
        "is what makes this the only graded series in the module. The fit "
        "converges and identifies nothing: the whole-brain locomotion index "
        "moves 4.7-6.4% at the largest gain the prior allows, against "
        "widths of 17-20% on the published ratios, so the posterior is its "
        "own prior. A null series could have bounded the gain from above; "
        "this one is not precise enough relative to what the illustrative "
        "operating point lets the model move."
    ),
    "diazepam/hot-plate/Lustyk2023": (
        "Bridge available and exercised end to end at two doses. The "
        "analgesia endpoint's only receptor-drivable target: diazepam is "
        "the one drug here whose rat brain partition is a Kp,uu rather "
        "than a Kp."
    ),
    "fentanyl/tail-flick/Yassen2008": (
        "Bridge UNNECESSARY and, separately, incomplete. Unnecessary "
        "because the source publishes a concentration-response rather than "
        "a dose-response, so no dose has to be converted. Incomplete "
        "because `receptor_drive_status()` is 'total-brain-only': no rat "
        "brain unbound fraction for fentanyl exists, so a plasma "
        "concentration cannot become a receptor-driving one. Not fitted, "
        "and the second, independent reason is the curve's steepness - see "
        "ENDPOINT_COVERAGE['analgesia']."
    ),
    "fentanyl/righting-reflex/Obert2025": (
        "Bridge available for the PK step and 'total-brain-only' for the "
        "receptor step. Recorded and not fitted for a third reason that "
        "would apply even if both were clean: a return-of-righting-reflex "
        "LATENCY is a threshold-crossing time, which every monotone stage "
        "of this model preserves, so it constrains the threshold and not "
        "the endpoint."
    ),
    "fentanyl/righting-reflex-ED50/Kissin1983": (
        "As [Obert2025], with a quantal ED50 in place of a latency, and the "
        "same three verdicts spelled out rather than cross-referenced: the "
        "PK step is available (fentanyl/rat/IV, [Yassen2008]); the receptor "
        "step is 'total-brain-only', so the dose cannot reach an occupancy; "
        "and a quantal ED50 is a threshold-crossing dose, which every "
        "monotone stage of this model preserves, so it would constrain the "
        "threshold rather than the endpoint even if the first two were "
        "clean."
    ),
    "diazepam/elevated-plus-maze/Graziani2022": (
        "Bridge available and exercised end to end, and then SPENT: 2 mg/kg "
        "IP -> plasma -> unbound brain "
        "nM, and the effect is the assay's own primary measure with a "
        "vehicle group, SEMs and group sizes. This is the entry P2.2 fits "
        "gaba_a_connection_modulation against; it is the first behavioural "
        "dataset in this project to become a likelihood target."
    ),
}


# ---------------------------------------------------------------------------
# Turning one entry into a comparable quantity (P2.2)
# ---------------------------------------------------------------------------
#
# A circuit-layer gain cannot be fitted against "89.0 s on the open arms",
# because nothing in this codebase produces seconds. What it does produce is
# `anxiety_index_readout()`, the CeA population's firing rate, a number
# bounded in (0, 1) that rises with anxiety. The two functions below turn a
# published EPM row into a quantity of the SAME shape - bounded in [0, 1],
# rising with anxiety - so that the comparison P2.2 makes is between two
# anxiety scales rather than between a rate and a duration.
#
# Neither function converts the model's units into behavioural units, and
# P2.2 does not fit an absolute readout value. It fits the DRUG/VEHICLE
# RATIO of these quantities, which is why no scale factor between "CeA rate"
# and "avoidance fraction" has to be invented: a common multiplicative scale
# cancels out of a ratio. The residual assumption, stated here because it is
# the one assumption in the P2.2 fit that no citation backs, is that the two
# scales change PROPORTIONALLY under drug. See docs/architecture.md, P2.2.

#: [BernalMorales2017] Bernal-Morales B, Cueto-Escobedo J, Guillen-Ruiz G,
#: Rodriguez-Landa JF, Contreras CM. "A Fatty Acids Mixture Reduces
#: Anxiety-Like Behaviors in Infant Rats Mediated by GABA-A Receptors."
#: Biomed Res Int. 2017;2017:8798546. doi:10.1155/2017/8798546, PMID
#: 29392140, PMC5748109, open access, read in full here. Prints the composite
#: EPM Anxiety Index formula used by `epm_composite_anxiety_index` verbatim:
#: "Anxiety Index (AI = 1 - [([Open arm time/Test duration] + [Open arm
#: entries/Total number of entries])/2])", attributing it to Cohen H, Matar
#: MA, Zohar J, "Animal models of post-traumatic stress disorder", Curr
#: Protoc Neurosci 2013;64:9.45, doi:10.1002/0471142301.ns0945s64, PMID
#: 23853112. The formula quoted here was read from [BernalMorales2017]'s own
#: full text; the Cohen chapter itself was not obtained, and that is why this
#: index is used only for the P2.2 sensitivity check and not for the fit.
EPM_COMPOSITE_ANXIETY_INDEX_SOURCE = (
    "Bernal-Morales et al., Biomed Res Int 2017;2017:8798546, "
    "doi:10.1155/2017/8798546 (PMC5748109), printing the index it attributes "
    "to Cohen, Matar & Zohar, Curr Protoc Neurosci 2013;64:9.45, "
    "doi:10.1002/0471142301.ns0945s64 (that chapter itself not obtained here)"
)


def open_arm_avoidance_fraction(effect, group_prefix, trial_duration_s):
    """`(mean, sd_of_mean)` of the fraction of the trial spent OFF the open
    arms, from one published group's open-arm time.

    This is the primary P2.2 quantity: it is the assay's own primary
    variable rescaled by the trial duration the source states, nothing more.
    `sd_of_mean` is the source's own SEM divided by the same duration; no
    uncertainty is invented and none is converted between SEM and SD.
    """
    if effect.dispersion_kind != "SEM":
        raise ValueError(
            "expected SEM-reported group means; got %r" % (effect.dispersion_kind,)
        )
    key = "%s open_arm_time_s" % group_prefix
    mean_s = float(effect.values[key])
    sem_s = float(effect.dispersion[key])
    duration = float(trial_duration_s)
    return 1.0 - mean_s / duration, sem_s / duration


def epm_composite_anxiety_index(effect, group_prefix, trial_duration_s):
    """The composite EPM Anxiety Index (see
    `EPM_COMPOSITE_ANXIETY_INDEX_SOURCE`), point estimate only.

    Used by P2.2 ONLY as a sensitivity check against
    `open_arm_avoidance_fraction`: it mixes a time term with an entry-count
    term, and the two terms share the open-arm entries count, so their
    covariance - which the source does not report - would be needed to
    propagate an honest uncertainty. No sd is returned rather than one being
    fabricated.
    """
    duration = float(trial_duration_s)
    time_term = float(effect.values["%s open_arm_time_s" % group_prefix]) / duration
    entry_term = (
        float(effect.values["%s open_arm_entries" % group_prefix])
        / float(effect.values["%s total_entries" % group_prefix])
    )
    return 1.0 - 0.5 * (time_term + entry_term)


def measure_ratio(effect, vehicle_key, drug_key):
    """`(ratio, sd)` of one published measure, drug group over vehicle group.

    The P7.3 generalisation of `anxiolysis_ratio` below. That function is
    specific to the elevated plus maze, because the quantity P2.2 needed was
    open-arm AVOIDANCE - one minus a fraction - and building it needs the
    trial duration. Most published behavioural measures are already on a
    ratio scale with a real zero (crossing episodes, immobility seconds,
    latency), and for those the drug/vehicle ratio is the measure itself,
    divided.

    `sd` is the first-order (delta method) propagation of the two groups'
    own dispersions, in whatever units the source printed them, and is
    therefore literature-derived rather than an engineering allowance. The
    dispersion kind is NOT converted: an SEM in gives an SEM-scaled sd out,
    and the caller is responsible for knowing which it asked for, exactly as
    `DISPERSION_KINDS` says.
    """
    if effect.dispersion is None:
        raise ValueError("this effect prints no dispersion, so no sd exists")
    mean_v = float(effect.values[vehicle_key])
    mean_d = float(effect.values[drug_key])
    sd_v = float(effect.dispersion[vehicle_key])
    sd_d = float(effect.dispersion[drug_key])
    if mean_v == 0.0:
        raise ValueError(
            "%r has a vehicle mean of zero, so no drug/vehicle ratio exists"
            % (vehicle_key,)
        )
    ratio = mean_d / mean_v
    sd = abs(ratio) * np.sqrt((sd_d / mean_d) ** 2 + (sd_v / mean_v) ** 2)
    return ratio, sd


def anxiolysis_ratio(effect, vehicle_prefix, drug_prefix, trial_duration_s,
                     index=open_arm_avoidance_fraction):
    """`(ratio, sd)` of drug-group to vehicle-group anxiety on `index`.

    The ratio is what P2.2's likelihood compares against the model's own
    drug/baseline anxiety-index ratio. `sd` is the first-order (delta
    method) propagation of the two groups' own SEMs, and is therefore a
    literature-derived likelihood width rather than the kind of explicitly
    declared engineering allowance `evidence.OCCUPANCY_LIKELIHOOD_SD` is.
    Raises if `index` supplies no uncertainty (the composite index above).
    """
    vehicle = index(effect, vehicle_prefix, trial_duration_s)
    drug = index(effect, drug_prefix, trial_duration_s)
    if not isinstance(vehicle, tuple):
        raise ValueError(
            "%r returns a point estimate only and cannot carry a likelihood "
            "width; it is a sensitivity check, not a target" % (index,)
        )
    (mean_v, sd_v), (mean_d, sd_d) = vehicle, drug
    ratio = mean_d / mean_v
    sd = abs(ratio) * np.sqrt((sd_d / mean_d) ** 2 + (sd_v / mean_v) ** 2)
    return ratio, sd


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
        "INTRAVENOUS fentanyl dose.\n"
        "----\n"
        "RE-RUN IN P2.2, still empty. All 421 open-access Europe PMC "
        "articles matching (fentanyl AND ('nucleus accumbens' OR "
        "microdialysis) AND dopamine AND rat) were fetched in full and "
        "screened for an intravenous fentanyl mg/kg or ug/kg dosing "
        "statement together with an accumbens dopamine percentage. 19 "
        "articles carried an IV fentanyl dose; none reports accumbens "
        "dopamine as a percentage of its own baseline. The IV fentanyl "
        "literature in the rat is a self-administration literature (0.25-2.5 "
        "ug/kg/infusion), which measures intake, not dialysate dopamine.\n"
        "One near-miss worth naming because it changes what a later pass "
        "should do, not what this one can: [Yassen2008], already this "
        "codebase's fentanyl rat IV PK source (drug_pk), is itself a PK/PD "
        "paper. It reports a concentration-anchored fentanyl "
        "pharmacodynamic relationship in the rat - the probability of "
        "complete antinociception (tail flick) as a logistic function of "
        "biophase concentration, odds ratio 3.03 (95% CI 1.87-4.21) per "
        "ng/mL, T(1/2,ke0) 6.08 min (4.43-7.73) - that needs no dose bridge "
        "at all. It is not usable for mu_opioid_coupling, which is the "
        "VTA_GABA gain read out as incentive salience: antinociception is a "
        "PAG endpoint, and using it for the reward gain would be exactly the "
        "endpoint substitution rule 2 forbids in the drug dimension. It is a "
        "real candidate target for mu_opioid_multi_region_coupling's "
        "PAG_Interneuron gain, whose readout (analgesia_index) is the "
        "matching one, and that is a different task from this one."
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
        "plus an oral ka (which would make [Gao2021] reachable).\n"
        "----\n"
        "RE-RUN IN P2.2, still empty, and the second search this pass ran "
        "says why more sharply. All 1303 open-access Europe PMC articles "
        "matching (haloperidol AND rat AND ('locomotor activity' OR "
        "catalepsy OR 'plasma concentration')) were fetched in full and "
        "screened for a haloperidol IV mention or a haloperidol plasma "
        "concentration alongside a locomotor/catalepsy endpoint. 19 "
        "survived; every one of them either doses IP/SC/PO for the "
        "behaviour, or is a review, or is human. The single closest article, "
        "Lei K, He GF, Zhang CL, et al., Exp Anim 2018;67(2):163-173, "
        "doi:10.1538/expanim.17-0091 (PMC5955748, read in full), does "
        "measure rat haloperidol plasma kinetics after BOTH an intravenous "
        "(0.7 mg/kg) and an intragastric (1.4 mg/kg) dose - the pair that "
        "would yield an absolute oral bioavailability - but its own "
        "behavioural arm is the oral one, in MK-801-treated rats, and the "
        "basal-ganglia circuit here has no NMDA-antagonist term, so its "
        "locomotor numbers are not comparable with locomotor_activity_"
        "readout even if its F were extracted.\n"
        "A second, route-free way to reach this gain was searched for and "
        "also not found: a rat study reporting measured D2 RECEPTOR "
        "OCCUPANCY alongside a locomotor or cataleptic effect would let "
        "d2_coupling be fitted with no PK step at all, since occupancy is "
        "what the circuit layer consumes. The two papers that do exactly "
        "this - Wadenberg ML, Kapur S, Soliman A, Jones C, Vaccarino F, "
        "Psychopharmacology 2000;150(4):422-9 (PMID 10958084) and Johnson M, "
        "Kozielska M, Pilla Reddy V, et al., Pharm Res 2014;31(10):2605-17, "
        "doi:10.1007/s11095-014-1358-7 (PMID 24792824) - are both closed "
        "access, with no PMC or repository copy reachable here, and neither "
        "abstract carries the occupancy-effect pairs.\n"
        "What would close it, restated: haloperidol rat IP/SC F and ka, OR "
        "the full text of either occupancy-versus-behaviour paper above.\n"
        "----\n"
        "RE-RUN IN P7.3, still empty, and BOTH of the two papers the "
        "paragraph above calls the cheapest way to unblock this system are "
        "now retired. Rule 1 applies to claims copied from this project's "
        "own prose, so both were re-checked rather than inherited.\n"
        "(a) Wadenberg ML, Kapur S, Soliman A, Jones C, Vaccarino F, "
        "Psychopharmacology 2000;150(4):422-9, PMID 10958084, IS NOT A "
        "HALOPERIDOL STUDY. Its abstract is about raclopride throughout: "
        "'Using [3H]-raclopride, cold raclopride (0.01-0.2 mg/kg) produced "
        "16-77% D2 receptor occupancy and no catalepsy.' Haloperidol "
        "appears nowhere in its design, so it fails rule 2 on drug identity "
        "before access is a question, and its endpoints (catalepsy, "
        "conditioned avoidance) are outside the seven P7.1 endpoints "
        "besides. It is still closed (Unpaywall is_oa false, no PMC, no "
        "repository copy on OpenAIRE, DiVA, TSpace or CORE); that no longer "
        "matters.\n"
        "(b) Johnson M, Kozielska M, Pilla Reddy V, et al., Pharm Res "
        "2014;31(10):2605-17, doi:10.1007/s11095-014-1358-7, PMID 24792824, "
        "WAS obtained in full (green OA, University of Groningen), and it "
        "CONTAINS NO MEASURED OCCUPANCY. Its Methods say so: 'it was not "
        "possible to measure D2RO in the same animals and plasma drug "
        "concentrations were not available for these rats'. Every occupancy "
        "in it is simulated from earlier PBPK/PD models, its behavioural "
        "endpoint is catalepsy, its strain is given only as 'Wiga SPF rats' "
        "and its sex is never stated. The measured occupancy it rests on is "
        "in Johnson M, et al., Pharm Res 2011;28(10):2490-504 (PMC3170473, "
        "olanzapine) and Kozielska M, et al., Pharm Res 2012;29(7):1932-48 "
        "(risperidone/paliperidone) - neither of which is haloperidol.\n"
        "One genuinely new near-miss, and it is the best IV lead that "
        "exists: Jankowski MM, Ignatowska-Jankowska BM, Glac W, Wiergowski "
        "M, Kazmierska-Grebowska P, Swiergiel AH, Clin Exp Pharmacol "
        "Physiol 2023;50(6):453-462, doi:10.1111/1440-1681.13762, PMID "
        "36802086, whose open bioRxiv preprint (2021.11.28.470229) states "
        "'Haloperidol (1 mg/kg) or water for injections (1 ml/kg) were "
        "infused intravenously in freely moving rats' and whose Figure S1 "
        "legend describes a FIVE-POINT IV dose-response - 'Haloperidol "
        "(HAL) in doses of 0.1 mg/kg, 0.25 mg/kg; 0.5 mg/kg; 1.0 mg/kg, 2.5 "
        "mg/kg was administered i.v. 10 minutes before' - against "
        "cocaine-induced locomotion, 'Figure shows mean +/- SD'. Every "
        "number is in that figure and no n is stated for it. It is the only "
        "intravenous haloperidol behavioural dose-response in the "
        "literature, and one email to its authors would convert it.\n"
        "A route-free anchor DOES exist and is not usable for these "
        "endpoints: Wadenberg ML, Soliman A, VanderSpek SC, Kapur S, "
        "Neuropsychopharmacology 2001;25(5):633-641, PMID 11682246, Table 1 "
        "prints measured striatal D2 occupancy against dose for haloperidol "
        "(0.025 mg/kg 56 +/- 4.0%, 0.05 68 +/- 5.7%, 0.1 88 +/- 1.8%, 0.5 "
        "94 +/- 1.21%, 1.0 97 +/- 0.6%, n = 4-8, SD, 120 min after "
        "SUBCUTANEOUS dosing). Its behavioural columns are catalepsy and a "
        "conditioned avoidance ED50, neither of which is one of the seven "
        "endpoints, so using it would be the endpoint substitution rule 2 "
        "forbids. Its own Results text and Table 1 also disagree about "
        "risperidone (text '29-71%', table 24 and 78) and quetiapine (text "
        "'57-81%', table lowest 58), which is worth knowing before anyone "
        "quotes it."
    ),
    ("diazepam", "rat", "IP", "elevated plus maze open-arm time"): (
        "SUPERSEDED IN P2.2, and kept rather than deleted because the way it "
        "was wrong is the useful part. This record read 'Zero of 600 "
        "matched'. Re-running the same Europe PMC query in P2.2 returns 724 "
        "open-access records and one of them, [Graziani2022] (PMC9728632), "
        "satisfies every one of this record's own three criteria: it never "
        "mentions mice, it states 'the benzodiazepine diazepam (Sigma, USA) "
        "at 2 mg/kg' with 'All drugs and vehicle were injected by the ip "
        "route', and its Table 2 prints open-arm time as 30.7+/-6.4 s "
        "(vehicle, n=11) against 89.0+/-13.0 s (diazepam, n=8). The zero was "
        "produced by the screen's table-detection step, not by the "
        "literature. The finding to carry forward is therefore 'the rat EPM "
        "literature tabulates diazepam open-arm times rarely - one article "
        "in 724 - but it does tabulate them', and a screen that reports zero "
        "hits should be checked against a hand-verified positive before it "
        "is believed. The original record follows unchanged.\n"
        "----\n"
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


# ---------------------------------------------------------------------------
# The retired P2.2 likelihood target (moved here from calibration/evidence.py
# by P11.3). Between P2.2 and P11.3 this was the ONE behavioural-readout
# calibration target in the codebase, and the gain it fitted -
# `gaba_a_connection_modulation`'s 0.0742 - was the only non-illustrative
# circuit-layer parameter the project ever had. Both facts ended with the
# species pivot: the observation is Wistar-rat behaviour, and a rat dataset
# may not calibrate a human-primary model. See docs/heldout-failure-analysis.md.
#
# The derivation below is unchanged from its production form, so the
# historical posterior in results/calibration/gaba_a_summary.csv stays
# reproducible from this quarantined module. Nothing in production reads it.
#
#    [Graziani2022], via `DIAZEPAM_EPM_GRAZIANI2022`: diazepam 2 mg/kg IP in
#    drug-naive male Wistar rats, elevated plus maze 5 min later, vehicle
#    (n=11) and drug (n=8) open-arm times with SEMs. Everything below is
#    COMPUTED from that entry rather than restated, so the two cannot drift,
#    and the exposure is computed by pushing the dose through the Phase 1
#    bridge (`brain_exposure`, i.e. the same solve_ivp path
#    `engine.Simulation` runs) rather than being quoted.
#
#    Two choices this target made, both of them arguable and therefore both
#    stated here rather than buried:
#
#    - The exposure timepoint is 60 min. [Graziani2022] states a 1 h
#      pre-trial interval for its acute IP injections but names its test
#      peptide, not diazepam, in that sentence, so the entry itself records
#      no timepoint and refuses to invent one. 60 min is the source's own
#      stated acute-trial interval and is passed explicitly here.
#      `EXPOSURE_TIMEPOINT_SENSITIVITY_MIN` carries the alternatives P2.2
#      refit at; the benzodiazepine site is near saturation across the whole
#      window, so the choice barely moves the posterior, and a test asserts
#      that rather than this comment claiming it.
#    - The body weight is the midpoint of the source's 300-380 g band. It
#      cancels: the rat IP diazepam parameter set is per-kg. Also asserted
#      by a test, not claimed here.
#
#    Unlike an occupancy likelihood width, this one is NOT an engineering
#    allowance: it is the delta-method propagation of [Graziani2022]'s own
#    two printed SEMs through the ratio - the first likelihood width in this
#    codebase that a source actually supplied.
# ---------------------------------------------------------------------------
GABA_A_EPM_EXPOSURE_TIMEPOINT_MIN = 60.0
EXPOSURE_TIMEPOINT_SENSITIVITY_MIN = (30.0, 45.0, 60.0)
_GRAZIANI_ENTRY = DIAZEPAM_EPM_GRAZIANI2022
_GRAZIANI_BODY_WEIGHT_KG = sum(GRAZIANI2022_BODY_WEIGHT_BAND_KG) / 2.0


def gaba_a_epm_unbound_brain_nM(timepoint_min=GABA_A_EPM_EXPOSURE_TIMEPOINT_MIN):
    """[Graziani2022]'s 2 mg/kg IP dose as an unbound BRAIN concentration in
    nM at `timepoint_min`, through the Phase 1 bridge.

    Exposed as a function rather than a constant because P2.2's timepoint
    sensitivity check needs it at more than one time, and because a caller
    should be able to see that the number is integrated, not quoted.
    """
    exposure = _GRAZIANI_ENTRY.brain_exposure(
        _GRAZIANI_BODY_WEIGHT_KG, timepoint_min=timepoint_min
    )
    if not exposure.drives_receptor:
        raise RuntimeError(
            "%s did not yield an unbound brain concentration (basis %r); a "
            "receptor may not be driven by anything else"
            % (_GRAZIANI_ENTRY.key, exposure.brain_basis)
        )
    return exposure.brain_nM


_ANXIOLYSIS_RATIO, _ANXIOLYSIS_RATIO_SD = anxiolysis_ratio(
    _GRAZIANI_ENTRY.effect,
    vehicle_prefix="vehicle",
    drug_prefix="diazepam 2 mg/kg",
    trial_duration_s=GRAZIANI2022_EPM_TRIAL_DURATION_S,
)

GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE = {
    "citation": (
        "Graziani D, et al. Braz J Med Biol Res. 2022;55:e12314. "
        "doi:10.1590/1414-431X2022e12314 (PMC9728632)"
    ),
    "evidence_key": _GRAZIANI_ENTRY.key,
    "dose_mg_per_kg": max(_GRAZIANI_ENTRY.doses_mg_per_kg),
    "route": _GRAZIANI_ENTRY.route,
    "species": _GRAZIANI_ENTRY.species,
    # Drug/vehicle ratio of open-arm AVOIDANCE (1 - open-arm time / trial
    # duration), the bounded, rises-with-anxiety quantity the model's own
    # anxiety index is compared against. < 1 means anxiolysis.
    "anxiety_ratio_target_mean": _ANXIOLYSIS_RATIO,
    "anxiety_ratio_target_sd": _ANXIOLYSIS_RATIO_SD,
    "anxiety_ratio_sd_is_source_propagated": True,
    # Same ratio under the composite EPM Anxiety Index, for the P2.2
    # sensitivity check only; no sd, because the source reports no
    # covariance between its time and entry terms.
    "anxiety_ratio_composite_index_mean": (
        epm_composite_anxiety_index(
            _GRAZIANI_ENTRY.effect, "diazepam 2 mg/kg",
            GRAZIANI2022_EPM_TRIAL_DURATION_S)
        / epm_composite_anxiety_index(
            _GRAZIANI_ENTRY.effect, "vehicle",
            GRAZIANI2022_EPM_TRIAL_DURATION_S)
    ),
    "composite_index_source": EPM_COMPOSITE_ANXIETY_INDEX_SOURCE,
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
    "DIAZEPAM_EPM_GRAZIANI2022",
    "DIAZEPAM_OPEN_FIELD_GRAZIANI2022",
    "DIAZEPAM_LOCOMOTION_SAGARA2008",
    "DIAZEPAM_HOT_PLATE_LUSTYK2023",
    "FENTANYL_TAIL_FLICK_YASSEN2008",
    "FENTANYL_RIGHTING_REFLEX_OBERT2025",
    "FENTANYL_RIGHTING_REFLEX_KISSIN1983",
    "ENDPOINT_COVERAGE",
    "ENDPOINT_COVERAGE_VERDICTS",
    "endpoint_coverage",
    "measure_ratio",
    "YASSEN2008_ALPHA",
    "YASSEN2008_BETA_PER_NG_PER_ML",
    "YASSEN2008_STEEPNESS",
    "HILL_ONE_STEEPNESS",
    "yassen2008_antinociception_probability",
    "yassen2008_concentration_at",
    "OBERT2025_RORR_MIN_BY_DOSE_MG_PER_KG",
    "OBERT2025_RORR_IQR_MIN_BY_DOSE_MG_PER_KG",
    "KISSIN1983_FENTANYL_RR_ED50_MG_PER_KG",
    "KISSIN1983_FENTANYL_RR_ED50_FIDUCIAL_LIMITS_MG_PER_KG",
    "SAGARA2008_ACTIVITY_WINDOW_MIN",
    "SAGARA2008_PRETREATMENT_MIN",
    "SAGARA2008_BODY_WEIGHT_BAND_KG",
    "LUSTYK2023_MIN_GROUP_SIZE",
    "LUSTYK2023_PRETREATMENT_MIN",
    "LUSTYK2023_BODY_WEIGHT_BAND_KG",
    "GRAZIANI2022_OPEN_FIELD_DURATION_S",
    "GRAZIANI2022_EPM_TRIAL_DURATION_S",
    "GRAZIANI2022_BODY_WEIGHT_BAND_KG",
    "EPM_COMPOSITE_ANXIETY_INDEX_SOURCE",
    "BEHAVIORAL_EVIDENCE",
    "BEHAVIORAL_EVIDENCE_BY_KEY",
    "BRIDGE_VERDICT",
    "SEARCHED_AND_EMPTY_BEHAVIORAL",
    "bridgeable_entries",
    "unbridgeable_entries",
    "calibration_ready_entries",
    "open_arm_avoidance_fraction",
    "epm_composite_anxiety_index",
    "anxiolysis_ratio",
    "GABA_A_EPM_EXPOSURE_TIMEPOINT_MIN",
    "EXPOSURE_TIMEPOINT_SENSITIVITY_MIN",
    "gaba_a_epm_unbound_brain_nM",
    "GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE",
]
