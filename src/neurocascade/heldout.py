"""P8.2: the held-out compounds, their published binding, and what is missing.

Every compound in this module is one this project has NEVER fitted anything
against. The training set is three drugs - fentanyl, haloperidol and diazepam
- and it is small because `drug_pk` only ever built four (drug, species,
route) triples. `test_heldout.py::test_no_held_out_compound_is_in_the_training_
set` reads the training set off the behavioural-evidence module (quarantined
under `legacy/` by P11.3) rather than restating it here, so a compound that
later acquires a fitting target cannot sit in this module unnoticed. That is
the acceptance criterion "no held-out compound silently moved into the
training set", expressed as an assertion.

What a record in this module is, and is not
-------------------------------------------

It is a transcription of published measurements with their provenance. It is
not a parameter set: several compounds here cannot be simulated from a dose at
all, and the module's job is to say so precisely rather than to fill the gap.
Every field is either a number with a citation, or a `SearchRecord` naming
what was searched and why each candidate failed. There is no third option and
no default.

The four things a prediction needs, and how often we have them
--------------------------------------------------------------

To predict a behavioural profile from a dose, the chain needs

    Ki table -> PK model -> plasma-to-BRAIN-UNBOUND partition -> occupancy
             -> activation -> mechanism gain x density -> circuit -> endpoint

and P8.2's headline finding is about the third link. `brain_partition.py`
(P1.5) refuses to turn a total brain:plasma ratio Kp into the unbound brain
concentration a receptor binds without a measured brain unbound fraction
`fu_brain`, because `Cu_brain = fu_brain * Kp * C_plasma` and fu_brain for a
lipophilic base is 0.001-0.05, so using Kp in its place overstates free drug
by one to three orders of magnitude. That refusal was written for fentanyl,
where P1.5 had already found no rat brain unbound fraction exists.

Searching independently for every compound here reproduced that result almost
uniformly. It is recorded per compound in `PARTITION_SEARCHES`, and the
summary is that a total brain:plasma ratio is common in this literature and a
brain unbound fraction is close to absent. That is a fact about rodent
neuropharmacology's reporting conventions rather than about any one drug, and
it is the single largest reason this task reports fewer end-to-end predictions
than the brief anticipated.

Censored Ki, and the convention this module uses
------------------------------------------------

A screening panel reports "> 10000 nM" for a compound that failed to inhibit
at the screening ceiling. P8.1 deliberately declined to pick between this
codebase's two incompatible conventions (`opioid.ASSAY_CEILING_NM`
instantiates at the bound and flags it; `monoamine_transporters.MISSING_KI`
omits the ligand) and left the choice to the caller.

This module is that caller, and it chooses OMIT, with the ceiling recorded.
A censored value is a statement that affinity is WEAKER than the bound, and
instantiating a ligand at the bound would assert an affinity the assay
explicitly failed to detect - which at a target with endogenous tone would
manufacture a displacement that was measured not to happen. `TESTED_INACTIVE`
keeps every one of them, with its ceiling, because "tested and inactive at
10 uM" and "never tested" are different facts and only the first one licenses
calling a model of this compound complete.
"""

import numpy as np


class SearchRecord(object):
    """A searched-and-empty result, as data rather than as prose.

    Track E's template, made into an object so a test can assert that a gap is
    still documented and a report can print the gaps without hand-maintaining
    a second list of them.
    """

    __slots__ = ("what", "queries", "candidates", "conclusion", "searched")

    def __init__(self, what, queries, candidates, conclusion, searched=True):
        if not what or not conclusion:
            raise ValueError(
                "a search record needs what was sought and what was "
                "concluded; an undocumented gap is indistinguishable from one "
                "nobody looked for"
            )
        if searched and not queries:
            raise ValueError(
                "%r claims to have been searched but lists no queries. If it "
                "genuinely was not searched, pass searched=False and say so: "
                "'we looked and found nothing' and 'nobody looked' are "
                "different facts, and only the first one is evidence about "
                "the literature." % (what,)
            )
        if not searched and queries:
            raise ValueError(
                "%r is marked not-searched but lists queries" % (what,)
            )
        self.what = what
        #: False marks a gap that is a PROCESS failure rather than a property
        #: of the literature. The report counts the two separately, because
        #: presenting an unrun search as a negative result would be the most
        #: consequential kind of overclaiming available in this task.
        self.searched = bool(searched)
        self.queries = tuple(queries)
        #: `(candidate, why it was unusable)` pairs. May be empty when the
        #: searches returned nothing at all, which is itself a stronger
        #: negative than a list of near-misses.
        self.candidates = tuple(candidates)
        self.conclusion = conclusion

    def __repr__(self):
        return "SearchRecord(%r)" % (self.what,)


class KiRecord(object):
    """One published affinity for one compound at one target.

    `ki_nM` is the value; `censored_above` is set instead when the assay only
    established that affinity is weaker than its ceiling. Exactly one of the
    two is not None, enforced, because a censored value silently read as a
    measured one is the failure mode this class exists to prevent.
    """

    __slots__ = ("target", "ki_nM", "censored_above", "species", "preparation",
                 "radioligand", "source", "note")

    def __init__(self, target, source, ki_nM=None, censored_above=None,
                 species=None, preparation=None, radioligand=None, note=None):
        if (ki_nM is None) == (censored_above is None):
            raise ValueError(
                "%s: give exactly one of ki_nM (a measurement) or "
                "censored_above (an assay ceiling the compound did not reach)"
                % target
            )
        for name, value in (("ki_nM", ki_nM), ("censored_above", censored_above)):
            if value is not None and not (value > 0.0 and np.isfinite(value)):
                raise ValueError("%s: %s must be finite and positive" % (target, name))
        if not source:
            raise ValueError("%s: an uncited affinity is an invented one" % target)
        self.target = target
        self.ki_nM = None if ki_nM is None else float(ki_nM)
        self.censored_above = (
            None if censored_above is None else float(censored_above)
        )
        self.species = species
        self.preparation = preparation
        self.radioligand = radioligand
        self.source = source
        self.note = note

    @property
    def is_censored(self):
        return self.ki_nM is None

    def __repr__(self):
        return "KiRecord(%s, %s)" % (
            self.target,
            "censored >%g nM" % self.censored_above if self.is_censored
            else "%g nM" % self.ki_nM,
        )


class EfficacyRecord(object):
    """A functional efficacy, which no Ki table contains.

    P8.1's `MissingEfficacyError` exists because a binding assay measures how
    tightly a compound sits on a receptor and says nothing about what the
    occupied receptor then does. Every entry here therefore cites a FUNCTIONAL
    experiment - GTP-gamma-S, cAMP, electrophysiology - and `assay` records
    which, so that a reader can see that the efficacy and the affinity came
    from different measurements as they must.
    """

    __slots__ = ("target", "efficacy", "kind", "assay", "source", "note")

    #: The functional classes this project distinguishes. `partial_agonist` is
    #: not a separate kind: it is `agonist` with an efficacy below 1.
    KINDS = ("agonist", "antagonist", "inverse_agonist", "channel_blocker",
             "positive_allosteric_modulator", "uptake_blocker",
             "substrate_releaser")

    def __init__(self, target, efficacy, kind, assay, source, note=None):
        if kind not in self.KINDS:
            raise ValueError(
                "%s: unknown efficacy kind %r; expected one of %s"
                % (target, kind, ", ".join(self.KINDS))
            )
        if not (0.0 <= float(efficacy) <= 1.5):
            raise ValueError(
                "%s: efficacy %r is outside [0, 1.5]; this scale is relative "
                "to a full agonist and a value above 1.5 is almost always a "
                "percentage that was not divided by 100" % (target, efficacy)
            )
        if kind == "antagonist" and float(efficacy) != 0.0:
            raise ValueError(
                "%s: a neutral antagonist has efficacy exactly 0; got %r. Use "
                "kind='agonist' for a partial agonist." % (target, efficacy)
            )
        if not assay or not source:
            raise ValueError(
                "%s: an efficacy needs the functional assay it came from and "
                "a citation; it cannot be read off a Ki table" % target
            )
        self.target = target
        self.efficacy = float(efficacy)
        self.kind = kind
        self.assay = assay
        self.source = source
        self.note = note

    def __repr__(self):
        return "EfficacyRecord(%s, %s, %.2f)" % (
            self.target, self.kind, self.efficacy
        )


class OccupancyObservation(object):
    """A published in vivo receptor occupancy, the held-out data for tier 1.

    This is the measurement that tests the receptor layer with NO circuit
    parameter in the chain: PK, partition and Ki determine it completely. It
    is therefore the only tier of P8.2 whose prediction has no unfitted
    quantity anywhere in it.
    """

    __slots__ = ("compound", "target", "species", "route", "dose_mg_per_kg",
                 "time_min", "occupancy", "occupancy_sd", "n", "method",
                 "source", "note")

    def __init__(self, compound, target, species, route, dose_mg_per_kg,
                 occupancy, method, source, time_min=None, occupancy_sd=None,
                 n=None, note=None):
        if not (0.0 <= float(occupancy) <= 1.0):
            raise ValueError(
                "%s/%s: occupancy is a fraction in [0, 1], not a percentage; "
                "got %r" % (compound, target, occupancy)
            )
        if not source or not method:
            raise ValueError("%s/%s: needs a method and a citation"
                             % (compound, target))
        self.compound = compound
        self.target = target
        self.species = species
        self.route = route
        self.dose_mg_per_kg = float(dose_mg_per_kg)
        self.occupancy = float(occupancy)
        self.occupancy_sd = None if occupancy_sd is None else float(occupancy_sd)
        self.n = n
        self.time_min = time_min
        self.method = method
        self.source = source
        self.note = note

    def __repr__(self):
        return "OccupancyObservation(%s@%s, %g mg/kg %s, %.0f%%)" % (
            self.compound, self.target, self.dose_mg_per_kg, self.route,
            100.0 * self.occupancy,
        )


#: How a behavioural number was reported. P7.2 and P7.3 both concluded that
#: "this literature plots everything", and P8.2's seven independent searches
#: reproduced it, so the distinction is carried as data and counted in the
#: report rather than being described in prose.
REPORTING_KINDS = ("tabulated", "plotted-only", "text-only", "unverified")


class BehaviouralObservation(object):
    """A published behavioural effect, the held-out data for tiers 2 and 3.

    Stored as a drug/vehicle RATIO with a standard deviation on that ratio,
    because the model's endpoint index has an arbitrary scale and only the
    ratio is comparable - P2.2's assumption, inherited knowingly and with the
    same cost, which is that the unknown assay scale `k` is the same in both
    groups and therefore cancels.

    `direction` is the sign of the published effect and is recorded
    SEPARATELY from the ratio rather than derived from it, so that an entry
    whose ratio could not be extracted from a figure can still contribute to
    the direction test that tier 2 runs.
    """

    __slots__ = ("compound", "endpoint", "species", "route", "dose_mg_per_kg",
                 "measure", "direction", "ratio", "ratio_sd", "n_vehicle",
                 "n_drug", "window_min", "reporting", "source", "note")

    DIRECTIONS = ("increase", "decrease", "no-change")

    def __init__(self, compound, endpoint, species, route, dose_mg_per_kg,
                 measure, direction, reporting, source, ratio=None,
                 ratio_sd=None, n_vehicle=None, n_drug=None, window_min=None,
                 note=None):
        if direction not in self.DIRECTIONS:
            raise ValueError("%s: unknown direction %r" % (compound, direction))
        if reporting not in REPORTING_KINDS:
            raise ValueError("%s: unknown reporting kind %r" % (compound, reporting))
        if reporting == "tabulated" and ratio is None:
            raise ValueError(
                "%s/%s: reporting is 'tabulated' but no ratio was extracted, "
                "which is a contradiction - if the numbers are in a table, "
                "the ratio is computable" % (compound, endpoint)
            )
        if ratio is not None and not (ratio > 0.0):
            raise ValueError("%s: a ratio of group means must be positive"
                             % compound)
        if not source:
            raise ValueError("%s: needs a citation" % compound)
        self.compound = compound
        self.endpoint = endpoint
        self.species = species
        self.route = route
        self.dose_mg_per_kg = float(dose_mg_per_kg)
        self.measure = measure
        self.direction = direction
        self.ratio = None if ratio is None else float(ratio)
        self.ratio_sd = None if ratio_sd is None else float(ratio_sd)
        self.n_vehicle = n_vehicle
        self.n_drug = n_drug
        self.window_min = window_min
        self.reporting = reporting
        self.source = source
        self.note = note

    @classmethod
    def from_open_arm_percent(cls, vehicle_percent, vehicle_sd, drug_percent,
                              drug_sd, n_vehicle, n_drug, **kwargs):
        """Build an anxiety observation from elevated-plus-maze open-arm time.

        The conversion is the one `endpoint_calibration` already uses for
        P2.2's diazepam contrast, restated here rather than re-invented:
        the model's anxiety index counts evidence pointing towards MORE
        anxiety, so the comparable published quantity is open-arm
        AVOIDANCE, `1 - open_arm_fraction`, and the ratio is
        `avoidance_drug / avoidance_vehicle`. A drug that increases open-arm
        time therefore produces a ratio BELOW one, which is what
        [Graziani2022]'s 0.7835 is.

        The dispersion is propagated by the delta method on a ratio of two
        independent group means, using the standard error of the mean
        (`sd / sqrt(n)`) rather than the group standard deviation - a ratio
        of MEANS needs the uncertainty of the means. P7.3 made the same
        conversion explicit at the point of use for [Lustyk2023], for the
        same reason.
        """
        veh = 1.0 - vehicle_percent / 100.0
        drug = 1.0 - drug_percent / 100.0
        if veh <= 0.0 or drug <= 0.0:
            raise ValueError(
                "open-arm time at or above 100% leaves no avoidance to take a "
                "ratio of"
            )
        sem_veh = (vehicle_sd / 100.0) / np.sqrt(n_vehicle)
        sem_drug = (drug_sd / 100.0) / np.sqrt(n_drug)
        ratio = drug / veh
        ratio_sd = ratio * np.sqrt(
            (sem_drug / drug) ** 2 + (sem_veh / veh) ** 2
        )
        return cls(ratio=float(ratio), ratio_sd=float(ratio_sd),
                   n_vehicle=n_vehicle, n_drug=n_drug, **kwargs)

    @property
    def is_quantitative(self):
        """True when this entry can enter tier 3 (a magnitude comparison).

        Needs both a ratio and a dispersion on it: a ratio with no error bar
        cannot say whether a model missed it by a little or by a lot, and
        reporting a residual in units of an absent standard deviation is how
        an unfalsifiable comparison gets made.
        """
        return self.ratio is not None and self.ratio_sd is not None

    def __repr__(self):
        return "BehaviouralObservation(%s/%s, %g mg/kg, %s, %s)" % (
            self.compound, self.endpoint, self.dose_mg_per_kg, self.direction,
            self.reporting,
        )


class HeldOutCompound(object):
    """One compound: what it binds, what it does there, and what is missing."""

    __slots__ = ("name", "mechanism_class", "ki", "efficacy", "tested_inactive",
                 "never_tested", "occupancy_observations",
                 "behavioural_observations", "pk", "partition", "searches",
                 "note")

    def __init__(self, name, mechanism_class, ki, efficacy, note,
                 tested_inactive=(), never_tested=(),
                 occupancy_observations=(), behavioural_observations=(),
                 pk=None, partition=None, searches=()):
        self.name = name
        self.mechanism_class = mechanism_class
        self.ki = tuple(ki)
        self.efficacy = {record.target: record for record in efficacy}
        self.tested_inactive = tuple(tested_inactive)
        self.never_tested = tuple(never_tested)
        self.occupancy_observations = tuple(occupancy_observations)
        self.behavioural_observations = tuple(behavioural_observations)
        self.pk = pk
        self.partition = partition
        self.searches = tuple(searches)
        self.note = note

        measured = {r.target for r in self.ki if not r.is_censored}
        missing = measured - set(self.efficacy)
        if missing:
            raise ValueError(
                "%s: every target with a measured Ki needs a functional "
                "efficacy, and these have none: %s. P8.1 raises the same "
                "refusal from the other side; a Ki table has no efficacy "
                "column and defaulting one is the cheapest available way to "
                "invent a number." % (name, ", ".join(sorted(missing)))
            )
        spurious = set(self.efficacy) - measured
        if spurious:
            raise ValueError(
                "%s: efficacy given for targets with no measured Ki behind "
                "them: %s" % (name, ", ".join(sorted(spurious)))
            )

    @property
    def modelled_targets(self):
        """The targets with a measured Ki that this model actually carries."""
        from . import fingerprint as _fp

        return tuple(sorted(
            _fp.resolve_target(r.target) for r in self.ki
            if not r.is_censored and _fp.is_modelled(r.target)
        ))

    def fingerprint(self, pools=None):
        """A P8.1 `BindingFingerprint` built from the measured Ki values.

        Censored entries are omitted - see the module docstring - and targets
        this model does not carry go to `BindingFingerprint.unmodelled` with
        the reason, rather than being dropped.

        Each entry carries its own row's provenance (species, preparation,
        radioligand, citation) and its efficacy's SEPARATE citation, because
        the two came from different experiments and P8.1's entry keeps them in
        different fields for exactly that reason.
        """
        from . import fingerprint as _fp

        entries, unmodelled = {}, {}
        for record in self.ki:
            if record.is_censored:
                unmodelled[record.target] = (
                    "censored: the assay established only that affinity is "
                    "weaker than %g nM" % record.censored_above
                )
                continue
            if not _fp.is_modelled(record.target):
                unmodelled[record.target] = "not a target this model carries"
                continue
            canonical = _fp.resolve_target(record.target)
            functional = self.efficacy[record.target]
            entries[canonical] = _fp.FingerprintEntry(
                target=canonical,
                ki_nM=record.ki_nM,
                efficacy=functional.efficacy,
                efficacy_source="%s (%s)" % (functional.source, functional.assay),
                species=record.species,
                source=record.preparation,
                hot_ligand=record.radioligand,
                reference=record.source,
                pool=(pools or {}).get(canonical, "orthosteric"),
                note=record.note or "",
            )
        return _fp.BindingFingerprint(
            self.name, entries, unmodelled=unmodelled or None
        )

    def __repr__(self):
        return "HeldOutCompound(%r, %d Ki, %d occupancy, %d behavioural)" % (
            self.name, len(self.ki), len(self.occupancy_observations),
            len(self.behavioural_observations),
        )


# ---------------------------------------------------------------------------
# The compounds
#
# Each block is a transcription with provenance. Where a search came back
# empty it is a `SearchRecord` rather than a silence, and the report counts
# them.
# ---------------------------------------------------------------------------


# --- ketamine --------------------------------------------------------------
#
# Mechanism class: NMDA open-channel blocker. Chosen for the held-out set
# because it is the one compound whose whole-brain mechanism gain is NOT free:
# `glutamate.GLUTAMATE_CONNECTION_GAIN` is pinned at 1.0 by algebra, so
# ketamine yields a behavioural magnitude prediction with no fitted parameter
# anywhere in it. That makes it the sharpest test in this task.

KETAMINE_KI = (
    KiRecord(
        "NMDA", ki_nM=659.0, species="rat (probable)",
        preparation="PDSP primary screen", radioligand="[3H]MK-801 (probable)",
        source="[Roth2013] PLoS One 8:e59334, PMID 23527166; corrected S1 in "
               "the 2018 erratum PMID 29566092",
        note="(RS)-ketamine, pKi 6.18 +/- 0.07. The radioligand and tissue are "
             "not stated in the retrieved text; PDSP's NMDA assay is "
             "conventionally [3H]MK-801 on rat forebrain, and the site "
             "assignment is therefore PROBABLE rather than confirmed. "
             "Cross-check, same site, different lab: [Ebert1997] Eur J "
             "Pharmacol 333:99, PMID 9311667, [3H]MK-801 on rat brain "
             "homogenate, gives (S) 300 nM and (R) 1400 nM. The racemic value "
             "used here sits between them, which is consistent; the two were "
             "measured in different laboratories and are deliberately not "
             "pooled.",
    ),
    KiRecord(
        "DAT", ki_nM=62900.0, species="rat", preparation="HEK293 stable line",
        radioligand="uptake inhibition, not a radioligand displacement",
        source="[Nishimura1998] Anesthesiology 88:768, PMID 9523822",
        note="Weak but MEASURED, and consistent with the >10 uM screening "
             "ceiling above rather than contradicting it.",
    ),
    KiRecord(
        "SERT", ki_nM=162000.0, species="rat", preparation="HEK293 stable line",
        radioligand="uptake inhibition",
        source="[Nishimura1998] Anesthesiology 88:768, PMID 9523822",
    ),
    KiRecord(
        "NET", ki_nM=66800.0, species="HUMAN", preparation="HEK293 stable line",
        radioligand="uptake inhibition",
        source="[Nishimura1998] Anesthesiology 88:768, PMID 9523822",
        note="HUMAN transporter, where the DAT and SERT values from the SAME "
             "paper are rat. The species split is within one publication and "
             "is carried rather than reconciled.",
    ),
)

KETAMINE_EFFICACY = (
    EfficacyRecord(
        "NMDA", 0.0, "channel_blocker",
        "whole-cell voltage clamp, use- and voltage-dependent open-channel "
        "block; forward rate 3e4-3e5 /M/s, reverse rate ~0.2 /s",
        "[MacDonald1991] J Physiol 432:483, PMID 1832184",
        note="Cultured MOUSE hippocampal neurones. The authors state their "
             "rate constants are the product of the true rate and 1/Po, so "
             "both are LOWER BOUNDS rather than measurements; their ratio is "
             "unaffected. This model uses only the classification (efficacy "
             "0, channel pool) and not the rates, so the caveat does not "
             "propagate into any number here - but it is why a fingerprint "
             "koff at this site must not be read as a measured timescale.",
    ),
    EfficacyRecord(
        "DAT", 0.0, "uptake_blocker", "uptake inhibition in HEK293",
        "[Nishimura1998] Anesthesiology 88:768, PMID 9523822",
    ),
    EfficacyRecord(
        "SERT", 0.0, "uptake_blocker", "uptake inhibition in HEK293",
        "[Nishimura1998] Anesthesiology 88:768, PMID 9523822",
    ),
    EfficacyRecord(
        "NET", 0.0, "uptake_blocker", "uptake inhibition in HEK293",
        "[Nishimura1998] Anesthesiology 88:768, PMID 9523822",
    ),
)

#: Tested and found inactive at the panel's ceiling. Not the same fact as
#: "never tested", and the difference is what lets this compound's model be
#: called complete over the targets it covers.
KETAMINE_TESTED_INACTIVE = tuple(
    (target, 10000.0,
     "[Roth2013] PDSP panel, PMID 23527166; erratum PMID 29566092. Panel "
     "footnote: where <50% inhibition was found on primary screen, Ki is set "
     "to >10000 nM.")
    for target in (
        "5HT1A", "5HT2A", "alpha1", "alpha2", "beta", "H1", "M1", "M2",
        "alpha4beta2", "CB1", "D1", "D2", "GABA_A_BZD", "MOR", "DOR", "KOR",
    )
)

KETAMINE_NEVER_TESTED = (
    ("AMPA", "absent from the Roth PDSP panel; no column for it"),
    ("A2A", "absent from the Roth PDSP panel"),
    ("alpha7", "the panel's nicotinic columns are alpha2beta2, alpha2beta4, "
               "alpha3beta2, alpha3beta4, alpha4beta2 and alpha4beta4; there "
               "is no alpha7 column"),
)

KETAMINE_BEHAVIOUR = (
    BehaviouralObservation(
        "ketamine", "locomotion", "rat", "IV", 5.0,
        measure="open-field activity after a 5 mg/kg IV bolus",
        direction="increase", reporting="unverified",
        source="[Radford2017] Pharmacol Biochem Behav 153:130, PMID 28024909",
        note="Male Sprague-Dawley, IV, with matched plasma concentrations "
             "(peak 4100 ng/mL at this dose). Direction is stated in the "
             "abstract; means, SEM and n were not obtained - the journal is "
             "not in PMC. This paper pairs four behavioural endpoints with "
             "matched plasma concentrations in one cohort and is the single "
             "most valuable unobtained document for this compound.",
    ),
    BehaviouralObservation(
        "ketamine", "sensorimotor_gating", "rat", "IV", 5.0,
        measure="prepulse inhibition", direction="decrease",
        reporting="unverified",
        source="[Radford2017] Pharmacol Biochem Behav 153:130, PMID 28024909",
    ),
    BehaviouralObservation(
        "ketamine", "sedation", "rat", "IV", 9.6,
        measure="ED50 for loss of the righting reflex, 9.6 mg/kg "
                "(95% CI 8.5-10.9), n = 14",
        direction="increase", reporting="text-only",
        source="[Harvey2015] Pharmacology 96:226, PMID 26352278",
        note="A QUANTAL threshold-crossing dose. P7.3's refutation 4 applies "
             "exactly: every stage from plasma concentration to the endpoint "
             "index is monotone, so a threshold crossing is preserved and the "
             "ED50 constrains WHERE the threshold sits on the concentration "
             "axis and nothing about the endpoint's regional weights. It "
             "calibrates drug_pk, not endpoints, and is recorded here so that "
             "it cannot be mistaken for an endpoint constraint later.",
    ),
)

KETAMINE_SEARCHES = (
    SearchRecord(
        what="a brain unbound fraction (fu_brain) or a Kp,uu for ketamine in "
             "rat or mouse",
        queries=(
            "ketamine brain to plasma ratio Kp rat mouse brain plasma "
            "partition coefficient unbound fu,brain",
            "ketamine unbound brain fraction fu,brain brain homogenate "
            "binding Kp,uu rodent equilibrium dialysis",
            '"ketamine" "Kp,uu" OR "unbound brain-to-plasma" microdialysis '
            "rat unbound brain concentration NMDA",
        ),
        candidates=(
            ("PMC8280316, diffusion kinetics of ketamine in brain tissue",
             "reports a brain:aCSF partition of ~2.76, which is an in vitro "
             "TISSUE:BUFFER partition and neither a Kp,uu nor a brain free "
             "fraction"),
            ("Hammarlund-Udenaes methodology reviews, PMID 35411506",
             "describes how Kp,uu is measured without reporting ketamine's"),
        ),
        conclusion="None exists. Total-brain Kp is available (rat 2.25, mouse "
                   "1.32-1.45) and cannot drive receptor occupancy, so "
                   "ketamine is total-brain-only in P1.5's sense - the same "
                   "verdict fentanyl already carries. Plasma protein binding "
                   "is additionally CONCENTRATION-DEPENDENT (32-68% bound "
                   "across 0.1-10 uM in both species), so fu_plasma is not a "
                   "constant either.",
    ),
    SearchRecord(
        what="numeric in vivo NMDA-channel occupancy per dose in a rodent",
        queries=(
            "ketamine in vivo [3H]MK-801 binding occupancy rat dose",
            "ketamine NMDA receptor occupancy PET rodent ex vivo "
            "autoradiography",
        ),
        candidates=(
            ("[Fernandes2015] Eur J Pharmacol 766:1, PMID 26325093",
             "reports dose- and time-dependent inhibition of in vivo "
             "[3H]MK-801 binding reaching ~100%, but no per-dose percentage "
             "with n in the abstract, and the full text was not obtained. An "
             "ED50 of 4.4 mg/kg surfaced in a search snippet and is NOT in "
             "the abstract; it is deliberately not recorded as a number."),
            ("[Lord2013] Eur J Pharmacol 715:21, PMID 23872376",
             "establishes the time course cleanly - maximal at 15 min, no "
             "significant occupancy at 1 h, mouse hippocampus, 30 mg/kg SC - "
             "but the occupancy percentages and n are not in the abstract"),
        ),
        conclusion="Rodent NMDA-channel occupancy after ketamine is measured "
                   "and published, and no per-dose percentage with a "
                   "dispersion could be obtained. Ketamine therefore "
                   "contributes no tier-1 observation.",
    ),
    SearchRecord(
        what="a tabulated behavioural dose-response with three or more doses "
             "plus vehicle",
        queries=(
            "ketamine locomotor activity dose response rat table mean SEM",
            "ketamine prepulse inhibition dose response tabulated",
        ),
        candidates=(
            ("[Hetzler1985] Pharmacol Biochem Behav 22:653, PMID 3991775",
             "the right design - hooded rats, IP, saline/1/10/50/100 mg/kg, "
             "60 min - but the journal is not in PMC and tabulated-vs-plotted "
             "could not be verified"),
            ("[Radford2017] PMID 28024909",
             "four endpoints with matched plasma concentrations, also not in "
             "PMC; direction only"),
        ),
        conclusion="Not met. Both best candidates are in Pharmacology "
                   "Biochemistry and Behavior, neither is in PubMed Central, "
                   "and no numeric behavioural table was obtained for any "
                   "endpoint.",
    ),
)


# --- caffeine --------------------------------------------------------------
#
# Mechanism class: adenosine receptor antagonist. Chosen because it is the
# cleanest test of the endogenous-tone question: A2A is a modelled target,
# and P6.2 measured no basal adenosine, so the model predicts caffeine does
# EXACTLY NOTHING. That is a falsifiable structural prediction and it is
# wrong, which is worth more than a fitted curve.

CAFFEINE_KI = (
    KiRecord(
        "A2A", ki_nM=45000.0, species="rat",
        preparation="rat brain membranes",
        radioligand="see source; compiled in [Fredholm2011] Table 4",
        source="[Daly1991] Pharmacology 42:309, PMID 1658821, as compiled in "
               "[Fredholm2011] Pharmacol Rev 63:1, PMID 21303899, Table 4",
        note="RAT, deliberately, not the widely quoted human value. Human A2A "
             "is 9,560-23,400 nM, i.e. 2-4x tighter, and using it in a rodent "
             "model would be the cross-species substitution rule 2 forbids. "
             "The rat literature spans 23,000-48,000 nM across six "
             "determinations ([Daly1993] 23,000; [Muller2000] 32,500; "
             "[Grahner1994] 43,000; [Daly1991] 45,000; [Jacobson1999] "
             "48,000); the value used is the middle of that range and the "
             "spread is 2.1-fold. Native rat striatal membranes by functional "
             "adenylate cyclase give a Ki of 27,000 nM ([Daly1983] Cell Mol "
             "Neurobiol 3:69, PMID 6309393, full text read), which is a "
             "FUNCTIONAL Ki rather than a displacement constant and is "
             "carried as a cross-check rather than merged.",
    ),
)

CAFFEINE_EFFICACY = (
    EfficacyRecord(
        "A2A", 0.0, "antagonist",
        "surmountable antagonism of 2-chloroadenosine-stimulated adenylate "
        "cyclase in rat striatal membranes; inhibition overcome by high "
        "agonist concentrations",
        "[Daly1983] Cell Mol Neurobiol 3:69, PMID 6309393",
        note="Neutral antagonist is the better-supported default for normal "
             "rodent brain, and the literature is genuinely split. "
             "[Bennett2013] Mol Pharmacol 83:949, PMID 23429888, found "
             "caffeine binds active- and inactive-state thermostabilised "
             "human A2A with equal affinity and classified it a NEUTRAL "
             "antagonist. [FernandezDuenas2014] ACS Chem Biol 9:2496, PMID "
             "25268872, found ~10% suppression of basal cAMP, i.e. INVERSE "
             "agonism - but only in 6-OHDA-lesioned mouse striatum, not in "
             "sham, and [Daly1983] found 100 uM caffeine had no significant "
             "effect on basal adenylate cyclase in native rat striatum. This "
             "model has no elevated constitutive A2A tone, so the neutral "
             "classification is the one that applies.",
    ),
)

CAFFEINE_TESTED_INACTIVE = (
    ("NMDA", 20000000.0,
     "[Vyleta2008] PLoS One 3:e3155, PMID 18781199. Mouse cultured "
     "neocortical neurones: I_NMDA 3.09 +/- 0.59 nA control vs 2.92 +/- 0.56 "
     "nA with 20 mM caffeine, n = 6, P = 0.21. A null at 20 millimolar."),
    ("AMPA", 7000000.0,
     "[Vyleta2008] PMID 18781199. Glutamate-activated current IC50 ~10 mM "
     "peak, ~7 mM steady-state, n = 7. Active only at concentrations two "
     "orders of magnitude above the adenosine range."),
)

CAFFEINE_NEVER_TESTED = (
    ("D1", "the frequently mis-cited [Ferre1990] Eur J Pharmacol 179:295, "
           "PMID 2194822, measured PARAXANTHINE (Ki 72.1 uM), a metabolite, "
           "not caffeine. No caffeine D1 affinity was found."),
    ("D2", "same paper; paraxanthine did not displace [3H]spiperone. No "
           "caffeine D2 affinity was found."),
    ("5HT1A", "no primary displacement data found"),
    ("5HT2A", "no primary displacement data found"),
    ("alpha1", "no primary displacement data found"),
    ("alpha2", "no primary displacement data found"),
    ("beta", "no primary displacement data found"),
    ("H1", "no primary displacement data found"),
    ("M1", "no primary displacement data found"),
    ("M2", "no primary displacement data found"),
    ("alpha4beta2", "no primary displacement data found"),
    ("alpha7", "no primary displacement data found"),
    ("CB1", "no primary displacement data found"),
    ("MOR", "no primary displacement data found"),
    ("DOR", "no primary displacement data found"),
    ("KOR", "no primary displacement data found"),
    ("DAT", "no primary displacement data found"),
    ("SERT", "no primary displacement data found"),
    ("NET", "no primary displacement data found"),
)

CAFFEINE_OCCUPANCY = (
    OccupancyObservation(
        "caffeine", "A2A", "mouse", "IP", 1.0, occupancy=0.32,
        time_min=30.0,
        method="in vivo displacement of striatal [3H]SCH 58261 accumulation",
        source="[ElYacoubi2001] Eur J Neurosci 14:1567, PMID 11722618",
        note="CD1 mice. The only in vivo A2A occupancy for caffeine in any "
             "rodent that was found.",
    ),
    OccupancyObservation(
        "caffeine", "A2A", "mouse", "IP", 6.25, occupancy=0.66,
        time_min=30.0,
        method="in vivo displacement of striatal [3H]SCH 58261 accumulation",
        source="[ElYacoubi2001] Eur J Neurosci 14:1567, PMID 11722618",
    ),
)

CAFFEINE_BEHAVIOUR = (
    BehaviouralObservation(
        "caffeine", "locomotion", "mouse", "IP", 12.5,
        measure="Digiscan activity counts, non-habituated CD1 mice, "
                "15-30 min bin",
        direction="increase", ratio=2.0, reporting="plotted-only",
        n_vehicle=8, n_drug=8, window_min=(15.0, 30.0),
        source="[ElYacoubi2000] Br J Pharmacol 129:1465, PMID 10742303, "
               "PMC1571962",
        note="The 2-fold figure is the ONE numeric value stated in the text "
             "('a 2-fold increase in locomotion was reached during the second "
             "sampling period'); every other value is in a figure. No "
             "dispersion is available, so this entry carries a ratio and no "
             "ratio_sd and is therefore NOT quantitative in this module's "
             "sense - it can enter the direction test and not the magnitude "
             "test. Biphasic: stimulation at 6.25-25 mg/kg, depression at "
             "100 mg/kg, turnover between 25 and 100 mg/kg. In HABITUATED "
             "mice 100 mg/kg induced no change at all, so the depressant limb "
             "is context-dependent rather than intrinsic. In A2A-knockout "
             "mice caffeine dose-dependently DECREASED locomotion across the "
             "whole range, which is the genetic evidence that the stimulant "
             "limb is A2A-mediated.",
    ),
    BehaviouralObservation(
        "caffeine", "locomotion", "mouse", "IP", 100.0,
        measure="Digiscan activity counts, non-habituated CD1 mice",
        direction="decrease", reporting="plotted-only",
        n_vehicle=8, n_drug=8,
        source="[ElYacoubi2000] Br J Pharmacol 129:1465, PMID 10742303",
    ),
    BehaviouralObservation(
        "caffeine", "sedation", "mouse", "IP", 10.0,
        measure="wakefulness; increased in wild-type and A1-knockout mice but "
                "NOT in A2A-knockout mice",
        direction="decrease", reporting="unverified",
        source="[Huang2005] Nat Neurosci 8:858, PMID 15965471",
        note="Three doses (5, 10, 15 mg/kg). 'decrease' here means decreased "
             "SEDATION, i.e. increased wakefulness. The A2A-knockout result "
             "is the mechanistic anchor: caffeine-induced wakefulness is "
             "A2A-dependent.",
    ),
)

CAFFEINE_SEARCHES = (
    SearchRecord(
        what="caffeine Ki at MOUSE A1 and MOUSE A2A",
        queries=(
            "caffeine mouse brain adenosine receptor binding",
            "mouse adenosine A1 receptor cDNA cloning pharmacological "
            "characterization",
            "adenosine receptor species comparison rat mouse human caffeine "
            "affinity",
        ),
        candidates=(
            ("[Maemoto1997] Br J Pharmacol 122:1202, PMID 9401787, full PDF "
             "read", "its Table 3 covers nine antagonists across rat, mouse, "
             "guinea-pig and human cortex, and caffeine is absent from all of "
             "them"),
            ("[Fredholm2011] Pharmacol Rev Table 4 and [Muller2011] Handb Exp "
             "Pharmacol Table 1, both read verbatim",
             "the two most complete species-resolved caffeine tables in the "
             "literature; neither contains any mouse A1 or mouse A2A caffeine "
             "entry, though both carry mouse entries for other ligands"),
            ("ChEMBL, all 212 caffeine x adenosine-receptor activity records "
             "enumerated", "the only Mus musculus entries are A2B"),
        ),
        conclusion="No mouse caffeine affinity at A1 or A2A exists. The "
                   "occupancy observations above are in MOUSE and the Ki used "
                   "is RAT, and that mismatch is carried explicitly rather "
                   "than hidden: it is the reason caffeine's tier-1 residual "
                   "cannot be read as a pure test of the model.",
    ),
    SearchRecord(
        what="a rodent brain:plasma partition and brain unbound fraction for "
             "caffeine",
        queries=(
            "caffeine brain plasma Kp rat unbound brain fraction",
            "caffeine pharmacokinetics rat volume of distribution clearance",
        ),
        candidates=(
            ("[Liu2006] Drug Metab Dispos 34:1443, PMID 16760229",
             "measured fu_brain by both brain-slice and brain-homogenate "
             "methods and names caffeine as a rapid-penetration compound, but "
             "the publisher returned HTTP 403 and the caffeine ROW could not "
             "be extracted; the five-compound Kp,uu range of 0.29-3.9 is not "
             "caffeine's value and is deliberately not used as one"),
            ("[Nakazono1992] Chem Pharm Bull 40:2510, PMID 1446372",
             "rat brain and CSF microdialysis with validated recovery, but "
             "the Kp is not in the abstract and the full text was not "
             "obtained"),
        ),
        conclusion="No usable caffeine Kp or Kp,uu was obtained, and no "
                   "rodent plasma protein binding either - the two rodent "
                   "binding values that surface in searches belong to "
                   "THEOBROMINE (fu 0.90) and PARAXANTHINE (fu 0.85), both "
                   "metabolites, and neither may be used for caffeine.",
    ),
    SearchRecord(
        what="a caffeine locomotor dose-response with three or more doses "
             "plus vehicle, tabulated with means, SEM and n",
        queries=(
            "caffeine locomotor activity dose response mice table mean SEM",
            "caffeine biphasic locomotor dose response tabulated rat",
        ),
        candidates=(
            ("[ElYacoubi2000] PMID 10742303, PMC1571962, full text read",
             "the right design exactly - 6.25/12.5/25/100 mg/kg IP plus "
             "vehicle, n = 8-10, Digiscan - and every value is in a figure "
             "caption. One number appears in the running text and is "
             "recorded above."),
        ),
        conclusion="Not met, and the miss is by reporting format rather than "
                   "by study design. This is P7.2's and P7.3's finding - this "
                   "literature plots everything - reproduced by a search that "
                   "was looking at a different drug and a different decade.",
    ),
    SearchRecord(
        what="in vivo A2A occupancy in the RAT, to match the rat Ki",
        queries=(
            "caffeine A2A receptor occupancy rat PET preladenant SCH442416",
            "caffeine ex vivo binding adenosine receptor rat brain occupancy "
            "dose",
        ),
        candidates=(
            ("[Paul2014] J Nucl Med 55:315, PMID 24434292",
             "gives rat A1 occupancy - 65.9% at 4 mg/kg IP and 98.5% at "
             "40 mg/kg IP, n = 6 per dose, Wistar, [11C]MPDX - which is A1 "
             "and not A2A; A1 is not a target this model carries"),
        ),
        conclusion="Every rodent caffeine occupancy study found measures A1. "
                   "No rat A2A occupancy exists, so the only A2A observations "
                   "available are the mouse ones above.",
    ),
)


# --- cocaine ---------------------------------------------------------------
#
# Mechanism class: monoamine uptake blocker. Chosen as the deliberate
# contrast with amphetamine - same three targets, opposite mechanism (pure
# blocker vs substrate releaser) - and because it is the clearest case of a
# compound whose behavioural mechanism the whole-brain model has NO path for
# at all.

COCAINE_KI = (
    KiRecord(
        "DAT", ki_nM=211.0, species="rat", preparation="rat brain synaptosomes",
        radioligand="[3H]dopamine uptake inhibition",
        source="[Baumann2013] Neuropsychopharmacology 38:552, PMID 23072836, "
               "Table 1",
        note="N = 3. Chosen over the human-transfected panel because "
             "amphetamine's values come from the SAME table in the SAME "
             "laboratory, which is what makes the blocker/releaser contrast "
             "between the two compounds a controlled comparison rather than a "
             "cross-study one. The human HEK293 value is 768 nM "
             "([Simmler2013] Br J Pharmacol 168:458, PMID 22897747), 3.6-fold "
             "weaker; the two panels are not mixed.",
    ),
    KiRecord(
        "NET", ki_nM=292.0, species="rat", preparation="rat brain synaptosomes",
        radioligand="[3H]norepinephrine uptake inhibition",
        source="[Baumann2013] PMID 23072836, Table 1",
    ),
    KiRecord(
        "SERT", ki_nM=313.0, species="rat", preparation="rat brain synaptosomes",
        radioligand="[3H]5-HT uptake inhibition",
        source="[Baumann2013] PMID 23072836, Table 1",
        note="Cocaine is essentially non-selective across the three "
             "transporters, a 1.5-fold spread, in sharp contrast to "
             "amphetamine's ~37-fold DAT-over-SERT selectivity measured in the "
             "same table.",
    ),
    KiRecord(
        "M2", ki_nM=18800.0, species="guinea-pig and rat",
        preparation="heart and brain membranes", radioligand="[3H]QNB",
        source="[Sharkey1988] J Pharmacol Exp Ther 246:1048, PMID 3047364",
        note="Genuine but 60-89x weaker than the transporters.",
    ),
    KiRecord(
        "M1", ki_nM=61400.0, species="rat",
        preparation="Sprague-Dawley rat brain membranes",
        radioligand="[3H]pirenzepine",
        source="[Grundt2004] Bioorg Med Chem Lett 14:3295, PMID 15149693; "
               "independently [Kulkarni2004] PMID 15189035 and [Zou2006] "
               "PMID 17034144, all three reporting 61.4 uM",
    ),
)

COCAINE_EFFICACY = (
    EfficacyRecord(
        "DAT", 0.0, "uptake_blocker",
        "release assay with [3H]MPP+: EC50 151 nM but Emax capped at "
        "29 +/- 1%, against 102 +/- 1% for amphetamine in the same table",
        "[Baumann2013] PMID 23072836",
        note="THE DISCRIMINATOR IS Emax, NOT EC50. Cocaine registers a "
             "release signal, and the paper attributes it to passive leak of "
             "radiolabelled substrate normally counteracted by ongoing "
             "uptake; MDPV, an uncontested pure blocker, shows the identical "
             "low-Emax pattern. A model that classified on EC50 alone would "
             "call cocaine a releaser. Independently confirmed in a different "
             "system and laboratory: no DA or 5-HT efflux in human "
             "transporter-transfected HEK293 ([Simmler2013] PMID 22897747).",
    ),
    EfficacyRecord(
        "NET", 0.0, "uptake_blocker",
        "release EC50 2190 nM with Emax 34 +/- 6%", "[Baumann2013] PMID 23072836",
    ),
    EfficacyRecord(
        "SERT", 0.0, "uptake_blocker",
        "release assay: INACTIVE, no measurable release at all",
        "[Baumann2013] PMID 23072836",
    ),
    EfficacyRecord(
        "M2", 0.0, "antagonist",
        "GppNHp does not shift the competition curve, and cocaine reverses "
        "methacholine-induced inhibition of guinea-pig atrial contraction at "
        "a similar concentration",
        "[Sharkey1988] PMID 3047364",
    ),
    EfficacyRecord(
        "M1", 0.0, "antagonist",
        "binding only; antagonism assigned by class and by the M2 functional "
        "result in [Sharkey1988]",
        "[Grundt2004] PMID 15149693",
        note="This is the weakest efficacy assignment in the module: it is "
             "not backed by its own functional assay at M1. It is recorded as "
             "such rather than presented as measured.",
    ),
)

COCAINE_TESTED_INACTIVE = tuple(
    (target, ceiling,
     "[Simmler2013] Br J Pharmacol 168:458, PMID 22897747, PMC3572571, "
     "Table 1; human recombinant receptors")
    for target, ceiling in (
        ("5HT1A", 20000.0), ("5HT2A", 13000.0), ("alpha1", 6000.0),
        ("alpha2", 20000.0), ("D1", 13600.0), ("D2", 30000.0),
        ("H1", 14400.0),
    )
) + (
    ("NMDA", 10000.0,
     "[He1993] J Med Chem 36:4075, PMID 8258830. [3H]TCP displacement from "
     "the phencyclidine site, rat brain homogenates."),
)

COCAINE_NEVER_TESTED = (
    ("AMPA", "no direct cocaine binding or functional potency measurement "
             "found"),
    ("GABA_A_BZD", "no [3H]flunitrazepam or [3H]Ro15-1788 displacement study "
                   "for cocaine found. A GABA-A IC50 of 127 uM exists "
                   "([Chebib2000] PMID 10780899) but is from a review and is "
                   "not site-specific."),
    ("beta", "the only source is a broad panel whose cocaine record reports "
             "DAT, NET and SERT all >30 uM, contradicting two independent "
             "labs, so the whole record is unreliable and is not used"),
    ("CB1", "same unreliable panel; no usable source"),
    ("A2A", "same unreliable panel; no usable source"),
    ("MOR", "same unreliable panel; no usable source"),
    ("DOR", "same unreliable panel; no usable source"),
    ("KOR", "same unreliable panel; no usable source"),
    ("alpha4beta2", "[Francis2000] Mol Pharmacol 58:109, PMID 10860932, "
                    "demonstrates alpha4-subunit-dependent noncompetitive "
                    "inhibition in the 'low micromolar range' but the numeric "
                    "IC50 was not retrievable"),
    ("alpha7", "only COCAINE METHIODIDE, a different quaternary compound, has "
               "a measured alpha7 number (Ki ~200 nM). Attributing it to "
               "cocaine would be a compound substitution."),
)

COCAINE_OCCUPANCY = (
    OccupancyObservation(
        "cocaine", "DAT", "mouse", "IV", 0.25, occupancy=0.50,
        method="displacement of in vivo [3H]cocaine binding in striatum",
        source="[Gatley1999] Psychopharmacology 146:93, PMID 10485970",
        note="Dose is mg/kg of the hydrochloride salt. Only two points are "
             "stated numerically in the abstract; n and time after dosing are "
             "in the paywalled full text. Note how LOW these doses are "
             "relative to the 5-40 mg/kg IP used throughout the behavioural "
             "literature.",
    ),
    OccupancyObservation(
        "cocaine", "DAT", "mouse", "IV", 1.0, occupancy=0.80,
        method="displacement of in vivo [3H]cocaine binding in striatum",
        source="[Gatley1999] Psychopharmacology 146:93, PMID 10485970",
    ),
)

COCAINE_BEHAVIOUR = (
    BehaviouralObservation(
        "cocaine", "locomotion", "rat", "SC", 10.0,
        measure="forward locomotion; the lowest dose reaching significance",
        direction="increase", reporting="plotted-only", n_drug=7,
        source="[Baumann2013] Neuropsychopharmacology 38:552, PMID 23072836, "
               "Figure 4c-d",
        note="Doses 3-17 mg/kg SC, n = 6-7 sessions per dose. Only the "
             "significance thresholds are stated numerically: locomotion from "
             "10 mg/kg, stereotypy from 3 mg/kg. All values plotted.",
    ),
    BehaviouralObservation(
        "cocaine", "locomotion", "mouse", "IP", 15.0,
        measure="distance travelled in cm, 60 min open field, C57BL/6J "
                "wild-type controls",
        direction="increase", reporting="plotted-only", n_drug=9,
        source="Front Mol Neurosci 2024, PMID 39737113, PMC11683150, "
               "Figures 2-3",
        note="Meets the design criterion exactly - vehicle plus 5, 10, 15 and "
             "40 mg/kg IP, n = 7-11 per group - and every mean and SEM is in "
             "a figure. No table anywhere in the paper.",
    ),
    BehaviouralObservation(
        "cocaine", "anxiety", "rat", "IP", 10.0,
        measure="elevated plus maze, in rats pre-selected as 'nonanxious': "
                "increased latency to first open-arm entry, fewer open-arm "
                "entries, less time on open arms",
        direction="increase", reporting="text-only",
        source="[Rogerio1992] Pharmacol Biochem Behav 43:631, PMID 1438502",
        note="The same authors' earlier work found 5, 10 and 20 mg/kg failed "
             "to produce a significant effect in NAIVE rats - a genuine "
             "three-dose null. The effect here is conditional on baseline "
             "trait anxiety, so a single population mean cannot reproduce it. "
             "No means, SEM or n obtainable.",
    ),
)

COCAINE_SEARCHES = (
    SearchRecord(
        what="a Kp,uu or brain unbound fraction for cocaine in any rodent",
        queries=(
            'cocaine ("brain to plasma" OR "brain:plasma" OR Kp,uu OR '
            '"unbound brain") rat',
            "cocaine brain homogenate binding unbound fraction equilibrium "
            "dialysis",
        ),
        candidates=(
            ("[Benuck1987] J Pharmacol Exp Ther 243:144, PMID 3668848",
             "gives a mouse total brain:plasma ratio of ~7 averaged over "
             "5-60 min after IP dosing, which is a TOTAL ratio and cannot "
             "give an unbound brain concentration"),
            ("[DeVane1989] Life Sci 45:1271, PMID 2811597",
             "gives a PREGNANT rat brain:plasma of ~1, which neither matches "
             "the mouse value nor supplies a free fraction; the two are "
             "carried as an unresolved conflict rather than averaged"),
        ),
        conclusion="Nothing found. Rat plasma protein binding IS available "
                   "and well characterised (fu 0.62-0.69, concentration-"
                   "independent, [Sukbuntherng1996] J Pharm Sci 85:567, PMID "
                   "8773950), which makes cocaine the compound in this set "
                   "where the missing half of the partition is most sharply "
                   "isolated: the plasma side is measured and the brain side "
                   "is not.",
    ),
    SearchRecord(
        what="a rat dose-occupancy curve for DAT, to match the rat Ki",
        queries=(
            '"dopamine transporter occupancy" cocaine rat',
            "cocaine ex vivo (WIN 35,428 OR RTI-55) binding occupancy dose "
            "rat",
        ),
        candidates=(
            ("[Wee2006] Neuropsychopharmacology 31:351, PMID 15957006",
             "gives an in vivo binding RATE in rat striatum - 25% of "
             "[3H]WIN 35,428 displaced at 5.8 min - but no dose-occupancy "
             "curve"),
            ("[Fowler1998] Synapse 28:111, PMID 9450511",
             "a full dose-occupancy curve with ED50 0.17-0.27 mg/kg, in the "
             "BABOON. The same paper shows slow-kinetics tracers give ED50s "
             "of 3-7 mg/kg, a 20-fold tracer artefact, which is a warning "
             "about comparing occupancy numbers across studies."),
        ),
        conclusion="The only rodent dose-occupancy relationship is the mouse "
                   "one recorded above, and the Ki used is rat. Same "
                   "species mismatch as caffeine, from the opposite "
                   "direction.",
    ),
    SearchRecord(
        what="a tabulated cocaine microdialysis or behavioural dose-response",
        queries=(
            'cocaine microdialysis "dose-response" (NAc OR striatum) dopamine '
            '"% of baseline"',
            'cocaine "locomotor activity" Table ("photobeam breaks" OR '
            '"distance travelled") mg/kg',
        ),
        candidates=(
            ("[DiChiara1988] PNAS 85:5274, PMID 2899326, PMC281732",
             "the canonical demonstration, and the PMC deposit is scanned "
             "page images only - no machine-readable text, no doses, no n"),
            ("two locomotor studies meeting the design criteria "
             "([Baumann2013]; PMC11683150)",
             "both present every value in figures"),
        ),
        conclusion="Not met for any endpoint. Roughly 800 titles were "
                   "screened across Europe PMC and PubMed; every cocaine "
                   "dose-response located is a figure.",
    ),
)


# --- d-amphetamine ---------------------------------------------------------
#
# Mechanism class: monoamine substrate releaser. The deliberate partner to
# cocaine: the same three targets, measured in the same laboratory, with the
# opposite mechanism. Whether this model can tell the two apart is a test that
# needs no circuit at all.

AMPHETAMINE_KI = (
    KiRecord(
        "DAT", ki_nM=122.0, species="rat", preparation="rat brain synaptosomes",
        radioligand="[3H]dopamine uptake inhibition",
        source="[Schindler2021] J Pharmacol Exp Ther 376:118, PMID 33082158, "
               "PMC7788351, Table 1",
        note="(S)-amphetamine sulfate, NIDA. n = 3 in triplicate. This is the "
             "UPTAKE-INHIBITION constant, which is what a Ki table reports "
             "and what the fingerprint path consumes. It is NOT the quantity "
             "that carries amphetamine's mechanism: the release EC50 in the "
             "same preparation is 5 +/- 1 nM with Emax 102%, i.e. 24-fold "
             "more potent than the uptake IC50. See AMPHETAMINE_RELEASE.",
    ),
    KiRecord(
        "NET", ki_nM=69.0, species="rat", preparation="rat brain synaptosomes",
        radioligand="[3H]norepinephrine uptake inhibition",
        source="[Schindler2021] PMID 33082158, Table 1",
    ),
    KiRecord(
        "alpha2", ki_nM=2800.0, species="human recombinant",
        preparation="radioligand displacement",
        radioligand="see source", source="[Simmler2013] Br J Pharmacol "
                                         "168:458, PMID 22897747, Table 1",
        note="The only receptor in Simmler's fourteen-target panel where "
             "amphetamine has a real number rather than a '>' bound. Still "
             "23-40x weaker than the transporters, and HUMAN where the "
             "transporter values used here are rat.",
    ),
)

#: The substrate-release parameters, which are what actually distinguishes
#: amphetamine from cocaine and which a Ki table does not contain. P8.1's
#: `FingerprintEntry` accepts `release_ec50` and `release_emax` precisely
#: because a fingerprint transporter entry is a pure blocker until a caller
#: supplies these, and this is the caller supplying them.
AMPHETAMINE_RELEASE = {
    "DAT": (5.0, 1.02,
            "[Schindler2021] PMID 33082158, Table 1: EC50 5 +/- 1 nM, "
            "Emax 102%, rat synaptosomes, [3H]MPP+ release, maximal release "
            "defined by 10 uM tyramine"),
    "NET": (8.0, 0.96,
            "[Schindler2021] PMID 33082158, Table 1: EC50 8 +/- 1 nM, "
            "Emax 96%"),
}

AMPHETAMINE_EFFICACY = (
    EfficacyRecord(
        "DAT", 0.0, "substrate_releaser",
        "release of preloaded [3H]MPP+ from rat synaptosomes: EC50 5 +/- 1 nM "
        "with Emax 102% of the tyramine-defined maximum",
        "[Schindler2021] PMID 33082158",
        note="Efficacy is 0 in the ORTHOSTERIC sense the receptor layer uses - "
             "amphetamine does not activate a G protein at the transporter - "
             "and the releasing action is carried in AMPHETAMINE_RELEASE "
             "instead, which is where `transport.py` consumes it. Full "
             "releasing efficacy (96-103%) at EC50s 15-20x BELOW the "
             "uptake-inhibition IC50s in the same preparation is the direct "
             "contrast with cocaine's 29-34% cap in [Baumann2013]'s table, "
             "and the two laboratories are the same one.",
    ),
    EfficacyRecord(
        "NET", 0.0, "substrate_releaser",
        "release of preloaded [3H]MPP+: EC50 8 +/- 1 nM, Emax 96%",
        "[Schindler2021] PMID 33082158",
    ),
    EfficacyRecord(
        "alpha2", 0.0, "antagonist",
        "binding only; no functional assay at alpha2 was obtained",
        "[Simmler2013] PMID 22897747",
        note="Like cocaine's M1 entry, this efficacy is assigned by class "
             "rather than measured, and is flagged as such rather than "
             "presented as a functional result.",
    ),
)

AMPHETAMINE_TESTED_INACTIVE = tuple(
    (target, ceiling,
     "[Simmler2013] Br J Pharmacol 168:458, PMID 22897747, PMC3572571, "
     "Table 1; human recombinant receptors, radioligand displacement")
    for target, ceiling in (
        ("SERT", 25000.0), ("5HT1A", 6740.0), ("5HT2A", 13000.0),
        ("alpha1", 6000.0), ("D1", 13600.0), ("D2", 30000.0), ("H1", 14400.0),
    )
) + (
    ("NMDA", 300000.0,
     "[Berger2009] Bioorg Med Chem 17:3456, PMID 19345586. [3H]MK-801, rat "
     "brain neuronal membrane, Ki 300 uM."),
)

AMPHETAMINE_NEVER_TESTED = tuple(
    (target,
     "no measurement of d-amphetamine found. Corroborating evidence that this "
     "is a true absence rather than a search failure: the full ChEMBL "
     "activity sets for dextroamphetamine (CHEMBL612, 115 activities, 51 with "
     "an affinity value) and racemic amphetamine (CHEMBL405, 211 activities, "
     "25 with a value) contain no entry for it, and [Simmler2013]'s "
     "fourteen-target panel did not include it.")
    for target in ("MOR", "DOR", "KOR", "M1", "M2", "alpha4beta2", "alpha7",
                   "AMPA", "CB1", "A2A", "GABA_A_BZD", "beta")
)

AMPHETAMINE_BEHAVIOUR = (
    BehaviouralObservation(
        "amphetamine", "locomotion", "rat", "SC", 0.5,
        measure="crossovers (quadrant transitions), 0-60 min: 71 +/- 12",
        direction="increase", reporting="tabulated", ratio=1.0, n_drug=7,
        window_min=(0.0, 60.0),
        source="[Kuczenski1991] J Neurosci 11:2703, PMID 1715389, PMC6575248, "
               "Table 2",
        note="TABULATED, and still not usable as a magnitude. The tables in "
             "this paper have NO VEHICLE GROUP - the comparators are cocaine "
             "and fencamfamine - so a drug/vehicle ratio cannot be formed, "
             "and the `ratio` field is set to 1.0 as a placeholder that "
             "carries no dispersion and therefore fails `is_quantitative`. "
             "The two d-amphetamine doses are also reported over "
             "non-overlapping time windows (0-60 min at 0.5 mg/kg, 0-40 min "
             "at 2.5 mg/kg), so they do not form a dose-response either.",
    ),
    BehaviouralObservation(
        "amphetamine", "locomotion", "rat", "SC", 2.5,
        measure="crossovers, 0-40 min: 176 +/- 21",
        direction="increase", reporting="tabulated", ratio=1.0, n_drug=7,
        window_min=(0.0, 40.0),
        source="[Kuczenski1991] PMID 1715389, Table 4",
    ),
)

AMPHETAMINE_SEARCHES = (
    SearchRecord(
        what="a Kp,uu or brain unbound fraction for amphetamine in rat or "
             "mouse",
        queries=(
            "amphetamine brain to plasma ratio Kp rat Kp,uu unbound",
            'Europe PMC REST: "amphetamine" AND "Kp,uu"',
            'PMC: "amphetamine" AND "Vu,brain"',
            "Friden Hammarlund-Udenaes brain slice Kp,uu amphetamine dataset "
            "rat unbound brain",
        ),
        candidates=(
            ("Europe PMC, 25 results reviewed",
             "every one an M4/M1/mGlu PAM or unrelated CNS-discovery paper "
             "using amphetamine as a behavioural tool; none reports an "
             "amphetamine Kp,uu"),
            ("[Clausing1995] J Pharmacol Exp Ther 274:614, PMID 7636721",
             "gives paired plasma, total brain tissue and brain "
             "extracellular-fluid concentrations after 5 mg/kg SC - 1.7 uM "
             "plasma, 25 uM caudate tissue, 7.9 uM caudate ECF - which is the "
             "closest anything in this whole task comes to a measured free "
             "brain concentration, but the paper prints no Kp and no free "
             "fraction, and computing one from the means would be this "
             "project's arithmetic rather than a published value"),
            ("the standard Kp,uu compilations (Friden 2009 PMID 19764786, "
             "Friden 2011 PMID 21149540, Doran 2005, Summerfield 2007, "
             "Kalvass 2002, Loryan 2013)",
             "all paywalled or 403; whether amphetamine is a member of any of "
             "them is UNRESOLVED rather than confirmed absent"),
        ),
        conclusion="No measured value obtained. This is the fourth "
                   "independent clean negative in this task's seven searches "
                   "and the pattern is now the finding rather than the "
                   "accident. Note the honest caveat above: for amphetamine "
                   "specifically the compilations could not be opened, so the "
                   "negative is weaker than caffeine's or cocaine's.",
    ),
    SearchRecord(
        what="rodent DAT occupancy by amphetamine",
        queries=(
            "amphetamine dopamine transporter occupancy rat PET [11C]PE2I "
            "displacement percent occupancy mg/kg",
        ),
        candidates=(
            ("every DAT-occupancy hit returned",
             "either a different drug (modafinil, cocaine) or human"),
            ("[Johansson2024] J Cereb Blood Flow Metab 44:434, PMID 37882727",
             "the only in vivo rodent occupancy datum for d-amphetamine, and "
             "it is D2 rather than DAT: ~45% in nucleus accumbens, ~40% in "
             "caudate-putamen, ~35% in globus pallidus after 2 mg/kg IV, "
             "peaking at ~22 min, stated with a tilde in prose with no table. "
             "It is a measure of DOPAMINE RELEASE displacing [11C]raclopride, "
             "not of amphetamine occupying anything, so it cannot test this "
             "model's binding layer."),
        ),
        conclusion="No rodent DAT occupancy exists. Mechanistically "
                   "unsurprising - amphetamine is a substrate, and "
                   "substrate-induced internalisation plus competition make "
                   "classical occupancy hard to define - but recorded as a "
                   "real gap. Amphetamine therefore contributes no tier-1 "
                   "observation.",
    ),
    SearchRecord(
        what="rat or mouse plasma free fraction for amphetamine",
        queries=("amphetamine plasma protein binding rat mouse equilibrium "
                 "dialysis",),
        candidates=(
            ("[Baggot1972] Biochem Pharmacol 21:1813, PMID 4630398",
             "unambiguously the right primary - equilibrium dialysis, species "
             "panel including rat and mouse, binding to albumin only - and "
             "the per-species table is paywalled. Only the summary bound "
             "'<45% in all species' is available. A secondary source appears "
             "to cite it for fu = 0.6, and that value is UNVERIFIED and "
             "deliberately not used."),
        ),
        conclusion="Only an upper bound on binding was obtained, which is not "
                   "a free fraction.",
    ),
    SearchRecord(
        what="prepulse inhibition, elevated plus maze, conditioned place "
             "preference and working-memory dose-response for amphetamine",
        queries=(),
        candidates=(),
        searched=False,
        conclusion="NOT SEARCHED, and recorded here so it is not miscounted "
                   "as an absence of literature. These four endpoints were "
                   "delegated to a sub-search that was dropped rather than "
                   "run, and zero queries were issued. Live leads that were "
                   "never followed: Swerdlow (rat PPI), Geyer and "
                   "Ralph-Williams (mouse PPI), Bardo (CPP). This is the one "
                   "gap in this module that is a process failure rather than "
                   "a property of the literature.",
    ),
)


# --- alprazolam ------------------------------------------------------------
#
# Mechanism class: GABA-A benzodiazepine-site positive allosteric modulator.
# The most informative compound in the set, for two reasons that pull in
# opposite directions. It is the ONLY one of the seven with a complete,
# measured plasma-to-brain-unbound partition, so the link that blocks every
# other compound is open here. And it is the only one with a tabulated,
# multi-dose behavioural series with a vehicle group, means, SDs and n.
#
# It shares the benzodiazepine site with diazepam, which IS a training drug.
# That does not make alprazolam a training compound - no fit in this
# repository has ever used an alprazolam number, and the test asserts it -
# but it does mean alprazolam is the one held-out compound whose target has a
# gain that P2.2 and P7.3 both tried to constrain. That makes it the sharpest
# available test of whether those attempts transfer.

ALPRAZOLAM_KI = (
    KiRecord(
        "GABA_A_BZD", ki_nM=3.3, species="rat",
        preparation="rat cortex homogenates",
        radioligand="[3H]flumazenil",
        source="[Albaugh2002] J Med Chem 45:5043, PMID 12408715",
        note="Four independent rat determinations span 3.3-7.9 nM: 3.3 nM "
             "([Albaugh2002], [3H]flumazenil, cortex), 4.6 nM DIRECT "
             "SATURATION of [3H]alprazolam itself ([McCabe1990] Pharmacol "
             "Biochem Behav 37:365, PMID 1964224, Bmax 2.6 pmol/mg), 5.0-8.0 "
             "nM across cerebellum, striatum and spinal cord ([Massotti1991] "
             "J Pharmacol Exp Ther 256:1154, PMID 1848629), and 7.9 nM "
             "([Sanna1999] Arzneimittelforschung 49:88, PMID 10083975, "
             "[3H]flunitrazepam). A 2.4-fold spread, tight by this "
             "codebase's standards. The lowest is used and the range is "
             "reported rather than averaged. Subtype-resolved human "
             "recombinant values are essentially flat across the "
             "diazepam-sensitive subtypes - alpha1 0.8, alpha2 0.59, alpha3 "
             "1.43, alpha5 1.54 nM - and 10 uM at the diazepam-INSENSITIVE "
             "alpha6 ([Clayton2015] Int J Med Chem 2015:430248, PMID "
             "26682068), which is the pharmacology behind P8.1's note that "
             "mapping this site onto Gabra1 over-counts.",
    ),
)

ALPRAZOLAM_EFFICACY = (
    EfficacyRecord(
        "GABA_A_BZD", 1.0, "positive_allosteric_modulator",
        "automated planar patch clamp, human alpha1beta2gamma2 in CHO cells, "
        "GABA at 2 uM (~EC10): Emax 2.84x GABA alone (95% CI 2.65-3.03), "
        "EC50 14.7 nM (95% CI 8.41-24.8), n >= 4",
        "[Norman2026] Commun Chem 9:155, PMID 41946818, PMC13068902, Table 2",
        note="FULL modulator, and the efficacy of 1.0 is on this codebase's "
             "relative-to-full-agonist scale rather than being the raw "
             "2.84-fold potentiation. Three independent lines put it in the "
             "full class: [Giusti1993] J Pharmacol Exp Ther 266:1018, PMID "
             "8394902, classifies alprazolam explicitly as a 'full allosteric "
             "modulator' against bretazenil as the partial one; in "
             "[Norman2026]'s own Table 2 alprazolam's Emax of 2.84 sits "
             "inside the diazepam/clonazepam/lorazepam cluster (2.88-3.29) "
             "and far above bretazenil's 1.39; and in Xenopus oocytes "
             "expressing alpha1beta2gamma2S it gives a 98% increase in "
             "GABA-evoked chloride current with EC50 56 nM, abolished by "
             "flumazenil ([Sanna1999] PMID 10083975). The site is ALLOSTERIC "
             "and the structure says so directly: alprazolam occupies the "
             "alpha1+/gamma2- interface while GABA occupies the two "
             "beta3+/alpha1- interfaces ([Masiulis2019] Nature 565:454, PMID "
             "30602790, PDB 6HUO, 3.26 A cryo-EM).",
    ),
)

#: 131 targets tested in one panel, one hit. The cleanest selectivity
#: statement in this module, and the reason alprazolam is the compound whose
#: single-target prediction is least confounded.
ALPRAZOLAM_TESTED_INACTIVE = tuple(
    (target, 10000.0,
     "[DrugMatrix] NTP in vitro pharmacology panel, ChEMBL document "
     "CHEMBL1909046. Recorded ceiling, verbatim: 'Inhibition < 50% @ 10 uM "
     "and thus dose-response curve not measured.' 131 targets were tested "
     "and the GABA-A flunitrazepam site was the ONLY one returning a "
     "measurable Ki.")
    for target in (
        "D1", "D2", "5HT1A", "5HT2A", "alpha1", "alpha2", "beta", "H1",
        "M1", "M2", "alpha4beta2", "NMDA", "AMPA", "CB1", "A2A",
        "MOR", "DOR", "KOR", "DAT", "SERT", "NET",
    )
)

ALPRAZOLAM_NEVER_TESTED = (
    ("alpha7", "DrugMatrix's only nicotinic assay uses [125I]epibatidine, "
               "which is alpha4beta2-preferring and reports nothing about "
               "alpha7. No alpha-bungarotoxin assay for alprazolam exists in "
               "ChEMBL or PDSP. This is not-tested, which is a different "
               "fact from the twenty-one inactives above."),
)

ALPRAZOLAM_OCCUPANCY = (
    OccupancyObservation(
        "alprazolam", "GABA_A_BZD", "mouse", "IP", 0.90, occupancy=0.50,
        method="ex vivo competitive [3H]flunitrazepam binding; ED50 for 50% "
               "occupancy reported as 2.9 umol/kg",
        source="[Cho1986] J Med Chem 29:1346, PMID 3016261",
        note="The dose in mg/kg is THIS PROJECT'S ARITHMETIC: 2.9 umol/kg x "
             "308.76 g/mol = 0.90 mg/kg. The published quantity is the molar "
             "ED50. Route, strain and n are not recoverable from the "
             "accessible record; a parenteral route is likely because the "
             "paper concerns parenteral prodrugs, and is NOT confirmed. "
             "Recorded with those caveats rather than omitted, because it is "
             "the only numeric rodent occupancy value for this compound.",
    ),
)

ALPRAZOLAM_BEHAVIOUR = (
    BehaviouralObservation.from_open_arm_percent(
        vehicle_percent=19.7, vehicle_sd=8.57,
        drug_percent=20.9, drug_sd=11.89, n_vehicle=10, n_drug=10,
        compound="alprazolam", endpoint="anxiety", species="rat", route="IP",
        dose_mg_per_kg=0.1,
        measure="elevated plus maze, % time in open arms, 30 min "
                "pre-treatment; converted to open-arm avoidance",
        direction="no-change", reporting="tabulated",
        source="[Engel2018] Front Pharmacol 9:1225, PMID 30455643, "
               "PMC6230983, Table 1",
        note="Wistar or CD rats, 240-330 g; the paper does not state which "
             "strain went to which experiment. Unpaired t = 0.2589, df = 18, "
             "not significant. This dose was run against its OWN vehicle "
             "group in a two-group experiment, which is why the vehicle "
             "figures differ from the other two doses.",
    ),
    BehaviouralObservation.from_open_arm_percent(
        vehicle_percent=26.6, vehicle_sd=9.96,
        drug_percent=35.97, drug_sd=8.7, n_vehicle=10, n_drug=10,
        compound="alprazolam", endpoint="anxiety", species="rat", route="IP",
        dose_mg_per_kg=0.3,
        measure="elevated plus maze, % time in open arms; converted to "
                "open-arm avoidance",
        direction="decrease", reporting="tabulated",
        source="[Engel2018] PMID 30455643, Table 1",
        note="F(2,27) = 3.884, p < 0.05 for this dose. The only significant "
             "alprazolam effect in the series.",
    ),
    BehaviouralObservation.from_open_arm_percent(
        vehicle_percent=26.6, vehicle_sd=9.96,
        drug_percent=27.6, drug_sd=14.07, n_vehicle=10, n_drug=10,
        compound="alprazolam", endpoint="anxiety", species="rat", route="IP",
        dose_mg_per_kg=1.0,
        measure="elevated plus maze, % time in open arms; converted to "
                "open-arm avoidance",
        direction="no-change", reporting="tabulated",
        source="[Engel2018] PMID 30455643, Table 1",
        note="Not significant, and the paper notes sedative effects at this "
             "dose QUALITATIVELY. Together with the 0.3 mg/kg point this "
             "makes the series NON-MONOTONE in dose, which is the property "
             "this task's monotonicity test is about: every stage of the "
             "model from concentration to endpoint index is monotone, so no "
             "gain can reproduce a curve that rises and falls.",
    ),
    BehaviouralObservation(
        "alprazolam", "locomotion", "mouse", "PO", 0.94,
        measure="ED50 for inhibition of spontaneous motor activity, "
                "0.94 mg/kg p.o.",
        direction="decrease", reporting="text-only",
        source="[Hirai1982] J Med Chem 25:1466, PMID 7154007",
        note="An ED50 point estimate with no group means, no dispersion and "
             "no n - the reporting convention of the entire 1976-1982 Upjohn "
             "series, which is the bulk of the numeric alprazolam behavioural "
             "literature. Companion values from the same paper: rotarod 2.22, "
             "loss of righting-reflex potentiation 0.25-0.36 mg/kg p.o.",
    ),
)

ALPRAZOLAM_SEARCHES = (
    SearchRecord(
        what="rodent dose-to-concentration pharmacokinetics for alprazolam: a "
             "volume of distribution and a clearance in the same species and "
             "route",
        queries=(
            "alprazolam pharmacokinetics rat intravenous clearance volume of "
            "distribution",
            "alprazolam rat subcutaneous pharmacokinetics Vd CL half-life",
            "alprazolam mouse pharmacokinetics bioavailability",
        ),
        candidates=(
            ("[Canfield2023] Toxicol Appl Pharmacol 465:116459, PMID 36907383",
             "12 male Sprague-Dawley rats at 2 mg/kg SC with Vd, CL, t1/2, "
             "Cmax, Tmax and AUC all measured - exactly the parameter set "
             "needed. Nominally hybrid open access, and ScienceDirect, "
             "ResearchGate, the Elsevier text-mining API, Unpaywall, "
             "Semantic Scholar, scholar.archive.org and EPA HERO all "
             "returned 403 or no PDF. Only the qualitative comparative "
             "statements are citable."),
            ("[Wang1999] Drug Metab Dispos 27:855, PMID 10421610",
             "rat 1.25 mg/kg IV with biexponential decay and an absolute oral "
             "F of ~30%; the CL, Vd and t1/2 are in the paywalled body"),
            ("[Lau1997] J Pharmacol Exp Ther 281:1013, PMID 9190832 and "
             "[Lau1996] Psychopharmacology 126:115, PMID 8856830",
             "give a rat terminal half-life of 0.44 h and 32 min "
             "respectively, consistent with each other - but a half-life "
             "alone fixes only the elimination rate and NOT the volume, so "
             "it cannot turn a mg/kg dose into a concentration"),
            ("[Horinkova2020] Prague Med Rep 121:244, PMID 33270012",
             "fully open access and read in full: female Wistar, 20 mg/kg IP, "
             "n = 8 per group, tissue distribution as % of dose at 30 min. "
             "The brain figure is 0.32% of dose. Actual concentrations are in "
             "Figure 1 only, with no numeric table, so no brain:blood ratio "
             "can be taken from it."),
        ),
        conclusion="No usable dose-to-concentration model exists in any "
                   "accessible source. This is the decisive gap for "
                   "alprazolam and it is the exact mirror image of cocaine's: "
                   "cocaine has a complete rat IV disposition and no brain "
                   "partition, alprazolam has a complete brain partition and "
                   "no disposition. Neither compound can be simulated from a "
                   "dose, for opposite reasons.",
    ),
    SearchRecord(
        what="a fitted rodent dose-occupancy curve for the benzodiazepine "
             "site",
        queries=(
            "(triazolobenzodiazepine OR alprazolam) AND flumazenil AND PET "
            "AND (rat OR mouse OR rodent)",
            "alprazolam benzodiazepine receptor occupancy ex vivo mouse dose",
        ),
        candidates=(
            ("rodent PET or SPECT occupancy of alprazolam",
             "returns ZERO records. The mechanistic reason is published: "
             "[Dobbs1995] Nucl Med Biol 22:459, PMID 7550022, found "
             "[11C]alprazolam brain uptake is only ~1% of injected dose and "
             "that the compound's own pharmacological activity prevents "
             "administering enough mass for blockade."),
            ("[Sanger1993] Psychopharmacology 111:315, PMID 7870969",
             "its abstract confirms it contains per-region rat in vivo "
             "[3H]flumazenil inhibition potencies for nine compounds "
             "INCLUDING alprazolam; paywalled on every route tried"),
        ),
        conclusion="Only the single mouse ex vivo ED50 recorded above. A "
                   "further complication is documented and replicated: four "
                   "independent mouse studies ([Miller1987] PMID 2889803, "
                   "[Mosaddeghi1992] PMID 1331409, [Burke1994] PMID 7913233, "
                   "[Kaplan1990] PMID 2167493) report that LOW-dose "
                   "alprazolam paradoxically INCREASES in vivo "
                   "benzodiazepine-site radioligand binding rather than "
                   "displacing it, which [Miller1987] attributes to an "
                   "increase in receptor number. Any monotone occupancy "
                   "function fitted to alprazolam will be wrong below about "
                   "0.1 mg/kg IP in the mouse, and this model's occupancy "
                   "function is monotone by construction.",
    ),
)


# --- morphine --------------------------------------------------------------
#
# Morphine is the required opioid contrast to fentanyl.  Only the MOR entry
# is carried here: the model's opioid module can represent MOR, while the
# additional delta/kappa values in mixed-species panels would not improve a
# dose-to-behaviour claim and would make the provenance less clean.

MORPHINE_KI = (
    KiRecord(
        "MOR", ki_nM=1.1, species="human cloned receptor",
        preparation="CHO membrane competition binding",
        radioligand="[3H]DAMGO",
        source="[Toll1998] NIDA Res Monogr 178:440-466, PMID 9686407",
        note="Table 5, human cloned MOR. This is kept separate from the "
             "rat-brain values in later panels rather than averaging across "
             "species and preparations.",
    ),
)

MORPHINE_EFFICACY = (
    EfficacyRecord(
        "MOR", efficacy=0.625, kind="agonist",
        assay="agonist-stimulated [35S]GTP-gamma-S binding in rat thalamic "
              "membranes",
        source="[Selley1997] Mol Pharmacol 51:87-96, PMID 9016350",
        note="Derived midpoint of the source's reported 55-70% of DAMGO "
             "maximal stimulation. This is a range summary, not a claim "
             "that the assay measured 0.625 exactly.",
    ),
)

MORPHINE_SEARCHES = (
    SearchRecord(
        what="a receptor-drivable rat brain partition Kp,uu for morphine",
        queries=(
            "morphine rat brain plasma Kp,uu unbound brain fraction",
            "morphine unbound brain concentration rat microdialysis plasma",
        ),
        candidates=(
            ("[Kalvass2007] JPET 323:346-355, PMID 17646430",
             "reports unbound brain concentrations for several opioids in "
             "mice, not morphine in rat; it cannot be substituted across "
             "species"),
        ),
        conclusion="No species-matched, receptor-drivable morphine brain "
                   "partition was obtained.",
    ),
    SearchRecord(
        what="a sourced rat dose-to-concentration PK model for morphine",
        queries=(
            "morphine rat intravenous pharmacokinetics Vd clearance dose",
            "morphine rat intraperitoneal pharmacokinetics absorption",
        ),
        candidates=(),
        conclusion="No route in drug_pk.py has a complete morphine parameter "
                   "set, so fentanyl or another opioid must not be reused.",
    ),
    SearchRecord(
        what="a quantitative rodent morphine dose-response with a matched "
             "vehicle group",
        queries=(
            "morphine rat analgesia dose response vehicle group tail flick "
            "tabulated",
        ),
        candidates=(
            ("the endpoint library morphine papers",
             "they establish region-level mechanism or use local doses, "
             "not a systemic dose-response that closes this model's chain"),
        ),
        conclusion="Behavioural evidence exists, but no observation is added "
                   "here until dose, route, assay and dispersion are matched.",
    ),
)


# --- clozapine -------------------------------------------------------------
#
# Clozapine is the required antipsychotic contrast.  The single D2 record is
# deliberately conservative: broader PDSP panels mix species, radioligands
# and assay conditions, while this task needs an auditable target join.

CLOZAPINE_KI = (
    KiRecord(
        "D2", ki_nM=75.0, species="human cloned receptor",
        preparation="recombinant receptor binding",
        radioligand="dopamine-receptor radioligand",
        source="[Seeman2013] J Recept Signal Transduct Res 33:1-5, "
               "PMC3894721",
        note="The review's Table 1 and discussion identify approximately "
             "75 nM as the appropriate clozapine D2 Ki; the fast-off "
             "assay context is retained rather than mixed with a different "
             "radioligand's lower apparent value.",
    ),
)

CLOZAPINE_EFFICACY = (
    EfficacyRecord(
        "D2", efficacy=0.0, kind="inverse_agonist",
        assay="constitutive-activity inhibition in a G-protein-coupled D2 "
              "functional assay",
        source="[Burstein2005] Eur J Pharmacol 519:1-13, PMID 16135699",
        note="Clozapine itself was classified as an inverse agonist at D2; "
             "the efficacy magnitude is intentionally not used for a "
             "behavioural fit in this held-out record, only the functional "
             "class and sign.",
    ),
)

CLOZAPINE_SEARCHES = (
    SearchRecord(
        what="a receptor-drivable rat brain partition Kp,uu for clozapine",
        queries=(
            "clozapine rat brain plasma Kp,uu unbound brain fraction",
            "clozapine unbound brain concentration rat microdialysis plasma",
        ),
        candidates=(),
        conclusion="No species-matched, receptor-drivable clozapine brain "
                   "partition was obtained.",
    ),
    SearchRecord(
        what="a sourced rat dose-to-concentration PK model for clozapine",
        queries=(
            "clozapine rat intravenous pharmacokinetics Vd clearance dose",
            "clozapine rat intraperitoneal pharmacokinetics absorption",
        ),
        candidates=(),
        conclusion="No route in drug_pk.py has a complete clozapine parameter "
                   "set; a human clinical PK value cannot be substituted.",
    ),
    SearchRecord(
        what="a quantitative rodent clozapine dose-response linked to the "
             "modelled D2 endpoint",
        queries=(
            "clozapine rat locomotion dose response D2 receptor vehicle "
            "tabulated",
        ),
        candidates=(),
        conclusion="No matched quantitative observation was added before the "
                   "dose-to-brain chain can be closed.",
    ),
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def _compound(name, mechanism_class, prefix, note, release=None, **extra):
    """Assemble one `HeldOutCompound` from this module's own constants.

    Takes the constants by name rather than by value so that a block defined
    above and never wired in here fails loudly at import instead of being
    silently absent from the registry.
    """
    globals_ = globals()
    return HeldOutCompound(
        name=name,
        mechanism_class=mechanism_class,
        ki=globals_["%s_KI" % prefix],
        efficacy=globals_["%s_EFFICACY" % prefix],
        tested_inactive=globals_.get("%s_TESTED_INACTIVE" % prefix, ()),
        never_tested=globals_.get("%s_NEVER_TESTED" % prefix, ()),
        occupancy_observations=globals_.get("%s_OCCUPANCY" % prefix, ()),
        behavioural_observations=globals_.get("%s_BEHAVIOUR" % prefix, ()),
        searches=globals_.get("%s_SEARCHES" % prefix, ()),
        note=note,
        **extra
    )


HELD_OUT = {}


def _register(compound):
    if compound.name in HELD_OUT:
        raise ValueError("%s is already registered" % compound.name)
    HELD_OUT[compound.name] = compound
    return compound


_register(_compound(
    "ketamine", "NMDA open-channel blocker", "KETAMINE",
    "The only compound in the set whose whole-brain mechanism gain is not "
    "free: glutamate.GLUTAMATE_CONNECTION_GAIN is pinned at 1.0 by algebra, "
    "so its behavioural magnitude is a prediction with no fitted parameter "
    "anywhere in it.",
))

_register(_compound(
    "caffeine", "adenosine receptor antagonist", "CAFFEINE",
    "The cleanest test of the endogenous-tone question. A2A is modelled and "
    "P6.2 measured no basal adenosine, so this model predicts caffeine has "
    "exactly no behavioural effect at any dose and at any gain.",
))

_register(_compound(
    "cocaine", "monoamine uptake blocker", "COCAINE",
    "Has a complete rat intravenous disposition and no brain unbound "
    "fraction: the exact mirror of alprazolam.",
))

_register(_compound(
    "amphetamine", "monoamine substrate releaser", "AMPHETAMINE",
    "The controlled contrast with cocaine - same three targets, same "
    "laboratory, opposite mechanism - which is a distinction this model can "
    "represent at the transporter and cannot carry to a behavioural endpoint.",
))

_register(_compound(
    "alprazolam", "GABA-A benzodiazepine-site positive allosteric modulator",
    "ALPRAZOLAM",
    "The only compound with a complete measured plasma-to-brain-unbound "
    "partition, and the only one with a tabulated multi-dose behavioural "
    "series with a vehicle group. It has no dose-to-concentration model, "
    "which is the mirror of cocaine's gap.",
))

_register(_compound(
    "morphine", "mu-opioid receptor agonist", "MORPHINE",
    "The required opioid held-out contrast to fentanyl. Its MOR binding and "
    "functional class are sourced, but no species-matched PK/brain-partition "
    "chain is silently borrowed from fentanyl.",
))

_register(_compound(
    "clozapine", "D2 inverse agonist", "CLOZAPINE",
    "The required antipsychotic held-out contrast. The D2 binding and inverse-"
    "agonist classification are sourced, but the model refuses to invent "
    "rodent PK, brain partition or a behavioural magnitude.",
))


#: The drugs this project HAS fitted against. Not restated as a literal: it
#: is read off the evidence module, so a drug that acquires a fitting target
#: later is picked up automatically and the disjointness test starts failing
#: instead of quietly passing.
#:
#: Since P11.3 that module is the QUARANTINED one: every dose-to-behaviour
#: dataset this project ever fitted lived in `calibration.behavioral_
#: evidence` until the species pivot moved it to `legacy/`. Reading the set
#: off the quarantined copy keeps the definition historical-and-complete -
#: these are the drugs the retired bridge actually touched - while a future
#: HUMAN behavioural dataset will re-enter through a production module and
#: must be added here explicitly when it does.
def training_compounds():
    """The set of drug names any calibration in this repository has used."""
    from .legacy import behavioral_evidence as _evidence

    return {entry.drug for entry in _evidence.BEHAVIORAL_EVIDENCE}


def held_out_compounds():
    return dict(HELD_OUT)


__all__ = [
    "HELD_OUT",
    "REPORTING_KINDS",
    "BehaviouralObservation",
    "EfficacyRecord",
    "HeldOutCompound",
    "KiRecord",
    "OccupancyObservation",
    "SearchRecord",
    "held_out_compounds",
    "training_compounds",
]
