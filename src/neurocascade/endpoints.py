"""P7.1: named behavioural endpoints as multi-region readouts.

    from neurocascade import endpoints, connectome

    ont = connectome.load_ontology(path=...)
    built = connectome.build_circuit_network(strength, voxel_counts=vox,
                                             resolution="regions")
    ep = endpoints.ENDPOINTS["locomotion"]
    resolved = ep.resolve(built.network, ont)
    print(resolved.report())
    score = resolved.readout.score(built.network, rates)

`behavior.BehaviorReadout` has existed since v0 and is unchanged: a name and
a `{node name: weight}` dict, scored as `sum_i w_i r_i`. What did not exist
was any *named* readout with a *reason* for its weights. Every readout in
`scenarios.py` and the Phase 5 modules is one or two hand-picked nodes with a
weight of 1.0 and a comment saying it is illustrative. This module is the
library the roadmap's Phase 7 asks for: seven endpoints, each a readout over
several Allen-CCF regions, each region placed in its endpoint by a cited
experiment in a rodent. Across the 69 terms the techniques are optogenetic
(31), chemogenetic (11), microinjection (11), lesion (7), inactivation (5),
electrical stimulation (2) and genetic (2).

**No term is an imaging term**, and that is a finding rather than a
preference. The brief asks for "lesion or imaging evidence"; sweeps for a
brain-wide c-Fos or light-sheet activation map of any of these seven assays
came back empty (SEARCHED_AND_EMPTY records them). `technique="imaging"` is
accepted and `ResolvedEndpoint.report()` counts correlative terms
separately, so the machinery is there and reads zero - which is the honest
state of it, not dead code to be deleted.

What is cited and what is illustrative, stated once
---------------------------------------------------

This is the load-bearing distinction in the whole module, so it is worth
being blunt about it.

**Cited:** which regions belong in the readout, and the SIGN of each. Every
term below carries a real experiment whose direction is unambiguous - a
lesion that lowered the measure means the region's activity raises it, an
optogenetic activation that raised the measure means the same thing more
directly. The citation is on the term, not on the endpoint, so it is visible
which experiment put each region there.

**Illustrative:** the MAGNITUDE of every weight. No lesion study reports a
linear decoder coefficient, and no combination of them does either: a
paper that says "NAc lesion abolished amphetamine hyperlocomotion" tells you
the sign and nothing about how a 0.1 change in NAc firing rate trades off
against a 0.1 change in caudoputamen firing rate. So every term's magnitude
is 1 before normalisation, i.e. **the regions in an endpoint are weighted
EQUALLY**, and `Endpoint.weights_are_illustrative` is True for all seven.
P7.3 is the task that fits the magnitudes against real dose-response curves
and is the only task allowed to remove the label.

Equal weighting is a choice and not the absence of one, so here is what it
asserts and what it costs. It asserts that the evidence identifies a set and
a direction and nothing finer, which is true of this literature. It costs
the obvious thing: a region whose lesion abolishes the behaviour and a region
whose lesion shifts it by 20% enter with the same coefficient. The
alternative - reading effect sizes out of the papers and calling those
weights - would be worse, because those effect sizes are measured in
different assays, in different species, against different manipulations
(a fibre-sparing excitotoxic lesion and a 10 Hz optogenetic train are not
the same perturbation), and combining them into one linear decoder would be
inventing a number out of numbers that were never commensurable. Recording
the direction and refusing the magnitude is the honest form of what this
literature supports.

Sign convention, and the one inference it makes
-----------------------------------------------

`sign=+1` means "more activity in this region, more of the measured
quantity"; `sign=-1` means the opposite. **No term writes a sign down.**
`EndpointTerm` takes two fields - `activity`, what the cited experiment did
to the region's firing, and `measure`, what happened to this endpoint's
score - and derives::

    sign = +1 if activity == measure else -1

which reads, in the four cases::

    lesion/inhibition LOWERS the measure   ->  sign = +1
    lesion/inhibition RAISES the measure   ->  sign = -1
    activation RAISES the measure          ->  sign = +1
    activation LOWERS the measure          ->  sign = -1

Passing `sign=` raises. The indirection exists because the sign of a lesion
result is the one thing here that is easy to get backwards through fatigue
rather than ignorance: "the lesion reduced open-arm time" and "the lesion
reduced anxiety" are opposite claims about the same experiment and differ by
one polarity conversion. Forcing that conversion to happen once, in a
machine-readable field, lets the tests re-derive every sign in the module
instead of trusting it.

The inference the convention makes is that a region's contribution is
monotone in its population firing rate, which is what a linear readout means
and is exactly the assumption P7.2 exists to relax where data supports it.
It is a real assumption and it is wrong in named cases, which are recorded
rather than hidden: the rostral ventromedial medulla reverses sign with its
own drive within one paper and one assay ([ZhuoGebhart1997]), and accumbens
and ventral hippocampus are inverted-U on prepulse inhibition. A term whose
direction is genuinely contested carries `contested=True`, is listed in the
report, and is EXCLUDED from the default readout. Passing
`include_contested=True` puts it back, so a test can measure what it is
worth instead of the module asserting that it is worth nothing.

`contested` is deliberately narrow: it means TWO studies manipulating the
REGION reported opposite directions for THIS endpoint. Sub-population
heterogeneity alone does not qualify, because every region has cell types
that differ and a flag that fires on all 69 terms carries no information;
that goes in the term's `note`.

Resolution: an endpoint does not know what parcellation it will be read at
-----------------------------------------------------------------------------

`connectome/build.py` builds at "regions" (291 Knox rows), "subdivisions"
(60) or "major_divisions" (12), and the same seven endpoints have to work at
all three. A term names CCF structures; `resolve()` maps each structure onto
whichever node the network actually has, by exactly four rules, and reports
which rule fired for every term:

1. **The structure IS a node.** Full term weight on it. (`"exact"`)
2. **An ancestor is a node.** The nearest such ancestor carries the full
   term weight: at `major_divisions` the ACB term lands on STR. (`"ancestor"`)
3. **Descendants are nodes.** The term weight is SPLIT across them, in
   proportion to `region_sizes` when given and equally otherwise. The split
   is the right arithmetic and not a convenience: a parent region's mean rate
   is the volume-weighted mean of its children's rates, so
   `w * r_parent = sum_i (w * v_i / V) * r_i`. With no sizes supplied the
   equal split is an extra illustrative assumption and is reported as one.
   (`"descendants"`)
4. **Nothing.** The term is dropped and named in the report. (`"unmapped"`)

Collapse, and why the default is the mean
-----------------------------------------

At a coarse parcellation several terms land on one node - SNr, CUN, PPN, SNc
and VTA all become MB. The default `collapse="mean"` gives that node the MEAN
of the terms that landed on it, so an endpoint does not silently reweight
itself towards whichever region the CCF happens to subdivide most finely.
`collapse="sum"` is available and is the other defensible reading (each
experiment is independent evidence), and `tests/test_endpoints.py` runs both.

**A term's own structures are a different question and sum rather than
average.** [Meyers2006]'s manipulation covers the dorsal hippocampus, which
is CA1 + CA3 + DG; that is ONE unit of evidence, not three, so the term's
weight is split 1/N across its structures and those parts sum back within
the term BEFORE the mean/sum rule is applied across terms. Get the order
wrong in either direction and the hippocampus either outweighs a
single-structure term threefold at fine resolution or arrives at a third of
its weight at coarse resolution. Both directions have a test.

When terms of OPPOSITE sign land on the same node they cancel, under either
rule. That is not a bug to be papered over: it is the parcellation saying it
is too coarse to express this endpoint, and `ResolvedEndpoint.cancellations`
names every node where it happened so the caller can see it rather than read
a quietly deflated score.

The mean rule has a second consequence, in the opposite direction from the
one it was chosen for, and it is worth stating because it is surprising.
Mean collapse followed by L1 normalisation makes a coarse readout
ANTI-correlated with evidence density: at `major_divisions`, locomotion's MB
absorbs five of the eight active terms and ends up with the SMALLEST weight
(+0.167) while HY, STR and MY carry one term each at +0.278, because MB's
five terms partly cancel and are then averaged. Both effects are real and
`collapse="sum"` trades one for the other. Neither is a reason to prefer a
coarse parcellation, which is the actual conclusion.

Normalisation, so that seven endpoints are on one scale
-------------------------------------------------------

After resolution the weights are scaled so that `sum_i |w_i| == 1`. A
Wilson-Cowan rate is in [0, 1], so a normalised score is in [-1, 1] and the
seven endpoints are directly comparable. Normalising AFTER resolution rather
than before means a dropped or unmapped term does not shrink the score of
the terms that survived; it means the score is always a weighted mean over
the regions the network can actually see. `normalize=False` returns the raw
+/-1 magnitudes for a caller who wants to do their own scaling.

The resting score is NOT zero and is not meant to be. Every node of a
freshly built network rests at 0.5, so `baseline_score()` is
`0.5 * sum_i w_i`, which is zero only for an endpoint whose positive and
negative weights happen to balance. What a drug does to an endpoint is
`score - baseline_score`, and `ResolvedEndpoint.delta()` is that
subtraction, because a difference from the drug-free run is the only thing
this readout can mean before P7.3 fits an absolute scale.

Seven endpoints, and two views of one of them
---------------------------------------------

`ENDPOINTS` holds exactly the seven the roadmap names and a test asserts the
count. `ASSAY_VARIANTS` holds two RESTRICTIONS of `analgesia`, built by
`Endpoint.restrict_to_assay`, and they are views rather than an eighth
endpoint: every term in them is a term of `analgesia`.

They exist because **TWO** papers ran both assays in one preparation and
found them to dissociate, in opposite manipulation directions:
[Pastoriza1996] (rat, medial frontal lesion: hot-plate latency +82%, tail
flick unchanged) and [Hao2025] (mouse, retrosplenial-to-cingulate
activation: thermal responses facilitated, "did not influence spinal
nociceptive responses in the tail-flick test"). Two more dissociate
tail flick from a DIFFERENT supraspinal measure without running hot plate -
[Roberts1994] (vocalization threshold down, tail flick unchanged) and
[MorganFranklin1990] (formalin analgesia abolished, tail flick intact) -
which supports the same spinal/supraspinal split without being evidence
about hot plate.

That count is smaller than the one first written here, and the correction
is worth keeping: the draft claimed six papers and included [Xie2023] and
[Samineni2017], which ran NEITHER assay. This module's own citation entry
for [Samineni2017] says so verbatim. Two papers, not six, and the
structural decision still follows: one shared weight vector is contradicted
by [Pastoriza1996] and [Hao2025] directly, and averaging over them would be
the substitution rule 2 forbids. So the module keeps the union endpoint the
roadmap asks for and publishes the split next to it. The two select
genuinely different regions: only the supraspinal assay reaches the
cingulate, only the spinal one reaches the submedial thalamus.

`EndpointTerm.assays` is a tuple of tokens matched by EXACT membership,
not prose matched as a substring, and that is not fussiness. Under
substring matching, ACAd's assay - "hot plate, tail flick explicitly
unchanged" - joined the tail-flick restriction with its sign intact, which
is the one term that most sharply does not belong there. Negations do not
survive substring matching. Caveats live in `note`.

What this module deliberately does not do
-----------------------------------------

It does not fit anything, it does not convert a score into an assay unit
(seconds of tail-flick latency, centimetres travelled, percent PPI), and it
does not predict a dose-response curve. All three are P7.3. It does not
introduce a nonlinearity; that is P7.2. And it does not touch
`behavior.BehaviorReadout`, `scenarios.py` or any Phase 5 module: the
existing one- and two-node readouts are still there and still what the
existing tests exercise, and `resolve()` returns a plain `BehaviorReadout`
so everything that already scores one keeps working.

It also does not read out the hand-built scenarios. Their nodes are
populations ("VTA_DA"), not CCF regions, and `resolve` ignores any node name
the ontology cannot resolve - which a test pins, because silently matching
"VTA_DA" to VTA would be worse than dropping it.
"""

from .behavior import BehaviorReadout


#: Every weight magnitude in this module is illustrative pending P7.3. The
#: string is the provenance tag carried on `Endpoint.weight_provenance` and
#: is asserted by `tests/test_endpoints.py` for all seven endpoints.
ILLUSTRATIVE_WEIGHTS = "illustrative-equal-magnitude"

#: The provenance tag for the extra assumption rule 3 makes when it has no
#: region sizes to split a term across descendants with.
ILLUSTRATIVE_EQUAL_SPLIT = "illustrative-equal-split"

VALID_COLLAPSE = ("mean", "sum")

#: Below this a weight is arithmetic residue rather than a contribution.
#: Weights here are O(1/number of terms), so 1e-12 is many orders below
#: anything meaningful and far above double-precision cancellation error.
_TINY = 1e-12

#: The evidence classes a term may cite, strongest first. This is an
#: ordering of how directly the experiment establishes that the region's
#: activity drives the measure, and `EndpointTerm` refuses anything else so
#: that "c-Fos went up" cannot be recorded as though it were a lesion.
EVIDENCE_TECHNIQUES = (
    "lesion",             # permanent excitotoxic/electrolytic/neurotoxic
    "inactivation",       # reversible SILENCING and nothing else:
                          # muscimol, lidocaine, TTX, cooling. The dividing
                          # line against `microinjection` below is whether
                          # the agent's effect on the region's firing needs
                          # an argument. Muscimol alone does not; muscimol
                          # WITH bicuculline, or with morphine, does, and
                          # those terms are `microinjection`.
    "chemogenetic",       # DREADD activation or inhibition
    "optogenetic",        # ChR2/halorhodopsin/Arch
    "genetic",            # constitutive or conditional deletion or
                          # knockdown of a gene in a genetically defined
                          # population, region-defining or not
    "microinjection",     # drug into the region, behaviour out. The ONLY
                          # technique here whose `activity` field is an
                          # INFERENCE rather than the manipulation itself:
                          # muscimol and bicuculline are unambiguous, but
                          # "opioid into vlPAG" means "more vlPAG output"
                          # only via disinhibition, and "D1 agonist into
                          # accumbens" has no unambiguous reading at all
                          # because D1 and D2 receptors sit on populations
                          # with opposite output signs. Every term where
                          # the inference is not straightforward says so
                          # in its `finding` or its `note`.
    "stimulation",        # electrical
    "imaging",            # c-Fos, fMRI, photometry: CORRELATIVE
)

#: The techniques that are correlative rather than causal. A term citing one
#: of these is still a term: a region that is reliably active during the
#: behaviour belongs in the readout. It fixes the direction less firmly than
#: a lesion does, so `ResolvedEndpoint.report()` counts the two separately.
CORRELATIVE_TECHNIQUES = frozenset(["imaging"])


DIRECTIONS = ("increase", "decrease")


class EndpointTerm(object):
    """One region's place in an endpoint, and the experiment that put it there.

    structures: CCF acronyms this term is about. More than one only when the
        anatomical entity the paper names is more than one CCF structure
        (the tuberomammillary nucleus is TMd + TMv; "hippocampus" is CA1 +
        CA3 + DG).
    activity: "increase" or "decrease" - what the cited experiment did to
        this region's activity (an excitotoxic lesion is "decrease", a ChR2
        train is "increase"), or for a correlative technique, the direction
        of the observed activity change.
    measure: "increase" or "decrease" - what happened to THIS ENDPOINT'S
        SCORE as the endpoint's own `direction` string defines it. The
        polarity conversion from the assay's units (open-arm seconds,
        tail-flick latency, percent PPI) into the endpoint's score happens
        exactly once, here, in the open.

    The SIGN IS DERIVED from those two: `+1` when they agree, `-1` when they
    disagree. Nothing writes a sign down by hand, which is why
    `tests/test_endpoints.py` can re-derive every one of them.

    citation: a key into `CITATIONS`.
    species: "mouse" or "rat", as reported by the cited paper. The model is
        mouse (roadmap); a rat term is a rat term and says so.
    technique: one of `EVIDENCE_TECHNIQUES`.
    assay: the behavioural test(s) the CITED experiment actually ran, as a
        string or a tuple of them, drawn from a controlled vocabulary
        rather than written as prose. This is not decoration. The modern
        circuit literature has largely migrated to Hargreaves radiant heat
        and von Frey, so a region can have excellent causal evidence in an
        assay this endpoint is not defined against, and the honest thing is
        to record which. `Endpoint.restrict_to_assay` matches these by
        EXACT membership, which is why they are tokens and not sentences:
        a prose caveat like "(not tail flick)" would otherwise match a
        restriction to tail flick. Caveats belong in `note`.
    finding: the paper's result, in its own direction, in one or two
        sentences.
    contested: True when TWO studies manipulating this REGION report
        opposite directions for this endpoint. Sub-population heterogeneity
        alone is not enough - every region has cell types that differ, and
        flagging that as contested would flag everything. It goes in `note`.
        A contested term is excluded from the default readout and kept in
        the record.
    inferred: a short string naming what had to be REASONED rather than
        measured to fill in `activity` or `measure`, or None when both come
        straight off the experiment. Eight of the 69 terms carry one, and
        they are the eight a reader should discount first:

        * the manipulation was one synapse away from the region (anxiety
          SUB, gating PRNc);
        * a drug's effect on the region's firing had to be reasoned
          (analgesia PAG and DR from opioid disinhibition; gating PL/ILA
          from a D1 agonist; gating ACB from dopamine, which has no
          unambiguous direction at all);
        * the assay was not one this endpoint is defined against, so
          `measure` is a translation (analgesia PB, MD).

        `ResolvedEndpoint.report()` counts them, so the number is visible
        next to the weights rather than buried in a note.
    """

    def __init__(self, structures, activity, measure, citation, species,
                 technique, assay, finding, contested=False, note=None,
                 inferred=None):
        if isinstance(structures, str):
            structures = (structures,)
        structures = tuple(structures)
        if not structures:
            raise ValueError("an EndpointTerm needs at least one structure")
        if activity not in DIRECTIONS or measure not in DIRECTIONS:
            raise ValueError(
                "activity and measure must each be one of %s, got %r and %r"
                % (list(DIRECTIONS), activity, measure)
            )
        if technique not in EVIDENCE_TECHNIQUES:
            raise ValueError(
                "technique must be one of %s, got %r"
                % (list(EVIDENCE_TECHNIQUES), technique)
            )
        if not citation:
            raise ValueError(
                "every term needs a citation key; a region with no cited "
                "experiment does not belong in an endpoint"
            )
        if not finding:
            raise ValueError(
                "every term needs the finding written out, so that the sign "
                "can be checked against the experiment rather than trusted"
            )
        if isinstance(assay, str):
            assay = (assay,)
        assay = tuple(assay)
        if not assay:
            raise ValueError(
                "every term needs the assay the cited experiment ran; see "
                "the class docstring for why it is not decoration"
            )
        self.structures = structures
        self.activity = activity
        self.measure = measure
        self.sign = 1 if activity == measure else -1
        self.citation = citation
        self.species = species
        self.technique = technique
        self.assays = assay
        self.finding = finding
        self.contested = bool(contested)
        self.note = note
        self.inferred = inferred

    @property
    def assay(self):
        """The assay tokens as one display string."""
        return ", ".join(self.assays)

    def ran_assay(self, needle):
        """Exact, case-insensitive membership in `assays`."""
        needle = needle.lower()
        return any(a.lower() == needle for a in self.assays)

    @property
    def is_correlative(self):
        return self.technique in CORRELATIVE_TECHNIQUES

    def __repr__(self):
        return "EndpointTerm(%s, sign=%+d, %s)" % (
            "+".join(self.structures), self.sign, self.citation
        )


class Endpoint(object):
    """A named behavioural endpoint and the regions that read it out.

    key: the stable identifier, e.g. "locomotion".
    name: human-readable.
    assay: the behavioural assay(s) the endpoint is defined against.
    direction: what a HIGHER score means, spelled out. Every endpoint in
        this module has one and a test asserts it is non-empty, because a
        readout whose polarity is not written down is a sign error waiting
        to happen.
    terms: the `EndpointTerm`s.
    """

    weight_provenance = ILLUSTRATIVE_WEIGHTS
    weights_are_illustrative = True

    def __init__(self, key, name, assay, direction, terms, notes=None):
        self.key = key
        self.name = name
        self.assay = assay
        self.direction = direction
        self.terms = tuple(terms)
        self.notes = notes
        if len(self.regions()) < 2:
            raise ValueError(
                "%r resolves to fewer than two distinct CCF structures; the "
                "point of P7.1 is that an endpoint is a MULTI-region readout"
                % (key,)
            )

    def regions(self):
        """Every CCF acronym named by any term, in first-appearance order."""
        seen = []
        for term in self.terms:
            for acronym in term.structures:
                if acronym not in seen:
                    seen.append(acronym)
        return tuple(seen)

    def citations(self):
        """The citation keys used, in first-appearance order."""
        seen = []
        for term in self.terms:
            if term.citation not in seen:
                seen.append(term.citation)
        return tuple(seen)

    def active_terms(self, include_contested=False):
        if include_contested:
            return self.terms
        return tuple(t for t in self.terms if not t.contested)

    def check_species(self, target_species):
        """Raise unless every term's provenance species is `target_species`.

        The endpoint layer's face of the P11.2 species gate, built to the
        same contract as `BrainPartition.check_species`: a parameter (here,
        a term) whose cited experiment ran in another species may not drive
        a readout for this one. Every term in this module cites mouse or rat
        work -- that is what P7.1 had -- so a human target raises today,
        naming the species and the terms; the human endpoint library is
        P17's build, and until it exists nothing can quietly resolve a mouse
        lesion onto a human network by passing `target_species=` below.
        """
        from .species_gate import UnsourcedParameterError, normalize_species

        target = normalize_species(target_species)
        if target is None:
            raise ValueError("target_species is required; there is no "
                             "species-neutral prediction to license")
        offenders = [
            term for term in self.terms
            if normalize_species(term.species) not in (None, target)
        ]
        if offenders:
            by_species = {}
            for term in offenders:
                by_species.setdefault(normalize_species(term.species), [])\
                    .append("+".join(term.structures))
            detail = "; ".join(
                "%s: %s" % (species, ", ".join(sorted(structures)))
                for species, structures in sorted(by_species.items())
            )
            raise UnsourcedParameterError(
                "endpoint %r: %d of %d terms were established in another "
                "species than %r (%s), so they cannot drive a %r readout. "
                "The human endpoint library is Phase 17's build; substituting"
                " another species' experiment for it is rule 2 of this "
                "codebase."
                % (self.key, len(offenders), len(self.terms), target, detail,
                   target)
            )

    def resolve(self, network, ontology, region_sizes=None, collapse="mean",
                normalize=True, include_contested=False, target_species=None):
        """Map this endpoint onto whatever nodes `network` actually has.

        network: a `CircuitNetwork`, or any iterable of node-name strings.
        ontology: a `connectome.StructureOntology`.
        region_sizes: optional {acronym: size} used to split a term across
            descendant nodes (rule 3). Voxel counts are the natural choice.
        target_species: optional species of the brain being predicted for.
            When given, the P11.2 species gate runs first and refuses any
            term whose cited experiment was not performed in that species.
            The existing rodent machinery omits it and is unchanged; the
            future human consumer must pass it.
        """
        if target_species is not None:
            self.check_species(target_species)
        return _resolve(
            self, network, ontology, region_sizes=region_sizes,
            collapse=collapse, normalize=normalize,
            include_contested=include_contested,
        )

    def restrict_to_assay(self, needle, key=None, name=None, assay=None):
        """A copy of this endpoint keeping only terms whose assay matches.

        `needle` must EQUAL one of a term's assay tokens, case-insensitively.
        This exists for one reason, recorded in full in
        docs/architecture.md: six independent papers dissociate tail-flick
        from hot-plate in BOTH directions, so a single shared weight vector
        for "analgesia" is contradicted by all six. `ASSAY_VARIANTS` holds
        the two restrictions this module builds.
        """
        terms = [t for t in self.terms if t.ran_assay(needle)]
        return Endpoint(
            key=key or "%s_%s" % (self.key, needle.replace(" ", "_")),
            name=name or "%s (%s only)" % (self.name, needle),
            assay=assay or needle,
            direction=self.direction,
            terms=terms,
            notes=(
                "Restricted from %r to the terms whose CITED experiment ran "
                "this assay. %s" % (self.key, self.notes or "")
            ).strip(),
        )

    def summary(self):
        lines = [
            "%s (%s)" % (self.name, self.key),
            "  assay:     %s" % self.assay,
            "  direction: %s" % self.direction,
            "  weights:   %s (all magnitudes equal; P7.3 fits them)"
            % self.weight_provenance,
        ]
        for term in self.terms:
            lines.append(
                "  %-14s %+d  %-15s %-5s %s%s"
                % (
                    "+".join(term.structures),
                    term.sign,
                    term.technique,
                    term.species,
                    term.citation,
                    "  [CONTESTED]" if term.contested else "",
                )
            )
            lines.append("                    assay: %s" % term.assay)
            if term.inferred:
                lines.append(
                    "                    INFERRED DIRECTION: %s"
                    % term.inferred
                )
        if self.notes:
            lines.append("  note: %s" % self.notes)
        return "\n".join(lines)

    def __repr__(self):
        return "Endpoint(%r, %d terms, %d regions)" % (
            self.key, len(self.terms), len(self.regions())
        )


class TermPlacement(object):
    """Where one term's structures ended up, and by which rule."""

    def __init__(self, term, structure, rule, nodes, weights):
        self.term = term
        self.structure = structure
        self.rule = rule            # exact | ancestor | descendants | unmapped
        self.nodes = tuple(nodes)
        self.weights = tuple(weights)

    def __repr__(self):
        return "TermPlacement(%s -> %s, %s)" % (
            self.structure, ",".join(self.nodes) or "-", self.rule
        )


class ResolvedEndpoint(object):
    """An `Endpoint` bound to one network's nodes.

    readout: a plain `behavior.BehaviorReadout`, so everything downstream
        that already knows how to score one keeps working.
    """

    def __init__(self, endpoint, readout, placements, weights, collapse,
                 normalize, include_contested, cancellations, equal_splits):
        self.endpoint = endpoint
        self.readout = readout
        self.placements = tuple(placements)
        self.weights = dict(weights)
        self.collapse = collapse
        self.normalize = normalize
        self.include_contested = include_contested
        self.cancellations = tuple(cancellations)
        self.equal_splits = tuple(equal_splits)

    @property
    def unmapped(self):
        return tuple(
            p.structure for p in self.placements if p.rule == "unmapped"
        )

    @property
    def mapped_nodes(self):
        return tuple(sorted(self.weights))

    @property
    def mapped_terms(self):
        """The terms that actually reached a node of this network.

        `Endpoint.active_terms()` is what was offered; this is what landed.
        The report counts evidence over THIS, because "8 causal terms" next
        to four nodes would describe the library rather than the readout.
        """
        landed = []
        for placement in self.placements:
            if placement.rule != "unmapped" and placement.term not in landed:
                landed.append(placement.term)
        return tuple(landed)

    def score(self, network, rates):
        return self.readout.score(network, rates)

    def baseline_score(self, resting_rate=0.5):
        """The score with every node at `resting_rate`.

        A freshly built `connectome` network rests at 0.5 everywhere, so
        this is the number a drug effect is measured against.
        """
        return resting_rate * sum(self.weights.values())

    def delta(self, network, rates, baseline_rates=None):
        """Change in score relative to a baseline.

        `baseline_rates=None` uses the flat resting network (every node at
        0.5). Pass the drug-free run's rates for anything else.
        """
        if baseline_rates is None:
            base = self.baseline_score()
        else:
            base = self.readout.score(network, baseline_rates)
        return self.readout.score(network, rates) - base

    def report(self):
        ep = self.endpoint
        landed = self.mapped_terms
        n_causal = sum(1 for t in landed if not t.is_correlative)
        n_corr = sum(1 for t in landed if t.is_correlative)
        n_inferred = sum(1 for t in landed if t.inferred)
        lines = [
            "%s (%s) resolved onto %d node(s)"
            % (ep.name, ep.key, len(self.weights)),
            "  direction:  %s" % ep.direction,
            "  weights:    %s, collapse=%s, normalize=%s"
            % (ep.weight_provenance, self.collapse, self.normalize),
            "  evidence:   %d of %d term(s) landed; %d causal, %d "
            "correlative, %d with an inferred direction"
            % (len(landed), len(ep.active_terms(self.include_contested)),
               n_causal, n_corr, n_inferred),
        ]
        for node in sorted(self.weights, key=lambda k: -abs(self.weights[k])):
            lines.append("    %-10s %+.4f" % (node, self.weights[node]))
        by_rule = {}
        for p in self.placements:
            by_rule.setdefault(p.rule, []).append(p.structure)
        for rule in ("exact", "ancestor", "descendants", "unmapped"):
            if rule in by_rule:
                lines.append(
                    "  %-12s %s" % (rule + ":", ", ".join(by_rule[rule]))
                )
        if self.equal_splits:
            lines.append(
                "  equal split (%s): %s"
                % (ILLUSTRATIVE_EQUAL_SPLIT, ", ".join(self.equal_splits))
            )
        if self.cancellations:
            lines.append(
                "  SIGN CANCELLATION at %s: this parcellation is too coarse "
                "to express those terms separately"
                % ", ".join(self.cancellations)
            )
        return "\n".join(lines)

    def __repr__(self):
        return "ResolvedEndpoint(%r, %d nodes)" % (
            self.endpoint.key, len(self.weights)
        )


def _node_names(network):
    if hasattr(network, "nodes"):
        return [node.name for node in network.nodes]
    return list(network)


def _resolve(endpoint, network, ontology, region_sizes, collapse, normalize,
             include_contested):
    if collapse not in VALID_COLLAPSE:
        raise ValueError(
            "collapse must be one of %s, got %r" % (list(VALID_COLLAPSE),
                                                    collapse)
        )
    names = _node_names(network)
    name_set = set(names)
    node_ids = {}
    for name in names:
        try:
            node_ids[name] = ontology.id_for_acronym(name)
        except KeyError:
            # A node this ontology does not name cannot carry an anatomical
            # term. Hand-built scenario nodes ("VTA_DA") land here, and
            # that is the right outcome: this module reads out CCF regions.
            continue

    placements = []
    contributions = {}   # node -> list of signed weights
    equal_splits = []

    for term in endpoint.active_terms(include_contested):
        # A term is ONE unit of evidence however many CCF structures the
        # paper's region spans. "Dorsal hippocampus" is CA1 + CA3 + DG and
        # must not outweigh a single-structure term threefold just because
        # the CCF subdivides it more finely. Within a term the structures
        # therefore SUM back to one unit; the mean/sum choice below is
        # about several DIFFERENT terms landing on one node, which is a
        # different question.
        per_structure = 1.0 / len(term.structures)
        within_term = {}
        for acronym in term.structures:
            nodes, weights, rule, split_was_equal = _place(
                acronym, ontology, name_set, node_ids, region_sizes
            )
            weights = [per_structure * w for w in weights]
            placements.append(
                TermPlacement(term, acronym, rule, nodes,
                              [term.sign * w for w in weights])
            )
            if split_was_equal:
                equal_splits.append(acronym)
            for node, weight in zip(nodes, weights):
                within_term[node] = within_term.get(node, 0.0) + weight
        for node, weight in within_term.items():
            contributions.setdefault(node, []).append(term.sign * weight)

    weights = {}
    cancellations = []
    for node, parts in contributions.items():
        total = sum(parts)
        if collapse == "mean":
            total = total / float(len(parts))
        # Zero parts carry no sign and must not be counted as negative, or a
        # node that took two same-signed terms would be reported as a
        # cancellation. `_place`'s partial-`region_sizes` fix removed the
        # only route to a zero part, so this is belt-and-braces rather than
        # a live fix; the exact-zero TOTAL below is reachable and is not,
        # since two opposite-signed terms cancelling leave a node that
        # should be named in `cancellations` and absent from `weights`
        # rather than present as a 1e-17 residue.
        signs = {1 if p > 0.0 else -1 for p in parts if abs(p) > _TINY}
        if len(signs) > 1:
            cancellations.append(node)
        if abs(total) > _TINY:
            weights[node] = total

    if normalize:
        scale = sum(abs(w) for w in weights.values())
        if scale > 0.0:
            weights = {k: v / scale for k, v in weights.items()}

    readout = BehaviorReadout(endpoint.key, weights)
    return ResolvedEndpoint(
        endpoint=endpoint,
        readout=readout,
        placements=placements,
        weights=weights,
        collapse=collapse,
        normalize=normalize,
        include_contested=include_contested,
        cancellations=sorted(cancellations),
        equal_splits=sorted(set(equal_splits)),
    )


def _place(acronym, ontology, name_set, node_ids, region_sizes):
    """The four rules of the module docstring, for one CCF acronym."""
    if acronym in name_set:
        return (acronym,), (1.0,), "exact", False

    try:
        structure_id = ontology.id_for_acronym(acronym)
    except KeyError:
        return (), (), "unmapped", False

    # Rule 2: the nearest ancestor that is a node.
    for ancestor_id in ontology.ancestors(structure_id):
        ancestor = ontology[ancestor_id].acronym
        if ancestor in name_set:
            return (ancestor,), (1.0,), "ancestor", False

    # Rule 3: the nodes that are descendants of this structure.
    descendants = [
        name for name, node_id in node_ids.items()
        if ontology.is_descendant_of(node_id, structure_id)
    ]
    if descendants:
        descendants.sort()
        # PARTIAL coverage is not usable. Taking `region_sizes.get(d, 0.0)`
        # would give the covered descendants the whole term and the missing
        # ones nothing, while reporting a volume-weighted split - a silent
        # unilateral collapse wearing the label of an exact calculation. So
        # the volume split needs a positive size for EVERY descendant, and
        # anything less falls back to the equal split and says so.
        sizes = [
            float(region_sizes.get(d, 0.0)) if region_sizes else 0.0
            for d in descendants
        ]
        if all(s > 0.0 for s in sizes):
            total = sum(sizes)
            return tuple(descendants), tuple(s / total for s in sizes), \
                "descendants", False
        share = 1.0 / len(descendants)
        return tuple(descendants), (share,) * len(descendants), \
            "descendants", True

    return (), (), "unmapped", False


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

#: Every source that puts a region into an endpoint below. Each entry names
#: the species, the technique and the assay, because all three decide what
#: the term is worth. PMIDs were verified against PubMed's own esummary
#: records (`scripts/verify_pmids.py` re-runs the check).
CITATIONS = {
    # --- locomotion --------------------------------------------------------
    "Liu2020": (
        "Liu D, Li W, Ma C, Zheng W, Yao Y, Tso CF, Zhong P, Chen X, Song JH, "
        "Choi W, Paik SB, Han H, Dan Y. A common hub for sleep and motor "
        "control in the substantia nigra. Science 2020;367(6476):440-445, "
        "PMID:31974254. MOUSE, optogenetic and chemogenetic, plus recording. "
        "SNr GAD2+ GABAergic neurons are preferentially active in states of "
        "LOW motor activity; activating them biases behavioural transitions "
        "toward less motor activity, inactivating them does the reverse."
    ),
    "Caggiano2018": (
        "Caggiano V, Leiras R, Goñi-Erro H, Masini D, Bellardita C, Bouvier "
        "J, Caldeira V, Fisone G, Kiehn O. Midbrain circuits that set "
        "locomotor speed and gait selection. Nature 2018;553(7689):455-460, "
        "PMID:29342142. MOUSE, optogenetic. Glutamatergic cuneiform neurons "
        "drive locomotion and are UNIQUELY able to elicit high-speed "
        "synchronous-gait escape locomotion, which glutamatergic PPN neurons "
        "cannot."
    ),
    "Masini2022": (
        "Masini D, Kiehn O. Targeted activation of midbrain neurons restores "
        "locomotor function in mouse models of parkinsonism. Nat Commun "
        "2022;13(1):504, PMID:35082287. MOUSE, chemogenetic and optogenetic. "
        "Excitation of CAUDAL glutamatergic PPN neurons normalized an "
        "otherwise severe locomotor deficit; the local GABAergic population "
        "gave only slow locomotion."
    ),
    "daSilva2018": (
        "da Silva JA, Tecuapetla F, Paixão V, Costa RM. Dopamine neuron "
        "activity before action initiation gates and invigorates future "
        "movements. Nature 2018;554(7691):244-248, PMID:29420469. MOUSE, "
        "optogenetic plus recording. Inhibiting SNc dopamine neurons during "
        "immobility reduced the probability and vigour of future movements; "
        "brief activation increased both. Manipulations AFTER initiation had "
        "no effect, so the sign is about initiation, not gait."
    ),
    "Jing2019": (
        "Jing MY, Han X, Zhao TY, Wang ZY, Lu GY, Wu N, Song R, Li J. "
        "Re-examining the role of ventral tegmental area dopaminergic neurons "
        "in motor activity and reinforcement by chemogenetic and optogenetic "
        "manipulation in mice. Metab Brain Dis 2019;34(5):1421-1430, "
        "PMID:31313126. MOUSE (DAT-Cre), chemogenetic (hM3Dq/hM4Di) and "
        "optogenetic, OPEN FIELD. Activation or inhibition of VTA dopamine "
        "neurons significantly increased or decreased open-field locomotor "
        "activity respectively - bidirectional, in this endpoint's own assay."
    ),
    "Kosse2017": (
        "Kosse C, Schöne C, Bracey E, Burdakov D. Orexin-driven GAD65 network "
        "of the lateral hypothalamus sets physical activity in mice. Proc "
        "Natl Acad Sci U S A 2017;114(17):4525-4530, PMID:28396414. MOUSE, "
        "optogenetic and chemogenetic with cell-type specific deep-brain "
        "recording. GAD65-LH bursts precede running bouts; silencing "
        "depressed voluntary locomotion, overactivation produced "
        "hyperlocomotion. Necessity and sufficiency in one paper."
    ),
    "Attachaipanich2023": (
        "Attachaipanich S, Ozawa T, Macpherson T, Hikida T. Dual Roles for "
        "Nucleus Accumbens Core Dopamine D1-Expressing Neurons Projecting to "
        "the Substantia Nigra Pars Reticulata in Limbic and Motor Control in "
        "Male Mice. eNeuro 2023;10(6), PMID:37236787. MOUSE (male), "
        "optogenetic terminal stimulation. Stimulating nucleus accumbens core "
        "D1-MSN terminals in SNr increased locomotion when bilateral and "
        "produced contralateral turning when unilateral. The paper's own "
        "asymmetry, recorded here: INHIBITION of the pathway altered neither "
        "locomotion nor reward."
    ),
    "Capelli2017": (
        "Capelli P, Pivetta C, Soledad Esposito M, Arber S. Locomotor speed "
        "control circuits in the caudal brainstem. Nature "
        "2017;551(7680):373-377, PMID:29059682. MOUSE, optogenetic. "
        "Glutamatergic neurons of the lateral paragigantocellular nucleus "
        "(CCF PGRNl) are essential to support high-speed locomotion and "
        "positively tune speed. The same paper reports glycinergic neurons "
        "that induce behavioural arrest, in caudal-brainstem territory the "
        "CCF does not separate - see SEARCHED_AND_EMPTY."
    ),
    "Kravitz2010": (
        "Kravitz AV, Freeze BS, Parker PR, Kay K, Thwin MT, Deisseroth K, "
        "Kreitzer AC. Regulation of parkinsonian motor behaviours by "
        "optogenetic control of basal ganglia circuitry. Nature "
        "2010;466(7306):622-6, PMID:20613723. MOUSE (D1-Cre / D2-Cre), "
        "optogenetic. Bilateral excitation of INDIRECT-pathway (D2) striatal "
        "projection neurons produced freezing, bradykinesia and fewer "
        "locomotor initiations; excitation of DIRECT-pathway (D1) neurons "
        "reduced freezing and increased locomotion. Two intermingled, roughly "
        "equinumerous populations in one CCF structure with OPPOSITE signs, "
        "which is why CP is contested."
    ),
    "Pamukcu2020": (
        "Pamukcu A, Cui Q, Xenias HS, Berceau BL, Augustine EC, Fan I, "
        "Chalasani S, Hantman AW, Lerner TN, Boca SM, Chan CS. Parvalbumin(+) "
        "and Npas1(+) Pallidal Neurons Have Distinct Circuit Topology and "
        "Function. J Neurosci 2020;40(41):7855-7876, PMID:32868462. MOUSE, "
        "optogenetic. Stated by the authors: 'PV+ neurons and Npas1+ neurons "
        "promoted and suppressed locomotion, respectively.' Within-nucleus "
        "sign inversion, hence GPe is contested."
    ),
    "Guillaumin2021": (
        "Guillaumin A, Serra GP, Georges F, Wallén-Mackenzie Å. Experimental "
        "investigation into the role of the subthalamic nucleus (STN) in "
        "motor control using optogenetics in mice. Brain Res "
        "2021;1755:147226, PMID:33358727. MOUSE (C57BL/6), bidirectional "
        "optogenetics in freely moving animals: 'selective optogenetic "
        "inhibition of the STN enhances locomotion while its excitation "
        "reduces locomotion'."
    ),
    "Callahan2024": (
        "Callahan JW, Morales JC, Atherton JF, Wang D, Kostic S, Bevan MD. "
        "Movement-related increases in subthalamic activity optimize "
        "locomotion. Cell Rep 2024;43(8):114495, PMID:39068661. MOUSE "
        "(C57BL/6), recording plus optogenetics on a head-fixed treadmill. "
        "The opposite of [Guillaumin2021] at a shorter timescale: 'brief "
        "optogenetic inhibition of the lateral STN ... slows and prematurely "
        "terminates locomotion'. The two are not reconcilable at one node, "
        "which is what STN's contested flag records."
    ),
    # --- anxiety -----------------------------------------------------------
    "Jimenez2018": (
        "Jimenez JC, Su K, Goldberg AR, Luna VM, Biane JS, Ordek G, Zhou P, "
        "Ong SK, Wright MA, Zweifel L, Paninski L, Hen R, Kheirbek MA. "
        "Anxiety Cells in a Hippocampal-Hypothalamic Circuit. Neuron "
        "2018;97(3):670-683.e6, PMID:29397273. MOUSE, freely moving calcium "
        "imaging plus optogenetics, ELEVATED PLUS MAZE. Ventral CA1 is "
        "enriched in anxiety cells; activating vCA1 terminals in LHA (but not "
        "in basal amygdala) increased anxiety and avoidance."
    ),
    "Yan2022": (
        "Yan JJ, Ding XJ, He T, Chen AX, Zhang W, Yu ZX, Cheng XY, Wei CY, Hu "
        "QD, Liu XY, Zhang YL, He M, Xie ZY, Zha X, Xu C, Cao P, Li H, Xu XH. "
        "A circuit from the ventral subiculum to anterior hypothalamic "
        "nucleus GABAergic neurons essential for anxiety-like behavioral "
        "avoidance. Nat Commun 2022;13(1):7464, PMID:36463200. MOUSE (male), "
        "recording plus optogenetics, ELEVATED PLUS MAZE. Retrograde tracing "
        "identifies ventral SUBICULUM as the input driving open-arm "
        "avoidance; the causal inhibition was applied at AHN, so SUB's sign "
        "here is one synapse removed."
    ),
    "Anthony2014": (
        "Anthony TE, Dee N, Bernard A, Lerchner W, Heintz N, Anderson DJ. "
        "Control of stress-induced persistent anxiety by an extra-amygdala "
        "septohypothalamic circuit. Cell 2014;156(3):522-36, PMID:24485458. "
        "MOUSE (C57BL/6), optogenetic. Against the classical view that the "
        "lateral septum suppresses anxiety, LS Crfr2+ GABAergic projection "
        "neurons to anterior hypothalamus ENHANCE stress-induced anxiety: "
        "transient activation promotes and inhibition suppresses it."
    ),
    "Borodovitsyna2020": (
        "Borodovitsyna O, Duffy BC, Pickering AE, Chandler DJ. Anatomically "
        "and functionally distinct locus coeruleus efferents mediate opposing "
        "effects on anxiety-like behavior. Neurobiol Stress 2020;13:100284, "
        "PMID:33344735. RAT, chemogenetic, ELEVATED PLUS MAZE. LC neurons "
        "projecting to the central amygdala PROMOTE anxiety-like behaviour "
        "while LC neurons projecting to mPFC promote exploration - opposing "
        "efferents within one nucleus, recorded on the term."
    ),
    "Chen2020": (
        "Chen L, Lu YP, Chen HY, Huang SN, Guo YR, Zhang JY, Li QX, Luo CY, "
        "Lin SW, Chen ZN, Hu LH, Wang WX, Li HY, Cai P, Yu CX. Ventral "
        "tegmental area GABAergic neurons induce anxiety-like behaviors and "
        "promote palatable food intake. Neuropharmacology 2020;173:108114, "
        "PMID:32376186. MOUSE, optogenetic. Photoactivation of VTA GABAergic "
        "neurons 'drove a series of anxiety-like behaviors in the open field, "
        "elevated plus maze, and marble-burying test'. A GABAergic "
        "subpopulation, not the dopaminergic majority - see the term's note."
    ),
    "Adhikari2015": (
        "Adhikari A, Lerner TN, Finkelstein J, Pak S, Jennings JH, Davidson "
        "TJ, Ferenczi E, Gunaydin LA, Mirzabekov JJ, Ye L, Kim SY, Lei A, "
        "Deisseroth K. Basomedial amygdala mediates top-down control of "
        "anxiety and fear. Nature 2015;527(7577):179-85, PMID:26536109. "
        "MOUSE, optogenetic, ELEVATED PLUS MAZE. BMA activation DECREASED "
        "high-anxiety states, and the ventral-mPFC to BMA projection "
        "implements top-down reduction of anxiety at baseline and after "
        "stress."
    ),
    "FelixOrtiz2013": (
        "Felix-Ortiz AC, Beyeler A, Seo C, Leppla CA, Wildes CP, Tye KM. BLA "
        "to vHPC inputs modulate anxiety-related behaviors. Neuron "
        "2013;79(4):658-64, PMID:23972595. MOUSE (C57BL/6, male), optogenetic "
        "at terminals, ELEVATED PLUS MAZE. Activating BLA to ventral CA1 "
        "synapses 'acutely and robustly increased anxiety-related behaviors'; "
        "inhibiting them decreased anxiety."
    ),
    "Tye2011": (
        "Tye KM, Prakash R, Kim SY, Fenno LE, Grosenick L, Zarabi H, Thompson "
        "KR, Gradinaru V, Ramakrishnan C, Deisseroth K. Amygdala circuitry "
        "mediating reversible and bidirectional control of anxiety. Nature "
        "2011;471(7338):358-62, PMID:21389985. MOUSE, optogenetic, ELEVATED "
        "PLUS MAZE. Stimulating BLA TERMINALS in CeA was ANXIOLYTIC and "
        "inhibiting them anxiogenic, while direct stimulation of BLA SOMATA "
        "had NO effect - which the authors attribute to recruiting "
        "antagonistic downstream structures. This is why BLA is contested "
        "rather than simply positive."
    ),
    "Kim2013": (
        "Kim SY, Adhikari A, Lee SY, Marshel JH, Kim CK, Mallory CS, Lo M, "
        "Pak S, Mattis J, Lim BK, Malenka RC, Warden MR, Neve R, Tye KM, "
        "Deisseroth K. Diverging neural pathways assemble a behavioural state "
        "from separable features in anxiety. Nature 2013;496(7444):219-23, "
        "PMID:23515158. MOUSE, optogenetic, ELEVATED PLUS MAZE. OVAL BNST "
        "activity promoted anxious-state features while ANTERODORSAL BNST "
        "activity was anxiolytic; the adBNST to lateral hypothalamus "
        "projection specifically reduced risk-avoidance. Two subregions of "
        "one CCF structure with opposite signs, hence BST is contested."
    ),
    "Sun2023": (
        "Sun Y, Qian L, Xu L, Hunt S, Sah P. Somatostatin neurons in the "
        "central amygdala mediate anxiety by disinhibition of the central "
        "sublenticular extended amygdala. Mol Psychiatry "
        "2023;28(10):4163-4174, PMID:33005027. MOUSE, optogenetic. Activating "
        "the CeL somatostatin input to the sublenticular extended amygdala "
        "'has no effect on movement, but is anxiogenic in both open field and "
        "elevated plus maze'."
    ),
    "Green2020": (
        "Green TA, Baracz SJ, Everett NA, Robinson KJ, Cornish JL. "
        "Differential effects of GABA(A) receptor activation in the prelimbic "
        "and orbitofrontal cortices on anxiety. Psychopharmacology (Berl) "
        "2020;237(11):3237-3247, PMID:32666257. RAT (Wistar, male, n=33 "
        "prelimbic), muscimol microinjection, ELEVATED PLUS MAZE. Muscimol "
        "into PL REDUCED anxiety-like behaviour with no locomotor confound; "
        "the same muscimol into orbitofrontal cortex had no effect."
    ),
    "deVisser2011": (
        "de Visser L, Baars AM, van 't Klooster J, van den Bos R. Transient "
        "inactivation of the medial prefrontal cortex affects both anxiety "
        "and decision-making in male wistar rats. Front Neurosci 2011;5:102, "
        "PMID:21927595. RAT (Wistar, male), muscimol plus baclofen into mPFC "
        "(PL AND ILA together), ELEVATED PLUS MAZE. Inactivation INCREASED "
        "anxiety with no change in general activity - the opposite direction "
        "to [Green2020], which is what makes PL contested."
    ),
    "Bi2013": (
        "Bi LL, Wang J, Luo ZY, Chen SP, Geng F, Chen YH, Li SJ, Yuan CH, Lin "
        "S, Gao TM. Enhanced excitability in the infralimbic cortex produces "
        "anxiety-like behaviors. Neuropharmacology 2013;72:148-56, "
        "PMID:23643746. MOUSE (C57BL/6, male), intra-infralimbic "
        "microinjection, ELEVATED PLUS MAZE. Bicuculline (local "
        "disinhibition) decreased open-arm exploration; CNQX (local "
        "inactivation) produced the opposite, anxiolytic effect. AP5 did "
        "nothing and there was no motor or appetite confound."
    ),
    "Li2021": (
        "Li Z, Rizzi G, Tan KR. Zona incerta subpopulations differentially "
        "encode and modulate anxiety. Sci Adv 2021;7(37):eabf6709, "
        "PMID:34516764. MOUSE, c-Fos plus single-cell calcium imaging plus "
        "optogenetics, ELEVATED PLUS MAZE. Somatostatin+ zona incerta "
        "activation induced anxiety while calretinin+ activation and Vglut2+ "
        "inhibition were anxiolytic - opposite directions for one CCF "
        "structure inside one paper, hence contested. Intra-ZI diazepam was "
        "anxiolytic, which matters for the benzodiazepine ligands this "
        "codebase already carries."
    ),
    # --- analgesia ---------------------------------------------------------
    "Morgan2014": (
        "Morgan MM, Reid RA, Stormann TM, Lautermilch NJ. Opioid selective "
        "antinociception following microinjection into the periaqueductal "
        "gray of the rat. J Pain 2014;15(11):1102-1109, PMID:25106089. RAT "
        "(Sprague-Dawley, both sexes), ventrolateral PAG microinjection, HOT "
        "PLATE. Oxycodone (32-188 ug), morphine and fentanyl into vlPAG each "
        "produced a dose-dependent INCREASE in hot-plate latency; methadone "
        "and buprenorphine did not. Opioid in vlPAG disinhibits its output "
        "neurons, which is the step from 'drug in the region' to 'more "
        "activity in the region'."
    ),
    "Samineni2017": (
        "Samineni VK, Grajales-Reyes JG, Copits BA, O'Brien DE, Trigg SL, "
        "Gomez AM, Bruchas MR, Gereau RW 4th. Divergent Modulation of "
        "Nociception by Glutamatergic and GABAergic Neuronal Subpopulations "
        "in the Periaqueductal Gray. eNeuro 2017;4(2), PMID:28374016. MOUSE "
        "(C57BL/6, male), cell-type-specific DREADDs. Activating "
        "GLUTAMATERGIC vlPAG neurons or inhibiting GABAergic ones suppresses "
        "nociception; the inverse facilitates it. Assay is Hargreaves and von "
        "Frey, NOT tail-flick or hot plate, so it supports PAG's sign without "
        "being an on-assay term."
    ),
    "Urban1994": (
        "Urban MO, Smith DJ. Nuclei within the rostral ventromedial medulla "
        "mediating morphine antinociception from the periaqueductal gray. "
        "Brain Res 1994;652(1):9-16, PMID:7953726. RAT, TAIL FLICK. Morphine "
        "(6 nmol) into PAG inhibited the tail-flick reflex, and lidocaine "
        "block of raphe magnus with the adjacent gigantocellular and lateral "
        "paragigantocellular nuclei COMPLETELY abolished it."
    ),
    "Mitchell1998": (
        "Mitchell JM, Lowe D, Fields HL. The contribution of the rostral "
        "ventromedial medulla to the antinociceptive effects of systemic "
        "morphine in restrained and unrestrained rats. Neuroscience "
        "1998;87(1):123-33, PMID:9722146. RAT, RVM lidocaine or muscimol, "
        "TAIL FLICK. Muscimol into RVM COMPLETELY reversed systemic morphine "
        "analgesia; morphine (5 ug) into RVM significantly increased "
        "tail-flick latency. RVM inactivation did not change baseline latency "
        "in unrestrained rats, so this tone is state-dependent rather than "
        "standing."
    ),
    "ZhuoGebhart1997": (
        "Zhuo M, Gebhart GF. Biphasic modulation of spinal nociceptive "
        "transmission from the medullary raphe nuclei in the rat. J "
        "Neurophysiol 1997;78(2):746-58, PMID:9307109. RAT, TAIL FLICK. At 43 "
        "of 86 RVM sites LOW-intensity stimulation (5-25 uA) FACILITATED the "
        "tail-flick reflex and higher intensity (50-200 uA) INHIBITED it; "
        "glutamate microinjection reproduced the biphasic effect (5 nmol "
        "facilitates, 50 nmol inhibits) at the SAME sites. The clearest "
        "published case in this module of a region whose readout cannot be "
        "linear in its firing rate, and the concrete target P7.2 should be "
        "tested against."
    ),
    "Shah2025": (
        "Shah DP, Sharma PR, Agarwal R, Barik A. A "
        "septo-hypothalamic-medullary circuit directs stress-induced "
        "analgesia. Elife 2025;13, PMID:39831900. MOUSE (C57BL/6, male), "
        "optogenetics, chemogenetics and photometry, using BOTH HOT PLATE AND "
        "TAIL FLICK. Chemogenetic activation of RVM neurons postsynaptic to "
        "lateral hypothalamus produced more licks and LOWER hot-plate "
        "latency, i.e. an ON-cell-like, negatively signed RVM population. "
        "Disengaging them is what produces stress-induced analgesia."
    ),
    "Wei1999": (
        "Wei F, Dubner R, Ren K. Nucleus reticularis gigantocellularis and "
        "nucleus raphe magnus in the brain stem exert opposite effects on "
        "behavioral hyperalgesia and spinal Fos protein expression after "
        "peripheral inflammation. Pain 1999;80(1-2):127-41, PMID:10204725. "
        "RAT, soma-selective ibotenic-acid lesion. Lesion of the "
        "gigantocellular nucleus ATTENUATED inflammatory hyperalgesia; "
        "extending the lesion into raphe magnus cancelled the effect. "
        "Opposite signs for two adjacent nuclei in the same animals. Assay is "
        "CFA thermal hyperalgesia, NOT tail-flick or hot plate."
    ),
    "Jones1986": (
        "Jones SL, Gebhart GF. Characterization of coeruleospinal inhibition "
        "of the nociceptive tail-flick reflex in the rat: mediation by spinal "
        "alpha 2-adrenoceptors. Brain Res 1986;364(2):315-30, PMID:2868781. "
        "RAT, electrical AND glutamate stimulation of locus coeruleus / "
        "subcoeruleus, TAIL FLICK. Stimulation INHIBITED the tail-flick "
        "reflex with the lowest thresholds (25 uA or less) in LC proper; "
        "glutamate microinjection reproduced it, so the effect is on cell "
        "bodies and not fibres of passage, and intrathecal yohimbine or "
        "phentolamine raised thresholds 83-94%, so it is a spinal alpha-2 "
        "mechanism."
    ),
    "Hirschberg2017": (
        "Hirschberg S, Li Y, Randall A, Kremer EJ, Pickering AE. Functional "
        "dichotomy in spinal- vs prefrontal-projecting locus coeruleus "
        "modules splits descending noradrenergic analgesia from ascending "
        "aversion and anxiety in rats. Elife 2017;6, PMID:29027903. RAT, "
        "retrograde PSAM/PSEM chemogenetics. The LC spinal module gives "
        "robust lateralised antinociception while the LC prefrontal module "
        "exacerbates spontaneous pain, aversion and anxiety. The single "
        "clearest link between this module's analgesia and anxiety endpoints, "
        "and the reason LC's note names both. Assay is Hargreaves and von "
        "Frey."
    ),
    "Campion2016": (
        "Campion KN, Saville KA, Morgan MM. Relative contribution of the "
        "dorsal raphe nucleus and ventrolateral periaqueductal gray to "
        "morphine antinociception and tolerance in the rat. Eur J Neurosci "
        "2016;44(9):2667-2672, PMID:27564986. RAT (Sprague-Dawley, male), "
        "cumulative morphine microinjection, HOT PLATE. Morphine was MORE "
        "potent in vlPAG (D50 3.3 ug) than in lateral (4.3 ug) or medial "
        "dorsal raphe (5.8 ug), and produced antinociception on 94% of vlPAG "
        "injections against 68-78% of DRN injections - a head-to-head potency "
        "comparison in one preparation."
    ),
    "Xie2023": (
        "Xie L, Wu H, Chen Q, Xu F, Li H, Xu Q, Jiao C, Sun L, Ullah R, Chen "
        "X. Divergent modulation of pain and anxiety by GABAergic neurons in "
        "the ventrolateral periaqueductal gray and dorsal raphe. "
        "Neuropsychopharmacology 2023;48(10):1509-1519, PMID:36526697. MOUSE, "
        "chemogenetics, chronic ablation and optogenetics. Dorsal raphe and "
        "vlPAG GABAergic neurons had OPPOSING effects: inhibiting or ablating "
        "DR neurons promoted nociception while activating them ameliorated "
        "inflammatory pain. Assay is Hargreaves, von Frey and elevated plus "
        "maze."
    ),
    "Petronilho2012": (
        "Petronilho A, Reis GM, Dias QM, Fais RS, Prado WA. Antinociceptive "
        "effect of stimulating the zona incerta with glutamate in rats. "
        "Pharmacol Biochem Behav 2012;101(3):360-8, PMID:22327011. RAT "
        "(Wistar, male), glutamate (7 ug) microinjection into zona incerta, "
        "TAIL FLICK. INCREASED tail-flick latency with no Rota-rod deficit; "
        "injections near but outside ZI were ineffective; the effect was "
        "abolished by dorsolateral-funiculus lesion and by naloxone, "
        "methysergide and phenoxybenzamine, so it is a genuine descending "
        "opioid / serotonergic / adrenergic mechanism."
    ),
    "Han2015": (
        "Han S, Soleiman MT, Soden ME, Zweifel LS, Palmiter RD. Elucidating "
        "an Affective Pain Circuit that Creates a Threat Memory. Cell "
        "2015;162(2):363-374, PMID:26186190. MOUSE, genetic silencing plus "
        "optogenetics. Silencing CGRP+ parabrachial neurons blocks pain "
        "responses and threat-memory formation; optogenetic stimulation "
        "produces defensive responses. Both hot plate and tail flick are "
        "indexed for this article but the full text could not be retrieved "
        "here, so the per-assay latency direction is NOT claimed - only the "
        "sign is taken from it."
    ),
    "Pastoriza1996": (
        "Pastoriza LN, Morrow TJ, Casey KL. Medial frontal cortex lesions "
        "selectively attenuate the hot plate response: possible nocifensive "
        "apraxia in the rat. Pain 1996;64(1):11-17, PMID:8867243. RAT "
        "(Sprague-Dawley, male), bilateral excitotoxic lesion of medial "
        "frontal cortex including cingulate. HOT-PLATE latency increased by "
        "82% while TAIL-FLICK latency and formalin scores were UNCHANGED, "
        "with righting and foot-lifting intact. The single best demonstration "
        "that this module's two analgesia assays need different weight "
        "vectors."
    ),
    "Hao2025": (
        "Hao S, Xue M, Chen QY, Wan J, Ma YJ, Shi W, Chen X, Li XH, Lu JS, Xu "
        "F, Bi GQ, Tao W, Zhuo M. Supraspinal facilitation of painful stimuli "
        "by glutamatergic innervation from the retrosplenial to the anterior "
        "cingulate cortex. PLoS Biol 2025;23(1):e3003011, PMID:39869643. "
        "MOUSE (C57BL/6, male), opto- and chemogenetic activation of "
        "retrosplenial to anterior cingulate glutamatergic projections. "
        "Significantly FACILITATED behavioural responses to mechanical and "
        "thermal noxious stimuli and, stated explicitly, 'did not influence "
        "spinal nociceptive responses in the tail-flick test'. "
        "Gain-of-function confirmation of [Pastoriza1996]'s dissociation, in "
        "mouse, in the opposite manipulation direction."
    ),
    "Chen2024": (
        "Chen C, Niehaus JK, Dinc F, Huang KL, Barnette AL, Tassou A, Shuster "
        "SA, Wang L, Lemire A, Menon V, Ritola K, Hantman AW, Zeng H, "
        "Schnitzer MJ, Scherrer G. Neural circuit basis of placebo pain "
        "relief. Nature 2024;632(8027):1092-1100, PMID:39048016. MOUSE "
        "(C57BL/6, both sexes), HOT PLATE among the assays. Activating "
        "rostral anterior cingulate to pontine projection neurons elicited "
        "analgesia even without placebo conditioning, and inhibiting them "
        "decreased pain thresholds - a POSITIVELY signed rACC efferent "
        "against the negatively signed bulk-ACC lesion result. Recorded on "
        "ACAd's note rather than used, because the bulk manipulations agree "
        "with each other."
    ),
    "Zhang1995": (
        "Zhang YQ, Tang JS, Yuan B, Jia H. Inhibitory effects of electrical "
        "stimulation of thalamic nucleus submedius area on the rat tail flick "
        "reflex. Brain Res 1995;696(1-2):205-12, PMID:8574670. RAT "
        "(Sprague-Dawley), electrical stimulation, TAIL FLICK. Unilateral "
        "stimulation of the ventral submedial thalamic nucleus markedly and "
        "intensity-dependently DEPRESSED the tail-flick reflex, and a "
        "ventrolateral orbital cortex lesion eliminated the effect."
    ),
    "Roberts1994": (
        "Roberts VJ, Dong WK. The effect of thalamic nucleus submedius "
        "lesions on nociceptive responding in rats. Pain 1994;57(3):341-349, "
        "PMID:7936712. RAT. Submedial lesions LOWERED the vocalization "
        "threshold (more pain) but produced NO change in tail-flick latency, "
        "from which the authors conclude the nucleus acts on supraspinally "
        "mediated inhibition and not on spinally mediated responses."
    ),
    "Meda2019": (
        "Meda KS, Patel T, Braz JM, Malik R, Turner ML, Seifikar H, Basbaum "
        "AI, Sohal VS. Microcircuit Mechanisms through which Mediodorsal "
        "Thalamic Input to Anterior Cingulate Cortex Exacerbates Pain-Related "
        "Aversion. Neuron 2019;102(5):944-959.e3, PMID:31030955. MOUSE "
        "(male), optogenetics in nerve-injury and chemotherapy neuropathy "
        "models. Activating mediodorsal thalamic inputs to anterior cingulate "
        "elicited pain-related AVERSION in both models while BLA inputs to "
        "the same target did the opposite. Assay is conditioned place "
        "aversion, not tail flick or hot plate."
    ),
    "Rashvand2014": (
        "Rashvand M, Khajavai A, Parviz M, Hasanein P, Keshavarz M. GABAA "
        "receptors are involved in the analgesic effects of morphine "
        "microinjected into the central nucleus of the amygdala. Clin Exp "
        "Pharmacol Physiol 2014;41(5):338-44, PMID:24606334. RAT (male; "
        "strain not stated in the retrieved record, which is itself recorded "
        "here), bilateral intra-CeA muscimol against bicuculline, TAIL FLICK. "
        "'Microinjection of bicuculline or muscimol in combination with "
        "morphine into the CeA increased and decreased TFL, respectively' - a "
        "bidirectional manipulation of the region's own excitability, which "
        "is why it is preferred here over the lesion papers."
    ),
    "Manning1995": (
        "Manning BH, Mayer DJ. The central nucleus of the amygdala "
        "contributes to the production of morphine antinociception in the rat "
        "tail-flick test. J Neurosci 1995;15(12):8199-213, PMID:8613754. RAT "
        "(Sprague-Dawley, male), bilateral NMDA lesion and lidocaine "
        "inactivation, TAIL FLICK. Lesions of CeA, but not of basolateral or "
        "medial amygdala, abolished the antinociception produced by 2.5 mg/kg "
        "morphine, without producing thermal hyperalgesia at baseline - so "
        "the tone is opioid-recruited rather than standing."
    ),
    "Andreoli2017": (
        "Andreoli M, Marketkar T, Dimitrov E. Contribution of amygdala CRF "
        "neurons to chronic pain. Exp Neurol 2017;298(Pt A):1-12, "
        "PMID:28830762. MOUSE (C57BL/6 background, male, CRF-Cre), "
        "Cre-dependent DREADDs plus projection ablation, TAIL FLICK. "
        "Inhibiting CeA CRF neurons RESTORED stress-induced tail-flick "
        "analgesia in neuropathic mice, so that molecularly defined "
        "subpopulation is negatively signed against the nucleus-level result. "
        "Recorded on CEA's note."
    ),
    "Wilson2019": (
        "Wilson TD, Valdivia S, Khan A, Ahn HS, Adke AP, Martinez Gonzalez S, "
        "Sugimura YK, Carrasquillo Y. Dual and Opposing Functions of the "
        "Central Amygdala in the Modulation of Pain. Cell Rep "
        "2019;29(2):332-346.e5, PMID:31597095. MOUSE (male), DREADDs and "
        "ChR2. CeA PKCdelta+ cells are sensitized by nerve injury and "
        "increase pain-related responses while CeA somatostatin+ cells drive "
        "antinociception - and, stated explicitly, the somatostatin "
        "activation reversed TACTILE but NOT THERMAL hypersensitivity, so no "
        "positive thermal sign is taken from it. Assay is Hargreaves, "
        "acetone, von Frey and Randall-Selitto."
    ),
    "MorganFranklin1990": (
        "Morgan MJ, Franklin KB. 6-Hydroxydopamine lesions of the ventral "
        "tegmentum abolish D-amphetamine and morphine analgesia in the "
        "formalin test but not in the tail flick test. Brain Res "
        "1990;519(1-2):144-9, PMID:2118819. RAT, bilateral 6-OHDA lesion. The "
        "reason VTA carries NO analgesia weight in this module: morphine "
        "analgesia in the TAIL-FLICK test was completely unchanged. An "
        "explicit measured zero, not an absence of evidence."
    ),
    # --- reward ------------------------------------------------------------
    "Ilango2014": (
        "Ilango A, Kesner AJ, Keller KL, Stuber GD, Bonci A, Ikemoto S. "
        "Similar roles of substantia nigra and ventral tegmental dopamine "
        "neurons in reward and aversion. J Neurosci 2014;34(3):817-22, "
        "PMID:24431440. MOUSE (TH-Cre), bidirectional optogenetics, "
        "CONDITIONED PLACE PREFERENCE. ChR2 mice SOUGHT the "
        "VTA-photostimulation compartment while NpHR mice AVOIDED the "
        "compartment where VTA photoinhibition occurred - both directions in "
        "one assay, which is what makes it the sign citation."
    ),
    "Tan2012": (
        "Tan KR, Yvon C, Turiault M, Mirzabekov JJ, Doehner J, Labouèbe G, "
        "Deisseroth K, Tye KM, Lüscher C. GABA neurons of the VTA drive "
        "conditioned place aversion. Neuron 2012;73(6):1173-83, "
        "PMID:22445344. MOUSE, optogenetic. The opposite-signed cell type "
        "inside VTA, recorded on the term's note."
    ),
    "Esmaeili2012": (
        "Esmaeili MH, Sahraei H, Ali-Beig H, Ardehari-Ghaleh M, Mohammadian "
        "Z, Zardooz H, Salimi SH, Shams J, Noroozzadeh A. Transient "
        "inactivation of the nucleus accumbens reduces both the expression "
        "and acquisition of morphine-induced conditioned place preference in "
        "rats. Pharmacol Biochem Behav 2012;102(2):249-56, PMID:22580069. "
        "RAT, lidocaine inactivation. Inactivation of EITHER accumbens "
        "subregion reduced acquisition and expression of morphine conditioned "
        "place preference, which is the region-level result; the core/shell "
        "and D1/D2 disagreements are on the term's note."
    ),
    "Lobo2010": (
        "Lobo MK, Covington HE 3rd, Chaudhury D, Friedman AK, Sun H, "
        "Damez-Werno D, Dietz DM, Zaman S, Koo JW, Kennedy PJ, Mouzon E, "
        "Mogri M, Neve RL, Deisseroth K, Han MH, Nestler EJ. Cell "
        "type-specific loss of BDNF signaling mimics optogenetic control of "
        "cocaine reward. Science 2010;330(6002):385-90, PMID:20947769. MOUSE "
        "(D1-Cre, D2-Cre), optogenetic plus conditional knockout. 'Activation "
        "of D2+ neurons ... suppresses cocaine reward, with opposite effects "
        "induced by activation of D1+ neurons.'"
    ),
    "SoaresCunha2020": (
        "Soares-Cunha C, de Vasconcelos NAP, Coimbra B, Domingues AV, Silva "
        "JM, Loureiro-Campos E, Gaspar R, Sotiropoulos I, Sousa N, Rodrigues "
        "AJ. Nucleus accumbens medium spiny neurons subtypes signal both "
        "reward and aversion. Mol Psychiatry 2020;25(12):3241-3255, "
        "PMID:31462765. MOUSE, optogenetic. 'Brief D1- or D2-MSN optogenetic "
        "stimulation elicited positive reinforcement and enhanced cocaine "
        "conditioning. Conversely, prolonged activation induced aversion' - "
        "the D2 sign depends on stimulation duration rather than on cell "
        "identity, which the authors say means the classical striatal "
        "opposition needs reconsidering."
    ),
    "Amohashemi2022": (
        "Amohashemi E, Reisi P, Alaei H. Involvement of GABAA receptors of "
        "lateral habenula in the acquisition and expression phases of "
        "morphine-induced place preference in male rats. Behav Pharmacol "
        "2022;33(7):452-465, PMID:36148835. RAT (Wistar), "
        "intra-lateral-habenula pharmacology, CONDITIONED PLACE PREFERENCE. "
        "Muscimol (habenular inhibition) ENHANCED morphine place preference "
        "while bicuculline (disinhibition) REDUCED acquisition, with no "
        "locomotor confound. Bidirectional and directly on this endpoint's "
        "assay."
    ),
    "Faget2018": (
        "Faget L, Zell V, Souter E, McPherson A, Ressler R, Gutierrez-Reed N, "
        "Yoo JH, Dulcis D, Hnasko TS. Opponent control of behavioral "
        "reinforcement by inhibitory and excitatory projections from the "
        "ventral pallidum. Nat Commun 2018;9(1):849, PMID:29487284. MOUSE, "
        "optogenetic. Ventral pallidal GABA neuron activation drove positive "
        "reinforcement (especially via the projection to VTA) while glutamate "
        "neuron activation drove avoidance (especially via the projection to "
        "lateral habenula). The GABAergic population is the majority; the "
        "opponent split is on the term's note. The CCF has no 'VP' acronym - "
        "ventral pallidum is SI plus MA."
    ),
    "Qi2014": (
        "Qi J, Zhang S, Wang HL, Wang H, de Jesus Aceves Buendia J, Hoffman "
        "AF, Lupica CR, Seal RP, Morales M. A glutamatergic reward input from "
        "the dorsal raphe to ventral tegmental area dopamine neurons. Nat "
        "Commun 2014;5:5390, PMID:25388237. MOUSE (C57BL/6) AND RAT "
        "(Sprague-Dawley), optogenetic. Photoactivation of dorsal raphe "
        "VGluT3 terminals in VTA released accumbens dopamine, reinforced "
        "instrumental behaviour and ESTABLISHED CONDITIONED PLACE "
        "PREFERENCES."
    ),
    "McDevitt2014": (
        "McDevitt RA, Tiran-Cappello A, Shen H, Balderas I, Britt JP, Marino "
        "RAM, Chung SL, Richie CT, Harvey BK, Bonci A. Serotonergic versus "
        "nonserotonergic dorsal raphe projection neurons: differential "
        "participation in reward circuitry. Cell Rep 2014;8(6):1857-1869, "
        "PMID:25242321. MOUSE. Dorsal raphe reinforcement is driven primarily "
        "by NON-serotonergic neurons projecting to VTA - the sign is stable, "
        "the transmitter attribution is not, and this is recorded on DR's "
        "note rather than resolved."
    ),
    "Zarrindast2004": (
        "Zarrindast MR, Ahmadi S, Haeri-Rohani A, Rezayof A, Jafari MR, "
        "Jafari-Sabet M. GABA(A) receptors in the basolateral amygdala are "
        "involved in mediating morphine reward. Brain Res 2004;1006(1):49-58, "
        "PMID:15047023. RAT (Wistar), intra-BLA pharmacology, CONDITIONED "
        "PLACE PREFERENCE. Muscimol (0.25-0.5 ug; local inhibition) DECREASED "
        "acquisition of morphine place preference, and bicuculline (local "
        "disinhibition) with a sub-threshold 1 mg/kg morphine dose ELICITED "
        "significant place preference. The authors' own caveat, carried here: "
        "high bicuculline doses reduced locomotion."
    ),
    "Jiang2021": (
        "Jiang C, Wang X, Le Q, Liu P, Liu C, Wang Z, He G, Zheng P, Wang F, "
        "Ma L. Morphine coordinates SST and PV interneurons in the prelimbic "
        "cortex to disinhibit pyramidal neurons and enhance reward. Mol "
        "Psychiatry 2021;26(4):1178-1193, PMID:31413370. MOUSE (SST-Cre and "
        "PV-Cre backcrossed to C57BL/6J), CONDITIONED PLACE PREFERENCE. "
        "Morphine DISINHIBITS prelimbic pyramidal neurons via mu-opioid "
        "receptors on parvalbumin interneurons and delta-opioid receptor / "
        "Rac1 signalling in somatostatin interneurons, and both of the latter "
        "are REQUIRED for morphine place preference."
    ),
    "Wayman2018": (
        "Wayman WN, Woodward JJ. Chemogenetic Excitation of "
        "Accumbens-Projecting Infralimbic Cortical Neurons Blocks "
        "Toluene-Induced Conditioned Place Preference. J Neurosci "
        "2018;38(6):1462-1471, PMID:29317484. RAT (adolescent "
        "Sprague-Dawley), DREADDs plus whole-cell recording, CONDITIONED "
        "PLACE PREFERENCE. Toluene place preference went with REDUCED "
        "excitability of infralimbic layer 5/6 neurons projecting to "
        "accumbens shell, and 'chemogenetic reversal of the toluene-induced "
        "decrease in IL5/6-NAc shell neurons blocked the expression of "
        "toluene-induced CPP', while the prelimbic to accumbens-core "
        "population showed no such change. Infralimbic activity is therefore "
        "NEGATIVELY signed for reward, opposite to prelimbic."
    ),
    "Liu2022": (
        "Liu M, Tan X, Liu E, Hang Z, Song R, Mu S, Han W, Yue Q, Sun J. "
        "Inactivation of the Lateral Hypothalamus Attenuates "
        "Methamphetamine-Induced Conditioned Place Preference through "
        "Regulation of Kcnq3 Expression. Int J Mol Sci 2022;23(13), "
        "PMID:35806315. MOUSE (C57BL/6), c-Fos plus chemogenetic inhibition "
        "plus slice recording, CONDITIONED PLACE PREFERENCE. Repeated "
        "methamphetamine activates c-Fos in lateral hypothalamic neurons in "
        "place-preference animals, and chemogenetic inhibition of the lateral "
        "hypothalamus disrupted the behaviour."
    ),
    "Stamatakis2016": (
        "Stamatakis AM, Van Swieten M, Basiri ML, Blair GA, Kantak P, Stuber "
        "GD. Lateral Hypothalamic Area Glutamatergic Neurons and Their "
        "Projections to the Lateral Habenula Regulate Feeding and Reward. J "
        "Neurosci 2016;36(2):302-11, PMID:26758824. MOUSE, bidirectional "
        "optogenetics. 'Inhibition of LHA-LHb glutamatergic fibers produced a "
        "real-time place preference, whereas stimulation ... had the opposite "
        "effect' - the opposite-signed subpopulation, recorded on LHA's note."
    ),
    "Meyers2006": (
        "Meyers RA, Zavala AR, Speer CM, Neisewander JL. Dorsal hippocampus "
        "inhibition disrupts acquisition and expression, but not "
        "consolidation, of cocaine conditioned place preference. Behav "
        "Neurosci 2006;120(2):401-12, PMID:16719704. RAT, "
        "intra-dorsal-hippocampus muscimol, CONDITIONED PLACE PREFERENCE. "
        "Disrupted acquisition and expression but NOT consolidation, with "
        "infusion into the overlying cortex as the anatomical control. The "
        "manipulation covers the dorsal hippocampus as a whole, so the term "
        "names CA1, CA3 and DG together and splits one unit of evidence "
        "between them."
    ),
    "Jehl2025": (
        "Jehl J, Ciscato M, Vicq E, Guyon N, Dejean de la Batie G, Mondoloni "
        "S, Frangieh J, Mohayyaei M, Nguyen C, Pons S, Maskos U, Hardelin JP, "
        "Marti F, Corringer PJ, Faure P, Mourot A. The interpeduncular "
        "nucleus blunts the rewarding effect of nicotine. Neuron "
        "2025;113(12):1898-1907.e6, PMID:40262615. MOUSE (beta-4 nicotinic "
        "receptor knock-in), chemogenetics plus optogenetics, CONDITIONED "
        "PLACE PREFERENCE. Blocking nicotine-induced interpeduncular "
        "activation enhanced VTA responses and INCREASED the drug's rewarding "
        "effect; optogenetic inhibition of the projection to laterodorsal "
        "tegmentum did the same. 'The IPN acts as a regulatory brake on the "
        "nicotine reward circuit.'"
    ),
    # --- sedation ----------------------------------------------------------
    "Carter2010": (
        "Carter ME, Yizhar O, Chikahisa S, Nguyen H, Adamantidis A, Nishino "
        "S, Deisseroth K, de Lecea L. Tuning arousal with optogenetic "
        "modulation of locus coeruleus neurons. Nat Neurosci "
        "2010;13(12):1526-33, PMID:21037585. MOUSE, optogenetics with "
        "EEG/EMG. Frequency-dependent ChR2 stimulation of locus coeruleus "
        "noradrenergic neurons caused immediate sleep-to-wake transitions and "
        "increased locomotor arousal, and halorhodopsin inhibition suppressed "
        "wakefulness."
    ),
    "Adamantidis2007": (
        "Adamantidis AR, Zhang F, Aravanis AM, Deisseroth K, de Lecea L. "
        "Neural substrates of awakening probed with optogenetic control of "
        "hypocretin neurons. Nature 2007;450(7168):420-4, PMID:17943086. "
        "MOUSE, optogenetics with polysomnography. Photostimulating "
        "hypocretin/orexin neurons of the lateral hypothalamus increased the "
        "probability of transition to wakefulness from both NREM and REM and "
        "shortened wake latency, at 5-30 Hz but not at 1 Hz."
    ),
    "Fujita2017": (
        "Fujita A, Bonnavion P, Wilson MH, Mickelsen LE, Bloit J, de Lecea L, "
        "Jackson AC. Hypothalamic Tuberomammillary Nucleus Neurons: "
        "Electrophysiological Diversity and Essential Role in Arousal "
        "Stability. J Neurosci 2017;37(39):9574-9592, PMID:28874450. MOUSE "
        "(Hdc-Cre), optogenetic silencing with EEG. Acute silencing of "
        "histaminergic tuberomammillary neurons during wakefulness was "
        "SUFFICIENT to rapidly and selectively induce slow-wave sleep."
    ),
    "Yu2015": (
        "Yu X, Ye Z, Houston CM, Zecharia AY, Ma Y, Zhang Z, Uygun DS, Parker "
        "S, Vyssotski AL, Yustos R, Franks NP, Brickley SG, Wisden W. "
        "Wakefulness Is Governed by GABA and Histamine Cotransmission. Neuron "
        "2015;87(1):164-78, PMID:26094607. MOUSE. Knocking down vesicular "
        "GABA transport in tuberomammillary neurons produced sustained "
        "hyperwakefulness, so the GABA these neurons co-release is a BRAKE on "
        "their own histaminergic wake drive. A within-region sign conflict, "
        "recorded on the term's note."
    ),
    "Anaclet2015": (
        "Anaclet C, Pedersen NP, Ferrari LL, Venner A, Bass CE, Arrigoni E, "
        "Fuller PM. Basal forebrain control of wakefulness and cortical "
        "rhythms. Nat Commun 2015;6:8744, PMID:26524973. MOUSE, DREADDs with "
        "EEG/EMG. Chemogenetic activation of basal forebrain GABAergic "
        "neurons produced sustained wakefulness and fast cortical rhythms "
        "while chemogenetic INHIBITION INCREASED SLEEP; cholinergic or "
        "glutamatergic activation changed state consolidation and EEG but not "
        "total wake time."
    ),
    "Xu2015": (
        "Xu M, Chung S, Zhang S, Zhong P, Ma C, Chang WC, Weissbourd B, Sakai "
        "N, Luo L, Nishino S, Dan Y. Basal forebrain circuit for sleep-wake "
        "control. Nat Neurosci 2015;18(11):1641-7, PMID:26457552. MOUSE, "
        "optogenetics. Activating cholinergic, glutamatergic AND parvalbumin+ "
        "GABAergic basal forebrain neurons each rapidly induced wakefulness, "
        "whereas somatostatin+ GABAergic activation PROMOTED NREM - the "
        "embedded sleep-promoting subpopulation, recorded on the term's note."
    ),
    "Kroeger2017": (
        "Kroeger D, Ferrari LL, Petit G, Mahoney CE, Fuller PM, Arrigoni E, "
        "Scammell TE. Cholinergic, Glutamatergic, and GABAergic Neurons of "
        "the Pedunculopontine Tegmental Nucleus Have Distinct Effects on "
        "Sleep/Wake Behavior in Mice. J Neurosci 2017;37(5):1352-1366, "
        "PMID:28039375. MOUSE, DREADDs with EEG/EMG. Chemogenetic activation "
        "of GLUTAMATERGIC pedunculopontine neurons induced prolonged "
        "wakefulness and inhibition reduced wake and increased NREM; "
        "cholinergic activation only altered NREM EEG spectra."
    ),
    "Kaur2017": (
        "Kaur S, Wang JL, Ferrari L, Thankachan S, Kroeger D, Venner A, "
        "Lazarus M, Wellman A, Arrigoni E, Fuller PM, Saper CB. A Genetically "
        "Defined Circuit for Arousal from Sleep during Hypercapnia. Neuron "
        "2017;96(5):1153-1167.e5, PMID:29103805. MOUSE, chemogenetics and "
        "optogenetics. Activating external lateral parabrachial CGRP neurons "
        "caused wakefulness, and optogenetic inhibition prevented CO2-evoked "
        "arousal - though not arousal to tone or shaking, so the wake drive "
        "is stimulus-gated rather than tonic."
    ),
    "EbanRothschild2016": (
        "Eban-Rothschild A, Rothschild G, Giardino WJ, Jones JR, de Lecea L. "
        "VTA dopaminergic neurons regulate ethologically relevant sleep-wake "
        "behaviors. Nat Neurosci 2016;19(10):1356-66, PMID:27595385. MOUSE, "
        "photometry with DREADDs and optogenetics and polysomnography. "
        "Chemogenetic and optogenetic INHIBITION of VTA dopamine neurons "
        "suppressed wakefulness even against salient stimuli, and stimulation "
        "initiated and maintained wake."
    ),
    "Kroeger2018": (
        "Kroeger D, Absi G, Gagliardi C, Bandaru SS, Madara JC, Ferrari LL, "
        "Arrigoni E, Münzberg H, Scammell TE, Saper CB, Vetrivelan R. Galanin "
        "neurons in the ventrolateral preoptic area promote sleep and heat "
        "loss in mice. Nat Commun 2018;9(1):4129, PMID:30297727. MOUSE, "
        "optogenetics and DREADDs. Photostimulation of ventrolateral preoptic "
        "galanin neurons at 1-4 Hz promoted sleep, optogenetic inhibition "
        "reduced sleep, and chemogenetic activation increased sleep and "
        "shortened sleep latency in an insomnia model. The authors' own "
        "artifact warning is carried on the term: above 8 Hz, conduction "
        "block caused paradoxical waking, which is not a sign flip."
    ),
    "Lu2000": (
        "Lu J, Greco MA, Shiromani P, Saper CB. Effect of lesions of the "
        "ventrolateral preoptic nucleus on NREM and REM sleep. J Neurosci "
        "2000;20(10):3830-42, PMID:10804223. RAT, ibotenate lesion. Lesions "
        "of the ventrolateral preoptic cell cluster cut NREM time by 50-60% "
        "and delta power by 60-70%, and the residual Fos+ cell count linearly "
        "predicted NREM time (r = 0.77). The classic lesion anchor behind the "
        "same sign [Kroeger2018] establishes in mouse."
    ),
    "Oikonomou2019": (
        "Oikonomou G, Altermatt M, Zhang RW, Coughlin GM, Montz C, Gradinaru "
        "V, Prober DA. The Serotonergic Raphe Promote Sleep in Zebrafish and "
        "Mice. Neuron 2019;103(4):686-701.e8, PMID:31248729. MOUSE (and "
        "zebrafish), ablation plus optogenetics. Against the classical "
        "assumption, raphe ablation INCREASED wakefulness and impaired the "
        "homeostatic sleep-deprivation response, and TONIC optogenetic "
        "stimulation near baseline rates INDUCED SLEEP - while BURST "
        "stimulation induced wake. A firing-mode-dependent sign."
    ),
    "Cho2017": (
        "Cho JR, Treweek JB, Robinson JE, Xiao C, Bremner LR, Greenbaum A, "
        "Gradinaru V. Dorsal Raphe Dopamine Neurons Modulate Arousal and "
        "Promote Wakefulness by Salient Stimuli. Neuron "
        "2017;94(6):1205-1219.e8, PMID:28602690. MOUSE. Dorsal raphe DOPAMINE "
        "neurons - a different cell type in the same voxels as "
        "[Oikonomou2019]'s serotonergic ones - promote wakefulness. The pair "
        "is why DR is the contested term of the sedation endpoint."
    ),
    # --- working memory ----------------------------------------------------
    "Kamigaki2017": (
        "Kamigaki T, Dan Y. Delay activity of specific prefrontal interneuron "
        "subtypes modulates memory-guided behavior. Nat Neurosci "
        "2017;20(6):854-863, PMID:28436982. MOUSE, two-photon calcium imaging "
        "plus optogenetics. Optogenetically inhibiting dorsomedial prefrontal "
        "pyramidal cells (by driving somatostatin+ or parvalbumin+ "
        "interneurons) during the DELAY ALONE impaired delayed Go/No-Go "
        "performance, and activating VIP+ interneurons (disinhibiting "
        "pyramids) ENHANCED performance. The bidirectional result is what "
        "fixes the sign."
    ),
    "Bolkan2017": (
        "Bolkan SS, Stujenske JM, Parnaudeau S, Spellman TJ, Rauffenbart C, "
        "Abbas AI, Harris AZ, Gordon JA, Kellendonk C. Thalamic projections "
        "sustain prefrontal activity during working memory maintenance. Nat "
        "Neurosci 2017;20(7):987-996, PMID:28481349. MOUSE, "
        "projection-specific optogenetics plus in vivo recording, T-maze "
        "delayed non-match-to-sample. Mediodorsal thalamus to prefrontal "
        "input supports the MAINTENANCE phase, prefrontal delay-tuned spiking "
        "required it, and ENHANCING mediodorsal excitability was SUFFICIENT "
        "to enhance task performance."
    ),
    "Viena2018": (
        "Viena TD, Linley SB, Vertes RP. Inactivation of nucleus reuniens "
        "impairs spatial working memory and behavioral flexibility in the "
        "rat. Hippocampus 2018;28(4):297-311, PMID:29357198. RAT, muscimol "
        "and procaine microinfusion. Muscimol into nucleus reuniens impaired "
        "delayed non-match-to-sample T-maze alternation at 30, 60 and 120 s "
        "delays and abolished the win-shift strategy."
    ),
    "Hallock2016": (
        "Hallock HL, Wang A, Griffin AL. Ventral Midline Thalamus Is Critical "
        "for Hippocampal-Prefrontal Synchrony and Spatial Working Memory. J "
        "Neurosci 2016;36(32):8372-89, PMID:27511010. RAT. Reuniens/rhomboid "
        "inactivation abolished the working-memory-specific synchrony between "
        "dorsal hippocampus and prefrontal cortex. The mechanism behind "
        "[Viena2018]'s behavioural result."
    ),
    "Spellman2015": (
        "Spellman T, Rigotti M, Ahmari SE, Fusi S, Gogos JA, Gordon JA. "
        "Hippocampal-prefrontal input supports spatial encoding in working "
        "memory. Nature 2015;522(7556):309-14, PMID:26053122. MOUSE, "
        "terminal-specific optogenetics plus recording, T-maze spatial "
        "working memory. Inhibiting direct hippocampal to prefrontal "
        "afferents impaired ENCODING (but not maintenance or retrieval), and "
        "prefrontal cue representations depended on hippocampal input only "
        "during that phase."
    ),
    "Song2020": (
        "Song D, Wang D, Yang Q, Yan T, Wang Z, Yan Y, Zhao J, Xie Z, Liu Y, "
        "Ke Z, Qazi TJ, Li Y, Wu Y, Shi Q, Lang Y, Zhang H, Huang T, Wang C, "
        "Quan Z, Qing H. The lateralization of left hippocampal CA3 during "
        "the retrieval of spatial working memory. Nat Commun 2020;11(1):2901, "
        "PMID:32518226. MOUSE, optogenetics plus in vivo recording. "
        "Optogenetic suppression of CA3 pyramidal neurons disrupted spatial "
        "working memory selectively in the CHOICE (retrieval) phase, and "
        "suppressing medial-septal parvalbumin neurons projecting to CA3 "
        "reproduced the impairment."
    ),
    "Nakashiba2008": (
        "Nakashiba T, Young JZ, McHugh TJ, Buhl DL, Tonegawa S. Transgenic "
        "inhibition of synaptic transmission reveals role of CA3 output in "
        "hippocampal learning. Science 2008;319(5867):1260-4, PMID:18218862. "
        "MOUSE, inducible tetanus-toxin block of the trisynaptic path. CA3 "
        "OUTPUT is dispensable for incremental spatial learning and required "
        "mainly for one-trial and pattern-completion memory, which is why "
        "CA3's weight here is not claimed to be large. Recorded on the term's "
        "note."
    ),
    "Gemzik2021": (
        "Gemzik ZM, Donahue MM, Griffin AL. Optogenetic suppression of the "
        "medial septum impairs working memory maintenance. Learn Mem "
        "2021;28(10):361-370, PMID:34526381. RAT, phase-specific "
        "optogenetics. Timed suppression of the medial septum impaired choice "
        "accuracy on a working-memory-dependent conditional discrimination "
        "only during cue sampling, and on a delayed non-match-to-place task "
        "only during the delay, with no deficit on the "
        "working-memory-INdependent variant."
    ),
    # --- sensorimotor gating -----------------------------------------------
    "Li1998": (
        "Li L, Korngut LM, Frost BJ, Beninger RJ. Prepulse inhibition "
        "following lesions of the inferior colliculus: prepulse intensity "
        "functions. Physiol Behav 1998;65(1):133-9, PMID:9811375. RAT, "
        "unilateral ibotenate lesion of inferior colliculus. LOWERED prepulse "
        "inhibition across prepulse intensities (29-49 dB SPL) WITHOUT "
        "increasing baseline startle magnitude - the startle control is what "
        "makes it a gating effect and not a reflex-strength artifact."
    ),
    "Fendt1994": (
        "Fendt M, Koch M, Schnitzler HU. Sensorimotor gating deficit after "
        "lesions of the superior colliculus. Neuroreport 1994;5(14):1725-8, "
        "PMID:7827317. RAT, excitotoxic lesion. Superior colliculus lesions "
        "DECREASED prepulse inhibition without affecting baseline startle or "
        "footshock sensitization."
    ),
    "Fendt1999": (
        "Fendt M. Enhancement of prepulse inhibition after blockade of GABA "
        "activity within the superior colliculus. Brain Res 1999;833(1):81-5, "
        "PMID:10375679. RAT, intracerebral picrotoxin. GABA-A blockade in the "
        "superior colliculus (i.e. moderate collicular activation) ENHANCED "
        "prepulse inhibition without changing baseline startle or spontaneous "
        "motor activity. Loss- and gain-of-function agreeing is rare in this "
        "literature."
    ),
    "Kodsi1995": (
        "Kodsi MH, Swerdlow NR. Ventral pallidal GABA-A receptors regulate "
        "prepulse inhibition of acoustic startle. Brain Res "
        "1995;684(1):26-35, PMID:7583201. RAT, intra-ventral-pallidum "
        "picrotoxin. GABA-A blockade in medial and central ventral pallidum "
        "(i.e. pallidal DISINHIBITION) REDUCED prepulse inhibition with no "
        "effect on startle amplitude or latency, while GABA-B blockade did "
        "nothing. The ventral pallidum is SI plus MA in the CCF; there is no "
        "'VP' acronym."
    ),
    "Wan1997": (
        "Wan FJ, Swerdlow NR. The basolateral amygdala regulates sensorimotor "
        "gating of acoustic startle in the rat. Neuroscience "
        "1997;76(3):715-24, PMID:9135045. RAT, quinolinic-acid lesion plus "
        "intracerebral AP5. Basolateral amygdala lesions REDUCED prepulse "
        "inhibition without changing startle amplitude, and intra-BLA AP5 "
        "dose-dependently reduced it - reducing basolateral glutamatergic "
        "drive lowers gating. Haloperidol did not rescue, so the effect is "
        "dopamine-independent."
    ),
    "Davis1982": (
        "Davis M, Gendelman DS, Tischler MD, Gendelman PM. A primary acoustic "
        "startle circuit: lesion and stimulation studies. J Neurosci "
        "1982;2(6):791-805, PMID:7086484. RAT, lesion plus electrical "
        "stimulation. The caudal pontine reticular nucleus is the obligatory "
        "sensorimotor relay of the startle circuit: bilateral lesions abolish "
        "acoustic startle and stimulation there evokes it at about 5 ms "
        "latency. The same paper reports that lesions of the gigantocellular "
        "and ORAL pontine reticular nuclei FAIL to abolish startle, which is "
        "why PRNr gets no weight."
    ),
    "Cano2021": (
        "Cano JC, Huang W, Fénelon K. The amygdala modulates prepulse "
        "inhibition of the auditory startle reflex through excitatory inputs "
        "to the caudal pontine reticular nucleus. BMC Biol 2021;19(1):116, "
        "PMID:34082731. MOUSE (wild-type plus GlyT2-eGFP/GlyT2-Cre), "
        "optogenetics with slice recording. Photo-inhibition of GLYCINERGIC "
        "caudal pontine interneurons DECREASED prepulse inhibition, while "
        "optogenetic inhibition of the central-amygdala input to the same "
        "nucleus decreased it and activation induced partial gating. Two "
        "anti-correlated populations in one CCF structure, which is why PRNc "
        "is contested."
    ),
    "Koch1993": (
        "Koch M, Kungel M, Herbert H. Cholinergic neurons in the "
        "pedunculopontine tegmental nucleus are involved in the mediation of "
        "prepulse inhibition of the acoustic startle response in the rat. Exp "
        "Brain Res 1993;97(1):71-82, PMID:8131833. RAT, retrograde tracing "
        "plus ChAT immunohistochemistry plus iontophoresis plus lesion. "
        "Neurotoxic lesions of cholinergic pedunculopontine neurons "
        "significantly REDUCED prepulse inhibition with no change in baseline "
        "startle."
    ),
    "Azzopardi2018": (
        "Azzopardi E, Louttit AG, DeOliveira C, Laviolette SR, Schmid S. The "
        "Role of Cholinergic Midbrain Neurons in Startle and Prepulse "
        "Inhibition. J Neurosci 2018;38(41):8798-8808, PMID:30171090. RAT "
        "(male and female), optogenetics. The direct contradiction of "
        "[Koch1993]: transient optogenetic activation of SPECIFICALLY "
        "cholinergic pedunculopontine neurons did NOT inhibit startle - it "
        "ENHANCED startle and produced prepulse FACILITATION - and the "
        "authors conclude that non-cholinergic pedunculopontine neurons "
        "mediate gating. The pedunculopontine weight must therefore not be "
        "routed through cholinergic receptor occupancy in the drug layer."
    ),
    "Yang2024": (
        "Yang C, Chen X, Xu J, Chen W. SKF82958, a dopamine D1 receptor "
        "agonist, disrupts prepulse inhibition in the medial prefrontal "
        "cortex and nucleus accumbens in C57BL/6J mice. Behav Pharmacol "
        "2024;35(4):193-200, PMID:38567425. MOUSE, intracerebral "
        "microinfusion. A D1 agonist into medial prefrontal cortex or into "
        "nucleus accumbens disrupted prepulse inhibition, but not into "
        "ventral hippocampus."
    ),
    "Lacroix2000": (
        "Lacroix L, Spinelli S, White W, Feldon J. The effects of ibotenic "
        "acid lesions of the medial and lateral prefrontal cortex on latent "
        "inhibition, prepulse inhibition and amphetamine-induced "
        "hyperlocomotion. Neuroscience 2000;97(3):459-68, PMID:10828529. RAT, "
        "ibotenate lesion. Medial prefrontal lesions INCREASED prepulse "
        "inhibition, the opposite direction to [Yang2024]'s pharmacology, "
        "which is why the prefrontal gating term is contested."
    ),
    "Wang2015": (
        "Wang J, Li G, Xu Y, Zhang WN. Hyperactivity and disruption of "
        "prepulse inhibition induced by NMDA infusion of the rat ventral "
        "hippocampus: Comparison of uni- and bilateral stimulation. Neurosci "
        "Lett 2015;594:150-4, PMID:25841789. RAT, intracerebral NMDA. "
        "Bilateral NMDA into ventral hippocampus induced hyperactivity and "
        "DISRUPTED prepulse inhibition, bilateral more than unilateral - "
        "increased ventral hippocampal activity lowers gating."
    ),
    "Zhang2002": (
        "Zhang WN, Bast T, Feldon J. Prepulse inhibition in rats with "
        "temporary inhibition/inactivation of ventral or dorsal hippocampus. "
        "Pharmacol Biochem Behav 2002;73(4):929-40, PMID:12213540. RAT, "
        "muscimol or TTX. Inactivation of EITHER ventral or dorsal "
        "hippocampus also REDUCED prepulse inhibition, and the deficit was "
        "neuroleptic-resistant. Together with [Wang2015] this makes the "
        "hippocampal contribution an inverted U rather than monotone, which a "
        "linear readout cannot hold - hence contested."
    ),
    "Kodsi1994": (
        "Kodsi MH, Swerdlow NR. Quinolinic acid lesions of the ventral "
        "striatum reduce sensorimotor gating of acoustic startle in rats. "
        "Brain Res 1994;643(1-2):59-65, PMID:8032933. RAT, quinolinic-acid "
        "lesion. Ventral striatal lesions also REDUCED prepulse inhibition "
        "and raised startle amplitude, and intra-ventral-pallidum muscimol "
        "restored gating. The opposite direction to the accumbens "
        "pharmacology, hence the accumbens gating term is contested."
    ),
    "Swerdlow1990": (
        "Swerdlow NR, Braff DL, Geyer MA. GABAergic projection from nucleus "
        "accumbens to ventral pallidum mediates dopamine-induced sensorimotor "
        "gating deficits of acoustic startle in rats. Brain Res "
        "1990;532(1-2):146-50, PMID:2282510. RAT, intracerebral infusion. "
        "Dopamine into nucleus accumbens disrupted prepulse inhibition, and "
        "muscimol into ventral pallidum reversed that deficit."
    ),
    # --- cited from a term's NOTE or from SEARCHED_AND_EMPTY, never as a
    # --- term's own `citation`: these bound, contest or explain a sign
    # --- rather than establish one ------------------------------------
    "Freeze2013": (
        "Freeze BS, Kravitz AV, Hammack N, Berke JD, Kreitzer AC. Control of "
        "basal ganglia output by direct and indirect pathway projection "
        "neurons. J Neurosci 2013;33(47):18531-9, PMID:24259575. MOUSE, "
        "optogenetic with single-unit recording. Direct-pathway-driven "
        "movement initiation tracked the INHIBITED subpopulation of SNr "
        "neurons and indirect-pathway motor suppression tracked the EXCITED "
        "one. Cited from [Liu2020]'s term as the same gate seen from the "
        "striatal side."
    ),
    "PadillaCoreano2016": (
        "Padilla-Coreano N, Bolkan SS, Pierce GM, Blackman DR, Hardin WD, "
        "Garcia-Garcia AL, Spellman TJ, Gordon JA. Direct ventral "
        "hippocampal-prefrontal input is required for anxiety-related neural "
        "activity and behavior. Neuron 2016;89(4):857-66, PMID:26853301. "
        "MOUSE, terminal optogenetic inhibition, ELEVATED PLUS MAZE. "
        "Inhibiting the ventral hippocampal input to prefrontal cortex "
        "disrupted anxiety and the prefrontal representation of aversion. "
        "Cited from the anxiety CA1 term as agreeing with [Jimenez2018]."
    ),
    "Parfitt2017": (
        "Parfitt GM, Nguyen R, Bang JY, Aqrabawi AJ, Tran MM, Seo DK, "
        "Richards BA, Kim JC. Bidirectional control of anxiety-related "
        "behaviors in mice: role of inputs arising from the ventral "
        "hippocampus to the lateral septum and medial prefrontal cortex. "
        "Neuropsychopharmacology 2017;42(8):1715-1728, PMID:28294135. MOUSE, "
        "projection-specific DREADDs, ELEVATED PLUS MAZE. Activating the "
        "lateral-septum-projecting ventral CA1 population DECREASED anxiety "
        "while the prefrontal-projecting one PROMOTED it - two largely "
        "non-overlapping populations. Cited from the anxiety CA1 term as the "
        "projection that runs the other way."
    ),
    "Gao2023": (
        "Gao F, Huang J, Huang GB, You QL, Yao S, Zhao ST, Liu J, Wu CH, Chen "
        "GF, Liu SM, Yu Z, Zhou YL, Ning YP, Liu S, Hu BJ, Sun XD. Elevated "
        "prelimbic cortex-to-basolateral amygdala circuit activity mediates "
        "comorbid anxiety-like behaviors associated with chronic pain. J Clin "
        "Invest 2023;133(9):e166356, PMID:36917193. MOUSE, optogenetic, "
        "ELEVATED PLUS MAZE. Prelimbic neurons projecting to basolateral "
        "amygdala are hyperactive in chronic pain and suppressing that "
        "circuit reversed anxiety-like behaviour. Cited from the anxiety PL "
        "term as agreeing with [Green2020] against [deVisser2011]."
    ),
    "Bouvier2015": (
        "Bouvier J, Caggiano V, Leiras R, Caldeira V, Bellardita C, Balueva "
        "K, Fuchs A, Kiehn O. Descending command neurons in the brainstem "
        "that halt locomotion. Cell 2015;163(5):1191-1203, PMID:26590422. "
        "MOUSE, optogenetic. Activating V2a neurons of the rostral medulla "
        "STOPS ongoing locomotion and inactivating them decreases spontaneous "
        "stopping. Cited from SEARCHED_AND_EMPTY: this negatively signed "
        "stop-cell population is why no medullary arrest term exists, because "
        "'rostral medulla V2a' does not map onto one CCF region."
    ),
    # --- the ontology this module's acronyms come from ----------------------
    "AllenCCF": (
        "Allen Institute for Brain Science. Allen Mouse Brain Common "
        "Coordinate Framework v3 structure graph (id 1), as ingested in P3.1 "
        "(`connectome/sources.py`, checksum-verified). Every acronym in this "
        "module was checked against that file; the ones that do NOT exist and "
        "the substitutions made for them are in SEARCHED_AND_EMPTY."
    ),
}


# ---------------------------------------------------------------------------
# The seven endpoints
# ---------------------------------------------------------------------------

_T = EndpointTerm


LOCOMOTION = Endpoint(
    key="locomotion",
    name="Locomotor activity",
    assay="open-field / photobeam ambulation; drug-induced hyperlocomotion",
    direction="higher = more locomotion (greater distance travelled)",
    terms=[
        _T("SNr", "increase", "decrease", "Liu2020", "mouse", "optogenetic",
           "open field",
           "GAD2+ SNr GABAergic neurons are preferentially active in states "
           "of low motor activity; activating them biased transitions "
           "toward less motor activity and inactivating them did the "
           "reverse. A GABAergic basal-ganglia output nucleus that gates "
           "movement, established in both directions.",
           note="[Freeze2013]'s single-unit result is the same gate seen "
                "from the striatal side: movement initiation tracked the "
                "INHIBITED SNr subpopulation."),
        _T("CUN", "increase", "increase", "Caggiano2018", "mouse",
           "optogenetic", ("open field", "treadmill locomotion"),
           "Glutamatergic cuneiform neurons drive locomotion and are "
           "uniquely able to elicit high-speed synchronous-gait escape "
           "locomotion. The strongest single-region locomotor driver in "
           "this endpoint."),
        _T("PPN", "increase", "increase", "Masini2022", "mouse",
           "chemogenetic", "open field",
           "Excitation of caudal glutamatergic pedunculopontine neurons "
           "normalized an otherwise severe locomotor deficit; the local "
           "GABAergic population gave only slow locomotion.",
           note="[Caggiano2018] finds the pedunculopontine contribution "
                "restricted to slower alternating gaits, so this term is "
                "the same sign as CUN with a smaller true gain - a "
                "difference the equal illustrative magnitude cannot "
                "express and P7.3 should recover."),
        _T("SNc", "decrease", "decrease", "daSilva2018", "mouse",
           "optogenetic", "movement initiation",
           "Inhibiting SNc dopamine neurons during immobility reduced the "
           "probability and vigour of future movements; brief activation "
           "increased both. Manipulations after initiation had no effect.",
           note="The sign is about INITIATION, not ongoing gait, which a "
                "rate readout does not distinguish."),
        _T("VTA", "increase", "increase", "Jing2019", "mouse",
           "chemogenetic", "open field",
           "Chemogenetic activation or inhibition of VTA dopamine neurons "
           "significantly increased or decreased open-field locomotor "
           "activity respectively - bidirectional, in this endpoint's own "
           "assay."),
        _T("LHA", "decrease", "decrease", "Kosse2017", "mouse",
           "chemogenetic", ("open field", "voluntary running"),
           "Chemogenetic silencing of natural GAD65 lateral-hypothalamic "
           "activity depressed voluntary locomotion while overactivation "
           "produced hyperlocomotion, with the bursts preceding running "
           "bouts. Necessity and sufficiency in one paper."),
        _T("ACB", "increase", "increase", "Attachaipanich2023", "mouse",
           "optogenetic", "open field",
           "Stimulating accumbens-core D1-MSN terminals in SNr increased "
           "locomotion when bilateral and produced contralateral turning "
           "when unilateral.",
           note="The paper's own asymmetry: inhibiting the same pathway "
                "changed nothing. And the CCF gives ACB no children, so "
                "the core/shell dissociation that most of this literature "
                "turns on is structurally unrepresentable here."),
        _T("PGRNl", "increase", "increase", "Capelli2017", "mouse",
           "optogenetic", ("open field", "treadmill locomotion"),
           "Glutamatergic neurons of the lateral paragigantocellular "
           "nucleus are essential to support high-speed locomotion and "
           "positively tune speed via their input to the mesencephalic "
           "locomotor region.",
           note="The same paper's glycinergic arrest neurons sit in "
                "caudal-brainstem territory the CCF does not separate; "
                "see SEARCHED_AND_EMPTY."),
        # --- contested -----------------------------------------------------
        _T("CP", "increase", "increase", "Kravitz2010", "mouse",
           "optogenetic", "open field",
           "Excitation of direct-pathway (D1) striatal projection neurons "
           "reduced freezing and increased locomotion; excitation of "
           "indirect-pathway (D2) neurons did the opposite.",
           contested=True,
           note="[Kravitz2010] contests itself. The two populations are "
                "intermingled and roughly equinumerous inside one CCF "
                "structure, so a scalar CP rate has no defensible sign at "
                "all. Recorded, excluded by default, and available with "
                "include_contested=True."),
        _T("GPe", "increase", "increase", "Pamukcu2020", "mouse",
           "optogenetic", "open field",
           "'PV+ neurons and Npas1+ neurons promoted and suppressed "
           "locomotion, respectively.' The parvalbumin class is the larger "
           "one, so the bulk nucleus leans positive.",
           contested=True,
           note="[Pamukcu2020] contests itself: a within-nucleus sign "
                "inversion, the same shape as CP's."),
        _T("STN", "increase", "decrease", "Guillaumin2021", "mouse",
           "optogenetic", "open field",
           "'Selective optogenetic inhibition of the STN enhances "
           "locomotion while its excitation reduces locomotion.'",
           contested=True,
           note="[Callahan2024] reports the opposite at a shorter "
                "timescale and in the lateral subregion: brief inhibition "
                "slows and prematurely terminates locomotion. Timescale "
                "and subregion, neither of which one CCF node can hold."),
    ],
    notes="Seven of the eight uncontested terms are positive, so this "
          "readout is close to a signed mean rate over the locomotor "
          "system with SNr subtracted. That is what the evidence supports "
          "and is worth knowing before reading a small score change.",
)


ANXIETY = Endpoint(
    key="anxiety",
    name="Anxiety-like behaviour",
    assay="elevated plus maze (open-arm time and entries)",
    direction="higher = MORE anxious = LESS open-arm time",
    terms=[
        _T("CA1", "increase", "increase", "Jimenez2018", "mouse",
           "optogenetic", "elevated plus maze",
           "Ventral CA1 is enriched in anxiety cells, and activating vCA1 "
           "terminals in the lateral hypothalamic area increased anxiety "
           "and avoidance.",
           note="[PadillaCoreano2016]'s vHPC-to-mPFC inhibition agrees. "
                "[Parfitt2017] finds the lateral-septum-projecting vCA1 "
                "population anxiolytic, so one of three projections runs "
                "the other way."),
        _T("SUB", "decrease", "decrease", "Yan2022", "mouse", "optogenetic",
           "elevated plus maze",
           "Exploration-triggered inhibition of the anterior hypothalamic "
           "target reduced open-arm avoidance, and retrograde tracing "
           "identifies ventral subiculum as the input driving it.",
           inferred="manipulation was at AHN, one synapse downstream; "
                    "nothing was done to subicular activity",
           note="The causal manipulation was applied one synapse "
                "downstream, so this term is weaker than CA1's."),
        _T("LSr", "increase", "increase", "Anthony2014", "mouse",
           "optogenetic", "elevated plus maze",
           "Lateral septal Crfr2+ GABAergic projection neurons to anterior "
           "hypothalamus ENHANCE stress-induced anxiety: transient "
           "activation promotes and inhibition suppresses it.",
           note="This is against the classical view that the lateral "
                "septum suppresses anxiety. No primary study manipulating "
                "the region in the opposite direction on this assay was "
                "found, so it is not flagged contested - the disagreement "
                "is with reviews, not with an experiment."),
        _T("LC", "increase", "increase", "Borodovitsyna2020", "rat",
           "chemogenetic", "elevated plus maze",
           "Locus coeruleus neurons projecting to the central amygdala "
           "promote anxiety-like behaviour.",
           note="The same paper's LC-to-mPFC population promotes "
                "exploration, i.e. opposing efferents within one nucleus. "
                "[Hirschberg2017] is the analgesia-side counterpart and "
                "is why LC appears in two endpoints with different signs."),
        _T("VTA", "increase", "increase", "Chen2020", "mouse",
           "optogenetic",
           ("elevated plus maze", "open field", "marble burying"),
           "Photoactivation of VTA GABAergic neurons 'drove a series of "
           "anxiety-like behaviors in the open field, elevated plus maze, "
           "and marble-burying test'.",
           note="A GABAergic subpopulation, not the dopaminergic majority "
                "that carries VTA's locomotion and reward terms."),
        _T("BMA", "increase", "decrease", "Adhikari2015", "mouse",
           "optogenetic", "elevated plus maze",
           "Basomedial amygdala activation DECREASED high-anxiety states, "
           "and the ventral-mPFC to BMA projection implements top-down "
           "reduction of anxiety at baseline and after stress.",
           note="The only negatively signed uncontested term in this "
                "endpoint."),
        # --- contested -----------------------------------------------------
        _T("BLA", "increase", "increase", "FelixOrtiz2013", "mouse",
           "optogenetic", "elevated plus maze",
           "Activating BLA-to-ventral-CA1 synapses 'acutely and robustly "
           "increased anxiety-related behaviors'; inhibiting them "
           "decreased anxiety.",
           contested=True,
           note="[Tye2011] found BLA terminals in CeA ANXIOLYTIC and "
                "direct stimulation of BLA SOMATA to have NO effect, "
                "attributing that to antagonistic downstream structures. "
                "Whole-nucleus drive is empirically near null, which a "
                "somatic rate readout cannot represent."),
        _T("BST", "increase", "increase", "Kim2013", "mouse", "optogenetic",
           "elevated plus maze",
           "Oval BNST activity promoted anxious-state features while "
           "anterodorsal BNST activity was anxiolytic, and the adBNST-to-"
           "lateral-hypothalamus projection specifically reduced "
           "risk-avoidance.",
           contested=True,
           note="[Kim2013] contests itself: two subregions of one CCF "
                "structure with opposite signs, inside one paper."),
        _T("CEA", "increase", "increase", "Sun2023", "mouse", "optogenetic",
           ("elevated plus maze", "open field"),
           "Activating the central-lateral somatostatin input to the "
           "sublenticular extended amygdala 'has no effect on movement, "
           "but is anxiogenic in both open field and elevated plus maze'.",
           contested=True,
           note="[Tye2011]'s BLA-to-CeA stimulation - which excites CeA - "
                "was anxiolytic, the opposite direction. The paper's own "
                "target, the central sublenticular extended amygdala, is "
                "not a CCF acronym; the nearest structure is SI."),
        _T("PL", "decrease", "decrease", "Green2020", "rat", "inactivation",
           ("elevated plus maze", "emergence test"),
           "Muscimol into prelimbic cortex REDUCED anxiety-like behaviour "
           "with no locomotor confound, while the same muscimol into "
           "orbitofrontal cortex had no effect.",
           contested=True,
           note="[deVisser2011] inactivated PL and ILA together and found "
                "the opposite, INCREASED anxiety. [Gao2023]'s PL-to-BLA "
                "suppression agrees with Green. The field's reconciling "
                "hypothesis is that the sign is set by which efferent is "
                "engaged, not by the subregion label."),
        _T("ILA", "increase", "increase", "Bi2013", "mouse",
           "microinjection", "elevated plus maze",
           "Bicuculline in infralimbic cortex (local disinhibition) "
           "decreased open-arm exploration; CNQX (local inactivation) "
           "produced the opposite, anxiolytic effect; AP5 did nothing.",
           contested=True,
           note="[Adhikari2015]'s ventral-mPFC-to-BMA projection is "
                "anxiolytic and [deVisser2011]'s bulk inactivation "
                "increased anxiety, both against this direction."),
        _T("ZI", "increase", "increase", "Li2021", "mouse", "optogenetic",
           "elevated plus maze",
           "Somatostatin+ zona incerta activation induced anxiety while "
           "calretinin+ activation and Vglut2+ inhibition were anxiolytic.",
           contested=True,
           note="[Li2021] contests itself: opposite directions for one CCF "
                "structure inside one paper. Worth keeping visible "
                "because intra-ZI diazepam "
                "was anxiolytic in the same study, and this codebase "
                "already carries diazepam."),
    ],
    notes="Six of the twelve terms are contested, the highest fraction of "
          "any endpoint except sensorimotor gating, and every one of them "
          "is contested for the same structural reason: the amygdala, "
          "BNST, mPFC and zona incerta each contain populations with "
          "opposite anxiety signs that one region-level rate cannot "
          "separate.",
)


ANALGESIA = Endpoint(
    key="analgesia",
    name="Analgesia (thermal antinociception)",
    assay="tail flick (spinal reflex) and hot plate (supraspinal)",
    direction="higher = MORE analgesia = LONGER withdrawal latency",
    terms=[
        _T("PAG", "increase", "increase", "Morgan2014", "rat",
           "microinjection", "hot plate",
           "Oxycodone, morphine and fentanyl into ventrolateral PAG each "
           "produced a dose-dependent increase in hot-plate latency. "
           "Opioid in vlPAG disinhibits its output neurons, which is the "
           "step from 'drug in the region' to 'more activity in it'.",
           inferred="opioid in vlPAG raises its output only via "
                    "disinhibition of the local GABAergic tone",
           note="[Samineni2017] establishes the same sign in MOUSE with "
                "cell-type-specific DREADDs but on Hargreaves rather than "
                "on either of this endpoint's assays; [Urban1994] is the "
                "tail-flick counterpart. The CCF has one PAG structure, "
                "so no ventrolateral subdivision is available."),
        _T("LC", "increase", "increase", "Jones1986", "rat", "stimulation",
           "tail flick",
           "Electrical and glutamate stimulation of locus coeruleus "
           "inhibited the tail-flick reflex, with thresholds of 25 uA or "
           "less in LC proper; glutamate reproduced it, so the effect is "
           "on cell bodies rather than fibres of passage, and intrathecal "
           "alpha-2 antagonists raised thresholds 83-94%.",
           note="[Hirschberg2017] splits this into a spinal module that "
                "is analgesic and a prefrontal module that exacerbates "
                "pain, aversion and anxiety - the same nucleus carrying "
                "this endpoint and the anxiety endpoint through different "
                "efferents."),
        _T("DR", "increase", "increase", "Campion2016", "rat",
           "microinjection", "hot plate",
           "Cumulative morphine microinjection produced antinociception on "
           "68-78% of dorsal raphe injections against 94% of vlPAG "
           "injections, with D50 5.8 and 4.3 ug against vlPAG's 3.3 ug - "
           "a head-to-head potency comparison in one preparation.",
           inferred="same opioid disinhibition step as PAG's term",
           note="The activity direction rests on the same opioid "
                "disinhibition inference PAG's term spells out. "
                "[Xie2023] finds dorsal raphe and vlPAG GABAergic neurons "
                "OPPOSING on Hargreaves, so this term is weaker than "
                "PAG's in a way the equal magnitude does not show."),
        _T("ZI", "increase", "increase", "Petronilho2012", "rat",
           "microinjection", "tail flick",
           "Glutamate (7 ug) into zona incerta increased tail-flick "
           "latency with no Rota-rod deficit; injections near but outside "
           "the nucleus were ineffective, and the effect was abolished by "
           "dorsolateral-funiculus lesion and by naloxone, methysergide "
           "and phenoxybenzamine.",
           note="Glutamate into the region is as direct an activity "
                "increase as this literature offers."),
        _T("SMT", "increase", "increase", "Zhang1995", "rat", "stimulation",
           "tail flick",
           "Unilateral electrical stimulation of the ventral submedial "
           "thalamic nucleus markedly and intensity-dependently depressed "
           "the tail-flick reflex, and a ventrolateral orbital lesion "
           "eliminated the effect.",
           note="[Roberts1994]'s lesion lowered the vocalization threshold "
                "but left tail-flick latency unchanged, which the authors "
                "read as a supraspinal-only role. The two are not "
                "contradictory - stimulation can recruit a pathway that "
                "carries no standing tone - but it does bound the term."),
        _T("CEA", "increase", "increase", "Rashvand2014", "rat",
           "microinjection", "tail flick",
           "Bicuculline and muscimol into the central amygdala, with "
           "morphine, increased and decreased tail-flick latency "
           "respectively - a bidirectional manipulation of the region's "
           "own excitability.",
           note="[Manning1995]'s lesion agrees at nucleus level. The "
                "molecularly defined subpopulations run the other way: "
                "[Andreoli2017]'s CRF+ neurons and [Wilson2019]'s "
                "PKCdelta+ neurons are both negatively signed, and "
                "[Wilson2019]'s somatostatin+ neurons reversed TACTILE "
                "but explicitly NOT thermal hypersensitivity, so no "
                "positive thermal sign is taken from them."),
        _T("GRN", "decrease", "increase", "Wei1999", "rat", "lesion",
           "CFA thermal hyperalgesia",
           "Soma-selective lesion of the gigantocellular nucleus "
           "attenuated inflammatory hyperalgesia; extending the lesion "
           "into raphe magnus cancelled the effect. Opposite signs for two "
           "adjacent nuclei in the same animals.",
           note="Off-assay, and said so. [ZhuoGebhart1997] did include "
                "gigantocellular sites in its tail-flick modulation, so "
                "the region does reach the reflex."),
        _T("PB", "increase", "decrease", "Han2015", "mouse", "genetic",
           "defensive responses",
           "Silencing CGRP+ parabrachial neurons blocks pain responses and "
           "threat-memory formation while optogenetic stimulation produces "
           "defensive responses.",
           inferred="the assay is defensive responding, so `measure` is "
                    "a translation into withdrawal latency, not a reading",
           note="Both hot plate and tail flick are indexed for this "
                "article but the full text could not be retrieved, so the "
                "per-assay latency direction is NOT claimed - only the "
                "sign."),
        _T("ACAd", "decrease", "increase", "Pastoriza1996", "rat", "lesion",
           "hot plate",
           "Bilateral excitotoxic lesion of medial frontal cortex "
           "including cingulate increased HOT-PLATE latency by 82% while "
           "TAIL-FLICK latency and formalin scores were unchanged, with "
           "righting and foot-lifting intact.",
           note="[Hao2025] confirms the same dissociation in mouse from "
                "the opposite manipulation direction. [Chen2024] finds a "
                "rostral-cingulate efferent that is positively signed; "
                "recorded rather than used, because the two bulk "
                "manipulations agree with each other."),
        _T("MD", "increase", "decrease", "Meda2019", "mouse", "optogenetic",
           "conditioned place aversion",
           "Activating mediodorsal thalamic inputs to anterior cingulate "
           "elicited pain-related aversion in nerve-injury and "
           "chemotherapy neuropathy models, while basolateral amygdala "
           "inputs to the same target did the opposite.",
           inferred="the assay is place aversion, so `measure` is a "
                    "translation into withdrawal latency, not a reading",
           note="Off-assay. MD also carries a positive working-memory "
                "term, which is not a contradiction - different endpoints."),
        # --- contested -----------------------------------------------------
        _T("RM", "increase", "increase", "Mitchell1998", "rat",
           "inactivation", "tail flick",
           "Muscimol into the rostral ventromedial medulla completely "
           "reversed systemic morphine analgesia, and morphine (5 ug) into "
           "it significantly increased tail-flick latency. Inactivation "
           "did not change baseline latency in unrestrained rats, so the "
           "tone is state-dependent rather than standing.",
           contested=True,
           note="[ZhuoGebhart1997] is the reason this is contested and is "
                "the most useful citation in the module: at the SAME "
                "sites, low-intensity stimulation FACILITATED the "
                "tail-flick reflex and higher intensity INHIBITED it, "
                "reproduced by 5 vs 50 nmol glutamate. [Shah2025] adds a "
                "mouse ON-cell-like population that lowers hot-plate "
                "latency. This region is the concrete case P7.2's "
                "nonlinear decoder should be tested against; no linear "
                "weight can be right for it. The CCF has no RVM structure "
                "- it is RM plus medial GRN and PGRNl."),
    ],
    notes="Six positive and four negative uncontested terms. THREE of "
          "them come from assays this endpoint is not defined against - "
          "GRN (CFA thermal hyperalgesia), PB (defensive responding) and "
          "MD (conditioned place aversion) - as do one supporting "
          "citation each for PAG, DR and LC. The assay field records that "
          "term by term and ASSAY_VARIANTS is the restriction that drops "
          "all three; PB's and MD's `inferred` field additionally says "
          "that their `measure` is a translation into withdrawal latency "
          "rather than a reading of one.",
)


REWARD = Endpoint(
    key="reward",
    name="Reward",
    assay="conditioned place preference",
    direction="higher = MORE preference for the drug-paired context",
    terms=[
        _T("VTA", "increase", "increase", "Ilango2014", "mouse",
           "optogenetic", "conditioned place preference",
           "ChR2 mice sought the VTA-photostimulation compartment while "
           "halorhodopsin mice avoided the compartment where VTA "
           "photoinhibition occurred - both directions in one assay.",
           note="[Tan2012]'s VTA GABA neurons drive place AVERSION, the "
                "opposite-signed cell type in the same voxels."),
        _T("ACB", "decrease", "decrease", "Esmaeili2012", "rat",
           "inactivation", "conditioned place preference",
           "Lidocaine inactivation of either accumbens subregion reduced "
           "acquisition AND expression of morphine place preference.",
           note="Two disagreements the CCF cannot express, both recorded: "
                "[Lobo2010] has D1 activation raising and D2 activation "
                "suppressing cocaine reward while [SoaresCunha2020] finds "
                "the D2 sign depends on stimulation DURATION, not cell "
                "identity; and the core/shell dissociation is "
                "drug-dependent and opposite-signed, with no children "
                "under ACB in the CCF to hold it."),
        _T("LH", "decrease", "increase", "Amohashemi2022", "rat",
           "microinjection", "conditioned place preference",
           "Muscimol into lateral habenula (inhibition) ENHANCED morphine "
           "place preference while bicuculline (disinhibition) REDUCED "
           "acquisition, with no locomotor confound.",
           note="CCF trap worth naming: 'LH' is the LATERAL HABENULA. The "
                "lateral hypothalamus is 'LHA' and is a separate term in "
                "this same endpoint with the opposite sign."),
        _T(("SI", "MA"), "increase", "increase", "Faget2018", "mouse",
           "optogenetic",
           ("conditioned place preference", "real-time place preference"),
           "Ventral pallidal GABA neuron activation drove positive "
           "reinforcement, especially through the projection to VTA.",
           note="The same paper's glutamatergic minority drives avoidance "
                "through the lateral habenula - a genuine opponent split "
                "rather than a disagreement. The CCF has no 'VP' acronym: "
                "ventral pallidum is SI plus MA, and this one term is "
                "split between them."),
        _T("DR", "increase", "increase", "Qi2014", "mouse", "optogenetic",
           "conditioned place preference",
           "Photoactivation of dorsal raphe VGluT3 terminals in VTA "
           "released accumbens dopamine, reinforced instrumental behaviour "
           "and established conditioned place preferences.",
           note="[McDevitt2014] attributes the reinforcement primarily to "
                "NON-serotonergic raphe neurons. The sign is stable; the "
                "transmitter is not, and this module does not commit to "
                "it."),
        _T("BLA", "decrease", "decrease", "Zarrindast2004", "rat",
           "microinjection", "conditioned place preference",
           "Muscimol into basolateral amygdala decreased acquisition of "
           "morphine place preference, and bicuculline with a "
           "sub-threshold 1 mg/kg morphine dose elicited significant place "
           "preference.",
           note="The authors' own caveat, carried: high bicuculline doses "
                "reduced locomotion."),
        _T("PL", "increase", "increase", "Jiang2021", "mouse", "genetic",
           "conditioned place preference",
           "Morphine disinhibits prelimbic pyramidal neurons via mu-opioid "
           "receptors on parvalbumin interneurons and delta-opioid / Rac1 "
           "signalling in somatostatin interneurons, both of which are "
           "required for morphine place preference."),
        _T("ILA", "increase", "decrease", "Wayman2018", "rat",
           "chemogenetic", "conditioned place preference",
           "Toluene place preference went with REDUCED excitability of "
           "infralimbic layer 5/6 neurons projecting to accumbens shell, "
           "and chemogenetic EXCITATION of those neurons BLOCKED "
           "expression of the preference.",
           note="PL and ILA carry opposite signs in this endpoint, "
                "reproduced across two drug classes and two labs. Any "
                "readout that treats 'mPFC' as one node gets this wrong "
                "by construction, which is why they are separate terms."),
        _T("LHA", "decrease", "decrease", "Liu2022", "mouse",
           "chemogenetic", "conditioned place preference",
           "Chemogenetic inhibition of the lateral hypothalamus disrupted "
           "methamphetamine place preference, in animals where the drug "
           "had activated c-Fos there.",
           note="[Stamatakis2016] finds that INHIBITING the "
                "lateral-hypothalamic glutamatergic projection to lateral "
                "habenula produces place preference, i.e. an "
                "opposite-signed subpopulation."),
        _T(("CA1", "CA3", "DG"), "decrease", "decrease", "Meyers2006",
           "rat", "inactivation", "conditioned place preference",
           "Muscimol into dorsal hippocampus disrupted acquisition and "
           "expression, but not consolidation, of cocaine place "
           "preference, with infusion into the overlying cortex as the "
           "anatomical control.",
           note="The manipulation covers the dorsal hippocampus as a "
                "whole, so this is ONE unit of evidence split three ways "
                "rather than three terms."),
        _T("IPN", "decrease", "increase", "Jehl2025", "mouse",
           "chemogenetic", "conditioned place preference",
           "Blocking nicotine-induced interpeduncular activation enhanced "
           "VTA responses and increased the drug's rewarding effect; "
           "optogenetic inhibition of the projection to laterodorsal "
           "tegmentum did the same. 'The IPN acts as a regulatory brake on "
           "the nicotine reward circuit.'"),
    ],
    notes="Eleven terms, no contested ones, and three of the eleven "
          "negative. That is not because this literature agrees - it is "
          "because the disagreements here are between SUBPOPULATIONS "
          "inside a region rather than between two studies of the region, "
          "which this module records on the term's note rather than in the "
          "contested flag.",
)


SEDATION = Endpoint(
    key="sedation",
    name="Sedation",
    assay="reduced spontaneous locomotion, loss of righting reflex, "
          "EEG-defined NREM, sleep latency",
    direction="higher = MORE sedated (less wakefulness)",
    terms=[
        _T("LC", "increase", "decrease", "Carter2010", "mouse",
           "optogenetic", "EEG sleep-wake state",
           "Frequency-dependent stimulation of locus coeruleus "
           "noradrenergic neurons caused immediate sleep-to-wake "
           "transitions and increased locomotor arousal; halorhodopsin "
           "inhibition suppressed wakefulness."),
        _T("LHA", "increase", "decrease", "Adamantidis2007", "mouse",
           "optogenetic", ("polysomnography", "sleep latency"),
           "Photostimulating hypocretin/orexin neurons increased the "
           "probability of transition to wakefulness from both NREM and "
           "REM and shortened wake latency, at 5-30 Hz but not at 1 Hz."),
        _T(("TMd", "TMv"), "decrease", "increase", "Fujita2017", "mouse",
           "optogenetic", "EEG sleep-wake state",
           "Acute optogenetic silencing of histaminergic tuberomammillary "
           "neurons during wakefulness was SUFFICIENT to rapidly and "
           "selectively induce slow-wave sleep.",
           note="[Yu2015] shows the GABA these same neurons co-release is "
                "a brake on their own histaminergic wake drive - a "
                "within-region sign conflict that argues the true "
                "magnitude is smaller, not that the sign flips. The CCF "
                "has no whole-nucleus TM row in the Knox region set, so "
                "the term names both children and splits one unit of "
                "evidence between them."),
        _T(("NDB", "SI", "MA"), "decrease", "increase", "Anaclet2015",
           "mouse", "chemogenetic", "EEG sleep-wake state",
           "Chemogenetic INHIBITION of basal forebrain GABAergic neurons "
           "INCREASED SLEEP, while activation produced sustained "
           "wakefulness and fast cortical rhythms. Cholinergic and "
           "glutamatergic activation changed state consolidation and EEG "
           "but not total wake time.",
           note="[Xu2015] finds parvalbumin+, cholinergic and "
                "glutamatergic basal-forebrain activation each waking, but "
                "somatostatin+ activation PROMOTING NREM - an embedded "
                "sleep-promoting subpopulation."),
        _T("PPN", "increase", "decrease", "Kroeger2017", "mouse",
           "chemogenetic", "EEG sleep-wake state",
           "Chemogenetic activation of glutamatergic pedunculopontine "
           "neurons induced prolonged wakefulness and inhibition reduced "
           "wake and increased NREM.",
           note="PPN is positive for locomotion and negative for sedation "
                "in this module, which is not a conflict - the two "
                "endpoints are different. It is also positive for "
                "sensorimotor gating, and [Azzopardi2018] says that "
                "weight must NOT be routed through cholinergic receptor "
                "occupancy."),
        _T("PB", "increase", "decrease", "Kaur2017", "mouse",
           "chemogenetic", "EEG sleep-wake state",
           "Activating external lateral parabrachial CGRP neurons caused "
           "wakefulness, and optogenetic inhibition prevented CO2-evoked "
           "arousal.",
           note="The paper's own limit: inhibition did not prevent arousal "
                "to a tone or to shaking, so the wake drive is "
                "stimulus-gated rather than tonic."),
        _T("VTA", "decrease", "increase", "EbanRothschild2016", "mouse",
           "chemogenetic", "polysomnography",
           "Chemogenetic and optogenetic inhibition of VTA dopamine "
           "neurons suppressed wakefulness even against salient stimuli, "
           "and stimulation initiated and maintained wake."),
        _T("VLPO", "increase", "increase", "Kroeger2018", "mouse",
           "optogenetic", ("EEG sleep-wake state", "sleep latency"),
           "Photostimulation of ventrolateral preoptic galanin neurons at "
           "1-4 Hz promoted sleep, optogenetic inhibition reduced sleep, "
           "and chemogenetic activation shortened sleep latency in an "
           "insomnia model.",
           note="The authors' artifact warning, carried: above 8 Hz, "
                "conduction block caused paradoxical waking, which is not "
                "a sign flip. [Lu2000]'s rat lesion is the classic anchor "
                "for the same sign - NREM time down 50-60%, with residual "
                "Fos+ cell count predicting NREM time at r = 0.77."),
        # --- contested -----------------------------------------------------
        _T("DR", "decrease", "decrease", "Oikonomou2019", "mouse",
           "optogenetic", "EEG sleep-wake state",
           "Raphe ablation INCREASED wakefulness and impaired the "
           "homeostatic sleep-deprivation response, and tonic optogenetic "
           "stimulation near baseline rates INDUCED SLEEP - against the "
           "classical assumption that the serotonergic raphe is "
           "wake-promoting.",
           contested=True,
           note="Contested twice over: BURST stimulation in the same paper "
                "induced wake, so the sign depends on firing mode; and "
                "[Cho2017]'s dorsal raphe DOPAMINE neurons, in the same "
                "voxels, promote wakefulness. The lowest-confidence term "
                "in this endpoint."),
    ],
    notes="Seven of the nine terms are the ascending arousal system and "
          "are therefore NEGATIVE. Two are positive: the ventrolateral "
          "preoptic nucleus, and the contested dorsal raphe. So the "
          "DEFAULT readout - contested excluded - is seven negative "
          "against one positive, which is close to minus the mean rate of "
          "the arousal nuclei. That is what the sleep-wake literature "
          "actually says and is worth seeing stated rather than "
          "discovered.",
)


WORKING_MEMORY = Endpoint(
    key="working_memory",
    name="Working memory",
    assay="delayed alternation, delayed non-match-to-sample/place, "
          "spontaneous alternation",
    direction="higher = BETTER working memory (more correct choices)",
    terms=[
        _T(("PL", "ILA"), "decrease", "decrease", "Kamigaki2017", "mouse",
           "optogenetic",
           ("delayed Go/No-Go", "delayed two-alternative forced choice"),
           "Inhibiting dorsomedial prefrontal pyramidal cells during the "
           "DELAY ALONE impaired performance, and disinhibiting them by "
           "driving VIP+ interneurons ENHANCED it and sharpened "
           "action-plan coding. The bidirectional result is what fixes "
           "the sign.",
           note="The manipulation was of dorsomedial prefrontal cortex as "
                "a whole, so this is one unit of evidence split between "
                "PL and ILA rather than two terms. Note that these two "
                "carry OPPOSITE signs in the reward endpoint, where the "
                "evidence does dissociate them."),
        _T("MD", "increase", "increase", "Bolkan2017", "mouse",
           "optogenetic", "delayed non-match-to-sample",
           "Mediodorsal-to-prefrontal input supports the maintenance "
           "phase, prefrontal delay-tuned spiking required it, and "
           "ENHANCING mediodorsal excitability was SUFFICIENT to enhance "
           "task performance - a gain-of-function, not just a lesion.",
           note="MD carries a negative analgesia term from [Meda2019]; "
                "different endpoints, no contradiction."),
        _T("RE", "decrease", "decrease", "Viena2018", "rat",
           "inactivation", "delayed non-match-to-sample",
           "Muscimol into nucleus reuniens impaired delayed "
           "non-match-to-sample alternation at 30, 60 and 120 s delays and "
           "abolished the win-shift strategy.",
           note="[Hallock2016] supplies the mechanism - reuniens "
                "inactivation abolishes the working-memory-specific "
                "hippocampal-prefrontal synchrony. Every usable reuniens "
                "study is RAT; see SEARCHED_AND_EMPTY."),
        _T("CA1", "decrease", "decrease", "Spellman2015", "mouse",
           "optogenetic", "spatial working memory T-maze",
           "Inhibiting direct hippocampal-to-prefrontal afferents impaired "
           "ENCODING, but not maintenance or retrieval, and prefrontal cue "
           "representations depended on hippocampal input only during that "
           "phase."),
        _T("CA3", "decrease", "decrease", "Song2020", "mouse",
           "optogenetic", "spatial working memory T-maze",
           "Optogenetic suppression of CA3 pyramidal neurons disrupted "
           "spatial working memory selectively in the choice (retrieval) "
           "phase, and suppressing medial-septal parvalbumin neurons "
           "projecting to CA3 reproduced it.",
           note="[Nakashiba2008] bounds this: CA3 OUTPUT is dispensable "
                "for incremental spatial learning and required mainly for "
                "one-trial and pattern-completion memory, so a large CA3 "
                "weight would overclaim."),
        _T("MS", "decrease", "decrease", "Gemzik2021", "rat",
           "optogenetic",
           ("conditional discrimination", "delayed non-match-to-place"),
           "Timed suppression of the medial septum impaired choice "
           "accuracy only during cue sampling on one task and only during "
           "the delay on the other, with no deficit on the "
           "working-memory-INdependent variant."),
    ],
    notes="Every term is POSITIVE, and no negatively signed region was "
          "found. That is a real property of this literature and not a "
          "convenience: the searches recorded in SEARCHED_AND_EMPTY turned "
          "up no region whose activation impairs working memory. The "
          "consequence is that this readout is, up to scale, the mean rate "
          "over its seven regions (six terms, one of which names PL and "
          "ILA together), so a drug that shifts global excitability will "
          "move it even with no circuit-specific effect. The "
          "inverted-U that prefrontal dopamine is famous for is exactly "
          "the shape a linear all-positive readout cannot hold, and is a "
          "second target for P7.2 alongside [ZhuoGebhart1997]'s raphe.",
)


SENSORIMOTOR_GATING = Endpoint(
    key="sensorimotor_gating",
    name="Sensorimotor gating",
    assay="prepulse inhibition of the acoustic startle reflex",
    direction="higher = MORE prepulse inhibition = better gating",
    terms=[
        _T("IC", "decrease", "decrease", "Li1998", "rat", "lesion",
           "prepulse inhibition",
           "Unilateral ibotenate lesion of inferior colliculus lowered "
           "prepulse inhibition across prepulse intensities (29-49 dB SPL) "
           "WITHOUT increasing baseline startle magnitude.",
           note="The startle-amplitude control is what makes this a gating "
                "effect rather than a reflex-strength artifact."),
        _T("SCm", "decrease", "decrease", "Fendt1994", "rat", "lesion",
           "prepulse inhibition",
           "Excitotoxic superior colliculus lesions decreased prepulse "
           "inhibition without affecting baseline startle or footshock "
           "sensitization.",
           note="[Fendt1999] is the gain-of-function half: picrotoxin into "
                "the same structure ENHANCED prepulse inhibition, again "
                "with no change in baseline startle. Loss- and "
                "gain-of-function agreeing is rare in this literature. "
                "The deep and intermediate layers are SCm, not SCs."),
        _T(("SI", "MA"), "increase", "decrease", "Kodsi1995", "rat",
           "microinjection", "prepulse inhibition",
           "GABA-A blockade in medial and central ventral pallidum - "
           "i.e. pallidal disinhibition - REDUCED prepulse inhibition with "
           "no effect on startle amplitude or latency, while GABA-B "
           "blockade did nothing.",
           note="The most internally consistent entry in this endpoint: "
                "[Swerdlow1990]'s intra-pallidal muscimol reversed an "
                "accumbens-driven gating deficit and [Kodsi1994]'s "
                "restored gating in ventral-striatum-lesioned rats, both "
                "in the same direction. Ventral pallidum is SI plus MA."),
        _T("BLA", "decrease", "decrease", "Wan1997", "rat", "lesion",
           "prepulse inhibition",
           "Quinolinic-acid basolateral amygdala lesions reduced prepulse "
           "inhibition without changing startle amplitude, and intra-BLA "
           "AP5 reduced it dose-dependently. Haloperidol did not rescue, "
           "so the effect is dopamine-independent."),
        # --- contested -----------------------------------------------------
        _T("PRNc", "increase", "decrease", "Davis1982", "rat", "lesion",
           "acoustic startle amplitude",
           "The caudal pontine reticular nucleus is the obligatory "
           "sensorimotor relay of the startle circuit - bilateral lesions "
           "abolish startle and stimulation evokes it at about 5 ms - and "
           "prepulse inhibition is implemented as inhibition OF these "
           "giant reticulospinal neurons, so more output means less "
           "gating.",
           contested=True,
           inferred="the cited experiment measured startle amplitude; "
                    "`measure` comes from the mechanistic argument that "
                    "gating is inhibition OF this nucleus",
           note="[Cano2021] shows the other population in the same "
                "structure runs the other way: photo-inhibiting "
                "GLYCINERGIC caudal pontine interneurons DECREASED "
                "prepulse inhibition. Two anti-correlated populations, one "
                "CCF node. The same paper is also the negative control "
                "for PRNr: lesions of the ORAL pontine reticular nucleus "
                "fail to abolish startle, so PRNr gets no weight."),
        _T("PPN", "decrease", "decrease", "Koch1993", "rat", "lesion",
           "prepulse inhibition",
           "Neurotoxic lesions of cholinergic pedunculopontine neurons "
           "significantly reduced prepulse inhibition with no change in "
           "baseline startle.",
           contested=True,
           note="[Azzopardi2018] contradicts it directly: transient "
                "optogenetic activation of specifically cholinergic "
                "pedunculopontine neurons ENHANCED startle and produced "
                "prepulse FACILITATION, from which the authors conclude "
                "that non-cholinergic neurons mediate gating. Whatever "
                "weight this region ends up with must NOT be routed "
                "through cholinergic receptor occupancy in the drug "
                "layer."),
        _T(("PL", "ILA"), "increase", "decrease", "Yang2024", "mouse",
           "microinjection", "prepulse inhibition",
           "A dopamine D1 agonist infused into medial prefrontal cortex "
           "disrupted prepulse inhibition.",
           contested=True,
           inferred="a D1 agonist is read as raising cortical activity",
           note="[Lacroix2000]'s ibotenate lesions of the same cortex "
                "INCREASED prepulse inhibition. Pharmacology and lesion "
                "disagree on sign, and no modern circuit-specific "
                "optogenetic prefrontal gating study was found; see "
                "SEARCHED_AND_EMPTY. The activity direction is doubly "
                "weak here: a D1 agonist is read as raising cortical "
                "activity, which is the usual account of a Gs-coupled "
                "receptor on pyramidal cells and is still an inference "
                "rather than the manipulation."),
        _T(("CA1", "SUB"), "increase", "decrease", "Wang2015", "rat",
           "microinjection", "prepulse inhibition",
           "Bilateral NMDA into ventral hippocampus induced hyperactivity "
           "and disrupted prepulse inhibition, bilateral more than "
           "unilateral.",
           contested=True,
           note="[Zhang2002] finds that muscimol or TTX inactivation of "
                "either ventral or dorsal hippocampus ALSO reduced "
                "prepulse inhibition. Together those make the hippocampal "
                "contribution an inverted U, and the sign here captures "
                "only the hyperactivation limb - the one drugs move."),
        _T("ACB", "increase", "decrease", "Swerdlow1990", "rat",
           "microinjection", "prepulse inhibition",
           "Dopamine infused into nucleus accumbens disrupted prepulse "
           "inhibition, and muscimol into ventral pallidum reversed that "
           "deficit.",
           contested=True,
           inferred="dopamine in the accumbens has NO unambiguous "
                    "activity direction: it excites D1- and inhibits "
                    "D2-expressing projection neurons",
           note="[Kodsi1994]'s quinolinic-acid lesions of ventral striatum "
                "ALSO reduced prepulse inhibition and raised startle "
                "amplitude - the same inverted-U problem as the "
                "hippocampus. And dopamine into the accumbens has no "
                "unambiguous activity direction at all, because it "
                "excites D1-expressing and inhibits D2-expressing "
                "projection neurons, which is the very split CP's "
                "locomotion term is contested for."),
    ],
    notes="Five of the nine terms are contested, the highest fraction in "
          "the module, and the reason is specific to this assay: the "
          "forebrain gating circuit is inverted-U in accumbens and "
          "hippocampus (both hyperactivation AND inactivation lower "
          "prepulse inhibition), the pontine relay contains two "
          "anti-correlated populations, and the prefrontal pharmacology "
          "and lesion literatures disagree on sign. This endpoint is also "
          "the one carrying the most cross-species extrapolation: eight of "
          "its nine terms are RAT."
)


#: The seven endpoints P7.1 asks for, in roadmap order. A test asserts the
#: count, because "seven endpoints" is the acceptance criterion.
ENDPOINT_ORDER = (
    "locomotion",
    "anxiety",
    "analgesia",
    "reward",
    "sedation",
    "working_memory",
    "sensorimotor_gating",
)

ENDPOINTS = {
    endpoint.key: endpoint
    for endpoint in (
        LOCOMOTION,
        ANXIETY,
        ANALGESIA,
        REWARD,
        SEDATION,
        WORKING_MEMORY,
        SENSORIMOTOR_GATING,
    )
}

#: Assay-restricted views of `analgesia`, and the one place in this module
#: where the evidence forced a structural decision rather than a weight.
#:
#: TWO papers ran BOTH assays and found them to dissociate, in opposite
#: manipulation directions: [Pastoriza1996] (lesion: hot-plate latency +82%,
#: tail flick unchanged) and [Hao2025] (activation: thermal facilitated,
#: tail flick unchanged). Two more dissociate tail flick from a different
#: supraspinal measure without running hot plate: [Roberts1994] (submedial
#: lesion: vocalization threshold down, tail flick unchanged) and
#: [MorganFranklin1990] (VTA lesion: formalin analgesia abolished, tail
#: flick intact). [Xie2023] and [Samineni2017] are NOT in this list, though
#: an earlier draft of this comment put them there: they ran neither assay,
#: which is what their own citation entries say.
#:
#: One shared weight vector is contradicted by the first two directly, and
#: the honest response is to keep the union endpoint the roadmap asks for
#: AND publish the two restrictions next to it, rather than to average.
#:
#: These are NOT extra members of the seven. They are views, and every
#: term in them is a term of `analgesia`.
ASSAY_VARIANTS = {
    "analgesia_tail_flick": ANALGESIA.restrict_to_assay(
        "tail flick",
        key="analgesia_tail_flick",
        name="Analgesia, tail-flick only (spinal reflex)",
        assay="tail flick",
    ),
    "analgesia_hot_plate": ANALGESIA.restrict_to_assay(
        "hot plate",
        key="analgesia_hot_plate",
        name="Analgesia, hot-plate only (supraspinal)",
        assay="hot plate",
    ),
}


SEARCHED_AND_EMPTY = """
P7.1: searched and empty, or found and deliberately not used
============================================================

Track E's template: a search that returned nothing usable is a result, and
it is recorded with what was searched and why each candidate failed.

REGIONS WITH NO USABLE EVIDENCE FOR AN ENDPOINT THEY OBVIOUSLY BELONG IN

* MOp / MOs and LOCOMOTION. Several query formulations for motor cortex
  against an open-field or photobeam ambulation readout returned only
  neurotoxicology, stroke and corticospinal-plasticity papers with no
  causal manipulation tied to an ambulation measure. Primary motor cortex
  carries NO locomotion weight in this module, which is a gap and not a
  claim that it is uninvolved.
* SSp and ANALGESIA. Queries pairing somatosensory cortex with hot plate,
  tail flick or thermal nociception returned three hits, all unusable (two
  human or rat imaging-electrophysiology papers and one 1989
  Chinese-language electroacupuncture study). The one strong causal mouse
  paper - optogenetically induced gamma in hindlimb S1 enhancing
  nociceptive sensitivity - uses von Frey MECHANICAL testing only. Rule 2
  of this codebase forbids substituting a different measurement for the
  one being modelled, so SSp carries no analgesia weight rather than an
  extrapolated one.
* AId / AIp (insular cortex) and ANALGESIA. Thirty-odd queries. The one
  modern paper with a hot plate (posterior insula CaMKIIa DREADDs) reports
  that ACTIVATION did not change hot-plate latency while INHIBITION
  lowered it AND lowered core body temperature; direct electrical
  activation of insular cortex produced an explicit NULL on the tail-flick
  reflex; and the rostral agranular insula has an antinociception-
  triggering pharmacology that points the opposite way from its GABA-B
  output population. For thermal latency specifically the honest reading
  of the retrieved primary data is "unresolved", so insular cortex carries
  no analgesia weight.
* CEA and REWARD. Five optogenetic or chemogenetic central-amygdala papers
  with a place-conditioning readout were retrieved and none was usable
  (CeA-to-SNc anxiety in an MPTP model, a Cre-line tool paper, VTA-to-CeA
  pain relief, CeA foraging). No central-amygdala reward weight.
* ACB and WORKING MEMORY. The nearest hits were a D2-receptor odour-span
  study and a cholinergic-depletion study, neither a regional inactivation
  on a delayed-alternation or DNMS assay. No accumbens working-memory
  weight.
* NDB / SI and WORKING MEMORY. Queries returned Alzheimer's-model and
  gamma-oscillation papers, none a direct basal-forebrain inactivation
  against a delayed-alternation readout. MS ([Gemzik2021]) was used
  instead, because it IS clean. The basal forebrain therefore carries a
  sedation weight and no working-memory weight.
* DG and SUB, WORKING MEMORY. No dentate- or subiculum-SELECTIVE causal
  working-memory study was found; every hit was a whole-hippocampus or
  fimbria-fornix manipulation, or an LTP-correlation paper. Both are
  omitted rather than given a weight borrowed from a lesion that did not
  isolate them.
* MRN / CUN and SEDATION. The modern cuneiform optogenetic literature is
  about locomotor speed and gait, not sleep-wake state; a mouse-restricted
  query for either against sleep or arousal returned zero. CUN therefore
  carries a locomotion weight and no sedation weight.
* GRN and SENSORIMOTOR GATING. One irrelevant hit, and [Davis1982]
  explicitly reports that lesions of the gigantocellular nucleus FAIL to
  abolish startle. Dropped.
* PRNr and SENSORIMOTOR GATING. [Davis1982] is again the negative control:
  startle cannot be evoked electrically from the ORAL pontine reticular
  nucleus and lesions of it do not abolish startle. This is a real
  anatomical trap given that the CCF names the two PRNc and PRNr, and
  PRNr deliberately carries no weight.
* Modern circuit-specific mPFC manipulation against PREPULSE INHIBITION.
  Queries surfaced only pharmacological-infusion and developmental papers.
  This is why the prefrontal gating term rests on drug microinfusion and
  is contradicted by the lesion literature.
* WHOLE-BRAIN IMAGING PRIORS, for every endpoint. Separate sweeps for a
  brain-wide c-Fos or light-sheet activation map of elevated-plus-maze
  open-arm exposure, of hot plate or tail flick, and of psychostimulant
  hyperlocomotion or place preference all came back empty of anything
  on-point. The one relevant methods paper validated its pipeline on
  haloperidol but reports no behavioural causality. Every regional weight
  in this module therefore comes from a causal manipulation, and NONE
  from an imaging prior - not by preference, but because no usable imaging
  prior exists.

MEASURED NULLS, WHICH ARE STRONGER THAN AN ABSENCE

* VTA carries NO analgesia weight, and that is a measurement.
  [MorganFranklin1990]: bilateral 6-OHDA lesion of the ventral tegmentum
  abolished amphetamine and morphine analgesia in the FORMALIN test but
  left morphine analgesia in the TAIL-FLICK test completely unchanged. An
  explicit zero, not a gap.
* There is no paper showing chemogenetic Gq excitation of VTA dopamine
  neurons produces place preference. All sufficiency evidence for that is
  optogenetic, and the one Gi-DREADD study reported a NULL on cocaine
  place preference while still reducing the drug's locomotor effect.

ACRONYMS THAT DO NOT EXIST IN THE CCF, AND WHAT WAS DONE INSTEAD

Every acronym in this module was checked against the P3.1 structure graph
([AllenCCF]). Four anatomical entities this literature names have no CCF
structure:

    ventral pallidum   -> SI + MA (there is no "VP" acronym; PALv exists
                          in the ontology but is not a Knox region row)
    RVM                -> RM + medial GRN + PGRNl. [Urban1994] shows the
                          effective tissue spans all three, so the RM term
                          is narrower than the papers behind it.
    SLEAc              -> not in the CCF at all; nearest is SI. Named on
                          [Sun2023]'s term rather than substituted.
    MLR, RMTg          -> not CCF structures. The mesencephalic locomotor
                          region is approximately CUN + PPN, which is why
                          those are two separate terms.

Two further CCF traps are recorded because getting either wrong would
silently invert a whole node:

    LH  is the LATERAL HABENULA, parent EPI. The lateral hypothalamus is
        LHA. Both appear in the reward endpoint, WITH OPPOSITE SIGNS.
    PPT is the POSTERIOR PRETECTAL nucleus. The pedunculopontine
        tegmental nucleus is PPN.

STRUCTURAL LIMITS THE CCF IMPOSES ON THIS MODULE

* ACB has no children in the CCF, so the core/shell dissociation that most
  of the place-preference literature turns on cannot be represented. One
  study dissociates the two endpoints INSIDE the nucleus in a single
  experiment - core to locomotion, medial shell to place preference - and
  a single ACB node averages over them.
* CP contains two intermingled, roughly equinumerous populations with
  opposite locomotor signs, so a scalar CP rate has no defensible sign.
  It is recorded contested and excluded by default.
* The stop-cell population of [Capelli2017] and [Bouvier2015] does not map
  onto one CCF region; only the positively signed lateral
  paragigantocellular term is taken.

SPECIES, STATED ONCE

The roadmap fixes the species at MOUSE. Of the 69 terms in the seven
endpoints, the sedation, working-memory and locomotion evidence is
predominantly mouse; the analgesia evidence is predominantly RAT, because
the tail-flick and hot-plate causal literature is 1980s-2010s rat
microinjection pharmacology and the modern mouse circuit work migrated to
Hargreaves and von Frey; and sensorimotor gating is eight of nine rat.
Every term carries its own species field, and no term's species was
changed to match the model.
"""
