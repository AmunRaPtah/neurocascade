"""P3.2: which sign a connectome edge gets, and on whose authority.

The problem this module exists to solve
---------------------------------------

Both published mouse connectomes are built from anterograde bulk tracer
fluorescence. Every weight in them is non-negative by construction, and the
measurement carries no sign at all: a dense projection from the
caudoputamen and a dense projection from the primary motor cortex look
identical to the tracer and do opposite things to the target. A
Wilson-Cowan circuit needs the sign, because `connectivity[i, j]` is a
signed drive coefficient.

The sign is therefore *not* in the connectome data (P3.1 says so in as many
words) and has to come from somewhere else. The one honest source is the
neurotransmitter phenotype of the source region's **projection** neurons:
a region whose long-range output neurons release glutamate drives its
targets, one whose long-range output neurons release GABA (or, in the
midbrain and hindbrain, GABA and/or glycine) suppresses them.

The rule, stated once
---------------------

    sign(edge j -> i) = sign of the transmitter released by the
                        LONG-RANGE PROJECTION neurons of source region j,

        glutamatergic  -> +1
        GABAergic      -> -1
        anything else  -> no sign (see "Regions that get no sign" below)

Three things follow from the words in that sentence, and each of them is a
decision, not a detail:

1. **The sign is a property of the source region only**, not of the pair.
   That is what makes it assignable at all from region-level anatomy, and
   it is an approximation: a source with two projection-neuron populations
   of opposite sign that innervate different targets is represented by
   whichever one predominates. The exceptions list below names the cases
   where this is known to be wrong.

2. **Projection neurons, not neurons.** Cortex contains a substantial
   minority of GABAergic interneurons (no percentage is quoted here,
   because none was sourced for this task and none is needed), and
   essentially all of them are local, so cortical *edges* are excitatory
   even though cortical *tissue* is not purely excitatory. The same
   argument runs the other way in the striatum, whose GABAergic spiny
   neurons both project and dominate numerically. So counting cells in a
   region would give the wrong answer in general; what matters is who owns
   the axons that leave.

3. **The rule is applied at the nearest ANCESTOR with a rule**, using the
   CCF hierarchy P3.1 already loads. `SIGN_RULES` below therefore states
   each rule at the coarsest level of the ontology at which it is true, and
   a more specific rule anywhere below it wins. `TH -> glutamatergic` plus
   `RT -> GABAergic` is the whole thalamic story in two lines, and adding a
   third line later cannot silently change the first two.

Regions that get no sign, and why that is the right answer
----------------------------------------------------------

Three cases fall outside the glutamate/GABA dichotomy, and all three are
recorded as an explicit transmitter class with **no sign** rather than
being pushed to +1 or -1:

* **Neuromodulatory nuclei** (VTA, SNc: dopaminergic; DR: serotonergic;
  LC: noradrenergic). A dopaminergic axon does not add a term to its
  target's net synaptic input; it changes how the target responds, through
  metabotropic receptors, which is precisely the layer this codebase
  already models explicitly (`ReceptorSystem` + `coupling`). Giving VTA a
  +1 in the connectivity matrix would double-count the mechanism the
  receptor layer exists to represent, and Phase 6 of the roadmap
  ("neuromodulatory projection systems") is where these edges are supposed
  to be handled. They are left unsigned here on purpose, not by omission.

* **Genuinely mixed output nuclei** (the basal forebrain: PALv, PALm).
  Their long-range output is cholinergic *and* GABAergic *and*
  glutamatergic in comparable measure, so no single sign is defensible.

* **Regions with no rule** - most of the hypothalamus, midbrain tegmentum,
  pons and medulla. Region-level transmitter identity there is either
  mixed or was not established here from a citable source, and this module
  refuses to guess. On the full Knox et al. 2019 291-region ipsilateral
  block the rules below sign **151 of 291 regions, carrying 90.9% of all
  outgoing tracer weight**; the 140 unsigned regions (131 with no rule, 5
  mixed, 4 neuromodulatory) carry the remaining 9.1%. Those numbers are
  asserted by a test, so they cannot drift silently, and they are the
  honest statement of how far this rule reaches.

  The sharpest consequence, stated rather than buried: **no region of the
  pons or the medulla has a sign**, so a network built at major-division
  resolution has two of its twelve nodes with no outgoing edges at all.
  That is a true statement about this rule table, it is visible in
  `BuildReport.summary()`, and it is the first thing an extension of this
  table should fix.

Known exceptions that are NOT encoded, listed so they are not forgotten
-----------------------------------------------------------------------

Each of these is a real violation of the rule above, is a minority of the
projection in question, and would need its own literature to encode as
anything other than a guess:

* **Cortical and hippocampal long-range GABAergic projection neurons.**
  They exist, they are a small minority of cortical projection neurons,
  and they are the reason "cortex is excitatory" is a statement about
  predominance rather than a fact about every axon [Melzer2020].
* **The pallidocortical projection is GABAergic AND cholinergic** - GPe
  sends both to frontal cortex [Saunders2015]. GPe is signed -1 here on
  the GABAergic majority.
* **Co-release.** Glutamate/GABA dual-transmitter neurons exist (62 of
  5,322 clusters brain-wide), and midbrain/hindbrain GABAergic clusters
  commonly co-release glycine [Yao2023]. Glycine is inhibitory, so the
  sign is unaffected; dual glutamate/GABA release has no single sign and
  is not represented.
* **IGL** (intergeniculate leaflet) is GABAergic/NPYergic and inherits
  thalamus's +1 here. It is a very small structure; encoding it was not
  done because no source for it was checked in this task.
* **Thalamic GABAergic local neurons** are present in mouse visual
  thalamus but are close to absent elsewhere in rodent dorsal thalamus
  [Arcelli1997], which is what makes the blanket `TH -> +1` defensible in
  this species and would not make it defensible in primate.
* **Striatal cholinergic interneurons** are the striatum's famous
  non-GABAergic cells, and they are interneurons: they do not leave the
  region, so they do not affect an edge's sign [Gerfen2011].

What would replace this table
-----------------------------

A hand-curated table of ~20 rules is what this task can support honestly.
The thing that would supersede it is [Yao2023]/[Zhang2023]: a whole-brain,
CCF-registered, mouse single-cell atlas that assigns a neurotransmitter to
every one of 5,322 clusters with a spatial location. Ingesting those
per-cluster assignments and computing a per-region projection-neuron
transmitter composition would turn every entry below into a measured
fraction, and would give the unsigned 9.1% an answer. That is a data
ingestion task in the shape of P3.1 and P4.1, not something to fake here.
"""

# --- transmitter classes ---------------------------------------------------

# Hoisted to `neurocascade.transmitter_sign` in P11.1 so the human layer need
# not import the mouse package for them. Re-exported here so every existing
# reference through this module keeps working, and so the signing rule stays
# documented where it is applied.
from ...transmitter_sign import (  # noqa: F401
    CHOLINERGIC,
    DOPAMINERGIC,
    GABAERGIC,
    GLUTAMATERGIC,
    MIXED,
    NORADRENERGIC,
    SEROTONERGIC,
    TRANSMITTER_SIGN,
)


# --- citations -------------------------------------------------------------

CITATIONS = {
    "Yao2023": (
        "Yao Z, van Velthoven CTJ, Kunst M, et al. 'A high-resolution "
        "transcriptomic and spatial atlas of cell types in the whole mouse "
        "brain.' Nature. 2023;624(7991):317-332. doi:10.1038/s41586-023-06812-z. "
        "Mouse, whole brain, CCF-registered: 5,322 transcriptomic clusters each "
        "with a neurotransmitter assignment; reports glutamate/GABA as the "
        "dominant dichotomy, glycine co-release in midbrain/hindbrain GABAergic "
        "clusters, and 62 glutamate-GABA dual-transmitter clusters."
    ),
    "Zhang2023": (
        "Zhang M, Pan X, Jung W, et al. 'Molecularly defined and spatially "
        "resolved cell atlas of the whole mouse brain.' Nature. "
        "2023;624(7991):343-354. doi:10.1038/s41586-023-06808-9. Mouse, whole "
        "brain, MERFISH; the spatial companion to Yao2023."
    ),
    "Melzer2020": (
        "Melzer S, Monyer H. 'Diversity and function of corticopetal and "
        "corticofugal GABAergic projection neurons.' Nature Reviews "
        "Neuroscience. 2020;21(9):499-515. doi:10.1038/s41583-020-0344-9. "
        "Review, rodent: long-range GABAergic projection neurons of cortex and "
        "hippocampus exist and are a minority of cortical projection neurons."
    ),
    "Arcelli1997": (
        "Arcelli P, Frassoni C, Regondi MC, De Biasi S, Spreafico R. 'GABAergic "
        "neurons in mammalian thalamus: a marker of thalamic complexity?' Brain "
        "Research Bulletin. 1997;42(1):27-37. doi:10.1016/S0361-9230(96)00107-4. "
        "Comparative (bat, mouse, rat, guinea pig, rabbit, cat, monkey, human): "
        "GABAergic local-circuit neurons are numerous in the visual thalamus of "
        "all species but are close to absent elsewhere in rodent dorsal "
        "thalamus (<1% of neurons in guinea-pig ventrolateral nucleus), and "
        "increase with species complexity."
    ),
    "Pinault2004": (
        "Pinault D. 'The thalamic reticular nucleus: structure, function and "
        "concept.' Brain Research Reviews. 2004;46(1):1-31. "
        "doi:10.1016/j.brainresrev.2004.04.008. Rodent: RT is a sheet of "
        "GABAergic neurons that innervates the dorsal thalamus and, uniquely "
        "among thalamic nuclei, does not project to cortex."
    ),
    "Gerfen2011": (
        "Gerfen CR, Surmeier DJ. 'Modulation of striatal projection systems by "
        "dopamine.' Annual Review of Neuroscience. 2011;34:441-466. "
        "doi:10.1146/annurev-neuro-061010-113641. Rodent: the striatum's output "
        "is carried by GABAergic spiny projection neurons in two classes (D1 "
        "direct, D2 indirect); the non-GABAergic striatal cells (cholinergic "
        "interneurons) are interneurons."
    ),
    "Smith1998": (
        "Smith Y, Bevan MD, Shink E, Bolam JP. 'Microcircuitry of the direct and "
        "indirect pathways of the basal ganglia.' Neuroscience. "
        "1998;86(2):353-387. doi:10.1016/S0306-4522(98)00004-9. Rat and primate: "
        "GPe, GPi/entopeduncular nucleus and SNr projection neurons are "
        "GABAergic; the subthalamic nucleus is the one excitatory node of the "
        "basal ganglia."
    ),
    "Saunders2015": (
        "Saunders A, Oldenburg IA, Berezovskii VK, et al. 'A direct GABAergic "
        "output from the basal ganglia to frontal cortex.' Nature. "
        "2015;521(7550):85-89. doi:10.1038/nature14179. Mouse: GPe projects "
        "directly to frontal cortex via two cell types, one GABAergic and one "
        "ChAT-expressing."
    ),
    "Do2016": (
        "Do JP, Xu M, Lee SH, Mu WQ, Zhang S, Cui J, Zhang F, Dan Y. 'Cell "
        "type-specific long-range connections of basal forebrain circuit.' "
        "eLife. 2016;5:e13214. doi:10.7554/eLife.13214. Mouse: whole-brain "
        "input and output mapping of basal forebrain cholinergic, glutamatergic "
        "and PV+/SOM+ GABAergic neurons; the three transmitter classes have "
        "distinct long-range outputs, so the region's output is not of one sign."
    ),
    "Schweizer2014": (
        "Schweizer N, Pupe S, Arvidsson E, Nordenankar K, Smith-Anttila CJA, "
        "Mahmoudi S, Andren A, Dumas S, Rajagopalan A, Levesque D, Leao RN, "
        "Wallen-Mackenzie A. 'Limiting glutamate transmission in a "
        "Vglut2-expressing subpopulation of the subthalamic nucleus is "
        "sufficient to cause hyperlocomotion.' PNAS. 2014;111(21):7837-7842. "
        "doi:10.1073/pnas.1323499111. Mouse: STN neurons are Vglut2-expressing "
        "glutamatergic neurons, and removing Vglut2 from them is sufficient to "
        "change behaviour."
    ),
    "Mitrofanis2005": (
        "Mitrofanis J. 'Some certainty for the \"zone of uncertainty\"? "
        "Exploring the function of the zona incerta.' Neuroscience. "
        "2005;130(1):1-15. doi:10.1016/j.neuroscience.2004.08.017. Rodent: the "
        "majority of zona incerta neurons are GABAergic, and they project "
        "widely."
    ),
    "Morales2017": (
        "Morales M, Margolis EB. 'Ventral tegmental area: cellular "
        "heterogeneity, connectivity and behaviour.' Nature Reviews "
        "Neuroscience. 2017;18(2):73-85. doi:10.1038/nrn.2016.165. Rodent: VTA "
        "output is carried by dopaminergic, GABAergic and glutamatergic "
        "projection neurons, with dopamine-releasing neurons the defining "
        "population; several co-release."
    ),
    "Uusisaari2011": (
        "Uusisaari M, Knopfel T. 'Functional classification of neurons in the "
        "mouse lateral cerebellar nuclei.' The Cerebellum. 2011;10(4):637-646. "
        "doi:10.1007/s12311-010-0240-3. Mouse: the cerebellar nuclei contain "
        "large glutamatergic projection neurons alongside GABAergic "
        "(nucleo-olivary) and glycinergic types, identified with GAD67-GFP and "
        "GlyT2-GFP lines."
    ),
}


# --- the rules -------------------------------------------------------------


class TransmitterRule(object):
    """One region-level statement of projection-neuron transmitter identity.

    `acronym` and `structure_id` are BOTH written down and are checked
    against each other when the rule set is resolved against an ontology.
    Either alone would fail quietly: an id alone is unreadable and would
    survive a re-parenting, an acronym alone would survive an upstream
    reassignment of that acronym to a different structure. Together, a
    disagreement is a loud error, which is the same discipline
    `ontology.MAJOR_DIVISION_IDS` uses.
    """

    __slots__ = ("acronym", "structure_id", "transmitter", "citations", "note")

    def __init__(self, acronym, structure_id, transmitter, citations, note):
        if transmitter not in TRANSMITTER_SIGN:
            raise ValueError("unknown transmitter class %r" % (transmitter,))
        missing = [key for key in citations if key not in CITATIONS]
        if missing:
            raise ValueError(
                "rule for %s cites unknown key(s) %s" % (acronym, missing)
            )
        if not citations:
            raise ValueError(
                "rule for %s has no citation; an unsourced sign is exactly "
                "what this module exists to prevent" % acronym
            )
        self.acronym = acronym
        self.structure_id = structure_id
        self.transmitter = transmitter
        self.citations = tuple(citations)
        self.note = note

    @property
    def sign(self):
        return TRANSMITTER_SIGN[self.transmitter]

    def __repr__(self):
        return "TransmitterRule(%r, %d, %r)" % (
            self.acronym,
            self.structure_id,
            self.transmitter,
        )


#: The sign-assignment table. Each rule is stated at the COARSEST level of
#: the CCF ontology at which it holds, and any rule below it wins for its
#: own subtree (see `SignRules.assign`). Order here is the CCF's own
#: top-down order and has no effect on lookup.
SIGN_RULES = (
    # --- cerebral cortex and its subplate: glutamatergic pyramidal output ---
    TransmitterRule(
        "Isocortex", 315, GLUTAMATERGIC, ("Yao2023", "Melzer2020"),
        "Isocortical long-range output is carried by glutamatergic pyramidal "
        "neurons; the GABAergic minority is overwhelmingly local. Long-range "
        "GABAergic cortical projection neurons exist and are a minority "
        "[Melzer2020].",
    ),
    TransmitterRule(
        "OLF", 698, GLUTAMATERGIC, ("Yao2023",),
        "Olfactory-area output neurons (mitral/tufted cells of MOB, piriform "
        "and anterior olfactory pyramidal cells) are glutamatergic. The "
        "olfactory bulb's very large GABAergic granule/periglomerular "
        "population is local and axonless.",
    ),
    TransmitterRule(
        "HPF", 1089, GLUTAMATERGIC, ("Yao2023", "Melzer2020"),
        "Hippocampal formation output (CA1/CA3 pyramidal, dentate granule, "
        "entorhinal and subicular principal cells) is glutamatergic; "
        "hippocampo-septal long-range GABAergic cells are the documented "
        "minority exception [Melzer2020].",
    ),
    TransmitterRule(
        "CTXsp", 703, GLUTAMATERGIC, ("Yao2023",),
        "Cortical subplate (claustrum, endopiriform, and the pallial amygdalar "
        "nuclei LA/BLA/BMA/PA) projection neurons are glutamatergic "
        "pyramidal-type cells. Note the CCF places the GABAergic amygdalar "
        "nuclei CEA/MEA in STR/sAMY, not here, so this rule does not have to "
        "carve them out.",
    ),
    # --- striatum and pallidum: GABAergic projection output ---
    TransmitterRule(
        "STR", 477, GABAERGIC, ("Gerfen2011", "Yao2023"),
        "Striatal output in all four CCF subdivisions (STRd caudoputamen, STRv "
        "accumbens/olfactory tubercle, LSX lateral septum, sAMY central and "
        "medial amygdala) is carried by GABAergic spiny projection neurons. "
        "Cholinergic cells here are interneurons and do not leave the region.",
    ),
    TransmitterRule(
        "PAL", 803, GABAERGIC, ("Smith1998", "Yao2023"),
        "Pallidal projection neurons (GPe, GPi/entopeduncular, BST) are "
        "GABAergic. GPe additionally sends a ChAT+ projection to frontal "
        "cortex [Saunders2015], which this sign does not represent.",
    ),
    TransmitterRule(
        "PALv", 835, MIXED, ("Do2016",),
        "Ventral pallidum / substantia innominata is basal forebrain: its "
        "long-range output is cholinergic AND glutamatergic AND GABAergic in "
        "comparable measure, with different targets per class. No single sign "
        "is defensible, so it gets none.",
    ),
    TransmitterRule(
        "PALm", 826, MIXED, ("Do2016",),
        "Medial septal complex / diagonal band, same argument as PALv: the "
        "septohippocampal projection is cholinergic and GABAergic and "
        "glutamatergic.",
    ),
    # --- thalamus: glutamatergic relay, with the one classical exception ---
    TransmitterRule(
        "TH", 549, GLUTAMATERGIC, ("Arcelli1997", "Yao2023"),
        "Thalamocortical and thalamostriatal relay neurons are glutamatergic. "
        "This blanket rule is defensible in MOUSE specifically: rodent dorsal "
        "thalamus outside the visual nuclei is almost devoid of GABAergic "
        "local neurons [Arcelli1997], and none of them project out anyway.",
    ),
    TransmitterRule(
        "RT", 262, GABAERGIC, ("Pinault2004", "Yao2023"),
        "The thalamic reticular nucleus is the exception to TH: a GABAergic "
        "sheet whose entire output inhibits the dorsal thalamus.",
    ),
    # --- hypothalamus: no default; two nuclei with unambiguous identity ---
    TransmitterRule(
        "STN", 470, GLUTAMATERGIC, ("Schweizer2014", "Smith1998"),
        "The subthalamic nucleus is glutamatergic (Vglut2) and is the "
        "excitatory node of the indirect pathway this codebase already "
        "models. The CCF places STN inside HY, which has no default rule, so "
        "without this entry the D2 scenario's key excitatory edge would be "
        "unsigned.",
    ),
    TransmitterRule(
        "ZI", 797, GABAERGIC, ("Mitrofanis2005", "Yao2023"),
        "Zona incerta is predominantly GABAergic and projects very widely.",
    ),
    # --- midbrain: no default; nuclei this project models by name ---
    TransmitterRule(
        "SNr", 381, GABAERGIC, ("Smith1998", "Yao2023"),
        "Substantia nigra pars reticulata is the GABAergic output nucleus of "
        "the basal ganglia.",
    ),
    TransmitterRule(
        "VTA", 749, DOPAMINERGIC, ("Morales2017", "Yao2023"),
        "VTA's defining projection neurons are dopaminergic (with GABAergic "
        "and glutamatergic projection populations alongside). Dopamine acts "
        "through the metabotropic receptor layer this codebase models "
        "explicitly, so VTA gets a transmitter class and NO sign rather than "
        "a fabricated +1.",
    ),
    TransmitterRule(
        "SNc", 374, DOPAMINERGIC, ("Morales2017", "Yao2023"),
        "Substantia nigra pars compacta: the nigrostriatal dopamine "
        "projection. Same argument as VTA.",
    ),
    TransmitterRule(
        "DR", 872, SEROTONERGIC, ("Yao2023",),
        "Dorsal raphe: the principal ascending serotonergic projection "
        "nucleus (with glutamatergic and GABAergic populations). Unsigned for "
        "the same reason as VTA.",
    ),
    # --- pons ---
    TransmitterRule(
        "LC", 147, NORADRENERGIC, ("Yao2023",),
        "Locus coeruleus: the sole source of cortical noradrenaline. Unsigned "
        "for the same reason as VTA; it is also a coupled node in this "
        "codebase's multi-region opioid scenario, so naming it here matters.",
    ),
    # --- cerebellum ---
    TransmitterRule(
        "CBX", 528, GABAERGIC, ("Yao2023",),
        "Cerebellar cortex output is the Purkinje cell axon, which is "
        "GABAergic.",
    ),
    TransmitterRule(
        "CBN", 519, GLUTAMATERGIC, ("Uusisaari2011", "Yao2023"),
        "The cerebellar nuclei carry the cerebellum's output to the rest of "
        "the brain through large glutamatergic projection neurons; the "
        "GABAergic nucleo-olivary and glycinergic cells are the minority "
        "exception this sign does not represent.",
    ),
)


# --- resolution against an ontology ----------------------------------------


class SignAssignment(object):
    """What sign one region got, and which rule gave it.

    `rule` is None when no ancestor of the region carries a rule; then
    `transmitter` is None and `sign` is None. That is a first-class outcome,
    not a failure: see the module docstring.
    """

    __slots__ = ("structure_id", "acronym", "rule")

    def __init__(self, structure_id, acronym, rule):
        self.structure_id = structure_id
        self.acronym = acronym
        self.rule = rule

    @property
    def transmitter(self):
        return None if self.rule is None else self.rule.transmitter

    @property
    def sign(self):
        return None if self.rule is None else self.rule.sign

    @property
    def is_signed(self):
        return self.sign is not None

    @property
    def citations(self):
        return () if self.rule is None else self.rule.citations

    @property
    def rule_acronym(self):
        """The ancestor the rule was stated on ('Isocortex' for MOp)."""
        return None if self.rule is None else self.rule.acronym

    def __repr__(self):
        return "SignAssignment(%r, %r, sign=%r)" % (
            self.acronym,
            self.transmitter,
            self.sign,
        )


class SignRules(object):
    """`SIGN_RULES` bound to one ontology, with nearest-ancestor lookup."""

    def __init__(self, ontology, rules=SIGN_RULES):
        self.ontology = ontology
        self.rules = tuple(rules)
        by_id = {}
        for rule in self.rules:
            structure = ontology[rule.structure_id]
            if structure.acronym != rule.acronym:
                raise ValueError(
                    "sign rule names structure %d as %r but this ontology "
                    "calls it %r; the CCF has been revised under this table "
                    "and the rule must be re-checked against the literature, "
                    "not silently re-pointed"
                    % (rule.structure_id, rule.acronym, structure.acronym)
                )
            if rule.structure_id in by_id:
                raise ValueError(
                    "two sign rules for structure %r" % (rule.acronym,)
                )
            by_id[rule.structure_id] = rule
        self._by_id = by_id

    def assign(self, structure_id):
        """Sign assignment for one region, from its nearest ruled ancestor."""
        structure = self.ontology[structure_id]
        anchor = self.ontology.ancestor_at(structure.id, self._by_id.keys())
        return SignAssignment(
            structure_id=structure.id,
            acronym=structure.acronym,
            rule=None if anchor is None else self._by_id[anchor],
        )

    def assign_all(self, structure_ids):
        return [self.assign(i) for i in structure_ids]

    def citations_used(self, structure_ids):
        """Citation keys actually reached by these regions, sorted."""
        used = set()
        for assignment in self.assign_all(structure_ids):
            used.update(assignment.citations)
        return sorted(used)
