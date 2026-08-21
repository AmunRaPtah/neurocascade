"""Which sign a human connectome edge gets, and on whose authority.

The problem this module exists to solve
---------------------------------------

The human structural matrix ingested by `human/sources.py` is diffusion
tractography. dMRI reconstructs white-matter bundles: it can say HOW MUCH
fiber there is between two regions, but it can say neither the DIRECTION of
the projection nor its SIGN (a streamline carries no excitatory/inhibitory
tag). The mouse connectomes in `connectome/` are anterograde tracer data and
are still unsigned by construction; the human matrix is unsigned for a
second, harder reason - the measurement itself has no direction. So the sign
of every human edge has to come from outside the matrix, exactly as it does
for the mouse edges, and the one honest outside source is the
neurotransmitter phenotype of the SOURCE region's long-range projection
neurons.

The rule, stated once (same rule as `connectome/neurotransmitter.py`)
-------------------------------------------------------------------------

    sign(edge j -> i) = sign of the transmitter released by the
                        LONG-RANGE PROJECTION neurons of source region j,

        glutamatergic  -> +1
        GABAergic      -> -1
        anything else  -> no sign

The transmitter classes and their `TRANSMITTER_SIGN` map are imported from
`connectome/neurotransmitter.py`, because that part is species-neutral model
physics (a glutamatergic drive is excitatory in a mouse or a human
Wilson-Cowan node). What is human-specific - and therefore lives HERE, not
there - is the per-region assignment table and every citation behind it.
`connectome/neurotransmitter.py` assigns signs through the CCF ontology
hierarchy; the human Desikan-Killiany atlas is a flat 82-region list, so the
human table assigns signs per atlas region directly (a cortex-wide rule plus
one rule per subcortical nucleus).

Three things follow from the rule, and each is a decision
---------------------------------------------------------

1. **The sign is a property of the source region, applied per direction.**
   Because dMRI is undirected, one symmetric weight `W[i, j]` is entered
   twice: the `i -> j` direction is signed by region `i`'s transmitter and
   the `j -> i` direction by region `j`'s. For a cortico-striatal bundle
   this produces `+W` cortex->striatum and `-W` striatum->cortex. Whether
   the bundle really carries fibres in BOTH directions is exactly what dMRI
   cannot measure; the convention here is "sign each direction by that
   direction's source", and `build.py` says so in its docstring. The
   alternative - signing the whole symmetric matrix by one side - would
   silently assert a direction the measurement does not have.

2. **Projection neurons, not neurons.** Cortex contains a large minority of
   GABAergic cells, but they are overwhelmingly local interneurons; the
   axons that leave the cortex are the glutamatergic pyramidal ones. The
   same argument runs the other way in the striatum: its GABAergic spiny
   projection neurons dominate the output, while the non-GABAergic cells
   (cholinergic interneurons) do not leave. Counting cells would give the
   wrong sign in both cases.

3. **A genuinely mixed region gets no sign rather than a guessed one.** The
   Desikan-Killiany atlas collapses the whole amygdala into one coarse
   region (`Lamyg`/`Ramyg`). The basolateral complex is pallial/cortical
   and glutamatergic; the central and medial nuclei are striatal-like and
   GABAergic. One region containing both has no single defensible sign, so
   it is `MIXED` and unsigned. `build.py` excludes unsigned source columns
   by default and records how much of the fibre weight that cost.

What the table reaches
----------------------

All 82 ENIGMA regions are covered: 68 cortical areas (glutamatergic), the
hippocampus and thalamus (glutamatergic), the four striatal/pallidal output
nuclei accumbens, caudate, putamen and pallidum (GABAergic, 8 regions), and
the amygdala (mixed, 2 regions). That is 80 signed regions carrying their
projection weight, and 2 regions left unsigned because the atlas's own
coarseness refuses a sign. Those counts are asserted by a test so they
cannot drift silently.

The obvious next increment
--------------------------

The four GABAergic nuclei here are signed from classical, primate-inclusive
basal-ganglia anatomy, and the cortex from the canonical glutamatergic
pyramidal story. The mouse table carries a much finer per-nucleus table
backed by single-cell atlases (Yao2023/Zhang2023); the human analogue would
be a whole-brain human transcriptomic atlas (e.g. the Allen Human Brain
Atlas or HCA cell-type data) mapped onto the same Desikan-Killiany labels.
Until then this table is the coarse, cited version, and its coarseness is
stated rather than hidden.
"""

from ..transmitter_sign import (  # species-neutral: see P11.1
    CHOLINERGIC,
    DOPAMINERGIC,
    GABAERGIC,
    GLUTAMATERGIC,
    MIXED,
    NORADRENERGIC,
    SEROTONERGIC,
    TRANSMITTER_SIGN,
)

__all__ = [
    "CITATIONS",
    "RegionSignRule",
    "SignAssignment",
    "SignRules",
    "SIGN_RULES",
    "TRANSMITTER_SIGN",
    "GLUTAMATERGIC",
    "GABAERGIC",
    "MIXED",
    "DOPAMINERGIC",
    "SEROTONERGIC",
    "NORADRENERGIC",
    "CHOLINERGIC",
]


# --- citations -------------------------------------------------------------
#
# These are the human-side citations only. The species-neutral transmitter
# class semantics are shared with `connectome/neurotransmitter.py`, which is
# why its constants are imported above; its CITATIONS table is mouse work and
# is deliberately NOT imported here.

CITATIONS = {
    "DeFelipe1992": (
        "DeFelipe J, Farinas I. 'The pyramidal neuron of the cerebral cortex: "
        "morphological and chemical characteristics in the synaptic inputs.' "
        "Progress in Neurobiology. 1992;39(6):563-607. "
        "doi:10.1016/0301-0082(92)90015-7. The cerebral cortex's principal "
        "long-range projection neurons are the glutamatergic pyramidal cells."
    ),
    "Melzer2020": (
        "Melzer S, Monyer H. 'Diversity and function of corticopetal and "
        "corticofugal GABAergic projection neurons.' Nature Reviews "
        "Neuroscience. 2020;21(9):499-515. doi:10.1038/s41583-020-0344-9. "
        "Review, rodent: long-range GABAergic projection neurons of cortex and "
        "hippocampus exist and are a minority of cortical projection neurons; "
        "the point this table relies on is the majority, glutamatergic output."
    ),
    "Fonnum1984": (
        "Fonnum F. 'Glutamate: a neurotransmitter in mammalian brain.' Journal "
        "of Neurochemistry. 1984;42(1):1-11. doi:10.1111/j.1471-4159.1984."
        "tb09689.x. Glutamate is the major excitatory transmitter of the "
        "principal neurons of the hippocampal formation and cortex."
    ),
    "Jones2007": (
        "Jones EG. 'The Thalamus.' 2nd ed. Cambridge University Press; 2007. "
        "ISBN 978-0-521-85881-1. Thalamocortical and thalamostriatal relay "
        "neurons are glutamatergic projection neurons in the primate and "
        "human thalamus."
    ),
    "Arcelli1997": (
        "Arcelli P, Frassoni C, Regondi MC, De Biasi S, Spreafico R. 'GABAergic "
        "neurons in mammalian thalamus: a marker of thalamic complexity?' Brain "
        "Research Bulletin. 1997;42(1):27-37. doi:10.1016/S0361-9230(96)00107-4. "
        "Comparative (bat, mouse, rat, guinea pig, rabbit, cat, monkey, human): "
        "GABAergic local-circuit neurons are numerous in the visual thalamus "
        "of all species and increase with species complexity, but they are "
        "local-circuit cells that do not project out of the thalamus; the "
        "long-range relay output stays glutamatergic."
    ),
    "Gerfen2011": (
        "Gerfen CR, Surmeier DJ. 'Modulation of striatal projection systems by "
        "dopamine.' Annual Review of Neuroscience. 2011;34:441-466. "
        "doi:10.1146/annurev-neuro-061010-113641. Rodent: the striatum's "
        "long-range output is carried by GABAergic spiny projection neurons; "
        "the non-GABAergic striatal cells are cholinergic interneurons that do "
        "not leave the region. The GABAergic-output identity of the spiny "
        "projection neurons is conserved across mammals."
    ),
    "Yelnik2002": (
        "Yelnik J. 'Functional anatomy of the basal ganglia.' Movement "
        "Disorders. 2002;17 Suppl 3:S15-21. doi:10.1002/mds.10138. Human basal "
        "ganglia: the caudate, putamen and nucleus accumbens are striatal, and "
        "their projection neurons are the GABAergic medium spiny neurons."
    ),
    "Smith1998": (
        "Smith Y, Bevan MD, Shink E, Bolam JP. 'Microcircuitry of the direct "
        "and indirect pathways of the basal ganglia.' Neuroscience. "
        "1998;86(2):353-387. doi:10.1016/S0306-4522(98)00004-9. Rat and "
        "primate: GPe and GPi (the pallidum's two segments) have GABAergic "
        "projection neurons; the subthalamic nucleus is the one excitatory "
        "node of the basal ganglia and is not in this atlas."
    ),
    "Parent1995": (
        "Parent A, Hazrati LN. 'Functional anatomy of the basal ganglia. I. "
        "The cortico-basal ganglia-thalamo-cortical loop.' Brain Research "
        "Reviews. 1995;20(1):91-127. doi:10.1016/0165-0173(94)00007-C. Primate "
        "and human: the pallidum is the GABAergic output station of the basal "
        "ganglia to the thalamus."
    ),
    "Sah2003": (
        "Sah P, Faber ESL, Lopez De Armentia M, Power J. 'The amygdaloid "
        "complex: anatomy and physiology.' Physiological Reviews. "
        "2003;83(3):803-834. doi:10.1152/physrev.00002.2003. The basolateral "
        "amygdala's principal neurons are glutamatergic (cortical-like), while "
        "the central and medial nuclei are GABAergic (striatal-like); the "
        "whole-amygdala region therefore has no single projection-neuron "
        "transmitter."
    ),
}


# --- the rules -------------------------------------------------------------


class RegionSignRule(object):
    """One region-level statement of projection-neuron transmitter identity.

    The human atlas is flat, so a rule is keyed by `kind` plus `suffix`:
    one cortex-wide rule (`kind="cortex"`, `suffix=None`) and one rule per
    subcortical nucleus. Both are checked for uniqueness when the rule set is
    resolved, so a duplicated or contradictory rule is a loud error.
    """

    __slots__ = ("kind", "suffix", "transmitter", "citations", "note")

    def __init__(self, kind, suffix, transmitter, citations, note):
        if kind not in ("cortex", "subcortex"):
            raise ValueError("unknown atlas kind %r" % (kind,))
        if kind == "cortex" and suffix is not None:
            raise ValueError(
                "the cortex-wide rule must have suffix=None; a per-area "
                "cortical rule would silently re-introduce a hierarchy this "
                "atlas does not have"
            )
        if kind == "subcortex" and not suffix:
            raise ValueError("a subcortical rule needs a nucleus suffix")
        if transmitter not in TRANSMITTER_SIGN:
            raise ValueError("unknown transmitter class %r" % (transmitter,))
        missing = [key for key in citations if key not in CITATIONS]
        if missing:
            raise ValueError(
                "rule for %s cites unknown key(s) %s" % (kind, missing)
            )
        if not citations:
            raise ValueError(
                "rule for %s has no citation; an unsourced sign is exactly "
                "what this module exists to prevent" % kind
            )
        self.kind = kind
        self.suffix = suffix
        self.transmitter = transmitter
        self.citations = tuple(citations)
        self.note = note

    @property
    def sign(self):
        return TRANSMITTER_SIGN[self.transmitter]

    def __repr__(self):
        return "RegionSignRule(%r, %r, %r)" % (
            self.kind,
            self.suffix,
            self.transmitter,
        )


_CORTEX_NOTE = (
    "Cerebral-cortex long-range output is carried by glutamatergic pyramidal "
    "projection neurons; the GABAergic population is overwhelmingly local "
    "interneurons. Long-range GABAergic cortical projection neurons exist and "
    "are a minority [Melzer2020]."
)
_HIPPO_NOTE = (
    "Hippocampal formation output (CA pyramidal cells, dentate granule cells, "
    "subiculum and entorhinal principal cells) is glutamatergic; long-range "
    "hippocampal GABAergic projection cells are the documented minority "
    "[Melzer2020]."
)
_THAL_NOTE = (
    "Thalamocortical and thalamostriatal relay neurons are glutamatergic "
    "[Jones2007]. Human thalamus has more GABAergic local-circuit neurons "
    "than rodent dorsal thalamus [Arcelli1997], but they do not project out, "
    "so the long-range relay sign stays +1."
)
_STRIATAL_NOTE = (
    "Striatal output is carried by GABAergic medium spiny projection neurons "
    "[Gerfen2011, Yelnik2002]; the striatal cholinergic cells are "
    "interneurons and do not leave the region."
)
_PALLIDAL_NOTE = (
    "The pallidum (external and internal segments) is the basal ganglia's "
    "GABAergic output station [Smith1998, Parent1995]."
)
_AMYGDALA_NOTE = (
    "The Desikan-Killiany 'amyg' label is one coarse region containing the "
    "glutamatergic basolateral complex AND the GABAergic central/medial "
    "nuclei [Sah2003]. No single projection-neuron transmitter is defensible, "
    "so the region is MIXED and unsigned rather than guessed."
)

#: The per-region sign table. One cortex-wide rule plus one rule for each of
#: the seven subcortical nuclei in the ENIGMA 'aparc' atlas. Every rule is
#: cited; order has no effect on lookup.
SIGN_RULES = (
    RegionSignRule(
        "cortex", None, GLUTAMATERGIC, ("DeFelipe1992", "Melzer2020"),
        _CORTEX_NOTE,
    ),
    RegionSignRule(
        "subcortex", "hippo", GLUTAMATERGIC, ("Fonnum1984", "Melzer2020"),
        _HIPPO_NOTE,
    ),
    RegionSignRule(
        "subcortex", "thal", GLUTAMATERGIC, ("Jones2007", "Arcelli1997"),
        _THAL_NOTE,
    ),
    RegionSignRule(
        "subcortex", "accumb", GABAERGIC, ("Gerfen2011", "Yelnik2002"),
        _STRIATAL_NOTE,
    ),
    RegionSignRule(
        "subcortex", "caud", GABAERGIC, ("Gerfen2011", "Yelnik2002"),
        _STRIATAL_NOTE,
    ),
    RegionSignRule(
        "subcortex", "put", GABAERGIC, ("Gerfen2011", "Yelnik2002"),
        _STRIATAL_NOTE,
    ),
    RegionSignRule(
        "subcortex", "pal", GABAERGIC, ("Smith1998", "Parent1995"),
        _PALLIDAL_NOTE,
    ),
    RegionSignRule(
        "subcortex", "amyg", MIXED, ("Sah2003",),
        _AMYGDALA_NOTE,
    ),
)


# --- resolution against the flat atlas -------------------------------------


class SignAssignment(object):
    """What sign one region got, and which rule gave it.

    `rule` is None when no rule matches the region (impossible for a valid
    atlas today, because the table covers every suffix the atlas parser
    accepts, but a first-class outcome rather than an error): then
    `transmitter` and `sign` are None.
    """

    __slots__ = ("region", "rule")

    def __init__(self, region, rule):
        self.region = region
        self.rule = rule

    @property
    def label(self):
        return self.region.label

    @property
    def kind(self):
        return self.region.kind

    @property
    def suffix(self):
        return self.region.suffix

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

    def __repr__(self):
        return "SignAssignment(%r, %r, sign=%r)" % (
            self.label,
            self.transmitter,
            self.sign,
        )


class SignRules(object):
    """`SIGN_RULES` resolved into a per-region lookup.

    The cortex-wide rule is the default for every cortical region; the
    subcortical rules are keyed by nucleus suffix. A region whose `(kind,
    suffix)` matches nothing is unsigned, never guessed.
    """

    def __init__(self, rules=SIGN_RULES):
        self.rules = tuple(rules)
        cortex_rules = [r for r in self.rules if r.kind == "cortex"]
        if len(cortex_rules) != 1:
            raise ValueError(
                "expected exactly one cortex-wide sign rule, got %d"
                % len(cortex_rules)
            )
        self._cortex_rule = cortex_rules[0]
        self._subcortical = {}
        for rule in self.rules:
            if rule.kind != "subcortex":
                continue
            if rule.suffix in self._subcortical:
                raise ValueError(
                    "two sign rules for subcortical nucleus %r" % (rule.suffix,)
                )
            self._subcortical[rule.suffix] = rule

    def assign(self, region):
        """Sign assignment for one `atlas.Region`."""
        if region.kind == "cortex":
            return SignAssignment(region, self._cortex_rule)
        return SignAssignment(region, self._subcortical.get(region.suffix))

    def assign_all(self, atlas):
        """Sign assignment for every region of a `HumanAtlas`, in order."""
        return [self.assign(region) for region in atlas.regions]

    def citations_used(self, atlas):
        """Citation keys actually reached by this atlas, sorted."""
        used = set()
        for assignment in self.assign_all(atlas):
            used.update(assignment.citations)
        return sorted(used)
