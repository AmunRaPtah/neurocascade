"""Illustrative parameter presets for worked examples.

IMPORTANT: the kon/koff/ke/tau/gain/weight values below are hand-set to
produce qualitatively and mechanistically correct dynamics (a mu-opioid
agonist disinhibits VTA dopamine neurons; an antagonist blocks and can
displace an already-bound agonist). They are NOT fit to or sourced from any
specific receptor-binding or pharmacokinetic dataset: treat them as
placeholders awaiting calibration against real Ki/kon/koff and PK
literature, the same way the CCT model's priors were literature-elicited
before use. Do not cite these numbers as measured values.
"""

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite


def mu_opioid_ligands():
    fentanyl = Ligand(name="fentanyl", kon=1.0, koff=0.15, efficacy=1.0)
    naloxone = Ligand(name="naloxone", kon=1.2, koff=0.05, efficacy=0.0)
    return [fentanyl, naloxone]


def mu_opioid_receptor_site():
    return ReceptorSite(name="mu_opioid_on_VTA_GABA", ligands=mu_opioid_ligands())


def mu_opioid_pk_models():
    # Relative ordering (naloxone clears faster than fentanyl) is meant to
    # be directionally right; absolute values are illustrative only.
    return [OneCompartmentPK(ke=0.05), OneCompartmentPK(ke=0.20)]


def reward_circuit_network():
    """VTA -> NAc -> PFC reward circuit, textbook opioid-disinhibition wiring:
    mu-agonist suppresses VTA GABAergic interneurons, which tonically inhibit
    VTA dopamine neurons; suppressing the brake increases DA firing, which
    drives NAc, which drives PFC; PFC exerts top-down feedback control.
    """
    nodes = [
        CircuitNode(name="VTA_GABA", tau=1.0, baseline_input=0.5, gain=4.0),
        CircuitNode(name="VTA_DA", tau=1.5, baseline_input=0.2, gain=4.0),
        CircuitNode(name="NAc", tau=2.0, baseline_input=0.0, gain=4.0),
        CircuitNode(name="PFC", tau=3.0, baseline_input=0.3, gain=4.0),
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    W[idx["VTA_DA"]][idx["VTA_GABA"]] = -2.5  # local GABA inhibits DA neurons
    W[idx["NAc"]][idx["VTA_DA"]] = 2.0  # mesolimbic dopamine projection
    W[idx["PFC"]][idx["NAc"]] = 1.5  # NAc drives PFC
    W[idx["VTA_DA"]][idx["PFC"]] = -0.8  # top-down feedback control
    W[idx["VTA_GABA"]][idx["PFC"]] = 0.6  # top-down recruits local inhibition
    return CircuitNetwork(nodes=nodes, connectivity=W)


def mu_opioid_coupling():
    # Mu receptors are Gi-coupled and sit on VTA GABAergic interneurons:
    # agonist activation SUPPRESSES that node's drive (disinhibition).
    return {"VTA_GABA": -3.0}


def incentive_salience_readout():
    return BehaviorReadout(
        name="incentive_salience", weights={"NAc": 0.7, "VTA_DA": 0.3}
    )


# ---------------------------------------------------------------------------
# Multi-region mu-opioid receptor density map (Track C1)
# SAME receptor/ligand pair (fentanyl/naloxone), THREE differently-wired
# target nodes across three otherwise-unconnected downstream systems
# ---------------------------------------------------------------------------
#
# Every scenario above this point couples one receptor's activation into
# exactly one circuit node via a single flat gain
# (`mu_opioid_coupling() == {"VTA_GABA": -3.0}`). Real mu-opioid receptors
# are not confined to VTA GABAergic interneurons: they are expressed, at
# different relative densities, across many brain regions simultaneously,
# and a single systemic dose (e.g. IV fentanyl) produces several distinct
# behavioral effects at once because of that -- reward, analgesia, and
# sedation are not independent drug actions, they are the SAME receptor
# activation reaching different circuits.
#
# Does this need a new engine mechanism, or does the existing `coupling`
# dict already generalize? Looking at `Simulation.rhs`:
#
#     for node_name, gain in self.coupling.items():
#         idx = self.network.index[node_name]
#         drive[idx] += gain * activation
#
# This loop already iterates over every entry in `coupling`, entirely
# independently: nothing about it assumes a single-key dict. A receptor
# whose activation should drive N nodes at N different gains is already
# expressible today as `{"node_a": g_a, "node_b": g_b, "node_c": g_c, ...}`,
# with no change to `receptor.py`, `network.py`, or `engine.py`. That is a
# real, load-bearing difference from the GABA-A/benzodiazepine case (which
# needed `connection_modulation` because allosteric modulation is not
# "add activation to a node's drive" at all, for any single node): here the
# *shape* of the mechanism -- additive drive on a targeted node -- is
# unchanged, only the *number* of targets is greater than one. Building a
# separate "density map" class or a `{node: (density, gain)}` two-tuple
# structure on top of this would be machinery this codebase does not need:
# it would let you factor a node's number into "density * mechanism
# strength", but nothing downstream of `coupling` ever needs that
# factorization done FOR it -- the product is all `Simulation.rhs` uses.
# So no engine extension was made for this track; see docs/architecture.md
# for the fuller writeup of this judgment call, including the one real
# subtlety it surfaced (density category and node-coupling-gain magnitude
# are not the same axis -- see the LC gain below).
#
# The three target nodes chosen here are real, well-described mu-opioid
# targets, each via a different cellular mechanism:
#
# 1. VTA GABAergic interneurons (`VTA_GABA`, reused unchanged from
#    `mu_opioid_coupling`/`reward_circuit_network` above): Gi-coupled
#    mu-opioid receptors on local GABAergic interneurons tonically
#    inhibiting VTA dopamine neurons; agonist activation suppresses the
#    interneuron, disinhibiting dopamine firing -> reward/incentive
#    salience.
# 2. Periaqueductal gray local circuit (`PAG_Interneuron` -| `PAG_Output`,
#    new): the classic descending-analgesia mechanism (Fields & Basbaum's
#    PAG-RVM circuit). Mu-opioid receptors on PAG GABAergic interneurons
#    that tonically inhibit PAG output neurons; agonist activation
#    suppresses the interneuron, disinhibiting PAG output projecting to the
#    rostral ventromedial medulla and, from there, descending inhibition of
#    spinal nociceptive transmission -> analgesia. Mechanistically this is
#    the SAME "Gi-coupled-receptor-on-a-GABA-interneuron disinhibits an
#    output population" motif as VTA_GABA, just in a different circuit, so
#    it is wired identically (same tau/baseline/gain/weight shape as
#    VTA_GABA -| VTA_DA).
# 3. Locus coeruleus (`LC`, new): mu-opioid receptors sit directly on LC
#    noradrenergic neurons themselves (Gi-coupled hyperpolarization), not on
#    an interposed local interneuron, the classic mechanism behind opioid
#    sedation and (via LC's role in the withdrawal noradrenergic surge)
#    opioid dependence. Modeled here as a single isolated node directly
#    coupled to receptor activation, with no local circuit needed -- a
#    second demonstration that `coupling` targets are agnostic to whether
#    the underlying biology is direct or interneuron-mediated.
#
# `PAG_Interneuron`/`PAG_Output` and `LC` are deliberately left UNCONNECTED
# to the reward circuit (and to each other) in `multi_region_mu_opioid_
# network()`: wiring them together would assert an anatomical projection
# this scenario doesn't intend to claim. The point being demonstrated is
# that one receptor's activation can supply drive to several genuinely
# separate downstream systems at once, not that those systems interact with
# each other.
#
# Relative density citations (qualitative/relative ordering, NOT absolute
# Bmax numbers, which none of these sources report in comparable units):
#   [Mansour1987] Mansour A, Khachaturian H, Lewis ME, Akil H, Watson SJ.
#                 "Autoradiographic differentiation of mu, delta, and kappa
#                 opioid receptors in the rat forebrain and midbrain."
#                 J Neurosci. 1987;7(8):2445-2464. Foundational quantitative
#                 mu-receptor autoradiography mapping across these and other
#                 regions.
#   [Hawkins1988] Cited via Mansour et al.'s review (Mansour A, Khachaturian
#                 H, Lewis ME, Akil H, Watson SJ. "Anatomy of CNS opioid
#                 receptors." Trends Neurosci. 1988;11(7):308-14): mu
#                 receptor binding density in periaqueductal gray and raphe
#                 nuclei is described as MODERATE, not among the highest-
#                 density regions.
#   [TempelZukin1987] Tempel A, Zukin RS. "Neuroanatomical patterns of the
#                 mu, delta, and kappa opioid receptors of rat brain as
#                 determined by quantitative in vitro autoradiography."
#                 Proc Natl Acad Sci. 1987;84(12):4308-12 (PubMed 3035579).
#                 Reports mu receptors most densely localized in patches in
#                 the striatum/nucleus accumbens, several cortical/
#                 hippocampal/thalamic regions, and the LOCUS COERULEUS --
#                 i.e. LC and striatal/NAc patches rank among the HIGHEST
#                 mu-receptor-density regions identified, a distinctly
#                 higher category than the "moderate" PAG/raphe density
#                 above.
#   [VanBockstaele] Van Bockstaele EJ et al. (ultrastructural
#                 immunoelectron-microscopy series on rat/cat LC, e.g.
#                 "Ultrastructural evidence for prominent distribution of
#                 the mu-opioid receptor at extrasynaptic sites on
#                 noradrenergic dendrites in the rat nucleus locus
#                 coeruleus") corroborates dense mu-opioid receptor
#                 expression directly on LC noradrenergic neurons
#                 themselves (not via a local interneuron), consistent with
#                 the direct-coupling mechanism used for `LC` below.
#
# A genuinely interesting wrinkle surfaced by taking this literature
# seriously (same spirit as Track A's naloxone-koff finding): LC is the
# HIGHEST-density region of these three per [TempelZukin1987], with VTA and
# PAG both in the "moderate" tier per [Hawkins1988]/[Mansour1987] -- yet the
# gain chosen for `LC` below is the SMALLEST of the three, not the largest.
# That is not a contradiction, it is the reason a raw density number cannot
# be dropped into `coupling` unmodified: `coupling`'s gain is a NODE-LEVEL
# drive-coupling constant in this mean-field abstraction, which folds
# together (a) receptor density and (b) how much a unit of receptor
# activation actually changes that node's net input given the surrounding
# circuit mechanism. VTA_GABA and PAG_Interneuron's activation is amplified
# by an interposed GABAergic-interneuron-disinhibition step reaching a
# separate output node one synapse downstream; LC's activation acts
# directly on the same node being read out, with no such amplification
# step. So even though LC has the highest receptor density of the three,
# its illustrative node-coupling gain is smaller, and that is a real
# property of this level of abstraction, not an error: density sets which
# nodes plausibly deserve a nonzero coupling entry at all (all three do),
# it does not by itself set the gain's magnitude. All three magnitudes
# below remain hand-tuned illustrative numbers chosen to produce clear,
# checkable qualitative dynamics (see docs/architecture.md) -- they are not
# a density-to-gain unit conversion.


def multi_region_mu_opioid_network():
    """Reward circuit (VTA_GABA/VTA_DA/NAc/PFC, unchanged from
    `reward_circuit_network()`) plus two additional, deliberately
    unconnected sub-systems the same mu-opioid receptor also innervates in
    real anatomy: a PAG descending-analgesia local circuit and a standalone
    LC (noradrenergic tone) node. See the section docstring above for the
    full mechanistic and citation rationale.
    """
    nodes = [
        # --- reward circuit: identical to reward_circuit_network() ---
        CircuitNode(name="VTA_GABA", tau=1.0, baseline_input=0.5, gain=4.0),
        CircuitNode(name="VTA_DA", tau=1.5, baseline_input=0.2, gain=4.0),
        CircuitNode(name="NAc", tau=2.0, baseline_input=0.0, gain=4.0),
        CircuitNode(name="PFC", tau=3.0, baseline_input=0.3, gain=4.0),
        # --- PAG descending-analgesia local circuit (new) ---
        CircuitNode(name="PAG_Interneuron", tau=1.0, baseline_input=0.5, gain=4.0),
        CircuitNode(name="PAG_Output", tau=1.5, baseline_input=0.2, gain=4.0),
        # --- LC noradrenergic tone (new); direct receptor action, no local
        # interneuron in this abstraction ---
        CircuitNode(name="LC", tau=1.0, baseline_input=0.0, gain=4.0),
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    # reward circuit wiring, unchanged
    W[idx["VTA_DA"]][idx["VTA_GABA"]] = -2.5
    W[idx["NAc"]][idx["VTA_DA"]] = 2.0
    W[idx["PFC"]][idx["NAc"]] = 1.5
    W[idx["VTA_DA"]][idx["PFC"]] = -0.8
    W[idx["VTA_GABA"]][idx["PFC"]] = 0.6
    # PAG local circuit: same GABAergic-interneuron-disinhibits-output shape
    # as VTA_GABA -| VTA_DA above
    W[idx["PAG_Output"]][idx["PAG_Interneuron"]] = -2.5
    # LC: no connectivity edges at all -- see docstring above for why that's
    # the point, not an oversight.
    return CircuitNetwork(nodes=nodes, connectivity=W)


def mu_opioid_multi_region_coupling():
    # Relative gain magnitudes are illustrative and hand-tuned for
    # qualitative clarity, NOT a density-to-gain unit conversion; see the
    # section docstring above for why LC's real receptor density (highest
    # of the three, per literature) does not translate into the largest
    # gain here.
    return {
        "VTA_GABA": -3.0,  # reward; unchanged from mu_opioid_coupling()
        "PAG_Interneuron": -2.5,  # analgesia; moderate density, same
        # interneuron-disinhibition mechanism as VTA_GABA
        "LC": -1.5,  # sedation/arousal; direct action, no amplification
        # step, despite highest literature-reported density of the three
    }


def analgesia_index_readout():
    return BehaviorReadout(name="analgesia_index", weights={"PAG_Output": 1.0})


def arousal_index_readout():
    # LC firing rate itself: mu-opioid agonism suppresses it (sedation), so
    # this readout should FALL, unlike incentive_salience/analgesia_index
    # which both RISE with agonist.
    return BehaviorReadout(name="arousal_index", weights={"LC": 1.0})


# ---------------------------------------------------------------------------
# D2 dopamine receptor / antipsychotic-relevant circuit
# (basal ganglia indirect pathway)
# ---------------------------------------------------------------------------
#
# Mechanistic rationale: D2 receptors are Gi-coupled and densely expressed on
# striatal "indirect pathway" medium spiny neurons (MSNs). Dopamine acting at
# D2 SUPPRESSES those MSNs; suppressing them disinhibits GPe, which inhibits
# STN, which reduces excitatory drive onto GPi, which disinhibits thalamus:
# net effect, dopamine acting at D2 facilitates thalamocortical (motor)
# output. This is the same "disinhibition via a Gi-coupled receptor" motif as
# the mu-opioid scenario, just relayed through a couple more basal-ganglia
# synapses instead of landing directly on the reward circuit's own node, so
# it maps onto ReceptorSite/CircuitNetwork/Simulation without any class
# changes.
#
# D2 antagonists (haloperidol and other typical antipsychotics) block that
# suppression, and clinically that's the same mechanistic thread connecting
# "blocks psychotic symptoms" (mesolimbic D2 blockade) to "can cause
# bradykinesia/parkinsonism" (nigrostriatal/indirect-pathway D2 blockade) for
# this drug class. This scenario models the indirect-pathway/motor side of
# that story, so the readout below is locomotor activity, not a
# psychosis-symptom index.
#
# As with mu-opioid, all kon/koff/ke/circuit values below are illustrative
# placeholders chosen to get the qualitative mechanism right, not fit to a
# Ki/PK dataset. Do not cite these numbers as measured values.


def d2_dopamine_ligands():
    dopamine = Ligand(name="dopamine", kon=1.0, koff=0.5, efficacy=1.0)
    # Slower koff than dopamine: directionally consistent with the high,
    # sustained D2 receptor occupancy typical antipsychotics show in PET
    # occupancy studies, though the absolute values here are illustrative.
    haloperidol = Ligand(name="haloperidol", kon=1.0, koff=0.02, efficacy=0.0)
    return [dopamine, haloperidol]


def d2_receptor_site():
    return ReceptorSite(
        name="D2_on_striatal_indirect_MSN", ligands=d2_dopamine_ligands()
    )


def d2_pk_models():
    # Synaptic dopamine is cleared/recycled essentially immediately; a
    # typical antipsychotic has a day-scale clinical half-life. Relative
    # ordering is meant to be directionally right, not the absolute values.
    return [OneCompartmentPK(ke=0.3), OneCompartmentPK(ke=0.03)]


def basal_ganglia_indirect_pathway_network():
    """Striatum(D2 MSN) -| GPe -| STN -> GPi -| Thalamus: the classic
    indirect ("no-go") pathway. Kept as one node per nucleus, textbook-wiring
    legible, rather than folding synapses together.
    """
    # baseline_input values are chosen so each node sits near the sensitive
    # (non-saturating) part of its sigmoid at rest, given its steady-state
    # input from upstream nodes. Otherwise a downstream node like GPi pins
    # near 0 or 1 and stops responding to anything upstream of it.
    nodes = [
        CircuitNode(name="Striatum_D2_MSN", tau=1.0, baseline_input=0.0, gain=4.0),
        CircuitNode(name="GPe", tau=1.0, baseline_input=1.0, gain=4.0),
        CircuitNode(name="STN", tau=1.0, baseline_input=1.0, gain=4.0),
        CircuitNode(name="GPi", tau=1.5, baseline_input=-1.0, gain=4.0),
        CircuitNode(name="Thalamus", tau=2.0, baseline_input=1.0, gain=4.0),
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    W[idx["GPe"]][idx["Striatum_D2_MSN"]] = -2.0  # indirect MSNs inhibit GPe
    W[idx["STN"]][idx["GPe"]] = -2.0  # GPe inhibits STN
    W[idx["GPi"]][idx["STN"]] = 2.0  # STN excites GPi
    W[idx["Thalamus"]][idx["GPi"]] = -2.0  # GPi inhibits thalamus
    return CircuitNetwork(nodes=nodes, connectivity=W)


def d2_coupling():
    # Gi-coupled D2 receptor on indirect-pathway MSNs: agonist activation
    # SUPPRESSES that node's drive (same disinhibition motif as mu-opioid).
    return {"Striatum_D2_MSN": -3.0}


def locomotor_activity_readout():
    return BehaviorReadout(name="locomotor_activity", weights={"Thalamus": 1.0})


# ---------------------------------------------------------------------------
# GABA-A / benzodiazepine circuit (amygdala, anxiety-relevant)
# ALLOSTERIC MODULATION, not competitive occupancy at GABA's own site
# ---------------------------------------------------------------------------
#
# Benzodiazepines (e.g. diazepam) do NOT compete with GABA for the same
# orthosteric binding site the way naloxone competes with fentanyl at mu
# receptors. They bind a distinct allosteric site on the GABA-A receptor (at
# an alpha/gamma subunit interface) and act as a POSITIVE ALLOSTERIC
# MODULATOR: they don't gate the channel themselves, they increase how
# effectively GABA (bound at its own separate orthosteric site) opens the
# channel. Flumazenil, in contrast, DOES compete with diazepam at that same
# benzodiazepine site, which is the real mechanism of benzodiazepine
# overdose reversal, but it has little effect on its own, because it isn't
# displacing GABA anywhere.
#
# ReceptorSite/Ligand model the competitive diazepam-vs-flumazenil
# interaction at the benzodiazepine site correctly, unmodified: those two do
# share one pool, exactly like fentanyl/naloxone. What does NOT fit the
# existing engine is how that occupancy couples into circuit dynamics:
# Simulation's `coupling` argument adds receptor activation directly and
# additively to a node's drive, which is right for a receptor that itself
# gates a current (mu-opioid, D2), but wrong for an allosteric modulator that
# has no drive of its own: it only changes the *gain* of an already-present
# GABAergic synapse. Modeling it with `coupling` would imply diazepam has an
# intrinsic excitatory/inhibitory action independent of any ongoing GABAergic
# transmission, which is not how benzodiazepines work and would misrepresent
# them as behaving like a conventional GPCR partial agonist.
#
# Rather than force that wrong shape, this scenario uses the small,
# additive-only `Simulation(..., connection_modulation=...)` extension added
# in engine.py alongside it: it multiplies one existing (already-inhibitory)
# connectivity weight by `1 + gain * activation`, representing potentiation
# of that GABAergic synapse's effect rather than an independent drive on
# either node. `efficacy` for diazepam is repurposed here as "potentiation
# strength" (1.0 = the illustrative gain used below) rather than direct
# channel activation; flumazenil keeps efficacy=0, a neutral competitor at
# the benzodiazepine site, the same simplification the mu-opioid scenario
# already makes for naloxone. `receptor.py` itself needed no changes.
#
# Circuit: local GABAergic interneurons tonically inhibit basolateral
# amygdala (BLA) principal neurons; BLA principal neurons drive central
# amygdala (CeA) output, the standard fear/anxiety output relay. Potentiating
# the interneuron -> BLA synapse (the GABA-A synapse benzodiazepines act on)
# suppresses BLA output and, downstream, the anxiety_index readout,
# consistent with benzodiazepines' clinical anxiolytic effect. As above, all
# kon/koff/ke/circuit values are illustrative placeholders, not measured
# values.


def gaba_a_benzodiazepine_ligands():
    diazepam = Ligand(name="diazepam", kon=1.0, koff=0.1, efficacy=1.0)
    # Faster on/off than diazepam: mirrors flumazenil's real short clinical
    # half-life, which is exactly why repeat dosing is needed to sustain
    # benzodiazepine-overdose reversal, the same clearance-mismatch problem
    # naloxone has against longer-acting opioids.
    flumazenil = Ligand(name="flumazenil", kon=1.5, koff=0.3, efficacy=0.0)
    return [diazepam, flumazenil]


def gaba_a_receptor_site():
    return ReceptorSite(
        name="GABA_A_benzodiazepine_site_on_BLA_interneuron_synapse",
        ligands=gaba_a_benzodiazepine_ligands(),
    )


def gaba_a_pk_models():
    # Diazepam clears very slowly (day-scale clinical half-life, including
    # active metabolites); flumazenil clears fast (hour-scale), the same
    # directional point as the koff choice above.
    return [OneCompartmentPK(ke=0.02), OneCompartmentPK(ke=0.5)]


def amygdala_anxiety_circuit_network():
    """GABA_Interneuron -| BLA -> CeA: local inhibition onto amygdala
    principal neurons, which drive the central-amygdala fear/anxiety output
    relay.
    """
    # baseline_input values (and BLA's lower gain) are chosen so BLA and CeA
    # sit near the sensitive part of their sigmoids at rest, the same reason
    # given for the D2 basal-ganglia network above; without that, CeA pins
    # near 1.0 and stops responding to anything upstream of it.
    nodes = [
        CircuitNode(name="GABA_Interneuron", tau=1.0, baseline_input=0.5, gain=4.0),
        CircuitNode(name="BLA", tau=1.5, baseline_input=2.2, gain=1.5),
        CircuitNode(name="CeA", tau=2.0, baseline_input=-1.0, gain=4.0),
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    W[idx["BLA"]][idx["GABA_Interneuron"]] = -2.5  # the GABA-A synapse benzos potentiate
    W[idx["CeA"]][idx["BLA"]] = 2.0  # BLA principal neurons drive CeA output
    return CircuitNetwork(nodes=nodes, connectivity=W)


def gaba_a_coupling():
    # No direct drive coupling: diazepam/flumazenil act purely through
    # connection_modulation below, not by driving any node directly.
    return {}


def gaba_a_connection_modulation():
    return {("BLA", "GABA_Interneuron"): 0.5}


def anxiety_index_readout():
    return BehaviorReadout(name="anxiety_index", weights={"CeA": 1.0})


# ---------------------------------------------------------------------------
# Calibrated mu-opioid receptor + fentanyl/naloxone PK, grounded in published
# data (Track A). Units: concentration in nM, time in minutes, kon in
# nM^-1 min^-1, koff/ke in min^-1. Everything below this line is a real-data
# parameter set, kept deliberately separate from the illustrative functions
# above so the two are never confused. The reward circuit (`reward_circuit_
# network`, `mu_opioid_coupling`, `incentive_salience_readout`) is reused
# as-is: this pass only calibrates receptor binding and PK, not circuit
# connectivity/gain/tau, which have no literature target and remain
# illustrative (see docs/architecture.md).
#
# Sources (see inline citations per parameter below):
#   [Volpe2011]  Volpe DA, McMahon Tobin GA, Mellon RD, Katki AG, Parker RJ,
#                Colatsky T, Kropp TJ, Verbois SL. "Uniform assessment and
#                ranking of opioid mu receptor binding constants for selected
#                opioid drugs." Regul Toxicol Pharmacol. 2011;59(3):385-90.
#                doi:10.1016/j.yrtph.2010.12.007. Single standardized
#                [3H]-DAMGO competition-binding assay on a recombinant human
#                MOR membrane preparation; fentanyl and naloxone were run in
#                the SAME assay, so their Ki's are directly comparable
#                (fentanyl Ki = 1.346 nM; naloxone, used as the assay's
#                reference ligand, Ki = 1.518 +/- 0.065 nM, IC50 = 5.926 +/-
#                0.253 nM, both figures read directly from the paper text).
#   [Mann2022]   Mann J, Samieegohar M, Chaturbedi A, Zirkle J, Han X,
#                Ahmadi SF, Eshleman A, Janowsky A, Wolfrum K, Swanson T,
#                Bloom S, Dahan A, Olofsen E, Florian J, Strauss DG, Li Z.
#                "Development of a Translational Model to Assess the Impact
#                of Opioid Overdose and Naloxone Dosing on Respiratory
#                Depression and Cardiac Arrest." Clin Pharmacol Ther.
#                2022;112(5):1020-1032. doi:10.1002/cpt.2696. FDA/CDER
#                mechanistic PK-PD model of naloxone reversal; mu-receptor
#                association/dissociation rate constants fit per-ligand from
#                in vitro binding-kinetics assays. Code + fit parameters are
#                public: github.com/FDA/Mechanistic-PK-PD-Model-to-Rescue-
#                Opioid-Overdose, Ligand_Data/{Fentanyl_A,NALOXONE}_
#                parameters/*_pars.txt: fentanyl koff = 0.004331 s^-1
#                (t1/2(diss) ~ 160 s, i.e. ~2.7 min, independently
#                corroborated as "~4 min" by a computational residence-time
#                study, Mahinthichaichan et al. 2021 JACS Au
#                doi:10.1021/jacsau.1c00341); naloxone koff = 0.03959 s^-1
#                (t1/2(diss) ~ 17.5 s). NOTE: Mann et al.'s Kon values use a
#                nonstandard Hill-exponent parametrization (units
#                pM^-n * s^-1, n ~ 0.84-0.86) that isn't dimensionally
#                compatible with this codebase's strictly bimolecular (n=1)
#                occupancy ODE, so we do not reuse their Kon directly; see
#                the kon derivation below instead.
#   [FDA-label]  Fentanyl Citrate Injection, USP prescribing information
#                (accessdata.fda.gov/drugsatfda_docs/label/.../016619...
#                lbl.pdf; figures corroborated across multiple label
#                revisions 2016-2025). Three-compartment IV PK: distribution
#                t1/2 = 1.7 min, redistribution t1/2 = 13 min, terminal
#                elimination t1/2 = 219 min, Vd = 4 L/kg.
#   [Naloxone-PK] Handal KA, Schauben JL, Salamone FR. "Naloxone." Ann Emerg
#                Med. 1983;9(4):438-45 (dosing-nomogram pharmacokinetic
#                analysis; biexponential decline). Corroborated by a more
#                recent review, Rzasa Lynn R, Galinkin JL. "Clinical
#                Pharmacokinetics and Pharmacodynamics of Naloxone." Clin
#                Pharmacokinet. 2024. IV serum half-life ~30-81 min, mean
#                ~64 min.
#   [Moss2020]   Moss RB, Pryor MM, Baillie R, Kudrycki K, Friedrich C,
#                Reed M, Carlo DJ. "Higher naloxone dosing in a quantitative
#                systems pharmacology model that predicts naloxone-fentanyl
#                competition at the opioid mu receptor level." PLoS ONE.
#                2020;15(6):e0234683. doi:10.1371/journal.pone.0234683
#                (open access, PMC7297366). Independent QSP model using
#                fentanyl Ki = 1.35 nM and naloxone Ki = 1.115 nM (close to
#                the [Volpe2011] values above, corroborating that ~1-1.5 nM
#                range) and reporting baseline (no naloxone) mu-receptor
#                occupancy of 73%/97%/98.9% at fentanyl exposure levels of
#                25/50/75 ng/mL, used below as a validation target, with
#                caveats (see test_scenarios_calibrated.py): that paper's
#                occupancy is computed from an internal brain/effect-site
#                concentration produced by its own multi-compartment brain
#                PK submodel, not the plasma-equivalent single concentration
#                variable this codebase tracks, so we validate the
#                direction/order of magnitude (near-saturating occupancy at
#                overdose-range exposure), not an exact percentage match.
#
# Derivation of kon: none of the sources above report a kon for fentanyl or
# naloxone in units compatible with this codebase's ODE
# (d(occ)/dt = kon*C*(1-sum(occ)) - koff*occ, which requires kon in
# nM^-1 min^-1). We derive it from the two numbers that ARE independently
# measured and dimensionally simple: Ki (a true equilibrium constant) and
# koff (a clean first-order rate constant), via the standard reversible
# bimolecular relationship Kd = koff / kon, i.e. kon = koff / Kd, taking
# Kd ~= Ki (the usual approximation for a reversible orthosteric ligand,
# same one implicit in the Cheng-Prusoff equation Volpe et al. used to
# compute Ki from IC50).
def mu_opioid_ligands_calibrated():
    fentanyl = Ligand(
        name="fentanyl",
        kon=0.1931,   # nM^-1 min^-1; koff/Ki = (0.004331 s^-1 * 60) / 1.346 nM
        koff=0.2599,  # min^-1; 0.004331 s^-1 [Mann2022]
        efficacy=1.0,
    )
    naloxone = Ligand(
        name="naloxone",
        kon=1.5648,   # nM^-1 min^-1; koff/Ki = (0.03959 s^-1 * 60) / 1.518 nM
        koff=2.3754,  # min^-1; 0.03959 s^-1 [Mann2022]
        efficacy=0.0,
    )
    return [fentanyl, naloxone]


def mu_opioid_receptor_site_calibrated():
    return ReceptorSite(name="mu_opioid_on_VTA_GABA", ligands=mu_opioid_ligands_calibrated())


def mu_opioid_pk_models_calibrated():
    # One-compartment collapse of each drug's real multi-compartment IV PK,
    # chosen to match this scenario's acute-bolus/reversal timescale
    # (minutes to ~1 h) rather than steady-state/terminal disposition. This
    # is an explicit, cited simplification, not a claim that either drug
    # truly follows one-compartment kinetics.
    #   fentanyl ke: ln(2) / 13 min, the REDISTRIBUTION-phase half-life
    #     [FDA-label], the dominant decline in the first ~1 h after a
    #     single IV bolus (the reason fentanyl's clinical duration of action
    #     is short despite a long, 219 min, terminal half-life driven by
    #     slow release from fat/muscle depots). Using the terminal phase
    #     instead would over-predict how long a single bolus stays on
    #     board within this scenario's time window.
    #   naloxone ke: ln(2) / 64 min, the mean reported IV serum half-life
    #     [Naloxone-PK].
    # Net effect: in this one-compartment approximation, fentanyl's
    # concentration falls FASTER than naloxone's (ke 0.0533 vs 0.0108
    # min^-1), the reverse of the illustrative placeholder's assumed
    # ordering. This is a real and clinically important asymmetry: it is
    # naloxone's short RECEPTOR residence time (see koff above, ~17.5 s vs
    # fentanyl's ~160 s) and short duration of antagonism, not a faster
    # plasma clearance, that drives naloxone reversal to wear off before a
    # large/long-acting fentanyl exposure does (renarcotization risk), a
    # phenomenon that requires fentanyl's slow terminal elimination phase
    # to manifest, which this one-compartment collapse cannot represent.
    return [OneCompartmentPK(ke=0.05332), OneCompartmentPK(ke=0.01083)]
