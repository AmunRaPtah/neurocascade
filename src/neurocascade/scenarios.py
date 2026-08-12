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
from .engine import ReceptorSystem
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
    # ILLUSTRATIVE. This gain still has no literature target. Track E
    # (docs/architecture.md) searched for a real fentanyl-induced NAc/VTA
    # dopamine release effect size and found the right paper (Yoshida et al.
    # 1999) confirms the mechanism but publishes no accessible figure; P2.2
    # re-ran that search over all 421 open-access Europe PMC articles
    # matching fentanyl/accumbens/dopamine/rat and found no percent-of-
    # baseline dopamine figure at an INTRAVENOUS dose, which is the only
    # fentanyl route drug_pk has kinetics for. See
    # calibration.behavioral_evidence.SEARCHED_AND_EMPTY_BEHAVIORAL and
    # docs/architecture.md's P2.2 section, which also names the one
    # concentration-anchored fentanyl PD dataset that does exist
    # ([Yassen2008]) and why it belongs to the PAG/analgesia gain rather
    # than to this one.
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
    # gain here. As with mu_opioid_coupling() above, Track E
    # (docs/architecture.md) looked for a real behavioral anchor for these
    # gains and did not find one usable; they remain illustrative.
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
    # ILLUSTRATIVE. This gain still has no literature target. Track E
    # (docs/architecture.md) found real haloperidol-specific locomotor-
    # suppression data (Gao et al. 2021) that is dose-based (mg/kg) by a
    # route this codebase has no haloperidol kinetics for. P2.2 re-ran that
    # search over all 1303 open-access Europe PMC articles matching
    # haloperidol/rat/locomotor-or-catalepsy-or-plasma-concentration and
    # confirmed the non-intersection: the behavioural literature doses IP,
    # SC or PO essentially without exception, and drug_pk has haloperidol
    # kinetics for IV only. P2.2 also searched for the route-free
    # alternative (a rat study pairing measured D2 occupancy with a
    # locomotor or cataleptic effect, which would need no PK step at all);
    # the two papers that do that are closed access. See
    # calibration.behavioral_evidence.SEARCHED_AND_EMPTY_BEHAVIORAL and
    # docs/architecture.md's P2.2 section.
    return {"Striatum_D2_MSN": -3.0}


def locomotor_activity_readout():
    return BehaviorReadout(name="locomotor_activity", weights={"Thalamus": 1.0})


# ---------------------------------------------------------------------------
# Combined scenario: mu-opioid + D2 receptor systems on ONE shared network
# ---------------------------------------------------------------------------
#
# Demonstrates the multi-receptor-system extension to `Simulation`
# (`engine.ReceptorSystem` + `Simulation(network, receptor_systems=[...])`):
# a mu-opioid receptor system on VTA_GABA and a D2 receptor system on
# Striatum_D2_MSN, each with its own PK models, its own ligands, and its own
# `coupling`, driving one shared 9-node circuit simultaneously, instead of
# two separate `Simulation` objects on two separate networks.
#
# Design choice for the combined network: this is a straight union of the
# reward circuit's own internal connectivity (VTA_GABA -> VTA_DA -> NAc ->
# PFC) and the basal ganglia indirect pathway's own internal connectivity
# (Striatum_D2_MSN -| GPe -| STN -> GPi -| Thalamus), unchanged from
# `reward_circuit_network()`/`basal_ganglia_indirect_pathway_network()`,
# with NO added connection between the two blocks. A real brain doesn't
# literally partition this way (corticostriatal and mesostriatal
# projections really do link these two systems), but no literature-sourced
# *weight* for such a cross-connection was identified (the same kind of gap
# already documented in docs/architecture.md for the within-circuit
# weights), and inventing one just to manufacture a cross-circuit
# interaction would be exactly the mechanistically dishonest shortcut this
# codebase has deliberately avoided elsewhere (see the GABA-A scenario's
# coupling-vs-connection_modulation discussion above). The real, previously
# unbuilt capability this demonstrates is that both receptor systems now
# read and write ONE shared state vector and are integrated by ONE
# `solve_ivp` call, not that these two illustrative circuits causally talk
# to each other. `tests/test_multi_receptor_simulation.py` checks both that
# this runs correctly and that each block's dynamics, with the other
# system's ligands zeroed out, exactly reproduce that system's own
# standalone single-receptor-system `Simulation`; a separate engine-level
# test (not scenario-specific) demonstrates what genuinely happens when two
# systems DO target the same connection (the multiplicative composition
# rule documented in `engine.py`).


def combined_mu_opioid_d2_network():
    """The reward circuit and the basal ganglia indirect pathway as one
    shared 9-node `CircuitNetwork` (block-diagonal union of their own
    connectivity; see module-level comment above for why no cross-block
    connection is added).
    """
    reward = reward_circuit_network()
    bg = basal_ganglia_indirect_pathway_network()
    nodes = list(reward.nodes) + list(bg.nodes)
    combined_index = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    for sub_network in (reward, bg):
        for post_name, i in sub_network.index.items():
            for pre_name, j in sub_network.index.items():
                weight = sub_network.connectivity[i, j]
                if weight != 0.0:
                    W[combined_index[post_name]][combined_index[pre_name]] = weight
    return CircuitNetwork(nodes=nodes, connectivity=W)


def mu_opioid_receptor_system():
    """The mu-opioid stack (illustrative parameters) bundled as a
    `ReceptorSystem`, ready to combine with other systems on a shared
    network via `Simulation(network, receptor_systems=[...])`."""
    return ReceptorSystem(
        pk_models=mu_opioid_pk_models(),
        receptor_site=mu_opioid_receptor_site(),
        coupling=mu_opioid_coupling(),
    )


def d2_receptor_system():
    """The D2 stack (illustrative parameters) bundled as a
    `ReceptorSystem`, ready to combine with other systems on a shared
    network via `Simulation(network, receptor_systems=[...])`."""
    return ReceptorSystem(
        pk_models=d2_pk_models(),
        receptor_site=d2_receptor_site(),
        coupling=d2_coupling(),
    )


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
    # ILLUSTRATIVE, and deliberately kept that way. This is the
    # illustrative section's own function, the counterpart of every other
    # hand-set number above it, and P2.2 left its value alone so that the
    # illustrative and calibrated sections stay distinguishable at a
    # glance. The gain that P2.2 actually fitted, against real published
    # elevated-plus-maze data, is in
    # gaba_a_connection_modulation_calibrated() at the bottom of this file.
    # This function's KEYS - which connection diazepam modulates - are what
    # the calibration reuses; only the value differs.
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


# ---------------------------------------------------------------------------
# Calibrated GABA-A / benzodiazepine-site receptor + diazepam/flumazenil PK,
# grounded in published data. Same units as the mu-opioid calibrated section
# above: concentration in nM, time in minutes, kon in nM^-1 min^-1, koff/ke
# in min^-1. Everything below this line is a real-data parameter set for
# `gaba_a_*`, kept separate from the illustrative functions above. The
# amygdala circuit (`amygdala_anxiety_circuit_network`, `gaba_a_coupling`,
# `anxiety_index_readout`) and the connection_modulation GAIN itself are
# reused as-is: this pass only calibrates the benzodiazepine-site receptor
# binding and PK, not circuit connectivity/gain/tau or the potentiation-gain
# magnitude, none of which have a literature target (see
# gaba_a_connection_modulation_calibrated() below and docs/architecture.md).
#
# Sources (see inline citations per parameter below):
#   [Mohler1977]   Möhler H, Okada T. "Properties of 3H-diazepam binding to
#                  benzodiazepine receptors in rat cerebral cortex." Life
#                  Sci. 1977;20(12):2101-2110. doi:10.1016/0024-3205(77)
#                  90191-6. Direct [3H]diazepam saturation binding, rat
#                  cerebral cortex synaptosomal membranes: Kd = 3.6 +/- 0.1
#                  nM. (The companion Möhler & Okada Science 1977 paper,
#                  doi:10.1126/science.918669, first reported the CNS-
#                  specific benzodiazepine receptor qualitatively; this Life
#                  Sci companion paper is where the quantitative Kd is.)
#   [Mohler1981]   Möhler H, Burkard WP, Keller HH, Richards JG, Haefely W.
#                  "Benzodiazepine antagonist Ro 15-1788: binding
#                  characteristics and interaction with drug-induced changes
#                  in dopamine turnover and cerebellar cGMP levels." J
#                  Neurochem. 1981;37(3):714-722. PMID 6268754. Direct
#                  [3H]Ro 15-1788 (flumazenil) saturation binding, same rat
#                  cerebral cortex preparation: Kd = 1.0 +/- 0.1 nM;
#                  unlabeled Ro 15-1788 displacing [3H]diazepam in the same
#                  tissue: IC50 = 2.3 +/- 0.6 nM. Both figures point the same
#                  direction: flumazenil binds this site more tightly than
#                  diazepam does.
#   [Sieghart1995] Sieghart W. "Structure and pharmacology of gamma-
#                  aminobutyric acidA receptor subtypes." Pharmacol Rev.
#                  1995;47(2):181-234. Subunit-resolved [3H]flumazenil Kd
#                  ~1.0-1.5 nM at the major synaptic (alpha1/2/3-beta2-
#                  gamma2) subtypes, corroborating [Mohler1981]'s whole-
#                  cortex value.
#   [Mandelli1978] Mandelli M, Tognoni G, Garattini S. "Clinical
#                  pharmacokinetics of diazepam." Clin Pharmacokinet.
#                  1978;3(1):72-91. doi:10.2165/00003088-197803010-00005.
#                  Two-compartment IV kinetics: a distribution phase with a
#                  half-life of about 1 h precedes the much longer
#                  elimination phase.
#   [Valium-label] VALIUM (diazepam) FDA prescribing information
#                  (accessdata.fda.gov/drugsatfda_docs/label/2016/
#                  013263s094lbl.pdf; figures corroborated across label
#                  revisions and by StatPearls, NCBI Bookshelf NBK537022).
#                  Terminal elimination half-life up to 48 h (StatPearls
#                  reports a mean of ~46 h); active metabolite
#                  N-desmethyldiazepam (desmethyldiazepam/nordazepam)
#                  terminal half-life up to 100 h; diazepam and its
#                  metabolites ~98% plasma-protein-bound.
#   [Romazicon-label] ROMAZICON (flumazenil) FDA prescribing information,
#                  NDA 20-073 (accessdata.fda.gov/drugsatfda_docs/label/
#                  2007/020073s016lbl.pdf and prior revisions). Mean
#                  elimination half-life 54 min (CV 21%, range 41-79 min)
#                  measured after a 5-min IV infusion of 1 mg in normal
#                  volunteers.
#   [Savic1991]    Savic I, Widén L, Stone-Elander S. "Feasibility of
#                  reversing benzodiazepine tolerance with flumazenil."
#                  Lancet. 1991;337(8734):133-137. doi:10.1016/0140-6736
#                  (91)90799-u. Human in vivo receptor-occupancy data for
#                  flumazenil itself: a 1.5 mg IV dose occupied ~55% of
#                  benzodiazepine binding sites in 12 patients; 15 mg
#                  occupied nearly all of them. Used below as a validation
#                  target for the calibrated flumazenil Ki (a real,
#                  independent, human in vivo occupancy measurement, not
#                  fit to any parameter in this codebase).
#
# Derivation of kon: unlike the mu-opioid section above (where koff was
# independently measured for both ligands and kon had to be derived), here
# it is the other way around and, if anything, a harder gap: extensive
# searching (classic radioligand-kinetics papers, e.g. Brown CL, Martin IL,
# "Kinetics of [3H]Ro 15-1788 binding to membrane-bound rat brain
# benzodiazepine receptors," J Neurochem. 1984;42:918-923, which reports
# only flumazenil's kinetics, qualitatively (monoexponential dissociation),
# not diazepam's, and is paywalled so its exact numeric rate constants could
# not be retrieved; Speth RC, Wastek GJ, Yamamura HI, Life Sci.
# 1978/79;24:351-358, which studied flunitrazepam, not diazepam or
# flumazenil) turned up no published kon or koff for diazepam or flumazenil
# at this site, for either ligand, in units compatible with this codebase's
# ODE (nM^-1 min^-1 kon, min^-1 koff), from a standardized modern assay.
# The one quantity independently measured for both ligands, by the same
# research group's preparations, is Ki (Kd) ([Mohler1977], [Mohler1981]).
# Rather than invent an unsourced koff (or kon) for each ligand
# independently, we make one explicit, clearly-flagged modeling assumption:
# both ligands share the same association rate constant, kon_shared. This is
# physically motivated (both are small, comparably lipophilic molecules
# binding the same physical pocket, so their access/collision rate to that
# site is a plausible quantity to treat as similar in the absence of
# ligand-specific data), but it is not a measured value: it is chosen only
# to match the order of magnitude of this codebase's existing mu-opioid
# calibrated kon values, for unit consistency, not sourced from any
# citation. Each ligand's koff is then derived from that shared kon and its
# own real, cited Ki via Kd = koff / kon (the same Kd ~= Ki approximation
# used in the mu-opioid section above): koff_i = kon_shared * Ki_i.
def gaba_a_benzodiazepine_ligands_calibrated():
    kon_shared = 0.2  # nM^-1 min^-1; assumed, not measured, see derivation
    # note above. Chosen only to set an overall timescale comparable to
    # this codebase's mu-opioid calibrated kon values; the real, cited
    # numbers below are each ligand's Ki, not this kon.
    diazepam = Ligand(
        name="diazepam",
        kon=kon_shared,
        koff=0.72,  # min^-1; kon_shared * Ki_diazepam = 0.2 * 3.6 [Mohler1977]
        efficacy=1.0,  # potentiation strength, not channel activation; see
        # gaba_a_benzodiazepine_ligands() docstring above for why efficacy
        # is repurposed this way for this connection_modulation-coupled site.
    )
    flumazenil = Ligand(
        name="flumazenil",
        kon=kon_shared,
        koff=0.20,  # min^-1; kon_shared * Ki_flumazenil = 0.2 * 1.0 [Mohler1981]
        efficacy=0.0,  # neutral competitive antagonist at this site
    )
    return [diazepam, flumazenil]


def gaba_a_receptor_site_calibrated():
    return ReceptorSite(
        name="GABA_A_benzodiazepine_site_on_BLA_interneuron_synapse",
        ligands=gaba_a_benzodiazepine_ligands_calibrated(),
    )


def gaba_a_pk_models_calibrated():
    # One-compartment collapse of each drug's real multi-phase clinical PK,
    # chosen (like the mu-opioid section above) to match this scenario's
    # acute-bolus timescale (minutes to ~1 h; the existing gaba_a tests run
    # to t=40 min) rather than diazepam's multi-day terminal disposition.
    #   diazepam ke: ln(2) / 60 min, the DISTRIBUTION-phase half-life
    #     [Mandelli1978], the dominant decline in the first ~1-2 h after a
    #     single IV/oral dose (the classic pharmacological explanation for
    #     why diazepam's clinical sedative effect is much shorter than its
    #     ~46 h parent / ~100 h active-metabolite terminal half-life
    #     [Valium-label]: the drug redistributes out of the CNS into
    #     peripheral fat long before it is actually eliminated).
    #   flumazenil ke: ln(2) / 54 min, the mean reported IV elimination
    #     half-life [Romazicon-label].
    # A genuinely interesting, real finding surfaced by this pairing: at
    # this scenario's acute timescale, diazepam's own relevant clearance
    # (via its ~60 min distribution phase) and flumazenil's clearance
    # (~54 min) come out nearly the same order of magnitude, not the
    # dramatic "long-acting drug vs. short-acting antagonist" asymmetry the
    # illustrative placeholder assumed (ke=0.02 vs 0.5 min^-1, a 25x
    # difference). The real, dramatic long-vs-short asymmetry between these
    # two drugs shows up only on the multi-hour-to-multi-day scale, driven
    # by diazepam's active metabolite desmethyldiazepam (terminal half-life
    # up to 100 h [Valium-label]), which continues to occupy and potentiate
    # the benzodiazepine site long after parent diazepam has redistributed
    # away. That second active compound is something this one-compartment,
    # single-species PK model cannot represent, exactly analogous to the
    # mu-opioid section's renarcotization gap (fentanyl's slow terminal
    # elimination phase vs. this codebase's one-compartment collapse).
    return [OneCompartmentPK(ke=0.011552), OneCompartmentPK(ke=0.012836)]


#: Posterior mean of `gaba_a_connection_modulation_gain` (P2.2), the first
#: circuit-layer parameter in this project fitted against real data rather
#: than hand-set. Posterior: mean 0.0742, sd 0.0189, 89% ETI [0.043, 0.106],
#: r_hat 1.02, ESS(bulk) 309; full summary in
#: `results/calibration/gaba_a_summary.csv`, `InferenceData` alongside it,
#: and `tests/test_gaba_a_scenario_calibrated.py::test_calibrated_
#: modulation_gain_matches_the_committed_posterior` checks this constant
#: against that artefact so the two cannot drift apart.
#:
#: Fitted against [Graziani2022] (Graziani D, et al., Braz J Med Biol Res
#: 2022;55:e12314, doi:10.1590/1414-431X2022e12314, PMC9728632): diazepam
#: 2 mg/kg IP in drug-naive male Wistar rats, elevated plus maze, vehicle
#: (n=11) 30.7 +/- 6.4 s versus diazepam (n=8) 89.0 +/- 13.0 s on the open
#: arms. The dose reaches the receptor through the Phase 1 bridge
#: (drug_pk.diazepam_rat_ip_pk + brain_partition; 32.3 nM unbound brain at
#: 60 min), and the comparison with the model is made as a DRUG/VEHICLE
#: RATIO of anxiety, because `anxiety_index_readout()` returns a firing rate
#: and the plus maze returns seconds, with no published constant between
#: them. A ratio needs no such constant; what it does assume is that the two
#: anxiety scales move proportionally under drug, which is the one
#: assumption in this fit that no citation backs. It is stated here, in
#: `calibration/circuit_forward.py`, and in docs/architecture.md's P2.2
#: section, and its consequence is bounded by refitting against a second,
#: independently defined EPM anxiety index.
GABA_A_CONNECTION_MODULATION_GAIN = 0.0742


def gaba_a_connection_modulation_calibrated():
    # CALIBRATED as of P2.2 (see GABA_A_CONNECTION_MODULATION_GAIN above for
    # the source, the posterior and the one assumption). This is the only
    # circuit-layer number in this module that is no longer illustrative:
    # the potentiation gain scaling the interneuron -> BLA synapse under
    # diazepam. Every other circuit-layer quantity in this file - the
    # amygdala network's own weights, taus, gains and baseline inputs
    # included, and `gaba_a_connection_modulation()` above - is unchanged
    # and still illustrative.
    #
    # What the fit moved, and by how much: the hand-set placeholder was 0.5,
    # roughly seven times the fitted value, and it predicted a 43%
    # suppression of the anxiety index against the ~11% the published
    # elevated-plus-maze effect actually supports. So this is a real,
    # data-driven correction, not a confirmation of a lucky guess.
    return {
        connection: GABA_A_CONNECTION_MODULATION_GAIN
        for connection in gaba_a_connection_modulation()
    }


# ---------------------------------------------------------------------------
# Calibrated D2 receptor + dopamine/haloperidol PK, grounded in published
# data. Units: concentration in nM, time in minutes, kon in nM^-1 min^-1,
# koff/ke in min^-1 (same convention as the mu-opioid calibration above).
# Everything below this line is a real-data parameter set, kept deliberately
# separate from the illustrative functions above so the two are never
# confused. The basal ganglia indirect-pathway circuit
# (`basal_ganglia_indirect_pathway_network`, `d2_coupling`,
# `locomotor_activity_readout`) is reused as-is: this pass only calibrates
# receptor binding and PK, not circuit connectivity/gain/tau, which have no
# literature target and remain illustrative (see docs/architecture.md).
#
# Unlike the mu-opioid pass, NO kon-from-Ki/koff derivation trick was needed
# here: both sources below report kon AND koff directly, already measured
# (not derived), in units that convert cleanly to nM^-1 min^-1 / min^-1.
# What IS a real limitation, flagged explicitly: the two ligands were NOT
# measured in the same assay against the same competitor the way fentanyl
# and naloxone were in Volpe et al. 2011 (both antagonists there, one
# standardized [3H]-DAMGO assay). Haloperidol's numbers come from a
# TR-FRET competition-binding assay (an antagonist-vs-fluorescent-antagonist
# format); dopamine's come from a functional GIRK-channel-current assay (an
# agonist-vs-electrophysiological-response format), because no source was
# found reporting both ligands' kon/koff from one directly comparable assay.
# The two are corroborated independently (each study's own koff/kon ratio
# reproduces that same study's own reported equilibrium Kd, see the
# consistency check below), so each number is trustworthy on its own, but
# the fentanyl/naloxone-style "directly comparable in the same experiment"
# guarantee does not extend to this pair.
#
# Sources (see inline citations per parameter below):
#   [Sykes2017]  Sykes DA, Moore H, Stott L, Holliday N, Javitch JA, Lane JR,
#                Charlton SJ. "Extrapyramidal side effects of antipsychotics
#                are linked to their association kinetics at dopamine D2
#                receptors." Nat Commun. 2017;8:763.
#                doi:10.1038/s41467-017-00716-z. Open access (PMC5624946).
#                Time-resolved FRET (TR-FRET) competition-kinetics assay on
#                human D2L receptors; Table 1 reports kon and koff for
#                unlabeled haloperidol (and ~20 other antipsychotics)
#                competing against a fluorescent antagonist tracer:
#                koff = 0.65 +/- 0.07 min^-1 (t1/2(diss) = 1.07 min),
#                kon = (2.13 +/- 0.52) x 10^9 M^-1 min^-1, pKd = 9.49 +/- 0.08,
#                pKi = 9.48 +/- 0.08 (all read directly from the paper's
#                Table 1, verified against the article's full-text XML, not
#                a secondary source).
#   [Agren2021]  Agren R, Stepniewski TM, Zeberg H, Selent J, Sahlholm K.
#                "Dopamine D2 Receptor Agonist Binding Kinetics-Role of a
#                Conserved Serine Residue." Int J Mol Sci. 2021;22(8):4078.
#                doi:10.3390/ijms22084078. Open access (PMC8071183). Human
#                D2 receptor co-expressed with GIRK1/4 + RGS4 in Xenopus
#                oocytes, voltage-clamped; dopamine-induced GIRK current
#                activation/deactivation kinetics used to fit binding rate
#                constants directly (a functional assay, not a radioligand/
#                fluorescent-tracer competition assay): koff = 0.197 +/-
#                0.012 s^-1 (n=6), kon = (9.70 +/- 1.23) x 10^7 M^-1 s^-1,
#                pKd = 8.69 +/- 0.14, EC50 = 20 nM (pEC50 = 7.70 +/- 0.07).
#   [Kudo1999]   Kudo S, Ishizaki T. "Pharmacokinetics of haloperidol: an
#                update." Clin Pharmacokinet. 1999;37(6):435-56.
#                doi:10.2165/00003088-199937060-00001. A full-length
#                pharmacokinetic review; single-oral-dose terminal half-life
#                widely reported in the ~14-37 h range across studies
#                (corroborated by, e.g., a bioequivalence study reporting
#                ~62-66 h in one cohort and population-PK studies reporting
#                ~20-30 h in others), with chronic dosing sometimes producing
#                much longer apparent half-lives. We use ~20 h, a commonly
#                cited representative single-dose value, NOT a claim that
#                haloperidol truly follows one-compartment kinetics (same
#                explicit-simplification caveat as fentanyl/naloxone above).
#   [Kapur1997]  Kapur S, Zipursky R, Roy P, Jones C, Remington G, Reed K,
#                Houle S. "The relationship between D2 receptor occupancy
#                and plasma levels on low dose oral haloperidol: a PET
#                study." Psychopharmacology (Berl). 1997;131(2):148-52.
#                doi:10.1007/s002130050277. PET with [11C]-raclopride in 12
#                patients on 1-5 mg/day oral haloperidol; plasma levels
#                0.5-5.8 ng/mL corresponded to 53-88% D2 occupancy, fit to a
#                saturating hyperbola: 50% occupancy at 0.51 ng/mL, 80%
#                occupancy at 2.0 ng/mL. Used below as a validation target
#                (see tests/test_d2_scenario_calibrated.py), converted to nM
#                using haloperidol's molar mass (375.86 g/mol).
#   [Farde1988]  Farde L, Wiesel FA, Halldin C, Sedvall G. "Central
#                D2-dopamine receptor occupancy in schizophrenic patients
#                treated with antipsychotic drugs." Arch Gen Psychiatry.
#                1988;45(1):71-6. doi:10.1001/archpsyc.1988.01800250087012.
#                Foundational PET occupancy study establishing that clinical
#                antipsychotic dosing (11 chemically distinct drugs,
#                including haloperidol) produces 65-85% D2 occupancy; the
#                broader ~65-70% (efficacy) / ~80% (EPS threshold) framing
#                cited throughout this section traces to this and the
#                Kapur/Nordstrom PET literature that followed it.
#   [Garris1994] Garris PA, Ciolkowski EL, Pastore P, Wightman RM. "Efflux
#                of dopamine from the synaptic cleft in the nucleus
#                accumbens of the rat brain." J Neurosci. 1994;14(10):
#                6084-93. PMID:7931564. Fast-scan cyclic voltammetry in
#                vivo: a single stimulus pulse produces a transient ~0.25
#                microM extracellular dopamine peak, removed by uptake on a
#                50-100 ms timescale. Cited below ONLY as context for why no
#                literal ke is used for dopamine (see
#                `d2_pk_models_calibrated`); this is NOT a first-order
#                elimination-rate-constant measurement compatible with
#                `OneCompartmentPK`, and no such literal-ke value for
#                dopamine was found in the literature (see comment below).
#
# On the folklore this scenario was built to check, not assume: the
# original prompt for this section (and a widely repeated pharmacology
# folk-summary) asserts "haloperidol is well known for having an unusually
# SLOW D2 dissociation rate, and this is mechanistically why its EPS/motor
# side effects are prolonged." That traces to Kapur S, Seeman P.
# "Antipsychotic agents differ in how fast they come off the dopamine D2
# receptors. Implications for atypical antipsychotic action." J Psychiatry
# Neurosci. 2000;25(2):161-6 (PMID:10740989), which reported koff varying
# ~1000-fold (0.002 to 3.013 min^-1) across nine antipsychotics and framed
# "slow koff" as THE distinguishing feature of typical vs atypical
# antipsychotics. Checking this against later, larger, more rigorous kinetic
# panels does NOT hold up as stated:
#   - [Sykes2017]'s own panel (haloperidol among ~20 antipsychotics, all
#     measured in the SAME TR-FRET assay) puts haloperidol's koff
#     (0.65 min^-1, t1/2 ~1.07 min) toward the FAST end even among typical
#     antipsychotics: fluphenazine (koff 0.040, t1/2 17.3 min), spiperone
#     (koff 0.038, t1/2 18.2 min), nemonapride (koff 0.018, t1/2 38.5 min),
#     and (+)-butaclamol (koff 0.026, t1/2 26.7 min) all dissociate far more
#     slowly than haloperidol in this dataset, and haloperidol's koff is
#     actually FASTER than the atypical risperidone's (koff 0.43, t1/2
#     1.61 min) in the same panel.
#   - Sahlholm K, Zeberg H, Nilsson J, Ogren SO, Fuxe K, Arhem P. "The
#     fast-off hypothesis revisited: A functional kinetic study of
#     antipsychotic antagonism of the dopamine D2 receptor." Eur
#     Neuropsychopharmacol. 2016;26(3):467-76.
#     doi:10.1016/j.euroneuro.2016.01.001, using live-cell electrophysiology
#     rather than radioligand binding, found haloperidol's recovery from
#     blockade only 2.5- to 6.4-fold slower than the atypicals amisulpride,
#     clozapine, and quetiapine, not the ~1000-fold range the original
#     hypothesis implied.
#   - A published critique of [Sykes2017] itself, Zeberg H, Sahlholm K.
#     "Antipsychotics with similar association kinetics at dopamine D2
#     receptors differ in extrapyramidal side-effects." Nat Commun.
#     2018;9:3521. doi:10.1038/s41467-018-04489-x, argues it is kon
#     (association rate), not koff, that best predicts an antipsychotic's
#     EPS liability; haloperidol's kon in [Sykes2017] (2.13 x 10^9
#     M^-1 min^-1) is among the highest of the whole panel, which fits that
#     alternative story at least as well as "haloperidol has a slow koff."
# Net honest conclusion: haloperidol's dissociation from D2 is NOT
# dramatically slower than every atypical antipsychotic's the way the
# original folklore claims (that ~1000-fold framing has not held up under
# more rigorous, single-assay, larger panels); what the most recent,
# largest single-assay dataset ([Sykes2017]) actually shows is that haloperidol
# has an unusually HIGH association rate (kon), and a koff that is slow
# only relative to the endogenous agonist dopamine itself (dopamine's koff,
# from [Agren2021], is ~18x faster than haloperidol's, 11.82 vs 0.65
# min^-1), not relative to other antipsychotics. This scenario's
# `Ligand.koff`/`Ligand.kon` fields do carry the real, correctly-directed
# dopamine-vs-haloperidol asymmetry (dopamine fast on/off, haloperidol much
# slower), which is what actually matters for this codebase's competitive-
# occupancy ODE; it does NOT encode the (real but separate, and not
# reproducible with this codebase's two-parameter-per-ligand Ligand class)
# haloperidol-vs-other-antipsychotics kon story above, which is included
# here purely as a documented, cited correction to an assumption this
# section was explicitly asked to verify rather than take on faith.
def d2_dopamine_ligands_calibrated():
    dopamine = Ligand(
        name="dopamine",
        kon=5.82,    # nM^-1 min^-1; (9.70e7 M^-1 s^-1) * 60 * 1e-9 [Agren2021]
        koff=11.82,  # min^-1; 0.197 s^-1 * 60 [Agren2021]
        efficacy=1.0,
    )
    haloperidol = Ligand(
        name="haloperidol",
        kon=2.13,   # nM^-1 min^-1; 2.13e9 M^-1 min^-1 [Sykes2017]
        koff=0.65,  # min^-1 [Sykes2017]
        efficacy=0.0,
    )
    return [dopamine, haloperidol]


def d2_receptor_site_calibrated():
    return ReceptorSite(
        name="D2_on_striatal_indirect_MSN", ligands=d2_dopamine_ligands_calibrated()
    )


def d2_pk_models_calibrated():
    # haloperidol ke: ln(2) / (20 h * 60 min/h), using ~20 h as a commonly
    # cited representative single-oral-dose terminal half-life [Kudo1999]
    # (real reported range ~14-37 h single-dose, and considerably longer
    # under chronic dosing in some patients; see [Kudo1999] citation above
    # for the full caveat). This is the same kind of explicit one-
    # compartment collapse of real multi-phase kinetics used for
    # fentanyl/naloxone above, not a claim haloperidol truly has one-
    # compartment kinetics.
    #
    # dopamine: there is no real first-order plasma/synaptic elimination
    # rate constant to calibrate against, because dopamine here is not a
    # dosed drug: it's a synaptic transmitter cleared almost entirely by
    # active transporter reuptake (DAT) on a sub-second timescale
    # ([Garris1994]: extracellular dopamine from a single release event is
    # cleared within ~50-100 ms in vivo), which is a saturable
    # Michaelis-Menten transporter process, not first-order, and is
    # ~3-4 orders of magnitude faster than this codebase's minute-scale
    # time unit. Using the real timescale literally (a ke of very roughly
    # 500-1000 min^-1) would make dopamine's concentration state variable
    # vanish within a negligible fraction of one integration step for any
    # simulation window this codebase actually runs (tens of minutes),
    # collapsing a "dopamine dose" to a physically-real but practically
    # uninformative instantaneous delta function. No published first-order
    # ke for dopamine clearance was found (nor would one be expected, given
    # the real mechanism is transporter-saturable, not first-order), so
    # ke=5.0 below is an EXPLICIT, FLAGGED PROXY chosen only to preserve the
    # qualitative ordering this codebase's dynamics actually need (dopamine
    # clears far faster than haloperidol, t1/2 ~8.3 s here vs ~20 h for
    # haloperidol) at a timescale this simulation can still resolve. Do not
    # cite 5.0 as a measured dopamine clearance rate.
    return [OneCompartmentPK(ke=5.0), OneCompartmentPK(ke=0.00057762)]
