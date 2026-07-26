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
