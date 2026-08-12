"""P3.3: the hand-built scenarios, re-wired from the connectome.

    from neurocascade import connectome_scenarios as cs

    circuit = cs.scenario_connectome()              # 291-region Knox build
    ported = cs.port_scenario(cs.SCENARIO_PORTS["d2_indirect"], circuit)
    ported.network                                   # a plain CircuitNetwork
    print(ported.report.summary())

Nothing in `scenarios.py` moved. Every hand-built network, coupling, ligand
set and readout is still there, is still what the existing tests exercise,
and is what this module ports *from*: a `ScenarioPort` holds a reference to
the hand-built factory and each ported network is built by replacing that
network's connectivity, node by node, with the published tracer number for
the corresponding pair of CCF regions.

What "map a scenario onto connectome regions" actually requires
---------------------------------------------------------------

The hand-built scenarios do not have one node per brain region. They have
one node per *population*, and three of the four scenarios turn on a pair of
populations inside a single CCF region:

    VTA_GABA / VTA_DA           both are VTA (CCF 749)
    PAG_Interneuron / PAG_Output  both are PAG (CCF 795)
    GABA_Interneuron / BLA        both are BLA (CCF 295)
    Striatum_D2_MSN               one of two intermingled cell types in CP

A region-level tracer matrix cannot separate any of those, and P3.2 zeroes
the within-region diagonal on purpose (a Wilson-Cowan node already carries
its local recurrence in its gain). So a port has to say, for every edge,
which of four things the connectome has to offer, and record which:

1. **Both endpoints in different regions, source region signed.** The
   published `connection_density` for that ordered pair, times the source's
   transmitter sign, is the weight. Provenance ``"connectome"``.
2. **Both endpoints in the same region.** The connectome is structurally
   silent. The hand-built weight is carried over unchanged and stays
   ILLUSTRATIVE. Provenance ``"intra_regional"``.
3. **Source region carries no sign** (dopaminergic, serotonergic,
   noradrenergic, or no rule at all - see `connectome/neurotransmitter.py`).
   There is no fast-synaptic weight to take. Provenance
   ``"neuromodulatory_illustrative"`` if the hand-built edge is carried over
   (the default, so the scenario's own mechanism survives to be tested) or
   ``"unsigned_source"`` if it is dropped. Roadmap Phase 6 is what replaces
   these with a projection-system model; until then, a retained edge here is
   exactly as illustrative as it was in `scenarios.py`.
4. **The source node is a subpopulation whose long-range output the region's
   row does not isolate.** Two cases: a local interneuron has no long-range
   output at all (``"local_interneuron_source"``, weight 0), and a
   cell-type-defined projection node can carry an explicit, cited
   `Restriction` naming the targets that cell type actually innervates
   (``"restricted"`` for the pairs it excludes). Exactly one `Restriction`
   is declared in this module and its cost is reported as a fraction of the
   source region's published outgoing weight.

Two approximations are made rather than hidden
----------------------------------------------

**Long-range input is shared by every population in the target region.**
`PL -> VTA` is one published number and both `VTA_DA` and `VTA_GABA` sit in
those voxels, so both receive it. The alternative - giving the whole
projection to the principal population - would assert that cortical afferents
avoid local interneurons, which is false (feedforward inhibition is the
standard motif). Neither choice is measured by this data.

**Long-range output leaves only from the projection population.** A node
whose role is ``"local_interneuron"`` gets no outgoing inter-regional edge,
which is what "local" means.

**A ported scenario is a sub-network, not a small brain.** Every path
through the other ~280 regions has been deleted, exactly as
`connectome/fixtures/` warns. What is being tested here is whether the
scenario's own qualitative result survives having its *weights* replaced by
published ones, not whether four regions are a model of anything.

The weight scale, and why this module does not need to choose one
-----------------------------------------------------------------

P3.2 refused to pick a `weight_scale` because P2.3 showed coupling gain and
connection weight are degenerate against behavioural data, and reported that
at 291 nodes the unrescaled densities drive every node into saturation
(signed row sums -17.6 to +32.5 against an illustrative `gain=4.0`).

At scenario scale that problem does not arise, and the numbers say so: the
mean per-node total input weight `mean_i sum_j |W_ij|` of the ported networks
at `weight_scale=1.0` is 2.0 (reward), 1.7 (multi-region), 3.7 (D2) and 1.6
(GABA-A) against 1.9, 1.4, 1.6 and 1.5 for the corresponding hand-built
networks - within a factor of 2.3 everywhere. So this module's default is
P3.2's default, `weight_scale=1.0`, meaning *the published number, unrescaled*,
and no new constant is introduced. The whole-brain saturation P3.2 reported is
a property of summing 291 rows, not of the densities themselves.
`weight_scale` is still a keyword, and `tests/test_connectome_scenarios.py`
sweeps it, because "the result survives at exactly one scale" and "the result
survives" are different claims.

Baselines: `"hand_built"` (default) and `"rest_balanced"`
---------------------------------------------------------

The default port changes the connectivity and *nothing else*: `tau`, `gain`,
`threshold` and `baseline_input` are carried over from the hand-built node,
so a change in a result is attributable to the connectome alone.

That is the right default and it has one known weakness, which the D2
scenario runs straight into: `baseline_input` in the hand-built scenarios was
chosen *against the hand-built weights*, to put each node in the sensitive
part of its sigmoid. `rest_balanced_baselines()` restates that intent as an
arithmetic rule instead of a hand-set number, and the rule is not invented
here - applied to the hand-built basal-ganglia network it reproduces that
network's own `baseline_input` values exactly (0, 1, 1, -1, 1), which is
asserted by a test. Passing `baselines="rest_balanced"` applies the same rule
to the ported weights. It is a preview of P3.4, which owns baseline stability
at scale; this module uses it only as a diagnostic, to separate "the result
died because a node saturated" from "the result died because the anatomy
disagrees".

What this module found
----------------------

Three of the four ported scenarios reproduce their qualitative result -
agonism, inert antagonism alone, and antagonist reversal - with published
weights and no retuning. The D2 indirect-pathway scenario does not, under
either baseline policy, and `docs/architecture.md`'s P3.3 section records the
diagnosis: a pallidothalamic projection from GPe that the textbook
box-and-arrow diagram does not have, and that is ~7.6x denser than the GPi
one the diagram is built on.
"""

import numpy as np

from . import connectome as _connectome
from . import scenarios
from .circuit import CircuitNode
from .engine import Simulation
from .network import CircuitNetwork

#: A scenario node is either the population whose axons leave the region
#: (and therefore appear in a tracer matrix) or a local interneuron
#: population (which does not).
POPULATION_ROLES = ("projection", "local_interneuron")

#: What to do with an edge whose source region has no transmitter sign.
#: "retain_illustrative" keeps the hand-built weight, flagged; "drop" zeroes
#: it. There is deliberately no option that invents a sign, which is the
#: same refusal `connectome/build.py` makes.
UNSIGNED_EDGE_POLICIES = ("retain_illustrative", "drop")

BASELINE_POLICIES = ("hand_built", "rest_balanced")

PROVENANCES = (
    "connectome",
    "intra_regional",
    "neuromodulatory_illustrative",
    "unsigned_source",
    "local_interneuron_source",
    "restricted",
)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


class Restriction(object):
    """The targets a cell-type-defined node's axons actually reach.

    A region's row in the connectome is the sum over every projection cell
    type in it. When a scenario node names one of those cell types, the row
    over-claims, and there is no way to split it from this data. A
    `Restriction` is the explicit, cited alternative to pretending otherwise:
    it names the allowed targets and says on whose authority, and the
    `PortReport` records what fraction of the source region's published
    outgoing weight it threw away.

    It is a modelling assertion, not a measurement. The measurement that
    would replace it is Cre-line-specific tracer data (Allen has Drd1a-Cre
    and Adora2a-Cre experiments), which is a P3.1-shaped ingestion task and
    is not attempted here.
    """

    def __init__(self, targets, reason, citations):
        if not targets:
            raise ValueError("a Restriction with no targets deletes the node's output")
        if not citations:
            raise ValueError(
                "a Restriction is an anatomical claim and needs a citation; "
                "%r has none" % (reason,)
            )
        self.targets = tuple(targets)
        self.reason = reason
        self.citations = tuple(citations)

    def allows(self, target_region):
        return target_region in self.targets

    def __repr__(self):
        return "Restriction(targets=%r)" % (self.targets,)


class RegionMapping(object):
    """One scenario node placed on one CCF region."""

    def __init__(self, node, region, role, restriction=None, note=""):
        if role not in POPULATION_ROLES:
            raise ValueError(
                "role must be one of %s, got %r" % (POPULATION_ROLES, role)
            )
        if role == "local_interneuron" and restriction is not None:
            raise ValueError(
                "a local interneuron has no long-range output to restrict (%s)" % node
            )
        self.node = node
        self.region = region
        self.role = role
        self.restriction = restriction
        self.note = note

    def __repr__(self):
        return "RegionMapping(%r -> %r, %s)" % (self.node, self.region, self.role)


class ScenarioPort(object):
    """A hand-built scenario, its region map, and how to simulate it.

    `hand_built` is called for the original `CircuitNetwork` - which is
    never modified - and `simulation(network)` builds the same `Simulation`
    the scenario's own tests build, on whichever network it is handed. That
    is what makes hand-built and ported runs comparable: one code path, two
    connectivity matrices.
    """

    def __init__(
        self,
        key,
        title,
        hand_built,
        mappings,
        simulation,
        readouts,
        agonist,
        antagonist,
        notes="",
    ):
        self.key = key
        self.title = title
        self.hand_built = hand_built
        self.mappings = tuple(mappings)
        self._simulation = simulation
        #: readout-name -> (BehaviorReadout factory, the direction the
        #: scenario claims the agonist moves it: +1 up, -1 down)
        self.readouts = dict(readouts)
        self.agonist = agonist
        self.antagonist = antagonist
        self.notes = notes
        self.by_node = {m.node: m for m in self.mappings}
        names = [node.name for node in hand_built().nodes]
        missing = [n for n in names if n not in self.by_node]
        if missing:
            raise ValueError("%s: unmapped scenario nodes %s" % (key, missing))
        extra = [m.node for m in self.mappings if m.node not in names]
        if extra:
            raise ValueError("%s: mapping names no such node %s" % (key, extra))
        self.node_names = names

    def regions(self):
        return [self.by_node[n].region for n in self.node_names]

    def simulation(self, network):
        return self._simulation(network)

    def doses(self, agonist=0.0, antagonist=0.0):
        """Dose vector in this scenario's own ligand order."""
        return self.simulation(self.hand_built()).dose_vector(
            **{self.agonist: agonist, self.antagonist: antagonist}
        )

    def __repr__(self):
        return "ScenarioPort(%r, %d nodes)" % (self.key, len(self.node_names))


# ---------------------------------------------------------------------------
# The port itself
# ---------------------------------------------------------------------------


class PortedEdge(object):
    """One entry of the ported connectivity matrix and where it came from.

    `published_weight` is what the connectome offers for this ordered pair of
    regions, whether or not the port used it, and `None` when the connectome
    structurally has nothing to offer (both endpoints in the same region).
    Keeping it separate from `weight` is what lets the report say how much
    published signal a `Restriction` or an unsigned source discarded, instead
    of showing a zero and leaving the reader to work out whether that zero
    was in the data or was a decision.
    """

    def __init__(
        self, post, pre, weight, provenance, hand_built_weight, published_weight,
        note="",
    ):
        self.post = post
        self.pre = pre
        self.weight = weight
        self.provenance = provenance
        self.hand_built_weight = hand_built_weight
        self.published_weight = published_weight
        self.note = note

    @property
    def discards_published_weight(self):
        """A non-zero published weight this port chose not to use."""
        return (
            self.provenance != "connectome"
            and self.published_weight is not None
            and self.published_weight != 0.0
        )

    @property
    def is_from_connectome(self):
        return self.provenance == "connectome"

    @property
    def is_illustrative(self):
        return self.provenance in ("intra_regional", "neuromodulatory_illustrative")

    @property
    def contradicts_hand_built(self):
        """The connectome answered, and it disagrees with the hand-built edge.

        Disagreement means either a sign flip or an edge the hand-built
        network asserted and the tracer data does not carry (below
        `CONTRADICTION_FLOOR` of the hand-built magnitude). Only defined for
        `"connectome"` edges: nothing else had an answer to disagree with.
        """
        if not self.is_from_connectome or self.hand_built_weight == 0.0:
            return False
        if self.weight == 0.0:
            return True
        if np.sign(self.weight) != np.sign(self.hand_built_weight):
            return True
        return abs(self.weight) < CONTRADICTION_FLOOR * abs(self.hand_built_weight)

    def __repr__(self):
        return "PortedEdge(%s -> %s, %+.4g, %s)" % (
            self.pre,
            self.post,
            self.weight,
            self.provenance,
        )


#: An edge counts as contradicted when the published weight is under 1% of
#: the hand-built one. 1% is a reporting threshold, not a model parameter:
#: it separates "the tracer data has essentially nothing here" from "the
#: tracer data has less here than the hand-built guess". Nothing in the
#: dynamics depends on it.
CONTRADICTION_FLOOR = 0.01


class PortReport(object):
    def __init__(
        self, port, by_node, circuit, weight_scale, unsigned_edges, baselines, edges
    ):
        self.port = port
        #: The mapping actually used, which is `port.by_node` unless the
        #: caller passed `overrides`. The report has to read this rather than
        #: the declaration, or a counterfactual build would print the
        #: scenario's declared regions while carrying different weights.
        self.by_node = dict(by_node)
        self.dataset_key = circuit.report.dataset_key
        self.citation = circuit.report.citation
        self.units = circuit.report.units
        self.resolution = circuit.report.resolution
        self.weight_scale = weight_scale
        self.unsigned_edges = unsigned_edges
        self.baselines = baselines
        self.edges = list(edges)
        self.discarded_by_restriction = {}

    def by_provenance(self, provenance):
        return [e for e in self.edges if e.provenance == provenance]

    def counts(self):
        """Edges per provenance, counting only the ones that say something.

        An edge is "trivial" when it is zero in the ported network, was zero
        in the hand-built one, and the connectome has nothing there either -
        i.e. all three sources agree there is no connection. Those are most
        of an n^2 matrix and counting them would drown the rest.
        """
        counts = {p: 0 for p in PROVENANCES}
        for edge in self.edges:
            if (
                edge.weight != 0.0
                or edge.hand_built_weight != 0.0
                or edge.discards_published_weight
            ):
                counts[edge.provenance] += 1
        return counts

    def discarded_edges(self):
        return [e for e in self.edges if e.discards_published_weight]

    def contradictions(self):
        return [e for e in self.edges if e.contradicts_hand_built]

    def illustrative_edges(self):
        return [e for e in self.edges if e.is_illustrative and e.weight != 0.0]

    def restriction_citations(self):
        keys = set()
        for mapping in self.by_node.values():
            if mapping.restriction is not None:
                keys.update(mapping.restriction.citations)
        return sorted(keys)

    def summary(self):
        counts = self.counts()
        lines = [
            "ported scenario %r (%s)" % (self.port.key, self.port.title),
            "  connectome: %s at resolution %r, weight_scale=%g"
            % (self.dataset_key, self.resolution, self.weight_scale),
            "  baselines=%r  unsigned_edges=%r" % (self.baselines, self.unsigned_edges),
            "  node -> CCF region:",
        ]
        for name in self.port.node_names:
            mapping = self.by_node[name]
            lines.append(
                "    %-18s %-5s %s%s"
                % (
                    name,
                    mapping.region,
                    mapping.role,
                    "" if mapping.restriction is None
                    else "  projects to %s only" % (mapping.restriction.targets,),
                )
            )
        lines.append("  edges by provenance (non-trivial only):")
        for provenance in PROVENANCES:
            lines.append("    %-30s %d" % (provenance, counts[provenance]))
        illustrative = self.illustrative_edges()
        if illustrative:
            lines.append(
                "  ILLUSTRATIVE weights carried over from scenarios.py (the "
                "connectome has no number for these):"
            )
            for edge in illustrative:
                lines.append(
                    "    %-18s <- %-18s %+8.4f  %s"
                    % (edge.post, edge.pre, edge.weight, edge.provenance)
                )
        contradictions = self.contradictions()
        if contradictions:
            lines.append("  edges where the published data contradicts the hand-built guess:")
            for edge in contradictions:
                lines.append(
                    "    %-18s <- %-18s hand %+8.4f  published %+10.6f"
                    % (edge.post, edge.pre, edge.hand_built_weight, edge.weight)
                )
        discarded = self.discarded_edges()
        if discarded:
            lines.append(
                "  published weights this map deliberately does NOT use "
                "(the node is a subpopulation, or has no long-range output):"
            )
            for edge in discarded:
                lines.append(
                    "    %-18s <- %-18s published %+10.4f  %s"
                    % (edge.post, edge.pre, edge.published_weight, edge.provenance)
                )
        for node, fraction in sorted(self.discarded_by_restriction.items()):
            lines.append(
                "  restriction on %s discarded %.1f%% of that region's published "
                "outgoing weight" % (node, 100.0 * fraction)
            )
        for citation in self.restriction_citations():
            lines.append("    %s" % citation)
        return "\n".join(lines)


class PortedCircuit(object):
    def __init__(self, port, network, report):
        self.port = port
        self.network = network
        self.report = report

    def weight(self, pre, post):
        """Weight of the edge `pre -> post`, by scenario node name."""
        i = self.network.index[post]
        j = self.network.index[pre]
        return float(self.network.connectivity[i, j])

    def simulation(self):
        return self.port.simulation(self.network)

    def __repr__(self):
        return "PortedCircuit(%r, %d nodes)" % (self.port.key, self.network.n)


def port_scenario(
    port,
    circuit,
    weight_scale=1.0,
    unsigned_edges="retain_illustrative",
    baselines="hand_built",
    overrides=None,
):
    """Rebuild `port`'s hand-built network with connectome weights.

    `circuit` is a `connectome.ConnectomeCircuit` at resolution "regions".
    `overrides` optionally replaces individual `RegionMapping`s by node name,
    which is how the tests explore counterfactual maps (a different thalamic
    nucleus, an unrestricted striatal node) without a second declaration of
    the whole scenario.
    """
    if unsigned_edges not in UNSIGNED_EDGE_POLICIES:
        raise ValueError(
            "unsigned_edges must be one of %s, got %r"
            % (UNSIGNED_EDGE_POLICIES, unsigned_edges)
        )
    if baselines not in BASELINE_POLICIES:
        raise ValueError(
            "baselines must be one of %s, got %r" % (BASELINE_POLICIES, baselines)
        )
    if circuit.report.resolution != "regions":
        raise ValueError(
            "a scenario maps onto individual CCF regions, so it needs a "
            "connectome built at resolution='regions'; got %r"
            % circuit.report.resolution
        )

    by_node = dict(port.by_node)
    if overrides:
        for node, mapping in overrides.items():
            if node not in by_node:
                raise ValueError("%s has no node %r to override" % (port.key, node))
            by_node[node] = mapping

    signed = {a.acronym: a for a in circuit.report.fine_assignments}
    for mapping in by_node.values():
        if mapping.region not in signed:
            raise ValueError(
                "%s maps %s onto region %r, which this connectome does not "
                "contain" % (port.key, mapping.node, mapping.region)
            )
    # Two projection nodes in one region would each receive that region's
    # whole published outgoing weight, delivering the same projection twice.
    # That is exactly the double-count the local-interneuron rule exists to
    # prevent, so it is refused rather than left to a reader to notice.
    projections = {}
    for mapping in by_node.values():
        if mapping.role != "projection":
            continue
        if mapping.region in projections:
            raise ValueError(
                "%s puts two projection populations in region %s (%s and %s), "
                "which would deliver that region's published output twice; at "
                "most one node per region may be a projection population"
                % (port.key, mapping.region, projections[mapping.region], mapping.node)
            )
        projections[mapping.region] = mapping.node

    hand = port.hand_built()
    names = port.node_names
    n = len(names)
    W = np.zeros((n, n))
    edges = []
    for i, post in enumerate(names):
        for j, pre in enumerate(names):
            hand_weight = float(hand.connectivity[hand.index[post], hand.index[pre]])
            source = by_node[pre]
            target = by_node[post]
            weight, provenance, published, note = _edge(
                source, target, circuit, signed, hand_weight, weight_scale, unsigned_edges
            )
            W[i, j] = weight
            edges.append(
                PortedEdge(post, pre, weight, provenance, hand_weight, published, note)
            )

    nodes = [
        CircuitNode(
            name=node.name,
            tau=node.tau,
            baseline_input=node.baseline_input,
            gain=node.gain,
            threshold=node.threshold,
        )
        for node in hand.nodes
    ]
    network = CircuitNetwork(nodes=nodes, connectivity=W)
    if baselines == "rest_balanced":
        network = rest_balanced_baselines(network)

    report = PortReport(
        port, by_node, circuit, weight_scale, unsigned_edges, baselines, edges
    )
    report.discarded_by_restriction = _restriction_cost(by_node, circuit)
    return PortedCircuit(port, network, report)


def _edge(source, target, circuit, signed, hand_weight, weight_scale, unsigned_edges):
    """`(weight, provenance, published_weight, note)` for one ordered pair."""
    if source.region == target.region:
        # P3.2 zeroes the within-region diagonal on purpose: for a bulk
        # tracer it is mostly local collaterals, and a Wilson-Cowan node
        # already carries its local recurrence in its gain. So the
        # connectome has nothing to say about a within-region edge, and the
        # hand-built number survives, still illustrative.
        return (
            hand_weight,
            "intra_regional",
            None,
            "connectome is silent within a region",
        )
    # What the tracer data offers here, used or not. An unsigned source has
    # a zero column in the built connectome by construction, so this is 0.0
    # for those, which is exactly the point: there is nothing to discard.
    published = circuit.weight(source.region, target.region)
    if source.role == "local_interneuron":
        return 0.0, "local_interneuron_source", published, "no long-range output"
    if source.restriction is not None and not source.restriction.allows(target.region):
        return 0.0, "restricted", published, source.restriction.reason
    if not signed[source.region].is_signed:
        transmitter = signed[source.region].transmitter
        note = "source region %s is %s" % (
            source.region,
            "unruled" if transmitter is None else transmitter,
        )
        if unsigned_edges == "retain_illustrative" and hand_weight != 0.0:
            return hand_weight, "neuromodulatory_illustrative", published, note
        return 0.0, "unsigned_source", published, note
    return weight_scale * published, "connectome", published, ""


def _restriction_cost(by_node, circuit):
    """Fraction of each restricted region's outgoing weight the map discards.

    Measured over the whole connectome, not over the scenario's node set,
    because the question a reader will ask is "how much of what this region
    actually does did you declare away", and the answer has to be about the
    region, not about the four other nodes that happened to be in the
    scenario.
    """
    cost = {}
    for node, mapping in by_node.items():
        if mapping.restriction is None:
            continue
        row = np.abs(circuit.connectivity[:, circuit.index_of(mapping.region)])
        total = float(row.sum())
        kept = sum(
            float(abs(circuit.weight(mapping.region, target)))
            for target in mapping.restriction.targets
            if target in circuit.acronyms
        )
        cost[node] = 0.0 if total == 0.0 else 1.0 - kept / total
    return cost


def rest_balanced_baselines(network, target_rate=0.5):
    """`baseline_input` that makes every node rest at `target_rate`, exactly.

    A Wilson-Cowan node is at equilibrium when `r_i = sigmoid(net_i)`, and
    `sigmoid(x) = 0.5` exactly at `x = threshold`, so requiring every node to
    rest at 0.5 is one line of algebra rather than a fit:

        baseline_i = threshold_i - sum_j W[i, j] * target_rate

    This is not a new modelling idea introduced by P3.3. It is the rule the
    hand-built basal-ganglia network was already built with - "baseline_input
    values are chosen so each node sits near the sensitive (non-saturating)
    part of its sigmoid at rest", `scenarios.py` - restated as arithmetic.
    Applied to that network's own weights it returns that network's own
    numbers, (0, 1, 1, -1, 1), which a test asserts.

    Being an exact fixed point is not the same as being the fixed point the
    system settles into: whether it is stable, and whether it is the one an
    integration from r = 0 finds, are properties of `W`, and for the ported
    D2 network they are the substance of P3.3's finding. P3.4 owns the
    general version of this problem.
    """
    W = network.connectivity
    nodes = [
        CircuitNode(
            name=node.name,
            tau=node.tau,
            baseline_input=node.threshold - float(W[i].sum()) * target_rate,
            gain=node.gain,
            threshold=node.threshold,
        )
        for i, node in enumerate(network.nodes)
    ]
    return CircuitNetwork(nodes=nodes, connectivity=W)


# ---------------------------------------------------------------------------
# Loading a connectome to port onto
# ---------------------------------------------------------------------------


def scenario_connectome(source="cache", cache_dir=None, allow_download=False):
    """A `ConnectomeCircuit` at region resolution, for porting onto.

    `source="cache"` reads the checksum-verified Knox et al. 2019 files
    (291 regions); `source="fixture"` reads the committed 15-region
    fixtures. Those fifteen regions are exactly the ones the scenarios name,
    and because a region-pair connection density is a per-region quantity,
    the two sources give *identical* weights for every pair a scenario uses.
    `tests/test_connectome_scenarios.py` asserts that equality against the
    real files rather than asserting it here, so the claim is checked rather
    than believed. It is what lets the whole regression suite run offline.
    """
    if source == "fixture":
        ontology = _connectome.load_fixture_ontology()
        strength, voxels = _connectome.load_fixture_build_inputs(ontology=ontology)
    elif source == "cache":
        ontology = _connectome.load_ontology(
            cache_dir=cache_dir, allow_download=allow_download
        )
        strength, voxels = _connectome.load_knox_inputs(
            cache_dir=cache_dir, ontology=ontology, allow_download=allow_download
        )
    else:
        raise ValueError("source must be 'cache' or 'fixture', got %r" % (source,))
    return _connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )


# ---------------------------------------------------------------------------
# The declarations, one per hand-built scenario
# ---------------------------------------------------------------------------
#
# Region choices that are not a literal name match, and why:
#
#   NAc  -> ACB   The CCF's nucleus accumbens. A name change, not a
#                 modelling choice.
#   PFC  -> PL    The mouse CCF has no "prefrontal cortex" structure. PL
#                 (prelimbic area) is the area most consistently treated as
#                 the rodent medial-prefrontal homologue in the
#                 cortico-basal-ganglia literature, and it is the one the
#                 mesocortical dopamine projection is usually described to.
#                 ILA and ACAd are the defensible alternatives; the tests
#                 run the port with each of them and report whether the
#                 result depends on the choice.
#   Thalamus -> MD  The hand-built node is generic. MD is the thalamic
#                 nucleus in the connectome fixture, and Knox's GPi -> MD,
#                 GPi -> VAL, GPi -> VM, GPi -> CM and GPi -> PF densities
#                 are all within a factor of four of each other
#                 (-0.054 to -0.181), so the choice does not carry the
#                 result. A full-data test re-runs the scenario with each.
#
# Every other node maps onto the CCF structure of the same name.

_STRIATAL_INDIRECT_RESTRICTION = Restriction(
    targets=("GPe",),
    reason=(
        "Striatum_D2_MSN is the indirect-pathway (D2/A2a) spiny projection "
        "neuron, whose long-range axon terminates in the external pallidum; "
        "the direct-pathway (D1) neurons intermingled with it in the same "
        "caudoputamen voxels are the ones that innervate GPi/EPN and SNr. "
        "A region-level tracer matrix sums both cell types into one CP row "
        "(Knox et al. 2019 gives CP -> GPe -12.56, CP -> GPi -12.21 and "
        "CP -> SNr -16.34, i.e. the direct-pathway targets carry MORE weight "
        "than the indirect one), so without this restriction the node is not "
        "the cell type the scenario names."
    ),
    citations=(
        "[Gerfen2011] Gerfen CR, Surmeier DJ. 'Modulation of striatal "
        "projection systems by dopamine.' Annu Rev Neurosci. 2011;34:441-466. "
        "doi:10.1146/annurev-neuro-061010-113641. Mouse/rat: D1-expressing "
        "spiny neurons project to SNr/GPi, D2/A2a-expressing spiny neurons "
        "project to GPe.",
        "[Albin1989] Albin RL, Young AB, Penney JB. 'The functional anatomy "
        "of basal ganglia disorders.' Trends Neurosci. 1989;12(10):366-375. "
        "doi:10.1016/0166-2236(89)90074-x. The direct/indirect formulation "
        "the hand-built scenario encodes.",
    ),
)


def _mu_simulation(network):
    coupling = (
        scenarios.mu_opioid_multi_region_coupling()
        if "LC" in network.index
        else scenarios.mu_opioid_coupling()
    )
    return Simulation(
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        network=network,
        coupling=coupling,
    )


def _d2_simulation(network):
    return Simulation(
        pk_models=scenarios.d2_pk_models(),
        receptor_site=scenarios.d2_receptor_site(),
        network=network,
        coupling=scenarios.d2_coupling(),
    )


def _gaba_a_simulation(network):
    return Simulation(
        pk_models=scenarios.gaba_a_pk_models(),
        receptor_site=scenarios.gaba_a_receptor_site(),
        network=network,
        coupling=scenarios.gaba_a_coupling(),
        connection_modulation=scenarios.gaba_a_connection_modulation(),
    )


REWARD_PORT = ScenarioPort(
    key="mu_opioid_reward",
    title="mu-opioid VTA -> NAc -> PFC reward circuit",
    hand_built=scenarios.reward_circuit_network,
    simulation=_mu_simulation,
    agonist="fentanyl",
    antagonist="naloxone",
    readouts={"incentive_salience": (scenarios.incentive_salience_readout, +1)},
    mappings=[
        RegionMapping(
            "VTA_GABA", "VTA", "local_interneuron",
            note="local GABAergic interneurons inside the VTA",
        ),
        RegionMapping(
            "VTA_DA", "VTA", "projection",
            note="the VTA's dopaminergic projection population; the region "
                 "carries no sign in connectome/neurotransmitter.py, so its "
                 "outgoing edges are the unsigned-source case",
        ),
        RegionMapping("NAc", "ACB", "projection"),
        RegionMapping("PFC", "PL", "projection"),
    ],
    notes=(
        "The one edge the connectome cannot supply here is the mesolimbic "
        "dopamine projection VTA_DA -> NAc, and that is by construction: "
        "P3.2 leaves VTA unsigned because a dopaminergic axon does not add a "
        "synaptic term to its target, it changes how the target responds, "
        "which is what the receptor layer already models. Roadmap Phase 6 "
        "replaces it."
    ),
)


MULTI_REGION_PORT = ScenarioPort(
    key="mu_opioid_multi_region",
    title="mu-opioid reward + PAG analgesia + LC arousal",
    hand_built=scenarios.multi_region_mu_opioid_network,
    simulation=_mu_simulation,
    agonist="fentanyl",
    antagonist="naloxone",
    readouts={
        "incentive_salience": (scenarios.incentive_salience_readout, +1),
        "analgesia_index": (scenarios.analgesia_index_readout, +1),
        "arousal_index": (scenarios.arousal_index_readout, -1),
    },
    mappings=[
        RegionMapping("VTA_GABA", "VTA", "local_interneuron"),
        RegionMapping("VTA_DA", "VTA", "projection"),
        RegionMapping("NAc", "ACB", "projection"),
        RegionMapping("PFC", "PL", "projection"),
        RegionMapping(
            "PAG_Interneuron", "PAG", "local_interneuron",
            note="mu-opioid-bearing GABAergic interneurons in the PAG",
        ),
        RegionMapping(
            "PAG_Output", "PAG", "projection",
            note="PAG has no sign rule at all in connectome/"
                 "neurotransmitter.py, so its outgoing edges are unsigned",
        ),
        RegionMapping("LC", "LC", "projection"),
    ],
)


D2_PORT = ScenarioPort(
    key="d2_indirect",
    title="D2 basal-ganglia indirect pathway",
    hand_built=scenarios.basal_ganglia_indirect_pathway_network,
    simulation=_d2_simulation,
    agonist="dopamine",
    antagonist="haloperidol",
    readouts={"locomotor_activity": (scenarios.locomotor_activity_readout, +1)},
    mappings=[
        RegionMapping(
            "Striatum_D2_MSN", "CP", "projection",
            restriction=_STRIATAL_INDIRECT_RESTRICTION,
        ),
        RegionMapping("GPe", "GPe", "projection"),
        RegionMapping("STN", "STN", "projection"),
        RegionMapping("GPi", "GPi", "projection"),
        RegionMapping(
            "Thalamus", "MD", "projection",
            note="the hand-built node is generic 'thalamus'; see the module "
                 "comment on why MD and why the choice does not carry the "
                 "result",
        ),
    ],
    notes=(
        "This is the scenario whose qualitative result does NOT survive the "
        "port. See docs/architecture.md, P3.3."
    ),
)


GABA_A_PORT = ScenarioPort(
    key="gaba_a_amygdala",
    title="GABA-A / benzodiazepine amygdala anxiety circuit",
    hand_built=scenarios.amygdala_anxiety_circuit_network,
    simulation=_gaba_a_simulation,
    agonist="diazepam",
    antagonist="flumazenil",
    readouts={"anxiety_index": (scenarios.anxiety_index_readout, -1)},
    mappings=[
        RegionMapping(
            "GABA_Interneuron", "BLA", "local_interneuron",
            note="the local GABAergic interneurons whose synapse onto BLA "
                 "principal cells is the one benzodiazepines potentiate",
        ),
        RegionMapping("BLA", "BLA", "projection"),
        RegionMapping("CeA", "CEA", "projection"),
    ],
    notes=(
        "The modulated connection (BLA <- GABA_Interneuron) is intra-BLA, so "
        "the connectome is silent about it by construction and the "
        "illustrative weight is carried over. What the port replaces is the "
        "BLA -> CeA weight, which becomes Knox et al. 2019's published "
        "BLA -> CEA density."
    ),
)


SCENARIO_PORTS = {
    port.key: port
    for port in (REWARD_PORT, MULTI_REGION_PORT, D2_PORT, GABA_A_PORT)
}
