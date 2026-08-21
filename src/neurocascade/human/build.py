"""Turn a human structural matrix into a species-neutral `CircuitNetwork`.

    from neurocascade import human

    conn = human.load_connectome()
    built = human.build_circuit_network(conn)
    print(built.report.summary())

The engine (`circuit.py`, `network.py`, `engine.py`) is species-neutral:
it takes `CircuitNode`s and a `connectivity` matrix and integrates them.
The mouse connectome builder (`connectome/build.py`) does the same job for
the Allen matrices. This module is the human half: it constructs the same
`CircuitNetwork` from the ENIGMA Desikan-Killiany matrix, and it adds no
new physics.

The one thing it must do that the mouse builder does not
---------------------------------------------------------

The mouse matrix is signed (from cited neurotransmitter identity) and
directed (tracer injection). The human matrix is NEITHER: dMRI
tractography is undirected by construction and carries no
excitatory/inhibitory sign. The mouse builder signs each fine source region
before aggregation; this builder signs each source region in place, from the
cited per-region table in `human/neurotransmitter.py`:

* cortex, hippocampus and thalamus -> glutamatergic (+1)
* accumbens, caudate, putamen, pallidum -> GABAergic (-1)
* amygdala -> MIXED, unsigned, because one coarse Desikan-Killiany region
  contains a glutamatergic basolateral complex AND a GABAergic
  central/medial complex and has no single defensible sign

Because the matrix is symmetric, one published weight `W[i, j]` is entered
twice and signed once per direction: the `i -> j` direction is signed by
region `i`'s transmitter and `j -> i` by region `j`'s. Whether the bundle
truly carries fibres both ways is exactly what dMRI cannot measure; this is
a modelling convention, stated here rather than hidden. Unsigned source
regions (the amygdala) have their outgoing columns excluded by default and
the excluded fraction reported; `unsigned_sources="raise"` refuses to build
instead.

Node parameters are the same ILLUSTRATIVE Wilson-Cowan defaults the mouse
builder uses (`connectome/build.py`'s `DEFAULT_NODE`): no human time
constant or transfer-function slope has been fitted, and the builder does
not pretend otherwise. `stability.stabilize()` rescales the weights and
solves for a balanced baseline exactly as it does for a mouse-built network,
because the network it returns is the same type.
"""

import numpy as np

from ..circuit import CircuitNode
from ..network import CircuitNetwork
from ..circuit import DEFAULT_NODE
from . import neurotransmitter as nt

UNSIGNED_POLICIES = ("exclude", "raise")


class BuildReport(object):
    """What the build decided, in a form a test can assert on."""

    def __init__(
        self,
        dataset_key,
        citation,
        units,
        atlas,
        labels,
        weight_scale,
        node_parameters,
        assignments,
        signed_weight_fraction,
        excluded_weight_fraction,
        excluded_by_transmitter,
        unsigned_sources,
    ):
        self.dataset_key = dataset_key
        self.citation = citation
        self.units = units
        self.atlas = atlas
        self.labels = list(labels)
        self.weight_scale = weight_scale
        self.node_parameters = dict(node_parameters)
        self.assignments = list(assignments)
        self.signed_weight_fraction = signed_weight_fraction
        self.excluded_weight_fraction = excluded_weight_fraction
        self.excluded_by_transmitter = dict(excluded_by_transmitter)
        self.unsigned_sources = unsigned_sources

    @property
    def n_nodes(self):
        return len(self.labels)

    def signed_regions(self):
        return [a for a in self.assignments if a.is_signed]

    def unsigned_regions(self):
        return [a for a in self.assignments if not a.is_signed]

    @property
    def inhibitory_sources(self):
        """Region labels whose outgoing edges are signed inhibitory (-1)."""
        return [a.label for a in self.assignments if a.sign == -1.0]

    def transmitter_counts(self):
        counts = {}
        for assignment in self.assignments:
            key = assignment.transmitter
            counts[key] = counts.get(key, 0) + 1
        return counts

    def citations(self):
        keys = set()
        for assignment in self.assignments:
            keys.update(assignment.citations)
        return sorted(keys)

    def summary(self):
        counts = self.transmitter_counts()
        lines = [
            "human connectome network: %s" % self.dataset_key,
            "  %d nodes (%d cortical + %d subcortical)"
            % (self.n_nodes, len(self.atlas.cortical()), len(self.atlas.subcortical())),
            "  weights: %s" % self.units,
            "  weight_scale=%g  unsigned_sources=%r"
            % (self.weight_scale, self.unsigned_sources),
            "  node parameters (ILLUSTRATIVE): %s"
            % ", ".join(
                "%s=%g" % (k, v) for k, v in sorted(self.node_parameters.items())
            ),
            "  sign assignment (per source region):",
        ]
        for name in sorted(counts, key=lambda k: (k is None, k)):
            label = "no rule" if name is None else name
            sign = None if name is None else nt.TRANSMITTER_SIGN[name]
            lines.append(
                "    %-14s %3d region(s)  sign=%s"
                % (label, counts[name], "none" if sign is None else "%+d" % sign)
            )
        lines.append(
            "  fibre weight signed: %.1f%%  excluded (unsigned sources): %.1f%%"
            % (
                100.0 * self.signed_weight_fraction,
                100.0 * self.excluded_weight_fraction,
            )
        )
        if self.inhibitory_sources:
            lines.append(
                "  inhibitory sources: %s" % ", ".join(self.inhibitory_sources)
            )
        if self.unsigned_regions():
            lines.append(
                "  unsigned sources excluded: %s"
                % ", ".join(a.label for a in self.unsigned_regions())
            )
        return "\n".join(lines)


class HumanCircuit(object):
    """A `CircuitNetwork` plus the provenance of every number in it."""

    def __init__(self, network, labels, connectivity, report):
        self.network = network
        self.labels = list(labels)
        self.connectivity = connectivity
        self.report = report

    @property
    def n(self):
        return len(self.labels)

    def index_of(self, label):
        return self.labels.index(label)

    def weight(self, source_label, target_label):
        """Signed weight of the projection source -> target.

        Named arguments in anatomical order, because `connectivity[i, j]`
        is `j -> i` and getting that backwards is this package's standing
        hazard (see `matrix.py`). For a symmetric dMRI weight signed per
        direction, `weight(a, b)` and `weight(b, a)` differ exactly when
        `a` and `b` have opposite transmitter signs.
        """
        return float(
            self.connectivity[self.index_of(target_label), self.index_of(source_label)]
        )

    def __repr__(self):
        return "HumanCircuit(%r, %d nodes)" % (self.report.dataset_key, self.n)


def build_circuit_network(
    connectome,
    weight_scale=1.0,
    node_parameters=None,
    unsigned_sources="exclude",
    rules=None,
):
    """Build a `CircuitNetwork` from a `HumanStructuralConnectome`.

    Every non-zero weight is signed by the SOURCE region's cited transmitter
    identity (`human/neurotransmitter.py`): +1 glutamatergic, -1 GABAergic.
    The symmetric dMRI matrix is entered once and signed per direction, so
    a cortico-striatal bundle becomes `+W` one way and `-W` the other.
    Unsigned source columns (the mixed amygdala) are excluded by default;
    `unsigned_sources="raise"` refuses to build instead.
    """
    if unsigned_sources not in UNSIGNED_POLICIES:
        raise ValueError(
            "unsigned_sources must be one of %s, got %r"
            % (UNSIGNED_POLICIES, unsigned_sources)
        )
    if node_parameters is None:
        node_parameters = dict(DEFAULT_NODE)

    labels = list(connectome.labels)
    atlas = connectome.atlas
    rules = nt.SignRules() if rules is None else rules
    assignments = rules.assign_all(atlas)

    if not any(a.is_signed for a in assignments):
        raise ValueError(
            "no region in this connectome has a sign rule; a network built "
            "from it would be empty"
        )
    if unsigned_sources == "raise" and not all(a.is_signed for a in assignments):
        unnamed = [a.label for a in assignments if not a.is_signed]
        raise ValueError(
            "unsigned_sources='raise' and %d of %d regions have no sign: %s"
            % (len(unnamed), len(assignments), ", ".join(unnamed))
        )

    # Sign each source column: +1, -1, or 0.0 (excluded unsigned source).
    signs = np.array(
        [0.0 if a.sign is None else float(a.sign) for a in assignments]
    )
    total_weight = float(connectome.weights.sum())
    signed_mask = np.array([a.is_signed for a in assignments], dtype=bool)

    excluded_by_transmitter = {}
    for column, assignment in enumerate(assignments):
        if assignment.is_signed:
            continue
        key = assignment.transmitter
        excluded_by_transmitter[key] = excluded_by_transmitter.get(
            key, 0.0
        ) + float(connectome.weights[:, column].sum())
    if total_weight > 0.0:
        excluded_by_transmitter = {
            k: v / total_weight for k, v in excluded_by_transmitter.items()
        }

    signed_weight = float(connectome.weights[:, signed_mask].sum())
    connectivity = weight_scale * connectome.weights * signs[None, :]

    nodes = [CircuitNode(name=label, **node_parameters) for label in labels]
    network = CircuitNetwork(nodes=nodes, connectivity=connectivity)

    report = BuildReport(
        dataset_key=connectome.dataset_key,
        citation=connectome.citation,
        units=(
            "%s, multiplied per direction by the source region's transmitter "
            "sign (+1 glutamatergic, -1 GABAergic); unsigned source columns "
            "are zero" % connectome.units
        ),
        atlas=atlas,
        labels=labels,
        weight_scale=weight_scale,
        node_parameters=node_parameters,
        assignments=assignments,
        signed_weight_fraction=(
            0.0 if total_weight == 0.0 else signed_weight / total_weight
        ),
        excluded_weight_fraction=(
            0.0 if total_weight == 0.0 else 1.0 - signed_weight / total_weight
        ),
        excluded_by_transmitter=excluded_by_transmitter,
        unsigned_sources=unsigned_sources,
    )
    return HumanCircuit(
        network=network,
        labels=labels,
        connectivity=connectivity,
        report=report,
    )
