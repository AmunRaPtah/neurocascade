"""P3.2: turn a published connectome into a `CircuitNetwork`.

    from neurocascade import connectome

    strength, voxels = connectome.load_knox_inputs()
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="major_divisions"
    )
    built.network            # a plain CircuitNetwork, ready for Simulation
    print(built.report.summary())

Nothing in `network.py` or `circuit.py` changed to make this work: this
module *constructs* the existing `CircuitNetwork(nodes, connectivity)` and
adds no method, no attribute and no keyword to it, so any external
importer of `neurocascade.CircuitNetwork` is unaffected.

The four decisions this module makes, and why each is a decision
----------------------------------------------------------------

**1. Which published number becomes a weight: connection DENSITY, not
normalized connection density.** P3.1's default matrix is the normalized
connection density, `s_ij / (|X_i| |Y_j|)`, which divides out both regions'
voxel counts. That is the right measure for asking "how strongly are these
two regions connected per unit tissue on each side", and it is the wrong
measure for a rate model's drive term. In

    tau_i dr_i/dt = -r_i + sigmoid(sum_j W[i,j] r_j + drive_i)

`r_j` is a mean rate *per neuron* in region `j`, so the drive that region
`j` delivers scales with how many neurons it has: dividing by `|X_j|`
would say that a large source and a tiny nucleus deliver the same input at
the same firing rate. Dividing by the *target's* size, on the other hand,
is exactly right, because the drive that matters is input per target
neuron. `s_ij / |Y_j|` is Knox et al. 2019's `connection_density`. So the
builder takes the raw `connection_strength` matrix plus voxel counts and
forms the density itself, which is also what makes coarse resolutions
possible at all (see `volumes.py`: only the raw strength is additive).

**2. Sign comes from `neurotransmitter.py` and is applied per SOURCE, at
the fine level, before aggregation.** Signing after aggregation would be a
different and worse model: "midbrain" has no sign of its own, but SNr
inside it does, so a coarse-grained edge should be the signed sum of its
members' edges, not an unsigned sum with a sign guessed afterwards. Doing it in this order means the coarse matrix has
net-excitatory and net-inhibitory entries that were computed rather than
assigned, and it is the reason a 12-node network built here is not simply a
cartoon.

**3. Unsigned sources are dropped, loudly, by default.** A region whose
projection neurons are dopaminergic, serotonergic, noradrenergic or of
genuinely mixed identity has no sign in a fast-synaptic connectivity
matrix; so does a region for which no rule exists. `unsigned_sources` is
"exclude" by default, which zeroes those columns and records in the report
exactly which regions and what fraction of the tracer weight went with
them. It can be set to "raise" for a build that must be complete. What it
cannot be set to is a sign, because that would be a fabricated number.

**4. The self-connection diagonal is dropped WITHIN a region and kept
BETWEEN regions.** Both published matrices have large non-zero diagonals -
a region's projection to itself, which for a bulk tracer is mostly local
intra-regional collaterals, and which is the largest entry in 109 of the
291 Knox rows. A Wilson-Cowan node already carries its own local recurrence
in its gain and threshold, so counting it again as an edge would
double-count it; that is the "drop_within_region" default. At a coarse
resolution the same diagonal entry means something different - MOp -> MOs
lands on the isocortex diagonal and is a genuine inter-areal projection -
so the fine diagonal is zeroed *before* aggregation and the coarse
diagonal that survives is real connectivity. `self_connections="keep"` and
`"drop"` are the two other honest options and both are available.

Resolutions, and the "roughly 43" in the roadmap
------------------------------------------------

    "regions"          the dataset's own region set: 291 for Knox et al.
                       2019, 213 for Oh et al. 2014
    "subdivisions"     one CCF level below the twelve major divisions:
                       60 regions for both datasets
    "major_divisions"  the twelve divisions Knox et al. 2019 published as
                       their own source/target mask

The roadmap asks for "major divisions -> ~43 regions -> ~213 regions". The
213 is Oh et al. 2014's region count and is reproduced exactly. **The 43 is
not a whole-brain parcellation**: it is the number of isocortical areas in
CCFv3 ("43 isocortical areas and their layers, 314 subcortical grey matter
structures, 81 fibre tracts and 8 ventricular structures", Wang et al.
2020, Cell 181:936-953, doi:10.1016/j.cell.2020.04.007), and it appears in
this data as a checkable fact rather than as a resolution: exactly 43 of
Knox's 291 regions lie in the isocortex, which is asserted by a test. The
CCF level that actually sits between the twelve divisions and the region
set is 60 regions wide, so that is what `"subdivisions"` is, and this
module reports its own count instead of claiming a number it does not
have. No search was made for a published 43-region whole-brain mouse
parcellation to match the roadmap's wording, because the arithmetic above
already explains where the number came from.

What is NOT built here
----------------------

**Both hemispheres.** Every network built here is one hemisphere: the
ipsilateral square block. Both datasets inject tracer into one hemisphere
only, so a bilateral network requires assuming left-right symmetry AND
deciding what to do with the five midline structures that have an
ipsilateral column and no contralateral one (RM, RO, RPA, OV, EW). Neither
is hard; both are modelling choices with no data behind them in these
files, and inventing them silently inside a builder is exactly the habit
this codebase does not have. The contralateral block is loaded, kept and
untouched by P3.1, so a later task can spend it.

**A weight scale, a baseline drive, or any node parameter.** `tau`, `gain`,
`threshold` and `baseline_input` are ILLUSTRATIVE defaults carried over
from the hand-built scenarios (`DEFAULT_NODE`), and `weight_scale` defaults
to 1.0, meaning the published density numbers enter the matrix unrescaled.
That is deliberate, and it is not a neutral choice masquerading as one: the
raw densities are not small in Wilson-Cowan terms. Individual entries are
(median non-zero |w| = 2.2e-3 at region resolution), but the row sums are
not: the 291-node network's signed input weight per node runs from -17.6 to
+32.5 with a median of +7.4, so with the illustrative `gain=4.0` every node
is driven deep into saturation at rest. A network built from this module
with all defaults is therefore *not* a ready-to-simulate model, and this
paragraph is the warning. It is also the GPi-saturation lesson from the D2
scenario reappearing at 291 nodes, which is exactly what P3.4 exists to
solve; a test asserts the row-sum range so the claim cannot rot.

The reason this module does not fix it by picking a scale is P2.3: a
coupling gain and a connection weight are degenerate against behavioural
data - only their product is identified - so choosing a scale here would be
choosing a calibration, and that scale would then propagate silently into
every result labelled "connectome-based". Setting `weight_scale` is a
one-line, explicit, illustrative act by the caller; P3.4 owns the
principled version of it.
"""

import numpy as np

from ..circuit import CircuitNode
from ..network import CircuitNetwork
from . import neurotransmitter as nt
from .ontology import MAJOR_DIVISION_IDS

RESOLUTIONS = ("regions", "subdivisions", "major_divisions")

UNSIGNED_POLICIES = ("exclude", "raise")

SELF_CONNECTION_POLICIES = ("drop_within_region", "drop", "keep")

#: ILLUSTRATIVE node parameters. `tau=1.0`, `gain=4.0`, `threshold=0.0` and
#: `baseline_input=0.0` are the values the hand-built scenarios in
#: `scenarios.py` use, and they are placeholders there too: no mouse
#: population time constant or transfer-function slope has been fitted or
#: sourced for any of these regions. They are here so a network can be
#: built at all, and P3.4 (baseline stability at scale) is the task that
#: has to replace `baseline_input` with something principled.
DEFAULT_NODE = {"tau": 1.0, "baseline_input": 0.0, "gain": 4.0, "threshold": 0.0}


# --- parcellations ---------------------------------------------------------


def _canonical_order(ontology, structure_ids):
    """CCF drawing order (`graph_order`), id as the tie-break.

    Ordering by the ontology rather than by the file's row order means a
    Knox-built and an Oh-built network at the same resolution list their
    shared regions in the same sequence, which is what makes P3.3's
    "does the result survive changing the connectome" comparison possible.
    """
    def key(structure_id):
        structure = ontology[structure_id]
        order = structure.graph_order
        return (order is None, -1 if order is None else order, structure.id)

    return sorted({int(i) for i in structure_ids}, key=key)


def parcellation(ontology, region_ids, resolution):
    """`(coarse_ids, {fine_id: coarse_id})` for one resolution.

    `coarse_ids` is in canonical CCF order. Every fine region must land in
    exactly one coarse region; a region that cannot be placed raises, since
    a whole-brain network silently missing a division is the failure this
    package is built to avoid.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(
            "resolution must be one of %s, got %r" % (RESOLUTIONS, resolution)
        )
    ids = [int(i) for i in region_ids]

    if resolution == "regions":
        mapping = {i: i for i in ids}
    elif resolution == "major_divisions":
        mapping = {}
        for i in ids:
            division = ontology.major_division(i)
            if division is None:
                raise ValueError(
                    "region %s (%d) is outside all twelve major divisions, so "
                    "it cannot be aggregated; the connectome should not "
                    "contain it" % (ontology[i].acronym, i)
                )
            mapping[i] = division
    else:  # subdivisions
        mapping = {i: _subdivision(ontology, i) for i in ids}

    return _canonical_order(ontology, mapping.values()), mapping


def _subdivision(ontology, structure_id):
    """The CCF structure one level below this region's major division.

    'One level below' is the literal tree step: for MOp the path is
    root/grey/CH/CTX/CTXpl/Isocortex/MO/MOp, the division is Isocortex, and
    the subdivision is MO. A region that IS a direct child of its division
    (PL, CLA) is its own subdivision, and a region that IS a division (only
    possible if a dataset lists one) maps to itself.
    """
    path = list(reversed(ontology.ancestors(structure_id, include_self=True)))
    for position, ancestor in enumerate(path):
        if ancestor in MAJOR_DIVISION_IDS:
            return path[min(position + 1, len(path) - 1)]
    raise ValueError(
        "region %s (%d) is outside all twelve major divisions, so it has no "
        "subdivision" % (ontology[structure_id].acronym, structure_id)
    )


# --- the built object and its report ---------------------------------------


class BuildReport(object):
    """Everything the build decided, in a form a test can assert on.

    The point of collecting this rather than logging it is that the two
    facts most likely to be over-read - how much of the brain got a sign,
    and what the weights actually mean - are then values, not prose.
    """

    def __init__(
        self,
        dataset_key,
        citation,
        units,
        resolution,
        region_ids,
        acronyms,
        node_assignments,
        fine_region_ids,
        fine_assignments,
        signed_weight_fraction,
        excluded_weight_fraction,
        excluded_by_transmitter,
        unsigned_sources,
        self_connections,
        weight_scale,
        node_parameters,
    ):
        self.dataset_key = dataset_key
        self.citation = citation
        self.units = units
        self.resolution = resolution
        self.region_ids = list(region_ids)
        self.acronyms = list(acronyms)
        # What the sign table says about the COARSE structures themselves
        # (Isocortex -> glutamatergic, MB -> no rule). Informational only:
        # no edge was signed with these. Edges were signed per fine source
        # region, before aggregation, from `fine_assignments`.
        self.node_assignments = list(node_assignments)
        self.fine_region_ids = list(fine_region_ids)
        self.fine_assignments = list(fine_assignments)
        self.signed_weight_fraction = signed_weight_fraction
        self.excluded_weight_fraction = excluded_weight_fraction
        self.excluded_by_transmitter = dict(excluded_by_transmitter)
        self.unsigned_sources = unsigned_sources
        self.self_connections = self_connections
        self.weight_scale = weight_scale
        self.node_parameters = dict(node_parameters)

    @property
    def n_nodes(self):
        return len(self.region_ids)

    @property
    def n_fine_regions(self):
        return len(self.fine_region_ids)

    def signed_fine_regions(self):
        return [a for a in self.fine_assignments if a.is_signed]

    def unsigned_fine_regions(self):
        return [a for a in self.fine_assignments if not a.is_signed]

    def transmitter_counts(self):
        counts = {}
        for assignment in self.fine_assignments:
            key = assignment.transmitter
            counts[key] = counts.get(key, 0) + 1
        return counts

    def citations(self):
        keys = set()
        for assignment in self.fine_assignments:
            keys.update(assignment.citations)
        return sorted(keys)

    def summary(self):
        counts = self.transmitter_counts()
        lines = [
            "connectome network: %s at resolution %r"
            % (self.dataset_key, self.resolution),
            "  %d nodes aggregated from %d source regions"
            % (self.n_nodes, self.n_fine_regions),
            "  weights: %s" % self.units,
            "  weight_scale=%g  self_connections=%r  unsigned_sources=%r"
            % (self.weight_scale, self.self_connections, self.unsigned_sources),
            "  node parameters (ILLUSTRATIVE): %s"
            % ", ".join(
                "%s=%g" % (k, v) for k, v in sorted(self.node_parameters.items())
            ),
            "  sign assignment over the %d source regions:"
            % self.n_fine_regions,
        ]
        for name in sorted(counts, key=lambda k: (k is None, k)):
            label = "no rule" if name is None else name
            sign = None if name is None else nt.TRANSMITTER_SIGN[name]
            lines.append(
                "    %-14s %3d region(s)  sign=%s"
                % (label, counts[name], "none" if sign is None else "%+d" % sign)
            )
        lines.append(
            "  outgoing tracer weight signed: %.1f%%  excluded: %.1f%%"
            % (100.0 * self.signed_weight_fraction, 100.0 * self.excluded_weight_fraction)
        )
        return "\n".join(lines)


class ConnectomeCircuit(object):
    """A `CircuitNetwork` plus the provenance of every number in it."""

    def __init__(self, network, region_ids, acronyms, connectivity, report):
        self.network = network
        self.region_ids = list(region_ids)
        self.acronyms = list(acronyms)
        self.connectivity = connectivity
        self.report = report

    @property
    def n(self):
        return len(self.region_ids)

    def index_of(self, acronym):
        return self.acronyms.index(acronym)

    def weight(self, source_acronym, target_acronym):
        """Signed weight of the projection source -> target.

        Named arguments in anatomical order, because `connectivity[i, j]`
        is `j -> i` and getting that backwards is this package's standing
        hazard (see `matrix.py`).
        """
        return float(
            self.connectivity[
                self.index_of(target_acronym), self.index_of(source_acronym)
            ]
        )

    def __repr__(self):
        return "ConnectomeCircuit(%r, %d nodes)" % (
            self.report.resolution,
            self.n,
        )


# --- the builder -----------------------------------------------------------


def build_circuit_network(
    connectome,
    voxel_counts=None,
    resolution="regions",
    unsigned_sources="exclude",
    self_connections="drop_within_region",
    weight_scale=1.0,
    node_parameters=None,
    rules=None,
):
    """Build a one-hemisphere `CircuitNetwork` from a `RegionalConnectome`.

    `connectome` must carry an ontology (the loaders attach one) and, for
    any resolution other than "regions", must be the raw
    `connection_strength` matrix with `voxel_counts` from
    `volumes.target_voxel_counts`, because that is the only published
    normalization that can be aggregated (see `volumes.py`).
    """
    if unsigned_sources not in UNSIGNED_POLICIES:
        raise ValueError(
            "unsigned_sources must be one of %s, got %r"
            % (UNSIGNED_POLICIES, unsigned_sources)
        )
    if self_connections not in SELF_CONNECTION_POLICIES:
        raise ValueError(
            "self_connections must be one of %s, got %r"
            % (SELF_CONNECTION_POLICIES, self_connections)
        )
    ontology = connectome.ontology
    if ontology is None:
        raise ValueError(
            "this connectome was loaded without an ontology, so its regions "
            "cannot be named, placed in the hierarchy or signed; pass "
            "ontology= to the loader"
        )

    if voxel_counts is not None and connectome.dataset_key not in _AGGREGABLE_KEYS:
        # Dividing an already-normalized matrix by a voxel count produces a
        # number with no interpretation at all, and the units string would
        # then claim it is a connection density. Refuse instead.
        raise ValueError(
            "voxel_counts turns a raw summed weight into a connection "
            "density, so it only applies to %s; %s is already normalized by "
            "a region size"
            % (sorted(_AGGREGABLE_KEYS), connectome.dataset_key)
        )

    aggregating = resolution != "regions"
    if aggregating:
        # Order matters: "this matrix is the wrong normalization to
        # aggregate at all" is a more fundamental refusal than "you did not
        # pass voxel counts", and a caller holding an Oh et al. 2014 matrix
        # should be told the first thing rather than sent to fetch voxel
        # counts that do not exist for it.
        if connectome.dataset_key not in _AGGREGABLE_KEYS:
            raise ValueError(
                "%s cannot be aggregated: its entries are already divided by "
                "a region size, and a sum of per-size means is not the mean "
                "of the union. Aggregation needs the raw summed weight, "
                "which of the manifest's matrices is %s."
                % (connectome.dataset_key, sorted(_AGGREGABLE_KEYS))
            )
        if voxel_counts is None:
            raise ValueError(
                "resolution %r aggregates regions, which needs target voxel "
                "counts: a connection density is a per-target-voxel mean, so "
                "coarse density = (summed strength) / (summed target voxels). "
                "Pass voxel_counts=volumes.target_voxel_counts(strength, "
                "density), or build at resolution='regions'." % resolution
            )

    ids, strength = _square_ipsilateral_source_major(connectome, ontology)
    rules = nt.SignRules(ontology) if rules is None else rules
    fine_assignments = rules.assign_all(ids)

    total_weight = float(strength.sum())
    signed_mask = np.array([a.is_signed for a in fine_assignments], dtype=bool)
    if not signed_mask.any():
        raise ValueError(
            "no source region in this connectome has a sign rule; a network "
            "built from it would be empty"
        )
    if unsigned_sources == "raise" and not signed_mask.all():
        unnamed = [a.acronym for a in fine_assignments if not a.is_signed]
        raise ValueError(
            "unsigned_sources='raise' and %d of %d source regions have no "
            "sign: %s" % (len(unnamed), len(ids), ", ".join(unnamed))
        )

    signs = np.array(
        [0.0 if a.sign is None else float(a.sign) for a in fine_assignments]
    )
    signed_weight = float(strength[signed_mask].sum())
    excluded_by_transmitter = {}
    for row, assignment in enumerate(fine_assignments):
        if assignment.is_signed:
            continue
        key = assignment.transmitter
        excluded_by_transmitter[key] = excluded_by_transmitter.get(key, 0.0) + float(
            strength[row].sum()
        )
    if total_weight > 0.0:
        excluded_by_transmitter = {
            k: v / total_weight for k, v in excluded_by_transmitter.items()
        }

    if self_connections in ("drop_within_region", "drop"):
        strength = strength.copy()
        np.fill_diagonal(strength, 0.0)

    coarse_ids, mapping = parcellation(ontology, ids, resolution)
    group = _group_matrix(ids, coarse_ids, mapping)

    # Sign is applied to the SOURCE (a row of the source-major matrix)
    # before aggregation; see the module docstring.
    signed_strength = signs[:, None] * strength
    aggregated = group @ signed_strength @ group.T

    if voxel_counts is not None:
        target_voxels = _coarse_target_voxels(coarse_ids, ids, mapping, voxel_counts)
        aggregated = aggregated / target_voxels[None, :]
        units = (
            "connection density, signed: summed voxel-pair model weight from "
            "the source region, divided by the TARGET region's voxel count "
            "(Knox et al. 2019 'connection_density'), multiplied by the "
            "source region's transmitter sign (+1 glutamatergic, -1 "
            "GABAergic); dimensionless"
        )
    else:
        units = "%s, multiplied by the source region's transmitter sign" % (
            connectome.units,
        )

    if self_connections == "drop":
        np.fill_diagonal(aggregated, 0.0)

    # source-major -> CircuitNetwork's connectivity[i, j] == j onto i.
    connectivity = weight_scale * aggregated.T

    acronyms = [ontology[i].acronym for i in coarse_ids]
    parameters = dict(DEFAULT_NODE)
    if node_parameters:
        parameters.update(node_parameters)
    nodes = [CircuitNode(name=acronym, **parameters) for acronym in acronyms]
    network = CircuitNetwork(nodes=nodes, connectivity=connectivity)

    report = BuildReport(
        dataset_key=connectome.dataset_key,
        citation=connectome.citation,
        units=units,
        resolution=resolution,
        region_ids=coarse_ids,
        acronyms=acronyms,
        node_assignments=rules.assign_all(coarse_ids),
        fine_region_ids=ids,
        fine_assignments=fine_assignments,
        signed_weight_fraction=(
            0.0 if total_weight == 0.0 else signed_weight / total_weight
        ),
        excluded_weight_fraction=(
            0.0 if total_weight == 0.0 else 1.0 - signed_weight / total_weight
        ),
        excluded_by_transmitter=excluded_by_transmitter,
        unsigned_sources=unsigned_sources,
        self_connections=self_connections,
        weight_scale=weight_scale,
        node_parameters=parameters,
    )
    return ConnectomeCircuit(
        network=network,
        region_ids=coarse_ids,
        acronyms=acronyms,
        connectivity=connectivity,
        report=report,
    )


#: The manifest keys whose entries are plain summed voxel-pair weights, and
#: are therefore additive over both source and target regions.
_AGGREGABLE_KEYS = frozenset(["knox2019_connection_strength"])


def _square_ipsilateral_source_major(connectome, ontology):
    """`(ids, S)` with `S[a, b]` the weight from region `a` onto region `b`.

    Both axes are the same regions in canonical CCF order. This is
    `RegionalConnectome.square_ipsilateral()` without its final transpose:
    the builder does its aggregation source-major (so the target-voxel
    division is a column operation, matching the published normalization's
    own orientation) and transposes exactly once, at the end, into the
    `CircuitNetwork` convention.

    Reordering to canonical order here rather than at the end matters for
    one small, sharp reason: it makes the fine-region order and the node
    order identical at `resolution="regions"`, so a caller can zip
    `report.fine_assignments` against `acronyms` without checking whether
    the file happened to be written in ontology order.
    """
    ipsi = connectome.hemisphere("ipsi")
    source_ids = [int(i) for i in ipsi.source_ids]
    order = {int(i): k for k, i in enumerate(ipsi.target_ids)}
    if set(order) != set(source_ids):
        raise ValueError(
            "%s: the ipsilateral target set differs from the source set, so "
            "there is no square block to build a network from"
            % connectome.dataset_key
        )
    ids = _canonical_order(ontology, source_ids)
    row_of = {i: k for k, i in enumerate(source_ids)}
    rows = [row_of[i] for i in ids]
    columns = [order[i] for i in ids]
    return ids, ipsi.weights[np.ix_(rows, columns)]


def _group_matrix(fine_ids, coarse_ids, mapping):
    """0/1 indicator, `G[coarse, fine]`."""
    position = {i: k for k, i in enumerate(coarse_ids)}
    group = np.zeros((len(coarse_ids), len(fine_ids)))
    for column, fine_id in enumerate(fine_ids):
        group[position[mapping[fine_id]], column] = 1.0
    return group


def _coarse_target_voxels(coarse_ids, fine_ids, mapping, voxel_counts):
    """Voxel count of each coarse region: the sum over its fine members.

    Deliberately the sum over the members *present in this dataset*, not
    the CCF volume of the coarse structure, which would include voxels the
    connectome has no rows for and would silently shrink every weight.
    """
    totals = dict((i, 0.0) for i in coarse_ids)
    for fine_id in fine_ids:
        key = ("ipsi", int(fine_id))
        if key not in voxel_counts:
            raise KeyError(
                "no ipsilateral voxel count for structure %d; the voxel "
                "counts and the connectome are not over the same regions"
                % fine_id
            )
        totals[mapping[fine_id]] += voxel_counts[key]
    missing = [i for i in coarse_ids if totals[i] <= 0.0]
    if missing:
        raise ValueError("coarse regions with zero voxels: %s" % missing)
    return np.array([totals[i] for i in coarse_ids])


# --- convenience loaders ---------------------------------------------------


def load_knox_inputs(cache_dir=None, ontology=None, allow_download=False):
    """`(strength, voxel_counts)` for the Knox et al. 2019 matrices.

    Reads two of the manifest's four Knox files from the verified cache and
    recovers the voxel counts from their ratio (`volumes.py`). No new
    source, no new number.
    """
    from .matrix import load_connectome
    from .ontology import load_ontology
    from .volumes import target_voxel_counts

    if ontology is None:
        ontology = load_ontology(cache_dir=cache_dir, allow_download=allow_download)
    strength = load_connectome(
        key="knox2019_connection_strength",
        ontology=ontology,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    density = load_connectome(
        key="knox2019_connection_density",
        ontology=ontology,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    return strength, target_voxel_counts(strength, density)
