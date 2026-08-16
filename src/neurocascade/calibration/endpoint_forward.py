"""P7.3: the forward model from a brain concentration to a P7.1 endpoint.

What this is, and what `circuit_forward.py` next to it is not
-------------------------------------------------------------

`circuit_forward.py` (P2.2) answers "given one receptor occupancy and one
circuit-layer gain, what does a THREE-NODE hand-built scenario's readout
do?". Its networks are populations (`VTA_GABA`, `CeA`), so no P7.1 endpoint
can be read on them: `endpoints.Endpoint.resolve` maps CCF acronyms onto
node names, and a hand-built scenario has none.

This module answers the same question on the WHOLE-BRAIN network, whose
nodes ARE CCF acronyms (`connectome/build.py` names every node
`ontology[i].acronym`), so all seven P7.1 endpoints resolve on it with
nothing unmapped. That is what makes a P7.3 fit a fit of the ENDPOINT
rather than of a scenario that happens to share a region with it.

Nothing here is reimplemented. The network is P3.2's build, the operating
point is P3.4's `stability.stabilize`, the densities are P4.2's
`relative_density`, the steady state is P2.2's
`circuit_forward.steady_state_rates` (which is network-agnostic and was
only ever exercised at 3-5 nodes), and the readout is P7.1's
`ResolvedEndpoint`.

The chain, and the one place a number is fitted
-----------------------------------------------

    dose (mg/kg, route, species)
        --Phase 1 bridge (drug_pk + brain_partition)-->
    unbound brain concentration C (nM)
        --ReceptorSite.occupancy_rhs at the calibrated Ki-->
    activation a in [0, 1]                      (one scalar, whole brain)
        --coupling[region] = mechanism_gain * rho[region]  (P4.2)-->
    per-region drive (or per-region synaptic modulation)
        --CircuitNetwork.rhs to steady state-->
    291 regional rates
        --P7.1 ResolvedEndpoint, equal illustrative magnitudes-->
    one endpoint index

`mechanism_gain` is the ONLY free parameter in that chain, and it is the
one P7.3 fits. Everything upstream of it is calibrated (PK, partition, Ki)
or measured (rho); everything downstream is P7.1's cited membership and
sign with an illustrative equal magnitude. See
`docs/architecture.md`'s P7.3 section for why the magnitudes cannot be
fitted from this data and therefore keep their label.

The reduction that makes an MCMC fit affordable
-----------------------------------------------

The gain and the activation enter the circuit ONLY as their product. For
the drive mechanism,

    drive_i = baseline_i + mechanism_gain * rho_i * activation

and for the modulation mechanism,

    W_ij <- W_ij * (1 + mechanism_gain * rho_i * activation)   (W_ij < 0)

so in both cases the steady state is a function of the single scalar

    x = mechanism_gain * activation

This is exact rather than approximate, and
`tests/test_endpoint_forward.py::test_only_the_product_of_gain_and_
activation_matters` pins it by evaluating three (gain, activation) pairs
with the same product and requiring identical rates. It matters because
one 291-node steady state costs seconds: a `ResponseCurve` over a grid of
x is computed once and interpolated inside the sampler, instead of
integrating 291 coupled ODEs per MCMC draw. The interpolation error is
MEASURED against held-out grid points rather than assumed small
(`ResponseCurve.interpolation_error`).

The index, and the zero it needs
--------------------------------

P2.2 compared a drug/vehicle RATIO, which needs the model's readout and
the assay's measure to be proportional with a common zero. Its readout was
one node's firing rate, where zero is a meaningful state.

A P7.1 endpoint score is not that. It is `sum_i w_i r_i` with signs on both
sides, `sum_i |w_i| = 1`, so at the drug-free rest state (P3.4 balances
every node to r = 0.5) it is `0.5 * sum_i w_i`, which is +0.5 for working
memory, -0.375 for sedation, and +0.1 for analgesia. A ratio of a quantity
whose zero is an accident of which regions happened to be cited is not a
quantity any assay can be compared with.

`directed_activity_index` is this module's answer:

    A = sum_{w_i > 0} w_i r_i  +  sum_{w_i < 0} |w_i| (1 - r_i)
      = score + sum_{w_i < 0} |w_i|

i.e. the fraction of the endpoint's cited evidence that is currently
pointing in the direction that INCREASES the behaviour. Three properties,
each of which is why it is this and not the raw score:

* It is bounded in [0, 1] with a zero that means something in every
  endpoint: every pro-behaviour region silent AND every anti-behaviour
  region saturated.
* At the drug-free state it is EXACTLY 0.5 for all seven endpoints,
  because `sum_i |w_i| = 1` and `r = 0.5`. So no endpoint's drug/vehicle
  ratio is conditioned on a near-zero denominator, which the raw score's
  +0.1 for analgesia would have been.
* It is an affine function of the score with a fixed offset, so it
  reorders nothing and adds no free parameter.

It is still an assumption - it assumes a linear readout has its zero where
all the evidence points the wrong way - and it is declared here, on
`directed_activity_index`, and in docs/architecture.md rather than buried.

What is illustrative in this path
---------------------------------

Everything except the receptor layer, and P7.3 removes the label from
none of it except the gain it actually fits:

* `connectome.DEFAULT_NODE` (tau 1.0, gain 4.0, threshold 0.0) - P3.2.
* `stability.DEFAULT_OPERATING_GAIN = 0.9`, which sets `weight_scale` and
  the balanced baselines - P3.4.
* P4.2's `density = k * expression_energy` proportionality.
* Every P7.1 endpoint weight MAGNITUDE.

So an index this module returns is a shape, not a magnitude, exactly as
P4.2 says of a density map. What a fit against it can identify is
discussed in `docs/architecture.md`'s P7.3 section, and the answer is
"one scalar per receptor system, conditional on all of the above".
"""

import numpy as np

from .. import connectome as _connectome
from .. import endpoints as _endpoints
from .. import expression as _expression
from .. import stability as _stability
from ..connectome import build as _build
from . import circuit_forward as cfwd

#: How the whole-brain density map is normalised. `"max"` puts the
#: peak-expressing region at rho = 1, so `mechanism_gain` reads as "the
#: action a fully occupied receptor pool has on the region with the most of
#: it", and every other region is a measured fraction of that.
#:
#: P4.2 warns that a `max`-normalised map depends on which regions are in
#: the matrix, which is why `scenario_maps.py` uses `reference-region` for
#: a three-node scenario. Here the region set IS the whole brain at a named
#: resolution, so the maximum is over everything the atlas measures and is
#: not an artefact of a scenario's membership; `reference-region` would
#: instead require picking one region out of 291, which is the arbitrary
#: choice this avoids. `tests/test_endpoint_forward.py` pins the peak
#: region of each gene so a silent change in the atlas cannot move the
#: meaning of the gain without failing a test.
DENSITY_NORMALIZATION = "max"

#: The ISH plane/combination policy, matching `expression/scenario_maps.py`
#: so that a whole-brain map and a scenario map are the same measurement
#: read at different scales.
PLANE = "coronal-else-sagittal"
COMBINE = "mean"

#: How a receptor system acts on the circuit. Both are mechanisms
#: `engine.Simulation` already implements; this module reproduces them at
#: whole-brain scale through `circuit_forward`, which implements both.
#:
#: ``"drive_coupling"``
#:     The receptor adds a drive to the region it sits in, scaled by that
#:     region's density. The mu-opioid scenarios' mechanism.
#: ``"connection_modulation"``
#:     The receptor scales the strength of the inhibitory synapses in the
#:     region it sits in. The benzodiazepine mechanism: diazepam does not
#:     drive a cell, it potentiates GABA-A currents at synapses that are
#:     already there, which is why `scenarios.gaba_a_coupling()` is empty
#:     and `gaba_a_connection_modulation()` is not.
MECHANISMS = ("drive_coupling", "connection_modulation")

#: The drug-free value of `directed_activity_index` for every endpoint on a
#: P3.4-stabilised network, exactly. See the module docstring.
BASELINE_INDEX = 0.5


def directed_activity_index(resolved, rates_by_node):
    """The endpoint's cited evidence, as the fraction currently pointing
    towards MORE of the behaviour. In [0, 1]; see the module docstring for
    why the raw score cannot be used and what the zero assumes.

    `resolved` is a `ResolvedEndpoint`; `rates_by_node` maps node name to
    rate. Only the endpoint's own nodes are read.
    """
    total = 0.0
    for node, weight in resolved.readout.weights.items():
        rate = float(rates_by_node[node])
        total += weight * rate if weight > 0 else abs(weight) * (1.0 - rate)
    return total


class WholeBrainDrugModel(object):
    """One receptor system acting on one whole-brain network.

    Parameters
    ----------
    network : CircuitNetwork
        Nodes named by CCF acronym, already at a usable operating point
        (`stability.stabilize`). This class does NOT stabilise for you: the
        operating point is P3.4's decision and carries P3.4's illustrative
        operating gain, so it is made by the caller and visible in the
        caller's code.
    ontology : connectome.StructureOntology
    densities : {node acronym: rho}
        P4.2 relative densities. Nodes absent from this mapping receive no
        drug action, and `unmeasured_nodes` must name them: a region the
        ISH data does not reach is not a region where the drug has no
        target, and `density.as_coupling` raises rather than let those two
        be confused. This class takes the same position and requires the
        caller to have made the exclusion explicitly.
    mechanism : one of `MECHANISMS`
    unmeasured_nodes : tuple of node names
        The regions excluded above, carried so that a report can say which
        they were and whether any endpoint reads one.
    region_sizes : {acronym: voxels} or None
        Passed to `Endpoint.resolve` for the descendant-split rule.
    """

    def __init__(self, network, ontology, densities, mechanism, name,
                 unmeasured_nodes=(), region_sizes=None):
        if mechanism not in MECHANISMS:
            raise ValueError(
                "mechanism must be one of %s; got %r"
                % (", ".join(MECHANISMS), mechanism)
            )
        node_names = [node.name for node in network.nodes]
        unknown = sorted(set(densities) - set(node_names))
        if unknown:
            raise ValueError(
                "densities name nodes the network does not have: %s"
                % ", ".join(unknown)
            )
        covered = set(densities) | set(unmeasured_nodes)
        uncovered = sorted(set(node_names) - covered)
        if uncovered:
            raise ValueError(
                "%d node(s) have neither a density nor a place in "
                "unmeasured_nodes: %s. A region with no measurement has to "
                "be named, not dropped." % (len(uncovered), ", ".join(uncovered))
            )
        if not np.all(np.isfinite(list(densities.values()))):
            raise ValueError("densities must all be finite")

        self.network = network
        self.ontology = ontology
        self.densities = dict(densities)
        self.mechanism = mechanism
        self.name = str(name)
        self.unmeasured_nodes = tuple(unmeasured_nodes)
        self.region_sizes = region_sizes
        self._baseline_rates = None
        self._resolved = {}
        self._negative_edges = None

    # --- the drug's action on the circuit ---------------------------------

    @property
    def baseline_rates(self):
        """The drug-free steady state, integrated once and cached.

        Cacheable for the same reason P2.2's baseline is: at zero drug the
        drive contribution is `x * rho * 0` and every modulation factor is
        `1 + x * rho * 0`, so the drug-free leg cannot depend on anything
        this module fits.
        """
        if self._baseline_rates is None:
            self._baseline_rates = cfwd.steady_state_rates(self.network)
        return self._baseline_rates

    def _connection_modulation(self):
        """`{(post, pre): rho_post}` over every INHIBITORY edge.

        The receptor sits on the postsynaptic region, so the density that
        scales a synapse is the density of the region receiving it. Only
        negative (inhibitory) edges are modulated, because a benzodiazepine
        potentiates GABA-A currents and there are no GABA-A currents at an
        excitatory synapse; `connectome/neurotransmitter.py` is what makes
        the sign of an edge a statement about its transmitter.
        """
        if self._negative_edges is None:
            names = [node.name for node in self.network.nodes]
            W = np.asarray(self.network.connectivity, dtype=float)
            rows, cols = np.nonzero(W < 0.0)
            edges = {}
            for i, j in zip(rows, cols):
                rho = self.densities.get(names[i])
                if rho is None:
                    continue
                edges[(names[i], names[j])] = float(rho)
            self._negative_edges = edges
        return self._negative_edges

    def rates(self, x, warm_start=True, method=None, use_sparse_jacobian=True,
              solver=None):
        """Steady-state rates at `x = mechanism_gain * activation`.

        Returns an array in the network's node order. `x = 0` returns the
        drug-free state (asserted by a test, not by this docstring).
        """
        x = float(x)
        initial = self.baseline_rates if warm_start else None
        if self.mechanism == "drive_coupling":
            return cfwd.steady_state_rates(
                self.network,
                coupling={node: x * rho for node, rho in self.densities.items()},
                activation=1.0,
                initial_rates=initial,
                method="RK45" if method is None else method,
                use_sparse_jacobian=use_sparse_jacobian,
                solver="integration" if solver is None else solver,
            )
        return cfwd.steady_state_rates(
            self.network,
            connection_modulation={
                edge: x * rho for edge, rho in self._connection_modulation().items()
            },
            activation=1.0,
            initial_rates=initial,
            method="RK45" if method is None else method,
            use_sparse_jacobian=use_sparse_jacobian,
            solver="integration" if solver is None else solver,
        )

    # --- the readout -------------------------------------------------------

    def resolved_endpoint(self, endpoint_key):
        """P7.1's `ResolvedEndpoint` for one endpoint on this network."""
        if endpoint_key not in self._resolved:
            endpoint = _endpoints.ENDPOINTS[endpoint_key]
            self._resolved[endpoint_key] = endpoint.resolve(
                self.network, self.ontology, region_sizes=self.region_sizes
            )
        return self._resolved[endpoint_key]

    def index(self, endpoint_key, rates):
        """`directed_activity_index` of one endpoint at `rates`."""
        names = [node.name for node in self.network.nodes]
        return directed_activity_index(
            self.resolved_endpoint(endpoint_key), dict(zip(names, rates))
        )

    def endpoint_regions_without_density(self, endpoint_key):
        """The endpoint's own nodes that are in `unmeasured_nodes`.

        The number that decides whether excluding the unmeasured regions
        touches this endpoint directly or only through the network. Empty
        is the answer P7.3 reports for all seven, and it is computed here
        rather than claimed in prose.
        """
        unmeasured = set(self.unmeasured_nodes)
        return tuple(
            node
            for node in sorted(self.resolved_endpoint(endpoint_key).readout.weights)
            if node in unmeasured
        )

    def response_curve(self, endpoint_keys, x_grid):
        """A `ResponseCurve` over `x_grid` for each endpoint in one sweep.

        One steady state per grid point, shared by every endpoint, because
        the rates do not depend on which readout is applied to them.
        """
        x_grid = np.asarray(sorted(float(x) for x in x_grid), dtype=float)
        values = {key: [] for key in endpoint_keys}
        for x in x_grid:
            rates = self.rates(x)
            for key in endpoint_keys:
                values[key].append(self.index(key, rates))
        return ResponseCurve(
            model_name=self.name,
            x_grid=x_grid,
            values={k: np.asarray(v, dtype=float) for k, v in values.items()},
        )


class ResponseCurve(object):
    """The endpoint index as a function of `x = mechanism_gain * activation`.

    Exists so an MCMC sampler can evaluate the forward model without
    integrating a 291-node ODE per draw. That is a numerical shortcut and
    it is treated as one: `interpolation_error` re-integrates at points
    BETWEEN the grid points and reports the largest disagreement, so the
    error is a measured number in the results file rather than an
    assumption in a docstring.
    """

    def __init__(self, model_name, x_grid, values):
        self.model_name = str(model_name)
        self.x_grid = np.asarray(x_grid, dtype=float)
        if np.any(np.diff(self.x_grid) <= 0):
            raise ValueError("x_grid must be strictly increasing")
        self.values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        for key, v in self.values.items():
            if v.shape != self.x_grid.shape:
                raise ValueError("%s: %r values for %r grid points"
                                 % (key, v.shape, self.x_grid.shape))

    @property
    def endpoint_keys(self):
        return tuple(sorted(self.values))

    def __call__(self, endpoint_key, x):
        """Piecewise-linear interpolation of the index at `x`.

        Linear rather than cubic on purpose: a monotone spline would be
        smoother, and a cubic one can OVERSHOOT between grid points, which
        would let the interpolant leave [0, 1] near saturation and put a
        value the circuit cannot produce into a likelihood. Refining the
        grid is the honest way to reduce this error, and
        `interpolation_error` is what says whether it needs refining.
        """
        x = np.asarray(x, dtype=float)
        if np.any(x < self.x_grid[0]) or np.any(x > self.x_grid[-1]):
            raise ValueError(
                "x outside the tabulated range [%g, %g]; extrapolating a "
                "circuit response is not something this curve may do"
                % (self.x_grid[0], self.x_grid[-1])
            )
        return np.interp(x, self.x_grid, self.values[endpoint_key])

    def interpolation_error(self, model, endpoint_keys=None):
        """`{endpoint: max |interpolated - integrated|}` at grid midpoints."""
        keys = tuple(endpoint_keys or self.endpoint_keys)
        midpoints = 0.5 * (self.x_grid[:-1] + self.x_grid[1:])
        worst = {key: 0.0 for key in keys}
        for x in midpoints:
            rates = model.rates(x)
            for key in keys:
                worst[key] = max(
                    worst[key], abs(model.index(key, rates) - float(self(key, x)))
                )
        return worst


def whole_brain_inputs(allow_download=False):
    """`(ontology, strength, voxel_counts, atlas_source, description)`.

    The full published Knox matrices when the gitignored `data/connectome`
    cache is present, and the committed fifteen-region fixtures otherwise -
    the same cache-else-fixture rule `scripts/report_endpoints.py` and
    `examples/behavioral_endpoint_library.py` already use, so a clean
    checkout runs everything at fixture scale and says so.
    """
    from ..connectome import fetch, sources

    cached = (
        fetch.is_cached("knox2019_connection_strength")
        and fetch.is_cached("knox2019_connection_density")
        and fetch.is_cached(sources.ONTOLOGY_KEY)
    )
    if cached:
        ontology = _connectome.load_ontology()
        strength, voxels = _connectome.load_knox_inputs(
            ontology=ontology, allow_download=allow_download
        )
        return (ontology, strength, voxels, "cache",
                "the full published Knox et al. 2019 matrices")
    ontology = _connectome.load_fixture_ontology()
    strength, voxels = _connectome.load_fixture_build_inputs(ontology=ontology)
    return (ontology, strength, voxels, "fixture",
            "the committed fifteen-region fixtures")


def node_voxel_counts(ontology, circuit, voxel_counts):
    """`{node acronym: voxel count}`, summed over the fine regions in it.

    The `region_sizes` argument `Endpoint.resolve` uses for its
    descendant-split rule. Lifted verbatim in behaviour from
    `scripts/report_endpoints.py`, which needs the same thing.
    """
    report = circuit.report
    _, mapping = _build.parcellation(
        ontology, report.fine_region_ids, report.resolution
    )
    totals = {i: 0.0 for i in report.region_ids}
    for fine in report.fine_region_ids:
        totals[mapping[fine]] += float(voxel_counts[("ipsi", int(fine))])
    return {ontology[i].acronym: totals[i] for i in report.region_ids}


def build_whole_brain_model(gene, mechanism, name, resolution="regions",
                            inputs=None, atlas=None, allow_download=False):
    """Assemble a `WholeBrainDrugModel` for one receptor gene.

    Returns `(model, provenance)`, where `provenance` records the four
    choices that are not this module's to make silently: which matrices
    were loaded, the P3.4 operating gain, the density normalisation, and
    which regions the atlas does not measure.

    The network is stabilised here (P3.4) because an unstabilised
    whole-brain network has 218 of 291 nodes pinned at the ceiling of their
    sigmoid (`results/whole_brain_stability.txt`), where a drug changes
    nothing and a fit would be measuring saturation rather than
    pharmacology.
    """
    if inputs is None:
        ontology, strength, voxels, source, description = whole_brain_inputs(
            allow_download=allow_download
        )
    else:
        ontology, strength, voxels = inputs
        source, description = "caller", "inputs supplied by the caller"

    circuit = _build.build_circuit_network(
        strength, voxel_counts=voxels, resolution=resolution
    )
    stabilized = _stability.stabilize(circuit.network)

    if atlas is None:
        if source == "fixture":
            atlas = _expression.load_fixture_atlas(symbols=[gene], ontology=ontology)
        else:
            atlas = _expression.load_expression_atlas(
                symbols=[gene], ontology=ontology
            )
    energy = atlas.energy_matrix(
        circuit.region_ids, symbols=[gene], plane=PLANE, combine=COMBINE
    )
    density_map = _expression.relative_density(
        energy, gene, normalization=DENSITY_NORMALIZATION
    )
    values = np.asarray(density_map.values, dtype=float).ravel()

    densities, unmeasured = {}, []
    for acronym, value in zip(circuit.acronyms, values):
        if np.isfinite(value):
            densities[acronym] = float(value)
        else:
            unmeasured.append(acronym)

    model = WholeBrainDrugModel(
        network=stabilized.network,
        ontology=ontology,
        densities=densities,
        mechanism=mechanism,
        name=name,
        unmeasured_nodes=tuple(unmeasured),
        region_sizes=node_voxel_counts(ontology, circuit, voxels),
    )
    provenance = {
        "gene": gene,
        "mechanism": mechanism,
        "resolution": resolution,
        "n_nodes": circuit.n,
        "connectome_source": source,
        "connectome_description": description,
        "operating_gain": _stability.DEFAULT_OPERATING_GAIN,
        "weight_scale": stabilized.weight_scale,
        "density_normalization": DENSITY_NORMALIZATION,
        "density_plane": PLANE,
        "density_combine": COMBINE,
        "unmeasured_nodes": tuple(unmeasured),
        "peak_density_node": max(densities, key=densities.get) if densities else None,
    }
    return model, provenance


__all__ = [
    "MECHANISMS",
    "DENSITY_NORMALIZATION",
    "build_whole_brain_model",
    "node_voxel_counts",
    "whole_brain_inputs",
    "BASELINE_INDEX",
    "directed_activity_index",
    "WholeBrainDrugModel",
    "ResponseCurve",
]
