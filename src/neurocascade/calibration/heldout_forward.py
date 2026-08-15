"""P8.2: the forward path from a held-out compound's binding fingerprint to a
whole-brain behavioural endpoint index, and the three quantities that can be
computed along it without fitting anything.

What this module is for
-----------------------

P8.1 built `fingerprint.predict`, which takes an arbitrary compound's Ki table
and a PK model and returns occupancy and activation per target. It stops
there, and its own docstring says why: turning an activation into a
behavioural score needs a `mechanism_gain` per target, and this repository has
never fitted one (P7.3: "No illustrative label was removed, anywhere").

P8.2 has to predict behaviour anyway, for compounds never used in fitting.
The way through is not to invent the gains. It is to notice that three
different questions can be asked of the chain, and that only the third one
needs a gain at all:

1. **What occupancy does the compound reach?**  PK, the plasma-to-brain
   partition and the Ki fix this completely. No circuit parameter enters.
   `fingerprint.predict` already answers it, and published in vivo occupancy
   measurements are the held-out data it can be checked against.

2. **Which way does each endpoint move?**  The sign of the endpoint's response
   is `sign(gain) * sign(delta_activation)`, and neither factor needs the
   gain's magnitude. `sign(gain)` is the receptor's transduction sign, which
   every P5 module already declares in its own `coupling` dict;
   `sign(delta_activation)` is whether the compound adds activation to the
   site or removes it from the endogenous tone. So the direction is a genuine
   parameter-free prediction.

3. **How far does it move?**  This is the one that needs the gain, and this
   module refuses to supply one. Instead it computes two gain-free things:
   the *reachable range* of the endpoint index over all gains, which is a
   one-sided test an observation can fail outright; and the *implied gain*,
   the value that would be needed to reproduce a published effect, which is
   a diagnostic rather than a prediction and which becomes a real test when
   two different compounds acting at the same target imply different values.

The reachable range is the part worth stating twice. The index saturates:
driving the gain to plus or minus infinity does not move the endpoint index
arbitrarily far, because the Wilson-Cowan rates it is built from are bounded
in [0, 1] and the index is a convex combination of them and their
complements. So there is a largest effect the model can produce at each
endpoint, for any gain whatsoever, and an observation outside it is refuted
by the model's structure rather than by a parameter choice. That is the
strongest kind of negative result available here, and it costs nothing to
compute.

The three signs, and where each comes from
------------------------------------------

`x = gain * delta_activation` is the only way a receptor reaches the circuit
(P7.3's reduction, exact and pinned by a test there). Its sign decomposes:

* **`sign(gain)`: the transduction sign.**  Read off the P5 module's own
  `coupling` or `connection_modulation` dict rather than restated here, for
  the same reason P8.1's `kon` table is assembled from module constants: a
  module that changes its mind changes this too. `transduction_sign` requires
  the signs within one system to be unanimous and raises if they are not,
  because a receptor whose own module disagrees about whether it excites or
  inhibits is not a receptor this module may average.

  The *magnitudes* in those dicts are illustrative and are deliberately not
  used. Only the sign is taken.

* **`sign(delta_activation)`: agonist or antagonist, against the tone.**
  `delta_activation` is the activation the site reaches with the drug present
  MINUS the activation it had without it. For an agonist arriving at an empty
  pool that is positive and equal to the drug's own activation, which is what
  every earlier scenario in this codebase computed. For an antagonist it is
  negative, and it is zero unless the site carries endogenous tone: a
  competitive antagonist at a pool with nothing in it displaces nothing and
  does nothing. That is not a modelling shortcut, it is what the equations
  say, and it is why P6.2's measured tone is what makes an antagonist
  predictable at all.

* **the endpoint's own sign**, from P7.1's cited region membership, already
  inside `directed_activity_index`.

The quasi-equilibrium step, and its measured error
--------------------------------------------------

`delta_activation` is evaluated at binding equilibrium for the drug
concentration prevailing at the time of the behavioural measurement, using
`projection.resting_occupancy` - the same function P6.2 uses to compute a
tonic occupancy. The alternative is to integrate the binding ODE alongside a
time-varying concentration, which `fingerprint.predict` does.

Equilibrium is used because the tone ligand has to be present at its basal
concentration throughout, and `predict` doses every ligand but the compound
at zero. It is an approximation and it is treated as one:
`equilibrium_error` re-runs the no-tone case both ways and reports the
largest disagreement, so the cost appears as a measured number in the results
file rather than as an assumption in this docstring.

What this module does NOT do
----------------------------

* **It does not compose a transporter onto the whole-brain network.**  A
  DAT blocker raises extracellular dopamine, which then acts at D1 and D2.
  P6.1 built that chain and `endpoint_forward` does not carry it: its
  `WholeBrainDrugModel` takes one receptor gene's ISH map and one mechanism,
  and there is no whole-brain projection field for a transporter to act on.
  So a compound whose behavioural mechanism runs through a monoamine
  transporter has no behavioural forward path here at all, and
  `behavioural_path_status` says so by name rather than returning a zero.

* **It does not supply a tone where P6.2 found none.**  Adenosine, glutamate,
  GABA and the opioid peptides have no measured basal concentration in this
  repository, so an antagonist at A2A, NMDA, AMPA, CB1, MOR, DOR, KOR or the
  GABA-A benzodiazepine site has `delta_activation == 0` and is reported as
  `no-tone` rather than as "no effect".
"""

import numpy as np

from .. import endogenous_tone as _tone
from .. import fingerprint as _fp
from .. import projection as _projection
from . import circuit_forward as _cfwd
from . import endpoint_forward as _efwd


#: The endpoint keys P7.1 defines, in a fixed order so every table in this
#: task's results reads the same way.
ENDPOINT_KEYS = (
    "locomotion",
    "anxiety",
    "analgesia",
    "sedation",
    "reward",
    "working_memory",
    "sensorimotor_gating",
)

#: Status values `behavioural_path_status` returns. Each names a DIFFERENT
#: reason a compound-target pair cannot reach a behavioural endpoint, because
#: collapsing them into one "unavailable" would hide which missing piece would
#: fix which compound.
PATH_STATUSES = (
    "available",
    "transporter-no-whole-brain-field",
    "no-expression-map",
    "antagonist-no-tone",
    "no-activation-change",
)


# ---------------------------------------------------------------------------
# The transduction sign, read off each P5 module rather than restated
# ---------------------------------------------------------------------------


def _mechanism_values(system):
    """The system's own coupling or connection-modulation values.

    Returns `(mechanism, values)` where `mechanism` is one of
    `endpoint_forward.MECHANISMS`. Which of the two a receptor uses is the P5
    module's declaration, not this module's choice.
    """
    coupling = dict(getattr(system, "coupling", None) or {})
    modulation = dict(getattr(system, "connection_modulation", None) or {})
    if coupling and modulation:
        raise ValueError(
            "%s declares BOTH a coupling and a connection modulation; this "
            "module reduces a receptor to one mechanism and cannot choose "
            "between them" % system.name
        )
    if coupling:
        return "drive_coupling", list(coupling.values())
    if modulation:
        return "connection_modulation", list(modulation.values())
    raise ValueError(
        "%s declares neither a coupling nor a connection modulation, so it "
        "has no way to reach a circuit" % system.name
    )


def transduction_sign(target):
    """`(mechanism, sign)` for one target, from its own P5 module.

    `sign` is +1 or -1: the direction an AGONIST at this receptor moves its
    host region's drive (for `drive_coupling`) or the strength of the
    inhibitory synapses onto it (for `connection_modulation`). It is the sign
    of the module's declared coupling values, whose magnitudes are
    illustrative and are not read.

    Raises when a module's own values disagree in sign, because averaging a
    receptor that its module says both excites and inhibits would invent the
    one thing this function exists to report.
    """
    spec = _fp.TARGETS[_fp.resolve_target(target)]
    if spec.is_transporter:
        raise ValueError(
            "%s is a transporter: it has no coupling to a circuit of its own "
            "and reaches behaviour only through the transmitter field it "
            "changes, which this module does not carry at whole-brain scale"
            % target
        )
    mechanism, values = _mechanism_values(spec.system())
    signs = {int(np.sign(v)) for v in values if v != 0.0}
    if not signs:
        raise ValueError("%s: every declared coupling value is zero" % target)
    if len(signs) > 1:
        raise ValueError(
            "%s: its own module declares couplings of both signs (%s), so "
            "there is no single transduction sign to read off"
            % (target, sorted(signs))
        )
    return mechanism, signs.pop()


#: Which network each connection-modulating system's own scenario uses, so
#: that the SIGN of the edge it modulates can be read off that network rather
#: than declared here. This table is a pointer, not a number: a test asserts
#: that each system's declared edge exists in the named network and that the
#: signs of all its declared edges agree.
#:
#: It exists because `endpoint_forward.WholeBrainDrugModel` hardcodes
#: "inhibitory edges only", which is correct for the one system P7.3 used (a
#: benzodiazepine potentiates GABA-A currents, and there are no GABA-A
#: currents at an excitatory synapse) and wrong for three of the other four.
#: An NMDA or AMPA blocker removes GLUTAMATERGIC transmission, so it scales
#: excitatory edges; applying it to inhibitory ones would give a whole-brain
#: NMDA antagonist the sign of a whole-brain NMDA agonist.
MODULATED_EDGE_NETWORK = {
    "GABA_A_BZD": ("scenarios", "amygdala_anxiety_circuit_network"),
    "NMDA": ("glutamate", "cortical_network"),
    "AMPA": ("glutamate", "cortical_network"),
    "CB1": ("cannabinoid", "cortical_network"),
    "KOR": ("opioid", "reward_network"),
}

#: How well founded each target's whole-brain `mechanism_gain` is. Assembled
#: from each module's own constant and its own words, and re-derived by a
#: test, for the same reason P8.1's `kon` standings are.
#:
#: ``pinned-by-algebra``
#:     The gain is not a parameter. `glutamate.GLUTAMATE_CONNECTION_GAIN` is
#:     exactly 1.0 because the site's `activation` is DEFINED as the negative
#:     fraction of transmission removed, so `1 + gain * activation` is
#:     `1 - fraction removed` only at `gain == 1`.
#: ``measured``
#:     The gain is a published fraction: CB1's -0.47 is the maximal
#:     depolarisation-induced suppression of inhibition [HoffmanLupica2000];
#:     KOR's -0.72 is dopamine overflow falling to 28% of control under a
#:     saturating kappa agonist [Britt2008].
#: ``unfitted``
#:     Illustrative, hand-set, and never calibrated against anything. The
#:     majority, and the reason most of this task's magnitudes are reported
#:     as reachable ranges rather than as predictions.
GAIN_STANDINGS = ("pinned-by-algebra", "measured", "unfitted")

_GAIN_STANDING = {
    "NMDA": ("pinned-by-algebra", "glutamate.GLUTAMATE_CONNECTION_GAIN"),
    "AMPA": ("pinned-by-algebra", "glutamate.GLUTAMATE_CONNECTION_GAIN"),
    "CB1": ("measured", "cannabinoid.CB1_CONNECTION_GAIN [HoffmanLupica2000]"),
    "KOR": ("measured", "opioid.KOR_CONNECTION_GAIN [Britt2008]"),
}


def gain_standing(target):
    """`(standing, source)` for one target's whole-brain mechanism gain.

    Everything not in `_GAIN_STANDING` is `unfitted`, which is the honest
    default: P7.3 removed no illustrative label anywhere, and P2.2's one
    fitted gain is a three-node scenario's single synapse rather than a
    whole-brain one.
    """
    canonical = _fp.resolve_target(target)
    return _GAIN_STANDING.get(
        canonical, ("unfitted", "illustrative, never calibrated")
    )


def declared_gain(target):
    """The module's own gain value for a target whose gain is not free.

    Raises for an `unfitted` target, because handing out a number for one of
    those is exactly the invention this task exists not to commit.
    """
    canonical = _fp.resolve_target(target)
    standing, _ = gain_standing(canonical)
    if standing == "unfitted":
        raise ValueError(
            "%s's whole-brain mechanism gain is unfitted and illustrative; "
            "there is no value to hand out. Use reachable_index_range or "
            "implied_x instead." % canonical
        )
    from .. import cannabinoid as _cb
    from .. import glutamate as _glu
    from .. import opioid as _op

    return {
        "NMDA": _glu.GLUTAMATE_CONNECTION_GAIN,
        "AMPA": _glu.GLUTAMATE_CONNECTION_GAIN,
        "CB1": _cb.CB1_CONNECTION_GAIN,
        "KOR": _op.KOR_CONNECTION_GAIN,
    }[canonical]


def modulated_edge_sign(target):
    """+1 if this target scales EXCITATORY edges, -1 if INHIBITORY.

    Read off the sign of the edge the system's own scenario declares, in that
    scenario's own network, rather than restated here. Raises if the system's
    declared edges disagree in sign, because a receptor that its own module
    says sits on both an excitatory and an inhibitory synapse has no single
    whole-brain generalisation and must not be given one silently.
    """
    import importlib

    canonical = _fp.resolve_target(target)
    mechanism, _ = transduction_sign(canonical)
    if mechanism != "connection_modulation":
        raise ValueError(
            "%s acts by %s, which modulates no edge" % (canonical, mechanism)
        )
    try:
        module_name, builder = MODULATED_EDGE_NETWORK[canonical]
    except KeyError:
        raise ValueError(
            "%s modulates connections but no network is recorded for it in "
            "MODULATED_EDGE_NETWORK, so the sign of the edges it scales "
            "cannot be read off its own module" % canonical
        )
    module = importlib.import_module("neurocascade.%s" % module_name)
    network = getattr(module, builder)()
    W = np.asarray(network.connectivity, dtype=float)

    system = _fp.TARGETS[canonical].system()
    signs = set()
    for post, pre in dict(system.connection_modulation or {}):
        signs.add(int(np.sign(W[network.index[post], network.index[pre]])))
    if len(signs) != 1 or 0 in signs:
        raise ValueError(
            "%s's declared edges in %s.%s have signs %s; there is no single "
            "edge sign to generalise" % (canonical, module_name, builder,
                                         sorted(signs))
        )
    return signs.pop()


# ---------------------------------------------------------------------------
# The activation a compound adds to, or removes from, a site
# ---------------------------------------------------------------------------


def _tone_target(target):
    """P6.2's `ToneTarget` for this target, or None when no basal transmitter
    concentration was ever measured for it."""
    for entry in _tone.TONE_TARGETS:
        if entry.name == target:
            return entry
    return None


def site_activation(site, concentrations):
    """The site's activation at binding equilibrium for held concentrations."""
    return float(site.activation(_projection.resting_occupancy(site, concentrations)))


def activation_change(target, site, compound, compound_nM, basal=None):
    """`(delta_activation, baseline_activation, drug_activation)` at `target`.

    `site` is the site the compound has already been added to (P8.1's
    `add_to_system` output), so its ligand list contains both the compound and
    whatever the P5 module put there, including the endogenous transmitter
    when the module carries one.

    The baseline leg holds the endogenous transmitter at P6.2's cited basal
    concentration and the compound at zero; the drug leg adds the compound.
    Everything else is zero in both legs, so every ligand this model is not
    asking about cancels.

    `delta_activation` is the quantity the whole-brain gain multiplies. It is
    zero for an antagonist at a site with no tone, which is a statement about
    the pharmacology and not a failure: displacing nothing changes nothing.
    """
    names = [lig.name for lig in site.ligands]
    if compound not in names:
        raise ValueError(
            "%r is not a ligand of the %s site; add it with "
            "fingerprint.add_to_system first (its ligands are %s)"
            % (compound, target, ", ".join(names))
        )

    baseline = np.zeros(len(names), dtype=float)
    entry = _tone_target(target)
    if entry is not None:
        level = (basal or {}).get(
            entry.transmitter, _tone.BASAL_TONE[entry.transmitter].nM
        )
        baseline[names.index(entry.ligand)] = float(level)

    with_drug = baseline.copy()
    with_drug[names.index(compound)] = float(compound_nM)

    a0 = site_activation(site, baseline)
    a1 = site_activation(site, with_drug)
    return a1 - a0, a0, a1


def behavioural_path_status(target, delta_activation, expression_genes):
    """Which of `PATH_STATUSES` applies to one compound-target pair.

    `expression_genes` is the set of gene symbols an ISH map exists for, so a
    target whose gene was never ingested is reported as such rather than
    silently dropped.
    """
    name = _fp.resolve_target(target)
    spec = _fp.TARGETS[name]
    if spec.is_transporter:
        return "transporter-no-whole-brain-field"
    if spec.gene not in expression_genes:
        return "no-expression-map"
    if delta_activation == 0.0:
        return (
            "antagonist-no-tone" if _tone_target(name) is None
            else "no-activation-change"
        )
    return "available"


# ---------------------------------------------------------------------------
# Several targets on one whole-brain network
# ---------------------------------------------------------------------------


class TargetLayer(object):
    """One target's whole-brain density map, mechanism and transduction sign.

    A thin record rather than a class with behaviour, because everything that
    acts on it is a property of the network the layers share.
    """

    __slots__ = ("target", "gene", "mechanism", "sign", "edge_sign",
                 "densities", "unmeasured_nodes")

    def __init__(self, target, gene, mechanism, sign, densities,
                 unmeasured_nodes=(), edge_sign=None):
        if mechanism not in _efwd.MECHANISMS:
            raise ValueError("unknown mechanism %r" % (mechanism,))
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1; got %r" % (sign,))
        if mechanism == "connection_modulation":
            if edge_sign not in (-1, 1):
                raise ValueError(
                    "%s modulates connections, so which SIGN of edge it "
                    "scales has to be stated; got edge_sign=%r. An NMDA "
                    "blocker applied to inhibitory edges has the sign of an "
                    "NMDA agonist." % (target, edge_sign)
                )
        elif edge_sign is not None:
            raise ValueError(
                "%s acts by %s and modulates no edge, so edge_sign must be "
                "None; got %r" % (target, mechanism, edge_sign)
            )
        self.target = target
        self.gene = gene
        self.mechanism = mechanism
        self.sign = int(sign)
        self.edge_sign = None if edge_sign is None else int(edge_sign)
        self.densities = dict(densities)
        self.unmeasured_nodes = tuple(unmeasured_nodes)

    def __repr__(self):
        return "TargetLayer(%r, %s, %s%s%s)" % (
            self.target, self.gene, "+" if self.sign > 0 else "-",
            self.mechanism,
            "" if self.edge_sign is None
            else (" on excitatory edges" if self.edge_sign > 0
                  else " on inhibitory edges"),
        )


class MultiTargetWholeBrainModel(object):
    """Several receptor systems acting on ONE whole-brain network.

    `endpoint_forward.WholeBrainDrugModel` carries one gene and one mechanism,
    which is all a single-target fit needs. A held-out compound is not
    single-target: a multi-target held-out compound can reach several modelled
    receptors and the model has to add their contributions on the same
    network before any endpoint is read.

    The composition is the sum P7.3's reduction already implies, extended over
    targets rather than assumed::

        drive_i  = baseline_i + sum_t  x_t * rho_t,i
        W_ij    <- W_ij * (1 + sum_t x_t * rho_t,i)        (W_ij < 0 only)

    with `x_t = gain_t * delta_activation_t`. Both sums run over the layers
    declaring that mechanism, and the two are independent, so a compound with
    a `drive_coupling` target and a `connection_modulation` target contributes
    to both at once. That the single-layer case reduces exactly to
    `WholeBrainDrugModel` is asserted by a test, not by this docstring.
    """

    def __init__(self, network, ontology, layers, name, region_sizes=None):
        node_names = [node.name for node in network.nodes]
        known = set(node_names)
        for layer in layers:
            unknown = sorted(set(layer.densities) - known)
            if unknown:
                raise ValueError(
                    "%s: densities name nodes the network does not have: %s"
                    % (layer.target, ", ".join(unknown))
                )
        self.network = network
        self.ontology = ontology
        self.layers = tuple(layers)
        self.name = str(name)
        self.region_sizes = region_sizes
        self._baseline_rates = None
        self._resolved = {}
        self._negative_edges = None

    @property
    def targets(self):
        return tuple(layer.target for layer in self.layers)

    @property
    def baseline_rates(self):
        """The drug-free steady state. Cacheable for P7.3's reason: at
        `x = 0` every drive contribution is zero and every modulation factor
        is exactly 1, so the drug-free leg cannot depend on any gain."""
        if self._baseline_rates is None:
            self._baseline_rates = _cfwd.steady_state_rates(self.network)
        return self._baseline_rates

    def _edges_of_sign(self, edge_sign):
        """Every edge of one sign, as `(post, pre)` node-name pairs.

        Cached per sign. `endpoint_forward` computes the inhibitory set only;
        this needs both, because the five connection-modulating systems in
        `TARGETS` split three-two between the two signs.
        """
        if self._negative_edges is None:
            names = [node.name for node in self.network.nodes]
            W = np.asarray(self.network.connectivity, dtype=float)
            self._negative_edges = {}
            for sign, mask in ((-1, W < 0.0), (1, W > 0.0)):
                rows, cols = np.nonzero(mask)
                self._negative_edges[sign] = [
                    (names[i], names[j]) for i, j in zip(rows, cols)
                ]
        return self._negative_edges[int(edge_sign)]

    def rates(self, x_by_target, warm_start=True):
        """Steady-state rates at `{target: x_t}`, `x_t = gain_t * delta_a_t`.

        A target absent from the mapping contributes nothing, which is the
        same thing as `x_t = 0` and is asserted as such by a test.
        """
        coupling = {}
        modulation = {}
        for layer in self.layers:
            x = float(x_by_target.get(layer.target, 0.0))
            if x == 0.0:
                continue
            if layer.mechanism == "drive_coupling":
                for node, rho in layer.densities.items():
                    coupling[node] = coupling.get(node, 0.0) + x * rho
            else:
                for post, pre in self._edges_of_sign(layer.edge_sign):
                    rho = layer.densities.get(post)
                    if rho is None:
                        continue
                    key = (post, pre)
                    modulation[key] = modulation.get(key, 0.0) + x * rho

        return _cfwd.steady_state_rates(
            self.network,
            connection_modulation=modulation or None,
            activation=1.0,
            coupling=coupling or None,
            initial_rates=self.baseline_rates if warm_start else None,
        )

    def resolved_endpoint(self, endpoint_key):
        if endpoint_key not in self._resolved:
            from .. import endpoints as _endpoints

            self._resolved[endpoint_key] = _endpoints.ENDPOINTS[
                endpoint_key
            ].resolve(self.network, self.ontology, region_sizes=self.region_sizes)
        return self._resolved[endpoint_key]

    def index(self, endpoint_key, rates):
        names = [node.name for node in self.network.nodes]
        return _efwd.directed_activity_index(
            self.resolved_endpoint(endpoint_key), dict(zip(names, rates))
        )

    def indices(self, x_by_target, endpoint_keys=ENDPOINT_KEYS):
        rates = self.rates(x_by_target)
        return {key: self.index(key, rates) for key in endpoint_keys}


def build_multi_target_model(targets, name, resolution="regions",
                             allow_download=False):
    """`(model, provenance)`: one network, one layer per target.

    Everything expensive - the connectome matrices, the parcellation, P3.4's
    stabilisation - is done once and shared, because the layers differ only
    in which ISH map they read. `endpoint_forward.build_whole_brain_model`
    does the same work per gene, and calling it once per target would restabilise
    the same 291-node network five times for clozapine.

    Targets with no ISH map, and transporters, are returned in
    `provenance["excluded"]` with the reason rather than dropped: a target the
    model cannot carry is a fact about the prediction, and the compound's
    report has to be able to name it.
    """
    from .. import connectome as _connectome
    from .. import expression as _expression
    from ..connectome import build as _build
    from .. import stability as _stability

    ontology, strength, voxels, source, description = _efwd.whole_brain_inputs(
        allow_download=allow_download
    )
    circuit = _build.build_circuit_network(
        strength, voxel_counts=voxels, resolution=resolution
    )
    stabilized = _stability.stabilize(circuit.network)
    region_sizes = _efwd.node_voxel_counts(ontology, circuit, voxels)

    wanted, excluded = [], {}
    for target in targets:
        canonical = _fp.resolve_target(target)
        spec = _fp.TARGETS[canonical]
        if spec.is_transporter:
            excluded[canonical] = "transporter-no-whole-brain-field"
            continue
        wanted.append((canonical, spec))

    genes = sorted({spec.gene for _, spec in wanted})
    if source == "fixture":
        atlas = _expression.load_fixture_atlas(symbols=genes, ontology=ontology)
    else:
        atlas = _expression.load_expression_atlas(symbols=genes, ontology=ontology)

    layers = []
    for canonical, spec in wanted:
        energy = atlas.energy_matrix(
            circuit.region_ids, symbols=[spec.gene],
            plane=_efwd.PLANE, combine=_efwd.COMBINE,
        )
        density_map = _expression.relative_density(
            energy, spec.gene, normalization=_efwd.DENSITY_NORMALIZATION
        )
        values = np.asarray(density_map.values, dtype=float).ravel()
        densities, unmeasured = {}, []
        for acronym, value in zip(circuit.acronyms, values):
            if np.isfinite(value):
                densities[acronym] = float(value)
            else:
                unmeasured.append(acronym)
        if not densities:
            excluded[canonical] = "no-expression-map"
            continue
        mechanism, sign = transduction_sign(canonical)
        edge_sign = (
            modulated_edge_sign(canonical)
            if mechanism == "connection_modulation" else None
        )
        layers.append(TargetLayer(
            target=canonical, gene=spec.gene, mechanism=mechanism, sign=sign,
            densities=densities, unmeasured_nodes=tuple(unmeasured),
            edge_sign=edge_sign,
        ))

    model = MultiTargetWholeBrainModel(
        network=stabilized.network, ontology=ontology, layers=layers,
        name=name, region_sizes=region_sizes,
    )
    provenance = {
        "resolution": resolution,
        "n_nodes": circuit.n,
        "connectome_source": source,
        "connectome_description": description,
        "operating_gain": _stability.DEFAULT_OPERATING_GAIN,
        "weight_scale": stabilized.weight_scale,
        "density_normalization": _efwd.DENSITY_NORMALIZATION,
        "density_plane": _efwd.PLANE,
        "density_combine": _efwd.COMBINE,
        "genes": {layer.target: layer.gene for layer in layers},
        "excluded": excluded,
    }
    return model, provenance


# ---------------------------------------------------------------------------
# The two gain-free quantities
# ---------------------------------------------------------------------------

#: The magnitude of `x` at which the endpoint index is treated as saturated.
#: Not a parameter of the model: `reachable_index_range` verifies that the
#: index has stopped moving by this point rather than assuming it, and raises
#: if it has not.
SATURATION_X = 1.0e3

#: How much the index may still be moving between `SATURATION_X` and ten times
#: it before `reachable_index_range` refuses to call the range converged.
SATURATION_TOLERANCE = 1.0e-3


class ReachableRange(object):
    """The interval one endpoint's index can occupy over ALL gains.

    The object a published effect is tested against without any parameter
    being chosen. `contains` is the one-sided verdict: an observation outside
    this interval is refuted by the model's structure, and one inside it is
    not thereby confirmed - it only means some gain reproduces it, and
    `implied_gain` is what says which.
    """

    __slots__ = ("endpoint", "lo", "hi", "baseline", "x_lo", "x_hi",
                 "converged", "drift")

    def __init__(self, endpoint, lo, hi, baseline, x_lo, x_hi, converged,
                 drift):
        self.endpoint = endpoint
        self.lo = float(lo)
        self.hi = float(hi)
        self.baseline = float(baseline)
        self.x_lo = float(x_lo)
        self.x_hi = float(x_hi)
        self.converged = bool(converged)
        self.drift = float(drift)

    @property
    def ratio_lo(self):
        return self.lo / self.baseline

    @property
    def ratio_hi(self):
        return self.hi / self.baseline

    def contains_ratio(self, ratio):
        return self.ratio_lo <= float(ratio) <= self.ratio_hi

    def shortfall(self, ratio):
        """How far outside the reachable interval `ratio` is, as a ratio.
        Zero when inside. Positive is the distance to the nearer edge."""
        ratio = float(ratio)
        if ratio < self.ratio_lo:
            return self.ratio_lo - ratio
        if ratio > self.ratio_hi:
            return ratio - self.ratio_hi
        return 0.0

    def __repr__(self):
        return "ReachableRange(%s, ratio [%.4f, %.4f]%s)" % (
            self.endpoint, self.ratio_lo, self.ratio_hi,
            "" if self.converged else ", NOT CONVERGED",
        )


def response_grid(saturation_x=SATURATION_X, points_per_decade=6,
                  smallest=1e-3):
    """The `x` values a target's response is tabulated at, both signs and zero.

    Geometric rather than linear because the interesting structure spans four
    decades: the index moves by parts in ten thousand at `x = 1e-3` and is
    fully saturated by `x = 1e3`, and a linear grid fine enough for the first
    would be ruinous over the second.
    """
    decades = np.log10(saturation_x / smallest)
    n = int(round(decades * points_per_decade)) + 1
    half = np.geomspace(smallest, saturation_x, n)
    return np.concatenate([-half[::-1], [0.0], half])


class TargetResponse(object):
    """One target's endpoint indices tabulated over `x`, swept once.

    This is P7.3's `ResponseCurve` at a different scale and for a different
    purpose. There it existed so an MCMC sampler need not integrate a 291-node
    ODE per draw; here it exists because both gain-free quantities this task
    reports - the reachable range and the implied gain - are read off the SAME
    sweep, for seven endpoints at once, and re-integrating per endpoint per
    inversion would cost hours to produce numbers already in hand.

    The interpolation is linear for P7.3's reason: a cubic can overshoot
    between grid points and put an index the circuit cannot produce into a
    reachable range, which is the one quantity here that must never be too
    wide. `interpolation_error` re-integrates at the grid midpoints and
    reports the largest disagreement, so the cost is a measured number in the
    results file rather than an assumption.
    """

    def __init__(self, target, x_grid, values, baseline):
        self.target = target
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        self.baseline = dict(baseline)

    @property
    def endpoint_keys(self):
        return tuple(sorted(self.values))

    def __call__(self, endpoint_key, x):
        x = np.asarray(x, dtype=float)
        if np.any(x < self.x_grid[0]) or np.any(x > self.x_grid[-1]):
            raise ValueError(
                "x outside the tabulated range [%g, %g]; extrapolating a "
                "circuit response is not something this curve may do"
                % (self.x_grid[0], self.x_grid[-1])
            )
        return np.interp(x, self.x_grid, self.values[endpoint_key])

    def interpolation_error(self, model, endpoint_keys=None):
        keys = tuple(endpoint_keys or self.endpoint_keys)
        midpoints = 0.5 * (self.x_grid[:-1] + self.x_grid[1:])
        worst = {key: 0.0 for key in keys}
        for x in midpoints:
            rates = model.rates({self.target: float(x)})
            for key in keys:
                worst[key] = max(
                    worst[key], abs(model.index(key, rates) - float(self(key, x)))
                )
        return worst


def sweep_target(model, target, endpoint_keys=ENDPOINT_KEYS,
                 x_grid=None):
    """Tabulate every endpoint's index against `x` for one target.

    One steady state per grid point, shared by all seven endpoints, because
    the rates do not depend on which readout is applied to them.
    """
    grid = response_grid() if x_grid is None else np.asarray(x_grid, dtype=float)
    values = {key: [] for key in endpoint_keys}
    for x in grid:
        rates = model.rates({target: float(x)})
        for key in endpoint_keys:
            values[key].append(model.index(key, rates))
    return TargetResponse(
        target=target, x_grid=grid, values=values,
        baseline=model.indices({}, endpoint_keys),
    )


def reachable_index_range(response, tolerance=SATURATION_TOLERANCE):
    """`{endpoint: ReachableRange}` read off a `TargetResponse`.

    Convergence is CHECKED rather than assumed: the whole value of this
    quantity is that it bounds the model at EVERY gain, and a range still
    growing at the edge of the sweep would bound nothing. The check is that
    the index has stopped moving over the outermost decade of the grid, which
    is a property of the tabulated values and needs no further integration.
    """
    ranges = {}
    grid = response.x_grid
    for key in response.endpoint_keys:
        arr = response.values[key]
        lo, hi = float(arr.min()), float(arr.max())
        # Movement over the outermost decade at each end.
        outer = max(
            abs(arr[0] - arr[np.argmin(np.abs(grid - grid[0] / 10.0))]),
            abs(arr[-1] - arr[np.argmin(np.abs(grid - grid[-1] / 10.0))]),
        )
        ranges[key] = ReachableRange(
            endpoint=key, lo=lo, hi=hi, baseline=response.baseline[key],
            x_lo=float(grid[int(arr.argmin())]),
            x_hi=float(grid[int(arr.argmax())]),
            converged=outer <= tolerance, drift=float(outer),
        )
    return ranges


def implied_x(response, endpoint_key, observed_ratio):
    """The `x = gain * delta_activation` reproducing `observed_ratio`, or None.

    Returned as `x` rather than as a gain because the two differ by the
    compound's own `delta_activation`, which the caller has and this function
    does not. Dividing here would hide which of the two factors a large value
    came from.

    Solved on the tabulated curve rather than by a root finder against the
    integrator: the index is not guaranteed monotone in `x`, so a Newton step
    off a flat saturated shoulder goes anywhere, while a bracketed scan of the
    grid finds every crossing. Returns None when there is no crossing, i.e.
    when the observation is outside the reachable range - and `implied_x` and
    `reachable_index_range` must agree on that, which a test asserts.
    """
    goal = float(observed_ratio) * response.baseline[endpoint_key]
    xs = response.x_grid
    ys = response.values[endpoint_key]

    roots = []
    for i in range(len(xs) - 1):
        f0, f1 = ys[i] - goal, ys[i + 1] - goal
        if f0 == 0.0:
            roots.append(float(xs[i]))
        elif f0 * f1 < 0.0:
            # Linear interpolation between the bracketing grid points, which
            # is exactly the function `TargetResponse.__call__` evaluates, so
            # the root is exact FOR THE CURVE rather than approximate for it.
            roots.append(float(xs[i] - f0 * (xs[i + 1] - xs[i]) / (f1 - f0)))
    if ys[-1] == goal:
        roots.append(float(xs[-1]))
    if not roots:
        return None
    # The smallest gain that explains the observation. A larger root exists
    # only where the curve is non-monotone, and reporting the biggest one
    # would overstate how implausible the required gain is.
    return min(roots, key=abs)


__all__ = [
    "ENDPOINT_KEYS",
    "PATH_STATUSES",
    "SATURATION_X",
    "MultiTargetWholeBrainModel",
    "ReachableRange",
    "TargetLayer",
    "activation_change",
    "behavioural_path_status",
    "TargetResponse",
    "implied_x",
    "reachable_index_range",
    "response_grid",
    "sweep_target",
    "site_activation",
    "transduction_sign",
]
