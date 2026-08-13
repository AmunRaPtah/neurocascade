import numpy as np
from scipy.integrate import solve_ivp

from .pk import as_ode_pk_model
from .projection import as_projection_source


class ReceptorSystem(object):
    """One receptor site's full pharmacology stack, bundled so several of
    these can drive the same shared `CircuitNetwork` at once (see
    `Simulation(network, receptor_systems=[...])`).

    pk_models   : one PK model per ligand on `receptor_site` (same order).
    receptor_site: the `ReceptorSite` this system's ligands compete at.
    coupling    : dict of node_name -> gain, exactly like the single-system
                  `Simulation.coupling` this replaces: this system's
                  receptor activation is added to that node's drive scaled
                  by gain.
    connection_modulation: optional dict of (post_node_name, pre_node_name)
                  -> gain, exactly like the single-system
                  `Simulation.connection_modulation` this replaces.
    name        : optional label (defaults to `receptor_site.name`), used
                  only for error messages/repr.

    `n` is the site's LIGAND count (one PK model and one `doses` entry
    each); `n_states` is how many ODE states the site owns. They are equal
    for a plain `ReceptorSite` and differ for a
    `receptor.DesensitizingReceptorSite`, which carries a second state per
    orthosteric ligand plus a channel-block pool (P5.4). Sites that predate
    P5.4 have no `n_states` attribute at all, hence the `getattr` default,
    so a duck-typed external receptor site keeps working unchanged.

    transmitter_sources: optional dict of ligand_name -> transporter-system
                  name. THIS IS THE PHASE 6 INTERFACE (P5.8). A ligand named
                  here has its concentration taken from that transporter
                  system's extracellular transmitter state instead of from
                  its own PK model, which is what closes the loop
                  "transporter blocked -> transmitter accumulates ->
                  occupancy at the receptor rises". The ligand keeps a PK
                  model (the engine sizes `doses` by ligand count either
                  way); that model's state is integrated and then ignored
                  for this ligand's binding, so the natural thing to give it
                  is a `pk.OneCompartmentPK` dosed at 0. See
                  `Simulation`'s docstring, section "Transporter systems".

                  P6.1 WIDENS THIS, without changing what a string means: a
                  value may instead be a `projection.ProjectionSource`
                  (equivalently a `(projection_system_name, region)` pair),
                  in which case the ligand binds that projection system's
                  concentration in that named region. A plain string still
                  names a transporter system and behaves exactly as it did.
                  A receptor that should sit in MANY regions at once is a
                  `projection.RegionalReceptorSystem`, not this class.
    """

    #: This class is the one-block, one-node receptor system; the regional
    #: code path in `Simulation.rhs` is opt-in through
    #: `projection.RegionalReceptorSystem`.
    is_regional = False
    n_regions = 1

    def __init__(
        self,
        pk_models,
        receptor_site,
        coupling,
        connection_modulation=None,
        name=None,
        transmitter_sources=None,
    ):
        if len(pk_models) != receptor_site.n:
            raise ValueError("need one PK model per ligand on the receptor site")
        self.pk_models = list(pk_models)
        self.receptor_site = receptor_site
        self.coupling = dict(coupling)
        self.connection_modulation = dict(connection_modulation or {})
        self.name = name or receptor_site.name
        self.n = receptor_site.n
        self.n_states = int(getattr(receptor_site, "n_states", receptor_site.n))
        self.transmitter_sources = dict(transmitter_sources or {})
        known = {lig.name for lig in receptor_site.ligands}
        unknown = set(self.transmitter_sources) - known
        if unknown:
            raise ValueError(
                "transmitter_sources names ligands that are not on this "
                "receptor site: %s" % sorted(unknown)
            )

    def __repr__(self):
        return "ReceptorSystem(name=%r, n_ligands=%d)" % (self.name, self.n)


class TransporterSystem(object):
    """One transporter's full stack: drugs, their PK, and the transmitter
    pool the transporter clears (P5.8).

    The deliberate parallel to `ReceptorSystem`, and the deliberate
    differences:

    * it has NO `coupling` and NO `connection_modulation`, because a
      transporter produces no activation for a node to read. Its entire
      output is one number, the extracellular transmitter concentration,
      which lives in the shared state vector and is reported as
      `result["transmitter"]`;
    * its state does not start at zero. A transporter's pool starts at the
      transmitter's drug-free basal concentration;
    * `transporter.release_rate` is the input Phase 6 replaces. P6.1's
      projection-system abstraction ends in "region-wise transmitter
      concentration"; this class is the object that owns that concentration
      and integrates it, and the release term is the hook a firing-rate-
      driven release model plugs into.

    pk_models  : one PK model per ligand on `transporter` (same order).
    transporter: a `transport.MonoamineTransporter`.
    name       : optional label (defaults to `transporter.name`); it is what
                 a `ReceptorSystem.transmitter_sources` entry refers to, so
                 it must be unique within one `Simulation`.
    """

    def __init__(self, pk_models, transporter, name=None):
        if len(pk_models) != transporter.n:
            raise ValueError("need one PK model per ligand on the transporter")
        self.pk_models = list(pk_models)
        self.transporter = transporter
        self.name = name or transporter.name
        self.n = transporter.n
        self.n_states = int(getattr(transporter, "n_states", 1))
        self.transmitter = transporter.transmitter

    def __repr__(self):
        return "TransporterSystem(name=%r, transmitter=%r, n_ligands=%d)" % (
            self.name,
            self.transmitter,
            self.n,
        )


class Simulation(object):
    """Wires one or more (pharmacokinetics -> receptor binding) stacks into
    a single shared circuit network into one coupled ODE system and
    integrates it with RK45 (as in the CCT model).

    Two ways to construct a Simulation:

    1. Single receptor system (original v0 API, unchanged, still the normal
       case for a scenario that models one receptor site on one circuit)::

           Simulation(pk_models=[...], receptor_site=site, network=net,
                      coupling={...}, connection_modulation={...})

    2. Multiple receptor systems acting on one shared circuit at once (e.g.
       a mu-opioid system on VTA_GABA and a D2 system on Striatum_D2_MSN,
       both coupling into one combined network)::

           Simulation(network=net, receptor_systems=[mu_system, d2_system])

       where each entry is a `ReceptorSystem` (or anything with the same
       `pk_models`/`receptor_site`/`coupling`/`connection_modulation`/`n`
       attributes). Passing both forms at once is an error; passing
       neither is an error.

    3. With one or more TRANSPORTER systems (P5.8), which may accompany
       receptor systems or stand alone::

           Simulation(network=net, transporter_systems=[dat_system])
           Simulation(network=net, receptor_systems=[d2], transporter_systems=[dat])

       A transporter is not a receptor (see `transport.py`): it produces no
       activation, it owns one state -- the extracellular transmitter
       concentration -- and that state is what Phase 6 consumes.

    4. With one or more PROJECTION systems (P6.1), which turn a source
       nucleus's firing rate into a region-wise transmitter concentration
       field::

           Simulation(network=net,
                      receptor_systems=[d2_in_striatum_and_cortex],
                      projection_systems=[vta_dopamine])

       A projection system is not a receptor either (see `projection.py`):
       it produces no activation, it owns one state PER TARGET REGION -- that
       region's extracellular transmitter concentration -- and those states
       are what a `projection.RegionalReceptorSystem`, or a
       `ReceptorSystem` whose `transmitter_sources` names a
       `(system, region)` pair, binds.

    State vector layout, generalizing to k receptor systems with n_1..n_k
    ligands respectively (r = sum(n_i) ligands; reduces to the original
    [C_1..C_d, occ_1..occ_d, r_1..r_n] layout when k == 1 and every ligand's
    PK model owns a single state):

        y = [<PK states of every ligand, in flattened ligand order>,  # d of them
             <receptor states of system 1>, <system 2>, ...,         # r of them
             <transmitter state of transporter system 1>, ...,       # m of them
             <transmitter field of projection system 1>, ...,        # p of them
             r_1, ..., r_n]                                          # shared rates

    i.e. every ligand's PK state block, in system order, then every system's
    receptor-state block, in the same order, then one transmitter
    concentration per transporter system, then one concentration per
    (projection system, target region), then one shared set of circuit
    firing rates. `doses` passed to `run()` must be given in that same
    flattened LIGAND order -- every receptor system's ligands first, in
    system order, then every transporter system's, then every projection
    system's -- one entry per ligand (see `dose_vector()` for a name-based
    helper that builds it for you). With no transporter systems m is 0 and
    with no projection systems p is 0, so the layout is bit-for-bit the
    pre-P5.8 one for every simulation that declares neither.

    A system's receptor-state block is one occupancy per ligand for the
    ordinary two-state `ReceptorSite`, so for every scenario before P5.4
    `self.r == self.n_ligands` and this layout is exactly the v0 one. A
    `receptor.DesensitizingReceptorSite` (P5.4, nicotinic receptors) owns
    more states than it has ligands -- a desensitized state per orthosteric
    ligand and a channel-block pool -- so the two counts are tracked
    separately and `state_rows(i)` gives system `i`'s rows.

    A ligand's PK block is as long as its PK model's `n_states` (1 for
    `OneCompartmentPK` and for an IV `DosedOneCompartmentPK`; 2 for an
    extravascular one-compartment model, which carries a depot; 2-3 for the
    two-compartment models), so `self.d`, the total PK state count, is >= r
    rather than equal to it as in v0. By the PK protocol's convention the
    *first* entry of each ligand's block is the central-compartment
    concentration the receptor site binds, and `self.concentration_rows`
    holds those row indices, so `result["concentrations"]` keeps its v0
    shape of (n_ligands, n_times) whatever PK models are in use. Keeping the
    PK states in the shared vector, rather than precomputing a concentration
    trace per ligand and interpolating it, is the same "everything reacts to
    everything else's current state" decision documented at the top of
    docs/architecture.md, and it is what makes a mid-run second dose (see
    `examples/fentanyl_reward_circuit.py`) a state edit rather than a
    re-derivation.

    Plasma is not what a receptor binds (P1.5): each ligand's PK model may
    carry a `brain_partition`, in which case the occupancy ODE is driven by
    that ligand's unbound BRAIN concentration rather than by its plasma
    concentration. The conversion is one constant per ligand
    (`receptor_concentration_factors`, exactly 1.0 when no partition is
    attached), so `result["concentrations"]` keeps its original meaning and
    values (plasma) and `result["receptor_concentrations"]` is what the
    receptor layer saw. See pk.py's protocol section and
    docs/architecture.md, P1.5.

    Composing multiple systems' contributions to the shared circuit:

    - `coupling`: each system's receptor activation is added to its target
      node(s)' drive, scaled by that system's own gain. Contributions from
      different systems targeting the same node are simply summed: drive is
      additive by construction in the Wilson-Cowan input term, so two
      independent receptor systems driving the same node is exactly "two
      independent inputs summed", the ordinary linear superposition already
      implied by the circuit equation. No new assumption is introduced by
      allowing more than one system to contribute to it.
    - `connection_modulation`: each system's `(1 + gain * activation)`
      scaling factor for a targeted connection is composed
      *multiplicatively* across systems when more than one system modulates
      the exact same `(post_node, pre_node)` connection, i.e. the effective
      weight is `base_weight * factor_1 * factor_2 * ...`. This is a
      deliberate choice, not the only possible one: summing the gain terms
      before adding 1 (`1 + sum(gain_i * activation_i)`) was the
      alternative considered. Each `connection_modulation` entry represents
      an independent allosteric/gain modulator acting on the *same physical
      synapse's transmission efficacy*, and independent multiplicative
      effects on efficacy are what "gain modulation" ordinarily means
      mechanistically, comparable to two independent multiplicative
      discounts rather than two additive ones. For example, if two
      different modulators each independently double a synapse's effective
      strength, the synapse should end up 4x, not 3x, its base strength.
      Summing the gain terms first would let one modulator's effect be
      diluted or amplified by whatever the other modulator's activation
      happens to be doing at that instant, which has no clean mechanistic
      reading. In the overwhelmingly common case relevant to this codebase,
      a single receptor system modulating any given connection,
      multiplicative and additive composition are identical (there's
      nothing to compose), so this choice only matters once two systems are
      deliberately wired to modulate the exact same connection, which is an
      unusual, explicit scenario design choice rather than the default.

    Transporter systems, and the Phase 6 hand-off (P5.8)
    ----------------------------------------------------

    A `TransporterSystem` contributes nothing to `drive` and nothing to
    `connectivity`. What it does is integrate one extracellular transmitter
    concentration alongside everything else, driven by its own drugs'
    concentrations. Two consequences worth stating:

    * `result["transmitter"]` is a (m, n_times) array in nM, one row per
      transporter system in the order given, and
      `transmitter_index[name]` maps a system name to its row. **This array
      is the documented interface Phase 6's projection systems consume.**
    * a receptor system may declare `transmitter_sources={ligand: system}`,
      in which case that ligand's binding is driven by that transporter's
      transmitter state instead of by its own PK model. That is the closed
      loop -- block the transporter, the transmitter accumulates, occupancy
      at the receptor rises -- and it is opt-in, so every result predating
      P5.8 is unchanged. Receptor binding does NOT deplete the transmitter
      pool here: the receptor population is a vanishing sink beside uptake,
      and modelling it as one would need a receptor density in absolute
      molar terms, which P4.2 explicitly does not provide.

    Projection systems, and what makes them different (P6.1)
    --------------------------------------------------------

    A `projection.ProjectionSystem` is a `TransporterSystem` with two things
    added and nothing taken away: its release term is driven by a CIRCUIT
    NODE's firing rate rather than being a constant, and it owns one pool per
    TARGET REGION rather than one pool. That first difference is the one that
    matters structurally, because it is the first term in this codebase's
    right-hand side that carries information from the circuit layer BACK into
    the concentration layer. Every module from P5.1 to P5.8 had to record the
    autoreceptor loop as open for exactly this reason; with a projection
    system attached it is closed.

    * `result["projection"]` is a (p, n_times) array in nM, one row per
      (projection system, target region), and `projection_index[(name,
      region)]` maps a pair to its row.
    * `projection.RegionalReceptorSystem` is the receptor side: one receptor
      type, one occupancy block per region, each block reading its own
      region's row and driving its own region's node.
    """

    def __init__(
        self,
        network=None,
        receptor_systems=None,
        pk_models=None,
        receptor_site=None,
        coupling=None,
        connection_modulation=None,
        transporter_systems=None,
        projection_systems=None,
    ):
        if network is None:
            raise TypeError("Simulation requires a network")

        legacy_kwargs_given = (
            pk_models is not None
            or receptor_site is not None
            or coupling is not None
            or connection_modulation is not None
        )
        transporter_systems = list(transporter_systems or [])
        projection_systems = list(projection_systems or [])
        if receptor_systems is None:
            if pk_models is None or receptor_site is None or coupling is None:
                # A transporter-only Simulation is legitimate (P5.8): the
                # transmitter pool is a complete model on its own, and the
                # receptor layer is what Phase 6 will attach to it. A
                # projection-only one is legitimate for the same reason
                # (P6.1): the transmitter FIELD is a complete model on its
                # own, and it is what the receptor layer attaches to.
                if not transporter_systems and not projection_systems:
                    raise TypeError(
                        "Simulation needs receptor_systems=[...], "
                        "transporter_systems=[...], projection_systems=[...], "
                        "or the single-receptor-system trio pk_models=/"
                        "receptor_site=/coupling= "
                        "(connection_modulation= optional)"
                    )
                receptor_systems = []
            else:
                receptor_systems = [
                    ReceptorSystem(
                        pk_models, receptor_site, coupling, connection_modulation
                    )
                ]
        elif legacy_kwargs_given:
            raise TypeError(
                "pass either receptor_systems=[...] or the legacy single-"
                "system kwargs (pk_models=/receptor_site=/coupling=/"
                "connection_modulation=), not both"
            )

        receptor_systems = list(receptor_systems)
        if not receptor_systems and not transporter_systems and not projection_systems:
            raise ValueError(
                "need at least one receptor, transporter or projection system"
            )

        transporter_names = [system.name for system in transporter_systems]
        if len(set(transporter_names)) != len(transporter_names):
            raise ValueError(
                "transporter system names must be unique within a Simulation; "
                "they are what transmitter_sources refers to"
            )
        projection_names = [system.name for system in projection_systems]
        if len(set(projection_names)) != len(projection_names):
            raise ValueError(
                "projection system names must be unique within a Simulation; "
                "they are what transmitter_sources refers to"
            )
        shared = set(transporter_names) & set(projection_names)
        if shared:
            raise ValueError(
                "transporter and projection system names must be disjoint, so "
                "that a bare string in transmitter_sources is unambiguous; "
                "shared: %s" % sorted(shared)
            )
        for system in projection_systems:
            for node in system.source_nodes:
                if node not in network.index:
                    raise ValueError(
                        "projection system %r is driven by node %r, which is "
                        "not in this network; have %s"
                        % (system.name, node, sorted(network.index))
                    )

        self.network = network
        self.receptor_systems = receptor_systems
        self.transporter_systems = transporter_systems
        self.projection_systems = projection_systems
        self._n_per_system = [system.n for system in receptor_systems]
        self._offsets = np.cumsum([0] + self._n_per_system[:-1]).tolist()
        # Receptor-state block lengths, which equal the ligand counts for
        # every two-state site and exceed them for a desensitizing one
        # (P5.4). Two offset tables are therefore needed: one to slice the
        # per-ligand concentrations out of the PK region, one to slice the
        # per-system state block out of the occupancy region.
        self._n_states_per_system = [
            int(getattr(system, "n_states", system.n)) for system in receptor_systems
        ]
        self._state_offsets = np.cumsum([0] + self._n_states_per_system[:-1]).tolist()

        # Transporter ligands sit AFTER every receptor ligand in the flat
        # `doses` order, so adding a transporter system cannot renumber an
        # existing receptor system's doses.
        self._n_per_transporter = [system.n for system in transporter_systems]
        self._transporter_offsets = (
            np.cumsum([sum(self._n_per_system)] + self._n_per_transporter[:-1]).tolist()
            if transporter_systems
            else []
        )
        # Projection ligands sit after every receptor and transporter ligand,
        # for the same reason: adding a projection system must not renumber
        # anything that already exists.
        self._n_per_projection = [system.n for system in projection_systems]
        self._projection_offsets = (
            np.cumsum(
                [sum(self._n_per_system) + sum(self._n_per_transporter)]
                + self._n_per_projection[:-1]
            ).tolist()
            if projection_systems
            else []
        )

        #: Number of ligands across all systems: the length of `doses`.
        self.n_ligands = (
            sum(self._n_per_system)
            + sum(self._n_per_transporter)
            + sum(self._n_per_projection)
        )
        #: Number of receptor STATES across all systems: the length of the
        #: occupancy region of y. Equal to the receptor-ligand count unless a
        #: desensitizing site is present.
        self.r = sum(self._n_states_per_system)
        #: Number of TRANSMITTER states: one per transporter system, 0 before
        #: P5.8 and 0 for every simulation that declares none.
        self.m = sum(
            int(getattr(system, "n_states", 1)) for system in transporter_systems
        )
        #: Number of PROJECTION states: one per (system, target region), 0
        #: before P6.1 and 0 for every simulation that declares none.
        self.p = sum(int(system.n_states) for system in projection_systems)
        self.c = network.n

        # Flat, ligand-ordered PK view. Every model is normalized to the ODE
        # protocol (pk.as_ode_pk_model), so the engine only ever calls
        # initial_state/concentration_rhs/n_states and never needs to know
        # whether a ligand is on a bolus-concentration model or a full
        # dose-driven two-compartment one.
        self._ode_pk_models = []
        self._pk_blocks = []  # (start, stop) into the PK region of y, per ligand
        concentration_rows = []
        receptor_factors = []
        cursor = 0
        for system in (
            list(receptor_systems)
            + list(transporter_systems)
            + list(projection_systems)
        ):
            for model in getattr(system, "pk_models", []):
                ode_model = as_ode_pk_model(model)
                n_states = int(ode_model.n_states)
                self._ode_pk_models.append(ode_model)
                self._pk_blocks.append((cursor, cursor + n_states))
                concentration_rows.append(cursor)  # central concentration first
                receptor_factors.append(
                    float(getattr(ode_model, "receptor_concentration_factor", 1.0))
                )
                cursor += n_states
        self.d = cursor  # total PK states; == self.r iff every model is 1-state
        self.concentration_rows = np.array(concentration_rows, dtype=int)
        #: Per-ligand constant converting the central (plasma/blood)
        #: concentration into the concentration the receptor actually binds
        #: (unbound brain, when that ligand's PK model carries a
        #: `brain_partition`). Exactly 1.0 for every plasma-driven model.
        #: `getattr(..., 1.0)` keeps duck-typed external PK models working
        #: without change. See pk.py's protocol and docs/architecture.md, P1.5.
        self.receptor_concentration_factors = np.array(receptor_factors, dtype=float)
        #: True when no ligand carries a partition, in which case `rhs` hands
        #: the plasma array itself to the receptor layer rather than a scaled
        #: copy. Not only an allocation saved per RHS call: it makes "no
        #: partition reproduces the pre-P1.5 result" an identity of objects
        #: rather than of values, so it cannot drift.
        self._all_receptor_factors_are_one = bool(
            np.all(self.receptor_concentration_factors == 1.0)
        )

        #: Where each transporter system's block starts in the transmitter
        #: region. One state each today, so this is 0, 1, 2, ...; kept as a
        #: cumulative table so a multi-pool transporter would not silently
        #: misalign anything.
        self._n_states_per_transporter = [
            int(getattr(system, "n_states", 1)) for system in transporter_systems
        ]
        self._transmitter_state_offsets = np.cumsum(
            [0] + self._n_states_per_transporter[:-1]
        ).tolist()
        #: Row index into `result["transmitter"]` for each transporter
        #: system, by name. Public because it is half of the Phase 6
        #: interface: the other half is `result["transmitter"]` itself.
        self.transmitter_index = {
            system.name: offset
            for system, offset in zip(
                transporter_systems, self._transmitter_state_offsets
            )
        }
        #: Where each projection system's block starts in the projection
        #: region, and the (system name, region) -> row map that is the
        #: public read of the transmitter FIELD (P6.1).
        self._n_states_per_projection = [
            int(system.n_states) for system in projection_systems
        ]
        self._projection_state_offsets = np.cumsum(
            [0] + self._n_states_per_projection[:-1]
        ).tolist()
        self.projection_index = {}
        for system, offset in zip(projection_systems, self._projection_state_offsets):
            for region in system.regions:
                self.projection_index[(system.name, region)] = offset + system.row(
                    region
                )
        self._projection_by_name = {
            system.name: system for system in projection_systems
        }
        #: Row indices into the shared `rates` vector of each projection
        #: system's source nuclei, in that system's own `sources` order.
        #: Resolved once so the RHS does no name lookup.
        self._projection_source_rows = [
            np.array(
                [network.index[node] for node in system.source_nodes], dtype=int
            )
            for system in projection_systems
        ]

        # Pool rows. Transporter pools and projection pools are addressed by
        # one index space -- transporters first, then projections -- so that
        # a single override table serves both and P5.8's rows keep their
        # exact meaning. `_pool_row_of_projection` is that offset.
        self._pool_row_of_projection = self.m

        #: (flat ligand index, pool row) pairs, resolved once, for every
        #: ligand of a NON-regional receptor system whose concentration comes
        #: from a transmitter pool rather than from its own PK model. P5.8
        #: built this for transporters; P6.1 lets a pair name a projection
        #: system's region instead.
        self._transmitter_overrides = []
        for system, offset in zip(receptor_systems, self._offsets):
            sources = getattr(system, "transmitter_sources", None)
            if not sources or getattr(system, "is_regional", False):
                continue
            positions = {
                lig.name: i for i, lig in enumerate(system.receptor_site.ligands)
            }
            for ligand_name, source in sources.items():
                self._transmitter_overrides.append(
                    (
                        offset + positions[ligand_name],
                        self._resolve_pool_row(system.name, ligand_name, source),
                    )
                )

        #: Per regional receptor system, the per-region override table:
        #: a list of (ligand index WITHIN the system, array of pool rows, one
        #: per region). Resolved once, so the RHS does no name lookup.
        self._regional_overrides = []
        for system in receptor_systems:
            if not getattr(system, "is_regional", False):
                self._regional_overrides.append([])
                continue
            positions = {
                lig.name: i for i, lig in enumerate(system.receptor_site.ligands)
            }
            table = []
            for ligand_name, source in system.transmitter_sources.items():
                rows = [
                    self._resolve_pool_row(
                        system.name, ligand_name, source, region=region
                    )
                    for region in system.regions
                ]
                table.append((positions[ligand_name], np.array(rows, dtype=int)))
            self._regional_overrides.append(table)

    def _resolve_pool_row(self, system_name, ligand_name, source, region=None):
        """Turn one `transmitter_sources` value into a row of the combined
        (transporter pools | projection pools) index space.

        `region` is the block's own region for a regional receptor system,
        used when the source names a projection system without one.
        """
        projection = as_projection_source(
            source, projection_names=self._projection_by_name, region=region
        )
        if projection is None:
            if source not in self.transmitter_index:
                raise ValueError(
                    "receptor system %r sources ligand %r from transporter "
                    "system %r, which is not in this Simulation; have %s"
                    % (system_name, ligand_name, source,
                       sorted(self.transmitter_index))
                )
            return self.transmitter_index[source]
        target_region = projection.region if projection.region is not None else region
        key = (projection.system, target_region)
        if key not in self.projection_index:
            raise ValueError(
                "receptor system %r sources ligand %r from projection system "
                "%r region %r, which is not in this Simulation; have %s"
                % (system_name, ligand_name, projection.system, target_region,
                   sorted(self.projection_index))
            )
        return self._pool_row_of_projection + self.projection_index[key]

    # Backward-compatible single-system attribute access.
    # These proxy to the sole ReceptorSystem when there is exactly one
    # (whether the caller used the legacy kwargs or passed
    # receptor_systems=[a_single_system]), so old code/tests that read
    # sim.pk_models / sim.receptor_site / sim.coupling /
    # sim.connection_modulation directly keep working unmodified. They
    # raise for a genuinely multi-system Simulation, where a single such
    # attribute would be ambiguous; use sim.receptor_systems[i].<attr>
    # instead in that case.
    def _sole_system(self, attr_name):
        if len(self.receptor_systems) != 1:
            raise AttributeError(
                "%r is only available on a single-receptor-system Simulation; "
                "this Simulation has %d receptor systems, use "
                "sim.receptor_systems[i].%s instead"
                % (attr_name, len(self.receptor_systems), attr_name)
            )
        return getattr(self.receptor_systems[0], attr_name)

    @property
    def pk_models(self):
        return self._sole_system("pk_models")

    @property
    def receptor_site(self):
        return self._sole_system("receptor_site")

    @property
    def coupling(self):
        return self._sole_system("coupling")

    @property
    def connection_modulation(self):
        return self._sole_system("connection_modulation")

    def dose_vector(self, **dose_by_ligand_name):
        """Build a flat per-ligand `doses` array (for `run()`) from
        ligand-name -> dose kwargs; ligands not mentioned default to 0.
        Mainly useful once there's more than one receptor system and the
        flat per-system ligand ordering is easy to get wrong by hand.

        Each entry is in whatever unit that ligand's own PK model doses in:
        a bolus concentration for `OneCompartmentPK`, mg/kg for the
        dose-administered models. 0 means "no drug" for either.
        """
        doses = np.zeros(self.n_ligands)
        offset = 0
        seen = set()
        for system in self.receptor_systems:
            for lig in system.receptor_site.ligands:
                if lig.name in dose_by_ligand_name:
                    doses[offset] = dose_by_ligand_name[lig.name]
                    seen.add(lig.name)
                offset += 1
        for system in self.transporter_systems:
            for lig in system.transporter.ligands:
                if lig.name in dose_by_ligand_name:
                    doses[offset] = dose_by_ligand_name[lig.name]
                    seen.add(lig.name)
                offset += 1
        for system in self.projection_systems:
            for lig in system.ligands:
                if lig.name in dose_by_ligand_name:
                    doses[offset] = dose_by_ligand_name[lig.name]
                    seen.add(lig.name)
                offset += 1
        unknown = set(dose_by_ligand_name) - seen
        if unknown:
            raise ValueError("unknown ligand name(s): %s" % sorted(unknown))
        return doses

    def state_rows(self, system_index):
        """Row indices, into `result["occupancy"]`, of one system's states.

        For a plain two-state site those rows ARE the per-ligand
        occupancies, one each, in ligand order. For a
        `DesensitizingReceptorSite` the block is longer than the ligand
        count and its internal layout belongs to the site; pass the sliced
        rows to `site.split_states(...)` rather than indexing them by hand.
        """
        start = self._state_offsets[system_index]
        return np.arange(start, start + self._n_states_per_system[system_index])

    def transmitter_rows(self, system_index):
        """Row indices, into `result["transmitter"]`, of one transporter
        system's states. One row each today; kept as a range so a future
        multi-pool transporter does not change the call sites."""
        start = self._transmitter_state_offsets[system_index]
        return np.arange(start, start + self._n_states_per_transporter[system_index])

    def projection_rows(self, system_index):
        """Row indices, into `result["projection"]`, of one projection
        system's target regions, in that system's own target order."""
        start = self._projection_state_offsets[system_index]
        return np.arange(start, start + self._n_states_per_projection[system_index])

    def projection_trace(self, result, system_name, region):
        """The (n_times,) concentration trace of one target region of one
        projection system, in nM. A named read, so a call site never has to
        reconstruct the row arithmetic."""
        return result["projection"][self.projection_index[(system_name, region)]]

    def _unpack(self, y):
        """Split y into (PK states, occupancies, transmitters, projection
        field, rates). The first return value is the whole PK region, which
        is per-ligand *concentration* only when every PK model owns a single
        state; use `self.concentration_rows` to pull the per-ligand
        concentrations out of it in the general case. The transmitter region
        is empty for every Simulation with no transporter systems, and the
        projection region for every one with no projection systems, which
        together is every Simulation that predates P5.8."""
        d, r, m, p = self.d, self.r, self.m, self.p
        pk_state = y[0:d]
        occ = y[d : d + r]
        transmitter = y[d + r : d + r + m]
        projection = y[d + r + m : d + r + m + p]
        rates = y[d + r + m + p : d + r + m + p + self.c]
        return pk_state, occ, transmitter, projection, rates

    def _pools(self, transmitter, projection):
        """The combined (transporter pools | projection pools) vector the
        override tables index into. Avoids an allocation in the two common
        cases where only one kind of pool exists."""
        if self.p == 0:
            return transmitter
        if self.m == 0:
            return projection
        return np.concatenate([transmitter, projection])

    def rhs(self, t, y):
        pk_state, occ, transmitter, projection, rates = self._unpack(y)
        pools = self._pools(transmitter, projection)

        d_pk = np.empty(self.d)
        for model, (start, stop) in zip(self._ode_pk_models, self._pk_blocks):
            d_pk[start:stop] = model.concentration_rhs(pk_state[start:stop])
        # One concentration per ligand, in the same flattened ligand order
        # the receptor systems are laid out in. `C` is the central-compartment
        # (plasma/blood) concentration; `C_receptor` is what the receptor
        # binds, which is unbound BRAIN concentration for any ligand whose PK
        # model carries a brain partition and identically `C` for one that
        # does not (P1.5).
        C = pk_state[self.concentration_rows]
        C_receptor = (
            C
            if self._all_receptor_factors_are_one
            else C * self.receptor_concentration_factors
        )
        if self._transmitter_overrides:
            # The Phase 6 hand-off, applied before the receptor layer runs:
            # a ligand declared in `transmitter_sources` binds the
            # extracellular transmitter concentration the transporter layer
            # is maintaining, not the plasma concentration its own PK model
            # carries. `C` is already a fresh array (fancy indexing copies),
            # but copy explicitly so that this cannot silently start writing
            # into the state vector if that ever changes.
            C_receptor = np.array(C_receptor, dtype=float)
            for ligand_row, pool_row in self._transmitter_overrides:
                C_receptor[ligand_row] = pools[pool_row]

        d_occ = np.empty(self.r)
        drive = self.network.baseline_drive()
        modulation_factors = {}  # (post_name, pre_name) -> composed factor

        for system, offset, s_offset, n_states, overrides in zip(
            self.receptor_systems,
            self._offsets,
            self._state_offsets,
            self._n_states_per_system,
            self._regional_overrides,
        ):
            n = system.n
            C_i = C_receptor[offset : offset + n]
            occ_i = occ[s_offset : s_offset + n_states]

            if not getattr(system, "is_regional", False):
                # The pre-P6.1 path, unchanged: one state block, one
                # activation, one set of coupled nodes.
                d_occ[s_offset : s_offset + n_states] = (
                    system.receptor_site.occupancy_rhs(occ_i, C_i)
                )

                activation = system.receptor_site.activation(occ_i)

                for node_name, gain in system.coupling.items():
                    idx = self.network.index[node_name]
                    drive[idx] += gain * activation

                for key, gain in system.connection_modulation.items():
                    factor = 1.0 + gain * activation
                    modulation_factors[key] = modulation_factors.get(key, 1.0) * factor
                continue

            # The P6.1 path: the SAME receptor site, one state block per
            # region, each block reading its own region's transmitter
            # concentration and driving its own region's node.
            site = system.receptor_site
            block = system.block
            for k, region in enumerate(system.regions):
                if overrides:
                    C_k = np.array(C_i, dtype=float)
                    for ligand_row, pool_rows in overrides:
                        C_k[ligand_row] = pools[pool_rows[k]]
                else:
                    C_k = C_i
                lo = s_offset + k * block
                sub = occ[lo : lo + block]
                d_occ[lo : lo + block] = site.occupancy_rhs(sub, C_k)

                # The receptor DENSITY factor: P4.2's `rho` in
                # `coupling = mechanism_gain * rho`, defaulting to 1.0.
                activation = site.activation(sub) * system.density[region]

                for node_name, gain in system.coupling.get(region, {}).items():
                    idx = self.network.index[node_name]
                    drive[idx] += gain * activation

                for key, gain in system.connection_modulation.get(region, {}).items():
                    factor = 1.0 + gain * activation
                    modulation_factors[key] = modulation_factors.get(key, 1.0) * factor

        connectivity = None
        if modulation_factors:
            connectivity = self.network.connectivity.copy()
            for (post_name, pre_name), factor in modulation_factors.items():
                i = self.network.index[post_name]
                j = self.network.index[pre_name]
                connectivity[i, j] = connectivity[i, j] * factor

        # The transporter layer. It contributes nothing to `drive` and
        # nothing to `connectivity`: its whole output is the transmitter
        # concentration it maintains, which the receptor layer above has
        # already read through `transmitter_sources` if anything asked for
        # it. Placed after the receptor loop so the ordering of the two
        # reads is explicit rather than incidental.
        d_transmitter = np.empty(self.m)
        cursor = 0
        for system, offset in zip(
            self.transporter_systems, self._transporter_offsets
        ):
            n = system.n
            # A cursor rather than the loop index, so that a transporter that
            # ever owns more than one pool cannot silently misalign the block.
            start, cursor = cursor, cursor + system.n_states
            # A transporter's drugs are plasma-driven exactly like a
            # receptor's, so they take the same brain-partition correction
            # (P1.5); the transporter ligands sit past every receptor ligand
            # in this array, so no `transmitter_sources` override can touch
            # them.
            C_i = C_receptor[offset : offset + n]
            d_transmitter[start:cursor] = system.transporter.transmitter_rhs(
                transmitter[start:cursor], C_i
            )

        # The projection layer (P6.1). Like the transporter layer it adds
        # nothing to `drive` and nothing to `connectivity`; unlike it, its
        # release term READS a circuit rate, which is the one place in this
        # right-hand side where information travels from the circuit layer
        # back into the concentration layer.
        d_projection = np.empty(self.p)
        cursor = 0
        for system, offset, source_rows in zip(
            self.projection_systems,
            self._projection_offsets,
            self._projection_source_rows,
        ):
            n = system.n
            start, cursor = cursor, cursor + system.n_states
            C_i = C_receptor[offset : offset + n]
            d_projection[start:cursor] = system.concentration_rhs(
                projection[start:cursor], rates[source_rows], C_i
            )

        d_rates = self.network.rhs(rates, drive, connectivity=connectivity)

        return np.concatenate(
            [d_pk, d_occ, d_transmitter, d_projection, d_rates]
        )

    def initial_pk_state(self, doses):
        """Assemble the PK region of y0 from one dose per ligand, by asking
        each ligand's own PK model to turn its dose into its own initial
        state (`pk.initial_state`). This is the only place dose semantics
        live in the engine, which is why the engine needs no branch on
        route, bioavailability or model order."""
        doses = np.asarray(doses, dtype=float)
        if doses.shape != (self.n_ligands,):
            raise ValueError(
                "doses must have length %d (one per ligand)" % self.n_ligands
            )
        blocks = [
            model.initial_state(dose)
            for model, dose in zip(self._ode_pk_models, doses)
        ]
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def initial_transmitter_state(self):
        """The transmitter region of y0: each transporter's drug-free basal
        extracellular concentration, in nM.

        Unlike every other region of the state vector this does NOT start at
        zero, and that is the point: there is transmitter in the cleft before
        any drug is given, so a transporter blocker has something to act on.
        It is also the one place where a "no drug" run is not a run in which
        every state is initially zero, which is why the check that a drug-free
        transporter run is exactly flat is a test rather than an assumption.
        """
        blocks = [
            system.transporter.initial_state() for system in self.transporter_systems
        ]
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def initial_projection_state(self):
        """The projection region of y0: each (system, region) pool's
        drug-free extracellular concentration in nM, with every source
        nucleus at its own resting firing rate.

        Like the transporter region and unlike everything else, this does not
        start at zero. Unlike the transporter region, only ONE region per
        system is a measured number: the rest are what the density map and
        the uptake kinetics predict (see `projection.ProjectionSystem`).
        """
        blocks = [system.initial_state() for system in self.projection_systems]
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def resting_occupancy_state(self):
        """The occupancy region of y0 with TONIC TRANSMITTER already bound.

        Zero everywhere except at ligands whose concentration comes from a
        transmitter pool (`transmitter_sources`), which start at the
        equilibrium occupancy their pool's drug-free concentration supports.

        Opt-in through `run(occupancy0="resting")`, so nothing that predates
        it moves. It exists because "the drug-free brain" and "the brain with
        every receptor empty" are not the same state once there is endogenous
        transmitter, and starting from the second one puts a start-up
        transient on every node that carries a tonically occupied receptor:
        the tonic compensation in `projection.compensate_baselines` subtracts
        a standing drive that the first integration steps have not yet
        produced. The transient is brief (occupancy equilibrates in seconds
        against a circuit relaxing over minutes) and it is exactly as large
        as the standing drive, which for the striatal D2 receptor is 2.13
        units, so it is not small.
        """
        from .projection import resting_occupancy

        pools = self._pools(
            self.initial_transmitter_state(), self.initial_projection_state()
        )
        blocks = []
        for index, system in enumerate(self.receptor_systems):
            site = system.receptor_site
            sources = getattr(system, "transmitter_sources", None)
            n_states = self._n_states_per_system[index]
            if not sources:
                blocks.append(np.zeros(n_states))
                continue
            positions = {lig.name: i for i, lig in enumerate(site.ligands)}
            if not getattr(system, "is_regional", False):
                offset = self._offsets[index]
                concentrations = np.zeros(system.n)
                for ligand_name, source in sources.items():
                    row = self._resolve_pool_row(system.name, ligand_name, source)
                    concentrations[positions[ligand_name]] = pools[row]
                blocks.append(resting_occupancy(site, concentrations))
                continue
            for region in system.regions:
                concentrations = np.zeros(system.n)
                for ligand_name, source in sources.items():
                    row = self._resolve_pool_row(
                        system.name, ligand_name, source, region=region
                    )
                    concentrations[positions[ligand_name]] = pools[row]
                blocks.append(resting_occupancy(site, concentrations))
        return np.concatenate(blocks) if blocks else np.zeros(0)

    def add_dose(self, y, doses):
        """Return a copy of state vector `y` with a further administration
        applied: each ligand's PK block gets that ligand's own
        `initial_state(dose)` ADDED to it.

        This is what a second dose partway through a run physically is in a
        linear PK model (superposition of a second administration onto
        whatever is already on board), and it is why the PK layer has to
        live in the state vector: for an extravascular route the new dose
        enters the depot, not the plasma, so "bump the concentration" would
        be the wrong edit. See `examples/fentanyl_reward_circuit.py` for the
        delayed-naloxone use.
        """
        y = np.array(y, dtype=float)
        doses = np.asarray(doses, dtype=float)
        if doses.shape != (self.n_ligands,):
            raise ValueError(
                "doses must have length %d (one per ligand)" % self.n_ligands
            )
        for model, (start, stop), dose in zip(
            self._ode_pk_models, self._pk_blocks, doses
        ):
            if dose != 0.0:
                y[start:stop] += model.initial_state(dose)
        return y

    def unpack_solution(self, sol):
        """Split a raw `solve_ivp` solution over this Simulation's state
        vector into the same dict `run()` returns. Exists so callers that
        drive `solve_ivp` themselves (mid-run dosing, in the examples) don't
        have to hand-reproduce the layout."""
        d, r, m, p = self.d, self.r, self.m, self.p
        plasma = sol.y[self.concentration_rows]
        return {
            "t": sol.t,
            # Unchanged meaning and unchanged values: the central-compartment
            # (plasma/blood) concentration.
            "concentrations": plasma,
            # Explicit alias, so a call site can say which compartment it
            # means instead of relying on the historical name.
            "plasma_concentrations": plasma,
            # What the receptor layer actually bound: unbound brain
            # concentration for ligands carrying a brain partition, and the
            # plasma array itself when none does.
            "receptor_concentrations": (
                plasma
                if self._all_receptor_factors_are_one
                else plasma * self.receptor_concentration_factors[:, None]
            ),
            "pk_states": sol.y[0:d],
            # One row per receptor STATE, in system order: per-ligand
            # occupancy for an ordinary site, and the full
            # bound/desensitized/blocked block for a desensitizing one. Use
            # `state_rows(i)` plus the site's `split_states` to read the
            # latter (P5.4).
            "occupancy": sol.y[d : d + r],
            # One row per transporter system, in nM, in the order the
            # systems were given: the extracellular transmitter
            # concentration each one maintains. Empty (shape (0, n_times))
            # when there are no transporter systems, so every pre-P5.8
            # caller sees the key and no change. THIS IS WHAT PHASE 6
            # CONSUMES; `transmitter_index` maps a system name to its row.
            "transmitter": sol.y[d + r : d + r + m],
            # One row per (projection system, target region), in nM, in
            # system order and then in each system's own target order: the
            # region-wise transmitter FIELD (P6.1). Empty (shape
            # (0, n_times)) when there are no projection systems, so every
            # pre-P6.1 caller sees the key and no change. `projection_index`
            # maps a (system name, region) pair to its row and
            # `projection_trace()` reads one by name.
            "projection": sol.y[d + r + m : d + r + m + p],
            "rates": sol.y[d + r + m + p : d + r + m + p + self.c],
        }

    #: Integrator used by `run()` unless overridden. Every result in this
    #: codebase before P5.5 was produced with RK45 and still is: `method` is
    #: additive and defaults to this.
    DEFAULT_METHOD = "RK45"

    def run(self, dose_concentrations=None, t_span=None, t_eval=None, rates0=None,
            doses=None, method=None, occupancy0=None):
        """Integrate the coupled system from one dose per ligand.

        `method` is passed straight to `solve_ivp` and defaults to
        `DEFAULT_METHOD` ("RK45"), so every pre-existing call is unchanged.
        It exists because P5.5 is the first module whose binding rate
        constants are MEASURED rather than illustrative: [Dravid2007]'s
        association rate for MK-801 at the NMDA channel site is
        0.234 nM^-1 min^-1, so a micromolar dose gives the occupancy ODE a
        pseudo-first-order rate of ~10^4 min^-1 while the circuit relaxes
        with tau = 1.5 min. That is four orders of magnitude of timescale
        separation in one state vector, i.e. a stiff system, and an explicit
        Runge-Kutta method pays for it in step count rather than in
        accuracy: the same 60-minute run takes ~125 s under RK45 and well
        under a second under LSODA, to the same answer. Passing
        `method="LSODA"` is therefore a statement about the integrator, not
        about the model, and a test asserts the two methods agree.

        `doses` (equivalently `dose_concentrations`, kept as the original
        name so no existing call site changes): one entry per ligand,
        flattened across all receptor systems in the same order as
        `self.receptor_systems` (each system's ligands in the order given to
        its `ReceptorSite`). See `dose_vector()` for a name-based way to
        build this without tracking that order by hand.

        Each entry is in the unit that ligand's PK model doses in: an
        achieved initial plasma concentration for `OneCompartmentPK` (the v0
        meaning, unchanged), or an administered dose in mg/kg for the
        dose-administered models in `pk.py`. In both cases 0 means "no
        drug", so dosed-vs-baseline comparisons are unit-agnostic.

        `occupancy0` (P6.1): `None` (the default, and every result before
        P6.1) starts every receptor empty. `"resting"` starts the ligands fed
        by a transmitter pool at the equilibrium occupancy that pool's
        drug-free concentration supports, which is what a drug-free brain
        actually is once there is endogenous transmitter; see
        `resting_occupancy_state`. An array is taken as the occupancy region
        of y0 directly.
        """
        if doses is not None and dose_concentrations is not None:
            raise TypeError("pass doses= or dose_concentrations=, not both")
        if doses is None:
            doses = dose_concentrations
        if doses is None:
            raise TypeError("run() requires doses= (a.k.a. dose_concentrations=)")
        if t_span is None:
            raise TypeError("run() requires t_span=")

        pk0 = self.initial_pk_state(doses)
        if occupancy0 is None:
            occ0 = np.zeros(self.r)
        elif isinstance(occupancy0, str):
            if occupancy0 != "resting":
                raise ValueError(
                    "occupancy0 must be None, 'resting', or an array; got %r"
                    % (occupancy0,)
                )
            occ0 = self.resting_occupancy_state()
        else:
            occ0 = np.asarray(occupancy0, dtype=float)
            if occ0.shape != (self.r,):
                raise ValueError(
                    "occupancy0 must have length %d (one per receptor state)"
                    % self.r
                )
        transmitter0 = self.initial_transmitter_state()
        projection0 = self.initial_projection_state()
        if rates0 is None:
            rates0 = np.zeros(self.c)
        y0 = np.concatenate(
            [pk0, occ0, transmitter0, projection0, np.asarray(rates0, dtype=float)]
        )

        sol = solve_ivp(
            self.rhs,
            t_span,
            y0,
            method=self.DEFAULT_METHOD if method is None else method,
            t_eval=t_eval,
            max_step=0.1,
        )
        if not sol.success:
            raise RuntimeError("integration failed: %s" % sol.message)

        # "concentrations" stays (n_ligands, n_times) as in v0 even when a PK
        # model carries extra states; "pk_states" is the full PK region,
        # (d, n_times), depot and peripheral compartments included, for
        # inspecting or validating the PK layer itself.
        return self.unpack_solution(sol)
