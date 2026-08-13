import numpy as np
from scipy.integrate import solve_ivp

from .pk import as_ode_pk_model


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
    """

    def __init__(self, pk_models, receptor_site, coupling, connection_modulation=None, name=None):
        if len(pk_models) != receptor_site.n:
            raise ValueError("need one PK model per ligand on the receptor site")
        self.pk_models = list(pk_models)
        self.receptor_site = receptor_site
        self.coupling = dict(coupling)
        self.connection_modulation = dict(connection_modulation or {})
        self.name = name or receptor_site.name
        self.n = receptor_site.n
        self.n_states = int(getattr(receptor_site, "n_states", receptor_site.n))

    def __repr__(self):
        return "ReceptorSystem(name=%r, n_ligands=%d)" % (self.name, self.n)


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

    State vector layout, generalizing to k receptor systems with n_1..n_k
    ligands respectively (r = sum(n_i) ligands; reduces to the original
    [C_1..C_d, occ_1..occ_d, r_1..r_n] layout when k == 1 and every ligand's
    PK model owns a single state):

        y = [<PK states of every ligand, in flattened ligand order>,  # d of them
             <receptor states of system 1>, <system 2>, ...,         # r of them
             r_1, ..., r_n]                                          # shared rates

    i.e. every ligand's PK state block, in system order, then every system's
    receptor-state block, in the same order, then one shared set of circuit
    firing rates. `doses` passed to `run()` must be given in that same
    flattened per-system LIGAND order, one entry per ligand (see
    `dose_vector()` for a name-based helper that builds it for you).

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
    """

    def __init__(
        self,
        network=None,
        receptor_systems=None,
        pk_models=None,
        receptor_site=None,
        coupling=None,
        connection_modulation=None,
    ):
        if network is None:
            raise TypeError("Simulation requires a network")

        legacy_kwargs_given = (
            pk_models is not None
            or receptor_site is not None
            or coupling is not None
            or connection_modulation is not None
        )
        if receptor_systems is None:
            if pk_models is None or receptor_site is None or coupling is None:
                raise TypeError(
                    "Simulation needs either receptor_systems=[...] or the "
                    "single-receptor-system trio pk_models=/receptor_site=/"
                    "coupling= (connection_modulation= optional)"
                )
            receptor_systems = [
                ReceptorSystem(pk_models, receptor_site, coupling, connection_modulation)
            ]
        elif legacy_kwargs_given:
            raise TypeError(
                "pass either receptor_systems=[...] or the legacy single-"
                "system kwargs (pk_models=/receptor_site=/coupling=/"
                "connection_modulation=), not both"
            )

        receptor_systems = list(receptor_systems)
        if not receptor_systems:
            raise ValueError("need at least one receptor system")

        self.network = network
        self.receptor_systems = receptor_systems
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

        #: Number of ligands across all systems: the length of `doses`.
        self.n_ligands = sum(self._n_per_system)
        #: Number of receptor STATES across all systems: the length of the
        #: occupancy region of y. Equal to `n_ligands` unless a
        #: desensitizing site is present.
        self.r = sum(self._n_states_per_system)
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
        for system in receptor_systems:
            for model in system.pk_models:
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

    def _unpack(self, y):
        """Split y into (PK states, occupancies, rates). The first return
        value is the whole PK region, which is per-ligand *concentration*
        only when every PK model owns a single state; use
        `self.concentration_rows` to pull the per-ligand concentrations out
        of it in the general case."""
        d, r = self.d, self.r
        pk_state = y[0:d]
        occ = y[d : d + r]
        rates = y[d + r : d + r + self.c]
        return pk_state, occ, rates

    def rhs(self, t, y):
        pk_state, occ, rates = self._unpack(y)

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

        d_occ = np.empty(self.r)
        drive = self.network.baseline_drive()
        modulation_factors = {}  # (post_name, pre_name) -> composed factor

        for system, offset, s_offset, n_states in zip(
            self.receptor_systems,
            self._offsets,
            self._state_offsets,
            self._n_states_per_system,
        ):
            n = system.n
            C_i = C_receptor[offset : offset + n]
            occ_i = occ[s_offset : s_offset + n_states]

            d_occ[s_offset : s_offset + n_states] = system.receptor_site.occupancy_rhs(
                occ_i, C_i
            )

            activation = system.receptor_site.activation(occ_i)

            for node_name, gain in system.coupling.items():
                idx = self.network.index[node_name]
                drive[idx] += gain * activation

            for key, gain in system.connection_modulation.items():
                factor = 1.0 + gain * activation
                modulation_factors[key] = modulation_factors.get(key, 1.0) * factor

        connectivity = None
        if modulation_factors:
            connectivity = self.network.connectivity.copy()
            for (post_name, pre_name), factor in modulation_factors.items():
                i = self.network.index[post_name]
                j = self.network.index[pre_name]
                connectivity[i, j] = connectivity[i, j] * factor

        d_rates = self.network.rhs(rates, drive, connectivity=connectivity)

        return np.concatenate([d_pk, d_occ, d_rates])

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
        d, r = self.d, self.r
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
            "rates": sol.y[d + r : d + r + self.c],
        }

    #: Integrator used by `run()` unless overridden. Every result in this
    #: codebase before P5.5 was produced with RK45 and still is: `method` is
    #: additive and defaults to this.
    DEFAULT_METHOD = "RK45"

    def run(self, dose_concentrations=None, t_span=None, t_eval=None, rates0=None,
            doses=None, method=None):
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
        occ0 = np.zeros(self.r)
        if rates0 is None:
            rates0 = np.zeros(self.c)
        y0 = np.concatenate([pk0, occ0, np.asarray(rates0, dtype=float)])

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
