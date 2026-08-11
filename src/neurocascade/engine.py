import numpy as np
from scipy.integrate import solve_ivp


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
    ligands respectively (d = r = sum(n_i); reduces to the original
    [C_1..C_d, occ_1..occ_d, r_1..r_n] layout when k == 1):

        y = [C_1^(1)..C_{n_1}^(1), C_1^(2)..C_{n_2}^(2), ...,   # all systems'
             occ_1^(1)..occ_{n_1}^(1), occ_1^(2)..occ_{n_2}^(2), ...,  # concentrations,
             r_1, ..., r_n,                                     # then occupancies,
             a_1, ..., a_m]                                     # then shared rates,
                                                                 # then PK aux states

    i.e. every system's concentrations, in system order, then every
    system's occupancies, in the same order, then one shared set of circuit
    firing rates, and finally any auxiliary PK states (depot and peripheral
    compartment amounts, in nmol) owned by dose-driven PK models, in the
    same flattened ligand order. Models that own no auxiliary state
    (`OneCompartmentPK`, and `OneCompartmentDosePK` on the IV route)
    contribute nothing to that trailing block, so a simulation built
    entirely from them has bit-for-bit the original state vector.

    The aux block is appended *after* the rates rather than interleaved with
    the concentrations so that the index of every pre-existing quantity is
    unchanged: adding a two-compartment PK model to one ligand must not
    move another ligand's occupancy or a circuit node's rate within `y`.

    `dose_concentrations` passed to `run()` must be given in that same
    flattened per-system ligand order (see `dose_vector()` for a name-based
    helper that builds it for you). It is only meaningful for PK models that
    are dosed as an achieved bolus concentration; a model that carries its
    own administered dose (`OneCompartmentDosePK`, `TwoCompartmentDosePK`)
    supplies its own initial state and rejects a nonzero entry here.

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

        self.d = sum(self._n_per_system)
        self.r = self.d
        self.c = network.n

        # Flattened per-ligand PK bookkeeping: one entry per ligand across
        # all systems, in the same order as the concentration block, giving
        # that ligand's PK model and the slice of the trailing auxiliary
        # block it owns. Resolved once here rather than in `rhs`, which runs
        # thousands of times per `run()`.
        self._pk_entries = []
        aux_offset = 0
        for system in receptor_systems:
            for pk in system.pk_models:
                n_aux = int(getattr(pk, "n_aux", 0))
                self._pk_entries.append((pk, aux_offset, n_aux))
                aux_offset += n_aux
        self.a = aux_offset

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

    def dose_vector(self, **concentration_by_ligand_name):
        """Build a flat `dose_concentrations` array (for `run()`) from
        ligand-name -> concentration kwargs; ligands not mentioned default
        to 0. Mainly useful once there's more than one receptor system and
        the flat per-system ligand ordering is easy to get wrong by hand.
        """
        doses = np.zeros(self.d)
        offset = 0
        seen = set()
        for system in self.receptor_systems:
            for lig in system.receptor_site.ligands:
                if lig.name in concentration_by_ligand_name:
                    doses[offset] = concentration_by_ligand_name[lig.name]
                    seen.add(lig.name)
                offset += 1
        unknown = set(concentration_by_ligand_name) - seen
        if unknown:
            raise ValueError("unknown ligand name(s): %s" % sorted(unknown))
        return doses

    def _unpack(self, y):
        d, r, c = self.d, self.r, self.c
        C = y[0:d]
        occ = y[d : d + r]
        rates = y[d + r : d + r + c]
        aux = y[d + r + c : d + r + c + self.a]
        return C, occ, rates, aux

    def _pk_rhs(self, C, aux, dC, d_aux):
        """Fill in every ligand's dC/dt and its auxiliary PK derivatives.

        Accepts both the original PK protocol (a bare `rhs(C)`, which owns
        no auxiliary state) and the dose-driven one
        (`concentration_rhs(C, aux) -> (dC, d_aux)`), so a scenario can mix
        bolus-concentration and administered-dose ligands in one
        simulation, which is what a reversal experiment needs: an agonist
        given as a real mg/kg dose and an antagonist titrated to a target
        concentration.
        """
        for i, (pk, aux_offset, n_aux) in enumerate(self._pk_entries):
            concentration_rhs = getattr(pk, "concentration_rhs", None)
            if concentration_rhs is None:
                dC[i] = pk.rhs(C[i])
                continue
            aux_i = aux[aux_offset : aux_offset + n_aux]
            dC_i, d_aux_i = concentration_rhs(C[i], aux_i)
            dC[i] = dC_i
            if n_aux:
                d_aux[aux_offset : aux_offset + n_aux] = d_aux_i

    def rhs(self, t, y):
        C, occ, rates, aux = self._unpack(y)

        dC = np.empty(self.d)
        d_occ = np.empty(self.r)
        d_aux = np.empty(self.a)
        drive = self.network.baseline_drive()
        modulation_factors = {}  # (post_name, pre_name) -> composed factor

        self._pk_rhs(C, aux, dC, d_aux)

        for system, offset in zip(self.receptor_systems, self._offsets):
            n = system.n
            C_i = C[offset : offset + n]
            occ_i = occ[offset : offset + n]

            d_occ[offset : offset + n] = system.receptor_site.occupancy_rhs(occ_i, C_i)

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

        return np.concatenate([dC, d_occ, d_rates, d_aux])

    def initial_pk_state(self, dose_concentrations=None):
        """The t=0 concentration vector and auxiliary PK vector implied by
        each ligand's PK model, given the optional bolus concentrations.

        A bolus-dosed model (`OneCompartmentPK`) takes its entry of
        `dose_concentrations` as its initial concentration, exactly as
        before. A dose-driven model computes its own initial state from the
        administered dose it was constructed with, and rejects a nonzero
        entry here rather than silently letting one of the two dosing
        statements win.
        """
        if dose_concentrations is None:
            bolus = np.zeros(self.d)
        else:
            bolus = np.asarray(dose_concentrations, dtype=float)
            if bolus.shape != (self.r,):
                raise ValueError("dose_concentrations must have length %d" % self.r)

        C0 = np.zeros(self.d)
        aux0 = np.zeros(self.a)
        for i, (pk, aux_offset, n_aux) in enumerate(self._pk_entries):
            initial_state = getattr(pk, "initial_state", None)
            if initial_state is None:
                C0[i] = bolus[i]
                continue
            C0_i, aux0_i = initial_state(bolus[i])
            C0[i] = C0_i
            if n_aux:
                aux0[aux_offset : aux_offset + n_aux] = aux0_i
        return C0, aux0

    def run(self, dose_concentrations=None, t_span=None, t_eval=None, rates0=None):
        """dose_concentrations: initial plasma concentration per ligand
        (bolus at t=0), flattened across all receptor systems in the same
        order as `self.receptor_systems` (each system's ligands in the
        order given to its `ReceptorSite`). See `dose_vector()` for a
        name-based way to build this without tracking that order by hand.

        It may be omitted entirely when every ligand's PK model carries its
        own administered dose, since in that case there is no achieved
        concentration for the caller to state."""
        if t_span is None:
            raise TypeError("run() requires t_span")
        C0, aux0 = self.initial_pk_state(dose_concentrations)
        occ0 = np.zeros(self.r)
        if rates0 is None:
            rates0 = np.zeros(self.c)
        y0 = np.concatenate([C0, occ0, np.asarray(rates0, dtype=float), aux0])

        sol = solve_ivp(
            self.rhs, t_span, y0, method="RK45", t_eval=t_eval, max_step=0.1
        )
        if not sol.success:
            raise RuntimeError("integration failed: %s" % sol.message)

        d, r, c = self.d, self.r, self.c
        return {
            "t": sol.t,
            "concentrations": sol.y[0:d],
            "occupancy": sol.y[d : d + r],
            "rates": sol.y[d + r : d + r + c],
            "pk_aux": sol.y[d + r + c : d + r + c + self.a],
        }
