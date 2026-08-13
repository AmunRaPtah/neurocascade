import numpy as np


class Ligand:
    """A drug competing for a receptor site.

    efficacy: 1.0 = full agonist, 0.0 = neutral antagonist (occupies but does
    not activate), fractional = partial agonist, negative = inverse agonist.
    kon/koff set the association/dissociation rate constants; affinity is
    koff/kon (lower = tighter binding).
    """

    def __init__(self, name, kon, koff, efficacy):
        self.name = name
        self.kon = kon
        self.koff = koff
        self.efficacy = efficacy


class ReceptorSite:
    """A receptor population competitively bound by one or more ligands.

    State is the fractional occupancy owned by each ligand (they share one
    finite pool of receptor, so occupancies sum to at most 1). This is what
    lets an antagonist (e.g. naloxone) displace an agonist (e.g. fentanyl)
    already bound at the same site.

    Two counts, equal here and deliberately separated so a subclass can
    make them differ (see `DesensitizingReceptorSite`):

      `n`        how many LIGANDS the site carries. This is what the engine
                 sizes the per-ligand PK models and the `doses` vector by.
      `n_states` how many ODE STATES the site owns inside the shared state
                 vector. Two states per ligand, or an extra pool, is a
                 receptor mechanism, not extra drugs.
    """

    def __init__(self, name, ligands):
        self.name = name
        self.ligands = list(ligands)
        self.n = len(self.ligands)
        #: One state per ligand for the plain two-state (free/bound) site.
        self.n_states = self.n

    def occupancy_rhs(self, occupancy, concentrations):
        occupancy = np.asarray(occupancy, dtype=float)
        concentrations = np.asarray(concentrations, dtype=float)
        free_fraction = 1.0 - occupancy.sum()
        d_occ = np.empty(self.n)
        for i, lig in enumerate(self.ligands):
            d_occ[i] = lig.kon * concentrations[i] * free_fraction - lig.koff * occupancy[i]
        return d_occ

    def activation(self, occupancy):
        return sum(o * lig.efficacy for o, lig in zip(occupancy, self.ligands))


class DesensitizingReceptorSite(ReceptorSite):
    """A receptor that also has a bound-but-NON-CONDUCTING state (P5.4).

    Why this exists at all
    ----------------------

    `ReceptorSite` above is a two-state model: a receptor is either free or
    bound, and bound receptor contributes `efficacy` to activation. That is
    adequate for a G-protein-coupled receptor on this codebase's timescale,
    and every receptor system before P5.4 is one. It cannot represent a
    nicotinic acetylcholine receptor, because the defining property of
    nicotinic pharmacology is that an agonist drives the receptor into a
    third state that is still AGONIST-OCCUPIED and no longer CONDUCTING
    (Katz & Thesleff, 1957). In a two-state model "occupied" and "active"
    are the same variable, so there is no way to write that down: raising
    the agonist concentration can only ever raise activation.

    The cycle implemented here, per orthosteric ligand `i`:

        free --kon_i*C_i--> bound_i --k_des_i--> desensitized_i
        bound_i --koff_i--> free
        desensitized_i --k_rec_i--> bound_i

    i.e.::

        free    = 1 - sum(bound) - sum(desensitized)
        d bound_i/dt        = kon_i*C_i*free - koff_i*bound_i
                              - k_des_i*bound_i + k_rec_i*desensitized_i
        d desensitized_i/dt = k_des_i*bound_i - k_rec_i*desensitized_i

    Only `bound` contributes to `activation`; `desensitized` occupies the
    pool (so it still competes) and contributes nothing. Ligand does not
    dissociate directly from the desensitized state, which is the standard
    Katz-Thesleff cyclic arrangement and is what makes the desensitized
    state a HIGH-AFFINITY state: at equilibrium the apparent affinity of
    the whole site for ligand `i` is `Kd_i / (1 + k_des_i/k_rec_i)` rather
    than `Kd_i`. See `apparent_affinity_factor`.

    Set `k_des_i = 0` for a ligand that should not desensitize the receptor
    (a competitive antagonist occupies the orthosteric site but does not
    drive the conformational change); its desensitized state then stays at
    zero for all time.

    Channel blockers
    ----------------

    `channel_blockers` is an optional second, INDEPENDENT ligand pool for
    non-competitive open-channel blockers (mecamylamine at a neuronal
    nicotinic receptor). They do not compete for the orthosteric pool --
    they bind in the channel lumen, a structurally distinct site -- so
    modelling them as extra entries in the competitive pool would assert a
    mechanism the measurements contradict. Instead they own their own
    shared pool and scale activation multiplicatively::

        d blocked_j/dt = konb_j*Cb_j*(1 - sum(blocked)) - koffb_j*blocked_j
        activation     = (1 - sum(blocked)) * sum(bound_i * efficacy_i)

    which reproduces exactly what a functional IC50 for non-competitive
    inhibition measures (activation halved when the blocker is at its Kd)
    and deliberately does not represent the use- and voltage-dependence of
    real open-channel block.

    State layout (length `n_states = 2*n_orthosteric + n_blockers`)::

        [bound_0..bound_{p-1}, desens_0..desens_{p-1}, blocked_0..blocked_{b-1}]

    Ligand/concentration layout (length `n = n_orthosteric + n_blockers`),
    which is also the order of this site's PK models and `doses` entries::

        [orthosteric_0..orthosteric_{p-1}, blocker_0..blocker_{b-1}]
    """

    def __init__(self, name, ligands, k_des, k_rec, channel_blockers=None):
        self.name = name
        self.orthosteric_ligands = list(ligands)
        self.channel_blockers = list(channel_blockers or [])
        if not self.orthosteric_ligands:
            raise ValueError("a desensitizing site needs at least one orthosteric ligand")
        self.ligands = self.orthosteric_ligands + self.channel_blockers
        self.n = len(self.ligands)
        self.n_orthosteric = len(self.orthosteric_ligands)
        self.n_blockers = len(self.channel_blockers)
        self.n_states = 2 * self.n_orthosteric + self.n_blockers
        self.k_des = self._as_rate_vector(k_des, "k_des")
        self.k_rec = self._as_rate_vector(k_rec, "k_rec")
        if np.any(self.k_des < 0) or np.any(self.k_rec < 0):
            raise ValueError("k_des and k_rec must be non-negative rate constants")
        if np.any((self.k_des > 0) & (self.k_rec <= 0)):
            raise ValueError(
                "a ligand that desensitizes the receptor (k_des > 0) needs a "
                "positive recovery rate k_rec, or the receptor would never "
                "come back"
            )

    def _as_rate_vector(self, value, label):
        """Accept one rate for the whole site, or one per orthosteric ligand."""
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size == 1:
            return np.full(self.n_orthosteric, float(arr[0]))
        if arr.size != self.n_orthosteric:
            raise ValueError(
                "%s must be a scalar or one value per orthosteric ligand "
                "(%d), got %d" % (label, self.n_orthosteric, arr.size)
            )
        return arr.astype(float)

    def split_states(self, state):
        """(bound, desensitized, blocked) views onto one state block."""
        state = np.asarray(state, dtype=float)
        p = self.n_orthosteric
        return state[0:p], state[p : 2 * p], state[2 * p : 2 * p + self.n_blockers]

    def free_fraction(self, state):
        """Fraction of the receptor pool that is neither bound nor desensitized."""
        bound, desens, _ = self.split_states(state)
        return 1.0 - bound.sum() - desens.sum()

    def desensitized_fraction(self, state):
        _, desens, _ = self.split_states(state)
        return float(desens.sum())

    def blocked_fraction(self, state):
        _, _, blocked = self.split_states(state)
        return float(blocked.sum())

    def apparent_affinity_factor(self):
        """`1 + k_des/k_rec` per orthosteric ligand.

        The factor by which trapping ligand in the desensitized state
        tightens the site's apparent equilibrium affinity: total bound
        (conducting + desensitized) is half-maximal at `Kd / factor`, not at
        `Kd`. This is the model's prediction for the gap between an
        activation EC50 and a desensitization IC50 measured on the same
        receptor, and P5.4 compares it against measured values rather than
        fitting it to them.
        """
        return 1.0 + self.k_des / self.k_rec

    def occupancy_rhs(self, state, concentrations):
        state = np.asarray(state, dtype=float)
        concentrations = np.asarray(concentrations, dtype=float)
        p, b = self.n_orthosteric, self.n_blockers
        bound, desens, blocked = self.split_states(state)

        free = 1.0 - bound.sum() - desens.sum()
        d_state = np.empty(self.n_states)
        for i, lig in enumerate(self.orthosteric_ligands):
            association = lig.kon * concentrations[i] * free
            to_desens = self.k_des[i] * bound[i]
            from_desens = self.k_rec[i] * desens[i]
            d_state[i] = association - lig.koff * bound[i] - to_desens + from_desens
            d_state[p + i] = to_desens - from_desens

        if b:
            free_channel = 1.0 - blocked.sum()
            for j, lig in enumerate(self.channel_blockers):
                d_state[2 * p + j] = (
                    lig.kon * concentrations[p + j] * free_channel
                    - lig.koff * blocked[j]
                )
        return d_state

    def activation(self, state):
        bound, _, blocked = self.split_states(state)
        conducting = sum(
            o * lig.efficacy for o, lig in zip(bound, self.orthosteric_ligands)
        )
        if self.n_blockers:
            conducting = conducting * (1.0 - blocked.sum())
        return conducting
