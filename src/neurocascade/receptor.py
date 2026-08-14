import numpy as np


def ligand_from_ki(name, ki_nM, kon, efficacy):
    """A `Ligand` built from an equilibrium affinity and one association rate.

    THE CONVERSION, AND THE TWO ASSUMPTIONS IN IT
    ---------------------------------------------

    This engine's occupancy ODE needs a rate pair::

        d occ/dt = kon * C * (1 - sum(occ)) - koff * occ

    A binding table publishes one number per (ligand, target): an inhibition
    constant `Ki` in nM. Two things have to be assumed to get from one to
    two, and they are not the same kind of assumption.

    **1. `Kd = Ki`.** A competition binding assay measures the concentration
    of unlabelled ligand that halves specific radioligand binding, converted
    to `Ki` by Cheng-Prusoff. For a reversible competitive orthosteric
    ligand that `Ki` *is* the equilibrium dissociation constant of the
    ligand-receptor complex, so this is the standard reading of the datum,
    not a modelling liberty. It fails for a ligand that is not competitive at
    the labelled site (an allosteric modulator, an open-channel blocker), and
    a Ki table does not say which ligands those are -- see
    `fingerprint.py`, which makes the caller declare the pool.

    **2. `koff = kon * Ki`, with `kon` supplied rather than measured.** This
    is the load-bearing one and it is stated here rather than left implicit:

        THE Ki FIXES THE RATIO koff/kon. IT FIXES NEITHER RATE. SPLITTING IT
        INTO A PAIR REQUIRES ONE MORE NUMBER, AND THAT NUMBER IS `kon`.

    What that does and does not cost:

    * The ligand's *affinity* is reproduced EXACTLY. `koff/kon` is `Ki` by
      construction, so every equilibrium quantity -- fractional occupancy at
      a given concentration, the competitive ratio between two ligands
      sharing a pool, an IC50 -- is the measured one and carries no
      assumption beyond (1).
    * The ligand's *approach timescale* is entirely `kon`'s. How fast
      occupancy equilibrates after a concentration step, and therefore
      whether binding is fast or slow compared with the circuit's own
      relaxation, is set by the supplied `kon` and by nothing in the
      binding table.

    So a `kon` chosen for the whole site makes every ligand at that site
    equilibrate on a comparable timescale while keeping each ligand's
    measured affinity intact. That is derivation 3 in this codebase's
    four-way taxonomy (see `opioid.py`'s docstring), and it is what
    `acetylcholine.py`, `glutamate.py`, `cannabinoid.py`, `opioid.py`,
    `serotonin.py`, `noradrenaline.py`, `histamine.py`, `adenosine.py` and
    `scenarios.gaba_a_benzodiazepine_ligands_calibrated` each already do by
    hand. This function is that arithmetic, in one place, with the
    assumption written down next to it.

    Where `kon` comes from is NOT this function's business and must not be:
    a site's own measured radioligand association rate, a shared
    illustrative value, or a value borrowed from a neighbouring subtype are
    three different provenances with three different standings, and
    `fingerprint.TargetSpec` carries which one applies to each target so a
    caller can read it off rather than infer it.

    Parameters
    ----------
    name : str
    ki_nM : float
        The equilibrium affinity, in nM. Must be positive and finite: a
        censored ">10000 nM" cell is a bound, not a measurement, and the
        caller has to decide what to do with it (`opioid.is_censored` and
        `monoamine_transporters.MISSING_KI` are this codebase's two existing
        answers, and they differ).
    kon : float
        nM^-1 min^-1. See above.
    efficacy : float
        Fraction of the site's reference full agonist's Emax. NOT derivable
        from a Ki: an affinity says how tightly a ligand binds and says
        nothing about what the bound receptor then does. It has to come from
        a functional assay, and the caller supplies it.
    """
    ki_nM = float(ki_nM)
    kon = float(kon)
    if not np.isfinite(ki_nM) or ki_nM <= 0.0:
        raise ValueError(
            "ki_nM must be a positive, finite affinity in nM; got %r for %r"
            % (ki_nM, name)
        )
    if not np.isfinite(kon) or kon <= 0.0:
        raise ValueError(
            "kon must be a positive, finite association rate constant in "
            "nM^-1 min^-1; got %r for %r" % (kon, name)
        )
    return Ligand(name=name, kon=kon, koff=kon * ki_nM, efficacy=float(efficacy))


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


class ChannelBlockingReceptorSite(ReceptorSite):
    """A receptor with a competitive pool AND an independent blocker pool.

    Two structurally distinct sites, so two pools (P5.5):

      * the ORTHOSTERIC pool, exactly `ReceptorSite`: one finite pool of
        receptor that every orthosteric ligand competes for, so an
        orthosteric antagonist displaces the agonist;
      * the CHANNEL-BLOCKER pool, a second finite pool of the same receptors
        seen from inside the ion channel. A non-competitive open-channel
        blocker does not displace the agonist -- that is the defining
        measurement -- so giving it a seat in the competitive pool would
        assert a mechanism the data contradict.

    ::

        d bound_i/dt   = kon_i*C_i*(1 - sum(bound)) - koff_i*bound_i
        d blocked_j/dt = konb_j*Cb_j*(1 - sum(blocked)) - koffb_j*blocked_j
        activation     = (1 - sum(blocked)) * sum(bound_i * efficacy_i)

    which is what a functional IC50 for non-competitive inhibition measures
    (activation halved when the blocker sits at its Kd) and deliberately not
    the use- and voltage-dependence of real open-channel block, for which
    this engine has neither a membrane potential nor a channel state.

    State layout (length `n_states == n == n_orthosteric + n_blockers`)::

        [bound_0..bound_{p-1}, blocked_0..blocked_{b-1}]

    which is also the ligand/concentration/`doses` order.

    Relationship to `DesensitizingReceptorSite` below: that class is this
    mechanism plus a desensitized state per orthosteric ligand, introduced
    by P5.4 because nicotinic receptors need both at once. The two are kept
    as separate classes rather than one inheriting from the other because
    their state layouts interleave differently (the desensitized states sit
    between the bound and blocked blocks), so sharing an implementation
    would buy nothing but a shared indexing bug.
    """

    def __init__(self, name, ligands, channel_blockers=None):
        self.name = name
        self.orthosteric_ligands = list(ligands)
        self.channel_blockers = list(channel_blockers or [])
        if not self.orthosteric_ligands:
            raise ValueError("a channel-blocking site needs at least one orthosteric ligand")
        self.ligands = self.orthosteric_ligands + self.channel_blockers
        self.n = len(self.ligands)
        self.n_orthosteric = len(self.orthosteric_ligands)
        self.n_blockers = len(self.channel_blockers)
        self.n_states = self.n

    def split_states(self, state):
        """(bound, blocked) views onto one state block."""
        state = np.asarray(state, dtype=float)
        p = self.n_orthosteric
        return state[0:p], state[p : p + self.n_blockers]

    def free_fraction(self, state):
        bound, _ = self.split_states(state)
        return 1.0 - bound.sum()

    def blocked_fraction(self, state):
        _, blocked = self.split_states(state)
        return float(blocked.sum())

    def occupancy_rhs(self, state, concentrations):
        state = np.asarray(state, dtype=float)
        concentrations = np.asarray(concentrations, dtype=float)
        p = self.n_orthosteric
        bound, blocked = self.split_states(state)

        d_state = np.empty(self.n_states)
        free = 1.0 - bound.sum()
        for i, lig in enumerate(self.orthosteric_ligands):
            d_state[i] = lig.kon * concentrations[i] * free - lig.koff * bound[i]

        if self.n_blockers:
            free_channel = 1.0 - blocked.sum()
            for j, lig in enumerate(self.channel_blockers):
                d_state[p + j] = (
                    lig.kon * concentrations[p + j] * free_channel
                    - lig.koff * blocked[j]
                )
        return d_state

    def activation(self, state):
        bound, blocked = self.split_states(state)
        conducting = sum(
            o * lig.efficacy for o, lig in zip(bound, self.orthosteric_ligands)
        )
        if self.n_blockers:
            conducting = conducting * (1.0 - blocked.sum())
        return conducting


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
