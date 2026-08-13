"""The transporter mechanism: clearance, not signalling (P5.8).

This module is to `monoamine_transporters.py` what `receptor.py` is to
`opioid.py`: the mechanism classes, with no drug and no citation in them.
The cited constants live next door.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes. So `km` is nM, `vmax` is nM/min, `clearance_rate` is min^-1, and a
drug concentration handed to any method here is nM.

Why a transporter is not a `ReceptorSite`
-----------------------------------------

Every target class in this codebase before P5.8 is a receptor, and every one
of them is the same shape: ligands compete for a finite pool, occupancy is a
state, and `activation = sum(occupancy_i * efficacy_i)` is a NUMBER THE
CIRCUIT READS. A transporter breaks all three.

1. **It has no efficacy and produces no signal.** Nothing downstream reads
   "DAT occupancy". Blocking DAT does not drive a node; it raises the
   extracellular dopamine concentration, and *that* is what receptors then
   bind. The output of a transporter is a concentration, not an activation.
2. **Its own substrate is the thing being modelled.** A receptor's ligands
   are drugs; a transporter's principal substrate is the transmitter itself,
   whose concentration is the state variable. The drugs are inhibitors of a
   process that acts on that state.
3. **Its measured constants are enzyme constants, not binding constants.**
   What is published for a transporter is `Km` and `Vmax` for transport of
   the natural substrate, and `Ki`/`IC50` for inhibition of that transport.
   Those are Michaelis-Menten quantities. Writing them into a `kon`/`koff`
   occupancy model would be a re-derivation with no measurement behind it.

So the state a `MonoamineTransporter` owns is one number, the extracellular
transmitter concentration, and its right-hand side is a mass balance::

    dC/dt = release + efflux(drugs) - uptake(C, drugs)

Why the inhibitors get no occupancy state, which is the same decision again
--------------------------------------------------------------------------

An uptake inhibitor does occupy the transporter, and one could give it a
`kon`/`koff` pair and a state. This module does not, for two reasons:

* Nothing published for these five drugs is a transporter association rate
  constant. Deriving `kon` and `koff` from a Ki, as the receptor modules do
  under their "derivation 3", buys a timescale that is invented rather than
  measured, and here it would be the ONLY timescale in the mechanism, since
  the inhibitor's occupancy has no other consequence.
* Transporter turnover is fast compared with everything else in this model.
  The uptake process itself relaxes the transmitter pool with a time
  constant of `Km/Vmax`, which for mouse striatal dopamine is 52 ms, more
  than three orders of magnitude below the circuit's 1.5 min. Assuming that
  inhibitor binding reaches quasi-equilibrium on the circuit's timescale is
  therefore the same approximation already implicit in using a
  Michaelis-Menten rate law at all.

The rapid-equilibrium competitive-inhibition form is then the textbook one:
an inhibitor at concentration `I` with inhibition constant `Ki` multiplies the
apparent `Km` by `1 + I/Ki`, and several inhibitors compose additively inside
that bracket because they compete for the same site as the substrate::

    a      = 1 + sum_i (I_i / Ki_i)
    uptake = Vmax * C / (Km * a + C)

`inhibition_factor()` is `a`.

Substrates are not blockers, and the difference is two terms
------------------------------------------------------------

A pure blocker (cocaine, methylphenidate, fluoxetine, atomoxetine) binds the
transporter and is not carried across the membrane. A substrate-type releaser
(amphetamine) is carried across, and that has a consequence a blocker's
mechanism cannot produce: transmitter is counter-transported OUT. The two
mechanisms are represented by two different pieces of a `TransporterLigand`:

    ki           -> the drug appears in `a`, raising the apparent Km.
                    EVERY ligand has this, releasers included: a substrate
                    competes with the transmitter for the same site.
    release_ec50 -> the drug ALSO adds an efflux term. ONLY a substrate has
                    this, and `is_substrate` is exactly `release_ec50 is not
                    None`.

so that::

    efflux = release_rate * capacity * sum_j (emax_j * S_j / (EC50_j + S_j))

`emax_j` is the substrate's releasing efficacy relative to a full substrate,
which is a measured quantity; `capacity` is how much efflux a full substrate
at saturation can drive relative to the transporter's own basal release rate,
which is not (see `monoamine_transporters.py`).

The steady state, which is the reason this shape was chosen
------------------------------------------------------------

With blockers only and a fixed release rate, the drug-free steady state `Cb`
satisfies `release = Vmax*Cb/(Km+Cb)`, and the inhibited steady state solves
to *exactly*

    C_ss = a * Cb

with `Km`, `Vmax` and `Cb` all cancelling. The fold-rise in extracellular
transmitter a pure uptake blocker produces is therefore `1 + I/Ki` and
NOTHING ELSE: it depends on one measured constant per drug and on no
illustrative one. `steady_state_fold()` returns it, and a test asserts the
cancellation against the integrated ODE rather than trusting the algebra.

That is worth stating plainly because it is unusual in this codebase: the
central quantitative prediction of the transporter layer carries no
illustrative parameter at all.

The first-order limit, and why it is a constructor and not a special case
-------------------------------------------------------------------------

`Km` and `Vmax` are separately measured for DAT and SERT. For NET what is
published (see `monoamine_transporters.py`) is a clearance HALF-LIFE, which
fixes the ratio `Vmax/Km` and says nothing about either alone. In the regime
this model runs in -- basal transmitter is 5 nM against a Km of 160 nM -- the
Michaelis-Menten law is within 3% of its own linear limit

    uptake = (Vmax/Km) * C / a

so `MonoamineTransporter.first_order(clearance_rate=...)` builds a
transporter from the measured ratio alone rather than from an invented
splitting of it into two numbers. The steady-state law above holds exactly in
this form too (it is the `Km >> C` case of the same cancellation), so nothing
is given up except the saturation behaviour at concentrations this model
never reaches.
"""

import numpy as np


class TransporterLigand(object):
    """A drug acting at a monoamine transporter.

    Parameters
    ----------
    name : str
        Ligand name; also the key `engine.Simulation.dose_vector` uses.
    ki : float or None
        nM. Inhibition constant for transport of the natural substrate. The
        drug raises the apparent Km by `1 + C/ki`. `None` means *no measured
        constant is available for this drug at this transporter*, which is
        NOT the same as "the drug does not act there": a `None` ligand
        contributes nothing and `monoamine_transporters.MISSING_KI` records
        why, so the gap is visible instead of being an implied infinity.
    release_ec50 : float or None
        nM. Half-maximal concentration for substrate-type release (reverse
        transport) of the transmitter. `None` -> this drug is a pure blocker.
    release_emax : float
        Releasing efficacy relative to a full substrate, i.e. the measured
        Emax of the release assay expressed as a fraction. Only meaningful
        when `release_ec50` is set.
    """

    def __init__(self, name, ki=None, release_ec50=None, release_emax=0.0):
        if ki is None and release_ec50 is None:
            raise ValueError(
                "ligand %r has neither an inhibition constant nor a release "
                "EC50, so it has no mechanism at this transporter" % (name,)
            )
        if ki is not None and ki <= 0.0:
            raise ValueError("ki must be positive, got %r for %r" % (ki, name))
        if release_ec50 is not None:
            if release_ec50 <= 0.0:
                raise ValueError(
                    "release_ec50 must be positive, got %r for %r"
                    % (release_ec50, name)
                )
            if release_emax <= 0.0:
                raise ValueError(
                    "%r has a release EC50 but no releasing efficacy; a "
                    "substrate that releases nothing is a pure blocker and "
                    "should be built with release_ec50=None" % (name,)
                )
        self.name = name
        self.ki = None if ki is None else float(ki)
        self.release_ec50 = None if release_ec50 is None else float(release_ec50)
        self.release_emax = float(release_emax)

    @property
    def is_substrate(self):
        """True for a substrate-type releaser, False for a pure blocker.

        This is the P5.8 distinction, and it is a property of the measurement
        rather than of the drug's class name: a substrate is a drug for which
        a release EC50 was measured in an assay that separates release from
        uptake inhibition.
        """
        return self.release_ec50 is not None

    @property
    def is_blocker(self):
        """True for a drug that inhibits uptake and is not transported."""
        return self.ki is not None and not self.is_substrate

    def inhibition_term(self, concentration):
        """`C / ki`, or 0.0 when this ligand has no measured `ki`."""
        if self.ki is None:
            return 0.0
        return float(concentration) / self.ki

    def release_fraction(self, concentration):
        """`emax * C / (EC50 + C)`, or 0.0 for a pure blocker."""
        if not self.is_substrate:
            return 0.0
        c = float(concentration)
        return self.release_emax * c / (self.release_ec50 + c)

    def __repr__(self):
        kind = "substrate" if self.is_substrate else "blocker"
        return "TransporterLigand(name=%r, %s, ki=%r)" % (self.name, kind, self.ki)


class MonoamineTransporter(object):
    """One transporter clearing one extracellular transmitter pool.

    Owns exactly ONE ODE state, the extracellular transmitter concentration
    in nM, which is the quantity Phase 6's projection systems consume (see
    `engine.TransporterSystem` and the module docstring of
    `monoamine_transporters.py`).

    Parameters
    ----------
    name : str
    transmitter : str
        The transmitter this transporter clears ("dopamine", "serotonin",
        "noradrenaline"). Used as the key a receptor system's
        `transmitter_sources` resolves against, and reported in results.
    ligands : sequence of TransporterLigand
        In a fixed order, which is also the order of this transporter's PK
        models and of its entries in the flat `doses` vector.
    km, vmax : float or None
        nM and nM/min. The Michaelis-Menten pair. Give both, or give neither
        and use `first_order()`.
    clearance_rate : float or None
        min^-1. `Vmax/Km`. Set by `first_order()`; mutually exclusive with
        the pair above.
    basal : float
        nM. The drug-free steady-state extracellular concentration, which
        fixes the release rate (see `basal_release_rate`).
    release_rate : float or None
        nM/min. Defaults to the value that makes the drug-free steady state
        equal `basal`. **This is the Phase 6 hand-off**: P6.1 replaces this
        constant with a term driven by the source nucleus's firing rate, and
        nothing else in the mechanism has to change.
    release_capacity : float
        How much efflux a full substrate at saturating concentration drives,
        as a multiple of `release_rate`. Not a measured quantity; see
        `monoamine_transporters.RELEASE_CAPACITY`.
    """

    #: One ODE state: the extracellular transmitter concentration.
    n_states = 1

    def __init__(
        self,
        name,
        transmitter,
        ligands,
        km=None,
        vmax=None,
        clearance_rate=None,
        basal=None,
        release_rate=None,
        release_capacity=0.0,
    ):
        michaelis = km is not None or vmax is not None
        if michaelis and clearance_rate is not None:
            raise ValueError(
                "give either the Michaelis-Menten pair (km, vmax) or a "
                "first-order clearance_rate, not both"
            )
        if michaelis and (km is None or vmax is None):
            raise ValueError("the Michaelis-Menten form needs both km and vmax")
        if not michaelis and clearance_rate is None:
            raise ValueError("a transporter needs (km, vmax) or clearance_rate")
        if basal is None or basal <= 0.0:
            raise ValueError(
                "basal must be a positive drug-free extracellular "
                "concentration in nM, got %r" % (basal,)
            )
        if release_capacity < 0.0:
            raise ValueError("release_capacity must be non-negative")

        self.name = name
        self.transmitter = transmitter
        self.ligands = list(ligands)
        self.n = len(self.ligands)
        self.km = None if km is None else float(km)
        self.vmax = None if vmax is None else float(vmax)
        self.clearance_rate = (
            None if clearance_rate is None else float(clearance_rate)
        )
        self.basal = float(basal)
        self.release_capacity = float(release_capacity)
        self.release_rate = (
            self.basal_release_rate() if release_rate is None else float(release_rate)
        )

        names = [lig.name for lig in self.ligands]
        if len(set(names)) != len(names):
            raise ValueError("duplicate ligand name at transporter %r" % (name,))

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def first_order(cls, name, transmitter, ligands, clearance_rate, **kwargs):
        """Build from a measured clearance RATE rather than from Km and Vmax.

        For a transporter whose published kinetics fix only the ratio
        `Vmax/Km` (a clearance half-life), which is NET's case. See the
        module docstring for why this is a constructor rather than a fudge.
        """
        return cls(
            name=name,
            transmitter=transmitter,
            ligands=ligands,
            clearance_rate=clearance_rate,
            **kwargs
        )

    @property
    def is_michaelis_menten(self):
        return self.km is not None

    @property
    def effective_clearance_rate(self):
        """min^-1. `Vmax/Km`, whichever form the transporter was built in.

        The quantity that sets how fast the transmitter pool relaxes, and the
        one thing the two forms have in common.
        """
        if self.is_michaelis_menten:
            return self.vmax / self.km
        return self.clearance_rate

    # -----------------------------------------------------------------
    # The mechanism
    # -----------------------------------------------------------------

    def basal_release_rate(self):
        """nM/min. The release rate whose drug-free steady state is `basal`.

        DERIVED, not assumed: it is what the mass balance requires given the
        measured uptake kinetics and the measured basal concentration, i.e.
        `Vmax*Cb/(Km+Cb)` in the Michaelis-Menten form and `k*Cb` in the
        first-order one. It is a release rate for the transmitter as a whole,
        not a claim about vesicular release per action potential.
        """
        if self.is_michaelis_menten:
            return self.vmax * self.basal / (self.km + self.basal)
        return self.clearance_rate * self.basal

    def inhibition_factor(self, concentrations):
        """`1 + sum_i C_i/Ki_i`: the factor the apparent Km is raised by.

        Substrates are included, because a substrate competes with the
        transmitter for the same site; what distinguishes them is the
        additional efflux term, not the absence of this one.
        """
        concentrations = np.asarray(concentrations, dtype=float)
        total = 1.0
        for lig, c in zip(self.ligands, concentrations):
            total += lig.inhibition_term(c)
        return total

    def apparent_km(self, concentrations):
        """nM, or None in the first-order form where there is no Km."""
        if not self.is_michaelis_menten:
            return None
        return self.km * self.inhibition_factor(concentrations)

    def uptake(self, transmitter, concentrations):
        """nM/min removed from the extracellular pool."""
        c = float(transmitter)
        if c <= 0.0:
            return 0.0
        a = self.inhibition_factor(concentrations)
        if self.is_michaelis_menten:
            return self.vmax * c / (self.km * a + c)
        return self.clearance_rate * c / a

    def release_drive(self, concentrations):
        """The dimensionless `sum_j emax_j * S_j/(EC50_j + S_j)`.

        Exactly zero when no substrate is present at nonzero concentration,
        which is what makes "a pure blocker cannot release transmitter" an
        identity of this code rather than a claim about its parameters.
        """
        concentrations = np.asarray(concentrations, dtype=float)
        total = 0.0
        for lig, c in zip(self.ligands, concentrations):
            total += lig.release_fraction(c)
        return total

    def efflux(self, concentrations):
        """nM/min added to the extracellular pool by substrate-driven
        reverse transport."""
        return self.release_rate * self.release_capacity * self.release_drive(
            concentrations
        )

    def transmitter_rhs(self, state, concentrations):
        """d[transmitter]/dt, shape (1,), for the shared state vector."""
        state = np.asarray(state, dtype=float)
        c = state[0]
        d = (
            self.release_rate
            + self.efflux(concentrations)
            - self.uptake(c, concentrations)
        )
        return np.array([d], dtype=float)

    def initial_state(self):
        """The drug-free extracellular concentration, shape (1,).

        A transporter's state does NOT start at zero: there is transmitter in
        the cleft before any drug is given. This is the one place where the
        transporter layer differs structurally from every receptor layer in
        this codebase, all of which start from an empty pool.
        """
        return np.array([self.basal], dtype=float)

    # -----------------------------------------------------------------
    # Analytic predictions
    # -----------------------------------------------------------------

    def steady_state_fold(self, concentrations):
        """The predicted steady-state transmitter concentration as a multiple
        of `basal`.

        With blockers only this is EXACTLY `inhibition_factor`, with Km, Vmax
        and basal all cancelling out (see the module docstring); the general
        form below adds the substrate's efflux. In the first-order form the
        expression is exact for any release rate; in the Michaelis-Menten
        form the efflux part is exact in the `basal << Km` regime the
        measured constants put this model in, and `steady_state()` solves the
        full quadratic when that matters.
        """
        a = self.inhibition_factor(concentrations)
        return a * (1.0 + self.release_capacity * self.release_drive(concentrations))

    def steady_state(self, concentrations):
        """nM. The exact steady-state extracellular concentration.

        Solves `release + efflux = uptake` rather than assuming the linear
        regime, so that a test can check `steady_state_fold` against it and
        report the real gap instead of asserting the approximation.
        """
        total_in = self.release_rate + self.efflux(concentrations)
        a = self.inhibition_factor(concentrations)
        if not self.is_michaelis_menten:
            return total_in * a / self.clearance_rate
        if total_in >= self.vmax:
            return float("inf")  # uptake cannot keep up at any concentration
        return total_in * self.km * a / (self.vmax - total_in)

    def __repr__(self):
        return "MonoamineTransporter(name=%r, transmitter=%r, n_ligands=%d)" % (
            self.name,
            self.transmitter,
            self.n,
        )
