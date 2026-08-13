"""P6.1: the neuromodulatory projection-system mechanism.

This module is to `projection_systems.py` what `receptor.py` is to
`opioid.py` and `transport.py` is to `monoamine_transporters.py`: the
mechanism classes, with no drug, no nucleus and no citation in them. The
cited constants live next door.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, firing rate the dimensionless Wilson-Cowan population rate in
[0, 1] that `network.CircuitNetwork` integrates.

What this replaces
------------------

Every receptor system before P6.1 is one `ReceptorSite` whose activation is
ONE number added to ONE node's drive:

    drive[node] += gain * activation(occupancy)

That is a receptor sitting on a designated node. It is the right shape for a
locally released transmitter and the wrong shape for a neuromodulator,
because a dopamine receptor is not a term added to the striatum: it is the
target end of a projection that leaves one small midbrain nucleus and
innervates much of the forebrain at once. The quantity that couples the two
ends is not an occupancy, it is a CONCENTRATION FIELD, and the chain the
brief names is

    source nucleus firing rate
      -> transmitter release, scaled by that region's projection density
      -> region-wise extracellular transmitter concentration
      -> receptor occupancy AT EVERY TARGET REGION
      -> drive at every one of those regions' nodes.

So this module owns two objects, and the split between them is the point:

* `ProjectionSystem` is the SOURCE side. It owns one ODE state per target
  region -- that region's extracellular transmitter concentration -- and a
  mass balance driven by the source node's firing rate. It produces no
  activation and touches no node's drive, exactly like
  `transport.MonoamineTransporter`, of which it is the firing-rate-driven
  generalization.
* `RegionalReceptorSystem` is the TARGET side. It is one receptor type
  expressed in many regions at once: one occupancy state block per region,
  each driven by that region's transmitter concentration, each contributing
  to its own region's node.

Neither replaces `engine.ReceptorSystem`. A receptor whose ligand is a dosed
drug reaching the whole brain through plasma still has one occupancy and one
coupling, and every scenario built before P6.1 still says exactly that.

The mass balance, and what is new in it
---------------------------------------

For target region `k` of a projection system whose source node currently
fires at rate `r`::

    R_k(r) = release_scale * rho_k * L(r) * (1 + capacity * release_drive(drugs))
    dC_k/dt = R_k(r) - uptake(C_k, drugs)

`rho_k` is the region's projection density (dimensionless, relative), `L` is
the release law (below), and `uptake`/`release_drive`/`capacity` are the
clearance object's, i.e. P5.8's mechanism unchanged. Setting `L == 1` and
using a single region recovers `transport.MonoamineTransporter`'s equation
term for term, and a test asserts that numerically rather than by reading.

`release_scale` is DERIVED, not chosen. It is fixed by requiring that at the
source's resting firing rate the REFERENCE region reproduces the measured
basal extracellular concentration that the clearance object already carries::

    release_scale = clearance.basal_release_rate() / rho_reference

so the projection layer introduces no new absolute concentration scale. This
is the same move `transport.MonoamineTransporter.basal_release_rate` makes
(a release rate that is what the mass balance requires given the measured
uptake kinetics and the measured basal level), lifted one level: the
measured basal level pins ONE region, and every other region's basal
concentration is then a PREDICTION of the density map rather than an input.

In the first-order clearance limit that prediction is exactly proportional,
`C_k / C_reference = rho_k / rho_reference`. Under Michaelis-Menten
clearance it is CONVEX in the density rather than proportional, because
`C(R) = R*Km/(Vmax - R)` is convex: a region at a quarter of the reference's
density sits at slightly LESS than a quarter of its concentration, and a
region above the reference sits at more than proportionally more, because
uptake saturates and has less headroom the harder it is pushed. At the
concentrations the measured constants put this model in (5 nM of dopamine
against a Km of 160 nM) the departure from proportionality is 1-2%, which is
why it is stated here and asserted by a test rather than corrected for. Two
things follow that are worth stating rather than discovering later:

* a region denser than the reference can demand more release than `Vmax` can
  clear, in which case there is no finite steady state. That is not a
  numerical accident, it is the model saying that uniform uptake kinetics
  cannot support arbitrarily dense innervation, and `ProjectionSystem`
  raises rather than silently returning an enormous basal level.
* uptake kinetics are held UNIFORM across regions. `Km` and `Vmax` are
  measured in one region (striatum, for dopamine) and there is no measured
  region-wise `Vmax` map in this repository. The alternative -- scaling
  `Vmax` with `rho_k` too, on the ground that transporter density tracks
  innervation density -- would make every region's basal concentration
  identical to the reference's, i.e. would erase exactly the regional
  variation this abstraction exists to produce. Neither choice is measured;
  this one is stated here and its consequence (basal concentration scales
  with density) is what P6.3's densities will be judged against.

The release law
---------------

`ProportionalRelease(rest_rate)` is `L(r) = max(r, 0) / rest_rate`: release
rate proportional to firing rate, normalized so that `L(rest_rate) == 1`.
Three properties of that choice, in the order they matter:

1. **The model consumes only the RATIO.** `rest_rate` is the source node's
   own drug-free resting rate, read off the network, and `L` is
   dimensionless. No firing rate in Hz enters any equation here. The Hz
   values that `projection_systems.py` cites are context for what the
   dimensionless rate stands for, not parameters.
2. **Proportionality is an assumption with a name.** It is the assumption a
   spike-count model of release makes: a fixed amount of transmitter per
   action potential, so rate of release = rate of spikes times quantum. It
   is not exact -- short-term depression makes per-pulse release fall with
   frequency, and burst firing releases supralinearly through it -- and
   `PowerLawRelease` exists so that a test can vary the exponent and report
   what the conclusion costs, not so that an exponent can be tuned.
3. **Zero is a floor, not a wall.** A Wilson-Cowan rate cannot go negative,
   but clipping is written down anyway so that `L` is total.

What this abstraction deliberately does NOT model
-------------------------------------------------

* **Space inside a region.** Each target region is ONE well-mixed
  compartment. See `projection_systems.py` for the volume-transmission
  timescale argument that licenses this and for its cited numbers; the short
  version is that the relevant mixing length is the spacing between release
  sites (micrometres), not the width of a region (of order a millimetre),
  and diffusion over the former is a millisecond process against a circuit
  time constant of 1.0-1.5 min.
* **Conduction delay** from nucleus to terminal field. Axonal conduction and
  the release cascade take milliseconds to tens of milliseconds; the circuit
  integrates on minutes.
* **Autoreceptor control of release.** A somatodendritic autoreceptor here
  changes the SOURCE NODE's firing rate, and release follows, which is the
  loop P5.1 and P5.3 recorded as open. Terminal autoreceptors, which gate
  release locally without changing somatic firing, are not represented.
* **Corelease and cotransmission**, and any transmitter-specific vesicular
  machinery.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Release laws
# ---------------------------------------------------------------------------


class ProportionalRelease(object):
    """`L(r) = max(r, 0) / rest_rate`: release proportional to firing rate.

    Normalized so that a source node sitting at its own resting rate
    releases at exactly the rate the reference region's measured basal
    concentration implies. See the module docstring.
    """

    def __init__(self, rest_rate):
        rest_rate = float(rest_rate)
        if rest_rate <= 0.0:
            raise ValueError(
                "rest_rate must be the source node's positive drug-free "
                "resting rate, got %r" % (rest_rate,)
            )
        self.rest_rate = rest_rate

    def __call__(self, rate):
        return max(float(rate), 0.0) / self.rest_rate

    def __repr__(self):
        return "ProportionalRelease(rest_rate=%r)" % (self.rest_rate,)


class PowerLawRelease(object):
    """`L(r) = (max(r, 0) / rest_rate) ** exponent`.

    Exists for sensitivity analysis only. `exponent=1.0` is
    `ProportionalRelease` exactly; `exponent > 1` is the supralinear
    burst-release direction and `exponent < 1` the short-term-depression
    direction. No exponent other than 1.0 is used by any result in this
    codebase, and none is fitted; the class is here so that a test can ask
    what a conclusion costs when the release law is wrong.
    """

    def __init__(self, rest_rate, exponent):
        rest_rate = float(rest_rate)
        if rest_rate <= 0.0:
            raise ValueError("rest_rate must be positive, got %r" % (rest_rate,))
        if float(exponent) <= 0.0:
            raise ValueError("exponent must be positive, got %r" % (exponent,))
        self.rest_rate = rest_rate
        self.exponent = float(exponent)

    def __call__(self, rate):
        return (max(float(rate), 0.0) / self.rest_rate) ** self.exponent

    def __repr__(self):
        return "PowerLawRelease(rest_rate=%r, exponent=%r)" % (
            self.rest_rate,
            self.exponent,
        )


# ---------------------------------------------------------------------------
# Clearance
# ---------------------------------------------------------------------------


class FirstOrderClearance(object):
    """Removal of a transmitter from an extracellular pool at a rate
    proportional to its concentration, with optional competitive inhibition.

    Why this exists beside `transport.MonoamineTransporter`
    -------------------------------------------------------

    Two of the six transmitter systems P6.1 covers are not cleared by a
    plasma-membrane transporter at all. Acetylcholine is HYDROLYSED in the
    extracellular space by acetylcholinesterase, and histamine has no
    reuptake carrier and is inactivated by methylation. Neither is a
    monoamine and neither has a `Km`/`Vmax` pair for transport, so building
    them out of a class called `MonoamineTransporter` would name a mechanism
    that is not the one acting.

    What it shares with that class is the ALGEBRA, deliberately and
    exactly: first-order removal `k*C`, competitive inhibition raising the
    apparent constant by `1 + sum(I_i/Ki_i)`, and substrate-driven efflux.
    `test_projection.py` asserts that this class and
    `MonoamineTransporter.first_order` produce identical numbers for the
    same rate constant, so the duplication cannot drift into a difference.

    Parameters
    ----------
    name, transmitter : str
    rate : float
        min^-1. The first-order clearance rate constant, i.e. `ln(2)/t_half`
        for a measured clearance half-life.
    basal : float
        nM. The drug-free extracellular concentration in the REFERENCE
        region, which is what fixes a projection system's release scale.
    ligands : sequence
        Objects with `inhibition_term(c)` and `release_fraction(c)`, i.e.
        `transport.TransporterLigand`. Usually empty: an inhibitor of
        acetylcholinesterase would go here, and none is currently sourced.
    release_capacity : float
        As in `transport.MonoamineTransporter`.
    """

    #: One pool per region; the projection system owns the state, not this.
    is_michaelis_menten = False

    def __init__(
        self, name, transmitter, rate, basal, ligands=(), release_capacity=0.0
    ):
        rate = float(rate)
        if rate <= 0.0:
            raise ValueError("clearance rate must be positive, got %r" % (rate,))
        if basal is None or float(basal) <= 0.0:
            raise ValueError(
                "basal must be a positive drug-free extracellular "
                "concentration in nM, got %r" % (basal,)
            )
        if float(release_capacity) < 0.0:
            raise ValueError("release_capacity must be non-negative")
        self.name = name
        self.transmitter = transmitter
        self.clearance_rate = rate
        self.km = None
        self.vmax = None
        self.basal = float(basal)
        self.ligands = list(ligands)
        self.n = len(self.ligands)
        self.release_capacity = float(release_capacity)
        names = [lig.name for lig in self.ligands]
        if len(set(names)) != len(names):
            raise ValueError("duplicate ligand name at clearance %r" % (name,))

    @property
    def effective_clearance_rate(self):
        return self.clearance_rate

    def basal_release_rate(self):
        """nM/min. The release rate whose drug-free steady state is `basal`."""
        return self.clearance_rate * self.basal

    def inhibition_factor(self, concentrations):
        concentrations = np.asarray(concentrations, dtype=float)
        total = 1.0
        for lig, c in zip(self.ligands, concentrations):
            total += lig.inhibition_term(c)
        return total

    def release_drive(self, concentrations):
        concentrations = np.asarray(concentrations, dtype=float)
        total = 0.0
        for lig, c in zip(self.ligands, concentrations):
            total += lig.release_fraction(c)
        return total

    def uptake(self, transmitter, concentrations):
        """nM/min removed from the extracellular pool."""
        c = float(transmitter)
        if c <= 0.0:
            return 0.0
        return self.clearance_rate * c / self.inhibition_factor(concentrations)

    def __repr__(self):
        return "FirstOrderClearance(name=%r, transmitter=%r, rate=%r)" % (
            self.name,
            self.transmitter,
            self.clearance_rate,
        )


def concentration_for_uptake(clearance, target):
    """nM. The concentration at which `clearance` removes `target` nM/min.

    The inverse of the drug-free uptake law, which is what turns a region's
    release rate into that region's basal concentration. Raises when the
    demand exceeds what a Michaelis-Menten `Vmax` can clear at any
    concentration; see the module docstring for why that is a statement and
    not a bug.
    """
    target = float(target)
    if target < 0.0:
        raise ValueError("uptake target must be non-negative, got %r" % (target,))
    if target == 0.0:
        return 0.0
    if getattr(clearance, "is_michaelis_menten", False):
        vmax = float(clearance.vmax)
        if target >= vmax:
            raise ValueError(
                "a release rate of %.4g nM/min exceeds this clearance's Vmax "
                "of %.4g nM/min, so no finite steady state exists. With "
                "uniform uptake kinetics that is the model refusing to "
                "support an innervation density this far above the reference "
                "region's; see projection.py's module docstring."
                % (target, vmax)
            )
        return target * float(clearance.km) / (vmax - target)
    return target / float(clearance.effective_clearance_rate)


# ---------------------------------------------------------------------------
# The source side
# ---------------------------------------------------------------------------


class SourceNucleus(object):
    """One nucleus of neurons releasing a transmitter into a shared field.

    A `ProjectionSystem` takes a LIST of these rather than one node, because
    a transmitter field usually has more than one source and the alternative
    would double-count. Dopamine is the case that forces it: VTA and SNc
    innervate overlapping forebrain territory, and modelling them as two
    independent systems each pinned to the measured striatal basal dopamine
    would put twice the measured dopamine in the striatum. One field, two
    sources, each with its own density map, adds up correctly by
    construction.

    Parameters
    ----------
    node : str
        The circuit node whose firing rate drives this nucleus's release.
    rest_rate : float
        That node's drug-free resting rate, dimensionless. Required rather
        than defaulted: it is a property of the network the system is
        attached to, and getting it wrong silently rescales every
        concentration the system produces. Every rest-balanced network in
        this codebase puts its nodes at `stability.MAX_SENSITIVITY_RATE`.
    release : callable, optional
        The release law `L(rate)`, defaulting to
        `ProportionalRelease(rest_rate)` so that `L(rest_rate) == 1`.
    """

    def __init__(self, node, rest_rate, release=None):
        self.node = node
        self.rest_rate = float(rest_rate)
        self.release = ProportionalRelease(rest_rate) if release is None else release

    def __repr__(self):
        return "SourceNucleus(node=%r, rest_rate=%r)" % (self.node, self.rest_rate)


class ProjectionTarget(object):
    """One innervated region: a well-mixed extracellular transmitter pool.

    Parameters
    ----------
    region : str
        The region's name. This is the key a `RegionalReceptorSystem` and a
        `ReceptorSystem.transmitter_sources` entry resolve against, and it is
        NOT required to be a circuit node name: a projection can maintain a
        transmitter pool in a region this scenario carries no node for.
    density : float or dict
        Relative projection density, dimensionless and non-negative. A float
        for a single-source system; `{source_node: density}` when the system
        has more than one source nucleus, because the density maps of two
        nuclei innervating the same region are different maps. Only RATIOS
        are used, so the overall scale is free and one target (the
        reference) is pinned to the clearance object's measured basal
        concentration.
    provenance : str
        Where the density came from. `"illustrative"` is the honest default
        for P6.1, which builds the abstraction; P6.3 is the task that
        derives real densities from the Phase 3 connectome and replaces
        these. Carried through to `ProjectionSystem.density_report()` so a
        caller can see at a glance which numbers are placeholders.
    """

    def __init__(self, region, density, provenance="illustrative"):
        if isinstance(density, dict):
            values = {key: float(value) for key, value in density.items()}
        else:
            values = float(density)
        for value in (values.values() if isinstance(values, dict) else [values]):
            if value < 0.0:
                raise ValueError(
                    "projection density must be non-negative, got %r for %r"
                    % (value, region)
                )
        self.region = region
        self.density = values
        self.provenance = provenance

    def __repr__(self):
        return "ProjectionTarget(region=%r, density=%r, provenance=%r)" % (
            self.region,
            self.density,
            self.provenance,
        )


class ProjectionSystem(object):
    """One transmitter's field, and the nuclei whose firing maintains it.

    Owns one ODE state per target region: that region's extracellular
    transmitter concentration in nM. Produces NO activation and adds nothing
    to any node's drive -- exactly like `engine.TransporterSystem`, and for
    the same reason (see `transport.py`'s docstring). What reads its state is
    a receptor system, through `RegionalReceptorSystem` or through a
    `ReceptorSystem.transmitter_sources` entry naming `(system, region)`.

    Parameters
    ----------
    name : str
        Unique within one `Simulation`; it is what a receptor system's
        `transmitter_sources` refers to.
    transmitter : str
        "dopamine", "serotonin", "noradrenaline", "acetylcholine",
        "histamine". Reported, and checked against the clearance object.
    sources : sequence of SourceNucleus
        The nuclei whose firing drives release into this field. Every node
        named must be a node of the `Simulation`'s network. More than one is
        the normal case; see `SourceNucleus`.
    targets : sequence of ProjectionTarget
        In a fixed order, which is the order of this system's ODE states.
    clearance : object
        `transport.MonoamineTransporter` or `FirstOrderClearance`, or
        anything with the same `uptake`/`inhibition_factor`/`release_drive`/
        `release_capacity`/`basal`/`basal_release_rate`/`ligands` interface.
        ONE clearance object serves every region: see the module docstring on
        uniform uptake kinetics.
    pk_models : sequence, optional
        One per ligand on `clearance`, in that order, and empty when the
        clearance carries no drug. A projection system's drugs are
        plasma-driven exactly like a receptor's, so they take the same
        brain-partition correction (P1.5).
    reference_region : str, optional
        The region whose drug-free concentration is the clearance object's
        measured `basal`. Defaults to the densest target, which is the only
        choice that cannot make another region's demand exceed `Vmax` when
        it would not otherwise.
    """

    def __init__(
        self,
        name,
        transmitter,
        sources,
        targets,
        clearance,
        pk_models=None,
        reference_region=None,
    ):
        sources = list(sources)
        if not sources:
            raise ValueError("a projection system needs at least one source nucleus")
        source_nodes = [s.node for s in sources]
        if len(set(source_nodes)) != len(source_nodes):
            raise ValueError(
                "duplicate source node in projection system %r" % (name,)
            )
        targets = list(targets)
        if not targets:
            raise ValueError("a projection system needs at least one target region")
        regions = [t.region for t in targets]
        if len(set(regions)) != len(regions):
            raise ValueError("duplicate target region in projection system %r" % (name,))
        clearance_transmitter = getattr(clearance, "transmitter", None)
        if clearance_transmitter is not None and clearance_transmitter != transmitter:
            raise ValueError(
                "projection system %r carries transmitter %r but its clearance "
                "object clears %r" % (name, transmitter, clearance_transmitter)
            )

        self.name = name
        self.transmitter = transmitter
        self.sources = sources
        self.source_nodes = tuple(source_nodes)
        self.targets = targets
        self.regions = tuple(regions)
        self.clearance = clearance

        #: One ODE state per target region.
        self.n_states = len(self.targets)
        #: Ligand count, for the flat `doses` vector: the clearance object's
        #: drugs. Zero for a transmitter whose clearance nothing in this
        #: codebase has a constant for.
        self.ligands = list(getattr(clearance, "ligands", []))
        self.n = len(self.ligands)
        self.pk_models = list(pk_models or [])
        if len(self.pk_models) != self.n:
            raise ValueError(
                "projection system %r needs one PK model per clearance ligand "
                "(%d), got %d" % (name, self.n, len(self.pk_models))
            )

        # Normalize each target's density into {source_node: rho}. A bare
        # number is only unambiguous when there is exactly one source.
        self._density = {}
        for target in targets:
            if isinstance(target.density, dict):
                unknown = set(target.density) - set(source_nodes)
                if unknown:
                    raise ValueError(
                        "target %r of projection system %r gives a density for "
                        "source node(s) %s, which this system does not have; "
                        "have %s"
                        % (target.region, name, sorted(unknown), sorted(source_nodes))
                    )
                self._density[target.region] = {
                    node: float(target.density.get(node, 0.0))
                    for node in source_nodes
                }
            else:
                if len(sources) != 1:
                    raise ValueError(
                        "projection system %r has %d source nuclei, so target "
                        "%r needs a {source_node: density} map rather than a "
                        "single number" % (name, len(sources), target.region)
                    )
                self._density[target.region] = {source_nodes[0]: float(target.density)}
        self._row = {t.region: i for i, t in enumerate(targets)}
        self._total = {
            region: sum(values.values()) for region, values in self._density.items()
        }
        if not any(self._total.values()):
            raise ValueError(
                "every target of projection system %r has zero density, so the "
                "system releases nothing anywhere" % (name,)
            )
        #: Per target region, the density vector in `sources` order. Held as
        #: an array so the RHS does no dict lookup.
        self._density_matrix = np.array(
            [[self._density[t.region][node] for node in source_nodes] for t in targets],
            dtype=float,
        )

        if reference_region is None:
            reference_region = max(regions, key=lambda region: self._total[region])
        if reference_region not in self._density:
            raise ValueError(
                "reference_region %r is not a target of projection system %r"
                % (reference_region, name)
            )
        if self._total[reference_region] <= 0.0:
            raise ValueError(
                "reference_region %r has zero projection density" % (reference_region,)
            )
        self.reference_region = reference_region

        #: nM/min per unit projection density, with every source at its own
        #: resting rate. DERIVED from the clearance object's measured basal
        #: concentration; see the module docstring. Not a free parameter.
        self.release_scale = (
            clearance.basal_release_rate() / self._total[reference_region]
        )

        # Fail loudly at construction rather than at integration time if a
        # density map demands more release than uptake can clear.
        self._basal = np.array(
            [self._basal_concentration(region) for region in self.regions],
            dtype=float,
        )

    # -----------------------------------------------------------------
    # The map
    # -----------------------------------------------------------------

    def density(self, region, source_node=None):
        """Projection density of one target region, total or from one source."""
        if source_node is None:
            return self._total[region]
        return self._density[region][source_node]

    def row(self, region):
        """Index of one target region's state within this system's block."""
        return self._row[region]

    def relative_density(self, region):
        """Total projection density as a multiple of the reference region's."""
        return self._total[region] / self._total[self.reference_region]

    def rest_rates(self):
        """The sources' resting rates, in `sources` order."""
        return np.array([s.rest_rate for s in self.sources], dtype=float)

    def density_report(self):
        """(region, total density, relative density, provenance) per target."""
        return [
            (
                t.region,
                self._total[t.region],
                self.relative_density(t.region),
                t.provenance,
            )
            for t in self.targets
        ]

    # -----------------------------------------------------------------
    # The mechanism
    # -----------------------------------------------------------------

    def _release_law(self, source_rates):
        """`L_s(r_s)` per source, in `sources` order."""
        source_rates = np.atleast_1d(np.asarray(source_rates, dtype=float))
        if source_rates.shape != (len(self.sources),):
            raise ValueError(
                "projection system %r has %d source nuclei, got %d firing rates"
                % (self.name, len(self.sources), source_rates.size)
            )
        return np.array(
            [s.release(r) for s, r in zip(self.sources, source_rates)], dtype=float
        )

    def release_rate(self, region, source_rates, concentrations=None):
        """nM/min entering `region`'s extracellular pool.

        `release_scale * sum_s rho_{region,s} * L_s(rate_s)`, times the
        substrate-driven efflux factor when an uptake substrate
        (amphetamine) is on board. Efflux is multiplied into the region's own
        release rate rather than added as a separate constant, which is what
        makes it scale with innervation density: reverse transport needs
        transporters, and transporters sit on the same axons the release
        does.
        """
        law = self._release_law(source_rates)
        drive = self.release_scale * float(
            self._density_matrix[self._row[region]] @ law
        )
        capacity = float(getattr(self.clearance, "release_capacity", 0.0))
        if capacity and concentrations is not None:
            drive = drive * (1.0 + capacity * self.clearance.release_drive(concentrations))
        return drive

    def _basal_concentration(self, region):
        target = self.release_scale * self._total[region]
        try:
            return concentration_for_uptake(self.clearance, target)
        except ValueError as exc:
            raise ValueError(
                "projection system %r, target region %r: %s" % (self.name, region, exc)
            )

    def basal_concentration(self, region):
        """nM. This region's drug-free extracellular concentration, with the
        source node at its resting rate.

        A PREDICTION for every region except the reference one, whose value
        is the clearance object's measured basal by construction.
        """
        return float(self._basal[self._row[region]])

    def initial_state(self):
        """The drug-free concentration field, shape `(n_states,)`.

        Like `transport.MonoamineTransporter.initial_state` and unlike every
        receptor block in this codebase, this does NOT start at zero: there
        is transmitter in the extracellular space before any drug is given,
        and that is what makes an antagonist given alone do anything.
        """
        return self._basal.copy()

    def concentration_rhs(self, state, source_rates, concentrations):
        """d[transmitter]/dt per target region, shape `(n_states,)`."""
        state = np.asarray(state, dtype=float)
        law = self._release_law(source_rates)
        factor = 1.0
        capacity = float(getattr(self.clearance, "release_capacity", 0.0))
        if capacity:
            factor = 1.0 + capacity * self.clearance.release_drive(concentrations)
        release = self.release_scale * factor * (self._density_matrix @ law)
        out = np.empty(self.n_states)
        for i in range(self.n_states):
            out[i] = release[i] - self.clearance.uptake(state[i], concentrations)
        return out

    def steady_state(self, region, source_rates, concentrations=None):
        """nM. The concentration `region` settles at for held firing rates.

        Exact, by inverting the uptake law, so a test can check the
        integrated ODE against it rather than against itself.
        """
        if concentrations is None:
            concentrations = np.zeros(self.n)
        release = self.release_rate(region, source_rates, concentrations)
        inhibition = self.clearance.inhibition_factor(concentrations)
        if getattr(self.clearance, "is_michaelis_menten", False):
            vmax = float(self.clearance.vmax)
            if release >= vmax:
                return float("inf")
            return release * float(self.clearance.km) * inhibition / (vmax - release)
        return release * inhibition / float(self.clearance.effective_clearance_rate)

    def __repr__(self):
        return (
            "ProjectionSystem(name=%r, transmitter=%r, sources=%r, "
            "n_targets=%d)" % (self.name, self.transmitter, list(self.source_nodes),
                               self.n_states)
        )


# ---------------------------------------------------------------------------
# The target side
# ---------------------------------------------------------------------------


class ProjectionSource(object):
    """`(system, region)`: a receptor ligand's concentration taken from one
    region's pool of one projection system.

    The P6.1 widening of P5.8's `transmitter_sources`, whose values were
    bare transporter-system names. A plain string still means a transporter
    system and behaves exactly as it did; this object means a projection
    system, and carries the region because a projection has more than one.
    """

    def __init__(self, system, region):
        self.system = system
        self.region = region

    def __iter__(self):  # so `tuple(...)` and unpacking work
        return iter((self.system, self.region))

    def __eq__(self, other):
        if isinstance(other, ProjectionSource):
            return (self.system, self.region) == (other.system, other.region)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.system, self.region) == other
        return NotImplemented

    def __hash__(self):
        return hash((self.system, self.region))

    def __repr__(self):
        return "ProjectionSource(system=%r, region=%r)" % (self.system, self.region)


def as_projection_source(value, projection_names=(), region=None):
    """Normalize a `transmitter_sources` value, or return None for one that
    names a transporter system.

    A `ProjectionSource`, or a 2-tuple/list, is always a projection source.
    A bare string is a projection source when it is the name of one of
    `projection_names` and a transporter-system name otherwise, which is
    P5.8's meaning preserved exactly: a `Simulation` requires transporter and
    projection system names to be disjoint, so the string is never
    ambiguous. When the string names a projection system the region is
    `region`, i.e. the receptor block's own, which is the whole point of
    `RegionalReceptorSystem`.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value in projection_names:
            return ProjectionSource(value, region)
        return None
    if isinstance(value, ProjectionSource):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return ProjectionSource(value[0], value[1])
    raise ValueError(
        "a transmitter_sources value must be a transporter-system name (str), "
        "a projection-system name (str), or a (projection_system, region) "
        "pair, got %r" % (value,)
    )


class RegionalReceptorSystem(object):
    """One receptor type expressed across many regions at once (P6.1).

    The target end of the chain in the module docstring, and the thing that
    replaces "a receptor sits on one designated node". Compared with
    `engine.ReceptorSystem`:

    * it owns ONE occupancy state block PER REGION, all with the same
      `receptor_site` definition (the site object is stateless, so it is
      shared rather than copied);
    * its ligands, and therefore its PK models and its entries in the flat
      `doses` vector, are shared across regions. A dosed drug reaches every
      region through the same plasma, so duplicating its PK per region would
      integrate the same trajectory `n_regions` times to no purpose;
    * `coupling` is per region, and each region's activation contributes to
      that region's node only;
    * `transmitter_sources` names a projection SYSTEM without a region: the
      region is whichever block is being evaluated, which is the entire
      point of the class.

    Parameters
    ----------
    pk_models : sequence
        One per ligand on `receptor_site`, shared across regions.
    receptor_site : ReceptorSite
        Shared. Its `occupancy_rhs`/`activation` are pure functions of the
        state block handed to them, which is what makes sharing correct.
    regions : sequence of str
        In a fixed order, which is the order of the state blocks. Every one
        must be a target region of the projection system(s) this system
        sources from.
    coupling : dict
        `{region: gain}` (the region's own node carries the receptor) or
        `{region: {node_name: gain}}` (it drives some other node, or more
        than one). A region absent from `coupling` contributes to no node,
        which is legitimate: the receptor is there and its occupancy is
        tracked, and this scenario carries no node for what it does.
    transmitter_sources : dict, optional
        `{ligand_name: projection_system_name}`. Also accepts a
        `ProjectionSource`/2-tuple or a bare transporter-system name for a
        ligand whose pool is the SAME in every region.
    connection_modulation : dict, optional
        `{region: {(post_node, pre_node): gain}}`, composed exactly as
        `engine.ReceptorSystem`'s is.
    density : dict, optional
        `{region: rho}`, a relative RECEPTOR density multiplying that
        region's coupling gain. Defaults to 1.0 everywhere. This is P4.2's
        missing factor `rho` in `coupling = mechanism_gain * rho`, exposed
        here because a projection system is the first structure in this
        codebase for which "the same receptor in twenty regions" is a single
        object that could carry it.
    name : str, optional
    """

    #: Marks the regional code path in `engine.Simulation.rhs`.
    is_regional = True

    def __init__(
        self,
        pk_models,
        receptor_site,
        regions,
        coupling,
        transmitter_sources=None,
        connection_modulation=None,
        density=None,
        name=None,
    ):
        if len(pk_models) != receptor_site.n:
            raise ValueError("need one PK model per ligand on the receptor site")
        regions = list(regions)
        if not regions:
            raise ValueError("a regional receptor system needs at least one region")
        if len(set(regions)) != len(regions):
            raise ValueError("duplicate region in a regional receptor system")

        self.pk_models = list(pk_models)
        self.receptor_site = receptor_site
        self.regions = tuple(regions)
        self.n_regions = len(self.regions)
        self.name = name or receptor_site.name
        self.n = receptor_site.n
        #: Per-region block length, and the total.
        self.block = int(getattr(receptor_site, "n_states", receptor_site.n))
        self.n_states = self.block * self.n_regions

        self.coupling = self._normalize_coupling(coupling)
        self.connection_modulation = {
            region: dict(entries)
            for region, entries in dict(connection_modulation or {}).items()
        }
        unknown = set(self.connection_modulation) - set(self.regions)
        if unknown:
            raise ValueError(
                "connection_modulation names regions this system does not "
                "carry: %s" % sorted(unknown)
            )

        self.density = {region: 1.0 for region in self.regions}
        for region, value in dict(density or {}).items():
            if region not in self.density:
                raise ValueError(
                    "density names region %r, which this system does not carry"
                    % (region,)
                )
            if float(value) < 0.0:
                raise ValueError("receptor density must be non-negative")
            self.density[region] = float(value)

        self.transmitter_sources = dict(transmitter_sources or {})
        known = {lig.name for lig in receptor_site.ligands}
        unknown = set(self.transmitter_sources) - known
        if unknown:
            raise ValueError(
                "transmitter_sources names ligands that are not on this "
                "receptor site: %s" % sorted(unknown)
            )

    def _normalize_coupling(self, coupling):
        out = {}
        for region, value in dict(coupling or {}).items():
            if region not in self.regions:
                raise ValueError(
                    "coupling names region %r, which this system does not carry; "
                    "have %s" % (region, sorted(self.regions))
                )
            if isinstance(value, dict):
                out[region] = dict(value)
            else:
                out[region] = {region: float(value)}
        return out

    def state_rows(self, region):
        """Row indices of one region's state block, within this system's
        block of `result["occupancy"]`."""
        k = self.regions.index(region)
        return np.arange(k * self.block, (k + 1) * self.block)

    def __repr__(self):
        return "RegionalReceptorSystem(name=%r, n_regions=%d, n_ligands=%d)" % (
            self.name,
            self.n_regions,
            self.n,
        )


# ---------------------------------------------------------------------------
# Resting occupancy and the standing drive it creates
# ---------------------------------------------------------------------------


def resting_occupancy(receptor_site, concentrations, t_end=None):
    """The occupancy state block a site settles at for held concentrations.

    Integrated rather than solved in closed form, so that it is correct for
    every site class in `receptor.py` -- competitive, channel-blocking and
    desensitizing -- rather than only for the two-state one. Used at SETUP
    time (to compute the standing drive tonic transmitter imposes), never
    inside an RHS.
    """
    from scipy.integrate import solve_ivp

    concentrations = np.asarray(concentrations, dtype=float)
    n_states = int(getattr(receptor_site, "n_states", receptor_site.n))
    if t_end is None:
        # Long enough that the slowest first-order component has decayed:
        # 40 / koff_min, floored so a fast site still gets a real span.
        koffs = [
            float(getattr(lig, "koff", 0.0)) for lig in receptor_site.ligands
        ]
        kons = [
            float(getattr(lig, "kon", 0.0)) * float(c)
            for lig, c in zip(receptor_site.ligands, concentrations)
        ]
        rates = [r for r in koffs + kons if r > 0.0]
        t_end = 40.0 / min(rates) if rates else 1.0
        t_end = float(np.clip(t_end, 1.0, 1e6))
    sol = solve_ivp(
        lambda t, y: receptor_site.occupancy_rhs(y, concentrations),
        (0.0, t_end),
        np.zeros(n_states),
        method="LSODA",
        rtol=1e-10,
        atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError("resting occupancy integration failed: %s" % sol.message)
    return sol.y[:, -1]


def standing_drive(regional_systems, projection_systems):
    """`{node_name: drive}` contributed at rest by tonic transmitter.

    P5.8 found, and P3.4 found before it, that any mechanism adding a
    STANDING term to a node's drive invalidates a stability certificate
    computed without it. Tonic transmitter is exactly such a mechanism, and
    a projection system creates one at every target region at once, so the
    P5.8 fix (`monoamine_transporters.tonic_compensated_network`, which
    hand-adjusted a single node) has to become a computation.

    Returns what `compensate_baselines` should subtract: for each region of
    each regional receptor system, the site's resting activation at that
    region's drug-free transmitter concentration, times the region's
    receptor density, times each coupling gain.
    """
    by_name = {system.name: system for system in projection_systems}
    drive = {}
    for system in regional_systems:
        if not getattr(system, "is_regional", False):
            raise TypeError(
                "standing_drive takes RegionalReceptorSystems; %r is a "
                "single-block receptor system, whose tonic drive is P5.8's "
                "monoamine_transporters.tonic_compensated_network"
                % (getattr(system, "name", system),)
            )
        ligand_names = [lig.name for lig in system.receptor_site.ligands]
        for region in system.regions:
            concentrations = np.zeros(system.n)
            for ligand_name, source in system.transmitter_sources.items():
                projection = as_projection_source(
                    source, projection_names=by_name, region=region
                )
                if projection is None:
                    continue  # a transporter pool: not this function's business
                target_region = (
                    region if projection.region is None else projection.region
                )
                pool = by_name[projection.system]
                concentrations[ligand_names.index(ligand_name)] = (
                    pool.basal_concentration(target_region)
                )
            if not np.any(concentrations):
                continue
            activation = system.receptor_site.activation(
                resting_occupancy(system.receptor_site, concentrations)
            )
            scaled = activation * system.density[region]
            for node, gain in system.coupling.get(region, {}).items():
                drive[node] = drive.get(node, 0.0) + gain * scaled
    return drive


def compensate_baselines(network, drive):
    """A copy of `network` with each node's `baseline_input` reduced by the
    standing drive tonic transmitter puts on it.

    The general form of `monoamine_transporters.tonic_compensated_network`,
    and it asserts exactly what that function asserts and no more: a real
    brain region's resting firing rate is what it is WITH tonic
    neuromodulator present, so the tonic term belongs INSIDE the baseline the
    node is balanced around rather than stacked on top of it. It does not
    decide how much of a region's resting drive is dopaminergic; it says the
    network's own calibration point already includes whatever that is.

    Changes `baseline_input` only. No weight, tau, gain or threshold moves.
    """
    from .circuit import CircuitNode
    from .network import CircuitNetwork

    unknown = set(drive) - set(network.index)
    if unknown:
        raise ValueError(
            "standing drive names nodes this network does not have: %s"
            % sorted(unknown)
        )
    nodes = [
        CircuitNode(
            name=node.name,
            tau=node.tau,
            baseline_input=node.baseline_input - drive.get(node.name, 0.0),
            gain=node.gain,
            threshold=node.threshold,
        )
        for node in network.nodes
    ]
    return CircuitNetwork(nodes=nodes, connectivity=network.connectivity.copy())
