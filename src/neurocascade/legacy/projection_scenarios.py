"""P6.1: the existing scenarios, re-wired through projection systems.

    from neurocascade import projection_scenarios as ps

    ported = ps.port("serotonin")
    print(ported.report())
    network, sim, result = ported.run(**{"8-OH-DPAT": 100.0})

Nothing in `serotonin.py`, `noradrenaline.py`, `acetylcholine.py`,
`histamine.py` or `monoamine_transporters.py` moved. Every hand-built
network, receptor site, ligand set, coupling gain and readout is still
there, is still what the existing tests exercise, and is what this module
ports *from*: this is P3.3's relationship to `scenarios.py`, repeated one
layer up.

What a port changes, in exactly three places
--------------------------------------------

1. **Where the transmitter comes from.** Before: the transmitter is a
   `Ligand` you inject a bolus of, cleared by an ILLUSTRATIVE first-order
   rate (`serotonin.SEROTONIN_KE`, `noradrenaline.NORADRENALINE_KE`,
   `acetylcholine.ACETYLCHOLINE_KE`, `histamine.HISTAMINE_KE`, all 5.0
   min^-1 and all labelled "the real quantity belongs to P5.8 and P6.2").
   After: the transmitter is a field maintained by a
   `projection.ProjectionSystem` whose release is driven by the source
   nucleus's firing rate and whose clearance is the measured uptake process
   where one exists. **Two of those four illustrative clearance proxies are
   replaced by measured constants** ([JohnJones2007]'s Km/Vmax for serotonin,
   [Park2012]'s clearance rate for noradrenaline). Histamine's and
   acetylcholine's are replaced by differently-labelled illustrative ones,
   because no measured extracellular clearance rate constant was found for
   either; `projection_systems.py` records both searches. Histamine does gain
   a measured BASAL level in the exchange, and acetylcholine gains nothing
   but the structure.

2. **Where the receptor is.** Before: one `ReceptorSystem` with one
   occupancy per ligand and a `coupling` dict that may name several nodes,
   so a receptor in the raphe and the same receptor in the cortex share ONE
   occupancy state. After: a `projection.RegionalReceptorSystem` with one
   occupancy block PER REGION, each driven by its own region's transmitter
   concentration. That difference is the whole of P6.1's "replace the
   current pattern in which a receptor sits on one designated node", and it
   is not cosmetic: 5-HT1A in the dorsal raphe and 5-HT1A in the cortex now
   have different occupancies, because they see different serotonin.

3. **What the network contains.** A projection needs its source nucleus to
   BE a node. Three of the five ports therefore add isolated nucleus nodes
   the hand-built scenario did not carry: the median raphe for serotonin,
   the PPT and LDT for acetylcholine, and the VTA and SNc for dopamine.
   Each is added exactly as the hand-built scenarios add their own nucleus
   node -- no edges, `baseline_input = 0`, so it rests at 0.5 -- and none
   of them changes any existing node's parameters.

Everything else is carried over unchanged: binding constants, efficacies,
coupling gains, connection weights, taus, gains, and readouts.

The one arithmetic correction, and why it is not a fourth change
-----------------------------------------------------------------

Tonic transmitter occupies receptors at rest, and an occupied receptor puts
a STANDING term on its node's drive. P3.4 found that such a term invalidates
a stability certificate computed without it, P5.7 hit it with a presynaptic
modulator, P5.8 hit it with tonic dopamine and fixed it by hand for one
node. `projection.standing_drive` computes it and
`projection.compensate_baselines` folds it into `baseline_input`, which is
the same assertion P5.8's `tonic_compensated_network` makes and no stronger:
a real region's resting firing rate is what it is WITH tonic neuromodulator
present, so the tonic term belongs inside the baseline the region is
balanced around. It changes baselines only. A test asserts that every ported
network's drug-free run is flat at 0.5 and that removing the compensation
saturates something.

The density maps are DERIVED from the connectome since P6.3
-----------------------------------------------------------

P6.1 gave every `ProjectionTarget` density 1.0 from every source, tagged
`provenance="illustrative-uniform"`, and said "P6.3 is the task that derives
real densities from the Phase 3 connectome". It is, and
`projection_density.py` is where the derivation and its citations live.
`port(key)` now builds from that table; `port(key, density_map="uniform")`
still builds P6.1's map, kept as an object so that every number this task
moved can be RE-RUN at the old densities rather than remembered.

Three things follow, and each is a change to what a port asserts:

1. **The reference region is now the region the measurement was made in.**
   Under a uniform map every region's concentration equalled the
   reference's, so which region was the reference did not matter. With a
   real map it is the only region whose concentration is a measurement.
   Two ports therefore gain a target region they have no node for -- the
   serotonin and acetylcholine ports gain a caudoputamen pool, because
   [Mathews2004] and [Vinson1997] measured there -- and the histamine port
   gains a medial-hypothalamus pool for [Itoh1991]'s probe. A pool with no
   node is what `ProjectionTarget` was built to allow.
2. **Regions no longer agree.** Cortical dopamine is 0.19 nM against
   striatal 5.0 nM, cortical serotonin 10.0 nM against striatal 2.8 nM,
   cortical acetylcholine 10.1 nM against striatal 18.4 nM, cortical
   histamine 0.016 nM against hypothalamic 7.8 nM. Two of those are
   checkable against held-out measurements and one of the two fails; see
   `projection_density.validation_report()`.
3. **The one density that is still illustrative is the nucleus's own.** A
   source nucleus's extracellular pool is somatodendritic and no
   anterograde tracer measures it, so each nuclear region keeps P6.1's
   value (the reference region's concentration) under the honest label
   `illustrative-somatodendritic`. It is the density every autoreceptor in
   this module sits in, so the tests sweep it.

What the ports are checked for
------------------------------

`tests/test_projection_scenarios.py` asserts, for every port:

* the drug-free run is flat (the tonic compensation is right);
* the direction of every drug effect the hand-built scenario reported is
  unchanged;
* **the autoreceptor loop is closed**: an agonist at the somatodendritic
  autoreceptor lowers the source nucleus's firing AND lowers the transmitter
  concentration in the terminal field AND lowers the terminal receptors'
  occupancy. `serotonin.py`'s and `noradrenaline.py`'s own tests assert that
  this is IMPOSSIBLE in the un-ported scenario, and both remain true and
  green, because a port is a new object rather than an edit;
* **an antagonist given alone is not inert**, because there is tonic
  transmitter to displace. Every module from P5.1 to P5.7 had to record the
  opposite;
* **the closed loop is stable**, by P3.4's check run on the WHOLE coupled
  system rather than on the circuit alone (`stability.assess_system`). The
  autoreceptor's feedback path leaves the circuit, goes through the
  transmitter concentration and the receptor occupancy, and comes back as a
  drive, so it is invisible to a Jacobian computed from
  `network.connectivity`. P6.3 is the task that closes the third of these
  loops (D2 on the VTA and SNc) and the first that has to check them.
"""

import numpy as np

from .. import acetylcholine as _ach
from .. import histamine as _ha
from .. import monoamine_transporters as _mt
from .. import noradrenaline as _na
from .. import projection as _projection
from . import projection_density as _pdens
from . import projection_systems as _ps
from .. import scenarios as _scenarios
from .. import serotonin as _5ht
from ..circuit import CircuitNode
from ..engine import Simulation
from ..network import CircuitNetwork
from ..projection import ProjectionTarget, RegionalReceptorSystem


#: The parameters an added nucleus node gets. Copied from the hand-built
#: scenarios' own nucleus nodes (`serotonin._NODES[0]`, `noradrenaline`,
#: `acetylcholine`, `histamine`, all identical): no edges, tau 1.0,
#: baseline_input 0.0 and gain 4.0, so `sigmoid(0)` puts it at rest exactly
#: at `stability.MAX_SENSITIVITY_RATE`.
NUCLEUS_NODE = {"tau": 1.0, "baseline_input": 0.0, "gain": 4.0}

#: The dimensionless resting rate every node in every ported network sits at,
#: which is what the release law is normalized to. Not a new number: it is
#: `stability.MAX_SENSITIVITY_RATE` and it is what each hand-built scenario's
#: `rest_rates()` already returns.
REST_RATE = 0.5

#: The provenance tag P6.1's uniform densities carry. Reachable through
#: `port(key, density_map="uniform")`, which is how every P6.3 change is
#: reported as a difference rather than asserted.
DENSITY_PROVENANCE = "illustrative-uniform"

#: The density maps `port()` accepts. `"connectome"` is P6.3's derived
#: table; `"uniform"` is P6.1's.
DENSITY_MAPS = ("connectome", "uniform")

#: The D2 coupling gain, `scenarios.d2_coupling()`'s own, reused unchanged
#: for the postsynaptic striatal receptor AND for the somatodendritic
#: autoreceptor P6.3 adds to the VTA and SNc. ILLUSTRATIVE, exactly as it is
#: there; see the dopamine port's notes for why reuse rather than a second
#: number.
D2_GAIN = _scenarios.d2_coupling()["Striatum_D2_MSN"]


def add_nodes(network, names):
    """A copy of `network` with isolated nucleus nodes appended.

    No edges and no change to any existing node, so the base network's
    dynamics are untouched and the added nodes rest at 0.5.
    """
    names = [name for name in names if name not in network.index]
    if not names:
        return network
    nodes = list(network.nodes) + [
        CircuitNode(name=name, **NUCLEUS_NODE) for name in names
    ]
    n = len(nodes)
    W = np.zeros((n, n))
    W[: network.n, : network.n] = network.connectivity
    return CircuitNetwork(nodes=nodes, connectivity=W)


class ReceptorPlacement(object):
    """One receptor type, the regions that carry it, and the node each of
    those regions drives.

    `site` and `pk_models` are zero-argument callables returning the
    hand-built scenario's own objects, so a port cannot quietly redefine a
    receptor: it can only move it.
    """

    def __init__(self, name, site, pk_models, transmitter_ligand, coupling):
        self.name = name
        self.site = site
        self.pk_models = pk_models
        self.transmitter_ligand = transmitter_ligand
        self.coupling = dict(coupling)  # region -> {node: gain}

    @property
    def regions(self):
        return tuple(self.coupling)

    def open_loop(self):
        """A copy with every AUTORECEPTOR coupling gain set to zero.

        An autoreceptor is a receptor whose region is one of the system's
        own source nuclei, so "which coupling entries are the feedback path"
        is a question the region names answer and nothing has to declare.
        Zeroing the gain rather than deleting the region keeps the occupancy
        block, the concentration and the state layout identical, so an
        open-loop and a closed-loop port differ in ONE number per
        autoreceptor and in nothing else. That is what makes
        `stability.assess_system` on the two of them a measurement of the
        loop rather than of two different models.
        """
        coupling = {
            region: (
                {node: 0.0 for node in nodes}
                if region in _pdens.NUCLEAR_REGIONS
                else dict(nodes)
            )
            for region, nodes in self.coupling.items()
        }
        return ReceptorPlacement(
            name=self.name,
            site=self.site,
            pk_models=self.pk_models,
            transmitter_ligand=self.transmitter_ligand,
            coupling=coupling,
        )

    def build(self, projection_name):
        return RegionalReceptorSystem(
            pk_models=self.pk_models(),
            receptor_site=self.site(),
            regions=self.regions,
            coupling=self.coupling,
            transmitter_sources={self.transmitter_ligand: projection_name},
            name=self.name,
        )


class ScenarioPort(object):
    """The recipe for one port: which hand-built network, which nuclei, which
    target regions, and where each receptor goes."""

    def __init__(
        self,
        key,
        transmitter,
        base_network,
        nucleus_nodes,
        regions,
        placements,
        readouts,
        source_readout,
        terminal_readout,
        notes="",
    ):
        self.key = key
        self.transmitter = transmitter
        self.base_network = base_network
        #: CCF nucleus key -> circuit node name.
        self.nucleus_nodes = dict(nucleus_nodes)
        #: Target region name -> the CCF structure it stands for. Regions are
        #: named by CCF acronym so P6.3 can look their rows up directly; the
        #: node they drive is in each placement's `coupling`.
        #:
        #: P6.3 requires the REFERENCE region -- the one whose concentration
        #: is this transmitter's cited measurement -- to be among them,
        #: because with a real density map every other region's
        #: concentration is a prediction relative to it. Two ports gain a
        #: region with no node because of this, which is exactly what
        #: `ProjectionTarget` allows.
        self.regions = dict(regions)
        self.reference_region = _pdens.REFERENCE_REGION[transmitter]
        if self.reference_region not in self.regions:
            raise ValueError(
                "port %r must carry its transmitter's reference region %r "
                "(where %s's cited basal concentration was measured) as a "
                "target region; it has %s"
                % (key, self.reference_region, transmitter, sorted(self.regions))
            )
        unknown = set(self.regions) - set(_pdens.REGION_STRUCTURES)
        if unknown:
            raise ValueError(
                "port %r names target regions with no CCF structures in "
                "projection_density.REGION_STRUCTURES, so no density can be "
                "derived for them: %s" % (key, sorted(unknown))
            )
        self.placements = list(placements)
        self.readouts = dict(readouts)
        self.source_readout = source_readout
        self.terminal_readout = terminal_readout
        self.notes = notes


def _targets(port, densities=None, density_map="connectome", table=None):
    """One `ProjectionTarget` per region, from the P6.3 derived table.

    `density_map="uniform"` rebuilds P6.1's map (1.0 from every nucleus into
    every region) so that any result can be re-run at the old densities.
    `densities` overrides individual regions and is used by exactly one
    caller, a test that has to show what a hand-made map would do; nothing in
    the modelling path passes it.
    """
    if density_map not in DENSITY_MAPS:
        raise ValueError(
            "density_map must be one of %s, got %r" % (DENSITY_MAPS, density_map)
        )
    if table is None:
        table = (
            _pdens.load_table()
            if density_map == "connectome"
            else _pdens.uniform_table()
        )
    nuclei = _ps.NUCLEI_BY_TRANSMITTER[port.transmitter]
    derived = table.target_densities(
        list(port.regions),
        nuclei,
        reference=port.reference_region,
        nodes=port.nucleus_nodes,
    )
    densities = dict(densities or {})
    out = []
    for region in port.regions:
        if region in densities:
            out.append(
                ProjectionTarget(region, dict(densities[region]), provenance="override")
            )
            continue
        provenance = (
            table.provenance(region)
            if density_map == "connectome"
            else DENSITY_PROVENANCE
        )
        out.append(ProjectionTarget(region, derived[region], provenance=provenance))
    return out


class PortedScenario(object):
    """A hand-built scenario re-expressed through a projection system.

    Attributes
    ----------
    base_network : the hand-built network, plus any added nucleus nodes and
        before the tonic compensation. Kept so a test can show what the
        compensation is worth.
    network : `base_network` with the standing tonic drive folded into the
        baselines. This is the one `run()` uses.
    projection : the `ProjectionSystem`.
    receptor_systems : the `RegionalReceptorSystem`s.
    """

    def __init__(self, port, densities=None, basal=None,
                 density_map="connectome", table=None, autoreceptor=True):
        self.port = port
        self.key = port.key
        self.transmitter = port.transmitter
        self.basal = basal
        self.density_map = density_map
        self.autoreceptor = bool(autoreceptor)
        self.base_network = add_nodes(
            port.base_network(), sorted(port.nucleus_nodes.values())
        )
        self.projection = _ps.projection_system(
            port.transmitter,
            _targets(port, densities, density_map=density_map, table=table),
            REST_RATE,
            nodes=port.nucleus_nodes,
            name=port.transmitter,
            clearance=(
                None
                if basal is None
                else _ps.CLEARANCE_BUILDERS[port.transmitter](basal=basal)
            ),
            reference_region=port.reference_region,
        )
        self.receptor_systems = [
            (placement if self.autoreceptor else placement.open_loop()).build(
                port.transmitter
            )
            for placement in port.placements
        ]
        self.standing_drive = _projection.standing_drive(
            self.receptor_systems, [self.projection]
        )
        self.network = _projection.compensate_baselines(
            self.base_network, self.standing_drive
        )

    # -----------------------------------------------------------------

    def simulation(self, network=None):
        return Simulation(
            network=self.network if network is None else network,
            receptor_systems=self.receptor_systems,
            projection_systems=[self.projection],
        )

    def run(self, t_end=30.0, n_points=121, method="LSODA", network=None,
            rates0=None, occupancy0="resting", **doses):
        """Integrate one dosing configuration; returns (network, sim, result).

        `method` defaults to LSODA for P5.8's reason, restated: the measured
        uptake kinetics put a 52 ms time constant on the dopamine pool
        against a circuit relaxing over 1-2 min, which is a stiff system, and
        a test asserts RK45 and LSODA agree so the choice is a statement
        about the integrator and not about the model.

        `occupancy0="resting"` is the default HERE and not in `Simulation`:
        the tonic compensation subtracts a standing drive that an
        empty-receptor start has not yet produced, so starting from zero
        occupancy would put a start-up transient on every tonically occupied
        node of exactly the size of that standing drive (2.13 units on
        `Striatum_D2_MSN`). A test asserts both that the default run is flat
        and that `occupancy0=None` reproduces the transient.
        """
        network = self.network if network is None else network
        sim = self.simulation(network=network)
        if rates0 is None:
            rates0 = np.full(network.n, REST_RATE)
        result = sim.run(
            doses=sim.dose_vector(**doses),
            t_span=(0.0, t_end),
            t_eval=np.linspace(0.0, t_end, n_points),
            rates0=rates0,
            method=method,
            occupancy0=occupancy0,
        )
        return network, sim, result

    # -----------------------------------------------------------------

    def rate_trace(self, network, result, node):
        return result["rates"][network.index[node]]

    def field_trace(self, sim, result, region):
        """nM. One target region's transmitter concentration over the run."""
        return sim.projection_trace(result, self.transmitter, region)

    def field_fold(self, sim, result, region):
        """The same trace as a multiple of that region's drug-free level,
        which is the unit every conclusion here is stated in.

        P6.1 had a second reason for it: two of the five fields had a unit
        scale rather than a measured basal level, and a fold was the only
        thing they supported at all. P6.2 removed that reason -- all five
        carry a cited concentration now -- and the fold is kept because it is
        the quantity that is provably independent of the basal level at a
        held source rate, which is what makes a comparison across the five
        ports mean anything."""
        basal = self.projection.basal_concentration(region)
        return self.field_trace(sim, result, region) / basal

    def occupancy_trace(self, sim, result, system_name, region, ligand):
        """One region's occupancy of one ligand at one receptor system."""
        index = [s.name for s in self.receptor_systems].index(system_name)
        system = self.receptor_systems[index]
        rows = sim.state_rows(index)[system.state_rows(region)]
        block = result["occupancy"][rows]
        names = [lig.name for lig in system.receptor_site.ligands]
        return block[names.index(ligand)]

    def report(self):
        lines = []
        lines.append("Ported scenario: %s (%s)" % (self.key, self.transmitter))
        lines.append("=" * 72)
        if self.port.notes:
            lines.append(self.port.notes.strip())
            lines.append("")
        lines.append("Source nuclei (CCF -> circuit node, rest rate %.2f):" % REST_RATE)
        hand_built = {node.name for node in self.port.base_network().nodes}
        for ccf, node in sorted(self.port.nucleus_nodes.items()):
            added = (
                "already in the hand-built network"
                if node in hand_built
                else "added by the port"
            )
            lines.append(
                "  %-5s %-6s -> %-18s %s"
                % (ccf, _ps.NUCLEI[ccf]["acronym"], node, added)
            )
        lines.append("")
        lines.append("Target regions and their drug-free concentration:")
        for region, density, relative, provenance in self.projection.density_report():
            lines.append(
                "  %-6s (%s)  density %.4g (%.4gx reference, %s)  basal %.4g nM"
                % (
                    region,
                    self.port.regions[region],
                    density,
                    relative,
                    provenance,
                    self.projection.basal_concentration(region),
                )
            )
        lines.append("")
        lines.append("Receptors, per region:")
        for system in self.receptor_systems:
            for region in system.regions:
                nodes = ", ".join(
                    "%s%+.3f" % (node, gain)
                    for node, gain in sorted(system.coupling.get(region, {}).items())
                )
                lines.append("  %-12s %-6s -> %s" % (system.name, region, nodes or "-"))
        lines.append("")
        lines.append("Standing drive folded into baselines by the tonic compensation:")
        if not self.standing_drive:
            lines.append("  (none: no receptor here is occupied at rest)")
        for node, value in sorted(self.standing_drive.items()):
            lines.append("  %-18s %+.4f" % (node, value))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The five ports
# ---------------------------------------------------------------------------


def _serotonin_port():
    return ScenarioPort(
        key="serotonin",
        transmitter="serotonin",
        base_network=_5ht.raphe_cortical_network,
        nucleus_nodes={"DR": "DRN_5HT", "CS": "MRN_5HT"},
        regions={
            "DR": "Dorsal nucleus raphe",
            "CS": "Superior central nucleus raphe (median raphe)",
            "PFC": "Prelimbic cortex (the scenario's PFC_Pyramidal)",
            "CP": (
                "Caudoputamen: the REFERENCE region, where [Mathews2004] "
                "measured 2.8 nM by zero net flux. No node and no receptor"
            ),
        },
        placements=[
            ReceptorPlacement(
                name="5HT1A",
                site=_5ht.receptor_site_5ht1a,
                pk_models=_5ht.pk_models_5ht1a,
                transmitter_ligand="serotonin",
                coupling={
                    "DR": {"DRN_5HT": _5ht.HT1A_MECHANISM_GAIN},
                    "CS": {"MRN_5HT": _5ht.HT1A_MECHANISM_GAIN},
                    "PFC": {"PFC_Pyramidal": _5ht.HT1A_MECHANISM_GAIN},
                },
            ),
            ReceptorPlacement(
                name="5HT2A",
                site=_5ht.receptor_site_5ht2a,
                pk_models=_5ht.pk_models_5ht2a,
                transmitter_ligand="serotonin",
                coupling={"PFC": {"PFC_Pyramidal": _5ht.HT2A_MECHANISM_GAIN}},
            ),
        ],
        readouts={
            "serotonergic_tone": _5ht.serotonergic_tone_readout(),
            "cortical_output": _5ht.cortical_output_readout(),
        },
        source_readout="DRN_5HT",
        terminal_readout="PFC_Pyramidal",
        notes="""
        P6.3: the caudoputamen pool is the reference region and carries
        [Mathews2004]'s 2.8 nM. The cortical concentration is then a
        PREDICTION of the derived density map, and the same paper measured
        it in the same mice: 9.97 nM predicted against 1.4 nM measured. The
        DENSITY ratio is 3.42 against a measured 2.0-fold the other way, i.e.
        6.8-fold out and in the wrong direction, and the concentration is a
        little above 3.42 x 2.8 because Michaelis-Menten clearance is convex.
        That failure is reported by
        `projection_density.validation_report()` rather than tuned away, and
        its leading suspect is that a Knox et al. 2019 row is a nucleus's
        WHOLE projection rather than its serotonergic subtype's: the dorsal
        raphe row alone predicts a striatum:cortex ratio of 4.3 against a
        measured 2.0, and it is the median raphe row that inverts it.

        The port that closes P5.1's open loop. 5-HT1A becomes ONE receptor
        type in THREE regions with three independent occupancies: the two
        raphe nuclei, where it is the somatodendritic autoreceptor, and the
        cortex, where it is the postsynaptic heteroreceptor. The hand-built
        scenario gave both roles a single shared occupancy state, because a
        `ReceptorSystem` has exactly one. The median raphe node is added by
        the port because the brief names MRN and a projection needs its
        source to be a node; it carries the same autoreceptor and no other
        receptor, and it is why an 8-OH-DPAT effect on the cortical field
        here is smaller than it would be with a single source.
        """,
    )


def _noradrenaline_port():
    return ScenarioPort(
        key="noradrenaline",
        transmitter="noradrenaline",
        base_network=_na.locus_coeruleus_cortical_network,
        nucleus_nodes={"LC": "LC_NA"},
        regions={
            "LC": "Locus ceruleus",
            "PFC": "Prelimbic/infralimbic cortex (the scenario's PFC_Pyramidal)",
        },
        placements=[
            ReceptorPlacement(
                name="alpha1",
                site=_na.receptor_site_alpha1,
                pk_models=_na.pk_models_alpha1,
                transmitter_ligand="noradrenaline",
                coupling={"PFC": {"PFC_Pyramidal": _na.ALPHA1_MECHANISM_GAIN}},
            ),
            ReceptorPlacement(
                name="alpha2",
                site=_na.receptor_site_alpha2,
                pk_models=_na.pk_models_alpha2,
                transmitter_ligand="noradrenaline",
                coupling={"LC": {"LC_NA": _na.ALPHA2_MECHANISM_GAIN}},
            ),
            ReceptorPlacement(
                name="beta",
                site=_na.receptor_site_beta,
                pk_models=_na.pk_models_beta,
                transmitter_ligand="noradrenaline",
                coupling={"PFC": {"PFC_Pyramidal": _na.BETA_MECHANISM_GAIN}},
            ),
        ],
        readouts={
            "noradrenergic_tone": _na.noradrenergic_tone_readout(),
            "cortical_output": _na.cortical_output_readout(),
        },
        source_readout="LC_NA",
        terminal_readout="PFC_Pyramidal",
        notes="""
        The port `noradrenaline.test_the_autoreceptor_is_an_open_loop_until_
        phase_6` was written in anticipation of. That test asserts, of the
        HAND-BUILT scenario, that clonidine moves the LC node while cortical
        noradrenaline, cortical occupancy and cortical firing do not, and it
        gives the structural reason: every ligand's concentration is driven
        by a PK model whose right-hand side is a function of its own state
        only. Here the noradrenaline concentration is driven by a projection
        system whose right-hand side reads `LC_NA`'s rate, so the reason no
        longer holds and the second half moves. Both statements are true, of
        two different objects, and both are tested.

        Noradrenaline's field was in FOLD-OF-BASAL units in P6.1, because
        P5.8 searched for a rodent basal extracellular noradrenaline
        concentration and found none. P6.2 found it -- [PanLai1995],
        1.59 +/- 0.37 nM by zero net flux in the medial prefrontal cortex of
        conscious rats -- so the field is in nM here and the alpha2
        autoreceptor at the LC carries 5.6% tonic occupancy rather than the
        3.6% the unit scale implied.
        """,
    )


def _acetylcholine_port():
    cortical = {"PFC": {"PFC_Pyramidal": _ach.CORTICAL_MECHANISM_GAIN}}
    return ScenarioPort(
        key="acetylcholine",
        transmitter="acetylcholine",
        base_network=_ach.basal_forebrain_cortical_network,
        nucleus_nodes={
            "NDB": "BF_Cholinergic",
            "PPN": "PPN_Cholinergic",
            "LDT": "LDT_Cholinergic",
        },
        regions={
            "NDB": "Diagonal band nucleus (basal forebrain)",
            "PPN": "Pedunculopontine nucleus",
            "LDT": "Laterodorsal tegmental nucleus",
            "PFC": "Prelimbic cortex (the scenario's PFC_Pyramidal)",
            "CP": (
                "Caudoputamen: the REFERENCE region, where [Vinson1997] "
                "measured 18.4 nM by no-net-flux microdialysis. No node and "
                "no receptor, and see the notes for what it costs"
            ),
        },
        placements=[
            ReceptorPlacement(
                name="alpha4beta2",
                site=_ach.receptor_site_alpha4beta2,
                pk_models=_ach.pk_models_alpha4beta2,
                transmitter_ligand="acetylcholine",
                coupling=dict(cortical),
            ),
            ReceptorPlacement(
                name="alpha7",
                site=_ach.receptor_site_alpha7,
                pk_models=_ach.pk_models_alpha7,
                transmitter_ligand="acetylcholine",
                coupling=dict(cortical),
            ),
            ReceptorPlacement(
                name="M1",
                site=_ach.receptor_site_m1,
                pk_models=_ach.pk_models_muscarinic,
                transmitter_ligand="acetylcholine",
                coupling=dict(cortical),
            ),
            ReceptorPlacement(
                name="M2",
                site=_ach.receptor_site_m2,
                pk_models=_ach.pk_models_muscarinic,
                transmitter_ligand="acetylcholine",
                coupling={
                    "NDB": {"BF_Cholinergic": _ach.M2_MECHANISM_GAIN},
                    "PPN": {"PPN_Cholinergic": _ach.M2_MECHANISM_GAIN},
                    "LDT": {"LDT_Cholinergic": _ach.M2_MECHANISM_GAIN},
                },
            ),
        ],
        readouts={
            "cholinergic_tone": _ach.cholinergic_tone_readout(),
            "cortical_output": _ach.cortical_output_readout(),
        },
        source_readout="BF_Cholinergic",
        terminal_readout="PFC_Pyramidal",
        notes="""
        Three source nuclei into one field, which is what the brief's
        "basal forebrain, PPT/LDT" grouping is. The M2 autoreceptor is
        placed in all three, which is what makes the loop closable; the
        PPT and LDT nodes are added by the port for the same reason the
        median raphe is added to the serotonin one.

        P6.3 makes the striatal caveat below LOAD-BEARING rather than
        marginal, and that is worth saying plainly. The caudoputamen is now
        the reference region, so [Vinson1997]'s 18.4 nM sets the absolute
        scale of the whole field, and the model attributes striatal
        acetylcholine entirely to the three projection nuclei when most of
        it is interneuronal. If interneurons supply the bulk of it, the
        derived release scale is too high by that factor and every region's
        predicted concentration is too high with it. Every fold-change this
        port reports is invariant to the error, because it multiplies all
        regions alike, and a test asserts that invariance.

        Acetylcholine's field is still the weakest of the five. Its basal
        concentration is a measurement since P6.2 ([Vinson1997], 18.4 nM by
        no-net-flux microdialysis), but in RAT STRIATUM, where extracellular
        acetylcholine comes from tonically active interneurons rather than
        from the three projection nuclei this port is built on; and its
        clearance rate constant is still illustrative. So the field has an
        absolute scale and its DYNAMICS are illustrative. A test asserts that
        no reported conclusion moves when the illustrative clearance rate is
        swept over two orders of magnitude, and `endogenous_tone.py` reports
        every acetylcholine number at [DeBundel2008]'s 4.6-fold lower value
        as well.
        """,
    )


def _histamine_port():
    return ScenarioPort(
        key="histamine",
        transmitter="histamine",
        base_network=_ha.tuberomammillary_cortical_network,
        nucleus_nodes={"TM": "TMN_HA"},
        regions={
            "TM": "Tuberomammillary nucleus",
            "PFC": "Prelimbic cortex (the scenario's PFC_Pyramidal)",
            "MEZ": (
                "Medial hypothalamic zone minus the tuberomammillary nucleus "
                "itself: the REFERENCE region, where [Itoh1991] put its "
                "probe. No node and no receptor"
            ),
        },
        placements=[
            ReceptorPlacement(
                name="H1",
                site=_ha.receptor_site_h1,
                pk_models=_ha.pk_models_h1,
                transmitter_ligand="histamine",
                coupling={"PFC": {"PFC_Pyramidal": _ha.H1_MECHANISM_GAIN}},
            )
        ],
        readouts={
            "histaminergic_tone": _ha.histaminergic_tone_readout(),
            "cortical_output": _ha.cortical_output_readout(),
        },
        source_readout="TMN_HA",
        terminal_readout="PFC_Pyramidal",
        notes="""
        The port with NO closed loop, and the reason is worth naming rather
        than working around: the histamine autoreceptor is H3, and P5.6 did
        not build an H3 receptor. So the tuberomammillary node carries no
        receptor, nothing can change its firing, and the histamine field is
        constant in every run here. What the port does demonstrate is the
        other half of a projection system, the one P5.8 first showed for
        dopamine: there is now tonic histamine at the cortical H1 receptor,
        so mepyramine given alone is no longer inert. P5.6 had to write that
        it was.

        P6.3 shrinks that tonic histamine 489-fold, and it is the largest
        single number this task moves. [Itoh1991]'s 7.8 nM was measured in
        the MEDIAL HYPOTHALAMUS, so that is the reference region; the
        derived map makes the tuberomammillary projection into the prelimbic
        cortex 489 times thinner than into the hypothalamus around it, hence
        0.016 nM of cortical histamine. P6.1's uniform map put 7.8 nM in
        both places, which was never a claim about the cortex. This is one,
        and nothing checks it: no cortical extracellular histamine
        measurement was found, so what the port now says about mepyramine
        rests entirely on a tracer ratio. `summary()` reports the mepyramine
        effect under both maps side by side rather than replacing one number
        with the other.
        """,
    )


def _dopamine_port():
    return ScenarioPort(
        key="dopamine",
        transmitter="dopamine",
        base_network=_mt.stabilized_indirect_pathway_network,
        nucleus_nodes={"VTA": "VTA_DA", "SNc": "SNc_DA"},
        regions={
            "CP": (
                "Caudoputamen (the scenario's Striatum_D2_MSN), and the "
                "REFERENCE region: [Bosse2011] measured 5.0 nM here"
            ),
            "VTA": "Ventral tegmental area, the D2 autoreceptor's own pool",
            "SNc": "Substantia nigra pars compacta, likewise",
        },
        placements=[
            ReceptorPlacement(
                name="D2",
                site=_scenarios.d2_receptor_site_calibrated,
                pk_models=_scenarios.d2_pk_models_calibrated,
                transmitter_ligand="dopamine",
                coupling={
                    "CP": {"Striatum_D2_MSN": D2_GAIN},
                    "VTA": {"VTA_DA": D2_GAIN},
                    "SNc": {"SNc_DA": D2_GAIN},
                },
            )
        ],
        readouts={
            "locomotor_activity": _scenarios.locomotor_activity_readout(),
        },
        source_readout="SNc_DA",
        terminal_readout="Thalamus",
        notes="""
        P5.8's transporter-fed D2 scenario, with the transporter's single
        pool replaced by a two-source projection field. It is the port that
        exercises the OTHER generalization: one region fed by two nuclei
        (VTA and SNc), where the serotonin port exercises one nucleus feeding
        several regions.

        P6.3 closes its loop. P6.1 left the D2 AUTORECEPTOR on the VTA and
        SNc nodes out on purpose, recording that "it is real, and adding it
        would have closed the dopamine loop, but it would also have added
        anatomy and a coupling gain the hand-built scenario does not have,
        to a port whose job is regression". P6.3's brief asks for it by
        name, so D2 becomes one receptor type in THREE regions: the
        caudoputamen, where it is the postsynaptic receptor on the
        indirect-pathway spiny neuron, and the two midbrain nuclei, where it
        is the somatodendritic autoreceptor on the dopamine neuron itself.

        The autoreceptor's gain is NOT a new number: it is
        `scenarios.d2_coupling()`'s own -3.0, reused unchanged. That follows
        the rule `serotonin.HT1A_MECHANISM_GAIN` already states for the
        5-HT1A autoreceptor and heteroreceptor -- "the same Gi-coupled
        receptor reducing the drive of the cell that carries it, and no data
        in reach distinguishes its per-receptor efficacy" between the two
        cells -- and it stays ILLUSTRATIVE for exactly the reason
        `d2_coupling` records at length. What the reuse buys is that
        closing the loop introduces no number at all: the anatomy is the
        derived density map's, the receptor is P2.2's calibrated site, and
        the gain is the one already in the file.

        The uptake blockers travel with the clearance object, so cocaine,
        methylphenidate, amphetamine and atomoxetine act here exactly as they
        do in P5.8, and haloperidol competes at the D2 site with 5.0 nM of
        endogenous dopamine, as it does there. What is new is that the
        dopamine is now released at a rate proportional to a midbrain node's
        firing, so silencing that node lowers it -- and now that the
        autoreceptor is in place, lowering it feeds back onto the firing.
        """,
    )


SCENARIO_PORTS = {
    "serotonin": _serotonin_port,
    "noradrenaline": _noradrenaline_port,
    "acetylcholine": _acetylcholine_port,
    "histamine": _histamine_port,
    "dopamine": _dopamine_port,
}

PORT_ORDER = ("dopamine", "serotonin", "noradrenaline", "acetylcholine", "histamine")


def port(key, densities=None, basal=None, density_map="connectome", table=None,
         autoreceptor=True):
    """Build one ported scenario by name.

    `density_map="uniform"` rebuilds the port at P6.1's illustrative-uniform
    densities, which is what makes every P6.3 change reportable as a
    difference. `densities` overrides individual regions; see `_targets`.

    `autoreceptor=False` OPENS every autoreceptor loop in the port by
    zeroing its coupling gain and changing nothing else, which is how P6.3
    measures what closing them costs and whether it costs stability.

    `basal` overrides the cited endogenous tone of this port's transmitter,
    in nM. It exists for `endogenous_tone.py`, which has to re-run a port at
    a counterfactual tone (P6.1's unit scale, or no tone at all) to report
    what the measured tone is worth; nothing in the modelling path passes it.
    """
    if key not in SCENARIO_PORTS:
        raise ValueError(
            "unknown port %r; have %s" % (key, sorted(SCENARIO_PORTS))
        )
    return PortedScenario(
        SCENARIO_PORTS[key](),
        densities=densities,
        basal=basal,
        density_map=density_map,
        table=table,
        autoreceptor=autoreceptor,
    )


def all_ports(density_map="connectome"):
    """Every ported scenario, in `PORT_ORDER`."""
    return [port(key, density_map=density_map) for key in PORT_ORDER]


# ---------------------------------------------------------------------------
# The two results the ports exist to produce
# ---------------------------------------------------------------------------

#: Per port, a drug acting at the somatodendritic autoreceptor, its dose in
#: nM, and the direction it should move the source nucleus's firing. The
#: doses are 100 nM throughout, the value the hand-built scenarios' own tests
#: use, so no port is shown at a dose chosen to flatter it.
#:
#: Acetylcholine's entry is an ANTAGONIST, not an agonist, because the only
#: muscarinic drug in `acetylcholine.py` is scopolamine and the M2 agonist is
#: acetylcholine itself, which the port no longer doses. Blocking an
#: inhibitory autoreceptor disinhibits the nucleus, so the expected direction
#: is the opposite one, and that sign flip is a prediction rather than a
#: convention.
AUTORECEPTOR_PROBE = {
    "serotonin": ("8-OH-DPAT", 100.0, -1),
    "noradrenaline": ("clonidine", 100.0, -1),
    "acetylcholine": ("scopolamine", 100.0, +1),
}

#: The ports whose transmitter field carries a MEASURED absolute
#: concentration scale, and in which a change in the field is therefore
#: visible as a change in receptor occupancy.
#:
#: **P6.2 made this ALL FIVE.** In P6.1 it was three, and the other two sat
#: at a unit scale of 1.0 nM from gaps P5.8 and P6.1 had recorded, so their
#: tonic occupancy was ~1e-4 and the closed loop was invisible at the
#: receptor. [PanLai1995] supplies the noradrenaline concentration
#: (1.59 nM, conscious rat mPFC, zero net flux) and [Vinson1997] the
#: acetylcholine one (18.4 nM, rat striatum, no-net-flux), so both fields now
#: carry an absolute scale and both loops are visible at the receptor. What
#: each one is worth, and by how much every number in this module moved, is
#: in `endogenous_tone.py`.
MEASURED_BASAL = ("dopamine", "serotonin", "noradrenaline", "acetylcholine", "histamine")

#: The basal concentrations P6.1 ran with, kept so that `endogenous_tone.py`
#: can re-run each port at them and report the change rather than assert it.
#: Only the two unit scales are listed; the other three transmitters' basal
#: levels are unchanged since P5.8 and P6.1.
P61_UNIT_SCALE_BASAL = {"noradrenaline": 1.0, "acetylcholine": 1.0}

#: Per port, an antagonist with no agonist alongside it. Before P6.1 (and
#: before P5.8 for dopamine) every one of these was inert by construction.
LONE_ANTAGONIST = {
    "serotonin": ("WAY-100635", 100.0),
    "noradrenaline": ("yohimbine", 100.0),
    "acetylcholine": ("scopolamine", 100.0),
    "histamine": ("mepyramine", 100.0),
    "dopamine": ("haloperidol", 100.0),
}


def closed_loop_effect(key, t_end=30.0, basal=None):
    """The four numbers that say the loop is closed, for one port.

    Returns a dict with the change, from the drug-free run, in: the source
    nucleus's rate, the terminal field's transmitter concentration as a fold
    of basal, the terminal region's occupancy by the transmitter itself, and
    the terminal node's rate. A closed loop moves all four; the hand-built
    scenario moves only the first.
    """
    if key not in AUTORECEPTOR_PROBE:
        raise ValueError("port %r has no autoreceptor probe" % (key,))
    scenario = port(key, basal=basal)
    drug, dose, direction = AUTORECEPTOR_PROBE[key]
    ligand = scenario.projection.transmitter
    source = scenario.port.source_readout
    terminal = scenario.port.terminal_readout

    net0, sim0, base = scenario.run(t_end=t_end)
    net1, sim1, dosed = scenario.run(t_end=t_end, **{drug: dose})

    terminal_system = next(
        s for s in scenario.receptor_systems if "PFC" in s.regions
    )
    return {
        "drug": drug,
        "dose_nM": dose,
        "direction": direction,
        "measured_basal": key in MEASURED_BASAL,
        "source_rate": (
            scenario.rate_trace(net1, dosed, source)[-1]
            - scenario.rate_trace(net0, base, source)[-1]
        ),
        "terminal_field_fold": (
            scenario.field_fold(sim1, dosed, "PFC")[-1]
            - scenario.field_fold(sim0, base, "PFC")[-1]
        ),
        "terminal_occupancy": (
            scenario.occupancy_trace(
                sim1, dosed, terminal_system.name, "PFC", ligand
            )[-1]
            - scenario.occupancy_trace(
                sim0, base, terminal_system.name, "PFC", ligand
            )[-1]
        ),
        "terminal_rate": (
            scenario.rate_trace(net1, dosed, terminal)[-1]
            - scenario.rate_trace(net0, base, terminal)[-1]
        ),
    }


def lone_antagonist_effect(key, t_end=30.0, basal=None):
    """The change in the terminal readout when an antagonist is given with no
    agonist alongside it. Nonzero only because there is tonic transmitter."""
    scenario = port(key, basal=basal)
    drug, dose = LONE_ANTAGONIST[key]
    terminal = scenario.port.terminal_readout
    net0, _, base = scenario.run(t_end=t_end)
    net1, _, dosed = scenario.run(t_end=t_end, **{drug: dose})
    return {
        "drug": drug,
        "dose_nM": dose,
        "terminal_rate": (
            scenario.rate_trace(net1, dosed, terminal)[-1]
            - scenario.rate_trace(net0, base, terminal)[-1]
        ),
    }


def autoreceptor_self_weight(key, region, delta=0.01, density_map="connectome"):
    """The effective SELF-CONNECTION one closed autoreceptor loop adds.

    The loop this task closes does not appear in `network.connectivity`: it
    leaves the circuit at the nucleus's firing rate, goes through that
    region's transmitter concentration and the autoreceptor's occupancy, and
    comes back as a term in the same node's drive. On the slow manifold --
    the concentration pool relaxes in 0.05-2 s and the occupancy in seconds,
    against a circuit time constant of 1-1.5 min -- that is exactly a
    self-weight:

        w_eff = d(drive at the nucleus) / d(rate of the nucleus)

    evaluated with the pool and the occupancy at their steady states for the
    held rate. Computed by central difference rather than by differentiating
    a chain of four functions, because two of the four (the site's occupancy
    law, the clearance's uptake law) have no analytic derivative in this
    codebase and a difference of exact values beats a derivative of
    approximations.
    """
    scenario = port(key, density_map=density_map)
    projection = scenario.projection
    node = None
    for nucleus, name in scenario.port.nucleus_nodes.items():
        if nucleus == region:
            node = name
    if node is None:
        raise ValueError(
            "region %r is not a source nucleus of the %s port" % (region, key)
        )
    rates = np.array([s.rest_rate for s in projection.sources], dtype=float)
    row = list(projection.source_nodes).index(node)

    def drive_at(rate):
        held = rates.copy()
        held[row] = rate
        concentration = projection.steady_state(region, held)
        total = 0.0
        for system in scenario.receptor_systems:
            if region not in system.regions:
                continue
            names = [lig.name for lig in system.receptor_site.ligands]
            vector = np.zeros(system.n)
            for ligand_name, source in system.transmitter_sources.items():
                if _projection.as_projection_source(
                    source, projection_names={projection.name: projection},
                    region=region,
                ) is None:
                    continue
                vector[names.index(ligand_name)] = concentration
            activation = system.receptor_site.activation(
                _projection.resting_occupancy(system.receptor_site, vector)
            )
            scaled = activation * system.density[region]
            total += sum(
                gain * scaled for gain in system.coupling.get(region, {}).values()
            )
        return total

    rest = float(rates[row])
    return (drive_at(rest + delta) - drive_at(rest - delta)) / (2.0 * delta)


def stability_report(t_end=30.0):
    """P3.4's check on the WHOLE coupled system, with each loop open and
    closed. The third acceptance criterion of P6.3, as a computation."""
    from .. import stability as _stability

    lines = [
        "Closed-loop stability (P6.3): P3.4's check on the whole coupled "
        "system",
        "=" * 78,
        "",
        "`stability.assess` cannot see an autoreceptor loop: the feedback "
        "path leaves the",
        "circuit through a transmitter concentration and a receptor "
        "occupancy, and neither is",
        "an entry of `network.connectivity`. So the check is run on "
        "`Simulation.rhs` instead,",
        "with every state in it, once with the loops OPEN and once CLOSED.",
        "",
        "%-14s %-7s %6s %11s %11s %10s %9s"
        % ("port", "loop", "ok", "max Re eig", "residual", "spread", "min sens"),
    ]
    for key in PORT_ORDER:
        for closed in (False, True):
            scenario = port(key, autoreceptor=closed)
            assessment = _stability.assess_system(scenario.simulation())
            lines.append(
                "%-14s %-7s %6s %+11.3e %11.2e %10.2e %9.4f"
                % (
                    key,
                    "closed" if closed else "open",
                    "ok" if assessment.ok else "NOT OK",
                    assessment.abscissa,
                    assessment.residuals.max(),
                    assessment.spread,
                    assessment.saturation.min_sensitivity,
                )
            )
    lines.append("")
    lines.append(
        "The effective SELF-WEIGHT each closed loop adds to its nucleus, "
        "which is what the"
    )
    lines.append(
        "circuit-only Jacobian is missing. Negative is negative feedback."
    )
    lines.append("")
    lines.append("%-14s %-6s %14s" % ("port", "region", "w_eff"))
    for key in PORT_ORDER:
        scenario = port(key)
        for nucleus in scenario.port.nucleus_nodes:
            if nucleus not in scenario.projection.regions:
                continue
            carries = any(
                nucleus in system.regions and system.coupling.get(nucleus)
                for system in scenario.receptor_systems
            )
            if not carries:
                continue
            lines.append(
                "%-14s %-6s %14.6f"
                % (key, nucleus, autoreceptor_self_weight(key, nucleus))
            )
    return "\n".join(lines)


def summary():
    """Both results for every port, as text. Written to `results/` by
    `scripts/report_projection_systems.py`."""
    lines = ["Ported scenarios (P6.1, re-run at P6.2's endogenous tone)", "=" * 72, ""]
    lines.append(
        "The autoreceptor loop, closed: change from the drug-free run at the"
    )
    lines.append("end of a 30 min run. All four columns move only if the loop is")
    lines.append("closed; the hand-built scenario moves the first and nothing else.")
    lines.append("")
    lines.append(
        "%-14s %-12s %10s %12s %12s %11s  %s"
        % (
            "port",
            "drug",
            "source r",
            "field fold",
            "occupancy",
            "terminal r",
            "basal scale",
        )
    )
    for key in PORT_ORDER:
        if key not in AUTORECEPTOR_PROBE:
            continue
        effect = closed_loop_effect(key)
        lines.append(
            "%-14s %-12s %+10.4f %+12.4f %+12.2e %+11.4f  %s"
            % (
                key,
                effect["drug"],
                effect["source_rate"],
                effect["terminal_field_fold"],
                effect["terminal_occupancy"],
                effect["terminal_rate"],
                "measured" if effect["measured_basal"] else "UNIT (see MEASURED_BASAL)",
            )
        )
    lines.append("")
    lines.append(
        "In P6.1 the noradrenaline and acetylcholine fields sat at a unit scale"
    )
    lines.append(
        "of 1.0 nM, so the loop was closed in the concentration and invisible at"
    )
    lines.append(
        "the receptor. P6.2 supplies both measurements, so every row above is a"
    )
    lines.append(
        "measured scale; `endogenous_tone.py` reports how far each one moved."
    )
    lines.append("")
    lines.append("An antagonist given ALONE is no longer inert")
    lines.append("%-14s %-14s %12s  %s" % ("port", "antagonist", "terminal r", "basal scale"))
    for key in PORT_ORDER:
        effect = lone_antagonist_effect(key)
        lines.append(
            "%-14s %-14s %+12.5f  %s"
            % (
                key,
                effect["drug"],
                effect["terminal_rate"],
                "measured" if key in MEASURED_BASAL else "UNIT",
            )
        )
    return "\n".join(lines)
