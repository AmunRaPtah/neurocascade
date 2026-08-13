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

The density maps are ILLUSTRATIVE AND UNIFORM, on purpose
----------------------------------------------------------

Every `ProjectionTarget` built here has density 1.0 from every source that
innervates it, tagged `provenance="illustrative-uniform"`. That is the map
that asserts nothing: no ordering between regions, because P6.1 has no
measured ordering to assert. P6.3 is the task that derives real densities
from the Phase 3 connectome, and the shape of what it will change is already
visible in `ProjectionSystem.density_report()`.

Two consequences worth naming. Under a uniform map every target region's
predicted basal concentration equals the reference region's measured one, so
no port here claims that cortical dopamine is lower than striatal dopamine
even though the model would say so given a real density map (see
`test_projection.py`, which exercises a non-uniform map). And where a field
has two or more sources, a uniform map makes them equal contributors, so
the dopamine port's VTA and SNc move striatal dopamine by the same amount,
which is not the anatomy and is not claimed to be.

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
  opposite.
"""

import numpy as np

from . import acetylcholine as _ach
from . import histamine as _ha
from . import monoamine_transporters as _mt
from . import noradrenaline as _na
from . import projection as _projection
from . import projection_systems as _ps
from . import scenarios as _scenarios
from . import serotonin as _5ht
from .circuit import CircuitNode
from .engine import Simulation
from .network import CircuitNetwork
from .projection import ProjectionTarget, RegionalReceptorSystem


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

#: The provenance tag every density in this module carries.
DENSITY_PROVENANCE = "illustrative-uniform"


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
        self.regions = dict(regions)
        self.placements = list(placements)
        self.readouts = dict(readouts)
        self.source_readout = source_readout
        self.terminal_readout = terminal_readout
        self.notes = notes


def _targets(port, densities=None):
    """One `ProjectionTarget` per region.

    Uniform (1.0 from every nucleus into every region) unless `densities`
    overrides it, in which case the provenance says so. The override exists
    for two callers and no third: a test that has to show the abstraction
    CAN produce region-wise differences, and P6.3, which is the task that
    supplies real densities.
    """
    uniform = {node: 1.0 for node in port.nucleus_nodes.values()}
    densities = dict(densities or {})
    return [
        ProjectionTarget(
            region,
            dict(densities.get(region, uniform)),
            provenance=(
                "override" if region in densities else DENSITY_PROVENANCE
            ),
        )
        for region in port.regions
    ]


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

    def __init__(self, port, densities=None):
        self.port = port
        self.key = port.key
        self.transmitter = port.transmitter
        self.base_network = add_nodes(
            port.base_network(), sorted(port.nucleus_nodes.values())
        )
        self.projection = _ps.projection_system(
            port.transmitter,
            _targets(port, densities),
            REST_RATE,
            nodes=port.nucleus_nodes,
            name=port.transmitter,
        )
        self.receptor_systems = [
            placement.build(port.transmitter) for placement in port.placements
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
        which is the unit every conclusion here is stated in and the only
        unit the two illustrative-basal fields (acetylcholine, and
        noradrenaline through P5.8's unit scale) support at all."""
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
                "  %-6s (%s)  density %.2f (%.2fx reference, %s)  basal %.4g nM"
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
            "PFC": "Prelimbic/infralimbic cortex (the scenario's PFC_Pyramidal)",
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

        Noradrenaline's field is in FOLD-OF-BASAL units, not nM: P5.8
        searched for a rodent basal extracellular noradrenaline concentration
        and found none, so NET carries a unit scale.
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
            "PFC": "Prelimbic/infralimbic cortex (the scenario's PFC_Pyramidal)",
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

        Acetylcholine's field is the weakest of the five and is labelled as
        such: neither its clearance rate constant nor its basal
        concentration is a measurement this project could reach (see
        `projection_systems.py`, "Searched and empty"), so the field is in
        fold-of-basal units and its dynamics are illustrative. What the port
        demonstrates for acetylcholine is the STRUCTURE, and a test asserts
        that no reported conclusion moves when the illustrative clearance
        rate is swept over two orders of magnitude.
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
            "PFC": "Prelimbic/infralimbic cortex (the scenario's PFC_Pyramidal)",
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
        7.8 nM of it ([Itoh1991], with four stated caveats), so mepyramine
        given alone is no longer inert. P5.6 had to write that it was.
        """,
    )


def _dopamine_port():
    return ScenarioPort(
        key="dopamine",
        transmitter="dopamine",
        base_network=_mt.stabilized_indirect_pathway_network,
        nucleus_nodes={"VTA": "VTA_DA", "SNc": "SNc_DA"},
        regions={"CP": "Caudoputamen (the scenario's Striatum_D2_MSN)"},
        placements=[
            ReceptorPlacement(
                name="D2",
                site=_scenarios.d2_receptor_site_calibrated,
                pk_models=_scenarios.d2_pk_models_calibrated,
                transmitter_ligand="dopamine",
                coupling={
                    "CP": {
                        "Striatum_D2_MSN": _scenarios.d2_coupling()["Striatum_D2_MSN"]
                    }
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
        several regions. Under the uniform density map the two contribute
        equally, which is not the anatomy; P6.3 is what fixes that.

        The uptake blockers travel with the clearance object, so cocaine,
        methylphenidate, amphetamine and atomoxetine act here exactly as they
        do in P5.8, and haloperidol competes at the D2 site with 5.0 nM of
        endogenous dopamine, as it does there. What is new is that the
        dopamine is now released at a rate proportional to a midbrain node's
        firing, so silencing that node lowers it.
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


def port(key, densities=None):
    """Build one ported scenario by name.

    `densities` overrides the uniform map for the named regions; see
    `_targets`. Nothing in this module passes it.
    """
    if key not in SCENARIO_PORTS:
        raise ValueError(
            "unknown port %r; have %s" % (key, sorted(SCENARIO_PORTS))
        )
    return PortedScenario(SCENARIO_PORTS[key](), densities=densities)


def all_ports():
    """Every ported scenario, in `PORT_ORDER`."""
    return [port(key) for key in PORT_ORDER]


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
#: The other two are in fold-of-basal units, from gaps P5.8 and P6.1
#: recorded: no rodent basal extracellular noradrenaline concentration and no
#: recovery-corrected basal acetylcholine concentration were found. Both
#: fields therefore sit at 1.0 nM, which is three to four orders of magnitude
#: below either transmitter's Kd at any receptor in this codebase, so tonic
#: occupancy is ~1e-4 and a halving of the field moves it by ~5e-5. **The
#: loop is closed in the concentration and invisible at the receptor**, and
#: that is a consequence of a missing measurement rather than of the
#: abstraction. It is asserted as such by a test rather than glossed.
MEASURED_BASAL = ("dopamine", "serotonin", "histamine")

#: Per port, an antagonist with no agonist alongside it. Before P6.1 (and
#: before P5.8 for dopamine) every one of these was inert by construction.
LONE_ANTAGONIST = {
    "serotonin": ("WAY-100635", 100.0),
    "noradrenaline": ("yohimbine", 100.0),
    "acetylcholine": ("scopolamine", 100.0),
    "histamine": ("mepyramine", 100.0),
    "dopamine": ("haloperidol", 100.0),
}


def closed_loop_effect(key, t_end=30.0):
    """The four numbers that say the loop is closed, for one port.

    Returns a dict with the change, from the drug-free run, in: the source
    nucleus's rate, the terminal field's transmitter concentration as a fold
    of basal, the terminal region's occupancy by the transmitter itself, and
    the terminal node's rate. A closed loop moves all four; the hand-built
    scenario moves only the first.
    """
    if key not in AUTORECEPTOR_PROBE:
        raise ValueError("port %r has no autoreceptor probe" % (key,))
    scenario = port(key)
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


def lone_antagonist_effect(key, t_end=30.0):
    """The change in the terminal readout when an antagonist is given with no
    agonist alongside it. Nonzero only because there is tonic transmitter."""
    scenario = port(key)
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


def summary():
    """Both results for every port, as text. Written to `results/` by
    `scripts/report_projection_systems.py`."""
    lines = ["Ported scenarios (P6.1)", "=" * 72, ""]
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
        "Where the basal concentration is a unit rather than a measurement the"
    )
    lines.append(
        "field moves and the receptor cannot see it: 1.0 nM is 3-4 orders below"
    )
    lines.append("any Kd in this codebase. That is a missing measurement, not a")
    lines.append("property of the abstraction.")
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
