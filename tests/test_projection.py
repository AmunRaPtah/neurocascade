"""P6.1: the projection-system mechanism.

Mechanism only. The cited constants are `test_projection_systems.py`'s and
the ported scenarios are `test_projection_scenarios.py`'s, the same split
`test_receptor.py` / `test_opioid_scenario.py` already uses.
"""

import numpy as np
import pytest

from neurocascade import monoamine_transporters as mt
from neurocascade import projection
from neurocascade import scenarios
from neurocascade import transport
from neurocascade.circuit import CircuitNode
from neurocascade.engine import ReceptorSystem, Simulation
from neurocascade.network import CircuitNetwork
from neurocascade.projection import (
    FirstOrderClearance,
    PowerLawRelease,
    ProjectionSource,
    ProjectionSystem,
    ProjectionTarget,
    ProportionalRelease,
    RegionalReceptorSystem,
    SourceNucleus,
    as_projection_source,
    compensate_baselines,
    concentration_for_uptake,
    standing_drive,
)
from neurocascade.receptor import Ligand, ReceptorSite


REST = 0.5


# ---------------------------------------------------------------------------
# Release laws
# ---------------------------------------------------------------------------


def test_proportional_release_is_unity_at_rest_and_linear_around_it():
    law = ProportionalRelease(REST)
    assert law(REST) == pytest.approx(1.0)
    assert law(2 * REST) == pytest.approx(2.0)
    assert law(0.0) == pytest.approx(0.0)


def test_proportional_release_floors_at_zero_rather_than_going_negative():
    # A Wilson-Cowan rate cannot be negative, but the law is total anyway so
    # that a caller handing it one gets zero release instead of negative
    # release, which would be transmitter flowing backwards into the axon.
    assert ProportionalRelease(REST)(-1.0) == 0.0


def test_power_law_release_with_exponent_one_is_the_proportional_law():
    # The sensitivity knob must reduce EXACTLY to the law every result uses,
    # or a sweep over it would not be a sweep around the reported value.
    proportional = ProportionalRelease(REST)
    power = PowerLawRelease(REST, exponent=1.0)
    for rate in (0.0, 0.1, REST, 0.9, 1.0):
        assert power(rate) == pytest.approx(proportional(rate))


def test_release_laws_reject_a_non_positive_rest_rate():
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError):
            ProportionalRelease(bad)
        with pytest.raises(ValueError):
            PowerLawRelease(bad, exponent=1.0)
    with pytest.raises(ValueError):
        PowerLawRelease(REST, exponent=0.0)


# ---------------------------------------------------------------------------
# Clearance
# ---------------------------------------------------------------------------


def test_first_order_clearance_agrees_with_the_transporter_it_duplicates():
    # `FirstOrderClearance` exists because acetylcholine is not cleared by a
    # monoamine transporter, and it shares the ALGEBRA of
    # `MonoamineTransporter.first_order` deliberately. This test is what stops
    # the duplication drifting into a difference.
    rate, basal = 30.6, 4.0
    ligands = [transport.TransporterLigand("blocker", ki=100.0)]
    enzyme = FirstOrderClearance(
        "AChE", "acetylcholine", rate=rate, basal=basal, ligands=ligands
    )
    carrier = transport.MonoamineTransporter.first_order(
        "NET",
        "noradrenaline",
        [transport.TransporterLigand("blocker", ki=100.0)],
        clearance_rate=rate,
        basal=basal,
    )
    assert enzyme.basal_release_rate() == pytest.approx(carrier.basal_release_rate())
    assert enzyme.effective_clearance_rate == pytest.approx(
        carrier.effective_clearance_rate
    )
    for drug in (0.0, 50.0, 1000.0):
        concentrations = np.array([drug])
        assert enzyme.inhibition_factor(concentrations) == pytest.approx(
            carrier.inhibition_factor(concentrations)
        )
        for c in (0.0, basal, 10 * basal):
            assert enzyme.uptake(c, concentrations) == pytest.approx(
                carrier.uptake(c, concentrations)
            )


def test_first_order_clearance_rejects_impossible_constants():
    for kwargs in (
        dict(rate=0.0, basal=1.0),
        dict(rate=-1.0, basal=1.0),
        dict(rate=1.0, basal=0.0),
        dict(rate=1.0, basal=None),
    ):
        with pytest.raises(ValueError):
            FirstOrderClearance("x", "acetylcholine", **kwargs)


def test_concentration_for_uptake_inverts_the_uptake_law():
    for clearance in (mt.dat(), mt.net(), FirstOrderClearance("e", "t", 5.0, 2.0)):
        for target in (1e-3, 1.0, 10.0):
            c = concentration_for_uptake(clearance, target)
            assert clearance.uptake(c, np.zeros(clearance.n)) == pytest.approx(target)
    assert concentration_for_uptake(mt.dat(), 0.0) == 0.0


def test_concentration_for_uptake_refuses_a_demand_above_vmax():
    dat = mt.dat()
    with pytest.raises(ValueError, match="Vmax"):
        concentration_for_uptake(dat, dat.vmax)


# ---------------------------------------------------------------------------
# ProjectionSystem: the mass balance
# ---------------------------------------------------------------------------


def one_source_system(density=1.0, clearance=None, extra=None):
    clearance = mt.dat() if clearance is None else clearance
    targets = [ProjectionTarget("CP", density)]
    if extra is not None:
        targets.append(ProjectionTarget("PFC", extra))
    return ProjectionSystem(
        name="dopamine",
        transmitter=clearance.transmitter,
        sources=[SourceNucleus("SNc", REST)],
        targets=targets,
        clearance=clearance,
        pk_models=mt.pk_models([lig.name for lig in clearance.ligands]),
        reference_region="CP",
    )


def test_a_single_region_at_rest_reproduces_the_transporter_it_generalizes():
    # P6.1 is P5.8's mechanism with the release constant replaced by a
    # firing-rate-driven term. With one region, one source at its rest rate
    # and the same clearance object, the two right-hand sides must agree
    # TERM FOR TERM, not merely qualitatively.
    dat = mt.dat()
    system = one_source_system(clearance=dat)
    drugs = np.zeros(dat.n)
    for c in (1.0, dat.basal, 20.0):
        projected = system.concentration_rhs(np.array([c]), [REST], drugs)
        transported = dat.transmitter_rhs(np.array([c]), drugs)
        assert projected[0] == pytest.approx(transported[0])


def test_the_reference_region_reproduces_the_measured_basal_concentration():
    # The release scale is DERIVED, not chosen: it is whatever makes the
    # reference region sit at the clearance object's measured basal level.
    for clearance in (mt.dat(), mt.sert(), mt.net()):
        system = ProjectionSystem(
            name=clearance.transmitter,
            transmitter=clearance.transmitter,
            sources=[SourceNucleus("N", REST)],
            targets=[ProjectionTarget("A", 1.0), ProjectionTarget("B", 0.25)],
            clearance=clearance,
            pk_models=mt.pk_models([lig.name for lig in clearance.ligands]),
            reference_region="A",
        )
        assert system.basal_concentration("A") == pytest.approx(clearance.basal)


def test_a_sparser_region_predicts_a_lower_basal_concentration():
    # This is the abstraction's first genuine prediction: given a density map
    # and uniform uptake kinetics, every non-reference region's drug-free
    # concentration follows. In the first-order limit it is exactly
    # proportional; under Michaelis-Menten the density-to-concentration map is
    # CONVEX, so a region at a quarter of the reference's density sits at
    # slightly LESS than a quarter of its concentration. The direction of the
    # departure is asserted rather than its size.
    linear = ProjectionSystem(
        name="noradrenaline",
        transmitter="noradrenaline",
        sources=[SourceNucleus("LC", REST)],
        targets=[ProjectionTarget("A", 1.0), ProjectionTarget("B", 0.25)],
        clearance=mt.net(),
        pk_models=mt.pk_models([lig.name for lig in mt.net().ligands]),
        reference_region="A",
    )
    ratio = linear.basal_concentration("B") / linear.basal_concentration("A")
    assert ratio == pytest.approx(0.25)

    saturating = one_source_system(density=1.0, extra=0.25)
    mm_ratio = (
        saturating.basal_concentration("PFC") / saturating.basal_concentration("CP")
    )
    assert mm_ratio < 0.25
    # Convex, but only just: 5 nM against a Km of 160 nM is deep enough into
    # the linear regime that the departure from proportionality is ~1%.
    assert mm_ratio > 0.24


def test_a_region_denser_than_uptake_can_serve_is_an_error_not_a_number():
    # With uniform uptake kinetics there is a density above which no finite
    # steady state exists. That is the model saying something; it must not be
    # a silent 1e30.
    dat = mt.dat()
    huge = dat.vmax / dat.basal_release_rate() * 2.0
    with pytest.raises(ValueError, match="Vmax"):
        one_source_system(density=1.0, extra=huge)


def test_integrating_the_field_reaches_the_algebraic_steady_state():
    from scipy.integrate import solve_ivp

    system = one_source_system(extra=0.3)
    drugs = np.zeros(system.n)
    for rate in (REST, 0.8, 0.2):
        sol = solve_ivp(
            lambda t, y: system.concentration_rhs(y, [rate], drugs),
            (0.0, 20.0),
            system.initial_state(),
            method="LSODA",
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol.success
        for region, value in zip(system.regions, sol.y[:, -1]):
            assert value == pytest.approx(system.steady_state(region, [rate]), rel=1e-6)


def test_doubling_the_firing_rate_doubles_the_field_in_the_linear_regime():
    # The chain the brief names, at its simplest: firing rate -> release ->
    # concentration. Asserted in the first-order clearance form, where the
    # proportionality is exact.
    system = ProjectionSystem(
        name="noradrenaline",
        transmitter="noradrenaline",
        sources=[SourceNucleus("LC", REST)],
        targets=[ProjectionTarget("PFC", 1.0)],
        clearance=mt.net(),
        pk_models=mt.pk_models([lig.name for lig in mt.net().ligands]),
    )
    basal = system.basal_concentration("PFC")
    assert system.steady_state("PFC", [2 * REST]) == pytest.approx(2 * basal)
    assert system.steady_state("PFC", [0.0]) == pytest.approx(0.0)


def test_two_source_nuclei_add_into_one_field():
    # The VTA/SNc case, which is why `sources` is a list. Two nuclei at half
    # the density each must produce exactly what one at full density does,
    # and silencing one must halve the release rather than the field's
    # existence.
    dat = mt.dat()
    pk = mt.pk_models([lig.name for lig in dat.ligands])
    two = ProjectionSystem(
        name="dopamine",
        transmitter="dopamine",
        sources=[SourceNucleus("VTA", REST), SourceNucleus("SNc", REST)],
        targets=[ProjectionTarget("CP", {"VTA": 0.5, "SNc": 0.5})],
        clearance=dat,
        pk_models=pk,
    )
    one = one_source_system(density=1.0)
    assert two.basal_concentration("CP") == pytest.approx(one.basal_concentration("CP"))
    assert two.release_rate("CP", [REST, REST]) == pytest.approx(
        one.release_rate("CP", [REST])
    )
    assert two.release_rate("CP", [0.0, REST]) == pytest.approx(
        0.5 * one.release_rate("CP", [REST])
    )


def test_a_bare_density_is_rejected_when_there_is_more_than_one_source():
    dat = mt.dat()
    with pytest.raises(ValueError, match="source_node"):
        ProjectionSystem(
            name="dopamine",
            transmitter="dopamine",
            sources=[SourceNucleus("VTA", REST), SourceNucleus("SNc", REST)],
            targets=[ProjectionTarget("CP", 1.0)],
            clearance=dat,
            pk_models=mt.pk_models([lig.name for lig in dat.ligands]),
        )


def test_a_density_naming_an_unknown_source_is_an_error():
    dat = mt.dat()
    with pytest.raises(ValueError, match="does not have"):
        ProjectionSystem(
            name="dopamine",
            transmitter="dopamine",
            sources=[SourceNucleus("VTA", REST)],
            targets=[ProjectionTarget("CP", {"SNc": 1.0})],
            clearance=dat,
            pk_models=mt.pk_models([lig.name for lig in dat.ligands]),
        )


def test_a_projection_system_rejects_a_clearance_for_another_transmitter():
    with pytest.raises(ValueError, match="clears"):
        ProjectionSystem(
            name="serotonin",
            transmitter="serotonin",
            sources=[SourceNucleus("DR", REST)],
            targets=[ProjectionTarget("PFC", 1.0)],
            clearance=mt.dat(),
            pk_models=mt.pk_models([lig.name for lig in mt.dat().ligands]),
        )


def test_an_uptake_blocker_raises_every_region_by_the_same_factor():
    # P5.8's cancellation law, lifted to a field: with blockers only, the
    # steady state is `(1 + I/Ki)` times basal in EVERY region, whatever the
    # density map, because the density cancels along with Km, Vmax and basal.
    system = one_source_system(extra=0.3)
    drugs = np.zeros(system.n)
    drugs[[lig.name for lig in system.ligands].index("cocaine")] = 490.0  # 1x Ki
    folds = [
        system.steady_state(region, [REST], drugs)
        / system.basal_concentration(region)
        for region in system.regions
    ]
    assert folds[0] == pytest.approx(folds[1], rel=1e-2)
    assert folds[0] == pytest.approx(2.0, rel=1e-2)


def test_projection_target_and_system_validation():
    with pytest.raises(ValueError):
        ProjectionTarget("CP", -1.0)
    dat = mt.dat()
    pk = mt.pk_models([lig.name for lig in dat.ligands])
    with pytest.raises(ValueError, match="at least one target"):
        ProjectionSystem("d", "dopamine", [SourceNucleus("VTA", REST)], [], dat, pk)
    with pytest.raises(ValueError, match="at least one source"):
        ProjectionSystem("d", "dopamine", [], [ProjectionTarget("CP", 1.0)], dat, pk)
    with pytest.raises(ValueError, match="duplicate target"):
        ProjectionSystem(
            "d",
            "dopamine",
            [SourceNucleus("VTA", REST)],
            [ProjectionTarget("CP", 1.0), ProjectionTarget("CP", 1.0)],
            dat,
            pk,
        )
    with pytest.raises(ValueError, match="releases nothing"):
        ProjectionSystem(
            "d", "dopamine", [SourceNucleus("VTA", REST)],
            [ProjectionTarget("CP", 0.0)], dat, pk,
        )
    with pytest.raises(ValueError, match="one PK model per"):
        ProjectionSystem(
            "d", "dopamine", [SourceNucleus("VTA", REST)],
            [ProjectionTarget("CP", 1.0)], dat, [],
        )


# ---------------------------------------------------------------------------
# RegionalReceptorSystem
# ---------------------------------------------------------------------------


def two_state_site():
    return ReceptorSite(
        "R",
        [
            Ligand("dopamine", kon=11.82, koff=5.82, efficacy=1.0),
            Ligand("blocker", kon=1.0, koff=1.0, efficacy=0.0),
        ],
    )


def test_a_regional_system_owns_one_state_block_per_region():
    site = two_state_site()
    system = RegionalReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        regions=["CP", "PFC"],
        coupling={"CP": -3.0, "PFC": {"PFC_Pyramidal": 1.0}},
    )
    assert system.n == 2  # ligands, SHARED across regions
    assert system.n_states == 4  # two regions x two states
    assert list(system.state_rows("CP")) == [0, 1]
    assert list(system.state_rows("PFC")) == [2, 3]
    # A bare gain means "the region's own node".
    assert system.coupling["CP"] == {"CP": -3.0}
    assert system.coupling["PFC"] == {"PFC_Pyramidal": 1.0}


def _pk():
    from neurocascade.pk import OneCompartmentPK

    return OneCompartmentPK(ke=1.0)


def test_a_regional_system_rejects_regions_it_does_not_carry():
    site = two_state_site()
    for kwargs in (
        dict(coupling={"MOp": 1.0}),
        dict(coupling={"CP": 1.0}, density={"MOp": 1.0}),
        dict(coupling={"CP": 1.0}, connection_modulation={"MOp": {("a", "b"): 1.0}}),
    ):
        with pytest.raises(ValueError):
            RegionalReceptorSystem(
                pk_models=[_pk() for _ in site.ligands],
                receptor_site=site,
                regions=["CP", "PFC"],
                **kwargs
            )
    with pytest.raises(ValueError, match="not on this receptor site"):
        RegionalReceptorSystem(
            pk_models=[_pk() for _ in site.ligands],
            receptor_site=site,
            regions=["CP"],
            coupling={"CP": 1.0},
            transmitter_sources={"nonsense": "dopamine"},
        )


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def test_a_bare_string_means_a_transporter_unless_it_names_a_projection():
    # P5.8's meaning is preserved exactly, which is why a `Simulation`
    # requires the two name spaces to be disjoint.
    assert as_projection_source("DAT") is None
    assert as_projection_source("DAT", projection_names={"dopamine"}) is None
    assert as_projection_source(
        "dopamine", projection_names={"dopamine"}, region="CP"
    ) == ProjectionSource("dopamine", "CP")
    assert as_projection_source(("dopamine", "PFC")) == ProjectionSource(
        "dopamine", "PFC"
    )
    with pytest.raises(ValueError):
        as_projection_source(3.0)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def simple_network(names=("SNc", "CP", "PFC_Pyramidal")):
    nodes = [
        CircuitNode(name=name, tau=1.0, baseline_input=0.0, gain=4.0)
        for name in names
    ]
    return CircuitNetwork(nodes=nodes, connectivity=np.zeros((len(names), len(names))))


def field_and_receptors(regions=("CP", "PFC_Pyramidal"), densities=(1.0, 0.3)):
    dat = mt.dat()
    field = ProjectionSystem(
        name="dopamine",
        transmitter="dopamine",
        sources=[SourceNucleus("SNc", REST)],
        targets=[
            ProjectionTarget(region, density)
            for region, density in zip(regions, densities)
        ],
        clearance=dat,
        pk_models=mt.pk_models([lig.name for lig in dat.ligands]),
        reference_region=regions[0],
    )
    site = two_state_site()
    receptors = RegionalReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        regions=list(regions),
        coupling={region: 1.0 for region in regions},
        transmitter_sources={"dopamine": "dopamine"},
        name="D2ish",
    )
    return field, receptors


def test_a_projection_only_simulation_is_legitimate_and_flat_without_drug():
    field, _ = field_and_receptors()
    sim = Simulation(network=simple_network(), projection_systems=[field])
    assert sim.r == 0 and sim.m == 0 and sim.p == 2
    result = sim.run(
        doses=np.zeros(sim.n_ligands),
        t_span=(0.0, 10.0),
        t_eval=np.linspace(0.0, 10.0, 21),
        rates0=np.full(3, REST),
        method="LSODA",
    )
    for region in field.regions:
        trace = sim.projection_trace(result, "dopamine", region)
        assert np.allclose(trace, field.basal_concentration(region), rtol=1e-8)


def test_the_projection_block_is_empty_and_the_layout_unchanged_without_one():
    # Every simulation predating P6.1 must see the same state vector it saw.
    site = two_state_site()
    sim = Simulation(
        network=simple_network(),
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        coupling={"CP": 1.0},
    )
    assert sim.p == 0
    assert sim.projection_index == {}
    result = sim.run(
        doses=np.zeros(sim.n_ligands), t_span=(0.0, 1.0), t_eval=[0.0, 1.0]
    )
    assert result["projection"].shape == (0, 2)


def test_the_same_receptor_in_two_regions_has_two_different_occupancies():
    # THE point of P6.1: before it, one `ReceptorSystem` coupling to two
    # nodes had ONE occupancy for both. Here the sparser region's field is
    # lower, so its occupancy is lower, and the two nodes get different drive.
    field, receptors = field_and_receptors()
    sim = Simulation(
        network=simple_network(),
        receptor_systems=[receptors],
        projection_systems=[field],
    )
    result = sim.run(
        doses=np.zeros(sim.n_ligands),
        t_span=(0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 41),
        rates0=np.full(3, REST),
        method="LSODA",
        occupancy0="resting",
    )
    dense = result["occupancy"][receptors.state_rows("CP")][0, -1]
    sparse = result["occupancy"][receptors.state_rows("PFC_Pyramidal")][0, -1]
    assert dense > sparse > 0.0
    # And the difference is the field's, not an artefact: equal densities
    # give equal occupancies.
    equal_field, equal_receptors = field_and_receptors(densities=(1.0, 1.0))
    equal_sim = Simulation(
        network=simple_network(),
        receptor_systems=[equal_receptors],
        projection_systems=[equal_field],
    )
    equal = equal_sim.run(
        doses=np.zeros(equal_sim.n_ligands),
        t_span=(0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 41),
        rates0=np.full(3, REST),
        method="LSODA",
        occupancy0="resting",
    )
    a = equal["occupancy"][equal_receptors.state_rows("CP")][0, -1]
    b = equal["occupancy"][equal_receptors.state_rows("PFC_Pyramidal")][0, -1]
    assert a == pytest.approx(b)


def test_the_field_follows_the_source_nucleus_rate_inside_the_ode():
    # The one place information travels from the circuit layer back into the
    # concentration layer. Held at a lower rate by a negative baseline, the
    # source node fires less and every target region's concentration falls.
    field, receptors = field_and_receptors()
    quiet = simple_network()
    quiet.nodes[0].baseline_input = -2.0
    sim = Simulation(
        network=quiet, receptor_systems=[receptors], projection_systems=[field]
    )
    rates0 = np.full(3, REST)
    result = sim.run(
        doses=np.zeros(sim.n_ligands),
        t_span=(0.0, 30.0),
        t_eval=np.linspace(0.0, 30.0, 61),
        rates0=rates0,
        method="LSODA",
        occupancy0="resting",
    )
    source_rate = result["rates"][quiet.index["SNc"]][-1]
    assert source_rate < 0.1
    for region in field.regions:
        trace = sim.projection_trace(result, "dopamine", region)
        assert trace[-1] < 0.3 * field.basal_concentration(region)
        assert trace[-1] == pytest.approx(
            field.steady_state(region, [source_rate]), rel=1e-3
        )


def test_a_regions_receptor_density_scales_only_that_regions_drive():
    # P4.2's missing factor `rho` in `coupling = mechanism_gain * rho`, which
    # a projection system is the first structure here that could carry. Two
    # regions with the SAME transmitter concentration and different receptor
    # densities must put different drive on their nodes and identical
    # occupancy in their blocks: density scales the coupling, not the binding.
    field, _ = field_and_receptors(densities=(1.0, 1.0))
    site = two_state_site()
    receptors = RegionalReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        regions=["CP", "PFC_Pyramidal"],
        coupling={"CP": 1.0, "PFC_Pyramidal": 1.0},
        density={"PFC_Pyramidal": 0.25},
        transmitter_sources={"dopamine": "dopamine"},
    )
    sim = Simulation(
        network=simple_network(),
        receptor_systems=[receptors],
        projection_systems=[field],
    )
    result = sim.run(
        doses=np.zeros(sim.n_ligands),
        t_span=(0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 41),
        rates0=np.full(3, REST),
        method="LSODA",
        occupancy0="resting",
    )
    dense = result["occupancy"][receptors.state_rows("CP")][0, -1]
    sparse = result["occupancy"][receptors.state_rows("PFC_Pyramidal")][0, -1]
    assert dense == pytest.approx(sparse)  # same field, same occupancy
    rates = result["rates"]
    assert rates[1, -1] > rates[2, -1]  # four times the drive on CP's node


def test_a_regional_system_can_modulate_a_connection_per_region():
    # `connection_modulation` is per region for a regional system, and it
    # composes multiplicatively across systems exactly as
    # `ReceptorSystem`'s does. Exercised because it is live code in the RHS
    # and no ported scenario uses it.
    network = simple_network()
    network.connectivity[1, 2] = 0.5  # PFC_Pyramidal -> CP
    field, _ = field_and_receptors()
    site = two_state_site()
    receptors = RegionalReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        regions=["CP", "PFC_Pyramidal"],
        coupling={},
        connection_modulation={"CP": {("CP", "PFC_Pyramidal"): -1.0}},
        transmitter_sources={"dopamine": "dopamine"},
    )
    sim = Simulation(
        network=network, receptor_systems=[receptors], projection_systems=[field]
    )
    kwargs = dict(
        doses=np.zeros(sim.n_ligands),
        t_span=(0.0, 20.0),
        t_eval=np.linspace(0.0, 20.0, 41),
        rates0=np.full(3, REST),
        method="LSODA",
        occupancy0="resting",
    )
    modulated = sim.run(**kwargs)

    unmodulated = Simulation(
        network=network,
        receptor_systems=[
            RegionalReceptorSystem(
                pk_models=[_pk() for _ in site.ligands],
                receptor_site=site,
                regions=["CP", "PFC_Pyramidal"],
                coupling={},
                transmitter_sources={"dopamine": "dopamine"},
            )
        ],
        projection_systems=[field],
    ).run(**kwargs)
    # The tonic occupancy scales the +0.5 weight down, so CP receives less.
    assert modulated["rates"][1, -1] < unmodulated["rates"][1, -1]


def test_projection_and_transporter_system_names_must_be_disjoint():
    field, receptors = field_and_receptors()
    field.name = "DAT"
    with pytest.raises(ValueError, match="disjoint"):
        Simulation(
            network=simple_network(),
            transporter_systems=[mt.transporter_system("DAT")],
            projection_systems=[field],
        )


def test_a_projection_source_node_must_be_in_the_network():
    field, _ = field_and_receptors()
    with pytest.raises(ValueError, match="not in this network"):
        Simulation(
            network=simple_network(names=("CP", "PFC_Pyramidal")),
            projection_systems=[field],
        )


def test_projection_ligands_sit_after_every_other_ligand_in_the_dose_vector():
    # Adding a projection system must not renumber an existing system's doses.
    site = two_state_site()
    field, _ = field_and_receptors()
    plain = ReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        coupling={"CP": 1.0},
        name="plain",
    )
    without = Simulation(network=simple_network(), receptor_systems=[plain])
    with_field = Simulation(
        network=simple_network(),
        receptor_systems=[plain],
        projection_systems=[field],
    )
    a = without.dose_vector(blocker=5.0)
    b = with_field.dose_vector(blocker=5.0)
    assert np.array_equal(a, b[: a.size])
    assert with_field.n_ligands == without.n_ligands + field.n
    assert with_field.dose_vector(cocaine=1.0)[a.size :].sum() == 1.0


# ---------------------------------------------------------------------------
# Tonic drive and the compensation it forces
# ---------------------------------------------------------------------------


def test_standing_drive_reproduces_p5_8_s_hand_computed_striatal_value():
    # P5.8 computed this by hand for one node: 0.711 tonic D2 occupancy times
    # a coupling gain of -3.0 is a standing -2.133 on Striatum_D2_MSN. The
    # general computation must land on the same number, or one of the two is
    # wrong.
    from neurocascade.legacy import projection_scenarios as ps

    ported = ps.port("dopamine")
    expected = (
        scenarios.d2_coupling()["Striatum_D2_MSN"] * mt.resting_d2_occupancy()
    )
    assert ported.standing_drive["Striatum_D2_MSN"] == pytest.approx(expected, rel=1e-6)


def test_compensate_baselines_moves_baselines_and_nothing_else():
    network = simple_network()
    compensated = compensate_baselines(network, {"CP": -2.0})
    assert np.array_equal(compensated.connectivity, network.connectivity)
    for before, after in zip(network.nodes, compensated.nodes):
        assert (after.tau, after.gain, after.threshold) == (
            before.tau,
            before.gain,
            before.threshold,
        )
    assert compensated.nodes[network.index["CP"]].baseline_input == pytest.approx(
        network.nodes[network.index["CP"]].baseline_input + 2.0
    )
    with pytest.raises(ValueError, match="does not have"):
        compensate_baselines(network, {"nowhere": 1.0})


def test_resting_occupancy_matches_the_closed_form_for_a_competitive_site():
    site = two_state_site()
    concentrations = np.array([5.0, 0.0])
    occupancy = projection.resting_occupancy(site, concentrations)
    dopamine = site.ligands[0]
    kd = dopamine.koff / dopamine.kon
    assert occupancy[0] == pytest.approx(5.0 / (5.0 + kd), rel=1e-6)
    assert occupancy[1] == pytest.approx(0.0, abs=1e-12)


def test_standing_drive_ignores_a_transporter_sourced_ligand():
    # `standing_drive` is about projection fields; a transporter pool's tonic
    # drive is P5.8's `tonic_compensated_network` and is not this function's
    # business. Asserted so the two cannot silently double-count.
    site = two_state_site()
    system = RegionalReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        regions=["CP"],
        coupling={"CP": -3.0},
        transmitter_sources={"dopamine": "DAT"},
    )
    assert standing_drive([system], []) == {}


def test_standing_drive_refuses_a_single_block_receptor_system():
    site = two_state_site()
    plain = ReceptorSystem(
        pk_models=[_pk() for _ in site.ligands],
        receptor_site=site,
        coupling={"CP": -3.0},
        name="plain",
    )
    with pytest.raises(TypeError, match="RegionalReceptorSystem"):
        standing_drive([plain], [])
