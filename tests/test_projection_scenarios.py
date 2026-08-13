"""P6.1: the hand-built scenarios, ported onto projection systems.

This is the regression suite the acceptance criterion "existing scenarios
ported and regression-tested" names. It asserts three kinds of thing:

* the port did not disturb what it ported FROM (the hand-built modules are
  untouched, and their own test files still pass unmodified);
* the ported scenario is well-posed (flat without drug, integrator-agnostic,
  stable);
* the two results the port exists to produce -- a CLOSED autoreceptor loop
  and a NON-INERT lone antagonist -- are there, with the honest statement of
  where they are invisible and why.
"""

import numpy as np
import pytest

from neurocascade import acetylcholine as ach
from neurocascade import histamine as ha
from neurocascade import monoamine_transporters as mt
from neurocascade import noradrenaline as na
from neurocascade import projection_scenarios as ps
from neurocascade import scenarios
from neurocascade import serotonin as ht
from neurocascade import stability


PORTS = ps.PORT_ORDER


@pytest.fixture(scope="module")
def ported():
    return {key: ps.port(key) for key in PORTS}


# ---------------------------------------------------------------------------
# The port did not disturb what it ported from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PORTS)
def test_the_hand_built_network_is_unchanged_apart_from_added_nucleus_nodes(
    ported, key
):
    scenario = ported[key]
    base = scenario.port.base_network()
    extended = scenario.base_network
    assert extended.n >= base.n
    for original, carried in zip(base.nodes, extended.nodes[: base.n]):
        assert carried.name == original.name
        assert carried.tau == original.tau
        assert carried.gain == original.gain
        assert carried.threshold == original.threshold
        assert carried.baseline_input == original.baseline_input
    assert np.array_equal(extended.connectivity[: base.n, : base.n], base.connectivity)
    # Added nodes are isolated in both directions.
    assert not extended.connectivity[base.n :, :].any()
    assert not extended.connectivity[:, base.n :].any()


@pytest.mark.parametrize("key", PORTS)
def test_the_tonic_compensation_moves_baselines_and_nothing_else(ported, key):
    scenario = ported[key]
    base, compensated = scenario.base_network, scenario.network
    assert np.array_equal(base.connectivity, compensated.connectivity)
    for before, after in zip(base.nodes, compensated.nodes):
        assert (after.name, after.tau, after.gain, after.threshold) == (
            before.name,
            before.tau,
            before.gain,
            before.threshold,
        )
        expected = before.baseline_input - scenario.standing_drive.get(before.name, 0.0)
        assert after.baseline_input == pytest.approx(expected)


def test_every_coupling_gain_is_the_hand_built_module_s_own():
    # A port may move a receptor. It may not redefine one.
    gains = {
        ("serotonin", "5HT1A"): ht.HT1A_MECHANISM_GAIN,
        ("serotonin", "5HT2A"): ht.HT2A_MECHANISM_GAIN,
        ("noradrenaline", "alpha1"): na.ALPHA1_MECHANISM_GAIN,
        ("noradrenaline", "alpha2"): na.ALPHA2_MECHANISM_GAIN,
        ("noradrenaline", "beta"): na.BETA_MECHANISM_GAIN,
        ("acetylcholine", "M1"): ach.CORTICAL_MECHANISM_GAIN,
        ("acetylcholine", "M2"): ach.M2_MECHANISM_GAIN,
        ("acetylcholine", "alpha7"): ach.CORTICAL_MECHANISM_GAIN,
        ("acetylcholine", "alpha4beta2"): ach.CORTICAL_MECHANISM_GAIN,
        ("histamine", "H1"): ha.H1_MECHANISM_GAIN,
        ("dopamine", "D2"): scenarios.d2_coupling()["Striatum_D2_MSN"],
    }
    for (key, name), expected in gains.items():
        scenario = ps.port(key)
        system = next(s for s in scenario.receptor_systems if s.name == name)
        for region, nodes in system.coupling.items():
            for gain in nodes.values():
                assert gain == pytest.approx(expected), (key, name, region)


def test_every_receptor_site_is_the_hand_built_module_s_own():
    scenario = ps.port("serotonin")
    ported_site = next(
        s for s in scenario.receptor_systems if s.name == "5HT1A"
    ).receptor_site
    reference = ht.receptor_site_5ht1a()
    assert [lig.name for lig in ported_site.ligands] == [
        lig.name for lig in reference.ligands
    ]
    for a, b in zip(ported_site.ligands, reference.ligands):
        assert (a.kon, a.koff, a.efficacy) == (b.kon, b.koff, b.efficacy)


# ---------------------------------------------------------------------------
# The ported scenario is well-posed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PORTS)
def test_a_drug_free_run_is_flat_at_the_resting_rate(ported, key):
    scenario = ported[key]
    network, _, result = scenario.run()
    assert np.abs(result["rates"] - ps.REST_RATE).max() < 1e-6


@pytest.mark.parametrize("key", PORTS)
def test_a_drug_free_field_is_flat_at_its_predicted_basal_level(ported, key):
    scenario = ported[key]
    _, sim, result = scenario.run()
    for region in scenario.projection.regions:
        trace = scenario.field_trace(sim, result, region)
        expected = scenario.projection.basal_concentration(region)
        # The tolerance is `solve_ivp`'s own default rtol of 1e-3, tightened
        # by an order of magnitude; the measured worst case is 3.8e-6, in the
        # serotonin field, where the pool's 2.1 s time constant against a
        # 30 min run gives the step-size controller the widest range to place
        # steps in. A bound taken from the integrator rather than fitted to
        # the residual, which is P5.3's rule.
        assert np.allclose(trace, expected, rtol=1e-4)


def test_starting_from_empty_receptors_reproduces_the_transient_the_default_avoids():
    # The tonic compensation subtracts a standing drive that an
    # empty-receptor start has not yet produced. `occupancy0="resting"` is
    # the default in `PortedScenario.run` for that reason, and the size of
    # what it avoids is asserted rather than assumed.
    scenario = ps.port("dopamine")
    _, _, resting = scenario.run()
    _, _, empty = scenario.run(occupancy0=None)
    assert np.abs(resting["rates"] - ps.REST_RATE).max() < 1e-6
    assert np.abs(empty["rates"] - ps.REST_RATE).max() > 0.02
    # Both settle to the same place; only the start differs.
    assert np.allclose(resting["rates"][:, -1], empty["rates"][:, -1], atol=1e-5)


def test_the_two_integrators_agree_on_a_ported_scenario():
    # P5.8's argument, re-checked: the measured uptake kinetics make this a
    # stiff system, so LSODA is the default. That is a statement about the
    # integrator and it has to be shown, not asserted.
    scenario = ps.port("serotonin")
    _, _, lsoda = scenario.run(**{"8-OH-DPAT": 100.0})
    _, _, rk45 = scenario.run(method="RK45", **{"8-OH-DPAT": 100.0})
    assert np.abs(lsoda["rates"] - rk45["rates"]).max() < 1e-3


@pytest.mark.parametrize("key", PORTS)
def test_every_ported_network_keeps_p3_4_s_global_uniqueness_certificate(ported, key):
    # Adding isolated nucleus nodes and shifting baselines cannot change the
    # contraction number, which depends on weights and gains only. Asserted
    # anyway, because "cannot" is what P3.4 exists to check rather than
    # assume.
    scenario = ported[key]
    assert stability.contraction_number(scenario.network) < 1.0


@pytest.mark.parametrize("key", PORTS)
def test_no_node_of_a_ported_scenario_sits_against_a_sigmoid_rail(ported, key):
    # P3.4's saturation lesson, for the fourth time (P3.4, P5.7, P5.8, here).
    # Tonic transmitter is exactly the kind of standing term that pins a node
    # if the compensation is wrong.
    scenario = ported[key]
    _, _, result = scenario.run()
    assert result["rates"].min() > stability.SATURATION_FLOOR
    assert result["rates"].max() < 1.0 - stability.SATURATION_FLOOR


# ---------------------------------------------------------------------------
# The density maps are labelled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PORTS)
def test_every_density_this_module_supplies_is_labelled_illustrative(ported, key):
    scenario = ported[key]
    report = scenario.projection.density_report()
    assert report
    for region, density, relative, provenance in report:
        assert provenance == ps.DENSITY_PROVENANCE
        assert "illustrative" in provenance
        # Uniform: every region is the reference region's equal.
        assert relative == pytest.approx(1.0)


@pytest.mark.parametrize("key", PORTS)
def test_a_uniform_map_puts_every_region_at_the_reference_basal_level(ported, key):
    scenario = ported[key]
    levels = {
        region: scenario.projection.basal_concentration(region)
        for region in scenario.projection.regions
    }
    assert len(set(round(v, 12) for v in levels.values())) == 1


# ---------------------------------------------------------------------------
# Result one: the autoreceptor loop is closed
# ---------------------------------------------------------------------------


def test_the_serotonin_autoreceptor_loop_is_closed_end_to_end():
    # The result P6.1 exists to produce, in the one system where every link
    # in the chain carries a measured concentration scale. 8-OH-DPAT at the
    # somatodendritic 5-HT1A autoreceptor:
    #   raphe firing falls -> release falls -> CORTICAL serotonin falls ->
    #   cortical 5-HT1A occupancy by serotonin falls -> cortical firing falls.
    # `test_serotonin_scenario.py` asserts that links 2 to 5 are IMPOSSIBLE
    # in the hand-built scenario, and that test still passes unmodified,
    # because a port is a new object rather than an edit.
    effect = ps.closed_loop_effect("serotonin")
    assert effect["drug"] == "8-OH-DPAT"
    assert effect["source_rate"] < -0.1
    assert effect["terminal_field_fold"] < -0.5
    assert effect["terminal_occupancy"] < -0.1
    assert effect["terminal_rate"] < -0.1


def test_a_uniform_density_map_gives_every_region_the_same_occupancy():
    # The structural half of the result, and the honest reading of it. 5-HT1A
    # is ONE receptor type in THREE regions with three independent occupancy
    # blocks, which a single `ReceptorSystem` cannot express at all. Under the
    # uniform density map those three blocks nevertheless agree to machine
    # precision, and that is the map's content rather than a bug: fed by the
    # same nuclei at the same density and reached by the same plasma drug,
    # three regions have no reason to differ. What the port demonstrates here
    # is the structure; what makes the structure produce different numbers is
    # a real density map, which is P6.3's.
    scenario = ps.port("serotonin")
    _, sim, result = scenario.run(**{"8-OH-DPAT": 100.0})
    occupancies = [
        scenario.occupancy_trace(sim, result, "5HT1A", region, "serotonin")[-1]
        for region in ("DR", "CS", "PFC")
    ]
    assert occupancies[1] == pytest.approx(occupancies[0], rel=1e-9)
    assert occupancies[2] == pytest.approx(occupancies[0], rel=1e-9)


def test_a_non_uniform_density_map_separates_the_raphe_from_the_cortex():
    # The counterfactual that makes the test above a statement about the map
    # and not about the abstraction. Give the cortex a quarter of the raphe's
    # innervation and the same three blocks stop agreeing: the cortical
    # serotonin concentration falls, and with it the cortical occupancy and
    # the cortical node's drive.
    sparse = {"PFC": {"DRN_5HT": 0.25, "MRN_5HT": 0.25}}
    scenario = ps.port("serotonin", densities=sparse)
    assert scenario.projection.relative_density("PFC") == pytest.approx(0.25)
    assert scenario.projection.basal_concentration(
        "PFC"
    ) < scenario.projection.basal_concentration("DR")

    _, sim, result = scenario.run()
    occupancies = {
        region: scenario.occupancy_trace(sim, result, "5HT1A", region, "serotonin")[-1]
        for region in ("DR", "CS", "PFC")
    }
    assert occupancies["DR"] == pytest.approx(occupancies["CS"], rel=1e-9)
    assert occupancies["PFC"] < 0.9 * occupancies["DR"]
    # And the override is labelled, so a density map can never quietly stop
    # being illustrative.
    provenances = {
        region: provenance
        for region, _, _, provenance in scenario.projection.density_report()
    }
    assert provenances["PFC"] == "override"
    assert provenances["DR"] == ps.DENSITY_PROVENANCE


def test_the_noradrenaline_loop_closes_in_the_field_and_barely_at_the_receptor():
    # P6.1 wrote this test to report a GAP: clonidine lowers LC firing and
    # halves the cortical noradrenaline FIELD, which is the loop P5.3's
    # `test_the_autoreceptor_is_an_open_loop_until_phase_6` says the
    # hand-built scenario cannot express, and the cortical receptors did not
    # move at all, because NET carried a UNIT SCALE of 1.0 nM in the absence
    # of a measurement.
    #
    # P6.2 supplies the measurement ([PanLai1995], 1.59 nM by zero net flux
    # in conscious rat mPFC), so the last assertion flips: the field is now
    # a measured scale. What did NOT change is the smallness, and that is now
    # a statement about a number rather than about its absence. Cortical
    # alpha1 and beta really do sit four and three orders of magnitude below
    # their Kd at 1.59 nM, so halving the field really does move them by
    # ~5e-5. The bounds are TIGHTENED rather than loosened: the effect must
    # now be nonzero as well as small, which is the half P6.1 could not
    # assert.
    effect = ps.closed_loop_effect("noradrenaline")
    assert effect["source_rate"] < -0.1
    assert effect["terminal_field_fold"] < -0.3
    assert 1e-6 < abs(effect["terminal_occupancy"]) < 1e-3
    assert 1e-6 < abs(effect["terminal_rate"]) < 1e-3
    assert "noradrenaline" in ps.MEASURED_BASAL


def test_the_acetylcholine_autoreceptor_probe_moves_the_nucleus_the_right_way():
    # Scopolamine is an ANTAGONIST at the M2 autoreceptor, so blocking it
    # disinhibits the nucleus and RAISES release: the opposite sign from the
    # other two probes, and a prediction of the wiring rather than a
    # convention.
    effect = ps.closed_loop_effect("acetylcholine")
    assert effect["direction"] == +1
    assert effect["source_rate"] > 0.0
    assert effect["terminal_field_fold"] > 0.0
    # P6.1 asserted the opposite of the line below, because acetylcholine's
    # basal concentration was a unit scale. [Vinson1997] supplies it in P6.2
    # (18.4 nM, rat striatum, no-net-flux), so the field carries a measured
    # absolute scale and the cortical occupancy the probe moves is 248-fold
    # larger than it was. The sign prediction this test exists for is
    # untouched by that.
    assert "acetylcholine" in ps.MEASURED_BASAL
    assert effect["terminal_occupancy"] > 1e-6


@pytest.mark.parametrize("key", sorted(ps.AUTORECEPTOR_PROBE))
def test_every_autoreceptor_probe_moves_its_nucleus_in_the_predicted_direction(key):
    effect = ps.closed_loop_effect(key)
    assert np.sign(effect["source_rate"]) == effect["direction"]
    # And the field follows the nucleus, which is the link that did not exist
    # before P6.1.
    assert np.sign(effect["terminal_field_fold"]) == effect["direction"]


# ---------------------------------------------------------------------------
# Result two: an antagonist given alone is no longer inert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ("dopamine", "serotonin", "histamine"))
def test_a_lone_antagonist_is_not_inert_where_the_basal_level_is_measured(key):
    # Every module from P5.1 to P5.7 had to write "an antagonist given alone
    # is inert here, because P6.2 has supplied no endogenous transmitter to
    # displace". P5.8 falsified that for one transmitter at one site; this
    # falsifies it for three transmitters across a projection field.
    effect = ps.lone_antagonist_effect(key)
    assert abs(effect["terminal_rate"]) > 1e-3
    assert key in ps.MEASURED_BASAL


def test_the_dopamine_port_reproduces_p5_8_s_haloperidol_result():
    # The regression that matters most: P5.8 reported that 100 nM haloperidol
    # on the transporter-fed D2 system moves the locomotor readout by -0.200.
    # Replacing the single transporter pool with a two-source projection
    # field at its resting rate must not move that number, because at rest
    # the field reproduces the transporter's mass balance term for term.
    ported = ps.lone_antagonist_effect("dopamine")["terminal_rate"]

    transporter_network = mt.tonic_compensated_network()
    systems = [mt.d2_receptor_system_on_transporter_dopamine()]
    _, _, base = mt.run(
        receptor_systems=systems, network=transporter_network,
        rates0=np.full(transporter_network.n, 0.5),
    )
    _, _, dosed = mt.run(
        receptor_systems=systems, network=transporter_network,
        rates0=np.full(transporter_network.n, 0.5), haloperidol=100.0,
    )
    row = transporter_network.index["Thalamus"]
    transporter = dosed["rates"][row][-1] - base["rates"][row][-1]

    assert ported == pytest.approx(transporter, abs=2e-3)
    assert ported == pytest.approx(-0.200, abs=5e-3)


def test_the_ported_dopamine_field_still_answers_to_uptake_blockade():
    # The uptake blockers travel with the clearance object, so P5.8's
    # psychostimulant chain survives the port: cocaine raises the field, the
    # field raises D2 occupancy, occupancy raises the readout.
    scenario = ps.port("dopamine")
    net0, sim0, base = scenario.run()
    net1, sim1, dosed = scenario.run(cocaine=1000.0)
    fold = scenario.field_fold(sim1, dosed, "CP")[-1]
    # P5.8's exact law, `1 + [drug]/Ki`, evaluated at the concentration
    # cocaine has actually decayed to by the end of the run rather than at
    # the dose: `TOOL_COMPOUND_KE` halves it over 30 minutes, and comparing
    # against the dose would be comparing against the wrong number.
    ligands = [lig.name for lig in scenario.projection.ligands]
    row = sim1.n_ligands - scenario.projection.n + ligands.index("cocaine")
    present = dosed["concentrations"][row][-1]
    assert present == pytest.approx(500.0, rel=0.01)
    assert fold == pytest.approx(1.0 + present / mt.ki("cocaine", "DAT"), rel=0.02)
    occupancy = scenario.occupancy_trace(sim1, dosed, "D2", "CP", "dopamine")[-1]
    resting = scenario.occupancy_trace(sim0, base, "D2", "CP", "dopamine")[-1]
    assert occupancy > resting
    row = net1.index["Thalamus"]
    assert dosed["rates"][row][-1] > base["rates"][row][-1]


# ---------------------------------------------------------------------------
# The reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PORTS)
def test_each_port_reports_its_nuclei_regions_and_standing_drive(ported, key):
    text = ported[key].report()
    for node in ported[key].port.nucleus_nodes.values():
        assert node in text
    for region in ported[key].projection.regions:
        assert region in text
    assert ps.DENSITY_PROVENANCE in text


def test_the_summary_names_both_results_and_what_P6_2_changed():
    # P6.1 asserted that the string "UNIT" appeared here, because two of the
    # five rows had to carry that caveat. P6.2 measured both, so no row does,
    # and what the summary has to say instead is which rows moved and where
    # to find by how much. Same requirement, moved forward one task: the
    # summary must not present five measured scales as though they had always
    # been five.
    text = ps.summary()
    assert "autoreceptor loop" in text
    assert "no longer inert" in text
    assert "UNIT" not in text
    assert "unit scale" in text and "endogenous_tone" in text
    for key in PORTS:
        assert key in text
