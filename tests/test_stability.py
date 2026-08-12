"""P3.4: the saturation check, the baseline procedure, and the certificate.

Everything here runs offline off the committed fifteen-region fixtures
except the block at the bottom that needs the gitignored `data/connectome`
cache and skips without it. The offline tests cover every acceptance
criterion; the cache-gated ones repeat them on all 291 published regions,
which is the network the claim "whole-brain" is actually about.

Six groups:

* **The measure.** That `sensitivity` really is the Jacobian's own
  coefficient and not a rate band someone picked, and that
  `SATURATION_FLOOR`'s two readings agree.
* **The procedure.** That `balanced_baselines` makes the target rates an
  exact fixed point, for uniform and per-node targets, and that it
  reproduces the hand-built basal-ganglia network's own numbers.
* **The certificate.** That `kappa < 1` behaves like a contraction should,
  and that it is sufficient rather than necessary.
* **The failures it has to catch.** The original GPi draft, reconstructed;
  P3.3's ported GPe; and the distinction between a node that is silent at
  rest because the drug recruits it and a node that is simply dead.
* **The whole-brain network**, at all three resolutions.
* **A correction to P3.3**, which analysed the ported D2 network at an
  exact fixed point that turns out to be unstable.
"""

import numpy as np
import pytest

from neurocascade import connectome, connectome_scenarios as cs, scenarios, stability
from neurocascade.circuit import CircuitNode, sigmoid
from neurocascade.network import CircuitNetwork


HAND_BUILT_NETWORKS = {
    "reward": scenarios.reward_circuit_network,
    "multi_region": scenarios.multi_region_mu_opioid_network,
    "basal_ganglia": scenarios.basal_ganglia_indirect_pathway_network,
    "amygdala": scenarios.amygdala_anxiety_circuit_network,
    "combined": scenarios.combined_mu_opioid_d2_network,
}


@pytest.fixture(scope="module")
def circuit():
    return cs.scenario_connectome(source="fixture")


@pytest.fixture(scope="module")
def fixture_build():
    ontology = connectome.load_fixture_ontology()
    return connectome.load_fixture_build_inputs(ontology=ontology)


def full_activation_drive(port, network):
    """Baseline drive plus every receptor system's coupling at activation 1.

    The limit the scenario's own agonist approaches, computed without
    integrating the pharmacokinetics: `Simulation` adds
    `gain * activation` to each coupled node's drive, so activation = 1 is
    this vector. Used to ask whether a node that is saturated at rest is
    one the drug recruits, or one that is dead.
    """
    drive = network.baseline_drive().copy()
    for system in port.simulation(network).receptor_systems:
        for name, gain in system.coupling.items():
            if name in network.index:
                drive[network.index[name]] += gain
    return drive


# ---------------------------------------------------------------------------
# The measure of saturation
# ---------------------------------------------------------------------------


def test_sensitivity_is_one_at_the_midpoint_and_zero_at_the_rails():
    assert stability.sensitivity(0.5) == 1.0
    assert stability.sensitivity(0.0) == 0.0
    assert stability.sensitivity(1.0) == 0.0
    assert stability.sensitivity(0.25) == pytest.approx(stability.sensitivity(0.75))
    assert stability.MAX_SENSITIVITY_RATE == 0.5


def test_the_sensitivity_is_the_sigmoids_own_slope_over_its_maximum():
    """Not a chosen band: the derivative of the model's own transfer function.

    Differentiate `sigmoid` numerically at the net input that produces each
    rate and compare against `gain * r * (1 - r)`, then against
    `sensitivity * gain / 4`.
    """
    gain = 4.0
    node = CircuitNode(name="n", tau=1.0, gain=gain)
    network = CircuitNetwork([node], [[0.0]])
    for net_input in (-2.0, -0.5, 0.0, 0.3, 1.7):
        rate = sigmoid(net_input, gain=gain)
        step = 1e-6
        numerical = (
            sigmoid(net_input + step, gain=gain) - sigmoid(net_input - step, gain=gain)
        ) / (2 * step)
        analytic = stability.sigmoid_slopes(network, [rate])[0]
        assert analytic == pytest.approx(numerical, rel=1e-6)
        assert analytic == pytest.approx(stability.sensitivity(rate) * gain / 4.0)


def test_the_saturation_floor_has_the_two_readings_its_docstring_claims():
    """0.01 is a reporting convention, and both readings of it must hold.

    (a) a node at the floor is within 0.0025 of a rail; (b) at the
    illustrative gain=4 the flag fires at |net - threshold| = 1.50, just
    outside the [-1, 1] band `docs/architecture.md`'s original GPi note
    used. If either arithmetic changes, the docstring is wrong.
    """
    floor = stability.SATURATION_FLOOR
    rate = 0.5 * (1.0 - np.sqrt(1.0 - floor))
    assert stability.sensitivity(rate) == pytest.approx(floor)
    assert rate == pytest.approx(0.0025063, abs=1e-6)
    net_offset = -np.log(rate / (1.0 - rate)) / 4.0
    assert net_offset == pytest.approx(1.4966, abs=1e-3)


def test_the_operating_matrix_defaults_to_the_worst_case_for_stability():
    """`rates=None` means r=1/2, where every node's slope is largest.

    That is what makes `contraction_number` an upper bound over the whole
    state space rather than a statement about one operating point.
    """
    network = scenarios.basal_ganglia_indirect_pathway_network()
    gains = np.array([node.gain for node in network.nodes])
    expected = (gains[:, None] / 4.0) * network.connectivity
    assert np.allclose(stability.operating_matrix(network), expected)
    for rates in (np.full(network.n, 0.2), np.full(network.n, 0.9)):
        assert stability.contraction_number(
            network, rates
        ) <= stability.contraction_number(network)


def test_the_jacobian_matches_a_numerical_derivative_of_the_rhs():
    network = scenarios.multi_region_mu_opioid_network()
    drive = network.baseline_drive()
    rates = np.linspace(0.2, 0.8, network.n)
    analytic = stability.jacobian(network, rates)
    numerical = np.zeros((network.n, network.n))
    step = 1e-6
    for j in range(network.n):
        up, down = rates.copy(), rates.copy()
        up[j] += step
        down[j] -= step
        numerical[:, j] = (network.rhs(up, drive) - network.rhs(down, drive)) / (
            2 * step
        )
    assert np.allclose(analytic, numerical, atol=1e-6)


# ---------------------------------------------------------------------------
# The procedure: baseline drive at scale
# ---------------------------------------------------------------------------


def test_balanced_baselines_reproduces_the_hand_built_basal_ganglia_numbers():
    """The credential that makes this a restatement, not a new decision.

    `scenarios.py` says its baselines "are chosen so each node sits near the
    sensitive (non-saturating) part of its sigmoid at rest". Solving that
    sentence returns the five numbers that are written there.
    """
    hand = scenarios.basal_ganglia_indirect_pathway_network()
    balanced = stability.balanced_baselines(hand)
    assert [node.baseline_input for node in balanced.nodes] == [0.0, 1.0, 1.0, -1.0, 1.0]
    assert [node.baseline_input for node in hand.nodes] == [0.0, 1.0, 1.0, -1.0, 1.0]


@pytest.mark.parametrize("key", sorted(HAND_BUILT_NETWORKS))
def test_balanced_baselines_makes_the_target_an_exact_fixed_point(key):
    network = stability.balanced_baselines(HAND_BUILT_NETWORKS[key]())
    rates = np.full(network.n, 0.5)
    assert np.allclose(network.rhs(rates, network.baseline_drive()), 0.0, atol=1e-14)


def test_balanced_baselines_is_exact_for_per_node_target_rates():
    """The general form, `b_i = theta_i + logit(r*_i)/g_i - sum_j W[i,j] r*_j`.

    Heterogeneous targets are supported so that region-wise baseline firing
    rates have somewhere to go when this project sources any; nothing today
    uses anything but the uniform 1/2.
    """
    network = scenarios.multi_region_mu_opioid_network()
    targets = np.linspace(0.15, 0.85, network.n)
    balanced = stability.balanced_baselines(network, target_rates=targets)
    assert np.allclose(
        balanced.rhs(targets, balanced.baseline_drive()), 0.0, atol=1e-14
    )


def test_balanced_baselines_rejects_target_rates_a_logistic_node_cannot_reach():
    network = scenarios.amygdala_anxiety_circuit_network()
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="strictly inside"):
            stability.balanced_baselines(network, target_rates=bad)


def test_balanced_baselines_touches_nothing_but_the_baseline_input():
    network = scenarios.multi_region_mu_opioid_network()
    before = [node.baseline_input for node in network.nodes]
    balanced = stability.balanced_baselines(network)
    assert np.array_equal(balanced.connectivity, network.connectivity)
    assert [node.baseline_input for node in network.nodes] == before
    for old, new in zip(network.nodes, balanced.nodes):
        assert (old.name, old.tau, old.gain, old.threshold) == (
            new.name,
            new.tau,
            new.gain,
            new.threshold,
        )


def test_p3_3s_rest_balanced_baselines_is_this_rule(circuit):
    """P3.3's name still exists and now delegates here; same numbers."""
    for key in sorted(cs.SCENARIO_PORTS):
        network = cs.port_scenario(cs.SCENARIO_PORTS[key], circuit).network
        a = cs.rest_balanced_baselines(network)
        b = stability.balanced_baselines(network)
        assert [n.baseline_input for n in a.nodes] == [n.baseline_input for n in b.nodes]


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------


def test_the_contraction_number_scales_with_the_weights():
    network = scenarios.basal_ganglia_indirect_pathway_network()
    kappa = stability.contraction_number(network)
    assert kappa == pytest.approx(2.0)
    for scale in (0.1, 0.5, 3.0):
        scaled = stability.rescale_weights(network, scale)
        assert stability.contraction_number(scaled) == pytest.approx(scale * kappa)
    assert stability.critical_weight_scale(network) == pytest.approx(1.0 / kappa)


def test_weight_scale_for_operating_gain_hits_the_requested_loop_gain():
    network = scenarios.multi_region_mu_opioid_network()
    for rho in (0.2, 0.9, 1.5):
        scale = stability.weight_scale_for_operating_gain(network, rho)
        scaled = stability.rescale_weights(network, scale)
        assert stability.contraction_number(scaled) == pytest.approx(rho)


def test_an_edgeless_network_needs_no_rescaling():
    node = CircuitNode(name="lone", tau=1.0, gain=4.0)
    network = CircuitNetwork([node], [[0.0]])
    assert stability.contraction_number(network) == 0.0
    assert stability.critical_weight_scale(network) == np.inf
    assert stability.weight_scale_for_operating_gain(network) == 1.0


@pytest.mark.parametrize("key", sorted(HAND_BUILT_NETWORKS))
def test_stabilize_certifies_and_de_saturates_every_hand_built_network(key):
    """The procedure, end to end, on networks it was not designed against."""
    stabilized = stability.stabilize(HAND_BUILT_NETWORKS[key]())
    assert stabilized.certified
    assert stabilized.contraction_after == pytest.approx(
        stability.DEFAULT_OPERATING_GAIN
    )
    assessment = stability.assess(stabilized.network, n_initial_conditions=8)
    assert assessment.ok, assessment.summary()
    assert assessment.globally_certified
    assert assessment.saturation.min_sensitivity == pytest.approx(1.0, abs=1e-6)


def test_a_certified_network_reaches_the_same_point_from_every_start():
    """What `kappa < 1` claims, checked against the integrator.

    Twelve initial conditions including both corners; the certificate says
    they must all land on the same point, so a disagreement anywhere is a
    disagreement between the theorem and the model actually being solved.
    """
    stabilized = stability.stabilize(scenarios.combined_mu_opioid_d2_network())
    assessment = stability.assess(stabilized.network, n_initial_conditions=12, seed=7)
    assert assessment.n_initial_conditions == 12
    assert assessment.unique
    # Endpoint agreement at ~1e-8 is the integrator's own noise at
    # rtol=1e-8; 1e-6 is two orders above it and four below anything the
    # rest of this suite measures.
    assert assessment.spread < 1e-6
    assert np.allclose(assessment.rates, 0.5, atol=1e-6)


def test_the_certificate_is_sufficient_and_not_necessary():
    """kappa >= 1 is not a verdict, and `assess` must not treat it as one.

    The hand-built basal-ganglia network has kappa = 2 and is nonetheless
    stable, unique across starts and unsaturated - which is why the report
    carries the local criterion separately and why `ok` does not consult
    `globally_certified`.
    """
    assessment = stability.assess(scenarios.basal_ganglia_indirect_pathway_network())
    assert not assessment.globally_certified
    assert assessment.contraction == pytest.approx(2.0)
    assert assessment.locally_stable
    assert assessment.unique
    assert assessment.ok


# ---------------------------------------------------------------------------
# The failures the check exists to catch
# ---------------------------------------------------------------------------


def test_the_original_gpi_draft_is_caught():
    """The lesson itself, reconstructed and now automated.

    `docs/architecture.md`: "the first parameter draft made GPi saturate
    near its sigmoid's ceiling at baseline (net input far outside [-1, 1]
    before the gain multiplier), so it stopped responding to anything
    upstream of it". Putting a baseline like that back into the shipped
    network reproduces it, and the check names GPi without being told to
    look at GPi.

    The `3.0` below is a test input, not a proposed parameter: it is any
    value that puts GPi's net input outside the band the note describes,
    and nothing in `scenarios.py` is changed by this test.
    """
    shipped = scenarios.basal_ganglia_indirect_pathway_network()
    draft = CircuitNetwork(
        nodes=[
            CircuitNode(
                name=node.name,
                tau=node.tau,
                baseline_input=3.0 if node.name == "GPi" else node.baseline_input,
                gain=node.gain,
                threshold=node.threshold,
            )
            for node in shipped.nodes
        ],
        connectivity=shipped.connectivity,
    )
    assessment = stability.assess(draft)
    assert not assessment.ok
    assert [state.name for state in assessment.saturation.saturated] == ["GPi"]
    assert assessment.saturation.worst.rail == "ceiling"
    # ...and the shipped network, whose docstring makes the claim, passes.
    assert stability.assess(shipped).ok


def test_the_shipped_hand_built_networks_that_claim_the_property_have_it():
    for key in ("basal_ganglia", "amygdala"):
        assessment = stability.assess(HAND_BUILT_NETWORKS[key]())
        assert assessment.ok, assessment.summary()
    assert stability.assess(
        scenarios.amygdala_anxiety_circuit_network()
    ).saturation.min_sensitivity == pytest.approx(0.42, abs=1e-3)


def test_the_reward_circuit_rests_with_two_nodes_against_a_rail():
    """A recorded fact about existing scenarios, not a new failure.

    The mu-opioid circuits are built as disinhibition relays: VTA_GABA
    holds VTA_DA down at rest and the agonist releases it, and the PAG
    circuit repeats the motif. So the nodes the drug recruits are silent
    at baseline, and the check says so. Whether that is a defect is settled
    by the next test, not by this one.
    """
    reward = stability.assess(scenarios.reward_circuit_network())
    assert [state.name for state in reward.saturation.saturated] == ["VTA_DA"]
    assert not reward.ok

    multi = stability.assess(scenarios.multi_region_mu_opioid_network())
    assert sorted(state.name for state in multi.saturation.saturated) == [
        "PAG_Output",
        "VTA_DA",
    ]
    for state in multi.saturation.saturated:
        assert state.rail == "floor"


def test_a_recruited_node_leaves_the_rail_and_a_dead_one_does_not(circuit):
    """The distinction that makes the check usable rather than pedantic.

    Saturated at rest is not the same as saturated always. Under the
    agonist at full receptor activation the mu-opioid circuit's silent
    nodes come off the floor (and the drug's own targets go to it instead,
    which is what a disinhibition relay looks like), so nothing is
    saturated in BOTH states. P3.3's ported GPe is: it is against the rail
    with and without the drug, which is what "it stopped responding to
    anything upstream" means.
    """
    port = cs.SCENARIO_PORTS["mu_opioid_multi_region"]
    network = port.hand_built()
    rest = stability.assess(network).saturation
    drugged = stability.assess(
        network, drive=full_activation_drive(port, network)
    ).saturation
    at_rest = {state.name for state in rest.saturated}
    with_drug = {state.name for state in drugged.saturated}
    assert at_rest == {"VTA_DA", "PAG_Output"}
    assert with_drug == {"VTA_GABA", "PAG_Interneuron", "LC"}
    assert at_rest & with_drug == set()

    d2 = cs.SCENARIO_PORTS["d2_indirect"]
    ported = cs.port_scenario(d2, circuit, baselines="rest_balanced").network
    ported_rest = {
        state.name for state in stability.assess(ported).saturation.saturated
    }
    ported_drugged = {
        state.name
        for state in stability.assess(
            ported, drive=full_activation_drive(d2, ported)
        ).saturation.saturated
    }
    assert "GPe" in ported_rest & ported_drugged


def test_the_ported_d2_network_saturates_gpe_with_hand_built_baselines(circuit):
    """P3.3's cause 1, now found by the automated check rather than by hand."""
    port = cs.SCENARIO_PORTS["d2_indirect"]
    ported = cs.port_scenario(port, circuit).network
    assessment = stability.assess(ported)
    assert [state.name for state in assessment.saturation.saturated] == ["GPe"]
    assert assessment.saturation.worst.rail == "floor"


# ---------------------------------------------------------------------------
# A correction to P3.3: an exact fixed point that is not a rest state
# ---------------------------------------------------------------------------


def test_the_rebalanced_ported_d2_fixed_point_is_exact_but_unstable(circuit):
    """P3.3 analysed this network sitting on a repeller.

    `rest_balanced_baselines` makes r = 1/2 an exact fixed point of the
    ported D2 network - `rhs` there is identically zero - and P3.3 read the
    scenario off a run started exactly there. The linearization says that
    point is unstable, and integrating from anywhere else confirms it: the
    network is bistable, and neither branch is r = 1/2.

    The cause is anatomical rather than numerical. The published
    striatopallidal density is -12.56 and the pallidostriatal return is
    -0.22; two inhibitory links in series are positive feedback, and at
    gain 4 that loop is strong enough to make CP and GPe a latch.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    network = cs.port_scenario(port, circuit, baselines="rest_balanced").network
    half = np.full(network.n, 0.5)

    assert np.allclose(network.rhs(half, network.baseline_drive()), 0.0, atol=1e-14)
    assert stability.spectral_abscissa(network, half) > 0.0

    low, _ = stability.steady_state(network, initial_rates=np.zeros(network.n))
    high, _ = stability.steady_state(network, initial_rates=np.ones(network.n))
    assert abs(low[network.index["GPe"]] - high[network.index["GPe"]]) > 0.99

    assessment = stability.assess(network)
    assert not assessment.unique
    assert not assessment.ok

    loop = (
        network.connectivity[network.index["GPe"], network.index["Striatum_D2_MSN"]]
        * network.connectivity[network.index["Striatum_D2_MSN"], network.index["GPe"]]
    )
    assert loop > 0.0  # inhibition times inhibition: positive feedback


def test_p3_3s_d2_inversion_survives_on_a_certified_stable_network(circuit):
    """The part of P3.3's diagnosis that the correction above leaves standing.

    Put the ported D2 network through the P3.4 procedure and the bistability
    is gone: r = 1/2 is the unique globally attracting rest state and nothing
    is saturated. The chain then runs with textbook signs at every step -
    striatum down, GPe up, STN down, GPi down - and the thalamus falls
    anyway, never rising above its drug-free value at any time. So the
    pallidothalamic finding is a property of the anatomy and not an artefact
    of reading the scenario off an unstable equilibrium.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    stabilized = stability.stabilize(cs.port_scenario(port, circuit).network)
    network = stabilized.network
    assessment = stability.assess(network)
    assert assessment.ok and assessment.globally_certified

    simulation = port.simulation(network)
    t_eval = np.linspace(0.0, 40.0, 200)
    free = simulation.run(
        dose_concentrations=port.doses(),
        t_span=(0.0, 40.0),
        t_eval=t_eval,
        rates0=assessment.rates,
    )["rates"]
    agonist = simulation.run(
        dose_concentrations=port.doses(agonist=1.0),
        t_span=(0.0, 40.0),
        t_eval=t_eval,
        rates0=assessment.rates,
    )["rates"]

    index = network.index
    deviation = agonist - free
    assert deviation[index["Striatum_D2_MSN"]].min() < -1e-4  # suppressed
    assert deviation[index["GPe"]].max() > 1e-4  # disinhibited
    assert deviation[index["STN"]].min() < -1e-5  # inhibited by GPe
    assert deviation[index["GPi"]].min() < -1e-5  # loses STN drive
    # ...and the readout node falls, never rises.
    assert deviation[index["Thalamus"]].max() <= 1e-12
    assert deviation[index["Thalamus"]].min() < -1e-3


@pytest.mark.parametrize(
    "key", ["gaba_a_amygdala", "mu_opioid_multi_region", "mu_opioid_reward"]
)
def test_the_surviving_ported_scenarios_survive_the_stabilized_operating_point(
    key, circuit
):
    """P3.3's three surviving results, re-run where the check passes.

    Stabilizing rescales every published weight by one scalar, so it cannot
    change any relative weight; what it can do is change an effect size, and
    it does (all three shrink with the loop gain). The sign is what has to
    survive, and does.
    """
    port = cs.SCENARIO_PORTS[key]
    stabilized = stability.stabilize(cs.port_scenario(port, circuit).network)
    network = stabilized.network
    assessment = stability.assess(network)
    assert assessment.ok, assessment.summary()

    simulation = port.simulation(network)
    t_eval = np.linspace(0.0, 40.0, 200)

    def score(readout, doses):
        rates = simulation.run(
            dose_concentrations=doses,
            t_span=(0.0, 40.0),
            t_eval=t_eval,
            rates0=assessment.rates,
        )["rates"]
        return np.array(
            [readout.score(network, rates[:, i]) for i in range(len(t_eval))]
        )

    for name, (factory, direction) in sorted(port.readouts.items()):
        readout = factory()
        baseline = score(readout, port.doses())
        agonist = score(readout, port.doses(agonist=1.0))
        effect = (
            agonist.max() - baseline.max()
            if direction > 0
            else baseline[-1] - agonist.min()
        )
        assert effect > 1e-3, "%s / %s: %g" % (key, name, effect)


# ---------------------------------------------------------------------------
# The whole-brain network, off the committed fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolution", connectome.RESOLUTIONS)
def test_a_network_built_with_all_defaults_saturates(resolution, fixture_build):
    """`build.py`'s docstring warning, turned into a check.

    "A network built from this module with all defaults is therefore not a
    ready-to-simulate model". At every resolution the published densities
    with `weight_scale=1` and `baseline_input=0` put most of the brain
    against a rail.
    """
    strength, voxels = fixture_build
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution=resolution
    )
    assessment = stability.assess(built.network, n_initial_conditions=3)
    assert not assessment.ok
    assert assessment.saturation.saturated
    assert assessment.contraction > 1.0


@pytest.mark.parametrize("resolution", connectome.RESOLUTIONS)
def test_the_procedure_gives_every_node_a_stable_sensitive_rest_state(
    resolution, fixture_build
):
    """The acceptance criterion, at fixture scale: build, stabilize, verify."""
    strength, voxels = fixture_build
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution=resolution
    )
    stabilized = stability.stabilize(built.network)
    assert stabilized.certified
    assessment = stability.assess(stabilized.network, n_initial_conditions=8)
    assert assessment.ok, assessment.summary()
    assert assessment.unique
    assert assessment.locally_stable
    assert np.allclose(assessment.saturation.sensitivities, 1.0, atol=1e-6)


def test_stabilizing_changes_one_global_scalar_and_the_baselines_only(fixture_build):
    """What the procedure is allowed to touch, and what it must not.

    The connectome fixes the relative weights; the procedure may not move
    them. Every ratio of non-zero weights is preserved exactly, and `tau`,
    `gain` and `threshold` come over untouched.
    """
    strength, voxels = fixture_build
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )
    stabilized = stability.stabilize(built.network)
    before = built.network.connectivity
    after = stabilized.network.connectivity
    assert np.allclose(after, stabilized.weight_scale * before, rtol=0, atol=0)
    nonzero = before != 0.0
    assert np.allclose(
        after[nonzero] / before[nonzero], stabilized.weight_scale, rtol=1e-12
    )
    for old, new in zip(built.network.nodes, stabilized.network.nodes):
        assert (old.name, old.tau, old.gain, old.threshold) == (
            new.name,
            new.tau,
            new.gain,
            new.threshold,
        )


def test_the_default_operating_gain_is_labelled_illustrative():
    assert stability.DEFAULT_OPERATING_GAIN == 0.9
    assert "ILLUSTRATIVE" in stability.__doc__
    assert (
        "ILLUSTRATIVE"
        in stability.stabilize(
            scenarios.amygdala_anxiety_circuit_network()
        ).summary()
    )


# ---------------------------------------------------------------------------
# The whole brain: all 291 published regions, cache-gated
# ---------------------------------------------------------------------------


_FULL_DATA = (
    connectome.is_cached("knox2019_connection_strength")
    and connectome.is_cached("knox2019_connection_density")
    and connectome.is_cached(connectome.ONTOLOGY_KEY)
)

full_data_only = pytest.mark.skipif(
    not _FULL_DATA,
    reason="needs the gitignored data/connectome cache; "
    "run scripts/fetch_connectome.py",
)


@pytest.fixture(scope="module")
def knox_inputs():
    if not _FULL_DATA:
        pytest.skip("no cache")
    ontology = connectome.load_ontology()
    return connectome.load_knox_inputs(ontology=ontology)


@full_data_only
def test_the_whole_brain_network_as_built_is_not_simulatable(knox_inputs):
    """The number `build.py` warns about, measured on all 291 regions."""
    strength, voxels = knox_inputs
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )
    assert built.network.n == 291
    assert stability.contraction_number(built.network) == pytest.approx(
        44.704, rel=1e-3
    )
    assert stability.critical_weight_scale(built.network) == pytest.approx(
        0.02237, rel=1e-3
    )
    assessment = stability.assess(built.network, n_initial_conditions=3)
    assert len(assessment.saturation.saturated) == 218
    assert not assessment.ok


@full_data_only
def test_balancing_alone_is_not_enough_at_whole_brain_scale(knox_inputs):
    """Why the procedure needs step 3 and not only step 2.

    Balanced baselines make r = 1/2 an exact fixed point of the unscaled
    291-region network, and it is a repeller: three initial conditions give
    two different saturated answers a full unit apart.
    """
    strength, voxels = knox_inputs
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )
    balanced = stability.balanced_baselines(built.network)
    half = np.full(balanced.n, 0.5)
    assert np.allclose(balanced.rhs(half, balanced.baseline_drive()), 0.0, atol=1e-12)
    assert stability.spectral_abscissa(balanced, half) > 0.0

    assessment = stability.assess(balanced, n_initial_conditions=3)
    assert not assessment.unique
    assert assessment.spread > 0.9
    assert len(assessment.saturation.saturated) == 194


@full_data_only
@pytest.mark.parametrize("resolution", connectome.RESOLUTIONS)
def test_the_whole_brain_network_has_a_stable_non_saturating_rest_state(
    resolution, knox_inputs
):
    """P3.4's acceptance criterion, on the published whole-brain connectome.

    Every node sensitive, one fixed point, reached from every start, with
    the contraction certificate to say there is no other one to find.
    """
    strength, voxels = knox_inputs
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution=resolution
    )
    stabilized = stability.stabilize(built.network)
    assessment = stability.assess(stabilized.network, n_initial_conditions=5)
    assert assessment.ok, assessment.summary()
    assert assessment.globally_certified
    assert assessment.unique
    assert np.allclose(assessment.saturation.sensitivities, 1.0, atol=1e-6)
    assert assessment.saturation.min_sensitivity > 1.0 - 1e-6


@full_data_only
def test_the_stabilized_whole_brain_network_still_propagates(knox_inputs):
    """A network made stable by shrinking everything would pass every check.

    So: the linear response of every region to a unit drive step at the
    caudoputamen. It reaches most of the brain, and its largest responses
    are the caudoputamen's own basal-ganglia targets, all of them negative
    because CP's projection neurons are GABAergic.
    """
    strength, voxels = knox_inputs
    built = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )
    network = stability.stabilize(built.network).network
    A = stability.operating_matrix(network)
    slopes = stability.sigmoid_slopes(network, np.full(network.n, 0.5))
    gain = np.linalg.solve(np.eye(network.n) - A, np.diag(slopes))

    response = gain[:, built.acronyms.index("CP")]
    assert (np.abs(response) > 1e-3).sum() > 100
    ranked = [built.acronyms[k] for k in np.argsort(-np.abs(response))[:6]]
    assert ranked[0] == "CP"
    assert set(ranked[1:]) == {"SNr", "SNc", "GPe", "GPi", "STN"}
    for acronym in ranked[1:]:
        assert response[built.acronyms.index(acronym)] < 0.0

    # The excitatory counterpart: MOp's projection neurons are
    # glutamatergic, its largest responses are thalamic and cortical, and
    # every one of them is positive.
    response = gain[:, built.acronyms.index("MOp")]
    assert (np.abs(response) > 1e-3).sum() > 100
    ranked = [built.acronyms[k] for k in np.argsort(-np.abs(response))[:5]]
    assert ranked[0] == "MOp"
    assert set(ranked[1:]) == {"PCN", "VAL", "PO", "SSp-m"}
    for acronym in ranked[1:]:
        assert response[built.acronyms.index(acronym)] > 0.0


@full_data_only
def test_the_fixture_and_the_full_connectome_agree_on_the_critical_scale():
    """Why the offline tests are about the same machinery, said in numbers.

    The fifteen-region fixture is a slice, so its kappa is its own and not
    the whole brain's - that is expected and is the reason the whole-brain
    numbers are asserted against the real files above. What must agree is
    the arithmetic: kappa/critical scale/operating gain are consistent with
    each other on both.
    """
    for source in ("fixture", "cache"):
        built = cs.scenario_connectome(source=source)
        kappa = stability.contraction_number(built.network)
        assert stability.critical_weight_scale(built.network) == pytest.approx(
            1.0 / kappa
        )
        stabilized = stability.stabilize(built.network, operating_gain=0.75)
        assert stabilized.contraction_after == pytest.approx(0.75)
        assert stabilized.weight_scale == pytest.approx(0.75 / kappa)
