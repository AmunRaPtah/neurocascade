"""P3.3: the hand-built scenarios re-wired from the connectome.

Everything here runs offline off the committed fixtures except the handful
of tests at the bottom that need the gitignored cache and skip without it.
That is not a compromise: the fixtures are a slice of the checksum-verified
published files, and a region-pair connection density is a per-region-pair
quantity, so the fifteen fixture regions carry *exactly* the numbers the
291-region files do for every pair a scenario uses.
`test_fixture_and_full_connectome_give_the_same_ported_networks` asserts that
against the real files rather than leaving it as a claim.

Three groups of tests:

* **Declarations and provenance.** That every scenario node lands on a real
  CCF region, that every weight in a ported network can say where it came
  from, and that the two categories which are still ILLUSTRATIVE
  (within-region edges and unsigned/neuromodulatory sources) are exactly the
  ones the connectome structurally cannot supply.
* **The qualitative regression.** Agonism, inert antagonism alone, and
  antagonist reversal, for each scenario, hand-built and ported, under both
  baseline policies.
* **The D2 failure, and its diagnosis.** The basal-ganglia scenario does not
  survive the port. These tests pin down *why*, in numbers, so the finding
  cannot quietly rot into "it works now" or "it never worked".
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade import connectome, connectome_scenarios as cs, scenarios


T_SPAN = (0.0, 40.0)
N_POINTS = 200
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], N_POINTS)

#: A directional claim ("the agonist moves the readout this way") needs a
#: margin above integration noise, not a tolerance on a measured quantity.
#: The smallest surviving effect in this suite is ~2.6e-3, so 1e-3 keeps a
#: factor of 2.6 and still rejects the ~1e-16 non-effects the antagonist-
#: alone runs produce.
DIRECTIONAL_MARGIN = 1e-3
#: An antagonist with nothing to antagonize should be inert to solver noise.
INERT_TOLERANCE = 1e-8


@pytest.fixture(scope="module")
def circuit():
    return cs.scenario_connectome(source="fixture")


def ports():
    return [cs.SCENARIO_PORTS[k] for k in sorted(cs.SCENARIO_PORTS)]


def port_keys():
    return sorted(cs.SCENARIO_PORTS)


# ---------------------------------------------------------------------------
# Helpers: one code path for hand-built and ported runs
# ---------------------------------------------------------------------------


def trace(port, network, readout, doses, rates0=None):
    sim = port.simulation(network)
    result = sim.run(
        dose_concentrations=doses, t_span=T_SPAN, t_eval=T_EVAL, rates0=rates0
    )
    return np.array(
        [readout.score(network, result["rates"][:, i]) for i in range(N_POINTS)]
    )


def agonist_effect(port, network, readout, direction, rates0=None):
    """Signed size of the agonist's effect, in the direction the scenario claims.

    Positive means "the readout moved the way the mechanism says it should".
    Comparing maxima for a rising readout and the baseline endpoint against
    the agonist minimum for a falling one is what the hand-built scenarios'
    own tests already do; this keeps that convention so the ported and
    hand-built numbers are the same measurement.
    """
    baseline = trace(port, network, readout, port.doses(), rates0)
    agonist = trace(port, network, readout, port.doses(agonist=1.0), rates0)
    if direction > 0:
        return agonist.max() - baseline.max()
    return baseline[-1] - agonist.min()


def antagonist_deviation(port, network, readout, rates0=None):
    baseline = trace(port, network, readout, port.doses(), rates0)
    antagonist = trace(port, network, readout, port.doses(antagonist=1.0), rates0)
    return abs(antagonist[-1] - baseline[-1])


def reversal_effect(port, network, readout, direction, rates0=None):
    """Agonist for 20 units, then an antagonist bolus; how far the readout
    comes back. Positive means "back toward baseline"."""
    sim = port.simulation(network)
    split = 0.5 * (T_SPAN[0] + T_SPAN[1])
    first = np.linspace(T_SPAN[0], split, N_POINTS)
    second = np.linspace(split, T_SPAN[1], N_POINTS)

    phase1 = sim.run(
        dose_concentrations=port.doses(agonist=1.0),
        t_span=(T_SPAN[0], split),
        t_eval=first,
        rates0=rates0,
    )
    at_split = readout.score(network, phase1["rates"][:, -1])

    y = np.concatenate(
        [phase1["pk_states"][:, -1], phase1["occupancy"][:, -1], phase1["rates"][:, -1]]
    )
    y = sim.add_dose(y, port.doses(antagonist=3.0))
    sol = solve_ivp(
        sim.rhs, (split, T_SPAN[1]), y, method="RK45", t_eval=second, max_step=0.1
    )
    assert sol.success
    rates = sol.y[sim.d + sim.r :]
    after = np.array(
        [readout.score(network, rates[:, i]) for i in range(len(second))]
    )
    return at_split - after.min() if direction > 0 else after.max() - at_split


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", port_keys())
def test_every_scenario_node_is_mapped_onto_a_real_ccf_region(key, circuit):
    port = cs.SCENARIO_PORTS[key]
    hand = port.hand_built()
    assert sorted(port.by_node) == sorted(node.name for node in hand.nodes)
    for mapping in port.mappings:
        assert mapping.region in circuit.acronyms
        assert mapping.role in cs.POPULATION_ROLES


def test_the_scenarios_that_share_a_region_are_the_ones_the_connectome_cannot_split():
    """Three scenarios turn on two populations inside one CCF region.

    This is the fact that makes P3.3 a design problem rather than a lookup,
    so it is asserted rather than described: VTA, PAG and BLA each carry two
    scenario nodes, and a region-level tracer matrix has one row for each.
    """
    shared = {}
    for port in ports():
        for mapping in port.mappings:
            shared.setdefault((port.key, mapping.region), []).append(mapping.node)
    doubled = {k: v for k, v in shared.items() if len(v) > 1}
    assert doubled == {
        ("mu_opioid_reward", "VTA"): ["VTA_GABA", "VTA_DA"],
        ("mu_opioid_multi_region", "VTA"): ["VTA_GABA", "VTA_DA"],
        ("mu_opioid_multi_region", "PAG"): ["PAG_Interneuron", "PAG_Output"],
        ("gaba_a_amygdala", "BLA"): ["GABA_Interneuron", "BLA"],
    }
    for (_, _), nodes in doubled.items():
        roles = {
            m.role
            for port in ports()
            for m in port.mappings
            if m.node in nodes
        }
        assert roles == {"local_interneuron", "projection"}


def test_a_restriction_without_a_citation_is_refused():
    with pytest.raises(ValueError, match="needs a citation"):
        cs.Restriction(targets=("GPe",), reason="because", citations=())
    with pytest.raises(ValueError, match="no targets"):
        cs.Restriction(targets=(), reason="because", citations=("x",))


def test_only_one_restriction_is_declared_and_it_is_cited(circuit):
    restricted = [
        (port.key, m.node, m.restriction)
        for port in ports()
        for m in port.mappings
        if m.restriction is not None
    ]
    assert len(restricted) == 1
    key, node, restriction = restricted[0]
    assert (key, node) == ("d2_indirect", "Striatum_D2_MSN")
    assert restriction.targets == ("GPe",)
    assert len(restriction.citations) == 2
    for citation in restriction.citations:
        assert citation.startswith("[")
    for target in restriction.targets:
        assert target in circuit.acronyms


def test_a_local_interneuron_cannot_carry_a_restriction():
    with pytest.raises(ValueError, match="no long-range output to restrict"):
        cs.RegionMapping(
            "x", "VTA", "local_interneuron",
            restriction=cs.Restriction(("GPe",), "r", ("c",)),
        )


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="role must be one of"):
        cs.RegionMapping("x", "VTA", "excitatory")


# ---------------------------------------------------------------------------
# Provenance: every weight can say where it came from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", port_keys())
def test_every_weight_is_either_published_or_labelled_illustrative(key, circuit):
    port = cs.SCENARIO_PORTS[key]
    ported = cs.port_scenario(port, circuit)
    hand = port.hand_built()
    seen = set()
    for edge in ported.report.edges:
        assert edge.provenance in cs.PROVENANCES
        seen.add((edge.post, edge.pre))
        i = ported.network.index[edge.post]
        j = ported.network.index[edge.pre]
        assert ported.network.connectivity[i, j] == edge.weight
        source = port.by_node[edge.pre]
        target = port.by_node[edge.post]
        if edge.provenance == "connectome":
            assert edge.weight == circuit.weight(source.region, target.region)
        elif edge.provenance in ("intra_regional", "neuromodulatory_illustrative"):
            assert edge.weight == edge.hand_built_weight
        else:
            assert edge.weight == 0.0
    # every ordered pair accounted for, none twice
    assert len(seen) == len(port.node_names) ** 2
    assert len(ported.report.edges) == len(seen)
    assert ported.network.n == hand.n


@pytest.mark.parametrize("key", port_keys())
def test_illustrative_edges_are_exactly_the_ones_the_connectome_cannot_supply(
    key, circuit
):
    """An edge keeps a hand-built number only for a structural reason.

    Either both endpoints are the same CCF region (P3.2 drops that diagonal
    on purpose) or the source region carries no transmitter sign. There is
    no third case, and in particular no case where a published number
    existed and was passed over.
    """
    port = cs.SCENARIO_PORTS[key]
    ported = cs.port_scenario(port, circuit)
    signed = {a.acronym: a for a in circuit.report.fine_assignments}
    for edge in ported.report.illustrative_edges():
        source = port.by_node[edge.pre]
        target = port.by_node[edge.post]
        assert (
            source.region == target.region or not signed[source.region].is_signed
        ), edge


def test_a_local_interneuron_has_no_long_range_output(circuit):
    port = cs.SCENARIO_PORTS["mu_opioid_multi_region"]
    ported = cs.port_scenario(port, circuit)
    for interneuron in ("VTA_GABA", "PAG_Interneuron"):
        column = ported.network.connectivity[:, ported.network.index[interneuron]]
        region = port.by_node[interneuron].region
        for node, weight in zip(port.node_names, column):
            if port.by_node[node].region != region:
                assert weight == 0.0, (interneuron, node)


def test_the_local_interneuron_rule_prevents_double_counting_a_projection(circuit):
    """Two scenario nodes live in BLA and there is one published BLA -> CEA
    projection, so it must arrive exactly once.

    This is what the "long-range output leaves only from the projection
    population" approximation buys, and the report names the published number
    it declined rather than hiding a zero.
    """
    port = cs.SCENARIO_PORTS["gaba_a_amygdala"]
    ported = cs.port_scenario(port, circuit)
    assert ported.weight("BLA", "CeA") == circuit.weight("BLA", "CEA")
    assert ported.weight("GABA_Interneuron", "CeA") == 0.0
    discarded = ported.report.discarded_edges()
    assert [(e.post, e.pre, e.provenance) for e in discarded] == [
        ("CeA", "GABA_Interneuron", "local_interneuron_source")
    ]
    assert discarded[0].published_weight == circuit.weight("BLA", "CEA")


def test_long_range_input_reaches_every_population_in_the_target_region(circuit):
    """The other approximation, stated in the module docstring, asserted here.

    `PL -> VTA` is one published number and both VTA populations sit in those
    voxels, so both receive it. Giving the whole projection to the principal
    population instead would assert that cortical afferents avoid local
    interneurons, which is the opposite of the feedforward-inhibition motif
    the scenario's own hand-built `PFC -> VTA_GABA` edge was drawn for.
    """
    port = cs.SCENARIO_PORTS["mu_opioid_reward"]
    ported = cs.port_scenario(port, circuit)
    published = circuit.weight("PL", "VTA")
    assert published > 0.0
    assert ported.weight("PFC", "VTA_DA") == published
    assert ported.weight("PFC", "VTA_GABA") == published
    # and the intra-VTA edge is still the illustrative one, not a copy of a
    # published number
    assert ported.weight("VTA_GABA", "VTA_DA") == -2.5


def test_the_mesolimbic_dopamine_edge_is_flagged_not_fabricated(circuit):
    """VTA carries no sign, so VTA_DA -> NAc has no published weight.

    P3.2's rule leaves dopaminergic sources unsigned because a dopaminergic
    axon changes how its target responds rather than adding a synaptic term,
    which is the layer the receptor model already occupies. The port keeps
    the hand-built weight so the scenario's mechanism can still be tested,
    and labels it; `unsigned_edges="drop"` removes it instead. What no
    setting does is give it a published-looking number.
    """
    port = cs.SCENARIO_PORTS["mu_opioid_reward"]
    retained = cs.port_scenario(port, circuit)
    edge = [
        e
        for e in retained.report.edges
        if (e.post, e.pre) == ("NAc", "VTA_DA")
    ][0]
    assert edge.provenance == "neuromodulatory_illustrative"
    assert edge.weight == 2.0  # scenarios.reward_circuit_network()'s own number
    assert circuit.weight("VTA", "ACB") == 0.0

    dropped = cs.port_scenario(port, circuit, unsigned_edges="drop")
    assert dropped.weight("VTA_DA", "NAc") == 0.0
    assert [
        e for e in dropped.report.edges if (e.post, e.pre) == ("NAc", "VTA_DA")
    ][0].provenance == "unsigned_source"


def test_the_published_data_contradicts_two_hand_built_reward_edges(circuit):
    """Two of the reward circuit's hand-drawn arrows are not what the tracer says.

    Both are reported rather than silently overwritten, because they are the
    substance of what porting a scenario buys: `PFC -| VTA_DA` was drawn as
    net inhibitory feedback and the corticotegmental projection is
    glutamatergic; `NAc -> PFC` was drawn excitatory and accumbal output is
    GABAergic, and in Knox et al. 2019 it is also two orders of magnitude
    weaker than the guess.
    """
    port = cs.SCENARIO_PORTS["mu_opioid_reward"]
    ported = cs.port_scenario(port, circuit)
    contradictions = {
        (e.post, e.pre): e for e in ported.report.contradictions()
    }
    assert set(contradictions) == {("VTA_DA", "PFC"), ("PFC", "NAc")}

    flipped = contradictions[("VTA_DA", "PFC")]
    assert flipped.hand_built_weight < 0 < flipped.weight
    assert flipped.weight == pytest.approx(0.500810, rel=1e-5)

    vanished = contradictions[("PFC", "NAc")]
    assert vanished.hand_built_weight == 1.5
    assert vanished.weight == pytest.approx(-0.009820, rel=1e-4)


def test_the_restriction_reports_what_it_threw_away(circuit):
    port = cs.SCENARIO_PORTS["d2_indirect"]
    ported = cs.port_scenario(port, circuit)
    cost = ported.report.discarded_by_restriction
    assert set(cost) == {"Striatum_D2_MSN"}
    # Measured over the whole loaded connectome, so it is a statement about
    # the caudoputamen and not about the four other scenario nodes. On the
    # fifteen-region fixture that is 79%; on the 291-region files it is 91%
    # (a full-data test below).
    assert 0.7 < cost["Striatum_D2_MSN"] < 0.85
    restricted = [
        e for e in ported.report.edges if e.provenance == "restricted"
    ]
    assert {e.post for e in restricted} == {"STN", "GPi", "Thalamus"}
    assert all(e.weight == 0.0 for e in restricted)
    # the discarded published weights, and the citation for discarding them,
    # both appear in the summary rather than only in the source
    text = ported.report.summary()
    assert "-12.2113  restricted" in text
    assert "[Gerfen2011]" in text
    assert ported.report.restriction_citations() == sorted(
        cs.SCENARIO_PORTS["d2_indirect"].by_node["Striatum_D2_MSN"].restriction.citations
    )


def test_the_striatal_row_contains_the_direct_pathway_too(circuit):
    """Why the restriction exists at all, in the data's own numbers.

    If the caudoputamen row were the indirect pathway, its pallidal-external
    target would dominate. It does not: the two direct-pathway targets carry
    as much or more.
    """
    assert circuit.weight("CP", "GPe") == pytest.approx(-12.558671, rel=1e-6)
    assert circuit.weight("CP", "GPi") == pytest.approx(-12.211333, rel=1e-6)
    assert circuit.weight("CP", "SNr") == pytest.approx(-16.336, rel=1e-4)
    assert abs(circuit.weight("CP", "SNr")) > abs(circuit.weight("CP", "GPe"))


# ---------------------------------------------------------------------------
# Knobs and refusals
# ---------------------------------------------------------------------------


def test_weight_scale_rescales_the_published_edges_and_nothing_else(circuit):
    port = cs.SCENARIO_PORTS["mu_opioid_reward"]
    plain = cs.port_scenario(port, circuit)
    scaled = cs.port_scenario(port, circuit, weight_scale=0.1)
    for a, b in zip(plain.report.edges, scaled.report.edges):
        assert (a.post, a.pre, a.provenance) == (b.post, b.pre, b.provenance)
        if a.provenance == "connectome":
            assert b.weight == pytest.approx(0.1 * a.weight)
        else:
            assert b.weight == a.weight


def test_the_default_scale_is_the_published_number_unrescaled(circuit):
    port = cs.SCENARIO_PORTS["gaba_a_amygdala"]
    ported = cs.port_scenario(port, circuit)
    assert ported.report.weight_scale == 1.0
    assert ported.weight("BLA", "CeA") == circuit.weight("BLA", "CEA")


def test_ported_networks_carry_a_comparable_total_input_weight(circuit):
    """The module docstring's justification for not choosing a scale.

    P3.2 reported that at 291 nodes the unrescaled densities saturate every
    node. At scenario scale they do not, and this is the measurement that
    says so: mean per-node total input weight, ported against hand-built,
    within a factor of 2.5 for every scenario.
    """
    for port in ports():
        hand = port.hand_built()
        ported = cs.port_scenario(port, circuit).network
        hand_drive = np.abs(hand.connectivity).sum(axis=1).mean()
        ported_drive = np.abs(ported.connectivity).sum(axis=1).mean()
        ratio = ported_drive / hand_drive
        assert 0.4 < ratio < 2.5, (port.key, ratio)


def test_porting_does_not_touch_the_hand_built_scenario(circuit):
    """Acceptance criterion: the hand-built networks are retained, not replaced."""
    for port in ports():
        before = port.hand_built()
        cs.port_scenario(port, circuit)
        after = port.hand_built()
        assert np.array_equal(before.connectivity, after.connectivity)
        assert [n.baseline_input for n in before.nodes] == [
            n.baseline_input for n in after.nodes
        ]
    # and the module the tests of record import is untouched
    assert scenarios.basal_ganglia_indirect_pathway_network().connectivity[1, 0] == -2.0
    assert scenarios.reward_circuit_network().connectivity[1, 0] == -2.5


def test_bad_policies_and_bad_overrides_are_refused(circuit):
    port = cs.SCENARIO_PORTS["gaba_a_amygdala"]
    with pytest.raises(ValueError, match="unsigned_edges must be one of"):
        cs.port_scenario(port, circuit, unsigned_edges="excitatory")
    with pytest.raises(ValueError, match="baselines must be one of"):
        cs.port_scenario(port, circuit, baselines="fitted")
    with pytest.raises(ValueError, match="has no node"):
        cs.port_scenario(
            port, circuit, overrides={"Amygdala": cs.RegionMapping("x", "BLA", "projection")}
        )
    with pytest.raises(ValueError, match="does not contain"):
        cs.port_scenario(
            port,
            circuit,
            overrides={"CeA": cs.RegionMapping("CeA", "VISp", "projection")},
        )
    # Two projection populations in one region would deliver that region's
    # published output twice - the double-count the local-interneuron rule
    # exists to prevent. Refused, not silently summed.
    with pytest.raises(ValueError, match="two projection populations"):
        cs.port_scenario(
            port,
            circuit,
            overrides={
                "GABA_Interneuron": cs.RegionMapping(
                    "GABA_Interneuron", "BLA", "projection"
                )
            },
        )
    with pytest.raises(ValueError, match="source must be"):
        cs.scenario_connectome(source="knox")


def test_an_overridden_map_is_what_the_report_prints(circuit):
    """A counterfactual build must not advertise the declared regions.

    The report reads the mapping actually used, so a run that swaps the
    output nucleus says SNr and carries SNr's weights, rather than printing
    the scenario's declaration over someone else's numbers.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    ported = cs.port_scenario(
        port,
        circuit,
        overrides={"GPi": cs.RegionMapping("GPi", "SNr", "projection")},
    )
    assert "GPi                SNr" in ported.report.summary()
    assert ported.weight("STN", "GPi") == circuit.weight("STN", "SNr")
    # ...and the declaration itself is unchanged for the next caller
    assert port.by_node["GPi"].region == "GPi"


def test_a_coarse_connectome_is_refused(circuit):
    strength, voxels = connectome.load_fixture_build_inputs()
    coarse = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="major_divisions"
    )
    with pytest.raises(ValueError, match="resolution='regions'"):
        cs.port_scenario(cs.SCENARIO_PORTS["gaba_a_amygdala"], coarse)


# ---------------------------------------------------------------------------
# rest_balanced_baselines: a rule, and where it came from
# ---------------------------------------------------------------------------


def test_the_baseline_rule_reproduces_the_hand_built_basal_ganglia_baselines():
    """The rule is not invented here; it is what that scenario already did.

    `scenarios.basal_ganglia_indirect_pathway_network` says its
    `baseline_input` values "are chosen so each node sits near the sensitive
    (non-saturating) part of its sigmoid at rest". Solving that sentence
    gives back its five numbers exactly, which is what licenses using the
    same rule on the ported weights.
    """
    hand = scenarios.basal_ganglia_indirect_pathway_network()
    rebalanced = cs.rest_balanced_baselines(hand)
    assert [n.baseline_input for n in hand.nodes] == [0.0, 1.0, 1.0, -1.0, 1.0]
    assert [n.baseline_input for n in rebalanced.nodes] == pytest.approx(
        [n.baseline_input for n in hand.nodes]
    )
    assert np.array_equal(rebalanced.connectivity, hand.connectivity)


def test_the_rebalanced_rest_state_is_an_exact_fixed_point(circuit):
    for port in ports():
        network = cs.port_scenario(port, circuit, baselines="rest_balanced").network
        rates = np.full(network.n, 0.5)
        assert np.allclose(
            network.rhs(rates, network.baseline_drive()), 0.0, atol=1e-12
        ), port.key


def test_rebalancing_leaves_the_other_node_parameters_alone(circuit):
    port = cs.SCENARIO_PORTS["gaba_a_amygdala"]
    plain = cs.port_scenario(port, circuit).network
    rebalanced = cs.port_scenario(port, circuit, baselines="rest_balanced").network
    assert np.array_equal(plain.connectivity, rebalanced.connectivity)
    for a, b in zip(plain.nodes, rebalanced.nodes):
        assert (a.name, a.tau, a.gain, a.threshold) == (b.name, b.tau, b.gain, b.threshold)


# ---------------------------------------------------------------------------
# The qualitative regression, scenario by scenario
# ---------------------------------------------------------------------------


SURVIVING = ["gaba_a_amygdala", "mu_opioid_multi_region", "mu_opioid_reward"]


@pytest.mark.parametrize("key", SURVIVING)
@pytest.mark.parametrize("baselines", cs.BASELINE_POLICIES)
def test_agonism_survives_the_port(key, baselines, circuit):
    port = cs.SCENARIO_PORTS[key]
    ported = cs.port_scenario(port, circuit, baselines=baselines).network
    hand = port.hand_built()
    if baselines == "rest_balanced":
        hand = cs.rest_balanced_baselines(hand)
    for name, (make_readout, direction) in port.readouts.items():
        readout = make_readout()
        assert (
            agonist_effect(port, hand, readout, direction) > DIRECTIONAL_MARGIN
        ), (key, name, "hand-built")
        assert (
            agonist_effect(port, ported, readout, direction) > DIRECTIONAL_MARGIN
        ), (key, name, "ported")


@pytest.mark.parametrize("key", port_keys())
@pytest.mark.parametrize("baselines", cs.BASELINE_POLICIES)
def test_antagonist_alone_stays_inert_on_the_ported_network(key, baselines, circuit):
    """Inert antagonism is the one result that survives every scenario.

    It should: with no agonist on board there is nothing to displace, so a
    zero-efficacy ligand can only matter through a bug. That includes the D2
    scenario, whose agonist result does not survive.
    """
    port = cs.SCENARIO_PORTS[key]
    ported = cs.port_scenario(port, circuit, baselines=baselines).network
    for name, (make_readout, _) in port.readouts.items():
        deviation = antagonist_deviation(port, ported, make_readout())
        assert deviation < INERT_TOLERANCE, (key, name, deviation)


@pytest.mark.parametrize("key", SURVIVING)
@pytest.mark.parametrize("baselines", cs.BASELINE_POLICIES)
def test_antagonist_reversal_survives_the_port(key, baselines, circuit):
    port = cs.SCENARIO_PORTS[key]
    ported = cs.port_scenario(port, circuit, baselines=baselines).network
    for name, (make_readout, direction) in port.readouts.items():
        effect = reversal_effect(port, ported, make_readout(), direction)
        assert effect > DIRECTIONAL_MARGIN, (key, name, effect)


def test_the_ported_analgesia_and_arousal_effects_still_point_opposite_ways(circuit):
    """One occupancy trace, three readouts, two directions - the multi-region
    scenario's whole point, checked on published wiring."""
    port = cs.SCENARIO_PORTS["mu_opioid_multi_region"]
    network = cs.port_scenario(port, circuit).network
    baseline = trace(port, network, scenarios.arousal_index_readout(), port.doses())
    agonist = trace(
        port, network, scenarios.arousal_index_readout(), port.doses(agonist=1.0)
    )
    assert agonist[-1] < baseline[-1] - 0.05
    analgesia_base = trace(
        port, network, scenarios.analgesia_index_readout(), port.doses()
    )
    analgesia_drug = trace(
        port, network, scenarios.analgesia_index_readout(), port.doses(agonist=1.0)
    )
    assert analgesia_drug[-1] > analgesia_base[-1] + 0.1


# ---------------------------------------------------------------------------
# The D2 scenario: the failure, and its diagnosis
# ---------------------------------------------------------------------------


def d2_locomotor_effect(network, rates0=None):
    port = cs.SCENARIO_PORTS["d2_indirect"]
    return agonist_effect(
        port, network, scenarios.locomotor_activity_readout(), +1, rates0
    )


def test_the_d2_indirect_pathway_result_does_not_survive_the_port(circuit):
    """The documented failure. Diagnosed by the four tests below.

    Hand-built, a D2 agonist raises the locomotor readout by ~0.11. Ported,
    the effect is zero to seven decimal places under either baseline policy.
    This test exists so that the failure is a checked fact: if a later change
    makes it pass, that change has to explain itself here.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    assert d2_locomotor_effect(port.hand_built()) > 0.1

    hand_baselines = cs.port_scenario(port, circuit).network
    assert abs(d2_locomotor_effect(hand_baselines)) < 1e-4

    balanced = cs.port_scenario(port, circuit, baselines="rest_balanced").network
    assert abs(d2_locomotor_effect(balanced, np.full(5, 0.5))) < 1e-4


def test_the_d2_response_is_not_merely_absent_but_inverted(circuit):
    """A sharper statement of the failure than "the effect is zero".

    Comparing the two trajectories pointwise rather than by their maxima:
    hand-built, the agonist run is never below its own drug-free run and
    rises 0.47 above it. Ported, under either baseline policy, it is never
    ABOVE its drug-free run at any time, and goes about 0.3 below. The D2
    agonist does something to the ported circuit; what it does is the
    opposite of what the scenario claims.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    readout = scenarios.locomotor_activity_readout()

    def difference(network, rates0=None):
        baseline = trace(port, network, readout, port.doses(), rates0)
        agonist = trace(port, network, readout, port.doses(agonist=1.0), rates0)
        return agonist - baseline

    hand = difference(port.hand_built())
    assert hand.max() > 0.4
    assert hand.min() > -1e-6

    for baselines, rates0 in (("hand_built", None), ("rest_balanced", np.full(5, 0.5))):
        network = cs.port_scenario(port, circuit, baselines=baselines).network
        ported = difference(network, rates0)
        assert ported.max() < 1e-6, baselines
        assert ported.min() < -0.3, baselines


def test_diagnosis_1_the_hand_built_baselines_saturate_the_ported_pallidum(circuit):
    """First cause: a baseline chosen against one W does not fit another one.

    GPe's `baseline_input` of 1.0 was set against a hand-built striatopallidal
    weight of -2.0. Knox et al. 2019's published striatopallidal density is
    -12.56, so at rest GPe receives about -9 and the illustrative gain of 4
    pins it at zero. Nothing downstream of it can then respond to anything.
    This is the GPi-saturation lesson of `scenarios.py` reappearing, and it is
    P3.4's problem, not the connectome's.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    hand = port.hand_built()
    ported = cs.port_scenario(port, circuit).network

    sim = port.simulation(ported)
    rest = sim.run(
        dose_concentrations=port.doses(), t_span=T_SPAN, t_eval=T_EVAL
    )["rates"][:, -1]
    assert rest[ported.index["GPe"]] < 1e-3

    hand_sim = port.simulation(hand)
    hand_rest = hand_sim.run(
        dose_concentrations=port.doses(), t_span=T_SPAN, t_eval=T_EVAL
    )["rates"][:, -1]
    assert hand_rest == pytest.approx(np.full(5, 0.5), abs=1e-3)

    assert ported.connectivity[ported.index["GPe"], ported.index["Striatum_D2_MSN"]] == (
        pytest.approx(-12.558671, rel=1e-6)
    )
    assert hand.nodes[hand.index["GPe"]].baseline_input == 1.0


def test_diagnosis_2_with_baselines_rebalanced_the_pathway_inverts_instead(circuit):
    """Second cause, and the substantive one: the thalamus moves the wrong way.

    Rebalanced, every node rests at exactly 0.5 and nothing is saturated, so
    the chain is free to run - and it runs backwards. Striatum falls, GPe
    rises, STN falls and GPi falls, all as the indirect-pathway diagram says,
    and the thalamus then falls too instead of rising.

    P3.4 corrects one thing about the setup, without changing the finding:
    that r = 1/2 fixed point is exact but UNSTABLE (the ported CP<->GPe pair
    is a two-inhibitory-link positive-feedback latch), so `rates0` here is
    sitting on a repeller. `tests/test_stability.py` shows both halves of
    that - the instability
    (`test_the_rebalanced_ported_d2_fixed_point_is_exact_but_unstable`) and
    the fact that the inversion this test measures survives on a
    certified-stable operating point where the rest state is the one the
    network actually finds
    (`test_p3_3s_d2_inversion_survives_on_a_certified_stable_network`).
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    network = cs.port_scenario(port, circuit, baselines="rest_balanced").network
    sim = port.simulation(network)
    rates0 = np.full(network.n, 0.5)
    agonist = sim.run(
        dose_concentrations=port.doses(agonist=1.0),
        t_span=T_SPAN,
        t_eval=T_EVAL,
        rates0=rates0,
    )["rates"]

    index = network.index
    assert agonist[index["Striatum_D2_MSN"]].min() < 0.01  # D2 agonism suppresses it
    assert agonist[index["GPe"]].max() > 0.99              # disinhibited, as intended
    assert agonist[index["STN"]].min() < 0.02              # inhibited by GPe
    assert agonist[index["GPi"]].min() < 0.11              # loses STN drive
    # ...and the readout falls anyway.
    assert agonist[index["Thalamus"]].max() <= 0.5 + 1e-9
    assert agonist[index["Thalamus"]].min() < 0.2


def test_diagnosis_3_the_pallidothalamic_edge_is_what_inverts_it(circuit):
    """The single number responsible: GPe -> thalamus is 7.6x GPi -> thalamus.

    The textbook indirect-pathway diagram routes GPe's entire influence on the
    thalamus through STN and GPi. Knox et al. 2019 has GPe projecting to the
    thalamus directly, seven and a half times more densely than GPi does, so
    when GPe rises the thalamus is inhibited harder than the falling GPi
    disinhibits it. A direct pallidothalamic projection is not an artefact of
    this dataset: Mastro KJ, Bouchard RS, Holt HAK, Gittis AH. "Transgenic
    mouse lines subdivide external segment of the globus pallidus (GPe)
    neurons and reveal distinct GPe output pathways." J Neurosci.
    2014;34(6):2087-2099. doi:10.1523/JNEUROSCI.4646-13.2014 reports mouse GPe
    axons in the reticular thalamic nucleus from both GPe subpopulations and
    dense PV-GPe projections to the parafascicular nucleus. Whether the
    MD-specific density Knox reports is a genuine GPe -> MD projection or
    signal shared with the neighbouring RT in a voxel-scale interpolated model
    is not something these files can settle.
    """
    ratio = circuit.weight("GPe", "MD") / circuit.weight("GPi", "MD")
    assert ratio == pytest.approx(7.58, abs=0.05)

    port = cs.SCENARIO_PORTS["d2_indirect"]
    counterfactual = cs.RegionMapping(
        "GPe",
        "GPe",
        "projection",
        restriction=cs.Restriction(
            targets=("STN", "GPi"),
            reason="counterfactual only: the classic box-and-arrow diagram "
                   "gives GPe no thalamic projection",
            citations=(
                "[Albin1989] Albin RL, Young AB, Penney JB. Trends Neurosci. "
                "1989;12(10):366-375 - the diagram, NOT a claim that GPe has "
                "no thalamic projection; Mastro et al. 2014 shows it does.",
            ),
        ),
    )
    restored = cs.port_scenario(
        port, circuit, baselines="rest_balanced", overrides={"GPe": counterfactual}
    ).network
    assert d2_locomotor_effect(restored, np.full(5, 0.5)) > 0.01


def test_diagnosis_4_swapping_the_output_nucleus_alone_does_not_rescue_it(circuit):
    """The rodent objection, tested rather than assumed.

    The hand-built diagram uses GPi as the basal-ganglia output nucleus, which
    is the primate arrangement; in rodents GPi is the small entopeduncular
    nucleus and SNr is the main output. Remapping the scenario's `GPi` node
    onto SNr does not restore the result, which is what identifies the
    pallidothalamic edge above as the cause rather than the choice of output
    nucleus.
    """
    port = cs.SCENARIO_PORTS["d2_indirect"]
    snr = cs.RegionMapping("GPi", "SNr", "projection")
    network = cs.port_scenario(
        port, circuit, baselines="rest_balanced", overrides={"GPi": snr}
    ).network
    assert abs(d2_locomotor_effect(network, np.full(5, 0.5))) < 1e-3


# ---------------------------------------------------------------------------
# Full-data extras: skipped without the gitignored cache
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
def full_circuit():
    if not _FULL_DATA:
        pytest.skip("no cache")
    return cs.scenario_connectome(source="cache")


@full_data_only
def test_fixture_and_full_connectome_give_the_same_ported_networks(
    circuit, full_circuit
):
    """The claim that lets this whole suite run offline, checked.

    A ported network only ever reads region-pair entries, and a connection
    density is a per-region-pair quantity, so slicing the published file down
    to fifteen regions cannot change any of them. Exact equality, not
    approximate.
    """
    assert full_circuit.report.dataset_key == circuit.report.dataset_key
    assert len(full_circuit.acronyms) == 291
    for port in ports():
        a = cs.port_scenario(port, circuit).network
        b = cs.port_scenario(port, full_circuit).network
        assert np.array_equal(a.connectivity, b.connectivity), port.key


@full_data_only
def test_the_restriction_discards_most_of_the_caudoputamen_row(full_circuit):
    port = cs.SCENARIO_PORTS["d2_indirect"]
    cost = cs.port_scenario(port, full_circuit).report.discarded_by_restriction
    # Over all 291 regions, GPe carries 9.5% of CP's outgoing weight.
    assert cost["Striatum_D2_MSN"] == pytest.approx(0.905, abs=0.005)


@full_data_only
@pytest.mark.parametrize("nucleus", ["VAL", "VM", "CM", "PF"])
def test_the_d2_failure_does_not_depend_on_which_thalamic_nucleus_is_used(
    nucleus, full_circuit
):
    """MD was chosen because it is in the fixture. It is not why D2 fails.

    Every candidate motor/limbic thalamic target has the same 7-9x
    GPe-over-GPi pallidothalamic ratio, and the scenario fails on each.
    """
    ratio = full_circuit.weight("GPe", nucleus) / full_circuit.weight("GPi", nucleus)
    assert 7.0 < ratio < 9.0
    port = cs.SCENARIO_PORTS["d2_indirect"]
    network = cs.port_scenario(
        port,
        full_circuit,
        baselines="rest_balanced",
        overrides={"Thalamus": cs.RegionMapping("Thalamus", nucleus, "projection")},
    ).network
    assert d2_locomotor_effect(network, np.full(5, 0.5)) < 1e-3


@full_data_only
@pytest.mark.parametrize("area", ["ILA", "ACAd"])
def test_the_reward_result_does_not_depend_on_which_prefrontal_area_is_used(
    area, full_circuit
):
    """PL is the declared mouse mPFC node; the alternatives give the same result."""
    port = cs.SCENARIO_PORTS["mu_opioid_reward"]
    network = cs.port_scenario(
        port,
        full_circuit,
        overrides={"PFC": cs.RegionMapping("PFC", area, "projection")},
    ).network
    effect = agonist_effect(
        port, network, scenarios.incentive_salience_readout(), +1
    )
    assert effect > DIRECTIONAL_MARGIN


@full_data_only
def test_the_summary_says_the_things_that_get_over_read(full_circuit):
    text = cs.port_scenario(
        cs.SCENARIO_PORTS["mu_opioid_reward"], full_circuit
    ).report.summary()
    assert "ILLUSTRATIVE" in text
    assert "knox2019_connection_strength" in text
    assert "weight_scale=1" in text
    assert "NAc                <- VTA_DA" in text
    assert "+2.0000  neuromodulatory_illustrative" in text
    assert "contradicts the hand-built guess" in text
