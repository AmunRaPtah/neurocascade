"""P3.2: building a `CircuitNetwork` from the connectome.

Everything here runs offline off the committed fixtures (real published
data, fifteen regions of it; see `fixtures/PROVENANCE.md`), except the
five tests at the bottom that check the real 291-region matrices and skip
when the gitignored cache is absent. Those are extras: every acceptance
criterion of P3.2 is covered by the offline tests.

Three of these tests are load-bearing in a way worth naming, because they
are the ones that would catch the mistakes that produce a healthy-looking
but wrong whole-brain model:

* `test_region_resolution_reproduces_the_published_connection_density`
  rebuilds Knox et al. 2019's own `connection_density` file from the raw
  strength matrix and the recovered voxel counts, exactly. If the
  aggregation arithmetic or the voxel-count recovery were wrong, this
  fails.
* `test_orientation_and_sign_are_both_right_on_a_known_pair` asserts
  MOp->CP and CP->MOp separately, with opposite signs. A transposed
  connectome or an inverted sign convention cannot pass it by accident.
* `test_circuit_network_api_is_unchanged` pins the constructor signatures
  of `CircuitNetwork` and `CircuitNode`, which is the acceptance criterion
  about soma-twin's import path.
"""

import inspect

import numpy as np
import pytest

from neurocascade import CircuitNetwork
from neurocascade.legacy import connectome
from neurocascade.circuit import CircuitNode
from neurocascade.legacy.connectome import (
    build,
    fetch,
    neurotransmitter as nt,
    sources,
    volumes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ccf():
    return connectome.load_fixture_ontology()


@pytest.fixture(scope="module")
def inputs(ccf):
    """(strength matrix, target voxel counts) off the committed fixtures."""
    return connectome.load_fixture_build_inputs(ontology=ccf)


@pytest.fixture(scope="module")
def rules(ccf):
    return nt.SignRules(ccf)


def built(inputs, **kwargs):
    strength, voxel_counts = inputs
    return connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, **kwargs
    )


# ---------------------------------------------------------------------------
# The sign-assignment rule
# ---------------------------------------------------------------------------


def test_every_rule_names_a_structure_this_ontology_agrees_about(ccf, rules):
    # SignRules' constructor cross-checks acronym against id for every rule;
    # this asserts it actually ran over the whole table rather than a subset.
    assert len(rules.rules) == len(nt.SIGN_RULES)
    for rule in nt.SIGN_RULES:
        assert ccf[rule.structure_id].acronym == rule.acronym


def test_every_rule_carries_a_citation_that_exists():
    for rule in nt.SIGN_RULES:
        assert rule.citations, rule.acronym
        for key in rule.citations:
            assert key in nt.CITATIONS
            # A citation is a real one: author, year, journal, species.
            assert len(nt.CITATIONS[key]) > 100


def test_a_rule_without_a_citation_is_refused():
    with pytest.raises(ValueError, match="no citation"):
        nt.TransmitterRule("XX", 1, nt.GLUTAMATERGIC, (), "")
    with pytest.raises(ValueError, match="unknown key"):
        nt.TransmitterRule("XX", 1, nt.GLUTAMATERGIC, ("NotAPaper2099",), "")
    with pytest.raises(ValueError, match="unknown transmitter"):
        nt.TransmitterRule("XX", 1, "vibes", ("Yao2023",), "")


def test_only_glutamate_and_gaba_carry_a_sign():
    signed = {k for k, v in nt.TRANSMITTER_SIGN.items() if v is not None}
    assert signed == {nt.GLUTAMATERGIC, nt.GABAERGIC}
    assert nt.TRANSMITTER_SIGN[nt.GLUTAMATERGIC] == 1.0
    assert nt.TRANSMITTER_SIGN[nt.GABAERGIC] == -1.0


def test_a_renamed_structure_raises_rather_than_being_re_pointed(ccf):
    wrong = (
        nt.TransmitterRule("NotMOp", 985, nt.GLUTAMATERGIC, ("Yao2023",), "test"),
    )
    with pytest.raises(ValueError, match="has been revised"):
        nt.SignRules(ccf, rules=wrong)


def test_sign_comes_from_the_nearest_ruled_ancestor(ccf, rules):
    # Cortex and thalamus: excitatory, via a division-level rule.
    for acronym, anchor in [("MOp", "Isocortex"), ("PL", "Isocortex"),
                            ("MD", "TH"), ("BLA", "CTXsp")]:
        assignment = rules.assign(ccf.id_for_acronym(acronym))
        assert assignment.transmitter == nt.GLUTAMATERGIC
        assert assignment.sign == 1.0
        assert assignment.rule_acronym == anchor

    # Striatum and pallidum: inhibitory, likewise.
    for acronym, anchor in [("CP", "STR"), ("ACB", "STR"), ("CEA", "STR"),
                            ("GPe", "PAL"), ("GPi", "PAL")]:
        assignment = rules.assign(ccf.id_for_acronym(acronym))
        assert assignment.transmitter == nt.GABAERGIC
        assert assignment.sign == -1.0
        assert assignment.rule_acronym == anchor


def test_a_specific_rule_beats_the_division_it_sits_in(ccf, rules):
    # RT is inside TH, whose rule is +1; RT's own rule must win.
    thalamus = rules.assign(ccf.id_for_acronym("MD"))
    reticular = rules.assign(ccf.id_for_acronym("RT"))
    assert thalamus.sign == 1.0
    assert reticular.sign == -1.0
    assert reticular.rule_acronym == "RT"

    # PALv is inside PAL, whose rule is -1; PALv is mixed, so it loses its
    # sign rather than inheriting one.
    assert rules.assign(ccf.id_for_acronym("PAL")).sign == -1.0
    ventral = rules.assign(ccf.id_for_acronym("PALv"))
    assert ventral.transmitter == nt.MIXED
    assert ventral.sign is None


def test_neuromodulatory_nuclei_are_classified_and_left_unsigned(ccf, rules):
    for acronym, transmitter in [
        ("VTA", nt.DOPAMINERGIC),
        ("SNc", nt.DOPAMINERGIC),
        ("DR", nt.SEROTONERGIC),
        ("LC", nt.NORADRENERGIC),
    ]:
        assignment = rules.assign(ccf.id_for_acronym(acronym))
        assert assignment.transmitter == transmitter
        assert assignment.sign is None
        assert not assignment.is_signed
        assert assignment.citations  # classified, not merely unknown


def test_a_region_outside_every_rule_is_unassigned_not_guessed(ccf, rules):
    # PAG is in the midbrain, which has no default rule: its transmitter
    # composition is genuinely mixed and no source was established here.
    assignment = rules.assign(ccf.id_for_acronym("PAG"))
    assert assignment.rule is None
    assert assignment.transmitter is None
    assert assignment.sign is None
    assert assignment.citations == ()


def test_stn_is_signed_excitatory_despite_sitting_in_the_hypothalamus(ccf, rules):
    # The CCF puts STN under HY, which has no default. Without STN's own
    # rule the D2 indirect pathway's one excitatory edge would vanish.
    assert ccf.major_division(ccf.id_for_acronym("STN")) == ccf.id_for_acronym("HY")
    assignment = rules.assign(ccf.id_for_acronym("STN"))
    assert assignment.sign == 1.0
    assert assignment.rule_acronym == "STN"


# ---------------------------------------------------------------------------
# Voxel-count recovery
# ---------------------------------------------------------------------------


def test_voxel_counts_recovered_from_the_published_ratio_are_whole_numbers(inputs, ccf):
    _, voxel_counts = inputs
    counts = {ccf[i].acronym: v for (h, i), v in voxel_counts.items() if h == "ipsi"}
    assert counts["MOp"] == 5658.0
    assert counts["CP"] == 13009.0
    assert counts["LC"] == 7.0
    for value in counts.values():
        assert value == round(value)
        assert value > 0


def test_source_and_target_recovery_agree_for_the_same_regions(ccf):
    strength = connectome.load_fixture_connectome(
        "knox2019_connection_strength", ccf
    )
    density = connectome.load_fixture_connectome(
        "knox2019_connection_density", ccf
    )
    normalized = connectome.load_fixture_connectome(
        "knox2019_normalized_connection_density", ccf
    )
    # |Y| from strength/density; |X| from density/normalized density. Two
    # different pairs of published files, one answer.
    targets = volumes.target_voxel_counts(strength, density)
    sources_ = volumes.source_voxel_counts(density, normalized)
    for structure_id, count in sources_.items():
        assert targets[("ipsi", structure_id)] == count


def test_pairing_two_matrices_that_are_not_one_normalization_apart_raises(ccf):
    strength = connectome.load_fixture_connectome(
        "knox2019_connection_strength", ccf
    )
    normalized = connectome.load_fixture_connectome(
        "knox2019_normalized_connection_density", ccf
    )
    # strength / normalized_density = |X_i| * |Y_j|, which is constant along
    # NEITHER axis, so this must be refused rather than silently averaged.
    with pytest.raises(ValueError, match="not constant"):
        volumes.target_voxel_counts(strength, normalized)


def test_mismatched_matrices_are_refused(ccf):
    strength = connectome.load_fixture_connectome(
        "knox2019_connection_strength", ccf
    )
    density = connectome.load_fixture_connectome(
        "knox2019_connection_density", ccf
    )
    with pytest.raises(ValueError, match="different shapes"):
        volumes.target_voxel_counts(strength, density.hemisphere("ipsi"))


# ---------------------------------------------------------------------------
# Parcellations
# ---------------------------------------------------------------------------


def test_the_three_resolutions_are_named_and_ordered_coarsest_last():
    assert build.RESOLUTIONS == ("regions", "subdivisions", "major_divisions")


def test_parcellation_maps_every_region_at_every_resolution(ccf, inputs):
    strength, _ = inputs
    ids = [int(i) for i in strength.source_ids]
    for resolution in build.RESOLUTIONS:
        coarse_ids, mapping = build.parcellation(ccf, ids, resolution)
        assert set(mapping) == set(ids)
        assert set(mapping.values()) == set(coarse_ids)
        assert len(coarse_ids) == len(set(coarse_ids))


def test_parcellation_uses_the_ccf_hierarchy_not_a_hand_written_list(ccf, inputs):
    strength, _ = inputs
    ids = [int(i) for i in strength.source_ids]
    _, subdivision = build.parcellation(ccf, ids, "subdivisions")
    _, division = build.parcellation(ccf, ids, "major_divisions")
    name = lambda mapping, acronym: ccf[mapping[ccf.id_for_acronym(acronym)]].acronym

    assert name(subdivision, "MOp") == "MO"
    assert name(subdivision, "GPe") == "PALd"
    assert name(subdivision, "GPi") == "PALd"  # two regions, one node
    assert name(subdivision, "PL") == "PL"  # already a division child
    assert name(division, "MOp") == "Isocortex"
    assert name(division, "CP") == "STR"
    assert name(division, "STN") == "HY"  # the CCF's placement, not ours


def test_an_unknown_resolution_raises(ccf):
    with pytest.raises(ValueError, match="resolution must be one of"):
        build.parcellation(ccf, [985], "voxels")


# ---------------------------------------------------------------------------
# The network builds, at all three resolutions
# ---------------------------------------------------------------------------


def test_network_builds_at_all_three_resolutions(inputs):
    expected = {"regions": 15, "subdivisions": 12, "major_divisions": 8}
    for resolution in build.RESOLUTIONS:
        circuit = built(inputs, resolution=resolution)
        network = circuit.network
        assert isinstance(network, CircuitNetwork)
        assert network.n == expected[resolution]
        assert network.connectivity.shape == (network.n, network.n)
        assert np.all(np.isfinite(network.connectivity))
        assert [node.name for node in network.nodes] == circuit.acronyms
        assert len(set(circuit.acronyms)) == network.n


def test_nodes_are_named_by_ccf_acronym_in_canonical_order(inputs):
    circuit = built(inputs, resolution="major_divisions")
    # Eight of the twelve divisions are in the fixture, in CCF order.
    assert circuit.acronyms == [
        "Isocortex", "CTXsp", "STR", "PAL", "TH", "HY", "MB", "P"
    ]


def test_built_network_runs_through_the_existing_rhs(inputs):
    circuit = built(inputs, resolution="major_divisions", weight_scale=0.05)
    network = circuit.network
    rates = np.full(network.n, 0.1)
    derivative = network.rhs(rates, network.baseline_drive())
    assert derivative.shape == (network.n,)
    assert np.all(np.isfinite(derivative))


def test_region_resolution_reproduces_the_published_connection_density(inputs, ccf):
    """The builder's arithmetic, checked against Allen's own published file.

    At `resolution='regions'` with the diagonal kept, a signed connection
    density is exactly the published `connection_density` matrix with each
    source's rows multiplied by its sign. Any error in the voxel-count
    recovery, the aggregation or the transpose shows up here.
    """
    circuit = built(inputs, resolution="regions", self_connections="keep")
    published = connectome.load_fixture_connectome(
        "knox2019_connection_density", ccf
    )
    ids, target_major = published.square_ipsilateral()
    order = [list(ids).index(i) for i in circuit.region_ids]
    expected = target_major[np.ix_(order, order)]
    signs = np.array(
        [0.0 if a.sign is None else a.sign for a in circuit.report.fine_assignments]
    )
    assert np.array_equal(circuit.connectivity, expected * signs[None, :])


def test_orientation_and_sign_are_both_right_on_a_known_pair(inputs):
    """MOp -> CP is excitatory and dense; CP -> MOp is inhibitory and weak.

    Asserting the two directions separately, with their published
    magnitudes, is what a transposed load cannot survive: the two numbers
    differ by an order of magnitude AND by sign.
    """
    circuit = built(inputs, resolution="regions")
    corticostriatal = circuit.weight("MOp", "CP")
    striatocortical = circuit.weight("CP", "MOp")
    assert corticostriatal == pytest.approx(3.059432191758332)
    assert striatocortical == pytest.approx(-0.2455127775174401)
    # and `weight(source, target)` really is connectivity[target, source]
    i = circuit.index_of("CP")
    j = circuit.index_of("MOp")
    assert circuit.connectivity[i, j] == corticostriatal
    assert circuit.connectivity[j, i] == striatocortical


def test_the_indirect_pathway_comes_out_with_textbook_polarity(inputs):
    """Nothing here was tuned; the signs are the rule table meeting the data.

    This is the check that says the sign rule is doing anatomy rather than
    bookkeeping: the D2 indirect pathway this codebase already models by
    hand falls out of the connectome with the right polarity at every step,
    and so does the GABA-A amygdala circuit's BLA -> CeA -> PAG.
    """
    circuit = built(inputs, resolution="regions")
    assert circuit.weight("MOp", "CP") > 0  # corticostriatal, glutamatergic
    assert circuit.weight("CP", "GPe") < 0  # striatopallidal, GABAergic
    assert circuit.weight("GPe", "STN") < 0  # pallidosubthalamic, GABAergic
    assert circuit.weight("STN", "GPi") > 0  # subthalamic, glutamatergic
    assert circuit.weight("GPi", "MD") < 0  # pallidothalamic, GABAergic
    assert circuit.weight("SNr", "MD") < 0  # nigrothalamic, GABAergic
    assert circuit.weight("MD", "MOp") > 0  # thalamocortical, glutamatergic
    assert circuit.weight("BLA", "CEA") > 0  # pallial amygdala, glutamatergic
    assert circuit.weight("CEA", "PAG") < 0  # central amygdala, GABAergic


def test_sign_is_a_property_of_the_source_column(inputs):
    circuit = built(inputs, resolution="regions")
    W = circuit.connectivity
    for column, assignment in enumerate(circuit.report.fine_assignments):
        if assignment.sign == 1.0:
            assert np.all(W[:, column] >= 0.0), assignment.acronym
            assert np.any(W[:, column] > 0.0), assignment.acronym
        elif assignment.sign == -1.0:
            assert np.all(W[:, column] <= 0.0), assignment.acronym
            assert np.any(W[:, column] < 0.0), assignment.acronym
        else:
            assert np.all(W[:, column] == 0.0), assignment.acronym


def test_aggregation_is_a_signed_sum_over_a_summed_target_volume(inputs, ccf):
    """Recompute one coarse entry by hand from the fixture's own numbers."""
    strength, voxel_counts = inputs
    circuit = built(inputs, resolution="major_divisions")

    ids = [int(i) for i in strength.source_ids]
    ipsi = strength.hemisphere("ipsi")
    column_of = {int(i): k for k, i in enumerate(ipsi.target_ids)}
    rules_ = nt.SignRules(ccf)

    sources_in_str = [i for i in ids if ccf.major_division(i) == ccf.id_for_acronym("STR")]
    targets_in_pal = [i for i in ids if ccf.major_division(i) == ccf.id_for_acronym("PAL")]
    total = 0.0
    for source_id in sources_in_str:
        sign = rules_.assign(source_id).sign
        row = ids.index(source_id)
        for target_id in targets_in_pal:
            if target_id == source_id:
                continue  # self_connections='drop_within_region'
            total += sign * ipsi.weights[row, column_of[target_id]]
    volume = sum(voxel_counts[("ipsi", i)] for i in targets_in_pal)

    assert circuit.weight("STR", "PAL") == pytest.approx(total / volume)
    assert circuit.weight("STR", "PAL") < 0.0  # striatopallidal is inhibitory


def test_self_connection_policies_do_what_they_say(inputs):
    kept = built(inputs, resolution="regions", self_connections="keep")
    dropped = built(inputs, resolution="regions")
    signed = [
        k for k, a in enumerate(kept.report.fine_assignments) if a.is_signed
    ]
    assert np.all(np.diag(kept.connectivity)[signed] != 0.0)
    assert np.all(np.diag(dropped.connectivity) == 0.0)

    # At a coarse resolution the surviving diagonal is inter-areal
    # connectivity (MOp <-> PL inside Isocortex), not local recurrence.
    coarse = built(inputs, resolution="major_divisions")
    assert coarse.weight("Isocortex", "Isocortex") > 0.0
    flat = built(inputs, resolution="major_divisions", self_connections="drop")
    assert flat.weight("Isocortex", "Isocortex") == 0.0
    assert flat.weight("Isocortex", "STR") == coarse.weight("Isocortex", "STR")


def test_weight_scale_is_the_only_rescaling_and_it_is_linear(inputs):
    plain = built(inputs, resolution="subdivisions")
    scaled = built(inputs, resolution="subdivisions", weight_scale=0.01)
    assert np.allclose(scaled.connectivity, 0.01 * plain.connectivity)
    assert plain.report.weight_scale == 1.0


def test_unsigned_sources_can_be_made_fatal_but_never_signed(inputs):
    with pytest.raises(ValueError, match="have no sign"):
        built(inputs, resolution="regions", unsigned_sources="raise")
    with pytest.raises(ValueError, match="unsigned_sources must be one of"):
        built(inputs, resolution="regions", unsigned_sources="excitatory")


def test_illustrative_node_parameters_are_labelled_and_overridable(inputs):
    circuit = built(inputs, resolution="major_divisions")
    assert circuit.report.node_parameters == build.DEFAULT_NODE
    assert "ILLUSTRATIVE" in circuit.report.summary()
    for node in circuit.network.nodes:
        assert node.tau == build.DEFAULT_NODE["tau"]
        assert node.gain == build.DEFAULT_NODE["gain"]
        assert node.baseline_input == build.DEFAULT_NODE["baseline_input"]
    custom = built(
        inputs, resolution="major_divisions", node_parameters={"tau": 2.5}
    )
    assert all(node.tau == 2.5 for node in custom.network.nodes)
    assert all(node.gain == 4.0 for node in custom.network.nodes)


# ---------------------------------------------------------------------------
# The refusals: things the builder will not do quietly
# ---------------------------------------------------------------------------


def test_aggregating_without_voxel_counts_is_refused(inputs):
    strength, _ = inputs
    with pytest.raises(ValueError, match="target voxel counts"):
        connectome.build_circuit_network(strength, resolution="subdivisions")


def test_aggregating_an_already_normalized_matrix_is_refused(ccf):
    normalized = connectome.load_fixture_connectome(
        "knox2019_normalized_connection_density", ccf
    )
    with pytest.raises(ValueError, match="cannot be aggregated"):
        connectome.build_circuit_network(
            normalized, resolution="major_divisions"
        )


def test_dividing_an_already_normalized_matrix_by_voxels_is_refused(ccf, inputs):
    """The failure that would otherwise be silent and mislabelled.

    Handing voxel counts with a matrix that already divided out a region
    size produces a number with no interpretation, and the report would
    then call it a connection density.
    """
    _, voxel_counts = inputs
    normalized = connectome.load_fixture_connectome(
        "knox2019_normalized_connection_density", ccf
    )
    with pytest.raises(ValueError, match="already normalized"):
        connectome.build_circuit_network(
            normalized, voxel_counts=voxel_counts, resolution="regions"
        )


def test_a_connectome_without_an_ontology_is_refused():
    from neurocascade.legacy.connectome import matrix as matrix_module

    bare = matrix_module.read_connectome(
        "knox2019_connection_strength",
        connectome.fixtures.KNOX_FIXTURE_PATHS["knox2019_connection_strength"],
    )
    with pytest.raises(ValueError, match="without an ontology"):
        connectome.build_circuit_network(bare)


def test_the_second_connectome_builds_at_its_own_resolution(ccf):
    """Oh et al. 2014 is usable, in its own units, and says so.

    Oh's matrix is a normalized connection STRENGTH: already divided by the
    source region's size, so it cannot be aggregated (no voxel counts are
    recoverable from a single file), but it can still carry a
    native-resolution network for P3.3's "does this survive changing the
    connectome" comparison. The units string must not claim it is a
    density.
    """
    oh = connectome.load_fixture_connectome("oh2014_supplementary_table_3", ccf)
    circuit = connectome.build_circuit_network(oh, resolution="regions")
    assert circuit.n == 14  # LC is absent from Oh et al. 2014
    assert "LC" not in circuit.acronyms
    assert "NOT a connection density" in circuit.report.units
    assert circuit.weight("MOp", "CP") > 0.0
    assert circuit.weight("CP", "GPe") < 0.0
    # Oh's homogeneous model gives CP -> MOp exactly zero where Knox's
    # voxel model gives it a small weight: a real difference between the
    # two fits, and the reason P3.3 gets to ask which results depend on it.
    assert circuit.weight("CP", "MOp") == 0.0
    with pytest.raises(ValueError, match="cannot be aggregated"):
        connectome.build_circuit_network(oh, resolution="major_divisions")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_accounts_for_all_of_the_weight(inputs):
    circuit = built(inputs, resolution="regions")
    report = circuit.report
    assert report.signed_weight_fraction + report.excluded_weight_fraction == (
        pytest.approx(1.0)
    )
    assert sum(report.excluded_by_transmitter.values()) == pytest.approx(
        report.excluded_weight_fraction
    )
    assert len(report.signed_fine_regions()) + len(report.unsigned_fine_regions()) == 15
    assert report.citations()  # the sign of every edge is attributable


def test_a_coarse_node_can_have_edges_its_own_structure_has_no_rule_for(inputs):
    """`node_assignments` is informational; edges were signed per fine region.

    MB has no rule of its own, but SNr inside it does, so the midbrain node
    has a real (inhibitory) outgoing column. If sign had been assigned after
    aggregation instead of before, that column would be empty.
    """
    circuit = built(inputs, resolution="major_divisions")
    midbrain = [a for a in circuit.report.node_assignments if a.acronym == "MB"][0]
    assert midbrain.rule is None
    column = circuit.connectivity[:, circuit.index_of("MB")]
    assert np.any(column != 0.0)
    assert circuit.weight("MB", "TH") < 0.0  # SNr and SNr alone, in this slice


def test_report_summary_states_the_things_that_get_over_read(inputs):
    text = built(inputs, resolution="subdivisions").report.summary()
    assert "knox2019_connection_strength" in text
    assert "subdivisions" in text
    assert "connection density" in text
    assert "weight_scale=1" in text
    assert "signed" in text


# ---------------------------------------------------------------------------
# The API this task promised not to move
# ---------------------------------------------------------------------------


def test_circuit_network_api_is_unchanged():
    """P3.2 must not move `CircuitNetwork`; external importers depend on it.

    The builder constructs the existing class and adds nothing to it, so
    this pins the two constructor signatures and the public attributes an
    outside caller (soma-twin) reaches for.
    """
    assert list(inspect.signature(CircuitNetwork.__init__).parameters) == [
        "self",
        "nodes",
        "connectivity",
    ]
    assert list(inspect.signature(CircuitNode.__init__).parameters) == [
        "self",
        "name",
        "tau",
        "baseline_input",
        "gain",
        "threshold",
    ]
    assert list(inspect.signature(CircuitNetwork.rhs).parameters) == [
        "self",
        "rates",
        "drive",
        "connectivity",
    ]
    network = CircuitNetwork(
        nodes=[CircuitNode(name="a", tau=1.0)], connectivity=[[0.0]]
    )
    assert network.n == 1
    assert network.index == {"a": 0}
    assert network.baseline_drive().tolist() == [0.0]


def test_the_builder_returns_a_plain_circuit_network(inputs):
    circuit = built(inputs, resolution="major_divisions")
    assert type(circuit.network) is CircuitNetwork
    assert all(type(node) is CircuitNode for node in circuit.network.nodes)


# ---------------------------------------------------------------------------
# The full published matrices, when the gitignored cache is present
# ---------------------------------------------------------------------------

_FULL_DATA = (
    fetch.is_cached("knox2019_connection_strength")
    and fetch.is_cached("knox2019_connection_density")
    and fetch.is_cached(sources.ONTOLOGY_KEY)
)

full_data_only = pytest.mark.skipif(
    not _FULL_DATA,
    reason="needs the gitignored data/connectome cache; "
    "run scripts/fetch_connectome.py",
)


@pytest.fixture(scope="module")
def full_inputs():
    if not _FULL_DATA:
        pytest.skip("no cache")
    return connectome.load_knox_inputs()


@full_data_only
def test_full_network_builds_at_all_three_resolutions(full_inputs):
    strength, voxel_counts = full_inputs
    sizes = {}
    for resolution in build.RESOLUTIONS:
        circuit = connectome.build_circuit_network(
            strength, voxel_counts=voxel_counts, resolution=resolution
        )
        sizes[resolution] = circuit.n
        assert np.all(np.isfinite(circuit.connectivity))
    # 291 published regions, the 60-wide CCF level below the divisions, and
    # the twelve divisions Knox et al. published as their own mask.
    assert sizes == {"regions": 291, "subdivisions": 60, "major_divisions": 12}


@full_data_only
def test_the_roadmaps_43_is_the_isocortex_not_a_resolution(full_inputs):
    strength, _ = full_inputs
    ontology = strength.ontology
    isocortex = ontology.id_for_acronym("Isocortex")
    regions = [int(i) for i in strength.source_ids]
    assert sum(ontology.is_descendant_of(i, isocortex) for i in regions) == 43


@full_data_only
def test_sign_coverage_over_the_real_brain_is_what_the_docs_claim(full_inputs):
    strength, voxel_counts = full_inputs
    circuit = connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, resolution="regions"
    )
    report = circuit.report
    assert report.n_fine_regions == 291
    assert len(report.signed_fine_regions()) == 151
    assert report.signed_weight_fraction == pytest.approx(0.909, abs=5e-4)
    counts = report.transmitter_counts()
    assert counts[nt.GLUTAMATERGIC] == 116
    assert counts[nt.GABAERGIC] == 35
    assert counts[None] == 131

    # The documented hole: no pons or medulla region has a sign, so at
    # division resolution two nodes have no outgoing edges at all.
    coarse = connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, resolution="major_divisions"
    )
    silent = [
        acronym
        for column, acronym in enumerate(coarse.acronyms)
        if not np.any(coarse.connectivity[:, column])
    ]
    assert silent == ["P", "MY"]


@full_data_only
def test_full_region_build_reproduces_the_published_density(full_inputs):
    strength, voxel_counts = full_inputs
    circuit = connectome.build_circuit_network(
        strength,
        voxel_counts=voxel_counts,
        resolution="regions",
        self_connections="keep",
    )
    published = connectome.load_connectome(
        key="knox2019_connection_density", ontology=strength.ontology
    )
    ids, target_major = published.square_ipsilateral()
    order = [list(ids).index(i) for i in circuit.region_ids]
    expected = target_major[np.ix_(order, order)]
    signs = np.array(
        [0.0 if a.sign is None else a.sign for a in circuit.report.fine_assignments]
    )
    assert np.array_equal(circuit.connectivity, expected * signs[None, :])


@full_data_only
def test_raw_weights_saturate_the_illustrative_sigmoid_at_scale_one(full_inputs):
    """The warning in `build.py`'s docstring, as an assertion.

    `weight_scale=1.0` is the published number, not a usable coupling: the
    per-node signed input weight is order +10, which with `gain=4.0` pins
    every node at the top of its sigmoid. P3.4 is the task that fixes this;
    until then nothing should be able to claim these defaults simulate.
    """
    strength, voxel_counts = full_inputs
    circuit = connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, resolution="regions"
    )
    row_sums = circuit.connectivity.sum(axis=1)
    assert row_sums.min() == pytest.approx(-17.6, abs=0.5)
    assert np.median(row_sums) == pytest.approx(7.4, abs=0.5)
    assert row_sums.max() == pytest.approx(32.5, abs=0.5)

    rates = np.full(circuit.n, 0.5)
    drive = circuit.network.baseline_drive()
    net_input = circuit.connectivity @ rates + drive
    saturated = np.mean(np.abs(net_input) * build.DEFAULT_NODE["gain"] > 5.0)
    assert saturated > 0.5
