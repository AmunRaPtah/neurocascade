"""P7.1: the behavioural endpoint library.

Everything here runs offline off the committed connectome fixtures (real
published data, fifteen regions of it) plus the fixture ontology, except the
handful at the bottom that need the gitignored 291-region cache and skip
without it.

Three of these tests are the load-bearing ones:

* `test_every_weight_is_labelled_illustrative` is the acceptance criterion
  the whole task turns on, and it checks the label on the endpoint AND that
  no term carries a magnitude of its own.
* `test_every_term_cites_a_real_entry_and_states_its_finding` is what stops
  a region being added to a readout because it felt right.
* `test_signs_match_the_recorded_finding_direction` re-derives each term's
  sign from the machine-readable direction fields of the record, so a term
  whose prose says "lesion reduced the measure" cannot carry a negative
  sign.
"""

import re

import pytest

from neurocascade import CircuitNetwork, endpoints
from neurocascade.legacy import connectome
from neurocascade.behavior import BehaviorReadout
from neurocascade.circuit import CircuitNode
from neurocascade.legacy.connectome import build, fetch, sources


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ccf():
    return connectome.load_fixture_ontology()


@pytest.fixture(scope="module")
def fixture_inputs(ccf):
    return connectome.load_fixture_build_inputs(ontology=ccf)


@pytest.fixture(scope="module")
def fixture_network(ccf, fixture_inputs):
    strength, voxel_counts = fixture_inputs
    return connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, resolution="regions"
    ).network


def flat_rates(network, value=0.5):
    return [value] * len(network.nodes)


# ---------------------------------------------------------------------------
# The acceptance criteria
# ---------------------------------------------------------------------------


def test_there_are_seven_endpoints_and_they_are_the_roadmaps_seven():
    assert len(endpoints.ENDPOINTS) == 7
    assert set(endpoints.ENDPOINTS) == set(endpoints.ENDPOINT_ORDER)
    assert set(endpoints.ENDPOINT_ORDER) == {
        "locomotion",
        "anxiety",
        "analgesia",
        "reward",
        "sedation",
        "working_memory",
        "sensorimotor_gating",
    }


def test_every_endpoint_is_a_multi_region_readout():
    """P7.1's own words: 'each of them a multi-region readout'."""
    for key in endpoints.ENDPOINT_ORDER:
        endpoint = endpoints.ENDPOINTS[key]
        # The whole library, and the part that survives excluding the
        # contested terms, both have to be multi-region. The second is the
        # one that matters, because it is what `resolve` builds by default.
        assert len(endpoint.regions()) >= 4, key
        active_regions = {
            acronym
            for term in endpoint.active_terms()
            for acronym in term.structures
        }
        assert len(active_regions) >= 3, (key, sorted(active_regions))


def test_every_weight_is_labelled_illustrative():
    """The acceptance criterion this whole task turns on.

    Two halves: the endpoint declares its weights illustrative, and no
    term is able to carry a magnitude of its own that could quietly
    become a fitted number. `EndpointTerm` has no magnitude attribute at
    all, which is the strongest form of that guarantee.
    """
    allowed = {
        "structures", "activity", "measure", "sign", "citation", "species",
        "technique", "assays", "finding", "contested", "note", "inferred",
    }
    for key, endpoint in endpoints.ENDPOINTS.items():
        assert endpoint.weights_are_illustrative is True, key
        assert endpoint.weight_provenance == endpoints.ILLUSTRATIVE_WEIGHTS
        for term in endpoint.terms:
            # An allowlist rather than "not hasattr(term, 'magnitude')":
            # the point is that there is NOWHERE for a fitted number to
            # live on a term, not that two particular names are absent.
            assert set(vars(term)) == allowed, (key, set(vars(term)) ^ allowed)
            assert term.sign in (1, -1)
    for key, variant in endpoints.ASSAY_VARIANTS.items():
        assert variant.weights_are_illustrative is True, key


def test_every_term_cites_a_real_entry_and_states_its_finding():
    """What stops a region being added because it felt right."""
    for key, endpoint in endpoints.ENDPOINTS.items():
        for term in endpoint.terms:
            assert term.citation in endpoints.CITATIONS, (key, term.citation)
            assert len(term.finding) > 60, (key, term.citation)
            assert term.species in ("mouse", "rat"), (key, term.species)
            assert term.technique in endpoints.EVIDENCE_TECHNIQUES
            assert term.assays, (key, term.citation)


#: Every term's expected sign, read by hand out of its `finding` prose and
#: its endpoint's `direction` string, and written down HERE rather than
#: computed from the module. This is the point: `sign` is derived inside
#: `EndpointTerm` from `activity` and `measure`, so re-deriving it with the
#: same formula proves nothing at all. A second, independent transcription
#: is the only thing that can catch a term whose fields were filled in
#: backwards, and a mutation of any one entry below fails the test.
#:
#: Keyed (endpoint, "+".join(structures)). Read each one as:
#:   locomotion SNr  -1  "activating them biased transitions toward LESS
#:                        motor activity" and the score is more locomotion
#:   analgesia  ACAd -1  "lesion INCREASED hot-plate latency" and the score
#:                        is longer latency, so lesion raised the score and
#:                        the region's activity lowers it
EXPECTED_SIGNS = {
    ("locomotion", "SNr"): -1,
    ("locomotion", "CUN"): +1,
    ("locomotion", "PPN"): +1,
    ("locomotion", "SNc"): +1,
    ("locomotion", "VTA"): +1,
    ("locomotion", "LHA"): +1,
    ("locomotion", "ACB"): +1,
    ("locomotion", "PGRNl"): +1,
    ("locomotion", "CP"): +1,
    ("locomotion", "GPe"): +1,
    ("locomotion", "STN"): -1,

    ("anxiety", "CA1"): +1,
    ("anxiety", "SUB"): +1,
    ("anxiety", "LSr"): +1,
    ("anxiety", "LC"): +1,
    ("anxiety", "VTA"): +1,
    ("anxiety", "BMA"): -1,
    ("anxiety", "BLA"): +1,
    ("anxiety", "BST"): +1,
    ("anxiety", "CEA"): +1,
    ("anxiety", "PL"): +1,
    ("anxiety", "ILA"): +1,
    ("anxiety", "ZI"): +1,

    ("analgesia", "PAG"): +1,
    ("analgesia", "LC"): +1,
    ("analgesia", "DR"): +1,
    ("analgesia", "ZI"): +1,
    ("analgesia", "SMT"): +1,
    ("analgesia", "CEA"): +1,
    ("analgesia", "GRN"): -1,
    ("analgesia", "PB"): -1,
    ("analgesia", "ACAd"): -1,
    ("analgesia", "MD"): -1,
    ("analgesia", "RM"): +1,

    ("reward", "VTA"): +1,
    ("reward", "ACB"): +1,
    ("reward", "LH"): -1,
    ("reward", "SI+MA"): +1,
    ("reward", "DR"): +1,
    ("reward", "BLA"): +1,
    ("reward", "PL"): +1,
    ("reward", "ILA"): -1,
    ("reward", "LHA"): +1,
    ("reward", "CA1+CA3+DG"): +1,
    ("reward", "IPN"): -1,

    ("sedation", "LC"): -1,
    ("sedation", "LHA"): -1,
    ("sedation", "TMd+TMv"): -1,
    ("sedation", "NDB+SI+MA"): -1,
    ("sedation", "PPN"): -1,
    ("sedation", "PB"): -1,
    ("sedation", "VTA"): -1,
    ("sedation", "VLPO"): +1,
    ("sedation", "DR"): +1,

    ("working_memory", "PL+ILA"): +1,
    ("working_memory", "MD"): +1,
    ("working_memory", "RE"): +1,
    ("working_memory", "CA1"): +1,
    ("working_memory", "CA3"): +1,
    ("working_memory", "MS"): +1,

    ("sensorimotor_gating", "IC"): +1,
    ("sensorimotor_gating", "SCm"): +1,
    ("sensorimotor_gating", "SI+MA"): -1,
    ("sensorimotor_gating", "BLA"): +1,
    ("sensorimotor_gating", "PRNc"): -1,
    ("sensorimotor_gating", "PPN"): +1,
    ("sensorimotor_gating", "PL+ILA"): -1,
    ("sensorimotor_gating", "CA1+SUB"): -1,
    ("sensorimotor_gating", "ACB"): -1,
}


def test_every_sign_matches_an_independent_hand_transcription():
    """The test that actually pins the signs.

    Re-deriving `sign` from `activity` and `measure` would re-run the
    constructor's own formula and could not fail. `EXPECTED_SIGNS` above is
    a second transcription, made by reading each term's `finding` against
    its endpoint's `direction`, so flipping any single term in the module
    fails here.
    """
    actual = {
        (key, "+".join(term.structures)): term.sign
        for key, endpoint in endpoints.ENDPOINTS.items()
        for term in endpoint.terms
    }
    assert actual == EXPECTED_SIGNS


def test_the_sign_table_covers_every_term_and_nothing_else():
    """Stops the table above from silently falling behind the module."""
    keys = [
        (key, "+".join(term.structures))
        for key, endpoint in endpoints.ENDPOINTS.items()
        for term in endpoint.terms
    ]
    assert len(keys) == len(set(keys)) == len(EXPECTED_SIGNS) == 69


def test_sign_is_derived_and_cannot_be_passed_in():
    """`EndpointTerm` computes `sign`; there is no way to hand it one."""
    for key, endpoint in endpoints.ENDPOINTS.items():
        for term in endpoint.terms:
            assert term.sign == (1 if term.activity == term.measure else -1)
    with pytest.raises(TypeError):
        endpoints.EndpointTerm(
            "CP", activity="increase", measure="increase", sign=-1,
            citation="Kravitz2010", species="mouse", technique="lesion",
            assay="open field", finding="x" * 80,
        )


def test_every_citation_is_used_and_every_used_citation_exists():
    """No orphan citations in either direction.

    An unused citation is either a region that was dropped without the
    drop being recorded, or dead prose. Both are worth catching. The
    exceptions are the ones deliberately cited from a term's NOTE or from
    SEARCHED_AND_EMPTY rather than from a term's `citation` field, which
    is where a contradicting or bounding paper belongs.
    """
    used = {
        term.citation
        for endpoint in endpoints.ENDPOINTS.values()
        for term in endpoint.terms
    }
    assert used <= set(endpoints.CITATIONS)

    prose = "\n".join(
        [endpoints.SEARCHED_AND_EMPTY]
        + [term.note or "" for e in endpoints.ENDPOINTS.values()
           for term in e.terms]
        + [term.inferred or "" for e in endpoints.ENDPOINTS.values()
           for term in e.terms]
        + [e.notes or "" for e in endpoints.ENDPOINTS.values()]
        + [endpoints.__doc__]
    )
    for key in endpoints.CITATIONS:
        assert key in used or ("[%s]" % key) in prose, key

    # The direction the first version of this test forgot, and the one that
    # matters more: a [Bracketed2019] key in the prose that resolves to
    # nothing is a citation the reader cannot follow. Five of these existed
    # and were added to CITATIONS rather than deleted from the prose,
    # because each was bounding or contesting a sign.
    cited_in_prose = set(re.findall(r"\[([A-Z][A-Za-z0-9]+)\]", prose))
    unresolvable = sorted(cited_in_prose - set(endpoints.CITATIONS))
    assert unresolvable == [], unresolvable


def test_contested_terms_are_excluded_by_default_and_available_on_request():
    contested = [
        (key, term)
        for key, endpoint in endpoints.ENDPOINTS.items()
        for term in endpoint.terms
        if term.contested
    ]
    # Not a vacuous test: this literature really does disagree with
    # itself, and if it stopped being recorded that would be the bug.
    assert len(contested) >= 10
    for key, term in contested:
        endpoint = endpoints.ENDPOINTS[key]
        assert term not in endpoint.active_terms()
        assert term in endpoint.active_terms(include_contested=True)
        # A contested term must say what contests it.
        assert term.note and "[" in term.note, (key, term.citation)


def test_the_species_of_every_term_is_the_papers_own():
    """Rule 2: no term's species was changed to match the model.

    The model is mouse. This asserts the module still contains rat terms,
    because a library in which every species had quietly become mouse is
    exactly the failure the rule exists to prevent.
    """
    species = [
        term.species
        for endpoint in endpoints.ENDPOINTS.values()
        for term in endpoint.terms
    ]
    assert "rat" in species and "mouse" in species
    # Analgesia in particular is rat-dominated, which the module says.
    analgesia = [t.species for t in endpoints.ENDPOINTS["analgesia"].terms]
    assert analgesia.count("rat") > analgesia.count("mouse")


# ---------------------------------------------------------------------------
# The assay dissociation
# ---------------------------------------------------------------------------


def test_assay_variants_are_views_of_analgesia_and_not_an_eighth_endpoint():
    analgesia = endpoints.ENDPOINTS["analgesia"]
    for key, variant in endpoints.ASSAY_VARIANTS.items():
        assert key not in endpoints.ENDPOINTS
        assert len(variant.regions()) >= 2
        # Subset AND complete: a `restrict_to_assay` that dropped half the
        # matching terms would pass a subset check.
        needle = variant.assay
        expected = [t for t in analgesia.terms if t.ran_assay(needle)]
        assert list(variant.terms) == expected, key
        assert expected


def test_the_two_analgesia_assays_select_different_regions():
    """The dissociation six papers report, made a property of the object.

    If tail-flick and hot-plate selected the same regions, the split would
    be decoration. They do not: only the supraspinal assay reaches the
    cingulate, and only the spinal one reaches the submedial thalamus.
    """
    tf = set(endpoints.ASSAY_VARIANTS["analgesia_tail_flick"].regions())
    hp = set(endpoints.ASSAY_VARIANTS["analgesia_hot_plate"].regions())
    assert tf and hp
    assert tf != hp
    assert "ACAd" in hp and "ACAd" not in tf
    assert "SMT" in tf and "SMT" not in hp


def test_restrict_to_assay_matches_tokens_exactly_not_as_a_substring():
    """The bug this field's structure exists to prevent.

    Several terms' notes and findings say a paper explicitly did NOT
    change tail-flick latency. Under substring matching those terms would
    join the tail-flick variant and invert its meaning.
    """
    analgesia = endpoints.ENDPOINTS["analgesia"]
    acad = [t for t in analgesia.terms if t.structures == ("ACAd",)][0]
    assert "tail" in acad.finding.lower()
    assert not acad.ran_assay("tail flick")
    assert acad.ran_assay("hot plate")
    assert acad.ran_assay("HOT PLATE")


# ---------------------------------------------------------------------------
# Resolution onto a network
# ---------------------------------------------------------------------------


def test_resolution_onto_the_fixture_network(ccf, fixture_network):
    """Fifteen real regions: some terms land, some cannot, all reported."""
    resolved = endpoints.ENDPOINTS["locomotion"].resolve(fixture_network, ccf)
    nodes = {node.name for node in fixture_network.nodes}
    assert set(resolved.weights) <= nodes
    # SNr, VTA and ACB are in the fixture; CUN, PGRNl and LHA are not.
    assert {"SNr", "VTA", "ACB"} <= set(resolved.weights)
    assert "CUN" in resolved.unmapped
    assert resolved.weights["SNr"] < 0 < resolved.weights["VTA"]
    assert "unmapped" in resolved.report()


def test_weights_are_normalised_to_unit_absolute_sum(ccf, fixture_network):
    for key in endpoints.ENDPOINT_ORDER:
        resolved = endpoints.ENDPOINTS[key].resolve(fixture_network, ccf)
        # Not `if not weights: continue` - an endpoint that resolved onto
        # nothing would make this test pass by saying nothing.
        assert resolved.weights, key
        total = sum(abs(w) for w in resolved.weights.values())
        assert total == pytest.approx(1.0), key


def test_normalisation_can_be_turned_off(ccf, fixture_network):
    resolved = endpoints.ENDPOINTS["locomotion"].resolve(
        fixture_network, ccf, normalize=False
    )
    assert all(abs(w) == pytest.approx(1.0) for w in resolved.weights.values())


def test_the_readout_is_a_plain_behavior_readout(ccf, fixture_network):
    """P7.1 adds a library, not a new decoder class.

    Everything downstream that already scores a `BehaviorReadout` keeps
    working, which is the same promise P3.2 made about `CircuitNetwork`.
    """
    resolved = endpoints.ENDPOINTS["reward"].resolve(fixture_network, ccf)
    assert type(resolved.readout) is BehaviorReadout
    rates = flat_rates(fixture_network)
    assert resolved.readout.score(fixture_network, rates) == pytest.approx(
        resolved.score(fixture_network, rates)
    )


def test_score_at_rest_is_half_the_signed_weight_sum(ccf, fixture_network):
    for key in endpoints.ENDPOINT_ORDER:
        resolved = endpoints.ENDPOINTS[key].resolve(fixture_network, ccf)
        rates = flat_rates(fixture_network, 0.5)
        assert resolved.score(fixture_network, rates) == pytest.approx(
            resolved.baseline_score()
        )
        assert resolved.delta(fixture_network, rates) == pytest.approx(0.0)
        # `baseline_score` takes a rate, so check a second one - otherwise
        # an implementation that hardcoded 0.5 would pass.
        assert resolved.score(
            fixture_network, flat_rates(fixture_network, 0.25)
        ) == pytest.approx(resolved.baseline_score(0.25))


def test_score_uses_each_nodes_own_rate(ccf, fixture_network):
    """A uniform rate vector cannot detect a node-index misalignment.

    Every other score test here sets all rates equal, so `BehaviorReadout`
    could read the wrong node throughout and still agree. This one gives
    every node a distinct rate and computes the expected score by hand from
    the weight dict.
    """
    resolved = endpoints.ENDPOINTS["analgesia"].resolve(fixture_network, ccf)
    rates = [0.1 + 0.05 * i for i in range(len(fixture_network.nodes))]
    expected = sum(
        weight * rates[fixture_network.index[name]]
        for name, weight in resolved.weights.items()
    )
    assert resolved.score(fixture_network, rates) == pytest.approx(expected)
    # And the hand sum is not accidentally the uniform one.
    assert expected != pytest.approx(resolved.baseline_score())


def test_delta_has_the_sign_the_endpoints_direction_string_promises(
    ccf, fixture_network
):
    """Raise a positively weighted node and the score must go up.

    This is the test that would catch a polarity inversion between the
    weights and the `direction` string, which is the single most likely
    way to get a whole endpoint backwards.
    """
    resolved = endpoints.ENDPOINTS["locomotion"].resolve(fixture_network, ccf)
    base = flat_rates(fixture_network)
    for name, weight in resolved.weights.items():
        rates = list(base)
        rates[fixture_network.index[name]] += 0.1
        delta = resolved.delta(fixture_network, rates)
        assert (delta > 0) == (weight > 0), name
    # SNr is the inhibitory basal-ganglia output: more SNr, less movement.
    rates = list(base)
    rates[fixture_network.index["SNr"]] += 0.2
    assert resolved.delta(fixture_network, rates) < 0


def test_a_term_is_one_unit_of_evidence_however_many_structures_it_names(ccf):
    """Dorsal hippocampus is CA1+CA3+DG and must not outweigh BLA 3:1."""
    reward = endpoints.ENDPOINTS["reward"]
    nodes = ["CA1", "CA3", "DG", "BLA", "VTA"]
    resolved = reward.resolve(nodes, ccf, normalize=False)
    hippocampal = sum(
        resolved.weights[n]
        for n in ("CA1", "CA3", "DG")
        if n in resolved.weights
    )
    assert hippocampal == pytest.approx(resolved.weights["BLA"])
    assert resolved.weights["CA1"] == pytest.approx(1.0 / 3.0)


def test_the_same_term_collapsed_onto_one_node_keeps_its_full_weight(ccf):
    """The other half of the previous test, at a coarse parcellation.

    Splitting a term across its structures and then collapsing them onto
    one ancestor node has to reconstitute the term, not divide it. This is
    why terms sum WITHIN themselves and only then average ACROSS each
    other.
    """
    reward = endpoints.ENDPOINTS["reward"]
    fine = reward.resolve(["CA1", "CA3", "DG", "BLA"], ccf, normalize=False)
    coarse = reward.resolve(["HPF", "BLA"], ccf, normalize=False)
    assert coarse.weights["HPF"] == pytest.approx(
        sum(fine.weights[n] for n in ("CA1", "CA3", "DG"))
    )
    assert coarse.weights["HPF"] == pytest.approx(coarse.weights["BLA"])


def test_ancestor_and_descendant_placement_rules(ccf):
    """Rules 2 and 3 of the module docstring, on real CCF structures."""
    locomotion = endpoints.ENDPOINTS["locomotion"]

    # Rule 2: no ACB node, but its ancestor STR is one.
    resolved = locomotion.resolve(["STR", "SNr"], ccf, normalize=False)
    rules = {p.structure: p.rule for p in resolved.placements}
    assert rules["ACB"] == "ancestor"
    assert rules["SNr"] == "exact"
    assert resolved.weights["STR"] == pytest.approx(1.0)

    # Rule 3: an endpoint term for a parent structure splits across the
    # descendant nodes that exist. `reward`'s hippocampal term names CA1,
    # CA3 and DG; ask for a network whose only hippocampal nodes are the
    # CA1 subfields and the term has to split again.
    parent = endpoints.EndpointTerm(
        "HPF", "increase", "increase", "Meyers2006", "rat", "inactivation",
        "conditioned place preference", "x" * 80,
    )
    probe = endpoints.Endpoint(
        key="probe", name="probe", assay="probe",
        direction="higher = more probe",
        terms=[parent, endpoints.EndpointTerm(
            "SNr", "increase", "increase", "Liu2020", "mouse", "optogenetic",
            "open field", "y" * 80)],
    )
    resolved = probe.resolve(["CA1", "CA3", "DG", "SNr"], ccf,
                             normalize=False)
    rules = {p.structure: p.rule for p in resolved.placements}
    assert rules["HPF"] == "descendants"
    assert sum(resolved.weights[n] for n in ("CA1", "CA3", "DG")) == \
        pytest.approx(1.0)
    assert resolved.equal_splits == ("HPF",)
    assert endpoints.ILLUSTRATIVE_EQUAL_SPLIT in resolved.report()

    # With sizes, the split is volume-weighted instead, and says so.
    sized = probe.resolve(
        ["CA1", "CA3", "DG", "SNr"], ccf, normalize=False,
        region_sizes={"CA1": 3.0, "CA3": 1.0, "DG": 1.0, "SNr": 1.0},
    )
    assert sized.equal_splits == ()
    assert sized.weights["CA1"] == pytest.approx(0.6)


def test_collapse_mean_and_sum_differ_only_where_terms_pile_up(ccf):
    locomotion = endpoints.ENDPOINTS["locomotion"]
    nodes = ["MB", "STR", "HY", "MY"]
    mean = locomotion.resolve(nodes, ccf, normalize=False, collapse="mean")
    total = locomotion.resolve(nodes, ccf, normalize=False, collapse="sum")
    # STR takes one term (ACB), so the two rules agree there.
    assert mean.weights["STR"] == pytest.approx(total.weights["STR"])
    # MB takes four of locomotion's terms here - SNr, PPN, SNc and VTA -
    # three positive and one negative, so sum is 2 and mean is 2/4. It is
    # four rather than five because the committed FIXTURE ontology is a
    # fifteen-region slice that does not contain CUN, LHA or PGRNl; the
    # full-cache test below runs the same endpoint over the real
    # structure graph.
    assert {p.structure for p in total.placements if p.rule == "unmapped"} \
        == {"CUN", "LHA", "PGRNl"}
    assert total.weights["MB"] == pytest.approx(2.0)
    assert mean.weights["MB"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        locomotion.resolve(nodes, ccf, collapse="median")


def test_sign_cancellation_is_reported_rather_than_hidden(ccf):
    """A coarse parcellation that cannot express an endpoint must say so."""
    locomotion = endpoints.ENDPOINTS["locomotion"]
    coarse = locomotion.resolve(["MB", "STR", "HY", "MY"], ccf)
    assert "MB" in coarse.cancellations
    assert "SIGN CANCELLATION" in coarse.report()
    # And a fine parcellation with the same terms must not.
    fine = locomotion.resolve(
        ["SNr", "CUN", "PPN", "SNc", "VTA", "LHA", "ACB", "PGRNl"], ccf
    )
    assert fine.cancellations == ()


def test_including_the_contested_terms_changes_what_is_measured(
    ccf, fixture_network
):
    """Not a formality: the contested terms are worth something.

    If including them changed nothing there would be no point recording
    them, and no point excluding them either.
    """
    anxiety = endpoints.ENDPOINTS["anxiety"]
    default = anxiety.resolve(fixture_network, ccf)
    everything = anxiety.resolve(fixture_network, ccf,
                                 include_contested=True)
    assert set(everything.weights) > set(default.weights)
    assert "BLA" not in default.weights
    assert "BLA" in everything.weights


def test_nodes_the_ontology_cannot_name_are_ignored_not_crashed_on(ccf):
    """Hand-built scenario nodes are not CCF regions and must not match."""
    hand_built = ["VTA_DA", "VTA_GABA", "PAG_Output", "NAc", "SNr"]
    resolved = endpoints.ENDPOINTS["locomotion"].resolve(hand_built, ccf)
    assert set(resolved.weights) == {"SNr"}


def test_every_summary_and_report_renders(ccf, fixture_network):
    for key in endpoints.ENDPOINT_ORDER:
        endpoint = endpoints.ENDPOINTS[key]
        text = endpoint.summary()
        assert endpoint.direction in text
        assert endpoints.ILLUSTRATIVE_WEIGHTS in text
        assert endpoint.resolve(fixture_network, ccf).report()
    for variant in endpoints.ASSAY_VARIANTS.values():
        assert variant.summary()


def test_a_term_needs_structures_a_citation_a_finding_and_an_assay():
    good = dict(
        structures="CP", activity="increase", measure="increase",
        citation="Kravitz2010", species="mouse", technique="lesion",
        assay="open field", finding="x" * 80,
    )
    endpoints.EndpointTerm(**good)
    for field, bad in (
        ("structures", ()), ("citation", ""), ("finding", ""),
        ("assay", ()), ("technique", "vibes"), ("activity", "sideways"),
        ("measure", "sideways"),
    ):
        kwargs = dict(good)
        kwargs[field] = bad
        with pytest.raises(ValueError):
            endpoints.EndpointTerm(**kwargs)


def test_an_endpoint_refuses_to_be_single_region():
    """The 'multi-region' in P7.1's brief is enforced by the constructor."""
    term = endpoints.EndpointTerm(
        "CP", "increase", "increase", "Kravitz2010", "mouse", "lesion",
        "open field", "x" * 80,
    )
    with pytest.raises(ValueError):
        endpoints.Endpoint(
            key="single", name="single", assay="a", direction="higher = more",
            terms=[term],
        )


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
def full_ontology():
    if not _FULL_DATA:
        pytest.skip("no cache")
    return connectome.load_ontology()


@pytest.fixture(scope="module")
def full_inputs(full_ontology):
    if not _FULL_DATA:
        pytest.skip("no cache")
    return connectome.load_knox_inputs(ontology=full_ontology)


@full_data_only
def test_every_acronym_used_is_a_real_ccf_structure(full_ontology):
    """No invented acronyms, checked against the P3.1 structure graph."""
    for key, endpoint in endpoints.ENDPOINTS.items():
        for acronym in endpoint.regions():
            full_ontology.id_for_acronym(acronym)


@full_data_only
def test_every_acronym_used_is_a_row_of_the_published_connectome(full_inputs):
    """Stronger than the previous test, and the one that matters.

    A CCF structure that is not a Knox source region can still be
    resolved (by ancestor), but a whole-brain readout built at
    resolution='regions' would silently lose it. Any exception has to be
    a deliberate one, named here.
    """
    strength, _ = full_inputs
    rows = set(strength.source_acronyms())
    missing = sorted(
        {
            acronym
            for endpoint in endpoints.ENDPOINTS.values()
            for acronym in endpoint.regions()
        }
        - rows
    )
    assert missing == []


@full_data_only
def test_every_endpoint_resolves_at_all_three_resolutions(
    full_ontology, full_inputs
):
    strength, voxel_counts = full_inputs
    for resolution in build.RESOLUTIONS:
        circuit = connectome.build_circuit_network(
            strength, voxel_counts=voxel_counts, resolution=resolution
        )
        for key in endpoints.ENDPOINT_ORDER:
            resolved = endpoints.ENDPOINTS[key].resolve(
                circuit.network, full_ontology
            )
            assert resolved.weights, (key, resolution)
            assert resolved.unmapped == (), (key, resolution)
            if resolution == "regions":
                # At full resolution every term keeps its own node, so
                # nothing collapses and nothing can cancel.
                assert resolved.cancellations == (), key
                assert all(
                    p.rule == "exact" for p in resolved.placements
                ), key


@full_data_only
def test_coarsening_destroys_endpoints_and_the_report_says_which(
    full_ontology, full_inputs
):
    """The number a reader should see before trusting a division-level run.

    Twelve major divisions cannot express seven endpoints that were
    defined over nuclei, and the honest thing is to measure how badly
    rather than to assert that it is fine.
    """
    strength, voxel_counts = full_inputs
    fine = connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, resolution="regions"
    ).network
    coarse = connectome.build_circuit_network(
        strength, voxel_counts=voxel_counts, resolution="major_divisions"
    ).network
    cancelled = []
    for key in endpoints.ENDPOINT_ORDER:
        endpoint = endpoints.ENDPOINTS[key]
        n_fine = len(endpoint.resolve(fine, full_ontology).weights)
        resolved = endpoint.resolve(coarse, full_ontology)
        assert len(resolved.weights) < n_fine, key
        if resolved.cancellations:
            cancelled.append(key)
    # Locomotion is the clearest case: SNr, CUN, PPN, SNc and VTA all
    # become MB, and SNr's sign is the opposite of the other four.
    assert "locomotion" in cancelled


def test_cancellation_is_reported_for_opposite_signs_and_only_for_those(ccf):
    """Both halves of the cancellation rule, on constructed endpoints.

    Review found that `{1 if p > 0 else -1 for p in parts}` classified a
    ZERO part as negative, so two same-signed terms reaching one node
    through a zero-size descendant split were reported as cancelling. The
    partial-`region_sizes` fix below removed the only route to a zero part,
    so that exact repro is no longer reachable and the guard in `_resolve`
    is now belt-and-braces. What IS reachable, and is what this pins, is
    the rule either side of it: same signs never cancel, opposite signs
    always do, and an exact cancellation drops the node from `weights`
    while still naming it in `cancellations`.
    """
    def probe(second_measure):
        return endpoints.Endpoint(
            key="probe", name="probe", assay="probe",
            direction="higher = more probe",
            terms=[
                endpoints.EndpointTerm(
                    "SNr", "increase", "increase", "Liu2020", "mouse",
                    "optogenetic", "open field", "x" * 80),
                endpoints.EndpointTerm(
                    "VTA", "increase", second_measure, "Ilango2014", "mouse",
                    "optogenetic", "open field", "y" * 80),
            ],
        )

    # Same sign, one node (both are midbrain): no cancellation.
    agree = probe("increase").resolve(["MB"], ccf, normalize=False)
    assert agree.cancellations == ()
    assert agree.weights["MB"] == pytest.approx(1.0)
    assert "SIGN CANCELLATION" not in agree.report()

    # Opposite signs, exactly cancelling: reported, and the node is gone
    # from the weights rather than sitting there as a float residue.
    disagree = probe("decrease").resolve(["MB"], ccf, normalize=False)
    assert disagree.cancellations == ("MB",)
    assert "MB" not in disagree.weights
    assert "SIGN CANCELLATION" in disagree.report()

    # And at a resolution that separates them, neither happens.
    apart = probe("decrease").resolve(["SNr", "VTA"], ccf, normalize=False)
    assert apart.cancellations == ()
    assert apart.weights == {"SNr": pytest.approx(1.0),
                             "VTA": pytest.approx(-1.0)}


def test_partial_region_sizes_fall_back_to_the_equal_split_and_say_so(ccf):
    """Found by review: a silent unilateral collapse wearing an exact label.

    `region_sizes.get(d, 0.0)` gave the covered descendants the whole term
    and the missing ones nothing, while reporting a volume-weighted split.
    A caller passing voxel counts from a build has no guarantee of covering
    every node, so this is reachable.
    """
    probe = endpoints.Endpoint(
        key="probe", name="probe", assay="probe",
        direction="higher = more probe",
        terms=[
            endpoints.EndpointTerm(
                "HPF", "increase", "increase", "Meyers2006", "rat",
                "inactivation", "conditioned place preference", "x" * 80),
            endpoints.EndpointTerm(
                "SNr", "increase", "increase", "Liu2020", "mouse",
                "optogenetic", "open field", "y" * 80),
        ],
    )
    partial = probe.resolve(
        ["CA1", "CA3", "DG", "SNr"], ccf, normalize=False,
        region_sizes={"CA1": 3.0},
    )
    assert partial.weights["CA1"] == pytest.approx(1.0 / 3.0)
    assert partial.weights["CA3"] == pytest.approx(1.0 / 3.0)
    assert partial.equal_splits == ("HPF",)
    assert endpoints.ILLUSTRATIVE_EQUAL_SPLIT in partial.report()


def test_the_report_counts_the_terms_that_landed_not_the_ones_offered(
    ccf, fixture_network
):
    """'8 causal terms' next to 4 nodes would describe the library."""
    resolved = endpoints.ENDPOINTS["locomotion"].resolve(fixture_network, ccf)
    offered = endpoints.ENDPOINTS["locomotion"].active_terms()
    assert len(resolved.mapped_terms) < len(offered)
    assert "%d of %d term(s) landed" % (
        len(resolved.mapped_terms), len(offered)
    ) in resolved.report()


def test_terms_whose_direction_is_inferred_are_flagged_and_counted(
    ccf, fixture_network
):
    """The eight terms a reader should discount first, made visible.

    Not every `activity`/`measure` came straight off the manipulation:
    some are one synapse away, some rest on a drug's presumed effect on
    firing, and two translate a non-latency assay into this endpoint's
    score. `inferred` names which, and the summary and report surface it.
    """
    flagged = [
        (key, term)
        for key, endpoint in endpoints.ENDPOINTS.items()
        for term in endpoint.terms
        if term.inferred
    ]
    assert len(flagged) == 8
    for key, term in flagged:
        assert len(term.inferred) > 20, (key, term.citation)
        assert "INFERRED DIRECTION" in endpoints.ENDPOINTS[key].summary()
    # The microinjection terms whose agent silences and nothing else do NOT
    # need the flag, so it is not simply "every microinjection".
    by_technique = {
        term.technique
        for _, term in flagged
    }
    assert by_technique != {"microinjection"}
    resolved = endpoints.ENDPOINTS["analgesia"].resolve(fixture_network, ccf)
    assert "with an inferred direction" in resolved.report()
