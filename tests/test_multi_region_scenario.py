"""The multi-region mu-opioid scenario, under both density maps.

P4.3 replaced the hand-written relative-density map with the Allen Oprm1
ISH one. Every qualitative claim this scenario makes is therefore asserted
twice, once under each map: that is the regression comparison the
substitution owes, and it is why `MAPS` is a parameter of nearly every test
below rather than a fixture.

The claims themselves are unchanged from before the substitution. Nothing
here was relaxed to accommodate the new numbers; what changed instead is
the illustrative mechanism-gain scale, for the reason
`scenarios.mu_opioid_mechanism_gain()` documents and
`test_the_gain_scale_rule_is_the_one_documented` pins.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade import scenarios
from neurocascade.legacy import connectome
from neurocascade.engine import Simulation
from neurocascade.legacy.expression import fetch, scenario_maps, sources


MAPS = ("atlas", "handwritten")


def coupling_for(which):
    if which == "atlas":
        return scenarios.mu_opioid_multi_region_coupling()
    return scenarios.mu_opioid_multi_region_coupling_handwritten()


def build_sim(which="atlas"):
    return Simulation(
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        network=scenarios.multi_region_mu_opioid_network(),
        coupling=coupling_for(which),
    )


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", MAPS)
def test_multi_region_coupling_targets_at_least_three_nodes(which):
    coupling = coupling_for(which)
    assert len(coupling) >= 3
    assert set(coupling) == {"VTA_GABA", "PAG_Interneuron", "LC"}
    # all three gains are nonzero and inhibitory (Gi-coupled receptor)
    assert all(gain < 0 for gain in coupling.values())


def test_the_atlas_map_is_mechanism_gain_times_measured_density():
    """P4.2's factorization, executed rather than described: every coupling
    is one illustrative scale times one measured, dimensionless density."""
    coupling = scenarios.mu_opioid_multi_region_coupling()
    gain = scenarios.mu_opioid_mechanism_gain()
    density_map = scenario_maps.scenario_density_map(
        "Oprm1",
        scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES,
        scenarios.MU_OPIOID_REFERENCE_NODE,
    )
    for node, structure_id in scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES.items():
        assert coupling[node] == pytest.approx(
            gain * density_map.for_region(structure_id)
        )
    # The reference region is 1.0 by construction, so the reference node's
    # coupling IS the mechanism gain and the other two are read against it.
    assert coupling[scenarios.MU_OPIOID_REFERENCE_NODE] == pytest.approx(gain)


def test_the_measured_density_orders_the_three_regions_lc_pag_vta():
    """The substantive content of the substitution. The hand-written map
    put PAG above LC; the transcript map puts LC highest, which is the
    order [TempelZukin1987] reported and the hand-tuned numbers did not
    encode. The disagreement with [Erbs2015]'s protein atlas, which orders
    them the other way by one semiquantitative tier, is recorded in
    `expression/anatomy.py` and is not smoothed over here."""
    density_map = scenario_maps.scenario_density_map(
        "Oprm1",
        scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES,
        scenarios.MU_OPIOID_REFERENCE_NODE,
    )
    vta, pag, lc = (
        density_map.for_region(scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES[node])
        for node in ("VTA_GABA", "PAG_Interneuron", "LC")
    )
    assert vta == pytest.approx(1.0)
    assert pag == pytest.approx(2.523, abs=0.01)
    assert lc == pytest.approx(4.275, abs=0.01)
    assert lc > pag > vta


def test_the_density_is_invariant_to_which_other_regions_are_in_the_matrix():
    """Why `reference-region` and not `max` or `mean`: dividing by one named
    region makes the answer depend only on the rows of the regions the
    scenario names. That is what lets the committed fixture and the 7.5 MB
    cache give the same three numbers, which the `real_data` test below
    then checks against the cache itself."""
    nodes = scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES
    three = scenario_maps.scenario_density_map(
        "Oprm1", nodes, scenarios.MU_OPIOID_REFERENCE_NODE
    )

    ontology = connectome.load_fixture_ontology()
    atlas = scenario_maps.load_atlas(["Oprm1"], ontology=ontology)
    wide_ids = sorted(ontology.structures)
    from neurocascade.legacy.expression import density as density_module

    wide = density_module.relative_density(
        atlas.energy_matrix(
            wide_ids,
            symbols=["Oprm1"],
            plane=scenario_maps.PLANE,
            combine=scenario_maps.COMBINE,
        ),
        "Oprm1",
        normalization="reference-region",
        reference=nodes[scenarios.MU_OPIOID_REFERENCE_NODE],
    )
    assert len(wide_ids) > 3 * len(nodes)
    for structure_id in nodes.values():
        assert three.for_region(structure_id) == pytest.approx(
            wide.for_region(structure_id)
        )


def test_the_gain_scale_rule_is_the_one_documented():
    """The ratios are data; the scale is illustrative and was set by a
    stated rule, so the rule is checked rather than the number trusted.

    The rule: the geometric mean of the two interneuron-motif couplings
    (VTA_GABA, PAG_Interneuron) stays where the hand-tuning put it. LC is
    excluded because its density moved most, so including it would let the
    least constrained node set the scale for the other two.
    """
    atlas = scenarios.mu_opioid_multi_region_coupling()
    hand = scenarios.mu_opioid_multi_region_coupling_handwritten()
    motif = ("VTA_GABA", "PAG_Interneuron")

    def geometric_mean(coupling, nodes):
        return float(np.exp(np.mean(np.log([abs(coupling[n]) for n in nodes]))))

    assert geometric_mean(atlas, motif) == pytest.approx(
        geometric_mean(hand, motif), rel=0.01
    )
    # And the substitution is a genuine change, not a rescaling: the
    # ordering of the three couplings is different.
    order = lambda c: sorted(c, key=lambda k: abs(c[k]))  # noqa: E731
    assert order(atlas) != order(hand)
    assert order(atlas) == ["VTA_GABA", "PAG_Interneuron", "LC"]
    assert order(hand) == ["LC", "PAG_Interneuron", "VTA_GABA"]


def test_one_mechanism_gain_covers_all_three_nodes():
    """The correction P4.3 makes to the old rationale: `Simulation.rhs`
    adds `gain * activation` to a node's OWN drive, and the downstream
    amplification the old map folded into LC's smaller gain is already
    carried by the connectivity weight, so folding it in again counted it
    twice. One gain, three densities."""
    coupling = scenarios.mu_opioid_multi_region_coupling()
    gain = scenarios.mu_opioid_mechanism_gain()
    assert isinstance(gain, float) and gain < 0.0
    ratios = sorted(value / gain for value in coupling.values())
    assert ratios == pytest.approx([1.0, 2.523, 4.275], abs=0.01)


def test_the_structures_are_the_ones_the_connectome_port_uses():
    """One anatomy, named twice; if the two ever disagree the density would
    be read for a different region than the one the ported network wires."""
    from neurocascade.legacy import connectome_scenarios as cs

    ontology = connectome.load_fixture_ontology()
    port = cs.SCENARIO_PORTS["mu_opioid_multi_region"]
    for node, structure_id in scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES.items():
        assert ontology.acronym_for_id(structure_id) == port.by_node[node].region


# ---------------------------------------------------------------------------
# The qualitative results, under BOTH maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", MAPS)
def test_state_arrays_have_expected_shapes(which):
    sim = build_sim(which)
    t_eval = np.linspace(0, 10, 50)
    result = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    assert result["concentrations"].shape == (2, 50)
    assert result["occupancy"].shape == (2, 50)
    # VTA_GABA, VTA_DA, NAc, PFC, PAG_Interneuron, PAG_Output, LC
    assert result["rates"].shape == (7, 50)
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)


@pytest.mark.parametrize("which", MAPS)
def test_agonist_raises_incentive_salience_above_baseline(which):
    """Sanity check that expanding the reward circuit's network with two
    additional, unconnected sub-systems (PAG, LC) doesn't perturb the
    original reward-circuit dynamics: this should behave the same as
    test_engine.py's identically-named test on the smaller 4-node network.
    """
    sim = build_sim(which)
    readout = scenarios.incentive_salience_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_scores = np.array(
        [readout.score(sim.network, baseline["rates"][:, i]) for i in range(len(t_eval))]
    )
    agonist_scores = np.array(
        [readout.score(sim.network, agonist["rates"][:, i]) for i in range(len(t_eval))]
    )
    assert agonist_scores.max() > baseline_scores.max()


@pytest.mark.parametrize("which", MAPS)
def test_agonist_raises_analgesia_index_above_baseline(which):
    sim = build_sim(which)
    readout = scenarios.analgesia_index_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    agonist_final = readout.score(sim.network, agonist["rates"][:, -1])
    # Agonist disinhibits PAG_Output (via PAG_Interneuron suppression), so
    # descending-analgesia output, and therefore analgesia_index, should
    # rise well above baseline.
    assert agonist_final > baseline_final + 0.1


@pytest.mark.parametrize("which", MAPS)
def test_agonist_lowers_arousal_index_below_baseline(which):
    sim = build_sim(which)
    readout = scenarios.arousal_index_readout()

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    agonist_final = readout.score(sim.network, agonist["rates"][:, -1])
    # LC is coupled directly (not via disinhibition): agonist activation
    # suppresses LC firing directly, so arousal_index should fall below
    # baseline (illustrative opioid-sedation effect).
    assert agonist_final < baseline_final - 0.05


@pytest.mark.parametrize("which", MAPS)
def test_antagonist_alone_stays_near_baseline_on_all_three_readouts(which):
    sim = build_sim(which)
    readouts = {
        "incentive_salience": scenarios.incentive_salience_readout(),
        "analgesia_index": scenarios.analgesia_index_readout(),
        "arousal_index": scenarios.arousal_index_readout(),
    }

    t_eval = np.linspace(0, 40, 200)
    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    antagonist = sim.run(dose_concentrations=[0.0, 1.0], t_span=(0, 40), t_eval=t_eval)

    for readout in readouts.values():
        baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
        antagonist_final = readout.score(sim.network, antagonist["rates"][:, -1])
        assert abs(antagonist_final - baseline_final) < 1e-2


@pytest.mark.parametrize("which", MAPS)
def test_naloxone_reverses_all_three_simultaneous_effects(which):
    """The key demonstration: one antagonist, competing at ONE shared
    receptor pool, reverses reward, analgesia, AND sedation at once, because
    all three come from the same occupancy trace, not three independent
    drug actions on three independent receptors.
    """
    sim = build_sim(which)
    readouts = {
        "incentive_salience": scenarios.incentive_salience_readout(),
        "analgesia_index": scenarios.analgesia_index_readout(),
        "arousal_index": scenarios.arousal_index_readout(),
    }
    t_span = (0, 40)
    t_split = 20.0

    t_eval_1 = np.linspace(t_span[0], t_split, 200)
    phase1 = sim.run(
        dose_concentrations=[1.0, 0.0], t_span=(t_span[0], t_split), t_eval=t_eval_1
    )
    scores_at_split = {
        name: readout.score(sim.network, phase1["rates"][:, -1])
        for name, readout in readouts.items()
    }

    C0 = phase1["concentrations"][:, -1].copy()
    C0[1] += 3.0  # naloxone bolus injected at t_split
    occ0 = phase1["occupancy"][:, -1]
    rates0 = phase1["rates"][:, -1]
    y0 = np.concatenate([C0, occ0, rates0])

    t_eval_2 = np.linspace(t_split, t_span[1], 200)
    sol = solve_ivp(
        sim.rhs, (t_split, t_span[1]), y0, method="RK45", t_eval=t_eval_2, max_step=0.1
    )
    d, r = sim.d, sim.r
    rates2 = sol.y[d + r : d + r + sim.c]

    # incentive_salience and analgesia_index were elevated by fentanyl, so
    # naloxone should pull them back DOWN toward baseline; arousal_index was
    # suppressed by fentanyl, so naloxone should pull it back UP.
    for name in ("incentive_salience", "analgesia_index"):
        readout = readouts[name]
        scores_after = np.array(
            [readout.score(sim.network, rates2[:, i]) for i in range(len(t_eval_2))]
        )
        assert scores_after.min() < scores_at_split[name]

    readout = readouts["arousal_index"]
    scores_after = np.array(
        [readout.score(sim.network, rates2[:, i]) for i in range(len(t_eval_2))]
    )
    assert scores_after.max() > scores_at_split["arousal_index"]


# ---------------------------------------------------------------------------
# The real files. An extra: it skips when the gitignored cache is absent.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not fetch.is_cached(sources.unionize_key("Oprm1")),
    reason="data/expression/ is not populated; run scripts/fetch_expression.py",
)
def test_real_cache_gives_the_same_coupling_as_the_committed_fixture():
    """The claim `scenario_maps.py` makes in prose, checked against the
    7.5 MB file it is a claim about."""
    from_fixture = scenarios.mu_opioid_multi_region_coupling(source="fixture")
    from_cache = scenarios.mu_opioid_multi_region_coupling(source="cache")
    assert set(from_fixture) == set(from_cache)
    for node, value in from_fixture.items():
        assert value == pytest.approx(from_cache[node], rel=1e-12)
