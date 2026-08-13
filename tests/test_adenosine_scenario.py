"""P5.6c: the adenosine A2A receptor system.

What is under test:

  * the CONSTANTS, and the two things about them that a reader must not be
    allowed to forget -- that adenosine's "Kd" is a functional potency
    because its binding affinity cannot be measured, and that caffeine is
    not an A2A-selective drug;
  * the ANATOMY, which here is a string: A2A goes on the node
    `scenarios.d2_coupling()` already names, because [Schiffmann1991] found
    the striatal A2 receptor exclusively on the enkephalinergic MSN;
  * the CIRCUIT RESULT: the sign chain from an A2A agonist through four
    synapses to the thalamic readout, and its opposition to D2.
"""

import numpy as np
import pytest

from neurocascade import (
    acetylcholine,
    adenosine as ad,
    cannabinoid,
    dopamine,
    glutamate,
    histamine,
    scenarios,
    serotonin,
    stability,
)
from neurocascade.engine import Simulation


T_END = 60.0
BOLUS = 10000.0  # nM adenosine, illustrative
CAFFEINE_BOLUS = 1000000.0  # nM: caffeine's Kd here is 45 uM, so this is ~22x


def _run(systems=None, t_end=T_END, n_points=241, network=None, **doses):
    network = ad.striatal_network() if network is None else network
    sim = Simulation(
        network=network,
        receptor_systems=ad.adenosine_receptor_systems() if systems is None else systems,
    )
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, n_points),
        rates0=ad.rest_rates(network),
        method="LSODA",
    )
    return network, sim, result


def _final(result, node_name, network):
    return result["rates"][network.index[node_name]][-1]


# ---------------------------------------------------------------------------
# The two things a reader must not forget
# ---------------------------------------------------------------------------


def test_the_adenosine_affinity_is_a_functional_potency_and_says_so():
    """[Fredholm2011]'s Table 3 footnotes the adenosine row "Data are from
    functional studies", and its text explains why there is no alternative:
    adenosine is metabolized and regenerated inside any preparation used to
    measure it. This is the cleanest searched-and-empty in Phase 5, because
    the search ends in a published explanation of why the number cannot
    exist rather than in "not found"."""
    assert ad.A2A_KD_ADENOSINE_RAT == 150.0
    assert ad.A2A_KD_ADENOSINE_HUMAN == 310.0
    doc = ad.__doc__
    assert "exceedingly difficult to determine this affinity" in doc
    assert "functional" in doc


def test_caffeine_is_not_an_a2a_selective_drug():
    """The honest headline. 0.91 in rat means non-selective within any
    single measurement's error; 0.46 in human means A1-PREFERRING. This
    model has no A1 receptor, so a caffeine dose here produces one half of
    caffeine's action and none of the other."""
    assert ad.caffeine_a1_a2a_selectivity("rat") == pytest.approx(0.911, rel=0.01)
    assert ad.caffeine_a1_a2a_selectivity("human") == pytest.approx(0.457, rel=0.01)
    for species in ad.SPECIES_SETS:
        assert ad.caffeine_a1_a2a_selectivity(species) < 1.0


def test_adenosine_itself_is_tighter_at_a1_than_at_a2a():
    """So the receptor this module builds is not the one the endogenous
    ligand engages first either. 2.1-fold in rat, 3.1-fold in human."""
    assert ad.A2A_KD_ADENOSINE_RAT / ad.A1_KD_ADENOSINE_RAT == pytest.approx(
        2.05, rel=0.01
    )
    assert ad.A2A_KD_ADENOSINE_HUMAN / ad.A1_KD_ADENOSINE_HUMAN == pytest.approx(
        3.1, rel=0.01
    )


def test_no_ligand_is_ever_built_from_the_a1_constants():
    """A1 is out of the brief and must not be smuggled in. The A1 numbers
    exist only so the non-selectivity above can be asserted."""
    for species in ad.SPECIES_SETS:
        for lig in ad.ligands_a2a(species=species):
            kd = lig.koff / lig.kon
            assert kd not in (
                ad.A2A_KD_CAFFEINE_A1_RAT,
                ad.A2A_KD_CAFFEINE_A1_HUMAN,
                ad.A1_KD_ADENOSINE_RAT,
                ad.A1_KD_ADENOSINE_HUMAN,
            )


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------


def test_the_default_species_set_is_rat():
    assert ad.adenosine_kd() == ad.A2A_KD_ADENOSINE_RAT
    assert ad.caffeine_kd() == ad.A2A_KD_CAFFEINE_RAT
    with pytest.raises(ValueError):
        ad.adenosine_kd("mouse")


def test_the_species_choice_changes_the_caffeine_to_adenosine_ratio_fourfold():
    """The one real consequence of the species set: both affinities move in
    the same direction between species, so the individual values shift ~2x
    and the RATIO shifts 4x."""
    assert ad.caffeine_over_adenosine_affinity_ratio("rat") == pytest.approx(300.0)
    assert ad.caffeine_over_adenosine_affinity_ratio("human") == pytest.approx(
        75.5, rel=0.01
    )


def test_derivation_3_reproduces_necas_measured_koff_within_a_factor_of_three():
    """The internal check on sharing NECA's kon with adenosine. NECA is
    adenosine with one amide substitution, so this is the closest chemotype
    match derivation 3 has been given anywhere in this codebase, and the
    derived koff lands within 2.5-fold of NECA's own measured one."""
    adenosine_ligand, _ = ad.ligands_a2a("rat")
    derived_koff = adenosine_ligand.koff
    assert derived_koff == pytest.approx(0.075, rel=1e-9)
    assert 1.0 < derived_koff / ad.A2A_KOFF_NECA < 3.0


def test_necas_own_rate_pair_reproduces_its_published_kinetic_kd():
    """A transcription check on the [Guo2012] row: 0.03 / 5.0e-4 = 60 nM
    against that table's printed 58 nM."""
    assert ad.A2A_KOFF_NECA / ad.A2A_KON_NECA == pytest.approx(60.0)
    assert ad.A2A_KOFF_NECA / ad.A2A_KON_NECA == pytest.approx(
        ad.A2A_KD_NECA_KINETIC, rel=0.05
    )


def test_the_guo_panel_spans_320_fold_so_derivation_3_is_flagged_not_safe():
    span = ad.A2A_KON_LUF5835 / ad.A2A_KON_CGS21680
    assert span == pytest.approx(320.0, rel=0.01)
    assert span > 100.0


def test_every_ligand_reproduces_its_cited_affinity_exactly():
    for species in ad.SPECIES_SETS:
        adenosine_ligand, caffeine = ad.ligands_a2a(species=species)
        assert adenosine_ligand.koff / adenosine_ligand.kon == pytest.approx(
            ad.adenosine_kd(species), rel=1e-12
        )
        assert caffeine.koff / caffeine.kon == pytest.approx(
            ad.caffeine_kd(species), rel=1e-12
        )


def test_caffeine_is_zero_efficacy_and_that_is_flagged():
    _, caffeine = ad.ligands_a2a()
    assert caffeine.efficacy == 0.0
    assert ad.A2A_EFFICACY_CAFFEINE == 0.0


def test_the_clearance_constants_are_the_shared_ones():
    assert ad.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE
    assert ad.TOOL_COMPOUND_KE == histamine.TOOL_COMPOUND_KE
    assert ad.TOOL_COMPOUND_KE == cannabinoid.TOOL_COMPOUND_KE
    assert ad.ADENOSINE_KE == dopamine.DOPAMINE_KE
    assert ad.ADENOSINE_KE == glutamate.GLUTAMATE_KE


# ---------------------------------------------------------------------------
# The anatomy, which is a string
# ---------------------------------------------------------------------------


def test_a2a_lands_on_the_node_the_d2_scenario_already_names():
    """[Schiffmann1991]: the striatal A2 receptor is expressed exclusively
    by enkephalinergic (indirect-pathway) MSNs. Of the five nodes in this
    network it belongs on exactly one, and that one already carries D2."""
    assert ad.A2A_NODE == "Striatum_D2_MSN"
    assert set(ad.coupling_a2a()) == set(scenarios.d2_coupling())


def test_the_network_is_the_d2_scenarios_own_and_introduces_nothing_new():
    """This module adds no new illustrative circuit-layer number at all,
    which after P5.5 makes it the second Phase 5 module able to say so."""
    theirs = scenarios.basal_ganglia_indirect_pathway_network()
    ours = ad.striatal_network()
    assert [n.name for n in ours.nodes] == [n.name for n in theirs.nodes]
    assert np.array_equal(ours.connectivity, theirs.connectivity)
    assert ours.baseline_drive() == pytest.approx(theirs.baseline_drive())


def test_the_two_gains_have_opposite_signs_and_unequal_magnitudes():
    """[Ferre1993] establishes the DIRECTION of the A2/D2 opposition, not
    its relative magnitude, so the two gains oppose (which is measured) and
    differ in size (which is an artefact of two illustrative choices made at
    different times). No A2A-versus-D2 potency comparison should be read out
    of this module, and this test is where that is written down."""
    a2a = ad.coupling_a2a()[ad.A2A_NODE]
    d2 = scenarios.d2_coupling()[ad.A2A_NODE]
    assert a2a > 0.0 > d2
    assert abs(a2a) != abs(d2)


def test_a_saturating_a2a_agonist_does_not_rail_the_striatal_node():
    network = ad.striatal_network()
    drive = network.baseline_drive()
    drive[network.index[ad.A2A_NODE]] += ad.A2A_MECHANISM_GAIN
    rates, residual = stability.steady_state(network, drive=drive)
    assert residual < 1e-6
    check = stability.check_saturation(network, rates, drive=drive)
    assert check.ok


# ---------------------------------------------------------------------------
# The circuit result
# ---------------------------------------------------------------------------


def test_a_drug_free_run_is_exactly_flat_at_the_reused_networks_fixed_point():
    _, _, result = _run()
    assert result["rates"] == pytest.approx(0.5, abs=1e-6)


def test_the_reused_network_is_NOT_certified_globally_unique_and_that_is_recorded():
    """Its weights are +/-2.0, so its contraction number is 2.0 rather than
    the < 1 P3.4's certificate needs. Inherited from the D2 scenario since
    Track A and changed by nothing here; asserted so the fact is on the
    record. Every run in this module does converge to the 0.5 fixed point,
    which is a weaker statement than certified uniqueness and is the one
    this module is entitled to make."""
    network = ad.striatal_network()
    assert stability.contraction_number(network) == pytest.approx(2.0, abs=1e-9)
    rates, residual = stability.steady_state(network)
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert rates == pytest.approx(ad.rest_rates(network), abs=1e-9)


def test_an_a2a_agonist_excites_the_striatopallidal_node():
    """Gs/Golf, and the direction [Ferre1993] measured as increased GABA
    release from the striatopallidal terminal."""
    network, _, result = _run(adenosine=BOLUS)
    assert _final(result, "Striatum_D2_MSN", network) > 0.5


def test_the_sign_chain_through_four_synapses_reaches_the_thalamus_inverted():
    """Striatum -| GPe -| STN -> GPi -| Thalamus: three inhibitory links and
    one excitatory, so an increase at the striatum has to arrive at the
    thalamus as a decrease. Easy to get wrong by eye, so it is checked node
    by node rather than only at the end."""
    network, _, result = _run(adenosine=BOLUS)
    assert _final(result, "Striatum_D2_MSN", network) > 0.5
    assert _final(result, "GPe", network) < 0.5
    assert _final(result, "STN", network) > 0.5
    assert _final(result, "GPi", network) > 0.5
    assert _final(result, "Thalamus", network) < 0.5


def test_the_effect_is_larger_under_the_rat_set_than_the_human_one():
    """Adenosine's potency estimate is 150 nM in rat and 310 nM in human, so
    at one dose the rat set gives the higher occupancy. The direction is
    identical under both, which is what the species robustness claim is."""
    network = ad.striatal_network()
    results = {}
    for species in ad.SPECIES_SETS:
        _, _, result = _run(
            systems=ad.adenosine_receptor_systems(species=species),
            adenosine=BOLUS,
        )
        results[species] = _final(result, "Thalamus", network)
    assert results["rat"] < results["human"] < 0.5


def test_caffeine_alone_is_inert_because_there_is_no_endogenous_adenosine():
    """The standing P6.2 limitation, and for caffeine it is the difference
    between a model of the drug and a model of the receptor: caffeine's
    entire mechanism is the removal of an ongoing adenosine action, and
    there is no ongoing action here to remove. [Ledent1997] is the
    measurement this cannot yet be tested against."""
    network, _, dosed = _run(caffeine=CAFFEINE_BOLUS)
    _, _, control = _run()
    assert dosed["rates"] == pytest.approx(control["rates"], abs=1e-6)


def test_caffeine_reverses_the_adenosine_response():
    network, _, agonist = _run(adenosine=BOLUS)
    _, _, both = _run(adenosine=BOLUS, caffeine=CAFFEINE_BOLUS)
    agonist_effect = 0.5 - _final(agonist, "Thalamus", network)
    blocked_effect = 0.5 - _final(both, "Thalamus", network)
    assert agonist_effect > 0.0
    assert blocked_effect < 0.5 * agonist_effect


def test_a_300_fold_weaker_competitor_needs_a_large_dose_to_do_it():
    """Which is the model's version of "caffeine is a weak antagonist". The
    ratio is the published one and is not tunable: 45 uM against 150 nM."""
    network, _, small = _run(adenosine=BOLUS, caffeine=BOLUS)
    _, _, large = _run(adenosine=BOLUS, caffeine=CAFFEINE_BOLUS)
    assert _final(small, "Thalamus", network) < _final(large, "Thalamus", network)


# ---------------------------------------------------------------------------
# A2A and D2 on one node
# ---------------------------------------------------------------------------


def test_a2a_and_d2_run_together_on_one_node_without_engine_changes():
    systems = ad.adenosine_and_dopamine_receptor_systems()
    assert len(systems) == 2
    assert systems[0].coupling == {"Striatum_D2_MSN": ad.A2A_MECHANISM_GAIN}
    assert systems[1].coupling == scenarios.d2_coupling()
    network, _, result = _run(systems=systems, adenosine=BOLUS)
    assert _final(result, "Striatum_D2_MSN", network) > 0.5


def test_a_dopamine_agonist_opposes_the_a2a_agonist_on_the_shared_node():
    """[Ferre1993]'s functional opposition, reproduced without asserting any
    molecular mechanism for it (no heteromer; see the module docstring)."""
    systems = ad.adenosine_and_dopamine_receptor_systems()
    network = ad.striatal_network()
    _, _, a2a_only = _run(systems=systems, adenosine=BOLUS)
    _, _, both = _run(systems=systems, adenosine=BOLUS, dopamine=BOLUS)
    assert _final(both, "Striatum_D2_MSN", network) < _final(
        a2a_only, "Striatum_D2_MSN", network
    )
    assert _final(both, "Thalamus", network) > _final(a2a_only, "Thalamus", network)


# ---------------------------------------------------------------------------
# Readout bookkeeping
# ---------------------------------------------------------------------------


def test_the_locomotor_readout_is_the_d2_scenarios_own():
    ours = ad.locomotor_activity_readout()
    theirs = scenarios.locomotor_activity_readout()
    assert ours.name == theirs.name
    assert ours.weights == theirs.weights


def test_the_striatopallidal_readout_names_the_receptor_bearing_node():
    assert ad.striatopallidal_output_readout().weights == {ad.A2A_NODE: 1.0}


def test_this_module_does_not_disturb_the_other_p5_6_modules():
    """P5.6 ships three unrelated receptor systems; the only thing they
    share is the discipline, and nothing here should have changed the
    cortical loop the other two use."""
    assert histamine.CORTICAL_GAIN == cannabinoid.CORTICAL_GAIN
    assert acetylcholine.TOOL_COMPOUND_KE == ad.TOOL_COMPOUND_KE
