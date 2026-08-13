"""P5.6a: the histamine H1 receptor system.

Three kinds of thing are under test:

  * the CONSTANTS, and specifically the agreements between independently
    measured ones. The load-bearing one is that diphenhydramine's
    derivation-2 association rate (a residence time from [Bosma2018] divided
    into an affinity from [Booth2002]) reproduces mepyramine's MEASURED
    association rate from [Bosma2017] to 3%, which nothing forced;
  * the AFFINITY-STATE problem, i.e. that every qualitative result holds
    under both of histamine's published affinities;
  * the CIRCUIT-LEVEL effect: an H1 agonist raises cortical output, an H1
    antagonist reverses it, and an H1 antagonist alone does nothing (the
    standing no-endogenous-tone limitation, asserted so that P6.2 changes a
    written-down expectation rather than a silence).
"""

import numpy as np
import pytest

from neurocascade import (
    acetylcholine,
    dopamine,
    glutamate,
    histamine as ha,
    noradrenaline,
    serotonin,
    stability,
)
from neurocascade.engine import Simulation


T_END = 60.0
BOLUS = 100000.0  # nM, illustrative, as for every tool compound in Phase 5


def _run(t_end=T_END, affinity_state="sf9", n_points=241, **doses):
    network = ha.tuberomammillary_cortical_network()
    sim = Simulation(
        network=network,
        receptor_systems=ha.histaminergic_receptor_systems(
            affinity_state=affinity_state
        ),
    )
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, n_points),
        rates0=ha.rest_rates(network),
        method="LSODA",
    )
    return network, sim, result


def _pyramidal(result):
    return result["rates"][1]


def _peak_deviation(result, row=1):
    trace = result["rates"][row] - 0.5
    return trace[np.argmax(np.abs(trace))]


# ---------------------------------------------------------------------------
# The constants and the agreements between them
# ---------------------------------------------------------------------------


def test_diphenhydramine_kd_is_the_mean_of_the_two_published_determinations():
    """And that mean is pKi 7.90, which is what IUPHAR records for
    [Booth2002] -- the check that the PDSP/BindingDB extraction is the right
    paper's data."""
    assert ha.H1_KD_DIPHENHYDRAMINE == pytest.approx(12.6, abs=1e-9)
    p_ki = 9.0 - np.log10(ha.H1_KD_DIPHENHYDRAMINE)
    assert p_ki == pytest.approx(7.9, abs=0.005)


def test_derivation_2_reproduces_the_measured_association_rate_of_a_different_drug():
    """The strongest arithmetic result in this module.

    Diphenhydramine's kon is never measured anywhere used here. It is
    `koff / Kd` with koff from a residence time in [Bosma2018] and Kd from a
    competition experiment in [Booth2002] -- a different laboratory, a
    different cell line, a different decade. It lands within 3% of
    mepyramine's kon, MEASURED in a third experiment. Nothing in the
    construction could have produced that agreement.
    """
    ratio = ha.H1_KON_DIPHENHYDRAMINE / ha.H1_KON_MEPYRAMINE
    assert ratio == pytest.approx(0.968, abs=0.005)
    assert abs(1.0 - ratio) < 0.05


def test_the_kon_panel_spans_far_more_than_the_two_agreeing_ligands_do():
    """Why the agreement above is evidence for the chemotype refinement of
    derivation 3 rather than evidence that all H1 ligands associate alike:
    [Bosma2017]'s own panel spans 330-fold."""
    kons = list(ha.H1_KON_PANEL.values())
    span = max(kons) / min(kons)
    assert span == pytest.approx(333.3, rel=0.01)
    assert span > 100.0


def test_the_panel_rows_reproduce_their_own_published_residence_times():
    """koff and RT are the same measurement, so `1/koff` must land inside
    the published RT +/- its published error. The comparison is made against
    the paper's OWN error bars rather than against a tolerance chosen here,
    because the two columns are printed to different numbers of significant
    figures (doxepin's koff is quoted as 0.06, which is 16.7 min against a
    printed RT of 22 +/- 7) and inventing a tolerance would hide exactly the
    transcription error this test is for.
    """
    published_rt = {  # min, mean +/- error, [Bosma2017] Table 2
        "mepyramine": (3.9, 0.6),
        "doxepin": (22.0, 7.0),
        "triprolidine": (3.5, 0.4),
        "desloratadine": (160.0, 50.0),
        "olopatadine": (170.0, 10.0),
        "levocetirizine": (140.0, 20.0),
        "acrivastine": (15.6, 0.9),
    }
    for name, koff in ha.H1_KOFF_PANEL.items():
        mean, error = published_rt[name]
        assert mean - error <= 1.0 / koff <= mean + error, name


def test_mepyramine_affinity_agrees_across_four_independent_routes():
    """1.10 to 3.16 nM from two laboratories, three methods and two
    temperatures: a 2.9-fold spread, the tightest in this module."""
    values = list(ha.mepyramine_affinity_spread().values())
    assert max(values) / min(values) == pytest.approx(2.89, rel=0.02)
    assert max(values) / min(values) < 3.5


def test_the_histamine_affinity_spread_is_recorded_and_is_33_fold():
    """The module cannot hold both at once; it must be able to say how far
    apart they are."""
    spread = ha.H1_KD_HISTAMINE_COS7 / ha.H1_KD_HISTAMINE_SF9
    assert spread == pytest.approx(33.1, rel=0.02)


def test_the_default_affinity_state_is_the_sf9_one():
    assert ha.histamine_kd() == ha.H1_KD_HISTAMINE_SF9
    assert ha.histamine_kd("cos7") == ha.H1_KD_HISTAMINE_COS7
    with pytest.raises(ValueError):
        ha.histamine_kd("brain")


def test_every_ligand_reproduces_its_cited_affinity_exactly():
    """Derivations 1, 2 and 3 all have to give back `koff/kon == Kd`."""
    for state in ha.AFFINITY_STATES:
        histamine, diphenhydramine, mepyramine = ha.ligands_h1(affinity_state=state)
        assert histamine.koff / histamine.kon == pytest.approx(
            ha.histamine_kd(state), rel=1e-12
        )
        assert diphenhydramine.koff / diphenhydramine.kon == pytest.approx(
            ha.H1_KD_DIPHENHYDRAMINE, rel=1e-12
        )
        assert mepyramine.koff / mepyramine.kon == pytest.approx(1.4, rel=1e-12)


def test_diphenhydramines_margin_over_the_transmitter_is_190_fold_and_that_is_small():
    """Recorded as a number because the competitive displacement seen in the
    scenario below is a consequence of it, not a defect."""
    _, diphenhydramine, _ = ha.ligands_h1()
    margin = ha.histamine_kd() / (diphenhydramine.koff / diphenhydramine.kon)
    assert margin == pytest.approx(190.4, rel=0.01)


def test_both_antagonists_are_zero_efficacy_and_share_one_flagged_constant():
    """One constant rather than two literal zeros, so that the flagged
    simplification (both are really INVERSE agonists, [Bakker2000]) has a
    single place to be read and a single place to change."""
    _, diphenhydramine, mepyramine = ha.ligands_h1()
    assert diphenhydramine.efficacy is ha.H1_EFFICACY_ANTAGONIST
    assert mepyramine.efficacy is ha.H1_EFFICACY_ANTAGONIST
    assert ha.H1_EFFICACY_ANTAGONIST == 0.0


def test_the_tool_compound_clearance_is_shared_with_every_earlier_module():
    assert ha.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE
    assert ha.TOOL_COMPOUND_KE == dopamine.TOOL_COMPOUND_KE
    assert ha.TOOL_COMPOUND_KE == noradrenaline.TOOL_COMPOUND_KE
    assert ha.TOOL_COMPOUND_KE == acetylcholine.TOOL_COMPOUND_KE
    assert ha.TOOL_COMPOUND_KE == glutamate.TOOL_COMPOUND_KE


def test_the_transmitter_clearance_proxy_is_shared_too():
    assert ha.HISTAMINE_KE == serotonin.SEROTONIN_KE
    assert ha.HISTAMINE_KE == dopamine.DOPAMINE_KE
    assert ha.HISTAMINE_KE == noradrenaline.NORADRENALINE_KE
    assert ha.HISTAMINE_KE == glutamate.GLUTAMATE_KE


# ---------------------------------------------------------------------------
# The circuit
# ---------------------------------------------------------------------------


def test_the_network_is_the_shared_cortical_loop_plus_an_isolated_source():
    network = ha.tuberomammillary_cortical_network()
    assert [node.name for node in network.nodes] == [
        "TMN_HA",
        "PFC_Pyramidal",
        "PFC_GABA",
    ]
    W = network.connectivity
    assert W[0].tolist() == [0.0, 0.0, 0.0]  # TMN has no inputs
    assert W[:, 0].tolist() == [0.0, 0.0, 0.0]  # and no outputs
    assert W[1, 2] == ha.CORTICAL_INHIBITION_WEIGHT
    assert W[2, 1] == ha.CORTICAL_RECURRENT_WEIGHT


def test_the_cortical_loop_is_character_for_character_the_shared_one():
    """P5.5's argument: the result has to come from the connectivity every
    other Phase 5 scenario already ran on."""
    assert ha.CORTICAL_INHIBITION_WEIGHT == glutamate.CORTICAL_INHIBITION_WEIGHT
    assert ha.CORTICAL_RECURRENT_WEIGHT == glutamate.CORTICAL_RECURRENT_WEIGHT
    assert ha.CORTICAL_GAIN == glutamate.CORTICAL_GAIN
    assert ha.CORTICAL_INHIBITION_WEIGHT == noradrenaline.CORTICAL_INHIBITION_WEIGHT
    assert ha.CORTICAL_RECURRENT_WEIGHT == noradrenaline.CORTICAL_RECURRENT_WEIGHT


def test_rule_1_the_baselines_are_the_ones_stability_derives():
    network = ha.tuberomammillary_cortical_network()
    balanced = stability.balanced_baselines(network)
    assert balanced.baseline_drive() == pytest.approx(
        np.array([0.0, 0.45, -0.4]), abs=1e-9
    )
    assert balanced.baseline_drive() == pytest.approx(
        network.baseline_drive(), abs=1e-9
    )


def test_rule_2_the_network_has_exactly_one_fixed_point():
    network = ha.tuberomammillary_cortical_network()
    assert stability.contraction_number(network) == pytest.approx(0.9, abs=1e-9)
    assert stability.contraction_number(network) < 1.0


def test_rule_3_a_saturating_agonist_dose_does_not_rail_the_cortical_node():
    """The per-receptor unit budget: gain 1.0, one receptor on the node."""
    network = ha.tuberomammillary_cortical_network()
    drive = network.baseline_drive()
    drive[1] += ha.H1_MECHANISM_GAIN  # full occupancy by a full agonist
    rates, residual = stability.steady_state(network, drive=drive)
    assert residual < 1e-6
    check = stability.check_saturation(network, rates, drive=drive)
    assert check.ok
    assert check.min_sensitivity > stability.SATURATION_FLOOR


def test_a_drug_free_run_is_exactly_flat():
    _, _, result = _run()
    assert result["rates"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# The pharmacology, at the circuit level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(ha.AFFINITY_STATES))
def test_histamine_raises_cortical_output_under_both_affinity_states(state):
    _, _, result = _run(affinity_state=state, histamine=BOLUS)
    assert _peak_deviation(result) > 0.005


def test_the_affinity_state_changes_the_size_of_the_effect_but_not_its_sign():
    """The honest form of the 33-fold uncertainty: the direction is robust,
    the magnitude is not, and the ratio is reported rather than smoothed."""
    _, _, sf9 = _run(affinity_state="sf9", histamine=BOLUS)
    _, _, cos7 = _run(affinity_state="cos7", histamine=BOLUS)
    strong = _peak_deviation(sf9)
    weak = _peak_deviation(cos7)
    assert strong > 0.0 and weak > 0.0
    assert strong > weak


def test_diphenhydramine_alone_is_inert_because_there_is_no_endogenous_tone():
    """The standing P6.2 limitation, written down so the day tone exists the
    expectation that must change is already stated. Real diphenhydramine
    given alone is sedating; this model cannot produce that, because a
    competitive antagonist with nothing to compete against does nothing."""
    _, _, result = _run(diphenhydramine=BOLUS)
    assert result["rates"] == pytest.approx(0.5, abs=1e-6)


def test_mepyramine_alone_is_inert_for_the_same_reason():
    _, _, result = _run(mepyramine=BOLUS)
    assert result["rates"] == pytest.approx(0.5, abs=1e-6)


def test_diphenhydramine_reverses_the_histamine_response():
    _, _, agonist = _run(histamine=BOLUS)
    _, _, both = _run(histamine=BOLUS, diphenhydramine=BOLUS)
    assert _peak_deviation(both) < 0.5 * _peak_deviation(agonist)


def test_mepyramine_is_the_more_effective_blocker_of_the_two():
    """It should be: 1.4 nM against 12.6 nM, a 9-fold tighter competitor at
    the same pool, and the two are dosed identically here."""
    _, _, with_diphenhydramine = _run(histamine=BOLUS, diphenhydramine=BOLUS)
    _, _, with_mepyramine = _run(histamine=BOLUS, mepyramine=BOLUS)
    assert _peak_deviation(with_mepyramine) < _peak_deviation(with_diphenhydramine)


def test_a_large_histamine_dose_displaces_the_antagonist():
    """Competitive means surmountable. The 190-fold margin makes this
    visible at doses the scenario actually uses, which is a consequence of
    the published affinities and not a tuning choice."""
    _, _, small = _run(histamine=BOLUS, diphenhydramine=BOLUS)
    _, _, large = _run(histamine=100.0 * BOLUS, diphenhydramine=BOLUS)
    assert _peak_deviation(large) > _peak_deviation(small)


def test_the_source_nucleus_node_never_moves():
    """TMN_HA carries no receptor and has no edges; it is a placeholder for
    P6.1 and the tests should notice if it silently acquires an input."""
    _, _, result = _run(histamine=BOLUS, diphenhydramine=BOLUS)
    assert result["rates"][0] == pytest.approx(0.5, abs=1e-9)


def test_the_interneuron_follows_the_pyramidal_node_through_the_loop():
    """The only path from H1 to PFC_GABA is the +0.8 recurrent weight, so
    the interneuron has to move in the same direction and later."""
    _, _, result = _run(histamine=BOLUS)
    pyramidal = _pyramidal(result) - 0.5
    interneuron = result["rates"][2] - 0.5
    assert interneuron.max() > 0.0
    assert np.argmax(interneuron) >= np.argmax(pyramidal)


# ---------------------------------------------------------------------------
# Shared-readout bookkeeping
# ---------------------------------------------------------------------------


def test_the_cortical_readout_is_the_same_object_as_every_other_modules():
    readouts = [
        ha.cortical_output_readout(),
        serotonin.cortical_output_readout(),
        noradrenaline.cortical_output_readout(),
        acetylcholine.cortical_output_readout(),
        glutamate.cortical_output_readout(),
    ]
    for readout in readouts[1:]:
        assert readout.name == readouts[0].name
        assert readout.weights == readouts[0].weights


def test_the_tone_readout_names_the_source_nucleus():
    readout = ha.histaminergic_tone_readout()
    assert readout.weights == {"TMN_HA": 1.0}
