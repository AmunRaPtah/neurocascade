"""P5.6b: cannabinoid CB1 as connection modulation.

What is under test, and the three kinds are deliberately separate:

  * the MECHANISM CHOICE, as a set of identities rather than a numeric
    result: empty coupling, a gain equal to a measured suppression, a
    modulated weight that never changes sign, and a drug-free run that
    reproduces the drug-free network exactly;
  * the CONSTANTS and the honest limits of the compiled affinities -- in
    particular that the rimonabant-versus-THC affinity ordering is NOT
    resolved by the published ranges, which is a finding and is asserted as
    one;
  * the CIRCUIT RESULT: presynaptic CB1 activation disinhibits the
    pyramidal node, does so by at most the measured fraction, and leaves the
    presynaptic node's own rate to move only through the loop.
"""

import numpy as np
import pytest

from neurocascade import (
    acetylcholine,
    cannabinoid as cb,
    dopamine,
    glutamate,
    histamine,
    noradrenaline,
    serotonin,
    stability,
)
from neurocascade.engine import Simulation


T_END = 60.0
BOLUS = 100000.0  # nM, illustrative
SATURATING = 10000000.0  # nM: >20000x the weakest published Kd here


def _run(t_end=T_END, affinity="midpoint", thc_efficacy=cb.CB1_EFFICACY_THC,
         connections=None, n_points=241, **doses):
    network = cb.cortical_network()
    sim = Simulation(
        network=network,
        receptor_systems=cb.cannabinoid_receptor_systems(
            affinity=affinity, thc_efficacy=thc_efficacy, connections=connections
        ),
    )
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, n_points),
        rates0=cb.rest_rates(network),
        method="LSODA",
    )
    return network, sim, result


def _peak_deviation(result, row=0):
    trace = result["rates"][row] - 0.5
    return trace[np.argmax(np.abs(trace))]


# ---------------------------------------------------------------------------
# The mechanism choice, as identities
# ---------------------------------------------------------------------------


def test_cb1_adds_no_drive_to_any_node():
    """The strongest form of P3.4's saturation rule, and the whole point of
    the module: a presynaptic receptor cannot be a `coupling` entry."""
    assert cb.coupling_cb1() == {}
    for system in cb.cannabinoid_receptor_systems():
        assert system.coupling == {}


def test_the_gain_is_the_measured_maximal_suppression_and_nothing_else():
    assert cb.CB1_MAX_SUPPRESSED_FRACTION == 0.47
    assert cb.CB1_CONNECTION_GAIN == -0.47
    assert cb.connection_modulation_cb1() == {
        ("PFC_Pyramidal", "PFC_GABA"): -0.47
    }


def test_the_measurement_was_made_far_enough_above_its_own_ec50_to_be_maximal():
    """36-fold, i.e. 97% of a Hill-1 plateau. This is what licenses calling
    47% a maximum rather than a point on a curve."""
    margin = cb.saturating_margin_of_the_measured_maximum()
    assert margin == pytest.approx(36.2, rel=0.01)
    hill_fraction = margin / (1.0 + margin)
    assert hill_fraction > 0.97


def test_the_engines_factor_equals_one_minus_the_fraction_removed():
    """`1 + gain*activation` has to be `1 - 0.47*activation`; a sign error
    here would turn disinhibition into extra inhibition."""
    for activation in (0.0, 0.25, 0.5, 1.0):
        factor = 1.0 + cb.CB1_CONNECTION_GAIN * activation
        assert factor == pytest.approx(1.0 - 0.47 * activation)
        assert cb.modulated_weight(activation) == pytest.approx(
            cb.CORTICAL_INHIBITION_WEIGHT * factor
        )


def test_the_modulated_weight_never_changes_sign():
    """A `connection_modulation` gain below -1 would silently turn an
    inhibitory synapse excitatory, and nothing in the engine forbids it. The
    measured gain is -0.47, so at full activation the synapse retains 53% of
    its strength and stays inhibitory."""
    base = cb.CORTICAL_INHIBITION_WEIGHT
    for activation in np.linspace(0.0, 1.0, 21):
        weight = cb.modulated_weight(activation)
        assert weight < 0.0
        # weakened but never abolished, and never past 53% of base
        assert base - 1e-12 <= weight <= base * 0.53 + 1e-12


def test_a_saturating_full_agonist_removes_exactly_the_measured_fraction():
    """The engine's own arithmetic, checked against the module's, at a dose
    that leaves the pool essentially fully occupied by 2-AG."""
    network, sim, result = _run(**{"2ag": SATURATING})
    system = sim.receptor_systems[0]
    occupancy = result["occupancy"][sim.state_rows(0), :]
    activation = system.receptor_site.activation(occupancy[:, 1])
    assert activation > 0.99
    factor = 1.0 + cb.CB1_CONNECTION_GAIN * activation
    assert factor == pytest.approx(0.53, abs=0.01)


def test_a_drug_free_run_reproduces_the_drug_free_network_exactly():
    _, _, result = _run()
    assert result["rates"] == pytest.approx(0.5, abs=1e-6)


def test_cb1_and_glutamate_modulate_disjoint_connections_of_the_same_loop():
    """Not a design convenience: it is what the signs of the connectivity
    matrix already mean. It is also what makes a combined
    glutamate + cannabinoid scenario well posed without a new decision."""
    assert cb.GABAERGIC_CONNECTION == ("PFC_Pyramidal", "PFC_GABA")
    assert cb.GLUTAMATERGIC_CONNECTION == glutamate.GLUTAMATERGIC_CONNECTION
    assert cb.GABAERGIC_CONNECTION != cb.GLUTAMATERGIC_CONNECTION
    network = cb.cortical_network()
    i, j = (network.index[n] for n in cb.GABAERGIC_CONNECTION)
    assert network.connectivity[i, j] < 0.0  # the GABAergic one
    i, j = (network.index[n] for n in cb.GLUTAMATERGIC_CONNECTION)
    assert network.connectivity[i, j] > 0.0  # the glutamatergic one


def test_this_module_adds_no_new_receptor_site_class():
    """The brief said CB1 does not fit the postsynaptic receptor pattern.
    The answer turned out to be that the codebase already had the pattern it
    does fit, so unlike P5.4 and P5.5 nothing was added to receptor.py."""
    from neurocascade.receptor import ReceptorSite

    site = cb.receptor_site_cb1()
    assert type(site) is ReceptorSite


# ---------------------------------------------------------------------------
# The constants, and what the compiled ranges do and do not settle
# ---------------------------------------------------------------------------


def test_the_bounds_are_the_published_ones():
    assert cb.CB1_KD_BOUNDS["thc"] == (5.05, 80.3)
    assert cb.CB1_KD_BOUNDS["anandamide"] == (61.0, 543.0)
    assert cb.CB1_KD_BOUNDS["2ag"] == (58.3, 472.0)
    assert cb.CB1_KD_BOUNDS["rimonabant"] == (1.8, 12.3)


def test_the_midpoint_is_the_geometric_mean_and_the_endpoints_are_published():
    """The midpoint is a DERIVED SUMMARY of two cited numbers, not a
    measurement: no experiment reported 20.1 nM for THC. The two endpoints
    are the published values, and every qualitative claim in this file is
    asserted under all three."""
    for name, (low, high) in cb.CB1_KD_BOUNDS.items():
        assert cb.cb1_kd(name, "midpoint") == pytest.approx(np.sqrt(low * high))
        assert cb.cb1_kd(name, "tight") == low
        assert cb.cb1_kd(name, "weak") == high
        assert low < cb.cb1_kd(name, "midpoint") < high
    assert cb.cb1_kd("thc") == pytest.approx(20.14, rel=0.01)
    with pytest.raises(ValueError):
        cb.cb1_kd("thc", "mean")


def test_only_one_of_the_four_affinity_orderings_is_resolved():
    """A finding, not an omission, and a wider one than expected.

    Of the orderings this module would like to be able to state, exactly
    one survives every combination of published values: rimonabant
    (1.8-12.3 nM) is tighter than both endocannabinoids (58.3-472 and
    61-543 nM). THC's compiled range (5.05-80.3 nM) OVERLAPS both
    endocannabinoids' and also overlaps rimonabant's, so neither
    "rimonabant is tighter than THC" nor "THC is tighter than anandamide"
    is settled by [Pertwee2010]'s table. That review says as much in prose
    about one of them -- anandamide's "affinity for the CB1 receptor is
    also similar to that of Delta9-THC" -- and this test is the arithmetic
    version, so the model can never quietly claim a selectivity the
    compiled data does not contain.
    """
    resolved = {
        (tighter, weaker)
        for tighter in cb.CB1_KD_BOUNDS
        for weaker in cb.CB1_KD_BOUNDS
        if tighter != weaker and cb.affinity_ordering_is_resolved(tighter, weaker)
    }
    assert resolved == {("rimonabant", "2ag"), ("rimonabant", "anandamide")}


def test_the_unresolved_orderings_still_have_a_definite_midpoint_ordering():
    """Which is exactly the trap: the midpoints look decisive and the data
    is not, so the ordering is only ever used through
    `affinity_ordering_is_resolved`."""
    assert cb.cb1_kd("rimonabant") < cb.cb1_kd("thc") < cb.cb1_kd("2ag")
    assert cb.cb1_kd("rimonabant", "weak") > cb.cb1_kd("thc", "tight")
    assert cb.cb1_kd("thc", "weak") > cb.cb1_kd("anandamide", "tight")


def test_the_two_endocannabinoids_are_separated_by_efficacy_not_affinity():
    """Their midpoint affinities are within 1.1-fold; their measured
    efficacies differ by 1.6-fold. Carrying only one of them would have
    meant choosing between 1.0 and 0.61 with nothing to justify it."""
    ratio = cb.cb1_kd("anandamide") / cb.cb1_kd("2ag")
    assert 0.9 < ratio < 1.2
    assert cb.CB1_EFFICACY_2AG / cb.CB1_EFFICACY_ANANDAMIDE == pytest.approx(
        1.64, rel=0.01
    )


def test_the_efficacy_ordering_is_2ag_then_anandamide_then_thc():
    assert cb.CB1_EFFICACY_2AG > cb.CB1_EFFICACY_ANANDAMIDE
    assert cb.CB1_EFFICACY_ANANDAMIDE > cb.CB1_EFFICACY_THC
    assert cb.CB1_EFFICACY_THC > cb.CB1_EFFICACY_RIMONABANT
    # and it survives every published reading of THC's efficacy
    for value in (cb.CB1_EFFICACY_THC,) + cb.CB1_EFFICACY_THC_ALTERNATIVES:
        assert value < cb.CB1_EFFICACY_ANANDAMIDE


def test_rimonabant_is_zero_efficacy_and_that_is_flagged_as_a_simplification():
    assert cb.CB1_EFFICACY_RIMONABANT == 0.0
    ligands = {lig.name: lig for lig in cb.ligands_cb1()}
    assert ligands["rimonabant"].efficacy == 0.0


def test_every_ligand_reproduces_its_chosen_affinity_exactly():
    """The illustrative kon cannot leak into an affinity."""
    for affinity in cb.AFFINITY_CHOICES:
        for lig in cb.ligands_cb1(affinity=affinity):
            assert lig.koff / lig.kon == pytest.approx(
                cb.cb1_kd(lig.name, affinity), rel=1e-12
            )


def test_the_illustrative_kon_is_shared_with_the_modules_that_also_had_none():
    assert cb.ILLUSTRATIVE_KON == acetylcholine.ILLUSTRATIVE_KON
    assert cb.ILLUSTRATIVE_KON == glutamate.ILLUSTRATIVE_KON


def test_the_clearance_constants_are_the_shared_ones():
    assert cb.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE
    assert cb.TOOL_COMPOUND_KE == histamine.TOOL_COMPOUND_KE
    assert cb.ENDOCANNABINOID_KE == dopamine.DOPAMINE_KE
    assert cb.ENDOCANNABINOID_KE == noradrenaline.NORADRENALINE_KE


# ---------------------------------------------------------------------------
# The circuit
# ---------------------------------------------------------------------------


def test_the_network_is_p5_5s_loop_character_for_character():
    assert cb.CORTICAL_INHIBITION_WEIGHT == glutamate.CORTICAL_INHIBITION_WEIGHT
    assert cb.CORTICAL_RECURRENT_WEIGHT == glutamate.CORTICAL_RECURRENT_WEIGHT
    assert cb.CORTICAL_GAIN == glutamate.CORTICAL_GAIN
    assert np.array_equal(
        cb.cortical_network().connectivity, glutamate.cortical_network().connectivity
    )


def test_rule_1_and_rule_2_hold_for_that_network():
    network = cb.cortical_network()
    balanced = stability.balanced_baselines(network)
    assert balanced.baseline_drive() == pytest.approx(
        np.array([0.45, -0.4]), abs=1e-9
    )
    assert stability.contraction_number(network) == pytest.approx(0.9, abs=1e-9)


# ---------------------------------------------------------------------------
# The pharmacology, at the circuit level
# ---------------------------------------------------------------------------


def test_a_cb1_agonist_disinhibits_the_pyramidal_node():
    """Presynaptic CB1 removes GABA release, so the same interneuron firing
    delivers less inhibition and the pyramidal node rises."""
    _, _, result = _run(**{"2ag": BOLUS})
    assert _peak_deviation(result, row=0) > 0.01


def test_the_disinhibition_is_bounded_by_the_measured_fraction():
    """Even at a hugely saturating dose the pyramidal node cannot exceed the
    rate the 53%-strength synapse supports. A model in which a saturating
    agonist abolished inhibition would be asserting something no experiment
    supports."""
    network = cb.cortical_network()
    W = network.connectivity.copy()
    W[0, 1] = cb.modulated_weight(1.0)
    bounded = type(network)(nodes=network.nodes, connectivity=W)
    ceiling, residual = stability.steady_state(bounded)
    assert residual < 1e-6

    _, _, result = _run(**{"2ag": SATURATING})
    assert result["rates"][0].max() <= ceiling[0] + 1e-3


def test_the_agonists_rank_by_efficacy_at_a_shared_saturating_dose():
    """At a dose that saturates the pool for all three, what separates them
    is `efficacy`, and the order has to be the measured one."""
    peaks = {}
    for name in ("2ag", "anandamide", "thc"):
        _, _, result = _run(**{name: SATURATING})
        peaks[name] = _peak_deviation(result, row=0)
    assert peaks["2ag"] > peaks["anandamide"] > peaks["thc"] > 0.0


@pytest.mark.parametrize("thc_efficacy", (0.20, 0.22, 0.31))
def test_the_thc_result_survives_every_published_reading_of_its_efficacy(
    thc_efficacy,
):
    _, _, result = _run(thc_efficacy=thc_efficacy, **{"thc": SATURATING})
    assert 0.0 < _peak_deviation(result, row=0) < 0.06


@pytest.mark.parametrize("affinity", list(cb.AFFINITY_CHOICES))
def test_the_disinhibition_direction_survives_every_affinity_choice(affinity):
    _, _, result = _run(affinity=affinity, **{"2ag": SATURATING})
    assert _peak_deviation(result, row=0) > 0.01


def test_rimonabant_alone_is_inert_and_that_matches_the_slice_measurement():
    """[HoffmanLupica2000] applied SR141716A alone to the same slices and
    got 115 +/- 17% of control, i.e. no significant effect, because an acute
    slice has no ongoing endocannabinoid tone. This model predicts inertness
    for the same reason, so for once the standing no-endogenous-tone
    limitation has an experiment agreeing with it. It is still a limitation:
    an intact animal does have tone and rimonabant does have effects alone.
    """
    _, _, result = _run(**{"rimonabant": SATURATING})
    assert result["rates"] == pytest.approx(0.5, abs=1e-6)
    lower = cb.CB1_ANTAGONIST_ALONE_IPSC_FRACTION - 0.17
    upper = cb.CB1_ANTAGONIST_ALONE_IPSC_FRACTION + 0.17
    assert lower <= 1.0 <= upper  # the measurement includes "no change"


def test_rimonabant_reverses_the_endocannabinoid_response():
    _, _, agonist = _run(**{"2ag": BOLUS})
    _, _, both = _run(**{"2ag": BOLUS, "rimonabant": SATURATING})
    assert _peak_deviation(both, row=0) < 0.25 * _peak_deviation(agonist, row=0)


def test_the_presynaptic_node_moves_only_through_the_loop_and_moves_later():
    """Presynaptic inhibition of release does not change the presynaptic
    cell's own firing -- which is exactly why this receptor needed
    `connection_modulation` and not `coupling`. PFC_GABA moves only because
    the pyramidal node it receives excitation from has risen, so it must lag
    that node."""
    _, _, result = _run(**{"2ag": BOLUS})
    pyramidal = result["rates"][0] - 0.5
    interneuron = result["rates"][1] - 0.5
    assert interneuron.max() > 0.0
    assert np.argmax(interneuron) > np.argmax(pyramidal)


def test_a_cb1_agonist_on_the_glutamatergic_connection_would_go_the_other_way():
    """Not built (see the module's "what was deliberately not built"), and
    tested here so the sign consequence of leaving it out is written down:
    CB1 on the excitatory terminal would REDUCE the pyramidal node's drive
    to the interneuron and therefore act against the disinhibition this
    module models. The module over-states a CB1 agonist's net excitatory
    effect by whatever that would have removed."""
    _, _, gabaergic = _run(**{"2ag": SATURATING})
    _, _, glutamatergic = _run(
        connections=[cb.GLUTAMATERGIC_CONNECTION], **{"2ag": SATURATING}
    )
    assert _peak_deviation(gabaergic, row=0) > 0.0
    assert _peak_deviation(glutamatergic, row=1) < 0.0


# ---------------------------------------------------------------------------
# Shared-readout bookkeeping
# ---------------------------------------------------------------------------


def test_the_cortical_readout_is_the_shared_one():
    readouts = [
        cb.cortical_output_readout(),
        serotonin.cortical_output_readout(),
        noradrenaline.cortical_output_readout(),
        acetylcholine.cortical_output_readout(),
        glutamate.cortical_output_readout(),
        histamine.cortical_output_readout(),
    ]
    for readout in readouts[1:]:
        assert readout.name == readouts[0].name
        assert readout.weights == readouts[0].weights


def test_the_interneuron_readout_matches_p5_5s():
    assert (
        cb.cortical_interneuron_readout().weights
        == glutamate.cortical_interneuron_readout().weights
    )
