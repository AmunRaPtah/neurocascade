"""P5.5: glutamate NMDA and AMPA receptors.

Three things are under test here and they are of different kinds:

  * the CONSTANTS, and specifically the internal agreements between
    independently measured ones (MK-801's rate pair against two equilibrium
    binding constants, NBQX's Schild pA2 against its apparent Ki), which are
    the checks that would catch a transcription error;
  * the DOUBLE-COUNTING resolution, which is a set of identities rather than
    a numeric result: empty coupling, a gain pinned at 1, shares summing to
    1, and a drug-free run that reproduces the drug-free network exactly;
  * the CIRCUIT-LEVEL effect of ketamine-like NMDA antagonism, against
    [Homayoun2007]'s direction, including the conditions under which the
    model's version of it stops holding.

Every simulation here uses `method="LSODA"`; see
`test_the_two_integrators_agree` for why, and `engine.Simulation.run` for
what changed.
"""

import numpy as np
import pytest

from neurocascade import acetylcholine, dopamine, noradrenaline, serotonin, stability
from neurocascade import glutamate as glu
from neurocascade.engine import Simulation
from neurocascade.receptor import ChannelBlockingReceptorSite, Ligand


T_END = 60.0
KETAMINE_DOSE = 50000.0  # nM bolus, illustrative like every tool compound here


def _run(network, systems, t_end=T_END, rates0=None, n_points=241, **doses):
    sim = Simulation(network=network, receptor_systems=systems)
    if rates0 is None:
        rates0 = glu.rest_rates(network)
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, n_points),
        rates0=rates0,
        method="LSODA",
    )
    return sim, result


def _deviation(result, node_index):
    """Rate trace minus the drug-free rest rate, 0.5."""
    return result["rates"][node_index] - 0.5


def _extreme(trace):
    """The signed value of largest magnitude in a trace."""
    return trace[np.argmax(np.abs(trace))]


# ---------------------------------------------------------------------------
# The new receptor mechanism
# ---------------------------------------------------------------------------


def _one_ligand_occupancy(concentration, kd):
    """Equilibrium occupancy of one pool by one ligand."""
    quotient = concentration / kd
    return quotient / (1.0 + quotient)


def _toy_site():
    agonist = Ligand(name="agonist", kon=0.01, koff=1.0, efficacy=1.0)  # Kd 100
    blocker = Ligand(name="blocker", kon=0.01, koff=1.0, efficacy=0.0)  # Kd 100
    return ChannelBlockingReceptorSite(
        name="toy", ligands=[agonist], channel_blockers=[blocker]
    )


def test_a_channel_blocker_halves_activation_at_its_kd_without_displacing_the_agonist():
    """The defining property of NON-competitive block, and the reason ketamine
    and MK-801 are not in the orthosteric pool."""
    site = _toy_site()
    saturating_agonist = 1e6  # >> Kd, so the orthosteric pool is nearly full

    def equilibrium(blocker_concentration):
        """Each pool is one ligand against one pool, so `C/Kd / (1 + C/Kd)`
        each, and the site's own RHS is asserted to vanish there."""
        state = np.array(
            [
                _one_ligand_occupancy(saturating_agonist, 100.0),
                _one_ligand_occupancy(blocker_concentration, 100.0),
            ]
        )
        assert np.allclose(
            site.occupancy_rhs(state, [saturating_agonist, blocker_concentration]),
            0.0,
            atol=1e-8,
        )
        return state

    without = equilibrium(0.0)
    at_kd = equilibrium(100.0)

    # Activation halves...
    assert site.activation(at_kd) == pytest.approx(0.5 * site.activation(without), rel=0.02)
    # ...and the agonist's own occupancy does not move, which is what
    # "does not compete" means and what a competitive antagonist would break.
    bound_without, _ = site.split_states(without)
    bound_at_kd, _ = site.split_states(at_kd)
    assert bound_at_kd[0] == pytest.approx(bound_without[0], rel=1e-6)


def test_the_channel_blocker_pool_is_a_separate_finite_pool():
    site = _toy_site()
    state = np.array([0.4, 0.9])
    assert site.free_fraction(state) == pytest.approx(0.6)
    assert site.blocked_fraction(state) == pytest.approx(0.9)
    # A ligand count and a state count that agree: unlike P5.4's
    # desensitizing site, this one adds no state per ligand.
    assert site.n_states == site.n == 2


# ---------------------------------------------------------------------------
# The magnesium block
# ---------------------------------------------------------------------------


def test_the_mg_block_is_monotone_in_voltage_and_relieved_by_depolarization():
    voltages = np.linspace(-78.0, 32.0, 200)  # [ChiuCarter2022]'s fitted range
    for condition in glu.MG_BLOCK_FITS:
        unblocked = glu.mg_unblocked_fraction(voltages, condition=condition)
        assert np.all(np.diff(unblocked) > 0.0)
        assert 0.0 < unblocked[0] < unblocked[-1] < 1.0


def test_the_measured_midpoints_are_reproduced():
    """Each fit's own printed midpoint is where it crosses 0.5, which is the
    cheapest available check that the prefactor and slope were transcribed as
    a pair."""
    for condition, (_pre, _slope, midpoint, _mg) in glu.MG_BLOCK_FITS.items():
        assert glu.mg_unblocked_fraction(midpoint, condition=condition) == pytest.approx(
            0.5, abs=0.01
        )


def test_at_the_measured_resting_potential_the_block_removes_most_of_the_nmda_conductance():
    """The constraint the whole module sits under: at rest, an excitatory
    weight is essentially pure AMPA."""
    for condition, resting in glu.RESTING_POTENTIAL_MV.items():
        unblocked = float(glu.mg_unblocked_fraction(resting, condition=condition))
        assert unblocked < 0.04
        assert glu.nmda_share(resting, condition=condition) < 0.01


def test_the_three_ionic_conditions_are_ordered_by_their_magnesium():
    """More Mg2+, deeper block, at every voltage in the fitted range."""
    voltages = np.linspace(-78.0, 32.0, 50)
    classic = glu.mg_unblocked_fraction(voltages, condition="classic")  # 1.0 mM
    asleep = glu.mg_unblocked_fraction(voltages, condition="asleep")  # 0.8 mM
    awake = glu.mg_unblocked_fraction(voltages, condition="awake")  # 0.7 mM
    assert np.all(classic < asleep)
    assert np.all(asleep < awake)
    # And the spread between the extreme conditions is worth a number rather
    # than a shrug: at -60 mV the classic recipe blocks 1.4-fold harder.
    ratio = float(
        glu.mg_unblocked_fraction(-60.0, condition="awake")
        / glu.mg_unblocked_fraction(-60.0, condition="classic")
    )
    assert 1.3 < ratio < 1.5


def test_the_jahr_stevens_form_is_recorded_and_differs_from_the_one_used():
    """[JahrStevens1990]'s constants could not be verified against the paper
    (HTTP 403), so they are recorded and not used. This test says how far
    apart the two functions are rather than letting the choice be invisible."""
    voltages = np.linspace(-78.0, 32.0, 50)
    used = glu.mg_unblocked_fraction(voltages, condition="classic")  # 1.0 mM Mg2+
    canonical = glu.jahr_stevens_unblocked_fraction(voltages, mg_mM=1.0)
    # Same shape, same monotonicity, same limits...
    assert np.all(np.diff(canonical) > 0.0)
    # ...and the same order of magnitude everywhere, but not the same curve.
    # At -60 mV in 1 mM Mg2+ the canonical form is the DEEPER block by 1.2-fold
    # and at -40 mV by 1.4-fold, so using it in place of the measured fit would
    # shrink every NMDA effect in this module rather than change its sign.
    for voltage, expected in ((-60.0, 0.82), (-40.0, 0.71)):
        ratio = float(
            glu.jahr_stevens_unblocked_fraction(voltage, mg_mM=1.0)
            / glu.mg_unblocked_fraction(voltage, condition="classic")
        )
        assert ratio == pytest.approx(expected, rel=0.05)
        assert ratio < 1.0


# ---------------------------------------------------------------------------
# Shares: the arithmetic that makes the partition a partition
# ---------------------------------------------------------------------------


def test_the_two_receptor_shares_of_a_weight_sum_to_exactly_one():
    """Not approximately. This identity IS the double-counting resolution:
    the receptor layer divides an existing weight, it does not add to one."""
    for condition in glu.MG_BLOCK_FITS:
        for v in np.linspace(-90.0, 30.0, 25):
            assert (
                glu.nmda_share(v, condition=condition)
                + glu.ampa_share(v, condition=condition)
            ) == pytest.approx(1.0, abs=1e-15)


def test_the_nmda_share_grows_with_depolarization_and_never_exceeds_the_ratio():
    voltages = np.linspace(-90.0, 30.0, 100)
    shares = np.array([glu.nmda_share(v) for v in voltages])
    assert np.all(np.diff(shares) > 0.0)
    # The ceiling is the fully unblocked partition, ratio / (1 + ratio).
    ceiling = glu.NMDA_AMPA_CONDUCTANCE_RATIO / (1.0 + glu.NMDA_AMPA_CONDUCTANCE_RATIO)
    assert shares.max() < ceiling
    assert shares.max() == pytest.approx(ceiling, rel=0.1)


def test_the_nmda_ampa_ratio_is_the_measured_number_with_no_correction_applied():
    """[Myme2003]'s 27% is a PEAK-current ratio and its NMDA component decays
    over 69 ms against an AMPA EPSC's few ms, so the share of transmitted
    charge is larger. The direction of that bias is known and the size is
    not, so the module records it as a lower bound and applies no correction
    factor. This test is the guard against one appearing quietly."""
    assert glu.NMDA_AMPA_RATIO_IS_A_LOWER_BOUND is True
    assert glu.NMDA_AMPA_CONDUCTANCE_RATIO == glu.MYME2003_NMDA_AMPA_RATIO_PFC
    # The visual-cortex value from the same paper is the second reading, and
    # the two agree to 4%: the ratio is a property of cortical synapses, not
    # of prefrontal ones.
    assert glu.MYME2003_NMDA_AMPA_RATIO_VISUAL == pytest.approx(
        glu.MYME2003_NMDA_AMPA_RATIO_PFC, rel=0.05
    )


def test_the_instantiated_sites_carry_complementary_shares():
    nmda = glu.receptor_site_nmda()
    ampa = glu.receptor_site_ampa()
    assert nmda.synaptic_share + ampa.synaptic_share == pytest.approx(1.0, abs=1e-15)
    assert nmda.synaptic_share == pytest.approx(glu.nmda_share(), abs=1e-15)


def test_a_share_outside_the_unit_interval_is_refused():
    orthosteric, blockers = glu.ligands_nmda()
    with pytest.raises(ValueError):
        glu.GlutamatergicTransmissionSite(
            name="bad", ligands=orthosteric, synaptic_share=1.5, channel_blockers=blockers
        )


# ---------------------------------------------------------------------------
# Constants: the agreements between independently measured numbers
# ---------------------------------------------------------------------------


def test_mk801_is_the_first_derivation_one_ligand_in_a_phase_five_module():
    """Both rate constants measured ([Dravid2007] Table 5), so the affinity is
    a PREDICTION rather than an input, and two other sources measured it."""
    ligand = glu.ligand_mk801()
    kd_from_rates = ligand.koff / ligand.kon
    # [Dravid2007]'s own printed Kd for the same pair.
    assert kd_from_rates == pytest.approx(glu.DRAVID2007_MK801_KD, rel=0.08)
    # [Wong1986]'s independent [3H]MK-801 binding Kd in rat brain membranes,
    # a different laboratory, preparation and method.
    assert kd_from_rates == pytest.approx(glu.WONG1986_MK801_BINDING_KD, rel=0.20)
    # And the two published equilibrium constants agree with each other to 13%.
    assert glu.DRAVID2007_MK801_KD == pytest.approx(
        glu.WONG1986_MK801_BINDING_KD, rel=0.13
    )


def test_the_mk801_unblocking_rate_is_reported_as_wrong_rather_than_quietly_used():
    """[Huettner1988] measured recovery from MK-801 block at ~90 min at -70 to
    -80 mV; this module's koff is a 5.8 s time constant. The gap is ~650-fold
    and is the module's largest known error."""
    tau_model_min = 1.0 / glu.NMDA_MK801_KOFF
    ratio = glu.HUETTNER1988_MK801_RECOVERY_TAU_HYPERPOLARIZED / tau_model_min
    assert ratio == pytest.approx(650.0, rel=0.05)
    # The depolarized measurement is the one the model is closest to, and it
    # is still 14-fold away, so the disagreement is not an artefact of
    # picking the most hyperpolarized number.
    assert (
        glu.HUETTNER1988_MK801_RECOVERY_TAU_DEPOLARIZED / tau_model_min
    ) == pytest.approx(14.4, rel=0.05)


def test_nbqx_schild_and_apparent_ki_agree_and_kainate_is_not_spared():
    """[Randle1992] measured NBQX's affinity two ways in one paper."""
    kd_from_pa2 = 10.0 ** (9.0 - glu.RANDLE1992_NBQX_PA2_VS_AMPA)
    assert kd_from_pa2 == pytest.approx(glu.AMPA_KD_NBQX, rel=0.5)
    assert 1.0 < kd_from_pa2 / glu.AMPA_KD_NBQX < 1.6
    # "AMPA antagonist" at these concentrations means "AMPA AND kainate".
    assert glu.KAINATE_KD_NBQX / glu.AMPA_KD_NBQX == pytest.approx(1.24, rel=0.05)
    # Intact tissue is 14-fold less sensitive than the expression system, in
    # the same paper: the error bar on every affinity in this module.
    assert glu.AMPA_IC50_NBQX_HIPPOCAMPAL_SLICE / glu.AMPA_KD_NBQX == pytest.approx(
        14.3, rel=0.05
    )


def test_the_two_ampa_glutamate_potencies_from_one_paper_span_twenty_five_fold():
    """[Patneau1990] measured glutamate at two components of the same
    receptor population; this model has one state for it."""
    ratio = glu.AMPA_KD_GLUTAMATE / glu.AMPA_KD_GLUTAMATE_NONDESENSITIZING
    assert ratio == pytest.approx(25.3, rel=0.05)


def test_glutamate_is_two_hundred_fold_more_potent_at_nmda_than_at_ampa():
    """The headline contrast, from ONE laboratory, ONE preparation, and this
    project's species ([Patneau1990], mouse)."""
    assert glu.AMPA_KD_GLUTAMATE / glu.NMDA_KD_GLUTAMATE == pytest.approx(209.0, rel=0.05)


def test_the_three_ketamine_potencies_span_sixteen_fold_and_magnesium_explains_part():
    """[Kotermanski2009] in 0 and 1 mM Mg2+, and [Dravid2007] Mg2+-free at a
    different voltage."""
    assert glu.NMDA_KD_KETAMINE / glu.NMDA_KD_KETAMINE_MG_FREE == pytest.approx(
        16.2, rel=0.05
    )
    # The two Mg2+-free measurements, 26 mV apart, differ 10-fold, with the
    # more hyperpolarized one the tighter -- the direction open-channel block
    # requires.
    assert glu.NMDA_KD_KETAMINE_MG_FREE_DRAVID / glu.NMDA_KD_KETAMINE_MG_FREE == (
        pytest.approx(10.0, rel=0.05)
    )
    # The instantiated constant is the 1 mM Mg2+ one.
    assert glu.NMDA_KD_KETAMINE == glu.NMDA_SUBTYPE_IC50_KETAMINE_1MM_MG["2A"]


def test_every_affinity_is_reproduced_exactly_by_its_ligand():
    """Derivation 3's guarantee: only the approach timescale is illustrative."""
    orthosteric, blockers = glu.ligands_nmda()
    assert orthosteric[0].koff / orthosteric[0].kon == pytest.approx(
        glu.NMDA_KD_GLUTAMATE, rel=1e-12
    )
    ketamine, _mk801 = blockers
    assert ketamine.koff / ketamine.kon == pytest.approx(glu.NMDA_KD_KETAMINE, rel=1e-12)
    glutamate, nbqx = glu.ligands_ampa()
    assert glutamate.koff / glutamate.kon == pytest.approx(glu.AMPA_KD_GLUTAMATE, rel=1e-12)
    assert nbqx.koff / nbqx.kon == pytest.approx(glu.AMPA_KD_NBQX, rel=1e-12)


def test_the_orthosteric_pools_share_the_codebases_one_illustrative_kon():
    assert glu.ILLUSTRATIVE_KON == acetylcholine.ILLUSTRATIVE_KON
    # ...and the channel-blocker pool does not need it, because it has a
    # measured one.
    assert glu.NMDA_CHANNEL_KON != glu.ILLUSTRATIVE_KON
    assert glu.NMDA_CHANNEL_KON == pytest.approx(0.234, rel=1e-9)


def test_the_illustrative_clearance_constants_match_the_rest_of_the_codebase():
    for other in (
        serotonin.TOOL_COMPOUND_KE,
        dopamine.TOOL_COMPOUND_KE,
        noradrenaline.TOOL_COMPOUND_KE,
        acetylcholine.TOOL_COMPOUND_KE,
    ):
        assert glu.TOOL_COMPOUND_KE == other
    for other in (
        serotonin.SEROTONIN_KE,
        dopamine.DOPAMINE_KE,
        noradrenaline.NORADRENALINE_KE,
        acetylcholine.ACETYLCHOLINE_KE,
    ):
        assert glu.GLUTAMATE_KE == other


def test_one_molecule_clears_at_one_rate_across_both_systems():
    """`dose_vector(glutamate=...)` matches by name across systems, so both
    systems must give glutamate the same PK model."""
    systems = glu.glutamatergic_receptor_systems()
    rates = []
    for system in systems:
        for ligand, model in zip(system.receptor_site.ligands, system.pk_models):
            if ligand.name == "glutamate":
                rates.append(model.ke)
    assert len(rates) == 2
    assert rates[0] == rates[1] == glu.GLUTAMATE_KE


def test_the_cortical_readout_is_the_same_object_shape_as_every_other_module():
    mine = glu.cortical_output_readout()
    for other in (
        serotonin.cortical_output_readout(),
        noradrenaline.cortical_output_readout(),
        acetylcholine.cortical_output_readout(),
    ):
        assert mine.name == other.name
        assert mine.weights == other.weights


# ---------------------------------------------------------------------------
# The double-counting resolution, as identities
# ---------------------------------------------------------------------------


def test_no_glutamate_system_couples_to_node_drive():
    """The single most important assertion in this file. A non-empty
    `coupling` here would put glutamatergic transmission into the model twice:
    once as W[i,j]*r_j and once as gain*activation."""
    for system in glu.glutamatergic_receptor_systems():
        assert system.coupling == {}


def test_the_connection_gain_is_pinned_at_one_by_the_algebra():
    """`1 + gain*activation` has to be `1 - fraction removed`, so the gain is
    not a parameter. This module therefore introduces no illustrative
    coupling gain at all, unlike every P5 module before it."""
    assert glu.GLUTAMATE_CONNECTION_GAIN == 1.0
    site = glu.receptor_site_nmda()
    state = np.zeros(site.n_states)
    state[1] = 0.6  # ketamine occupancy of the channel pool
    removed = site.removed_weight_fraction(state)
    factor = 1.0 + glu.GLUTAMATE_CONNECTION_GAIN * site.activation(state)
    assert factor == pytest.approx(1.0 - removed, abs=1e-15)
    assert removed == pytest.approx(site.synaptic_share * 0.6, abs=1e-15)


def test_only_the_glutamatergic_connection_is_modulated():
    """The return connection is GABAergic and this module does not touch it.
    That asymmetry is what the signs of the connectivity matrix already mean,
    not a choice made here."""
    for system in glu.glutamatergic_receptor_systems():
        assert set(system.connection_modulation) == {glu.GLUTAMATERGIC_CONNECTION}
    network = glu.cortical_network()
    i = network.index["PFC_GABA"]
    j = network.index["PFC_Pyramidal"]
    assert network.connectivity[i, j] > 0.0  # excitatory: glutamatergic
    assert network.connectivity[j, i] < 0.0  # inhibitory: GABAergic, untouched


def test_with_no_drug_the_network_is_exactly_the_drug_free_one():
    """The regression guard for every result that predates this module: with
    no drug on board every activation is exactly zero, every modulation factor
    is exactly 1, and the nodes do not move off their rest rates."""
    network = glu.cortical_network()
    _sim, result = _run(network, glu.glutamatergic_receptor_systems())
    assert np.allclose(result["rates"], 0.5, atol=1e-9)


def test_the_circuit_layer_cannot_saturate_because_no_drive_is_added():
    """P5.3's rule 3, in the strongest form available: this module's receptor
    systems add nothing to any node's drive, at any dose, so the saturation
    budget is untouched."""
    network = glu.cortical_network()
    _sim, result = _run(
        network,
        glu.glutamatergic_receptor_systems(),
        t_end=30.0,
        n_points=61,
        ketamine=1.0e6,
        mk801=1.0e6,
        nbqx=1.0e6,
    )
    # A saturating dose of every antagonist in the module at once: the
    # glutamatergic weight is driven to essentially zero, and even then no
    # node reaches a rail, because nothing was added to any node's drive.
    for column in range(result["rates"].shape[1]):
        report = stability.check_saturation(network, result["rates"][:, column])
        assert report.ok, report.summary()


def test_the_network_satisfies_the_same_two_certificates_as_every_other_scenario():
    network = glu.cortical_network()
    assert stability.contraction_number(network) == pytest.approx(0.9, abs=1e-9)
    balanced = stability.balanced_baselines(network)
    for node, rebalanced in zip(network.nodes, balanced.nodes):
        assert node.baseline_input == pytest.approx(rebalanced.baseline_input, abs=1e-12)


# ---------------------------------------------------------------------------
# The circuit-level effect of ketamine-like NMDA antagonism
# ---------------------------------------------------------------------------


def test_ketamine_disinhibits_the_pyramidal_population():
    """[Homayoun2007]: MK-801 in awake rats reduced firing in ~69% of
    fast-spiking interneurons and increased it in 86% of regular-spiking
    (pyramidal) neurons in prefrontal cortex. This is a DIRECTION test, which
    is what that result is; nothing here is fitted to it."""
    network = glu.cortical_network()
    _sim, result = _run(
        network, [glu.receptor_system_nmda()], ketamine=KETAMINE_DOSE
    )
    pyramidal = _deviation(result, network.index["PFC_Pyramidal"])
    interneuron = _deviation(result, network.index["PFC_GABA"])

    assert pyramidal.max() > 0.01  # pyramidal firing UP
    assert interneuron.min() < -0.01  # interneuron firing DOWN
    assert np.all(pyramidal >= -1e-9)  # and never the other way round
    assert np.all(interneuron <= 1e-9)


def test_the_interneuron_moves_before_the_pyramidal_neuron_does():
    """[Homayoun2007] report that interneuron inhibition PRECEDED pyramidal
    excitation. In this model that ordering is not put in: the interneuron is
    postsynaptic to the blocked connection and the pyramidal node is one
    synapse further away."""
    network = glu.cortical_network()
    _sim, result = _run(
        network, [glu.receptor_system_nmda()], ketamine=KETAMINE_DOSE, n_points=241
    )
    t = result["t"]
    threshold = 1e-3
    first_pyramidal = t[np.argmax(np.abs(_deviation(result, 0)) > threshold)]
    first_interneuron = t[np.argmax(np.abs(_deviation(result, 1)) > threshold)]
    assert first_interneuron < first_pyramidal


def test_mk801_produces_the_same_direction_as_ketamine():
    """The drug [Homayoun2007] actually used, at the same site by the same
    mechanism."""
    network = glu.cortical_network()
    _sim, result = _run(network, [glu.receptor_system_nmda()], mk801=1000.0)
    assert _deviation(result, 0).max() > 0.01
    assert _deviation(result, 1).min() < -0.01


def test_the_effect_grows_monotonically_with_dose():
    network = glu.cortical_network()
    effects = []
    for dose in [0.0, 2000.0, 10000.0, 50000.0, 250000.0]:
        _sim, result = _run(
            network, [glu.receptor_system_nmda()], t_end=30.0, n_points=61, ketamine=dose
        )
        effects.append(_deviation(result, 0).max())
    assert effects[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(effects) > 0.0)


def test_the_effect_is_a_monotone_function_of_the_operating_voltage():
    """The module's most load-bearing prediction: NMDA's leverage on the
    circuit is set by the Mg2+ block, so it collapses at resting potential and
    grows with depolarization. Nothing here is free -- the voltage dependence
    comes entirely from [ChiuCarter2022]'s measured fit."""
    network = glu.cortical_network()
    voltages = [-82.7, -70.0, -60.0, -50.0, -40.0, -20.0, 0.0]
    effects = []
    for v in voltages:
        _sim, result = _run(
            network,
            [glu.receptor_system_nmda(v_mV=v)],
            t_end=30.0,
            n_points=61,
            ketamine=KETAMINE_DOSE,
        )
        effects.append(_deviation(result, 0).max())
    assert np.all(np.diff(effects) > 0.0)
    # At the measured resting potential the effect is an order of magnitude
    # smaller than at the illustrative operating point, and at 0 mV it is
    # twice as large: a 27-fold range, all of it from the Mg2+ block.
    assert effects[0] < 0.002
    assert effects[-1] / effects[0] > 20.0


def test_the_direction_survives_the_whole_measured_voltage_range_and_all_three_baths():
    """The headline result is asserted across the measured range rather than
    at `ILLUSTRATIVE_OPERATING_VOLTAGE`, because that constant is the one
    illustrative number in the module's receptor arithmetic."""
    network = glu.cortical_network()
    for condition in glu.MG_BLOCK_FITS:
        for v in (-78.0, -40.0, 32.0):
            _sim, result = _run(
                network,
                [glu.receptor_system_nmda(v_mV=v, condition=condition)],
                t_end=20.0,
                n_points=41,
                ketamine=KETAMINE_DOSE,
            )
            assert _deviation(result, 0).max() > 0.0
            assert _deviation(result, 1).min() < 0.0


def test_both_nodes_get_the_same_nmda_share_despite_the_measurement_that_says_otherwise():
    """[Bogart2018] measured the NMDA:AMPA ratio 8-10 fold SMALLER at
    fast-spiking interneurons than at pyramidal cells, i.e. the opposite of
    what would make [Homayoun2007]'s result easy. The model does not encode a
    per-target share, and the disinhibition therefore comes from the loop's
    topology; this test pins both halves of that."""
    for _input, (fsi, pyramidal) in glu.BOGART2018_FSI_VS_PYRAMIDAL_RATIO.items():
        assert fsi < pyramidal
    site = glu.receptor_site_nmda()
    assert isinstance(site.synaptic_share, float)  # one share, not a per-node map
    systems = glu.glutamatergic_receptor_systems()
    assert all(len(s.connection_modulation) == 1 for s in systems)


def test_recurrent_excitation_can_reverse_the_direction_and_where():
    """The honest boundary on the result. In a two-population loop the only
    glutamatergic connection is the one onto the interneuron, so NMDA block
    disinhibits unconditionally. Add recurrent pyramidal excitation, which
    NMDA block also removes, and the two effects oppose. This measures where
    the balance tips instead of asserting it cannot."""
    connections = [glu.GLUTAMATERGIC_CONNECTION, glu.RECURRENT_EXCITATION_CONNECTION]

    def peak_pyramidal_deviation(recurrent):
        network = stability.balanced_baselines(
            glu.cortical_network(recurrent_excitation=recurrent)
        )
        _sim, result = _run(
            network,
            [glu.receptor_system_nmda(connections=connections)],
            t_end=30.0,
            n_points=61,
            rates0=np.full(network.n, 0.5),
            ketamine=KETAMINE_DOSE,
        )
        return _extreme(_deviation(result, 0)), stability.contraction_number(network)

    # Wherever P3.4's global-uniqueness certificate still holds, the
    # disinhibition direction holds with it.
    for recurrent in (0.0, 0.1, 0.2):
        deviation, contraction = peak_pyramidal_deviation(recurrent)
        assert contraction < 1.0
        assert deviation > 0.0

    # It reverses only well outside that certificate.
    deviation, contraction = peak_pyramidal_deviation(0.8)
    assert deviation < 0.0
    assert contraction > 1.0


# ---------------------------------------------------------------------------
# Competitive versus uncompetitive: the reason for two mechanisms
# ---------------------------------------------------------------------------


def _competitive_equilibrium(kds, concentrations):
    """Analytic occupancies of one competitive pool."""
    quotients = np.asarray(concentrations, dtype=float) / np.asarray(kds, dtype=float)
    return quotients / (1.0 + quotients.sum())


def test_the_competitive_pool_matches_the_analytic_equilibrium():
    """A guard that `GlutamatergicTransmissionSite` changed what is DONE with
    the binding arithmetic and not the arithmetic itself: the same
    concentrations through the plain `ReceptorSite` every other receptor in
    this codebase uses give the same answer."""
    site = glu.receptor_site_ampa()
    plain = glu.plain_ampa_binding_site()
    kds = [glu.AMPA_KD_GLUTAMATE, glu.AMPA_KD_NBQX]
    concentrations = [1.0e6, 500.0]
    expected = _competitive_equilibrium(kds, concentrations)

    assert np.allclose(site.occupancy_rhs(expected, concentrations), 0.0, atol=1e-12)
    assert np.allclose(plain.occupancy_rhs(expected, concentrations), 0.0, atol=1e-12)


def test_glutamate_reverses_nbqx_but_not_ketamine():
    """The whole reason the two antagonists are represented differently.
    NBQX competes for the agonist site ([Randle1992]'s Schild slopes near
    unity), so ambient glutamate displaces it; ketamine binds in the channel
    lumen and does not compete, so ambient glutamate does nothing to it."""
    ampa = glu.receptor_site_ampa()
    nbqx_block = []
    for glutamate in (0.0, 1.0e5, 1.0e6, 5.0e6):
        state = _competitive_equilibrium(
            [glu.AMPA_KD_GLUTAMATE, glu.AMPA_KD_NBQX], [glutamate, 500.0]
        )
        assert np.allclose(ampa.occupancy_rhs(state, [glutamate, 500.0]), 0.0, atol=1e-8)
        nbqx_block.append(ampa.blocked_transmission_fraction(state))
    assert np.all(np.diff(nbqx_block) < 0.0)
    assert nbqx_block[-1] < 0.5 * nbqx_block[0]

    nmda = glu.receptor_site_nmda()
    ketamine_block = []
    for glutamate in (0.0, 1.0e5, 1.0e6, 5.0e6):
        # The channel pool's equilibrium does not involve the orthosteric
        # concentrations at all, which is the point: it is written out
        # independently of the glutamate occupancy beside it.
        state = np.array(
            [
                _one_ligand_occupancy(glutamate, glu.NMDA_KD_GLUTAMATE),
                _one_ligand_occupancy(KETAMINE_DOSE, glu.NMDA_KD_KETAMINE),
                0.0,  # no MK-801 on board
            ]
        )
        concentrations = [glutamate, KETAMINE_DOSE, 0.0]
        assert np.allclose(nmda.occupancy_rhs(state, concentrations), 0.0, atol=1e-8)
        ketamine_block.append(nmda.blocked_transmission_fraction(state))
    assert np.allclose(ketamine_block, ketamine_block[0], atol=1e-12)


def test_an_antagonist_alone_is_not_inert_here_unlike_every_module_before_this_one():
    """Every scenario from P5.1 to P5.4 had to record that its antagonists are
    inert without endogenous transmitter tone, because there was no agonist
    for them to displace. Glutamate antagonists are not inert, and the reason
    is the double-counting resolution: the transmission they remove is already
    in the connectivity weight, so it does not need a dose to exist."""
    network = glu.cortical_network()
    _sim, result = _run(network, [glu.receptor_system_ampa()], nbqx=1000.0)
    assert _deviation(result, 0).max() > 0.1
    assert _deviation(result, 1).min() < -0.1


def test_blocking_ampa_moves_the_circuit_far_more_than_blocking_nmda():
    """A direct consequence of the Mg2+ block: at any physiological operating
    potential AMPA receptors carry most of the weight, so an AMPA antagonist
    has most of the leverage. This is what makes ketamine's small effect the
    expected one rather than a disappointing one."""
    network = glu.cortical_network()
    _sim, ampa = _run(network, [glu.receptor_system_ampa()], t_end=30.0, n_points=61,
                      nbqx=1000.0)
    _sim2, nmda = _run(network, [glu.receptor_system_nmda()], t_end=30.0, n_points=61,
                       ketamine=KETAMINE_DOSE)
    assert _deviation(ampa, 0).max() > 5.0 * _deviation(nmda, 0).max()


def test_the_two_systems_compose_multiplicatively_and_the_cross_term_is_small():
    """`engine.Simulation` composes several systems' modulation factors
    multiplicatively; the module docstring writes the partition additively.
    They differ by exactly the cross term, and it is only nonzero when an NMDA
    blocker and an AMPA blocker are on board at once."""
    nmda = glu.receptor_site_nmda()
    ampa = glu.receptor_site_ampa()
    nmda_state = np.array([0.0, 0.8, 0.0])  # ketamine at 80% of the channel pool
    ampa_state = np.array([0.0, 0.9])  # NBQX at 90% of the orthosteric pool

    removed_nmda = nmda.removed_weight_fraction(nmda_state)
    removed_ampa = ampa.removed_weight_fraction(ampa_state)
    multiplicative = (1.0 - removed_nmda) * (1.0 - removed_ampa)
    additive = 1.0 - removed_nmda - removed_ampa
    cross_term = multiplicative - additive
    assert cross_term == pytest.approx(removed_nmda * removed_ampa, abs=1e-15)
    assert 0.0 < cross_term < 0.07

    # With only one blocker present the two forms are identical, which is the
    # case every test above runs in.
    assert (1.0 - removed_nmda) * 1.0 == pytest.approx(1.0 - removed_nmda, abs=1e-15)


# ---------------------------------------------------------------------------
# The integrator
# ---------------------------------------------------------------------------


def test_the_two_integrators_agree():
    """`engine.Simulation.run` gained a `method` argument for P5.5 because a
    MEASURED association rate constant makes this system stiff. The default is
    still RK45 and nothing before this module changes; this asserts that the
    stiff solver is a statement about the integrator and not about the model.
    The dose and span are small so the explicit method stays affordable."""
    network = glu.cortical_network()
    sim = Simulation(network=network, receptor_systems=[glu.receptor_system_nmda()])
    kwargs = dict(
        doses=sim.dose_vector(ketamine=2000.0),
        t_span=(0.0, 5.0),
        t_eval=[5.0],
        rates0=glu.rest_rates(network),
    )
    explicit = sim.run(**kwargs)
    stiff = sim.run(method="LSODA", **kwargs)
    assert np.allclose(explicit["rates"], stiff["rates"], atol=1e-6)
    assert Simulation.DEFAULT_METHOD == "RK45"
