"""P5.7: the kappa and delta opioid receptors.

What is under test:

  * the CONSTANTS, and the four things about them a reader must not be
    allowed to forget -- naloxone is 1.8-fold mu-over-kappa selective,
    fentanyl is a 30% partial agonist at kappa, two cells of the affinity
    table are censored bounds rather than measurements, and the zeros in
    the efficacy table are measurements rather than the modelling
    simplification every module before this one had to flag;
  * the ANATOMY, which is three placements each forced by a named
    experiment, and one of them (kappa on a connection rather than a node)
    forced by three;
  * the CIRCUIT RESULT the brief asks for: kappa agonism moving the
    incentive-salience readout in the direction OPPOSITE to mu agonism, on
    the network the mu scenario has used since Track C1;
  * the RE-RUN: what adding the cross-affinities does to the established mu
    results, on both networks, quantitatively.
"""

import numpy as np
import pytest

from neurocascade import (
    acetylcholine,
    adenosine,
    cannabinoid,
    dopamine,
    histamine,
    noradrenaline,
    opioid as op,
    scenarios,
    serotonin,
    stability,
)
from neurocascade.engine import Simulation


T_END = 40.0
N_POINTS = 201

#: nM. Roughly 7x fentanyl's calibrated mu Kd of 1.346 nM: a saturating mu
#: exposure without being an absurd one.
FENTANYL_DOSE = 10.0

#: nM. Four orders of magnitude above it. NOT a plausible exposure; it is
#: how "not mu-selective in the limit" is turned into a measurement.
FENTANYL_LIMIT_DOSE = 1.0e5

#: nM. 500x U50488's kappa Ki of 0.2 nM.
U50488_DOSE = 100.0
NALOXONE_DOSE = 100.0


@pytest.fixture(scope="module")
def stabilized():
    return op.stabilized_reward_network()


def run_on(network, systems, rates0, t_end=T_END, **doses):
    sim = Simulation(network=network, receptor_systems=systems)
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, N_POINTS),
        rates0=rates0,
        method="LSODA",
    )
    return sim, result


def extreme_delta(readout, network, dosed, drug_free):
    """Signed furthest excursion of a dosed run from the drug-free run."""
    delta = op.readout_trace(readout, network, dosed) - op.readout_trace(
        readout, network, drug_free
    )
    return delta[np.argmax(np.abs(delta))]


# ---------------------------------------------------------------------------
# The four things a reader must not forget
# ---------------------------------------------------------------------------


def test_naloxone_is_not_a_mu_selective_antagonist():
    """The headline the brief exists for. [Toll1998]'s human cloned
    receptors put naloxone at 1.4 nM at mu and 2.5 nM at kappa: 1.79-fold,
    which is inside the spread of a single competition curve. Any naloxone
    exposure that blocks mu has blocked kappa."""
    assert op.ki("naloxone", "mu") == 1.4
    assert op.ki("naloxone", "kappa") == 2.5
    assert op.naloxone_mu_over_kappa_selectivity() == pytest.approx(1.786, abs=0.01)
    assert op.naloxone_mu_over_kappa_selectivity() < 2.0
    # And the guinea-pig brain column, from the same table, agrees that it
    # is single-digit-fold rather than orders of magnitude.
    assert op.naloxone_mu_over_kappa_selectivity("guinea_pig_brain") < 3.0
    # Delta is the one place naloxone really is selective.
    assert op.selectivity("naloxone", "mu", "delta") == pytest.approx(48.2, abs=0.1)


def test_fentanyl_is_a_partial_agonist_at_kappa_not_an_inert_occupier():
    """The half of "not mu-selective in the limit" a Ki table cannot show.
    [Toll1998] Table 7: fentanyl stimulates [35S]GTPyS at kappa to 30 +/- 4%
    of U69,593's maximum, and at delta to 86 +/- 19% of DPDPE's. So at high
    exposure fentanyl is not merely occupying the receptor that opposes its
    own mu action, it is activating it."""
    kappa, mu = op.fentanyl_is_a_partial_kappa_agonist()
    assert kappa == 0.30
    assert mu == 1.00
    assert 0.0 < kappa < mu
    assert op.efficacy("fentanyl", "delta") == 0.86
    selectivity = op.fentanyl_mu_selectivity()
    assert selectivity["kappa"] == pytest.approx(121.1, abs=0.1)
    assert selectivity["delta"] == pytest.approx(218.1, abs=0.1)


def test_two_affinities_are_censored_bounds_and_are_used_as_bounds():
    """[Toll1998] printed ">10,000" for U50488 at delta and for
    Leu-enkephalin at kappa. Instantiating a censored cell AT the bound is
    the smallest value consistent with the datum, so it MAXIMISES the
    modelled cross-activity: every "this ligand does nothing there"
    conclusion drawn from those two cells is therefore conservative."""
    censored = {
        (ligand, receptor)
        for ligand in op.LIGANDS
        for receptor in op.RECEPTORS
        if op.is_censored(ligand, receptor)
    }
    assert censored == {("U50488", "delta"), ("leu_enkephalin", "kappa")}
    assert op.ki("U50488", "delta") == op.ASSAY_CEILING_NM
    # Every non-censored value is strictly inside the assay.
    for ligand in op.LIGANDS:
        for receptor in op.RECEPTORS:
            if (ligand, receptor) not in censored:
                assert op.ki(ligand, receptor) < op.ASSAY_CEILING_NM
    # And the bound is a bound: no dispersion is recorded for those cells.
    assert op.KI_HUMAN_CHO_SD["U50488"]["delta"] is None
    assert op.KI_HUMAN_CHO_SD["leu_enkephalin"]["kappa"] is None


def test_the_antagonist_efficacies_are_measured_not_assumed():
    """The first time in this codebase that `efficacy=0.0` is a measurement.

    Every module from P5.1 to P5.6 had to flag a neutral-antagonist efficacy
    of 0.0 as a modelling simplification standing in for unmeasured inverse
    agonism. [Toll1998] ran naloxone, nor-BNI and naltrindole in the
    [35S]GTPyS functional assay at all three human receptors and each was
    'flat' at all three, and ran U50488 at mu and delta with the same
    result. Eleven measured zeros."""
    assert len(op.MEASURED_FLAT) == 11
    for ligand, receptor in op.MEASURED_FLAT:
        assert op.efficacy(ligand, receptor) == 0.0
    # Nothing is silently zero: every zero efficacy in the table is one of
    # the eleven measured ones.
    zeros = {
        (ligand, receptor)
        for ligand in op.LIGANDS
        for receptor in op.RECEPTORS
        if op.efficacy(ligand, receptor) == 0.0
    }
    assert zeros == set(op.MEASURED_FLAT)


# ---------------------------------------------------------------------------
# The rest of the constants
# ---------------------------------------------------------------------------


def test_dynorphin_is_barely_a_selective_kappa_ligand():
    """4.53-fold kappa over mu, and a full agonist at all three receptors in
    the same functional panel. The endogenous "kappa peptide" is not a
    selective ligand at equilibrium; its selectivity in vivo is a matter of
    where it is released, which is a Phase 6 property this model lacks."""
    assert op.selectivity("dynorphin_A", "kappa", "mu") == pytest.approx(4.53, abs=0.01)
    for receptor in op.RECEPTORS:
        assert op.efficacy("dynorphin_A", receptor) > 0.95


def test_leu_enkephalin_binds_mu_tightly_and_barely_activates_it():
    """Affinity and efficacy point different ways for one peptide at one
    receptor: Ki 7.4 nM at mu, within 3.5-fold of its delta Ki, but only
    11 +/- 4% of DAMGO's maximal stimulation there against 104 +/- 21% of
    DPDPE's at delta. A model carrying only Ki would get this backwards,
    which is why both [Toll1998] tables are used and not just Table 5."""
    assert op.ki("leu_enkephalin", "mu") == 7.4
    assert op.ki("leu_enkephalin", "delta") == 2.1
    assert op.selectivity("leu_enkephalin", "delta", "mu") < 4.0
    assert op.efficacy("leu_enkephalin", "mu") == 0.11
    assert op.efficacy("leu_enkephalin", "delta") == 1.04
    # Efficacies above 1 are printed as measured and not clipped: 104 +/- 21%
    # is indistinguishable from the reference agonist, and rounding it to
    # 1.0 would replace a measurement with a tidier number.
    assert op.efficacy("leu_enkephalin", "delta") > 1.0


def test_the_selective_antagonists_are_not_very_selective_at_equilibrium():
    """nor-BNI is 28.5-fold kappa-over-delta and naltrindole 31.5-fold
    delta-over-mu, both far short of the "highly selective" they are
    routinely called. nor-BNI's famous selectivity in vivo is a DURATION
    phenomenon that this module explicitly does not represent."""
    assert op.selectivity("nor_BNI", "kappa", "delta") == pytest.approx(28.5, abs=0.1)
    assert op.selectivity("nor_BNI", "kappa", "mu") == pytest.approx(105.0, abs=0.1)
    assert op.selectivity("naltrindole", "delta", "mu") == pytest.approx(31.5, abs=0.1)
    assert "duration" in op.__doc__


def test_the_preparation_disagreements_are_recorded_not_averaged():
    """[Toll1998] measured the same twenty-one cells in human CHO cells and
    in guinea-pig brain, which is a direct measurement of what a preparation
    is worth. Nineteen agree within 3.5-fold; naltrindole at mu misses by
    31.5-fold and dynorphin A at delta by 7.4-fold. The kappa column, which
    this module's headline results rest on, is the one they agree on best."""
    outliers = {("naltrindole", "mu"), ("dynorphin_A", "delta")}
    assert op.preparation_disagreement("naltrindole", "mu") == pytest.approx(
        31.5, abs=0.1
    )
    assert op.preparation_disagreement("dynorphin_A", "delta") == pytest.approx(
        7.36, abs=0.01
    )
    for ligand in op.LIGANDS:
        for receptor in op.RECEPTORS:
            if (ligand, receptor) in outliers:
                continue
            assert op.preparation_disagreement(ligand, receptor) < 3.5
    for ligand in op.LIGANDS:
        assert op.preparation_disagreement(ligand, "kappa") < 1.8
    # In the guinea-pig column naltrindole is barely delta-selective at all.
    assert op.selectivity(
        "naltrindole", "delta", "mu", "guinea_pig_brain"
    ) == pytest.approx(2.22, abs=0.01)


def test_the_cross_assay_check_that_licenses_mixing_two_papers():
    """The calibrated mu site is [Volpe2011]'s; the kappa and delta sites
    are [Toll1998]'s. The two laboratories' mu numbers for the same two
    ligands differ by 1.9-fold and 1.08-fold, which is what makes putting
    them in one simulation defensible rather than sloppy."""
    agreement = op.mu_assay_agreement()
    assert agreement["fentanyl"] == pytest.approx(0.520, abs=0.005)
    assert agreement["naloxone"] == pytest.approx(0.922, abs=0.005)
    for ratio in agreement.values():
        assert 0.4 < ratio < 2.5


# ---------------------------------------------------------------------------
# Derivation 3, and the checks that make it a derivation
# ---------------------------------------------------------------------------


def test_kappa_derivation_3_reproduces_the_radioligands_own_affinity():
    """Every kappa ligand shares [3H]diprenorphine's MEASURED kon from
    [Wang2021]. The check: that kon and the same paper's measured koff give
    a kinetic Kd of 0.178 nM, against [Toll1998]'s independently measured
    equilibrium Ki of 0.2 nM for diprenorphine at kappa-CHO. Two
    laboratories, two assay formats, 1.1-fold."""
    kinetic, equilibrium = op.kappa_kinetic_check()
    assert kinetic == pytest.approx(0.1785, abs=0.001)
    assert equilibrium == 0.2
    assert max(kinetic, equilibrium) / min(kinetic, equilibrium) < 1.2


def test_delta_derivation_3_uses_the_component_that_matches_equilibrium():
    """[Vaughn1989]'s association is biphasic, so a component has to be
    chosen and the choice has to be a rule rather than a preference. The
    slow component's kinetic Kd is 0.201 nM against the same paper's
    saturation Kd of 0.328 nM (1.6-fold); the fast component's is 0.0059 nM
    (56-fold off). The slow one is used."""
    slow, fast, saturation = op.delta_kinetic_check()
    assert slow == pytest.approx(0.2014, abs=0.001)
    assert fast == pytest.approx(0.00586, abs=0.0001)
    assert saturation / slow < 2.0
    assert saturation / fast > 50.0
    assert op.SITE_KON["delta"] == op.DELTA_KON_PCL_DPDPE


def test_every_ligand_reproduces_its_own_published_kd_at_every_site():
    """Whatever derivation 3 gets wrong, it cannot get the AFFINITY wrong:
    `koff = kon * Kd` reproduces the Kd exactly for any kon. That is why
    this module's conclusions are stated at equilibrium and never in terms
    of a rate."""
    for receptor, site in (
        ("kappa", op.receptor_site_kappa()),
        ("delta", op.receptor_site_delta()),
        ("mu", op.receptor_site_mu()),
    ):
        assert [lig.name for lig in site.ligands] == list(op.LIGANDS)
        for lig in site.ligands:
            assert lig.koff / lig.kon == pytest.approx(op.ki(lig.name, receptor))
            assert lig.efficacy == op.efficacy(lig.name, receptor)


def test_the_two_sites_use_their_own_measured_association_rates():
    """A ten-fold gap between the kappa and delta kon values, and it is a
    gap between two radioligands in two preparations rather than a property
    of the two receptors. Asserted so that nobody reads it as one."""
    kappa = op.receptor_site_kappa().ligands
    delta = op.receptor_site_delta().ligands
    assert all(lig.kon == op.KAPPA_KON_DIPRENORPHINE for lig in kappa)
    assert all(lig.kon == op.DELTA_KON_PCL_DPDPE for lig in delta)
    assert op.KAPPA_KON_DIPRENORPHINE / op.DELTA_KON_PCL_DPDPE == pytest.approx(
        9.8, abs=0.1
    )


def test_the_tool_compound_clearance_is_the_one_every_module_shares():
    assert op.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE
    assert op.TOOL_COMPOUND_KE == dopamine.TOOL_COMPOUND_KE
    assert op.TOOL_COMPOUND_KE == noradrenaline.TOOL_COMPOUND_KE
    assert op.TOOL_COMPOUND_KE == acetylcholine.TOOL_COMPOUND_KE
    assert op.TOOL_COMPOUND_KE == histamine.TOOL_COMPOUND_KE
    assert op.TOOL_COMPOUND_KE == adenosine.TOOL_COMPOUND_KE


def test_fentanyl_and_naloxone_keep_their_calibrated_pharmacokinetics():
    """The same drug has the same kinetics at all three of its receptors, so
    the two mu drugs carry Track A's calibrated `ke` here rather than the
    shared illustrative one."""
    calibrated = scenarios.mu_opioid_pk_models_calibrated()
    models = op.pk_models()
    by_name = dict(zip(op.LIGANDS, models))
    assert by_name["fentanyl"].ke == pytest.approx(calibrated[0].ke)
    assert by_name["naloxone"].ke == pytest.approx(calibrated[1].ke)
    # And the peptides clear far faster than any dosed small molecule.
    assert by_name["dynorphin_A"].ke == op.PEPTIDE_KE
    assert op.PEPTIDE_KE > 100 * op.TOOL_COMPOUND_KE


# ---------------------------------------------------------------------------
# The anatomy, which is where each receptor goes and why
# ---------------------------------------------------------------------------


def test_kappa_is_a_connection_modulator_and_has_no_coupling_entry():
    """[Svingos2001] found kappa labelling in the accumbens mainly
    PRESYNAPTIC, 51% of it co-expressing the dopamine transporter;
    [Devine1993] found intra-VTA U-50,488H inert; [Britt2008] measured the
    effect at the terminal. A presynaptic receptor scaling one identified
    synapse is `connection_modulation`, and giving it a `coupling` entry
    would assert that a kappa agonist drives the accumbens independently of
    whether dopamine is being released. Same argument as CB1's."""
    assert op.coupling_kappa() == {}
    modulation = op.connection_modulation_kappa()
    assert list(modulation) == [("NAc", "VTA_DA")]
    assert modulation[("NAc", "VTA_DA")] == op.KOR_CONNECTION_GAIN
    system = op.receptor_system_kappa()
    assert system.coupling == {}
    assert system.connection_modulation == modulation


def test_delta_sits_on_the_node_the_microdialysis_experiment_localised_it_to():
    """[Devine1993]: intra-VTA DPDPE raises ventral striatal dopamine
    dose-orderly, naltrindole blocks it, CTOP does not. That is the VTA
    GABAergic interneuron with a negative gain, the same disinhibition motif
    mu uses on the same node."""
    assert op.DELTA_NODE == "VTA_GABA"
    assert op.coupling_delta() == {"VTA_GABA": op.DELTA_MECHANISM_GAIN}
    assert op.DELTA_MECHANISM_GAIN < 0.0
    # The node is one mu already occupies, and that is the point.
    assert op.DELTA_NODE in scenarios.mu_opioid_multi_region_coupling()


def test_the_delta_gain_does_not_claim_a_potency_comparison_with_mu():
    """A2A's refusal, repeated. Matching delta's magnitude to mu's -1.72 on
    the shared node would assert that a unit of delta activation and a unit
    of mu activation move that interneuron equally, which nothing measured
    supports. [Devine1993]'s 100- to 1000-fold DPDPE/DAMGO concentration gap
    is a LIGAND potency statement, already carried by the Kd's."""
    assert op.DELTA_MECHANISM_GAIN == -1.0
    mu_gain = scenarios.mu_opioid_multi_region_coupling()["VTA_GABA"]
    assert abs(op.DELTA_MECHANISM_GAIN) != abs(mu_gain)
    # The potency difference [Devine1993] did measure lives in the Kd's, not
    # in the gain, and the module docstring records which paper said it.
    assert "1000-fold higher concentrations" in op.__doc__
    assert op.ki("leu_enkephalin", "delta") < op.ki("leu_enkephalin", "mu")


def test_the_kappa_gain_is_the_measured_maximal_suppression():
    """[Britt2008]: saturating BRL52537 cut single-pulse-evoked accumbal
    dopamine overflow to 28 +/- 5% of control. The engine forms
    `1 + gain * activation`, so the gain is -(1 - 0.28)."""
    assert op.KOR_MAX_SUPPRESSED_FRACTION == 0.72
    assert op.KOR_CONNECTION_GAIN == -op.KOR_MAX_SUPPRESSED_FRACTION
    # A saturating full agonist leaves 28% of the projection.
    assert op.modulated_mesolimbic_weight(1.0) == pytest.approx(2.0 * 0.28)
    # And with no drug on board, nothing changes at all.
    assert op.modulated_mesolimbic_weight(0.0) == 2.0


def test_a_kappa_agonist_can_neither_silence_nor_invert_the_projection():
    """Two failure modes a `connection_modulation` gain can have and this
    one does not. A gain below -1 would turn the excitatory mesolimbic
    weight inhibitory and nothing in the engine forbids it; a gain of -1
    would let a saturating agonist abolish a projection, which no experiment
    supports."""
    assert -1.0 < op.KOR_CONNECTION_GAIN < 0.0
    for activation in np.linspace(0.0, 1.05, 22):
        assert op.modulated_mesolimbic_weight(activation) > 0.0
    assert op.modulated_mesolimbic_weight(1.0) > 0.25 * 2.0


def test_the_network_is_the_mu_scenarios_own_weights_and_topology():
    """No new circuit is drawn. `stabilize()` multiplies every weight by ONE
    global scalar and re-solves the baselines; relative weights, topology,
    taus and gains are untouched."""
    hand = op.reward_network()
    assert hand.connectivity.shape == (7, 7)
    stabilized = op.stabilized_reward_network()
    assert [node.name for node in stabilized.nodes] == [
        node.name for node in hand.nodes
    ]
    scale = stabilized.connectivity[
        stabilized.index["NAc"], stabilized.index["VTA_DA"]
    ] / hand.connectivity[hand.index["NAc"], hand.index["VTA_DA"]]
    nonzero = hand.connectivity != 0.0
    assert np.allclose(
        stabilized.connectivity[nonzero] / hand.connectivity[nonzero], scale
    )
    for a, b in zip(stabilized.nodes, hand.nodes):
        assert a.tau == b.tau and a.gain == b.gain


def test_the_stabilized_network_has_the_certificate_the_others_do_not():
    """The reason this module cannot use Track C1's operating point. On the
    hand baselines VTA_DA sits saturated OFF, so the projection kappa
    modulates carries nothing; re-balancing the baselines alone leaves a
    contraction number of 2.63 and the loop does not settle. Only the fully
    stabilized network has a unique globally attracting fixed point."""
    hand = op.reward_network()
    rest, _ = stability.steady_state(hand)
    assert rest[hand.index["VTA_DA"]] < 1e-4  # saturated off

    balanced_only = stability.balanced_baselines(hand)
    assert stability.contraction_number(balanced_only) > 1.0

    stabilized = op.stabilized_reward_network()
    assert stability.contraction_number(stabilized) == pytest.approx(
        op.OPERATING_GAIN
    )
    assert stability.contraction_number(stabilized) < 1.0
    for initial in stability.default_initial_conditions(stabilized.n, count=4):
        rates, residual = stability.steady_state(stabilized, initial_rates=initial)
        assert residual < 1e-6
        assert np.allclose(rates, 0.5, atol=1e-6)


def test_a_presynaptic_modulator_of_a_silent_projection_does_nothing(stabilized):
    """The finding that forced the network choice, as a measurement rather
    than a sentence.

    Compared AT THE OPERATING POINT, which is what the claim is about: on
    Track C1's baselines VTA_DA settles saturated off, the mesolimbic
    projection transmits nothing, and the same kappa system at the same dose
    and above 99% occupancy moves the accumbal readout by less than 1e-4.
    On the stabilized network it moves it by a fifth of a unit. Nothing
    about the pharmacology differs between the two runs.

    (During Track C1's start-up transient VTA_DA is briefly non-zero and the
    modulator does have an effect, about 0.04. That is a transient, not an
    operating point, and it is why this test reads the endpoint.)"""
    hand = op.reward_network()
    systems = op.opioid_receptor_systems()
    readout = op.accumbal_drive_readout()

    _, hand_base = run_on(hand, systems, None)
    sim, hand_dosed = run_on(hand, systems, None, U50488=U50488_DOSE)
    kappa_occupancy = hand_dosed["occupancy"][sim.state_rows(1)][
        op.LIGANDS.index("U50488")
    ]
    assert kappa_occupancy[-1] > 0.99  # the receptor is still occupied at the end
    hand_effect = readout.score(hand, hand_dosed["rates"][:, -1]) - readout.score(
        hand, hand_base["rates"][:, -1]
    )
    assert abs(hand_effect) < 1e-4

    rest = op.rest_rates(stabilized)
    _, stable_base = run_on(stabilized, systems, rest)
    _, stable_dosed = run_on(stabilized, systems, rest, U50488=U50488_DOSE)
    stable_effect = readout.score(
        stabilized, stable_dosed["rates"][:, -1]
    ) - readout.score(stabilized, stable_base["rates"][:, -1])
    assert abs(stable_effect) > 0.1
    assert abs(stable_effect) > 1000 * abs(hand_effect)


# ---------------------------------------------------------------------------
# The result the brief asks for
# ---------------------------------------------------------------------------


def test_kappa_agonism_has_the_opposite_valence_to_mu_agonism(stabilized):
    """THE acceptance criterion. One network, one readout, two agonists:
    fentanyl raises incentive salience and U50488 lowers it.

    Neither sign is chosen here. Mu's is Track C1's disinhibition wiring;
    kappa's is [DiChiara1988], who gave freely-moving rats fentanyl and
    U-50,488 -- the two drugs in this test -- and measured accumbal dopamine
    going up under the first and down under the second."""
    systems = op.opioid_receptor_systems()
    rest = op.rest_rates(stabilized)
    readout = op.incentive_salience_readout()

    _, base = run_on(stabilized, systems, rest)
    _, mu_agonist = run_on(stabilized, systems, rest, fentanyl=FENTANYL_DOSE)
    _, kappa_agonist = run_on(stabilized, systems, rest, U50488=U50488_DOSE)

    mu_effect = extreme_delta(readout, stabilized, mu_agonist, base)
    kappa_effect = extreme_delta(readout, stabilized, kappa_agonist, base)

    assert mu_effect > 0.05
    assert kappa_effect < -0.05
    assert np.sign(mu_effect) == -np.sign(kappa_effect)


def test_the_endogenous_kappa_peptide_has_the_same_valence_as_the_tool_drug(
    stabilized,
):
    """U50488 is a synthetic tool compound; dynorphin A is what the receptor
    is for. Both lower the readout, and dynorphin does it despite being a
    full agonist at mu and delta too, because 1.7 nM at kappa against 7.7
    and 42.7 nM elsewhere still decides which receptor fills first."""
    systems = op.opioid_receptor_systems()
    rest = op.rest_rates(stabilized)
    readout = op.accumbal_drive_readout()

    _, base = run_on(stabilized, systems, rest)
    _, dyn = run_on(stabilized, systems, rest, dynorphin_A=1000.0)
    assert extreme_delta(readout, stabilized, dyn, base) < -0.05


def test_the_delta_agonist_has_the_same_valence_as_the_mu_agonist(stabilized):
    """[Devine1993] put delta and mu on the same side: intra-VTA DPDPE and
    intra-VTA DAMGO both raised ventral striatal dopamine. Leu-enkephalin,
    which is 3.5-fold delta-preferring and a full agonist there, raises the
    readout, so the opposite-valence result above is specific to kappa and
    not a generic property of adding a receptor."""
    systems = op.opioid_receptor_systems()
    rest = op.rest_rates(stabilized)
    readout = op.incentive_salience_readout()

    _, base = run_on(stabilized, systems, rest)
    _, enk = run_on(stabilized, systems, rest, leu_enkephalin=100.0)
    assert extreme_delta(readout, stabilized, enk, base) > 0.05


def test_the_kappa_antagonist_is_inert_alone_and_reverses_the_agonist(stabilized):
    """[Britt2008] measured both halves of this: nor-BNI (1 uM) "mostly
    restored the DA signal" after the agonist, and "alone had no direct
    effect on DA overflow". The model agrees with the second for a reason
    the experiment did not have -- P6.2 has supplied no endogenous dynorphin
    to displace -- and that difference is recorded rather than glossed."""
    systems = op.opioid_receptor_systems()
    rest = op.rest_rates(stabilized)
    readout = op.accumbal_drive_readout()

    _, base = run_on(stabilized, systems, rest)
    _, antagonist_alone = run_on(stabilized, systems, rest, nor_BNI=1000.0)
    assert abs(extreme_delta(readout, stabilized, antagonist_alone, base)) < 1e-6

    _, agonist = run_on(stabilized, systems, rest, U50488=U50488_DOSE)
    _, reversed_ = run_on(
        stabilized, systems, rest, U50488=U50488_DOSE, nor_BNI=1000.0
    )
    agonist_effect = extreme_delta(readout, stabilized, agonist, base)
    reversed_effect = extreme_delta(readout, stabilized, reversed_, base)
    assert abs(reversed_effect) < 0.2 * abs(agonist_effect)


def test_every_antagonist_alone_leaves_the_drug_free_run_exactly_alone(stabilized):
    """Three antagonists, three receptors, and eleven measured zeros in the
    efficacy table: none of them can move a model with no transmitter tone.
    Asserted as an identity rather than a tolerance, because with
    `activation == 0` the coupling term and the modulation factor are
    exactly 0 and exactly 1."""
    systems = op.opioid_receptor_systems()
    rest = op.rest_rates(stabilized)
    _, base = run_on(stabilized, systems, rest)
    for antagonist in ("naloxone", "nor_BNI", "naltrindole"):
        _, dosed = run_on(stabilized, systems, rest, **{antagonist: 1000.0})
        for readout in (
            op.incentive_salience_readout(),
            op.accumbal_drive_readout(),
            scenarios.analgesia_index_readout(),
            scenarios.arousal_index_readout(),
        ):
            assert abs(extreme_delta(readout, stabilized, dosed, base)) < 1e-9


# ---------------------------------------------------------------------------
# The re-run: what the cross-affinities change
# ---------------------------------------------------------------------------


def test_the_three_established_mu_results_survive_the_cross_affinities(stabilized):
    """The regression the brief asks for. Reward up, analgesia up, arousal
    down under fentanyl: every one of the multi-region scenario's
    qualitative claims still holds once fentanyl and naloxone are also
    binding kappa and delta."""
    rest = op.rest_rates(stabilized)
    for systems in (op.mu_only_receptor_systems(), op.opioid_receptor_systems()):
        _, base = run_on(stabilized, systems, rest)
        _, dosed = run_on(stabilized, systems, rest, fentanyl=FENTANYL_DOSE)
        assert (
            extreme_delta(op.incentive_salience_readout(), stabilized, dosed, base)
            > 0.1
        )
        assert (
            extreme_delta(
                scenarios.analgesia_index_readout(), stabilized, dosed, base
            )
            > 0.1
        )
        assert (
            extreme_delta(scenarios.arousal_index_readout(), stabilized, dosed, base)
            < -0.1
        )


def test_at_a_clinical_exposure_the_cross_affinities_barely_matter(stabilized):
    """Quantified, because "barely" is not a result. At 10 nM fentanyl --
    about 7x its mu Kd and 0.07x its kappa Kd -- the cross-affinities move
    incentive salience by under 5%, and the PAG and LC readouts not at all,
    since neither the kappa nor the delta system touches those nodes."""
    rest = op.rest_rates(stabilized)
    mu_only, full = op.mu_only_receptor_systems(), op.opioid_receptor_systems()
    readout = op.incentive_salience_readout()

    _, base_mu = run_on(stabilized, mu_only, rest)
    _, dosed_mu = run_on(stabilized, mu_only, rest, fentanyl=FENTANYL_DOSE)
    _, base_full = run_on(stabilized, full, rest)
    _, dosed_full = run_on(stabilized, full, rest, fentanyl=FENTANYL_DOSE)

    effect_mu = extreme_delta(readout, stabilized, dosed_mu, base_mu)
    effect_full = extreme_delta(readout, stabilized, dosed_full, base_full)
    assert abs(effect_full - effect_mu) / abs(effect_mu) < 0.05

    for other in (
        scenarios.analgesia_index_readout(),
        scenarios.arousal_index_readout(),
    ):
        a = extreme_delta(other, stabilized, dosed_mu, base_mu)
        b = extreme_delta(other, stabilized, dosed_full, base_full)
        assert b == pytest.approx(a, abs=1e-6)


def test_in_the_limit_fentanyl_works_against_itself(stabilized):
    """"Not mu-selective in the limit", as a number. Raise fentanyl to
    100 uM -- four orders of magnitude above a clinical exposure, and above
    its 84.8 nM kappa Ki -- and its own 30% partial agonism at kappa
    suppresses the mesolimbic projection it is trying to disinhibit. The
    reward effect is more than 20% smaller than the mu-only model predicts,
    while the PAG and LC readouts, which no kappa or delta system touches,
    are unchanged."""
    rest = op.rest_rates(stabilized)
    mu_only, full = op.mu_only_receptor_systems(), op.opioid_receptor_systems()
    readout = op.incentive_salience_readout()

    _, base_mu = run_on(stabilized, mu_only, rest)
    _, dosed_mu = run_on(stabilized, mu_only, rest, fentanyl=FENTANYL_LIMIT_DOSE)
    _, base_full = run_on(stabilized, full, rest)
    _, dosed_full = run_on(stabilized, full, rest, fentanyl=FENTANYL_LIMIT_DOSE)

    effect_mu = extreme_delta(readout, stabilized, dosed_mu, base_mu)
    effect_full = extreme_delta(readout, stabilized, dosed_full, base_full)
    assert effect_mu > 0.0 and effect_full > 0.0
    assert effect_full < effect_mu
    assert (effect_mu - effect_full) / effect_mu > 0.2

    unchanged = extreme_delta(
        scenarios.arousal_index_readout(), stabilized, dosed_full, base_full
    )
    assert unchanged == pytest.approx(
        extreme_delta(
            scenarios.arousal_index_readout(), stabilized, dosed_mu, base_mu
        ),
        abs=1e-6,
    )


def test_naloxone_reverses_kappa_at_a_concentration_close_to_the_mu_one(stabilized):
    """The brief's reversal sentence, measured. With each agonist dosed at
    7.5x its OWN Kd so the two competitions start from the same occupancy,
    the naloxone concentration that halves fentanyl's mu occupancy and the
    one that halves U50488's kappa occupancy are within a factor of two of
    each other -- as 1.4 nM and 2.5 nM require, and as a model treating
    naloxone as mu-selective would not predict."""
    rest = op.rest_rates(stabilized)
    systems = op.opioid_receptor_systems()
    kd_mu = 0.2599 / 0.1931  # scenarios.mu_opioid_ligands_calibrated() fentanyl
    kd_kappa = op.ki("U50488", "kappa")
    u50_index = op.LIGANDS.index("U50488")

    def peak(system_index, ligand_index, **doses):
        sim, result = run_on(stabilized, systems, rest, **doses)
        return result["occupancy"][sim.state_rows(system_index)][ligand_index].max()

    alone_mu = peak(0, 0, fentanyl=7.5 * kd_mu)
    alone_kappa = peak(1, u50_index, U50488=7.5 * kd_kappa)
    assert alone_mu > 0.8 and alone_kappa > 0.8

    ladder = [10 ** (k / 4.0) for k in range(0, 17)]

    def half_dose(system_index, ligand_index, agonist, alone):
        for dose in ladder:
            doses = dict(agonist)
            doses["naloxone"] = dose
            if peak(system_index, ligand_index, **doses) <= 0.5 * alone:
                return dose
        return None

    mu_dose = half_dose(0, 0, {"fentanyl": 7.5 * kd_mu}, alone_mu)
    kappa_dose = half_dose(
        1, u50_index, {"U50488": 7.5 * kd_kappa}, alone_kappa
    )
    assert mu_dose is not None and kappa_dose is not None
    assert 0.5 < kappa_dose / mu_dose < 4.0


def test_the_single_assay_configuration_gives_the_same_conclusions(stabilized):
    """The control for mixing two papers. Swapping the [Volpe2011]-based mu
    site for [Toll1998]'s own changes fentanyl's mu Kd by 1.9-fold and
    nothing qualitative: same three signs, same opposite-valence result."""
    rest = op.rest_rates(stabilized)
    systems = op.single_assay_receptor_systems()
    _, base = run_on(stabilized, systems, rest)
    _, fen = run_on(stabilized, systems, rest, fentanyl=FENTANYL_DOSE)
    _, kap = run_on(stabilized, systems, rest, U50488=U50488_DOSE)

    readout = op.incentive_salience_readout()
    assert extreme_delta(readout, stabilized, fen, base) > 0.1
    assert extreme_delta(readout, stabilized, kap, base) < -0.05
    assert (
        extreme_delta(scenarios.analgesia_index_readout(), stabilized, fen, base) > 0.1
    )
    assert (
        extreme_delta(scenarios.arousal_index_readout(), stabilized, fen, base) < -0.1
    )


# ---------------------------------------------------------------------------
# Housekeeping the module's own claims depend on
# ---------------------------------------------------------------------------


def test_the_readout_is_the_mu_scenarios_own(stabilized):
    """The opposite-valence result is a statement about the readout the mu
    scenario has always used, not about one chosen to produce it."""
    theirs = scenarios.incentive_salience_readout()
    ours = op.incentive_salience_readout()
    assert ours.name == theirs.name
    assert ours.weights == theirs.weights


def test_the_drug_free_run_is_identical_with_and_without_the_new_systems(
    stabilized,
):
    """Adding two receptor systems changes nothing until a drug is dosed:
    with zero activation the delta coupling term is exactly zero and the
    kappa modulation factor is exactly one."""
    rest = op.rest_rates(stabilized)
    _, mu_only = run_on(stabilized, op.mu_only_receptor_systems(), rest)
    _, full = run_on(stabilized, op.opioid_receptor_systems(), rest)
    assert np.allclose(mu_only["rates"], full["rates"], atol=1e-12)
    assert np.allclose(full["rates"], 0.5, atol=1e-9)


def test_the_mu_system_is_track_as_unchanged_one():
    """P5.7 adds receptors; it does not touch the calibrated mu site."""
    system = op.mu_receptor_system_calibrated()
    ligands = system.receptor_site.ligands
    assert [lig.name for lig in ligands] == ["fentanyl", "naloxone"]
    reference = scenarios.mu_opioid_ligands_calibrated()
    for ours, theirs in zip(ligands, reference):
        assert (ours.kon, ours.koff, ours.efficacy) == (
            theirs.kon,
            theirs.koff,
            theirs.efficacy,
        )
    assert system.coupling == scenarios.mu_opioid_multi_region_coupling()


def test_receptor_and_engine_modules_needed_no_change_for_p5_7():
    """The kappa mechanism reuses `connection_modulation`, which P5.6b's CB1
    module established, and the delta mechanism reuses `coupling`. Asserted
    by using the two shapes that already exist rather than by inspecting
    source: this module builds only plain `ReceptorSite`s."""
    from neurocascade.receptor import ReceptorSite

    for site in (
        op.receptor_site_kappa(),
        op.receptor_site_delta(),
        op.receptor_site_mu(),
    ):
        assert type(site) is ReceptorSite
        assert site.n == site.n_states == len(op.LIGANDS)
    # And CB1 is the precedent, not a coincidence: same construction.
    assert cannabinoid.CB1_CONNECTION_GAIN < 0.0
    assert op.KOR_CONNECTION_GAIN < 0.0


def test_unknown_ligands_receptors_and_preparations_are_errors():
    with pytest.raises(ValueError):
        op.ki("morphine", "mu")
    with pytest.raises(ValueError):
        op.ki("fentanyl", "sigma")
    with pytest.raises(ValueError):
        op.ki("fentanyl", "mu", "rat_brain")
    with pytest.raises(ValueError):
        op.efficacy("morphine", "mu")
