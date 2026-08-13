"""Tests for the P5.2 dopamine D1 / direct-pathway receptor system.

`dopamine.py`'s module docstring carries the full source list and the
derivation of every constant; each test below restates the published figure
or the stated design rule it checks against, so a silent edit to the module
cannot pass.

Three things are being checked, and they are different in kind:

  * the BINDING layer against published numbers ([Sahlholm2025],
    [Agren2021], [Billard1984], [Reader1989]);
  * the CIRCUIT layer against the three stated illustrative-parameter rules
    (unit weights, rest at maximum sensitivity, no saturation) -- not
    against literature, because it has none;
  * the PHARMACOLOGY against the direct/indirect model's qualitative
    predictions ([Albin1989], [Gerfen1990], [Kravitz2010]).
"""

import numpy as np

from neurocascade import dopamine, scenarios, serotonin, stability
from neurocascade.engine import ReceptorSystem, Simulation
from neurocascade.receptor import ReceptorSite

T_SPAN = (0.0, 20.0)
T_EVAL = np.linspace(0.0, 20.0, 400)

#: nM. The haloperidol concentration the pharmacology tests below use, and
#: the reason it is not larger: haloperidol's measured kon is 2.13 nM^-1
#: min^-1 [Sykes2017], so at 2000 nM its pseudo-first-order association
#: rate is ~4300 min^-1 and RK45 spends twenty times longer on the same
#: qualitative answer. `tests/test_d2_scenario_calibrated.py` records the
#: same trade for the same reason. At 200 nM haloperidol is still 650x its
#: own Kd (0.305 nM) and still takes the majority of the shared D2 pool
#: away from 1000 nM dopamine, which is what these tests need.
HALOPERIDOL_DOSE_nM = 200.0


def by_name(ligands):
    return {ligand.name: ligand for ligand in ligands}


def kd(ligand):
    return ligand.koff / ligand.kon


def equilibrium_occupancy(ligand, concentration):
    """The single-ligand equilibrium `C / (C + Kd)`, verified to BE the
    fixed point of `ReceptorSite.occupancy_rhs` rather than assumed.

    Checking the algebraic fixed point instead of integrating to it keeps
    this exact and cheap even for SCH-23390, whose 178-minute residence
    time makes numerical equilibration take hundreds of simulated minutes.
    """
    occupancy = concentration / (concentration + kd(ligand))
    site = ReceptorSite(name="site", ligands=[ligand])
    residual = site.occupancy_rhs([occupancy], [concentration])[0]
    assert abs(residual) < 1e-12
    return occupancy


def build_simulation(network=None, receptor_systems=None):
    network = dopamine.cortico_basal_ganglia_network() if network is None else network
    systems = (
        dopamine.dopamine_receptor_systems()
        if receptor_systems is None
        else receptor_systems
    )
    return Simulation(network=network, receptor_systems=systems)


def locomotor_trace(sim, **doses):
    """Locomotor readout over time, started from the network's rest state.

    Starting at `rest_rates()` rather than at zero rates is what makes the
    drug-free trace exactly flat, so every deviation below is the drug and
    not the start-up transient (see `dopamine.rest_rates`).
    """
    readout = dopamine.locomotor_activity_readout()
    result = sim.run(
        dose_concentrations=sim.dose_vector(**doses),
        t_span=T_SPAN,
        t_eval=T_EVAL,
        rates0=dopamine.rest_rates(sim.network),
    )
    scores = np.array(
        [
            readout.score(sim.network, result["rates"][:, i])
            for i in range(len(T_EVAL))
        ]
    )
    return result, scores


# ---------------------------------------------------------------------------
# The binding layer: does the module reproduce the published numbers?
# ---------------------------------------------------------------------------


def test_d1_rate_constants_reproduce_the_published_affinities():
    # For a single ligand at equilibrium occ = C / (C + koff/kon), so
    # occupancy at C = Kd must be exactly 0.5, and that Kd must land on the
    # published affinity or the conversion from the paper's units into this
    # codebase's is wrong.
    #   dopamine       pKd 5.969 -> Kd 1074 nM  [Sahlholm2025], hD1/GIRK/oocyte
    #   dihydrexidine  pKd 7.002 -> Kd 99.5 nM  [Sahlholm2025], same table
    #   SCH-23390      Kd 0.53 nM               [Billard1984], rat striatum
    ligands = by_name(dopamine.d1_dopamine_ligands())

    # Every ligand's own Kd is where its occupancy is a half, by definition.
    for ligand in ligands.values():
        assert abs(equilibrium_occupancy(ligand, kd(ligand)) - 0.5) < 1e-12

    # ...and that Kd has to match the published affinity. For the two
    # agonists the rate constants and the pKd are INDEPENDENT entries in
    # [Sahlholm2025]'s Table 1, so they only have to agree, and the
    # paper's own two numbers differ by 0.5%; SCH-23390's Kd is the input
    # to derivation 3 and must come back out exactly.
    published = {
        "dopamine": 10.0 ** (9.0 - dopamine.D1_PKD_DOPAMINE),
        "dihydrexidine": 10.0 ** (9.0 - 7.002),
        "SCH-23390": dopamine.D1_KD_SCH23390,
    }
    tolerances = {"dopamine": 0.01, "dihydrexidine": 0.01, "SCH-23390": 1e-12}
    for name, affinity in published.items():
        assert abs(kd(ligands[name]) - affinity) < tolerances[name] * affinity


def test_dopamine_d1_unit_conversion_is_the_published_rate_constants():
    # 122325 M^-1 s^-1 and 0.132 s^-1 [Sahlholm2025], into this codebase's
    # nM^-1 min^-1 and min^-1.
    for constant, published in (
        (dopamine.D1_KON_DOPAMINE, 122325 * 60 * 1e-9),
        (dopamine.D1_KOFF_DOPAMINE, 0.132 * 60),
        (dopamine.D1_KON_DIHYDREXIDINE, 952419 * 60 * 1e-9),
        (dopamine.D1_KOFF_DIHYDREXIDINE, 0.095 * 60),
    ):
        assert abs(constant - published) < 1e-12 * published


def test_dihydrexidine_is_a_strong_partial_agonist_not_a_full_one():
    # [Sahlholm2025] Table 1: Emax 0.808 +/- 0.044 of the response to 30 uM
    # dopamine in the same oocyte. The textbook shorthand "full D1 agonist"
    # hides a ~19% deficit that Ligand.efficacy carries directly.
    ligands = by_name(dopamine.d1_dopamine_ligands())
    assert ligands["dihydrexidine"].efficacy == 0.808
    assert 0.0 < ligands["dihydrexidine"].efficacy < ligands["dopamine"].efficacy
    # SCH-23390 is the antagonist of the set; its zero is a documented
    # modelling simplification (inverse agonism is not representable
    # without a constitutive-activity term), not a measured Emax.
    assert ligands["SCH-23390"].efficacy == 0.0


def test_dopamine_binds_d2_far_more_tightly_than_d1_in_matched_assays():
    # The one prediction in this module that is free of illustrative
    # parameters. Both numbers are koff/kon from GIRK-current kinetics in
    # Xenopus oocytes expressing the human receptor, from one laboratory:
    #   D2  koff 0.197 s^-1, kon 9.70e7 M^-1 s^-1  -> Kd 2.03 nM  [Agren2021]
    #   D1  koff 0.132 s^-1, kon 1.22e5 M^-1 s^-1  -> Kd 1079 nM  [Sahlholm2025]
    # i.e. dopamine needs ~500x more of itself to occupy D1 than D2. This is
    # the quantitative content of "tonic dopamine engages D2, phasic
    # dopamine is needed for D1".
    d1_dopamine = by_name(dopamine.d1_dopamine_ligands())["dopamine"]
    d2_dopamine = by_name(scenarios.d2_dopamine_ligands_calibrated())["dopamine"]

    assert abs(kd(d2_dopamine) - 2.03) < 0.01
    assert abs(kd(d1_dopamine) - 1079.0) < 1.0
    ratio = kd(d1_dopamine) / kd(d2_dopamine)
    assert 500.0 < ratio < 560.0

    # At a tonic-scale dopamine concentration the two sites are in
    # completely different regimes; at [Garris1994]'s ~250 nM phasic
    # transient peak the D1 site is finally engaged.
    assert equilibrium_occupancy(d2_dopamine, 20.0) > 0.9
    assert equilibrium_occupancy(d1_dopamine, 20.0) < 0.02
    assert equilibrium_occupancy(d1_dopamine, 250.0) > 0.18


def test_sch23390_derived_koff_is_bracketed_by_the_independent_observation():
    # Derivation 3: koff = (dopamine's measured D1 kon) * (SCH-23390's
    # measured Kd) = 0.0073395 * 0.53 [Sahlholm2025], [Billard1984].
    ligands = by_name(dopamine.d1_dopamine_ligands())
    sch = ligands["SCH-23390"]
    assert sch.kon == dopamine.D1_KON_DOPAMINE
    assert abs(sch.koff - dopamine.D1_KON_DOPAMINE * dopamine.D1_KD_SCH23390) < 1e-15

    # The check that makes the assumption falsifiable: [Reader1989] reports
    # 40-60% of [3H]SCH23390 binding displaced 2 h after initiating
    # dissociation in rabbit membranes, which brackets the half-life at
    # 120*ln2 / -ln(0.6) = 163 min (40% gone) to 120*ln2 / -ln(0.4) = 91 min
    # (60% gone). The derived value is 178 min: ~9% beyond the slow end.
    # Close enough that derivation 3 is not contradicted, far enough that
    # this is a sanity check and not a calibration -- and it is a third
    # species, which is why no number is taken from it.
    half_life = np.log(2.0) / sch.koff
    slow_end = 120.0 * np.log(2.0) / -np.log(0.6)
    fast_end = 120.0 * np.log(2.0) / -np.log(0.4)
    assert abs(half_life - 178.19) < 0.05
    assert 88.0 < fast_end < 92.0
    assert 160.0 < slow_end < 165.0
    assert half_life < 1.15 * slow_end


def test_sch23390_reverses_by_not_letting_go_rather_than_by_excess():
    # The third distinct antagonist-reversal mechanism in this codebase.
    # naloxone (koff 2.375 min^-1) is FASTER off than the agonist it
    # reverses and needs molar excess; WAY-100635 (koff 0.0078) and
    # SCH-23390 (koff 0.00389) are near-irreversible on this timescale.
    ligands = by_name(dopamine.d1_dopamine_ligands())
    sch = ligands["SCH-23390"]
    assert sch.koff < ligands["dihydrexidine"].koff / 1000.0
    assert sch.koff < ligands["dopamine"].koff / 1000.0
    way = by_name(serotonin.ligands_5ht1a())["WAY-100635"]
    assert sch.koff < way.koff  # slower still than P5.1's slow antagonist


def test_the_two_illustrative_pk_constants_are_shared_rather_than_invented():
    # One dopamine dose is applied to BOTH receptor systems by name, so the
    # two systems must clear it at the same rate or the model describes one
    # molecule with two elimination rate constants.
    assert dopamine.DOPAMINE_KE == scenarios.d2_pk_models_calibrated()[0].ke
    d1_models = dopamine.d1_pk_models()
    assert d1_models[0].ke == dopamine.DOPAMINE_KE
    # dihydrexidine and SCH-23390 share ONE illustrative ke rather than two
    # invented ones, and it is deliberately the same number P5.1 gave its
    # four tool compounds for the same stated reason (see "Searched and
    # empty" in both module docstrings).
    assert d1_models[1].ke == d1_models[2].ke == dopamine.TOOL_COMPOUND_KE
    assert dopamine.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE


# ---------------------------------------------------------------------------
# The circuit layer: does it obey its own three stated rules?
# ---------------------------------------------------------------------------


def test_both_pathways_are_present_on_one_network_with_textbook_signs():
    network = dopamine.cortico_basal_ganglia_network()
    W = network.connectivity
    idx = network.index

    # One striatum, two populations, one output nucleus, one readout node.
    for name in (
        "Cortex_M1",
        "Striatum_D1_MSN",
        "Striatum_D2_MSN",
        "GPe",
        "STN",
        "GPi",
        "Thalamus",
    ):
        assert name in idx

    w = dopamine.BG_WEIGHT
    # Shared corticostriatal input, identical onto both populations.
    assert W[idx["Striatum_D1_MSN"], idx["Cortex_M1"]] == w
    assert W[idx["Striatum_D2_MSN"], idx["Cortex_M1"]] == w
    # DIRECT: striatonigral, GABAergic, monosynaptic onto the output.
    assert W[idx["GPi"], idx["Striatum_D1_MSN"]] == -w
    # INDIRECT: striatopallidal -| GPe -| STN -> output.
    assert W[idx["GPe"], idx["Striatum_D2_MSN"]] == -w
    assert W[idx["STN"], idx["GPe"]] == -w
    assert W[idx["GPi"], idx["STN"]] == w
    # Shared output onto the locomotor readout's node.
    assert W[idx["Thalamus"], idx["GPi"]] == -w
    # ...and nothing else. Seven edges, no cross-block cheats.
    assert int(np.count_nonzero(W)) == 7

    # The readout really is the D2 scenario's own readout, on the node the
    # two pathways reconverge onto.
    assert (
        dopamine.locomotor_activity_readout().weights
        == scenarios.locomotor_activity_readout().weights
    )

    # The D2 node name is the one the existing calibrated D2 site couples
    # to, so the two pathways share a network rather than a resemblance.
    assert set(scenarios.d2_coupling()) == set(dopamine.d2_coupling())


def test_baselines_are_exactly_the_maximum_sensitivity_rule():
    # Rule 2: baseline_input puts every node at rate 0.5 at rest, the
    # maximum-sensitivity point stability.MAX_SENSITIVITY_RATE derives.
    # Given rule 1's unit weights that determines all seven values, so
    # stability.balanced_baselines must reproduce them exactly.
    network = dopamine.cortico_basal_ganglia_network()
    balanced = stability.balanced_baselines(
        network, target_rates=stability.MAX_SENSITIVITY_RATE
    )
    for node, rebuilt in zip(network.nodes, balanced.nodes):
        assert abs(node.baseline_input - rebuilt.baseline_input) < 1e-12

    # The value worth naming: at the output nucleus the rule gives exactly
    # 0.0, i.e. at rest the direct and indirect inputs to GPi cancel. That
    # is a consequence of rules 1 and 2, not a target.
    assert network.nodes[network.index["GPi"]].baseline_input == 0.0

    rates, residual = stability.steady_state(network)
    assert residual < 1e-6
    assert np.allclose(rates, 0.5, atol=1e-6)


def test_the_network_is_an_acyclic_cascade_and_that_is_the_uniqueness_proof():
    # The connectivity matrix is nilpotent: the graph is a directed acyclic
    # cascade, so each node's steady rate is a function of its upstream
    # nodes alone and the fixed point is unique and globally attracting by
    # topological order, whatever the taus.
    network = dopamine.cortico_basal_ganglia_network()
    W = network.connectivity
    assert np.allclose(np.linalg.matrix_power(W, network.n), 0.0)

    # P3.4's contraction certificate is SUFFICIENT, not necessary, and this
    # network does not satisfy it. Asserting that keeps the docstring's
    # claim honest: acyclicity is doing the work, not kappa.
    assessment = stability.assess(network)
    assert not assessment.globally_certified
    assert abs(assessment.contraction - np.sqrt(2.0)) < 1e-6

    # Empirically, everything P3.4's gate asks for holds anyway.
    assert assessment.converged
    assert assessment.unique
    assert assessment.locally_stable
    assert assessment.saturation.ok
    assert assessment.ok


def test_a_drug_free_run_sits_exactly_at_rest():
    sim = build_simulation()
    _result, scores = locomotor_trace(sim)
    assert np.allclose(scores, 0.5, atol=1e-6)


def test_rule_three_no_node_saturates_and_it_is_what_sets_the_gain_magnitude():
    # Rule 3: under a saturating dose of a full agonist at either receptor,
    # no node falls below stability.SATURATION_FLOOR. That rule is what
    # picks magnitude 1.0 for the two mechanism gains rather than reusing
    # scenarios.d2_coupling()'s -3.0, which was set against a different
    # network. This test asserts both halves of that claim.
    sim = build_simulation()
    for doses in ({"dihydrexidine": 3000.0}, {"dopamine": 3000.0}):
        result, _scores = locomotor_trace(sim, **doses)
        sensitivities = stability.sensitivity(result["rates"])
        assert sensitivities.min() > stability.SATURATION_FLOOR
        assert sensitivities.min() > 0.15

    # ...and the counterfactual, which is the reason the number is 1.0.
    network = dopamine.cortico_basal_ganglia_network()
    railed = Simulation(
        network=network,
        receptor_systems=[
            ReceptorSystem(
                pk_models=dopamine.d1_pk_models(),
                receptor_site=dopamine.d1_receptor_site(),
                coupling={"Striatum_D1_MSN": 3.0},
                name="D1_gain_3",
            )
        ],
    )
    result, _ = locomotor_trace(railed, dihydrexidine=3000.0)
    assert stability.sensitivity(result["rates"]).min() < stability.SATURATION_FLOOR


def test_cortical_drive_alone_almost_cancels_at_the_output_nucleus():
    # A consequence of rules 1 and 2 rather than a claim about the brain:
    # the shared corticostriatal input drives both MSN populations
    # identically, and their two paths to GPi have opposite net sign, so to
    # first order at the operating point a change in cortical drive does
    # not move the output nucleus at all. What moves it is dopamine acting
    # differentially at D1 and D2. The residual is second order (the direct
    # path passes through one sigmoid, the indirect through three).
    network = dopamine.cortico_basal_ganglia_network()
    idx = network.index
    for delta in (0.1, -0.1):
        drive = network.baseline_drive().copy()
        drive[idx["Cortex_M1"]] += delta
        rates, residual = stability.steady_state(network, drive=drive)
        assert residual < 1e-6
        msn_shift = abs(rates[idx["Striatum_D1_MSN"]] - 0.5)
        assert abs(rates[idx["Striatum_D2_MSN"]] - 0.5) == msn_shift
        assert msn_shift > 0.05  # the input really did reach the striatum
        assert abs(rates[idx["GPi"]] - 0.5) < 0.05 * msn_shift


# ---------------------------------------------------------------------------
# The pathways: Albin/DeLong's opposition, tested at the pathway level
# ---------------------------------------------------------------------------


def test_the_two_pathways_oppose_each_other_on_the_locomotor_readout():
    # [Kravitz2010] is a statement about PATHWAYS, not about drugs:
    # optogenetic excitation of direct-pathway MSNs increased locomotion,
    # excitation of indirect-pathway MSNs produced freezing and
    # bradykinesia. So it is tested by driving each MSN node directly,
    # with no ligand anywhere, which is as close as a rate model gets to
    # channelrhodopsin.
    network = dopamine.cortico_basal_ganglia_network()
    readout = dopamine.locomotor_activity_readout()
    idx = network.index

    baseline, _ = stability.steady_state(network)
    baseline_score = readout.score(network, baseline)

    scores = {}
    for node in ("Striatum_D1_MSN", "Striatum_D2_MSN"):
        drive = network.baseline_drive().copy()
        drive[idx[node]] += 1.0  # excite this population only
        rates, residual = stability.steady_state(network, drive=drive)
        assert residual < 1e-6
        scores[node] = readout.score(network, rates)

    assert scores["Striatum_D1_MSN"] > baseline_score
    assert scores["Striatum_D2_MSN"] < baseline_score


def test_each_receptor_moves_only_its_own_striatal_population():
    # [Gerfen1990]: D1 with the striatonigral population, D2 with the
    # striatopallidal one. A D1 drug must not move the D2-bearing node and
    # vice versa -- if it did, the two `coupling` dicts would be wrong,
    # and the "opposing effects" result below would be an artefact of
    # cross-talk rather than of two pathways.
    sim = build_simulation()
    idx = sim.network.index

    # A D1-only ligand drives the D1-bearing population and leaves the
    # D2-bearing one exactly at rest.
    d1_only, _ = locomotor_trace(sim, dihydrexidine=1000.0)
    assert np.allclose(d1_only["rates"][idx["Striatum_D2_MSN"]], 0.5, atol=1e-6)
    assert d1_only["rates"][idx["Striatum_D1_MSN"]].max() > 0.9

    # The converse, which has to be run against a dopamine background
    # because a D2 antagonist alone has nothing to displace: adding
    # haloperidol changes the D2-bearing population and leaves the
    # D1-bearing one where dopamine alone put it. The 1e-3 rather than 0
    # is RK45 step-size selection, the same residual P5.1's open-loop test
    # documents: one step is chosen for the whole shared state vector, so
    # adding a ligand perturbs the last digits of every other trace.
    plain, _ = locomotor_trace(sim, dopamine=1000.0)
    blocked, _ = locomotor_trace(sim, dopamine=1000.0, haloperidol=HALOPERIDOL_DOSE_nM)
    d1_trace_shift = np.abs(
        blocked["rates"][idx["Striatum_D1_MSN"]]
        - plain["rates"][idx["Striatum_D1_MSN"]]
    ).max()
    d2_trace_shift = np.abs(
        blocked["rates"][idx["Striatum_D2_MSN"]]
        - plain["rates"][idx["Striatum_D2_MSN"]]
    ).max()
    assert d1_trace_shift < 1e-3
    assert d2_trace_shift > 0.1


# ---------------------------------------------------------------------------
# The pharmacology: opposing D1 and D2 drugs on ONE locomotor readout
# ---------------------------------------------------------------------------


def test_a_d1_agonist_and_a_d2_antagonist_oppose_on_the_same_readout():
    # THE result this task exists to produce. One network, one readout
    # function, one shared dopamine dose; two drugs acting at two different
    # dopamine receptors on two different striatal populations, moving that
    # one readout in OPPOSITE directions relative to dopamine alone.
    #
    # dihydrexidine (D1 agonist, Kd 99.7 nM [Sahlholm2025]) adds drive to
    # the direct pathway, which inhibits GPi and releases the thalamus:
    # locomotion up. haloperidol (D2 antagonist, Kd 0.305 nM [Sykes2017])
    # displaces dopamine from the indirect pathway's D2 receptors, removing
    # the disinhibition dopamine was providing: locomotion down. Both are
    # the direction [Albin1989]/[Kravitz2010] predict for the pathway each
    # one acts on.
    sim = build_simulation()

    _, dopamine_alone = locomotor_trace(sim, dopamine=1000.0)
    _, plus_d1_agonist = locomotor_trace(sim, dopamine=1000.0, dihydrexidine=1000.0)
    _, plus_d2_antagonist = locomotor_trace(
        sim, dopamine=1000.0, haloperidol=HALOPERIDOL_DOSE_nM
    )

    assert plus_d1_agonist.max() > dopamine_alone.max()
    assert plus_d2_antagonist.max() < dopamine_alone.max()
    # Not a rounding-error separation in either direction. The drug-free
    # trace is flat to 1e-6 (test_a_drug_free_run_sits_exactly_at_rest), so
    # these are four orders of magnitude above the integrator's own noise.
    # The two thresholds differ because the two arms are not given equally
    # saturating doses: dihydrexidine at 10x its Kd nearly fills the D1
    # pool, while haloperidol at HALOPERIDOL_DOSE_nM takes only the
    # majority of the D2 pool from 1000 nM dopamine.
    assert plus_d1_agonist.max() - dopamine_alone.max() > 0.1
    assert dopamine_alone.max() - plus_d2_antagonist.max() > 0.04
    # Stated as the ordering it is, so a sign flip anywhere fails loudly.
    assert plus_d2_antagonist.max() < dopamine_alone.max() < plus_d1_agonist.max()


def test_the_two_opposing_drugs_partly_cancel_when_given_together():
    # The thing two separate models could not produce. Because both
    # pathways reconverge on one GPi node, giving both drugs at once lands
    # BETWEEN the two single-drug results instead of reproducing either --
    # the D2 blockade eats part of the D1 agonist's gain.
    sim = build_simulation()

    _, with_d1_agonist = locomotor_trace(sim, dopamine=1000.0, dihydrexidine=1000.0)
    _, with_d2_antagonist = locomotor_trace(sim, dopamine=1000.0, haloperidol=HALOPERIDOL_DOSE_nM)
    _, with_both = locomotor_trace(
        sim, dopamine=1000.0, dihydrexidine=1000.0, haloperidol=HALOPERIDOL_DOSE_nM
    )

    assert with_d2_antagonist.max() < with_both.max() < with_d1_agonist.max()


def test_both_antagonists_alone_are_inert_until_phase_6():
    # With no endogenous dopamine tone an antagonist has nothing to
    # displace, so it does nothing -- the same gap naloxone, haloperidol
    # and WAY-100635 have in every other scenario here. Real SCH-23390 and
    # real haloperidol given alone both suppress locomotion, because real
    # striatum has tonic dopamine. Supplying that tone is P6.2, and the day
    # it lands this expectation is already written down.
    sim = build_simulation()
    for doses in ({"SCH-23390": 2000.0}, {"haloperidol": HALOPERIDOL_DOSE_nM}):
        _, scores = locomotor_trace(sim, **doses)
        assert np.allclose(scores, 0.5, atol=1e-6)


def test_sch23390_reverses_the_d1_agonist_on_the_shared_network():
    # Competitive antagonism at the D1 pool: SCH-23390's derived Kd
    # (0.53 nM [Billard1984]) is ~190x tighter than dihydrexidine's
    # measured one (99.7 nM [Sahlholm2025]), so it wins the shared pool and
    # the direct-pathway drive collapses back toward rest.
    sim = build_simulation()

    _, agonist = locomotor_trace(sim, dihydrexidine=300.0)
    _, blocked = locomotor_trace(sim, dihydrexidine=300.0, **{"SCH-23390": 1000.0})

    assert agonist.max() > 0.75
    assert blocked.max() < agonist.max()
    # It does not merely blunt the agonist, it returns the readout to rest,
    # which is what a 190-fold affinity advantage plus a 178-minute
    # residence time should do.
    assert abs(blocked[-1] - 0.5) < 0.01


def test_haloperidol_reverses_dopamine_at_d2_on_the_shared_network():
    # The existing D2 result, re-run on the wider network, so that adding
    # the direct pathway is shown not to have broken the indirect one.
    sim = build_simulation()
    idx = sim.network.index

    result_agonist, agonist = locomotor_trace(sim, dopamine=1000.0)
    result_blocked, blocked = locomotor_trace(
        sim, dopamine=1000.0, haloperidol=HALOPERIDOL_DOSE_nM
    )

    assert agonist.max() > 0.5
    assert blocked.max() < agonist.max()

    # The mechanism, not just the readout: haloperidol keeps the
    # indirect-pathway MSN closer to rest.
    assert (
        result_blocked["rates"][idx["Striatum_D2_MSN"]].min()
        > result_agonist["rates"][idx["Striatum_D2_MSN"]].min()
    )


def test_dopamine_alone_drives_both_arms_the_same_way_which_is_the_real_biology():
    # Worth asserting because it is the result a reader expects to be
    # opposite and is not. D1 and D2 have opposite receptor-level signs on
    # their own MSNs, and both of those changes RELEASE the thalamus:
    # exciting the direct pathway inhibits GPi, and inhibiting the indirect
    # pathway also ends up inhibiting GPi (via GPe -| STN). So dopamine
    # acting at both receptors at once raises locomotion through both arms.
    # The opposition in this module is between pathway ACTIVATIONS
    # ([Kravitz2010], tested above) and between a D1 agonist and a D2
    # antagonist, not between D1 and D2 agonism.
    sim = build_simulation()
    idx = sim.network.index

    result, scores = locomotor_trace(sim, dopamine=3000.0)
    assert scores.max() > 0.5
    # Opposite signs at the two striatal populations...
    assert result["rates"][idx["Striatum_D1_MSN"]].max() > 0.5
    assert result["rates"][idx["Striatum_D2_MSN"]].min() < 0.5
    # ...and one direction at the output nucleus.
    assert result["rates"][idx["GPi"]].min() < 0.5
