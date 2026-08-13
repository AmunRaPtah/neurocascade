"""Tests for the P5.1 serotonin 5-HT1A / 5-HT2A receptor systems.

`serotonin.py`'s module docstring carries the full source list and the
derivation of every constant; each test below restates the published figure
it checks against, so a silent edit to the module cannot pass.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade import serotonin, stability
from neurocascade.engine import Simulation
from neurocascade.receptor import ReceptorSite

AFFINITY_STATES = serotonin.HT2A_AFFINITY_STATES


def equilibrium_occupancy(ligand, concentration, dt=0.01, steps=200000):
    """Integrate one ligand's occupancy to equilibrium at a fixed [C]."""
    site = ReceptorSite(name="site", ligands=[ligand])
    occ = np.array([0.0])
    for _ in range(steps):
        occ = occ + dt * site.occupancy_rhs(occ, [concentration])
    return float(occ[0])


def by_name(ligands):
    return {ligand.name: ligand for ligand in ligands}


# ---------------------------------------------------------------------------
# The binding layer: does the module reproduce the published numbers?
# ---------------------------------------------------------------------------


def test_5ht1a_rate_constants_reproduce_the_published_affinities():
    # For a single ligand at equilibrium, occ = C / (C + koff/kon), so
    # occupancy at C = the published affinity must be 0.5 or the conversion
    # from the paper's units into this codebase's is wrong.
    #   serotonin  Ki 8.7 nM   [Satala2016], vs [3H]-8-OH-DPAT, h5-HT1A/HEK293
    #   8-OH-DPAT  Kd 2.9 nM   [Satala2016], saturation, same preparation
    #   WAY-100635 Kd 0.087 nM [Parkel2006], saturation, rat hippocampus
    published = {
        "serotonin": 8.7,
        "8-OH-DPAT": 2.9,
        "WAY-100635": 0.087,
    }
    for ligand in serotonin.ligands_5ht1a():
        affinity = published[ligand.name]
        assert abs(ligand.koff / ligand.kon - affinity) < 1e-3 * affinity
        assert abs(equilibrium_occupancy(ligand, affinity) - 0.5) < 1e-3


@pytest.mark.parametrize("affinity_state", AFFINITY_STATES)
def test_5ht2a_rate_constants_reproduce_the_published_affinities(affinity_state):
    #   serotonin   Kd 150 nM live-cell NanoBRET [Denomme2024], or Ki 750 nM
    #               against [3H]ketanserin in rat cortex [McKenna1989] in
    #               the antagonist-radioligand set
    #   DOI         Ki 19.05 nM (pKi 7.72) vs [3H]ketanserin [Seggel1990]
    #   ketanserin  Kd 1.0 nM saturation, rat cortex [McKenna1989]
    published = {
        "serotonin": (
            150.0 if affinity_state == "live-cell" else 750.0
        ),
        "DOI": 19.05,
        "ketanserin": 1.0,
    }
    for ligand in serotonin.ligands_5ht2a(affinity_state=affinity_state):
        affinity = published[ligand.name]
        assert abs(ligand.koff / ligand.kon - affinity) < 1e-3 * affinity
        assert abs(equilibrium_occupancy(ligand, affinity) - 0.5) < 1e-3


def test_serotonin_5ht2a_rate_constants_are_the_measured_pair_not_a_derivation():
    # [Denomme2024] measured BOTH constants for serotonin at human 5-HT2A by
    # NanoBRET in live HEK293 cells: kon = 0.42e6 M^-1 s^-1 and
    # koff = 0.063 s^-1. This is the only ligand in the module that needs no
    # derivation at either site, and it is the reason the live-cell affinity
    # state is the default.
    serotonin_2a = by_name(serotonin.ligands_5ht2a())["serotonin"]
    assert serotonin_2a.kon == pytest.approx(0.42e6 * 60 * 1e-9, rel=1e-6)
    assert serotonin_2a.koff == pytest.approx(0.063 * 60, rel=1e-6)


def test_8ohdpat_single_state_koff_lies_between_the_measured_phases():
    # [Satala2016] measured a BIPHASIC dissociation for [3H]-8-OH-DPAT,
    # koff(fast) = 2.33 min^-1 and koff(slow) = 0.21 min^-1, the signature
    # of an agonist seeing a G-protein-coupled and an uncoupled state.
    # `ReceptorSite` has one pool, so the module uses the single-state
    # collapse kon * Kd, with both factors measured in that same paper. A
    # collapse of a two-phase decay has to land between the two phases; if
    # it does not, the three numbers are not describing one experiment.
    dpat = by_name(serotonin.ligands_5ht1a())["8-OH-DPAT"]
    assert dpat.koff == pytest.approx(
        serotonin.HT1A_KON * serotonin.HT1A_KD_8OHDPAT, rel=1e-3
    )
    assert serotonin.HT1A_KOFF_8OHDPAT_SLOW < dpat.koff
    assert dpat.koff < serotonin.HT1A_KOFF_8OHDPAT_FAST


def test_way100635_is_a_nearly_irreversible_antagonist():
    # [Parkel2006]: koff = 7.8e-3 min^-1 in rat hippocampal membranes, so
    # slow that the authors report equilibrium is not reached before 7.5 h
    # of incubation. That is a dissociation half-life of ~89 min and ~130x
    # slower than the agonist it competes with here. [Forster1995] saw the
    # same fact from the pharmacology side: insurmountable antagonism at
    # high concentrations in isolated tissue.
    ligands = by_name(serotonin.ligands_5ht1a())
    way, dpat = ligands["WAY-100635"], ligands["8-OH-DPAT"]
    dissociation_half_life = np.log(2.0) / way.koff
    assert 85.0 < dissociation_half_life < 95.0
    assert dpat.koff / way.koff > 100.0


def test_efficacies_are_the_published_relative_maxima():
    # [NewmanTancredi2003], one [35S]GTPgammaS panel in rat hippocampal
    # membranes, Emax relative to 5-HT = 100: (+)8-OH-DPAT 85,
    # WAY100,635 0. [Cussac2008], h5-HT2A Gq/11-coupled calcium
    # mobilisation: DOI 81.3% of the serotonin maximum. Serotonin is 1.0 at
    # both sites by construction: it is the reference of both scales.
    ht1a = by_name(serotonin.ligands_5ht1a())
    assert ht1a["serotonin"].efficacy == 1.0
    assert ht1a["8-OH-DPAT"].efficacy == 0.85
    assert ht1a["WAY-100635"].efficacy == 0.0

    ht2a = by_name(serotonin.ligands_5ht2a())
    assert ht2a["serotonin"].efficacy == 1.0
    assert ht2a["DOI"].efficacy == 0.813
    assert ht2a["ketanserin"].efficacy == 0.0


def test_both_selective_agonists_are_partial_not_full():
    # Worth its own assertion because the textbook shorthand ("8-OH-DPAT,
    # the prototypical 5-HT1A agonist"; "DOI, the 5-HT2A agonist") hides a
    # ~15-20% efficacy deficit that both cited functional panels measure and
    # that `Ligand.efficacy` can carry directly.
    ht1a = by_name(serotonin.ligands_5ht1a())
    ht2a = by_name(serotonin.ligands_5ht2a())
    for agonist, reference in (
        (ht1a["8-OH-DPAT"], ht1a["serotonin"]),
        (ht2a["DOI"], ht2a["serotonin"]),
    ):
        assert 0.8 <= agonist.efficacy < reference.efficacy


def test_serotonin_engages_5ht1a_at_far_lower_concentrations_than_5ht2a():
    # Not a modelling choice: 8.7 nM at 5-HT1A [Satala2016] against ~150 nM
    # at 5-HT2A [Denomme2024], two independent labs, a ~17-fold split. At a
    # concentration between the two affinities the 5-HT1A pool is most of
    # the way occupied while the 5-HT2A pool has barely started.
    serotonin_1a = by_name(serotonin.ligands_5ht1a())["serotonin"]
    serotonin_2a = by_name(serotonin.ligands_5ht2a())["serotonin"]
    kd_ratio = (serotonin_2a.koff / serotonin_2a.kon) / (
        serotonin_1a.koff / serotonin_1a.kon
    )
    assert 15.0 < kd_ratio < 20.0

    occupancy_1a = equilibrium_occupancy(serotonin_1a, 40.0)
    occupancy_2a = equilibrium_occupancy(serotonin_2a, 40.0)
    assert occupancy_1a > 0.8
    assert occupancy_2a < 0.25


def test_affinity_state_changes_serotonin_only_and_rejects_unknown_states():
    live = by_name(serotonin.ligands_5ht2a("live-cell"))
    membrane = by_name(serotonin.ligands_5ht2a("antagonist-radioligand"))
    for name in ("DOI", "ketanserin"):
        assert live[name].kon == membrane[name].kon
        assert live[name].koff == membrane[name].koff
    assert membrane["serotonin"].koff > live["serotonin"].koff
    # The bounded consequence recorded in the module docstring: the
    # ketanserin/serotonin affinity ratio is 150 in one set and 750 in the
    # other, and nothing else moves.
    def ratio(ligands):
        return (ligands["serotonin"].koff / ligands["serotonin"].kon) / (
            ligands["ketanserin"].koff / ligands["ketanserin"].kon
        )

    assert ratio(live) == pytest.approx(150.0, rel=1e-3)
    assert ratio(membrane) == pytest.approx(750.0, rel=1e-3)

    with pytest.raises(ValueError):
        serotonin.ligands_5ht2a(affinity_state="agonist-radioligand")


# ---------------------------------------------------------------------------
# The circuit layer
# ---------------------------------------------------------------------------


def build_sim(affinity_state="live-cell"):
    return Simulation(
        network=serotonin.raphe_cortical_network(),
        receptor_systems=serotonin.serotonin_receptor_systems(
            affinity_state=affinity_state
        ),
    )


def run(sim, t_end=60.0, n=300, **doses):
    t_eval = np.linspace(0.0, t_end, n)
    return sim.run(
        doses=sim.dose_vector(**doses), t_span=(0.0, t_end), t_eval=t_eval
    )


def final_scores(sim, result, readouts):
    return [
        readout.score(sim.network, result["rates"][:, -1]) for readout in readouts
    ]


def test_network_rests_at_maximum_sensitivity_and_carries_the_p34_certificate():
    # The two stated rules the illustrative circuit values follow (see the
    # section comment in serotonin.py): every node at rate 0.5 at rest, the
    # maximum-sensitivity point `stability.MAX_SENSITIVITY_RATE` derives,
    # and a contraction number below 1, P3.4's global-uniqueness
    # certificate. Neither is a fit to anything; both are checkable.
    network = serotonin.raphe_cortical_network()
    assessment = stability.assess(network)
    assert assessment.converged
    assert assessment.unique
    assert assessment.locally_stable
    assert assessment.globally_certified
    assert assessment.contraction == pytest.approx(0.9, rel=1e-6)
    assert np.allclose(assessment.rates, stability.MAX_SENSITIVITY_RATE, atol=1e-6)


@pytest.mark.parametrize("affinity_state", AFFINITY_STATES)
def test_5ht1a_agonist_suppresses_raphe_firing(affinity_state):
    # The autoreceptor, as an agonist demonstration: 8-OH-DPAT activating
    # the somatodendritic 5-HT1A receptor on dorsal raphe serotonin neurons
    # suppresses their firing ([Sprouse1987] for the electrophysiology this
    # stands for).
    sim = build_sim(affinity_state)
    tone = serotonin.serotonergic_tone_readout()

    baseline = run(sim)
    agonist = run(sim, **{"8-OH-DPAT": 100.0})

    (baseline_tone,) = final_scores(sim, baseline, [tone])
    (agonist_tone,) = final_scores(sim, agonist, [tone])
    assert agonist_tone < baseline_tone - 0.1


@pytest.mark.parametrize("affinity_state", AFFINITY_STATES)
def test_5ht2a_agonist_raises_cortical_output(affinity_state):
    # DOI at the Gq-coupled 5-HT2A receptor on layer V pyramidal neurons
    # raises their firing ([AghajanianMarek1997] for the postsynaptic
    # excitation this stands for; [Puig2003] for why only the pyramidal
    # node carries the receptor here).
    sim = build_sim(affinity_state)
    cortex = serotonin.cortical_output_readout()

    baseline = run(sim)
    agonist = run(sim, DOI=100.0)

    (baseline_cortex,) = final_scores(sim, baseline, [cortex])
    (agonist_cortex,) = final_scores(sim, agonist, [cortex])
    assert agonist_cortex > baseline_cortex + 0.1


@pytest.mark.parametrize("affinity_state", AFFINITY_STATES)
def test_either_antagonist_alone_stays_at_baseline(affinity_state):
    # Both antagonists have nothing to displace in a model with no
    # endogenous serotonin tone, so alone they must do nothing, the same
    # check the naloxone and haloperidol scenarios make. Real WAY-100635
    # given alone is NOT inert in vivo, because real raphe neurons carry
    # tonic autoreceptor activation; supplying that tone is P6.2, and this
    # test is the marker for where it will show up.
    sim = build_sim(affinity_state)
    readouts = [
        serotonin.serotonergic_tone_readout(),
        serotonin.cortical_output_readout(),
    ]
    baseline = final_scores(sim, run(sim), readouts)

    for antagonist in ({"WAY-100635": 100.0}, {"ketanserin": 100.0}):
        blocked = final_scores(sim, run(sim, **antagonist), readouts)
        assert np.allclose(blocked, baseline, atol=1e-3)


@pytest.mark.parametrize("affinity_state", AFFINITY_STATES)
def test_way100635_reverses_the_8ohdpat_suppression_of_raphe_firing(affinity_state):
    # Displacement at the shared 5-HT1A pool. Unlike naloxone against
    # fentanyl (which needs a large molar excess because its koff is the
    # FASTER of the two), WAY-100635 wins by not letting go: koff
    # 0.0078 min^-1 against 8-OH-DPAT's 0.986 [Parkel2006], [Satala2016].
    sim = build_sim(affinity_state)
    tone = serotonin.serotonergic_tone_readout()

    t_split = 30.0
    phase1 = sim.run(
        doses=sim.dose_vector(**{"8-OH-DPAT": 100.0}),
        t_span=(0.0, t_split),
        t_eval=np.linspace(0.0, t_split, 200),
    )
    y = np.concatenate(
        [
            phase1["pk_states"][:, -1],
            phase1["occupancy"][:, -1],
            phase1["rates"][:, -1],
        ]
    )
    y = sim.add_dose(y, sim.dose_vector(**{"WAY-100635": 1000.0}))
    solution = solve_ivp(
        sim.rhs,
        (t_split, 60.0),
        y,
        method="RK45",
        t_eval=np.linspace(t_split, 60.0, 200),
        max_step=0.1,
    )
    assert solution.success
    d, r = sim.d, sim.r
    occupancy = solution.y[d : d + r, -1]
    rates = solution.y[d + r : d + r + sim.c, -1]

    ligand_index = {
        ligand.name: i
        for i, ligand in enumerate(sim.receptor_systems[0].receptor_site.ligands)
    }
    assert occupancy[ligand_index["WAY-100635"]] > 0.9
    assert occupancy[ligand_index["8-OH-DPAT"]] < 0.05

    baseline = run(sim)
    (baseline_tone,) = final_scores(sim, baseline, [tone])
    reversed_tone = tone.score(sim.network, rates)
    suppressed_tone = tone.score(sim.network, phase1["rates"][:, -1])
    assert suppressed_tone < baseline_tone - 0.1
    assert abs(reversed_tone - baseline_tone) < 1e-2


@pytest.mark.parametrize("affinity_state", AFFINITY_STATES)
def test_ketanserin_reverses_the_doi_cortical_excitation(affinity_state):
    # The same reversal at the other site: ketanserin's Kd of 1.0 nM
    # [McKenna1989] against DOI's 19.05 nM [Seggel1990] is a 19-fold
    # affinity advantage, and its derived koff (0.0252 min^-1) is 19x
    # slower, so a molar excess displaces DOI within the window.
    sim = build_sim(affinity_state)
    cortex = serotonin.cortical_output_readout()

    t_split = 30.0
    phase1 = sim.run(
        doses=sim.dose_vector(DOI=100.0),
        t_span=(0.0, t_split),
        t_eval=np.linspace(0.0, t_split, 200),
    )
    y = np.concatenate(
        [
            phase1["pk_states"][:, -1],
            phase1["occupancy"][:, -1],
            phase1["rates"][:, -1],
        ]
    )
    y = sim.add_dose(y, sim.dose_vector(ketanserin=1000.0))
    solution = solve_ivp(
        sim.rhs,
        (t_split, 60.0),
        y,
        method="RK45",
        t_eval=np.linspace(t_split, 60.0, 200),
        max_step=0.1,
    )
    assert solution.success
    d, r = sim.d, sim.r
    occupancy = solution.y[d : d + r, -1]
    rates = solution.y[d + r : d + r + sim.c, -1]

    ht2a = sim.receptor_systems[1]
    offset = sim.r - ht2a.n
    ligand_index = {
        ligand.name: offset + i
        for i, ligand in enumerate(ht2a.receptor_site.ligands)
    }
    assert occupancy[ligand_index["ketanserin"]] > 0.9
    assert occupancy[ligand_index["DOI"]] < 0.05

    (baseline_cortex,) = final_scores(sim, run(sim), [cortex])
    excited_cortex = cortex.score(sim.network, phase1["rates"][:, -1])
    reversed_cortex = cortex.score(sim.network, rates)
    assert excited_cortex > baseline_cortex + 0.1
    assert abs(reversed_cortex - baseline_cortex) < 1e-2


def test_the_two_receptors_move_the_cortical_node_in_opposite_directions():
    # One node, two receptors of opposite sign, which is the anatomy
    # [AmargosBosch2004] and [Santana2017] report for mPFC pyramidal
    # neurons: Gi-coupled 5-HT1A and Gq-coupled 5-HT2A on the same cell.
    # Given together, the two drugs land between their separate effects.
    sim = build_sim()
    cortex = serotonin.cortical_output_readout()

    (baseline,) = final_scores(sim, run(sim), [cortex])
    (with_dpat,) = final_scores(sim, run(sim, **{"8-OH-DPAT": 100.0}), [cortex])
    (with_doi,) = final_scores(sim, run(sim, DOI=100.0), [cortex])
    (with_both,) = final_scores(
        sim, run(sim, **{"8-OH-DPAT": 100.0, "DOI": 100.0}), [cortex]
    )

    assert with_dpat < baseline < with_doi
    assert with_dpat < with_both < with_doi


def test_dose_vector_applies_one_serotonin_dose_to_both_receptor_pools():
    # Serotonin is two `Ligand` objects here, one per receptor pool, with
    # two concentration states. `dose_vector` matches by name across
    # systems, so a single serotonin= keyword reaches both, which is what
    # one rise in serotonin physically is.
    sim = build_sim()
    doses = sim.dose_vector(serotonin=50.0)
    serotonin_positions = [
        i
        for i, ligand in enumerate(
            [
                ligand
                for system in sim.receptor_systems
                for ligand in system.receptor_site.ligands
            ]
        )
        if ligand.name == "serotonin"
    ]
    assert len(serotonin_positions) == 2
    assert np.allclose(doses[serotonin_positions], 50.0)
    assert doses.sum() == pytest.approx(100.0)


def test_autoreceptor_is_an_open_loop_until_phase_6():
    # The documented gap, asserted rather than described. A 5-HT1A agonist
    # visibly suppresses the raphe node, and that suppression reaches
    # nothing: the serotonin concentration at the cortical 5-HT2A pool, and
    # therefore its occupancy, are unchanged with and without the agonist,
    # because concentrations in this engine come from the PK layer alone.
    # Closing the loop (P6.1 release, P6.2 endogenous tone) must break this
    # test, which is the point of writing it.
    #
    # "Unchanged" is checked to 1e-4 relative and not to the bit, for a
    # numerical reason with no physics in it: RK45 chooses one step size
    # for the whole shared state vector, so adding a ligand perturbs the
    # step sequence and hence the last digits of every other trace. The
    # residual is ~2e-6 relative here, four orders below the ~0.4 change
    # the same dose produces in the raphe rate.
    sim = build_sim()
    tone = serotonin.serotonergic_tone_readout()
    t_end, n = 10.0, 200

    without = run(sim, t_end=t_end, n=n, serotonin=300.0)
    with_agonist = run(sim, t_end=t_end, n=n, serotonin=300.0, **{"8-OH-DPAT": 100.0})

    assert (
        tone.score(sim.network, with_agonist["rates"][:, -1])
        < tone.score(sim.network, without["rates"][:, -1]) - 0.1
    )

    ligands = [
        ligand
        for system in sim.receptor_systems
        for ligand in system.receptor_site.ligands
    ]
    ht2a_serotonin = len(sim.receptor_systems[0].receptor_site.ligands)
    assert ligands[ht2a_serotonin].name == "serotonin"
    for trace in ("concentrations", "occupancy"):
        undosed = without[trace][ht2a_serotonin]
        dosed = with_agonist[trace][ht2a_serotonin]
        scale = max(np.abs(undosed).max(), 1e-12)
        assert np.abs(dosed - undosed).max() / scale < 1e-4
