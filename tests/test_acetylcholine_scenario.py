"""P5.4: acetylcholine at nicotinic (alpha4beta2, alpha7) and muscarinic
(M1-M5) receptors.

Structure follows P5.1-P5.3: first the cited constants, then the
derivations, then what the numbers say, then the circuit rules, then the
scenario. Integration is used sparingly and over short windows, because the
nicotinic sites carry `koff` values of several hundred per minute and RK45
over the shared state vector is not cheap there; every claim that can be
made at the algebraic fixed point is made there instead.
"""

import numpy as np
import pytest

from neurocascade import (
    DesensitizingReceptorSite,
    ReceptorSite,
    Simulation,
    acetylcholine,
    dopamine,
    noradrenaline,
    serotonin,
    stability,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def site_steady_state(site, concentrations):
    """The site's fixed point, written down rather than integrated to.

    P5.3 had to do this once, for the whole-cell alpha2 set whose `koff` of
    129194 min^-1 made RK45 impractical. Here the reason is the opposite
    end of the same problem: alpha4beta2's recovery constant is 86.9 min,
    so relaxation to equilibrium takes hours of model time while the
    fastest `koff` in the same site is 150 min^-1, and an explicit
    integration would need ~10^6 steps per assertion. The fixed point is
    algebraic, so it is written down:

        bound_i       = (C_i / Kd_i) * free
        desens_i      = (k_des_i / k_rec_i) * bound_i
        free          = 1 / (1 + sum_i (C_i/Kd_i) * (1 + k_des_i/k_rec_i))
        blocked_j     = (C_j / Kdb_j) / (1 + sum_j C_j / Kdb_j)

    `test_the_algebraic_fixed_point_is_a_fixed_point` checks this against
    `occupancy_rhs` directly, so the shortcut cannot silently disagree with
    the model it is a shortcut for.
    """
    concentrations = np.asarray(concentrations, dtype=float)
    if not isinstance(site, DesensitizingReceptorSite):
        ratios = np.array(
            [
                lig.kon * c / lig.koff
                for lig, c in zip(site.ligands, concentrations)
            ]
        )
        return ratios / (1.0 + ratios.sum())

    p = site.n_orthosteric
    ratios = np.array(
        [
            lig.kon * concentrations[i] / lig.koff
            for i, lig in enumerate(site.orthosteric_ligands)
        ]
    )
    trap = site.k_des / site.k_rec
    free = 1.0 / (1.0 + (ratios * (1.0 + trap)).sum())
    bound = ratios * free
    desens = trap * bound

    blocker_ratios = np.array(
        [
            lig.kon * concentrations[p + j] / lig.koff
            for j, lig in enumerate(site.channel_blockers)
        ]
    )
    blocked = (
        blocker_ratios / (1.0 + blocker_ratios.sum())
        if site.n_blockers
        else np.zeros(0)
    )
    return np.concatenate([bound, desens, blocked])


def total_bound(site, state):
    bound, desens, _ = site.split_states(state)
    return bound.sum() + desens.sum()


def test_the_algebraic_fixed_point_is_a_fixed_point():
    # The helper every equilibrium assertion below leans on, checked
    # against the model's own RHS at both nicotinic sites and both
    # muscarinic ones, with all ligands present at once.
    cases = (
        (acetylcholine.receptor_site_alpha4beta2(), [500.0, 300.0, 20000.0]),
        (acetylcholine.receptor_site_alpha7(), [30000.0, 5000.0, 3000.0]),
        (acetylcholine.receptor_site_m1(), [20000.0, 5.0]),
        (acetylcholine.receptor_site_m2(), [4000.0, 2.0]),
    )
    for site, concentrations in cases:
        state = site_steady_state(site, concentrations)
        assert np.allclose(site.occupancy_rhs(state, concentrations), 0.0, atol=1e-12)
        assert np.all(state >= 0.0)
        if isinstance(site, DesensitizingReceptorSite):
            assert site.free_fraction(state) > 0.0
            assert site.blocked_fraction(state) <= 1.0
        else:
            assert state.sum() <= 1.0


@pytest.fixture(scope="module")
def network():
    return acetylcholine.basal_forebrain_cortical_network()


@pytest.fixture(scope="module")
def sim(network):
    return Simulation(
        network=network, receptor_systems=acetylcholine.cholinergic_receptor_systems()
    )


# ---------------------------------------------------------------------------
# The cited constants
# ---------------------------------------------------------------------------


def test_fenster1997_table_1_is_transcribed_as_published():
    # rat subunits in Xenopus oocytes, one laboratory, one experiment.
    # Nicotine activation EC50, calcium-containing / calcium-free.
    assert acetylcholine.ALPHA4BETA2_KD_NICOTINE == 15000.0
    assert acetylcholine.ALPHA4BETA2_KD_NICOTINE_CALCIUM_FREE == 12000.0
    assert acetylcholine.ALPHA7_KD_NICOTINE == 90000.0
    assert acetylcholine.ALPHA7_KD_NICOTINE_CALCIUM_FREE == 234000.0
    # Nicotine desensitization IC50.
    assert acetylcholine.ALPHA4BETA2_DESENSITIZATION_IC50_NICOTINE == 61.0
    assert acetylcholine.ALPHA7_DESENSITIZATION_IC50_NICOTINE == 1300.0
    assert acetylcholine.ALPHA7_DESENSITIZATION_IC50_NICOTINE_CALCIUM_FREE == 6200.0
    # Time constants, in minutes.
    assert acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION == 3.7
    assert acetylcholine.ALPHA4BETA2_TAU_RECOVERY == 86.9
    assert acetylcholine.ALPHA7_TAU_DESENSITIZATION == 0.5
    assert acetylcholine.ALPHA7_TAU_RECOVERY == 1.9
    # The three rows this module does not build, kept so the comparison is
    # one experiment wide rather than two quotations.
    assert acetylcholine.FENSTER1997_OTHER_SUBTYPES == {
        "alpha3beta4": (32.5, 42.6),
        "alpha4beta4": (35.5, 413.8),
        "alpha3beta2": (5.0, 11.7),
    }


def test_the_desensitization_constants_are_used_as_rates_of_the_published_taus():
    a4b2 = acetylcholine.receptor_site_alpha4beta2()
    a7 = acetylcholine.receptor_site_alpha7()
    assert a4b2.k_des[0] == pytest.approx(1.0 / 3.7)
    assert a4b2.k_rec[0] == pytest.approx(1.0 / 86.9)
    assert a7.k_des[0] == pytest.approx(1.0 / 0.5)
    assert a7.k_rec[0] == pytest.approx(1.0 / 1.9)
    # Both agonists at a site share the site's rates: derivation 3 moved
    # from a binding rate to a gating rate, stated in the module docstring.
    assert a4b2.k_des[0] == a4b2.k_des[1]
    assert a7.k_rec[0] == a7.k_rec[1]


def test_the_second_reading_of_the_same_two_rates_is_kept_as_the_error_bar():
    # [Fenster1999], same laboratory, same preparation.
    assert acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION_FAST_1999 == 1.4
    assert acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION_SLOW_1999 == 17.0
    assert acetylcholine.ALPHA4BETA2_TAU_RECOVERY_1999 == 43.0
    # The recovery constants differ 2.0-fold; anything this module says
    # about recovery timing is good to about a factor of two and no better.
    ratio = (
        acetylcholine.ALPHA4BETA2_TAU_RECOVERY
        / acetylcholine.ALPHA4BETA2_TAU_RECOVERY_1999
    )
    assert 1.9 < ratio < 2.1
    # ...and the single 3.7 min onset sits between the 1999 biphasic pair,
    # which is the most that can be said for it.
    assert (
        acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION_FAST_1999
        < acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION
        < acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION_SLOW_1999
    )


def test_acetylcholine_affinities_are_the_published_potencies():
    assert acetylcholine.ALPHA4BETA2_KD_ACETYLCHOLINE == 1600.0  # [Sanders2022] HS
    assert acetylcholine.ALPHA4BETA2_KD_ACETYLCHOLINE_LOW_SENSITIVITY == 37100.0
    assert acetylcholine.ALPHA7_KD_ACETYLCHOLINE == 30000.0  # [PapkePapke2002]
    # The stoichiometry spread, which is this module's counterpart of
    # P5.3's alpha1-subtype spread.
    assert (
        acetylcholine.ALPHA4BETA2_KD_ACETYLCHOLINE_LOW_SENSITIVITY
        / acetylcholine.ALPHA4BETA2_KD_ACETYLCHOLINE
    ) == pytest.approx(23.2, rel=0.01)
    # The alpha7 net-charge estimate is at least fivefold tighter than the
    # peak-current literature it replaces, which is the paper's own claim
    # and the reason the net-charge number is the one used.
    assert (
        acetylcholine.ALPHA7_KD_ACETYLCHOLINE_PEAK_CURRENT_LOW
        / acetylcholine.ALPHA7_KD_ACETYLCHOLINE
    ) > 3.5
    assert (
        acetylcholine.ALPHA7_KD_ACETYLCHOLINE_PEAK_CURRENT_HIGH
        / acetylcholine.ALPHA7_KD_ACETYLCHOLINE
    ) > 11.0


def test_mecamylamine_constants_and_the_disagreement_between_two_laboratories():
    # [Papke2013] Table 3, high-sensitivity stoichiometry, peak current.
    assert acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE_R == 19000.0
    assert acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE_S == 17000.0
    assert acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE == 18000.0
    # ...and the low-sensitivity stoichiometry runs the OTHER way from the
    # agonist's preference: 7.5-fold more sensitive to the blocker.
    assert (
        acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE
        / acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE_LOW_SENSITIVITY
    ) == pytest.approx(7.5)

    # [ChavezNoriega1997]: ~50% block at 3 uM at BOTH human subtypes.
    assert acetylcholine.ALPHA7_IC50_MECAMYLAMINE == 3000.0
    assert acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE_CHAVEZ == 3000.0
    # The two laboratories disagree 6-fold at alpha4beta2, and the module
    # says so rather than picking quietly.
    assert (
        acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE
        / acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE_CHAVEZ
    ) == pytest.approx(6.0)


def test_muscarinic_tables_cover_all_five_subtypes_for_both_ligands():
    assert set(acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE) == set(
        acetylcholine.MUSCARINIC_SUBTYPES
    )
    assert set(acetylcholine.MUSCARINIC_PKI_SCOPOLAMINE) == set(
        acetylcholine.MUSCARINIC_SUBTYPES
    )
    assert acetylcholine.MUSCARINIC_PKI_SCOPOLAMINE["M3"] == (9.4, 9.4)
    assert acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE["M2"] == (4.3, 6.5)
    for subtype in acetylcholine.MUSCARINIC_SUBTYPES:
        low, high = acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE[subtype]
        assert low <= high
        low, high = acetylcholine.MUSCARINIC_PKI_SCOPOLAMINE[subtype]
        assert low <= high


def test_the_rat_acetylcholine_values_almost_fall_inside_the_human_ranges():
    # The only species check available for the muscarinic constants, and it
    # is recorded with the one place it fails rather than as a clean pass.
    # Only rat M2 (6.4) actually lands inside its human range. Rat M1 (5.0)
    # and rat M3 (5.6) each sit slightly ABOVE the human upper bound, by
    # 0.1 and 0.2 log units, i.e. the curated rat affinity is 1.26-fold and
    # 1.58-fold tighter than the tightest human reading. Both overshoots
    # are far smaller than the widths of the human ranges themselves (up to
    # 158-fold at M2), so this is not read as a species difference; it is
    # read as further evidence that the curated agonist numbers are a
    # mixture of affinity states rather than one quantity.
    inside, outside = [], []
    for subtype, rat in acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE_RAT.items():
        low, high = acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE[subtype]
        (inside if low <= rat <= high else outside).append(subtype)
    assert inside == ["M2"]
    assert sorted(outside) == ["M1", "M3"]
    overshoot = {
        subtype: 10.0
        ** (
            acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE_RAT[subtype]
            - acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE[subtype][1]
        )
        for subtype in outside
    }
    assert overshoot["M1"] == pytest.approx(1.26, rel=0.01)
    assert overshoot["M3"] == pytest.approx(1.58, rel=0.01)
    human_width = max(
        10.0 ** (high - low)
        for low, high in acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE.values()
    )
    assert all(value < human_width for value in overshoot.values())


def test_muscarinic_kd_is_the_geometric_midpoint_of_the_recorded_range():
    # M1 acetylcholine 4.3-4.9 -> pKi 4.6 -> 25.1 uM.
    assert acetylcholine.muscarinic_kd("M1", "acetylcholine") == pytest.approx(
        10.0 ** (9.0 - 4.6)
    )
    # M1 scopolamine is a single value, so no midpoint is taken.
    assert acetylcholine.muscarinic_kd("M1", "scopolamine") == pytest.approx(1.0)
    assert acetylcholine.muscarinic_kd("M5", "scopolamine") == pytest.approx(
        10.0 ** (9.0 - 8.7)
    )
    with pytest.raises(ValueError):
        acetylcholine.muscarinic_kd("M6", "acetylcholine")


def test_scopolamine_is_essentially_non_selective_across_the_five_subtypes():
    # The claim the five-subtype table exists to make: scopolamine's pKi
    # spans 8.7 to 9.4, a factor of 5 in Kd, so "scopolamine blocks the
    # muscarinic component" is a subtype-independent statement here in the
    # way "prazosin blocks the alpha1 component" was in P5.3. Acetylcholine
    # is not: its curated range spans 4.3 to 6.5, a factor of 158.
    kds = [
        acetylcholine.muscarinic_kd(s, "scopolamine")
        for s in acetylcholine.MUSCARINIC_SUBTYPES
    ]
    assert max(kds) / min(kds) == pytest.approx(5.0, rel=0.02)

    ach_pki = [
        v for pair in acetylcholine.MUSCARINIC_PKI_ACETYLCHOLINE.values() for v in pair
    ]
    assert 10.0 ** (max(ach_pki) - min(ach_pki)) == pytest.approx(158.5, rel=0.01)

    # ...and scopolamine outbinds acetylcholine at every one of the five,
    # which is why it works as a tool at all. The margin is narrowest at M5
    # (398-fold), because M5 carries the tightest curated acetylcholine
    # value in the table and one of the weaker scopolamine ones.
    ratios = {
        subtype: acetylcholine.muscarinic_kd(subtype, "acetylcholine")
        / acetylcholine.muscarinic_kd(subtype, "scopolamine")
        for subtype in acetylcholine.MUSCARINIC_SUBTYPES
    }
    assert min(ratios, key=ratios.get) == "M5"
    assert ratios["M5"] == pytest.approx(398.1, rel=0.01)
    assert all(ratio > 390.0 for ratio in ratios.values())


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def test_every_koff_is_the_illustrative_kon_times_the_measured_affinity():
    assert acetylcholine.ILLUSTRATIVE_KON == 0.01
    checks = [
        (acetylcholine.receptor_site_alpha4beta2(), [
            acetylcholine.ALPHA4BETA2_KD_ACETYLCHOLINE,
            acetylcholine.ALPHA4BETA2_KD_NICOTINE,
            acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE,
        ]),
        (acetylcholine.receptor_site_alpha7(), [
            acetylcholine.ALPHA7_KD_ACETYLCHOLINE,
            acetylcholine.ALPHA7_KD_NICOTINE,
            acetylcholine.ALPHA7_IC50_MECAMYLAMINE,
        ]),
        (acetylcholine.receptor_site_m1(), [
            acetylcholine.muscarinic_kd("M1", "acetylcholine"),
            acetylcholine.muscarinic_kd("M1", "scopolamine"),
        ]),
        (acetylcholine.receptor_site_m2(), [
            acetylcholine.muscarinic_kd("M2", "acetylcholine"),
            acetylcholine.muscarinic_kd("M2", "scopolamine"),
        ]),
    ]
    for site, kds in checks:
        assert len(site.ligands) == len(kds)
        for lig, kd in zip(site.ligands, kds):
            assert lig.kon == acetylcholine.ILLUSTRATIVE_KON
            # The measured affinity is reproduced exactly; only the
            # timescale is illustrative.
            assert lig.koff / lig.kon == pytest.approx(kd, rel=1e-12)


def test_the_illustrative_kon_keeps_binding_faster_than_desensitization():
    # The constraint that sets the value: binding must not become the
    # rate-limiting step in front of the measured gating constants, or the
    # illustrative number would be silently deciding this module's results.
    for site in (
        acetylcholine.receptor_site_alpha4beta2(),
        acetylcholine.receptor_site_alpha7(),
    ):
        for i, lig in enumerate(site.orthosteric_ligands):
            assert lig.koff > 20.0 * site.k_des[i]
    # ...and it sits at the slow end of the only measured kon panel this
    # codebase has, so it does not assert unusually fast binding either.
    assert (
        noradrenaline.BETA_KON_SOTALOL
        < acetylcholine.ILLUSTRATIVE_KON
        < noradrenaline.BETA_KON_FORMOTEROL
    )


def test_the_muscarinic_sites_are_ordinary_two_state_sites_on_purpose():
    for site in (acetylcholine.receptor_site_m1(), acetylcholine.receptor_site_m2()):
        assert isinstance(site, ReceptorSite)
        assert not isinstance(site, DesensitizingReceptorSite)
        assert site.n_states == site.n == 2
    for site in (
        acetylcholine.receptor_site_alpha4beta2(),
        acetylcholine.receptor_site_alpha7(),
    ):
        assert isinstance(site, DesensitizingReceptorSite)
        assert site.n == 3  # two agonists plus the channel blocker
        assert site.n_states == 5


# ---------------------------------------------------------------------------
# The result the task exists to produce
# ---------------------------------------------------------------------------


def test_nicotine_desensitizes_alpha4beta2_far_below_the_concentration_that_activates_it():
    # [Fenster1997]'s central finding, and the one a two-state occupancy
    # model cannot express: activation EC50 15 uM, desensitization IC50
    # 0.061 uM, a 246-fold window.
    site = acetylcholine.receptor_site_alpha4beta2()
    window = (
        acetylcholine.ALPHA4BETA2_KD_NICOTINE
        / acetylcholine.ALPHA4BETA2_DESENSITIZATION_IC50_NICOTINE
    )
    assert window == pytest.approx(245.9, rel=0.01)

    # Inside the window: at 300 nM nicotine (the concentration
    # [Fenster1999] used, and a tobacco-related one) 32% of the receptor
    # pool is parked in the non-conducting state and 1.3% is conducting.
    # Their ratio is k_des/k_rec exactly, at every concentration.
    state = site_steady_state(site, [0.0, 300.0, 0.0])
    bound, desens, _ = site.split_states(state)
    assert desens[1] == pytest.approx(0.315, rel=0.01)
    assert bound[1] == pytest.approx(0.0134, rel=0.01)
    assert desens[1] / bound[1] == pytest.approx(site.k_des[1] / site.k_rec[1])

    # What the two-state model says about the same 300 nM: 2% of the pool
    # bound, all of it conducting, and no unavailable population at all.
    # The pharmacologically central quantity, the fraction of receptor
    # withdrawn from what endogenous acetylcholine could recruit, is 0 in
    # that model and 0.32 in this one.
    two_state = ReceptorSite(name="two_state", ligands=site.orthosteric_ligands)
    kd = acetylcholine.ALPHA4BETA2_KD_NICOTINE
    two_state_occ = site_steady_state(two_state, [0.0, 300.0])
    assert two_state.n_states == 2
    assert two_state_occ[1] == pytest.approx(300.0 / (300.0 + kd), rel=1e-9)
    assert two_state.activation(two_state_occ) == pytest.approx(two_state_occ[1])
    assert desens[1] > 16.0 * two_state_occ[1]


def test_the_two_state_model_cannot_reproduce_that_and_this_says_how():
    # A plain ReceptorSite given nicotine's measured affinity is monotone
    # in concentration and has no state to put a desensitized receptor in,
    # so its activation at 61 nM (the measured desensitization IC50) is
    # 0.4% of maximal rather than half of anything.
    site = acetylcholine.receptor_site_alpha4beta2()
    plain = ReceptorSite(name="plain", ligands=site.orthosteric_ligands)
    kd = acetylcholine.ALPHA4BETA2_KD_NICOTINE
    ic50 = acetylcholine.ALPHA4BETA2_DESENSITIZATION_IC50_NICOTINE

    occ_at_ic50 = ic50 / (ic50 + kd)
    assert occ_at_ic50 < 0.005
    assert plain.activation([0.0, occ_at_ic50]) == pytest.approx(occ_at_ic50)

    # The desensitizing site puts 8.7% of the pool into a non-conducting,
    # ligand-occupied state at the same concentration, 21-fold more than
    # the two-state model has bound at all. That is the state the two-state
    # model does not have.
    state = site_steady_state(site, [0.0, ic50, 0.0])
    assert site.desensitized_fraction(state) == pytest.approx(0.0869, rel=0.01)
    assert site.desensitized_fraction(state) / occ_at_ic50 == pytest.approx(
        21.4, rel=0.01
    )

    # 8.7% and not the ~50% the measured IC50 implies: this is the same
    # 10-fold shortfall reported in
    # test_the_predicted_affinity_shift_undershoots_the_measured_one_tenfold,
    # seen from the concentration axis instead of the affinity axis. The
    # model reaches half-desensitization at 10.0x the measured IC50.
    apparent_kd = acetylcholine.ALPHA4BETA2_KD_NICOTINE / site.apparent_affinity_factor()[1]
    at_apparent_kd = site_steady_state(site, [0.0, apparent_kd, 0.0])
    assert total_bound(site, at_apparent_kd) == pytest.approx(0.5, rel=1e-9)
    assert apparent_kd / ic50 == pytest.approx(10.04, rel=0.01)


@pytest.mark.parametrize("nicotine_efficacy", (1.0, 0.5))
def test_the_desensitization_results_do_not_depend_on_nicotines_efficacy(
    nicotine_efficacy,
):
    # NICOTINE_EFFICACY is a stated simplification, not a measurement, so
    # the results that matter are re-run at half of it.
    site = acetylcholine.receptor_site_alpha4beta2(nicotine_efficacy=nicotine_efficacy)
    state = site_steady_state(site, [0.0, 300.0, 0.0])
    bound, desens, _ = site.split_states(state)
    assert desens[1] > 9.0 * bound[1]
    assert site.desensitized_fraction(state) > 0.3


def test_alpha7_desensitizes_faster_and_recovers_faster_than_alpha4beta2():
    a4b2 = acetylcholine.receptor_site_alpha4beta2()
    a7 = acetylcholine.receptor_site_alpha7()
    # 7.4-fold faster onset, 46-fold faster recovery. Both from one table.
    assert a7.k_des[0] / a4b2.k_des[0] == pytest.approx(7.4, rel=0.01)
    assert a7.k_rec[0] / a4b2.k_rec[0] == pytest.approx(45.7, rel=0.01)

    # ...and the spread across [Fenster1997]'s five subtypes is the scale
    # against which those two ratios should be read: 71-fold in onset and
    # 218-fold in recovery, in one experiment.
    taus = list(acetylcholine.FENSTER1997_OTHER_SUBTYPES.values()) + [
        (
            acetylcholine.ALPHA4BETA2_TAU_DESENSITIZATION,
            acetylcholine.ALPHA4BETA2_TAU_RECOVERY,
        ),
        (acetylcholine.ALPHA7_TAU_DESENSITIZATION, acetylcholine.ALPHA7_TAU_RECOVERY),
    ]
    onsets = [t[0] for t in taus]
    recoveries = [t[1] for t in taus]
    assert max(onsets) / min(onsets) == pytest.approx(71.0, rel=0.01)
    assert max(recoveries) / min(recoveries) == pytest.approx(217.8, rel=0.01)


def test_the_predicted_affinity_shift_undershoots_the_measured_one_tenfold():
    # The prediction the model makes and gets wrong, reported rather than
    # fitted away. `1 + k_des/k_rec` is fixed by two measured constants;
    # the measured EC50/IC50 ratio is a third measurement of the same
    # quantity, from the same table.
    cases = (
        (
            acetylcholine.receptor_site_alpha4beta2(),
            acetylcholine.ALPHA4BETA2_KD_NICOTINE,
            acetylcholine.ALPHA4BETA2_DESENSITIZATION_IC50_NICOTINE,
            24.49,
            10.04,
        ),
        (
            acetylcholine.receptor_site_alpha7(),
            acetylcholine.ALPHA7_KD_NICOTINE,
            acetylcholine.ALPHA7_DESENSITIZATION_IC50_NICOTINE,
            4.80,
            14.42,
        ),
    )
    for site, kd, ic50, expected_factor, expected_shortfall in cases:
        factor = site.apparent_affinity_factor()[1]
        assert factor == pytest.approx(expected_factor, rel=0.001)
        measured = kd / ic50
        assert measured / factor == pytest.approx(expected_shortfall, rel=0.001)
        # The direction is the part that would falsify the single-
        # desensitized-state simplification if it flipped, so it is
        # asserted separately from the magnitude.
        assert factor < measured

    # The model's factor is not a free parameter: it is what the site
    # actually does. Half-maximal TOTAL binding sits at Kd / factor.
    site = acetylcholine.receptor_site_alpha7()
    apparent_kd = acetylcholine.ALPHA7_KD_NICOTINE / site.apparent_affinity_factor()[1]
    state = site_steady_state(site, [0.0, apparent_kd, 0.0])
    assert total_bound(site, state) == pytest.approx(0.5, rel=0.02)


def test_mecamylamine_blocks_without_displacing_acetylcholine():
    # Non-competitive, and this is what that costs the model to say: the
    # agonist's occupancy is unchanged and only the conducting fraction
    # falls. Modelling mecamylamine as a competitor would have moved the
    # first number too.
    for site, ic50 in (
        (acetylcholine.receptor_site_alpha4beta2(), acetylcholine.ALPHA4BETA2_IC50_MECAMYLAMINE),
        (acetylcholine.receptor_site_alpha7(), acetylcholine.ALPHA7_IC50_MECAMYLAMINE),
    ):
        ach_conc = 10.0 * site.orthosteric_ligands[0].koff / site.orthosteric_ligands[0].kon
        alone = site_steady_state(site, [ach_conc, 0.0, 0.0])
        blocked = site_steady_state(site, [ach_conc, 0.0, ic50])

        bound_alone, desens_alone, _ = site.split_states(alone)
        bound_blocked, desens_blocked, blocked_pool = site.split_states(blocked)
        assert bound_blocked[0] == pytest.approx(bound_alone[0], rel=1e-4)
        assert desens_blocked[0] == pytest.approx(desens_alone[0], rel=1e-4)
        # Half the channels are blocked at the blocker's own IC50, which is
        # exactly what a non-competitive IC50 measures.
        assert blocked_pool[0] == pytest.approx(0.5, rel=1e-3)
        assert site.activation(blocked) == pytest.approx(
            0.5 * site.activation(alone), rel=1e-3
        )


def test_scopolamine_does_displace_acetylcholine_at_the_muscarinic_sites():
    # The contrast: at M1 scopolamine and acetylcholine share one pool, so
    # adding scopolamine both blocks and displaces. `ReceptorSite` needed no
    # change for this, the same finding P5.1 recorded for
    # diazepam/flumazenil.
    site = acetylcholine.receptor_site_m1()
    ach_kd = acetylcholine.muscarinic_kd("M1", "acetylcholine")
    scop_kd = acetylcholine.muscarinic_kd("M1", "scopolamine")

    alone = site_steady_state(site, [10.0 * ach_kd, 0.0])
    with_scopolamine = site_steady_state(site, [10.0 * ach_kd, 100.0 * scop_kd])
    assert alone[0] > 0.9
    assert with_scopolamine[0] < 0.5 * alone[0]
    assert with_scopolamine[1] > 0.4
    assert site.activation(with_scopolamine) < 0.5 * site.activation(alone)


# ---------------------------------------------------------------------------
# The circuit, and P5.3's three rules
# ---------------------------------------------------------------------------


def test_rule_one_rest_is_exactly_the_maximum_sensitivity_point(network):
    balanced = stability.balanced_baselines(
        network, target_rates=stability.MAX_SENSITIVITY_RATE
    )
    for node, rebuilt in zip(network.nodes, balanced.nodes):
        assert abs(node.baseline_input - rebuilt.baseline_input) < 1e-12
    rates, residual = stability.steady_state(network)
    assert residual < 1e-6
    assert np.allclose(rates, 0.5, atol=1e-6)
    assert np.allclose(acetylcholine.rest_rates(network), rates, atol=1e-6)


def test_rule_two_the_contraction_certificate_holds(network):
    assessment = stability.assess(network)
    assert abs(assessment.contraction - 0.9) < 1e-9
    assert assessment.globally_certified
    assert assessment.unique
    assert assessment.ok


def test_rule_three_splits_one_unit_of_drive_between_three_receptors(network):
    # P5.3 established the per-NODE unit budget with two receptors on the
    # cortical node; this node carries three, so each gets a third.
    assert acetylcholine.CORTICAL_MECHANISM_GAIN == pytest.approx(1.0 / 3.0)
    assert 3 * acetylcholine.CORTICAL_MECHANISM_GAIN == pytest.approx(1.0)
    assert acetylcholine.M2_MECHANISM_GAIN == -1.0

    idx = network.index
    for total_drive, must_be_ok in ((1.0, True), (3.0, False)):
        drive = network.baseline_drive().copy()
        drive[idx["PFC_Pyramidal"]] += total_drive
        rates, residual = stability.steady_state(network, drive=drive)
        assert residual < 1e-6
        report = stability.check_saturation(network, rates, drive=drive)
        assert report.ok is must_be_ok
        if not must_be_ok:
            # The counterfactual that makes the rule a rule: at P5.1's
            # per-receptor unit the node rails.
            assert report.min_sensitivity < stability.SATURATION_FLOOR

    drive = network.baseline_drive().copy()
    drive[idx["BF_Cholinergic"]] += acetylcholine.M2_MECHANISM_GAIN
    rates, residual = stability.steady_state(network, drive=drive)
    assert residual < 1e-6
    assert stability.check_saturation(network, rates, drive=drive).ok
    assert rates[idx["BF_Cholinergic"]] < 0.03


def test_the_cortical_loop_is_the_one_p5_1_and_p5_3_already_built(network):
    assert (
        acetylcholine.CORTICAL_INHIBITION_WEIGHT
        == noradrenaline.CORTICAL_INHIBITION_WEIGHT
    )
    assert (
        acetylcholine.CORTICAL_RECURRENT_WEIGHT
        == noradrenaline.CORTICAL_RECURRENT_WEIGHT
    )
    # P5.1 hard-codes the two weights inside its network builder, so the
    # comparison against it has to go through the built matrices.
    raphe = serotonin.raphe_cortical_network()
    idx, r_idx = network.index, raphe.index
    W = np.asarray(network.connectivity)
    R = np.asarray(raphe.connectivity)
    assert (
        W[idx["PFC_Pyramidal"], idx["PFC_GABA"]]
        == R[r_idx["PFC_Pyramidal"], r_idx["PFC_GABA"]]
    )
    assert (
        W[idx["PFC_GABA"], idx["PFC_Pyramidal"]]
        == R[r_idx["PFC_GABA"], r_idx["PFC_Pyramidal"]]
    )
    # ...and the readout on the shared cortical node is literally the same.
    assert (
        acetylcholine.cortical_output_readout().weights
        == noradrenaline.cortical_output_readout().weights
        == serotonin.cortical_output_readout().weights
    )


def test_the_shared_illustrative_clearance_constants_have_not_drifted():
    assert acetylcholine.TOOL_COMPOUND_KE == noradrenaline.TOOL_COMPOUND_KE
    assert acetylcholine.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE
    assert acetylcholine.TOOL_COMPOUND_KE == dopamine.TOOL_COMPOUND_KE
    assert acetylcholine.ACETYLCHOLINE_KE == noradrenaline.NORADRENALINE_KE
    assert acetylcholine.ACETYLCHOLINE_KE == serotonin.SEROTONIN_KE
    assert acetylcholine.ACETYLCHOLINE_KE == dopamine.DOPAMINE_KE


def test_one_molecule_of_acetylcholine_clears_at_one_rate_across_four_systems():
    systems = acetylcholine.cholinergic_receptor_systems()
    kes = []
    for system in systems:
        for lig, model in zip(system.receptor_site.ligands, system.pk_models):
            if lig.name == "acetylcholine":
                kes.append(model.ke)
    assert len(kes) == 4
    assert len(set(kes)) == 1


# ---------------------------------------------------------------------------
# The scenario, wired together
# ---------------------------------------------------------------------------


def test_the_state_vector_has_more_receptor_states_than_ligands(sim):
    # Four systems, ten ligands (3 + 3 + 2 + 2), fourteen receptor states
    # (5 + 5 + 2 + 2). This inequality did not exist before P5.4 and it is
    # the engine-level shape of the whole task.
    assert sim.n_ligands == 10
    assert sim.r == 14
    assert sim.d == 10  # one-compartment PK, one state per ligand
    assert list(sim.state_rows(0)) == [0, 1, 2, 3, 4]
    assert list(sim.state_rows(2)) == [10, 11]
    assert list(sim.state_rows(3)) == [12, 13]
    assert sim.dose_vector(nicotine=1.0).shape == (10,)


def test_dose_vector_reaches_both_nicotinic_copies_of_one_drug(sim):
    doses = sim.dose_vector(nicotine=7.0, mecamylamine=3.0, scopolamine=1.0)
    assert list(doses) == [0.0, 7.0, 3.0, 0.0, 7.0, 3.0, 0.0, 1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        sim.dose_vector(varenicline=1.0)


def test_a_drug_free_run_sits_exactly_at_rest(sim):
    result = sim.run(
        doses=np.zeros(sim.n_ligands),
        t_span=(0.0, 10.0),
        t_eval=np.linspace(0.0, 10.0, 11),
        rates0=acetylcholine.rest_rates(),
    )
    assert np.allclose(result["rates"], 0.5, atol=1e-6)
    assert np.allclose(result["occupancy"], 0.0, atol=1e-9)


def test_a_nicotine_dose_leaves_the_two_nicotinic_sites_in_different_states(sim):
    # The pharmacology, run through the whole coupled system rather than at
    # the site level: one dose, two receptors on one node, and by ten
    # minutes alpha4beta2 is mostly desensitized while alpha7 -- which
    # recovers 46-fold faster -- holds a much larger conducting share of
    # whatever it has bound.
    result = sim.run(
        doses=sim.dose_vector(nicotine=1e4),
        t_span=(0.0, 10.0),
        t_eval=np.linspace(0.0, 10.0, 21),
        rates0=acetylcholine.rest_rates(),
    )
    systems = sim.receptor_systems
    a4b2, a7 = systems[0].receptor_site, systems[1].receptor_site
    end_a4b2 = result["occupancy"][sim.state_rows(0)][:, -1]
    end_a7 = result["occupancy"][sim.state_rows(1)][:, -1]

    bound4, desens4, _ = a4b2.split_states(end_a4b2)
    bound7, desens7, _ = a7.split_states(end_a7)
    assert desens4[1] > 4.0 * bound4[1]
    assert desens4[1] / (bound4[1] + desens4[1]) > desens7[1] / (bound7[1] + desens7[1])

    # Nothing is ever negative or over the pool, in the full run.
    for rows, site in ((sim.state_rows(0), a4b2), (sim.state_rows(1), a7)):
        block = result["occupancy"][rows]
        assert np.all(block >= -1e-9)
        assert np.all(block[:2].sum(axis=0) + block[2:4].sum(axis=0) <= 1.0 + 1e-9)


def test_desensitization_outlives_the_transmitter_that_caused_it(sim):
    # The clearest single statement of what the third state buys, and it
    # falls out of the two measured recovery constants. `ACETYLCHOLINE_KE`
    # clears the transmitter with a half-life of 8 seconds, so by ten
    # minutes every conducting and every competitive occupancy has returned
    # to zero -- but alpha4beta2, whose measured recovery constant is 86.9
    # min, is still an eighth desensitized, while alpha7, at 1.9 min, has
    # emptied. A two-state model has nowhere to put that eighth.
    result = sim.run(
        doses=sim.dose_vector(acetylcholine=2e4),
        t_span=(0.0, 10.0),
        t_eval=np.linspace(0.0, 10.0, 11),
        rates0=acetylcholine.rest_rates(),
    )
    a4b2 = sim.receptor_systems[0].receptor_site
    a7 = sim.receptor_systems[1].receptor_site
    end4 = result["occupancy"][sim.state_rows(0)][:, -1]
    end7 = result["occupancy"][sim.state_rows(1)][:, -1]

    assert result["concentrations"][0, -1] < 1e-6  # the transmitter is gone
    bound4, desens4, _ = a4b2.split_states(end4)
    bound7, desens7, _ = a7.split_states(end7)
    assert bound4[0] < 1e-3
    assert bound7[0] < 1e-3
    assert desens4[0] > 0.1
    assert desens7[0] < 0.01
    # The muscarinic sites, which have no third state, are simply empty.
    assert np.allclose(result["occupancy"][sim.state_rows(2)][:, -1], 0.0, atol=1e-6)


def test_an_antagonist_alone_is_inert_because_there_is_no_endogenous_tone(sim):
    # True of scopolamine and of mecamylamine, and it is a limitation
    # rather than a result: real scopolamine and real mecamylamine both do
    # plenty on their own, because a real basal forebrain fires tonically
    # and a real cortex has acetylcholine in it. P6.2 is where that
    # changes, and this assertion is what should then fail.
    for drug in ("scopolamine", "mecamylamine"):
        result = sim.run(
            doses=sim.dose_vector(**{drug: 1e4}),
            t_span=(0.0, 10.0),
            t_eval=np.linspace(0.0, 10.0, 11),
            rates0=acetylcholine.rest_rates(),
        )
        assert np.allclose(result["rates"], 0.5, atol=1e-6)


def test_readouts_are_the_shapes_the_other_transmitter_modules_use():
    tone = acetylcholine.cholinergic_tone_readout()
    assert tone.name == "cholinergic_tone"
    assert tone.weights == {"BF_Cholinergic": 1.0}
    assert acetylcholine.cortical_output_readout().name == "cortical_output"
