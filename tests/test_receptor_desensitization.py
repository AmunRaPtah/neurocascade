"""P5.4: the desensitized receptor state, tested on its own.

These tests are about `receptor.DesensitizingReceptorSite` as a mechanism,
with hand-picked round-number rate constants and no pharmacology in them.
The literature constants that drive it live in
`tests/test_acetylcholine_scenario.py`.
"""

import numpy as np
import pytest

from neurocascade import DesensitizingReceptorSite, Ligand, ReceptorSite


def _integrate(site, concentrations, minutes, dt=1e-3, state=None):
    """Forward Euler over the site's own RHS; small dt, short windows."""
    state = np.zeros(site.n_states) if state is None else np.array(state, float)
    for _ in range(int(minutes / dt)):
        state = state + dt * site.occupancy_rhs(state, concentrations)
    return state


def _agonist(kon=1.0, koff=1.0, efficacy=1.0, name="agonist"):
    return Ligand(name=name, kon=kon, koff=koff, efficacy=efficacy)


# ---------------------------------------------------------------------------
# Shape: what the site owns and how it reports it
# ---------------------------------------------------------------------------


def test_ligand_count_and_state_count_are_different_numbers():
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(name="a"), _agonist(name="b")],
        k_des=0.5,
        k_rec=0.1,
        channel_blockers=[_agonist(name="blocker", efficacy=0.0)],
    )
    # three ligands -> three PK models and three doses; seven states.
    assert site.n == 3
    assert [lig.name for lig in site.ligands] == ["a", "b", "blocker"]
    assert site.n_states == 2 * 2 + 1


def test_plain_receptor_site_reports_equal_counts():
    # The attribute has to exist on the base class too, or the engine would
    # need a branch per site type.
    site = ReceptorSite(name="site", ligands=[_agonist()])
    assert site.n_states == site.n == 1


def test_split_states_partitions_the_block_in_order():
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(name="a"), _agonist(name="b")],
        k_des=0.5,
        k_rec=0.1,
        channel_blockers=[_agonist(name="blocker", efficacy=0.0)],
    )
    block = np.array([0.1, 0.2, 0.03, 0.04, 0.7])
    bound, desens, blocked = site.split_states(block)
    assert list(bound) == [0.1, 0.2]
    assert list(desens) == [0.03, 0.04]
    assert list(blocked) == [0.7]
    assert site.free_fraction(block) == pytest.approx(1.0 - 0.1 - 0.2 - 0.03 - 0.04)
    assert site.desensitized_fraction(block) == pytest.approx(0.07)
    assert site.blocked_fraction(block) == pytest.approx(0.7)


def test_rates_may_be_given_per_ligand_and_are_validated():
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(name="a"), _agonist(name="b")],
        k_des=[0.5, 0.0],
        k_rec=[0.1, 0.1],
    )
    assert list(site.k_des) == [0.5, 0.0]

    with pytest.raises(ValueError):
        DesensitizingReceptorSite(
            name="site", ligands=[_agonist()], k_des=[0.1, 0.2], k_rec=0.1
        )
    with pytest.raises(ValueError):
        DesensitizingReceptorSite(name="site", ligands=[_agonist()], k_des=-1.0, k_rec=0.1)
    with pytest.raises(ValueError):
        # k_des > 0 with k_rec == 0 is a receptor that never comes back.
        DesensitizingReceptorSite(name="site", ligands=[_agonist()], k_des=0.5, k_rec=0.0)
    with pytest.raises(ValueError):
        DesensitizingReceptorSite(name="site", ligands=[], k_des=0.0, k_rec=1.0)


# ---------------------------------------------------------------------------
# Reduction: with no desensitization it must BE the old model
# ---------------------------------------------------------------------------


def test_zero_desensitization_rate_reproduces_the_two_state_site_exactly():
    ligands = [_agonist(kon=1.0, koff=0.2, name="ago"), _agonist(kon=0.7, koff=0.05, efficacy=0.0, name="anta")]
    plain = ReceptorSite(name="plain", ligands=ligands)
    desens = DesensitizingReceptorSite(
        name="desens", ligands=ligands, k_des=0.0, k_rec=1.0
    )
    occ = np.array([0.3, 0.25])
    state = np.array([0.3, 0.25, 0.0, 0.0])
    concentrations = [0.4, 1.1]

    d_plain = plain.occupancy_rhs(occ, concentrations)
    d_desens = desens.occupancy_rhs(state, concentrations)
    assert np.allclose(d_desens[:2], d_plain)
    assert np.allclose(d_desens[2:], 0.0)
    assert desens.activation(state) == pytest.approx(plain.activation(occ))


def test_a_ligand_with_zero_k_des_never_desensitizes_the_receptor():
    # An orthosteric competitive antagonist occupies the site without
    # driving the conformational change; its desensitized state must stay
    # at zero for all time even while the agonist's fills.
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(name="ago"), _agonist(name="anta", efficacy=0.0)],
        k_des=[1.0, 0.0],
        k_rec=[0.2, 0.2],
    )
    state = _integrate(site, [1.0, 1.0], minutes=5.0)
    _, desens, _ = site.split_states(state)
    assert desens[0] > 0.05
    assert desens[1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Conservation and bounds
# ---------------------------------------------------------------------------


def test_the_receptor_pool_is_conserved_and_never_leaves_the_simplex():
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(kon=5.0, koff=0.1, name="a"), _agonist(kon=2.0, koff=0.4, name="b")],
        k_des=1.5,
        k_rec=0.05,
        channel_blockers=[_agonist(kon=3.0, koff=0.3, efficacy=0.0, name="blocker")],
    )
    state = np.zeros(site.n_states)
    dt = 1e-3
    for _ in range(20000):
        state = state + dt * site.occupancy_rhs(state, [10.0, 10.0, 10.0])
        bound, desens, blocked = site.split_states(state)
        assert np.all(state >= -1e-12)
        assert bound.sum() + desens.sum() <= 1.0 + 1e-9
        assert blocked.sum() <= 1.0 + 1e-9
    assert site.free_fraction(state) >= -1e-9


def test_desensitized_receptor_still_occupies_the_pool():
    # Two ligands, only one of which desensitizes. The desensitized
    # population has to keep competing, or a desensitizing agonist would
    # look like it was disappearing from the receptor.
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(kon=1.0, koff=0.1, name="a"), _agonist(kon=1.0, koff=0.1, name="b")],
        k_des=[2.0, 0.0],
        k_rec=[0.02, 0.02],
    )
    state = _integrate(site, [1.0, 1.0], minutes=20.0)
    bound, desens, _ = site.split_states(state)
    # 'a' is mostly parked in the desensitized state, and that parked
    # population is what keeps 'b' from taking the whole pool.
    assert desens[0] > bound[0]
    assert bound[1] < 1.0 - desens[0]
    assert site.free_fraction(state) == pytest.approx(
        1.0 - bound.sum() - desens.sum()
    )


# ---------------------------------------------------------------------------
# The mechanism itself
# ---------------------------------------------------------------------------


def test_desensitized_receptor_contributes_nothing_to_activation():
    site = DesensitizingReceptorSite(
        name="site", ligands=[_agonist(efficacy=1.0)], k_des=1.0, k_rec=0.1
    )
    all_bound = np.array([0.6, 0.0])
    all_desensitized = np.array([0.0, 0.6])
    assert site.activation(all_bound) == pytest.approx(0.6)
    assert site.activation(all_desensitized) == pytest.approx(0.0)


def test_activation_peaks_and_then_falls_under_a_constant_agonist():
    # The property no two-state site can have: with the agonist held
    # constant, activation rises, peaks, and decays to a lower plateau as
    # the receptor accumulates in the desensitized state.
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(kon=2.0, koff=0.5)],
        k_des=1.0,
        k_rec=0.05,
    )
    dt = 1e-3
    state = np.zeros(site.n_states)
    trace = []
    for _ in range(30000):
        state = state + dt * site.occupancy_rhs(state, [1.0])
        trace.append(site.activation(state))
    trace = np.array(trace)
    peak = int(np.argmax(trace))
    assert 0 < peak < len(trace) - 1  # a genuine interior maximum
    assert trace[-1] < 0.5 * trace[peak]

    # The same site with k_des = 0 is monotone, which is the contrast.
    plain = DesensitizingReceptorSite(
        name="plain", ligands=[_agonist(kon=2.0, koff=0.5)], k_des=0.0, k_rec=0.05
    )
    state = np.zeros(plain.n_states)
    monotone = []
    for _ in range(30000):
        state = state + dt * plain.occupancy_rhs(state, [1.0])
        monotone.append(plain.activation(state))
    assert np.all(np.diff(np.array(monotone)) >= -1e-12)


def test_steady_state_desensitized_to_bound_ratio_is_k_des_over_k_rec():
    site = DesensitizingReceptorSite(
        name="site", ligands=[_agonist(kon=1.0, koff=0.5)], k_des=0.8, k_rec=0.2
    )
    state = _integrate(site, [2.0], minutes=200.0, dt=2e-3)
    bound, desens, _ = site.split_states(state)
    assert desens[0] / bound[0] == pytest.approx(0.8 / 0.2, rel=1e-3)
    assert np.allclose(site.occupancy_rhs(state, [2.0]), 0.0, atol=1e-6)


def test_apparent_affinity_is_tightened_by_exactly_one_plus_k_des_over_k_rec():
    # The Katz-Thesleff consequence: total bound (conducting +
    # desensitized) is half-maximal at Kd / (1 + k_des/k_rec).
    kon, koff = 1.0, 4.0  # Kd = 4
    k_des, k_rec = 3.0, 1.0  # factor 4 -> apparent Kd 1.0
    site = DesensitizingReceptorSite(
        name="site", ligands=[_agonist(kon=kon, koff=koff)], k_des=k_des, k_rec=k_rec
    )
    assert site.apparent_affinity_factor()[0] == pytest.approx(4.0)

    apparent_kd = (koff / kon) / site.apparent_affinity_factor()[0]
    state = _integrate(site, [apparent_kd], minutes=200.0, dt=2e-3)
    bound, desens, _ = site.split_states(state)
    assert bound[0] + desens[0] == pytest.approx(0.5, rel=1e-3)

    # ... and a two-state site with the same Kd is only 20% occupied there.
    plain = ReceptorSite(name="plain", ligands=[_agonist(kon=kon, koff=koff)])
    assert apparent_kd / (apparent_kd + koff / kon) == pytest.approx(0.2)
    assert plain.n_states == 1


def _wash_out(site, loaded, minutes, dt=1e-3):
    state = np.array(loaded, dtype=float)
    zero = np.zeros(site.n)
    trace = []
    for step in range(int(minutes / dt)):
        state = state + dt * site.occupancy_rhs(state, zero)
        trace.append(site.desensitized_fraction(state))
    return state, np.array(trace)


def test_recovery_after_washout_is_limited_by_k_rec_AND_by_re_entry():
    # Recovery in the Katz-Thesleff cycle is not simply exp(-k_rec * t):
    # a receptor leaving the desensitized state arrives BOUND, and from
    # there it can either unbind (koff) or desensitize again (k_des). Only
    # the koff branch is real recovery, so the observed rate is
    # k_rec * koff / (koff + k_des). This test pins both limits, because
    # the fast-desensitizing limit is the one the alpha7 constants live in.
    k_rec, k_des = 0.25, 5.0

    fast_unbinding = DesensitizingReceptorSite(
        name="fast_off", ligands=[_agonist(kon=5.0, koff=50.0)], k_des=k_des, k_rec=k_rec
    )
    loaded = _integrate(fast_unbinding, [10.0], minutes=40.0, dt=2e-4)
    assert fast_unbinding.desensitized_fraction(loaded) > 0.7

    expected_rate = k_rec * 50.0 / (50.0 + k_des)
    half_life = np.log(2.0) / expected_rate
    state, _ = _wash_out(fast_unbinding, loaded, half_life, dt=2e-4)
    assert fast_unbinding.desensitized_fraction(state) == pytest.approx(
        0.5 * fast_unbinding.desensitized_fraction(loaded), rel=0.05
    )

    # With koff far BELOW k_des the same k_rec produces a much slower
    # recovery, because most receptors that recover simply desensitize
    # again before they can unbind.
    slow_unbinding = DesensitizingReceptorSite(
        name="slow_off", ligands=[_agonist(kon=5.0, koff=0.5)], k_des=k_des, k_rec=k_rec
    )
    loaded_slow = _integrate(slow_unbinding, [10.0], minutes=40.0)
    state_slow, _ = _wash_out(slow_unbinding, loaded_slow, half_life)
    assert slow_unbinding.desensitized_fraction(state_slow) > 0.8 * (
        slow_unbinding.desensitized_fraction(loaded_slow)
    )

    # Both recover completely given enough time; the desensitized state is
    # a delay, not a sink.
    for site, loaded_state in (
        (fast_unbinding, loaded),
        (slow_unbinding, loaded_slow),
    ):
        state, _ = _wash_out(site, loaded_state, 400.0, dt=2e-3)
        assert site.free_fraction(state) > 0.99


def test_recovery_is_monotone_and_never_overshoots():
    site = DesensitizingReceptorSite(
        name="site", ligands=[_agonist(kon=5.0, koff=20.0)], k_des=2.0, k_rec=0.5
    )
    loaded = _integrate(site, [10.0], minutes=20.0, dt=2e-4)
    _, trace = _wash_out(site, loaded, 20.0, dt=2e-4)
    assert np.all(np.diff(trace) <= 1e-12)
    assert trace[-1] < 1e-3


# ---------------------------------------------------------------------------
# The channel blocker: not a competitor
# ---------------------------------------------------------------------------


def test_channel_blocker_does_not_compete_for_the_orthosteric_pool():
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(kon=1.0, koff=1.0)],
        k_des=0.0,
        k_rec=1.0,
        channel_blockers=[_agonist(kon=1.0, koff=1.0, efficacy=0.0, name="blocker")],
    )
    without = _integrate(site, [1.0, 0.0], minutes=50.0)
    with_blocker = _integrate(site, [1.0, 100.0], minutes=50.0)
    bound_without, _, _ = site.split_states(without)
    bound_with, _, blocked = site.split_states(with_blocker)

    # Agonist occupancy is untouched: this is what "non-competitive" means,
    # and it is what modelling mecamylamine in the competitive pool would
    # have got wrong.
    assert bound_with[0] == pytest.approx(bound_without[0], rel=1e-6)
    assert blocked[0] > 0.98
    assert site.activation(with_blocker) < 0.02 * site.activation(without)


def test_blocker_at_its_own_kd_halves_activation():
    kd = 3.0
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(kon=1.0, koff=1.0)],
        k_des=0.0,
        k_rec=1.0,
        channel_blockers=[
            _agonist(kon=1.0, koff=kd, efficacy=0.0, name="blocker")
        ],
    )
    free = _integrate(site, [1.0, 0.0], minutes=80.0)
    blocked = _integrate(site, [1.0, kd], minutes=80.0)
    assert site.activation(blocked) == pytest.approx(0.5 * site.activation(free), rel=1e-3)


def test_two_blockers_share_one_channel_pool():
    site = DesensitizingReceptorSite(
        name="site",
        ligands=[_agonist(kon=1.0, koff=1.0)],
        k_des=0.0,
        k_rec=1.0,
        channel_blockers=[
            _agonist(kon=1.0, koff=1.0, efficacy=0.0, name="b1"),
            _agonist(kon=1.0, koff=1.0, efficacy=0.0, name="b2"),
        ],
    )
    state = _integrate(site, [1.0, 5.0, 5.0], minutes=80.0)
    assert site.blocked_fraction(state) <= 1.0 + 1e-9
    _, _, blocked = site.split_states(state)
    assert blocked[0] == pytest.approx(blocked[1], rel=1e-6)
