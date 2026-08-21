"""P8.2: the three tiers of held-out validation, and the verdicts they reach.

Split as this repository splits calibration tests: the RULE (checked on
constructed inputs whose answer is known), and the RESULT (the verdicts the
committed data actually produce, asserted so they cannot drift silently).
"""

import numpy as np
import pytest

from neurocascade import heldout as ho
from neurocascade.calibration import heldout_forward as hf
from neurocascade.calibration import heldout_validation as hv


COMPOUNDS = sorted(ho.HELD_OUT)


# ---------------------------------------------------------------------------
# The rule: coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPOUNDS)
def test_every_compound_gets_a_verdict_on_every_chain_link(name):
    status = hv.chain_status(ho.HELD_OUT[name])
    assert set(status.links) == set(hv.CHAIN_LINKS)
    for link in hv.CHAIN_LINKS:
        assert isinstance(status.links[link], bool)
        assert status.detail[link].strip(), link


def test_a_half_life_alone_does_not_count_as_pharmacokinetics():
    """`chain_status` asks `drug_pk` whether it can BUILD a model, not
    whether some PK number was found. Alprazolam has a well-replicated rat
    terminal half-life and no volume of distribution, and a half-life alone
    cannot turn a mg/kg dose into a concentration. Counting it would make the
    coverage table claim the opposite of what the code can do."""
    status = hv.chain_status(ho.HELD_OUT["alprazolam"])
    assert status.links["pharmacokinetics"] is False
    assert "builds no model" in status.detail["pharmacokinetics"]


def test_no_held_out_compound_has_a_complete_chain():
    """The headline result of P8.2, asserted rather than described. If a
    compound ever does acquire a complete chain this test must be changed in
    the same commit that adds the missing link, which is the point."""
    complete = [
        name for name in COMPOUNDS
        if hv.chain_status(ho.HELD_OUT[name]).complete
    ]
    assert complete == [], complete


def test_the_missing_link_is_the_same_one_for_every_compound():
    """Not seven separate gaps: one gap, seven times. Every compound is
    missing the plasma-to-brain-unbound partition, and that is what makes
    this a finding about the literature rather than about the compounds."""
    for name in COMPOUNDS:
        status = hv.chain_status(ho.HELD_OUT[name])
        assert "brain_partition" in status.missing, name


# ---------------------------------------------------------------------------
# The rule: the activation sign, and what makes an antagonist inert
# ---------------------------------------------------------------------------


def test_an_agonist_adds_activation_and_a_blocker_at_a_tone_free_site_does_not():
    assert hv._delta_activation_sign("MOR", "agonist") == +1
    assert hv._delta_activation_sign("GABA_A_BZD",
                                     "positive_allosteric_modulator") == +1
    # A2A, NMDA and the benzodiazepine site have no measured basal
    # transmitter in this repository, so a blocker there displaces nothing.
    assert hv._delta_activation_sign("A2A", "antagonist") == 0
    assert hv._delta_activation_sign("NMDA", "channel_blocker") == 0
    # D2 does have a measured tone, so a blocker there has something to take.
    assert hv._delta_activation_sign("D2", "antagonist") == -1


def test_a_transporter_ligand_reaches_no_endpoint_by_either_sign():
    for kind in ("uptake_blocker", "substrate_releaser"):
        assert hv._delta_activation_sign("DAT", kind) == 0


def test_caffeine_is_predicted_to_do_nothing_and_that_is_falsifiable():
    """The model's least defensible prediction, and it must not escape being
    scored. A2A is modelled, caffeine is an antagonist there, and P6.2
    measured no basal adenosine, so the predicted effect is exactly zero at
    every dose and every gain."""
    caffeine = ho.HELD_OUT["caffeine"]
    assert caffeine.modelled_targets == ("A2A",)
    assert hv.behavioural_path_status_for(caffeine, "A2A") == "antagonist-no-tone"
    assert hv.delta_activation_sign_for(caffeine, "A2A") == 0


# ---------------------------------------------------------------------------
# The rule: the tone ceiling
# ---------------------------------------------------------------------------


def test_the_tone_ceiling_covers_exactly_the_targets_with_a_measured_tone():
    from neurocascade.legacy import endogenous_tone as tone

    assert set(hv.tone_ceiling()) == {t.name for t in tone.TONE_TARGETS}


def test_d2_is_the_only_target_where_a_blocker_can_remove_much():
    """A measured consequence of P6.2's tone table with real predictive
    force: at every target but D2 and 5HT1A, a full blocker moves activation
    by under 6%, however potent it is. Several antagonists in this set are
    therefore bounded to near-inertness before any dose is chosen."""
    ceiling = hv.tone_ceiling()
    assert ceiling["D2"] > 0.5
    assert ceiling["5HT1A"] > 0.2
    for target in ("H1", "M1", "M2", "alpha1", "beta", "alpha7", "D1"):
        assert ceiling[target] < 0.01, target


# ---------------------------------------------------------------------------
# The rule: monotonicity
# ---------------------------------------------------------------------------


def _observation(dose, ratio, sd):
    return ho.BehaviouralObservation(
        "probe", "anxiety", "rat", "IP", dose, measure="m",
        direction="decrease", reporting="tabulated", source="s",
        ratio=ratio, ratio_sd=sd, n_vehicle=10, n_drug=10,
    )


def test_a_monotone_series_does_not_turn_over():
    series = [_observation(0.1, 1.0, 0.01), _observation(0.3, 0.9, 0.01),
              _observation(1.0, 0.8, 0.01)]
    verdict = hv.monotonicity_verdict("probe", "anxiety", series)
    assert verdict.turns_over is False
    assert verdict.significant is False


def test_a_turnover_outside_the_error_bars_refutes():
    series = [_observation(0.1, 1.0, 0.01), _observation(0.3, 0.5, 0.01),
              _observation(1.0, 1.0, 0.01)]
    verdict = hv.monotonicity_verdict("probe", "anxiety", series)
    assert verdict.turns_over is True
    assert verdict.significant is True


def test_a_turnover_inside_the_error_bars_refutes_nothing():
    """The guard that keeps this test honest. A non-monotone point estimate
    buried in noise is not evidence against anything, and reporting it as a
    refutation would be the mirror image of widening a tolerance to pass."""
    series = [_observation(0.1, 1.0, 0.5), _observation(0.3, 0.9, 0.5),
              _observation(1.0, 1.0, 0.5)]
    verdict = hv.monotonicity_verdict("probe", "anxiety", series)
    assert verdict.turns_over is True
    assert verdict.significant is False


def test_two_dose_levels_cannot_be_tested_for_a_turnover():
    series = [_observation(0.1, 1.0, 0.01), _observation(1.0, 0.5, 0.01)]
    assert hv.monotonicity_verdict("probe", "anxiety", series) is None


def test_both_steps_must_resolve_for_a_turnover_to_count():
    """A monotone series with one noisy point should not be called a
    turnover. Here the fall is resolved and the rise is not."""
    series = [_observation(0.1, 1.0, 0.01), _observation(0.3, 0.5, 0.01),
              _observation(1.0, 0.52, 0.4)]
    verdict = hv.monotonicity_verdict("probe", "anxiety", series)
    assert verdict.turns_over is True
    assert verdict.significant is False


# ---------------------------------------------------------------------------
# The result: what the committed data actually says
# ---------------------------------------------------------------------------


def test_the_alprazolam_series_turns_over_at_one_sigma_and_not_at_two():
    """The sharpest single result in P8.2, and it is deliberately reported at
    both thresholds. The turnover in [Engel2018]'s elevated-plus-maze series
    exceeds one standard deviation on the rising and the falling step, and
    does not survive 1.96, so it is suggestive rather than decisive. Claiming
    only the threshold that passes would be choosing the test after seeing
    the data."""
    series = ho.HELD_OUT["alprazolam"].behavioural_observations
    at_one = hv.monotonicity_verdict("alprazolam", "anxiety", series, n_sigma=1.0)
    at_two = hv.monotonicity_verdict("alprazolam", "anxiety", series, n_sigma=1.96)
    assert at_one.turns_over is True
    assert at_one.significant is True
    assert at_two.significant is False


def test_only_one_compound_has_three_quantitative_dose_levels():
    """Which is why the monotonicity test, the strongest instrument in this
    task, runs exactly once."""
    testable = []
    for name in COMPOUNDS:
        compound = ho.HELD_OUT[name]
        for endpoint in {o.endpoint for o in compound.behavioural_observations}:
            if hv.monotonicity_verdict(
                name, endpoint, compound.behavioural_observations
            ) is not None:
                testable.append((name, endpoint))
    assert testable == [("alprazolam", "anxiety")]


def test_every_occupancy_observation_is_blocked_and_the_blocker_is_named():
    """Tier 1 is the tier with no unfitted parameter in it, so what blocks it
    is named per compound rather than summarised away."""
    total = 0
    for name in COMPOUNDS:
        gap = hv.occupancy_gap(ho.HELD_OUT[name])
        total += gap["observations"]
        if gap["observations"]:
            assert gap["blocked_by"], name
            assert "brain_partition" in gap["blocked_by"], name
    assert total >= 5, total


# ---------------------------------------------------------------------------
# The result: reachability and the magnitude error
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bzd_response():
    model, _ = hf.build_multi_target_model(["GABA_A_BZD"], "test-bzd")
    return hf.sweep_target(
        model, "GABA_A_BZD", x_grid=hf.response_grid(points_per_decade=2)
    )


def test_the_alprazolam_effect_is_within_the_models_reach(bzd_response):
    """And that is not a success. Being reachable means some gain reproduces
    it; the gain is unidentified, and `magnitude_result` says so in its
    verdict rather than reporting a zero residual as a hit."""
    observation = [
        o for o in ho.HELD_OUT["alprazolam"].behavioural_observations
        if o.is_quantitative and o.dose_mg_per_kg == 0.3
    ][0]
    result = hv.magnitude_result(
        "alprazolam", "GABA_A_BZD", bzd_response, observation
    )
    assert result.verdict == "reachable-gain-unidentified"
    assert result.residual_sd == 0.0
    assert result.implied_x is not None


def test_an_unreachable_observation_reports_its_error_in_sigma(bzd_response):
    """The quantitative error this task promises per compound per endpoint,
    exercised on a constructed observation outside the reachable set so that
    the arithmetic is checkable by hand."""
    ranges = hf.reachable_index_range(bzd_response)
    beyond = ranges["analgesia"].ratio_hi + 0.2
    observation = ho.BehaviouralObservation(
        "probe", "analgesia", "rat", "IP", 1.0, measure="m",
        direction="increase", reporting="tabulated", source="s",
        ratio=beyond, ratio_sd=0.1,
    )
    result = hv.magnitude_result("probe", "GABA_A_BZD", bzd_response, observation)
    assert result.verdict == "out-of-reach"
    assert result.residual_sd == pytest.approx(0.2 / 0.1, rel=1e-6)


def test_the_predicted_direction_of_alprazolam_anxiety_is_a_decrease(
    bzd_response
):
    """A parameter-free prediction that is CORRECT, and worth pinning as
    such: a benzodiazepine-site modulator potentiates inhibition brain-wide,
    which lowers the anxiety index, and [Engel2018]'s rats spent more time on
    the open arms. The direction test is the one tier of this task the model
    passes."""
    compound = ho.HELD_OUT["alprazolam"]
    x_sign = (hv.delta_activation_sign_for(compound, "GABA_A_BZD")
              * hf.transduction_sign("GABA_A_BZD")[1])
    assert x_sign == +1
    direction, slope = hv.endpoint_direction(bzd_response, "anxiety", x_sign)
    assert direction == "decrease"
    assert slope < 0.0
    observed = [
        o for o in compound.behavioural_observations
        if o.is_quantitative and o.dose_mg_per_kg == 0.3
    ][0]
    assert observed.direction == direction
