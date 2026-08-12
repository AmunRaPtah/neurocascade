"""Tests for `neurocascade.calibration.identifiability` (P2.3).

P2.3's product is a set of CLAIMS about what the behavioural evidence can
and cannot determine, so what is worth testing is the claims, not the
plotting. In particular:

1. The closed-form reduction the whole structural argument rests on really
   does reproduce the integrated forward model. If it did not, the
   "eight parameters, four combinations" result would be about a different
   model than the one this codebase runs.
2. The two exact degeneracies are exact: time constants do not move a
   steady-state observable at all, and a circuit relabelled to preserve
   {A, B, C, D} predicts the same thing.
3. The Fisher information matrix built from one observation has rank one,
   whatever it is evaluated on.
4. The profile machinery reports an open interval as open rather than
   silently clipping it to the scan range, and the drug gain's regime-A
   interval agrees with P2.2's independently sampled posterior - which is
   the cross-check that this module's likelihood is the same likelihood
   `bayesian_calibration.py` samples.
5. The conclusions the repository states are the conclusions the code
   computes: no weight and no tau is identifiable, and nothing here has
   quietly added a circuit parameter to the production fit.

The MCMC-based tests are `slow` for the same reason
`tests/test_bayesian_calibration.py` is: they run real chains.
"""

import numpy as np
import pytest

from neurocascade import scenarios
from neurocascade.calibration import behavioral_evidence
from neurocascade.calibration import circuit_forward as cfwd
from neurocascade.calibration import identifiability as ident
from neurocascade.calibration.priors import (
    CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS,
    FREE_PARAM_NAMES,
)


#: Grid resolution for the tests that need a whole report. Nine points per
#: parameter is far too coarse to LOCATE a confidence bound, and is plenty to
#: tell a flat profile from a bounded one, which is all the verdicts turn on.
#: The production artifacts under results/identifiability/ are generated at 41
#: by `scripts/run_identifiability.py`.
_TEST_GRID_POINTS = 9


@pytest.fixture(scope="module")
def report():
    """One report shared by every test that needs it: each one costs a few
    hundred circuit integrations, and recomputing it per test would put this
    module over the per-task gate's time budget for no extra coverage."""
    return ident.identifiability_report(grid_points=_TEST_GRID_POINTS)


# ---------------------------------------------------------------------------
# 1. The reduction is a reduction OF THIS MODEL.
# ---------------------------------------------------------------------------
def test_closed_form_reproduces_the_integrated_forward_model():
    """The structural argument is made in closed form; the codebase runs an
    integrator. If those two disagree the argument is about a different
    model, so this is the test the rest of the module rests on."""
    activation = ident.diazepam_activation()
    combos = ident.structural_combinations()
    for gain in (0.0, 0.0742, 0.5, 2.0):
        integrated = ident.predicted_anxiety_ratio(None, gain, activation)
        closed = ident.closed_form_ratio(
            combos["A"], combos["B"], combos["C"], combos["D"], gain, activation)
        assert closed == pytest.approx(integrated, abs=1e-7)


def test_closed_form_still_matches_away_from_the_nominal_circuit():
    """Not just at one point: the reduction has to hold over the region the
    profile likelihood actually explores."""
    activation = ident.diazepam_activation()
    for overrides in (
        {"gain_BLA": 0.4, "baseline_input_BLA": 0.7},
        {"weight_CeA__BLA": 12.0, "gain_CeA": 0.9},
        {"weight_BLA__GABA_Interneuron": -0.4, "baseline_input_CeA": -1.9},
    ):
        combos = ident.structural_combinations(overrides)
        integrated = ident.predicted_anxiety_ratio(overrides, 0.3, activation)
        closed = ident.closed_form_ratio(
            combos["A"], combos["B"], combos["C"], combos["D"], 0.3, activation)
        assert closed == pytest.approx(integrated, abs=1e-6)


def test_the_prediction_at_the_scenario_values_reproduces_p2_2s_target():
    """This module's forward model is P2.2's forward model, so at P2.2's
    fitted gain it must land on P2.2's observation."""
    predicted = ident.predicted_anxiety_ratio()
    assert predicted == pytest.approx(ident.OBSERVED_RATIO, abs=0.2 * ident.OBSERVED_RATIO_SD)


# ---------------------------------------------------------------------------
# 2. The two exact degeneracies.
# ---------------------------------------------------------------------------
def test_time_constants_do_not_move_the_observable_at_all():
    """A steady state does not depend on how fast it is reached. Checked over
    the whole scan range, not just at a perturbation, because the claim is
    that the derivative is zero everywhere and not merely small locally."""
    activation = ident.diazepam_activation()
    reference = ident.predicted_anxiety_ratio(None, 0.0742, activation)
    for name in [p.name for p in ident.AMYGDALA_PARAMETERS if p.kind == "tau"]:
        param = ident.AMYGDALA_PARAMETERS_BY_NAME[name]
        for value in (param.scan[0], param.nominal * 0.5, param.scan[1]):
            moved = ident.predicted_anxiety_ratio({name: value}, 0.0742, activation)
            assert moved == pytest.approx(reference, abs=1e-6)


def test_a_relabelled_circuit_predicts_the_same_thing_for_every_gain():
    """The {A,B,C,D} collapse, demonstrated by construction rather than
    inferred from flat profiles."""
    activation = ident.diazepam_activation()
    equivalent = ident.equivalent_parameterisation()
    for gain in (0.0, 0.05, 0.0742, 0.5, 3.0):
        assert ident.predicted_anxiety_ratio(equivalent, gain, activation) == pytest.approx(
            ident.predicted_anxiety_ratio(None, gain, activation), abs=1e-6
        )


def test_the_relabelled_circuit_is_genuinely_different():
    """Guards the test above from passing trivially: at least one parameter
    has to have moved by more than a factor of two, or 'two circuits, one
    prediction' would be a rounding difference."""
    equivalent = ident.equivalent_parameterisation()
    ratios = [
        abs(value / ident.AMYGDALA_PARAMETERS_BY_NAME[name].nominal)
        for name, value in equivalent.items()
    ]
    assert max(ratios) > 2.0 or min(ratios) < 0.5
    assert len(equivalent) >= 6  # it relabels most of the circuit, not one number


def test_the_combinations_are_unchanged_by_the_relabelling():
    original = ident.structural_combinations()
    relabelled = ident.structural_combinations(ident.equivalent_parameterisation())
    for key in original:
        assert relabelled[key] == pytest.approx(original[key], rel=1e-12)


# ---------------------------------------------------------------------------
# 3. One observation, one degree of freedom.
# ---------------------------------------------------------------------------
def test_fisher_information_from_one_observation_has_rank_one():
    values, _ = ident.fisher_spectrum()
    assert values[0] > 1.0
    # Every other eigenvalue is at machine precision relative to the first.
    assert np.max(np.abs(values[1:])) < 1e-9 * values[0]


def test_the_informed_direction_is_dominated_by_the_drug_gain():
    """Not a claim that it HAS to be - it is a fact about this circuit, and
    it is the reason P2.2 spending the one degree of freedom on the gain was
    the right call rather than an arbitrary one."""
    names = [p.name for p in ident.AMYGDALA_PARAMETERS]
    _, vectors = ident.fisher_spectrum(names)
    leading = np.abs(vectors[:, 0])
    gain_index = names.index("gaba_a_connection_modulation_gain")
    assert leading[gain_index] == pytest.approx(leading.max())


def test_taus_have_no_component_in_the_informed_direction():
    names = [p.name for p in ident.AMYGDALA_PARAMETERS]
    _, vectors = ident.fisher_spectrum(names)
    leading = np.abs(vectors[:, 0])
    for name in [p.name for p in ident.AMYGDALA_PARAMETERS if p.kind == "tau"]:
        assert leading[names.index(name)] < 1e-6


# ---------------------------------------------------------------------------
# 4. The profile machinery behaves.
# ---------------------------------------------------------------------------
def test_profile_interval_reports_an_open_side_as_open():
    grid = np.linspace(0.0, 1.0, 11)
    flat = np.zeros_like(grid)
    assert ident.profile_interval(grid, flat) == (None, None)

    one_sided = np.where(grid < 0.2, 10.0, 0.0)
    lower, upper = ident.profile_interval(grid, one_sided)
    assert lower is not None and upper is None

    parabola = 100.0 * (grid - 0.5) ** 2
    lower, upper = ident.profile_interval(grid, parabola)
    assert lower is not None and upper is not None
    assert lower < 0.5 < upper


def test_bounds_at_searched_edge_distinguishes_a_box_bound_from_a_data_bound():
    grid = np.linspace(0.0, 1.0, 11)
    interior = 100.0 * (grid - 0.5) ** 2
    assert ident.bounds_at_searched_edge(
        grid, ident.profile_interval(grid, interior)) == (False, False)
    # A profile that only exceeds the threshold in the outermost cell.
    edge = np.where((grid <= 0.0) | (grid >= 1.0), 10.0, 0.0)
    assert ident.bounds_at_searched_edge(
        grid, ident.profile_interval(grid, edge)) == (True, True)


def test_the_drug_gains_own_profile_agrees_with_p2_2s_posterior():
    """The likelihood cross-check. P2.2 sampled a posterior for this gain
    with PyMC; this module computes a profile likelihood for it with an
    optimiser. Different machinery, same likelihood, so the 95% profile
    interval must bracket P2.2's 89% ETI [0.0431, 0.1062] and the fitted
    0.0742 must sit inside it."""
    grid, profile, _ = ident.profile_chi_square(
        "gaba_a_connection_modulation_gain", free_drug_gain=False, points=61)
    lower, upper = ident.profile_interval(grid, profile)
    assert lower is not None and upper is not None
    assert lower < scenarios.GABA_A_CONNECTION_MODULATION_GAIN < upper
    assert lower <= 0.0431 and upper >= 0.1062


_ARBITRARY_CIRCUITS = (
    None,
    {"weight_CeA__BLA": 0.5, "gain_BLA": 9.0},
    {"baseline_input_CeA": 0.1, "tau_BLA": 12.0},
    {"gain_CeA": 0.6, "baseline_input_BLA": 4.0, "weight_BLA__GABA_Interneuron": -11.0},
)


def test_the_mechanistic_null_is_rejected_whatever_the_circuit():
    """The one statement in this analysis that survives freeing every
    circuit parameter: a drug gain of exactly zero predicts a ratio of
    exactly 1 for EVERY circuit, so its chi-square does not depend on any
    illustrative number.

    Asserted twice, because it is an algebraic identity of the model and
    only approximately a property of the integrator. The closed form has to
    give exactly 1.0 (both legs are literally the same expression at gain
    zero); the integrated path has to agree to the integrator's own declared
    tolerance and no better, which is why the second bound is read off
    `circuit_forward._STEADY_STATE_RTOL` rather than chosen.
    """
    activation = ident.diazepam_activation()
    for overrides in _ARBITRARY_CIRCUITS + (ident.equivalent_parameterisation(),):
        combos = ident.structural_combinations(overrides)
        assert ident.closed_form_ratio(
            combos["A"], combos["B"], combos["C"], combos["D"], 0.0, activation
        ) == 1.0
        ratio = ident.predicted_anxiety_ratio(overrides, 0.0, activation)
        assert ratio == pytest.approx(1.0, abs=100.0 * cfwd._STEADY_STATE_RTOL)
    assert ident.chi_square(1.0) > ident.PROFILE_THRESHOLD_CHI2


def test_the_nulls_residual_is_integrator_error_and_shrinks_with_tolerance():
    """Backs the tolerance in the test above rather than asserting it: the
    residual at gain zero is not a property of the model, so tightening the
    integration by three orders of magnitude has to shrink it."""
    activation = ident.diazepam_activation()
    overrides = {"gain_CeA": 0.6, "baseline_input_BLA": 4.0}
    network = ident.build_amygdala_network(overrides)
    readout = scenarios.anxiety_index_readout()
    modulation = {ident.MODULATED_CONNECTION: 0.0}

    def ratio(rtol, atol):
        baseline = cfwd.steady_state_rates(
            network, modulation, 0.0, rtol=rtol, atol=atol)
        drugged = cfwd.steady_state_rates(
            network, modulation, activation, initial_rates=baseline,
            rtol=rtol, atol=atol)
        return readout.score(network, drugged) / readout.score(network, baseline)

    loose = abs(ratio(1e-8, 1e-11) - 1.0)
    tight = abs(ratio(1e-11, 1e-14) - 1.0)
    assert tight < loose
    assert tight < 1e-10


def test_a_time_constants_profile_is_flat_in_every_regime():
    grid, profile, _ = ident.profile_chi_square(
        "tau_BLA", free_drug_gain=False, points=9)
    assert np.max(profile) < 1e-6
    assert ident.profile_interval(grid, profile) == (None, None)


# ---------------------------------------------------------------------------
# 5. The conclusions, and the guard on the production fit.
# ---------------------------------------------------------------------------
def test_no_weight_and_no_tau_comes_out_identifiable(report):
    """P2.3's headline result, computed rather than asserted. If a future
    task adds behavioural data that changes this, this test fails and the
    prose in docs/architecture.md has to be updated with it."""
    by_name = {row["parameter"]: row for row in report}
    assert set(by_name) == {p.name for p in ident.AMYGDALA_PARAMETERS}
    for name in ident.P2_3_TARGET_PARAMETER_NAMES:
        assert by_name[name]["verdict"] != "identifiable", (
            "%s came out identifiable; scenarios.py's illustrative label for "
            "it and docs/architecture.md's P2.3 section both need revisiting"
            % name
        )
    for name in [p.name for p in ident.AMYGDALA_PARAMETERS if p.kind == "tau"]:
        assert by_name[name]["verdict"] == "structurally-unidentifiable"
    assert by_name["gaba_a_connection_modulation_gain"]["verdict"] != \
        "structurally-unidentifiable"


def test_every_verdict_is_one_of_the_declared_ones(report):
    for row in report:
        assert row["verdict"] in ident.VERDICTS


def test_freeing_the_drug_gain_removes_the_apparent_bounds(report):
    """The pedagogical core of the analysis: several parameters look
    identifiable while ten illustrative numbers are held fixed, and stop
    looking that way as soon as one of them is freed. Stated as a count so
    it cannot pass by accident on a single parameter."""
    circuit_rows = [row for row in report if row["kind"] != "drug_gain"]
    bounded_with_gain_fixed = sum(row["bounded_with_gain_fixed"] for row in circuit_rows)
    bounded_with_gain_free = sum(row["bounded_with_gain_free"] for row in circuit_rows)
    assert bounded_with_gain_fixed >= 4
    assert bounded_with_gain_free < bounded_with_gain_fixed


def test_the_three_circuits_without_data_are_reported_as_such():
    rows = ident.circuits_without_behavioural_evidence()
    assert len(rows) == 3
    assert {row["circuit"] for row in rows} == set(ident.CIRCUIT_BUILDERS) - {
        "amygdala_anxiety_circuit (GABA-A)"
    }
    for row in rows:
        assert row["behavioural_observations"] == 0
        assert row["circuit_parameters"] > 0
        assert row["verdict"] == "structurally-unidentifiable"


def test_the_observation_count_is_read_from_the_evidence_module():
    """The 'one observation' the rank-one argument rests on is counted, not
    assumed. A future task adding a second calibration-ready behavioural
    entry has to come back here."""
    counts = ident.behavioural_observation_counts()
    assert sum(counts.values()) == len(behavioral_evidence.calibration_ready_entries())
    assert counts["amygdala_anxiety_circuit (GABA-A)"] == 1


def test_the_evidence_key_matches_the_one_the_production_fit_declares():
    assert set(ident.CIRCUIT_OF_BEHAVIOURAL_EVIDENCE) == set(
        CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS.values()
    )


def test_this_module_adds_no_parameter_to_the_production_fit():
    """P2.3 analyses; it does not calibrate. Nothing here may leak into
    `priors.PRIOR_SPECS`, which is what
    `test_bayesian_calibration.py::test_only_evidence_backed_circuit_layer_
    parameters_are_fit` polices."""
    circuit_names = {p.name for p in ident.AMYGDALA_PARAMETERS}
    leaked = circuit_names.intersection(FREE_PARAM_NAMES)
    assert leaked == {"gaba_a_connection_modulation_gain"}, (
        "P2.3 introduced %s into the production fit; it is an identifiability "
        "analysis and must not" % sorted(leaked - {"gaba_a_connection_modulation_gain"})
    )


def test_no_illustrative_circuit_parameter_changed_value():
    """P2.3 changes no number. The nominal values this analysis reports are
    the ones `scenarios.py` carries, and the amygdala circuit's are exactly
    the v0 hand-set ones."""
    network = scenarios.amygdala_anxiety_circuit_network()
    node = {n.name: n for n in network.nodes}
    assert (node["GABA_Interneuron"].tau, node["GABA_Interneuron"].baseline_input,
            node["GABA_Interneuron"].gain) == (1.0, 0.5, 4.0)
    assert (node["BLA"].tau, node["BLA"].baseline_input, node["BLA"].gain) == (1.5, 2.2, 1.5)
    assert (node["CeA"].tau, node["CeA"].baseline_input, node["CeA"].gain) == (2.0, -1.0, 4.0)
    assert network.connectivity[network.index["BLA"], network.index["GABA_Interneuron"]] == -2.5
    assert network.connectivity[network.index["CeA"], network.index["BLA"]] == 2.0


def test_scan_ranges_are_derived_rather_than_typed_in():
    """Every scan range must trace to either the 0.1x-10x uninformed-prior
    convention or the sigmoid saturation width, and must bracket the
    parameter's own nominal value."""
    for param in ident.AMYGDALA_PARAMETERS:
        low, high = param.scan
        assert low <= param.nominal <= high, param.name
        assert param.scan_basis
        if param.kind in ("tau", "node_gain", "weight"):
            span = max(abs(low), abs(high)) / min(abs(low), abs(high))
            assert span == pytest.approx(
                ident.SCAN_DECADES[1] / ident.SCAN_DECADES[0], rel=1e-9)


def test_the_reduced_box_contains_the_scenarios_circuit():
    box = ident.reduced_parameter_box()
    combos = ident.structural_combinations()
    for key, value in combos.items():
        low, high = box[key]
        assert low <= value <= high, key


# ---------------------------------------------------------------------------
# Posterior-correlation view. Real chains, hence slow.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_a_time_constants_posterior_is_exactly_its_prior():
    """Structural non-identifiability, seen from the sampler: the likelihood
    cannot depend on a parameter the observable does not depend on, so the
    marginal has to be the uniform prior it started from."""
    _, _, statistics = ident.run_joint_diagnostic(
        "tau_BLA", draws=400, tune=400, chains=2)
    assert statistics["posterior_sd_over_prior_sd"] == pytest.approx(1.0, abs=0.12)
    assert abs(statistics["posterior_correlation"]) < 0.2


@pytest.mark.slow
def test_a_connectivity_weight_trades_off_against_the_drug_gain():
    """Practical non-identifiability, seen from the sampler: the joint
    posterior is a ridge, so the two parameters are strongly correlated and
    the weight's own marginal is barely narrower than its prior."""
    _, _, statistics = ident.run_joint_diagnostic(
        "weight_CeA__BLA", draws=400, tune=400, chains=2)
    assert abs(statistics["posterior_correlation"]) > 0.5
    assert statistics["posterior_sd_over_prior_sd"] > 0.5
