"""Tests for the Bayesian parameter-fitting scaffold in
`neurocascade.calibration` (Track C2). Three receptor systems, each its own
INDEPENDENT `pm.Model`/`DEMetropolisZ` chain (see `bayesian_calibration.py`'s
module docstring for why: none of the three systems shares a parameter or a
likelihood term with another, and combining them into one joint model was
tried and produced a measured convergence failure, documented in
docs/architecture.md). Two things are checked per system, mirroring how
`tests/test_scenarios_calibrated.py` etc. sanity-check Track A's work, plus
one cross-cutting synthetic-recovery check:

1. A FAST/cheap smoke test of each system's real model (a few hundred
   DEMetropolisZ draws across 2 chains, not a full production chain): does
   it run end to end without error and produce a structurally valid
   `arviz.InferenceData` posterior (right variable names, right shapes, no
   NaN/inf)? This does NOT assert tight convergence (r_hat/ESS thresholds);
   a run this short is not expected to fully converge, only to execute
   correctly. Real, longer diagnostic runs' r_hat/ESS per system are
   reported separately (see docs/architecture.md and the task report), not
   asserted here as a test, because CI-speed draw counts and genuine MCMC
   convergence are different questions.
2. A synthetic-data self-test: build a full `engine.Simulation` (PK model +
   `ReceptorSite`, the actual simulation engine, not just the equilibrium
   forward-model shortcut used for speed elsewhere in this module) with a
   KNOWN, made-up Kd, generate a synthetic "observed" occupancy from it,
   then run the same calibration machinery (`build_single_kd_model`) with a
   wide, uninformative prior and check the known Kd is recovered.
"""

import inspect

import numpy as np
import pytest

from neurocascade.calibration import bayesian_calibration as bc
from neurocascade.calibration import behavioral_evidence
from neurocascade.calibration import evidence as ev
from neurocascade.calibration.priors import (
    CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS,
    FREE_PARAM_NAMES,
    LIKELIHOOD_UPDATED_PARAM_NAMES,
    PRIOR_ONLY_PARAM_NAMES,
    PRIOR_SPECS,
)
from neurocascade.engine import Simulation
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import Ligand, ReceptorSite

# Every test here runs real DEMetropolisZ chains: 409s for the 7 of them,
# against roughly 20s for the other ~130 tests in the suite combined. That is
# why the whole module is `slow` and sits out the per-task gate. It is NOT
# excused from passing. `autonomous_runner.py --verify-full` runs it daily and
# reports through the digest.
pytestmark = pytest.mark.slow

# Expected shape of each system's "predicted" Deterministic (number of
# equilibrium-occupancy values its forward model returns), used by the
# smoke tests below; must match each `_simulate_*_raw` function's return
# length in bayesian_calibration.py.
_PREDICTED_LENGTH = {"mu_opioid": 1, "d2": 2, "gaba_a": 3}

#: Which entries of each system's `predicted` vector are receptor
#: occupancies (bounded in [0, 1] by construction) and which are something
#: else. GABA-A's third entry is P2.2's behavioural quantity, a drug/vehicle
#: ratio of the anxiety index, which is bounded ABOVE by 1 rather than
#: below: the fitted gain cannot be negative, so diazepam cannot raise the
#: anxiety index, and exactly 1 is reachable at gain 0.
_OCCUPANCY_ENTRIES = {"mu_opioid": (0,), "d2": (0, 1), "gaba_a": (0, 1)}
_RATIO_ENTRIES = {"mu_opioid": (), "d2": (), "gaba_a": (2,)}


def _run_smoke_test_for_system(system_name):
    # A few hundred total draws across 2 chains: cheap enough for a test
    # suite, not a claim of production-quality convergence (see module
    # docstring). `save=False` keeps this test from writing into
    # results/calibration/ (the real artifacts are produced by
    # `bayesian_calibration.main()`, run separately; see
    # docs/architecture.md for the diagnostics that run produced).
    idata, prior_idata, summary = bc.run_calibration(
        system_name, draws=120, tune=120, chains=2, cores=1, progressbar=False, save=False
    )

    param_names = bc.SYSTEM_PARAM_NAMES[system_name]
    for name in param_names:
        assert name in idata.posterior.data_vars
        values = idata.posterior[name].values
        assert values.shape == (2, 120)
        assert np.all(np.isfinite(values))

    assert "predicted" in idata.posterior.data_vars
    predicted = idata.posterior["predicted"].values
    assert predicted.shape == (2, 120, _PREDICTED_LENGTH[system_name])
    assert np.all(np.isfinite(predicted))
    for index in _OCCUPANCY_ENTRIES[system_name]:
        occupancies = predicted[:, :, index]
        assert np.all(occupancies >= 0.0)
        assert np.all(occupancies <= 1.0)
    for index in _RATIO_ENTRIES[system_name]:
        ratios = predicted[:, :, index]
        assert np.all(ratios > 0.0)
        # 1e-9 is integrator noise around the exactly-attainable value 1,
        # not a tolerance on a fitted quantity.
        assert np.all(ratios <= 1.0 + 1e-9)

    assert prior_idata is not None
    assert len(summary) == len(param_names)
    assert "r_hat" in summary.columns
    assert "ess_bulk" in summary.columns


def test_smoke_mu_opioid_calibration_runs_and_produces_valid_posterior():
    _run_smoke_test_for_system("mu_opioid")


def test_smoke_d2_calibration_runs_and_produces_valid_posterior():
    _run_smoke_test_for_system("d2")


def test_smoke_gaba_a_calibration_runs_and_produces_valid_posterior():
    _run_smoke_test_for_system("gaba_a")


def test_system_param_names_partition_all_free_parameters_with_no_overlap():
    # Every one of the eight priors.py parameters belongs to exactly one of
    # the three independent per-system models (bc.SYSTEM_PARAM_NAMES), and
    # no parameter is double-counted across systems, which is exactly the
    # "no shared parameters between systems" premise the independent-model
    # restructuring in bayesian_calibration.py's module docstring relies on.
    all_system_params = [
        name for names in bc.SYSTEM_PARAM_NAMES.values() for name in names
    ]
    assert sorted(all_system_params) == sorted(FREE_PARAM_NAMES)
    assert len(all_system_params) == len(set(all_system_params))


def test_only_evidence_backed_circuit_layer_parameters_are_fit():
    """The successor to P2.1's `test_no_circuit_layer_parameter_is_fit`.

    That test asserted a blanket "no free parameter here is a circuit-layer
    quantity", which was the correct guard while Track E and P2.1 could find
    no behavioural dataset at all. P2.2 found one ([Graziani2022]), so the
    blanket ban would now block a fit that has real data behind it. It is
    replaced, not deleted, and the replacement is STRICTER in the way that
    matters: a circuit-layer parameter may enter this scaffold only if it
    names a behavioural evidence entry, that entry exists, and that entry is
    still calibration-ready when this test runs. An uncalibrated circuit
    parameter is refused exactly as before.
    """
    circuit_layer_markers = (
        "coupling",
        "connection_modulation",
        "tau",
        "gain",
        "baseline_input",
        "weight",
    )
    ready_keys = {
        entry.key for entry in behavioral_evidence.calibration_ready_entries()
    }

    for name in FREE_PARAM_NAMES:
        looks_like_circuit_layer = any(
            marker in name.lower() for marker in circuit_layer_markers
        )
        declared_circuit_layer = name in CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS

        if not declared_circuit_layer:
            # Unchanged from P2.1: anything not declared as an
            # evidence-backed circuit parameter must be a receptor-binding
            # constant, by name and by family.
            assert not looks_like_circuit_layer, (
                f"{name} looks like a circuit-layer parameter but is not "
                "declared in priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS with "
                "the dataset that calibrates it"
            )
            assert name.startswith(("Ki_", "kon_", "koff_")), (
                f"{name} is not one of the receptor-binding parameter "
                "families this scaffold fits (Ki_/kon_/koff_)"
            )
            continue

        # A declared circuit-layer parameter has to earn its place.
        evidence_key = CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS[name]
        assert evidence_key in behavioral_evidence.BEHAVIORAL_EVIDENCE_BY_KEY, (
            f"{name} names {evidence_key!r}, which is not in the behavioural "
            "evidence module"
        )
        assert evidence_key in ready_keys, (
            f"{name} is fitted against {evidence_key!r}, which is no longer "
            "calibration-ready. Either the evidence regressed or the fit "
            "should not be running."
        )
        entry = behavioral_evidence.BEHAVIORAL_EVIDENCE_BY_KEY[evidence_key]
        assert entry.is_bridgeable
        assert entry.effect.has_error_bars


def test_every_declared_circuit_layer_parameter_is_actually_fit():
    """The other direction: a name may not sit in
    CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS unless it is a real free parameter
    that a likelihood term actually updates. Without this, the declaration
    list could be used to launder a parameter that no data touches."""
    for name in CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS:
        assert name in FREE_PARAM_NAMES
        assert name in LIKELIHOOD_UPDATED_PARAM_NAMES
        assert name not in PRIOR_ONLY_PARAM_NAMES


def test_prior_only_parameters_have_no_matching_likelihood_term():
    # Documents/guards the honesty claim made in priors.py and
    # docs/architecture.md: four of the eight free parameters
    # (Ki_naloxone, kon_dopamine, koff_dopamine, Ki_diazepam) have a real
    # reported sd but no likelihood term in evidence.py/the per-system model
    # builders updating them against data in this pass. Checked here by
    # construction (they don't appear anywhere in the relevant system's
    # `_simulate_*_raw` parameter lookups), not by re-deriving a
    # posterior/prior equality numerically (which would need a very long
    # chain to state precisely).
    assert set(LIKELIHOOD_UPDATED_PARAM_NAMES) | set(PRIOR_ONLY_PARAM_NAMES) == set(
        FREE_PARAM_NAMES
    )
    assert set(LIKELIHOOD_UPDATED_PARAM_NAMES) & set(PRIOR_ONLY_PARAM_NAMES) == set()
    # Ki_diazepam_nM left this set in P2.2 and that is a real change, not a
    # bookkeeping one: the [Graziani2022] behavioural term is the first
    # likelihood in this scaffold whose value depends on how much diazepam
    # is bound, so diazepam's affinity is now touched by data. The other
    # three are unchanged, still informed-prior-only.
    assert set(PRIOR_ONLY_PARAM_NAMES) == {
        "Ki_naloxone_nM",
        "kon_dopamine_nM_min",
        "koff_dopamine_min",
    }

    raw_sources = {
        name: inspect.getsource(fn)
        for name, fn in [
            ("mu_opioid", bc._simulate_mu_opioid_raw),
            ("d2", bc._simulate_d2_raw),
            ("gaba_a", bc._simulate_gaba_a_raw),
        ]
    }
    for name in PRIOR_ONLY_PARAM_NAMES:
        for system_name, source in raw_sources.items():
            assert name not in source, (
                f"{name} (informed-prior-only) unexpectedly referenced in "
                f"{system_name}'s forward model"
            )


# ---------------------------------------------------------------------------
# P2.2: the circuit-layer gain. These four are deterministic (a root find on
# the forward model, milliseconds), not MCMC runs: each one asks a question
# about the SHAPE of the problem the sampler is given, and the answers must
# hold whether or not a chain has been run.
# ---------------------------------------------------------------------------

def _gain_reproducing(target_ratio, brain_nM=None, Ki_diazepam=3.6):
    """The gain at which the forward model reproduces `target_ratio`
    exactly. With one free gain and one scalar target this is the maximum
    of the likelihood, and the posterior sits on it up to the prior."""
    from scipy.optimize import brentq

    def residual(gain):
        vector = np.array([1.0, Ki_diazepam, gain])
        return bc._simulate_gaba_a_raw(vector, epm_brain_nM=brain_nM)[2] - target_ratio

    return brentq(residual, 0.0, 5.0, xtol=1e-9)


def test_the_published_effect_lands_well_inside_the_gain_prior():
    """A uniform prior only reports a fit if the data localises the
    posterior strictly inside it. The [Graziani2022] target, and its whole
    +/-3 sd interval, must map to gains clear of both bounds - otherwise the
    posterior summary would be describing the box, not the evidence."""
    lower, upper = PRIOR_SPECS["gaba_a_connection_modulation_gain"][1:]
    target = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_mean"]
    sd = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_sd"]

    # A weaker anxiolysis (larger ratio) maps to a smaller gain, so the
    # +3 sd end of the target is the one that probes the LOWER bound.
    gain_at_weakest = _gain_reproducing(min(target + 3.0 * sd, 0.999999))
    gain_at_strongest = _gain_reproducing(target - 3.0 * sd)
    assert lower < gain_at_weakest < gain_at_strongest < upper

    # Stated the other way round, which is the way that actually shows the
    # posterior is not piled against a bound: what the model predicts AT
    # each bound has to be far from what was observed, in units of the
    # observation's own uncertainty. The lower bound is the mechanistic
    # null - a gain of zero is "diazepam does not potentiate this synapse",
    # which predicts a ratio of exactly 1 - so this doubles as the
    # significance of the fit.
    for bound in (lower, upper):
        predicted = bc._simulate_gaba_a_raw(np.array([1.0, 3.6, bound]))[2]
        assert abs(predicted - target) / sd > 3.0, bound


def test_the_gain_is_identified_by_the_data_not_by_the_prior():
    """The property that makes this a fit rather than a restatement of the
    prior: the target's own uncertainty maps to a gain interval far
    narrower than the prior's range."""
    lower, upper = PRIOR_SPECS["gaba_a_connection_modulation_gain"][1:]
    target = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_mean"]
    sd = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_sd"]
    interval = _gain_reproducing(target - sd) - _gain_reproducing(target + sd)
    assert 0.0 < interval < 0.05 * (upper - lower)


def test_the_fit_barely_moves_over_the_unstated_exposure_timepoint():
    """[Graziani2022] does not state its diazepam pre-test interval, and the
    entry refuses to invent one; `evidence.py` passes 60 min explicitly.
    This is the check that the choice does not carry the result: across the
    whole plausible window the benzodiazepine site stays near saturation, so
    every timepoint implies essentially the same gain."""
    target = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_mean"]
    sd = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_sd"]
    one_sd_in_gain = _gain_reproducing(target - sd) - _gain_reproducing(target + sd)

    gains = [
        _gain_reproducing(target, brain_nM=ev.gaba_a_epm_unbound_brain_nM(t))
        for t in ev.EXPOSURE_TIMEPOINT_SENSITIVITY_MIN
    ]
    assert max(gains) - min(gains) < 0.5 * one_sd_in_gain


def test_the_fit_is_reported_under_a_second_behavioural_index_too():
    """The one assumption in this fit that no citation backs is that the
    model's anxiety index and the behavioural one move proportionally. It
    cannot be tested directly, but its consequence can be bounded: refitting
    against an independently defined EPM anxiety index built from the same
    paper's numbers must not change the gain by more than the target's own
    uncertainty already does. If it did, the reported gain would be an
    artefact of the index rather than of the drug."""
    primary = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_mean"]
    sd = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_sd"]
    composite = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE[
        "anxiety_ratio_composite_index_mean"
    ]
    spread = abs(_gain_reproducing(composite) - _gain_reproducing(primary))
    one_sd_in_gain = _gain_reproducing(primary - sd) - _gain_reproducing(primary + sd)
    assert spread < one_sd_in_gain


def _generate_synthetic_engine_observation(true_kd_nM, concentration_nM, t_span=(0, 240)):
    """Generate a synthetic "observed" equilibrium occupancy from a REAL
    `engine.Simulation` run (PK model + `ReceptorSite`, the actual
    simulation classes this codebase's forward model is supposed to reuse,
    not the lighter equilibrium-only shortcut `forward.py` uses elsewhere
    in this module for MCMC speed). A single ligand, no circuit coupling
    (coupling={}), on a minimal one-node network, run out to a long enough
    time span that its PK has fully delivered the bolus and its receptor
    occupancy has equilibrated, so the final-time occupancy is a valid
    equilibrium observation.
    """
    from neurocascade.circuit import CircuitNode
    from neurocascade.network import CircuitNetwork

    # kon fixed at the same reference value forward.py uses for a
    # Ki/Kd-only ligand; koff = true_kd * kon reproduces the desired Kd.
    kon = 1.0
    koff = true_kd_nM * kon
    ligand = Ligand(name="synthetic_probe", kon=kon, koff=koff, efficacy=1.0)
    receptor_site = ReceptorSite(name="synthetic_site", ligands=[ligand])
    pk_models = [OneCompartmentPK(ke=1e-6)]  # negligible clearance over t_span
    node = CircuitNode(name="Isolated", tau=1.0, baseline_input=0.0, gain=1.0)
    network = CircuitNetwork(nodes=[node], connectivity=[[0.0]])

    sim = Simulation(
        pk_models=pk_models, receptor_site=receptor_site, network=network, coupling={},
    )
    t_eval = np.linspace(t_span[0], t_span[1], 50)
    result = sim.run(
        dose_concentrations=[concentration_nM], t_span=t_span, t_eval=t_eval
    )
    return float(result["occupancy"][0, -1])


def test_synthetic_recovery_of_known_kd_via_engine_simulation():
    # Generate synthetic data from a REAL engine.Simulation run at a KNOWN
    # Kd deliberately different from any of this module's real literature
    # values (2.5 nM; none of the eight priors.py point estimates is 2.5),
    # so recovering it is a genuine check, not a coincidence of matching
    # some other constant already in the code.
    true_kd = 2.5  # nM
    concentration = 20.0  # nM; well above true_kd so the equilibrium
    # occupancy is in a sensitive (non-saturating, non-floor) part of the
    # C/(C+Kd) curve, giving the likelihood real leverage on Kd.
    synthetic_observed_occupancy = _generate_synthetic_engine_observation(
        true_kd, concentration
    )
    # Cross-check the synthetic data itself against the closed-form
    # equilibrium formula this codebase's occupancy ODE is known to satisfy
    # (see e.g. test_calibrated_kon_koff_reproduce_ki_at_equilibrium in
    # tests/test_scenarios_calibrated.py): occ = C / (C + Kd).
    expected_occupancy = concentration / (concentration + true_kd)
    assert abs(synthetic_observed_occupancy - expected_occupancy) < 1e-3

    model = bc.build_single_kd_model(
        prior_lower=0.1,
        prior_upper=25.0,
        concentration=concentration,
        observed_occupancy=synthetic_observed_occupancy,
        sd=0.01,  # tight: this "observation" is a clean synthetic value,
        # not a real literature figure with known systematic model-mismatch
        # (contrast evidence.py's OCCUPANCY_LIKELIHOOD_SD=0.07).
    )
    import pymc as pm

    with model:
        step = pm.DEMetropolisZ()
        idata = pm.sample(
            draws=300, tune=300, chains=2, cores=1, step=step,
            random_seed=20260726, progressbar=False,
        )

    posterior_kd = idata.posterior["kd"].values
    assert np.all(np.isfinite(posterior_kd))
    recovered_mean = float(posterior_kd.mean())
    recovered_sd = float(posterior_kd.std())

    # Recovery tolerance: within 25% of the true value, a loose bound
    # appropriate for a cheap, short DEMetropolisZ chain (this is a
    # sanity check that the machinery finds the right neighborhood, not a
    # precision benchmark).
    assert abs(recovered_mean - true_kd) < 0.25 * true_kd
    # The true value should also fall within a generous multiple of the
    # posterior's own spread, guarding against a recovered mean that's
    # "close" only by chance while the posterior is wildly overconfident
    # or underdispersed.
    assert abs(recovered_mean - true_kd) < 4 * recovered_sd
