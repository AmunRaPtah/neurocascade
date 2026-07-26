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

from neurocascade.calibration import bayesian_calibration as bc
from neurocascade.calibration.priors import (
    FREE_PARAM_NAMES,
    LIKELIHOOD_UPDATED_PARAM_NAMES,
    PRIOR_ONLY_PARAM_NAMES,
)
from neurocascade.engine import Simulation
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import Ligand, ReceptorSite

# Expected shape of each system's "predicted" Deterministic (number of
# equilibrium-occupancy values its forward model returns), used by the
# smoke tests below; must match each `_simulate_*_raw` function's return
# length in bayesian_calibration.py.
_PREDICTED_LENGTH = {"mu_opioid": 1, "d2": 2, "gaba_a": 2}


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
    assert np.all(predicted >= 0.0)
    assert np.all(predicted <= 1.0)

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
    assert set(PRIOR_ONLY_PARAM_NAMES) == {
        "Ki_naloxone_nM",
        "kon_dopamine_nM_min",
        "koff_dopamine_min",
        "Ki_diazepam_nM",
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
