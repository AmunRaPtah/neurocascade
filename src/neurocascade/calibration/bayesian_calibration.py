"""Bayesian calibration of neurocascade's free parameters (see `priors.py`)
against real published validation targets.

Eight of the nine are receptor-binding constants fitted against the three
occupancy targets already cited in `scenarios.py`'s calibrated sections (see
`evidence.py`): Moss et al. 2020 for mu-opioid/fentanyl, Kapur et al. 1997
for D2/haloperidol, Savic et al. 1991 for GABA-A/flumazenil.

The ninth, added in P2.2, is the first CIRCUIT-layer parameter this scaffold
has ever fitted: `gaba_a_connection_modulation_gain`, against
[Graziani2022]'s rat elevated-plus-maze data, reached through the Phase 1
dose-to-brain-concentration bridge and compared with the model as a
drug/vehicle RATIO (the behavioural readout has no published scale; a ratio
needs none). See `circuit_forward.py` and docs/architecture.md's P2.2
section, which also records why the other two coupling gains P2.2 was asked
to fit are still unfitted.

Mirrors `/root/projects/cct-bayesian-validation/bayesian_calibration.py`:
the forward model is a black-box ODE solve (`ReceptorSite.occupancy_rhs` via
`scipy.integrate.solve_ivp`, see `forward.py`), not differentiable through
PyTensor, so it's wrapped with `pytensor.compile.ops.as_op` and sampled with
a gradient-free step method (`DEMetropolisZ`) rather than NUTS.

THREE INDEPENDENT `pm.Model`s, one per receptor system (mu-opioid, D2,
GABA-A), each sampled with its own `DEMetropolisZ` chain over only that
system's own parameters, rather than one combined 8-parameter model. An
earlier version of this module DID combine all eight parameters into one
`pm.Model`/one shared `DEMetropolisZ` step (matching how
`cct-bayesian-validation` samples its own 14 mutually-DEPENDENT parameters,
all of which feed one coupled ODE state vector). That doesn't apply here:
none of neurocascade's three receptor systems shares a parameter or a
likelihood term with another (`priors.py`'s per-parameter source citations
and `evidence.py`'s three validation targets are each scoped to exactly one
system), so there is no statistical reason to sample them jointly. Doing so
anyway produced a real, measured convergence failure (max r_hat 1.92, min
bulk ESS 5 on the combined model): `DEMetropolisZ` draws every proposal from
ONE shared history across the full parameter vector, so a hard-to-satisfy
move in one system's block (see the D2 section below) got accepted or
rejected as part of a joint move, dragging down the effective sample size of
every other, otherwise easy, parameter too, even though nothing in their
likelihoods depends on that system. Splitting into three independent models,
each with its own chain, removes that coupling entirely; see
docs/architecture.md's "Bayesian parameter fitting" section for the full
diagnostic history, including why a `pm.CompoundStep` with per-block
`vars=` (the alternative that would still be "one model") was tried first
and found blocked by a version-specific PyMC 6.0.1 limitation.
"""

import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from .. import scenarios
from . import circuit_forward as cfwd
from . import evidence as ev
from . import forward as fwd
from .priors import PRIOR_SPECS

SEED = 20260726

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "calibration")

# Which of the eight priors.py parameters belong to each independent model,
# in the order each system's own `_simulate_*_raw` function below expects.
SYSTEM_PARAM_NAMES = {
    "mu_opioid": ["Ki_fentanyl_nM", "Ki_naloxone_nM"],
    "d2": ["kon_haloperidol_nM_min", "koff_haloperidol_min",
           "kon_dopamine_nM_min", "koff_dopamine_min"],
    "gaba_a": ["Ki_flumazenil_nM", "Ki_diazepam_nM",
               "gaba_a_connection_modulation_gain"],
}


def _make_rv(name):
    """Build one free RV from its `priors.py` spec (shared by all three
    per-system model builders below)."""
    spec = PRIOR_SPECS[name]
    if spec[0] == "uniform":
        _, lo, hi = spec
        return pm.Uniform(name, lower=lo, upper=hi)
    _, mean, sd, lo, hi = spec
    return pm.TruncatedNormal(name, mu=mean, sigma=sd, lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# Mu-opioid: 2 parameters (Ki_fentanyl_nM, Ki_naloxone_nM), 1 likelihood term
# (Moss et al. 2020, on fentanyl only; naloxone is informed-prior-only, see
# priors.py).
# ---------------------------------------------------------------------------
def _simulate_mu_opioid_raw(vec):
    Ki_fentanyl = vec[0]
    occ = fwd.equilibrium_occupancy_from_kd(
        Ki_fentanyl, ev.MU_OPIOID_FENTANYL_OCCUPANCY_EVIDENCE["concentration_nM"], efficacy=1.0
    )
    return np.array([occ])


@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def _simulate_mu_opioid(vec):
    return _simulate_mu_opioid_raw(vec)


def build_mu_opioid_model():
    with pm.Model() as model:
        Ki_fentanyl = _make_rv("Ki_fentanyl_nM")
        _Ki_naloxone = _make_rv("Ki_naloxone_nM")  # informed prior only, no likelihood term
        vec = pt.stack([Ki_fentanyl, _Ki_naloxone])
        predicted = pm.Deterministic("predicted", _simulate_mu_opioid(vec))
        pm.Normal(
            "fentanyl_occupancy_obs",
            mu=predicted[0],
            sigma=ev.MU_OPIOID_FENTANYL_OCCUPANCY_EVIDENCE["occupancy_target_sd"],
            observed=ev.MU_OPIOID_FENTANYL_OCCUPANCY_EVIDENCE["occupancy_target_mean"],
        )
    return model


# ---------------------------------------------------------------------------
# D2: 4 parameters (kon/koff for haloperidol AND dopamine), 2 likelihood
# terms (Kapur et al. 1997, on haloperidol only; dopamine is
# informed-prior-only, see priors.py).
# ---------------------------------------------------------------------------
def _simulate_d2_raw(vec):
    kon_halo, koff_halo = vec[0], vec[1]
    ed50 = fwd.equilibrium_occupancy(
        kon_halo, koff_halo, ev.D2_HALOPERIDOL_OCCUPANCY_EVIDENCE["ed50_concentration_nM"],
        efficacy=0.0,
    )
    ed80 = fwd.equilibrium_occupancy(
        kon_halo, koff_halo, ev.D2_HALOPERIDOL_OCCUPANCY_EVIDENCE["ed80_concentration_nM"],
        efficacy=0.0,
    )
    return np.array([ed50, ed80])


@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def _simulate_d2(vec):
    return _simulate_d2_raw(vec)


def build_d2_model():
    with pm.Model() as model:
        kon_halo = _make_rv("kon_haloperidol_nM_min")
        koff_halo = _make_rv("koff_haloperidol_min")
        _kon_dop = _make_rv("kon_dopamine_nM_min")  # informed prior only
        _koff_dop = _make_rv("koff_dopamine_min")  # informed prior only
        vec = pt.stack([kon_halo, koff_halo, _kon_dop, _koff_dop])
        predicted = pm.Deterministic("predicted", _simulate_d2(vec))
        pm.Normal(
            "haloperidol_ed50_occupancy_obs",
            mu=predicted[0],
            sigma=ev.D2_HALOPERIDOL_OCCUPANCY_EVIDENCE["occupancy_target_sd"],
            observed=ev.D2_HALOPERIDOL_OCCUPANCY_EVIDENCE["ed50_occupancy_target_mean"],
        )
        pm.Normal(
            "haloperidol_ed80_occupancy_obs",
            mu=predicted[1],
            sigma=ev.D2_HALOPERIDOL_OCCUPANCY_EVIDENCE["occupancy_target_sd"],
            observed=ev.D2_HALOPERIDOL_OCCUPANCY_EVIDENCE["ed80_occupancy_target_mean"],
        )
    return model


# ---------------------------------------------------------------------------
# GABA-A: 3 parameters (Ki_flumazenil_nM, Ki_diazepam_nM, and, new in P2.2,
# the CIRCUIT-layer gaba_a_connection_modulation_gain), 3 likelihood terms
# (Savic et al. 1991 on flumazenil occupancy, twice; [Graziani2022]'s
# elevated-plus-maze anxiolysis ratio once).
#
# This is the only one of the three systems that gained a circuit-layer
# parameter in P2.2, and the shortfall is data, not machinery:
# `behavioral_evidence.SEARCHED_AND_EMPTY_BEHAVIORAL` records the sweeps
# re-run in this pass for the other two (421 open-access fentanyl/accumbens
# articles, 1303 haloperidol/behaviour articles), both still empty, and
# docs/architecture.md's P2.2 section states what would close each.
#
# The behavioural term reads BOTH the gain and Ki_diazepam. The diazepam
# exposure [Graziani2022]'s dose produces is fixed - it comes from the Phase
# 1 bridge, not from any sampled parameter - but how much of the site that
# exposure occupies depends on the sampled Ki, and the circuit only ever
# sees occupancy. That is why Ki_diazepam leaves priors.py's
# PRIOR_ONLY_PARAM_NAMES in this pass, and why this term is computed inside
# one `as_op` rather than split across two.
# ---------------------------------------------------------------------------
#: The unbound BRAIN concentration [Graziani2022]'s 2 mg/kg IP dose reaches
#: at the exposure timepoint, computed once at import through the Phase 1
#: bridge (`evidence.gaba_a_epm_unbound_brain_nM`, which integrates rather
#: than solving in closed form). It does not depend on any sampled
#: parameter, so recomputing it per draw would only cost time.
_GABA_A_EPM_BRAIN_NM = ev.gaba_a_epm_unbound_brain_nM()

#: The amygdala circuit's drug-free steady state. Independent of every
#: sampled parameter (at zero activation the modulation factor is exactly 1
#: whatever the gain), so it is integrated once here rather than once per
#: MCMC step; `circuit_forward.readout_ratio` accepts it as an argument
#: instead of caching it invisibly.
_GABA_A_BASELINE_RATES = cfwd.steady_state_rates(
    scenarios.amygdala_anxiety_circuit_network(),
    scenarios.gaba_a_connection_modulation(),
    activation=0.0,
    coupling=scenarios.gaba_a_coupling(),
)


def _simulate_gaba_a_raw(vec, epm_brain_nM=None):
    Ki_flumazenil, Ki_diazepam, modulation_gain = vec[0], vec[1], vec[2]
    low = fwd.equilibrium_occupancy_from_kd(
        Ki_flumazenil, ev.GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE["low_dose_concentration_nM"],
        efficacy=0.0,
    )
    high = fwd.equilibrium_occupancy_from_kd(
        Ki_flumazenil, ev.GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE["high_dose_concentration_nM"],
        efficacy=0.0,
    )
    # Diazepam's activation of the benzodiazepine site at that exposure.
    # Diazepam's efficacy at this site is 1.0 in scenarios.py (see that
    # module on why efficacy carries potentiation strength for a
    # connection_modulation-coupled site), so activation is occupancy.
    brain_nM = _GABA_A_EPM_BRAIN_NM if epm_brain_nM is None else epm_brain_nM
    activation = fwd.equilibrium_occupancy_from_kd(Ki_diazepam, brain_nM, efficacy=1.0)
    anxiety_ratio = cfwd.readout_ratio(
        scenarios.amygdala_anxiety_circuit_network(),
        scenarios.anxiety_index_readout(),
        connection_modulation={
            connection: modulation_gain
            for connection in scenarios.gaba_a_connection_modulation()
        },
        activation=activation,
        coupling=scenarios.gaba_a_coupling(),
        baseline_rates=_GABA_A_BASELINE_RATES,
    )
    return np.array([low, high, anxiety_ratio])


@as_op(itypes=[pt.dvector], otypes=[pt.dvector])
def _simulate_gaba_a(vec):
    return _simulate_gaba_a_raw(vec)


def build_gaba_a_model():
    with pm.Model() as model:
        Ki_flumazenil = _make_rv("Ki_flumazenil_nM")
        Ki_diazepam = _make_rv("Ki_diazepam_nM")
        modulation_gain = _make_rv("gaba_a_connection_modulation_gain")
        vec = pt.stack([Ki_flumazenil, Ki_diazepam, modulation_gain])
        predicted = pm.Deterministic("predicted", _simulate_gaba_a(vec))
        pm.Normal(
            "flumazenil_low_dose_occupancy_obs",
            mu=predicted[0],
            sigma=ev.GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE["occupancy_target_sd"],
            observed=ev.GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE["low_dose_occupancy_target_mean"],
        )
        pm.Normal(
            "flumazenil_high_dose_occupancy_obs",
            mu=predicted[1],
            sigma=ev.GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE["occupancy_target_sd"],
            observed=ev.GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE["high_dose_occupancy_target_mean"],
        )
        pm.Normal(
            "diazepam_epm_anxiolysis_ratio_obs",
            mu=predicted[2],
            sigma=ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_sd"],
            observed=ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_mean"],
        )
    return model


BUILDERS = {
    "mu_opioid": build_mu_opioid_model,
    "d2": build_d2_model,
    "gaba_a": build_gaba_a_model,
}


def build_model(system_name):
    """Build the independent `pm.Model` for one receptor system.
    `system_name` is one of `BUILDERS`' keys ("mu_opioid", "d2", "gaba_a")."""
    if system_name not in BUILDERS:
        raise ValueError(
            "unknown system_name %r, expected one of %s" % (system_name, sorted(BUILDERS))
        )
    return BUILDERS[system_name]()


def run_calibration(system_name, draws=1000, tune=1000, chains=4, cores=1, progressbar=False,
                     save=True, seed=SEED):
    """Build and sample ONE receptor system's independent model.
    `draws`/`tune`/`chains` are exposed as arguments so tests can run a
    cheap smoke-test configuration (a few hundred draws, 1-2 chains) while
    `main()` below uses a fuller production-scale configuration by default.
    """
    model = build_model(system_name)
    with model:
        prior_idata = pm.sample_prior_predictive(draws=min(draws, 500), random_seed=seed)
        step = pm.DEMetropolisZ()
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores, step=step,
            random_seed=seed, progressbar=progressbar,
        )

    summary = az.summary(idata, var_names=SYSTEM_PARAM_NAMES[system_name])

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary.to_csv(os.path.join(RESULTS_DIR, f"{system_name}_summary.csv"))
        idata.to_netcdf(os.path.join(RESULTS_DIR, f"{system_name}_posterior.nc"))
        prior_idata.to_netcdf(os.path.join(RESULTS_DIR, f"{system_name}_prior.nc"))

    return idata, prior_idata, summary


def run_all_calibrations(draws=1000, tune=1000, chains=4, cores=1, progressbar=False,
                          save=True, seed=SEED):
    """Run all three independent per-system calibrations and return
    {system_name: (idata, prior_idata, summary)}."""
    return {
        system_name: run_calibration(
            system_name, draws=draws, tune=tune, chains=chains, cores=cores,
            progressbar=progressbar, save=save, seed=seed,
        )
        for system_name in BUILDERS
    }


def build_single_kd_model(prior_lower, prior_upper, concentration, observed_occupancy, sd,
                           efficacy=1.0):
    """Minimal one-parameter `pm.Model` fitting a single Kd against one
    equilibrium-occupancy observation at a fixed concentration, using the
    same `ReceptorSite`-based forward model as the per-system builders
    above, just scaled down to one parameter and one likelihood term. Used
    by the synthetic-data self-test in `tests/test_bayesian_calibration.py`
    to check that this scaffold can actually recover a known parameter from
    data generated by a real `engine.Simulation` run, at a scale small
    enough to sample quickly; also reusable directly for any future
    single-Kd calibration need this codebase doesn't have yet.
    """

    # Wrapped as a length-1 dvector, not a dscalar: the PyTensor/Numba
    # as_op backend used here misbehaves on scalar-typed as_op outputs
    # (observed while building this test: a scalar `as_op` raised
    # "can't unbox array from PyObject into native value" inside PyMC's
    # test-point evaluation), while the vector form (used throughout this
    # module) round-trips cleanly.
    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def simulate_occupancy(kd_vec):
        return np.array(
            [fwd.equilibrium_occupancy_from_kd(float(kd_vec[0]), concentration, efficacy=efficacy)]
        )

    with pm.Model() as model:
        kd = pm.Uniform("kd", lower=prior_lower, upper=prior_upper)
        predicted = pm.Deterministic(
            "predicted_occupancy", simulate_occupancy(pt.stack([kd]))[0]
        )
        pm.Normal("occupancy_obs", mu=predicted, sigma=sd, observed=observed_occupancy)
    return model


def main():
    results = run_all_calibrations(draws=1000, tune=1000, chains=4, cores=1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    diagnostics_lines = []
    for system_name, (idata, prior_idata, summary) in results.items():
        print(f"=== {system_name} ===")
        print(summary.to_string())

        r_hat_max = float(pd.to_numeric(summary["r_hat"], errors="coerce").max())
        ess_min = float(pd.to_numeric(summary["ess_bulk"], errors="coerce").min())
        print(f"Max r_hat: {r_hat_max:.3f}  (want < 1.05 for good convergence)")
        print(f"Min ESS (bulk): {ess_min:.0f}\n")
        diagnostics_lines.append(f"{system_name}: max_r_hat={r_hat_max} min_ess_bulk={ess_min}")

    with open(os.path.join(RESULTS_DIR, "calibration_diagnostics.txt"), "w") as f:
        f.write("\n".join(diagnostics_lines) + "\n")

    print(f"Saved per-system posterior/prior/summary files under {RESULTS_DIR}/")
    return results


if __name__ == "__main__":
    main()
