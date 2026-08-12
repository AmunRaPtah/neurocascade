"""Bayesian parameter-fitting scaffold for neurocascade's three
literature-calibrated receptor systems (mu-opioid, D2, GABA-A benzodiazepine
site; see `scenarios.py`'s "Calibrated ..." sections and
`docs/architecture.md`'s "Bayesian parameter fitting" section).

Mirrors the pattern used by the sibling project
`/root/projects/cct-bayesian-validation`, which solved the equivalent
Bayesian-fitting problem for a related pharmacology ODE model: PyMC with a
gradient-free step method (`DEMetropolisZ`, since the ODE forward model
isn't differentiable through PyTensor, so it's wrapped with
`pytensor.compile.ops.as_op`), a two-tier informed/uninformed prior scheme,
a Gaussian likelihood built from real literature validation targets, and
results saved as `arviz` `InferenceData` plus a posterior summary table.

Module layout (one shared module per CCT-sibling responsibility, adapted to
neurocascade's simpler structure): unlike the CCT model, where all axes
share one coupled ODE state vector, each of neurocascade's three receptor
systems' own forward model is *fully independent* of the other two (mu-
opioid, D2, and GABA-A each have their own `ReceptorSite`/ligands/PK with no
shared state), so there is no mechanistic need for the CCT-style single
14-parameter coupled forward model. What is shared, and worth combining into
one calibration run per the project brief, is the *sampling infrastructure*
(one `pm.Model`, one `DEMetropolisZ` sampler, one posterior artifact)
covering all three systems' parameters and likelihood terms at once, purely
for calibration-workflow convenience. The three systems' posteriors would
come out identical if fit in three separate `pm.sample()` calls, since none
of the three likelihood terms below reads another system's parameters.

  - priors.py:   two-tier (informed/uninformed) `PRIOR_SPECS` for the 8 free
                 receptor-binding parameters this pass fits, one shared
                 dict across all three systems.
  - evidence.py: the real literature validation targets already present in
                 `scenarios.py`'s calibrated sections (Moss et al. 2020 for
                 mu-opioid, Kapur et al. 1997 for D2, Savic et al. 1991 for
                 GABA-A), turned into Gaussian-likelihood (mean, sd) targets.
  - behavioral_evidence.py: (P2.1) the DOSE-to-BEHAVIOUR datasets, as
                 distinct from evidence.py's concentration-to-occupancy ones.
                 Holds the three papers Track E named, re-examined through
                 the Phase 1 dose-to-brain-concentration bridge, plus one
                 additional rat/IP diazepam dataset that exercises the bridge
                 end to end. Nothing in it is a likelihood target yet, and
                 the module records per entry exactly why not.
  - forward.py:  the forward model, built on `ReceptorSite.occupancy_rhs`
                 (integrated via `scipy.integrate.solve_ivp`, the same
                 integrator `engine.py` uses elsewhere) rather than a
                 hand-derived closed-form occupancy formula.
  - bayesian_calibration.py: builds the combined `pm.Model` (8 parameters, 5
                 likelihood terms), samples with `DEMetropolisZ`, and saves
                 posterior + prior-predictive `InferenceData` plus a summary
                 table, mirroring
                 `cct-bayesian-validation/bayesian_calibration.py`'s
                 `build_model()`/`main()` structure.

What this scaffold deliberately does NOT fit (see `docs/architecture.md` and
the module docstrings below for the full reasoning): circuit-layer
weights/gains/taus in any scenario. Every calibrated section in
`scenarios.py` explicitly documents that no literature target exists for
those parameters, and `evidence.py`'s validation targets are all
receptor-occupancy measurements (Moss/Kapur/Savic), not behavioral-readout
data, so a circuit-layer likelihood term would have nothing real to compare
against. Forcing a fit there would be exactly the kind of invented number
this codebase's documentation repeatedly warns against.

Track E (`docs/architecture.md`, "Track E: circuit-layer calibration
attempt") searched specifically for real behavioral anchors for
`mu_opioid_coupling()`/`d2_coupling()`/`gaba_a_connection_modulation()`'s
gains and confirmed this gap by name, per system, rather than leaving it a
general statement: real papers on the right drug/mechanism were found for
all three, but none supplied a number usable without either substituting a
different drug's measured effect or inventing a dose-to-concentration PK
submodel this codebase has for no drug. `FREE_PARAM_NAMES` in `priors.py`
still contains only the eight receptor-binding parameters from Track C2;
`tests/test_bayesian_calibration.py::test_no_circuit_layer_parameter_is_fit`
guards that by construction.
"""
