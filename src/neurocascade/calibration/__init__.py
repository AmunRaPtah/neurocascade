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
  - behavioral_evidence.py: (P2.1, extended in P2.2) the DOSE-to-BEHAVIOUR
                 datasets, as distinct from evidence.py's
                 concentration-to-occupancy ones. Holds the three papers
                 Track E named, re-examined through the Phase 1
                 dose-to-brain-concentration bridge, plus two further
                 rat/IP diazepam datasets - one that exercises the bridge
                 end to end and one, added in P2.2, that is an actual
                 likelihood target. The module records per entry exactly
                 why each of the others is not.
  - forward.py:  the RECEPTOR-layer forward model, built on
                 `ReceptorSite.occupancy_rhs` (integrated via
                 `scipy.integrate.solve_ivp`, the same integrator
                 `engine.py` uses elsewhere) rather than a hand-derived
                 closed-form occupancy formula.
  - circuit_forward.py: (P2.2) the CIRCUIT-layer forward model, built the
                 same way on `CircuitNetwork.rhs`. Returns a drug/vehicle
                 RATIO of a behavioural readout, because the readout has no
                 published scale and a ratio needs none.
  - bayesian_calibration.py: builds the combined `pm.Model` (8 parameters, 5
                 likelihood terms), samples with `DEMetropolisZ`, and saves
                 posterior + prior-predictive `InferenceData` plus a summary
                 table, mirroring
                 `cct-bayesian-validation/bayesian_calibration.py`'s
                 `build_model()`/`main()` structure.

What this scaffold fits at the CIRCUIT layer, and what it still does not
(see `docs/architecture.md`'s P2.2 section for the full reasoning).

Through Track C2 and P2.1 the answer was "nothing": every validation target
in `evidence.py` was a receptor-occupancy measurement (Moss/Kapur/Savic),
and Track E had searched per system for a real behavioural anchor for
`mu_opioid_coupling()`/`d2_coupling()`/`gaba_a_connection_modulation()`'s
gains and found none usable without either substituting a different drug's
measured effect or inventing a dose-to-concentration PK submodel. A
circuit-layer likelihood term would have had nothing real to compare
against.

P2.2 changed that for exactly ONE of the three. `gaba_a_connection_
modulation`'s gain is now fitted against [Graziani2022], a rat/IP diazepam
elevated-plus-maze dataset with a vehicle group, printed SEMs and group
sizes, reached through the Phase 1 bridge. `mu_opioid_coupling` and
`d2_coupling` remain unfitted and their scenario functions remain labelled
illustrative: P2.2 re-ran both searches at full scale (421 open-access
fentanyl/accumbens articles, 1303 haloperidol/behaviour articles) and both
came back empty, for reasons recorded per system in
`behavioral_evidence.SEARCHED_AND_EMPTY_BEHAVIORAL`.

`priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` is the list of circuit-layer
parameters this scaffold fits and the dataset behind each, and
`tests/test_bayesian_calibration.py::test_only_evidence_backed_circuit_
layer_parameters_are_fit` enforces it: a circuit-layer parameter may be here
only if it names a behavioural entry that is still calibration-ready when
the test runs, and anything not so declared must still be a `Ki_`/`kon_`/
`koff_` receptor-binding constant. That test replaced P2.1's blanket
`test_no_circuit_layer_parameter_is_fit`.
"""
