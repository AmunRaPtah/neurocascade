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

  - priors.py:   two-tier (informed/uninformed) `PRIOR_SPECS` for the seven
                 free receptor-binding parameters this pass fits, one shared
                 dict across all three systems.
  - evidence.py: the real literature validation targets already present in
                 `scenarios.py`'s calibrated sections (Moss et al. 2020 for
                 mu-opioid, Kapur et al. 1997 for D2, Savic et al. 1991 for
                 GABA-A), turned into Gaussian-likelihood (mean, sd) targets.
                 All three are HUMAN measurements; since P11.3 they are also
                 the only behavioural-adjacent targets this package holds.
  - forward.py:  the RECEPTOR-layer forward model, built on
                 `ReceptorSite.occupancy_rhs` (integrated via
                 `scipy.integrate.solve_ivp`, the same integrator
                 `engine.py` uses elsewhere) rather than a hand-derived
                 closed-form occupancy formula.
  - circuit_forward.py: (P2.2) the CIRCUIT-layer forward model, built the
                 same way on `CircuitNetwork.rhs`. Returns a drug/vehicle
                 RATIO of a behavioural readout, because the readout has no
                 published scale and a ratio needs none.
  - bayesian_calibration.py: builds one independent `pm.Model` per system,
                 samples each with `DEMetropolisZ`, and saves posterior +
                 prior-predictive `InferenceData` plus a summary table,
                 mirroring
                 `cct-bayesian-validation/bayesian_calibration.py`'s
                 `build_model()`/`main()` structure.
  - identifiability.py / endpoint_calibration.py / heldout_forward.py:
                 the P2.3/P7.3/P8 machinery, all species-agnostic and all
                 kept. Since P11.3 they run against explicitly-labelled
                 SYNTHETIC observations (`identifiability.py`'s scalar,
                 `endpoint_calibration.synthetic_target`) because the rodent
                 datasets that fed them moved out with the species pivot.
  - behavioral_evidence.py: RETIRED by P11.3 to
                 `neurocascade/legacy/behavioral_evidence.py`, where the full
                 dose-to-BEHAVIOUR dataset set it held (eleven entries, ten
                 rat and one mouse, no human) stays importable, tested and
                 reproducible under the quarantine policy. No human dataset
                 has replaced it yet; `SEARCHED_AND_EMPTY_BEHAVIORAL` there
                 records what the searches found and why each entry fell
                 short of its own gate.

What this scaffold fits at the CIRCUIT layer: nothing, again
(see `docs/architecture.md`'s P2.2 and P11.3 sections for the reasoning).

Through Track C2 and P2.1 the answer was "nothing": every validation target
in `evidence.py` was a receptor-occupancy measurement (Moss/Kapur/Savic),
and Track E had searched per system for a real behavioural anchor for
`mu_opioid_coupling()`/`d2_coupling()`/`gaba_a_connection_modulation()`'s
gains and found none usable without either substituting a different drug's
measured effect or inventing a dose-to-concentration PK submodel. A
circuit-layer likelihood term would have had nothing real to compare
against.

Between P2.2 and P11.3 the answer was "one", and only ever one:
`gaba_a_connection_modulation`'s gain, fitted against [Graziani2022], a
rat/IP diazepam elevated-plus-maze dataset reached through the Phase 1
bridge. The v2 human-primary roadmap retired that fit - the observation is
rat behaviour, and under the species pivot no rodent dataset may calibrate
any layer of a human-primary model - so the answer is "nothing" again, now
with the blocker named precisely: HUMAN behavioural evidence through the
bridge, which no search has yet produced. The fitting machinery is kept
warm on synthetic observations (`bayesian_calibration.build_circuit_gain_
model`, `identifiability.py`'s labelled scalar) so that the day such a
dataset passes the gate, nothing needs rebuilding.

`priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` is the register of circuit-layer
parameters this scaffold calibrates and the dataset behind each - EMPTY
since P11.3, and enforced as empty by
`tests/test_bayesian_calibration.py::test_no_circuit_layer_parameter_
claims_to_be_calibrated`. Anything entering it must name HUMAN behavioural
evidence in the same change that adds it; anything not registered must be
a `Ki_`/`kon_`/`koff_` receptor-binding constant. This is P2.1's original
blanket rule restored, with P11.3's reason attached.
"""
