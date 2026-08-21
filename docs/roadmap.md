# Roadmap: receptor → circuit → behavior, at whole-brain scale

> **SUPERSEDED as to species, 2026-08-21.** Phases 1-10 below are complete and
> are retained as the record of the mouse build. The species decision stated in
> "What 'the entire brain' means here" was reversed: neurocascade is
> human-primary. See `roadmap-human.md` for the v2 plan (Phases 11-18), the
> five-layer audit behind the reversal, and the quarantine policy for rodent
> material. Everything in this document about engine structure, the ordering
> constraint and the honesty discipline still holds.

## What "the entire brain" means here, concretely

The v0 to v1 model is three hand-built circuits (4-11 nodes each) carrying three
receptor systems, with binding/PK constants literature-calibrated and every
circuit-layer parameter explicitly labeled illustrative. The target state is:

- **Receptor layer:** every modeled receptor's density defined for *every* brain
  region from one consistent atlas source, not a hand-written map per scenario.
- **Circuit layer:** one whole-brain directed, weighted network from a real
  connectome, not a per-scenario hand-drawn graph.
- **Behavior layer:** named behavioral endpoints fit against real published
  dose-response data, not illustrative linear readouts.
- **The end product:** `predict(binding_fingerprint) → behavioral profile`. Give
  the model a compound's receptor affinity profile and a dose, and it returns a
  predicted behavioral signature, validated against held-out drugs never used in
  fitting.

That is a defensible whole-brain, receptor-to-behavior model. It is *not* a
cellular-resolution or spiking simulation, and it is not a claim to simulate
cognition. Population-rate (Wilson-Cowan) resolution over an atlas parcellation
is the level where receptor pharmacology and behavioral endpoints actually meet
the available data, and the level this codebase already works at.

**Species: mouse.** (SUPERSEDED, see the note at the top of this file. The
reasoning below optimized for the circuit layer and paid for it in the other
four; `roadmap-human.md` records the layer-by-layer audit.) The Allen Mouse
Brain Connectivity Atlas is the only
real, quantitative, *directed* whole-brain connectome (tract-tracing, not
diffusion inference), and the Allen Mouse Brain Atlas provides matched
region-wise receptor-gene expression from the same ontology. The behavioral
pharmacology this model must fit against (locomotor activity, elevated plus
maze, tail-flick, CPP) is rodent as well. Human is a later, separate track;
the human connectome is undirected and diffusion-derived, which would weaken
the circuit layer, not strengthen it. That track has now begun on the
`human-connectome-layer` branch (`src/neurocascade/human/`): the ENIGMA
Toolbox Desikan-Killiany structural matrix is ingested and signed per source
region from cited transmitter identity (`human/neurotransmitter.py`), with
the undirected/unsigned caveat stated there rather than hidden.

## The ordering constraint that drives everything

Track E established that circuit-layer and behavioral calibration is blocked,
and diagnosed exactly why: real behavioral pharmacology reports **administered
dose (mg/kg)**, while this codebase's state variable is **plasma
concentration**, because `OneCompartmentPK` starts from an already-achieved
bolus concentration rather than an administered dose. Every quantitative
behavioral number found was dose-based.

That single missing piece, a real dose→concentration→brain-concentration
submodel, is what gates the entire behavioral half of the project. It is
built first, and it is buildable honestly: absorption, bioavailability,
volume of distribution, clearance, and brain:plasma partition coefficients
are published quantities for these drugs, not invented ones. Phase 1 exists
to remove that blocker; Phases 2 and 7 spend it.

---

## Phase 1: The dose→brain-concentration bridge

Unblocks all behavioral calibration. Nothing downstream of Phase 2 is honest
without it.

- **P1.1** Extend `pk.py` from bolus-concentration to administered dose: one-
  and two-compartment models with first-order absorption, parameterized by
  dose (mg/kg), bioavailability `F`, absorption rate `ka`, volume of
  distribution `Vd`, and clearance `CL`, with route (IV / IP / SC / PO) and
  body weight. IV bolus must reduce exactly to current behavior so no existing
  result moves.
- **P1.2 to P1.4** Literature PK parameter sets for fentanyl, haloperidol, and
  diazepam in rat and mouse, per route, each with citation and units, following
  the same sourcing discipline as the existing binding calibrations.
- **P1.5** Plasma→brain partition (`Kp` / `Kp,uu`) per drug: receptors sit in
  brain tissue, not plasma, and the ratio is drug-specific and published.
- **P1.6** Validate simulated concentration-time curves against published PK
  curves (Cmax, Tmax, half-life, AUC) with quantitative tolerances.

## Phase 2: Circuit-layer calibration (spending the Phase 1 unblock)

- **P2.1** Evidence module of real dose→behavior datasets that Track E found
  but could not use: the exact papers it named, now usable through the PK
  bridge.
- **P2.2** Bayesian fit of the three coupling gains
  (`mu_opioid_coupling`, `d2_coupling`, `gaba_a_connection_modulation`)
  against those targets, using the existing per-system independent-model
  pattern that converged cleanly.
- **P2.3** Fit connectivity weights and time constants where identifiable, and
  report explicitly which remain unidentifiable from behavioral data alone.
  An unidentifiable parameter stays labeled illustrative until data removes
  the label.

## Phase 3: Whole-brain connectome backbone

- **P3.1** Ingest the Allen Mouse Brain Connectivity Atlas normalized
  connection-density matrix and structure ontology into a versioned data module.
- **P3.2** Build `CircuitNetwork` from the connectome at selectable resolution
  (major divisions → ~43 regions → ~213 regions), with excitatory/inhibitory
  sign assignment derived from region neurotransmitter identity rather than
  guessed.
- **P3.3** Map every existing hand-built scenario onto connectome regions and
  regression-test that the established qualitative results (agonism, inert
  antagonism, antagonist reversal) survive the substitution.
- **P3.4** Whole-brain baseline stability: every node must sit in the sensitive
  part of its sigmoid at rest, at scale. This is the GPi-saturation lesson
  from the D2 scenario generalized into an automated check.

## Phase 4: Whole-brain receptor density maps

- **P4.1** Ingest Allen Mouse Brain Atlas ISH expression energy per receptor
  gene per region, for every gene the model uses.
- **P4.2** Convert expression to a density scale used by the receptor layer,
  documenting the proportionality assumption explicitly as an assumption.
- **P4.3** Validate against independently known distributions (Drd1/Drd2
  striatum-dominant, Oprm1 thalamus/striatum, Gabra1 cortex/cerebellum). A map
  that fails these is a bug, not a finding.

## Phase 5: Breadth of receptor systems

Each system follows the Phase 1 sourcing discipline: literature Ki/kon/koff,
efficacy, citations, tests.

- **P5.1** Serotonin 5-HT1A, 5-HT2A · **P5.2** Dopamine D1 / direct pathway
- **P5.3** Noradrenaline α1, α2, β · **P5.4** Acetylcholine nicotinic + muscarinic
- **P5.5** Glutamate NMDA, AMPA · **P5.6** Histamine H1, cannabinoid CB1, adenosine A2A
- **P5.7** Opioid kappa, delta · **P5.8** Monoamine transporters DAT/SERT/NET
  (required before any psychostimulant or SSRI can be modeled at all)

## Phase 6: Neuromodulatory projection systems

The step that makes this a brain rather than a large graph. A monoamine
receptor is not a term added to one node: it is the target end of a diffuse
projection system.

- **P6.1** Projection-system abstraction: source nucleus firing rate →
  transmitter release → region-wise transmitter concentration weighted by
  projection density → occupancy at every target region.
- **P6.2** Endogenous transmitter tone, so administered drugs compete with real
  transmitter at the same receptor pool rather than acting on an empty one.
- **P6.3** Projection densities for VTA/SNc, DRN/MRN, LC, basal forebrain, and
  TMN derived from the Phase 3 connectome.

## Phase 7: Behavior layer, made real

- **P7.1** Endpoint library: locomotion, anxiety, analgesia, reward, sedation,
  working memory, sensorimotor gating, each of them a multi-region readout.
- **P7.2** Nonlinear decoder, only where data supports one over the linear readout.
- **P7.3** Fit each endpoint against real dose-response curves via the Phase 1 bridge.

## Phase 8: Prediction and held-out validation

- **P8.1** Accept an arbitrary compound's binding fingerprint (PDSP-style Ki
  table) and construct its ligand set automatically.
- **P8.2** Predict behavioral profiles for drugs never used in fitting
  (morphine, clozapine, alprazolam, amphetamine) from binding alone, and report
  the errors quantitatively.
- **P8.3** Failure analysis: where the model is wrong and what would fix it.
  This section is expected to be substantial and is not a sign of failure.

## Phase 9: Numerics at scale

Sparse Jacobian, vectorized RHS, steady-state solver, benchmarks. ~200-400
nodes × ~15 receptor systems × multiple ligands is a few thousand coupled
states; the RK45-over-a-dense-RHS approach that works at 4 nodes will not.

## Phase 10: Outputs

Figures, a documented prediction API, and a manuscript skeleton.

---

## Standing discipline (applies to every task)

1. **Never invent a number.** A parameter is either literature-sourced with a
   citation, or explicitly labeled illustrative. "Searched and empty" is a
   valid, publishable result; Track E is the template.
2. **Never substitute a different drug's measured effect** for the one being
   modeled.
3. Tests must pass before any work is pushed.
4. Commit and push after each task, not at the end of a session.
5. When a parameter is calibrated, remove its illustrative label in the *same*
   commit that adds the calibration, and cite the source in `docs/architecture.md`.
