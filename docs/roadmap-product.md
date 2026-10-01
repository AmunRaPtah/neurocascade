# Product roadmap (Phases 28-32), from 2026-10-02

This supersedes `docs/roadmap-human.md` as the plan the backlog follows. The
human roadmap stays as the record of Phases 11-27 and of the frozen layers'
unfreezing conditions.

## What neurocascade is for

A tool for CNS drug discovery teams. A team gives it a compound's **preclinical
package** and gets back the predicted **human** consequences: human PK, the dose
that engages the target, occupancy over time across a receptor panel, and the
expected effect and liability profile, each with an uncertainty band whose
width is measured, not asserted.

Input, the package a discovery team actually holds before first-in-human:

- receptor binding panel (Ki) with functional efficacy per target;
- in-vitro ADME: plasma unbound fraction, brain-tissue binding, microsomal or
  hepatocyte intrinsic clearance;
- rodent PK and rodent Kp,uu;
- optionally rodent receptor occupancy and behavioural dose-response.

Output: predicted human CL, Vss, half-life and oral exposure; predicted dose
for the class's target-engagement occupancy; occupancy-time courses across the
panel at that dose; probabilities for named clinical effect categories; the
nearest benchmark drugs; provenance and band for every number.

The manuscript is ancillary. It writes up the benchmark result.

## Why the plan changed

By P25.3 the shipped product predicted striatal D2 occupancy for five
antipsychotics and refused everything else, because every layer had to be a
measured human value. A compound in discovery has no measured human value at
all, so that rule excludes the product's whole use case. Two rules change:

1. **Translation is the product, not a contamination.** Rodent and in-vitro
   data are the input. Using them to predict a human quantity is allowed when
   it is an explicit, named translation step whose error has been measured on
   the benchmark. Silently substituting one species' number for another's is
   still forbidden.
2. **Calibrated bands replace refusals.** A missing optional input widens the
   band, by an amount measured on the benchmark. A refusal is only for a
   missing core input (no binding data at all, no clearance information of any
   kind).

Everything else about the project's discipline stands: no invented numbers,
cited sources, searched-and-empty records, ledger rows, no tolerance widening.

## The one test that decides usefulness

**Retrospective benchmark.** About 150 CNS drugs, each given only preclinical-
type data (in-vitro and animal measurements), scored against what was observed
in humans: PK, clinical dose, PET occupancy where it exists, and labelled
adverse-event profile. A drug-level held-out set is fixed and hashed in Phase 28
before anything is fitted. Every headline number reported to a user is a
held-out number.

## Phases

| Phase | Builds | Gate (thresholds fixed before scoring) |
|---|---|---|
| 28 Benchmark | drug list and locked split; receptor panel Ki + efficacy; preclinical ADME/PK; human truth (PK, dose, PET); clinical effect labels | benchmark frozen, coverage reported per field |
| 29 Translation | species gate allows labelled translation inputs; human PK from preclinical; human occupancy from Ki + Kp,uu; predicted human dose | held-out fold-error for CL and dose; MAE for PET occupancy |
| 30 Effects | occupancy-profile to clinical-effect model vs baselines; failure analysis | held-out AUROC with bootstrap CI, beats single-target and nearest-neighbour baselines |
| 31 Mechanism | circuit engine as explanation; does it add held-out predictive value | reported either way |
| 32 Product | compound schema + `predict_compound()`; CLI + pip; one-page report; web UI; blind demo on held-out compounds | a held-out compound run end to end from a package file |

Phase 29's PK translation is the main risk: rodent-to-human dose prediction is
often off by 3-10-fold. It runs first so that a too-wide band is found in days.

## Data sources (all free)

- Binding: PDSP Ki database, ChEMBL (API), IUPHAR/BPS Guide to Pharmacology
  (efficacy, action type).
- Human PK truth: Lombardo et al. 2018 (Drug Metab Dispos 46:1466, 1352 drugs,
  IV human PK); FDA labels via openFDA for oral parameters.
- Preclinical PK and binding: published rat PK compilations, Kp,uu compilations
  (Fridén 2009, Loryan 2022 and their sources), ChEMBL ADME assays.
- Clinical dose: WHO ATC/DDD index; label dose ranges.
- Clinical effects: FDA label adverse-reaction tables (openFDA), SIDER 4.1.
- PET occupancy: this repo's existing registries plus literature.

## Parked

P24.3 (manuscript), P26.1-26.2 (EPS on 7-9 drugs), P27.1-27.3 (CLI and report
for the two-drug product, folded into Phase 32). They are marked `blocked` in
the backlog with the reason.
