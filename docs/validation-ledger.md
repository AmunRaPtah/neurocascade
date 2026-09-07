# Validation ledger

**This file, not `tasks/backlog.json`, is the measure of progress.**

The backlog reached 70/70 on 2026-08-29 and 77/77 on 2026-09-03 while the model
could predict exactly one measured human quantity. Task count measures code
written. This ledger measures claims that survived contact with a number nobody
in this repository chose.

## Rules

1. A row is added only when a prediction has been compared to a measured value
   that was **not** used to fit it. Contracts, registries, gates and schemas
   never earn a row.
2. Every row names its held-out error, the split that produced it, and the
   source of the measurement.
3. A failed comparison earns a row. A negative result is a result; deleting it
   is the one thing this project has never done and will not start doing.
4. `searched-and-empty` earns a row in the second table, not the first.
5. No row may be added by the same task that produced the prediction without
   the comparison data having existed beforehand. If the data arrived after the
   prediction was published, say so in the row, as P19.6 did for Kapur 1996.

## Validated claims

| # | Claim | Held-out test | Error | Measured source | Status |
|---|---|---|---|---|---|
| V1 | Plasma haloperidol concentration → striatal D2 occupancy (hyperbola, EC50 1.8849 nM) | Split B, patient-disjoint leave-one-study-out | MAE 0.0351 occupancy units | Nyberg and Regenthal depot PET series | **validated, narrow** |
| V2 | Same, total-to-free basis conversion (fraction 0.09) | Split C, Videbaek IV free-plasma subjects (n=4) | MAE 0.0620, bias +0.0601 one-signed | Videbaek | **fails, systematically over-predicts** |
| V3 | Same, using externally published EC50 values | Split D, Kapur 1997 (1.3569 nM) and Fitzgerald 2000 (1.0642 nM) | MAE 0.0939, bias +0.0939, all 8 errors positive | published constants | **disagreement with literature, unresolved** |
| V4 | Oral dose → steady-state striatal D2 occupancy, whole chain | Kapur 1996, 2 mg/day, reported range 0.53–0.74 | single-dose peak 0.5253 misses low by 0.0047; regimen-matched steady-state mean 0.5635 contained | Kapur 1996 | **weak, not blind**: the 2 mg peak was published before the test ran |
| V5 | Terminal half-life implied by 13 FDA-transcribed population-PK parameter sets (7 applications, 5 drugs) | approved-label terminal half-life, a different FDA surface (openFDA/DailyMed) from the review PDFs the parameters came from, and non-compartmental rather than model-derived | **5 of 13 within twofold, 8 outside.** Passes: UZEDY parent 3.21 h vs 3 h (1.07x), UZEDY 9-OH 11.5 h vs 21 h (1.83x), ZYPREXA oral 37.4 h and LYBALVI 49.8 h and RELPREVV 34.3 h all inside 21–54 h. Failures: SUSTENNA 48.8 vs 23 (2.12x), MAINTENA and ASIMTUFII 168.0 vs 75 (2.24x), BARHEMSYS 13.6 vs 4–5 (2.72x), ERZOFRI 293.8 vs 23 (12.8x), TRINZA 353.8 vs 23 (15.4x), ARISTADA 4221.6 vs 75 (56.3x), ARISTADA INITIO 12296.2 vs 75 (164x) | FDA labels; parameters from Drugs@FDA Clinical Pharmacology reviews | **mostly fails, and the failures are the finding.** Every immediate-release row passes; every failure but one is a depot, where the label half-life describes site release and the model's slowest eigenvalue does not. The two worst carry 5.0% and 18.0% of AUC in that phase, so their "terminal" root is a release constant parked in a peripheral compartment. RELPREVV is the control that rules out "depots cannot be checked": it is a depot, it models release explicitly, and it passes at 1.00x. **Independence is partial**: 4 of the 5 label sources are ANDA generic copies of the innovator label, and amisulpride's label quotes clearances from the very model under test. No tolerance was widened; twofold was fixed before the numbers were computed |

### Standing caveats on V1–V4

- The four points V1 is fitted on come from two studies of eight patients each,
  measured at weeks 1 and 4, so they are **not four independent observations**.
  The profile interval is narrower than the data justifies.
- All four fitted points carry `tracer=None` **and** `region=None`. The
  "striatal" label on the project's central constant is inherited from context,
  not printed by either source.
- The EC50 is fitted on depot intramuscular steady state and driven as oral.
  This is the largest unrepaired mismatch in the chain.
- 11 of 15 registered PET observations have `region=None`; tracer spread across
  raclopride, IBZM and FLB 457 sets a 2–2.5× floor larger than every held-out
  error above.

## Untested and unvalidated

| Layer | State | What it would take |
|---|---|---|
| Oral dose → plasma concentration | **untested, not failed.** `chain_testable_observations()` returns 0 | a study printing per-subject oral dose, plasma concentration and occupancy in one table |
| Occupancy → effect (`D2Effect`) | **not a prediction.** `emax=1.0, baseline=0.0` makes `abs(effect)` equal occupancy identically | a human dose-response set identifying an Emax and a baseline |
| Effect-site kinetics (ke0) | **absent.** Chain maps plasma to occupancy instantaneously | a fitted human ke0; the four `HUMAN_KE0_RECORDS` are illustrative |
| Occupancy → clinical outcome | **falsified as a sole determinant.** Siafis 2023: haloperidol 11 mg (EPS OR 4.7) vs risperidone 16 mg (OR 3.3) at ~87% occupancy both | the off-target fingerprint test, Phase 23 |
| Any circuit-layer parameter | **all illustrative.** The only fitted one, `GABA_A_CONNECTION_MODULATION_GAIN`, was rat-fitted and retired in the human pivot | frozen; blocked on data, not effort |
| Any behavioural endpoint | **none.** No test anywhere compares a predicted behaviour to a measured one | out of scope for v1 |
| Mu-opioid and GABA-A chains | **scaffolding only.** Every PK parameter and receptor Ki is `None` behind a gate | frozen for v1 |
| Any drug other than haloperidol → D2 occupancy | **no prediction exists yet, and P21.1 did not create one.** P21.2 established that plasma EC50 values are sourceable for six drugs plus a seventh second-hand. P21.1 then turned the hardcoded haloperidol constant into a drug-keyed registry carrying basis, analyte, Emax convention, region, tracer and provenance per row. That is a schema, and by rule 1 a schema never earns a row: P21.1 **produced no comparison against an externally measured number**, made no new prediction, and deliberately added **no drug**. The registry still has exactly one entry, haloperidol, holding the same float it held before; a test asserts the predictions are bit-identical to the pre-refactor code | ingesting the six reachable drugs with their printed qualifiers, then a held-out occupancy series per drug that was not used to fit that drug's EC50 |
| Multi-drug dose → plasma concentration, in the chain | **sourced but not connected.** P21.3 transcribed 17 fitted parameter sets across 7 drugs from FDA reviews and validated 13 of them against label half-lives (row V5), but none is wired into `dose_to_occupancy`. `_require_pk_record` is haloperidol-only by construction and refuses anything that is not two-compartment, absolute and single-analyte; most of the new registry is none of those | a decision about what the chain does with an apparent parameter set (`CL/F`, `V/F`) and with a parent-plus-metabolite fit. That is a modelling choice, not a transcription, and is deliberately not made here |
| Inter-individual variability from the FDA reviews | **transcribed, unused.** Every P21.3 row that printed IIV carries it with its scale (`CV%`, `variance`, `SD`), shrinkage where given, and an inter-occasion flag, because the scale convention differs between reviews. Nothing consumes any of it | a population simulation layer, which does not exist |
| Quetiapine → D2 occupancy | **outside the model's form, not merely unfitted.** Trough occupancy falls to single digits within a dosing interval, and both primaries that attempted the fit report it failed | effect-site kinetics (ke0), which the chain does not have; a direct-effect plasma model cannot represent this drug at any EC50 |

## Searched and empty

Recorded as real results, per the standing honesty discipline.

| Sought | Searched | Outcome |
|---|---|---|
| Haloperidol IV/oral crossover for absolute F | P19.2 | Holley 1983 (PMID 6831826) prints only CL and Vss; Chang 1992 (PMID 1579624) prints no micro-constants; both paywalled |
| Volume and Q allometric exponents | P19.2 | source prints an exponent for clearance only; chain refuses off-70 kg rather than defaulting |
| Per-subject rows behind Kapur 1997 / Fitzgerald 2000 | P19.3 | unreachable; the V3 disagreement stays unresolved |
| Extrastriatal D2 | P19.6 | Tuppurainen 2009 excluded (n=2, no plasma); Kessler 2005 fallypride file is a 564-byte HTML stub |
| Adult diazepam population PK | P14.4 | none exists; available model is pediatric |
| Held-out compounds with a complete dose-to-behaviour chain | P8.3 | none of seven has one |
| US approved oral dose range for amisulpride | P20.6 | not approved as an antipsychotic in the United States: its only application is Barhemsys, intravenous, for postoperative nausea and vomiting. Excluded from the P22.2 label-dose test; retained for the P22.3 PET test, where the European evidence applies |
| The pooled per-drug ED50 table in Uchida 2011 (PMID 21508857) | P21.2 | the paper is unreachable: Unpaywall returns `is_oa false / closed` with no OA location, Europe PMC returns `inPMC N / inEPMC N`, and the abstract prints per-drug prediction errors with no ED50 and no concentration unit. **The numbers are reachable even though the paper is not**: all five ED50 values, in both the free-Emax and Emax-fixed-100% parameterizations, are quoted verbatim in open-access secondaries including Uchida's own 2014 restatement. Recorded as second-hand provenance, never as a read of the primary |
| A plasma EC50 for quetiapine D2 occupancy | P21.2 | none exists. Vernaleken 2010 "did not furnish valid fits for EC50 values in any brain region"; Gefvert 2001 states the correlation "could not be calculated"; de Greef 2011 and Uchida 2011 both excluded the drug. The only readable number (583 ng/mL) is review-reported from SPECT with unverified attribution. This is a property of the drug's occupancy time course, not of the search |
| An FDA-sourced occupancy-concentration relation for any antipsychotic | P21.2 | none. Drugs@FDA checked for ziprasidone (NDA 20-825), aripiprazole (NDA 21-436) and paliperidone (NDA 21-999): labels print in-vitro Ki not in-vivo EC50, older biopharmaceutics reviews are scanned and unindexed. Phase 21's "FDA-sourced" title applies to the PK layer (P21.3), not the occupancy layer. **Corrected by P21.3:** this row originally also claimed "the Invega clinical pharmacology review 404s at every URL variant". It does not 404. NDA 021999 fetches normally once the TOC extension is read as `.html` rather than `.cfm`; its population-PK section is *withheld*, 13 consecutive pages replaced by a full-page Trade Secret pull-sheet. A retryable routing failure and a permanent redaction are opposite facts that look identical from a script |
| A population-PK parameter set for quetiapine, from any FDA source | P21.3 | none, and not for an access reason. NDA 022047 (SEROQUEL XR, review 2007-05-10) was fetched and read: no population-PK analysis was ever performed. Recorded as `not-fitted`, which is distinct from redacted or unextractable because there is nothing to retry for. Quetiapine is now empty on both Phase 21 layers, for two unrelated reasons: no plasma EC50 (P21.2, a property of its occupancy time course) and no fitted popPK (here) |
| A population-PK parameter set for clozapine or ziprasidone | P21.3 | none. Clozapine VERSACLOZ (NDA 203479, 2012-09-04) was fetched and contains no population-PK analysis (`not-fitted`). The older clozapine and ziprasidone applications predate routine electronic posting of review documents, so nothing is available to fetch |
| A population-PK parameter set for oral aripiprazole | P21.3 | none. NDA 021436 (ABILIFY, 2007-10-19) and NDA 216655 (OPIPZA, 2024-07-19) were both fetched and read; neither performed a population-PK analysis. Every aripiprazole parameter set in the registry comes from a depot application. Population PK entered this therapeutic area with the long-acting injectables, not with the oral originals |
| Printed final parameter estimates for RISVAN risperidone (NDA 214835) | P21.3 | none, from this source. The review discusses the population-PK model and reports its validation, but no parameter table is printed anywhere in the document. Recorded as `estimates-not-printed` rather than as a failed extraction, since repeated OCR of a table that was never typeset returns nothing |
| Independence of Uchida 2011's haloperidol pool from this repository's own fit | P21.2 | undeterminable. Uchida's Emax-constrained haloperidol ED50 of 0.70 ng/mL sits 1.2% from the 0.708 ng/mL fitted here in P19.4, but the list of studies behind his n=35 pool is inside the paywalled paper, so overlap with Nyberg 1995 and Regenthal 1997 cannot be ruled out. **Not counted as a replication and not entered in the validated table** |
| A current innovator oral haloperidol label | P20.6 | none. Haldol oral tablets were discontinued, so all 68 current US oral labels are ANDA copies of a withdrawn label, and all use the pre-modern indication wording "management of manifestations of psychotic disorders". The dose range is still usable, but it is a generic's copy and is marked as such on the row |
