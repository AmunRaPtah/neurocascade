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
| An FDA-sourced occupancy-concentration relation for any antipsychotic | P21.2 | none. Drugs@FDA checked for ziprasidone (NDA 20-825), aripiprazole (NDA 21-436) and paliperidone (NDA 21-999): labels print in-vitro Ki not in-vivo EC50, older biopharmaceutics reviews are scanned and unindexed, and the Invega clinical pharmacology review 404s at every URL variant. Phase 21's "FDA-sourced" title applies to the PK layer (P21.3), not the occupancy layer |
| Independence of Uchida 2011's haloperidol pool from this repository's own fit | P21.2 | undeterminable. Uchida's Emax-constrained haloperidol ED50 of 0.70 ng/mL sits 1.2% from the 0.708 ng/mL fitted here in P19.4, but the list of studies behind his n=35 pool is inside the paywalled paper, so overlap with Nyberg 1995 and Regenthal 1997 cannot be ruled out. **Not counted as a replication and not entered in the validated table** |
| A current innovator oral haloperidol label | P20.6 | none. Haldol oral tablets were discontinued, so all 68 current US oral labels are ANDA copies of a withdrawn label, and all use the pre-modern indication wording "management of manifestations of psychotic disorders". The dose range is still usable, but it is a generic's copy and is marked as such on the row |
