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
| Plasma concentration → D2 occupancy for any drug other than haloperidol | **sourceable but unsourced.** P21.2 audited reachability and found primary human PET values for 8 of 9 candidate drugs. No value is loaded, no ingestion code exists and no prediction was made, so nothing here is testable yet | P21.3 onward: ingest per drug with the concentration basis (parent, active moiety, serum, whole blood), the Emax convention and the study population recorded alongside each constant, then hold out against measured PET in P22.3. Record is `docs/p21-2-multi-drug-ec50-reachability.md` |

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
| A current innovator oral haloperidol label | P20.6 | none. Haldol oral tablets were discontinued, so all 68 current US oral labels are ANDA copies of a withdrawn label, and all use the pre-modern indication wording "management of manifestations of psychotic disorders". The dose range is still usable, but it is a generic's copy and is marked as such on the row |
| A paliperidone plasma ED50 for D2 occupancy at primary strength | P21.2 | Arakawa 2008 (PMID 18058087) does report striatal ED50 6.65 ng/mL, but its free abstract prints the ED50 only in mg/day, Springer is paywalled, and the green-OA flag on the DOI points at QST repository record 45147, which is metadata-only with zero file objects in both its JSON and OAI-PMH forms. The number is readable only in a Janssen medical-information page and in US patent 9439906B2, both secondary. Recorded as an attributed secondary reading and not written into the model. Paliperidone is the one drug of nine that failed the reachability audit |
| The Uchida 2011 pooled per-drug ED50 table (PMID 21508857) | P21.2 | closed at every free route: Unpaywall `oa_status: closed`, absent from PMC and Europe PMC, LWW 403, Ovid abstract-only, nothing at CORE, BASE, Toronto or Keio. Its own abstract also shows it only ever covered five drugs (risperidone, clozapine, olanzapine, haloperidol, ziprasidone), so it never held a quetiapine, aripiprazole, amisulpride or paliperidone value. Obtaining it would not have produced this phase's table, which contradicts the P21.2 brief's premise |
