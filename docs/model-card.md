# Model card: neurocascade v1, oral dose to striatal D2 occupancy

This card describes what `neurocascade.human.predict(drug, dose_mg, regimen)`
returns and how far that output can be trusted. The limits come first because
the model is narrow. It predicts one quantity, a striatal dopamine D2 receptor
occupancy-time curve, for two drugs. Every other question a reader might bring
to a "drug to brain to behaviour" model is out of scope.

How to read the citations. Each claim below ends with one or more bracketed
references to a row of [`validation-ledger.md`](validation-ledger.md). `[V4]`
is row V4 of the validated-claims table. `[ledger: Extrastriatal D2]` is the
row of the untested or searched-and-empty table whose first cell begins with
that text. Passages set as block quotes are carried verbatim from the P19.7
scope document, [`p19-7-scope-and-failure-analysis.md`](p19-7-scope-and-failure-analysis.md).
`tests/test_model_card.py` checks that every claim carries a citation, that
every citation resolves to a ledger row, that every quote is verbatim, and that
the headline numbers are the ones the code computes.

## 1. Do not use this model for

- **Predicting behaviour, a clinical outcome, or a side effect.** The output
  is an occupancy. No test anywhere in the repository compares a predicted
  behaviour to a measured one. Occupancy is already falsified as the sole
  determinant of outcome: Siafis 2023 reports haloperidol (EPS odds ratio 4.7)
  and risperidone (3.3) at about the same 87% occupancy, and V11 places
  clozapine's 65% concentration 2.94x above the top of its effective plasma range. [ledger: Any behavioural endpoint] [ledger: Occupancy → clinical outcome] [V11]

  > Saying "5 mg reaches the occupancy Kapur 2000 associated with extrapyramidal side effects" is a statement about a number. Saying "5 mg causes extrapyramidal side effects" is a behavioural prediction, and this chain does not make it.

  Ledger: [ledger: Occupancy → clinical outcome]

- **Using the serotonin-dopamine fingerprint to rank drugs for extrapyramidal
  risk.** The P23.3 test of that hypothesis is indeterminate: the 5-HT2A:D2
  ratio trends in the hypothesised direction (R² 0.314 to 0.636), but at n = 7
  the nested test gives p = 0.133. `predict()` does not return off-target
  occupancy. The Ki-derived off-target curves elsewhere in the repository carry
  an error floor of up to 0.368 occupancy units. [V12] [ledger: Occupancy → clinical outcome]

- **Any drug other than haloperidol and olanzapine.** `predict()` refuses
  every other drug before computing anything and names every missing layer.
  The refusals are listed in section 7. Risperidone and aripiprazole are
  refused on principle, not merely unfitted: the parent-only path misprices
  risperidone's occupancy by up to 0.568. Quetiapine lies outside the model's
  form. The mu-opioid and GABA-A chains are scaffolding with no parameters. [ledger: Public prediction interface] [ledger: Risperidone or aripiprazole → D2 occupancy] [ledger: Quetiapine → D2 occupancy] [ledger: Mu-opioid and GABA-A chains]

- **Any region outside the striatum.** [ledger: Extrastriatal D2] [V1]

  > Extrastriatal D2 is out of scope and is not weakly supported. It is not supported at all.

  > Applying this constant to a cortical or thalamic region would be a category error, not an extrapolation.

  Ledger: [ledger: Extrastriatal D2]

- **Any route but oral, including the depot route the haloperidol constant
  was fitted on.** The haloperidol PK set is oral and apparent, with F never
  identified, and the search for an IV crossover came back empty. [ledger: Haloperidol IV/oral crossover for absolute F] [V1]

  > There is no IM record of any kind, which is the ironic corollary of the regimen mismatch: the chain cannot simulate the route its own EC50 was measured on.

  Ledger: [V1]

- **Any body weight other than the 70 kg reference, or any individual
  patient.** No sourced set prints volume or Q exponents, and the interface
  takes no weight argument. Every prediction is a typical-value prediction
  with no interval, because no between-subject variability or covariate is
  applied. [ledger: Volume and Q allometric exponents] [ledger: Between-subject variability and covariates]

- **Chronic dosing beyond ordinary accumulation.** This means tolerance,
  receptor regulation, or occupancy after weeks of treatment. [ledger: Chronic pharmacology]

  > The chain has no time-varying pharmacology. There is no tolerance term, no receptor up- or down-regulation, no autoreceptor adaptation, and no accumulation term beyond ordinary two-compartment superposition.

  > Chronic dosing is therefore out of scope in the strong sense: the chain does not merely lack a chronic term, its residuals show where the missing term would go.

  Ledger: [ledger: Chronic pharmacology]

- **Any time-resolved question about the brain that runs ahead of or lags
  behind plasma.** The chain has no effect compartment. [ledger: Effect-site kinetics (ke0)] [ledger: Quetiapine → D2 occupancy]

  > Kurose 2020 (PMID 32726002) reports central occupancy decaying more slowly than plasma concentration, which this chain cannot represent at all.

  Ledger: [ledger: Effect-site kinetics (ke0)]

- **Whole-brain, circuit or network simulation, read as a drug prediction.**
  Every circuit-layer parameter is illustrative and those layers are frozen
  for v1. `predict()` does not reach them. [ledger: Any circuit-layer parameter]

- **Reading the transducer output as a second result.** For haloperidol,
  `D2Effect` with Emax 1 and baseline 0 is the occupancy curve with a sign.
  The transducer refuses aripiprazole by name. [ledger: Occupancy → effect (`D2Effect`)]

## 2. Unrepaired mismatches

Each row is a known way the output can be wrong. None has been repaired; the
ledger cell cited on each row says what data would repair it. [V1] [ledger: Multi-drug chain, layer by layer (P22.5)]

| Mismatch | Drug | What it does to an output | Ledger |
|---|---|---|---|
| Regimen and route: the EC50 was fitted on depot intramuscular steady state, and the chain is driven by oral doses | haloperidol | The one whole-chain test misses on a single-dose reading (peak 0.5253 against the 0.53–0.74 range) and is contained only on a steady-state reading computed after the miss was seen | [V1] [V4] |
| The dose-to-plasma arm has never been tested against an occupancy | both | No observation pairs an oral dose with a measured occupancy in the same subjects (haloperidol 0 pairs, olanzapine no PET set) | [ledger: Oral dose → plasma concentration] [ledger: Multi-drug chain, layer by layer (P22.5)] |
| No held-out occupancy series for olanzapine | olanzapine | The olanzapine EC50 has only ever been checked through the dose domain (V8), never against a held-out PET point | [ledger: Held-out PET observations] [V8] |
| Basis: total versus free plasma | haloperidol | Converting with free fraction 0.09 systematically over-predicts the free-plasma subjects (MAE 0.0620, bias +0.0601). The two bases are never pooled | [V2] |
| The fitted constant disagrees with the published constants | haloperidol | Kapur 1997's and Fitzgerald 2000's EC50s over-predict all 8 points (MAE 0.0939). The disagreement is unresolved because the per-subject data are unreachable | [V3] [ledger: Per-subject rows behind Kapur 1997 / Fitzgerald 2000] |
| Region, and for haloperidol also tracer, are not printed on the fitting observations | both | The 'striatal' label is inherited from context for both constants. Haloperidol's four fitted points carry neither field, and tracer spread across the registered PET set sets a 2–2.5x floor, larger than every held-out error. Olanzapine's constant prints [11C]raclopride but no region | [V1] [ledger: Multi-drug chain, layer by layer (P22.5)] |
| The fit is not independent | haloperidol | Four group means from two studies of eight patients each, at weeks 1 and 4. The profile interval is narrower than the data justify | [V1] |
| Thresholds measured in one drug are applied to another | olanzapine | The 65/72/78% thresholds were measured in haloperidol-treated patients (Kapur 2000). Placing olanzapine against them transfers a haloperidol relation, and the output labels it `same_drug=False` | [ledger: Public prediction interface] |
| Single-dose readings against steady-state thresholds | both | The thresholds are steady-state relations. A single-dose peak placed against them is flagged as a regimen mismatch | [ledger: Public prediction interface] [V4] |
| The label-dose pass has little discriminating power | both | The V8 window scales exactly with EC50. Olanzapine passes for any EC50 within 0.73–2.9x of its value and haloperidol within 0.48–1.93x, which is no wider than the tracer floor | [V8] |
| Population exchangeability is assumed | both | PK, PET and threshold observations come from different people, recruited under different routes and regimens, and nothing tests whether they can be composed | [ledger: Populations behind the two drivable chains] |
| Typical values only | both | No between-subject variability and no covariates are applied: smoking, sex, race, age, CYP2D6 | [ledger: Between-subject variability and covariates] |
| Absorption lag dropped | olanzapine | LYBALVI's ALAG of 0.782 h is not carried. Steady-state statistics are exactly unchanged, but a single-dose curve runs early by that lag | [ledger: Olanzapine single-dose time course] |
| Active metabolite ignored | haloperidol | Reduced haloperidol is bounded at 0.0030 worst-case occupancy error, but the bound rests on a rat affinity | [ledger: Reduced haloperidol's contribution to V1's occupancy] [ledger: A human D2 affinity for reduced haloperidol] |
| No effect compartment | both | Occupancy follows plasma instantaneously. Any lag, hysteresis or slow off-rate in the brain is absent | [ledger: Effect-site kinetics (ke0)] |
| No chronic pharmacology | both | Weeks-long adaptation is absent, and the haloperidol fit's week-4 residuals show where it would go | [ledger: Chronic pharmacology] |

## 3. The populations the observations came from

The haloperidol chain is assembled from groups that share no subject. P19.7
described them as follows. [V1] [V2] [V4] [ledger: Populations behind the two drivable chains]

> **PK set.** Pilla Reddy 2013 (PMID 24113674): 122 subjects, of whom 20 are healthy volunteers and 102 are patients with schizophrenia; 538 concentrations; 1-60 mg/day; oral in every arm.

> **Occupancy set.** The four fitted points come from two depot intramuscular studies of eight patients each, two independent patient groups in total. Route, formulation and regimen all differ from the PK set.

> **The whole-chain comparison point.** Kapur 1996, seven patients, oral, steady-state daily.

> **The free-basis set.** Videbaek, four subjects, IV, free plasma. Disqualified for the oral case and used only as split C's out-of-basis probe.

> So the chain composes an oral PK model fitted mostly in schizophrenia patients with an occupancy constant fitted in depot-treated patients, and validates the composition against a third group of seven oral patients. No subject appears in more than one of these sets. Whether the sets are exchangeable is **not determined in the repo**; the honest statement is that the composition assumes exchangeability and nothing tests it.

> Sex, age, ancestry, smoking status and CYP2D6 genotype are not recorded for any of these sets in this repository. Haloperidol is a CYP2D6 substrate, and none of the machinery here represents that.

Ledger: [V1] [V2] [V4] [ledger: Populations behind the two drivable chains]

The olanzapine chain postdates P19.7, so the following comes from the
olanzapine rows rather than from the P19.7 document. The PK set is the FDA
population-PK model in the LYBALVI review (NDA 213378). It pools healthy adults
and adults with schizophrenia across the ALK3831 programme. Its typical values
describe a healthy 36-year-old, 70 kg, non-Black male non-smoker who took
olanzapine co-formulated with samidorphan. The LYBALVI label states that there
is no PK interaction between the two components. [V8] [ledger: Populations behind the two drivable chains]

The olanzapine EC50, de Greef 2011 (5.29 ng/mL, fitted Emax 0.875), is a pooled
fit to published [11C]raclopride PET studies. Those studies used single and
multiple oral doses in healthy volunteers and patients with schizophrenia, and
the fit prints no region. Whether the PK population and the PET population are
exchangeable is not tested, as for haloperidol. [V8] [ledger: Populations behind the two drivable chains]

The thresholds come from Kapur 2000, a study of haloperidol-treated patients
given 1.0 or 2.5 mg/day for 2 weeks. [ledger: Public prediction interface]

## 4. Intended use

- **What it answers.** For a typical 70 kg adult taking oral haloperidol or
  olanzapine, it gives the predicted striatal D2 occupancy over time, either
  over one 24 h interval at once-daily steady state or over 72 h after a
  single dose. It also reports where that occupancy sits against the sourced
  Kapur 2000 and Uchida 2011 thresholds, flagging whether each threshold was
  measured in the same drug and under the same regimen. [ledger: Public prediction interface] [V4] [V8]
- **The reading closest to the evidence** is the once-daily steady-state mean
  for haloperidol. That is the regimen class the EC50 was fitted on and the
  statistic V8 pre-registered. It is still a depot fit driven orally. [V1] [V4] [V8]
- **Who it is for.** Researchers asking an occupancy question who will carry
  the limitations above into whatever they conclude. It is not for
  prescribing. A dose that the model places at or above 0.78 occupancy is a
  statement about occupancy, not a prediction of extrapyramidal symptoms. [ledger: Occupancy → clinical outcome] [ledger: Any behavioural endpoint]
- **Refusals are part of the intended use.** A refused drug comes back with
  every missing layer named. That tells the caller which data would have to
  exist before the drug could be predicted. [ledger: Public prediction interface] [ledger: Multi-drug chain, layer by layer (P22.5)]

## 5. What the model can do, and how well

All of the D2 occupancy evidence in the validated-claims table is listed
below, failures included. Only V4 and V8 run the whole dose-to-occupancy chain;
V1 to V3 and V9 to V10 test the plasma-to-occupancy hyperbola alone. [V1] [V4] [V8] [V9]

| Claim | Result | Status | Ledger |
|---|---|---|---|
| Haloperidol plasma to striatal D2 occupancy, EC50 1.8849 nM, patient-disjoint split | MAE 0.0351 | validated, narrow | [V1] |
| Same, after conversion to the free-plasma basis | MAE 0.0620, bias +0.0601 | fails | [V2] |
| Same, with the published Kapur 1997 and Fitzgerald 2000 EC50s | MAE 0.0939, every error positive | unresolved disagreement | [V3] |
| Haloperidol oral 2 mg/day, whole chain, against Kapur 1996's range 0.53–0.74 | single-dose peak 0.5253 misses; steady-state mean 0.5635 contained, computed after the miss | weak, not blind | [V4] |
| Terminal half-life of FDA popPK sets against label half-lives | 5 of 13 within twofold; every immediate-release set passes, every failure but one is a depot | mostly fails | [V5] |
| Once-daily dose window for 0.65–0.78 mean occupancy against the label dose range | haloperidol 2.89–5.56 mg/day vs 1.0–15.0, olanzapine 5.77–16.4 mg/day vs 10.0–20.0; 2 of 2 scored pass, 6 refused | passes, weakly discriminating | [V8] |
| Aripiprazole plasma to occupancy, published EC50, leave-one-drug-out | MAE 0.0244 | validated, narrow, not drivable by the chain | [V9] |
| Ziprasidone plasma to occupancy, published EC50 | one point, error −0.424 | fails, anecdotal | [V10] |
| Clozapine's 65% concentration against its effective plasma range | needs 1764 ng/mL, 2.94x above the range top | the occupancy-threshold hypothesis is falsified for clozapine | [V11] |
| Risperidone active-moiety ED50 from paliperidone under equipotency | fold error 1.696 | fails | [V6] |
| Aripiprazole metabolite exposure ratio from two EC50 bases | derived 1.00 vs label 0.40 | fails | [V7] |

Only the V1, V4 and V8 rows describe what `predict()` computes, and a test
asserts that its output is bit-identical to those computations. The other rows
bound how far the approach generalises. They are not capabilities of the
interface. [ledger: Public prediction interface] [V1] [V4] [V8]

## 6. How to read an output

- `Prediction.provenance` is a gate flag, not a grade. It reads `"measured"`
  whenever no layer is illustrative, including for haloperidol, whose
  occupancy layer is `fitted-in-repo`. The per-layer provenance is in
  `Prediction.layers`. [V1] [ledger: Public prediction interface]
- Each `ThresholdPosition` carries `same_drug` and `same_regimen`. When either
  is false, the comparison is a transfer, not a measurement. [ledger: Public prediction interface]
- `Prediction.caveats` always includes the occupancy-is-not-the-outcome
  statement. It should go with any figure that is shown next to a threshold. [ledger: Occupancy → clinical outcome]
- `Prediction.metabolite_caveats` carries the reduced-haloperidol bound. [ledger: Reduced haloperidol's contribution to V1's occupancy]

## 7. Drugs refused today, and at which layers

| Drug | Refused at | Ledger |
|---|---|---|
| risperidone | active-metabolite, pk, occupancy, exposure | [ledger: Risperidone or aripiprazole → D2 occupancy] [ledger: Public prediction interface] |
| aripiprazole | active-metabolite, occupancy, exposure | [ledger: Risperidone or aripiprazole → D2 occupancy] [ledger: Public prediction interface] |
| paliperidone | pk, occupancy, exposure | [ledger: Multi-drug chain, layer by layer (P22.5)] |
| quetiapine | occupancy, exposure | [ledger: Quetiapine → D2 occupancy] [ledger: A population-PK parameter set for quetiapine] |
| ziprasidone | pk, occupancy, exposure | [ledger: A population-PK parameter set for clozapine or ziprasidone] |
| clozapine | pk, occupancy, exposure | [ledger: A population-PK parameter set for clozapine or ziprasidone] |
| amisulpride | pk, occupancy, exposure | [ledger: Multi-drug chain, layer by layer (P22.5)] |
| fentanyl | pk, occupancy | [ledger: Mu-opioid and GABA-A chains] |
| diazepam | pk, occupancy, exposure | [ledger: Mu-opioid and GABA-A chains] [ledger: Adult diazepam population PK] |

Since P25.2, quetiapine and aripiprazole clear the pk layer, through the journal-literature records Zheng 2024 and Knights 2015, and every other layer still refuses them. [V14] V8's table still records both as refused at pk, because that result is frozen at the P22 sources. [V8]

Supported: haloperidol and olanzapine, the two drugs V8 scored. [V8] [ledger: Public prediction interface]
