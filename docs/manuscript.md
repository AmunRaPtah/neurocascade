# A mechanistic oral-dose-to-striatal-D2-occupancy chain for antipsychotics, validated against FDA label dose ranges and held-out PET

**Research article — journal-direct draft (v1 deliverable, P24.3).**

This is the manuscript for the deliverable the human track actually reached: a
mechanistic chain that converts an oral antipsychotic dose into a predicted
striatal dopamine D2 receptor occupancy-time curve, for the two drugs that clear
every layer, and the held-out comparisons that chain survived or failed.

It is drafted from [`validation-ledger.md`](validation-ledger.md), not from the
roadmap. Every quantitative claim ends with a bracketed reference to a ledger
row: `[V4]` is a row of the validated-claims table; `[ledger: X]` is the row of
the untested or searched-and-empty tables whose first cell begins with `X`.
`tests/test_manuscript.py` checks that every quantitative claim carries such a
citation, that every citation resolves to exactly one ledger row, that the
negative and indeterminate rows are reported, that the headline numbers are the
ones the code computes now, and that the frozen layers are named as out of scope
rather than claimed. It supersedes [`manuscript-skeleton.md`](manuscript-skeleton.md),
which was written at mouse scale before the human pivot and is retained only as
the record of that earlier scope.

## Abstract

### Background

Antipsychotic drugs are routinely described by a chain from plasma exposure to
dopamine D2 receptor occupancy to clinical effect, but the links in that chain
are measured to very different standards, and a model that joins them is only as
trustworthy as its weakest step. We built a transparent, provenance-preserving
chain for one quantity — striatal D2 occupancy from an oral dose — and tested it
only against numbers that were not used to fit it.

### Methods

A two-compartment population pharmacokinetic model converts oral dose to plasma
concentration; a competitive-binding hyperbola converts plasma concentration to
striatal D2 occupancy under a drug-specific in-vivo EC50. Every parameter is
literature-sourced with its species, route and provenance recorded, or the drug
is refused. The chain was validated by patient-disjoint leave-one-study-out on
haloperidol, by leave-one-drug-out held-out PET, against FDA approved-label
dose ranges transcribed only after the predictions were committed, and against a
clinically-effective plasma range for clozapine. The off-target serotonin–dopamine
hypothesis for extrapyramidal risk was tested by pre-registered regression.

### Results

For haloperidol, plasma→occupancy predicts a patient-disjoint held-out set to a
mean absolute error of 0.0351 occupancy units. The whole oral chain places
haloperidol's and olanzapine's clinically-used dose windows within the FDA label
ranges (2 of 2 scored pass). Aripiprazole plasma→occupancy holds out at MAE
0.0244. Against these successes, converting to a free-plasma basis over-predicts
systematically (MAE 0.0620), externally published EC50s disagree with the fitted
one (MAE 0.0939), the FDA-parameter half-lives fail for 8 of 13 sets, two
active-metabolite predictions miss by 1.7× and 2.5×, a single ziprasidone point
misses by 0.42, and clozapine's modelled 65%-occupancy concentration sits 2.94×
above the top of its effective plasma range — reproducing a real clinical anomaly
and falsifying occupancy-threshold-as-sole-mediator for that drug. The
serotonin–dopamine fingerprint trends in the hypothesised direction but is
indeterminate at seven drugs (p = 0.133).

### Conclusions

A single receptor-occupancy quantity can be predicted from an oral dose with a
held-out error of a few occupancy units, but only for the two drugs whose every
layer is sourced, and only within a tracer- and region-provenance floor that
exceeds some of the held-out errors. The chain predicts no behaviour, simulates
no circuit or network, and refuses every drug and route it cannot source. Its
documented failures are reported at the size they came out and are part of the
contribution.

### Keywords

Dopamine D2 occupancy; antipsychotics; population pharmacokinetics; positron
emission tomography; held-out validation; pre-registration; receptor occupancy;
provenance.

## 1. Introduction

### 1.1 The chain and its weakest links

The received account of antipsychotic action runs from an oral dose, through a
plasma concentration, to dopamine D2 receptor occupancy in the striatum, to a
clinical effect. Each arrow is a different kind of claim. Dose-to-plasma is a
pharmacokinetic model fitted in hundreds of subjects; plasma-to-occupancy is a
binding relation fitted in a few dozen; occupancy-to-effect is a population
association that, as we confirm below, no single occupancy threshold explains.

The purpose of this work is not to assert the whole chain but to measure how far
down it the evidence actually reaches, and to refuse to go further. We therefore
built the chain so that it predicts exactly one quantity — striatal D2 occupancy
as a function of time after an oral dose — and so that it declines to run for any
drug, route, region or body weight whose parameters are not sourced
[ledger: Public prediction interface].

### 1.2 What this manuscript does not claim

The chain makes no behavioural or clinical prediction: its output is an
occupancy, and no test anywhere in the project compares a predicted behaviour to
a measured one [ledger: Any behavioural endpoint]. It performs
no whole-brain, circuit or network simulation: every circuit-layer parameter is illustrative
and those layers are frozen for this release [ledger: Any circuit-layer parameter].
The mu-opioid and GABA-A chains are scaffolding with no sourced parameters
[ledger: Mu-opioid and GABA-A chains]. Saying that a dose reaches a given
occupancy is a statement about a number, not a prediction of a symptom; the
distinction is enforced in the code and carried on every output
[ledger: Occupancy → clinical outcome].

### 1.3 Contributions

1. A provenance-preserving oral-dose-to-striatal-D2-occupancy chain for
   haloperidol and olanzapine, the two drugs that clear every layer
   [ledger: Public prediction interface].
2. Held-out validations on four independent surfaces — patient-disjoint PET,
   leave-one-drug-out PET, FDA label dose ranges, and a clinical plasma range —
   with every success and every failure reported [V1] [V8] [V9] [V11].
3. A pre-registered test of the serotonin–dopamine hypothesis for extrapyramidal
   risk, reported as indeterminate rather than as its suggestive trend [V12].
4. An explicit, test-enforced boundary: refusals naming the missing layer for
   every drug the chain cannot source [ledger: Multi-drug chain, layer by layer (P22.5)].

## 2. Materials and methods

### 2.1 Provenance discipline

Every numeric parameter is either literature-sourced with author, year, species
and route recorded, or labelled illustrative, or the drug is refused before a
number is read. No parameter was invented, no tolerance was widened to pass a
test, and no other drug's, route's or species' measured value was substituted for
the one being modelled. Where a value was sought and not found, that is recorded
as a searched-and-empty result rather than filled.

### 2.2 Dose to plasma concentration

Oral dose is converted to plasma concentration by a two-compartment population
pharmacokinetic model. Haloperidol uses Pilla Reddy 2013 (PMID 24113674),
fitted in 122 subjects (20 healthy volunteers, 102 patients with schizophrenia),
oral 1–60 mg/day [ledger: Populations behind the two drivable chains]. Olanzapine
uses the FDA population-PK model in the LYBALVI review (NDA 213378), of which
exactly one of the 23 transcribed FDA parameter rows meets the chain's criteria
and is bridged into it [ledger: Multi-drug chain, layer by layer (P22.5)]. The
model refuses any body weight other than the 70 kg reference, because no sourced
set prints volume or inter-compartmental-clearance allometric exponents
[ledger: Volume and Q allometric exponents].

### 2.3 Plasma concentration to occupancy

Occupancy follows a competitive-binding hyperbola, `occupancy = Emax · C /
(C + EC50)`, under a drug-specific in-vivo EC50. For haloperidol the fitted
striatal constant is EC50 = 1.8849 nM [V1]; for olanzapine it is de Greef 2011's
pooled [11C]raclopride fit, 5.29 ng/mL with a fitted Emax of 0.875 [V8]. The
occupancy follows plasma instantaneously: the chain has no effect compartment,
so any lag, hysteresis or slow brain off-rate is absent [ledger: Effect-site kinetics (ke0)].

### 2.4 Refusal as a design feature

`predict(drug, dose_mg, regimen)` resolves the PK set, the EC50, and the brain
partition record by drug key and refuses, naming every missing layer, if any is
absent. It never falls back to another drug's parameters. Risperidone and
aripiprazole are refused on principle, not merely unfitted, because the
parent-only concentration path is ill-posed for them [ledger: Risperidone or aripiprazole → D2 occupancy].
Quetiapine is refused because its trough occupancy collapses within a dosing
interval, outside the form of a direct-effect model [ledger: Quetiapine → D2 occupancy].

### 2.5 Pre-registration

The Phase 22 metrics, denominators, tie-breaks and pass criteria were fixed in
`docs/p22-preregistration.md` before any comparison was computed, and a
negative result was committed to in advance [V8]. The label dose ranges were
transcribed only after the predicted dose windows were committed [V8]. The
serotonin–dopamine regression's denominator, model, criterion and
indeterminate-reporting rule were likewise fixed in advance [V12].

## 3. Results

### 3.1 Haloperidol plasma→occupancy holds out narrowly

On a patient-disjoint leave-one-study-out split of the Nyberg and Regenthal depot
PET series, the plasma→occupancy hyperbola with EC50 1.8849 nM predicts held-out
striatal D2 occupancy to a mean absolute error of 0.0351 occupancy units [V1].
This is the chain's single clearest validated claim, and it is narrow: the four
fitted points are group means from two studies of eight patients each, measured
at weeks 1 and 4, so they are not four independent observations, and the EC50 was
fitted on depot intramuscular steady state while the chain is driven orally [V1].

### 3.2 The whole oral chain lands at clinically-used doses

The once-daily oral dose window predicted to give 0.65–0.78 steady-state mean
striatal occupancy is haloperidol 2.89–5.56 mg/day and
olanzapine 5.77–16.4 mg/day [V8]. Scored against FDA approved-label adult oral dose ranges transcribed
only after the predictions were committed, both pass the pre-registered rule
(haloperidol vs label 1.0–15.0, e = +0.0149; olanzapine vs label 10.0–20.0,
e = −0.1625); 2 of 2 scored drugs pass and 6 are refused [V8]. The test is
weakly discriminating: because dose scales exactly with EC50 in this chain,
olanzapine would pass for any EC50 within 0.73–2.9× of its value and haloperidol
within 0.48–1.93×, no wider than the tracer floor [V8].

The one whole-chain occupancy comparison, haloperidol 2 mg/day against Kapur 1996's
reported range 0.53–0.74, gives a single-dose peak of 0.5253 that misses the
bottom of the range by 0.0047, and a regimen-matched steady-state mean of 0.5635
that is contained [V4]. This comparison is weak and not blind: the 2 mg peak was
published before the test ran [V4].

### 3.3 A second drug holds out in the concentration domain

Leave-one-drug-out held-out PET, pre-registered and run with the drug's own EC50
never fitted in this repository, gives for aripiprazole plasma→occupancy a mean
absolute error of 0.0244 over six region-observations in three independent dose
groups, passing the pre-registered ≤0.10 criterion [V9]. Aripiprazole is validated
on this axis but is not drivable by the chain, because it has no sourced oral PK
[V9].

### 3.4 The failures, reported at their size

The failures are results and are reported without softening.

- **Free-plasma basis (V2).** Converting to the free-plasma basis with fraction
  0.09 and testing on Videbaek's IV free-plasma subjects over-predicts
  systematically: MAE 0.0620, bias +0.0601, one-signed [V2].
- **Published EC50s (V3).** Kapur 1997's (1.3569 nM) and Fitzgerald 2000's
  (1.0642 nM) externally published EC50s over-predict all eight held-out points:
  MAE 0.0939, every error positive; the disagreement is unresolved because the
  per-subject data are unreachable [V3] [ledger: Per-subject rows behind Kapur 1997 / Fitzgerald 2000].
- **FDA-parameter half-lives (V5).** The terminal half-life implied by 13 FDA
  population-PK parameter sets matches the approved-label half-life within twofold
  for only 5 of 13; every immediate-release set passes and every failure but one
  is a depot, where the slowest eigenvalue is a release constant, not elimination
  [V5].
- **Active metabolites (V6, V7).** The risperidone active-moiety ED50 predicted
  from paliperidone under the label's equipotency assertion is 6.523 ng/mL against
  a measured 11.06 ng/mL, a fold error of 1.696 [V6]. The dehydro-aripiprazole:aripiprazole
  exposure ratio derived from two EC50 bases is 1.00 against a label value of
  0.40, a fold error of 2.5, which is why aripiprazole remains refused for the
  oral chain [V7].
- **Ziprasidone (V10).** A single linkable held-out point gives an error of
  −0.424; it is anecdotal (n = 1) and implies an EC50 roughly 6× below the fitted
  one [V10].

### 3.5 Clozapine: the anomaly reproduced

The pre-registered clozapine falsification compares the chain's 65%-striatal-occupancy
plasma concentration to the clinically-effective plasma range nobody in this
repository chose. Reaching 65% occupancy needs 1764 ng/mL, 2.94× above the top of
the AGNP therapeutic reference range (350–600 ng/mL); across the effective range
the chain predicts only 27–39% occupancy [V11]. The untuned chain therefore puts
clozapine's 65% concentration about threefold above where clozapine is clinically
effective, reproducing the real anomaly and falsifying
occupancy-threshold-as-sole-mediator for clozapine [V11].

### 3.6 The serotonin–dopamine fingerprint: indeterminate

The project's founding observation is that D2 occupancy alone does not determine
extrapyramidal risk — Siafis 2023 reports haloperidol and risperidone at about
the same 87% occupancy with different EPS odds ratios
[ledger: Occupancy → clinical outcome]. We tested the Meltzer hypothesis that the
5-HT2A:D2 occupancy ratio explains the residual, by a pre-registered nested
regression over the seven drugs that have both a Lako 2013 dose→D2 formula and a
Siafis EPS odds ratio. Adding the fingerprint moves R² from 0.314 to 0.636 with
the ratio coefficient negative (−0.396, the hypothesised direction), but the
nested F(1, 4) = 3.54 gives p = 0.133, and power to detect even a large increment
is 0.20 [V12]. By the pre-registered rule this is **indeterminate**: the data
confirm nothing and reject nothing, and about thirteen drugs would be needed at
the observed effect size, more than the eight any dose→D2 formula covers [V12].
The 5-HT2A side is additionally the weaker, Ki-derived quantity, carrying an error
floor of up to 0.368 occupancy units [V12].

## 4. Discussion

### 4.1 What can be claimed

For two drugs, an oral dose can be turned into a striatal D2 occupancy-time curve
whose held-out error on the plasma→occupancy step is a few occupancy units [V1],
and whose untuned dose windows land inside the clinically-used ranges [V8]. A
third drug's occupancy step holds out on independent PET [V9]. These are real,
if narrow, successes for a fully provenance-tracked chain.

### 4.2 What cannot

Every validated claim is bounded by a provenance floor larger than some of the
errors themselves: the fitted points carry no printed region and, for haloperidol,
no printed tracer, and tracer spread across the registered PET set sets a 2–2.5×
floor [V1]. The chain's largest unrepaired mismatch is that its EC50 was fitted on
depot intramuscular steady state and is driven orally — it cannot even simulate
the route its own constant was measured on [V1]. The dose-to-plasma arm has never
been tested against a measured occupancy in the same subjects for either drug
[ledger: Oral dose → plasma concentration]. The occupancy→outcome step is
falsified as a sole determinant [V11] [ledger: Occupancy → clinical outcome], and
the one mechanistic hypothesis that might rescue it is underpowered at seven drugs
[V12]. Populations behind the PK, PET and threshold observations share no subjects
and their exchangeability is assumed, not tested
[ledger: Populations behind the two drivable chains].

### 4.3 The refused drugs are the research agenda

Of nine census drugs, two are drivable end-to-end and seven stop at a missing
layer [ledger: Multi-drug chain, layer by layer (P22.5)]. The single repair that
could add drivable drugs without new measurement is a journal-literature search
for oral adult population PK for the pk-refused drugs, which has never been done —
Phase 21 searched FDA review documents only
[ledger: Oral adult population PK for clozapine]. Closing the occupancy→outcome
question needs either more drugs with a dose→D2 formula or an in-vivo
dose→5-HT2A-occupancy model, both searched-and-empty
[ledger: A plasma EC50 for quetiapine D2 occupancy].

## 5. Limitations

The chain predicts one quantity, striatal D2 occupancy, for two drugs, by the
oral route, at the 70 kg reference weight, as a typical-value prediction with no
interval [ledger: Between-subject variability and covariates]. It has no effect
compartment [ledger: Effect-site kinetics (ke0)] and no chronic pharmacology, and
the haloperidol fit's week-4 residuals show where a chronic term would go
[ledger: Chronic pharmacology]. For olanzapine the absorption lag is dropped,
which leaves steady-state statistics unchanged but shifts a single-dose curve
[ledger: Olanzapine single-dose time course]. No held-out occupancy series exists
for olanzapine, so its EC50 has only ever been checked through the dose domain
[ledger: Held-out PET observations]. The reduced-haloperidol active metabolite is
bounded at a 0.0030 worst-case occupancy error, but on a rat affinity, the only
one that exists [ledger: Reduced haloperidol's contribution to V1's occupancy]
[ledger: A human D2 affinity for reduced haloperidol]. Thresholds are a
haloperidol relation (Kapur 2000) applied to olanzapine as a labelled cross-drug
transfer [ledger: Public prediction interface]. The aripiprazole partial-agonist
transducer is refused because no single in-vivo human intrinsic-activity value
exists [ledger: A single in-vivo human D2 intrinsic-activity value for aripiprazole].

## 6. Reproducibility and data availability

Every headline number in this manuscript is recomputed by the code and pinned by
`tests/test_manuscript.py`: the haloperidol single-dose peak (0.5253) and
steady-state mean (0.5635) from `predict` [V4]; the dose windows (2.89–5.56 and
5.77–16.4 mg/day) from `predict_window` [V8]; the EC50 (1.8849 nM) and olanzapine
Emax (0.875, de Greef 2011 5.29 ng/mL) from the EC50 registry [V1] [V8]; and the
clozapine, held-out-PET and fingerprint figures from their modules [V9] [V11] [V12].
The supported and refused drug lists are checked against the prediction interface.
The full source-level rationale and citations are in
[`architecture.md`](architecture.md); the pre-registration is in
[`p22-preregistration.md`](p22-preregistration.md); the fingerprint write-up is in
[`p23-4-fingerprint-result.md`](p23-4-fingerprint-result.md); the deliverable's
scope and intended use are in [`model-card.md`](model-card.md).

## 7. References

1. Pilla Reddy V, et al. Population pharmacokinetics of haloperidol. 2013. PMID 24113674.
2. Nyberg S, et al. D2 occupancy from depot haloperidol PET. (ledger V1 source.)
3. Regenthal R, et al. D2 occupancy from depot haloperidol PET. (ledger V1 source.)
4. Kapur S, et al. Haloperidol dose, plasma and striatal D2 occupancy. 1996. (ledger V4 source.)
5. Kapur S, et al. 1997; Fitzgerald P, et al. 2000. Published haloperidol D2 EC50 values. (ledger V3 sources.)
6. Videbaek C, et al. IV haloperidol free-plasma occupancy. (ledger V2 source.)
7. de Greef R, et al. Olanzapine D2 occupancy. 2011. PMC3032087. AAPS J 13(1):107-11.
8. Remington G, et al. Risperidone active-moiety D2 ED50. 2006. PMID 16513859.
9. Grunder G, et al. Aripiprazole D2 occupancy, parent and moiety bases. 2008. PMID 18381901. Am J Psychiatry 165:988-95.
10. Kim E, et al. Aripiprazole plasma D2 occupancy. 2012. (ledger V9/V7 source.)
11. Ito H, et al. Aripiprazole single-dose D2 occupancy. 2012. PMC3460902. (ledger V9 held-out source.)
12. Suzuki T, et al. Ziprasidone D2 occupancy. 2013. PMID 23417515. PMC3679209.
13. Bench CJ, et al. Ziprasidone D2 occupancy. 1993/1996. (ledger V10 source.)
14. Grunder G, et al. Clozapine putamen D2 EC50. 2006. (ledger V11 source.)
15. Hiemke C, et al. AGNP consensus therapeutic reference ranges. 2018. PMID 28910830.
16. Siafis S, et al. Antipsychotic dose, occupancy and adverse effects. 2023. PMID 37537284. PMC10618092.
17. Lako IM, et al. Dose–D2-occupancy formulas. 2013. PMID 23948784.
18. Leucht S, et al. Antipsychotic dose–response and ED95. 2020. PMID 31838873.
19. Kapur S, et al. Relationship between D2 occupancy, clinical response and side effects. 2000.
