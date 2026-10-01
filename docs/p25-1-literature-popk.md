# P25.1: adult oral population PK for six antipsychotics, from the journal literature

P22.5 found that seven of the nine census drugs stall at the PK layer, and that
PK had only been searched for in FDA review documents (P21.3). This task searched
the journal literature (Europe PMC, PubMed, PMC full text) for published adult
population-PK models of oral clozapine, quetiapine, ziprasidone, aripiprazole,
risperidone and paliperidone ER.

**Result: eight models transcribed for five drugs. Ziprasidone is searched and
empty. Nine implied terminal half-lives were compared against the approved US
label, and 6 of 9 fall within twofold.** The three failures are all models fitted
to sparse trough TDM data, and in each case the culprit is a poorly identified
volume. The records, search log and comparison live in
`src/neurocascade/human/literature_popk.py`. The ledger row is V13.

## What was searched, per drug

Every drug got the same five queries, run on 2026-10-01; the hit counts are in
`DrugSearch.queries`. Ziprasidone and paliperidone got extra queries. Every
candidate model the queries surfaced is recorded as transcribed or refused with
one of six barriers: `paywalled`, `abstract-only`, `conference-abstract-only`,
`summariser-only`, `out-of-scope`, `not-transcribed`.

| drug | transcribed | outcome |
|---|---|---|
| clozapine | Li 2012 (PMID 22820910), Reeves 2024 (36793249) | 2 models, 12 refusals |
| quetiapine | Chen 2024 (39635430), Zheng 2024 (39698209) | 2 models; both borrow ka from the paywalled Zhou 2015 |
| ziprasidone | none | **searched and empty** |
| aripiprazole | Koue 2007 (17978491), Knights 2015 (25821065) | 2 models |
| risperidone | Størset 2024 (38963454) | 1 model; every adult oral model co-fits 9-OH-risperidone |
| paliperidone ER | Korell 2017 (28614899), Online Supplement 2 | 1 model; build set includes adolescents |

### Ziprasidone: why it is empty

The indexed literature has exactly two adult oral population-PK models of
ziprasidone:

- **Wessels 2011, PMID 21209243 (the CATIE cohort).** Unpaywall reports it
  closed with no open-access location, the publisher PDF returns HTTP 403, and
  PubMed and Europe PMC carry no abstract.
- **Bao 2024, PMID 38918237.** Unpaywall reports it closed and the Springer PDF
  URL returns an HTML page. Its abstract prints CL 18.74 L/h and V 110.24 L with
  ka fixed at 0.5 h⁻¹. It does not say how many compartments the model has, what
  the variability is, what form the covariate effects take, or whether CL and V
  are apparent.

Abstract numbers are noted on the candidate and are not transcribed: a record
needs the structure, and an abstract does not give it. No sponsor model was
published in a journal. P21.3 found the FDA route empty for the same drug, so
ziprasidone is now empty in both corpora, for access reasons that would be
cleared by a subscription. That is a different kind of gap from quetiapine's
`not-fitted` in P21.3.

## Rule: a summary is not a transcription

Two open-access candidates were reachable only through a fetch tool that
returns a model-written summary of the page:

- **Kim 2008 (aripiprazole, PMC2675764).** Every raw route failed: Europe PMC
  full text returns HTTP 500, PMC serves a browser check to scripts, and Wiley
  returns 403. The summary gave the final V/F as 192 L on one request and 193 L
  on another.
- **Feng 2008 (risperidone, PMC2661978).** The same raw routes failed in the
  same ways.

A tool that changes a digit between two reads of the same table is not a
source. Both are refused as `summariser-only`, and every value in the registry
was read from a raw copy: Europe PMC XML, a PDF, the PMC HTML page, or a
supplementary control stream.

## Three things the literature needed that the FDA schema did not

1. **Fixed parameters carry their origin.** Five of the eight models fix the
   absorption constant to another study's value, and both quetiapine models fix
   it to Zhou 2015, which is itself paywalled. A fixed value is not an estimate
   from the data at hand.
2. **The variability scale can be `unstated`.** Chen 2024 and Zheng 2024 print
   an ω for CL/F without saying whether it is a variance, an SD or a CV.
3. **Disposition is mapped per entity, and a parallel elimination rate is
   allowed.** Reeves 2024's parent leaves both by CL/F and by a metabolic rate
   constant km. Evaluating the model that way reproduces the paper's own printed
   reference half-life (36.8 h against its 36 h). Without km it would be 50 h.
   That match is the check that km was read correctly.

## The comparison

The quantity compared is the same as P21.3's (ledger V5), for the same reason:
`/F` cancels in V/CL, so an apparent oral set gives a real half-life. The
twofold threshold is reused unchanged. The label half-lives come from openFDA:
- **New this task:** clozapine (NDA 019758, steady-state mean 12 h) and
  quetiapine (NDA 020639, 6 h). Both are innovator labels.
- **Reused from P21.3:** risperidone, aripiprazole and paliperidone.

| drug | source | entity | predicted t½ (h) | label t½ (h) | fold | terminal AUC frac | cmt | |
|---|---|---|---|---|---|---|---|---|
| clozapine | Li 2012 | parent | 16.65 | 12 | 1.39 | 1.000 | 1 | PASS |
| clozapine | Reeves 2024 | parent | 36.80 | 12 | 3.07 | 1.000 | 1 | FAIL |
| quetiapine | Chen 2024 | parent | 14.45 | 6 | 2.41 | 1.000 | 1 | FAIL |
| quetiapine | Zheng 2024 | parent | 4.83 | 6 | 1.24 | 1.000 | 1 | PASS |
| aripiprazole | Koue 2007 | parent | 59.78 | 75 | 1.25 | 0.964 | 2 | PASS |
| aripiprazole | Knights 2015 | parent | 65.35 | 75 | 1.15 | 0.933 | 2 | PASS |
| risperidone | Størset 2024 | parent | 4.56 | 3 | 1.52 | 1.000 | 1 | PASS |
| risperidone | Størset 2024 | metabolite | 8.32 | 21 | 2.52 | 1.000 | 1 | FAIL |
| paliperidone | Korell 2017 | parent | 26.02–32.85 | 23 | 1.43 | 0.789 | 2 | PASS |

Choices fixed before computing, and stated here so they cannot be mistaken for
tuning:

- **Clozapine is compared against the label's steady-state mean, 12 h.** The
  label's range of 4–66 h would contain every prediction and so test nothing.
- **Koue 2007 uses CYP2D6 Group 1** (normal metabolisers). Group 2 gives 75.1 h
  and Group 3 107.8 h, both closer to the label's 75 h, so Group 1 is the harder
  test.
- **Korell 2017 has no reference creatinine clearance.** CrCL enters clearance
  additively and the source prints no typical value. Picking one would invent a
  subject, so the half-life is reported over the source's own printed CrCL
  range, 0 to the 150 mL/min cap. The fold is taken at the worse end, and both
  ends pass.

### What the failures share

All three failing models were fitted to sparse routine-TDM samples drawn near
trough:
- **Reeves 2024:** η-shrinkage on V/F is 87%.
- **Chen 2024:** V/F has SE 33.5% and a bootstrap 90% interval of 1222–5163 L.
- **Størset 2024's metabolite:** one compartment, no between-subject
  variability on volume, samples 9–30 h after dose.

Trough concentrations identify clearance well and volume poorly, and the
half-life is V/CL. So the half-life inherits the volume's weakness while the
clearance is fine.

Two pairs make the point directly:
- **Clozapine.** The two models agree on clearance (21.9 against 20.2 L/h) and
  differ 2.8-fold on volume (526 against 1460 L). One passes and one fails.
- **Quetiapine.** Chen 2024 and Zheng 2024 come from the same group, use the
  same method and borrow the same ka. Their volumes differ 4.6-fold, and their
  verdicts differ.

The dense-sampling models all pass: Koue 2007, healthy subjects sampled to
366 h; Knights 2015, more than 13,500 observations; and Korell 2017. **The
lesson for the chain is that a TDM-derived V/F should not be trusted for a time
course, even when its CL/F is fine for a steady-state mean.**

### How independent this is

- **Knights 2015 is the sponsor's model, from the same clinical programme as
  the aripiprazole label.** Its agreement is the least independent row here.
- **The risperidone, aripiprazole and paliperidone labels are ANDA copies**, as
  P21.3 recorded. The clozapine and quetiapine labels are innovator labels.
- **Label half-lives are rounded** ("about 6 hours"), so folds near 1 are not
  evidence of precision.

## Two corrections to the brief

The brief was right that only FDA documents had been searched. Reading the
sources corrected two expectations it carried (rule 11):

- **"Thyssen/Vermeulen 2010 oral paliperidone" does not exist.** Thyssen 2010
  (PMID 20528007) is an oral *risperidone* model. The original adult oral
  paliperidone ER model, Cirincione 2007, was only ever an ASCPT abstract
  (Clin Pharmacol Ther 81:S19) with no table. The one parameter table in the
  journal literature is Korell 2017's supplement, and its main text calls it "a
  previously developed population PK model" whose source is a J&J internal
  report on children and adolescents.
- **No adult oral risperidone model fits parent alone or the active-moiety
  sum.** All of them co-fit 9-hydroxyrisperidone. P21.4 refuses a parent-only
  path for risperidone, and these models do not change that.

## What this does not do

- **Nothing here is wired into the dose-to-occupancy chain.** The rows that are
  single-entity, mamillary and apparent with a printed reference weight would
  pass `fda_pk_bridge`'s criteria:
  - Chen 2024 and Zheng 2024 (quetiapine);
  - Knights 2015 (aripiprazole, 74.19 kg).

  Neither drug has a usable occupancy layer: quetiapine is outside the
  direct-effect form (P21.2) and aripiprazole is refused at the analyte and
  transducer layers (P21.4, P21.5). Clozapine has an EC50 (V11) and now two
  sourced PK sets, but Li 2012 prints no reference weight and Reeves 2024 needs
  a parallel-elimination slot the chain does not have.
- **Covariate relations are transcribed and not evaluated.** Reeves 2024 prints
  its age effect two incompatible ways: Table 2 labels it "exponential" with
  β = −0.013, while the Results formula is (age−40)^0.013.
