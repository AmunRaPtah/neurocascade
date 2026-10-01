# P22.1: pre-registration of the Phase 22 validation metrics

Committed before any P22 result exists. Git history is the proof of ordering:
this document's commit precedes every commit that computes, transcribes or
compares a Phase 22 number, and no P22.2, P22.3 or P22.4 result may cite an
analysis that is not either specified here or listed in that result's own
"Deviations from pre-registration" section.

P19.6 recorded that its Kapur 1996 comparison was not blind: the 2 mg
single-dose peak had been published in `docs/p19-5-dose-to-occupancy-chain.md`
before the comparison ran, and the steady-state figure was computed after the
single-dose miss was seen. That caveat is why this document exists and comes
first.

**A negative result will be reported and is publishable.** That is not a
concession, it is the design: `docs/validation-ledger.md` rule 3 states that a
failed comparison earns a row, and rows V2, V3, V5, V6 and V7 are failures that
were reported at the size they came out. Phase 22 continues that. If every drug
misses, the phase's deliverable is the documented miss.

## What this pre-registration can and cannot claim

Pre-registration here buys **procedural** blindness: the metrics, denominators
and pass criteria are fixed before any comparison is computed. It cannot
buy epistemic blindness. The contamination already in view, stated so it cannot
be discovered later and read as concealment:

1. **Haloperidol's predicted window is already published.** P19.5/P19.6 printed
   2.89 mg/day (65%) to 5.56 mg/day (78%), and the P22.2 task brief restates
   it. The haloperidol row of P22.2 is therefore *not blind* and will carry
   that caveat the way V4 and V6 do. It is kept in the denominator because
   dropping the anchor drug after publishing its prediction would be a worse
   distortion than flagging it.
2. **The label dosage text sits on disk.** P20.6 snapshotted all eight usable
   US oral labels to `data/fda/` (SHA-256-pinned in
   `neurocascade.human.fda_sources`) and deliberately did not transcribe dose
   ranges into numbers; but the fetch script has a `--show` mode, the text was
   viewable, and P20.6's ledger row quotes haloperidol's indication wording.
   The claim available is that no label dose range has been transcribed into
   this repository as a number, not that none was ever seen.
3. **Antipsychotic dose ranges are common clinical knowledge**, present in the
   training data of every model that has worked on this repository. No
   procedure can remove that.
4. **The P22.3 source landscape was already surveyed.** P21.2 read the
   per-drug EC50 values that will be the *inputs* of P22.3. The held-out side
   is protected not by ignorance but by the study-disjointness rule below.
5. **The clozapine anomaly is famous and its expected direction is stated in
   the P22.4 brief itself.** What pre-registration adds there is a fixed
   mapping from outcome to reading, so the interpretation cannot bend to the
   number.

## Commitments common to all three tests

- **C1. No post-hoc changes.** Once a comparison for a drug has been seen, no
  parameter, Emax convention, region choice, plasma basis, dosing-frequency
  assumption, threshold or tolerance affecting that drug may be changed. If a
  genuine error in this document is found, the correction is a deviation:
  recorded in the result document's "Deviations from pre-registration"
  section, with the pre-registered analysis still computed and reported
  alongside the corrected one.
- **C2. No silent drops.** Every drug in a test's fixed denominator appears in
  the result as exactly one of: a scored comparison; a named-layer refusal
  (which layer of the chain refused it and why); or a searched-and-empty
  record for the ground-truth side. "Unscored" is never "pass" and never
  disappears from the count.
- **C3. Modeling choices precede comparisons.** The ledger records that wiring
  the P21.3 FDA parameter sets into the chain requires a modeling decision
  (what to do with apparent `CL/F`, `V/F` sets; with parent-plus-metabolite
  fits) that was deliberately not made in P21.3. Any such decision must be
  made, coded, tested and committed *before* the ground-truth number for any
  drug it affects is transcribed or compared, and may not be revisited for
  that drug afterwards.
- **C4. Numbers are regenerated, not copied.** Every reported figure is
  computed by code in this repository at result time (the P22.5/P19.7
  standard), and every transcribed ground-truth value quotes the sentence and
  pins the source (application number and `effective_time` for labels; PMID,
  tracer, region, analyte basis and regimen for PET observations).
- **C5. Conventions.** Occupancy is a fraction in [0, 1]. Doses are total oral
  mg/day in a 70 kg reference subject. Logs are base 10. ng/mL ↔ nM
  conversions use the molecular weight recorded on the drug's registry record.
  Doses are reported to three significant figures and occupancies to three
  decimals; ties and boundary cases are evaluated at that printed precision,
  and a value exactly on a boundary resolves *against* the more interesting
  reading (specified per test below).
- **C6. Ledger rows regardless of direction.** Each of P22.2, P22.3 and P22.4
  adds its row(s) to `docs/validation-ledger.md` whether it passes, fails, or
  comes out empty.

---

## P22.2: label-dose test against FDA approved dose ranges

### Denominator, fixed now

The eight drugs with `status="ok"` in `neurocascade.human.fda_sources.LABELS`
(committed in P20.6, before this document): **haloperidol, risperidone,
paliperidone, olanzapine, quetiapine, ziprasidone, aripiprazole, clozapine.**
Amisulpride is excluded, per the searched-and-empty record in that same module:
it has no US antipsychotic approval, so the US route cannot supply ground
truth. It remains in the P22.3 denominator.

Two exclusions from scoring are predictable now and are pre-declared so they
cannot look like post-hoc trimming:

- **Quetiapine** is expected to be unscorable: P21.2 established that no
  plasma EC50 exists for it anywhere, because the occupancy-plasma fits failed
  in the primaries, and the ledger records the drug as outside the model's
  form (no ke0). If that changes it will be because a sourced constant was
  found, not because one was manufactured.
- **Risperidone** is refused on the parent-only axis by machine
  (`require_parent_only_path`), and the ledger states the active-moiety axis
  is the only well-posed one. If P22.2 wires a moiety-basis chain, both the
  EC50 (Remington 2006, active moiety) and the PK concentration it is applied
  to must be on the moiety basis; a basis mismatch is a refusal under C2, not
  an approximation. The same basis-match requirement applies to every drug.

### Predicted quantity

For each drug the chain can drive end-to-end without an illustrative
parameter and without violating a standing refusal:

- **D65** = the total daily oral dose at which the **mean occupancy over one
  dosing interval at steady state under once-daily dosing** equals 0.65, via
  `dose_reaching_occupancy(0.65, regimen="steady-state-daily", drug=key)`
  (superposition of the chain's own single-dose curve, n_doses=40, 5-minute
  step, 70 kg).
- **D78** = the same at 0.78.
- Predicted window **W = [D65, D78]**.

The targets 0.65 and 0.78 are exactly the sourced clinical-window thresholds
carried in `CLINICAL_WINDOW_PROVENANCE` (response, Kapur 2000; EPS, Kapur
2000 / Farde 1992, with 78 used rather than the synthesized 80). The
steady-state **mean** statistic is fixed here because the thresholds come from
trough-agnostic clinical PET scans at unspecified times post-dose; peak and
trough are reported alongside but do not enter the metric.

Once-daily dosing is fixed for every drug, including drugs whose labels dose
twice or three times daily. This is a known, recorded limitation, uniform
across drugs, not a per-drug knob. No dosing-frequency adjustment may be made
after a comparison is seen (C1).

The drug's registered EC50 is used with **its own Emax convention**: if the
source fitted Emax free, the hyperbola is `Emax · C / (C + EC50)`. If a target
occupancy exceeds the registered Emax, that bound is **unreachable at any
dose**: recorded as such, the drug scores overlap = fail, and its midpoint
error is undefined (reported as undefined, never imputed, never dropped from
the failure count).

### Ground-truth extraction rule, fixed before any label is read as a number

From the pinned snapshot (SHA-256 verified against `fda_sources`) of each
drug's DOSAGE AND ADMINISTRATION section, for **schizophrenia, adults, oral**:

- **L_low** = the lowest dose the label presents as a recommended target or
  effective dose (the lower end of the recommended/target range, or the single
  recommended dose). Initial titration steps that the label instructs to
  increase from are excluded, unless the label states the initial dose is
  itself within the effective range.
- **L_high** = the maximum recommended daily dose. Where acute and maintenance
  carry different caps, the larger label-sanctioned cap is used.
- Where the label gives separate acute and maintenance recommended ranges, the
  label range is their union.
- Units: total mg/day; divided doses are summed.
- Every transcription quotes the sentence(s) verbatim. If two defensible
  readings of one label differ, both are transcribed with their sentences, the
  **narrower** range is used for scoring (harder to overlap, conservative
  against the model), and both are reported.
- A label recommending a single dose yields a point range, L_low = L_high.

### Ordering rule inside P22.2

The predicted windows for all drivable drugs are computed and **committed
before** the commit that transcribes any label range into numbers, so git
history shows prediction-before-ground-truth for every drug except
haloperidol, which is pre-flagged as not blind (its window is already
published *and* its usable dose range was noted in P20.6's ledger row).

### Metrics and pass criteria, fixed now

Per scored drug:

1. **Overlap**: does W intersect [L_low, L_high]? Closed intervals; touching
   at an endpoint counts as overlap (the tie resolves *toward* overlap, which
   is the less interesting reading given the phase is looking for
   discriminating misses).
2. **Log midpoint error**: e = log10( sqrt(D65·D78) / sqrt(L_low·L_high) ),
   geometric midpoints because dose is a ratio-scale quantity. Undefined when
   either window is unreachable/empty.

**A drug passes iff overlap is true AND |e| ≤ log10(2).** The twofold
criterion reuses the precedent fixed in advance for V5, and finer
discrimination is not defensible anyway: the tracer and region provenance on
the EC50 side carries a 2–2.5× floor (P19.7, P21.2 trap 3).

Aggregate reporting: n_pass / n_scored / 8, per-drug signed e, median |e| and
max |e| over scored drugs. **No phase-level binary pass is defined**; the
per-drug table is the deliverable, and by C6 it is reported whole.

---

## P22.3: leave-one-drug-out against published PET occupancy

### Denominator, fixed now

The nine-drug census of P21.2: **haloperidol, risperidone (active moiety),
olanzapine, clozapine, ziprasidone, amisulpride, aripiprazole, paliperidone,
quetiapine.** Pre-declared expectations, so they cannot read as trimming:
quetiapine is expected searched-and-empty (no EC50 exists, structurally);
paliperidone's only EC50 is second-hand (sponsor restatement of Arakawa 2008)
and any paliperidone row carries that provenance class explicitly.

### Design

For drug X: nothing is fitted on X's held-out observations. The prediction is
`occ = Emax_r · C / (C + EC50_r)` at each observed concentration, where
(EC50_r, Emax_r) is X's registered constant from its source study, on its
recorded plasma basis and analyte basis. "Fit nothing on X" here means: the
registered constant comes from a source study, and the test observations must
be disjoint from that study (below); no in-repo refitting on X's data occurs.

A **whole-chain** (dose → occupancy) comparison is additionally run for any
observation that pairs an administered dose with a differenceable occupancy
under a chain-compatible regimen (the `chain_testable_observations` pattern),
reported separately from the concentration-based test.

### Eligibility of a held-out observation, fixed now

An observation enters the test iff all of:

1. human in-vivo PET or SPECT D2/D3 occupancy with a paired plasma or serum
   concentration from the same visit;
2. the concentration is on the **same analyte basis** (parent vs active
   moiety) and **plasma basis** (total vs free) as X's registered EC50:
   basis conversion using a free fraction is not permitted as a rescue (the
   V2 lesson: the one conversion tried failed one-signed);
3. its study population is **disjoint from the study(ies) behind X's
   registered EC50**, and the disjointness is establishable from what was
   read. If it cannot be established (a pooled fit whose study list is
   unreachable, the Uchida 2011 situation), the observation is ineligible.
   Undeterminable independence is treated as dependence.

Haloperidol's version of this test is P19.6 and is not re-scored; haloperidol
enters P22.3 only through observations from studies disjoint from Nyberg 1995
and Regenthal 1997 (and disjoint from the P19.3 external-relation sources
already used in split D).

### Tracer rule, fixed now (carried from P19.6/P21.2)

The tracer is recorded on every observation. Primary metrics are computed
within tracer family (same family as the EC50 source's tracer where possible).
The raclopride / IBZM / FLB 457 (and fallypride) spread is 2–2.5×; pooling
across tracers appears only as a clearly labeled secondary figure, never as
the scored metric. The regimen (single-dose vs steady state) is likewise
recorded per observation; a regimen mismatch with the EC50 source is flagged
on the row and reported, not silently pooled.

### Metrics and pass criteria, fixed now

Per observation: signed error = predicted − observed, in occupancy fraction.
Per drug: MAE, mean signed bias, n, and containment where the source printed a
range (containment is the weak secondary, per P19.6). Standardized errors are
computed only where the source printed an SD, and otherwise reported as *not
computable*, never as zero.

**A drug passes iff its per-drug MAE ≤ 0.10 occupancy units on same-tracer
observations** (≤ 0.15 if only cross-tracer observations exist, and then
labeled cross-tracer). Rationale, fixed in advance: 0.10 is the
"somebody else's constant" precision: P19.6's split D, predicting haloperidol
with externally published constants, gave MAE 0.094, and this test asks
exactly that class of question. No minimum n, but n is printed with every
aggregate and n < 3 is flagged *anecdotal*. A drug with no eligible
observation earns a searched-and-empty ledger row naming what was searched
(C2). A tie (MAE exactly 0.100 at printed precision) resolves to **pass**,
the less interesting reading for a phase hunting for discriminating misses.

---

## P22.4: the clozapine anomaly, as a falsification test

### Inputs, fixed now

- Clozapine's registered EC50 record(s) as they stand when P22.4 runs,
  ingested under P22.2/P22.3's rules before any P22.4 number is computed.
  **Primary = the striatal (putamen) row**: the 0.65/0.78 thresholds are
  striatal measurements, so the striatal constant is the like-for-like choice,
  with every other registered candidate (cortical region rows,
  alternative Emax conventions such as Gründer 2006's fitted-Emax putamen
  950 ng/mL vs Uchida's free-Emax 105 ng/mL with Emax 60%) reported as
  sensitivity, in full, none chosen after the fact.
- Label ground truth: the VERSACLOZ snapshot (NDA203479) already pinned in
  `fda_sources`, transcribed under the P22.2 extraction rule.

### Computed quantities

- **D65** for clozapine, by the identical procedure as P22.2 (steady-state
  daily mean, once-daily, 70 kg, registered Emax convention).
- Predicted occupancy at L_low and at L_high.

### Fixed mapping from outcome to reading

- **Outcome A: anomaly reproduced**: D65 does not exist (registered
  Emax ≤ 0.65) **or** D65 > L_high. Reading: the chain, untuned, places
  clozapine's 65% dose above the doses at which clozapine is clinically
  effective, i.e. it reproduces the real anomaly, and *occupancy-threshold-
  as-sole-mediator is falsified for clozapine*. This is a model success and a
  hypothesis failure, and is reported as both.
- **Outcome B: anomaly absent**: D65 ≤ L_high. Reading: the anomaly does not
  appear through this chain's numbers. That contradicts the received PET
  literature (clozapine effective at 20–67% occupancy; Nordström 1995
  explicitly declined to fit a hyperbola to it) and is reported as an
  unresolved disagreement between the chain and that literature, not as
  a vindication of the occupancy threshold.
- **Tie**: D65 exactly equal to L_high at printed precision resolves to
  **Outcome B**, the reading *against* the more interesting result.

Both outcomes are publishable and neither triggers any adjustment: **nothing
is tuned to make clozapine fit**, in either direction (C1). If the sensitivity
rows straddle the A/B boundary (plausible, since the registered candidates
span 4.6× on the Emax convention alone), the primary (striatal putamen) row
decides the headline and the straddle itself is reported as the finding it is.

### What P22.4 is not

It is not a test of the chain's clozapine PK layer (that is P22.2's business)
and it is not scored with P22.2's overlap metric. It is a directional
falsification test of one hypothesis (occupancy in the 0.65–0.78 window as
the sole mediator of antipsychotic response) using the chain as the
instrument. Its ledger row records which outcome obtained and against which
EC50 record.

---

## P23.3: does the off-target fingerprint explain the EPS residual that D2 occupancy cannot

Added for Phase 23's thesis test. The P22.1 document is the pre-registration
home per the task brief; this section is committed in its own commit, before the
commit that adds the regression code and its result, so git history shows
criterion-before-number the way the P22.2–P22.4 sections do.

### What this section can and cannot claim (contamination inventory)

Procedural blindness only, and **less than P22.2–P22.4 had.** Rule 12 required
the load-bearing assumption (that the inputs for an eight-drug regression exist
and are sourceable) to be checked *before* building on it. Checking it meant
assembling the inputs and running the regression once during feasibility
scoping. So the result was seen before this text was committed: this section is
**not blind**, and is flagged the way the haloperidol row of P22.2 is. Three
further contaminations, stated so they cannot later read as concealment:

1. The Siafis 2023 contrast (haloperidol 11 mg EPS OR 4.7 vs risperidone 16 mg
   OR 3.3 at ~87% D2) and the Meltzer 5-HT2A hypothesis are in the task brief
   itself and throughout P23.1/P23.2 and the ledger. The *direction* of the
   hypothesis was never unknown.
2. The eight Lako-2013 drugs and the Siafis per-drug ORs are already published
   numbers; nothing here is a blind prediction of an unseen quantity.
3. The 5-HT2A occupancy side has no in-vivo occupancy model (there is no
   Lako-equivalent for 5-HT2A, searched-and-empty below), so it is computed from
   in-vitro Ki — the *weaker quantity* P23.2 measured at a 0.368-occupancy-unit
   floor. The predictor inherits that weakness by construction.

What is nonetheless fixed in advance and honored by the code with no post-hoc
tuning: the denominator, the model form, the predictor definitions, the
criterion, the decision rule, and — most importantly — the commitment to report
an indeterminate result as indeterminate rather than reading a trend as a
finding (the task's explicit instruction).

### Hypothesis

- **H1**: across the drugs with both a Lako-2013 dose→D2-occupancy formula and a
  Siafis-2023 EPS odds ratio, the **5-HT2A:D2 occupancy ratio at the drug's ED95
  dose** explains variance in ln(EPS OR) beyond D2 occupancy at ED95 alone, and
  it enters **negatively** (more 5-HT2A-selective → lower EPS; the Meltzer
  mechanism).
- **H0**: the ratio adds no explanatory variance (β on the ratio term = 0).

### Denominator, fixed now

The eight drugs for which Lako et al. 2013 published a dose→D2-occupancy formula
(amisulpride, aripiprazole, clozapine, haloperidol, olanzapine, quetiapine,
risperidone, ziprasidone), **minus clozapine**, which has neither an ED95 in
Leucht 2020 (not estimated — too few fixed-dose studies) nor a usable EPS OR in
Siafis 2023 ("no usable data"). Both are pre-existing facts, not choices made
here. Denominator = **7**: haloperidol, risperidone, olanzapine, quetiapine,
aripiprazole, ziprasidone, amisulpride. No drug may be dropped from the primary
analysis after the result is seen; leave-one-drug-out is a pre-declared
sensitivity, below, reported never scored.

### Model, fixed now

- **Outcome** Y = ln(OR), OR = the Siafis 2023 per-drug **maximum** EPS odds
  ratio vs placebo (the headline reported numbers; transcribed verbatim with
  dose and CI). The anchor mismatch — occupancy evaluated at ED95, OR being the
  maximum (which occurs at each drug's top studied dose) — is a known limitation,
  uniform across drugs, pre-declared and not a per-drug knob.
- **occ_D2** = D2 occupancy fraction at ED95 = OccMax·D/(D+OccD50) with the Lako
  2013 (OccMax, OccD50) for the drug and D = the Leucht 2020 ED95 (mg/day).
- **occ_5HT2A** = 5-HT2A occupancy at ED95, computed on the P23.2 principle that
  one plasma concentration drives every receptor: OccD50_5HT2A =
  OccD50_D2·(Ki_5HT2A/Ki_D2), OccMax_5HT2A = 100% (**stated assumption**), Ki =
  PDSP human-median. This is **Ki-derived and weaker** (P23.2) and additionally
  assumes the in-vitro affinity ratio transfers to the in-vivo dose–ED50 ratio.
- **ratio** = occ_5HT2A / occ_D2; the predictor is log10(ratio).
- **Model 0** (null): Y ~ 1 + occ_D2. **Model 1**: Y ~ 1 + occ_D2 + log10(ratio).

### Criterion and decision rule, fixed now

Nested partial F-test (Model 1 vs Model 0) at α = 0.05.

- **SUPPORTED** iff p < 0.05 **and** β_ratio < 0 (hypothesized sign).
- **INDETERMINATE** iff not supported **and** the achievable power to detect even
  a large increment (Cohen f² ≥ 0.35) is < 0.80 — i.e. the test could not have
  reliably found a real effect, so a non-significant result distinguishes
  neither way.
- **NULL-FAVORED** only if p ≥ 0.05 **and** power ≥ 0.80 (an equivalence-style
  read). **Pre-declared unreachable**: at n = 7, df = (1, 4), power to detect
  f² = 0.35 is ≈ 0.20, so a non-significant result here is INDETERMINATE by
  construction, stated in advance so the negative is not over-read in either
  direction.

### Sensitivity (pre-declared, reported not scored)

Median Ki is primary; the PDSP min/max Ki span and leave-one-drug-out are
reported as robustness, in full. The two structural confounds are named now:
aripiprazole is a D2 **partial agonist** (Siafis found partial agonists carry
small ORs despite high occupancy; P21.5 refuses its transducer), and quetiapine
sits far **below** the EPS occupancy threshold at ED95 (28% D2). A result that
depends on the in/out of either is, itself, an indeterminate result.

### Searched-and-empty, recorded now

An in-vivo 5-HT2A **occupancy** model (a Lako-equivalent dose→5-HT2A-occupancy
function) for these drugs: none exists. 5-HT2A occupancy is reported only in
scattered single-dose PET studies at non-matched doses, so the 5-HT2A side of
the ratio cannot be put on the same in-vivo footing as D2 and is Ki-derived.

---

## What this document does not constrain

Engineering choices with no ground-truth contact (module layout, refusal
mechanics, test structure, how the P21.3 registry is threaded into the chain)
remain free, subject to C3: any choice that affects a predicted number must be
committed before the affected drug's ground truth is transcribed or compared.
P22.5 (failure analysis) is descriptive and needs no pre-registration beyond
C4, which binds it to regenerate every number from code.
