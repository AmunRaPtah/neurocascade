# P25.2: literature population-PK records in the occupancy chain

Design note. It was written and committed before any code. Results are appended
at the end, below the line marked "Result".

## The question

P25.1 transcribed eight adult oral popPK models from the journal literature. It
did not wire any of them into the dose-to-occupancy chain. The chain's PK gate,
`dose_to_occupancy._require_pk_record`, accepts only:

- one shape of record: two compartments, absolute units, a printed reference
  weight;
- from two sources: the hand-transcribed haloperidol row and the FDA rows that
  `fda_pk_bridge` converts.

Widening it is a modelling decision, so the rules come first.

## Two gates, not one

There are two questions, and they are kept apart:

1. **Can a literature record become a `HumanPKRecord` at all?** This is
   decided by the new module `literature_pk_bridge`, which plays the role
   `fda_pk_bridge` plays for FDA rows. It answers per record, with a tuple of
   reasons.
2. **Can the chain integrate a `HumanPKRecord` of that shape?** This is
   decided by `_require_pk_record`. It is widened from "two compartments" to
   "one or two compartments", and nothing else changes in it.

A record has to pass both.

## What the literature bridge accepts

A P25.1 record converts only if every one of these holds:

| rule | why |
|---|---|
| analyte is `parent` (no co-fitted metabolite) | The same rule as `fda_pk_bridge` rule 2. A joint fit's parent parameters were estimated together with the metabolite's, and three of the four co-fits here also need a slot the chain lacks: interconversion, or parallel elimination. Metabolites are handled by the active-metabolite gate (P21.4), not by collapsing a joint fit. |
| no parallel-elimination slot | Reeves 2024's `km`. For one compartment it could be folded into an effective clearance (CL + km·V). That would be a derived clearance no source printed, and the record fails rule 7 anyway. |
| 1 or 2 compartments, CL/V1 (+Q2/V2) mapped for the parent | These are the structures the repository's PK classes integrate and that P25.1's records contain. A three-compartment model is refused because no literature record needs one and no test would cover it. |
| units L and L/h, rate constants 1/h | A per-kg set (Koue 2007) has no reference weight, so placing it at one would mean choosing a subject the source did not define. It is refused, not converted. |
| apparent (/F), F left unset | As for haloperidol and the FDA bridge: F cancels out of an oral prediction, and the EC50 downstream was fitted against oral plasma that already carries the true F. |
| reference weight printed | `HumanPKRecord` refuses absolute values without one, for the reason recorded in P19.2. Li 2012 prints none. |
| first-order absorption with a printed ka; a lag may be dropped; any other input is refused | The PK classes have a first-order depot and nothing else. A lag is dropped and named, exactly as `fda_pk_bridge` rule 4 does. A zero-order release phase (Korell 2017, D1 = 25 h) is not a time shift: it reshapes the whole profile, so it is refused. A **fixed** ka (borrowed from another study) is accepted, and the borrowing is carried into the record's notes. P25.1 recorded the origin of each fixed ka so that a consumer could do exactly this. |
| a reference subject exists | Korell 2017's clearance has an additive creatinine-clearance term and no reference CrCL. Its typical subject is undefined, as P25.1 found when it had to report the half-life as an interval. |
| the implied terminal half-life passed V13 | P25.1's lesson: a sparse-TDM V/F is not trustworthy for a time course. The chain predicts time courses, single-dose peaks included, so a record whose own half-life is more than twofold off its label is refused. This uses V13's comparison and threshold unchanged. A record with no V13 pairing is refused as unchecked. |

### Inter-individual variability and covariates

**Inter-individual variability is not represented.** The chain predicts a
typical subject, as it already does for haloperidol and olanzapine. Variability
estimates stay on the P25.1 record and are named in the converted record's
notes as not carried.

**Covariates are not evaluated.** The prediction is for the source's own
reference subject (Zheng 2024: 70 kg. Knights 2015: 74.19 kg, 32 years, not a
CYP2D6 poor metaboliser, SEXO = 0). Every covariate relation is named in the
notes as dropped.

**The weight-scaling exponent is left unset on the converted record.** P25.1
transcribed the covariate relations as printed text and deliberately did not
evaluate them. Reading 0.75 out of a string here would be evaluating one. So a
literature-sourced record cannot be moved off its reference weight, and the
existing chain refuses that with its existing message.

### Which subject `predict()` uses

**`predict_occupancy` now defaults to the PK source's reference weight, not to
70 kg.** Knights 2015's reference subject weighs 74.19 kg. Predicting it at
70 kg would need a weight rule the converted record no longer carries.
Haloperidol and olanzapine both have a 70 kg reference, so their outputs are
unchanged; a regression test pins this. An explicit `body_weight_kg` behaves
exactly as before.

### Ordering and frozen results

**Literature records come last.** `human_pk_records()` lists hand-transcribed
rows first, then FDA-bridged, then literature-bridged. A drug that already has a
record keeps it as its default. No drug has records from two sources today.

**P22.2 and P22.5 are frozen.** Both are scored or pre-registered results that
recorded each drug's PK-layer status as it stood in P22. Their modules keep
consulting the P22 sources only, through `include_literature=False`. Without
that, quetiapine's recorded refusal in V8 would silently change after the fact.

## What is refused, by record (expected before running the code)

| record | expected | first reason |
|---|---|---|
| clozapine Li 2012 | refused | co-fits norclozapine; no reference weight |
| clozapine Reeves 2024 | refused | co-fit; parallel elimination; V13 fail |
| quetiapine Chen 2024 | refused | V13 fail (2.41-fold) |
| quetiapine Zheng 2024 | **accepted** | one compartment, 70 kg |
| aripiprazole Koue 2007 | refused | per-kg set, no reference weight; lag |
| aripiprazole Knights 2015 | **accepted** | two compartments, 74.19 kg |
| risperidone Størset 2024 | refused | co-fits 9-OH-risperidone |
| paliperidone Korell 2017 | refused | zero-order release; no reference CrCL |

**`predict()` still refuses both accepted drugs, and this is expected.** Quetiapine
has no registered D2 EC50: P21.2 found it outside the direct-effect form.
Aripiprazole is refused at the active-metabolite layer (P21.4). What changes is
that `pk` disappears from both drugs' refusal lists, so the missing layers named
are the real ones.

## The external check this enables

The bridge's rules only decide whether a record may enter the chain. The thing
that can be compared against data is the absorption phase, which P25.1 never
tested: V13 compared disposition (half-life) alone. For each accepted record, the
chain's own PK layer is integrated after one oral dose at the reference subject,
and its predicted time to peak plasma concentration is compared with the
approved innovator label:

- SEROQUEL (NDA 020639, SPL set 0584dda8-bc3c-48fe-1a90-79608f78e8a0, effective
  2025-01-22): "reaching peak plasma concentrations in 1.5 hours".
- ABILIFY (NDA 021436, SPL set c040bd1d-45b7-49f2-93ea-aed7220b30ac, effective
  2025-01-29): "peak plasma concentrations occurring within 3 hours to 5 hours".

Both were read from openFDA `drug/label.json` on 2026-10-01.

**The criterion is fixed here, before computing:** agreement within twofold,
with the fold measured to the nearer end of a label range and taken as 1 inside
it. This is V5's and V13's threshold and range convention, reused unchanged.

**Disclosure:** while writing this note I estimated Zheng 2024's one-compartment
Tmax in my head (about 1.8 h). Because the criterion is reused rather than chosen,
that estimate gave me no freedom over the result.

**Independence caveats:**

- Knights 2015 is the sponsor's model, from the ABILIFY programme.
- Zheng 2024's ka is fixed to Zhou 2015, whose authors include AstraZeneca
  staff. AstraZeneca also holds the SEROQUEL label, so the absorption constant
  and the label value may share underlying data.

---

## Result

**The bridge accepted and refused exactly the records the table above
predicted.** `literature_pk_bridge.bridge_survey()` prints every reason.

**`refusal_layers` changed for two drugs:**

- quetiapine: `occupancy, exposure`;
- aripiprazole: `active-metabolite, occupancy, exposure`.

**Haloperidol and olanzapine `predict()` outputs are unchanged.** Both regimens
were compared against the base commit at a relative tolerance of 1e-12, and in
fact they are bit-identical.

**Tmax (ledger V14):**

| drug | source | cmt | ka fixed | predicted Tmax (h) | label Tmax (h) | fold | |
|---|---|---|---|---|---|---|---|
| quetiapine | Zheng 2024 | 1 | yes | 1.77 | 1.5 | 1.18 | PASS |
| aripiprazole | Knights 2015 | 2 | no | 4.55 | 3–5 | 1.00 | PASS |

Zheng's integrated Tmax agrees with the closed form, ln(ka/k)/(ka−k) = 1.76 h,
to within the 1-minute grid.
