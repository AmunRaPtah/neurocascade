# P23.4: does the binding fingerprint explain what occupancy cannot — the result

This is the plain-language report of Phase 23's thesis test, written up to be
read on its own and to be lifted into the manuscript. Phase 23 is the task the
whole human track was built to reach, so the result is stated here without
hedging and without dressing a trend as a finding.

**One sentence: the result is indeterminate.** The serotonin–dopamine fingerprint
trends in the hypothesised direction but seven drugs cannot settle whether it
explains the extrapyramidal residual that D2 occupancy leaves, and the write-up
says below what number could.

The computation lives in
`src/neurocascade/human/fingerprint_eps_regression.py`; it was pre-registered in
`docs/p22-preregistration.md` (P23.3 section) and its reasoning is in
`docs/architecture.md` (Phase 23 part three). This document adds no number of its
own: every figure here is reproduced live by the module and checked against
ledger row **V12** by `tests/test_fingerprint_result.py`.

## The question, and the falsification it starts from

The project's founding observation is a falsification nobody in this repository
chose. Siafis et al. 2023 (PMC10618092, *Mol Psychiatry*; N = 37,193 across
dose-finding trials) reports haloperidol at 11 mg/day with an extrapyramidal
(EPS) odds ratio of ≈4.7 against risperidone at 16 mg/day with ≈3.3, while both
drugs sit at approximately the same ~87% striatal D2 occupancy. **Occupancy alone
does not determine the outcome.** That is already recorded as falsified-as-a-sole-
determinant in the ledger (the occupancy→clinical-outcome row, and V11 for
clozapine).

The widely held explanation — Meltzer's serotonin–dopamine hypothesis — is that a
drug's *off-target* profile, specifically its 5-HT2A affinity relative to D2,
buys EPS tolerability: the more 5-HT2A-selective the drug, the lower its EPS for a
given D2 occupancy. Phase 23 asks the quantitative version of that claim: **does
the 5-HT2A:D2 occupancy ratio explain variance in EPS risk beyond D2 occupancy
alone?**

## The dataset is forced, not chosen: n = 7

The denominator is the intersection of two pre-existing published sources, not a
selection made here:

- Siafis 2023 built its occupancy–response curves with the Lako et al. 2013
  (PMID 23948784) dose→D2-occupancy formulas, which exist for exactly **eight**
  drugs: amisulpride, aripiprazole, clozapine, haloperidol, olanzapine,
  quetiapine, risperidone, ziprasidone.
- **Clozapine drops** on two independent facts that predate this task: Leucht
  et al. 2020 (PMID 31838873) published no ED95 for it (too few fixed-dose
  studies), and Siafis 2023 reported "no usable data" for its EPS odds ratio.

That leaves **n = 7**. The "roughly eight drugs" the task anticipated is this
intersection, and the low power is a property of it, fixed before any number was
computed.

## The asymmetry at the heart of the test: strong D2, weak 5-HT2A

The two halves of the predictor ratio are not on the same footing, and saying so
is the honest shape of the result.

- **D2 occupancy at ED95** is the *strong* quantity: the Lako 2013 hyperbola
  `OccMax · dose / (dose + OccD50)` — PET/SPECT-fitted, the same quantity Siafis
  itself used — evaluated at each drug's Leucht 2020 ED95.
- **5-HT2A occupancy at ED95** has **no in-vivo equivalent**. There is no
  meta-analytic dose→5-HT2A-occupancy model for these drugs, only scattered
  single-dose PET at non-matched doses (recorded searched-and-empty in the
  pre-registration). So it is built the way P23.2 built every off-target
  occupancy: one plasma concentration drives every receptor, and the in-vitro
  5-HT2A:D2 affinity ratio (PDSP human-median Ki, full download 2026-10-01) is
  transferred onto the D2 dose-response — `OccD50_5HT2A = OccD50_D2 ·
  (Ki_5HT2A/Ki_D2)`, `OccMax_5HT2A = 100%` assumed.

That makes the 5-HT2A side the **weaker, Ki-derived quantity** P23.2 measured at a
0.368-occupancy-unit error floor, plus the added assumption that an in-vitro
affinity ratio transfers to an in-vivo dose–ED50 ratio. The module carries a
Ki-derived label on every 5-HT2A row so the quantity cannot be read without its
warning.

## The per-drug inputs

Occupancy fractions at ED95; outcome is the Siafis per-drug **maximum** EPS odds
ratio (the headline reported number, at each drug's top studied dose). The
regression is on `ln(OR)` because an OR is ratio-scale.

| drug | D2 occ | 5-HT2A occ (Ki-derived) | 5-HT2A:D2 ratio | EPS OR (max) |
|---|---|---|---|---|
| haloperidol | 0.833 | 0.168 | 0.201 | 7.56 |
| amisulpride | 0.677 | 0.003 | 0.004 | 5.79 |
| risperidone | 0.790 | 0.975 | 1.234 | 3.30 |
| ziprasidone | 0.677 | 0.979 | 1.446 | 3.06 |
| olanzapine | 0.677 | 0.937 | 1.384 | 2.73 |
| aripiprazole | 0.851 | 0.893 | 1.050 | 1.57 |
| quetiapine | 0.284 | 0.787 | 2.775 | 1.11 |

The anchor is not perfectly matched: occupancy is at ED95 while the OR is each
drug's maximum (at its top studied dose). That mismatch is uniform across drugs,
pre-declared, and not a per-drug knob; it was chosen over "OR at ED95" because the
latter is reachable only by digitising per-drug spline figures, a worse degree of
freedom. At n = 7 it does not change the verdict.

## The result: indeterminate

- **Model 0** (`ln OR ~ D2 occupancy`): R² = 0.314.
- **Model 1** (`+ log10(5-HT2A:D2 occupancy ratio)`): R² = 0.636, and the ratio
  coefficient is **negative, −0.396** — the Meltzer direction: more
  5-HT2A-selective → lower EPS.
- **Nested F-test**: F(1, 4) = 3.54, **p = 0.133**. Not significant at α = 0.05.

The increment is, as a point estimate, a very large effect (Cohen f² = 0.88) —
the fingerprint roughly doubles the explained variance. But the test could not
have reliably found even a conventionally large effect: with n = 7 and df = (1, 4),
the power to detect f² = 0.35 is **0.20**, and ~0.42 at the observed effect size.

By the pre-registered decision rule — not significant, and power to detect a large
effect below 0.80 — this is **INDETERMINATE**: the trend is consistent with the
hypothesis, and the data confirm nothing and reject nothing. Per the task's
explicit instruction, it is reported as a trend, not read as a finding.

## What number could settle it

The honest finding is that **seven drugs cannot settle this question.** To reach
80% power:

- at the **observed** effect size (f² = 0.88), you would need **n ≈ 13** drugs;
- at a merely conventionally **large** effect (f² = 0.35), **n ≈ 26**.

Either exceeds the eight drugs that have a Lako-style dose→D2-occupancy formula,
and no further such formula exists. So the number that would settle it is not
reachable by adding drugs to the present design. Two things would change that,
and both are themselves data this repository has searched for and not found:

1. an **in-vivo dose→5-HT2A-occupancy model** (a Lako-equivalent) for these
   drugs, which would put the 5-HT2A side on the same footing as D2 instead of
   leaving it Ki-derived — searched-and-empty (pre-registration, and P23.2's
   haloperidol cortical-5-HT2 search);
2. dose→D2-occupancy formulas for **more drugs** that also carry a Siafis EPS OR,
   which would grow n past 13 — none exists beyond the Lako eight.

Until one of those arrives, the correct statement is that the eight-drug design is
structurally underpowered, by the margin quantified above, to decide the Meltzer
hypothesis.

## Sensitivity: the verdict moves on single drugs

Leave-one-drug-out (pre-declared, reported not scored) confirms the reading. The
fingerprint sign is **stably negative across all seven deletions**, but
significance is fragile and never crosses 0.05:

- dropping aripiprazole gets closest (p = 0.059);
- dropping quetiapine collapses the D2-only R² from 0.31 to **0.01** — because
  quetiapine sits at only 28% D2 at its ED95, far below the EPS threshold, and
  is doing most of the work in the D2-only model.

A result that moves this much on one drug is itself an indeterminate result. Two
structural confounds are named, not trimmed: **aripiprazole** is a D2 *partial
agonist* (P21.5 refuses its transducer; Siafis notes partial agonists carry small
ORs even at high occupancy), and **quetiapine** sits below the occupancy range
where EPS appears at all.

## What this does and does not claim

This is a comparison of a predicted quantity (per-drug D2 and Ki-derived 5-HT2A
occupancy, and their ratio) against a measured clinical quantity nobody in this
repository chose (the Siafis 2023 EPS odds ratios, N = 37,193). It produced a
definite, pre-registered outcome: **indeterminate, underpowered, with the power
stated.** That closes Phase 23's thesis test and is recorded as ledger row V12.

It is **not** a validated claim that the fingerprint explains EPS, and it is not a
refutation of that hypothesis. The model is **not** claimed to predict behaviour,
EPS incidence, or any clinical endpoint for any drug beyond this single
regression against published odds ratios; no behavioural endpoint anywhere in this
repository has been compared to a measured behaviour (ledger, untested table). The
founding falsification stands — D2 occupancy alone does not determine EPS — and
the proposed mechanism for the residual remains, on this evidence, neither
confirmed nor ruled out.
