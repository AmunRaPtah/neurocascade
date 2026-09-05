# P19.6: held-out validation against measured occupancy

P19.4 fitted a plasma EC50 for haloperidol at the striatal D2 receptor and P19.5
built a dose-to-occupancy chain on top of it. Neither has been compared to an
occupancy the fit did not already see. This is that comparison.

`neurocascade.human.occupancy_validation` holds four pre-registered splits, a
whole-chain containment test against the one dose-and-occupancy observation in
the record, and a check of whether the clinical occupancy window falls at doses
people are actually given. The headline is one number:

    mean absolute error over 20 held-out occupancy predictions: 0.0662

That is six and a half occupancy points, against a fit whose in-sample residual
RMSE is 0.035. Two of the four splits are meaningfully worse than that, and this
document reports them at the size they came out.

## What was held out, and when it was decided

The set of registered observations is small enough that a split chosen after
seeing the errors would not be a held-out test at all. Each of the four rules
therefore carries a `forced_by` field naming what fixed it, and each has no free
parameter left to tune:

| split | n | basis | parameter | forced by |
|---|---|---|---|---|
| A: leave-one-out | 4 | total | refitted here on the other three points | exhaustive over the four points; no choice of which to hold out |
| B: leave-one-study-out | 4 | total | refitted here on the other study | exhaustive over the two studies; the only patient-disjoint partition available |
| C: free-basis set | 4 | free | the total-basis fit, converted by the free fraction | P19.4 already refused to pool total and free; the free set is whatever is on that basis |
| D: external parameter | 8 | total | published by other authors, no in-repo fit | P19.3 already recorded which published relations exist; both are used |

`build_folds` runs without touching `fit_plasma_ec50` at all. A test asserts
this by replacing the fitter with something that raises and checking that every
fold of every split still builds. The partition therefore cannot be a function of
the fit. `HeldOutFold.__post_init__` refuses to construct a fold whose test point
appears in its own training set, on the grounds that the error it would report is
an in-sample error wearing a held-out label. `_run_refitting_fold` additionally
checks that the refit consumed exactly as many points as the fold handed it,
because `fit_plasma_ec50` silently drops observations it does not recognise, and
a dropped training point would mean the reported error came from a training set
other than the declared one.

There is no fifth split. `split_rule("E-...")` raises, and says why.

## Split A: leave one of the four points out

Refit on three of the four total-basis points, predict the fourth.

| point | refitted EC50 (nM) | observed | predicted | error | in training-RMSE units |
|---|---|---|---|---|---|
| Nyberg week-1 | 1.9407 | 0.730 | 0.7033 | -0.0267 | -0.69 |
| Nyberg week-4 | 1.7785 | 0.520 | 0.5639 | +0.0439 | +1.25 |
| Regenthal week-1 | 1.8035 | 0.750 | 0.8019 | +0.0519 | +1.73 |
| Regenthal week-4 | 2.0392 | 0.530 | 0.4688 | -0.0612 | -2.13 |

MAE 0.0459, RMSE 0.0476, bias +0.0020, max absolute error 0.0612. All four are
inside the range their own source printed.

The bias is essentially zero, which is what leave-one-out on a one-parameter
model should give: the refits move the constant between 1.78 and 2.04 nM, and
the errors cancel. The last column is the error divided by the training residual
RMSE of that fold's own fit. It is not a standardised error and the module does
not call it one, for the reason given below.

## Split B: leave one study out

Each of the two depot studies contributes two points measured in the same
patients at week 1 and week 4, so split A holds out a point whose patients are
still in the training set. Split B removes that leak by holding out a whole
study.

| point | refitted EC50 (nM) | observed | predicted | error |
|---|---|---|---|---|
| Nyberg week-1 | 1.8171 | 0.730 | 0.7168 | -0.0132 |
| Nyberg week-4 | 1.8171 | 0.520 | 0.5586 | +0.0386 |
| Regenthal week-1 | 1.9403 | 0.750 | 0.7900 | +0.0400 |
| Regenthal week-4 | 1.9403 | 0.530 | 0.4812 | -0.0488 |

MAE 0.0351, RMSE 0.0376, bias +0.0042, max absolute error 0.0488. Per fold:
holding out Nyberg gives MAE 0.0259, holding out Regenthal gives 0.0444.

Split B is the *better* of the two, which is the opposite of what removing a leak
usually does. The explanation is not that the leak was harmless: it is that split
A's refits are fitted on three points and swing further (1.78 to 2.04 nM) than
split B's two-point-per-study fits (1.82 and 1.94 nM), so split A's predictions
move further from each held-out observation. With four points, this difference is
not evidence of anything. Both figures are reported; neither is picked.

## Split C: does the total-basis constant survive conversion to free plasma?

The four Videbæk subjects report free rather than total plasma. P19.4 refused to
pool the two bases, because they sit about elevenfold apart and that gap is
protein binding rather than biology. So this split takes the total-basis fit,
multiplies by the free fraction 0.090 to get 0.1696 nM, and predicts four
subjects that constant was never fitted on.

| subject | observed | predicted | error |
|---|---|---|---|
| Videbæk Table 2 (II) | 0.290 | 0.2862 | -0.0038 |
| Videbæk Table 2 (III) | 0.360 | 0.4395 | +0.0795 |
| Videbæk Table 2 (IV) | 0.250 | 0.2739 | +0.0239 |
| Videbæk Table 2 (V) | 0.160 | 0.3009 | +0.1409 |

**MAE 0.0620, RMSE 0.0818, bias +0.0601, max absolute error 0.1409.** This is one
of the two bad splits, and the bias is nearly the whole of the error: the
conversion over-predicts on three of the four subjects, badly on subject V, where
it predicts 0.30 against a measured 0.16.

The reason is visible in the repository's own numbers. Fitting these four points
directly on a free basis gives 0.2298 nM. The converted constant is 0.1696 nM, a
factor of 1.355 smaller, and a smaller EC50 means a higher predicted occupancy at
every concentration. So the free fraction of 0.090 and the total-basis fit are
not mutually consistent to better than about a third. Either the free fraction is
wrong for these subjects, or the two bases are not measuring the same free
concentration, or the hyperbola is not the right shape across the range. This
split does not distinguish between them; it records that the discrepancy exists
and how large it is.

Split C is not repaired by refitting on the free basis. A refit would produce a
smaller error and no held-out test.

## Split D: predicting with somebody else's constant

P19.3 recorded two published plasma-occupancy relations for haloperidol that were
derived by other authors: Kapur 1997's EC50 of 0.51 ng/mL and Fitzgerald 2000's
of 0.40 ng/mL, which are 1.3569 nM and 1.0642 nM at the haloperidol molecular
weight of 375.86 g/mol. Neither was fitted here. Both are used to predict the same
four depot points.

| point | source EC50 (nM) | observed | predicted | error |
|---|---|---|---|---|
| Nyberg week-1 | 1.3569 (Kapur 1997) | 0.730 | 0.7722 | +0.0422 |
| Nyberg week-4 | 1.3569 | 0.520 | 0.6290 | +0.1090 |
| Regenthal week-1 | 1.3569 | 0.750 | 0.8433 | +0.0933 |
| Regenthal week-4 | 1.3569 | 0.530 | 0.5702 | +0.0402 |
| Nyberg week-1 | 1.0642 (Fitzgerald 2000) | 0.730 | 0.8121 | +0.0821 |
| Nyberg week-4 | 1.0642 | 0.520 | 0.6837 | +0.1637 |
| Regenthal week-1 | 1.0642 | 0.750 | 0.8728 | +0.1228 |
| Regenthal week-4 | 1.0642 | 0.530 | 0.6284 | +0.0984 |

**MAE 0.0939, RMSE 0.1014, bias +0.0939, max absolute error 0.1637.** This is the
worst split in the task. Per fold: Kapur 1997's constant gives MAE 0.0712,
Fitzgerald 2000's gives 0.1167.

Bias equal to MAE means every one of the eight predictions is too high, without
exception. Both published constants are smaller than the 1.8849 nM fitted here,
and a smaller EC50 predicts more occupancy at the same plasma concentration. The
disagreement between published constants and this repository's fit is therefore
one-directional and around 1.4- to 1.8-fold in the constant, which shows up as
four to sixteen occupancy points on the prediction.

The pooled MAE of 0.0939 averages over a *choice of published constant* rather
than over held-out points. The two per-fold rows are the numbers that mean
something.

## Standardised error is reported as not computable, not as zero

The acceptance criterion asks for absolute and standardised error per point and
in aggregate. The absolute errors are above. The standardised errors do not
exist, and the module says so rather than manufacturing them.

`PETOccupancyComparison.standardized_error` returns `None` when the observation
has no `observed_sd`. Not one of the registered observations printed an SD: the
depot abstracts print a range across patients, the Videbæk table prints
per-subject values with no spread, and the published relations print a point
estimate. So `standardized_errors_available` is `False` for all four splits and
every summary line says *not computable, no source printed an SD*.

The `error_in_training_rmse_units` column in split A and B is deliberately not
called a standardised error. It divides by the residual scatter of the fit, which
is a property of this repository's model, not of the measurement. It answers
"is this miss large compared to how well the model fits at all", which is a
different and weaker question than "is this miss large compared to the
measurement noise". For splits C and D there is no in-repo fit at all, so the
column is `None` and the summary prints *no in-repo training residual* rather
than borrowing a residual from another split.

## Containment is a much weaker test than the error

Where a source printed a range, the module also asks whether the prediction falls
inside it. The counts:

| split | points with a range | contained |
|---|---|---|
| A | 4 | 4 |
| B | 4 | 4 |
| C | 0 | 0 |
| D | 8 | 8 |

Split D is contained on all eight points while carrying an MAE of 0.094. The
depot ranges are wide (Regenthal week-4 prints 0.12 to 0.89), so containment
there discriminates almost nothing, and reading the 8-of-8 as a pass would be a
mistake. Split C's zero out of zero is not a pass either: the Videbæk subjects
print no range, so the test could not be run. The signed errors above are the
numbers that carry information.

## The one whole-chain comparison, and it misses

Everything above tests the plasma-to-occupancy hyperbola. Testing the whole chain
needs an observation pairing an oral dose with an occupancy, and
`chain_testable_observations()` returns an empty tuple. Not as a claim in prose:
it is computed, and the summary states it in its first line.

    observations pairing an oral dose with a differenceable occupancy: 0

Every dose-bearing observation in the record fails one of three conditions.
Kapur 1996 has a 2 mg oral dose but reports a range with no central value, so
there is nothing to subtract a prediction from. The four Videbæk subjects are on
an intravenous infusion, which is a route P19.2 recorded no parameters for. The
four Xiberas points report a binding index rather than an occupancy. **The
dose-to-plasma arm of the P19.5 chain is therefore untested by P19.6.**

The one thing Kapur 1996 supports is a containment test, and the chain fails it:

| quantity | value |
|---|---|
| reported range, 2 mg daily, 7 patients | 0.53 to 0.74 |
| chain's single-dose peak at 2 mg | 0.5253 |
| contained | **no**, 0.0047 below the lower bound |

That is recorded as the result. It was not repaired by refitting anything.

The module then computes what the chain predicts for the regimen the patients
were actually on, 2 mg daily to steady state, by superposing the chain's own
single-dose curve. That gives trough 0.4718, mean 0.5635, peak 0.6592, with the
mean inside the reported range. Three caveats travel with that figure on the
result object itself, and they are repeated here because the number is easy to
misread as a second, passing test:

- **It was not blind.** The 2 mg single-dose peak this turns on was already
  published in `docs/p19-5-dose-to-occupancy-chain.md` before P19.6 ran, so the
  comparison was made against a number already in view.
- **The steady-state figure was computed after the single-dose miss was seen.**
  It explains the miss; it is not a second, passing test.
- **The source reports a range across seven patients, not a mean with a
  spread.** Containment says the prediction is somewhere among those patients and
  says nothing about where.

The honest reading is that a single-dose prediction misses a steady-state
observation by about half an occupancy point, and that the direction and size of
the miss are consistent with the accumulation the chain itself predicts. That is
a plausible explanation, not a confirmed one, and one observation cannot make it
more than that.

## The clinical occupancy window

P19.5 asked whether the clinical window appears at doses people are given. It
does, at the top of the ordinary range.

| outcome | occupancy | dose at steady state | single dose |
|---|---|---|---|
| clinical response | 65% | 2.89 mg/day | 3.36 mg |
| hyperprolactinemia | 72% | 4.02 mg/day | 4.65 mg |
| extrapyramidal side effects | 78% | 5.56 mg/day | 6.41 mg |
| extrapyramidal side effects, pooled reanalysis | 77% to 78% | 5.40 mg/day | 6.23 mg |

Response to EPS spans 2.9 to 5.6 mg/day, a factor of 1.92 in dose for 13
occupancy points. These are doses haloperidol is prescribed at, so the chain
reproduces the window rather than landing an order of magnitude away, and the
whole of that claim is the dose range. The errors above say the occupancy at any
given dose is good to roughly five to ten points, and the containment test says
the one whole-chain number available came out 0.005 low.

Two things about the window itself are recorded in
`CLINICAL_WINDOW_PROVENANCE` and should not be dropped when the table is quoted.
The familiar "65 to 80 percent" is a synthesis rather than a primary result; the
80 appears to be a rounding of Kapur 2000's 78, so this module uses 78. And the
primary sources print wider intervals than the synthesis does: Farde 1988 prints
65% to 85% for antipsychotic effect, Farde 1992 prints 70% to 89% for the
patients who developed EPS, and Uchida 2011's pooled reanalysis of 12 studies and
82 subjects gives 60 to 78 percent.

`OCCUPANCY_IS_NOT_THE_OUTCOME` carries the other half. Reaching 78 percent
occupancy is not the same as predicting extrapyramidal side effects: the chain
has no effect compartment and no ke0, the thresholds are group-level statistics
over small samples, the striatal region assignment is inferred rather than
printed by two of the three sources, and Siafis 2023 is cited for the point that
occupancy does not determine the side effect in an individual. The doses in that
table are where the chain crosses an occupancy, and nothing more.

## What P19.6 refuses

Each of these is a test in `tests/test_human_occupancy_validation.py`:

- a fold whose test point is in its own training set
- a fold with no test points
- a refit that consumed a different number of points than the fold declared
- a fifth split named after the fact
- a split rule on a plasma basis that is neither total nor free
- pooling total-basis and free-basis observations
- a standardised error where no source printed an SD
- a training-RMSE ratio where no fit ran in this repository
- a dose, dose count or time step the steady-state construction cannot use
- a target occupancy outside (0, 1), or a regimen the chain has no parameters for

## Integration

`as_heldout_results()` hands all 20 predictions to the P18.2 summariser as
`HeldOutHumanDrugResult` records, all with status `scored`. `summarize_heldout`
independently returns a mean absolute error of 0.0662, which matches the pooled
mean over the four splits exactly, a cross-check by code that knows nothing
about this drug.

The steady-state construction is superposition of the P19.5 single-dose curve on
a 24-hour interval, not a second pharmacokinetic model: it uses the same EC50 and
the same solver, and 40 doses is past convergence (the mean occupancy at 40 and
at 70 doses agree to 1e-10).

Nothing in this module refits anything in response to what it found. Splits C and
D are recorded as the misses they are.
