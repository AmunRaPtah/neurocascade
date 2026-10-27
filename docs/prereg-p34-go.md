# Pre-registration: P34.5 GO / NO-GO on a fresh held-out set

Written and sealed 2026-10-27, after the P33.4 NO-GO and before any Phase 34 work
exists. It follows the route `docs/prereg-p33-4-go.md` ("One look") sets for a
NO-GO: a new task, a new pre-registration and a fresh held-out set. The user asked
on 2026-10-27 for the build to keep running autonomously. That request is the human
decision the P33.4 NO-GO waited for. P32.x and P30.4 are now gated on P34.5 instead
of P33.4, and still need a GO.

## What P33.4 showed (train LODO, results/P33.4.json)

The candidate (multi-target occupancy into an L2 logistic) lost to its own
baselines on train data, before the held-out look: AUROC 0.57 / 0.34 / 0.08 / 0.00
for sedation / anticholinergic / EPS / orthostatic, against 0.61 to 0.94 for
plain occupancy_max. An AUROC near 0 from a leave-one-out logistic on about 15
drugs is the leave-one-out intercept artifact, not a real signal. Only 15 to 19 of
267 train drugs reached the common set, although about 200 drugs have labels. EPS
has 8 positives and orthostatic hypotension 3 across all 334 v2 drugs.

## Order (the anti-tuning rule)

1. P34.1 to P34.3 use **train drugs only**. The v2 held-out set is burned and
   stays stripped from session trees. It is not reused for anything gating.
2. P34.3 **freezes** the candidate: the code and fitted parameters are committed,
   and their sha256 goes into `results/P34.3.json`, before any fresh held-out truth
   exists.
3. P34.4 builds the fresh held-out truth only after that freeze, by the inclusion
   rule below. The P34.4 build scripts must not import or run the candidate.
4. P34.5 scores the frozen candidate once on the host. The first verdict binds
   (`HOME/go/P34.5.json`).

## Fresh held-out inclusion rule (fixed now, applied mechanically by P34.4)

- **Pool:** approved small-molecule drugs (ChEMBL max_phase 4, release used by
  P33.1 or later) that are NOT among the 465 P33.1 candidates. They also need
  pChEMBL >= 5.0 at one or more of the P33.1 anchor targets, OR an ATC code in
  N05, N06, N03, N07, A03, A04, G04BD, C02CA, R06. Every pool member is listed,
  including the ones later excluded.
- **Labels:** built with `scripts/v2_liabilities.py` and the liability rules file
  as frozen by P34.2. The rule is not re-tuned on the fresh set. The drug label
  source is openFDA. EU SmPC tables may be used only through the same
  placebo-subtracted rule, and only if P34.2 implemented and froze that parser on
  train drugs first.
- **Cmax and Ki:** the `scripts/v2_cmax.py` and `scripts/v2_binding.py` procedures
  as of P34.3's freeze.
- **Exclusion:** only for a missing input, recorded per drug with the reason. A
  drug may not be dropped for any other reason.
- **Power floor:** a liability is scored only if its fresh common set has at least
  8 positives and 8 negatives. If no liability reaches the floor, P34.4 ends with
  `TASK_RESULT: EMPTY` and names the gap. That is not a NO-GO, and it goes to the
  human.

## Rule

- **Statistic and decision:** identical to `docs/prereg-p33-4-go.md`. That means
  candidate AUROC minus best-baseline AUROC on the common set, a paired bootstrap of
  drugs (2000 resamples, seed 20261026, percentile 95% interval), and GO if the
  interval lies entirely above 0 for at least one scored liability.
  Implementation: `pipeline/go_verdict.py`, unchanged.
- **Baselines:** strongest-target Ki, 1-NN fingerprint (fitted on train drugs
  only), ATC-class prevalence (train prevalence), and plasma Cmax,u/Ki occupancy
  (occupancy_max). The candidate may not be one of these baselines.
- **NO-GO:** anything else. P32.x then stays held for the human. No Phase 35
  retry is started without them.

## Secondary, not gating

Human CL and Vss fold error against the train mean, as in P33.4.

## Disclosed weakness

The P33.4 train LODO table and the v2 held-out baseline AUROCs are known. The
fresh pool overlaps the v2 drugs in pharmacology, so it tests transfer to new
molecules, not to new mechanisms.
