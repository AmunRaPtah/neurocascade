# Pre-registration: P33.4 GO / NO-GO for the v2 chain

Written and sealed 2026-10-26, before any candidate model exists. It operationalises
the gate stated in P33.3's brief on 2026-10-21: "GO only if at least one liability
beats every baseline on held-out drugs with a CI excluding the best baseline".

## Why this exists

The 2026-10-26 audit found that P33.3 scored only baselines, because there was no
candidate. Its own doc says the GO condition was not met (docs/architecture.md, P33.3
section). The conductor had no gate check and went on to P30.3 and P32.1, so the
product surface was being built around the training mean. From now on, P32.x carries
`requires_go: ["P33.4"]`, and the conductor will not start them until a host-side
marker says GO.

## Rule

- **Population:** v2 held-out drugs (`guards.v2_heldout`), frozen v2 truth, with
  `human_pk_corrections.json` applied.
- **Liabilities:** those P33.3 scores (sedation, anticholinergic), plus any other v2
  liability that reaches at least 5 positives and 5 negatives on the common held-out
  set.
- **Baselines:** the P33.3 set, frozen as implemented in `chain_benchmark.py` at
  merge 7c14fbb: strongest-target Ki, 1-NN fingerprint, and ATC-class prevalence.
  This adds the plasma Cmax,u/Ki occupancy baseline once v2 Cmax exists.
- **Common set:** a drug counts only if the candidate and every baseline have a score
  for it.
- **Statistic:** AUROC of the candidate minus AUROC of the best baseline on the same
  drugs. The best baseline is the one with the highest held-out AUROC. The 95%
  interval comes from a paired bootstrap of drugs: 2000 resamples, seed 20261026,
  percentile method.
- **GO:** for at least one liability, the 95% interval lies entirely above 0.
- **NO-GO:** anything else, including every liability being underpowered.
- **Implementation:** `pipeline/go_verdict.py`. It is human-owned and cannot be
  edited by sessions.

## One look

The candidate is chosen on train leave-one-drug-out only. The held-out evaluation
runs once on the host through P33.4's `host_eval`. The conductor records the first
verdict in `HOME/go/P33.4.json`, and a later run cannot overwrite it. A NO-GO holds
P32.x until a human decides what comes next. A new candidate would need a new task, a
new pre-registration and a fresh held-out set. Retrying against this one is not
allowed.

## Secondary, not gating

Human CL and Vss fold error against the train mean, reported beside the verdict. A
liability GO with PK still at the mean means the product claims liabilities only, not
dose.

## Disclosed weakness

Held-out baseline AUROCs (0.56 to 0.63) were seen on 2026-10-25 in the P33.3 host
run, before this document was written. No candidate has been scored on held-out data.
Knowing the baselines are weak could still bias which kind of candidate gets built.
Because of that, the rule compares against the baselines on the same drugs and does
not use a fixed number.
