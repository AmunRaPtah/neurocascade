# Amendment P34.4b: UK SmPC labels for the fresh set (disclosed deviation)

Written and sealed 2026-10-28, after P34.4 returned EMPTY and before any SmPC
for a fresh drug has been fetched. It amends `docs/prereg-p34-go.md`, which
allows SmPC tables only if P34.2 froze an SmPC parser on train drugs first.
P34.2 did not (`results/P34.2.json` `smpc_parser.frozen: false`). Adding a parser
now is therefore a **deviation from the pre-registration**, and it is made after
the fresh label counts were seen. The user chose this route on 2026-10-28 after
being told both of these facts and that the likely result is still EMPTY.

## What had been seen when this was written

- The P34.4 openFDA label counts on the 37 fresh drugs (`results/P34.4.json`),
  positives / negatives: eps 0/20, sedation 1/19, weight_gain 0/0 (withheld
  under A7), anticholinergic 4/16, orthostatic_hypotension 0/20. Positives by
  name: anticholinergic alosetron, fesoterodine, nabilone, tolvaptan; sedation
  nabilone.
- The PR #49 train validation of the parser used here (commit `163fdc0`,
  branch `architect/P34.2`): 267 train drugs, 180 with an emc SmPC, 3 with an
  SmPC placebo-controlled table (cetirizine, levocetirizine, rivastigmine),
  12 compared labels, 12 agree, **0 openFDA-positive**.

## What had not been seen

- No candidate score, rank or AUROC on any fresh drug. P34.5 has not run.
- No SmPC page for any fresh drug. No fresh INN has been searched on emc.

The deviation can therefore move the fresh sample size, but not toward a
candidate result: the candidate (frozen in P34.3) and the P34.5 rule are
untouched, and labels are written before any fresh score exists.

## Parser

`scripts/v2_smpc.py` and `src/neurocascade/benchmark/v2/smpc_rules.json` from
PR #49 (commit `163fdc0`), taken unchanged. Their rules were written before
P34.4 built the fresh pool. The source is the UK electronic medicines
compendium (medicines.org.uk/emc), whose SmPCs follow the EU QRD template; that
substitution for "EU SmPC" is unchanged from PR #49 and disclosed there.

Precedence is unchanged: an SmPC label is used only for a drug whose openFDA
build gives no `placebo_subtracted` basis, and never replaces one. The openFDA
absolute-incidence fallback is never applied to an SmPC.

Train-only changes are allowed before the freeze. Each one is added to
`smpc_rules.json` `amendments` with a `made_after_seeing` naming the train
drugs that prompted it. No fresh INN may be searched, fetched or parsed before
the freeze.

## Validation bar (train only, fixed now)

The compared set is every train drug (`v2_partition()` train, not v2
held-out) with both an openFDA and an SmPC `placebo_subtracted` label, over
eps, sedation, anticholinergic and orthostatic_hypotension (weight_gain is
withheld under A7). The parser passes only if all three hold:

1. at least 3 compared labels are openFDA-positive;
2. the SmPC label is positive on at least 2/3 of those;
3. the SmPC label agrees on at least 90% of the openFDA-negative compared labels.

Condition 1 is a positive control. PR #49 was rejected because its 12
agreements were all negatives, which cannot show the parser ever finds a
positive. PR #49's own run fails condition 1. Unless today's emc pages differ,
the parser is expected to fail.

If the parser fails, it is not frozen and not applied. The result is
`EMPTY (SmPC parser failed train validation)`, no fresh drug is touched, and
the decision returns to the human.

## If it passes

1. Freeze: the sha256 of `scripts/v2_smpc.py` and `smpc_rules.json` is written
   to `results/P34.4b.json` and pinned in a test before any fresh fetch.
2. Apply once, to the 37 drugs in `truth_fresh/drugs.json` under the precedence
   rule. In practice that means the 20 drugs with no openFDA placebo table. The 38 pool
   members excluded in P34.4 for having no FDA label stay excluded. The frozen
   Cmax build reads values from a drug's FDA label (`scripts/v2_cmax.py`), so
   admitting them would be a second deviation.
3. Re-count against the unchanged power floor of 8 positives and 8 negatives per
   liability. Below the floor the result is EMPTY and goes to the human. At or
   above it, Cmax and Ki are built with the P34.2-frozen scripts and P34.5 runs
   under its unchanged rule, with this amendment cited in the verdict.

## Expected outcome, stated in advance

Most likely EMPTY. If the parser fails validation, the build stops at that
step. If it passes, at the train rate (3 of 180 SmPCs with a placebo table)
about 0.3 of the 20 eligible fresh drugs would gain a placebo table, and
anticholinergic needs at least 4 more positives.
