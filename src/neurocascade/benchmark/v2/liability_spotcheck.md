# P33.2 liability labels: hand spot-check (2026-10-21)

## Method

Twenty drugs were checked against the raw `adverse_reactions_table` HTML of the
selected SPL. For each drug, all five liabilities were compared cell by cell
(100 drug-liability cells): drug %, placebo %, arms and N, the term chosen,
the pooled risk difference and the binary label. The check used
`V.parse_table` grid dumps plus a reading of the original HTML. No LLM was used.

- Fixed set (15): haloperidol, perphenazine, pimozide, olanzapine, quetiapine,
  risperidone, aripiprazole, lurasidone, sertraline, fluoxetine, clomipramine,
  trazodone, mirtazapine, oxybutynin, prazosin.
- Random set (5, `random.Random(20261021)` over the universe): silodosin,
  milnacipran, tamsulosin, aclidinium, methylphenidate.

## Result

| Drug | Basis / source | Verdict |
|---|---|---|
| haloperidol | placebo_subtracted, ANDA078347 (empty caption) | correct (EPS 50.7 vs 16 -> 1) |
| perphenazine | prose-only label, labels null | correct (no table exists) |
| pimozide | placebo_subtracted, count table N=20 | **wrong before A6**: counts read as %; anticholinergic 0 -> 1 (dry mouth 25% vs 5%); EPS 40 vs 0, sedation 70 vs 25 |
| olanzapine | ANDA202862 table 2 | correct |
| quetiapine | Seroquel XR | correct |
| risperidone | table 10 (bipolar monotherapy, largest n) | correct |
| aripiprazole | Abilify Maintena NDA217006 (LAI), drug_column_unnamed | values faithful; caveat: the LAI label was selected, not the oral one |
| lurasidone | table 20 | **structural error before A5**: colspan-duplicated arms and unpooled "All" column; no label flip after fix |
| sertraline | placebo pool | correct |
| fluoxetine | MDD column group (A4) | correct |
| clomipramine | adult column group; "-" read as 0 | correct |
| trazodone | outpatient column group | correct by rule; see caveat 2 |
| mirtazapine | placebo table | correct (sedation 54 vs 18 -> 1; weight gain 12 vs 2 -> 1; anticholinergic 0 by ratio) |
| oxybutynin | Oxytrol, N/% sub-columns | correct |
| prazosin | prose-only, labels null | correct |
| silodosin | placebo table | correct |
| milnacipran | "All" column pooled | correct |
| tamsulosin | n-weighted pool of 0.4/0.8 mg | correct |
| aclidinium | all mapped terms absent_below_threshold | correct |
| methylphenidate | Body System / Preferred Term layout | **structural error before term-column fix**: term read from body-system column; no label flip |

**Discrepancy rate, before fixes:** 1 label flip in 100 cells (1%). Three of the
20 drugs (15%) had a structural extraction error (pimozide, lurasidone,
methylphenidate). **After fixes:** 0 of 100 cells disagree with the source table.

The fixed drugs became the development set for amendments A3-A6, so the
post-fix 0% is optimistic. Treat about 15% drug-level structural error as the
honest prior for tables not yet inspected.

## Post-hoc amendments (in `liability_rules.json`, made after seeing data)

The pre-specified rules were committed in 59928fd before any results.
Amendments A1-A6 were all made after seeing data:

- A1: exclude discontinuation tables.
- A2: break ties in primary-table n by row count.
- A3: normalise row-label variants.
- A4: split column groups.
- A5: handle colspan-duplicate arms and the "all" pooled column.
- A6: detect count tables, which requires an impossible rounded % before converting.

The margin rule (RD >= 5 pp AND drug >= 2x placebo) and the term map are unchanged.

## Caveats and failure modes

1. Label selection: the primary label is chosen by rule, and for some drugs
   that is a non-oral or LAI formulation (aripiprazole).
2. Term choice: the max-RD term decides the label. For trazodone, blurred
   vision (15 vs 4) would pass, but dry mouth (34 vs 20, the higher RD) fails
   the 2x ratio. This is a rule effect, not an extraction error.
3. Weight gain is under-called. Many labels report weight in Warnings tables,
   which are outside adverse_reactions_table.
4. Prose-only labels: 121 drugs (mostly old non-PLR labels) have no table.
   They fall back to `absolute_fallback`, with labels null or prose flags.
5. Unmatched terms. Example: pimozide "Rigidity", because the map has
   "muscle rigidity". It did not change the label here, but similar near-misses
   elsewhere could.
6. Unparsed tables: 19 tables had no recoverable structure (anaesthetics,
   clozapine table 0) and 44 had an unnamed drug column (flagged). Seven
   mention placebo with no placebo column (tacrine, vigabatrin and others).
   Two liabilities ended `failed`: labetalol sedation and raloxifene weight gain.
