# P29.6 — microsomal binding (fu_mic) and blood:plasma ratio (Rb) for the P29.2 CL arm

## Problem

The rebuilt P29.2 well-stirred-liver IVIVE clearance arm was level with the
train-mean floor on train (GMFE 2.855 vs ~2.75–2.82) and its error was a
*systematic under-prediction of lipophilic bases* — haloperidol 18-fold,
desipramine 8.7, chlorpromazine 7.5 — the signature of the omitted microsomal
binding (fu_mic) correction. The well-stirred model is

    CL_H = Q_H · fu_b · CLint_u / (Q_H + fu_b · CLint_u),
    fu_b = fu_p / Rb,   CLint_u = CLint / fu_mic.

The frozen arm set fu_mic = 1 and Rb = 1. Omitting fu_mic (< 1 for lipophilic
drugs) under-estimates CLint_u and therefore CL_H, worst for the most
lipophilic bases — exactly the observed pattern.

## What P29.6 adds

A sourced physicochemical record per drug (`src/neurocascade/benchmark/physchem.json`),
consumed by `human_pk_translation._ivive_cl(..., binding_corrected=True)`:

- **logP** — ChEMBL_37 ALogP, cited by CHEMBL id per drug (e.g. haloperidol
  CHEMBL54 4.43, chlorpromazine CHEMBL71 4.89, desipramine CHEMBL72 3.53).
- **pKa / ionisation class** — from PubChem (ChEMBL/DrugBank/HSDB), used only to
  pick the lipophilicity descriptor and the Rb acid/base default. Austin
  (2002)/Hallifax-Houston (2006) use logP for bases and logD7.4 for
  neutrals/acids. Every drug in the scored set is a base or a neutral
  benzodiazepine; no anionic acid is present, and each neutral lacks an
  ionisable group near pH 7.4, so logD7.4 = ALogP and ALogP is the descriptor
  throughout.
- **fu_mic** — Hallifax & Houston 2006 (PMID 16552024), the quadratic
  refinement of the Austin 2002 (PMID 12433825) logP microsomal-binding model:

      fu_mic = 1 / (1 + C · 10^(0.072·x² + 0.067·x − 1.126)),  x = logP.

  The coefficients 0.072 / 0.067 / −1.126 are Hallifax-Houston's; the file and
  module attribute them so, and name Austin 2002 as the model they refine.
- **Incubation protein concentration C** — a *declared default* of
  **0.5 mg/mL**: the ChEMBL CLint records carry no per-assay protein
  concentration, so a default is declared. 0.5 mg/mL is the standard human
  liver microsome substrate-depletion CLint assay concentration
  (Słoczyńska et al. 2019, Acta Pharm 69:345-361, PMID 31259741). The result
  is robust to this choice: C = 1.0 mg/mL gives train GMFE 1.61, C = 0.5 gives
  1.91; both beat the mean. The more conservative (higher-GMFE) 0.5 was chosen,
  not the value that minimises GMFE.

### Blood:plasma ratio (Rb) — measured where public, default otherwise

Rb enters as fu_b = fu_p / Rb. A **measured human whole-blood:plasma ratio** is
used where a primary source reports one directly:

| drug | Rb | source |
|------|----|--------|
| haloperidol | 0.79 | Cheng et al. 1987, Psychopharmacology (Berl) 91:410-414, PMID 3108922 (steady-state blood:plasma 0.79 ± 0.03 in patients) |
| clozapine | 0.88 | Flanagan et al. 2003, Br J Clin Pharmacol, PMID 12848787 (whole-blood ~10% below plasma) |
| risperidone | 0.67 | Mannens et al. 1994, Psychopharmacology (Berl), PMID 7531854 ("blood:plasma ratio averaged 0.67 in man") |

For every other drug no clean in-vivo whole-blood:plasma ratio is public, so the
**base/neutral default Rb = 1** is used (Obach 1999 acid/base convention, PMID
10534321) and the search is recorded per drug as searched-and-empty. Where only
a *different* quantity exists it is noted but not adopted, because converting it
to Rb would need an un-sourced assumption:

- desipramine — in-vitro RBC/plasma ≈ 0.78 (PMID 3982232); not a whole-blood
  ratio (conversion needs a haematocrit). Default.
- chlorpromazine — erythrocyte/plasma ≈ 0.5 (Lund 1980); same issue. Default.
- clomipramine — plasma/whole-blood 0.70–1.20 (PMID 1277478), spans 1. Default.
- citalopram — RBC accumulation in vitro (Fišar 2006, PMID 16816836); no clean
  in-vivo Rb. Default. (Applying the reported RBC-derived Rb > 1 would *reduce*
  citalopram's prediction, i.e. help it; it is deliberately not applied.)
- mirtazapine — only a postmortem blood:plasma ≈ 2.5; no antemortem Rb. Default.
- maprotiline — concentrates in erythrocytes (Maguire 1980) but no numeric Rb.
- promazine, chlordiazepoxide, lorazepam, oxazepam, trimipramine, trazodone —
  no public human whole-blood:plasma ratio found. Default.

Only directly-measured ratios are applied, and every measured ratio found is
applied regardless of whether it helps the fit (risperidone's 0.67 raises its
prediction and could overshoot; it is used anyway). This is the rule the brief
names — "measured Rb where public and the acid/base default otherwise" — made
explicit and auditable rather than silently dropped.

## Result (train, leave-one-drug-out, same 15 drugs as the frozen arm)

- **Binding-corrected train LOO CL: GMFE 1.911, 11/15 within 2-fold, 12/15 within
  3-fold.**
- **Train-mean baseline on the same 15 drugs: 2.817** (leave-one-out geometric
  mean of the observed CLs).
- **The arm now beats the mean (1.911 < 2.817)**, where the uncorrected arm did
  not (2.855).

Per-drug lipophilic-base fold error before → after:

| drug | before | after | fu_mic (C=0.5) | Rb |
|------|--------|-------|----------------|----|
| haloperidol | 18.36 | 5.32 | 0.343 | 0.79 |
| desipramine | 8.71 | 5.95 | 0.663 | 1.0 (default) |
| chlorpromazine | 7.55 | 1.85 | 0.193 | 1.0 (default) |

desipramine stays ~6-fold low: its available microsomal CLint under-represents
its (largely CYP2D6) clearance; fu_mic corrects binding, not a low CLint input.

## Scope and freezing

- The frozen uncorrected P29.2 arm is **unchanged**. `score_train()` /
  `score_heldout()` and the V20 conformal bands calibrated on them are
  untouched. The binding correction is a separate opt-in
  (`binding_corrected=True`, `score_train_binding_corrected()`), so no empirical
  bias factor is applied in it — the correction is mechanistic.
- Development used TRAIN only. Held-out binding-corrected CL is host-pending:
  physchem inputs (logP, Rb) are provided for the six held-out drugs with an
  IVIVE path (alprazolam, diazepam, fluphenazine, imipramine, nomifensine,
  venlafaxine) so the host can apply the frozen method; held-out truth is absent
  in the dev sandbox, where `score_heldout_binding_corrected()` returns
  searched-and-empty (see `heldout_binding_corrected_status()`).

## Verification of every PMID (checked against PubMed 2026-10-27)

16552024 Hallifax & Houston 2006 DMD · 12433825 Austin et al. 2002 DMD ·
31259741 Słoczyńska et al. 2019 Acta Pharm · 10534321 Obach 1999 DMD ·
3108922 Cheng et al. 1987 Psychopharmacology · 12848787 Flanagan et al. 2003
Br J Clin Pharmacol · 7531854 Mannens et al. 1994 Psychopharmacology ·
3982232, 1277478, 16816836 (partial-data sources cited but not adopted).
