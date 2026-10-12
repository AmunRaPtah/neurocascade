# P29.4: panel occupancy-time at clinical dose

## Result: searched-and-empty

No held-out clinical-dose occupancy-time prediction can be scored from the
frozen Phase 28 benchmark. The requested score is therefore **not estimable
(n = 0)**. This is a data result, not an empty prediction registry.

The observation side is not the blocker. Eight held-out drugs have at least
one PET dose/occupancy observation, and all eight have a P28.2 human Ki panel:

| drug | PET target(s) | panel targets | first missing prediction layer |
|---|---|---:|---|
| amisulpride | D2 | 14 | P29.2 human oral exposure; Kp,uu |
| asenapine | D2, 5-HT2A | 18 | preclinical ADME/PK; sublingual route |
| escitalopram | SERT | 7 | oral exposure/reference weight |
| lurasidone | D2 | 13 | oral exposure; Kp,uu |
| milnacipran | SERT | 2 | oral exposure; compatible P29.2 clearance input |
| paroxetine | SERT | 13 | oral exposure; Kp,uu transfer |
| quetiapine | D2, 5-HT2A | 40 | P29.2 clearance; oral exposure |
| venlafaxine | SERT | 6 | active metabolite; oral exposure |

The P29.4 denominator is consequently eight PET drugs, but the complete
prediction denominator is zero. No MAE or bias is reported per receptor. A
PET dose/occupancy observation is comparison data; using its dose-response
relation to fill the missing exposure model would put the answer into the
prediction.

## Load-bearing assumption checked first

P29.2 predicts human **IV** CL and Vss in units per kilogram. It does not
predict oral AUC, absolute bioavailability, oral CL/F, or a human reference
weight. A clinical-dose plasma curve therefore cannot be formed from its
output. The required relation is:

```
Cplasma,oral(t) = PK(dose_mg / reference_weight_kg, F, CL/kg, V/kg, ka, ...)
```

The benchmark has no same-entity, route-matched `F`, `CL/F`, oral AUC, or
reference weight for this eight-drug PET subset. Setting `F = 1`, multiplying
by an unreferenced 70 kg, borrowing another drug's F, or selecting a point
from a label dose range would each create a number forbidden by rules 1–2.
The P29.2 held-out CL error (n=5; GMFE 3.42) is therefore not an
oral exposure band. It cannot be attached to a curve that has not been
identified.

The second assumption is target exposure. A panel Ki can be put into the
competitive occupancy expression only with a brain exposure term, ordinarily
`Kp,uu * fu,p * Cplasma`. The frozen P28.3 records contain usable rat Kp,uu
only for quetiapine and venlafaxine among these eight. The P29.1 transfer
method is explicitly tagged and its error is measured against human CSF
surrogates, but it does not create missing drug-specific measurements. The
P23.2 record also establishes that an in-vitro Ki-derived occupancy is weaker
than the in-vivo PET-fitted D2 quantity: the haloperidol discrepancy reaches
0.368 occupancy units on a clinical curve. That warning is carried forward;
no unmeasured Ki-to-PET correction was fitted here.

Venlafaxine is the closest case: P29.2 has a CL prediction and P28.3 has rat
Kp,uu. It still cannot be scored honestly because venlafaxine's clinically
active O-desmethylvenlafaxine is not represented in the parent-only panel.
The primary human study reports a parent/metabolite exposure relationship
(Klamerus et al. 1992, PMID 1487561); dropping the metabolite would not be a
parent-only prediction of the SERT PET quantity.

## What was searched

The frozen `truth`, `binding`, and `adme` records were enumerated for all 55
held-out drugs, then the eight PET-positive records were checked against the
P29.2 models and P29.1 Kp,uu path. The external literature search was run on
2026-10-03 using PubMed/Europe PMC queries of the form:

* `"<drug>" population pharmacokinetics oral clearance volume absorption`
  for each of the eight drugs;
* `"<drug>" Kp,uu brain` and `"<drug>" unbound brain plasma` for each drug;
* `"<drug>" PET occupancy plasma concentration dose` for each PET target.

The usable candidates did not close the requested chain:

* Søgaard et al. 2005 (PMID 16291715) gives human escitalopram IV CL/V and
  oral Tmax/half-life, but not a preclinical-to-human oral bridge or brain
  Kp,uu. Liu et al. 2022 (PMID 35924062) gives a clinical escitalopram
  apparent model, not a preclinical input.
* Bartlett & van der Voort Maarschalk 2012 (PMID 22936407) describes
  asenapine sublingual absorption and Tmax qualitatively; the PET observation
  is a 100 microgram sub-therapeutic ORG 5222 dose (Andrée et al. 1997,
  PMID 9226735), not a clinical-dose oral record.
* Amisulpride clinical population-PK estimates exist (Zhang et al. 2023,
  PMID 37731733), but they are human apparent parameters and do not supply
  brain Kp,uu for the translation prediction.
* Zhou et al. 2015 (PMID 25975812) supplies a human quetiapine oral model,
  but it is a direct human PK model rather than P29.2's preclinical
  translation and does not solve the missing P29.2 error-calibrated oral
  bridge.
* Li et al. 2022 (PMID 36172189) supplies paroxetine human CL/F and V/F in a
  Chinese psychiatric population, with formulation, dose and sex effects;
  it is not a drug-specific preclinical translation input and does not supply
  Kp,uu.
* Li et al. 2012 (PMID 22653299) reports oral milnacipran human exposure and
  metabolite disposition, but no brain Kp,uu or P29.2-compatible oral bridge.
* Human venlafaxine PK reports include active-metabolite disposition
  (Klamerus et al. 1992, PMID 1487561). A parent-only curve would violate the
  same-entity/metabolite rule.
* Fridén et al. 2009 (PMID 19764786), Loryan et al. 2022 (PMCID PMC9246790),
  and the P28.3 ChEMBL_37 extraction were checked as the Kp,uu source set.
  The available values do not cover the eight drugs as a usable,
  species-tagged panel input.

These sources are useful future data leads, but taking their human clinical
PK as the prediction input would change P29.4 into a direct human-PK exercise
and would no longer measure the preclinical-to-human task. No number was
transcribed into the live model.

## Reopening conditions

P29.4 can be reopened when each scored drug has a same-entity, route-matched
oral exposure model driven by preclinical inputs, a source-defined reference
weight or a per-kg clinical dose, a species-tagged Kp,uu input, and an
active-metabolite treatment where applicable. The prediction must be frozen
before comparing it with the existing PET observations. At that point the
ledger should report MAE and signed bias separately for D2, 5-HT2A and SERT,
with the P29.2 error band carried on every predicted curve.
