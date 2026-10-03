# P29.5: oral exposure, preclinical oral inputs, and PET plasma pairs

## Result: partially closed; acceptance thresholds are not met

P29.5 searched the named primary and first-party sources before adding any
record. The oral-exposure truth gap is closed for all eight held-out PET
drugs. The PET dose/plasma gap is closed only for five of the eight, and the
preclinical oral-input gap remains below the train threshold. These are data
records, not fit inputs: held-out truth is comparison-only.

| Evidence layer | Coverage | Drugs with records | Remaining gap |
|---|---:|---|---|
| Human oral exposure truth | 8/8 PET drugs | amisulpride, asenapine, escitalopram, lurasidone, milnacipran, paroxetine, quetiapine, venlafaxine | none for the requested eight |
| PET dose/plasma pair | 5/8 PET drugs | amisulpride, asenapine, paroxetine, quetiapine, venlafaxine | escitalopram, lurasidone, milnacipran |
| Preclinical oral/clearance input, held-out PET drugs | 5/8 | asenapine, escitalopram, lurasidone, milnacipran, venlafaxine | amisulpride, paroxetine and quetiapine are searched-and-empty |
| Preclinical oral/clearance input, train split | 36/130 (27.7%) | inherited P28.3 CLint or in-vivo clearance/Vss fields | below the required 60% |

The eight oral records preserve the quantity actually printed by the source:
AUC/dose for amisulpride and quetiapine; CL/F for escitalopram, lurasidone,
paroxetine and venlafaxine; and absolute-F or an explicitly bounded absorbed
fraction for asenapine, lurasidone, milnacipran, paroxetine and venlafaxine.
Ranges and inequalities remain ranges and inequalities. The swallowed and
sublingual asenapine routes are separate. No F midpoint, body weight, active
metabolite substitution, or cross-drug value was introduced.

The PET records similarly preserve qualifiers. Asenapine is a regulatory
target-concentration model point, paroxetine is a plateau boundary, and
venlafaxine is a subject-level NET/active-moiety series rather than a parent
SERT measurement. Quetiapine's values are reported in nmol/L, not ng/mL;
the schema therefore uses `plasma_concentration` plus `plasma_units`.

## Search decisions

- Amisulpride oral AUC/dose uses the 200-mg human study (PMID 28793958).
  Its rat oral time-course paper (PMID 24619919) was searched, but the
  reachable record did not print a numeric F, CLint, or summary PK value.
- Asenapine uses the FDA NDA 022117 physician label for sublingual F and the
  swallowed-oral upper bound, and its pharmacology review for the rat F range.
- Escitalopram and lurasidone use their first-party labels for human CL/F or
  absorbed-fraction records. Lurasidone's rat oral F, IV clearance/Vss, and
  plasma binding use PMID 21838595.
- Milnacipran uses the Savella NDA 022256 label. Paroxetine uses NDA 021299
  and the regimen-specific human CL/F in PMID 18483540; its rat oral study
  (PMID 1826617) did not expose a numeric F or clearance in the reachable
  record.
- Quetiapine uses the open PET/PK study in PMID 21477416. Venlafaxine uses
  the Effexor XR NDA 020699 label and the active-moiety NET PET table in
  PMID 30649319.
- Searches for numerical paired PET plasma values for escitalopram (PMID
  27557550), lurasidone (PMID 23649882), and milnacipran (PMID 23067569)
  were empty in the reachable records; model EC50s, dose/occupancy summaries,
  and figure-only values were not promoted to pairs.

The complete records are in
`src/neurocascade/benchmark/p29_5/`: `oral_exposure.json`,
`pet_plasma_pairs.json`, and `preclinical_oral_inputs.json`. The loader in
`p29_5.py` reports the coverage counts and is not imported by any fitting
module. The frozen P28.1 split is unchanged.

Because 5/8 PET pairs and 36/130 train preclinical inputs do not meet the
acceptance criteria, this task remains an honest searched-and-empty/partial
result rather than a completed calibration.
