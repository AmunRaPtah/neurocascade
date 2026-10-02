# P29.3: predicted human target-engagement dose, scored held out

## Result: searched-and-empty

No held-out oral target-engagement dose can be scored from the frozen Phase 28
benchmark. The score is therefore **not estimable (n = 0)** for both clinical
dose and PET-anchored dose. This is a data result, not an empty prediction
registry.

The proposed chain would need, for a target with fractional occupancy `p`,

```
Cbrain,u = Kp,uu * fu,p * Cplasma,total
Ctarget,u = Ki * p / (1 - p)
Cplasma,total,target = Ctarget,u / (Kp,uu * fu,p)
Dtarget/kg/day = Cplasma,total,target * CL/kg * 1.44 / F
```

The final expression uses the steady-state mean concentration for a once-daily
oral regimen; for another interval, `1.44` would be replaced by the appropriate
minutes-per-day factor. To report `mg/day` rather than `mg/kg/day`, a
source-defined human reference weight is also required. Every quantity in the
equation must belong to the same drug, species, analyte, route, and regimen.

The frozen benchmark has none of the required oral-exposure fields: its truth
records contain IV/unspecified human CL and Vss, label dose ranges, and some
half-lives, but no same-entity oral absolute F, oral CL/F, or oral AUC. P29.2's
CL prediction is per kg because the preclinical records contain no body
weights. Thus multiplying the prediction by an unreferenced 70 kg, setting
`F = 1`, or borrowing F from another drug would each create a number that the
source did not measure. The dose range also does not define a unique point for
a pointwise fold-error score.

For N05A antipsychotics, the target threshold was pre-specified as 70% striatal
D2 occupancy, a value within the 65–70% efficacy range discussed by Kapur and
Seeman (2001, PMID 11729039) and the 70–80% therapeutic window reviewed by
Kapur et al. (2009, PMID 21143431). This threshold is not transferred to N05B,
N06A, or FAIL drugs: no class-level human target-engagement threshold was
found for those strata in the benchmark source set.

## What was searched

The repository search covered all 55 held-out records in
`benchmark/truth/<inn>.json`, all P29.2 prediction inputs and outputs, every
held-out `benchmark/adme/<inn>.json`, every human binding panel, the P29.1
Kp,uu records, and the P28.4 label transcriptions. It specifically searched
for oral F, CL/F, oral AUC, a reference body weight, a target-specific human
occupancy threshold, and a matched PET dose.

External candidates were checked before closing the task:

* de Greef et al. (2011, PMID 21184291; PMC3032087) supplies a useful
  antipsychotic dose/PK/PET modelling paper and reports a ziprasidone fed oral
  bioavailability value, but it does not supply a complete benchmark-wide,
  source-defined reference-weighted input set. Using its dose/occupancy fit as
  the prediction would also reuse the PET relationship being sought as an
  anchor.
* The FDA GEODON label (NDA 020825, revised 2007) states approximately 60%
  absolute bioavailability under fed conditions. That single value does not
  close the missing reference weight, and it is not available for all held-out
  drugs. The food effect is large and regimen-relevant, so it cannot be
  silently treated as a drug-invariant F.
* Fridén et al. (2009, PMID 19764786) supports the P29.1 rat-to-human Kp,uu
  transfer method, but its human Kp,uu,CSF overlap with this benchmark is only
  four drugs (bupropion, diazepam, amitriptyline, and thioridazine); it does
  not provide a human Kp,uu measurement for ziprasidone or a complete held-out
  panel.
* Lako et al. (2013, PMID 23948784) provides population dose-to-D2-occupancy
  functions for eight antipsychotics. Those are PET-derived dose/occupancy
  observations and are therefore comparison data, not a substitute for the
  missing oral PK bridge. Suzuki et al. (2013, PMID 23417515) similarly gives
  a ziprasidone 60 mg twice-daily PET time course, but its reported occupancy
  changes across the dosing interval and does not constitute a 70%-threshold
  point.

These candidates are usable as future anchors or method references, but none
provides the complete, independent, same-entity input needed to score the
requested frozen held-out split. No cross-drug F, body weight, Ki, occupancy,
or dose was substituted.

## Held-out census

`yes` means that the frozen repository contains at least one admissible value
for that layer; it does not mean that the complete chain is runnable. `D2 Ki`
is checked only for N05A, because the 70% threshold is not defined for the
other strata. `P29.2 CL` means a held-out CL prediction exists, not that its
clinical truth was used in fitting. `PET D2` means a PET dose/occupancy record
is present in the frozen truth; it is not used to fit this failed chain.

| drug | stratum | D2 Ki | rat Kp,uu | P29.2 CL | human fu | PET D2 | missing layer / reason not run |
|---|---|---:|---:|---:|---:|---:|---|
| alprazolam | N05B | — | no | no | yes | no | no N05B target threshold; oral F/CL/F absent |
| amineptine | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| amisulpride | N05A | yes | no | no | yes | yes | Kp,uu absent; oral F/CL/F absent |
| amitriptyline | N06A | — | yes | no | yes | no | no N06A target threshold; P29.2 CL absent |
| amoxapine | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| aprepitant | FAIL | — | no | no | no | no | no target threshold; binding and PK inputs absent |
| asenapine | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| benzoctamine | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| bifemelane | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| bitopertin | FAIL | — | no | no | no | no | no target threshold; binding and PK inputs absent |
| bradanicline | FAIL | — | no | no | yes | no | no target threshold; binding and PK inputs absent |
| butaperazine | N05A | no | no | no | no | no | human D2 Ki absent; oral PK absent |
| chlorproethazine | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| clobazam | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| diazepam | N05B | — | yes | yes | yes | no | no N05B target threshold; oral F/CL/F absent |
| dosulepin | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| droperidol | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| escitalopram | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| fabomotizole | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| fluanisone | N05A | no | no | no | no | no | human D2 Ki absent; human binding panel absent |
| fluoxetine | N06A | — | no | no | no | no | no N06A target threshold; Kp,uu and CL absent |
| fluphenazine | N05A | yes | no | no | yes | no | Kp,uu absent; oral F/CL/F absent |
| gedocarnil | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| imipramine | N06A | — | no | yes | yes | no | no N06A target threshold; Kp,uu absent |
| levosulpiride | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| lofepramine | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| lurasidone | N05A | yes | no | no | no | yes | Kp,uu absent; oral F/CL/F absent |
| mephenoxalone | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| milnacipran | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| moclobemide | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| moperone | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| mosapramine | N05A | no | no | no | no | no | human D2 Ki absent; oral PK absent |
| navacaprant | FAIL | — | no | no | yes | no | no target threshold; binding and PK inputs absent |
| nialamide | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| nomifensine | N06A | — | no | yes | yes | no | no N06A target threshold; Kp,uu absent |
| nordazepam | N05B | — | no | no | yes | no | no N05B target threshold; human binding panel absent |
| nortriptyline | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| oxitriptan | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| paroxetine | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| penfluridol | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| perazine | N05A | no | no | no | no | no | human D2 Ki absent; human binding panel absent |
| periciazine | N05A | no | no | no | no | no | human D2 Ki absent; oral PK absent |
| pinazepam | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| pipotiazine | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| quetiapine | N05A | yes | yes | no | no | yes | P29.2 CL absent and human fu absent; oral F/CL/F absent |
| reboxetine | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| tofisopam | N05B | — | no | no | no | no | no N05B target threshold; human binding panel absent |
| tranylcypromine | N06A | — | no | no | yes | no | no N06A target threshold; Kp,uu and CL absent |
| trifluoperazine | N05A | yes | no | no | yes | no | Kp,uu absent; oral F/CL/F absent |
| triflupromazine | N05A | yes | no | no | no | no | Kp,uu absent; oral F/CL/F absent |
| venlafaxine | N06A | — | yes | yes | yes | no | no N06A target threshold; target-specific occupancy threshold absent |
| verucerfont | FAIL | — | no | no | no | no | no target threshold; binding and PK inputs absent |
| vilazodone | N06A | — | no | no | no | no | no N06A target threshold; binding and PK inputs absent |
| ziprasidone | N05A | yes | yes | yes | yes | no | oral F/CL/F absent in benchmark; P29.2 CL is per kg and no source-defined reference weight exists for mg/day |
| zuclopenthixol | N05A | no | no | no | no | no | human D2 Ki absent; oral PK absent |

## Score

| comparison | n | fold-error | status |
|---|---:|---:|---|
| predicted target-engagement dose vs clinical dose | 0 | not estimable | searched-and-empty |
| predicted target-engagement dose vs PET-anchored dose | 0 | not estimable | searched-and-empty |

The held-out denominator remains all 55 drugs; drugs were not silently removed
from the split. A future task can produce a score only after adding a
same-entity oral exposure record (absolute F or CL/F), a source-defined human
reference weight compatible with P29.2's per-kg prediction, and a
pre-specified target/threshold for the non-N05A strata. A PET dose/occupancy
observation must remain held out from any fit used to make that score.
