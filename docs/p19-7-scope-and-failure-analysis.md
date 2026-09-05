# P19.7: scope and failure analysis for the human dose-to-occupancy chain

The P8.3 held-out failure analysis (`docs/heldout-failure-analysis.md`) set the
standard this document follows: state what was tested, attribute each failure to
a layer rather than to the model as a whole, and say what data would repair it.
That discipline held up under an autonomous driver, so it is reused here without
modification.

Every number below was regenerated from the code in this repository. Nothing was
copied from a brief. Where the brief for this task asked a question the code
cannot answer, the answer given is "not determined in the repo".

## Result in one paragraph

The chain in `neurocascade.human.dose_to_occupancy` maps an oral haloperidol dose
to a striatal D2 occupancy-time curve through five layers, of which one is
`measured` (the Pilla Reddy 2013 apparent oral PK set), one is `fitted-in-repo`
(the 1.8849 nM plasma EC50 from P19.4), and three are `structural`. No layer is
`illustrative`, which is why `_gate_illustrative` currently refuses nothing. What
P19.6 tested is the plasma-to-occupancy hyperbola alone:
`chain_testable_observations()` returns zero observations pairing an oral dose
with a differenceable occupancy, so **the dose-to-plasma arm of the chain is
untested**.
On the arm that was tested, held-out error is 0.0351 to 0.0939 MAE in occupancy
units depending on the split, and the one whole-chain comparison available
(Kapur 1996's 2 mg group range 0.53-0.74) misses by 0.0047 on a single-dose
reading and is contained on a regimen-matched steady-state reading computed
afterwards. The chain predicts an occupancy. It does not predict behaviour, and
nothing in this repository licenses reading it as if it did.

## Failure matrix

| Layer | Evidence | Attribution | Concrete fix and required data |
|---|---|---|---|
| dose to plasma | `chain_testable_observations()` returns 0; no observation in P19.3 pairs an oral dose with a differenceable occupancy | untested, not failed. The distinction matters: no split in `PRE_REGISTERED_SPLITS` touches this arm | a study printing per-subject oral dose, measured plasma concentration and measured occupancy in the same table. P19.3 searched and found none reachable |
| PK identifiability | Pilla Reddy 2013 never identified F, so CL/F 88 L/h, Q/F 233 L/h, Vc/F 669 L, Vp/F 2500 L are apparent | source limitation. `apparent_parameters=True` folds F into the values and forbids passing an F separately | an IV plus oral crossover in the same subjects. P19.2 recorded the haloperidol IV search as empty: Holley 1983 (PMID 6831826) printed only CL and Vss, Chang 1992 (PMID 1579624) never printed micro-constants, both paywalled |
| weight scaling | the source prints an allometric exponent for clearance (0.75) and for nothing else | refused rather than defaulted. `predict_occupancy` raises off the 70 kg reference unless a caller supplies `WeightScaling(volume_exponent, q_exponent, source)` | a population analysis printing volume and Q exponents, or a caller willing to name the textbook values as their own claim |
| exposure | the haloperidol `kp_uu` record is `measurement_kind="absent"`, `status="not-required"` | structural, and deliberate. The fitted EC50 already contains brain penetration, protein binding and endogenous dopamine competition; a Kp,uu would apply the correction twice. The chain re-reads the record every call and raises if a value appears | nothing, unless somebody adds a Kp,uu, at which point one of the two layers has to give up carrying brain penetration |
| occupancy constant | 1.8849 nM (0.708 ng/ml), 95% profile interval 1.382-2.525, asymptotic SE 0.176, residual RMSE 0.036 | fitted in repo on four group means from two studies, and the four are not independent: each study measured the same eight patients at weeks 1 and 4. The interval is narrower than two independent groups would justify | per-subject occupancy and plasma pairs, oral, with the region and tracer printed. Split B (patient-disjoint) gives MAE 0.0351, which bounds what the study-to-study spread costs |
| regimen mismatch | both fitting studies are depot intramuscular steady state; the chain being driven is oral | carried forward unrepaired from P19.4 and stated in the `occupancy` layer's own text | an oral steady-state PET set. Kapur 1996 is the only oral point reachable and it is a group range, not per-subject values |
| basis mismatch | split C converts the total-basis constant to free plasma with fraction 0.09 and predicts four Videbaek subjects: MAE 0.0620, bias +0.0601, max 0.1409 | the bias is one-signed, so the conversion is systematically over-predicting. Total and free are ~11-fold apart, which is protein binding, not biology, and the two are never pooled | a study printing both total and free concentrations in the same subjects |
| external parameters | split D uses Kapur 1997's 1.3569 nM and Fitzgerald 2000's 1.0642 nM: MAE 0.0939, bias +0.0939, max 0.1637, every one of eight errors positive | both published constants fall below this repository's lower profile bound, so both over-predict every point. This is a disagreement between the literature and the fit, not a bug | resolving it needs the per-subject data behind Kapur 1997 and Fitzgerald 2000, which P19.3 could not reach |
| tracer | 11 of 15 registered PET observations have `region=None`; the tracer spread across [11C]raclopride, [123I]IBZM and [76Br]FLB 457 sets a 2-2.5x floor | the four depot points the EC50 is fitted on carry `tracer=None` **and** `region=None`. See the region section below | occupancy sets that print the tracer and the region. Both fields exist in the schema precisely so that a missing one is visible |
| transduction | `D2Effect` has `baseline_occupancy=0.0`, `emax=1.0`, `direction=-1.0` (`src/neurocascade/human/d2_effect.py:8-10`) | structural. With Emax 1 and baseline 0, `abs(effect)` equals occupancy exactly, and a test asserts the identity. It is a re-expression, not a second prediction | a human dose-response set that identifies an Emax and a baseline. None exists in `src/` |
| effect-site kinetics | the chain maps plasma to occupancy instantaneously: no effect compartment, no ke0 | a known structural gap, named in `OCCUPANCY_IS_NOT_THE_OUTCOME`. Kurose 2020 (PMID 32726002) reports central occupancy decaying more slowly than plasma, which this chain cannot represent | a fitted human ke0. The four `HUMAN_KE0_RECORDS` are illustrative (see the register below) |

## What the chain supports, and what it does not

### Region: striatal, on an inherited attribution

The chain is labelled striatal D2 throughout, and the P19.3 registry does hold
striatal observations. But the four group means the EC50 was actually fitted on
carry `region=None` in the registry, alongside `tracer=None`. The "striatal"
label on this repository's central constant is therefore inherited from the
context of the sources, not printed by either of them in a field this repository
could record. Eleven of the fifteen registered observations have `region=None`.

Extrastriatal D2 is out of scope and is not weakly supported. It is not
supported at all. Tuppurainen 2009 is the only extrastriatal record considered
and it was excluded: extrastriatal only, n=2, no plasma concentrations. Kessler
2005's fallypride data was unobtainable (the retrieved file is a 564-byte HTML
stub). The tracers differ in a way that matters here: [76Br]FLB 457 and
[11C]raclopride are not interchangeable, and the 2-2.5x spread across tracers in
the P19.3 set is larger than every held-out error reported above. Applying this
constant to a cortical or thalamic region would be a category error, not an
extrapolation.

### Dosing regimen: fitted on steady state, driven as single dose

The EC50 was fitted on depot intramuscular steady-state observations. The chain's
published predictions in `docs/p19-5-dose-to-occupancy-chain.md` are single oral
doses. This is the single largest unrepaired mismatch in the chain and it is
visible in the one whole-chain test available:

    Kapur 1996, 2.0 mg steady-state-daily, reported range 0.53-0.74
      chain, single-dose peak:      0.5253   not contained (0.0047 below the bound)
      chain, matched steady state:  trough 0.4718, mean 0.5635, peak 0.6592
                                    mean contained: True

The steady-state figure is not a second, passing test. `kapur1996_containment()`
records three caveats in code, and they are load-bearing: (i) the comparison was
not blind, because the 2 mg single-dose peak was already published in the P19.5
doc before P19.6 ran; (ii) the steady-state figure was computed *after* the
single-dose miss was seen, so it explains the miss rather than independently
confirming anything; (iii) the source reports a range across seven patients, not
a mean with a spread, so containment says only that the prediction lands
somewhere among them.

### Acute versus chronic

The chain has no time-varying pharmacology. There is no tolerance term, no
receptor up- or down-regulation, no autoreceptor adaptation, and no
accumulation term beyond ordinary two-compartment superposition. Its terminal
half-life is about 31 hours on this parameter set, so a repeated-dose prediction
is arithmetic on that half-life and nothing more.

The fitting data contains a week-1 versus week-4 contrast within both studies
(Nyberg 0.73 → 0.52, Regenthal 0.75 → 0.53), and the chain does not model
whatever produced it. That contrast is absorbed into residual scatter: the
week-4 points are the ones the leave-one-out fit misses low (Regenthal week 4,
error -0.0612, -2.13 training RMSE). Chronic dosing is therefore out of scope in
the strong sense: the chain does not merely lack a chronic term, its residuals
show where the missing term would go.

### Populations, and whether the two sets describe comparable people

They do not, and the difference is not small.

- **PK set.** Pilla Reddy 2013 (PMID 24113674): 122 subjects, of whom 20 are
  healthy volunteers and 102 are patients with schizophrenia; 538 concentrations;
  1-60 mg/day; oral in every arm. The published IIV (CL 44.5 %CV, Vc 116 %CV,
  proportional residual error 0.44) is **not** in this repository's schema, so
  the chain runs on typical values with no between-subject variance at all.
- **Occupancy set.** The four fitted points come from two depot intramuscular
  studies of eight patients each, two independent patient groups in total.
  Route, formulation and regimen all differ from the PK set.
- **The whole-chain comparison point.** Kapur 1996, seven patients, oral,
  steady-state daily.
- **The free-basis set.** Videbaek, four subjects, IV, free plasma. Disqualified
  for the oral case and used only as split C's out-of-basis probe.

So the chain composes an oral PK model fitted mostly in schizophrenia patients
with an occupancy constant fitted in depot-treated patients, and validates the
composition against a third group of seven oral patients. No subject appears in
more than one of these sets. Whether the sets are exchangeable is **not
determined in the repo**; the honest statement is that the composition assumes
exchangeability and nothing tests it.

Sex, age, ancestry, smoking status and CYP2D6 genotype are not recorded for any
of these sets in this repository. Haloperidol is a CYP2D6 substrate, and none of
the machinery here represents that.

### Route

Oral only. `predict_occupancy` raises on route `IV` because P19.2 recorded the
haloperidol IV search as empty, and raises on any route with no record at all.
There is no IM record of any kind, which is the ironic corollary of the regimen
mismatch: the chain cannot simulate the route its own EC50 was measured on.

### What the pre-registered splits actually establish

Four splits, all against the plasma-to-occupancy hyperbola, all with
`forced_by` recorded on the rule:

| Split | n | MAE | RMSE | bias | max error |
|---|---|---|---|---|---|
| A, leave-one-out, total basis | 4 | 0.0459 | 0.0476 | +0.0020 | 0.0612 |
| B, leave-one-study-out, total basis | 4 | 0.0351 | 0.0376 | +0.0042 | 0.0488 |
| C, free-basis set | 4 | 0.0620 | 0.0818 | +0.0601 | 0.1409 |
| D, external published EC50 | 8 | 0.0939 | 0.1014 | +0.0939 | 0.1637 |

Split B's folds: holding out Nyberg gives MAE 0.0259, holding out Regenthal
0.0444. Split D's folds: Kapur 1997's constant gives MAE 0.0712, Fitzgerald
2000's gives 0.1167, and both biases are positive and equal to their MAEs
because every single error is one-signed.

Standardised errors are not computable for any split: no source printed an SD.
The B < A ordering is worth stating plainly, because it is the opposite of the
usual pattern, because the patient-disjoint split does better than leave-one-out. With
four points from two studies this is a small-sample artefact and should not be
read as evidence that the fit generalises across studies better than within
them.

## Illustrative constants still standing

The P19.1 opt-in gate (commit `2c374ec`) touched exactly three modules: `ke0.py`,
`mor_effect.py`, `gabaa_effect.py`, plus their tests. Those are the constants
carrying a machine-checkable `status="illustrative"` and an
`allow_illustrative=True` opt-in. The rest of this register lists constants that
are unfitted in a human but not gated by that mechanism, because a scope document
that only listed the gated ones would understate the exposure.

### Gated by P19.1

| Constant | Location | Value | What would retire it |
|---|---|---|---|
| `HumanKe0("cognitive_performance", ...)` | `src/neurocascade/human/ke0.py:81` | 0.10 /min | a human PK-PD study fitting an effect-compartment rate constant against a cognitive readout |
| `HumanKe0("motor_activity", ...)` | `src/neurocascade/human/ke0.py:82` | 0.08 /min | the same, against a motor readout |
| `HumanKe0("subjective_sedation", ...)` | `src/neurocascade/human/ke0.py:83` | 0.05 /min | the same, against a sedation scale |
| `HumanKe0("anxiety_state", ...)` | `src/neurocascade/human/ke0.py:84` | 0.04 /min | the same, against an anxiety scale |
| `HUMAN_MOR_RESERVE_PRIOR` | `src/neurocascade/human/mor_effect.py:91-95` | `reserve=1.5`, `emax=1.0` | a matched human MOR assay giving receptor reserve, replacing the prior rather than scaling it |
| `HUMAN_GABAA_OPERATIONAL_PRIOR` | `src/neurocascade/human/gabaa_effect.py:71-75` | `reserve=1.0` | a matched human GABA-A efficacy measurement; the record's own source string says "matched human efficacy pending" |

All four ke0 records carry the source string "human effector prior; fit
required". None is consumed by the dose-to-occupancy chain, which has no effect
compartment at all, which is itself the gap that makes them exist.

### Unfitted in a human, not gated by P19.1

| Constant | Location | Value | What would retire it |
|---|---|---|---|
| `DEFAULT_NODE` (Wilson-Cowan node parameters) | `src/neurocascade/circuit.py:33`, reached via `src/neurocascade/human/build.py:242` | `tau=1.0, baseline_input=0.0, gain=4.0, threshold=0.0` | a fitted human time constant and transfer-function slope. The builder labels them ILLUSTRATIVE in its docstring (`build.py:41`) and prints the label in its report (`build.py:127`), and they are the mouse builder's defaults reused unchanged |
| `DEFAULT_OPERATING_GAIN` | `src/neurocascade/stability.py:202` | 0.9 | behavioural data. The comment says so directly: "which value is right is a calibration this project has no behavioural data for (P2.3: only the product of coupling gain and connection weight is identified)". Consumed as the default at `src/neurocascade/human/scenario.py:177` and labelled ILLUSTRATIVE at `scenario.py:22` |
| `mechanism_gain` | no default in code; documented example at `src/neurocascade/human/receptors.py:124` shows `mechanism_gains=-0.5, # ILLUSTRATIVE: caller-supplied, never fitted` | caller-supplied | a fitted human mechanism gain. `MECHANISM_GAIN_NOTE` in `scenario.py` states as a fixed fact that none has ever been fitted in this repository, so any value a caller passes is illustrative and the provenance is the caller's |
| `fallback_tau` | `src/neurocascade/human/ei_time_constants.py:24` | 1.0 | measured regional E/I time constants. Only returned when a caller passes `allow_fallback=True`; the module keeps measured finite values separate from it |
| `D2Effect.baseline_occupancy`, `D2Effect.emax` | `src/neurocascade/human/d2_effect.py:8-9` | 0.0 and 1.0 | a human dose-response fit. These are not labelled illustrative because they are structural identity settings, but neither has been fitted in a human, and together they are what makes `abs(effect)` equal occupancy |

The chain-level consequence is stated in `predict_occupancy`'s provenance rule:
the prediction's provenance is the weakest claim any layer makes, so a single
illustrative layer would make the whole prediction illustrative. Today none of
the constants above sits inside the dose-to-occupancy chain, which is why it
reports `fitted-in-repo` rather than `illustrative`. The gate exists so that
adding one later cannot enter a prediction unannounced.

## Why nothing here is a behavioural claim

The repository holds its own falsification of the step from occupancy to
behaviour, in `OCCUPANCY_IS_NOT_THE_OUTCOME`:

- Siafis 2023 (PMC10618092, N=37,193) reports haloperidol at 11 mg/day with an
  extrapyramidal odds ratio of 4.7 against risperidone at 16 mg/day with 3.3, at
  approximately the same 87 percent occupancy. Occupancy alone does not
  determine the side effect.
- Kurose 2020 (PMID 32726002) reports central occupancy decaying more slowly
  than plasma concentration, which this chain cannot represent at all.

The clinical thresholds registered in `CLINICAL_OCCUPANCY_THRESHOLDS` and the
doses at which the chain reaches them are reference points, not endpoints:

| Threshold | Source | Occupancy | Steady-state dose | Single dose |
|---|---|---|---|---|
| clinical response | Kapur 2000, PMID 10739409 | 65% | 2.89 mg/day | 3.36 mg |
| hyperprolactinemia | Kapur 2000 | 72% | 4.02 mg/day | 4.65 mg |
| extrapyramidal side effects | Kapur 2000 | 78% | 5.56 mg/day | 6.41 mg |
| extrapyramidal, pooled reanalysis | Uchida 2011, PMID 21694629 | 77-78% | 5.40 mg/day | 6.23 mg |

`CLINICAL_WINDOW_PROVENANCE` records that the familiar "65 to 80 percent window"
is a synthesis no primary paper prints: Farde 1988 (PMID 2892477) prints 65-85%
as a range across eleven antipsychotics, Farde 1992 (PMID 1352677) prints 70-89%
for classical neuroleptics and no number at all for extrapyramidal side effects,
and the 80 upper bound was not found in any primary paper, and it appears to be a
rounding of Kapur 2000's 78. The module uses 78 and cites Kapur 2000 for it.
Uchida 2011's pooled reanalysis of 12 studies and 82 subjects revises the window
to 60-78% and describes 65-80 as something proposed from samples under twenty.

Saying "5 mg reaches the occupancy Kapur 2000 associated with extrapyramidal
side effects" is a statement about a number. Saying "5 mg causes extrapyramidal
side effects" is a behavioural prediction, and this chain does not make it.

## What the chain would need before a behavioural endpoint could hang off it

Six things, in dependency order. Items 1 and 2 are prerequisites for everything
after them.

1. **Human dose-response observations.** There are zero
   `HumanDoseResponseObservation` instances anywhere in `src/`; the only ones in
   the repository are fixtures in `tests/test_human_dose_response.py`.
   `fit_human_log_linear` requires `min_observations=3`, so the fit cannot run at
   all. Nothing downstream of this is reachable until it is filled.
2. **Endpoint weights.** `endpoint_score` raises with "weights are required; no
   human endpoint magnitudes are fitted". Four human endpoints are retained in
   `endpoint_status.py` and two rodent assays are retired; none carries a fitted
   magnitude.
3. **An effect compartment.** The chain maps plasma to occupancy instantaneously.
   Kurose 2020 says that is wrong in a direction that matters for any
   time-resolved behavioural endpoint. This needs a fitted human ke0, which means
   retiring the four illustrative records in `ke0.py` rather than opting into
   them.
4. **A transduction layer that is not an identity.** While Emax is 1 and baseline
   occupancy is 0, `effects()` returns the occupancy curve on a sign convention.
   A behavioural endpoint hanging off that would be hanging off occupancy with
   extra steps.
5. **Between-subject variance.** The PK schema carries no IIV, so every
   prediction is a typical-value prediction with no interval. Pilla Reddy 2013's
   published IIV (CL 44.5 %CV, Vc 116 %CV) would have to enter the schema before
   any statement about a population of patients could carry an uncertainty.
6. **A network layer that is not illustrative.** Any endpoint routed through the
   whole-brain scenario picks up `DEFAULT_NODE`, `DEFAULT_OPERATING_GAIN` and a
   caller-supplied `mechanism_gain`, none fitted in a human. Retiring
   `DEFAULT_OPERATING_GAIN` specifically requires behavioural data, which makes
   it circular with respect to item 1: you cannot calibrate the operating gain on
   behaviour and then predict behaviour with it and call the second thing a
   prediction.

`HUMAN_LAYER_FINDINGS` in `src/neurocascade/human/failure_analysis.py` already
records the state this implies: PK, BBB, receptor, connectivity and
effect-transduction are all `partial`, and behavioural validation is `blocked`.
P19 does not change that classification.

## Extending the same pattern to fentanyl and diazepam

The haloperidol chain took a `measured` PK record and a `fitted-in-repo`
occupancy constant and composed them. Neither drug currently has either half.
Checked programmatically against `missing_fields` rather than by eye:

**PK (`src/neurocascade/human/pk_parameters.py`).** Of the four records in
`HUMAN_PK_RECORDS`, only `haloperidol PO` (lines 223-258) is
`complete_for_model`. The fentanyl IV record (204-209) is incomplete. The
haloperidol IV (210-222) and diazepam IV (259-263) records are searched-and-empty
placeholders: they exist to record that a search happened and returned nothing,
which is why the chain can raise a specific error rather than a generic one.
There is no oral record for either drug.

Prerequisite: a human parameter set printing `V1_L`, `V2_L`, `Q12_L_per_h`,
`CL_L_per_h` and `ka_per_min` for the intended route. `predict_occupancy` refuses
on any one of the five being absent, and this is tested. For fentanyl the route
question is sharper than for haloperidol, since the clinically relevant routes
(IV, transdermal, transmucosal) each need their own absorption model, and the
chain's `ka_per_min` slot describes first-order oral absorption only.

**Receptor (`src/neurocascade/human/receptor_parameters.py`).** The D2 record
(113-184) is the only `usable` one. MOR (185) and GABA_A_BZD (186) both have
`ki_nM=None`, `efficacy=None`, `ec50_in_vivo_nM=None`, `ligand=None`, status
`pending-human-occupancy`, and the note "No rodent affinity is promoted."

Prerequisite: an **in-vivo plasma EC50**, not an in-vitro Ki.
`occupancy_constant_nM()` refuses to substitute a Ki without an explicitly named
correction factor, and P19.4 established why that refusal is not pedantry. Seven
primary human D2 Ki values span 68-fold (0.13 to 8.85 nM), and the resulting
EC50/Ki ratios run 0.213 to 14.50, straddling unity. The in-vitro literature
cannot even agree on the *direction* of the correction. There is no reason to
expect MOR or GABA-A to be better behaved.

Concretely, for each drug, the same four things the haloperidol chain needed:

1. A human PK record with all five parameters, route-matched, with the
   apparent-versus-absolute status stated.
2. A human PET occupancy set at the target receptor, with tracer and region
   printed, and with plasma concentrations measured at the same time. For MOR
   this means [11C]carfentanil or equivalent; for GABA-A benzodiazepine site,
   [11C]flumazenil. Tracer non-interchangeability applies to these systems too
   and the 2-2.5x floor observed for D2 is the reason to record the tracer field
   rather than assume it.
3. A ligand-specific efficacy with a stated reference compound and sign
   convention. The D2 efficacy (-0.95, Hall 1997 cAMP, with -0.83 from Roberts
   2005 GTPγS alongside) comes from a single laboratory, and no source outside it
   reports a number at all. Whether a comparable measurement exists for fentanyl
   at human MOR or diazepam at the human GABA-A benzodiazepine site is **not
   determined in the repo**.
4. A decision about the exposure layer. The haloperidol chain omits a
   brain-partition step because the EC50 was fitted on plasma in living people
   and already contains brain penetration. That reasoning transfers only if the
   new EC50 is fitted the same way. A Ki-derived constant would need a Kp,uu, and
   `neurocascade.human.kp_uu` holds no value for fentanyl or diazepam either.

Diazepam carries one extra obstacle not present for haloperidol: active
metabolites (desmethyldiazepam, oxazepam, temazepam) with their own affinities
and their own kinetics. A single-parent two-compartment model with one occupancy
hyperbola is structurally wrong for it in a way it is not obviously wrong for
haloperidol. This is a modelling decision, not a data gap, and it should be made
before any parameter is entered rather than after.

## Repair order

1. Fill the dose-to-plasma arm's test. Everything else is refinement of an arm
   that was measured; this is the arm that has never been checked. One study
   printing per-subject oral dose, plasma and occupancy would move the chain from
   "one arm tested" to "chain tested".
2. Resolve the regimen mismatch, by finding oral steady-state occupancy data or
   by restricting the chain's published predictions to steady state. The
   Kapur 1996 containment result already shows which way that would go.
3. Get the PK IIV into the schema, so predictions carry an interval.
4. Get region and tracer printed on the fitting observations, or find
   replacements that print them, so the "striatal" label stops being inherited.
5. Retire `DEFAULT_OPERATING_GAIN` and `DEFAULT_NODE` before any endpoint is
   routed through the network layer, being explicit that retiring the operating
   gain on behavioural data forecloses using it to predict behaviour.
6. Only then consider fentanyl or diazepam. Repeating the pattern with weaker
   inputs would produce a chain with the same shape and none of the standing.

## Standing limitations to preserve

- Never substitute another drug's PK for a drug with no record. The
  searched-and-empty placeholders exist so the refusal is specific.
- Never treat an in-vitro Ki as an in-vivo EC50. The 68-fold Ki spread and the
  unity-straddling ratio are the reason, and they are recorded in the D2 record's
  own notes.
- Never pool a total-plasma and a free-plasma basis. The ~11-fold gap is protein
  binding, not biology, and split C shows the one-signed bias that results from
  crossing it.
- Never treat tracers as interchangeable. The 2-2.5x spread across
  [11C]raclopride, [123I]IBZM and [76Br]FLB 457 exceeds every held-out error in
  this document.
- Never apply the striatal constant to an extrastriatal region.
- Never multiply plasma by a Kp,uu *and* use the fitted in-vivo EC50. The chain
  raises if a Kp,uu value appears, and that check should stay.
- Never report the transduction output as a second result while Emax is 1 and
  baseline occupancy is 0. It is the occupancy curve with a sign.
- Never call the steady-state containment result a passing test. It was computed
  after the single-dose miss was seen, against a number already in public view.
- Never call an occupancy a clinical outcome. Siafis 2023 is the falsification,
  it is already in the repository, and it should be cited whenever the threshold
  table is.
- Never let an illustrative constant into a prediction without the
  `allow_illustrative=True` opt-in, and never let the opt-in be the default
  anywhere.
