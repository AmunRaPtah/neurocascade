# Roadmap v2: human-primary, receptor → circuit → behavior

## The scope reversal, and why

v1 fixed the species at **mouse**, for a defensible reason: the Allen Mouse Brain
Connectivity Atlas is the only real, quantitative, *directed* whole-brain
connectome, and the behavioral pharmacology to fit against was rodent. See
`roadmap.md`, "Species: mouse".

That reasoning optimized for one layer, the circuit layer, and paid for it in
every other layer. The target has been restated: **neurocascade simulates human
brains.** Rodent data that does not transfer to humans is retired.

The pivot applies a principle the project already held.
`expression/autoradiography.py` states the rule:

> substituting rat autoradiography for a mouse expression map would be exactly
> the species-substitution this project refuses.

That rule was written to protect the mouse model. Applied to a human-primary
model, it retires the mouse layers on the project's own existing terms.

## What the audit found

Five layers, audited against the published human literature (2026-08-21):

| Layer | Human vs rodent | Reason |
|---|---|---|
| Receptor density | **STRONGER** | Measured protein density replaces an mRNA proxy |
| Directed connectivity | **WEAKER** | No human tracer ground truth can exist |
| Pharmacokinetics | **STRONGER** | 104 human fentanyl popPK models; rat has 1 |
| Kp,uu / brain partition | **WEAKER**, the weakest layer | 6 human compounds vs 70 rat |
| Behavioral endpoints | **STRONGER**, except reward and anxiety | Continuous readouts, N to 37,000 |

Three findings drive the phase ordering below.

### 1. The mRNA proxy was indefensible for GABA-A, and this is now measured

Hansen JY, Markello RD, Tuominen L, Nørgaard M, Kuzmin E, Palomero-Gallagher N,
Dagher A, Misic B. "Correspondence between gene expression and neurotransmitter
receptor and transporter density in the human brain." *NeuroImage*
2022;264:119671. PMID 36209794.

Across 27 receptors/transporters over 9 systems, gene expression corresponds to
measured density for **only four single-protein metabotropic receptors: 5-HT1A,
CB1, D2, MOR**. The authors' recommendation, verbatim: *"we recommend using
direct measures of receptor and transporter density."*

Per-gene, from `netneurolab/hansen_gene-receptor`:

| Gene → receptor | cortex ρ | p_spin |
|---|---|---|
| OPRM1 → MOR | 0.846 | 0.0019 |
| DRD2 → D2 | 0.706 | 0.0128 |
| GABRA1 → GABA-A | **0.244** | **0.559** |
| GABRG2 → GABA-A | **0.176** | **0.638** |
| all 19 GABR* subunits | max abs 0.41 | every p_spin ≥ 0.24 |

Read onto v1: the ISH-mRNA density map was quantitatively defensible for MOR and
D2 and indefensible for GABA-A. The benzodiazepine/amygdala system, which carried
the project's only fitted circuit parameter, had the weakest-supported density
map in the build. Retiring the ISH layer corrects this rather than costing
anything.

### 2. Direction is measurable in humans, just not by diffusion

Human dMRI is a symmetric physical process; any direction in a dMRI connectome is
imported from outside. Quantitatively (Maier-Hein 2017, *Nat Commun* 8:1349, PMID
29116093): **36 ± 17% of reconstructed streamlines connect regions that are not
connected**, 88 ± 58 invalid bundles per tractogram, **bundle precision 23 ± 9%**.

**F-TRACT** measures direction causally in living humans. Cortico-cortical evoked
potentials from intracranial stimulation: stimulation site → recording site *is* a
direction.

Lemaréchal J-D, Jedynak M, Trebaul L, et al. "A brain atlas of axonal and
synaptic delays based on modelling of cortico-cortical evoked potentials."
*Brain* 2022;145(5):1653–1667. PMID 35416942. **780 patients, 34,354
stimulations, 774,445 CCEPs**, on HCP-MMP1 (360 parcels). EBRAINS DOI
10.25493/5AM4-J3F, CC BY-NC-SA 4.0; matrices are explicitly asymmetric.

It also supplies **cortical maps of local excitatory and inhibitory synaptic time
constants**, parameters this engine currently labels ILLUSTRATIVE.

Coverage is clinically driven and incomplete. Where F-TRACT has no edge, macaque
tract-tracing (Markov 2014 FLNe/SLN) transferred by connectivity blueprints (Mars
2018, *eLife* 7:e35237, PMID 29749930) supplies a directed prior. **Every edge
carries a provenance flag recording whether its direction was measured or
inferred.** Homology transfer is worst exactly where a human model cares most,
in lateral/medial temporal, ACC and TPJ (Reid 2016, PMID 26515902), so the flag
is load-bearing rather than decorative.

Traps recorded so they are not re-discovered:
- TVB's `connectivity_76`/`_96`/`_192` carry ordinal {0,1,2,3} weights and `RM-`
  prefixes: they are **macaque CoCoMac tracer codes on human coordinates**, widely
  mis-cited as human DSI. Not usable.
- CoCoMac itself is frozen since 2014; a live pull gives ordinal codes with ~44%
  of cells "no data".
- core-nets.org returns HTTP 500; the Markov data mirror is
  `INM-6/multi-area-model` `.../data_multiarea/raw_data`.
- braingraph.org "directed connectomes" derive direction from edge-frequency
  ordering, a graph-statistics heuristic with no anatomical basis.

### 3. The three drug classes need three structurally different transducers

v1 used one receptor→effect structure for all systems. Human occupancy data shows
that is wrong, and the rodent build had no way to discover it.

- **D2 / haloperidol, occupancy-limited.** Effect tracks fractional occupancy
  near-linearly in the 60–85% band. Kapur 1997 (PMID 9201802): EC50 = 0.51 ng/mL,
  80% occupancy at 2.0 ng/mL, r²=0.84; prospectively validated at ED50 0.40 ng/mL
  with 3.89% mean prediction error (Fitzgerald 2000, PMID 10789875). Hill
  coefficient is 1 in every published human D2 fit.
- **μ-opioid / fentanyl, reserve-dominated.** Methadone at 18–90 mg/d produces
  **no displacement and no dose-occupancy relationship at all** (Melichar 2005,
  PMID 15347732). An occupancy→effect map is the wrong structure; use an
  amplification exponent on bound fraction.
- **GABA-A / diazepam, reserve-dominated and efficacy-dependent.** Lorazepam
  0.03 mg/kg IV, a therapeutic dose, occupies **<3% of benzodiazepine sites while
  producing marked sedation** (Sybirska 1993, PMID 8395663), meaning >97% receptor
  reserve. Occupancy is nearly irrelevant to effect magnitude; intrinsic efficacy
  is what varies. The right structure is the operational model of agonism.

A systematic correction is available for the binding layer: model-derived in-vivo
Ki runs **~3× more potent than cloned-human in-vitro Ki, R² = 0.94 across a
100-fold affinity range** (Petersson 2013, *AAPS J* 15(2):533–541, PMC3675734).

## Retirement policy: quarantine, not deletion

Rodent-only material moves to a `legacy/` namespace. It stays importable and
stays tested; it is excluded from the prediction API and from any human scenario
by an enforced species gate.

Deletion was rejected because rodent data retains two uses that a human-primary
model needs:

1. **Cross-species calibration anchors.** Morphine Kp,uu 0.29 rat vs 0.64 human
   (Ederoth 2004, PMID 15025740) and TPA023 occupancy EC50 9 ng/mL human vs 19–30
   rat/rhesus/baboon (Atack 2010, PMID 19779131) are the only matched pairs that
   quantify the transfer error. They are evidence *about* the pivot.
2. **The searched-and-empty records.** Five registries document what was looked
   for and not found. That is a real result and re-deriving it is expensive.

What quarantine forbids is a rodent value reaching a human prediction. The
species gate is a test, not a convention.

## Phases

### Phase 11: species quarantine and the human boundary
Establishes the boundary before any human data lands, so nothing can leak across
it while the tree is in motion.

- **P11.1** `legacy/` namespace; move rodent-only modules and their data.
- **P11.2** Enforced species gate on `fingerprint.predict()` and every scenario
  builder: a parameter whose provenance species is not human raises unless it is
  explicitly declared a cross-species anchor. Test-enforced.
- **P11.3** Retire the rodent behavioral bridge: all 11 datasets in
  `calibration/behavioral_evidence.py`, `decoder.py`'s 5 optogenetic series, and
  `GABA_A_CONNECTION_MODULATION_GAIN = 0.0742`. Record in
  `docs/heldout-failure-analysis.md` that the project's only non-illustrative
  circuit parameter did not survive the pivot, and why.

### Phase 12: human receptor density
- **P12.1** Ingest Hansen 2022 PET atlas (*Nat Neurosci* 25:1569, PMID 36303070;
  `netneurolab/hansen_receptors`, CC BY-NC-SA 4.0). 19 receptors/transporters, 9
  systems, n=1,238. MOR, D2 and GABA-A/BZ are all covered. Cammoun-033 carries 68
  cortical + 14 subcortical + brainstem.
- **P12.2** Absolute-scale anchor from Zilles & Palomero-Gallagher autoradiography
  in **fmol/mg protein** (*Front Neuroanat* 2017;11:78, PMID 28970785;
  `AlGoulas/receptor_principles`). 15 receptors × 44 areas × 3 laminar strata.
  Covers GABA-A and GABA-A/BZ. It omits MOR and D2, which therefore stay
  scale-free parameters with that stated.
- **P12.3** The parcellation gap: **there is no VTA in the Hansen atlas**, since
  it falls inside one undifferentiated brainstem parcel. Define VTA and midbrain
  nodes from Trutti 2021 (PMID 33580320) or CIT168 (Pauli 2018, PMID 29664465),
  neither of which carries receptor density. Record the consequence.
- **P12.4** Retire the ISH→density path; record the gene-expression
  correspondence result as the reason, with the per-gene numbers.

PET maps are **z-scored relative densities rather than absolute Bmax**, because
tracers came from different participants and scanners, and different tracers
report different quantities (BPND vs VT). Spatial topography is constrained;
absolute scale is not.

### Phase 13: directed human connectivity
- **P13.1** Ingest F-TRACT (EBRAINS 10.25493/5AM4-J3F) on HCP-MMP1.
- **P13.2** Per-edge provenance flag: measured / inferred / absent. No edge
  without one.
- **P13.3** Macaque FLNe + SLN prior via connectivity blueprints for uncovered
  edges. Read Huang, Wang, Triebkorn, … Jirsa, bioRxiv 2026-06-17, DOI
  10.64898/2026.06.16.732559 first, since it builds this exact construction and
  validates against F-TRACT.
- **P13.4** Human E/I synaptic time constants from F-TRACT replace the
  ILLUSTRATIVE node parameters.
- **P13.5** Whole-brain stability at human scale, P3.4 procedure.
- **P13.6** Demote the ENIGMA/HCP dMRI matrix to a provenance-flagged fallback.
  Its direction is assigned from source-region transmitter identity, an
  assumption layered on a direction-blind measurement, and must never be reported
  as measured.

### Phase 14: human pharmacokinetics
- **P14.1** `DosedThreeCompartmentPK`. `pk.py` supports 1- and 2-compartment only;
  **every canonical human fentanyl model is 3-compartment** (Shafer 1990, Scott &
  Stanski 1987, McClain & Hug 1980, Bae 2020).
- **P14.2** Human PK parameters for the three drugs. Two traps: Scott & Stanski
  1987 found **no age effect on fentanyl PK**, since the dose reduction with age
  is purely pharmacodynamic, so no age-on-CL covariate may cite it; and the
  Lombardo 2018 curated **fentanyl VDss = 0.89 L/kg is 4–6× below** the label and
  every compartmental model, so it must not be imported blind.
- **P14.3** Kp,uu with an explicit species-transfer factor:
  `Kp,uu_human = Kp,uu_rat × f_species`, f_species ≈ 1 for non-efflux substrates
  and 2–3 for P-gp substrates, exposed as a fitted parameter with those priors.
  Per drug: **diazepam** uses the real human value Kp,uu,CSF = 0.79 (Fridén 2009,
  PMID 19764786; table open in DiVA `diva2:349401`); **haloperidol** needs no
  Kp,uu at all, since inverting the human occupancy curve supplies the same
  quantity; **fentanyl** has no defensible human value, uses ≈1, and is flagged as
  the model's least-supported input.
- **P14.4** Human PK validation. Record the known hole: **no adult diazepam
  population PK model exists**, as all four are pediatric or methodological.

### Phase 15: receptor layer, human-sourced
- **P15.1** Flip defaults already coded in the tree: A2A `species="rat"` to human
  (note caffeine's A1/A2A selectivity flips 0.91 to 0.46, inverting a published
  conclusion); α2 `affinity_state` to `"whole-cell"`; transporters `KI` to
  `KI_HUMAN`.
- **P15.2** Replace the ~7 individual rodent rate constants whose affinities are
  already human (DOR kon, D1 SCH-23390 Kd, 5-HT1A WAY-100635 koff, 5-HT2A
  ketanserin/DOI, α1 kon and koff).
- **P15.3** Full re-parameterization where the tree is wholly rodent: NMDA and
  AMPA (`glutamate.py` contains zero occurrences of "human"), α7, α4β2, GABA-A/BZ,
  CB1.
- **P15.4** In-vivo vs in-vitro Ki correction (Petersson 2013), as an explicit,
  optional, cited transform, never silently applied.

D2, H1, MOR, KOR, β-adrenergic and M1/M2 are already human and transfer
untouched. `test_d2_scenario_calibrated.py` and `test_histamine_scenario.py` are
the reference templates.

### Phase 16: three receptor→effect transducers
- **P16.1** Occupancy-limited transducer (D2). Validation target: Lako 2013
  (PMID 23948784) OccMax 91.9%, OccD50 0.65 mg/day for haloperidol, with formulas
  openly reproduced in PMC10618092.
- **P16.2** Reserve-dominated amplification (μ-opioid). Port the open FDA
  receptor→ventilation model: `FDA/Mechanistic-PK-PD-Model-to-Rescue-Opioid-Overdose`
  (Mann 2022, PMID 35766413). Fentanyl K_on 3.081e-5, K_off 4.331e-3, n 0.8439;
  drive reduced by `CAR^P`. Open subject-level raw data on Dryad
  10.5061/dryad.j3tx95xdb (Moss 2022, 22 participants).
- **P16.3** Operational model of agonism (GABA-A), after Visser 2003 (PMID
  12490579): in-vivo affinity and efficacy separated, with K_PD(unbound) vs
  in-vitro affinity r = 0.93 and e_PD vs GABA-shift r = 0.95.

### Phase 17: human endpoints
- **P17.1** Rebuild the endpoint library on human readouts. Anchors: fentanyl
  isohypercapnic minute ventilation C50 0.42–2.3 ng/mL (Algera 2021, PMC7983936);
  remifentanil miosis EC50 0.541 ng/mL, γ 2.38, RSE 5–10% (Maharaj 2025,
  PMC11867082), the tightest estimates in the opioid PD literature; midazolam
  BIS IC50 70 ng/mL and MOAA/S IC50 80 ng/mL (Wiltshire 2012, PMID 22253270);
  prolactin (Petersson 2013 mechanistic model; Lin 2025 dose-response, N=23,128).
- **P17.2** ke0 is effector-specific and must not be shared across readouts: EEG
  0.147 min⁻¹ (t½ 4.7 min), respiratory t½ke0 16.4–17.2 min, pupil 11.1–12.8 min.
  Do not drive analgesia from the EEG ke0; it is ~10× too fast for real
  analgesic onset (Abad-Torrent 2022, PMID 36207192, n=100).
- **P17.3** Fit endpoints to human dose-response data.
- **P17.4** Retire `anxiety` (EPM), `reward` (CPP), `analgesia_tail_flick` and the
  LORR component of `sedation`, and record why: **these have no human analogue at
  all.** Reward has no published concentration-effect model for these classes in
  humans; the EPM's nearest human analogue is subjective VAS anxiolysis, which is
  among the *less* sensitive benzodiazepine biomarkers. This is a real narrowing
  of scope for the amygdala/GABA-A system and is to be stated, not papered over.

Do not hard-code a single fentanyl C50: the same group's estimates span 0.42 to
2.3 to 7.5 ng/mL and **they are not the same quantity** (effect-site vs
steady-state plasma, isohypercapnic clamp vs room air, physiologic vs
ventilation-only).

### Phase 18: human validation
- **P18.1** Validate the binding layer against human PET occupancy, something v1
  structurally could not do.
- **P18.2** Held-out human drug validation.
- **P18.3** Failure analysis, in the manner of P8.3.

Two falsification tests the model must survive: **occupancy alone does not
determine EPS** (haloperidol 11 mg/d gives OR 4.7 vs risperidone 16 mg/d OR 3.3 at
the same ~87% occupancy), and **central occupancy decays more slowly than plasma**
(Kurose 2020, PMID 32726002), so acute single-dose data needs a direct effect and
multi-day offset needs an effect compartment or explicit slow k_off.

### Phase 19: the haloperidol dose-to-occupancy chain

Added 2026-08-29, after an audit found that Phases 14 to 18 delivered contracts
and registries but almost no pharmacology: of the fifteen new modules under
`src/neurocascade/human/`, exactly one carries a sourced human number (diazepam
`kp_uu = 0.79`). Every human PK parameter and every receptor Ki sits at `None`
behind a gate that always fails, and no test compares a prediction to a measured
human observation because none exists in the tree. The backlog reading 70/70 is
true of tasks and false of science.

Phase 19 closes one gap end to end rather than widening the scaffolding:
**dose to plasma concentration to striatal D2 occupancy, for haloperidol,
checked against measured PET points never used in the fit.** Haloperidol is the
right first chain because human D2 occupancy versus plasma concentration is the
best-measured receptor-occupancy relationship available, which makes this the
first prediction in the repository that can be wrong in public.

- **P19.1** Gate the illustrative constants so `ke0`, the MOR reserve and the
  GABA-A reserve cannot be consumed without an explicit opt-in.
- **P19.2** Transcribe a human haloperidol PK parameter set. The trap is that
  human disposition is usually two-compartment while the record has V3/Q13 slots.
- **P19.3** Register real human D2 PET occupancy observations, per-point tracer
  recorded, through `dataset_manifest`.
- **P19.4** Fit or transcribe the in-vivo plasma EC50 and record the in-vitro Ki
  separately, making the P15.4 correction concrete instead of nominal.
- **P19.5** Wire the chain, identifying brain exposure from the occupancy
  relation rather than an unmeasured partition step.
- **P19.6** Held-out validation, reporting the real error.
- **P19.7** Scope statement and failure analysis to the P8.3 standard.

P19.6 inherits both falsification tests recorded under Phase 18: occupancy alone
does not determine EPS, and central occupancy decays more slowly than plasma
(Kurose 2020, PMID 32726002). The second one bounds the phase directly, since a
single-dose occupancy curve driven by plasma kinetics will mispredict the
multi-day offset unless an effect compartment or slow k_off is made explicit.
Recording that mismatch is an acceptable outcome for the phase; hiding it is not.

## Frozen for v1

Added 2026-09-05, after the audit that found the backlog at 77/77 while the
model could predict one measured human quantity. The three autonomous drivers
generate scaffolding wherever the backlog points them, so pointing them at a
layer that has no data produces contracts, registries and gates rather than
predictions. Phases 14 to 18 are what that looks like at scale.

The layers below are frozen for v1. Freezing is not deletion and not a judgement
that the layer is wrong: it is a statement that the constraint on each is a
number nobody has, so effort spent there buys nothing. Each carries the
condition that would unfreeze it. `tests/test_v1_frozen_layers.py` enforces
this: it imports the v1 entry points in a clean interpreter and fails if any
frozen module is reachable, and it fails if a frozen module loses its entry
here.

| Frozen | Why | What unfreezes it |
|---|---|---|
| `neurocascade.circuit`, `neurocascade.human.ei_time_constants` | every Wilson-Cowan parameter is illustrative; the only fitted one, `GABA_A_CONNECTION_MODULATION_GAIN`, was rat-fitted and retired at the pivot | human-measured E/I time constants and connection weights |
| `neurocascade.network`, `neurocascade.human.matrix`, `neurocascade.human.directed_prior` | whole-brain simulation rests on a connectome whose edge directions are inferred rather than measured, and the Jirsa directed-connectome URL is still unverified | a directed human connectome with measured directions |
| `neurocascade.behavior`, `neurocascade.endpoints`, `neurocascade.human.behavioral_endpoints` | no test anywhere compares a predicted behaviour to a measured one; the human endpoints were retired in P17.4 | a human dataset linking exposure to a graded behavioural measure, with an identifiable Emax and baseline |
| `neurocascade.opioid`, `neurocascade.human.mor_effect` | the mu-opioid chain is scaffolding: PK parameters and receptor Ki are `None` behind a gate | sourced human mu-opioid PK and Ki |
| `neurocascade.human.gabaa_effect` | the operational model of agonism has no human parameters, and P14.4 established that no adult diazepam population PK exists | an adult, not pediatric, benzodiazepine population PK model |

Two consequences bind Phases 21 to 24. No task may resume a frozen layer, which
`test_no_backlog_task_after_phase_20_touches_a_frozen_layer` checks against the
briefs themselves. And v1 claims nothing about behaviour: the deliverable is
oral dose to striatal D2 occupancy against a sourced threshold, which is the one
chain that has survived contact with a held-out number.

What is deliberately not frozen is the off-target binding fingerprint of
Phase 23. It is the only remaining test of the project's founding thesis, the
data for it is free and public, and unlike the circuit layer its blocking
constraint is work rather than an unobtainable measurement.

## Honesty discipline: unchanged

Every rule from v1 carries over verbatim. Never invent a parameter. Never
substitute another drug's, route's or species' measured effect. Never weaken a
test or widen a tolerance to reach green. Record searched-and-empty as a real
result. Label every uncalibrated parameter ILLUSTRATIVE.

One addition, specific to v2: **every parameter carries its provenance species,
and every connectome edge carries whether its direction was measured or
inferred.** Both are test-enforced. The pivot's whole value rests on not silently
importing a rodent number into a human prediction, and on not reporting an
inferred direction as a measured one.

## Known open questions, recorded rather than resolved

- The Jirsa directed-human-connectome repository URL is unverified (bioRxiv
  returned 429 on every attempt). Retry before starting P13.3.
- Whether the Brainbuilder 3D whole-brain autoradiography volumes (Funck 2025,
  PMID 40624139) are downloadable is unverified; only derived validation metrics
  are confirmed released.
- Whether Simcyp 4Brain is base-licence or a paid add-on is unverified.
- LeiCNS-PK3.0 has no author code deposit; the `nlmixr2lib` reimplementation is
  not author-endorsed.
