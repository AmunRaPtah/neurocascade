# NeuroCascade: a receptor-to-circuit model of pharmacological effects at whole-brain scale

**Manuscript skeleton — journal-direct research article draft**

This document is a Word-compatible Markdown skeleton for a full research
manuscript. It is organized for direct journal submission rather than a
preprint-first workflow. Bracketed notes identify prose, tables, analyses or
citations still needed in the next drafting pass; they are not scientific
claims. Quantitative claims already written here point to committed repository
artifacts so that the manuscript cannot silently outrun the evidence.

## Proposed abstract

### Background

Pharmacological effects emerge from interactions among drug exposure,
receptor occupancy, distributed circuits and behavioral readouts. A model
that joins these layers must preserve the distinction between measured
quantities, derived quantities and assumptions, particularly when a whole-brain
connectome is used to propagate a local receptor perturbation.

### Methods

NeuroCascade combines literature-sourced pharmacokinetic and binding records,
competitive receptor models, receptor-gene expression maps, a signed mouse
connectome, population-rate circuit dynamics and multi-region behavioral
readouts. The model accepts a binding fingerprint at the receptor layer and
supports whole-brain steady-state and endpoint analyses when the required
exposure, target and mechanism layers are available. Calibration and
identifiability analyses were separated from illustrative circuit parameters.
Numerical integration, sparse-Jacobian methods, vectorized right-hand sides
and a direct fixed-point solver were benchmarked on the same workloads.

### Results

The default regional connectome contains 291 regions and 43,527 non-zero
edges ([P9.1 benchmark](../results/benchmarks/p9_1_benchmark.json)). A
canonical receptor-fingerprint registry and a receptor-layer prediction API
were implemented ([binding-fingerprint report](../results/binding_fingerprints.txt);
[prediction API](prediction-api.md)). The available behavioral evidence
identified one drug-action direction but did not identify connectivity weights
or time constants ([P2.3 identifiability report](../results/identifiability/summary.txt)).
The seven-compound held-out set had zero complete dose-to-behavior chains;
therefore no arbitrary-compound behavioral accuracy estimate is reported
([held-out validation report](../results/heldout/heldout_validation.txt)).

### Conclusions

NeuroCascade provides a reproducible whole-brain receptor-to-circuit
scaffold, a usable receptor-layer prediction interface and a quantitative
account of the data required to close the exposure-to-behavior chain. Its
held-out experiment is a negative model-readiness result, not evidence that
the seven compounds' behavioral magnitudes were predicted. Route-specific PK,
receptor-drivable brain partition, transporter/projection fields, endogenous
tone and matched behavioral measurements are the immediate requirements for a
valid end-to-end test.

### Keywords

Receptor occupancy; pharmacokinetics; connectome; Wilson-Cowan model;
whole-brain modeling; pharmacology; held-out validation; identifiability;
reproducibility.

## 1. Introduction

### 1.1 Motivation

Drug effects are often described as a linear story from receptor binding to a
behavioral phenotype. The intervening steps are neither instantaneous nor
localized: exposure is route- and species-specific, receptor expression is
regionally heterogeneous, circuit interactions are signed and recurrent, and
behavioral assays measure endpoint-specific projections of distributed neural
activity. The central modeling question is therefore not whether a receptor
can be assigned to a circuit, but which links in the chain are measured well
enough to support prediction.

### 1.2 Scope and hypothesis

The hypothesis is deliberately layered: a receptor fingerprint plus a valid
exposure model should predict receptor-level concentration, occupancy and
activation; a validated target map and mechanism gain should then propagate
that perturbation through a signed whole-brain network. The stronger claim —
that an arbitrary administered dose can be converted into a quantitatively
accurate behavioral profile for an unseen compound — is treated as an
empirical test, not as a design assumption.

The model operates at atlas-parcellated population-rate resolution. It is not
a cellular-resolution or spiking simulation and does not claim to simulate
cognition. The mouse connectome and mouse receptor-expression atlas are used
because they provide a shared regional ontology; species transfer is outside
this manuscript's scope ([roadmap](roadmap.md)).

### 1.3 Contributions

1. A provenance-preserving receptor, PK and brain-partition layer with a
   public arbitrary-fingerprint API.
2. A signed, region-wise connectome model that composes multiple target
   layers on one network.
3. An explicit separation of calibrated, derived, assumed and illustrative
   parameter classes.
4. A reproducible numerical benchmark for regional steady-state analysis.
5. A held-out validation and failure analysis that distinguishes missing data
   links, mechanism reachability failures, structural readout failures and
   numerical limitations.

## 2. Materials and methods

### 2.1 Provenance and data policy

Every parameter is either linked to a literature measurement, derived by a
stated equation from measured inputs, or labeled illustrative. Missing,
unsearched and species-mismatched evidence remain distinct. The full
parameter inventory is maintained in the README's
[calibration-status table](../README.md#parameter-calibration-status), and
the architecture record contains the source-level rationale and citations
([architecture](architecture.md)).

### 2.2 Dose-to-brain exposure

Administered dose is converted to plasma concentration using route- and
species-specific compartment models where the required parameters exist.
The receptor layer uses receptor-drivable brain concentration, not total
brain:plasma ratio substituted as though it were unbound exposure. For
diazepam, the committed partition record derives `Kp,uu = 0.918` from a total
`Kp`, plasma unbound fraction and brain unbound fraction; the derivation and
source studies are recorded in the architecture document
([brain-partition section](architecture.md#phase-1-plasma-to-brain-partition-p15)).

The arbitrary-fingerprint API refuses a missing PK route, missing partition or
missing efficacy rather than silently borrowing another compound's value.

### 2.3 Receptor binding and activation

For receptor targets, ligand competition is represented with occupancy
dynamics rather than a static Hill curve. Fingerprint entries carry target,
`Ki`, efficacy and provenance. Where the model requires kinetic rates,
`koff = kon * Ki`; the standing of `kon` is recorded per target as measured,
subtype-borrowed, assumed or illustrative. Transporter targets use their
competitive uptake formulation and are not assigned receptor occupancy by
default. The public interface and worked examples are specified in
[prediction-api.md](prediction-api.md).

### 2.4 Receptor maps and the regional ontology

Receptor-gene expression is joined to the same Allen CCF ontology as the
connectome. Expression energy is converted to a relative density scale; it is
not treated as absolute receptor abundance. Proxy subunits, missing regions
and terminal-field mismatches remain explicit. Independent anatomical checks
are retained in the committed expression validation outputs
([expression coverage](../results/expression_coverage.txt),
[density validation](../results/expression_density_validation.txt),
[receptor-map validation](../results/receptor_map_validation.txt)).

### 2.5 Signed whole-brain network

The default regional network is based on the regionalized mouse connectome of
Knox et al. [1]. Connection signs are assigned from the transmitter identity
of source regions and are aggregated at selectable atlas resolutions. The
regional benchmark records 291 nodes, 43,527 non-zero edges and the exact
source/cache provenance ([P9.1 JSON](../results/benchmarks/p9_1_benchmark.json)).
The edge scale is an operating-scale assumption, not a fitted measurement.

### 2.6 Circuit dynamics and endpoint readouts

Each node follows a bounded Wilson-Cowan-style population-rate equation. Drug
action enters through receptor-specific coupling or connection-modulation
mechanisms. Projection systems add source-nucleus activity, region-wise
transmitter fields, clearance and receptor competition where the necessary
data are available. Endpoints are multi-region readouts for locomotion,
anxiety, analgesia, reward, sedation, working memory and sensorimotor gating.
Endpoint membership and sign are literature-backed; most endpoint weight
magnitudes remain illustrative. The nonlinear decoder comparison is retained
as a negative result in [decoder_comparison.txt](../results/decoder_comparison.txt).

### 2.7 Calibration and identifiability

Calibration was attempted only where the evidence supplied a compatible
exposure, endpoint definition and uncertainty. The GABA-A behavioral fit uses
the diazepam elevated-plus-maze ratio from Graziani et al. [4]. Circuit-layer
weights and time constants were then tested for structural and practical
identifiability using fixed-point algebra, Fisher information and profile
likelihood. The analysis found zero identifiable connectivity weights and zero
identifiable time constants; one drug-action gain was one-sided identifiable
([identifiability summary](../results/identifiability/summary.txt)).

### 2.8 Held-out validation design

Seven compounds — alprazolam, amphetamine, caffeine, clozapine, cocaine,
ketamine and morphine — were excluded from all fitting steps. Each compound
was checked against five links: binding, route/species PK, receptor-drivable
brain partition, behavioral path and observed behavioral data. Quantitative
error was computed only where the chain closed and the observation included
the necessary ratio and dispersion. The resulting attribution is preserved
in the [P8.3 failure analysis](heldout-failure-analysis.md), rather than
recast as a single accuracy score.

### 2.9 Numerical implementation and benchmarks

The original integration baseline, sparse-Jacobian stiff-solver comparison,
vectorized right-hand side and direct root fixed-point solver were benchmarked
on the same regional workloads. Solver agreement is reported relative to the
RK45 reference, and timing is treated as machine-specific. The committed
artifacts are [P9.1](p9.1-benchmark.md), [P9.2](p9.2-stiff-solver.md) and
[P9.3](p9.3-steady-state.md). No timing is presented as a biological result.

## 3. Results

### 3.1 A provenance-aware receptor-to-network scaffold

The completed model joins receptor-layer predictions to a signed regional
network and exposes separate APIs for receptor-level prediction and
whole-brain endpoint analysis. The distinction matters: `fingerprint.predict`
returns PK, brain concentration, occupancy and activation, whereas the
whole-brain builders consume a mechanism product and do not infer that product
from an arbitrary administered dose. See the [API guide](prediction-api.md)
and [P10.1 figure set](figures.md).

**Table 1. Model layers and epistemic status.**

| Layer | Output | Status in this study |
|---|---|---|
| Administered dose → plasma | Route/species concentration trajectory | Partial, literature-backed where available |
| Plasma → brain | Receptor-drivable concentration | Partial; total `Kp` is not substituted for `Kp,uu` |
| Binding fingerprint | Occupancy and activation | Target-specific; efficacy remains caller-supplied |
| Expression → density | Relative regional receptor map | Measured transcript map plus explicit relative-density assumption |
| Connectome → signed network | Regional propagation structure | Published structure/sign; scale illustrative |
| Circuit mechanism | Regional steady state | Mostly illustrative gains and node parameters |
| Endpoint readout | Named behavioral index | Membership/sign literature-backed; magnitudes mostly illustrative |

### 3.2 Whole-brain structure and reproducibility

The regional benchmark provides a reproducible computational reference: node
count, edge count, target set, resolution, solver settings and cached source
are recorded in machine-readable JSON. The P9.3 artifact additionally reports
the speedup and fixed-point agreement for each workload. These are engineering
results that make subsequent biological analyses tractable; they do not
validate the biological parameterization.

**Figure 1.** Region-wise activity strip and whole-brain model output.
Use `results/figures/region_activity_map.png`; caption must say that the
visual is CCF-ordered and is not an anatomical section.

### 3.3 Receptor-layer prediction

The fingerprint API reproduces the receptor-layer round trip for the tested
hand-built ligand records at the reported affinity level. It explicitly
surfaces the cost of supplying only `Ki`: at MOR, the fingerprint-derived
association-rate fallback does not reproduce the measured ligand-specific
residence times ([binding-fingerprint report](../results/binding_fingerprints.txt)).
This result supports an occupancy interface, not a claim that binding alone
determines downstream efficacy or timing.

**Figure 2.** PK and occupancy time course.
Use `results/figures/occupancy_time_course.png`; caption should identify the
diazepam example and state that it ends at receptor-layer occupancy.

### 3.4 Calibration and identifiability

The available behavioral calibration evidence supplies one quantitative
drug/vehicle ratio for the circuit-layer identifiability analysis. The Fisher
information is therefore rank one. Fixed-point structure further makes the
time constants absent from a steady-state endpoint and leaves circuit
parameters in degenerate combinations. The result is not that the circuit
is false; it is that the available endpoint cannot identify its internal
parameters. The full verdict table and profiles are committed under
`results/identifiability/`.

### 3.5 Held-out validation: a negative readiness result

The held-out registry contains seven compounds and zero complete dose-to-
behavior chains. Five occupancy observations cannot be scored because route-
matched PK and receptor-drivable brain partition are missing. Only alprazolam
has three quantitative dose levels with a dispersion; its reachable points
require an unidentified gain and therefore are not fitted predictive errors.
The direction table contains 15 scored observations and 4 matches, but those
counts mix reachable receptor paths, no-tone boundary conditions,
transporter-path gaps and evidence-format limitations. They must not be
reported as a single validated behavioral accuracy metric. See the
[authoritative report](../results/heldout/heldout_validation.txt),
[error table](../results/heldout/heldout_errors.csv) and
[failure analysis](heldout-failure-analysis.md).

**Figure 3.** Dose-response reachability diagnostics.
Use `results/figures/dose_response_curves.png` and state that the lines use
an illustrative gain bound, not a validated fit.

**Figure 4.** Held-out reachable-range diagnostic.
Use `results/figures/heldout_validation_scatter.png` and state that the plot
is not a fitted prediction because the exposure and gain links are incomplete.

### 3.6 Human thesis test: the serotonin–dopamine fingerprint and EPS

*(Human track; see [roadmap-human.md](roadmap-human.md). The sections above
describe the mouse scaffold. This subsection reports the one human-outcome test
the receptor layer was carried forward to make, and its result is the human
track's current headline.)*

The founding falsification is Siafis et al. 2023 (PMC10618092; N = 37,193):
haloperidol (11 mg/day, EPS odds ratio ≈4.7) and risperidone (16 mg/day, ≈3.3)
sit at approximately the same ~87% striatal D2 occupancy yet carry unequal
extrapyramidal risk, so D2 occupancy alone does not determine the clinical
outcome. Phase 23 tested the standard explanation — that the 5-HT2A:D2 occupancy
ratio (the Meltzer serotonin–dopamine fingerprint) explains the residual — as a
pre-registered nested regression on the seven drugs with both a Lako 2013
dose→D2-occupancy formula and a Siafis EPS odds ratio.

The result is **indeterminate.** Adding the fingerprint ratio raises R² from
0.314 to 0.636 with the ratio coefficient carrying the hypothesised negative sign
(−0.397), but the increment is not significant (F(1,4) = 3.54, p = 0.133) and the
design's power to detect even a large effect is 0.20. The 5-HT2A side is
Ki-derived rather than in-vivo (no dose→5-HT2A-occupancy model exists for these
drugs), and the verdict moves on single-drug deletions. Seven drugs cannot settle
the question; ~13 would be needed at the observed effect size (~26 at a
conventionally large one), more than the eight drugs any dose→D2 formula covers.
The full statement, the per-drug inputs and the number that would settle it are in
[`docs/p23-4-fingerprint-result.md`](p23-4-fingerprint-result.md), the regression
in `src/neurocascade/human/fingerprint_eps_regression.py`, and ledger row V12. No
behavioural endpoint is predicted: this is a regression against published odds
ratios, not a simulated behaviour.

## 4. Discussion

### 4.1 What the model establishes

NeuroCascade establishes a reproducible software and data architecture for
joining receptor pharmacology to a signed, atlas-level network. It makes the
interface between measured exposure and downstream mechanism explicit, and it
turns missing links into inspectable status rather than hidden defaults. The
whole-brain model is useful for structural reachability, sensitivity analysis,
and prioritizing measurements even where it is not yet a validated
compound-to-behavior predictor.

### 4.2 What the model does not establish

The study does not establish arbitrary-compound behavioral prediction. In the
human track it does not establish that the serotonin–dopamine fingerprint
explains extrapyramidal risk (§3.6): that test is underpowered at seven drugs and
is reported as indeterminate, neither a confirmation nor a refutation of the
Meltzer hypothesis. It does not establish that a transcript map is an absolute
receptor-density map,
that a connectome edge scale is measured, that illustrative circuit gains are
pharmacological constants, or that a reachable behavioral observation was
predicted when the gain was selected to reach it. It also does not establish
that a remaining receptor path is the mechanism of a transporter ligand.

### 4.3 Relation to prior work

The network backbone follows the regionalized mouse connectome of Knox et al.
[1], with the mesoscale connectome of Oh et al. [2] retained as a related
source. The CCF ontology follows the Allen atlas framework [3]. Receptor and
PK examples are anchored to the primary studies recorded in the architecture
document, including diazepam binding [5], the opioid kinetic/affinity records
[6], and the rat PK sources used by the drug-specific bridge [7–9]. The single
behavioral calibration datum is Graziani et al. [4].

### 4.4 What data would change the conclusion

The repair order is operational. First obtain matched route/species PK and
`Kp,uu` for one held-out compound. Then score occupancy before adding circuit
gains. Next obtain matched behavioral data with dose, route, species, timing,
mean, dispersion and sample size. Only after those links close should gains
be fit. Transporter/projection fields and endogenous adenosine/glutamate tone
must be added before interpreting the affected held-out compounds. The
alprazolam non-monotone anxiety pattern should be tested in a replicated,
uncertainty-resolved experiment before any non-monotone mechanism is added.
This sequence is copied in substance from P8.3's repair order
([held-out failure analysis](heldout-failure-analysis.md)).

## 5. Limitations

This section is intentionally derived from the standing P8.3 document rather
than drafted independently. The final manuscript should retain its layer
attributions:

1. **Dose-to-concentration PK:** all seven held-out compounds lack a usable
   model for the observed route/species pair. The needed repair is measured
   concentration-time data with route-specific `F`, `ka`, `Vd`, `CL` and
   uncertainty.
2. **Brain exposure:** all seven lack a receptor-drivable `Kp,uu` or equivalent
   unbound measurement. Total brain:plasma `Kp` must not be substituted.
3. **Occupancy validation:** five occupancy observations exist, but none can
   be scored until PK and brain partition are matched.
4. **Binding/species alignment:** caffeine and cocaine occupancy evidence is
   mouse while the carried affinity is rat; the ketamine NMDA assignment is
   explicitly probable. Same-species, same-preparation measurements are
   required.
5. **Transporter and projection reachability:** DAT, NET and SERT do not yet
   have whole-brain behavioral fields. Cocaine and the transporter component
   of amphetamine therefore cannot be interpreted through their intended
   mechanisms.
6. **Endogenous tone:** the A2A and NMDA antagonist paths are inert where the
   repository has no measured basal adenosine or glutamate tone. This is a
   boundary condition, not a calibrated no-effect finding.
7. **Behavioral evidence:** only a minority of observations provide a ratio
   and dispersion suitable for a quantitative residual; assay, route, species
   and reporting differences limit comparability.
8. **Behavioral shape:** the alprazolam anxiety series suggests a turnover at
   1 SD but not 1.96 SD. The monotone model cannot produce an inverted-U; a
   mechanistic extension requires replication and out-of-sample comparison.
9. **Circuit gains and endpoint magnitudes:** reachable observations with an
   unidentified gain are not validated predictions. Most gains and endpoint
   weights remain illustrative.
10. **Numerical reachability:** several outer gain sweeps are not converged;
    those boundaries must not be treated as definitive structural refutations
    until the range calculation is stabilized.

The detailed evidence, proposed repairs and standing rules are in
[`docs/heldout-failure-analysis.md`](heldout-failure-analysis.md).

## 6. Reproducibility and data availability

The source code, tests, generation scripts, machine-readable benchmark tables,
held-out CSVs and figure outputs are committed in the repository. Downloaded
Allen data are cached and checksummed rather than duplicated in the source
tree. The figure-generation entry point and figure caveats are documented in
[figures.md](figures.md). The prediction API and environment convention are
documented in [prediction-api.md](prediction-api.md).

**Analysis table plan.** The final manuscript should include:

- Table 1: calibration status by parameter class (from the README table).
- Table 2: regional network and solver benchmark metadata (from P9.1–P9.3).
- Table 3: identifiability verdicts (from `results/identifiability/summary.txt`).
- Table 4: held-out coverage and failure attribution (from
  `results/heldout/heldout_coverage.csv` and P8.3).

## 7. References

The complete source-level bibliography is maintained in
[`docs/architecture.md`](architecture.md). The manuscript's core references
are listed here in the citation form to be expanded and journal-formatted in
the next pass.

1. **Knox2019.** Knox JE, Harris KD, Graddis JD, Whitesell JD, Zeng H, Harris
   JA, Shea-Brown E, Mihalas S. High-resolution data-driven model of the mouse
   connectome. *Network Neuroscience*. 2019;3(1):217–236.
   doi:10.1162/netn_a_00066.
2. **Oh2014.** Oh SW, Harris JA, Ng L, et al. A mesoscale connectome of the
   mouse brain. *Nature*. 2014;508(7495):207–214.
   doi:10.1038/nature13186.
3. **Wang2020.** Wang Q, Ding SL, Li Y, et al. The Allen Mouse Brain Common
   Coordinate Framework: a 3D reference atlas. *Cell*. 2020.
   doi:10.1016/j.cell.2020.04.007.
4. **Graziani2022.** Graziani D, et al. Behavioral effects of a low molecular
   weight peptide fraction from *Phaseolus vulgaris* in rats. *Brazilian
   Journal of Medical and Biological Research*. 2022;55:e12314.
   doi:10.1590/1414-431X2022e12314; PMID 36477952.
5. **Mohler1977.** Möhler H, Okada T. Benzodiazepine receptor binding in rat
   cerebral cortex. *Life Sciences*. 1977;20:2101–2110.
   doi:10.1016/0024-3205(77)90191-6.
6. **Mann2022.** Mann A, et al. Mechanistic opioid receptor occupancy and
   reversal kinetics. *Clinical Pharmacology & Therapeutics*. 2022.
   doi:10.1002/cpt.2696.
7. **Yassen2008.** Yassen A, et al. Rat fentanyl pharmacokinetics and
   pharmacodynamics source used by the committed bridge. Full bibliographic
   details and the source-level parameter extraction are in
   `docs/architecture.md`.
8. **Friedman1986; Gustafsson2019.** Diazepam total brain:plasma and unbound
   brain partition sources used in the committed derivation; see the
   [brain-partition record](architecture.md#phase-1-plasma-to-brain-partition-p15)
   for complete details and caveats.
9. **Sagara2008; Lustyk2023.** Graded diazepam locomotion and analgesia
   evidence used for the endpoint reachability figures; complete citations
   and data extraction are recorded in `docs/architecture.md` and
   `docs/figures.md`.

## Quantitative claim ledger

This ledger is included to make the manuscript auditable during drafting.
Before submission, every new number must be added here or linked directly to
the corresponding committed result.

| Claim in this skeleton | Traceable artifact |
|---|---|
| 291 regional nodes and 43,527 non-zero edges | [`results/benchmarks/p9_1_benchmark.json`](../results/benchmarks/p9_1_benchmark.json) |
| Seven held-out compounds; zero complete chains | [`results/heldout/heldout_validation.txt`](../results/heldout/heldout_validation.txt) |
| Five occupancy observations unscorable because upstream links are absent | [`docs/heldout-failure-analysis.md`](heldout-failure-analysis.md), P8.2 report |
| One calibration-ready behavioral observation; rank-one information | [`results/identifiability/summary.txt`](../results/identifiability/summary.txt) |
| Zero identifiable connectivity weights and time constants | [`results/identifiability/summary.txt`](../results/identifiability/summary.txt) |
| Held-out direction table: 15 observations and 4 matches | [`results/heldout/heldout_errors.csv`](../results/heldout/heldout_errors.csv) |
| Diazepam `Kp,uu = 0.918` derivation | [`docs/architecture.md`](architecture.md#phase-1-plasma-to-brain-partition-p15) |
| Human fingerprint-vs-EPS test indeterminate (R² 0.314→0.636, p=0.133, power 0.20, n=7) | [`docs/p23-4-fingerprint-result.md`](p23-4-fingerprint-result.md); ledger row V12 |

## Author and journal placeholders

- Authors: [insert]
- Affiliations: [insert]
- Corresponding author: [insert]
- Target journal: [insert journal; journal-direct submission]
- Reporting checklist: [insert journal-specific checklist]
- Data/code statement: adapt Section 6 to journal wording.
- Figures: adapt captions in Section 3 and export publication-resolution files.
