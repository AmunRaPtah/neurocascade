# neurocascade

A simulation engine that models how a drug binding a receptor propagates
through brain circuits to produce a behavioral effect. Four coupled layers:

    drug -> receptor occupancy -> circuit dynamics -> inter-circuit network -> behavior

## Premise

Pharmacology usually stops at the receptor (Ki, occupancy, efficacy).
Systems neuroscience usually starts above it (circuit models, connectomes).
**neurocascade** tries to hold the whole causal chain in one simulation: a
drug's concentration over time drives receptor occupancy, occupancy drives
activation of the neurons that carry that receptor, and that activation
propagates through a network of coupled circuits to produce a predicted
behavioral readout, all as one integrated ODE system.

## Layers

- **Pharmacokinetics** (`pk.py`) - one-compartment elimination
  (`dC/dt = -ke*C`), plus one- and two-compartment models driven by an
  administered mg/kg dose with route, bioavailability, absorption rate,
  volume(s) of distribution and clearance. Drug-specific literature values
  live in `drug_pk.py`, never in `pk.py`.
- **Brain partition** (`brain_partition.py`) - converts a plasma
  concentration into the *unbound brain* concentration a receptor actually
  binds, from a published `Kp` plus a brain unbound fraction, or from a
  published `Kp,uu` plus a plasma unbound fraction. The two coefficients are
  not interchangeable and the module refuses to treat them as such: a total
  brain:plasma ratio with no measured `fu,brain` can report a total brain
  concentration and may not drive occupancy. Per-drug numbers live in
  `drug_pk.py`.
- **Receptor** (`receptor.py`) - competitive binding at one receptor site
  across one or more ligands (so an antagonist can displace an agonist
  already bound there); occupancy converts to a signed activation via each
  ligand's efficacy (agonist / antagonist / partial / inverse).
- **Circuit** (`circuit.py`) - a Wilson-Cowan-style population rate node:
  `tau * dr/dt = -r + sigmoid(input)`.
- **Network** (`network.py`) - multiple circuit nodes coupled by a signed
  connectivity matrix; receptor activation enters as extra drive on whichever
  node(s) actually carry that receptor.
- **Behavior** (`behavior.py`) - a named linear readout of circuit rates
  (e.g. "incentive salience" as a weighted combination of NAc and VTA
  dopamine-neuron firing rate).
- **Engine** (`engine.py`) - assembles all of the above into one state vector
  and integrates it with `scipy.solve_ivp` (RK45), the same integration
  approach used in the CCT model.

See [`docs/architecture.md`](docs/architecture.md) for the full design
rationale and the honesty notes on what's illustrative versus calibrated.

## Status: v0

The engine runs end to end on five worked examples, all reusing the same
`ReceptorSite`/`CircuitNetwork`/`BehaviorReadout`/`Simulation` classes:

- [`examples/fentanyl_reward_circuit.py`](examples/fentanyl_reward_circuit.py)
  simulates a mu-opioid agonist (fentanyl) and antagonist (naloxone) acting
  on a four-node VTA-NAc-PFC reward circuit (opioid-induced disinhibition of
  VTA dopamine neurons). Agonist raises the behavioral readout, antagonist
  alone does ~nothing, and antagonist given after the agonist reverses it,
  the same mechanism naloxone overdose rescue relies on.
- [`examples/d2_basal_ganglia_circuit.py`](examples/d2_basal_ganglia_circuit.py)
  simulates a D2 dopamine agonist and antagonist (haloperidol-like) on a
  five-node basal ganglia indirect-pathway circuit, the same
  Gi-coupled-disinhibition mechanism as mu-opioid, relayed one hop further.
- [`examples/gaba_a_benzodiazepine_circuit.py`](examples/gaba_a_benzodiazepine_circuit.py)
  simulates a benzodiazepine (diazepam) and its reversal agent (flumazenil)
  on a GABAergic-interneuron -> amygdala circuit. Benzodiazepines are
  positive allosteric modulators, not orthosteric competitors with GABA, so
  this scenario uses a small additive extension
  (`Simulation(..., connection_modulation=...)`, see
  [`docs/architecture.md`](docs/architecture.md)) that scales an existing
  connection's weight instead of adding drive to a node; `ReceptorSite`
  itself needed no change.
- [`examples/serotonin_raphe_cortex_circuit.py`](examples/serotonin_raphe_cortex_circuit.py)
  runs the 5-HT1A (Gi) and 5-HT2A (Gq) systems of `serotonin.py` on one
  shared raphe/cortex circuit: 8-OH-DPAT suppressing raphe firing through
  the somatodendritic autoreceptor, DOI exciting cortical pyramidal
  neurons, and each reversed by its antagonist (WAY-100635, ketanserin).
  Every binding constant and efficacy there is literature-sourced,
  including two fractional partial-agonist efficacies; the circuit and the
  tool-compound PK are illustrative. The autoreceptor is an open loop
  until Phase 6 supplies transmitter release, and a test asserts that.
- [`examples/dopamine_direct_indirect_pathways.py`](examples/dopamine_direct_indirect_pathways.py)
  runs the D1 (Gs, direct pathway) and D2 (Gi, indirect pathway) systems of
  `dopamine.py` on ONE seven-node cortico-basal-ganglia-thalamic circuit, so
  that a drug acting at both is simulated as one coupled system rather than
  as two models. A D1 agonist and a D2 antagonist move the same locomotor
  readout in opposite directions and partly cancel when given together,
  which only happens because both pathways reconverge on one output nucleus.
  Dopamine's affinity for D2 is ~530x higher than for D1, and that number
  comes entirely from two matched published measurements rather than from
  anything hand-set.

- [`examples/acetylcholine_nicotinic_muscarinic_circuit.py`](examples/acetylcholine_nicotinic_muscarinic_circuit.py)
  runs the nicotinic (alpha4beta2, alpha7) and muscarinic (M1, M2) systems
  of `acetylcholine.py` on one basal-forebrain/cortex circuit. Nicotinic
  receptors are the first in this codebase that a two-state occupancy model
  cannot represent: an agonist drives them into a state that is still
  occupied and no longer conducting, and nicotine desensitizes alpha4beta2
  246-fold below the concentration that activates it. So `receptor.py`
  gained `DesensitizingReceptorSite`, in the same additive way GABA-A
  allosteric modulation gained `connection_modulation`, with the
  desensitization and recovery rate constants literature-sourced from one
  experiment. Mecamylamine is modelled as the non-competitive open-channel
  blocker it is rather than as a competitor, and the muscarinic sites stay
  ordinary two-state sites on purpose.

- [`examples/opioid_kappa_delta_circuit.py`](examples/opioid_kappa_delta_circuit.py)
  adds the kappa and delta receptors of `opioid.py` to the network the
  mu-opioid scenario has used since the start, with every affinity and every
  efficacy from one laboratory's panel on human cloned mu/delta/kappa
  receptors, so the selectivity ratios are ratios of numbers measured in the
  same experiment. Two of those numbers are the point: **naloxone is
  1.79-fold selective for mu over kappa**, so any exposure that reverses a mu
  agonist has also blocked most of the kappa pool, and **fentanyl is a 30%
  partial agonist at kappa**, so above its kappa Kd it activates the receptor
  that opposes its own mu action. Re-running the existing mu scenario with
  those cross-affinities in place moves the reward readout by 1.5% at a
  clinical fentanyl exposure and by 28.9% four orders of magnitude above it;
  all three established mu results survive. A kappa agonist moves the
  scenario's own reward readout in the direction opposite to fentanyl, which
  is the effect Di Chiara and Imperato measured in 1988 with those same two
  drugs. Full tables in
  [`results/opioid_cross_affinity.txt`](results/opioid_cross_affinity.txt).

- [`examples/monoamine_transporter_psychostimulants.py`](examples/monoamine_transporter_psychostimulants.py)
  adds the first target class in this model that is **not a receptor**. A
  transporter produces no signal for a circuit node to read; it clears a
  transmitter, so blocking it raises that transmitter's extracellular
  concentration, and receptors then bind that. The mechanism is a
  Michaelis-Menten mass balance on one pool per transporter, and its central
  prediction falls out of the algebra with every uncertain constant
  cancelling: a pure uptake blocker raises the transmitter **exactly
  `1 + [drug]/Ki`-fold**, independent of Km, Vmax and the basal level.
  Amphetamine is represented as the transporter SUBSTRATE it is rather than
  as a blocker, which is worth 87.5-fold at DAT: it releases dopamine with an
  EC50 of 6.4 nM and blocks uptake with a Ki of 560 nM. Wiring the
  transporter's dopamine pool into the existing calibrated D2 site gives this
  codebase its **first endogenous transmitter tone** (a measured 5.0 nM, which
  puts D2 at 71% occupancy at rest), and with it the first configuration in
  which an antagonist given alone is not inert: haloperidol on its own now
  moves the circuit, because there is finally something for it to displace.
  Full tables in
  [`results/transporter_constants.txt`](results/transporter_constants.txt).

**The circuit time constants and connection weights are hand-set
placeholders**, chosen to get the mechanism right, not sourced from any
specific dataset. The receptor-binding constants and the PK are no longer:
`scenarios.py` carries a separate, literature-calibrated set of `*_
calibrated()` builders alongside the illustrative ones, and the two are
never mixed.

The circuit layer stays illustrative for a measured reason rather than for
want of trying. `results/identifiability/` holds a profile-likelihood and
Fisher-information analysis of every parameter in the one circuit that has
a published behavioural dataset behind it: the node time constants cannot
be identified by *any* endpoint measured after the circuit has equilibrated,
the node gains, baselines and weights enter the observable through only four
combinations rather than eight independent numbers, and a single scalar
observation carries a single degree of freedom. See
[`docs/architecture.md`](docs/architecture.md)'s P2.3 section for what data
would change that; the short answer is a dose-response series rather than
more single-dose studies.

## Quickstart

```
/root/projects/pardalos/.venv/bin/python examples/fentanyl_reward_circuit.py
/root/projects/pardalos/.venv/bin/python -m pytest tests/
```

## Layout

| Path                 | Purpose                                          |
|----------------------|---------------------------------------------------|
| `src/neurocascade/`  | The engine (pk, drug_pk, receptor, circuit, network, behavior, engine, scenarios) |
| `src/neurocascade/connectome/` | Allen mouse connectome + CCF ontology ingestion (P3.1), and the `CircuitNetwork` builder with its neurotransmitter sign rule (P3.2) |
| `src/neurocascade/connectome_scenarios.py` | The hand-built scenarios re-wired from the connectome, with per-edge provenance (P3.3) |
| `src/neurocascade/serotonin.py` | The 5-HT1A and 5-HT2A receptor systems, their literature binding constants and efficacies, and the raphe/cortex scenario they drive (P5.1) |
| `src/neurocascade/dopamine.py` | The D1 receptor, its literature binding constants and efficacies, and the shared direct+indirect basal-ganglia network the calibrated D2 system also couples into (P5.2) |
| `src/neurocascade/noradrenaline.py` | The alpha1, alpha2 and beta adrenoceptor systems, their literature binding constants and efficacies, and the locus-coeruleus/cortex scenario they drive (P5.3) |
| `src/neurocascade/acetylcholine.py` | The nicotinic (alpha4beta2, alpha7) and muscarinic (M1-M5) receptor systems, their literature binding constants, the measured desensitization and recovery rate constants the new `DesensitizingReceptorSite` consumes, and the basal-forebrain/cortex scenario they drive (P5.4) |
| `src/neurocascade/glutamate.py` | The NMDA and AMPA receptor systems, the measured Mg2+ block, and the double-counting resolution that makes them modulate an existing connectivity weight instead of adding drive (P5.5) |
| `src/neurocascade/histamine.py` | The H1 receptor system, its literature binding constants including the first derivation-2 rate pair in this codebase, and the tuberomammillary/cortex scenario it drives (P5.6) |
| `src/neurocascade/cannabinoid.py` | The CB1 receptor system, handled as connection modulation because the receptor is presynaptic and its signal retrograde, with a connection-modulation gain that is a measured IPSC suppression rather than an illustrative constant (P5.6) |
| `src/neurocascade/adenosine.py` | The A2A receptor system on the striatopallidal node the D2 scenario already carries, its literature binding constants, and the record that caffeine is not an A2A-selective drug and that adenosine's own affinity cannot be measured (P5.6) |
| `src/neurocascade/opioid.py` | The kappa and delta receptor systems, the cross-affinities and cross-efficacies of every ligand at all three opioid receptors from one single-laboratory panel, and the record that naloxone is 1.79-fold mu-over-kappa selective and fentanyl a partial kappa agonist (P5.7) |
| `src/neurocascade/transport.py` | The transporter MECHANISM, beside `receptor.py`: Michaelis-Menten uptake, rapid-equilibrium competitive inhibition, substrate-driven efflux, and one ODE state that is a concentration rather than an occupancy (P5.8) |
| `src/neurocascade/monoamine_transporters.py` | DAT, SERT and NET: mouse uptake kinetics and basal levels, cited constants for cocaine, methylphenidate, amphetamine, fluoxetine and atomoxetine, the substrate/blocker distinction as two measured constants, and the documented Phase 6 transmitter-concentration interface (P5.8) |
| `src/neurocascade/projection.py` | The projection-system MECHANISM: a source nucleus's firing rate driving release into a region-wise transmitter field, one ODE state per target region, plus `RegionalReceptorSystem`, which gives one receptor type an occupancy block per region instead of a single node (P6.1) |
| `src/neurocascade/projection_systems.py` | The nine source nuclei by CCF acronym and id (VTA, SNc, DR, CS, LC, NDB, PPN, LDT, TM), their clearance models, the cited volume-transmission timescale argument, and the constants recorded as illustrative or searched-and-empty (P6.1) |
| `src/neurocascade/projection_scenarios.py` | The hand-built monoamine, cholinergic and histaminergic scenarios ported onto projection systems, with the closed autoreceptor loop and the non-inert lone antagonist (P6.1), the D2 autoreceptor on the VTA and SNc, and the open/closed loop comparison the stability check is run on (P6.3) |
| `src/neurocascade/projection_density.py` | Per-system projection densities DERIVED from the Phase 3 connectome (committed as `projection_densities.csv` with its source checksums), the reference region each transmitter's measured basal level belongs to, the refused nuclear block with its evidence, and the validation of the predicted concentration ratios against held-out measurements including the [Lin2023] whole-brain fibre map (P6.3) |
| `src/neurocascade/endpoints.py` | The seven named behavioural endpoints (locomotion, anxiety, analgesia, reward, sedation, working memory, sensorimotor gating) as multi-region readouts over CCF regions, 69 terms each carrying the lesion/inactivation/opto/chemogenetic experiment that placed its region and fixed its sign, the 16 terms the literature contests recorded and excluded, the tail-flick / hot-plate assay split six papers force, and every weight MAGNITUDE labelled illustrative pending P7.3 (P7.1) |
| `src/neurocascade/decoder.py` | The held-out model comparison that decides whether a nonlinear decoder beats P7.1's linear readout, per endpoint: the argument that only a graded SINGLE-REGION manipulation can identify a decoder nonlinearity (a saturating dose-response cannot, because the circuit layer's own sigmoid already produces one), four model families compared by leave-one-out cross-validation under two weightings, the five published graded series that exist outside a figure, and the searched-and-empty record for the four endpoints with none. Nothing was adopted; every rejection carries its reason (P7.2) |
| `src/neurocascade/endogenous_tone.py` | The five projection systems' cited basal extracellular transmitter concentrations with species, region, method and caveats, the tonic occupancy and Gaddum dose ratio each imposes at every receptor that binds them, and the re-run comparison quantifying what P6.2 changed in P6.1's results (P6.2) |
| `src/neurocascade/stability.py` | Saturation check, closed-form baseline drive, and the contraction certificate for a stable non-saturating rest state (P3.4), plus `assess_system`, which runs the same four checks on the whole coupled ODE system so that a feedback loop leaving the circuit through a transmitter concentration is visible to it (P6.3) |
| `src/neurocascade/expression/` | Allen ISH receptor-gene expression ingestion over the same CCF ontology (P4.1), the expression-to-density conversion with its assumption, its measured error against mouse autoradiography, and the terminal-field mismatch registry (P4.2), and the cited anatomical validation of the resulting maps (P4.3) |
| `src/neurocascade/datafetch.py` | Checksummed download/verify/cache, shared by both ingestion packages |
| `scripts/`           | `fetch_connectome.py`, `fetch_expression.py` (download + checksum-verify source data), maintenance |
| `data/`              | Gitignored cache of downloaded source data; checksums live in `connectome/sources.py` and `expression/sources.py` |
| `examples/`          | Worked end-to-end scenarios                      |
| `tests/`             | Unit tests per layer                             |
| `docs/`              | Architecture and design notes                    |
| `results/`           | Generated figures; `calibration/` posteriors, `identifiability/` P2.3 analysis, `whole_brain_stability.txt` P3.4 verification, `expression_coverage.txt` P4.1 coverage, `expression_density_validation.txt` P4.2 density validation, `receptor_map_validation.txt` P4.3 anatomy validation, `opioid_cross_affinity.txt` P5.7 cross-affinity re-run, `transporter_constants.txt` P5.8 transporter constants, `projection_systems.txt` P6.1 projection systems, `endogenous_tone.txt` P6.2 endogenous transmitter tone, `projection_densities.txt` P6.3 derived densities, validation and closed-loop stability, `behavioral_endpoints.txt` P7.1 endpoint library at three resolutions, `decoder_comparison.txt` P7.2 decoder model comparison and its rejections |

### Connectome data

The Allen matrices are downloaded, not committed. The tests run offline off
small real-data fixtures, so this is only needed for whole-brain work:

```
/root/projects/pardalos/.venv/bin/python scripts/fetch_connectome.py
/root/projects/pardalos/.venv/bin/python scripts/fetch_connectome.py --list
```

Every file is verified against a committed SHA-256; a mismatch fails the
fetch rather than being accepted. See `docs/architecture.md`, "Phase 3:
ingesting the Allen mouse connectome (P3.1)".

With the cache present, a whole-brain network at any of three resolutions
(291 regions, 60 subdivisions, 12 major divisions) is two calls:

```python
from neurocascade import connectome

strength, voxels = connectome.load_knox_inputs()
built = connectome.build_circuit_network(
    strength, voxel_counts=voxels, resolution="major_divisions"
)
print(built.report.summary())   # sign coverage, units, what was excluded
```

Excitatory/inhibitory sign comes from region neurotransmitter identity
(`connectome/neurotransmitter.py`), which is cited per rule; regions the
rule cannot reach are left unsigned and reported rather than guessed. Read
"Phase 3: building the circuit from the connectome (P3.2)" in
`docs/architecture.md` before using the output: at `weight_scale=1.0` the
published weights saturate the illustrative sigmoid (218 of 291 nodes), so a
network built with all defaults is not ready to simulate.

### Baseline stability at scale

`stability.py` makes that a check rather than a surprise, and supplies the
rest state it is missing:

```python
from neurocascade import stability

network = stability.stabilize(built.network).network   # ready to simulate
print(stability.assess(network).summary())             # every node, every check
```

`stabilize` rescales every weight by one scalar so the network's
dimensionless loop gain is `DEFAULT_OPERATING_GAIN` (ILLUSTRATIVE, 0.9 -
below 1 the fixed point is provably unique and globally attracting), then
solves in closed form for the `baseline_input` that puts every node at its
most sensitive operating point. Relative weights are untouched. `assess`
integrates from several initial conditions and reports saturation per node,
local stability, and the certificate. On the published 291-region network
the result is a unique, stable, non-saturating rest state reached from every
start; `results/whole_brain_stability.txt` is the recorded run, produced by
`scripts/check_whole_brain_stability.py`. See "Phase 3: whole-brain baseline
stability (P3.4)" in `docs/architecture.md`.

### Receptor-gene expression data

The Allen ISH expression energies are pinned API queries, downloaded and
checksum-verified the same way, into a separate cache:

```
/root/projects/pardalos/.venv/bin/python scripts/fetch_expression.py
/root/projects/pardalos/.venv/bin/python scripts/fetch_expression.py --missing
```

`--missing` prints the searched-and-empty record: of the 43 receptor and
transporter genes the model uses or will use, 42 have ISH data and `Adra1b`
has none in any Allen mouse product. Nothing is substituted for it.

```python
from neurocascade import connectome, expression

ontology = connectome.load_ontology()
matrix   = connectome.load_connectome(ontology=ontology)
atlas    = expression.load_expression_atlas(ontology=ontology)

energy = atlas.for_connectome(matrix, plane="coronal-else-sagittal", combine="mean")
energy.values              # (42 genes, 291 regions), NaN where not measured
energy.for_gene("Oprm1")
```

`plane` and `combine` are required arguments with no default: a gene has
several ISH experiments in two sectioning geometries that disagree by up to
~30-fold in small structures, so choosing among them is a modelling decision
that P4.2 owns. 279 of the connectome's 291 regions carry ISH data;
`expression.MISSING_FROM_ISH` names the twelve that do not and
`docs/architecture.md`, "Phase 4: ingesting Allen receptor-gene expression
(P4.1)", explains why. Expression energy is transcript abundance in
arbitrary units and not a receptor density; the conversion, its assumption
and its measured error are P4.2, below.
`results/expression_coverage.txt` is the committed record of per-gene
coverage, the missing regions, and how much a gene's experiments disagree.

### Expression as a receptor density, and how far that holds

```python
from neurocascade import expression

rho = expression.relative_density(energy, "Drd1", normalization="max")
rho.as_coupling(mechanism_gain=-3.0, nodes={"striatum": 672})
```

The conversion assumes, within one gene, that density is proportional to
expression energy. That is stated as an assumption in
`expression.ASSUMPTION` and measured against published mouse receptor
autoradiography: over ten regions with measured D1 densities (Yoo et al.
2010) the map **ranks regions well** (Spearman 0.71) and **scales them
badly** (proportional R^2 0.31; log-log slope 0.46 where proportionality
needs 1.0). Across seven receptor systems (Lothmann et al. 2021) the quality
is receptor-specific: alpha-1 and GABA-A track measured density, AMPA does
not track it at all.

Some receptors are not in the region that transcribes them.
`expression.MISMATCHES` records those with citations, and
`relative_density()` refuses to convert `Slc6a3`, `Cnr1`, `Slc6a4` or
`Slc6a2` without `acknowledge_mismatch=True`: the dopamine transporter's
transcript puts 119x as much in SNr as in CP while [3H]mazindol binding puts
0.7x as much, a 172-fold misstatement that no correction here can fix.
`results/expression_density_validation.txt` is the committed record.

### Is it the map of the gene it claims to be

```python
from neurocascade import connectome, expression

atlas   = expression.load_expression_atlas(ontology=connectome.load_ontology())
results = expression.check_atlas(atlas)      # eighteen cited claims
```

A transposed matrix, a gene column off by one and a structure-id mis-join
all produce a map with the right shape and plausible numbers, so the maps
are checked against independently published anatomy: **Drd1, Drd2 and
Adora2a striatum-dominant, Oprm1 peaking in the medial habenula, Gabra1
cortical and pallidal rather than striatal, Cnr1 hippocampal**. Every
expectation carries its citation, species, method and whether it measured
mRNA or protein; claims are ordinal only, because P4.2 measured this
conversion to be a good ordering and a bad scale.

Seventeen of the eighteen behaved as recorded, including two that are
expected to **fail** and assert that they do: Cnr1's receptors sit on
striatal axon terminals in SNr and GPe (Herkenham et al. 1991) and D1's on
striatonigral terminals, so neither transcript map can reproduce the
protein distribution. The eighteenth is a 0.6% difference between two
regions at the floor of the Oprm1 map.
`results/receptor_map_validation.txt` is the committed record.

The first consumer is the multi-region mu-opioid scenario, whose
hand-written density map is now the Allen Oprm1 one:
`coupling = mechanism_gain * rho`, with rho = 1.00 (VTA), 2.52 (PAG), 4.27
(LC) and the gain still illustrative.
`scenarios.mu_opioid_multi_region_coupling_handwritten()` keeps the old map
as the regression baseline, and every qualitative result is asserted under
both.

### Scenarios on connectome regions

The hand-built scenarios are also runnable with their weights replaced by
published ones. Every entry of a ported network carries a provenance, and
the two categories that stay ILLUSTRATIVE (edges inside a single CCF region,
and edges out of a region with no transmitter sign) are exactly the ones the
connectome structurally cannot supply:

```python
from neurocascade import connectome_scenarios as cs

circuit = cs.scenario_connectome()          # or source="fixture", offline
ported = cs.port_scenario(cs.SCENARIO_PORTS["mu_opioid_reward"], circuit)
print(ported.report.summary())              # provenance, contradictions, costs
```

Agonism, inert antagonism and antagonist reversal survive the substitution
for the mu-opioid and GABA-A scenarios. The D2 basal-ganglia scenario's
result does not, and the diagnosis - a pallidothalamic projection the
textbook diagram does not have - is in "Phase 3: porting the hand-built
scenarios onto connectome regions (P3.3)" in `docs/architecture.md`. The
hand-built networks in `scenarios.py` are untouched; they are what the
comparison is against.

## Environment

Uses the shared Pardalos scientific Python stack (`scipy`, `numpy`,
`matplotlib`, `pytest`) at `/root/projects/pardalos/.venv`, same convention
as `cct-bayesian-validation`.

## Roadmap

- Calibrate receptor and circuit parameters against literature (Ki/kon/koff
  databases, connectivity estimates). Now covers mu-opioid, D2, and GABA-A;
  none of the three are calibrated yet.
- Add more receptor systems and circuits (e.g. D1 on the striatal direct
  pathway, GABA-A on cortical microcircuits beyond the amygdala).
- Multi-region receptor density maps, so one drug perturbs several network
  entry points at once rather than a single node.
- Replace the linear behavior readout with one fit against real behavioral
  endpoints once available. P7.1 built the named endpoint library
  (`endpoints.py`): seven multi-region readouts whose region membership and
  sign are cited, and whose weight magnitudes are all still illustrative.
  P7.2 (`decoder.py`) asked whether a nonlinear decoder beats that linear
  readout on held-out data, endpoint by endpoint, and the answer was no
  everywhere: five graded single-region series exist in the literature with
  per-level numbers outside a figure, and all five were rejected. P7.3 fits
  the magnitudes.
- Bayesian calibration (PyMC) of free parameters against literature
  behavioral data, following the CCT model's approach.
