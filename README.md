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

The engine runs end to end on three worked examples, all reusing the same
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
| `src/neurocascade/stability.py` | Saturation check, closed-form baseline drive, and the contraction certificate for a stable non-saturating rest state (P3.4) |
| `scripts/`           | `fetch_connectome.py` (download + checksum-verify source data), maintenance |
| `data/`              | Gitignored cache of downloaded source data; checksums live in `connectome/sources.py` |
| `examples/`          | Worked end-to-end scenarios                      |
| `tests/`             | Unit tests per layer                             |
| `docs/`              | Architecture and design notes                    |
| `results/`           | Generated figures; `calibration/` posteriors, `identifiability/` P2.3 analysis, `whole_brain_stability.txt` P3.4 verification |

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
  endpoints once available.
- Bayesian calibration (PyMC) of free parameters against literature
  behavioral data, following the CCT model's approach.
