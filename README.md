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

- **Pharmacokinetics** (`pk.py`) - one-compartment elimination, `dC/dt = -ke*C`.
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

The engine runs end to end on one worked example:
[`examples/fentanyl_reward_circuit.py`](examples/fentanyl_reward_circuit.py)
simulates a mu-opioid agonist (fentanyl) and antagonist (naloxone) acting on
a four-node VTA-NAc-PFC reward circuit (opioid-induced disinhibition of VTA
dopamine neurons). It reproduces the qualitatively correct pharmacology:
agonist raises the behavioral readout, antagonist alone does ~nothing, and
antagonist given after the agonist reverses it, the same mechanism naloxone
overdose rescue relies on.

**The receptor-binding constants, circuit time constants, and connection
weights are hand-set placeholders**, chosen to get the mechanism right, not
sourced from any specific dataset. Calibrating them against real Ki/kon/koff
and connectivity literature is the next phase, the same literature-elicited
priors approach used for the CCT model.

## Quickstart

```
/root/projects/pardalos/.venv/bin/python examples/fentanyl_reward_circuit.py
/root/projects/pardalos/.venv/bin/python -m pytest tests/
```

## Layout

| Path                 | Purpose                                          |
|----------------------|---------------------------------------------------|
| `src/neurocascade/`  | The engine (pk, receptor, circuit, network, behavior, engine, scenarios) |
| `examples/`          | Worked end-to-end scenarios                      |
| `tests/`             | Unit tests per layer                             |
| `docs/`              | Architecture and design notes                    |
| `results/`           | Generated figures                                |

## Environment

Uses the shared Pardalos scientific Python stack (`scipy`, `numpy`,
`matplotlib`, `pytest`) at `/root/projects/pardalos/.venv`, same convention
as `cct-bayesian-validation`.

## Roadmap

- Calibrate receptor and circuit parameters against literature (Ki/kon/koff
  databases, connectivity estimates).
- Add more receptor systems and circuits (dopaminergic D1/D2 on
  striatal direct/indirect pathways, GABA-A on cortical microcircuits).
- Multi-region receptor density maps, so one drug perturbs several network
  entry points at once rather than a single node.
- Replace the linear behavior readout with one fit against real behavioral
  endpoints once available.
- Bayesian calibration (PyMC) of free parameters against literature
  behavioral data, following the CCT model's approach.
