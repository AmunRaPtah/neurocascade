# Architecture

## Why one coupled ODE system instead of separate stages

It's tempting to simulate each layer separately: solve for drug PK, feed the
concentration curve into a receptor-binding calculation, feed that into a
circuit simulation, and so on. That works for a constant or slowly-varying
concentration, but drug concentration, receptor occupancy, and circuit state
all evolve on overlapping timescales, and occupancy depends on how much
receptor is currently free, which depends on circuit-driven feedback in
richer models. Treating the whole chain as one state vector integrated
together keeps the causality honest: everything reacts to everything else's
*current* state, not to a precomputed trace from an earlier, disconnected
simulation.

## State vector

For `d` competing ligands at one receptor site and `n` circuit nodes:

```
y = [C_1, ..., C_d,          # plasma concentration per ligand
     occ_1, ..., occ_d,       # fractional receptor occupancy per ligand
     r_1, ..., r_n]           # firing rate per circuit node
```

`Simulation.rhs(t, y)` in `engine.py` computes the derivative of the whole
vector in one pass and hands it to `scipy.integrate.solve_ivp` with
`method="RK45"`, mirroring the integration approach used in the CCT model
(`solve_ivp` + RK45 over a coupled nonlinear system).

## Receptor layer: why competitive occupancy, not a Hill equation

A simple Hill-equation occupancy curve (`occ = C / (C + EC50)`) is fine for a
single ligand at steady state, but it can't represent what happens when a
second ligand is added to a receptor pool that's already partly occupied by
the first, which is exactly the situation that matters most in this domain
(naloxone reversing fentanyl, an antagonist blocking an agonist mid-course).
`ReceptorSite` instead tracks one occupancy value per ligand from a shared
free-receptor pool:

```
d(occ_i)/dt = kon_i * C_i * (1 - sum(occ)) - koff_i * occ_i
```

so ligands actually compete for the same finite receptor, and the model can
answer "what happens if you give the antagonist five minutes after the
agonist" without any special-casing.

Activation is a signed sum: `activation = sum(occ_i * efficacy_i)`. Setting
`efficacy = 0` gives a neutral antagonist (blocks without activating);
`efficacy = 1` a full agonist; fractional values a partial agonist; negative
values an inverse agonist. This one number captures the pharmacological
category without a separate branch of code per ligand type.

## Circuit layer: Wilson-Cowan rate model

Each circuit node is a population-rate unit:

```
tau * dr/dt = -r + sigmoid(gain * (input - threshold))
```

`input` is the sum of the node's own baseline drive, weighted input from
other nodes (via the network's connectivity matrix), and, for whichever
node(s) actually carry the receptor in question, the receptor's activation
term. This is a standard, well-understood mean-field abstraction for
population activity: fine-grained enough to represent excitation/inhibition
and time-constant differences between populations, coarse enough to keep the
whole multi-region model tractable as one ODE system.

## Network layer: connectivity as a signed matrix

`CircuitNetwork.connectivity[i, j]` is the weight of the projection from
node `j` onto node `i` (positive = excitatory, negative = inhibitory). This
is deliberately just a matrix, not a graph library dependency: the
`v0` scenario only needs four nodes, and a bigger connectome can still be
expressed the same way (see `docs/build-slice.md`-style handoffs in sibling
projects for the "core stays dependency-light" convention this follows).

## Behavior layer: linear readout by design

`BehaviorReadout` is a named weighted sum over node rates. This is the
simplest thing that could possibly work, and that's the point for v0: it's
interpretable and checkable by hand (you can predict its output from the
circuit weights alone), so it won't hide a modeling mistake behind a fitted
black box. The natural upgrade path, once real behavioral endpoint data
exists to fit against, is a learned nonlinear decoder; that's out of scope
until there's something real to calibrate it to.

## What v0 gets right, and what it doesn't (yet)

**Mechanistically faithful:**
- Mu-opioid receptors are Gi-coupled and expressed on VTA GABAergic
  interneurons that tonically inhibit VTA dopamine neurons; an agonist there
  suppresses the GABA node, disinhibiting dopamine firing. That's the
  textbook opioid-reward mechanism and it's what the coupling in
  `scenarios.mu_opioid_coupling()` encodes.
- Competitive antagonism at a shared receptor pool is exactly how naloxone
  reverses opioid effects: it doesn't undo anything downstream, it takes the
  receptor back.

**Explicitly not yet real:**
- Circuit `tau`/connection weights/gains in `reward_circuit_network()` and
  the coupling gain in `mu_opioid_coupling()` are still hand-set to get
  qualitative behavior right, not fit to any specific dataset. They should
  not be cited as measured values. No literature target for VTA-NAc-PFC
  connection *weights* (as opposed to wiring direction/sign, which matches
  known circuit anatomy, see above) was identified during calibration;
  fitting these is future work.
- Only one receptor site and one circuit are modeled. A real drug usually
  hits several receptor subtypes across several regions simultaneously; the
  classes here (`ReceptorSite`, `CircuitNetwork`, coupling dict) are general
  enough to support that, but it hasn't been built or tested yet.

**Calibrated as of Track A (receptor/PK only, circuit still illustrative):**
`scenarios.py` now has a second, clearly-separated parameter set,
`mu_opioid_ligands_calibrated()` / `mu_opioid_receptor_site_calibrated()` /
`mu_opioid_pk_models_calibrated()`, grounded in published data instead of
hand-tuned placeholders (full citations and derivations are inline in that
section of `scenarios.py`):
- **Ki**: fentanyl 1.346 nM, naloxone 1.518 nM, both from the same
  standardized human-mu-opioid-receptor binding assay (Volpe et al. 2011,
  Regul Toxicol Pharmacol, doi:10.1016/j.yrtph.2010.12.007), so the two
  numbers are directly comparable.
- **koff**: fentanyl 0.004331 s^-1 (residence time ~160 s), naloxone
  0.03959 s^-1 (residence time ~17.5 s), both from in vitro binding-kinetics
  fits in an FDA mechanistic PK-PD model of naloxone reversal (Mann et al.
  2022, Clin Pharmacol Ther, doi:10.1002/cpt.2696; parameters public at
  github.com/FDA/Mechanistic-PK-PD-Model-to-Rescue-Opioid-Overdose).
  **kon is not directly reported in compatible units by any source found**;
  it's derived from these koff values and the Ki's above via
  Kd ~= Ki = koff / kon, a standard approximation, not an independently
  measured number, and flagged as such where it's used.
- **PK**: fentanyl ke from the FDA label's redistribution-phase half-life
  (13 min); naloxone ke from a mean reported IV serum half-life (~64 min,
  Handal et al. 1983; Rzasa Lynn & Galinkin 2024). Both real drugs follow
  multi-compartment kinetics (fentanyl distribution/redistribution/terminal
  half-lives of 1.7/13/219 min per the FDA label); collapsing each to one
  exponential for this codebase's one-compartment `OneCompartmentPK` is an
  explicit, documented simplification chosen to match the acute-bolus
  timescale this scenario simulates, not a claim that either drug is truly
  one-compartment.
- **Validation**: `tests/test_scenarios_calibrated.py` checks the
  calibrated fentanyl ligand reaches near-saturating occupancy (>90%) at a
  clinically-cited overdose concentration (50 ng/mL, ~148.6 nM), consistent
  with the 97% baseline occupancy an independent QSP model (Moss et al.
  2020, PLoS ONE, doi:10.1371/journal.pone.0234683, using the same 1.35 nM
  fentanyl Ki) reports at that exposure level. That comparison is
  directional/order-of-magnitude, not an exact match: Moss et al.'s
  occupancy is computed from an internal brain-concentration submodel this
  codebase doesn't have.
- **A genuinely interesting real finding surfaced by calibration**: the
  illustrative placeholder assumed naloxone had the slower koff ("stickier"
  binding) of the two ligands, to make displacement easy to reason about.
  The real data says the opposite: naloxone's koff is about 9x FASTER than
  fentanyl's, despite similar affinity. Naloxone reversal in the calibrated
  model still works (see `test_naloxone_reverses_calibrated_agonist_effect`)
  because clinical naloxone dosing puts it in large molar excess over the
  fentanyl present, not because it binds tighter or lingers longer. This
  is also the real pharmacological basis for renarcotization risk after
  naloxone reversal, which this one-compartment PK collapse cannot
  represent (see `mu_opioid_pk_models_calibrated()` for detail).
- The reward circuit itself (VTA-NAc-PFC wiring, gains, taus) is reused
  unchanged from the illustrative scenario in the calibrated one: only
  receptor binding and PK were in scope for this calibration pass.
