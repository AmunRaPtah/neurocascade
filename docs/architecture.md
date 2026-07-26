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

This is the single-receptor-site case. "Fourth extension: multiple
receptor systems on one shared circuit" below describes how this layout
generalizes when a `Simulation` combines several receptor sites on one
shared circuit at once.

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

## Second scenario: D2 dopamine receptor / basal ganglia indirect pathway

`scenarios.d2_*` reuses every class unmodified. D2 receptors are Gi-coupled,
same as mu-opioid, just sitting on a different node in a different circuit:
striatal "indirect pathway" medium spiny neurons (MSNs) instead of VTA
GABAergic interneurons. Dopamine acting at D2 suppresses those MSNs, which
disinhibits GPe, which inhibits STN, which reduces excitatory drive onto
GPi, which disinhibits thalamus, so the same "Gi-coupled receptor
disinhibits something two or three synapses downstream" motif as
mu-opioid, just relayed one hop further. `d2_dopamine_ligands()` gives
dopamine efficacy=1.0 (agonist) and haloperidol efficacy=0.0 (antagonist),
competing at one shared D2 pool exactly like fentanyl/naloxone at mu; the
readout (`locomotor_activity_readout()`, weighted on the `Thalamus` node)
reflects the indirect pathway's real role in motor facilitation, and is
deliberately not framed as a "psychosis index": this scenario models the
mesolimbic drug class's motor/basal-ganglia side, which is also the
mechanistic link to antipsychotics' extrapyramidal side effects, not the
mesolimbic antipsychotic-efficacy side.

One tuning lesson worth recording: the first parameter draft made GPi
saturate near its sigmoid's ceiling at baseline (net input far outside
[-1, 1] before the gain multiplier), so it stopped responding to anything
upstream of it and the agonist/antagonist barely changed the readout at
all. `basal_ganglia_indirect_pathway_network()`'s `baseline_input` values
are chosen so every node sits near the sensitive part of its sigmoid at
its own steady state; this is a generally applicable check for any new
multi-node relay chain added to this codebase, not a one-off fix.

## Third scenario, and why it needed a small engine extension: GABA-A / benzodiazepine

Benzodiazepines (e.g. diazepam) are a different pharmacological shape than
mu-opioid or D2: they do not compete with GABA for GABA's own orthosteric
binding site. They bind a separate allosteric site on the GABA-A receptor
and act as a positive allosteric modulator (PAM): GABA still gates the
channel, and diazepam just makes GABA more effective at it. Flumazenil, by
contrast, really does compete with diazepam at that separate benzodiazepine
site, which is the actual mechanism of benzodiazepine-overdose reversal, but
it does very little on its own since it isn't displacing GABA from
anywhere.

That splits into two questions the codebase had to answer honestly before
adding this scenario:

1. **Does the diazepam/flumazenil interaction fit `ReceptorSite` as-is?**
   Yes, unmodified. Diazepam and flumazenil genuinely share one pool at the
   benzodiazepine site, exactly the competitive-occupancy relationship
   `ReceptorSite` already models for fentanyl/naloxone. No change to
   `receptor.py` was needed.
2. **Does coupling that occupancy into circuit dynamics fit the existing
   `Simulation.coupling` mechanism?** No. `coupling` adds receptor
   activation directly and additively to a node's drive, which is right for
   a receptor that itself gates a current (mu-opioid, D2), but wrong for an
   allosteric modulator that has no drive of its own: it only changes the
   *gain* of an already-present GABAergic synapse. Using `coupling` here
   would have modeled diazepam as if it had an independent excitatory or
   inhibitory action, which is not how benzodiazepines work and would have
   been a mechanistically dishonest shortcut just to reuse the existing
   mechanism.

So `engine.py` gained one small, additive-only extension:
`Simulation(..., connection_modulation={...})`, a dict of
`(post_node, pre_node) -> gain`. Each evaluation, the targeted connectivity
entry is scaled by `1 + gain * activation` (via a new optional
`connectivity` override argument on `CircuitNetwork.rhs`, defaulting to
`self.connectivity` so every existing call site is unaffected) instead of
being added to a node's drive. This is a `Simulation`/`CircuitNetwork`-level
change only; `ReceptorSite`/`Ligand` are untouched, and a receptor can use
`coupling`, `connection_modulation`, both, or neither, depending on its
actual mechanism.

`scenarios.gaba_a_*` puts this to use in a local-interneuron -> basolateral
amygdala (BLA) -> central amygdala (CeA) circuit: local GABAergic
interneurons tonically inhibit BLA principal neurons, whose output drives
CeA, the standard fear/anxiety output relay. Diazepam potentiates the
interneuron -> BLA synapse (via `connection_modulation`), suppressing BLA
and, downstream, the `anxiety_index` readout, consistent with
benzodiazepines' clinical anxiolytic effect. Flumazenil given after
diazepam displaces some of diazepam's occupancy from the shared
benzodiazepine-site pool and partially reverses the effect, the same
overdose-reversal mechanism as the mu-opioid/naloxone case, and, like that
case, imperfectly and temporarily, since flumazenil's own illustrative
clearance is faster than diazepam's.

## Fourth extension: multiple receptor systems on one shared circuit

Every scenario up to this point models one receptor site and one circuit in
isolation. A real drug usually hits several receptor subtypes across
several regions at once, and a real circuit carries several receptor types
at once, so `Simulation` needed to support several receptor sites, each
with its own PK models, ligands, and coupling, all driving one shared
`CircuitNetwork` in a single integrated ODE run.

**API.** `engine.py` gained a small bundling class, `ReceptorSystem(
pk_models, receptor_site, coupling, connection_modulation=None)`, holding
exactly the four pieces `Simulation` previously took as top-level
arguments. `Simulation` now accepts either form:

```
# original v0 form, unchanged, still the normal single-receptor-site case
Simulation(pk_models=..., receptor_site=..., network=..., coupling=...,
           connection_modulation=...)

# new: several receptor systems on one shared network
Simulation(network=..., receptor_systems=[mu_system, d2_system, ...])
```

Passing both forms together, or neither, is a `TypeError`. Every existing
call site in this codebase uses the first form with keyword arguments, so
reordering `Simulation.__init__`'s parameters to put `network` first (to
make room for `receptor_systems`) doesn't break anything: no call site
relied on positional order. `sim.pk_models` / `sim.receptor_site` /
`sim.coupling` / `sim.connection_modulation` remain readable directly on a
`Simulation` when it has exactly one receptor system, whichever form built
it, via properties that proxy to that sole `ReceptorSystem`; they raise a
clear `AttributeError` (pointing at `sim.receptor_systems[i].<attr>`
instead) on a genuinely multi-system `Simulation`, where a single such
attribute would be ambiguous.

**State vector.** For k receptor systems with n_1..n_k ligands each
(d = r = sum(n_i)):

```
y = [C^(1)_1..C^(1)_{n_1}, C^(2)_1..C^(2)_{n_2}, ...,   # every system's concentrations,
     occ^(1)_1..occ^(1)_{n_1}, occ^(2)_1..occ^(2)_{n_2}, ...,  # then every system's occupancies,
     r_1, ..., r_n]                                     # then one shared set of circuit rates
```

in system order throughout. This reduces exactly to the original
`[C_1..C_d, occ_1..occ_d, r_1..r_n]` layout when k == 1, so `sim.d`,
`sim.r`, and `sim.c` still mark the same slice boundaries they always did
and any code that manually reconstructs state from `sim.rhs`/`solve_ivp`
output (several examples and tests do this for two-phase dosing) needed no
changes. `rhs()` slices each system's own `(C_i, occ_i)` block, computes
that system's PK derivative, occupancy derivative, and activation exactly
as before, then combines all systems' contributions to the shared circuit
before the one call to `network.rhs()`. A new `dose_vector(**by_ligand_
name)` helper on `Simulation` builds the flat, correctly-ordered
`dose_concentrations` array from ligand names, since hand-tracking the
per-system concatenation order gets easy to mix up once there's more than
one system.

**Composing contributions to the shared circuit.** Two different rules
apply to the two ways a receptor system can affect the network, and they
were chosen for different reasons:

- `coupling` (direct drive): contributions from different systems targeting
  the same node are summed. This isn't really a new design choice: the
  Wilson-Cowan input term (`net_input = W @ rates + drive`) is already a
  sum, so two independent receptor systems adding to the same node's drive
  is just two independent inputs superposing, exactly like two independent
  synaptic inputs would.
- `connection_modulation` (allosteric gain on an existing connection):
  when more than one receptor system targets the exact same `(post_node,
  pre_node)` connection, their `(1 + gain * activation)` factors are
  composed **multiplicatively**: `effective_weight = base_weight *
  factor_1 * factor_2 * ...`. The alternative considered was summing the
  gain terms before adding 1 (`1 + sum(gain_i * activation_i)`), matching
  how `coupling` composes. Multiplicative composition was chosen because
  each `connection_modulation` entry represents an independent modulator
  acting on the same physical synapse's transmission efficacy, and
  independent effects on an efficacy multiply, not add: if two different
  modulators each independently double a synapse's effective strength, the
  synapse ends up 4x its base strength, not 3x. Additive composition of the
  gain terms would let one modulator's effect be diluted or amplified by
  whatever the other modulator's activation happens to be doing at that
  instant, which has no clean mechanistic reading for "two independent
  allosteric effects on the same synapse." In the ordinary case relevant to
  every scenario in this codebase so far, a single receptor system
  modulating any given connection, multiplicative and additive composition
  are identical (there's nothing to compose yet), so this choice only
  becomes visible once a scenario deliberately wires two systems onto the
  same connection, which is an unusual, explicit design choice rather than
  a default outcome of adding a second receptor system.

**Combined scenario.** `scenarios.combined_mu_opioid_d2_network()` /
`mu_opioid_receptor_system()` / `d2_receptor_system()` put this to use: a
mu-opioid `ReceptorSystem` on `VTA_GABA` (the reward circuit) and a D2
`ReceptorSystem` on `Striatum_D2_MSN` (the basal ganglia indirect pathway)
now drive one shared 9-node network via `Simulation(network,
receptor_systems=[mu_system, d2_system])`. The combined network is a
straight union of both sub-circuits' own connectivity, unchanged, with no
added connection between the two blocks. A real brain doesn't literally
partition this way (corticostriatal and mesostriatal projections really do
link these two systems), but no literature-sourced *weight* for such a
cross-connection was identified, the same kind of gap already documented
above for the within-circuit weights, and inventing one just to
manufacture a cross-circuit interaction would be exactly the
mechanistically dishonest shortcut this codebase avoided when choosing
`connection_modulation` over `coupling` for the GABA-A scenario. The real
capability this demonstrates is that both receptor systems read and write
one shared state vector and are integrated by one `solve_ivp` call, not
that these two illustrative circuits causally talk to each other.
`tests/test_multi_receptor_simulation.py` checks: the combined simulation
integrates without error; each block's dynamics, with the other system's
ligands held at zero concentration, reproduce that system's own standalone
single-receptor-system `Simulation` to numerical integration tolerance
(confirming the block-diagonal network genuinely doesn't let one system
leak into the other's block); both readouts show their correct effect
under simultaneous dosing of both drugs; and, at the engine level with a
minimal synthetic network (mirroring `tests/test_engine.py`'s existing
`connection_modulation` unit tests), that two systems targeting the same
connection really do compose multiplicatively rather than additively,
confirming the design decision above against the actual derivative
`Simulation.rhs()` produces, not just against a description of it.

**What's now possible that wasn't before:** a scenario can model a drug (or
combination of drugs) that acts at more than one receptor subtype across
more than one brain region in a single coupled simulation, each site with
its own realistic PK and binding kinetics, all feeding into shared circuit
dynamics, which is what "Explicitly not yet real" below used to call out
as unbuilt. What's still not built: an actual cross-circuit connection
backed by a specific sourced weight (see the combined scenario's design
note above), and a scenario where two systems deliberately target the same
`connection_modulation` entry with real pharmacology behind each side
(the current test of that composition rule uses a synthetic network built
purely to exercise the arithmetic).

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
- D2 receptors are Gi-coupled on striatal indirect-pathway MSNs; the same
  disinhibition logic reused one hop further down the basal ganglia's
  indirect pathway (see above).
- Benzodiazepines are positive allosteric modulators of GABA-A, not
  orthosteric competitors with GABA; representing that mechanism required
  the small `connection_modulation` extension described above rather than
  reusing `coupling` as-is (see above).

**Explicitly not yet real:**
- `kon`/`koff`/`ke`/circuit `tau`/connection weights in the illustrative
  scenarios (mu-opioid, D2, GABA-A) are hand-set to get qualitative behavior
  right, not sourced from a specific Ki/kon/koff or PK dataset. They should
  not be cited as measured values. As of Track A, mu-opioid receptor
  binding and PK have a literature-calibrated alternative (see below); D2
  and GABA-A do not yet, and no scenario's circuit `tau`/connection
  weights/gains have a literature target — no dataset for VTA-NAc-PFC (or
  basal-ganglia, or amygdala) connection *weights* (as opposed to wiring
  direction/sign, which matches known circuit anatomy, see above) was
  identified. Calibrating the remaining scenarios, and longer-term fitting
  circuit parameters Bayesian-style the way the CCT model's 14 parameters
  were fit against a literature screen, is the next real piece of work, not
  a formality.
- Most scenarios in v0 still model one receptor site and one circuit in
  isolation. `Simulation` itself no longer requires that: it now accepts
  multiple `ReceptorSystem`s (each with its own PK models, ligands,
  coupling, and connection_modulation) driving one shared `CircuitNetwork`
  at once, and `scenarios.combined_mu_opioid_d2_network()` demonstrates it
  (see "Fourth extension: multiple receptor systems on one shared circuit"
  above). What's still missing is a cross-circuit *connection* backed by a
  sourced weight, i.e. the combined scenario's two sub-circuits share one
  state vector and one integration but don't causally affect each other,
  because no literature-sourced weight for such a connection was
  identified (same gap as the within-circuit weights above).

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
