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

This is the single-receptor-site case. "Fifth extension: multiple
receptor systems on one shared circuit" below describes how this layout
generalizes when a `Simulation` combines several receptor sites on one
shared circuit at once.

As of Phase 1 (P1.1) the first block is more precisely "each ligand's PK
*state*", which is one entry per ligand only while every ligand is on a
single-state PK model (`OneCompartmentPK`, or an IV dose-administered
one-compartment model). A drug given by an extravascular route carries a
depot state, and a two-compartment drug a peripheral state, in that same
block. See "Phase 1: the dose-to-concentration bridge (P1.1)" at the end of
this document; the layout above is unchanged for every scenario written
before it.

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

## Fourth scenario, and why it needed NO engine extension: multi-region mu-opioid receptor density map (Track C1)

The three scenarios above each couple one receptor's activation into
exactly one circuit node (`mu_opioid_coupling() == {"VTA_GABA": -3.0}`,
`d2_coupling() == {"Striatum_D2_MSN": -3.0}`). Real receptors are not
confined to a single node: mu-opioid receptors, for instance, are expressed
at meaningfully different densities across many brain regions
simultaneously, which is exactly why a single systemic opioid dose produces
several distinct effects at once (reward, analgesia, sedation) rather than
one.

The question this track had to answer first: does `Simulation.coupling`
already support that, or does it need a real extension? Looking at
`Simulation.rhs`:

```python
for node_name, gain in self.coupling.items():
    idx = self.network.index[node_name]
    drive[idx] += gain * activation
```

This loop already iterates over every entry in `coupling`, independently of
how many there are; nothing about it assumes a single-key dict. A receptor
whose activation should drive N nodes at N different gains was already
expressible as `{"node_a": g_a, "node_b": g_b, "node_c": g_c, ...}` before
this track started, with zero changes needed to `receptor.py`,
`network.py`, or `engine.py`. That is a materially different situation from
the GABA-A/benzodiazepine case above, which genuinely needed
`connection_modulation`: there, the *shape* of the mechanism itself was
wrong for any single node (additive drive doesn't represent an allosteric
modulator at all). Here the shape is right already; only the number of
targets was previously one by convention, not by any code limit. A
`{node: (density, gain)}` two-tuple structure, or a separate
`ReceptorDensityMap` class, would have been machinery this codebase doesn't
need: nothing downstream of `coupling` would ever consume a density and a
mechanism-strength number separately, only their product, which is exactly
what a plain `{node: gain}` entry already stores. So no engine.py/network.py
extension was made for this track; `scenarios.py` gained a new coupling
dict with three entries and a new network with three coupled nodes, and
that was sufficient.

`scenarios.multi_region_mu_opioid_network()` extends the existing VTA-NAc-
PFC reward circuit (unchanged) with two more, deliberately UNCONNECTED
sub-systems the same mu-opioid receptor also innervates in real anatomy:

1. **VTA_GABA** (reward, reused unchanged): Gi-coupled mu-opioid receptors
   on VTA GABAergic interneurons; agonist activation suppresses the
   interneuron, disinhibiting VTA dopamine firing.
2. **PAG_Interneuron -| PAG_Output** (analgesia, new): the classic
   descending-analgesia mechanism (Fields & Basbaum's PAG-RVM circuit).
   Mu-opioid receptors on PAG GABAergic interneurons tonically inhibit PAG
   output neurons projecting to the rostral ventromedial medulla, which
   relays descending inhibition of spinal nociceptive transmission; agonist
   activation suppresses the interneuron, disinhibiting that output. This
   is the SAME "Gi-coupled-receptor-on-a-GABA-interneuron disinhibits an
   output population" motif as VTA_GABA, so it is wired identically (same
   tau/baseline/gain/weight shape as `VTA_GABA -| VTA_DA`).
3. **LC** (sedation/arousal, new): mu-opioid receptors sit directly on
   locus coeruleus noradrenergic neurons (Gi-coupled hyperpolarization),
   not on an interposed local interneuron: the classic mechanism behind
   opioid sedation. Modeled as a single isolated node coupled directly to
   receptor activation, with no local circuit needed, demonstrating that
   `coupling` targets are agnostic to whether the underlying biology is
   direct or interneuron-mediated.

`PAG_Interneuron`/`PAG_Output` and `LC` are deliberately left unconnected to
the reward circuit and to each other: wiring them together would assert an
anatomical projection this scenario doesn't intend to claim. The point is
that one receptor's activation supplies drive to several genuinely separate
downstream systems at once, not that those systems interact.

**Relative density citations** (qualitative/relative ordering, not absolute
Bmax numbers, which none of these sources report in comparable units):
- [Mansour1987] Mansour A, Khachaturian H, Lewis ME, Akil H, Watson SJ.
  "Autoradiographic differentiation of mu, delta, and kappa opioid
  receptors in the rat forebrain and midbrain." J Neurosci.
  1987;7(8):2445-2464. Foundational quantitative mu-receptor autoradiography
  mapping across these and other regions.
- [Hawkins1988], cited via Mansour A, Khachaturian H, Lewis ME, Akil H,
  Watson SJ. "Anatomy of CNS opioid receptors." Trends Neurosci.
  1988;11(7):308-14: mu receptor binding density in periaqueductal gray and
  raphe nuclei is described as MODERATE, not among the highest-density
  regions.
- [TempelZukin1987] Tempel A, Zukin RS. "Neuroanatomical patterns of the
  mu, delta, and kappa opioid receptors of rat brain as determined by
  quantitative in vitro autoradiography." Proc Natl Acad Sci.
  1987;84(12):4308-12 (PubMed 3035579). Reports mu receptors most densely
  localized in patches in the striatum/nucleus accumbens, several
  cortical/hippocampal/thalamic regions, and the locus coeruleus: LC and
  striatal/NAc patches rank among the HIGHEST mu-receptor-density regions
  identified, a distinctly higher category than the "moderate" PAG/raphe
  density above.
- [VanBockstaele] Van Bockstaele EJ et al.'s ultrastructural
  immunoelectron-microscopy series on rat/cat locus coeruleus (e.g.
  "Ultrastructural evidence for prominent distribution of the mu-opioid
  receptor at extrasynaptic sites on noradrenergic dendrites in the rat
  nucleus locus coeruleus") corroborates dense mu-opioid receptor
  expression directly on LC noradrenergic neurons themselves, not via a
  local interneuron, consistent with the direct-coupling mechanism used for
  `LC` above.

**A genuinely interesting wrinkle surfaced by taking this literature
seriously** (same spirit as Track A's naloxone-koff finding): LC is the
highest-density region of these three per [TempelZukin1987], with VTA and
PAG both in the "moderate" tier per [Hawkins1988]/[Mansour1987], yet the
gain chosen for `LC` in `mu_opioid_multi_region_coupling()` is the smallest
of the three (-1.5 vs -3.0 for VTA_GABA and -2.5 for PAG_Interneuron), not
the largest. That is not a contradiction; it is exactly why a raw density
number cannot be dropped into `coupling` unmodified. `coupling`'s gain is a
node-level drive-coupling constant in this mean-field abstraction, which
folds together (a) receptor density and (b) how much a unit of receptor
activation actually changes that node's net input given the surrounding
circuit mechanism. VTA_GABA's and PAG_Interneuron's activation is amplified
by an interposed GABAergic-interneuron-disinhibition step reaching a
separate output node one synapse downstream; LC's activation acts directly
on the same node being read out, with no such amplification step. So even
though LC has the highest receptor density of the three, its illustrative
node-coupling gain is smaller, and that is a real property of this level of
abstraction, not an error: density supports which nodes plausibly deserve a
nonzero coupling entry at all (all three do), it does not by itself set the
gain's magnitude. All three magnitudes remain hand-tuned illustrative
numbers chosen to produce clear, checkable qualitative dynamics, not a
density-to-gain unit conversion.

Behaviorally, this lets one antagonist dose (naloxone) reverse three
simultaneous effects at once from a single competing occupancy trace:
`incentive_salience` (reward) and `analgesia_index` (weighted on
`PAG_Output`) both rise with fentanyl and fall back with naloxone;
`arousal_index` (weighted on `LC`) falls with fentanyl (sedation) and rises
back with naloxone. See `tests/test_multi_region_scenario.py` and
`examples/mu_opioid_multi_region_circuit.py`.

## Fifth extension: multiple receptor systems on one shared circuit

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
- Mu-opioid receptors are expressed, at different relative densities well
  documented across the autoradiography literature, in VTA, PAG, and locus
  coeruleus at once; a single fentanyl dose producing simultaneous reward,
  analgesia, and sedation effects (Track C1, see above) reflects that same
  receptor's activation reaching multiple real anatomical targets, not
  three independent drug actions.

**Explicitly not yet real:**
- `kon`/`koff`/`ke`/circuit `tau`/connection weights in the illustrative
  scenarios (mu-opioid, D2, GABA-A) are hand-set to get qualitative behavior
  right, not sourced from a specific Ki/kon/koff or PK dataset. They should
  not be cited as measured values. As of Track A, mu-opioid receptor
  binding and PK have a literature-calibrated alternative (see below); D2
  (see "Calibrated: D2" below) and the GABA-A benzodiazepine site (see
  "Calibrated: GABA-A" below) now do too. No scenario's circuit `tau`/connection
  weights/gains have a literature target: no dataset for VTA-NAc-PFC (or
  basal-ganglia, or amygdala) connection *weights* (as opposed to wiring
  direction/sign, which matches known circuit anatomy, see above) was
  identified. Calibrating the remaining scenarios, and longer-term fitting
  circuit parameters Bayesian-style the way the CCT model's 14 parameters
  were fit against a literature screen, is the next real piece of work, not
  a formality.
- Most scenarios in v0 still model one receptor site and one circuit in
  isolation. Two independent gaps used to be documented here and both are
  now closed differently: Track C1 (see above) showed that one receptor's
  activation can already drive multiple nodes, even across
  otherwise-unconnected sub-systems, at different gains, with no engine
  change (a coupling-breadth question, not an engine limit). Separately,
  `Simulation` itself no longer requires exactly one receptor SITE per
  simulation: it now accepts multiple `ReceptorSystem`s (each with its own
  PK models, ligands, coupling, and connection_modulation) driving one
  shared `CircuitNetwork` at once, and
  `scenarios.combined_mu_opioid_d2_network()` demonstrates it (see "Fifth
  extension: multiple receptor systems on one shared circuit" above). What's
  still missing is a cross-circuit *connection* backed by a sourced weight,
  i.e. the combined scenario's two sub-circuits share one state vector and
  one integration but don't causally affect each other, because no
  literature-sourced weight for such a connection was identified (same gap
  as the within-circuit weights above).

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

**Calibrated: GABA-A / benzodiazepine site (diazepam/flumazenil).**
`scenarios.py` has an analogous second parameter set for the third
scenario, `gaba_a_benzodiazepine_ligands_calibrated()` /
`gaba_a_receptor_site_calibrated()` / `gaba_a_pk_models_calibrated()` /
`gaba_a_connection_modulation_calibrated()`, grounded in published
receptor-binding and clinical PK data where it exists (full citations and
derivations are inline in that section of `scenarios.py`):
- **Ki (Kd)**: diazepam 3.6 nM (Möhler & Okada 1977, Life Sci,
  doi:10.1016/0024-3205(77)90191-6, direct [3H]diazepam saturation binding
  in rat cerebral cortex); flumazenil 1.0 nM (Möhler, Burkard, Keller,
  Richards, Haefely 1981, J Neurochem, PMID 6268754, direct
  [3H]Ro 15-1788 saturation binding in the same tissue preparation, with
  unlabeled Ro 15-1788 also displacing [3H]diazepam at IC50 = 2.3 nM in
  that same paper). Both figures agree on direction: flumazenil binds this
  site more tightly than diazepam does, corroborated by subunit-resolved
  data (Sieghart 1995, Pharmacol Rev, ~1.0-1.5 nM for flumazenil at the
  major synaptic subtypes).
- **koff/kon**: no source was found reporting kon or koff for diazepam or
  flumazenil at this site, for either ligand, in units compatible with this
  codebase's ODE, from a standardized assay, unlike the mu-opioid case
  where Mann et al. 2022 supplied koff for both ligands directly. The
  closest available kinetics paper (Brown & Martin 1984, J Neurochem
  42:918-923) characterizes only flumazenil's dissociation, qualitatively,
  and is paywalled. Given that gap, both ligands are assigned the same
  explicitly-flagged, not-independently-measured association rate constant
  (`kon_shared`), and koff is derived from that shared kon and each
  ligand's real Ki via koff = kon_shared * Ki, the same Kd = koff/kon
  relationship used in the mu-opioid section, just solved for the other
  variable because the piece that's actually missing from the literature
  is reversed between the two receptors.
- **PK**: diazepam ke from its distribution-phase half-life (~60 min,
  Mandelli, Tognoni, Garattini 1978, Clin Pharmacokinet); flumazenil ke
  from its mean reported IV elimination half-life (54 min, FDA ROMAZICON
  label, NDA 20-073). Both are one-compartment collapses of real
  multi-phase clinical PK, chosen to match this scenario's acute-bolus
  timescale, the same simplification the mu-opioid section makes and
  documents for fentanyl/naloxone.
- **Validation**: `tests/test_gaba_a_scenario_calibrated.py` checks the
  calibrated flumazenil Ki against Savic, Widén, Stone-Elander 1991
  (Lancet), which reports that a 1.5 mg IV dose of flumazenil occupied
  ~55% of human benzodiazepine binding sites while a 10x higher dose (15
  mg) occupied nearly all of them; without a dose-to-concentration PK
  submodel, the test reproduces the shape of that relationship (a
  concentration solved to give ~55% occupancy at equilibrium must reach
  near-saturation at 10x that concentration), not the exact 1.5 mg/15 mg
  dose mapping.
- **A genuinely interesting real finding surfaced by calibration, in two
  parts**: First, the illustrative placeholder assumed flumazenil had the
  faster koff ("weaker hold") of the two ligands, to mirror its short
  clinical duration of action. The real receptor-affinity data says the
  opposite: flumazenil's Ki is about 3.6x tighter than diazepam's, so under
  this codebase's shared-kon assumption, flumazenil actually has the
  SLOWER koff (longer receptor residence) of the two. Second, and
  connectedly, the illustrative placeholder also assumed a large PK
  clearance asymmetry (diazepam ke=0.02 vs flumazenil ke=0.5 min^-1, 25x
  apart); the real, cited clearance rates relevant to this scenario's
  acute timescale are close in magnitude (diazepam's ~60 min distribution-
  phase half-life vs. flumazenil's ~54 min elimination half-life). Put
  together: neither weaker/faster receptor binding nor a large acute-phase
  PK difference is what makes flumazenil need repeat dosing to sustain
  reversal of a benzodiazepine overdose in practice, a clinical need that
  is well documented in flumazenil's prescribing information, which
  recommends repeat boluses at 20-minute intervals for resedation. What
  actually drives it, per this literature review, is diazepam's active metabolite,
  N-desmethyldiazepam, with a terminal half-life of up to 100 hours,
  continuing to occupy and potentiate the benzodiazepine site long after
  parent diazepam has redistributed away and long after flumazenil's own
  ~54-minute half-life has cleared it. This is a second active compound
  this one-compartment, single-species PK model cannot represent, the same
  category of gap as the mu-opioid section's renarcotization caveat
  (fentanyl's slow terminal elimination phase versus that scenario's
  one-compartment collapse).
- As instructed by this calibration pass's own scope: the amygdala circuit
  itself (`amygdala_anxiety_circuit_network`, `gaba_a_coupling`,
  `anxiety_index_readout`) is reused unchanged, and the potentiation GAIN
  in `connection_modulation` is also reused unchanged and left
  explicitly illustrative in `gaba_a_connection_modulation_calibrated()`,
  because it is a circuit-level parameter, not a receptor-binding or PK
  constant, and no dataset quantifying it against real diazepam
  dose/exposure data was identified.

**Calibrated: D2** (receptor/PK only, circuit still illustrative):
`scenarios.py` also has a calibrated D2 parameter set,
`d2_dopamine_ligands_calibrated()` / `d2_receptor_site_calibrated()` /
`d2_pk_models_calibrated()`, following the same pattern as the mu-opioid
pass above (full citations and derivations are inline in that section of
`scenarios.py`):
- **kon/koff**: unlike the mu-opioid pass, both were found directly
  measured (not derived from Ki/Kd) for both ligands, though from two
  different assays rather than one shared one. Haloperidol: kon = 2.13
  nM^-1 min^-1, koff = 0.65 min^-1 (Sykes et al. 2017, Nat Commun,
  doi:10.1038/s41467-017-00716-z, a TR-FRET competition-kinetics assay on
  human D2L receptors covering ~20 antipsychotics). Dopamine: kon = 5.82
  nM^-1 min^-1, koff = 11.82 min^-1 (Agren et al. 2021, Int J Mol Sci,
  doi:10.3390/ijms22084078, a functional GIRK-channel-current kinetics
  assay in Xenopus oocytes). Each study's own koff/kon ratio reproduces
  that same study's own reported equilibrium Kd (haloperidol ~0.31 nM
  kinetic vs ~0.32 nM from its reported pKd; dopamine ~2.03 nM kinetic vs
  ~2.04 nM from its reported pKd), so both are internally consistent, but
  because the two ligands were not measured in one shared assay the
  fentanyl/naloxone-style direct-comparability guarantee does not extend
  to this pair.
- **PK**: haloperidol ke from a commonly cited representative single-dose
  oral terminal half-life (~20 h; reported range ~14-37 h across studies,
  Kudo & Ishizaki 1999, Clin Pharmacokinet,
  doi:10.2165/00003088-199937060-00001). Dopamine has no real first-order
  plasma/synaptic clearance rate constant to calibrate against: it is not
  a dosed drug here, and its real clearance (DAT reuptake, ~50-100 ms per
  release event, Garris et al. 1994, J Neurosci, PMID 7931564) is a
  saturable transporter process, not first-order, on a timescale
  3-4 orders of magnitude faster than this codebase's minute-scale time
  unit. `d2_pk_models_calibrated()` uses an explicitly flagged proxy
  (ke=5.0 min^-1, t1/2 ~8.3 s) chosen only to preserve the correct
  qualitative ordering (dopamine clears far faster than haloperidol) at a
  timescale this simulation can actually resolve; the real clearance
  number is documented in `scenarios.py` but deliberately not used
  literally, and is not cited as a measured ke.
- **Validation**: `tests/test_d2_scenario_calibrated.py` checks the
  calibrated haloperidol ligand's equilibrium occupancy at two clinical
  plasma concentrations from a PET occupancy study (Kapur et al. 1997,
  Psychopharmacology, doi:10.1007/s002130050277: 0.51 ng/mL for 50% D2
  occupancy, 2.0 ng/mL for 80%), converted to nM. The model predicts
  occupancy in the right direction at both (clearly partial-to-high at the
  lower concentration, clearly near-saturating, >90%, at the higher one)
  but runs somewhat higher than the clinical percentages, expected since
  haloperidol is ~89-93% plasma-protein-bound in vivo and this codebase
  has no protein-binding or brain/plasma-partitioning submodel; this is
  the same directional/order-of-magnitude validation style used for the
  fentanyl/Moss et al. comparison above, not an exact match.
- **A finding this calibration pass was specifically asked to verify
  rather than assume, and that did not hold up as stated**: a common
  pharmacology-folklore claim (and the framing this calibration pass
  started from) is that haloperidol has an unusually SLOW D2 dissociation
  rate compared with other antipsychotics, and that this alone explains
  its prolonged EPS/motor side effects. That traces to Kapur & Seeman 2000
  (J Psychiatry Neurosci, PMID 10740989), which reported koff varying
  ~1000-fold across nine antipsychotics and framed slow koff as the
  defining typical-vs-atypical distinction. Checking it against later,
  larger, single-assay kinetic panels does not support that framing:
  Sykes et al. 2017's own panel puts haloperidol's koff (0.65 min^-1,
  dissociation half-life ~1.07 min) toward the fast end even among typical
  antipsychotics (fluphenazine, spiperone, nemonapride, and
  (+)-butaclamol all dissociate several- to 30-fold more slowly than
  haloperidol in the same assay), and haloperidol's koff is actually
  faster than the atypical risperidone's in that same dataset. Sahlholm et
  al. 2016 (Eur Neuropsychopharmacol, doi:10.1016/j.euroneuro.2016.01.001)
  independently found haloperidol's recovery from blockade only 2.5- to
  6.4-fold slower than clozapine/quetiapine/amisulpride, not ~1000-fold. A
  published critique of Sykes et al. (Zeberg & Sahlholm 2018, Nat Commun,
  doi:10.1038/s41467-018-04489-x) argues it is haloperidol's unusually
  HIGH association rate (kon), not a slow koff, that better explains its
  EPS liability, and haloperidol's kon in the Sykes et al. dataset is among
  the highest of the whole panel. What the dopamine-vs-haloperidol
  comparison this scenario actually needs (and can represent) does hold:
  dopamine's real koff is ~18x faster than haloperidol's (11.82 vs
  0.65 min^-1), so the fast-agonist/slow-antagonist qualitative shape the
  illustrative placeholder assumed is correct; it is specifically the
  broader "haloperidol is uniquely sticky among antipsychotics" claim that
  the more rigorous data does not support, and that this codebase's
  two-ligand scenario has no way to represent either way, since it only
  ever models one agonist against one antagonist at a time.
- The basal ganglia indirect-pathway circuit itself
  (`basal_ganglia_indirect_pathway_network`, `d2_coupling`,
  `locomotor_activity_readout`) is reused unchanged from the illustrative
  scenario in the calibrated one: only receptor binding and PK were in
  scope for this calibration pass.

## Bayesian parameter fitting (Track C2)

Track A calibrated point estimates for all three receptor systems' binding
and PK constants against literature. This track asks the natural next
question: given those point estimates, and given that several of their
underlying citations also report a real standard deviation, can this
codebase run an actual Bayesian fit, the way the sibling project
`cct-bayesian-validation` fit the CCT model's 14 free parameters against
real behavioral evidence? `src/neurocascade/calibration/` is the resulting
scaffold. It mirrors that sibling project's method (PyMC, a gradient-free
step method since the ODE forward model isn't differentiable through
PyTensor, a two-tier informed/uninformed prior scheme, a Gaussian
likelihood built from real literature targets, results saved as `arviz`
`InferenceData`) while diverging from it in one structural respect, for a
reason discovered empirically rather than assumed up front (see "An honest
convergence story" below): the CCT model's 14 parameters all feed one
coupled ODE state vector and were fit in a single joint `pm.Model`, which is
right for that model, since those parameters genuinely interact. None of
neurocascade's three receptor systems (mu-opioid, D2, GABA-A) shares a
parameter or a likelihood term with another: each has its own independent
`ReceptorSite`/ligands/PK, and each of the three validation targets in
`evidence.py` is scoped to exactly one system. So this scaffold runs THREE
INDEPENDENT `pm.Model`s, one per system, each with its own `DEMetropolisZ`
chain, rather than one combined model across all eight parameters.

### What is, and is not, honestly fittable here

Checking this before writing any code, per this track's brief: circuit-layer
weights/gains/taus have no literature target in any of the three calibrated
sections above, a gap `docs/architecture.md` has now documented three times
over ("Explicitly not yet real" above; each "Calibrated: ..." section's own
closing note). That is still true. No circuit-layer parameter is fit here,
and none should be until a real behavioral or connectivity dataset exists to
fit it against; the module docstring in `calibration/__init__.py` states
this plainly rather than quietly limiting scope without saying why.

What IS fittable: the receptor-binding constants (Ki/Kd, kon, koff) in the
three calibrated sections, several of which trace to a citation that also
reports a real standard deviation, plus three real validation targets
already cited in those same sections: Moss et al. 2020's fentanyl occupancy
figure (mu-opioid), Kapur et al. 1997's PET occupancy curve (D2), and Savic
et al. 1991's flumazenil occupancy figure (GABA-A). `calibration/evidence.py`
turns these into Gaussian likelihood targets rather than inventing new ones.

### Scope and module layout

Eight free parameters across the three systems (`calibration/priors.py`):

| Parameter | System | Tier | Source | Likelihood term? |
|---|---|---|---|---|
| `Ki_fentanyl_nM` | mu-opioid | uninformed (uniform, 0.1x-10x point estimate) | Volpe et al. 2011 (no reported sd) | Moss et al. 2020 |
| `Ki_naloxone_nM` | mu-opioid | informed (real sd) | Volpe et al. 2011, 1.518 +/- 0.065 nM | none |
| `kon_haloperidol_nM_min` | D2 | informed (real sd) | Sykes et al. 2017, 2.13 +/- 0.52 | Kapur et al. 1997 |
| `koff_haloperidol_min` | D2 | informed (real sd) | Sykes et al. 2017, 0.65 +/- 0.07 | Kapur et al. 1997 |
| `kon_dopamine_nM_min` | D2 | informed (real sd) | Agren et al. 2021, 5.82 +/- 0.738 | none |
| `koff_dopamine_min` | D2 | informed (real sd) | Agren et al. 2021, 11.82 +/- 0.72 | none |
| `Ki_flumazenil_nM` | GABA-A | informed (real sd) | Mohler et al. 1981, 1.0 +/- 0.1 nM | Savic et al. 1991 |
| `Ki_diazepam_nM` | GABA-A | informed (real sd) | Mohler & Okada 1977, 3.6 +/- 0.1 nM | none |

Four of the eight (`Ki_naloxone_nM`, `kon_dopamine_nM_min`,
`koff_dopamine_min`, `Ki_diazepam_nM`) have a real reported sd but no
matching likelihood term in this pass: their posterior is their prior,
unchanged by data, the same honest "informed-prior-only" category CCT's own
`priors.py` documents for several of its own elicitation-only parameters.
`priors.py`'s `PRIOR_ONLY_PARAM_NAMES`/`LIKELIHOOD_UPDATED_PARAM_NAMES`
record this distinction in code, and
`tests/test_bayesian_calibration.py::test_prior_only_parameters_have_no_matching_likelihood_term`
checks it holds by construction.

An identifiability point worth stating plainly (`calibration/forward.py`'s
docstring covers this in full): all three validation targets are
equilibrium, single-ligand occupancy measurements, which depend only on the
ratio Kd = koff/kon, not on kon and koff individually. For mu-opioid
fentanyl and GABA-A flumazenil, only a Ki point estimate exists anyway, so
`forward.equilibrium_occupancy_from_kd` fixes an arbitrary reference kon and
fits Kd directly. For D2 haloperidol, both kon and koff are independently
measured with their own real sd (Sykes et al. 2017), so both are sampled as
separate parameters with their own informed priors, while the Kapur et al.
1997 likelihood term constrains only their ratio; the forward model
(`forward.equilibrium_occupancy`) integrates the actual
`ReceptorSite.occupancy_rhs` ODE via `scipy.integrate.solve_ivp` (the same
integrator `engine.py` uses) rather than substituting the closed-form
two-state formula, so this scaffold stays built on the real receptor class
rather than reimplementing its mechanism.

The synthetic-recovery self-test
(`tests/test_bayesian_calibration.py::test_synthetic_recovery_of_known_kd_via_engine_simulation`)
goes one step further than the receptor-only forward model used for
speed elsewhere: it builds a full `engine.Simulation` (a real `OneCompartmentPK`
model, `ReceptorSite`, and a minimal one-node `CircuitNetwork`) with a known,
made-up Kd, runs it to equilibrium, and confirms the calibration machinery
recovers that known Kd from the resulting synthetic occupancy, so the
scaffold's genuine, full-engine reuse (not just the equilibrium shortcut) is
checked at least once, not merely asserted in a docstring.

### An honest convergence story: why this scaffold runs three independent models, not one

The first version of this scaffold combined all eight parameters into one
`pm.Model` with one shared `DEMetropolisZ` chain, matching
`cct-bayesian-validation/bayesian_calibration.py`'s own pattern over its 14
mutually-dependent parameters. Run at production scale (4 chains, 1000
tuning steps, 1000 draws each), it did **not** converge: max r_hat 1.92, min
bulk ESS 5, a bad run by any normal MCMC standard.

Diagnosing rather than hiding it: the haloperidol block was the proximate
cause. `kon_haloperidol_nM_min`/`koff_haloperidol_min`'s informed priors,
from Sykes et al. 2017's kinetic assay, imply Kd = koff/kon ~ 0.3 nM. The
Kapur et al. 1997 PET occupancy target implies a clinical-apparent Kd
roughly 4x larger (~1.3-1.4 nM), a real and expected gap given haloperidol's
~89-93% plasma protein binding, which this codebase's simple two-state
occupancy model doesn't correct for (the same gap
`test_d2_scenario_calibrated.py` already documents directionally). That is a
genuine prior-likelihood tension, not a bug, and fitting the haloperidol
block alone, in isolation, against just the Kapur likelihood converged
cleanly (r_hat 1.01-1.03), confirming the model and forward-model machinery
were sound; what wasn't sound was combining it with the other two,
mechanistically unrelated, systems in one sampler. A single `DEMetropolisZ`
spanning all eight parameters draws every proposal from one shared history
across all three systems at once, so a joint move is accepted or rejected as
a whole: the haloperidol block's harder geometry dragged down the
acceptance rate, and therefore the effective sample size, of the other five
parameters too, even though none of their likelihood terms reads
haloperidol's parameters at all.

The first fix attempted was to keep one `pm.Model` but give each
receptor-system block its own `DEMetropolisZ` step inside a
`pm.CompoundStep` (`vars=[...]` restricted per block). That hit a
version-specific limitation in this environment's PyMC 6.0.1:
`DEMetropolisZ`'s internal proposal-history buffer is sized to the model's
full dimensionality regardless of the `vars` subset passed to it, so a block
covering fewer than all eight parameters raised `ValueError: operands could
not be broadcast together with shapes (2,) (8,)` inside its own `astep`.

The actual fix, once the diagnosis above was taken seriously rather than
patched around: there was never a statistical reason to sample all eight
parameters in one model in the first place. Mu-opioid, D2, and GABA-A share
no parameter and no likelihood term, so `calibration/bayesian_calibration.py`
now builds THREE INDEPENDENT `pm.Model`s (`build_mu_opioid_model()`,
`build_d2_model()`, `build_gaba_a_model()`), each sampled with its own
`DEMetropolisZ` chain via `run_calibration(system_name, ...)`. This is
simpler than the blocked `CompoundStep` workaround, not just a fallback:
it's the same fix the isolated-haloperidol diagnostic run already
demonstrated, generalized to all three systems, with no version-specific
sampler limitation involved.

**Production-scale diagnostics for all three, run at 4 chains x 1000 tuning
steps x 1000 draws each (`DEMetropolisZ`), independently:**

| System | Free parameters | Max r_hat | Min ESS (bulk) |
|---|---|---|---|
| mu-opioid | `Ki_fentanyl_nM`, `Ki_naloxone_nM` | 1.01 | 585 |
| D2 | `kon_haloperidol_nM_min`, `koff_haloperidol_min`, `kon_dopamine_nM_min`, `koff_dopamine_min` | 1.02 | 284 |
| GABA-A | `Ki_flumazenil_nM`, `Ki_diazepam_nM` | 1.01 | 578 |

All three converge cleanly (r_hat <= 1.02, well under the usual < 1.05
threshold; ESS in the hundreds, not single digits). D2's posterior
concretely shows the protein-binding compromise discussed above:
`kon_haloperidol_nM_min` posterior mean ~0.73 (vs. its 2.13 prior mean, a
real, data-driven shift, since kon's looser prior sd absorbs most of the
adjustment needed to reconcile the kinetic and clinical-apparent Kd) while
`koff_haloperidol_min` posterior mean ~0.68 stays close to its 0.65 prior
mean (its tighter prior sd resists moving as far), exactly the
per-parameter-uncertainty-weighted compromise Bayesian updating should
produce. Each system's full posterior/prior `InferenceData` and summary
table are saved separately under `results/calibration/`
(`{system}_posterior.nc`, `{system}_prior.nc`, `{system}_summary.csv`), plus
a combined `calibration_diagnostics.txt` with each system's max r_hat/min
ESS.

### Tests and dependencies

`tests/test_bayesian_calibration.py` adds: one fast smoke test per system
(~120 draws, 2 chains each; checks it runs end to end and produces a
structurally valid posterior, without asserting tight convergence, since a
run this short isn't expected to fully converge either); a check that the
three systems' parameter lists partition all eight free parameters with no
overlap (the premise the independent-model split relies on); the
prior-only-parameter bookkeeping check described above (now checking all
three systems' forward-model source, not one combined one); and the
synthetic-recovery self-test (6 tests total in this file). All pre-existing
55 tests continue to pass unmodified; the full suite (61 tests total)
passes.

PyMC (6.0.1), PyTensor (3.0.7), and ArviZ (1.2.0) were already installed in
the shared `/root/projects/pardalos/.venv` this project runs tests from
(confirmed with `python -c "import pymc"` before writing any code); no new
dependency was installed, and this repository has no `requirements.txt` or
`pyproject.toml` to update (it relies on `tests/conftest.py`'s `sys.path`
shim into the shared venv, the same convention `cct-bayesian-validation`
uses).

## Track E: circuit-layer calibration attempt (searched, not found)

Every calibration pass so far ("Explicitly not yet real" above; each
"Calibrated: ..." section's own closing note; Track C2's own scoping
paragraph) states that circuit-layer parameters (`CircuitNode.tau`/
`baseline_input`/`gain`, `CircuitNetwork.connectivity` weights, and each
receptor system's `coupling`/`connection_modulation` gain) have no
literature target and remain illustrative, but every one of those searches
was aimed at binding/PK data, not specifically at circuit-level or
behavioral-endpoint data. This track ran that more targeted search directly
for the three natural fit targets Track C2 already identified by name
(`incentive_salience_readout`/`mu_opioid_coupling`'s gain;
`locomotor_activity_readout`/`d2_coupling`'s gain;
`anxiety_index_readout`/`gaba_a_connection_modulation`'s gain), following
the same method the sibling project `cct-bayesian-validation` used to fit
its own circuit/structural parameters (phi1/phi2/phi3/delta_MEM/delta_BUS)
against real single-agent behavioral effect sizes (e.g. "40-60% CPP
reduction," Wee et al. 2007, via CCT's own Table S1).

**Conclusion up front, so it isn't buried:** no circuit-layer parameter is
fit in this pass either. This is a "searched and empty" result for all
three systems, not a "not yet attempted" one: specific real papers reporting
the exact endpoints this track's brief named were found and read (abstracts,
and full text where accessible) for all three systems, and none produced a
number this codebase could honestly turn into a Gaussian-likelihood target
without either (a) substituting a different drug's measured effect for the
one this scenario actually models, which is a category of shortcut this
codebase's own binding/PK calibration passes never take even under real
pressure to do so (see below), or (b) inventing a dose-to-concentration
pharmacokinetic submodel this codebase deliberately does not have for any
drug, anywhere (`pk.py`'s `OneCompartmentPK` explicitly starts from an
already-achieved bolus concentration, not an administered dose; see "State
vector" above). Both would be exactly the kind of invented/forced number
this track's brief and the rest of this codebase's documentation repeatedly
warn against.

### A structural reason this gap is harder than the binding/PK gap, found while searching

Track C2's three receptor-occupancy validation targets (Moss et al. 2020,
Kapur et al. 1997, Savic et al. 1991) are all usable for exactly one shared
reason: each one reports drug effect as a function of a directly measured
**plasma concentration** (Moss: ng/mL fentanyl exposure; Kapur: ng/mL
haloperidol plasma level from the same PET session; Savic: an IV dose
converted, via the existing closed-form occupancy relationship, to an
equilibrium concentration). Concentration is the one unit this codebase's
model natively speaks (`OneCompartmentPK`, `ReceptorSite.occupancy_rhs`),
so no bridging assumption beyond ordinary unit conversion was ever needed
for those three.

Real quantitative rodent *behavioral* pharmacology, by contrast, is reported
overwhelmingly as **administered dose** (mg/kg), not achieved plasma/brain
concentration, because behavioral-endpoint studies essentially never also
run simultaneous plasma pharmacokinetics the way a PET occupancy study
routinely does (occupancy IS the measured relationship between plasma level
and receptor binding; a locomotor or elevated-plus-maze study has no reason
to draw blood). Every quantitative behavioral number this track found (see
per-system findings below) was dose-based for exactly that reason. Bridging
dose (mg/kg, species- and route-specific) to this codebase's concentration
state variable would require inventing an absorption/bioavailability/volume-
of-distribution submodel that does not exist for fentanyl, haloperidol, or
diazepam in this codebase, and inventing one solely to force this fit would
be a materially bigger, less-honest leap than any unit conversion Track A or
Track C2 ever made (those always stayed within "the same physical quantity,
expressed in different units," e.g. s^-1 to min^-1, or ng/mL to nM at fixed
molar mass; a dose-to-concentration submodel is a new mechanistic claim
about drug disposition, not a unit conversion).

### Per-system findings

**Mu-opioid (fentanyl -> NAc/VTA dopamine release, for
`incentive_salience_readout()`/`mu_opioid_coupling()`).** The directly
correct paper exists and was found: Yoshida Y, Koide S, Hirose N, Takada K,
Tomiyama K, Koshikawa N, Cools AR. "Fentanyl increases dopamine release in
rat nucleus accumbens: involvement of mesolimbic mu- and delta-2-opioid
receptors." Neuroscience. 1999;92(4):1357-65. doi:10.1016/s0306-4522(99)
00046-9, PMID 10426490. It has the right drug (fentanyl), the right brain
region (NAc, with the source of the effect localized to VTA and NAc via
local microdialysis-probe infusion), the right technique (in vivo
microdialysis in freely moving rats), and the right mechanism
(naloxone-reversible, mu-and-delta-2-opioid-receptor-mediated): exactly the
same causal chain `mu_opioid_coupling()` encodes. Its abstract confirms
fentanyl dose-dependently increased NAc dopamine at 5 and 10 ug/kg IV. What
could not be obtained: the actual percent-of-baseline dopamine figure at
either dose.
The paper is not open access (confirmed via Europe PMC: `isOpenAccess: N`,
no PMC full text), and repeated direct fetch attempts against the publisher
page and PubMed returned only the abstract text quoted above, not the
numeric results in the paper's figures/tables. A mechanistically identical
same-class analog number DOES exist and was verified directly (full abstract
fetched and cross-checked): Steinmiller CL, Maisonneuve IM, Glick SD.
"Effects of dextromethorphan on dopamine release in the nucleus accumbens:
interactions with morphine." Pharmacol Biochem Behav. 2003;74(4):803-10.
doi:10.1016/S0091-3057(02)01080-8, PMID 12667894, reports that acute
morphine (5 mg/kg i.p.) increased NAc dopamine to ~175% of basal levels: a
real, matched-unit, percent-of-baseline figure. Morphine and fentanyl are
both full mu-opioid agonists acting through the same
VTA-GABA-interneuron-disinhibition
mechanism this scenario models, so the analogy is mechanistically close, but
it is morphine's measured number, not fentanyl's own: Track A's own binding
calibration never substitutes one ligand's measured constant for a
different ligand's missing one (fentanyl's and naloxone's Ki/koff are each
that ligand's own citation; GABA-A's `kon_shared` assumption, the closest
precedent to a "borrowed" value in this codebase, still keeps each ligand's
own measured Ki and only shares one *unmeasured* rate constant motivated by
physical similarity, not "morphine's dopamine-release magnitude stands in
for fentanyl's"). Using the Steinmiller number as if it were fentanyl's own
would break that discipline, so it is reported here and not used as a
likelihood target.

**D2/haloperidol (dose-dependent locomotor suppression, for
`locomotor_activity_readout()`/`d2_coupling()`).** Real, haloperidol-specific
quantitative behavioral data was found and verified directly: Gao L, Hao C,
Ma R, Chen J, Zhang G, Chen Y. "Synthesis and biological evaluation of a new
class of multi-target heterocycle piperazine derivatives as potential
antipsychotics." RSC Adv. 2021;11:16931-16941. doi:10.1039/d1ra02426d (open
access, PMC9031908) reports haloperidol's own ED50 for reversing MK-801-
induced hyperlocomotion in mice as 0.11 mg/kg (oral), alongside risperidone's
0.02 mg/kg in the same assay. This is the right drug and a standard
antipsychotic-behavioral-pharmacology paradigm (pharmacologically-induced
hyperlocomotion reversal, the same family as the amphitamine-induced-
hyperlocomotion-reversal endpoint this track's brief named), not a
same-class substitute. It was still not used, for the structural reason
above: an ED50 expressed in mg/kg PO in mice has no route into this
codebase's nM-concentration-driven occupancy/circuit model without an
invented oral-absorption/bioavailability submodel, and, separately, an ED50
alone (by construction, "50% of that drug's own maximal effect") is not
actually a usable Gaussian-likelihood target on its own: it would also
require knowing the corresponding concentration and the drug's maximal
achievable effect size, neither of which this one-paper data point
supplies. The existing Kapur et al. 1997 occupancy curve already reused by
Track C2 (0.51 ng/mL -> 50% D2 occupancy, 2.0 ng/mL -> 80%) is the one
concentration-anchored,
already-calibrated haloperidol dataset in this codebase, but it is a
receptor-occupancy curve, not a behavioral-effect curve (it says nothing
about locomotor activity), so it cannot substitute for a real behavioral
target either without inventing an occupancy-to-locomotor-suppression
mapping this literature search did not turn up in matched, usable form.

**GABA-A/diazepam (dose-dependent anxiolysis, for
`anxiety_index_readout()`/`gaba_a_connection_modulation()`).** Diazepam's
own dose-dependent elevated-plus-maze effect is extremely well established
qualitatively and at specific doses (commonly 1-2 mg/kg in rats/mice across
many independent studies found in this search, e.g. Pellow, Chopin, File,
Briley 1985, J Neurosci Methods 14:149-167, doi:10.1016/0165-0270(85)
90031-7, the field's own foundational EPM validation paper), but a clean,
directly-quotable percent-open-arm-time figure specifically for diazepam
(as opposed to its own bar-chart/figure data, not extractable from the
full-text sources this search could access) was not found. A
mechanistically identical same-class analog number WAS found and verified
directly (full text fetched): an open-access 2026 paper (Biomarker-Based
Pharmacological Characterization of ENX-102, PMC12564412) used
chlordiazepoxide, not diazepam, as its positive control and reports vehicle
rats spent 9.4% (+/-2.5% SE) of time in the EPM's open arms versus 39.1%
(+/-4.9% SE) after chlordiazepoxide 5.0 mg/kg: again a real, matched-unit,
percent figure, but for a different (if closely related, same-site,
non-selective GABA-A benzodiazepine-site) drug. It is again not substituted
for diazepam's own missing number, for the same reason given in the
mu-opioid case above. This system also inherits the same dose-vs-
concentration structural gap described above: even a diazepam-specific
percent figure at a stated mg/kg dose would still need a dose-to-
concentration bridge this codebase does not have to connect to
`gaba_a_connection_modulation()`'s gain.

### What this leaves calibrated vs. illustrative

No change to any scenario's circuit-layer numbers: `mu_opioid_coupling()`,
`mu_opioid_multi_region_coupling()`, `d2_coupling()`,
`gaba_a_connection_modulation()` (and, unchanged as always,
`gaba_a_connection_modulation_calibrated()`, which already carried this
exact caveat forward from Track A), and every `CircuitNode`/
`CircuitNetwork` `tau`/`baseline_input`/`gain`/connectivity weight in every
scenario, remain hand-set illustrative placeholders, now with three
specific, searched-and-verified real papers on file per system (two of
three genuinely about the right drug) documenting exactly why none closes
this gap yet, rather than a blanket "no dataset was identified" note. Two
rules keep this a documented, revisitable gap instead of a quietly-forced
number: never substitute one ligand's or drug's measured value for
another's, and never invent a submodel, such as a dose-to-concentration PK
bridge, solely to manufacture a fit.
`tests/test_bayesian_calibration.py::test_no_circuit_layer_parameter_is_fit`
guards, by construction, that none of this scaffold's free parameters is a
circuit-layer quantity, so this claim cannot silently go stale if a future
pass adds one without updating this section.

## Phase 1: the dose-to-concentration bridge (P1.1)

### Why this was the blocking gap, not just a missing feature

Track E above ends by naming the exact reason circuit-layer calibration
could not proceed: real quantitative behavioral pharmacology reports an
**administered dose in mg/kg**, while this codebase's only state variable on
the drug side was an **achieved plasma concentration**, because
`OneCompartmentPK` starts from a bolus concentration rather than from a dose.
Every behavioral number Track E found and verified (haloperidol's 0.11 mg/kg
ED50 for reversing MK-801 hyperlocomotion; fentanyl 5 and 10 ug/kg IV for NAc
dopamine release; diazepam 1-2 mg/kg on the elevated plus maze;
chlordiazepoxide 5.0 mg/kg) was dose-based, and none of them had a route into
a model whose input is nM. That is a *representational* gap rather than a
data gap: the numbers exist and were read, and the model simply could not
accept them as inputs.

Track E also stated the condition under which closing this gap would be
legitimate: not "invent a dose-to-concentration submodel so a fit succeeds",
which is what it declined to do, but build the standard, mechanism-explicit
compartmental structure whose parameters (F, ka, Vd, CL) are themselves
published, measured quantities for these drugs. That is the split this task
takes: **P1.1 adds the structure and no numbers; P1.2-P1.4 add the drugs'
literature parameters, each with its own citation.** `pk.py` contains no
drug-specific constant of any kind after this change, and every dosed model
raises rather than defaulting when a required disposition parameter is
missing (see "Refusing to default", below). Until P1.2-P1.4 land, Track E's
conclusion is unchanged and no illustrative label has been removed
anywhere.

### What was added

`pk.py` gains `DosedOneCompartmentPK` and `DosedTwoCompartmentPK`, both
parameterized by route (IV / IP / SC / PO), body weight, bioavailability F,
absorption rate `ka`, volume(s) of distribution, and clearance(s), and dosed
in mg/kg. The equations are the ordinary textbook linear compartmental ones
(see each class docstring for the exact form and for the concentration-form
derivation of the two-compartment return term, which carries Q/V1 and not
Q/V2). `OneCompartmentPK` is unchanged, in behavior and in its `rhs(C)`
signature.

### Keeping PK inside the coupled state vector

The alternative implementation, and the more common one in PK/PD code, is to
solve the PK analytically (the Bateman function is closed-form) and feed the
resulting concentration trace into the receptor layer by interpolation. That
was rejected for the same reason the top of this document gives for the
overall design: an interpolated trace is a precomputed input, not a state,
and it silently forecloses everything this project is heading toward, where
concentration depends on the current state of the rest of the system
(saturable elimination, an active metabolite, brain uptake driven by a
transporter that other drugs compete for, mid-run second doses).

So PK models own state instead. The protocol, documented at the top of
`pk.py`, is `n_states` / `initial_state(dose)` / `concentration_rhs(state)` /
`concentration(state)`, with the convention that **state[0] of each ligand's
block is the central-compartment concentration the receptor site binds**.
`Simulation`'s state vector generalizes accordingly:

```
y = [<PK block of every ligand, in flattened ligand order>,   # self.d entries
     occ_1, ..., occ_r,                                        # one per ligand
     r_1, ..., r_n]                                            # circuit rates
```

`self.d` is now the total number of PK states rather than the ligand count,
and equals the ligand count exactly when every model is single-state, which
is why the v0 layout, and every existing scenario, test and example built on
it, is unchanged. `self.concentration_rows` holds the row index of each
ligand's central concentration, so `result["concentrations"]` keeps its
`(n_ligands, n_times)` shape whatever mix of PK models is in play, and
`result["pk_states"]` exposes the full block (depot, peripheral compartment)
for inspecting the PK layer itself. `Simulation.rhs` asks each model for its
own block's derivative and never branches on route or model type.

Two engine helpers exist because the extra states made the old hand-rolled
idioms wrong: `Simulation.add_dose(y, doses)` applies a further
administration by *adding* each ligand's `initial_state(dose)` to its block
(for an SC/PO drug a later dose enters the depot, not the plasma, so the
previous `C0[1] += dose` idiom in the reversal examples would have been
quietly wrong for a dose-administered model), and
`Simulation.unpack_solution(sol)` replaces the manual `sol.y[0:d]` slicing
those examples repeated. The three reversal examples now use both, and
produce byte-identical figures to before the change.

### Design decisions worth recording

**A dose is whatever that model doses in.** `Simulation.run(doses=[...])`
(still accepted under its original name `dose_concentrations=`) passes each
per-ligand entry to that ligand's own PK model, which converts it to an
initial state. For `OneCompartmentPK` that entry is an achieved bolus
concentration, exactly as in v0; for a dosed model it is mg/kg. The engine
holds no dose semantics at all. The alternative, baking the dose into the PK
model at construction, was rejected because Phases 2 and 7 exist to sweep
dose against a behavioral endpoint, and that has to be a per-run quantity,
not a per-object one. Zero means "no drug" under both conventions, so every
existing baseline-vs-dosed comparison keeps working unchanged.

**The depot is carried in central-compartment concentration equivalents.**
An extravascular dose's depot state D is defined as the central
concentration that drug would produce if it were absorbed instantaneously,
so `dD/dt = -ka*D` and `dC/dt = +ka*D - ke*C` are unit-consistent without a
second (mass) unit in the state vector. Bioavailability is applied once, at
the dose (`D(0) = F * Dose / V_central`), matching the usual NONMEM `F1`
convention; for a linear model that is algebraically identical to applying F
to the absorption flux instead. The non-bioavailable fraction is simply
never entered into the system: this model does not claim to represent *where*
it was lost (first pass, incomplete absorption, degradation at the injection
site), only that it did not reach the systemic circulation.

**Refusing to default.** An extravascular route requires an explicit F and
ka; both raise `ValueError` if omitted. Defaulting F to 1.0 would be
inventing the single number that most changes achieved concentration, in the
exact style this codebase forbids, and it would do so invisibly. IV, by
contrast, *sets* F = 1 and rejects a `ka`, because the bioavailability of an
intravenous bolus is 1 by definition rather than by assumption. Negative
doses, non-positive volumes/clearances and unknown routes all raise too.

**Units are in the parameter names.** `Vd_L_per_kg`,
`CL_mL_per_min_per_kg`, `ka_per_min`, `molar_mass_g_per_mol`. The
literature reports each of these in several conventions (L/h/kg vs
mL/min/kg, h^-1 vs min^-1), and a silently mis-scaled clearance is
indistinguishable from an invented number in its effect on results. Time is
minutes throughout, matching the existing calibrated `kon`/`koff`/`ke`.
Concentration comes out in mg/L unless a molar mass is supplied, in which
case the model converts to nM, the unit the calibrated receptor constants
use; `model.concentration_unit` records which. Conversion helpers
(`per_hour_to_per_min`, `clearance_L_per_h_per_kg_to_mL_per_min_per_kg`,
`half_life_to_rate_constant`, `concentration_mg_per_L_to_nM`) exist so that
P1.2-P1.4 can write a literature value in the unit its paper reports and
leave the conversion visible in the diff.

**Body weight cancels, and is kept anyway.** With dose in mg/kg and Vd/CL
per kg, body weight cancels exactly out of both C0 and ke, so a mouse and a
rat given the same mg/kg have the same concentration-time course under this
parameterization (`test_body_weight_cancels_for_per_kg_parameters_but_
absolute_amounts_track_it`). It is still a required parameter because the
absolute quantities it produces (`dose_mg`, `V_central_L`, `CL_L_per_min`)
are what published PK tables report and what P1.6's Cmax/AUC validation will
compare against, and because it is not ignored: the class exposes those
absolute values rather than pretending weight played no part.

**Derived quantities, not fitted ones.** `terminal_half_life_min`,
`distribution_half_life_min`, `tmax_min()`, `hybrid_rate_constants()` and
`auc_0_inf()` are closed forms of the parameters already given (the
Bateman stationary point; the negated eigenvalues of the 2x2 disposition
matrix, which satisfy `alpha+beta = k10+k12+k21` and `alpha*beta = k10*k21`;
and `AUC = F*Dose/CL`, which holds for any linear model eliminating only
from the central compartment). They add no information; they exist so P1.6
can compare against published Cmax/Tmax/half-life/AUC without recomputing
them at the call site.

### The reduction that had to be exact

`DosedOneCompartmentPK(route="IV", ...)` with F = 1 is `dC/dt = -(CL/Vd) C`
started at `C0 = Dose*BW/(Vd*BW)`: the *same differential equation* as
`OneCompartmentPK(ke=CL/Vd)`, with the initial concentration derived instead
of stated. So the requirement is not "close agreement" but identity, and
`tests/test_pk.py::test_iv_bolus_reduces_exactly_to_one_compartment_pk`
holds it to that: two `Simulation`s over the same 2-node circuit, one on
each PK model, compared with `np.array_equal` on concentrations, occupancy
*and* firing rates, i.e. bit-for-bit rather than within a tolerance. The
test's parameter set (Vd 0.5 L/kg, CL 25 mL/min/kg, 25 g body weight, 1
mg/kg) is chosen so the dose arithmetic is exact in binary floating point
(ke = 0.05 min^-1, C0 = 2.0), which is what makes bit-equality a fair
requirement rather than a flaky one; it also reuses the same ke = 0.05 the
existing illustrative scenarios carry. The rest of `test_pk.py` checks the
new machinery against closed-form solutions (single exponential, Bateman
function, two-compartment biexponential, mass balance into elimination,
AUC = F*Dose/CL), the refusals listed above, and the engine integration
(mixed 2-state and 2-state-plus-depot models in one `Simulation`, matching
the same models integrated standalone).

### What this does not do

- It adds no PK parameter for any drug. Fentanyl, haloperidol and diazepam
  are still not dosable in mg/kg by this codebase until P1.2-P1.4 source
  their F/ka/Vd/CL per species and route.
- It models plasma, not brain. Receptors sit in tissue, and the
  plasma-to-brain partition (Kp / Kp,uu) is a separate, drug-specific,
  published quantity that P1.5 adds. Nothing here should be read as a claim
  about brain concentration.
- It is linear PK only: first-order absorption and elimination, no saturable
  (Michaelis-Menten) clearance, no enterohepatic recirculation, no
  metabolite compartments, no protein-binding term. Diazepam's active
  metabolites in particular are a known issue that P1.4 is required to
  address explicitly rather than silently.
- It does not validate any simulated curve against a published one. That is
  P1.6, and the closed-form checks in `test_pk.py` are self-consistency
  checks of the implementation, not evidence that any drug behaves this way.

## Phase 1: literature PK parameters, haloperidol in rat (P1.3)

P1.1 built the structure and deliberately shipped no drug constants. P1.3 is
the first task to put real numbers into it, for one drug in one species:
haloperidol in the rat. The numbers live in a new module,
`src/neurocascade/drug_pk.py`, so that `pk.py` keeps its property of
containing no drug-specific constant of any kind.

### What was found, and what it supports

**IV, rat, fully sourced.** Cheng YF, Paalzow LK. "Linear pharmacokinetics of
haloperidol in the rat." Biopharm Drug Dispos. 1992;13(1):69-76.
doi:10.1002/bdd.2510130106, PMID 1554878. Rat, IV bolus at 0.5, 1.0 and
2.5 mg/kg plus 12 h infusions. Its abstract (quoted verbatim in
`drug_pk.py`, read from PubMed rather than from a secondary summary)
reports total **blood** clearance 83 mL/min/kg, Vc 5.5 L/kg, V_beta
11.1 L/kg, Vss 9.6 L/kg, terminal half-life 1.5 h, and linear disposition
over that dose range. Bioavailability of the IV route is 1 by definition.
The full text is closed access (checked via Unpaywall: `is_oa` false, no
repository copy), so rat strain, sex, body weight, anaesthesia state and
assay could not be recorded; that is a limitation of the record, not a gap
filled by assumption.

Two models are built from that one source:

- `haloperidol_rat_iv_pk` (primary), two-compartment, because the source's
  disposition is explicitly biexponential. Vc and Vss are used directly;
  V2 = Vss - Vc = 4.1 L/kg follows from the definition of Vss.
- `haloperidol_rat_iv_one_compartment_pk`, a documented collapse with
  Vd = V_beta = 11.1 L/kg, which preserves the terminal half-life and total
  exposure (AUC = Dose/CL) and, as its docstring and a test both state,
  understates the post-bolus peak about two-fold because the real drug
  starts in Vc rather than V_beta.

**Q had to be derived, and the derivation is ill-conditioned.** Published in
vivo PK tables rarely report the inter-compartmental clearance Q, and this
one does not, so the two-compartment model cannot be built from it without
solving for Q. `pk.intercompartmental_clearance_from_terminal_rate` inverts
the standard identities (`alpha+beta = k10+k12+k21`, `alpha*beta = k10*k21`)
to give `Q = beta (k10 - beta) / (k10/V2 - beta (1/V1 + 1/V2))`. That is
arithmetic on numbers the paper already reported, not a new number, and
`test_intercompartmental_clearance_solves_back_to_the_model_it_came_from`
in `test_pk.py` round-trips it against models with a known Q. But the denominator
is a small difference of comparable terms, so the answer is sensitive to how
the source's own rounding is read:

| anchor | beta (min^-1) | terminal t1/2 | derived Q (mL/min/kg) |
|---|---|---|---|
| V_beta = 11.1 L/kg, so beta = CL/V_beta | 0.0074775 | 92.7 min = 1.545 h | 114.5 |
| reported "1.5 h", so beta = ln2/90 min | 0.0077016 | 90.0 min | 141.6 |

The model anchors on V_beta, for a stated reason rather than convenience:
V_beta is printed to three significant figures and "1.5 h" to two, and the
half-life the V_beta anchor implies (1.545 h) sits inside the interval that
"1.5 h" rounds from (1.45-1.55 h), so that choice reproduces both reported
numbers within their stated precision while the reverse choice would put
V_beta at 10.8 rather than 11.1 L/kg. Both values are kept in code
(`HALOPERIDOL_RAT_IV_Q_SENSITIVITY_ML_PER_MIN_PER_KG`) and a test pins the
~24% spread, because any later use of the distribution phase (P1.6's
Cmax/Tmax validation especially) inherits that uncertainty and should not
discover it by surprise.

**The half-life test's tolerance is the source's rounding, not a chosen
percentage.** `test_drug_pk.py` integrates the model through the same ODE
protocol the engine uses, fits the terminal slope by log-linear regression
over 300-600 min (well past the ~12 min distribution phase), and requires
the result to fall in 1.45-1.55 h, the interval "1.5 h" constrains. It lands
at 92.70 min, and the test asserts that exact value too, so the real
agreement is visible rather than hidden behind a bound. This satisfies P1.3's
"simulated half-life matches the cited value" criterion without widening
anything: no tolerance in this task was chosen to make a check pass.

### What was searched for and not found

**F and ka for IP and SC in rat: absent.** This is the parameter pair the
task most wanted, since the locomotor-activity literature Phase 2 must fit
against dosed IP and SC, and it is not in the reachable literature.
Searched (August 2026): PubMed (`haloperidol AND rat* AND pharmacokinetic*
AND (subcutaneous OR intraperitoneal OR bioavailability)`, 37 hits, none an
IP or SC PK study), Europe PMC including open-access full text
(`haloperidol AND rats AND intraperitoneal AND Cmax AND Tmax`;
`haloperidol AND rats AND subcutaneous AND pharmacokinetics AND plasma AND
Cmax`), Europe PMC title/abstract sweep of all 34 rat-haloperidol
pharmacokinetics papers, and web search for a rat IV-vs-IP or IV-vs-SC
crossover. Candidates found and the specific reason each is unusable:

- Pizzolato G, Soncrant TT, Rapoport SI. J Neurochem 1984;43:724-32 (PMID
  6540296) dosed rats 0.5 and 1 mg/kg i.p. and relates cerebral glucose
  utilization to *estimated* brain concentrations; no measured
  concentration-time data, no IV arm.
- Castaing N et al. J Anal Toxicol 2006;30:419-25 (PMID 16959133) dosed
  1 mg/kg i.p. but is a postmortem-redistribution study with no in-life
  profile.
- Kapur S, VanderSpek SC, Brownlee BA, Nobrega JN. J Pharmacol Exp Ther
  2003;305:625-31 (PMID 12606608) is the standard reference for
  occupancy-matched haloperidol dosing in rats and states it established
  "the relationship between dose, plasma levels, and in vivo dopamine D2
  receptor occupancy", but the full text is closed (publisher returns 403;
  no PMC or repository copy), and the abstract gives only doses
  (0.04-0.08 mg/kg single, 0.25 mg/kg/day by pump), names its routes only
  as "daily injections or osmotic minipump infusions", and reports no PK
  parameter beyond the qualitative claim that rodent antipsychotic
  half-life is 4-6x faster than human.
- Kapur S, Barsoum SC, Seeman P (Neuropsychopharmacology 2000) gives an SC
  dose-occupancy curve (ED50 0.02 mg/kg s.c.), but occupancy is not
  concentration and no SC plasma profile is reported.
- Johnson M et al. Pharm Res 2014;31:2605-17 (PMID 24792824) fits rat plasma
  PK for haloperidol from IV/IP/SC data, and is the single most likely
  source of a real ka and F. Its only open-access copy is a green
  repository record whose PDF endpoint is bot-blocked (403 to every fetch
  attempt tried, including a browser-context fetch), so the parameter tables
  could not be read. Quoting them from a summary of a summary would be
  citing numbers nobody on this project has seen, which is the failure mode
  this codebase's rules exist to prevent. This is the first place a future
  pass with library access should look.
- The haloperidol nanoparticle/intranasal literature (e.g. the SLN/NLC
  studies reviewed in PMC8948700) reports brain-targeting ratios against IV
  or intranasal solution, not IP or SC bioavailability of the free drug.

So `drug_pk.haloperidol_rat_pk("IP"/"SC"/"PO", ...)` raises
`UnsourcedParameterError`, and the exception message *is* the search record
above. A test asserts that no `HALOPERIDOL_RAT_*_F` or `*_KA*` constant
other than the IV `F = 1` exists, so the most likely future mistake, adding
`F = 1.0` for SC on the grounds that SC bypasses first pass, fails the suite
rather than passing silently.

One physiological fact is recorded alongside that gap and deliberately not
converted into a number: the measured blood clearance (83 mL/min/kg) exceeds
rat hepatic blood flow (~55 mL/min/kg; Davies B, Morris T. Pharm Res
1993;10:1093-5, PMID 8378254), so hepatic extraction of the portal-drained
fraction of an IP dose is near-complete and F(IP) is certainly far below 1.
"Far below 1" is not a value, and the same inequality shows elimination is
not purely hepatic, which leaves even the residual direction uncertain.
`drug_pk.hepatic_extraction_is_high_note()` returns the two cited numbers so
the inequality is checkable in a test rather than asserted in prose.

**Rat blood:plasma ratio: absent.** Cheng & Paalzow report blood clearance
and blood-based volumes, so this parameter set produces whole-**blood**
concentrations, while the occupancy anchor this codebase already uses (Kapur
et al. 1997, in `scenarios.py`) is a human **plasma** concentration.
Converting between them needs Rb, which was searched for in rat and not
found. The nearest datum is human (plasma and whole-blood haloperidol
concentrations not significantly different in patients on depot haloperidol,
i.e. Rb ~ 1, with the reduced-haloperidol metabolite behaving very
differently at ~3.6). It is a different species, so it is recorded as
context and no Rb is applied; any comparison of this model's output against
a plasma measurement inherits that unquantified gap. This matters for P1.5
and P1.6 and is flagged there rather than left implicit.

### What this changes, and what it does not

- It removes no illustrative label. Every circuit-layer parameter, and the
  binding constants, are exactly where Track A/E left them. What changed is
  that haloperidol can now be dosed in mg/kg in the rat **by the IV route
  only**.
- It does not unblock Phase 2 for haloperidol. The behavioral literature
  Track E named uses IP, SC and PO dosing, and those routes still have no
  F and no ka, so the dose-to-concentration bridge is built but not yet
  crossable for the route the behavior data uses. That is a narrower and
  more specific blocker than Track E's, and P1.6 or a later pass with
  journal access can close it by reading one paper (Johnson 2014) or by
  finding a rat IV/SC crossover.
- It is blood, not plasma, and plasma, not brain. P1.5's Kp is still
  required before any of this is a statement about what a receptor sees.

### A note on P1.2's state

While reading the repository to start this task: the commit
`595a25b P1.2: Literature PK parameters: fentanyl (rat/mouse)` touches only
`tasks/backlog.json`, and `logs/task-P1.2.log` shows that tick's agent
never started (`--dangerously-skip-permissions cannot be used with
root/sudo privileges`) before the runner recorded the task as complete. No
fentanyl PK parameter set exists in the tree. P1.3 therefore had to create
`drug_pk.py` itself, and it is structured for one drug/species/route record
per block so P1.2 can be redone into it without rework. The backlog file is
left alone (the runner owns it), but P1.2 should be reopened.

## Phase 1: literature PK parameters, fentanyl in rat and mouse (P1.2)

P1.2 is the second drug to get real numbers, and it went into the block
structure P1.3 left behind in `src/neurocascade/drug_pk.py` without needing
to change `pk.py` at all. (P1.2 was originally recorded as done in error;
see "A note on P1.2's state" in the P1.3 section above. This is the redo.)

The short version: **one route, in one species, is fully sourced and
modelled: rat, intravenous.** Rat SC, rat IP and rat PO each have real
published data that fall short in a different, specific way, and each of
those ways is recorded as a searched-and-empty entry rather than patched.
The **mouse is empty for every route**, and the rat parameters are not
scaled into it.

### What was found, and what it supports

**IV, rat, fully sourced.** Yassen A, Olofsen E, Kan J, Dahan A, Danhof M.
"Pharmacokinetic-pharmacodynamic modeling of the effectiveness and safety of
buprenorphine and fentanyl in rats." Pharm Res. 2008;25(1):183-93.
doi:10.1007/s11095-007-9440-z, PMID 17914664. Open access, read in full from
PMC2190336, not from an abstract and not from a secondary summary. Male
Wistar rats (Charles River NL), 225-250 g, chronically cannulated (femoral
artery for sampling, jugular vein for dosing), conscious; fentanyl
0.015-0.085 mg/kg as a zero-order IV infusion over 20-40 min; arterial
plasma by LC-MS/MS, LLOQ 0.118 ng/mL. The paper selected among one-, two-
and three-compartment models and reports, in Table II, a NONMEM population
two-compartment fit:

| parameter | estimate | 95% CI | bootstrap (1,000 replicates) |
|---|---|---|---|
| Cl | 17.0 mL/min | 16.9–18.1 | 16.6 |
| V1 | 250 mL | 218–282 | 255 |
| V2 | 440 mL | 404–476 | 444 |
| Q | 9.7 mL/min | 8.3–11.1 | 9.7 |

This is a better evidential position than haloperidol's in three ways worth
naming, because they change what the tests can assert. The source reports
**Q**, so nothing has to be back-solved and none of P1.3's ill-conditioning
applies here. It reports a **bootstrap**, so the point estimate can be shown
not to be an artefact of one fit. And it reports **confidence intervals on
every parameter**, so the uncertainty on quantities derived from them can be
propagated rather than asserted.

It is worse in one way: it reports **no half-life**, of any phase. So for
fentanyl the half-life is the derived quantity and the parameter set is the
primary one, the exact inverse of the haloperidol record.

Two models are built from that one source:

- `fentanyl_rat_iv_pk` (primary), two-compartment. Derived terminal half-life
  **53.60 min**, distribution half-life **5.98 min**, Vss = V1 + V2 = 690 mL,
  Vz = CL/beta = 1314.5 mL.
- `fentanyl_rat_iv_one_compartment_pk`, a documented collapse with
  Vd = Vz, which preserves the terminal phase and AUC = Dose/CL and
  understates the bolus peak **5.3-fold** (Vz/V1 = 1314.5/250). That is a far
  worse trade than haloperidol's two-fold, and the docstring says so: for
  fentanyl the distribution phase is most of what happens in the first half
  hour, which is most of what the drug does.

**Absolute parameters, not per-kg ones, and why that is stored as such.**
[Yassen2008] publishes absolute mL and mL/min for a rat in a 225-250 g band
and never normalises by weight. `drug_pk.py` therefore stores the absolute
numbers and divides by the caller's `body_weight_kg` at construction. The
alternative (picking one weight inside the band and baking per-kg constants
in) would have hidden a choice inside a constant. The division is safe
because the disposition rate constants (k10 = Cl/V1, k12 = Q/V1, k21 = Q/V2)
and therefore both half-lives are weight-invariant; only the concentration a
mg/kg dose achieves scales with the weight passed.
`test_derived_half_lives_do_not_depend_on_the_body_weight_passed` proves
that rather than asserting it, and also checks that `V_central_L`,
`V_peripheral_L` and `CL_L_per_min` come back out as the paper's absolute
numbers whichever weight went in.

**The derived half-life carries its source's uncertainty.**
`fentanyl_rat_iv_terminal_half_life_interval_min()` enumerates the 16 corners
of the (Cl, V1, V2, Q) 95% CI box and returns **44.3–63.6 min**. That is an
outer bound, not a joint confidence region, because the four estimates are
correlated and the covariance is not published. The tests treat it as an
outer bound: the half-life test asserts the exact derived value (53.60 min,
to 0.05 min), and
the interval is asserted separately so its width is on the record for P1.6.

**How the "simulated half-life matches the cited half-life" criterion is
met.** There is no cited rat IV half-life to match, so the check is stated
precisely instead of stretched: `test_fentanyl_pk.py` integrates the model
through the same ODE protocol `engine.Simulation` uses, fits the terminal
slope by log-linear regression over 180-360 min (thirty distribution
half-lives past the dose), and requires agreement with the closed-form
half-life the source's own four parameters imply, to a relative 1e-4. It
lands at 53.60 min. The one *reported* rat fentanyl half-life anywhere in
this record, Canfield & Sprague's 1.10 h by the SC route, is a different
strain, dose and route and is not used as a tolerance for the IV model; it
is recorded below as an independent cross-check.

### What was searched for and not found

**SC, rat: disposition sourced, absorption absent.** Canfield JR, Sprague JE.
"In vivo pharmacokinetic, pharmacodynamic and brain concentration comparison
of fentanyl and para-fluorofentanyl in rats." Arch Toxicol.
2025;99(1):287-97. doi:10.1007/s00204-024-03887-z, PMID 39419833. Open
access, read in full from PMC11748481. Six jugular-vein-cannulated male
Sprague-Dawley rats, 326.92 ± 3.99 g, 300 µg/kg SC, plasma LC-MS/MS,
non-compartmental analysis. Table 1 (fentanyl): t1/2 1.10 ± 0.01 h,
AUC(0-∞) 112.83 ± 10.82 ng/mL·h, Cmax 82.12 ± 11.78 ng/mL, Tmax 0.5 h,
Clp/F 917.60 ± 110.11 mL/h, VD/F 1524.90 ± 368.52 mL (mean ± SEM).

Every one of those numbers is stored as a constant. None of them is enough
to build an SC model, for two independent reasons:

1. **ka is unresolved.** The sampling grid is 0, 30, 60, 120, 240, 480 min,
   so the reported "Tmax 0.5 h" is the *first post-dose sample*, not a
   located peak. The true peak lies somewhere in (0, 30] min. That bounds ka
   from below and measures nothing.
2. **The reported means are incompatible with a one-compartment absorption
   model at any ka.** For any such model the whole curve sits under
   C = F·Dose/V, and F·Dose/V = AUC(0-∞)·ke exactly, so `Cmax ≤ AUC·ke` is a
   structural ceiling. Here AUC·ke = 112.83 × ln2/1.10 = 71.10 ng/mL against
   a reported Cmax of 82.12 ng/mL, so the peak is **15.5% above the
   ceiling**.
   `fentanyl_rat_sc_one_compartment_ceiling()` returns all three numbers and
   a test pins the ratio.

Reason 2 is not a criticism of the paper: it is the fingerprint of a
distribution phase, and it agrees with [Yassen2008]'s finding that rat
fentanyl needs two disposition compartments. But a two-compartment SC model
would need that distribution phase parameterised, and a first sample at
30 min cannot do it either. Also recorded: the paper's own Vz/F has two
readings, the printed mean 1524.9 mL and the 1456.2 mL that its CL/F and
half-life imply, 4.5% apart because a mean of per-animal ratios is not the
ratio of the means. `fentanyl_rat_sc_apparent_volume_ml()` returns both.

Other SC candidates and why each is unusable: Canfield, Fry & Sprague, Drug
Alcohol Depend 2025;272:112710 (PMID 40382854) doses the same route at
50-200 µg/kg but reports the same apparent parameters with no IV comparator;
Kendall et al., J Am Assoc Lab Anim Sci 2014;53(5):478-84 (PMC4181689) is
the only other rodent study found reporting a subcutaneous fentanyl
concentration-time profile, and it dosed a *sustained-release* formulation,
whose half-life is set by the formulation's release rate rather than by
fentanyl's disposition. The paper says so itself: "due to the short duration
of action of Fent and Butp, only the SR formulations were assessed".

**IP, rat: sourced, and self-contradictory by a factor of two.** Almousa AA,
Ikeda R, Wada M, Kuroda N, Hanajiri RK, Nakashima K. J Chromatogr B.
2011;879(27):2941-4. doi:10.1016/j.jchromb.2011.08.029, PMID 21917532;
author copy read in full from the Nagasaki University repository
(hdl 10069/26235). Three male Wistar rats, 295-320 g, 0.1 mg/kg IP, plasma
at 0/10/15/20/25/30/40/90/120/180 min, HPLC-UV (LOD 0.8 ng/mL, calibrators
5-100 ng/mL). Table 3: Tmax 28.3 ± 2.9 min, Cmax 21.3 ± 4.9 ng/mL,
T1/2 198.3 ± 44.7 min, AUC(0-180) 1560 ± 290 ng·min/mL, MRT(0-180)
75.3 ± 8.7 min, CL 19.4 ± 2.6 mL/min.

This is the only rodent fentanyl dataset found with a *resolved* Tmax, which
would make it the only source of a real ka. It is still not usable:

- No IV arm, so no F, and its "CL" is defined by the paper as
  dose/AUC(0-180), a truncated AUC, so it is an upper bound on an apparent
  clearance, not a clearance.
- Its own three primary numbers do not agree. Reconstructing the curve from
  its Cmax, Tmax and T1/2 (ke = ln2/T1/2; F·D/V = Cmax·exp(ke·Tmax), which
  is exact at the peak; ka from Tmax = ln(ka/ke)/(ka−ke)) and integrating to
  180 min gives **3044 ng·min/mL against a reported 1560**, a factor of
  1.95. `fentanyl_rat_ip_auc_consistency()` returns both and a test pins the
  ratio.
- The most likely single cause is T1/2 = 198 min. It was fitted on 90-180
  min points sitting at or below the assay's lowest calibrator, in n = 3
  rats, and it is roughly three times the terminal half-life two
  better-powered rat studies give (53.6 min derived from [Yassen2008] by IV,
  66 min reported by Canfield & Sprague by SC).

What *does* corroborate: the apparent volume implied by that study's Cmax
and Tmax is **4.25 L/kg** (`fentanyl_rat_ip_apparent_volume_ml()`), against
Canfield & Sprague's SC VD/F of 4.45-4.66 L/kg by its two readings: two
labs, strains, routes and assays within ~5% on the one quantity both can
estimate. So the record is "partly corroborated, and internally
inconsistent where it matters", not "wrong". Other IP candidates: Immelman
et al., Can J Anaesth 1990;37(1):63-8 (PMID 2295107) reports serum fentanyl after IP
dosing but only as a comparator arm for extradural/subdural analgesia, with
no PK parameters and no IV arm; Yu et al., Toxicol Sci 2018;164(1):353-62 (PMID 29669042)
gives rat brain-versus-plasma exposure but only after IV dosing.

**PO, rat: F sourced, ka absent.** Schneider E, Brune K. "Distribution of
fentanyl in rats: an autoradiographic study." Naunyn Schmiedebergs Arch
Pharmacol. 1985;331(4):359-63. doi:10.1007/BF00500820, PMID 4094625.
Intragastric against intravenous fentanyl in the rat: "Its oral
bioavailability was found to be only 1.5%." This is the **only absolute
bioavailability for fentanyl in a rodent by any route** that this project
found, and it is stored as `FENTANYL_RAT_PO_F = 0.015`. It is still not a
route: a whole-body autoradiography paper reports no absorption-phase
parameter, and no other rat oral fentanyl concentration-time dataset turned
up. PO is recorded for completeness rather than need, since the rodent
behavioural pharmacology this project must fit against does not dose
fentanyl orally.

That 1.5% is also the strongest single argument against the tempting
shortcut of setting F(SC) = 1 because subcutaneous dosing bypasses first
pass. It shows how far below 1 an extravascular route can be for this
compound. The supporting inequality is kept checkable rather than asserted:
`fentanyl_hepatic_extraction_note()` returns 71.6 mL/min/kg of plasma
clearance ([Yassen2008], at the midpoint weight) against ~55 mL/min/kg of
rat hepatic blood flow ([DaviesMorris1993]). Unlike the haloperidol case,
this is *not* claimed to prove extrahepatic elimination, because converting
plasma clearance to blood clearance needs a rat blood:plasma ratio that was
searched for and not found; it establishes high extraction and stops there.

**Mouse: empty for every route.** No Vd, no CL, no terminal half-life and no
F for fentanyl in the mouse was found, by any route. Searched (August 2026):
PubMed (`fentanyl AND (mice OR mouse) AND pharmacokinetic* AND (serum OR
plasma OR clearance OR half-life)`, 11 hits, every abstract read; a broader
44-hit sweep screened by title), Europe PMC restricted to open access for
mouse fentanyl noncompartmental/Cmax/AUC/half-life full text, Unpaywall on
every closed candidate, and a web search for a repository or thesis copy of
the canonical study. Candidates and the specific reason each fails:

- Kalvass JC, Olson ER, Cassidy MP, Selley DE, Pollack GM. J Pharmacol Exp
  Ther 2007;323:346-55 (PMID 17646430, doi:10.1124/jpet.107.119560) is *the*
  source: seven opioids including fentanyl, equipotent subcutaneous doses in
  mice, serum and brain concentration time courses, unbound fractions. It is
  closed access (Unpaywall `is_oa` false with no repository copy, the
  publisher returns 403 to the article page and to both full-text PDF paths,
  and there is no PMC deposit), and its abstract reports no PK parameter.
  **This is the first paper a future pass with journal access should read.**
  It would close this gap and very likely P1.5's mouse Kp,uu as well.
- Kendall LV, Hansen RJ, Dorsey K, Kang S, Lunghofer PJ, Gustafson DL.
  J Am Assoc Lab Anim Sci 2014;53(5):478-84 (PMC4181689, read in full):
  sustained-release formulation only, as above. Female CD1 mice 20-27 g,
  3.5 mg/kg SC interscapular, Tmax 2 h, Cmax 62.1 ng/mL, t1/2 6.28 h,
  AUC(inf) 288 h·ng/mL, Cl/Fpred 12.2 L/h/kg.
- Shao et al., Drug Metab Dispos 2026;54(7):100319 (PMID 42424960) measured
  mouse fentanyl PK after an IV dose as the anchor for a cross-species
  PBPK-PD model; closed access.
- The mouse fentanyl-*analogue* literature (Bilel et al. on
  isobutyrylfentanyl and 4F-furanylfentanyl; Chen et al., Int J Mol Sci 2026
  on 25 analogues) reports PK for the analogues. Substituting an analogue's
  disposition for fentanyl's is the cross-drug substitution rule 2 forbids.
- The mouse anti-fentanyl vaccine and monoclonal-antibody literature reports
  serum and brain fentanyl under antibody sequestration, a deliberately
  perturbed system, and reports PK parameters for the antibody.

So `fentanyl_mouse_pk` raises `UnsourcedParameterError` for every route,
carrying that record, and it exists purely so that mouse-facing code gets
the record rather than an `AttributeError` and a temptation to reach for the
rat function. The rat parameters are **not** scaled allometrically into the
mouse. A single 10× body-mass allometric step with no mouse datum to check
it against produces a number with a citation attached that nobody measured,
which is the exact failure mode this codebase's rules exist to prevent.

### Design decisions worth recording

- **Apparent parameters are named apparent.** The SC and IP disposition
  constants are all divided by an unknown F, so they are named
  `..._CL_OVER_F_...` and `..._VD_OVER_F_...`. A test asserts that no
  `FENTANYL_RAT_SC_CL_ML_PER_H` or `FENTANYL_RAT_IP_CL_ML_PER_MIN` exists,
  because the single most likely way this block decays is someone dropping
  the `OVER_F` while tidying and thereby turning an apparent clearance into
  a clearance.
- **The unusable datasets are kept as data, not only as prose.** Every
  number from Canfield & Sprague and from Almousa et al. is a module
  constant, and the two contradictions are functions
  (`fentanyl_rat_sc_one_compartment_ceiling`,
  `fentanyl_rat_ip_auc_consistency`) whose arithmetic a test re-runs. A
  future pass that finds the missing ka can build the model without
  re-reading the papers, and cannot quietly inherit the contradiction.
- **Refusals distinguish "absent" from "present and unusable".** Rat SC has
  published disposition and no absorption; rat IP has a published but
  self-contradictory set; rat PO has a measured F and no absorption; mouse
  has nothing. Each raises a different record, because "not available" and
  "available and unusable" call for different next steps.
- **A separate test file.** `tests/test_fentanyl_pk.py` rather than an
  extension of `test_drug_pk.py`, because the two drugs sit in inverted
  evidential positions (haloperidol: half-life reported, Q derived;
  fentanyl: Q reported, half-life derived) and the tests have to be shaped
  differently as a result. Merging them would have blurred that.

### What this changes, and what it does not

- It removes no illustrative label. Every circuit-layer parameter and every
  binding constant is exactly where Track A/E left it. What changed is that
  fentanyl can now be dosed in mg/kg in the **rat, by the IV route only**.
- The existing calibrated mu-opioid scenario is untouched. That scenario
  still uses a one-compartment model with a *human* FDA-label redistribution
  half-life (13 min); this block is rat, plasma, and two-compartment, and
  the two are not merged here. A rat cross-check worth noting: 0.05 mg/kg IV
  into V1 = 250 mL for a 237.5 g rat is 141 nM, the same order as the
  148.6 nM (50 ng/mL) human overdose exposure the calibrated mu-opioid
  evidence uses. That is weak evidence on its own, but a unit error would have
  shown up as orders of magnitude, and a test asserts it.
- It does not unblock Phase 2 for fentanyl. The rodent behavioural
  literature doses fentanyl SC and IP, and both routes still lack a ka.
  That is a narrower blocker than Track E's, and a specific one: what is
  needed is a rat fentanyl study with sampling in the first 30 minutes and,
  ideally, a matched IV arm. Neither this record's SC source (first sample
  at 30 min) nor its IP source (self-inconsistent, n = 3) is it.
- It is plasma, not brain. P1.5's Kp is still required before any of this
  is a statement about what a receptor sees, and for fentanyl P1.5 also
  inherits the P-glycoprotein question: Yu et al., Toxicol Sci
  2018;164(1):353-62 (PMID 29669042) report that co-administering the P-gp
  inhibitor tariquidar raises the rat unbound brain-to-plasma concentration
  ratio 2.9-fold at 40 µg/kg fentanyl. A single partition coefficient will
  not represent a ratio that moves that much with a co-administered drug.

## Phase 1: literature PK parameters, diazepam in rat and mouse (P1.4)

P1.4 is the third drug, and the first one where **no single paper covers a
single route**. Haloperidol and fentanyl each had one source per modelled
route; diazepam's intravenous disposition, intraperitoneal disposition,
intraperitoneal absorption and oral data come from four different papers,
spanning 1976 to 2025. That changed what the task is: less transcription,
more deciding which seams may be crossed and recording every one that is.

The short version: **two routes are modelled in the rat, IV and IP.** IV is
a single-paper collapse. IP is an explicit cross-study composite and is
flagged as one everywhere it appears. PO has a complete, modern, open-access
apparent parameter set that turns out **not to identify an absorption rate
constant**, so it is recorded and not implemented. SC and the whole mouse are
empty. And diazepam brought a question the other two drugs did not: an
active metabolite that demonstrably matters inside the simulated window,
which P1.4 was required to address explicitly and does, as a documented model
limitation.

### What was found, and what it supports

**IV, rat, single source.** Klotz U, Antonin KH, Bieck PR. "Pharmacokinetics
and plasma binding of diazepam in man, dog, rabbit, guinea pig and rat."
J Pharmacol Exp Ther. 1976;199(1):67-73, PMID 789856. Closed access
(Unpaywall `is_oa` false, no repository copy, publisher 403), so the abstract
is the whole record, and it is quoted verbatim in `drug_pk.py` rather than
paraphrased. Rat: **terminal half-life 1.1 h, total plasma clearance
81.6 mL/min/kg**, plasma binding 86.3% (diazepam) and 90.5%
(desmethyldiazepam), disposition explicitly biexponential and fitted with a
two-compartment open model.

`diazepam_rat_iv_pk` is a one-compartment collapse with
`Vd = V_beta = CL/beta = 7.77 L/kg`, which reproduces the 1.1 h terminal
half-life exactly and keeps `AUC = Dose/CL`. The limitation is worse than
the other two drugs' collapses and is stated as such: Klotz et al. report
**no central volume and no Vss for any species**, so the distribution phase
is not recoverable at all. Haloperidol's collapse understates the bolus peak
2-fold and fentanyl's 5.3-fold, both computable because both sources printed
a central volume; here the factor cannot be computed, only bounded below by
1. The paper's own alpha half-life figure, "0.3-1.0 hours", is a range
across all four animal species and is stored under the name
`DIAZEPAM_ALPHA_HALF_LIFE_RANGE_ALL_SPECIES_H` with a test asserting no
rat-specific distribution half-life constant exists, because taking the
middle of a four-species range and calling it the rat's is precisely the
invented number this codebase forbids.

**IP, rat, a two-paper composite - the route Phase 2 actually needs.** The
diazepam target Track E named is the elevated plus maze at 1-2 mg/kg, and
that literature doses intraperitoneally. No rat study found reports both the
disposition and the absorption for this route, so the model is built from
two, and both halves are named after their own paper in code:

- *Disposition:* Friedman H, Abernethy DR, Greenblatt DJ, Shader RI.
  Psychopharmacology (Berl). 1986;88(3):267-70, doi:10.1007/BF00180822,
  PMID 3083446. Closed access; abstract is the record. Single 5 mg/kg IP
  dose, four rats sacrificed at each of ten times from 5 min to 6 h, plasma
  and brain assayed for diazepam and desmethyldiazepam. Plasma half-life
  **0.88 h**, brain half-life 0.89 h, **apparent Vd 19.3 L/kg**, **apparent
  clearance 255 mL/min/kg**, free fraction 19.6%, brain:plasma 4.5 ± 0.1.
- *Absorption:* Hironaka T, Fuchino K, Fujii T. J Pharmacol Exp Ther.
  1984;229(3):809-15, PMID 6726659. Closed access; abstract is the record.
  0.9 mg/kg IP in **21-day-old** male rats: **ka = 0.158/min**, with k12
  0.015, k21 0.136 and kel 0.013/min.

There is no IV arm in either, so the volume and clearance are apparent, and
they are named `..._VD_OVER_F_...` and `..._CL_OVER_F_...` with a test
asserting the un-suffixed names do not exist.

**The apparent parameterisation, and why it is not a loophole.** `pk.py`
refuses to build an extravascular model without an explicit F, on purpose
(P1.1's "refusing to default"). P1.4 adds one sanctioned alternative,
`apparent_parameters=True`, because the diazepam IP data are the case that
rule was not written for. For a one-compartment extravascular model,

    C(t) = Dose / (V/F) · ka/(ka − ke) · (e^(−ke·t) − e^(−ka·t)),
    ke = (CL/F)/(V/F),

and **F appears nowhere**. It is not that F is unknown and we are pretending;
it is that F is not separately identifiable from, and not separately needed
for, the concentration a given mg/kg dose achieves. So the flag sets `F = 1`
as bookkeeping, records `apparent_parameters = True` and
`F_is_measured = False`, and says so in `repr`. What such a model cannot
report is a true Vd, a true CL, or the absorbed fraction - and asking it for
those is asking for the one number that was divided out.
`test_pk.py::test_apparent_parameterisation_is_flagged_and_cannot_double_apply_f`
holds the distinction: the apparent and measured-F=1 models produce
bit-comparable curves and differ only in what they claim, which is exactly
why the flag has to exist. Passing an F alongside apparent parameters raises,
and the flag is rejected outright for IV.

**How the "simulated half-life matches the cited value" criterion is met,
twice, without widening anything.**

| route | model | simulated t½ | cited t½ | note |
|---|---|---|---|---|
| IV | Klotz collapse | 66.00 min | 1.1 h = 66 min | inside the 1.05–1.15 h interval "1.1 h" rounds from; exact by construction |
| IP | Friedman + Hironaka | 52.46 min | 0.88 h = 52.8 min | **−0.64%**, and the gap is asserted at that value |

The IP row is the interesting one. Friedman et al. print three mutually
constrained numbers, and CL/F ÷ V/F = ln2/52.46 min, not ln2/52.8 min. The
model is built from the volume and clearance, because those are what set the
concentration scale, so it lands at 52.46. Rather than widening the tolerance
to swallow the printed 0.88 h, `diazepam_rat_ip_reported_half_life_consistency()`
returns the real relative error and the test asserts it exactly and requires
it to stay under 1%. Both readings are inside what the printed precision of
19.3 and 255 leaves open, so the source is internally consistent; the point
is that the check reports the discrepancy instead of hiding it.

**What licenses the composite, and what does not.** The one available check
is that the three independent rat diazepam studies in this record agree on
the disposition envelope: 59.7 min (Hironaka's own constants, derived through
the two-compartment identities), 52.8 min (Friedman), 66.0 min (Klotz IV).
A **25% spread** across two routes, three assays and two decades, asserted at
that width by `test_the_three_rat_studies_agree_on_the_disposition_envelope`.
This is deliberately not called good agreement, and it is a weak check: three
studies agreeing on elimination does not prove an absorption constant
transfers between them. It only rules out the failure mode where the ka
source's animals were handling the drug so differently that nothing about
them transfers.

**The composite's most likely error, stated with its direction.** ka =
0.158/min puts Tmax at **17.1 min**, while Friedman et al.'s first post-dose
sample - the one that would locate the true peak in those animals - is at
5 min. If their curve was already falling at 5 min, the true ka is
several-fold higher and this model absorbs too slowly, understating early
concentrations.
That cannot be settled: the paper is closed and its abstract reports no Tmax.
And the ka that *is* available comes from 21-day-old weanlings at a 5.6-fold
lower dose, which is recorded as data (`DIAZEPAM_RAT_IP_KA_SOURCE_AGE_DAYS`)
so a test can assert the caveat is still on the record. A weanling rat is not
a small adult.

**Bioavailability, derived and kept derived.** F = (AUC/Dose)_extravascular ÷
(AUC/Dose)_IV is algebraically identical to CL_IV ÷ (CL/F)_extravascular, so
the two clearances already published give:

| route | derivation | F |
|---|---|---|
| IV | by definition | 1 |
| IP | 81.6 ÷ 255 | **0.320** |
| PO | 81.6 ÷ 996.3 (printed CL/F), or ÷ 915.7 (Dose/AUC) | **0.082–0.089** |

These are cross-study derivations and inherit everything that differs between
a 1976 GLC-ECD study and a 1986 or 2025 one. So they are returned by
functions, not stored as constants, and a test asserts that no
`DIAZEPAM_RAT_IP_F` or `DIAZEPAM_RAT_PO_F` constant exists - because the
likely decay path is someone promoting a derived number to a measured-looking
one. Critically, **the IP model does not use its F**: the apparent
parameterisation needs none. The numbers are recorded because P1.4 asked for
an F by route, and because ~32% by IP against ~8% orally is the coherent
shape for a high-extraction drug given by a partly-portal route.
`diazepam_hepatic_extraction_note()` keeps the supporting inequality
checkable: 81.6 mL/min/kg of plasma clearance against ~55 mL/min/kg of rat
hepatic blood flow (Davies & Morris 1993). As with fentanyl, that is not
claimed to prove extrahepatic elimination - a rat blood:plasma ratio would be
needed and was searched for and not found - it establishes high extraction
and stops.

### What was searched for and not found

**PO, rat: sourced, and it does not identify ka.** Hua A, Wang L, Zhu Y,
Zhou Q, Mamun AA, Wang S, Wu F, Xu M. "Effects of Acacetin on the
Pharmacokinetics of Diazepam in vivo and in vitro." Iran J Pharm Res.
2025;24(1):e164825, doi:10.5812/ijpr-164825, PMID 41477118, open access
(PMC12749239, read in full). Its **control arm** is a real rat oral diazepam
dataset: six male Sprague-Dawley rats, 220 ± 20 g, diazepam 10 mg/kg by
gavage, plasma at 0.083–8 h by UPLC-MS/MS. Table 1: Cmax 57.75 ± 24.81 µg/L,
Tmax 1.33 ± 0.52 h, t½z 1.81 ± 1.26 h, AUC(0–∞) 182.02 µg/L·h, CLz/F
59.78 ± 18.43 L/h/kg, Vz/F 164.39 ± 141.97 L/kg. Every number is stored.

No model is built from it, for a reason that is arithmetic rather than
judgement. Two absorption rate constants come out of the same paper depending
on which of its numbers you read, both computed with the same
ke = ln2/1.81 h:

- from the reported **Tmax** alone, solving Tmax = ln(ka/ke)/(ka − ke) at
  1.33 h: **ka = 0.0217/min**;
- from the reported **Cmax and AUC** alone - for any one-compartment
  extravascular model F·Dose/V = AUC(0–∞)·ke exactly, and
  Cmax/(F·Dose/V) = (ke/ka)^(ke/(ka−ke)), a function of ka/ke only; the
  reported Cmax is 82.9% of AUC·ke, needing ka/ke = 15.6, i.e.
  **ka = 0.0996/min** - which would put Tmax at 29.5 min, not 80.

**4.6-fold apart** (`diazepam_rat_po_ka_is_not_identified()`, with a test).
Unlike the fentanyl SC case, the reported Cmax does *not* breach the
structural ceiling AUC·ke, so a one-compartment model is not excluded here;
what is excluded is any single ka reproducing the study's own three headline
numbers. Two smaller inconsistencies are pinned alongside it: printed Vz/F is
5.0% from (CL/F)/λz, and printed CL/F is 8.1% from Dose/AUC - both the usual
mean-of-per-animal-ratios artefact, the same one the fentanyl SC record
found. The likely cause of the ka split is ordinary and not a fault in the
paper: a mean of per-animal Cmax on a discrete grid is biased upward relative
to the mean curve's peak. But an absorption constant that moves 4.6-fold
depending on which reported number it is read from is not one to build on.
Two further caveats recorded rather than smoothed: the controls were
vehicle-pretreated (DMSO/PEG200/water i.p. for two weeks) rather than naive,
which is not obviously neutral for a CYP-cleared drug, and the
terminal-phase estimates carry ~80% relative dispersion.

The study that would fix this was found and could not be read: Díaz-García
JM, Oliver-Botana J, Fos-Galve D, J Pharm Sci 1992;81(8):768-72
(doi:10.1002/jps.2600810810, PMID 1403721) with its companion Eur J Drug
Metab Pharmacokinet 1991;Spec No 3:94-101 (PMID 1820943) gave **4 mg/kg IV
and 4 mg/kg oral in the same rats** - exactly the crossover needed - but both
are closed access and neither abstract prints a single control-group value,
reporting only that the cirrhotic group's half-life and volume rose. That is
the first paper a future pass with journal access should read for this route.

**SC, rat: nothing at all.** No F, no ka, no Cmax/Tmax, no apparent clearance
or volume. Searched (August 2026): Europe PMC `TITLE:"diazepam" AND
(TITLE:"subcutaneous" OR ABSTRACT:"subcutaneous")` (36 hits, every title
screened), plus a rat/plasma/kinetics-qualified variant, plus the sweeps run
for the oral route. Every subcutaneous hit is one of three things: a
different drug in the same paper (rectal diazepam against subcutaneous
morphine-scopolamine, PMID 4072591), the word used for a non-diazepam
co-treatment, or a chronic-dependence study dosing by implanted capsule or
minipump - where the implant's release rate, not diazepam's absorption, sets
the profile, the same objection that disqualified the sustained-release
fentanyl studies in P1.2. This route is recorded for completeness rather than
need, since the EPM literature doses IP and that route is modelled. F(SC) is
not set to 1 on the grounds that SC bypasses first pass, for a reason this
block makes concrete rather than assert: the one extravascular route with a
derivable F here comes out at 8–9%.

**Mouse: empty for every route.** Searched (August 2026): Europe PMC
`TITLE:"diazepam" AND (TITLE:"mice" OR TITLE:"mouse")` (444 hits screened by
title), three further clearance/half-life/volume-qualified queries including
one sorted by citation count, a `TITLE:"benzodiazepine"` cross-query, web
search for the classic mouse disposition studies, and Unpaywall on every
closed candidate. Candidates and why each fails:

- Greenblatt DJ, Sethy VH. Psychopharmacology (Berl) 1990;102(3):373-8
  (doi:10.1007/BF02244106, PMID 1979181) is the closest thing to the right
  study: male CF-1 mice, IV diazepam at 3 and 10 µmol/kg, **plasma and brain
  concentrations from 1 min to 24 h**, with ex vivo ³H-flunitrazepam
  displacement occupancy in the same animals. Closed access, no PMC deposit,
  and its abstract reports no PK parameter at all - only that diazepam
  "entered brain rapidly, and was also cleared rapidly". **This is the first
  paper a future pass with journal access should read**; it would close this
  gap and P1.5's mouse brain:plasma partition together.
- File SE, Greenblatt DJ, Martin IL, Brown C, Psychopharmacology
  1985;86:137-41 reports mouse brain diazepam concentrations and receptor
  occupancy across strains, but as correlates of anticonvulsant duration, not
  as a parameterised time course; closed.
- The Marcucci/Garattini series (Eur J Pharmacol 1968-70; Br J Pharmacol
  1968;34:667P-668P and 1970;39:455P-457P) compared mouse and rat brain and
  blood levels after 5 mg/kg IV and is the origin of the finding that
  N-desmethyl metabolites accumulate in mouse brain but not rat brain. The
  Eur J Pharmacol papers are closed; the two Br J Pharmacol items are two-
  and three-page **scanned meeting proceedings** whose text is not
  machine-readable. No number from them is quoted here.
- Takahashi Y et al., Br J Pharmacol 2026, doi:10.1111/bph.70430,
  PMID 42037156, gives IV diazepam PK in midgestational
  **pregnant** mice with fetal Cmax for diazepam, nordazepam, oxazepam and
  oxazepam-glucuronide. Pregnancy is a deliberately altered disposition
  state, the reported parameters are fetal-transfer ratios rather than
  maternal PK, and it is closed access.
- The intranasal-formulation mouse literature (e.g. PMC12963071, 2026: C57
  mice, 1 mg/kg intranasal, plasma 49.1 ng/mL and brain 156.3 ng/g at 10 min)
  reports two-timepoint concentrations in pilocarpine-treated animals, no PK
  parameter, and a route this model does not have.

So `diazepam_mouse_pk` raises for every route and the rat parameters are not
scaled allometrically. Beyond the general objection to a single unchecked 10×
allometric step, this drug has a specific one: rat and mouse are documented
to differ in *which metabolite dominates*, so rat numbers would import the
wrong metabolic picture along with the wrong disposition.

**Rat blood:plasma ratio: absent.** Klotz et al. state they measured diazepam
"in blood and plasma", so the ratio was within that paper's reach, but the
abstract reports only the bound fractions and the full text is closed. All
three diazepam parameter sets here are therefore plasma, which is at least
self-consistent, and the missing Rb is why the clearance-versus-hepatic-flow
comparison is left as an inequality about extraction.

### The active-metabolite question, answered

P1.4 was required to state explicitly whether nordiazepam, oxazepam or
temazepam matter on the timescale this model simulates. **Nordiazepam
(desmethyldiazepam) does, and it is not modelled.** That is recorded as a
documented limitation, in code as `DIAZEPAM_ACTIVE_METABOLITE_LIMITATION`
with a test that fails if the record loses any leg of the argument. Three
facts settle it, each from a primary source read for this task:

1. **Exposure.** After diazepam 10 mg/kg PO in the rat, nordiazepam's plasma
   AUC(0–∞) is 139.73 µg/L·h against the parent's 182.02. In *molar* terms -
   the units that matter for receptor occupancy, at 270.71 vs 284.74 g/mol -
   that is 0.516 vs 0.639 µmol/L·h: the metabolite carries **81% of the
   parent's molar exposure** (Hua et al. 2025, Tables 1 and 2, same animals).
   The mass-based ratio, 0.768, understates it by 5%, so the conversion is
   done in code (`diazepam_nordiazepam_molar_exposure_ratio()`) rather than
   in a comment. Independently, by the IP route, Friedman et al. 1986 find
   nordiazepam in rat plasma and brain with half-lives of 1.11 and 1.09 h
   against diazepam's 0.88 and 0.89 h, and brain:plasma 3.5 against 4.5.
2. **Potency.** Nordiazepam is not a bystander at the site this codebase
   models. Gobbi M, Barone D, Mennini T, Garattini S, J Pharm Pharmacol
   1987;39(5):388-91 (doi:10.1111/j.2042-7158.1987.tb03404.x, PMID 2886589)
   find, in rat cerebral cortex and cerebellum, that diazepam and
   desmethyldiazepam have **equal affinity** at the central benzodiazepine
   receptor while desmethyldiazepam "acts as a partial agonist" there. No
   numeric Ki appears in the reachable abstract and the full text is closed,
   so no Ki is recorded - the equal-affinity and partial-agonist findings are
   qualitative and are used qualitatively.
3. **Timescale.** Nordiazepam's rat Tmax is 1.25 h by the oral route and it
   is already measurable in plasma and brain at Friedman et al.'s early IP
   time points. The existing GABA-A scenario runs to t = 40 min and the EPM
   protocols Phase 2 must fit test 30–60 min after an IP dose. The metabolite
   is inside the window by both routes measured - this is not a late
   accumulation effect a short simulation could ignore.

**Consequence, stated and not patched:** every diazepam simulation this
codebase produces **understates total benzodiazepine-site occupancy**,
because roughly 80% more equal-affinity ligand, at reduced efficacy, is
present and unmodelled. This sharpens, with numbers, the qualitative caveat
`scenarios.gaba_a_pk_models_calibrated()` already carried from Track A.

It is not fixed here, and the reason is the codebase's first rule. The fix is
structural - a metabolite compartment feeding a second `Ligand` at the same
`ReceptorSite` - and it needs two numbers this project does not have: a rat
formation fraction for the demethylation step (the reported nordiazepam
CL/F of 85.13 L/h/kg is apparent in *both* the parent's F and the formation
fraction, so it does not give one), and a numeric Ki and efficacy for
nordiazepam at the benzodiazepine site. Inventing an efficacy for a partial
agonist is exactly the kind of number this codebase forbids, and
`test_active_metabolite_limitation_is_recorded_and_not_modelled` asserts both
that the record is intact and that no nordiazepam ligand has quietly appeared
in `scenarios.gaba_a_benzodiazepine_ligands_calibrated()`.

**Temazepam and oxazepam** are not quantified for the rat by either rat
dataset in this record: Hua et al. measured diazepam and nordiazepam only,
Friedman et al. diazepam and desmethyldiazepam only. A rat plasma or brain
concentration-time course for either after diazepam was searched for and not
found, so nothing is asserted about their magnitude in the rat either way.
They are demonstrably measurable in the *mouse* (Takahashi et al. 2026 report
nordazepam and oxazepam in midgestational mice after IV diazepam, both
exceeding the parent's fetal Cmax), which is one more reason the mouse is not
modelled from rat numbers.

### Design decisions worth recording

- **`apparent_parameters` is a new flag on `pk.py`, not a new default.** It
  is the minimum structural change that lets an honest extravascular model
  exist without an F, and it is refused for IV, refused when an F is passed
  alongside it, and visible in `repr`. P1.1's refusal to default F is
  otherwise untouched.
- **Composites are named after their sources.** `diazepam_rat_ip_pk`'s
  `name` string carries both papers, its docstring says which half came from
  where, and the ka source's animal age and dose are module constants so a
  test can assert the caveat has not been dropped. The alternative - one
  parameter set with four citations in a comment - would have made the seam
  invisible within a release.
- **Derived numbers are functions; measured numbers are constants.** Both
  bioavailabilities and every consistency check are functions that recompute
  from the transcribed values. A test asserts the corresponding constants do
  *not* exist. This is the same discipline P1.2 applied to apparent
  clearances, generalised: the module's shape should make it hard to mistake
  a derivation for a measurement.
- **The real errors are asserted at their real values.** −0.64% (IP half-life
  vs printed), 25% (three-study disposition spread), 4.6× (PO ka ambiguity),
  5.0% and 8.1% (PO internal inconsistencies). No tolerance in this task was
  widened to make a check pass; where a model and its source disagree, the
  test states the size of the disagreement.
- **A third test file.** `tests/test_diazepam_pk.py`, for the same reason
  P1.2 got its own: this drug's evidential position is a third distinct one
  (no single source covers a route), and it needs a class of assertion the
  other two files do not have - that the composite's seams stay visible.

### What this changes, and what it does not

- It removes no illustrative label. Every circuit-layer parameter and every
  binding constant is where Track A/E left it. What changed is that diazepam
  can now be dosed in mg/kg in the rat by the IV and **IP** routes.
- **It is the first task that partially unblocks Phase 2.** Haloperidol and
  fentanyl both ended with "the behavioural literature uses a route we cannot
  model". Diazepam does not: the EPM literature doses IP, and IP is modelled.
  The remaining blockers for the GABA-A system are therefore no longer
  PK-structural - they are Track E's original one (no directly quotable
  diazepam percent-open-arm-time figure) plus P1.5's brain partition plus the
  unmodelled metabolite above. That is a different and much more tractable
  list than the one Track E left.
- **The margin over Ki is smaller than it looks, which is why P1.5 matters.**
  1 mg/kg IP in a 250 g rat peaks at 145 nM total plasma, about 40× the
  3.6 nM benzodiazepine-site Ki the calibrated scenario uses - but only 19.6%
  of that is unbound, so unbound plasma peaks near 28 nM, about 8× Ki. Brain
  total is a further 4.5-fold higher and brain binding is unknown here.
  Whether the receptor sees 8× Ki or 40× Ki is exactly P1.5's question, and
  `test_ip_dose_reaches_concentrations_above_the_calibrated_ki` records that
  the answer is not obvious rather than asserting a comfortable margin.
- It is plasma, not brain. Unusually for this project, the brain half of the
  answer is already partly in hand for this drug: Friedman et al. report a
  rat brain:plasma ratio of 4.5 ± 0.1 for diazepam and 3.5 ± 0.2 for
  nordiazepam, stored as `DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO` and untouched
  by P1.4, because converting a total-brain:total-plasma ratio into the
  Kp,uu a receptor layer needs is P1.5's job and needs a brain unbound
  fraction this task did not search for.

## Phase 1: plasma-to-brain partition (P1.5)

### The gap this closes, and the one it turns out to open

P1.1 through P1.4 built a dose → **plasma** bridge and said so at every
step: "It models plasma, not brain. Receptors sit in tissue, and the
plasma-to-brain partition (Kp / Kp,uu) is a separate, drug-specific,
published quantity that P1.5 adds." P1.4 sharpened it into a number: 1 mg/kg
diazepam IP in a 250 g rat peaks at 145 nM total plasma, ~40× the 3.6 nM
benzodiazepine-site Ki, but only 19.6% of that is unbound and brain is a
further 4.5-fold up, so "whether the receptor sees 8× Ki or 40× Ki is exactly
P1.5's question".

That framing was slightly too generous to the pre-P1.5 code, and the
correction is the substance of this task. The occupancy ODE is

    d(occ_i)/dt = kon_i * C_i * (1 - sum(occ)) - koff_i * occ_i,

with every calibrated `kon` in this codebase derived as `koff / Kd` from an
*in vitro binding-assay* Kd - Volpe et al.'s [3H]-DAMGO competition for
fentanyl/naloxone, Möhler & Okada's [3H]diazepam saturation binding, Kapur's
D2 numbers. A binding-assay Kd is expressed in **free ligand concentration**.
So `C_i` was never allowed to be a total plasma concentration in the first
place: it has to be the free concentration at the receptor, which is unbound
drug in brain. The pre-P1.5 model was not "plasma instead of brain", it was
"total plasma instead of unbound brain", which is two errors pointing in
opposite directions (protein binding lowers it, tissue partition raises
it) and no reason for them to cancel.

### Kp and Kp,uu are not two names for the same thing

With C_p, C_b the total plasma and total brain concentrations, Cu_p, Cu_b the
unbound ones, and fu_p, fu_b the unbound fractions:

    Kp    = C_b / C_p
    Kp,uu = Cu_b / Cu_p = Kp * fu_b / fu_p

`Kp` is dominated by **tissue binding**. Haloperidol is the cleanest
illustration available in this project's own sourced numbers: in the mouse
its Kp is 13.0 and its Kp,uu is 1.06, because fu_brain is 0.007. A Kp of 13
sounds like avid brain uptake and is in fact passive equilibrium plus very
heavy binding to brain lipid and protein. Multiplying a plasma concentration
by 13 and feeding it to the occupancy ODE would overstate the driving
concentration by 1/fu_b ≈ 140-fold.

So the layer added here refuses that substitution structurally rather than
warning about it in a comment. `brain_partition.BrainPartition` requires an
explicit `basis` of `"Kp"` or `"Kp,uu"` (no default), and:

- a Kp-basis partition **with no measured fu_brain** exposes
  `total_brain_factor` and raises `UnusablePartitionError` from
  `unbound_brain_factor`;
- a Kp,uu-basis partition **with no measured fu_plasma** also raises, because
  `pk.py` produces *total* plasma and Cu_b = Kp,uu · fu_p · C_p;
- attaching a non-drivable partition to a PK model raises **at construction**,
  with the drug's own citation in the message, not at the first RHS
  evaluation.

This is the same "Refusing to default" rule P1.1 applied to bioavailability,
applied to the factor that most changes occupancy.

### Why the partition is a scalar and not a compartment

The alternative implementation is a brain compartment with its own state and
its own influx/efflux clearances. It was rejected for this task on evidence,
not convenience: both drugs for which an equilibration observation could be
found equilibrate fast relative to this codebase's minute-scale time unit.
Friedman et al. measured a diazepam brain half-life of 0.89 h against a
plasma half-life of 0.88 h in the same animals - parallel decline, i.e. brain
tracks plasma. Hug & Murphy report that after 50 µg/kg IV fentanyl,
"concentrations in brain, heart, and lung equilibrated with that in plasma
before 1.5 min and declined at the same rate". Under that condition an
instantaneous linear partition is the correct reduction, and a kinetic brain
compartment would need influx/efflux clearances that are not published for
these drugs - it would add three invented numbers to model a transient that
lasts under two minutes.

The assumption is tracked rather than assumed globally:
`BrainPartition.equilibration_evidence` holds the published observation, and
`assumes_rapid_equilibrium_unverified` is True where none was found (which is
the case for haloperidol). A drug that is a strong P-gp substrate with a long
brain equilibration half-life would need the larger model, and that is
recorded as a limitation rather than approximated.

Because the partition is linear, the whole layer collapses to one scalar per
ligand. `pk.py` gains `receptor_concentration_factor` (exactly 1.0 with no
partition) and `receptor_concentration(state)`; `engine.Simulation` holds
`receptor_concentration_factors` and computes `C_receptor = C * factors` once
per RHS call, driving `occupancy_rhs` from that instead of from `C`. Keeping
it a scalar rather than a callable makes the linearity assumption impossible
to violate by accident and keeps the per-step work vectorized.

`result["concentrations"]` keeps its original name, meaning and values
(central compartment, i.e. plasma or blood); `result["plasma_concentrations"]`
is an explicit alias for call sites that want to say which they mean, and
`result["receptor_concentrations"]` is what the receptor layer actually saw.
`model.receptor_compartment` returns `"unbound brain"` or `"total plasma (no
brain partition attached)"`, so "this simulation ran on plasma" is an
inspectable fact rather than something a reader has to infer from an absent
attribute.

### What was found, per drug

Coverage came out uneven, and the unevenness is the result rather than an
accident of effort: the literature reports Kp far more often than Kp,uu,
because Kp needs only two assays and Kp,uu needs a brain-tissue-binding
measurement as well.

**Diazepam, rat - complete, and attached by default.** Every factor exists
and two of the three come from the same study and the same animals as the IP
disposition parameters P1.4 already uses:

| quantity | value | source |
|---|---|---|
| Kp (total brain : total plasma) | 4.5 ± 0.1 | [Friedman1986], rat, 5 mg/kg IP |
| fu,plasma | 0.196 | [Friedman1986], same study |
| fu,brain (whole brain) | 0.040 ± 0.006 | [Gustafsson2019], Sprague-Dawley rat, brain-homogenate equilibrium dialysis |
| **Kp,uu (derived)** | **0.918** | Kp · fu,brain / fu,plasma |

[Gustafsson2019] (Sci Rep 9:5308, open access, Table 2 read directly) is a
different laboratory and a different method, so this is a two-study
composite, labeled as one exactly like P1.4's IP disposition model. Its
independent corroboration is [Friden2009], which measured rat diazepam
Kp,uu = 1.07 directly, 14% from the derived 0.918.

The disagreement worth recording is in the *other* coefficient: [Friden2009]
reports Kp = 2.28 against [Friedman1986]'s 4.5, a factor of 2. That the two
studies agree twice as well on Kp,uu as on Kp is the behaviour the
distinction predicts, and it is asserted as a test
(`test_the_two_rat_diazepam_kp_values_disagree_and_that_is_recorded`) rather
than mentioned: Kp depends on the tissue-binding assay and on when brain and
plasma were sampled, Kp,uu divides both out.

One asymmetry is worth stating because it is counter-intuitive: the factor
the model actually multiplies by is `fu_brain · Kp = 0.18`, which does **not**
contain fu_plasma. So the 43% disagreement between [Friedman1986]'s
fu_plasma of 0.196 and [Klotz1976]'s 0.137 - recorded in P1.4 and left
unresolved - moves the *reported* Kp,uu (0.918 vs 1.31) and does not move the
receptor-driving concentration at all.

[Gustafsson2019]'s central finding is that brain tissue binding is **not**
homogeneous across regions (rat diazepam fu,brain runs 0.037 in frontal
cortex to 0.055 in cerebellum, a factor of 1.5). The default partition uses
the whole-brain value because a non-regional model is not entitled to more;
the regional values are stored, because Phases 3-4 make this codebase
regional and this is precisely the number that will stop being scalar then.

**Haloperidol, rat - a Kp, and nothing else.** [Sunderland1987] reports a rat
brain:plasma ratio of 22 (abstract quoted verbatim; the full text is closed,
so dose, route, timing, strain, sex and n were not obtainable, and the assay
is a *radioreceptor* assay, which measures neuroleptic-equivalent D2-binding
activity rather than chemically specific haloperidol). No rat fu,brain was
found, so this cannot yield an unbound brain concentration and is not
attached to the rat IV model, which reports itself as plasma-driven.

**Fentanyl, rat - a Kp read off a figure, and nothing else.**
[Canfield2025] - the same open-access study, the same six 355 g male
Sprague-Dawley rats and the same 300 µg/kg SC dose that already supply this
module's fentanyl SC disposition constants - reports region-resolved
brain:plasma ratios at 1 h post-dose. The catch is that it reports them
**only in Fig. 5B**: there is no printed value in text, table or supplement.
The constants recorded are therefore figure reads (hippocampus ~15, medulla
~18, striatum ~16, frontal cortex ~17, ±2 ratio units), explicitly labeled as
such in `BrainPartition.source`, and cross-checked against the same figure's
panel A: mean brain concentration ~1185 ng/g divided by mean ratio 16.5 gives
a 1 h plasma concentration of 71.8 ng/mL, against 59.9 ng/mL propagated from
the paper's *printed* Cmax (82.12 ng/mL at 0.5 h) and half-life (1.10 h) - a
20% gap in the direction the crude mono-exponential estimate is biased
(`fentanyl_rat_brain_plasma_figure_read_cross_check`, and a one-sided test).
No rat fu,brain was found here either, so fentanyl is also plasma-driven.

**Mouse, diazepam and haloperidol - complete, and deliberately unused.**
[Doran2005] (Kp) and its companion [Maurer2005] (fu,plasma, fu,brain) give
diazepam 2.00 / 0.098 / 0.050 → Kp,uu 1.02, and haloperidol 13.0 / 0.087 /
0.007 → Kp,uu 1.06. Both primaries are closed access; the values reach this
project as attributed quotes in [Ma2024]'s Table 1 (open access, read
directly), and that second-hand status is recorded on every constant. They
are recorded and **not** used, because this module has no mouse PK for either
drug - a partition is not a licence to invent the kinetics it multiplies -
and the roadmap's eventual move to mouse should start from a sourced number
rather than a search. The one check available on a second-hand quote is
internal consistency: Kp · fu,b / fu,p reproduces the printed Kp,uu for both
drugs to within the table's rounding, which a transcription error in any of
the four numbers would break.

### What was searched for and not found

Recorded as data in `SEARCHED_AND_EMPTY_BRAIN_PARTITION`, so the error a
caller hits is the search record, following the Track E / P1.2-P1.4 pattern.

**Rat fu,brain for fentanyl.** Yu et al. (Toxicol Sci 2018;164:353-62) *did*
measure a rat unbound brain-to-plasma ratio for fentanyl at 40 µg/kg IV - the
abstract says so, reporting that tariquidar raised it 2.9-fold - but gives no
baseline value, and although Unpaywall marks the article open access, the
Oxford Academic PDF endpoint returns 403 to every automated request tried
(direct, browser-headered, and via a text-extraction proxy), so the value was
never seen and no reachable paper restates it. Kalvass et al. (JPET
2007;323:346-55) reports unbound brain concentrations for fentanyl and six
other opioids but in *mice* and behind a paywall - wrong species and
unreadable. Björkman et al. (Anesthesiology 1990;72:865-73) measured
steady-state tissue/**blood** partition coefficients for fentanyl in 13 rat
tissues including brain, which would be a better-controlled Kp than a figure
read, and is likewise 403-blocked at the publisher. A rat brain value of 4
appears in Table 1 of Front Pharmacol 2025 (PMC12575352) under the heading
"Kp of fentanyl in rats from report", with 13 tissue columns matching
Björkman's 13 tissues - but that table carries no citation in its text, so
the attribution is an inference and the number is not used. It leaves an
unresolved tension on the record: 4 (steady-state infusion, tissue:blood)
against ~16 (1 h post-SC, tissue:plasma), a 4-fold disagreement only a
readable primary can settle.

**Rat fu,brain and fu,plasma for haloperidol.** Loryan et al. (Mol Psychiatry
2016;21:1527-36) is exactly the right study - regional rat Kp,uu,brain *and*
regional rat fu,brain for haloperidol and four other antipsychotics in awake
Sprague-Dawley rats - and is closed access with no repository copy Unpaywall
or DiVA would produce; its abstract reports only qualitative regional
findings and fold-differences. [Gustafsson2019], which *is* open and supplies
the diazepam fu,brain used here, cites that paper for exactly these
haloperidol numbers without reproducing them. Kapetanovic et al. (JPET
1982;221:434-8) measured rat plasma and regional brain haloperidol by GC after
0.5 mg/kg IP and reports brain/plasma ratios, but only their variation with
age is in the abstract. Tsuneizumi et al. (Biol Psychiatry 1992;32:817-24)
gives rat brain:serum ratios for nine neuroleptics by HPLC-EC - which would
be both chemically specific and a good check on [Sunderland1987]'s
radioreceptor assay - but its abstract reports only the across-drug range
0.18-62.5.

**What was deliberately not done about the haloperidol gap.** The mouse
fu,brain of 0.007 is in hand. Applying it to the rat Kp of 22 would give a
plausible-looking rat Kp,uu of about 1.8, and it is not done: it would be a
cross-species substitution in the one factor that carries all the difference
between the two coefficients, which is rule 2. Worth noting that this
project's own data gives some support to the species-independence claim that
would license it - rat diazepam fu,brain 0.040 [Gustafsson2019] against mouse
0.050 [Maurer2005], 25% apart - and that this is exactly the argument that
should not be used to move a number by a factor of 140.

**Human, every drug.** None found, which is why the pre-P1.5 calibrated
scenarios were left alone. `mu_opioid_pk_models_calibrated`,
`gaba_a_pk_models_calibrated` and `d2_pk_models_calibrated` run on *human*
clinical PK (the fentanyl citrate label's 13 min redistribution half-life,
naloxone's ~64 min serum half-life, [Mandelli1978]'s diazepam distribution
phase, [Kudo1999]'s haloperidol terminal half-life) and dose in an achieved
plasma concentration rather than mg/kg. Attaching a rat partition to human
kinetics is the substitution rule 2 forbids, and human brain:plasma data for
these drugs is postmortem forensic material, where postmortem redistribution
makes a brain:blood ratio a property of the interval since death as much as
of the drug.

### Existing scenario results: nothing moved, and why

Every pre-P1.5 result is unchanged, bit for bit, and this is structural
rather than fortunate. A PK model with no partition has
`receptor_concentration_factor` exactly `1.0` (a float literal, not a
computed 0.9999...), so `C_receptor = C * 1.0` is the identical array; the
engine's occupancy path is otherwise untouched. Every scenario in
`scenarios.py` - illustrative and calibrated alike - is built on
`OneCompartmentPK` with no partition, for the human-PK reason above, so all
of them take that path. `test_the_human_gap_is_recorded_and_the_legacy_
scenarios_stay_plasma_driven` asserts it directly on all three calibrated
model sets rather than leaving it to the other tests to notice, and the
pre-existing suite (154 gate tests plus 7 `slow`) passes unchanged.

The rat dose-driven models in `drug_pk.py` are the ones that changed, and
only diazepam's changed behaviourally: `diazepam_rat_iv_pk`,
`diazepam_rat_ip_pk` and `diazepam_rat_pk` now attach the rat partition by
default and drive occupancy with unbound brain concentration at 0.18× plasma.
Their *plasma* output is unchanged - same `initial_state`, same
`concentration_rhs`, same terminal half-life, same AUC
(`test_the_partition_does_not_touch_the_plasma_curve`) - because P1.5 adds a
layer rather than perturbing P1.1-P1.4. The haloperidol and fentanyl rat
builders accept the same argument and resolve it to "no partition", for the
sourcing reason above.

The default is the attaching one (`brain_partition=USE_SOURCED_BRAIN_
PARTITION`, a sentinel distinct from `None`, which means "plasma-driven, and
I know it"). A default of `None` would have silently reproduced the bug this
task exists to fix, every time Phase 2 built a model without thinking about
it.

### The answer to P1.4's question

P1.4 ended by asking whether the benzodiazepine site sees 8× Ki or 40× Ki at
1 mg/kg diazepam IP in a 250 g rat. It sees **7.3× Ki**: 145.1 nM peak total
plasma × 0.18 = 26.1 nM unbound brain, against the 3.6 nM [Mohler1977] Kd.

The near-coincidence with the unbound-*plasma* estimate P1.4 offered as the
lower bound (28.4 nM, 7.9× Ki) is not luck and is worth stating, because it
is the single most useful thing Kp,uu tells you about this drug: Kp,uu = 0.918
means diazepam sits at passive equilibrium across the rat blood-brain
barrier, so unbound brain ≈ unbound plasma, and the 4.5-fold total-brain
enrichment is entirely brain tissue binding that the receptor never sees. The
40× figure was the wrong answer, and it was the one the model would have used
before this task.

### What this does not do

- It removes no illustrative label, and adds no PK parameter. Every
  circuit-layer parameter and every binding constant is where Track A/E left
  it.
- It does not give fentanyl or haloperidol a brain-driven receptor. Both have
  a cited rat Kp and neither has a rat fu,brain, so both remain plasma-driven
  and say so. That is P1.5's honest coverage, not a placeholder: the two
  searched-and-empty records above name the exact papers that would close each
  gap if they became readable.
- It does not model brain uptake kinetics, regional partition, active
  transport, or a metabolite's own partition. Nordiazepam's rat brain:plasma
  ratio of 3.5 ± 0.2 is in [Friedman1986] alongside the parent's 4.5 and is
  *not* recorded as a partition here, because P1.4 already established that
  nordiazepam has no ligand in this model to attach it to; that remains the
  metabolite limitation P1.4 recorded, now with the note that its brain side
  is also known and also unusable until the metabolite is modeled.
- It does not validate any simulated brain concentration against a published
  one. The checks here are consistency checks (derivation identities,
  independent-source agreement, figure-read cross-checks); comparing a
  simulated concentration-time curve to a measured one is still P1.6.

---

## Phase 1: validating the simulated curves against published PK (P1.6)

### The problem this task had to solve before it could do anything

P1.2-P1.5 built five PK models out of published *parameter sets*: a
clearance, some volumes, an absorption constant. The obvious way to "validate
simulated concentration-time curves against published PK curves" is to run
each model, compute Cmax/Tmax/half-life/AUC off the trajectory, and compare
them to the paper the parameters came from. That comparison cannot fail for
any reason connected to a rat. AUC would come back as Dose/CL because
Dose/CL is what was typed in; the terminal half-life would come back as
ln2/beta for the same reason. Reporting agreement of that kind as validation
against the literature is the quantitative equivalent of citing yourself, and
it is exactly the trap this task was most likely to fall into, because it
produces a table of green ticks.

So `pk_validation.py` splits every comparison into three tiers and refuses to
let them be read as one number:

| tier | target is | what a pass means | n |
|---|---|---|---|
| `ROUND_TRIP` | the model's own closed-form value | the integrator, the NCA and the unit arithmetic work | 16 |
| `SOURCE_CONSISTENCY` | a number the parameterising source *printed* but that was not an input | the source agrees with itself at the precision it printed | 3 |
| `INDEPENDENT` | a study that contributed no parameter to any model | the model is right about the rat | 15 |

Only the third tier can say anything, and 9 of its 15 comparisons say the
model is wrong.

### Finding independent data was most of the work

`drug_pk.py`'s existing searched-and-empty records were written looking for
the things needed to *build* a model: an F and a ka. P1.6 needs something
different and weaker - any second measurement of the same drug in the same
species - so the searches were redone from that angle, and two studies that
were useless for P1.2-P1.4 turned out to be exactly what P1.6 needed.

- **[Lei2018]** Lei K, He GF, Zhang CL, et al., *Exp Anim* 2018;67(2):163-73,
  doi 10.1538/expanim.17-0091, PMID 29225304, open access (PMC5955748, read
  in full). A herb-drug interaction study whose control arm is a clean rat
  haloperidol PK experiment: male Sprague-Dawley 180-200 g, n = 7, 0.7 mg/kg
  i.v., HPLC-MS/MS on plasma, Table 2 - AUC(0-inf) 188.4 ± 20.5 ng·h/mL,
  t½ 2.9 ± 0.7 h, CL 3.8 ± 0.4 L/h/kg. It contributes nothing to any model
  here, and it is the only independent rat haloperidol concentration-time
  dataset the project has ever found.
- **[Rautiola2019]** Rautiola D, Maglalang PD, Cheryala N, et al.,
  *J Pharmacol Exp Ther* 2019;370(3):796-805, doi 10.1124/jpet.118.255943,
  PMID 30837282, open access (PMC6806351, read in full). An intranasal
  diazepam-prodrug study whose IV reference arm is a clean rat diazepam PK
  experiment: male Sprague-Dawley 0.271 ± 0.023 kg, 1.00 mg/kg IV bolus,
  naive-pooled NCA, Table 2 - C0 (extrapolated) 441 ng/mL, t½,z 32.9 min,
  AUC(0-15 min) 1910 ± 70 and AUC(0-90 min) 5430 ± 190 min·ng/mL (± S.E.).

Both were checked for internal consistency before use rather than trusted.
[Lei2018]'s IV column hangs together (Dose/AUC = 3.715 L/h/kg against a
printed CL of 3.8, a 2.2% mean-of-per-animal-ratios gap); its *oral* column
does not (the printed CL/F of 17.1 L/h/kg implies an oral AUC of 82 ng·h/mL
against the 43.3 it prints, a 1.9-fold contradiction), so nothing from the
oral column is used as a target and that is recorded in the module.

P1.3 named one specific follow-up for a later pass - Johnson et al.,
*Pharm Res* 2014;31:2605-17 (doi 10.1007/s11095-014-1358-7, PMID 24792824),
which fits rat haloperidol plasma PK from IV/IP/SC data and would close the
missing F and ka for the routes the behavioural literature uses. It was
retried here. Unpaywall now reports it as open access with a repository copy
at research.rug.nl, and the record page does list a PDF, but the file
endpoint returns 403 to a browser-identified request; the parameter tables
were still not readable. So P1.3's blocker stands exactly as written, and the
one thing this pass adds to it is that the obstacle is a repository
misconfiguration rather than a missing deposit, which is worth knowing for
whoever tries next.

For fentanyl the search came up empty and the record says so
(`SEARCHED_AND_EMPTY_VALIDATION[("fentanyl","rat","IV")]`): Ohtsuka et al.
2007 dosed rats 0.03 mg/kg IV, squarely inside [Yassen2008]'s own dose range,
and is closed access with an abstract that prints no parameter; Yu et al.
2018 *Toxicol Sci* has full rat IV plasma and brain profiles at 40 and
80 µg/kg and the publisher returns 403 to both the article and the PDF with
no PMC deposit; and a 2025 QSAR-PBPK paper prints an "observed" rat fentanyl
row (CLsys 0.493 L/h, i.e. 8.2 mL/min against [Yassen2008]'s 17.0 - a
disagreement that would have been the most interesting number in this
section) but attributes it only to "experimentally measured data reported in
the literature" and names no source, which is the same objection that kept
the Groningen haloperidol tables out of P1.3. Fentanyl is therefore validated
cross-route only, and that limitation is the reason its two failures are the
hardest to interpret.

### Criteria, and why none of them is a dial

Each comparison declares a `criterion_kind` chosen by *what the target number
is*, before looking at how close the model got:

- `NUMERICAL` (round trip): |rel err| ≤ 0.5%, which is ~25× the largest error
  the simulation grid actually produces.
- `DISPERSION`: inside observed mean ± 2 × the dispersion the source prints.
  SEM is inflated to SD by √n where n is stated; where the design pools
  different n at different time points ([Rautiola2019]'s leapfrog sampling)
  the SEM is used unconverted, which makes the interval about 2× too narrow -
  i.e. it makes the model's job *harder*, and `dispersion_is_understated`
  flags every record where that happens.
- `TWO_FOLD`: 0.5 ≤ predicted/observed ≤ 2.0, the conventional
  predictive-performance criterion of the PBPK literature, used where the
  source prints no dispersion at all.
- `INTERVAL`: inside the interval the source's own printed rounding leaves
  open ("1.5 h" → 87-93 min). This is the same device P1.3 used for the
  haloperidol half-life instead of inventing a percentage.
- `UPPER_BOUND`: a structural impossibility test rather than a fit - an
  implied bioavailability must be ≤ 1.

Both the dispersion and the two-fold verdict are computed for *every* record
regardless of which is primary, so a comparison that passes one and fails the
other says so, and several do.

The guarantee that no tolerance was widened to reach green is structural
rather than a promise: `tests/test_pk_validation.py` asserts the **relative
error itself** for every independent comparison to five significant figures,
and asserts the failing set by key. Widening a criterion does not make any of
those assertions pass; it makes `test_the_failing_comparisons_are_exactly_
these` fail. The only way to move a comparison out of the failing set is to
change the model or the target, and either way a test says so out loud.

### What the independent tier actually found

#### Haloperidol vs [Lei2018]: 25% low on exposure, 47% low on half-life

| quantity | model | [Lei2018] | error | verdict |
|---|---|---|---|---|
| AUC(0-inf) at 0.7 mg/kg | 140.6 ng·h/mL | 188.4 ± 20.5 | **−25.4%** (z = −2.33) | FAIL |
| clearance | 4.98 L/h/kg | 3.8 ± 0.4 | **+31.1%** (z = +2.95) | FAIL |
| terminal half-life | 92.7 min | 174 ± 42 | **−46.7%** (z = −1.94) | pass |

The AUC and clearance rows are one finding measured twice, not two
corroborating hits, and the module says so. The half-life row is worse than
its verdict: it is a 1.88-fold under-prediction that clears the dispersion
criterion at z = -1.94 only because [Lei2018]'s SD is 24% of its own mean,
and scrapes inside the two-fold criterion at 0.53x. Calling that agreement
would be overclaiming, and the test that pins it is named accordingly.

The clearance gap has a specific candidate explanation that is deliberately
*not* applied. [ChengPaalzow1992] reports **blood** clearance and blood-based
volumes; [Lei2018] measured **plasma**. CL_blood = CL_plasma / Rb, so a rat
haloperidol blood:plasma ratio of 0.76 would reconcile the two exactly. That
is a physically ordinary value, and no rat Rb for haloperidol exists - the
gap was already recorded in
`drug_pk.SEARCHED_AND_EMPTY[("haloperidol","rat","blood:plasma")]` during
P1.3. So the discrepancy is reported at full size with the candidate named,
rather than dissolved by a conversion factor nobody measured. That is the
rule-1 discipline applied to an *error* rather than to a parameter: you may
not invent a number to explain a miss away any more than to make a fit
succeed.

The half-life gap is not covered by that escape route: a terminal half-life
is invariant to a time-independent blood:plasma ratio. The most likely cause
is sampling design - [Lei2018] sampled to 24 h with HPLC-MS/MS, and the 1992
paper's window and assay sensitivity are unknown because it is closed
access - but that is a hypothesis, and the reported error is −46.7%.

#### Diazepam vs [Rautiola2019]: the peak is 3.4× low, and now we know by how much

| quantity | model | [Rautiola2019] | error | verdict |
|---|---|---|---|---|
| C0 (the IV bolus Cmax) | 128.7 ng/mL | 441 | **−70.8%**, i.e. 3.43× low | FAIL |
| AUC(0-15 min) | 1786 ng·min/mL | 1910 ± 70 | −6.5% | pass |
| AUC(0-90 min) | 7493 ng·min/mL | 5430 ± 190 | **+38.0%** (z = +10.9) | FAIL |
| terminal half-life | 66.0 min | 32.9 | **+100.6%**, 2.006× | FAIL |

This is the most informative block in the task, because P1.4 predicted the
first row and could not size it. `diazepam_rat_iv_pk`'s docstring says the
model starts at Dose/V_beta where the real drug starts at Dose/Vc, that
Vc < V_beta always, and that "here it cannot be computed at all, only bounded
below by 1" because [Klotz1976] publishes no central volume. [Rautiola2019]'s
extrapolated C0 sizes it: Dose/C0 = 2.27 L/kg is an empirical central volume
against the model's V_beta of 7.77 L/kg, a factor of 3.4. That number is an
observation from a different study and does **not** become a parameter -
no `DIAZEPAM_RAT_IV_VC_L_PER_KG` was added, and a test asserts its absence -
but the open question P1.4 left is now answered quantitatively.

The two AUC rows disagree in *sign*, and that is the shape of the error
rather than noise: the missing distribution phase makes the model start 3.4×
too low, the too-long half-life makes it fall too slowly, and over the first
15 minutes those two errors nearly cancel (−6.5%). By 90 minutes the second
one dominates (+38.0%). A single AUC number would have hidden this
completely.

Both AUC comparisons are made over the source's **own** windows rather than
extrapolated to infinity, because that study stopped sampling at 90 min and
reported no AUC(0-inf). The model's AUC(0-inf) is 2.26× the observed
90-minute AUC; quoting that as a +126% error would have been manufacturing a
discrepancy out of the study design, and a test pins the 2.26 factor so the
choice stays visible.

The half-life row fails the two-fold criterion by 0.3%, which is close enough
that the verdict alone is misleading and the fold error (2.006) is what gets
reported. It is also the one comparison where the *observed* value is the
more suspect of the two: [Rautiola2019] sampled to 90 min, under three of its
own half-lives, so its λz may still contain distribution. [Klotz1976]'s 1.1 h
is the other candidate and its sampling window is not readable, because that
paper is closed access and its abstract states only that disposition was
biexponential in all four species it studied. Neither value can be declared
correct from here. The model is pinned to [Klotz1976], so the disagreement is
reported at full size.

#### Fentanyl: no same-route check exists, and both cross-route checks fail

With no independent rat IV dataset, the [Yassen2008] disposition can only be
held against the two extravascular rat studies `drug_pk.py` records but does
not model. Two kinds of check are possible, and the structural one is the
stronger:

For any route CL/F ≥ CL, because F ≤ 1. So CL(IV) ÷ (CL/F)(extravascular) is
an implied bioavailability. A value above 1 is an impossibility: the two
studies cannot both describe rat fentanyl.

| comparison | implied F | verdict |
|---|---|---|
| [Yassen2008] IV ÷ [CanfieldSprague2025] SC | **1.530** | FAIL |
| [Yassen2008] IV ÷ [Almousa2011] IP | **1.135** (a *lower* bound) | FAIL |
| [Klotz1976] IV ÷ [Friedman1986] IP (diazepam) | 0.320 | pass |
| [Klotz1976] IV ÷ [Hua2025] PO (diazepam) | 0.082 | pass |
| [ChengPaalzow1992] IV ÷ [Lei2018] PO (haloperidol) | 0.291 | pass |

The last three rows are the control group, and they are why the first two are
evidence about fentanyl rather than about the test: the same arithmetic on
cross-study pairs assembled the same way gives ordinary bioavailabilities for
the other two drugs. (The haloperidol row is also a check on the
cross-study substitution itself: [Lei2018]'s own within-study answer is
23.0 ± 5.7%, and swapping in the 1992 blood clearance moves it to 29% - the
substitution costs about a quarter of the answer and does not change its sign
or order of magnitude.)

The Almousa row is the weaker of the two failures and is labelled as such: it
is a lower bound, because that paper's clearance is Dose/AUC(0-180) on a
truncated AUC so the true CL/F is smaller and the implied F larger still, and
that source is independently known to contradict itself two-fold
(`fentanyl_rat_ip_auc_consistency`, from P1.2). It corroborates the direction,
not the magnitude.

Half-life and exposure tell the same story more mildly:

| quantity | model | observed | error | verdict |
|---|---|---|---|---|
| terminal half-life vs SC | 53.6 min | 66.0 ± 0.6 (SEM) | −18.8% | FAIL (dispersion), pass (two-fold) |
| terminal half-life vs IP | 53.6 min | 198.3 ± 44.7 | −73.0% | FAIL both |
| AUC(0-inf) at 0.3 mg/kg vs SC | 69.9 ng·h/mL | 112.8 ± 10.8 (SEM) | −38.1% | pass both |

The AUC row is worth reading carefully, because it is the one place where a
"pass" is the misleading answer. The model has no SC route, so what it can
offer for 0.3 mg/kg SC is the F = 1 value - an **upper bound** on any SC
prediction it could make. That upper bound lands 38% *below* the observed
mean. It clears the dispersion criterion because [CanfieldSprague2025]'s
between-animal spread is wide (SEM 10.82 on n = 6, so an SD of 26.5 on a mean
of 112.8), and it clears the two-fold criterion at 0.62x. The same
disagreement reads as a pass one way round and as an impossibility the other
way round, which is precisely why the structural row is kept alongside it
instead of being replaced by it.

What would explain a fentanyl clearance that is too high by half: the SC
study dosed 0.300 mg/kg against [Yassen2008]'s 0.015-0.085 mg/kg, a
3.5-20x gap, so saturable elimination at the higher dose is the leading
candidate.
The studies also differ in strain (Wistar vs Sprague-Dawley), weight (237 vs
327 g) and administration (20-40 min infusion vs bolus). None of this is
resolvable without the same-route dataset the searched-and-empty record says
is behind a paywall.

### Tmax: not validated, for a structural reason

Tmax is one of the four quantities P1.6 was asked to validate and it is the
one that could not be validated at all. This is recorded as data
(`pk_validation.TMAX_IS_NOT_VALIDATABLE`) with a test asserting it is still
on the record, because a silently missing quantity looks like an oversight.

Four of the five models are IV bolus models, whose Tmax is 0 by construction:
there is no absorption phase to get right. The fifth, `diazepam_rat_ip_pk`, is
the only absorption model in the project; it predicts Tmax = 17.1 min, and no
rat intraperitoneal diazepam study in reach reports an observed Tmax.
[Friedman1986], whose disposition the composite uses, is closed access with
no Tmax in its abstract and a first post-dose sample at 5 min. The search was
extended beyond P1.4's sweep specifically for a rat IP diazepam Tmax; the
nearest hit, Nelson et al. 2024 (PMC11296087, read in full), does dose male
Sprague-Dawley rats diazepam IP at 0.75-6 mg/kg with LC-MS plasma sampling,
but it samples 10 min after the *last* of three hourly injections and reports
no non-compartmental parameters, so there is no single-dose ascending limb in
it to read a Tmax from.

Observed rat Tmax values do exist in this project's records - fentanyl IP
28.3 min, fentanyl SC 0.5 h (and that one is the first sampling time, not a
resolved peak), diazepam PO 1.33 h - but every one of them belongs to a route
with no model, and each of those routes is unmodelled precisely because its
ka was never measured. ka is the parameter Tmax would test. **Validating Tmax
and building the missing absorption models are the same missing piece, not
two.** That is a sharper statement of the gap than P1.2-P1.4 had, and it is
the most useful thing this task produced for Phase 2 planning.

### The source-consistency tier, and its one failure

Three comparisons hold a simulated curve against a number the parameterising
source printed but did not contribute:

- **haloperidol, 92.7 min against a printed "1.5 h"** (+3.0%): passes,
  because it sits inside the 87-93 min that two significant figures leave
  open. P1.3 chose to anchor on the printed V_beta (three significant
  figures) rather than the printed half-life (two); this is the check that
  the choice did not push the terminal phase outside what the paper allows.
- **fentanyl, point estimates vs the source's own 1,000-replicate bootstrap**
  (+2.1%): passes, and both sit inside the 44.3-63.6 min the source's
  confidence intervals leave open. [Yassen2008] publishes no half-life at
  all, so this is the only within-paper check available for that drug.
- **diazepam IP, 52.46 min against a printed "0.88 h"** (−0.64%): **fails**
  its interval by 0.07% at the low edge. This one is the source's arithmetic
  and not the model's, and the module shows why: ln2/((CL/F)/(V/F)) from the
  printed 255 and 19.3 gives 52.23-52.71 min across those two numbers' own
  rounding, and the printed 52.8 min lies just outside *that*. The source's
  three printed values are mutually inconsistent at the fourth significant
  figure, the model reproduces two of them exactly, and it therefore cannot
  reproduce the third. `diazepam_rat_ip_reported_half_life_consistency` from
  P1.4 already pinned the same −0.64%; P1.6 keeps the failure visible instead
  of reclassifying it.

### What this task changes, and what it does not

- **No parameter changed and no illustrative label moved.** Not one number in
  `drug_pk.py`, `pk.py` or `scenarios.py` was touched. A validation task that
  edits the thing it is validating is not a validation task.
- **No tolerance was widened.** 10 of 34 comparisons fail. The failing set is
  asserted by key, and every independent comparison's relative error is
  asserted to five significant figures.
- **The models are still the best available and are still wrong by these
  amounts.** Nothing here says to stop using them; it says what Phase 2 is
  buying when it spends the PK bridge. The concrete carry-forwards: rat
  haloperidol exposure is over-cleared by ~31% and its terminal phase is
  ~1.9× too fast; rat diazepam's IV bolus peak is ~3.4× too low for the first
  minutes and its exposure over an hour and a half is ~38% too high; rat
  fentanyl's clearance is high enough to be structurally incompatible with
  both published extravascular datasets, by at least 13% and probably 53%.
- **Two named papers would close the biggest gap.**
  `SEARCHED_AND_EMPTY_VALIDATION[("fentanyl","rat","IV")]` names Ohtsuka et
  al. 2007 and Yu et al. 2018, either of which is a rat IV fentanyl
  concentration-time dataset and either of which would turn the drug's two
  cross-route failures into a same-route answer. Both are paywalled rather
  than nonexistent, which is a more tractable problem than the ones Track E
  hit. The Tmax gap needs something else again: a rat IP diazepam study with
  a resolved ascending limb, which `TMAX_IS_NOT_VALIDATABLE` records as
  searched and not found.
- **It validates plasma/blood curves only.** The brain concentrations P1.5
  introduced are the plasma curve times a scalar, so nothing here tests the
  partition itself; validating a simulated *brain* concentration against a
  measured one needs brain time-course data and is not attempted.

## Phase 2: dose-to-behaviour evidence module (P2.1)

### The question this task actually had to answer

Track E, above, ends with a specific diagnosis: circuit-layer calibration is
blocked because real behavioural pharmacology reports an administered **dose**
while this codebase's state variable was a **concentration**, and no honest
route existed between them. Phase 1 built that route. P2.1 is the first task
to spend it, and its brief names the three papers Track E named: Yoshida et
al. 1999 (fentanyl → nucleus accumbens dopamine), Gao et al. 2021 (haloperidol
locomotor suppression), Pellow et al. 1985 (diazepam → elevated plus maze).

So the question is narrow and answerable: **with the bridge in place, are
those three usable now?**

**They are not, and none of the three fails for the reason Track E predicted.**
That is the finding, and `src/neurocascade/calibration/behavioral_evidence.py`
is the record of it. The blocker moved, per system, to a different and more
specific place:

| paper | bridge covers its drug/species/route? | effect size obtainable? | what actually blocks it now |
|---|---|---|---|
| Yoshida 1999 (fentanyl, rat, IV) | **yes**, via `drug_pk.fentanyl_rat_iv_pk`, [Yassen2008] | no | the 1999 article is closed access; its abstract carries neither the doses nor the percentages |
| Gao 2021 (haloperidol, mouse, PO) | no: no mouse haloperidol PK at all, no haloperidol PO in any species | yes (ED50 0.11 mg/kg) | route/species coverage, *and* Track E's untouched second objection that an ED50 is not a likelihood target |
| Pellow 1985 (diazepam, rat, route unstated) | unanswerable, because the source gives neither dose nor route | no | dataset access; there is nothing for a bridge to convert |

Two of those are worth stating plainly because they change what the next task
should do. For **fentanyl**, Phase 1 did its job completely: the moment a dose
and a percent-of-baseline figure are obtained from that paper,
`BehavioralEvidenceEntry.brain_exposure` runs on it unchanged. "Get a 1999
closed-access PDF" is a materially cheaper problem than "build a
pharmacokinetic submodel this project has for no drug". For **haloperidol**,
the gap is structural in a way no single paper closes: the routes the rat PK
literature covers (IV only, [ChengPaalzow1992]) and the routes the behavioural
literature uses (IP, SC, PO, essentially without exception in every paper
screened) do not intersect for this drug. That is recorded in
`SEARCHED_AND_EMPTY_BEHAVIORAL[("haloperidol","rat","IV","locomotor
activity")]` together with the two things that would close it.

### A correction to Track E, made rather than inherited

Track E states that Yoshida et al.'s abstract "confirms fentanyl
dose-dependently increased NAc dopamine at 5 and 10 ug/kg IV". Re-checked in
this pass against PubMed EFetch, Semantic Scholar, OpenAlex and the publisher
page, that could not be reproduced: the abstract PubMed carries states every
dose as a bare unit, "(microg/kg)" or "(nmol)", and contains no dose value at
all. The two numbers may well be right; nothing this project can reach
confirms them.

They are therefore **not stored as a dose**. The entry's
`provenance_gaps["doses_mg_per_kg"]` holds the claim, the list of sources
checked, and the reason each did not confirm it, and
`test_track_e_unverified_fentanyl_dose_claim_is_not_stored_as_a_dose` asserts
that the entry's dose stays `None`. Rule 1 of this codebase, never invent a
number, applies to numbers copied from our own prose exactly as it applies to
numbers copied from a paper, and this is the first time that has actually
bitten. What the PubMed record *does* establish is used: species and strain
(MeSH "Rats, Sprague-Dawley"), sex (MeSH "Male"), route (MeSH "Injections,
Intravenous") and the endpoint.

### Why there is a fourth entry, and what it is for

A module of three absences would be an accurate record and a useless one: it
would not demonstrate that the bridge works on real behavioural dosing, which
is the acceptance criterion this task was actually given. So the module also
carries **Stickling & Rosenkranz 2024** (Behav Neurosci 139(2):60-73,
doi:10.1037/bne0000612, PMC11966652, read in full): male Sprague Dawley rats
(Envigo), PND 90-95, a single **1.5 mg/kg IP** diazepam injection, elevated
plus maze **60 min** later for a 5 min trial at 25-30 lux, n = 12-13 per
group, with means and SEMs printed in its Table 1.

Its role is to exercise the bridge, and it stands on its own sourcing: the same
drug, the same species and the same assay as Pellow, found independently. It is
in the module because `drug_pk` has a rat IP diazepam model and diazepam/rat is the
one drug/species pair in this codebase with a *receptor-drivable* brain
partition (Kp,uu 0.918 × fu,plasma 0.196, P1.5). So it converts all the way:

    1.5 mg/kg IP  →  Cmax 218 nM plasma (Tmax ≈ 17 min)
                  →  135 nM plasma at the source's own 60 min timepoint
                  →  **24.3 nM unbound brain**

That last number is the quantity a `ReceptorSite` may legitimately bind, and
it lands in a pharmacologically sensible range against diazepam's
benzodiazepine-site affinity. It is the first time in this project that a
published *behavioural* dose has been carried through to a receptor-driving
concentration.

Two caveats travel with it, on the entry rather than in a footnote, because
they are why it is recorded and not yet fitted:

1. The paper's introduction says "diazepam (1 mg/kg)" while its Methods and
   Figure 1 legend both say 1.5 mg/kg. The Methods value is used; the
   discrepancy is stored.
2. The only EPM measure it prints numerically is "closed arm preference"
   (saline 77.5 ± 12.1 s, diazepam 76.7 ± 24.4 s), and the paper never defines
   that measure. Its open-arm times, which would be the usable quantity, are
   in Figure 4A only. A target whose operational definition is unknown cannot
   be compared with a model readout, so `calibration_ready_entries()` returns
   empty and `test_no_entry_is_calibration_ready_yet_and_that_is_recorded`
   pins that.

### Design decisions worth recording

**A missing field is `None` *and* is listed in `provenance_gaps`, enforced by
the constructor.** `BehavioralEvidenceEntry.__init__` raises if any of
`strain`, `sex`, `body_weight_kg`, `doses_mg_per_kg`, `timepoint_min` or
`route` is absent without a recorded reason, and raises again if a gap is
recorded for a field that does have a value. This is the Track E discipline
turned into a type invariant: "not reported" becomes a fact with a citation
context attached rather than a silent hole. `test_constructor_refuses_an_
unexplained_gap` proves the enforcement bites.

**Bridge coverage is computed, never asserted.** `pk_route_status()` calls the
`drug_pk` builder and reports one of `available` / `no-dosing-recorded` /
`unsourced` / `no-species`, so the verdicts in this module cannot go stale: if
a later phase sources haloperidol IP kinetics, the Gao entry's status changes
by itself and `test_every_entry_has_a_bridge_verdict_and_a_matching_status`
fails until the prose is updated to match. The alternative, a hand-written
"unusable" flag, is exactly the kind of claim that rots.

**`brain_exposure` integrates rather than solving in closed form.** It goes
through `pk_validation.simulate_curve`, i.e. the same `solve_ivp` path
`engine.Simulation` runs, so the concentration it reports is what a simulation
would actually see, integrator error included. It also refuses to invent a
timepoint or a dose: an entry that has neither raises rather than defaulting.

**No `sd` is exposed anywhere in this module.** `evidence.py` has
`OCCUPANCY_LIKELIHOOD_SD`, an explicitly-flagged engineering allowance, and it
is defensible there because a target existed and only its uncertainty had to
be supplied. Here no computable target exists at all, so a likelihood width
would be an allowance for nothing. P2.2 is where that question belongs.

**Body weight is not defaulted, and where it does not matter, that is proved
rather than claimed.** The Stickling entry records no body weight (the paper
reports PND, not grams). For that entry it is harmless, because the rat IP
diazepam parameter set is per-kg, so weight cancels, and
`test_bridged_concentration_is_weight_invariant_for_the_per_kg_model` asserts
it instead of the docstring asserting it. It would *not* be harmless for
fentanyl, whose [Yassen2008] parameters are absolute mL and mL/min;
`PK_SOURCE_BODY_WEIGHT_BANDS_KG` records the 225-250 g band those estimates
came from as the defensible choice for a caller who must pick one.

**Rule 2 is enforced structurally.** Track E found real, usable numbers for
morphine (NAc dopamine) and chlordiazepoxide (EPM open-arm time) and refused
to use them for fentanyl and diazepam. `test_no_cross_drug_substitution_is_
present` scans every entry's endpoint, measure and group labels for the names
of the obvious substitutes, so that refusal cannot be quietly undone later.

### What was searched and not found

`SEARCHED_AND_EMPTY_BEHAVIORAL` carries three records, one per system, each
naming what was searched, which candidates were found, why each is unusable,
and what would close it. The diazepam one is the largest sweep this project
has run: all 600 open-access Europe PMC articles matching (diazepam AND
"elevated plus" AND rats) were fetched in full and screened programmatically
for an article that (a) never mentions mice, (b) contains a diazepam mg/kg IP
dosing statement, and (c) contains a table with an open-arm measure carrying a
"±". **Zero of 600 matched.** The rat EPM literature reports open-arm data in
bar charts; the studies that tabulate it are overwhelmingly mouse studies, and
this codebase has no mouse diazepam PK
(`drug_pk.SEARCHED_AND_EMPTY[("diazepam","mouse","*")]`).

The fentanyl record names the one near-miss worth knowing about:
Sustkova-Fiserova et al., Int J Mol Sci 2017;18(11):2486,
doi:10.3390/ijms18112486 (PMC5713452) is the right
drug, species and region and prints real percent-of-baseline numbers, but its
systemic fentanyl is 30 µg/kg **subcutaneous**, a route
`drug_pk.SEARCHED_AND_EMPTY[("fentanyl","rat","SC")]` already records as
having no obtainable `ka`, and its analytes are anandamide, 2-AG and GABA
rather than dopamine.

### What this task changes, and what it does not

- **No parameter changed and no illustrative label moved.** Every circuit-layer
  gain, weight, tau and baseline in `scenarios.py` is exactly as illustrative
  as it was before P2.1, for the reasons above.
  `tests/test_bayesian_calibration.py::test_no_circuit_layer_parameter_is_fit`
  continues to guard that by construction.
- **Nothing here is a fit.** `calibration_ready_entries()` returns empty, and a
  test pins it. P2.2 inherits a module with one entry that reaches a receptor-
  driving concentration and no entry that yet has a comparable model readout;
  the honest next moves are named per entry rather than assumed.
- **The Phase 1 bridge is validated as a behavioural-dosing tool, once.** One
  real published behavioural dose now goes mg/kg → plasma → unbound brain nM
  through the production code path. That is one, not three, and the module
  says which two are missing and why.

## Phase 2: Bayesian fit of the coupling gains (P2.2)

### The task asked for three fits; one is honest, and this section is mostly about why

P2.2's brief is to fit `mu_opioid_coupling`, `d2_coupling` and
`gaba_a_connection_modulation` against the P2.1 evidence, as three
independent per-system models. P2.1 handed this task a module in which
`calibration_ready_entries()` returned **empty** (no entry had a usable
effect size and a bridge and a defined measure at once), so the brief as
written could not be executed against the evidence as inherited. Two things
follow, and both were done:

1. The literature searches P2.1 ran were **re-run at the same scale**, per
   system, because "P2.1 found nothing" is not the same claim as "there is
   nothing." That produced one genuine find (below) and two confirmations of
   emptiness with sharper reasons than before.
2. Exactly one coupling gain is fitted:
   **`gaba_a_connection_modulation`**. `mu_opioid_coupling` and `d2_coupling`
   remain illustrative, and the illustrative label was not touched for either.

So the acceptance criterion "three independent models, all converged" is met
in the sense that this scaffold still runs and converges three independent
per-system models, and is **not** met in the sense the brief intended: only
one of the three names a circuit-layer parameter. That is stated here rather
than smoothed over, and the per-system reasons are in
`behavioral_evidence.SEARCHED_AND_EMPTY_BEHAVIORAL`.

### The find, and the correction to P2.1 it forces

**[Graziani2022]** Graziani D, et al. "Behavioral effects of a low molecular
weight peptide fraction from *Phaseolus vulgaris* in rats." Braz J Med Biol
Res. 2022;55:e12314. doi:10.1590/1414-431X2022e12314, PMID 36477952,
PMC9728632, open access, read in full. Male Wistar rats, 300–380 g,
drug-naive, no disease model. Diazepam 2 mg/kg, "All drugs and vehicle were
injected by the ip route", 5 min elevated plus maze. Table 2, "Data are
reported as means±SE":

| group | n | open-arm time (s) | open-arm entries | total entries |
|---|---|---|---|---|
| vehicle | 11 | 30.7 ± 6.4 | 1.8 ± 0.4 | 7.5 ± 0.9 |
| diazepam 2 mg/kg | 8 | 89.0 ± 13.0 | 6.4 ± 0.7 | 12.8 ± 0.6 |

P2.1's widest sweep, over "all 600 open-access Europe PMC articles matching
(diazepam AND 'elevated plus' AND rats)", screened for articles that never
mention mice, carry a diazepam mg/kg IP dosing statement, and tabulate an
open-arm measure with a "±", reported **"Zero of 600 matched."**
[Graziani2022] satisfies all three of those criteria and is in that result
set (the same query returns 724 records when re-run). The zero came from the
screen's table-detection step, not from the literature. The record in
`SEARCHED_AND_EMPTY_BEHAVIORAL` is marked SUPERSEDED **in place**, with the
original text kept underneath, because the interesting part is the failure
mode: an automated screen that returns zero should be checked against a
hand-verified positive before it is believed. The corrected finding is that
the rat EPM literature tabulates diazepam open-arm times *rarely*, one
article in 724, and not that it never does.

The table's units are asserted rather than assumed, which matters because
the source prints none on those columns: open + closed + centre time is
302.7 s (vehicle) and 299.9 s (diazepam) against the stated 5 min trial, and
open + closed entries reproduce the printed totals in both rows.
`test_graziani_table_is_internally_consistent_so_its_units_are_not_assumed`
checks both.

### The real blocker, which was neither Track E's nor P2.1's

Track E said the blocker was dose-versus-concentration. Phase 1 fixed that.
P2.1 said the blocker had moved to dataset access. For GABA-A that was true
and [Graziani2022] closes it, and then a third blocker appears, which no
earlier pass had to confront because no earlier pass had a dataset to
confront it with:

**this codebase's behavioural readouts have no scale.** `anxiety_index_readout()`
returns the CeA population's firing rate, a number in (0, 1). The elevated
plus maze returns seconds on the open arms. Nothing published converts one
into the other, and a conversion constant invented to make the fit work
would be exactly the invented number rule 1 forbids, and a *bigger* invention
than the dose→concentration submodel Track E refused to invent, because at
least that one would have been a claim about drug disposition rather than a
claim about what a firing rate means.

What P2.2 does instead is fit a **ratio**. Write the unknown behavioural
scale as `b = f(a)`, `a` the model's anxiety index. If `f` is approximately
linear across the range the drug moves the animal over, then

    b(drug) / b(vehicle)  ≈  a(drug) / a(vehicle)

and every unknown constant in `f` cancels. Both sides are then dimensionless
drug/vehicle ratios of a bounded, rises-with-anxiety quantity. On the data
side that quantity is **open-arm avoidance**, `1 − open-arm time / trial
duration`: the assay's own primary variable divided by the trial duration
the source states, and nothing else.

    vehicle  1 − 30.7/300 = 0.8977 ± 0.0213
    diazepam 1 − 89.0/300 = 0.7033 ± 0.0433
    ratio    0.7835 ± 0.0517

That ±0.0517 is the delta-method propagation of the paper's own two SEMs. It
is **the first likelihood width in this codebase that a source actually
supplies**: every earlier one, `evidence.OCCUPANCY_LIKELIHOOD_SD` included,
was a declared engineering allowance. The observed effect is 4.2 of its own
standard deviations away from "no effect", which is why the gain comes out
identified rather than prior-shaped.

**The proportionality assumption is the one thing in this fit no citation
backs, and it is not hidden.** It is stated in `circuit_forward.py`'s module
docstring, in `gaba_a_connection_modulation()`, and here. It cannot be
tested directly, but its consequence can be bounded, and is: the fit is
repeated against an independently defined composite EPM Anxiety Index
(`1 − (open-arm time/duration + open-arm entries/total entries)/2`, the
formula as printed by [BernalMorales2017], which attributes it to Cohen,
Matar & Zohar 2013, a chapter this project did not obtain, which is why
that index is used only as a check). The composite gives a ratio of 0.7259
against the primary 0.7835, and the gain those two imply differs by less
than the gain interval the primary target's own ±1 SEM already spans.
`test_the_fit_is_reported_under_a_second_behavioural_index_too` pins that.

What the ratio formulation deliberately does **not** do: it does not assume
the model's baseline anxiety index equals the vehicle animals' anxiety, so
it neither fits nor blesses any of the amygdala circuit's illustrative
`baseline_input`/`tau`/`gain`/connectivity values. Those still set how
*sensitively* the ratio depends on the fitted gain, which is a real
dependency and is why P2.3, not P2.2, is where they belong.

### What is actually fitted, and how

`calibration/circuit_forward.py` is the circuit-layer counterpart to
`forward.py`. Given a gain and a receptor activation it integrates
`CircuitNetwork.rhs`, the same function `engine.Simulation.rhs` calls, via
the same `solve_ivp`, to steady state, twice, once with the drug and once
without, and returns the readout ratio. Nothing about the circuit is
reimplemented for calibration, exactly as `forward.py` integrates the real
`ReceptorSite.occupancy_rhs` rather than substituting the closed form.

The chain, end to end, for one MCMC draw:

    2 mg/kg IP  --Phase 1 bridge (drug_pk + brain_partition, integrated)-->
    32.3 nM unbound brain diazepam at 60 min
                --ReceptorSite.occupancy_rhs at sampled Ki_diazepam-->
    activation 0.90
                --(1 + gain·activation) on the BLA<-interneuron weight,
                  CircuitNetwork.rhs to steady state, twice-->
    anxiety-index ratio
                --Gaussian likelihood, sd from [Graziani2022]'s SEMs-->
    posterior over the gain

Three approximations are made and each is checked rather than argued:

- **Activation is clamped** at its equilibrium value while the circuit
  relaxes, instead of co-integrating the decaying plasma concentration.
  Defensible because the site is near saturation across the whole plausible
  pre-test window while the circuit's slowest tau is 2 min.
  `test_clamped_activation_agrees_with_a_full_engine_simulation` builds the
  same scenario as a real `engine.Simulation` and requires the two ratios to
  agree within 2%.
- **The steady state is integrated for 30 τ at rtol 1e-8** rather than
  solved. Checked against a 10× longer, 1000× tighter integration and
  against the network's own rhs being zero there.
- **The drug-free leg is computed once** and passed in, because at zero
  activation every modulation factor is exactly `1 + gain·0`. A test
  requires passing it to change no answer, so it is a cache and not an
  approximation.

**The exposure timepoint is the one place the source is silent.**
[Graziani2022] states "In the acute trials, animals received ip injections
(100 µL) of PV3 doses 1 h before trials", its acute-trial interval, but the
sentence names the test peptide, not diazepam. The entry therefore stores
**no** timepoint and `brain_exposure` refuses to invent one;
`evidence.py` passes 60 min explicitly and says where it came from. This is
recorded as a gap and then shown not to matter: at 30/45/60 min the bridge
gives 47.5/39.4/32.3 nM unbound brain, all far above diazepam's 3.6 nM Kd,
so the implied gain moves by less than half the target's own ±1 SEM
interval (`test_the_fit_barely_moves_over_the_unstated_exposure_timepoint`).

### The prior, and the bound that was nearly active

`gaba_a_connection_modulation_gain` is uninformed, and more so than that
label usually means here: the other uninformed prior in `priors.py`
(`Ki_fentanyl_nM`) is uniform over 0.1×–10× a *measured* point estimate,
while no measurement of a circuit-layer gain exists in any literature this
project has reached across Track E, P2.1 and P2.2.

Applying the existing 0.1×–10× convention to the only number available, the
hand-set illustrative 0.5, would have given Uniform(0.05, 5.0), and the
posterior sits near 0.074. **That lower bound would have been active**, and
the reported "fit" would have been partly a description of the box. Caught
before sampling, not after. The bounds are taken from the mechanism instead:

- **lower 0.0**, the mechanistic null, "diazepam does not potentiate this
  synapse", which predicts a ratio of exactly 1. It is a meaningful,
  falsifiable value rather than a numerical floor, and the data rejects it
  at >4 sd.
- **upper 5.0**: at this exposure a gain of 5 multiplies the GABAergic
  weight by ~5.5, far beyond any benzodiazepine potentiation. A wide box,
  not a claim.

`test_the_published_effect_lands_well_inside_the_gain_prior` asserts the
model's prediction at *each* bound is more than 3 sd from the observation,
which is the statement that actually shows the posterior is not piled
against either one.

### Results

Three independent `pm.Model`s, `DEMetropolisZ`, 4 chains × 1000 tuning steps
× 1000 draws each, unchanged in structure from Track C2. Full summaries in
`results/calibration/{system}_summary.csv`, posteriors and priors as
`InferenceData` alongside, diagnostics in `calibration_diagnostics.txt`.

| System | Free parameters | Max r_hat | Min ESS (bulk) |
|---|---|---|---|
| mu-opioid | `Ki_fentanyl_nM`, `Ki_naloxone_nM` | 1.01 | 585 |
| D2 | `kon_haloperidol_nM_min`, `koff_haloperidol_min`, `kon_dopamine_nM_min`, `koff_dopamine_min` | 1.02 | 284 |
| GABA-A | `Ki_flumazenil_nM`, `Ki_diazepam_nM`, **`gaba_a_connection_modulation_gain`** | 1.02 | 265 |

All three converge (r_hat ≤ 1.02 against the usual < 1.05 threshold, ESS in
the hundreds). mu-opioid and D2 reproduce Track C2's numbers exactly, which
they should: neither model changed, and the seed did not either.

The fitted gain:

    gaba_a_connection_modulation_gain
        posterior mean 0.0742, sd 0.0189, 89% ETI [0.0431, 0.1062]
        r_hat 1.02, ESS(bulk) 309, MCSE(mean) 0.0011

**The hand-set illustrative value was 0.5**, about seven times the fitted
one. At the exposure [Graziani2022]'s dose produces, 0.5 predicts a 43%
suppression of the anxiety index where the published effect supports ~11%.
So the fit is a real correction rather than a confirmation of a lucky guess,
and this is the first time in this project that a circuit-layer number has
been told what it should be by data.

The GABA-A model grew from two parameters to three, and one further thing
changed with it: **`Ki_diazepam_nM` left `PRIOR_ONLY_PARAM_NAMES`.** The
behavioural term is the first likelihood in this scaffold whose value
depends on how much diazepam is actually bound, so diazepam's affinity now
has data behind it and not only a prior. The update is weak by
construction, because at near-saturating occupancy the behavioural ratio
constrains the gain sharply and the Ki barely, and the posterior duly sits
on its prior (3.603 ± 0.097 against a 3.6 ± 0.1 prior). That is reported
rather than overstated.

### The guard test, replaced rather than deleted

`test_no_circuit_layer_parameter_is_fit` asserted a blanket "no free
parameter in this scaffold is a circuit-layer quantity". That was the right
guard while no behavioural dataset existed; with one, it would block a fit
that has real data behind it. It is replaced by
`test_only_evidence_backed_circuit_layer_parameters_are_fit`, which is
stricter where it counts: a circuit-layer parameter may enter this scaffold
only if `priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` names the behavioural
entry that calibrates it, that entry exists, and that entry is **still**
calibration-ready when the test runs. Anything not so declared must still be
a `Ki_`/`kon_`/`koff_` receptor-binding constant, exactly as before. A
second test closes the other direction: a name in that mapping must be a
real free parameter that a likelihood term actually updates, so the
declaration list cannot be used to launder an untouched parameter.

### What the other two systems needed and did not get

**mu-opioid.** All 421 open-access Europe PMC articles matching (fentanyl
AND ("nucleus accumbens" OR microdialysis) AND dopamine AND rat) were
fetched in full and screened. 19 carry an intravenous fentanyl dose; none
reports accumbens dopamine as a percentage of its own baseline. The rat IV
fentanyl literature is a self-administration literature (0.25–2.5 µg/kg per
infusion), which measures intake, not dialysate dopamine. IV is the only
fentanyl route `drug_pk` has kinetics for.

One near-miss changes what a later task should do. **[Yassen2008], already
this codebase's fentanyl rat IV PK source, is itself a PK/PD paper.** It
reports a concentration-anchored fentanyl pharmacodynamic relationship in
the rat (probability of complete antinociception as a logistic function of
biophase concentration; odds ratio 3.03, 95% CI 1.87–4.21, per ng/mL;
T½,ke0 6.08 min, 4.43–7.73) that needs no dose bridge at all. It is not
usable for `mu_opioid_coupling`, which is the VTA_GABA gain read out as
incentive salience: antinociception is a PAG endpoint, and using it for the
reward gain would be endpoint substitution of exactly the kind rule 2
forbids across drugs. It *is* a real candidate for
`mu_opioid_multi_region_coupling()["PAG_Interneuron"]`, whose readout
(`analgesia_index`) is the matching one, and therefore a different parameter from the one
P2.2 was asked to fit, and therefore a different task.

**D2.** All 1303 open-access Europe PMC articles matching (haloperidol AND
rat AND ("locomotor activity" OR catalepsy OR "plasma concentration")) were
fetched and screened for an IV mention or a plasma concentration alongside a
locomotor/cataleptic endpoint. 19 survived; every one doses IP, SC or PO for
the behaviour, or is a review, or is human. IV is the only route `drug_pk`
has haloperidol kinetics for, and that non-intersection is now confirmed by
a second, independent sweep rather than inferred.

The closest article, Lei K, et al., Exp Anim 2018;67(2):163–173,
doi:10.1538/expanim.17-0091 (PMC5955748, read in full), measures rat
haloperidol plasma kinetics after **both** an intravenous (0.7 mg/kg) and an
intragastric (1.4 mg/kg) dose, the pair that yields an absolute oral
bioavailability, but its behavioural arm is the oral one in MK-801-treated
rats, and the basal-ganglia circuit here has no NMDA-antagonist term, so its
locomotor numbers are not comparable with `locomotor_activity_readout` even
if F were extracted from it.

A second, route-free route was searched for and also not found: a rat study
reporting measured **D2 receptor occupancy alongside a locomotor or
cataleptic effect** would let `d2_coupling` be fitted with no PK step at
all, since occupancy is what the circuit layer consumes. The two papers that
do exactly this, Wadenberg ML, Kapur S, Soliman A, Jones C, Vaccarino F,
Psychopharmacology 2000;150(4):422–9 (PMID 10958084), and Johnson M,
Kozielska M, Pilla Reddy V, et al., Pharm Res 2014;31(10):2605–17,
doi:10.1007/s11095-014-1358-7 (PMID 24792824), are both closed access with
no reachable repository copy, and neither abstract carries the
occupancy–effect pairs. That is the single cheapest thing that would unblock
this system, and it is a document-access problem, not a modelling one.

### What this task changes, and what it does not

- **One illustrative label moved, and only one.**
  `gaba_a_connection_modulation_calibrated()` now returns the fitted gain
  with its citation and posterior, in this same commit. Every other
  circuit-layer number in `scenarios.py`, including
  `gaba_a_connection_modulation()` itself, the *illustrative* section's
  function, which is deliberately left at its hand-set 0.5 so the two
  sections stay distinguishable, is exactly as illustrative as before.
- **`mu_opioid_coupling` and `d2_coupling` are unchanged.** Their docstrings
  now point at the P2.2 sweeps as well as Track E's.
- **The behavioural evidence module gained its first usable entry**, and
  `calibration_ready_entries()` is no longer empty. The test that pinned it
  empty is updated to name the entry, so a fourth entry cannot drift into
  the fit unnoticed.
- **No tolerance was widened and no test was skipped.** The one
  pre-existing assertion the new gain interacts with,
  `test_calibrated_diazepam_lowers_anxiety_index_below_baseline`, still
  passes on its original threshold.

## Phase 2: identifiability of connectivity weights and time constants (P2.3)

### The task, and the answer

P2.3's brief is to attempt to fit the amygdala circuit's **connectivity
weights** and **node time constants** against the same behavioural evidence
P2.2 used, and to say which of them are identifiable at all. The answer is
**none of them**, and the interesting part is that this is provable rather
than merely observed: two of the three reasons are exact structural
properties of the model, and only the third is a statement about how much
data happens to exist.

`src/neurocascade/calibration/identifiability.py` runs the analysis,
`scripts/run_identifiability.py` produces the artifacts under
`results/identifiability/`, and `tests/test_identifiability.py` pins the
claims. Every circuit-layer weight, tau, node gain and baseline input in
`scenarios.py` keeps its illustrative label, and no number in that file
changed.

### Why this needed three layers and not one profile plot

A profile likelihood answers "given everything else, what does the data say
about this parameter?" - and in this model the "given everything else" is
carrying almost all of the weight. Running the profiles alone would have
produced a table with several parameters marked identifiable, and every one
of those marks would have been an artefact. So the analysis is built in
three layers, weakest assumption first, and each layer subsumes the one
before it.

**Layer 1, structural: two exact degeneracies, proved.**

The behavioural observable is a drug/vehicle ratio of a **steady-state**
readout (the elevated plus maze runs 60 min after injection; the circuit's
slowest tau is 2 min, so it has equilibrated by a factor of 30 before the
animal is on the maze). Two things follow exactly.

*Time constants drop out entirely.* The Wilson-Cowan fixed point solves
`r* = sigmoid(W r* + drive)`; `tau` appears only in `tau_i dr_i/dt = -r_i +
sigmoid(...)`, i.e. only in how fast the approach is, never in where it
lands. So `d(ratio)/d(tau_i)` is identically zero. Measured, the three tau
sensitivities come out at 3e-8 to 1.7e-7 against non-tau sensitivities of
4.3e-2 to 2.8, five orders of magnitude apart and consistent with the
integrator's own rtol of 1e-8 rather than with any real dependence.
`test_time_constants_do_not_move_the_observable_at_all` sweeps each tau over
its whole 0.1x-10x range and requires the prediction not to move.

*The eight remaining parameters collapse into four combinations.* The
amygdala circuit is feedforward (`GABA_Interneuron -| BLA -> CeA`), so its
fixed point can be written in closed form, and when it is, the three node
gains, three baseline inputs and two connectivity weights enter only through

    A = gain_BLA * W[BLA, GABA_Interneuron] * L(gain_I * baseline_I)
    B = gain_BLA * (baseline_BLA - threshold_BLA)
    C = gain_CeA * W[CeA, BLA]
    D = gain_CeA * (baseline_CeA - threshold_CeA)

with the drug entering as `A -> A * (1 + g * activation)`, so that the entire
observable is

    ratio = L(C*L(A*(1 + g*a) + B) + D) / L(C*L(A + B) + D)

Eight parameters, four combinations: a **four-dimensional exact degeneracy**
on top of the three dead tau directions. This is not inferred from flat
profiles. `equivalent_parameterisation()` constructs a numerically very
different amygdala circuit with identical `{A,B,C,D}` - node gains
4/1.5/4 -> 2/0.6/1.1, the GABAergic weight -2.5 -> -7.53, the CeA weight
2.0 -> 7.27 - and `results/identifiability/structural_degeneracy.png` shows
the two predicting the same ratio at every drug gain, to within 2e-9, i.e.
to the integrator's tolerance and no worse.

The closed form is used for the argument and the integrator is used for the
model, so the two are checked against each other rather than assumed equal:
`test_closed_form_reproduces_the_integrated_forward_model` and
`test_closed_form_still_matches_away_from_the_nominal_circuit` require
agreement to 1e-7 and 1e-6 at the nominal circuit and at three perturbed
ones. If they ever disagree, the structural argument would be about a
different model than the one this codebase runs, which is the failure mode
worth guarding.

**Layer 2, information content: one datum, one direction.**

`behavioral_evidence.calibration_ready_entries()` returns exactly one entry.
For a scalar Gaussian observation the Fisher information matrix of any
parameter vector is `F = s s^T / sd^2`, which is **rank one** however many
parameters it is built over. Computed over all twelve parameters, its
eigenvalues are

    3.09e+03,  then 5.0e-13 and below

i.e. one informed direction and eleven at machine zero
(`results/identifiability/fisher_spectrum.png`). The single informed
eigenvector is dominated by the drug-action gain, with exactly zero
component on any tau.

That is the whole budget. P2.2 spent it on `gaba_a_connection_modulation_
gain`, and this analysis is where that choice gets defended rather than
assumed: the gain is the only parameter in the set that is about the **drug**
rather than about the circuit, so a drug/vehicle contrast is the matching
kind of evidence for it and is not the matching kind of evidence for a
synaptic weight or a membrane time constant.

**Layer 3, profile likelihood, in three regimes.**

`results/identifiability/profile_likelihood_amygdala.png` shows, per
parameter, the 95% (delta-chi-square = 3.84) profile in two regimes, and
`reduced_profile_likelihood.png` the third:

- **A** - only the scanned parameter free; every other circuit number,
  including P2.2's gain, held at its current value.
- **B** - the scanned parameter *and* the drug gain free, the gain
  re-optimised at every grid point.
- **C** - the scanned parameter and every other circuit parameter free.
  Because the observable depends on the originals only through
  `{A,B,C,D}`, profiling in that reduced space **is** regime C, which is
  what makes it computable at all.

In regime A, several parameters have finite intervals. In regime B most of
those intervals open up. In regime C nothing is localised. The three
regimes differ only in how many illustrative numbers are being treated as
known, and that is the point:
`test_freeing_the_drug_gain_removes_the_apparent_bounds` requires at least
four parameters to be bounded in regime A and strictly fewer in regime B,
so the collapse cannot quietly stop happening.

`weight_BLA__GABA_Interneuron` is the clearest case. In regime B it has a
finite interval, which looks like the connectivity-weight fit P2.3 was asked
for. It is not one: that weight enters only inside `A`, so its "interval" is
an interval on `A` re-labelled through a fixed `gain_BLA` and a fixed
interneuron rate, both illustrative.

### The explicit list

Amygdala anxiety circuit, the only circuit in the codebase with a
calibration-ready behavioural observation attached to it:

| parameter | kind | verdict | why |
|---|---|---|---|
| `tau_GABA_Interneuron` | tau | **structurally unidentifiable** | steady-state observable; sensitivity 1.1e-7 |
| `tau_BLA` | tau | **structurally unidentifiable** | sensitivity 3.1e-8 |
| `tau_CeA` | tau | **structurally unidentifiable** | sensitivity -1.7e-7 |
| `gain_GABA_Interneuron` | node gain | **structurally degenerate** | enters only inside `A` |
| `gain_BLA` | node gain | **structurally degenerate** | enters only inside `A` and `B` |
| `gain_CeA` | node gain | **structurally degenerate** | enters only inside `C` and `D` |
| `baseline_input_GABA_Interneuron` | baseline | **structurally degenerate** | enters only inside `A` |
| `baseline_input_BLA` | baseline | **structurally degenerate** | enters only inside `B` |
| `baseline_input_CeA` | baseline | **structurally degenerate** | enters only inside `D` |
| `weight_BLA__GABA_Interneuron` | connectivity | **structurally degenerate** | enters only inside `A` |
| `weight_CeA__BLA` | connectivity | **structurally degenerate** | enters only inside `C` |
| `gaba_a_connection_modulation_gain` | drug action | **one-sided identifiable** | bounded away from zero, unbounded above once the circuit is free |

Reward (mu-opioid, 17 circuit parameters), basal-ganglia indirect pathway
(D2, 19) and multi-region mu-opioid (27): **all unidentifiable, trivially**.
Those circuits have zero calibration-ready behavioural observations
(`behavioural_observation_counts()` counts them from the evidence module
rather than asserting it), so their Fisher information matrix is the zero
matrix. That is 63 further circuit-layer parameters with no data behind
them, and saying so by name is more useful than saying "the rest".

**So: zero identifiable connectivity weights, zero identifiable time
constants, and one partially identifiable drug-action gain.** Every one of
the eleven circuit-structure parameters keeps its illustrative label.
`gaba_a_connection_modulation_calibrated()` keeps the P2.2 fit it already
had, which was never illustrative and is not being made illustrative here;
what it gains is the conditionality statement below.

### What P2.3 says about P2.2, said plainly

P2.2 fitted `gaba_a_connection_modulation_gain = 0.0742 [0.0431, 0.1062]`
and recorded that the circuit's illustrative values "set how *sensitively*
the ratio depends on the fitted gain, which is a real dependency and is why
P2.3, not P2.2, is where they belong." P2.3 quantifies that, and the answer
is stronger than "some dependency":

- **Conditional on the illustrative circuit**, the gain is well determined.
  The regime-A profile interval is
  [0.0390, 0.1108], which brackets P2.2's independently sampled 89% ETI, as
  it must - `test_the_drug_gains_own_profile_agrees_with_p2_2s_posterior`
  makes that a cross-check between an optimiser here and PyMC there, over
  what should be the same likelihood.
- **Unconditionally**, that interval is gone. In regime C the gain's profile
  is flat from the bottom of the searched range up to the top of its prior
  box: with the circuit free, a large gain acting on a weak GABAergic
  synapse and a small gain acting on a strong one are indistinguishable from
  one drug/vehicle ratio.
- **What survives unconditionally is one thing, and it is a real one.** A
  drug gain of *exactly zero* predicts a ratio of *exactly one* for every
  possible circuit - both legs of the ratio become the same expression - so
  its delta-chi-square of 17.5 (4.2 sd) does not depend on a single
  illustrative number. The mechanistic null "diazepam does not potentiate
  this synapse" is rejected by the data, unconditionally.
  `test_the_mechanistic_null_is_rejected_whatever_the_circuit` asserts the
  exact-1.0 identity in closed form and to integrator tolerance in the
  integrated path, over four arbitrary circuits.

This does not retract P2.2's fit and does not move its label. It says what
the fit is: a posterior for one gain *given* a circuit this project has not
measured, whose one circuit-independent content is the rejection of zero.

### Design decisions worth recording

**The verdict "bounded" is not the verdict "calibrated", and the threshold
that separates them is borrowed, not invented.** The reduced-space profiles
do produce finite bounds - `A`'s 95% interval is [-84.6, -4.4e-4] - and
reporting those bare would be misleading, because both ends are just the
corners where the circuit saturates and no drug effect of *any* size is
possible. Excluding "the circuit is dead" is not measuring a parameter. So a
both-sided interval is only called identifiable if it is also narrower than
`IGNORANCE_SPAN_RATIO`, which is the width of this codebase's own
UNINFORMED prior box (`priors.py`'s 0.1x-10x convention), i.e. the span
this project already uses to mean "we know essentially nothing". An interval
spanning 1.9e5 fails that by more than three orders of magnitude. Reusing
an existing convention rather than picking a new number keeps this from
being one more free dial.

**Bounds are labelled by where they land.** `bounds_at_searched_edge()`
reports whether each finite endpoint sits in the outermost searched cell,
because a crossing in the interior is the data excluding a value while a
crossing at the edge is the box running out and would move if the box moved.
Both occur here and they read very differently.

**The scan ranges are derived, not chosen.** Strictly-signed parameters
(taus, node gains, connectivity weights) get 0.1x-10x their `scenarios.py`
value, which is `priors.py`'s existing UNINFORMED convention. `baseline_
input` can legitimately be zero or negative, so a multiplicative range is
meaningless for it; it gets `+/- 4.595/gain`, the additive shift over which
that node's own logistic sweeps 1% to 99% of its range. The drug gain gets
P2.2's mechanism-derived prior box. `test_scan_ranges_are_derived_rather_
than_typed_in` checks every range brackets its own nominal value and that
the multiplicative ones all have exactly the same span.

**The inner optimiser is an exhaustive lattice, not a gradient method, and
that choice is a correctness matter rather than a performance one.** The
reduced-space profile's inner problem is a 4-parameter minimisation of a
non-convex chi-square over a box spanning several orders of magnitude.
Multistart L-BFGS-B fails on it, and it fails in the dangerous direction: a
stuck optimiser returns a chi-square that is too **high**, which is reported
as identifiability that is not there. The first version of this analysis did
exactly that, reporting the drug gain as bounded above at 0.15 when a
feasible parameter set at gain 5 can be written down by hand. Replacing it
with a vectorised lattice plus local polish removed the bound. Since
chi-square is bounded below by zero, any feasible point found is an upper
bound on the profile, so a profile reported as flat is the conservative
claim: the truth can only be flatter, never more identifiable.

**Nothing here enters `priors.PRIOR_SPECS`.** The joint models this module
builds are diagnostics that measure a posterior correlation, not
calibrations, so they carry their own local prior specs and never pass
through `test_only_evidence_backed_circuit_layer_parameters_are_fit`.
`test_this_module_adds_no_parameter_to_the_production_fit` asserts that the
only overlap between this module's parameter list and `FREE_PARAM_NAMES` is
the one P2.2 already declared.

**Every verdict is computed, none is written down.** `classify()` encodes no
expected answer; `test_no_weight_and_no_tau_comes_out_identifiable` runs the
real analysis and fails if a future task's data changes the result, with a
message naming the two things that would then have to be revisited (the
illustrative label, and this section).

### The posterior-correlation view

The rank argument and the profile likelihoods say the same thing, and a
reader who wants to see it rather than accept it gets
`results/identifiability/posterior_correlation.png`: two joint
`DEMetropolisZ` posteriors over the drug gain and one circuit parameter,
against the single [Graziani2022] observation, with uniform priors over
exactly the same boxes the profiles scan.

- `weight_CeA__BLA` and the gain trade off along a ridge: posterior
  correlation **+0.965**, and the weight's own marginal is only 16% narrower
  than its uniform prior (posterior sd / prior sd = 0.840). Practical
  non-identifiability, seen directly.
- `tau_BLA`'s marginal **is** its prior (posterior sd / prior sd = 0.954)
  and its correlation with the gain is **+0.063**, i.e. zero: structural
  non-identifiability, seen directly. A likelihood cannot depend on a
  parameter the observable does not depend on.

**One of those two chains did not converge, and that is reported rather
than dropped.** The `tau_BLA` run is clean (max r_hat 1.01, min bulk ESS
205). The `weight_CeA__BLA` run is not: max r_hat **1.11** against the usual
< 1.05, min bulk ESS **14**. Those numbers are in
`results/identifiability/joint_posterior_diagnostics.csv` and are not
presented as a converged posterior. The failure is the expected one and is
part of the result rather than an obstacle to it: `DEMetropolisZ` proposes
from a shared history, and on a one-dimensional ridge with no mode there is
nothing for the chains to agree about, so they wander along the ridge at
different rates. A well-mixed chain would have been the surprising outcome.
What the run therefore supports is the qualitative shape - a ridge, a
near-prior marginal - and not a precise correlation coefficient, which is
why the two `slow` tests assert `|corr| > 0.5` and `posterior sd / prior sd
> 0.5` rather than the specific values above. The quantitative
non-identifiability statement rests on the profile likelihoods and the
rank-one Fisher matrix, which are deterministic and need no chain.

Both are tests as well as figures
(`test_a_time_constants_posterior_is_exactly_its_prior`,
`test_a_connectivity_weight_trades_off_against_the_drug_gain`), marked
`slow` for the same reason `test_bayesian_calibration.py` is: they run real
chains.

### What would actually identify these, which is the useful output

This analysis says more than "no". It says what each class of parameter
needs, and the three answers are different, which matters for what Phases 3
and 7 should collect.

- **Time constants need a time-resolved measurement, and no amount of
  endpoint data of any kind will do.** The obstruction is not sample size;
  an equilibrium observable is exactly independent of tau. A behavioural or
  physiological readout sampled on the 1-2 min timescale of the circuit's
  own relaxation would identify them; a hundred more elevated-plus-maze
  studies would not. This is a constraint on P7.1's endpoint library, not
  on the size of P2.1's evidence module.
- **The four combinations `{A,B,C,D}` need at least four independent
  behavioural observations, and the cheapest source is a dose-response
  curve.** Every entry in the evidence module is currently a single dose,
  which gives one activation `a` and therefore one equation. A published
  rat elevated-plus-maze **dose series** for diazepam would give one
  equation per dose against the same four unknowns. That is the single
  highest-value thing a future P2.1-style pass could look for, and it is a
  different search from the one P2.2 ran (which screened for *any* usable
  dataset, and found one).
- **The individual parameters inside each combination need node-level
  observables, not more behaviour.** `anxiety_index_readout()` reads CeA
  alone, so `gain_CeA` and `W[CeA, BLA]` can never be separated by it. A
  readout that also scores BLA would split `A` from `B`; a measured
  connection density would fix `W` directly and let the behavioural datum
  constrain the gain instead. That is what Phase 3's Allen connectome
  backbone actually buys, stated as an identifiability gain rather than as
  a data-source upgrade.

### What this task changes, and what it does not

- **No parameter changed and no illustrative label moved**, in either
  direction. `test_no_illustrative_circuit_parameter_changed_value` pins the
  amygdala circuit's eleven numbers at their v0 hand-set values.
- **P2.2's fit is unchanged and now carries its conditionality explicitly.**
  `GABA_A_CONNECTION_MODULATION_GAIN` still returns 0.0742;
  `gaba_a_connection_modulation_calibrated()`'s comment now records that the
  point value is conditional on the illustrative circuit and that the
  circuit-independent content of the fit is the rejection of zero.
- **Three circuits are now on record as having no behavioural data at all**,
  by name and by parameter count, computed from the evidence module rather
  than asserted.
- **No tolerance was widened and no test was skipped.** One tolerance in a
  new test is set from `circuit_forward._STEADY_STATE_RTOL` rather than
  chosen, and a second test backs it by showing the residual it admits
  shrinks by three orders of magnitude when the integration is tightened.

## Phase 3: ingesting the Allen mouse connectome (P3.1)

### What this task is, and the one thing it must not become

P3.1 is an ingestion task. It downloads two published mouse connectome
matrices and the CCF structure ontology, verifies them, and hands back a
directed weighted matrix whose rows and columns are named brain regions.
It does **not** build a `CircuitNetwork`, assign excitatory/inhibitory
signs, choose a resolution, or move any existing parameter. Those are
P3.2, P3.3 and P3.4, and keeping them out was the main design pressure
here, because almost every convenience one might add to a loader (drop the
diagonal, threshold weak connections, row-normalise, symmetrise) is
actually a modelling decision with consequences for the circuit layer.
`RegionalConnectome` therefore stores the published numbers unmodified and
offers `without_self_connections()` as an explicit method rather than a
default.

The new package is `src/neurocascade/connectome/`:

    sources.py   the manifest: URLs, SHA-256 digests, sizes, citations,
                 and what each matrix's numbers mean
    fetch.py     download, verify, cache; the only network code
    ontology.py  the CCF structure graph as a queryable hierarchy
    matrix.py    RegionalConnectome plus one reader per published format
    fixtures/    real data, fifteen regions of it, committed

plus `scripts/fetch_connectome.py` (the CLI) and
`scripts/make_connectome_fixture.py` (regenerates the fixtures from the
verified cache).

### Two datasets, and the region that decided which one is primary

The roadmap names the Allen Mouse Brain Connectivity Atlas; the task named
Oh et al. 2014 and Knox et al. 2019. Both are ingested.

- **[Knox2019]** Knox JE, Harris KD, Graddis N, Whitesell JD, Zeng H,
  Harris JA, Shea-Brown E, Mihalas S. "High-resolution data-driven model
  of the mouse connectome." Network Neuroscience. 2019;3(1):217-236.
  doi:10.1162/netn_a_00066. Regionalized voxel model, **291** grey-matter
  CCF regions, published as four plain CSVs on Allen's static publication
  download server.
- **[Oh2014]** Oh SW, Harris JA, Ng L, et al. "A mesoscale connectome of
  the mouse brain." Nature. 2014;508(7495):207-214.
  doi:10.1038/nature13186. The homogeneous (region-by-region non-negative
  least squares) model, **213** regions, published as Supplementary Table 3.

Knox is the default. The recency argument and the region-count argument
are both real but generic; the argument that actually settled it is
specific to this codebase. The two region sets overlap in 202 acronyms:
11 Oh regions are absent from Knox, and 89 Knox regions are absent from
Oh. One of those 89 is **LC, the locus coeruleus**, a coupled node in
this codebase's existing multi-region mu-opioid scenario, and the whole
sedation/arousal arm of it. A connectome that cannot represent LC cannot
carry a scenario this project has already built and tested. So Knox is
primary and Oh is the second opinion.

Oh is worth keeping for two things Knox's regionalized CSVs do not have:
per-connection **p-values**, and a different model class (regional
homogeneity rather than voxel-scale smoothing), so P3.2/P3.3 can ask
whether a whole-brain result survives changing the connectome underneath
it. That difference is in the model, not the data: Oh fit 469 tracer
experiments and Knox fit 428 of the 491 then available from the same Allen
atlas, citing Oh as the source, so agreement between the two matrices is
evidence about modelling assumptions and not independent replication of
the anatomy. The manifest says so in as many words, because "we checked it
against a second connectome" is exactly the sentence that would otherwise
be over-read later.

### The naming trap: density and strength are not synonyms

The task asked for "the normalized connection density matrix". Knox
publishes a file with exactly that name. Oh's Supplementary Table 3 does
not contain one, and calling it that would have been wrong in a way no
test would have caught.

[Knox2019] publishes four regionalizations of the voxel-scale weight
matrix. Writing `w_ij` for the summed voxel-pair weight from source region
X (|X| voxels) to target region Y (|Y| voxels):

    connection_strength             w_ij                neither size divided out
    connection_density              w_ij / |Y|          target size only
    normalized_connection_strength  w_ij / |X|          source size only
    normalized_connection_density   w_ij / (|X| |Y|)    both

The paper's Methods define the first, third and fourth symbolically; all
four are defined in code in the Allen Institute's companion package
`mouse_connectivity_models` (`RegionalizedModel`, whose own docstring
example states `normalized_connection_density.shape == (291, 577)`, the
published files' exact shape, and which builds them from the voxel array,
pinning these CSVs to the regionalized *voxel* model rather than to the
paper's recomputed homogeneous one). So the density is the *mean
voxel-to-voxel model weight* between a source voxel and a target voxel,
which is the right measure when regions differ in volume by orders of
magnitude, as CP and LC do.

[Oh2014]'s Supplementary Notes define their `w_X,Y` as "the amount of
segmented signal activated in the target region by infecting one voxel in
the source region ... proportional to the average out-degree of neurons
projecting from the source to the target". That is a normalized connection
**strength**: the target region's size is not divided out. Knox make the
same point from the other side, noting that their normalized connection
strength is the normalization "necessary to compare directly with the
homogeneous model".

Consequence recorded in the code: every `DataFile` carries a `units`
string quoted from its own source, `RegionalConnectome` propagates it onto
the loaded object, and there is a test asserting that Oh's units string
says "NOT a connection density". The two datasets' numbers are not
commensurable and nothing in this package pretends otherwise. The one file whose
normalization the paper does not write out, `connection_density.csv.gz`,
is manifested with its definition taken from that companion
implementation and attributed to it, rather than inferred by elimination
from the paper: inferring it would have produced the right answer here and
the habit of guessing definitions everywhere else.

### Orientation, and why it gets its own accessor

This is the failure mode that would have been hardest to notice. Both
publications *print* their matrices source-major (sources on rows), and
[Knox2019] is explicit that this is the transpose of their own
mathematics: "we depict sources as rows and targets as columns (W^T), the
opposite of our mathematical convention". This codebase's
`CircuitNetwork.connectivity[i, j]` is "from `j` onto `i`", i.e. the
mathematical convention.

A silently transposed connectome produces a whole-brain model that is
wrong about every projection's direction while looking completely healthy:
same sparsity, same weight distribution, same degree statistics, plausible
dynamics. So:

- `RegionalConnectome.weights` keeps the file's own source-major
  orientation, so the array and the cached bytes never disagree about what
  row 0 is;
- `projection_matrix()` and `square_ipsilateral()` are the only places a
  transpose happens, and they say so in their names and docstrings;
- the regression test asserts the VTA→ACB and ACB→VTA weights
  *separately* (1.07e-4 and 4.69e-4 in the normalized density), which a
  transposed load cannot pass by accident, rather than checking a shape.

### Three properties of the real files, found by looking rather than assumed

**The ipsilateral and contralateral blocks are not the same width.**
Knox's matrix is 291 sources x 577 targets: 291 ipsilateral and only 286
contralateral. The five missing contralateral targets are RM, RO and RPA
(nucleus raphe magnus, obscurus, pallidus), OV (vascular organ of the
lamina terminalis) and EW (Edinger-Westphal nucleus): midline structures
with no contralateral counterpart. So a ragged column count is the correct
parse, not an error, and the reader does not require the blocks to match.

**Oh's p-value sheets carry the literal Excel token `#NUM!`, and it means
something.** On the published file, `#NUM!` appears in 28415 ipsilateral
and 29501 contralateral p-value cells, and the equivalence

    p-value cell is '#NUM!'   <->   connection weight is exactly 0

holds in both hemispheres with no exceptions, which is what one would
expect from a model that only assigns confidence to connections it gave
non-zero weight. The reader maps the token to **NaN**, and refuses a blank
cell outright. Mapping it to 0.0 would have been the single most damaging
available mistake, because a p-value of zero reads as maximal confidence
in exactly the connections the model says do not exist. That refusal is
tested, and the fixture was deliberately extended (with STN) so that the
committed fixture actually contains 184 of these cells rather than
exercising the path only on data that is not committed.

**All 1327 CCF acronyms in structure graph 1 are distinct.** That is what
makes Oh's acronym-labelled matrix joinable to Knox's id-labelled one at
all. `StructureOntology` checks it on load and raises rather than silently
keeping the last of a duplicate pair, because the join would otherwise
mis-assign rows in a way that produces a valid-looking matrix.

### Reproducibility: what is committed and what is not

The constraint is that the source data is about 8 MB (and the voxel-scale
factors, deliberately not manifested, are another 635 MB), which must not
enter git, while the build has to be reproducible from a clean checkout.
The split:

- **Committed:** the manifest (`sources.py`) with each file's URL,
  SHA-256, byte count, retrieval date, citation and units; the fetch
  script; the loaders; and ~44 kB of real-data fixtures.
- **Gitignored:** `data/connectome/`, holding exactly the published bytes
  under the manifest's filenames. Nothing writes derived or converted
  files there, so there is never a stale intermediate to invalidate.

`python scripts/fetch_connectome.py` on an empty cache reproduced all six
files byte-identically to the manifest, and `--verify-only` re-checks a
populated cache. `--checksums` prints the manifest in `sha256sum -c`
format so the cache can be verified with the standard tool rather than
with this package's own code checking itself.

**A checksum mismatch is a hard failure with no override.** The bytes are
kept as `<filename>.rejected` for inspection and nothing downstream sees
them. The temptation this forecloses is obvious and worth naming: when the
upstream file changes, editing the expected digest makes the build green
in one line and silently moves every result that depends on the
connectome. The error message says not to, and the two legitimate
responses (re-download, or read the diff and bump the manifest in a commit
that explains what moved) are spelled out in it.

**One of the six sources is a live API and is checksummed anyway.** The
ontology comes from `structure_graph_download/1.json`, which Allen renders
from a database per request; the other five sit on static archive servers.
It was verified byte-stable across two requests, but nothing stops Allen
revising the CCF. Pinning it means a revision shows up as a dated,
loud failure that a human reads, instead of a region rename or a
re-parenting propagating into every whole-brain result with no record.
That is the intended behaviour of the checksum, not a limitation of it.

### The fixtures are real data, and why that mattered more than it looks

The tests must run offline. The obvious way to do that is a hand-written
3x3 matrix of made-up numbers, and it would have tested nothing that
matters: not the two-row CSV header, not the ragged ipsi/contra width, not
the acronym-to-id join, not the `#NUM!` tokens, and not the fact that a
real region's self-projection is non-zero in every one of the 291 rows
(and the largest entry in 109 of them).

So each fixture is a *slice of the checksum-verified published file,
written back out in that file's own format*: fifteen regions instead of
291, ~44 kB instead of ~8 MB, same parsers. `PROVENANCE.md` records which
verified digest each was cut from, and the generator calls `fetch.verify()`
on all three upstream files before reading any of them, so a fixture can
never be cut from unverified bytes.

The fifteen regions are not arbitrary: they are the CCF regions this
codebase's existing scenarios already name (VTA, ACB, PL, CP, GPe, STN,
GPi, SNr, PAG, LC, BLA, CEA) plus cortical, thalamic and raphe partners
(MOp, MD, DR), spanning eight of the twelve major divisions. That makes
the fixture directly useful to P3.3, whose job is mapping exactly those
hand-built scenarios onto connectome regions, and it means the numbers the
tests assert on are numbers about regions this project actually models.

Two honest caveats are recorded in `PROVENANCE.md` rather than glossed:
the CSV fixture is bit-identical to the published doubles (Python's
shortest round-tripping `repr`), while the xlsx fixture round-trips to 16
significant figures only (1 ulp; openpyxl serialises floats as `%.16g`):
a property of the file format, not a tolerance chosen to make anything
pass. And a fifteen-region slice of a 291-region whole-brain fit is a test
input, not a model: every path through the other 276 regions has been
deleted.

### What was searched for and not found, or found and deliberately not taken

- **The full voxel-scale model** (`nodes.csv.gz`, 490 MB;
  `weights.csv.gz`, 145 MB) is published in the same Knox directory and is
  **not** in the manifest. P3.1's target is a regional matrix, and a
  checksum may not be written down for bytes that have not been retrieved
  and hashed. If a later task needs voxel resolution, it downloads them,
  hashes them, and adds them.
- **Oh's Supplementary Table 1** (region names and acronyms) was not
  ingested: the CCF structure graph already supplies every name, full name
  and hierarchy position, from the same ontology both matrices index into,
  so a second name table would be a second source of truth for the same
  fact.
- **A static, versioned copy of the CCF ontology** was searched for on
  Allen's `informatics-archive` server, which does host versioned CCF
  directories (`ccf_2015` ... `ccf_2022`). Those contain annotation
  volumes, structure masks, meshes and centres, but no structure-graph
  table; `structure_tree_safe_2017.csv` and the equivalent paths return
  404. So the live API endpoint is the only route to the ontology, which
  is why the volatility note above exists rather than a nicer solution.
- **Neurotransmitter identity per region**, which P3.2 needs to give the
  connectome signs, is not in any of these files. The tracer data behind
  both matrices is bulk anterograde fluorescence and carries no sign at
  all: every weight in both datasets is non-negative by construction, and
  `RegionalConnectome` treats a negative weight as a parse failure rather
  than as an inhibitory projection. Sourcing sign is P3.2's problem and it
  is a genuinely separate search.

### What this task changes, and what it does not

- **No parameter changed, no illustrative label moved, no scenario
  touched.** The connectome package is additive; nothing in `engine.py`,
  `network.py`, `scenarios.py` or any calibration module imports it yet.
  Every existing test result is unchanged.
- **No number in this package is a modelling choice.** Every value is
  either a published matrix entry, a measured SHA-256, or a CCF structure
  id taken from a published mask file. There is nothing here to label
  illustrative, and nothing here calibrates anything.
- **P2.3's identifiability finding gets its first piece of relief, but
  only in principle.** P2.3 recorded that a measured connection density
  would fix `W` directly and let a behavioural datum constrain the gain
  instead of the product. This task supplies the measured densities. It
  does not yet convert one into a `CircuitNetwork` weight, and the
  conversion is not a rescaling: a normalized connection density is a
  mean voxel-to-voxel tracer weight, dimensionless and unsigned, while
  `CircuitNetwork.connectivity` is a signed drive coefficient in a
  Wilson-Cowan mean-field. What that conversion may honestly assume is
  P3.2's question, and it should be answered before any claim that the
  circuit layer is "connectome-based".
- **47 new tests.** 46 of them run offline off the committed fixtures; the
  47th checks the real 291-region file and skips when the gitignored cache
  is absent. It is an extra, not a substitute: every acceptance criterion
  is covered by the 46 offline tests.

## Phase 3: building the circuit from the connectome (P3.2)

### What this task is, and the line it must not cross

P3.1 ingested two published matrices and refused to make a single
modelling decision with them. P3.2 is where the decisions get made, and
there are only four of them that matter: which published number becomes a
weight, where the sign comes from, what a "resolution" is, and what
happens to the regions the sign rule cannot reach. Each is recorded below
with the alternative that was rejected, because every one of them has a
convenient wrong answer that would produce a whole-brain network that
looks completely healthy.

The line that must not be crossed is the one this codebase has held
since Track A: no number may enter that is not either published, derived
from published numbers by stated arithmetic, or labelled illustrative.
P3.2 adds exactly one *kind* of new content to the project - a
categorical, region-level claim about neurotransmitter identity - and it
is sourced the same way a `Ki` is.

The new modules are

    neurotransmitter.py  the sign-assignment rule, its citations and its
                         exceptions
    volumes.py           region voxel counts, recovered from ratios of the
                         published matrices
    build.py             parcellation, aggregation, and the builder

and nothing in `network.py`, `circuit.py`, `engine.py` or `scenarios.py`
was touched.

### The sign rule

    sign(edge j -> i) = sign of the transmitter released by the
                        long-range PROJECTION neurons of source region j
        glutamatergic +1,  GABAergic -1,  anything else: no sign

Three properties of that sentence are load-bearing.

**Sign is a property of the source region alone.** That is what makes it
assignable from region-level anatomy at all, and it is an approximation
whose failure mode (a source with two projection populations of opposite
sign innervating different targets) is named in the module's exception
list rather than hidden.

**Projection neurons, not neurons.** Cortex holds a substantial GABAergic
minority by cell count and its *edges* are still excitatory, because
essentially all of those interneurons are local. The striatum runs the
other way: its GABAergic spiny neurons both project and dominate
numerically. A rule that counted cells would get cortex wrong; a rule that
counts the axons leaving the region gets both right.

**The rule is applied at the nearest ANCESTOR with a rule**, over the CCF
hierarchy P3.1 already loads. Each entry is stated at the coarsest level
of the ontology at which it is true (`TH -> +1`), and a more specific
entry below it wins (`RT -> -1`). Nineteen rules cover the brain this
way, and adding a twentieth cannot silently move any of the other
nineteen.

The table is sourced per entry: Isocortex/OLF/HPF/CTXsp glutamatergic
[Yao2023, Melzer2020]; STR and PAL GABAergic [Gerfen2011, Smith1998];
TH glutamatergic with RT the exception [Arcelli1997, Pinault2004]; STN
glutamatergic [Schweizer2014]; ZI GABAergic [Mitrofanis2005]; SNr
GABAergic [Smith1998]; CBX GABAergic and CBN glutamatergic
[Uusisaari2011]; VTA/SNc dopaminergic [Morales2017], DR serotonergic and
LC noradrenergic [Yao2023]; PALv/PALm mixed [Do2016]. Full citations with
species and what each supports are in `neurotransmitter.CITATIONS`, and a
test asserts every rule carries at least one.

Two sourcing choices are worth stating explicitly. The **thalamic** rule
is defensible in mouse and would not be in primate: [Arcelli1997] is a
comparative study showing GABAergic local neurons in dorsal thalamus
increase with species complexity and are close to absent in rodent
outside the visual nuclei. Species discipline cuts both ways here, and
this is a case where it favours the simple rule rather than blocking it.
And [Yao2023] (the whole-brain mouse single-cell atlas) is cited for
region-level transmitter identity but **not ingested**: its per-cluster
neurotransmitter assignments over 5,322 clusters are what would replace
this hand-curated table with measured per-region compositions, and doing
that is a P3.1-shaped data ingestion task, not something to approximate
here. That is written into the module docstring as the intended
successor.

### The regions that get no sign, and the number that matters

Three classes fall outside glutamate/GABA and are recorded with a
transmitter class and **no sign** rather than being pushed to +/-1:

- **Neuromodulatory nuclei** (VTA, SNc, DR, LC). A dopaminergic axon does
  not add a term to its target's synaptic input; it changes how the
  target responds, through metabotropic receptors - which is precisely
  the layer this codebase already models explicitly, and precisely what
  roadmap Phase 6 is for. Giving VTA a +1 in the connectivity matrix
  would double-count the mechanism the receptor layer exists to
  represent. This is the same argument the multi-region opioid scenario
  already makes about LC's receptor density not setting its coupling
  gain.
- **Genuinely mixed output nuclei** (basal forebrain PALv/PALm), whose
  long-range output is cholinergic and GABAergic and glutamatergic in
  comparable measure [Do2016].
- **Regions with no rule**: most of hypothalamus, midbrain tegmentum,
  pons and medulla.

On the full 291-region ipsilateral block the rules sign **151 of 291
regions, carrying 90.9% of all outgoing tracer weight**; the 140 unsigned
regions (131 no-rule, 5 mixed, 4 neuromodulatory) carry 9.1%. Both
numbers are asserted by a test so they cannot drift, and they are the
honest measure of how far the rule reaches. The sharpest consequence is
stated rather than buried: **no pons or medulla region has a sign**, so a
twelve-node major-division network has two nodes with no outgoing edges
at all. A test asserts that those two nodes are exactly `P` and `MY`, so
the hole is a checked fact rather than something a reader has to notice.

### Which published number becomes a weight

P3.1's default matrix is the *normalized* connection density,
`s_ij/(|X_i||Y_j|)`. The builder deliberately uses the **connection
density**, `s_ij/|Y_j|`, instead. The reason is in the Wilson-Cowan
equation: `r_j` is a mean rate *per neuron*, so the drive a source
delivers scales with how many neurons it has, and dividing by the
source's size `|X_j|` would assert that a 13,009-voxel caudoputamen and a
7-voxel locus coeruleus deliver the same input at the same firing rate.
Dividing by the *target's* size is right for the opposite reason: what
enters the target's sigmoid is input per target neuron. So P3.1's default
is the right measure for the question "how connected are these two
regions" and the wrong one for a rate model's drive, and this is recorded
here rather than left as an unexplained key change.

### Voxel counts, recovered rather than fetched

Aggregating regions requires the region sizes, because a normalized
quantity does not sum. The coarse density of a region pair is
`(sum of s_ij) / (sum of |Y_j|)`, i.e. what Knox et al. would have
published had they regionalized at that level.

Those voxel counts are not in any file this project fetched - and they
did not need to be. They are recovered from ratios of matrices already in
P3.1's manifest:

    connection_strength / connection_density                        = |Y_j|
    normalized_connection_strength / normalized_connection_density  = |Y_j|
    connection_density / normalized_connection_density              = |X_i|
    connection_strength / normalized_connection_strength            = |X_i|

On the published 291 x 577 files each ratio is constant along its axis to
2.2e-16 (one ulp) and **every recovered value is an exact integer**. The
291 ipsilateral regions sum to 209,134 voxels = 209.1 mm^3 at the model's
100 um grid, the right order for one hemisphere of mouse grey matter.
That is the same fact derived four ways, and it is also an independent
test of P3.1's `units` strings: if Allen ever re-published
`connection_density.csv.gz` with the source size divided out instead of
the target size, `volumes.py`'s constancy check fails loudly rather than
producing a plausible wrong network. No new source, no new checksum, and
no new number entered the project to get this.

### Resolutions, and where the roadmap's "43" actually comes from

    "regions"          the dataset's own set: 291 (Knox) or 213 (Oh)
    "subdivisions"     one CCF level below the divisions: 60 for both
    "major_divisions"  the twelve Knox et al. published as their own mask

The roadmap asks for "major divisions -> ~43 regions -> ~213 regions".
The 213 is Oh et al. 2014's region count and is reproduced exactly. **The
43 is not a whole-brain parcellation.** It is the number of isocortical
areas in CCFv3 - "43 isocortical areas and their layers, 314 subcortical
grey matter structures, 81 fibre tracts and 8 ventricular structures"
(Wang Q, Ding SL, Li Y, et al. "The Allen Mouse Brain Common Coordinate
Framework: A 3D Reference Atlas." Cell. 2020;181(4):936-953.
doi:10.1016/j.cell.2020.04.007) - and it appears in this data as a
checkable fact rather than as a resolution: exactly 43 of Knox's 291
regions lie in the isocortex, which a test asserts.

Every natural whole-brain cut of the CCF was computed before concluding
that: nearest-ancestor at tree depth <=4/5/6 gives 16/70/212 groups, the
CCF's own `st_level` 6 gives 38 but splits VIS into one node while
lumping seventeen frontal areas into "Isocortex", and one literal level
below the twelve divisions gives **60** for both datasets, uniformly (MO,
SS, VIS, ..., DORsm, DORpm, MBmot, PALd, ...). 60 is the honest answer,
so `"subdivisions"` is 60 and the module reports its own count rather
than claiming a number it does not have.

### Sign is applied before aggregation, and why that is not cosmetic

Signing after aggregation would be a different and worse model. "Midbrain"
has no sign of its own, but SNr inside it does, so a coarse edge out of MB
is the signed sum of its members' edges - which today means SNr's
GABAergic output and nothing else, and means something richer the moment a
rule is added for another midbrain nucleus. Doing it in this order means
the twelve-node matrix has
net-excitatory and net-inhibitory entries that were *computed* from 291
regions rather than assigned by hand, which is the difference between a
coarse model and a cartoon. It also means a coarse node's own
`SignAssignment` (reported as `node_assignments`) is informational only -
no edge was signed with it - and the report says so.

### Two policies that would otherwise be silent defaults

**Unsigned sources are excluded, loudly.** `unsigned_sources="exclude"`
zeroes those columns and the report records which regions and what
fraction of tracer weight left with them. `"raise"` is available for a
build that must be complete. There is deliberately no option that assigns
them a sign.

**The self-connection diagonal is dropped within a region and kept
between regions.** Both matrices have large non-zero diagonals - the
largest entry in 109 of Knox's 291 rows - which for a bulk tracer is
mostly local intra-regional collaterals, and a Wilson-Cowan node already
carries its local recurrence in its gain and threshold. So the fine
diagonal is zeroed *before* aggregation, and the coarse diagonal that
survives (MOp -> MOs landing on the isocortex diagonal) is genuine
inter-areal connectivity. `"keep"` and `"drop"` are both available; the
default is stated in the report.

### The check that says the rule is doing anatomy

Nothing in this task was tuned against a circuit, so the following is a
result rather than a construction: reading the signed 291-region matrix
off, the D2 indirect pathway this codebase already models by hand comes
out with textbook polarity at every step -

    MOp -> CP  +3.06     corticostriatal, glutamatergic
    CP  -> GPe -12.56    striatopallidal, GABAergic
    GPe -> STN -2.38     pallidosubthalamic, GABAergic
    STN -> GPi +0.32     subthalamic, glutamatergic
    GPi -> MD  -0.11     pallidothalamic, GABAergic
    SNr -> MD  -0.27     nigrothalamic, GABAergic
    MD  -> MOp +0.04     thalamocortical, glutamatergic

and so does the GABA-A scenario's amygdala circuit (BLA -> CEA +1.65,
CEA -> PAG -0.04). Nine signs, all correct, from nineteen rules and a
tracer matrix that contains no signs at all. A test asserts all nine, and
it is also the first evidence that P3.3 (porting the hand-built scenarios
onto connectome regions) has a chance of working.

### The finding this task hands to P3.4

`weight_scale` defaults to 1.0, meaning the published densities enter the
matrix unrescaled, and that is **not a usable coupling**. Individual
entries are small (median non-zero |w| = 2.2e-3 at region resolution) but
the row sums are not: the signed input weight per node runs from -17.6 to
+32.5 with a median of +7.4. With the illustrative `gain=4.0` that pins
every node at the top of its sigmoid. This is the GPi-saturation lesson
from the D2 scenario reappearing at 291 nodes, and it is exactly what
P3.4 exists to fix. A test asserts the row-sum range so the claim cannot
rot, and `build.py`'s docstring says in as many words that a network built
with all defaults is not ready to simulate.

The reason the builder does not simply pick a scale is P2.3: coupling gain
and connection weight are degenerate against behavioural data - only their
product is identified - so choosing a scale here would be choosing a
calibration, and it would then propagate silently into every result
labelled "connectome-based". P2.3's relief still stands, but it is
narrower than it looked: the connectome fixes the *relative* weights
within `W` and leaves one global scale to be identified.

### What was deliberately not built

**Both hemispheres.** Every network here is the ipsilateral square block.
Both datasets inject tracer into one hemisphere only, so a bilateral
network needs a left-right symmetry assumption *and* a decision about the
five midline structures (RM, RO, RPA, OV, EW) that have an ipsilateral
column and no contralateral one - duplicating a midline nucleus across
hemispheres is anatomically wrong, and dropping it removes real regions
from a "whole-brain" network. Neither choice has data behind it in these
files. The contralateral block is loaded and untouched, so a later task
can spend it deliberately.

**Any node parameter.** `tau=1.0`, `gain=4.0`, `threshold=0.0`,
`baseline_input=0.0` (`build.DEFAULT_NODE`) are the hand-built scenarios'
illustrative values, are labelled ILLUSTRATIVE in the code and in every
`BuildReport.summary()`, and no mouse population time constant or
transfer-function slope was sourced for any region. P3.4 owns
`baseline_input`.

### Effect on the existing codebase

- **`CircuitNetwork` and `CircuitNode` are unchanged** - no method, no
  attribute, no keyword. The builder constructs the existing classes and
  returns a plain `CircuitNetwork` inside a `ConnectomeCircuit` wrapper
  that carries the provenance. A test pins both constructor signatures and
  `rhs`'s, which is the acceptance criterion about soma-twin's import path
  turned into something that fails if anyone moves it.
- **No existing parameter, scenario or calibration moved.** Nothing in
  `engine.py`, `scenarios.py` or `calibration/` imports this module yet;
  P3.3 is the task that connects them.
- **The fixtures grew**: two more Knox CSVs (strength and density, needed
  for the voxel-count recovery and the aggregation) and seven more
  ontology structures, so that `SignRules`' acronym-vs-id cross-check runs
  against committed data. One existing test asserted the fixture ontology
  has 48 structures; it now asserts 55, with the reason in the test. That
  is a fact about the fixture changing because the fixture legitimately
  grew, not an expectation weakened to get to green.
- **45 new tests**, 40 of them offline off the fixtures and 5 skipped
  without the gitignored cache. The three that carry the most weight are
  the one that rebuilds Allen's own published `connection_density` file
  from the raw strength matrix and the recovered voxel counts (exactly,
  `array_equal`), the one that asserts MOp->CP and CP->MOp separately with
  opposite signs (which a transpose or a sign inversion cannot pass), and
  the one that pins the `CircuitNetwork` API.

## Phase 3: porting the hand-built scenarios onto connectome regions (P3.3)

### The task, and the answer

P3.2 ended with a signed 291-region matrix and one piece of evidence that
this task might work: nine textbook basal-ganglia and amygdala signs fell
out of the tracer data with no tuning. P3.3 is where the scenarios
themselves are re-wired from that matrix and the qualitative results are
re-run.

Three of the four ported scenarios reproduce all three results - agonism,
inert antagonism alone, antagonist reversal - with published weights, no
retuning, and no rescaling. **The D2 basal-ganglia indirect pathway does
not**, and most of this section is the diagnosis, because the reason is a
specific, quantified disagreement between the textbook box-and-arrow
diagram the scenario encodes and what the mouse connectome actually
contains.

The new module is `connectome_scenarios.py`, at the top level rather than
inside `connectome/`, because it depends on both the data layer and
`scenarios.py` and the arrow has to point that way. **Nothing in
`scenarios.py`, `network.py`, `circuit.py` or `engine.py` was touched.**
Every hand-built network is still there, is still what
`tests/test_d2_scenario.py` and its siblings exercise, and is what the port
reads *from*: a `ScenarioPort` holds the hand-built factory, and a ported
network is that network with its connectivity replaced entry by entry.

### The thing that makes this a design problem rather than a lookup

The hand-built scenarios do not have one node per brain region. They have
one node per *population*, and every one of the four scenarios has at least
one node the CCF cannot address on its own. Three of them place a pair of
populations inside a single region:

    VTA_GABA / VTA_DA             both are VTA   (CCF 749)
    PAG_Interneuron / PAG_Output  both are PAG   (CCF 795)
    GABA_Interneuron / BLA        both are BLA   (CCF 295)

and the fourth, D2, names a cell type instead:

    Striatum_D2_MSN               one of two intermingled cell types in CP

A region-level tracer matrix has one row for each of those regions, and
P3.2 zeroes the within-region diagonal deliberately (for a bulk tracer it
is mostly local collaterals, and a Wilson-Cowan node already carries its
local recurrence in its gain). So the single most important edge in each of
those three scenarios - the GABAergic interneuron that inhibits the
projection population next to it, which is where every one of these drugs
acts - is an edge the connectome is *structurally* silent about. It is not
missing data; it is the wrong resolution for the question.

The port therefore refuses to produce a bare matrix. Every entry carries a
provenance, and there are six:

    connectome                    the published density x the source's sign
    intra_regional                same region both ends; hand-built weight
                                  retained, still ILLUSTRATIVE
    neuromodulatory_illustrative  source region carries no sign; hand-built
                                  weight retained, still ILLUSTRATIVE
    unsigned_source               same, with the weight dropped instead
    local_interneuron_source      a local population has no long-range axon
    restricted                    a cell-type node's declared target set

`PortReport.summary()` prints the illustrative edges, the edges where the
published data *contradicts* the hand-built guess, and the published weights
the map deliberately declined to use. Those three lists are the honest
content of a "connectome-based" scenario, and a test asserts each of them.

### Two approximations, made rather than hidden

**Long-range input is shared by every population in the target region.**
`PL -> VTA` is one published number and both `VTA_DA` and `VTA_GABA` sit in
those voxels, so both receive it. The alternative - giving the whole
projection to the principal population - would assert that cortical
afferents avoid local interneurons, which is the opposite of the
feedforward-inhibition motif the scenario's own hand-drawn
`PFC -> VTA_GABA` edge exists to represent.

**Long-range output leaves only from the projection population.** This one
is not cosmetic: without it the single published `BLA -> CEA` projection
would be delivered twice, once from each of the two BLA-resident nodes -
which is why `port_scenario` refuses a map with two projection populations
in one region rather than summing them. The
report names the published number it declined (`CeA <- GABA_Interneuron,
published +1.6489`) rather than showing a zero and leaving the reader to
work out whether that zero was in the data or was a decision. In the
mu-opioid scenario the same rule costs nothing at all, because VTA and PAG
are unsigned sources and their published outgoing weight is already zero.

**A ported scenario is a sub-network, not a small brain.** Every path
through the other ~285 regions has been deleted, exactly as
`connectome/fixtures/` warns about slicing. What is being tested is whether
a scenario's qualitative result survives having its *weights* replaced by
published ones - not whether four regions are a model of anything.

### The one Restriction, and why it needs a citation

`Striatum_D2_MSN` is the only node in this module that names a cell type
rather than a population-in-a-region, and the caudoputamen row over-claims
for it badly. In Knox et al. 2019:

    CP -> GPe   -12.56    indirect pathway (D2/A2a spiny neurons)
    CP -> GPi   -12.21    direct pathway   (D1 spiny neurons)
    CP -> SNr   -16.34    direct pathway   (D1 spiny neurons)

The direct-pathway targets carry *more* weight than the indirect one, and
the two cell types are intermingled in the same voxels, so a region-level
matrix cannot separate them at all. Without a restriction the node is not
the cell type the scenario names; with one, the port is making an
anatomical claim and has to say on whose authority. `Restriction` therefore
refuses to be constructed without a citation, the same way
`connectome/neurotransmitter.py`'s rules do: here Gerfen & Surmeier 2011
(Annu Rev Neurosci 34:441-466, doi:10.1146/annurev-neuro-061010-113641) for
D1 -> SNr/GPi and D2/A2a -> GPe in mouse and rat, and Albin, Young & Penney
1989 (Trends Neurosci 12:366-375) for the direct/indirect formulation the
scenario encodes. The report states the cost: over the full 291-region
matrix the restriction discards **90.5%** of the caudoputamen's published
outgoing weight.

What would replace the declaration with a measurement is Cre-line-specific
tracer data - Allen has Drd1a-Cre and Adora2a-Cre experiments - which is a
P3.1-shaped ingestion task and is named in the module rather than
approximated here, the same way P3.2 named [Yao2023] as the successor to
its hand-curated transmitter table.

### No new constant: why weight_scale stays at P3.2's 1.0

P3.2 refused to pick a `weight_scale` (P2.3: coupling gain and connection
weight are degenerate against behavioural data, so choosing a scale is
choosing a calibration) and reported that at 291 nodes the unrescaled
densities drive every node into saturation - signed row sums from -17.6 to
+32.5 against an illustrative `gain=4.0`.

At scenario scale that does not happen, and the measurement says so.
Mean per-node total input weight, `mean_i sum_j |W_ij|`:

    scenario                 hand-built   ported   ratio
    mu_opioid_reward            1.850      2.015    1.09
    mu_opioid_multi_region      1.414      1.690    1.20
    d2_indirect                 1.600      3.668    2.29
    gaba_a_amygdala             1.500      1.562    1.04

So the port's default is P3.2's default - the published number, unrescaled
- and **P3.3 introduces no new numeric constant at all**. The whole-brain
saturation P3.2 reported is a property of summing 291 rows, not of the
densities themselves. `weight_scale` remains a keyword and the tests assert
it rescales the published edges and only those, because "the result
survives at exactly one scale" and "the result survives" are different
claims.

### The baseline rule, and where it came from

The default port changes the connectivity and nothing else: `tau`, `gain`,
`threshold` and `baseline_input` come over unchanged from the hand-built
node, so any change in a result is attributable to the connectome alone.
That is the right default and it has one known weakness, which the D2
scenario runs straight into: `baseline_input` in the hand-built scenarios
was chosen *against the hand-built weights*.

`rest_balanced_baselines()` restates that intent as arithmetic. A
Wilson-Cowan node is at equilibrium when `r_i = sigmoid(net_i)` and
`sigmoid(x) = 0.5` exactly at `x = threshold`, so

    baseline_i = threshold_i - sum_j W[i, j] * 0.5

makes `r = 0.5` an exact fixed point for every node, in closed form, with
nothing fitted. The reason this is a legitimate move rather than a new
modelling decision is that it is not new: applied to the hand-built
basal-ganglia network's own weights it returns that network's own
`baseline_input` values, `(0, 1, 1, -1, 1)`, exactly. `scenarios.py` says
those values "are chosen so each node sits near the sensitive
(non-saturating) part of its sigmoid at rest"; solving that sentence gives
back the five numbers. A test asserts it. P3.4 owns the general version of
this problem; P3.3 uses it only as a diagnostic, to separate "the result
died because a node saturated" from "the result died because the anatomy
disagrees".

### What survived

Agonism, inert antagonism alone and antagonist reversal, ported, under both
baseline policies, for `mu_opioid_reward`, `mu_opioid_multi_region` and
`gaba_a_amygdala`. Effect sizes move in both directions - two of the four
grow, two shrink - and none inverts; with hand-built baselines the ported
agonist effects are +0.144 (incentive salience, hand-built +0.235), +0.806
(analgesia, +0.607), +0.620 (arousal, +0.500) and +0.341 (anxiety, +0.499),
and every antagonist-alone deviation is at
solver noise: the largest across all four scenarios and both baseline
policies is 3.3e-16.

The multi-region scenario's actual point survives too: one occupancy trace
still drives three readouts in two directions at once (analgesia and
incentive salience up, arousal down) on published wiring.

Two hand-drawn arrows in the reward circuit turn out to be wrong, and the
port reports both rather than silently overwriting them:

    PFC -| VTA_DA   hand -0.800   published +0.5008   sign flip
    NAc ->  PFC     hand +1.500   published -0.0098   sign flip, 150x weaker

The first is the more instructive. `PFC -| VTA_DA = -0.8` was a shortcut for
disynaptic top-down inhibition; the corticotegmental projection is
glutamatergic, and in the ported network the inhibition is represented
explicitly instead, by the same cortical input also exciting `VTA_GABA`.
The second is accumbal output being GABAergic, which the hand-drawn "NAc
drives PFC" had backwards, and which the tracer data also finds two orders
of magnitude weaker than the guess. Neither reverses the scenario's result.

### What did not survive: the D2 indirect pathway

Hand-built, a D2 agonist raises the locomotor readout by +0.110. Ported, the
effect is zero to four decimal places under either baseline policy - and
comparing the two trajectories pointwise rather than by their maxima says
something sharper. Hand-built, the agonist run is never below its own
drug-free run and rises 0.472 above it. Ported, it is never *above* its
drug-free run at any time, and goes 0.311 (hand-built baselines) or 0.326
(rebalanced) below it. **The agonist does something to the ported circuit;
what it does is the opposite of what the scenario claims.** There are two
independent causes, and the second is the interesting one.

**Cause 1: a baseline chosen against one W does not fit another one.** GPe's
`baseline_input = 1.0` was set against a hand-built striatopallidal weight
of -2.0. The published striatopallidal density is -12.56, so at rest GPe
receives about -9, and `gain=4.0` pins it at zero. Nothing downstream of it
can respond to anything. This is the GPi-saturation lesson of `scenarios.py`
reappearing at five nodes, it is exactly what P3.4 exists to fix, and it is
not a statement about the connectome.

**Cause 2, once cause 1 is removed: the pathway inverts.** With baselines
rebalanced, every node rests at exactly 0.5, nothing is saturated, and the
chain runs - correctly, at every step but the last. Under a D2 agonist:

    Striatum_D2_MSN   0.500 -> 0.007   suppressed by the Gi-coupled receptor
    GPe               0.500 -> 1.000   disinhibited, as the diagram says
    STN               0.500 -> 0.014   inhibited by GPe, as the diagram says
    GPi               0.500 -> 0.104   loses STN drive, as the diagram says
    Thalamus          0.500 -> 0.174   FALLS

The locomotor readout is the thalamus, so the scenario's result is not
merely damped, it is reversed.

**The single number responsible is the pallidothalamic projection.** Knox et
al. 2019 has GPe projecting to the thalamus directly, and much more densely
than GPi does:

    GPe -> MD   -0.859     GPi -> MD   -0.113     ratio 7.58
    GPe -> VAL  -0.473     GPi -> VAL  -0.054     ratio 8.82
    GPe -> VM   -0.336     GPi -> VM   -0.046     ratio 7.27
    GPe -> CM   -0.750     GPi -> CM   -0.097     ratio 7.73
    GPe -> PF   -1.598     GPi -> PF   -0.181     ratio 8.82

The textbook indirect-pathway diagram routes GPe's entire influence on the
thalamus through STN and GPi and gives it no thalamic projection of its
own. The mouse connectome gives it one seven to nine times denser than the
GPi projection the diagram is built on, at every candidate motor or limbic
thalamic nucleus. So when GPe rises, the thalamus is inhibited harder than
the falling GPi disinhibits it, and the sign of the whole pathway follows
GPe rather than GPi.

Three checks make that a diagnosis rather than a guess, and all three are
tests:

* Removing only the GPe -> thalamus edge (as an explicitly labelled
  counterfactual `Restriction`, not as a shipped default) restores the
  result: +0.039.
* Remapping the scenario's `GPi` node onto SNr does *not* restore it. That
  matters, because "the hand-built diagram uses the primate output nucleus;
  in rodents SNr is the main basal-ganglia output" is the obvious first
  objection, and it is not the cause.
* The failure is identical with VAL, VM, CM or PF in place of MD, so it is
  not an artefact of MD being the thalamic nucleus that happens to be in the
  test fixture.

**Is the GPe -> thalamus edge real, or an artefact?** It is not obviously an
artefact. Mastro KJ, Bouchard RS, Holt HAK, Gittis AH. "Transgenic mouse
lines subdivide external segment of the globus pallidus (GPe) neurons and
reveal distinct GPe output pathways." J Neurosci. 2014;34(6):2087-2099.
doi:10.1523/JNEUROSCI.4646-13.2014 reports mouse GPe axons in the reticular
thalamic nucleus from both GPe subpopulations ("axonal projections from both
populations of neurons were found in the RT with no significant difference
in density") and dense PV-GPe projections to the parafascicular nucleus.
What these files cannot settle is whether the *MD-specific* density Knox
reports is a genuine GPe -> MD projection or signal shared with the
neighbouring reticular nucleus in a voxel-scale interpolated model. That
would need single-neuron reconstructions, not a regional matrix, and it is
recorded here as the open question rather than resolved by assertion.

The honest summary is therefore not "the connectome broke the D2 scenario".
It is that **the hand-built D2 scenario is a drawing of the Albin-DeLong
diagram, and the diagram is a statement about which projections matter, not
a list of which projections exist.** Replacing its weights with measured
ones replaces that editorial judgment with a measurement, and the
measurement disagrees. Fixing it properly needs the cell-type-resolved
striatal output the `Restriction` currently stands in for, and a decision
about the pallidothalamic projection that has anatomy behind it rather than
a diagram - neither of which is a number this task could invent.

### Why the whole regression suite runs offline, and the check that says so

A ported network only ever reads region-pair entries of the matrix, and a
connection density is a per-region-pair quantity, so slicing the published
file down to the fifteen fixture regions cannot change any of them. That is
a claim, so it is checked rather than believed:
`test_fixture_and_full_connectome_give_the_same_ported_networks` builds
every port off both sources and asserts `np.array_equal` on the
connectivity. It passes exactly, for all four scenarios. Nine tests in all
need the gitignored cache and skip without it; the other eight are extras
(the thalamic and prefrontal robustness sweeps, the 90.5% restriction cost,
the report summary), and every acceptance criterion is covered offline.

### What this task changes, and what it does not

- **`scenarios.py` is untouched.** Every hand-built network, coupling,
  ligand set, readout and calibrated constant is where it was, and a test
  asserts that porting does not mutate them. The acceptance criterion is
  "hand-built networks retained for comparison, not deleted"; they are also
  what the comparison is *against*, so deleting them would have removed the
  experiment.
- **No new numeric constant enters the project.** `weight_scale` stays 1.0,
  the node parameters come from `scenarios.py`, the weights come from Knox
  et al. 2019, and the one derived quantity (`rest_balanced_baselines`) is
  closed-form arithmetic that reproduces an existing scenario's own numbers.
  The single reporting threshold in the module, `CONTRADICTION_FLOOR = 0.01`,
  decides what gets *printed* and nothing else.
- **No label moved.** The intra-regional and neuromodulatory edges are still
  ILLUSTRATIVE, and are now labelled per edge in a report rather than only in
  a module docstring.
- **70 new tests**, 61 of them offline off the fixtures and 9 skipped without
  the cache. The five that carry the most weight are the four D2 diagnosis
  tests and the fixture-equals-full-data one.

## Phase 3: whole-brain baseline stability (P3.4)

### The task, and the shape of the answer

This codebase learned the same lesson twice by hand. The D2 scenario's
first parameter draft put GPi's net input far outside `[-1, 1]` before
the gain multiplier, so GPi sat against its sigmoid's ceiling at rest and
"stopped responding to anything upstream of it". P3.3's port of that same
scenario onto published weights hit it again: GPe's hand-set
`baseline_input = 1.0` was chosen against a hand-drawn striatopallidal
weight of `-2.0`, met the published density of `-12.56`, and was pinned
at zero. Both times the diagnosis was made by printing five rates and
looking at them.

At 291 nodes that is not available, so P3.4 turns the lesson into three
things: a **measure** of saturation that is not a band anyone chose, a
**procedure** for assigning baseline drive that is closed-form and
therefore scales, and a **certificate** that says the procedure's answer
is not merely *a* fixed point but the only one, reachable from anywhere.
They live in one new module, `src/neurocascade/stability.py`, which
imports only `circuit.py` and `network.py` and works on any
`CircuitNetwork` - hand-built, ported or connectome-built - because the
failure it is about is not a property of the connectome.

Nothing in `network.py`, `circuit.py`, `engine.py`, `scenarios.py` or
`connectome/build.py` changed behaviourally. `connectome_scenarios.py`'s
`rest_balanced_baselines` now delegates to the general rule and returns
identical numbers, which a test asserts.

### The measure: sensitivity, not a rate band

The obvious check is "is every rate inside `[0.05, 0.95]`", and the
numbers in it would be invented. The non-invented version starts from
what "saturated" means: the node's output has stopped depending on its
input, which is a statement about a derivative. Linearizing

    tau_i dr_i/dt = -r_i + S_i(net_i),   net_i = sum_j W[i,j] r_j + b_i

gives `J = diag(1/tau) (-I + diag(S_i'(net_i)) W)`, and at a fixed point
(`r_i = S_i(net_i)`) the slope is the rate's own quadratic,
`S_i' = g_i r_i (1 - r_i)`. That factor is exactly the channel through
which everything upstream of node `i` reaches everything downstream of
it; if it is zero, the node's row of the Jacobian is dead and the node is
a constant. Dividing by its own maximum, `g_i/4` at `r_i = 1/2`, removes
the gain and leaves a dimensionless index:

    sensitivity_i = S_i'(net_i) / (g_i/4) = 4 r_i (1 - r_i)   in (0, 1]

so 1 is the most responsive a node can be and 0 is a rail. This is the
linearization's own coefficient written in units where 1 is the best
available, not a threshold on rate.

`SATURATION_FLOOR = 0.01` is the one reporting convention, and it is
labelled as such: nothing in the codebase is fitted against it and every
report carries the raw sensitivities so another floor can be applied
afterwards. It is 0.01 because two independent readings land there. A
node at `sensitivity = 0.01` transmits 1% of the input change an
optimally-placed node would, i.e. sits within 0.0025 of a rail; and
solving `4r(1-r) = 0.01` gives `|g (net - theta)| = 5.99`, so at the
illustrative `gain = 4.0` the flag fires at `|net - theta| = 1.50` -
just outside the `[-1, 1]` band the original GPi note in this document
already used to describe the failure it saw. A test asserts both
arithmetics, so the docstring cannot drift from the constant.

### The procedure, in four steps

**1. Choose the operating point.** `r*_i = 1/2` for every node. This is a
derivation and not a preference: `S_i'` is maximized at `S_i = 1/2`, so
`r* = 1/2` is the unique point at which every node is as far from
saturation as it can be. Per-node targets are supported and exact (step 2
below), and are where region-wise baseline firing rates would go; this
project has sourced none, so a heterogeneous `r*` today would mean
inventing 291 numbers.

**2. Solve for the drive, in closed form.** `r*` is an exact fixed point
iff `S_i(net_i) = r*_i`, i.e. `net_i = theta_i + logit(r*_i)/g_i`, so

    b_i = theta_i + logit(r*_i)/g_i - sum_j W[i,j] r*_j

At `r* = 1/2` the logit term vanishes and this is P3.3's
`rest_balanced_baselines`, whose credential stands unchanged: applied to
the hand-built basal-ganglia network's own weights it returns that
network's own `baseline_input` values `(0, 1, 1, -1, 1)` exactly. This is
one matrix-vector product; nothing is searched, iterated or tuned per
node, and that is the whole reason the procedure scales.

**3. Certify.** Being an exact fixed point is not being a stable one, and
this is where the task earns its keep. Two conditions:

    LOCAL   max Re eig(J) < 0,  J = diag(1/tau)(-I + diag(S') W)
            <=> r* is locally asymptotically stable

    GLOBAL  kappa = sigma_max(diag(g/4) W) < 1
            ==>  r* is the UNIQUE fixed point, and every trajectory from
                 every initial condition converges to it exponentially

The global condition is Kelly's contraction argument - Kelly DG,
"Stability in contractive nonlinear neural networks", IEEE Trans Biomed
Eng. 1990;37(3):231-242, doi:10.1109/10.52325, whose abstract states it
in the form this codebase needs: "if the map WF(x) is a contraction, the
system has a unique equilibrium which is globally asymptotically stable".
Kelly writes the current-form equation `mu x' = -x + p + WF(x)`; the
translation to this codebase's rate form is three lines and is written
out in `stability.py` rather than left as a reference, because the
Lyapunov function that makes it tau-independent (`V = (1/2) sum_i tau_i
u_i^2`) is the part that matters for a network whose time constants
differ per node. The bound uses `S_i' <= g_i/4` everywhere, which is why
`kappa` is evaluated at `r = 1/2` regardless of where the network sits.

The local criterion is deliberately not written as
`max Re eig(diag(S')W) < 1`. That is the same statement only when every
`tau` is equal; `J`'s eigenvalues are that matrix's rescaled row by row,
and a positive diagonal scaling can move the eigenvalues of a
non-symmetric matrix. Every network built in this project happens to have
uniform `tau` at connectome resolution and non-uniform `tau` in the
hand-built scenarios, so `spectral_abscissa` computes `eig(J)` and does
not take the shortcut.

**4. Verify by integration.** A certificate that is never checked against
the integrator is a certificate about a different model than the one this
package solves. `stability.assess` integrates from several spread,
seeded initial conditions (all-silent, all-saturated, the target, then
uniform draws), checks that they all land on the same point, that
`max |dr/dt|` there is at solver noise, and that every node clears the
floor.

### The one free number, and why it is dimensionless

Steps 1-4 say whether a given `(W, gain)` has a usable rest state. They
do not say what to do when it does not, and the whole-brain connectome
does not. P2.3 already established that a coupling gain and a connection
weight are degenerate against behavioural data - only their product is
identified - so the honest move is to name the product rather than to
pick a `weight_scale`. Define the **operating gain**

    rho = kappa(weight_scale * W) = weight_scale * kappa(W)

a single dimensionless number that is the network's loop gain at its most
sensitive operating point, and set `weight_scale = rho / kappa(W)`. That
is `stability.stabilize()`. `DEFAULT_OPERATING_GAIN = 0.9` is
ILLUSTRATIVE and labelled so in the module, in `StabilizedNetwork.
summary()` and in the committed results file; what it replaces is the
*pair* (`weight_scale`, `gain`) that P3.2 deliberately refused to choose,
with one number that has a meaning (`rho < 1` is the certificate,
`rho -> 1` is the edge of instability) and that leaves every relative
weight the connectome fixed exactly where it was. A test asserts that
stabilizing preserves every non-zero weight ratio to 1e-12 and touches no
`tau`, `gain` or `threshold`.

### The whole-brain result

Knox et al. 2019 connection density, signed per P3.2, one hemisphere.
The full run is `scripts/check_whole_brain_stability.py` and its output is
committed at `results/whole_brain_stability.txt`.

    resolution        nodes   kappa   critical weight_scale
    regions             291   44.70          0.02237
    subdivisions         60   35.15          0.02845
    major_divisions      12   30.57          0.03271

At region resolution, as `connectome/build.py` ships it
(`weight_scale=1.0`, `baseline_input=0.0`): **218 of 291 nodes are past
the saturation floor** - 210 against the ceiling and 8 against the floor,
the latter being the regions whose input is dominated by GABAergic
sources (GPe, GPi, SNr, SNc, PS and three cerebellar cortical lobules).
That is P3.2's warning measured rather than asserted.

Balancing the baselines and *not* rescaling is worse in an instructive
way: `r = 1/2` becomes an exact fixed point of the 291-node network
(`max |dr/dt|` there is 0 to machine precision) and it is a **repeller**.
Three initial conditions give two different saturated answers a full unit
apart, and 194 of 291 nodes end up past the floor. Step 2 without step 3
produces a rest state that exists and that the model never visits.

With the procedure applied at `rho = 0.9`
(`weight_scale = 0.0201324` at region resolution):

    fixed point residual max |dr/dt| = 1.6e-08
    same point from all 5 starts, spread = 2.0e-08
    local: max Re eig(J) = -0.678
    global: kappa = 0.9 < 1, unique and globally attracting
    saturation: min sensitivity 1.000, no node below the floor

and identically at `subdivisions` and `major_divisions`. That is the
acceptance criterion: a stable, unique, non-saturating fixed point on the
published whole-brain connectome, reached from every initial condition
tried, with a certificate saying there is no other one to find.

### The check a stabilized network has to survive, and does

A network can be made to pass every test above by shrinking every weight
until nothing propagates, so the last block of the results file asks
whether anything still does. The linear response of every region to a
unit drive step at the caudoputamen, `(I - A)^-1 diag(S')`:

    CP  +1.000   SNr -0.325   SNc -0.295   GPe -0.251   GPi -0.240   STN -0.150

166 of 291 regions move by more than 1e-3. The six largest responses are
the caudoputamen and its five basal-ganglia targets, every one of them
negative because CP's projection neurons are GABAergic - which is the
same "the rule is doing anatomy" check P3.2 made, now run through the
dynamics rather than off the matrix. A step at MOp instead reaches 193
regions and its largest responses are thalamic (PCN, VAL, PO) and
somatosensory (SSp-m), all positive. Both are asserted by tests.

### What the check says about the scenarios that already existed

Run over every hand-built network in `scenarios.py`, at the drug-free
baseline:

    basal_ganglia_indirect_pathway   min sensitivity 1.000   passes
    amygdala_anxiety_circuit         min sensitivity 0.420   passes
    reward_circuit                   VTA_DA at 4.9e-06       flagged
    multi_region_mu_opioid           + PAG_Output at 3.3e-04 flagged
    combined_mu_opioid_d2            VTA_DA                  flagged

The two networks whose docstrings make the claim satisfy it. The
mu-opioid circuits do not, and that is a real property of them rather
than a new bug: they are built as disinhibition relays, so the node the
drug recruits (`VTA_DA`, `PAG_Output`) is *meant* to be silent at rest.

This is the point at which a check that only looks at the resting state
would be pedantic rather than useful, so P3.4 draws the distinction
explicitly and tests it. Evaluate the same check at the drive the
scenario's own agonist approaches (baseline plus each receptor system's
coupling at activation 1):

    at rest        VTA_DA, PAG_Output              against the floor
    with drug      VTA_GABA, PAG_Interneuron, LC   against the floor
    in both        (none)

The saturated set is disjoint between the two states. Every node that is
silent at rest is one the drug recruits, and the nodes that go to a rail
under the drug are the drug's own Gi-coupled targets - which is precisely
what a disinhibition relay looks like from the outside. Contrast P3.3's
ported D2 network, where `GPe` is against the rail *with and without* the
drug: that is the intersection being non-empty, and it is what "it
stopped responding to anything upstream" actually means. So the criterion
worth carrying forward is not "no node may be saturated at rest" but
**"no node may be saturated in every state the drug can produce"**, and
`stability.assess` supports it by accepting a drive vector.

No parameter in `scenarios.py` was changed on the strength of any of
this. The flagged nodes are recorded in tests as facts about the shipped
scenarios, with the drugged-state test alongside them saying why those
scenarios still work.

### A correction to P3.3, found by the certificate

P3.3's D2 section says: "With baselines rebalanced, every node rests at
exactly 0.5, nothing is saturated, and the chain runs". The first half is
true and the conclusion drawn from it needed one more step, which is
exactly the step P3.4 exists to add.

`r = 1/2` is an exact fixed point of the rebalanced ported D2 network -
`rhs` there is identically zero - and it is **unstable**: the Jacobian's
spectral abscissa at that point is `+0.733`. The network is bistable, and
neither branch is `r = 1/2`:

    from r = 0   [0.3345, 0.9996, 0.0135, 0.1044, 0.1737]
    from r = 1   [0.6655, 0.0004, 0.9865, 0.8956, 0.8263]

The cause is anatomical and not numerical. The published striatopallidal
density is `-12.56` and the pallidostriatal return is `-0.22`; two
inhibitory links in series are positive feedback, and at `gain = 4.0`
that loop is strong enough to make CP and GPe a latch. P3.3 therefore
read its rebalanced D2 trajectories off a repeller, with `rates0` set to
exactly `0.5` - which is why the drug-free run stayed at `0.5` (an
integration started exactly on a fixed point does not move) while the
drugged run fell off it.

**The finding survives the correction, and is now better supported.** Put
the same ported network through the P3.4 procedure at `rho = 0.9`: the
bistability is gone, `r = 1/2` is the unique globally attracting rest
state, no node is saturated, and under a D2 agonist the chain runs with
textbook signs at every step - striatum down, GPe up, STN down, GPi down
- and the thalamus falls anyway, never rising above its drug-free value
at any time (peak deviation `-0.020`). So the pallidothalamic diagnosis
P3.3 reached is a property of the anatomy rather than an artefact of the
operating point it was measured at. Both halves are tests.

P3.3's own tests are unchanged and still pass: their assertions are about
trajectories from `rates0 = 0.5`, and those trajectories are what they
were. What changed is the docstring of the one test whose prose implied
that `0.5` was a rest state the network would find, which now points at
the two P3.4 tests that say what it actually is.

### The three surviving ported scenarios, re-run where the check passes

Stabilizing multiplies every published weight by one scalar, so it cannot
change a relative weight; what it can change is an effect size, and it
does. At `rho = 0.9`, all three of P3.3's surviving results keep their
sign and their mechanism:

    mu_opioid_reward        incentive salience   +0.216
    mu_opioid_multi_region  analgesia +0.330  arousal +0.493  salience +0.214
    gaba_a_amygdala         anxiety index        +0.103

and they shrink monotonically with `rho` (at `rho = 0.2`: +0.035, +0.088,
+0.005), which is what a loop gain should do and is another reason `rho`
rather than `weight_scale` is the parameter worth naming. The sign is the
part that has to survive, and a test asserts it for every readout.

### What was deliberately not done

**No region-wise baseline firing rates were sourced: searched, and the
one good candidate is structurally unusable.** The uniform `r* = 1/2` is
a derivation from the sigmoid, not a claim that every mouse brain region
fires at the same rate, and `balanced_baselines` accepts a per-node
target vector precisely so that real data can be spent on it.

What was searched for: a whole-brain, region-wise, awake-mouse baseline
firing-rate table aligned to the Allen CCF parcellation. What exists:

- **IBL Brain-Wide Map** (International Brain Laboratory. "A brain-wide
  map of neural activity during complex behaviour." Nature. 2025.
  doi:10.1038/s41586-025-09235-0), 621,733 neurons, 699 Neuropixels
  insertions, **279 brain areas**, mouse, CCF-registered. This is the
  right shape - it is the only dataset that covers a comparable fraction
  of the brain to Knox et al.'s 291 regions.
- **Allen Brain Observatory Neuropixels** (de Vries, Siegle et al.),
  awake mouse, spontaneous *and* evoked, but visual cortex, hippocampus
  and visual thalamus only - roughly a dozen of the 291 regions.

Neither is usable here, and the reason is not coverage. A Wilson-Cowan
`r_i` is **the fraction of the population that is active**, dimensionless
and bounded in `[0, 1]`, not a firing rate in spikes/s. Converting a
published median firing rate into an `r*_i` requires a per-region
*maximum* firing rate to divide by - a second table that does not exist
for 291 CCF regions in any species - and the quotient would then be read
through `gain = 4.0` and `threshold = 0.0`, both of which are themselves
ILLUSTRATIVE. Substituting spikes/s for an activation fraction would be
inventing the denominator, so the honest position is the derived uniform
`r* = 1/2` plus the parameter hook, and the record of what would have to
exist for that to change. It is the same class of gap as Track E's, and
it belongs to the same later phase that sources per-region node
parameters at all.

**`rho` was not calibrated.** It is one ILLUSTRATIVE number, and by
P2.3's degeneracy argument it is not identifiable from anything this
project currently has. What P3.4 changes is that the free parameter is
now *one* dimensionless quantity with a stated meaning and a stated
critical value, instead of two coupled ones with neither.

**No sparse or iterative linear algebra.** `kappa` is a dense SVD, which
is fine at 291 nodes (the 291-node `assess` is dominated by the Python
loop inside `CircuitNetwork.rhs`, not by the algebra) and is not fine at
the few thousand states Phase 9 is about. `stability.py` is written so
that the two places it would have to change are `contraction_number` and
`jacobian`.

**Both hemispheres, still.** Every network here is the ipsilateral block,
for P3.2's reasons.

### Effect on the existing codebase

- **One new module** (`stability.py`), one new script
  (`scripts/check_whole_brain_stability.py`), one committed results file
  (`results/whole_brain_stability.txt`).
- **No behaviour changed anywhere else.**
  `connectome_scenarios.rest_balanced_baselines` delegates to
  `stability.balanced_baselines` and a test asserts the numbers are
  identical for all four ports; `connectome/build.py` gained documentation
  only.
- **No numeric constant entered the project except one labelled
  ILLUSTRATIVE** (`DEFAULT_OPERATING_GAIN = 0.9`) and one labelled a
  reporting convention (`SATURATION_FLOOR = 0.01`, whose arithmetic is
  asserted by a test). `MAX_SENSITIVITY_RATE = 0.5` is the argmax of the
  logistic slope, and the integration tolerances are numerical settings
  that are arguments everywhere they are used.
- **No label moved.** Every ILLUSTRATIVE label in `build.py` and
  `scenarios.py` is where it was.
- **50 new tests**, 43 of them offline off the fifteen-region fixtures and
  7 skipped without the gitignored cache. The ones that carry the most
  weight are the whole-brain assessment at all three resolutions, the
  reconstruction of the original GPi draft (the check names GPi without
  being told to look at it), the recruited-versus-dead pair, and the two
  that state and then resolve the correction to P3.3.

## Phase 4: ingesting Allen receptor-gene expression (P4.1)

### What this task is, and the two things it must not become

P4.1 is an ingestion task, the P3.1 shape applied to a second body of
external data. It pins one Allen Brain Atlas query per receptor gene,
verifies the bytes, and hands back a (gene x region) expression-energy
matrix whose columns are the connectome's regions in the connectome's
order. It does **not** convert expression to a receptor density, and it does
not pick which of a gene's several ISH experiments is "the" map. Those are
P4.2 and they are the two places this task could have quietly become a
modelling task instead of an ingestion one.

The new package is `src/neurocascade/expression/`:

    genes.py      the 43 receptor/transporter genes and what each encodes
    sources.py    the manifest: query URLs, digests, citations, units, and
                  the searched-and-empty record
    fetch.py      download, verify, cache; thin over `datafetch.Fetcher`
    unionize.py   readers for the two published shapes
    atlas.py      ExpressionAtlas / RegionalExpression
    fixtures/     real data, eight genes and 55 structures of it, committed

plus `scripts/fetch_expression.py` (the CLI),
`scripts/build_expression_manifest.py` (measures the digests),
`scripts/make_expression_fixture.py` (regenerates the fixtures from the
verified cache) and `scripts/report_expression_coverage.py`, which writes
the committed `results/expression_coverage.txt`. That file exists so the
two things this task must not hide - the gene with no data, and the regions
the ISH data cannot reach - are readable in a clean checkout without the
gitignored caches and without running anything.

### One fetcher, two manifests

P3.1's `connectome/fetch.py` and this package's `expression/fetch.py` are
now both thin bindings of `neurocascade/datafetch.py`, parameterised by a
manifest module and a cache directory. The alternative was copying 150
lines and editing two names, which would have left two implementations of
"what happens on a checksum mismatch" free to drift apart. The behaviour
that must not drift is exactly the strict part: reject, keep the bytes as
`.rejected`, and say in the error message that editing the expected digest
is not one of the legitimate responses. `connectome.fetch`'s public
function names and signatures are unchanged and its 47 tests are untouched.

The caches are deliberately separate (`data/connectome/`,
`data/expression/`) so re-fetching the more volatile ISH half cannot
disturb the connectome, and so `sha256sum -c` can be run against either
directory alone.

### The gene list, and why a gene table rather than a list of strings

The task named "Oprm1, Oprk1, Oprd1, Drd1, Drd2, Gabra1-5, Gabrb, Htr1a,
Htr2a, Chrm1-5, Chrna4/7, Adra1/2, Adrb1/2, Grin1/2a/2b, Gria1-4, Hrh1,
Cnr1, Adora2a, Slc6a2/3/4". Expanded to actual mouse symbols that is 43
genes: `Gabrb` becomes Gabrb1/2/3, `Adra1/2` becomes Adra1a/1b/1d and
Adra2a/2b/2c, `Adrb1/2` becomes Adrb1 and Adrb2. The expansion is written
out as a set in the tests so a later edit cannot silently drop a subunit.

Three things made a table (symbol, Entrez id, protein, receptor system,
consuming phase) preferable to a list of symbols:

- **A receptor is not always a gene.** There is no `Gabaa` to look up;
  GABA-A is a pentamer and this project's benzodiazepine site exists only
  on alpha1/2/3/5-containing receptors. `Gabra4` is ingested precisely so
  P4.2 can *exclude* it from a diazepam-site map deliberately rather than
  pool it in by accident; the table says so and a test asserts the flag.
- **A gene is not always the receptor's name.** DAT is `Slc6a3`, SERT
  `Slc6a4`, NET `Slc6a2`.
- **Symbols move.** The mouse D1 gene was `Drd1a` and is now `Drd1`. A
  query for the old symbol returns *zero datasets*, not an error, so a
  symbol drift would have produced an empty map indistinguishable from
  "not expressed". Every entry therefore carries its NCBI Entrez id, and
  `build_expression_manifest.py` checks all 43 against Allen's own gene
  table before it measures a single digest. All 43 resolved.

### Which quantity, and how the definition was verified

Allen's ISH pipeline segments each image into expressing and non-expressing
pixels, grids the result into 200 um voxels registered to the adult (P56)
reference space, and unionizes those voxels over every CCF structure. The
structure-level quantity the roadmap wants is

    expression energy = (sum of intensity of expressing pixels in A)
                        / (sum of all pixels in A)

which is Allen's own documented definition. That was not taken on trust.
Over all 843 graph-1 unionizes of the coronal Oprm1 experiment,
`sum_expressing_pixel_intensity / sum_pixels` reproduces the published
`expression_energy` to a maximum relative error of 1.1e-5, and
`sum_expressing_pixels / sum_pixels` reproduces `expression_density` to
9.6e-6: consistent with the API's six significant figures and nothing else.
A test re-runs both identities on the committed fixture rows.

**Expression energy is not a receptor density**, and nothing in this
package pretends otherwise. It is transcript abundance in arbitrary units,
not even comparable in absolute terms between genes, because probe
efficiency and segmentation threshold differ per experiment. The `units`
string on every manifest entry says this in full. P4.2 owns the conversion
and owns stating its proportionality assumption as an assumption.

### The provenance problem: a live API, and what was searched for instead

Every file in this manifest is a live query, which is the weaker of the two
situations P3.1 distinguished, and it was chosen only after looking for the
stronger one.

- `informatics-archive/october-2014/mouse_expression/` **is** a genuinely
  static archive of the adult-mouse expression data, and it publishes
  per-experiment *voxel grids*, not structure unionizes. Turning those into
  per-region numbers means re-implementing Allen's unionization against
  `P56_Mouse_gridAnnotation.zip` and publishing our arithmetic as if it
  were theirs. The published unionize is the quantity the roadmap asks for;
  recomputing an approximation of it from a static file would have traded a
  provenance problem for a correctness problem, which is the worse trade.
- No static dump of the unionizes exists on the archive. Its
  `current-release` tree holds `brain_observatory`, `mouse_annotation`,
  `mouse_ccf` and `rna_seq` only; the dated trees (`october-2014`,
  `june-2016`, `september-2017`) hold projection data, cell types,
  annotation volumes and the voxel grids.

So the queries are pinned by digest and labelled `volatility="live-api"`,
the same treatment P3.1 gave the CCF structure graph for the same reason.
Both the index query and a per-gene query were verified byte-identical
across repeated requests. What the checksum buys is not immutability, which
nobody can promise for an API; it is that a new experiment or a pipeline
re-run becomes a dated, loud failure instead of a silent move in every
receptor map.

Two details fall straight out of that. **Row order is pinned explicitly**
in both queries, because RMA's default order is undocumented and a permuted
response would fail the digest for no real reason. (`order$eq
'section_data_sets.id'` is rejected by the API as ambiguous on the
SectionDataSet model, so the index orders by `genes.acronym`,`data_sets.id`
instead: found by trying it, not by reading it anywhere.) And the manifest
records **row and experiment counts** per gene alongside the digest,
because the digest cannot detect a *truncated* response: RMA paginates, and
a short response hashes perfectly consistently. `build_expression_manifest.py`
cross-checks each CSV's row count against the API's own `total_rows` before
pinning, and the loader re-checks offline.

### Sharing the connectome's ontology, and the twelve regions where it fails

Every query filters `structure[graph_id$eq1]`: the same CCF structure graph
the Knox et al. 2019 connectome is indexed by and that
`connectome/ontology.py` already loads. This package has no ontology of its
own and refuses to name a region without one being passed in. That is what
makes the acceptance criterion "regions line up exactly" enforceable in the
query rather than patched up downstream, and a test asserts the filter is
in every criteria string.

Measured coverage: **279 of the connectome's 291 source regions** carry ISH
unionizes, 95.9%. The twelve that do not are AP, ASO, EW, ICB, IV, NR, PST,
PVa, VISa, VISli, VISpor and VISrl, and they are the *same twelve* in six
independent experiments (Oprm1, Drd2, Gabra1, Cnr1, Slc6a3, Grin1), which
is what identifies the cause as structural rather than gene-specific. The
ISH grid is registered to the pre-CCFv3 P56 reference space (id 9 coronal,
10 sagittal) while Knox et al. regionalize onto CCFv3; four of the twelve
(VISa, VISli, VISpor, VISrl) are visual areas CCFv3 delineates and the
older annotation does not, and the rest are small nuclei in the same
position. `sources.MISSING_FROM_ISH` records them and a test on the real
files asserts the list is exactly right, so it cannot be quietly edited.

This is a real limitation of the alignment and it is recorded as one. It is
not a reason to prefer a different connectome: LC, the region that decided
P3.1's choice of Knox over Oh, is present in both layers.

### One invariant found worth enforcing: plane implies reference space

Coronal experiments are registered to Allen reference space 9 ("P56 Brain")
and sagittal ones to space 10 ("P56 Brain (L/R Flipped)"), with no
exceptions across all 106 indexed experiments. `SectionDataSet` checks the
two against each other on construction and raises rather than accepting a
differently registered experiment, because such an experiment would not be
commensurable with the rest and pooling it into a gene's map is exactly the
kind of thing that produces a plausible-looking wrong answer. Both spaces
predate CCFv3, which is the same fact that causes the twelve missing
regions above; recording it as an enforced invariant rather than a comment
means a future Allen re-registration announces itself.

### The two decisions the loader refuses to make

P3.1's `RegionalConnectome` offers `without_self_connections()` as an
explicit method rather than a default, because every convenience a loader
might add is a modelling decision. The same pressure exists here and it is
sharper, because a gene does not have one expression map.

**Which plane.** Allen's sagittal pipeline sections and processes the left
hemisphere only; coronal experiments cover the whole brain, and Allen's own
Mouse Brain Atlas documentation puts the coronal set at 4,104 genes with the
rest of the genome sectioned sagittally. Measured on this data: Oprm1's
coronal experiment carries unionizes for 279 of the 291
connectome regions, its two sagittal siblings for 262 and 274. Three of the
42 genes with data (Adra2c, Adrb2, Htr2a) are **sagittal-only**, so a
coronal-only atlas loses them entirely.

**Which experiment.** Oprm1 has three experiments, Chrm4 five, and they
disagree far more than one would like. Measured over the 291 connectome
regions, writing `r` for the max/min ratio across the experiments that
report a region, the median `r` per gene ranges from **1.18 (Gria3) to 31.5
(Chrm3)**, with Oprm1 at 6.2, Drd1 at 8.8, Adora2a at 11.6 and Slc6a3 at
9.3. Only 5 of the 38 multi-experiment genes have a median `r` below 1.4.
(The per-gene maxima are much larger still, up to 1e9, but they are
dominated by regions where one experiment saw essentially nothing and are
not worth reading; the median is.) `results/expression_coverage.txt` has
the whole table.

Averaging is defensible; taking the coronal experiment alone is defensible;
treating the spread as the measurement uncertainty it evidently is may be
the most defensible of the three. Which is right depends on what the
receptor layer needs, which is P4.2's question, and a loader with a default
would have hidden the size of the question.

So `energy_matrix()` takes `plane` and `combine` as **required** keyword
arguments and raises a `TypeError` naming the reason if either is omitted.
`plane` is one of `coronal` / `sagittal` / `any` /
`coronal-else-sagittal`; the last picks per gene and never pools the two
geometries for one gene, so a gene's map always comes from one sectioning
plane. `combine` is `mean` / `median` / `max` / `single`, and `single`
raises unless the plane filter left exactly one experiment: the strict
option, under which no averaging of anything happens at all. Every returned
matrix records which policy produced it, which experiment ids went into it,
and how many experiments contributed to each individual cell.

### NaN is not zero, again

An absent unionize row means *not measured*, and it is NaN. It is never
0.0. This is the same refusal P3.1 made for Oh et al. 2014's `#NUM!`
p-value cells, and it matters more here rather than less: a zero in a
receptor map is the claim "a drug acting at this receptor has no target in
this region", which is a scientific assertion, while NaN is the truth.

There is a second, sharper distinction underneath it: an unmeasured region
comes back NaN and is reported, while a structure id that is not in the
ontology at all is *refused*. Both would otherwise produce an identical
all-NaN column, and one of them is a mis-join.

`RegionalExpression` enforces the NaN correspondence as an invariant - a
finite value with no contributing experiment, or a NaN with contributors, is
a construction error - and reports the gaps through
`genes_without_values()`, `regions_without_values()` and `coverage()`
rather than leaving a caller to discover them. The combining rules are
NaN-aware for the same reason: an experiment that does not report a region
does not vote, so it cannot drag that region's mean toward zero. A test
constructs the zero-filled alternative and asserts the real values are
strictly larger everywhere the two differ.

### Adra1b: searched and empty

**`Adra1b` (adrenergic alpha-1B receptor, Entrez 11548) has no in situ
hybridization experiment in any Allen mouse product.**

What was searched, on 2026-08-12: `model::Gene[acronym$eq'Adra1b']` returns
three records, two of them *Mus musculus* (Allen gene ids 11335 and
500726721, both Entrez 11548, "adrenergic receptor, alpha 1b"), so the gene
is in Allen's table and this is not a spelling or symbol-drift problem.
`model::SectionDataSet,rma::criteria,genes[acronym$eq'Adra1b']` with **no
product filter and no `failed` filter**, the widest query the API supports,
returns `total_rows = 0`.

The consequence, recorded rather than worked around: alpha-1B stays in
`genes.RECEPTOR_GENES` with no expression map, and any alpha-1 map built
from this package is explicitly alpha-1A + alpha-1D only. Substituting
another alpha-1 subtype's map for it would be inventing a number and is
refused; a test asserts that `Adra1b` is absent from the available symbols,
that its record says what was searched, and that asking the atlas for it
raises an error containing the search rather than a bare `KeyError`.

### Four experiments that exist and yielded nothing

Of the 106 non-failed ISH experiments the index lists for these genes,
**four have no structure unionizes in any graph**: 282 (Cnr1), 359 (Drd2),
526 (Gria1) and 534 (Grin1), all from the original sagittal release. Not
merely none in graph 1: none at all.

Nothing in the API's `SectionDataSet` record explains why, and the first
plausible explanation checked was wrong, which is worth recording. The
`expression` flag is false for 28 of the 106, so it looked like the
distinguishing field; but 24 of those 28 do have unionizes, so it is not.
Neither is `delegate`, `failed`, `failed_facet`, `section_thickness`,
`weight`, `reference_space_id`, nor any channel field: every one of them is
identical between experiment 282 and its sibling 284, same gene, same
plane, one with unionizes and one without. The cause is not recoverable
from this data, so it is not guessed at.

What is done instead is to keep the four visible. The index query does not
filter them out; adding `[expression$eq'true']` would have made the counts
add up and made the fact invisible (and would also have dropped 24
experiments that are fine). `ExpressionAtlas.experiments_without_unionizes()`
reports them, and the fixture deliberately includes two (282 and 359) so
the index-versus-data mismatch path runs offline on real data.

### The fixtures, and the ontology they do not have

Eight genes and 55 structures, ~155 kB, cut from the verified responses and
written back out in their own formats. The structures are exactly the 55 in
the *connectome's* committed fixture ontology, and there is deliberately no
expression-specific ontology fixture: P4.1's whole claim is that the two
layers are indexed by the same structure graph, and two fixture ontologies
free to drift apart would let the offline tests keep passing after that
stopped being true.

The eight genes were chosen for what each one exercises, not for coverage.
`Htr2a` is the sagittal-only case, so `plane="coronal"` produces an all-NaN
row that has to be reported rather than read as "not expressed". `Drd1`'s
experiment 354 reports only 30 of the 55 structures, the widest real
coverage gap available. `Cnr1` and `Drd2` each carry one of the four
indexed-but-empty experiments. `Oprm1`, `Drd2` and `Gabra1` are the three
receptor systems the model already simulates; `Adora2a` and `Slc6a3` are
P4.3 validation targets. Rows are copied through `csv.reader`/`csv.writer`
without ever being parsed as floats, so the fixtures are character-identical
to the bytes Allen served, and a real-data test asserts the arrays match
over all 55 structures of all 24 experiments.

### Anatomy, as a bug check rather than a finding

P4.3 owns the validation. Some of its checks are already running here,
because a mis-joined or mis-attributed ingestion would otherwise look
completely healthy - same sparsity, same distribution, plausible numbers -
and these are the cheapest available evidence that it is not:

- **Drd1, Drd2, Adora2a peak in CP / ACB.** On the full 291-region matrix
  the top regions are CLA, CP, EPd, ACB, OT for Drd1 and Drd2, and CP, CLA,
  ACB, OT for Adora2a.
- **Oprm1 peaks in the medial habenula**, then PVT, RH, CM, IAM, IMD: the
  midline thalamus. That is the classic Oprm1 distribution and it is the
  roadmap's stated P4.3 expectation.
- **Slc6a3 (DAT) sits in SNr/IPN/SNc/VTA, not in the striatum.** DAT is
  expressed by dopamine neurons, so it belongs in their somata; the
  striatum is where dopamine is *released*. A DAT map peaking in CP would
  mean projection data had been attached to a gene. (SNr above SNc is the
  well-known spillover between two adjacent structures at 200 um, not a
  parse error.)
- **Slc6a4 (SERT) peaks in the dorsal raphe**, and Cnr1 in cerebellar
  cortex and amygdala.

Four of these (Drd2, Adora2a, Slc6a3, Oprm1) run offline on the fifteen
fixture regions; the full versions run on the real 291-region matrix and
skip without the gitignored cache.

### What this task changes, and what it does not

- **No parameter changed, no illustrative label moved, no scenario
  touched.** The expression package is additive; nothing in `engine.py`,
  `receptor.py`, `scenarios.py`, `connectome/` or any calibration module
  imports it. Every existing result is unchanged.
- **No number in this package is a modelling choice.** Every value is a
  published unionize, a measured SHA-256, a measured row count, or an NCBI
  Entrez id. There is nothing here to label illustrative and nothing here
  calibrates anything.
- **`connectome/fetch.py` was refactored, not changed.** Its function
  names, signatures and behaviour are identical; `DataFile` moved to
  `datafetch.py` and is re-exported. All 47 P3.1 tests pass untouched.
- **65 new tests.** 59 run offline off the committed fixtures; 6 check the
  full 42-gene responses and skip when the gitignored cache is absent. They
  are extras, not substitutes: every acceptance criterion is covered
  offline.

## Phase 4: expression energy as a receptor density (P4.2)

### What this task is, and the sentence it exists to make honest

P4.1 stopped at expression energy and said, in four places, that the
conversion to a density and the statement of its assumption belong to P4.2.
This is that task. The whole of it reduces to one line and what surrounds
it:

    ASSUMPTION: within one gene, receptor density in a region is
    proportional to that gene's Allen ISH expression energy in that region.
    density(region) = k_gene * energy(region).

Four new modules, none of which changes an existing parameter:

    density.py          the conversion, the assumption as a named constant,
                        and the density-times-mechanism-gain factorization
    mismatch.py         the genes whose receptor protein is not in the
                        region that transcribes it, with the error measured
    autoradiography.py  published mouse receptor densities, transcribed
                        from two open-access papers with full conditions
    validation.py       the comparison, reported as three numbers

plus `scripts/report_density_validation.py`, which writes the committed
`results/expression_density_validation.txt`, and 43 tests, 38 of which run
offline.

### What "the density scale the receptor layer consumes" actually is

The task's phrasing invites a wrong answer. `ReceptorSite` holds
*fractional* occupancy; the binding ODE is dimensionless and needs no
density at all, and that is not an oversight, because competitive occupancy
really is density-independent when the ligand pool is not depleted by
binding. There is no `Bmax` field waiting to be filled in.

Density enters one step later, in `Simulation.rhs`, which adds
`gain * activation` to a node's drive. That single number folds two things
together, and P4.2 supplies exactly one of them:

    coupling[node] = mechanism_gain[node] * rho[region]

`rho` is what `relative_density()` returns: dimensionless, per region,
normalized by an explicitly chosen policy. `mechanism_gain` stays whatever
it already is - illustrative in `scenarios.py`, fitted in P2.2 - and keeps
its label.

The reason for the split is a lesson this document already records. The
multi-region mu-opioid scenario gives LC the *smallest* coupling gain of its
three targets even though LC has the *highest* published mu-receptor density
of the three, because LC's receptor acts directly on the node being read out
while the other two act through an interposed interneuron-disinhibition step
that amplifies them. Density and gain are different axes. A P4.2 that
returned "the coupling constant" would have quietly asserted they were the
same one, and would have made every future density improvement look like a
recalibration of circuit mechanism. `ReceptorDensityMap.as_coupling()`
performs the multiplication so that the factorization is executable and
testable rather than only described here.

### Normalization has no default, for P4.1's reason

`relative_density()` requires `normalization=`, exactly as `energy_matrix()`
requires `plane=` and `combine=`. "Relative to the peak region", "relative
to the average region" and "relative to one named region" are three
different quantities whose difference is invisible in the output array, and
`reference-region` is the one a caller wants when a scenario already has one
calibrated node and the others should be expressed against it. `none` passes
energy through and keeps P4.1's units string, which says in full that the
number is not a density.

The exponent of the assumed relation is a module constant,
`PROPORTIONALITY_EXPONENT = 1.0`, so the one place the assumption enters the
arithmetic is greppable. It is a parameter rather than a hard-coded 1
because the measurement below finds the real exponent is nowhere near 1 -
but the parameter's default stays 1.0 and no per-gene exponent table is
shipped, because one gene's measured exponent is not evidence about another
gene's and a table of them would invite exactly that misuse.

### The comparison: Drd1 against measured mouse D1 density

The acceptance criterion is a comparison against independent density data
for at least one receptor. The primary one is:

  [Yoo2010] Yoo JH, Bailey A, Ansonoff M, Pintar JE, Matifas A, Kieffer BL,
            Kitchen I. "Lack of genotype effect on D1, D2 receptors and
            dopamine transporter binding in triple MOP-, DOP-, and
            KOP-opioid receptor knockout mice of three different genetic
            backgrounds." Synapse. 2010;64(7):520-527. PMID 20196137,
            doi 10.1002/syn.20757. Male wild-type mice >9 weeks, three
            backgrounds (congenic B6, coisogenic 129, B6 x 129),
            quantitative in vitro autoradiography, [3H]SCH-23390 4 nM for
            D1, [3H]raclopride 4 nM for D2, [3H]mazindol 4 nM with 0.3 uM
            desipramine for DAT, fmol/mg tissue equivalent.

It is the primary one for a specific reason beyond being mouse and
quantitative: it measured D1, D2 and the dopamine transporter *in the same
animals on the same films*, which is what lets the terminal-field section
below separate "the conversion is wrong" from "this receptor is trafficked
away from the cell that transcribes it".

Ten of its fifteen D1 regions join unambiguously to CCF structures. The result,
under `plane="coronal"`, `combine="mean"`:

| quantity | value | what it answers |
|---|---|---|
| Spearman rho | **+0.709** (p = 0.022, n = 10) | does the map rank regions correctly? |
| Pearson r | +0.601 | |
| proportional R^2 (through 0) | **+0.310** | is one constant enough? |
| log-log slope | **0.460 +/- 0.105** | is the relation proportional at all? |
| transcript dynamic range | 340x | |
| measured dynamic range | 20x | |

**The map is a good ordering and a bad scale.** Region ranks agree; a single
`k_gene` explains a third of the variance; and the log-log slope is more
than four standard errors below the 1.0 that proportionality requires. The
worst regions are reported, not dropped: VTA's measured D1 is 29x what the
proportional model predicts, thalamus 8.8x, hypothalamus 6.3x. All three are
low-transcript regions, which is the compression the slope describes.

Two explanations for a slope near 0.5 are live and this data cannot separate
them. Saturating translation and trafficking would compress a wide
transcript range into a narrow protein range. So would an autoradiographic
detection floor: [3H]SCH-23390 in thalamus, hypothalamus, hippocampus and
VTA all read 20-25 fmol/mg, suspiciously flat for four unrelated regions,
and a floor there would flatten the low end artificially. Because the two
cannot be distinguished here, the exponent is reported and **not applied**.
Applying 0.46 would be fitting a shape to one gene's confound.

### The one honest use of the word "absolute"

`AbsoluteScale` exists and there is exactly one path to constructing it:
`validation.fit_absolute_scale()`, from a comparison against measured
densities of that gene in that species. It carries the fit's R^2 with it,
because `k_Drd1 = 17 fmol/mg per energy unit` fitted at R^2 = 0.31 is a real
number and a bad one, and separating the constant from its quality is how a
bad number turns into a quoted number. `to_absolute()` refuses a different
gene with an error that says why: expression energy is not comparable
between genes, because probe and segmentation threshold differ per
experiment, so a `k` fitted on Drd1 says nothing about Oprm1. Extending it
would be inventing a number.

### The presynaptic / terminal-field mismatch, measured

The task named this as the important failure case and it is. In situ
hybridization measures mRNA and mRNA is in the cell body; the protein is
wherever the neuron trafficked it, and for a large class of receptors that
is the axon terminal, in a region that transcribes none of the gene. For a
*circuit* model this is worse than imprecision: it puts the receptor in the
wrong node, and the model then propagates that into a behavioural
prediction.

Two cases have both sides measured in mouse, so both are numbers. Both are
quoted here under `plane="coronal"`, the tests' policy;
`results/expression_density_validation.txt` recomputes them under
`coronal-else-sagittal` and gets 120x/173 and 63.2x/31.9, which is the
size of the policy's effect on them:

- **Slc6a3 (DAT).** Transcript puts 119x as much in SNr as in CP.
  [3H]mazindol binding puts 0.69x as much - the striatum has slightly *more*
  than the substantia nigra. The proportionality assumption misstates the
  SNr:CP ratio by a factor of **172**. This is not a receptor map that needs
  refining; it is a map of where the transporter is made.
- **Drd1.** Transcript puts 63x as much in CP as in SNr. [3H]SCH-23390 puts
  2.0x as much, because nigral D1 protein sits on striatonigral terminals.
  The error factor is **32**. SNr is out of the headline comparison for an
  unrelated reason - CCF has no combined "substantia nigra" structure to
  join Yoo et al.'s region to - but the effect of the exclusion is to drop
  the single worst-disagreeing region, so the comparison is reported *with*
  SNr included as well (rho falls 0.709 -> 0.591, proportional R^2 0.310 ->
  0.147). Excluding the region that disagrees most and reporting only the
  better number would have been the dishonest move available here. The nigrostriatal bundle is the same mechanism caught in
  transit: 91 fmol/mg of D1 binding in a fibre tract with no cell bodies at
  all.

Six further genes are documented and not quantified, because no mouse table
with both sides exists for them: Cnr1 (striatal transcript, pallidal and
nigral protein - [Herkenham1991] PMID 1909204, [Mailleux1992] PMID 1376455),
Slc6a4 (PMID 8783244), Slc6a2 (PMID 10753308), Oprm1's medial-habenula to
interpeduncular projection ([Gardon2014] PMID 25086313), Adra2a, and Drd2 -
whose record cites the transcript map itself rather than a paper, because
unlike the transporters SNc and VTA do express Drd2, so both ends of the
nigrostriatal projection are already in the map and its `partial` severity
follows from that observation.

Each record carries a severity: `terminal-field` when the dominant pool is
elsewhere, `partial` when a named minority pool is.

Two design decisions came out of this:

**A `terminal-field` gene refuses to convert silently.** `relative_density()`
raises for Slc6a3, Cnr1, Slc6a4 and Slc6a2 unless the caller passes
`acknowledge_mismatch=True`. The flag changes no arithmetic. Its whole
function is that using a dopamine-transporter *transcript* map as a
dopamine-transporter *density* map has to be something a caller typed. This
is P4.1's refusal-to-default pattern applied to a hazard rather than a
choice.

**No correction is applied.** The tempting one - push each source nucleus's
transcript out along the P3.1 connectome's projection weights - needs the
fraction of each receptor pool that is trafficked to terminals, per gene,
and that fraction is not published for any gene here. Multiplying by a
plausible-looking fraction is exactly the invented number this project
forbids. Correcting these maps is what Phase 6's projection-system
abstraction is for, and Phase 6 goes and gets the projection-density and
release data it would need. Until then the honest output is a flagged map
with a measured error bar.

**The genes deliberately absent from the registry are a claim too.** The
ionotropic receptors (`Grin*`, `Gria*`, `Gabra*`, `Gabrb*`) are
predominantly postsynaptic on the dendrites of the cell that transcribes
them, which is the case the proportionality assumption is built for. Their
absence is an assertion about them, recorded as one in `mismatch.py`'s
docstring, not an oversight.

### Breadth: seven receptor systems, and the AMPA null result

One receptor satisfies the acceptance criterion; one receptor also makes it
impossible to tell whether 0.71 is typical. So a second dataset:

  [Lothmann2021] Lothmann K, Amunts K, Herold C. "The Neurotransmitter
            Receptor Architecture of the Mouse Olfactory System."
            Front Neuroanat. 2021;15:632549. PMID 33967704,
            doi 10.3389/fnana.2021.632549. Ten male C57BL/6 mice, 27 weeks,
            quantitative autoradiography with ten tritiated ligands over
            fourteen olfactory-system regions, converted to Bmax at
            saturation, fmol/mg protein.

Same fourteen regions, same expression policy, one gene per row (Spearman
rho, n = 14 in every row shown). The full 25-row table is in
`results/expression_density_validation.txt`; these are the rows that carry
the result, and the one omitted AMPA subunit, Gria3, is +0.132 over its 13
reported regions, which changes nothing about the paragraph below it:

| ligand | gene | rho | | ligand | gene | rho |
|---|---|---|---|---|---|---|
| [3H]prazosin (a1) | Adra1d | **+0.763** | | [3H]muscimol | Gabra1 | **+0.723** |
| [3H]prazosin (a1) | Adra1a | **+0.714** | | [3H]flumazenil (BZ) | Gabra3 | **+0.789** |
| [3H]MK-801 | Grin2a | **+0.754** | | [3H]flumazenil (BZ) | Gabra1 | +0.578 |
| [3H]MK-801 | Grin1 | +0.090 | | [3H]SCH-23390 | Drd1 | +0.512 |
| [3H]AMPA | Gria1 | -0.182 | | [3H]RX-821002 (a2) | Adra2a | +0.116 |
| [3H]AMPA | Gria2 | -0.007 | | [3H]RX-821002 (a2) | Adra2b | **-0.688** |
| [3H]AMPA | Gria4 | -0.209 | | [3H]RX-821002 (a2) | Adra2c | -0.433 |

Three things worth saying about that table.

**The conversion's quality is receptor-specific, not global.** Alpha-1 and
GABA-A track the measured density about as well as D1 did; *no* AMPA subunit
tracks AMPA binding at all, in either direction. A whole-brain receptor map
built on this assumption inherits that unevenness receptor by receptor, and
the honest form of a P4.2 deliverable is therefore a per-receptor quality
number rather than one headline.

**A subunit's transcript is not its receptor's density, and the spread
inside a family shows it.** Grin2a correlates at +0.75 while Grin1 - the
obligatory subunit present in every NMDA receptor - correlates at +0.09.
Gabra1 at +0.72 while Gabra2 is at -0.25. Which subunit's map should stand
for the assembled receptor is not answerable from this data, and summing
them is not available either, because energies are not comparable across
genes. This is the pentamer problem P4.1's gene table warned about, now with
numbers on it.

**Alpha-2 is the worst case, and only half of it has an explanation.**
Adra2a at +0.12 and Adra2b at -0.69 are the sweep's failures. Adra2a has a
documented mechanism and a `partial` mismatch record: alpha-2A is the
locus-coeruleus autoreceptor on noradrenergic terminals as well as a
postsynaptic cortical receptor, and a binding assay cannot separate the two
pools. Adra2b's *negative* correlation has no such explanation and is not
given one here. What can be said about it is that all three alpha-2
transcripts sit near the floor in these regions (median energy 0.06-0.13
against Gabra1's 6.5-20 in the same structures), so the ordering being
measured may be mostly noise - which is a caveat, not a mechanism, and is
recorded as the former.

The benzodiazepine row deserves a note of its own: it is the site this
codebase's diazepam scenario actually acts at, and it is compared against
Gabra1/2/3/5 only. Gabra4 is excluded because alpha4-containing pentamers
are classical-benzodiazepine-insensitive. P4.1 ingested Gabra4 specifically
so that P4.2 could exclude it deliberately rather than pool it in by
accident; this is that exclusion, and a test asserts it.

### How much P4.1's deferred choice was worth

P4.1 refused to pick a plane policy or a combining rule and handed the
choice to P4.2. P4.2's answer, for the D1 comparison, is that it was worth
almost nothing: all fourteen plane x combine policies give Spearman between
**0.709 and 0.794**. The refusal cost one required keyword argument and
bought the ability to say that, which is the right trade. It would not have
been possible to say it if the loader had defaulted.

For the record the headline numbers use `plane="coronal"`, `combine="mean"`
in the tests (the fixture's Drd1 has one coronal experiment, so this is the
least reduced option available) and `plane="coronal-else-sagittal"`,
`combine="mean"` in the report, where three sagittal-only genes would
otherwise vanish.

### Two smaller decisions

**Region joins are name-based and every one is written down.** The
autoradiography papers delineate on their own atlases (Hof et al. 2000;
Franklin & Paxinos), the ISH unionizes are over CCF structures, and
`REGION_MAPPING` records a reason string for each of the 29 published region
names - including the six that are deliberately left unmapped. "Nucleus
accumbens core" and "shell" are unmapped because CCF's ACB is undivided and
merging them would be an averaging rule invented by this module; "amygdala"
because CCF splits the amygdalar nuclei between CTXsp and sAMY; "substantia
nigra" because CCF has SNc and SNr as siblings with no combined parent and
the two differ by an order of magnitude in Drd1 transcript; "nigrostriatal
bundle" because it is a fibre tract; "subthalamic area" because the phrase
is not CCF's STN. Each of those is a comparison point that would have
measured the atlas join rather than the conversion.

**Both datasets are mouse, and that was a constraint on which datasets could
be used.** Kitchen et al. 1997's mu-opioid autoradiography is the obvious
first candidate for a receptor this model already simulates, and it is
paywalled; the open-access mu-opioid alternative found (Effah et al. 2024,
EJN, PMC11612844) publishes its region-wise MOPr densities only as bar
figures, with a whole-brain mean in the one numeric table. So there is no
Oprm1 comparison here. Substituting the abundant *rat* mu-opioid
autoradiography would have violated rule 2 for the sake of covering a
receptor the model uses, and it was not done. What Oprm1 has instead is the
qualitative anatomy check P4.1 already runs (medial habenula, then midline
thalamus) and a documented, unquantified mismatch record.
`autoradiography.SEARCHED_AND_EMPTY` carries both this search and the
extrastriatal-D2 one in the Track E format, with the queries and the dates,
so neither gets repeated and neither turns into a silent gap.

### Fixture extension, and why it did not touch the connectome matrices

The offline comparison needs OT, CLA, EPd, ACA and HIP, which the committed
fixture ontology did not have. Adding them to `FIXTURE_REGION_IDS` would
have changed the committed Knox and Oh matrices and every number the P3.x
tests assert on them, for no gain, since P4.2 never touches a connection
weight. So they went into a new `EXTRA_ONTOLOGY_REGION_IDS`: ontology
entries with no matrix row. The shared fixture ontology went from 55
structures to 61 (the five, plus EPd's parent EP), the expression fixtures
were recut over the larger structure set from the same verified bytes, and
two assertions that count fixture rows were updated. The Knox CSVs are
byte-identical. There is still exactly one fixture ontology, which is the
property P4.1 said must not be lost.

The payoff is that the acceptance criterion runs in the offline gate: the
D1 comparison, both measured displacements and the policy sweep are
fixture-backed, and a `real_data` test asserts the full 291-region matrix
gives the same rho to three decimals, so a fixture that stopped being a
slice of the data would fail loudly.

### What this task changes, and what it does not

- **No parameter changed, no illustrative label moved, no scenario
  touched.** Nothing in `engine.py`, `receptor.py`, `scenarios.py`,
  `connectome_scenarios.py` or any calibration module imports the new code.
  Every existing result is unchanged, and the whole suite passes with the
  fixture extension.
- **One new fitted constant exists**, `k_Drd1 = 17 fmol/mg tissue
  equivalent per expression-energy unit`, and it is not stored anywhere: it
  is recomputed from the published table every time, carries its own R^2 of
  0.31, and refuses to be applied to any other gene.
- **The three numbers that matter are negative results and are reported as
  the headline.** Proportional R^2 of 0.31, a log-log slope of 0.46 where
  proportionality needs 1.0, and a 172-fold misplacement for the dopamine
  transporter. No tolerance was widened and no region was dropped to
  improve any of them.
- **43 new tests**, 38 offline and 5 requiring the gitignored cache.

## Phase 4: validating the receptor maps against known anatomy (P4.3)

### The task, and the sentence in it that sets the standard

The roadmap's P4.3 line ends "A map that fails these is a bug, not a
finding", and that sentence is the whole design. Everything in
`expression/anatomy.py` is a bug detector rather than a result: a
transposed matrix, a gene column shifted by one, a structure-id mis-join or
a normalization applied along the wrong axis all produce a receptor map
with the right shape, the right sparsity and entirely plausible numbers,
and anatomy is the cheapest thing that tells them apart from a real map.
A2A is a striatal receptor; alpha-1 GABA-A is a cortical one; mu-opioid
peaks in the medial habenula. A map that says otherwise is broken, and the
right response is to go and find the defect.

Three modules and two test files:

    anatomy.py        the cited expectations, as executable claims, and the
                      checker that measures them
    scenario_maps.py  the join from a gene symbol plus {node: structure id}
                      to the density factor a scenario's coupling needs
    scenarios.py      the multi-region mu-opioid map, now atlas-derived

plus `scripts/report_receptor_map_validation.py`, which writes the
committed `results/receptor_map_validation.txt`, and the tests: the two
files `tests/test_expression_anatomy.py` and
`tests/test_multi_region_scenario.py` now collect 42 items where the second
of them used to collect 7, and 5 of the 42 need the gitignored cache.

### Six receptors, and what each source actually is

The acceptance bar is five receptors with cited expected distributions.
There are six, and the constraint that decided which six was not
importance but whether a *mouse* source exists that says something
falsifiable about regions this project can resolve.

| gene | source | species | modality |
|---|---|---|---|
| Drd1 | [Yoo2010] (P4.2's transcription, reused) | mouse | protein, quantitative |
| Drd2 | [Yoo2010] + [Bello2011] | mouse | protein / genetic |
| Oprm1 | [Erbs2015] mu-mCherry knock-in atlas | mouse | protein |
| Gabra1 | [Hoertnagl2013] | mouse | **mRNA** |
| Cnr1 | [Marsicano1999]; [Herkenham1991], [Mailleux1992] | mouse; rat | mRNA; protein |
| Adora2a | [Wang2022] A2AR-tag knock-in | mouse | protein |

  [Erbs2015] Erbs E, Faget L, Scherrer G, et al. "A mu-delta opioid
            receptor brain atlas reveals neuronal co-occurrence in
            subcortical networks." Brain Struct Funct. 2015;220(2):677-702.
            PMID 24623156. Table 1's MOR column, read from the PMC deposit:
            MHb ++++, CPu +++, AcbC ++, PVA +++, PV ++, MD +, VTA ++,
            SNC +/++, SNR +, LC +, LPAG +/++, VLPAG +, DLPAG +, DMPAG -,
            M1/M2 +, PrL +, CA1/CA3 +, DG +, CbCx -. Scale: "- not
            detectable, +/- weak, + moderate, ++ dense, +++ very dense".
  [Hoertnagl2013] Hoertnagl H, Tasan RO, Wieselthaler A, Kirchmair E,
            Sieghart W, Sperk G. "Patterns of mRNA and protein expression
            for 12 GABAA receptor subunits in the mouse brain."
            Neuroscience. 2013;236:345-372. PMID 23337532. alpha1 ISH:
            cortex layer 4 +++, layers 2/3/6 ++, CA1 +++, DG ++(+), CA3 +,
            caudate putamen (+), globus pallidus +++, STN +++, SNr +++,
            SNc ++, cerebellar Purkinje +++, granule ++, molecular +.
  [Wang2022] Wang M, Li Z, Song Y, et al. "Genetic tagging of the adenosine
            A2A receptor reveals its heterogeneous expression in brain
            regions." Front Neuroanat. 2022;16:978641. PMID 36059431.
  [Marsicano1999] Marsicano G, Lutz B. Eur J Neurosci. 1999;11(12):4213-25.
            PMID 10594647.
  [Mailleux1992] Mailleux P, Vanderhaeghen JJ. Neuroscience.
            1992;48(3):655-68. PMID 1376455. RAT.
  [Herkenham1991] Herkenham M, Lynn AB, de Costa BR, Richfield EK. Brain
            Res. 1991;547(2):267-74. PMID 1909204. RAT.
  [Bello2011] Bello EP, Mateo Y, Gelman DM, et al. Nat Neurosci.
            2011;14(8):1033-8. PMID 21743470.

[Hoertnagl2013] is the one whose modality matches the map exactly, and it
is the most valuable of the six for that reason alone. Every other source
measured protein or binding, so every other claim is crossing the
mRNA-to-protein step that P4.2 measured separately and found to be worth a
factor of 172 for the dopamine transporter. `AnatomySource.modality` is a
required field with only two legal values precisely so that this is
visible per claim rather than buried.

**Two rat sources, and what they are allowed to do.** Cnr1's protein claim
is [Herkenham1991] and its "no CB1 mRNA in dorsal thalamus" half leans on
[Mailleux1992]; both are rat, both say so in their record, and neither
supplies a number. They are used for a presence/absence ordering whose
mouse half [Marsicano1999] independently supports. P4.2 refused rat
mu-opioid autoradiography for a *quantitative* comparison and that refusal
still stands; what would have been dishonest here is quoting a rat density
and calling the difference the conversion's error.

### Three rules that make an anatomical expectation testable

**Ordinal, never proportional.** A claim is `min(map[high]) >
max(map[low])` and nothing else. P4.2 measured this conversion to be a good
ordering (Spearman 0.71 against mouse D1 autoradiography) and a bad scale
(proportional R^2 0.31, log-log slope 0.46 where proportionality needs
1.0), so an ordering is the only thing it can be validated as. Every
`ClaimResult` reports its fold-difference and no test asserts on one.

**Asserted only at two tiers or more.** Five of the six sources are
semiquantitative: [Erbs2015] scores on `-/+/++/+++`, [Hoertnagl2013] on
`-/(+)/+/++/+++`. One tier is one observer's judgement of one section, and
P4.2's compression means one tier of protein can correspond to anything
from no difference to a 20-fold difference of transcript. So each claim
carries a `separation` counted from the source's own labels *before* any
measurement was made, and the test suite asserts a claim only at
`separation >= 2`. Fourteen of the eighteen claims clear that bar.

The obvious risk of such a rule is that it becomes a place to put
inconvenient results, so the four single-tier claims are pinned by a test
that asserts *which* of them hold: three do, one does not, and any change
to any of the four fails. The rule is not a filter for failures.

**A divergence is a claim too.** Two claims are recorded with
`expected_to_hold=False`, and the `Claim` constructor refuses one unless it
carries a mechanism and a citation for that mechanism - otherwise
"expected to fail" is just a dropped inconvenient result with a nicer name.
Both are genes `mismatch.py` already flags:

- **Cnr1: SNr and GPe above CP.** [Herkenham1991]: "In rat brain, the
  highest receptor densities are in the globus pallidus and substantia
  nigra pars reticulata." The transcript map puts SNr at 0.23 and GPe at
  0.34 of CP. The same paper's ibotenic-acid lesion is the mechanism: those
  receptors sit on the axons and terminals of striatal projection neurons.
- **Drd1: SNr above cortex.** Yoo et al. measure mouse nigral D1 at 206.3
  against 22.8 fmol/mg in motor cortex; the transcript puts SNr at 0.29 of
  MOp. Mechanism [Gerfen2011]: striatonigral terminals.

An expected failure that is asserted to fail still fails loudly if somebody
"fixes" it, and its measured size is the error bar on using that gene's map.

### The result: seventeen of eighteen, and the one that did not

Fourteen asserted claims all behaved as recorded. Of the four reported
single-tier claims, three held and one did not: **Oprm1, cortex above
cerebellar cortex**, where [Erbs2015] scores M1/M2 `+` against CbCx `-`.

Traced rather than left. MOp reads 0.1150 and CBX 0.1157 - a 0.6%
difference, with both regions at the 11th percentile of the whole-brain
Oprm1 distribution (min 0.0495, 10th percentile 0.1082, median 0.3307).
There is no ordering in the map to be right or wrong about; two regions at
the floor are not inverted, they are indistinguishable. Two further
measurements say the same thing. The disagreement disappears entirely under
`plane="any"`, which pools Oprm1's three experiments, and P4.1 measured
Oprm1's median between-experiment ratio at 6.2, so a 0.6% gap is far inside
the data's own spread. And a mis-join would move a region by orders of
magnitude, which is exactly what the same gene's MH does in the other
direction: 6.04, 52x these two, at the 99.6th percentile.

So: no ingestion or normalization defect was found, and the one
disagreement is a floor effect with two independent measurements of its
size. That is the honest form of "the validation passed", and it is
recorded that way rather than as eighteen ticks.

### The claustrum, run down and left as an open question

The most conspicuous oddity in these maps is that **CLA is the single
highest region of both the Drd1 and the Drd2 transcript maps, above CP**,
and second for Adora2a. P4.1 recorded the fact without comment. The cheap
explanation is partial-volume contamination: the claustrum is a thin sheet
against the striatum and the ISH grid is 200 um, so CLA voxels could be
carrying striatal signal.

That hypothesis makes a testable prediction, because the geometry is one
geometry: a single mixing fraction should explain CLA across all genes.
Fitting `CLA = f*CP + (1-f)*AIv` per gene over the 31 genes where CP and
insular cortex differ enough to be informative gives f from **-0.37**
(Gria4; Chrna4, Adrb1 and Htr2a are also near zero, i.e. CLA reads as
cortex) to **+2.23** (Adra2b; Drd1 1.44 and Gabra2 1.45 read as more
striatal than the striatum), median 0.64, interquartile range 0.30 to 0.94.
One geometric mixture cannot produce that spread, so the hypothesis is
refuted by its own prediction. Independently, Slc6a3 ranks CLA 241st of
291, so the claustrum is not a globally "hot" structure either.

Independent corroboration for the D1 half: Yoo et al. measured mouse
claustral D1 at 87.3 fmol/mg, fourth of their fifteen regions and 3.8x
motor cortex. So high claustral D1 is real in protein, and the Drd1
CLA-above-MOp claim is asserted for that reason. What remains unexplained
is claustral Adora2a at 82% of CP, for which no source in reach says
anything either way. Recorded as an open question about claustral biology
with the refutation of the artefact hypothesis attached, and deliberately
not asserted as a claim in either direction.

### One cross-gene check, and the only legitimate shape for one

P4.2's rule is that expression energy is not comparable between genes:
probe, riboprobe length and segmentation threshold differ per experiment.
That rules out "is Drd2 higher than Drd1 in VTA", which is exactly the
question that would catch two gene rows swapped into each other. The shape
that survives the rule is a ratio of *within-gene* ratios:

    (Drd2[VTA] / Drd2[CP]) / (Drd1[VTA] / Drd1[CP]) = 62.2
    (Drd2[SNc] / Drd2[CP]) / (Drd1[SNc] / Drd1[CP]) = 28.9

Both halves are internal to one gene, so the probe difference cancels
rather than being ignored. Well above 1 is the D2-autoreceptor anatomy
[Bello2011] against a midbrain D1 that Yoo et al. measure at cortical
levels (VTA 24.8 against motor cortex 22.8 fmol/mg). Near 1 would mean the
rows had been shuffled.

### Replacing the hand-written mu-opioid map

The third acceptance criterion. `mu_opioid_multi_region_coupling()` used to
return three hand-tuned numbers whose docstring cited [TempelZukin1987] and
[Mansour1987] for a qualitative density ordering and then, correctly, said
the numbers were not a density-to-gain conversion. P4.2 gave that folding a
factorization, so the replacement is exact rather than a rescaling:

    coupling[node] = mechanism_gain * rho[region]

`rho` now comes from the Allen Oprm1 ISH map over VTA (749), PAG (795) and
LC (147); `mechanism_gain` stays hand-set and stays labelled illustrative.

| node | region | rho (VTA = 1) | atlas coupling | hand-written |
|---|---|---|---|---|
| VTA_GABA | VTA | 1.00 | -1.72 | -3.0 |
| PAG_Interneuron | PAG | 2.52 | -4.34 | -2.5 |
| LC | LC | 4.27 | -7.35 | -1.5 |

**The normalization is `reference-region`, and that is load-bearing.**
Dividing the row by one named region makes the result depend only on the
rows of the regions the scenario names, where dividing by the peak or the
mean does not. That is what lets the committed 69-structure fixture and the
gitignored 7.5 MB cache give the same three numbers - the fixtures are
byte-identical slices, so this is an identity and not an approximation -
and it is why `scenarios.py` can import the expression package without
making every existing scenario test require the cache. A `real_data` test
checks the identity against the cache rather than leaving it as prose.

**The ordering changes, and the direction is the one the old citations
already asked for.** [TempelZukin1987] put LC among the highest-density mu
regions and the hand-tuned numbers gave it the smallest coupling. The
transcript map puts LC 4.3x VTA. It disagrees with [Erbs2015], the one
mouse whole-brain mu-opioid *protein* atlas, which scores VTA ++ against
LPAG +/++ and LC +, i.e. the reverse order by one tier. Both are recorded
as Oprm1 claims with their citations and neither is used to adjust the
other; what the disagreement bounds is how much of the analgesia/sedation
split is data, and the answer is "a 4.3-fold transcript spread against one
semiquantitative protein tier".

The plane policy is worth something here, unlike in P4.2's correlation: LC
above VTA survives all fourteen plane x combine policies, but PAG above VTA
does not - the sagittal experiments put PAG at 0.82-0.90 of VTA where the
single whole-brain coronal experiment puts it at 2.52. The scenario uses
`coronal-else-sagittal`, i.e. the coronal experiment, because the sagittal
pipeline processes one hemisphere only. `results/receptor_map_validation.txt`
prints the whole table.

### A correction the substitution forced: one mechanism gain, not three

Factoring density out made the old rationale for LC's smaller gain
inspectable, and it was double-counting. `Simulation.rhs` computes
`drive[idx] += gain * activation`: `coupling` is added to a node's *own*
input, and the amplification that reaches `VTA_DA` and `PAG_Output` one
synapse later is carried by the connectivity weight (-2.5 in both cases).
"LC's activation is not amplified" is therefore a statement about the
readout - which `arousal_index_readout()` already expresses by reading LC's
own rate - and not about how much receptor activation moves LC's drive.
Folded into the gain as well, it was counted twice.

With that removed, nothing distinguishes the three nodes at this layer
except receptor density, and no data in reach distinguishes per-receptor
efficacy between a VTA interneuron, a PAG interneuron and an LC
noradrenergic neuron. So there is one illustrative gain and the map
supplies all three ratios. Asserting a per-cell-type efficacy difference on
top of a measured density difference would have been inventing a number.

### The scale of that gain, and the P3.4 lesson it ran into

rho is dimensionless and its normalization is a choice, so the gain's scale
is not determined by the data. Setting it by preserving VTA_GABA's old
-3.0 was the first thing tried, and it fails in an informative way: PAG's
coupling follows the density to -7.57, which drives the PAG interneuron out
of the sensitive part of its sigmoid, and under P3.4's rest-balanced
baselines the naloxone reversal of `analgesia_index` on the ported
connectome network falls to **1.9e-4**, below the suite's 1e-3 directional
margin. That is P3.4's saturation lesson reappearing under *drug drive*
rather than at rest, which is a case P3.4's procedure does not cover
because it balances the network at rest.

So the scale is set by a rule instead: leave the geometric mean of the two
interneuron-motif couplings where the hand-tuning put it, `sqrt(3.0 * 2.5 /
2.5227) = 1.724`, rounded to **-1.72**. LC is excluded from the rule
because its density moved most and including it would let the least
constrained node set the scale for the other two.

The value is not load-bearing, and that is measured rather than asserted.
Sweeping the scale over the four network-by-baseline combinations (hand-
built and ported, `hand_built` and `rest_balanced` baselines) x three
readouts x agonism and reversal, every qualitative result survives for any
scale in roughly [1.3, 2.0]. The tightest directional margin at -1.72 is
2.7e-3, against **2.6e-3 for the hand-written map itself**: the
substitution does not cost margin, it moves where the margin is.

A note on what this is and is not. Re-tuning an explicitly illustrative,
hand-set number after changing the factor it multiplies is what the
illustrative label has always meant for these gains, and the rule and the
surviving interval are both written down. It would have been tuning if a
tolerance had been widened, if a test had been weakened, or if the value
had been picked by searching for one that passed; none of those happened,
and the failure that motivated it is reported with its number.

### What runs offline, and the seven regions that made that possible

Every one of the eighteen claims runs in the offline gate. That needed
seven more structures in the shared fixture ontology - CA1, CA3, DG, MH,
PVT, SNc, CBX - which went into a new `ANATOMY_ONLY_REGION_IDS` alongside
P4.2's `AUTORADIOGRAPHY_ONLY_REGION_IDS`, and the expression fixtures were
recut over the enlarged set from the same verified bytes. The ontology went
from 61 structures to 69 (five new regions, since SNc and CBX were already
there for P3.2's sign table, plus their three ancestors CA, EPI and MTN).
The Knox matrices are byte-identical, because these are ontology entries
with no matrix row.

Splitting the two tuples rather than growing one was deliberate. P4.2's
test asserts that every region it added is a region it compares, and
widening that to the union would have made it assert nothing; now each task
owns its block and P4.3 has the converse test, that every region it added
is named by a claim.

### What this task changes, and what it does not

- **`scenarios.py` now imports the expression package.** P4.1 and P4.2 were
  both additive and said so; this is the task that spends that. The import
  is lazy and inside the function, and it reads the committed fixture by
  default, so no existing test acquired a dependency on the gitignored
  cache.
- **One illustrative number changed value and one changed shape.** The
  multi-region mu-opioid coupling is now `-1.72 * rho`, and its three
  per-node gains became one gain plus a measured map. The illustrative
  label stays on the gain, because nothing calibrated it; the density
  factor is data and carries P4.2's assumption with it.
- **No other parameter moved, no other scenario touched, no tolerance
  widened.** `mu_opioid_coupling()` (the single-region VTA scenario, and
  the one P2.2 fitted a posterior around) is untouched, as are the D2 and
  GABA-A scenarios.
- **The headline result is a negative one and is reported as the
  headline**: seventeen of eighteen claims behaved as their records say,
  the eighteenth is a 0.6% difference between two regions at the floor of
  the map, and the validation rules out ingestion defects while ruling out
  nothing about the mRNA-to-protein step.
- **35 net new test items**: the two P4.3 test files collect 42 where the
  pre-existing multi-region suite collected 7, 5 of them needing the
  gitignored cache. Every one of the eighteen claims runs in the offline
  gate, and every qualitative result of the multi-region scenario is now
  asserted under both density maps.
- **Two existing tests changed, both because a number they pinned moved
  for a stated reason.** `test_a_recruited_node_leaves_the_rail_and_a_dead_
  one_does_not` no longer lists VTA_GABA among the nodes the agonist rails,
  because at -1.72 it lands at sensitivity 0.28 rather than 0.002; the
  property that test exists for, no node saturated both at rest and under
  drug, is unchanged, and the test now also asserts VTA_GABA is driven down
  without being railed. `test_the_fixture_ontology_covers_the_offline_
  comparison` now iterates P4.2's own five ontology-only regions instead of
  the union of P4.2's and P4.3's, because the union would have made it
  assert nothing.

---

## Phase 5: serotonin 5-HT1A and 5-HT2A (P5.1)

### The task, and where its difficulty actually was

P5.1 asks for two receptor systems with literature Ki, kon, koff and
efficacy for serotonin itself and for four representative ligands
(8-OH-DPAT, WAY-100635, DOI, ketanserin), a scenario showing agonism,
antagonism and reversal, and the 5-HT1A autoreceptor's role documented as a
Phase 6 dependency. The new code is `src/neurocascade/serotonin.py`, its
tests are `tests/test_serotonin_scenario.py`, and the worked example is
`examples/serotonin_raphe_cortex_circuit.py`.

The difficulty was not the circuit and it was not the engine: this task
needed no engine change at all, for the same reason the multi-region
mu-opioid track did not. Two `ReceptorSystem`s on one shared
`CircuitNetwork` is what `Simulation(network, receptor_systems=[...])`
already does, and a receptor whose activation drives two nodes is what a
two-key `coupling` dict already does. The difficulty was the sourcing, and
specifically that the four ligands' rate constants are published in four
different states of completeness. Three derivations were needed rather than
one, and each is stated at the ligand it applies to:

1. **Both constants measured** - use them. 8-OH-DPAT's kon at 5-HT1A
   ([Satala2016]) and serotonin's kon and koff at 5-HT2A ([Denomme2024]).
2. **koff and an equilibrium affinity measured, kon missing** -
   `kon = koff / Kd`, the mu-opioid pattern. WAY-100635 ([Parkel2006]).
3. **Only an equilibrium affinity measured** - share the site's one
   measured kon and set `koff = kon * Ki`, the GABA-A pattern. Serotonin at
   5-HT1A, DOI and ketanserin at 5-HT2A.

Derivation 3 is the assumption in this task, and it is the same one the
GABA-A section makes, but better anchored. There, `kon_shared = 0.2` was
chosen to match the order of magnitude of the mu-opioid numbers and was
sourced from nothing. Here the shared kon is a *measured association rate
constant for a ligand binding the same site in the same kind of
preparation*: 0.34 nM^-1 min^-1 at 5-HT1A from [Satala2016], 0.0252
nM^-1 min^-1 at 5-HT2A from [Denomme2024]. What is assumed is that the
other ligands at that site associate at a comparable rate, not what that
rate is. The full source list, with species, preparation, radioligand and
temperature per number, is in `serotonin.py`'s module docstring and is not
repeated here.

### Two sources that carry most of the weight, and why they were preferred

**[Satala2016]** (Satała G et al., Mol Neurobiol 2016;53:6869-6881,
PMC5104769) is a paper about zinc, and its 5-HT1A pharmacology is the
control arm. That control arm happens to be exactly the [Volpe2011]-shaped
thing the D2 section looked for and could not find: one preparation (human
5-HT1A in HEK293), one radioligand ([3H]-8-OH-DPAT), and in it both a
saturation Kd for the agonist (2.9 nM), its association rate constant
(3.4e8 M^-1 min^-1), its dissociation rate constants, and serotonin's Ki
(8.7 nM) measured against it. Two of the three 5-HT1A ligands are therefore
directly comparable rather than merely each-trustworthy-alone, which is the
guarantee the dopamine/haloperidol pair does not have.

**[Denomme2024]** (Denomme N et al., Mol Pharmacol 2024;106:92-106) is a
paper about a novel antagonist, and again the reference ligand is what this
project needed: serotonin's kon and koff at human 5-HT2A measured by
NanoBRET *in live cells*. It is the only 5-HT2A binding-kinetics
measurement found for any ligand in this set, and being a live-cell
measurement is what makes it the anchor for the affinity-state problem
below.

Both were preferred over the obvious alternatives (a PDSP-style Ki table, a
review's summary table) because the derivations above need rate constants
and an assay description, not an affinity number stripped of its context.

### The affinity state problem, which is the largest real uncertainty here

[McKenna1989] measures serotonin's affinity for 5-HT2A twice in the same rat
cortex: **3.0 nM** against the agonist radioligand [125I]DOI and **750 nM**
against the antagonist radioligand [3H]ketanserin. That 250-fold spread is
not a discrepancy to be averaged away. It is the two-state behaviour of an
agonist at a G-protein-coupled receptor: the agonist radioligand labels the
G-protein-coupled high-affinity state, the antagonist radioligand labels the
whole population. The same split shows up for DOI itself, 0.71 nM at cloned
human 5-HT2A against [125I]DOI ([Nelson1999]) versus 19.05 nM in rat cortex
against [3H]ketanserin ([Seggel1990]).

`ReceptorSite` has one pool and one affinity per ligand, so one of them has
to be chosen, and choosing is a modelling decision that changes results by
more than any other number in this task. The choice made, and the three
reasons for it:

- Serotonin takes its **live-cell** value, Kd ~150 nM ([Denomme2024]).
  Intact cells contain the coupled and uncoupled states in whatever ratio
  they actually occur in, which is closer to the tissue the circuit layer
  is about than either membrane preparation is.
- It is the only affinity in the set that arrives with both rate constants
  measured, so choosing it means the reference ligand of both scales needs
  no derivation at all.
- It lands *between* the two membrane values, 3.0 and 750, which is what a
  mixed-state population should do. That is a consistency check across
  three independent labs, not a preference.

DOI and ketanserin then take antagonist-radioligand membrane values, so the
5-HT2A set is mixed-format, exactly the flag the D2 section raised for
dopamine-versus-haloperidol. The consequence is bounded rather than waved
at: with serotonin at 150 nM the model's ketanserin/serotonin affinity ratio
is 150, where an all-[3H]ketanserin set makes it 750. So
`ligands_5ht2a(affinity_state="antagonist-radioligand")` builds that second
set, and **every qualitative result is asserted under both** - the same
discipline P4.3 used when it kept the hand-written mu-opioid map alongside
the atlas-derived one. Nothing in the agonist, antagonist, reversal or
opposite-sign results depends on which state is chosen; what depends on it
is only how much ketanserin it takes to displace a given serotonin
concentration, and no result in this task turns on that number.

### Three findings, none of them assumed going in

**WAY-100635 is a nearly irreversible antagonist, and that inverts the
mu-opioid reversal story.** [Parkel2006] measures koff = 7.8e-3 min^-1 in
rat hippocampal membranes and remarks that equilibrium is not reached before
7.5 h of incubation. That is a dissociation half-life of 89 minutes, ~130x
slower than the agonist it competes with here. Track A's finding about
naloxone was that the antagonist has the *faster* off-rate and reverses only
by molar excess; 5-HT1A is the opposite case, where the antagonist reverses
because it does not let go. The two facts are visible in the same codebase
now, which is worth more than either alone. Independent corroboration comes
from a different kind of experiment entirely: [Forster1995] reports
WAY-100635 is an *insurmountable* antagonist at high concentrations in
guinea-pig ileum, which is what a residence time that long looks like from
the functional-pharmacology side.

**The 5-HT1A/5-HT2A affinity split for serotonin is large, and it is
measured.** 8.7 nM at 5-HT1A ([Satala2016]) against ~150 nM at 5-HT2A
([Denomme2024]), two independent labs, a ~17-fold difference. Every
statement this module makes about which receptor a given serotonin
concentration engages first follows from those two published numbers and
from nothing hand-set. It is the one prediction here that is entirely
free of illustrative parameters, and it is tested as such.

**Both selective agonists are strong partial agonists, not full ones.**
[NewmanTancredi2003] ran one [35S]GTPgammaS panel in rat hippocampal
membranes and reports, relative to 5-HT = 100: (+)8-OH-DPAT 85,
WAY100,635 0. [Cussac2008] reports DOI at 81.3% of the serotonin maximum in
h5-HT2A Gq/11-coupled calcium mobilisation. So two of this task's four
efficacies come from a single panel with the antagonist measured in the same
experiment as the agonist, and the textbook shorthand ("8-OH-DPAT, the
prototypical 5-HT1A agonist") hides a 15% efficacy deficit that
`Ligand.efficacy` carries directly. This is the first place in the codebase
where `efficacy` is a fractional literature value rather than 1.0 or 0.0.

### The autoreceptor, and the Phase 6 dependency stated as a test

The 5-HT1A autoreceptor's function is a feedback loop: serotonin in the
raphe activates it, raphe firing falls, and *less serotonin is released in
the forebrain*. This model can express the first half of that sentence and
not the second, because `Simulation`'s ligand concentrations come from the
PK layer alone and no circuit rate has a path back into them.

Three consequences, all recorded in `serotonin.py` and one of them asserted:

- **The autoreceptor is an open loop here.** It suppresses the node that
  carries it and stops.
  `test_autoreceptor_is_an_open_loop_until_phase_6` doses the model with
  serotonin plus 8-OH-DPAT, checks that the raphe node is suppressed by
  ~0.4, and checks that the serotonin concentration and occupancy at the
  cortical 5-HT2A pool are unchanged to within 1e-4 relative. Closing the
  loop in P6.1/P6.2 must break that test, which is why it is written as an
  assertion rather than a paragraph. (The residual is ~2e-6 and is
  integrator step-size selection, not physics: RK45 picks one step for the
  whole shared state vector, so adding a ligand perturbs the last digits of
  every other trace. That is why the tolerance is 1e-4 and not zero.)
- **No raphe -> cortex edge is wired.** In a Wilson-Cowan network a weight
  is a fixed-sign synaptic drive, and the raphe's influence on cortex is not
  that: it is receptor-mediated and of both signs at once, 5-HT1A inhibitory
  and 5-HT2A excitatory on the same pyramidal cell ([AmargosBosch2004],
  [Santana2017]). Representing it as a signed weight would double-count the
  receptor layer, which is the layer this codebase exists for. The
  `DRN_5HT` node is therefore unconnected, the same shape as `LC` in the
  multi-region mu-opioid scenario and for a related reason.
- **An antagonist given alone is inert here and is not inert in vivo.**
  Real WAY-100635 has effects alone because real raphe neurons carry tonic
  autoreceptor activation to remove. Supplying that tone is P6.2. The test
  that asserts inertness says so in its comment, so the day P6.2 lands the
  expectation is already written down.

One smaller finding fell out of the same reading. `expression/mismatch.py`'s
docstring named `Htr1a` as a `partial` terminal-field-mismatch case, and no
record was ever written for it. It is not one: [Riad2000] finds 5-HT1A
protein on cell bodies and dendrites in *both* its raphe autoreceptor and
its hippocampal heteroreceptor role, and not on axon terminals, which is
precisely the case the transcript-to-density proportionality assumption is
built for. (5-HT1B, in the same paper, is the opposite case and would belong
in that registry if the model used it.) The docstring is corrected in this
commit rather than inherited, and it now names the four genes that actually
have records.

### What the circuit is, and the two rules that set its illustrative values

Three nodes. `DRN_5HT` (dorsal raphe serotonin neurons, somatodendritic
5-HT1A autoreceptor, [Sprouse1987]); `PFC_Pyramidal` (layer V prefrontal
pyramidal neurons, carrying *both* postsynaptic 5-HT1A and 5-HT2A, which is
the anatomy [AmargosBosch2004] reports); `PFC_GABA` (local interneurons,
carrying no receptor, providing the recurrent inhibition that makes the
pyramidal response finite).

Every weight, tau, gain and baseline in it is illustrative, and "illustrative"
here does not mean "tuned until the tests passed". Two stated rules fix them:

1. **Baselines put every node at rate 0.5 at rest**, the maximum-sensitivity
   point `stability.MAX_SENSITIVITY_RATE` derives. Given the two loop
   weights that fixes `baseline_input` at +0.45 and -0.4 exactly; nothing is
   left to choose.
2. **The loop weights satisfy P3.4's global-uniqueness certificate.**
   `contraction_number = sigma_max(diag(g/4) W)` is 0.9 < 1, so this
   network has exactly one fixed point and every trajectory converges to it
   whatever the taus. `stability.assess` is called on it in the tests rather
   than the property being claimed in prose.

This is the first hand-built scenario in the codebase constructed to carry
P3.4's certificate from the start, instead of being checked against it
afterwards.

The two mechanism gains (`HT1A_MECHANISM_GAIN = -1.0`,
`HT2A_MECHANISM_GAIN = +1.0`) are illustrative and follow their own stated
rule: activation is a dimensionless fraction in [0, 1], so magnitude 1 means
a fully occupied receptor bound by a full agonist shifts its node's input by
one unit, which at gain 4 carries the node most of the way across its
sigmoid. Their signs are not free - 5-HT1A is Gi/o-coupled and
hyperpolarizing, 5-HT2A is Gq/11-coupled and depolarizing - and their equal
magnitudes are a consequence of the rule rather than a claim that the two
receptors are equally powerful. P2.3 already established that only the
product of a coupling gain and a connection weight is identifiable even when
behavioural data exists, so no behavioural anchor was sought for these: Phase
5 is a receptor-layer task and Phase 7 is where endpoints get fitted.

One gain serves both of 5-HT1A's target nodes, following P4.3's correction:
the same Gi-coupled receptor reducing the drive of the cell that carries it,
with no data in reach distinguishing its per-receptor efficacy between a
raphe serotonin neuron and a cortical pyramidal neuron. The density
difference between those regions is the missing `rho` factor of P4.2's
`coupling = mechanism_gain * rho`; see the next section for why it is not
supplied yet.

### What was deliberately not built

- **5-HT2A on cortical interneurons.** It is really there ([Willins1997]),
  and it is why [Puig2003] finds intravenous DOI exciting 21 of 56 and
  inhibiting 17 of 56 identified mPFC pyramidal neurons. Adding it needs the
  ratio of the interneuron gain to the pyramidal gain, and in this network
  that ratio sets both the size and, in the small-signal regime, the sign of
  the cortical response. Linearized at rest the response is
  `(u_pyr - w * u_gaba) / (1 + w * w')` with `w = 0.9`, `w' = 0.8`, so it
  changes sign once the interneuron gain exceeds `1/w = 1.11` times the
  pyramidal one; running the model at a 1 nM DOI dose confirms the flip
  between ratios 1.0 and 1.5. At a saturating dose the interneuron rails
  and the response stays positive but is compressed from +0.26 to +0.04 as
  the ratio goes from 0 to 2. So the ratio is load-bearing over the whole
  range, no source found supplies it, and supplying one would be choosing
  the sign of a published-as-mixed result by hand. The interneuron node
  exists and carries no drug target.
- **An atlas-derived density factor.** `Htr1a` and `Htr2a` are already in
  `expression/genes.py` tagged P5.1, and `scenario_maps.scenario_coupling`
  is one call away. It is not used because the committed expression fixture
  carries `Htr2a` and not `Htr1a`, so a density-derived 5-HT1A map would make
  every test in this module depend on the gitignored 7.5 MB cache - the
  precise failure mode P4.3's fixture discussion set out to avoid. The
  mechanism gains are already stated as the `mechanism_gain` half of P4.2's
  factorization, so the upgrade is additive when an `Htr1a` fixture slice
  lands.
- **A second 5-HT1A efficacy for the autoreceptor.** [NewmanTancredi2003]'s
  85% is measured at postsynaptic hippocampal receptors; 8-OH-DPAT is widely
  described as behaving as a *full* agonist at somatodendritic
  autoreceptors. That difference is receptor reserve - the same intrinsic
  efficacy producing a maximal response where receptors exceed the G
  proteins needed - and receptor reserve is not representable at all in a
  model whose activation term is linear in occupancy. Using 85% everywhere
  understates the autoreceptor response, which is the honest direction to be
  wrong in; the alternative is asserting a second, unmeasured efficacy.
- **An inverse-agonist ketanserin.** [Egan1998] shows competitive 5-HT2A
  antagonists reduce basal signalling at a constitutively active receptor,
  i.e. they are inverse agonists rather than neutral. `Ligand.efficacy`
  supports negative values, but `ReceptorSite.activation` is
  `sum(occupancy * efficacy)` with no constitutive term, so a negative
  efficacy would drive the node *below* its drug-free baseline, which is not
  what an inverse agonist does to a receptor with no constitutive tone.
  Ketanserin is modelled with efficacy 0.0 and that zero is labelled as the
  one modelling simplification among this task's efficacies, rather than as
  a measurement. Adding constitutive activity is an engine extension with
  its own literature requirement (how much constitutive activity, per
  receptor, in vivo).

### Searched and empty

- **Rodent pharmacokinetics for 8-OH-DPAT, WAY-100635, DOI and ketanserin.**
  Searched Europe PMC for each compound crossed with "pharmacokinetics",
  "half-life", "plasma" and "brain concentration" in rat and mouse. The hits
  are reviews of other drugs, PET occupancy studies that report occupancy
  over time without a plasma curve, and papers using these compounds as
  tools without measuring their disposition. So all four share ONE
  illustrative elimination rate constant, `TOOL_COMPOUND_KE = ln(2)/30 min`,
  rather than four separately invented ones: four guesses would assert a
  clearance *ordering* that nothing supports, where one asserts only that
  all four are still on board across the scenario's window. Phase 1's
  dose-to-brain bridge is what a real dataset would plug into.
- **Binding kinetics for DOI and ketanserin at 5-HT2A, and for serotonin at
  5-HT1A.** Searched for "[3H]ketanserin" with "association"/"dissociation
  rate constant", for 5-HT2A "residence time" with NanoBRET/TR-FRET, and for
  the 5-HT2A entries of [Sykes2017]-style antipsychotic kinetic panels. The
  classic [3H]ketanserin characterisation (Leysen JE et al., Mol Pharmacol
  1982;21:301-314) does report association and dissociation kinetics, but it
  is paywalled with no abstract in Europe PMC and its rate constants could
  not be read; nothing citable was obtained from it. Hence derivation 3 for
  those three ligands.
- **Serotonin's own clearance rate.** The same gap dopamine has in the D2
  section, for the same reason: 5-HT in the extracellular space is cleared
  by SERT-mediated reuptake, a saturable Michaelis-Menten transporter
  process on a sub-second-to-second timescale, not first-order elimination.
  `SEROTONIN_KE = 5.0 min^-1` is an explicit flagged proxy that preserves
  the only property the dynamics need - transmitter clears far faster than
  any dosed drug - at a timescale the integrator resolves. The real quantity
  belongs to P5.8 (SERT) and P6.2 (endogenous tone).

### What this task changes, and what it does not

- **One new module, and no change to any existing model file.**
  `serotonin.py` is additive; `engine.py`, `receptor.py`, `network.py` and
  `scenarios.py` are untouched. The only edit to pre-existing code outside
  the package export is the `expression/mismatch.py` docstring correction
  above, which changes no behaviour and no number.
- **No existing test changed, none was skipped, marked slow or weakened,
  and no tolerance was widened.** 24 new test items, all in the offline
  gate, running in ~40 s.
- **Six ligands' binding constants and efficacies are literature-sourced.**
  Four efficacies come from functional panels rather than being assumed to
  be 1.0 or 0.0, and two of the four are fractional.
- **The circuit and the pharmacokinetics remain illustrative and are
  labelled as such**, as in every scenario before Phase 7. What is different
  is that both sets of illustrative values now follow stated rules -
  maximum-sensitivity baselines, the P3.4 certificate, unit-magnitude
  mechanism gains, one shared tool-compound `ke` - so that a later
  calibration replaces a documented placeholder rather than an arbitrary
  one.

---

## Phase 5: dopamine D1 and the direct pathway (P5.2)

### The task, and the sentence in the brief that set the difficulty

P5.2 asks for the D1 receptor, the direct basal-ganglia pathway, and one
sentence more: "direct and indirect pathways must coexist on one shared
striatal network so that a drug acting at both is simulated correctly
rather than as two separate models." The new code is
`src/neurocascade/dopamine.py`, its tests are
`tests/test_dopamine_d1_scenario.py`, and the worked example is
`examples/dopamine_direct_indirect_pathways.py`.

That sentence is the whole design constraint, and it is a real one rather
than a stylistic preference. The striatum's two projection-neuron
populations are not two circuits: they are two populations of one nucleus,
receiving the same corticostriatal afferents, and their outputs reconverge
on the same basal-ganglia output nucleus two synapses later. Two separate
`Simulation`s would get the receptor layer right and the reconvergence
wrong - the second arm would never see what the first one did to the
shared output. Once both arms are on one `CircuitNetwork`, the interesting
result (a D1 agonist and a D2 antagonist partly cancelling at one readout)
falls out of the wiring instead of having to be asserted.

As with P5.1, no engine change was needed. Two `ReceptorSystem`s on one
shared network is what `Simulation(network, receptor_systems=[...])` has
done since the combined mu-opioid/D2 scenario; what is new here is that
the two systems are wired into the SAME anatomical structure with a real
convergence point, where that earlier scenario was a deliberately
block-diagonal union of two unconnected circuits.

### The pair of measurements this task turned on

The D1 binding constants come from **[Sahlholm2025]** (Sahlholm K,
Svensson P, Malo M, Andersson DR, Betari N. ACS Chem Neurosci
2025;16(13):2502-2512, doi:10.1021/acschemneuro.5c00270, open access
PMC12232300): human D1 co-expressed with GIRK1/4 in Xenopus oocytes,
two-electrode voltage clamp at 22 C, with `kon` read off the slope of
`k_obs` against agonist concentration and `koff` off the current
deactivation on washout. Table 1 gives five agonists; two are instantiated
(dopamine, dihydrexidine) and all five are recorded in the module
docstring, because a panel is worth more than a single row.

What makes it the right source is not that it exists but *whose* it is.
This codebase's D2 numbers already come from **[Agren2021]**, which is the
same laboratory, the same GIRK-kinetics method, the same expression system
and the same receptor host species; the D1 paper says in as many words
that its approach is "analogous to that earlier described for dopamine D2R
agonists" and cites [Agren2021] for it. So the D1/D2 comparison below is
the [Volpe2011]-style same-preparation guarantee that the D2 section
explicitly could not get for dopamine-versus-haloperidol, arriving two
sections later for the pair that actually needed it.

**Dopamine binds D2 about 530x more tightly than D1.** Kd 2.03 nM at D2
against 1079 nM at D1, both computed as `koff / kon` from the two papers'
own rate constants. That is the quantitative content of the textbook
statement that tonic dopamine engages D2 and phasic dopamine is needed to
engage D1, and this module gets it entirely from published numbers -
nothing about it is hand-set, which makes it the one prediction here free
of illustrative parameters. Set against [Garris1994]'s ~0.25 uM
extracellular transient from a single stimulus pulse, it says the phasic
peak sits near D1's Kd and three orders of magnitude above D2's, which is
the only regime in which the two arms of this circuit can be driven
differentially at all.

Two things are recorded alongside it because they bound how much weight it
can carry, and one of them was nearly missed:

- **The same two papers' EC50s differ by 31-fold, not 530-fold** (622 nM
  at D1, 20 nM at D2). The gap between the two ratios is receptor reserve
  and signal amplification. This model's `activation = sum(occupancy *
  efficacy)` is linear in occupancy with no amplification term, so it
  reproduces the 530-fold OCCUPANCY separation and not the 31-fold
  POTENCY separation. Which of the two a behavioural endpoint should be
  fitted against is a Phase 7 question; it is flagged rather than settled,
  and it is the same class of limitation as P5.1's receptor-reserve note
  about 8-OH-DPAT at the autoreceptor.
- **[Agren2021] co-expressed RGS4 and [Sahlholm2025] did not.** The D2
  paper says why: RGS4 accelerates GTP hydrolysis so that "the time course
  of the GIRK current more closely follows the time course of D2R agonist
  occupancy". RGS4 acts on Galpha_i/o and D1 reaches GIRK through Gs, so
  the D1 measurement has no equivalent. The D1 `koff` is therefore the
  more likely of the two to be rate-limited by G-protein cycle steps
  rather than by unbinding. If it is, the true D1 `koff` is FASTER and the
  true D1 Kd LARGER, which *widens* the gap rather than narrowing it. The
  caveat points away from the finding being an artefact, which is the only
  reason it is safe to lean on the number at all.

### SCH-23390, and a derivation checked against a third species

The D1 antagonist needed derivation 3 (share the site's one measured
`kon`, set `koff = kon * Kd`), because no association or dissociation rate
constant for SCH-23390 could be obtained. `koff = 0.0073395 * 0.53 =
0.00389 min^-1` gives a dissociation half-life of 178 minutes, which is a
large claim to get out of an assumption.

It is checked rather than asserted. [Reader1989] reports that in rabbit
cortical and striatal membranes "the dissociation kinetics were extremely
slow and only around 40-60% of the binding was displaced two hours after"
dissociation was initiated. That brackets the half-life at 91-163 minutes.
The derived 178 minutes is ~9% beyond the slow end: close enough that
derivation 3 is not contradicted, far enough that this is a sanity check
and not a calibration. No number is taken from [Reader1989], because it is
a third species and rule 2 is rule 2; it is used as a bracket, the test
says so, and the test asserts the bracket's own arithmetic so the claim
cannot rot.

That result puts SCH-23390 next to P5.1's WAY-100635 and completes a set
worth naming. This codebase now carries three distinct
antagonist-reversal mechanisms, all of them measured or derived rather
than chosen: naloxone reverses fentanyl by molar excess *because* its
`koff` is faster (2.375 min^-1); WAY-100635 (0.0078) and SCH-23390
(0.00389) reverse their agonists by not letting go. A model whose
antagonists all behaved the same way would have been hiding that.

### Which ligand, and the trade that decided it

The D1 agonist in the locomotor literature is SKF-81297 or SKF-38393. The
D1 agonist in this module is dihydrexidine, and that is a deliberate
trade. No binding-kinetics measurement was found for either SKF compound,
so both would need derivation 3 off a Ki measured in a different assay
format; dihydrexidine arrives with `kon`, `koff` AND a fractional efficacy
(0.808 of dopamine's Emax) measured in the same experiment, in the same
oocyte, as the reference agonist. P5.2 is a receptor-layer task, so the
ligand with the better-sourced constants won over the ligand with the
better-known behaviour. That trade inverts the moment Phase 7 needs a
compound with a published dose-response curve, and the module says so.

Dihydrexidine's 0.808 is the second fractional literature efficacy in the
codebase, after P5.1's 8-OH-DPAT (0.85) and DOI (0.813), and it hides the
same kind of shorthand: "dihydrexidine, the full D1 agonist" is a ~19%
overstatement in the assay that measured it.

### The sign of the D1 gain, which is the one thing chosen rather than read

D1 is Gs/olf-coupled, so the receptor-level sign is not in doubt. The sign
of its effect on a medium spiny neuron's FIRING RATE is, and the source
that establishes the effect also establishes the ambiguity:
[HernandezLopez1997] measures both signs in the same cells, inhibition of
evoked discharge at ~-80 mV and enhancement at ~-55 mV, the switch being
an L-type Ca2+ conductance only available from depolarized potentials.

A Wilson-Cowan node has no membrane potential and cannot carry a
state-dependent sign, so one had to be picked. The positive one is used,
for the reason that paper itself gives: the up-state is the state an MSN
is in when corticostriatal input is driving it, and this network's D1 node
is driven by exactly such an input and sits at rate 0.5 rather than at
rest. This is the single place in the module where a sign was chosen
rather than read off a measurement, and it is labelled as such at the
constant, in the docstring and here.

### The circuit, and three rules that make "illustrative" mean something

Seven nodes: `Cortex_M1`, `Striatum_D1_MSN`, `Striatum_D2_MSN`, `GPe`,
`STN`, `GPi`, `Thalamus`. Seven edges, and no others:

    Cortex_M1 -> Striatum_D1_MSN -| GPi -| Thalamus          DIRECT
    Cortex_M1 -> Striatum_D2_MSN -| GPe -| STN -> GPi        INDIRECT

`Striatum_D2_MSN`, `GPe`, `STN`, `GPi` and `Thalamus` keep the names the
existing five-node indirect-pathway scenario uses, so the calibrated D2
receptor site and PK models drop in unchanged and
`scenarios.locomotor_activity_readout()` scores the new network without
modification. That last point is not cosmetic: "opposing effects on the
same locomotor readout" is only a claim about the circuit if it is
literally the same readout function on the same node, and a test asserts
that the two modules' readout weights are equal objects rather than
lookalikes. `scenarios.basal_ganglia_indirect_pathway_network()` itself is
untouched.

Every weight, tau, gain and baseline is illustrative, and three stated
rules fix them so that "illustrative" does not mean "tuned until the tests
passed":

1. **Unit weights.** Every connection has magnitude 1.0 and the textbook
   sign. At the operating point a logistic node's slope is exactly 1, so a
   unit weight makes the per-synapse loop gain exactly 1 - a cascade that
   neither attenuates to nothing over five hops nor amplifies out of the
   sensitive region. Equal magnitudes are also what make the two arms
   comparable BY CONSTRUCTION, which is the point of the scenario: any
   asymmetry in a D1-versus-D2 result then comes from the topology and the
   published binding constants, not from a hand-set weight difference.
2. **Baselines put every node at rate 0.5 at rest.** Given rule 1 that
   determines all seven values exactly;
   `stability.balanced_baselines` reproduces them and a test asserts it.
   One value is worth naming: GPi's baseline comes out exactly 0.0, i.e.
   at rest the direct and indirect inputs to the output nucleus cancel.
   That is a consequence of rules 1 and 2, not a target.
3. **No node saturates under a saturating dose of a full agonist at
   either receptor.** This is what sets the mechanism gains to magnitude
   1.0, and it is the reason `scenarios.d2_coupling()`'s -3.0 is *not*
   reused: at magnitude 3 in this network the minimum sigmoid sensitivity
   under a saturating D1 agonist falls to 7.5e-4, an order of magnitude
   below `stability.SATURATION_FLOOR`, and a railed node stops responding
   to the other pathway. At magnitude 1 it is 0.199. The test asserts
   both halves, including the counterfactual - the rule has to be able to
   reject a value or it is not a rule.

The -3.0 was set against a five-node network with weight-2.0 connections
and is correct there; the honest reading is that a coupling gain is not
portable across networks because P2.3 established that only the *product*
of a gain and a weight is identified. Carrying the old number into a new
topology would have been carrying a calibration of a different system.

### Acyclicity, and why it is the better uniqueness argument here

The connectivity matrix is nilpotent: the graph is a directed acyclic
cascade. For a DAG the fixed point is unique and globally attracting for
free, by topological order - each node's steady rate is a function of its
upstream nodes' rates alone, so solving in order gives one and only one
solution whatever the taus.

That matters because P3.4's contraction certificate, which P5.1's
serotonin circuit was built to satisfy, **is not satisfied here**:
`kappa = sigma_max(diag(g/4) W) = 1.414`. Satisfying it would have meant
weights of 0.707 or below, which buys nothing this network does not
already have and costs the legibility of unit weights. The certificate is
sufficient, not necessary, and this is the first place in the codebase
where the distinction does real work. So the test asserts *both* that
`kappa >= 1` and that `W**7 == 0`, and runs `stability.assess` to confirm
convergence, uniqueness across five initial conditions, local stability
and non-saturation empirically as well. Writing down "the certificate
fails and here is the stronger argument that replaces it" is the point;
quietly rescaling the weights until `kappa < 1` would have hidden it.

The same observation applies retrospectively to every hand-built
basal-ganglia scenario in this codebase, all of which are cascades, and it
is why they were fine without a certificate nobody had computed for them
yet.

### One start-up detail that is not cosmetic

`Simulation.run` starts from zero rates by default. On a network built to
rest at 0.5 that means the first minutes of every trace are a start-up
transient rather than a drug effect, and comparisons between conditions
are then differences of two transients. `dopamine.rest_rates()` returns
the fixed point and every test and the example pass it as `rates0`, so the
drug-free trace is exactly flat to 1e-6 and every deviation is the drug. A
test asserts that flatness, which is what makes the small effects below
(the second-order cortical-drive residual, the segregation check) legible
at all.

### What the model says, and the result the brief asked for

All of these are qualitative direction checks, on an illustrative circuit,
against the direct/indirect model - not fits.

**The pathways oppose each other, tested as a statement about pathways.**
[Kravitz2010] is optogenetics, not pharmacology: exciting direct-pathway
MSNs increased locomotion, exciting indirect-pathway MSNs produced
freezing and bradykinesia. So it is tested by adding drive to each MSN
node directly with no ligand anywhere, which is as close as a rate model
gets to channelrhodopsin. Direct up raises the readout; indirect up lowers
it.

**A D1 agonist and a D2 antagonist move one readout in opposite
directions.** Against a 1000 nM dopamine background on the shared network,
1000 nM dihydrexidine raises the locomotor readout from 0.6055 to 0.8094
and 200 nM haloperidol lowers it to 0.5545. Both are the direction the pathway each
drug acts on predicts. This is the acceptance criterion, and it is one
`solve_ivp` call over one state vector.

**Given together they partly cancel**, landing between the two
single-drug results - the thing two separate models cannot produce,
because the cancellation happens at the shared GPi node and nowhere else.

**Cortical drive alone almost does not move the output.** A perturbation
of the shared corticostriatal input drives both MSN populations
identically, and their two paths to GPi have opposite net sign, so to
first order at the operating point the output nucleus does not move at
all: a 0.097 shift in both MSNs produces a 0.0024 shift in GPi, 2.5% of
it. The residual is second order, because the direct path passes through
one sigmoid and the indirect through three. This is a consequence of rules
1 and 2 rather than a claim about the brain, and it is stated that way -
but it is the reason the scenario is about dopamine acting
*differentially* on two populations rather than about drive.

### The one result readers expect to be opposite and is not

D1 and D2 agonism are **not** opposing on locomotion in this model, and
should not be. D1 activation excites the direct MSN, which inhibits GPi;
D2 activation inhibits the indirect MSN, which via GPe -| STN also ends up
inhibiting GPi. Opposite receptor-level signs, one direction at the output.
A test asserts exactly that - opposite signs at the two striatal
populations, one sign at GPi - because it is the kind of thing that would
otherwise get "fixed" by someone who remembered "D1 and D2 oppose each
other" without remembering what they oppose each other *at*. The
oppositions that are real here are between pathway activations
([Kravitz2010]) and between a D1 agonist and a D2 antagonist.

There is one further genuine D1-versus-D2 opposition, and this model
cannot express it: D2 autoreceptors on nigrostriatal terminals and on
SNc/VTA somata, where an agonist reduces dopamine release, which is the
mechanism behind the biphasic locomotor dose-response of D2 agonists. It
is blocked for exactly the reason P5.1's 5-HT1A autoreceptor was:
`Simulation`'s ligand concentrations come from the PK layer and no circuit
rate has a path back into them. P5.1's
`test_autoreceptor_is_an_open_loop_until_phase_6` already asserts that gap
in general form, so it is recorded here and not re-asserted.

### Searched and empty

- **Binding kinetics for SCH-23390 at D1.** Searched Europe PMC for
  "SCH 23390" crossed with "dissociation rate constant", "koff" and
  "residence time"; for D1 with "TR-FRET", "NanoBRET" and "association
  rate constant"; and through the Rinken group's D1 work, which is the
  same group [Parkel2006] came from in P5.1. Allikalt A, Kopanchuk S,
  Rinken A (Eur J Pharmacol 2018;839:40-46) report association and
  dissociation half-lives of 40 s and 70 s and Kd 5.2 nM - for the
  FLUORESCENT ligand BodipyFL-SKF83566, not for SCH-23390. Farrell CB,
  O'Boyle KM (Eur J Pharmacol 1994;268(1):79-88) is the obvious source,
  measures [3H]SCH 23390 dissociation from rat striatum, and is paywalled
  with an abstract that reports the *effect of dopamine on* k-1 without
  reporting k-1. Billard W et al. (Life Sci 1984;35:1885-1893) gives the
  saturation Kd (0.53 nM, rat striatum) that derivation 3 uses instead.
- **Rodent pharmacokinetics for dihydrexidine and SCH-23390.** Searched
  each compound crossed with "pharmacokinetics", "half-life", "plasma
  concentration" and "brain concentration" in rat and mouse. Nothing
  usable: dihydrexidine's hits are in-vitro pharmacology and reviews of
  tavapadon; SCH-23390's are studies using it as a tool at a stated mg/kg
  without measuring its disposition. Both therefore share ONE illustrative
  `ke`, deliberately the same number as P5.1's
  `serotonin.TOOL_COMPOUND_KE`, for P5.1's stated reason - two invented
  values would assert a clearance ORDERING nothing supports. A test
  asserts the two constants are equal so they cannot drift apart, and
  another asserts this module's dopamine `ke` equals the D2 set's, because
  one dopamine dose is applied to both systems by name and the alternative
  is one molecule with two elimination rate constants.
- **A locomotor dose-response usable as a circuit-layer target.** Not
  searched again: Track E and P2.2 already established the blocker for
  this whole drug class, which is that behavioural pharmacology reports
  mg/kg by IP/SC/PO and this codebase has no such kinetics for these
  compounds. Phase 1's bridge is where such a dataset enters.

### What was deliberately not built

- **A cortico-thalamo-cortical feedback edge.** `Thalamus -> Cortex_M1`
  exists anatomically, and adding it turns the acyclic cascade into a loop
  whose gain nothing in reach fixes - and the uniqueness argument above goes with it,
  since `kappa = 1.414` means the contraction certificate would not cover
  the loop either.
- **An atlas-derived density factor.** `Drd1` is already in
  `expression/genes.py` tagged P5.2, `Drd2` is already in the model, P4.3
  scored both as striatum-dominant, and the two mechanism gains are
  explicitly the `mechanism_gain` half of P4.2's
  `coupling = mechanism_gain * rho[region]`. Two measured facts stop the
  upgrade rather than one guessed one. First, `rho` is a per-REGION factor
  and here both receptors sit in the SAME region on two different CELL
  TYPES, which a regional ISH map cannot separate - a real limit of the
  P4.2 factorization at exactly this scenario. Second, the direct
  pathway's own anatomy is the terminal-field mismatch P4.3 already
  records as an expected FAILURE for `Drd1`: nigral D1 protein is on the
  terminals of striatonigral neurons whose somata, and so whose
  transcript, are in the caudoputamen, and P4.2 measured the resulting
  CP:SNr error at 32-fold. A transcript-derived `rho` for the direct
  pathway's output nucleus would be wrong by that factor, in a known
  direction. This scenario is therefore the one that would benefit most
  from a cell-type-resolved density source and is furthest from having
  one.
- **The connectome-ported version.** P3.3 established that the ported
  indirect pathway inverts, because Knox et al. give GPe a thalamic
  projection seven to nine times denser than GPi's and the Albin-DeLong
  diagram gives it none. That finding applies unchanged to the direct
  pathway's ported form and would need the cell-type-resolved striatal
  output P3.3's `Restriction` stands in for. This scenario is the diagram,
  and P3.3 already documents at length what the diagram is and is not.
- **Endogenous dopamine tone.** Without it an antagonist given alone is
  inert, as everywhere else here; real SCH-23390 and real haloperidol both
  suppress locomotion given alone. P6.2. The test that asserts inertness
  says so, so the day P6.2 lands the expectation is already written down.

### What this task changes, and what it does not

- **One new module, and no change to any existing model file.**
  `dopamine.py` is additive; `scenarios.py`, `engine.py`, `receptor.py`,
  `network.py` and `serotonin.py` are untouched. The only edit outside the
  new files is adding `dopamine` to the package exports.
- **The calibrated D2 parameter set is reused, not re-derived.**
  `d2_receptor_site_calibrated()` and `d2_pk_models_calibrated()` are the
  same objects; only the coupling gain differs, for the stated saturation
  reason. So the D1/D2 comparison is a comparison of two independently
  sourced parameter sets rather than of two sets fitted to make it come
  out.
- **No existing test changed, none skipped, marked slow or weakened, and
  no tolerance widened.** 21 new test items, all in the offline gate.
- **Three D1 ligands' binding constants and efficacies are
  literature-sourced**, one of the three efficacies fractional and
  measured, one antagonist `koff` derived by the codebase's stated
  derivation 3 and checked against an independent bracket.
- **The circuit remains illustrative and is labelled as such**, as in
  every scenario before Phase 7 - but its parameters now follow three
  rules with a rejection criterion attached to one of them, and the
  uniqueness argument for the network is written down and asserted rather
  than inherited.
