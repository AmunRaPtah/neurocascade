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
