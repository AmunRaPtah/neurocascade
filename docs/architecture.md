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
  binding and PK have a literature-calibrated alternative (see below); as of
  the GABA-A calibration pass described below, so does the GABA-A
  benzodiazepine site. D2 does not yet, and no scenario's circuit `tau`/connection
  weights/gains have a literature target — no dataset for VTA-NAc-PFC (or
  basal-ganglia, or amygdala) connection *weights* (as opposed to wiring
  direction/sign, which matches known circuit anatomy, see above) was
  identified. Calibrating the remaining scenarios, and longer-term fitting
  circuit parameters Bayesian-style the way the CCT model's 14 parameters
  were fit against a literature screen, is the next real piece of work, not
  a formality.
- Each scenario in v0 models one receptor site and one circuit in isolation
  (separate scenarios, not interacting ones). A real drug usually hits
  several receptor subtypes across several regions simultaneously, and a
  real circuit carries several receptor types at once; the classes here
  (`ReceptorSite`, `CircuitNetwork`, `coupling`, `connection_modulation`)
  are general enough to support combining them in one `Simulation`, but
  that hasn't been built or tested yet.

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
