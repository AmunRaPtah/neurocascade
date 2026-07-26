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
  binding and PK have a literature-calibrated alternative (see below); D2
  now does too (see "Calibrated: D2" below). GABA-A does not yet, and no
  scenario's circuit `tau`/connection
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
