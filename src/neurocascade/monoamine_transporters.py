"""P5.8: the monoamine transporters DAT, SERT and NET.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes. `km` and `ki` are nM, `vmax` is nM/min, `clearance_rate` is min^-1.

This module is to `transport.py` what `opioid.py` is to `receptor.py`: the
cited constants and the assembled systems. The mechanism, and the argument
for why a transporter is not a receptor, is in `transport.py`.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter): the Michaelis constant
and maximal velocity for dopamine and for serotonin uptake in MOUSE brain
tissue; the noradrenaline clearance rate constant; the drug-free basal
extracellular dopamine and serotonin concentrations in MOUSE brain; the
uptake-inhibition constants of cocaine, methylphenidate and amphetamine at
all three MOUSE transporters from one laboratory's one panel; the same three
drugs' constants in a second, independent mouse preparation; atomoxetine's
constants at all three human transporters from two laboratories; fluoxetine's
constant at the human serotonin transporter; and amphetamine's release
potency and releasing efficacy at all three transporters.

ILLUSTRATIVE: the basal extracellular noradrenaline concentration, which is a
unit scale rather than a measurement and which no result depends on (see
`NORADRENALINE_BASAL_UNIT_SCALE`); the absolute efflux capacity of a
saturating substrate (`RELEASE_CAPACITY`); the pharmacokinetics of all five
drugs, which share `TOOL_COMPOUND_KE` with every other P5 module; and the
demonstration circuit, which is `scenarios.basal_ganglia_indirect_pathway_
network()` reused through P3.4's stabilization procedure.

The sentence in the brief that sets the task
---------------------------------------------

"A transporter is not a receptor: blocking it raises synaptic transmitter
concentration rather than directly producing a signal."

That is a statement about model structure, and this module's first job is to
make it structural rather than rhetorical. `transport.py` gives a transporter
no efficacy, no occupancy state and no `coupling`; `engine.TransporterSystem`
gives it one state, the extracellular transmitter concentration; and the only
way anything downstream can respond to a transporter is by binding that
concentration. A test asserts that a `MonoamineTransporter` has no
`activation` method at all, so the distinction cannot decay into a naming
convention.

The prediction that carries no illustrative number
---------------------------------------------------

Because uptake is a Michaelis-Menten process and a blocker is competitive,
the drug-free and inhibited steady states of the transmitter pool differ by
EXACTLY the competitive-inhibition factor, with Km, Vmax and the basal
concentration all cancelling (the algebra is in `transport.py`):

    [transmitter]_ss / [transmitter]_basal  =  1 + [drug] / Ki

So the transporter layer's central quantitative prediction depends on ONE
measured constant per drug and on nothing illustrative whatever. What that
gives, for the three psychostimulants at the mouse transporters:

    drug              transporter   Ki (nM)   1 uM drug raises transmitter
    cocaine               DAT          490          3.04-fold
    cocaine               SERT         730          2.37-fold
    cocaine               NET          460          3.17-fold
    methylphenidate       DAT          260          4.85-fold
    methylphenidate       NET          170          6.88-fold
    methylphenidate       SERT      114370          1.01-fold
    amphetamine           NET          120          9.33-fold  (blockade only)

and the last row is the one that matters, because amphetamine's blockade is
NOT its main action. See below.

Substrate versus blocker, which is the second half of the brief
----------------------------------------------------------------

Cocaine, methylphenidate, fluoxetine and atomoxetine are pure uptake
blockers: they bind the transporter and are not carried across the membrane
([Rothman2012]'s definition, [Zhu2008]'s review, [HanGu2006]'s introduction).
Amphetamine is a SUBSTRATE. It is carried across, and the consequence is a
mechanism no blocker has: transmitter is counter-transported out of the
terminal ([Zhu2008]: "amphetamine acts by elevating the cytoplasmic DA
concentration, and thereby alters the DA gradient across the plasma
membrane, promoting reverse transport of DA and DA release").

In this module that distinction is two constants, not an adjective:

* every drug has a `ki` and therefore competes for the substrate site;
* only amphetamine has a `release_ec50` and a `release_emax`, and therefore
  only amphetamine contributes an efflux term.

`transport.MonoamineTransporter.release_drive()` is identically zero for any
combination of pure blockers at any concentration, which is what makes "a
blocker cannot release transmitter" a property of the code rather than of the
parameter values, and a test asserts it.

The measurement that makes the distinction quantitative, and it is a large
number: **amphetamine releases dopamine with an EC50 of 6.4 nM and inhibits
dopamine uptake with a Ki of 560 nM.** Its releasing action is 88-fold more
potent than its blocking action at the same transporter, from two different
laboratories' assays. At NET the ratio is 16-fold (7.4 against 120 nM) and at
SERT it is 12-fold (1960 against 23820 nM). `release_over_block_ratio()`
returns these, and a test asserts that modelling amphetamine as a blocker
alone would understate its potency at DAT by nearly two orders of magnitude.
That is the whole reason the brief says the distinction "needs to be
represented".

Where the constants came from
------------------------------

Uptake kinetics of the natural substrate, and the basal levels:

  [JohnJones2007] John CE, Jones SR. "Voltammetric characterization of the
                effect of monoamine uptake inhibitors and releasers on
                dopamine and serotonin uptake in mouse caudate-putamen and
                substantia nigra slices." Neuropharmacology.
                2007;52(8):1596-1605. doi:10.1016/j.neuropharm.2007.03.004
                (PMC2041899, open access). PMID:17459426. Male and female
                C57BL/6 MICE, 2-4 months, 400 um coronal slices, 32 C,
                oxygenated aCSF, fast-scan cyclic voltammetry at carbon-fibre
                microelectrodes on ENDOGENOUSLY released transmitter.
                Dopamine in caudate-putamen: Km 0.16 uM, Vmax 3.1 +/- 0.20
                uM/s. Serotonin in substantia nigra pars reticulata:
                Km 0.17 uM, Vmax 0.080 +/- 0.006 uM/s. Table I additionally
                gives apparent Ki for cocaine, methylphenidate, fluoxetine
                and amphetamine at both, in the same preparation; those are
                in `KI_MOUSE_SLICE` and are the second preparation this
                module compares against rather than the one it builds from.

  [Bosse2011]   Bosse KE, Maina FK, Birbeck JA, France MM, Roberts JJP,
                Colombo ML, Mathews TA. "Aberrant striatal dopamine
                transmitter dynamics in brain-derived neurotrophic
                factor-deficient mice." J Neurochem. 2011;120(3):385-395.
                doi:10.1111/j.1471-4159.2011.07531.x (PMC3385875, open
                access). MICE. Basal extracellular dopamine in
                caudate-putamen by ZERO NET FLUX microdialysis, wild-type:
                5.0 +/- 0.2 nM. Voltammetric Vmax in the same animals
                4.1 +/- 0.1 uM/s with Km fixed at 0.16 uM "based on an
                average literature value" -- i.e. an independent
                confirmation of [JohnJones2007]'s Km choice, and a 1.32-fold
                cross-laboratory check on its Vmax.

  [Mathews2004] Mathews TA, Fedele DE, Coppelli FM, Avila AM, Murphy DL,
                Andrews AM. "Gene dose-dependent alterations in extraneuronal
                serotonin but not dopamine in mice with reduced serotonin
                transporter expression." J Neurosci Methods.
                2004;140(1-2):169-181. doi:10.1016/j.jneumeth.2004.05.017.
                PMID:15589347. MICE, zero net flux microdialysis. Basal
                extracellular 5-HT in wild-type SERT(+/+): 2.8 +/- 1 nM in
                striatum, 1.4 +/- 0.4 nM in frontal cortex.

  [Park2012]    Park J, Bucher ES, Fontillas K, Owesson-White C, Ariansen JL,
                Carelli RM, Wightman RM. "Opposing catecholamine changes in
                the bed nucleus of the stria terminalis during intracranial
                self-stimulation and its extinction." Biol Psychiatry (the
                open-access record used here is PMC3311672, "In vivo
                voltammetry monitoring of electrically evoked extracellular
                norepinephrine in subregions of the bed nucleus of the stria
                terminalis"). Anaesthetized RATS, fast-scan cyclic
                voltammetry. Noradrenaline clearance half-life
                1.36 +/- 0.07 s in ventral BNST and 2.37 +/- 0.07 s in
                dorsomedial BNST. A clearance RATE, not a saturation
                constant: it fixes Vmax/Km and says nothing about either
                alone, which is why NET is built in `transport.py`'s
                first-order form.

Drug constants:

  [HanGu2006]   Han DD, Gu HH. "Comparison of the monoamine transporters
                from human and mouse in their sensitivities to
                psychostimulant drugs." BMC Pharmacol. 2006;6:6.
                doi:10.1186/1471-2210-6-6 (PMC1448202, open access).
                PMID:16515684. Intestine 407 cells transiently transfected
                with HUMAN or MOUSE DAT, SERT or NET cDNA; uptake of 50 nM
                [3H]-labelled substrate, 10 min, 20 C, with increasing drug
                concentrations; IC50 by nonlinear regression and
                KI = IC50/(1 + [S]/KM). Table 1, mean +/- SEM of 4-7
                experiments, both species side by side.

                This is the primary panel for the three psychostimulants,
                and it was preferred for three reasons: it is an UPTAKE
                INHIBITION constant, which is the quantity `transport.py`'s
                mechanism actually consumes, rather than a binding affinity
                at a radioligand site; it has a MOUSE column, and mouse is
                this project's species (docs/roadmap.md); and all nine
                stimulant/transporter cells come from one laboratory and one
                assay, so a selectivity ratio built from them compares two
                transporters rather than two laboratories -- the argument
                P5.7 made for [Toll1998] and made again here.

                The paper does not state the KM values used in its
                Cheng-Prusoff correction. With [S] = 50 nM and the substrate
                Km values in this module (160-170 nM), the correction factor
                is 1.3, so the reported KI values are about 30% below the
                raw IC50 values. Recorded because it is a real, unresolvable
                detail of a number every result here depends on, and because
                it bounds the error at well under two-fold.

  [Bymaster2002] Bymaster FP, Katner JS, Nelson DL, Hemrick-Luecke SK,
                Threlkeld PG, Heiligenstein JH, Morin SM, Gehlert DR, Perry
                KW. "Atomoxetine increases extracellular levels of
                norepinephrine and dopamine in prefrontal cortex of rat: a
                potential mechanism for efficacy in attention
                deficit/hyperactivity disorder." Neuropsychopharmacology.
                2002;27(5):699-711. PMID:12431845. Atomoxetine inhibited
                radioligand binding at clonal cell lines transfected with
                HUMAN NE, 5-HT and DA transporters with Ki of 5, 77 and
                1451 nM respectively.

  [Carlier1998] Carlier PR, Lo MM, Lo PC, Richelson E, Tatsumi M, Reynolds
                IJ, Sharma TA. "Synthesis of a potent wide-spectrum
                serotonin-, norepinephrine-, dopamine-reuptake inhibitor
                (SNDRI) and a species-selective dopamine-reuptake inhibitor
                based on the gamma-amino alcohol functional group." Bioorg
                Med Chem Lett. 1998;8(5):487-492. PMID:9871604. HUMAN
                transporters. Atomoxetine (as tomoxetine) Kd 2.03 nM at NET,
                8.9 nM at SERT, 1080 nM at DAT. The second laboratory for
                atomoxetine; not used to build a ligand. Read through the
                IUPHAR/BPS Guide to PHARMACOLOGY's curation of this paper
                (ligand 7118, interactions at targets 926/927/928), not from
                the paper itself, which was not reachable.

  [Tatsumi1997] Tatsumi M, Groshan K, Blakely RD, Richelson E.
                "Pharmacological profile of antidepressants and related
                compounds at human monoamine transporters." Eur J Pharmacol.
                1997;340(2-3):249-258. PMID:9537821. Equilibrium dissociation
                constants at HUMAN 5-HT, NE and DA transporters for 37
                antidepressants. Fluoxetine at SERT, Ki 3 nM.

                PROVENANCE, stated because it is weaker than this codebase's
                usual standard: the full table was not reachable (the paper
                is behind a subscription and no open reproduction of its
                fluoxetine row was found). The 3 nM figure is IUPHAR/BPS
                Guide to PHARMACOLOGY interaction 13001, which records
                `Ki = 3x10^-9 M` at human SERT citing PMID 9537821. IUPHAR's
                curation of a primary paper was accepted as a source by P5.7
                for exactly this kind of check; here it is the source itself
                rather than the check on one, and that is a difference worth
                naming. Fluoxetine's constants at NET and at DAT were not
                obtainable at all from that curation, and are recorded in
                `MISSING_KI` rather than filled in.

  [Rothman2012] Rothman RB, Partilla JS, Baumann MH, Lightfoot-Siordia C,
                Blough BE. "Studies of the biogenic amine transporters. 14.
                Identification of low-efficacy 'partial' substrates for the
                biogenic amine transporters." J Pharmacol Exp Ther.
                2012;341(1):251-262. doi:10.1124/jpet.111.188946 (PMC3364510,
                open access). PMID:22271821. RAT brain synaptosomes
                preloaded with 5 nM [3H]MPP+ (DAT, NET) or [3H]5-HT (SERT)
                in Krebs-phosphate buffer with 1 uM reserpine. Table 1,
                d-amphetamine: EC50 6.4 +/- 1 nM at DAT with Emax
                103 +/- 5%, 7.4 +/- 2.42 nM at NET with Emax 101 +/- 10%,
                1960 +/- 140 nM at SERT with Emax 104 +/- 2%.

                The paper is also the source of the OPERATIONAL definition
                this module's substrate/blocker split rests on: "Uptake
                inhibitors bind to the transporter, are not transported into
                the nerve terminal, but block the transport of endogenous
                substrates. Releasers are transporter substrates ...
                transported into the nerve terminal and, as a result,
                cytoplasmic neurotransmitters are countertransported out",
                distinguished experimentally by substrate reversal: a
                releaser's effect is attenuated by a blocking concentration
                of an uptake inhibitor, a blocker's is not.

  [Zhu2008]     Zhu J, Reith MEA. "Role of dopamine transporter in the
                action of psychostimulants, nicotine, and other drugs of
                abuse." CNS Neurol Disord Drug Targets. 2008;7(5):393-409.
                PMID:19128199 (PMC3133725, open access). Cited for the
                mechanism statements only, no number is taken from it:
                "Cocaine acts as a simple inhibitor that binds to the DAT and
                blocks its transport activity"; "amphetamines not only
                competitively inhibit DA reuptake and thereby increase
                synaptic DA levels, but also reverse transport activity,
                resulting in efflux of DA via the DAT".

Two preparations, and what the disagreement is worth
-----------------------------------------------------

[HanGu2006] measured cocaine, methylphenidate and amphetamine at cloned mouse
transporters in a cell line; [JohnJones2007] measured the same three drugs (and
fluoxetine) against endogenously released transmitter in mouse brain slices.
Two species-matched measurements of the same quantity by wholly different
methods is an unusually direct check, and it comes out in two halves:

* **At DAT the two agree within 1.4-fold on all three drugs.** Cocaine 490
  against 350 nM, methylphenidate 260 against 210 nM, amphetamine 560 against
  420 nM. A cloned mouse transporter in a gut-derived cell line and a mouse
  striatal terminal in a slice give the same number to within the spread of
  either measurement.
* **At SERT they do not.** Cocaine 730 against 10,000 nM, methylphenidate
  114,370 against 19,000 nM, amphetamine 23,820 against 23,000 nM: agreement
  for amphetamine, 13.7-fold and 6.0-fold disagreements for the other two.
  [JohnJones2007] states the caveat itself -- "direct comparisons of apparent
  Km values or Ki values between the DA and 5-HT systems may not be valid",
  and its serotonin measurements are made in substantia nigra with 30-pulse
  stimulation, where the released transmitter concentration far exceeds Km.

`preparation_agreement()` returns the fold disagreement per cell and a test
asserts both halves, including the failures.

**Fluoxetine is where the two views of the world are furthest apart, and it is
not hidden.** [JohnJones2007]'s apparent Ki for fluoxetine at serotonin uptake
in mouse slices is 59 uM; the human SERT binding constant is 3 nM. That is
roughly 20,000-fold, and it is not a rounding question: one of the two says
fluoxetine is a subnanomolar-potency drug and the other says it is nearly
inert. This module uses the binding constant, for reasons stated so they can
be argued with:

1. It is the constant the field carries, and it is the same KIND of quantity
   -- an equilibrium affinity at the site -- that every receptor module in
   this codebase is built from.
2. [JohnJones2007]'s own DAT column agrees with an independent cell-based
   mouse panel to 1.4-fold, and its fluoxetine cell disagrees with the
   published affinity by four orders of magnitude, so the outlier is one cell
   rather than the method.
3. The authors name the mechanism of the discrepancy themselves: their assay
   uses endogenously released transmitter rather than exogenously added, and
   they list "assay temperature, interactions with binding proteins,
   transporter internalization, drug sequestration or post-translational
   modifications" as the reasons their values "differ greatly from those
   reported in other studies". Fluoxetine is by far the most lipophilic drug
   in their panel, so slice sequestration hurts it most.

What that choice means for the model is stated rather than glossed: with
Ki = 3 nM, the exact steady-state law above says 1 uM fluoxetine would raise
extracellular 5-HT by 334-fold, which is not what SSRIs do in a living
animal. The missing piece is 5-HT1A/1B autoreceptor feedback, which curtails
the rise and which is P6.1/P6.2's, not P5.8's. `fluoxetine_fold_rise()`
exists so this over-prediction is a number in a test rather than an
embarrassment discovered later, and it is the sharpest illustration in this
codebase so far of why the projection-system layer is the next phase.

Two laboratories on atomoxetine
--------------------------------

[Bymaster2002] and [Carlier1998] measured atomoxetine at the same three human
transporters. NET 5 against 2.03 nM (2.5-fold), DAT 1451 against 1080 nM
(1.3-fold), SERT 77 against 8.9 nM (8.6-fold). The SERT cell is the
disagreement that matters, because it is the difference between "atomoxetine
is 15-fold NET-selective over SERT" and "4.4-fold". [Bymaster2002] is used,
being the paper about the compound itself and the one whose abstract was read
directly; [Carlier1998] is recorded in `KI_SECOND_LAB` and a test asserts the
spread rather than averaging it away.

Species: mouse where possible, human where not
-----------------------------------------------

The three psychostimulants use [HanGu2006]'s MOUSE column, which matches this
project's species. The two antidepressants have no mouse measurement in reach
and use human transporters. That mixing is bounded rather than ignored:
[HanGu2006] measured the same nine cells in both species, so the cost of a
human-for-mouse substitution is directly measurable in that paper's own data,
and `species_ratio()` returns it. Eight of the nine cells agree within
2.4-fold; the ninth, methylphenidate at DAT, is 4.3-fold (mouse 260 nM,
human 60 nM). So substituting a human constant for a mouse one is worth
roughly a factor of two to four here, which is smaller than the
between-laboratory spread on atomoxetine's SERT cell, and the ordering of
that comparison is the useful output.

Searched and empty, or found and deliberately not used
-------------------------------------------------------

* **Fluoxetine at NET and at DAT, as an equilibrium constant.** Searched
  Europe PMC, PubMed, the IUPHAR/BPS Guide to PHARMACOLOGY REST API
  (ligand 203, all transporter interactions) and ChEMBL for [Tatsumi1997]'s
  full row and for any open-access panel reporting fluoxetine at all three
  human transporters. IUPHAR curates only the SERT cell of that paper;
  the paper itself and every reproduction of its table found are behind
  subscriptions; ChEMBL's REST API returned HTTP 500 throughout. Recorded in
  `MISSING_KI` and left out of the model rather than filled from a
  secondary quotation. The consequence is bounded in the direction that
  matters: [JohnJones2007]'s independent mouse-tissue measurement puts
  fluoxetine's apparent Ki at dopamine uptake at 270 uM, so the action this
  module omits at DAT is one it also has evidence is negligible. At NET
  there is no such bound and the omission is unbounded, which is why
  fluoxetine's noradrenergic action is listed under "not built".
* **Michaelis-Menten Km and Vmax for noradrenaline uptake** in any rodent
  preparation. Searched voltammetry and synaptosome literature for a
  saturation analysis of NE uptake; what exists is clearance half-lives
  ([Park2012]) and release/uptake comparisons that report no saturation
  constants. Hence NET's first-order form, which uses the measured ratio and
  invents no splitting of it.
* **A basal extracellular noradrenaline concentration** by zero net flux or
  any other quantitative microdialysis method, in mouse or rat. Searched; the
  quantitative-microdialysis literature reachable here is dopamine and
  serotonin. Hence `NORADRENALINE_BASAL_UNIT_SCALE`, which is a unit and not
  a measurement, and the test that no NET conclusion depends on it.
* **[Rothman2001]** (Rothman RB et al., Synapse 2001;39:32-41, PMID
  11071707), the paper the brief's releaser framing comes from and the most
  cited source for amphetamine's release potencies. Subscription-only, and
  no open reproduction of its table with error terms was found;
  secondary quotations of "24.8 / 7.07 / 1765 nM" were found and NOT used,
  because a number read from a citing dissertation is not a citation.
  [Rothman2012] is the same laboratory's open-access measurement of the same
  quantities with the same assay and is used instead. The two differ at DAT
  (6.4 against 24.8 nM), which is a tracer difference: [Rothman2012] releases
  [3H]MPP+ and [Rothman2001] released [3H]dopamine.
* **Rodent pharmacokinetics for any of the five drugs.** Not adopted, for
  P5.3's and P5.7's two reasons: a real half-life paired with an
  illustrative release capacity buys nothing, and it would make an
  illustrative number look calibrated. `TOOL_COMPOUND_KE` is shared with
  every other P5 module and a test asserts the equality. Cocaine's real
  rodent half-life is minutes and methylphenidate's is under an hour, so
  no statement about the TIME COURSE of a psychostimulant effect should be
  read out of this module.
* **An absolute efflux capacity for a saturating substrate**, i.e. how many
  nM/min of transmitter amphetamine can drive out of the terminal in a
  preparation matched to the uptake measurements. [Rothman2012]'s Emax is a
  percentage of the assay's own maximum and carries no absolute rate; the in
  vivo measurements that would set it (microdialysis fold-increases after
  1 mg/kg amphetamine) are dose-based and run into the same Phase 1 bridge
  that Track E and P2.2 documented for every circuit gain in this project.
  Hence `RELEASE_CAPACITY`, labelled illustrative.

What was deliberately not built
--------------------------------

* **The vesicular monoamine transporter, VMAT2.** Amphetamine's efflux has
  two prongs ([Rothman2012]: it "promote[s] efflux of transmitter by a process
  of transporter-mediated exchange" AND "increase[s] cytoplasmic levels of
  transmitter by disrupting storage of transmitters in vesicles"), and only
  the first is here. Representing the second needs a cytoplasmic transmitter
  pool this model does not have, and it is the reason `RELEASE_CAPACITY`
  is a single lumped constant rather than a mechanism.
* **Transporter-mediated currents, trace-amine receptor signalling, and
  transporter trafficking.** All real amphetamine actions, none of them
  expressible as a term in a mass balance on the extracellular pool.
* **Region-wise transporter density.** `Slc6a3`, `Slc6a4` and `Slc6a2` ISH
  data are already in `data/expression/`, so P4.2's
  `coupling = mechanism_gain * rho` factorization could distribute Vmax over
  regions. It is not applied here: P5.8's brief is the mechanism and the
  constants, the measured Vmax values are region-specific measurements
  (caudate-putamen, substantia nigra) rather than whole-brain averages to be
  redistributed, and one transmitter pool per transporter is the right shape
  until P6.3 supplies projection densities. Named as the obvious next step.
* **Noradrenaline reuptake by DAT, and dopamine reuptake by NET**, which is
  the standard explanation for atomoxetine raising cortical DOPAMINE
  ([Bymaster2002]'s own finding). One transporter clears one pool here. The
  cross-clearance is a Phase 6 statement about which transporter is present
  in which region, not a P5.8 statement about a constant.
* **A behavioural endpoint.** The readouts stay circuit-level, as in every
  P5 module. Phase 7 names behaviours.
"""

import numpy as np

from .engine import ReceptorSystem, TransporterSystem
from .pk import OneCompartmentPK
from .transport import MonoamineTransporter, TransporterLigand


#: The three transporters, in the order every table below uses.
TRANSPORTERS = ("DAT", "SERT", "NET")

#: The transmitter each one clears.
TRANSMITTER = {"DAT": "dopamine", "SERT": "serotonin", "NET": "noradrenaline"}

#: The five drugs the brief names, psychostimulants first.
DRUGS = ("cocaine", "methylphenidate", "amphetamine", "fluoxetine", "atomoxetine")


# ---------------------------------------------------------------------------
# Uptake kinetics of the natural substrate
# ---------------------------------------------------------------------------

#: nM. Michaelis constant for DOPAMINE uptake, mouse caudate-putamen,
#: [JohnJones2007] (0.16 uM). [Bosse2011] independently fixed its own
#: voltammetric fits at the same 0.16 uM "based on an average literature
#: value", so this is the value two mouse laboratories work from.
KM_DOPAMINE = 160.0

#: nM/min. Maximal velocity of dopamine uptake, mouse caudate-putamen,
#: [JohnJones2007]: 3.1 +/- 0.20 uM/s = 3100 nM/s * 60.
VMAX_DOPAMINE = 186000.0

#: nM/min. The same quantity from [Bosse2011] in wild-type mice,
#: 4.1 +/- 0.1 uM/s. Not used to build the transporter: it is the
#: cross-laboratory check, 1.32-fold from the value above. `steady_state_fold`
#: is independent of Vmax, so the two choices differ only in how fast the
#: pool relaxes, and a test asserts that.
VMAX_DOPAMINE_SECOND_LAB = 246000.0

#: nM. Basal extracellular dopamine, mouse caudate-putamen, ZERO NET FLUX
#: microdialysis, [Bosse2011] wild-type: 5.0 +/- 0.2 nM.
BASAL_DOPAMINE = 5.0

#: nM. Michaelis constant for SEROTONIN uptake, mouse substantia nigra pars
#: reticulata, [JohnJones2007] (0.17 uM).
KM_SEROTONIN = 170.0

#: nM/min. Maximal velocity of serotonin uptake in the same preparation,
#: [JohnJones2007]: 0.080 +/- 0.006 uM/s = 80 nM/s * 60. Thirty-nine-fold
#: below dopamine's, in the same paper's own hands.
VMAX_SEROTONIN = 4800.0

#: nM. Basal extracellular serotonin, mouse striatum, ZERO NET FLUX
#: microdialysis, [Mathews2004] wild-type SERT(+/+): 2.8 +/- 1 nM. The same
#: paper's frontal cortex value is 1.4 +/- 0.4 nM.
BASAL_SEROTONIN = 2.8

#: s. Noradrenaline clearance half-life in rat ventral BNST, [Park2012]:
#: 1.36 +/- 0.07 s. The dorsomedial BNST value in the same paper is
#: 2.37 +/- 0.07 s; the ventral one is used because it is the terminal field
#: with the denser noradrenergic innervation in that paper's own account.
NORADRENALINE_CLEARANCE_HALF_LIFE_S = 1.36

#: min^-1. `ln(2) / (1.36/60)` = 30.6. DERIVED from the measured half-life
#: above, and equal to `Vmax/Km` for a transporter operating well below its
#: Km. This is the whole of what is published for NET, which is why NET is
#: built with `MonoamineTransporter.first_order` and has no Km or Vmax of its
#: own: splitting a measured ratio into two numbers would be inventing one.
NORADRENALINE_CLEARANCE_RATE = 60.0 * np.log(2.0) / NORADRENALINE_CLEARANCE_HALF_LIFE_S

#: nM. ILLUSTRATIVE UNIT SCALE, not a measurement of anything. No basal
#: extracellular noradrenaline concentration by zero net flux or any other
#: quantitative microdialysis method was found (see "Searched and empty").
#: Setting it to 1.0 makes NET's transmitter state numerically equal to its
#: own fold-of-basal, and the exact steady-state law in `transport.py` says
#: every fold-change this module reports is independent of it. A test asserts
#: that independence, so this number cannot quietly acquire meaning.
NORADRENALINE_BASAL_UNIT_SCALE = 1.0


# ---------------------------------------------------------------------------
# Uptake inhibition constants
# ---------------------------------------------------------------------------

#: nM. The table the transporters are built from.
#:
#: cocaine / methylphenidate / amphetamine: [HanGu2006] Table 1, MOUSE
#: transporter column, uptake inhibition of 50 nM [3H]substrate in
#: transfected Intestine 407 cells, Cheng-Prusoff corrected.
#: atomoxetine: [Bymaster2002], HUMAN transporters, radioligand binding.
#: fluoxetine: [Tatsumi1997] via IUPHAR curation, HUMAN SERT, radioligand
#: binding; its NET and DAT cells are in `MISSING_KI`, not here.
#:
#: A drug/transporter pair absent from this table is absent from the model,
#: which is the point: an unmeasured constant is a gap, not an infinity.
KI = {
    "cocaine": {"DAT": 490.0, "SERT": 730.0, "NET": 460.0},
    "methylphenidate": {"DAT": 260.0, "SERT": 114370.0, "NET": 170.0},
    "amphetamine": {"DAT": 560.0, "SERT": 23820.0, "NET": 120.0},
    "fluoxetine": {"SERT": 3.0},
    "atomoxetine": {"DAT": 1451.0, "SERT": 77.0, "NET": 5.0},
}

#: nM. The SEM [HanGu2006] prints beside each mouse KI above. `None` where
#: the source reports no dispersion (the two binding papers quote point
#: estimates).
KI_SEM = {
    "cocaine": {"DAT": 40.0, "SERT": 120.0, "NET": 60.0},
    "methylphenidate": {"DAT": 30.0, "SERT": 7610.0, "NET": 30.0},
    "amphetamine": {"DAT": 110.0, "SERT": 1710.0, "NET": 20.0},
    "fluoxetine": {"SERT": None},
    "atomoxetine": {"DAT": None, "SERT": None, "NET": None},
}

#: nM. [HanGu2006]'s HUMAN transporter column, same table, same assay, same
#: day. Not used to build anything: it is the direct measurement of what a
#: human-for-mouse substitution costs, which is the question the two
#: antidepressants raise. See `species_ratio()`.
KI_HUMAN = {
    "cocaine": {"DAT": 230.0, "SERT": 740.0, "NET": 480.0},
    "methylphenidate": {"DAT": 60.0, "SERT": 132430.0, "NET": 100.0},
    "amphetamine": {"DAT": 640.0, "SERT": 38460.0, "NET": 70.0},
}

#: nM. [JohnJones2007] Table I: apparent Ki against ENDOGENOUSLY released
#: transmitter in MOUSE brain slices, i.e. the same species as `KI`'s
#: stimulant rows but a wholly different preparation. Not used to build
#: anything; it is the second-preparation check, and the place the
#: fluoxetine question is posed. See `preparation_agreement()`.
KI_MOUSE_SLICE = {
    "cocaine": {"DAT": 350.0, "SERT": 10000.0},
    "methylphenidate": {"DAT": 210.0, "SERT": 19000.0},
    "amphetamine": {"DAT": 420.0, "SERT": 23000.0},
    "fluoxetine": {"DAT": 270000.0, "SERT": 59000.0},
}

#: nM. [Carlier1998]'s human transporter Kd values for atomoxetine, the
#: second laboratory. Not used to build anything. See
#: `atomoxetine_lab_disagreement()`.
KI_SECOND_LAB = {"atomoxetine": {"DAT": 1080.0, "SERT": 8.9, "NET": 2.03}}

#: Drug/transporter pairs with NO usable constant, and why. Kept as data
#: rather than as prose so that "this is a gap" is checkable, in the same
#: spirit as `opioid.MEASURED_FLAT`: a reader can ask the module what it does
#: not know.
MISSING_KI = {
    ("fluoxetine", "NET"): (
        "No equilibrium constant obtained. [Tatsumi1997] measured it and the "
        "table was not reachable; IUPHAR curates only that paper's SERT cell. "
        "Unbounded omission: fluoxetine's noradrenergic action is absent from "
        "this model and nothing here bounds its size."
    ),
    ("fluoxetine", "DAT"): (
        "No equilibrium constant obtained, same reason. Bounded omission: "
        "[JohnJones2007] measured an apparent Ki of 270 uM against dopamine "
        "uptake in mouse tissue, so the action left out is one there is "
        "independent evidence is negligible."
    ),
}


# ---------------------------------------------------------------------------
# Release: the substrate half of the mechanism
# ---------------------------------------------------------------------------

#: nM. [Rothman2012] Table 1, d-amphetamine, rat brain synaptosomes,
#: [3H]MPP+ release at DAT and NET and [3H]5-HT release at SERT. These are
#: RELEASE potencies and have nothing to do with uptake inhibition; the
#: distance between the two is the point (see `release_over_block_ratio`).
RELEASE_EC50 = {"amphetamine": {"DAT": 6.4, "SERT": 1960.0, "NET": 7.4}}

#: nM. The SEM printed beside each of them in the same table.
RELEASE_EC50_SEM = {"amphetamine": {"DAT": 1.0, "SERT": 140.0, "NET": 2.42}}

#: Releasing efficacy as a fraction of the assay's own maximum,
#: [Rothman2012] Table 1: 103 +/- 5%, 104 +/- 2%, 101 +/- 10%. Values above
#: 1.0 are as printed and are NOT clipped, for P5.7's reason: rounding a
#: measurement to a tidier number replaces it. d-Amphetamine is the paper's
#: full-efficacy reference, which is why all three sit at 1.0 within error.
RELEASE_EMAX = {"amphetamine": {"DAT": 1.03, "SERT": 1.04, "NET": 1.01}}

#: The drugs for which a release EC50 exists, i.e. the substrates. Everything
#: else in `DRUGS` is a pure blocker. This is a classification by
#: MEASUREMENT -- [Rothman2012]'s substrate-reversal criterion -- not by drug
#: class, and a test asserts it agrees with the mechanism the built ligands
#: end up carrying.
SUBSTRATES = frozenset(RELEASE_EC50)

#: Pure uptake blockers: bind, are not transported, release nothing.
#: [Zhu2008] for cocaine and methylphenidate, [Rothman2012]'s use of cocaine
#: as the blocking agent in its substrate-reversal experiments, and
#: [Bymaster2002] and [Tatsumi1997] describing atomoxetine and fluoxetine as
#: reuptake inhibitors.
PURE_BLOCKERS = frozenset(DRUGS) - SUBSTRATES

#: ILLUSTRATIVE. How much efflux a full substrate at saturating concentration
#: drives, as a multiple of the transporter's own basal release rate. 1.0 by
#: the same per-mechanism unit rule P5.1 introduced for receptor gains: a
#: saturating substrate doubles the rate at which transmitter appears in the
#: extracellular space.
#:
#: NOT a measurement, and no result stated as a release POTENCY depends on
#: it: [Rothman2012]'s Emax is a percentage of its own assay's maximum, and
#: the absolute rate that would calibrate this is an in vivo dose-response
#: measurement blocked by the same Phase 1 bridge Track E documented. This is
#: the one constant to change when such a measurement is in reach, and a test
#: asserts that the substrate-versus-blocker conclusions in this module hold
#: for any positive value of it.
RELEASE_CAPACITY = 1.0


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def _check_transporter(transporter):
    if transporter not in TRANSPORTERS:
        raise ValueError(
            "transporter must be one of %s, got %r" % (list(TRANSPORTERS), transporter)
        )


def ki(drug, transporter):
    """nM. The uptake-inhibition constant used to build the model, or None
    where none is available (see `MISSING_KI` for why)."""
    _check_transporter(transporter)
    if drug not in KI:
        raise ValueError("unknown drug %r; have %s" % (drug, sorted(KI)))
    return KI[drug].get(transporter)


def release_ec50(drug, transporter):
    """nM, or None for a pure blocker."""
    _check_transporter(transporter)
    return RELEASE_EC50.get(drug, {}).get(transporter)


def release_emax(drug, transporter):
    """Releasing efficacy as a fraction, or 0.0 for a pure blocker."""
    _check_transporter(transporter)
    return RELEASE_EMAX.get(drug, {}).get(transporter, 0.0)


def is_substrate(drug):
    """True for a substrate-type releaser, by [Rothman2012]'s criterion."""
    return drug in SUBSTRATES


def selectivity(drug, preferred, over):
    """`Ki(over) / Ki(preferred)`: how many fold the drug prefers one
    transporter to another, from one table and one assay.

    Returns None if either cell is missing, rather than a number built from
    two laboratories -- the same rule `opioid.selectivity` states.
    """
    a = ki(drug, preferred)
    b = ki(drug, over)
    if a is None or b is None:
        return None
    return b / a


def fold_rise(drug, concentration, transporter):
    """The exact steady-state fold-rise in extracellular transmitter a PURE
    BLOCKER at `concentration` nM produces: `1 + C/Ki`.

    Carries no illustrative constant at all -- Km, Vmax and the basal level
    all cancel out of the mass balance (see `transport.py`). Raises for a
    substrate, whose steady state is not this expression.
    """
    if is_substrate(drug):
        raise ValueError(
            "%r is a substrate: its steady state adds an efflux term, so use "
            "the transporter's own steady_state_fold()" % (drug,)
        )
    value = ki(drug, transporter)
    if value is None:
        return None
    return 1.0 + float(concentration) / value


def fluoxetine_fold_rise(concentration=1000.0):
    """The number this module over-predicts, kept as a function so it is
    tested rather than admitted in prose.

    334 at 1 uM fluoxetine. Real SSRIs raise extracellular 5-HT a few fold,
    not a few hundred, because 5-HT1A and 5-HT1B autoreceptors cut release
    as soon as the transmitter rises. This model has no path from a
    transmitter concentration back to a release rate, which is exactly P6.1's
    projection-system abstraction, so the open-loop prediction stands
    uncorrected. See the module docstring.
    """
    return fold_rise("fluoxetine", concentration, "SERT")


def release_over_block_ratio(drug="amphetamine"):
    """`Ki / EC50_release` per transporter: how many fold more potent the
    substrate's releasing action is than its blocking action.

    `{'DAT': 87.5, 'SERT': 12.2, 'NET': 16.2}` for amphetamine. The number
    the substrate/blocker distinction is worth: a model that represented
    amphetamine as a blocker alone would put its dopaminergic potency nearly
    two orders of magnitude too low.
    """
    out = {}
    for transporter in TRANSPORTERS:
        ec50 = release_ec50(drug, transporter)
        block = ki(drug, transporter)
        if ec50 is None or block is None:
            continue
        out[transporter] = block / ec50
    return out


def preparation_agreement():
    """`(drug, transporter) -> fold disagreement` between [HanGu2006]'s
    cloned mouse transporters and [JohnJones2007]'s mouse brain slices.

    Always >= 1. Three DAT cells sit under 1.4; the SERT cells run to 13.7,
    and fluoxetine at SERT -- where the model's constant is [Tatsumi1997]'s
    human binding value rather than [HanGu2006]'s -- is about 20,000. The
    module docstring argues which value to believe; this function is what a
    test asserts against.
    """
    out = {}
    for drug, row in KI_MOUSE_SLICE.items():
        for transporter, slice_value in row.items():
            reference = ki(drug, transporter)
            if reference is None:
                continue
            ratio = slice_value / reference
            out[(drug, transporter)] = ratio if ratio >= 1.0 else 1.0 / ratio
    return out


def species_ratio():
    """`(drug, transporter) -> fold difference` between [HanGu2006]'s own
    mouse and human columns.

    Always >= 1. The direct price of a human-for-mouse substitution, measured
    in one laboratory on one day, and the reason this module can say what
    using human constants for fluoxetine and atomoxetine costs instead of
    hoping it is small. Eight of nine cells are under 2.4; methylphenidate at
    DAT is 4.3.
    """
    out = {}
    for drug, row in KI_HUMAN.items():
        for transporter, human in row.items():
            mouse = ki(drug, transporter)
            if mouse is None:
                continue
            ratio = human / mouse
            out[(drug, transporter)] = ratio if ratio >= 1.0 else 1.0 / ratio
    return out


def atomoxetine_lab_disagreement():
    """`transporter -> fold difference` between [Bymaster2002] and
    [Carlier1998] for atomoxetine at the human transporters.

    `{'DAT': 1.34, 'SERT': 8.65, 'NET': 2.46}`. The SERT cell is the one that
    matters: it is the difference between calling atomoxetine 15-fold
    NET-selective over SERT and calling it 4.4-fold.
    """
    out = {}
    for transporter, other in KI_SECOND_LAB["atomoxetine"].items():
        used = ki("atomoxetine", transporter)
        ratio = other / used
        out[transporter] = ratio if ratio >= 1.0 else 1.0 / ratio
    return out


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, deliberately the same number as
#: `serotonin.TOOL_COMPOUND_KE`, `dopamine.TOOL_COMPOUND_KE`,
#: `noradrenaline.TOOL_COMPOUND_KE`, `acetylcholine.TOOL_COMPOUND_KE`,
#: `histamine.TOOL_COMPOUND_KE`, `adenosine.TOOL_COMPOUND_KE` and
#: `opioid.TOOL_COMPOUND_KE`; a test asserts the equality. Carried by all
#: five drugs. Do not cite it as any of their half-lives in any species, and
#: note especially that cocaine's rodent half-life is minutes.
TOOL_COMPOUND_KE = 0.023105


def pk_models(drugs):
    """One PK model per drug, in the order given."""
    return [OneCompartmentPK(ke=TOOL_COMPOUND_KE) for _ in drugs]


# ---------------------------------------------------------------------------
# The transporters
# ---------------------------------------------------------------------------


def drugs_at(transporter):
    """The drugs this module can put at one transporter, in `DRUGS` order.

    A drug is here if it has a measured constant of either kind at this
    transporter. Fluoxetine is therefore absent from DAT and NET, which is
    `MISSING_KI` showing through into the model rather than being papered
    over.
    """
    _check_transporter(transporter)
    return tuple(
        drug
        for drug in DRUGS
        if ki(drug, transporter) is not None
        or release_ec50(drug, transporter) is not None
    )


def ligands_at(transporter):
    """The `transport.TransporterLigand` list for one transporter."""
    return [
        TransporterLigand(
            name=drug,
            ki=ki(drug, transporter),
            release_ec50=release_ec50(drug, transporter),
            release_emax=release_emax(drug, transporter),
        )
        for drug in drugs_at(transporter)
    ]


def dat(release_capacity=RELEASE_CAPACITY):
    """The dopamine transporter, Michaelis-Menten form, mouse constants."""
    return MonoamineTransporter(
        name="DAT",
        transmitter="dopamine",
        ligands=ligands_at("DAT"),
        km=KM_DOPAMINE,
        vmax=VMAX_DOPAMINE,
        basal=BASAL_DOPAMINE,
        release_capacity=release_capacity,
    )


def sert(release_capacity=RELEASE_CAPACITY):
    """The serotonin transporter, Michaelis-Menten form, mouse constants."""
    return MonoamineTransporter(
        name="SERT",
        transmitter="serotonin",
        ligands=ligands_at("SERT"),
        km=KM_SEROTONIN,
        vmax=VMAX_SEROTONIN,
        basal=BASAL_SEROTONIN,
        release_capacity=release_capacity,
    )


def net(release_capacity=RELEASE_CAPACITY):
    """The noradrenaline transporter, FIRST-ORDER form.

    The only one of the three built this way, and the reason is a gap rather
    than a preference: what is published for NET is a clearance half-life,
    which fixes `Vmax/Km` and neither of them. See
    `NORADRENALINE_CLEARANCE_RATE` and `transport.py`'s module docstring.
    """
    return MonoamineTransporter.first_order(
        name="NET",
        transmitter="noradrenaline",
        ligands=ligands_at("NET"),
        clearance_rate=NORADRENALINE_CLEARANCE_RATE,
        basal=NORADRENALINE_BASAL_UNIT_SCALE,
        release_capacity=release_capacity,
    )


TRANSPORTER_BUILDERS = {"DAT": dat, "SERT": sert, "NET": net}


def transporter_system(transporter, release_capacity=RELEASE_CAPACITY):
    """One transporter wired into the engine, PK models included."""
    _check_transporter(transporter)
    built = TRANSPORTER_BUILDERS[transporter](release_capacity=release_capacity)
    return TransporterSystem(
        pk_models=pk_models(built.ligands), transporter=built, name=transporter
    )


def monoamine_transporter_systems(release_capacity=RELEASE_CAPACITY):
    """All three, in `TRANSPORTERS` order.

    Dosing `cocaine` through `Simulation.dose_vector` now reaches all three
    at their measured constants at once, which is the whole point of a
    single-panel source: cocaine is not a dopamine drug, it is a
    three-transporter drug whose constants at the three differ by 1.6-fold.
    """
    return [
        transporter_system(name, release_capacity=release_capacity)
        for name in TRANSPORTERS
    ]


# ---------------------------------------------------------------------------
# The Phase 6 interface, demonstrated
# ---------------------------------------------------------------------------

#: Which transporter supplies which endogenous transmitter, as a name map a
#: `ReceptorSystem.transmitter_sources` entry is built from. This dict IS the
#: interface contract in its smallest form: a receptor whose ligand is the
#: endogenous transmitter names the transporter system that maintains it, and
#: the engine substitutes that state for the ligand's PK concentration.
#:
#: What P6.1 changes and what it does not: P6.1 makes the RELEASE term of
#: each transporter a function of its source nucleus's firing rate, and P6.3
#: distributes one transmitter pool into region-wise pools weighted by
#: projection density. Neither touches this map or the engine plumbing behind
#: it; both replace `MonoamineTransporter.release_rate` and multiply the
#: number of pools. That is why the release rate is a plain settable
#: attribute rather than a derived constant.
TRANSMITTER_SOURCE = {"dopamine": "DAT", "serotonin": "SERT", "noradrenaline": "NET"}


def d2_receptor_system_on_transporter_dopamine():
    """The existing calibrated D2 system, driven by DAT's dopamine pool.

    Nothing about the receptor is new: `scenarios.d2_receptor_site_
    calibrated()` and `scenarios.d2_coupling()` are unchanged, with
    [Agren2021]'s measured dopamine kon/koff and [Sykes2017]'s haloperidol
    constants. The single change is `transmitter_sources`, which makes the
    `dopamine` ligand bind the extracellular concentration DAT is
    maintaining instead of a dose someone injected by hand.

    That is small in code and large in what it fixes. Every module from P5.1
    to P5.7 had to write "an antagonist given alone is inert here, because
    P6.2 has supplied no endogenous transmitter to displace". With this
    wiring there IS endogenous dopamine -- 5.0 nM of it, measured
    ([Bosse2011]), sustained by a measured uptake process -- so haloperidol
    given alone is not inert, and a test asserts it. The tone is local rather
    than projection-derived, so this closes the gap for one transmitter at
    one site and not the general case, which is still P6.2's.
    """
    from . import scenarios

    return ReceptorSystem(
        pk_models=scenarios.d2_pk_models_calibrated(),
        receptor_site=scenarios.d2_receptor_site_calibrated(),
        coupling=scenarios.d2_coupling(),
        name="D2_on_transporter_dopamine",
        transmitter_sources={"dopamine": TRANSMITTER_SOURCE["dopamine"]},
    )


#: What `stabilize()` sets the post-scaling contraction number to; P3.4's own
#: default, repeated here for the reason `opioid.OPERATING_GAIN` gives.
OPERATING_GAIN = 0.9


def stabilized_indirect_pathway_network():
    """`scenarios.basal_ganglia_indirect_pathway_network()` put through P3.4's
    stabilization procedure.

    The same five nodes, weights, taus and gains; one global weight scalar
    and re-solved baseline inputs, so the drug-free fixed point is 0.5 at
    every node and P3.4's global-uniqueness certificate holds. Used for the
    same reason `opioid.stabilized_reward_network()` is: the effect this
    module demonstrates is a change of a few hundredths in a rate, and the
    hand-set baselines put GPi close enough to saturation (P3.4's own
    finding, and the reason that check exists) that a start-up transient
    would bury it.
    """
    from . import scenarios, stability

    return stability.stabilize(
        scenarios.basal_ganglia_indirect_pathway_network(),
        operating_gain=OPERATING_GAIN,
    ).network


def resting_d2_occupancy(basal=BASAL_DOPAMINE):
    """D2 occupancy by endogenous dopamine at its measured basal level.

    `0.711`, from two measurements and no fitting: [Agren2021]'s dopamine
    kon and koff give the calibrated D2 site a Kd of 11.82/5.82 = 2.03 nM,
    and [Bosse2011]'s zero-net-flux basal dopamine is 5.0 nM, so the site
    sits at `C/(C+Kd)` = 71% occupied before any drug is given.

    That number is worth stating on its own, because it is the first
    quantitative statement of endogenous tone in this codebase and it is
    large: the striatal D2 receptor is not a mostly-empty pool waiting for a
    drug, it is a mostly-occupied one. Every "an antagonist alone is inert"
    caveat written since P5.1 was a consequence of assuming the opposite.
    """
    from . import scenarios

    dopamine = scenarios.d2_dopamine_ligands_calibrated()[0]
    kd = dopamine.koff / dopamine.kon
    return basal / (basal + kd)


def tonic_compensated_network(occupancy=None):
    """`stabilized_indirect_pathway_network()` with the tonic D2 drive folded
    into the baselines, so the drug-free fixed point is 0.5 again.

    Why this is necessary rather than fussy, and it is P3.4's lesson for the
    third time. `resting_d2_occupancy()` is 0.711 and `scenarios.d2_coupling()`
    is -3.0, so simply attaching the transporter-fed D2 system to the
    stabilized network adds a standing -2.13 to Striatum_D2_MSN's drive. That
    pins the node at a rate of 2e-4, i.e. saturated OFF, and a saturated node
    cannot respond to anything: the certificate P3.4 issues for this network
    is a certificate for the DRUG-FREE network, and tonic transmitter is not
    drug-free.

    The fix is the one the biology already implies: a real striatum's resting
    firing rate is what it is WITH tonic dopamine present, so the tonic
    inhibition belongs in the baseline the node is balanced around, not on
    top of it. This function adds `+3.0 * 0.711` to that one node's
    `baseline_input` and leaves every other node, weight, tau and gain
    untouched. A test asserts that the resulting drug-free run is flat at 0.5
    and that removing the compensation reproduces the saturation, so the
    correction is visible rather than assumed.

    Note what this does NOT do: it does not decide how much of the striatum's
    resting drive is dopaminergic. It asserts that the network's own
    calibration point already includes whatever that is, which is the
    weakest statement that makes the question askable.
    """
    from .circuit import CircuitNode
    from .network import CircuitNetwork
    from . import scenarios

    if occupancy is None:
        occupancy = resting_d2_occupancy()
    base = stabilized_indirect_pathway_network()
    coupling = scenarios.d2_coupling()
    nodes = []
    for node in base.nodes:
        offset = -coupling.get(node.name, 0.0) * occupancy
        nodes.append(
            CircuitNode(
                name=node.name,
                tau=node.tau,
                baseline_input=node.baseline_input + offset,
                gain=node.gain,
                threshold=node.threshold,
            )
        )
    return CircuitNetwork(nodes=nodes, connectivity=base.connectivity)


def run(
    transporters=None,
    receptor_systems=None,
    network=None,
    t_end=30.0,
    n_points=121,
    method="LSODA",
    rates0=None,
    **doses
):
    """Integrate one configuration and return `(network, sim, result)`.

    Shared by the tests, the example and the report so the three cannot drift
    apart in their integration settings.

    `method` defaults to LSODA and that is not cosmetic. The measured uptake
    constants put a time constant of `Km/Vmax` = 52 ms on the dopamine pool
    against a circuit relaxing over 1-2 minutes: more than three orders of
    magnitude of timescale separation in one state vector, the same stiffness
    problem P5.5 hit with MK-801's measured association rate and the same
    answer. A test asserts RK45 and LSODA agree, so the choice is a statement
    about the integrator rather than about the model.
    """
    from .engine import Simulation

    if transporters is None:
        transporters = monoamine_transporter_systems()
    network = stabilized_indirect_pathway_network() if network is None else network
    sim = Simulation(
        network=network,
        receptor_systems=receptor_systems,
        transporter_systems=transporters,
    )
    if rates0 is None:
        rates0 = np.full(network.n, 0.5)
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, n_points),
        rates0=rates0,
        method=method,
    )
    return network, sim, result


def transmitter_trace(sim, result, transporter):
    """The extracellular transmitter concentration of one transporter system
    over the whole run, in nM. The Phase 6 read, spelled out."""
    return result["transmitter"][sim.transmitter_index[transporter]]


def transmitter_fold(sim, result, transporter):
    """The same trace as a multiple of that transmitter's drug-free basal
    level, which is the unit every conclusion in this module is stated in."""
    system = next(s for s in sim.transporter_systems if s.name == transporter)
    return transmitter_trace(sim, result, transporter) / system.transporter.basal
