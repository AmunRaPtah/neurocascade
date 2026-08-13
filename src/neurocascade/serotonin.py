"""P5.1: the serotonin 5-HT1A (Gi) and 5-HT2A (Gq) receptor systems.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter below): every binding
constant and every efficacy. Six ligands over two receptor sites --
serotonin, 8-OH-DPAT and WAY-100635 at 5-HT1A; serotonin, DOI and
ketanserin at 5-HT2A.

ILLUSTRATIVE (hand-set, labelled, no literature target): the raphe/cortex
circuit's weights, taus, gains and baseline inputs; the two mechanism gains
that turn receptor activation into node drive; and the pharmacokinetics of
all four tool compounds, for which no rodent PK was found (see "Searched
and empty" below). This is the same split as the mu-opioid, D2 and GABA-A
calibrated sections in `scenarios.py`: this task calibrates the receptor
layer, not the circuit layer.

The two receptors, and why they belong in one module
----------------------------------------------------

5-HT1A is Gi/o-coupled and hyperpolarizing. It sits in two places that
matter here:

  * **somatodendritically on dorsal raphe serotonin neurons**, where it is
    the autoreceptor that shuts the serotonin system down when serotonin
    rises ([Sprouse1987]);
  * **postsynaptically on cortical pyramidal neurons**, where it is
    inhibitory ([AmargosBosch2004], [Santana2017]).

5-HT2A is Gq/11-coupled and depolarizing, densest on layer V cortical
pyramidal neurons ([Willins1997], [AghajanianMarek1997]) -- the same
neurons that carry postsynaptic 5-HT1A ([AmargosBosch2004] shows the two
co-expressed in single mPFC pyramidal cells and interacting in vivo). One
transmitter therefore reaches one cortical neuron through two receptors of
opposite sign, and which one dominates is a function of their affinities
for serotonin, not of anything hand-set. That is the reason both systems
are built in one module and put on one shared network rather than in two
separate scenarios.

The autoreceptor, and the Phase 6 dependency it creates
-------------------------------------------------------

The 5-HT1A autoreceptor's actual function is a feedback loop: serotonin
released in the raphe activates it, raphe firing falls, and *less serotonin
is released in the forebrain*. This model can express the first half of that
sentence and not the second.

`Simulation`'s ligand concentrations come from the PK layer alone. Nothing
in `rhs` lets a circuit node's firing rate change a ligand's concentration,
so `DRN_5HT`'s rate has no path back to the serotonin concentration the
cortical 5-HT1A and 5-HT2A sites see. The autoreceptor here is an open
loop: it suppresses the node that carries it, and stops.

Closing it is exactly what Phase 6 is for. P6.1 is the projection-system
abstraction (source nucleus firing rate -> transmitter release ->
region-wise transmitter concentration), and P6.2 is endogenous transmitter
tone. Until both exist, this module deliberately does NOT wire a
`DRN_5HT -> cortex` edge into the network: in a Wilson-Cowan network a
weight is a fixed-sign synaptic drive, and the raphe's influence on cortex
is not that -- it is receptor-mediated and of both signs at once (5-HT1A
inhibitory, 5-HT2A excitatory, on the same cell). Representing it as a
signed weight would double-count the receptor layer that is the whole point
of this codebase. `tests/test_serotonin_scenario.py::
test_autoreceptor_is_an_open_loop_until_phase_6` asserts the gap rather
than describing it, so that closing it in Phase 6 must break a test.

Two further consequences of the same gap, recorded so Phase 6 inherits them:

  * With no endogenous serotonin tone, an antagonist given alone has
    nothing to displace and does nothing (same as naloxone and haloperidol
    in the existing scenarios). Real WAY-100635 given alone *does* have
    effects, because real raphe neurons have tonic 5-HT1A autoreceptor
    activation to remove. That is P6.2, not a bug here.
  * [Puig2003] reports that DOI in mPFC *increases* dorsal raphe 5-HT
    neuron firing and prefrontal 5-HT release, i.e. there is a
    cortex-to-raphe descending loop as well. Also P6.

Sources
-------

5-HT1A binding and efficacy:

  [Satala2016]  Satala G, Duszynska B, Stachowicz K, Rafalo A, Pochwat B,
                Luckhart C, Albert PR, Daigle M, Tanaka KF, Hen R, Lenda T,
                Nowak G, Bojarski AJ, Szewczyk B. "Concentration-Dependent
                Dual Mode of Zn Action at Serotonin 5-HT1A Receptors: In
                Vitro and In Vivo Studies." Mol Neurobiol.
                2016;53(10):6869-6881. doi:10.1007/s12035-015-9586-3 (open
                access, PMC5104769). Human 5-HT1A stably expressed in
                HEK293 cells, agonist radioligand [3H]-8-OH-DPAT, room
                temperature. Control (Zn-free) values, i.e. the arm of that
                paper without added zinc, as reported: saturation
                Kd = 2.9 +/- 0.1 nM; association rate constant
                kon = (3.4 +/- 0.4) x 10^8 M^-1 min^-1; BIPHASIC
                dissociation, koff(fast) = 2.33 +/- 0.31 min^-1 and
                koff(slow) = 0.21 +/- 0.02 min^-1; serotonin Ki = 8.7 +/-
                0.6 nM in the same assay. Both numbers this module needs
                for the agonist pair therefore come from one preparation
                and one radioligand, the [Volpe2011]-style guarantee the
                D2 section could not get.
  [Parkel2006]  Parkel S, Rinken A. "Characteristics of binding of
                [3H]WAY100635 to rat hippocampal membranes." Neurochem Res.
                2006;31(9):1135-1140. doi:10.1007/s11064-006-9135-6. Rat
                hippocampal membranes, 25 C: saturation Kd = 87 +/- 4 pM
                (Bmax 15.1 +/- 0.2 fmol/mg protein) and dissociation rate
                constant koff = (7.8 +/- 1.1) x 10^-3 min^-1 -- so slow the
                authors note equilibrium is not reached before 7.5 h of
                incubation. They also report positive cooperativity in the
                association phase, which is why the kon below is DERIVED
                from koff and Kd rather than taken from their association
                fits (see "Derivations").
  [NewmanTancredi2003] Newman-Tancredi A, Rivet JM, Cussac D, Touzard M,
                Chaput C, Marini L, Millan MJ. "Comparison of hippocampal G
                protein activation by 5-HT1A receptor agonists and the
                atypical antipsychotics clozapine and S16924." Naunyn
                Schmiedebergs Arch Pharmacol. 2003;368(3):188-199.
                doi:10.1007/s00210-003-0788-2. [35S]GTPgammaS binding in
                rat hippocampal membranes, i.e. POSTSYNAPTIC 5-HT1A
                receptors in native tissue. Maximum efficacy relative to
                5-HT = 100: (+)8-OH-DPAT 85, flesinoxan 62, eltoprazine 60,
                buspirone 41, clozapine 22, pindolol 7, WAY100,635 0,
                spiperone 0. Two of this module's three 5-HT1A efficacies
                are read straight off that one list.
  [Forster1995] Forster EA, Cliffe IA, Bill DJ, Dover GM, Jones D, Reilly
                Y, Fletcher A. "A pharmacological profile of the selective
                silent 5-HT1A receptor antagonist, WAY-100635." Eur J
                Pharmacol. 1995;281(1):81-88.
                doi:10.1016/0014-2999(95)00234-c. pIC50 8.87 displacing
                [3H]8-OH-DPAT from rat hippocampal membranes, >100-fold
                selective; "no evidence of any 5-HT1A receptor agonist or
                partial agonist activity"; and, in guinea-pig ileum, an
                INSURMOUNTABLE antagonist at high concentrations. That last
                observation and [Parkel2006]'s koff are the same fact seen
                from two directions (see "What the numbers turned out to
                say").
  [Fletcher1996] Fletcher A, Forster EA, Bill DJ, Brown G, Cliffe IA,
                Hartley JE, Jones DE, McLenachan A, Stanhope KJ, Critchley
                DJ, Childs KJ, Middlefell VC, Lanfumey L, Corradetti R,
                Laporte AM, Gozlan H, Hamon M, Dourish CT.
                "Electrophysiological, biochemical, neurohormonal and
                behavioural studies with WAY-100635, a potent, selective
                and silent 5-HT1A receptor antagonist." Behav Brain Res.
                1996;73(1-2):337-353. doi:10.1016/0166-4328(96)00118-0.
                IC50 1.35 nM against [3H]8-OH-DPAT in rat hippocampus; the
                paper that established "silent" as opposed to "partial
                agonist" for this compound.

5-HT2A binding and efficacy:

  [Denomme2024] Denomme N, Hernandez CC, Bock HA, Ohana RF, Bakshi S,
                Sherwood AM, McCorvy JD, Daley PF, Callaway WB, Hull JM,
                Alt A, Isom LL, Cozzi NV. "N-(4-Bromo-2,5-
                Dimethoxyphenethyl)-6-(4-Phenylbutoxy)Hexan-1-Amine (XOB):
                A Novel Phenylalkylamine Antagonist of Serotonin 2A
                Receptors and Voltage-Gated Sodium Channels." Mol
                Pharmacol. 2024;106(2):92-106.
                doi:10.1124/molpharm.123.000837. NanoBRET kinetic binding
                at HiBiT-tagged human 5-HT2A receptors in live HEK293
                cells, room temperature: serotonin kon = 0.42 x 10^6
                M^-1 s^-1, koff = 0.063 s^-1, Kd ~150 nM. Both rate
                constants measured, in one live-cell assay -- the only
                5-HT2A binding-kinetics measurement found for any ligand in
                this set.
  [McKenna1989] McKenna DJ, Peroutka SJ. "Differentiation of 5-
                hydroxytryptamine2 receptor subtypes using
                125I-R-(-)2,5-dimethoxy-4-iodo-phenylisopropylamine and
                3H-ketanserin." J Neurosci. 1989;9(10):3482-3490.
                doi:10.1523/jneurosci.09-10-03482.1989. Rat cortex, one
                paper, both radioligands: [3H]ketanserin saturation
                Kd = 1.0 +/- 0.1 nM (Bmax 11 +/- 0.4 pmol/g tissue) and
                [125I]R-(-)DOI saturation Kd = 0.63 +/- 0.09 nM (Bmax
                2.5 +/- 0.2 pmol/g); serotonin Ki = 750 +/- 50 nM against
                [3H]ketanserin and 3.0 +/- 0.7 nM against [125I]DOI. That
                250-fold spread in ONE tissue is the agonist high-/low-
                affinity state problem, and it is the largest single
                uncertainty in this module (see "The affinity state
                problem").
  [Seggel1990]  Seggel MR, Yousif MY, Lyon RA, Titeler M, Roth BL, Suba EA,
                Glennon RA. "A structure-affinity study of the binding of
                4-substituted analogues of 1-(2,5-dimethoxyphenyl)-2-
                aminopropane at 5-HT2 serotonin receptors." J Med Chem.
                1990;33(3):1032-1036. DOI pKi 7.72 (19.1 nM) at
                [3H]ketanserin-labelled rat cortical 5-HT2 sites. Read from
                the compilation in [Hemanth2023] Table 1-A rather than from
                the paper itself, which is paywalled; the same table gives
                DOI pKi 9.15 (0.71 nM) at cloned human 5-HT2A against
                [125I]DOI from [Nelson1999] (Nelson DL, Lucaites VL,
                Wainscott DB, Glennon RA. Naunyn Schmiedebergs Arch
                Pharmacol. 1999;359(1):1-6), the same two-state spread
                [McKenna1989] measures for serotonin.
  [Hemanth2023] Hemanth P, Nistala P, Nguyen VT, Eltit JM, Glennon RA,
                Dukat M. "Binding and functional structure-activity
                similarities of 4-substituted 2,5-dimethoxyphenyl
                isopropylamine analogues at 5-HT2A and 5-HT2B serotonin
                receptors." Front Pharmacol. 2023;14:1101290.
                doi:10.3389/fphar.2023.1101290 (open access, PMC9902381).
                Source of the two DOI affinities above, each attributed to
                its primary.
  [Cussac2008]  Cussac D, Boutet-Robinet E, Ailhaud MC, Newman-Tancredi A,
                Martel JC, Danty N, Rauly-Lestienne I. "Agonist-directed
                trafficking of signalling at serotonin 5-HT2A, 5-HT2B and
                5-HT2C-VSV receptors mediated Gq/11 activation and calcium
                mobilisation in CHO cells." Eur J Pharmacol.
                2008;594(1-3):32-38. DOI Emax = 81.3% of the serotonin
                maximum in h5-HT2A Gq/11-coupled calcium mobilisation. The
                value is quoted here as reported in [Wallach2023] (Wallach
                J, Cao AB, Calkins MM, et al. "Identification of 5-HT2A
                receptor signaling pathways associated with psychedelic
                potential." Nat Commun. 2023;14:8221.
                doi:10.1038/s41467-023-44016-1, open access), which cites
                [Cussac2008] for it while comparing it with LSD (84.6%) and
                lisuride (48.6%) from the same assay.
  [Egan1998]    Egan C, Herrick-Davis K, Teitler M. "Creation of a
                constitutively activated state of the 5-HT2A receptor by
                site-directed mutagenesis: revelation of inverse agonist
                activity of antagonists." Ann N Y Acad Sci.
                1998;861:136-139. At a constitutively active 5-HT2A mutant,
                every competitive antagonist tested reduced basal inositol
                phosphate accumulation, i.e. behaved as an inverse agonist.
                Cited for what this module CANNOT represent (see
                "ketanserin" below).

Anatomy and mechanism (no numbers taken from these; they justify which node
carries which receptor and with what sign):

  [Sprouse1987] Sprouse JS, Aghajanian GK. "Electrophysiological responses
                of serotoninergic dorsal raphe neurons to 5-HT1A and 5-HT1B
                agonists." Synapse. 1987;1(1):3-9.
                doi:10.1002/syn.890010103. 8-OH-DPAT suppresses dorsal
                raphe 5-HT neuron firing through the somatodendritic 5-HT1A
                autoreceptor.
  [Riad2000]    Riad M, Garcia S, Watkins KC, Jodoin N, Doucet E, Langlois
                X, el Mestikawy S, Hamon M, Descarries L. "Somatodendritic
                localization of 5-HT1A and preterminal axonal localization
                of 5-HT1B serotonin receptors in adult rat brain." J Comp
                Neurol. 2000;417(2):181-194. 5-HT1A protein is
                somatodendritic in BOTH its autoreceptor (raphe) and
                heteroreceptor (hippocampus) roles, and absent from axon
                terminals. This is why `Htr1a` is not a terminal-field
                mismatch case (see `expression/mismatch.py`).
  [AmargosBosch2004] Amargos-Bosch M, Bortolozzi A, Puig MV, Serrats J,
                Adell A, Celada P, Toth M, Mengod G, Artigas F.
                "Co-expression and in vivo interaction of serotonin1A and
                serotonin2A receptors in pyramidal neurons of prefrontal
                cortex." Cereb Cortex. 2004;14(3):281-299.
                doi:10.1093/cercor/bhg128.
  [Santana2017] Santana N, Artigas F. "Laminar and Cellular Distribution of
                Monoamine Receptors in Rat Medial Prefrontal Cortex." Front
                Neuroanat. 2017;11:87. doi:10.3389/fnana.2017.00087 (open
                access).
  [Willins1997] Willins DL, Deutch AY, Roth BL. "Serotonin 5-HT2A receptors
                are expressed on pyramidal cells and interneurons in the
                rat cortex." Synapse. 1997;27(1):79-82.
  [AghajanianMarek1997] Aghajanian GK, Marek GJ. "Serotonin induces
                excitatory postsynaptic potentials in apical dendrites of
                neocortical pyramidal cells." Neuropharmacology.
                1997;36(4-5):589-599. doi:10.1016/s0028-3908(97)00051-8.
  [Puig2003]    Puig MV, Celada P, Diaz-Mataix L, Artigas F. "In vivo
                modulation of the activity of pyramidal neurons in the rat
                medial prefrontal cortex by 5-HT2A receptors: relationship
                to thalamocortical afferents." Cereb Cortex.
                2003;13(8):870-882. doi:10.1093/cercor/13.8.870. IV DOI
                excited 21/56 and inhibited 17/56 identified mPFC pyramidal
                neurons. The reason this module puts 5-HT2A on the
                pyramidal node only (see "What was deliberately not
                built").

Derivations
-----------

Both sites need (kon, koff) per ligand and the literature supplies
different subsets per ligand, so three different derivations are used and
each is stated at its ligand:

1. **Both constants measured** -- use them: 8-OH-DPAT's kon and serotonin's
   kon/koff at 5-HT2A.
2. **koff and Kd measured, kon missing** -- kon = koff / Kd, the mu-opioid
   pattern (`scenarios.mu_opioid_ligands_calibrated`). Used for
   WAY-100635.
3. **Only an equilibrium affinity measured** -- share the site's one
   MEASURED kon and set koff = kon * Ki, the GABA-A pattern
   (`scenarios.gaba_a_benzodiazepine_ligands_calibrated`). Used for
   serotonin at 5-HT1A, and for DOI and ketanserin at 5-HT2A.

Derivation 3 is the same assumption the GABA-A section makes and it is
still an assumption, but it is better anchored here: there, `kon_shared`
was picked to match the order of magnitude of another receptor system's
numbers and was sourced from nothing; here it is a measured association
rate constant for a ligand binding *the same site in the same kind of
preparation* ([Satala2016] for 5-HT1A, [Denomme2024] for 5-HT2A). What is
assumed is that the other ligands at that site associate at a similar rate,
not what that rate is.

What the numbers turned out to say
----------------------------------

Three findings worth recording, because none of them was assumed going in:

* **WAY-100635 is a nearly irreversible antagonist.** koff = 0.0078 min^-1
  is a dissociation half-life of 89 minutes, ~130x slower than the agonist
  8-OH-DPAT's derived koff and ~300x slower than this codebase's calibrated
  naloxone (2.375 min^-1). [Forster1995] independently reports
  insurmountable antagonism at high concentrations in an isolated-tissue
  functional assay, which is what a residence time that long looks like
  from the pharmacology side. The practical consequence in this model is
  the opposite of the mu-opioid scenario's: naloxone reverses fentanyl only
  when given in large molar excess (because its koff is FASTER than
  fentanyl's), whereas WAY-100635 reverses 8-OH-DPAT at modest excess
  because it simply does not let go.
* **The 5-HT1A/5-HT2A affinity split for serotonin is large and it is
  measured, not chosen.** Serotonin's Kd is 8.7 nM at 5-HT1A and ~150 nM at
  5-HT2A: a ~17-fold difference from two independent labs. Everything this
  module predicts about which receptor a given serotonin concentration
  engages first follows from those two published numbers and from nothing
  hand-set.
* **Both selective agonists are strong partial agonists, not full ones.**
  8-OH-DPAT reaches 85% of serotonin's maximum at postsynaptic 5-HT1A
  ([NewmanTancredi2003]) and DOI 81.3% at 5-HT2A ([Cussac2008]). The
  textbook shorthand "8-OH-DPAT, the prototypical 5-HT1A agonist" hides a
  15% efficacy deficit that this codebase's `Ligand.efficacy` can carry
  directly.

The affinity state problem, and the one number it forced a choice on
--------------------------------------------------------------------

[McKenna1989] measures serotonin's affinity for 5-HT2A twice in the same
rat cortex and gets 3.0 nM against an agonist radioligand and 750 nM
against an antagonist radioligand. That is not a discrepancy, it is the
two-state behaviour of an agonist at a G-protein-coupled receptor: the
agonist radioligand labels the G-protein-coupled high-affinity state, the
antagonist radioligand labels the whole population. `ReceptorSite` has one
pool and one affinity per ligand, so one of them has to be chosen.

This module takes serotonin's LIVE-CELL value, Kd ~150 nM
([Denomme2024]), for three reasons: it is a measurement in intact cells
where the coupled and uncoupled states are present in whatever ratio they
actually occur in; it comes with the only measured kon and koff for any
ligand at this site, so choosing it means the reference ligand needs no
derivation at all; and it lands between the two membrane values (3.0 and
750), which is what a mixed-state population should do.

DOI and ketanserin then take their affinities from antagonist-radioligand
membrane assays, which is a mixed-format set. The consequence is bounded
rather than waved at: with 5-HT at 150 nM the model's ketanserin/serotonin
affinity ratio is 150, where an all-[3H]ketanserin set would make it 750.
`ligands_5ht2a(affinity_state="antagonist-radioligand")` builds that second
set, and every qualitative result in `tests/test_serotonin_scenario.py` is
asserted under both, the same discipline P4.3 used for the hand-written
versus atlas-derived mu-opioid density maps.

Searched and empty
------------------

* **Rodent pharmacokinetics for 8-OH-DPAT, WAY-100635, DOI and
  ketanserin.** Searched Europe PMC for each compound crossed with
  "pharmacokinetics" / "half-life" / "plasma" / "brain concentration" in
  rat and mouse. Nothing usable came back: the hits are reviews of other
  drugs, PET occupancy studies that report occupancy over time without a
  plasma curve, and papers using these compounds as tools without
  measuring their disposition. So all four get ONE shared illustrative
  elimination rate constant (`TOOL_COMPOUND_KE`) rather than four
  hand-picked ones, because four different guesses would assert a
  clearance ORDERING that no source supports. Phase 1's dose->brain
  bridge (`drug_pk.py`) is the machinery this would plug into the moment a
  real PK dataset appears for any of them.
* **Binding kinetics (kon or koff) for DOI and ketanserin at 5-HT2A, and
  for serotonin at 5-HT1A.** Searched for "[3H]ketanserin" with
  "association"/"dissociation rate constant", for 5-HT2A "residence time"
  with NanoBRET/TR-FRET, and for the 5-HT2A entries of the antipsychotic
  kinetic panels ([Sykes2017]-style). The classic [3H]ketanserin
  characterisation (Leysen JE, Niemegeers CJ, Van Nueten JM, Laduron PM.
  Mol Pharmacol. 1982;21(2):301-314) does report association and
  dissociation kinetics, but it is paywalled with no abstract in Europe
  PMC, so its rate constants could not be read; nothing citable was
  obtained from it. Hence derivation 3 for those three ligands.

What was deliberately not built
-------------------------------

* **5-HT2A on cortical interneurons.** It is really there ([Willins1997]),
  and it is why [Puig2003] finds DOI exciting 21 of 56 and inhibiting 17 of
  56 identified mPFC pyramidal neurons. Adding it would need the ratio of
  the interneuron gain to the pyramidal gain, and that ratio sets both the
  size and, near rest, the SIGN of the cortical response: linearized at
  rest the response is `(u_pyr - 0.9 u_gaba) / 1.72`, which changes sign
  once the interneuron gain passes 1.11x the pyramidal one, and at a
  saturating dose the interneuron rails and the response is compressed from
  +0.26 to +0.04 as the ratio goes from 0 to 2. No source found gives that
  ratio, so putting a number there would be choosing the sign of a
  published-as-mixed result by hand. The interneuron node exists and
  carries no 5-HT2A; `PFC_GABA` is in the network for the recurrent
  inhibition it provides, not as a drug target.
* **An atlas-derived density factor.** P4.2/P4.3 factor a coupling into
  `mechanism_gain * rho[region]` with rho from the Allen ISH maps, and
  `Htr1a`/`Htr2a` are already in `expression/genes.py` for exactly that.
  It is not used here because the committed expression fixture carries
  `Htr2a` and not `Htr1a`, so a density-derived 5-HT1A map would make
  every test in this module require the gitignored 7.5 MB cache. The
  hook is one call to `expression.scenario_maps.scenario_coupling` once an
  `Htr1a` fixture slice exists; `HT1A_MECHANISM_GAIN` and
  `HT2A_MECHANISM_GAIN` are already the `mechanism_gain` half of that
  factorization, stated on their own.
* **A 5-HT1A partial-agonist efficacy that differs between raphe and
  cortex.** [NewmanTancredi2003]'s 85% is measured at postsynaptic
  (hippocampal) receptors. 8-OH-DPAT is widely described as behaving as a
  FULL agonist at somatodendritic autoreceptors, which is a receptor-
  reserve effect: the same intrinsic efficacy produces a maximal response
  where receptors are in excess of the G proteins needed. `Ligand.efficacy`
  is one number per ligand per site, and receptor reserve is not
  representable at all in a model whose activation term is linear in
  occupancy. Using 85% everywhere therefore understates the autoreceptor
  response, and that is the honest direction to be wrong in: the
  alternative is asserting a second, unmeasured efficacy.
"""

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite

# ---------------------------------------------------------------------------
# 5-HT1A: measured constants, and the ligands built from them
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. MEASURED association rate constant of [3H]-8-OH-DPAT at
#: human 5-HT1A: (3.4 +/- 0.4) x 10^8 M^-1 min^-1 [Satala2016], i.e.
#: 3.4e8 * 1e-9 = 0.34 nM^-1 min^-1. Used for 8-OH-DPAT itself, and shared
#: with serotonin by derivation 3.
HT1A_KON = 0.34

#: nM. MEASURED saturation Kd of [3H]-8-OH-DPAT, human 5-HT1A/HEK293
#: [Satala2016].
HT1A_KD_8OHDPAT = 2.9

#: min^-1. The two MEASURED dissociation phases of [3H]-8-OH-DPAT
#: [Satala2016]: a fast and a slow component, the standard signature of an
#: agonist seeing two affinity states. `ReceptorSite` has one pool, so the
#: koff actually used is `HT1A_KON * HT1A_KD_8OHDPAT` -- the single-state
#: collapse that reproduces the paper's own saturation Kd. That collapse is
#: not free to be anything: it must land between these two phases, and
#: `tests/test_serotonin_scenario.py` asserts that it does.
HT1A_KOFF_8OHDPAT_FAST = 2.33
HT1A_KOFF_8OHDPAT_SLOW = 0.21

#: nM. MEASURED Ki of serotonin against [3H]-8-OH-DPAT in the same assay
#: [Satala2016].
HT1A_KI_SEROTONIN = 8.7

#: nM and min^-1. MEASURED saturation Kd (87 +/- 4 pM) and dissociation
#: rate constant ((7.8 +/- 1.1) x 10^-3 min^-1) of [3H]WAY100635 in rat
#: hippocampal membranes at 25 C [Parkel2006].
HT1A_KD_WAY100635 = 0.087
HT1A_KOFF_WAY100635 = 0.0078

#: Efficacies relative to serotonin = 1.0, from ONE [35S]GTPgammaS panel in
#: rat hippocampal membranes [NewmanTancredi2003]: (+)8-OH-DPAT 85,
#: WAY100,635 0.
HT1A_EFFICACY_8OHDPAT = 0.85
HT1A_EFFICACY_WAY100635 = 0.0


def ligands_5ht1a():
    """serotonin / 8-OH-DPAT / WAY-100635 at 5-HT1A, in that order.

    Every affinity and every efficacy below is a published measurement; see
    the module docstring for the sources and for which of the three
    derivations each ligand's rate constants come from.
    """
    serotonin = Ligand(
        name="serotonin",
        # Derivation 3: the site's one measured kon, shared.
        kon=HT1A_KON,
        # min^-1; HT1A_KON * Ki = 0.34 * 8.7 [Satala2016].
        koff=2.958,
        # The reference agonist of [NewmanTancredi2003]'s efficacy scale,
        # by construction rather than by measurement.
        efficacy=1.0,
    )
    dpat = Ligand(
        name="8-OH-DPAT",
        # Derivation 1, measured [Satala2016].
        kon=HT1A_KON,
        # min^-1; HT1A_KON * Kd = 0.34 * 2.9 [Satala2016]. The single-state
        # collapse of a measured biphasic dissociation (2.33 fast / 0.21
        # slow); it reproduces that paper's own saturation Kd and sits
        # between its two phases.
        koff=0.986,
        efficacy=HT1A_EFFICACY_8OHDPAT,
    )
    way = Ligand(
        name="WAY-100635",
        # Derivation 2: koff / Kd = 0.0078 / 0.087 [Parkel2006].
        kon=0.08966,
        koff=HT1A_KOFF_WAY100635,  # min^-1, measured [Parkel2006]
        efficacy=HT1A_EFFICACY_WAY100635,  # silent antagonist
        # [Forster1995], [Fletcher1996], [NewmanTancredi2003]: Emax 0 in
        # [35S]GTPgammaS, and no agonist or partial-agonist activity in any
        # functional assay those papers ran. Unlike ketanserin below, this
        # zero is measured rather than a modelling simplification.
    )
    return [serotonin, dpat, way]


def receptor_site_5ht1a():
    """One competitive 5-HT1A pool.

    ONE pool for both the raphe autoreceptor and the cortical postsynaptic
    receptor, which is an approximation and a defensible one here: the two
    populations are the same receptor protein, and in this model they see
    the same ligand concentration (the PK layer is a single systemic
    compartment), so their occupancies would be identical anyway. It stops
    being defensible the moment Phase 6 gives the raphe and the cortex
    different local serotonin concentrations, which is the point at which
    this becomes two sites.
    """
    return ReceptorSite(name="5HT1A_raphe_and_cortex", ligands=ligands_5ht1a())


# ---------------------------------------------------------------------------
# 5-HT2A: measured constants, and the ligands built from them
# ---------------------------------------------------------------------------

#: nM^-1 min^-1 and min^-1. MEASURED association and dissociation rate
#: constants of serotonin at human 5-HT2A in live HEK293 cells by NanoBRET
#: [Denomme2024]: kon = 0.42e6 M^-1 s^-1 = 0.42e6 * 60 * 1e-9 nM^-1 min^-1;
#: koff = 0.063 s^-1 * 60. Their ratio is the paper's reported Kd of
#: ~150 nM, which is the consistency check in the tests.
HT2A_KON = 0.0252
HT2A_KOFF_SEROTONIN = 3.78

#: nM. Serotonin's Kd at 5-HT2A implied by the two measured rate constants
#: above, and the affinity this module uses for it. See "The affinity state
#: problem" in the module docstring for why the live-cell number and not
#: [McKenna1989]'s 3.0 nM (agonist radioligand) or 750 nM (antagonist
#: radioligand).
HT2A_KD_SEROTONIN = 150.0

#: nM. MEASURED saturation Kd of [3H]ketanserin in rat cortex
#: [McKenna1989].
HT2A_KD_KETANSERIN = 1.0

#: nM. DOI at [3H]ketanserin-labelled rat cortical 5-HT2 sites: pKi 7.72
#: [Seggel1990], i.e. 10^(9 - 7.72) = 19.05 nM.
HT2A_KI_DOI = 19.05

#: nM. The same two quantities in the OTHER affinity state, kept because
#: they are what `affinity_state="antagonist-radioligand"` uses and because
#: the spread between the states is the honest error bar on this site's
#: numbers: serotonin Ki 750 nM against [3H]ketanserin [McKenna1989];
#: 3.0 nM against [125I]DOI in the same paper; DOI 0.71 nM (pKi 9.15) at
#: cloned human 5-HT2A against [125I]DOI [Nelson1999].
HT2A_KI_SEROTONIN_ANTAGONIST_STATE = 750.0
HT2A_KI_SEROTONIN_AGONIST_STATE = 3.0
HT2A_KI_DOI_AGONIST_STATE = 0.71

#: Efficacy relative to serotonin = 1.0: DOI Emax 81.3% of the serotonin
#: maximum in h5-HT2A Gq/11-coupled calcium mobilisation [Cussac2008].
HT2A_EFFICACY_DOI = 0.813

#: Ketanserin. 0.0 is a MODELLING SIMPLIFICATION: the only efficacy in this
#: module set by an argument rather than by an assay result, where
#: WAY-100635's 0.0 is a measured Emax. [Egan1998] shows competitive
#: 5-HT2A antagonists reduce basal signalling at a constitutively active
#: receptor, i.e. they are inverse agonists rather than neutral. An inverse
#: agonist can only be expressed against constitutive activity, and
#: `ReceptorSite.activation` is `sum(occupancy * efficacy)` with no
#: constitutive term: a negative efficacy here would drive the node BELOW
#: its drug-free baseline, which is not what an inverse agonist does to a
#: receptor with no constitutive tone. Adding a constitutive-activity term
#: is a real engine extension with its own literature requirement (how much
#: constitutive activity, per receptor, in vivo) and is not in this task.
HT2A_EFFICACY_KETANSERIN = 0.0

#: The affinity sets `ligands_5ht2a` can build. See "The affinity state
#: problem" in the module docstring.
HT2A_AFFINITY_STATES = ("live-cell", "antagonist-radioligand")


def ligands_5ht2a(affinity_state="live-cell"):
    """serotonin / DOI / ketanserin at 5-HT2A, in that order.

    `affinity_state` selects serotonin's affinity, which is the one number
    at this site that the literature reports differently depending on the
    assay format:

      "live-cell" (default)      Kd 150 nM, with kon and koff both measured
                                 in intact cells [Denomme2024].
      "antagonist-radioligand"   Ki 750 nM against [3H]ketanserin in rat
                                 cortex [McKenna1989], which is the same
                                 format DOI's and ketanserin's affinities
                                 come from, at the cost of overriding a
                                 directly measured koff with a derived one.

    DOI and ketanserin are identical in both sets. The tests assert every
    qualitative result under both.
    """
    if affinity_state not in HT2A_AFFINITY_STATES:
        raise ValueError(
            "affinity_state must be one of %s, got %r"
            % (list(HT2A_AFFINITY_STATES), affinity_state)
        )

    if affinity_state == "live-cell":
        serotonin = Ligand(
            name="serotonin",
            # Derivation 1: both constants measured [Denomme2024].
            kon=HT2A_KON,
            koff=HT2A_KOFF_SEROTONIN,
            efficacy=1.0,
        )
    else:
        serotonin = Ligand(
            name="serotonin",
            # Derivation 3, with the site's measured kon and the
            # antagonist-radioligand Ki: koff = 0.0252 * 750 [McKenna1989].
            kon=HT2A_KON,
            koff=18.9,
            efficacy=1.0,
        )
    doi = Ligand(
        name="DOI",
        # Derivation 3: the site's measured kon, shared.
        kon=HT2A_KON,
        # min^-1; HT2A_KON * Ki = 0.0252 * 19.05 [Seggel1990].
        koff=0.48006,
        efficacy=HT2A_EFFICACY_DOI,
    )
    ketanserin = Ligand(
        name="ketanserin",
        kon=HT2A_KON,
        # min^-1; HT2A_KON * Kd = 0.0252 * 1.0 [McKenna1989].
        koff=0.0252,
        efficacy=HT2A_EFFICACY_KETANSERIN,  # simplification; see the constant
    )
    return [serotonin, doi, ketanserin]


def receptor_site_5ht2a(affinity_state="live-cell"):
    """One competitive 5-HT2A pool, on the cortical pyramidal node."""
    return ReceptorSite(
        name="5HT2A_on_cortical_pyramidal",
        ligands=ligands_5ht2a(affinity_state=affinity_state),
    )


# ---------------------------------------------------------------------------
# Pharmacokinetics: ILLUSTRATIVE, and one number rather than four
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, one elimination rate constant
#: shared by all four tool compounds (8-OH-DPAT, WAY-100635, DOI,
#: ketanserin). No rodent PK was found for any of them (see "Searched and
#: empty" in the module docstring), and four separately invented values
#: would assert a clearance ordering nothing supports; one shared value
#: asserts only that all four are still on board over this scenario's
#: ~1 h window, which is the property the occupancy demonstrations need.
#: Do not cite this as a measured half-life for any of these compounds.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY, exactly as for dopamine in
#: `scenarios.d2_pk_models_calibrated`: serotonin here is not a dosed drug
#: but a transmitter, cleared by SERT-mediated reuptake, which is a
#: saturable Michaelis-Menten transporter process rather than first-order
#: elimination and runs on a sub-second-to-second timescale, three to four
#: orders faster than this codebase's minute-scale time unit. Taken
#: literally it would collapse a serotonin "dose" into a delta function
#: inside one integration step. 5.0 min^-1 (t1/2 ~8.3 s) preserves the only
#: property the dynamics need -- transmitter clears much faster than any
#: dosed drug -- at a timescale the integrator resolves. The real quantity
#: belongs to P5.8 (SERT) and P6.2 (endogenous tone), not here.
SEROTONIN_KE = 5.0


def pk_models_5ht1a():
    """One PK model per 5-HT1A ligand, in `ligands_5ht1a()` order."""
    return [
        OneCompartmentPK(ke=SEROTONIN_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


def pk_models_5ht2a():
    """One PK model per 5-HT2A ligand, in `ligands_5ht2a()` order."""
    return [
        OneCompartmentPK(ke=SEROTONIN_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: ILLUSTRATIVE throughout
# ---------------------------------------------------------------------------
#
# Three nodes, and the edge that is missing on purpose:
#
#   DRN_5HT          dorsal raphe serotonin neurons. Carries the
#                    somatodendritic 5-HT1A autoreceptor [Sprouse1987],
#                    [Riad2000]. NO edges: see the module docstring. This is
#                    the same shape as `LC` in the multi-region mu-opioid
#                    scenario, and for a related reason -- a receptor
#                    target whose downstream projection this codebase
#                    cannot yet represent.
#   PFC_Pyramidal    layer V prefrontal pyramidal neurons. Carries BOTH
#                    postsynaptic 5-HT1A (Gi, inhibitory) and 5-HT2A (Gq,
#                    excitatory) on the same node, which is the anatomy
#                    [AmargosBosch2004] and [Santana2017] report.
#   PFC_GABA         local cortical GABAergic interneurons. Carries no
#                    receptor here (see "What was deliberately not built"),
#                    and exists to give the pyramidal node the recurrent
#                    inhibition that makes its response finite.
#
# Every weight, tau, gain and baseline_input below is ILLUSTRATIVE. Two
# stated rules set them, so that "illustrative" does not mean "tuned until
# the tests passed":
#
#   1. Baselines put every node at rate 0.5 at rest with no drug, the
#      maximum-sensitivity point `stability.MAX_SENSITIVITY_RATE` derives.
#      With W[Pyr][GABA] = -0.9 and W[GABA][Pyr] = 0.8 that fixes
#      baseline_input = +0.45 for the pyramidal node and -0.4 for the
#      interneuron, exactly; nothing is left to choose.
#   2. The loop weights satisfy P3.4's global-uniqueness certificate,
#      `stability.contraction_number` = sigma_max(diag(g/4) W) < 1, which
#      with gain 4 everywhere is max(0.9, 0.8) = 0.9. So this network has
#      exactly one fixed point and every trajectory converges to it,
#      whatever the taus - the property P3.4 built the check for, asserted
#      on this network in the tests.





def raphe_cortical_network():
    """DRN_5HT (isolated) + PFC_Pyramidal -| PFC_GABA local cortical loop.

    See the section comment above for what each node is and why there is no
    raphe -> cortex edge.
    """
    nodes = [
        CircuitNode(name="DRN_5HT", tau=1.0, baseline_input=0.0, gain=4.0),
        CircuitNode(name="PFC_Pyramidal", tau=1.5, baseline_input=0.45, gain=4.0),
        CircuitNode(name="PFC_GABA", tau=1.0, baseline_input=-0.4, gain=4.0),
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    W[idx["PFC_Pyramidal"]][idx["PFC_GABA"]] = -0.9  # local inhibition
    W[idx["PFC_GABA"]][idx["PFC_Pyramidal"]] = 0.8  # recurrent excitation
    return CircuitNetwork(nodes=nodes, connectivity=W)


#: ILLUSTRATIVE. How much a unit of 5-HT1A activation changes the drive of
#: the node carrying it. Negative because 5-HT1A is Gi/o-coupled and
#: hyperpolarizing.
#:
#: Magnitude 1.0 by a stated rule rather than by tuning: activation is a
#: dimensionless fraction in [0, 1], so a magnitude of 1 means a fully
#: occupied receptor bound by a full agonist shifts the node's input by one
#: unit, which at gain 4 carries it most of the way across its sigmoid.
#: That is what a saturating dose of a full agonist should do. The value is
#: NOT identifiable from anything published: P2.3 established that only the
#: product of a coupling gain and a connection weight is identifiable even
#: when behavioural data exists, and no behavioural anchor was sought here
#: because Phase 5 is a receptor-layer task.
#:
#: ONE gain for both target nodes, following P4.3's correction: the same
#: Gi-coupled receptor reducing the drive of the cell that carries it, and
#: no data in reach distinguishes its per-receptor efficacy between a raphe
#: serotonin neuron and a cortical pyramidal neuron. The DENSITY difference
#: between those regions is the missing factor `rho` of P4.2's
#: `coupling = mechanism_gain * rho`; see "What was deliberately not built".
HT1A_MECHANISM_GAIN = -1.0

#: ILLUSTRATIVE. The 5-HT2A counterpart. Positive: 5-HT2A is Gq/11-coupled
#: and depolarizing ([AghajanianMarek1997] for the postsynaptic excitation
#: of layer V pyramidal cells this stands for). Same magnitude rule as
#: above; the equal magnitude of the two gains is a consequence of that
#: rule, not a claim that the two receptors are equally powerful.
HT2A_MECHANISM_GAIN = 1.0


def coupling_5ht1a():
    """5-HT1A activation -> drive, on the raphe autoreceptor node and the
    cortical pyramidal node."""
    return {
        "DRN_5HT": HT1A_MECHANISM_GAIN,  # somatodendritic autoreceptor
        "PFC_Pyramidal": HT1A_MECHANISM_GAIN,  # postsynaptic heteroreceptor
    }


def coupling_5ht2a():
    """5-HT2A activation -> drive, on the cortical pyramidal node only."""
    return {"PFC_Pyramidal": HT2A_MECHANISM_GAIN}


def receptor_system_5ht1a():
    """The 5-HT1A stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_5ht1a(),
        receptor_site=receptor_site_5ht1a(),
        coupling=coupling_5ht1a(),
        name="5HT1A",
    )


def receptor_system_5ht2a(affinity_state="live-cell"):
    """The 5-HT2A stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_5ht2a(),
        receptor_site=receptor_site_5ht2a(affinity_state=affinity_state),
        coupling=coupling_5ht2a(),
        name="5HT2A",
    )


def serotonin_receptor_systems(affinity_state="live-cell"):
    """Both systems, 5-HT1A first, ready for `Simulation(network, ...)`.

    Serotonin appears once per system, as two separate `Ligand` objects
    with two separate concentration states. That is not a duplication bug:
    they are two distinct receptor pools and the engine tracks occupancy
    per pool. It does mean a serotonin "dose" has to be applied to both
    entries to represent one rise in serotonin, which
    `Simulation.dose_vector(serotonin=...)` does by itself, because it
    matches ligands by name across every system.
    """
    return [
        receptor_system_5ht1a(),
        receptor_system_5ht2a(affinity_state=affinity_state),
    ]


def serotonergic_tone_readout():
    """Dorsal raphe serotonin-neuron firing.

    Falls under a 5-HT1A agonist: that is the autoreceptor. This is the
    same shape of readout as `scenarios.arousal_index_readout` (one node's
    own rate, suppressed by a Gi-coupled agonist), and like it, it is a
    circuit-level quantity and not a fitted behavioural endpoint. Naming a
    behaviour would require Phase 7.
    """
    return BehaviorReadout(name="serotonergic_tone", weights={"DRN_5HT": 1.0})


def cortical_output_readout():
    """Layer V prefrontal pyramidal firing: the node both receptors reach."""
    return BehaviorReadout(name="cortical_output", weights={"PFC_Pyramidal": 1.0})
