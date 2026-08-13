"""P5.3: the noradrenaline alpha1, alpha2 and beta adrenoceptor systems.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter below): every affinity
and every efficacy, for five ligands at three receptor sites --
noradrenaline at alpha1A, alpha2A and beta1; prazosin at alpha1A;
clonidine and yohimbine at alpha2A; propranolol at beta1. Two rate
constants are measured directly (prazosin's `koff` at alpha1, propranolol's
`kon` and `koff` at beta1); the rest are derivation 3 (below) off measured
equilibrium affinities.

ILLUSTRATIVE (hand-set, labelled, no literature target): the three-node
locus-coeruleus/cortex circuit's weights, taus, gains and baseline inputs;
the three mechanism gains that turn receptor activation into node drive;
and the pharmacokinetics of all four dosed drugs plus the noradrenaline
clearance proxy. As in every scenario before Phase 7, this task calibrates
the receptor layer, not the circuit layer.

Why three sites and not one
---------------------------

Noradrenaline is the only transmitter in this codebase so far that acts
through three pharmacologically distinct receptor FAMILIES with three
different G proteins, and the whole point of the noradrenergic system is
that one released transmitter hits all three at once with different
affinities:

  * **alpha1** (alpha1A/1B/1D) is Gq/11-coupled. On a cortical pyramidal
    neuron alpha1 activation closes a resting K+ conductance and
    depolarizes ([McCormick1991]): excitatory.
  * **alpha2** (alpha2A/2B/2C) is Gi/o-coupled. On a locus-coeruleus
    noradrenergic neuron the somatodendritic alpha2A receptor is the
    AUTORECEPTOR: an agonist hyperpolarizes it and firing falls
    ([Cedarbaum1977], [Callado2000]): inhibitory.
  * **beta** (beta1/2/3) is Gs-coupled. On a cortical pyramidal neuron
    beta1 activation reduces the slow calcium-dependent K+ current and the
    spike-frequency adaptation it produces ([Madison1986]), so the
    same firing drive yields more spikes: excitatory.

So this module builds ONE `CircuitNetwork` and puts THREE `ReceptorSystem`s
on it, two of them coupling into the SAME cortical node with the same sign
and one into the source nucleus with the opposite sign. That needs no
engine change: `Simulation(network, receptor_systems=[...])` sums each
system's `coupling` contribution into the shared drive vector, and drive is
additive by construction in the Wilson-Cowan input term (see `engine.py`).
It is the same shape as P5.1's two serotonin systems and P5.2's D1/D2 pair,
with one new feature -- two receptor systems whose gains land on one node
and add rather than oppose.

Sources
-------

The affinities all come from ONE laboratory's programme of whole-cell
radioligand binding on CHO cells expressing a single human adrenoceptor
subtype, four papers wide. That matters more than any individual number:
it is the [Volpe2011]-style same-preparation guarantee that P5.2 had to go
looking for, extended across three receptor families, so that
"noradrenaline prefers receptor X to receptor Y" is a comparison rather
than a quotation of two unrelated experiments.

  [Proudman2021] Proudman RGW, Baker JG. "The selectivity of
                alpha-adrenoceptor agonists for the human alpha1A, alpha1B,
                and alpha1D-adrenoceptors." Pharmacol Res Perspect.
                2021;9(4):e00799. doi:10.1002/prp2.799 (PMC8343220).
                HUMAN alpha1A/1B/1D in CHO cells, [3H]-prazosin WHOLE-CELL
                binding, 37 C. Log KD for (-)-noradrenaline:
                  alpha1A -4.81 +/- 0.10 (n=7)   -> Kd 15488 nM
                  alpha1B -3.79 +/- 0.09 (n=9)   -> Kd 162181 nM
                  alpha1D -5.48 +/- 0.18 (n=8)   -> Kd 3311 nM
                and, in the same paper's functional tables, an alpha1A
                log EC50 of -8.61 +/- 0.09 for intracellular calcium
                mobilization with an intrinsic efficacy ratio (KD/EC50) of
                3.80. The three subtypes are all recorded because the
                49-fold spread in noradrenaline's affinity across them is
                the honest error bar on any single-alpha1 statement; see
                "What the numbers turned out to say".
  [Proudman2020] Proudman RGW, Pupo AS, Baker JG. "The affinity and
                selectivity of alpha-adrenoceptor antagonists,
                antidepressants, and antipsychotics for the human alpha1A,
                alpha1B, and alpha1D-adrenoceptors." Pharmacol Res
                Perspect. 2020;8(4):e00602. doi:10.1002/prp2.602
                (PMC7327383). Same cells, same radioligand, same assay.
                PRAZOSIN log KD: alpha1A -9.07 +/- 0.04 (n=9),
                alpha1B -8.74 +/- 0.06 (n=8), alpha1D -9.07 +/- 0.23
                (n=10), i.e. Kd 0.85, 1.8 and 0.85 nM -- essentially
                non-selective, which is why the prazosin result below does
                not depend on which alpha1 subtype the cortical node is
                taken to carry.
  [Proudman2022] Proudman RGW, Akinaga J, Baker JG. "The signaling and
                selectivity of alpha-adrenoceptor agonists for the human
                alpha2A, alpha2B and alpha2C-adrenoceptors and comparison
                with human alpha1 and beta-adrenoceptors." Pharmacol Res
                Perspect. 2022;10(5):e01003. doi:10.1002/prp2.1003
                (PMC9471048, open access). HUMAN alpha2A/2B/2C in CHO
                cells, [3H]-rauwolscine WHOLE-CELL binding. Log KD:
                  noradrenaline  alpha2A -3.57 +/- 0.03 (n=9) -> 269153 nM
                                 alpha2B -3.52, alpha2C -4.49
                  clonidine      alpha2A -6.72 +/- 0.03 (n=5) -> 190.5 nM
                                 alpha2B -6.34, alpha2C -6.56
                Its alpha2A table also gives, in the same rows, the Gi
                log IC50 (CRE-SPAP in the presence of forskolin) and the
                intrinsic efficacy ratio KD/IC50:
                  noradrenaline  log IC50 -6.60 +/- 0.12, log ratio 3.03
                  clonidine      log IC50 -8.18 +/- 0.04, log ratio 1.46
                Both are used below: the IC50s for the affinity-versus-
                potency finding, the efficacy ratios as the second reading
                of clonidine's partial agonism.
  [Proudman2022b] Proudman RGW, Akinaga J, Baker JG. "The affinity and
                selectivity of alpha-adrenoceptor antagonists,
                antidepressants and antipsychotics for the human alpha2A,
                alpha2B, and alpha2C-adrenoceptors and comparison with
                human alpha1 and beta-adrenoceptors." Pharmacol Res
                Perspect. 2022;10(2):e00936. doi:10.1002/prp2.936
                (PMC8882856). Same cells and assay. YOHIMBINE log KD at
                alpha2A: -8.48 +/- 0.07 (n=5) against [3H]-rauwolscine and
                -8.58 +/- 0.03 (n=5) against [3H]-RX821002 in the same
                paper, i.e. Kd 3.31 and 2.63 nM. That 1.26-fold spread
                between two radioligands in one laboratory is the smallest
                honest error bar available for any number in this module,
                and it is worth stating precisely because the noradrenaline
                numbers below have a 10000-fold one.
  [Baker2010]   Baker JG. "The selectivity of beta-adrenoceptor agonists at
                human beta1-, beta2- and beta3-adrenoceptors." Br J
                Pharmacol. 2010;160(5):1048-1061.
                doi:10.1111/j.1476-5381.2010.00754.x (PMC2936015). HUMAN
                beta1/2/3 in CHO cells, [3H]-CGP 12177 WHOLE-CELL binding.
                Noradrenaline log KD: beta1 -5.74 +/- 0.03 (n=5) ->
                1819.7 nM; beta2 -5.41 +/- 0.07; beta3 -5.53 +/- 0.10. Its
                beta1 functional table gives noradrenaline log EC50
                -7.94 +/- 0.13 in [3H]-cAMP accumulation, a maximal
                response of 102.4 +/- 3.6% of isoprenaline's (n=10) and an
                intrinsic efficacy ratio of 2.20. The 102.4% is where this
                module's noradrenaline efficacy of 1.0 stops being a
                convention and becomes a measurement.
  [Baker2005]   Baker JG. "The selectivity of beta-adrenoceptor antagonists
                at the human beta1, beta2 and beta3 adrenoceptors." Br J
                Pharmacol. 2005;144(3):317-322.
                doi:10.1038/sj.bjp.0706048 (PMC1576008). Same cells, same
                radioligand. PROPRANOLOL log KD: beta1 -8.16 +/- 0.08
                (n=5) -> 6.92 nM; beta2 -9.08 +/- 0.06 (n=8) -> 0.83 nM;
                beta3 -6.93 +/- 0.11 (n=7). NOT the number this module
                uses -- see the propranolol discrepancy below -- but the
                one that keeps the beta site comparable to the alpha sites,
                and the one that shows propranolol is beta2-preferring
                rather than beta1-selective.

Binding kinetics. Three papers, one per site, and they are the reason this
module has rate constants at all rather than affinities alone:

  [Sykes2022]   Sykes DA, Jimenez-Roses M, Reilly J, Fairhurst RA, Charlton
                SJ, Veprintsev DB. "Exploring the kinetic selectivity of
                drugs targeting the beta1-adrenoceptor." Pharmacol Res
                Perspect. 2022;10(4):e00978. doi:10.1002/prp2.978
                (PMC9237807, open access). HUMAN beta1 in CHO membranes,
                [3H]-DHA, competition-association method. Table 1 gives
                koff, kon, residence time, pKd and pKi for 18 ligands. The
                two rows used here, and the four that make the panel
                load-bearing:
                  ligand        koff (min^-1)  kon (M^-1 min^-1)  pKd
                  propranolol   0.91 +/- 0.03  1.28 +/- 0.22 e9   9.13
                  isoprenaline  4.77 +/- 1.74  2.51 +/- 0.44 e7   6.78
                  salbutamol    9.81 +/- 2.60  1.96 +/- 0.77 e7   4.91
                  formoterol    9.97 +/- 2.43  1.68 +/- 0.11 e7   6.25
                  salmeterol    14.3 +/- 3.65  2.63 +/- 0.21 e7   6.29
                  carvedilol    0.09 +/- 0.02  1.97 +/- 0.39 e9  10.36
                See "What derivation 3 assumes, and the first test of it in
                this codebase" for why those six rows matter more than the
                two that are instantiated.
  [Quinton2010] Quinton L, Girard E, Maiga A, et al. "Isolation and
                pharmacological characterization of AdTx1, a natural
                peptide displaying specific insurmountable antagonism of
                the alpha1A-adrenoceptor." Br J Pharmacol.
                2010;159(2):316-325. doi:10.1111/j.1476-5381.2009.00532.x
                (PMC2825353). The paper is about a mamba-venom toxin; the
                number used here is from its control arm: dissociation of
                [3H]-prazosin from RAT BRAIN SYNAPTOSOMES at room
                temperature, initiated with 1 uM cold prazosin,
                koff = 0.043 +/- 0.015 min^-1 (n=4). Rat brain is the
                tissue this whole project is about, which is why this is
                preferred to the same constant measured in a peripheral
                preparation.
  [Tanaka2004]  Tanaka T, Zhang L, Suzuki F, Muramatsu I. "Alpha-1
                adrenoceptors: evaluation of receptor subtype-binding
                kinetics in intact arterial tissues and comparison with
                membrane binding." Br J Pharmacol. 2004;141(3):468-476.
                doi:10.1038/sj.bjp.0705627. Intact RAT TAIL ARTERY segments
                at 4 C: [3H]-prazosin Kon = 0.048 +/- 0.006 nM^-1 min^-1
                (n=4), Koff = 0.078 +/- 0.013 min^-1 (n=3). Reported in
                exactly this codebase's units. NOTE, because it is not
                cosmetic: the paper states a kinetic pKD of 9.3 (Kd 0.50
                nM) from those two constants, but 0.078/0.048 is 1.63 nM
                (pKD 8.79), so the paper's own three numbers are mutually
                inconsistent by 3.2-fold. Its Kon is used here and its
                Koff is not; see "Derivations".
  [Gerhardt1990] Gerhardt MA, Wade SM, Neubig RR. "p-[125I]iodoclonidine is
                a partial agonist at the alpha 2-adrenergic receptor." Mol
                Pharmacol. 1990;38(2):214-221. PMID:1974694. HUMAN PLATELET
                membranes (the classical alpha2A preparation): kon
                8.0 +/- 2.7 x 10^6 M^-1 s^-1 and koff
                2.0 +/- 0.8 x 10^-3 s^-1 at room temperature, Scatchard
                Kd 1.2 +/- 0.1 nM. Same internal-consistency caveat as
                [Tanaka2004] and in the same direction: kon * Kd is
                9.6e-3 s^-1, 4.8-fold faster than the reported koff, so
                only the kon is used. The paper is also this module's
                citation for the fact that halogenated clonidine analogues
                are PARTIAL agonists at alpha2 receptors.

The noradrenaline affinity-state problem needs its own sources, because it
is the largest real uncertainty in this task:

  [UPrichard1978] U'Prichard DC, Snyder SH. "3H-Catecholamine binding to
                alpha-receptors in rat brain: enhancement by reserpine."
                Eur J Pharmacol. 1978;51(2):145-155. PMID:212279. RAT
                CORTEX membranes at 25 C, with the AGONIST itself as the
                radioligand: (-)-3H-noradrenaline binds saturably with
                Kd = 27 nM ((+/-)-3H-adrenaline, 16.7 nM), 10-12 pmol/g
                tissue, and the paper's own conclusion is that the
                3H-catecholamines label the same alpha-noradrenergic
                receptor as 3H-clonidine, i.e. the alpha2 site. This is
                the alpha2 "agonist-state" affinity below.
  [Salama1982]  Salama AI, Lin LL, Repp LD, U'Prichard DC. "Magnesium
                reduces affinities of antagonists at rat cortex alpha
                2-adrenergic receptors labeled with 3H-clonidine: evidence
                for heterogeneity of alpha 2-receptor conformations with
                respect to antagonists." Life Sci. 1982;30(15):1305-1311.
                PMID:6283288. RAT CORTEX membranes: 3H-clonidine labels TWO
                sites, apparent Kd ~1.0 nM ("super-high") and ~5.9 nM
                ("high"); 100 uM GTP reduces the number of super-high
                sites by 45% and reduces clonidine's affinity while
                INCREASING yohimbine's, and 10 mM Mg2+ does the opposite.
                Two things come from this paper: clonidine's agonist-state
                affinity below, and the mechanism -- guanine-nucleotide-
                and cation-dependent receptor states -- that makes the
                10000-fold noradrenaline spread a two-state phenomenon
                rather than a contradiction.
  [Ernsberger1987] Ernsberger P, U'Prichard DC. "Modulation of agonist and
                antagonist interactions at kidney alpha 1-adrenoceptors by
                nucleotides and metal ions." Eur J Pharmacol.
                1987;133(2):165-176. PMID:2880738. Cited for one fact only:
                noradrenaline competition curves at [3H]-prazosin-labelled
                alpha1 sites are BIPHASIC and Na+/GTP selectively remove
                the high-affinity component, so the same two-state problem
                exists at alpha1. No number is taken from it (it is rat
                kidney, and its two Kd values were not obtainable), which
                is why the alpha1 site here has ONE affinity set and the
                alpha2 site has two.

Efficacy:

  [Jasper1998]  Jasper JR, Lesnick JD, Chang LK, Yamanishi SS, Chang TK,
                Hsu SA, Daunt DA, Bonhaus DW, Eglen RM. "Ligand efficacy
                and potency at recombinant alpha2 adrenergic receptors:
                agonist-mediated [35S]GTPgammaS binding." Biochem
                Pharmacol. 1998;55(7):1035-1043.
                doi:10.1016/s0006-2952(97)00631-x. PMID:9605427. Cloned
                mouse alpha2a/d and human alpha2a/2b/2c receptors,
                [35S]GTPgammaS incorporation into membranes, intrinsic
                activity reported RELATIVE TO NORADRENALINE = 1.0 measured
                in the same panel. Human alpha2a: UK-14304 1.02,
                dexmedetomidine 0.73, ST-91 0.63, guanabenz 0.38,
                CLONIDINE 0.32, NE 1.00. The paper also states that
                antagonists did not increase [35S]GTPgammaS binding at all,
                which is where yohimbine's efficacy of 0.0 comes from.
  [Millan2000]  Millan MJ, Newman-Tancredi A, Audinot V, et al. "Agonist
                and antagonist actions of yohimbine as compared to
                fluparoxan at alpha2-adrenergic receptors, serotonin
                5-HT1A, 5-HT1B, 5-HT1D and dopamine D2 and D3 receptors."
                Synapse. 2000;35(2):79-95. PMID:10611634. Independent
                corroboration of yohimbine's zero efficacy: in
                [35S]GTPgammaS protocols yohimbine "exerts antagonist
                actions at h-alpha2A-AR" while behaving as a PARTIAL AGONIST
                at h5-HT1A -- a reminder that yohimbine is not a clean
                alpha2 tool in vivo, which this single-site model does not
                represent.
  [Pohjanoksa1998] Pohjanoksa K, Jansson CC, Luomala K, Marjamaki A,
                Savola JM, Scheinin M. "Subtype-specific stimulation of
                [35S]GTPgammaS binding by recombinant alpha2-adrenoceptors."
                Eur J Pharmacol. 1998;348(2-3):249-259. PMID:9760042. The
                disagreement, recorded rather than hidden: in this
                laboratory's membranes clonidine was a weak partial agonist
                at alpha2B and "appeared inactive at alpha2A and alpha2C".
                See `ALPHA2_EFFICACY_CLONIDINE` for why a low-intrinsic-
                efficacy partial agonist reads as inactive in a low-reserve
                membrane preparation and as potent in a high-reserve
                whole-cell one, and why 0.32 is used with that spread
                stated.
  [Baker2003]   Baker JG, Hall IP, Hill SJ. "Agonist and inverse agonist
                actions of beta-blockers at the human beta2-adrenoceptor
                provide evidence for agonist-directed signalling." Mol
                Pharmacol. 2003;64(6):1357-1369.
                doi:10.1124/mol.64.6.1357. PMID:14645666. Cited for the
                fact that propranolol is an INVERSE agonist rather than a
                neutral antagonist at a constitutively active beta
                receptor, which is why its efficacy of 0.0 here is the
                same flagged modelling simplification P5.1 recorded for
                ketanserin and P5.2 for SCH-23390.

Anatomy and mechanism (no binding numbers taken from these; they fix which
node carries which receptor and with what sign):

  [Cedarbaum1977] Cedarbaum JM, Aghajanian GK. "Catecholamine receptors on
                locus coeruleus neurons: pharmacological characterization."
                Eur J Pharmacol. 1977;44(4):375-385.
                doi:10.1016/0014-2999(77)90312-0. PMID:330174. The result
                the alpha2 gain's SIGN comes from: clonidine depresses the
                firing of rat locus-coeruleus neurons, the effect is
                antagonized by alpha-adrenoceptor antagonists including
                yohimbine and piperoxan, and the receptor's antagonist rank
                order is the alpha2 one.
  [Callado2000] Callado LF, Stamford JA. "Spatiotemporal interaction of
                alpha(2) autoreceptors and noradrenaline transporters in
                the rat locus coeruleus: implications for volume
                transmission." J Neurochem. 2000;74(6):2350-2358.
                doi:10.1046/j.1471-4159.2000.0742350.x. PMID:10820195.
                Cited for the somatodendritic release / autoreceptor /
                transporter arrangement in the LC that the open-loop
                limitation below is about.
  [McCormick1991] McCormick DA, Pape HC, Williamson A. "Actions of
                norepinephrine in the cerebral cortex and thalamus:
                implications for function of the central noradrenergic
                system." Prog Brain Res. 1991;88:293-305.
                doi:10.1016/s0079-6123(08)63817-0. PMID:1726028. The
                alpha1 sign: noradrenaline depolarizes cortical and
                thalamic neurons through alpha1 receptors by reducing a
                resting K+ conductance.
  [Madison1986] Madison DV, Nicoll RA. "Actions of noradrenaline recorded
                intracellularly in rat hippocampal CA1 pyramidal neurones,
                in vitro." J Physiol. 1986;372:221-244.
                doi:10.1113/jphysiol.1986.sp016006. PMID:2873241. The beta
                sign: noradrenaline acting at beta receptors blocks the
                calcium-activated potassium current underlying
                spike-frequency adaptation, so a pyramidal cell fires more
                spikes for the same depolarizing drive. This is the
                cleanest available justification for a POSITIVE beta gain
                on a rate node, because "less adaptation" is a
                gain/threshold change in a rate model rather than a
                synaptic input -- and it is an approximation, because this
                engine has one additive drive term per node and no gain
                term to put it in.
  [Berridge2003] Berridge CW, Waterhouse BD. "The locus
                coeruleus-noradrenergic system: modulation of behavioral
                state and state-dependent cognitive processes." Brain Res
                Brain Res Rev. 2003;42(1):33-84.
                doi:10.1016/s0165-0173(03)00143-7. PMID:12668290. The
                projection-system anatomy this module deliberately does NOT
                wire as an edge; see "What was deliberately not built".

Derivations
-----------

The same three-way split P5.1 established:

1. **Both constants measured** -- use them. Propranolol at beta1
   ([Sykes2022]); prazosin at alpha1, from two different papers
   ([Tanaka2004] kon, [Quinton2010] koff) with the equilibrium affinity of
   a third ([Proudman2020]) as the check.
2. **koff and an equilibrium affinity measured, kon missing** --
   `kon = koff / Kd`. Not needed here.
3. **Only an equilibrium affinity measured** -- share the site's measured
   `kon` and set `koff = kon * Kd`. Noradrenaline at all three sites, and
   clonidine and yohimbine at alpha2.

Prazosin is the strongest constant in this module and the only one in the
codebase so far that three independent measurements agree on:

    kon  = 0.048 nM^-1 min^-1   rat tail artery, 4 C        [Tanaka2004]
    koff = 0.043 min^-1         rat brain synaptosomes, RT  [Quinton2010]
    -> Kd = 0.896 nM
    Kd   = 0.851 nM             human alpha1A CHO, 37 C     [Proudman2020]

5.3% apart, from three preparations, two species and three temperatures.
Nothing was fitted to make that happen, and a test asserts it.
[Tanaka2004]'s own koff (0.078) is not used because that paper's kon, koff
and reported pKD do not agree with each other (above); the value adopted,
0.043, sits between its koff (0.078) and the koff its stated pKD would
imply with its own kon (0.048 * 0.50 = 0.024). Being bracketed by the two
readings of the paper it was NOT taken from is the same kind of check the
P5.2 SCH-23390 derivation got from [Reader1989], and a test asserts the
bracket's arithmetic so the claim cannot rot.

What derivation 3 assumes, and the first test of it in this codebase
--------------------------------------------------------------------

Derivation 3's content is "the other ligands at this site associate at a
rate comparable to the one measured ligand". P5.1 and P5.2 both used it and
neither could check it, because neither site had more than one measured
kon. [Sykes2022]'s beta1 panel has EIGHTEEN, and it says two things:

  * The assumption is FALSE across chemotypes. kon at one site, one
    preparation, one method, spans 8.47e6 (sotalol) to 1.97e9 (carvedilol)
    M^-1 min^-1 -- a factor of 230.
  * It holds well WITHIN a chemotype. Every catecholamine-like agonist in
    the panel lands in a narrow band: isoprenaline 2.51e7, salbutamol
    1.96e7, formoterol 1.68e7, salmeterol 2.63e7 M^-1 min^-1, a factor of
    1.6 across all four.

So derivation 3 is applied here in a refined form -- **share the measured
kon of the ligand chemically closest to the one being derived, and say
which** -- and the beta1 site is the evidence for the refinement rather
than a place where it is assumed. Noradrenaline at beta1 therefore takes
ISOPRENALINE's measured kon (a catecholamine full agonist measured in the
same experiment) and not propranolol's, which is 51-fold faster and would
have asserted that a catecholamine associates like an aryloxypropanolamine.

The two remaining shared kons cross chemotypes and are labelled as the
weaker assumptions they are: noradrenaline at alpha1 borrows prazosin's
(a quinazoline), and noradrenaline and clonidine at alpha2 borrow
p-iodoclonidine's (an imidazoline). What follows from the beta1 panel is
that those `koff` values could be wrong by up to ~100-fold. What does NOT
follow is that any equilibrium result is wrong: derivation 3 sets
`koff = kon * Kd`, so the measured affinity is reproduced EXACTLY whatever
the shared kon is, and the entire uncertainty lands on the timescale over
which occupancy approaches equilibrium. Every result in this module that
depends on a rate rather than on an affinity says so.

The noradrenaline affinity-state problem, which is 10000-fold here
------------------------------------------------------------------

P5.1 met this at 5-HT2A, where serotonin's published affinity varied
250-fold with the radioligand's own pharmacology. At alpha2 it is worse:

    noradrenaline at alpha2A, whole-cell competition against the
    antagonist [3H]-rauwolscine, human receptor:      Kd 269153 nM
      [Proudman2022]
    noradrenaline at rat cortex alpha2 sites, saturation binding of
    (-)-3H-noradrenaline ITSELF:                      Kd     27 nM
      [UPrichard1978]

Ten thousand-fold, and the IUPHAR/BPS Guide to PHARMACOLOGY records the
published range for noradrenaline at human alpha2A as pKi 3.6-7.4, i.e. it
brackets both. Neither number is wrong. An agonist radioligand labels the
G-protein-coupled high-affinity state; a competition curve against an
antagonist radioligand reports a weighted view of the whole population; and
[Salama1982] and [Ernsberger1987] show the balance moving with GTP, Na+ and
Mg2+, which is the mechanism. `ReceptorSite` has one pool and one affinity
per ligand, so this module cannot hold both at once, and the choice changes
more than any other number here.

Two sets are therefore built, `AFFINITY_STATES` selects between them, and
the tests assert the qualitative results under both:

  "agonist-state" (default)  noradrenaline Kd 27 nM [UPrichard1978] and
                             clonidine Kd 5.9 nM [Salama1982], both
                             measured with an agonist radioligand in RAT
                             CORTEX membranes; yohimbine keeps its
                             [Proudman2022b] value.
  "whole-cell"               noradrenaline Kd 269153 nM and clonidine
                             Kd 190.5 nM [Proudman2022], the values that
                             are matched to the alpha1 and beta sites.

The default is the agonist-state one, by a stated rule rather than by
preference: **the preparation should match the receptor being modelled.**
The alpha2 site here is a somatodendritic autoreceptor on a
G-protein-coupled noradrenergic neuron in rodent brain, and rat cortical
membranes with an agonist radioligand are the closest available preparation
to that; a human receptor in a CHO cell is not. This is the same rule P5.1
used to pick serotonin's live-cell affinity at 5-HT2A, applied to a case
where it selects the other kind of assay -- which is the point of having a
rule.

Three consequences, all recorded because they are costs:

  * **The alpha2 site is then NOT matched to the alpha1 and beta sites.**
    The cross-receptor comparison below is therefore computed from the
    whole-cell constants at all three sites, which are all defined in this
    module, and it is asserted algebraically rather than by integration.
  * **The state choice flips the answer to "which adrenoceptor does
    noradrenaline engage first".** Whole-cell: beta1 (1820 nM) before
    alpha1A (15488) before alpha2A (269153). Agonist-state at alpha2:
    alpha2A (27 nM) first, by a factor of 67 over beta1. Both orderings are
    published measurements, and this module can produce either. That is
    the honest headline of this task, and it is tested as two explicit
    orderings rather than smoothed into one.
  * **The whole-cell noradrenaline set cannot be integrated at all.** With
    the alpha2 site's shared kon, `koff = 0.48 * 269153 = 129194 min^-1`,
    a relaxation time of 0.3 ms. `Simulation.run` uses explicit RK45,
    whose stability limit then forces steps of ~2e-5 min for the whole
    shared state vector: a 10-minute window that takes 3 s under the
    default set did not finish in 240 s under the whole-cell set. So
    whole-cell alpha2 results are asserted at the algebraic fixed point
    (`ReceptorSite.occupancy_rhs` residual = 0), which is exact and cheap,
    and the reason is this measurement rather than a preference. A stiff
    or quasi-steady-state occupancy solver is Phase 9's business; the
    number above is what it would buy.

What the numbers turned out to say
----------------------------------

**Noradrenaline's highest affinity in the matched panel is for beta1, not
for either alpha.** 1820 nM at beta1 against 15488 nM at alpha1A and
269153 nM at alpha2A, all from whole-cell binding on human receptors in CHO
cells in one laboratory. The textbook ordering of noradrenaline as a potent
alpha agonist and a weak beta agonist is a statement about POTENCY, and the
same papers' functional data give exactly that ordering:

    site     Kd (nM)   functional EC50/IC50 (nM)   intrinsic efficacy ratio
    alpha1A    15488    2.45  (Ca2+ mobilization)          3.80
    beta1       1820   11.5   (cAMP accumulation)          2.20
    alpha2A   269153  251     (Gi, CRE-SPAP)               3.03

Affinity says beta1 first; potency says alpha1A first. The gap is receptor
reserve and signal amplification, and it is 3.8 log units at alpha1A. This
model's `activation = sum(occupancy * efficacy)` is linear in occupancy
with no amplification term, so it reproduces the AFFINITY ordering and not
the POTENCY ordering, and both are asserted in the tests with the numbers
above. It is the same limitation P5.2 flagged for the 530-fold-versus-
31-fold dopamine D1/D2 gap, one receptor family wider and an order of
magnitude larger.

**Prazosin is essentially non-selective across the alpha1 subtypes and
noradrenaline is not.** Prazosin Kd 0.85 / 1.8 / 0.85 nM at alpha1A/1B/1D
([Proudman2020]) against noradrenaline's 15488 / 162181 / 3311
([Proudman2021]), a 49-fold spread. So "prazosin blocks the alpha1
component" is subtype-independent here, while "noradrenaline engages beta1
before alpha1" has a subtype-dependent MARGIN (8.5-fold at alpha1A, 89-fold
at alpha1B, 1.8-fold at alpha1D) but survives at every subtype, because
beta1's 1820 nM is tighter than all three. The claim is therefore made in
the form that survives, and the margin is stated as a range.

**Propranolol's two published affinities at beta1 differ 9.7-fold**, and
this module uses the kinetic one. [Sykes2022]'s rate constants give
koff/kon = 0.911 nM, which agrees with that paper's own equilibrium pKd of
9.13 (0.74 nM) to 4%; [Baker2005]'s whole-cell competition gives 6.92 nM.
Derivation 1 wins over the matched-panel value here because a measured
kon/koff pair is worth more to this codebase than set consistency, and
because nothing in this module's qualitative results turns on the
difference: propranolol is 2000-fold tighter than noradrenaline at beta1
under the kinetic number and 263-fold tighter under the whole-cell one, and
`ligands_beta(propranolol_affinity="whole-cell")` builds the second set so
that "nothing turns on it" is a tested claim rather than an assurance. The
two candidate preparations differ in temperature (37 C vs whatever the
membrane assay ran at), radioligand ([3H]-CGP 12177 vs [3H]-DHA) and
membrane-versus-whole-cell format, and no attempt is made here to say
which is right.

**Clonidine is a weak partial agonist, and the published spread on how
weak is the widest efficacy uncertainty in this codebase.** [Jasper1998]
measures intrinsic activity 0.32 relative to noradrenaline in the same
[35S]GTPgammaS panel, which is the number used. [Proudman2022]'s intrinsic
efficacy ratios put it at 10^(1.46-3.03) = 2.7% of noradrenaline's, and
[Pohjanoksa1998] could not detect any alpha2A response to it at all. Those
three are not really in conflict: they are one low-intrinsic-efficacy
ligand read out in three systems with different receptor reserve. See
`ALPHA2_EFFICACY_CLONIDINE` for which quantity `Ligand.efficacy` actually
is and why the Emax ratio was chosen for consistency with P5.1 and P5.2.

Searched and empty, or found and deliberately not used
------------------------------------------------------

* **Rodent pharmacokinetics for prazosin, clonidine, yohimbine and
  propranolol.** Unlike P5.1's and P5.2's tool compounds, these four are
  clinical drugs and rodent PK for them DOES exist -- e.g. a single
  intravenous dose of rac-propranolol in the rat gives a biphasic curve
  with a terminal half-life of ~2.3 h (Toxicokinetics of a single
  intravenous dose of rac-propranolol versus optically pure propranolol in
  the rat, PMID:8593256). It is recorded here and NOT used, for two
  reasons. First, what `drug_pk.py` and Phase 1 consume is a full
  parameter set per route (F, ka, Vd, CL and a brain partition), not a
  terminal half-life, and half-sourcing one of the four would put a real
  citation next to three invented numbers and thereby assert a clearance
  ORDERING that nothing in reach supports. Second, nothing in this
  receptor-layer task depends on the elimination rate beyond "all four are
  still on board across the scenario's window". So all four share ONE
  illustrative `TOOL_COMPOUND_KE`, deliberately the same number as
  `serotonin.TOOL_COMPOUND_KE` and `dopamine.TOOL_COMPOUND_KE`, and a test
  asserts the equality. Do not cite it as a half-life for any of them.
* **Binding kinetics for noradrenaline itself, at any adrenoceptor.**
  Searched Europe PMC for the catecholamines crossed with "association rate
  constant", "dissociation rate constant", "residence time", NanoBRET and
  TR-FRET, and through the kinetic panels that exist for these receptors
  ([Sykes2022] at beta1 and its beta2 predecessors). Every kinetic panel
  found measures antagonists and synthetic agonists; the endogenous
  transmitter is in none of them. Hence derivation 3 at all three sites for
  noradrenaline, with the chemotype refinement above.
* **Binding kinetics for clonidine and yohimbine.** [Gerhardt1990] measures
  a clonidine ANALOGUE (p-iodoclonidine) and is used for its kon only;
  nothing was found for clonidine or yohimbine themselves. Searched
  additionally for "[3H]rauwolscine" and "[3H]RX821002" crossed with
  association/dissociation kinetics: [Tanaka2004]-style tissue-segment
  work and the radioligand characterizations report equilibrium constants
  and, where kinetics appear at all, report them for the radioligand
  rather than for the competitor.
* **Noradrenaline's own clearance rate.** The same gap dopamine and
  serotonin have, for the same reason: extracellular noradrenaline is
  cleared by NET-mediated reuptake, a saturable Michaelis-Menten
  transporter process on a sub-second timescale, not first-order
  elimination. `NORADRENALINE_KE` is an explicit flagged proxy set to the
  same value as `dopamine.DOPAMINE_KE` and `serotonin.SEROTONIN_KE`; the
  real quantity belongs to P5.8 (NET) and P6.2 (endogenous tone).
* **A behavioural dose-response usable as a circuit-layer target.** Not
  searched again. Track E and P2.2 established the blocker for the whole
  class -- behavioural pharmacology reports mg/kg by IP/SC/PO and this
  module has no such kinetics -- and Phase 1's bridge is where such a
  dataset enters.

What was deliberately not built
-------------------------------

* **The LC -> cortex edge.** The locus coeruleus projects to essentially
  the whole forebrain ([Berridge2003]) and this network has no
  edge from `LC_NA` to `PFC_Pyramidal`. That is P5.1's decision for P5.1's
  reason, and it is if anything stronger here: in a Wilson-Cowan network a
  weight is a fixed-sign synaptic drive, and the LC's influence on a
  cortical pyramidal cell is receptor-mediated and of BOTH SIGNS AT ONCE
  through three receptor families -- alpha1 and beta excitatory, alpha2
  inhibitory, on cells that carry all three. Representing that as one
  signed weight would double-count the receptor layer, which is the layer
  this codebase exists for. Closing the loop properly is P6.1: source
  firing rate -> release -> region-wise transmitter concentration ->
  occupancy.
* **Postsynaptic alpha2A in the cortex.** It is really there and it is
  functionally important: [Wang2007] (Wang M, Ramos BP, Paspalas CD, et al.
  "Alpha2A-adrenoceptors strengthen working memory networks by inhibiting
  cAMP-HCN channel signaling in prefrontal cortex." Cell.
  2007;129(2):397-410, PMID:17448997) shows alpha2A stimulation on
  pyramidal-cell dendritic spines closing HCN channels and STRENGTHENING
  network connections. That is the same state-dependent-sign problem P5.2
  met with D1: the receptor is Gi-coupled, so the cAMP sign is not in
  doubt, but the effect on a rate node is an increase in the efficacy of
  incoming input rather than a decrease in firing, and this model has one
  scalar drive per node. Adding it would mean choosing that sign by hand
  with no measurement that settles it for a rate model, so `alpha2` here is
  the autoreceptor only and this paragraph is the record of what is
  missing.
* **The alpha1/beta subtype split.** `Adra1a`, `Adra1b`, `Adra1d`,
  `Adra2a`, `Adra2b`, `Adra2c`, `Adrb1` and `Adrb2` are all already in
  `expression/genes.py` tagged P5.3, and every affinity above is reported
  per subtype, so a subtype-resolved version is a table lookup away. It is
  not built because the three sites here are pharmacological pools, not
  regions: the question a subtype split would answer ("which subtype does
  the cortical pyramidal node carry?") is a cell-type-resolved question
  that a regional ISH map cannot answer -- the same limit P5.2 recorded for
  D1-versus-D2 in one striatum. What IS done instead is to state every
  claim in the form that survives the subtype uncertainty, and to test that
  it does.
* **An atlas-derived density factor.** Same position as P5.1 and P5.2. The
  three mechanism gains are explicitly the `mechanism_gain` half of P4.2's
  `coupling = mechanism_gain * rho[region]`, so the upgrade is additive,
  and it is not taken because the committed expression fixture does not
  carry these genes and a density-derived map would make every test in this
  module depend on the gitignored cache -- the failure mode P4.3's fixture
  discussion exists to avoid.
* **Endogenous noradrenergic tone.** Without it an antagonist given alone
  is inert, as everywhere else in this codebase. Real prazosin, real
  yohimbine and real propranolol all have effects given alone, because a
  real LC fires tonically and a real cortex has noradrenaline in it. P6.2.
  The tests that assert inertness say so, so the day P6.2 lands the
  expectation is already written down.
* **The imidazoline story.** Clonidine binds non-adrenergic
  imidazoline I1 sites with an affinity comparable to its alpha2 one, which
  is why the antihypertensive mechanism of clonidine-like drugs was argued
  about for two decades. A single-site alpha2 model cannot represent that,
  and it is stated here so that a clonidine result in this codebase is not
  read as a claim about the drug's mechanism in vivo.
"""

import numpy as np

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite


def _kd_from_log(log_kd):
    """nM from a Baker-laboratory `log KD` (which is log10 of molar Kd).

    Every affinity in this module arrives as a log KD in molar units, so
    the conversion is written once here rather than open-coded eleven
    times: `Kd(nM) = 10^(9 + log_KD)`.
    """
    return 10.0 ** (9.0 + log_kd)


# ---------------------------------------------------------------------------
# alpha1: measured constants, and the ligands built from them
# ---------------------------------------------------------------------------

#: MEASURED log KD values of (-)-noradrenaline at the three human alpha1
#: subtypes, [3H]-prazosin whole-cell binding in CHO cells [Proudman2021].
#: All three are kept because the 49-fold spread across them is the honest
#: error bar on any single-alpha1 claim; only alpha1A is instantiated.
ALPHA1A_LOG_KD_NORADRENALINE = -4.81
ALPHA1B_LOG_KD_NORADRENALINE = -3.79
ALPHA1D_LOG_KD_NORADRENALINE = -5.48

#: MEASURED log KD values of prazosin at the same three subtypes, same
#: laboratory, same radioligand, same assay [Proudman2020]. Essentially
#: non-selective (0.85 / 1.8 / 0.85 nM), which is what makes the prazosin
#: results here subtype-independent.
ALPHA1A_LOG_KD_PRAZOSIN = -9.07
ALPHA1B_LOG_KD_PRAZOSIN = -8.74
ALPHA1D_LOG_KD_PRAZOSIN = -9.07

#: nM. The two alpha1A affinities this module uses, in nM.
ALPHA1_KD_NORADRENALINE = _kd_from_log(ALPHA1A_LOG_KD_NORADRENALINE)  # 15488.2
ALPHA1_KD_PRAZOSIN_BINDING = _kd_from_log(ALPHA1A_LOG_KD_PRAZOSIN)  # 0.8511

#: nM^-1 min^-1. MEASURED association rate constant of [3H]-prazosin at
#: alpha1 adrenoceptors in intact rat tail artery segments at 4 C
#: [Tanaka2004], reported in exactly these units. Used for prazosin itself
#: and shared with noradrenaline by derivation 3.
ALPHA1_KON_PRAZOSIN = 0.048

#: min^-1. MEASURED dissociation rate constant of [3H]-prazosin from RAT
#: BRAIN synaptosomes at room temperature [Quinton2010]. Preferred over
#: [Tanaka2004]'s own 0.078 min^-1 because it is brain tissue and because
#: [Tanaka2004]'s kon, koff and reported pKD are mutually inconsistent by
#: 3.2-fold; 0.043 sits between that paper's two readings (0.024 implied by
#: its pKD with its kon, and 0.078 measured), which is the bracket the
#: tests assert.
ALPHA1_KOFF_PRAZOSIN = 0.043

#: min^-1. DERIVED (derivation 3): `ALPHA1_KON_PRAZOSIN *
#: ALPHA1_KD_NORADRENALINE` = 0.048 * 15488.166. A dissociation half-life
#: of 0.056 s. The shared kon crosses chemotypes (quinazoline ->
#: catecholamine), so this RATE could be off by up to ~100-fold on the
#: [Sykes2022] panel's evidence; the AFFINITY it encodes is exact by
#: construction. See "What derivation 3 assumes" in the module docstring.
ALPHA1_KOFF_NORADRENALINE = ALPHA1_KON_PRAZOSIN * ALPHA1_KD_NORADRENALINE

#: Prazosin's efficacy. 0.0 is the same MODELLING SIMPLIFICATION P5.1
#: flagged for ketanserin and P5.2 for SCH-23390: competitive
#: alpha1-antagonists behave as inverse agonists at constitutively active
#: receptors, and `ReceptorSite.activation` has no constitutive term for a
#: negative efficacy to act against, so a negative value would drive the
#: node BELOW its drug-free baseline, which is not what an inverse agonist
#: does to a receptor with no constitutive tone.
ALPHA1_EFFICACY_PRAZOSIN = 0.0


def ligands_alpha1():
    """noradrenaline / prazosin at alpha1A, in that order."""
    noradrenaline = Ligand(
        name="noradrenaline",
        # Derivation 3: the site's measured kon, shared.
        kon=ALPHA1_KON_PRAZOSIN,
        koff=ALPHA1_KOFF_NORADRENALINE,
        # The reference agonist of this module's efficacy scale. At beta1
        # this is a measurement (102.4% of isoprenaline, [Baker2010]); at
        # alpha1A and alpha2A noradrenaline is the reference the panels
        # normalize to, so 1.0 is a definition there.
        efficacy=1.0,
    )
    prazosin = Ligand(
        name="prazosin",
        kon=ALPHA1_KON_PRAZOSIN,  # derivation 1, measured [Tanaka2004]
        koff=ALPHA1_KOFF_PRAZOSIN,  # derivation 1, measured [Quinton2010]
        efficacy=ALPHA1_EFFICACY_PRAZOSIN,  # simplification; see the constant
    )
    return [noradrenaline, prazosin]


def receptor_site_alpha1():
    """One competitive alpha1 pool, on the cortical pyramidal node.

    ONE pool and ONE subtype (alpha1A). alpha1 receptors sit on many other
    cells, including cortical interneurons and the cerebral vasculature;
    this site is the postsynaptic pyramidal one, which is the only node in
    this network that could carry it.
    """
    return ReceptorSite(name="alpha1_on_cortical_pyramidal", ligands=ligands_alpha1())


# ---------------------------------------------------------------------------
# alpha2: two affinity states, and the autoreceptor
# ---------------------------------------------------------------------------

#: MEASURED log KD values at human alpha2A, [3H]-rauwolscine whole-cell
#: binding in CHO cells [Proudman2022] (agonists) and [Proudman2022b]
#: (yohimbine). The alpha2B/2C values are recorded in the module docstring.
ALPHA2A_LOG_KD_NORADRENALINE = -3.57
ALPHA2A_LOG_KD_CLONIDINE = -6.72
ALPHA2A_LOG_KD_YOHIMBINE = -8.48

#: MEASURED log KD of yohimbine at alpha2A in the SAME paper against the
#: other radioligand, [3H]-RX821002: -8.58. The 1.26-fold spread between
#: the two is this module's smallest error bar and is kept as such.
ALPHA2A_LOG_KD_YOHIMBINE_RX821002 = -8.58

#: nM. The whole-cell affinity set, in nM.
ALPHA2_KD_NORADRENALINE_WHOLE_CELL = _kd_from_log(
    ALPHA2A_LOG_KD_NORADRENALINE
)  # 269153.5
ALPHA2_KD_CLONIDINE_WHOLE_CELL = _kd_from_log(ALPHA2A_LOG_KD_CLONIDINE)  # 190.55
ALPHA2_KD_YOHIMBINE = _kd_from_log(ALPHA2A_LOG_KD_YOHIMBINE)  # 3.3113

#: nM. MEASURED saturation Kd of (-)-3H-noradrenaline -- the AGONIST itself
#: as the radioligand -- in rat cortex membranes at 25 C [UPrichard1978].
#: The same paper reports 16.7 nM for (+/-)-3H-adrenaline and concludes
#: that the 3H-catecholamines label the same site as 3H-clonidine, i.e. the
#: alpha2 site. This is the G-protein-coupled high-affinity state, and it
#: is 10000-fold tighter than the whole-cell value above.
ALPHA2_KD_NORADRENALINE_AGONIST_STATE = 27.0

#: nM. MEASURED apparent Kd values of the two 3H-clonidine binding sites in
#: rat cortex membranes [Salama1982]: ~1.0 nM ("super-high") and ~5.9 nM
#: ("high"). The "high" site is used for the agonist-state set, because it
#: is the larger population and because [Salama1982] shows the super-high
#: component is the GTP-sensitive one -- i.e. the one that a receptor in a
#: living neuron with GTP present has less of. The super-high value is kept
#: so the choice is visible.
ALPHA2_KD_CLONIDINE_AGONIST_STATE = 5.9
ALPHA2_KD_CLONIDINE_SUPERHIGH_STATE = 1.0

#: nM^-1 min^-1. MEASURED association rate constant of
#: p-[125I]iodoclonidine at alpha2 receptors in human platelet membranes
#: (the classical alpha2A preparation) at room temperature [Gerhardt1990]:
#: 8.0e6 M^-1 s^-1 * 60 * 1e-9. Shared by all three alpha2 ligands
#: (derivation 3). Only the kon is taken from that paper: its own kon * Kd
#: is 4.8-fold faster than its reported koff, so the pair is not
#: self-consistent and is not used as one.
ALPHA2_KON = 0.48

#: Efficacy relative to noradrenaline = 1.0. MEASURED intrinsic activity of
#: clonidine at cloned HUMAN alpha2a in [35S]GTPgammaS binding, in a panel
#: where noradrenaline was run as the reference in the same experiment
#: [Jasper1998].
#:
#: This is an Emax RATIO, chosen for consistency with P5.1's and P5.2's
#: fractional efficacies (8-OH-DPAT 0.85, DOI 0.813, dihydrexidine 0.808),
#: and it is the lowest one in the codebase so far. What `Ligand.efficacy`
#: multiplies is OCCUPANCY, so the quantity it formally wants is relative
#: INTRINSIC efficacy, not Emax; those two coincide only in a system with
#: no receptor reserve. [Proudman2022]'s intrinsic efficacy ratios give
#: clonidine 10^(1.46 - 3.03) = 2.7% of noradrenaline's, and
#: [Pohjanoksa1998] could not detect an alpha2A response to clonidine at
#: all. So the published spread on this one number is 0.32 / 0.027 / ~0,
#: which is one low-efficacy ligand read out in three systems of different
#: receptor reserve rather than three disagreeing measurements. 0.32 is
#: used, the spread is stated, and every result that depends on clonidine
#: being a WEAKER agonist than noradrenaline holds across all three
#: readings.
ALPHA2_EFFICACY_CLONIDINE = 0.32

#: Yohimbine's efficacy. MEASURED, not a simplification: [Jasper1998]
#: reports that antagonists did not increase [35S]GTPgammaS binding at any
#: alpha2 subtype, and [Millan2000] finds yohimbine exerts antagonist
#: actions at h-alpha2A in the same kind of protocol. This is the same
#: standing as WAY-100635's zero in P5.1 and unlike prazosin's and
#: propranolol's here.
ALPHA2_EFFICACY_YOHIMBINE = 0.0

#: The affinity sets `ligands_alpha2` can build. See "The noradrenaline
#: affinity-state problem" in the module docstring for why there are two,
#: why the default is the agonist-state one, and what each costs.
AFFINITY_STATES = ("agonist-state", "whole-cell")


def ligands_alpha2(affinity_state="agonist-state"):
    """noradrenaline / clonidine / yohimbine at alpha2A, in that order.

    `affinity_state` selects the two agonists' affinities:

      "agonist-state" (default)  noradrenaline Kd 27 nM [UPrichard1978] and
                                 clonidine Kd 5.9 nM [Salama1982], both
                                 measured against an agonist radioligand in
                                 rat cortex membranes -- the preparation
                                 closest to the somatodendritic
                                 autoreceptor this site stands for.
      "whole-cell"               noradrenaline Kd 269153 nM and clonidine
                                 Kd 190.5 nM [Proudman2022], matched to
                                 this module's alpha1 and beta sites. Any
                                 run that DOSES noradrenaline through this
                                 set is numerically prohibitive (koff
                                 129194 min^-1); use the algebraic fixed
                                 point instead, as the tests do.

    Yohimbine is identical in both sets, and that is itself an
    approximation with a known direction: [Salama1982] shows GTP INCREASES
    yohimbine's affinity at rat cortex alpha2 sites while decreasing
    clonidine's, so an antagonist's affinity is not state-free either. The
    effect there is a factor of a few, against 10000-fold for
    noradrenaline, which is why one number is carried and flagged rather
    than two.
    """
    if affinity_state not in AFFINITY_STATES:
        raise ValueError(
            "affinity_state must be one of %s, got %r"
            % (list(AFFINITY_STATES), affinity_state)
        )

    if affinity_state == "agonist-state":
        kd_noradrenaline = ALPHA2_KD_NORADRENALINE_AGONIST_STATE
        kd_clonidine = ALPHA2_KD_CLONIDINE_AGONIST_STATE
    else:
        kd_noradrenaline = ALPHA2_KD_NORADRENALINE_WHOLE_CELL
        kd_clonidine = ALPHA2_KD_CLONIDINE_WHOLE_CELL

    noradrenaline = Ligand(
        name="noradrenaline",
        # Derivation 3, with the site's shared measured kon.
        kon=ALPHA2_KON,
        koff=ALPHA2_KON * kd_noradrenaline,
        efficacy=1.0,
    )
    clonidine = Ligand(
        name="clonidine",
        kon=ALPHA2_KON,  # derivation 3
        koff=ALPHA2_KON * kd_clonidine,
        efficacy=ALPHA2_EFFICACY_CLONIDINE,
    )
    yohimbine = Ligand(
        name="yohimbine",
        kon=ALPHA2_KON,  # derivation 3
        koff=ALPHA2_KON * ALPHA2_KD_YOHIMBINE,
        efficacy=ALPHA2_EFFICACY_YOHIMBINE,  # measured zero; see the constant
    )
    return [noradrenaline, clonidine, yohimbine]


def receptor_site_alpha2(affinity_state="agonist-state"):
    """One competitive alpha2 pool: the LC somatodendritic AUTORECEPTOR.

    ONE pool, on the `LC_NA` node, and deliberately not also on the
    cortical node. alpha2A really is postsynaptic in the cortex as well
    ([Wang2007]), and the reason it is not modelled there is a sign
    ambiguity a rate node cannot carry; see "What was deliberately not
    built" in the module docstring.
    """
    return ReceptorSite(
        name="alpha2_autoreceptor_on_LC",
        ligands=ligands_alpha2(affinity_state=affinity_state),
    )


# ---------------------------------------------------------------------------
# beta: the site with measured kinetics, and the one panel that checks
# derivation 3
# ---------------------------------------------------------------------------

#: MEASURED log KD of noradrenaline at the three human beta subtypes,
#: [3H]-CGP 12177 whole-cell binding in CHO cells [Baker2010].
BETA1_LOG_KD_NORADRENALINE = -5.74
BETA2_LOG_KD_NORADRENALINE = -5.41
BETA3_LOG_KD_NORADRENALINE = -5.53

#: nM. The beta1 affinity this module uses.
BETA_KD_NORADRENALINE = _kd_from_log(BETA1_LOG_KD_NORADRENALINE)  # 1819.7

#: MEASURED log KD of propranolol at the three human beta subtypes, same
#: laboratory, same radioligand, same assay [Baker2005]. Note beta2
#: (-9.08) is TIGHTER than beta1 (-8.16): propranolol is beta2-preferring,
#: not beta1-selective, which is the source of its bronchospasm liability.
BETA1_LOG_KD_PROPRANOLOL = -8.16
BETA2_LOG_KD_PROPRANOLOL = -9.08
BETA3_LOG_KD_PROPRANOLOL = -6.93

#: nM. Propranolol at beta1 from whole-cell competition binding
#: [Baker2005]: 6.92 nM. NOT the default -- see the next two constants and
#: the discrepancy paragraph in the module docstring -- but selectable, so
#: that "no result turns on which of the two is used" is tested.
BETA_KD_PROPRANOLOL_WHOLE_CELL = _kd_from_log(BETA1_LOG_KD_PROPRANOLOL)

#: nM^-1 min^-1 and min^-1. MEASURED rate constants of propranolol at human
#: beta1 by competition association against [3H]-DHA [Sykes2022]:
#: 1.28e9 M^-1 min^-1 * 1e-9, and 0.91 min^-1. Their ratio is 0.711 nM
#: against that paper's own equilibrium pKd of 9.13 (0.741 nM), a 4%
#: agreement, which is the units check in the tests. Residence time 1.1
#: min: propranolol is a FAST-off antagonist, unlike P5.1's WAY-100635 and
#: P5.2's SCH-23390.
BETA_KON_PROPRANOLOL = 1.28
BETA_KOFF_PROPRANOLOL = 0.91

#: nM^-1 min^-1. MEASURED association rate constant of ISOPRENALINE at
#: human beta1 in the same table and the same experiment [Sykes2022]:
#: 2.51e7 M^-1 min^-1 * 1e-9. This, not propranolol's, is what
#: noradrenaline's derivation 3 shares, because isoprenaline is the
#: catecholamine full agonist of the panel and propranolol's kon is
#: 51-fold faster. The panel's four catecholamine-like agonists span only
#: 1.6-fold (1.68e7 to 2.63e7) while the panel as a whole spans 230-fold,
#: which is the evidence for the chemotype refinement of derivation 3.
BETA_KON_ISOPRENALINE = 0.0251

#: MEASURED kon values of the other three catecholamine-like agonists in
#: the [Sykes2022] beta1 panel, and of the two extremes of the whole panel,
#: in nM^-1 min^-1. Kept as constants rather than prose because a test
#: asserts the 1.6-fold-within-chemotype / 230-fold-across-panel claim that
#: derivation 3's refinement rests on.
BETA_KON_SALBUTAMOL = 0.0196
BETA_KON_FORMOTEROL = 0.0168
BETA_KON_SALMETEROL = 0.0263
BETA_KON_SOTALOL = 0.00847
BETA_KON_CARVEDILOL = 1.97

#: min^-1. DERIVED (derivation 3): `BETA_KON_ISOPRENALINE *
#: BETA_KD_NORADRENALINE` = 0.0251 * 1819.70.
BETA_KOFF_NORADRENALINE = BETA_KON_ISOPRENALINE * BETA_KD_NORADRENALINE

#: MEASURED. Noradrenaline's maximal response at human beta1 in
#: [3H]-cAMP accumulation is 102.4 +/- 3.6% of isoprenaline's (n=10)
#: [Baker2010], so noradrenaline is a FULL agonist at beta1 and this
#: module's efficacy of 1.0 for it is measured here rather than assumed.
BETA_EMAX_NORADRENALINE_VS_ISOPRENALINE = 1.024

#: Propranolol's efficacy. 0.0 is the third instance of the same MODELLING
#: SIMPLIFICATION (after ketanserin in P5.1 and SCH-23390 in P5.2):
#: [Baker2003] shows propranolol is an inverse agonist at a constitutively
#: active beta receptor, and this engine has no constitutive term for a
#: negative efficacy to act against.
BETA_EFFICACY_PROPRANOLOL = 0.0

#: The propranolol affinity sets `ligands_beta` can build.
PROPRANOLOL_AFFINITIES = ("kinetic", "whole-cell")


def ligands_beta(propranolol_affinity="kinetic"):
    """noradrenaline / propranolol at beta1, in that order.

    `propranolol_affinity` selects between the two published beta1
    affinities, which differ 9.7-fold:

      "kinetic" (default)  kon 1.28 nM^-1 min^-1 and koff 0.91 min^-1, both
                           measured [Sykes2022], Kd 0.711 nM. Derivation 1.
      "whole-cell"         Kd 6.92 nM [Baker2005], matched to this module's
                           alpha sites, with [Sykes2022]'s kon retained so
                           that only the affinity changes.

    Noradrenaline is identical in both.
    """
    if propranolol_affinity not in PROPRANOLOL_AFFINITIES:
        raise ValueError(
            "propranolol_affinity must be one of %s, got %r"
            % (list(PROPRANOLOL_AFFINITIES), propranolol_affinity)
        )

    noradrenaline = Ligand(
        name="noradrenaline",
        # Derivation 3, with the site's measured CATECHOLAMINE kon.
        kon=BETA_KON_ISOPRENALINE,
        koff=BETA_KOFF_NORADRENALINE,
        efficacy=1.0,  # measured full agonist here [Baker2010]
    )
    if propranolol_affinity == "kinetic":
        propranolol = Ligand(
            name="propranolol",
            kon=BETA_KON_PROPRANOLOL,  # derivation 1, measured [Sykes2022]
            koff=BETA_KOFF_PROPRANOLOL,  # derivation 1, measured [Sykes2022]
            efficacy=BETA_EFFICACY_PROPRANOLOL,
        )
    else:
        propranolol = Ligand(
            name="propranolol",
            kon=BETA_KON_PROPRANOLOL,
            # Derivation 3 off the whole-cell affinity, with the measured
            # kon kept: 1.28 * 6.9183 [Baker2005].
            koff=BETA_KON_PROPRANOLOL * BETA_KD_PROPRANOLOL_WHOLE_CELL,
            efficacy=BETA_EFFICACY_PROPRANOLOL,
        )
    return [noradrenaline, propranolol]


def receptor_site_beta(propranolol_affinity="kinetic"):
    """One competitive beta1 pool, on the cortical pyramidal node."""
    return ReceptorSite(
        name="beta1_on_cortical_pyramidal",
        ligands=ligands_beta(propranolol_affinity=propranolol_affinity),
    )


# ---------------------------------------------------------------------------
# Pharmacokinetics: one illustrative number for four drugs, one proxy for
# the transmitter
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, shared by prazosin, clonidine,
#: yohimbine and propranolol. Deliberately the same number as
#: `serotonin.TOOL_COMPOUND_KE` and `dopamine.TOOL_COMPOUND_KE`, and a test
#: asserts the equality so they cannot drift apart. Unlike those two
#: modules' compounds these four are clinical drugs with real published
#: rodent PK; see "Searched and empty, or found and deliberately not used"
#: in the module docstring for why it is not half-adopted here. Do not cite
#: this as a measured half-life for any of them.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY, the same one and for the same reason as
#: `dopamine.DOPAMINE_KE` and `serotonin.SEROTONIN_KE`: noradrenaline here
#: is a transmitter, not a dosed drug, and is cleared from the
#: extracellular space by NET-mediated reuptake -- a saturable
#: Michaelis-Menten transporter process on a sub-second timescale, not
#: first-order elimination. Taken literally that would collapse a
#: noradrenaline "dose" into a delta function inside one integration step.
#: 5.0 min^-1 (t1/2 ~8.3 s) preserves the only property the dynamics need,
#: that the transmitter clears far faster than any dosed drug, at a
#: timescale the integrator resolves. It is why the cortical response to a
#: noradrenaline dose below is much smaller than the same occupancy held
#: constant would produce, and a test states that explicitly. The real
#: quantity belongs to P5.8 (NET) and P6.2 (endogenous tone).
NORADRENALINE_KE = 5.0


def pk_models_alpha1():
    """One PK model per alpha1 ligand, in `ligands_alpha1()` order."""
    return [
        OneCompartmentPK(ke=NORADRENALINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


def pk_models_alpha2():
    """One PK model per alpha2 ligand, in `ligands_alpha2()` order."""
    return [
        OneCompartmentPK(ke=NORADRENALINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


def pk_models_beta():
    """One PK model per beta ligand, in `ligands_beta()` order."""
    return [
        OneCompartmentPK(ke=NORADRENALINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: ILLUSTRATIVE throughout, built to three stated rules
# ---------------------------------------------------------------------------
#
# Three nodes, and one edge that is missing on purpose:
#
#   LC_NA            locus coeruleus noradrenergic neurons. Carries the
#                    somatodendritic alpha2A AUTORECEPTOR ([Cedarbaum1977],
#                    [Callado2000]). NO edges: see the
#                    module docstring. Same shape as P5.1's `DRN_5HT` and
#                    for the same reason.
#   PFC_Pyramidal    layer V prefrontal pyramidal neurons. Carries BOTH
#                    alpha1 (Gq, excitatory, [McCormick1991]) and beta1
#                    (Gs, excitatory, [Madison1986]) -- two receptor
#                    systems whose gains land on ONE node and ADD, which is
#                    what is new about this network relative to P5.1's.
#   PFC_GABA         local cortical GABAergic interneurons. Carries no
#                    receptor here, and exists to give the pyramidal node
#                    the recurrent inhibition that makes its response
#                    finite.
#
# Every weight, tau, gain and baseline_input below is ILLUSTRATIVE, and
# three stated rules fix them so that "illustrative" does not mean "tuned
# until the tests passed":
#
#   1. **Baselines put every node at rate 0.5 at rest with no drug**, the
#      maximum-sensitivity point `stability.MAX_SENSITIVITY_RATE` derives.
#      With the two loop weights below that fixes baseline_input at +0.45
#      for the pyramidal node and -0.4 for the interneuron exactly, and
#      `stability.balanced_baselines` reproduces them; a test asserts it.
#   2. **The loop weights satisfy P3.4's global-uniqueness certificate**,
#      `contraction_number = sigma_max(diag(g/4) W) = 0.9 < 1`, so this
#      network has exactly one fixed point and every trajectory converges
#      to it whatever the taus. The weights are P5.1's, unchanged, because
#      the cortical microcircuit being represented is the same one and
#      changing them would have implied a difference that no source
#      supports.
#   3. **No node saturates under a saturating dose of a full agonist at
#      every receptor the node carries.** This is the rule that sets the
#      mechanism gains, and it is the first place in this codebase where it
#      bites on a NODE rather than on a receptor: `PFC_Pyramidal` carries
#      two excitatory receptors, so P5.1's per-receptor unit rule would put
#      +2.0 of drive on it at saturation, which rails it (minimum sigmoid
#      sensitivity 0.0044, below `stability.SATURATION_FLOOR` of 0.01). The
#      unit budget is therefore per NODE: the two cortical gains are +0.5
#      each, they sum to exactly the one unit P5.1's rule allots, and the
#      LC's single receptor keeps magnitude 1.0. A test asserts both halves
#      including the counterfactual, because a rule that cannot reject a
#      value is not a rule.
#
# Splitting the unit budget equally between alpha1 and beta is also what
# keeps this scenario honest: the two receptors are given the same gain by
# construction, so any asymmetry in an alpha1-versus-beta result comes from
# the published binding constants and from nothing hand-set.

#: ILLUSTRATIVE. The cortical loop weights, unchanged from P5.1's
#: `serotonin.raphe_cortical_network`; see rule 2 above.
CORTICAL_INHIBITION_WEIGHT = -0.9
CORTICAL_RECURRENT_WEIGHT = 0.8

#: ILLUSTRATIVE, and the same value every hand-built scenario in this
#: codebase uses.
LC_CORTEX_GAIN = 4.0

#: The node parameters, as one table so the rules above can be checked
#: against them by eye. `baseline_input` is fully determined by rule 1.
_NODES = (
    # name,            tau,  baseline_input
    ("LC_NA", 1.0, 0.0),
    ("PFC_Pyramidal", 1.5, 0.45),
    ("PFC_GABA", 1.0, -0.4),
)


def locus_coeruleus_cortical_network():
    """LC_NA (isolated) + PFC_Pyramidal -| PFC_GABA local cortical loop.

    See the section comment above for what each node is, which rule fixes
    each number, and why there is no LC -> cortex edge.
    """
    nodes = [
        CircuitNode(
            name=name, tau=tau, baseline_input=baseline, gain=LC_CORTEX_GAIN
        )
        for name, tau, baseline in _NODES
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    W[idx["PFC_Pyramidal"]][idx["PFC_GABA"]] = CORTICAL_INHIBITION_WEIGHT
    W[idx["PFC_GABA"]][idx["PFC_Pyramidal"]] = CORTICAL_RECURRENT_WEIGHT
    return CircuitNetwork(nodes=nodes, connectivity=W)


def rest_rates(network=None):
    """The rate vector rule 1 puts every node at with no drug on board.

    `Simulation.run` starts from zero rates by default, which for this
    network would make the first minutes of every trace a start-up
    transient rather than a drug effect. Passing `rates0=rest_rates()`
    starts at the fixed point instead, so a drug-free run is exactly flat
    and every deviation in a dosed run is the drug. This is P5.2's
    `dopamine.rest_rates` habit, adopted here because the effects this
    module has to show are small: the noradrenaline clearance proxy
    (`NORADRENALINE_KE`) removes the transmitter in seconds, so a cortical
    node with tau 1.5 min never reaches the rate its peak occupancy would
    support, and a start-up transient would swamp what is left.
    """
    n = len(_NODES) if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Coupling: three mechanism gains, two of them sharing one node
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE. How much a unit of alpha1 activation changes the drive of
#: the cortical pyramidal node. POSITIVE: alpha1 is Gq/11-coupled and
#: noradrenaline depolarizes cortical neurons through it by reducing a
#: resting K+ conductance ([McCormick1991]).
#:
#: Magnitude 0.5 by rule 3 above, not by tuning: this node carries TWO
#: excitatory receptors and the unit budget is per node, so the two share
#: it equally. Equal shares are also what make an alpha1-versus-beta
#: comparison a statement about the published affinities rather than about
#: two hand-set gains. Neither value is identifiable from anything
#: published -- P2.3 established that only the PRODUCT of a coupling gain
#: and a connection weight is identified even when behavioural data exists
#: -- and both are the `mechanism_gain` half of P4.2's
#: `coupling = mechanism_gain * rho`.
ALPHA1_MECHANISM_GAIN = 0.5

#: ILLUSTRATIVE. The beta1 counterpart on the same node, same magnitude by
#: the same rule. POSITIVE: beta is Gs-coupled, and noradrenaline acting at
#: beta receptors removes the calcium-activated K+ current underlying
#: spike-frequency adaptation ([Madison1986]), so the same drive
#: produces more spikes. That is a gain change rather than a synaptic
#: input, which is the cleanest justification available for putting it in
#: the drive term of a rate node -- and it is an approximation, stated
#: here, because a drive term and a gain term are not the same thing.
BETA_MECHANISM_GAIN = 0.5

#: ILLUSTRATIVE. The alpha2 autoreceptor gain on the LC node. NEGATIVE:
#: alpha2 is Gi/o-coupled, and clonidine applied to locus-coeruleus neurons
#: depresses their firing, an effect blocked by yohimbine
#: ([Cedarbaum1977]). Magnitude 1.0, P5.1's per-receptor unit rule, because
#: this node carries only one receptor. The difference in magnitude from
#: the two cortical gains is a consequence of rule 3's per-node budget and
#: NOT a claim that the autoreceptor is twice as powerful as either
#: cortical receptor.
ALPHA2_MECHANISM_GAIN = -1.0


def coupling_alpha1():
    """alpha1 activation -> drive, on the cortical pyramidal node."""
    return {"PFC_Pyramidal": ALPHA1_MECHANISM_GAIN}


def coupling_alpha2():
    """alpha2 activation -> drive, on the LC autoreceptor node."""
    return {"LC_NA": ALPHA2_MECHANISM_GAIN}


def coupling_beta():
    """beta1 activation -> drive, on the cortical pyramidal node."""
    return {"PFC_Pyramidal": BETA_MECHANISM_GAIN}


def receptor_system_alpha1():
    """The alpha1 stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_alpha1(),
        receptor_site=receptor_site_alpha1(),
        coupling=coupling_alpha1(),
        name="alpha1",
    )


def receptor_system_alpha2(affinity_state="agonist-state"):
    """The alpha2 stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_alpha2(),
        receptor_site=receptor_site_alpha2(affinity_state=affinity_state),
        coupling=coupling_alpha2(),
        name="alpha2",
    )


def receptor_system_beta(propranolol_affinity="kinetic"):
    """The beta1 stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_beta(),
        receptor_site=receptor_site_beta(propranolol_affinity=propranolol_affinity),
        coupling=coupling_beta(),
        name="beta",
    )


def noradrenergic_receptor_systems(
    affinity_state="agonist-state", propranolol_affinity="kinetic"
):
    """All three systems, alpha1 / alpha2 / beta, for `Simulation(...)`.

    Noradrenaline appears once per system, as three separate `Ligand`
    objects with three separate concentration states. That is not a
    duplication bug: they are three distinct receptor pools and the engine
    tracks occupancy per pool. One rise in noradrenaline is applied to all
    three entries by `Simulation.dose_vector(noradrenaline=...)`, which
    matches ligands by name across every system -- which is why all three
    systems have to give noradrenaline the SAME `ke`, or the model would
    describe one molecule clearing at three different rates. A test asserts
    that.
    """
    return [
        receptor_system_alpha1(),
        receptor_system_alpha2(affinity_state=affinity_state),
        receptor_system_beta(propranolol_affinity=propranolol_affinity),
    ]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def noradrenergic_tone_readout():
    """Locus-coeruleus noradrenergic-neuron firing.

    Falls under an alpha2 agonist: that is the autoreceptor. The same shape
    of readout as `serotonin.serotonergic_tone_readout` and
    `scenarios.arousal_index_readout` -- one node's own rate, suppressed by
    a Gi-coupled agonist -- and like them a circuit-level quantity rather
    than a fitted behavioural endpoint. Naming a behaviour (sedation, for
    the record, is what an alpha2 agonist produces) requires Phase 7.
    """
    return BehaviorReadout(name="noradrenergic_tone", weights={"LC_NA": 1.0})


def cortical_output_readout():
    """Layer V prefrontal pyramidal firing: the node alpha1 and beta share.

    Deliberately the same name, node and weight as
    `serotonin.cortical_output_readout`, and a test asserts the two are
    equal. The two modules describe the same cortical neuron with
    different receptors on it, so the day a combined monoamine scenario
    exists it should score both through one readout rather than through two
    lookalikes.
    """
    return BehaviorReadout(name="cortical_output", weights={"PFC_Pyramidal": 1.0})
