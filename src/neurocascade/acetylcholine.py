"""P5.4: the acetylcholine nicotinic and muscarinic receptor systems.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff`, `k_des`, `k_rec` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter below): every
equilibrium affinity at all four instantiated sites, the affinities of
scopolamine and of acetylcholine at all five muscarinic subtypes, the
channel-block potency of mecamylamine at both nicotinic sites, and -- the
constants this task exists for -- the DESENSITIZATION and RECOVERY rate
constants of the alpha4beta2 and alpha7 nicotinic receptors, both measured
in one laboratory in one preparation ([Fenster1997]).

ILLUSTRATIVE (hand-set, labelled, no literature target): one shared
association rate constant `ILLUSTRATIVE_KON` (no measured `kon` for any of
these ligands at any of these receptors was reached; see "Searched and
empty"), the three-node basal-forebrain/cortex circuit's weights, taus,
gains and baseline inputs, the four mechanism gains, and the
pharmacokinetics of the three dosed drugs plus the acetylcholine clearance
proxy. As in every scenario before Phase 7, this task calibrates the
receptor layer, not the circuit layer.

Why the two-state occupancy model was not enough
------------------------------------------------

Every receptor before this one fits `ReceptorSite`: a receptor is free or
bound, and bound receptor activates in proportion to `efficacy`. "Occupied"
and "active" are the same variable, so in that model raising an agonist's
concentration can only ever raise activation.

That is false for a nicotinic receptor, and not as a second-order detail:
it is the single most important fact about nicotinic pharmacology.
[Katz1957] established at the motor end-plate that acetylcholine drives the
receptor into a state that is still AGONIST-OCCUPIED and no longer
CONDUCTING, and that this desensitized state is entered and left with its
own rate constants. [Fenster1997] measured those rate constants for the two
neuronal subtypes modelled here and found the consequence that makes the
whole thing pharmacologically central:

    nicotine at rat alpha4beta2, activation EC50        15 uM
    nicotine at rat alpha4beta2, desensitization IC50    0.061 uM

A 246-fold window. Over most of that window nicotine's dominant action on
alpha4beta2 receptors is to REMOVE them from the available pool, not to
activate them -- and plasma nicotine in a smoker sits inside it. A
two-state `ReceptorSite` given nicotine's binding affinity produces a
monotonically increasing activation curve and cannot express this at all:
there is no state for the receptor to be in. Representing it by fitting a
smaller `efficacy` would be worse than useless, because efficacy is
concentration-independent and desensitization is not, and because the
receptor would then recover instantly rather than over the 87 minutes
[Fenster1997] measured.

So `receptor.py` gained `DesensitizingReceptorSite`, in exactly the way
GABA-A allosteric modulation made `engine.py` gain `connection_modulation`
(docs/architecture.md, "Third scenario"): the existing mechanism was the
wrong SHAPE, not merely the wrong size, and reusing it would have been a
mechanistically dishonest shortcut. See that class's docstring for the
equations and docs/architecture.md, P5.4, for the design argument.

Two smaller mechanisms come with it, both for the same reason:

  * **Mecamylamine is not a competitive antagonist.** It is a
    non-competitive, voltage-dependent OPEN-CHANNEL BLOCKER ([Papke2001]),
    and [Arias2010] locates its site in the channel lumen, between the
    non-polar (17') and outer (20') rings of the transmembrane domain --
    structurally distinct from the agonist site. Putting it in the
    competitive pool would assert that it displaces acetylcholine, which is
    the one thing it demonstrably does not do. It therefore gets its own
    pool and scales activation multiplicatively.
  * **The muscarinic sites are NOT desensitizing sites here** and stay
    plain `ReceptorSite`s. That is a modelling claim and it is made on
    purpose: muscarinic receptors are G-protein-coupled and their
    agonist-induced desensitization is a phosphorylation/arrestin process
    on a timescale of minutes to hours with no rate constant in reach for
    these ligands, whereas nicotinic desensitization is an intrinsic
    conformational change of the channel with measured rate constants. The
    contrast between the two halves of this module is the point: one
    transmitter, two receptor superfamilies, and only one of them needs the
    new state.

Sources
-------

The nicotinic kinetics, which are what this task is for:

  [Fenster1997] Fenster CP, Rains MF, Noerager B, Quick MW, Lester RAJ.
                "Influence of subunit composition on desensitization of
                neuronal acetylcholine receptors at low concentrations of
                nicotine." J Neurosci. 1997;17(15):5747-5759.
                doi:10.1523/JNEUROSCI.17-15-05747.1997 (PMID:9221773,
                PMC6573211, open access). RAT subunits in Xenopus oocytes,
                two-electrode voltage clamp, 20-25 C. ONE laboratory, ONE
                preparation, five subunit combinations, and every number
                this module's nicotinic sites turn on. Table 1:

                  subtype  nicotine   nicotine    tau_d (min)   recovery
                           EC50 (uM)  IC50 (uM)   +Ca    -Ca    tau (min)
                  a3b4     62 / 65    1.15        32.5   36.8   42.6
                  a4b4      9 / 11    0.042       35.5   16.2   413.8
                  a3b2    123 / 75    0.33        ~5      3.9    11.7
                  a4b2     15 / 12    0.061        3.7    2.7    86.9
                  a7       90 / 234   1.3 (+Ca)    0.5    1.9     1.9
                                      6.2 (-Ca)

                (EC50 given +Ca / -Ca; tau_d from a single exponential fit
                at near-IC50 nicotine; recovery from a single exponential
                fit after washout of 1 uM nicotine, 10 uM for alpha7.)
                Errors, where the table gives them: a4b2 tau_d 3.7 +/- 0.9
                (n=4), recovery 86.9 +/- 64.7 (n=4); a7 tau_d 0.5 +/- 0.03
                (n=3), recovery 1.9 +/- 0.5 (n=5). The alpha7 nicotine
                concentration-response has a Hill coefficient of 1.1 and
                EC50s of 79 uM (3.6 mM Ca) and 90 uM (1.8 mM Ca). The three
                other rows are kept in this docstring because they are what
                makes the alpha4beta2-versus-alpha7 contrast a comparison
                rather than two quotations: within one experiment the
                desensitization onset spans 71-fold and the recovery
                spans 218-fold across subtypes.
  [Fenster1999] Fenster CP, Beckman ML, Parker JC, Sheffield EB, Whitworth
                TL, Quick MW, Lester RA. "Regulation of alpha4beta2
                nicotinic receptor desensitization by calcium and protein
                kinase C." Mol Pharmacol. 1999;55(3):432-443.
                (PMID:10051526). Same laboratory, same preparation, a
                second and independent reading of the same two rate
                constants: 300 nM nicotine for 30 min desensitizes
                alpha4beta2 biphasically with tau_f = 1.4 min (65% relative
                amplitude) and tau_s = 17 min, and recovery is a single
                exponential with tau_rec = 43 min. Used here as the error
                bar on [Fenster1997]'s single-exponential 3.7 min and
                86.9 min rather than as a competing value; see "How much
                the desensitization constants are actually worth".
  [Katz1957]    Katz B, Thesleff S. "A study of the desensitization
                produced by acetylcholine at the motor end-plate." J
                Physiol. 1957;138(1):63-80. doi:10.1113/jphysiol.1957.sp005838
                (PMID:13463799). The cyclic scheme
                `free <-> bound <-> desensitized` that
                `DesensitizingReceptorSite` implements, and the origin of
                the fact that the desensitized state is the HIGH-affinity
                one. No number is taken from it; the mechanism is.

Nicotinic equilibrium affinities:

  [Sanders2022] Sanders VR, Sweeney A, Topf M, Millar NS.
                "Stoichiometry-Selective Antagonism of alpha4beta2
                Nicotinic Acetylcholine Receptors by Fluoroquinolone
                Antibiotics." ACS Chem Neurosci. 2022;13(12):1805-1817.
                doi:10.1021/acschemneuro.2c00200 (PMID:35657695,
                PMC9204775). HUMAN alpha4 and beta2 in Xenopus oocytes,
                two-electrode voltage clamp at -60 mV, concatemers fixing
                the stoichiometry. ACETYLCHOLINE EC50:
                  high-sensitivity (alpha4)2(beta2)3   1.6 +/- 0.1 uM
                                                       (n=5, nH 0.83+/-0.2)
                  low-sensitivity  (alpha4)3(beta2)2  37.1 +/- 6.8 uM
                                                       (n=3, nH 0.74+/-0.1)
                Both are recorded; the 23-fold spread between them is the
                stoichiometry uncertainty on any single "alpha4beta2"
                statement, and it is this module's counterpart of P5.3's
                alpha1-subtype spread.
  [PapkePapke2002] Papke RL, Porter Papke JK. "Comparative pharmacology of
                rat and human alpha7 nAChR conducted with net charge
                analysis." Br J Pharmacol. 2002;137(1):49-61.
                doi:10.1038/sj.bjp.0704833 (PMID:12183330, PMC1573461, open
                access). Xenopus oocytes, room temperature. The ACh EC50 at
                alpha7 depends on whether the measurement corrects for
                desensitization DURING the measurement: net-charge analysis
                of rat alpha7 gives EC50 = 30 uM (with a residual-inhibition
                IC50 of 68 uM in the same fit), whereas the paper's Table 3
                of previously published PEAK-CURRENT estimates averages
                "about 200 uM and range[s] from 107-334 uM", and the authors
                state that "analysis of net charge lowers the estimate for
                the ACh EC50 by at least a factor of five". That sentence is
                this module's independent evidence that a receptor which
                desensitizes cannot be characterized by an occupancy model
                that has no desensitized state: the number one measures
                depends on how much of the receptor desensitized while one
                was measuring it.

Nicotinic channel block:

  [Papke2013]   Papke RL, Stokes C, Muldoon P, Damaj MI. "Similar activity
                of mecamylamine stereoisomers in vitro and in vivo." Eur J
                Pharmacol. 2013;720(1-3):264-275.
                doi:10.1016/j.ejphar.2013.10.018 (PMID:24161916,
                PMC3992716, open access). Xenopus oocytes, -60 mV. Table 3
                peak-current IC50 at alpha4beta2:
                  (alpha4)2(beta2)3  R-mecamylamine 19 +/- 5 uM
                                     S-mecamylamine 17 +/- 4 uM
                  (alpha4)3(beta2)2  R-mecamylamine 2.6 +/- 0.3 uM
                                     S-mecamylamine 2.2 +/- 0.2 uM
                (net-charge IC50s are 1.3/1.0 and 0.24/0.16 uM
                respectively). n >= 4 oocytes per condition. alpha7 was NOT
                tested in that study; the authors state mecamylamine is
                "least potent for alpha7".
  [ChavezNoriega1997] Chavez-Noriega LE, Crona JH, Washburn MS, Urrutia A,
                Elliott KJ, Johnson EC. "Pharmacological characterization
                of recombinant human neuronal nicotinic acetylcholine
                receptors h-alpha2beta2, h-alpha2beta4, h-alpha3beta2,
                h-alpha3beta4, h-alpha4beta2, h-alpha4beta4 and h-alpha7
                expressed in Xenopus oocytes." J Pharmacol Exp Ther.
                1997;280(1):346-356 (PMID:8996215). The alpha7
                mecamylamine number, and the only one found that measures
                alpha4beta2 and alpha7 in the same experiment: "MEC (3 uM)
                inhibited h-alpha2beta4 and h-alpha4beta4 nAChRs by > 80%,
                whereas h-alpha2beta2, h-alpha4beta2 and h-alpha7 nAChRs
                were inhibited by approximately 50%." Half-block at 3 uM at
                both of this module's subtypes. It is a single-concentration
                readout rather than a fitted IC50, and is used as such.
  [Papke2001]   Papke RL, Sanberg PR, Shytle RD. "Analysis of mecamylamine
                stereoisomers on human nicotinic receptor subtypes." J
                Pharmacol Exp Ther. 2001;297(2):646-656 (PMID:11303054).
                The mechanism, not a number: "Mecamylamine inhibition of
                neuronal nAChR was noncompetitive and voltage dependent",
                and its selectivity "was manifested primarily in terms of
                slow recovery rates from mecamylamine-induced inhibition".
  [Arias2010]   Arias HR, Feuerbach D, Targowska-Duda KM, Russell M,
                Jozwiak K. "Tricyclic antidepressants and mecamylamine bind
                to different sites in the human alpha4beta2 nicotinic
                receptor ion channel." Int J Biochem Cell Biol.
                2010;42(6):1007-1018. doi:10.1016/j.biocel.2010.03.002
                (PMID:20223294, PMC4605392). HEK293-h-alpha4beta2
                membranes, room temperature. Cited for the LOCATION of the
                mecamylamine site -- the channel lumen, "between the
                non-polar (position 17') and outer (position 20') rings" --
                which is why this module gives it a pool of its own instead
                of a seat in the competitive one. Its [3H]imipramine
                competition IC50s (135 +/- 19 uM resting, 209 +/- 34 uM
                desensitized, nH ~0.55) are recorded and NOT used: they
                measure displacement from a different luminal site, not
                functional block.

Muscarinic affinities. These are the weakest constants in the module and
the reason is stated rather than hidden:

  [GtoPdb2026]  Harding SD, Armstrong JF, Faccenda E, et al. IUPHAR/BPS
                Guide to PHARMACOLOGY, release 2026.2 (2026-06-15),
                `interactions.csv`. https://www.guidetopharmacology.org.
                A CURATED DATABASE, not a primary measurement, and used
                here for the muscarinic constants alone because no
                single-laboratory M1-M5 panel carrying BOTH acetylcholine
                and scopolamine was reachable in full text (see "Searched
                and empty"). Human pKi as curated, with the primary
                references the Guide attributes each entry to:
                  acetylcholine  M1 4.3-4.9   [Jakubik1997], [Keov2013]
                                 M2 4.3-6.5   [Cheng2002], [Jakubik1997],
                                              [Keov2013]
                                 M3 4.5-5.4   same three
                                 M4 4.5-5.6   [Jakubik1997], [Keov2013]
                                 M5 6.1       [Cheng2002]
                  scopolamine    M1 9.0       [Huang2001], [Croy2016]
                                 M2 8.7       [Huang2001], [Bolden1992]
                                 M3 9.4       [Huang2001], [Bolden1992]
                                 M4 9.1-9.5   [Huang2001], [Bolden1992]
                                 M5 8.7       [Bolden1992]
                The Guide also records rat acetylcholine pKi of 5.0 (M1),
                6.4 (M2) and 5.6 (M3), which bracket the human ranges and
                are recorded below for the same reason.
  [Bolden1992]  Bolden C, Cusack B, Richelson E. "Antagonism by
                antimuscarinic and neuroleptic compounds at the five cloned
                human muscarinic cholinergic receptors expressed in Chinese
                hamster ovary cells." J Pharmacol Exp Ther.
                1992;260(2):576-580 (PMID:1346637). The single-laboratory
                M1-M5 SCOPOLAMINE panel: [3H]quinuclidinyl benzilate
                competition on CHO-K1 membranes expressing each of the five
                human subtypes. It is the only source reached that covers
                M5, and [GtoPdb2026] attributes the M2-M5 scopolamine
                entries above to it. Its full text is behind a
                Cloudflare-protected publisher site and could not be
                retrieved, so the numbers used are the Guide's transcription
                of it and are labelled as such.
  [Huang2001]   Huang F, Buchwald P, Browne CE, Farag HH, Wu WM, Ji F,
                Hochhaus G, Bodor N. "Receptor binding studies of soft
                anticholinergic agents." AAPS PharmSci. 2001;3(4):E30.
                doi:10.1208/ps030430 (PMID:12049493, PMC2751219). Four
                cloned human muscarinic subtypes (m1-m4), and the second
                laboratory behind the scopolamine entries at M1-M4.
  [Jakubik1997] Jakubik J, Bacakova L, El-Fakahany EE, Tucek S. "Positive
                cooperativity of acetylcholine and other agonists with
                allosteric ligands on muscarinic acetylcholine receptors."
                Mol Pharmacol. 1997;52(1):172-179. doi:10.1124/mol.52.1.172
                (PMID:9224827). CHO membranes stably expressing M1-M4,
                acetylcholine competing against [3H]N-methylscopolamine.
                One of the primaries behind the acetylcholine entries.
  [Keov2013]    Keov P, Lopez L, Devine SM, et al. "Molecular mechanisms of
                bitopic ligand engagement with the M1 muscarinic
                acetylcholine receptor." Mol Pharmacol. 2013;84(3):425-437
                (PMID:23798605). A second primary behind the acetylcholine
                entries.
  [Cheng2002]   Cheng K, Khurana S, Chen Y, Kennedy RH, Zimniak P, Raufman
                JP. "Lithocholylcholine, a bile acid/acetylcholine hybrid,
                is a muscarinic receptor antagonist." J Pharmacol Exp Ther.
                2002;303(1):29-35. doi:10.1124/jpet.102.036376
                (PMID:12235229). CHO cells expressing each of the five
                subtypes; the primary behind the M5 acetylcholine entry.
  [Croy2016]    Croy CH, Chan WY, Castetter AM, Watt ML, Quets AT, Felder
                CC. "Characterization of PCS1055, a novel muscarinic M4
                receptor antagonist." Eur J Pharmacol. 2016;782:70-76.
                doi:10.1016/j.ejphar.2016.04.022 (PMID:27085897).
                [3H]NMS competition; a second primary behind the M1
                scopolamine entry.

Anatomy and mechanism (no binding numbers taken from these; they fix which
node carries which receptor and with what sign):

  [McCormick1986] McCormick DA, Prince DA. "Mechanisms of action of
                acetylcholine in the guinea-pig cerebral cortex in vitro."
                J Physiol. 1986;375:169-194.
                doi:10.1113/jphysiol.1986.sp016112 (PMID:2879035). The M1
                sign: acetylcholine slowly depolarizes cortical pyramidal
                neurons through a muscarinic receptor by reducing a
                potassium conductance, and blocks the calcium-activated
                potassium conductance underlying the after-hyperpolarization.
                Excitatory on a rate node, and the same shape of argument
                P5.3 used from [McCormick1991] for alpha1.
  [Kassam2008]  Kassam SM, Herman PM, Goodfellow NM, Alves NC, Lambe EK.
                "Developmental excitation of corticothalamic neurons by
                nicotinic acetylcholine receptors." J Neurosci.
                2008;28(35):8756-8764.
                doi:10.1523/JNEUROSCI.2645-08.2008 (PMID:18753377). The
                alpha4beta2 sign and location: nicotinic receptors
                containing alpha4 and beta2 subunits directly depolarize
                layer VI prefrontal pyramidal neurons.
  [Alkondon1993] Alkondon M, Albuquerque EX. "Diversity of nicotinic
                acetylcholine receptors in rat hippocampal neurons. I.
                Pharmacological and functional evidence for distinct
                structural subtypes." J Pharmacol Exp Ther.
                1993;265(3):1455-1473 (PMID:8510022). The alpha7 sign and
                its defining kinetic signature on central neurons: the
                alpha-bungarotoxin-sensitive, methyllycaconitine-sensitive
                type IA current is a fast, inward, rapidly desensitizing
                cationic current.
  [Zhang2002]   Zhang W, Basile AS, Gomeza J, Volpicelli LA, Levey AI, Wess
                J. "Characterization of central inhibitory muscarinic
                autoreceptors by the use of muscarinic acetylcholine
                receptor knock-out mice." J Neurosci. 2002;22(5):1709-1717.
                doi:10.1523/JNEUROSCI.22-05-01709.2002 (PMID:11880500). The
                M2 sign and location, from knock-out mice rather than from
                a selective drug: M2 (with M4) mediates the inhibitory
                muscarinic autoreceptor that suppresses acetylcholine
                release in cortex and hippocampus.

Derivations
-----------

P5.1's three-way split, with a fourth case this task adds:

1. **Both binding constants measured** -- use them. Not available anywhere
   in this module.
2. **koff and an equilibrium affinity measured, kon missing** --
   `kon = koff / Kd`. Not available either.
3. **Only an equilibrium affinity measured** -- share the site's measured
   `kon` and set `koff = kon * Kd`. There is no measured `kon` at ANY of
   these four sites, so the shared value is `ILLUSTRATIVE_KON`, an
   explicitly illustrative constant rather than a borrowed measurement.
   Every affinity is still reproduced EXACTLY (`koff = kon * Kd` by
   construction); the entire uncertainty lands on the timescale over which
   binding approaches equilibrium, and P5.3's [Sykes2022] panel says that
   timescale could be wrong by up to ~230-fold. Every result in this module
   that depends on a BINDING rate rather than on an affinity says so.
   Nothing in this module's headline results does, because the rates that
   matter here are the desensitization rates, and those are measured.
4. **A potency, not an affinity.** Every nicotinic number available is a
   functional EC50 or IC50 from voltage-clamp, not a binding Kd. This
   module uses them as the site's Kd and says so at each constant. The
   assumption is that activation is proportional to conducting-state
   occupancy with no receptor reserve, which is exactly the assumption
   `activation = sum(occupancy * efficacy)` already makes; it is the same
   approximation P5.2 and P5.3 flagged when their affinity and potency
   orderings disagreed, applied here in the only direction the data allow.

How much the desensitization constants are actually worth
---------------------------------------------------------

Two independent readings exist for alpha4beta2, from one laboratory:

    quantity          [Fenster1997]        [Fenster1999]
    desensitization   3.7 +/- 0.9 min      1.4 min (65%) / 17 min
                      (single exponential) (biphasic)
    recovery          86.9 +/- 64.7 min    43 min

The recovery constants differ 2.0-fold and the [Fenster1997] value's own
standard error is 74% of its mean, so anything this module says about
recovery TIMING is good to about a factor of two and no better. The
desensitization onset is worse than that in a way a single number hides:
[Fenster1997] itself reports (its Table 2) that beta2-containing receptors
need TWO components to describe the decay -- tau_slow 90.7 +/- 5.9 s and
tau_fast 5.4 +/- 1.5 s for alpha4beta2 -- and concludes that "these data
imply multiple desensitized states for these receptors".
`DesensitizingReceptorSite` has ONE desensitized state per ligand and
therefore cannot hold two components. `DESENSITIZATION_READINGS` carries
all three alpha4beta2 readings so the choice is visible, the default is
[Fenster1997]'s single-exponential fit because it is the number measured in
the same table and the same experiment as everything else here, and a test
asserts the 2.0-fold recovery spread rather than letting it be forgotten.

The prediction this model makes, and gets wrong by 10-fold
----------------------------------------------------------

In the Katz-Thesleff cycle, trapping ligand in a desensitized state that
cannot dissociate makes the site's APPARENT equilibrium affinity tighter
than its conducting-state affinity by exactly `1 + k_des/k_rec` (see
`DesensitizingReceptorSite.apparent_affinity_factor`). That is not a free
parameter here: both rates are measured, so the factor is a PREDICTION, and
[Fenster1997] measured the thing it predicts -- the gap between an
activation EC50 and a desensitization IC50 on the same receptor in the same
experiment.

    site       1 + k_des/k_rec   EC50/IC50 measured   model / measured
    alpha4beta2      24.5              246                 1 / 10.0
    alpha7            4.8               69                 1 / 14.4

The model reproduces the DIRECTION at both sites, reproduces an order of
magnitude of the shift, and under-predicts it by 10 to 14-fold -- in the
same direction and by a similar factor at two receptors whose measured
rates differ 7.4-fold (`k_des`) and 45.7-fold (`k_rec`). That consistency
is what
makes it a finding rather than noise, and the mechanism [Fenster1997] names
is the obvious candidate: with more than one desensitized state, and in
particular with the deep slow state that a 30-minute exposure reaches, the
true trapping ratio is larger than the one a single-exponential fit to the
first few minutes reports. A test asserts both factors so the claim cannot
rot, and asserts the direction (the model under-predicts, never over-
predicts) separately, because the direction is the part that would falsify
the single-state simplification if it flipped.

No number here was adjusted to close that gap. Fitting `k_rec` to make the
factor come out at 246 would have produced a receptor that recovers from
desensitization in 8 minutes rather than the 43-87 minutes two experiments
measured, which is a worse model wearing a better-looking number.

Searched and empty, or found and deliberately not used
------------------------------------------------------

* **Association rate constants for any of these ligands at any of these
  receptors.** Searched Europe PMC for acetylcholine, nicotine,
  scopolamine and mecamylamine crossed with "association rate constant",
  "dissociation rate constant", "binding kinetics" and "residence time",
  and through the kinetic literature for both superfamilies. What exists is
  either for a radioligand rather than a competitor
  ([3H]N-methylscopolamine, [125I]alpha-bungarotoxin), or for a different
  receptor family, or -- [Sykes2009] (Sykes DA, Dowling MR, Charlton SJ,
  "Exploring the mechanism of agonist efficacy: a relationship between
  efficacy and agonist dissociation rate at the muscarinic M3 receptor",
  Mol Pharmacol 2009;76(3):543-551, PMID:19498041), which measures exactly
  the quantity wanted for acetylcholine at M3 -- behind a
  Cloudflare-protected publisher site whose full text could not be
  retrieved. `ILLUSTRATIVE_KON` is therefore an explicit illustrative
  constant shared by all four sites, and the [Sykes2009] reference is
  recorded here because it is the specific paper that would remove it.
* **A relative maximal response for nicotine against acetylcholine at
  alpha4beta2 or alpha7 in one preparation.** [ChavezNoriega1997] states
  only that "ACh was one of the most efficacious agonists at all hnAChRs,
  except at h-alpha3beta2"; [Fenster1997] normalizes nicotine responses to
  ACh test pulses without reporting a ratio; the alpha7 literature reached
  reports Imax relative to ACh for nornicotine (~50%, [Papke2007]) and for
  a series of experimental partial agonists, but not for nicotine itself.
  `NICOTINE_EFFICACY` is therefore 1.0, i.e. nicotine is modelled as a full
  agonist at both sites, and that is a stated simplification rather than a
  measurement. It is in the conventional direction -- partial agonists such
  as varenicline are routinely benchmarked as a fraction of nicotine's
  maximum -- and no result in this module depends on it, because both
  nicotinic results are about desensitization, which is driven by occupancy
  and not by efficacy. A test asserts that the desensitization results hold
  at NICOTINE_EFFICACY = 0.5 as well.
* **A single-laboratory M1-M5 panel carrying both acetylcholine and
  scopolamine.** [Bolden1992] is the M1-M5 scopolamine panel and does not
  report acetylcholine; [Jakubik1997] is an acetylcholine panel and stops
  at M4; [Wood1999] (Wood MD, Murkitt KL, Ho M, et al., "Functional
  comparison of muscarinic partial agonists at muscarinic receptor
  subtypes hM1, hM2, hM3, hM4 and hM5 using microphysiometry", Br J
  Pharmacol 1999;126(7):1620-1624, PMID:10323594) covers all five but uses
  CARBACHOL as its reference full agonist rather than acetylcholine, and
  substituting carbachol's potency for acetylcholine's would be exactly the
  ligand substitution this project forbids. Hence [GtoPdb2026], with the
  primaries named and the curated status labelled.
* **Rodent pharmacokinetics for nicotine, scopolamine and mecamylamine.**
  All three have published rodent PK; nicotine's is extensive. Not used,
  for P5.3's two reasons, unchanged: `drug_pk.py` consumes a full
  per-route parameter set (F, ka, Vd, CL, brain partition) rather than a
  half-life, and half-sourcing one drug would put a real citation beside
  two invented numbers and thereby assert a clearance ORDERING nothing
  supports. All three share one illustrative `TOOL_COMPOUND_KE`,
  deliberately the same number as `serotonin.TOOL_COMPOUND_KE`,
  `dopamine.TOOL_COMPOUND_KE` and `noradrenaline.TOOL_COMPOUND_KE`, and a
  test asserts the equality. Do not cite it as a half-life for any of them.
* **Acetylcholine's own clearance rate.** The same gap the other three
  transmitters have, and the most extreme case of it: extracellular
  acetylcholine is hydrolysed by acetylcholinesterase, one of the fastest
  enzymes known, on a sub-millisecond timescale. Taken literally that
  collapses an acetylcholine "dose" into a delta function far inside one
  integration step. `ACETYLCHOLINE_KE` is an explicit flagged proxy set to
  the same 5.0 min^-1 as `dopamine.DOPAMINE_KE`, `serotonin.SEROTONIN_KE`
  and `noradrenaline.NORADRENALINE_KE`; the real quantity belongs to P6.2
  (endogenous tone).
* **A behavioural dose-response usable as a circuit-layer target.** Not
  searched again, for the reason Track E and P2.2 established: behavioural
  pharmacology reports mg/kg by IP/SC/PO and this module has no such
  kinetics.

What was deliberately not built
-------------------------------

* **Voltage- and use-dependence of mecamylamine's block.** [Papke2001]
  shows the block is both, and this engine has no membrane potential and no
  channel state for the "use" in use-dependence to be a function of. The
  multiplicative form here reproduces what a functional IC50 measures at
  one holding potential (-60 mV) and nothing more. What follows is that
  mecamylamine's block in this model does not grow with agonist
  concentration, and real open-channel block does.
* **The alpha4beta2 stoichiometry split.** [Sanders2022] measures ACh EC50
  at 1.6 uM and 37.1 uM for (alpha4)2(beta2)3 and (alpha4)3(beta2)2, and
  [Papke2013] measures mecamylamine at 17-19 uM and 2.2-2.6 uM for the same
  two -- i.e. the two stoichiometries differ 23-fold in one direction for
  the agonist and 7-fold in the OTHER direction for the blocker. Both are
  recorded as constants; the high-sensitivity form is instantiated,
  because it is the one nanomolar nicotine engages and this module's
  headline result is about nanomolar nicotine. This is a receptor-assembly
  question, not a regional one, so no ISH map could settle it, exactly as
  P5.3 recorded for its subtype split.
* **The M3, M4 and M5 sites.** Their acetylcholine and scopolamine
  affinities are all defined below, because the task is a five-subtype one
  and because the cross-subtype comparison is the honest way to state how
  non-selective scopolamine is. They are not instantiated because this
  three-node circuit has no node whose identity would fix which of them it
  carries: M4 is the striatal autoreceptor and there is no striatum here,
  M3 and M5 are sparse in cortex, and inventing a node to hold a receptor
  is the failure mode P5.2 recorded.
* **The basal forebrain -> cortex edge.** Same decision, and the same
  reason, as P5.1's DRN and P5.3's LC: acetylcholine's effect on a cortical
  pyramidal cell is receptor-mediated and this module gives that cell three
  different acetylcholine receptors. A signed weight from `BF_Cholinergic`
  to `PFC_Pyramidal` would double-count the receptor layer, which is the
  layer this codebase exists for. P6.1.
* **Endogenous cholinergic tone.** Without it scopolamine and mecamylamine
  given alone are inert, as everywhere else in this codebase, and real
  scopolamine is not. P6.2. The tests that assert inertness say so.
* **An atlas-derived density factor.** Same position as P5.1 to P5.3.
  `Chrm1` to `Chrm5`, `Chrna4` and `Chrna7` are all already in
  `expression/genes.py` tagged P5.4, so the upgrade is a table lookup away;
  it is not taken because the committed expression fixture does not carry
  these genes and a density-derived map would make every test in this
  module depend on the gitignored cache.
* **Nicotine's upregulation of alpha4beta2 receptor number.** It is real,
  it is downstream of exactly the desensitized state this module now has
  ([Fenster1999] is a paper about that link), and it happens over days.
  This model has no receptor-number state and no timescale that long.
"""

import numpy as np

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import (
    DesensitizingReceptorSite,
    Ligand,
    ReceptorSite,
    ligand_from_ki,
)


def _kd_from_pki(p_ki):
    """nM from a pKi (which is -log10 of molar Ki): `Kd(nM) = 10^(9 - pKi)`."""
    return 10.0 ** (9.0 - p_ki)


def _rate_from_time_constant(tau_min):
    """min^-1 from a time constant in minutes. Every desensitization and
    recovery constant in the literature is reported as a time constant, and
    `DesensitizingReceptorSite` wants a rate, so the inversion is written
    once here rather than open-coded at each constant."""
    return 1.0 / tau_min


# ---------------------------------------------------------------------------
# The one illustrative binding rate constant, shared by all four sites
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. ILLUSTRATIVE. There is no measured association rate
#: constant for acetylcholine, nicotine, scopolamine or mecamylamine at any
#: of the four receptors modelled here (see "Searched and empty"), so
#: derivation 3's "share the site's measured kon" has nothing to share and
#: this is an explicit hand-set number instead of a borrowed measurement.
#:
#: 0.01 nM^-1 min^-1 is chosen by two stated constraints rather than by
#: taste. It is at the slow end of the only measured kon panel this codebase
#: has ([Sykes2022] at beta1, 0.00847 to 1.97 nM^-1 min^-1 across 18
#: ligands), so it does not assert unusually fast binding; and it is fast
#: enough that binding equilibrates in well under a minute at every affinity
#: below, so it does not become the rate-limiting step in front of the
#: measured desensitization constants, which are what this module is for. A
#: test asserts the second property at every site.
#:
#: Every `koff` derived from it reproduces its site's measured AFFINITY
#: exactly; only the approach timescale is illustrative.
ILLUSTRATIVE_KON = 0.01


# ---------------------------------------------------------------------------
# alpha4beta2: the site the desensitization window belongs to
# ---------------------------------------------------------------------------

#: nM. MEASURED acetylcholine EC50 at the HIGH-SENSITIVITY human
#: (alpha4)2(beta2)3 concatemer in Xenopus oocytes, 1.6 +/- 0.1 uM (n=5,
#: Hill 0.83) [Sanders2022]. Derivation 4: a functional potency used as the
#: site's conducting-state Kd.
ALPHA4BETA2_KD_ACETYLCHOLINE = 1600.0

#: nM. MEASURED acetylcholine EC50 at the LOW-SENSITIVITY
#: (alpha4)3(beta2)2 concatemer in the same experiment, 37.1 +/- 6.8 uM
#: (n=3) [Sanders2022]. Not instantiated; kept because the 23-fold spread
#: between the two stoichiometries is the honest error bar on any
#: "alpha4beta2" statement, and a test asserts the ratio.
ALPHA4BETA2_KD_ACETYLCHOLINE_LOW_SENSITIVITY = 37100.0

#: nM. MEASURED nicotine activation EC50 at rat alpha4beta2 in
#: calcium-containing medium, 15 uM [Fenster1997] Table 1. The
#: calcium-free value is 12 uM; the calcium-containing one is used
#: throughout this module because brain extracellular fluid contains
#: calcium and because [Fenster1997] shows calcium changes alpha7's
#: numbers materially.
ALPHA4BETA2_KD_NICOTINE = 15000.0
ALPHA4BETA2_KD_NICOTINE_CALCIUM_FREE = 12000.0

#: nM. MEASURED nicotine desensitization IC50 at rat alpha4beta2, 0.061 uM
#: [Fenster1997] Table 1, which that paper calls "the apparent affinity of
#: nicotine for the desensitized state of the receptor". NOT a constant the
#: model is given: it is the quantity the model's
#: `apparent_affinity_factor` predicts, and the comparison is in the module
#: docstring and in the tests.
ALPHA4BETA2_DESENSITIZATION_IC50_NICOTINE = 61.0

#: min. MEASURED time constants at rat alpha4beta2 [Fenster1997] Table 1:
#: a single exponential fit to the development of desensitization at
#: near-IC50 nicotine (3.7 +/- 0.9 min, n=4, calcium-containing; 2.7 +/- 0.3
#: calcium-free), and to recovery after washout of 1 uM nicotine
#: (86.9 +/- 64.7 min, n=4).
ALPHA4BETA2_TAU_DESENSITIZATION = 3.7
ALPHA4BETA2_TAU_DESENSITIZATION_CALCIUM_FREE = 2.7
ALPHA4BETA2_TAU_RECOVERY = 86.9

#: min. MEASURED, [Fenster1999], same laboratory and preparation: 300 nM
#: nicotine for 30 min desensitizes alpha4beta2 biphasically (tau_f 1.4 min
#: at 65% relative amplitude, tau_s 17 min) and recovery is a single
#: exponential of 43 min. The second reading of the same two quantities, and
#: this module's error bar on them; see "How much the desensitization
#: constants are actually worth".
ALPHA4BETA2_TAU_DESENSITIZATION_FAST_1999 = 1.4
ALPHA4BETA2_TAU_DESENSITIZATION_SLOW_1999 = 17.0
ALPHA4BETA2_TAU_RECOVERY_1999 = 43.0

#: sec. MEASURED, [Fenster1997] Table 2: the two components of the current
#: decay during a brief nicotine pulse at alpha4beta2, tau_slow 90.7 +/- 5.9
#: and tau_fast 5.4 +/- 1.5 s (calcium-containing), with the fast component
#: carrying 40.3 +/- 3.2% of the amplitude. Recorded, not used: they are the
#: direct evidence that this receptor has more than one desensitized state,
#: which is what `DesensitizingReceptorSite` deliberately does not
#: represent.
ALPHA4BETA2_PULSE_TAU_SLOW_SEC = 90.7
ALPHA4BETA2_PULSE_TAU_FAST_SEC = 5.4

#: nM. MEASURED mecamylamine peak-current IC50 at the high-sensitivity
#: (alpha4)2(beta2)3 receptor [Papke2013] Table 3: R-mecamylamine
#: 19 +/- 5 uM, S-mecamylamine 17 +/- 4 uM. Mecamylamine as used is racemic
#: and that paper's whole point is that the stereoisomers behave alike, so
#: the value here is the mean of the two measured enantiomers, 18 uM, and
#: the two are kept separately so that is checkable.
ALPHA4BETA2_IC50_MECAMYLAMINE_R = 19000.0
ALPHA4BETA2_IC50_MECAMYLAMINE_S = 17000.0
ALPHA4BETA2_IC50_MECAMYLAMINE = 0.5 * (
    ALPHA4BETA2_IC50_MECAMYLAMINE_R + ALPHA4BETA2_IC50_MECAMYLAMINE_S
)

#: nM. MEASURED mecamylamine peak-current IC50 at the LOW-sensitivity
#: (alpha4)3(beta2)2 receptor in the same table, 2.6 +/- 0.3 (R) and
#: 2.2 +/- 0.2 uM (S). Recorded because it runs OPPOSITE to the agonist
#: stoichiometry preference: the stoichiometry acetylcholine prefers 23-fold
#: is the one mecamylamine blocks 7-fold less well.
ALPHA4BETA2_IC50_MECAMYLAMINE_LOW_SENSITIVITY = 2400.0


# ---------------------------------------------------------------------------
# alpha7: the same three constants, 4 to 46-fold different
# ---------------------------------------------------------------------------

#: nM. MEASURED acetylcholine EC50 at rat alpha7 by NET CHARGE analysis,
#: 30 uM [PapkePapke2002]. The peak-current literature that paper tabulates
#: averages ~200 uM over a 107-334 uM range, and the authors state net
#: charge lowers the estimate "by at least a factor of five". The net-charge
#: value is used, by a stated rule: it is the estimate that corrects for the
#: desensitization occurring DURING the measurement, and this module's whole
#: subject is that desensitization. The peak-current range is kept below.
ALPHA7_KD_ACETYLCHOLINE = 30000.0
ALPHA7_KD_ACETYLCHOLINE_PEAK_CURRENT_LOW = 107000.0
ALPHA7_KD_ACETYLCHOLINE_PEAK_CURRENT_HIGH = 334000.0

#: nM. MEASURED nicotine activation EC50 at rat alpha7 in 1.8 mM calcium,
#: 90 uM (Hill 1.1, n=4) [Fenster1997]; 79 uM in 3.6 mM calcium and 234 uM
#: (n=10) calcium-free. The calcium-containing value is used, as at
#: alpha4beta2.
ALPHA7_KD_NICOTINE = 90000.0
ALPHA7_KD_NICOTINE_CALCIUM_FREE = 234000.0

#: nM. MEASURED nicotine desensitization IC50 at rat alpha7, 1.3 uM in
#: calcium-containing and 6.2 uM in calcium-free medium [Fenster1997]. As at
#: alpha4beta2 this is a prediction target, not an input.
ALPHA7_DESENSITIZATION_IC50_NICOTINE = 1300.0
ALPHA7_DESENSITIZATION_IC50_NICOTINE_CALCIUM_FREE = 6200.0

#: min. MEASURED time constants at rat alpha7 [Fenster1997] Table 1:
#: desensitization 0.5 +/- 0.03 min (n=3, calcium-containing) and
#: 1.9 +/- 0.5 min (n=4, calcium-free), recovery 1.9 +/- 0.5 min (n=5). The
#: table prints 1.9 +/- 0.5 twice in this row with different n; the column
#: layout of the other four rows makes the first the calcium-free tau_d and
#: the second the recovery constant, and the paper's own statement that
#: calcium had "little observable effect on the time course of
#: desensitization (t = 2.7; p > 0.05)" is consistent with 0.5 and 1.9 being
#: the two tau_d values at those sample sizes.
ALPHA7_TAU_DESENSITIZATION = 0.5
ALPHA7_TAU_DESENSITIZATION_CALCIUM_FREE = 1.9
ALPHA7_TAU_RECOVERY = 1.9

#: nM. MEASURED half-block concentration of mecamylamine at human alpha7:
#: [ChavezNoriega1997] reports that 3 uM mecamylamine inhibited h-alpha7
#: (and h-alpha4beta2) "by approximately 50%". A single-concentration
#: readout rather than a fitted IC50, and the only measurement reached that
#: puts the two subtypes in one experiment. [Papke2013] did not test alpha7
#: and states mecamylamine is "least potent for alpha7"; the value here is
#: 6-fold MORE potent than this module's alpha4beta2 number, which is the
#: opposite ordering, and that disagreement is recorded rather than
#: resolved -- the two come from different laboratories, different subunit
#: species and different stoichiometries.
ALPHA7_IC50_MECAMYLAMINE = 3000.0

#: The same single-concentration readout at h-alpha4beta2 in
#: [ChavezNoriega1997]: also ~50% inhibition at 3 uM. Kept so the
#: disagreement with [Papke2013]'s 18 uM is a number rather than a remark.
ALPHA4BETA2_IC50_MECAMYLAMINE_CHAVEZ = 3000.0


# ---------------------------------------------------------------------------
# The other three nicotinic rows of [Fenster1997] Table 1
# ---------------------------------------------------------------------------
#
# Not instantiated. They are here because one experiment measuring five
# subunit combinations is what turns "alpha7 desensitizes faster than
# alpha4beta2" from a quotation into a comparison, and because the spread
# across them is the scale against which this module's two sites should be
# read. A test asserts that spread.

#: min. MEASURED [Fenster1997] Table 1, calcium-containing medium:
#: (tau_desensitization, tau_recovery) for the three subtypes this module
#: does not build.
FENSTER1997_OTHER_SUBTYPES = {
    "alpha3beta4": (32.5, 42.6),
    "alpha4beta4": (35.5, 413.8),
    "alpha3beta2": (5.0, 11.7),
}


# ---------------------------------------------------------------------------
# Muscarinic: five subtypes of constants, two sites built
# ---------------------------------------------------------------------------

#: CURATED pKi of ACETYLCHOLINE at the five HUMAN muscarinic subtypes
#: [GtoPdb2026], as (low, high) where the Guide records a range and
#: (value, value) where it records one number. Primaries per subtype are in
#: the module docstring. The 160-fold M2 range is not a transcription error:
#: agonist binding to a muscarinic receptor is biphasic and
#: guanine-nucleotide-sensitive, so a curated range over several
#: laboratories brackets the G-protein-coupled high-affinity state and the
#: uncoupled low-affinity one at once. This is the same problem P5.3 met at
#: alpha2 and, unlike there, no single-laboratory pair of state-resolved
#: values for acetylcholine was reached, so this module carries ONE
#: midpoint per subtype and states the range instead of offering two
#: selectable sets.
MUSCARINIC_PKI_ACETYLCHOLINE = {
    "M1": (4.3, 4.9),
    "M2": (4.3, 6.5),
    "M3": (4.5, 5.4),
    "M4": (4.5, 5.6),
    "M5": (6.1, 6.1),
}

#: CURATED pKi of acetylcholine at RAT M1, M2 and M3 [GtoPdb2026]. Recorded
#: because all three sit inside the corresponding human ranges above, which
#: is the only species check available here.
MUSCARINIC_PKI_ACETYLCHOLINE_RAT = {"M1": 5.0, "M2": 6.4, "M3": 5.6}

#: CURATED pKi of SCOPOLAMINE at the five human muscarinic subtypes
#: [GtoPdb2026], attributed to [Bolden1992] (all five), [Huang2001] (M1-M4)
#: and [Croy2016] (M1). Same (low, high) convention.
MUSCARINIC_PKI_SCOPOLAMINE = {
    "M1": (9.0, 9.0),
    "M2": (8.7, 8.7),
    "M3": (9.4, 9.4),
    "M4": (9.1, 9.5),
    "M5": (8.7, 8.7),
}

MUSCARINIC_SUBTYPES = ("M1", "M2", "M3", "M4", "M5")


def muscarinic_kd(subtype, ligand):
    """nM Kd for `ligand` at muscarinic `subtype`, from the tables above.

    Where the Guide records a RANGE, the value returned is the midpoint in
    pKi, i.e. the geometric mean of the two Kd endpoints. That midpoint is a
    DERIVED number and is labelled as one everywhere it is used; both
    endpoints stay available in `MUSCARINIC_PKI_*` so the width of what is
    being collapsed is never out of sight.
    """
    table = {
        "acetylcholine": MUSCARINIC_PKI_ACETYLCHOLINE,
        "scopolamine": MUSCARINIC_PKI_SCOPOLAMINE,
    }[ligand]
    if subtype not in table:
        raise ValueError("unknown muscarinic subtype %r" % (subtype,))
    low, high = table[subtype]
    return _kd_from_pki(0.5 * (low + high))


# ---------------------------------------------------------------------------
# Efficacies
# ---------------------------------------------------------------------------

#: Acetylcholine's efficacy, 1.0, at every site. This is a DEFINITION here
#: rather than a measurement, and a defensible one: acetylcholine is the
#: endogenous transmitter and the reference full agonist that every panel
#: cited above normalizes to ([GtoPdb2026] labels it a full agonist at all
#: five muscarinic subtypes; [ChavezNoriega1997] finds it "one of the most
#: efficacious agonists at all hnAChRs" except h-alpha3beta2, which this
#: module does not build).
ACETYLCHOLINE_EFFICACY = 1.0

#: Nicotine's efficacy at both nicotinic sites. 1.0, and this is a STATED
#: SIMPLIFICATION rather than a measurement: no relative maximal response
#: for nicotine against acetylcholine at alpha4beta2 or alpha7 in one
#: preparation was reached (see "Searched and empty"). Nothing in this
#: module's results depends on it -- both nicotinic results are about
#: desensitization, which `DesensitizingReceptorSite` drives from occupancy
#: and not from efficacy -- and a test asserts that by re-running them at
#: 0.5.
NICOTINE_EFFICACY = 1.0

#: Scopolamine's efficacy. 0.0 is the fourth instance of the same MODELLING
#: SIMPLIFICATION this codebase has flagged for ketanserin (P5.1),
#: SCH-23390 (P5.2) and prazosin and propranolol (P5.3): scopolamine is an
#: inverse agonist at constitutively active muscarinic receptors, and
#: `ReceptorSite.activation` has no constitutive term for a negative
#: efficacy to act against, so a negative value would drive the node BELOW
#: its drug-free baseline, which is not what an inverse agonist does to a
#: receptor with no constitutive tone.
SCOPOLAMINE_EFFICACY = 0.0

#: Mecamylamine's efficacy is not used at all: it is a channel blocker, it
#: never enters `activation`'s weighted sum, and this zero exists only so
#: the `Ligand` constructor has a value. Its pharmacology is entirely in
#: the multiplicative block term.
MECAMYLAMINE_EFFICACY = 0.0


# ---------------------------------------------------------------------------
# Ligands and sites
# ---------------------------------------------------------------------------


def _ligand(name, kd, efficacy, kon=ILLUSTRATIVE_KON):
    """Derivation 3 with an illustrative kon: `koff = kon * Kd`, so the
    measured affinity is exact and only the timescale is hand-set.

    The arithmetic and the assumption behind it are `receptor.ligand_from_ki`
    (P8.1), which is where the four modules that each wrote this line by hand
    now share one statement of it."""
    return ligand_from_ki(name=name, ki_nM=kd, kon=kon, efficacy=efficacy)


def ligands_alpha4beta2(nicotine_efficacy=NICOTINE_EFFICACY):
    """(orthosteric, blockers) for alpha4beta2.

    Orthosteric: acetylcholine, nicotine, in that order. Blocker:
    mecamylamine. The engine sees them concatenated, so the site's ligand
    and dose order is acetylcholine / nicotine / mecamylamine.
    """
    orthosteric = [
        _ligand("acetylcholine", ALPHA4BETA2_KD_ACETYLCHOLINE, ACETYLCHOLINE_EFFICACY),
        _ligand("nicotine", ALPHA4BETA2_KD_NICOTINE, nicotine_efficacy),
    ]
    blockers = [
        _ligand(
            "mecamylamine", ALPHA4BETA2_IC50_MECAMYLAMINE, MECAMYLAMINE_EFFICACY
        )
    ]
    return orthosteric, blockers


def ligands_alpha7(nicotine_efficacy=NICOTINE_EFFICACY):
    """(orthosteric, blockers) for alpha7, same order as alpha4beta2."""
    orthosteric = [
        _ligand("acetylcholine", ALPHA7_KD_ACETYLCHOLINE, ACETYLCHOLINE_EFFICACY),
        _ligand("nicotine", ALPHA7_KD_NICOTINE, nicotine_efficacy),
    ]
    blockers = [
        _ligand("mecamylamine", ALPHA7_IC50_MECAMYLAMINE, MECAMYLAMINE_EFFICACY)
    ]
    return orthosteric, blockers


def receptor_site_alpha4beta2(nicotine_efficacy=NICOTINE_EFFICACY):
    """The high-sensitivity (alpha4)2(beta2)3 pool, on the cortical
    pyramidal node ([Kassam2008]).

    Both agonists get the SAME desensitization and recovery rates, because
    [Fenster1997] measured them with nicotine and no acetylcholine
    equivalent was found. That is derivation 3's assumption moved from a
    binding rate to a gating rate: it says the conformational change is a
    property of the receptor rather than of which agonist opened it, which
    is what the Katz-Thesleff scheme asserts and what
    [PapkePapke2002]'s ACh-driven desensitization at alpha7 is consistent
    with, but it is not separately measured here.
    """
    orthosteric, blockers = ligands_alpha4beta2(nicotine_efficacy=nicotine_efficacy)
    return DesensitizingReceptorSite(
        name="alpha4beta2_on_cortical_pyramidal",
        ligands=orthosteric,
        k_des=_rate_from_time_constant(ALPHA4BETA2_TAU_DESENSITIZATION),
        k_rec=_rate_from_time_constant(ALPHA4BETA2_TAU_RECOVERY),
        channel_blockers=blockers,
    )


def receptor_site_alpha7(nicotine_efficacy=NICOTINE_EFFICACY):
    """The homomeric alpha7 pool, on the same cortical pyramidal node
    ([Alkondon1993]), with the same two agonists and the same blocker and
    nothing in common with alpha4beta2 except that.

    Putting both nicotinic sites on ONE node is deliberate: it makes every
    alpha7-versus-alpha4beta2 difference this module reports a consequence
    of the published constants rather than of two hand-set gains, the same
    device P5.3 used for its two cortical adrenoceptors.
    """
    orthosteric, blockers = ligands_alpha7(nicotine_efficacy=nicotine_efficacy)
    return DesensitizingReceptorSite(
        name="alpha7_on_cortical_pyramidal",
        ligands=orthosteric,
        k_des=_rate_from_time_constant(ALPHA7_TAU_DESENSITIZATION),
        k_rec=_rate_from_time_constant(ALPHA7_TAU_RECOVERY),
        channel_blockers=blockers,
    )


def ligands_muscarinic(subtype):
    """acetylcholine / scopolamine at one muscarinic subtype, in that order.

    Ordinary competitive ligands at one pool: scopolamine really does
    compete with acetylcholine for the orthosteric site, so this needs
    nothing `ReceptorSite` did not already do -- the same finding P5.1
    recorded for diazepam/flumazenil at the benzodiazepine site.
    """
    return [
        _ligand(
            "acetylcholine",
            muscarinic_kd(subtype, "acetylcholine"),
            ACETYLCHOLINE_EFFICACY,
        ),
        _ligand(
            "scopolamine", muscarinic_kd(subtype, "scopolamine"), SCOPOLAMINE_EFFICACY
        ),
    ]


def receptor_site_m1():
    """One competitive M1 pool, on the cortical pyramidal node.

    NOT a `DesensitizingReceptorSite`, on purpose: see "Why the two-state
    occupancy model was not enough" in the module docstring.
    """
    return ReceptorSite(name="M1_on_cortical_pyramidal", ligands=ligands_muscarinic("M1"))


def receptor_site_m2():
    """One competitive M2 pool: the basal-forebrain cholinergic AUTORECEPTOR
    ([Zhang2002])."""
    return ReceptorSite(
        name="M2_autoreceptor_on_basal_forebrain", ligands=ligands_muscarinic("M2")
    )


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, shared by nicotine, scopolamine and
#: mecamylamine, and deliberately the same number as
#: `serotonin.TOOL_COMPOUND_KE`, `dopamine.TOOL_COMPOUND_KE` and
#: `noradrenaline.TOOL_COMPOUND_KE`; a test asserts the equality so they
#: cannot drift apart. Do not cite it as a half-life for any of the three.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY, the same value and the same reason as
#: `dopamine.DOPAMINE_KE`, `serotonin.SEROTONIN_KE` and
#: `noradrenaline.NORADRENALINE_KE`, and the most extreme case of it in the
#: codebase: extracellular acetylcholine is hydrolysed by
#: acetylcholinesterase on a sub-millisecond timescale, not eliminated by a
#: first-order process on a scale of seconds. 5.0 min^-1 (t1/2 ~8.3 s)
#: preserves the only property the dynamics need -- that the transmitter
#: clears far faster than any dosed drug -- at a timescale the integrator
#: resolves. The real quantity belongs to P6.2.
ACETYLCHOLINE_KE = 5.0


def pk_models_alpha4beta2():
    """One PK model per alpha4beta2 ligand, in site ligand order
    (acetylcholine, nicotine, mecamylamine)."""
    return [
        OneCompartmentPK(ke=ACETYLCHOLINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


def pk_models_alpha7():
    """One PK model per alpha7 ligand, same order."""
    return pk_models_alpha4beta2()


def pk_models_muscarinic():
    """One PK model per muscarinic ligand (acetylcholine, scopolamine)."""
    return [
        OneCompartmentPK(ke=ACETYLCHOLINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: ILLUSTRATIVE throughout, built to P5.3's three stated rules
# ---------------------------------------------------------------------------
#
# Three nodes, and one edge that is missing on purpose:
#
#   BF_Cholinergic   basal forebrain cholinergic neurons (nucleus basalis /
#                    medial septum). Carries the M2 AUTORECEPTOR
#                    ([Zhang2002]). NO edges, for P5.1's and P5.3's reason.
#   PFC_Pyramidal    layer V/VI prefrontal pyramidal neurons. Carries
#                    THREE acetylcholine receptors at once -- M1 (Gq,
#                    excitatory, [McCormick1986]), alpha4beta2 (cation
#                    channel, excitatory, [Kassam2008]) and alpha7 (cation
#                    channel, excitatory, [Alkondon1993]) -- which is what
#                    is new about this network relative to P5.3's two.
#   PFC_GABA         local cortical GABAergic interneurons. Carries no
#                    receptor here, and exists to give the pyramidal node
#                    the recurrent inhibition that makes its response
#                    finite.
#
# The three rules, unchanged from P5.3 because the circuit being
# represented is the same cortical microcircuit:
#
#   1. Baselines put every node at rate 0.5 at rest with no drug, the
#      maximum-sensitivity point `stability.MAX_SENSITIVITY_RATE` derives.
#      With the loop weights below that fixes baseline_input at +0.45 for
#      the pyramidal node and -0.4 for the interneuron exactly, and
#      `stability.balanced_baselines` reproduces them.
#   2. The loop weights satisfy P3.4's global-uniqueness certificate,
#      contraction_number = 0.9 < 1, so the network has exactly one fixed
#      point. The weights are P5.1's and P5.3's, unchanged.
#   3. No node saturates under a saturating dose of a full agonist at every
#      receptor the node carries. The unit budget is per NODE, as P5.3
#      established: `PFC_Pyramidal` now carries THREE excitatory receptors,
#      so each gets 1/3 of the one-unit budget, and the basal-forebrain
#      node's single receptor keeps magnitude 1.0. Equal thirds are also
#      what make an alpha7-versus-alpha4beta2-versus-M1 comparison a
#      statement about published constants rather than about three hand-set
#      gains.

#: ILLUSTRATIVE. Unchanged from `serotonin.raphe_cortical_network` and
#: `noradrenaline.locus_coeruleus_cortical_network`; see rule 2.
CORTICAL_INHIBITION_WEIGHT = -0.9
CORTICAL_RECURRENT_WEIGHT = 0.8

#: ILLUSTRATIVE, and the same value every hand-built scenario uses.
BF_CORTEX_GAIN = 4.0

#: The node parameters. `baseline_input` is fully determined by rule 1.
_NODES = (
    # name,            tau,  baseline_input
    ("BF_Cholinergic", 1.0, 0.0),
    ("PFC_Pyramidal", 1.5, 0.45),
    ("PFC_GABA", 1.0, -0.4),
)


def basal_forebrain_cortical_network():
    """BF_Cholinergic (isolated) + PFC_Pyramidal -| PFC_GABA cortical loop."""
    nodes = [
        CircuitNode(name=name, tau=tau, baseline_input=baseline, gain=BF_CORTEX_GAIN)
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

    P5.2's and P5.3's habit, and needed here for the same reason and one
    more: `ACETYLCHOLINE_KE` clears the transmitter in seconds, so a
    cortical node with tau 1.5 min never reaches the rate its peak occupancy
    would support, and a start-up transient would swamp what is left.
    """
    n = len(_NODES) if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Coupling: four mechanism gains, three of them sharing one node
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE. One unit of drive per NODE, split equally between the three
#: receptors the cortical pyramidal node carries (rule 3). POSITIVE for all
#: three: M1 is Gq-coupled and acetylcholine depolarizes cortical pyramidal
#: neurons through it by reducing a potassium conductance
#: ([McCormick1986]); alpha4beta2 and alpha7 are cation channels whose
#: opening depolarizes the same cell ([Kassam2008], [Alkondon1993]).
#: Not identifiable from anything published (P2.3), and the
#: `mechanism_gain` half of P4.2's `coupling = mechanism_gain * rho`.
CORTICAL_MECHANISM_GAIN = 1.0 / 3.0

#: ILLUSTRATIVE. The M2 autoreceptor gain on the basal forebrain node.
#: NEGATIVE: M2 is Gi/o-coupled and is the inhibitory autoreceptor that
#: suppresses acetylcholine release, identified in M2 knock-out mice
#: ([Zhang2002]). Magnitude 1.0 by rule 3's per-node budget, and NOT a
#: claim that the autoreceptor is three times as powerful as any cortical
#: receptor.
M2_MECHANISM_GAIN = -1.0


def coupling_cortical():
    """Any of the three cortical receptors' activation -> pyramidal drive."""
    return {"PFC_Pyramidal": CORTICAL_MECHANISM_GAIN}


def coupling_m2():
    """M2 activation -> drive, on the basal forebrain autoreceptor node."""
    return {"BF_Cholinergic": M2_MECHANISM_GAIN}


def receptor_system_alpha4beta2(nicotine_efficacy=NICOTINE_EFFICACY):
    return ReceptorSystem(
        pk_models=pk_models_alpha4beta2(),
        receptor_site=receptor_site_alpha4beta2(nicotine_efficacy=nicotine_efficacy),
        coupling=coupling_cortical(),
        name="alpha4beta2",
    )


def receptor_system_alpha7(nicotine_efficacy=NICOTINE_EFFICACY):
    return ReceptorSystem(
        pk_models=pk_models_alpha7(),
        receptor_site=receptor_site_alpha7(nicotine_efficacy=nicotine_efficacy),
        coupling=coupling_cortical(),
        name="alpha7",
    )


def receptor_system_m1():
    return ReceptorSystem(
        pk_models=pk_models_muscarinic(),
        receptor_site=receptor_site_m1(),
        coupling=coupling_cortical(),
        name="M1",
    )


def receptor_system_m2():
    return ReceptorSystem(
        pk_models=pk_models_muscarinic(),
        receptor_site=receptor_site_m2(),
        coupling=coupling_m2(),
        name="M2",
    )


def cholinergic_receptor_systems(nicotine_efficacy=NICOTINE_EFFICACY):
    """All four systems, alpha4beta2 / alpha7 / M1 / M2, for `Simulation`.

    Acetylcholine appears once per system, as four separate `Ligand`
    objects with four separate concentration states, exactly as
    noradrenaline does across P5.3's three systems: they are four distinct
    receptor pools and the engine tracks state per pool. One rise in
    acetylcholine is applied to all four by
    `Simulation.dose_vector(acetylcholine=...)`, which matches ligands by
    name across every system -- which is why all four systems must give
    acetylcholine the SAME `ke`, or the model would describe one molecule
    clearing at four different rates. A test asserts that.
    """
    return [
        receptor_system_alpha4beta2(nicotine_efficacy=nicotine_efficacy),
        receptor_system_alpha7(nicotine_efficacy=nicotine_efficacy),
        receptor_system_m1(),
        receptor_system_m2(),
    ]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def cholinergic_tone_readout():
    """Basal-forebrain cholinergic-neuron firing.

    Falls under a muscarinic agonist: that is the autoreceptor. The same
    shape as `serotonin.serotonergic_tone_readout` and
    `noradrenaline.noradrenergic_tone_readout`, and like them a
    circuit-level quantity rather than a fitted behavioural endpoint.
    """
    return BehaviorReadout(name="cholinergic_tone", weights={"BF_Cholinergic": 1.0})


def cortical_output_readout():
    """Layer V/VI prefrontal pyramidal firing: the node all three cortical
    acetylcholine receptors share.

    Deliberately the same name, node and weight as
    `serotonin.cortical_output_readout` and
    `noradrenaline.cortical_output_readout`; a test asserts all three are
    equal, so a combined monoamine/acetylcholine scenario would score them
    through one readout rather than three lookalikes.
    """
    return BehaviorReadout(name="cortical_output", weights={"PFC_Pyramidal": 1.0})
