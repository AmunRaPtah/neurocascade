"""P5.2: the dopamine D1 receptor and the direct basal-ganglia pathway.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter below): every D1
binding constant and every D1 efficacy. Three ligands at one D1 site --
dopamine, dihydrexidine and SCH-23390. The D2 half of this module is not
re-derived: it reuses `scenarios.d2_receptor_site_calibrated()` and
`scenarios.d2_pk_models_calibrated()` unchanged, so the D1-vs-D2
comparison below is a comparison of two published parameter sets and not
of two parameter sets fitted to make it come out.

ILLUSTRATIVE (hand-set, labelled, no literature target): the seven-node
cortico-basal-ganglia-thalamic circuit's weights, taus, gains and baseline
inputs; the two mechanism gains that turn receptor activation into node
drive; and the pharmacokinetics of dihydrexidine and SCH-23390, for which
no rodent PK was found (see "Searched and empty"). This is the same split
as every scenario before Phase 7: this task calibrates the receptor layer,
not the circuit layer.

Why D1 and D2 have to share one network
---------------------------------------

The striatum's two projection-neuron populations are not two circuits. They
are two populations of the same nucleus, receiving the same corticostriatal
afferents, releasing GABA onto two different targets, and converging again
on the same basal-ganglia output nucleus:

  * **direct / striatonigral MSNs** carry D1 ([Gerfen1990]) and project
    monosynaptically to the output nucleus (GPi/SNr), inhibiting it;
  * **indirect / striatopallidal MSNs** carry D2 ([Gerfen1990]) and project
    to GPe, from which the chain runs GPe -| STN -> GPi.

A drug that binds both receptors therefore acts on ONE structure through
two arms with opposite receptor-level signs (D1 is Gs/olf-coupled and
depolarizing on the cell that carries it; D2 is Gi/o-coupled and
hyperpolarizing on the cell that carries it), and the arms reconverge two
synapses later. Simulating that as two separate models would get the
receptor layer right and the convergence wrong: the second arm would never
see what the first one did to the shared output nucleus. So this module
builds ONE `CircuitNetwork` carrying both pathways and puts two
`ReceptorSystem`s on it, which is what
`Simulation(network, receptor_systems=[...])` already supports without any
engine change (same as P5.1's two serotonin systems on one cortical node).

`scenarios.basal_ganglia_indirect_pathway_network()` is left exactly as it
was; this is an additional, wider network, not a replacement, and the D2
receptor site and PK it uses are the same objects.

Sources
-------

D1 binding and efficacy:

  [Sahlholm2025] Sahlholm K, Svensson P, Malo M, Andersson DR, Betari N.
                "Estimation of Dopamine D1 Receptor Agonist Binding
                Kinetics Using Time-Resolved Functional Assays: Relation to
                Agonist-Induced Receptor Internalization by Investigational
                Antiparkinsonian Therapeutics." ACS Chem Neurosci.
                2025;16(13):2502-2512. doi:10.1021/acschemneuro.5c00270
                (open access, PMC12232300). HUMAN D1 receptor co-expressed
                with GIRK1/4 in Xenopus laevis oocytes, two-electrode
                voltage clamp at -80 mV, 22 C; GIRK current activation and
                deactivation on agonist application and washout are used to
                estimate kon and koff. Table 1, read from the article's
                full-text XML:

                  ligand         pEC50 (EC50 nM)  efficacy   koff (s^-1)   kon (M^-1 s^-1)   pKd
                  dopamine       6.207 (622)      1.022      0.132+-0.010  122325+-37072     5.969+-0.090
                  A77636         7.382 (41.5)     1.173      0.025+-0.004  903422+-78561     7.556+-0.028
                  dihydrexidine  7.361 (43.5)     0.808      0.095+-0.005  952419+-174431    7.002+-0.054
                  apomorphine    5.816 (1527)     0.133      0.090+-0.016  6910+-8354        4.883+-0.309
                  tavapadon      6.864 (137)      0.106      0.027+-0.008  41157+-28432      6.179+-0.149

                "efficacy" is Emax relative to 30 uM dopamine in the same
                oocyte, which is why dopamine's own entry is 1.022 rather
                than exactly 1. Two of the five (dopamine, dihydrexidine)
                are instantiated below; the other three are recorded here
                because they are what makes this a panel rather than a
                single measurement.
  [Agren2021]   Agren R, Stepniewski TM, Zeberg H, Selent J, Sahlholm K.
                "Dopamine D2 Receptor Agonist Binding Kinetics-Role of a
                Conserved Serine Residue." Int J Mol Sci. 2021;22(8):4078.
                doi:10.3390/ijms22084078 (open access, PMC8071183). The D2
                counterpart of [Sahlholm2025], already cited and already
                used by `scenarios.d2_dopamine_ligands_calibrated()`: human
                D2 with GIRK1/4 AND RGS4 in Xenopus oocytes, same
                laboratory, same GIRK-kinetics method (the D1 paper says in
                as many words that its approach is "analogous to that
                earlier described for dopamine D2R agonists" and cites this
                paper for it). Dopamine at D2: koff 0.197 +/- 0.012 s^-1,
                kon (9.70 +/- 1.23) x 10^7 M^-1 s^-1, pKd 8.69 +/- 0.14,
                EC50 20 nM. This pairing is what lets the D1/D2 affinity
                comparison below be a comparison rather than an
                apples-to-oranges quote (see "What the numbers turned out
                to say", and the RGS4 asymmetry recorded there).
  [Billard1984] Billard W, Ruperto V, Crosby G, Iorio LC, Barnett A.
                "Characterization of the binding of 3H-SCH 23390, a
                selective D-1 receptor antagonist ligand, in rat striatum."
                Life Sci. 1984;35(18):1885-1893.
                doi:10.1016/0024-3205(84)90540-x. PMID:6387355. RAT
                striatal membranes: saturable, stereospecific, one-site
                binding of [3H]SCH 23390, Scatchard KD = 0.53 nM, Bmax
                69 pmol/g wet weight. The equilibrium affinity SCH-23390's
                rate constants are derived from below.
  [Reader1989]  Reader TA, Grondin L, Montreuil B, Dewar KM. "Dopamine D1
                receptors labelled with [3H]SCH23390 in rabbit cerebral
                cortex and neostriatum. Equilibrium binding, kinetics and
                selectivity." Naunyn Schmiedebergs Arch Pharmacol.
                1989;340(6):617-625. doi:10.1007/bf00717736. PMID:2575710.
                RABBIT cortex and caudate-putamen membranes: association of
                [3H]SCH23390 rapid, "the dissociation kinetics were
                extremely slow and only around 40-60% of the binding was
                displaced two hours after the addition of either
                S(+)-butaclamol or 30 volumes of buffer"; saturation
                Kd(25 C) 0.31 nM (cortex) and 0.13 nM (caudate-putamen).
                Used ONLY as an independent qualitative bracket on the
                derived SCH-23390 koff (see "Derivations"); no number is
                taken from it, because it is a third species.
  [Reader1988]  Reader TA, Briere R, Gottberg E, Diop L, Grondin L.
                "Specific [3H]SCH23390 binding to dopamine D1 receptors in
                cerebral cortex and neostriatum: evidence for
                heterogeneities in affinity states and cortical
                distribution." J Neurochem. 1988;50(2):451-463.
                doi:10.1111/j.1471-4159.1988.tb02932.x. Cited for the
                finding that [3H]SCH23390 dissociation kinetics and
                competition curves both indicate high- and low-affinity
                states, i.e. the same two-state problem P5.1 met at 5-HT2A.
  [Bourne2001]  Bourne JA. "SCH 23390: the first selective dopamine
                D1-like receptor antagonist." CNS Drug Rev.
                2001;7(4):399-414.
                doi:10.1111/j.1527-3458.2001.tb00207.x (open access,
                PMC6741643). A REVIEW, quoted here only for the widely
                repeated Ki = 0.2 nM (D1) and 0.3 nM (D5) figures, which
                are the same order as [Billard1984]'s 0.53 nM and are NOT
                used as this module's number: the review states them
                without attributing them to a primary experiment. It is
                also the source of the caveat that SCH-23390 binds 5-HT2
                and 5-HT2C with high affinity in vitro, which this
                single-site model does not represent.

Anatomy and mechanism (no binding numbers taken from these; they justify
which node carries which receptor, with what sign, and what the pathways
do):

  [Albin1989]   Albin RL, Young AB, Penney JB. "The functional anatomy of
                basal ganglia disorders." Trends Neurosci.
                1989;12(10):366-375. doi:10.1016/0166-2236(89)90074-x.
                PMID:2479133. The direct/indirect model this circuit draws.
  [DeLong1990]  DeLong MR. "Primate models of movement disorders of basal
                ganglia origin." Trends Neurosci. 1990;13(7):281-285.
                doi:10.1016/0166-2236(90)90110-v. PMID:1695404. The other
                half of the same model.
  [Gerfen1990]  Gerfen CR, Engber TM, Mahan LC, Susel Z, Chase TN, Monsma
                FJ, Sibley DR. "D1 and D2 dopamine receptor-regulated gene
                expression of striatonigral and striatopallidal neurons."
                Science. 1990;250(4986):1429-1432.
                doi:10.1126/science.2147780. PMID:2147780. The segregation
                this module's two coupling dicts encode: D1 mRNA with
                substance P in striatoNIGRAL neurons, D2 mRNA with
                enkephalin in striatoPALLIDAL neurons, each responding to
                its own agonist after 6-OHDA lesion.
  [Kravitz2010] Kravitz AV, Freeze BS, Parker PR, Kay K, Thwin MT,
                Deisseroth K, Kreitzer AC. "Regulation of parkinsonian
                motor behaviours by optogenetic control of basal ganglia
                circuitry." Nature. 2010;466(7306):622-626.
                doi:10.1038/nature09159. PMID:20613723. The empirical test
                of the direct/indirect model, in D1-Cre and D2-Cre mice:
                bilateral excitation of INDIRECT-pathway MSNs produced
                increased freezing, bradykinesia and fewer locomotor
                initiations; excitation of DIRECT-pathway MSNs reduced
                freezing and increased locomotion. This is the paper the
                opposing-pathway test in
                `tests/test_dopamine_d1_scenario.py` is written against; it
                is a statement about PATHWAYS, not about drugs, and the
                test checks it that way (drive each MSN node directly, read
                the locomotor readout).
  [HernandezLopez1997] Hernandez-Lopez S, Bargas J, Surmeier DJ, Reyes A,
                Galarraga E. "D1 receptor activation enhances evoked
                discharge in neostriatal medium spiny neurons by modulating
                an L-type Ca2+ conductance." J Neurosci.
                1997;17(9):3334-3342.
                doi:10.1523/jneurosci.17-09-03334.1997. PMID:9096166. The
                source of D1's POSITIVE sign here, and of the caveat
                attached to it: D1 agonists inhibit evoked discharge at
                hyperpolarized potentials (~-80 mV) and ENHANCE it at
                depolarized ones (~-55 mV), via an L-type Ca2+ current. The
                sign this module uses is the depolarized-state one, which
                is the state a corticostriatally driven MSN is in; see
                "What the positive D1 gain does and does not claim".
  [Galarraga1997] Galarraga E, Hernandez-Lopez S, Reyes A, Barral J,
                Bargas J. "Dopamine facilitates striatal EPSPs through an
                L-type Ca2+ conductance." Neuroreport.
                1997;8(9):2183-2186. doi:10.1097/00001756-199707070-00019.
                PMID:9243608. The same effect seen on the synaptic
                response rather than on evoked discharge.
  [Garris1994]  Garris PA, Ciolkowski EL, Pastore P, Wightman RM. "Efflux
                of dopamine from the synaptic cleft in the nucleus
                accumbens of the rat brain." J Neurosci.
                1994;14(10):6084-6093. PMID:7931564. Already cited by
                `scenarios.d2_pk_models_calibrated` for why dopamine gets
                no literal `ke`; cited again here for the ~0.25 uM
                extracellular transient a single stimulus pulse produces,
                which is the concentration scale at which the D1/D2
                affinity split below actually matters.

Derivations
-----------

Same three-way split P5.1 established, and each is stated at its ligand:

1. **Both constants measured** -- use them. Dopamine and dihydrexidine at
   D1 ([Sahlholm2025]).
2. **koff and an equilibrium affinity measured, kon missing** --
   `kon = koff / Kd`. Not needed here.
3. **Only an equilibrium affinity measured** -- share the site's one
   measured kon and set `koff = kon * Kd`. Used for SCH-23390, with
   dopamine's measured D1 `kon` and [Billard1984]'s saturation Kd.

Derivation 3 gives SCH-23390 `koff = 0.0073395 * 0.53 = 0.00389 min^-1`, a
dissociation half-life of 178 min. That is a strong claim to get out of an
assumption, so it is checked against the one independent kinetic
observation that exists: [Reader1989] reports 40-60% of [3H]SCH23390
binding displaced two hours after initiating dissociation, which brackets
the half-life at 91-163 min. The derived 178 min is ~9% beyond the slow end
of that bracket -- close enough that the assumption is not contradicted,
far enough that it is not a match. The bracket is rabbit and the derivation
is human-kon/rat-Kd, so it is a sanity check and is written as one in the
tests, not as a calibration.

The mixed-format caveat is the same one the D2 section already raises for
dopamine-versus-haloperidol: dopamine's and dihydrexidine's constants come
from a functional GIRK assay on human D1 in oocytes, SCH-23390's affinity
from a [3H]SCH 23390 saturation curve on rat striatal membranes. Nothing
in this module's qualitative results turns on SCH-23390's exact affinity
(it is 190-fold tighter than dihydrexidine under any of the published
numbers, 0.2, 0.31, 0.13 or 0.53 nM), which is why one number was chosen
and flagged rather than a second parameter set built as at 5-HT2A.

What the numbers turned out to say
----------------------------------

**Dopamine's affinity for D2 is ~530x higher than for D1, measured in two
matched assays from one laboratory.** Kd 2.03 nM at D2 ([Agren2021]) versus
1079 nM at D1 ([Sahlholm2025]), both from GIRK-current kinetics in Xenopus
oocytes expressing the human receptor, both as `koff / kon`. This is the
quantitative content of the textbook "D2 is the high-affinity receptor,
D1 the low-affinity one, so tonic dopamine engages D2 and phasic dopamine
is needed for D1", and this module gets it for free from two published
numbers rather than by choosing anything. Against [Garris1994]'s ~0.25 uM
extracellular transient it says the phasic peak sits near D1's Kd and
three orders of magnitude above D2's, which is the regime in which the two
arms of the circuit can be driven differentially at all.

Two honesty notes on that ratio, both of which are recorded because they
change how much weight it can carry:

  * The same two papers' EC50s differ by only 31-fold (622 nM at D1, 20 nM
    at D2), not 530-fold. The gap between the Kd ratio and the EC50 ratio
    is receptor reserve and signal amplification, and this model's
    `activation = sum(occupancy * efficacy)` is linear in occupancy with
    no amplification term, so it reproduces the 530-fold OCCUPANCY
    separation and not the 31-fold POTENCY separation. Which of the two a
    behavioural endpoint should be fitted against is a Phase 7 question,
    and it is flagged here rather than settled.
  * [Agren2021] co-expressed RGS4 precisely so that the GIRK current
    "more closely follows the time course of D2R agonist occupancy";
    [Sahlholm2025] did not (RGS4 acts on Galpha_i/o, and D1 reaches GIRK
    through Gs). So the D1 koff is the more likely of the two to be
    rate-limited by G-protein cycle steps rather than by unbinding. If it
    is, the true D1 koff is FASTER and the true D1 Kd LARGER than the
    published one, which widens the D1/D2 gap rather than narrowing it.
    The caveat therefore points away from the finding being an artefact.

**Dihydrexidine is a strong partial agonist, not a full one.** Emax 0.808
of dopamine's in the same oocyte ([Sahlholm2025]). This is the second
place in the codebase where `Ligand.efficacy` carries a fractional
literature value rather than 1.0 or 0.0 (P5.1's 8-OH-DPAT and DOI were the
first), and again the textbook shorthand -- "dihydrexidine, the full D1
agonist" -- hides the deficit.

**SCH-23390 is a very slowly dissociating antagonist**, like P5.1's
WAY-100635 and unlike this codebase's naloxone. Its derived residence time
(178 min) is ~1500x dihydrexidine's (koff 5.7 min^-1, t1/2 7.3 s), so it
reverses a D1 agonist by not letting go rather than by molar excess. That
is now the second such case in the codebase and the third distinct
antagonist-reversal mechanism (naloxone: fast-off, reverses only in molar
excess; WAY-100635 and SCH-23390: near-irreversible).

What the positive D1 gain does and does not claim
-------------------------------------------------

D1 is Gs/olf-coupled, so the receptor-level sign is not in doubt; what is
in doubt is the sign of its effect on a medium spiny neuron's FIRING RATE,
because [HernandezLopez1997] measures both signs in the same cells:
inhibition of evoked discharge at ~-80 mV, enhancement at ~-55 mV, the
switch being an L-type Ca2+ conductance that is only available from
depolarized potentials. A Wilson-Cowan node has no membrane potential, so
it cannot carry a state-dependent sign; it has to be given one.

The one used here is the depolarized-state sign (`D1_MECHANISM_GAIN > 0`),
for the reason [HernandezLopez1997] itself gives: the up-state is the state
an MSN is in when corticostriatal input is driving it, and this network's
D1 node is driven by exactly such an input (`Cortex_M1 -> Striatum_D1_MSN`)
and sits at rate 0.5 rather than at rest. It is a modelling choice made
against a published state-dependence, not a fact, and it is the single
place in this module where a sign was chosen rather than read off.

Searched and empty
------------------

* **Rodent pharmacokinetics for dihydrexidine and SCH-23390.** Searched
  Europe PMC for each compound crossed with "pharmacokinetics",
  "half-life", "plasma concentration" and "brain concentration" in rat and
  mouse. Nothing usable: dihydrexidine's hits are in-vitro pharmacology
  and reviews of tavapadon; SCH-23390's are studies using it as a tool at
  a stated mg/kg without measuring its disposition. Both therefore share
  ONE illustrative elimination rate constant, `TOOL_COMPOUND_KE`, which is
  deliberately the same number as P5.1's `serotonin.TOOL_COMPOUND_KE` and
  for the same reason: two separately invented values would assert a
  clearance ORDERING that nothing supports, where one shared value asserts
  only that both are still on board across this scenario's window. A test
  asserts the two constants are equal so they cannot drift apart.
* **Binding kinetics for SCH-23390 at D1.** Searched Europe PMC for
  "SCH 23390" crossed with "dissociation rate constant", "koff" and
  "residence time"; for D1 with "TR-FRET"/"NanoBRET"/"association rate
  constant"; and through the Rinken group's D1 fluorescence-anisotropy
  work, which reports association and dissociation half-lives for the
  FLUORESCENT ligand BodipyFL-SKF83566 (40 s and 70 s, Kd 5.2 nM;
  Allikalt A, Kopanchuk S, Rinken A, Eur J Pharmacol 2018;839:40-46) and
  not for SCH-23390 itself. Farrell CB, O'Boyle KM, Eur J Pharmacol
  1994;268(1):79-88 measures [3H]SCH 23390 dissociation from rat striatum
  and is the obvious source, but is paywalled and its abstract reports the
  effect of dopamine on k-1 without reporting k-1. Hence derivation 3.
* **A locomotor dose-response for dihydrexidine or SCH-23390 usable as a
  circuit-layer target.** Not searched exhaustively, because Track E and
  P2.2 already established the blocker for the whole class: behavioural
  pharmacology reports mg/kg by IP/SC/PO routes, and nothing in this
  module has the PK to convert that to a concentration. The
  `drug_pk`/Phase 1 bridge is where such a dataset would enter.

What was deliberately not built
-------------------------------

* **SKF-81297 / SKF-38393, the D1 agonists the locomotor literature
  actually uses.** No binding-kinetics measurement was found for either,
  so both would need derivation 3 off a Ki from a different assay format,
  where dihydrexidine arrives with kon, koff AND efficacy measured in the
  same experiment as dopamine. The task is a receptor-layer task, so the
  ligand with the better-sourced constants was preferred over the ligand
  with the better-known behaviour. Adding SKF-81297 becomes worthwhile the
  moment Phase 7 needs a compound with a published dose-response curve.
* **The D2 autoreceptor.** D2 receptors also sit presynaptically on
  nigrostriatal dopamine terminals and somatodendritically on SNc/VTA
  dopamine neurons, where an agonist REDUCES dopamine release. That is the
  mechanism behind the biphasic locomotor dose-response of D2 agonists
  (low doses suppress locomotion, higher doses increase it), and it is a
  genuine D1-versus-D2 opposition this module cannot express -- for
  exactly the reason P5.1's 5-HT1A autoreceptor could not be closed:
  `Simulation`'s ligand concentrations come from the PK layer and no
  circuit rate has a path back into them. It is the same P6.1/P6.2
  dependency, and `serotonin.py`'s
  `test_autoreceptor_is_an_open_loop_until_phase_6` already asserts the
  gap in general form, so it is not re-asserted here.
* **Endogenous dopamine tone.** With none, an antagonist given alone is
  inert, as everywhere else in this codebase. Real SCH-23390 and real
  haloperidol given alone both suppress locomotion, because real striatum
  has tonic dopamine to remove. P6.2.
* **A cortico-thalamo-cortical feedback edge.** `Thalamus -> Cortex_M1`
  exists anatomically, and adding it would turn an acyclic cascade into a
  loop whose gain nothing in reach fixes. See the circuit section below
  for what acyclicity buys and why it was kept.
* **An atlas-derived density factor.** `Drd1` is already in
  `expression/genes.py` tagged P5.2, `Drd2` is already in the model, and
  P4.3 scored both as striatum-dominant against mouse autoradiography. The
  mechanism gains below are explicitly the `mechanism_gain` half of P4.2's
  `coupling = mechanism_gain * rho[region]`, so the upgrade would be
  additive. Two things stop it, and both are already measured elsewhere in
  this codebase rather than guessed at here:

    - `rho` is a per-REGION factor, and here the two receptors sit in the
      SAME region (caudoputamen) on two different CELL TYPES. A regional
      ISH map cannot separate them, which is a real limit of the P4.2
      factorization at exactly this scenario.
    - The direct pathway's own anatomy is the terminal-field mismatch P4.3
      recorded as an expected FAILURE for `Drd1`: nigral D1 protein sits on
      the terminals of striatonigral neurons whose somata, and therefore
      whose `Drd1` transcript, are in the caudoputamen, and P4.2 measured
      the resulting CP:SNr error at 32-fold. A transcript-derived `rho` for
      the direct pathway's output would be wrong by that factor in a known
      direction.
"""

import numpy as np

from . import scenarios
from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite

# ---------------------------------------------------------------------------
# D1: measured constants, and the ligands built from them
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. MEASURED association rate constant of DOPAMINE at human
#: D1 from GIRK-current activation kinetics [Sahlholm2025]:
#: 122325 M^-1 s^-1 * 60 s/min * 1e-9 M/nM. Also the shared kon derivation
#: 3 uses for SCH-23390.
D1_KON_DOPAMINE = 0.0073395

#: min^-1. MEASURED dissociation rate constant of dopamine at human D1
#: [Sahlholm2025]: 0.132 s^-1 * 60.
D1_KOFF_DOPAMINE = 7.92

#: The same paper's independently reported equilibrium affinity for
#: dopamine at D1, pKd = 5.969 (Kd = 1074 nM). The rate constants above
#: give koff/kon = 1079 nM, so the paper's own kinetic and equilibrium
#: numbers agree to 0.5%; that agreement is the units check in the tests.
D1_PKD_DOPAMINE = 5.969

#: nM^-1 min^-1 and min^-1. MEASURED rate constants of DIHYDREXIDINE at
#: human D1 in the same assay and the same table [Sahlholm2025]:
#: 952419 M^-1 s^-1 * 60 * 1e-9, and 0.095 s^-1 * 60. Their ratio is
#: 99.7 nM against the paper's reported pKd 7.002 (99.5 nM).
D1_KON_DIHYDREXIDINE = 0.05714514
D1_KOFF_DIHYDREXIDINE = 5.7

#: Efficacy relative to dopamine = 1.0. MEASURED: dihydrexidine's Emax is
#: 0.808 +/- 0.044 of the response to 30 uM dopamine in the same oocyte
#: [Sahlholm2025]. A strong PARTIAL agonist, not the full agonist the
#: shorthand suggests.
D1_EFFICACY_DIHYDREXIDINE = 0.808

#: nM. MEASURED saturation Kd of [3H]SCH 23390 in rat striatal membranes,
#: Scatchard analysis, one-site fit [Billard1984]. The equilibrium affinity
#: derivation 3 turns into a koff below.
D1_KD_SCH23390 = 0.53

#: min^-1. DERIVED (derivation 3): `D1_KON_DOPAMINE * D1_KD_SCH23390`.
#: 0.0073395 * 0.53 = 0.003889935, a dissociation half-life of 178 min.
#: The assumption is that SCH-23390 associates at a rate comparable to
#: dopamine's at the same site, not that the rate is any particular value.
#: See the module docstring for the [Reader1989] bracket (91-163 min) this
#: is checked against.
D1_KOFF_SCH23390 = 0.003889935

#: SCH-23390's efficacy. 0.0 is the same MODELLING SIMPLIFICATION P5.1
#: flagged for ketanserin, and for the same reason: SCH-23390 is reported
#: as an inverse agonist at constitutively active D1 receptors, and
#: `ReceptorSite.activation` has no constitutive term for a negative
#: efficacy to act against, so a negative value here would drive the node
#: BELOW its drug-free baseline, which is not what an inverse agonist does
#: to a receptor with no constitutive tone. Unlike WAY-100635's measured
#: Emax of 0, this zero is an argument.
D1_EFFICACY_SCH23390 = 0.0


def d1_dopamine_ligands():
    """dopamine / dihydrexidine / SCH-23390 at D1, in that order.

    Every affinity and every efficacy here is a published measurement
    except SCH-23390's koff, which is derivation 3 off a published
    equilibrium Kd; see the module docstring.
    """
    dopamine = Ligand(
        name="dopamine",
        kon=D1_KON_DOPAMINE,  # derivation 1, measured [Sahlholm2025]
        koff=D1_KOFF_DOPAMINE,  # derivation 1, measured [Sahlholm2025]
        # The reference agonist of [Sahlholm2025]'s efficacy scale by
        # construction. The paper's own fitted value for it is 1.022,
        # because the scale is normalized to 30 uM dopamine rather than to
        # dopamine's fitted Emax; 1.0 is the definition, not a rounding.
        efficacy=1.0,
    )
    dihydrexidine = Ligand(
        name="dihydrexidine",
        kon=D1_KON_DIHYDREXIDINE,  # derivation 1, measured [Sahlholm2025]
        koff=D1_KOFF_DIHYDREXIDINE,  # derivation 1, measured [Sahlholm2025]
        efficacy=D1_EFFICACY_DIHYDREXIDINE,
    )
    sch23390 = Ligand(
        name="SCH-23390",
        # Derivation 3: the site's one measured kon, shared.
        kon=D1_KON_DOPAMINE,
        koff=D1_KOFF_SCH23390,
        efficacy=D1_EFFICACY_SCH23390,  # simplification; see the constant
    )
    return [dopamine, dihydrexidine, sch23390]


def d1_receptor_site():
    """One competitive D1 pool, on the direct-pathway (striatonigral) MSN.

    ONE pool, on one node, because [Gerfen1990] puts D1 mRNA in the
    striatonigral population and D2 mRNA in the striatopallidal one. D1 is
    also expressed in cortex and elsewhere; this site is the striatal one,
    and the network below has no other node that could carry it.
    """
    return ReceptorSite(
        name="D1_on_striatal_direct_MSN", ligands=d1_dopamine_ligands()
    )


# ---------------------------------------------------------------------------
# Pharmacokinetics: dopamine's flagged proxy, and one number for two tools
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE PROXY, imported wholesale from
#: `scenarios.d2_pk_models_calibrated`'s reasoning rather than re-argued:
#: dopamine here is a transmitter, not a dosed drug, cleared by DAT-mediated
#: reuptake, which is a saturable Michaelis-Menten transporter process on a
#: sub-second timescale ([Garris1994]) and not first-order elimination.
#: Taken literally it would collapse a dopamine "dose" into a delta function
#: inside one integration step. 5.0 min^-1 (t1/2 ~8.3 s) preserves the only
#: property the dynamics need -- transmitter clears far faster than any
#: dosed drug -- at a timescale the integrator resolves. It is the SAME
#: value the D2 calibrated set uses, and a test asserts that, because the
#: two systems are given one shared dopamine dose and would otherwise
#: describe the same molecule clearing at two different rates.
DOPAMINE_KE = 5.0

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, one elimination rate constant
#: shared by dihydrexidine and SCH-23390, for which no rodent PK was found
#: (see "Searched and empty"). Deliberately the same number as
#: `serotonin.TOOL_COMPOUND_KE`, for the same stated reason; a test asserts
#: the equality. Do not cite this as a measured half-life for either
#: compound.
TOOL_COMPOUND_KE = 0.023105


def d1_pk_models():
    """One PK model per D1 ligand, in `d1_dopamine_ligands()` order."""
    return [
        OneCompartmentPK(ke=DOPAMINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: ILLUSTRATIVE throughout, built to three stated rules
# ---------------------------------------------------------------------------
#
# Seven nodes, one shared striatum, two pathways:
#
#   Cortex_M1          motor cortex. The corticostriatal source, projecting
#                      to BOTH striatal populations with the same weight.
#                      Carries no receptor.
#   Striatum_D1_MSN    direct-pathway / striatonigral medium spiny neurons.
#                      Carries D1 [Gerfen1990].
#   Striatum_D2_MSN    indirect-pathway / striatopallidal MSNs. Carries D2
#                      [Gerfen1990]. Same node NAME as in
#                      `scenarios.basal_ganglia_indirect_pathway_network`,
#                      so the existing calibrated D2 site drops in.
#   GPe, STN           the indirect pathway's two relays.
#   GPi                the output nucleus. In a rodent SNr is the main
#                      basal-ganglia output; P3.3 checked that swapping the
#                      two changes nothing in this codebase's ported
#                      scenarios, so the primate-flavoured name is kept for
#                      continuity with the existing scenario.
#   Thalamus           the locomotor readout's node, unchanged from
#                      `scenarios.locomotor_activity_readout()`.
#
#   Cortex_M1 -> Striatum_D1_MSN -| GPi -| Thalamus            (DIRECT)
#   Cortex_M1 -> Striatum_D2_MSN -| GPe -| STN -> GPi          (INDIRECT)
#
# Every weight, tau, gain and baseline_input is ILLUSTRATIVE. Three stated
# rules fix them, so that "illustrative" does not mean "tuned until the
# tests passed":
#
#   1. **Unit weights.** Every connection has magnitude `BG_WEIGHT` = 1.0
#      and the textbook sign. At the operating point below a logistic node's
#      slope is exactly 1, so a unit weight makes the per-synapse loop gain
#      exactly 1: a cascade that neither attenuates its signal to nothing
#      over five hops nor amplifies it out of the sensitive region. Equal
#      magnitudes are also what makes the direct and indirect arms
#      comparable BY CONSTRUCTION, which is the point of this scenario --
#      any asymmetry in a D1-versus-D2 result then comes from the wiring
#      topology and the published binding constants, not from a hand-set
#      weight difference.
#   2. **Baselines put every node at rate 0.5 at rest**,
#      `stability.MAX_SENSITIVITY_RATE`. Given rule 1 that fixes all seven
#      baseline_inputs exactly and leaves nothing to choose;
#      `stability.balanced_baselines` reproduces the values below, and a
#      test asserts it. Note what falls out at the output nucleus: GPi's
#      baseline is exactly 0.0, i.e. at rest the direct and indirect inputs
#      to it cancel. That is a consequence of rules 1 and 2, not a target.
#   3. **No node saturates under a saturating dose of a full agonist at
#      either receptor.** This is what sets the mechanism gains' magnitude
#      to 1.0 rather than reusing `scenarios.d2_coupling()`'s -3.0: at
#      magnitude 3 in this network the minimum sigmoid sensitivity under a
#      saturating D1 agonist falls to 7.5e-4, well below
#      `stability.SATURATION_FLOOR` (0.01), and a railed node stops
#      responding to the other pathway -- the GPi-saturation lesson of the
#      original D2 scenario and of P3.4. At magnitude 1 the minimum
#      sensitivity is 0.199. A test asserts the floor is respected.
#
# What acyclicity buys, and why it is preferred here to P3.4's certificate.
# The connectivity matrix is nilpotent (`W**7 == 0`): the graph is a
# directed acyclic cascade. For a DAG the fixed point is unique and globally
# attracting for free, by topological order -- each node's steady rate is a
# function of its upstream nodes' rates alone, so solving in order gives one
# and only one solution, whatever the taus. P3.4's contraction certificate
# (`kappa = sigma_max(diag(g/4) W) < 1`) is SUFFICIENT, not necessary, and
# this network does not satisfy it (`kappa = 1.414`); it does not need to.
# Both facts are asserted in the tests rather than claimed here, and
# `stability.assess` is run on the network to confirm convergence,
# uniqueness across five initial conditions, local stability and
# non-saturation empirically as well.

#: ILLUSTRATIVE. The magnitude of every connection in the network; see
#: rule 1 above for why 1.0 and why they are all equal.
BG_WEIGHT = 1.0

#: The node parameters, as one table so the rules above can be checked
#: against them by eye. `baseline_input` is fully determined by rule 2.
_NODES = (
    # name,                tau,  baseline_input
    ("Cortex_M1", 1.0, 0.0),
    ("Striatum_D1_MSN", 1.0, -0.5),
    ("Striatum_D2_MSN", 1.0, -0.5),
    ("GPe", 1.0, 0.5),
    ("STN", 1.0, 0.5),
    ("GPi", 1.5, 0.0),
    ("Thalamus", 2.0, 0.5),
)

#: ILLUSTRATIVE, and the same value every hand-built scenario in this
#: codebase uses.
BG_GAIN = 4.0


def cortico_basal_ganglia_network():
    """Direct AND indirect pathways on one shared striatal network.

    Seven nodes; see the section comment above for what each one is, which
    rule fixes each number, and what acyclicity guarantees.
    """
    nodes = [
        CircuitNode(name=name, tau=tau, baseline_input=baseline, gain=BG_GAIN)
        for name, tau, baseline in _NODES
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    w = BG_WEIGHT
    # Corticostriatal, glutamatergic, onto BOTH populations equally.
    W[idx["Striatum_D1_MSN"]][idx["Cortex_M1"]] = w
    W[idx["Striatum_D2_MSN"]][idx["Cortex_M1"]] = w
    # DIRECT: striatonigral, GABAergic, monosynaptic onto the output.
    W[idx["GPi"]][idx["Striatum_D1_MSN"]] = -w
    # INDIRECT: striatopallidal -| GPe -| STN -> output.
    W[idx["GPe"]][idx["Striatum_D2_MSN"]] = -w
    W[idx["STN"]][idx["GPe"]] = -w
    W[idx["GPi"]][idx["STN"]] = w
    # Pallido-/nigro-thalamic, GABAergic.
    W[idx["Thalamus"]][idx["GPi"]] = -w
    return CircuitNetwork(nodes=nodes, connectivity=W)


def rest_rates(network=None):
    """The rate vector rule 2 puts every node at with no drug on board.

    `Simulation.run` starts from zero rates by default, which for this
    network means the first several minutes of every trace are a start-up
    transient rather than a drug effect. Passing `rates0=rest_rates()`
    starts at the fixed point instead, so a drug-free run is exactly flat
    and every deviation in a dosed run is the drug. The tests do that.
    """
    n = len(_NODES) if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Coupling: two mechanism gains, equal in magnitude and opposite in sign
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE. How much a unit of D1 activation changes the drive of the
#: direct-pathway MSN. POSITIVE: D1 is Gs/olf-coupled, and in the
#: depolarized state a corticostriatally driven MSN is in, D1 activation
#: enhances evoked discharge ([HernandezLopez1997], [Galarraga1997]). See
#: "What the positive D1 gain does and does not claim" in the module
#: docstring for the state-dependence this sign chooses between.
#:
#: Magnitude 1.0 by rule 3 above (no node saturates), which coincides with
#: P5.1's rule that a fully occupied receptor bound by a full agonist should
#: shift its node's input by one unit. The value is NOT identifiable from
#: anything published: P2.3 established that only the product of a coupling
#: gain and a connection weight is identified even when behavioural data
#: exists. It is the `mechanism_gain` half of P4.2's
#: `coupling = mechanism_gain * rho`.
D1_MECHANISM_GAIN = 1.0

#: ILLUSTRATIVE. The D2 counterpart on the indirect-pathway MSN. NEGATIVE:
#: D2 is Gi/o-coupled and hyperpolarizing on the cell that carries it, the
#: same sign `scenarios.d2_coupling()` uses.
#:
#: Equal in MAGNITUDE to `D1_MECHANISM_GAIN` by rule 1's logic: this
#: scenario exists to compare the two arms, so giving them different gains
#: would put the answer into the parameters. It differs from
#: `scenarios.d2_coupling()`'s -3.0, and that is deliberate rather than an
#: oversight: -3.0 was set against a five-node network with weight-2.0
#: connections and rails this seven-node one (rule 3). The original scenario
#: is untouched.
D2_MECHANISM_GAIN = -1.0


def d1_coupling():
    """D1 activation -> drive, on the direct-pathway MSN only."""
    return {"Striatum_D1_MSN": D1_MECHANISM_GAIN}


def d2_coupling():
    """D2 activation -> drive, on the indirect-pathway MSN only.

    Not `scenarios.d2_coupling()`: same node, same sign, different
    magnitude, for the reason recorded at `D2_MECHANISM_GAIN`.
    """
    return {"Striatum_D2_MSN": D2_MECHANISM_GAIN}


def receptor_system_d1():
    """The D1 stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=d1_pk_models(),
        receptor_site=d1_receptor_site(),
        coupling=d1_coupling(),
        name="D1",
    )


def receptor_system_d2():
    """The D2 stack as a `ReceptorSystem`, on this module's network.

    The receptor site and the PK models are `scenarios`' calibrated D2
    objects, unchanged and un-re-derived; only the coupling gain differs
    (see `D2_MECHANISM_GAIN`).
    """
    return ReceptorSystem(
        pk_models=scenarios.d2_pk_models_calibrated(),
        receptor_site=scenarios.d2_receptor_site_calibrated(),
        coupling=d2_coupling(),
        name="D2",
    )


def dopamine_receptor_systems():
    """Both systems, D1 first, ready for `Simulation(network, ...)`.

    Dopamine appears once per system, as two separate `Ligand` objects with
    two separate concentration states -- exactly the shape P5.1's serotonin
    module has, and not a duplication bug: they are two distinct receptor
    pools and the engine tracks occupancy per pool. One rise in dopamine is
    applied to both entries by `Simulation.dose_vector(dopamine=...)`,
    which matches ligands by name across every system. That is why
    `DOPAMINE_KE` here has to equal the D2 set's dopamine `ke`.
    """
    return [receptor_system_d1(), receptor_system_d2()]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def locomotor_activity_readout():
    """The SAME locomotor readout the D2-only indirect-pathway scenario uses.

    Returned through `scenarios` rather than rebuilt, so that a D1 result
    and a D2 result in this module are scored by one function on one node,
    which is what makes "opposing effects on the same readout" a claim
    about the circuit rather than about two different measurements.
    """
    return scenarios.locomotor_activity_readout()


def direct_pathway_readout():
    """Direct-pathway (striatonigral, D1-bearing) MSN firing."""
    return BehaviorReadout(
        name="direct_pathway_activity", weights={"Striatum_D1_MSN": 1.0}
    )


def indirect_pathway_readout():
    """Indirect-pathway (striatopallidal, D2-bearing) MSN firing."""
    return BehaviorReadout(
        name="indirect_pathway_activity", weights={"Striatum_D2_MSN": 1.0}
    )


def basal_ganglia_output_readout():
    """Output-nucleus (GPi/SNr) firing: where the two pathways reconverge."""
    return BehaviorReadout(name="basal_ganglia_output", weights={"GPi": 1.0})
