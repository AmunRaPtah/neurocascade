"""P5.6a: the histamine H1 receptor system.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter below): the affinities
of histamine, mepyramine and diphenhydramine at the human H1 receptor;
mepyramine's association AND dissociation rate constants (derivation 1);
diphenhydramine's dissociation rate constant, from a measured residence
time (derivation 2, the first instance of it in this codebase); and the
three efficacies.

ILLUSTRATIVE (hand-set, labelled, no literature target): the three-node
tuberomammillary/cortex circuit's weights, taus, gains and baseline inputs;
the one mechanism gain that turns H1 activation into node drive; and the
pharmacokinetics of the two dosed drugs plus the histamine clearance proxy.
As in every scenario before Phase 7, this task calibrates the receptor
layer, not the circuit layer.

Why H1 and not H1/H2/H3
-----------------------

The roadmap asks for H1 specifically, and the reason is the behaviour: an
H1 antagonist that reaches the brain is a sedative, which is why
first-generation antihistamines are sold as sleep aids and why the
second-generation ones were engineered not to cross the blood-brain
barrier. The receptor arithmetic behind that is:

  * **H1** is Gq/11-coupled and POSTSYNAPTIC on the targets of the
    tuberomammillary histaminergic projection. On a thalamic relay neuron
    histamine produces a slow depolarization associated with a DECREASE in
    membrane conductance -- i.e. closure of a resting K+ conductance -- and
    that depolarization is blocked by the H1 antagonists pyrilamine
    (mepyramine), triprolidine or DIPHENHYDRAMINE ([McCormick1991]). The
    functional consequence measured in the same paper is a switch of
    thalamic firing from rhythmic burst discharge to single-spike activity,
    which is the cellular signature of arousal. So the sign of the H1 gain
    on a rate node is not a guess: more H1 activation, more firing.
  * H2 (Gs) is present on the same cells and does something different in
    the same experiment (enhancement of Ih), and H3 is an autoreceptor.
    Neither is built here. See "What was deliberately not built".

Sources
-------

Affinities. Unlike P5.3's adrenoceptor panel there is no single laboratory
that measured all three of these ligands in one preparation, and the
consequence is stated rather than hidden: the histamine affinity comes from
one pair of papers and the two antagonist affinities from another, so
"diphenhydramine is ~190-fold tighter than histamine at H1" composes two
laboratories rather than quoting one.

  [Booth2002]   Booth RG, Moniri NH, Bakker RA, Choksi NY, Nix WB,
                Timmerman H, Leurs R. "A novel phenylaminotetralin
                radioligand reveals a subpopulation of histamine H1
                receptors." J Pharmacol Exp Ther. 2002;302(1):328-336.
                doi:10.1124/jpet.302.1.328. PMID:12065734. CLONED HUMAN H1
                receptor in CHO cell membranes, [3H]mepyramine competition
                binding (the paper's control radioligand; its own novel
                radioligand is [3H](-)-trans-H2-PAT). Two independent
                determinations per ligand at the human receptor:
                  pyrilamine (mepyramine)  Ki 1.01 and 1.18 nM
                  DIPHENHYDRAMINE          Ki 11.7 and 13.5 nM
                  triprolidine             Ki 1.11 and 1.53 nM
                  chlorpheniramine         Ki 2.67 and 4.81 nM
                and, at the guinea-pig receptor in the same experiment,
                diphenhydramine 9.20 and 9.57 nM.
                ACCESS NOTE, recorded because it matters for reproducing
                this: the publisher's full text returned HTTP 403 to every
                retrieval attempt, so the numbers above are this paper's own
                data as deposited by the authors' laboratory in the NIMH
                PDSP Ki database and served through BindingDB entry 5364
                (https://www.bindingdb.org/entry/5364). The check that this
                is the right paper's data rather than a mis-join: the mean
                of the two diphenhydramine determinations, 12.6 nM, is
                pKi 7.90, and the IUPHAR/BPS Guide to PHARMACOLOGY records
                exactly "diphenhydramine, human H1, pKi 7.9" attributed to
                this reference.
  [Jongejan2005] Jongejan A, Bruysters M, Ballesteros JA, Haaksma E, Bakker
                RA, Pardo L, Leurs R. "Linking agonist binding to histamine
                H1 receptor activation." Nat Chem Biol. 2005;1(2):98-103.
                doi:10.1038/nchembio714. WILD-TYPE HUMAN H1 receptor
                transiently expressed in COS-7 cells, [3H]mepyramine
                displacement: HISTAMINE Ki 79,433 nM (pKi 4.10). The
                paper's subject is the mutational dissection of agonist
                binding versus activation, i.e. exactly the affinity-versus-
                potency gap discussed below.
  [Biselli2021] Biselli S, Bresinsky M, Tropmann K, Forster L, Honisch C,
                Buschauer A, Bernhardt G, Pockes S. "Pharmacological
                characterization of a new series of carbamoylguanidines
                reveals potent agonism at the H2R and D3R." Eur J Med Chem.
                2021;214:113190. doi:10.1016/j.ejmech.2021.113190. HUMAN H1
                receptor expressed in Sf9 insect cells, [3H]mepyramine
                displacement: HISTAMINE Ki 2399 nM (pKi 5.62). The same
                value, from the same laboratory's Sf9 system and with the
                radioligand's own Kd of 4.5 nM stated, appears in
                [Tropmann2021] (Tropmann K, Bresinsky M, Forster L, et al.
                J Med Chem. 2021;64:8684, doi:10.1021/acs.jmedchem.1c00692).
                Both of these are med-chem papers in which histamine is the
                reference agonist; the affinities were read from ChEMBL's
                curation of them (documents CHEMBL4813951, CHEMBL5149991)
                because both publishers' full texts are paywalled.

Binding kinetics. Two papers from one laboratory, and between them they
give this module something P5.1 to P5.5 never had: two independently
obtained association rate constants at the same site that can be checked
against each other.

  [Bosma2017]   Bosma R, Witt G, Vaas LAI, Josimovic I, Gribbon P, Vischer
                HF, Gul S, Leurs R. "The Target Residence Time of
                Antihistamines Determines Their Antagonism of the G
                Protein-Coupled Histamine H1 Receptor." Front Pharmacol.
                2017;8:667. doi:10.3389/fphar.2017.00667 (PMC5627017, open
                access). HUMAN H1R transiently expressed in HEK293T cells,
                [3H]mepyramine competition-association, 25 C for the
                kinetics and 37 C for the equilibrium competition, 50 mM
                phosphate buffer pH 7.4. Its reference radioligand row is
                the one used here:
                  MEPYRAMINE  kon 200 +/- 50 x 10^6 M^-1 min^-1
                              koff 0.28 +/- 0.05 min^-1
                              RT 3.9 +/- 0.6 min, pKd,calc 8.8, pKi 8.5
                and the panel it sits in spans kon from 0.6 to 200 x 10^6
                M^-1 min^-1 across ten antihistamines, a factor of 330 --
                the same "derivation 3 is false across chemotypes" warning
                [Sykes2022]'s beta1 panel gave P5.3 (cited in full in
                `noradrenaline.py`).
  [Bosma2018]   Bosma R, van den Bor J, Vischer HF, Labeaga L, Leurs R.
                "The long duration of action of the second generation
                antihistamine bilastine coincides with its long residence
                time at the histamine H1 receptor." Eur J Pharmacol.
                2018;838:107-111. doi:10.1016/j.ejphar.2018.09.011. Same
                laboratory, same human H1R [3H]mepyramine method. The row
                used here is its SHORT-residence-time reference compound:
                  DIPHENHYDRAMINE  residence time 0.41 +/- 0.1 min
                against bilastine's 73 +/- 5 min and fexofenadine's
                60 +/- 20 min. Only the residence time is reported for
                diphenhydramine in the abstract, and the publisher's full
                text returned HTTP 403, so only the residence time is used;
                see "Derivations".

Efficacy:

  [Bakker2000]  Bakker RA, Wieland K, Timmerman H, Leurs R. "Constitutive
                activity of the histamine H1 receptor reveals inverse
                agonism of histamine H1 receptor antagonists." Eur J
                Pharmacol. 2000;387(1):R5-R7.
                doi:10.1016/s0014-2999(99)00803-1. PMID:10633171. Transient
                expression of wild-type human H1R in COS-7 cells raises
                basal inositol trisphosphate with no agonist present, and
                H1 "antagonists" REDUCE it. This is the citation for the
                efficacy simplification recorded at
                `H1_EFFICACY_ANTAGONIST`, and it is the fourth instance of
                the same one (ketanserin P5.1, SCH-23390 P5.2, prazosin and
                propranolol P5.3).

Anatomy and mechanism (no binding numbers taken from these):

  [McCormick1991] McCormick DA, Williamson A. "Modulation of neuronal firing
                mode in cat and guinea pig LGNd by histamine: possible
                cellular mechanisms of histaminergic control of arousal." J
                Neurosci. 1991;11(10):3188-3199.
                doi:10.1523/JNEUROSCI.11-10-03188.1991 (PMC6575455).
                PMID:1658246. The H1 sign, and the only mechanism paper in
                this module that names the exact drug modelled here:
                histamine slowly depolarizes LGNd relay neurons with a
                DECREASE in membrane conductance, reversing near EK, and the
                depolarization is blocked by pyrilamine, triprolidine or
                diphenhydramine. Histamine halts rhythmic 1-4 Hz burst
                discharge and replaces it with single-spike activity.
  [McCormick1989] McCormick DA, Williamson A. "Convergence and divergence of
                neurotransmitter action in human cerebral cortex." Proc Natl
                Acad Sci USA. 1989;86(20):8098-8102.
                doi:10.1073/pnas.86.20.8098 (PMC298222). PMID:2573061.
                HUMAN cortical pyramidal cells in vitro: acetylcholine,
                histamine, noradrenaline and serotonin all increase spiking
                by reducing spike-frequency adaptation, converging on the
                same potassium currents. Cited for two things: that the
                cortical node is a defensible place to put H1, and that this
                module's excitatory H1 gain is the same KIND of
                approximation P5.3 flagged for beta (an adaptation change
                represented as a drive change, because a rate node has one
                additive input term and no adaptation variable).
  [Haas2003]    Haas H, Panula P. "The role of histamine and the
                tuberomamillary nucleus in the nervous system." Nat Rev
                Neurosci. 2003;4(2):121-130. doi:10.1038/nrn1034.
                PMID:12563283. The projection-system anatomy this module
                deliberately does NOT wire as an edge, for P5.1's reason.

Derivations
-----------

The same three-way split P5.1 established, and this module is the first to
use all three at one site:

1. **Both constants measured** -- use them. MEPYRAMINE ([Bosma2017]).
2. **koff and an equilibrium affinity measured, kon missing** --
   `kon = koff / Kd`. DIPHENHYDRAMINE: koff from [Bosma2018]'s residence
   time, Kd from [Booth2002]. P5.3 wrote this rule down and then recorded
   "not needed here"; this is its first use.
3. **Only an equilibrium affinity measured** -- share the site's measured
   `kon` and set `koff = kon * Kd`. HISTAMINE, sharing mepyramine's.

The check that derivation 2 buys, and it is the strongest arithmetic
result in this module
----------------------------------------------------------------------

Derivation 2 produces a NUMBER THAT NOTHING FORCED. Diphenhydramine's
implied association rate is

    kon = koff / Kd = (1 / 0.41 min) / 12.6 nM = 0.194 nM^-1 min^-1

and mepyramine's MEASURED association rate, from a different paper about a
different drug, is 0.200 nM^-1 min^-1. Three percent apart. The two inputs
are a residence time from [Bosma2018], an equilibrium affinity from
[Booth2002] (a different laboratory, a different cell line, a different
decade) and nothing else; the agreement was not available to be tuned, and
a test asserts it.

What it means is the useful part. Derivation 3's content is "ligands at
this site associate at a comparable rate", and P5.3 established from
[Sykes2022]'s 18-ligand beta1 panel that this is FALSE across chemotypes
and holds within one. Mepyramine (an ethylenediamine) and diphenhydramine
(an ethanolamine) are both first-generation antihistamines built around the
same protonated-amine-plus-two-aryl-rings pharmacophore, and their kon
values agree to 3% while [Bosma2017]'s full panel of ten antihistamines
spans 330-fold. That is a second, independent confirmation of the chemotype
refinement, at a second receptor, and it is why histamine -- which is NOT
of that chemotype -- gets its shared kon with an explicit warning attached.

The histamine affinity problem, which is 33-fold here and 800-fold against
potency
---------------------------------------------------------------------------

P5.1 met this at 5-HT2A (250-fold), P5.3 at alpha2 (10,000-fold). At H1:

    histamine at human H1, [3H]mepyramine displacement,
      wild-type receptor in COS-7 cells:                 Kd 79,433 nM
      [Jongejan2005]
    histamine at human H1, [3H]mepyramine displacement,
      receptor expressed in Sf9 insect cells:            Kd  2,399 nM
      [Biselli2021], [Tropmann2021]

33-fold, and the IUPHAR/BPS Guide to PHARMACOLOGY records the published
range for histamine at human H1 as pKi 4.7-5.9, which brackets both. On top
of that sits a functional gap of a different kind: histamine's EC50 for
H1-mediated phosphoinositide turnover is in the high-nanomolar to low-
micromolar range, i.e. up to ~800-fold left of the COS-7 binding number.
Neither the binding spread nor the binding-versus-function gap is a
contradiction; both are the G-protein-coupling/receptor-reserve story P5.3
documented at length, and [Jongejan2005]'s entire subject is the mutational
separation of the binding step from the activation step.

`ReceptorSite` has one pool and one affinity per ligand, so this module
cannot hold both at once. Two sets are therefore built, `AFFINITY_STATES`
selects between them, and the tests assert the qualitative results under
both. The default is the Sf9 one, by P5.3's stated rule rather than by
preference: **the preparation should match the receptor being modelled.**
The H1 receptor here is a postsynaptic Gq-coupled receptor on a neuron, and
the Sf9 measurement is the one made in a membrane preparation in which the
receptor is functionally coupled; it is also the one consistent with the
tighter end of the published range and therefore the CONSERVATIVE choice
for a model in which an antagonist has to compete with the agonist.

What the numbers turned out to say
----------------------------------

**Diphenhydramine's selectivity over the transmitter is 190-fold, and that
is small.** Kd 12.6 nM against histamine's 2,399 nM in the default set.
Compare P5.3's propranolol at beta1 (2,000-fold over noradrenaline) or
P5.5's NBQX at AMPA. The consequence is visible in the scenario: a dose of
diphenhydramine that fully occupies H1 in the absence of transmitter is
substantially displaced by a large histamine dose, which is the competitive
mechanism working correctly and not a defect.

**Mepyramine's affinity agrees across three routes to it.** koff/kon from
[Bosma2017]'s kinetics is 1.40 nM; that paper's own equilibrium pKi at 37 C
is 8.5 (3.2 nM) and its pKd,calc 8.8 (1.6 nM); [Booth2002]'s competition
binding in a different laboratory gives a mean of 1.10 nM. The full spread
is 2.9-fold, which is the smallest cross-laboratory spread of any agonist-
or-antagonist pair in this module and is asserted as such.

Searched and empty, or found and deliberately not used
------------------------------------------------------

* **Binding kinetics for histamine itself, at H1.** [Bosma2017] and
  [Bosma2018] are kinetic panels of ANTAGONISTS; the endogenous transmitter
  is in neither, exactly as P5.3 found for noradrenaline at all three
  adrenoceptors. Searched additionally for [3H]histamine association or
  dissociation at H1: the H1 radioligand literature is a mepyramine/
  pyrilamine literature, because [3H]histamine labels H3 and H4 sites with
  far higher affinity and is not usable as an H1 agonist radioligand.
  Hence derivation 3 for histamine, with the chemotype warning.
* **Rodent pharmacokinetics for diphenhydramine.** Real and published --
  diphenhydramine is a clinical drug -- and NOT adopted here, for exactly
  P5.3's two reasons: what `drug_pk.py` consumes is a full per-route set
  (F, ka, Vd, CL, brain partition) rather than a half-life, and nothing in
  this receptor-layer task depends on the elimination rate beyond "still on
  board across the scenario's window". `TOOL_COMPOUND_KE` is shared with
  the four modules before this one and a test asserts the equality. Do not
  cite it as a half-life for diphenhydramine or mepyramine.
* **A brain histamine concentration to dose against.** The quantity a
  physiological scenario would want is extracellular histamine in the
  cortex or thalamus of a waking rodent. This is a P6.2 quantity (see
  `HISTAMINE_KE`), and doses here are illustrative bolus concentrations as
  everywhere else in Phase 5.
* **A behavioural dose-response usable as a circuit-layer target.** Not
  searched again; Track E and P2.2 settled the class. Sedation is a real,
  heavily quantified endpoint for diphenhydramine and it is a Phase 7
  target, not a Phase 5 one.

What was deliberately not built
-------------------------------

* **The TMN -> cortex edge.** P5.1's decision for P5.1's reason and P5.3's
  reinforcement of it: the tuberomammillary projection's influence is
  receptor-mediated and of more than one sign at once (H1 excitatory, H3
  presynaptically inhibitory), and a single signed weight would double-count
  the receptor layer. P6.1 closes the loop properly.
* **H2 and H3.** H2 is on the same thalamic neurons in the same experiment
  ([McCormick1991]) and enhances Ih; H3 is the autoreceptor and the target
  of a different drug class entirely. Neither is in the P5.6 brief, and
  adding H2 would need a second sign decision for a current this model has
  no representation of.
* **The peripheral H1 receptor.** Diphenhydramine's anti-allergic action is
  peripheral and its sedation is central; only the central receptor is
  here, and no claim about the peripheral one should be read out of this
  module.
* **Endogenous histaminergic tone.** Without it an antagonist given alone
  is inert, as everywhere else in this codebase before P6.2 -- and here
  that is a particularly conspicuous falsehood, because the entire clinical
  point of diphenhydramine is what it does given alone. The tests that
  assert inertness say so.
"""

import numpy as np

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite


# ---------------------------------------------------------------------------
# Affinities
# ---------------------------------------------------------------------------

#: nM. MEASURED Ki of histamine at the wild-type human H1 receptor
#: transiently expressed in COS-7 cells, [3H]mepyramine displacement
#: [Jongejan2005]. pKi 4.10.
H1_KD_HISTAMINE_COS7 = 79432.82

#: nM. MEASURED Ki of histamine at the human H1 receptor expressed in Sf9
#: insect cells, [3H]mepyramine displacement [Biselli2021] (the identical
#: value is reported by [Tropmann2021] from the same system, alongside the
#: radioligand's own Kd of 4.5 nM). pKi 5.62. The default; see "The
#: histamine affinity problem" in the module docstring.
H1_KD_HISTAMINE_SF9 = 2398.83

#: nM. MEASURED Kd of [3H]mepyramine itself in the Sf9 preparation
#: [Tropmann2021]. Recorded because a competition Ki is only as good as the
#: radioligand Kd the Cheng-Prusoff correction used, and because it is
#: within 3-fold of every other mepyramine number in this module.
H1_KD_MEPYRAMINE_RADIOLIGAND_SF9 = 4.5

#: nM. The two independent determinations of DIPHENHYDRAMINE's Ki at the
#: cloned human H1 receptor in CHO membranes, [3H]mepyramine competition
#: [Booth2002]. Their mean is what the model uses; the pair is kept so the
#: replicate spread (15%) is visible next to the 33-fold spread on
#: histamine.
H1_KI_DIPHENHYDRAMINE_DETERMINATIONS = (11.7, 13.5)

#: nM. MEAN of the two determinations above: 12.6 nM, pKi 7.90. The
#: cross-check that this is [Booth2002]'s number and not a mis-join is that
#: the IUPHAR/BPS Guide to PHARMACOLOGY records "diphenhydramine, human H1,
#: pKi 7.9" attributed to exactly this reference.
H1_KD_DIPHENHYDRAMINE = float(np.mean(H1_KI_DIPHENHYDRAMINE_DETERMINATIONS))

#: nM. The two determinations of pyrilamine (mepyramine) at the human H1
#: receptor in the same [Booth2002] experiment, kept for the three-way
#: agreement check in `mepyramine_affinity_spread()`.
H1_KI_MEPYRAMINE_DETERMINATIONS = (1.01, 1.18)

#: nM. Guinea-pig H1 determinations of diphenhydramine in the SAME
#: [Booth2002] experiment. Not used; recorded because the human/guinea-pig
#: ratio (12.6 / 9.39 = 1.34) is this module's only direct measurement of a
#: species difference, and it is small -- unlike the H1 species differences
#: [Seifert2003] (J Pharmacol Exp Ther 2003;305:1104) reports for bulkier
#: ligands, which run up to ~10-fold.
H1_KI_DIPHENHYDRAMINE_GUINEA_PIG_DETERMINATIONS = (9.20, 9.57)


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. MEASURED association rate constant of mepyramine at the
#: human H1 receptor, competition-association against [3H]mepyramine in
#: HEK293T membranes at 25 C [Bosma2017]: 200 +/- 50 x 10^6 M^-1 min^-1,
#: i.e. 200e6 * 1e-9. Derivation 1 for mepyramine; shared with histamine by
#: derivation 3, WITH the chemotype warning -- [Bosma2017]'s own ten-ligand
#: panel spans 330-fold in kon, so this rate is the weakest thing in the
#: module even though the affinity it encodes is exact.
H1_KON_MEPYRAMINE = 0.200

#: min^-1. MEASURED dissociation rate constant of mepyramine in the same
#: experiment [Bosma2017]: 0.28 +/- 0.05 min^-1, residence time 3.9 min.
H1_KOFF_MEPYRAMINE = 0.28

#: min. MEASURED residence time of DIPHENHYDRAMINE at the human H1 receptor,
#: same laboratory and same [3H]mepyramine method [Bosma2018]: 0.41 +/- 0.1
#: min. Diphenhydramine is the short-residence-time reference compound of
#: that paper, against bilastine's 73 min and fexofenadine's 60 min.
H1_RESIDENCE_TIME_DIPHENHYDRAMINE = 0.41

#: min^-1. MEASURED, as `1 / RT`: 2.439 min^-1. This is a definition, not a
#: derivation -- residence time IS 1/koff.
H1_KOFF_DIPHENHYDRAMINE = 1.0 / H1_RESIDENCE_TIME_DIPHENHYDRAMINE

#: nM^-1 min^-1. DERIVATION 2, and the first use of it in this codebase:
#: `koff / Kd` with koff from [Bosma2018] and Kd from [Booth2002].
#: 2.439 / 12.6 = 0.1936. Compare `H1_KON_MEPYRAMINE`, MEASURED at 0.200 in
#: yet a third experiment: 3% apart, with nothing tuned. See "The check that
#: derivation 2 buys" in the module docstring.
H1_KON_DIPHENHYDRAMINE = H1_KOFF_DIPHENHYDRAMINE / H1_KD_DIPHENHYDRAMINE

#: MEASURED association rate constants of the ten antihistamines in
#: [Bosma2017]'s Table 2, in nM^-1 min^-1 (their 10^6 M^-1 min^-1 units
#: times 1e-3). Kept as data rather than prose because the tests assert the
#: 330-fold span that makes derivation 3 a flagged assumption rather than a
#: safe one.
H1_KON_PANEL = {
    "mepyramine": 0.200,
    "doxepin": 0.070,
    "triprolidine": 0.036,
    "desloratadine": 0.025,
    "olopatadine": 0.0018,
    "levocetirizine": 0.0012,
    "acrivastine": 0.0006,
}

#: MEASURED dissociation rate constants from the same table, min^-1.
H1_KOFF_PANEL = {
    "mepyramine": 0.28,
    "doxepin": 0.06,
    "triprolidine": 0.30,
    "desloratadine": 0.008,
    "olopatadine": 0.006,
    "levocetirizine": 0.008,
    "acrivastine": 0.065,
}


# ---------------------------------------------------------------------------
# Efficacy
# ---------------------------------------------------------------------------

#: Histamine's efficacy. 1.0 by DEFINITION rather than by measurement: it is
#: the reference agonist of every H1 functional assay cited here, so its
#: Emax ratio against itself is 1 and no experiment could have said
#: otherwise. Same standing as noradrenaline's 1.0 at alpha1A and alpha2A in
#: P5.3, and unlike its 1.0 at beta1 there, which [Baker2010] measured.
H1_EFFICACY_HISTAMINE = 1.0

#: The efficacy given to BOTH antagonists here. 0.0 is the same MODELLING
#: SIMPLIFICATION flagged in P5.1 (ketanserin), P5.2 (SCH-23390) and twice
#: in P5.3 (prazosin, propranolol): [Bakker2000] shows the human H1 receptor
#: is constitutively active and that H1 "antagonists" are INVERSE agonists,
#: which is now considered a component of their anti-allergic mechanism.
#: `ReceptorSite.activation` has no constitutive term for a negative
#: efficacy to act against, so a negative value would drive the node BELOW
#: its drug-free baseline -- which is not what an inverse agonist does to a
#: receptor pool with no constitutive tone represented.
H1_EFFICACY_ANTAGONIST = 0.0


#: The affinity sets `ligands_h1` can build. See "The histamine affinity
#: problem" in the module docstring.
AFFINITY_STATES = ("sf9", "cos7")


def histamine_kd(affinity_state="sf9"):
    """The histamine Kd of the named published preparation, in nM."""
    if affinity_state not in AFFINITY_STATES:
        raise ValueError(
            "affinity_state must be one of %s, got %r"
            % (list(AFFINITY_STATES), affinity_state)
        )
    return (
        H1_KD_HISTAMINE_SF9 if affinity_state == "sf9" else H1_KD_HISTAMINE_COS7
    )


def mepyramine_affinity_spread():
    """The four routes to mepyramine's H1 affinity, in nM, smallest first.

    Not used by the model; it exists so the agreement can be asserted rather
    than asserted-about. The four are: [Bosma2017]'s kinetic koff/kon, that
    same paper's pKd,calc and its 37 C pKi, and [Booth2002]'s mean
    competition Ki from a different laboratory.
    """
    kinetic = H1_KOFF_MEPYRAMINE / H1_KON_MEPYRAMINE
    return {
        "booth2002_competition_mean": float(
            np.mean(H1_KI_MEPYRAMINE_DETERMINATIONS)
        ),
        "bosma2017_kinetic_koff_over_kon": kinetic,
        "bosma2017_pkd_calc": 10.0 ** (9.0 - 8.8),
        "bosma2017_pki_37C": 10.0 ** (9.0 - 8.5),
    }


def ligands_h1(affinity_state="sf9"):
    """histamine / diphenhydramine / mepyramine at H1, in that order.

    `affinity_state` selects histamine's affinity only; both antagonists are
    identical in both sets, which is itself an approximation with a known
    direction -- an antagonist's affinity is not state-free either -- and it
    is a much smaller one here than the 33-fold on the agonist.
    """
    kd_histamine = histamine_kd(affinity_state)
    histamine = Ligand(
        name="histamine",
        # Derivation 3, sharing the site's measured antagonist kon across a
        # chemotype boundary. The AFFINITY below is exact by construction;
        # only the timescale over which occupancy approaches it carries this
        # assumption. See the module docstring.
        kon=H1_KON_MEPYRAMINE,
        koff=H1_KON_MEPYRAMINE * kd_histamine,
        efficacy=H1_EFFICACY_HISTAMINE,
    )
    diphenhydramine = Ligand(
        name="diphenhydramine",
        kon=H1_KON_DIPHENHYDRAMINE,  # derivation 2
        koff=H1_KOFF_DIPHENHYDRAMINE,  # measured, 1/RT [Bosma2018]
        efficacy=H1_EFFICACY_ANTAGONIST,  # simplification; see the constant
    )
    mepyramine = Ligand(
        name="mepyramine",
        kon=H1_KON_MEPYRAMINE,  # derivation 1, measured [Bosma2017]
        koff=H1_KOFF_MEPYRAMINE,  # derivation 1, measured [Bosma2017]
        efficacy=H1_EFFICACY_ANTAGONIST,
    )
    return [histamine, diphenhydramine, mepyramine]


def receptor_site_h1(affinity_state="sf9"):
    """One competitive H1 pool, postsynaptic on the cortical node.

    ONE pool. H1 receptors sit on many other cells -- thalamic relay
    neurons, where [McCormick1991] measured the response this module's sign
    comes from, and cerebral vasculature, and mast cells. This site is the
    postsynaptic neuronal one.
    """
    return ReceptorSite(
        name="H1_on_cortical_pyramidal",
        ligands=ligands_h1(affinity_state=affinity_state),
    )


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, shared by diphenhydramine and
#: mepyramine, and deliberately the same number as `serotonin`,
#: `dopamine`, `noradrenaline`, `acetylcholine` and `glutamate`'s
#: `TOOL_COMPOUND_KE`; a test asserts the equality. Do not cite it as a
#: half-life for either drug.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY, the same value and the same reason as the
#: five transmitter clearance constants before it. Extracellular histamine
#: is not cleared by first-order elimination: it is taken up by astrocytes
#: and degraded by histamine N-methyltransferase, and there is no neuronal
#: histamine transporter at all. 5.0 min^-1 (t1/2 ~8.3 s) preserves the only
#: property the dynamics need -- that the transmitter clears far faster than
#: any dosed drug -- at a timescale the integrator resolves. The real
#: quantity belongs to P6.2.
HISTAMINE_KE = 5.0


def pk_models_h1():
    """One PK model per H1 ligand, in `ligands_h1()` order."""
    return [
        OneCompartmentPK(ke=HISTAMINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: the shared cortical loop plus an isolated source nucleus
# ---------------------------------------------------------------------------
#
# Three nodes, and one edge that is missing on purpose:
#
#   TMN_HA          tuberomammillary histaminergic neurons. Carries no
#                   receptor here (H3 is the autoreceptor and is not built),
#                   and has NO edges: see the module docstring. Same shape
#                   and same reason as P5.1's `DRN_5HT`, P5.3's `LC_NA` and
#                   P5.4's `BF_ACh`.
#   PFC_Pyramidal   layer V/VI prefrontal pyramidal neurons. Carries the H1
#                   receptor (Gq, excitatory, [McCormick1991],
#                   [McCormick1989]).
#   PFC_GABA        local cortical GABAergic interneurons.
#
# The two loop weights, the gain and the two baseline inputs are
# character-for-character those of `serotonin.raphe_cortical_network`,
# `noradrenaline.locus_coeruleus_cortical_network`,
# `acetylcholine.basal_forebrain_cortical_network` and
# `glutamate.cortical_network`. Reusing them is the point, and it is P5.5's
# argument: a result has to be a consequence of the connectivity every other
# P5 scenario already ran on, not of a loop drawn to produce it.
#
# The three rules, unchanged from P5.3:
#
#   1. Baselines put every node at rate 0.5 at rest with no drug,
#      `stability.MAX_SENSITIVITY_RATE`; `stability.balanced_baselines`
#      reproduces +0.45 and -0.4 exactly and a test asserts it.
#   2. contraction_number = 0.9 < 1, P3.4's global-uniqueness certificate.
#   3. No node saturates under a saturating dose of a full agonist at every
#      receptor the node carries. `PFC_Pyramidal` carries ONE receptor here,
#      so the per-receptor unit budget applies and the gain is +1.0.

#: ILLUSTRATIVE. Unchanged from every hand-built cortical loop in this
#: codebase.
CORTICAL_INHIBITION_WEIGHT = -0.9
CORTICAL_RECURRENT_WEIGHT = 0.8

#: ILLUSTRATIVE, and the same value every hand-built scenario uses.
CORTICAL_GAIN = 4.0

#: The node parameters. `baseline_input` is fully determined by rule 1.
_NODES = (
    # name,           tau,  baseline_input
    ("TMN_HA", 1.0, 0.0),
    ("PFC_Pyramidal", 1.5, 0.45),
    ("PFC_GABA", 1.0, -0.4),
)


def tuberomammillary_cortical_network():
    """TMN_HA (isolated) + PFC_Pyramidal -| PFC_GABA cortical loop."""
    nodes = [
        CircuitNode(name=name, tau=tau, baseline_input=baseline, gain=CORTICAL_GAIN)
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

    Same reason as P5.2 and P5.3: `HISTAMINE_KE` clears the transmitter in
    seconds, so a cortical node with tau 1.5 min never reaches the rate its
    peak occupancy would support, and a start-up transient from zero would
    swamp what is left.
    """
    n = len(_NODES) if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE. How much a unit of H1 activation changes the drive of the
#: cortical pyramidal node. POSITIVE: H1 is Gq/11-coupled, and histamine
#: acting at it closes a resting K+ conductance and depolarizes the cell
#: ([McCormick1991] in thalamus, where the block by diphenhydramine itself
#: was demonstrated; [McCormick1989] in cortex, where the readout is reduced
#: spike-frequency adaptation).
#:
#: Magnitude 1.0 by P5.1's per-receptor unit rule, because this node carries
#: one receptor in this scenario. Not identifiable from anything published
#: (P2.3 established that only the PRODUCT of a coupling gain and a
#: connection weight is identified even where behavioural data exists), and
#: it is the `mechanism_gain` half of P4.2's `coupling = mechanism_gain *
#: rho[region]`.
#:
#: The approximation inside the sign, stated because P5.3 stated it for
#: beta: what [McCormick1989] measures in cortex is reduced ADAPTATION,
#: which in a rate model is a gain/threshold change rather than an additive
#: input, and this engine has one additive drive term per node and no
#: adaptation variable. The thalamic result is a genuine depolarization and
#: therefore a cleaner match to a drive term; the cortical node is used
#: anyway, for the loop-reuse reason above.
H1_MECHANISM_GAIN = 1.0


def coupling_h1():
    """H1 activation -> drive, on the cortical pyramidal node."""
    return {"PFC_Pyramidal": H1_MECHANISM_GAIN}


def receptor_system_h1(affinity_state="sf9"):
    """The H1 stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_h1(),
        receptor_site=receptor_site_h1(affinity_state=affinity_state),
        coupling=coupling_h1(),
        name="H1",
    )


def histaminergic_receptor_systems(affinity_state="sf9"):
    """The H1 system alone, in list form for `Simulation(...)`."""
    return [receptor_system_h1(affinity_state=affinity_state)]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def cortical_output_readout():
    """Layer V/VI prefrontal pyramidal firing.

    Deliberately the same name, node and weight as
    `serotonin.cortical_output_readout`, `noradrenaline`,
    `acetylcholine` and `glutamate`'s; a test asserts all five are equal.
    This is an AROUSAL-DIRECTION circuit quantity and NOT a sedation
    endpoint: naming sedation requires Phase 7, and the fact that
    diphenhydramine's sedation is the single most quantified thing about
    the drug makes it a good Phase 7 target rather than a licence to name
    it here.
    """
    return BehaviorReadout(name="cortical_output", weights={"PFC_Pyramidal": 1.0})


def histaminergic_tone_readout():
    """Tuberomammillary histaminergic-neuron firing.

    Flat in every scenario in this module, because TMN_HA carries no
    receptor and has no edges. It exists so that the day P6.1 gives the
    projection system a source, the readout it should move is already
    written down -- the same role `serotonin.serotonergic_tone_readout` and
    `noradrenaline.noradrenergic_tone_readout` play.
    """
    return BehaviorReadout(name="histaminergic_tone", weights={"TMN_HA": 1.0})
