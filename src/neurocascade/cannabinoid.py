"""P5.6b: the cannabinoid CB1 receptor system, as connection modulation.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

WHY CB1 IS NOT A `coupling` ENTRY, WHICH IS WHAT THIS MODULE IS ABOUT
====================================================================

Every receptor system in P5.1 to P5.4 adds DRIVE to a node: the receptor
activates, and `coupling[node] * activation` is added to that node's
Wilson-Cowan input. That shape encodes a specific claim -- "this receptor
sits on the modelled population and changes how much it fires" -- and it is
the right claim for a postsynaptic Gq, Gi or Gs receptor.

It is the wrong claim for CB1, and the reason is anatomical rather than
computational. CB1 is **presynaptic**. It is expressed on axon terminals,
where it inhibits transmitter release; in the hippocampus it clusters on
the terminals of cholecystokinin-positive GABAergic interneurons
([Katona1999]), and the receptor that matters for the effect modelled here
is not on the neuron whose rate would change but on the terminal of the
neuron projecting onto it. And the physiological signal that activates it
runs BACKWARDS: depolarizing a pyramidal neuron makes it release
endocannabinoids that travel retrogradely to the interneuron terminal and
transiently suppress GABA release ([Wilson2001], [OhnoShosaku2001]). A
`coupling` entry has no way to say either of those things. It has one
index, the postsynaptic node, and no notion of which input it acts on.

`connection_modulation` says exactly both of them. Its key is a
`(post_node, pre_node)` PAIR, so it names a specific projection, and its
effect is multiplicative on that projection's weight::

    W_eff[post, pre] = W[post, pre] * (1 + gain * activation)

which is what presynaptic inhibition of release IS in a rate model: the
same presynaptic firing produces less postsynaptic effect. This is the
mechanism P5.5 used for glutamate and the mechanism GABA-A allosteric
modulation has used since `engine.py` first grew the feature for
diazepam (`scenarios.gaba_a_connection_modulation`). CB1 is the third
occupant, and the first for which the reason is PRESYNAPTIC LOCALIZATION
rather than allosteric potentiation (diazepam) or avoidance of a double
count (glutamate).

The gain is a measurement, not an illustrative constant
=======================================================

P5.5's glutamate gain was pinned at 1.0 by algebra, which made it the first
P5 module with no illustrative coupling gain. This module is the second,
and by a different route: **the gain here is a measured number.**

[HoffmanLupica2000] recorded evoked GABA-A IPSCs in CA1 pyramidal neurons
of rat hippocampal slices and applied the CB1 agonist WIN 55,212-2. The
IPSC was reduced with an EC50 of 138 nM, and at 5 uM -- 36-fold above that
EC50, i.e. saturating -- the reduction was **47 +/- 4% (n = 12)**. The
effect was blocked by SR141716A (rimonabant) and not by naloxone, was
presynaptic (it changed the frequency of action-potential-dependent
spontaneous IPSCs without touching miniature IPSCs), and was abolished by
cadmium, i.e. it works through presynaptic voltage-dependent calcium
channels.

So the maximal fraction of a GABAergic connection that a saturating CB1
agonist removes is 0.47, and this module's `connection_modulation` gain is
`-CB1_MAX_SUPPRESSED_FRACTION = -0.47`. At saturation by a full agonist the
factor is `1 - 0.47 = 0.53` and the inhibitory weight retains 53% of its
strength; with no drug on board the activation is exactly zero, the factor
is exactly 1, and this module changes nothing about any result that
predates it. A test asserts both ends.

Two things that follows from putting the measurement in the gain:

1. **A CB1 agonist in this model cannot silence inhibition.** It can remove
   at most 47% of it, because that is what was measured. A model in which a
   saturating agonist abolished the synapse would be asserting something no
   experiment supports, and the ceiling is the reason the disinhibition
   result below is modest rather than dramatic.
2. **The sign of the gain is negative and the sign of the weight is
   negative, and those are different facts.** The weight
   `W[PFC_Pyramidal, PFC_GABA] = -0.9` is negative because the synapse is
   inhibitory; the gain is negative because CB1 REMOVES transmission. Their
   product is a weaker inhibition -- disinhibition -- and not a sign flip:
   `-0.9 * 0.53 = -0.477` is still inhibitory. A test asserts that the
   modulated weight never changes sign, because a `connection_modulation`
   gain below -1 would silently turn an inhibitory synapse excitatory, and
   nothing in the engine forbids it.

Where the difference from P5.5 is, since both use the same feature
-----------------------------------------------------------------

P5.5 had to put its fraction inside the receptor SITE
(`GlutamatergicTransmissionSite.activation` returns a negative number and
the gain is +1), because two glutamate systems partition ONE weight between
them and `share_NMDA + share_AMPA == 1` is an identity the site has to
enforce. CB1 has no partition to enforce: it is one receptor with one
measured maximal effect on one connection. So the fraction lives in the
gain, exactly where GABA-A's potentiation gain lives, and `activation` is
the ordinary positive `sum(occupancy * efficacy)` that every module before
P5.5 used. Nothing new is added to `receptor.py` by this module, and that
is the intended outcome: the brief said CB1 "does not fit the postsynaptic
receptor pattern", and the answer turned out to be that the codebase
already had the pattern it does fit.

The measurement's tissue does not match the loop's label
--------------------------------------------------------

Stated up front because it is the largest single weakness here.
[HoffmanLupica2000]'s 47% is rat hippocampal CA1: pyramidal neuron,
GABAergic interneuron, evoked IPSC. The network this module runs on is the
two-node loop labelled `PFC_Pyramidal` / `PFC_GABA` that P5.1, P5.3, P5.4
and P5.5 all share, so a hippocampal number is being applied to a
cortically-labelled loop. Both readings of that are recorded:

  * In favour: the loop is a generic pyramidal/interneuron microcircuit
    with illustrative weights, not a claim about prefrontal cortex in
    particular, and CB1 is expressed on CCK-positive interneuron terminals
    in neocortex as well as hippocampus. Reusing the shared loop is what
    makes this module's result comparable with P5.1 to P5.5's, which is
    worth more here than a second hand-drawn network would be.
  * Against: 47% is a number from one preparation and this module applies
    it outside it. If a neocortical maximal suppression is measured and
    differs, `CB1_MAX_SUPPRESSED_FRACTION` is the one constant to change.

The honest version of the claim this module supports is therefore "a
saturating CB1 agonist disinhibits a pyramidal/interneuron loop by removing
the measured fraction of its inhibitory synapse", not "THC raises
prefrontal firing by X%".

Sources
-------

Affinities. There is no single-laboratory panel in reach that measured all
four ligands, so this module takes all four from ONE COMPILATION instead,
which is the next best thing and is at least internally consistent:

  [Pertwee2010] Pertwee RG, Howlett AC, Abood ME, Alexander SPH, Di Marzo
                V, Elphick MR, Greasley PJ, Hansen HS, Kunos G, Mackie K,
                Mechoulam R, Ross RA. "International Union of Basic and
                Clinical Pharmacology. LXXIX. Cannabinoid receptors and
                their ligands: beyond CB1 and CB2." Pharmacol Rev.
                2010;62(4):588-631. doi:10.1124/pr.110.003004 (PMC2993256,
                open access). Its Table 1 gives CB1 Ki values, measured
                "usually [3H]CP55940, but sometimes [3H]SR141716A,
                [3H]R-(+)-WIN55212, or HU-243", as RANGES over the published
                literature:
                  (-)-Delta9-THC        5.05 - 80.3 nM
                  anandamide              61 - 543  nM
                  2-AG                  58.3 and 472 nM (two values, not a
                                        range: the paper prints both)
                  rimonabant (SR141716A) 1.8 - 12.3 nM
                  CP55940                0.5 - 5.0  nM
                  R-(+)-WIN55212        1.89 - 123  nM
                The last two are not instantiated and are recorded because
                they are the reference full agonists the efficacies below
                are measured against.

Efficacy. Three papers, three preparations, three reference agonists, and
that is a weakness recorded rather than smoothed:

  [Savinainen2001] Savinainen JR, Jarvinen T, Laine K, Laitinen JT.
                "Despite substantial degradation, 2-arachidonoylglycerol is
                a potent full efficacy agonist mediating CB1 receptor-
                dependent G-protein activation in rat cerebellar membranes."
                Br J Pharmacol. 2001;134(3):664-672.
                doi:10.1038/sj.bjp.0704297 (PMC1572991, open access). RAT
                cerebellar membranes, agonist-stimulated [35S]GTPgammaS
                binding, with GDP raised to 10^-4 M specifically to separate
                efficacies that look identical at low GDP. Conclusion, in
                the title: 2-AG is a FULL efficacy agonist, and at the
                discriminating GDP concentration its efficacy exceeds
                CP-55,940's. This is why 2-AG rather than CP55940 is this
                module's efficacy reference.
  [Liu2018]     Liu Y, Ji L, Eno M, Kudalkar S, Li AL, Schimpgen M, et al.
                J Med Chem. 2018;61(18):8639. ANANDAMIDE Emax 61% relative
                to 500 nM CP-55,940, rat CB1 receptor in HEK293 cells,
                inhibition of forskolin-stimulated cAMP accumulation. Read
                from ChEMBL's curation (document CHEMBL4196195); the
                publisher's text is paywalled.
  [Cheng2012]   Cheng YX, Pourashraf M, Luo X, Srivastava S, Walpole C,
                Salois D, et al. Bioorg Med Chem Lett. 2012;22(4):1619.
                Delta9-THC Emax 20% relative to R-(+)-WIN55,212-2, HUMAN CB1
                in HEK293-EBNA cells, [35S]GTPgammaS binding. Read from
                ChEMBL (document CHEMBL1949601). Two further readings of the
                same quantity, from [Mahardhika2024] (J Med Chem
                2024;67:9896, ChEMBL document CHEMBL5500402), are 22%
                (beta-arrestin recruitment, relative to CP55940) and 31%
                (cAMP, relative to forskolin) at human CB1 in CHO cells. The
                three span 20-31%, which is the honest error bar on THC's
                efficacy and is narrow enough that nothing here turns on
                which is used.
  [Landsman1997] Landsman RS, Burkey TH, Consroe P, Roeske WR, Yamamura HI.
                "SR141716A is an inverse agonist at the human cannabinoid
                CB1 receptor." Eur J Pharmacol. 1997;334(1):R1-R2.
                doi:10.1016/s0014-2999(97)01160-6. PMID:9346339. Human CB1
                in CHO membranes: WIN 55,212-2 stimulated [35S]GTPgammaS
                binding 76.3% ABOVE basal while SR141716A produced a 22.3%
                DECREASE below it. The citation for the efficacy
                simplification at `CB1_EFFICACY_RIMONABANT`.

Mechanism and anatomy (no binding numbers taken from these):

  [HoffmanLupica2000] Hoffman AF, Lupica CR. "Mechanisms of cannabinoid
                inhibition of GABA(A) synaptic transmission in the
                hippocampus." J Neurosci. 2000;20(7):2470-2479.
                doi:10.1523/JNEUROSCI.20-07-02470.2000 (PMC6772239).
                PMID:10729327. RAT hippocampal slices (Sprague-Dawley,
                P14-30), whole-cell voltage clamp of CA1 pyramidal neurons
                at -80 mV, room temperature. The source of this module's
                gain, of its EC50 cross-check, and of the antagonist-alone
                result below.
  [Katona1999]  Katona I, Sperlagh B, Sik A, Kafalvi A, Vizi ES, Mackie K,
                Freund TF. "Presynaptically located CB1 cannabinoid
                receptors regulate GABA release from axon terminals of
                specific hippocampal interneurons." J Neurosci.
                1999;19(11):4544-4558.
                doi:10.1523/JNEUROSCI.19-11-04544.1999. PMID:10341254. The
                localization: CB1 on the axon terminals of a specific
                interneuron class, not on the pyramidal cell.
  [Wilson2001]  Wilson RI, Nicoll RA. "Endogenous cannabinoids mediate
                retrograde signalling at hippocampal synapses." Nature.
                2001;410(6828):588-592. doi:10.1038/35069076. PMID:11279497.
                The retrograde direction: depolarization-induced suppression
                of inhibition is mediated by endocannabinoids released from
                the depolarized pyramidal neuron acting on CB1 on its own
                inhibitory inputs.
  [OhnoShosaku2001] Ohno-Shosaku T, Maejima T, Kano M. "Endogenous
                cannabinoids mediate retrograde signals from depolarized
                postsynaptic neurons to presynaptic terminals." Neuron.
                2001;29(3):729-738. doi:10.1016/s0896-6273(01)00247-1.
                PMID:11301031. The independent, simultaneous demonstration
                of the same thing in cultured hippocampal neurons.

The one result here that is NOT the usual Phase 5 caveat
--------------------------------------------------------

Every module from P5.1 to P5.4 had to record that its antagonist given
ALONE is inert, because there is no endogenous transmitter tone in this
codebase for the antagonist to remove (P6.2). P5.5 broke that pattern in
one direction: a glutamate antagonist alone is NOT inert, because the
transmission it removes is `W[i,j]` and needs no dose.

CB1 restores the pattern, and for once the model's inertness is a MEASURED
fact rather than a modelling artefact. [HoffmanLupica2000] applied
SR141716A alone to the same slices and the evoked IPSC was 115 +/- 17% of
control (n = 9), i.e. no significant effect. In an acute slice with no
ongoing postsynaptic depolarization there is no tonic endocannabinoid
release for rimonabant to block, so rimonabant alone does nothing -- which
is exactly what this model predicts and for the same reason. The test that
asserts rimonabant's inertness cites that number, and it is the first time
in Phase 5 that the standing "no endogenous tone" limitation has an
experiment agreeing with it rather than contradicting it.

The limitation is not thereby cancelled. In an intact, behaving animal
there IS endocannabinoid tone and rimonabant does have effects given alone;
the drug was withdrawn from the market because of them. What the slice
result establishes is narrower and worth having: the inertness in this
model is a correct prediction FOR A PREPARATION WITHOUT TONE, so the day
P6.2 supplies tone, the change in the prediction is the thing to check.

Searched and empty
------------------

* **Binding kinetics at CB1, for any ligand.** Searched for association and
  dissociation rate constants, residence times, competition-association and
  NanoBRET kinetics at CB1: the cannabinoid binding literature is an
  equilibrium-Ki literature, and the kinetic-panel programmes that exist for
  adrenoceptors ([Sykes2022], cited in `noradrenaline.py`), the H1
  receptor ([Bosma2017], cited in `histamine.py`) and the A2A
  receptor ([Guo2012]) have no CB1 counterpart in reach. The likely reason
  is chemical rather than accidental: these are extremely lipophilic ligands
  that partition into membrane before reaching the receptor, and CB1's
  orthosteric site is entered through a lipid portal, so a
  solution-phase kon is not straightforwardly measurable. Consequence: ALL
  FOUR ligands share `ILLUSTRATIVE_KON`, deliberately the same hand-set 0.01
  nM^-1 min^-1 as `acetylcholine.ILLUSTRATIVE_KON` and
  `glutamate.ILLUSTRATIVE_KON`, and a test asserts the equality. Every
  affinity is reproduced EXACTLY by construction (`koff = kon * Kd`); only
  the timescale over which occupancy approaches equilibrium is illustrative,
  and no result in this module is stated in terms of a rate.
* **A single-laboratory panel measuring THC, anandamide, 2-AG and
  rimonabant at CB1.** [Felder1995] (Mol Pharmacol 48:443) and
  [Showalter1996] (J Pharmacol Exp Ther 278:989) are the two candidates the
  IUPHAR tables are built from and both publishers returned HTTP 403 to
  every retrieval attempt. Hence [Pertwee2010]'s compiled ranges, which at
  least come from one editorial process.
* **A brain endocannabinoid concentration to dose against.** 2-AG and
  anandamide are synthesized on demand from membrane precursors and are not
  stored or released like a classical transmitter, so "extracellular
  concentration" is a quantity with a much weaker meaning here than for
  dopamine or acetylcholine. P6.2's problem, and a harder instance of it
  than any transmitter before it.
* **Rodent pharmacokinetics for THC and rimonabant.** Same position, and
  for the same two reasons, as P5.3's four clinical drugs: what Phase 1
  consumes is a full per-route parameter set, and nothing in this
  receptor-layer task depends on the elimination rate.
"""

import numpy as np

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite, ligand_from_ki


# ---------------------------------------------------------------------------
# Affinities: one compilation, reported as ranges, instantiated explicitly
# ---------------------------------------------------------------------------

#: nM. The CB1 Ki values of [Pertwee2010] Table 1, as `(low, high)` pairs
#: exactly as that table prints them. For 2-AG the table prints two discrete
#: values rather than a range ("58.3, 472"); the pair is stored the same way
#: because what the model needs from either is the same thing, a bracket.
#: Radioligand: usually [3H]CP55940.
CB1_KD_BOUNDS = {
    "thc": (5.05, 80.3),
    "anandamide": (61.0, 543.0),
    "2ag": (58.3, 472.0),
    "rimonabant": (1.8, 12.3),
}

#: nM. The same table's values for the two reference FULL agonists, which
#: this module does not instantiate but does need to name, because the
#: efficacies below are Emax ratios against them.
CB1_KD_BOUNDS_REFERENCE_AGONISTS = {
    "cp55940": (0.5, 5.0),
    "win55212": (1.89, 123.0),
}

#: The affinity choices `ligands_cb1` can build.
#:
#: "midpoint" (default) is the GEOMETRIC MEAN of each ligand's published
#: bounds. It is labelled a DERIVED SUMMARY of two cited numbers, in the
#: same category as derivation 3's `koff = kon * Kd`, and explicitly NOT a
#: measurement: no experiment reported 20.1 nM for THC. It is the default
#: because a single scenario needs one number and the geometric mean is the
#: only choice that is symmetric in the log-affinity the bounds are
#: effectively quoted in.
#:
#: "tight" and "weak" instantiate the two endpoints, which ARE published
#: values. Every qualitative claim in the tests is asserted under all three,
#: and the claims that do NOT survive are reported as findings rather than
#: quietly dropped -- see `affinity_ordering_is_resolved()`.
AFFINITY_CHOICES = ("midpoint", "tight", "weak")


def cb1_kd(ligand, affinity="midpoint"):
    """The Kd used for one CB1 ligand, in nM, under one affinity choice."""
    if affinity not in AFFINITY_CHOICES:
        raise ValueError(
            "affinity must be one of %s, got %r" % (list(AFFINITY_CHOICES), affinity)
        )
    low, high = CB1_KD_BOUNDS[ligand]
    if affinity == "tight":
        return low
    if affinity == "weak":
        return high
    return float(np.sqrt(low * high))


def affinity_ordering_is_resolved(tighter, weaker):
    """Does the published literature settle that `tighter` binds CB1 more
    tightly than `weaker`?

    True only when the two ligands' [Pertwee2010] bounds do not overlap,
    i.e. when the claim survives every combination of published values.

    This exists because MOST of the orderings this module would like to
    state are not resolved, which was not expected and is the honest
    headline of the affinity half of this task. Of the twelve ordered pairs
    of the four ligands, exactly TWO survive: rimonabant (1.8-12.3 nM) is
    tighter than 2-AG (58.3-472) and than anandamide (61-543). THC's range
    (5.05-80.3) overlaps every other ligand's, so neither "rimonabant binds
    CB1 more tightly than THC" (midpoints 4.7 against 20.1 nM) nor "THC
    binds more tightly than anandamide" (20.1 against 182 nM) is settled by
    the compiled literature, even though both look decisive at the
    midpoints. [Pertwee2010] states one of these in prose -- anandamide's
    "affinity for the CB1 receptor is also similar to that of Delta9-THC"
    -- and this function is the arithmetic version, so the model never
    quietly claims a selectivity the data does not contain. A test asserts
    the exact set.
    """
    return CB1_KD_BOUNDS[tighter][1] < CB1_KD_BOUNDS[weaker][0]


# ---------------------------------------------------------------------------
# Efficacy
# ---------------------------------------------------------------------------

#: 2-AG's efficacy. 1.0, and this module's efficacy REFERENCE, because
#: [Savinainen2001] measured it to be a full-efficacy agonist at rat brain
#: CB1 -- exceeding CP-55,940's efficacy under the high-GDP conditions that
#: separate efficacies at all. Choosing the endogenous ligand rather than a
#: synthetic reference is also what makes the ratios below mean "fraction of
#: what the brain's own agonist does", which is the quantity the circuit
#: layer wants.
CB1_EFFICACY_2AG = 1.0

#: Anandamide's efficacy. MEASURED Emax of 61% relative to 500 nM CP-55,940
#: at rat CB1 in HEK293 cells, cAMP readout [Liu2018]. The reference agonist
#: is CP-55,940 rather than this module's 2-AG, so this ratio and the
#: reference are not from one panel; [Savinainen2001] puts 2-AG at or above
#: CP-55,940, which makes 0.61 if anything an OVER-estimate of anandamide
#: relative to 2-AG. The direction of the approximation is stated because
#: the qualitative claim -- anandamide is a partial agonist and 2-AG is not
#: -- is what the tests assert, and it is robust to that direction.
CB1_EFFICACY_ANANDAMIDE = 0.61

#: Delta9-THC's efficacy. MEASURED Emax of 20% relative to R-(+)-WIN55,212-2
#: at human CB1, [35S]GTPgammaS [Cheng2012]. The three published readings of
#: this quantity found (20%, 22%, 31%; see the module docstring) span 1.55-
#: fold, which is narrow by this codebase's standards -- P5.3's clonidine
#: efficacy spanned 0.32 / 0.027 / undetectable. The tests assert that no
#: result here changes across 0.20 to 0.31.
CB1_EFFICACY_THC = 0.20

#: The alternative published readings of THC's efficacy, kept so the
#: robustness claim above is testable rather than asserted.
CB1_EFFICACY_THC_ALTERNATIVES = (0.22, 0.31)

#: Rimonabant's efficacy. 0.0 is the fifth instance of the same MODELLING
#: SIMPLIFICATION (ketanserin P5.1, SCH-23390 P5.2, prazosin and
#: propranolol P5.3, the antihistamines in `histamine.py`):
#: [Landsman1997] measured SR141716A to REDUCE basal [35S]GTPgammaS binding
#: by 22.3%, so it is an inverse agonist and not a neutral antagonist, and
#: `ReceptorSite.activation` has no constitutive term for a negative
#: efficacy to act against.
#:
#: It bites harder here than in any earlier module, and that is worth
#: saying: [Landsman1997] measured SR141716A to take basal [35S]GTPgammaS
#: binding 22.3% BELOW basal in a system where a full agonist raised it
#: 76.3% above, so the constitutive activity being discarded here is about a
#: quarter of the drug-free signal rather than a rounding error. What a
#: correct treatment would need is a constitutive-activity term in
#: `ReceptorSite`, which would change every module in the codebase.
CB1_EFFICACY_RIMONABANT = 0.0


# ---------------------------------------------------------------------------
# Kinetics: none published, so one illustrative rate for every ligand
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. ILLUSTRATIVE, and deliberately the same hand-set value as
#: `acetylcholine.ILLUSTRATIVE_KON` and `glutamate.ILLUSTRATIVE_KON`; a test
#: asserts the equality. No CB1 binding kinetics were found for any ligand
#: (see "Searched and empty"). Every affinity below is reproduced EXACTLY by
#: `koff = kon * Kd` whatever this value is; only the approach timescale
#: carries it, and no result in this module is stated in terms of a rate.
ILLUSTRATIVE_KON = 0.01


def _ligand(name, kd, efficacy):
    """Derivation 3 with an illustrative rather than a measured shared kon:
    `koff = kon * Kd`, so the measured affinity is exact.

    The arithmetic and the assumption behind it are `receptor.ligand_from_ki`
    (P8.1)."""
    return ligand_from_ki(
        name=name, ki_nM=kd, kon=ILLUSTRATIVE_KON, efficacy=efficacy
    )


def ligands_cb1(affinity="midpoint", thc_efficacy=CB1_EFFICACY_THC):
    """2-AG / anandamide / THC / rimonabant at CB1, in that order.

    Both endogenous ligands are carried, not one. They are not
    interchangeable: [Savinainen2001] found 2-AG is a full-efficacy agonist
    and more potent than anandamide at rat brain CB1, while the compiled
    affinities put them within 1.1-fold of each other at the midpoint. A
    model that carried only "the endocannabinoid" would have to choose
    between an efficacy of 1.0 and one of 0.61 with nothing to justify the
    choice; carrying both makes the difference visible and testable.

    `thc_efficacy` exists so the tests can re-run every result across the
    three published readings of it (0.20, 0.22, 0.31) rather than assert
    robustness without checking.
    """
    return [
        _ligand("2ag", cb1_kd("2ag", affinity), CB1_EFFICACY_2AG),
        _ligand("anandamide", cb1_kd("anandamide", affinity), CB1_EFFICACY_ANANDAMIDE),
        _ligand("thc", cb1_kd("thc", affinity), thc_efficacy),
        _ligand("rimonabant", cb1_kd("rimonabant", affinity), CB1_EFFICACY_RIMONABANT),
    ]


def receptor_site_cb1(affinity="midpoint", thc_efficacy=CB1_EFFICACY_THC):
    """One competitive CB1 pool, on a GABAergic axon TERMINAL.

    An ordinary `ReceptorSite`, deliberately: the four ligands genuinely do
    compete for one orthosteric pool, and nothing about CB1's presynaptic
    location changes the binding arithmetic. What changes is what is done
    with `activation`, and that is the receptor SYSTEM's business
    (`connection_modulation`), not the site's. This is the opposite of
    P5.4's and P5.5's finding, where a new mechanism needed a new site
    class, and it is the reason this module adds nothing to `receptor.py`.
    """
    return ReceptorSite(
        name="CB1_on_GABAergic_terminal",
        ligands=ligands_cb1(affinity=affinity, thc_efficacy=thc_efficacy),
    )


# ---------------------------------------------------------------------------
# The measured connection-modulation gain
# ---------------------------------------------------------------------------

#: MEASURED. The fraction of an evoked GABA-A IPSC that a SATURATING
#: concentration of the CB1 agonist WIN 55,212-2 removes in rat hippocampal
#: CA1: 47 +/- 4% at 5 uM, n = 12 [HoffmanLupica2000]. This is the maximal
#: effect, not a fitted scale: 5 uM is 36-fold above that paper's own EC50
#: of 138 nM.
CB1_MAX_SUPPRESSED_FRACTION = 0.47

#: nM. MEASURED EC50 of WIN 55,212-2 for that suppression
#: [HoffmanLupica2000]. Not used to build any ligand -- WIN 55,212-2 is not
#: in this module's ligand set and a functional EC50 in slice is not a
#: binding Kd -- and recorded for the comparison in
#: `saturating_margin_of_the_measured_maximum()`, which is what licenses
#: calling 47% a maximum.
CB1_WIN55212_IPSC_EC50 = 138.0

#: nM. The concentration at which the 47% was measured.
CB1_WIN55212_SATURATING_CONCENTRATION = 5000.0

#: MEASURED. Evoked IPSC amplitude as a fraction of control when the CB1
#: antagonist SR141716A was applied ALONE to the same slices: 1.15 +/- 0.17,
#: n = 9, i.e. no significant effect [HoffmanLupica2000]. The experimental
#: counterpart of this codebase's standing "an antagonist alone is inert
#: without endogenous tone" caveat; see the module docstring.
CB1_ANTAGONIST_ALONE_IPSC_FRACTION = 1.15

#: The `connection_modulation` gain. NEGATIVE and MEASURED: the engine forms
#: `1 + gain * activation`, so `gain = -CB1_MAX_SUPPRESSED_FRACTION` makes a
#: saturating full agonist leave `1 - 0.47 = 0.53` of the synapse. This is
#: the second P5 module with no illustrative coupling gain, after P5.5, and
#: the first whose gain is a measurement rather than an algebraic identity.
CB1_CONNECTION_GAIN = -CB1_MAX_SUPPRESSED_FRACTION


def saturating_margin_of_the_measured_maximum():
    """How far above its own EC50 the 47% was measured, in EC50 units.

    36-fold, which is where the word "maximal" in
    `CB1_MAX_SUPPRESSED_FRACTION`'s docstring comes from. At 36x EC50 a
    one-site Hill-1 curve is at 97% of its maximum, so treating 47% as the
    plateau under-states the true plateau by at most ~3%.
    """
    return CB1_WIN55212_SATURATING_CONCENTRATION / CB1_WIN55212_IPSC_EC50


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, shared by THC and rimonabant, and
#: deliberately the same number as the `TOOL_COMPOUND_KE` of every module
#: from P5.1 on; a test asserts the equality. Do not cite it as a half-life
#: for either drug -- THC's real terminal half-life is famously long, and
#: nothing in this module depends on that.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY, shared by 2-AG and anandamide, the same value
#: and broadly the same reason as the transmitter clearance constants before
#: it -- with one difference worth naming. The earlier proxies stood in for
#: a saturable transporter (DAT, SERT, NET, EAAT); the endocannabinoids are
#: instead synthesized on demand and hydrolysed in place, 2-AG by
#: monoacylglycerol lipase and anandamide by fatty-acid amide hydrolase, so
#: what this constant abstracts is an enzymatic sink rather than a
#: transporter. The two are given the SAME value even though their
#: hydrolases differ, because giving them different ones would assert a
#: clearance ORDERING that nothing sourced here supports. P6.2.
ENDOCANNABINOID_KE = 5.0


def pk_models_cb1():
    """One PK model per CB1 ligand, in `ligands_cb1()` order."""
    return [
        OneCompartmentPK(ke=ENDOCANNABINOID_KE),
        OneCompartmentPK(ke=ENDOCANNABINOID_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: the shared cortical loop, and the OTHER connection
# ---------------------------------------------------------------------------
#
# Two nodes and no new illustrative weight, exactly as in P5.5:
#
#   PFC_Pyramidal   pyramidal neurons. In [HoffmanLupica2000]'s and
#                   [Wilson2001]'s preparation this is the CA1 pyramidal
#                   cell, and it is the cell that RELEASES the retrograde
#                   messenger. It carries no receptor in this module.
#   PFC_GABA        local GABAergic interneurons. The CB1 receptor is on
#                   THIS node's axon terminals ([Katona1999]), which is why
#                   the modulated connection is keyed (PFC_Pyramidal,
#                   PFC_GABA) -- postsynaptic node first, presynaptic node
#                   second, per `engine.Simulation`'s convention.
#
# The weights, gain and baselines are character-for-character
# `glutamate.cortical_network`'s, which are in turn P5.1's, P5.3's and
# P5.4's minus their source-nucleus node.
#
# EXACTLY ONE of the two connections is GABAergic: PFC_GABA -> PFC_Pyramidal
# at -0.9. The other, PFC_Pyramidal -> PFC_GABA at +0.8, is the
# glutamatergic one P5.5 modulates. So P5.5 and P5.6 modulate DISJOINT
# connections of the same loop, which is not a coincidence and is not a
# design convenience: it is what the signs of the connectivity matrix
# already mean, and it makes a combined glutamate+cannabinoid scenario
# well-posed without any new decision. A test asserts the disjointness.
#
# CB1 is also expressed on glutamatergic terminals, and that IS the same
# connection P5.5 modulates. It is not built here; see "What was
# deliberately not built" below the network.

#: ILLUSTRATIVE. Unchanged from every hand-built cortical loop here.
CORTICAL_INHIBITION_WEIGHT = -0.9
CORTICAL_RECURRENT_WEIGHT = 0.8

#: ILLUSTRATIVE, and the same value every hand-built scenario uses.
CORTICAL_GAIN = 4.0

#: The node parameters. `baseline_input` is fully determined by rule 1
#: (every node at rate 0.5 at rest); `stability.balanced_baselines`
#: reproduces both and a test asserts it.
_NODES = (
    # name,           tau,  baseline_input
    ("PFC_Pyramidal", 1.5, 0.45),
    ("PFC_GABA", 1.0, -0.4),
)

#: The one GABAergic connection in the loop, as the engine's
#: `connection_modulation` key: (postsynaptic, presynaptic). CB1 sits on the
#: presynaptic member's terminal.
GABAERGIC_CONNECTION = ("PFC_Pyramidal", "PFC_GABA")

#: The glutamatergic connection, which `glutamate.py` modulates and this
#: module does not touch. Named here only so a test can assert the two
#: modules are disjoint.
GLUTAMATERGIC_CONNECTION = ("PFC_GABA", "PFC_Pyramidal")


def cortical_network():
    """PFC_Pyramidal -| PFC_GABA cortical loop, unchanged from P5.5's."""
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

    As in P5.5, starting here rather than at zero makes a control run
    EXACTLY flat: with no drug the activation is exactly zero, every
    modulation factor is exactly 1, and the network is identically the
    drug-free one.
    """
    n = len(_NODES) if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# The receptor system: empty coupling, one modulated connection
# ---------------------------------------------------------------------------


def coupling_cb1():
    """EMPTY, on purpose.

    CB1 adds no drive to any node. It is presynaptic, so the only thing it
    can do in this model is change what an existing projection delivers, and
    that is `connection_modulation`'s job. The consequence for P3.4's
    saturation rule is the strongest possible form of it, the same one P5.5
    could state: this module cannot saturate a node, because it never adds
    anything to a node's input.
    """
    return {}


def connection_modulation_cb1(connections=None):
    """`{(post, pre): -0.47}` for each GABAergic connection modulated."""
    if connections is None:
        connections = [GABAERGIC_CONNECTION]
    return {key: CB1_CONNECTION_GAIN for key in connections}


def receptor_system_cb1(
    affinity="midpoint", thc_efficacy=CB1_EFFICACY_THC, connections=None
):
    """The CB1 stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_cb1(),
        receptor_site=receptor_site_cb1(
            affinity=affinity, thc_efficacy=thc_efficacy
        ),
        coupling=coupling_cb1(),
        connection_modulation=connection_modulation_cb1(connections),
        name="CB1",
    )


def cannabinoid_receptor_systems(
    affinity="midpoint", thc_efficacy=CB1_EFFICACY_THC, connections=None
):
    """The CB1 system alone, in list form for `Simulation(...)`."""
    return [
        receptor_system_cb1(
            affinity=affinity, thc_efficacy=thc_efficacy, connections=connections
        )
    ]


def modulated_weight(activation, base_weight=CORTICAL_INHIBITION_WEIGHT):
    """The effective inhibitory weight at a given CB1 activation.

    Written out because it is the whole module in one line and because the
    tests use it to check the engine reproduces it: the engine's
    `1 + gain * activation` with `gain = -0.47` has to equal
    `1 - 0.47 * activation`.
    """
    return base_weight * (1.0 + CB1_CONNECTION_GAIN * activation)


# ---------------------------------------------------------------------------
# What was deliberately not built
# ---------------------------------------------------------------------------
#
# * **CB1 on glutamatergic terminals.** Real, and in some regions the
#   dominant CB1 population. It would be one more entry in
#   `connection_modulation`, on GLUTAMATERGIC_CONNECTION -- the connection
#   P5.5 already modulates, so the engine would compose the two factors
#   multiplicatively and the machinery needs no change. It is not built
#   because the number that would set its gain, a measured maximal
#   suppression of an evoked EPSC by a saturating CB1 agonist in the same
#   preparation as [HoffmanLupica2000]'s IPSC measurement, was not found;
#   inventing one, or reusing the 47% measured on a different synapse,
#   would be exactly the substitution rule 2 forbids. Consequence: this
#   module models the DISINHIBITORY half of cannabinoid action only, and
#   therefore over-states the net excitatory effect of a CB1 agonist by
#   whatever the suppression of excitatory transmission would have removed.
#   `connections=` is a parameter of every constructor above so that the
#   day the number exists, adding it is a one-line change.
# * **The retrograde loop closed as a loop.** [Wilson2001] and
#   [OhnoShosaku2001] describe a feedback: the postsynaptic rate determines
#   endocannabinoid release, which determines the presynaptic suppression,
#   which changes the postsynaptic rate. This module has the second and
#   third links and not the first, because the first is
#   "rate -> transmitter concentration", which is exactly P6.1's
#   projection-system abstraction and is deliberately not reinvented here.
#   Until it exists, 2-AG and anandamide are dosed like drugs, and the
#   depolarization-induced part of depolarization-induced suppression of
#   inhibition is absent.
# * **CB2, and CB1's non-Gi signalling.** Out of the P5.6 brief.


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def cortical_output_readout():
    """Pyramidal firing: the node CB1 disinhibits.

    Deliberately the same name, node and weight as
    `serotonin.cortical_output_readout` and its counterparts in
    `noradrenaline`, `acetylcholine`, `glutamate` and `histamine`; a test
    asserts all six are equal.
    """
    return BehaviorReadout(name="cortical_output", weights={"PFC_Pyramidal": 1.0})


def cortical_interneuron_readout():
    """Local GABAergic interneuron firing.

    The node whose TERMINAL carries the receptor. Its own rate is not what
    CB1 changes -- presynaptic inhibition of release leaves the presynaptic
    cell's firing alone, which is precisely why this receptor needed
    `connection_modulation` rather than `coupling` -- so this readout moves
    only through the loop, as a second-order consequence of the pyramidal
    node's change. A test asserts that ordering.
    """
    return BehaviorReadout(name="cortical_interneuron", weights={"PFC_GABA": 1.0})
