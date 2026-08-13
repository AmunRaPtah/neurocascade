"""P5.6c: the adenosine A2A receptor system, and the caffeine problem.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter): caffeine's affinity
at A2A and at A1, in rat and in human; adenosine's A2A "affinity", with the
large caveat below that it is not a binding measurement and cannot be one;
and a measured association rate constant at A2A shared by derivation 3.

ILLUSTRATIVE: the five-node basal-ganglia network, which is
`scenarios.basal_ganglia_indirect_pathway_network()` reused unchanged; the
one mechanism gain; and the pharmacokinetics.

Why A2A, and why it is the only receptor in Phase 5 whose scenario is
already written
---------------------------------------------------------------------

A2A is the caffeine target, and its anatomy is the sharpest in this
codebase. [Schiffmann1991] showed by in situ hybridization that the
striatal A2 receptor is expressed EXCLUSIVELY by the enkephalinergic
medium spiny neurons and not by the substance-P or cholinergic ones: it is
a marker of the indirect pathway. [Ferre1993] then showed by microdialysis
in awake rats that A2 and D2 receptors are co-localized on that same
GABAergic striatopallidal neuron and act on it in opposition, A2
stimulation inhibiting and A2 blockade potentiating the effects of D2
stimulation.

`scenarios.basal_ganglia_indirect_pathway_network()` already has that
neuron as a node, `Striatum_D2_MSN`, and already carries a D2 receptor
system on it. So A2A is not given a new circuit here: it is added to the
one the D2 scenario has used since Track A, on the SAME node, with the
OPPOSITE sign, and both of those facts are measured rather than chosen.
That makes this the first Phase 5 module whose receptor lands on an
existing node of an existing scenario rather than on a loop drawn for it,
and it is the first time two receptor systems in this codebase share a node
because an experiment says they share a neuron.

Sources
-------

Affinities. Both come from the IUPHAR/BPS adenosine-receptor nomenclature
report, which is a compilation rather than a primary paper, and that is a
deliberate choice explained under "Why a compilation":

  [Fredholm2011] Fredholm BB, IJzerman AP, Jacobson KA, Linden J, Muller CE.
                "International Union of Basic and Clinical Pharmacology.
                LXXXI. Nomenclature and classification of adenosine
                receptors -- an update." Pharmacol Rev. 2011;63(1):1-34.
                doi:10.1124/pr.110.003285 (PMC3061413, open access).
                Its Table 3 (agonists) row 1, footnoted "Data are from
                functional studies" and attributed to Yan et al., 2003:
                  ADENOSINE   A1 ~100 (h)   A2A 310 (h)   A2B 15,000 (h)
                                            A3 290 (h)
                              A1   73 (r)   A2A 150 (r)   A2B  5,100 (r)
                                            A3 6,500 (r)
                Its Table 4 (antagonists) row 32:
                  CAFFEINE    A1 10,700 (h) [Jacobson et al., 1999]
                              A2A 23,400 (h) [Abo-Salem et al., 2004]
                              A2B 33,800 (h) [Borrmann et al., 2009]
                              A3 13,300 (h) [Jacobson et al., 1999]
                              A1 41,000 (r) [Grahner et al., 1994]
                              A2A 45,000 (r) [Daly et al., 1991]
                Every one of those primary papers was checked for reach and
                every one is behind a paywall (Karger for [Daly1991], ASPET
                for [AboSalem2004]); the values above are therefore this
                review's tabulation of them, with the review's own
                attribution recorded so a future pass can replace any single
                entry without re-deriving the set.
  [Guo2012]     Guo D, Mulder-Krieger T, IJzerman AP, Heitman LH.
                "Functional efficacy of adenosine A2A receptor agonists is
                positively correlated to their receptor residence time." Br
                J Pharmacol. 2012;166(6):1846-1859.
                doi:10.1111/j.1476-5381.2012.01897.x (PMC3402809, open
                access). HUMAN A2A in HEK293 membranes, competition
                association against [3H]ZM241385, 5 C, 25 mM Tris-HCl pH
                7.4 + 5 mM MgCl2 + 0.1% CHAPS, n = 3 in duplicate. Table 3,
                the two rows used here:
                  NECA       kon 5.0 +/- 0.6 x 10^5 M^-1 min^-1
                             koff 0.03 +/- 0.01 min^-1, RT 35 min,
                             kinetic KD 58 nM
                  CGS21680   kon 5.0 +/- 1.0 x 10^4 M^-1 min^-1
                             koff 0.02 min^-1, RT 53 min, KD 380 nM
                and the panel's full kon span, 5.0e4 to 1.6e7 M^-1 min^-1
                across ten agonists, i.e. 320-fold, which is the same
                warning about derivation 3 that P5.3 took from [Sykes2022]
                (cited in full in `noradrenaline.py`)
                and `histamine.py` from [Bosma2017].

Mechanism and anatomy (no binding numbers taken from these):

  [Schiffmann1991] Schiffmann SN, Jacobs O, Vanderhaeghen JJ. "Striatal
                restricted adenosine A2 receptor (RDC8) is expressed by
                enkephalin but not by substance P neurons: an in situ
                hybridization histochemistry study." J Neurochem.
                1991;57(3):1062-1067.
                doi:10.1111/j.1471-4159.1991.tb08257.x. PMID:1713612. RAT
                caudate-putamen. The reason A2A goes on `Striatum_D2_MSN`
                and on no other node of that network.
  [Ferre1993]   Ferre S, O'Connor WT, Fuxe K, Ungerstedt U. "The
                striopallidal neuron: a main locus for adenosine-dopamine
                interactions in the brain." J Neurosci.
                1993;13(12):5402-5406.
                doi:10.1523/JNEUROSCI.13-12-05402.1993 (PMC6576405).
                PMID:8254382. Awake freely-moving RATS, intrastriatal
                microdialysis with GABA measured in the globus pallidus:
                the striopallidal terminal's GABA release is where A2-D2
                interaction happens, A2 stimulation and D2 blockade acting
                in the same direction and A2 blockade opposing D2 blockade.
                The reason the A2A gain is POSITIVE on a node whose D2 gain
                is negative.
  [Ledent1997]  Ledent C, Vaugeois JM, Schiffmann SN, Pedrazzini T, El
                Yacoubi M, Vanderhaeghen JJ, Costentin J, Heath JK, Vassart
                G, Parmentier M. "Aggressiveness, hypoalgesia and high blood
                pressure in mice lacking the adenosine A2a receptor."
                Nature. 1997;388(6643):674-678. doi:10.1038/41771.
                PMID:9262401. MOUSE. The causal test of the whole story:
                in A2A-knockout mice caffeine does not merely lose its
                stimulant effect on exploratory activity, it BECOMES A
                DEPRESSANT. Cited for the direction this module's scenario
                has to reproduce, and for the fact that it is not the whole
                story -- see "The A1 problem".

Why a compilation, when the standing rule prefers primaries
-----------------------------------------------------------

The rule is to prefer primary sources when they can be reached, and the
record of trying is part of the result. For caffeine, the four primaries
[Fredholm2011] names are all paywalled and none was reachable. For
ADENOSINE the situation is different and more interesting, because there is
no primary to reach: [Fredholm2011] states, in the text accompanying
Table 3, that

    "it is exceedingly difficult to determine this affinity, because
    adenosine is rapidly metabolized and rapidly formed in biological
    preparations, including membrane preparations. Therefore, if metabolism
    of adenosine is prevented, endogenous adenosine accumulates to confound
    the measurements"

and concludes that one "must rely on determination of the potency of
adenosine in functional assays (see an estimate based on functional studies
in Table 3)". The table's own footnote a says the adenosine row is
functional data. So:

**The endogenous ligand of this receptor has no measured binding affinity,
and the reason is a property of the ligand rather than a gap in the
literature.** Every number in `A2A_KD_ADENOSINE_*` below is a functional
potency being used as a Kd. That is the same approximation P5.2 to P5.5
made whenever a functional EC50 stood in for an affinity, and it is flagged
here in the strongest terms because for this receptor it is not optional.

Which species set, and why the answer is rat
--------------------------------------------

Both a rat and a human set are defined and `SPECIES_SETS` selects between
them. The default is RAT, by the preparation-match rule P5.3 established:
every mechanism paper this module depends on is rodent ([Schiffmann1991]
rat, [Ferre1993] rat, [Ledent1997] mouse), the circuit is the rodent basal
ganglia, and the roadmap's target species is mouse. Choosing the human set
would pair rodent anatomy with human pharmacology for no reason.

The cost is stated: [Guo2012]'s measured kon, which derivation 3 shares to
both ligands, is from the HUMAN receptor. So the default set is rat
affinities with a human association rate, and what that means precisely is
that the AFFINITIES are rat and exact by construction while the TIMESCALE
on which occupancy approaches them is human-derived. No result below is
stated in terms of a rate.

The species choice matters less than it might: caffeine's A2A affinity is
45,000 nM in rat and 23,400 nM in human, 1.9-fold apart, and adenosine's is
150 against 310 nM, 2.1-fold apart -- in the SAME direction, so the
caffeine-to-adenosine affinity RATIO is 300-fold in rat and 75-fold in
human. That 4-fold difference in the ratio is the one thing the species
choice does change, and the tests assert every qualitative result under
both sets.

The A1 problem, which is the honest headline of this module
-----------------------------------------------------------

**Caffeine is not an A2A drug.** In rat its affinity is 41,000 nM at A1 and
45,000 nM at A2A: 1.1-fold apart, i.e. non-selective within the error of
any single measurement. In human it is 10,700 against 23,400, and there it
is 2.2-fold A1-PREFERRING. On no reading is caffeine A2A-selective.

This model has an A2A receptor and no A1 receptor, so a caffeine dose here
produces the A2A-mediated half of caffeine's action and none of the
A1-mediated half. Three consequences, all recorded:

1. **The dose-response is wrong even where the direction is right.** A
   caffeine concentration that half-occupies A2A also half-occupies A1, and
   in this model the second does nothing. Any potency statement from this
   module is a statement about a hypothetical A2A-selective xanthine.
2. **The direction is nevertheless the right one to test against**, because
   [Ledent1997]'s A2A-knockout result is a direct causal demonstration that
   the STIMULANT component specifically requires A2A. What A1 blockade
   contributes is real but is not what the knockout removed.
3. **A1 is not in the P5.6 brief and is not smuggled in.**
   `A2A_KD_CAFFEINE_A1_*` are defined and are used by exactly one function,
   `caffeine_a1_a2a_selectivity()`, which exists so the non-selectivity can
   be asserted by a test rather than described in prose. No ligand is built
   from them.

Derivations
-----------

1. **Both constants measured** -- not available for either ligand here.
2. **koff and an affinity measured, kon missing** -- not used here.
3. **Only an affinity (or here, a potency) measured** -- share the site's
   measured `kon` and set `koff = kon * Kd`. Both ligands.

Derivation 3 is applied in P5.3's refined form, "share the measured kon of
the ligand chemically closest to the one being derived, and say which":

  * ADENOSINE takes NECA's kon (5.0e5 M^-1 min^-1). NECA is
    5'-N-ethylcarboxamidoadenosine -- adenosine with one amide substitution
    -- so this is the closest chemotype match derivation 3 has been given
    anywhere in this codebase, closer than P5.3's isoprenaline-for-
    noradrenaline. The resulting koff, 0.075 min^-1, is within 2.5-fold of
    NECA's own MEASURED koff of 0.03 min^-1, which is the internal check.
  * CAFFEINE takes the same kon, and this one crosses a chemotype boundary
    (adenosine nucleoside to xanthine). On [Guo2012]'s own evidence -- a
    320-fold kon span across its ten agonists -- that rate could be wrong
    by two orders of magnitude. The AFFINITY it encodes is exact whatever
    it is, and the module's results are equilibrium results.

Searched and empty
------------------

* **A primary binding affinity for adenosine at A2A.** Does not exist, for
  the reason [Fredholm2011] gives verbatim above. This is the cleanest
  searched-and-empty in Phase 5: the search terminates not in "not found"
  but in a published explanation of why it cannot be found.
* **Binding kinetics for caffeine, at any adenosine receptor.**
  [Guo2012]'s panel is agonists; its antagonist counterpart in the same
  laboratory characterizes [3H]ZM241385 rather than the xanthines. Searched
  for caffeine crossed with association/dissociation rate constants and
  residence time and found nothing usable. Hence derivation 3 across a
  chemotype for caffeine.
* **A rodent A2A-mediated dose-response usable as a circuit-layer target.**
  [Ledent1997] gives a knockout-versus-wild-type direction rather than a
  dose-response, and the caffeine locomotor literature is dose-based
  (mg/kg, IP) with no caffeine kinetics in `drug_pk.py` to bridge it. Same
  class of blocker Track E and P2.2 established. Caffeine is, however, an
  unusually good Phase 7 candidate precisely because [Ledent1997] provides
  a genetic control arm.
* **Rodent pharmacokinetics for caffeine.** Published and not adopted, for
  P5.3's two reasons. `TOOL_COMPOUND_KE` is shared and a test asserts it.

What was deliberately not built
-------------------------------

* **A1, A2B, A3.** Out of the brief. A1's absence is the module's largest
  quantified weakness and is recorded above rather than hidden.
* **The A2A-D2 receptor heteromer.** [Ferre1993]'s interaction is
  frequently modelled as an allosteric interaction within an A2A-D2
  heteromer, in which A2A occupancy REDUCES D2's affinity for dopamine.
  This engine has no representation of one receptor changing another's
  affinity, and adding one for a single scenario would be a large change
  justified by a mechanism that is still argued about. What is built
  instead is the uncontroversial part: two receptors on one node with
  opposite-signed gains, which reproduces the functional opposition
  [Ferre1993] measured without asserting its molecular basis. Said plainly
  so that an A2A/D2 result from this codebase is not read as a claim about
  heteromers.
* **Endogenous adenosine tone.** The same gap as every module before this
  one, and a particularly consequential one here: extracellular adenosine
  in the striatum is not zero, caffeine's entire mechanism is the removal
  of an ongoing adenosine action, and this model has no ongoing action to
  remove. So caffeine given ALONE is inert here. That is the standing P6.2
  limitation and it is, for caffeine, the difference between a model of the
  drug and a model of the receptor. The test that asserts the inertness
  says so.
"""

import numpy as np

from .behavior import BehaviorReadout
from .engine import ReceptorSystem
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite
from .scenarios import basal_ganglia_indirect_pathway_network


# ---------------------------------------------------------------------------
# Affinities
# ---------------------------------------------------------------------------

#: nM. [Fredholm2011] Table 3, adenosine row, RAT: A2A 150. FOOTNOTED IN
#: THAT TABLE AS FUNCTIONAL DATA, not binding. See "Why a compilation" in
#: the module docstring: adenosine's binding affinity at its own receptors
#: cannot be measured, because adenosine is metabolized and regenerated in
#: any preparation used to measure it, and that review says so explicitly.
A2A_KD_ADENOSINE_RAT = 150.0

#: nM. The same table's HUMAN value: 310.
A2A_KD_ADENOSINE_HUMAN = 310.0

#: nM. The same table's A1 values for adenosine, rat and human. Not used to
#: build any ligand; recorded because adenosine is 2.1-fold (rat) to 3.1-
#: fold (human) TIGHTER at A1 than at A2A, so the receptor this module
#: builds is not the one the endogenous ligand engages first.
A1_KD_ADENOSINE_RAT = 73.0
A1_KD_ADENOSINE_HUMAN = 100.0

#: nM. [Fredholm2011] Table 4, caffeine row, A2A. Rat 45,000 attributed to
#: Daly et al. 1991; human 23,400 attributed to Abo-Salem et al. 2004.
A2A_KD_CAFFEINE_RAT = 45000.0
A2A_KD_CAFFEINE_HUMAN = 23400.0

#: nM. The same row's A1 values. Rat 41,000 (Grahner et al., 1994); human
#: 10,700 (Jacobson et al., 1999). Used by `caffeine_a1_a2a_selectivity()`
#: only, and it is the number that makes "caffeine is an A2A drug" false.
A2A_KD_CAFFEINE_A1_RAT = 41000.0
A2A_KD_CAFFEINE_A1_HUMAN = 10700.0

#: The species sets `ligands_a2a` can build. Default rat; see the module
#: docstring for the preparation-match argument and for what it costs.
SPECIES_SETS = ("rat", "human")


def _species_pair(rat, human, species):
    if species not in SPECIES_SETS:
        raise ValueError(
            "species must be one of %s, got %r" % (list(SPECIES_SETS), species)
        )
    return rat if species == "rat" else human


def adenosine_kd(species="rat"):
    """Adenosine's A2A "affinity", in nM. A FUNCTIONAL POTENCY; see above."""
    return _species_pair(A2A_KD_ADENOSINE_RAT, A2A_KD_ADENOSINE_HUMAN, species)


def caffeine_kd(species="rat"):
    """Caffeine's A2A affinity, in nM."""
    return _species_pair(A2A_KD_CAFFEINE_RAT, A2A_KD_CAFFEINE_HUMAN, species)


def caffeine_a1_a2a_selectivity(species="rat"):
    """`Kd(A1) / Kd(A2A)` for caffeine: >1 means A1-preferring.

    0.91 in rat and 0.46 in human. Neither is a selectivity; both say
    caffeine engages A1 at least as readily as A2A, and the human value says
    it prefers A1 by 2.2-fold. This function exists so that "caffeine is not
    an A2A-selective drug" is a test rather than a paragraph. A1 is not
    modelled, and the consequences are enumerated in the module docstring.
    """
    a1 = _species_pair(A2A_KD_CAFFEINE_A1_RAT, A2A_KD_CAFFEINE_A1_HUMAN, species)
    return a1 / caffeine_kd(species)


def caffeine_over_adenosine_affinity_ratio(species="rat"):
    """How much weaker caffeine is than adenosine at A2A.

    300-fold in rat, 75-fold in human. This is the quantity that decides
    what a caffeine dose has to be to compete with transmitter once P6.2
    supplies transmitter, and its 4-fold species dependence is the one real
    consequence of the `SPECIES_SETS` choice.
    """
    return caffeine_kd(species) / adenosine_kd(species)


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. MEASURED association rate constant of NECA at the HUMAN
#: A2A receptor, competition association against [3H]ZM241385 in HEK293
#: membranes at 5 C [Guo2012] Table 3: 5.0 +/- 0.6 x 10^5 M^-1 min^-1, i.e.
#: 5.0e5 * 1e-9. Shared by both ligands (derivation 3). For adenosine that
#: is the closest chemotype match in this codebase (NECA is adenosine with
#: a 5'-N-ethylcarboxamide); for caffeine it crosses from nucleoside to
#: xanthine and could be wrong by ~100-fold, as [Guo2012]'s own 320-fold
#: kon span across its panel shows.
A2A_KON_NECA = 5.0e-4

#: min^-1. NECA's MEASURED dissociation rate constant in the same table,
#: residence time 35 min. Not used to build a ligand; it is the check on
#: derivation 3 -- adenosine's derived koff comes out within 2.5-fold of it.
A2A_KOFF_NECA = 0.03

#: nM. NECA's kinetically derived KD from the same table, `koff/kon`.
A2A_KD_NECA_KINETIC = 58.0

#: MEASURED kon values, in nM^-1 min^-1, of the extremes of [Guo2012]'s
#: ten-agonist panel: CGS21680 at 5.0e4 M^-1 min^-1 and LUF5835 at
#: 1.6e7 M^-1 min^-1. Kept as constants because a test asserts the 320-fold
#: span that makes derivation 3 a flagged assumption here.
A2A_KON_CGS21680 = 5.0e-5
A2A_KON_LUF5835 = 1.6e-2


# ---------------------------------------------------------------------------
# Efficacy
# ---------------------------------------------------------------------------

#: Adenosine's efficacy. 1.0 by DEFINITION: it is the endogenous agonist and
#: the reference every A2A functional assay normalizes to. Same standing as
#: histamine's 1.0 at H1 and noradrenaline's at alpha1A.
A2A_EFFICACY_ADENOSINE = 1.0

#: Caffeine's efficacy. 0.0, the same MODELLING SIMPLIFICATION recorded five
#: times before this module. Caffeine is conventionally described as a
#: competitive antagonist at A2A rather than as an inverse agonist, so this
#: is the mildest instance of the simplification in the codebase; it is
#: still flagged, because A2A is constitutively active and a xanthine's
#: measured effect on basal activity is not identically zero.
A2A_EFFICACY_CAFFEINE = 0.0


def ligands_a2a(species="rat"):
    """adenosine / caffeine at A2A, in that order."""
    adenosine = Ligand(
        name="adenosine",
        kon=A2A_KON_NECA,  # derivation 3, closest available chemotype
        koff=A2A_KON_NECA * adenosine_kd(species),
        efficacy=A2A_EFFICACY_ADENOSINE,
    )
    caffeine = Ligand(
        name="caffeine",
        kon=A2A_KON_NECA,  # derivation 3, across a chemotype boundary
        koff=A2A_KON_NECA * caffeine_kd(species),
        efficacy=A2A_EFFICACY_CAFFEINE,
    )
    return [adenosine, caffeine]


def receptor_site_a2a(species="rat"):
    """One competitive A2A pool, on the striatopallidal MSN.

    ONE pool and ONE node, and for once that is a measurement rather than a
    modelling convenience: [Schiffmann1991] found the striatal A2 receptor
    exclusively on enkephalinergic neurons, so of the five nodes in the
    basal-ganglia network this receptor belongs on exactly one. A2A is also
    expressed outside the striatum, at much lower density; nothing here
    represents that.
    """
    return ReceptorSite(
        name="A2A_on_striatopallidal_MSN", ligands=ligands_a2a(species=species)
    )


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min for caffeine, deliberately the same
#: number as every other module's `TOOL_COMPOUND_KE`; a test asserts the
#: equality. Do not cite it as caffeine's half-life in any species.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY for adenosine, the same value and the same
#: reason as the transmitter clearance constants before it. Extracellular
#: adenosine is cleared by equilibrative nucleoside transporters and by
#: adenosine deaminase and adenosine kinase, i.e. by saturable transport
#: plus enzymatic removal, on a sub-second-to-second timescale, not by
#: first-order elimination -- which is the SAME property that makes its
#: binding affinity unmeasurable (see the module docstring). 5.0 min^-1
#: (t1/2 ~8.3 s) preserves the only property the dynamics need, that the
#: transmitter clears far faster than the dosed drug, at a timescale the
#: integrator resolves. P6.2.
ADENOSINE_KE = 5.0


def pk_models_a2a():
    """One PK model per A2A ligand, in `ligands_a2a()` order."""
    return [
        OneCompartmentPK(ke=ADENOSINE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: reused, not built
# ---------------------------------------------------------------------------

#: The node A2A sits on. Named as a constant because the whole anatomical
#: argument of this module reduces to this one string agreeing with
#: `scenarios.d2_coupling()`'s, and a test asserts that agreement.
A2A_NODE = "Striatum_D2_MSN"


def striatal_network():
    """`scenarios.basal_ganglia_indirect_pathway_network()`, unchanged.

    Deliberately the same object, not a copy: Striatum_D2_MSN -| GPe -| STN
    -> GPi -| Thalamus, with the weights, taus, gains and baselines the D2
    scenario has used since Track A. Reusing it is what makes an A2A result
    comparable with a D2 result, and it means this module introduces no new
    illustrative circuit-layer number at all -- the second module in Phase 5
    able to say that, after P5.5, and by a different route (P5.5 added no
    coupling gain; this one adds no network).
    """
    return basal_ganglia_indirect_pathway_network()


def rest_rates(network=None):
    """The drug-free fixed point of that network: 0.5 at every node.

    Not a coincidence and not chosen here. The D2 scenario's hand-set
    `baseline_input` values `(0, 1, 1, -1, 1)` are exactly what
    `stability.balanced_baselines` derives for `r* = 1/2` given those
    weights, which is the credential that function's own docstring cites,
    so the network this module reuses already sits at the maximum-
    sensitivity point with no drug on board. `stability.steady_state`
    returns it with a residual of exactly 0.

    Passing `rates0=rest_rates()` makes a drug-free run exactly flat and
    every deviation in a dosed run the drug, the same habit P5.2, P5.3 and
    P5.5 adopted. It matters more here than elsewhere: without it the first
    minutes of every trace are a start-up transient, and comparing a dosed
    run against a control then compares two transients whose integrator
    step sequences differ.

    ONE PROPERTY OF THIS NETWORK IS WORTH RECORDING RATHER THAN INHERITING
    SILENTLY: its contraction number is 2.0, not the < 1 that P3.4's
    global-uniqueness certificate needs, because its weights are +/-2.0
    rather than the cortical loop's +/-0.9. So this fixed point is a fixed
    point and is the one every run in this module converges to, but it is
    not CERTIFIED unique the way the cortical loop's is. That is a property
    of the basal-ganglia network as the D2 scenario has had it since Track
    A, and this module changes no weight of it; a test asserts the number so
    the fact is on the record rather than discovered later.
    """
    n = 5 if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE. How much a unit of A2A activation changes the drive of the
#: striatopallidal node. POSITIVE: A2A is Gs/Golf-coupled, and the
#: functional result it has to reproduce is [Ferre1993]'s -- A2 stimulation
#: acts on the striatopallidal neuron in the direction OPPOSITE to D2
#: stimulation, which in this network carries a gain of -3.0.
#:
#: Magnitude 1.0 by P5.1's per-receptor unit rule, and NOT 3.0. That is a
#: deliberate refusal, and it is the one place a reader could be misled:
#: giving A2A the same magnitude as D2 with the opposite sign would look
#: like the natural encoding of "they oppose each other", and it would be a
#: claim that the two receptors are equally powerful on that neuron. Nothing
#: measured supports that claim. [Ferre1993] establishes the DIRECTION of
#: the interaction, not its relative magnitude. So the two gains differ in
#: magnitude, the difference is an artefact of two different illustrative
#: choices made at different times, and no A2A-versus-D2 potency comparison
#: should be read out of this module. A test states that explicitly.
#:
#: Not identifiable from anything published (P2.3), and it is the
#: `mechanism_gain` half of P4.2's `coupling = mechanism_gain * rho`.
A2A_MECHANISM_GAIN = 1.0


def coupling_a2a():
    """A2A activation -> drive, on the striatopallidal node."""
    return {A2A_NODE: A2A_MECHANISM_GAIN}


def receptor_system_a2a(species="rat"):
    """The A2A stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models_a2a(),
        receptor_site=receptor_site_a2a(species=species),
        coupling=coupling_a2a(),
        name="A2A",
    )


def adenosine_receptor_systems(species="rat"):
    """The A2A system alone, in list form for `Simulation(...)`."""
    return [receptor_system_a2a(species=species)]


def adenosine_and_dopamine_receptor_systems(species="rat"):
    """A2A and D2 on ONE node of ONE network, the [Ferre1993] arrangement.

    Uses `scenarios.d2_receptor_system()` unchanged, so the D2 half of this
    is Track A's calibrated mu-opioid-era parameter set and nothing here
    touches it. The two systems' `coupling` dicts both name
    `Striatum_D2_MSN` and the engine sums their contributions into that
    node's drive, which is the ordinary linear superposition
    `engine.Simulation` documents -- and which is, for once, physiologically
    exactly right: two receptors on one neuron feeding one membrane.
    """
    from . import scenarios

    return [receptor_system_a2a(species=species), scenarios.d2_receptor_system()]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def locomotor_activity_readout():
    """Thalamic firing, the output of the indirect pathway.

    Deliberately the same name, node and weight as
    `scenarios.locomotor_activity_readout`, and a test asserts the two are
    equal: this module is adding a receptor to that scenario, not proposing
    a different readout of it. The name is inherited from Track A and is
    older than this codebase's rule about not naming behaviours before
    Phase 7; it is a circuit-level quantity, and the direction it moves
    under A2A blockade is what [Ledent1997] is cited for, not its
    magnitude.
    """
    return BehaviorReadout(name="locomotor_activity", weights={"Thalamus": 1.0})


def striatopallidal_output_readout():
    """Firing of the node A2A and D2 share.

    The node where the receptor arithmetic happens, as opposed to the
    thalamic node four synapses downstream where the sign has been flipped
    twice. Reported separately because the sign chain
    Striatum -| GPe -| STN -> GPi -| Thalamus is easy to get wrong by eye
    and a test checks it end to end.
    """
    return BehaviorReadout(
        name="striatopallidal_output", weights={A2A_NODE: 1.0}
    )
