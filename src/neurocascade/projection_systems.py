"""P6.1: the six neuromodulatory projection systems, with their citations.

`projection.py` holds the mechanism and no numbers; this module holds the
numbers, the nuclei, and the reasoning behind each one, in the same
relationship `opioid.py` has to `receptor.py`.

    from neurocascade import projection_systems as ps

    print(ps.timescale_report())               # the volume-transmission table
    field = ps.dopamine_projection(
        targets=[projection.ProjectionTarget("CP", {"SNc": 1.0, "VTA": 0.2}),
                 projection.ProjectionTarget("PL", {"VTA": 0.05})],
        rest_rate=0.5)

Species: mouse where the constant exists in mouse, and the species is stated
for every constant that is not.

======================================================================
THE VOLUME-TRANSMISSION TIMESCALE ASSUMPTION, WHICH IS WHAT P6.1 BUYS
======================================================================

The abstraction says a target region has ONE extracellular transmitter
concentration. That is a spatial approximation, and it is the load-bearing
one, so it is stated here as an argument with numbers rather than as a
modelling convenience.

**The claim is NOT that a region is diffusively well mixed.** It is not.
Dopamine's effective diffusion coefficient in striatal extracellular space is
`D* = 322 um^2/s` ([Dreyer2010], Table 1). A region is of order a
millimetre across, and `x^2 / (6 D*)` for `x = 1 mm` is 518 s, longer than
most simulations in this codebase run. Diffusion does not mix a region and this
model does not claim it does.

**The claim is that the transmitter field is spatially near-uniform at the
scale of the innervation mesh, and that the mesh is what a region-wise
average is an average over.** The relevant length is the spacing between
release sites, not the width of a region:

- striatal dopamine release-site density `rho = 0.104 um^-3` ([Dreyer2010],
  Table 1), i.e. a mean nearest-site spacing of `rho^(-1/3) = 2.1 um`;
- [Ozcete2024] independently: "On average, there is one dopamine release site
  every ~4 um of dopamine axon, which leads to an estimate of one site per
  25 um^3 of the striatum", i.e. 2.9 um. The two agree within 1.4-fold from
  different measurements.

Diffusion over 2.1 um takes `x^2 / (6 D*) = 2.3 ms`. So on any timescale
longer than a few milliseconds, every point in the tissue is within diffusive
reach of many release sites, and the concentration a receptor sees is the
mesh-averaged one. That average is what `ProjectionSystem` integrates.

**The chemistry agrees with the geometry.** Extracellular clearance of each
transmitter, expressed as the relaxation time constant of its own pool, is
also far below the circuit's:

    transmitter      clearance time constant   source
    dopamine         52 ms   (Km/Vmax)         [JohnJones2007], mouse CP
    serotonin        2.1 s   (Km/Vmax)         [JohnJones2007], mouse SNr
    noradrenaline    2.0 s   (1/k)             [Park2012], rat BNST
    acetylcholine    ~tens of ms (bounded)     [Ozcete2024]; ILLUSTRATIVE here
    histamine        not measured              ILLUSTRATIVE here

against a Wilson-Cowan population time constant of 1.0-1.5 min in every
scenario in this codebase. The smallest of those ratios (serotonin, 2.1 s
against 90 s) is still 42-fold.

**What follows, and what does not.**

- FOLLOWS: a region's transmitter concentration can be carried as one ODE
  state whose relaxation is fast compared with the circuit's, so the field
  tracks the source's firing with a lag that is negligible at circuit
  resolution. `test_projection_systems.py` asserts that conclusion the
  expensive way, by integrating and comparing against the algebraic steady
  state, rather than by asserting the ratio.
- FOLLOWS: there is no need for an explicit diffusion operator, and adding
  one at region resolution would be false precision: every structure in the
  connectome parcellation is orders of magnitude wider than the release-site
  spacing the field is smooth over.
- DOES NOT FOLLOW: that concentration gradients WITHIN a region are absent.
  They are not. Striatal dopamine is patchy at the 10-100 um scale, and
  [Ozcete2024] reviews the release-site heterogeneity that makes it so. The
  model's region-wise concentration is an average over that structure, and
  any prediction that depends on the variance rather than the mean is outside
  what this abstraction can say.
- DOES NOT FOLLOW: that the fast pool can be eliminated adiabatically. It
  could be, and P6.1 deliberately does not: keeping the concentration as a
  state is what lets a drug act on it (an uptake blocker changes the pool's
  own kinetics, not just its steady state) and is the same
  "everything reacts to everything else's current state" decision the top of
  docs/architecture.md records for the PK layer.

**Why volume transmission at all, rather than synapse-by-synapse.** The
ultrastructure says the transmitter that reaches a neuromodulatory receptor
mostly did not cross a synaptic cleft: for dopamine "only ~1% of the dopamine
varicosities have them [synaptic appositions]" ([Ozcete2024], reviewing the
quantitative EM literature), for serotonin "synaptic appositions are overall
rare", for noradrenaline they "are largely absent". A receptor-occupancy
model driven by a region-wise extracellular concentration is therefore not a
coarse stand-in for a synaptic model; for these three transmitters it is
closer to the anatomy than a synaptic model would be.

For ACETYLCHOLINE the same question is open, and it is recorded as open
rather than assumed to go the same way. [Sarter2009] reviews studies
concluding "the great majority of cholinergic terminals in the cortex of rats
and humans form synaptic contacts" alongside studies contradicting it, and
says plainly that "the reasons for the discrepancy between these studies
remain unclear". The acetylcholine field here is built on the same
abstraction as the other four because one uniform mechanism was worth more
than a per-transmitter special case, and NOT because the anatomy was found to
demand it. No histamine varicosity fraction was sought.

======================================================================
THE RELEASE LAW
======================================================================

`projection.ProportionalRelease` makes release rate proportional to firing
rate, normalized to 1 at the source node's own resting rate. Three notes:

1. It is a spike-count model of release: a fixed quantity of transmitter per
   action potential. That is exactly the assumption [Dreyer2010] builds its
   dopamine model on ("N0 = 3000" molecules per fusion event, a fixed release
   probability per spike, and tonic firing as "random Poisson spikes at an
   average rate of 4 Hz").
2. It is known to be imperfect. Short-term depression makes per-pulse release
   fall with frequency, and burst firing can release supralinearly.
   `projection.PowerLawRelease` exists so a test can price that, and no
   exponent other than 1.0 produces any number reported here.
3. **No firing rate in Hz enters any equation.** The model consumes the ratio
   `r / r_rest` of dimensionless Wilson-Cowan rates. The Hz values in
   `FIRING_RATES` below are cited context for what that dimensionless rest
   rate stands for, and a test asserts they are not used in any computation
   by checking that every reported result is unchanged when they are altered.

======================================================================
WHAT IS MEASURED HERE AND WHAT IS NOT
======================================================================

MEASURED, and reused from the modules that sourced them:
  dopamine Km/Vmax/basal, serotonin Km/Vmax/basal, noradrenaline clearance
  rate and (from P6.2) noradrenaline basal, and every uptake-inhibition
  constant -- all of P5.8's panel, through `monoamine_transporters.py`.
  histamine basal extracellular concentration ([Itoh1991]).
  acetylcholine basal extracellular concentration ([Vinson1997], added by
  P6.2).
  The volume-transmission geometry above ([Dreyer2010], [Ozcete2024]).
  The source nuclei's identities and CCF acronyms ([AllenCCF]).
  The firing-rate context ([Ozcete2024], [Devilbiss2011], [Takahashi2006]).

  **All five transmitters now carry a cited basal concentration**, which is
  P6.2's acceptance criterion; `endogenous_tone.BASAL_TONE` is the single
  registry, with each one's species, region, method and caveats attached, and
  `endogenous_tone.tone_report()` prints what each one buys at each receptor.

NOT MEASURED, and labelled:
  every PROJECTION DENSITY. P6.1 builds the abstraction; P6.3 is the task
  that derives densities from the Phase 3 connectome. Every density this
  module or `projection_scenarios.py` supplies carries
  `provenance="illustrative"` and `ProjectionSystem.density_report()` shows
  it.
  acetylcholine's and histamine's extracellular clearance rate constants.
  See "Searched and empty" below and `ILLUSTRATIVE_CLEARANCE`.

======================================================================
SEARCHED AND EMPTY
======================================================================

**A first-order extracellular clearance rate constant for acetylcholine in
any rodent brain preparation.** What exists is the enzyme's turnover number
("one molecule of AChE can hydrolyse 5,000 molecules of ACh per second",
[Sarter2009], citing the enzymology literature), which is not a clearance
rate constant without an extracellular AChE concentration, and functional
delays that BOUND the transient's lifetime at tens of milliseconds
([Ozcete2024]: the cholinergic-axon-stimulation to dopamine-release delay is
"~10 ms"; "M4-IPSCs have a lag of ~40 ms and a rise time of ~80 ms"). The
bound is used to justify the ILLUSTRATIVE value's order of magnitude and is
not presented as a measurement of it.

**A basal extracellular acetylcholine concentration corrected for probe
recovery.** Reviewed dialysate values "range between 0.1-3 nM" and are
recognised as near the detection limit without a cholinesterase inhibitor in
the perfusate, which itself perturbs the quantity being measured. A dialysate
concentration is not an extracellular concentration, no zero-net-flux
acetylcholine measurement was found, and the review is a review. So
`ACETYLCHOLINE_BASAL_UNIT_SCALE = 1.0` is a UNIT, exactly as P5.8's
`NORADRENALINE_BASAL_UNIT_SCALE` is, and a test asserts that every fold-change
this module reports is invariant to it.

**FOUND BY P6.2.** [Vinson1997] is the quantitative-microdialysis
measurement, in rat striatum with no esterase inhibitor: 18.4 +/- 11.8 nM.
P6.1's search looked for the phrase "zero net flux" and for reviews of basal
cholinergic tone; the paper that carries the number is titled as a study of
NEOSTIGMINE's effect on the extraction fraction, and reports the drug-free
control concentration in passing. The record above is left standing because
what it says about DIALYSATE values remains true and is why [DeBundel2008]'s
0.40 nM is used only as the lower bound it is. `ACETYLCHOLINE_BASAL`
replaces the unit scale, with the three caveats attached to it, and the
fold-invariance test is kept because it is what shows the port's conclusions
do not turn on the choice.

**A first-order extracellular clearance rate constant for histamine.**
[Cumming1991] measured the decline of extracellular histamine in rat striatum
after synthesis inhibition and found half-lives of "32 min and 7.7 h". Those
are FOUND AND DELIBERATELY NOT USED as a clearance rate: they measure how
fast the releasable pool runs down when synthesis stops, not how fast
released histamine leaves the extracellular space, and using them as the
latter would be a category error of a factor unknown. The same paper reports
that extracellular histamine was "insensitive to the addition of
tetrodotoxin" and fell only 20% in calcium-free medium, so a substantial part
of what a histamine microdialysis probe collects may not be neuronal at all;
that caveat is attached to `HISTAMINE_BASAL` as well.

**A mouse basal extracellular noradrenaline concentration**, still, exactly
as P5.8 recorded. NET is therefore built in the unit-scale form and its field
is a fold-of-basal field.

**PARTLY FOUND BY P6.2, in the wrong species.** [PanLai1995] measured
1.59 +/- 0.37 nM by zero net flux in the medial prefrontal cortex of
CONSCIOUS RATS, so the noradrenaline field now carries an absolute
concentration scale and `monoamine_transporters.BASAL_NORADRENALINE`
replaces the unit scale. The MOUSE value is still missing: [Ihalainen1999]
is the mouse comparison and reports dialysate rather than recovery-corrected
levels, which is a regional ordering and not a concentration.

**Discharge rates in Hz for identified cholinergic basal forebrain or
PPT/LDT neurons.** [Lee2005] and [Boucetta2014] are the juxtacellular
identified-cell recordings, and both are behind Journal of Neuroscience's
403 for the fetcher used here; the qualitative result (cholinergic basal
forebrain cells "discharge in bursts at maximal rates during active waking
and paradoxical sleep") was reachable and the numbers were not. Since the
model consumes no Hz value, this gap costs a line of documentation and
nothing else, and it is recorded rather than filled with a number from a
citing secondary source.
"""

import numpy as np

from . import monoamine_transporters as _mt
from . import projection as _projection
from .projection import (
    ProjectionSystem,
    ProjectionTarget,
    SourceNucleus,
)


CITATIONS = {
    "Dreyer2010": (
        "Dreyer JK, Herrik KF, Berg RW, Hounsgaard JD. Influence of phasic "
        "and tonic dopamine release on receptor activation. J Neurosci "
        "2010;30:14273-14283. Computational model of striatal dopamine "
        "volume transmission with no free parameters; Table 1 supplies the "
        "release-site density, the effective diffusion constant and the "
        "extracellular volume fraction used here."
    ),
    "Ozcete2024": (
        "Ozcete OD, Banerjee A, Kaeser PS. Mechanisms of neuromodulatory "
        "volume transmission. Mol Psychiatry 2024, PMC11540752. Open-access "
        "review; used for the release-site spacing cross-check, the "
        "asynaptic-varicosity fractions, the acetylcholine degradation and "
        "response-latency statements, and the tonic firing-rate ranges."
    ),
    "JohnJones2007": (
        "John CE, Jones SR. Voltammetric characterization of the effect of "
        "monoamine uptake inhibitors and releasers on dopamine and serotonin "
        "uptake in mouse caudate-putamen and substantia nigra slices. "
        "Neuropharmacology 2007;52:1596-1605, PMC2041899. C57BL/6 mouse, the "
        "source of this codebase's dopamine and serotonin Km and Vmax."
    ),
    "Park2012": (
        "Park J, Takmakov P, Wightman RM (as cited by P5.8). Noradrenaline "
        "clearance half-life 1.36 +/- 0.07 s in rat ventral BNST by in vivo "
        "voltammetry, which fixes Vmax/Km and nothing else."
    ),
    "Bosse2011": (
        "Bosse KE et al. J Neurochem 2011;120:385-395, PMC3385875. Mouse "
        "basal extracellular dopamine by zero net flux, 5.0 +/- 0.2 nM."
    ),
    "Mathews2004": (
        "Mathews TA et al. J Neurosci Methods 2004;140:169-181. Mouse basal "
        "extracellular serotonin by zero net flux, 2.8 +/- 1 nM in striatum."
    ),
    "Itoh1991": (
        "Itoh Y, Oishi R, Nishibori M, Saeki K. Characterization of histamine "
        "release from the rat hypothalamus as measured by in vivo "
        "microdialysis. J Neurochem 1991;56:769-774. Anaesthetized rat, "
        "medial hypothalamus: 'The basal HA output was almost constant ... "
        "the mean value being 7.1 pg/30 min'; 'extracellular HA concentration "
        "was assumed to be 7.8 nM, by a calculation from in vitro recovery "
        "through the dialysis membrane'. Potassium-evoked release was "
        "calcium-dependent (+170%)."
    ),
    "Cumming1991": (
        "Cumming P, Damsma G, Fibiger HC, Vincent SR. Characterization of "
        "extracellular histamine in the striatum and bed nucleus of the "
        "stria terminalis of the rat: an in vivo microdialysis study. "
        "J Neurochem 1991;56:1797-1803. Half-lives of 32 min and 7.7 h for "
        "the decline after synthesis inhibition (NOT a clearance rate; see "
        "the module docstring), and the tetrodotoxin/calcium caveat."
    ),
    "Devilbiss2011": (
        "Devilbiss DM, Waterhouse BD. Phasic and tonic patterns of locus "
        "coeruleus output differentially modulate sensory network function "
        "in the awake rat. J Neurophysiol 2011;105:69-87, PMC3023368. "
        "'Tonic activity is exemplified by stochastic discharge across a "
        "range of relatively slow rates (0.1-5.0 Hz)'; the authors' own "
        "units 'averaged 0.88 +/- 0.23 (SE) Hz under control, quiet resting "
        "conditions'."
    ),
    "Takahashi2006": (
        "Takahashi K, Lin JS, Sakai K. Neuronal activity of histaminergic "
        "tuberomammillary neurons during wake-sleep states in the mouse. "
        "J Neurosci 2006;26:10292-10298. Identified histamine neurons in "
        "non-anaesthetized head-restrained MICE: 'slow (<10 Hz) tonic, "
        "repetitive, irregular firing' in waking, silent in slow-wave and "
        "paradoxical sleep."
    ),
    "Sarter2009": (
        "Sarter M, Parikh V, Howe WM. Phasic acetylcholine release and the "
        "volume transmission hypothesis: time to move on. Nat Rev Neurosci "
        "2009;10:383-390, PMC2699581. Source of the acetylcholinesterase "
        "turnover figure ('one molecule of AChE can hydrolyse 5,000 "
        "molecules of ACh per second') and of the unresolved status of the "
        "cortical cholinergic synaptic-contact fraction."
    ),
    "Lee2005": (
        "Lee MG, Hassani OK, Alonso A, Jones BE. Cholinergic basal forebrain "
        "neurons burst with theta during waking and paradoxical sleep. "
        "J Neurosci 2005;25:4365-4369. Juxtacellularly identified cells; "
        "the paper's Hz values were not reachable (see 'Searched and "
        "empty')."
    ),
    "Vinson1997": (
        "Vinson PN, Justice JB Jr. Effect of neostigmine on concentration "
        "and extraction fraction of acetylcholine using quantitative "
        "microdialysis. J Neurosci Methods 1997;73:61-67, PMID:9130679. "
        "Rat STRIATUM, no-net-flux quantitative microdialysis: 'The control "
        "ACh concentration was determined to be 18.4 +/- 11.8 nM (n = 10; "
        "mean +/- S.E.M.)' with no cholinesterase inhibitor in the "
        "perfusate, rising linearly to 581 +/- 109 nM under 300 nM "
        "neostigmine. Added by P6.2; the source of `ACETYLCHOLINE_BASAL`."
    ),
    "DeBundel2008": (
        "De Bundel D, Sarre S, Van Eeckhaut A, Smolders I, Michotte Y. "
        "Critical evaluation of acetylcholine determination in rat brain "
        "microdialysates using ion-pair liquid chromatography with "
        "amperometric detection. Sensors 2008;8:5171-5185, PMC3705495. "
        "'The concentration of acetylcholine in 30 uL microdialysates "
        "obtained under basal conditions from the hippocampus of freely "
        "moving rats was 0.40 +/- 0.12 nM (mean +/- SD, n = 30)', with the "
        "authors' own estimate that 'the in vivo recovery for acetylcholine "
        "in microdialysis samples was approximately 10%'. The independent "
        "check on [Vinson1997], 4.6-fold below it."
    ),
    "PanLai1995": (
        "Pan WH, Lai YJ. Anesthetics decreased the microdialysis extraction "
        "fraction of norepinephrine but not dopamine in the medial "
        "prefrontal cortex. Synapse 1995;21:85-92, PMID:8525466. Conscious "
        "rat medial prefrontal cortex by ZERO NET FLUX: noradrenaline "
        "1.59 +/- 0.37 nM at 51 +/- 8% in vivo recovery, dopamine "
        "2.38 +/- 0.70 nM at 41 +/- 6%. Added by P6.2; the source of "
        "`monoamine_transporters.BASAL_NORADRENALINE`, and the only "
        "measured DOPAMINE concentration in this codebase from a region "
        "other than the striatum."
    ),
    "Abercrombie1988": (
        "Abercrombie ED, Keller RW Jr, Zigmond MJ. Characterization of "
        "hippocampal norepinephrine release as measured by microdialysis "
        "perfusion: pharmacological and behavioral studies. Neuroscience "
        "1988;27:897-904, PMID:3252176. Freely-moving rat hippocampus: "
        "'Correcting for recovery (21%), the concentration of norepinephrine "
        "in the extracellular fluid of hippocampus under resting conditions "
        "was estimated to be approximately 14 nM'. The cross-check on "
        "[PanLai1995], 8.8-fold above it. Its closing sentence is also the "
        "clearest statement of this phase's own premise: 'this suggests "
        "that nerve terminal density is an important factor in determining "
        "the concentration of catecholamines in the extracellular fluid'."
    ),
    "Ihalainen1999": (
        "Ihalainen JA, Riekkinen P Jr, Feenstra MG. Comparison of dopamine "
        "and noradrenaline release in mouse prefrontal cortex, striatum and "
        "hippocampus using microdialysis. Neurosci Lett 1999;277:71-74. "
        "MOUSE, dialysate rather than recovery-corrected: 'Basal release of "
        "NA was similar in these areas, but DA in striatum was 13-30 times "
        "higher than in other areas'. Consumed by no equation; it is the "
        "regional ORDERING that P6.3's density map has to reproduce."
    ),
    "AllenCCF": (
        "Allen Mouse Brain Common Coordinate Framework structure graph 1, "
        "the ontology already vendored at data/connectome/. Every source "
        "nucleus below is named by its CCF acronym and id so that P6.3 can "
        "look up its connectivity row without a second naming convention."
    ),
}


# ---------------------------------------------------------------------------
# Volume transmission: the geometry, and the arithmetic the docstring quotes
# ---------------------------------------------------------------------------

#: um^2/s. Effective (tortuosity-corrected) diffusion constant for dopamine
#: in striatal extracellular space. [Dreyer2010], Table 1: "D* = 322 um2/s".
DOPAMINE_EFFECTIVE_DIFFUSION = 322.0

#: Dimensionless. Extracellular volume fraction. [Dreyer2010], Table 1:
#: "alpha = 0.21". Reported for completeness; nothing here consumes it,
#: because the model carries concentrations rather than amounts.
EXTRACELLULAR_VOLUME_FRACTION = 0.21

#: um^-3. Density of dopamine release sites in striatum. [Dreyer2010],
#: Table 1: "rho = 0.104 um-3".
DOPAMINE_RELEASE_SITE_DENSITY = 0.104

#: um^3 per release site, from [Ozcete2024]: "one site per 25 um3 of the
#: striatum". The independent cross-check on the line above.
DOPAMINE_RELEASE_SITE_VOLUME_OZCETE = 25.0

#: min. The Wilson-Cowan population time constant every scenario in this
#: codebase uses for a cortical/striatal node, and the denominator of every
#: timescale ratio below. Not a new number: it is `tau` in the existing
#: networks.
CIRCUIT_TAU_MIN = 1.5


def release_site_spacing(density=DOPAMINE_RELEASE_SITE_DENSITY):
    """um. Mean nearest-release-site spacing, `density ** (-1/3)`.

    The length scale over which the transmitter field has to be smooth for a
    region-wise concentration to mean anything. 2.12 um for [Dreyer2010]'s
    density and 2.92 um for [Ozcete2024]'s, which is the 1.4-fold agreement
    the module docstring quotes.
    """
    return float(density) ** (-1.0 / 3.0)


def diffusion_time_s(distance_um, diffusion=DOPAMINE_EFFECTIVE_DIFFUSION):
    """s. `x^2 / (6 D)`: the characteristic time for three-dimensional
    diffusion over `distance_um`.

    The 6 rather than 2 is the three-dimensional mean-square-displacement
    convention, `<r^2> = 6 D t`. Using 2 (one dimension) would make every
    number here three times larger and would not change any conclusion,
    which is the point of stating the convention rather than the number.
    """
    return float(distance_um) ** 2 / (6.0 * float(diffusion))


#: min. Extracellular clearance time constant per transmitter, computed from
#: the constants the earlier phases sourced rather than restated here. For a
#: Michaelis-Menten transporter in the linear regime (which is where the
#: measured basal levels put every one of these) the pool relaxes with
#: `Km/Vmax`; for a first-order one with `1/k`.
def clearance_time_constant_min(clearance):
    """min. `Km/Vmax`, or `1/k` for a first-order clearance."""
    rate = float(clearance.effective_clearance_rate)
    return 1.0 / rate


def timescale_report():
    """The volume-transmission timescale table, computed, as text.

    Exists so the argument in the module docstring is a computation a reader
    can re-run rather than a paragraph. `scripts/report_projection_systems.py`
    writes it to `results/`.
    """
    lines = []
    lines.append("Volume-transmission timescales (P6.1)")
    lines.append("=" * 72)
    lines.append("")
    spacing_dreyer = release_site_spacing()
    spacing_ozcete = DOPAMINE_RELEASE_SITE_VOLUME_OZCETE ** (1.0 / 3.0)
    lines.append(
        "Mean dopamine release-site spacing: %.2f um [Dreyer2010], "
        "%.2f um [Ozcete2024] (%.2f-fold apart)"
        % (
            spacing_dreyer,
            spacing_ozcete,
            max(spacing_dreyer, spacing_ozcete) / min(spacing_dreyer, spacing_ozcete),
        )
    )
    lines.append(
        "Diffusion over that spacing at D* = %.0f um^2/s: %.1f ms"
        % (DOPAMINE_EFFECTIVE_DIFFUSION, 1e3 * diffusion_time_s(spacing_dreyer))
    )
    lines.append(
        "Diffusion across a 1 mm region at the same D*: %.0f s "
        "(so a region is NOT diffusively mixed; see the module docstring)"
        % diffusion_time_s(1000.0)
    )
    lines.append("")
    lines.append(
        "%-16s %14s %12s %10s" % ("transmitter", "clearance tau", "circuit tau", "ratio")
    )
    tau_circuit_s = 60.0 * CIRCUIT_TAU_MIN
    for name, builder in sorted(CLEARANCE_BUILDERS.items()):
        clearance = builder()
        tau_s = 60.0 * clearance_time_constant_min(clearance)
        lines.append(
            "%-16s %11.4g s %9.4g s %10.0f"
            % (name, tau_s, tau_circuit_s, tau_circuit_s / tau_s)
        )
    lines.append("")
    lines.append("ILLUSTRATIVE clearance rates: %s" % ", ".join(sorted(ILLUSTRATIVE_CLEARANCE)))
    lines.append("")
    lines.append("Source nuclei")
    lines.append("-" * 72)
    for key in NUCLEUS_ORDER:
        nucleus = NUCLEI[key]
        lines.append(
            "%-6s %-38s %-14s %s"
            % (
                nucleus["acronym"],
                nucleus["name"],
                nucleus["transmitter"],
                nucleus["firing"],
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Firing-rate context. NOT consumed by any equation; see the module docstring.
# ---------------------------------------------------------------------------

#: What the dimensionless resting rate of each source node stands for, in Hz,
#: with the species the measurement was made in. A test asserts that no
#: result in this module or in `projection_scenarios.py` changes when these
#: strings change, because the release law consumes the RATIO of
#: dimensionless Wilson-Cowan rates and nothing else.
FIRING_RATES = {
    "dopamine": (
        "tonic (pacemaker) 0.2-10 Hz, bursts >10 Hz [Ozcete2024]; "
        "[Dreyer2010] models tonic activity as Poisson spikes at 4 Hz"
    ),
    "serotonin": "tonic 0.03-3 Hz, bursts up to 17 Hz [Ozcete2024]",
    "noradrenaline": (
        "tonic 0.1-5.0 Hz, 0.88 +/- 0.23 Hz at quiet rest, awake rat "
        "[Devilbiss2011]"
    ),
    "acetylcholine": (
        "no reachable Hz value; identified cholinergic basal forebrain cells "
        "burst at maximal rates in active waking and paradoxical sleep "
        "[Lee2005]. SEARCHED AND EMPTY, see the module docstring"
    ),
    "histamine": (
        "slow (<10 Hz) tonic irregular firing in waking, silent in slow-wave "
        "and paradoxical sleep; MOUSE [Takahashi2006]"
    ),
}


# ---------------------------------------------------------------------------
# The nuclei
# ---------------------------------------------------------------------------

#: The eight source nuclei the brief names, by Allen CCF acronym and id.
#: `node` is the default circuit-node name a scenario gives the nucleus,
#: which is the CCF acronym: P6.3 has to look these rows up in the connectome
#: and a second naming convention would be a second place to be wrong.
NUCLEI = {
    "VTA": {
        "acronym": "VTA",
        "ccf_id": 749,
        "name": "Ventral tegmental area",
        "transmitter": "dopamine",
        "firing": FIRING_RATES["dopamine"],
    },
    "SNc": {
        "acronym": "SNc",
        "ccf_id": 374,
        "name": "Substantia nigra, compact part",
        "transmitter": "dopamine",
        "firing": FIRING_RATES["dopamine"],
    },
    "DR": {
        "acronym": "DR",
        "ccf_id": 872,
        "name": "Dorsal nucleus raphe",
        "transmitter": "serotonin",
        "firing": FIRING_RATES["serotonin"],
    },
    "CS": {
        "acronym": "CS",
        "ccf_id": 679,
        "name": "Superior central nucleus raphe (median raphe)",
        "transmitter": "serotonin",
        "firing": FIRING_RATES["serotonin"],
    },
    "LC": {
        "acronym": "LC",
        "ccf_id": 147,
        "name": "Locus ceruleus",
        "transmitter": "noradrenaline",
        "firing": FIRING_RATES["noradrenaline"],
    },
    "NDB": {
        "acronym": "NDB",
        "ccf_id": 596,
        "name": "Diagonal band nucleus (basal forebrain)",
        "transmitter": "acetylcholine",
        "firing": FIRING_RATES["acetylcholine"],
    },
    "PPN": {
        "acronym": "PPN",
        "ccf_id": 1052,
        "name": "Pedunculopontine nucleus",
        "transmitter": "acetylcholine",
        "firing": FIRING_RATES["acetylcholine"],
    },
    "LDT": {
        "acronym": "LDT",
        "ccf_id": 162,
        "name": "Laterodorsal tegmental nucleus",
        "transmitter": "acetylcholine",
        "firing": FIRING_RATES["acetylcholine"],
    },
    "TM": {
        "acronym": "TM",
        "ccf_id": 557,
        "name": "Tuberomammillary nucleus",
        "transmitter": "histamine",
        "firing": FIRING_RATES["histamine"],
    },
}

NUCLEUS_ORDER = ("VTA", "SNc", "DR", "CS", "LC", "NDB", "PPN", "LDT", "TM")

#: The brief's grouping: which nuclei feed which transmitter's field. The
#: basal forebrain and the PPT/LDT are listed as one acetylcholine field with
#: three sources because they are three nuclei releasing one transmitter into
#: overlapping territory, which is exactly the case `SourceNucleus` exists
#: for. Nothing here asserts that their density maps are the same, and they
#: are not: the basal forebrain innervates cortex and hippocampus, the
#: PPT/LDT the thalamus and the midbrain.
NUCLEI_BY_TRANSMITTER = {
    "dopamine": ("VTA", "SNc"),
    "serotonin": ("DR", "CS"),
    "noradrenaline": ("LC",),
    "acetylcholine": ("NDB", "PPN", "LDT"),
    "histamine": ("TM",),
}


# ---------------------------------------------------------------------------
# Clearance: measured where measured, ILLUSTRATIVE where not
# ---------------------------------------------------------------------------

#: nM. Basal extracellular histamine. [Itoh1991], ANAESTHETIZED RAT, medial
#: hypothalamus, "assumed to be 7.8 nM, by a calculation from in vitro
#: recovery through the dialysis membrane".
#:
#: Three caveats travel with this number and are not hidden: it is rat, not
#: mouse; it is the hypothalamus, i.e. the tuberomammillary nucleus's own
#: neighbourhood rather than a forebrain target; and it is a recovery-corrected
#: dialysate value rather than a zero-net-flux one, which is the method
#: [Bosse2011] and [Mathews2004] used for the two numbers this codebase
#: trusts most. [Cumming1991] adds a fourth: extracellular histamine in
#: striatum was tetrodotoxin-insensitive and only 20% calcium-dependent, so
#: some of what a probe collects may not be neuronal. It is used because it
#: is a real measurement of the right quantity and the alternative is a unit
#: scale, and `histamine_fold_rise_is_basal_independent` in the tests shows
#: which conclusions would survive either way.
HISTAMINE_BASAL = 7.8

#: nM. Basal extracellular acetylcholine, RAT STRIATUM, QUANTITATIVE
#: (no-net-flux) microdialysis with NO cholinesterase inhibitor in the
#: perfusate, [Vinson1997]: "The control ACh concentration was determined to
#: be 18.4 +/- 11.8 nM (n = 10; mean +/- S.E.M.)".
#:
#: P6.1 recorded this as searched-and-empty and used a unit scale of 1.0;
#: rule 3 says the label goes in the commit that supplies the calibration,
#: which is this one. Three caveats travel with it:
#:
#: * RAT, not mouse.
#: * STRIATUM, where extracellular acetylcholine comes from tonically active
#:   cholinergic INTERNEURONS rather than from the basal-forebrain and
#:   PPT/LDT projections this module's acetylcholine field is built from.
#:   That is a source mismatch, not only a region mismatch, and it is the
#:   weakest thing about this constant. It is used anyway because it is the
#:   only primary quantitative-microdialysis measurement of the right
#:   quantity that this project could reach, and because the alternative is
#:   a unit scale, which asserts nothing at all.
#: * The dispersion is large: +/- 11.8 on 18.4 with n = 10.
#:
#: The independent check is [DeBundel2008], which measured 0.40 +/- 0.12 nM
#: in hippocampal DIALYSATE of freely-moving rats without an esterase
#: inhibitor and put its own in vivo recovery for acetylcholine at ~10%,
#: i.e. ~4 nM extracellular: 4.6-fold below this number, in a different
#: region, by a method this codebase trusts less. `endogenous_tone.py`
#: reports every acetylcholine conclusion at BOTH values, so no claim here
#: rests on choosing between them.
ACETYLCHOLINE_BASAL = 18.4

#: nM. [DeBundel2008]'s hippocampal dialysate value. A DIALYSATE
#: concentration is a strict LOWER BOUND on the extracellular one, because
#: dialysate = extraction_fraction * extracellular and the extraction
#: fraction is at most 1. Recorded as the bound it is, and used only in the
#: sensitivity range above.
ACETYLCHOLINE_BASAL_DIALYSATE_LOWER_BOUND = 0.40

#: nM. The same paper's own recovery-corrected reading of that bound
#: ("approximately 10%" in vivo recovery). The low end of the acetylcholine
#: sensitivity range.
ACETYLCHOLINE_BASAL_HIPPOCAMPAL_ESTIMATE = 4.0

#: min^-1. ILLUSTRATIVE. Acetylcholine's extracellular clearance rate
#: constant is not measured (see "Searched and empty"). The order of
#: magnitude is taken from a cited BOUND rather than invented freely:
#: [Ozcete2024] reports a ~10 ms delay from cholinergic axon stimulation to
#: dopamine release and a ~40 ms lag with ~80 ms rise for M4-mediated IPSCs,
#: so an acetylcholine transient's lifetime is tens of milliseconds. A
#: half-life of 40 ms gives `ln(2)/40 ms = 17.3 s^-1 = 1040 min^-1`.
#: DO NOT CITE THIS AS A MEASURED CLEARANCE RATE. What it has to be right
#: about is only that the pool is fast compared with the circuit, and
#: `test_projection_systems.py` sweeps it over two orders of magnitude and
#: asserts no reported result moves.
ACETYLCHOLINE_CLEARANCE_HALF_LIFE_S = 0.040
ACETYLCHOLINE_CLEARANCE_RATE = 60.0 * np.log(2.0) / ACETYLCHOLINE_CLEARANCE_HALF_LIFE_S

#: min^-1. ILLUSTRATIVE. Histamine has no plasma-membrane reuptake carrier
#: and no measured extracellular clearance rate constant; [Cumming1991]'s
#: 32 min half-life measures pool run-down after synthesis inhibition and is
#: deliberately not used here (see "Searched and empty"). The value chosen is
#: the SLOWEST of the three measured monoamine clearance rates in this
#: codebase, serotonin's `Vmax/Km`, on the stated ground that a transmitter
#: cleared by an enzyme rather than a transporter is not cleared faster than
#: the slowest transporter-cleared one. That is a direction, not a
#: measurement, and it is the reason this constant is labelled rather than
#: cited. Same sweep test as acetylcholine's.
HISTAMINE_CLEARANCE_RATE = _mt.VMAX_SEROTONIN / _mt.KM_SEROTONIN

#: The two transmitters whose clearance rate constant is illustrative.
ILLUSTRATIVE_CLEARANCE = ("acetylcholine", "histamine")


def dopamine_clearance(release_capacity=_mt.RELEASE_CAPACITY, basal=None):
    """DAT, from P5.8. Km/Vmax [JohnJones2007], basal [Bosse2011], and the
    uptake-inhibition panel for cocaine, methylphenidate, amphetamine and
    atomoxetine.

    `basal` overrides the cited tone; see `monoamine_transporters.dat`.
    """
    return _mt.dat(release_capacity=release_capacity, basal=basal)


def serotonin_clearance(release_capacity=_mt.RELEASE_CAPACITY, basal=None):
    """SERT, from P5.8. Km/Vmax [JohnJones2007], basal [Mathews2004]."""
    return _mt.sert(release_capacity=release_capacity, basal=basal)


def noradrenaline_clearance(release_capacity=_mt.RELEASE_CAPACITY, basal=None):
    """NET, from P5.8, in the first-order form [Park2012] forces. Its basal
    is [PanLai1995]'s zero-net-flux measurement since P6.2, where P6.1 had a
    unit scale here."""
    return _mt.net(release_capacity=release_capacity, basal=basal)


def acetylcholine_clearance(rate=None, basal=None):
    """Acetylcholinesterase hydrolysis, as a first-order process.

    The rate is illustrative (`ACETYLCHOLINE_CLEARANCE_RATE`); the basal
    level is [Vinson1997]'s measurement (`ACETYLCHOLINE_BASAL`), with its
    three caveats. No ligand: this codebase sources no
    cholinesterase-inhibitor constant, and an empty ligand list is how a gap
    is represented here rather than an implied infinite Ki (P5.8's rule).
    """
    return _projection.FirstOrderClearance(
        name="AChE",
        transmitter="acetylcholine",
        rate=ACETYLCHOLINE_CLEARANCE_RATE if rate is None else rate,
        basal=ACETYLCHOLINE_BASAL if basal is None else basal,
    )


def histamine_clearance(rate=None, basal=None):
    """Histamine inactivation, as a first-order process.

    Rate illustrative (`HISTAMINE_CLEARANCE_RATE`); basal measured but with
    four stated caveats (`HISTAMINE_BASAL`).
    """
    return _projection.FirstOrderClearance(
        name="HNMT",
        transmitter="histamine",
        rate=HISTAMINE_CLEARANCE_RATE if rate is None else rate,
        basal=HISTAMINE_BASAL if basal is None else basal,
    )


CLEARANCE_BUILDERS = {
    "dopamine": dopamine_clearance,
    "serotonin": serotonin_clearance,
    "noradrenaline": noradrenaline_clearance,
    "acetylcholine": acetylcholine_clearance,
    "histamine": histamine_clearance,
}


# ---------------------------------------------------------------------------
# Building a projection system
# ---------------------------------------------------------------------------


def source_nuclei(transmitter, rest_rate, nodes=None):
    """The `SourceNucleus` list for one transmitter's field.

    `nodes` optionally renames the circuit nodes; by default a nucleus's node
    is its CCF acronym, which is what P6.3 will need. `rest_rate` is the
    dimensionless resting rate of those nodes, one value or one per nucleus.
    """
    keys = NUCLEI_BY_TRANSMITTER[transmitter]
    nodes = dict(nodes or {})
    rates = np.atleast_1d(np.asarray(rest_rate, dtype=float))
    if rates.size == 1:
        rates = np.full(len(keys), float(rates[0]))
    if rates.size != len(keys):
        raise ValueError(
            "%s has %d source nuclei, got %d resting rates"
            % (transmitter, len(keys), rates.size)
        )
    return [
        SourceNucleus(node=nodes.get(key, NUCLEI[key]["acronym"]), rest_rate=rate)
        for key, rate in zip(keys, rates)
    ]


def projection_system(
    transmitter,
    targets,
    rest_rate,
    nodes=None,
    name=None,
    clearance=None,
    pk_models=None,
    reference_region=None,
):
    """Build one transmitter's projection system.

    `targets` is the caller's, because a density map is scenario-specific
    until P6.3 supplies a real one; every `ProjectionTarget` built by
    `projection_scenarios.py` carries `provenance="illustrative"`.

    The clearance object's ligands come with it (DAT carries cocaine,
    methylphenidate, amphetamine and atomoxetine), so `pk_models` defaults to
    P5.8's tool-compound models for exactly those drugs.
    """
    if transmitter not in CLEARANCE_BUILDERS:
        raise ValueError(
            "no clearance model for transmitter %r; have %s"
            % (transmitter, sorted(CLEARANCE_BUILDERS))
        )
    if clearance is None:
        clearance = CLEARANCE_BUILDERS[transmitter]()
    if pk_models is None:
        pk_models = _mt.pk_models([lig.name for lig in getattr(clearance, "ligands", [])])
    return ProjectionSystem(
        name=name or transmitter,
        transmitter=transmitter,
        sources=source_nuclei(transmitter, rest_rate, nodes=nodes),
        targets=list(targets),
        clearance=clearance,
        pk_models=pk_models,
        reference_region=reference_region,
    )


def dopamine_projection(targets, rest_rate, **kwargs):
    """VTA + SNc -> one dopamine field. See `projection_system`."""
    return projection_system("dopamine", targets, rest_rate, **kwargs)


def serotonin_projection(targets, rest_rate, **kwargs):
    """DR + CS (median raphe) -> one serotonin field."""
    return projection_system("serotonin", targets, rest_rate, **kwargs)


def noradrenaline_projection(targets, rest_rate, **kwargs):
    """LC -> one noradrenaline field, in fold-of-basal units."""
    return projection_system("noradrenaline", targets, rest_rate, **kwargs)


def acetylcholine_projection(targets, rest_rate, **kwargs):
    """NDB + PPN + LDT -> one acetylcholine field, in fold-of-basal units."""
    return projection_system("acetylcholine", targets, rest_rate, **kwargs)


def histamine_projection(targets, rest_rate, **kwargs):
    """TM -> one histamine field."""
    return projection_system("histamine", targets, rest_rate, **kwargs)


PROJECTION_BUILDERS = {
    "dopamine": dopamine_projection,
    "serotonin": serotonin_projection,
    "noradrenaline": noradrenaline_projection,
    "acetylcholine": acetylcholine_projection,
    "histamine": histamine_projection,
}


def illustrative_targets(regions, provenance="illustrative"):
    """`[ProjectionTarget(region, density_map), ...]` from
    `{region: density_map}`, all tagged with one provenance.

    A one-line way for a scenario to declare a density map and be honest
    about it at the same time.
    """
    return [
        ProjectionTarget(region, density, provenance=provenance)
        for region, density in regions.items()
    ]
