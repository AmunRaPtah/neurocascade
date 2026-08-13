"""P6.2: endogenous transmitter tone at every projection system's receptors.

The brief: "Give every projection system a baseline endogenous transmitter
tone with cited basal concentrations, so that administered drugs compete at a
receptor pool that is already partly occupied by real transmitter rather than
an empty one. This changes existing results: report by how much."

This module is the REGISTRY and the REPORT. It holds no mechanism: the
competitive pool is `receptor.ReceptorSite`'s and has been since v0, the
concentration field is `projection.ProjectionSystem`'s from P6.1, and the
arithmetic of what tone costs a drug is `projection.tone_dose_ratio`. What
P6.2 adds is (a) a cited basal concentration for the two transmitters that
had a unit scale instead of one, (b) one place where all five are recorded
with their species, region, method and caveats, and (c) the quantified
answer to "by how much does this change existing results", computed by
re-running the scenarios rather than remembered.

    from neurocascade import endogenous_tone as tone

    print(tone.tone_report())        # tone, Kd and tonic occupancy per receptor
    print(tone.change_report())      # P6.1's numbers against P6.2's, re-run

======================================================================
WHAT "COMPETES AT A POOL THAT IS ALREADY PARTLY OCCUPIED" MEANS, EXACTLY
======================================================================

One receptor population, one finite pool, several ligands. At equilibrium,
with `x_j = C_j / Kd_j`,

    occupancy_i = x_i / (1 + sum_j x_j)

so tone enters an administered drug's occupancy in exactly one place, the
denominator, and the drug concentration needed for ANY given occupancy is
multiplied by

    dose ratio = 1 + C_tone / Kd_tone

That number depends on the tone and on the tone's affinity alone. Not on the
drug, not on its efficacy, not on the coupling gain downstream. So one
number per (receptor, transmitter) pair says what P6.2 costs every drug at
that receptor, and `tone_report()` is a table of them. `test_endogenous_tone`
asserts the ratio against the integrated ODE rather than against this
algebra.

Two consequences worth separating, because they point in opposite
directions:

* an AGONIST is weakened by tone. It has to displace transmitter that is
  already there, and it inherits a partly-activated baseline, so the same
  dose produces a smaller CHANGE in activation.
* an ANTAGONIST is what tone makes possible at all. With an empty pool a
  neutral antagonist occupies receptor, contributes zero efficacy, and
  changes nothing: every module from P5.1 to P5.7 had to write that sentence
  down. With tone it displaces transmitter that WAS contributing, so it
  produces a signal, and its size is set by the tonic occupancy it removes.

======================================================================
WHERE THE FIVE BASAL CONCENTRATIONS CAME FROM
======================================================================

    transmitter     nM     species  region              method
    dopamine        5.0    mouse    caudate-putamen     zero net flux
    serotonin       2.8    mouse    striatum            zero net flux
    noradrenaline   1.59   rat      medial PFC          zero net flux
    acetylcholine   18.4   rat      striatum            no net flux
    histamine       7.8    rat      medial hypothalamus in vitro recovery

Three of those predate P6.2 ([Bosse2011], [Mathews2004], [Itoh1991]) and are
reused unchanged from `monoamine_transporters.py` and
`projection_systems.py`. Two are new here and are the reason this task is
not bookkeeping:

* NORADRENALINE. P5.8 searched for this and recorded it empty; P6.1 searched
  again and recorded it empty again; `NORADRENALINE_BASAL_UNIT_SCALE = 1.0`
  stood in its place, labelled a unit. [PanLai1995] measured it by ZERO NET
  FLUX in the medial prefrontal cortex of CONSCIOUS rats: 1.59 +/- 0.37 nM,
  with a 51 +/- 8% in vivo recovery in the same animals. The reason two
  searches missed it is worth recording: the paper is titled as a study of
  ANAESTHETIC effects on the extraction fraction, and reports the conscious
  control group's basal concentration in passing.
* ACETYLCHOLINE. Same history, same outcome. [Vinson1997] measured it by
  quantitative (no-net-flux) microdialysis in rat striatum with no
  cholinesterase inhibitor in the perfusate: 18.4 +/- 11.8 nM. That paper is
  titled as a study of NEOSTIGMINE's effect on the extraction fraction.

Each carries its caveats at the constant rather than in a footnote (see
`monoamine_transporters.BASAL_NORADRENALINE` and
`projection_systems.ACETYLCHOLINE_BASAL`), and each has an independent
cross-check that DISAGREES, which is reported rather than averaged away:

    transmitter     used                cross-check                    ratio
    noradrenaline   1.59 nM rat mPFC    ~14 nM rat hippocampus         8.8x above
                    zero net flux       [Abercrombie1988]
    acetylcholine   18.4 nM rat CP      ~4 nM rat hippocampus          4.6x below
                    no net flux         [DeBundel2008]

Both cross-checks are dialysate values scaled by a probe-recovery estimate,
and both are from hippocampus. Both used values are zero/no-net-flux, which
measures the concentration at which the probe neither gains nor loses
analyte and so needs no recovery estimate at all; in vitro recovery exceeds
in vivo recovery, so a dialysate value corrected by it is biased UP. That
argument, and not the size of the number, is what picks the used value in
each case.

The two disagreements therefore point in OPPOSITE directions, and saying so
is the point of carrying both. For noradrenaline the used value is the
smaller, so it understates the tonic occupancy and understates every claim
P6.2 makes from it. For acetylcholine the used value is the LARGER, which is
the uncomfortable direction, and it is why `SENSITIVITY` re-runs the whole
table at [DeBundel2008]'s 4 nM and why the result of that re-run is
reported rather than filed.

======================================================================
THE ONE CONSTANT THAT IS IN THE WRONG PLACE, STATED PLAINLY
======================================================================

[Vinson1997]'s acetylcholine measurement is in the STRIATUM, where
extracellular acetylcholine is released by tonically active cholinergic
INTERNEURONS. The acetylcholine projection system in this codebase has three
sources -- the diagonal band, the pedunculopontine and the laterodorsal
tegmental nuclei -- and none of them is a striatal interneuron. So the
number is a measurement of extracellular acetylcholine, in a rodent, by the
right method, in the wrong compartment of the cholinergic system.

It is used because the alternative is the unit scale it replaces, which
asserts nothing; because at three of the four cholinergic receptors the
tonic occupancies it produces are small (0.073% at M1, 0.29% at alpha7,
0.46% at M2) so no conclusion there rests on its exact value; and because
`SENSITIVITY` reports every acetylcholine number at [DeBundel2008]'s
4.6-fold lower hippocampal value as well, so a reader can see the whole
supportable range. It is NOT presented
as a basal-forebrain-derived cortical acetylcholine concentration, because
it is not one.

The FOURTH cholinergic receptor is the exception and needs saying: alpha4beta2
sits at 22% tonic occupancy, because P5.4's desensitized state holds the pool
and tightens the site's apparent affinity 24.5-fold, from a Kd of 1600 nM to
an apparent 65 nM. That is the one place where this constant's value carries
a conclusion, and it is exactly the place where its 4.6-fold uncertainty
matters most: at [DeBundel2008]'s 4 nM the same number is 5.8%. Both are
reported and neither is presented as settled.

======================================================================
WHAT CHANGED, AND WHAT DID NOT
======================================================================

`change_report()` computes this rather than restating it, so the numbers in
docs/architecture.md are a transcript of a run. In outline:

* dopamine, serotonin and histamine: NOTHING changes. Their basal levels
  were already measured and are untouched, and a test asserts every one of
  P6.1's dopamine, serotonin and histamine results reproduces to 1e-9.
* noradrenaline: the field gains a 1.59-fold absolute scale. Tonic occupancy
  rises 1.59-fold at alpha1 and beta (both far below their Kd, so the
  response is linear) and 1.56-fold at the alpha2 autoreceptor, 3.571% ->
  5.561%, where saturation has begun to bite. Yohimbine given alone moves
  the cortical readout 2.4-fold further (+1.2e-5 -> +2.9e-5) and clonidine's
  closed loop 1.5-fold further.
* acetylcholine: the field gains an 18.4-fold absolute scale. Tonic occupancy
  rises 18.3 to 18.4-fold at alpha7, M1 and M2, all three of which are far
  below their Kd, and 14.58-fold at alpha4beta2, 1.507% -> 21.97%, which is
  sublinear because that site is no longer far below its apparent Kd.
  Scopolamine given alone moves the cortical readout 15.5-fold further
  (-8e-6 -> -1.24e-4).

======================================================================
WHAT THIS MODULE DELIBERATELY DOES NOT DO
======================================================================

* It does not give a receptor a tone its own transmitter's concentration
  cannot supply. Tone here is always `basal concentration / Kd` at a site
  that already carries that transmitter as an orthosteric ligand, with the
  Kd sourced by whichever P5 module built the site.
* It does not touch a hand-built scenario. A tone is added by PORTING a
  scenario onto a projection system, which is P6.1's pattern and its
  regression guarantee; `serotonin.py` and the rest are unmoved, exactly as
  P6.1 left them.
* It does not invent a tone for a transmitter with no projection system.
  GABA, glutamate, adenosine, the opioid peptides and the endocannabinoids
  all have real tonic levels and none of them is a diffuse projection
  system, so none is in scope for a task about projection systems.
"""

import numpy as np

from . import acetylcholine as _ach
from . import dopamine as _da
from . import histamine as _ha
from . import monoamine_transporters as _mt
from . import noradrenaline as _na
from . import projection as _projection
from . import projection_scenarios as _ports
from . import projection_systems as _ps
from . import scenarios as _scenarios
from . import serotonin as _5ht


CITATIONS = {
    "PanLai1995": _ps.CITATIONS["PanLai1995"],
    "Vinson1997": _ps.CITATIONS["Vinson1997"],
    "DeBundel2008": _ps.CITATIONS["DeBundel2008"],
    "Abercrombie1988": _ps.CITATIONS["Abercrombie1988"],
    "Ihalainen1999": _ps.CITATIONS["Ihalainen1999"],
    "Bosse2011": _ps.CITATIONS["Bosse2011"],
    "Mathews2004": _ps.CITATIONS["Mathews2004"],
    "Itoh1991": _ps.CITATIONS["Itoh1991"],
}


# ---------------------------------------------------------------------------
# 1. The registry
# ---------------------------------------------------------------------------


class BasalTone(object):
    """One projection system's drug-free extracellular transmitter level.

    Every field is about provenance except `nM`. That is deliberate: a basal
    concentration is only usable if the species, the region and the method
    travel with it, and this codebase's rule is that they travel attached to
    the number rather than in prose somewhere else.

    `cross_check` is an independently measured value for the same quantity
    that DISAGREES, with the factor. It is carried rather than reconciled.
    """

    def __init__(
        self,
        transmitter,
        nM,
        species,
        region,
        method,
        citation,
        caveats=(),
        cross_check=None,
    ):
        self.transmitter = transmitter
        self.nM = float(nM)
        self.species = species
        self.region = region
        self.method = method
        self.citation = citation
        self.caveats = tuple(caveats)
        #: (value_nM, citation, note) or None.
        self.cross_check = cross_check

    @property
    def cross_check_ratio(self):
        if self.cross_check is None:
            return None
        return self.cross_check[0] / self.nM

    def __repr__(self):
        return "BasalTone(%r, %.4g nM, %s %s, %s)" % (
            self.transmitter,
            self.nM,
            self.species,
            self.region,
            self.citation,
        )


#: The five projection systems' endogenous tone, one cited entry each. This
#: is P6.2's first acceptance criterion, and `test_endogenous_tone` asserts
#: that every transmitter in `projection_systems.CLEARANCE_BUILDERS` appears
#: here with a citation and that no entry is a unit scale.
BASAL_TONE = {
    "dopamine": BasalTone(
        transmitter="dopamine",
        nM=_mt.BASAL_DOPAMINE,
        species="mouse",
        region="caudate-putamen",
        method="zero net flux microdialysis",
        citation="Bosse2011",
        caveats=(
            "wild-type controls of a BDNF-deficient study, 5.0 +/- 0.2 nM",
        ),
        cross_check=(
            2.38,
            "PanLai1995",
            "conscious rat medial PFC by zero net flux, 2.38 +/- 0.70 nM: a "
            "DIFFERENT REGION rather than a disagreement, and the only "
            "measured dopamine concentration in this codebase from outside "
            "the striatum. [Ihalainen1999] puts mouse striatal dopamine "
            "13-30 fold above cortical and hippocampal dopamine in dialysate, "
            "which is the ordering P6.3's density map has to reproduce and "
            "which this 2.1-fold cross-region ratio does NOT reproduce.",
        ),
    ),
    "serotonin": BasalTone(
        transmitter="serotonin",
        nM=_mt.BASAL_SEROTONIN,
        species="mouse",
        region="striatum",
        method="zero net flux microdialysis",
        citation="Mathews2004",
        caveats=(
            "wild-type SERT(+/+), 2.8 +/- 1 nM; the same paper's frontal "
            "cortex value is 1.4 +/- 0.4 nM",
        ),
        cross_check=None,
    ),
    "noradrenaline": BasalTone(
        transmitter="noradrenaline",
        nM=_mt.BASAL_NORADRENALINE,
        species="rat",
        region="medial prefrontal cortex",
        method="zero net flux microdialysis, conscious",
        citation="PanLai1995",
        caveats=(
            "RAT, where dopamine's and serotonin's basal levels here are mouse",
            "the conscious group is used; the same paper's anaesthetized "
            "groups read 4.38 +/- 1.39 and 3.67 +/- 0.90 nM and are "
            "deliberately not used, because anaesthesia halved the extraction "
            "fraction in the paper's own finding",
            "no mouse value was found; that gap is still open",
        ),
        cross_check=(
            14.0,
            "Abercrombie1988",
            "rat hippocampus, dialysate corrected for a 21% in vitro "
            "recovery, ~14 nM: 8.8-fold above the value used. In vitro "
            "recovery exceeds in vivo recovery, so a value corrected by it "
            "is biased UP, and zero net flux is the method this codebase "
            "prefers; the smaller value is used and the disagreement is "
            "reported rather than split.",
        ),
    ),
    "acetylcholine": BasalTone(
        transmitter="acetylcholine",
        nM=_ps.ACETYLCHOLINE_BASAL,
        species="rat",
        region="striatum",
        method="no-net-flux quantitative microdialysis, no esterase inhibitor",
        citation="Vinson1997",
        caveats=(
            "RAT, not mouse",
            "STRIATUM, where extracellular acetylcholine comes from tonically "
            "active interneurons rather than from the three projection nuclei "
            "this codebase's acetylcholine field is built on. A source "
            "mismatch as well as a region mismatch; see the module docstring",
            "large dispersion: +/- 11.8 nM on 18.4 nM with n = 10",
        ),
        cross_check=(
            _ps.ACETYLCHOLINE_BASAL_HIPPOCAMPAL_ESTIMATE,
            "DeBundel2008",
            "rat hippocampus: 0.40 +/- 0.12 nM in dialysate at an in vivo "
            "recovery the authors put at ~10%, i.e. ~4 nM extracellular, "
            "4.6-fold below the value used. The dialysate number itself is a "
            "strict LOWER BOUND on the extracellular concentration, because "
            "dialysate = extraction fraction * extracellular and the "
            "extraction fraction is at most 1.",
        ),
    ),
    "histamine": BasalTone(
        transmitter="histamine",
        nM=_ps.HISTAMINE_BASAL,
        species="rat",
        region="medial hypothalamus",
        method="dialysate corrected for in vitro recovery, anaesthetized",
        citation="Itoh1991",
        caveats=(
            "RAT, not mouse",
            "the hypothalamus is the tuberomammillary nucleus's own "
            "neighbourhood rather than a forebrain target",
            "recovery-corrected dialysate rather than zero net flux",
            "[Cumming1991] found striatal extracellular histamine "
            "tetrodotoxin-insensitive and only 20% calcium-dependent, so part "
            "of what a probe collects may not be neuronal",
        ),
        cross_check=None,
    ),
}

#: The two transmitters whose tone P6.2 added, and what stood there before.
#: Used by `change_report()` so the comparison is against a recorded value
#: rather than a remembered one.
ADDED_BY_P6_2 = dict(_ports.P61_UNIT_SCALE_BASAL)

#: Per transmitter, the alternative basal concentration the sensitivity pass
#: re-runs everything at: the cross-check where one exists.
SENSITIVITY = {
    name: tone.cross_check[0]
    for name, tone in BASAL_TONE.items()
    if tone.cross_check is not None
}


# ---------------------------------------------------------------------------
# 2. Every receptor in this codebase that one of these five transmitters
#    binds, whether or not a port uses it
# ---------------------------------------------------------------------------


class ToneTarget(object):
    """One receptor site, and the endogenous ligand that tonically occupies
    it.

    `site` is a zero-argument callable returning the P5 module's own object,
    so this module can only READ a receptor, never redefine one. `ported`
    records whether a `projection_scenarios.py` port actually places this
    receptor in a field; the ones that are not ported are listed anyway,
    because the tonic occupancy of a receptor is a property of the receptor
    and the transmitter rather than of whether a scenario happens to use it.

    A site whose endogenous ligand's binding constants are ILLUSTRATIVE is
    deliberately absent, because a tonic occupancy computed from a
    placeholder Kd would be a placeholder number wearing a citation. That is
    why v0's `scenarios.d2_receptor_site()` is not here and P2.2's calibrated
    one is.
    """

    def __init__(self, name, transmitter, ligand, site, source, ported, note=""):
        self.name = name
        self.transmitter = transmitter
        self.ligand = ligand
        self.site = site
        self.source = source
        self.ported = ported
        self.note = note

    def __repr__(self):
        return "ToneTarget(%r, %s)" % (self.name, self.transmitter)


TONE_TARGETS = (
    ToneTarget(
        "D1", "dopamine", "dopamine", _da.d1_receptor_site,
        "dopamine.py (P5.2)", ported=False,
        note="the direct pathway; no port places it in a field yet",
    ),
    ToneTarget(
        "D2", "dopamine", "dopamine", _scenarios.d2_receptor_site_calibrated,
        "scenarios.py, P2.2-calibrated", ported=True,
        note=(
            "the CALIBRATED site ([Agren2021] dopamine, [Sykes2017] "
            "haloperidol), which is the one every port and every P5.8 result "
            "uses. `scenarios.d2_receptor_site()` is v0's illustrative "
            "placeholder pair and is deliberately absent from this table: a "
            "tonic occupancy computed from an illustrative Kd would be an "
            "illustrative number wearing a citation."
        ),
    ),
    ToneTarget(
        "5HT1A", "serotonin", "serotonin", _5ht.receptor_site_5ht1a,
        "serotonin.py (P5.1)", ported=True,
    ),
    ToneTarget(
        "5HT2A", "serotonin", "serotonin", _5ht.receptor_site_5ht2a,
        "serotonin.py (P5.1)", ported=True,
    ),
    ToneTarget(
        "alpha1", "noradrenaline", "noradrenaline", _na.receptor_site_alpha1,
        "noradrenaline.py (P5.3)", ported=True,
    ),
    ToneTarget(
        "alpha2", "noradrenaline", "noradrenaline", _na.receptor_site_alpha2,
        "noradrenaline.py (P5.3)", ported=True,
    ),
    ToneTarget(
        "beta", "noradrenaline", "noradrenaline", _na.receptor_site_beta,
        "noradrenaline.py (P5.3)", ported=True,
    ),
    ToneTarget(
        "alpha4beta2", "acetylcholine", "acetylcholine",
        _ach.receptor_site_alpha4beta2, "acetylcholine.py (P5.4)", ported=True,
        note="desensitizing site: the Kd column is the APPARENT one",
    ),
    ToneTarget(
        "alpha7", "acetylcholine", "acetylcholine", _ach.receptor_site_alpha7,
        "acetylcholine.py (P5.4)", ported=True,
        note="desensitizing site: the Kd column is the APPARENT one",
    ),
    ToneTarget(
        "M1", "acetylcholine", "acetylcholine", _ach.receptor_site_m1,
        "acetylcholine.py (P5.4)", ported=True,
    ),
    ToneTarget(
        "M2", "acetylcholine", "acetylcholine", _ach.receptor_site_m2,
        "acetylcholine.py (P5.4)", ported=True,
    ),
    ToneTarget(
        "H1", "histamine", "histamine", _ha.receptor_site_h1,
        "histamine.py (P5.6)", ported=True,
    ),
)


def tone_concentrations(site, ligand, concentration):
    """The site's full concentration vector with one ligand held at
    `concentration` and every other at zero."""
    names = [lig.name for lig in site.ligands]
    vector = np.zeros(len(names))
    vector[names.index(ligand)] = float(concentration)
    return vector


def tonic_occupancy(target, basal=None):
    """Fractional occupancy of `target`'s pool by its endogenous transmitter
    at the cited basal concentration, INTEGRATED.

    Integrated rather than evaluated in closed form, so that the number is
    the one the ODE system actually produces for every site class, including
    the two desensitizing nicotinic ones whose pool is held by a bound state
    plus a desensitized state.
    """
    site = target.site()
    if basal is None:
        basal = BASAL_TONE[target.transmitter].nM
    concentrations = tone_concentrations(site, target.ligand, basal)
    state = _projection.resting_occupancy(site, concentrations)
    names = [lig.name for lig in site.ligands]
    index = names.index(target.ligand)
    occupied = float(state[index])
    factor = getattr(site, "apparent_affinity_factor", None)
    if factor is not None:
        # bound + desensitized is what holds the pool; `state` carries them
        # in two blocks and P5.4's `split_states` is the one that knows where.
        bound, desens, _ = site.split_states(state)
        occupied = float(bound[index] + desens[index])
    return occupied


def tonic_activation(target, basal=None):
    """The activation the endogenous transmitter alone produces at rest.
    This is the quantity `projection.standing_drive` multiplies by a coupling
    gain, so it is what the tonic compensation folds into a baseline."""
    site = target.site()
    if basal is None:
        basal = BASAL_TONE[target.transmitter].nM
    concentrations = tone_concentrations(site, target.ligand, basal)
    return float(site.activation(_projection.resting_occupancy(site, concentrations)))


def dose_ratio(target, basal=None):
    """`1 + C_tone / Kd_tone`: the factor by which endogenous tone multiplies
    the concentration of ANY drug needed to reach a given occupancy at this
    receptor. See `projection.tone_dose_ratio`."""
    site = target.site()
    if basal is None:
        basal = BASAL_TONE[target.transmitter].nM
    return _projection.tone_dose_ratio(
        site, tone_concentrations(site, target.ligand, basal)
    )


def transmitter_kd(target):
    """The endogenous transmitter's (apparent) dissociation constant at this
    site, in nM."""
    return _projection.orthosteric_affinities(target.site())[target.ligand]


def tone_table(basal=None):
    """One row per receptor: transmitter, tone, Kd, tonic occupancy, dose
    ratio. `basal` optionally overrides the tone per transmitter."""
    basal = dict(basal or {})
    rows = []
    for target in TONE_TARGETS:
        level = basal.get(target.transmitter, BASAL_TONE[target.transmitter].nM)
        rows.append(
            {
                "receptor": target.name,
                "transmitter": target.transmitter,
                "basal_nM": level,
                "kd_nM": transmitter_kd(target),
                "occupancy": tonic_occupancy(target, basal=level),
                "activation": tonic_activation(target, basal=level),
                "dose_ratio": dose_ratio(target, basal=level),
                "ported": target.ported,
                "note": target.note,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 3. What changed, computed rather than remembered
# ---------------------------------------------------------------------------

#: The ports P6.2 moves, and the ones it must leave bit-for-bit alone.
CHANGED_PORTS = ("noradrenaline", "acetylcholine")
UNCHANGED_PORTS = ("dopamine", "serotonin", "histamine")


def _p61_basal(key):
    """The basal concentration P6.1 ran port `key` at, or None if P6.2 did
    not move it."""
    return ADDED_BY_P6_2.get(key)


def port_change(key, t_end=30.0):
    """The port's own two results at P6.1's basal level and at P6.2's.

    Returns `{"before": ..., "after": ..., "basal_before": ...}` where each
    of before/after carries `lone_antagonist` and, where the port has one,
    `closed_loop`. For a port P6.2 did not move, `before` is `after` by
    construction and a test asserts the two are identical to 1e-9 rather
    than skipping the comparison.
    """
    before_basal = _p61_basal(key)
    out = {
        "port": key,
        "basal_before": before_basal,
        "basal_after": BASAL_TONE[key].nM,
    }
    for label, level in (("before", before_basal), ("after", None)):
        entry = {"lone_antagonist": _ports.lone_antagonist_effect(key, t_end=t_end, basal=level)}
        if key in _ports.AUTORECEPTOR_PROBE:
            entry["closed_loop"] = _ports.closed_loop_effect(key, t_end=t_end, basal=level)
        out[label] = entry
    return out


def empty_pool_change(key, t_end=30.0, empty=1e-9):
    """What the port's lone antagonist does against an EMPTY pool.

    The counterfactual the brief's sentence names: "rather than acting on an
    empty one". A neutral antagonist with no transmitter to displace is
    inert, which is the sentence P5.1 to P5.7 all had to write; this runs the
    same port with the tone driven to `empty` nM and reports both.

    `empty` is a small positive number rather than zero because the release
    scale is derived from the basal level and a zero-tone field has no scale
    at all; 1e-9 nM is nine orders below every Kd in this codebase, so the
    resulting occupancy is numerically indistinguishable from an empty pool.
    """
    with_tone = _ports.lone_antagonist_effect(key, t_end=t_end)
    without = _ports.lone_antagonist_effect(key, t_end=t_end, basal=empty)
    return {
        "port": key,
        "drug": with_tone["drug"],
        "dose_nM": with_tone["dose_nM"],
        "with_tone": with_tone["terminal_rate"],
        "empty_pool": without["terminal_rate"],
    }


# ---------------------------------------------------------------------------
# 4. Reports
# ---------------------------------------------------------------------------


def basal_report():
    lines = ["Endogenous transmitter tone, per projection system (P6.2)", "=" * 72, ""]
    lines.append(
        "%-14s %8s  %-7s %-24s %-52s %s"
        % ("transmitter", "nM", "species", "region", "method", "citation")
    )
    for name in _ps.NUCLEI_BY_TRANSMITTER:
        tone = BASAL_TONE[name]
        lines.append(
            "%-14s %8.4g  %-7s %-24s %-52s [%s]"
            % (
                tone.transmitter,
                tone.nM,
                tone.species,
                tone.region,
                tone.method,
                tone.citation,
            )
        )
    lines.append("")
    lines.append("Independent cross-checks, carried rather than reconciled:")
    for name in _ps.NUCLEI_BY_TRANSMITTER:
        tone = BASAL_TONE[name]
        if tone.cross_check is None:
            lines.append("  %-14s (none found)" % tone.transmitter)
            continue
        value, citation, note = tone.cross_check
        lines.append(
            "  %-14s %.4g nM [%s], %.2fx the value used"
            % (tone.transmitter, value, citation, value / tone.nM)
        )
        for chunk in _wrap(note, 66):
            lines.append("      " + chunk)
    lines.append("")
    lines.append("Caveats, attached to each constant:")
    for name in _ps.NUCLEI_BY_TRANSMITTER:
        tone = BASAL_TONE[name]
        lines.append("  %s:" % tone.transmitter)
        for caveat in tone.caveats:
            for i, chunk in enumerate(_wrap(caveat, 64)):
                lines.append("    %s %s" % ("*" if i == 0 else " ", chunk))
    return "\n".join(lines)


def tone_report(basal=None):
    lines = ["Tonic occupancy before any drug is given", "=" * 72, ""]
    lines.append(
        "The pool an administered drug arrives at. `dose ratio` is the factor"
    )
    lines.append(
        "by which this tone multiplies the concentration of ANY drug needed to"
    )
    lines.append("reach a given occupancy at that receptor: 1 + C/Kd, exactly.")
    lines.append("")
    lines.append(
        "%-12s %-14s %9s %11s %11s %10s  %s"
        % ("receptor", "transmitter", "tone nM", "Kd nM", "occupancy", "dose ratio", "ported")
    )
    for row in tone_table(basal=basal):
        lines.append(
            "%-12s %-14s %9.4g %11.4g %11.3e %10.4f  %s"
            % (
                row["receptor"],
                row["transmitter"],
                row["basal_nM"],
                row["kd_nM"],
                row["occupancy"],
                row["dose_ratio"],
                "yes" if row["ported"] else "no",
            )
        )
    lines.append("")
    lines.append("Notes:")
    for target in TONE_TARGETS:
        if not target.note:
            continue
        for i, chunk in enumerate(_wrap(target.note, 58)):
            lines.append("  %-12s %s" % (target.name if i == 0 else "", chunk))
    return "\n".join(lines)


def sensitivity_report():
    lines = ["The same table at every cross-check value", "=" * 72, ""]
    lines.append(
        "Each block re-runs the tonic occupancies with ONE transmitter's tone"
    )
    lines.append(
        "set to its disagreeing independent measurement, so the reader sees the"
    )
    lines.append("range the literature supports rather than one number.")
    for transmitter, value in sorted(SENSITIVITY.items()):
        tone = BASAL_TONE[transmitter]
        citation = tone.cross_check[1]
        lines.append("")
        lines.append(
            "%s: %.4g nM [%s] instead of %.4g nM [%s]"
            % (transmitter, value, citation, tone.nM, tone.citation)
        )
        lines.append(
            "  %-12s %11s %11s %10s" % ("receptor", "used", "cross-check", "ratio")
        )
        base = {row["receptor"]: row for row in tone_table()}
        for row in tone_table(basal={transmitter: value}):
            if row["transmitter"] != transmitter:
                continue
            reference = base[row["receptor"]]["occupancy"]
            lines.append(
                "  %-12s %11.3e %11.3e %10.3f"
                % (
                    row["receptor"],
                    reference,
                    row["occupancy"],
                    (row["occupancy"] / reference) if reference else float("nan"),
                )
            )
    return "\n".join(lines)


def change_report(t_end=30.0):
    lines = ["What P6.2 changes in P6.1's results", "=" * 72, ""]
    lines.append(
        "Every number below is produced by re-running the port, once at the"
    )
    lines.append(
        "basal concentration P6.1 used and once at P6.2's cited one. The three"
    )
    lines.append(
        "ports whose tone was already measured are re-run too, so that 'nothing"
    )
    lines.append("moved' is a measurement rather than an assumption.")
    lines.append("")
    lines.append(
        "%-14s %10s %10s %13s %13s"
        % ("port", "basal P6.1", "basal P6.2", "antagonist r", "was")
    )
    for key in _ports.PORT_ORDER:
        change = port_change(key, t_end=t_end)
        before = change["before"]["lone_antagonist"]["terminal_rate"]
        after = change["after"]["lone_antagonist"]["terminal_rate"]
        lines.append(
            "%-14s %10s %10.4g %+13.6f %+13.6f"
            % (
                key,
                ("%.4g" % change["basal_before"]) if change["basal_before"] else "same",
                change["basal_after"],
                after,
                before,
            )
        )
    lines.append("")
    lines.append("The autoreceptor loop, at both tones:")
    lines.append(
        "%-14s %-12s %12s %12s %12s %12s"
        % ("port", "drug", "occ P6.2", "occ P6.1", "rate P6.2", "rate P6.1")
    )
    for key in _ports.PORT_ORDER:
        if key not in _ports.AUTORECEPTOR_PROBE:
            continue
        change = port_change(key, t_end=t_end)
        after = change["after"]["closed_loop"]
        before = change["before"]["closed_loop"]
        lines.append(
            "%-14s %-12s %+12.3e %+12.3e %+12.5f %+12.5f"
            % (
                key,
                after["drug"],
                after["terminal_occupancy"],
                before["terminal_occupancy"],
                after["terminal_rate"],
                before["terminal_rate"],
            )
        )
    lines.append("")
    lines.append("The counterfactual the brief names: an EMPTY pool.")
    lines.append(
        "%-14s %-14s %14s %14s"
        % ("port", "antagonist", "with tone", "empty pool")
    )
    for key in _ports.PORT_ORDER:
        effect = empty_pool_change(key, t_end=t_end)
        lines.append(
            "%-14s %-14s %+14.6f %+14.6f"
            % (key, effect["drug"], effect["with_tone"], effect["empty_pool"])
        )
    return "\n".join(lines)


def summary(t_end=30.0):
    return "\n\n".join(
        [
            basal_report(),
            tone_report(),
            sensitivity_report(),
            change_report(t_end=t_end),
        ]
    )


def _wrap(text, width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
