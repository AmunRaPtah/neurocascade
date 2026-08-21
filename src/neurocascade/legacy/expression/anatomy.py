"""P4.3: the receptor maps against independently known distributions.

What this module is for, in one sentence
----------------------------------------

P4.1 ingested Allen ISH expression energy and P4.2 converted it to a
relative density under a stated proportionality assumption; neither of them
can tell you whether the resulting map is *the map of the gene it claims to
be*. A transposed matrix, a shifted gene column, a structure-id mis-join or
a normalization applied along the wrong axis all produce a map with the
right shape, the right sparsity and entirely plausible numbers. The only
cheap detector is anatomy: mu-opioid receptors are dense in the medial
habenula, A2A is a striatal receptor, alpha-1 GABA-A is a cortical one, and
a map that says otherwise is broken.

So the claims here are **bug detectors, not findings**. A failing
concordant claim means the ingestion or the normalization is wrong, and the
right response is to go and find the defect, not to write a paragraph about
what the brain might be doing.

The three rules this module follows
-----------------------------------

**1. Every expectation is cited, with species and modality.** Each claim
names an `AnatomySource` carrying the paper, its PMID/DOI, the species, the
method, and whether the source measured *mRNA* or *protein/binding*. That
last field is not decoration: a transcript map should be checked against
transcript expectations, and every place where a protein expectation is
used instead is a place where the mRNA-protein divergence P4.2 measured can
masquerade as an ingestion bug.

**2. Claims are ordinal, never proportional.** P4.2 measured the
transcript-to-density relation over the one gene with a matched mouse
density map and found a good *ordering* (Spearman ~0.71) and a bad *scale*
(a fit through the origin explains 31% of the variance; the log-log slope
is 0.46 where proportionality needs 1.0). A map that is a good ordering and
a bad scale can only be validated as an ordering. So a claim is always of
the form "every region in `high` is above every region in `low`", and the
measured fold-difference is reported but never asserted against a threshold
derived from the source's own magnitudes.

**3. A claim is asserted only if the source separates the two groups by at
least two of its own tiers.** Almost every anatomical source here is
semiquantitative: Erbs et al. score MOR density on a `-`/`+`/`++`/`+++`
scale, Hoertnagl et al. on `-`/`(+)`/`+`/`++`/`+++`. One tier is one
observer's eyeball judgement of one autoradiogram, and rule 2's compression
means one tier of protein can be anything from no difference to a 20-fold
difference of transcript. Claims separated by a single tier are therefore
*reported* and not *asserted*: `Claim.assertable` is the pre-registered
rule, decided per claim from the source's own labels before any measurement
was made, and `results/receptor_map_validation.txt` prints the outcome of
every claim including the reported ones, so nothing is hidden by it.

Divergences are claims too
--------------------------

Two of the claims below are expected to FAIL, and they are recorded with
`expected_to_hold=False` plus a mechanism and a citation for the mechanism.
Cnr1 is the clearest case: [Herkenham1991] measured the highest CB1 binding
densities in the brain in globus pallidus and substantia nigra pars
reticulata, and showed by striatal lesion that those receptors sit on the
axons and terminals of striatal projection neurons - so the transcript is
in the caudoputamen and the protein is two synapses away. The transcript
map cannot reproduce that and must not be "corrected" until Phase 6 has the
projection data to do it honestly (see `mismatch.py`).

Recording those as claims rather than dropping them is the point. An
expected failure that is asserted to fail still fails loudly if somebody
"fixes" it, and its measured size is the error bar on using the map for
that gene.
"""

import numpy as np

from . import mismatch as _mismatch
from .autoradiography import YOO2010

#: The plane and combine policy the P4.3 checks are quoted under. P4.2
#: measured the policy to be worth almost nothing for the ordering these
#: claims test (all fourteen plane x combine policies gave Spearman between
#: 0.709 and 0.794 on the D1 comparison), and `policy_agreement()` re-checks
#: that here, over these claims, rather than assuming it carries across.
DEFAULT_PLANE = "coronal-else-sagittal"
DEFAULT_COMBINE = "mean"

#: A claim is asserted by the test suite only when its source separates the
#: two region groups by at least this many of the source's own tiers. See
#: rule 3 in the module docstring.
ASSERTABLE_SEPARATION = 2.0


class AnatomySource(object):
    """One published anatomical distribution, and what kind of thing it measured."""

    def __init__(self, key, citation, species, method, modality,
                 pmid="", doi="", scale="", note=""):
        if modality not in ("mRNA", "protein"):
            raise ValueError(
                "modality must be 'mRNA' or 'protein', got %r. A transcript "
                "map compared against a protein expectation is a different "
                "test from one compared against a transcript expectation, "
                "and the difference has to be visible in the record."
                % (modality,)
            )
        self.key = key
        self.citation = citation
        self.species = species
        self.method = method
        self.modality = modality
        self.pmid = pmid
        self.doi = doi
        self.scale = scale
        self.note = note

    def __repr__(self):
        return "AnatomySource(%r, %s, %s)" % (self.key, self.species, self.modality)


ERBS2015 = AnatomySource(
    key="erbs2015",
    citation=(
        "Erbs E, Faget L, Scherrer G, Matifas A, Filliol D, Vonesch JL, "
        "Koch M, Kessler P, Hentsch D, Birling MC, Koutsourakis M, Vasseur "
        "L, Veinante P, Kieffer BL, Massotte D. 'A mu-delta opioid receptor "
        "brain atlas reveals neuronal co-occurrence in subcortical "
        "networks.' Brain Struct Funct. 2015;220(2):677-702."
    ),
    pmid="24623156",
    doi="10.1007/s00429-014-0717-9",
    species="Mus musculus",
    method=(
        "mu-mCherry knock-in mice (MOR fused to mCherry at the endogenous "
        "locus); whole-brain fluorescence, scored region by region"
    ),
    modality="protein",
    scale="- not detectable, +/- weak, + moderate, ++ dense, +++ very dense "
    "(the published Table 1 also uses ++++ for its three densest entries, "
    "MHb, fr and LPBE, above the legend's top tier)",
    note=(
        "Table 1 values used here, read from the PMC deposit: MHb ++++, CPu "
        "+++, AcbC ++, AcbSh ++, PVA +++, PV ++, MD +, VTA ++, SNC +/++, "
        "SNR +, LC +, DLPAG +, LPAG +/++, VLPAG +, DMPAG -, M1/M2 +, PrL +, "
        "S1/S2 +/++, CA1/CA3 +, DG +, CbCx -. The paper states its "
        "distributions are 'in full agreement with the previous reports in "
        "mice and rats based on ligand binding'."
    ),
)

HORTNAGL2013 = AnatomySource(
    key="hortnagl2013",
    citation=(
        "Hoertnagl H, Tasan RO, Wieselthaler A, Kirchmair E, Sieghart W, "
        "Sperk G. 'Patterns of mRNA and protein expression for 12 GABAA "
        "receptor subunits in the mouse brain.' Neuroscience. "
        "2013;236:345-372."
    ),
    pmid="23337532",
    doi="10.1016/j.neuroscience.2013.01.008",
    species="Mus musculus",
    method="in situ hybridization (and immunohistochemistry on the same "
    "material), mouse brain excluding brain stem",
    modality="mRNA",
    scale="- no labeling, (+) faint, + weak, ++ moderate, +++ strong",
    note=(
        "alpha1 ISH column, read from the PMC deposit's Tables 2-9: cortex "
        "layer 4 +++, layers 2/3/6 ++, layers 1/5 +, piriform +++; CA1 +++, "
        "dentate gyrus ++(+), CA3 +, dorsal/ventral subiculum +++; caudate "
        "putamen (+), accumbens core (+), accumbens shell (+), globus "
        "pallidus +++, subthalamic +++, substantia nigra reticulata +++, "
        "substantia nigra compacta ++; cerebellar Purkinje layer +++, "
        "granule layer ++, molecular layer +, cerebellar nuclei +++. This "
        "is the ONLY mouse ISH source among the six and it is the one whose "
        "modality matches the map exactly."
    ),
)

WANG2022 = AnatomySource(
    key="wang2022",
    citation=(
        "Wang M, Li Z, Song Y, Sun Q, Deng L, Lin Z, Zeng Y, Qiu C, Lin J, "
        "Guo H, Chen J, Guo W. 'Genetic tagging of the adenosine A2A "
        "receptor reveals its heterogeneous expression in brain regions.' "
        "Front Neuroanat. 2022;16:978641."
    ),
    pmid="36059431",
    doi="10.3389/fnana.2022.978641",
    species="Mus musculus",
    method=(
        "A2AR-tag knock-in mice (3xHA-2xMyc fused to the A2AR C-terminus by "
        "CRISPR/Cas9); anti-HA immunostaining across the brain"
    ),
    modality="protein",
    scale="verbal: 'most prominent and intense', 'lighter than', 'very "
    "lightly labeled', 'a few discrete cells', 'no signal was detected'",
    note=(
        "Quoted: 'The most prominent and intense labeling was observed in "
        "the neuropil of the entire striatum and extended into the nucleus "
        "accumbens and olfactory tubercles'; 'Immunoreactivity in the "
        "globus pallidus was lighter than in the striatum'; 'In the "
        "hippocampus, very lightly labeled cells with evident pyramidal "
        "morphology could be seen in the CA1 and CA2 regions, but not in "
        "the dentate gyrus'; 'a few discrete and bright labeled cells could "
        "be seen in the deep layers (layer V or VI) in certain areas of the "
        "cerebral cortex'; 'no signal was detected in the midbrain or "
        "cerebellum'. The paper's own contribution is finding A2A OUTSIDE "
        "the striatum, so it is if anything biased against the "
        "striatum-specificity claim tested here."
    ),
)

MARSICANO1999 = AnatomySource(
    key="marsicano1999",
    citation=(
        "Marsicano G, Lutz B. 'Expression of the cannabinoid receptor CB1 "
        "in distinct neuronal subpopulations in the adult mouse forebrain.' "
        "Eur J Neurosci. 1999;11(12):4213-4225."
    ),
    pmid="10594647",
    doi="10.1046/j.1460-9568.1999.00847.x",
    species="Mus musculus",
    method="double in situ hybridization at single-cell resolution, mouse "
    "forebrain",
    modality="mRNA",
    scale="two-level, by design: the paper's finding is 'a clear "
    "distinction between neurons containing CB1 mRNA either at high levels "
    "or low levels'",
    note=(
        "Quoted: 'The majority of high CB1-expressing cells are GABAergic "
        "neurons belonging mainly to the cholecystokinin-positive and "
        "parvalbumin-negative type of interneurons (basket cells)'; 'In the "
        "hippocampus, amygdala and entorhinal cortex area, CB1 mRNA is "
        "present at low but significant levels in many non-GABAergic "
        "cells'. The regions this paper works in - hippocampus, amygdala, "
        "cortex - are therefore CB1-mRNA-bearing; it is a forebrain study "
        "and says nothing about thalamus, which is why the thalamic half of "
        "the claim below leans on [Mailleux1992]."
    ),
)

MAILLEUX1992 = AnatomySource(
    key="mailleux1992",
    citation=(
        "Mailleux P, Vanderhaeghen JJ. 'Distribution of neuronal "
        "cannabinoid receptor in the adult rat brain: a comparative "
        "receptor binding radioautography and in situ hybridization "
        "histochemistry.' Neuroscience. 1992;48(3):655-668."
    ),
    pmid="1376455",
    doi="10.1016/0306-4522(92)90409-u",
    species="Rattus norvegicus",
    method="[3H]CP55,940 binding radioautography and oligonucleotide in "
    "situ hybridization in the same study",
    modality="mRNA",
    scale="verbal: slight / moderate / dense / very dense",
    note=(
        "RAT, and used here only for the one thing no mouse source in reach "
        "supplies: the list of regions where CB1 mRNA is absent. Its "
        "regional survey names cortex, hippocampus, basal ganglia, septum, "
        "amygdala, habenula, hypothalamus, interpeduncular nucleus, central "
        "gray, pons, cerebellum and spinal cord, and no dorsal thalamic "
        "nucleus at all. No quantity from it is used, for the species "
        "reason P4.2 gives for refusing rat mu-opioid autoradiography; what "
        "is used is a presence/absence ordering that the mouse study "
        "[Marsicano1999] independently supports on its own (forebrain) half."
    ),
)

HERKENHAM1991 = AnatomySource(
    key="herkenham1991",
    citation=(
        "Herkenham M, Lynn AB, de Costa BR, Richfield EK. 'Neuronal "
        "localization of cannabinoid receptors in the basal ganglia of the "
        "rat.' Brain Res. 1991;547(2):267-274."
    ),
    pmid="1909204",
    doi="10.1016/0006-8993(91)90970-7",
    species="Rattus norvegicus",
    method="[3H]CP55,940 section binding autoradiography with ibotenic-acid "
    "striatal lesions and 6-OHDA nigrostriatal lesions",
    modality="protein",
    scale="verbal: highest / dense",
    note=(
        "Quoted: 'In rat brain, the highest receptor densities are in the "
        "globus pallidus and substantia nigra pars reticulata. Receptors "
        "are also dense in the caudate-putamen.' and 'The data show that "
        "cannabinoid receptors in the basal ganglia are neuronally located "
        "on striatal projection neurons, including their axons and "
        "terminals.' RAT, and used here only as the mechanism citation for "
        "a divergence that is expected to fail - it sets no threshold and "
        "supplies no number to the map."
    ),
)

GERFEN2011 = AnatomySource(
    key="gerfen2011",
    citation=(
        "Gerfen CR, Surmeier DJ. 'Modulation of striatal projection systems "
        "by dopamine.' Annu Rev Neurosci. 2011;34:441-466."
    ),
    pmid="21469956",
    doi="10.1146/annurev-neuro-061010-113641",
    species="Mus musculus / Rattus norvegicus",
    method="review of the direct/indirect pathway anatomy",
    modality="protein",
    scale="not a density source",
    note=(
        "Cited only for the mechanism of the Drd1 divergence: D1-expressing "
        "spiny projection neurons have their somata in the caudoputamen and "
        "their terminals in SNr/GPi, so nigral D1 protein is striatal in "
        "origin. Already cited by P3.3 for the same anatomy in "
        "`connectome_scenarios._STRIATAL_INDIRECT_RESTRICTION`."
    ),
)

BELLO2011 = AnatomySource(
    key="bello2011",
    citation=(
        "Bello EP, Mateo Y, Gelman DM, Noain D, Shin JH, Low MJ, Alvarez "
        "VA, Lovinger DM, Rubinstein M. 'Cocaine supersensitivity and "
        "enhanced motivation for reward in mice lacking dopamine D2 "
        "autoreceptors.' Nat Neurosci. 2011;14(8):1033-1038."
    ),
    pmid="21743470",
    doi="10.1038/nn.2862",
    species="Mus musculus",
    method="conditional deletion of Drd2 from dopaminergic neurons "
    "(Drd2-loxP x DAT-Cre), with the loss verified in midbrain tissue",
    modality="mRNA",
    scale="not a density source: presence/absence in one cell population",
    note=(
        "Cited for one fact the striatum-only autoradiography cannot "
        "supply: midbrain dopamine neurons of the mouse transcribe Drd2, so "
        "SNc and VTA are expected to carry Drd2 message while Drd1 message "
        "there stays at cortical levels. That contrast is the cross-gene "
        "check in `midbrain_dopamine_contrast()`, which is computed as a "
        "ratio of within-gene ratios so no expression energy is ever "
        "compared across genes."
    ),
)

YOO2010_SOURCE = AnatomySource(
    key="yoo2010",
    citation=YOO2010.citation,
    pmid=YOO2010.pmid,
    doi=YOO2010.doi,
    species=YOO2010.species,
    method=YOO2010.method,
    modality="protein",
    scale="quantitative, fmol/mg tissue equivalent",
    note=(
        "The same transcription P4.2 uses for its quantitative comparison, "
        "reused here for ordinal claims. D1 (B6 column): caudate putamen "
        "409.2, olfactory tubercle 386.0, accumbens core 248.8, substantia "
        "nigra 206.3, claustrum 87.3, cingulate cortex 34.4, hippocampus "
        "24.0, VTA 24.8, motor cortex 22.8, thalamus 20.7. D2: caudate "
        "putamen 70.4, olfactory tubercle 51.2, accumbens core 40.6, "
        "accumbens shell 30.7 and NOTHING else, because the paper reports "
        "[3H]raclopride binding outside the striatum as below the film's "
        "usable range - which is what makes 'D2 is striatal' a citable "
        "statement from this source rather than an absence of data."
    ),
)

ANATOMY_SOURCES = {
    source.key: source
    for source in (
        ERBS2015,
        HORTNAGL2013,
        WANG2022,
        MARSICANO1999,
        MAILLEUX1992,
        HERKENHAM1991,
        GERFEN2011,
        BELLO2011,
        YOO2010_SOURCE,
    )
}


class Claim(object):
    """One ordinal expectation about one gene's map.

    The assertion is `min(map[high]) > max(map[low])`: every named high
    region above every named low region. Nothing about the *size* of the
    difference is asserted, for the reason in rule 2 of the module
    docstring; `ClaimResult.ratio` reports it.
    """

    def __init__(self, symbol, high, low, source, separation, evidence,
                 expected_to_hold=True, mechanism="", mechanism_source=None,
                 note=""):
        if not high or not low:
            raise ValueError("a claim needs at least one high and one low region")
        overlap = set(high) & set(low)
        if overlap:
            raise ValueError(
                "%s: %s appears on both sides of the claim"
                % (symbol, sorted(overlap))
            )
        if separation < 0:
            raise ValueError("separation is a count of tiers and cannot be negative")
        if not expected_to_hold and not (mechanism and mechanism_source):
            raise ValueError(
                "a claim expected to fail must carry the mechanism that "
                "makes it fail and a citation for that mechanism, or it is "
                "just a dropped inconvenient result"
            )
        self.symbol = symbol
        self.high = tuple(high)
        self.low = tuple(low)
        self.source = source
        self.separation = float(separation)
        self.evidence = evidence
        self.expected_to_hold = bool(expected_to_hold)
        self.mechanism = mechanism
        self.mechanism_source = mechanism_source
        self.note = note

    def __repr__(self):
        return "Claim(%s: %s > %s%s)" % (
            self.symbol,
            "/".join(self.high),
            "/".join(self.low),
            "" if self.expected_to_hold else ", EXPECTED TO FAIL",
        )

    @property
    def assertable(self):
        """Whether the test suite asserts this claim. See rule 3."""
        return self.separation >= ASSERTABLE_SEPARATION

    @property
    def regions(self):
        return self.high + self.low

    def describe(self):
        return "%s: %s above %s [%s, %s, %s]" % (
            self.symbol,
            ", ".join(self.high),
            ", ".join(self.low),
            self.source.key,
            self.source.species,
            self.source.modality,
        )


# ---------------------------------------------------------------------------
# The claims
#
# Six receptors, chosen because each one is a receptor this project either
# already simulates (Oprm1, Drd2, Gabra1) or names in the roadmap's P4.3
# line (Drd1, Cnr1, Adora2a), and because between them they cover four
# different kinds of source: quantitative mouse autoradiography, two mouse
# knock-in protein atlases, and the one mouse ISH study whose modality
# matches the map exactly.
# ---------------------------------------------------------------------------

CLAIMS = (
    # --- Drd1 -------------------------------------------------------------
    Claim(
        symbol="Drd1",
        high=("CP", "ACB", "OT"),
        low=("MOp", "MD", "VTA", "CA1"),
        source=YOO2010_SOURCE,
        separation=4.0,
        evidence=(
            "[3H]SCH-23390 binding, B6 column: CP 409.2, OT 386.0, ACB core "
            "248.8 against motor cortex 22.8, thalamus 20.7, VTA 24.8, "
            "hippocampus 24.0 fmol/mg. The smallest striatal/extrastriatal "
            "ratio in that set is 10-fold, which on a quantitative source "
            "is scored well past the two-tier bar."
        ),
    ),
    Claim(
        symbol="Drd1",
        high=("CLA",),
        low=("MOp",),
        source=YOO2010_SOURCE,
        separation=2.0,
        evidence=(
            "Claustrum 87.3 against motor cortex 22.8 fmol/mg, 3.8-fold. "
            "Included deliberately: CLA is the single highest region of the "
            "Drd1 and Drd2 transcript maps, above CP, and this is the only "
            "independent mouse measurement of claustral D1 available to say "
            "whether that is anatomy or artefact."
        ),
    ),
    Claim(
        symbol="Drd1",
        high=("SNr",),
        low=("MOp",),
        source=YOO2010_SOURCE,
        separation=4.0,
        evidence=(
            "Substantia nigra 206.3 against motor cortex 22.8 fmol/mg, "
            "9-fold. Expected to FAIL on a transcript map."
        ),
        expected_to_hold=False,
        mechanism=(
            "Nigral D1 protein is on the terminals of striatonigral "
            "projection neurons whose somata, and therefore whose Drd1 "
            "message, are in the caudoputamen. P4.2 measured the resulting "
            "CP:SNr error at 32-fold; this claim is the same fact stated "
            "against cortex so that it reads as an anatomical expectation "
            "rather than as a ratio."
        ),
        mechanism_source=GERFEN2011,
    ),

    # --- Drd2 -------------------------------------------------------------
    Claim(
        symbol="Drd2",
        high=("CP", "ACB", "OT"),
        low=("MOp", "PL", "CA1"),
        source=YOO2010_SOURCE,
        separation=4.0,
        evidence=(
            "[3H]raclopride binding is reported for CP 70.4, OT 51.2, ACB "
            "core 40.6, ACB shell 30.7 fmol/mg and for nothing else in the "
            "brain, because the paper reports extrastriatal binding as "
            "below the film's usable range."
        ),
    ),
    Claim(
        symbol="Drd2",
        high=("SNc", "VTA"),
        low=("MOp", "PL"),
        source=BELLO2011,
        separation=2.0,
        evidence=(
            "Drd2 is deleted from midbrain dopamine neurons in this mouse "
            "line and the D2 autoreceptor function is lost, so those "
            "neurons transcribe Drd2; the isocortex has no comparable "
            "Drd2-defined population."
        ),
    ),

    # --- Oprm1 ------------------------------------------------------------
    Claim(
        symbol="Oprm1",
        high=("MH",),
        low=("MOp", "PL", "CA1", "CA3", "DG", "CBX"),
        source=ERBS2015,
        separation=3.0,
        evidence="MHb ++++ against M1/M2 +, PrL +, CA1/CA3 +, DG +, CbCx -.",
    ),
    Claim(
        symbol="Oprm1",
        high=("CP",),
        low=("MOp", "CBX"),
        source=ERBS2015,
        separation=2.0,
        evidence=(
            "CPu +++ against M1/M2 + and CbCx -. Striatal MOR is patchy "
            "(striosomal) and a structure unionize averages the patches with "
            "the matrix around them, so this is the claim of the six most "
            "likely to be compressed by the region-level resolution; it is "
            "asserted anyway because two tiers is two tiers."
        ),
    ),
    Claim(
        symbol="Oprm1",
        high=("PVT",),
        low=("MOp",),
        source=ERBS2015,
        separation=1.0,
        evidence="PV ++ against M1/M2 +. One tier: reported, not asserted.",
    ),
    Claim(
        symbol="Oprm1",
        high=("LC", "PAG", "VTA"),
        low=("CBX",),
        source=ERBS2015,
        separation=1.0,
        evidence=(
            "LC +, LPAG +/++ (DLPAG +, VLPAG +, DMPAG -), VTA ++ against "
            "CbCx -. One tier for LC and PAG: reported, not asserted. These "
            "are the three regions the multi-region mu-opioid scenario "
            "couples to, so their measured values are printed by the report "
            "whatever the outcome."
        ),
    ),
    Claim(
        symbol="Oprm1",
        high=("MOp",),
        low=("CBX",),
        source=ERBS2015,
        separation=1.0,
        evidence=(
            "M1/M2 + against CbCx -. One tier, at the bottom of the scale, "
            "between a 'moderate' and a 'not detectable': reported, not "
            "asserted. This is the claim P4.3 found to be the one ordinal "
            "disagreement of the six-receptor sweep, and its measured size "
            "is 1%, i.e. the two regions are indistinguishable in the "
            "transcript map rather than inverted in it."
        ),
    ),

    # --- Gabra1 -----------------------------------------------------------
    Claim(
        symbol="Gabra1",
        high=("MOp", "PL", "ACA"),
        low=("CP", "ACB"),
        source=HORTNAGL2013,
        separation=3.0,
        evidence=(
            "Cortex layer 4 +++, layers 2/3 and 6 ++, against caudate "
            "putamen (+) and accumbens core/shell (+): three tiers on this "
            "paper's own five-level scale, and the mouse ISH modality "
            "matches the map."
        ),
    ),
    Claim(
        symbol="Gabra1",
        high=("GPe", "SNr", "STN"),
        low=("CP", "ACB"),
        source=HORTNAGL2013,
        separation=3.0,
        evidence=(
            "Globus pallidus +++, subthalamic +++, substantia nigra "
            "reticulata +++ against caudate putamen (+). The basal ganglia "
            "reversal is the sharpest single test in this module: an "
            "alpha1 map that looked striatal would be a swapped gene row."
        ),
    ),
    Claim(
        symbol="Gabra1",
        high=("CA1",),
        low=("CA3",),
        source=HORTNAGL2013,
        separation=2.0,
        evidence=(
            "CA1 +++ against CA3 +, and the text states it in words: 'The "
            "alpha1 mRNA and protein were both abundantly found in the "
            "dentate gyrus and hilus, CA1 region as well as in the dorsal "
            "and ventral subiculum, whereas the expression in the CA3 area "
            "was considerably weaker.'"
        ),
    ),
    Claim(
        symbol="Gabra1",
        high=("CBX",),
        low=("CP",),
        source=HORTNAGL2013,
        separation=2.0,
        evidence=(
            "Cerebellar Purkinje layer +++ and granule layer ++ against "
            "caudate putamen (+). Note what this claim does NOT say: the "
            "roadmap's shorthand is 'Gabra1 high in cortex and cerebellum', "
            "but this paper scores cerebellar alpha1 mRNA BELOW cortical "
            "layer 4, and reports separately that in cerebellum (among "
            "other regions) 'mRNA levels did not reflect protein levels, "
            "indicating that the protein is located far distantly from the "
            "cell body'. The map agrees with the mRNA half."
        ),
    ),

    # --- Cnr1 -------------------------------------------------------------
    Claim(
        symbol="Cnr1",
        high=("CA1", "CA3", "DG", "MOp", "PL", "BLA"),
        low=("MD",),
        source=MARSICANO1999,
        separation=2.0,
        evidence=(
            "The mouse study works in hippocampus, cortex and amygdala "
            "because those are where CB1-expressing neurons are, and its "
            "high-expressing population is a CCK-positive interneuron "
            "class present in all three. The thalamic side of the claim is "
            "[Mailleux1992]'s regional survey, which names no dorsal "
            "thalamic nucleus among the CB1 mRNA regions."
        ),
    ),
    Claim(
        symbol="Cnr1",
        high=("SNr", "GPe"),
        low=("CP",),
        source=HERKENHAM1991,
        separation=2.0,
        evidence=(
            "'In rat brain, the highest receptor densities are in the "
            "globus pallidus and substantia nigra pars reticulata. "
            "Receptors are also dense in the caudate-putamen.' Expected to "
            "FAIL on a transcript map."
        ),
        expected_to_hold=False,
        mechanism=(
            "The same paper's lesion experiment shows why: ibotenic acid in "
            "the caudoputamen removed cannabinoid binding from the globus "
            "pallidus, entopeduncular nucleus and substantia nigra pars "
            "reticulata, so those receptors are on the axons and terminals "
            "of striatal projection neurons and the message that made them "
            "is in the striatum. `mismatch.py` records Cnr1 as a "
            "terminal-field gene for this reason and refuses to convert it "
            "to a density without an explicit acknowledgement."
        ),
        mechanism_source=HERKENHAM1991,
    ),

    # --- Adora2a ----------------------------------------------------------
    Claim(
        symbol="Adora2a",
        high=("CP", "ACB", "OT"),
        low=("MD", "VTA", "SNc", "CBX", "CA1", "MOp", "PL"),
        source=WANG2022,
        separation=3.0,
        evidence=(
            "'The most prominent and intense labeling was observed in the "
            "neuropil of the entire striatum and extended into the nucleus "
            "accumbens and olfactory tubercles' against 'no signal was "
            "detected in the midbrain or cerebellum', 'very lightly "
            "labeled cells' in CA1/CA2 and 'a few discrete and bright "
            "labeled cells' in deep cortical layers."
        ),
    ),
    Claim(
        symbol="Adora2a",
        high=("CP",),
        low=("GPe",),
        source=WANG2022,
        separation=1.0,
        evidence=(
            "'Immunoreactivity in the globus pallidus was lighter than in "
            "the striatum.' One tier: reported, not asserted. Worth "
            "reporting because pallidal A2A is the striatopallidal terminal "
            "field, i.e. the one place in this gene's map where the "
            "terminal-field mechanism could show up."
        ),
    ),
)

CLAIMS_BY_SYMBOL = {}
for _claim in CLAIMS:
    CLAIMS_BY_SYMBOL.setdefault(_claim.symbol, []).append(_claim)

#: The receptors this module validates. Five is the acceptance bar; six is
#: what there are honest mouse-or-explicitly-labelled sources for.
VALIDATED_SYMBOLS = tuple(sorted(CLAIMS_BY_SYMBOL))


def claim_regions():
    """Every region acronym any claim names, sorted."""
    seen = set()
    for claim in CLAIMS:
        seen.update(claim.regions)
    return sorted(seen)


class ClaimResult(object):
    """What one claim measured."""

    def __init__(self, claim, values, missing):
        self.claim = claim
        self.values = dict(values)
        self.missing = tuple(missing)
        highs = [self.values[r] for r in claim.high if r in self.values]
        lows = [self.values[r] for r in claim.low if r in self.values]
        self.min_high = min(highs) if highs else float("nan")
        self.max_low = max(lows) if lows else float("nan")
        if self.missing:
            self.holds = None
        else:
            self.holds = bool(self.min_high > self.max_low)

    @property
    def ratio(self):
        """min(high) / max(low). Reported, never asserted against."""
        if not np.isfinite(self.min_high) or not np.isfinite(self.max_low):
            return float("nan")
        if self.max_low <= 0.0:
            return float("inf")
        return self.min_high / self.max_low

    @property
    def ok(self):
        """Did the map do what the claim's record says it should?

        `None` when a region has no measurement, because an unmeasured
        region is not evidence either way - the same refusal to read NaN as
        zero that P4.1 makes in the loader.
        """
        if self.holds is None:
            return None
        return self.holds == self.claim.expected_to_hold

    def __repr__(self):
        return "ClaimResult(%r, holds=%r, ratio=%.3g)" % (
            self.claim.describe(),
            self.holds,
            self.ratio,
        )

    def summary(self):
        state = "unmeasured" if self.holds is None else (
            "as expected" if self.ok else "UNEXPECTED"
        )
        return "%-8s %-42s min(high)=%.4g max(low)=%.4g ratio=%.3g  %s" % (
            self.claim.symbol,
            "%s > %s" % ("/".join(self.claim.high), "/".join(self.claim.low)),
            self.min_high,
            self.max_low,
            self.ratio,
            state,
        )


def check(expression, claims=CLAIMS, ontology=None):
    """Evaluate claims against a `RegionalExpression`.

    `expression` must carry an ontology (or one must be passed) because
    claims name regions by acronym: an acronym is what a citation gives you
    and a structure id is what the data is keyed by, and doing that join in
    one place is what keeps a mistyped id from becoming a silently absent
    region.
    """
    ontology = ontology or expression.ontology
    if ontology is None:
        raise ValueError(
            "claims name regions by acronym, so an ontology is required to "
            "resolve them against the matrix's structure ids"
        )
    index = {int(i): n for n, i in enumerate(expression.structure_ids)}
    results = []
    for claim in claims:
        values, missing = {}, []
        row = expression.for_gene(claim.symbol)
        for acronym in claim.regions:
            structure_id = int(ontology.id_for_acronym(acronym))
            position = index.get(structure_id)
            if position is None:
                missing.append(acronym)
                continue
            value = float(row[position])
            if not np.isfinite(value):
                missing.append(acronym)
                continue
            values[acronym] = value
        results.append(ClaimResult(claim, values, missing))
    return results


def check_atlas(atlas, symbols=None, plane=DEFAULT_PLANE,
                combine=DEFAULT_COMBINE, ontology=None, claims=CLAIMS):
    """`check()` over exactly the regions the claims name.

    Deliberately not over the connectome's 291 regions: several claims name
    structures the connectome does not carry a row for (CBX, CA1, MH), and
    a validation that could only see modelled regions would be unable to
    test the claims that most cheaply detect a mis-join.
    """
    ontology = ontology or atlas.ontology
    if symbols is not None:
        claims = [c for c in claims if c.symbol in set(symbols)]
    wanted = sorted(set(r for c in claims for r in c.regions))
    structure_ids = [int(ontology.id_for_acronym(a)) for a in wanted]
    gene_symbols = sorted(set(c.symbol for c in claims))
    expression = atlas.energy_matrix(
        structure_ids, symbols=gene_symbols, plane=plane, combine=combine
    )
    return check(expression, claims=claims, ontology=ontology)


def failures(results):
    """The results whose outcome is not what their claim's record says."""
    return [r for r in results if r.ok is False]


def unmeasured(results):
    return [r for r in results if r.ok is None]


def asserted(results):
    """Results the test suite asserts on. See rule 3 of the docstring."""
    return [r for r in results if r.claim.assertable]


def midbrain_dopamine_contrast(expression):
    """Is Drd2 midbrain-weighted relative to Drd1, in the same map?

    The one cross-gene check in this module, and the shape it has to take
    to be legitimate. Expression energy is NOT comparable between genes -
    probe and segmentation threshold differ per experiment - so this
    compares a *ratio of within-gene ratios*:

        (Drd2[VTA] / Drd2[CP])  /  (Drd1[VTA] / Drd1[CP])

    Both halves are dimensionless and both are internal to one gene, so the
    probe difference cancels rather than being ignored. A value well above
    1 says the two gene rows carry different anatomy in the expected
    direction: D2 is an autoreceptor on midbrain dopamine neurons
    [Bello2011] while D1 in the midbrain sits at cortical levels
    ([Yoo2010]: VTA 24.8 against motor cortex 22.8 fmol/mg). A value near 1
    would say the rows had been shuffled into each other.
    """
    ontology = expression.ontology
    index = {int(i): n for n, i in enumerate(expression.structure_ids)}

    def value(symbol, acronym):
        position = index[int(ontology.id_for_acronym(acronym))]
        return float(expression.for_gene(symbol)[position])

    out = {}
    for region in ("VTA", "SNc"):
        d2 = value("Drd2", region) / value("Drd2", "CP")
        d1 = value("Drd1", region) / value("Drd1", "CP")
        out[region] = d2 / d1
    return out


def policy_agreement(atlas, ontology=None, planes=("coronal", "coronal-else-sagittal",
                                                   "any"),
                     combines=("mean", "median", "max")):
    """Every claim's outcome under every plane x combine policy.

    P4.1 refused to default the two policies and P4.2 measured what the
    refusal was worth for its D1 correlation. This is the same measurement
    for these claims: `{(plane, combine): n_unexpected}`, so a policy that
    quietly changes an anatomical conclusion cannot hide inside the one
    policy the tests happen to use.
    """
    out = {}
    for plane in planes:
        for combine in combines:
            results = check_atlas(
                atlas, plane=plane, combine=combine, ontology=ontology
            )
            out[(plane, combine)] = {
                "unexpected": len(failures(results)),
                "unmeasured": len(unmeasured(results)),
            }
    return out


def mismatch_context(symbol):
    """The `mismatch.py` record for a gene, if it has one.

    P4.2 flagged four genes whose protein is not where their transcript is.
    Two of this module's expected failures are exactly those genes, and
    tying the two records together in code keeps the anatomy claim and the
    conversion warning from drifting apart.
    """
    return _mismatch.mismatch_for(symbol)
