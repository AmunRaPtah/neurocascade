"""P4.2: published receptor *protein* densities, to test the conversion against.

Why this module exists
----------------------

P4.2 converts Allen ISH expression energy into the density scale the
receptor layer consumes. The conversion rests on an assumption - density is
proportional to transcript abundance - and an assumption that is never
checked against an independent measurement is indistinguishable from a
guess. This module holds the independent measurement: *quantitative in vitro
receptor autoradiography in mouse brain*, region by region, in fmol/mg,
transcribed from published tables with the ligand, displacer, strain, sex,
age and animal count each number was measured under.

Nothing here is fitted, smoothed, rescaled or averaged across studies. Each
`LigandBinding` is one published table column. Where a paper reports several
genetic backgrounds, all of them are recorded, because the spread between
them is the only estimate of measurement uncertainty available and dropping
it would make the comparison look tighter than the data is.

The two datasets, and why these two
-----------------------------------

**Yoo et al. 2010** is the primary. It is mouse, it is quantitative, its D1
table covers fifteen regions spanning a twenty-fold density range including
the striatum and the cortex, and - the reason it is primary rather than merely
useful - it measured D1 receptors, the dopamine transporter and D2
receptors *in the same animals with the same films*, which is exactly the
combination needed to separate "the conversion is wrong" from "this
particular receptor is trafficked away from the cell that transcribes it".
Its D1 table is the acceptance-criterion comparison; its [3H]mazindol
table is what makes the DAT terminal-field mismatch a measured number
rather than a caveat (see `mismatch.py`).

**Lothmann et al. 2021** is the breadth check. Fourteen regions of the
olfactory system, ten tritiated ligands, converted to Bmax at saturation by
the Zilles-lab protocol, so seven of this project's receptor systems can be
compared at once instead of one. It is the evidence that the conversion's
quality is *receptor-specific*: over the same fourteen regions the
correlation ranges from strongly positive to indistinguishable from zero
depending on which receptor is asked about.

Both are open access and both are mouse, which matters: substituting rat
autoradiography for a mouse expression map would be exactly the
species-substitution this project refuses.

What a comparison here can and cannot establish
-----------------------------------------------

It is a *rank and scale* comparison across regions within one receptor, and
nothing else:

- **Never across genes.** Expression energy is not comparable between
  genes - probe, riboprobe length and segmentation threshold all differ per
  experiment - so "Gabra1 is bigger than Drd1" is not a claim this data can
  make, and no function here computes one.
- **Region delineations do not match.** The autoradiography papers
  delineate regions on their own cytoarchitectonic atlases (Hof et al. 2000
  for Lothmann; Franklin & Paxinos for Yoo), while the ISH unionizes are
  over CCF structures. `REGION_MAPPING` below is a name-based join, region
  by region, with every judgment call written down and every region that
  cannot be mapped unambiguously listed as unmapped rather than forced.
- **A single-concentration binding value is not a Bmax.** Yoo et al.
  incubated at one 4 nM concentration, so their numbers are proportional to
  Bmax only where Kd is regionally uniform. Lothmann et al. converted to
  Bmax with the ligand's Kd, which is the stronger quantity. Each
  `LigandBinding` records which it is in `quantity`.
"""

import numpy as np

#: Sentinel for a published region that deliberately has no CCF counterpart.
UNMAPPED = None


class LigandBinding(object):
    """One published table column: one ligand's regional binding densities.

    `values` is keyed by the *published* region name, not by CCF acronym, so
    that what is stored is what was printed. `REGION_MAPPING` does the join,
    and `mapped()` applies it.
    """

    def __init__(
        self,
        receptor,
        ligand,
        concentration_nm,
        displacer,
        quantity,
        units,
        genes,
        dataset,
        values,
        note="",
    ):
        self.receptor = receptor
        self.ligand = ligand
        self.concentration_nm = concentration_nm
        self.displacer = displacer
        self.quantity = quantity
        self.units = units
        self.genes = tuple(genes)
        self.dataset = dataset
        self.values = dict(values)
        self.note = note
        for name, value in self.values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "%s in %r is %r; a published binding density is finite "
                    "and positive" % (self.receptor, name, value)
                )

    def __repr__(self):
        return "LigandBinding(%r, %s, %d regions)" % (
            self.receptor,
            self.ligand,
            len(self.values),
        )

    def mapped(self, ontology=None):
        """`{CCF acronym: density}`, dropping the regions with no mapping.

        The dropped ones are not silently dropped: `unmapped()` lists them
        with the reason each was left out.
        """
        out = {}
        for name, value in self.values.items():
            acronym = REGION_MAPPING[name].acronym
            if acronym is UNMAPPED:
                continue
            if acronym in out:
                raise ValueError(
                    "two published regions map to %s; the join has to be "
                    "one-to-one or the merge rule becomes an invisible "
                    "modelling choice" % acronym
                )
            out[acronym] = value
        if ontology is not None:
            for acronym in out:
                ontology.id_for_acronym(acronym)
        return out

    def unmapped(self):
        """`[(published name, why it has no CCF counterpart)]`."""
        return [
            (name, REGION_MAPPING[name].reason)
            for name in self.values
            if REGION_MAPPING[name].acronym is UNMAPPED
        ]


class AutoradiographyDataset(object):
    """One published study, and the conditions its numbers were measured in."""

    def __init__(
        self,
        key,
        citation,
        pmid,
        doi,
        species,
        strain,
        sex,
        age,
        n_animals,
        method,
        note="",
    ):
        self.key = key
        self.citation = citation
        self.pmid = pmid
        self.doi = doi
        self.species = species
        self.strain = strain
        self.sex = sex
        self.age = age
        self.n_animals = n_animals
        self.method = method
        self.note = note

    def __repr__(self):
        return "AutoradiographyDataset(%r)" % (self.key,)


class MappedRegion(object):
    """One published region name and the CCF structure it was joined to."""

    __slots__ = ("published", "acronym", "reason")

    def __init__(self, published, acronym, reason):
        self.published = published
        self.acronym = acronym
        self.reason = reason


def _mapping(rows):
    table = {}
    for published, acronym, reason in rows:
        table[published] = MappedRegion(published, acronym, reason)
    return table


# ---------------------------------------------------------------------------
# The published-region -> CCF-structure join
# ---------------------------------------------------------------------------
#
# Every entry is a judgment, so every entry has a reason. The ones that are
# left unmapped are left unmapped on purpose: a coarse published region with
# no single CCF counterpart, or a fibre bundle where "expression energy"
# means nothing, would each contribute a comparison point whose disagreement
# says something about the atlas join rather than about the conversion.

REGION_MAPPING = _mapping([
    # --- Yoo et al. 2010 -----------------------------------------------
    ("Caudate putamen", "CP", "one CCF structure, same name"),
    (
        "Nucleus accumbens core",
        UNMAPPED,
        "CCF's ACB is undivided, so core and shell would both join to it; "
        "merging them here would be an averaging rule invented by this "
        "module. `ACCUMBENS_SUBDIVISIONS` records the two values instead.",
    ),
    (
        "Nucleus accumbens shell",
        UNMAPPED,
        "the other half of the same problem: CCF's ACB is one structure and "
        "this paper measured two, so joining either one to ACB would be "
        "picking a subdivision to stand for the whole",
    ),
    ("Olfactory tubercle", "OT", "one CCF structure, same name"),
    ("Claustrum", "CLA", "one CCF structure, same name"),
    (
        "Dorsal endopiriform",
        "EPd",
        "CCF: endopiriform nucleus, dorsal part",
    ),
    (
        "Cingulate cortex",
        "ACA",
        "CCF's anterior cingulate area; the paper's coronal levels are "
        "forebrain, so the retrosplenial (posterior cingulate) cortex is "
        "not what was sampled",
    ),
    ("Motor cortex", "MO", "CCF's somatomotor areas, MOp plus MOs"),
    (
        "Nigrostriatal bundle",
        UNMAPPED,
        "a fibre tract. Its D1 binding is receptor protein in axonal "
        "transit, which is the mechanism `mismatch.py` is about, and it is "
        "precisely why a transcript map cannot have a counterpart for it.",
    ),
    (
        "Amygdala",
        UNMAPPED,
        "not one CCF structure: the amygdalar nuclei are split between "
        "CTXsp (BLA, LA, BMA, PA) and sAMY (CEA, MEA, IA), and choosing a "
        "subset to stand for 'amygdala' would be inventing the join",
    ),
    ("Thalamus", "TH", "the CCF division"),
    ("Hypothalamus", "HY", "the CCF division"),
    (
        "Hippocampus",
        "HIP",
        "CCF's hippocampal region (CA fields plus DG), not HPF, which also "
        "contains the retrohippocampal areas (ENT, SUB, ...) that a coronal "
        "'hippocampus' region of interest does not include",
    ),
    (
        "Substantia nigra",
        UNMAPPED,
        "CCF has SNc and SNr as siblings with no combined parent, and the "
        "two differ by an order of magnitude in Drd1 and Slc6a3 transcript. "
        "`SUBSTANTIA_NIGRA_ALTERNATIVES` carries the value against both, "
        "and `mismatch.py` uses it as the terminal-field case rather than "
        "burying it in a correlation.",
    ),
    (
        "Subthalamic area",
        UNMAPPED,
        "'subthalamic area' is not CCF's STN: the paper's phrase covers the "
        "subthalamic region around it, and the join would be a guess",
    ),
    ("Ventral tegmental area", "VTA", "one CCF structure, same name"),
    # --- Lothmann et al. 2021 -------------------------------------------
    ("Main olfactory bulb", "MOB", "one CCF structure, same name"),
    ("Accessory olfactory bulb", "AOB", "one CCF structure, same name"),
    (
        "Anterior olfactory cortex",
        "AON",
        "the paper's AOC is CCF's anterior olfactory nucleus; its own "
        "figure legend gives AOC for 'anterior olfactory nucleus'",
    ),
    ("Taenia tecta, dorsal", "TTd", "CCF: taenia tecta, dorsal part"),
    ("Taenia tecta, ventral", "TTv", "CCF: taenia tecta, ventral part"),
    ("Dorsal peduncular cortex", "DP", "CCF: dorsal peduncular area"),
    (
        "Endopiriform nucleus (dorsal)",
        "EPd",
        "CCF: endopiriform nucleus, dorsal part",
    ),
    ("Piriform cortex", "PIR", "CCF: piriform area"),
    (
        "Entorhinal cortex, lateral",
        "ENTl",
        "CCF: entorhinal area, lateral part",
    ),
    (
        "Entorhinal cortex, medial",
        "ENTm",
        "CCF: entorhinal area, medial part, dorsal zone. The CCF also has "
        "ENTmv; a coronal ROI labelled 'medial entorhinal cortex' is the "
        "dorsal zone at these levels.",
    ),
    ("Orbitofrontal cortex, medial", "ORBm", "CCF: orbital area, medial part"),
    (
        "Orbitofrontal cortex, ventrolateral",
        "ORBvl",
        "CCF: orbital area, ventrolateral part",
    ),
    ("Orbitofrontal cortex, lateral", "ORBl", "CCF: orbital area, lateral part"),
])


# ---------------------------------------------------------------------------
# Yoo et al. 2010
# ---------------------------------------------------------------------------

YOO2010 = AutoradiographyDataset(
    key="yoo2010",
    citation=(
        "Yoo JH, Bailey A, Ansonoff M, Pintar JE, Matifas A, Kieffer BL, "
        "Kitchen I. Lack of genotype effect on D1, D2 receptors and dopamine "
        "transporter binding in triple MOP-, DOP-, and KOP-opioid receptor "
        "knockout mice of three different genetic backgrounds. "
        "Synapse. 2010;64(7):520-527."
    ),
    pmid="20196137",
    doi="10.1002/syn.20757",
    species="Mus musculus",
    strain="wild-type of three backgrounds: congenic C57BL/6 (B6), "
    "coisogenic 129, and B6 x 129",
    sex="male",
    age=">9 weeks",
    n_animals="5-8 per genotype per background",
    method=(
        "quantitative in vitro receptor autoradiography, 20 um coronal "
        "sections, single-concentration incubation at 4 nM, tritium "
        "microscale standards, densitometry"
    ),
    note=(
        "The knockout arm is irrelevant here and is not recorded: only the "
        "wild-type columns are transcribed. The paper's finding is that "
        "genotype did not matter, so the wild-type values are ordinary "
        "mouse dopamine-system densities and not a lesioned preparation."
    ),
)

#: `{region: (B6, 129, B6 x 129)}` wild-type values, fmol/mg tissue
#: equivalent, transcribed from Yoo et al. 2010 Tables I-III. The three
#: columns are three genetic backgrounds measured on the same films.
_YOO_D1 = {
    "Caudate putamen": (409.2, 393.0, 329.4),
    "Nucleus accumbens core": (248.8, 236.0, 211.3),
    "Nucleus accumbens shell": (226.6, 235.6, 196.3),
    "Olfactory tubercle": (386.0, 330.5, 247.4),
    "Claustrum": (87.3, 102.1, 71.2),
    "Dorsal endopiriform": (93.4, 95.9, 71.3),
    "Cingulate cortex": (34.4, 35.7, 27.6),
    "Motor cortex": (22.8, 24.1, 22.7),
    "Nigrostriatal bundle": (91.3, 122.1, 69.6),
    "Amygdala": (48.2, 49.2, 30.2),
    "Thalamus": (20.7, 24.3, 17.9),
    "Hypothalamus": (23.5, 21.3, 12.8),
    "Hippocampus": (24.0, 33.2, 24.7),
    "Substantia nigra": (206.3, 232.5, 118.0),
    "Ventral tegmental area": (24.8, 31.5, 17.7),
}

_YOO_D2 = {
    "Caudate putamen": (70.4, 89.1, 78.2),
    "Olfactory tubercle": (51.2, 62.9, 58.5),
    "Nucleus accumbens core": (40.6, 48.9, 39.8),
    "Nucleus accumbens shell": (30.7, 43.5, 36.8),
}

_YOO_DAT = {
    "Caudate putamen": (111.4, 108.3, 113.4),
    "Olfactory tubercle": (67.9, 87.1, 91.1),
    "Nucleus accumbens core": (73.1, 68.2, 60.8),
    "Nucleus accumbens shell": (58.0, 48.9, 38.8),
    "Subthalamic area": (52.2, 41.9, 54.6),
    "Substantia nigra": (77.1, 62.3, 77.5),
    "Ventral tegmental area": (75.7, 63.0, 114.5),
}

#: Which of the three backgrounds a comparison uses by default, and why.
#: Allen's ISH panel is C57BL/6J, so B6 is the background whose animals are
#: closest to the animals the transcript map was measured in. The other two
#: are kept so `background_spread()` can say how much that choice is worth.
PRIMARY_BACKGROUND = "B6"
BACKGROUNDS = ("B6", "129", "B6 x 129")


def _column(table, background):
    index = BACKGROUNDS.index(background)
    return {name: values[index] for name, values in table.items()}


def _yoo_binding(receptor, ligand, nm, displacer, genes, table, note,
                 background=PRIMARY_BACKGROUND):
    return LigandBinding(
        receptor=receptor,
        ligand=ligand,
        concentration_nm=nm,
        displacer=displacer,
        quantity="specific binding at a single %g nM concentration; NOT a "
        "saturation Bmax, so it tracks Bmax only where Kd is regionally "
        "uniform" % nm,
        units="fmol/mg tissue equivalent",
        genes=genes,
        dataset=YOO2010,
        values=_column(table, background),
        note=note,
    )


def yoo2010_d1(background=PRIMARY_BACKGROUND):
    return _yoo_binding(
        "D1", "[3H]SCH-23390", 4.0, "10 uM cis-flupentixol", ("Drd1", "Drd5"),
        _YOO_D1,
        "SCH-23390 does not separate D1 from D5, and this project has no "
        "Drd5 map (Drd5 is not in genes.RECEPTOR_GENES). Drd5 transcript is "
        "sparse relative to Drd1 outside the hippocampus and thalamus, but "
        "the comparison is against Drd1 alone and that is a real "
        "attribution gap, not a rounding error. 1 mM mianserin was "
        "co-incubated to block 5-HT2/5-HT1c binding.",
        background,
    )


def yoo2010_d2(background=PRIMARY_BACKGROUND):
    return _yoo_binding(
        "D2", "[3H]raclopride", 4.0, "10 uM sulpiride", ("Drd2",), _YOO_D2,
        "Only four regions, two of them the accumbens core and shell that "
        "CCF does not separate: raclopride binding outside the striatum was "
        "below the film's usable range. Two joinable points is not a rank "
        "correlation and no function here computes one; the value is as an "
        "ordering check, CP above OT in both transcript and binding.",
        background,
    )


def yoo2010_dat(background=PRIMARY_BACKGROUND):
    return _yoo_binding(
        "DAT", "[3H]mazindol", 4.0, "10 uM mazindol", ("Slc6a3",), _YOO_DAT,
        "0.3 uM desipramine was co-incubated to block the noradrenaline "
        "transporter, so this is DAT and not NET. This is the table that "
        "makes the dopamine transporter's soma-versus-terminal mismatch a "
        "measured number; see mismatch.py.",
        background,
    )


#: The two accumbens subdivisions Yoo et al. report and CCF does not have.
#: Kept out of the joined comparison (see `REGION_MAPPING`) and kept here so
#: they are still readable, since ACB is a region this project's scenarios
#: actually use.
ACCUMBENS_SUBDIVISIONS = {
    "D1": {
        "Nucleus accumbens core": _YOO_D1["Nucleus accumbens core"],
        "Nucleus accumbens shell": _YOO_D1["Nucleus accumbens shell"],
    },
    "D2": {
        "Nucleus accumbens core": _YOO_D2["Nucleus accumbens core"],
        "Nucleus accumbens shell": _YOO_D2["Nucleus accumbens shell"],
    },
    "DAT": {
        "Nucleus accumbens core": _YOO_DAT["Nucleus accumbens core"],
        "Nucleus accumbens shell": _YOO_DAT["Nucleus accumbens shell"],
    },
}

#: Yoo et al.'s "substantia nigra" against each CCF structure it could mean.
#: The value is the same number twice; what differs is the transcript it
#: gets compared against, and the two differ enough that reporting one
#: without the other would be a choice disguised as a lookup.
SUBSTANTIA_NIGRA_ALTERNATIVES = ("SNc", "SNr")


# ---------------------------------------------------------------------------
# Lothmann et al. 2021
# ---------------------------------------------------------------------------

LOTHMANN2021 = AutoradiographyDataset(
    key="lothmann2021",
    citation=(
        "Lothmann K, Amunts K, Herold C. The Neurotransmitter Receptor "
        "Architecture of the Mouse Olfactory System. "
        "Front Neuroanat. 2021;15:632549."
    ),
    pmid="33967704",
    doi="10.3389/fnana.2021.632549",
    species="Mus musculus",
    strain="C57BL/6 (Janvier)",
    sex="male",
    age="27 weeks",
    n_animals="10",
    method=(
        "quantitative in vitro receptor autoradiography, 16 um coronal "
        "sections, tritium microscale standards, densities converted to "
        "Bmax at saturation with each ligand's Kd (Zilles et al. 2002 "
        "protocol)"
    ),
    note=(
        "Regions were delineated on the Hof et al. 2000 cytoarchitectonic "
        "atlas of the C57BL/6 mouse, not on the CCF. The published table "
        "prints the NMDA column twice; it is stored once here."
    ),
)

#: Lothmann et al. 2021 Table 1, mean Bmax in fmol/mg protein. Column order:
#: AMPA, kainate, NMDA, mGlu2/3, GABA-A, GABA-A(BZ), GABA-B, alpha1, alpha2,
#: D1/5.
_LOTHMANN_COLUMNS = (
    "AMPA", "kainate", "NMDA", "mGlu2/3", "GABA-A", "GABA-A(BZ)",
    "GABA-B", "alpha1", "alpha2", "D1/5",
)
_LOTHMANN = {
    "Main olfactory bulb": (727, 1281, 954, 2180, 803, 3867, 2144, 547, 497, 61),
    "Accessory olfactory bulb": (909, 1328, 1203, 3948, 950, 6409, 3617, 675, 547, 100),
    "Anterior olfactory cortex": (1044, 1432, 1445, 2365, 566, 2038, 4223, 338, 1809, 171),
    "Taenia tecta, dorsal": (1151, 1356, 1878, 2977, 906, 3662, 4736, 308, 1589, 357),
    "Taenia tecta, ventral": (1361, 1022, 1716, 2825, 788, 3214, 4316, 321, 1377, 315),
    "Dorsal peduncular cortex": (1069, 1624, 1909, 3442, 1048, 4125, 5048, 317, 1031, 360),
    "Endopiriform nucleus (dorsal)": (789, 1480, 1239, 1836, 695, 2791, 4059, 238, 1032, 888),
    "Piriform cortex": (1060, 785, 1821, 3328, 1034, 3961, 4853, 448, 1027, 242),
    "Entorhinal cortex, lateral": (1664, 1054, 3037, 3070, 1297, 3627, 5894, 314, 1771, 345),
    "Entorhinal cortex, medial": (1580, 1040, 2091, 3081, 1190, 3553, 6201, 304, 1770, 330),
    "Orbitofrontal cortex, medial": (1089, 1095, 2292, 3867, 1553, 4894, 5925, 502, 650, 175),
    "Orbitofrontal cortex, ventrolateral": (1034, 995, 2101, 4024, 1520, 4762, 5909, 495, 570, 131),
    "Orbitofrontal cortex, lateral": (1100, 1039, 2168, 3983, 1393, 4269, 5003, 513, 564, 211),
    "Olfactory tubercle": (1079, 964, 1638, 3964, 853, 3048, 2995, 204, 770, 5340),
}

#: Ligand, and the genes of this project whose product it labels. A gene
#: absent from `genes.RECEPTOR_GENES` is named anyway, because an unmodelled
#: subunit contributing to the measured signal is an attribution gap that
#: has to be visible.
_LOTHMANN_LIGANDS = {
    "AMPA": (
        "[3H]AMPA", ("Gria1", "Gria2", "Gria3", "Gria4"),
        "labels the assembled AMPA receptor; all four GluA subunit maps are "
        "ingested, and no rule says which subunit's transcript should "
        "predict the tetramer's binding",
    ),
    "kainate": (
        "[3H]kainate", (),
        "no kainate-receptor gene (Grik*) is ingested; recorded for "
        "completeness and not comparable",
    ),
    "NMDA": (
        "[3H]MK-801", ("Grin1", "Grin2a", "Grin2b"),
        "MK-801 is an open-channel blocker, so it labels the *activatable* "
        "channel population under the assay's activating conditions rather "
        "than every assembled receptor",
    ),
    "mGlu2/3": (
        "[3H]LY-341495", (),
        "no metabotropic glutamate gene (Grm*) is ingested",
    ),
    "GABA-A": (
        "[3H]muscimol", (
            "Gabra1", "Gabra2", "Gabra3", "Gabra4", "Gabra5",
            "Gabrb1", "Gabrb2", "Gabrb3",
        ),
        "the orthosteric GABA site of every assembled GABA-A pentamer, "
        "benzodiazepine-sensitive or not",
    ),
    "GABA-A(BZ)": (
        "[3H]flumazenil", ("Gabra1", "Gabra2", "Gabra3", "Gabra5"),
        "the benzodiazepine site, which exists only on alpha1/2/3/5- "
        "containing pentamers. This is the site this codebase's diazepam "
        "scenario acts at, and alpha4 is excluded here for exactly the "
        "reason genes.py ingests it separately.",
    ),
    "GABA-B": (
        "[3H]CGP-54626", (),
        "no GABA-B gene (Gabbr1/2) is ingested",
    ),
    "alpha1": (
        "[3H]prazosin", ("Adra1a", "Adra1d"),
        "alpha-1A + alpha-1D only: Adra1b has no Allen ISH experiment at "
        "all (sources.GENES_WITHOUT_DATA), so part of the measured signal "
        "has no transcript map by construction",
    ),
    "alpha2": (
        "[3H]RX-821002", ("Adra2a", "Adra2b", "Adra2c"),
        "all three alpha-2 subtypes are ingested",
    ),
    "D1/5": (
        "[3H]SCH-23390", ("Drd1", "Drd5"),
        "D1 and D5 together; Drd5 is not ingested",
    ),
}


def lothmann2021(receptor):
    """One receptor's column of Lothmann et al. 2021 Table 1."""
    if receptor not in _LOTHMANN_COLUMNS:
        raise KeyError(
            "%r is not a column of this table; it has %s"
            % (receptor, ", ".join(_LOTHMANN_COLUMNS))
        )
    index = _LOTHMANN_COLUMNS.index(receptor)
    ligand, genes, note = _LOTHMANN_LIGANDS[receptor]
    return LigandBinding(
        receptor=receptor,
        ligand=ligand,
        concentration_nm=None,
        displacer=None,
        quantity="Bmax at saturation, computed from the measured binding "
        "and the ligand's Kd",
        units="fmol/mg protein",
        genes=genes,
        dataset=LOTHMANN2021,
        values={name: row[index] for name, row in _LOTHMANN.items()},
        note=note,
    )


def lothmann2021_receptors():
    """Every column of the table, in published order."""
    return list(_LOTHMANN_COLUMNS)


def comparable_lothmann_receptors():
    """The columns at least one ingested gene can be compared against."""
    return [r for r in _LOTHMANN_COLUMNS if _LOTHMANN_LIGANDS[r][1]]


def background_spread():
    """How much Yoo et al.'s three genetic backgrounds disagree.

    The honest error bar on any single number taken from that table: same
    films, same protocol, same species, three mouse backgrounds. Returned as
    `{receptor: (median max/min ratio, worst region, worst ratio)}`.
    """
    out = {}
    for receptor, table in (("D1", _YOO_D1), ("D2", _YOO_D2), ("DAT", _YOO_DAT)):
        ratios = []
        for name, values in table.items():
            ratios.append((max(values) / min(values), name))
        ratios.sort()
        out[receptor] = (
            float(np.median([r for r, _ in ratios])),
            ratios[-1][1],
            float(ratios[-1][0]),
        )
    return out


DATASETS = {
    YOO2010.key: YOO2010,
    LOTHMANN2021.key: LOTHMANN2021,
}


# ---------------------------------------------------------------------------
# Searched and empty
# ---------------------------------------------------------------------------
#
# Track E's template: a search that came back unusable is a result, and the
# search is recorded so nobody repeats it and nobody quietly substitutes
# something else for it.

SEARCHED_AND_EMPTY = (
    {
        "receptor": "mu-opioid (Oprm1)",
        "why_wanted": (
            "Oprm1 is the receptor of this codebase's fentanyl scenarios, so "
            "it is the single most valuable comparison available - more "
            "valuable than D1, which no scenario currently uses."
        ),
        "searched": (
            "2026-08-12. Kitchen I, Slowe SJ, Matthes HW, Kieffer B. "
            "'Quantitative autoradiographic mapping of mu-, delta- and "
            "kappa-opioid receptors in knockout mice lacking the mu-opioid "
            "receptor gene.' Brain Res. 1997;778(1):73-88, PMID 9462879, is "
            "the canonical region-wise mouse [3H]DAMGO table and is behind "
            "an Elsevier paywall with no PMC deposit. A Europe PMC "
            "open-access full-text search for DAMGO + autoradiography + "
            "mouse + fmol/mg returned 14 records; the only one with "
            "region-wise mouse MOPr densities is Effah F, Nidadavolu P, de "
            "Gusmao Taveiros Silva NK, Wojtowicz M, Camarini R, Zimmer A, "
            "Bailey A. 'Differential temporal decline of cerebral oxytocin "
            "and mu-opioid receptor density during the aging process in "
            "mice.' Eur J Neurosci. 2024;60(11):6686-6703, PMC11612844, "
            "which prints its per-region values only as bar figures; its "
            "one numeric MOPr table is whole-brain means by age group."
        ),
        "conclusion": (
            "No Oprm1 comparison is made. The abundant RAT mu-opioid "
            "autoradiography was deliberately not substituted: a rat density "
            "map compared against a mouse transcript map would measure the "
            "species difference as if it were the conversion's error. What "
            "Oprm1 has instead is P4.1's qualitative anatomy check (medial "
            "habenula, then midline thalamus) and an unquantified "
            "terminal-field record in mismatch.py."
        ),
    },
    {
        "receptor": "D2 outside the striatum (Drd2)",
        "why_wanted": (
            "Drd2 is the receptor of the haloperidol scenario, and the four "
            "regions Yoo et al. report are all striatal, so the comparison "
            "has no dynamic range."
        ),
        "searched": (
            "2026-08-12. Yoo et al. 2010 state that [3H]raclopride binding "
            "outside the striatum was below their film's usable range, which "
            "is a property of the ligand and the method rather than of that "
            "study. No open-access mouse table with extrastriatal D2 "
            "densities in comparable units was found alongside it."
        ),
        "conclusion": (
            "Drd2 gets an ordering check on two joinable regions (CP above "
            "OT in both transcript and binding) and no correlation. Two "
            "points is not a correlation and none is computed."
        ),
    },
)
