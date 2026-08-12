"""Small committed fixtures: real connectome data, fifteen regions of it.

Why the fixtures are real data and not a synthetic matrix
---------------------------------------------------------

The tests have to run with no network (rule: the gate runs offline), and
the published files are megabytes that must not enter git. The obvious
shortcut - a hand-written 3x3 matrix of made-up numbers - would test the
array plumbing and nothing else: not the two-row CSV header, not the
ragged ipsi/contra column count, not the acronym-to-id join, not the fact
that a real region's self-projection is non-zero in every row of the
published matrix.

So each fixture is instead a *slice of the checksum-verified published
file, written back out in that file's own published format*: fifteen
regions instead of 291, a few kilobytes instead of a few megabytes, same
parser, same shape of header, same real numbers. `PROVENANCE.md` records
which verified bytes each was cut from, and
`scripts/make_connectome_fixture.py` regenerates them.

The fifteen regions are the ones this codebase's existing scenarios
already name (VTA, ACB, PL, CP, GPe, STN, GPi, SNr, PAG, LC, BLA, CEA)
plus cortical, thalamic and raphe partners (MOp, MD, DR), and they span
eight of the twelve major divisions. One of them, LC, exists in Knox et
al. 2019 and not in Oh et al. 2014, so the Oh fixture has fourteen rows
where the Knox fixture has fifteen. That asymmetry is kept on purpose: it
is a real property of the two datasets and the loaders have to survive it.
STN is in the set for a second reason: it is the region that gives the Oh
fixture some genuinely zero-weight connections, and therefore the `#NUM!`
p-value cells that exercise the reader's error-token path on real bytes.

These fixtures are for tests and for offline exploration. They are NOT a
substitute for the real matrices in any modelling result: fifteen
regions sliced out of a whole-brain model are not a fifteen-region model
of the brain, because every path through the other 276 regions has been
deleted.
"""

import os

FIXTURE_DIR = os.path.abspath(os.path.dirname(__file__))

#: CCF structure ids in the fixtures, in ontological order. Acronyms are
#: resolved from the ontology rather than written down a second time.
FIXTURE_REGION_IDS = (
    985,   # MOp   isocortex, primary motor area
    972,   # PL    isocortex, prelimbic area
    56,    # ACB   striatum, nucleus accumbens
    672,   # CP    striatum, caudoputamen
    295,   # BLA   cortical subplate, basolateral amygdalar nucleus
    536,   # CEA   striatum, central amygdalar nucleus
    1022,  # GPe   pallidum, globus pallidus, external segment
    1031,  # GPi   pallidum, globus pallidus, internal segment
    470,   # STN   hypothalamus (the CCF's placement), subthalamic nucleus
    362,   # MD    thalamus, mediodorsal nucleus
    749,   # VTA   midbrain, ventral tegmental area
    381,   # SNr   midbrain, substantia nigra, reticular part
    795,   # PAG   midbrain, periaqueductal gray
    872,   # DR    midbrain, dorsal nucleus raphe
    147,   # LC    pons, locus ceruleus
)

#: P4.2's five: the regions where Yoo et al. 2010 measured mouse D1 and DAT
#: binding densities and this codebase has ISH expression for the matching
#: gene, minus the ones already above.
AUTORADIOGRAPHY_ONLY_REGION_IDS = (
    754,   # OT    olfactory areas, olfactory tubercle
    583,   # CLA   cortical subplate, claustrum
    952,   # EPd   cortical subplate, endopiriform nucleus, dorsal part
    31,    # ACA   isocortex, anterior cingulate area
    1080,  # HIP   hippocampal formation, hippocampal region
)

#: P4.3's seven: the regions its cited expected distributions name and that
#: neither list above already had. Hippocampal subfields because Hoertnagl
#: et al. 2013 score Gabra1 mRNA +++ in CA1 and + in CA3; MH and PVT because
#: they are where Erbs et al. 2015 put the densest mu-opioid signal in the
#: brain; SNc because the D1/D2 midbrain contrast needs it; CBX because
#: "cerebellum" appears in three of the six receptors' expectations and the
#: connectome parcellates the cerebellum into lobules with no whole-cortex
#: row.
ANATOMY_ONLY_REGION_IDS = (
    382,   # CA1   hippocampal formation, field CA1
    463,   # CA3   hippocampal formation, field CA3
    726,   # DG    hippocampal formation, dentate gyrus
    483,   # MH    thalamus, medial habenula
    149,   # PVT   thalamus, paraventricular nucleus
    374,   # SNc   midbrain, substantia nigra, compact part
    528,   # CBX   cerebellum, cerebellar cortex
)

#: Structures that get NO row or column in the connectome fixture matrices
#: but that the shared fixture ontology must still resolve, because a
#: downstream package's fixture is cut against this ontology and needs them.
#: They are ontology entries only - adding them to `FIXTURE_REGION_IDS`
#: would change the committed Knox and Oh matrices and every number the P3.x
#: tests assert on them, for no gain, since neither P4.2 nor P4.3 touches a
#: connection weight.
EXTRA_ONTOLOGY_REGION_IDS = (
    AUTORADIOGRAPHY_ONLY_REGION_IDS + ANATOMY_ONLY_REGION_IDS
)

#: The region absent from Oh et al. 2014's 213-region set. See the module
#: docstring; `docs/architecture.md` records why it matters here.
FIXTURE_REGION_MISSING_FROM_OH2014 = 147

FIXTURE_CONNECTOME_KEY = "knox2019_normalized_connection_density"
FIXTURE_OH_KEY = "oh2014_supplementary_table_3"

#: The three Knox normalizations that have a fixture, and their filenames.
#: P3.1 needed only the normalized connection density. P3.2 added the other
#: two because building a network at a coarse resolution needs the RAW
#: summed weight (the only additive normalization) and needs the region
#: voxel counts, which are recovered from the strength/density ratio rather
#: than written down anywhere (`connectome/volumes.py`). All three are cut
#: from the same verified files in the same way, so this costs nothing but
#: two more small CSVs.
KNOX_FIXTURE_KEYS = (
    "knox2019_normalized_connection_density",
    "knox2019_connection_strength",
    "knox2019_connection_density",
)

ONTOLOGY_FIXTURE_PATH = os.path.join(FIXTURE_DIR, "allen_structure_graph_1_fixture.json")
KNOX_FIXTURE_PATHS = {
    key: os.path.join(FIXTURE_DIR, "%s_fixture.csv" % key) for key in KNOX_FIXTURE_KEYS
}
KNOX_FIXTURE_PATH = KNOX_FIXTURE_PATHS[FIXTURE_CONNECTOME_KEY]
OH_FIXTURE_PATH = os.path.join(
    FIXTURE_DIR, "oh2014_supplementary_table_3_fixture.xlsx"
)


def load_fixture_ontology():
    """The trimmed CCF structure graph: the fifteen regions and their ancestors."""
    from ..ontology import StructureOntology

    return StructureOntology.from_path(ONTOLOGY_FIXTURE_PATH)


def load_fixture_connectome(key=FIXTURE_CONNECTOME_KEY, ontology=None):
    """Load one fixture matrix. No cache, no network, no download path."""
    from ..matrix import read_connectome

    paths = dict(KNOX_FIXTURE_PATHS)
    paths[FIXTURE_OH_KEY] = OH_FIXTURE_PATH
    if key not in paths:
        raise KeyError(
            "no fixture for %r; fixtures exist for %s" % (key, sorted(paths))
        )
    if ontology is None:
        ontology = load_fixture_ontology()
    return read_connectome(key, paths[key], ontology=ontology)


def load_fixture_build_inputs(ontology=None):
    """`(strength, voxel_counts)` off the fixtures, for P3.2's builder.

    The offline mirror of `build.load_knox_inputs()`: same two matrices,
    same voxel-count recovery, fifteen regions instead of 291.
    """
    from ..volumes import target_voxel_counts

    if ontology is None:
        ontology = load_fixture_ontology()
    strength = load_fixture_connectome("knox2019_connection_strength", ontology)
    density = load_fixture_connectome("knox2019_connection_density", ontology)
    return strength, target_voxel_counts(strength, density)
