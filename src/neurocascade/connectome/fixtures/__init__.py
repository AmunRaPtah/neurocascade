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

#: The region absent from Oh et al. 2014's 213-region set. See the module
#: docstring; `docs/architecture.md` records why it matters here.
FIXTURE_REGION_MISSING_FROM_OH2014 = 147

FIXTURE_CONNECTOME_KEY = "knox2019_normalized_connection_density"
FIXTURE_OH_KEY = "oh2014_supplementary_table_3"

ONTOLOGY_FIXTURE_PATH = os.path.join(FIXTURE_DIR, "allen_structure_graph_1_fixture.json")
KNOX_FIXTURE_PATH = os.path.join(
    FIXTURE_DIR, "knox2019_normalized_connection_density_fixture.csv"
)
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

    paths = {
        FIXTURE_CONNECTOME_KEY: KNOX_FIXTURE_PATH,
        FIXTURE_OH_KEY: OH_FIXTURE_PATH,
    }
    if key not in paths:
        raise KeyError(
            "no fixture for %r; fixtures exist for %s" % (key, sorted(paths))
        )
    if ontology is None:
        ontology = load_fixture_ontology()
    return read_connectome(key, paths[key], ontology=ontology)
