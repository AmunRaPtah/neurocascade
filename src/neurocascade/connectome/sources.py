"""P3.1: the versioned manifest of external connectome data this package
ingests, and the only place a URL or a checksum is written down.

Why a manifest module instead of "just download it in the loader"
-----------------------------------------------------------------

Every other number in this codebase is pinned to a citation. A downloaded
matrix is no different: "the Allen connectome" is not one thing, it is a
specific model fitted to a specific data release, published on a specific
date, and the numbers move between releases. So each file below records

  * the exact URL it came from,
  * the SHA-256 of the exact bytes retrieved on `retrieved`,
  * the publication that produced it, and
  * what the numbers in it actually mean (`units`), taken from that
    publication's own methods, or, for the one measure the methods do not
    write out, from the authors' companion implementation with the
    attribution said out loud.

The checksum is the part that makes the build reproducible. `fetch.py`
refuses a file whose digest does not match, which converts "the upstream
data silently changed under us" from an invisible drift in every downstream
result into a loud, dated failure that a human has to look at. That is the
intended behaviour, not an inconvenience: see `VOLATILITY` below.

Two datasets, not one, and why
------------------------------

**Knox et al. 2019** is the primary. It is the regionalized voxel model:
291 grey-matter regions at an intermediate level of the CCF, published as
four plain CSVs (one per normalization) on a static Allen download server.
It is preferred because (a) it is the more recent fit, (b) it covers 291
regions rather than 213, and (c) its four normalizations are published
separately, so this codebase does not have to guess which normalization a
single unlabelled matrix used.

**Oh et al. 2014** is kept as a second matrix. It is what Knox et al. call
the "homogeneous model": a region-by-region non-negative least squares fit
over 213 regions, published as Supplementary Table 3 of the Nature paper.
It is worth ingesting alongside Knox for two reasons: it ships
per-connection p-values, which Knox's regionalized CSVs do not, and it is
a different model class fitted independently, so P3.2/P3.3 can ask whether
a whole-brain result depends on which connectome it was built from.

That independence is in the *model*, not in the *data*. Both fits are
built on anterograde rAAV tracer experiments in wild-type C57BL/6J mice
from the same Allen Mouse Brain Connectivity Atlas: Oh et al. fit 469
experiments, Knox et al. fit 428 of the 491 then available and cite Oh et
al. as the data source. The two experiment sets therefore overlap heavily.
Agreement between the matrices is evidence that a result does not hinge on
the modelling assumptions (regional homogeneity versus voxel-scale
smoothing); it is not independent replication of the underlying anatomy,
and this module should not be used to claim otherwise.

The 213 Oh regions are NOT a subset of the 291 Knox regions: 202 acronyms
are common, 11 Oh regions are absent from Knox's set, and 89 Knox regions
are absent from Oh's. One of those 89 matters immediately: **LC (locus
coeruleus, id 147) is present in Knox 2019 and absent from Oh 2014**, and
LC is a coupled node in this codebase's existing multi-region mu-opioid
scenario. That alone settles the "which one is primary" question on
grounds specific to this project rather than on general recency.

A naming trap that this module refuses to paper over
----------------------------------------------------

The task that produced this module asked for "the normalized connection
density matrix". Knox et al. 2019 publishes a file with exactly that name.
Oh et al. 2014's Supplementary Table 3 does **not** contain a connection
density: its own Supplementary Notes call the quantity "normalized
connection strength ... the amount of segmented signal activated in the
target region by infecting one voxel in the source region". Those are
different normalizations (density divides by the target region's size as
well; strength does not), and they differ by orders of magnitude across
region pairs of unequal size. Each entry's `units` field below therefore
carries the source's own words, and `matrix.py` propagates that string
onto the loaded object rather than letting a caller assume all matrices in
this module are commensurable. They are not.

Orientation
-----------

Both publications *print* their matrices with sources on rows and targets
on columns, and both note this explicitly. Knox et al. 2019 is the more
pointed about it: their mathematics uses W[target, source] and their
figures show W-transpose. This codebase's `CircuitNetwork.connectivity`
uses the mathematical convention, `connectivity[i, j]` = projection from
`j` onto `i`. The published-file orientation and the CircuitNetwork
orientation are therefore transposes of each other, which is exactly the
kind of thing that produces a plausible-looking but reversed whole-brain
model. `matrix.py` keeps the file's native (source-major) orientation in
`weights`, names it so, and provides one explicit transposing accessor;
it does not silently pick one.

The large voxel-scale files, and why they are not manifested
------------------------------------------------------------

The same Knox download directory also holds `nodes.csv.gz` (490 MB) and
`weights.csv.gz` (145 MB), the low-rank factors of the full voxel-scale
model. They are deliberately absent from `DATA_FILES`: this task's target
is a regional matrix, the regionalized CSVs are the published regional
form of that same model, and a checksum may not be written down for bytes
that have not actually been retrieved and hashed. If a later task needs
voxel resolution, it downloads them, hashes them, and adds them here.
"""

from ..datafetch import DataFile as _DataFile

MANIFEST_VERSION = "2026-08-12"

#: What a human should run when the cache is missing a file. Read by
#: `datafetch.Fetcher` so the error message names the right script.
FETCH_COMMAND = "scripts/fetch_connectome.py"

# Retrieval date shared by every entry below: all files were downloaded and
# hashed in one session while writing P3.1.
_RETRIEVED = "2026-08-12"

_KNOX_BASE = (
    "https://download.alleninstitute.org/publications/"
    "A_high_resolution_data-driven_model_of_the_mouse_connectome"
)

KNOX2019_CITATION = (
    "Knox JE, Harris KD, Graddis N, Whitesell JD, Zeng H, Harris JA, "
    "Shea-Brown E, Mihalas S. 'High-resolution data-driven model of the "
    "mouse connectome.' Network Neuroscience. 2019;3(1):217-236. "
    "doi:10.1162/netn_a_00066 (PMID 30793081, PMC6372022). "
    "Species: adult mouse (C57BL/6J), anterograde rAAV tract tracing, "
    "Allen Mouse Brain Connectivity Atlas."
)

OH2014_CITATION = (
    "Oh SW, Harris JA, Ng L, et al. 'A mesoscale connectome of the mouse "
    "brain.' Nature. 2014;508(7495):207-214. doi:10.1038/nature13186 "
    "(PMID 24695228, PMC5102064). Species: adult mouse (C57BL/6J), "
    "anterograde rAAV tract tracing, Allen Mouse Brain Connectivity Atlas."
)

ALLEN_ONTOLOGY_CITATION = (
    "Allen Institute for Brain Science. Allen Mouse Brain Common "
    "Coordinate Framework structure graph 1 (adult mouse), retrieved from "
    "the Allen Brain Atlas API. See Wang Q, Ding S-L, Li Y, et al. 'The "
    "Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas.' "
    "Cell. 2020;181(4):936-953.e20. doi:10.1016/j.cell.2020.04.007"
)


class DataFile(_DataFile):
    """One external file: where it comes from, what it hashes to, what it means.

    The behaviour lives in `neurocascade.datafetch.DataFile`, which P4.1's
    expression manifest also uses; this subclass exists only to default
    `retrieved` to this manifest's single retrieval date, since every entry
    below was downloaded and hashed in one session.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        _DataFile.__init__(self, *args, **kwargs)


# ---------------------------------------------------------------------------
# Knox et al. 2019 regionalized voxel model: four normalizations of the same
# 291-region directed matrix, all four computed from the VOXEL-scale model by
# summing over each region's 100 um voxels.
#
# The four definitions, with w_ij the summed voxel-pair weight from source
# region X (|X| voxels) to target region Y (|Y| voxels):
#
#   connection_strength             w_ij
#   connection_density              w_ij / |Y|          target size only
#   normalized_connection_strength  w_ij / |X|          source size only
#   normalized_connection_density   w_ij / (|X| |Y|)    both
#
# Sources for those four, not one: the paper's Methods ("Regionalized
# connectivity") define the first, third and fourth symbolically, and the
# Allen Institute's companion implementation defines all four in code, as
# the docstrings of `RegionalizedModel.connection_strength` /
# `.connection_density` / `.normalized_connection_strength` /
# `.normalized_connection_density` in
# https://github.com/AllenInstitute/mouse_connectivity_models
# (mcmodels/models/voxel/regionalized_model.py). That class's own docstring
# example states `normalized_connection_density.shape == (291, 577)`, which
# is exactly the shape of the published files, and it builds the matrix from
# `voxel_array`, which is what pins these CSVs to the regionalized VOXEL
# model rather than to the paper's recomputed homogeneous model.
#
# Note the paper writes its formulae in the target-major convention
# (W[target, source]) while the code and the published CSVs are source-major,
# so the symbolic forms in the paper and the divisions above are the same
# statement written two ways round. See the "Orientation" section above.
# ---------------------------------------------------------------------------

DATA_FILES = [
    DataFile(
        key="knox2019_normalized_connection_density",
        filename="knox2019_normalized_connection_density.csv.gz",
        url=_KNOX_BASE + "/normalized_connection_density.csv.gz",
        sha256="e202b7262882e91dcb9870a30b372ddecbb3cbcac6b532b60fa1b538ec37aa97",
        size_bytes=1622011,
        citation=KNOX2019_CITATION,
        description=(
            "Regionalized voxel model, 291 grey-matter CCF regions, "
            "directed, source rows x (291 ipsilateral + 286 contralateral) "
            "target columns. This is the matrix shown in the paper's "
            "Figure 2. Default matrix for this package."
        ),
        units=(
            "normalized connection density, w_ij / (|X| |Y|): the region-pair "
            "connection strength divided by the voxel counts of BOTH the "
            "source and the target region, i.e. the mean voxel-to-voxel model "
            "weight between the two regions. Non-negative, dimensionless per "
            "voxel pair. Knox 2019 Methods; RegionalizedModel."
            "normalized_connection_density in mouse_connectivity_models."
        ),
    ),
    DataFile(
        key="knox2019_normalized_connection_strength",
        filename="knox2019_normalized_connection_strength.csv.gz",
        url=_KNOX_BASE + "/normalized_connection_strength.csv.gz",
        sha256="b170761104ebf1b3d96d3800e83522a3aaafac3ec882855a20b49e488777523e",
        size_bytes=1595759,
        citation=KNOX2019_CITATION,
        description=(
            "Regionalized voxel model, same 291-region shape as the density "
            "file. Normalized by SOURCE region size only. Knox 2019 note "
            "this is the normalization that is directly comparable with the "
            "Oh 2014 homogeneous model."
        ),
        units=(
            "normalized connection strength, w_ij / |X|: connection strength "
            "divided by the SOURCE region's voxel count only. Knox 2019 "
            "Methods; RegionalizedModel.normalized_connection_strength in "
            "mouse_connectivity_models."
        ),
    ),
    DataFile(
        key="knox2019_connection_strength",
        filename="knox2019_connection_strength.csv.gz",
        url=_KNOX_BASE + "/connection_strength.csv.gz",
        sha256="c2df7d5ac4478bb01c78a06c577c1b58453a0674b4258c070f1e555b9ca5eca8",
        size_bytes=1552201,
        citation=KNOX2019_CITATION,
        description=(
            "Regionalized voxel model, same 291-region shape. Unnormalized: "
            "the summed voxel-voxel weight between the two regions, so it "
            "scales with both regions' volumes."
        ),
        units=(
            "connection strength, w_ij: the sum of voxel-to-voxel model "
            "weights over all voxel pairs in the two regions, with neither "
            "region's size divided out. Knox 2019 Methods; RegionalizedModel."
            "connection_strength in mouse_connectivity_models."
        ),
    ),
    DataFile(
        key="knox2019_connection_density",
        filename="knox2019_connection_density.csv.gz",
        url=_KNOX_BASE + "/connection_density.csv.gz",
        sha256="ecd4cc3cfb1ea1a33967de8f1d9183338c512edaaea5fbdb58211a4aa34d9862",
        size_bytes=1593081,
        citation=KNOX2019_CITATION,
        description=(
            "Regionalized voxel model, same 291-region shape. The one "
            "normalization the paper's Methods do not write out; its "
            "definition comes from the companion implementation instead."
        ),
        units=(
            "connection density, w_ij / |Y|: connection strength divided by "
            "the TARGET region's voxel count only. Not defined symbolically "
            "in Knox 2019's Methods; definition taken from RegionalizedModel."
            "connection_density in the Allen Institute's companion package "
            "mouse_connectivity_models."
        ),
    ),
    DataFile(
        key="oh2014_supplementary_table_3",
        filename="oh2014_supplementary_table_3.xlsx",
        url=(
            "https://media.springernature.com/original/springer-static/esm/"
            "art%3A10.1038%2Fnature13186/MediaObjects/"
            "41586_2014_BFnature13186_MOESM71_ESM.xlsx"
        ),
        sha256="eab971421b9597424d65547e3361e5b9d64c32bc15d21d51b94e5067482c5664",
        size_bytes=1570515,
        citation=OH2014_CITATION,
        description=(
            "Supplementary Table 3: the linear (homogeneous) model weights "
            "behind the paper's Figure 4a, 213 regions, rows = sources, "
            "columns = targets, labelled by CCF acronym. Four sheets: "
            "W_ipsi, PValue_ipsi, W_contra, PValue_contra. The p-value "
            "sheets are the per-connection statistical confidence; Knox "
            "2019's regionalized CSVs have no equivalent."
        ),
        units=(
            "normalized connection strength, w_X,Y (Oh 2014 Supplementary "
            "Notes): 'the amount of segmented signal activated in the target "
            "region by infecting one voxel in the source region ... "
            "proportional to the average out-degree of neurons projecting "
            "from the source to the target'. NOT a connection density: it is "
            "not divided by target region size."
        ),
    ),
    DataFile(
        key="allen_structure_graph_1",
        filename="allen_structure_graph_1.json",
        url="https://api.brain-map.org/api/v2/structure_graph_download/1.json",
        sha256="b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386",
        size_bytes=637735,
        citation=ALLEN_ONTOLOGY_CITATION,
        description=(
            "Adult mouse structure graph (ontology id 1): 1327 structures "
            "as one nested tree under 'root' (id 997), each with id, "
            "acronym, name, parent_structure_id, st_level, graph_order and "
            "colour. This is what turns a bare structure id in either "
            "connectivity matrix into a named region with a place in the "
            "hierarchy, which is what P3.2 needs to select a resolution."
        ),
        units="ontology; no numeric measurements",
        volatility="live-api",
    ),
]

DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

#: The file every loader defaults to when the caller does not name one.
DEFAULT_CONNECTIVITY_KEY = "knox2019_normalized_connection_density"
ONTOLOGY_KEY = "allen_structure_graph_1"

VOLATILITY = """
Five of the six files sit on static archive servers (Allen's publication
download server, Springer Nature's static ESM host) and are expected to be
byte-identical forever.

`allen_structure_graph_1` is different: it is a *live API endpoint* that
renders the ontology out of Allen's database on each request. It was
verified byte-stable across two requests two seconds apart while writing
this manifest, but nothing prevents Allen from revising the CCF ontology
and changing those bytes. That is precisely why it is checksummed: if the
ontology changes, the fetch fails loudly, someone reads the diff, decides
whether the region set this project uses moved, and bumps the manifest in a
commit that says so. A fetch script that silently accepted whatever the API
returned today would let a region rename or a re-parenting propagate into
every whole-brain result with no record.
"""


def data_file(key):
    """Look up one manifest entry, with a useful error listing the keys."""
    try:
        return DATA_FILES_BY_KEY[key]
    except KeyError:
        raise KeyError(
            "unknown data file %r; manifest %s has: %s"
            % (key, MANIFEST_VERSION, ", ".join(sorted(DATA_FILES_BY_KEY)))
        )


def connectivity_keys():
    """The manifest keys that hold a connectivity matrix (not the ontology)."""
    return [f.key for f in DATA_FILES if f.key != ONTOLOGY_KEY]
