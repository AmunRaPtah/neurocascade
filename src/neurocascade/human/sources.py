"""The versioned manifest of external human connectome data this package
ingests, and the only place a URL or a checksum is written down.

Why a second manifest instead of one more entry in `connectome/sources.py`
--------------------------------------------------------------------------

`connectome/sources.py` is the mouse manifest: Allen CCF ontology, Knox
et al. 2019 and Oh et al. 2014 matrices, all of it one species and one
coordinate system. The human data below is a different species, a
different atlas, a different file format and - decisively - a different
*kind* of measurement (diffusion tractography instead of tracer
injection). Keeping the two manifests in separate modules is what makes
"which species did this number come from" answerable by looking at the
import, instead of by remembering which key was which.

The data
--------

The human structural connectivity matrix is the ENIGMA Toolbox's
group-average Desikan-Killiany ("aparc") matrix with cortical and
subcortical regions in one 82 x 82 file (`strucMatrix_with_sctx.csv`):
68 cortical regions (34 per hemisphere, Desikan-Killiany 2006) plus 14
subcortical regions (nucleus accumbens, amygdala, caudate, hippocampus,
pallidum, putamen, thalamus; 7 per hemisphere). It was reconstructed with
MRtrix3 tractography from the diffusion MRI of unrelated healthy young
adults in the Human Connectome Project, thresholded with a
distance-dependent procedure, log-transformed, and averaged across
subjects. The numbers are streamline counts (fiber density).

The URLs are pinned to one commit of the ENIGMA Toolbox repository so the
bytes do not drift when `master` moves; the SHA-256 below is of the exact
bytes retrieved on `retrieved`. The repository is BSD-3-Clause; the
underlying HCP imaging data and the Desikan-Killiany atlas are cited
separately below.

Three honest differences from the mouse matrix, said out loud
-------------------------------------------------------------

1. **Undirected.** Diffusion tractography cannot tell the direction of a
   white-matter bundle, so the matrix is symmetric by construction
   (`W[i, j] == W[j, i]`). The mouse matrices are directed. `matrix.py`
   refuses a non-symmetric human file rather than pretending the
   direction is data.

2. **Unsigned.** dMRI carries no excitatory/inhibitory sign. The mouse
   package assigns signs from neurotransmitter identity
   (`connectome/neurotransmitter.py`); the human package does NOT invent
   one at ingestion. `build.py` applies a single, documented
   `coupling_sign` assumption and names the regions where the default is
   known to be wrong.

3. **Cortex-centred atlas.** Desikan-Killiany is a cortical parcellation;
   the 14 subcortical regions are the coarse classic nuclei, not the fine
   subdivisions the mouse CCF provides. That coarseness is a property of
   the human data, not a bug to be hidden.
"""

from ..datafetch import DataFile as _DataFile

MANIFEST_VERSION = "2026-08-16"

#: What a human should run when the cache is missing a file.
FETCH_COMMAND = "scripts/fetch_human_connectome.py"

# Retrieval date shared by every entry below: all files were downloaded and
# hashed in one session while writing the human layer.
_RETRIEVED = "2026-08-16"

# Pinned to one commit rather than `master`, so the bytes do not move.
_BASE = (
    "https://raw.githubusercontent.com/MICA-MNI/ENIGMA/"
    "b08974b55243060cbc1fad12c87048037446e8f7/"
    "enigmatoolbox/datasets/matrices/hcp_connectivity"
)

ENIGMA_CITATION = (
    "ENIGMA Toolbox group-average connectivity: Lariviere S, Paquola C, "
    "Park B-y, Royer J, Wang Y, Bernhardt BC, Misic B, et al. 'The ENIGMA "
    "Toolbox: multiscale neural contextualization of multisite "
    "neuroimaging datasets.' Nature Methods. 2021;18(7):698-700. "
    "doi:10.1038/s41592-021-01186-4. Species: human. Group-average "
    "structural connectivity reconstructed with MRtrix3 diffusion "
    "tractography in unrelated healthy young adults of the Human "
    "Connectome Project, parcellated with Desikan-Killiany (68 cortical "
    "regions) plus 14 subcortical regions. Retrieved from the ENIGMA "
    "Toolbox repository (github.com/MICA-MNI/ENIGMA, BSD-3-Clause)."
)

HCP_CITATION = (
    "Underlying imaging data: Van Essen DC, Smith SM, Barch DM, Behrens "
    "TEJ, Yacoub E, Ugurbil K; WU-Minn HCP Consortium. 'The WU-Minn Human "
    "Connectome Project: An overview.' NeuroImage. 2013;80:62-79. "
    "doi:10.1016/j.neuroimage.2013.05.041."
)

DK_CITATION = (
    "Parcellation: Desikan RS, Segonne F, Fischl B, Quinn BT, Dickerson "
    "BC, Blacker D, Buckner RL, Dale AM, Maguire RP, Hyman BT, Albert MS, "
    "Killiany RJ. 'An automated labeling system for subdividing the human "
    "cerebral cortex on MRI scans into gyral based regions of interest.' "
    "NeuroImage. 2006;31(3):968-980. doi:10.1016/j.neuroimage.2006.01.021."
)


class DataFile(_DataFile):
    """One external file: where it comes from, what it hashes to, what it means.

    The behaviour lives in `neurocascade.datafetch.DataFile`; this subclass
    only defaults `retrieved` to this manifest's single retrieval date.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        _DataFile.__init__(self, *args, **kwargs)


DATA_FILES = [
    DataFile(
        key="enigma_hcp_sc_matrix_aparc_with_sctx",
        filename="enigma_hcp_sc_matrix_aparc_with_sctx.csv",
        url=_BASE + "/strucMatrix_with_sctx.csv",
        sha256="ee3825eed9d83786fbfa4804c13c78ef4b94267ad48b517e2a732f2c05300634",
        size_bytes=25098,
        citation=ENIGMA_CITATION + " " + HCP_CITATION + " " + DK_CITATION,
        description=(
            "Group-average human structural connectivity, Desikan-Killiany "
            "('aparc') parcellation, cortical + subcortical in one 82 x 82 "
            "matrix: 68 cortical regions (34 per hemisphere) + 14 "
            "subcortical regions (7 per hemisphere). No header and no index "
            "column: plain comma-separated numbers in the order of the "
            "labels file. Symmetric (undirected) with a zero diagonal. This "
            "is the file behind the ENIGMA Toolbox's load_sc_as_one() "
            "'aparc' output."
        ),
        units=(
            "log-transformed streamline count (fiber density) between "
            "region pairs: the number of reconstructed streamlines "
            "connecting two regions, log-transformed to reduce "
            "connectivity-strength variance, group-averaged across "
            "unrelated healthy adults from the HCP. UNDIRECTED (symmetric "
            "by construction of diffusion tractography) and UNSIGNED (dMRI "
            "carries no excitatory/inhibitory sign). Non-negative; the "
            "diagonal is zero."
        ),
    ),
    DataFile(
        key="enigma_hcp_sc_labels_aparc_with_sctx",
        filename="enigma_hcp_sc_labels_aparc_with_sctx.csv",
        url=_BASE + "/strucLabels_with_sctx.csv",
        sha256="331333840228ccb9489d7b63f6d12c08bef9ee12ec0fbe95e27b7d86759ce97a",
        size_bytes=1249,
        citation=DK_CITATION + " " + ENIGMA_CITATION,
        description=(
            "The 82 region labels for the 'aparc' structural connectivity "
            "matrix, as one comma-separated row in matrix row/column order. "
            "Cortical labels use a hemisphere underscore prefix "
            "('L_bankssts', 'R_precentral'); subcortical labels use the "
            "hemisphere letter directly ('Laccumb', 'Rthal')."
        ),
        units="region labels; no numeric measurements",
    ),
]

DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

#: The matrix every loader defaults to when the caller does not name one.
DEFAULT_MATRIX_KEY = "enigma_hcp_sc_matrix_aparc_with_sctx"
LABELS_KEY = "enigma_hcp_sc_labels_aparc_with_sctx"

VOLATILITY = (
    "Both files are pinned to one commit of the ENIGMA Toolbox repository "
    "(raw.githubusercontent.com serves immutable per-commit bytes), so they "
    "are byte-stable for as long as that commit exists on GitHub. They are "
    "still checksummed: if the bytes ever differ, the fetch fails loudly and "
    "the manifest is reviewed, exactly as for the mouse data."
)


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
    """The manifest keys that hold a connectivity matrix (not the labels)."""
    return [f.key for f in DATA_FILES if f.key != LABELS_KEY]
