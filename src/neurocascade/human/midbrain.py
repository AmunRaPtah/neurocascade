"""Midbrain nodes defined from dedicated human atlases -- geometry only.

Why this module exists
----------------------

The mu-opioid reward scenario needs a ventral tegmental area node. The
atlas stack this package otherwise builds on cannot supply one: the
Cammoun ("Lausanne") scale033 parcellation that carries the Hansen 2022
PET receptor maps has NO VTA (`receptors.PARCELS` -- its last parcel is
the undifferentiated ``brainstem``, which `receptors.UNMAPPED_PARCELS`
already keeps out of the connectome), and the ENIGMA
Desikan-Killiany connectome has no midbrain regions at all. In vivo, the
human VTA falls inside that single brainstem parcel; there is no
measurement to average onto a VTA node, and the roadmap forbids papering
over that by borrowing the parcel's value.

So P12.3 does the two halves of the job separately and honestly:

* **Geometry** comes from dedicated midbrain atlases -- CIT168 (Pauli et
  al. 2018) to define the nuclei, Trutti et al. (2021) as an independent
  replication of the structure they jointly describe. Both are human,
  both are cited below, and the published volumes are transcribed into
  `NUCLEI`.
* **Receptor density stays ABSENT, and the absence is enforced rather
  than noted**: `MidbrainNode` carries no density field (its ``__slots__``
  make one impossible to attach), every read path into the PET layer
  refuses a midbrain label with `MidbrainDensityAbsent`, and
  `pet_value_from_brainstem_parcel` exists so that the one number the PET
  atlas DOES have for tissue containing the VTA -- the Cammoun-033
  brainstem parcel's own value -- can only be obtained through a call that
  says out loud it is not a VTA measurement. `SEARCHED_AND_EMPTY` records
  what was searched for a measured human VTA receptor density and why each
  candidate found was unusable; nothing here invents a number to fill the
  gap (rule 1), and none of it borrows a different receptor's or another
  species' measurement (rule 2).

The source atlases
------------------

CIT168 -- Pauli WM, Nili AN, Tyszka JM. "A high-resolution probabilistic
in vivo atlas of human subcortical brain nuclei." *Scientific Data*
2018;5:180063. PMID 29664465. doi:10.1038/sdata.2018.63. Probabilistic
segmentations of 16 subcortical nuclei from 168 adults (ages 22-35),
delineated on 700 um templates and distributed in MNI152NLin2009cAsym
1 mm space; deposited on OSF (project r2hvk) as the paper's Data Citation.
Its Table 1 publishes each nucleus' volume in microliters, "derived ...
by spatial integration of label probabilities over each hemisphere
independently" -- those are the numbers transcribed into `NUCLEI`, because
they are the peer-reviewed ones (see `DEPOSIT_VS_TABLE1` for why the
deposited image files are NOT the source of any product value here).

Trutti AC, Fontanesi L, Mulder MJ, Bazin PL, Hommel B, Forstmann BU. "A
probabilistic atlas of the human ventral tegmental area (VTA) based on
7 Tesla MRI data." *Brain Structure and Function* 2021;226(4):1155-1167.
PMID 33580320. doi:10.1007/s00429-021-02231-w. Manual delineation by two
independent raters in 27 healthy participants (19 female, mean age 24.5,
SD 4.8), keeping only voxels both raters called VTA, registered to
MNI152 2009b space; deposited on OSF (project 9pzj3).

Neither atlas carries ANY receptor measurement. That is a property of the
sources, not an oversight of this module: they are structural MRI
segmentations. What a receptor-atlas consumer would want from them --
density per nucleus -- does not exist anywhere we could find, which
`SEARCHED_AND_EMPTY` documents candidate by candidate.

What this module deliberately does NOT do
-----------------------------------------

* **No centroids, no wiring.** The deposited CIT168 images integrate
  systematically larger than the paper's own Table 1 (`DEPOSIT_VS_TABLE1`),
  by 18-66% depending on nucleus, for reasons the deposit does not
  document; exposing coordinates derived from those bytes would inherit an
  unexplained discrepancy. And no measured directed connectivity connects
  these nuclei to the ENIGMA connectome's 82 regions, so no network edges
  are built here -- Phase 13's problem, and inventing weights is forbidden
  (rule 1).
* **No averaging of the brainstem parcel onto anything.**
  `pet_value_from_brainstem_parcel` exists precisely to keep that refusal
  explicit at every call site that might be tempted.

Licences
--------

Both articles are CC BY 4.0 (each states it in its own copyright block).
The deposits' OSF projects carry NO licence field of their own (verified
through the OSF API, 2026-08-25); the files are used here under the terms
of their articles' CC BY 4.0 statements, with attribution recorded on
every manifest entry. This is stated rather than smoothed over: it is the
one licence question about this data that the sources do not answer
directly.

Offline by default
------------------

Same contract as the rest of ``human/``: nothing touches the network at
import time, loaders default to ``allow_download=False``, a cache miss
raises `MissingCachedFile` naming ``scripts/fetch_human_midbrain.py``.
Tests run off the committed fixtures -- a byte-identical copy of the
CIT168 label table (``info.csv``), which pins the registry offline; the
image files stay in the gitignored cache and the checks that parse them
skip when it is absent. See ``fixtures/PROVENANCE.md``.

    from neurocascade import human

    vta = human.midbrain.node("LVTA")
    vta.volume_ul          # 33.0, Pauli Table 1, left hemisphere
    vta.receptor_density("MOR")   # refuses: MidbrainDensityAbsent

    human.receptors.load_receptor("MOR").for_connectome_label("LVTA")
    #   -> MidbrainDensityAbsent, from the receptors side too
"""

import csv
import gzip
import hashlib
import io
import os
import re
import struct
import sys
import tarfile
import tempfile

import numpy as np

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401

__all__ = [
    # citations and licences
    "PAULI_CITATION",
    "PAULI_ARTICLE_LICENCE",
    "PAULI_DEPOSIT_NOTE",
    "TRUTTI_CITATION",
    "TRUTTI_ARTICLE_LICENCE",
    "TRUTTI_DEPOSIT_LICENCE_NOTE",
    "ATLAS_SPACES_NOTE",
    # manifest and fetch
    "MANIFEST_VERSION",
    "FETCH_COMMAND",
    "DEFAULT_CACHE_DIR",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "data_file",
    "cache_path",
    "verify",
    "is_cached",
    "download",
    "ensure",
    "fetch_all",
    "checksum_lines",
    "ChecksumMismatch",
    "MissingCachedFile",
    "CIT168_TAR_MEMBERS",
    # geometry: the nuclei and their published volumes
    "GEOMETRY_UNITS",
    "MidbrainNucleus",
    "NUCLEI",
    "SCENARIO_NUCLEI",
    "nucleus",
    # nodes: what the scenarios address
    "MidbrainNode",
    "NODES",
    "MIDBRAIN_NODE_LABELS",
    "node",
    "scenario_node_labels",
    # enforcement
    "MidbrainDensityAbsent",
    "node_for_request",
    "is_midbrain_request",
    "pet_value_from_brainstem_parcel",
    # records
    "SEARCHED_AND_EMPTY",
    "SEARCH_RECORDS_NOTE",
    "DENSITY_ABSENT_REASON",
    "searched_and_empty_reason",
    "CROSS_ATLAS_CONSISTENCY",
    "DEPOSIT_VS_TABLE1",
    "DEPOSIT_DISCREPANCY_NOTE",
    "TRUTTI_PUBLISHED_VOLUMES",
    # verification loaders (cached-image checks)
    "read_nifti",
    "extract_cit168_member",
    "cit168_probability_integrals",
    "trutti_mask_volumes",
]

# --- citations ---------------------------------------------------------------


def _cited(*parts):
    return " ".join(parts)


#: The atlas whose published volumes this module transcribes. Species:
#: human (in vivo MRI, n=168 adults ages 22-35).
PAULI_CITATION = _cited(
    "Pauli WM, Nili AN, Tyszka JM. 'A high-resolution probabilistic in "
    "vivo atlas of human subcortical brain nuclei.' Scientific Data "
    "2018;5:180063. PMID 29664465. doi:10.1038/sdata.2018.63.",
    "Probabilistic segmentations of 16 human subcortical nuclei from 168 "
    "adults (ages 22-35), delineated on 700 um templates, distributed in "
    "MNI152NLin2009cAsym 1 mm space; deposited on OSF project r2hvk "
    "('An In Vivo High Resolution Atlas of the Subcortical Human Brain') "
    "as the paper's Data Citation. Table 1 publishes per-nucleus volumes "
    "in microliters, 'derived ... by spatial integration of label "
    "probabilities over each hemisphere independently'.",
)

#: Stated in the article's own copyright block: "Open Access This article
#: is licensed under a Creative Commons Attribution 4.0 International
#: License" (with the CC0 waiver applying to the article's metadata files).
PAULI_ARTICLE_LICENCE = (
    "Creative Commons Attribution 4.0 International (CC BY 4.0): the "
    "article's own stated licence."
)

#: The deposit's licence situation, recorded rather than assumed: the OSF
#: project sets no licence field, so the article's statement is what binds.
PAULI_DEPOSIT_NOTE = (
    "The OSF deposit's own licence field is UNSET (verified via the OSF "
    "API, 2026-08-25); the files are used under the article's CC BY 4.0 "
    "statement with this attribution."
)

#: The independent midbrain atlas used as replication. Species: human
#: (in vivo 7T MRI, n=27 healthy participants).
TRUTTI_CITATION = _cited(
    "Trutti AC, Fontanesi L, Mulder MJ, Bazin PL, Hommel B, Forstmann BU. "
    "'A probabilistic atlas of the human ventral tegmental area (VTA) "
    "based on 7 Tesla MRI data.' Brain Structure and Function "
    "2021;226(4):1155-1167. PMID 33580320. doi:10.1007/s00429-021-02231-w.",
    "VTA delineated independently by two raters in 27 healthy "
    "participants (19 female, mean age 24.5, SD 4.8); voxels kept only "
    "where BOTH raters agreed; registered to MNI152 2009b space; "
    "deposited on OSF project 9pzj3. Its definition is broader than a "
    "cytoarchitectonic VTA: it covers the VTA complex including PBP, PN, "
    "IF, CLi and RLi, with the substantia nigra a boundary landmark only.",
)

TRUTTI_ARTICLE_LICENCE = (
    "Creative Commons Attribution 4.0 International (CC BY 4.0): the "
    "article's own stated licence ('(c) The Author(s) 2021, Open Access')."
)

TRUTTI_DEPOSIT_LICENCE_NOTE = (
    "The OSF deposit's own licence field is UNSET (verified via the OSF "
    "API, 2026-08-25); the files are used under the article's CC BY 4.0 "
    "statement with this attribution."
)

#: Why the two atlases' spaces are recorded but never mixed: CIT168 lives
#: in MNI152NLin2009cAsym, Trutti in MNI152 2009b. Different nonlinear
#: templates mean their voxels do NOT correspond one-to-one; this module
#: compares the two only through published scalar volumes, never voxelwise.
ATLAS_SPACES_NOTE = (
    "Spaces differ: CIT168 is distributed in MNI152NLin2009cAsym 1 mm "
    "space, Trutti in MNI152 2009b space. They are compared here only "
    "through published scalar volumes (see CROSS_ATLAS_CONSISTENCY), "
    "never voxel-to-voxel."
)

# --- manifest ---------------------------------------------------------------

MANIFEST_VERSION = "2026-08-25"

#: What a human should run when the cache is missing a file.
FETCH_COMMAND = "scripts/fetch_human_midbrain.py"

_RETRIEVED = "2026-08-25"

#: Default on-disk cache, shared with the other human layers: <repo>/data/human.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

#: Geometry quantities, verbatim in spirit from the sources. Recorded once
#: here because every manifest entry's units field repeats it: NOTHING in
#: this manifest is a receptor density.
GEOMETRY_UNITS = (
    "structural-MRI geometry only: group volumes in microliters (Pauli "
    "2018 Table 1) or voxel probabilities/volumes in cubic millimeters "
    "(Trutti 2021). NOT a receptor density, NOT a binding potential, and "
    "never readable as one through this module"
)


class DataFile(datafetch.DataFile):
    """One external file, defaulting `retrieved` to this manifest's date."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        datafetch.DataFile.__init__(self, *args, **kwargs)


_CITATION_BLOCK = " ".join([
    PAULI_CITATION, PAULI_ARTICLE_LICENCE, PAULI_DEPOSIT_NOTE,
    TRUTTI_CITATION, TRUTTI_ARTICLE_LICENCE, TRUTTI_DEPOSIT_LICENCE_NOTE,
])

_DATA = (
    # --- CIT168 (Pauli 2018): the defining atlas ---------------------------
    (
        "cit168_pauli2018_atlas_bundle",
        "cit168_pauli2018_atlas_bundle.tar.gz",
        "https://osf.io/download/mwqyb/",
        "36e326758c128df7ad53672de38fda9ca80117bfb50b02c51e6bdcde13ec134a",
        349362,
        "The CIT168 release bundle as the authors distribute it: one tar.gz "
        "holding the probabilistic atlas (16 volumes, one per nucleus, in "
        "MNI152NLin2009cAsym 1 mm space), the deterministic atlas, and the "
        "label table info.csv (16 rows: id, label, hemisphere='both', "
        "structure='subcortex'). VERIFICATION ONLY: no loader consumes the "
        "images; product values come from the peer-reviewed Table 1 "
        "transcribed into NUCLEI, because the deposited images integrate "
        "systematically larger than that table (see DEPOSIT_VS_TABLE1). "
        "Read by the fixture maker (which extracts info.csv byte-for-byte) "
        "and by the cache-present checks that quantify the discrepancy.",
    ),
    # --- Trutti 2021: the independent replication --------------------------
    (
        "trutti2021_vta_hem-l_proba27",
        "trutti2021_vta_hem-l_proba-27.nii.gz",
        "https://osf.io/download/zrhk2/",
        "a6904b835d40bc2b4b3d9fab2d3fd6b27e32fdf0d9644a3867ebc14935b83fe6",
        1292881,
        "Trutti 2021 left-hemisphere probabilistic VTA atlas (27 "
        "participants, MNI152 2009b space, 0.5 mm voxels). VERIFICATION "
        "ONLY: parsed by trutti_mask_volumes to confirm the released bytes "
        "reproduce the paper's published unthresholded atlas volume "
        "(2226.375 mm^3 left). Not consumed by any loader.",
    ),
    (
        "trutti2021_vta_hem-r_proba27",
        "trutti2021_vta_hem-r_proba-27.nii.gz",
        "https://osf.io/download/m8yks/",
        "5ed1ca88c55bfa919a392d7e3395616a171aa02367f5bc2701c19a73005e2e0c",
        1297822,
        "Trutti 2021 right-hemisphere probabilistic VTA atlas; same role "
        "as the left file (paper value 2368.00 mm^3 right).",
    ),
)

assert len(_DATA) == 3


def _entry(key, filename, url, sha, size, description):
    return DataFile(
        key=key,
        filename=filename,
        url=url,
        sha256=sha,
        size_bytes=size,
        citation=_CITATION_BLOCK,
        description=description,
        units=GEOMETRY_UNITS,
    )


DATA_FILES = [_entry(*args) for args in _DATA]

DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

#: Members of the CIT168 tarball, named exactly as archived, with the
#: sha256 and size of each INNER file. The Fetcher verifies the outer
#: bundle against the manifest above; these digests verify what gets
#: extracted from it, so a repacked-but-same-length bundle cannot smuggle
#: different bytes past `extract_cit168_member`. Measured 2026-08-25 from
#: the verified bundle.
CIT168_TAR_MEMBERS = {
    "atl-pauli2018/atl-pauli2018_space-MNI152NLin2009cAsym_hemi-both_probabilistic.nii.gz": (
        "7a91c36701f36d9d908479b862e65a879d15f54badcec6878e61d857ab570f66",
        857744,
    ),
    "atl-pauli2018/atl-pauli2018_space-MNI152NLin2009cAsym_hemi-both_deterministic.nii.gz": (
        "0072d6ea3c05d1e798743e7d1f732fbc9c50ee6969bc5e68c6bdba3472d26242",
        51026,
    ),
    "atl-pauli2018/atl-pauli2018_space-MNI152NLin2009cAsym_info.csv": (
        "4bbcbb9940ad42577ab45a2a5d08527f26fb7c9ee7962bb4ad40283450411af2",
        635,
    ),
}

#: Which member the fixture copies and the tests parse.
CIT168_INFO_MEMBER = "atl-pauli2018/atl-pauli2018_space-MNI152NLin2009cAsym_info.csv"
CIT168_PROBABILISTIC_MEMBER = (
    "atl-pauli2018/"
    "atl-pauli2018_space-MNI152NLin2009cAsym_hemi-both_probabilistic.nii.gz"
)

_FETCHER = datafetch.Fetcher(
    manifest=sys.modules[__name__],
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/human",
)


def data_file(key):
    """Look up one manifest entry, naming the alternatives on a miss."""
    try:
        return DATA_FILES_BY_KEY[key]
    except KeyError:
        raise KeyError(
            "unknown data file %r; manifest %s has: %s"
            % (key, MANIFEST_VERSION, ", ".join(sorted(DATA_FILES_BY_KEY)))
        )


def cache_path(key, cache_dir=None):
    return _FETCHER.cache_path(key, cache_dir)


def verify(key, cache_dir=None):
    return _FETCHER.verify(key, cache_dir)


def is_cached(key, cache_dir=None):
    return _FETCHER.is_cached(key, cache_dir)


def download(key, cache_dir=None, timeout=300, force=False):
    return _FETCHER.download(key, cache_dir=cache_dir, timeout=timeout, force=force)


def ensure(key, cache_dir=None, allow_download=True, timeout=300):
    return _FETCHER.ensure(
        key, cache_dir=cache_dir, allow_download=allow_download, timeout=timeout
    )


def fetch_all(keys=None, cache_dir=None, timeout=300, force=False, log=None):
    return _FETCHER.fetch_all(
        keys=keys, cache_dir=cache_dir, timeout=timeout, force=force, log=log
    )


def checksum_lines():
    return _FETCHER.checksum_lines()


# --- the nuclei, with their published volumes --------------------------------


class MidbrainNucleus(object):
    """One CIT168 nucleus and its published volume.

    Every field except ``model_role`` is transcribed from the source:
    ``label`` from the deposited info.csv, ``acronym`` and the volumes from
    Pauli 2018 Table 1. ``volume_ul_left`` / ``volume_ul_right`` are GROUP
    volumes in microliters -- macroscopic anatomy, not a receptor count,
    and typed accordingly: there is no density accessor on this class at
    all.
    """

    __slots__ = (
        "id", "label", "acronym", "volume_ul_left", "volume_ul_right",
        "laterality_percent", "model_role", "model_role_note",
    )

    def __init__(self, id, label, acronym, volume_ul_left, volume_ul_right,
                 laterality_percent, model_role, model_role_note=""):
        self.id = int(id)
        self.label = str(label)
        self.acronym = str(acronym)
        self.volume_ul_left = float(volume_ul_left)
        self.volume_ul_right = float(volume_ul_right)
        self.laterality_percent = float(laterality_percent)
        #: "scenario-node" (a `MidbrainNode` exists for it) or
        #: "geometry-only" (transcribed for completeness, consumed by no
        #: scenario).
        self.model_role = model_role
        self.model_role_note = model_role_note

    @property
    def citation(self):
        return " ".join([PAULI_CITATION, PAULI_ARTICLE_LICENCE])

    @property
    def geometry(self):
        return GEOMETRY_UNITS

    def __repr__(self):
        return "MidbrainNucleus(%d, %r, %.0f/%.0f ul)" % (
            self.id, self.label, self.volume_ul_left, self.volume_ul_right,
        )


_ROLE_VTA = "scenario-node"
_ROLE_GEOMETRY = "geometry-only"


#: All 16 CIT168 nuclei, keyed by the deposited volume index (the ``id``
#: column of info.csv, which is also the fourth-axis index of the
#: probabilistic image minus one). Volumes are Pauli 2018 Table 1 verbatim;
#: the VTA row is published under the acronym VTA with the name "Ventral
#: Tegmentum". A test re-checks ids and labels against the committed copy
#: of info.csv whenever the fixtures are present, and the full table
#: against the cached info.csv member when the data cache is present.
NUCLEI = {
    1: MidbrainNucleus(
        1, "putamen", "Pu", 5224, 5196, 0.3, _ROLE_GEOMETRY,
        "striatal; covered by the Cammoun-033 putamen parcels and the "
        "Hansen PET maps already",
    ),
    2: MidbrainNucleus(
        2, "caudate_nucleus", "Ca", 4493, 4543, -0.6, _ROLE_GEOMETRY,
        "striatal; likewise already parcellated upstream",
    ),
    3: MidbrainNucleus(
        3, "nucleus_accumbens", "NAC", 397, 399, -0.3, _ROLE_GEOMETRY,
        "already a connectome node via Laccumb/Raccumb",
    ),
    4: MidbrainNucleus(
        4, "extended_amygdala", "EXA", 134, 136, -1.0, _ROLE_GEOMETRY,
        "no scenario addresses it; amygdala coverage is upstream too",
    ),
    5: MidbrainNucleus(
        5, "globus_pallidus_external", "GPe", 696, 678, 1.3, _ROLE_GEOMETRY,
        "covered upstream by the pallidum parcels",
    ),
    6: MidbrainNucleus(
        6, "globus_pallidus_internal", "GPi", 383, 383, 0.0, _ROLE_GEOMETRY,
        "covered upstream by the pallidum parcels",
    ),
    7: MidbrainNucleus(
        7, "substantia_nigra_pars_compacta", "SNc", 132, 136, -1.5, _ROLE_VTA,
        "the nigrostriatal dopamine source; a dopaminergic scenario needs "
        "it as a distinct node, and no upstream parcellation resolves it",
    ),
    8: MidbrainNucleus(
        8, "red_nucleus", "RN", 301, 298, 0.5, _ROLE_GEOMETRY,
        "motor-cerebellar relay; no drug scenario targets it",
    ),
    9: MidbrainNucleus(
        9, "substantia_nigra_pars_reticulata", "SNr", 261, 269, -1.4,
        _ROLE_GEOMETRY,
        "GABAergic output of the basal ganglia; no scenario addresses it "
        "yet, unlike SNc whose dopaminergic role the scenarios model",
    ),
    10: MidbrainNucleus(
        10, "parabrachial_pigmented_nucleus", "PBP", 99, 98, 0.5, _ROLE_VTA,
        "part of the VTA complex proper (Trutti's broad VTA includes it); "
        "kept separate from VTA because CIT168 measures them separately",
    ),
    11: MidbrainNucleus(
        11, "ventral_tegmental_area", "VTA", 33, 33, 0.1, _ROLE_VTA,
        "THE node the mu-opioid reward scenario needs; published under "
        "the name 'Ventral Tegmentum'. Note the volume is an order of "
        "magnitude smaller than the VTA COMPLEX other atlases report, "
        "because CIT168 splits PBP out",
    ),
    12: MidbrainNucleus(
        12, "ventral_pallidum", "VeP", 68, 74, -4.3, _ROLE_GEOMETRY,
        "no scenario addresses it",
    ),
    13: MidbrainNucleus(
        13, "habenular_nuclei", "HN", 29, 27, 4.5, _ROLE_GEOMETRY,
        "epithalamus, not midbrain; transcribed because it is one table",
    ),
    14: MidbrainNucleus(
        14, "hypothalamus", "HTH", 604, 617, -1.1, _ROLE_GEOMETRY,
        "covered upstream by no parcel, but no scenario addresses it yet",
    ),
    15: MidbrainNucleus(
        15, "mammillary_nucleus", "MN", 64, 62, 2.0, _ROLE_GEOMETRY,
        "no scenario addresses it",
    ),
    16: MidbrainNucleus(
        16, "subthalamic_nucleus", "STH", 135, 128, 2.9, _ROLE_GEOMETRY,
        "no scenario addresses it",
    ),
}

assert len(NUCLEI) == 16

#: Nuclei some scenario layer needs (or will need) as circuit nodes.
SCENARIO_NUCLEI = tuple(n for n in NUCLEI.values() if n.model_role == _ROLE_VTA)

assert tuple(sorted(n.id for n in SCENARIO_NUCLEI)) == (7, 10, 11)

# --- the nodes ---------------------------------------------------------------


class MidbrainNode(object):
    """One hemisphere of one midbrain nucleus, addressed by node label.

    This is the type a future network builder would put in a node list.
    It carries IDENTITY and GEOMETRY and nothing else:

    * no density attribute exists, and ``__slots__`` means none can be
      attached later -- the absence of receptor density is structural,
      not documentary;
    * `receptor_density` refuses loudly and names what a real measurement
      would have to be;
    * the geometry is a published GROUP volume in microliters, carried
      with its citation so any downstream use inherits the source.

    Labels follow the ENIGMA subcortical convention (hemisphere letter +
    short name, like ``Laccumb``): LVTA, RVTA, LSNc, RSNc, LPBP, RPBP.
    """

    __slots__ = ("label", "nucleus_id", "hemisphere", "volume_ul")

    def __init__(self, label, nucleus_id, hemisphere, volume_ul):
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "nucleus_id", int(nucleus_id))
        object.__setattr__(self, "hemisphere", str(hemisphere))
        object.__setattr__(self, "volume_ul", float(volume_ul))

    def __setattr__(self, name, value):
        raise AttributeError(
            "%s attributes are fixed at construction; %r cannot be set "
            "(and no attribute can carry a receptor density -- see "
            "MidbrainDensityAbsent)" % (type(self).__name__, name)
        )

    @property
    def nucleus(self):
        """The `MidbrainNucleus` this node is one hemisphere of."""
        return NUCLEI[self.nucleus_id]

    @property
    def nucleus_label(self):
        return self.nucleus.label

    @property
    def acronym(self):
        return self.nucleus.acronym

    @property
    def citation(self):
        return self.nucleus.citation

    @property
    def geometry(self):
        return GEOMETRY_UNITS

    def __float__(self):
        raise TypeError(
            "%s is a brain region, not a number; its only quantity is "
            "%r (see .geometry)" % (self.label, self.volume_ul)
        )

    def receptor_density(self, receptor_key):
        """There is no such measurement. This method refuses, loudly.

        See `MidbrainDensityAbsent` -- raised with this node, the receptor
        asked about, and the reason, which is also the text of the record
        at `SEARCHED_AND_EMPTY`.
        """
        raise MidbrainDensityAbsent(node=self, receptor_key=receptor_key)

    def __repr__(self):
        return "MidbrainNode(%r, %s, %.0f ul, geometry only)" % (
            self.label, self.acronym, self.volume_ul,
        )


def _node_label(hemisphere, acronym):
    return hemisphere + acronym


#: The six scenario-addressable midbrain nodes, built from SCENARIO_NUCLEI.
NODES = {}
for _nucleus in SCENARIO_NUCLEI:
    for _hemisphere, _published_ul in (
        ("L", _nucleus.volume_ul_left),
        ("R", _nucleus.volume_ul_right),
    ):
        NODES[_node_label(_hemisphere, _nucleus.acronym)] = MidbrainNode(
            label=_node_label(_hemisphere, _nucleus.acronym),
            nucleus_id=_nucleus.id,
            hemisphere=_hemisphere,
            volume_ul=_published_ul,
        )

MIDBRAIN_NODE_LABELS = tuple(sorted(NODES))

assert MIDBRAIN_NODE_LABELS == ("LPBP", "LSNc", "LVTA", "RPBP", "RSNc", "RVTA")


def node(label):
    """One `MidbrainNode` by its label, naming the alternatives on a miss."""
    try:
        return NODES[label]
    except KeyError:
        raise KeyError(
            "%r is not a midbrain node label; this module defines %s"
            % (label, ", ".join(MIDBRAIN_NODE_LABELS))
        )


def scenario_node_labels():
    """The six labels a network builder would add, in registry order."""
    return MIDBRAIN_NODE_LABELS


# --- enforcement -------------------------------------------------------------


class MidbrainDensityAbsent(TypeError):
    """A receptor density was requested where none exists.

    Raised by `MidbrainNode.receptor_density` and by every read path into
    the PET receptor layer when a midbrain node's label is asked for.
    Carries `.node` (the `MidbrainNode`, or None when the caller asked by
    label alone), `.receptor_key`, and `.reason`.

    Deliberately a TypeError and not a KeyError: a KeyError reads as "look
    somewhere else", and there is nowhere else -- the honest responses are
    the ones the message lists.
    """

    def __init__(self, node=None, receptor_key=None, reason=None):
        if reason is None:
            reason = DENSITY_ABSENT_REASON
        TypeError.__init__(
            self,
            "%s%s carries NO receptor-density measurement: neither CIT168 "
            "(Pauli 2018) nor the Trutti 2021 VTA atlas measured receptors, "
            "and no measured density for the human %s was found anywhere "
            "(see neurocascade.human.midbrain.SEARCHED_AND_EMPTY). The "
            "Hansen PET atlas contains this territory only inside its "
            "undifferentiated Cammoun-033 brainstem parcel, and reading "
            "that parcel's value as a midbrain density is refused "
            "(midbrain.pet_value_from_brainstem_parcel) because it averages "
            "over tissue far beyond the node. %s%s"
            % (
                repr(node.label) + " " if node is not None else "",
                ("(%s)" % node.acronym) if node is not None else "This region",
                node.nucleus_label.replace("_", " ") if node is not None
                else "ventral midbrain",
                reason,
                " [receptor asked about: %s]" % receptor_key
                if receptor_key is not None else "",
            ),
        )
        self.node = node
        self.receptor_key = receptor_key
        self.reason = reason


_MIDBRAIN_REQUEST_ALIASES = {
    "VTA": "ventral_tegmental_area",
    "VENTRALTEGMENTUM": "ventral_tegmental_area",
    "VENTRALTEGMENTALAREA": "ventral_tegmental_area",
    "SNC": "substantia_nigra_pars_compacta",
    "SUBSTANTIANIGRACOMPACTA": "substantia_nigra_pars_compacta",
    "SUBSTANTIANIGRAPARSCOMPACTA": "substantia_nigra_pars_compacta",
    "PBP": "parabrachial_pigmented_nucleus",
    "PARABRACHIALPIGMENTEDNUCLEUS": "parabrachial_pigmented_nucleus",
}

_NORMALIZE = re.compile(r"[^A-Za-z]")


def is_midbrain_request(label):
    """True if a label names -- however spelled -- a node this module defines.

    Catches the spellings a caller plausibly tries before giving up:
    ``LVTA``/``RVTA`` (this module's own convention), bare ``VTA``,
    ``L_SNC``, ``vta``, ``substantia_nigra_pars_compacta``, and the same
    with separators. Exactness elsewhere matters more than cleverness
    here: anything NOT matching returns False and stays whatever the
    caller's own lookup does with it.
    """
    text = _NORMALIZE.sub("", str(label)).upper()
    if not text:
        return False
    candidates = [text]
    if len(text) > 1 and text[0] in ("L", "R"):
        candidates.append(text[1:])
    return any(candidate in _MIDBRAIN_REQUEST_ALIASES for candidate in candidates)


def node_for_request(label):
    """The `MidbrainNode` a request label refers to, or None.

    The inverse of `is_midbrain_request`: same normalization, but returns
    the node so the refuser can say WHICH region was meant. A hemisphere
    prefix selects the hemisphere; a bare alias like ``VTA`` returns the
    LEFT node (and callers that care about hemispheres should pass labels
    that do).
    """
    text = _NORMALIZE.sub("", str(label)).upper()
    if not text:
        return None
    hemisphere = ""
    candidates = [(text, "")]
    if len(text) > 1 and text[0] in ("L", "R"):
        candidates.append((text[1:], text[0]))
    for candidate, prefix in candidates:
        nucleus_label = _MIDBRAIN_REQUEST_ALIASES.get(candidate)
        if nucleus_label is None:
            continue
        hemisphere = prefix or "L"
        for nucleus in NUCLEI.values():
            if nucleus.label == nucleus_label:
                return NODES[hemisphere + nucleus.acronym]
    return None


def pet_value_from_brainstem_parcel(pet_map, node_or_label):
    """There is no such value. This function refuses, loudly.

    The Hansen PET atlas DOES contain a number for the Cammoun-033
    ``brainstem`` parcel, and the ventral midbrain lies inside that
    parcel's territory -- which is exactly why this function exists as a
    named API instead of leaving the temptation to a working lookup: the
    parcel covers pons and medulla as well as the midbrain, several
    centimetres of tissue, so its z-score is not a VTA, SNc or PBP
    measurement, and assigning it to a midbrain node is the
    "average the brainstem parcel onto it" move the roadmap forbids in
    terms.

    Raises `MidbrainDensityAbsent` always. Never returns. If a future
    phase wants a midbrain coupling coefficient, the honest routes are a
    tracer study that actually parcellates the midbrain, or a measured
    crosswalk -- not this function growing a return value.
    """
    target = (
        node_or_label if isinstance(node_or_label, MidbrainNode)
        else node_for_request(node_or_label)
    )
    if target is None:
        raise ValueError(
            "pet_value_from_brainstem_parcel refuses midbrain requests; %r "
            "does not name one of %s" % (node_or_label, ", ".join(MIDBRAIN_NODE_LABELS))
        )
    # Touch pet_map only to fail fast on the wrong type: the point is the
    # refusal, not the lookup it refuses to perform.
    if not hasattr(pet_map, "for_parcel"):
        raise TypeError(
            "expected a receptors.PETRelativeDensityMap-like object, got %r"
            % (type(pet_map).__name__,)
        )
    raise MidbrainDensityAbsent(
        node=target,
        receptor_key=getattr(getattr(pet_map, "receptor", None), "key", None),
        reason=(
            "This call IS the refusal: the containing Cammoun-033 brainstem "
            "parcel has a value (%s), but it is measured over pons and "
            "medulla together with the midbrain and is not a density for "
            "%s." % (
                getattr(type(pet_map), "__name__", "the map"),
                target.label,
            )
        ),
    )


# --- the searched-and-empty record --------------------------------------------

#: How the search behind `SEARCHED_AND_EMPTY` was run, so it can be
#: re-run rather than trusted. Everything went through the Europe PMC REST
#: API (www.ebi.ac.uk/europepmc/webservices/rest/search), resultType core,
#: plus full-text retrieval where a PMCID existed. Queries actually issued,
#: with the hit counts returned on 2026-08-25:
#:
#: * ``TITLE:"carfentanil" AND ABSTRACT:"positron emission tomography"
#:   AND ABSTRACT:"ventral tegmental area"`` -- **0 hits**;
#: * ``TITLE:"ventral tegmental area" AND (ABSTRACT:autoradiography OR
#:   ABSTRACT:"receptor density" OR ABSTRACT:Bmax)`` -- 10 hits, every
#:   one a rodent study;
#: * ``(TITLE:midbrain OR TITLE:"substantia nigra" OR TITLE:"ventral
#:   tegmentum") AND ABSTRACT:autoradiography AND ABSTRACT:human`` --
#:   8 hits: the three human midbrain autoradiography papers recorded
#:   individually below (all measuring targets other than MOR/D2), and
#:   five rodent/disease studies;
#: * earlier in P12.3, a broader carfentanil-PET sweep returned 157 hits,
#:   none of which parcellates a midbrain nucleus.
SEARCH_RECORDS_NOTE = (
    "Europe PMC REST search, queries and hit counts recorded in this "
    "module's docstring block (SEARCH_RECORDS_NOTE); no measured receptor "
    "density exists for the human ventral tegmental area or any "
    "neighbouring midbrain nucleus"
)

#: The one-sentence form of that negative result, used verbatim in every
#: `MidbrainDensityAbsent` whose caller did not supply its own reason.
#: Kept next to the record it summarizes so the two cannot drift apart.
DENSITY_ABSENT_REASON = (
    "A Europe PMC search over candidate sources found NO measured "
    "receptor density for any human midbrain node: each candidate was "
    "wrong region, wrong receptor target, wrong species, or a "
    "gene-expression proxy rather than a measurement "
    "(neurocascade.human.midbrain.SEARCHED_AND_EMPTY)"
)


#: Every candidate source for a measured human VTA receptor density that
#: the search surfaced, with why it cannot be used. House format:
#: {citation, status, what_was_tried, consequence}. The conclusion is the
#: negative result itself: MOR and D2 density on a human midbrain node
#: would have to be invented, and rule 1 forbids that -- so the midbrain
#: nodes carry geometry only, enforced by `MidbrainDensityAbsent`.
SEARCHED_AND_EMPTY = {
    "Pfeiffer1982_PMID6289997": {
        "citation": (
            "Pfeiffer A, Pasi A, Mehraein P, Herz A. 'Opiate receptor "
            "binding sites in human brain.' Brain Research "
            "1982;248:87-96. PMID 6289997. doi:10.1016/0006-8993(82)91150-7"
        ),
        "status": "full abstract reached; VTA not among the regions examined",
        "what_was_tried": (
            "Europe PMC abstract: competitive binding on human brain "
            "MEMBRANE HOMOGENATES -- 'mu-, delta-and chi-types of opiate "
            "binding were found in 5 different regions of the brain', none "
            "named or implied to be midbrain; homogenate binding over whole "
            "regions could not be parcellated to a VTA node even if one "
            "were listed"
        ),
        "consequence": (
            "no usable VTA density: wrong granularity (homogenates, five "
            "coarse regions) and no midbrain region reported"
        ),
    },
    "Itzhak1982_PMID6292630": {
        "citation": (
            "Itzhak Y, Bonnet KA, Groth J, Hiller JM, Simon EJ. 'Multiple "
            "opiate binding sites in human brain regions: evidence for "
            "kappa and sigma sites.' Life Sciences 1982;31:1363-1366. "
            "PMID 6292630. doi:10.1016/0024-3205(82)90382-4"
        ),
        "status": "metadata reached; no abstract available; VTA not indicated",
        "what_was_tried": (
            "Europe PMC record (no abstract online): homogenate binding in "
            "human brain regions characterising kappa and sigma sites; "
            "title and available metadata list no midbrain region"
        ),
        "consequence": (
            "no usable VTA density: kappa/sigma pharmacology is not the "
            "mu site being modelled, and no midbrain region is indicated"
        ),
    },
    "Gabilondo1995_PMID7552322": {
        "citation": (
            "Gabilondo AM, Meana JJ, Garcia-Sevilla JA. 'Increased density "
            "of mu-opioid receptors in the postmortem brain of suicide "
            "victims.' Brain Research 1995;682:245-250. PMID 7552322. "
            "doi:10.1016/0006-8993(95)00333-L"
        ),
        "status": "full abstract reached; measured regions exclude the midbrain",
        "what_was_tried": (
            "Europe PMC abstract: THE closest thing to a human mu-opioid "
            "absolute anchor outside cortex -- postmortem Bmax by "
            "[3H]DAGO binding in SUICIDE VICTIMS AND CONTROLS -- but the "
            "regions measured are frontal cortex, thalamus and caudate "
            "only"
        ),
        "consequence": (
            "no usable VTA density: right receptor, right species, right "
            "quantity (Bmax), wrong regions; extrapolating cortical or "
            "caudate values onto the VTA would be invention"
        ),
    },
    "Gabilondo1994_PMID7862885": {
        "citation": (
            "Gabilondo AM, Meana JJ, Barturen F, Sastre M, Garcia-Sevilla "
            "JA. 'mu-Opioid receptor and alpha 2-adrenoceptor agonist "
            "binding sites in the postmortem brain of heroin addicts.' "
            "Psychopharmacology 1994;115:135-140. PMID 7862885. "
            "doi:10.1007/BF02244763"
        ),
        "status": "full abstract reached; measured regions exclude the midbrain",
        "what_was_tried": (
            "Europe PMC abstract: postmortem [3H]DAGO mu-opioid Bmax in "
            "HEROIN ADDICTS, measured in frontal cortex, thalamus and "
            "caudate (plus hypothalamus for alpha-2)"
        ),
        "consequence": (
            "no usable VTA density: same shape as Gabilondo 1995 -- right "
            "receptor and species, no midbrain region"
        ),
    },
    "Allen1991_PMID1748734_angiotensin_II": {
        "citation": (
            "Allen AM, Paxinos G, McKinley MJ, Chai SY, Mendelsohn FAO. "
            "'Localization and characterization of angiotensin II "
            "receptor binding sites in the human basal ganglia, thalamus, "
            "midbrain, pons, and cerebellum.' J Comp Neurol 1991;312:291-298. "
            "PMID 1748734. doi:10.1002/cne.903120211"
        ),
        "status": (
            "full abstract reached; proves human midbrain autoradiography "
            "exists, but for the WRONG TARGET"
        ),
        "what_was_tried": (
            "Europe PMC abstract: quantitative in vitro autoradiography of "
            "HUMAN midbrain with 125I-[Sar1,Ile8]angiotensin II -- high "
            "density in substantia nigra pars compacta, interpeduncular "
            "nucleus and raphe; the technique reaches the structures this "
            "module cares about, but the ligand is angiotensin II"
        ),
        "consequence": (
            "no usable VTA density: substituting another target's binding "
            "for MOR or D2 is exactly what rule 2 forbids, however good "
            "the anatomical resolution"
        ),
    },
    "Ball1994_PMID7834343_excitatory_amino_acid": {
        "citation": (
            "Ball EF, Shaw PJ, Ince PG, Johnson M. 'The distribution of "
            "excitatory amino acid receptors in the normal human midbrain "
            "and basal ganglia ... a quantitative autoradiographic study "
            "using [3H]MK-801, [3H]glycine, [3H]CNQX and [3H]kainate.' "
            "Brain Research 1994;658:209-218. PMID 7834343. "
            "doi:10.1016/S0006-8993(09)90028-2"
        ),
        "status": (
            "full abstract reached; quantitative HUMAN midbrain "
            "autoradiography of glutamatergic sites, not MOR/D2"
        ),
        "what_was_tried": (
            "Europe PMC abstract: densities of NMDA, glycine-site, AMPA "
            "and kainate binding across normal human midbrain and basal "
            "ganglia -- the closest existing template for HOW a midbrain "
            "density map would look, for four targets none of which is "
            "mu-opioid or dopamine D2"
        ),
        "consequence": (
            "no usable VTA density: wrong receptors (rule 2); recorded "
            "because it shows the measurement is possible in principle "
            "and remains undone for the systems the scenarios model"
        ),
    },
    "Chai1990_PMID2153714_converting_enzyme": {
        "citation": (
            "Chai SY, McKenzie JS, McKinley MJ, Mendelsohn FAO. "
            "'Angiotensin converting enzyme in the human basal forebrain "
            "and midbrain visualized by in vitro autoradiography.' "
            "J Comp Neurol 1990;291:179-194. PMID 2153714. "
            "doi:10.1002/cne.902910203"
        ),
        "status": (
            "full abstract reached; human midbrain autoradiography of an "
            "enzyme, not a receptor of interest"
        ),
        "what_was_tried": (
            "Europe PMC abstract: 125I-351A labelling of angiotensin "
            "converting enzyme across human basal forebrain and midbrain"
        ),
        "consequence": (
            "no usable VTA density: an enzyme map, not a receptor density "
            "of any kind, let alone MOR or D2"
        ),
    },
    "Noppari2026_PMID41108393_carfentanil_PET": {
        "citation": (
            "Noppari T, Tuisku J, Lukkarinen L, Tani P, Lindberg N, Saure "
            "E, Lauerma H, et al. 'Aberrant type 2 dopamine and mu-opioid "
            "receptor availability in autism spectrum disorder.' Eur J "
            "Nucl Med Mol Imaging 2026;53:2069-2082. PMID 41108393. "
            "doi:10.1007/s00259-025-07620-5"
        ),
        "status": "full abstract reached; ROI set does not include the midbrain",
        "what_was_tried": (
            "Europe PMC abstract: the most recent in-vivo human "
            "[11C]carfentanil (MOR) + [11C]raclopride (D2R) PET study the "
            "search surfaced; its regional analysis covers striatum, "
            "nucleus accumbens, globus pallidus, hippocampi and cuneal/"
            "precuneal cortex -- no midbrain nucleus is parcellated"
        ),
        "consequence": (
            "no usable VTA density: right tracers, in vivo, but the "
            "midbrain is absent from the ROI set; PET resolution and "
            "partial-volume handling for a 33-ul nucleus are themselves "
            "open problems the paper does not claim to solve"
        ),
    },
    "Ceballos2026_PMID41845130_neuropeptide_atlas": {
        "citation": (
            "Ceballos EG, Farahani A, Liu ZQ, Milisav F, Hansen JY, "
            "Dagher A, Misic B. 'Organization of neuropeptide systems in "
            "the human brain.' Nature Neuroscience 2026;29:1212-1224. "
            "PMID 41845130. doi:10.1038/s41593-026-02236-w"
        ),
        "status": "full abstract reached; gene-transcription PROXY, not a measurement",
        "what_was_tried": (
            "Europe PMC abstract: a whole-brain atlas of 38 neuropeptide "
            "receptors explicitly built 'using gene transcription as a "
            "proxy'; includes subcortex, but the values are expression "
            "levels, not measured densities"
        ),
        "consequence": (
            "no usable VTA density: adopting it would reintroduce the "
            "mRNA-proxy assumption P12.4 exists to retire, and rule 1 "
            "does not allow calling a proxy a measurement because the "
            "direct one is missing"
        ),
    },
    "hansen2022_MOR_map_brainstem_parcel": {
        "citation": (
            "Hansen JY et al., Nat Neurosci 2022;25:1569-1581, PMID "
            "36303070 -- the in-vivo MOR map already ingested at "
            "receptors.py, parcellated with Cammoun scale033"
        ),
        "status": (
            "already ingested; contains the territory but NOT a midbrain "
            "measurement"
        ),
        "what_was_tried": (
            "The map's own parcellation: the ventral midbrain falls "
            "inside the single undifferentiated 'brainstem' parcel, which "
            "covers pons and medulla with it and has no Desikan-Killiany "
            "counterpart (UNMAPPED_PARCELS)"
        ),
        "consequence": (
            "no usable VTA density; the parcel's value is refused for "
            "midbrain nodes by construction (pet_value_from_brainstem_parcel)"
        ),
    },
    "rodent_VTA_autoradiography_literature": {
        "citation": (
            "Exemplar (of the 10 hits of the TITLE:'ventral tegmental "
            "area' x autoradiography/receptor-density Europe PMC query, "
            "all rat or mouse): Piras AP, Zhou Y, Schlussman SD, Ho A, "
            "Kreek MJ. Neuroscience 2010;169. PMID 20452406. "
            "doi:10.1016/j.neuroscience.2010.04.060"
        ),
        "status": (
            "wrong species, uniformly: every hit is rat or mouse tissue"
        ),
        "what_was_tried": (
            "The exemplar's abstract: in situ [35S]GTPgammaS binding "
            "autoradiography measuring kappa-opioid-receptor-stimulated "
            "G-protein activation IN THE RAT VENTRAL TEGMENTAL AREA -- "
            "the measurement this module wants, done properly, in the "
            "wrong species for a different receptor site"
        ),
        "consequence": (
            "no usable VTA density: rule 2 forbids substituting another "
            "species' measurement even where the structure is homologous"
        ),
    },
}


def searched_and_empty_reason(record_key):
    """One record's consequence text, raising KeyError for unknown keys."""
    try:
        return SEARCHED_AND_EMPTY[record_key]["consequence"]
    except KeyError:
        raise KeyError(
            "no search record %r; recorded searches are %s"
            % (record_key, ", ".join(sorted(SEARCHED_AND_EMPTY)))
        )


# --- cross-atlas consistency, and the deposit-vs-paper discrepancy -----------

#: Published Trutti volumes, verbatim from the paper's Results, pinned here
#: because the cache-present check reproduces two of them from the released
#: bytes. The individual-mask means are the paper's OWN statistic (not
#: recomputable from the group files); the unthresholded atlas volumes ARE
#: reproduced bit-exactly by `trutti_mask_volumes`.
TRUTTI_PUBLISHED_VOLUMES = {
    "individual_mask_mean_mm3_left": 137.35,
    "individual_mask_sd_mm3_left": 38.27,
    "individual_mask_mean_mm3_right": 138.80,
    "individual_mask_sd_mm3_right": 39.33,
    "unthresholded_atlas_volume_mm3_left": 2226.375,
    "unthresholded_atlas_volume_mm3_right": 2368.00,
}

#: What the two atlases agree on, and what that agreement does and does
#: not establish. Written out numerically so nobody has to take "consistent"
#: on faith -- or mistake it for validation.
CROSS_ATLAS_CONSISTENCY = {
    "trutti_vta_complex_individual_masks": (
        "137.35 mm^3 left (SD 38.27), 138.80 mm^3 right (SD 39.33): the "
        "mean volume of individually hand-delineated VTA masks over 27 "
        "participants (Trutti 2021 Results)"
    ),
    "pauli_table1_vta_plus_pbp": (
        "132 ul left (33 + 99), 131 ul right (33 + 98): CIT168's VTA plus "
        "its separately-measured parabrachial pigmented nucleus, the two "
        "labels whose union corresponds to Trutti's broader VTA complex "
        "(Table 1)"
    ),
    "agreement": (
        "~130-140 mm^3 per hemisphere from two independent samples, "
        "protocols and spaces; Trutti notes the same ~138 mm^3 equals "
        "postmortem reconstructions of VTA volume (Halliday & Tork 1986)"
    ),
    "what_it_is_not": (
        "validation, and not a voxelwise comparison either: the atlases "
        "live in different spaces (MNI152NLin2009cAsym vs MNI152 2009b), "
        "use different definitions (Trutti's VTA INCLUDES PBP; CIT168 "
        "separates them), and compare different quantities (mean binary "
        "mask vs sum of two probability integrals). Same macroscopic "
        "structure, described twice; no number here may be transferred "
        "from one atlas to the other"
    ),
}

#: Probability-weighted volumes computed FROM the deposited CIT168 images
#: (sum of voxel probabilities x voxel volume, per hemisphere, 1 mm^3
#: voxels), measured 2026-08-25 from checksum-verified bytes. Recorded so
#: the discrepancy with the paper's own Table 1 is a number anyone can
#: reproduce, not an assertion. Keyed by nucleus id -> (left, right) ul.
DEPOSIT_INTEGRALS_UL = {
    1: (6917.750428799806, 6796.640014830026),
    2: (5960.56458146247, 5998.083357566661),
    3: (513.003293290428, 512.1025555947172),
    4: (186.73585993906954, 189.67668930620633),
    5: (987.0541734697263, 944.1705837352318),
    6: (507.76730949800367, 506.92887935261155),
    7: (157.09909265335182, 165.34667914977),
    8: (388.70169387812166, 385.68692076526844),
    9: (339.5434487672061, 349.5324349844681),
    10: (118.12330204422778, 117.7892584312612),
    11: (38.87825696613868, 39.52323454378419),
    12: (94.5723738680008, 104.72796345378123),
    13: (47.43961690296419, 44.7099237231746),
    14: (841.4767603912044, 854.3106999646138),
    15: (87.78360759641055, 92.58360322715966),
    16: (186.5651310136891, 175.76385055788288),
}

#: The finding itself: EVERY deposited CIT168 nucleus integrates larger
#: than the paper's published volume for it -- 18% to 66% depending on
#: nucleus (min/max ratio over all 32 hemisphere values) -- while the
#: p>=0.5 thresholded cores are mostly SMALLER than Table 1 (the VTA core
#: is 9.3 vs the published 33 ul). So the deposited maps are neither the
#: thresholded cores nor consistent with the published integral, and the
#: deposit documents no transformation between the two. Hypotheses (labelled
#: as such, not established): the deposited files may post-date the paper's
#: tables (the retrieved bundle's internal dates are 2024-11-11, six years
#: after publication), or Table 1 may have been computed on differently-
#: processed labels. Nothing here is resolved by guessing, which is why the
#: response is to NOT use the deposit-derived geometry at all.
DEPOSIT_DISCREPANCY_NOTE = (
    "The deposited CIT168 probabilistic images yield probability-integrals "
    "18-66% ABOVE Pauli 2018 Table 1 for every one of the 16 nuclei (both "
    "hemispheres), while their p>=0.5 cores fall far BELOW it (VTA: 9.3 vs "
    "33 ul). Same stated method ('spatial integration of label "
    "probabilities over each hemisphere'), different answer, no documented "
    "explanation. Candidate causes are hypotheses only: the retrieved "
    "bundle's internal file dates are 2024-11-11 (post-dating the 2018 "
    "paper), so the served bytes may not be the ones the table describes. "
    "Consequence for this module: product values are the PEER-REVIEWED "
    "Table 1 transcriptions; no centroid, mask or coordinate derived from "
    "the deposit is exposed, because each would inherit an unexplained "
    "discrepancy. The comparison numbers stay reproducible: see "
    "DEPOSIT_INTEGRALS_UL and cit168_probability_integrals()."
)

#: The dict-shaped view the tests assert against: per nucleus, published
#: vs measured, with ratios. Built from the two registries above so the
#: two cannot drift apart.
DEPOSIT_VS_TABLE1 = {
    nucleus.id: {
        "label": nucleus.label,
        "table1_ul": (nucleus.volume_ul_left, nucleus.volume_ul_right),
        "deposit_integral_ul": DEPOSIT_INTEGRALS_UL[nucleus.id],
        "ratio": (
            DEPOSIT_INTEGRALS_UL[nucleus.id][0] / nucleus.volume_ul_left,
            DEPOSIT_INTEGRALS_UL[nucleus.id][1] / nucleus.volume_ul_right,
        ),
    }
    for nucleus in NUCLEI.values()
}

assert min(
    min(side["ratio"]) for side in DEPOSIT_VS_TABLE1.values()
) > 1.17
assert max(
    max(side["ratio"]) for side in DEPOSIT_VS_TABLE1.values()
) < 1.67


# --- minimal NIfTI-1 reader, and the verification loaders --------------------
#
# nibabel would be the usual tool; this project's dependency policy is numpy
# and scipy only, and the readers below touch the narrow subset of NIfTI-1
# these three files use (little-endian single-dataset gzip-compressed
# floats/uint8, no qform/sform interpretation needed because no coordinate
# is ever exposed). Validated against Trutti's published unthresholded
# atlas volumes, which the released bytes reproduce exactly.


def read_nifti(path):
    """Read one NIfTI-1 file into ``(array, voxel_sizes)``.

    ``array`` comes back in (i, j, k[, t]) order with float64 dtype;
    ``voxel_sizes`` is the (dx, dy, dz) triple from pixdim[1:4]. Handles
    gzip-compressed or raw files, and checks sizeof_hdr rather than
    misreading foreign bytes.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    return _parse_nifti_bytes(raw)


def _parse_nifti_bytes(raw):
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    header = raw[:348]
    sizeof_hdr = struct.unpack("<i", header[:4])[0]
    if sizeof_hdr != 348:
        raise ChecksumMismatch(
            "not a little-endian NIfTI-1 file (sizeof_hdr=%r): refusing to "
            "guess axis order or endianness" % (sizeof_hdr,)
        )
    dim = struct.unpack("<8h", header[40:56])
    ndim = dim[0]
    if not 3 <= ndim <= 4:
        raise ValueError("expected a 3-D or 4-D NIfTI volume, got dim[0]=%d" % ndim)
    datatype = struct.unpack("<h", header[70:72])[0]
    pixdim = struct.unpack("<8f", header[76:108])
    vox_offset = int(struct.unpack("<f", header[108:112])[0])
    dtypes = {
        2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
        256: np.int8, 512: np.uint16,
    }
    if datatype not in dtypes:
        raise ValueError("unsupported NIfTI datatype code %d" % datatype)
    shape_on_disk = tuple(dim[1:ndim + 1])
    count = int(np.prod(shape_on_disk))
    array = np.frombuffer(raw[vox_offset:], dtype=dtypes[datatype], count=count)
    # NIfTI stores x fastest; reshape to C-order reversed shape then undo
    # it, so the array reads (i=x, j=y, k=z[, t]) without transposing axes
    # silently in the caller's mental model.
    array = array.reshape(shape_on_disk[::-1]).transpose(tuple(reversed(range(ndim))))
    voxel_sizes = (float(pixdim[1]), float(pixdim[2]), float(pixdim[3]))
    return array.astype(np.float64), voxel_sizes


def extract_cit168_member(member_name, cache_dir=None):
    """Extract one CIT168 tarball member to a temporary directory.

    Verifies the OUTER bundle against the manifest first, then the INNER
    member against `CIT168_TAR_MEMBERS`, so the returned path holds bytes
    doubly accounted for. Caller owns the temp directory (which this
    function creates and does not clean up -- the extraction is small and
    the tests use it within one test's lifetime).
    """
    if member_name not in CIT168_TAR_MEMBERS:
        raise KeyError(
            "unknown CIT168 member %r; the bundle carries %s"
            % (member_name, ", ".join(sorted(CIT168_TAR_MEMBERS)))
        )
    expected_sha, expected_size = CIT168_TAR_MEMBERS[member_name]
    bundle_path = ensure("cit168_pauli2018_atlas_bundle", cache_dir=cache_dir)
    out_dir = tempfile.mkdtemp(prefix="neurocascade_cit168_")
    with tarfile.open(bundle_path, "r:gz") as archive:
        payload = archive.extractfile(member_name)
        if payload is None:
            raise MissingCachedFile(
                "%s is not a file member of the verified bundle at %s"
                % (member_name, bundle_path)
            )
        data = payload.read()
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expected_sha or len(data) != expected_size:
        raise ChecksumMismatch(
            "member %s of the verified CIT168 bundle has sha256 %s (%d "
            "bytes), expected %s (%d bytes): the bundle's outer digest "
            "matches the manifest but its contents changed upstream; do "
            "NOT edit either digest to match"
            % (member_name, actual_sha, len(data), expected_sha, expected_size)
        )
    destination = os.path.join(out_dir, os.path.basename(member_name))
    with open(destination, "wb") as handle:
        handle.write(data)
    return destination


def cit168_info_rows(cache_dir=None, allow_download=False):
    """The 16 rows of the CIT168 label table, as dicts, from verified bytes.

    Offline by default. Used by the tests to pin `NUCLEI` ids and labels
    against the deposited identity file; carries no geometry and implies
    no density. When the bundle is not cached (a clean checkout), falls
    back to the committed fixture -- which is a byte-identical digest-
    checked copy of this same member, so the identity check stays runnable
    offline without weakening what it checks.
    """
    if allow_download or is_cached("cit168_pauli2018_atlas_bundle", cache_dir):
        path = ensure(
            "cit168_pauli2018_atlas_bundle",
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        with tarfile.open(path, "r:gz") as archive:
            data = archive.extractfile(CIT168_INFO_MEMBER).read()
    else:
        fixture_fallback = os.path.join(
            os.path.dirname(__file__), "fixtures",
            "cit168_pauli2018_info_fixture.csv",
        )
        with open(fixture_fallback, "rb") as handle:
            data = handle.read()
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    if len(rows) != len(NUCLEI):
        raise ValueError(
            "CIT168 info.csv lists %d nuclei, this module's registry has %d"
            % (len(rows), len(NUCLEI))
        )
    return rows


def cit168_probability_integrals(cache_dir=None, allow_download=False):
    """Recompute the deposited CIT168 probability integrals, in ul.

    ``{nucleus id: (left, right)}`` from the VERIFIED bundle: sum of voxel
    probabilities times voxel volume over each hemisphere independently --
    the method Pauli 2018 states for Table 1. Verification only; no loader
    consumes the result, and the numbers it reproduces are recorded in
    `DEPOSIT_INTEGRALS_UL` precisely so this function can be checked
    against them.
    """
    member_path = extract_cit168_member(CIT168_PROBABILISTIC_MEMBER)
    try:
        array, voxel_sizes = read_nifti(member_path)
    finally:
        os.remove(member_path)
        os.rmdir(os.path.dirname(member_path))
    if array.ndim != 4 or array.shape[3] != len(NUCLEI):
        raise ValueError(
            "expected a 4-D probabilistic atlas with one volume per "
            "nucleus (%d), got shape %s" % (len(NUCLEI), array.shape)
        )
    voxel_ul = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
    middle = array.shape[0] // 2
    integrals = {}
    for nucleus_id in range(1, len(NUCLEI) + 1):
        volume = array[..., nucleus_id - 1]
        integrals[nucleus_id] = (
            float(volume[:middle].sum()) * voxel_ul,
            float(volume[middle:].sum()) * voxel_ul,
        )
    return integrals


_TRUTTI_MANIFEST_KEYS = {
    "L": "trutti2021_vta_hem-l_proba27",
    "R": "trutti2021_vta_hem-r_proba27",
}


def trutti_mask_volumes(cache_dir=None, allow_download=False):
    """Verify the Trutti releases reproduce the paper's published volumes.

    Returns ``{hemisphere: {"nonzero_voxels": int, "nonzero_extent_mm3":
    float, "probability_integral_mm3": float}}``. The paper states the
    unthresholded atlas volume as 2226.375 mm^3 (left) and 2368.00 mm^3
    (right); the nonzero-extent numbers here must equal them exactly, and
    a test enforces that. The probability integrals are recorded for
    completeness -- they measure the group MAP, a different quantity from
    any individual's VTA, and are never presented as one.
    """
    results = {}
    for hemisphere, key in sorted(_TRUTTI_MANIFEST_KEYS.items()):
        path = ensure(key, cache_dir=cache_dir, allow_download=allow_download)
        array, voxel_sizes = read_nifti(path)
        voxel_mm3 = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
        nonzero = int(np.count_nonzero(array))
        results[hemisphere] = {
            "nonzero_voxels": nonzero,
            "nonzero_extent_mm3": nonzero * voxel_mm3,
            "probability_integral_mm3": float(array.sum()) * voxel_mm3,
        }
    return results
