"""F-TRACT: measured directed connectivity between human cortical parcels.

Why this module exists
----------------------

Every other connectivity input to the human layer is *direction-blind*.
Diffusion MRI cannot see which way a bundle runs (``matrix.py`` refuses a
non-symmetric file rather than pretend otherwise), and the ENIGMA matrix's
direction is assigned afterwards from transmitter identity -- an
assumption layered on a symmetric measurement, flagged as such in
``build.py``.  Phase 13.6 demotes that matrix to a provenance-flagged
fallback.

F-TRACT is different: it measures direction *causally* by stimulating one
cortical site and recording evoked potentials at another.  780 epilepsy
patients, 34,354 stimulations, 774,445 CCEPs (Lemarechal et al. 2022).
Rows in the exported CSVs are the *stimulated* parcel (source) and
columns are the *recorded* parcel (target).  This is stated by the file
itself, not inferred.

Direction semantics
-------------------

* A finite value at ``M[i, j]`` means ``i`` (row) was stimulated and a
  response was measured at ``j`` (column).
* ``NaN`` means the ordered pair was never probed; zero and NaN are
  distinct states (28 measured cells are exactly 0.0 in probability).
* A reciprocal pair (both ``M[i, j]`` and ``M[j, i]`` finite) with
  different values is *asymmetric* -- the defining biological property
  of CCEP direction.  78.2% of reciprocal probability pairs are
  asymmetric.
* ``np.array_equal(M, M.T)`` is ``False`` for every feature.

How these data were sampled
---------------------------

Four acquisition properties bound what any model may claim from these
matrices: clinically driven (hence biased) electrode coverage, a patient
cohort that is not neurotypical, responses that may be polysynaptic, and a
group average with no per-edge dispersion.  They are enumerated with their
consequences in ``SAMPLING_LIMITATIONS`` and pinned by
``tests/test_human_ftract.py``.

What this module deliberately does NOT do
-----------------------------------------

* **No sign.**  CCEP probability is unsigned.  Excitatory vs inhibitory
  classification is a separate modelling choice (P13.4 uses F-TRACT's
  synaptic time constants, not this module, to assign E/I).
* **No coupling to the circuit layer.**  The 360-parcel HCP-MMP1
  parcellation is *not* the Desikan-Killiany atlas used by the PET and
  ENIGMA layers.  An explicit parcellation crosswalk is needed before
  F-TRACT edges can feed a ``CircuitNetwork`` -- Phase 13's problem,
  not this module's.

Licence decision
----------------

The Zenodo record (record 7015415, the actual bytes served) states in its
own description: *"The atlas is provided for free use for research use
only"*.  The Zenodo metadata page labels the record as CC BY 4.0.
The EBRAINS landing page (DOI 10.25493/5AM4-J3F) lists CC BY-NC-SA 4.0.

The deposit's own words take priority over a secondary description: this
module is bound to the research-use-only condition from the Zenodo record
description.  For a research project, both licence readings are
satisfied.  Both the discrepancy and the chosen source are recorded here
rather than smoothed over.

Data files
----------

``f-tract_v2112.zip`` (Zenodo record 7015415, version 21-12, 2021-12-09):
915 MB, containing CCEP-derived connectivity matrices across 11
parcellations, 2 age groups, and 19 features.  This module extracts only
the MNI-HCP-MMP1 adult cohort (ages 15-100) subset.  The three ingested
features:

* ``probability`` -- mean CCEP detection probability (float64, [0, 1]
  with NaN = never probed, 0.0 = probed and no response).
* ``amplitude`` -- median response amplitude (units undeclared by the
  release).
* ``axonal_conduction_delay`` -- median delay from stimulation to first
  response (units undeclared; fed to Wilson-Cowan propagation delays
  in P13.4).

Two additional features are loaded as one-dimensional parcel maps (not
full matrices) for P13.4:

* ``synaptic_excitatory_delay`` -- per-parcel excitatory synaptic time
  constant (units undeclared; stored in the release as tab-separated
  ``.txt``, NOT ``.csv``).
* ``synaptic_inhibitory_delay`` -- per-parcel inhibitory synaptic time
  constant.

The original P13.1 recon recorded these as "empty directory entries, no
data."  That is wrong for the v21-12 release as distributed: the ``.txt``
files are present and contain 360 parcel entries each, with NaN where
unmeasured.

Negative control: TVB matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three TVB ``connectivity_76/96/192`` archives are pinned as
verification-only manifest entries.  They are *not* F-TRACT data and
their rejection must be tested.  Signatures live in the committed fixture
``tvb_signatures_fixture.json``.

Offline by default
------------------

Same contract as the rest of ``human/``: nothing touches the network at
import time, loaders default to ``allow_download=False``, a cache miss
raises ``MissingCachedFile`` naming ``scripts/fetch_human_ftract.py``.
The inner member extraction also verifies the per-file sha256.
"""

import csv
import gzip
import hashlib
import json
import os
import sys
import zipfile

import numpy as np

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401


def _sha256_bytes(data):
    """Compute sha256 of raw bytes without writing to disk."""
    return hashlib.sha256(data).hexdigest()

# --- citations ---------------------------------------------------------------

LEMARECHAL_CITATION = (
    "Lemarechal J-D, Jedynak M, Trebaul L, Boyer A, Deman P, Tuyisenge V, "
    "Ayoubian L, et al. 'A brain atlas of axonal and synaptic delays based "
    "on modelling of cortico-cortical evoked potentials.' Brain "
    "2022;145(5):1653-1667. PMID 35416942. doi:10.1093/brain/awab362. "
    "780 patients, 34,354 stimulations, 774,445 CCEPs on HCP-MMP1 (360 "
    "parcels). EBRAINS DOI 10.25493/5AM4-J3F."
)

DAVID2013_CITATION = (
    "David O, Bastin J, Chabardes S, Minotti L, Kahane P. 'Probabilistic "
    "functional tractography of the human cortex.' NeuroImage "
    "2013;80:307-317. PMID 23632076. doi:10.1016/j.neuroimage.2013.05.075."
)

TREBAUL2018_CITATION = (
    "Trebaul L, Deman P, Tuyisenge V, Jedynak M, Hugues E, Boyer A, et al. "
    "'Probabilistic functional tractography of the human cortex revisited.' "
    "NeuroImage 2018;181:818-827. PMID 30036722. "
    "doi:10.1016/j.neuroimage.2018.07.039."
)

# --- licence decision --------------------------------------------------------

LICENCE_NOTE = (
    "The Zenodo record description states 'provided for free use for "
    "research use only.'  The Zenodo metadata page labels the record CC BY "
    "4.0; the EBRAINS landing page lists CC BY-NC-SA 4.0.  The deposit's "
    "own description is the binding statement.  This module is used under "
    "that research-use-only condition.  Both the discrepancy and the chosen "
    "source are recorded here."
)

ZENODO_CITATION = (
    "Source record: Zenodo record 7015415, version 21-12, published "
    "2021-12-09. " + LICENCE_NOTE
)

# --- sampling limitations ----------------------------------------------------
# Four properties of how these data were acquired, each with the consequence
# for a model that consumes them. They are recorded HERE, next to the
# matrices, rather than in prose elsewhere: a directed human edge is the
# scarcest thing in this project, and the temptation to read it as more than
# it is is correspondingly large. Every one of them is a property of the
# acquisition, not of this ingest, and none of them is fixable downstream.

SAMPLING_LIMITATIONS = (
    {
        "limitation": "Electrode placement is clinically driven, not "
                      "anatomically sampled: implants target suspected "
                      "epileptogenic cortex, which biases coverage toward "
                      "temporal and frontal regions.",
        "consequence": "Parcel-pair coverage is non-uniform and non-random. "
                       "A NaN cell means 'never probed', so absence of an "
                       "edge is NOT evidence of absence of a connection, and "
                       "COVERAGE's finite-cell counts must never be read as a "
                       "connection density.",
    },
    {
        "limitation": "The 780 participants are not neurotypical: all have "
                      "drug-resistant focal epilepsy and are recorded under "
                      "antiseizure medication, some within epileptogenic "
                      "tissue.",
        "consequence": "Probabilities and delays are measured in an altered "
                       "brain. Applying them to a model of a healthy human "
                       "brain is a transfer assumption, of the same kind the "
                       "species gate refuses across species, and it should be "
                       "stated wherever a prediction depends on it.",
    },
    {
        "limitation": "A CCEP response can be polysynaptic: a response "
                      "recorded at j after stimulating i may have travelled "
                      "through one or more relays.",
        "consequence": "A finite M[i, j] is a directed INFLUENCE, not a "
                       "monosynaptic projection, and axonal_conduction_delay "
                       "may sum several relays. Do not equate an F-TRACT edge "
                       "with an anatomical fibre.",
    },
    {
        "limitation": "The release is a group average only: no subject-level "
                      "matrices and no per-edge dispersion are published.",
        "consequence": "There is no uncertainty attached to any edge and no "
                       "route to individual-level prediction. A model built "
                       "on it predicts a cohort average, and any per-edge "
                       "error bar would have to be invented.",
    },
)

#: The same four, as one paragraph, for reports and figure captions.
SAMPLING_LIMITATIONS_NOTE = " ".join(
    "%s %s" % (item["limitation"], item["consequence"])
    for item in SAMPLING_LIMITATIONS
)

ALL_CITATIONS = " ".join([LEMARECHAL_CITATION, DAVID2013_CITATION, TREBAUL2018_CITATION, ZENODO_CITATION])

# --- manifest ----------------------------------------------------------------

MANIFEST_VERSION = "2026-08-25"

FETCH_COMMAND = "scripts/fetch_human_ftract.py"

_RETRIEVED = "2026-08-25"

DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

UNITS_NOTE = (
    "Units are NOT declared by the F-TRACT release for any feature.  The "
    "CSVs report only 'float64'.  Probability is bounded [0, 1] by "
    "definition; amplitude and delay values are assumed to be in SI-like "
    "units but the release provides no label.  Do not assume milliseconds "
    "without an independent source."
)


class DataFile(datafetch.DataFile):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        datafetch.DataFile.__init__(self, *args, **kwargs)


# Outer bundle: the full Zenodo zip.
# sha256 is locally computed (Zenodo publishes only md5); publisher md5
# is df03571e8fce5b030482f239dd0d04ab and is verified after download.
_BUNDLE_KEY = "ftract_v2112_bundle"

_BUNDLE = DataFile(
    key=_BUNDLE_KEY,
    filename="f-tract_v2112.zip",
    url="https://zenodo.org/records/7015415/files/f-tract_v2112.zip?download=1",
    sha256="9cea8ec2edb4803a7568cf507c1d6f7b208680733beac977f36280a5a6294c46",
    size_bytes=915288088,
    citation=ALL_CITATIONS,
    description=(
        "F-TRACT atlas release v21-12 (Zenodo record 7015415). The full "
        "915 MB archive containing CCEP-derived connectivity matrices "
        "across 11 parcellations, 2 age groups, and 19 features. This "
        "module extracts only the MNI-HCP-MMP1 adult cohort subset. "
        "Publisher md5: df03571e8fce5b030482f239dd0d04ab. sha256 above "
        "is locally computed and is not published by Zenodo."
    ),
    units=UNITS_NOTE,
)

# TVB negative-control archives (verification only, not consumed by any loader).
_TVB_DATA = [
    (
        "tvb_connectivity_76_zip",
        "connectivity_76.zip",
        "https://raw.githubusercontent.com/the-virtual-brain/tvb-data/master/tvb_data/connectivity/connectivity_76.zip",
        "8b856d5fa80a8593e01dc18b95efff829ca98d2daf43cea38fd3309fed726b2e",
        44190,
    ),
    (
        "tvb_connectivity_96_zip",
        "connectivity_96.zip",
        "https://raw.githubusercontent.com/the-virtual-brain/tvb-data/master/tvb_data/connectivity/connectivity_96.zip",
        "5f6d8b1512c7fb2c6ee50f214b0dd2d1b898625d2acacb9ffe7c3c2422b55cf7",
        52641,
    ),
    (
        "tvb_connectivity_192_zip",
        "connectivity_192.zip",
        "https://raw.githubusercontent.com/the-virtual-brain/tvb-data/master/tvb_data/connectivity/connectivity_192.zip",
        "d9626a029da18108c3a34cf717d52a665033f227810d71a0ed732c19e2cfd93d",
        61137,
    ),
]

_TVB_CITATION = (
    "TVB connectivity matrices: GPL v3 (github.com/the-virtual-brain/tvb-data). "
    "Not F-TRACT data; pinned here for negative-control rejection testing."
)

_TVB_ENTRIES = []
for _key, _fname, _url, _sha, _sz in _TVB_DATA:
    _TVB_ENTRIES.append(
        DataFile(
            key=_key,
            filename=_fname,
            url=_url,
            sha256=_sha,
            size_bytes=_sz,
            citation=_TVB_CITATION,
            description=(
                "TVB %s: a Virtual Brain connectivity archive, pinned for "
                "negative-control rejection testing. NOT F-TRACT data." % _fname
            ),
            units='arbitrary units (weights_unit="au" in info.txt)',
        )
    )

DATA_FILES = [_BUNDLE] + _TVB_ENTRIES
DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

# --- zip member paths and digests --------------------------------------------

#: The subdirectory inside the zip that holds the HCP-MMP1 adult cohort.
_ZIP_BASE = "ages_15_100/sr_8.40/seg_None_None/pl_200/MNI-HCP-MMP1"

#: Inner members of interest, with sha256 and byte-size measured from the
#: verified extraction.  The key is the feature name used as a directory
#: name inside the ``export/`` tree.  Values are (member_path, sha256, size).
FTRACT_ZIP_MEMBERS = {
    "probability": (
        _ZIP_BASE + "/export/probability/probability.csv",
        "4671eb8d7d0adb2b4755d415250675e2e8e6826e9b87463a56115caa1dfc97fc",
        708172,
    ),
    "amplitude": (
        _ZIP_BASE + "/export/amplitude/amplitude.csv",
        "71659a6486ac20c5aced3271913806315be36cd0df2aea2f476a11f2008d6b0d",
        749370,
    ),
    "axonal_conduction_delay": (
        _ZIP_BASE + "/export/axonal_conduction_delay/axonal_conduction_delay.csv",
        "3b9b09573e32b27a73ef832f0df26b98a87df06bd90e81cb467847900dcd4867",
        604868,
    ),
    "synaptic_excitatory_delay": (
        _ZIP_BASE + "/export/synaptic_excitatory_delay/synaptic_excitatory_delay.txt",
        "885e4d57bcb97435f0d360f88a89c005a8d4eac9099413d42bd3c80f26e1ab99",
        5806,
    ),
    "synaptic_inhibitory_delay": (
        _ZIP_BASE + "/export/synaptic_inhibitory_delay/synaptic_inhibitory_delay.txt",
        "3f8f79f9aeb4eaccc65b1dc3b38432c83b0fcadf1e4a5604afa33007d2010c07",
        5823,
    ),
}

#: Which features are full 360x360 matrices vs one-dimensional parcel maps.
MATRIX_FEATURES = ("probability", "amplitude", "axonal_conduction_delay")
DELAY_MAP_FEATURES = ("synaptic_excitatory_delay", "synaptic_inhibitory_delay")

#: Metadata about the ingested matrix features.
FEATURE_META = {
    "probability": {
        "statistic": "mean",
        "units": "CCEP detection probability [0, 1]; NaN = never probed",
    },
    "amplitude": {
        "statistic": "median",
        "units": "response amplitude (units not declared by the release)",
    },
    "axonal_conduction_delay": {
        "statistic": "median",
        "units": "conduction delay (units not declared by the release)",
    },
}

# --- measured-direction constants --------------------------------------------

ROWS_ARE_STIMULATED_NOTE = (
    "In every F-TRACT CSV, rows are the STIMULATED parcel (source) and "
    "columns are the RECORDED parcel (target).  This is stated by the "
    "file header ('stimulated parcels are listed vertically') and "
    "confirmed by the 78.2% asymmetry in reciprocal probability pairs: "
    "the direction is real, not an artefact of row/column labelling."
)

# --- pinned coverage and asymmetry numbers ------------------------------------
# Measured from the full adult-cohort MNI-HCP-MMP1 CSVs, NaN-aware.
# Tested against the cached full CSV when present; fixture tests cover
# the parsing logic only.

COVERAGE = {
    "probability": {
        "n_parcels": 360,
        "n_finite": 27066,
        "n_diagonal": 295,
        "n_reciprocal_pairs": 10789,
        "n_asymmetric_pairs": 8439,
        "asymmetry_percent": 78.2,
        "n_one_way": 5193,
        "symmetric_array_equal": False,
    },
    "amplitude": {
        "n_parcels": 360,
        "n_finite": 16050,
        "n_diagonal": 278,
        "n_reciprocal_pairs": 6176,
        "n_asymmetric_pairs": 6176,
        "asymmetry_percent": 100.0,
        "n_one_way": 3420,
        "symmetric_array_equal": False,
    },
    "axonal_conduction_delay": {
        "n_parcels": 360,
        "n_finite": 16050,
        "n_diagonal": 278,
        "n_reciprocal_pairs": 6176,
        "n_asymmetric_pairs": 6175,
        "asymmetry_percent": 100.0,
        "n_one_way": 3420,
        "symmetric_array_equal": False,
    },
}

#: Exactly 28 off-diagonal cells are 0.0 (probed, no response) in
#: probability; zero and NaN are distinct states.
N_EXACT_ZEROS_OFF_DIAGONAL = 28

# --- Fetcher -----------------------------------------------------------------

_FETCHER = datafetch.Fetcher(
    manifest=sys.modules[__name__],
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/human",
)


def data_file(key):
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


# --- extraction --------------------------------------------------------------

def _extracted_dir(cache_dir=None):
    """Return (and create) the directory where extracted members live."""
    base = os.path.join(cache_dir or DEFAULT_CACHE_DIR, "ftract", "extracted")
    os.makedirs(base, exist_ok=True)
    return base


def extract_member(feature_key, cache_dir=None, allow_download=True):
    """Extract and verify one inner member from the F-TRACT bundle.

    Returns the absolute path to the extracted file.  Verifies the
    per-member sha256 against ``FTRACT_ZIP_MEMBERS``.
    """
    if feature_key not in FTRACT_ZIP_MEMBERS:
        raise KeyError(
            "unknown F-TRACT feature %r; known: %s"
            % (feature_key, ", ".join(sorted(FTRACT_ZIP_MEMBERS)))
        )

    member_path, expected_sha, expected_size = FTRACT_ZIP_MEMBERS[feature_key]
    out_dir = os.path.join(_extracted_dir(cache_dir), feature_key)
    member_basename = os.path.basename(member_path)
    out_path = os.path.join(out_dir, member_basename)

    if os.path.exists(out_path):
        actual_size = os.path.getsize(out_path)
        if actual_size != expected_size:
            raise ChecksumMismatch(
                "%s: cached size %d != expected %d; re-extract"
                % (out_path, actual_size, expected_size)
            )
        actual_sha = sha256_of(out_path)
        if actual_sha == expected_sha:
            return out_path
        raise ChecksumMismatch(
            "%s: cached sha256 %s != expected %s; re-extract"
            % (out_path, actual_sha, expected_sha)
        )

    bundle_path = ensure(_BUNDLE_KEY, cache_dir=cache_dir, allow_download=allow_download)
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        with zf.open(member_path) as src:
            data = src.read()

    if len(data) != expected_size:
        raise ChecksumMismatch(
            "extracted %s: size %d != expected %d"
            % (member_path, len(data), expected_size)
        )
    actual_sha = _sha256_bytes(data)
    if actual_sha != expected_sha:
        raise ChecksumMismatch(
            "extracted %s: sha256 %s != expected %s"
            % (member_path, actual_sha, expected_sha)
        )

    with open(out_path, "wb") as dst:
        dst.write(data)

    return out_path


# --- CSV parsing -------------------------------------------------------------

_CSV_HEADER_LINES = 4  # comment lines before the header row


def _read_feature_csv(path):
    """Parse an F-TRACT feature CSV into (labels, matrix).

    Returns:
        labels: list of 360 parcel names (Glasser order, 180 L then 180 R).
        matrix: np.ndarray of shape (360, 360) with float64 values and NaN
                where the ordered pair was never probed.

    Raises ValueError on format violations.
    """
    with open(path, newline="") as fh:
        lines = fh.read().splitlines()

    if len(lines) < _CSV_HEADER_LINES + 2:
        raise ValueError(
            "%s: expected at least %d lines (comments + header + 1 data row), got %d"
            % (path, _CSV_HEADER_LINES + 2, len(lines))
        )

    # Verify comment structure.
    for i in range(_CSV_HEADER_LINES):
        if not lines[i].startswith("#"):
            raise ValueError(
                "%s: line %d expected to start with '#', got: %r"
                % (path, i, lines[i][:60])
            )

    header_line = lines[_CSV_HEADER_LINES]
    if not header_line.startswith("stimulated parcels,"):
        raise ValueError(
            "%s: header line expected to start with 'stimulated parcels,'; got: %r"
            % (path, header_line[:60])
        )

    labels = [cell for cell in header_line.split(",")[1:] if cell]
    n = len(labels)
    if n == 0:
        raise ValueError("%s: no column labels found" % path)

    row_labels = []
    rows = []
    for idx, line in enumerate(lines[_CSV_HEADER_LINES + 1 :], start=_CSV_HEADER_LINES + 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != n + 1:
            raise ValueError(
                "%s: line %d: expected %d comma-separated fields (1 label + %d values), got %d"
                % (path, idx, n + 1, n, len(parts))
            )
        row_label = parts[0]
        row_labels.append(row_label)
        rows.append(
            np.array([float(v) if v != "nan" else np.nan for v in parts[1:]], dtype=np.float64)
        )

    if len(row_labels) != n:
        raise ValueError(
            "%s: expected %d data rows (matching column count), got %d"
            % (path, n, len(row_labels))
        )
    if row_labels != labels:
        raise ValueError(
            "%s: row labels do not match column labels (first mismatch at index %d)"
            % (path, next(i for i, (a, b) in enumerate(zip(row_labels, labels)) if a != b))
        )

    matrix = np.array(rows, dtype=np.float64)
    return labels, matrix


def _read_synaptic_delay_map(path):
    """Parse a tab-separated F-TRACT synaptic delay file.

    Returns dict mapping parcel label -> float or NaN.
    """
    result = {}
    with open(path, newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    "%s: expected 2 tab-separated fields, got %d in: %r"
                    % (path, len(parts), line[:80])
                )
            label, val = parts
            try:
                result[label] = float(val)
            except ValueError:
                if val.lower() == "nan":
                    result[label] = float("nan")
                else:
                    raise ValueError(
                        "%s: cannot parse %r as float for parcel %r"
                        % (path, val, label)
                    )
    return result


# --- feature class -----------------------------------------------------------

class FTractFeature(object):
    """One F-TRACT feature: a 360x360 measured-direction matrix.

    Carries its name, the statistic used to aggregate across patients
    (mean or median), the parcel labels, and the matrix.  Provides
    measured/asymmetric coverage statistics and individual-cell lookups.
    """

    def __init__(self, name, labels, matrix):
        if name not in FEATURE_META:
            raise ValueError("unknown feature %r" % name)
        self.name = name
        self._meta = FEATURE_META[name]
        self.labels = list(labels)
        self.matrix = np.asarray(matrix, dtype=np.float64)

        n = self.matrix.shape[0]
        if self.matrix.ndim != 2 or self.matrix.shape != (n, n):
            raise ValueError("matrix must be square; got shape %s" % (self.matrix.shape,))
        if len(self.labels) != n:
            raise ValueError("labels count %d != matrix size %d" % (len(self.labels), n))

        self._label_index = {l: i for i, l in enumerate(self.labels)}

    @property
    def n(self):
        return self.matrix.shape[0]

    @property
    def statistic(self):
        return self._meta["statistic"]

    @property
    def units(self):
        return self._meta["units"]

    def __repr__(self):
        return "FTractFeature(%r, %d parcels, statistic=%r)" % (
            self.name, self.n, self.statistic,
        )

    def value(self, source, target):
        """The measured value for source -> target, or NaN if unprobed."""
        i = self._label_index[source]
        j = self._label_index[target]
        return self.matrix[i, j]
    def measured(self, source, target):
        """True if the ordered pair was probed (finite value)."""
        return np.isfinite(self.value(source, target))

    def coverage(self):
        """Coverage and asymmetry statistics as a dict."""
        M = self.matrix
        fin = np.isfinite(M)
        diag_mask = np.eye(self.n, dtype=bool)
        off_diag = ~diag_mask
        recip = fin & fin.T
        n_recip_off = int(np.sum(recip & off_diag)) // 2
        asym_off = recip & (M != M.T) & off_diag
        n_asym = int(np.sum(asym_off)) // 2
        oneway = fin & ~fin.T & off_diag
        return {
            "n_parcels": self.n,
            "n_finite": int(np.sum(fin)),
            "n_diagonal": int(np.sum(fin & diag_mask)),
            "n_reciprocal_pairs": n_recip_off,
            "n_asymmetric_pairs": n_asym,
            "asymmetry_percent": 100.0 * n_asym / max(n_recip_off, 1),
            "n_one_way": int(np.sum(oneway)),
            "symmetric_array_equal": bool(np.array_equal(M, M.T)),
        }

    def n_exact_zeros_off_diagonal(self):
        """Cells that are exactly 0.0 (probed, no response) off-diagonal."""
        off_diag = ~np.eye(self.n, dtype=bool)
        return int(np.sum((self.matrix == 0.0) & off_diag))

    def provenance(self):
        """This feature's coverage as a per-edge `ProvenanceMap`.

        Finite cells (exact zeros included: probed, no response) are
        MEASURED - CCEP stimulation resolved those directions directly;
        NaN cells are ABSENT. See `human/provenance.py`.
        """
        from . import provenance

        return provenance.ProvenanceMap.from_ftract(self)


# --- public loaders ----------------------------------------------------------

def load_feature(name, cache_dir=None, allow_download=False):
    """Load one matrix feature from the extracted F-TRACT bundle.

    Returns an ``FTractFeature``.  Offline by default.
    """
    path = extract_member(name, cache_dir=cache_dir, allow_download=allow_download)
    labels, matrix = _read_feature_csv(path)
    return FTractFeature(name, labels, matrix)


def load_synaptic_delays(feature_key, cache_dir=None, allow_download=False):
    """Load a synaptic delay parcel map.

    Returns a dict mapping parcel label -> float or NaN.
    """
    path = extract_member(feature_key, cache_dir=cache_dir, allow_download=allow_download)
    return _read_synaptic_delay_map(path)


# --- TVB rejection helpers ---------------------------------------------------

def _load_tvb_signatures():
    """Load the committed TVB signatures fixture."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "tvb_signatures_fixture.json"
    )
    with open(fixture_path) as fh:
        return json.load(fh)


def tvb_rejection_reasons(tvb_name):
    """Why a TVB matrix is not F-TRACT data, given its measured signatures.

    ``tvb_name`` is e.g. ``"76"``, ``"96"``, ``"192"``.
    Returns a list of strings explaining each rejection criterion.
    """
    sigs = _load_tvb_signatures()
    sig = sigs[tvb_name]
    reasons = []
    if sig["shape"] != [360, 360]:
        reasons.append("shape is (%d, %d), not (360, 360)" % tuple(sig["shape"]))
    if not sig["labels_glasser"]:
        reasons.append("labels are not HCP-MMP1 Glasser order")
    if sig["ordinal"]:
        reasons.append("weights are ordinal (unique values: %s)" % sig["unique_values"])
    if sig["weights_unit"] == "au":
        reasons.append("weights_unit is 'au' (arbitrary units), not a probability or physiological measurement")
    if sig["label_prefixes"]:
        reasons.append("labels carry CoCoMac-style prefixes: %s" % ", ".join(sig["label_prefixes"][:4]))
    if not reasons:
        reasons.append("matrix has no F-TRACT-signature features (shape, labels, or units match)")
    return reasons
