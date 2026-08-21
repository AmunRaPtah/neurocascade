"""The undirected, unsigned human structural connectivity matrix.

`HumanStructuralConnectome` holds the ENIGMA Toolbox group-average
Desikan-Killiany matrix plus its labels and atlas. It is the human
analogue of `connectome/matrix.py`'s `RegionalConnectome`, with the two
differences that are properties of the measurement rather than of this
code:

* **undirected** - diffusion tractography is symmetric by construction,
  so there is no source/target distinction and no hemisphere block. The
  single matrix is both `weights` and `weights.T`.
* **unsigned** - dMRI carries no excitatory/inhibitory sign, so there is
  no `neurotransmitter.py` here. Sign is a modelling assumption applied
  later, in `build.py`, and it is labelled as such.

The loader therefore has no orientation conversion: `projection_matrix()`
returns the same numbers it loaded, because for a symmetric matrix the
`CircuitNetwork` convention (`connectivity[i, j]` = from `j` onto `i`) and
the file's row/column convention are the same matrix.
"""

import csv

import numpy as np

from . import sources
from .atlas import atlas_from_labels


def _read_labels(labels_path):
    """Read the single comma-separated label row, with strict shape checks."""
    with open(labels_path, newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 1:
        raise ValueError(
            "%s: expected one comma-separated label row, got %d rows"
            % (labels_path, len(rows))
        )
    labels = [cell for cell in rows[0] if cell != ""]
    if not labels:
        raise ValueError("%s: empty label row" % labels_path)
    return labels


class HumanStructuralConnectome(object):
    """One published human structural matrix, with its provenance."""

    def __init__(self, weights, labels, atlas, dataset_key, citation, units):
        self.weights = np.asarray(weights, dtype=float)
        self.labels = list(labels)
        self.atlas = atlas
        self.dataset_key = dataset_key
        self.citation = citation
        self.units = units

        n = self.weights.shape[0]
        if self.weights.ndim != 2 or self.weights.shape != (n, n):
            raise ValueError("weights must be a square matrix")
        if len(self.labels) != n:
            raise ValueError(
                "labels count %d != matrix size %d" % (len(self.labels), n)
            )
        if len(self.atlas) != n:
            raise ValueError("atlas size %d != matrix size %d" % (len(self.atlas), n))
        if self.labels != self.atlas.labels:
            raise ValueError("labels do not match the atlas region order")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("%s contains non-finite weights" % dataset_key)
        if np.any(self.weights < 0.0):
            raise ValueError(
                "%s contains negative weights; streamline counts are "
                "non-negative by construction, so this is a parse failure"
                % dataset_key
            )
        if not np.array_equal(self.weights, self.weights.T):
            raise ValueError(
                "%s is not symmetric; diffusion tractography is undirected, "
                "so a non-symmetric file is a parse failure, not a directed "
                "measurement" % dataset_key
            )

    @property
    def n(self):
        return self.weights.shape[0]

    def __repr__(self):
        return "HumanStructuralConnectome(%r, %d regions)" % (
            self.dataset_key,
            self.n,
        )

    def projection_matrix(self):
        """`(matrix, labels)`, ready for `CircuitNetwork.connectivity`.

        For a symmetric (undirected) matrix there is nothing to transpose:
        `matrix[i, j]` is both the published entry and the weight of the
        coupling from region `j` onto region `i`, because they are the
        same number. Returns copies so callers cannot mutate the loaded
        data.
        """
        return self.weights.copy(), list(self.labels)


def read_connectome(matrix_path, labels_path, dataset_key=None):
    """Read a manifest file pair from explicit paths, no cache, no network."""
    if dataset_key is None:
        dataset_key = sources.DEFAULT_MATRIX_KEY
    matrix_entry = sources.data_file(dataset_key)

    labels = _read_labels(labels_path)
    weights = np.loadtxt(matrix_path, delimiter=",", dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(
            "%s: expected a square matrix, got shape %s" % (matrix_path, weights.shape)
        )
    if weights.shape[0] != len(labels):
        raise ValueError(
            "%s: %d labels but %d x %d matrix"
            % (labels_path, len(labels), weights.shape[0], weights.shape[1])
        )

    return HumanStructuralConnectome(
        weights=weights,
        labels=labels,
        atlas=atlas_from_labels(labels),
        dataset_key=dataset_key,
        citation=matrix_entry.citation,
        units=matrix_entry.units,
    )


def load_connectome(cache_dir=None, allow_download=False):
    """Load the default human structural matrix plus its labels.

    `allow_download` is False by default so that neither a test nor an
    ordinary import can start a download; `scripts/fetch_human_connectome.py`
    passes True.
    """
    from . import fetch

    matrix_path = fetch.ensure(
        sources.DEFAULT_MATRIX_KEY,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    labels_path = fetch.ensure(
        sources.LABELS_KEY,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    return read_connectome(matrix_path, labels_path)
