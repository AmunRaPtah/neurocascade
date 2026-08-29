"""Per-edge provenance flags for the human layer's directed connectivity.

Why this module exists
----------------------

The roadmap (``docs/roadmap-human.md``, P13.2) requires that every directed
edge the human layer uses carries one of three flags:

* ``MEASURED`` - the edge's existence *and direction* come from a measurement
  that resolves direction. The only current source is F-TRACT (CCEP
  responses to direct electrical stimulation in 780 epilepsy patients,
  `human/ftract.py`): if parcel ``i`` was stimulated and parcel ``j``
  responded, ``i -> j`` is a measured directed edge. A probed pair with no
  response (an exact zero) is still measured - the measurement happened;
  its result is "no detectable connection".
* ``INFERRED`` - the edge comes from a direction-blind measurement whose
  direction was assigned afterwards by convention. This is exactly the
  ENIGMA/HCP dMRI matrix (`human/matrix.py`) with `human/build.py`'s
  per-source transmitter sign: diffusion tractography cannot see which way
  a bundle runs, so each nonzero symmetric weight is entered once per
  direction and signed by the source region's cited transmitter identity.
  An inferred edge may describe anatomy that does not exist in that
  direction; it must never be reported as measured (enforced again in
  P13.6).
* ``ABSENT`` - no data covers this directed edge. F-TRACT NaN cells are
  pairs that were never both probed-and-responsive; dMRI zero cells carry
  no streamlines and contribute nothing to any built network.

The invariant this module enforces is the roadmap sentence: **no edge
without one**. A `ProvenanceMap` is constructed over a complete n x n grid
of ordered label pairs and refuses to hold anything else, so "what is the
provenance of edge i -> j?" always has an answer.

What this module deliberately does NOT do
-----------------------------------------

It does not merge parcellations. F-TRACT is HCP-MMP1 (360 cortical
parcels); the ENIGMA matrix is Desikan-Killiany plus subcortex (82
regions). `combine()` only accepts maps over identical label sets - a
measured MMP1 edge cannot be projected onto a DK node by assertion here.
Cross-parcellation transfer is P13.3's job, via macaque FLNe/SLN
blueprints validated against F-TRACT, not this module's.
"""

import numpy as np

#: Direction resolved by the measurement itself (F-TRACT CCEP).
MEASURED = "measured"

#: Direction-blind measurement, direction assigned by convention
#: (ENIGMA/HCP dMRI + per-source transmitter sign).
INFERRED = "inferred"

#: No data for this directed edge.
ABSENT = "absent"

#: All levels, weakest to strongest.
LEVELS = (ABSENT, INFERRED, MEASURED)

_LEVEL_RANK = {ABSENT: 0, INFERRED: 1, MEASURED: 2}


class ProvenanceError(ValueError):
    """A provenance requirement was not met."""


def _check_level(level):
    if level not in _LEVEL_RANK:
        raise ValueError(
            "level must be one of %s, got %r" % (", ".join(LEVELS), level)
        )
    return level


class ProvenanceMap(object):
    """One provenance flag per directed edge over a fixed label set.

    Constructed only through complete grids (`from_levels`, `full`, or the
    source-specific classmethods), so every ordered pair always carries
    exactly one of the three levels - the roadmap's "no edge without one".
    """

    def __init__(self, labels, flags):
        self.labels = list(labels)
        self._flags = np.array(flags, dtype=object)

        n = len(self.labels)
        if self._flags.shape != (n, n) or self._flags.ndim != 2:
            raise ValueError(
                "flags must be a %d x %d grid over labels, got shape %s"
                % (n, n, self._flags.shape)
            )
        unknown = sorted(
            {str(f) for f in self._flags.flat} - set(LEVELS)
        )
        if unknown:
            raise ProvenanceError(
                "flags contain values outside %s: %s"
                % (", ".join(LEVELS), ", ".join(repr(u) for u in unknown))
            )
        self._label_index = {label: i for i, label in enumerate(self.labels)}
        if len(self._label_index) != n:
            raise ValueError("labels contain duplicates")

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_levels(cls, labels, flags):
        """Wrap an already-complete grid of level strings."""
        return cls(labels, flags)

    @classmethod
    def full(cls, labels, level):
        """Every edge at one level (the starting point for `combine`)."""
        _check_level(level)
        n = len(labels)
        return cls(labels, np.full((n, n), level, dtype=object))

    @classmethod
    def from_ftract(cls, feature):
        """F-TRACT coverage as provenance.

        Finite cells (including exact zeros: probed, no response) are
        MEASURED - CCEP stimulation resolved that direction directly.
        NaN cells were never probed-and-responsive and are ABSENT.
        """
        flags = np.full(feature.matrix.shape, ABSENT, dtype=object)
        flags[np.isfinite(feature.matrix)] = MEASURED
        return cls(feature.labels, flags)

    @classmethod
    def from_structural(cls, connectome):
        """ENIGMA/HCP dMRI coverage as provenance.

        Every nonzero weight becomes INFERRED in BOTH directions - the
        matrix is symmetric, so neither direction was measured; each is a
        modelling convention layered on a direction-blind measurement.
        Zero weights are ABSENT (no streamlines, contributes nothing to
        any built network).
        """
        flags = np.full(connectome.weights.shape, ABSENT, dtype=object)
        flags[connectome.weights != 0.0] = INFERRED
        return cls(connectome.labels, flags)

    # -- lookups ------------------------------------------------------------

    @property
    def n(self):
        return len(self.labels)

    def index_of(self, label):
        try:
            return self._label_index[label]
        except KeyError:
            raise KeyError(
                "%r is not a label in this map (%d labels)" % (label, self.n)
            )

    def flag(self, source, target):
        """The provenance level of the directed edge source -> target."""
        return str(self._flags[self.index_of(source), self.index_of(target)])

    def flags_matrix(self):
        """A copy of the full n x n grid of level strings."""
        return self._flags.copy()

    def counts(self):
        """Edges per level, e.g. ``{"absent": 100, "inferred": 20, ...}``."""
        counts = dict.fromkeys(LEVELS, 0)
        for value in self._flags.flat:
            counts[str(value)] += 1
        return counts

    # -- enforcement --------------------------------------------------------

    def require(self, source, target, minimum=MEASURED):
        """Raise unless source -> target holds at least `minimum`.

        `minimum=MEASURED` (the default) refuses inferred edges - the
        guard every future consumer of directed human edges should call
        before trusting one. Ordering: absent < inferred < measured.
        """
        _check_level(minimum)
        actual = self.flag(source, target)
        if _LEVEL_RANK[actual] < _LEVEL_RANK[minimum]:
            raise ProvenanceError(
                "edge %r -> %r has provenance %r, which is below the "
                "required minimum %r" % (source, target, actual, minimum)
            )
        return actual

    def combine(self, other):
        """Elementwise strongest level of two maps over identical labels.

        MEASURED beats INFERRED beats ABSENT. Refuses different label sets:
        cross-parcellation transfer is P13.3's job, not an implicit
        relabelling here.
        """
        if self.labels != list(other.labels):
            raise ValueError(
                "cannot combine provenance maps over different label sets "
                "(%d vs %d labels)" % (self.n, other.n)
            )
        theirs = other.flags_matrix()
        merged = self._flags.copy()
        their_rank = np.vectorize(lambda v: _LEVEL_RANK[str(v)])(theirs)
        my_rank = np.vectorize(lambda v: _LEVEL_RANK[str(v)])(merged)
        upgrade = their_rank > my_rank
        merged[upgrade] = theirs[upgrade]
        return ProvenanceMap(self.labels, merged)

    def summary(self):
        counts = self.counts()
        total = self.n * self.n
        lines = [
            "edge provenance: %d directed pairs over %d labels" % (total, self.n),
            "  %-9s %6d (%5.1f%%)"
            % (MEASURED, counts[MEASURED], 100.0 * counts[MEASURED] / total),
            "  %-9s %6d (%5.1f%%)"
            % (INFERRED, counts[INFERRED], 100.0 * counts[INFERRED] / total),
            "  %-9s %6d (%5.1f%%)"
            % (ABSENT, counts[ABSENT], 100.0 * counts[ABSENT] / total),
        ]
        return "\n".join(lines)

    def __repr__(self):
        counts = self.counts()
        return "ProvenanceMap(%d labels; %d measured, %d inferred, %d absent)" % (
            self.n,
            counts[MEASURED],
            counts[INFERRED],
            counts[ABSENT],
        )
