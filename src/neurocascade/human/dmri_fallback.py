"""ENIGMA/HCP dMRI as a provenance-flagged human fallback (P13.6)."""

from dataclasses import dataclass
import numpy as np

from .provenance import ABSENT, INFERRED, MEASURED, ProvenanceMap

TRACTOGRAPHY_LIMITATIONS = (
    "Maier-Hein et al. 2017, Nat Commun 8:1349, PMID 29116093: "
    "36 +/- 17% of reconstructed streamlines connect regions that are not "
    "connected; 88 +/- 58 invalid bundles per tractogram; bundle precision "
    "23 +/- 9%."
)


@dataclass(frozen=True)
class DMRIFallback:
    """Structural matrix whose directed interpretation is explicitly inferred."""

    connectome: object

    @property
    def labels(self):
        return tuple(self.connectome.labels)

    @property
    def provenance(self):
        return ProvenanceMap.from_structural(self.connectome)

    @property
    def limitations(self):
        return TRACTOGRAPHY_LIMITATIONS

    def require_inferred(self, source, target):
        """Allow use as a fallback while refusing measured-level claims."""
        return self.provenance.require(source, target, minimum=INFERRED)

    def require_measured(self, source, target):
        """Always refuse: this fallback cannot supply measured direction."""
        actual = self.provenance.flag(source, target)
        raise ValueError(
            "dMRI fallback edge %r -> %r is %s, never measured" %
            (source, target, actual)
        )

    def compare_with_ftract(self, feature):
        """Compare direction only where the F-TRACT labels overlap exactly.

        Returns counts, not a falsely precise accuracy: dMRI and F-TRACT use
        different parcellations in the current project, so non-identical label
        sets produce an explicit no-comparison result.
        """
        if tuple(feature.labels) != self.labels:
            return {"status": "not_comparable", "reason": "parcellation labels differ", "n_compared": 0}
        measured = np.isfinite(feature.matrix)
        structural = self.connectome.weights != 0.0
        comparable = measured & structural
        reciprocal = comparable & comparable.T & ~np.eye(len(self.labels), dtype=bool)
        return {
            "status": "compared",
            "n_compared": int(comparable.sum()),
            "n_reciprocal_pairs": int(reciprocal.sum() // 2),
            "note": "F-TRACT finite cells are measured influence; dMRI directions remain inferred.",
        }
