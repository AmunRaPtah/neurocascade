"""Macaque directed connectivity prior for uncovered human edges (P13.3).

This module deliberately stores only a *prior*, never a measured human edge.
The Markov FLNe/SLN data are macaque tract-tracing observations transferred
through a stated homology relationship.  Consumers must combine the prior
with a human map; the resulting flags remain ``INFERRED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .provenance import ABSENT, INFERRED, MEASURED, ProvenanceMap

MARKOV_FLNE_CITATION = (
    "Markov NT et al. A weighted and directed interareal connectivity matrix "
    "for the macaque cerebral cortex. Cereb Cortex 2014;24(1):17-36. "
    "PMID 23010748. FLNe: 1,615 pathways; 29 of 91 areas injected."
)
MARKOV_SLN_CITATION = (
    "Markov NT et al. A weighted laminar connectivity matrix of macaque "
    "cerebral cortex. J Comp Neurol 2014;522(1):225-259. PMID 23983048."
)
MARS_BLUEPRINT_CITATION = (
    "Mars RB et al. Whole brain comparative anatomy using connectivity "
    "blueprints. eLife 2018;7:e35237. PMID 29749930."
)
REID_HOMOLOGY_CITATION = (
    "Reid AT et al. A cross-species comparison of structural and functional "
    "brain connectivity. NeuroImage 2016;125:311-331. PMID 26515902."
)
JIRSA_CITATION = (
    "Huang N, Wang HE, Triebkorn P, et al., Jirsa V. bioRxiv 2026-06-17. "
    "DOI 10.64898/2026.06.16.732559. Repository retry: unavailable in this run."
)

# This is a quality annotation, not a probability of biological equivalence.
WEAK_HOMOLOGY_REGIONS = (
    "lateral_temporal",
    "medial_temporal",
    "anterior_cingulate",
    "temporo_parietal_junction",
)
BEST_HOMOLOGY_CASES = ("local_ipsilateral", "homotopic")


@dataclass(frozen=True)
class DirectedPriorEdge:
    source: str
    target: str
    weight: float
    source_species: str = "macaque"
    provenance: str = INFERRED
    confidence: str = "unqualified"

    def __post_init__(self):
        if self.provenance != INFERRED:
            raise ValueError("macaque priors must be flagged inferred")
        if self.source_species != "macaque":
            raise ValueError("directed prior source_species must be macaque")
        if self.weight < 0.0:
            raise ValueError("prior weights must be non-negative")


class MacaqueDirectedPrior:
    """Sparse directed prior keyed by human labels after blueprint mapping."""

    def __init__(self, labels: Iterable[str], edges: Iterable[DirectedPriorEdge], *, mapping_note: str):
        self.labels = tuple(labels)
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels contain duplicates")
        allowed = set(self.labels)
        self.edges = tuple(edges)
        for edge in self.edges:
            if edge.source not in allowed or edge.target not in allowed:
                raise ValueError("prior edge labels must belong to the human label set")
            if edge.source == edge.target:
                raise ValueError("self-edges are not accepted in the directed prior")
        self.mapping_note = mapping_note

    @property
    def citations(self):
        return (MARKOV_FLNE_CITATION, MARKOV_SLN_CITATION, MARS_BLUEPRINT_CITATION, REID_HOMOLOGY_CITATION)

    def as_provenance(self, human_map: ProvenanceMap) -> ProvenanceMap:
        """Overlay covered human edges without upgrading measured direction.

        Only ABSENT human edges are upgraded to INFERRED. Existing MEASURED
        and INFERRED edges are retained, and label sets must match exactly.
        """
        if tuple(human_map.labels) != self.labels:
            raise ValueError("prior and human provenance labels do not match")
        flags = human_map.flags_matrix()
        index = {label: i for i, label in enumerate(self.labels)}
        for edge in self.edges:
            i, j = index[edge.source], index[edge.target]
            if flags[i, j] == ABSENT:
                flags[i, j] = INFERRED
        return ProvenanceMap.from_levels(self.labels, flags)

    def uncovered_edges(self, human_map: ProvenanceMap):
        """Return prior edges that target currently absent human directions."""
        if tuple(human_map.labels) != self.labels:
            raise ValueError("prior and human provenance labels do not match")
        return tuple(
            edge for edge in self.edges
            if human_map.flag(edge.source, edge.target) == ABSENT
        )


def prior_from_weight_mapping(labels: Iterable[str], weights: Mapping[tuple[str, str], float], *, mapping_note: str):
    """Build a prior from blueprint-transferred source/target weights."""
    return MacaqueDirectedPrior(
        labels,
        (DirectedPriorEdge(source, target, float(weight)) for (source, target), weight in weights.items()),
        mapping_note=mapping_note,
    )


HOMOLOGY_LIMITATIONS = {
    "best_supported": BEST_HOMOLOGY_CASES,
    "weakest_supported": WEAK_HOMOLOGY_REGIONS,
    "summary": (
        "Reid 2016 found generally poor cross-species correspondence: local "
        "ipsilateral and homotopic connections are the strongest cases, while "
        "long-range and heterotopic connections are weakest. Only 29 of 91 "
        "macaque areas were injected; human-expanded prefrontal and "
        "temporo-parietal cortex has no macaque counterpart."
    ),
}

JIRSA_RETRY_STATUS = {
    "attempted": True,
    "outcome": "unavailable",
    "citation": JIRSA_CITATION,
    "reason": "The preprint repository URL could not be verified during the build; no validation claim is made.",
}
