"""P13.3: macaque directed priors never masquerade as measured human data."""

import pytest

from neurocascade.human.directed_prior import (
    DirectedPriorEdge,
    HOMOLOGY_LIMITATIONS,
    JIRSA_RETRY_STATUS,
    MacaqueDirectedPrior,
)
from neurocascade.human.provenance import ABSENT, INFERRED, MEASURED, ProvenanceMap


def test_prior_edges_are_macaque_and_inferred():
    edge = DirectedPriorEdge("a", "b", 0.5)
    assert edge.source_species == "macaque"
    assert edge.provenance == INFERRED
    with pytest.raises(ValueError, match="must be flagged inferred"):
        DirectedPriorEdge("a", "b", 0.5, provenance=MEASURED)


def test_prior_upgrades_only_absent_directions():
    labels = ["a", "b"]
    human = ProvenanceMap.from_levels(labels, [[ABSENT, MEASURED], [ABSENT, ABSENT]])
    prior = MacaqueDirectedPrior(labels, [DirectedPriorEdge("a", "b", 1.0), DirectedPriorEdge("b", "a", 1.0)], mapping_note="test blueprint")
    merged = prior.as_provenance(human)
    assert merged.flag("b", "a") == INFERRED
    assert merged.flag("a", "b") == MEASURED
    assert merged.counts()[INFERRED] == 1


def test_prior_rejects_parcellation_mismatch():
    prior = MacaqueDirectedPrior(["a", "b"], [], mapping_note="test")
    with pytest.raises(ValueError, match="labels do not match"):
        prior.as_provenance(ProvenanceMap.full(["a", "c"], ABSENT))


def test_homology_limitations_and_jirsa_retry_are_recorded():
    assert "lateral_temporal" in HOMOLOGY_LIMITATIONS["weakest_supported"]
    assert "long-range" in HOMOLOGY_LIMITATIONS["summary"]
    assert JIRSA_RETRY_STATUS["attempted"] is True
    assert JIRSA_RETRY_STATUS["outcome"] == "unavailable"
