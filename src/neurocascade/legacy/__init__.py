"""Quarantined rodent-sourced material.

Everything in this package was built for the mouse-primary model of Phases
1 to 10. `docs/roadmap-human.md` reversed that scope decision: neurocascade is
human-primary, and rodent data that does not transfer to humans is retired.

The retirement policy, quoted from that roadmap:

    Rodent-only material moves to a `legacy/` namespace. It stays importable
    and stays tested; it is excluded from the prediction API and from any human
    scenario by an enforced species gate.

    Deletion was rejected because rodent data retains two uses that a
    human-primary model needs:

    1. Cross-species calibration anchors. Morphine Kp,uu 0.29 rat vs 0.64 human
       (Ederoth 2004, PMID 15025740) and TPA023 occupancy EC50 9 ng/mL human vs
       19-30 rat/rhesus/baboon (Atack 2010, PMID 19779131) are the only matched
       pairs that quantify the transfer error. They are evidence *about* the
       pivot.
    2. The searched-and-empty records. Five registries document what was looked
       for and not found. That is a real result and re-deriving it is expensive.

    What quarantine forbids is a rodent value reaching a human prediction. The
    species gate is a test, not a convention.

Nothing here is deleted, deprecated or broken. It is fully tested and it stays
that way: these modules remain the record of what the mouse build established,
and several of them hold the only quantitative statements this project has
about how badly a rodent number transfers.

What lives here
---------------
`connectome`
    Allen Mouse Brain Connectivity Atlas ingestion (P3.1): Knox 2019 voxel-model
    matrices, Oh 2014, the CCF structure graph, the ontology, and the
    transmitter sign rules keyed to CCF subtrees.
`expression`
    Allen ISH receptor-gene expression (P4.1 to P4.3): 43 mouse gene symbols,
    the unionize path, the expression-to-density proportionality, and the mouse
    autoradiography validation tables (Yoo 2010, Lothmann 2021).
`connectome_scenarios`
    P3.3's ports of the hand-built scenarios onto CCF regions.
`drug_pk`
    Rodent PK parameters (P1.2 to P1.4) and rodent brain partitions (P1.5), plus
    their searched-and-empty registries.
`pk_validation`
    P1.6's validation of the rodent PK curves against published rodent data.
`endogenous_tone`
    P6.2's five basal extracellular concentrations, all from rodent
    microdialysis, which cannot be performed in humans outside rare
    neurosurgical settings.
`projection_density`, `projection_scenarios`, `projection_systems`
    P6.1 and P6.3: projection densities derived from the mouse tracer matrix,
    and the nine source nuclei keyed to CCF acronyms and ids.
`behavioral_evidence`
    The rodent dose-to-behaviour bridge (P2.1 to P2.3, P7.3): all eleven
    published dose-to-behaviour datasets - ten rat, one mouse, zero human -
    the value types and delta-method ratio math around them, the per-endpoint
    coverage and searched-and-empty registries, and the derivation of the
    retired P2.2 anxiolysis target. Moved here by P11.3; it was the last
    rodent data production calibration read.
`rodent_series`
    P11.3 addition: decoder.py's five optogenetic/chemogenetic
    drive-response series (four mouse, one rat), with their citations. The
    `decoder` machinery that assessed them is species-agnostic and stayed in
    production with an empty library.

What deliberately did NOT move
------------------------------
The species-agnostic engine (`engine`, `network`, `circuit`, `behavior`,
`receptor`, `transport`, `pk`, `brain_partition`, `stability`, `projection`) is
mechanism with no species in it. The already-human material stayed too: the
three calibrated PK models in `scenarios.py` run on human clinical PK, and the
D2 and H1 receptor systems are human end to end.

See also
--------
`docs/roadmap-human.md`
    The five-layer audit, the phase plan, and the reasoning behind each
    retirement.
"""

from . import (
    behavioral_evidence,
    connectome,
    connectome_scenarios,
    drug_pk,
    endogenous_tone,
    expression,
    pk_validation,
    projection_density,
    projection_scenarios,
    projection_systems,
    rodent_series,
)

__all__ = [
    "behavioral_evidence",
    "connectome",
    "connectome_scenarios",
    "drug_pk",
    "endogenous_tone",
    "expression",
    "pk_validation",
    "projection_density",
    "projection_scenarios",
    "projection_systems",
    "rodent_series",
]
