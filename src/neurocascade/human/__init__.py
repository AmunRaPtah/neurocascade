"""The human data layer.

This package is the first deliberate move toward a human-first build. The
existing `neurocascade.connectome` package ingests the Allen MOUSE
connectome and CCF ontology; everything in `human/` ingests HUMAN data and
turns it into the same species-neutral `CircuitNetwork` the engine already
integrates. The physics in `circuit.py`, `network.py`, `receptor.py`,
`pk.py` and `engine.py` is unchanged and species-neutral: only the data
layer differs.

Read `sources.py` first: it is the manifest (URLs pinned to a commit,
checksums, citations, and what the matrix's numbers actually mean), and it
is the only place a number about the external data is written down.
`neurotransmitter.py` is the cited per-region sign table that turns the
unsigned dMRI matrix into a signed `CircuitNetwork` (glutamatergic cortex /
hippocampus / thalamus, GABAergic striatum / pallidum, mixed amygdala).

Offline by default
------------------

Same contract as `neurocascade.connectome`: nothing touches the network at
import time, every loader defaults to `allow_download=False`, and a cache
miss raises `MissingCachedFile` telling you to run
`scripts/fetch_human_connectome.py`. Tests run entirely off the small
committed fixtures in `fixtures/`, which are real slices of the verified
files.

    from neurocascade import human

    conn = human.load_connectome()          # verified cache
    built = human.build_circuit_network(conn)
    print(built.report.summary())

What has been built here
------------------------

This track ingests one real, citable human structural connectivity matrix -
the ENIGMA Toolbox's group-average Desikan-Killiany (68 cortical + 14
subcortical) diffusion-MRI tractography matrix - and builds a signed
`CircuitNetwork` from it. The sign is assigned per source region from cited
transmitter identity (`neurotransmitter.py`), the human analogue of
`connectome/neurotransmitter.py`. Three honest caveats are front-and-center
rather than hidden:

* dMRI tractography is UNDIRECTED and UNSIGNED. The mouse matrices are
  directed by tracer injection; the human matrix is symmetric and carries
  no direction, so each symmetric weight is entered twice and signed once
  per direction (`build.py` states this convention out loud).
* The sign is a per-source projection-neuron transmitter identity, not a
  measurement: cortex / hippocampus / thalamus are glutamatergic, striatum
  / pallidum are GABAergic, and the coarse whole-amygdala region is MIXED
  and left unsigned (excluded, or a hard error with
  `unsigned_sources="raise"`).
* The Desikan-Killiany atlas is a cortex-centred parcellation; the 14
  subcortical regions are the coarse classic nuclei, not the fine
  subdivisions the mouse CCF provides.
"""

from . import atlas, build, expression, fetch, fixtures, maps, matrix, neurotransmitter, scenario, sources
from .atlas import HumanAtlas, Region, atlas_from_labels, parse_label
from .build import (
    BuildReport,
    HumanCircuit,
    build_circuit_network,
)
from .maps import (
    MISSING_POLICIES,
    HumanCoupling,
    unmeasured_regions,
    whole_brain_coupling,
    whole_brain_nodes,
)
from .scenario import (
    MECHANISM_GAIN_NOTE,
    HumanScenario,
    HumanScenarioReport,
    whole_brain_simulation,
)
from .neurotransmitter import (
    CITATIONS,
    RegionSignRule,
    SIGN_RULES,
    SignAssignment,
    SignRules,
)
from .fetch import (
    DEFAULT_CACHE_DIR,
    ChecksumMismatch,
    MissingCachedFile,
    cache_path,
    checksum_lines,
    download,
    ensure,
    fetch_all,
    is_cached,
    sha256_of,
    verify,
)
from .matrix import (
    HumanStructuralConnectome,
    load_connectome,
    read_connectome,
)
from .expression import (
    ABSENT,
    ASSUMPTION,
    HumanGeneExpression,
    HumanReceptorDensityMap,
    HumanReceptorGene,
    HUMAN_RECEPTOR_GENES,
    NORMALIZATIONS,
    REGIONS_WITHOUT_SAMPLES,
    STABLE_SYMBOLS,
    UNTHRESHOLDED_ONLY,
    availability,
    load_expression,
    read_expression,
    relative_density,
)

__all__ = [
    "atlas",
    "build",
    "expression",
    "fetch",
    "fixtures",
    "maps",
    "matrix",
    "neurotransmitter",
    "scenario",
    "sources",
    "CITATIONS",
    "RegionSignRule",
    "SIGN_RULES",
    "SignAssignment",
    "SignRules",
    "HumanAtlas",
    "Region",
    "atlas_from_labels",
    "parse_label",
    "BuildReport",
    "HumanCircuit",
    "build_circuit_network",
    "MISSING_POLICIES",
    "HumanCoupling",
    "unmeasured_regions",
    "whole_brain_coupling",
    "whole_brain_nodes",
    "MECHANISM_GAIN_NOTE",
    "HumanScenario",
    "HumanScenarioReport",
    "whole_brain_simulation",
    "DEFAULT_CACHE_DIR",
    "ChecksumMismatch",
    "MissingCachedFile",
    "cache_path",
    "checksum_lines",
    "download",
    "ensure",
    "fetch_all",
    "is_cached",
    "sha256_of",
    "verify",
    "HumanStructuralConnectome",
    "load_connectome",
    "read_connectome",
    "ABSENT",
    "ASSUMPTION",
    "HumanGeneExpression",
    "HumanReceptorDensityMap",
    "HumanReceptorGene",
    "HUMAN_RECEPTOR_GENES",
    "NORMALIZATIONS",
    "REGIONS_WITHOUT_SAMPLES",
    "STABLE_SYMBOLS",
    "UNTHRESHOLDED_ONLY",
    "availability",
    "load_expression",
    "read_expression",
    "relative_density",
]
