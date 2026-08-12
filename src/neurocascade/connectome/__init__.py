"""P3.1: the whole-brain connectome backbone.

This package ingests two published mouse connectome matrices and the Allen
CCF structure ontology, and hands back a directed weighted matrix whose
rows and columns are named brain regions with a place in the anatomical
hierarchy. It is the data layer P3.2 builds a `CircuitNetwork` out of.

Read `sources.py` first: it is the manifest (URLs, checksums, citations,
and what each matrix's numbers actually mean), and it is the only place a
number about the external data is written down.

Offline by default
------------------

Nothing in this package touches the network at import time, and every
loader defaults to `allow_download=False`. A cache miss raises
`fetch.MissingCachedFile` telling you to run
`scripts/fetch_connectome.py`; it never silently starts a download. Tests
run entirely off the small committed fixtures in `fixtures/`, which are
real subsets of the real files (see `fixtures/PROVENANCE.md`).

    from neurocascade import connectome

    ontology = connectome.load_ontology()          # verified cache
    matrix = connectome.load_connectome(ontology=ontology)
    ids, W = matrix.square_ipsilateral()           # W[i, j]: j -> i
"""

from .sources import (
    DATA_FILES,
    DATA_FILES_BY_KEY,
    DEFAULT_CONNECTIVITY_KEY,
    MANIFEST_VERSION,
    ONTOLOGY_KEY,
    DataFile,
    connectivity_keys,
    data_file,
)
from .fetch import (
    DEFAULT_CACHE_DIR,
    ChecksumMismatch,
    MissingCachedFile,
    cache_path,
    checksum_lines,
    fetch_all,
    is_cached,
    sha256_of,
    verify,
)
from .ontology import (
    MAJOR_DIVISION_IDS,
    Structure,
    StructureOntology,
    load_ontology,
)
from .matrix import (
    HEMISPHERES,
    RegionalConnectome,
    load_connectome,
    read_connectome,
)
from .volumes import (
    source_voxel_counts,
    target_voxel_counts,
)
from .neurotransmitter import (
    CITATIONS,
    SIGN_RULES,
    TRANSMITTER_SIGN,
    SignAssignment,
    SignRules,
    TransmitterRule,
)
from .build import (
    DEFAULT_NODE,
    RESOLUTIONS,
    BuildReport,
    ConnectomeCircuit,
    build_circuit_network,
    load_knox_inputs,
    parcellation,
)
from .fixtures import (
    EXTRA_ONTOLOGY_REGION_IDS,
    FIXTURE_CONNECTOME_KEY,
    FIXTURE_DIR,
    FIXTURE_REGION_IDS,
    KNOX_FIXTURE_KEYS,
    load_fixture_build_inputs,
    load_fixture_connectome,
    load_fixture_ontology,
)

__all__ = [
    "CITATIONS",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CONNECTIVITY_KEY",
    "DEFAULT_NODE",
    "MAJOR_DIVISION_IDS",
    "MANIFEST_VERSION",
    "ONTOLOGY_KEY",
    "HEMISPHERES",
    "RESOLUTIONS",
    "SIGN_RULES",
    "TRANSMITTER_SIGN",
    "BuildReport",
    "ChecksumMismatch",
    "ConnectomeCircuit",
    "DataFile",
    "MissingCachedFile",
    "RegionalConnectome",
    "SignAssignment",
    "SignRules",
    "Structure",
    "StructureOntology",
    "TransmitterRule",
    "FIXTURE_CONNECTOME_KEY",
    "FIXTURE_DIR",
    "EXTRA_ONTOLOGY_REGION_IDS",
    "FIXTURE_REGION_IDS",
    "KNOX_FIXTURE_KEYS",
    "build_circuit_network",
    "cache_path",
    "checksum_lines",
    "connectivity_keys",
    "data_file",
    "fetch_all",
    "is_cached",
    "load_connectome",
    "load_fixture_build_inputs",
    "load_fixture_connectome",
    "load_fixture_ontology",
    "load_knox_inputs",
    "load_ontology",
    "parcellation",
    "read_connectome",
    "sha256_of",
    "source_voxel_counts",
    "target_voxel_counts",
    "verify",
]
