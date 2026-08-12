"""P4.1: whole-brain receptor-gene expression, over the connectome's ontology.

This package ingests Allen Mouse Brain Atlas in-situ hybridization
expression energy, per gene per CCF region, for every receptor and
transporter gene this project models or will model, and hands back a
(gene x region) matrix whose columns line up one-for-one with the P3.1
connectome's regions.

Read `sources.py` first: it is the manifest (query URLs, checksums,
citations, and what the numbers mean), the record of what was searched for
and not found, and the only place a number about the external data is
written down. Read `genes.py` for the gene list and why each gene is on it.

Offline by default
------------------

Nothing here touches the network at import time, and every loader defaults
to `allow_download=False`. A cache miss raises `fetch.MissingCachedFile`
telling you to run `scripts/fetch_expression.py`; it never silently starts
a download. Tests run entirely off the small committed fixtures in
`fixtures/`, which are real slices of the verified responses and which share
the connectome's committed fixture ontology.

    from neurocascade import connectome, expression

    ontology = connectome.load_ontology()
    matrix   = connectome.load_connectome(ontology=ontology)
    atlas    = expression.load_expression_atlas(ontology=ontology)

    energy = atlas.for_connectome(
        matrix, plane="coronal-else-sagittal", combine="mean"
    )
    energy.values      # (42 genes, 291 regions), NaN where not measured
    energy.for_gene("Oprm1")

`plane` and `combine` are required, not optional. A gene has several ISH
experiments in two sectioning geometries that disagree by up to ~30-fold in
small structures, so picking among them is a modelling decision this
package refuses to make silently; `atlas.py`'s docstring says why at
length, and P4.2 owns the choice.

What this package does NOT do
-----------------------------

It does not convert expression energy into a receptor density, a Bmax or a
concentration. Expression energy is a transcript-abundance measure in
arbitrary units; mRNA is not protein and protein is not binding sites. That
conversion is P4.2, and P4.2 owns stating its proportionality assumption as
an assumption.
"""

from .genes import (
    GENES_BY_SYMBOL,
    RECEPTOR_GENES,
    SYSTEMS,
    ReceptorGene,
    gene,
    genes_for_system,
    symbols,
)
from .sources import (
    ALLEN_ISH_CITATION,
    AVAILABLE_SYMBOLS,
    DATA_FILES,
    DATA_FILES_BY_KEY,
    DATASET_INDEX_KEY,
    EXPRESSION_ENERGY_UNITS,
    GENES_WITHOUT_DATA,
    GENES_WITHOUT_DATA_BY_SYMBOL,
    MANIFEST_VERSION,
    MISSING_FROM_ISH,
    SAGITTAL_ONLY_SYMBOLS,
    UnionizeFile,
    data_file,
    unionize_file,
    unionize_key,
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
from .unionize import (
    PLANES,
    REFERENCE_SPACE_BY_PLANE,
    UNIONIZE_COLUMNS,
    ExperimentUnionizes,
    SectionDataSet,
    SectionDataSetIndex,
    read_dataset_index,
    read_unionize_csv,
)
from .atlas import (
    COMBINE_RULES,
    PLANE_POLICIES,
    ExpressionAtlas,
    GeneExpression,
    RegionalExpression,
    check_counts,
    load_expression_atlas,
)
from .fixtures import (
    FIXTURE_COUNTS,
    FIXTURE_DIR,
    FIXTURE_SAGITTAL_ONLY_SYMBOL,
    FIXTURE_SYMBOLS,
    load_fixture_atlas,
    load_fixture_index,
)

__all__ = [
    "ALLEN_ISH_CITATION",
    "AVAILABLE_SYMBOLS",
    "COMBINE_RULES",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DATASET_INDEX_KEY",
    "DEFAULT_CACHE_DIR",
    "EXPRESSION_ENERGY_UNITS",
    "FIXTURE_COUNTS",
    "FIXTURE_DIR",
    "FIXTURE_SAGITTAL_ONLY_SYMBOL",
    "FIXTURE_SYMBOLS",
    "GENES_BY_SYMBOL",
    "GENES_WITHOUT_DATA",
    "GENES_WITHOUT_DATA_BY_SYMBOL",
    "MANIFEST_VERSION",
    "MISSING_FROM_ISH",
    "PLANES",
    "PLANE_POLICIES",
    "RECEPTOR_GENES",
    "REFERENCE_SPACE_BY_PLANE",
    "SAGITTAL_ONLY_SYMBOLS",
    "SYSTEMS",
    "UNIONIZE_COLUMNS",
    "ChecksumMismatch",
    "ExperimentUnionizes",
    "ExpressionAtlas",
    "GeneExpression",
    "MissingCachedFile",
    "ReceptorGene",
    "RegionalExpression",
    "SectionDataSet",
    "SectionDataSetIndex",
    "UnionizeFile",
    "cache_path",
    "check_counts",
    "checksum_lines",
    "data_file",
    "fetch_all",
    "gene",
    "genes_for_system",
    "is_cached",
    "load_expression_atlas",
    "load_fixture_atlas",
    "load_fixture_index",
    "read_dataset_index",
    "read_unionize_csv",
    "sha256_of",
    "symbols",
    "unionize_file",
    "unionize_key",
    "verify",
]
