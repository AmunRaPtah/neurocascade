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

Energy is not density: what P4.2 added
--------------------------------------

`atlas.py` returns expression energy, a transcript-abundance measure in
arbitrary units. `density.py` converts it to the relative-density scale the
receptor layer consumes, under one assumption stated as an assumption -
density is proportional to transcript, within a gene - and three modules
exist to keep that honest:

    density.py          the conversion, the assumption, and the
                        density-times-mechanism-gain factorization
    mismatch.py         the genes whose receptor protein is not in the
                        region that transcribes it, with the error measured
    autoradiography.py  published mouse receptor densities to test against
    validation.py       the test, reported as three numbers because rank
                        agreement and proportionality are different claims

    rho = expression.relative_density(energy, "Drd1", normalization="max")
    rho.as_coupling(mechanism_gain, {"striatum": 672})

Measured, not assumed: over ten mouse regions with published D1 densities
the map ranks regions well (Spearman 0.71) and scales them badly
(proportional R2 0.31). `results/expression_density_validation.txt` is the
committed record.

Is it the map of the gene it claims to be: what P4.3 added
----------------------------------------------------------

A transposed matrix, a gene column off by one and a structure-id mis-join
all produce a map with the right shape and plausible numbers. Anatomy is
what tells them apart:

    anatomy.py        eighteen cited ordinal expectations over six
                      receptors, each with species, method and modality,
                      including two that are expected to FAIL and say why
    scenario_maps.py  the join from a gene plus {node: structure id} to the
                      density factor a scenario's coupling needs

    results = expression.check_atlas(atlas)          # every claim, measured
    scenarios.mu_opioid_multi_region_coupling()      # the first consumer

Seventeen of the eighteen behaved as recorded; the one that did not is a
0.6% difference between two regions at the floor of the Oprm1 map.
`results/receptor_map_validation.txt` is the committed record.
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
from .density import (
    ASSUMPTION,
    NORMALIZATIONS,
    PROPORTIONALITY_EXPONENT,
    AbsoluteScale,
    ReceptorDensityMap,
    relative_density,
)
from .mismatch import (
    MISMATCHES,
    MISMATCHES_BY_SYMBOL,
    SEVERITIES,
    TERMINAL_FIELD_SYMBOLS,
    Displacement,
    TerminalFieldMismatch,
    measured_displacement,
    mismatch_for,
)
from .autoradiography import (
    BACKGROUNDS,
    DATASETS,
    LOTHMANN2021,
    PRIMARY_BACKGROUND,
    REGION_MAPPING,
    SEARCHED_AND_EMPTY,
    YOO2010,
    AutoradiographyDataset,
    LigandBinding,
    comparable_lothmann_receptors,
    lothmann2021,
    lothmann2021_receptors,
    yoo2010_d1,
    yoo2010_d2,
    yoo2010_dat,
)
from .validation import (
    DensityComparison,
    compare,
    compare_lothmann,
    dropped_regions,
    fit_absolute_scale,
    policy_sensitivity,
)
from .anatomy import (
    ANATOMY_SOURCES,
    CLAIMS,
    CLAIMS_BY_SYMBOL,
    VALIDATED_SYMBOLS,
    AnatomySource,
    Claim,
    ClaimResult,
    check,
    check_atlas,
    claim_regions,
)
from .scenario_maps import (
    scenario_coupling,
    scenario_density_map,
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
    "ANATOMY_SOURCES",
    "AnatomySource",
    "CLAIMS",
    "CLAIMS_BY_SYMBOL",
    "Claim",
    "ClaimResult",
    "VALIDATED_SYMBOLS",
    "ASSUMPTION",
    "AVAILABLE_SYMBOLS",
    "AbsoluteScale",
    "AutoradiographyDataset",
    "BACKGROUNDS",
    "COMBINE_RULES",
    "ChecksumMismatch",
    "DATASETS",
    "DATASET_INDEX_KEY",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DEFAULT_CACHE_DIR",
    "DensityComparison",
    "Displacement",
    "EXPRESSION_ENERGY_UNITS",
    "ExperimentUnionizes",
    "ExpressionAtlas",
    "FIXTURE_COUNTS",
    "FIXTURE_DIR",
    "FIXTURE_SAGITTAL_ONLY_SYMBOL",
    "FIXTURE_SYMBOLS",
    "GENES_BY_SYMBOL",
    "GENES_WITHOUT_DATA",
    "GENES_WITHOUT_DATA_BY_SYMBOL",
    "GeneExpression",
    "LOTHMANN2021",
    "LigandBinding",
    "MANIFEST_VERSION",
    "MISMATCHES",
    "MISMATCHES_BY_SYMBOL",
    "MISSING_FROM_ISH",
    "MissingCachedFile",
    "NORMALIZATIONS",
    "PLANES",
    "PLANE_POLICIES",
    "PRIMARY_BACKGROUND",
    "PROPORTIONALITY_EXPONENT",
    "RECEPTOR_GENES",
    "REFERENCE_SPACE_BY_PLANE",
    "REGION_MAPPING",
    "SEARCHED_AND_EMPTY",
    "ReceptorDensityMap",
    "ReceptorGene",
    "RegionalExpression",
    "SAGITTAL_ONLY_SYMBOLS",
    "SEVERITIES",
    "SYSTEMS",
    "SectionDataSet",
    "SectionDataSetIndex",
    "TERMINAL_FIELD_SYMBOLS",
    "TerminalFieldMismatch",
    "UNIONIZE_COLUMNS",
    "UnionizeFile",
    "YOO2010",
    "cache_path",
    "check_counts",
    "checksum_lines",
    "comparable_lothmann_receptors",
    "compare",
    "compare_lothmann",
    "data_file",
    "dropped_regions",
    "check",
    "check_atlas",
    "claim_regions",
    "fetch_all",
    "fit_absolute_scale",
    "gene",
    "genes_for_system",
    "is_cached",
    "load_expression_atlas",
    "load_fixture_atlas",
    "load_fixture_index",
    "lothmann2021",
    "lothmann2021_receptors",
    "measured_displacement",
    "mismatch_for",
    "policy_sensitivity",
    "read_dataset_index",
    "read_unionize_csv",
    "relative_density",
    "scenario_coupling",
    "scenario_density_map",
    "sha256_of",
    "symbols",
    "unionize_file",
    "unionize_key",
    "verify",
    "yoo2010_d1",
    "yoo2010_d2",
    "yoo2010_dat",
]
