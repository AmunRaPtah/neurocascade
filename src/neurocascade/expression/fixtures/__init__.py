"""Small committed fixtures: real Allen ISH expression data, eight genes of it.

Why the fixtures are real data and not a synthetic matrix
---------------------------------------------------------

Same reason as `connectome/fixtures/`: the tests have to run with no
network, the pinned queries return about 7.4 MB that must not enter git,
and a hand-written 3x3 matrix of invented numbers would test the array
plumbing and nothing that matters. It would not contain a gene whose four
experiments disagree, or an experiment that reports two thirds of the
structures the others do, or an indexed experiment with no unionize rows at
all - and every one of those is a real property of this data that the
loader has to survive.

So each fixture is a *slice of the checksum-verified pinned response,
written back out in that response's own format*: the same eleven-column
CSV header, the same `{"success": ..., "msg": [...]}` index envelope, the
same real numbers.

The ontology is the connectome's, deliberately
----------------------------------------------

The structures kept are exactly the 77 in
`connectome/fixtures/allen_structure_graph_1_fixture.json` (55 until P4.2
added the five autoradiography-comparison regions and one ancestor of them,
then 69 when P4.3 added the seven regions its cited expected distributions
name - CA1, CA3, DG, MH, PVT, SNc, CBX - and one ancestor, then 77 when
P6.1 added the five neuromodulatory source nuclei the projection systems
name - CS, NDB, PPN, LDT, TM - and three ancestors). There is no
expression-specific ontology fixture and there must not be one: P4.1's
whole claim is that expression and connectivity are indexed by the same CCF
structure graph, and two fixture ontologies free to drift apart would let
the tests keep passing after that stopped being true.

What each of the eight genes is here for
----------------------------------------

    Oprm1     the receptor the model already simulates; 3 experiments in
              both planes; its sagittal experiment 69860840 does not report
              DR, so the "absent row means NaN" path runs on real data
    Drd1      4 experiments, one of which (354) reports only 33 of the 61
              structures - the widest coverage gap in the fixture
    Drd2      the other receptor the model already simulates; striatal
    Gabra1    the third modelled system; cortex-dominant
    Htr2a     the only sagittal-ONLY gene in the fixture, so
              plane="coronal" produces an all-NaN row that has to be
              reported rather than read as "not expressed"
    Cnr1      its experiment 282 appears in the index and has no unionize
              rows in any structure graph, so the index-vs-data mismatch
              path runs offline
    Adora2a   striatal, and a P4.3 validation target
    Slc6a3    the dopamine transporter; confined to VTA/SNc, which is the
              sharpest "is this map anatomically real" check available

These fixtures are for tests and offline exploration. They are NOT a
substitute for the real files in any modelling result: 61 structures out of
the ~800 an experiment unionizes is a test input, not a receptor map.
"""

import os

FIXTURE_DIR = os.path.abspath(os.path.dirname(__file__))

#: The genes with a committed fixture. See the module docstring for why each.
FIXTURE_SYMBOLS = (
    "Oprm1",
    "Drd1",
    "Drd2",
    "Gabra1",
    "Htr2a",
    "Cnr1",
    "Adora2a",
    "Slc6a3",
)

#: The one sagittal-only gene in the fixture set.
FIXTURE_SAGITTAL_ONLY_SYMBOL = "Htr2a"

#: An indexed experiment with no unionize rows in any structure graph. Four
#: such experiments exist among the 106 in the full index (282 Cnr1, 359
#: Drd2, 526 Gria1, 534 Grin1, all from the original sagittal release);
#: nothing in their API metadata distinguishes them from the 102 that do
#: have unionizes. See `atlas.ExpressionAtlas.experiments_without_unionizes`.
FIXTURE_EXPERIMENT_WITHOUT_UNIONIZES = 282

INDEX_FIXTURE_NAME = "aba_ish_section_data_sets_fixture.json"


def unionize_fixture_name(symbol):
    return "aba_ish_unionize_%s_fixture.csv" % symbol


INDEX_FIXTURE_PATH = os.path.join(FIXTURE_DIR, INDEX_FIXTURE_NAME)
UNIONIZE_FIXTURE_PATHS = {
    symbol: os.path.join(FIXTURE_DIR, unionize_fixture_name(symbol))
    for symbol in FIXTURE_SYMBOLS
}

#: (n_rows, n_experiments) per fixture file, measured by
#: `scripts/make_expression_fixture.py` when it cut them. The manifest
#: records the same two numbers for the full files, and the loader checks
#: them for the same reason: a truncated read hashes consistently and would
#: otherwise pass every other check while silently dropping experiments.
FIXTURE_COUNTS = {
    "Oprm1": (230, 3),
    "Drd1": (268, 4),
    "Drd2": (231, 3),
    "Gabra1": (153, 2),
    "Htr2a": (77, 1),
    "Cnr1": (308, 4),
    "Adora2a": (231, 3),
    "Slc6a3": (307, 4),
}


def load_fixture_index():
    """The trimmed gene -> experiment index."""
    from ..unionize import read_dataset_index

    return read_dataset_index(INDEX_FIXTURE_PATH)


def load_fixture_atlas(symbols=None, ontology=None, index=None):
    """An `ExpressionAtlas` off the fixtures. No cache, no network.

    Shares the connectome's committed fixture ontology, which is the point:
    the offline tests exercise the same region alignment the real loader
    does, over the same structure ids.
    """
    from ... import connectome
    from ..atlas import ExpressionAtlas, GeneExpression, check_counts
    from ..genes import gene
    from ..unionize import read_unionize_csv
    from .. import sources

    if symbols is None:
        symbols = list(FIXTURE_SYMBOLS)
    if ontology is None:
        ontology = connectome.load_fixture_ontology()
    if index is None:
        index = load_fixture_index()

    loaded = {}
    for symbol in symbols:
        if symbol not in UNIONIZE_FIXTURE_PATHS:
            raise KeyError(
                "no expression fixture for %r; fixtures exist for %s"
                % (symbol, ", ".join(FIXTURE_SYMBOLS))
            )
        experiments = read_unionize_csv(
            UNIONIZE_FIXTURE_PATHS[symbol], index, expected_symbol=symbol
        )
        n_rows, n_datasets = FIXTURE_COUNTS[symbol]
        check_counts(unionize_fixture_name(symbol), n_rows, n_datasets, experiments)
        loaded[symbol] = GeneExpression(gene(symbol), experiments)

    return ExpressionAtlas(
        gene_expression=loaded,
        index=index,
        ontology=ontology,
        unavailable=sources.GENES_WITHOUT_DATA,
    )
