"""P4.1: the per-gene, per-region expression loader.

What a loaded object is
-----------------------

`ExpressionAtlas` holds, for every receptor gene in `genes.RECEPTOR_GENES`
that has Allen ISH data, every experiment's structure unionizes, keyed by
CCF graph-1 structure id - the same ids the P3.1 connectome uses - plus the
connectome's own `StructureOntology` for naming them. It also holds the
searched-and-empty record for the genes that have no data at all, because
that is a result and not an absence.

The two decisions this module refuses to make for you
-----------------------------------------------------

P3.1's `RegionalConnectome` stores published numbers unmodified and offers
`without_self_connections()` as an explicit method rather than a default,
because every convenience a loader might add is a modelling decision. The
same pressure exists here and it is sharper, because a gene does not have
*one* expression map:

**Which plane.** Allen's sagittal pipeline sections and processes the left
hemisphere only; coronal experiments cover the whole brain. Measured on
this data: a coronal experiment carries unionizes for 279 of the
connectome's 291 source regions, its sagittal siblings for 262 and 274.
Three of the 42 genes with data (Adra2c, Adrb2, Htr2a) are sagittal-only,
so a coronal-only atlas silently loses them unless somebody notices.

**Which experiment, or what to do with several.** Oprm1 has three
experiments, Chrm4 five. In the ventral tuberomammillary nucleus the three
Oprm1 experiments report expression energies of 0.285, 0.027 and 0.789.
Averaging them is defensible; so is taking the coronal one alone; so is
treating the spread as the measurement uncertainty it evidently is. Which
of those is right is a P4.2 question about what the receptor layer needs,
and a loader that quietly picked one would make the choice invisible.

So `energy_matrix()` takes `plane` and `combine` as **required** keyword
arguments. There is no default, on purpose. Every returned
`RegionalExpression` records which policy produced it and how many
experiments actually contributed to each cell.

NaN is not zero
---------------

A structure with no unionize row in a contributing experiment is NaN, never
0.0. Twelve connectome regions have no ISH unionize in any experiment (see
`sources.MISSING_FROM_ISH`), and individual experiments miss a few more.
Zero in a receptor map is the claim "a drug acting here has no target in
this region"; NaN is the truth, "not measured". This is the same refusal
P3.1 made for Oh et al. 2014's `#NUM!` cells, and `RegionalExpression`
reports the NaNs rather than leaving a caller to discover them.
"""

import numpy as np

from . import genes as _genes
from . import sources
from .unionize import read_dataset_index, read_unionize_csv

#: How to choose which experiments of a gene contribute.
#:
#:   "coronal"             coronal experiments only; a sagittal-only gene
#:                         yields an all-NaN row, and is named in the report
#:   "sagittal"            sagittal experiments only
#:   "any"                 every experiment, both planes pooled
#:   "coronal-else-sagittal"  per gene: coronal if it has any, else sagittal.
#:                         Never pools the two planes for one gene, so a
#:                         gene's map comes from one sectioning geometry.
PLANE_POLICIES = ("coronal", "sagittal", "any", "coronal-else-sagittal")

#: How to reduce several experiments to one number per region.
#:
#:   "mean"/"median"/"max" over the experiments that report the region;
#:                         an experiment that does not report it does not
#:                         drag the value toward zero, it simply does not vote
#:   "single"              require exactly one experiment after the plane
#:                         filter, and raise otherwise. The strict option:
#:                         no averaging of anything happens at all.
COMBINE_RULES = ("mean", "median", "max", "single")


class GeneExpression(object):
    """One gene's experiments, unreduced."""

    __slots__ = ("gene", "experiments")

    def __init__(self, gene, experiments):
        self.gene = gene
        self.experiments = tuple(experiments)
        for experiment in self.experiments:
            if experiment.dataset.symbol != gene.symbol:
                raise ValueError(
                    "experiment %d belongs to %s, not %s"
                    % (
                        experiment.dataset.id,
                        experiment.dataset.symbol,
                        gene.symbol,
                    )
                )

    def __repr__(self):
        return "GeneExpression(%r, %d experiments: %s)" % (
            self.gene.symbol,
            len(self.experiments),
            ", ".join(sorted(self.planes())),
        )

    def planes(self):
        return sorted(set(e.dataset.plane for e in self.experiments))

    def dataset_ids(self):
        return [e.dataset.id for e in self.experiments]

    def select(self, plane):
        """The experiments a plane policy admits, in file order."""
        if plane not in PLANE_POLICIES:
            raise ValueError(
                "plane must be one of %s, got %r" % (list(PLANE_POLICIES), plane)
            )
        if plane == "any":
            return list(self.experiments)
        if plane == "coronal-else-sagittal":
            coronal = [e for e in self.experiments if e.dataset.plane == "coronal"]
            if coronal:
                return coronal
            return [e for e in self.experiments if e.dataset.plane == "sagittal"]
        return [e for e in self.experiments if e.dataset.plane == plane]


class RegionalExpression(object):
    """A (gene x region) expression-energy matrix and how it was made.

    `values[g, r]` is the expression energy of `symbols[g]` in
    `structure_ids[r]`, or NaN if no admitted experiment reports that
    region. `n_experiments[g, r]` is how many experiments actually
    contributed to that cell, which is the difference between "one
    measurement" and "the mean of five" and is not recoverable from the
    values alone.
    """

    def __init__(
        self,
        values,
        n_experiments,
        symbols,
        structure_ids,
        plane,
        combine,
        dataset_ids,
        units,
        citation,
        ontology=None,
    ):
        self.values = np.asarray(values, dtype=float)
        self.n_experiments = np.asarray(n_experiments, dtype=int)
        self.symbols = tuple(symbols)
        self.structure_ids = np.asarray(structure_ids, dtype=int)
        self.plane = plane
        self.combine = combine
        self.dataset_ids = dict(dataset_ids)
        self.units = units
        self.citation = citation
        self.ontology = ontology

        shape = (len(self.symbols), self.structure_ids.size)
        if self.values.shape != shape:
            raise ValueError(
                "values has shape %s but there are %d genes and %d regions"
                % (self.values.shape, shape[0], shape[1])
            )
        if self.n_experiments.shape != shape:
            raise ValueError("n_experiments must have the same shape as values")
        finite = np.isfinite(self.values)
        if np.any(self.values[finite] < 0.0):
            raise ValueError(
                "negative expression energy; the published quantity is "
                "non-negative by construction, so this is a bug"
            )
        if np.any(np.isinf(self.values)):
            raise ValueError("infinite expression energy; this is a bug")
        # The invariant that keeps "not measured" from becoming "zero".
        if np.any(finite & (self.n_experiments == 0)):
            raise ValueError("a value with no contributing experiment is not a value")
        if np.any(~finite & (self.n_experiments > 0)):
            raise ValueError("a NaN with contributing experiments is a bug")

    def __repr__(self):
        return "RegionalExpression(%d genes x %d regions, plane=%r, combine=%r)" % (
            len(self.symbols),
            self.structure_ids.size,
            self.plane,
            self.combine,
        )

    @property
    def shape(self):
        return self.values.shape

    def for_gene(self, symbol):
        """One gene's row."""
        try:
            row = self.symbols.index(symbol)
        except ValueError:
            raise KeyError(
                "%r is not in this matrix; it has %s"
                % (symbol, ", ".join(self.symbols))
            )
        return self.values[row]

    def acronyms(self):
        """Region acronyms, from the connectome ontology."""
        if self.ontology is None:
            raise ValueError(
                "this matrix was built without an ontology, so region names "
                "are not available"
            )
        return [self.ontology[i].acronym for i in self.structure_ids]

    def genes_without_values(self):
        """Genes whose whole row is NaN under this plane policy.

        Under `plane="coronal"` this is exactly the sagittal-only genes, and
        it is the reason that policy is not a silent default.
        """
        return [
            symbol
            for symbol, row in zip(self.symbols, self.values)
            if not np.any(np.isfinite(row))
        ]

    def regions_without_values(self):
        """Structure ids no admitted experiment reports, for any gene."""
        column_ok = np.any(np.isfinite(self.values), axis=0)
        return [int(i) for i, ok in zip(self.structure_ids, column_ok) if not ok]

    def coverage(self):
        """Fraction of (gene, region) cells that carry a measurement."""
        if self.values.size == 0:
            return 0.0
        return float(np.isfinite(self.values).sum()) / self.values.size


class ExpressionAtlas(object):
    """Every ingested gene's ISH expression, over the connectome's ontology."""

    def __init__(self, gene_expression, index, ontology, unavailable=()):
        self.genes = dict(gene_expression)
        self.index = index
        self.ontology = ontology
        self.unavailable = tuple(unavailable)

    def __repr__(self):
        return "ExpressionAtlas(%d genes, %d without data, %d experiments)" % (
            len(self.genes),
            len(self.unavailable),
            sum(len(g.experiments) for g in self.genes.values()),
        )

    def __contains__(self, symbol):
        return symbol in self.genes

    def __getitem__(self, symbol):
        try:
            return self.genes[symbol]
        except KeyError:
            if symbol in sources.GENES_WITHOUT_DATA_BY_SYMBOL:
                entry = sources.GENES_WITHOUT_DATA_BY_SYMBOL[symbol]
                raise KeyError(
                    "%s has no Allen ISH expression data at all. %s %s"
                    % (symbol, entry["searched"], entry["conclusion"])
                )
            raise KeyError(
                "%r was not loaded; this atlas has %s"
                % (symbol, ", ".join(sorted(self.genes)))
            )

    def symbols(self):
        """Loaded gene symbols, in `genes.RECEPTOR_GENES` order."""
        return [g.symbol for g in _genes.RECEPTOR_GENES if g.symbol in self.genes]

    def missing_symbols(self):
        """Genes this project wants and Allen has no ISH experiment for."""
        return [entry["symbol"] for entry in self.unavailable]

    # --- the matrix --------------------------------------------------------

    def energy_matrix(self, structure_ids, symbols=None, plane=None, combine=None):
        """Build a (gene x region) expression-energy matrix.

        `plane` and `combine` are required; see `PLANE_POLICIES`,
        `COMBINE_RULES` and this module's docstring for why there is no
        default. Regions no admitted experiment reports come back NaN.
        """
        if plane is None or combine is None:
            raise TypeError(
                "energy_matrix() requires plane= and combine=. There is no "
                "default because a gene has several ISH experiments in two "
                "sectioning geometries that disagree by up to ~30-fold in "
                "small structures, so choosing among them is a modelling "
                "decision, not a loader convenience. plane in %s, combine in "
                "%s." % (list(PLANE_POLICIES), list(COMBINE_RULES))
            )
        if plane not in PLANE_POLICIES:
            raise ValueError(
                "plane must be one of %s, got %r" % (list(PLANE_POLICIES), plane)
            )
        if combine not in COMBINE_RULES:
            raise ValueError(
                "combine must be one of %s, got %r" % (list(COMBINE_RULES), combine)
            )
        if symbols is None:
            symbols = self.symbols()
        symbols = list(symbols)
        structure_ids = [int(i) for i in structure_ids]
        self._check_structures_exist(structure_ids)

        values = np.full((len(symbols), len(structure_ids)), np.nan)
        counts = np.zeros((len(symbols), len(structure_ids)), dtype=int)
        used = {}
        for row, symbol in enumerate(symbols):
            experiments = self[symbol].select(plane)
            used[symbol] = [e.dataset.id for e in experiments]
            if combine == "single" and len(experiments) != 1:
                raise ValueError(
                    "combine='single' requires exactly one experiment, but %s "
                    "has %d under plane=%r (%s). Pick a different plane policy "
                    "or a combine rule that says what to do with several."
                    % (symbol, len(experiments), plane, used[symbol])
                )
            if not experiments:
                continue
            stack = np.vstack([e.energy_for(structure_ids) for e in experiments])
            reported = np.isfinite(stack)
            counts[row] = reported.sum(axis=0)
            has_any = counts[row] > 0
            if np.any(has_any):
                values[row, has_any] = _reduce(combine, stack[:, has_any])
        return RegionalExpression(
            values=values,
            n_experiments=counts,
            symbols=symbols,
            structure_ids=structure_ids,
            plane=plane,
            combine=combine,
            dataset_ids=used,
            units=sources.EXPRESSION_ENERGY_UNITS,
            citation=sources.ALLEN_ISH_CITATION,
            ontology=self.ontology,
        )

    def for_connectome(self, connectome, symbols=None, plane=None, combine=None):
        """`energy_matrix` over a `RegionalConnectome`'s source regions.

        This is the alignment the roadmap asks for: the returned matrix's
        columns are the connectome's rows, in the connectome's order, so a
        region index means the same thing in both layers.
        """
        return self.energy_matrix(
            structure_ids=connectome.source_ids,
            symbols=symbols,
            plane=plane,
            combine=combine,
        )

    def _check_structures_exist(self, structure_ids):
        """A structure id outside the ontology is a caller bug, not a gap.

        The two look identical in the output otherwise: both produce an
        all-NaN column. But "this region was not measured" is a fact about
        the ISH data that the report exists to surface, while "this is not a
        region" is a mis-join, and a whole-brain map quietly built on a
        typo'd id is exactly the kind of bug that never announces itself.
        """
        if self.ontology is None:
            return
        unknown = sorted(set(i for i in structure_ids if i not in self.ontology))
        if unknown:
            raise KeyError(
                "structure ids %s are not in this ontology, so they are not "
                "regions at all. An unmeasured region comes back NaN and is "
                "reported; a non-existent one is refused, because the two are "
                "indistinguishable in the output." % unknown
            )

    # --- reporting ---------------------------------------------------------

    def coverage_report(self, structure_ids=None):
        """Per-gene experiment counts, planes, and region coverage.

        Everything here is measured from the loaded files. Nothing in it is
        written down anywhere else, so a change upstream shows up in the
        report rather than in a stale comment.
        """
        rows = []
        for symbol in self.symbols():
            gene_expression = self.genes[symbol]
            row = {
                "symbol": symbol,
                "protein": gene_expression.gene.protein,
                "system": gene_expression.gene.system,
                "n_experiments": len(gene_expression.experiments),
                "planes": gene_expression.planes(),
                "dataset_ids": gene_expression.dataset_ids(),
            }
            if structure_ids is not None:
                wanted = [int(i) for i in structure_ids]
                covered = set()
                for experiment in gene_expression.experiments:
                    covered.update(
                        i
                        for i, value in zip(wanted, experiment.energy_for(wanted))
                        if np.isfinite(value)
                    )
                row["n_regions_covered"] = len(covered)
                row["n_regions_requested"] = len(wanted)
                row["regions_missing"] = [i for i in wanted if i not in covered]
            rows.append(row)
        return rows

    def unavailable_report(self):
        """The searched-and-empty record, as loaded from the manifest."""
        return [dict(entry) for entry in self.unavailable]

    def experiments_without_unionizes(self):
        """Indexed experiments that contributed no rows to any loaded gene.

        On the full data these are four experiments (282 Cnr1, 359 Drd2,
        526 Gria1, 534 Grin1, all from the original sagittal release) that
        are indexed, non-failed, and have **no StructureUnionize records in
        any structure graph** - not merely none in graph 1.

        Nothing in the API's SectionDataSet record explains why. The
        `expression` flag is false for 28 of the 106 indexed experiments,
        but 24 of those 28 do have unionizes, so the flag does not
        distinguish these four; neither does `delegate`, `failed`,
        `failed_facet`, `section_thickness`, `weight`, `reference_space_id`
        or any channel field, all of which are identical between experiment
        282 and its sibling 284 from the same gene. So the cause is not
        recoverable from this data and is not guessed at here.

        The index query does not filter these out, because "this experiment
        exists and yielded nothing" is worth being able to see; this
        accessor is how it is seen, rather than by noticing that a count
        does not add up.

        Only meaningful when the atlas was loaded with every gene the index
        covers, so it reports only experiments whose gene IS loaded.
        """
        loaded = set()
        for gene_expression in self.genes.values():
            loaded.update(gene_expression.dataset_ids())
        return [
            dataset
            for dataset in self.index.datasets
            if dataset.symbol in self.genes and dataset.id not in loaded
        ]


def _reduce(combine, stack):
    """Reduce (n_experiments, n_regions) to (n_regions,), ignoring NaN.

    Every column passed in has at least one finite entry, so no all-NaN
    warning is possible and none is suppressed.
    """
    if combine in ("mean", "single"):
        return np.nanmean(stack, axis=0)
    if combine == "median":
        return np.nanmedian(stack, axis=0)
    if combine == "max":
        return np.nanmax(stack, axis=0)
    raise ValueError("unknown combine rule %r" % (combine,))  # pragma: no cover


def load_expression_atlas(
    ontology=None,
    symbols=None,
    cache_dir=None,
    allow_download=False,
    check_manifest_counts=True,
):
    """Load the ISH expression atlas from the verified cache.

    `ontology` defaults to the connectome's CCF structure graph, which is
    the point: one ontology, one meaning of "region". `allow_download` is
    False by default so no ordinary import or test can start a download.

    `check_manifest_counts` compares the parsed row and experiment counts
    with the ones recorded in the manifest. It is on by default because a
    truncated API response hashes consistently and would otherwise pass the
    checksum while silently dropping experiments.
    """
    from . import fetch

    if ontology is None:
        from ..connectome.ontology import load_ontology

        ontology = load_ontology(allow_download=allow_download)

    index_path = fetch.ensure(
        sources.DATASET_INDEX_KEY, cache_dir=cache_dir, allow_download=allow_download
    )
    index = read_dataset_index(index_path)

    if symbols is None:
        symbols = list(sources.AVAILABLE_SYMBOLS)
    loaded = {}
    for symbol in symbols:
        entry = sources.unionize_file(symbol)
        path = fetch.ensure(
            entry.key, cache_dir=cache_dir, allow_download=allow_download
        )
        experiments = read_unionize_csv(path, index, expected_symbol=symbol)
        if check_manifest_counts:
            check_counts(
                entry.key, entry.n_rows, entry.n_section_data_sets, experiments
            )
        loaded[symbol] = GeneExpression(_genes.gene(symbol), experiments)

    return ExpressionAtlas(
        gene_expression=loaded,
        index=index,
        ontology=ontology,
        unavailable=sources.GENES_WITHOUT_DATA,
    )


def check_counts(label, n_rows_expected, n_experiments_expected, experiments):
    """Compare what was parsed with what the manifest recorded.

    The SHA-256 already proves the bytes are the ones that were retrieved.
    This catches the other failure: a *truncated* API response, which RMA
    will happily return if a query outgrows its `num_rows`, and which would
    hash consistently while silently dropping experiments. The manifest
    records the counts precisely so the check can run offline afterwards.
    """
    n_rows = sum(e.n_structures for e in experiments)
    if n_rows != n_rows_expected:
        raise ValueError(
            "%s: parsed %d unionize rows, %d were recorded. The bytes match "
            "their digest, so this is a loader bug or a truncated pinned "
            "response, not an upstream change." % (label, n_rows, n_rows_expected)
        )
    if len(experiments) != n_experiments_expected:
        raise ValueError(
            "%s: parsed %d experiments, %d were recorded"
            % (label, len(experiments), n_experiments_expected)
        )
