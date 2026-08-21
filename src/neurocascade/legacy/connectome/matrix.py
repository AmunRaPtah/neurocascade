"""P3.1: the directed, weighted region-by-region connectome matrix.

What a loaded object is
-----------------------

`RegionalConnectome` holds one published matrix plus everything needed to
know what its numbers mean and which brain regions its rows and columns
are:

    weights            (n_source, n_target) float array, >= 0
    source_ids         (n_source,)  CCF structure ids, one per row
    target_ids         (n_target,)  CCF structure ids, one per column
    target_hemispheres (n_target,)  'ipsi' or 'contra', one per column
    units, citation, dataset_key    what the numbers are and where from

Three decisions in that shape are worth stating.

**Orientation is the file's, and converting is explicit.** `weights` is
source-major (`weights[i, j]` = projection from `source_ids[i]` to
`target_ids[j]`), because that is how both publications ship and print
their matrices, and because a loader that silently transposed would make
the cached bytes and the in-memory array disagree about what row 0 is.
This codebase's `CircuitNetwork.connectivity` uses the opposite,
mathematical convention (`connectivity[i, j]` = from `j` onto `i`), so
`projection_matrix()` exists and does exactly one thing: return the
transpose, target-major, with the ids to match. Everything downstream that
builds a `CircuitNetwork` goes through it. This is not pedantry: a
whole-brain model built on a silently transposed connectome is wrong in a
way that still produces plausible-looking dynamics, since both matrices
have the same sparsity statistics and the same weight distribution.

**Hemisphere is a property of a column, not a separate matrix.** Both
datasets publish an ipsilateral and a contralateral block over the same
source regions. Keeping them as one array with a per-column hemisphere
label means "the full directed graph over 2N target nodes" and "just the
ipsilateral square block" are both one slice away, and neither is
privileged. The alternative (two objects) makes the common
`ipsi + contra` whole-brain case the awkward one.

**Weights are left exactly as published.** No thresholding, no log
transform, no row normalization, no symmetrization, no removal of the
diagonal. Every one of those is a modelling choice with consequences for
the circuit layer, and P3.1's job is ingestion. `RegionalConnectome` does
check that what it loaded is finite and non-negative, because both
publications' models are non-negative by construction and a negative entry
would mean the file was misparsed rather than that the brain has an
inhibitory long-range projection. (Sign assignment from neurotransmitter
identity is P3.2's problem, and it is a separate one: the tracer data
these matrices come from is anterograde bulk fluorescence, which carries
no sign at all.)

The self-connection diagonal
----------------------------

Both matrices have non-zero ipsilateral diagonals: a region projects to
itself, which for a bulk tracer measurement mostly means local
intra-regional axon collaterals. That is real signal, not an artefact, but
it is also not the same quantity as an inter-regional projection, and a
Wilson-Cowan node already carries its own local recurrence in its gain and
threshold. `without_self_connections()` exists so P3.2 can make that call
deliberately; the loader does not make it.
"""

import csv
import gzip

import numpy as np

from . import sources

HEMISPHERES = ("ipsi", "contra")


class RegionalConnectome(object):
    """One published directed region-by-region matrix, with its provenance."""

    def __init__(
        self,
        weights,
        source_ids,
        target_ids,
        target_hemispheres,
        dataset_key,
        citation,
        units,
        ontology=None,
        pvalues=None,
    ):
        self.weights = np.asarray(weights, dtype=float)
        self.source_ids = np.asarray(source_ids, dtype=int)
        self.target_ids = np.asarray(target_ids, dtype=int)
        self.target_hemispheres = np.asarray(target_hemispheres, dtype=object)
        self.dataset_key = dataset_key
        self.citation = citation
        self.units = units
        self.ontology = ontology
        self.pvalues = None if pvalues is None else np.asarray(pvalues, dtype=float)

        n_source, n_target = self.weights.shape
        if self.source_ids.shape != (n_source,):
            raise ValueError(
                "source_ids has %d entries but the matrix has %d rows"
                % (self.source_ids.size, n_source)
            )
        if self.target_ids.shape != (n_target,):
            raise ValueError(
                "target_ids has %d entries but the matrix has %d columns"
                % (self.target_ids.size, n_target)
            )
        if self.target_hemispheres.shape != (n_target,):
            raise ValueError("target_hemispheres must have one entry per column")
        bad = set(self.target_hemispheres) - set(HEMISPHERES)
        if bad:
            raise ValueError("unknown hemisphere labels: %s" % sorted(bad))
        if len(set(self.source_ids.tolist())) != n_source:
            raise ValueError("duplicate source structure ids")
        pairs = list(zip(self.target_hemispheres.tolist(), self.target_ids.tolist()))
        if len(set(pairs)) != n_target:
            raise ValueError("duplicate (hemisphere, structure id) target columns")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError(
                "%s contains non-finite weights; the published models are "
                "finite everywhere, so this is a parse failure" % dataset_key
            )
        if np.any(self.weights < 0.0):
            raise ValueError(
                "%s contains negative weights; both published models are "
                "non-negative by construction (bounded/non-negative least "
                "squares), so this is a parse failure, not an inhibitory "
                "projection" % dataset_key
            )
        if self.pvalues is not None and self.pvalues.shape != self.weights.shape:
            raise ValueError("pvalues must have the same shape as weights")

    # --- shape and identity ------------------------------------------------

    @property
    def n_source(self):
        return self.weights.shape[0]

    @property
    def n_target(self):
        return self.weights.shape[1]

    def __repr__(self):
        return "RegionalConnectome(%r, %d sources x %d targets)" % (
            self.dataset_key,
            self.n_source,
            self.n_target,
        )

    def source_acronyms(self):
        return [self._ontology()[i].acronym for i in self.source_ids]

    def target_acronyms(self):
        """Hemisphere-tagged target labels, e.g. 'VTA' / 'VTA (contra)'."""
        ontology = self._ontology()
        return [
            ontology[i].acronym if h == "ipsi" else "%s (contra)" % ontology[i].acronym
            for i, h in zip(self.target_ids, self.target_hemispheres)
        ]

    def region_metadata(self):
        """One dict per source region: id, acronym, name, major division.

        This is the "region ontology metadata" half of what a loader has to
        return: enough to name every row and to know where it sits in the
        hierarchy, without handing the caller the whole 1327-node tree.
        """
        ontology = self._ontology()
        rows = []
        for structure_id in self.source_ids:
            structure = ontology[structure_id]
            division_id = ontology.major_division(structure_id)
            rows.append(
                {
                    "id": structure.id,
                    "acronym": structure.acronym,
                    "name": structure.name,
                    "major_division_id": division_id,
                    "major_division_acronym": (
                        None if division_id is None else ontology[division_id].acronym
                    ),
                }
            )
        return rows

    def _ontology(self):
        if self.ontology is None:
            raise ValueError(
                "this connectome was loaded without an ontology, so region "
                "names are not available; pass ontology= to the loader"
            )
        return self.ontology

    # --- slicing -----------------------------------------------------------

    def hemisphere(self, name):
        """Restrict to ipsilateral or contralateral target columns."""
        if name not in HEMISPHERES:
            raise ValueError("hemisphere must be one of %s, got %r" % (HEMISPHERES, name))
        keep = np.array([h == name for h in self.target_hemispheres], dtype=bool)
        return self._with_columns(keep)

    def subset(self, structure_ids):
        """Restrict rows AND columns to the given regions, in the given order.

        Raises for a region this dataset does not contain, rather than
        dropping it: a whole-brain model quietly built on eight of the ten
        regions the caller asked for is a bug that never announces itself.
        Both hemispheres of each requested region are kept where the dataset
        has them.
        """
        wanted = [int(i) for i in structure_ids]
        source_index = {int(i): k for k, i in enumerate(self.source_ids)}
        missing = [i for i in wanted if i not in source_index]
        if missing:
            raise KeyError(
                "%s has no source rows for structure ids %s"
                % (self.dataset_key, missing)
            )
        rows = [source_index[i] for i in wanted]

        column_index = self._column_index()
        columns = [
            column_index[(hemisphere, structure_id)]
            for hemisphere in HEMISPHERES
            for structure_id in wanted
            if (hemisphere, structure_id) in column_index
        ]
        weights = self.weights[np.ix_(rows, columns)]
        pvalues = (
            None if self.pvalues is None else self.pvalues[np.ix_(rows, columns)]
        )
        return RegionalConnectome(
            weights=weights,
            source_ids=[int(self.source_ids[r]) for r in rows],
            target_ids=[int(self.target_ids[c]) for c in columns],
            target_hemispheres=[self.target_hemispheres[c] for c in columns],
            dataset_key=self.dataset_key,
            citation=self.citation,
            units=self.units,
            ontology=self.ontology,
            pvalues=pvalues,
        )

    def _column_index(self):
        return {
            (str(h), int(i)): k
            for k, (h, i) in enumerate(zip(self.target_hemispheres, self.target_ids))
        }

    def _with_columns(self, keep):
        return RegionalConnectome(
            weights=self.weights[:, keep],
            source_ids=self.source_ids,
            target_ids=self.target_ids[keep],
            target_hemispheres=self.target_hemispheres[keep],
            dataset_key=self.dataset_key,
            citation=self.citation,
            units=self.units,
            ontology=self.ontology,
            pvalues=None if self.pvalues is None else self.pvalues[:, keep],
        )

    def without_self_connections(self):
        """Zero the ipsilateral diagonal; see the module docstring."""
        weights = self.weights.copy()
        column_index = self._column_index()
        for i, source_id in enumerate(self.source_ids):
            column = column_index.get(("ipsi", int(source_id)))
            if column is not None:
                weights[i, column] = 0.0
        return RegionalConnectome(
            weights=weights,
            source_ids=self.source_ids,
            target_ids=self.target_ids,
            target_hemispheres=self.target_hemispheres,
            dataset_key=self.dataset_key,
            citation=self.citation,
            units=self.units,
            ontology=self.ontology,
            pvalues=self.pvalues,
        )

    # --- conversion --------------------------------------------------------

    def projection_matrix(self):
        """`(matrix, source_ids, target_ids, target_hemispheres)`, target-major.

        `matrix[i, j]` is the weight of the projection from source `j` onto
        target `i`: the `CircuitNetwork.connectivity` convention, and the
        convention Knox et al. 2019 use in their own mathematics. This is
        the ONLY place the transpose happens.
        """
        return (
            self.weights.T.copy(),
            self.source_ids.copy(),
            self.target_ids.copy(),
            self.target_hemispheres.copy(),
        )

    def square_ipsilateral(self):
        """`(structure_ids, matrix)` for the ipsilateral block, target-major.

        Both datasets' ipsilateral target set equals their source set, so
        this block is square and its rows and columns are the same regions
        in the same order: the directed adjacency matrix of the one
        hemisphere, ready to become a `CircuitNetwork.connectivity`.
        """
        ipsi = self.hemisphere("ipsi")
        order = {int(i): k for k, i in enumerate(ipsi.target_ids)}
        if set(order) != set(int(i) for i in ipsi.source_ids):
            raise ValueError(
                "%s: ipsilateral target set differs from source set, so there "
                "is no square block" % self.dataset_key
            )
        columns = [order[int(i)] for i in ipsi.source_ids]
        weights = ipsi.weights[:, columns]
        return ipsi.source_ids.copy(), weights.T.copy()


# ---------------------------------------------------------------------------
# Readers, one per published file format.
# ---------------------------------------------------------------------------


def _open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "rt", newline="")


def read_knox2019_csv(path, dataset_key, citation, units, ontology=None):
    """Read one Knox et al. 2019 regionalized CSV.

    Layout, verified against the published files: a two-row header, where
    row 1 is a hemisphere label per column ('ipsi' then 'contra') and row 2
    is the target structure id per column, both with an empty first cell;
    then one row per source region, whose first cell is the source
    structure id.

    The full files are 291 sources x 577 targets (291 ipsilateral + 286
    contralateral). The five ipsilateral-only targets are RM, RO, RPA
    (raphe magnus/obscurus/pallidus), OV (vascular organ of the lamina
    terminalis) and EW (Edinger-Westphal nucleus): midline structures with
    no contralateral counterpart to project to. So a ragged ipsi/contra
    column count is expected, not a parse error, and this reader does not
    require the two blocks to match.
    """
    with _open_maybe_gzip(path) as handle:
        reader = csv.reader(handle)
        hemisphere_row = next(reader)
        id_row = next(reader)
        if len(hemisphere_row) != len(id_row):
            raise ValueError("%s: header rows differ in length" % path)
        if hemisphere_row[0] != "" or id_row[0] != "":
            raise ValueError(
                "%s: expected an empty top-left cell in both header rows, got "
                "%r / %r" % (path, hemisphere_row[0], id_row[0])
            )
        target_hemispheres = hemisphere_row[1:]
        target_ids = [int(value) for value in id_row[1:]]

        source_ids = []
        rows = []
        for line in reader:
            if not line or all(cell == "" for cell in line):
                continue
            if len(line) != len(id_row):
                raise ValueError(
                    "%s: data row for source %r has %d cells, header has %d"
                    % (path, line[0], len(line), len(id_row))
                )
            source_ids.append(int(line[0]))
            rows.append([float(value) for value in line[1:]])

    return RegionalConnectome(
        weights=np.array(rows, dtype=float),
        source_ids=source_ids,
        target_ids=target_ids,
        target_hemispheres=target_hemispheres,
        dataset_key=dataset_key,
        citation=citation,
        units=units,
        ontology=ontology,
    )


def read_oh2014_xlsx(path, dataset_key, citation, units, ontology):
    """Read Oh et al. 2014 Supplementary Table 3 (four sheets, acronym-keyed).

    An ontology is required, not optional: the file labels its rows and
    columns with CCF acronyms and this package keys everything by structure
    id so the two datasets can be compared at all. An acronym that does not
    resolve is an error rather than a dropped region.

    openpyxl is imported here rather than at module scope so that the Knox
    path - the default, and the one the tests exercise most - keeps this
    package's import-time dependencies at numpy alone.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:  # pragma: no cover - environment-dependent
        raise ImportError(
            "reading %s needs openpyxl (the file is .xlsx as published by "
            "Nature). The Knox et al. 2019 CSVs need no such dependency." % path
        )

    workbook = load_workbook(path, read_only=True, data_only=True)
    expected = ["W_ipsi", "PValue_ipsi", "W_contra", "PValue_contra"]
    missing = [name for name in expected if name not in workbook.sheetnames]
    if missing:
        raise ValueError(
            "%s: expected sheets %s, missing %s (found %s)"
            % (path, expected, missing, workbook.sheetnames)
        )

    def cell(value, sheet_name, allow_excel_error):
        # The p-value sheets carry the literal Excel error token '#NUM!' in
        # every cell whose connection weight is zero, and only there: on the
        # published file the equivalence (p is '#NUM!') <-> (W == 0) holds
        # exactly, in both hemispheres (28415 ipsilateral and 29501
        # contralateral cells). Oh et al. fit p-values only to connections
        # the model gave a non-zero weight, so the token means "no
        # connection, therefore no confidence value", and NaN is the honest
        # in-memory spelling of that. Zero would be a lie in the most
        # damaging possible direction, since a p-value of 0 reads as maximal
        # confidence. The same token in a WEIGHT sheet would be a parse
        # failure and is refused.
        if isinstance(value, str):
            if allow_excel_error and value.strip() == "#NUM!":
                return float("nan")
            raise ValueError(
                "%s sheet %s: unexpected non-numeric cell %r"
                % (path, sheet_name, value)
            )
        if value is None:
            # No cell in the published sheets is blank. Treating a blank as
            # zero is exactly the failure mode the '#NUM!' handling above
            # exists to avoid, so it is refused rather than defaulted.
            raise ValueError(
                "%s sheet %s: blank cell; the published table has none, so "
                "this file is not the published table" % (path, sheet_name)
            )
        return float(value)

    def read_sheet(name, allow_excel_error=False):
        grid = [list(row) for row in workbook[name].iter_rows(values_only=True)]
        header = [item for item in grid[0][1:] if item is not None]
        labels = []
        values = []
        for row in grid[1:]:
            if row[0] is None:
                continue
            labels.append(str(row[0]))
            values.append(
                [
                    cell(v, name, allow_excel_error)
                    for v in row[1: 1 + len(header)]
                ]
            )
        return [str(item) for item in header], labels, np.array(values, dtype=float)

    sheets = {
        name: read_sheet(name, allow_excel_error=name.startswith("PValue"))
        for name in expected
    }
    workbook.close()

    # All four sheets must be labelled identically, or the p-values would be
    # silently attached to the wrong connections. Nothing else in the file
    # would look wrong if they were not.
    ipsi_columns, ipsi_rows, ipsi_w = sheets["W_ipsi"]
    for name in expected[1:]:
        columns, rows, _ = sheets[name]
        if rows != ipsi_rows:
            raise ValueError(
                "%s: sheet %s has different source rows from W_ipsi" % (path, name)
            )
        if columns != ipsi_columns:
            raise ValueError(
                "%s: sheet %s has different target columns from W_ipsi"
                % (path, name)
            )
    contra_w = sheets["W_contra"][2]
    ipsi_p = sheets["PValue_ipsi"][2]
    contra_p = sheets["PValue_contra"][2]

    source_ids = [ontology.id_for_acronym(acronym) for acronym in ipsi_rows]
    column_ids = [ontology.id_for_acronym(acronym) for acronym in ipsi_columns]

    return RegionalConnectome(
        weights=np.hstack([ipsi_w, contra_w]),
        source_ids=source_ids,
        target_ids=column_ids + column_ids,
        target_hemispheres=(
            ["ipsi"] * len(column_ids) + ["contra"] * len(column_ids)
        ),
        dataset_key=dataset_key,
        citation=citation,
        units=units,
        ontology=ontology,
        pvalues=np.hstack([ipsi_p, contra_p]),
    )


_READERS = {
    "knox2019_normalized_connection_density": read_knox2019_csv,
    "knox2019_normalized_connection_strength": read_knox2019_csv,
    "knox2019_connection_strength": read_knox2019_csv,
    "knox2019_connection_density": read_knox2019_csv,
    "oh2014_supplementary_table_3": read_oh2014_xlsx,
}


def read_connectome(key, path, ontology=None):
    """Read a manifest file from an explicit path, no cache and no network."""
    entry = sources.data_file(key)
    reader = _READERS[key]
    if reader is read_oh2014_xlsx and ontology is None:
        raise ValueError(
            "%s is labelled by acronym, so an ontology is required to load it"
            % key
        )
    return reader(
        path,
        dataset_key=key,
        citation=entry.citation,
        units=entry.units,
        ontology=ontology,
    )


def load_connectome(
    key=None, ontology=None, cache_dir=None, allow_download=False, path=None
):
    """Load a connectivity matrix plus its region ontology metadata.

    Defaults to the Knox et al. 2019 normalized connection density matrix.
    `allow_download` is False by default so that neither a test nor an
    ordinary import can start a download; `scripts/fetch_connectome.py`
    passes True.
    """
    from . import fetch
    from .ontology import load_ontology

    if key is None:
        key = sources.DEFAULT_CONNECTIVITY_KEY
    if ontology is None:
        ontology = load_ontology(cache_dir=cache_dir, allow_download=allow_download)
    if path is None:
        path = fetch.ensure(key, cache_dir=cache_dir, allow_download=allow_download)
    return read_connectome(key, path, ontology=ontology)
