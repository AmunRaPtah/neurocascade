"""P4.1: readers for the two published shapes this package ingests - the
gene/experiment index, and one gene's structure unionizes.

What a structure unionize is
----------------------------

Allen's ISH pipeline produces, per experiment, a 200 um voxel grid of
expression energy registered to the adult mouse reference space. A
*unionize* is that grid summed over every voxel labelled with a given CCF
structure, which is what turns a volume into a per-region number. One row
is therefore one (experiment, structure) pair, and one gene has as many
rows as it has experiments times structures.

Three properties of the real files, found by looking
----------------------------------------------------

**A gene has more than one experiment, and they disagree.** Oprm1 has
three (one coronal, two sagittal); Chrm4 has five. In
the ventral tuberomammillary nucleus the three Oprm1 experiments report
expression energies of 0.285, 0.027 and 0.789 - a 29-fold spread for the
same gene in the same structure. So "the expression of Oprm1 in TMv" is not
a number this data hands you; it is a number somebody has to *decide* how
to compute, which is why nothing in this module combines experiments and
why `atlas.ExpressionAtlas.energy_matrix()` makes the caller say how.

**Coronal and sagittal are not interchangeable.** Allen's sagittal pipeline
sections and processes the left hemisphere only, while coronal experiments
cover the whole brain (Allen's own Mouse Brain Atlas documentation: the
coronal atlas covers 4,104 genes, and the rest of the genome was sectioned
sagittally with only the left hemisphere processed). Measured here: the
coronal Oprm1 experiment carries
unionizes for 279 of the connectome's 291 source regions, its two sagittal
siblings for 262 and 274. `plane` is therefore a first-class field on every
experiment rather than a footnote.

**Not every structure is in every experiment.** Coverage varies by a few
regions between experiments of the same gene (Slc6a3's coronal experiment
978 is missing three more regions than Oprm1's), and twelve connectome
regions are missing from all of them for reference-space reasons recorded
in `sources.MISSING_FROM_ISH`. An absent row means *not measured*, and this
module represents it as NaN and never as zero. That distinction is the same
one P3.1 made for Oh et al. 2014's `#NUM!` p-value cells, and it matters
more here, not less: a zero in a receptor map reads as "this drug has no
target in this region", which is a claim, whereas NaN reads as "we do not
know", which is the truth.
"""

import csv
import json

import numpy as np

#: The columns the API returns for model::StructureUnionize, in order. The
#: reader requires this header exactly: a reordered or extended header means
#: the API changed shape and the numeric columns can no longer be trusted to
#: be where the parser thinks they are.
UNIONIZE_COLUMNS = (
    "id",
    "section_data_set_id",
    "structure_id",
    "sum_pixels",
    "sum_expressing_pixels",
    "sum_pixel_intensity",
    "sum_expressing_pixel_intensity",
    "expression_density",
    "expression_energy",
    "voxel_energy_mean",
    "voxel_energy_cv",
)

PLANES = ("coronal", "sagittal")

#: Allen reference space ids the adult ISH data is registered to, by plane:
#: 9 is "P56 Brain", 10 is "P56 Brain (L/R Flipped)". Holds for all 106
#: experiments in the pinned index with no exceptions, which is what makes it
#: an invariant worth enforcing rather than a summary. Both spaces predate
#: CCFv3, which is the reason the connectome has twelve regions the ISH data
#: cannot reach; see `sources.MISSING_FROM_ISH`.
REFERENCE_SPACE_BY_PLANE = {"coronal": 9, "sagittal": 10}


class SectionDataSet(object):
    """One ISH experiment: which gene, which plane, which reference space.

    The plane and the reference space are checked against each other on
    construction. An experiment registered to something other than the P56
    space its plane implies is not comparable with the rest and must not be
    pooled into a gene's map silently; there is no such experiment in the
    pinned index, and this is how it stays that way.
    """

    __slots__ = ("id", "symbol", "plane", "reference_space_id")

    def __init__(self, id, symbol, plane, reference_space_id=None):
        if plane not in PLANES:
            raise ValueError(
                "unknown plane of section %r for section data set %s; "
                "expected one of %s" % (plane, id, list(PLANES))
            )
        expected = REFERENCE_SPACE_BY_PLANE[plane]
        if reference_space_id is not None and int(reference_space_id) != expected:
            raise ValueError(
                "section data set %s is %s but is registered to reference "
                "space %s, not the expected %d. A differently registered "
                "experiment is not commensurable with the others and must "
                "not be pooled into a gene's map."
                % (id, plane, reference_space_id, expected)
            )
        self.id = int(id)
        self.symbol = symbol
        self.plane = plane
        self.reference_space_id = expected

    def __repr__(self):
        return "SectionDataSet(%d, %r, %r)" % (self.id, self.symbol, self.plane)


class SectionDataSetIndex(object):
    """The gene -> experiment index, keyed both ways."""

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self._by_id = {}
        for dataset in self.datasets:
            if dataset.id in self._by_id:
                raise ValueError(
                    "section data set %d appears twice in the index" % dataset.id
                )
            self._by_id[dataset.id] = dataset
        self._by_symbol = {}
        for dataset in self.datasets:
            self._by_symbol.setdefault(dataset.symbol, []).append(dataset)

    def __len__(self):
        return len(self.datasets)

    def __repr__(self):
        return "SectionDataSetIndex(%d experiments, %d genes)" % (
            len(self.datasets),
            len(self._by_symbol),
        )

    def __getitem__(self, dataset_id):
        try:
            return self._by_id[int(dataset_id)]
        except KeyError:
            raise KeyError(
                "section data set %r is not in this index; the index is what "
                "says which experiments a map may be built from, so an "
                "unindexed experiment is a mismatch between two manifest "
                "files rather than something to guess about" % (dataset_id,)
            )

    def __contains__(self, dataset_id):
        return int(dataset_id) in self._by_id

    def symbols(self):
        return sorted(self._by_symbol)

    def for_gene(self, symbol):
        """Every experiment for one gene, index order. Empty list if none."""
        return list(self._by_symbol.get(symbol, []))

    def planes_for(self, symbol):
        return sorted(set(d.plane for d in self.for_gene(symbol)))


def read_dataset_index(path):
    """Parse the pinned `model::SectionDataSet` JSON response.

    Refuses a response the API marked unsuccessful, and refuses a truncated
    one (`num_rows < total_rows`), because a short index would silently drop
    experiments and the unionize files would then contain rows this package
    cannot attribute to a gene.
    """
    with open(path, "r") as handle:
        payload = json.load(handle)
    if not payload.get("success", False):
        raise ValueError(
            "%s is not a successful API response: %r" % (path, payload.get("msg"))
        )
    if payload.get("num_rows") != payload.get("total_rows"):
        raise ValueError(
            "%s is a truncated API response (%s of %s rows); the index must be "
            "complete or experiments go missing without notice"
            % (path, payload.get("num_rows"), payload.get("total_rows"))
        )
    datasets = []
    for row in payload["msg"]:
        plane = row["plane_of_section"]["name"]
        for gene in row["genes"]:
            datasets.append(
                SectionDataSet(
                    id=row["id"],
                    symbol=gene["acronym"],
                    plane=plane,
                    reference_space_id=row.get("reference_space_id"),
                )
            )
    return SectionDataSetIndex(datasets)


class ExperimentUnionizes(object):
    """One experiment's per-structure expression, as parallel arrays.

    `structure_ids` are CCF graph-1 ids, so they are directly comparable
    with the connectome's `source_ids`. No ontology is held here: naming a
    region is `atlas.py`'s job and it uses the connectome's ontology so
    there is one source of truth for what a region is called.
    """

    __slots__ = (
        "dataset",
        "structure_ids",
        "expression_energy",
        "expression_density",
        "sum_pixels",
        "_index",
    )

    def __init__(
        self,
        dataset,
        structure_ids,
        expression_energy,
        expression_density,
        sum_pixels,
    ):
        self.dataset = dataset
        self.structure_ids = np.asarray(structure_ids, dtype=int)
        self.expression_energy = np.asarray(expression_energy, dtype=float)
        self.expression_density = np.asarray(expression_density, dtype=float)
        self.sum_pixels = np.asarray(sum_pixels, dtype=float)
        n = self.structure_ids.size
        for name in ("expression_energy", "expression_density", "sum_pixels"):
            if getattr(self, name).shape != (n,):
                raise ValueError("%s has the wrong length" % name)
        if len(set(self.structure_ids.tolist())) != n:
            raise ValueError(
                "experiment %d unionizes the same structure twice" % dataset.id
            )
        for name in ("expression_energy", "expression_density", "sum_pixels"):
            values = getattr(self, name)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    "experiment %d has non-finite %s; the published pipeline "
                    "emits a number for every unionized structure, so this is "
                    "a parse failure" % (dataset.id, name)
                )
            if np.any(values < 0.0):
                raise ValueError(
                    "experiment %d has negative %s; expression energies, "
                    "densities and pixel counts are non-negative by "
                    "construction, so this is a parse failure"
                    % (dataset.id, name)
                )
        self._index = {int(i): k for k, i in enumerate(self.structure_ids)}

    def __repr__(self):
        return "ExperimentUnionizes(%d, %r, %r, %d structures)" % (
            self.dataset.id,
            self.dataset.symbol,
            self.dataset.plane,
            self.structure_ids.size,
        )

    @property
    def n_structures(self):
        return self.structure_ids.size

    def energy_for(self, structure_ids):
        """Expression energy for the requested structures, NaN where absent.

        NaN means "this experiment does not report this structure", which is
        not the same statement as zero expression and must never be spelled
        as one. See the module docstring.
        """
        return self._column_for(self.expression_energy, structure_ids)

    def density_for(self, structure_ids):
        return self._column_for(self.expression_density, structure_ids)

    def pixels_for(self, structure_ids):
        return self._column_for(self.sum_pixels, structure_ids)

    def _column_for(self, values, structure_ids):
        wanted = [int(i) for i in structure_ids]
        out = np.full(len(wanted), np.nan)
        for k, structure_id in enumerate(wanted):
            position = self._index.get(structure_id)
            if position is not None:
                out[k] = values[position]
        return out


def read_unionize_csv(path, index, expected_symbol=None):
    """Read one gene's pinned unionize CSV into per-experiment objects.

    `index` is the `SectionDataSetIndex`; every `section_data_set_id` in the
    file must appear in it, which is what attributes rows to a gene and a
    plane. A row whose experiment is unindexed is an error rather than a
    dropped row: the two files are pinned together and a mismatch between
    them means one of them moved.

    Rows are returned grouped by experiment, in the order the experiments
    first appear in the file (which the manifest's query pins to ascending
    experiment id).
    """
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if tuple(header) != UNIONIZE_COLUMNS:
            raise ValueError(
                "%s: expected the column header %s, got %s. A changed header "
                "means the API's response shape moved and the numeric columns "
                "can no longer be read positionally."
                % (path, list(UNIONIZE_COLUMNS), header)
            )
        columns = {name: i for i, name in enumerate(UNIONIZE_COLUMNS)}
        grouped = {}
        order = []
        for line in reader:
            if not line or all(cell == "" for cell in line):
                continue
            if len(line) != len(UNIONIZE_COLUMNS):
                raise ValueError(
                    "%s: row has %d cells, header has %d"
                    % (path, len(line), len(UNIONIZE_COLUMNS))
                )
            dataset_id = int(line[columns["section_data_set_id"]])
            if dataset_id not in grouped:
                grouped[dataset_id] = ([], [], [], [])
                order.append(dataset_id)
            structure_ids, energies, densities, pixels = grouped[dataset_id]
            structure_ids.append(int(line[columns["structure_id"]]))
            energies.append(float(line[columns["expression_energy"]]))
            densities.append(float(line[columns["expression_density"]]))
            pixels.append(float(line[columns["sum_pixels"]]))

    experiments = []
    for dataset_id in order:
        dataset = index[dataset_id]
        if expected_symbol is not None and dataset.symbol != expected_symbol:
            raise ValueError(
                "%s: experiment %d belongs to %s, not %s"
                % (path, dataset_id, dataset.symbol, expected_symbol)
            )
        structure_ids, energies, densities, pixels = grouped[dataset_id]
        experiments.append(
            ExperimentUnionizes(
                dataset=dataset,
                structure_ids=structure_ids,
                expression_energy=energies,
                expression_density=densities,
                sum_pixels=pixels,
            )
        )
    return experiments
