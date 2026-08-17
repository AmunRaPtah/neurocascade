"""Human receptor-gene expression, over the same Desikan-Killiany atlas as the
human connectome.

This module is the human analogue of `neurocascade.expression` (the Allen
MOUSE in-situ-hybridization package). It ingests the Allen Human Brain
Atlas (AHBA) microarray transcriptome, already processed into a
region x gene matrix on the ENIGMA Toolbox's Desikan-Killiany parcellation,
and hands back a gene x region matrix whose region order is identical to
`human.load_connectome()`'s 82-region atlas (68 cortical + 14 subcortical).
The connectome and the expression map therefore mean the same thing by
"region", which is the property the mouse package established in P4.1 and
the whole point of ingesting these two files together.

Why this file is small where the mouse package is large
-------------------------------------------------------

The mouse ISH package has to reason about experiments, sectioning planes,
unionize rows and a live API. The human data is one published CSV that the
ENIGMA Toolbox already aggregated: six AHBA donors, probe selection,
sample-to-parcel matching, normalization and donor aggregation are already
done upstream (Arnatkeviciute 2019; Markello 2021). So this module has no
plane policy and no combine rule; the one decision the mouse loader refuses
to make silently is already made and published here. What remains is the
same honesty work, and it is the bulk of this file:

* the manifest (`DATA_FILES`), the only place the URL and SHA-256 are
  written down;
* the gene registry (`HUMAN_RECEPTOR_GENES`) and, for every gene the model
  wants that this file does NOT contain, the searched-and-empty record
  (`UNTHRESHOLDED_ONLY`, `ABSENT`);
* the loader (`HumanGeneExpression`), which keeps NaN as "not measured" and
  refuses to turn it into zero;
* the conversion to a relative density (`relative_density`), under the same
  assumption the mouse package states as an assumption, restated here.

Offline by default, same as the rest of `human/`
------------------------------------------------

Nothing here touches the network at import time and `load_expression()`
defaults to `allow_download=False`; a cache miss raises
`fetch.MissingCachedFile` telling you to run
`scripts/fetch_human_expression.py`. Tests run off the small committed
fixture in `fixtures/`.

    from neurocascade import human

    conn = human.load_connectome()
    expr = human.load_expression()
    assert expr.region_labels == conn.labels    # one meaning of "region"

    rho = human.relative_density(expr, "DRD2", normalization="max")

What the numbers are, and what they are not
-------------------------------------------

The CSV's numbers are AHBA microarray expression values that abagen
processed (intensity-based probe filtering, differential-stability probe
selection, sample-to-parcel matching, normalization, aggregation within
parcels and across the six donors). The published values are measured to
lie in [0, 1]; a value outside that interval is therefore rejected as a
parse failure, not accepted as "expression".

They are **not** receptor densities. The same chain of caveats that
`neurocascade.expression.density` spells out applies verbatim: mRNA is not
protein, protein is not binding sites, and the receptor may not be in the
region that transcribes it. `relative_density` below converts a gene's row
into a dimensionless relative density under an assumption stated as an
assumption, exactly as the mouse package does.

Two regions, said out loud
--------------------------

Two Desikan-Killiany regions - the right frontal pole and the right
temporal pole (`R_frontalpole`, `R_temporalpole`) - were not matched to any
AHBA tissue sample and are therefore all-NaN in the published file. They
are `REGIONS_WITHOUT_SAMPLES`, kept as NaN rather than filled, because
zero is the claim "a drug acting here has no target" and that is not what
the data says.

Searched and empty, said out loud
---------------------------------

The ENIGMA release keeps only genes whose expression is reproducible across
donors (cross-donor correlation r >= 0.2), leaving 12,668 of the AHBA's
genes. Three of this model's genes survive only in the unthresholded
release (`UNTHRESHOLDED_ONLY`: ADRA1A, SLC6A3/DAT), and eight are absent
from BOTH releases (`ABSENT`: OPRD1/delta, DRD3, DRD5, ADRA2B, ADRB3,
ADORA2A, SLC6A2/NET, SLC6A4/SERT). None of those eleven is substituted
with another gene's map; a human-first model built on this layer must
record those targets as having no expression map, which is what the
registry records.
"""

import csv
import os
import sys

import numpy as np

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401
from . import atlas as _atlas

__all__ = [
    # citations
    "AHBA_CITATION",
    "ABAGEN_WORKFLOW_CITATION",
    "ABAGEN_TOOLBOX_CITATION",
    "ENIGMA_CITATION",
    # manifest
    "MANIFEST_VERSION",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DEFAULT_CACHE_DIR",
    "EXPRESSION_UNITS",
    "data_file",
    # gene registry
    "HumanReceptorGene",
    "HUMAN_RECEPTOR_GENES",
    "GENES_BY_SYMBOL",
    "SYSTEMS",
    "STABLE_SYMBOLS",
    "UNTHRESHOLDED_ONLY",
    "ABSENT",
    "REGIONS_WITHOUT_SAMPLES",
    "gene",
    "genes_for_system",
    "symbols",
    "availability",
    # fetch
    "ChecksumMismatch",
    "MissingCachedFile",
    "cache_path",
    "verify",
    "is_cached",
    "download",
    "ensure",
    "fetch_all",
    "checksum_lines",
    "sha256_of",
    # loader
    "HumanGeneExpression",
    "read_expression",
    "load_expression",
    # density
    "ASSUMPTION",
    "NORMALIZATIONS",
    "HumanReceptorDensityMap",
    "relative_density",
]


# --- citations -------------------------------------------------------------

AHBA_CITATION = (
    "Hawrylycz MJ, Lein ES, Guillozet-Bongaarts AL, Shen EH, Ng L, Miller "
    "JA, van de Lagemaat LN, Smith KA, Ebbert A, Riley ZL, et al. 'An "
    "anatomically comprehensive atlas of the adult human brain "
    "transcriptome.' Nature. 2012;489(7416):391-399. "
    "doi:10.1038/nature11405. The underlying data: post-mortem microarray "
    "expression from six adult human donor brains (Allen Human Brain "
    "Atlas), covering ~3,700 tissue samples across the whole brain."
)

ABAGEN_WORKFLOW_CITATION = (
    "Arnatkeviciute A, Fulcher BD, Fornito A. 'A practical guide to linking "
    "brain-wide gene expression and neuroimaging data.' NeuroImage. "
    "2019;189:353-367. doi:10.1016/j.neuroimage.2019.01.011. The "
    "processing pipeline: intensity-based filtering of microarray probes, "
    "selection of a representative probe per gene, matching of microarray "
    "samples to brain parcels, normalization, and aggregation within "
    "parcels and across donors."
)

ABAGEN_TOOLBOX_CITATION = (
    "Markello RD, Arnatkeviciute A, Poline JB, Fulcher BD, Fornito A, Misic "
    "B. 'Standardizing workflows in imaging transcriptomics with the abagen "
    "toolbox.' eLife. 2021;10:e72129. doi:10.7554/eLife.72129."
)

ENIGMA_CITATION = (
    "Lariviere S, Paquola C, Park B-y, Royer J, Wang Y, Bernhardt BC, Misic "
    "B, et al. 'The ENIGMA Toolbox: multiscale neural contextualization of "
    "multisite neuroimaging datasets.' Nature Methods. 2021;18(7):698-700. "
    "doi:10.1038/s41592-021-01186-4. Distributed the processed AHBA matrix "
    "on the Desikan-Killiany parcellation, keeping genes reproducible across "
    "donors at r >= 0.2."
)

EXPRESSION_CITATION = (
    AHBA_CITATION + " " + ABAGEN_WORKFLOW_CITATION + " " +
    ABAGEN_TOOLBOX_CITATION + " " + ENIGMA_CITATION
)


# --- manifest --------------------------------------------------------------

MANIFEST_VERSION = "2026-08-16"

#: What a human should run when the cache is missing a file.
FETCH_COMMAND = "scripts/fetch_human_expression.py"

#: The enigma-extra repository hosts the processed AHBA matrices that the
#: ENIGMA Toolbox's `fetch_ahba()` loads. Pinned to one commit.
_BASE = (
    "https://raw.githubusercontent.com/saratheriver/enigma-extra/"
    "8c9df6e0c37c66830360ad553807326035e180ed/ahba"
)

#: Default on-disk cache, shared with the human connectome: <repo>/data/human.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

EXPRESSION_UNITS = (
    "AHBA microarray expression, processed by the abagen workflow "
    "(intensity-based probe filtering, differential-stability probe "
    "selection, sample-to-parcel matching, normalization, and aggregation "
    "across six donors) on the Desikan-Killiany parcellation. Dimensionless; "
    "the published values are measured to lie in [0, 1]. NOT a receptor "
    "density and NOT comparable in absolute terms between genes."
)


class DataFile(datafetch.DataFile):
    """One external file, defaulting `retrieved` to this manifest's date."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", "2026-08-16")
        datafetch.DataFile.__init__(self, *args, **kwargs)


DATA_FILES = [
    DataFile(
        key="enigma_hcp_ahba_dk_expression",
        filename="allgenes_stable_r0.2.csv",
        url=_BASE + "/allgenes_stable_r0.2.csv",
        sha256="4c31801c130b8894cf86785353bb4d814df71e44ade9245ec6e585554cc6b185",
        size_bytes=19570163,
        citation=EXPRESSION_CITATION,
        description=(
            "AHBA microarray gene expression, processed by abagen onto the "
            "ENIGMA Toolbox's Desikan-Killiany parcellation (68 cortical + "
            "14 subcortical = 82 regions), keeping genes reproducible across "
            "the six donors at cross-donor correlation r >= 0.2 (12,668 "
            "genes). First column is the region label; the remaining columns "
            "are genes. The 82 region labels are in the SAME order as the "
            "human connectome's `enigma_hcp_sc_labels_aparc_with_sctx` file, "
            "so region i means the same region on both sides. Two regions "
            "(R_frontalpole, R_temporalpole) matched no tissue sample and "
            "are all-NaN."
        ),
        units=EXPRESSION_UNITS,
        volatility="static",
    ),
]

DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

#: The one expression file this layer currently ingests.
DEFAULT_EXPRESSION_KEY = "enigma_hcp_ahba_dk_expression"


def data_file(key):
    """Look up one manifest entry, naming the alternatives on a miss."""
    try:
        return DATA_FILES_BY_KEY[key]
    except KeyError:
        raise KeyError(
            "unknown data file %r; manifest %s has: %s"
            % (key, MANIFEST_VERSION, ", ".join(sorted(DATA_FILES_BY_KEY)))
        )


_FETCHER = datafetch.Fetcher(
    manifest=sys.modules[__name__],
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/human",
)


def cache_path(key, cache_dir=None):
    return _FETCHER.cache_path(key, cache_dir)


def verify(key, cache_dir=None):
    return _FETCHER.verify(key, cache_dir)


def is_cached(key, cache_dir=None):
    return _FETCHER.is_cached(key, cache_dir)


def download(key, cache_dir=None, timeout=300, force=False):
    return _FETCHER.download(key, cache_dir=cache_dir, timeout=timeout, force=force)


def ensure(key, cache_dir=None, allow_download=True, timeout=300):
    return _FETCHER.ensure(
        key, cache_dir=cache_dir, allow_download=allow_download, timeout=timeout
    )


def fetch_all(keys=None, cache_dir=None, timeout=300, force=False, log=None):
    return _FETCHER.fetch_all(
        keys=keys, cache_dir=cache_dir, timeout=timeout, force=force, log=log
    )


def checksum_lines():
    return _FETCHER.checksum_lines()


# --- the gene registry -----------------------------------------------------


class HumanReceptorGene(object):
    """One human gene, what it encodes, and whether this layer has its map."""

    __slots__ = ("symbol", "protein", "system", "note")

    def __init__(self, symbol, protein, system, note=""):
        self.symbol = symbol
        self.protein = protein
        self.system = system
        self.note = note

    def __repr__(self):
        return "HumanReceptorGene(%r, %r)" % (self.symbol, self.protein)


#: Human (Homo sapiens) receptor and transporter genes the model needs a map
#: for, in HGNC symbols. Ordered by receptor system, mirroring the mouse
#: `expression.genes.RECEPTOR_GENES` table. A gene's availability is recorded
#: separately below (`STABLE_SYMBOLS` / `UNTHRESHOLDED_ONLY` / `ABSENT`) so
#: that "the model needs it" and "the data has it" are two different,
#: greppable facts.
HUMAN_RECEPTOR_GENES = (
    # --- opioid ------------------------------------------------------------
    HumanReceptorGene("OPRM1", "mu-opioid receptor (MOR)", "opioid"),
    HumanReceptorGene("OPRK1", "kappa-opioid receptor (KOR)", "opioid"),
    HumanReceptorGene(
        "OPRD1", "delta-opioid receptor (DOR)", "opioid",
        "absent from both ENIGMA AHBA releases",
    ),
    # --- dopamine ----------------------------------------------------------
    HumanReceptorGene("DRD1", "dopamine D1 receptor", "dopamine"),
    HumanReceptorGene("DRD2", "dopamine D2 receptor", "dopamine"),
    HumanReceptorGene(
        "DRD3", "dopamine D3 receptor", "dopamine",
        "absent from both ENIGMA AHBA releases",
    ),
    HumanReceptorGene("DRD4", "dopamine D4 receptor", "dopamine"),
    HumanReceptorGene(
        "DRD5", "dopamine D5 receptor", "dopamine",
        "absent from both ENIGMA AHBA releases",
    ),
    # --- GABA-A ------------------------------------------------------------
    HumanReceptorGene("GABRA1", "GABA-A receptor alpha1 subunit", "gaba_a"),
    HumanReceptorGene("GABRA2", "GABA-A receptor alpha2 subunit", "gaba_a"),
    HumanReceptorGene("GABRA3", "GABA-A receptor alpha3 subunit", "gaba_a"),
    HumanReceptorGene("GABRA4", "GABA-A receptor alpha4 subunit", "gaba_a"),
    HumanReceptorGene("GABRA5", "GABA-A receptor alpha5 subunit", "gaba_a"),
    HumanReceptorGene("GABRB1", "GABA-A receptor beta1 subunit", "gaba_a"),
    HumanReceptorGene("GABRB2", "GABA-A receptor beta2 subunit", "gaba_a"),
    HumanReceptorGene("GABRB3", "GABA-A receptor beta3 subunit", "gaba_a"),
    HumanReceptorGene("GABRG2", "GABA-A receptor gamma2 subunit", "gaba_a"),
    # --- serotonin ---------------------------------------------------------
    HumanReceptorGene("HTR1A", "5-HT1A receptor", "serotonin"),
    HumanReceptorGene("HTR2A", "5-HT2A receptor", "serotonin"),
    HumanReceptorGene("HTR2C", "5-HT2C receptor", "serotonin"),
    # --- acetylcholine -----------------------------------------------------
    HumanReceptorGene("CHRM1", "muscarinic M1 receptor", "acetylcholine"),
    HumanReceptorGene("CHRM2", "muscarinic M2 receptor", "acetylcholine"),
    HumanReceptorGene("CHRM3", "muscarinic M3 receptor", "acetylcholine"),
    HumanReceptorGene("CHRM4", "muscarinic M4 receptor", "acetylcholine"),
    HumanReceptorGene("CHRM5", "muscarinic M5 receptor", "acetylcholine"),
    HumanReceptorGene("CHRNA4", "nicotinic acetylcholine receptor alpha4 subunit", "acetylcholine"),
    HumanReceptorGene("CHRNA7", "nicotinic acetylcholine receptor alpha7 subunit", "acetylcholine"),
    HumanReceptorGene("CHRNB2", "nicotinic acetylcholine receptor beta2 subunit", "acetylcholine"),
    # --- noradrenaline -----------------------------------------------------
    HumanReceptorGene(
        "ADRA1A", "adrenergic alpha-1A receptor", "noradrenaline",
        "only in the unthresholded ENIGMA AHBA release (r-1), not the "
        "r >= 0.2 stable set",
    ),
    HumanReceptorGene("ADRA1B", "adrenergic alpha-1B receptor", "noradrenaline"),
    HumanReceptorGene("ADRA1D", "adrenergic alpha-1D receptor", "noradrenaline"),
    HumanReceptorGene("ADRA2A", "adrenergic alpha-2A receptor", "noradrenaline"),
    HumanReceptorGene(
        "ADRA2B", "adrenergic alpha-2B receptor", "noradrenaline",
        "absent from both ENIGMA AHBA releases",
    ),
    HumanReceptorGene("ADRA2C", "adrenergic alpha-2C receptor", "noradrenaline"),
    HumanReceptorGene("ADRB1", "adrenergic beta-1 receptor", "noradrenaline"),
    HumanReceptorGene("ADRB2", "adrenergic beta-2 receptor", "noradrenaline"),
    HumanReceptorGene(
        "ADRB3", "adrenergic beta-3 receptor", "noradrenaline",
        "absent from both ENIGMA AHBA releases",
    ),
    # --- glutamate ---------------------------------------------------------
    HumanReceptorGene("GRIN1", "NMDA receptor GluN1 subunit", "glutamate"),
    HumanReceptorGene("GRIN2A", "NMDA receptor GluN2A subunit", "glutamate"),
    HumanReceptorGene("GRIN2B", "NMDA receptor GluN2B subunit", "glutamate"),
    HumanReceptorGene("GRIA1", "AMPA receptor GluA1 subunit", "glutamate"),
    HumanReceptorGene("GRIA2", "AMPA receptor GluA2 subunit", "glutamate"),
    HumanReceptorGene("GRIA3", "AMPA receptor GluA3 subunit", "glutamate"),
    HumanReceptorGene("GRIA4", "AMPA receptor GluA4 subunit", "glutamate"),
    # --- histamine, cannabinoid, adenosine ---------------------------------
    HumanReceptorGene("HRH1", "histamine H1 receptor", "histamine"),
    HumanReceptorGene("CNR1", "cannabinoid CB1 receptor", "cannabinoid"),
    HumanReceptorGene(
        "ADORA2A", "adenosine A2A receptor", "adenosine",
        "absent from both ENIGMA AHBA releases",
    ),
    # --- monoamine transporters -------------------------------------------
    HumanReceptorGene(
        "SLC6A2", "noradrenaline transporter (NET)", "transporter",
        "absent from both ENIGMA AHBA releases",
    ),
    HumanReceptorGene(
        "SLC6A3", "dopamine transporter (DAT)", "transporter",
        "only in the unthresholded ENIGMA AHBA release (r-1), not the "
        "r >= 0.2 stable set",
    ),
    HumanReceptorGene(
        "SLC6A4", "serotonin transporter (SERT)", "transporter",
        "absent from both ENIGMA AHBA releases",
    ),
)

GENES_BY_SYMBOL = {g.symbol: g for g in HUMAN_RECEPTOR_GENES}

#: Distinct receptor systems represented, in table order.
SYSTEMS = tuple(dict.fromkeys(g.system for g in HUMAN_RECEPTOR_GENES))

#: The genes the stable (r >= 0.2) ENIGMA AHBA file actually contains,
#: measured against the pinned file, in registry order. This is the set the
#: loader can load.
STABLE_SYMBOLS = (
    "OPRM1", "OPRK1",
    "DRD1", "DRD2", "DRD4",
    "GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5",
    "GABRB1", "GABRB2", "GABRB3", "GABRG2",
    "HTR1A", "HTR2A", "HTR2C",
    "CHRM1", "CHRM2", "CHRM3", "CHRM4", "CHRM5", "CHRNA4", "CHRNA7", "CHRNB2",
    "ADRA1B", "ADRA1D", "ADRA2A", "ADRA2C", "ADRB1", "ADRB2",
    "GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GRIA3", "GRIA4",
    "HRH1", "CNR1",
)

#: Genes present only in the unthresholded ENIGMA AHBA release
#: (`allgenes_stable_r-1.csv`), dropped from the r >= 0.2 stable set because
#: their expression was not reproducible across donors at that threshold.
UNTHRESHOLDED_ONLY = {
    "ADRA1A": (
        "present in the unthresholded ENIGMA AHBA release "
        "(allgenes_stable_r-1.csv) but not in the r >= 0.2 stable set, so "
        "its cross-donor reproducibility is below the threshold this layer "
        "ingests"
    ),
    "SLC6A3": (
        "present in the unthresholded ENIGMA AHBA release "
        "(allgenes_stable_r-1.csv) but not in the r >= 0.2 stable set, so "
        "its cross-donor reproducibility is below the threshold this layer "
        "ingests"
    ),
}

#: Genes absent from BOTH ENIGMA AHBA releases (stable and unthresholded).
#: Measured, not assumed: each was checked against the two pinned files'
#: headers. This is a searched-and-empty record, not a bug to be papered over.
ABSENT = {
    "OPRD1": (
        "no OPRD1 column in either ENIGMA AHBA release; the abagen pipeline "
        "did not retain a delta-opioid probe that passed its filters"
    ),
    "DRD3": (
        "no DRD3 column in either ENIGMA AHBA release"
    ),
    "DRD5": (
        "no DRD5 column in either ENIGMA AHBA release"
    ),
    "ADRA2B": (
        "no ADRA2B column in either ENIGMA AHBA release"
    ),
    "ADRB3": (
        "no ADRB3 column in either ENIGMA AHBA release"
    ),
    "ADORA2A": (
        "no ADORA2A column in either ENIGMA AHBA release"
    ),
    "SLC6A2": (
        "no SLC6A2 (NET) column in either ENIGMA AHBA release"
    ),
    "SLC6A4": (
        "no SLC6A4 (SERT) column in either ENIGMA AHBA release"
    ),
}

#: Desikan-Killiany regions the AHBA pipeline matched to no tissue sample,
#: and which are therefore all-NaN in the published matrix. Kept as NaN: a
#: missing sample is not the claim "no receptor here".
REGIONS_WITHOUT_SAMPLES = ("R_frontalpole", "R_temporalpole")


def gene(symbol):
    """Look up one gene, with an error naming the alternatives."""
    try:
        return GENES_BY_SYMBOL[symbol]
    except KeyError:
        raise KeyError(
            "%r is not a gene this layer ingests; the table has: %s"
            % (symbol, ", ".join(sorted(GENES_BY_SYMBOL)))
        )


def genes_for_system(system):
    """Every gene of one receptor system, in table order."""
    return [g for g in HUMAN_RECEPTOR_GENES if g.system == system]


def symbols():
    """Every gene symbol, in table order."""
    return [g.symbol for g in HUMAN_RECEPTOR_GENES]


def availability(symbol):
    """`"stable"`, `"unthresholded-only"`, or `"absent"` for one gene.

    The loader can only load `"stable"` genes from the pinned file. The
    other two standings are the searched-and-empty record.
    """
    gene(symbol)  # raises for an unknown symbol
    if symbol in STABLE_SYMBOLS:
        return "stable"
    if symbol in UNTHRESHOLDED_ONLY:
        return "unthresholded-only"
    return "absent"


# --- the loader ------------------------------------------------------------


class HumanGeneExpression(object):
    """A (gene x region) human expression matrix and how it was made.

    `values[g, r]` is the normalized AHBA microarray expression of
    `symbols[g]` in `region_labels[r]`, or NaN where the published file has
    no measurement for that (gene, region) pair. NaN is not zero: zero is
    the claim "a drug acting at this gene has no target in this region",
    and "not measured" is not that claim.
    """

    def __init__(self, values, symbols, region_labels, units, citation, dataset_key):
        self.values = np.asarray(values, dtype=float)
        self.symbols = tuple(symbols)
        self.region_labels = tuple(region_labels)
        self.units = units
        self.citation = citation
        self.dataset_key = dataset_key

        shape = (len(self.symbols), len(self.region_labels))
        if self.values.shape != shape:
            raise ValueError(
                "values has shape %s but there are %d genes and %d regions"
                % (self.values.shape, shape[0], shape[1])
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("duplicate gene symbol in expression matrix")
        if len(set(self.region_labels)) != len(self.region_labels):
            raise ValueError("duplicate region label in expression matrix")
        finite = np.isfinite(self.values)
        if np.any(self.values[finite] < 0.0) or np.any(self.values[finite] > 1.0):
            raise ValueError(
                "expression values outside [0, 1]; the published abagen-"
                "normalized values are bounded, so this is a parse failure"
            )
        if np.any(np.isinf(self.values)):
            raise ValueError("infinite expression value; this is a bug")

    def __repr__(self):
        return "HumanGeneExpression(%r, %d genes x %d regions)" % (
            self.dataset_key,
            len(self.symbols),
            len(self.region_labels),
        )

    @property
    def shape(self):
        return self.values.shape

    def index_of(self, symbol):
        try:
            return self.symbols.index(symbol)
        except ValueError:
            standing = availability(symbol)
            raise KeyError(
                "%r is not in this matrix (%s). It is %r in the manifest's "
                "searched-and-empty record: %s"
                % (
                    symbol,
                    self.dataset_key,
                    standing,
                    _standing_note(symbol, standing),
                )
            )

    def region_index_of(self, label):
        try:
            return self.region_labels.index(label)
        except ValueError:
            raise KeyError(
                "%r is not one of this matrix's %d region labels"
                % (label, len(self.region_labels))
            )

    def for_gene(self, symbol):
        """One gene's row over regions, in `region_labels` order."""
        return self.values[self.index_of(symbol)]

    def values_for(self, symbols):
        """A `(len(symbols), n_regions)` submatrix, NaN where unmeasured."""
        rows = [self.for_gene(s) for s in symbols]
        return np.vstack(rows)

    def genes_without_values(self):
        """Symbols whose whole row is NaN (none in the pinned file)."""
        return [
            symbol
            for symbol, row in zip(self.symbols, self.values)
            if not np.any(np.isfinite(row))
        ]

    def regions_without_values(self):
        """Region labels no loaded gene measures (the two unmatched regions)."""
        column_ok = np.any(np.isfinite(self.values), axis=0)
        return [
            label
            for label, ok in zip(self.region_labels, column_ok)
            if not ok
        ]

    def coverage(self):
        """Fraction of (gene, region) cells that carry a measurement."""
        if self.values.size == 0:
            return 0.0
        return float(np.isfinite(self.values).sum()) / self.values.size


def _standing_note(symbol, standing):
    if standing == "stable":
        return "present in the stable file and loadable"
    if standing == "unthresholded-only":
        return UNTHRESHOLDED_ONLY[symbol]
    return ABSENT[symbol]


def read_expression(matrix_path, dataset_key=None, symbols=None):
    """Read the AHBA CSV from an explicit path. No cache, no network.

    `symbols` restricts the loaded columns to a named gene subset (default:
    every stable gene in `HUMAN_RECEPTOR_GENES`).
    """
    if dataset_key is None:
        dataset_key = DEFAULT_EXPRESSION_KEY
    entry = data_file(dataset_key)

    with open(matrix_path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    if not header or header[0] != "label":
        raise ValueError(
            "%s: expected a leading 'label' column, got %r"
            % (matrix_path, header[0] if header else None)
        )
    gene_columns = header[1:]
    region_labels = [row[0] for row in rows]
    if not region_labels:
        raise ValueError("%s: no region rows" % matrix_path)

    if symbols is None:
        symbols = list(STABLE_SYMBOLS)
    symbols = list(symbols)
    missing = [s for s in symbols if s not in gene_columns]
    if missing:
        raise ValueError(
            "%s: genes not in the file: %s. Check availability() before "
            "requesting them." % (matrix_path, ", ".join(missing))
        )

    index = {s: gene_columns.index(s) for s in symbols}
    values = np.full((len(symbols), len(region_labels)), np.nan)
    for r, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                "%s: row %d has %d cells, header has %d"
                % (matrix_path, r + 2, len(row), len(header))
            )
        for g, symbol in enumerate(symbols):
            cell = row[index[symbol] + 1]
            if cell not in ("", "NaN", "nan", "NA", "None"):
                values[g, r] = float(cell)

    return HumanGeneExpression(
        values=values,
        symbols=symbols,
        region_labels=region_labels,
        units=entry.units,
        citation=entry.citation,
        dataset_key=dataset_key,
    )


def load_expression(cache_dir=None, allow_download=False, symbols=None):
    """Load the human expression matrix from the verified cache.

    `allow_download` is False by default so that neither a test nor an
    ordinary import can start a download;
    `scripts/fetch_human_expression.py` passes True.
    """
    path = ensure(
        DEFAULT_EXPRESSION_KEY,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    return read_expression(path, symbols=symbols)


# --- density ---------------------------------------------------------------

#: How a gene's row is turned into a dimensionless relative density. Same
#: meanings as `neurocascade.expression.density.NORMALIZATIONS`.
NORMALIZATIONS = ("max", "mean", "reference-region", "none")

ASSUMPTION = (
    "Within one gene, receptor density in a region is assumed proportional "
    "to AHBA microarray expression in that region: density = k_gene * "
    "expression, k_gene unknown and per gene. mRNA is not protein, protein "
    "is not binding sites, and for some genes the receptor protein is not "
    "even in the region that transcribes it. This is an assumption, not a "
    "measurement."
)


class HumanReceptorDensityMap(object):
    """One gene's per-region relative density, and how it was made.

    `values[i]` is the relative density of `symbol` in `region_labels[i]`,
    NaN where the expression matrix has no measurement for that region.
    """

    def __init__(
        self,
        values,
        region_labels,
        symbol,
        normalization,
        units,
        source,
        reference=None,
    ):
        self.values = np.asarray(values, dtype=float)
        self.region_labels = tuple(region_labels)
        self.symbol = symbol
        self.normalization = normalization
        self.units = units
        self.source = source
        self.reference = reference
        if self.values.shape != (len(self.region_labels),):
            raise ValueError(
                "%d values for %d regions"
                % (self.values.size, len(self.region_labels))
            )
        finite = np.isfinite(self.values)
        if np.any(self.values[finite] < 0.0):
            raise ValueError("negative relative density; this is a bug")

    def __repr__(self):
        return "HumanReceptorDensityMap(%r, %d regions, normalization=%r)" % (
            self.symbol,
            len(self.region_labels),
            self.normalization,
        )

    @property
    def assumption(self):
        return ASSUMPTION

    def for_region(self, label):
        try:
            index = self.region_labels.index(label)
        except ValueError:
            raise KeyError("%r is not one of this map's regions" % label)
        return float(self.values[index])

    def measured(self):
        """`(labels, values)` for the regions that have a value."""
        finite = np.isfinite(self.values)
        return (
            [self.region_labels[i] for i in range(len(self.values)) if finite[i]],
            self.values[finite],
        )

    def top(self, n=10):
        """The n highest regions, as `[(label, value)]`."""
        labels, values = self.measured()
        order = np.argsort(values)[::-1][:n]
        return [(labels[i], float(values[i])) for i in order]

    def as_coupling(self, mechanism_gain, nodes):
        """`{node: mechanism_gain * rho[region]}`, the receptor layer's input.

        `nodes` maps a circuit node name to a Desikan-Killiany region label.
        `mechanism_gain` is a single number or a per-node dict: it is the
        OTHER factor (how much a unit of receptor activation moves that node
        given the circuit around it) and is not this layer's to supply. A
        region with no measurement raises; substituting zero would assert
        the receptor is absent there.
        """
        gains = mechanism_gain
        if not isinstance(gains, dict):
            gains = {node: mechanism_gain for node in nodes}
        coupling = {}
        for node, label in nodes.items():
            if node not in gains:
                raise KeyError("no mechanism gain for node %r" % (node,))
            value = self.for_region(label)
            if not np.isfinite(value):
                raise ValueError(
                    "%s has no measured expression in %r (node %r), so this "
                    "map cannot supply its density factor. Substituting zero "
                    "would assert the receptor is absent there; it asserts "
                    "nothing." % (self.symbol, label, node)
                )
            coupling[node] = gains[node] * value
        return coupling


def relative_density(expression, symbol, normalization=None, reference=None):
    """Convert one gene's expression row into a relative density map.

    `normalization` is required; see `NORMALIZATIONS`. The same refusal as
    the mouse package: 'relative to the peak region', 'relative to the mean
    region' and 'relative to one named region' are three different
    quantities, so there is no default.
    """
    if normalization is None:
        raise TypeError(
            "relative_density() requires normalization=. There is no default "
            "because 'max', 'mean' and 'reference-region' are different "
            "quantities. normalization in %s." % (list(NORMALIZATIONS),)
        )
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            "normalization must be one of %s, got %r"
            % (list(NORMALIZATIONS), normalization)
        )

    row = np.array(expression.for_gene(symbol), dtype=float)
    finite = np.isfinite(row)
    if not np.any(finite):
        raise ValueError(
            "%s has no measured region in this matrix, so there is nothing "
            "to normalize" % symbol
        )

    values = np.full(row.shape, np.nan)
    values[finite] = row[finite]

    reference_label = None
    if normalization == "max":
        divisor = np.nanmax(values)
        units = "dimensionless, relative to the peak measured region"
    elif normalization == "mean":
        divisor = np.nanmean(values)
        units = "dimensionless, relative to the mean measured region"
    elif normalization == "reference-region":
        if reference is None:
            raise TypeError(
                "normalization='reference-region' needs reference=<region label>"
            )
        reference_label = reference
        if reference_label not in expression.region_labels:
            raise KeyError(
                "reference region %r is not in this matrix" % reference_label
            )
        divisor = values[expression.region_labels.index(reference_label)]
        if not np.isfinite(divisor) or divisor <= 0.0:
            raise ValueError(
                "reference region %r has no usable expression value (%r)"
                % (reference_label, divisor)
            )
        units = "dimensionless, relative to region %r" % reference_label
    else:
        divisor = 1.0
        units = expression.units

    if divisor <= 0.0:
        raise ValueError(
            "the normalizing divisor is %r; %s has no positive expression "
            "anywhere in this matrix" % (divisor, symbol)
        )

    return HumanReceptorDensityMap(
        values=values / divisor,
        region_labels=expression.region_labels,
        symbol=symbol,
        normalization=normalization,
        units=units,
        source=expression,
        reference=reference_label,
    )
