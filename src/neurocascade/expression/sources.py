"""P4.1: the versioned manifest of Allen Mouse Brain Atlas ISH expression
data this package ingests, and the only place a URL or a checksum for it is
written down.

This is the P3.1 manifest discipline applied to a second, unrelated body of
external data. Read `connectome/sources.py` first if you have not: the
contract (URL + measured SHA-256 + citation + units, a gitignored cache of
exactly the published bytes, and a hard failure on mismatch) is identical
and is implemented once in `neurocascade/datafetch.py`.

What the numbers are
--------------------

Allen's ISH pipeline segments each in-situ image into "expressing" and
"non-expressing" pixels, grids the result into 200 um cubic voxels
registered to the adult (P56) mouse reference space, and then *unionizes*
those voxels over every structure of the CCF ontology. The quantity this
project wants is the structure-level

    expression energy = (sum of intensity of expressing pixels in A)
                        / (sum of all pixels in A)

for a structure A, as defined in Allen's own ISH documentation and in Lein
et al. 2007. The published rows also carry

    expression density = sum_expressing_pixels / sum_pixels
    sum_pixels         = the structure's size in the ISH grid

Both identities were checked numerically against the retrieved rows rather
than taken on trust: over all 843 graph-1 unionizes of the coronal Oprm1
experiment (79912572), `sum_expressing_pixel_intensity / sum_pixels`
reproduces the published `expression_energy` to a maximum relative error of
1.1e-5, and `sum_expressing_pixels / sum_pixels` reproduces
`expression_density` to 9.6e-6 - both consistent with the API's six
significant figures and nothing else. So the columns mean what the
documentation says they mean, in these files.

**Expression energy is not a receptor density.** It is a
transcript-abundance measure of a gene, in arbitrary units, for a structure
delineated in an ISH-registered reference space. Converting it to anything
the receptor layer can use is P4.2's job and P4.2 owns stating the
proportionality assumption as an assumption. Nothing in this package does
that conversion.

Why the API and not a static archive
------------------------------------

Every file here is a **live API query**, which is the weaker of the two
provenance situations P3.1 distinguished, and it was chosen after looking
for the stronger one:

* `download.alleninstitute.org/informatics-archive/october-2014/mouse_expression/`
  is a genuinely static archive of the adult-mouse expression data, but it
  publishes per-experiment *voxel grids* (the 200 um energy volumes), not
  structure unionizes. Turning those into per-structure numbers would mean
  re-implementing Allen's unionization against
  `current-release/mouse_annotation/P56_Mouse_gridAnnotation.zip` and
  publishing our arithmetic as if it were theirs. The published unionize is
  the quantity the roadmap asks for; recomputing an approximation of it
  from a static file would trade a provenance problem for a correctness
  problem.
* No static dump of the structure unionizes was found on the informatics
  archive: its `current-release` tree holds `brain_observatory`,
  `mouse_annotation`, `mouse_ccf` and `rna_seq` only, and the dated trees
  (`october-2014`, `june-2016`, `september-2017`) hold projection data,
  cell types, annotation volumes and the voxel expression grids.

So each query URL is pinned with the SHA-256 of the bytes it returned on
`_RETRIEVED`, and every entry is labelled `volatility="live-api"`. Both the
index query and a per-gene unionize query were verified byte-identical
across repeated requests seconds apart. What the checksum buys is not
immutability, which nobody can promise for an API; it is that a change
upstream - a new ISH experiment for a gene, a re-run of the pipeline, a
revised ontology - surfaces as a dated, loud failure that a human reads,
instead of silently moving every downstream receptor map.

The ordering options in each query exist for the same reason. RMA's default
row order is not documented as stable, so both queries pin one explicitly:
without that the same rows could come back permuted and the digest would
fail for no real reason. (Note that `order$eq'section_data_sets.id'` is
rejected by the API as ambiguous on the SectionDataSet model, which is why
the index query orders by `genes.acronym`,`data_sets.id` instead.)

The ontology this shares with the connectome
--------------------------------------------

Every query filters `structure[graph_id$eq1]`: **CCF structure graph 1**,
the same graph the Knox et al. 2019 connectome is indexed by and the same
one `connectome/ontology.py` already loads. That is what makes "region"
mean the same thing on both sides, and it is why this package has no
ontology of its own - it takes a `StructureOntology` from the connectome
package and refuses to name a region without one.

Graph 1 unionizes cover 279 of the connectome's 291 source regions; the
twelve exceptions and why they are structural rather than gene-specific are
documented in `MISSING_FROM_ISH` below.
"""

import urllib.parse

from ..datafetch import DataFile as _DataFile
from . import genes as _genes

MANIFEST_VERSION = "2026-08-12"

#: What a human should run when the cache is missing a file. Read by
#: `datafetch.Fetcher` so the error message names the right script.
FETCH_COMMAND = "scripts/fetch_expression.py"

# Retrieval date shared by every entry below: every query was issued and
# hashed in one session while writing P4.1.
_RETRIEVED = "2026-08-12"

_API = "https://api.brain-map.org/api/v2/data/"

#: Allen product id 1 is the adult "Mouse Brain Atlas" ISH data set. Filtering
#: on it keeps the developing-mouse and spinal-cord products out.
MOUSE_BRAIN_ATLAS_PRODUCT_ID = 1

#: CCF structure graph 1: the adult mouse ontology, shared with the P3.1
#: connectome. See the module docstring.
STRUCTURE_GRAPH_ID = 1

ALLEN_ISH_CITATION = (
    "Lein ES, Hawrylycz MJ, Ao N, et al. 'Genome-wide atlas of gene "
    "expression in the adult mouse brain.' Nature. 2007;445(7124):168-176. "
    "doi:10.1038/nature05453 (PMID 17151600). Data retrieved from the Allen "
    "Brain Atlas API (Allen Mouse Brain Atlas, product id 1). Species: adult "
    "mouse (C57BL/6J), P56, in situ hybridization."
)

EXPRESSION_ENERGY_UNITS = (
    "expression energy: (sum of intensity of expressing pixels in the "
    "structure) / (sum of all pixels in the structure), from Allen's ISH "
    "informatics pipeline (200 um voxel grid, unionized over CCF structure "
    "graph 1). Arbitrary units, non-negative, NOT a receptor density and NOT "
    "comparable in absolute terms between genes, because the segmentation "
    "threshold and probe efficiency differ per experiment."
)


class UnionizeFile(_DataFile):
    """One gene's structure unionizes, plus what was counted in them.

    `n_rows` and `n_section_data_sets` are measured alongside the digest and
    checked by the loader. They are redundant with the SHA-256 for detecting
    an upstream change, but they are not redundant for detecting a
    *truncated* response: RMA paginates, and a query that silently returned
    only the first page would hash consistently and be quietly wrong. The
    loader compares them to what it actually parsed.

    `n_section_data_sets` counts the experiments **present in this file**,
    which for four genes is one fewer than the index lists for them: see
    `atlas.ExpressionAtlas.experiments_without_unionizes()`.
    """

    def __init__(self, symbol, n_rows, n_section_data_sets, **kwargs):
        self.symbol = symbol
        self.n_rows = n_rows
        self.n_section_data_sets = n_section_data_sets
        _DataFile.__init__(self, retrieved=_RETRIEVED, **kwargs)


def _url(endpoint, criteria):
    return _API + endpoint + "?" + urllib.parse.urlencode({"criteria": criteria})


def index_criteria(symbols):
    """RMA criteria for the gene -> ISH experiment index.

    One row per non-failed adult-mouse-atlas SectionDataSet for any of the
    requested genes, with the gene and the plane of section included. This is
    the file that says which experiments exist, which is what makes a
    per-gene unionize file interpretable: the unionize rows carry only a
    `section_data_set_id`.
    """
    acronyms = ",".join("'%s'" % symbol for symbol in symbols)
    return (
        "model::SectionDataSet,"
        "rma::criteria,[failed$eq'false'],products[id$eq%d],"
        "genes[acronym$in%s],"
        "rma::include,genes,plane_of_section,"
        "rma::options[order$eq'genes.acronym','data_sets.id']"
        "[num_rows$eq500][start_row$eq0]"
        % (MOUSE_BRAIN_ATLAS_PRODUCT_ID, acronyms)
    )


def index_url(symbols=None):
    """The exact URL the index file's bytes came from."""
    if symbols is None:
        symbols = _genes.symbols()
    return _url("query.json", index_criteria(symbols))


def unionize_criteria(symbol):
    """RMA criteria for one gene's graph-1 structure unionizes.

    Every non-failed adult-atlas experiment for the gene, every CCF graph-1
    structure it was unionized over, ordered by (experiment, structure) so
    the bytes are reproducible.
    """
    return (
        "model::StructureUnionize,"
        "rma::criteria,structure[graph_id$eq%d],"
        "section_data_set[failed$eq'false']"
        "(products[id$eq%d],genes[acronym$eq'%s']),"
        "rma::options[order$eq'structure_unionizes.section_data_set_id,"
        "structure_unionizes.structure_id']"
        "[num_rows$eq10000][start_row$eq0]"
        % (STRUCTURE_GRAPH_ID, MOUSE_BRAIN_ATLAS_PRODUCT_ID, symbol)
    )


def unionize_url(symbol):
    """The exact URL one gene's unionize CSV came from."""
    return _url("query.csv", unionize_criteria(symbol))


def unionize_key(symbol):
    return "aba_ish_unionize_%s" % symbol.lower()


# ---------------------------------------------------------------------------
# Genes searched for and not found.
#
# This is the searched-and-empty record the project's first rule requires, and
# it is a result rather than a gap to be papered over. Every symbol below was
# resolved in Allen's own gene table (so this is not a spelling problem) and
# then queried for SectionDataSets with NO filter at all - no product filter,
# no `failed` filter - which is the widest query the API supports.
# ---------------------------------------------------------------------------

GENES_WITHOUT_DATA = (
    {
        "symbol": "Adra1b",
        "entrez_id": 11548,
        "searched": (
            "Allen Brain Atlas API, 2026-08-12: model::Gene[acronym$eq'Adra1b'] "
            "returns three records, two of them Mus musculus (gene ids 11335 "
            "and 500726721, both Entrez 11548, 'adrenergic receptor, alpha 1b'), "
            "so the gene is in Allen's table. "
            "model::SectionDataSet,rma::criteria,genes[acronym$eq'Adra1b'] with "
            "no product filter and no failed filter returns total_rows = 0."
        ),
        "conclusion": (
            "No in situ hybridization experiment for Adra1b exists in any Allen "
            "mouse product. The gene is not droppable from the model's gene list "
            "- alpha-1B is a real adrenoceptor subtype that P5.3 will need - so "
            "it stays in `genes.RECEPTOR_GENES` with no expression map, and any "
            "alpha-1 map built from this package is explicitly alpha-1A + "
            "alpha-1D only. Substituting another alpha-1 subtype's map for it "
            "would be inventing a number."
        ),
    },
)

GENES_WITHOUT_DATA_BY_SYMBOL = {
    entry["symbol"]: entry for entry in GENES_WITHOUT_DATA
}

#: Genes with ISH data in one plane only. Not a "no data" case, but the
#: sagittal pipeline sections one hemisphere rather than the whole brain, so
#: these three have strictly weaker whole-brain coverage than the rest and a
#: coronal-only analysis cannot include them at all. Measured from the index
#: file, not asserted: see `atlas.ExpressionAtlas.coverage_report()`.
SAGITTAL_ONLY_SYMBOLS = ("Adra2c", "Adrb2", "Htr2a")

#: Connectome (Knox et al. 2019) source regions that have NO graph-1 unionize
#: in ANY of the ISH experiments retrieved here. Measured by intersecting the
#: 291 Knox source ids with the structure ids present in six independent
#: experiments (Oprm1, Drd2, Gabra1, Cnr1, Slc6a3, Grin1 coronal); the same
#: twelve are absent from every one of them, so this is a property of the
#: reference space the ISH data is registered to and not of any gene.
#:
#: The ISH grid is registered to the P56 adult mouse reference space that
#: predates CCFv3 (reference space id 9 for coronal experiments, 10 for
#: sagittal, "P56 Brain" and "P56 Brain (L/R Flipped)"), whereas Knox et al.
#: 2019 regionalize onto CCFv3. Four of the twelve (VISa, VISli, VISpor,
#: VISrl) are lateral/anterior visual areas that CCFv3 delineates and the
#: older annotation does not; the rest are small nuclei in the same position.
#: 279 of 291 regions, 95.9%, do carry ISH unionizes.
MISSING_FROM_ISH = (
    "AP",      # area postrema
    "ASO",     # accessory supraoptic group
    "EW",      # Edinger-Westphal nucleus
    "ICB",     # infracerebellar nucleus
    "IV",      # trochlear nucleus
    "NR",      # nucleus of Roller
    "PST",     # preparasubthalamic nucleus
    "PVa",     # periventricular hypothalamic nucleus, anterior part
    "VISa",    # anterior area
    "VISli",   # laterointermediate area
    "VISpor",  # postrhinal area
    "VISrl",   # rostrolateral visual area
)

DATASET_INDEX_KEY = "aba_ish_section_data_sets"

# ---------------------------------------------------------------------------
# The manifest.
#
# `_UNIONIZE_DIGESTS` is generated by `scripts/build_expression_manifest.py`
# from the bytes actually retrieved, and pasted here. It is not hand-written
# and it is not to be hand-edited to make a fetch pass; see
# `datafetch.ChecksumMismatch`.
#
#   symbol, sha256, size_bytes, n_rows, n_section_data_sets
# ---------------------------------------------------------------------------

_UNIONIZE_DIGESTS = (
    ("Oprm1", "a001196e12ebf5eb8f5c9f08ac84abebd4c7d640d065c84fd749f373b7f0556e", 224085, 2411, 3),
    ("Oprk1", "1639e8887977e56c0963674d14afb5e51ccec895e07a745038e36aaf3fd4b25b", 152094, 1623, 2),
    ("Oprd1", "5213866801c3d9a1f97ab4c7e8b3d86703e9a8193542b732c011cb45a674ccea", 151748, 1651, 2),
    ("Drd1", "d5a199f31aee728e18e5399d64ac0c338ff1f597ccd220b8b5473396ed0fef25", 233745, 2644, 4),
    ("Drd2", "035841983dfdff5b932bce6299acd8bb14d8ed119fea4ef09e7af08a0f0351c5", 217827, 2450, 3),
    ("Gabra1", "c3fb6bc96749c64788ff0eeb126d4792fe704bb25f6d6836fe141683470bac17", 148240, 1634, 2),
    ("Gabra2", "60acb06148502ef2eeae020cb6363a767d4c4a43569e6ac19e83118b1da7ddaa", 155777, 1659, 2),
    ("Gabra3", "56ba27f3da73fed7b5f1a60f5d830b8b2a5ec726f66ccc9cfbf509c15dfa8d49", 149297, 1642, 2),
    ("Gabra4", "168f518f3bc4e56f5c9b64a7f4deea2a1145d6f8dfc65221eed37235cf2370fd", 152999, 1661, 2),
    ("Gabra5", "4a4e09676399a0f6aacea3cfa0e45655ff567882bd9cc0330b40484a7acd55d2", 151691, 1664, 2),
    ("Gabrb1", "3211e183f2d892485237c372c21f224d90410170ed110fc419571a1ca5649457", 145416, 1636, 2),
    ("Gabrb2", "f84d9a5148dd0f3cf11ed213b7b485ce23d15448b2de6727451a64638e62587a", 294067, 3298, 4),
    ("Gabrb3", "0cda1e01f72cb7311fefe667919fab920b7898ffe6a07d22a341fbeb5581656c", 147251, 1620, 2),
    ("Htr1a", "54724fe730863f47ff3b8f8d9cb74ea4d071071f67027d28d7596b56a2726025", 151054, 1659, 2),
    ("Htr2a", "0e54f8ae2912e7d72bc11e6966d67c58e55e25e34e58f64355d2ac4d10b82717", 72979, 806, 1),
    ("Chrm1", "877fbeecb254ed0541877b62478435fe8382bc7dc0952f80f7afb30e98f928df", 150432, 1659, 2),
    ("Chrm2", "48a9f058f32dfc64767e2114e92644a75f178d3068d1913945873c07543c50fc", 147511, 1620, 2),
    ("Chrm3", "cc5bc4d354ba34174f79e21cfc425e59801cc34719d4a1022fa7236a0e271bcd", 290861, 3247, 4),
    ("Chrm4", "1a3d5ca6536647c670fe8102d43c290119e106ed477dfd6643be1c8833d07e12", 294482, 3349, 5),
    ("Chrm5", "98c4dcae517844f530e0c56dc27ce154335b18122c27184fde95b869dafb0c77", 153708, 1647, 2),
    ("Chrna4", "e60ccf5dcde260082c804e4883127689310b40354877981e2a206ab2dee48725", 215493, 2446, 3),
    ("Chrna7", "95090df06f1b9e990dd4dc198bfc9c804c2171376873741e7d5dc87eca660566", 148724, 1620, 2),
    ("Adra1a", "88d93b13caf4971157b35923074828c334d30fa2f526fe3699b1d24101ddf074", 225691, 2440, 3),
    ("Adra1d", "d22d3a7748d62164fc60e1f44bfeec0e8a47ba61edcef5054359cf0908d1500e", 149359, 1627, 2),
    ("Adra2a", "eba514abbf41b64f32e46f6223e7deeaeacccc98b838625ee2cf34b05a9c59ef", 157162, 1659, 2),
    ("Adra2b", "e6c984cc690010a778c73d17d4bc663c7fe44c6e21983cd353fc6b9197c501e8", 155230, 1637, 2),
    ("Adra2c", "cd5416422509fdcde567f969ec098ab70e2fd2b1bcd00ef05a3757c016b923dc", 77009, 816, 1),
    ("Adrb1", "b5d40c5f386667cae4a448860dee67d2a667be9c637ad3813d2f6d8d47bdc705", 153765, 1649, 2),
    ("Adrb2", "382813c22183c6bc0cd2a6c562609f18b35278002949bec237eacadfb4ee2a6c", 71892, 777, 1),
    ("Grin1", "4dee8a23c04be2c6fc4bea5d264e3a1952f69d0bd0a8a9c0f567b734dcf42e6a", 292829, 3310, 4),
    ("Grin2a", "e312981d404454692700c5975973eaa7c3786c43c3d0d875cf3fa8395bd5af28", 151660, 1649, 2),
    ("Grin2b", "88a0f9fafacd8d9e82a5394f383fc5fe61e63a0701f238aec940034e28c8c079", 149212, 1652, 2),
    ("Gria1", "74964c71338b1d5bb8e9b6eeafc29eadc3ef61ad8cdcfb981824f6d637c3c027", 147975, 1635, 2),
    ("Gria2", "e4a200fcea2f4f582777ed000a5c3e4b2d30d254a250ec60b820cf6c99404cd7", 223374, 2469, 3),
    ("Gria3", "d0a9be8b6ba26863ed51497926e3a3f296fa77209054bd2d75df4769857cae04", 150095, 1652, 2),
    ("Gria4", "3347dbe7bf24456666190616ee54cad1ff0625add9ad0919faa15ab6ec49777d", 149430, 1645, 2),
    ("Hrh1", "b5dd7553305529078a76d990098b1aea638ef3e391c1446e5e348ebfea4e220d", 145073, 1557, 2),
    ("Cnr1", "abdeeba67ea6bd6d360f02a7e1af9c30feef3036c122781bb9776e8ad2cbffdd", 290793, 3300, 4),
    ("Adora2a", "6af587c8701268060eb8d52ed0f65f9a1f440cd7afdb03429952752327e4e8d2", 227187, 2409, 3),
    ("Slc6a2", "e922a3b4960da9021032e9ae585cb6ce547ed93da3cab5f5e9aa103b9e181aba", 155451, 1637, 2),
    ("Slc6a3", "5183aa633c14a0d9f0030679c32e08516ecc65b5a437ffb0f282d83c7d58b41d", 293488, 3202, 4),
    ("Slc6a4", "8310dab7d0081474f84e5d25b18e44feeeba0d3db975092604e710e39c0d6b86", 153515, 1649, 2),
)

_INDEX_DIGEST = (
    "cf26412b7c574eebf93c8b6391f8ccbdd6bc3b050b1acfaa914bf3f268d6d43c",
    102969,
    106,
)


def _build_data_files():
    index_sha, index_size, _index_rows = _INDEX_DIGEST
    files = [
        _DataFile(
            key=DATASET_INDEX_KEY,
            filename="aba_ish_section_data_sets.json",
            url=index_url(),
            sha256=index_sha,
            size_bytes=index_size,
            citation=ALLEN_ISH_CITATION,
            description=(
                "One row per non-failed Allen Mouse Brain Atlas ISH experiment "
                "for the %d genes in `genes.RECEPTOR_GENES`, with the gene and "
                "the plane of section. The unionize files below carry only a "
                "section_data_set_id, so this is what makes them interpretable, "
                "and it is the record of exactly which experiments a map was "
                "built from." % len(_genes.RECEPTOR_GENES)
            ),
            units="experiment metadata; no numeric measurements",
            retrieved=_RETRIEVED,
            volatility="live-api",
        )
    ]
    for symbol, sha256, size_bytes, n_rows, n_datasets in _UNIONIZE_DIGESTS:
        files.append(
            UnionizeFile(
                symbol=symbol,
                n_rows=n_rows,
                n_section_data_sets=n_datasets,
                key=unionize_key(symbol),
                filename="aba_ish_unionize_%s.csv" % symbol,
                url=unionize_url(symbol),
                sha256=sha256,
                size_bytes=size_bytes,
                citation=ALLEN_ISH_CITATION,
                description=(
                    "Structure unionizes for %s (%s) over CCF structure graph "
                    "1: %d rows from %d ISH experiment(s)."
                    % (
                        symbol,
                        _genes.gene(symbol).protein,
                        n_rows,
                        n_datasets,
                    )
                ),
                units=EXPRESSION_ENERGY_UNITS,
                volatility="live-api",
            )
        )
    return files


DATA_FILES = _build_data_files()
DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

#: Gene symbols that have a manifest entry, i.e. that have ISH data.
AVAILABLE_SYMBOLS = tuple(f.symbol for f in DATA_FILES if isinstance(f, UnionizeFile))

VOLATILITY = """
Every file in this manifest is a live Allen Brain Atlas API query, not an
archive file, because Allen does not publish the structure unionizes as a
static dump (see the module docstring for what was searched). The bytes were
verified stable across repeated requests, but an API can be re-run,
re-normalised or extended at any time.

That is why they are checksummed. A new ISH experiment for one of these
genes, or a pipeline re-run, changes the digest, the fetch stops, and
someone decides whether the receptor maps move - rather than the maps moving
underneath a published result with no record. Re-running
`scripts/build_expression_manifest.py` regenerates the digest table; the
change belongs in a commit that says what moved and why.
"""


def data_file(key):
    """Look up one manifest entry, with a useful error listing the keys."""
    try:
        return DATA_FILES_BY_KEY[key]
    except KeyError:
        raise KeyError(
            "unknown data file %r; manifest %s has: %s"
            % (key, MANIFEST_VERSION, ", ".join(sorted(DATA_FILES_BY_KEY)))
        )


def unionize_file(symbol):
    """The manifest entry for one gene's unionizes.

    A gene that is in `genes.RECEPTOR_GENES` but has no ISH data raises with
    the searched-and-empty record rather than with a list of manifest keys:
    "there is no file for this" and "we looked and Allen has nothing" are
    different facts and only one of them is a typo.
    """
    if symbol in GENES_WITHOUT_DATA_BY_SYMBOL:
        entry = GENES_WITHOUT_DATA_BY_SYMBOL[symbol]
        raise KeyError(
            "%s has no Allen ISH expression data, so there is no file to "
            "load. %s %s" % (symbol, entry["searched"], entry["conclusion"])
        )
    return data_file(unionize_key(symbol))
