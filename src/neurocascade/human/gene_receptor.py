"""The measured reason the mRNA proxy was retired: Hansen et al. 2022's own
gene-expression-to-receptor-density correspondences, ingested as numbers.

P12.4 retires the ISH-expression-to-density path (`legacy/human_expression/`
since this phase). Retirement needs a MEASURED reason, not a preference, and
one exists -- the same group that produced the PET atlas this layer now reads
(`human/receptors.py`) tested the proxy itself:

Hansen JY, Markello RD, Tuominen L, Nørgaard M, Kuzmin E,
Palomero-Gallagher N, Dagher A, Misic B. "Correspondence between gene
expression and neurotransmitter receptor and transporter density in the
human brain." *NeuroImage* 2022;264:119671. PMID 36209794.
doi:10.1016/j.neuroimage.2022.119671.

Across 27 receptors, receptor binding-sites and transporters over 9
transmitter systems, using both PET and autoradiographic density measurements,
they find poor spatial correspondence between gene expression and density --
with exceptions, stated in the abstract verbatim::

    We find poor spatial correspondences between gene expression and density
    for all neurotransmitter receptors and transporters except four
    single-protein metabotropic receptors (5-HT1A, CB1, D2, and MOR).

and their recommendation, also verbatim::

    Altogether, we recommend using direct measures of receptor and
    transporter density when relating neurotransmitter systems to brain
    structure and function.

This module ingests the machine-readable per-gene result --
`results/PETcorrs_scale033_microarray.csv` from
`github.com/netneurolab/hansen_gene-receptor` at one pinned commit -- so the
reason is a committed table asserted by test, not prose in a document. Every
number in `tests/test_human_gene_receptor.py` is read back out of these bytes.

What the deposited columns are, read off the authors' own script
----------------------------------------------------------------

Each row is one gene-receptor pair: Spearman rho between a receptor's
parcellated density and its coding gene's regional AHBA microarray
expression, with a spatial-autocorrelation-preserving permutation p-value
("spin test", Alexander-Bloch 2018; Markello & Misic 2021). Verified against
`code/correlate_expression_density.py` in the pinned commit, because the
column NAMES alone would let a reader get two things wrong:

* ``cortex-rho``/``cortex-pspin`` are computed over the 34 LEFT-hemisphere
  Cammoun-033 cortical parcels (ids 42-75 of `region_info_scale033.xlsx`,
  which carry Desikan-Killiany labels), against a spin null built on that
  same hemisphere: 10,000 rotations, seed 1234, ONE-TAILED.
* ``cortex-pspin`` for a multimeric receptor's subunits is Benjamini-
  Hochberg FDR-corrected ACROSS THAT RECEPTOR'S SUBUNITS ONLY -- all 19
  GABA-A subunits together, all 11 alpha4beta2 subunits together. The
  single-protein receptors' values are UNcorrected. So a GABA-A subunit's
  p_spin and MOR's p_spin are not the same kind of number, and comparing
  their magnitudes naively flatters GABA-A.
* ``subcortex-rho``/``subcortex-pval`` come from a DIFFERENT test: a
  parametric one-sided Spearman (`alternative="greater"`) over the 15
  subcortical parcels, FDR-corrected within multimeric receptors likewise.
  The subcortical column is NOT a spin test, whatever its neighbours are.

The verdict that matters here
-----------------------------

Read the three systems the model carries (model keys in parentheses):

=================== ========== ========== =========================
gene -> density     cortex rho cortex p_spin   verdict
=================== ========== ========== =========================
OPRM1 -> MOR (MOR)      0.846      0.0019 correspondence
DRD2 -> D2 (D2)          0.706      0.0128 correspondence
GABRA1 -> GABAa         0.244      0.5590 none
(GABA_A_BZ)
GABRG2 -> GABAa          0.176      0.6380 none
all 19 GABR* subunits    max |rho|  min p_spin none for every subunit
                         0.408      0.237
=================== ========== ========== =========================

The mRNA proxy was therefore defensible where v1 used it least and
indefensible where v1 used it most: MOR and D2 maps carried no fitted
parameter, while the GABA-A map -- rho 0.244, p_spin 0.559 for its primary
subunit, and no subunit anywhere near significance after the correction the
paper itself applies -- supplied the density factor for the circuit whose
coupling gain was the only fitted parameter in the build (retired with the
rodent bridge in P11.3). `docs/heldout-failure-analysis.md` records that
consequence; this module supplies its numbers.

Corroboration in matched tissue, at the protein level: Zhao L, Mühleisen TW,
Pelzer DI, Burger B, Beins EC, Forstner AJ, Herms S, Hoffmann P, Amunts K,
Palomero-Gallagher N, Cichon S. "Relationships between neurotransmitter
receptor densities and expression levels of their corresponding genes in the
human hippocampus." *NeuroImage* 2023;273:120095. PMID 37030412. Autoradiography
and RNA-seq in the SAME tissue samples (7 subjects, cornu ammonis vs dentate
gyrus, 14 receptors, 43 genes): correlations "vary widely and the mean
correlation strength was weak-to-moderate", concluding receptor densities are
controlled by RNA expression "but also by multiple regionally specific
post-translational factors."

Offline by default, like the rest of `human/`: `load_correspondence()`
defaults to `allow_download=False`; run `scripts/fetch_human_gene_receptor.py`
to fill the cache. Tests run off the byte-identical committed fixture.

Licence, recorded rather than assumed: the ARTICLE is open access under
CC BY 4.0 (its own copyright block). The depositing repository carries no
licence file, and the PET densities it correlates were released by
`netneurolab/hansen_receptors` under **CC BY-NC-SA 4.0** -- the same terms
`human/receptors.py` already records, and the reason this table is committed
as a verbatim copy of the authors' bytes with attribution rather than
restated in a new arrangement.
"""

import csv
import os
import sys

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401

__all__ = [
    # citations
    "CORRESPONDENCE_CITATION",
    "REPOSITORY_COMMIT",
    "LICENCE_NOTE",
    "ZHAO_CITATION",
    "RECOMMENDATION_VERBATIM",
    "FOUR_RECEPTORS_VERBATIM",
    # manifest
    "MANIFEST_VERSION",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DEFAULT_CACHE_DIR",
    "CORRESPONDENCE_UNITS",
    "data_file",
    "cache_path",
    "verify",
    "is_cached",
    "download",
    "ensure",
    "fetch_all",
    "checksum_lines",
    "sha256_of",
    # the table
    "GeneReceptorPair",
    "CorrespondenceTable",
    "read_correspondence",
    "load_correspondence",
    "DEFAULT_CORRESPONDENCE_KEY",
    # the paper's own groupings, and this model's cross-reference
    "PAPER_CRITERION",
    "SIGNIFICANT_AND_LARGE_PET",
    "ROBUST_ACROSS_MODALITIES",
    "SYSTEMS",
    "MODEL_RECEPTOR_KEYS",
    # derived summaries
    "gabr_subunits",
    "gabr_summary",
]


# --- citations --------------------------------------------------------------

CORRESPONDENCE_CITATION = (
    "Hansen JY, Markello RD, Tuominen L, Norgaard M, Kuzmin E, "
    "Palomero-Gallagher N, Dagher A, Misic B. 'Correspondence between gene "
    "expression and neurotransmitter receptor and transporter density in "
    "the human brain.' NeuroImage. 2022;264:119671. PMID 36209794. "
    "doi:10.1016/j.neuroimage.2022.119671."
)

#: One commit of ``netneurolab/hansen_gene-receptor`` (the repository's HEAD
#: at retrieval, 2022-10-13). raw.githubusercontent.com serves immutable
#: bytes per commit, so the URL cannot drift.
REPOSITORY_COMMIT = "1ce5cbda3e0459f6cfc9af999e09d42fa748442f"

LICENCE_NOTE = (
    "Article: open access under CC BY 4.0 (the paper's own copyright "
    "block). Depositing repository github.com/netneurolab/"
    "hansen_gene-receptor: NO licence file present, checked 2026-08-25. "
    "The PET densities the correlations were run against were released by "
    "github.com/netneurolab/hansen_receptors under CC BY-NC-SA 4.0, the "
    "licence human/receptors.py records for the atlas this layer now reads "
    "instead of an mRNA map."
)

ZHAO_CITATION = (
    "Zhao L, Muhleisen TW, Pelzer DI, Burger B, Beins EC, Forstner AJ, "
    "Herms S, Hoffmann P, Amunts K, Palomero-Gallagher N, Cichon S. "
    "'Relationships between neurotransmitter receptor densities and "
    "expression levels of their corresponding genes in the human "
    "hippocampus.' NeuroImage. 2023;273:120095. PMID 37030412. In-vitro "
    "autoradiography and RNA-sequence in the SAME tissue samples (7 "
    "subjects): receptor-density/gene-expression correlations vary widely "
    "with weak-to-moderate mean strength; the authors conclude densities "
    "are controlled by RNA expression 'but also by multiple regionally "
    "specific post-translational factors'."
)

RECOMMENDATION_VERBATIM = (
    "we recommend using direct measures of receptor and transporter "
    "density when relating neurotransmitter systems to brain structure "
    "and function"
)

FOUR_RECEPTORS_VERBATIM = (
    "We find poor spatial correspondences between gene expression and "
    "density for all neurotransmitter receptors and transporters except "
    "four single-protein metabotropic receptors (5-HT1A, CB1, D2, and MOR)"
)


# --- manifest ---------------------------------------------------------------

MANIFEST_VERSION = "2026-08-25"

#: What a human should run when the cache is missing the file.
FETCH_COMMAND = "scripts/fetch_human_gene_receptor.py"

_BASE = (
    "https://raw.githubusercontent.com/netneurolab/hansen_gene-receptor/"
    + REPOSITORY_COMMIT
)

#: Default on-disk cache, shared with the other human layers: <repo>/data/human.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

CORRESPONDENCE_UNITS = (
    "dimensionless Spearman rank correlation (rho) between one gene's AHBA "
    "microarray expression and one receptor's parcellated density, with a "
    "spatial-permutation (spin) p-value; see the module docstring for which "
    "parcels and which multiple-comparison correction each column carries. "
    "A correlation coefficient, NOT a density, and not evidence by itself "
    "that either map is correct"
)

_CORRESPONDENCE_CITATION = " ".join(
    [CORRESPONDENCE_CITATION, LICENCE_NOTE]
)

#: Retrieved and hashed 2026-08-25; the URL was fetched twice and the bytes
#: compared identical before the digest was recorded.
_RETRIEVED = "2026-08-25"


class DataFile(datafetch.DataFile):
    """One external file, defaulting `retrieved` to this manifest's date."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        datafetch.DataFile.__init__(self, *args, **kwargs)


DATA_FILES = [
    DataFile(
        key="hansen_gene-receptor_PETcorrs_scale033_microarray",
        filename="hansen_gene-receptor_PETcorrs_scale033_microarray.csv",
        url=_BASE + "/results/PETcorrs_scale033_microarray.csv",
        sha256=(
            "11daa38a4504d56b163d122b99556bd08ce8cfb022dbe50c8dbf3b902475a616"
        ),
        size_bytes=4363,
        citation=_CORRESPONDENCE_CITATION,
        description=(
            "Per-gene correspondence between AHBA microarray expression and "
            "PET receptor/transporter density, at the Cammoun-033 "
            "parcellation: 46 gene-receptor pairs over 18 receptors and 9 "
            "transmitter systems, as deposited by Hansen et al. 2022 "
            "(NeuroImage 264:119671). Columns: an unnamed pandas index, "
            "'genes' (HGNC symbol), 'receptors' (the paper's receptor "
            "label), 'cortex-rho', 'cortex-pspin', 'subcortex-rho' and "
            "'subcortex-pval'. This is the table whose GABA-A rows are the "
            "measured reason the ISH-expression-to-density path was retired "
            "(P12.4)."
        ),
        units=CORRESPONDENCE_UNITS,
        volatility="static",
    ),
]

DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

#: The one correspondence file this layer ingests.
DEFAULT_CORRESPONDENCE_KEY = DATA_FILES[0].key


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


# --- the paper's own groupings, restated so the numbers can be read ---------

#: What the paper counts as a correspondence in its PET analysis, quoted from
#: its results section: "significant (one-tailed pspin < 0.05) and large
#: (r > 0.5) relationships with the expression of their corresponding genes".
PAPER_CRITERION = "one-tailed p_spin < 0.05 and rho > 0.5"

#: The genes whose pairs meet `PAPER_CRITERION` in THIS deposited table:
#: HTR1A, HTR2A, CNR1, DRD2, HRH3, CHRM1, OPRM1 -- the paper's list of seven,
#: reproduced here so a re-read of the table reproduces the paper's own count.
#: Every other pair fails at least one arm of the criterion; every GABA-A and
#: alpha4beta2 subunit fails both (their p_spins are the FDR-corrected ones).
SIGNIFICANT_AND_LARGE_PET = (
    "HTR1A",   # 5-HT1A   rho 0.881, p_spin 0.0019
    "HTR2A",   # 5-HT2A   rho 0.529, p_spin 0.0133
    "CNR1",    # CB1      rho 0.743, p_spin 0.0077
    "DRD2",    # D2       rho 0.706, p_spin 0.0128
    "HRH3",    # H3       rho 0.628, p_spin 0.0364
    "CHRM1",   # M1       rho 0.629, p_spin 0.0014
    "OPRM1",   # MOR      rho 0.846, p_spin 0.0019
)

#: The paper's HEADLINE set -- the four single-protein metabotropic receptors
#: whose correspondence survives beyond the PET dataset (5-HT2A, H3 and M1 do
#: not replicate autoradiographically; the discussion states M1 explicitly).
#: This is the set the abstract excepts, and the strongest statement the paper
#: makes FOR the proxy: four receptors out of twenty-seven.
ROBUST_ACROSS_MODALITIES = ("HTR1A", "CNR1", "DRD2", "OPRM1")

#: Which paper receptor label belongs to which transmitter system. Our
#: grouping for reading the table, following the paper's nine-system scope;
#: not a measurement and not load-bearing anywhere.
SYSTEMS = {
    "5HT1a": "serotonin",
    "5HT1b": "serotonin",
    "5HT2a": "serotonin",
    "5HT4": "serotonin",
    "5HT6": "serotonin",
    "5HTT": "serotonin",
    "A4B2": "acetylcholine",
    "M1": "acetylcholine",
    "VAChT": "acetylcholine",
    "CB1": "cannabinoid",
    "D1": "dopamine",
    "D2": "dopamine",
    "DAT": "dopamine",
    "GABAa": "gaba_a",
    "H3": "histamine",
    "mGluR5": "glutamate",
    "MOR": "opioid",
    "NET": "noradrenaline",
}

#: Paper receptor label -> this model's PET-atlas receptor key, for the three
#: systems the model carries. Hansen's "GABAa" PET map IS the [11C]flumazenil
#: benzodiazepine-site map (Norgaard et al., n=16) -- the same tracer cohort
#: `human/receptors.py` ingests as "GABA_A_BZ" -- so the correspondence
#: verdict attaches to that system directly. Labels absent here are receptors
#: this model does not yet carry a PET map for.
MODEL_RECEPTOR_KEYS = {
    "MOR": "MOR",
    "D2": "D2",
    "GABAa": "GABA_A_BZ",
}


# --- the table --------------------------------------------------------------


class GeneReceptorPair(object):
    """One gene-receptor row of the deposited correspondence table.

    `cortex_pspin` and `subcortex_pval` are NOT the same kind of number --
    see the module docstring: one is a one-tailed spin test over 34 left-
    hemisphere cortical parcels, the other a parametric one-sided Spearman
    p over 15 subcortical parcels; both are FDR-corrected within multimeric
    receptors' subunits and uncorrected elsewhere.
    """

    __slots__ = (
        "gene",
        "receptor_label",
        "system",
        "cortex_rho",
        "cortex_pspin",
        "subcortex_rho",
        "subcortex_pval",
    )

    def __init__(
        self,
        gene,
        receptor_label,
        cortex_rho,
        cortex_pspin,
        subcortex_rho,
        subcortex_pval,
    ):
        if receptor_label not in SYSTEMS:
            raise ValueError(
                "%r is not a receptor label this module knows; the table has "
                "%s. If the deposit grew a receptor, add it to SYSTEMS "
                "deliberately rather than silently." % (receptor_label,)
            )
        fields = {
            "gene": gene.strip(),
            "receptor_label": receptor_label.strip(),
            "system": SYSTEMS[receptor_label],
            "cortex_rho": float(cortex_rho),
            "cortex_pspin": float(cortex_pspin),
            "subcortex_rho": float(subcortex_rho),
            "subcortex_pval": float(subcortex_pval),
        }
        for name, value in fields.items():
            # Immutable after construction, so every assignment goes through
            # the base class, past the guarding `__setattr__` below.
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError(
            "GeneReceptorPair is immutable; the deposited numbers are not "
            "ours to edit"
        )

    def __repr__(self):
        return "GeneReceptorPair(%s->%s, cortex rho %.3f, p_spin %.4f)" % (
            self.gene,
            self.receptor_label,
            self.cortex_rho,
            self.cortex_pspin,
        )

    @property
    def model_receptor_key(self):
        """This model's PET-atlas key, or None for a receptor it does not carry."""
        return MODEL_RECEPTOR_KEYS.get(self.receptor_label)

    def is_significant_and_large(self):
        """The paper's own PET criterion, applied to this row.

        "Significant (one-tailed pspin < 0.05) and large (r > 0.5)". The r
        arm is signed on purpose: the paper means a strong POSITIVE
        association (its spin test is one-sided greater), so a strong
        negative rho must not pass.
        """
        return self.cortex_pspin < 0.05 and self.cortex_rho > 0.5

    def corresponds(self):
        """True for one of the four receptors the paper's conclusion excepts."""
        return self.gene in ROBUST_ACROSS_MODALITIES


class CorrespondenceTable(object):
    """Every deposited gene-receptor pair, and how to read the verdict."""

    def __init__(self, pairs, citation, dataset_key):
        self.pairs = tuple(pairs)
        self.citation = citation
        self.dataset_key = dataset_key
        genes = [p.gene for p in self.pairs]
        if len(set(genes)) != len(genes):
            raise ValueError("duplicate gene in correspondence table")
        seen = {(p.gene, p.receptor_label) for p in self.pairs}
        if len(seen) != len(self.pairs):
            raise ValueError("duplicate (gene, receptor) pair")

    def __repr__(self):
        return "CorrespondenceTable(%r, %d pairs)" % (
            self.dataset_key,
            len(self.pairs),
        )

    def __len__(self):
        return len(self.pairs)

    @property
    def genes(self):
        return tuple(p.gene for p in self.pairs)

    def for_gene(self, symbol):
        """The one pair for `symbol`. Multimeric subunits appear once each."""
        for pair in self.pairs:
            if pair.gene == symbol:
                return pair
        raise KeyError(
            "%r is not in the deposited table (%d pairs); it is either a "
            "gene the paper did not test or a symbol spelled differently"
            % (symbol, len(self.pairs))
        )

    def for_receptor(self, receptor_label):
        """Every pair against one receptor label, in table order."""
        return [
            p for p in self.pairs if p.receptor_label == receptor_label
        ]

    def for_model_receptor(self, model_key):
        """Every pair against a receptor this model carries, by OUR key."""
        labels = [
            label for label, key in MODEL_RECEPTOR_KEYS.items() if key == model_key
        ]
        if not labels:
            raise KeyError(
                "%r is not a receptor this module can cross-reference; the "
                "table resolves %s"
                % (model_key, sorted(set(MODEL_RECEPTOR_KEYS.values())))
            )
        return [p for p in self.pairs if p.receptor_label == labels[0]]

    def receptor_labels(self):
        """Distinct receptor labels, in first-appearance order."""
        return tuple(dict.fromkeys(p.receptor_label for p in self.pairs))

    def significant_and_large(self):
        """Pairs meeting `PAPER_CRITERION`, in table order."""
        return [p for p in self.pairs if p.is_significant_and_large()]

    def robust(self):
        """Pairs for the four receptors the paper's conclusion excepts."""
        return [p for p in self.pairs if p.corresponds()]


#: Genes prefixing the GABA-A receptor's subunit-encoding genes in the
#: deposited table. The paper: GABA-A "is a pentamer typically composed of
#: three primary subunits (alpha1, beta2, gamma2) but can be built out of a
#: total of nineteen different subunits"; the main text shows the three, and
#: the deposited table carries all nineteen.
_GABR_PREFIX = "GABR"


def gabr_subunits(table):
    """The 19 GABA-A subunit pairs, in table order."""
    return [p for p in table.pairs if p.gene.startswith(_GABR_PREFIX)]


def gabr_summary(table):
    """The GABA-A verdict as numbers: the worst case any subunit can claim.

    Returns `(max_abs_rho_pair, min_pspin_pair)` over the 19 subunits. A
    defender of the mRNA proxy gets to pick whichever subunit looks best;
    these two ARE those best cases, and both still fail the paper's
    criterion -- which is why the summary is stated as extrema rather than
    as GABRA1 alone.
    """
    subunits = gabr_subunits(table)
    if not subunits:
        raise ValueError("no GABR* subunits in this table")
    max_abs = max(subunits, key=lambda p: abs(p.cortex_rho))
    min_p = min(subunits, key=lambda p: p.cortex_pspin)
    return max_abs, min_p


def read_correspondence(path, dataset_key=None):
    """Read the deposited CSV from an explicit path. No cache, no network."""
    if dataset_key is None:
        dataset_key = DEFAULT_CORRESPONDENCE_KEY
    entry = data_file(dataset_key)

    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    required = ["genes", "receptors", "cortex-rho", "cortex-pspin",
                "subcortex-rho", "subcortex-pval"]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(
            "%s: deposited columns missing: %s; header is %s"
            % (path, ", ".join(missing), header)
        )
    index = {name: header.index(name) for name in required}

    pairs = []
    for line, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                "%s: row %d has %d cells, header has %d"
                % (path, line + 2, len(row), len(header))
            )
        try:
            pairs.append(
                GeneReceptorPair(
                    gene=row[index["genes"]].strip(),
                    receptor_label=row[index["receptors"]].strip(),
                    cortex_rho=float(row[index["cortex-rho"]]),
                    cortex_pspin=float(row[index["cortex-pspin"]]),
                    subcortex_rho=float(row[index["subcortex-rho"]]),
                    subcortex_pval=float(row[index["subcortex-pval"]]),
                )
            )
        except ValueError as error:
            raise ValueError("%s: row %d: %s" % (path, line + 2, error))

    if not pairs:
        raise ValueError("%s: no gene-receptor rows" % path)

    return CorrespondenceTable(
        pairs=pairs,
        citation=entry.citation,
        dataset_key=dataset_key,
    )


def load_correspondence(cache_dir=None, allow_download=False):
    """Load the deposited correspondence table from the verified cache.

    `allow_download` is False by default so neither a test nor an ordinary
    import can start a download; `scripts/fetch_human_gene_receptor.py`
    passes True.
    """
    path = ensure(
        DEFAULT_CORRESPONDENCE_KEY,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    return read_correspondence(path)
