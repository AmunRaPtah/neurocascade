"""The absolute-scale anchor: Zilles & Palomero-Gallagher autoradiography.

P12.1's PET atlas (`receptors.py`) constrains WHERE a receptor system acts
and deliberately leaves the scale free. This module is the other half of
that sentence for the one system it can serve: measured ABSOLUTE receptor
density, in fmol/mg protein, from quantitative in vitro receptor
autoradiography of human postmortem cortex.

The source
----------

Zilles K, Palomero-Gallagher N. "Multiple transmitter receptors in regions
and layers of the human cerebral cortex." *Frontiers in Neuroanatomy*
2017;11:78. PMID 28970785. doi:10.3389/fnana.2017.00078. (The article is
indexed as 2017 by PubMed; the deposit's own header writes "Zilles K,
Palomero-Gallagher N (2018)" for the same DOI -- volume 11 straddled the
year boundary. Same paper.)

Machine-readable form: ``github.com/AlGoulas/receptor_principles``,
pinned to one commit in the manifest below, whose data files are NumPy
arrays transcribed from the paper's Supplementary Table 2. The deposit
carries the MIT licence; the article itself is CC BY 4.0. Both are
recorded on every manifest entry.

15 receptors x 44 cytoarchitectonic areas x 3 laminar strata
(supragranular / granular / infragranular), every value a mean density in
fmol/mg protein over three postmortem brains:

    AMPA NMDA kainate GABAa GABAa/BZ GABAb m1 m2 m3 a4b2 a1 a2
    5-HT1a 5-HT2 D1

Two of these are systems the model already carries, keyed here by their
model names (`receptors.RECEPTORS`): ``GABA_A`` (published "GABAa",
[3H]muscimol labelling) and ``GABA_A_BZ`` (published "GABAa/BZ", the
benzodiazepine site). The benzodiazepine site is exactly the system P12.1
could not scale, because the Hansen flumazenil map is VT-derived rather
than a counted density. The remaining thirteen are ingested whole -- it is
one table -- but no scenario consumes them yet.

The sample, stated at the source and not buried
-----------------------------------------------

Three postmortem brains. Ages 72-77 years; two male, one female; no record
of neurological or psychiatric disease; postmortem delay 8-18 h. Coverage
is cortical ONLY: 44 iso- and periarchicortical areas. No subcortical
structure and no proper archicortex was measured. `SAMPLE` carries those
facts as fields, and every accessor that hands out a density also hands
out this provenance, so a downstream number cannot cite the anchor without
citing its n=3 aged cortex-only basis. What an fmol/mg value from three
elderly brains can support is an order-of-magnitude absolute scale with
real regional structure; what it cannot support is individual or
population precision, and nothing here pretends otherwise.

Who gets an absolute scale and who does not
-------------------------------------------

MOR and D2 are NOT in this table. There is no mu-opioid or D2-like
autoradiography in it to borrow, and rule 2 of this codebase forbids
substituting another receptor's measurement for the one being modelled --
including its near neighbours: D1 IS measured here (median 74 fmol/mg
protein), and using it for D2 would be exactly that substitution.
`SCALE_FREE_RECEPTORS` records both refusals with their reasons,
`absolute_anchor()` enforces them, and the reason travels in the exception
so a caller meeting it reads why rather than a bare KeyError.

The join this module deliberately does not build
------------------------------------------------

The 44 areas are Julich-style cytoarchitectonic areas; the connectome
scenarios use Desikan-Killiany labels. No measured crosswalk between them
exists in this deposit, so `AutoradiographicDensityTable.for_connectome_label`
refuses rather than improvising one. Building the join honestly needs the
Julich-Brain probabilistic maps via EBRAINS siibra; the subcortical and
archicortical fingerprints that would extend this anchor live in separate
papers reachable only through siibra, which `SUBCORTICAL_SOURCES` records
per paper as reached / not reached from this environment. Until such a
crosswalk lands, the anchor is consumable per cytoarchitectonic area and
nowhere else.

Offline by default
------------------

Same contract as the rest of ``human/``: nothing touches the network at
import time, loaders default to ``allow_download=False``, a cache miss
raises `MissingCachedFile` naming ``scripts/fetch_human_autoradiography.py``.
Tests run off byte-identical copies of the five verified ``.npy`` files in
``fixtures/``. The sixth file, the paper's Supplementary Table 2 xlsx, is
NOT shipped (same reasoning as P12.1's parcel-identity xlsx: fixtures must
stay readable without openpyxl); when the cache is present a test parses
it and re-checks all 1,980 stratum-level means against the arrays
cell-for-cell.

One real feature of the data, kept rather than cleaned up: area 24 --
periarchicortical agranular cingulate cortex -- has NO granular stratum.
Supplementary Table 2 prints '--' there and the .npy files encode it as 0.
A zero would assert a measured density of zero binding sites, which is not
what '--' says, so `for_area("24", ..., stratum="granular")` raises
`StratumAbsent` instead of returning 0. These are the only zeros in the
deposit: 15 receptors x 1 area x 1 stratum, and a test pins that count.

    from neurocascade import human

    table = human.autoradiography.load_table()
    v = table.for_area("V1", "GABA_A_BZ", stratum="granular")
    float(v)          # 3842.0 fmol/mg protein, explicitly
    table.sample      # the n=3, ages 72-77, cortex-only provenance

    human.autoradiography.absolute_anchor("MOR")   # refuses, with the reason
"""

import os
import sys

import numpy as np

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401

__all__ = [
    # citations and licences
    "ZILLES_CITATION",
    "ARTICLE_LICENCE",
    "DEPOSIT_CITATION",
    "DEPOSIT_LICENCE",
    # sample limitations
    "SampleProvenance",
    "SAMPLE",
    # units and types
    "ABSOLUTE_UNITS",
    "AbsoluteDensity",
    "AutoradiographicDensityTable",
    # strata and the agranular absence
    "STRATA",
    "StratumAbsent",
    "AGRANULAR_ABSENCES",
    # receptors
    "AutoradiographicReceptor",
    "RECEPTORS",
    "SCALE_FREE_RECEPTORS",
    "ScaleFreeReceptor",
    "scale_free_reason",
    "receptors_covered",
    # subcortical/archicortical sources: reached vs not reached
    "SUBCORTICAL_SOURCES",
    # manifest and fetch
    "MANIFEST_VERSION",
    "REPOSITORY_COMMIT",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DEFAULT_CACHE_DIR",
    "data_file",
    "cache_path",
    "verify",
    "is_cached",
    "download",
    "ensure",
    "fetch_all",
    "checksum_lines",
    "ChecksumMismatch",
    "MissingCachedFile",
    # loaders and verification anchor
    "load_stratum_array",
    "load_names",
    "load_table",
    "read_published_supplementary_table",
]

# --- citations --------------------------------------------------------------


def _cited(*parts):
    return " ".join(parts)


#: The paper this table comes from. PubMed indexes it as 2017;11:78; the
#: deposit's own header writes "(2018)" for the same DOI.
ZILLES_CITATION = _cited(
    "Zilles K, Palomero-Gallagher N. 'Multiple transmitter receptors in "
    "regions and layers of the human cerebral cortex.' Frontiers in "
    "Neuroanatomy 2017;11:78. PMID 28970785. doi:10.3389/fnana.2017.00078.",
    "Quantitative in vitro receptor autoradiography of human postmortem",
    "cerebral cortex: 15 receptors x 44 iso- and periarchicortical areas x 3",
    "laminar strata, mean densities in fmol/mg protein.",
)

ARTICLE_LICENCE = (
    "Creative Commons Attribution 4.0 (CC BY): the article's own stated "
    "licence ('The use, distribution or reproduction in other forums is "
    "permitted, provided the original author(s) and the copyright owner(s) "
    "are credited')."
)

DEPOSIT_CITATION = (
    "Machine-readable data: github.com/AlGoulas/receptor_principles "
    "(Goulas A et al.), recept_data/ NumPy arrays transcribed from the "
    "paper's Supplementary Table 2."
)

DEPOSIT_LICENCE = (
    "MIT License, Copyright (c) 2020 AlGoulas: the LICENSE file of the "
    "receptor_principles repository the .npy files come from."
)

#: Autoradiographic protocols per the paper's Methods: tritiated ligands
#: "according to previously published receptor protocols" (Zilles et al.
#: 2002a,b; Graebenitz et al. 2011; Palomero-Gallagher & Zilles 2017b).
#: The per-receptor ligand table itself (Supplementary Table S1) is not
#: part of the machine-readable deposit and is not reachable in full text;
#: the main text names [3H]muscimol for GABA-A. Nothing here depends on
#: which ligand labelled which site beyond the site identities the
#: receptor names already carry.
PROTOCOL_NOTE = (
    "Ligands: tritiated, per the paper's cited protocols (Zilles et al. "
    "2002a,b; Graebenitz et al. 2011; Palomero-Gallagher & Zilles 2017b); "
    "[3H]muscimol named in the text for GABA-A. Supplementary Table S1's "
    "per-receptor ligand list is not in the machine-readable deposit."
)

# --- sample limitations -----------------------------------------------------


class SampleProvenance(object):
    """Who was measured, stated as fields rather than prose.

    Every field exists so a downstream consumer can be refused on it: an
    fmol/mg anchor from three elderly postmortem cortices is an
    order-of-magnitude scale with real regional structure, and the fields
    below are what any statement built on it inherits.
    """

    __slots__ = (
        "n_donors", "ages", "sex", "state", "postmortem_delay", "coverage",
        "note",
    )

    def __init__(self, n_donors, ages, sex, state, postmortem_delay,
                 coverage, note):
        self.n_donors = int(n_donors)
        self.ages = ages
        self.sex = sex
        self.state = state
        self.postmortem_delay = postmortem_delay
        self.coverage = coverage
        self.note = note

    def __repr__(self):
        return "SampleProvenance(n=%d, ages=%s, %s)" % (
            self.n_donors, self.ages, self.coverage[:40] + "...",
        )

    def __str__(self):
        return (
            "%d postmortem brains, ages %s (%s); %s; postmortem delay %s; "
            "coverage: %s" % (
                self.n_donors, self.ages, self.sex, self.state,
                self.postmortem_delay, self.coverage,
            )
        )


#: The sample behind every number in this module, verbatim from the paper:
#: "Brains of three donors without any record of neurological or psychiatric
#: diseases"; ages 72-77; two male, one female; postmortem delay 8-18 h;
#: 44 iso- and periarchicortical areas, i.e. cortex only.
SAMPLE = SampleProvenance(
    n_donors=3,
    ages="72-77 years",
    sex="two male, one female",
    state="postmortem, no record of neurological or psychiatric disease",
    postmortem_delay="8-18 h",
    coverage=(
        "44 iso- and periarchicortical areas of the cerebral cortex; NO "
        "subcortical structure and NO proper archicortex was measured"
    ),
    note=(
        "An absolute scale from n=3 aged, cortex-only postmortem tissue "
        "supports order-of-magnitude anchoring with real regional "
        "structure, not individual or population precision"
    ),
)


# --- manifest ---------------------------------------------------------------

MANIFEST_VERSION = "2026-08-25"

#: What a human should run when the cache is missing a file.
FETCH_COMMAND = "scripts/fetch_human_autoradiography.py"

#: One commit of ``AlGoulas/receptor_principles`` (the repository HEAD at
#: retrieval, last pushed 2021-04-08). raw.githubusercontent.com serves
#: immutable bytes per commit; each URL below was fetched twice and the
#: bytes compared identical before the digest was recorded.
REPOSITORY_COMMIT = "1a8b8714271d99ffe62f81f93ac42f3e57c952f1"

_BASE = (
    "https://raw.githubusercontent.com/AlGoulas/receptor_principles/"
    + REPOSITORY_COMMIT + "/recept_data/"
)

_RETRIEVED = "2026-08-25"

#: Default on-disk cache, shared with the other human layers: <repo>/data/human.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

ABSOLUTE_UNITS = (
    "fmol/mg protein, mean specific binding measured by quantitative in "
    "vitro receptor autoradiography of human postmortem cortex. An "
    "ABSOLUTE binding-site density, unlike the z-scored relative maps in "
    "receptors.RELATIVE_UNITS -- but bounded by this module's SAMPLE-scale "
    "limits: n=3 donors, ages 72-77, cortical areas only"
)

_CITATION_BLOCK = " ".join([
    ZILLES_CITATION, ARTICLE_LICENCE, DEPOSIT_CITATION, DEPOSIT_LICENCE,
])

_XLSX_KEY = "zilles2017_supplementary_table_2_xlsx"


class DataFile(datafetch.DataFile):
    """One external file, defaulting `retrieved` to this manifest's date."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        datafetch.DataFile.__init__(self, *args, **kwargs)


#: (key, cache filename, upstream path, sha256, size, description)
_DATA = (
    (
        "zilles2017_ReceptorNames_npy",
        "zilles2017_ReceptorNames.npy",
        "ReceptorNames.npy",
        "f570401cbe1ad960a44b1c4f0f7db30466c60cb77aa13c5dfedff75d375c9171",
        608,
        "The 15 receptor labels, in the column order of every ReceptData "
        "array: AMPA, NMDA, kainate, GABAa, GABAa/BZ, GABAb, m1, m2, m3, "
        "a4b2, a1, a2, 5-HT1a, 5-HT2, D1.",
    ),
    (
        "zilles2017_RegionNames_npy",
        "zilles2017_RegionNames.npy",
        "RegionNames.npy",
        "c2e830b2885cccc4847abb3d0b9a9b0a9661840ba6742c6533b4e50cc6955440",
        656,
        "The 44 cytoarchitectonic area labels, in the row order of every "
        "ReceptData array: 1, 2, 3a, 3b, V1, V2d, V2v, V3A, V3d, V3v, V4v, "
        "FG1, FG2, 37B, 37L, 37M, 41, 42, 20, 21, 22, 36, 38, 5L, 5M, PGa, "
        "PGp, PFt, PFm, 24, 32, 23, 31, 4, 6, 8, 9, 10L, 10M, 11, 44, 45, "
        "46, 47.",
    ),
    (
        "zilles2017_ReceptData_S_npy",
        "zilles2017_ReceptData_S.npy",
        "ReceptData_S.npy",
        "876b00a55802bba9921b72865ad7b115a6c8c01abef07eb1f0ccc9eba50f15b4",
        1448,
        "SUPRAGRANDULAR stratum: 44 areas x 15 receptors, uint16, mean "
        "density in fmol/mg protein.",
    ),
    (
        "zilles2017_ReceptData_G_npy",
        "zilles2017_ReceptData_G.npy",
        "ReceptData_G.npy",
        "66517300cfc5d40bf1a8f792cb35fa19b9405e3b95922d2b0102569bfa423dd9",
        1448,
        "GRANULAR stratum: 44 areas x 15 receptors, uint16, mean density "
        "in fmol/mg protein. Area 24 (periarchicortical agranular cingulate "
        "cortex) has NO granular stratum: the deposit encodes its absence "
        "as 0, Supplementary Table 2 prints '--', and this module raises "
        "`StratumAbsent` rather than reading a zero.",
    ),
    (
        "zilles2017_ReceptData_I_npy",
        "zilles2017_ReceptData_I.npy",
        "ReceptData_I.npy",
        "ccf010fd061256e438b7fa33d48ff82efd3178f59af5db25f4336d5ae7acba5a",
        1448,
        "INFRAGRANULAR stratum: 44 areas x 15 receptors, uint16, mean "
        "density in fmol/mg protein.",
    ),
    (
        _XLSX_KEY,
        "zilles2017_ReceptData.xlsx",
        "ReceptData.xlsx",
        "d64bf50ff74f084f618de87a7eeb76f8bfc525526b5dcbf5b51240e3d87b463a",
        70718,
        "The paper's Supplementary Table 2 as deposited: per receptor, "
        "supragranular/granular/infragranular/all-layers means plus SD and "
        "CV over the 44 areas. NOT consumed by any loader; read only by "
        "`read_published_supplementary_table` to verify the .npy arrays "
        "cell-for-cell (1980 stratum-level means). SDs are stored NEGATIVE "
        "in the spreadsheet (floating-bar chart convention; CV = |SD|/mean "
        "holds). Its 'all layers' column is thickness-weighted, so it does "
        "not equal the unweighted stratum mean this module derives.",
    ),
)

assert len(_DATA) == 6


def _entry(key, filename, upstream_path, sha, size, description):
    return DataFile(
        key=key,
        filename=filename,
        url=_BASE + upstream_path,
        sha256=sha,
        size_bytes=size,
        citation=_CITATION_BLOCK,
        description=description,
        units=(
            ABSOLUTE_UNITS if key != _XLSX_KEY
            else "means in fmol/mg protein with SD and CV columns; see "
                 "autoradiography.SAMPLE for the sample they summarise"
        ),
    )


DATA_FILES = [_entry(*args) for args in _DATA]

DATA_FILES_BY_KEY = {f.key: f for f in DATA_FILES}

_FETCHER = datafetch.Fetcher(
    manifest=sys.modules[__name__],
    default_cache_dir=DEFAULT_CACHE_DIR,
    user_agent="neurocascade/human",
)


def data_file(key):
    """Look up one manifest entry, naming the alternatives on a miss."""
    try:
        return DATA_FILES_BY_KEY[key]
    except KeyError:
        raise KeyError(
            "unknown data file %r; manifest %s has: %s"
            % (key, MANIFEST_VERSION, ", ".join(sorted(DATA_FILES_BY_KEY)))
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


# --- strata, and the one absent stratum -------------------------------------

STRATA = ("supragranular", "granular", "infragranular")

_STRATUM_LETTERS = {
    "supragranular": "S",
    "granular": "G",
    "infragranular": "I",
}

#: Strata the publication reports no measurement for, and why. Keyed
#: (area label, stratum). Area 24 is periarchicortical agranular cingulate
#: cortex: "Only the agranular motor areas 4 and 6, as well as the
#: agranular cingulate area 24 did not follow this principal segregation
#: ... a typical receptor pattern indicating the presence of an inner
#: granular layer IV could not be demonstrated" -- and for area 24 the
#: granular column is '--' outright. The deposit encodes each absence as 0;
#: this module refuses to read one as a density.
AGRANULAR_ABSENCES = {
    ("24", "granular"): (
        "area 24 is periarchicortical agranular cingulate cortex: it has "
        "no inner granular layer IV, the publication reports '--' there, "
        "and the .npy deposit encodes the absence as 0. Zero would assert "
        "a measured density of zero binding sites, which is not what '--' "
        "says"
    ),
}


class StratumAbsent(KeyError):
    """A stratum the publication reports no measurement for (see
    `AGRANULAR_ABSENCES`). Not a zero density: the stratum does not exist
    in that area."""

    def __init__(self, area, stratum, reason):
        KeyError.__init__(
            self,
            "%s has no %s stratum measurement in this source: %s"
            % (area, stratum, reason),
        )
        self.area = area
        self.stratum = stratum
        self.reason = reason


# --- the receptor registry ---------------------------------------------------


class AutoradiographicReceptor(object):
    """One receptor's column of the table, under both of its names.

    ``key`` is how this module and the scenarios address it; for the two
    systems the model already carries (`receptors.RECEPTORS`) it is the
    model key ("GABA_A", "GABA_A_BZ"). ``published_name`` is the verbatim
    string in the deposit's ReceptorNames array, kept alongside so the
    transcription is checkable against the source rather than remembered.
    """

    __slots__ = ("key", "published_name", "family", "model_system_note")

    def __init__(self, key, published_name, family, model_system_note=""):
        self.key = key
        self.published_name = published_name
        self.family = family
        #: What the model does with it today, said plainly.
        self.model_system_note = model_system_note

    def __repr__(self):
        return "AutoradiographicReceptor(%r, published %r, %s)" % (
            self.key, self.published_name, self.family,
        )


_GABA_A_BZ_NOTE = (
    "THE system P12.1 could not scale: the Hansen flumazenil map is "
    "VT-derived ('Bmax' per Table S3, converted 'using autoradiography "
    "densities'), so this table supplies the absolute fmol/mg protein "
    "scale the PET atlas cannot"
)
_GABA_A_NOTE = (
    "the GABA-A site itself ([3H]muscimol), measured here but carried by "
    "no PET tracer in the P12.1 atlas"
)

RECEPTORS = {
    "AMPA": AutoradiographicReceptor("AMPA", "AMPA", "glutamatergic"),
    "NMDA": AutoradiographicReceptor("NMDA", "NMDA", "glutamatergic"),
    "kainate": AutoradiographicReceptor("kainate", "kainate", "glutamatergic"),
    "GABA_A": AutoradiographicReceptor(
        "GABA_A", "GABAa", "GABAergic", model_system_note=_GABA_A_NOTE,
    ),
    "GABA_A_BZ": AutoradiographicReceptor(
        "GABA_A_BZ", "GABAa/BZ", "GABAergic", model_system_note=_GABA_A_BZ_NOTE,
    ),
    "GABAb": AutoradiographicReceptor("GABAb", "GABAb", "GABAergic"),
    "m1": AutoradiographicReceptor("m1", "m1", "cholinergic muscarinic"),
    "m2": AutoradiographicReceptor("m2", "m2", "cholinergic muscarinic"),
    "m3": AutoradiographicReceptor("m3", "m3", "cholinergic muscarinic"),
    "a4b2": AutoradiographicReceptor("a4b2", "a4b2", "cholinergic nicotinic"),
    "a1": AutoradiographicReceptor("a1", "a1", "noradrenergic"),
    "a2": AutoradiographicReceptor("a2", "a2", "noradrenergic"),
    "5-HT1a": AutoradiographicReceptor("5-HT1a", "5-HT1a", "serotonergic"),
    "5-HT2": AutoradiographicReceptor("5-HT2", "5-HT2", "serotonergic"),
    "D1": AutoradiographicReceptor(
        "D1", "D1", "dopaminergic",
        model_system_note=(
            "measured; D2 is NOT, which is why SCALE_FREE_RECEPTORS['D2'] "
            "names this column explicitly: using D1's densities for D2 "
            "would be substituting another receptor's measurement"
        ),
    ),
}

_PUBLISHED_NAMES = tuple(r.published_name for r in RECEPTORS.values())

assert len(RECEPTORS) == 15
assert len(set(_PUBLISHED_NAMES)) == 15


# --- who stays scale-free ----------------------------------------------------

#: Model receptor systems this table does NOT cover, with the reason. The
#: refusal is the deliverable: an fmol/mg number for MOR or D2 cannot come
#: from this source, and inventing one from a neighbouring receptor's
#: column is the substitution this project forbids.
SCALE_FREE_RECEPTORS = {
    "MOR": (
        "not measured by this source: Zilles & Palomero-Gallagher "
        "autoradiographed 15 receptors (AMPA, NMDA, kainate, GABA-A, "
        "GABA-A/BZ, GABA-B, m1-m3, a4b2, a1, a2, 5-HT1A, 5-HT2, D1) and "
        "none of them is the mu-opioid receptor. No absolute fmol/mg "
        "anchor exists here to borrow, so MOR stays a scale-free "
        "parameter, exactly as receptors.absolute_density() states"
    ),
    "D2": (
        "not measured by this source: the dopaminergic column is D1 "
        "(median 74 fmol/mg protein across areas and strata), not D2. D1 "
        "being present makes the refusal sharper, not weaker: borrowing "
        "its column for D2 would substitute one receptor's density for "
        "another's, which rule 2 of this codebase forbids even between "
        "sister receptors. D2 therefore stays a scale-free parameter"
    ),
}


class ScaleFreeReceptor(TypeError):
    """Raised by `absolute_anchor` for a receptor this source does not
    cover. Carries `.reason`, so the refusal explains itself."""

    def __init__(self, receptor_key, reason):
        TypeError.__init__(
            self,
            "%s has no absolute-density anchor in this source and stays a "
            "scale-free parameter: %s" % (receptor_key, reason),
        )
        self.receptor_key = receptor_key
        self.reason = reason


def scale_free_reason(receptor_key):
    """Why one model system has no absolute anchor here (raises KeyError
    for anything neither covered nor recorded)."""
    try:
        return SCALE_FREE_RECEPTORS[receptor_key]
    except KeyError:
        raise KeyError(
            "%r is not recorded as scale-free; covered receptors are %s "
            "and recorded refusals are %s"
            % (receptor_key, ", ".join(receptors_covered()),
               ", ".join(sorted(SCALE_FREE_RECEPTORS)))
        )


def receptors_covered():
    """The receptor keys this table gives absolute densities for."""
    return tuple(RECEPTORS)


# --- the searched-and-not-reached record -------------------------------------

#: Subcortical and archicortical receptor fingerprints exist, but in
#: separate papers whose numeric tables are not in any machine-readable
#: deposit this environment could reach. Recorded per paper, the house way:
#: what was tried, what happened, and what would have been needed.
SUBCORTICAL_SOURCES = {
    "Kedo2018_amygdala_PMID29188378": {
        "citation": (
            "Kedo M, Zilles K, Palomero-Gallagher N. 'Receptor-driven, "
            "multimodal mapping of the human amygdala.' Brain Struct Funct "
            "2018;223(4):1637-1664. PMID 29188378. "
            "doi:10.1007/s00429-017-1577-x"
        ),
        "status": "searched and NOT reached",
        "what_was_tried": (
            "EBRAINS siibra API endpoints (siibra-apistable.apps.hbp.eu: "
            "DNS resolves, TCP connect times out at 20 s and 90 s; "
            "siibra-api.apps.hbp.eu, api.siibra.eu and "
            "siibra-api.opensource.ebrains.eu likewise unreachable); no "
            "PMC full text (Springer paywalled); GitHub search found no "
            "mirror of the numeric tables"
        ),
        "consequence": (
            "no amygdalar fmol/mg anchor is ingested; amygdalar densities "
            "stay PET-relative only"
        ),
    },
    "Zhao2023_hippocampus_PMID37030412": {
        "citation": (
            "Zhao T, Cichon J, ... 'Relationships between neurotransmitter "
            "receptor densities and expression levels in the human "
            "hippocampus.' NeuroImage 2023;273:120095. PMID 37030412. "
            "doi:10.1016/j.neuroimage.2023.120095"
        ),
        "status": "full text reached; numbers NOT reachable in machine-readable form",
        "what_was_tried": (
            "PMC10167541 full text retrieved: receptor densities are "
            "presented as autoradiographs with colour scales, not as a "
            "numeric table; the siibra route to the underlying values is "
            "unreachable as above"
        ),
        "consequence": (
            "no hippocampal fmol/mg anchor is ingested; hippocampal "
            "densities stay PET-relative only"
        ),
    },
    "PalomeroGallagher2009_cingulate_PMID19034899": {
        "citation": (
            "Palomero-Gallagher N, Vogt BA, Schleicher A, Mayberg HS, "
            "Zilles K. 'Receptor architecture of human cingulate cortex: "
            "evaluation of the four-region neurobiological model.' Hum "
            "Brain Mapp 2009;30(8):2552-2575. PMID 19034899. "
            "doi:10.1002/hbm.20667"
        ),
        "status": "full text reached; numbers NOT reachable as a complete table",
        "what_was_tried": (
            "PMC6870973 full text retrieved: absolute densities appear in "
            "figures and isolated in-text values (e.g. benzodiazepine "
            "sites up to 3195 fmol/mg protein), not as a complete per-area "
            "table; the siibra route is unreachable as above"
        ),
        "consequence": (
            "no cingulate-specific extension beyond this table's areas "
            "24/32/23/31 is ingested"
        ),
    },
}


# --- the absolute-density type ----------------------------------------------


class AbsoluteDensity(object):
    """One area-stratum-receptor mean density, in fmol/mg protein.

    Absolute, unlike `receptors.RelativeDensityValue`: ordering against
    numbers is meaningful across this whole table (one measurement family,
    one unit system), and averaging within a receptor produces another
    density. Multiplying by a mechanism gain returns a plain float -- a
    coupling coefficient, no longer a density. Multiplying two densities
    together is refused: the product of two concentrations is not a
    concentration. ``float(v)`` is the documented exit.
    """

    __slots__ = ("_value", "_area", "_stratum", "_receptor_key")

    def __init__(self, value, area, stratum, receptor_key):
        self._value = float(value)
        self._area = area
        self._stratum = stratum
        self._receptor_key = receptor_key

    @property
    def units(self):
        return ABSOLUTE_UNITS

    @property
    def area(self):
        return self._area

    @property
    def stratum(self):
        return self._stratum

    @property
    def receptor_key(self):
        return self._receptor_key

    def __float__(self):
        return self._value

    def __repr__(self):
        return "AbsoluteDensity(%g fmol/mg protein, %s/%s/%s)" % (
            self._value, self._receptor_key, self._area, self._stratum,
        )

    def __mul__(self, other):
        if isinstance(other, (AbsoluteDensity,)):
            raise TypeError(
                "cannot multiply two densities together: the product of two "
                "concentrations is not a concentration. Multiplying by a "
                "scalar gain is supported and returns a plain float"
            )
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self._value * float(other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, AbsoluteDensity):
            # A ratio of two densities of ONE receptor is dimensionless
            # laminar/areal structure -- legitimate, e.g. SG/IG ratios.
            return self._value / other._value
        if not isinstance(other, (int, float)):
            return NotImplemented
        return AbsoluteDensity(
            self._value / float(other),
            self._area, self._stratum, self._receptor_key,
        )

    def __add__(self, other):
        if not isinstance(other, AbsoluteDensity):
            return NotImplemented
        return AbsoluteDensity(
            self._value + other._value,
            "%s+%s" % (self._area, other._area),
            "%s+%s" % (self._stratum, other._stratum),
            self._require_same_receptor(other, "add"),
        )

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, AbsoluteDensity):
            return NotImplemented
        return AbsoluteDensity(
            self._value - other._value,
            "%s-%s" % (self._area, other._area),
            "%s-%s" % (self._stratum, other._stratum),
            self._require_same_receptor(other, "subtract"),
        )

    def _require_same_receptor(self, other, op):
        if other._receptor_key != self._receptor_key:
            raise TypeError(
                "cannot %s densities of two different receptors (%s vs %s): "
                "their sum or difference is no receptor's density. Average "
                "within one receptor, or compare them as floats"
                % (op, self._receptor_key, other._receptor_key)
            )
        return self._receptor_key

    def __neg__(self):
        raise TypeError(
            "negating a receptor density has no physical meaning; if you "
            "want a difference, subtract two densities of the SAME receptor"
        )

    def _comparable(self, other):
        if isinstance(other, AbsoluteDensity):
            return other._value
        if isinstance(other, (int, float)):
            return float(other)
        return NotImplemented

    def __eq__(self, other):
        value = self._comparable(other)
        if value is NotImplemented:
            return NotImplemented
        return self._value == value

    def __ne__(self, other):
        value = self._comparable(other)
        if value is NotImplemented:
            return NotImplemented
        return self._value != value

    def __lt__(self, other):
        value = self._comparable(other)
        if value is NotImplemented:
            return NotImplemented
        return self._value < value

    def __le__(self, other):
        value = self._comparable(other)
        if value is NotImplemented:
            return NotImplemented
        return self._value <= value

    def __gt__(self, other):
        value = self._comparable(other)
        if value is NotImplemented:
            return NotImplemented
        return self._value > value

    def __ge__(self, other):
        value = self._comparable(other)
        if value is NotImplemented:
            return NotImplemented
        return self._value >= value

    # Values are dict VALUES keyed by (area, stratum); never keys.
    __hash__ = None


# --- the table ---------------------------------------------------------------


class AutoradiographicDensityTable(object):
    """The whole measured table: 44 areas x 15 receptors x 3 strata.

    Constructed only by this module's loaders. Values leave through
    `for_area` / `strata_of` / `anchor` as `AbsoluteDensity`; the raw
    arrays stay private.
    """

    def __init__(self, strata_arrays, area_labels, receptor_keys, citation):
        area_labels = [str(a) for a in area_labels]
        receptor_keys = list(receptor_keys)
        if len(area_labels) != 44 or len(set(area_labels)) != 44:
            raise ValueError(
                "expected 44 distinct area labels, got %d" % len(area_labels)
            )
        if len(receptor_keys) != 15 or len(set(receptor_keys)) != 15:
            raise ValueError(
                "expected 15 distinct receptor keys, got %d" % len(receptor_keys)
            )
        if set(receptor_keys) != set(RECEPTORS):
            raise ValueError(
                "receptor keys must match this module's registry; expected "
                "%s, got %s" % (sorted(RECEPTORS), sorted(receptor_keys))
            )
        normalized = {}
        for stratum in STRATA:
            array = np.asarray(strata_arrays[stratum], dtype=float)
            if array.shape != (44, 15):
                raise ValueError(
                    "%s: expected shape (44, 15), got %s"
                    % (stratum, array.shape)
                )
            if not np.all(np.isfinite(array)):
                raise ValueError("%s: non-finite cell is a parse failure" % stratum)
            if np.any(array < 0.0):
                raise ValueError("%s: negative density" % stratum)
            allowed = np.zeros(array.shape, dtype=bool)
            for (area, absent_stratum) in AGRANULAR_ABSENCES:
                if absent_stratum == stratum:
                    allowed[area_labels.index(area), :] = True
            zeros = array == 0.0
            unexpected = zeros & ~allowed
            if np.any(unexpected):
                where = [
                    (area_labels[r], receptor_keys[c])
                    for r, c in zip(*np.nonzero(unexpected))
                ]
                raise ValueError(
                    "zero cells outside the registered absences %s: %s -- "
                    "zero would assert the receptor is absent from that "
                    "stratum, which is a finding, not a parse artefact"
                    % (sorted(AGRANULAR_ABSENCES), where[:5])
                )
            missing_absence = allowed & ~zeros
            if np.any(missing_absence):
                raise ValueError(
                    "%s: registered absence cells are not zero in the "
                    "deposit; re-check the transcription" % stratum
                )
            normalized[stratum] = array

        object.__setattr__(self, "_strata", normalized)
        object.__setattr__(self, "_areas", tuple(area_labels))
        object.__setattr__(self, "_receptors", tuple(receptor_keys))
        self.citation = citation
        self.sample = SAMPLE

    def __setattr__(self, name, value):
        if name in ("_strata", "_areas", "_receptors"):
            raise AttributeError("table contents are fixed at construction")
        object.__setattr__(self, name, value)

    def __repr__(self):
        return "AutoradiographicDensityTable(%d areas x %d receptors x %d strata)" % (
            len(self._areas), len(self._receptors), len(STRATA),
        )

    @property
    def units(self):
        return ABSOLUTE_UNITS

    @property
    def areas(self):
        """The 44 published area labels, in deposit row order."""
        return self._areas

    @property
    def receptors(self):
        """The 15 receptor keys, in deposit column order."""
        return self._receptors

    @property
    def strata(self):
        return STRATA

    def _index(self, area, receptor_key):
        if receptor_key not in RECEPTORS:
            raise KeyError(
                "%r is not one of this table's receptors: %s"
                % (receptor_key, ", ".join(self._receptors))
            )
        try:
            return self._areas.index(str(area)), self._receptors.index(receptor_key)
        except ValueError:
            raise KeyError(
                "%r is not one of this table's 44 areas: %s"
                % (area, ", ".join(self._areas))
            )

    def for_area(self, area, receptor_key, stratum):
        """One measured mean density, as an `AbsoluteDensity`.

        Raises `StratumAbsent` for the strata in `AGRANULAR_ABSENCES`
        rather than returning a zero.
        """
        if stratum not in STRATA:
            raise KeyError(
                "stratum %r is not one of %s" % (stratum, ", ".join(STRATA))
            )
        reason = AGRANULAR_ABSENCES.get((str(area), stratum))
        if reason is not None:
            raise StratumAbsent(area, stratum, reason)
        r, c = self._index(area, receptor_key)
        return AbsoluteDensity(
            self._strata[stratum][r, c], str(area), stratum, receptor_key,
        )

    def strata_of(self, area, receptor_key):
        """`{stratum: AbsoluteDensity}` for every MEASURED stratum."""
        result = {}
        for stratum in STRATA:
            try:
                result[stratum] = self.for_area(area, receptor_key, stratum)
            except StratumAbsent:
                continue
        return result

    def layer_mean(self, area, receptor_key):
        """Unweighted mean over the MEASURED strata, as `AbsoluteDensity`.

        DERIVED, and labelled so: this module's arithmetic over the
        published means, not a published quantity. The paper's own
        "all layers" column (Supplementary Table 2) is a
        thickness-weighted average over single layers and differs from it
        -- for V1 GABA-A/BZ this gives 3095.67 while the publication's
        all-layers mean is 3525 -- so never present one as the other.
        """
        measured = self.strata_of(area, receptor_key)
        if not measured:
            raise StratumAbsent(area, "all", "no stratum measured for %r" % area)
        total = sum(float(v) for v in measured.values())
        return AbsoluteDensity(
            total / len(measured), str(area), "layer-mean(unweighted)", receptor_key,
        )

    def anchor(self, receptor_key):
        """`{area: {stratum: AbsoluteDensity}}` for one receptor.

        The absolute-scale product this module exists to hand over: for
        GABA_A_BZ it is the fmol/mg protein scale the P12.1 PET map
        cannot carry. Areas keep the published cytoarchitectonic labels;
        see `for_connectome_label` for why there is no Desikan-Killiany
        view.
        """
        if receptor_key in SCALE_FREE_RECEPTORS:
            raise ScaleFreeReceptor(receptor_key, SCALE_FREE_RECEPTORS[receptor_key])
        if receptor_key not in RECEPTORS:
            raise KeyError(
                "%r is neither covered nor recorded scale-free; covered: %s; "
                "recorded scale-free: %s"
                % (receptor_key, ", ".join(receptors_covered()),
                   ", ".join(sorted(SCALE_FREE_RECEPTORS)))
            )
        return {
            area: self.strata_of(area, receptor_key) for area in self._areas
        }

    def top(self, receptor_key, stratum, n=5):
        """The n highest areas for one receptor/stratum."""
        values = [
            (area, self.for_area(area, receptor_key, stratum))
            for area in self._areas
            if (str(area), stratum) not in AGRANULAR_ABSENCES
        ]
        values.sort(key=lambda pair: -float(pair[1]))
        return values[:n]

    def for_connectome_label(self, label):
        """There is no such lookup. This method refuses, loudly.

        The 44 areas are Julich-style cytoarchitectonic areas; the
        connectome scenarios use Desikan-Killiany labels. No measured
        crosswalk between them exists in this deposit, and improvising
        one from label similarity would be anatomy invented for the sake
        of a join. Building it honestly needs the Julich-Brain
        probabilistic maps via EBRAINS siibra (searched, not reached from
        this environment -- see SUBCORTICAL_SOURCES). Until then this
        anchor is consumable per cytoarchitectonic area and nowhere else.
        """
        raise NotImplementedError(
            "no measured crosswalk exists between this table's 44 "
            "cytoarchitectonic areas and the Desikan-Killiany label %r, so "
            "no absolute density can be produced for it. The join needs "
            "Julich-Brain probabilistic maps via EBRAINS siibra; "
            "improvising one here would be invented anatomy" % (label,)
        )


# --- loaders -----------------------------------------------------------------


def load_names(cache_dir=None, allow_download=False):
    """`(area labels, receptor published names)` from the verified cache."""
    areas_path = ensure(
        "zilles2017_RegionNames_npy", cache_dir=cache_dir,
        allow_download=allow_download,
    )
    names_path = ensure(
        "zilles2017_ReceptorNames_npy", cache_dir=cache_dir,
        allow_download=allow_download,
    )
    areas = [str(x) for x in np.load(areas_path, allow_pickle=True)]
    names = [str(x) for x in np.load(names_path, allow_pickle=True)]
    if len(areas) != 44 or len(names) != 15:
        raise ValueError(
            "unexpected name-array lengths: %d areas, %d receptors"
            % (len(areas), len(names))
        )
    return areas, names


_STRATUM_MANIFEST_KEYS = {
    "supragranular": "zilles2017_ReceptData_S_npy",
    "granular": "zilles2017_ReceptData_G_npy",
    "infragranular": "zilles2017_ReceptData_I_npy",
}


def load_stratum_array(stratum, cache_dir=None, allow_download=False):
    """One stratum's (44, 15) uint16 array from the verified cache."""
    if stratum not in STRATA:
        raise KeyError("stratum %r is not one of %s" % (stratum, STRATA))
    path = ensure(
        _STRATUM_MANIFEST_KEYS[stratum],
        cache_dir=cache_dir, allow_download=allow_download,
    )
    return np.load(path, allow_pickle=True)


def load_table(cache_dir=None, allow_download=False):
    """Load and validate the whole table from the verified cache.

    Offline by default: a cold cache raises `MissingCachedFile` naming
    `scripts/fetch_human_autoradiography.py`.
    """
    areas, published_names = load_names(
        cache_dir=cache_dir, allow_download=allow_download
    )
    if tuple(published_names) != _PUBLISHED_NAMES:
        raise ValueError(
            "deposit receptor names %s do not match this module's registry "
            "%s; re-check the transcription before trusting any column"
            % (published_names, list(_PUBLISHED_NAMES))
        )
    strata_arrays = {
        stratum: load_stratum_array(
            stratum, cache_dir=cache_dir, allow_download=allow_download
        )
        for stratum in STRATA
    }
    return AutoradiographicDensityTable(
        strata_arrays=strata_arrays,
        area_labels=areas,
        receptor_keys=receptors_covered(),
        citation=" ".join([ZILLES_CITATION, PROTOCOL_NOTE]),
    )


def absolute_anchor(receptor_key, cache_dir=None, allow_download=False):
    """The absolute-scale anchor for one model receptor system.

    For GABA_A_BZ (and the other 14 measured receptors) this is
    `{area: {stratum: AbsoluteDensity}}`. For MOR and D2 it raises
    `ScaleFreeReceptor` carrying the recorded reason: they are not in
    this source, and no neighbouring receptor's column may stand in for
    them.
    """
    if receptor_key in SCALE_FREE_RECEPTORS:
        raise ScaleFreeReceptor(receptor_key, SCALE_FREE_RECEPTORS[receptor_key])
    return load_table(
        cache_dir=cache_dir, allow_download=allow_download
    ).anchor(receptor_key)


# --- verification anchor -----------------------------------------------------

#: Supplementary Table 2's block headers use Unicode subscripts; the .npy
#: deposit uses ASCII. Positional order matches (verified), and this map
#: is how the cell-for-cell test addresses the blocks by OUR keys.
XLSX_BLOCK_NAMES = {
    "AMPA": "AMPA",
    "NMDA": "NMDA",
    "kainate": "kainate",
    "GABA_A": "GABAᴀ",
    "GABA_A_BZ": "GABAᴀ/BZ",
    "GABAb": "GABAᴃ",
    "m1": "M₁",
    "m2": "M₂",
    "m3": "M₃",
    "a4b2": "α₄β₂",
    "a1": "α₁",
    "a2": "α₂",
    "5-HT1a": "5-HT₁ᴀ",
    "5-HT2": "5-HT₂",
    "D1": "D₁",
}


def read_published_supplementary_table(path):
    """Parse the cached Supplementary Table 2 xlsx into measured means.

    Returns ``{(published receptor name, area, stratum): mean}`` for all
    15 x 44 x 3 cells, with unmeasured strata (area 24's granular) absent
    from the dict exactly as the publication prints '--'. Verification
    only; no loader consumes the result. Requires openpyxl, imported
    lazily.
    """
    from openpyxl import load_workbook

    sheet = load_workbook(path, read_only=True)["Sheet1"]
    rows = list(sheet.iter_rows(values_only=True))
    header_rows = [i for i, r in enumerate(rows) if r and r[0] == "Region"]
    if len(header_rows) != 15:
        raise ValueError(
            "%s: expected 15 receptor blocks, found %d"
            % (path, len(header_rows))
        )
    reverse_published = {v: k for k, v in XLSX_BLOCK_NAMES.items()}

    blocks = []
    for header_row in header_rows:
        title = None
        for j in range(header_row - 1, -1, -1):
            if rows[j][0] is not None:
                title = rows[j][0]
                break
        if title not in reverse_published:
            raise ValueError(
                "%s: block titled %r is not a recognised receptor name"
                % (path, title)
            )
        blocks.append((reverse_published[title], header_row + 1))

    def cell(value):
        if value is None or value == "--":
            return None
        return value

    parsed = {}
    for index, (receptor_key, start) in enumerate(blocks):
        end = (
            blocks[index + 1][1] - 2
            if index + 1 < len(blocks)
            else len(rows)
        )
        seen_areas = set()
        for row in rows[start:end]:
            if row[0] is None:
                continue
            area = str(row[0])
            seen_areas.add(area)
            for column, stratum in ((1, "supragranular"), (4, "granular"),
                                    (7, "infragranular")):
                value = cell(row[column])
                if value is None:
                    if (area, stratum) not in AGRANULAR_ABSENCES:
                        raise ValueError(
                            "%s: %s/%s is empty but not a registered "
                            "absence" % (path, area, stratum)
                        )
                    continue
                parsed[(receptor_key, area, stratum)] = float(value)
        if len(seen_areas) != 44:
            raise ValueError(
                "%s: block %s covers %d areas, expected 44"
                % (path, receptor_key, len(seen_areas))
            )
    return parsed
