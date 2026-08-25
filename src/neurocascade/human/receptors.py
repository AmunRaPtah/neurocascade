"""The human PET receptor-density atlas: Hansen et al. 2022, ingested.

This is the human analogue of the mouse receptor-map layer
(``expression/``), and it replaces that layer's *measurement*, not just its
species: where the mouse package reads in-situ hybridization mRNA as a
proxy for density and states the proportionality assumption out loud, this
package reads measured in-vivo receptor availability. The correspondence
between the two is itself published: gene expression tracks measured
density well only for a handful of receptors (Hansen, Markello, Tuominen,
et al., *NeuroImage* 2022;264:119671, PMID 36209794 -- MOR r=0.846, D2
r=0.706, GABA-A max |r| over all GABR* subunits 0.41 with every p_spin >=
0.24), which is why ``roadmap-human.md`` retires the ISH layer (P12.4).

The source
----------

Hansen JY, Markello RD, Tuominen L, Norgaard M, Kuzmin E,
Palomero-Gallagher N, Dagher A, Misic B. "Mapping neurotransmitter systems
to the structural and functional organization of the human neocortex."
*Nature Neuroscience* 2022;25(11):1569-1581. PMID 36303070.
doi:10.1038/s41593-022-01186-3. Data and code:
``github.com/netneurolab/hansen_receptors``, pinned to one commit in the
manifest below.

**Licence: CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike). The
repository's ``LICENSE.md`` is the full Creative Commons legal code. This
module records the licence here, in ``fixtures/PROVENANCE.md``, and in
every citation string, because ShareAlike obligations attach to any
derived table we commit. Non-commercial research use is what this project
is; if that ever changes, this layer has to be re-licensed or removed.

19 receptors/transporters across 9 systems, from 1,238 healthy
participants. P12.1 ingests the three the existing scenarios need:

=============== ==========================================================
``"MOR"``       mu-opioid, [11C]carfentanil (Kantonen 2020, n=204)
``"D2"``        dopamine D2/D3, [11C]FLB457 (Smith n=37 + Sandiego n=55)
``"GABA_A_BZ"`` GABA-A benzodiazepine site, [11C]flumazenil (Norgaard
                2021, n=16)
=============== ==========================================================

Parcellation: Cammoun ("Lausanne") scale033 -- 83 parcels = 68 cortical +
14 subcortical + brainstem (Cammoun et al., *J Neurosci Methods*
2012;203(2):386-397, PMID 22001222). The subcortical parcels are the point
of choosing this scale: accumbens, amygdala, caudate, pallidum, putamen,
thalamus and hippocampus are exactly the nuclei the scenarios need and
exactly what a cortical-only parcellation cannot supply. At this coarsest
scale the cortical parcel names are the Desikan-Killiany names, so each of
the 83 parcels maps to one region of the ENIGMA Desikan-Killiany
connectome (``human/sources.py``) except the brainstem, which has no DK
counterpart and stays unmapped (`UNMAPPED_PARCELS`).

The central constraint: relative, not absolute
----------------------------------------------

These maps are **z-scored relative densities**. They are NOT Bmax, NOT
fmol/mg protein, NOT a binding-site concentration, and no operation
offered here can turn them into one. This is a property of the source, not
a choice of this module:

* different tracers came from different participants, scanners and cohorts
  (one MOR map is n=204 across variable scanners; the other n=39 on one
  HRRT), so their raw values share no scale;
* different tracers report different quantities. Per the repository's own
  Table S3: FLB457 and carfentanil are BPnd (SRTM); the flumazenil map is
  logged as "Bmax" but "originally Vt, converted to Bmax using
  autoradiography densities" -- a VT-derived quantity, not a measured
  binding-site count;
* the paper's own aggregation z-scores each tracer before combining them
  (``code/make_receptor_matrix.py`` in the pinned repository), so the
  published numbers are standardized deviations by construction.

The type system below carries that constraint rather than commenting on
it:

* `PETRelativeDensityMap` exposes **no numeric array attribute**. Every
  read goes through `for_parcel` / `for_connectome_label`, which return a
  `RelativeDensityValue`, never a bare float.
* `RelativeDensityValue` supports ordering, same-scale arithmetic, and
  multiplication by a mechanism gain (which returns a plain float: the
  product is a coupling coefficient and is no longer a density). It
  refuses addition/subtraction/division involving bare numbers, and it
  raises on any comparison or arithmetic across two different scales --
  so an accumbens z-score on the flb457 scale cannot silently meet one on
  the flumazenil scale.
* There is deliberately no conversion accessor. `absolute_density()`
  exists to refuse, loudly, and name what an honest version would need.
* ``float(value)`` is the documented exit from the type system: explicit,
  per value, and visible in review in a way ``map.values[3]`` is not.

What the numbers mean instead: a value of ``+2.5`` on
`NEOCORTICAL_SCALE` means "2.5 standard deviations above the mean
neocortical density of this tracer's own measure", nothing more.
Subcortical values reach beyond +/-4 because the reference distribution
is neocortical (the publication's aggregation script z-scores every tracer
over the same 68 cortical parcels -- pinned to the deposit bit-for-bit for
D2 by `read_published_cortical_columns` and its test) while striatal
binding is far outside the cortical range. A large |z| is a statement
about the reference distribution, not about receptor count.

Offline by default
------------------

Same contract as the rest of ``human/``: nothing touches the network at
import time, loaders default to ``allow_download=False``, a cache miss
raises `MissingCachedFile` naming ``scripts/fetch_human_receptors.py``.
Tests run off the committed fixtures in ``fixtures/``, which for the five
tracer files are complete verbatim copies of the published files (they are
one row of 83 numbers each) and for the published aggregation a verbatim
68 x 3 column slice. See ``fixtures/PROVENANCE.md``.

    from neurocascade import human

    mor = human.receptors.load_receptor("MOR")
    v = mor.for_connectome_label("Laccumb")     # RelativeDensityValue
    float(v)                                    # +4.198..., explicitly
    coupling = human.receptors.coupling_over_connectome(
        mor,
        mechanism_gains=-0.5,   # ILLUSTRATIVE: caller-supplied, never fitted
        labels=conn.labels,
    )
"""

import os
import sys

import numpy as np

from .. import datafetch
from ..datafetch import ChecksumMismatch, MissingCachedFile, sha256_of  # noqa: F401

__all__ = [
    # citations and licence
    "HANSEN_CITATION",
    "HANSEN_LICENCE",
    "CAMMOUN_CITATION",
    "TRACER_CITATIONS",
    # manifest
    "MANIFEST_VERSION",
    "DATA_FILES",
    "DATA_FILES_BY_KEY",
    "DEFAULT_CACHE_DIR",
    "REPOSITORY_COMMIT",
    "data_file",
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
    # parcellation and crosswalk
    "CammounParcel",
    "PARCELS",
    "N_PARCELS",
    "CORTICAL_PARCELS",
    "SUBCORTICAL_PARCELS",
    "BRAINSTEM_PARCEL",
    "CAMMOUN_TO_CONNECTOME",
    "CONNECTOME_TO_CAMMOUN",
    "UNMAPPED_PARCELS",
    "SUBCORTICAL_CONNECTOME_LABELS",
    "parcel_for_connectome_label",
    "connectome_label_for_parcel",
    # tracers and receptors
    "TracerProvenance",
    "TRACERS",
    "ReceptorProvenance",
    "RECEPTORS",
    "NOT_COMBINED",
    "receptors_ingested",
    # types
    "RelativeScale",
    "NEOCORTICAL_SCALE",
    "RELATIVE_UNITS",
    "RelativeDensityValue",
    "PETRelativeDensityMap",
    "ParcellatedPETMap",
    "combine_weighted",
    # loaders
    "read_parcellated",
    "load_tracer",
    "load_receptor",
    "load_receptors",
    # join to the connectome/scenarios
    "coupling_over_connectome",
    # verification anchor
    "PUBLISHED_MATRIX_KEY",
    "PUBLISHED_COLUMN_SCALES",
    "PUBLISHED_FIXTURE_COLUMNS",
    "read_published_cortical_columns",
    "read_published_column_slice",
]

# --- citations --------------------------------------------------------------


def _cited(title, journal, year, volume, pages, pmid, doi):
    return (
        "%s %s %s;%s:%s. PMID %s. doi:%s."
        % (title, journal, year, volume, pages, pmid, doi)
    )


#: The atlas this module ingests. Species: human (in vivo PET).
HANSEN_CITATION = (
    "Hansen JY, Markello RD, Tuominen L, Norgaard M, Kuzmin E, "
    "Palomero-Gallagher N, Dagher A, Misic B. 'Mapping neurotransmitter "
    "systems to the structural and functional organization of the human "
    "neocortex.' Nature Neuroscience 2022;25(11):1569-1581. "
    "PMID 36303070. doi:10.1038/s41593-022-01186-3. 19 receptors/"
    "transporters across 9 systems, 1238 healthy participants, parcellated "
    "with the Cammoun ('Lausanne') multiscale atlas. Data and code: "
    "github.com/netneurolab/hansen_receptors."
)

#: The licence the data carries, recorded wherever the data is used. CC
#: BY-NC-SA 4.0 binds derivatives: anything we commit that derives from
#: these files must carry the same licence terms.
HANSEN_LICENCE = (
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 "
    "International (CC BY-NC-SA 4.0): the hansen_receptors repository's "
    "LICENSE.md. Attribution required; non-commercial use only; derived "
    "works must be shared under the same licence."
)

CAMMOUN_CITATION = (
    "Parcellation: Cammoun L, Gigandet X, Meskaldji D, Thiran JP, Sporns O, "
    "Do KQ, Maeder P, Meuli R, Hagmann P. 'Mapping the human connectome at "
    "multiple scales with diffusion spectrum MRI.' J Neurosci Methods "
    "2012;203(2):386-397. PMID 22001222. "
    "doi:10.1016/j.jneumeth.2011.09.031. Scale033: 83 parcels = 68 cortical "
    "+ 14 subcortical + brainstem."
)

#: Primary reference per tracer, keyed by tracer key. These are the papers
#: that collected the PET data, which the repository's README asks users of
#: any single map to cite alongside Hansen et al. Bibliographic details
#: verified against PubMed 2026-08-24.
TRACER_CITATIONS = {
    "MU_carfentanil_hc204_kantonen": _cited(
        "Kantonen T, Karjalainen T, Isojarvi J, Nuutila P, Tuisku J, "
        "Rinne J, Kaasinen V, Karrasch M, Laine M, Vahlberg T, et al.",
        "'Interindividual variability and lateralization of mu-opioid "
        "receptors in the human brain.' NeuroImage",
        "2020", "217", "116922", "32407992",
        "10.1016/j.neuroimage.2020.116922",
    ),
    "MU_carfentanil_hc39_turtonen": _cited(
        "Turtonen O, Saarinen M, Nummenmaa L, Tuominen L.",
        "'Adult attachment system links with brain mu opioid receptor "
        "availability in vivo.' Biological Psychiatry: Cognitive "
        "Neuroscience and Neuroimaging",
        "2021", "6(3)", "360-369", "33431346",
        "10.1016/j.bpsc.2020.10.013",
    ),
    "D2_flb457_hc37_smith": _cited(
        "Smith CT, Crawford JL, Dang LC, Seaman KL, San Juan MD, Vijay A, "
        "Katz DT, Matuskey D, Cowan RL, Morris ED, Zald DH, "
        "Samanez-Larkin GR.",
        "'Partial-volume correction increases estimated dopamine D2-like "
        "receptor binding potential and reduces adult age differences.' "
        "J Cereb Blood Flow Metab (online 2017 Nov 1)",
        "2019", "39(5)", "822-833", "29090626",
        "10.1177/0271678X17737693",
    ),
    "D2_flb457_hc55_sandiego": _cited(
        "Sandiego CM, Gallezot JD, Lim K, Ropchan J, Lin SF, Gao H, "
        "Morris ED, Cosgrove KP.",
        "'Reference region modeling approaches for amphetamine challenge "
        "studies with [11C]FLB 457 and PET.' J Cereb Blood Flow Metab",
        "2015", "35(4)", "623-629", "25564239",
        "10.1038/jcbfm.2014.237",
    ),
    "GABAa-bz_flumazenil_hc16_norgaard": _cited(
        "Norgaard M, Beliveau V, Ganz M, Svarer C, Palner F, Keller SH, "
        "Jensen P, Greve DN, Knudsen GM.",
        "'A high-resolution in vivo atlas of the human brain's "
        "benzodiazepine binding site of GABA(A) receptors.' NeuroImage",
        "2021", "232", "117878", "33610745",
        "10.1016/j.neuroimage.2021.117878",
    ),
}

# --- manifest ---------------------------------------------------------------

MANIFEST_VERSION = "2026-08-24"

#: What a human should run when the cache is missing a file.
FETCH_COMMAND = "scripts/fetch_human_receptors.py"

#: One commit of ``netneurolab/hansen_receptors`` (the repository's HEAD at
#: retrieval, 2022-10-27). raw.githubusercontent.com serves immutable bytes
#: per commit, so these URLs cannot drift.
REPOSITORY_COMMIT = "f8b41da92a733f99368c1f39d4959731aa1316c1"

_BASE = "https://raw.githubusercontent.com/netneurolab/hansen_receptors/" + REPOSITORY_COMMIT

#: Default on-disk cache, shared with the other human layers: <repo>/data/human.
DEFAULT_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "human")
)

#: The paper's own aggregated neocortical matrix: not consumed by any
#: loader, read only by `read_published_cortical_columns` to prove this
#: module's re-aggregation matches the publication. Defined here so the
#: manifest can reference it.
PUBLISHED_MATRIX_KEY = "hansen_results_receptor_data_scale033"

#: Retrieved and hashed 2026-08-24; each URL was fetched twice and the bytes
#: compared identical before the digest was recorded.
_RETRIEVED = "2026-08-24"

_PET_UNITS = (
    "parcellated PET binding from one tracer, in that tracer's own reported "
    "quantity and on that cohort's own scale (see the tracer registry for "
    "whether it is BPnd or a VT-derived quantity). NOT comparable between "
    "tracers, NOT a receptor density, and never read as absolute"
)

_PUBLISHED_UNITS = (
    "z-scored relative densities over 68 neocortical parcels (dimensionless "
    "SD units), several tracers' quantities combined per the paper's own "
    "participant-weighted rule. NOT absolute; see receptors.RELATIVE_UNITS"
)


class DataFile(datafetch.DataFile):
    """One external file, defaulting `retrieved` to this manifest's date."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("retrieved", _RETRIEVED)
        datafetch.DataFile.__init__(self, *args, **kwargs)


_DATA = (
    # --- the five parcellated tracer maps ----------------------------------
    (
        "hansen_pet_scale033_MU_carfentanil_hc204_kantonen",
        "hansen_pet_scale033_MU_carfentanil_hc204_kantonen.csv",
        "data/PET_parcellated/scale033/MU_carfentanil_hc204_kantonen.csv",
        "abc343a5cd6dec899c35b65770b2b9d9828dfe8e74b9175b07cc6cb6ecda799e",
        2075,
        "[11C]carfentanil BPnd, 204 healthy participants (SRTM, occipital "
        "cortex reference; scanner variable per Kantonen et al. 2020). THE "
        "MOR map: this is the map Hansen et al. combine into their "
        "published MOR column. One comma-separated row of 83 values, in "
        "the Cammoun-033 parcel order of `region_info_scale033.xlsx`.",
    ),
    (
        "hansen_pet_scale033_MU_carfentanil_hc39_turtonen",
        "hansen_pet_scale033_MU_carfentanil_hc39_turtonen.csv",
        "data/PET_parcellated/scale033/MU_carfentanil_hc39_turtonen.csv",
        "546e46f5b70a23461a81f7c400eefc612349f4cffa4aec95f8c7f4315dff84d4",
        2075,
        "[11C]carfentanil BPnd, 39 healthy participants (SRTM, occipital "
        "cortex reference, HRRT; Turtonen et al. 2021). An INDEPENDENT "
        "replication cohort for the MOR topography, not combined into the "
        "product map: the published matrix uses the Kantonen map alone. "
        "Used here by one anatomy test (cross-cohort replication). Same "
        "83-value row format.",
    ),
    (
        "hansen_pet_scale033_D2_flb457_hc37_smith",
        "hansen_pet_scale033_D2_flb457_hc37_smith.csv",
        "data/PET_parcellated/scale033/D2_flb457_hc37_smith.csv",
        "0b4d1835d722e683241c81dcd9817adeb958c8a1a91fde5f92643a52dd869eb7",
        2075,
        "[11C]FLB457 BPnd, 37 healthy participants (SRTM, cerebellar "
        "reference, HRRT; Smith et al.). Half of THE D2 map.",
    ),
    (
        "hansen_pet_scale033_D2_flb457_hc55_sandiego",
        "hansen_pet_scale033_D2_flb457_hc55_sandiego.csv",
        "data/PET_parcellated/scale033/D2_flb457_hc55_sandiego.csv",
        "fc0c72b672b80837da5dfc1f5bf5c8bf6c3c043ebe3c228bd48897558dd8b4ef",
        2075,
        "[11C]FLB457 BPnd, 55 healthy participants (SRTM, cerebellar "
        "reference, HR+; Sandiego et al.). The other half of THE D2 map.",
    ),
    (
        "hansen_pet_scale033_GABAa-bz_flumazenil_hc16_norgaard",
        "hansen_pet_scale033_GABAa-bz_flumazenil_hc16_norgaard.csv",
        "data/PET_parcellated/scale033/GABAa-bz_flumazenil_hc16_norgaard.csv",
        "f76da8cd9f25fb344e5e1439366a7aeaa0a1db1b046ad6a780a40f8b802ba2f8",
        2075,
        "[11C]flumazenil at the GABA-A benzodiazepine site, 16 healthy "
        "participants (Logan analysis, HRRT; Norgaard et al. 2021). THE "
        "GABA-A/BZ map. Table S3 logs its parameter as 'Bmax' with the "
        "note 'originally Vt, converted to Bmax using autoradiography "
        "densities' - i.e. a VT-DERIVED quantity, not a measured "
        "binding-site count, which is one of the reasons the product maps "
        "here are typed relative-only.",
    ),
    # --- parcel identity ----------------------------------------------------
    (
        "hansen_lausanne_region_info_scale033",
        "hansen_lausanne_region_info_scale033.xlsx",
        "data/lausanne/region_info_scale033.xlsx",
        "a44d8403dd802dc9010653a11b6207be47bc4d507cd766e89f4c14c70314ff89",
        12263,
        "Parcel identity for the Cammoun-033 parcellation: id, label, "
        "hemisphere, structure (cortex/subcortex) per row, 83 rows in the "
        "order the parcellated CSVs' columns follow. Read by a test that "
        "pins the hard-coded PARCELS table in receptors.py against the "
        "published file; openpyxl parses it lazily.",
    ),
    # --- the published aggregation, as a verification anchor -----------------
    (
        PUBLISHED_MATRIX_KEY,
        "hansen_results_receptor_data_scale033.csv",
        "results/receptor_data_scale033.csv",
        "bf3e707fda6004749cec5571dc0720e12f8895c5827642e1f690787b0ff095e2",
        32482,
        "The paper's own aggregated 68-parcel (neocortex only) x 19-receptor "
        "matrix, as deposited. NOT consumed by any loader: it exists in this "
        "manifest so this module can verify itself against the publication "
        "bit-for-bit (`read_published_cortical_columns`). The deposit's "
        "columns are NOT on one shared scale -- verified numerically, see "
        "PUBLISHED_COLUMN_SCALES: D2 is a z-scored two-tracer combination "
        "(which pins our z-scoring basis and weights), MOR and GABA-A/BZ are "
        "single tracers' raw values (which pin parcel order and the parse). "
        "Columns follow data/receptor_names_pet.npy order; rows follow the "
        "same Cammoun-033 order with subcortical parcels dropped.",
    ),
)


def _entry(key, filename, upstream_path, sha, size, description):
    citation_bits = [HANSEN_CITATION, HANSEN_LICENCE]
    for tracer_key in TRACER_CITATIONS:
        if tracer_key in filename:
            citation_bits.append(TRACER_CITATIONS[tracer_key])
            break
    else:
        citation_bits.append(CAMMOUN_CITATION)
    if "region_info" in filename:
        # The xlsx carries no numbers, only parcel identity; its provenance
        # is its units.
        units = CAMMOUN_CITATION
    elif PUBLISHED_MATRIX_KEY in key:
        units = _PUBLISHED_UNITS
    else:
        units = _PET_UNITS
    return DataFile(
        key=key,
        filename=filename,
        url=_BASE + "/" + upstream_path,
        sha256=sha,
        size_bytes=size,
        citation=" ".join(citation_bits),
        description=description,
        units=units,
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


# --- the parcellation, and the crosswalk to the connectome ------------------


class CammounParcel(object):
    """One Cammoun-033 parcel: its published id, label, and what it is.

    ``connectome_label`` is the ENIGMA Desikan-Killiany label of the region
    this parcel corresponds to (`human/atlas.py`), or None for the
    brainstem, which has no Desikan-Killiany counterpart.
    """

    __slots__ = ("id", "label", "hemisphere", "structure", "connectome_label")

    def __init__(self, id, label, hemisphere, structure, connectome_label):
        self.id = int(id)
        self.label = label
        self.hemisphere = hemisphere
        self.structure = structure
        self.connectome_label = connectome_label

    def __repr__(self):
        return "CammounParcel(%d, %r, %s, %s)" % (
            self.id, self.label, self.hemisphere, self.structure,
        )


#: The 83 Cammoun-033 parcels, in the exact column order of every
#: parcellated CSV in the manifest. Transcribed once from the published
#: `region_info_scale033.xlsx`; a test re-checks this table against that
#: file whenever the cache is present, and a second test pins the cortical
#: block against the paper's own aggregated matrix, which would catch any
#: transcription error in 68 of the 83 positions even fully offline.
PARCELS = (
    CammounParcel(1, "R_lateralorbitofrontal", "R", "cortex", "R_lateralorbitofrontal"),
    CammounParcel(2, "R_parsorbitalis", "R", "cortex", "R_parsorbitalis"),
    CammounParcel(3, "R_frontalpole", "R", "cortex", "R_frontalpole"),
    CammounParcel(4, "R_medialorbitofrontal", "R", "cortex", "R_medialorbitofrontal"),
    CammounParcel(5, "R_parstriangularis", "R", "cortex", "R_parstriangularis"),
    CammounParcel(6, "R_parsopercularis", "R", "cortex", "R_parsopercularis"),
    CammounParcel(7, "R_rostralmiddlefrontal", "R", "cortex", "R_rostralmiddlefrontal"),
    CammounParcel(8, "R_superiorfrontal", "R", "cortex", "R_superiorfrontal"),
    CammounParcel(9, "R_caudalmiddlefrontal", "R", "cortex", "R_caudalmiddlefrontal"),
    CammounParcel(10, "R_precentral", "R", "cortex", "R_precentral"),
    CammounParcel(11, "R_paracentral", "R", "cortex", "R_paracentral"),
    CammounParcel(12, "R_rostralanteriorcingulate", "R", "cortex", "R_rostralanteriorcingulate"),
    CammounParcel(13, "R_caudalanteriorcingulate", "R", "cortex", "R_caudalanteriorcingulate"),
    CammounParcel(14, "R_posteriorcingulate", "R", "cortex", "R_posteriorcingulate"),
    CammounParcel(15, "R_isthmuscingulate", "R", "cortex", "R_isthmuscingulate"),
    CammounParcel(16, "R_postcentral", "R", "cortex", "R_postcentral"),
    CammounParcel(17, "R_supramarginal", "R", "cortex", "R_supramarginal"),
    CammounParcel(18, "R_superiorparietal", "R", "cortex", "R_superiorparietal"),
    CammounParcel(19, "R_inferiorparietal", "R", "cortex", "R_inferiorparietal"),
    CammounParcel(20, "R_precuneus", "R", "cortex", "R_precuneus"),
    CammounParcel(21, "R_cuneus", "R", "cortex", "R_cuneus"),
    CammounParcel(22, "R_pericalcarine", "R", "cortex", "R_pericalcarine"),
    CammounParcel(23, "R_lateraloccipital", "R", "cortex", "R_lateraloccipital"),
    CammounParcel(24, "R_lingual", "R", "cortex", "R_lingual"),
    CammounParcel(25, "R_fusiform", "R", "cortex", "R_fusiform"),
    CammounParcel(26, "R_parahippocampal", "R", "cortex", "R_parahippocampal"),
    CammounParcel(27, "R_entorhinal", "R", "cortex", "R_entorhinal"),
    CammounParcel(28, "R_temporalpole", "R", "cortex", "R_temporalpole"),
    CammounParcel(29, "R_inferiortemporal", "R", "cortex", "R_inferiortemporal"),
    CammounParcel(30, "R_middletemporal", "R", "cortex", "R_middletemporal"),
    CammounParcel(31, "R_bankssts", "R", "cortex", "R_bankssts"),
    CammounParcel(32, "R_superiortemporal", "R", "cortex", "R_superiortemporal"),
    CammounParcel(33, "R_transversetemporal", "R", "cortex", "R_transversetemporal"),
    CammounParcel(34, "R_insula", "R", "cortex", "R_insula"),
    CammounParcel(35, "R_thalamusproper", "R", "subcortex", "Rthal"),
    CammounParcel(36, "R_caudate", "R", "subcortex", "Rcaud"),
    CammounParcel(37, "R_putamen", "R", "subcortex", "Rput"),
    CammounParcel(38, "R_pallidum", "R", "subcortex", "Rpal"),
    CammounParcel(39, "R_accumbensarea", "R", "subcortex", "Raccumb"),
    CammounParcel(40, "R_hippocampus", "R", "subcortex", "Rhippo"),
    CammounParcel(41, "R_amygdala", "R", "subcortex", "Ramyg"),
    CammounParcel(42, "L_lateralorbitofrontal", "L", "cortex", "L_lateralorbitofrontal"),
    CammounParcel(43, "L_parsorbitalis", "L", "cortex", "L_parsorbitalis"),
    CammounParcel(44, "L_frontalpole", "L", "cortex", "L_frontalpole"),
    CammounParcel(45, "L_medialorbitofrontal", "L", "cortex", "L_medialorbitofrontal"),
    CammounParcel(46, "L_parstriangularis", "L", "cortex", "L_parstriangularis"),
    CammounParcel(47, "L_parsopercularis", "L", "cortex", "L_parsopercularis"),
    CammounParcel(48, "L_rostralmiddlefrontal", "L", "cortex", "L_rostralmiddlefrontal"),
    CammounParcel(49, "L_superiorfrontal", "L", "cortex", "L_superiorfrontal"),
    CammounParcel(50, "L_caudalmiddlefrontal", "L", "cortex", "L_caudalmiddlefrontal"),
    CammounParcel(51, "L_precentral", "L", "cortex", "L_precentral"),
    CammounParcel(52, "L_paracentral", "L", "cortex", "L_paracentral"),
    CammounParcel(53, "L_rostralanteriorcingulate", "L", "cortex", "L_rostralanteriorcingulate"),
    CammounParcel(54, "L_caudalanteriorcingulate", "L", "cortex", "L_caudalanteriorcingulate"),
    CammounParcel(55, "L_posteriorcingulate", "L", "cortex", "L_posteriorcingulate"),
    CammounParcel(56, "L_isthmuscingulate", "L", "cortex", "L_isthmuscingulate"),
    CammounParcel(57, "L_postcentral", "L", "cortex", "L_postcentral"),
    CammounParcel(58, "L_supramarginal", "L", "cortex", "L_supramarginal"),
    CammounParcel(59, "L_superiorparietal", "L", "cortex", "L_superiorparietal"),
    CammounParcel(60, "L_inferiorparietal", "L", "cortex", "L_inferiorparietal"),
    CammounParcel(61, "L_precuneus", "L", "cortex", "L_precuneus"),
    CammounParcel(62, "L_cuneus", "L", "cortex", "L_cuneus"),
    CammounParcel(63, "L_pericalcarine", "L", "cortex", "L_pericalcarine"),
    CammounParcel(64, "L_lateraloccipital", "L", "cortex", "L_lateraloccipital"),
    CammounParcel(65, "L_lingual", "L", "cortex", "L_lingual"),
    CammounParcel(66, "L_fusiform", "L", "cortex", "L_fusiform"),
    CammounParcel(67, "L_parahippocampal", "L", "cortex", "L_parahippocampal"),
    CammounParcel(68, "L_entorhinal", "L", "cortex", "L_entorhinal"),
    CammounParcel(69, "L_temporalpole", "L", "cortex", "L_temporalpole"),
    CammounParcel(70, "L_inferiortemporal", "L", "cortex", "L_inferiortemporal"),
    CammounParcel(71, "L_middletemporal", "L", "cortex", "L_middletemporal"),
    CammounParcel(72, "L_bankssts", "L", "cortex", "L_bankssts"),
    CammounParcel(73, "L_superiortemporal", "L", "cortex", "L_superiortemporal"),
    CammounParcel(74, "L_transversetemporal", "L", "cortex", "L_transversetemporal"),
    CammounParcel(75, "L_insula", "L", "cortex", "L_insula"),
    CammounParcel(76, "L_thalamusproper", "L", "subcortex", "Lthal"),
    CammounParcel(77, "L_caudate", "L", "subcortex", "Lcaud"),
    CammounParcel(78, "L_putamen", "L", "subcortex", "Lput"),
    CammounParcel(79, "L_pallidum", "L", "subcortex", "Lpal"),
    CammounParcel(80, "L_accumbensarea", "L", "subcortex", "Laccumb"),
    CammounParcel(81, "L_hippocampus", "L", "subcortex", "Lhippo"),
    CammounParcel(82, "L_amygdala", "L", "subcortex", "Lamyg"),
    CammounParcel(83, "brainstem", "L", "subcortex", None),
)

assert len(PARCELS) == 83

N_PARCELS = len(PARCELS)
CORTICAL_PARCEL_INDEXES = tuple(
    i for i, p in enumerate(PARCELS) if p.structure == "cortex"
)
assert len(CORTICAL_PARCEL_INDEXES) == 68

SUBCORTICAL_PARCEL_INDEXES = tuple(
    i
    for i, p in enumerate(PARCELS)
    if p.structure == "subcortex" and p.connectome_label is not None
)
assert len(SUBCORTICAL_PARCEL_INDEXES) == 14

BRAINSTEM_PARCEL = PARCELS[-1]

CAMMOUN_TO_CONNECTOME = {p.label: p.connectome_label for p in PARCELS}
CONNECTOME_TO_CAMMOUN = {
    p.connectome_label: p.label for p in PARCELS if p.connectome_label is not None
}

#: Parcels with no Desikan-Killiany counterpart, and therefore no entry in
#: the ENIGMA connectome the scenarios build on. Kept in the maps (the
#: measurement covers them); unreachable through the connectome join.
UNMAPPED_PARCELS = ("brainstem",)

#: The 14 subcortical ENIGMA connectome labels the scenarios use, i.e. the
#: parcels P12.1 exists to resolve. Order follows the parcellation file.
SUBCORTICAL_CONNECTOME_LABELS = tuple(
    PARCELS[i].connectome_label for i in SUBCORTICAL_PARCEL_INDEXES
)

_PARCEL_INDEX = {p.label: i for i, p in enumerate(PARCELS)}


def parcel_for_connectome_label(label):
    """The Cammoun-033 parcel label behind one ENIGMA connectome label.

    Raises for anything the parcellation does not cover, including the
    brainstem-shaped hole: the Hansen parcellation has a brainstem parcel
    and the Desikan-Killiany connectome does not, so a brainstem request
    from either side is refused rather than silently dropped.
    """
    try:
        return CONNECTOME_TO_CAMMOUN[label]
    except KeyError:
        raise KeyError(
            "%r is not a region the Cammoun-033 parcellation covers under a "
            "Desikan-Killiany name; it has 82 such regions plus %s. The 14 "
            "subcortical ones are %s."
            % (label, ", ".join(UNMAPPED_PARCELS),
               ", ".join(SUBCORTICAL_CONNECTOME_LABELS))
        )


def connectome_label_for_parcel(label):
    """Inverse of `parcel_for_connectome_label`; None for unmapped parcels."""
    return CAMMOUN_TO_CONNECTOME[label]


# --- the tracer and receptor registries -------------------------------------


class TracerProvenance(object):
    """One published tracer map: who measured it, what quantity it reports.

    ``quantity`` is the parameter the source paper modelled, verbatim in
    spirit from the repository's Table S3. It is recorded so that no one
    can later mistake two tracers' raw values for one unit system -- the
    type system forbids combining them anyway (`ParcellatedPETMap` offers
    no arithmetic), but the reason deserves a field rather than a comment.
    """

    __slots__ = (
        "key", "receptor_key", "quantity", "quantity_note", "reference_region",
        "modelling_method", "n_participants", "role", "citation",
    )

    def __init__(
        self,
        key,
        receptor_key,
        quantity,
        reference_region,
        modelling_method,
        n_participants,
        role,
        quantity_note="",
    ):
        self.key = key
        self.receptor_key = receptor_key
        self.quantity = quantity
        self.quantity_note = quantity_note
        self.reference_region = reference_region
        self.modelling_method = modelling_method
        self.n_participants = int(n_participants)
        #: "product" (combined into the map this module hands out) or
        #: "replication-check" (loaded by tests only).
        self.role = role
        self.citation = TRACER_CITATIONS[key]

    @property
    def manifest_key(self):
        return "hansen_pet_scale033_" + self.key

    @property
    def fixture_name(self):
        return "hansen_pet_scale033_%s_fixture.csv" % self.key

    def __repr__(self):
        return "TracerProvenance(%r, %s, n=%d)" % (
            self.key, self.quantity, self.n_participants,
        )


TRACERS = (
    TracerProvenance(
        key="MU_carfentanil_hc204_kantonen",
        receptor_key="MOR",
        quantity="BPnd",
        reference_region="occipital cortex",
        modelling_method="SRTM",
        n_participants=204,
        role="product",
        quantity_note=(
            "scanner variable across the pooled cohort, per the source "
            "paper's supplementary note; another reason the raw values "
            "carry no shared absolute scale"
        ),
    ),
    TracerProvenance(
        key="MU_carfentanil_hc39_turtonen",
        receptor_key="MOR",
        quantity="BPnd",
        reference_region="occipital cortex",
        modelling_method="SRTM",
        n_participants=39,
        role="replication-check",
    ),
    TracerProvenance(
        key="D2_flb457_hc37_smith",
        receptor_key="D2",
        quantity="BPnd",
        reference_region="cerebellum",
        modelling_method="SRTM",
        n_participants=37,
        role="product",
    ),
    TracerProvenance(
        key="D2_flb457_hc55_sandiego",
        receptor_key="D2",
        quantity="BPnd",
        reference_region="cerebellum",
        modelling_method="SRTM",
        n_participants=55,
        role="product",
    ),
    TracerProvenance(
        key="GABAa-bz_flumazenil_hc16_norgaard",
        receptor_key="GABA_A_BZ",
        quantity="Vt-derived ('Bmax' per Table S3)",
        reference_region="none (Logan analysis)",
        modelling_method="Logan analysis",
        n_participants=16,
        role="product",
        quantity_note=(
            "Table S3: 'originally Vt, converted to Bmax using "
            "autoradiography densities'. A distribution-volume-derived "
            "quantity, not a counted binding-site density; converting it "
            "to fmol/mg protein here would be inventing a calibration"
        ),
    ),
)

_TRACERS_BY_KEY = {t.key: t for t in TRACERS}


class ReceptorProvenance(object):
    """One receptor system's map: which tracers, combined how.

    ``combination`` records the published aggregation rule, because this
    module executes it rather than inventing one. For a single-tracer
    receptor it is the identity; for D2 it is Hansen et al.'s
    participant-count-weighted average of per-tracer z-scores, which their
    ``make_receptor_matrix.py`` applies as
    ``(zscore(a)*na + zscore(b)*nb) / (na + nb)``.
    """

    __slots__ = (
        "key", "name", "system", "tracer_keys", "weights", "combination",
        "ligands_note",
    )

    def __init__(
        self, key, name, system, tracer_keys, weights, combination,
        ligands_note="",
    ):
        self.key = key
        self.name = name
        self.system = system
        self.tracer_keys = tuple(tracer_keys)
        self.weights = tuple(float(w) for w in weights)
        self.combination = combination
        self.ligands_note = ligands_note

    @property
    def tracers(self):
        return tuple(_TRACERS_BY_KEY[k] for k in self.tracer_keys)

    def __repr__(self):
        return "ReceptorProvenance(%r, %d tracers)" % (self.key, len(self.tracer_keys))


#: The three systems P12.1 ingests, with the exact combination rule the
#: publication uses. Weights are participant counts (Table S3).
RECEPTORS = {
    "MOR": ReceptorProvenance(
        key="MOR",
        name="mu-opioid receptor",
        system="opioid",
        tracer_keys=("MU_carfentanil_hc204_kantonen",),
        weights=(1.0,),
        combination="single tracer, as published (no averaging)",
        ligands_note=(
            "[11C]carfentanil displacement is the fentanyl/morphine-class "
            "target; the scenario layer's mu-opioid site competes here"
        ),
    ),
    "D2": ReceptorProvenance(
        key="D2",
        name="dopamine D2/D3 receptor",
        system="dopamine",
        tracer_keys=("D2_flb457_hc37_smith", "D2_flb457_hc55_sandiego"),
        weights=(37.0, 55.0),
        combination=(
            "participant-count-weighted average of the two per-tracer "
            "z-scores, exactly the published rule "
            "(make_receptor_matrix.py: (z_a*37 + z_b*55)/92)"
        ),
        ligands_note=(
            "FLB457 is D2/D3 with extrastriatal sensitivity; the "
            "haloperidol scenario's D2 site competes here"
        ),
    ),
    "GABA_A_BZ": ReceptorProvenance(
        key="GABA_A_BZ",
        name="GABA-A benzodiazepine site",
        system="gaba_a",
        tracer_keys=("GABAa-bz_flumazenil_hc16_norgaard",),
        weights=(1.0,),
        combination="single tracer, as published (no averaging)",
        ligands_note=(
            "the benzodiazepine site of alpha1/2/3/5-containing GABA-A "
            "receptors; the diazepam scenario competes here"
        ),
    ),
}

#: Tracers present in the source at scale033 that P12.1 deliberately does
#: not ingest, and why. Recorded the way every searched-and-not-taken list
#: in this repository is: with the reason, so nobody re-litigates it by
#: re-downloading. (The other 16 receptors' scale033 maps also exist; they
#: are future phases, not exclusions, so they are not listed here.)
NOT_COMBINED = {
    "MU_carfentanil_hc39_turtonen":
        "ingested, but as a replication check only: the published MOR "
        "column combines no second tracer, and adding one would be a new "
        "analysis rather than an ingestion",
    "D2_fallypride_hc49_jaworska":
        "not ingested: [18F]fallypride is a different D2/D3 tracer whose "
        "published matrix does not combine it with FLB457; mixing tracers "
        "across quantities is exactly what the relative-only typing exists "
        "to prevent without the paper's own weighting rule",
    "D2_raclopride_hc7_alakurtti":
        "not ingested: [11C]raclopride has insufficient affinity for "
        "extrastriatal D2, n=7, and the published matrix does not use it",
    "GABAa_flumazenil_hc6_dukart":
        "not ingested: a Vt map (RPM) of GABA-A generally rather than the "
        "benzodiazepine site specifically, n=6, unused by the published "
        "matrix",
    "GABAa_FLUMAZENIL_c11":
        "not ingested: the Copenhagen NRU flumazenil release, unused by "
        "the published matrix and without per-tracer provenance in "
        "Table S3",
}


def receptors_ingested():
    """The receptor keys this module loads, in registry order."""
    return tuple(RECEPTORS)


# --- the relative-scale types ----------------------------------------------

#: What every number this module hands out means. Constant, not a
#: constructor argument: no caller can relabel a z-scored map as anything.
RELATIVE_UNITS = (
    "standard deviations from the mean of the reference population "
    "(dimensionless z-score). NOT Bmax, NOT fmol/mg protein, NOT a "
    "binding-site concentration, NOT comparable between tracers or between "
    "reference populations"
)


class RelativeScale(object):
    """The reference population one relative-density map is z-scored on.

    Two maps are comparable only when they carry the SAME scale object:
    same reference population, same tracer quantity after normalization.
    Comparability is decided by identity (`is`), not by field equality, so
    a caller cannot construct a look-alike scale and slip it past a check.
    """

    __slots__ = ("name", "reference_labels", "description")

    def __init__(self, name, reference_labels, description):
        self.name = name
        self.reference_labels = tuple(reference_labels)
        self.description = description

    @property
    def units(self):
        return RELATIVE_UNITS

    def __repr__(self):
        return "RelativeScale(%r, %d reference parcels)" % (
            self.name, len(self.reference_labels),
        )


#: The one scale this module produces: z-scored against the 68 neocortical
#: parcels, which is the population the publication itself z-scores over.
#: Keeping that basis is what makes this module's cortical output identical
#: to the published matrix (verified bit-for-bit by test), while extending
#: the same linear transform to the subcortical parcels and brainstem puts
#: the nuclei the scenarios need on the same axis instead of a second one.
NEOCORTICAL_SCALE = RelativeScale(
    name="neocortical-z",
    reference_labels=tuple(PARCELS[i].label for i in CORTICAL_PARCEL_INDEXES),
    description=(
        "each tracer's own measure standardized against its distribution "
        "over the 68 neocortical Cammoun-033 parcels (population mean and "
        "SD), then combined across tracers by participant-weighted average "
        "where the publication does so. Subcortical parcels and brainstem "
        "are transformed with the SAME mean and SD, so their z-scores are "
        "'how far outside the neocortical distribution', which is why "
        "striatal D2 exceeds +15"
    ),
)


def _check_same_scale(scale, other, op):
    if other._scale is not scale:
        raise TypeError(
            "cannot %s relative densities on different scales: %r vs %r. "
            "Different tracers report different quantities (BPnd vs "
            "VT-derived) normalized over different populations; combining "
            "them is the error this type exists to catch. Go through the "
            "published per-tracer z-scoring rule instead." % (op, scale, other._scale)
        )


class RelativeDensityValue(object):
    """One region's relative density, and nothing else.

    Not a float. Arithmetic that would treat it as an absolute density is
    a TypeError; arithmetic within one scale (hemisphere differences,
    weighted combinations) is allowed and stays typed; multiplication by a
    mechanism gain returns a plain float because the PRODUCT is a coupling
    coefficient, not a density. ``float(v)`` is the explicit way out.
    """

    __slots__ = ("_value", "_scale")

    def __init__(self, value, scale):
        self._value = float(value)
        self._scale = scale

    @property
    def scale(self):
        """The `RelativeScale` this value is expressed on."""
        return self._scale

    def __float__(self):
        return self._value

    def __repr__(self):
        return "RelativeDensityValue(z=%+.6g, %r)" % (self._value, self._scale.name)

    # -- allowed arithmetic -------------------------------------------------

    def __mul__(self, other):
        if isinstance(other, RelativeDensityValue):
            raise TypeError(_PRODUCT_OF_DENSITIES_MESSAGE)
        if not isinstance(other, (int, float)):
            return NotImplemented
        # The product of a gain and a relative density is a coupling
        # coefficient: a plain float, deliberately, because that is what
        # engine.Simulation consumes -- and deliberately NOT a density,
        # because a gain times a z-score is not a z-score either.
        return self._value * float(other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, RelativeDensityValue):
            raise TypeError(
                "cannot divide one relative density by another: the ratio "
                "of two z-scores is not a quantity this source measures"
            )
        if not isinstance(other, (int, float)):
            return NotImplemented
        return RelativeDensityValue(self._value / float(other), self._scale)

    def __add__(self, other):
        if not isinstance(other, RelativeDensityValue):
            return NotImplemented
        _check_same_scale(self._scale, other, "add")
        return RelativeDensityValue(self._value + other._value, self._scale)

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, RelativeDensityValue):
            return NotImplemented
        _check_same_scale(self._scale, other, "subtract")
        return RelativeDensityValue(self._value - other._value, self._scale)

    def __rsub__(self, other):
        if not isinstance(other, RelativeDensityValue):
            return NotImplemented
        _check_same_scale(self._scale, other, "subtract")
        return RelativeDensityValue(other._value - self._value, self._scale)

    def __neg__(self):
        return RelativeDensityValue(-self._value, self._scale)

    def __abs__(self):
        return RelativeDensityValue(abs(self._value), self._scale)

    # -- comparisons: ordering is meaningful, unit conversion is not -------

    def _comparable(self, other):
        if isinstance(other, RelativeDensityValue):
            _check_same_scale(self._scale, other, "compare")
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

    # Defining __eq__ removes the inherited hash; these values live in
    # dicts keyed by LABEL, never by themselves.
    __hash__ = None


_PRODUCT_OF_DENSITIES_MESSAGE = (
    "cannot multiply two relative densities together: the product of two "
    "z-scores is not a density on either scale. Multiplying by a scalar "
    "(a mechanism gain) is supported and returns a plain float"
)


class PETRelativeDensityMap(object):
    """One receptor's per-parcel relative density, and how it was made.

    Constructed only by this module's pipeline (`load_receptor`,
    `combine_weighted`): the constructor requires a `RelativeScale`
    instance, and there is no public numeric array to reach around it.
    Values arrive through `for_parcel` / `for_connectome_label` /
    `as_coupling`.
    """

    def __init__(self, values, receptor, scale):
        values = np.asarray(values, dtype=float)
        if not isinstance(scale, RelativeScale):
            raise TypeError(
                "scale must be a RelativeScale built by this module, got %r"
                % (type(scale).__name__,)
            )
        if values.shape != (N_PARCELS,):
            raise ValueError(
                "expected %d parcel values, got shape %s" % (N_PARCELS, values.shape)
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "non-finite relative density; these parcellated files have "
                "no missing cells, so this is a parse failure, not a gap to "
                "fill (there is no zero to substitute: zero would assert "
                "the receptor is absent)"
            )
        object.__setattr__(self, "_values", values)
        self.receptor = receptor
        self.scale = scale

    def __setattr__(self, name, value):
        if name == "_values":
            raise AttributeError("values are fixed at construction")
        object.__setattr__(self, name, value)

    def __repr__(self):
        return "PETRelativeDensityMap(%r, %d parcels, scale=%r)" % (
            self.receptor.key, N_PARCELS, self.scale.name,
        )

    # -- what the type refuses ---------------------------------------------

    @property
    def units(self):
        return RELATIVE_UNITS

    @property
    def citation(self):
        return " ".join([HANSEN_CITATION, HANSEN_LICENCE] + [
            t.citation for t in self.receptor.tracers
        ])

    def absolute_density(self):
        """There is no such conversion. This method refuses, loudly.

        A z-score carries no unit to convert. The independent anchor
        measured in fmol/mg protein now EXISTS (P12.2:
        ``neurocascade.human.autoradiography``, Zilles & Palomero-Gallagher
        2017), but it covers GABA-A and GABA-A/BZ only -- not MOR, not D2,
        which therefore stay scale-free (`autoradiography.
        SCALE_FREE_RECEPTORS`) -- and it is keyed by 44 Julich-style
        cytoarchitectonic areas rather than this parcellation's
        Desikan-Killiany labels. Applying it to THIS map would take a
        measured crosswalk between those two atlases, which does not exist
        and must not be improvised. So the conversion still cannot happen
        here, inside the type or around it.
        """
        raise TypeError(
            "%s is a %s map: its values are %s. No absolute density can be "
            "derived here. The autoradiography anchor of "
            "neurocascade.human.autoradiography is a separate measurement "
            "with its own citation, its own sample (n=3 postmortem brains, "
            "ages 72-77) and its own area labels; joining it to this map "
            "needs a measured cytoarchitectonic-to-Desikan-Killiany "
            "crosswalk, not a transformation."
            % (self.receptor.key, type(self).__name__, RELATIVE_UNITS)
        )

    # -- reads --------------------------------------------------------------

    @property
    def parcel_labels(self):
        return tuple(p.label for p in PARCELS)

    def for_parcel(self, label):
        """One Cammoun-033 parcel's value, as a `RelativeDensityValue`."""
        try:
            index = _PARCEL_INDEX[label]
        except KeyError:
            raise KeyError(
                "%r is not one of this map's %d Cammoun-033 parcel labels "
                "(see receptors.PARCELS)" % (label, N_PARCELS)
            )
        return RelativeDensityValue(float(self._values[index]), self.scale)

    def for_connectome_label(self, label):
        """One ENIGMA Desikan-Killiany region's value, via the crosswalk.

        This is the accessor the scenarios use: their node labels are the
        82-region connectome's (`human.load_connectome().labels`), and the
        14 subcortical ones are what P12.1 exists to resolve.
        """
        return self.for_parcel(parcel_for_connectome_label(label))

    def subcortical_values(self):
        """`{connectome label: RelativeDensityValue}` for the 14 nuclei."""
        return {
            PARCELS[i].connectome_label: self.for_parcel(PARCELS[i].label)
            for i in SUBCORTICAL_PARCEL_INDEXES
        }

    def measured(self):
        """`[(parcel label, RelativeDensityValue)]` for every parcel."""
        return [(p.label, self.for_parcel(p.label)) for p in PARCELS]

    def top(self, n=10):
        """The n highest parcels, as `[(label, RelativeDensityValue)]`."""
        order = np.argsort(self._values)[::-1][:n]
        return [(PARCELS[i].label, self.for_parcel(PARCELS[i].label)) for i in order]

    def as_coupling(self, mechanism_gains, nodes):
        """`{node: gain * rho[node's region]}`, the receptor layer's input.

        ``nodes`` maps a circuit node name to an ENIGMA Desikan-Killiany
        region label (the convention of ``human/build.py`` and
        ``human/maps.py``). ``mechanism_gains`` is a number or a per-node
        dict; it is the OTHER factor and is not this layer's to supply.
        The products are plain floats -- coupling coefficients, no longer
        densities.
        """
        gains = mechanism_gains
        if not isinstance(gains, dict):
            gains = {node: mechanism_gains for node in nodes}
        coupling = {}
        for node, label in nodes.items():
            if node not in gains:
                raise KeyError("no mechanism gain for node %r" % (node,))
            coupling[node] = self.for_connectome_label(label) * gains[node]
        return coupling


class ParcellatedPETMap(object):
    """One tracer's parcellated values, in that tracer's own quantity.

    The intermediate stage. Its values are BPnd, or a VT-derived quantity,
    on one cohort's scale -- meaningful WITHIN the tracer, meaningless
    across tracers, and therefore given no arithmetic at all: the only
    operation offered is `relative_to_cortex`, the published z-scoring
    step. That is the structural form of "different tracers report
    different quantities": you cannot combine them without first doing
    what the publication did to make them commensurable.
    """

    def __init__(self, values, tracer):
        values = np.asarray(values, dtype=float)
        if values.shape != (N_PARCELS,):
            raise ValueError(
                "expected %d parcels for %s, got shape %s"
                % (N_PARCELS, tracer.key, values.shape)
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "%s: non-finite parcellated value; the published files have "
                "no missing cells, so this is a parse failure" % tracer.key
            )
        object.__setattr__(self, "_values", values)
        self.tracer = tracer

    def __setattr__(self, name, value):
        if name == "_values":
            raise AttributeError("values are fixed at construction")
        object.__setattr__(self, name, value)

    def __repr__(self):
        return "ParcellatedPETMap(%r, %s, n=%d)" % (
            self.tracer.key, self.tracer.quantity, self.tracer.n_participants,
        )

    @property
    def citation(self):
        return " ".join([HANSEN_CITATION, HANSEN_LICENCE, self.tracer.citation])

    def for_parcel(self, label):
        """One parcel's RAW value in the tracer's own quantity.

        Returns a float, deliberately: the quantity is the tracer's own
        (`self.tracer.quantity`) and is recorded on the object. What this
        class refuses is not reading its values but COMBINING them with
        another tracer's.
        """
        try:
            index = _PARCEL_INDEX[label]
        except KeyError:
            raise KeyError(
                "%r is not one of this map's %d Cammoun-033 parcel labels"
                % (label, N_PARCELS)
            )
        return float(self._values[index])

    def reference_population_mean_and_sd(self):
        """`(mean, sd)` over the 68 neocortical parcels, population SD."""
        block = self._values[list(CORTICAL_PARCEL_INDEXES)]
        return float(block.mean()), float(block.std(ddof=0))

    def relative_to_cortex(self):
        """The published z-scoring step, producing a `PETRelativeDensityMap`.

        ``(x - mean_neocortex) / sd_neocortex`` for every parcel, including
        the subcortical ones and brainstem, which the reference population
        does not contain. Population SD (ddof=0), matching scipy's
        ``zscore`` default that the publication's aggregation script uses.
        """
        mean, sd = self.reference_population_mean_and_sd()
        if sd <= 0.0:
            raise ValueError(
                "%s: neocortical SD is %g; nothing to standardize"
                % (self.tracer.key, sd)
            )
        return PETRelativeDensityMap(
            values=(self._values - mean) / sd,
            receptor=_single_tracer_receptor(self.tracer),
            scale=NEOCORTICAL_SCALE,
        )


def _single_tracer_receptor(tracer):
    """A one-tracer `ReceptorProvenance` view of one z-scored tracer map.

    A per-tracer map's provenance names exactly the tracer it came from;
    what the map then *is* -- a whole receptor's product, one term of the
    published average, or a replication check -- depends on how many tracers
    cover that receptor in the registry, so the wording follows the
    registry rather than being asserted locally.
    """
    covering = RECEPTORS[tracer.receptor_key]
    if tracer.role != "product":
        # Checked before coverage: a replication cohort must never inherit
        # wording that makes it sound like part of the product map.
        combination = (
            "per-tracer neocortical z-score; ingested for replication "
            "checking only, not part of any product map"
        )
    elif len(covering.tracer_keys) == 1:
        combination = (
            "the receptor's only ingested tracer, z-scored against the "
            "neocortex once: the whole published product map, no averaging"
        )
    else:
        combination = (
            "per-tracer neocortical z-score; one input to this receptor's "
            "published multi-tracer combination"
        )
    return ReceptorProvenance(
        key=tracer.receptor_key,
        name="%s (%s alone)"
        % (
            _RECEPTOR_NAMES.get(tracer.receptor_key, tracer.receptor_key),
            tracer.key,
        ),
        system=_RECEPTOR_SYSTEMS.get(tracer.receptor_key, "unknown"),
        tracer_keys=(tracer.key,),
        weights=(1.0,),
        combination=combination,
    )


_RECEPTOR_NAMES = {k: v.name for k, v in RECEPTORS.items()}
_RECEPTOR_SYSTEMS = {k: v.system for k, v in RECEPTORS.items()}


def combine_weighted(maps, weights):
    """Combine same-scale relative maps by weighted average.

    This is Hansen et al.'s rule, applied AFTER each tracer is z-scored:
    ``(sum_i w_i * z_i) / sum_i w_i``. It accepts `PETRelativeDensityMap`s
    only -- raw `ParcellatedPETMap`s have no arithmetic precisely so this
    function cannot be fed incomparable quantities. All maps must carry
    the same scale object.
    """
    maps = list(maps)
    weights = [float(w) for w in weights]
    if not maps:
        raise ValueError("combine_weighted needs at least one map")
    if len(weights) != len(maps):
        raise ValueError("%d maps but %d weights" % (len(maps), len(weights)))
    if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
        raise ValueError(
            "weights must be non-negative with a positive sum, got %s" % (weights,)
        )
    scale = maps[0].scale
    total = np.zeros(N_PARCELS)
    for m, w in zip(maps, weights):
        if m.scale is not scale:
            raise TypeError(
                "cannot combine maps on different scales: %r vs %r"
                % (scale, m.scale)
            )
        if m.receptor.key != maps[0].receptor.key:
            raise ValueError(
                "refusing to combine different receptors (%r, %r): even "
                "commensurable z-scores of two different receptors do not "
                "average into a third thing"
                % (maps[0].receptor.key, m.receptor.key)
            )
        total += w * m._values
    combined = total / sum(weights)
    receptor = maps[0].receptor
    if len(maps) > 1:
        receptor = _multi_tracer_receptor(receptor, maps, weights)
    return PETRelativeDensityMap(values=combined, receptor=receptor, scale=scale)


def _multi_tracer_receptor(receptor, maps, weights):
    # Each input map's own provenance records which tracers went into it, so
    # the combined provenance is their concatenation -- no map carries a
    # bare tracer here.
    tracer_keys = tuple(k for m in maps for k in m.receptor.tracer_keys)
    return ReceptorProvenance(
        key=receptor.key,
        name=receptor.name,
        system=receptor.system,
        tracer_keys=tracer_keys,
        weights=tuple(weights),
        combination=(
            "participant-count-weighted average of per-tracer z-scores, "
            "the published rule"
        ),
    )


# --- loaders ----------------------------------------------------------------


def read_parcellated(path, tracer_key):
    """Read one parcellated tracer CSV from an explicit path.

    The published format is a single comma-separated row of 83 values in
    the Cammoun-033 parcel order (`PARCELS`), no header. Anything else is
    a parse failure.
    """
    tracer = _TRACERS_BY_KEY[tracer_key]
    values = np.loadtxt(path, delimiter=",", dtype=float, ndmin=1)
    if values.shape != (N_PARCELS,):
        raise ValueError(
            "%s: expected one row of %d parcels for %s, got shape %s"
            % (path, N_PARCELS, tracer.key, values.shape)
        )
    return ParcellatedPETMap(values=values, tracer=tracer)


def load_tracer(tracer_key, cache_dir=None, allow_download=False):
    """Load one tracer's parcellated map from the verified cache."""
    path = ensure(
        _TRACERS_BY_KEY[tracer_key].manifest_key,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    return read_parcellated(path, tracer_key)


def load_receptor(receptor_key, cache_dir=None, allow_download=False):
    """Load one receptor system's relative-density map.

    Runs the publication's own aggregation over the verified cache:
    per-tracer z-scoring against the neocortical parcels, then the
    participant-weighted average where the publication averages. Offline
    by default: a cold cache raises `MissingCachedFile`.
    """
    receptor = RECEPTORS[receptor_key]
    maps = []
    for tracer in receptor.tracers:
        raw = load_tracer(
            tracer.key, cache_dir=cache_dir, allow_download=allow_download
        )
        maps.append(raw.relative_to_cortex())
    if len(maps) == 1:
        return maps[0]
    return combine_weighted(maps, receptor.weights)


def load_receptors(keys=None, cache_dir=None, allow_download=False):
    """`{key: PETRelativeDensityMap}` for every (or the named) receptors."""
    keys = receptors_ingested() if keys is None else tuple(keys)
    return {
        key: load_receptor(
            key, cache_dir=cache_dir, allow_download=allow_download
        )
        for key in keys
    }


def coupling_over_connectome(map_or_maps, mechanism_gains, labels=None):
    """Coupling dicts over the ENIGMA connectome regions the scenarios build.

    ``map_or_maps`` is one `PETRelativeDensityMap` or a dict of them keyed
    by receptor; the result matches that shape, so a multi-system
    simulation gets ``{receptor: {node: gain * rho}}``. ``labels`` are
    ENIGMA Desikan-Killiany labels (default: all 82 mapped parcels); a
    label the Cammoun-033 parcellation does not cover raises, naming what
    IS covered.
    """
    if labels is None:
        labels = sorted(CONNECTOME_TO_CAMMOUN)
    labels = list(labels)
    nodes = {label: label for label in labels}
    if isinstance(map_or_maps, PETRelativeDensityMap):
        return map_or_maps.as_coupling(mechanism_gains, nodes)
    return {
        key: m.as_coupling(mechanism_gains, nodes)
        for key, m in map_or_maps.items()
    }


# --- the verification anchor ------------------------------------------------

#: Column index of each ingested receptor in the published 19-column matrix
#: (`data/receptor_names_pet.npy` order).
_PUBLISHED_COLUMNS = {"D2": 9, "GABA_A_BZ": 11, "MOR": 15}

#: What scale each ingested receptor's published column is ON. The deposited
#: matrix is heterogeneous -- verified numerically, not assumed: the D2
#: column has neocortical mean 0 / SD ~0.97 and equals our per-tracer
#: z-score combination bit-for-bit, while the MOR column equals the RAW
#: kantonen file bit-for-bit and the GABA-A/BZ column the RAW norgaard file.
#: The authors combined D2 (two tracers that had to be made commensurable)
#: on the z scale and deposited single-tracer receptors in their own raw
#: tracer units. So only D2's column anchors the z-scoring rule; MOR and
#: GABA-A/BZ's columns anchor parcel order and the raw read instead. Both
#: kinds of check are tests.
PUBLISHED_COLUMN_SCALES = {
    "D2": (
        "z-scored participant-weighted two-tracer combination "
        "(neocortical mean 0, SD < 1)"
    ),
    "MOR": "raw [11C]carfentanil BPnd (kantonen), not z-scored",
    "GABA_A_BZ": "raw flumazenil Vt-derived 'Bmax' (norgaard), not z-scored",
}

#: The published matrix's rows are the cortical parcels only, in PARCELS
#: order with subcortex and brainstem dropped.
_PUBLISHED_ROWS = list(CORTICAL_PARCEL_INDEXES)


def read_published_cortical_columns(path, receptor_key):
    """One receptor's column of the published 68-parcel matrix.

    This is the anchor this module verifies itself against, and the two
    directions of verification are stated rather than blurred:

    * ``D2`` -- the loader's combined map must equal this column
      bit-for-bit, because it is the same computation over the same inputs.
      This is what pins the z-scoring basis (the 68 neocortical parcels)
      and the weighting rule to the publication.
    * ``MOR``, ``GABA_A_BZ`` -- the loader's RAW `ParcellatedPETMap` must
      equal this column bit-for-bit; the deposited column was never
      z-scored, so it pins parcel order and the parse, not the z-scores.

    Reading it is verification only; no loader consumes it. See
    `PUBLISHED_COLUMN_SCALES` for what each column is on.
    """
    if receptor_key not in _PUBLISHED_COLUMNS:
        raise KeyError(
            "no published column recorded for %r; ingested receptors are %s"
            % (receptor_key, ", ".join(sorted(_PUBLISHED_COLUMNS)))
        )
    column = _PUBLISHED_COLUMNS[receptor_key]
    matrix = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
    if matrix.shape[1] <= column:
        raise ValueError(
            "%s: expected at least %d columns, got %d"
            % (path, column + 1, matrix.shape[1])
        )
    if matrix.shape[0] != len(_PUBLISHED_ROWS):
        raise ValueError(
            "%s: expected %d neocortical rows, got %d"
            % (path, len(_PUBLISHED_ROWS), matrix.shape[0])
        )
    return np.array(matrix[:, column], dtype=float)


#: Columns carried by the committed fixture slice of the published matrix
#: (`fixtures/hansen_results_receptor_data_scale033_fixture.csv`), in file
#: order. Defined here rather than in the fixture maker so the slice on
#: disk and the reader that checks it cannot disagree.
PUBLISHED_FIXTURE_COLUMNS = ("D2", "GABA_A_BZ", "MOR")


def read_published_column_slice(path, receptor_key):
    """One receptor's column of the committed 68 x 3 fixture slice.

    Same contract as `read_published_cortical_columns`, against the small
    committed copy the offline tests run on; the maker script writes the
    slice in `PUBLISHED_FIXTURE_COLUMNS` order, verbatim from the verified
    full matrix.
    """
    try:
        index = PUBLISHED_FIXTURE_COLUMNS.index(receptor_key)
    except ValueError:
        raise KeyError(
            "the fixture slice carries %s only, not %r"
            % (", ".join(PUBLISHED_FIXTURE_COLUMNS), receptor_key)
        )
    matrix = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
    expected = (len(_PUBLISHED_ROWS), len(PUBLISHED_FIXTURE_COLUMNS))
    if matrix.shape != expected:
        raise ValueError(
            "%s: expected shape %s, got %s" % (path, expected, matrix.shape)
        )
    return np.array(matrix[:, index], dtype=float)
