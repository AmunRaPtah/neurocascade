# Human connectome fixture provenance

Both fixture files are slices of the checksum-verified ENIGMA Toolbox
files recorded in `../sources.py` (manifest version 2026-08-16):

* `enigma_hcp_sc_matrix_aparc_with_sctx.csv`
  sha256 `ee3825eed9d83786fbfa4804c13c78ef4b94267ad48b517e2a732f2c05300634`
  (25098 bytes)

* `enigma_hcp_sc_labels_aparc_with_sctx.csv`
  sha256 `331333840228ccb9489d7b63f6d12c08bef9ee12ec0fbe95e27b7d86759ce97a`
  (1249 bytes)

Source commit of the ENIGMA Toolbox repository
(`enigmatoolbox/datasets/matrices/hcp_connectivity/`):

    b08974b55243060cbc1fad12c87048037446e8f7

The twelve fixture regions, in published file order:

    L_bankssts, R_bankssts, L_precentral, R_precentral,
    L_superiorfrontal, Laccumb, Lamyg, Lcaud, Lhippo, Lpal, Lput, Lthal

The matrix fixture is the 12 x 12 principal submatrix of the verified full
matrix at those labels; the label fixture is that one row. Every value is
copied verbatim from the published file (no rounding; the full file's
longest decimal run is five digits). `R_bankssts` genuinely has zero
non-zero entries against the other eleven chosen regions - a real sparsity
feature, kept rather than papered over. Regenerate with:

    .venv/bin/python scripts/make_human_fixture.py

The slice is for offline tests and exploration only. Twelve regions cut
out of an 82-region whole-brain matrix are not a twelve-region model of
the brain: every path through the other 70 regions has been deleted.

## AHBA expression fixture

`enigma_hcp_ahba_dk_expression_fixture.csv` is a verbatim slice of the
checksum-verified ENIGMA AHBA expression matrix recorded in
`../expression.py` (manifest version 2026-08-16):

* `allgenes_stable_r0.2.csv`
  sha256 `4c31801c130b8894cf86785353bb4d814df71e44ade9245ec6e585554cc6b185`
  (19570163 bytes)

Source commit of the `saratheriver/enigma-extra` repository (`ahba/`):

    8c9df6e0c37c66830360ad553807326035e180ed

The fixture is the 12 x 40 principal submatrix at the same twelve fixture
regions listed above (rows) and the forty stable receptor genes in
`expression.STABLE_SYMBOLS` (columns). Every value is copied verbatim from
the published file (no rounding). The twelve regions do not include the two
all-NaN regions (`R_frontalpole`, `R_temporalpole`), so the fixture is fully
finite by construction; the NaN bookkeeping is exercised by a synthetic
object and by the full-cache test, not by this slice. Regenerate with:

    .venv/bin/python scripts/make_human_expression_fixture.py

## Hansen 2022 PET receptor-atlas fixtures (P12.1)

Six files under this directory are slices or copies of the checksum-verified
Hansen JY et al., "Mapping neurotransmitter systems to the structural and
functional organization of the human neocortex", *Nat Neurosci*
2022;25(11):1569-1581 (PMID 36303070) release recorded in `../receptors.py`
(manifest version 2026-08-24, source commit of
`netneurolab/hansen_receptors`: `f8b41da92a733f99368c1f39d4959731aa1316c1`).

Licence: **CC BY-NC-SA 4.0**. The full manifest digests (all seven files,
including the Cammoun-033 parcel-identity xlsx that is *not* shipped here)
live in one place only, the manifest in `../receptors.py`; run
`.venv/bin/python scripts/fetch_human_receptors.py --checksums` for them.

Tracer fixtures - byte-identical copies of the verified cache files, so each
fixture's sha256 equals its manifest entry's:

* `hansen_pet_scale033_MU_carfentanil_hc204_kantonen_fixture.csv`
  (sha256 `abc343a5cd6dec899c35b65770b2b9d9828dfe8e74b9175b07cc6cb6ecda799e`)
* `hansen_pet_scale033_MU_carfentanil_hc39_turtonen_fixture.csv`
  (sha256 `546e46f5b70a23461a81f7c400eefc612349f4cffa4aec95f8c7f4315dff84d4`)
* `hansen_pet_scale033_D2_flb457_hc37_smith_fixture.csv`
  (sha256 `0b4d1835d722e683241c81dcd9817adeb958c8a1a91fde5f92643a52dd869eb7`)
* `hansen_pet_scale033_D2_flb457_hc55_sandiego_fixture.csv`
  (sha256 `fc0c72b672b80837da5dfc1f5bf5c8bf6c3c043ebe3c228bd48897558dd8b4ef`)
* `hansen_pet_scale033_GABAa-bz_flumazenil_hc16_norgaard_fixture.csv`
  (sha256 `f76da8cd9f25fb344e5e1439366a7aeaa0a1db1b046ad6a780a40f8b802ba2f8`)

They can be copies rather than slices because each is a single row of 83
numbers to begin with. One real feature is kept rather than cleaned up:
the kantonen map has an exact `0` at `R_pericalcarine`. The deposited
aggregation carries it too (our copy of that column matches bit-for-bit),
so the publication itself treats it as a measurement, and so do we.

Aggregation-matrix fixture:

* `hansen_results_receptor_data_scale033_fixture.csv` - a 68 x 3
  text-verbatim slice of the paper's deposited 68 x 19 matrix, carrying the
  three ingested receptors' columns in `receptors.PUBLISHED_FIXTURE_COLUMNS`
  order (D2, GABA-A/BZ, MOR). Note those columns are NOT on one shared scale
  in the deposit: D2 is a z-scored two-tracer combination, MOR and GABA-A/BZ
  are single tracers' raw values. See `receptors.PUBLISHED_COLUMN_SCALES`.

Both regenerate from the verified cache with:

    .venv/bin/python scripts/make_human_receptor_fixture.py

which refuses to run against a cold or unverified cache.

## Zilles & Palomero-Gallagher autoradiography fixtures (P12.2)

Five files under this directory are byte-identical copies of the
checksum-verified machine-readable absolute-density deposit recorded in
`../autoradiography.py` (manifest version 2026-08-25, source commit of
`AlGoulas/receptor_principles`:
`1a8b8714271d99ffe62f81f93ac42f3e57c952f1`, last upstream push 2021-04-08):

* `zilles2017_ReceptorNames_fixture.npy`
  (sha256 `f570401cbe1ad960a44b1c4f0f7db30466c60cb77aa13c5dfedff75d375c9171`)
* `zilles2017_RegionNames_fixture.npy`
  (sha256 `c2e830b2885cccc4847abb3d0b9a9b0a9661840ba6742c6533b4e50cc6955440`)
* `zilles2017_ReceptData_S_fixture.npy`
  (sha256 `876b00a55802bba9921b72865ad7b115a6c8c01abef07eb1f0ccc9eba50f15b4`)
* `zilles2017_ReceptData_G_fixture.npy`
  (sha256 `66517300cfc5d40bf1a8f792cb35fa19b9405e3b95922d2b0102569bfa423dd9`)
* `zilles2017_ReceptData_I_fixture.npy`
  (sha256 `ccf010fd061256e438b7fa33d48ff82efd3178f59af5db25f4336d5ae7acba5a`)

They can be copies rather than slices because each is a single small
array to begin with (15 names, 44 area labels, three 44 x 15 uint16 mean
densities in fmol/mg protein).

The deposit's sixth file -- the paper's Supplementary Table 2
(`ReceptData.xlsx`, sha256
`d64bf50ff74f084f618de87a7eeb76f8bfc525526b5dcbf5b51240e3d87b463a`,
70718 bytes) -- is deliberately NOT shipped here, for the same reason
P12.1 does not ship the parcel-identity xlsx: fixtures must stay readable
without openpyxl. When `data/human/` is populated, a cache-present test
parses it and re-checks all 1,965 measured stratum-level means against
the arrays cell-for-cell (1,980 minus area 24's 15 absent granular
values, which the spreadsheet prints as '--').

One real feature is kept rather than cleaned up: **area 24 has no
granular stratum.** It is periarchicortical agranular cingulate cortex;
the publication prints '--' and the .npy arrays encode 0. A zero would
assert a measured density of zero binding sites, which is not what '--'
says, so `for_area("24", ..., stratum="granular")` raises instead. These
are the only zeros in the deposit.

Licences: the article is CC BY 4.0; the `receptor_principles` deposit is
MIT (Copyright (c) 2020 AlGoulas). Both are recorded on every manifest
entry in `../autoradiography.py`. Regenerate the five fixtures from the
verified cache with:

    .venv/bin/python scripts/make_human_autoradiography_fixture.py

which refuses to run against a cold cache.

## CIT168 midbrain-atlas label table (P12.3)

One file under this directory is a byte-identical copy of the label table
inside the checksum-verified CIT168 release bundle recorded in
`../midbrain.py` (manifest version 2026-08-25):

* `cit168_pauli2018_info_fixture.csv`
  (sha256 `4bbcbb9940ad42577ab45a2a5d08527f26fb7c9ee7962bb4ad40283450411af2`,
  635 bytes) - member
  `atl-pauli2018/atl-pauli2018_space-MNI152NLin2009cAsym_info.csv` of the
  verified bundle, whose OUTER digest lives in the manifest there.

Source: Pauli WM, Nili AN, Tyszka JM, "A high-resolution probabilistic in
vivo atlas of human subcortical brain nuclei", *Scientific Data*
2018;5:180063 (PMID 29664465), OSF project r2hvk. The file carries no
numbers at all - 16 rows of `id, label, hemisphere='both',
structure='subcortex'` - and exists so the midbrain nucleus registry's
identities can be pinned offline against exactly what the authors
distributed. It pins identity only, never geometry: volumes come from the
paper's peer-reviewed Table 1 (`midbrain.NUCLEI`), because the deposited
images integrate systematically differently from that table
(`midbrain.DEPOSIT_DISCREPANCY_NOTE`).

The bundle's image members are deliberately NOT shipped here: at ~858 KB
apiece they dwarf every other committed fixture, nothing offline consumes
them, and no product value derives from them. When `data/human/` is
populated, cache-present tests parse the probabilistic member (after
re-verifying its inner digest) and reproduce the recorded deposit-vs-Table-1
integrals, and parse Trutti's two hemisphere maps to confirm they reproduce
that paper's published unthresholded atlas volumes exactly.

Regenerate from the verified cache with:

    .venv/bin/python scripts/make_human_midbrain_fixture.py

which refuses to run against a cold or unverified cache, or if the inner
member's digest has moved.

## F-TRACT measured directed connectivity (P13.1)

### 6-parcel verbatim CSV slices

Three feature slices (probability, amplitude, axonal_conduction_delay) and
two synaptic delay parcel maps, generated by
`scripts/make_human_ftract_fixture.py` from the extracted F-TRACT v21-12
release (Zenodo record 7015415, md5 `df03571e8fce5b030482f239dd0d04ab`).

The six parcels in each slice:

    L_V1, L_V6, L_V2, L_V3, R_V1, R_V6

These span both hemispheres and guarantee: NaN cells (never probed),
asymmetric reciprocal pairs (L_V1→R_V1 ≠ R_V1→L_V1), and the 0.0-vs-NaN
distinction (zero = probed, no response; NaN = never probed).

Each CSV slice contains the verbatim 4-comment-line header from the
original file (lines 0-3), a re-built header with just the 6 parcel
names, and 6 data rows. Every cell value is the original string token
from the published CSV, unmodified — no rounding, no float
re-serialisation. The synapse delay .txt files are tab-separated
parcel-value lines, subset to the same 6 parcels.

CSV file sizes: probability 708172 B, amplitude 749370 B,
axonal_conduction_delay 604868 B (full 360x360); slices are <1 KB each.

F-TRACT licence note: the Zenodo record description states "provided
for free use for research use only"; Zenodo metadata labels it CC BY 4.0;
EBRAINS lists CC BY-NC-SA 4.0. The deposit's own words take priority.
Recorded in `ftract.LICENCE_NOTE`.

Regenerate from the verified extracted cache with:

    .venv/bin/python scripts/make_human_ftract_fixture.py

### TVB connectivity signatures (negative-control rejection)

`tvb_signatures_fixture.json` contains measured signatures for the three
TVB connectivity archives (connectivity_76/96/192.zip, GPL v3, pinned by
sha256+size in `ftract.py`). These are NOT F-TRACT data and are used only
for rejection testing.

Key rejection properties per archive:

| Archive  | shape     | ordinal | labels_glasser | label_prefixes | weights_unit |
|----------|-----------|---------|----------------|----------------|-------------|
| _76      | (76, 76)  | False   | False          | []             | au          |
| _96      | (96, 96)  | True    | False          | BG, MM82a, RM, TM | au       |
| _192     | (192,192) | False   | False          | []             | au          |

Only _96 has ordinal {0,1,2,3} weights; all have `weights_unit = "au"`
(arbitrary units); none has HCP-MMP1 Glasser labels.
