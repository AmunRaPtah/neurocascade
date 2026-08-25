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
