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
