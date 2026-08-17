"""Small committed fixtures: real human connectome data, twelve regions.

Why real data and not a synthetic matrix: the same reason as
`connectome/fixtures` - a hand-written matrix would test array plumbing
and nothing else. The fixture below is a 12 x 12 slice of the
checksum-verified ENIGMA Desikan-Killiany matrix, written back out in the
published format (one label row + one plain numeric matrix). See
`PROVENANCE.md`; `scripts/make_human_fixture.py` regenerates it.

The twelve regions span both hemispheres, five cortical Desikan-Killiany
areas, and all seven subcortical nuclei. The subcortical set includes the
four GABAergic output nuclei (`accumb`, `caud`, `pal`, `put`) so the
builder's inhibitory-exception report is exercised on real labels.
"""

import os

FIXTURE_DIR = os.path.abspath(os.path.dirname(__file__))

FIXTURE_MATRIX_PATH = os.path.join(
    FIXTURE_DIR, "enigma_hcp_sc_matrix_aparc_with_sctx_fixture.csv"
)
FIXTURE_LABELS_PATH = os.path.join(
    FIXTURE_DIR, "enigma_hcp_sc_labels_aparc_with_sctx_fixture.csv"
)
FIXTURE_EXPRESSION_PATH = os.path.join(
    FIXTURE_DIR, "enigma_hcp_ahba_dk_expression_fixture.csv"
)


def load_fixture_connectome():
    """Load the fixture matrix. No cache, no network, no download path."""
    from ..matrix import read_connectome

    return read_connectome(FIXTURE_MATRIX_PATH, FIXTURE_LABELS_PATH)


def load_fixture_atlas():
    """The twelve-region atlas alone, without the matrix."""
    from ..atlas import atlas_from_labels
    from ..matrix import _read_labels

    return atlas_from_labels(_read_labels(FIXTURE_LABELS_PATH))


def load_fixture_expression():
    """The twelve-region x forty-gene AHBA expression slice.

    No cache, no network, no download path. The slice is verbatim real data
    from the checksum-verified ENIGMA AHBA file (see PROVENANCE.md);
    `scripts/make_human_expression_fixture.py` regenerates it.
    """
    from ..expression import read_expression

    return read_expression(FIXTURE_EXPRESSION_PATH)
