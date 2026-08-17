"""The human expression layer: ENIGMA AHBA Desikan-Killiany ingestion.

Every test runs offline, off the committed fixture in
`src/neurocascade/human/fixtures/`, which is a real twelve-region x
forty-gene slice of the checksum-verified ENIGMA AHBA file. See
`fixtures/PROVENANCE.md`. One test additionally checks the real 82-region
file and is skipped when the gitignored cache is absent.
"""

import numpy as np
import pytest

from neurocascade import human
from neurocascade.human import expression as expr
from neurocascade.human import fixtures


#: Reference values copied verbatim from the checksum-verified full file
#: `data/human/allgenes_stable_r0.2.csv` (manifest version 2026-08-16). These
#: pin the fixture against the real published data, so a regenerated or edited
#: fixture cannot drift without a test noticing.
DRD2_LCAUD = 0.9726974474405531
DRD2_LPUT = 0.8574508285623009
OPRM1_LACCUMB = 0.8342042449496524


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_expression():
    return fixtures.load_fixture_expression()


@pytest.fixture(scope="module")
def fixture_atlas():
    return fixtures.load_fixture_atlas()


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_manifest_entry_is_completely_specified():
    assert expr.DATA_FILES, "manifest is empty"
    for entry in expr.DATA_FILES:
        assert len(entry.sha256) == 64, entry.key
        assert set(entry.sha256) <= set("0123456789abcdef"), entry.key
        assert entry.size_bytes > 0, entry.key
        assert entry.url.startswith("https://"), entry.key
        assert entry.filename and "/" not in entry.filename, entry.key
        assert len(entry.citation) > 40, entry.key
        assert len(entry.units) > 20, entry.key
        assert len(entry.description) > 40, entry.key
        assert entry.retrieved
        assert entry.volatility == "static"


def test_url_is_pinned_to_a_commit():
    for entry in expr.DATA_FILES:
        assert "/enigma-extra/8c9df6e0c37c66830360ad553807326035e180ed/" in entry.url


def test_units_say_not_a_density_and_bounded():
    units = expr.DATA_FILES[0].units
    assert "NOT a receptor density" in units
    assert "[0, 1]" in units


def test_checksum_lines_are_sha256sum_c_input():
    lines = expr.checksum_lines()
    assert len(lines) == len(expr.DATA_FILES)
    by_filename = {entry.filename: entry.sha256 for entry in expr.DATA_FILES}
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert by_filename[filename] == digest


def test_missing_cache_entry_raises_instead_of_downloading(tmp_path):
    with pytest.raises(expr.MissingCachedFile):
        expr.ensure(
            expr.DEFAULT_EXPRESSION_KEY,
            cache_dir=str(tmp_path),
            allow_download=False,
        )


# ---------------------------------------------------------------------------
# The gene registry and the searched-and-empty record
# ---------------------------------------------------------------------------


def test_stable_symbols_are_unique_and_registered():
    assert len(set(expr.STABLE_SYMBOLS)) == len(expr.STABLE_SYMBOLS)
    for symbol in expr.STABLE_SYMBOLS:
        assert symbol in expr.GENES_BY_SYMBOL, symbol


def test_availability_partitions_the_registry_without_overlap():
    stable = set(expr.STABLE_SYMBOLS)
    unstable = set(expr.UNTHRESHOLDED_ONLY)
    absent = set(expr.ABSENT)
    assert not (stable & unstable)
    assert not (stable & absent)
    assert not (unstable & absent)
    assert stable | unstable | absent == set(expr.GENES_BY_SYMBOL)
    assert len(expr.STABLE_SYMBOLS) == 40
    assert len(expr.UNTHRESHOLDED_ONLY) == 2
    assert len(expr.ABSENT) == 8


def test_every_registry_gene_has_a_valid_availability():
    for symbol in expr.GENES_BY_SYMBOL:
        assert expr.availability(symbol) in ("stable", "unthresholded-only", "absent")


def test_known_searched_and_empty_records():
    # SERT/NET and the delta opioid receptor have no human expression map
    # in either ENIGMA AHBA release; DAT and alpha-1A survive only in the
    # unthresholded one.
    assert expr.availability("SLC6A4") == "absent"
    assert expr.availability("SLC6A2") == "absent"
    assert expr.availability("OPRD1") == "absent"
    assert expr.availability("ADORA2A") == "absent"
    assert expr.availability("DRD3") == "absent"
    assert expr.availability("SLC6A3") == "unthresholded-only"
    assert expr.availability("ADRA1A") == "unthresholded-only"
    assert expr.availability("DRD2") == "stable"


def test_unknown_gene_is_an_error_naming_the_table():
    with pytest.raises(KeyError):
        expr.gene("NOT_A_GENE")


def test_genes_for_system():
    assert {g.symbol for g in expr.genes_for_system("opioid")} >= {
        "OPRM1", "OPRK1", "OPRD1"
    }
    assert {g.symbol for g in expr.genes_for_system("transporter")} == {
        "SLC6A2", "SLC6A3", "SLC6A4"
    }


# ---------------------------------------------------------------------------
# The loader, off the fixture
# ---------------------------------------------------------------------------


def test_fixture_shape_and_labels(fixture_expression, fixture_atlas):
    assert fixture_expression.shape == (40, 12)
    assert list(fixture_expression.region_labels) == fixture_atlas.labels
    assert len(fixture_expression.symbols) == 40


def test_fixture_is_real_published_data(fixture_expression):
    # Reference values copied verbatim from the verified full file.
    drd2 = fixture_expression.for_gene("DRD2")
    assert drd2[fixture_expression.region_index_of("Lcaud")] == pytest.approx(DRD2_LCAUD)
    assert drd2[fixture_expression.region_index_of("Lput")] == pytest.approx(DRD2_LPUT)
    oprm1 = fixture_expression.for_gene("OPRM1")
    assert oprm1[fixture_expression.region_index_of("Laccumb")] == pytest.approx(OPRM1_LACCUMB)


def test_fixture_is_finite_and_bounded(fixture_expression):
    assert np.all(np.isfinite(fixture_expression.values))
    assert np.all(fixture_expression.values >= 0.0)
    assert np.all(fixture_expression.values <= 1.0)
    assert fixture_expression.coverage() == pytest.approx(1.0)


def test_fixture_has_no_missing_regions_or_genes(fixture_expression):
    assert fixture_expression.regions_without_values() == []
    assert fixture_expression.genes_without_values() == []


def test_values_for_submatrix(fixture_expression):
    block = fixture_expression.values_for(["DRD1", "DRD2"])
    assert block.shape == (2, 12)
    assert np.array_equal(block[0], fixture_expression.for_gene("DRD1"))


def test_for_gene_raises_for_absent_gene(fixture_expression):
    with pytest.raises(KeyError) as caught:
        fixture_expression.for_gene("SLC6A4")
    assert "absent" in str(caught.value)
    assert "SLC6A4" in str(caught.value)


def test_for_gene_raises_for_unthresholded_only_gene(fixture_expression):
    with pytest.raises(KeyError) as caught:
        fixture_expression.for_gene("SLC6A3")
    assert "unthresholded-only" in str(caught.value)


def test_anatomy_d2_and_d1_are_striatum_dominant(fixture_expression):
    for symbol in ("DRD1", "DRD2"):
        row = fixture_expression.for_gene(symbol)
        top = fixture_expression.region_labels[int(np.argmax(row))]
        assert top in ("Lcaud", "Lput", "Laccumb"), (symbol, top)


def test_anatomy_htr2a_and_gabra1_are_cortex_dominant(fixture_expression):
    htr2a = fixture_expression.for_gene("HTR2A")
    order = np.argsort(htr2a)[::-1]
    top_two = {fixture_expression.region_labels[i] for i in order[:2]}
    assert top_two <= {"L_bankssts", "R_bankssts", "L_precentral", "R_precentral"}

    gabra1 = fixture_expression.for_gene("GABRA1")
    assert gabra1[fixture_expression.region_index_of("R_precentral")] > gabra1[
        fixture_expression.region_index_of("Lcaud")
    ]


# ---------------------------------------------------------------------------
# Parse honesty: NaN is NaN, [0, 1] is enforced
# ---------------------------------------------------------------------------


def _write_csv(tmp_path, header, rows):
    import csv

    path = tmp_path / "expr.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return str(path)


def test_nan_cells_are_kept_as_nan(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2", "HTR2A", "GABRA1"],
        [
            ["Lcaud", "0.5", "0.1", "0.2"],
            ["Rfrontalpole", "NaN", "", "0.3"],
            ["Lput", "0.4", "0.2", "0.1"],
        ],
    )
    loaded = expr.read_expression(path, symbols=["DRD2", "HTR2A", "GABRA1"])
    assert np.isnan(loaded.for_gene("DRD2")[1])
    assert np.isnan(loaded.for_gene("HTR2A")[1])
    assert loaded.regions_without_values() == []  # GABRA1 disagrees, so no all-NaN col
    assert loaded.genes_without_values() == []


def test_all_nan_region_is_reported(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2"],
        [["Lcaud", "0.5"], ["Rfrontalpole", "NaN"]],
    )
    loaded = expr.read_expression(path, symbols=["DRD2"])
    assert loaded.regions_without_values() == ["Rfrontalpole"]


def test_value_outside_unit_interval_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2"],
        [["Lcaud", "1.2"]],
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expr.read_expression(path, symbols=["DRD2"])


def test_requesting_a_gene_the_file_lacks_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2"],
        [["Lcaud", "0.5"]],
    )
    with pytest.raises(ValueError, match="not in the file"):
        expr.read_expression(path, symbols=["DRD2", "SLC6A4"])


def test_ragged_row_rejected(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2", "HTR2A"],
        [["Lcaud", "0.5"]],
    )
    with pytest.raises(ValueError, match="cells"):
        expr.read_expression(path, symbols=["DRD2", "HTR2A"])


# ---------------------------------------------------------------------------
# Density conversion
# ---------------------------------------------------------------------------


def test_relative_density_max(fixture_expression):
    rho = expr.relative_density(fixture_expression, "DRD2", normalization="max")
    assert rho.symbol == "DRD2"
    assert np.nanmax(rho.values) == pytest.approx(1.0)
    assert rho.for_region("Lcaud") == pytest.approx(1.0)
    assert rho.for_region("Lput") == pytest.approx(DRD2_LPUT / DRD2_LCAUD)


def test_relative_density_mean(fixture_expression):
    rho = expr.relative_density(fixture_expression, "DRD2", normalization="mean")
    assert np.nanmean(rho.values) == pytest.approx(1.0)


def test_relative_density_reference_region(fixture_expression):
    rho = expr.relative_density(
        fixture_expression, "DRD2", normalization="reference-region", reference="Lput"
    )
    assert rho.for_region("Lput") == pytest.approx(1.0)
    assert rho.for_region("Lcaud") == pytest.approx(DRD2_LCAUD / DRD2_LPUT)


def test_relative_density_requires_normalization(fixture_expression):
    with pytest.raises(TypeError):
        expr.relative_density(fixture_expression, "DRD2")


def test_relative_density_rejects_unknown_normalization(fixture_expression):
    with pytest.raises(ValueError):
        expr.relative_density(fixture_expression, "DRD2", normalization="median")


def test_as_coupling_maps_nodes_to_regions(fixture_expression):
    rho = expr.relative_density(fixture_expression, "DRD2", normalization="max")
    coupling = rho.as_coupling(
        mechanism_gain=-3.0,
        nodes={"caudate": "Lcaud", "putamen": "Lput"},
    )
    assert coupling["caudate"] == pytest.approx(-3.0 * 1.0)
    assert coupling["putamen"] == pytest.approx(-3.0 * DRD2_LPUT / DRD2_LCAUD)


def test_as_coupling_refuses_an_unmeasured_region(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2"],
        [["Lcaud", "0.5"], ["Rfrontalpole", "NaN"]],
    )
    loaded = expr.read_expression(path, symbols=["DRD2"])
    rho = expr.relative_density(loaded, "DRD2", normalization="max")
    with pytest.raises(ValueError, match="no measured expression"):
        rho.as_coupling(-3.0, {"pole": "Rfrontalpole"})


# ---------------------------------------------------------------------------
# The full cached matrix, when present (optional, skipped offline)
# ---------------------------------------------------------------------------


def test_full_cached_expression_matches_the_connectome():
    try:
        conn = human.load_connectome(allow_download=False)
        loaded = human.load_expression(allow_download=False)
    except (human.fetch.MissingCachedFile, expr.MissingCachedFile):
        pytest.skip("data/human/ cache absent; run scripts/fetch_human_expression.py")
    assert loaded.shape == (40, 82)
    assert list(loaded.region_labels) == conn.labels
    assert set(loaded.regions_without_values()) == {"R_frontalpole", "R_temporalpole"}
    assert loaded.genes_without_values() == []
    assert loaded.coverage() == pytest.approx(80.0 / 82.0)
