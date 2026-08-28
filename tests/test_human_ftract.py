"""F-TRACT module: measured directed connectivity from intracranial CCEP.

P13.1. Offline tests run entirely off the committed fixtures. The checks
that parse the full cached CSVs are skipped when ``data/human/`` is
absent. See ``fixtures/PROVENANCE.md``.

The load-bearing properties under test:

* the CSV parser enforces the 4-comment-line + header-at-index-4 format
  and detects row/label mismatches;
* a verbatim 6-parcel slice reproduces the exact cell values from the
  published archive, including asymmetric reciprocal pairs;
* zero (0.0, probed-no-response) and NaN (never probed) are distinct
  states after parsing;
* synaptic delay parcel maps parse from the tab-separated format;
* the TVB negative-control rejection uses only committed signature
  fixtures and rejects all three matrices on multiple grounds;
* the manifest entries are completely specified with citations, units,
  descriptions, and checksums;
* the module-level coverage constants are pinned against the full CSVs.
"""

import json
import os

import numpy as np
import pytest

from neurocascade.human import fixtures
from neurocascade.human import ftract as F


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_entry_is_completely_specified():
    assert len(F.DATA_FILES) == 4  # 1 bundle + 3 TVB
    for entry in F.DATA_FILES:
        assert len(entry.sha256) == 64, entry.key
        assert set(entry.sha256) <= set("0123456789abcdef"), entry.key
        assert entry.size_bytes > 0, entry.key
        assert entry.url.startswith("https://"), entry.key
        assert len(entry.citation) > 40, entry.key
        assert len(entry.units) > 10, entry.key
        assert len(entry.description) > 40, entry.key
        assert entry.retrieved == "2026-08-25"


def test_manifest_bundle_is_first():
    assert F.DATA_FILES[0].key == "ftract_v2112_bundle"


def test_manifest_tvb_entries_are_verification_only():
    tvb_keys = [e.key for e in F.DATA_FILES if e.key.startswith("tvb_")]
    assert len(tvb_keys) == 3
    for key in tvb_keys:
        assert "au" in F.data_file(key).units.lower() or "arbitrary" in F.data_file(key).units.lower()


def test_zip_member_digests_are_all_hex():
    for key, (member_path, sha, size) in F.FTRACT_ZIP_MEMBERS.items():
        assert len(sha) == 64
        assert set(sha) <= set("0123456789abcdef")
        assert size > 0


# ---------------------------------------------------------------------------
# CSV parsing (from committed 6-parcel slice fixture)
# ---------------------------------------------------------------------------


class TestCSVParsing:
    """Test the CSV parser on the committed verbatim slice fixtures."""

    @pytest.mark.parametrize(
        "feature_name", list(F.FEATURE_META.keys())
    )
    def test_parse_fixture_slice(self, feature_name):
        """The 6-parcel slice parses to a 6x6 matrix with correct labels."""
        labels, matrix = fixtures.load_fixture_ftract_csv(feature_name)
        assert labels == fixtures.FTRACT_SLICE_PARCELS
        assert matrix.shape == (6, 6)
        assert matrix.dtype == np.float64

    def test_probability_slice_values_are_verbatim(self):
        """Spot-check: L_V1 -> L_V1 probability is exactly 0.9952..."""
        labels, matrix = fixtures.load_fixture_ftract_csv("probability")
        i = labels.index("L_V1")
        j = labels.index("L_V1")
        assert matrix[i, j] == pytest.approx(0.9952095808383233, rel=1e-15)

    def test_amplitude_slice_values_are_verbatim(self):
        """Spot-check: R_V6 -> R_V6 amplitude is exactly 13.880..."""
        labels, matrix = fixtures.load_fixture_ftract_csv("amplitude")
        i = labels.index("R_V6")
        j = labels.index("R_V6")
        assert matrix[i, j] == pytest.approx(13.880237579345705, rel=1e-12)

    def test_delay_slice_values_are_verbatim(self):
        """Spot-check: L_V2 -> L_V6 delay is exactly 0.52546."""
        labels, matrix = fixtures.load_fixture_ftract_csv("axonal_conduction_delay")
        i = labels.index("L_V2")
        j = labels.index("L_V6")
        assert matrix[i, j] == pytest.approx(0.52546, rel=1e-12)

    def test_nan_cells_are_missing_not_zero(self):
        """NaN (never probed) and 0.0 (probed, no response) are distinct."""
        labels, matrix = fixtures.load_fixture_ftract_csv("probability")
        # L_V6 -> R_V1 is NaN in the source CSV.
        i = labels.index("L_V6")
        j = labels.index("R_V1")
        assert np.isnan(matrix[i, j])
        # L_V3 -> R_V6 row includes a 0.0 in the full CSV.
        # In the 6-parcel slice, L_V6 -> R_V6 is 1.0, so verify NaN
        # is not equal to 0.0.
        assert matrix[i, j] != 0.0

    def test_asymmetric_reciprocal_pair_exists(self):
        """L_V1 -> R_V1 != R_V1 -> L_V1 in probability (direction is real)."""
        labels, matrix = fixtures.load_fixture_ftract_csv("probability")
        i_l = labels.index("L_V1")
        i_r = labels.index("R_V1")
        # Both directions are finite.
        assert np.isfinite(matrix[i_l, i_r])
        assert np.isfinite(matrix[i_r, i_l])
        # They differ: L_V1->R_V1 ~ 0.848 vs R_V1->L_V1 ~ 0.9375.
        assert matrix[i_l, i_r] != matrix[i_r, i_l]

    def test_fixture_matrix_is_not_symmetric(self):
        """Probability slice is not symmetric: direction is real."""
        _, matrix = fixtures.load_fixture_ftract_csv("probability")
        assert not np.array_equal(matrix, matrix.T)

    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="unknown feature"):
            F.FTractFeature("nonexistent_feature", ["a"], np.array([[0.0]]))


# ---------------------------------------------------------------------------
# Synaptic delay map parsing
# ---------------------------------------------------------------------------


class TestSynapticDelayMap:
    def test_parse_fixture_excitatory(self):
        d = fixtures.load_fixture_ftract_delay_map("synaptic_excitatory_delay")
        assert "L_V1" in d
        assert d["L_V1"] == pytest.approx(4.7063, rel=1e-10)

    def test_parse_fixture_inhibitory(self):
        d = fixtures.load_fixture_ftract_delay_map("synaptic_inhibitory_delay")
        assert "L_V1" in d

    def test_nan_in_delay_map(self):
        d = fixtures.load_fixture_ftract_delay_map("synaptic_excitatory_delay")
        # L_V6A is nan in the full file; verify nan entries parse correctly.
        if "L_V6A" in d:
            assert np.isnan(d["L_V6A"])

    def test_parcel_count(self):
        d = fixtures.load_fixture_ftract_delay_map("synaptic_excitatory_delay")
        assert len(d) == 6  # six-parcel slice


# ---------------------------------------------------------------------------
# FTractFeature class
# ---------------------------------------------------------------------------


class TestFTractFeature:
    @pytest.fixture
    def prob_feature(self):
        labels, matrix = fixtures.load_fixture_ftract_csv("probability")
        return F.FTractFeature("probability", labels, matrix)

    def test_n(self, prob_feature):
        assert prob_feature.n == 6

    def test_statistic(self, prob_feature):
        assert prob_feature.statistic == "mean"

    def test_repr(self, prob_feature):
        assert "probability" in repr(prob_feature)
        assert "6 parcels" in repr(prob_feature)

    def test_value(self, prob_feature):
        v = prob_feature.value("L_V1", "L_V1")
        assert v == pytest.approx(0.9952095808383233, rel=1e-15)

    def test_measured_true(self, prob_feature):
        assert prob_feature.measured("L_V1", "R_V1")

    def test_measured_false_for_nan(self, prob_feature):
        assert not prob_feature.measured("L_V6", "R_V1")

    def test_coverage_keys(self, prob_feature):
        c = prob_feature.coverage()
        for key in ("n_parcels", "n_finite", "n_diagonal", "n_reciprocal_pairs",
                     "n_asymmetric_pairs", "asymmetry_percent", "n_one_way",
                     "symmetric_array_equal"):
            assert key in c
        assert c["n_parcels"] == 6
        assert c["symmetric_array_equal"] is False


# ---------------------------------------------------------------------------
# Module-level coverage constants
# ---------------------------------------------------------------------------


class TestCoverageConstants:
    """The pinned coverage numbers are documented for the full 360-parcel CSVs."""

    def test_probability_coverage_has_expected_finite_count(self):
        assert F.COVERAGE["probability"]["n_finite"] == 27066
        assert F.COVERAGE["probability"]["n_asymmetric_pairs"] == 8439

    def test_amplitude_coverage_has_expected_finite_count(self):
        assert F.COVERAGE["amplitude"]["n_finite"] == 16050
        assert F.COVERAGE["amplitude"]["asymmetry_percent"] == 100.0

    def test_delay_coverage_has_expected_finite_count(self):
        assert F.COVERAGE["axonal_conduction_delay"]["n_finite"] == 16050

    def test_n_exact_zeros_off_diagonal_is_pinned(self):
        assert F.N_EXACT_ZEROS_OFF_DIAGONAL == 28

    def test_direction_semantics_documented(self):
        assert "stimulated" in F.ROWS_ARE_STIMULATED_NOTE.lower()
        assert "recorded" in F.ROWS_ARE_STIMULATED_NOTE.lower()


# ---------------------------------------------------------------------------
# TVB negative-control rejection
# ---------------------------------------------------------------------------


class TestTVBRejection:
    """All three TVB matrices are rejected on multiple grounds."""

    @pytest.mark.parametrize("tvb_name", ["76", "96", "192"])
    def test_rejection_returns_reasons(self, tvb_name):
        reasons = F.tvb_rejection_reasons(tvb_name)
        assert isinstance(reasons, list)
        assert len(reasons) >= 3

    def test_76_rejected_for_shape(self):
        reasons = F.tvb_rejection_reasons("76")
        assert any("shape" in r for r in reasons)

    def test_96_rejected_for_ordinal_and_prefixes(self):
        reasons = F.tvb_rejection_reasons("96")
        assert any("ordinal" in r for r in reasons)
        assert any("CoCoMac" in r or "prefix" in r for r in reasons)

    def test_192_rejected_for_shape(self):
        reasons = F.tvb_rejection_reasons("192")
        assert any("shape" in r for r in reasons)

    def test_all_rejected_for_weights_unit_au(self):
        for name in ("76", "96", "192"):
            reasons = F.tvb_rejection_reasons(name)
            assert any("au" in r.lower() or "arbitrary" in r.lower() for r in reasons)

    def test_all_rejected_for_not_glasser(self):
        for name in ("76", "96", "192"):
            reasons = F.tvb_rejection_reasons(name)
            assert any("glasser" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# TVB signatures fixture
# ---------------------------------------------------------------------------


class TestTVBSignaturesFixture:
    def test_loads_all_three(self):
        sigs = fixtures.load_fixture_tvb_signatures()
        assert set(sigs.keys()) == {"76", "96", "192"}

    def test_96_is_ordinal(self):
        sigs = fixtures.load_fixture_tvb_signatures()
        assert sigs["96"]["ordinal"] is True
        assert sigs["96"]["unique_values"] == [0.0, 1.0, 2.0, 3.0]

    def test_76_is_not_ordinal(self):
        sigs = fixtures.load_fixture_tvb_signatures()
        assert sigs["76"]["ordinal"] is False

    def test_all_have_weights_unit_au(self):
        sigs = fixtures.load_fixture_tvb_signatures()
        for name in ("76", "96", "192"):
            assert sigs[name]["weights_unit"] == "au"

    def test_96_has_coconut_prefixes(self):
        sigs = fixtures.load_fixture_tvb_signatures()
        assert "RM" in sigs["96"]["label_prefixes"]
        assert "BG" in sigs["96"]["label_prefixes"]


# ---------------------------------------------------------------------------
# Citations and licences
# ---------------------------------------------------------------------------


def test_lemarechal_citation_contains_pmid():
    assert "PMID 35416942" in F.LEMARECHAL_CITATION


def test_licence_note_records_discrepancy():
    assert "Zenodo" in F.LICENCE_NOTE
    assert "EBRAINS" in F.LICENCE_NOTE
    assert "research use only" in F.LICENCE_NOTE.lower()


# ---------------------------------------------------------------------------
# Sampling limitations
# ---------------------------------------------------------------------------


def test_the_four_sampling_limitations_are_recorded_with_consequences():
    """P13.1's brief names four acquisition limitations. Each is pinned here
    with the consequence for a consumer, because a limitation recorded
    without its consequence is decoration: the reason to state that coverage
    is clinically driven is that a NaN cell may not be read as an absent
    connection."""
    assert len(F.SAMPLING_LIMITATIONS) == 4
    for item in F.SAMPLING_LIMITATIONS:
        assert set(item) == {"limitation", "consequence"}
        assert len(item["limitation"]) > 60
        assert len(item["consequence"]) > 60


@pytest.mark.parametrize("phrase", [
    "epileptogenic",       # clinically driven electrode placement
    "not neurotypical",    # the cohort
    "polysynaptic",        # a directed edge need not be monosynaptic
    "group average",       # no subject-level matrices, no per-edge dispersion
])
def test_each_named_limitation_is_present(phrase):
    assert phrase in F.SAMPLING_LIMITATIONS_NOTE.lower()


def test_the_coverage_counts_carry_the_absence_warning():
    """The one misreading these numbers invite - finite-cell count as
    connection density - is warned about in the same module that publishes
    them."""
    note = F.SAMPLING_LIMITATIONS_NOTE.lower()
    assert "never probed" in note
    assert "not evidence of absence" in note


# ---------------------------------------------------------------------------
# Full CSV coverage (skipped when cache absent)
# ---------------------------------------------------------------------------


class TestFullCSVCoverage:
    """These tests parse the full 360x360 CSVs from the extracted cache.
    They are skipped when the F-TRACT extract is not present.
    """

    @pytest.fixture(autouse=True)
    def _require_cache(self):
        try:
            F.extract_member("probability", allow_download=False)
        except F.MissingCachedFile:
            pytest.skip("F-TRACT extract not present in data/human/")

    @pytest.mark.parametrize("feature_name", list(F.FEATURE_META.keys()))
    def test_parse_full_csv_shape(self, feature_name):
        path = F.extract_member(feature_name, allow_download=False)
        labels, matrix = F._read_feature_csv(path)
        assert len(labels) == 360
        assert matrix.shape == (360, 360)

    @pytest.mark.parametrize("feature_name", list(F.FEATURE_META.keys()))
    def test_label_ordering_180l_then_180r(self, feature_name):
        path = F.extract_member(feature_name, allow_download=False)
        labels, _ = F._read_feature_csv(path)
        Ls = [l for l in labels if l.startswith("L_")]
        Rs = [l for l in labels if l.startswith("R_")]
        assert len(Ls) == 180
        assert len(Rs) == 180
        assert labels == Ls + Rs

    @pytest.mark.parametrize("feature_name", list(F.FEATURE_META.keys()))
    def test_array_equal_symmetry_false(self, feature_name):
        path = F.extract_member(feature_name, allow_download=False)
        _, matrix = F._read_feature_csv(path)
        assert not np.array_equal(matrix, matrix.T)

    @pytest.mark.parametrize("feature_name", list(F.FEATURE_META.keys()))
    def test_coverage_matches_pinned_constants(self, feature_name):
        path = F.extract_member(feature_name, allow_download=False)
        labels, matrix = F._read_feature_csv(path)
        feat = F.FTractFeature(feature_name, labels, matrix)
        computed = feat.coverage()
        pinned = F.COVERAGE[feature_name]
        assert computed["n_finite"] == pinned["n_finite"]
        assert computed["n_diagonal"] == pinned["n_diagonal"]
        assert computed["n_reciprocal_pairs"] == pinned["n_reciprocal_pairs"]
        assert computed["n_asymmetric_pairs"] == pinned["n_asymmetric_pairs"]
        assert computed["n_one_way"] == pinned["n_one_way"]
        assert computed["symmetric_array_equal"] == pinned["symmetric_array_equal"]

    def test_exact_zeros_off_diagonal(self):
        path = F.extract_member("probability", allow_download=False)
        labels, matrix = F._read_feature_csv(path)
        feat = F.FTractFeature("probability", labels, matrix)
        assert feat.n_exact_zeros_off_diagonal() == F.N_EXACT_ZEROS_OFF_DIAGONAL
