"""The human data layer: ENIGMA Desikan-Killiany ingestion and building.

Every test runs offline, off the committed fixtures in
`src/neurocascade/human/fixtures/`, which are a real twelve-region slice
of the checksum-verified ENIGMA files. See `fixtures/PROVENANCE.md`.

One test (`test_full_cached_matrix_...`) additionally checks the real
82-region file and is skipped when the gitignored cache is absent. It is
an extra, not a substitute: everything the layer needs is covered by the
offline tests above it.
"""

import numpy as np
import pytest

from neurocascade import human
from neurocascade.human import atlas as atlas_module
from neurocascade.human import build, fetch, fixtures, matrix, sources
from neurocascade.human import neurotransmitter as nt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conn():
    return fixtures.load_fixture_connectome()


@pytest.fixture(scope="module")
def atlas():
    return fixtures.load_fixture_atlas()


# ---------------------------------------------------------------------------
# The manifest: the committed half of "reproducible without committing the
# binaries".
# ---------------------------------------------------------------------------


def test_every_manifest_entry_is_completely_specified():
    assert sources.DATA_FILES, "manifest is empty"
    for entry in sources.DATA_FILES:
        assert len(entry.sha256) == 64, entry.key
        assert set(entry.sha256) <= set("0123456789abcdef"), entry.key
        assert entry.size_bytes > 0, entry.key
        assert entry.url.startswith("https://"), entry.key
        assert entry.filename and "/" not in entry.filename, entry.key
        assert len(entry.citation) > 40, entry.key
        assert len(entry.units) > 20, entry.key
        assert len(entry.description) > 40, entry.key
        assert entry.retrieved


def test_manifest_keys_and_filenames_are_unique():
    keys = [entry.key for entry in sources.DATA_FILES]
    filenames = [entry.filename for entry in sources.DATA_FILES]
    assert len(set(keys)) == len(keys)
    assert len(set(filenames)) == len(filenames)


def test_default_and_labels_keys_resolve():
    assert sources.data_file(sources.DEFAULT_MATRIX_KEY)
    assert sources.data_file(sources.LABELS_KEY)
    assert sources.LABELS_KEY not in sources.connectivity_keys()
    assert sources.DEFAULT_MATRIX_KEY in sources.connectivity_keys()


def test_unknown_manifest_key_names_the_alternatives():
    with pytest.raises(KeyError) as caught:
        sources.data_file("no_such_dataset")
    assert sources.DEFAULT_MATRIX_KEY in str(caught.value)


def test_checksum_lines_are_sha256sum_c_input():
    lines = fetch.checksum_lines()
    assert len(lines) == len(sources.DATA_FILES)
    by_filename = {entry.filename: entry.sha256 for entry in sources.DATA_FILES}
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert by_filename[filename] == digest


def test_units_say_undirected_and_unsigned():
    units = sources.data_file(sources.DEFAULT_MATRIX_KEY).units
    assert "UNDIRECTED" in units
    assert "UNSIGNED" in units


def test_urls_are_pinned_to_a_commit():
    for entry in sources.DATA_FILES:
        assert "/ENIGMA/b08974b55243060cbc1fad12c87048037446e8f7/" in entry.url


# ---------------------------------------------------------------------------
# Fetch and verification, without touching the network
# ---------------------------------------------------------------------------


def test_missing_cache_entry_raises_instead_of_downloading(tmp_path):
    with pytest.raises(fetch.MissingCachedFile):
        fetch.ensure(
            sources.DEFAULT_MATRIX_KEY,
            cache_dir=str(tmp_path),
            allow_download=False,
        )


# ---------------------------------------------------------------------------
# The atlas: label parsing and region names
# ---------------------------------------------------------------------------


def test_atlas_parses_labels(atlas):
    assert len(atlas) == 12
    assert atlas["L_bankssts"].hemisphere == "L"
    assert atlas["L_bankssts"].kind == "cortex"
    assert atlas["L_bankssts"].name == "banks of the superior temporal sulcus"
    assert atlas["R_precentral"].hemisphere == "R"
    assert atlas["R_precentral"].name == "precentral gyrus"
    assert atlas["Laccumb"].kind == "subcortex"
    assert atlas["Laccumb"].name == "nucleus accumbens"
    assert atlas["Lthal"].name == "thalamus"


def test_atlas_cortex_and_subcortex_counts(atlas):
    assert len(atlas.cortical()) == 5
    assert len(atlas.subcortical()) == 7


def test_parse_label_rejects_unknown():
    with pytest.raises(ValueError):
        atlas_module.parse_label("X_bankssts")


def test_unknown_suffix_rejected():
    with pytest.raises(ValueError):
        atlas_module.atlas_from_labels(["L_notaregion"])


# ---------------------------------------------------------------------------
# The matrix: faithful, undirected, unsigned
# ---------------------------------------------------------------------------


def test_fixture_shape_and_labels(conn):
    assert conn.n == 12
    assert conn.labels == [
        "L_bankssts",
        "R_bankssts",
        "L_precentral",
        "R_precentral",
        "L_superiorfrontal",
        "Laccumb",
        "Lamyg",
        "Lcaud",
        "Lhippo",
        "Lpal",
        "Lput",
        "Lthal",
    ]


def test_fixture_is_real_published_data(conn):
    index = {label: i for i, label in enumerate(conn.labels)}
    # Reference values read from the full verified file; see PROVENANCE.md.
    assert conn.weights[index["L_precentral"], index["L_superiorfrontal"]] == pytest.approx(10.75)
    assert conn.weights[index["Laccumb"], index["Lput"]] == pytest.approx(9.112)
    assert conn.weights[index["Lpal"], index["Lput"]] == pytest.approx(10.929)
    assert conn.weights[index["Lthal"], index["L_precentral"]] == pytest.approx(9.6063)


def test_fixture_is_symmetric_nonnegative_finite(conn):
    assert np.array_equal(conn.weights, conn.weights.T)
    assert np.all(conn.weights >= 0)
    assert np.all(np.isfinite(conn.weights))


def test_projection_matrix_is_identity_for_symmetric(conn):
    weights, labels = conn.projection_matrix()
    assert np.array_equal(weights, conn.weights)
    assert labels == conn.labels


def test_read_connectome_rejects_asymmetric(tmp_path):
    # A non-symmetric matrix is a parse failure: tractography is undirected.
    matrix_file = tmp_path / "m.csv"
    labels_file = tmp_path / "l.csv"
    matrix_file.write_text("1,2\n0,1\n")
    labels_file.write_text("L_bankssts,R_bankssts\n")
    with pytest.raises(ValueError, match="not symmetric"):
        matrix.read_connectome(str(matrix_file), str(labels_file))


# ---------------------------------------------------------------------------
# The sign table: cited per-region transmitter identity
# ---------------------------------------------------------------------------


def test_every_sign_rule_is_completely_specified():
    assert nt.SIGN_RULES, "sign table is empty"
    for rule in nt.SIGN_RULES:
        assert rule.transmitter in nt.TRANSMITTER_SIGN, rule
        assert rule.citations, rule
        assert all(key in nt.CITATIONS for key in rule.citations), rule
        assert rule.note, rule


def test_sign_rules_cover_every_valid_atlas_region(atlas):
    rules = nt.SignRules()
    for region in atlas.regions:
        assert rules.assign(region).rule is not None, region.label


def test_fixture_sign_assignments(atlas):
    assignments = {a.label: a for a in nt.SignRules().assign_all(atlas)}
    assert assignments["L_bankssts"].sign == 1.0
    assert assignments["L_bankssts"].transmitter == nt.GLUTAMATERGIC
    assert assignments["Lhippo"].sign == 1.0
    assert assignments["Lthal"].sign == 1.0
    assert assignments["Laccumb"].sign == -1.0
    assert assignments["Laccumb"].transmitter == nt.GABAERGIC
    assert assignments["Lcaud"].sign == -1.0
    assert assignments["Lpal"].sign == -1.0
    assert assignments["Lput"].sign == -1.0
    assert assignments["Lamyg"].sign is None
    assert assignments["Lamyg"].transmitter == nt.MIXED
    assert not assignments["Lamyg"].is_signed


def test_every_cortical_region_is_glutamatergic(atlas):
    for assignment in nt.SignRules().assign_all(atlas):
        if assignment.kind == "cortex":
            assert assignment.sign == 1.0
            assert assignment.transmitter == nt.GLUTAMATERGIC


def test_duplicate_subcortical_rule_rejected():
    duplicate = nt.SIGN_RULES + (
        nt.RegionSignRule(
            "subcortex", "thal", nt.GABAERGIC, ("Jones2007",), "duplicate"
        ),
    )
    with pytest.raises(ValueError):
        nt.SignRules(duplicate)


def test_no_rule_is_unsigned_not_an_error():
    unknown = atlas_module.Region(
        id=0,
        label="L_madeup",
        hemisphere="L",
        kind="subcortex",
        suffix="madeup",
        name="made up nucleus",
    )
    assignment = nt.SignRules().assign(unknown)
    assert assignment.rule is None
    assert assignment.sign is None
    assert assignment.transmitter is None
    assert not assignment.is_signed


def test_citations_used_are_all_known(atlas):
    for key in nt.SignRules().citations_used(atlas):
        assert key in nt.CITATIONS


# ---------------------------------------------------------------------------
# The builder: a species-neutral CircuitNetwork signed per source region
# ---------------------------------------------------------------------------


def test_build_circuit_network(conn):
    built = human.build_circuit_network(conn)
    assert built.n == 12
    assert built.network.n == 12
    assert built.labels == conn.labels
    assert [node.name for node in built.network.nodes] == conn.labels


def test_unsigned_amygdala_column_is_excluded(conn):
    built = human.build_circuit_network(conn)
    amygdala = conn.labels.index("Lamyg")
    assert np.all(built.connectivity[:, amygdala] == 0.0)
    # ... while its INCOMING edges (row) from signed sources survive.
    assert np.any(built.connectivity[amygdala, :] != 0.0)


def test_cortical_column_stays_excitatory_and_striatal_column_inhibitory(conn):
    built = human.build_circuit_network(conn)
    cortex = conn.labels.index("L_precentral")
    striatum = conn.labels.index("Lcaud")
    assert np.all(built.connectivity[:, cortex] >= 0.0)
    assert np.any(built.connectivity[:, cortex] > 0.0)
    assert np.all(built.connectivity[:, striatum] <= 0.0)
    assert np.any(built.connectivity[:, striatum] < 0.0)


def test_weight_signs_each_direction_by_its_source(conn):
    built = human.build_circuit_network(conn)
    # Same-sign pair: stays symmetric and positive.
    assert built.weight("L_precentral", "L_superiorfrontal") == pytest.approx(10.75)
    assert built.weight("L_superiorfrontal", "L_precentral") == pytest.approx(10.75)
    # Opposite-sign pair: +W one way (cortex onto striatum), -W the other.
    assert built.weight("L_precentral", "Lcaud") == pytest.approx(11.163)
    assert built.weight("Lcaud", "L_precentral") == pytest.approx(-11.163)
    # Unsigned source -> any target is excluded...
    assert built.weight("Lamyg", "Lthal") == 0.0
    # ...but the reverse direction (thalamus onto amygdala) is signed +1.
    assert built.weight("Lthal", "Lamyg") == pytest.approx(8.0214)


def test_report_sign_counts_and_inhibitory_sources(conn):
    built = human.build_circuit_network(conn)
    assert built.report.transmitter_counts() == {
        nt.GLUTAMATERGIC: 7,
        nt.GABAERGIC: 4,
        nt.MIXED: 1,
    }
    assert built.report.inhibitory_sources == ["Laccumb", "Lcaud", "Lpal", "Lput"]
    assert [a.label for a in built.report.unsigned_regions()] == ["Lamyg"]
    summary = built.report.summary()
    assert "Lamyg" in summary
    assert "glutamatergic" in summary
    assert "gabaergic" in summary
    assert "mixed" in summary


def test_report_signed_weight_fraction(conn):
    built = human.build_circuit_network(conn)
    total = conn.weights.sum()
    signed_columns = np.array(
        [a.is_signed for a in built.report.assignments], dtype=bool
    )
    expected = conn.weights[:, signed_columns].sum() / total
    assert built.report.signed_weight_fraction == pytest.approx(expected)
    assert built.report.excluded_weight_fraction == pytest.approx(1.0 - expected)
    assert set(built.report.excluded_by_transmitter) == {nt.MIXED}
    amygdala = conn.labels.index("Lamyg")
    expected_mixed = conn.weights[:, amygdala].sum() / total
    assert built.report.excluded_by_transmitter[nt.MIXED] == pytest.approx(expected_mixed)


def test_unsigned_raise_names_amygdala(conn):
    with pytest.raises(ValueError) as caught:
        human.build_circuit_network(conn, unsigned_sources="raise")
    assert "Lamyg" in str(caught.value)


def test_bad_unsigned_policy_rejected(conn):
    with pytest.raises(ValueError):
        human.build_circuit_network(conn, unsigned_sources="guess")


# ---------------------------------------------------------------------------
# The full cached matrix, when present (optional, skipped offline)
# ---------------------------------------------------------------------------


def test_full_cached_matrix_is_82_regions():
    try:
        conn = human.load_connectome(allow_download=False)
    except fetch.MissingCachedFile:
        pytest.skip("data/human/ cache absent; run scripts/fetch_human_connectome.py")
    assert conn.n == 82
    assert len(conn.atlas.cortical()) == 68
    assert len(conn.atlas.subcortical()) == 14
    assert conn.atlas.hemisphere_counts() == {
        ("L", "cortex"): 34,
        ("R", "cortex"): 34,
        ("L", "subcortex"): 7,
        ("R", "subcortex"): 7,
    }
    built = human.build_circuit_network(conn)
    assert built.network.n == 82
    assert built.report.transmitter_counts() == {
        nt.GLUTAMATERGIC: 72,
        nt.GABAERGIC: 8,
        nt.MIXED: 2,
    }
    assert len(built.report.inhibitory_sources) == 8
    assert len(built.report.unsigned_regions()) == 2
    assert built.report.inhibitory_sources == [
        label
        for label in conn.labels
        if conn.atlas[label].suffix in ("accumb", "caud", "pal", "put")
    ]
    assert built.report.unsigned_regions()[0].label == "Lamyg"
    assert built.report.unsigned_regions()[1].label == "Ramyg"
