"""P3.1: ingestion of the Allen mouse connectome and the CCF ontology.

Every test here runs offline, off the committed fixtures in
`src/neurocascade/connectome/fixtures/`. Those fixtures are real published
data - fifteen regions sliced out of the checksum-verified full files, in
the published formats - so these tests exercise the same parsers, the same
two-row CSV header, the same `#NUM!` Excel tokens and the same acronym-to-
id join that the full matrices go through. See `fixtures/PROVENANCE.md`.

One test (`test_full_cached_matrix_...`) additionally checks the real
291-region file, and is skipped when the gitignored cache is absent. It is
an extra, not a substitute: everything the acceptance criteria ask for is
covered by the offline tests above it.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from neurocascade.legacy import connectome
from neurocascade.legacy.connectome import fetch, fixtures, matrix, ontology as ontology_module
from neurocascade.legacy.connectome import sources


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ccf():
    return connectome.load_fixture_ontology()


@pytest.fixture(scope="module")
def knox(ccf):
    return connectome.load_fixture_connectome(ontology=ccf)


@pytest.fixture(scope="module")
def oh(ccf):
    return connectome.load_fixture_connectome(
        fixtures.FIXTURE_OH_KEY, ontology=ccf
    )


# ---------------------------------------------------------------------------
# The manifest: the committed half of "reproducible without committing the
# binaries". If these degrade, the checksums stop meaning anything.
# ---------------------------------------------------------------------------


def test_every_manifest_entry_is_completely_specified():
    assert sources.DATA_FILES, "manifest is empty"
    for entry in sources.DATA_FILES:
        assert len(entry.sha256) == 64, entry.key
        assert set(entry.sha256) <= set("0123456789abcdef"), entry.key
        assert entry.size_bytes > 0, entry.key
        assert entry.url.startswith("https://"), entry.key
        assert entry.filename and "/" not in entry.filename, entry.key
        # A file with no citation and no units is a number with no source,
        # which is the one thing this codebase does not allow.
        assert len(entry.citation) > 40, entry.key
        assert len(entry.units) > 20, entry.key
        assert len(entry.description) > 40, entry.key
        assert entry.retrieved


def test_manifest_keys_and_filenames_are_unique():
    keys = [entry.key for entry in sources.DATA_FILES]
    filenames = [entry.filename for entry in sources.DATA_FILES]
    assert len(set(keys)) == len(keys)
    assert len(set(filenames)) == len(filenames)


def test_default_and_ontology_keys_resolve():
    assert sources.data_file(sources.DEFAULT_CONNECTIVITY_KEY)
    assert sources.data_file(sources.ONTOLOGY_KEY)
    assert sources.ONTOLOGY_KEY not in sources.connectivity_keys()
    assert sources.DEFAULT_CONNECTIVITY_KEY in sources.connectivity_keys()


def test_unknown_manifest_key_names_the_alternatives():
    with pytest.raises(KeyError) as caught:
        sources.data_file("no_such_dataset")
    assert sources.DEFAULT_CONNECTIVITY_KEY in str(caught.value)


def test_checksum_lines_are_sha256sum_c_input():
    lines = fetch.checksum_lines()
    assert len(lines) == len(sources.DATA_FILES)
    by_filename = {entry.filename: entry.sha256 for entry in sources.DATA_FILES}
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert by_filename[filename] == digest


def test_units_distinguish_density_from_strength():
    """The two datasets' matrices are NOT the same quantity.

    Knox's default file is a connection DENSITY (divided by both region
    sizes); Oh's Supplementary Table 3 is a connection STRENGTH. A caller
    who mixes them is comparing different units, so the manifest has to say
    so rather than let 'the Allen connectome' imply one number.
    """
    knox_units = sources.data_file(
        "knox2019_normalized_connection_density"
    ).units
    oh_units = sources.data_file("oh2014_supplementary_table_3").units
    assert "density" in knox_units
    assert "strength" in oh_units
    assert "NOT a connection density" in oh_units


# ---------------------------------------------------------------------------
# Fetch and verification, without touching the network
# ---------------------------------------------------------------------------


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    payload = b"connectome" * 5000
    path.write_bytes(payload)
    assert fetch.sha256_of(str(path)) == hashlib.sha256(payload).hexdigest()


def test_missing_cache_entry_raises_instead_of_downloading(tmp_path):
    with pytest.raises(fetch.MissingCachedFile):
        fetch.ensure(
            sources.DEFAULT_CONNECTIVITY_KEY,
            cache_dir=str(tmp_path),
            allow_download=False,
        )


def test_corrupted_cache_entry_is_rejected(tmp_path):
    entry = sources.data_file(sources.DEFAULT_CONNECTIVITY_KEY)
    (tmp_path / entry.filename).write_bytes(b"not the published matrix")
    with pytest.raises(fetch.ChecksumMismatch) as caught:
        fetch.verify(entry.key, cache_dir=str(tmp_path))
    message = str(caught.value)
    assert entry.sha256 in message
    assert "Do NOT edit the expected digest" in message


def test_is_cached_is_false_rather_than_raising(tmp_path):
    assert not fetch.is_cached(sources.ONTOLOGY_KEY, cache_dir=str(tmp_path))
    (tmp_path / sources.data_file(sources.ONTOLOGY_KEY).filename).write_bytes(b"x")
    assert not fetch.is_cached(sources.ONTOLOGY_KEY, cache_dir=str(tmp_path))


def test_loaders_do_not_download_by_default(tmp_path):
    """`allow_download` defaults to False everywhere a loader can be entered."""
    with pytest.raises(fetch.MissingCachedFile):
        ontology_module.load_ontology(cache_dir=str(tmp_path))
    with pytest.raises(fetch.MissingCachedFile):
        matrix.load_connectome(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# The ontology
# ---------------------------------------------------------------------------


def test_fixture_ontology_covers_the_fixture_regions_and_all_divisions(ccf):
    # 48 structures when P3.1 cut this fixture (the fifteen regions, their
    # ancestors, and the twelve divisions with theirs); 55 since P3.2 added
    # every structure its sign table names (RT, ZI, SNc, PALv, PALm, CBX,
    # CBN and their ancestors), so that `SignRules`' acronym-vs-id
    # cross-check runs against the committed fixture rather than only
    # against the gitignored full ontology; 61 since P4.2 added
    # `AUTORADIOGRAPHY_ONLY_REGION_IDS` (OT, CLA, EPd, ACA, HIP and EPd's
    # parent EP), which carry no connectome row and exist so the expression
    # fixture can be cut over the autoradiography comparison regions; 69
    # since P4.3 added `ANATOMY_ONLY_REGION_IDS`, the regions its cited
    # expected distributions name: CA1, CA3, DG, MH, PVT plus the three
    # ancestors they pull in (CA, EPI, MTN). SNc and CBX are in that tuple
    # too and cost nothing, because P3.2's sign table had already put them
    # in this ontology. 77 since P6.1 added
    # `PROJECTION_NUCLEUS_ONLY_REGION_IDS`, the five neuromodulatory source
    # nuclei not already present (CS, NDB, PPN, LDT, TM) plus the three
    # ancestors they pull in, so that `projection_systems.NUCLEI`'s CCF
    # acronyms and ids can be checked against a COMMITTED ontology rather
    # than only against the gitignored full one.
    assert len(ccf) == 77
    for structure_id in fixtures.EXTRA_ONTOLOGY_REGION_IDS:
        assert structure_id in ccf
        # An ontology-only entry, deliberately: it must NOT have acquired a
        # connectome row, because the committed matrices are P3.1's numbers.
        assert structure_id not in fixtures.FIXTURE_REGION_IDS
    for structure_id in fixtures.FIXTURE_REGION_IDS:
        assert structure_id in ccf
    divisions = ccf.major_divisions()
    assert len(divisions) == len(ontology_module.MAJOR_DIVISION_IDS)
    assert [d.acronym for d in divisions] == [
        "Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL",
        "TH", "HY", "MB", "P", "MY", "CB",
    ]


def test_structures_carry_id_acronym_and_full_name(ccf):
    vta = ccf[749]
    assert vta.acronym == "VTA"
    assert vta.name == "Ventral tegmental area"
    assert ccf.by_acronym("VTA").id == 749
    assert ccf.id_for_acronym("LC") == 147
    assert ccf.name_for_id(147) == "Locus ceruleus"


def test_hierarchy_walks_up_to_root(ccf):
    assert ccf.path_acronyms(749) == ["root", "grey", "BS", "MB", "MBmot", "VTA"]
    assert ccf.path_acronyms(147) == ["root", "grey", "BS", "HB", "P", "P-sat", "LC"]
    assert ccf.ancestors(749)[0] == ccf.by_acronym("MBmot").id
    assert ccf[ccf.ancestors(749)[-1]].acronym == "root"


def test_major_division_is_the_p3_2_aggregation_primitive(ccf):
    """Every fixture region maps to exactly one of the twelve divisions."""
    expected = {
        985: "Isocortex", 972: "Isocortex",
        56: "STR", 672: "STR", 536: "STR",
        295: "CTXsp",
        1022: "PAL", 1031: "PAL",
        470: "HY", 362: "TH",
        749: "MB", 381: "MB", 795: "MB", 872: "MB",
        147: "P",
    }
    for structure_id, acronym in expected.items():
        division = ccf.major_division(structure_id)
        assert division is not None, structure_id
        assert ccf[division].acronym == acronym


def test_structures_above_the_divisions_have_no_division(ccf):
    assert ccf.major_division(ccf.id_for_acronym("root")) is None
    assert ccf.major_division(ccf.id_for_acronym("grey")) is None
    # A division is its own division: aggregation has to be idempotent, or
    # aggregating twice would drop the coarse regions.
    assert ccf.major_division(313) == 313


def test_descendant_relations(ccf):
    assert ccf.is_descendant_of(749, 313)
    assert not ccf.is_descendant_of(313, 749)
    assert ccf.is_descendant_of(749, 749)
    assert 749 in ccf.descendants(313)
    assert 313 not in ccf.descendants(313)
    assert 313 in ccf.descendants(313, include_self=True)


def test_unknown_lookups_raise_with_the_key_in_the_message(ccf):
    with pytest.raises(KeyError) as caught:
        ccf[123456789]
    assert "123456789" in str(caught.value)
    with pytest.raises(KeyError) as caught:
        ccf.by_acronym("NOT_A_REGION")
    assert "NOT_A_REGION" in str(caught.value)


def test_duplicate_acronyms_are_refused_not_silently_collapsed():
    """The Oh 2014 loader joins on acronym, so ambiguity must be fatal."""
    payload = {
        "id": 1, "acronym": "root", "name": "root", "parent_structure_id": None,
        "children": [
            {"id": 2, "acronym": "X", "name": "first", "parent_structure_id": 1,
             "children": []},
            {"id": 3, "acronym": "X", "name": "second", "parent_structure_id": 1,
             "children": []},
        ],
    }
    with pytest.raises(ValueError) as caught:
        ontology_module.StructureOntology.from_json(payload)
    assert "duplicate acronym" in str(caught.value)


def test_ontology_parses_the_api_envelope_and_a_bare_root(tmp_path, ccf):
    with open(fixtures.ONTOLOGY_FIXTURE_PATH) as handle:
        payload = json.load(handle)
    assert "msg" in payload  # fixture keeps the published envelope
    bare = ontology_module.StructureOntology.from_json(payload["msg"][0])
    assert len(bare) == len(ccf)


# ---------------------------------------------------------------------------
# The matrix: shape, identity, provenance
# ---------------------------------------------------------------------------


def test_knox_fixture_shape_and_region_identity(knox, ccf):
    assert (knox.n_source, knox.n_target) == (15, 30)
    assert list(knox.source_ids) == list(fixtures.FIXTURE_REGION_IDS)
    assert knox.source_acronyms()[:4] == ["MOp", "PL", "ACB", "CP"]
    ipsi = knox.hemisphere("ipsi")
    contra = knox.hemisphere("contra")
    assert list(ipsi.target_ids) == list(fixtures.FIXTURE_REGION_IDS)
    assert list(contra.target_ids) == list(fixtures.FIXTURE_REGION_IDS)
    assert knox.target_acronyms()[15] == "MOp (contra)"


def test_loaded_matrix_carries_its_citation_and_units(knox):
    assert "Knox" in knox.citation and "2019" in knox.citation
    assert "netn_a_00066" in knox.citation
    assert "normalized connection density" in knox.units
    assert knox.dataset_key == "knox2019_normalized_connection_density"


def test_weights_are_finite_and_non_negative(knox, oh):
    for loaded in (knox, oh):
        assert np.all(np.isfinite(loaded.weights))
        assert np.all(loaded.weights >= 0.0)


def test_region_metadata_is_the_ontology_half_of_the_loader(knox):
    rows = knox.region_metadata()
    assert len(rows) == knox.n_source
    first = rows[0]
    assert first == {
        "id": 985,
        "acronym": "MOp",
        "name": "Primary motor area",
        "major_division_id": 315,
        "major_division_acronym": "Isocortex",
    }
    assert {row["major_division_acronym"] for row in rows} == {
        "Isocortex", "STR", "CTXsp", "PAL", "HY", "TH", "MB", "P"
    }


def test_matrix_without_an_ontology_still_loads_but_refuses_to_name_regions():
    loaded = matrix.read_connectome(
        fixtures.FIXTURE_CONNECTOME_KEY, fixtures.KNOX_FIXTURE_PATH
    )
    assert loaded.n_source == 15
    with pytest.raises(ValueError) as caught:
        loaded.region_metadata()
    assert "without an ontology" in str(caught.value)


# ---------------------------------------------------------------------------
# Orientation: the trap this whole module is arranged to avoid
# ---------------------------------------------------------------------------


def test_published_orientation_is_source_major(knox, ccf):
    """`weights[i, j]` is source i -> target j, as the files are published.

    Checked on a projection whose direction is not in doubt: the dopaminergic
    VTA -> nucleus accumbens projection and the accumbens -> VTA return
    projection have different weights, so a transposed load cannot pass this
    by symmetry.
    """
    source = list(knox.source_ids).index(749)   # VTA
    target = list(knox.hemisphere("ipsi").target_ids).index(56)  # ACB
    vta_to_acb = knox.hemisphere("ipsi").weights[source, target]
    acb_to_vta = knox.hemisphere("ipsi").weights[target, source]
    assert vta_to_acb == pytest.approx(0.0001070659046282291, rel=1e-12)
    assert acb_to_vta == pytest.approx(0.0004690753262541187, rel=1e-12)
    assert vta_to_acb != acb_to_vta


def test_projection_matrix_is_the_transpose_and_nothing_else(knox):
    projected, source_ids, target_ids, hemispheres = knox.projection_matrix()
    assert projected.shape == (knox.n_target, knox.n_source)
    assert np.array_equal(projected, knox.weights.T)
    assert list(source_ids) == list(knox.source_ids)
    assert list(target_ids) == list(knox.target_ids)
    assert list(hemispheres) == list(knox.target_hemispheres)


def test_square_ipsilateral_uses_the_circuitnetwork_convention(knox, ccf):
    """`W[i, j]` = from j onto i, matching `CircuitNetwork.connectivity`."""
    ids, W = knox.square_ipsilateral()
    assert list(ids) == list(fixtures.FIXTURE_REGION_IDS)
    assert W.shape == (15, 15)
    acronyms = [ccf[i].acronym for i in ids]
    onto_acb = W[acronyms.index("ACB"), acronyms.index("VTA")]
    onto_vta = W[acronyms.index("VTA"), acronyms.index("ACB")]
    assert onto_acb == pytest.approx(0.0001070659046282291, rel=1e-12)
    assert onto_vta == pytest.approx(0.0004690753262541187, rel=1e-12)
    # and it really is the transpose of the published block
    ipsi = knox.hemisphere("ipsi")
    assert np.array_equal(W, ipsi.weights.T)


def test_square_block_feeds_circuitnetwork_directly(knox, ccf):
    """The point of P3.1: the ingested matrix is a CircuitNetwork weight matrix.

    Not a claim that this is a good model - signs, gains and time constants
    are P3.2's problem, and a raw connection density is not a synaptic
    weight. It is a check that the shape and the index convention line up,
    which is the part a later task should not have to rediscover.
    """
    from neurocascade import CircuitNetwork, CircuitNode

    ids, W = knox.square_ipsilateral()
    nodes = [
        CircuitNode(name=ccf[i].acronym, tau=10.0, baseline_input=0.0)
        for i in ids
    ]
    network = CircuitNetwork(nodes, W)
    assert network.n == 15
    assert network.index["VTA"] == list(ids).index(749)
    rates = np.full(15, 0.1)
    derivative = network.rhs(rates, network.baseline_drive())
    assert derivative.shape == (15,)
    assert np.all(np.isfinite(derivative))


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def test_subset_preserves_requested_order_and_both_hemispheres(knox):
    small = knox.subset([749, 56, 972])
    assert list(small.source_ids) == [749, 56, 972]
    assert list(small.target_ids) == [749, 56, 972, 749, 56, 972]
    assert list(small.target_hemispheres) == ["ipsi"] * 3 + ["contra"] * 3
    full_ipsi = knox.hemisphere("ipsi")
    row = list(knox.source_ids).index(749)
    column = list(full_ipsi.target_ids).index(56)
    assert small.weights[0, 1] == full_ipsi.weights[row, column]


def test_subset_refuses_a_region_the_dataset_does_not_have(oh):
    """Oh et al. 2014 has no locus coeruleus; asking must fail loudly."""
    assert fixtures.FIXTURE_REGION_MISSING_FROM_OH2014 == 147
    with pytest.raises(KeyError) as caught:
        oh.subset([749, 147])
    assert "147" in str(caught.value)


def test_hemisphere_rejects_an_unknown_label(knox):
    with pytest.raises(ValueError):
        knox.hemisphere("both")


def test_without_self_connections_clears_only_the_ipsilateral_diagonal(knox):
    before = knox
    after = knox.without_self_connections()
    ipsi_before = before.hemisphere("ipsi").weights
    ipsi_after = after.hemisphere("ipsi").weights
    assert np.all(np.diag(ipsi_before) > 0.0), "fixture should have real self-weights"
    assert np.all(np.diag(ipsi_after) == 0.0)
    off_diagonal = ~np.eye(before.n_source, dtype=bool)
    assert np.array_equal(ipsi_before[off_diagonal], ipsi_after[off_diagonal])
    assert np.array_equal(
        before.hemisphere("contra").weights, after.hemisphere("contra").weights
    )
    assert np.any(before.weights != after.weights)


# ---------------------------------------------------------------------------
# Oh et al. 2014: the acronym join and the '#NUM!' p-values
# ---------------------------------------------------------------------------


def test_oh_fixture_shape_and_missing_lc(oh, knox):
    assert (oh.n_source, oh.n_target) == (14, 28)
    assert set(knox.source_ids) - set(oh.source_ids) == {147}
    assert list(oh.source_ids) == [
        i for i in fixtures.FIXTURE_REGION_IDS if i != 147
    ]


def test_oh_rows_are_joined_to_structure_ids_via_acronym(oh, ccf):
    assert oh.source_acronyms()[:4] == ["MOp", "PL", "ACB", "CP"]
    assert oh.region_metadata()[0]["name"] == "Primary motor area"


def test_oh_without_an_ontology_is_refused(ccf):
    with pytest.raises(ValueError) as caught:
        matrix.read_connectome(fixtures.FIXTURE_OH_KEY, fixtures.OH_FIXTURE_PATH)
    assert "ontology is required" in str(caught.value)


def test_excel_num_error_becomes_nan_exactly_where_the_weight_is_zero(oh):
    """`#NUM!` means 'no connection, so no p-value', and NaN is how that is
    spelled in memory. Zero would read as maximal confidence, which is the
    opposite of what the token means.
    """
    missing = np.isnan(oh.pvalues)
    assert missing.sum() == 184
    assert np.all(oh.weights[missing] == 0.0)
    assert np.all(oh.weights[~missing] > 0.0)
    assert np.all(oh.pvalues[~missing] >= 0.0)


def test_oh_pvalue_grid_matches_the_weight_grid(oh):
    assert oh.pvalues.shape == oh.weights.shape
    sliced = oh.subset([749, 56])
    assert sliced.pvalues.shape == sliced.weights.shape


# ---------------------------------------------------------------------------
# Parser rejection: a misparse must not look like data
# ---------------------------------------------------------------------------


def test_csv_reader_rejects_a_header_without_the_empty_corner(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("source,ipsi\nid,749\n749,0.1\n")
    with pytest.raises(ValueError) as caught:
        matrix.read_knox2019_csv(str(path), "k", "c", "u")
    assert "empty top-left" in str(caught.value)


def test_csv_reader_rejects_a_ragged_data_row(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text(",ipsi,ipsi\n,749,56\n749,0.1\n")
    with pytest.raises(ValueError) as caught:
        matrix.read_knox2019_csv(str(path), "k", "c", "u")
    assert "header has" in str(caught.value)


def test_xlsx_reader_rejects_sheets_whose_labels_disagree(tmp_path, ccf):
    """Misaligned p-value sheets would attach confidence to the wrong edges.

    Nothing else about the file would look wrong, so the reader checks all
    four sheets carry identical row and column labels instead of trusting
    the first one.
    """
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    for name in ("W_ipsi", "PValue_ipsi", "W_contra", "PValue_contra"):
        sheet = workbook.create_sheet(name)
        # PValue_contra names the regions in the other order.
        acronyms = ["VTA", "ACB"] if name != "PValue_contra" else ["ACB", "VTA"]
        sheet.append([None] + acronyms)
        for row, acronym in enumerate(acronyms):
            sheet.append([acronym, 0.1, 0.2])
    path = tmp_path / "mismatched.xlsx"
    workbook.save(str(path))

    with pytest.raises(ValueError) as caught:
        matrix.read_oh2014_xlsx(str(path), "k", "c", "u", ccf)
    assert "PValue_contra" in str(caught.value)


def test_xlsx_reader_rejects_a_blank_cell(tmp_path, ccf):
    """A blank p-value must not become a zero p-value."""
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    for name in ("W_ipsi", "PValue_ipsi", "W_contra", "PValue_contra"):
        sheet = workbook.create_sheet(name)
        sheet.append([None, "VTA", "ACB"])
        sheet.append(["VTA", 0.1, None if name == "PValue_ipsi" else 0.2])
        sheet.append(["ACB", 0.3, 0.4])
    path = tmp_path / "blank.xlsx"
    workbook.save(str(path))

    with pytest.raises(ValueError) as caught:
        matrix.read_oh2014_xlsx(str(path), "k", "c", "u", ccf)
    assert "blank cell" in str(caught.value)


def test_negative_weights_are_treated_as_a_parse_failure():
    with pytest.raises(ValueError) as caught:
        matrix.RegionalConnectome(
            weights=[[0.1, -0.2]],
            source_ids=[749],
            target_ids=[749, 56],
            target_hemispheres=["ipsi", "ipsi"],
            dataset_key="synthetic",
            citation="c",
            units="u",
        )
    assert "non-negative by construction" in str(caught.value)


def test_non_finite_weights_are_treated_as_a_parse_failure():
    with pytest.raises(ValueError) as caught:
        matrix.RegionalConnectome(
            weights=[[0.1, np.nan]],
            source_ids=[749],
            target_ids=[749, 56],
            target_hemispheres=["ipsi", "ipsi"],
            dataset_key="synthetic",
            citation="c",
            units="u",
        )
    assert "non-finite" in str(caught.value)


def test_label_arrays_must_match_the_matrix():
    with pytest.raises(ValueError):
        matrix.RegionalConnectome(
            weights=[[0.1, 0.2]],
            source_ids=[749, 56],
            target_ids=[749, 56],
            target_hemispheres=["ipsi", "ipsi"],
            dataset_key="synthetic", citation="c", units="u",
        )
    with pytest.raises(ValueError):
        matrix.RegionalConnectome(
            weights=[[0.1, 0.2]],
            source_ids=[749],
            target_ids=[749],
            target_hemispheres=["ipsi"],
            dataset_key="synthetic", citation="c", units="u",
        )
    with pytest.raises(ValueError):
        matrix.RegionalConnectome(
            weights=[[0.1, 0.2]],
            source_ids=[749],
            target_ids=[749, 749],
            target_hemispheres=["ipsi", "ipsi"],
            dataset_key="synthetic", citation="c", units="u",
        )


def test_unknown_hemisphere_label_is_refused():
    with pytest.raises(ValueError) as caught:
        matrix.RegionalConnectome(
            weights=[[0.1]],
            source_ids=[749],
            target_ids=[749],
            target_hemispheres=["left"],
            dataset_key="synthetic", citation="c", units="u",
        )
    assert "hemisphere" in str(caught.value)


# ---------------------------------------------------------------------------
# The real files, when they happen to be cached
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not fetch.is_cached(sources.DEFAULT_CONNECTIVITY_KEY)
    or not fetch.is_cached(sources.ONTOLOGY_KEY),
    reason=(
        "data/connectome/ is gitignored and not populated; run "
        "scripts/fetch_connectome.py. Every acceptance criterion is covered "
        "by the offline fixture tests above."
    ),
)
def test_full_cached_matrix_has_the_published_shape():
    ccf = connectome.load_ontology()
    full = connectome.load_connectome(ontology=ccf)
    # Knox et al. 2019: 291 grey-matter regions; 291 ipsilateral targets and
    # 286 contralateral, the five missing ones being midline structures.
    assert full.n_source == 291
    assert full.n_target == 577
    assert full.hemisphere("ipsi").n_target == 291
    assert full.hemisphere("contra").n_target == 286
    ipsi_only = set(full.hemisphere("ipsi").target_ids) - set(
        full.hemisphere("contra").target_ids
    )
    assert {ccf[i].acronym for i in ipsi_only} == {"RM", "RO", "RPA", "OV", "EW"}
    sliced = full.subset(list(fixtures.FIXTURE_REGION_IDS))
    assert np.array_equal(sliced.weights, connectome.load_fixture_connectome().weights)
