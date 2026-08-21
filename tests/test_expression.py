"""P4.1: Allen ISH expression ingestion.

Every test here runs offline off the committed fixtures, except the handful
at the bottom marked `real_data`, which check the full 42-gene pinned
responses and skip when the gitignored cache is absent. Those are extras,
not substitutes: every acceptance criterion is covered by the offline tests.
"""

import hashlib
import json
import os

import numpy as np
import pytest

from neurocascade.legacy import connectome, expression
from neurocascade.legacy.expression import atlas as atlas_module
from neurocascade.legacy.expression import fetch, fixtures, genes, sources, unionize


# ---------------------------------------------------------------------------
# The gene table
# ---------------------------------------------------------------------------


def test_every_gene_the_task_names_is_in_the_table():
    """The roadmap's gene list, expanded to actual mouse symbols.

    'Gabra1-5', 'Gabrb', 'Adra1/2', 'Adrb1/2', 'Chrm1-5' and 'Gria1-4' are
    families, not symbols; this is the expansion, written out so that a
    later edit that quietly drops a subunit fails here.
    """
    expected = {
        "Oprm1", "Oprk1", "Oprd1",
        "Drd1", "Drd2",
        "Gabra1", "Gabra2", "Gabra3", "Gabra4", "Gabra5",
        "Gabrb1", "Gabrb2", "Gabrb3",
        "Htr1a", "Htr2a",
        "Chrm1", "Chrm2", "Chrm3", "Chrm4", "Chrm5",
        "Chrna4", "Chrna7",
        "Adra1a", "Adra1b", "Adra1d",
        "Adra2a", "Adra2b", "Adra2c",
        "Adrb1", "Adrb2",
        "Grin1", "Grin2a", "Grin2b",
        "Gria1", "Gria2", "Gria3", "Gria4",
        "Hrh1", "Cnr1", "Adora2a",
        "Slc6a2", "Slc6a3", "Slc6a4",
    }
    assert set(genes.symbols()) == expected


def test_gene_symbols_and_entrez_ids_are_unique():
    symbols = [g.symbol for g in genes.RECEPTOR_GENES]
    entrez = [g.entrez_id for g in genes.RECEPTOR_GENES]
    assert len(set(symbols)) == len(symbols)
    assert len(set(entrez)) == len(entrez)


def test_gene_lookup_error_lists_the_table():
    with pytest.raises(KeyError) as caught:
        genes.gene("Drd1a")
    assert "Drd1" in str(caught.value)


def test_the_three_modelled_systems_are_marked_in_model():
    """Whatever else moves, the receptors the codebase already simulates
    must not be filed as future work."""
    for symbol in ("Oprm1", "Drd2", "Gabra1"):
        assert genes.gene(symbol).phase == "in model"


def test_gabra4_is_flagged_as_benzodiazepine_insensitive():
    """The one subunit that must not be pooled into a diazepam-site map."""
    assert "INsensitive" in genes.gene("Gabra4").note


# ---------------------------------------------------------------------------
# The manifest, and the searched-and-empty record
# ---------------------------------------------------------------------------


def test_manifest_covers_every_gene_that_has_data():
    """42 of the 43 genes have a manifest entry; the 43rd has a record."""
    manifested = set(sources.AVAILABLE_SYMBOLS)
    missing = set(sources.GENES_WITHOUT_DATA_BY_SYMBOL)
    assert manifested | missing == set(genes.symbols())
    assert not (manifested & missing)
    assert len(manifested) == 42
    assert missing == {"Adra1b"}


def test_adra1b_has_a_searched_and_empty_record_not_a_substitute():
    """Rule 1 and rule 2: no data is recorded as no data, and nothing else
    is put in its place."""
    entry = sources.GENES_WITHOUT_DATA_BY_SYMBOL["Adra1b"]
    assert entry["entrez_id"] == genes.gene("Adra1b").entrez_id
    # The record has to say what was searched, not just that nothing was found.
    assert "total_rows = 0" in entry["searched"]
    assert "no failed filter" in entry["searched"]
    # And it must not have been replaced by another subtype's map.
    assert "Adra1b" not in sources.AVAILABLE_SYMBOLS
    assert "inventing a number" in entry["conclusion"]


def test_asking_the_manifest_for_the_missing_gene_explains_itself():
    """"There is no file for this" and "we looked and Allen has nothing" are
    different facts; only one of them is a typo."""
    with pytest.raises(KeyError) as caught:
        sources.unionize_file("Adra1b")
    assert "no Allen ISH expression data" in str(caught.value)
    assert "total_rows = 0" in str(caught.value)


def test_every_manifest_entry_has_url_digest_citation_and_units():
    for entry in sources.DATA_FILES:
        assert entry.url.startswith("https://api.brain-map.org/")
        assert len(entry.sha256) == 64
        assert int(entry.sha256, 16) >= 0  # hex
        assert entry.size_bytes > 0
        assert "Lein" in entry.citation
        assert entry.units
        assert entry.retrieved == "2026-08-12"
        # Every one of these is a live query, and says so.
        assert entry.volatility == "live-api"


def test_manifest_urls_are_rebuilt_from_the_query_builders():
    """The URL in the manifest is not a hand-copied string: it is what
    `unionize_url()` produces, so the query and the digest cannot drift."""
    for entry in sources.DATA_FILES:
        if isinstance(entry, sources.UnionizeFile):
            assert entry.url == sources.unionize_url(entry.symbol)
        else:
            assert entry.url == sources.index_url()


def test_every_query_pins_the_connectome_structure_graph():
    """The acceptance criterion 'same ontology as the Phase 3 connectome'
    is enforced in the query itself, not downstream."""
    for symbol in sources.AVAILABLE_SYMBOLS:
        assert "structure[graph_id$eq1]" in sources.unionize_criteria(symbol)
    assert sources.STRUCTURE_GRAPH_ID == 1


def test_every_query_pins_a_row_order():
    """Without an explicit order the same rows could come back permuted and
    the digest would fail for no real reason."""
    assert "order$eq" in sources.index_criteria(["Oprm1"])
    assert "order$eq" in sources.unionize_criteria("Oprm1")


def test_unionize_entries_record_row_and_experiment_counts():
    entry = sources.unionize_file("Oprm1")
    assert entry.n_rows == 2411
    assert entry.n_section_data_sets == 3


def test_checksum_lines_are_sha256sum_c_input():
    lines = fetch.checksum_lines()
    assert len(lines) == len(sources.DATA_FILES)
    digest, filename = lines[0].split("  ")
    assert len(digest) == 64
    assert filename == sources.data_file(sources.DATASET_INDEX_KEY).filename


def test_data_file_error_lists_the_manifest():
    with pytest.raises(KeyError) as caught:
        sources.data_file("aba_ish_unionize_nope")
    assert "aba_ish_unionize_oprm1" in str(caught.value)


def test_expression_cache_is_separate_from_the_connectome_cache():
    assert fetch.DEFAULT_CACHE_DIR != connectome.DEFAULT_CACHE_DIR
    assert os.path.basename(fetch.DEFAULT_CACHE_DIR) == "expression"


# ---------------------------------------------------------------------------
# Fetch: verification is strict and offline-safe
# ---------------------------------------------------------------------------


def test_missing_cache_raises_and_names_the_script(tmp_path):
    with pytest.raises(fetch.MissingCachedFile) as caught:
        fetch.ensure(
            sources.DATASET_INDEX_KEY, cache_dir=str(tmp_path), allow_download=False
        )
    assert "fetch_expression.py" in str(caught.value)


def test_wrong_bytes_are_rejected_and_the_message_forbids_editing_the_digest(tmp_path):
    entry = sources.unionize_file("Oprm1")
    path = tmp_path / entry.filename
    path.write_bytes(b"not the pinned response")
    with pytest.raises(fetch.ChecksumMismatch) as caught:
        fetch.verify(entry.key, cache_dir=str(tmp_path))
    message = str(caught.value)
    assert entry.sha256 in message
    assert hashlib.sha256(b"not the pinned response").hexdigest() in message
    assert "Do NOT edit the expected digest" in message


def test_is_cached_never_raises_for_absence(tmp_path):
    assert not fetch.is_cached(sources.DATASET_INDEX_KEY, cache_dir=str(tmp_path))


def test_sha256_of_matches_hashlib(tmp_path):
    payload = b"expression energy is not a receptor density"
    path = tmp_path / "x"
    path.write_bytes(payload)
    assert fetch.sha256_of(str(path)) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# The readers
# ---------------------------------------------------------------------------


def test_index_reads_genes_and_planes():
    index = fixtures.load_fixture_index()
    assert len(index) == 26
    assert set(index.symbols()) == set(fixtures.FIXTURE_SYMBOLS)
    assert index[79912572].symbol == "Oprm1"
    assert index[79912572].plane == "coronal"
    assert index.planes_for("Htr2a") == ["sagittal"]
    assert index.planes_for("Oprm1") == ["coronal", "sagittal"]


def test_index_rejects_an_unsuccessful_response(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"success": False, "msg": "Data Access error"}))
    with pytest.raises(ValueError) as caught:
        unionize.read_dataset_index(str(path))
    assert "not a successful API response" in str(caught.value)


def test_index_rejects_a_truncated_response(tmp_path):
    """A short index silently drops experiments, and the unionize files then
    contain rows that cannot be attributed to a gene."""
    with open(fixtures.INDEX_FIXTURE_PATH) as handle:
        payload = json.load(handle)
    payload["total_rows"] = payload["num_rows"] + 1
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as caught:
        unionize.read_dataset_index(str(path))
    assert "truncated" in str(caught.value)


def test_plane_and_reference_space_are_checked_against_each_other():
    """Coronal ISH is registered to P56 Brain (id 9), sagittal to its L/R
    flipped counterpart (id 10), with no exceptions in the pinned index. An
    experiment registered elsewhere is not commensurable and must not be
    pooled into a gene's map."""
    index = fixtures.load_fixture_index()
    for dataset in index.datasets:
        assert (
            dataset.reference_space_id
            == unionize.REFERENCE_SPACE_BY_PLANE[dataset.plane]
        )
    with pytest.raises(ValueError) as caught:
        unionize.SectionDataSet(
            id=1, symbol="Oprm1", plane="coronal", reference_space_id=10
        )
    assert "not commensurable" in str(caught.value)


def test_unknown_plane_of_section_is_refused():
    with pytest.raises(ValueError) as caught:
        unionize.SectionDataSet(id=1, symbol="Oprm1", plane="horizontal")
    assert "plane of section" in str(caught.value)


def test_unionize_reader_requires_the_published_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,structure_id,expression_energy\n1,2,3\n")
    with pytest.raises(ValueError) as caught:
        unionize.read_unionize_csv(str(path), fixtures.load_fixture_index())
    assert "column header" in str(caught.value)


def test_unionize_reader_refuses_an_unindexed_experiment(tmp_path):
    """The index and the unionize file are pinned together; a mismatch means
    one of them moved and is not something to guess about."""
    with open(fixtures.UNIONIZE_FIXTURE_PATHS["Oprm1"]) as handle:
        lines = handle.read().splitlines()
    lines[1] = lines[1].replace(",69860840,", ",999999999,", 1)
    path = tmp_path / "orphan.csv"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(KeyError) as caught:
        unionize.read_unionize_csv(str(path), fixtures.load_fixture_index())
    assert "999999999" in str(caught.value)


def test_unionize_reader_refuses_rows_from_the_wrong_gene():
    with pytest.raises(ValueError) as caught:
        unionize.read_unionize_csv(
            fixtures.UNIONIZE_FIXTURE_PATHS["Oprm1"],
            fixtures.load_fixture_index(),
            expected_symbol="Drd2",
        )
    assert "belongs to Oprm1" in str(caught.value)


def test_experiment_arrays_are_finite_and_non_negative():
    atlas = fixtures.load_fixture_atlas()
    for expression_of_gene in atlas.genes.values():
        for experiment in expression_of_gene.experiments:
            assert np.all(np.isfinite(experiment.expression_energy))
            assert np.all(experiment.expression_energy >= 0.0)
            assert np.all(experiment.sum_pixels > 0.0)


def test_negative_energy_is_a_parse_failure():
    dataset = unionize.SectionDataSet(id=1, symbol="Oprm1", plane="coronal")
    with pytest.raises(ValueError) as caught:
        unionize.ExperimentUnionizes(
            dataset=dataset,
            structure_ids=[997],
            expression_energy=[-1.0],
            expression_density=[0.1],
            sum_pixels=[10.0],
        )
    assert "non-negative by construction" in str(caught.value)


def test_absent_structure_is_nan_and_never_zero():
    """The fixture's own coverage gap, on real data: Oprm1's sagittal
    experiment 69860840 has no row for DR (id 872)."""
    atlas = fixtures.load_fixture_atlas(symbols=["Oprm1"])
    by_id = {e.dataset.id: e for e in atlas["Oprm1"].experiments}
    dr = 872
    assert np.isnan(by_id[69860840].energy_for([dr])[0])
    assert np.isfinite(by_id[79912572].energy_for([dr])[0])


def test_expression_energy_matches_its_published_definition():
    """expression_energy == sum_expressing_pixel_intensity / sum_pixels, and
    expression_density == sum_expressing_pixels / sum_pixels, checked on the
    fixture rows rather than taken from the documentation on trust.

    The tolerance is the API's six significant figures, not a number chosen
    to make this pass: the raw columns are rounded before they are served.
    """
    path = fixtures.UNIONIZE_FIXTURE_PATHS["Oprm1"]
    with open(path) as handle:
        header = handle.readline().strip().split(",")
        column = {name: i for i, name in enumerate(header)}
        checked = 0
        for line in handle:
            cells = line.strip().split(",")
            sum_pixels = float(cells[column["sum_pixels"]])
            energy = float(cells[column["expression_energy"]])
            density = float(cells[column["expression_density"]])
            recomputed_energy = (
                float(cells[column["sum_expressing_pixel_intensity"]]) / sum_pixels
            )
            recomputed_density = (
                float(cells[column["sum_expressing_pixels"]]) / sum_pixels
            )
            if energy > 0:
                assert abs(recomputed_energy - energy) / energy < 1e-4
            if density > 0:
                assert abs(recomputed_density - density) / density < 1e-4
            checked += 1
    assert checked == fixtures.FIXTURE_COUNTS["Oprm1"][0]


# ---------------------------------------------------------------------------
# The atlas: alignment with the connectome
# ---------------------------------------------------------------------------


def test_atlas_uses_the_connectome_ontology_not_one_of_its_own():
    atlas = fixtures.load_fixture_atlas()
    ontology = connectome.load_fixture_ontology()
    assert atlas.ontology is not None
    assert set(atlas.ontology.structures) == set(ontology.structures)
    assert atlas.ontology.by_acronym("VTA").id == 749


def test_matrix_columns_are_the_connectome_rows_in_order():
    """The acceptance criterion, stated as an assertion: region index means
    the same thing on both sides."""
    ontology = connectome.load_fixture_ontology()
    matrix = connectome.load_fixture_connectome(ontology=ontology)
    atlas = fixtures.load_fixture_atlas(ontology=ontology)
    energy = atlas.for_connectome(
        matrix, plane="coronal-else-sagittal", combine="mean"
    )
    assert list(energy.structure_ids) == [int(i) for i in matrix.source_ids]
    assert energy.acronyms() == matrix.source_acronyms()
    assert energy.shape == (len(fixtures.FIXTURE_SYMBOLS), matrix.n_source)


def test_every_fixture_region_is_covered_by_the_ish_data():
    ontology = connectome.load_fixture_ontology()
    matrix = connectome.load_fixture_connectome(ontology=ontology)
    atlas = fixtures.load_fixture_atlas(ontology=ontology)
    energy = atlas.for_connectome(
        matrix, plane="coronal-else-sagittal", combine="mean"
    )
    assert energy.coverage() == 1.0
    assert energy.regions_without_values() == []
    assert energy.genes_without_values() == []


# ---------------------------------------------------------------------------
# The atlas: the two decisions it refuses to make
# ---------------------------------------------------------------------------


def test_energy_matrix_requires_plane_and_combine():
    atlas = fixtures.load_fixture_atlas()
    ids = [749]
    with pytest.raises(TypeError) as caught:
        atlas.energy_matrix(ids)
    assert "modelling decision" in str(caught.value)
    with pytest.raises(TypeError):
        atlas.energy_matrix(ids, plane="coronal")
    with pytest.raises(TypeError):
        atlas.energy_matrix(ids, combine="mean")


def test_unknown_plane_or_combine_is_refused():
    atlas = fixtures.load_fixture_atlas()
    with pytest.raises(ValueError):
        atlas.energy_matrix([749], plane="axial", combine="mean")
    with pytest.raises(ValueError):
        atlas.energy_matrix([749], plane="coronal", combine="geometric-mean")


def test_experiments_of_one_gene_disagree_substantially():
    """The measured reason `combine` has no default.

    Oprm1's three experiments in the fixture's own regions; the spread is a
    property of the data, so this test states it as a fact rather than
    asserting a specific ratio that a re-fetch could move.
    """
    atlas = fixtures.load_fixture_atlas(symbols=["Oprm1"])
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    stack = np.vstack(
        [e.energy_for(ids) for e in atlas["Oprm1"].experiments]
    )
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(stack, axis=0) / np.nanmin(stack, axis=0)
    assert np.nanmax(spread) > 3.0


def test_plane_policies_select_different_experiments():
    atlas = fixtures.load_fixture_atlas(symbols=["Oprm1"])
    gene_expression = atlas["Oprm1"]
    assert [e.dataset.id for e in gene_expression.select("coronal")] == [79912572]
    assert [e.dataset.id for e in gene_expression.select("sagittal")] == [
        69860840,
        80342234,
    ]
    assert len(gene_expression.select("any")) == 3
    assert [e.dataset.id for e in gene_expression.select("coronal-else-sagittal")] == [
        79912572
    ]


def test_coronal_else_sagittal_never_mixes_planes_for_one_gene():
    atlas = fixtures.load_fixture_atlas()
    for symbol in atlas.symbols():
        chosen = atlas[symbol].select("coronal-else-sagittal")
        assert len(set(e.dataset.plane for e in chosen)) == 1


def test_sagittal_only_gene_is_all_nan_under_coronal_and_is_reported():
    """The failure this makes impossible: reading 'we have no coronal data'
    as 'this gene is not expressed'."""
    atlas = fixtures.load_fixture_atlas()
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    energy = atlas.energy_matrix(ids, plane="coronal", combine="mean")
    assert energy.genes_without_values() == [fixtures.FIXTURE_SAGITTAL_ONLY_SYMBOL]
    row = energy.for_gene("Htr2a")
    assert np.all(np.isnan(row))
    assert not np.any(row == 0.0)
    # ... and the same gene does have data in the other plane.
    sagittal = atlas.energy_matrix(ids, plane="sagittal", combine="mean")
    assert np.all(np.isfinite(sagittal.for_gene("Htr2a")))


def test_a_structure_id_outside_the_ontology_is_refused():
    """An unmeasured region is NaN and reported; a non-existent one is an
    error, because both look like an all-NaN column otherwise."""
    atlas = fixtures.load_fixture_atlas()
    with pytest.raises(KeyError) as caught:
        atlas.energy_matrix([749, 12345678], plane="coronal", combine="mean")
    assert "12345678" in str(caught.value)
    assert "not regions at all" in str(caught.value)


def test_regions_without_values_are_reported():
    atlas = fixtures.load_fixture_atlas()
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    energy = atlas.energy_matrix(
        ids, symbols=["Htr2a"], plane="coronal", combine="mean"
    )
    assert energy.regions_without_values() == ids
    assert energy.coverage() == 0.0


def test_combine_single_refuses_to_average_silently():
    atlas = fixtures.load_fixture_atlas()
    ids = [749]
    with pytest.raises(ValueError) as caught:
        atlas.energy_matrix(ids, symbols=["Oprm1"], plane="any", combine="single")
    assert "exactly one experiment" in str(caught.value)
    # One experiment: allowed, and equal to that experiment's own numbers.
    single = atlas.energy_matrix(
        ids, symbols=["Oprm1"], plane="coronal", combine="single"
    )
    coronal = atlas["Oprm1"].select("coronal")[0]
    assert single.values[0, 0] == pytest.approx(coronal.energy_for(ids)[0])


def test_combine_rules_agree_with_numpy_on_real_rows():
    atlas = fixtures.load_fixture_atlas(symbols=["Drd1"])
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    stack = np.vstack([e.energy_for(ids) for e in atlas["Drd1"].experiments])
    for rule, reducer in (
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("max", np.nanmax),
    ):
        got = atlas.energy_matrix(ids, plane="any", combine=rule).for_gene("Drd1")
        np.testing.assert_allclose(got, reducer(stack, axis=0))


def test_n_experiments_records_who_actually_voted():
    """Drd1's experiment 354 reports only 33 of the fixture's 61 structures,
    so the count is not uniform across a row, and a caller can see it."""
    atlas = fixtures.load_fixture_atlas(symbols=["Drd1"])
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    energy = atlas.energy_matrix(ids, plane="any", combine="mean")
    counts = energy.n_experiments[0]
    assert counts.max() == 4
    assert counts.min() < 4
    # Every finite value has at least one contributor and vice versa.
    assert np.array_equal(np.isfinite(energy.values), energy.n_experiments > 0)


def test_missing_experiments_do_not_drag_a_mean_toward_zero():
    """An experiment that does not report a region does not vote. If NaN
    were treated as 0 the mean would be smaller; assert it is not."""
    atlas = fixtures.load_fixture_atlas(symbols=["Drd1"])
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    energy = atlas.energy_matrix(ids, plane="any", combine="mean")
    stack = np.vstack([e.energy_for(ids) for e in atlas["Drd1"].experiments])
    partial = np.isnan(stack).any(axis=0) & ~np.isnan(stack).all(axis=0)
    assert partial.any()
    zero_filled = np.nan_to_num(stack, nan=0.0).mean(axis=0)
    assert np.all(energy.values[0][partial] > zero_filled[partial])


def test_matrix_records_which_experiments_produced_it():
    atlas = fixtures.load_fixture_atlas(symbols=["Oprm1"])
    energy = atlas.energy_matrix([749], plane="coronal", combine="mean")
    assert energy.dataset_ids == {"Oprm1": [79912572]}
    assert energy.plane == "coronal"
    assert energy.combine == "mean"
    assert "expression energy" in energy.units
    assert "Lein" in energy.citation


# ---------------------------------------------------------------------------
# Reporting: what is missing has to be visible
# ---------------------------------------------------------------------------


def test_atlas_reports_the_gene_with_no_data():
    atlas = fixtures.load_fixture_atlas()
    assert atlas.missing_symbols() == ["Adra1b"]
    report = atlas.unavailable_report()
    assert report[0]["symbol"] == "Adra1b"


def test_asking_for_the_missing_gene_explains_rather_than_keyerrors_blankly():
    atlas = fixtures.load_fixture_atlas()
    with pytest.raises(KeyError) as caught:
        atlas["Adra1b"]
    message = str(caught.value)
    assert "no Allen ISH expression data" in message
    assert "total_rows = 0" in message


def test_indexed_experiments_with_no_unionizes_are_visible():
    """Two of the fixture's 26 indexed experiments produced no expression
    data at all. That is reported, not silently absorbed into a count."""
    atlas = fixtures.load_fixture_atlas()
    orphans = atlas.experiments_without_unionizes()
    assert sorted(d.id for d in orphans) == [282, 359]
    assert fixtures.FIXTURE_EXPERIMENT_WITHOUT_UNIONIZES in [d.id for d in orphans]


def test_coverage_report_is_measured_not_asserted():
    ontology = connectome.load_fixture_ontology()
    matrix = connectome.load_fixture_connectome(ontology=ontology)
    atlas = fixtures.load_fixture_atlas(ontology=ontology)
    rows = atlas.coverage_report(structure_ids=matrix.source_ids)
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["Htr2a"]["planes"] == ["sagittal"]
    assert by_symbol["Oprm1"]["planes"] == ["coronal", "sagittal"]
    assert by_symbol["Drd1"]["n_experiments"] == 4
    for row in rows:
        assert row["n_regions_covered"] == row["n_regions_requested"]
        assert row["regions_missing"] == []


def test_fixture_counts_are_checked_on_load():
    """The truncation guard: a short read hashes consistently and would pass
    every other check."""
    with pytest.raises(ValueError) as caught:
        atlas_module.check_counts(
            "fake.csv", 1, 1, fixtures.load_fixture_atlas(symbols=["Htr2a"])[
                "Htr2a"
            ].experiments
        )
    # Htr2a has one experiment and it reports every structure in the fixture
    # ontology, so its row count IS that ontology's size and moves every time
    # the fixture set grows (55 -> 61 with P4.2, 69 with P4.3, 77 with P6.1).
    # Taken from `FIXTURE_COUNTS`, which is the manifest the loader checks
    # against, so this cannot go stale again and the assertion is no weaker:
    # the point is that `check_counts` reports the count it actually parsed.
    expected_rows = fixtures.FIXTURE_COUNTS["Htr2a"][0]
    assert expected_rows == len(connectome.load_fixture_ontology().structures)
    assert "parsed %d unionize rows, 1 were recorded" % expected_rows in str(
        caught.value
    )


def test_regional_expression_refuses_a_value_with_no_contributor():
    with pytest.raises(ValueError) as caught:
        atlas_module.RegionalExpression(
            values=[[1.0]],
            n_experiments=[[0]],
            symbols=["Oprm1"],
            structure_ids=[749],
            plane="coronal",
            combine="mean",
            dataset_ids={},
            units="x",
            citation="y",
        )
    assert "no contributing experiment" in str(caught.value)


def test_regional_expression_refuses_negative_values():
    with pytest.raises(ValueError) as caught:
        atlas_module.RegionalExpression(
            values=[[-1.0]],
            n_experiments=[[1]],
            symbols=["Oprm1"],
            structure_ids=[749],
            plane="coronal",
            combine="mean",
            dataset_ids={},
            units="x",
            citation="y",
        )
    assert "non-negative" in str(caught.value)


# ---------------------------------------------------------------------------
# Anatomy: a map that fails these is a bug, not a finding
#
# P4.3 owns the full version of this check. These three run on the fixture
# now because they are the cheapest possible evidence that the ingestion is
# not silently transposed, mis-joined or attached to the wrong gene, and
# because a mis-join would otherwise look completely healthy.
# ---------------------------------------------------------------------------


def _fixture_energy(symbol, plane="coronal-else-sagittal"):
    ontology = connectome.load_fixture_ontology()
    atlas = fixtures.load_fixture_atlas(ontology=ontology)
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    energy = atlas.energy_matrix(ids, symbols=[symbol], plane=plane, combine="mean")
    return dict(zip(energy.acronyms(), energy.for_gene(symbol)))


def test_drd2_is_striatum_dominant():
    energy = _fixture_energy("Drd2")
    assert energy["CP"] > 5 * energy["MOp"]
    assert energy["ACB"] > 5 * energy["MOp"]
    assert max(energy, key=energy.get) in ("CP", "ACB")


def test_adora2a_is_striatum_dominant():
    """A2A is the indirect-pathway striatal marker; a map that put it in
    cortex would mean the gene columns were shuffled."""
    energy = _fixture_energy("Adora2a")
    assert max(energy, key=energy.get) in ("CP", "ACB")


def test_slc6a3_is_confined_to_the_dopaminergic_midbrain():
    """DAT is expressed by dopamine neurons, i.e. in their somata (VTA,
    SNc), not in their targets. The striatum is where dopamine is
    *released*; a DAT map peaking in CP would mean the ingestion had
    attached projection data to a gene."""
    energy = _fixture_energy("Slc6a3")
    assert energy["VTA"] > 5 * energy["CP"]
    assert energy["VTA"] > 5 * energy["MOp"]


def test_oprm1_is_thalamus_high_relative_to_cortex():
    """Roadmap P4.3's expectation for Oprm1 is thalamus/striatum-dominant.
    MD is the only thalamic region in the fixture."""
    energy = _fixture_energy("Oprm1")
    assert energy["MD"] > energy["MOp"]
    assert energy["MD"] > energy["PL"]


# ---------------------------------------------------------------------------
# The real files. Extras: they skip when the gitignored cache is absent.
# ---------------------------------------------------------------------------

_REAL_CACHE_PRESENT = fetch.is_cached(sources.DATASET_INDEX_KEY) and all(
    fetch.is_cached(sources.unionize_key(symbol))
    for symbol in sources.AVAILABLE_SYMBOLS
)

real_data = pytest.mark.skipif(
    not _REAL_CACHE_PRESENT,
    reason="data/expression/ is not populated; run scripts/fetch_expression.py",
)


@real_data
def test_real_atlas_loads_every_gene_that_has_data():
    atlas = expression.load_expression_atlas()
    assert sorted(atlas.symbols()) == sorted(sources.AVAILABLE_SYMBOLS)
    assert atlas.missing_symbols() == ["Adra1b"]


@real_data
def test_real_atlas_covers_279_of_the_connectomes_291_regions():
    """The measured alignment between the two Phase 3/4 data layers, and the
    twelve regions where it fails. `MISSING_FROM_ISH` records the twelve; if
    that list is ever quietly edited, this fails."""
    ontology = connectome.load_ontology()
    matrix = connectome.load_connectome(ontology=ontology)
    atlas = expression.load_expression_atlas(ontology=ontology)
    energy = atlas.for_connectome(
        matrix, plane="coronal-else-sagittal", combine="mean"
    )
    missing = [ontology[i].acronym for i in energy.regions_without_values()]
    assert sorted(missing) == sorted(sources.MISSING_FROM_ISH)
    assert matrix.n_source - len(missing) == 279


@real_data
def test_real_sagittal_only_genes_are_the_three_recorded():
    atlas = expression.load_expression_atlas()
    sagittal_only = sorted(
        symbol
        for symbol in atlas.symbols()
        if atlas[symbol].planes() == ["sagittal"]
    )
    assert sagittal_only == sorted(sources.SAGITTAL_ONLY_SYMBOLS)


@real_data
def test_real_orphan_experiments_are_the_four_recorded():
    atlas = expression.load_expression_atlas()
    assert sorted(d.id for d in atlas.experiments_without_unionizes()) == [
        282,
        359,
        526,
        534,
    ]


@real_data
def test_real_maps_reproduce_known_anatomy():
    """P4.3's validation, run early on the real 291-region matrix: the top
    region of each of these maps is not negotiable."""
    ontology = connectome.load_ontology()
    matrix = connectome.load_connectome(ontology=ontology)
    atlas = expression.load_expression_atlas(ontology=ontology)
    energy = atlas.for_connectome(
        matrix, plane="coronal-else-sagittal", combine="mean"
    )
    acronyms = energy.acronyms()

    def top(symbol, n):
        row = energy.for_gene(symbol)
        order = np.argsort(np.where(np.isfinite(row), row, -np.inf))[::-1]
        return [acronyms[i] for i in order[:n]]

    # Striatal markers: caudoputamen and accumbens in the top handful.
    assert "CP" in top("Drd1", 5)
    assert "CP" in top("Drd2", 5)
    assert "CP" in top("Adora2a", 5)
    # Oprm1: the medial habenula is the classic peak, thalamus next.
    assert "MH" in top("Oprm1", 3)
    # DAT and SERT sit in the cells that make the transmitter.
    assert set(top("Slc6a3", 4)) & {"SNc", "VTA"}
    assert "DR" in top("Slc6a4", 3)


@real_data
def test_real_fixtures_are_slices_of_the_real_files():
    """The fixtures are real data, and this is what says so."""
    ontology = connectome.load_fixture_ontology()
    fixture_atlas = fixtures.load_fixture_atlas(ontology=ontology)
    real_atlas = expression.load_expression_atlas(
        symbols=list(fixtures.FIXTURE_SYMBOLS), ontology=ontology
    )
    ids = sorted(ontology.structures)
    for symbol in fixtures.FIXTURE_SYMBOLS:
        pairs = zip(
            fixture_atlas[symbol].experiments, real_atlas[symbol].experiments
        )
        for from_fixture, from_file in pairs:
            assert from_fixture.dataset.id == from_file.dataset.id
            assert np.array_equal(
                from_fixture.energy_for(ids),
                from_file.energy_for(ids),
                equal_nan=True,
            )
