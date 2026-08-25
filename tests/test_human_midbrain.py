"""Midbrain nodes from dedicated human atlases, with density ABSENT.

P12.3. Offline tests run entirely off the committed fixtures; the checks
that parse the cached image files are skipped when `data/human/` is
absent. See `fixtures/PROVENANCE.md`.

The load-bearing properties under test:

* midbrain nodes exist, are defined by CIT168/Trutti citations, and carry
  GEOMETRY ONLY -- no density attribute can even be attached;
* every read path into the PET receptor layer REFUSES a midbrain label
  (`MidbrainDensityAbsent`), including through `as_coupling` and
  `coupling_over_connectome`;
* the refusal names what was searched and why each candidate was unusable,
  instead of a bare error;
* the brainstem parcel's own value stays reachable AS a brainstem-parcel
  value, but `pet_value_from_brainstem_parcel` refuses to dress it up as a
  midbrain measurement;
* the published volumes are pinned verbatim (Pauli 2018 Table 1), the
  registry identities are pinned against the committed CIT168 info.csv
  fixture, and -- when the cache is present -- the deposited images are
  shown to reproduce this module's recorded discrepancy numbers rather
  than being quietly trusted.
"""

import os

import pytest

from neurocascade.human import fixtures
from neurocascade.human import midbrain as M


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_manifest_entry_is_completely_specified():
    assert len(M.DATA_FILES) == 3
    for entry in M.DATA_FILES:
        assert len(entry.sha256) == 64, entry.key
        assert set(entry.sha256) <= set("0123456789abcdef"), entry.key
        assert entry.size_bytes > 0, entry.key
        assert entry.url.startswith("https://"), entry.key
        assert entry.filename and "/" not in entry.filename, entry.key
        assert len(entry.citation) > 40, entry.key
        assert len(entry.units) > 20, entry.key
        assert len(entry.description) > 40, entry.key
        assert entry.retrieved == "2026-08-25"
        assert entry.volatility == "static"


def test_units_say_geometry_not_density_on_every_entry():
    # The one thing that must never drift: nothing in this manifest is a
    # receptor measurement.
    for entry in M.DATA_FILES:
        assert "NOT a receptor density" in entry.units, entry.key


def test_both_atlases_and_licences_are_recorded_on_every_entry():
    for entry in M.DATA_FILES:
        assert "Pauli WM, Nili AN, Tyszka JM" in entry.citation, entry.key
        assert "PMID 29664465" in entry.citation, entry.key
        assert "Trutti AC, Fontanesi L" in entry.citation, entry.key
        assert "PMID 33580320" in entry.citation, entry.key
        assert "CC BY 4.0" in entry.citation, entry.key
        # ...including the licence question the sources do NOT answer.
        assert "licence field is UNSET" in entry.citation, entry.key


def test_citations_name_papers_spaces_and_deposits():
    text = " ".join([
        M.PAULI_CITATION, M.TRUTTI_CITATION, M.ATLAS_SPACES_NOTE,
    ])
    for phrase in (
        "Scientific Data 2018;5:180063",
        "Brain Structure and Function 2021;226(4):1155-1167",
        "MNI152NLin2009cAsym",
        "MNI152 2009b",
        "OSF project r2hvk",
        "OSF project 9pzj3",
    ):
        assert phrase in text, phrase


def test_checksum_lines_are_sha256sum_c_input():
    lines = M.checksum_lines()
    assert len(lines) == len(M.DATA_FILES)
    by_filename = {entry.filename: entry.sha256 for entry in M.DATA_FILES}
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert by_filename[filename] == digest


def test_unknown_data_file_key_names_the_alternatives():
    with pytest.raises(KeyError, match="unknown data file"):
        M.data_file("not_a_file")


def test_missing_cache_entry_raises_instead_of_downloading(tmp_path):
    with pytest.raises(M.MissingCachedFile):
        M.ensure(
            M.DATA_FILES[0].key, cache_dir=str(tmp_path), allow_download=False
        )


def test_is_cached_is_false_rather_than_raising(tmp_path):
    assert not M.is_cached("cit168_pauli2018_atlas_bundle", cache_dir=str(tmp_path))


def test_inner_tar_members_are_digested_too():
    # The outer bundle is verified by the Fetcher; these digests are what
    # makes an extraction trustworthy. All three members must be covered.
    assert len(M.CIT168_TAR_MEMBERS) == 3
    for name, (sha, size) in M.CIT168_TAR_MEMBERS.items():
        assert name.startswith("atl-pauli2018/"), name
        assert len(sha) == 64 and set(sha) <= set("0123456789abcdef")
        assert size > 0
    assert M.CIT168_INFO_MEMBER in M.CIT168_TAR_MEMBERS
    assert M.CIT168_PROBABILISTIC_MEMBER in M.CIT168_TAR_MEMBERS


# ---------------------------------------------------------------------------
# The nucleus registry: Pauli 2018 Table 1, transcribed verbatim
# ---------------------------------------------------------------------------


def test_registry_has_all_sixteen_nuclei_with_roles():
    assert len(M.NUCLEI) == 16
    roles = {n.model_role for n in M.NUCLEI.values()}
    assert roles <= {"scenario-node", "geometry-only"}
    scenario_ids = tuple(sorted(n.id for n in M.SCENARIO_NUCLEI))
    assert scenario_ids == (7, 10, 11)  # SNc, PBP, VTA
    for nucleus in M.NUCLEI.values():
        assert nucleus.label and nucleus.acronym, nucleus.id
        assert nucleus.volume_ul_left > 0 and nucleus.volume_ul_right > 0
        assert -100 < nucleus.laterality_percent < 100
        assert nucleus.model_role_note, nucleus.id


def test_key_volumes_are_pinned_verbatim_from_table_1():
    # Transcribed once from the paper; if any of them moves, a test fails
    # before a downstream number can.
    assert (M.NUCLEI[11].volume_ul_left, M.NUCLEI[11].volume_ul_right) == (33.0, 33.0)
    assert (M.NUCLEI[7].volume_ul_left, M.NUCLEI[7].volume_ul_right) == (132.0, 136.0)
    assert (M.NUCLEI[10].volume_ul_left, M.NUCLEI[10].volume_ul_right) == (99.0, 98.0)
    assert (M.NUCLEI[1].volume_ul_left, M.NUCLEI[1].volume_ul_right) == (5224.0, 5196.0)
    assert M.NUCLEI[11].acronym == "VTA"
    assert M.NUCLEI[7].acronym == "SNc"
    assert M.NUCLEI[10].acronym == "PBP"


def test_vta_row_records_that_the_published_name_differs():
    # Pauli's table calls it "Ventral Tegmentum"; the registry keeps the
    # deposited label and says why the volume is so small next to other
    # atlases' "VTA complex".
    note = M.NUCLEI[11].model_role_note
    assert "Ventral Tegmentum" in note
    assert "PBP" in note


def test_registry_identities_match_the_committed_fixture_csv():
    rows = fixtures.load_fixture_cit168_info()
    assert len(rows) == 16
    for row in rows:
        nucleus = M.NUCLEI[int(row["id"])]
        assert nucleus.label == row["label"], row["id"]
        assert row["hemisphere"] == "both", row["id"]
        assert row["structure"] == "subcortex", row["id"]
    assert {int(row["id"]) for row in rows} == set(M.NUCLEI)


def test_registry_identities_match_the_cached_info_member_when_present():
    path = M.cache_path("cit168_pauli2018_atlas_bundle")
    if not os.path.exists(path):
        pytest.skip(
            "data/human/ cache absent; run scripts/fetch_human_midbrain.py"
        )
    rows = M.cit168_info_rows()
    fixture_rows = fixtures.load_fixture_cit168_info()
    # Same bytes whether read through the verified bundle or the fixture.
    assert rows == fixture_rows
    for row in rows:
        assert M.NUCLEI[int(row["id"])].label == row["label"]


# ---------------------------------------------------------------------------
# The nodes: geometry only, structurally incapable of carrying a density
# ---------------------------------------------------------------------------


def test_six_nodes_exist_with_enigma_style_labels():
    assert M.scenario_node_labels() == (
        "LPBP", "LSNc", "LVTA", "RPBP", "RSNc", "RVTA",
    )
    for label in M.MIDBRAIN_NODE_LABELS:
        node = M.node(label)
        assert node.label == label
        assert node.hemisphere == label[0]
        assert node.nucleus.model_role == "scenario-node"


def test_node_volumes_match_their_nucleus_hemisphere():
    for label, node in M.NODES.items():
        expected = (
            node.nucleus.volume_ul_left
            if node.hemisphere == "L"
            else node.nucleus.volume_ul_right
        )
        assert node.volume_ul == expected, label


def test_node_carries_no_density_attribute_and_none_can_be_added():
    node = M.node("LVTA")
    with pytest.raises(AttributeError):
        node.density = 4.2
    with pytest.raises(AttributeError):
        node.rho = 0.5


def test_float_of_a_node_refuses():
    with pytest.raises(TypeError, match="not a number"):
        float(M.node("RVTA"))


def test_receptor_density_refuses_with_full_provenance():
    node = M.node("LVTA")
    with pytest.raises(M.MidbrainDensityAbsent) as excinfo:
        node.receptor_density("MOR")
    refusal = excinfo.value
    assert refusal.node is node
    assert refusal.receptor_key == "MOR"
    assert "SEARCHED_AND_EMPTY" in str(refusal)
    assert "brainstem" in str(refusal)


def test_node_geometry_travels_with_its_citation_and_units():
    node = M.node("RSNc")
    assert "PMID 29664465" in node.citation
    assert "NOT a receptor density" in node.geometry


# ---------------------------------------------------------------------------
# Enforcement on the receptors side
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mor_map():
    return fixtures.load_fixture_receptor("MOR")


def test_pet_map_for_connectome_label_refuses_midbrain_labels(mor_map):
    for label in ("LVTA", "RVTA", "LSNc", "RSNc", "LPBP", "RPBP"):
        with pytest.raises(M.MidbrainDensityAbsent) as excinfo:
            mor_map.for_connectome_label(label)
        assert excinfo.value.receptor_key == "MOR"
        assert excinfo.value.node is not None
        assert excinfo.value.node.label == label


def test_pet_map_refusal_names_the_region_it_was_asked_about(mor_map):
    with pytest.raises(M.MidbrainDensityAbsent, match="ventral tegmental area"):
        mor_map.for_connectome_label("LVTA")


def test_as_coupling_goes_through_the_same_refusal(mor_map):
    with pytest.raises(M.MidbrainDensityAbsent):
        mor_map.as_coupling(-0.5, {"reward": "LVTA"})


def test_coupling_over_connectome_refuses_a_midbrain_label(mor_map):
    from neurocascade.human import receptors as R

    with pytest.raises(M.MidbrainDensityAbsent):
        R.coupling_over_connectome(
            mor_map, -0.5, labels=["Laccumb", "Rput", "LVTA"]
        )


def test_non_midbrain_reads_are_untouched(mor_map):
    # The enforcement must not widen into a general lock-down: real
    # connectome labels keep working, and the brainstem parcel itself
    # stays readable AS the brainstem parcel.
    value = mor_map.for_connectome_label("Laccumb")
    assert isinstance(float(value), float)
    brainstem = mor_map.for_parcel("brainstem")
    assert isinstance(float(brainstem), float)


SPELLINGS_THAT_MUST_REFUSE = (
    "LVTA", "RVTA", "vta", "VTA", "L_SNC", "rSNC", "PBP", "lpbp",
    "L.VTA!", "VentralTegmentalArea", "substantia_nigra_pars_compacta",
)

LABELS_THAT_ARE_NOT_MIDBRAIN = (
    "Laccumb", "Raccumb", "Lput", "Rput", "Lpal", "Lcaud", "Rthal",
    "Lamyg", "Lhippo", "brainstem", "L_superiorfrontal", "", "L", "R",
)


@pytest.mark.parametrize("label", SPELLINGS_THAT_MUST_REFUSE)
def test_every_plausible_spelling_is_caught(mor_map, label):
    assert M.is_midbrain_request(label)
    assert M.node_for_request(label) is not None
    with pytest.raises(M.MidbrainDensityAbsent):
        mor_map.for_connectome_label(label)


@pytest.mark.parametrize("label", LABELS_THAT_ARE_NOT_MIDBRAIN)
def test_no_false_positives_against_real_labels(label):
    # Exactness elsewhere matters more than cleverness here: anything not
    # naming a midbrain node must fall through untouched.
    assert not M.is_midbrain_request(label)
    assert M.node_for_request(label) is None


def test_bare_alias_resolves_to_a_node_so_the_refusal_can_name_it():
    node = M.node_for_request("vta")
    assert node is not None and node.label in ("LVTA", "RVTA")


# ---------------------------------------------------------------------------
# pet_value_from_brainstem_parcel: the named refusal
# ---------------------------------------------------------------------------


def test_brainstem_parcel_value_is_refused_even_though_one_exists(mor_map):
    node = M.node("LVTA")
    with pytest.raises(M.MidbrainDensityAbsent) as excinfo:
        M.pet_value_from_brainstem_parcel(mor_map, node)
    message = str(excinfo.value)
    assert "This call IS the refusal" in message
    assert "pons and medulla" in message
    assert excinfo.value.receptor_key == "MOR"


def test_brainstem_refusal_accepts_labels_too(mor_map):
    with pytest.raises(M.MidbrainDensityAbsent):
        M.pet_value_from_brainstem_parcel(mor_map, "RVTA")


def test_brainstem_refusal_never_returns_a_number(mor_map):
    # Belt and braces: whatever a future refactor does to the message, the
    # function must keep raising for EVERY node.
    for label in M.MIDBRAIN_NODE_LABELS:
        with pytest.raises(M.MidbrainDensityAbsent):
            M.pet_value_from_brainstem_parcel(mor_map, label)


def test_brainstem_refusal_rejects_non_midbrain_requests():
    with pytest.raises(ValueError, match="does not name one of"):
        M.pet_value_from_brainstem_parcel(object(), "Laccumb")


def test_brainstem_refusal_rejects_the_wrong_object_type():
    with pytest.raises(TypeError, match="PETRelativeDensityMap-like"):
        M.pet_value_from_brainstem_parcel(object(), "LVTA")


# ---------------------------------------------------------------------------
# The searched-and-empty record
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("citation", "status", "what_was_tried", "consequence")


def test_search_record_exists_and_is_complete():
    assert len(M.SEARCHED_AND_EMPTY) >= 9
    for key, record in M.SEARCHED_AND_EMPTY.items():
        for field in _REQUIRED_FIELDS:
            assert field in record, key
            assert len(record[field]) > 30, (key, field)


def test_every_consequence_states_the_negative_result():
    for key, record in M.SEARCHED_AND_EMPTY.items():
        assert record["consequence"].startswith("no "), key


def test_specific_candidates_are_recorded_with_their_reasons():
    records = M.SEARCHED_AND_EMPTY
    # Human mu-opioid postmortem binding exists -- but never measured VTA.
    gabilondo = records["Gabilondo1995_PMID7552322"]
    assert "right receptor" in gabilondo["consequence"]
    assert "[3H]DAGO" in gabilondo["what_was_tried"]
    # Human midbrain autoradiography EXISTS -- for other targets.
    allen = records["Allen1991_PMID1748734_angiotensin_II"]
    assert "WRONG TARGET" in allen["status"]
    # In-vivo carfentanil PET studies parcellate no midbrain nucleus.
    noppari = records["Noppari2026_PMID41108393_carfentanil_PET"]
    assert "the midbrain is absent" in noppari["consequence"]
    # Gene-expression proxies stay out (that assumption is P12.4's to retire).
    ceballos = records["Ceballos2026_PMID41845130_neuropeptide_atlas"]
    assert "PROXY" in ceballos["status"]
    # Rodent work is abundant and unusable.
    rodent = records["rodent_VTA_autoradiography_literature"]
    assert "wrong species" in rodent["status"]
    assert "PMID 20452406" in rodent["citation"]
    # The already-ingested PET map contains the territory, not a measurement.
    hansen = records["hansen2022_MOR_map_brainstem_parcel"]
    assert "NOT a midbrain" in hansen["status"]


def test_every_record_cites_a_findable_identifier():
    for key, record in M.SEARCHED_AND_EMPTY.items():
        assert "PMID" in record["citation"], key


def test_search_method_is_recorded_so_it_can_be_rerun():
    assert "Europe PMC" in M.SEARCH_RECORDS_NOTE
    docstring_block = M.__doc__ or ""
    note = M.SEARCH_RECORDS_NOTE
    assert "queries" in note or "docstring" in note


def test_searched_and_empty_reason_lookup():
    reason = M.searched_and_empty_reason("Gabilondo1995_PMID7552322")
    assert reason.startswith("no ")
    with pytest.raises(KeyError, match="no search record"):
        M.searched_and_empty_reason("not_a_record")


# ---------------------------------------------------------------------------
# Cross-atlas consistency, and the deposit-vs-paper discrepancy
# ---------------------------------------------------------------------------


def test_cross_atlas_consistency_is_numerical_and_honest():
    record = M.CROSS_ATLAS_CONSISTENCY
    assert "137.35" in record["trutti_vta_complex_individual_masks"]
    assert "138.80" in record["trutti_vta_complex_individual_masks"]
    assert "132 ul left" in record["pauli_table1_vta_plus_pbp"]
    assert "131 ul right" in record["pauli_table1_vta_plus_pbp"]
    assert "validation" in record["what_it_is_not"]
    assert "never" in record["what_it_is_not"] or "no number" in record["what_it_is_not"]


def test_trutti_published_volumes_are_pinned_verbatim():
    volumes = M.TRUTTI_PUBLISHED_VOLUMES
    assert volumes["unthresholded_atlas_volume_mm3_left"] == 2226.375
    assert volumes["unthresholded_atlas_volume_mm3_right"] == 2368.0
    assert volumes["individual_mask_mean_mm3_left"] == 137.35
    assert volumes["individual_mask_mean_mm3_right"] == 138.80


def test_deposit_discrepancy_is_quantified_for_all_sixteen_nuclei():
    assert len(M.DEPOSIT_VS_TABLE1) == 16
    for nucleus_id, side in M.DEPOSIT_VS_TABLE1.items():
        nucleus = M.NUCLEI[nucleus_id]
        assert side["label"] == nucleus.label
        assert side["table1_ul"] == (
            nucleus.volume_ul_left, nucleus.volume_ul_right
        )
        left_ratio, right_ratio = side["ratio"]
        # Every deposited nucleus integrates LARGER than its published
        # volume, by between 18% and 66%: the discrepancy is systematic,
        # not noise, and the recorded bounds pin that claim.
        assert 1.17 < left_ratio < 1.67, (nucleus_id, left_ratio)
        assert 1.17 < right_ratio < 1.67, (nucleus_id, right_ratio)


def test_deposit_discrepancy_note_states_the_response():
    note = M.DEPOSIT_DISCREPANCY_NOTE
    assert "18-66%" in note
    assert "hypothes" in note          # causes labelled as hypotheses only
    assert "PEER-REVIEWED" in note     # product values come from Table 1
    assert "DEPOSIT_INTEGRALS_UL" in note


def test_recorded_integrals_match_the_registries_they_summarize():
    assert set(M.DEPOSIT_INTEGRALS_UL) == set(M.NUCLEI)
    for nucleus_id, integrals in M.DEPOSIT_INTEGRALS_UL.items():
        side = M.DEPOSIT_VS_TABLE1[nucleus_id]
        # The comparison view is DERIVED from the two registries; this is
        # what makes drift between them impossible.
        assert side["deposit_integral_ul"] == integrals
        left_ratio, right_ratio = side["ratio"]
        volume = M.NUCLEI[nucleus_id]
        assert abs(left_ratio - integrals[0] / volume.volume_ul_left) < 1e-15
        assert abs(right_ratio - integrals[1] / volume.volume_ul_right) < 1e-15


# ---------------------------------------------------------------------------
# The cached files, when present (optional, skipped offline)
# ---------------------------------------------------------------------------


def _cache_present():
    return all(
        os.path.exists(M.cache_path(entry.key)) for entry in M.DATA_FILES
    )


def test_cached_files_verify_against_the_manifest():
    if not _cache_present():
        pytest.skip("data/human/ cache absent; run scripts/fetch_human_midbrain.py")
    for entry in M.DATA_FILES:
        M.verify(entry.key)


def test_extracted_info_member_equals_the_committed_fixture():
    if not _cache_present():
        pytest.skip("data/human/ cache absent; run scripts/fetch_human_midbrain.py")
    member_path = M.extract_cit168_member(M.CIT168_INFO_MEMBER)
    try:
        with open(member_path, "rb") as cached_handle:
            cached = cached_handle.read()
        with open(fixtures.FIXTURE_CIT168_INFO_PATH, "rb") as fixture_handle:
            committed = fixture_handle.read()
        assert cached == committed
    finally:
        os.remove(member_path)
        os.rmdir(os.path.dirname(member_path))


def test_cached_probabilistic_integrals_reproduce_the_recorded_numbers():
    if not _cache_present():
        pytest.skip("data/human/ cache absent; run scripts/fetch_human_midbrain.py")
    measured = M.cit168_probability_integrals()
    # A voxel is 1 ul, so anything above a millionth of one is a real
    # difference, not floating-point dust; equality within this bound
    # means the same bytes were summed in the same order.
    tolerance = 1e-6
    for nucleus_id, recorded in M.DEPOSIT_INTEGRALS_UL.items():
        left, right = measured[nucleus_id]
        assert abs(left - recorded[0]) <= tolerance, nucleus_id
        assert abs(right - recorded[1]) <= tolerance, nucleus_id


def test_cached_trutti_maps_reproduce_the_paper_volumes_exactly():
    if not _cache_present():
        pytest.skip("data/human/ cache absent; run scripts/fetch_human_midbrain.py")
    measured = M.trutti_mask_volumes()
    published = M.TRUTTI_PUBLISHED_VOLUMES
    assert measured["L"]["nonzero_extent_mm3"] == (
        published["unthresholded_atlas_volume_mm3_left"]
    )
    assert measured["R"]["nonzero_extent_mm3"] == (
        published["unthresholded_atlas_volume_mm3_right"]
    )
    assert measured["L"]["nonzero_voxels"] == 17811
    assert measured["R"]["nonzero_voxels"] == 18944
