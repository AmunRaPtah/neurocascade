"""The absolute-scale anchor: Zilles & Palomero-Gallagher autoradiography.

P12.2. Every test runs offline off the byte-identical .npy fixtures in
`src/neurocascade/human/fixtures/`; a few additionally parse the cached
Supplementary Table 2 xlsx and are skipped when `data/human/` is absent.
See `fixtures/PROVENANCE.md`.

The load-bearing properties under test:

* the values ARE absolute (fmol/mg protein) and carry their sample's
  limitations with them, not behind them;
* MOR and D2 get NO anchor from this source, and the refusal says why
  rather than raising bare;
* area 24's absent granular stratum is an absence, not a zero;
* nothing here pretends to join to the Desikan-Killiany connectome.
"""

import os

import numpy as np
import pytest

from neurocascade.human import autoradiography as A
from neurocascade.human import fixtures


#: Reference values computed from the fixture bytes (which are the verified
#: published bytes) and copied here verbatim. If any of them moves, a test
#: fails before a downstream number can.
V1_GRANULAR_GABA_A_BZ = 3842.0
V1_GRANULAR_GABA_A = 2941.0
V1_SUPRAGRANULAR_GABA_A = 3201.0
AREA9_SUPRAGRANULAR_GABA_A_BZ = 2808.0
AREA46_INFRAGRANULAR_AMPA = 585.0
V1_LAYER_MEAN_GABA_A_BZ = 3095.6666666666665
GABA_A_BZ_MEASURED_MEDIAN = 2598.0

#: The one modulatory value that approaches the GABA/glutamate range:
#: alpha-1 at supragranular motor area 6. Kept as a pin rather than
#: smoothed over -- see the anatomy tests for how localized it is.
A1_AREA6_SUPRAGRANULAR_OUTLIER = 2287.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table():
    return fixtures.load_fixture_autoradiography()


@pytest.fixture(scope="module")
def gaba_bz(table):
    return table.anchor("GABA_A_BZ")


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_manifest_entry_is_completely_specified():
    assert len(A.DATA_FILES) == 6
    for entry in A.DATA_FILES:
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


def test_urls_are_pinned_to_a_commit():
    for entry in A.DATA_FILES:
        assert "/receptor_principles/%s/" % A.REPOSITORY_COMMIT in entry.url
    assert A.REPOSITORY_COMMIT == "1a8b8714271d99ffe62f81f93ac42f3e57c952f1"


def test_both_licences_are_recorded_on_every_entry():
    for entry in A.DATA_FILES:
        assert "MIT" in entry.citation, entry.key
        assert "CC BY" in entry.citation, entry.key


def test_citation_names_the_paper_and_the_deposit():
    text = A.ZILLES_CITATION + " " + A.DEPOSIT_CITATION
    for phrase in (
        "Zilles K, Palomero-Gallagher N",
        "Frontiers in Neuroanatomy 2017;11:78",
        "PMID 28970785",
        "10.3389/fnana.2017.00078",
        "AlGoulas/receptor_principles",
    ):
        assert phrase in text, phrase


def test_units_say_absolute_and_state_the_bounds():
    assert "fmol/mg protein" in A.ABSOLUTE_UNITS
    assert "ABSOLUTE" in A.ABSOLUTE_UNITS
    # ...and the bounds travel with the units, not in a README.
    assert "n=3 donors" in A.ABSOLUTE_UNITS or "SAMPLE" in A.ABSOLUTE_UNITS


def test_checksum_lines_are_sha256sum_c_input():
    lines = A.checksum_lines()
    assert len(lines) == len(A.DATA_FILES)
    by_filename = {entry.filename: entry.sha256 for entry in A.DATA_FILES}
    for line in lines:
        digest, filename = line.split("  ", 1)
        assert by_filename[filename] == digest


def test_missing_cache_entry_raises_instead_of_downloading(tmp_path):
    with pytest.raises(A.MissingCachedFile):
        A.ensure(A.DATA_FILES[0].key, cache_dir=str(tmp_path),
                 allow_download=False)


def test_unknown_data_file_key_names_the_alternatives():
    with pytest.raises(KeyError, match="unknown data file"):
        A.data_file("not_a_file")


# ---------------------------------------------------------------------------
# Sample limitations, recorded at the source
# ---------------------------------------------------------------------------


def test_sample_limitations_are_fields_not_footnotes():
    sample = A.SAMPLE
    assert sample.n_donors == 3
    assert "72-77" in sample.ages
    assert "two male" in sample.sex and "one female" in sample.sex
    assert "postmortem" in sample.state
    assert "8-18 h" in sample.postmortem_delay
    assert "cortical" in sample.coverage and "NO subcortical" in sample.coverage
    assert "NO proper archicortex" in sample.coverage


def test_every_table_hands_out_that_sample(table):
    assert table.sample is A.SAMPLE
    assert table.sample.n_donors == 3


# ---------------------------------------------------------------------------
# Receptor registry: who is covered, who stays scale-free
# ---------------------------------------------------------------------------


def test_registry_has_exactly_the_fifteen_measured_receptors(table):
    assert len(A.RECEPTORS) == 15
    assert set(table.receptors) == set(A.RECEPTORS)
    published = [r.published_name for r in A.RECEPTORS.values()]
    assert len(set(published)) == 15


def test_published_names_match_the_deposit_bytes():
    names = np.load(
        fixtures.FIXTURE_AUTORADIOGRAPHY_PATHS["zilles2017_ReceptorNames_npy"],
        allow_pickle=True,
    )
    assert tuple(str(x) for x in names) == tuple(
        r.published_name for r in A.RECEPTORS.values()
    )


def test_gaba_systems_use_the_model_keys_with_published_names_kept():
    assert A.RECEPTORS["GABA_A"].published_name == "GABAa"
    assert A.RECEPTORS["GABA_A_BZ"].published_name == "GABAa/BZ"
    # ...and say what they are FOR in the model.
    assert "flumazenil" in A.RECEPTORS["GABA_A_BZ"].model_system_note
    assert "VT-derived" in A.RECEPTORS["GABA_A_BZ"].model_system_note
    assert "[3H]muscimol" in A.RECEPTORS["GABA_A"].model_system_note


def test_d1_is_present_and_says_why_that_matters_for_d2():
    note = A.RECEPTORS["D1"].model_system_note
    assert "D2 is NOT" in note
    assert "substitut" in note.lower()


def test_mor_and_d2_are_recorded_scale_free():
    assert set(A.SCALE_FREE_RECEPTORS) == {"MOR", "D2"}
    for reason in A.SCALE_FREE_RECEPTORS.values():
        assert len(reason) > 80
    # MOR: the receptor simply is not among the 15.
    assert "none of them is the mu-opioid" in A.SCALE_FREE_RECEPTORS["MOR"]
    # D2: the sharper refusal -- its sister receptor D1 IS present.
    assert "D1" in A.SCALE_FREE_RECEPTORS["D2"]
    assert "forbids" in A.SCALE_FREE_RECEPTORS["D2"]


def test_absolute_anchor_refuses_mor_and_d2_with_the_reason():
    for key in ("MOR", "D2"):
        with pytest.raises(A.ScaleFreeReceptor) as excinfo:
            A.absolute_anchor(key)
        assert excinfo.value.receptor_key == key
        assert excinfo.value.reason == A.SCALE_FREE_RECEPTORS[key]
        assert "scale-free" in str(excinfo.value)


def test_scale_free_reason_lookup_refuses_unknown_keys():
    with pytest.raises(KeyError, match="scale-free"):
        A.scale_free_reason("H1")


def test_coverage_is_a_property_not_an_assertion_elsewhere():
    assert A.receptors_covered() == tuple(A.RECEPTORS)
    assert "MOR" not in A.receptors_covered()
    assert "D2" not in A.receptors_covered()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def test_table_shape_and_order_from_fixtures(table):
    assert len(table.areas) == 44
    assert len(set(table.areas)) == 44
    assert table.strata == ("supragranular", "granular", "infragranular")
    assert "V1" in table.areas and "24" in table.areas


def test_reference_values_are_pinned(table):
    assert float(table.for_area("V1", "GABA_A_BZ", "granular")) == \
        V1_GRANULAR_GABA_A_BZ
    assert float(table.for_area("V1", "GABA_A", "granular")) == \
        V1_GRANULAR_GABA_A
    assert float(table.for_area("V1", "GABA_A", "supragranular")) == \
        V1_SUPRAGRANULAR_GABA_A
    assert float(table.for_area("9", "GABA_A_BZ", "supragranular")) == \
        AREA9_SUPRAGRANULAR_GABA_A_BZ
    assert float(table.for_area("46", "AMPA", "infragranular")) == \
        AREA46_INFRAGRANULAR_AMPA


def test_anchor_returns_only_measured_strata(gaba_bz):
    assert set(gaba_bz) == set(fixtures.load_fixture_autoradiography().areas)
    assert set(gaba_bz["V1"]) == {"supragranular", "granular", "infragranular"}
    assert float(gaba_bz["V1"]["granular"]) == V1_GRANULAR_GABA_A_BZ


def test_unknown_area_receptor_or_stratum_name_the_knowns(table):
    with pytest.raises(KeyError, match="not one of this table's 44 areas"):
        table.for_area("Laccumb", "GABA_A", "granular")
    with pytest.raises(KeyError, match="not one of this table's receptors"):
        table.for_area("V1", "D2", "granular")
    with pytest.raises(KeyError, match="stratum"):
        table.for_area("V1", "GABA_A", "layer-iv")


def test_wrong_shape_arrays_are_rejected():
    with pytest.raises(ValueError, match="44 distinct"):
        A.AutoradiographicDensityTable(
            strata_arrays={s: np.zeros((43, 15)) for s in A.STRATA},
            area_labels=["a"] * 43,
            receptor_keys=A.receptors_covered(),
            citation="x",
        )
    good_areas = ["A%02d" % i for i in range(44)]
    with pytest.raises(ValueError, match="\\(44, 15\\)"):
        A.AutoradiographicDensityTable(
            strata_arrays={
                "supragranular": np.zeros((44, 14)),
                "granular": np.zeros((44, 15)),
                "infragranular": np.zeros((44, 15)),
            },
            area_labels=good_areas,
            receptor_keys=A.receptors_covered(),
            citation="x",
        )
    # Right count, wrong identities: a column set that is not this registry
    # cannot be silently relabelled into one.
    impostors = ["R%d" % i for i in range(15)]
    with pytest.raises(ValueError, match="must match this module's registry"):
        A.AutoradiographicDensityTable(
            strata_arrays={s: np.zeros((44, 15)) for s in A.STRATA},
            area_labels=good_areas,
            receptor_keys=impostors,
            citation="x",
        )


def test_unexpected_zero_is_rejected_not_normalized():
    areas = ["A%02d" % i for i in range(44)]
    arrays = {s: np.full((44, 15), 100.0) for s in A.STRATA}
    arrays["supragranular"][7, 3] = 0.0  # not a registered absence
    with pytest.raises(ValueError, match="zero cells outside"):
        A.AutoradiographicDensityTable(
            strata_arrays=arrays, area_labels=areas,
            receptor_keys=A.receptors_covered(), citation="x",
        )


def test_registered_absence_must_actually_be_zero_in_the_deposit():
    areas = list(fixtures.load_fixture_autoradiography().areas)
    index = areas.index("24")
    arrays = {
        s: np.asarray(np.load(
            fixtures.FIXTURE_AUTORADIOGRAPHY_PATHS[
                "zilles2017_ReceptData_%s_npy" % s[0].upper()
            ],
            allow_pickle=True,
        ), dtype=float)
        for s in A.STRATA
    }
    arrays["granular"][index, :] = 5.0  # absence cell filled in
    with pytest.raises(ValueError, match="registered absence cells are not zero"):
        A.AutoradiographicDensityTable(
            strata_arrays=arrays, area_labels=areas,
            receptor_keys=A.receptors_covered(), citation="x",
        )


def test_the_only_zeros_in_the_deposit_are_area_24_granular():
    counts = {}
    for stratum in A.STRATA:
        array = np.load(
            fixtures.FIXTURE_AUTORADIOGRAPHY_PATHS[
                "zilles2017_ReceptData_%s_npy" % stratum[0].upper()
            ],
            allow_pickle=True,
        )
        rows, cols = np.nonzero(np.asarray(array) == 0)
        if rows.size:
            counts[stratum] = (rows.tolist(), cols.tolist())
    assert set(counts) == {"granular"}
    rows, cols = counts["granular"]
    assert len(rows) == 15  # every receptor, one area
    assert set(rows) == {29}  # row 29 IS area 24 in the deposit's order
    assert set(cols) == set(range(15))


# ---------------------------------------------------------------------------
# Area 24: an absence, not a zero
# ---------------------------------------------------------------------------


def test_area_24_granular_raises_instead_of_returning_zero(table):
    with pytest.raises(A.StratumAbsent) as excinfo:
        table.for_area("24", "AMPA", "granular")
    assert excinfo.value.area == "24"
    assert excinfo.value.stratum == "granular"
    assert "agranular" in excinfo.value.reason
    assert "'--'" in excinfo.value.reason


def test_area_24_still_reports_both_existing_strata(table):
    strata = table.strata_of("24", "GABA_A_BZ")
    assert set(strata) == {"supragranular", "infragranular"}
    assert all(float(v) > 0 for v in strata.values())


def test_stratum_absent_is_a_key_error_subtype():
    assert issubclass(A.StratumAbsent, KeyError)


def test_layer_mean_skips_the_absent_stratum(table):
    mean = table.layer_mean("24", "GABA_A_BZ")
    expected = (
        float(table.for_area("24", "GABA_A_BZ", "supragranular"))
        + float(table.for_area("24", "GABA_A_BZ", "infragranular"))
    ) / 2.0
    assert float(mean) == pytest.approx(expected)


def test_layer_mean_is_the_derived_mean_and_labelled_so(table):
    for area, receptor in (("V1", "GABA_A_BZ"), ("46", "AMPA"),
                           ("38", "m2"), ("11", "NMDA")):
        mean = table.layer_mean(area, receptor)
        strata = table.strata_of(area, receptor)
        expected = sum(float(v) for v in strata.values()) / len(strata)
        assert float(mean) == pytest.approx(expected), (area, receptor)
        assert "layer-mean(unweighted)" in repr(mean)
    # ...and the docstring says it is NOT the publication's thickness-
    # weighted "all layers" column.
    assert "thickness-weighted" in A.AutoradiographicDensityTable.layer_mean.__doc__
    # The concrete divergence, at the cell this document quotes: V1
    # GABA-A/BZ derives to 3095.67, where the publication's all-layers
    # column reads 3525 (verified against the cached spreadsheet).
    v1_bz = table.layer_mean("V1", "GABA_A_BZ")
    assert float(v1_bz) == V1_LAYER_MEAN_GABA_A_BZ
    assert float(v1_bz) != 3525.0


# ---------------------------------------------------------------------------
# The type system
# ---------------------------------------------------------------------------


def test_values_are_typed_never_bare_floats(table):
    value = table.for_area("V1", "GABA_A_BZ", "granular")
    assert isinstance(value, A.AbsoluteDensity)
    assert not isinstance(value, float)
    assert value.units is A.ABSOLUTE_UNITS
    assert "fmol/mg protein" in value.units


def test_ordering_against_numbers_and_each_other(table):
    v1 = table.for_area("V1", "GABA_A_BZ", "granular")
    v2 = table.for_area("45", "GABA_A_BZ", "granular")
    assert v1 > 3000.0
    assert v1 >= V1_GRANULAR_GABA_A_BZ
    assert v1 > v2
    assert v2 < v1
    assert v1 == V1_GRANULAR_GABA_A_BZ
    with pytest.raises(TypeError):
        hash(v1)


def test_float_is_the_documented_exit(table):
    value = table.for_area("V1", "GABA_A_BZ", "granular")
    assert float(value) == V1_GRANULAR_GABA_A_BZ
    assert "%.1f" % value == "%.1f" % V1_GRANULAR_GABA_A_BZ


def test_gain_multiplication_returns_a_plain_float(table):
    product = 3.0 * table.for_area("V1", "GABA_A_BZ", "granular")
    assert type(product) is float
    assert product == pytest.approx(3.0 * V1_GRANULAR_GABA_A_BZ)
    reverse = table.for_area("V1", "GABA_A_BZ", "granular") * 2
    assert type(reverse) is float


def test_product_of_two_densities_is_refused(table):
    a = table.for_area("V1", "GABA_A_BZ", "granular")
    b = table.for_area("45", "GABA_A_BZ", "granular")
    with pytest.raises(TypeError, match="multiply two densities"):
        a * b


def test_same_receptor_arithmetic_stays_typed(table):
    v1 = table.for_area("V1", "GABA_A", "granular")
    m1 = table.for_area("4", "GABA_A", "granular")
    total = v1 + m1
    difference = v1 - m1
    scaled = m1 / 2.0
    assert isinstance(total, A.AbsoluteDensity)
    assert isinstance(difference, A.AbsoluteDensity)
    assert isinstance(scaled, A.AbsoluteDensity)
    assert float(total) == pytest.approx(V1_GRANULAR_GABA_A + float(m1))
    assert float(difference) == pytest.approx(V1_GRANULAR_GABA_A - float(m1))
    assert float(scaled) == pytest.approx(float(m1) / 2.0)


def test_cross_receptor_addition_and_subtraction_are_refused(table):
    gaba = table.for_area("V1", "GABA_A", "granular")
    bz = table.for_area("V1", "GABA_A_BZ", "granular")
    with pytest.raises(TypeError, match="different receptors"):
        gaba + bz
    with pytest.raises(TypeError, match="different receptors"):
        gaba - bz


def test_density_ratio_is_dimensionless_laminar_structure(table):
    ratio = table.for_area("V1", "GABA_A", "granular") / table.for_area(
        "4", "GABA_A", "granular"
    )
    assert type(ratio) is float
    assert ratio == pytest.approx(2.8498062015503876)


def test_negation_is_refused_as_meaningless(table):
    with pytest.raises(TypeError, match="negating"):
        -table.for_area("V1", "GABA_A", "granular")


def test_arithmetic_with_relative_densities_raises(table):
    """The two P12 layers are different measurement families; mixing them
    must fail loudly even though neither carries a shared scale."""
    pet = fixtures.load_fixture_receptor("GABA_A_BZ").for_connectome_label(
        "Laccumb"
    )
    auto = table.for_area("V1", "GABA_A_BZ", "granular")
    with pytest.raises(TypeError):
        pet + auto
    with pytest.raises(TypeError):
        auto + pet
    with pytest.raises(TypeError):
        auto * pet


def test_cross_scale_comparison_raises_both_ways(table):
    """An fmol/mg density and a z-score are not mutually orderable: neither
    type may let a comparison across measurement families silently compare
    their internal floats."""
    pet = fixtures.load_fixture_receptor("GABA_A_BZ").for_connectome_label(
        "Laccumb"
    )
    auto = table.for_area("V1", "GABA_A_BZ", "granular")
    with pytest.raises(TypeError):
        auto < pet
    with pytest.raises(TypeError):
        pet < auto
    with pytest.raises(TypeError):
        auto >= pet


# ---------------------------------------------------------------------------
# What the numbers say (the paper's own qualitative claims, pinned)
# ---------------------------------------------------------------------------


def test_anatomy_v1_leads_the_granular_stratum_for_both_gaba_systems(table):
    for receptor, expected in (("GABA_A", V1_GRANULAR_GABA_A),
                               ("GABA_A_BZ", V1_GRANULAR_GABA_A_BZ)):
        top = table.top(receptor, "granular", n=2)
        assert top[0][0] == "V1", (receptor, top)
        assert float(top[0][1]) == expected


def test_anatomy_supragranular_ge_infragranular_for_both_gaba_systems(table):
    for receptor in ("GABA_A", "GABA_A_BZ"):
        for area in table.areas:
            sg = table.for_area(area, receptor, "supragranular")
            ig = table.for_area(area, receptor, "infragranular")
            assert sg >= ig, (receptor, area)


def test_anatomy_bz_median_beats_every_modulatory_maximum(table):
    """The paper's claim that GABA/glutamate densities reach much higher
    than the modulatory ones, pinned numerically: the MEDIAN benzodiazepine
    site density over all measured cells exceeds each modulatory receptor's
    single highest cell anywhere in the table."""
    bz_cells = [
        float(v)
        for area in table.areas
        for v in table.strata_of(area, "GABA_A_BZ").values()
    ]
    median = float(np.median(bz_cells))
    assert median == GABA_A_BZ_MEASURED_MEDIAN

    def max_over_areas(receptor):
        return max(
            float(v)
            for area in table.areas
            for v in table.strata_of(area, receptor).values()
        )

    modulatory = ("m1", "m2", "m3", "a4b2", "a1", "a2",
                  "5-HT1a", "5-HT2", "D1")
    for receptor in modulatory:
        assert median > max_over_areas(receptor), receptor
    # The closest any modulatory cell gets is alpha-1 in motor cortex --
    # and even that stays ~300 fmol/mg protein below the median.
    assert max(max_over_areas(r) for r in modulatory) == \
        A1_AREA6_SUPRAGRANULAR_OUTLIER
    assert median - A1_AREA6_SUPRAGRANULAR_OUTLIER == pytest.approx(311.0)


def test_the_alpha1_outlier_is_at_supragranular_motor_area_6(table):
    assert float(table.for_area("6", "a1", "supragranular")) == \
        A1_AREA6_SUPRAGRANULAR_OUTLIER
    # It is a whole-area-6 feature, not one stratum: all three strata of
    # area 6 lead the receptor, and the best non-motor cell is far below.
    assert float(table.for_area("6", "a1", "granular")) == 1750.0
    assert float(table.for_area("6", "a1", "infragranular")) == 1563.0
    # Its neighbours are ordinary, so the feature is localized.
    assert float(table.for_area("4", "a1", "supragranular")) == 263.0
    assert float(table.for_area("8", "a1", "supragranular")) == 361.0


def test_densities_span_two_orders_of_magnitude_across_receptors(table):
    """Why the PET atlas could not be scaled by eye: D1 tops out at 135
    fmol/mg protein while GABA-A/BZ reaches 4213 in supragranular V2d."""
    d1_max = max(
        float(v)
        for area in table.areas
        for v in table.strata_of(area, "D1").values()
    )
    assert d1_max == 135.0
    assert float(table.for_area("V2d", "GABA_A_BZ", "supragranular")) == \
        4213.0
    assert float(table.for_area("V2v", "GABA_A_BZ", "supragranular")) == \
        4164.0


# ---------------------------------------------------------------------------
# The subcortical/archicortical sources: reached vs not reached
# ---------------------------------------------------------------------------


def test_subcortical_sources_are_recorded_individually():
    keys = set(A.SUBCORTICAL_SOURCES)
    assert keys == {
        "Kedo2018_amygdala_PMID29188378",
        "Zhao2023_hippocampus_PMID37030412",
        "PalomeroGallagher2009_cingulate_PMID19034899",
    }
    for key, record in A.SUBCORTICAL_SOURCES.items():
        assert "PMID" in record["citation"], key
        assert record["status"], key
        assert len(record["what_was_tried"]) > 60, key
        assert len(record["consequence"]) > 30, key


def test_kedo_amygdala_is_marked_not_reached():
    record = A.SUBCORTICAL_SOURCES["Kedo2018_amygdala_PMID29188378"]
    assert record["status"] == "searched and NOT reached"
    assert "siibra" in record["what_was_tried"]


def test_reached_papers_say_why_numbers_were_still_not_taken():
    for key in ("Zhao2023_hippocampus_PMID37030412",
                "PalomeroGallagher2009_cingulate_PMID19034899"):
        record = A.SUBCORTICAL_SOURCES[key]
        assert "reached" in record["status"]
        assert "machine-readable" in record["status"] or "complete table" \
            in record["status"]
        assert record["consequence"].startswith("no ")


# ---------------------------------------------------------------------------
# The join this module does not build
# ---------------------------------------------------------------------------


def test_connectome_label_lookup_refuses_and_names_what_would_build_it(table):
    with pytest.raises(NotImplementedError) as excinfo:
        table.for_connectome_label("Laccumb")
    message = str(excinfo.value)
    assert "crosswalk" in message
    assert "siibra" in message


# ---------------------------------------------------------------------------
# The cached files, when present (optional, skipped offline)
# ---------------------------------------------------------------------------


def test_cached_files_verify_against_the_manifest():
    paths = [A.cache_path(entry.key) for entry in A.DATA_FILES]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip(
            "data/human/ cache absent; run scripts/fetch_human_autoradiography.py"
        )
    for entry in A.DATA_FILES:
        A.verify(entry.key)


def test_cached_loaders_agree_with_the_fixture_build():
    fixture_table = fixtures.load_fixture_autoradiography()
    try:
        cached = A.load_table(cache_dir=A.DEFAULT_CACHE_DIR)
    except A.MissingCachedFile:
        pytest.skip(
            "data/human/ cache absent; run scripts/fetch_human_autoradiography.py"
        )
    assert cached.areas == fixture_table.areas
    assert cached.receptors == fixture_table.receptors
    for area in cached.areas:
        for receptor in cached.receptors:
            for stratum, value in cached.strata_of(area, receptor).items():
                assert float(value) == float(
                    fixture_table.strata_of(area, receptor)[stratum]
                ), (area, receptor, stratum)


_XLSX_ENTRY_KEY = [
    f.key for f in A.DATA_FILES if f.filename.endswith(".xlsx")
][0]


def test_cached_xlsx_matches_the_arrays_cell_for_cell(table):
    path = A.cache_path(_XLSX_ENTRY_KEY)
    if not os.path.exists(path):
        pytest.skip(
            "data/human/ cache absent; run scripts/fetch_human_autoradiography.py"
        )
    published = A.read_published_supplementary_table(path)

    measured = {}
    for area in table.areas:
        for receptor in table.receptors:
            for stratum, value in table.strata_of(area, receptor).items():
                measured[(receptor, str(area), stratum)] = float(value)

    assert len(measured) == 1965  # 1980 cells minus area 24's 15 absences
    assert set(published) == set(measured)
    mismatched = {
        key: (published[key], measured[key])
        for key in measured
        if published[key] != measured[key]
    }
    assert not mismatched

    # The absences really are '--' in the spreadsheet, not zeros.
    for receptor in table.receptors:
        assert (receptor, "24", "granular") not in published


def test_cached_xlsx_block_titles_are_the_published_unicode_names():
    for key, title in A.XLSX_BLOCK_NAMES.items():
        assert A.RECEPTORS[key].key == key
        assert isinstance(title, str) and title
    # The two GABA systems' blocks carry the published spellings.
    assert A.XLSX_BLOCK_NAMES["GABA_A"] == "GABAᴀ"
    assert A.XLSX_BLOCK_NAMES["GABA_A_BZ"] == "GABAᴀ/BZ"
