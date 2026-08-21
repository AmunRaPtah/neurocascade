"""P4.2: expression energy -> receptor density, and how wrong that is.

Everything up to the `real_data` block runs offline off the committed
fixtures, including the acceptance-criterion comparison (Drd1 against Yoo
et al. 2010's mouse D1 autoradiography) and both measured
soma-versus-terminal displacements. That is possible because P4.2 added the
five autoradiography-comparison regions to the shared fixture ontology
(`connectome.EXTRA_ONTOLOGY_REGION_IDS`), so the fixture expression slice
covers every region of the comparison. The `real_data` tests re-run the
same comparisons on the full 291-region matrix and sweep the seven
receptor systems of Lothmann et al. 2021; they skip without the gitignored
cache and they are extras, not substitutes.
"""

import numpy as np
import pytest

from neurocascade.legacy import connectome, expression
from neurocascade.legacy.expression import (
    autoradiography,
    density,
    fetch,
    fixtures,
    mismatch,
    sources,
    validation,
)


COMPARISON_ACRONYMS = (
    "ACA", "ACB", "CLA", "CP", "EPd", "HIP", "HY", "MO", "OT", "SNc", "SNr",
    "TH", "VTA",
)


@pytest.fixture
def ontology():
    return connectome.load_fixture_ontology()


@pytest.fixture
def structure_ids(ontology):
    return [ontology.id_for_acronym(a) for a in COMPARISON_ACRONYMS]


@pytest.fixture
def energy(ontology, structure_ids):
    atlas = fixtures.load_fixture_atlas(
        symbols=["Drd1", "Drd2", "Slc6a3"], ontology=ontology
    )
    return atlas.energy_matrix(
        structure_ids,
        symbols=["Drd1", "Drd2", "Slc6a3"],
        plane="coronal",
        combine="mean",
    )


# ---------------------------------------------------------------------------
# The conversion, and the assumption it is built on
# ---------------------------------------------------------------------------


def test_the_assumption_is_written_down_as_an_assumption():
    """P4.2's acceptance criterion, as a test: the proportionality claim has
    to be labelled an assumption in the code, not only in a doc."""
    text = density.ASSUMPTION.lower()
    assert "assumed proportional" in text
    assert "assumption, not a measurement" in text
    assert density.PROPORTIONALITY_EXPONENT == 1.0
    # And the module docstring has to carry it too, since that is what a
    # reader of the source sees first.
    assert "ASSUMPTION" in density.__doc__


def test_normalization_has_no_default(energy):
    with pytest.raises(TypeError) as caught:
        density.relative_density(energy, "Drd1")
    message = str(caught.value)
    assert "requires normalization=" in message
    for policy in density.NORMALIZATIONS:
        assert policy in message


def test_normalization_policies_mean_different_things(energy):
    by_max = density.relative_density(energy, "Drd1", normalization="max")
    by_mean = density.relative_density(energy, "Drd1", normalization="mean")
    raw = density.relative_density(energy, "Drd1", normalization="none")

    assert np.nanmax(by_max.values) == pytest.approx(1.0)
    assert np.nanmean(by_mean.values) == pytest.approx(1.0)
    np.testing.assert_allclose(raw.values, energy.for_gene("Drd1"))
    # Same ordering, different scale: normalization must not reorder regions.
    finite = np.isfinite(by_max.values)
    assert np.array_equal(
        np.argsort(by_max.values[finite]), np.argsort(by_mean.values[finite])
    )


def test_reference_region_normalization_is_one_at_the_reference(energy, ontology):
    reference = ontology.id_for_acronym("CP")
    rho = density.relative_density(
        energy, "Drd1", normalization="reference-region", reference=reference
    )
    assert rho.for_region(reference) == pytest.approx(1.0)
    assert rho.reference == reference
    with pytest.raises(TypeError):
        density.relative_density(energy, "Drd1", normalization="reference-region")


def test_units_say_what_the_number_is(energy):
    rho = density.relative_density(energy, "Drd1", normalization="max")
    assert "dimensionless" in rho.units
    raw = density.relative_density(energy, "Drd1", normalization="none")
    assert raw.units == energy.units
    assert "NOT a receptor density" in raw.units


def test_the_map_keeps_its_provenance(energy):
    rho = density.relative_density(energy, "Drd1", normalization="max")
    assert rho.source is energy
    assert rho.source.plane == "coronal"
    assert rho.source.combine == "mean"
    assert rho.source.dataset_ids["Drd1"]
    assert rho.assumption == density.ASSUMPTION


def _gabra1_with_a_gap(ontology):
    """A real fixture matrix with a real hole in it.

    Gabra1's sagittal experiment does not report the dorsal raphe, which is
    the only per-region gap in the fixture that is not a whole missing gene.
    """
    atlas = fixtures.load_fixture_atlas(symbols=["Gabra1"], ontology=ontology)
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    grid = atlas.energy_matrix(ids, plane="sagittal", combine="mean")
    row = grid.for_gene("Gabra1")
    assert np.isnan(row).any(), "fixture no longer exercises the NaN path"
    missing = [i for i, v in zip(ids, row) if not np.isfinite(v)]
    return grid, row, missing


def test_nan_survives_the_conversion(ontology):
    """An unmeasured region is unmeasured after normalizing too. A zero here
    would be the claim that a drug acting at this receptor has no target in
    the region, which is a different statement from 'not measured'."""
    grid, row, missing = _gabra1_with_a_gap(ontology)
    assert [ontology.acronym_for_id(i) for i in missing] == ["DR"]
    rho = density.relative_density(grid, "Gabra1", normalization="max")
    assert np.array_equal(np.isnan(rho.values), np.isnan(row))


def test_all_nan_gene_refuses_rather_than_returning_an_empty_map(ontology):
    atlas = fixtures.load_fixture_atlas(symbols=["Htr2a"], ontology=ontology)
    ids = [int(i) for i in connectome.FIXTURE_REGION_IDS]
    grid = atlas.energy_matrix(ids, plane="coronal", combine="mean")
    with pytest.raises(ValueError) as caught:
        density.relative_density(grid, "Htr2a", normalization="max")
    assert "no measured region" in str(caught.value)


# ---------------------------------------------------------------------------
# What the receptor layer consumes
# ---------------------------------------------------------------------------


def test_as_coupling_multiplies_density_by_a_separate_mechanism_gain(
    energy, ontology
):
    """The factorization P4.2 exists to make explicit: the coupling entry a
    circuit node gets is (mechanism gain) x (relative density), and P4.2
    supplies only the second."""
    rho = density.relative_density(energy, "Drd1", normalization="max")
    nodes = {"striatum": ontology.id_for_acronym("CP"),
             "cortex": ontology.id_for_acronym("MO")}
    coupling = rho.as_coupling(-3.0, nodes)
    assert coupling["striatum"] == pytest.approx(-3.0 * rho.for_region(nodes["striatum"]))
    assert coupling["cortex"] == pytest.approx(-3.0 * rho.for_region(nodes["cortex"]))
    # The striatum has two orders of magnitude more Drd1 transcript than
    # motor cortex, so the same mechanism gain gives a far larger entry.
    assert abs(coupling["striatum"]) > 20 * abs(coupling["cortex"])


def test_as_coupling_accepts_a_per_node_mechanism_gain(energy, ontology):
    rho = density.relative_density(energy, "Drd1", normalization="max")
    nodes = {"a": ontology.id_for_acronym("CP"), "b": ontology.id_for_acronym("OT")}
    coupling = rho.as_coupling({"a": -1.0, "b": 2.0}, nodes)
    assert coupling["a"] < 0.0 < coupling["b"]


def test_as_coupling_refuses_an_unmeasured_region(ontology):
    """Emitting 0.0 would turn 'not measured' into 'no target here'."""
    grid, _, missing = _gabra1_with_a_gap(ontology)
    rho = density.relative_density(grid, "Gabra1", normalization="max")
    with pytest.raises(ValueError) as caught:
        rho.as_coupling(1.0, {"raphe": missing[0]})
    assert "Substituting zero" in str(caught.value)


# ---------------------------------------------------------------------------
# The presynaptic / terminal-field mismatch
# ---------------------------------------------------------------------------


def test_a_terminal_field_gene_will_not_convert_silently(energy):
    """Slc6a3's transcript is in the midbrain and its protein is in the
    striatum. The map is still the best available, so it is not refused
    outright - but using it as a density map has to be something the caller
    typed."""
    with pytest.raises(ValueError) as caught:
        density.relative_density(energy, "Slc6a3", normalization="max")
    message = str(caught.value)
    assert "acknowledge_mismatch=True" in message
    assert "SNc" in message and "CP" in message

    rho = density.relative_density(
        energy, "Slc6a3", normalization="max", acknowledge_mismatch=True
    )
    assert rho.mismatch is mismatch.mismatch_for("Slc6a3")


def test_a_partial_mismatch_flags_but_does_not_block(energy):
    rho = density.relative_density(energy, "Drd1", normalization="max")
    assert rho.mismatch is not None
    assert rho.mismatch.severity == "partial"
    assert "SNr" in rho.mismatch.terminal_regions


def test_an_unaffected_gene_carries_no_flag(energy):
    rho = density.relative_density(energy, "Drd2", normalization="max")
    assert rho.mismatch is mismatch.mismatch_for("Drd2")
    assert rho.mismatch.severity == "partial"
    assert mismatch.mismatch_for("Gabra1") is None
    assert mismatch.mismatch_for("Grin1") is None
    assert mismatch.mismatch_for("Gria1") is None


def test_every_mismatch_record_cites_something():
    assert MISMATCH_SYMBOLS <= set(expression.GENES_BY_SYMBOL)
    for record in mismatch.MISMATCHES:
        assert record.citations, record.symbol
        for citation in record.citations:
            assert len(citation) > 40
        assert record.soma_regions and record.terminal_regions
        assert record.severity in mismatch.SEVERITIES


MISMATCH_SYMBOLS = set(mismatch.MISMATCHES_BY_SYMBOL)


def test_the_dopamine_transporter_displacement_is_two_orders_of_magnitude(energy):
    """The quantified failure case. Slc6a3 transcript says SNr has ~120x the
    striatum; [3H]mazindol binding says the striatum has slightly MORE than
    SNr. The proportionality assumption is not slightly wrong here."""
    displacement = mismatch.measured_displacement(energy, "Slc6a3")
    assert displacement.soma == "SNr" and displacement.terminal == "CP"
    assert displacement.energy_ratio == pytest.approx(118.7, rel=0.02)
    assert displacement.density_ratio == pytest.approx(77.1 / 111.4, rel=1e-6)
    assert displacement.factor == pytest.approx(171.5, rel=0.02)
    assert "misstates" in displacement.describe()


def test_the_d1_striatonigral_displacement_is_a_factor_of_thirty(energy):
    """D1's is smaller and in the other direction: the transcript map has
    the striatum 63x above SNr, the binding has it 2x above, because the
    nigral D1 protein is on striatonigral terminals."""
    displacement = mismatch.measured_displacement(energy, "Drd1")
    assert displacement.soma == "CP" and displacement.terminal == "SNr"
    assert displacement.energy_ratio == pytest.approx(63.3, rel=0.02)
    assert displacement.density_ratio == pytest.approx(409.2 / 206.3, rel=1e-6)
    assert displacement.factor == pytest.approx(31.9, rel=0.02)


def test_a_gene_with_no_measured_pair_returns_none(energy):
    assert mismatch.measured_displacement(energy, "Drd2") is None
    assert mismatch.measured_displacement(energy, "Gabra1") is None


# ---------------------------------------------------------------------------
# The published density tables
# ---------------------------------------------------------------------------


def test_the_autoradiography_tables_are_mouse():
    """Rule 2 of this codebase, as a test: no rat density stands in for a
    mouse one."""
    for dataset in autoradiography.DATASETS.values():
        assert dataset.species == "Mus musculus"
        assert dataset.pmid and dataset.doi
        assert dataset.n_animals


def test_every_published_region_has_a_mapping_decision():
    for getter in (
        autoradiography.yoo2010_d1,
        autoradiography.yoo2010_d2,
        autoradiography.yoo2010_dat,
    ):
        for name in getter().values:
            assert name in autoradiography.REGION_MAPPING, name
    for receptor in autoradiography.lothmann2021_receptors():
        for name in autoradiography.lothmann2021(receptor).values:
            assert name in autoradiography.REGION_MAPPING, name


def test_unmappable_regions_say_why_rather_than_disappearing():
    unmapped = dict(autoradiography.yoo2010_d1().unmapped())
    assert "Substantia nigra" in unmapped
    assert "Nigrostriatal bundle" in unmapped
    assert "Amygdala" in unmapped
    for reason in unmapped.values():
        assert len(reason) > 30


def test_mapped_acronyms_resolve_in_the_shared_ontology(ontology):
    """The join is to CCF acronyms, and CCF is the connectome's ontology.
    A mapping to a name the ontology does not have would be a silent hole."""
    full = connectome.load_ontology()
    for getter in (
        autoradiography.yoo2010_d1,
        autoradiography.yoo2010_d2,
        autoradiography.yoo2010_dat,
    ):
        for acronym in getter().mapped():
            full.id_for_acronym(acronym)
    for receptor in autoradiography.lothmann2021_receptors():
        for acronym in autoradiography.lothmann2021(receptor).mapped():
            full.id_for_acronym(acronym)


def test_the_fixture_ontology_covers_the_offline_comparison(ontology):
    for acronym in COMPARISON_ACRONYMS:
        ontology.id_for_acronym(acronym)
    # P4.3 added a second block of ontology-only regions for its anatomy
    # claims, so this asserts over P4.2's own five rather than over the
    # union: the property being protected is "every region P4.2 added is a
    # region P4.2 compares", and widening it to P4.3's would have made it
    # assert nothing.
    for structure_id in connectome.AUTORADIOGRAPHY_ONLY_REGION_IDS:
        assert ontology.acronym_for_id(structure_id) in COMPARISON_ACRONYMS


def test_ligands_name_the_genes_they_cannot_attribute():
    """[3H]SCH-23390 labels D1 and D5 and this project has no Drd5 map. That
    is an attribution gap and it has to be visible, not rounded away."""
    d1 = autoradiography.yoo2010_d1()
    assert "Drd5" in d1.genes
    assert "Drd5" not in expression.GENES_BY_SYMBOL
    assert "Drd5" in d1.note
    alpha1 = autoradiography.lothmann2021("alpha1")
    assert "Adra1b" in alpha1.note  # the searched-and-empty gene of P4.1
    assert "Adra1b" not in sources.AVAILABLE_SYMBOLS


def test_benzodiazepine_site_excludes_the_insensitive_subunit():
    """P4.1 ingested Gabra4 so P4.2 could exclude it deliberately. This is
    the exclusion."""
    bz = autoradiography.lothmann2021("GABA-A(BZ)")
    assert set(bz.genes) == {"Gabra1", "Gabra2", "Gabra3", "Gabra5"}
    orthosteric = autoradiography.lothmann2021("GABA-A")
    assert "Gabra4" in orthosteric.genes


def test_the_receptor_with_no_usable_density_data_is_recorded_as_such():
    """Oprm1 is the receptor this codebase actually simulates and the one
    with no usable mouse density table. Track E's rule: a search that came
    back empty is a result, written down with what was searched, not a
    silent gap - and not an excuse to substitute the rat data."""
    records = {entry["receptor"]: entry for entry in autoradiography.SEARCHED_AND_EMPTY}
    assert "mu-opioid (Oprm1)" in records
    oprm1 = records["mu-opioid (Oprm1)"]
    assert "9462879" in oprm1["searched"]  # Kitchen et al. 1997, paywalled
    assert "PMC11612844" in oprm1["searched"]  # figures only
    assert "RAT" in oprm1["conclusion"]
    for entry in autoradiography.SEARCHED_AND_EMPTY:
        assert entry["why_wanted"] and entry["searched"] and entry["conclusion"]
    # And no dataset here quietly covers it anyway.
    for getter in (autoradiography.yoo2010_d1, autoradiography.yoo2010_d2,
                   autoradiography.yoo2010_dat):
        assert "Oprm1" not in getter().genes
    for receptor in autoradiography.lothmann2021_receptors():
        assert "Oprm1" not in autoradiography.lothmann2021(receptor).genes


def test_background_spread_is_the_measurement_floor():
    spread = autoradiography.background_spread()
    for receptor in ("D1", "D2", "DAT"):
        median, _, worst = spread[receptor]
        assert 1.0 < median < 1.6
        assert worst < 2.1


# ---------------------------------------------------------------------------
# The comparison itself: the acceptance criterion
# ---------------------------------------------------------------------------


def test_drd1_ranks_regions_well_and_scales_them_badly(energy):
    """The headline P4.2 result, offline on real fixture rows.

    Ten regions, mouse both sides. The ordering is right (Spearman 0.71,
    p 0.02); the proportionality is not (R2 0.31 through the origin, and a
    log-log slope of 0.46 where proportionality would give 1.0). Both
    numbers are reported as they came out; neither tolerance was widened to
    make anything pass.
    """
    comparison = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    assert comparison.n == 10
    rho, p = comparison.spearman
    assert rho == pytest.approx(0.709, abs=0.001)
    assert p < 0.05
    assert comparison.proportional_r2 == pytest.approx(0.310, abs=0.002)
    slope, stderr, _ = comparison.loglog
    assert slope == pytest.approx(0.460, abs=0.002)
    # Proportionality (slope 1) is more than four standard errors away.
    assert (1.0 - slope) / stderr > 4.0


def test_the_transcript_dynamic_range_is_far_wider_than_the_measured_one(energy):
    """The compression the log-log slope describes, stated directly: 340x of
    transcript spread maps onto 20x of measured density."""
    comparison = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    transcript, measured = comparison.dynamic_ranges()
    assert transcript == pytest.approx(340.0, rel=0.02)
    assert measured == pytest.approx(19.8, rel=0.02)
    assert transcript / measured > 15.0


def test_the_worst_regions_are_reported_not_dropped(energy):
    comparison = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    worst = comparison.worst(3)
    assert [acronym for acronym, _, _ in worst] == ["VTA", "TH", "HY"]
    acronym, measured, predicted = worst[0]
    assert measured / predicted > 20.0


def test_joining_the_substantia_nigra_makes_the_fit_worse(energy):
    """The region the default mapping leaves out is the one that disagrees
    most, so the comparison has to be reported both ways or the exclusion
    would be doing work the reader cannot see."""
    with_sn = validation.compare(
        energy,
        "Drd1",
        autoradiography.yoo2010_d1(),
        extra_regions={"SNr": "Substantia nigra"},
    )
    without = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    assert with_sn.n == without.n + 1
    assert with_sn.spearman[0] < without.spearman[0]
    assert with_sn.proportional_r2 < without.proportional_r2
    assert with_sn.worst(1)[0][0] == "SNr"


def test_dropped_regions_are_enumerated_with_reasons(energy):
    dropped = dict(validation.dropped_regions(energy, "Drd1", autoradiography.yoo2010_d1()))
    assert "Substantia nigra" in dropped
    assert "Amygdala" in dropped
    assert all(reason for reason in dropped.values())


def test_the_absolute_scale_carries_its_fit_quality(energy):
    comparison = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    scale = validation.fit_absolute_scale(comparison)
    assert scale.symbol == "Drd1"
    assert scale.factor == pytest.approx(comparison.proportional_slope)
    assert scale.r_squared == pytest.approx(comparison.proportional_r2)
    assert scale.units == "fmol/mg tissue equivalent"
    assert scale.dataset is autoradiography.YOO2010


def test_an_absolute_scale_cannot_be_moved_to_another_gene(energy):
    """Expression energy is not comparable between genes, so k fitted on
    Drd1 says nothing about Drd2."""
    comparison = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    scale = validation.fit_absolute_scale(comparison)
    drd2 = density.relative_density(energy, "Drd2", normalization="none")
    with pytest.raises(ValueError) as caught:
        drd2.to_absolute(scale)
    assert "not comparable between genes" in str(caught.value)

    drd1 = density.relative_density(energy, "Drd1", normalization="none")
    absolute = drd1.to_absolute(scale)
    assert absolute.units == "fmol/mg tissue equivalent"
    assert absolute.absolute_scale is scale


def test_an_absolute_scale_refuses_an_already_normalized_map(energy):
    comparison = validation.compare(energy, "Drd1", autoradiography.yoo2010_d1())
    scale = validation.fit_absolute_scale(comparison)
    normalized = density.relative_density(energy, "Drd1", normalization="max")
    with pytest.raises(ValueError) as caught:
        normalized.to_absolute(scale)
    assert "normalization='none'" in str(caught.value)


def test_comparing_a_gene_against_a_ligand_that_does_not_bind_it_is_refused(energy):
    with pytest.raises(ValueError) as caught:
        validation.compare_lothmann(energy, "Drd2", "D1/5")
    assert "category error" in str(caught.value)


def test_a_comparison_of_two_regions_is_not_a_comparison():
    binding = autoradiography.yoo2010_d1()
    with pytest.raises(ValueError) as caught:
        validation.DensityComparison(
            "Drd1", binding, ["CP", "MO"], [672, 500], [1.0, 2.0], [3.0, 4.0]
        )
    assert "is not a comparison" in str(caught.value)


def test_the_plane_and_combine_choice_barely_moves_the_answer(
    ontology, structure_ids
):
    """P4.1 refused to pick a plane policy and handed the choice to P4.2.
    This is what the choice was worth for this comparison: every one of the
    fourteen policies gives Spearman between 0.70 and 0.80."""
    atlas = fixtures.load_fixture_atlas(symbols=["Drd1"], ontology=ontology)
    rows = validation.policy_sensitivity(
        atlas, "Drd1", autoradiography.yoo2010_d1(), structure_ids, ontology=ontology
    )
    assert len(rows) == 14
    rhos = [rho for _, _, _, rho in rows]
    assert min(rhos) == pytest.approx(0.709, abs=0.001)
    assert max(rhos) == pytest.approx(0.794, abs=0.001)


# ---------------------------------------------------------------------------
# The real 291-region files
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
def test_real_drd1_comparison_matches_the_fixture_one():
    """The fixture rows are the real rows, so the offline headline number
    must be the number the full files give. If these two ever disagree the
    fixture has stopped being a slice of the data."""
    full = connectome.load_ontology()
    atlas = expression.load_expression_atlas(ontology=full)
    binding = autoradiography.yoo2010_d1()
    ids = [full.id_for_acronym(a) for a in sorted(binding.mapped())]
    grid = atlas.energy_matrix(ids, symbols=["Drd1"], plane="coronal", combine="mean")
    comparison = validation.compare(grid, "Drd1", binding)
    assert comparison.n == 10
    assert comparison.spearman[0] == pytest.approx(0.709, abs=0.001)


@real_data
def test_real_sweep_shows_the_conversion_is_receptor_specific():
    """Lothmann et al. 2021: seven receptor systems, fourteen regions, one
    expression policy. The alpha-1 and GABA-A maps track the measured
    density; the AMPA maps do not track it at all. That spread is the P4.2
    result that a single headline correlation would hide."""
    full = connectome.load_ontology()
    atlas = expression.load_expression_atlas(ontology=full)

    def rho(receptor, symbol):
        binding = autoradiography.lothmann2021(receptor)
        ids = [full.id_for_acronym(a) for a in sorted(binding.mapped())]
        grid = atlas.energy_matrix(
            ids, symbols=[symbol], plane="coronal-else-sagittal", combine="mean"
        )
        return validation.compare(grid, symbol, binding).spearman

    assert rho("alpha1", "Adra1a")[0] == pytest.approx(0.714, abs=0.002)
    assert rho("alpha1", "Adra1d")[0] == pytest.approx(0.763, abs=0.002)
    assert rho("GABA-A", "Gabra1")[0] == pytest.approx(0.723, abs=0.002)
    assert rho("GABA-A(BZ)", "Gabra3")[0] == pytest.approx(0.789, abs=0.002)

    # Every AMPA subunit is uncorrelated with measured AMPA binding, and
    # this is reported rather than dropped.
    for subunit in ("Gria1", "Gria2", "Gria3", "Gria4"):
        statistic, pvalue = rho("AMPA", subunit)
        assert abs(statistic) < 0.25
        assert pvalue > 0.4


@real_data
def test_real_alpha2_is_the_worst_case_and_has_a_documented_reason():
    """Adra2a barely correlates with measured alpha-2 binding, and Adra2b
    correlates NEGATIVELY. alpha-2A is on locus coeruleus terminals as well
    as on the cortical cells that transcribe it, which is a `partial`
    mismatch record - so the failure has a documented mechanism rather than
    being an unexplained bad number."""
    full = connectome.load_ontology()
    atlas = expression.load_expression_atlas(ontology=full)
    binding = autoradiography.lothmann2021("alpha2")
    ids = [full.id_for_acronym(a) for a in sorted(binding.mapped())]
    grid = atlas.energy_matrix(
        ids, symbols=["Adra2a", "Adra2b"], plane="coronal-else-sagittal",
        combine="mean",
    )
    assert validation.compare(grid, "Adra2a", binding).spearman[0] < 0.2
    assert validation.compare(grid, "Adra2b", binding).spearman[0] < 0.0
    assert mismatch.mismatch_for("Adra2a").severity == "partial"
    assert "LC" in mismatch.mismatch_for("Adra2a").soma_regions


@real_data
def test_real_every_mismatch_region_is_a_real_ccf_structure():
    """The registry names regions as bare acronyms. A typo there would read
    as a plausible claim about anatomy and never fail anything, so it is
    checked against the full structure graph."""
    full = connectome.load_ontology()
    for record in mismatch.MISMATCHES:
        for acronym in record.soma_regions + record.terminal_regions:
            full.id_for_acronym(acronym)


@real_data
def test_real_dat_map_is_flagged_and_measurably_misplaced():
    full = connectome.load_ontology()
    atlas = expression.load_expression_atlas(ontology=full)
    ids = [full.id_for_acronym(a) for a in ("CP", "SNr")]
    grid = atlas.energy_matrix(
        ids, symbols=["Slc6a3"], plane="coronal", combine="mean"
    )
    displacement = mismatch.measured_displacement(grid, "Slc6a3")
    assert displacement.factor > 100.0
    with pytest.raises(ValueError):
        density.relative_density(grid, "Slc6a3", normalization="max")
