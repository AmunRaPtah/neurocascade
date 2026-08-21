"""P6.3: projection densities derived from the Phase 3 connectome.

Three kinds of assertion, in the order they matter:

* the committed table is REAL DERIVED DATA and can be recomputed. Offline,
  from the committed 15-region connectome fixture, for every pair the
  fixture carries; and against the whole 291-region verified cache in the
  `full_data_only` block at the bottom;
* the derivation is the formula the module claims and nothing else: the
  right normalization, the right aggregation, scale-invariant, and refusing
  a matrix it cannot aggregate;
* the two decisions that are not arithmetic -- the reference region and the
  refused nuclear block -- do what the module says they do, ON THE REAL
  NUMBERS rather than on a toy matrix.
"""

import os

import numpy as np
import pytest

from neurocascade.legacy import connectome
from neurocascade import projection as proj
from neurocascade.legacy import projection_density as pdens
from neurocascade.legacy import projection_scenarios as pscen
from neurocascade.legacy import projection_systems as ps
from neurocascade.legacy.connectome import fixtures, volumes


# ---------------------------------------------------------------------------
# Offline inputs: the committed 15-region fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_inputs():
    ontology = fixtures.load_fixture_ontology()
    strength, voxels = fixtures.load_fixture_build_inputs(ontology=ontology)
    return strength, voxels, ontology


@pytest.fixture(scope="module")
def table():
    return pdens.load_table()


#: The (nucleus, region) pairs the committed connectome FIXTURE can check
#: offline. The fixture's fifteen regions were chosen in P3.1 for the
#: hand-built scenarios, so it has source rows for VTA, DR and LC and target
#: columns for CP and PL, and no rows at all for SNc, CS, NDB, PPN, LDT,
#: TMd or TMv. Those are covered by the `full_data_only` test at the bottom,
#: which re-derives the WHOLE table from the verified cache.
FIXTURE_PAIRS = [
    (nucleus, region)
    for nucleus in ("VTA", "DR", "LC")
    for region in ("CP", "PFC", "VTA", "DR", "LC")
]


# ---------------------------------------------------------------------------
# 1. The committed table is derived data
# ---------------------------------------------------------------------------


def test_the_committed_table_exists_and_covers_every_nucleus_and_region(table):
    assert os.path.exists(pdens.TABLE_PATH)
    assert set(table.nuclei) == set(pdens.NUCLEUS_STRUCTURES)
    assert set(table.regions) == set(pdens.REGION_STRUCTURES)
    for nucleus in table.nuclei:
        for region in table.regions:
            assert table.density(region, nucleus) > 0.0


def test_the_committed_table_matches_a_fresh_derivation_from_the_fixture(
    fixture_inputs, table
):
    # The committed numbers are cut from the 291-region files and the
    # fixture is cut from the same verified bytes, so for a pair whose row
    # and column the fixture carries the two are the SAME number, not an
    # approximation of it: a connection density depends on one row, one
    # column and one voxel count. `connectome_scenarios.scenario_connectome`
    # relies on the identical fact for the circuit layer.
    strength, voxels, ontology = fixture_inputs
    for nucleus, region in FIXTURE_PAIRS:
        derived = pdens.derive_density(
            strength,
            voxels,
            ontology,
            pdens.NUCLEUS_STRUCTURES[nucleus],
            pdens.REGION_STRUCTURES[region],
        )
        assert derived == table.density(region, nucleus), (nucleus, region)


def test_the_derivation_is_summed_strength_over_summed_target_voxels(
    fixture_inputs
):
    # The formula, spelled out against the raw arrays rather than against the
    # helper that computes it, for one pair.
    strength, voxels, ontology = fixture_inputs
    ipsi = strength.hemisphere("ipsi")
    row = list(ipsi.source_ids).index(ontology.id_for_acronym("VTA"))
    column = list(ipsi.target_ids).index(ontology.id_for_acronym("CP"))
    expected = float(ipsi.weights[row, column]) / voxels[
        ("ipsi", ontology.id_for_acronym("CP"))
    ]
    assert pdens.derive_density(
        strength, voxels, ontology, ("VTA",), ("CP",)
    ) == pytest.approx(expected, rel=1e-15)


def test_a_multi_structure_region_sums_strength_and_sums_voxels(fixture_inputs):
    # The aggregation rule `volumes.py` establishes: only the raw summed
    # weight is additive, and the coarse density is the summed strength over
    # the SUMMED target voxels, never the mean of the two densities. TM is
    # the case that forces it (the Knox region set has no TM row, only TMd
    # and TMv), and this checks it on a pair the fixture does carry.
    strength, voxels, ontology = fixture_inputs
    both = pdens.derive_density(strength, voxels, ontology, ("VTA",), ("CP", "PL"))
    ipsi = strength.hemisphere("ipsi")
    row = list(ipsi.source_ids).index(ontology.id_for_acronym("VTA"))
    total = sum(
        float(ipsi.weights[row, list(ipsi.target_ids).index(
            ontology.id_for_acronym(acronym)
        )])
        for acronym in ("CP", "PL")
    )
    volume = sum(
        voxels[("ipsi", ontology.id_for_acronym(acronym))]
        for acronym in ("CP", "PL")
    )
    assert both == pytest.approx(total / volume, rel=1e-15)
    # And it is NOT the mean of the two per-region densities, which is the
    # mistake the rule exists to prevent.
    mean = 0.5 * (
        pdens.derive_density(strength, voxels, ontology, ("VTA",), ("CP",))
        + pdens.derive_density(strength, voxels, ontology, ("VTA",), ("PL",))
    )
    assert both != pytest.approx(mean, rel=1e-3)


def test_an_already_normalized_matrix_is_refused(fixture_inputs):
    _, voxels, ontology = fixture_inputs
    normalized = fixtures.load_fixture_connectome(
        "knox2019_connection_density", ontology
    )
    with pytest.raises(ValueError, match="unnormalized"):
        pdens.derive_density(normalized, voxels, ontology, ("VTA",), ("CP",))


def test_a_stale_committed_table_is_refused(tmp_path):
    # The table records which CCF structures each region name stood for when
    # it was written. If `REGION_STRUCTURES` changes and the table does not,
    # the numbers would silently be about different regions.
    path = str(tmp_path / "stale.csv")
    with open(pdens.TABLE_PATH) as handle:
        text = handle.read()
    with open(path, "w") as handle:
        handle.write(text.replace(",CP,terminal,CP,", ",CP,terminal,ACB,", 1))
    with pytest.raises(ValueError, match="stale"):
        pdens.load_table(path)


# ---------------------------------------------------------------------------
# 2. Only ratios are used
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", pscen.PORT_ORDER)
def test_scaling_the_whole_table_changes_no_concentration(key, table):
    # `ProjectionSystem.release_scale` is derived by requiring the reference
    # region to reproduce its measured basal level, so the density map's
    # overall scale is free. That is what makes it legitimate to compare
    # `connection_density` values across nuclei of very different size
    # without a normalization anyone chose.
    scaled = pdens.DensityTable(
        densities={k: 1000.0 * v for k, v in table.densities.items()},
        dataset_key=table.dataset_key,
        citation=table.citation,
        units=table.units,
        source="scaled",
    )
    base = pscen.port(key)
    other = pscen.port(key, table=scaled)
    for region in base.projection.regions:
        assert base.projection.basal_concentration(region) == pytest.approx(
            other.projection.basal_concentration(region), rel=1e-12
        )


# ---------------------------------------------------------------------------
# 3. The reference region is the region the measurement was made in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", pscen.PORT_ORDER)
def test_every_port_pins_its_measured_basal_level_to_the_measured_region(key):
    scenario = pscen.port(key)
    reference = pdens.REFERENCE_REGION[key]
    assert scenario.projection.reference_region == reference
    assert scenario.projection.basal_concentration(reference) == pytest.approx(
        ps.CLEARANCE_BUILDERS[key]().basal
    )


def test_a_port_that_leaves_out_its_reference_region_is_refused():
    port = pscen.SCENARIO_PORTS["serotonin"]()
    regions = dict(port.regions)
    regions.pop("CP")
    with pytest.raises(ValueError, match="reference region"):
        pscen.ScenarioPort(
            key=port.key,
            transmitter=port.transmitter,
            base_network=port.base_network,
            nucleus_nodes=port.nucleus_nodes,
            regions=regions,
            placements=port.placements,
            readouts=port.readouts,
            source_readout=port.source_readout,
            terminal_readout=port.terminal_readout,
        )


# ---------------------------------------------------------------------------
# 4. The nuclear block: refused, and why
# ---------------------------------------------------------------------------


def test_every_nuclear_region_is_the_reference_regions_equal(table):
    for key in pscen.PORT_ORDER:
        scenario = pscen.port(key)
        basal = ps.CLEARANCE_BUILDERS[key]().basal
        nuclear = [
            region
            for region in scenario.projection.regions
            if region in pdens.NUCLEAR_REGIONS
        ]
        assert nuclear, key
        for region in nuclear:
            assert scenario.projection.basal_concentration(
                region
            ) == pytest.approx(basal, rel=1e-9)
            assert scenario.projection.relative_density(region) == pytest.approx(
                1.0
            )


def test_a_nuclear_region_takes_release_from_its_own_nucleus_only(table):
    densities = table.target_densities(
        ["DR", "CS", "CP"], ("DR", "CS"), reference="CP"
    )
    reference_total = table.total("CP", ("DR", "CS"))
    assert densities["DR"]["DR"] == pytest.approx(reference_total)
    assert densities["DR"]["CS"] == 0.0
    assert densities["CS"]["CS"] == pytest.approx(reference_total)
    assert densities["CS"]["DR"] == 0.0
    # And a terminal region takes the derived density from every nucleus.
    assert densities["CP"]["DR"] == table.density("CP", "DR")
    assert densities["CP"]["CS"] == table.density("CP", "CS")


def test_the_tracer_self_projection_would_put_micromolar_serotonin_in_the_raphe(
    table
):
    # The numerical half of the module's case for refusing the nuclear
    # block, on the real committed numbers. This is what the tracer diagonal
    # says, and it is why the diagonal is not a projection density.
    nuclei = ps.NUCLEI_BY_TRANSMITTER["serotonin"]
    ratio = table.total("DR", nuclei) / table.total("CP", nuclei)
    assert ratio > 500.0
    clearance = ps.serotonin_clearance()
    demand = clearance.basal_release_rate() * ratio
    assert demand > clearance.vmax
    # ... and the mass balance then has NO finite steady state, which is the
    # model refusing the map rather than returning an enormous number.
    with pytest.raises(ValueError, match="no finite steady state"):
        proj.concentration_for_uptake(clearance, demand)


def test_the_nuclear_ratios_the_module_docstring_quotes_are_the_real_ones(table):
    # The table in "The model refuses the numbers", checked rather than
    # believed. Every nucleus's tracer self-density, over its density into
    # its transmitter's reference region.
    expected = {
        "VTA": 9.8, "SNc": 7.2, "DR": 931.0, "CS": 880.0,
        "LC": 1751.0, "NDB": 102.0, "PPN": 23.0, "LDT": 1543.0,
        # The exception the docstring names: histamine's reference region is
        # the hypothalamus around the nucleus, so nothing would have blown up
        # here. It is excluded on the mechanism argument alone.
        "TM": 1.27,
    }
    transmitter_of = {
        nucleus: transmitter
        for transmitter, keys in ps.NUCLEI_BY_TRANSMITTER.items()
        for nucleus in keys
    }
    for nucleus, quoted in expected.items():
        reference = pdens.REFERENCE_REGION[transmitter_of[nucleus]]
        ratio = table.density(nucleus, nucleus) / table.density(reference, nucleus)
        assert ratio == pytest.approx(quoted, rel=0.01), nucleus


# ---------------------------------------------------------------------------
# 5. The validation, including the part that fails
# ---------------------------------------------------------------------------


def test_the_validation_reports_one_pass_and_three_misses(table):
    rows = {
        (row["transmitter"], row["source"].split("]")[0]): row
        for row in table.validation_rows()
    }
    assert len(rows) == len(pdens.VALIDATION_RATIOS)
    by_transmitter = {}
    for row in table.validation_rows():
        by_transmitter.setdefault(row["transmitter"], []).append(row)

    # Dopamine against [Ihalainen1999]'s mouse 13-30x: IN RANGE, and it is
    # the only row that is. Against [Devoto2005]'s rat 6.3x it is 4.2x high,
    # and BOTH are reported, because the two measurements disagree with each
    # other by more than the model disagrees with either.
    dopamine = by_transmitter["dopamine"]
    assert dopamine[0]["verdict"] == "in range"
    assert dopamine[1]["error"] == pytest.approx(4.2, rel=0.05)
    assert all(row["same_direction"] for row in dopamine)

    # Serotonin: the failure. 6.8x out AND in the wrong direction, which is
    # the sharpest check in the module because [Mathews2004] measured both
    # regions in the same animals by the same method.
    serotonin = by_transmitter["serotonin"][0]
    assert serotonin["predicted"] == pytest.approx(0.292, rel=0.02)
    assert serotonin["error"] == pytest.approx(6.8, rel=0.05)
    assert not serotonin["same_direction"]

    # Noradrenaline: right direction, 2.2x low.
    noradrenaline = by_transmitter["noradrenaline"][0]
    assert noradrenaline["same_direction"]
    assert noradrenaline["error"] == pytest.approx(2.2, rel=0.05)

    # Acetylcholine: the second failure, and the larger one.
    acetylcholine = by_transmitter["acetylcholine"][0]
    assert not acetylcholine["same_direction"]
    assert acetylcholine["error"] > 10.0


def test_the_dorsal_raphe_row_alone_would_pass_the_check_the_pair_fails(table):
    # The diagnosis the serotonin port's notes state, as a number: with the
    # median raphe row left out, the striatum:cortex prediction is 4.3
    # against [Mathews2004]'s measured 2.0 -- the right direction and a
    # factor of 2 -- and it is the CS row that inverts it. That is what
    # points at cell-type non-specificity rather than at the formula.
    dr_only = table.total("CP", ("DR",)) / table.total("PFC", ("DR",))
    cs_only = table.total("CP", ("CS",)) / table.total("PFC", ("CS",))
    assert dr_only == pytest.approx(4.34, rel=0.02)
    assert cs_only == pytest.approx(0.208, rel=0.02)
    assert dr_only > 2.0 > cs_only


def test_the_lin2023_table_holds_no_injection_site_region():
    # [Lin2023]'s SI Table S2 lists the regions containing its AAV injection;
    # the authors exclude them from their own grading and so does this
    # module. The tuberomammillary nucleus itself is the one that matters.
    for acronym in ("TMd", "TMv", "LM", "SUM", "MM", "PH", "DMH", "VMH",
                    "LHA", "TU", "PSTN", "PMd", "PMv", "IF", "IPN"):
        assert acronym not in pdens.HISTAMINE_FIBRE_DENSITY


# ---------------------------------------------------------------------------
# 6. P6.1's map is still reachable
# ---------------------------------------------------------------------------


def test_the_uniform_table_reproduces_p61(table):
    uniform = pdens.uniform_table()
    for nucleus in table.nuclei:
        for region in table.regions:
            assert uniform.density(region, nucleus) == 1.0
    scenario = pscen.port("serotonin", density_map="uniform")
    levels = {
        region: scenario.projection.basal_concentration(region)
        for region in scenario.projection.regions
    }
    assert len(set(round(v, 12) for v in levels.values())) == 1


def test_the_reports_run_and_name_what_they_are_about(table):
    text = table.report()
    assert "nuclear" in text and "terminal" in text
    assert "REFUSED" in text
    for nucleus in table.nuclei:
        assert nucleus in text
    validation = table.validation_report()
    assert "OPPOSITE" in validation and "in range" in validation
    concentrations = pdens.concentration_report(table)
    assert "MEASURED" in concentrations and "predicted" in concentrations
    for transmitter in pscen.PORT_ORDER:
        assert transmitter in concentrations


# ---------------------------------------------------------------------------
# 7. Against the real 291-region files
# ---------------------------------------------------------------------------

full_data_only = pytest.mark.skipif(
    not os.path.exists(os.path.join("data", "connectome")),
    reason="needs the gitignored data/connectome cache; "
    "run scripts/fetch_connectome.py",
)


@pytest.fixture(scope="module")
def cache_inputs():
    ontology = connectome.load_ontology()
    strength, voxels = connectome.load_knox_inputs(ontology=ontology)
    return strength, voxels, ontology


@full_data_only
def test_the_committed_table_matches_the_whole_verified_cache(
    cache_inputs, table
):
    # The check the fixture cannot make: every one of the 162 committed
    # numbers, re-derived from the 291-region published matrices.
    strength, voxels, ontology = cache_inputs
    fresh = pdens.derive_table(strength, voxels, ontology)
    assert len(fresh) == len(table.densities)
    for (nucleus, region), value in fresh.items():
        assert value == table.density(region, nucleus), (nucleus, region)


@full_data_only
def test_the_medial_hypothalamus_reference_region_is_the_ontologys_own(
    cache_inputs
):
    # `MEZ_TERMINAL_FIELD` is written out rather than resolved at import
    # time, so it has to be checked against the ontology it claims to be: the
    # Knox source regions inside the medial hypothalamic zone, minus the
    # tuberomammillary nucleus, which is where [Itoh1991]'s probe was and
    # where this model's histamine field is NOT pinned to its own nucleus.
    strength, _, ontology = cache_inputs
    zone = ontology.id_for_acronym("MEZ")
    tm = set(pdens.NUCLEUS_STRUCTURES["TM"])
    members = [
        ontology[int(i)].acronym
        for i in strength.hemisphere("ipsi").source_ids
        if ontology.is_descendant_of(int(i), zone)
    ]
    assert set(pdens.MEZ_TERMINAL_FIELD) == set(members) - tm
    assert tm <= set(members)


@full_data_only
def test_the_derived_map_reproduces_lin2023s_measured_fibre_density(
    cache_inputs
):
    # THE STRONGEST CHECK IN THIS MODULE. [Lin2023] mapped histaminergic
    # fibres whole-brain in HDC-CreERT2 mice and registered the result to
    # Allen CCFv3, so it is mouse, cell-type specific, per-region and in the
    # same ontology as the connectome. If deriving a projection density from
    # a WILD-TYPE tracer row were a bad way to get a neuromodulatory
    # projection density, this is where it would show.
    strength, voxels, ontology = cache_inputs
    rows, spearman, log_pearson = pdens.fibre_density_rows(
        strength, voxels, ontology
    )
    assert len(rows) == len(pdens.HISTAMINE_FIBRE_DENSITY)
    assert spearman > 0.8
    assert log_pearson > 0.8
    # And it is not the injection site carrying it: the authors exclude the
    # 24 regions containing it, and dropping EVERY hypothalamic region -- a
    # stricter cut than theirs -- leaves the correlation where it was.
    _, without_hypothalamus, _ = pdens.fibre_density_rows(
        strength, voxels, ontology, exclude_hypothalamus=True
    )
    assert without_hypothalamus > 0.8
    assert abs(without_hypothalamus - spearman) < 0.05
    # A rank correlation against a SHUFFLED map is not 0.8; this pins the
    # claim to the pairing rather than to the spread of either variable.
    measured = np.array([row[2] for row in rows])
    derived = np.array([row[1] for row in rows])
    rng = np.random.default_rng(0)
    from scipy import stats

    shuffled = [
        abs(stats.spearmanr(derived, rng.permutation(measured)).statistic)
        for _ in range(200)
    ]
    assert max(shuffled) < spearman


@full_data_only
def test_the_fixture_and_the_cache_agree_on_every_pair_the_fixture_has(
    cache_inputs, fixture_inputs
):
    # The claim that lets the offline tests above stand for the real files.
    strength, voxels, ontology = cache_inputs
    f_strength, f_voxels, f_ontology = fixture_inputs
    for nucleus, region in FIXTURE_PAIRS:
        assert pdens.derive_density(
            strength, voxels, ontology,
            pdens.NUCLEUS_STRUCTURES[nucleus], pdens.REGION_STRUCTURES[region],
        ) == pdens.derive_density(
            f_strength, f_voxels, f_ontology,
            pdens.NUCLEUS_STRUCTURES[nucleus], pdens.REGION_STRUCTURES[region],
        )
