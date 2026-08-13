"""P6.1: the cited constants, the nuclei, and the volume-transmission
timescale argument the acceptance criteria ask to be documented.

The argument is a computation, not a paragraph, so these tests re-run it.
"""

import numpy as np
import pytest

from neurocascade import connectome
from neurocascade import monoamine_transporters as mt
from neurocascade import projection_systems as ps
from neurocascade.projection import ProjectionTarget


REST = 0.5


# ---------------------------------------------------------------------------
# The volume-transmission timescale argument
# ---------------------------------------------------------------------------


def test_the_two_release_site_densities_agree_within_a_factor_of_two():
    # [Dreyer2010]'s rho = 0.104 um^-3 and [Ozcete2024]'s "one site per
    # 25 um3" are independent measurements of the length scale the
    # well-mixed-region assumption rests on. They must agree, and the report
    # states by how much.
    dreyer = ps.release_site_spacing()
    ozcete = ps.DOPAMINE_RELEASE_SITE_VOLUME_OZCETE ** (1.0 / 3.0)
    assert dreyer == pytest.approx(2.13, abs=0.01)
    assert ozcete == pytest.approx(2.92, abs=0.01)
    assert max(dreyer, ozcete) / min(dreyer, ozcete) < 2.0


def test_diffusion_mixes_the_innervation_mesh_but_not_a_region():
    # The distinction the module docstring turns on, asserted both ways so
    # that neither half can be quietly dropped.
    mesh = ps.diffusion_time_s(ps.release_site_spacing())
    region = ps.diffusion_time_s(1000.0)
    assert mesh < 0.01  # milliseconds
    assert region > 300.0  # longer than most runs in this codebase
    assert region / mesh > 1e4


def test_every_transmitter_pool_relaxes_far_faster_than_the_circuit():
    # The chemistry half of the same argument. 40x is the floor because
    # serotonin's 2.1 s against a 90 s circuit time constant is the smallest
    # ratio of the five; the assertion is on the ratio, and the numbers it is
    # computed from are all elsewhere and cited.
    circuit_s = 60.0 * ps.CIRCUIT_TAU_MIN
    for name, builder in ps.CLEARANCE_BUILDERS.items():
        tau_s = 60.0 * ps.clearance_time_constant_min(builder())
        assert tau_s < circuit_s / 40.0, name


def test_the_measured_clearance_time_constants_are_the_published_ones():
    # Not new numbers: P5.8's, restated as time constants. Checked against
    # the constants module rather than against literals, so a change there
    # cannot leave this documentation stale.
    dopamine = 60.0 * ps.clearance_time_constant_min(ps.dopamine_clearance())
    assert dopamine == pytest.approx(60.0 * mt.KM_DOPAMINE / mt.VMAX_DOPAMINE)
    assert dopamine == pytest.approx(0.0516, abs=1e-4)  # 52 ms
    serotonin = 60.0 * ps.clearance_time_constant_min(ps.serotonin_clearance())
    assert serotonin == pytest.approx(60.0 * mt.KM_SEROTONIN / mt.VMAX_SEROTONIN)
    noradrenaline = 60.0 * ps.clearance_time_constant_min(ps.noradrenaline_clearance())
    assert noradrenaline == pytest.approx(
        mt.NORADRENALINE_CLEARANCE_HALF_LIFE_S / np.log(2.0), rel=1e-9
    )


def test_the_timescale_report_states_both_halves_of_the_argument():
    text = ps.timescale_report()
    assert "release-site spacing" in text
    assert "NOT diffusively mixed" in text
    for transmitter in ps.CLEARANCE_BUILDERS:
        assert transmitter in text
    for name in ps.ILLUSTRATIVE_CLEARANCE:
        assert name in text
    for key in ps.NUCLEUS_ORDER:
        assert ps.NUCLEI[key]["name"] in text


# ---------------------------------------------------------------------------
# What is measured and what is not
# ---------------------------------------------------------------------------


def test_the_three_monoamine_clearances_are_p5_8_s_objects_unchanged():
    for builder, reference in (
        (ps.dopamine_clearance, mt.dat()),
        (ps.serotonin_clearance, mt.sert()),
        (ps.noradrenaline_clearance, mt.net()),
    ):
        built = builder()
        assert built.km == reference.km
        assert built.vmax == reference.vmax
        assert built.clearance_rate == reference.clearance_rate
        assert built.basal == reference.basal
        assert [lig.name for lig in built.ligands] == [
            lig.name for lig in reference.ligands
        ]


def test_only_acetylcholine_and_histamine_carry_an_illustrative_clearance():
    assert set(ps.ILLUSTRATIVE_CLEARANCE) == {"acetylcholine", "histamine"}
    # And the two illustrative ones carry no drug, because no inhibition
    # constant for a cholinesterase or HNMT inhibitor is sourced here. An
    # empty ligand list is P5.8's way of representing a gap; an implied
    # infinite Ki is not.
    for name in ps.ILLUSTRATIVE_CLEARANCE:
        assert ps.CLEARANCE_BUILDERS[name]().ligands == []


def test_the_acetylcholine_basal_is_the_cited_measurement_and_the_fold_is_invariant():
    # P6.1 had a unit scale here, on the ground that no measurement was
    # reachable, and asserted that the FOLD was invariant to the choice so
    # that the unit could not quietly acquire meaning. P6.2 supplies the
    # measurement ([Vinson1997], rat striatum, no-net-flux, 18.4 nM), so the
    # first assertion changes and the invariance one is KEPT: what it now
    # asserts is that P6.2's change of this constant moved no fold this
    # module reports, which is half of `endogenous_tone.change_report()`'s
    # claim. The sweep still spans four orders of magnitude and now includes
    # P6.1's own 1.0 and [DeBundel2008]'s 4 nM.
    assert ps.ACETYLCHOLINE_BASAL == 18.4
    assert ps.acetylcholine_clearance().basal == 18.4
    assert "Vinson1997" in ps.CITATIONS
    assert "18.4 +/- 11.8 nM" in ps.CITATIONS["Vinson1997"]
    folds = []
    for basal in (0.01, 1.0, ps.ACETYLCHOLINE_BASAL_HIPPOCAMPAL_ESTIMATE,
                  ps.ACETYLCHOLINE_BASAL, 100.0):
        system = _one_region("acetylcholine", basal=basal)
        folds.append(
            [
                system.steady_state("PFC", _rates(system, rate))
                / system.basal_concentration("PFC")
                for rate in (0.25, REST, 1.0)
            ]
        )
    for other in folds[1:]:
        assert other == pytest.approx(folds[0], rel=1e-12)


def test_the_illustrative_clearance_rates_change_no_reported_fold():
    # The two illustrative clearance rates are TIMESCALES, not scales. The
    # steady-state fold a firing-rate change produces cannot depend on them,
    # and a sweep over two orders of magnitude asserts it.
    for transmitter in ps.ILLUSTRATIVE_CLEARANCE:
        base = None
        for factor in (0.1, 1.0, 10.0):
            builder = ps.CLEARANCE_BUILDERS[transmitter]
            reference = builder()
            clearance = builder(rate=reference.clearance_rate * factor)
            system = _one_region(transmitter, clearance=clearance)
            folds = [
                system.steady_state("PFC", _rates(system, rate))
                / system.basal_concentration("PFC")
                for rate in (0.25, REST, 1.0)
            ]
            if base is None:
                base = folds
            else:
                assert folds == pytest.approx(base, rel=1e-12)


def test_the_illustrative_acetylcholine_rate_matches_its_stated_derivation():
    # A labelled illustrative constant still has to be the number its own
    # docstring says it is, or the label hides an error rather than a choice.
    assert ps.ACETYLCHOLINE_CLEARANCE_HALF_LIFE_S == 0.040
    assert ps.ACETYLCHOLINE_CLEARANCE_RATE == pytest.approx(
        60.0 * np.log(2.0) / 0.040
    )
    assert ps.HISTAMINE_CLEARANCE_RATE == pytest.approx(
        mt.VMAX_SEROTONIN / mt.KM_SEROTONIN
    )


def test_the_histamine_basal_is_the_cited_measurement():
    assert ps.HISTAMINE_BASAL == 7.8
    assert ps.histamine_clearance().basal == 7.8
    assert "Itoh1991" in ps.CITATIONS
    assert "7.8 nM" in ps.CITATIONS["Itoh1991"]
    # And its caveats are recorded where a reader will meet them.
    assert "Cumming1991" in ps.CITATIONS
    assert "NOT a clearance rate" in ps.CITATIONS["Cumming1991"]


def _one_region(transmitter, clearance=None, basal=None):
    builder = ps.CLEARANCE_BUILDERS[transmitter]
    if clearance is None:
        clearance = builder(basal=basal) if basal is not None else builder()
    return ps.projection_system(
        transmitter,
        [ProjectionTarget("PFC", {node: 1.0 for node in _nodes(transmitter)})],
        REST,
        clearance=clearance,
    )


def _rates(system, rate):
    """One firing rate per source nucleus of `system`."""
    return [rate] * len(system.sources)


def _nodes(transmitter):
    return [ps.NUCLEI[key]["acronym"] for key in ps.NUCLEI_BY_TRANSMITTER[transmitter]]


# ---------------------------------------------------------------------------
# The nuclei
# ---------------------------------------------------------------------------


def test_every_nucleus_the_brief_names_is_present_with_its_transmitter():
    expected = {
        "dopamine": {"VTA", "SNc"},
        "serotonin": {"DR", "CS"},
        "noradrenaline": {"LC"},
        "acetylcholine": {"NDB", "PPN", "LDT"},
        "histamine": {"TM"},
    }
    for transmitter, keys in expected.items():
        assert set(ps.NUCLEI_BY_TRANSMITTER[transmitter]) == keys
        for key in keys:
            assert ps.NUCLEI[key]["transmitter"] == transmitter
    assert set(ps.NUCLEUS_ORDER) == set(ps.NUCLEI)


def test_every_nucleus_acronym_and_id_is_the_allen_ccf_s():
    # The nuclei are named by CCF acronym and id so P6.3 can look their
    # connectivity rows up without a second naming convention. That claim is
    # only worth making if the ids are right, so it is checked against the
    # COMMITTED fixture ontology: the full structure graph is gitignored, and
    # a check that only runs where someone happens to have downloaded it is
    # not a check. P6.1 added the five nuclei the fixture did not already
    # carry to `connectome.PROJECTION_NUCLEUS_ONLY_REGION_IDS` for exactly
    # this reason.
    ccf = connectome.load_fixture_ontology()
    for key, nucleus in ps.NUCLEI.items():
        assert nucleus["ccf_id"] in ccf.structures, key
        assert ccf.acronym_for_id(nucleus["ccf_id"]) == nucleus["acronym"], key
        assert ccf.name_for_id(nucleus["ccf_id"]).startswith(
            nucleus["name"].split(" (")[0]
        ), key
        assert nucleus["acronym"] == key
    # And every one of them is an ontology-only entry or an existing fixture
    # region, never a new matrix row: the committed Knox and Oh numbers are
    # P3.1's and P6.1 did not touch them.
    for key in ps.NUCLEUS_ORDER:
        ccf_id = ps.NUCLEI[key]["ccf_id"]
        assert (
            ccf_id in connectome.FIXTURE_REGION_IDS
            or ccf_id in connectome.PROJECTION_NUCLEUS_ONLY_REGION_IDS
            or ccf_id in connectome.ANATOMY_ONLY_REGION_IDS
        ), key


def test_the_firing_rates_are_context_and_not_parameters():
    # The release law consumes the RATIO of dimensionless Wilson-Cowan
    # rates. If any Hz value were reaching an equation, replacing the whole
    # table with nonsense would change a number; nothing does.
    system = _one_region("dopamine")
    before = system.steady_state("PFC", _rates(system, 0.8))
    original = dict(ps.FIRING_RATES)
    try:
        for key in ps.FIRING_RATES:
            ps.FIRING_RATES[key] = "nonsense"
        rebuilt = _one_region("dopamine")
        after = rebuilt.steady_state("PFC", _rates(rebuilt, 0.8))
    finally:
        ps.FIRING_RATES.update(original)
    assert after == before


def test_the_acetylcholine_firing_rate_gap_is_recorded_rather_than_filled():
    text = ps.FIRING_RATES["acetylcholine"]
    assert "SEARCHED AND EMPTY" in text
    assert "Lee2005" in text
    assert "Hz" not in text.replace("no reachable Hz value", "")


def test_source_nuclei_default_to_ccf_acronyms_and_accept_renaming():
    default = ps.source_nuclei("dopamine", REST)
    assert [s.node for s in default] == ["VTA", "SNc"]
    assert all(s.rest_rate == REST for s in default)
    renamed = ps.source_nuclei("dopamine", REST, nodes={"VTA": "VTA_DA"})
    assert [s.node for s in renamed] == ["VTA_DA", "SNc"]
    per_nucleus = ps.source_nuclei("dopamine", [0.4, 0.6])
    assert [s.rest_rate for s in per_nucleus] == [0.4, 0.6]
    with pytest.raises(ValueError, match="resting rates"):
        ps.source_nuclei("dopamine", [0.4, 0.5, 0.6])


def test_a_projection_system_is_built_for_every_transmitter_the_brief_names():
    for transmitter, builder in ps.PROJECTION_BUILDERS.items():
        targets = ps.illustrative_targets(
            {"PFC": {node: 1.0 for node in _nodes(transmitter)}}
        )
        system = builder(targets, REST)
        assert system.transmitter == transmitter
        assert list(system.source_nodes) == _nodes(transmitter)
        assert system.basal_concentration("PFC") > 0.0
        assert all(
            provenance == "illustrative"
            for _, _, _, provenance in system.density_report()
        )


def test_an_unknown_transmitter_is_an_error_rather_than_a_default():
    with pytest.raises(ValueError, match="no clearance model"):
        ps.projection_system("glutamate", [ProjectionTarget("PFC", 1.0)], REST)


def test_the_release_scale_is_derived_and_carries_no_free_parameter():
    # Stated in the module docstring as the reason the projection layer
    # introduces no new absolute concentration scale.
    for transmitter in ps.CLEARANCE_BUILDERS:
        clearance = ps.CLEARANCE_BUILDERS[transmitter]()
        system = _one_region(transmitter, clearance=clearance)
        density = system.density("PFC")
        assert system.release_scale == pytest.approx(
            clearance.basal_release_rate() / density
        )
        assert system.basal_concentration("PFC") == pytest.approx(clearance.basal)
