import hashlib

import pytest

from neurocascade.human.pet_observations import (
    FIXTURE_DIR,
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    _FIXTURES,
    ReportedOccupancyRelation,
    haloperidol_d2_manifests,
    haloperidol_d2_observations,
    occupancy_observations,
)
from neurocascade.human.pet_validation import (
    compare_pet_occupancy,
    observations_by_tracer,
    require_single_tracer,
)


def test_every_transcription_still_matches_its_recorded_digest():
    for dataset_id, (filename, expected) in _FIXTURES.items():
        digest = hashlib.sha256((FIXTURE_DIR / filename).read_bytes()).hexdigest()
        assert digest == expected, dataset_id


def test_a_changed_transcription_is_refused_rather_than_loaded(tmp_path, monkeypatch):
    # An edit to a committed observation, for instance to make a downstream
    # prediction land, has to fail loudly rather than pass silently.
    import neurocascade.human.pet_observations as module

    original = dict(module._FIXTURES)
    filename = original["videbaek2001"][0]
    monkeypatch.setitem(module._FIXTURES, "videbaek2001", (filename, "0" * 64))
    with pytest.raises(ValueError, match="has changed since it was recorded"):
        module.haloperidol_d2_observations("videbaek2001")


def test_manifests_validate_and_declare_themselves_transcribed():
    manifests = haloperidol_d2_manifests()
    assert len(manifests) == len(_FIXTURES)
    for manifest in manifests:
        assert manifest.layer == "pet"
        assert manifest.derivation == "transcribed"
        assert manifest.local_path
        assert manifest.checksum
        # The checksum locks our reading, not its fidelity to the paper, and
        # the manifest has to say so rather than let a reader assume otherwise.
        assert "fidelity to the source" in manifest.checksum_proves


def test_every_observation_names_its_tracer_or_records_that_it_cannot():
    observations = haloperidol_d2_observations()
    assert len(observations) == 15
    untraced = [item for item in observations if item.tracer is None]
    # Nyberg 1995 and Regenthal 1997 print occupancies and plasma levels in
    # their abstracts and never name the ligand, two points each.
    assert len(untraced) == 4
    for item in untraced:
        assert "does not name the tracer" in item.notes or "names neither the tracer" in item.notes, item.notes
    with pytest.raises(ValueError, match="do not name a tracer"):
        observations_by_tracer(observations)


def test_tracers_present_are_the_three_that_were_actually_reachable():
    tracers = sorted(observations_by_tracer(
        [item for item in haloperidol_d2_observations() if item.tracer]
    ))
    assert tracers == ["[11C]raclopride", "[123I]IBZM", "[76Br]FLB 457"]
    with pytest.raises(ValueError, match="must not be pooled"):
        require_single_tracer([item for item in haloperidol_d2_observations() if item.tracer])


def test_the_xiberas_points_are_binding_indices_and_cannot_be_differenced():
    points = haloperidol_d2_observations("xiberas2001")
    assert len(points) == 4
    for point in points:
        assert point.quantity == "binding-index"
        assert point.region == "striatum"
        assert point.value_origin == "table"
        assert point.plasma_concentration_basis == "total"
        with pytest.raises(ValueError, match="did not report an occupancy"):
            compare_pet_occupancy(point, 0.8)
    doses = [point.dose for point in points]
    assert doses == [3.0, 8.0, 15.0, 60.0]


def test_the_videbaek_points_are_free_plasma_occupancies():
    points = haloperidol_d2_observations("videbaek2001")
    assert len(points) == 4
    assert [point.observed_occupancy for point in points] == [0.29, 0.36, 0.25, 0.16]
    assert [point.plasma_concentration_nM for point in points] == [0.068, 0.133, 0.064, 0.073]
    for point in points:
        assert point.tracer == "[123I]IBZM"
        assert point.plasma_concentration_basis == "free"
        assert point.dose_regimen == "intravenous-infusion"
    # Each point is its own subject, so a per-subject hyperbola is identified:
    # occupancy = C / (C + Kd) recovers the Kd column the paper prints.
    for point, printed_kd in zip(points, [0.16, 0.24, 0.19, 0.39]):
        occupancy = point.observed_occupancy
        implied = point.plasma_concentration_nM * (1.0 - occupancy) / occupancy
        assert implied == pytest.approx(printed_kd, abs=0.01)


def test_the_kapur_points_are_intervals_with_no_invented_mean():
    for dataset_id, low, high in (
        ("kapur1996", 0.53, 0.74),
        ("kapur1997", 0.53, 0.88),
        ("kapur2000", 0.38, 0.87),
    ):
        (point,) = haloperidol_d2_observations(dataset_id)
        assert point.tracer == "[11C]raclopride"
        assert point.is_interval_only
        assert point.observed_range_low == low
        assert point.observed_range_high == high
        with pytest.raises(ValueError, match="range"):
            compare_pet_occupancy(point, 0.7)


def test_the_depot_points_carry_means_ranges_and_plasma_levels():
    nyberg = haloperidol_d2_observations("nyberg1995")
    assert [point.observed_occupancy for point in nyberg] == [0.73, 0.52]
    assert [point.plasma_concentration_nM for point in nyberg] == [4.6, 2.3]
    regenthal = haloperidol_d2_observations("regenthal1997")
    assert [point.observed_occupancy for point in regenthal] == [0.75, 0.53]
    assert [point.plasma_concentration_nM for point in regenthal] == [7.3, 1.8]
    for point in nyberg + regenthal:
        assert point.dose_regimen == "depot-injection"
        assert point.dose is None
        assert point.plasma_concentration_basis == "total"
        assert point.n == 8


def test_occupancy_observations_drops_everything_that_is_not_a_usable_occupancy():
    usable = occupancy_observations()
    # Four depot group means and the four Videbaek subjects. The three Kapur
    # intervals have no central value and the four Xiberas points are binding
    # indices, so eleven of fifteen registered points cannot be differenced.
    assert len(usable) == 8
    for point in usable:
        assert point.quantity == "occupancy"
        assert point.observed_occupancy is not None


def test_the_two_reported_relations_disagree_and_are_kept_apart():
    kapur, fitzgerald = HALOPERIDOL_D2_REPORTED_RELATIONS
    assert kapur.label == "kapur1997"
    assert fitzgerald.label == "fitzgerald2000"
    assert kapur.ec50_ng_per_mL == 0.51
    assert fitzgerald.ec50_ng_per_mL == 0.40
    # Kapur's abstract prints 80% occupancy at 2.0 ng/ml, which is what its own
    # stated 50% point implies. That internal consistency is the only check
    # available on a paper whose full text could not be reached.
    assert kapur.occupancy_at(2.04) == pytest.approx(0.80, abs=0.005)
    assert kapur.ec50_nM == pytest.approx(1.357, abs=1e-3)
    assert fitzgerald.ec50_nM == pytest.approx(1.064, abs=1e-3)
    # The two are 28% apart. Nothing here averages them.
    assert kapur.ec50_ng_per_mL / fitzgerald.ec50_ng_per_mL == pytest.approx(1.275)


def test_a_reported_relation_rejects_impossible_values():
    values = dict(
        label="x",
        compound="haloperidol",
        target="D2",
        tracer="[11C]raclopride",
        region=None,
        n=1,
        model="hyperbola",
        ec50_ng_per_mL=0.5,
        ec50_as_printed="0.5 ng/ml",
        plasma_concentration_basis="total",
        source="source",
        source_locator="abstract",
        value_origin="text",
    )
    with pytest.raises(ValueError, match="ec50_ng_per_mL"):
        ReportedOccupancyRelation(**{**values, "ec50_ng_per_mL": 0.0})
    with pytest.raises(ValueError, match="n must be"):
        ReportedOccupancyRelation(**{**values, "n": 0})
    with pytest.raises(ValueError, match="total or free"):
        ReportedOccupancyRelation(**{**values, "plasma_concentration_basis": "bound"})
    with pytest.raises(ValueError, match="must not be negative"):
        ReportedOccupancyRelation(**values).occupancy_at(-1.0)


def test_an_unknown_dataset_id_is_an_error_not_an_empty_set():
    with pytest.raises(KeyError, match="unknown dataset_id"):
        haloperidol_d2_observations("farde1988")
