"""Tests for `neurocascade.calibration.behavioral_evidence` (P2.1).

Three things are checked, matching the three acceptance criteria of the task:

1. **Provenance is complete or explicitly gapped.** Every entry carries drug,
   species, strain, sex, route, dose, timepoint, endpoint, effect and error
   bars, and any of those that is absent is absent *with a recorded reason*.
   The entry constructor enforces the pairing; these tests prove the
   enforcement bites and that the shipped entries satisfy it.
2. **Each bridgeable entry converts to a brain concentration** through the
   Phase 1 machinery (`drug_pk` parameter set -> integrated PK curve ->
   `brain_partition`), with the physical sanity of the result asserted
   (positive, dose-proportional where the model is linear, weight-invariant
   where the parameterisation is per kg).
3. **Every entry that remains unusable says why**, and the reason is
   *computed* from `drug_pk`'s own coverage rather than asserted in prose, so
   these tests fail if a later phase sources a route and nobody revisits the
   verdicts.
"""

import numpy as np
import pytest

from neurocascade.legacy import drug_pk
from neurocascade.calibration import behavioral_evidence as be


# ---------------------------------------------------------------------------
# 1. Provenance
# ---------------------------------------------------------------------------


def test_every_entry_has_a_citation_and_an_endpoint():
    assert be.BEHAVIORAL_EVIDENCE, "the module must not be empty"
    for entry in be.BEHAVIORAL_EVIDENCE:
        assert entry.citation.strip()
        assert entry.endpoint.strip()
        assert entry.source_access.strip()
        # doi or PMID or PMC id: something a reader can resolve.
        assert any(
            tag in entry.citation for tag in ("doi:", "PMID", "PMC")
        ), entry.key


def test_every_absent_provenance_field_carries_a_reason():
    for entry in be.BEHAVIORAL_EVIDENCE:
        for field in be.GAPPABLE_FIELDS:
            if getattr(entry, field) is None:
                assert field in entry.provenance_gaps, (entry.key, field)
                assert len(entry.provenance_gaps[field]) > 20, (entry.key, field)


def test_constructor_refuses_an_unexplained_gap():
    with pytest.raises(ValueError, match="no recorded reason"):
        be.BehavioralEvidenceEntry(
            key="test/unexplained",
            drug="diazepam",
            species="rat",
            strain="Wistar",
            sex="male",
            route="IP",
            doses_mg_per_kg=(1.0,),
            timepoint_min=None,  # gap, not declared
            endpoint="elevated plus maze",
            effect=be.UnavailableEffect("open-arm time", "not published"),
            citation="doi:10.0000/test",
            source_access="test",
            body_weight_kg=0.25,
        )


def test_constructor_refuses_a_gap_recorded_for_a_field_that_has_a_value():
    with pytest.raises(ValueError, match="recorded as provenance gaps"):
        be.BehavioralEvidenceEntry(
            key="test/spurious",
            drug="diazepam",
            species="rat",
            strain="Wistar",
            sex="male",
            route="IP",
            doses_mg_per_kg=(1.0,),
            timepoint_min=30.0,
            endpoint="elevated plus maze",
            effect=be.UnavailableEffect("open-arm time", "not published"),
            citation="doi:10.0000/test",
            source_access="test",
            body_weight_kg=0.25,
            provenance_gaps={"timepoint_min": "but it has one"},
        )


def test_effect_requires_a_dispersion_kind_when_error_bars_are_given():
    with pytest.raises(ValueError, match="dispersion_kind"):
        be.PublishedEffect(
            kind="group_means",
            measure="open-arm time",
            units="s",
            values={"vehicle": 10.0, "drug": 20.0},
            dispersion={"vehicle": 1.0, "drug": 2.0},
        )


def test_effect_refuses_a_dispersion_that_does_not_match_its_values():
    with pytest.raises(ValueError, match="do not match"):
        be.PublishedEffect(
            kind="group_means",
            measure="open-arm time",
            units="s",
            values={"vehicle": 10.0, "drug": 20.0},
            dispersion={"vehicle": 1.0},
            dispersion_kind="SEM",
        )


def test_uncited_entry_is_refused():
    with pytest.raises(ValueError, match="required"):
        be.BehavioralEvidenceEntry(
            key="test/uncited",
            drug="diazepam",
            species="rat",
            strain="Wistar",
            sex="male",
            route="IP",
            doses_mg_per_kg=(1.0,),
            timepoint_min=30.0,
            endpoint="elevated plus maze",
            effect=be.UnavailableEffect("open-arm time", "not published"),
            citation="",
            source_access="test",
            body_weight_kg=0.25,
        )


def test_the_one_entry_with_published_error_bars_records_them_as_published():
    entry = be.DIAZEPAM_EPM_STICKLING2024
    effect = entry.effect
    assert isinstance(effect, be.PublishedEffect)
    assert effect.has_error_bars
    assert effect.dispersion_kind == "SEM"
    # Values as printed in the source's Table 1, non-stressed rats.
    assert effect.values == {"saline": 77.5, "diazepam 1.5 mg/kg": 76.7}
    assert effect.dispersion == {"saline": 12.1, "diazepam 1.5 mg/kg": 24.4}
    assert effect.group_sizes["saline"] == 12
    assert entry.route == "IP"
    assert entry.doses_mg_per_kg == (1.5,)
    assert entry.timepoint_min == 60.0
    assert entry.strain == "Sprague Dawley (Envigo)"
    assert entry.sex == "male"


def test_track_e_unverified_fentanyl_dose_claim_is_not_stored_as_a_dose():
    """Rule 1 applied to this repository's own prose.

    Track E states [Yoshida1999] used 5 and 10 ug/kg IV. That could not be
    re-verified, so the entry must carry no dose at all, and must say why.
    """
    entry = be.FENTANYL_NAC_DOPAMINE_YOSHIDA1999
    assert entry.doses_mg_per_kg is None
    reason = entry.provenance_gaps["doses_mg_per_kg"]
    assert reason == be.TRACK_E_UNVERIFIED_FENTANYL_DOSE_CLAIM
    assert "5 and 10 ug/kg" in reason
    assert "microg/kg" in reason  # what the abstract actually says


# ---------------------------------------------------------------------------
# 2. The bridge: dose -> brain concentration
# ---------------------------------------------------------------------------


def test_exactly_the_expected_entries_are_bridgeable():
    """P2.2's version of this test listed two entries and said, in a
    comment, that both were diazepam/rat/IP because that is "the one
    drug/species/route triple this codebase has a PK model AND a
    receptor-drivable brain partition for". P7.3 added six entries, three of
    them fentanyl/rat/IV, and doing so split that sentence in two: `drug_pk`
    builds a PK model for fentanyl/rat/IV, so those entries ARE bridgeable
    in the sense this function means, and their brain partition is
    total-brain-only, so they cannot drive a receptor. The distinction is
    now a separate method (`receptor_drive_status`) with its own test below,
    and this one is widened to the six entries whose PK step works.
    """
    bridgeable = {e.key for e in be.bridgeable_entries()}
    assert bridgeable == {
        "diazepam/elevated-plus-maze/Stickling2024",
        "diazepam/elevated-plus-maze/Graziani2022",
        "diazepam/open-field/Graziani2022",
        "diazepam/locomotor-activity/Sagara2008",
        "diazepam/hot-plate/Lustyk2023",
        "fentanyl/tail-flick/Yassen2008",
        "fentanyl/righting-reflex/Obert2025",
        "fentanyl/righting-reflex-ED50/Kissin1983",
    }
    assert len(be.unbridgeable_entries()) == len(be.BEHAVIORAL_EVIDENCE) - 8


def test_only_the_diazepam_entries_can_actually_drive_a_receptor():
    """The distinction P7.3 had to add, and the reason its analgesia and
    sedation fentanyl datasets are recorded rather than fitted.

    `drug_pk.brain_partition_status('fentanyl', 'rat')` is
    'total-brain-only': no rat brain unbound fraction for fentanyl was
    found (P1.5), so its Kp of ~16.5 is mostly tissue binding and must not
    be used as if it were a Kp,uu. Computed from `drug_pk`, so sourcing
    that number later flips these verdicts without this test being edited.
    """
    drivable = {e.key for e in be.BEHAVIORAL_EVIDENCE if e.can_drive_receptor}
    assert all(k.startswith("diazepam/") for k in drivable), drivable
    for key in ("fentanyl/tail-flick/Yassen2008",
                "fentanyl/righting-reflex/Obert2025",
                "fentanyl/righting-reflex-ED50/Kissin1983"):
        entry = be.BEHAVIORAL_EVIDENCE_BY_KEY[key]
        assert entry.is_bridgeable
        assert entry.receptor_drive_status()[0] == "total-brain-only"


def test_bridgeable_entry_converts_a_published_dose_to_an_unbound_brain_nM():
    """The acceptance criterion, exercised end to end.

    1.5 mg/kg IP diazepam in a rat, evaluated 60 min after dosing (the
    source's own timepoint), through `drug_pk.diazepam_rat_ip_pk` and the
    sourced rat diazepam Kp,uu, must give a finite positive UNBOUND BRAIN
    concentration in nM.
    """
    entry = be.DIAZEPAM_EPM_STICKLING2024
    exposure = entry.brain_exposure(body_weight_kg=0.35)

    assert exposure.dose_mg_per_kg == 1.5
    assert exposure.timepoint_min == 60.0
    assert exposure.plasma_nM > 0.0
    assert np.isfinite(exposure.plasma_nM)
    assert exposure.brain_nM is not None
    assert exposure.brain_basis == "unbound brain"
    assert exposure.drives_receptor
    assert exposure.partition_status == "receptor-drivable"
    assert exposure.receptor_compartment == "unbound brain"

    # Unbound brain sits below total plasma for this drug, because
    # Kp,uu * fu_plasma = 0.918 * 0.196 < 1 (P1.5's derivation).
    partition = drug_pk.diazepam_rat_brain_partition()
    assert exposure.brain_nM == pytest.approx(
        partition.unbound_brain_factor * exposure.plasma_nM, rel=1e-12
    )
    assert 0.0 < exposure.brain_nM < exposure.plasma_nM

    # The peak of the curve is above the 60 min value, i.e. the entry's
    # timepoint is on the declining limb: Tmax of this model is ~17 min.
    assert exposure.cmax_nM > exposure.plasma_nM


def test_bridged_concentration_is_dose_proportional():
    """Both dosed PK models here are linear, so doubling the dose must
    exactly double the concentration. This is what makes a mg/kg dose usable
    as a calibration input at all."""
    entry = be.DIAZEPAM_EPM_STICKLING2024
    single = entry.brain_exposure(body_weight_kg=0.35, dose_mg_per_kg=1.5)
    double = entry.brain_exposure(body_weight_kg=0.35, dose_mg_per_kg=3.0)
    assert double.brain_nM == pytest.approx(2.0 * single.brain_nM, rel=1e-6)


def test_bridged_concentration_is_weight_invariant_for_the_per_kg_model():
    """The Stickling entry records no body weight, and this is why that gap
    is harmless for this particular entry: the rat IP diazepam parameter set
    is per kg (V/F 19.3 L/kg, CL/F 255 mL/min/kg), so the concentration a
    mg/kg dose produces does not depend on the animal's size."""
    entry = be.DIAZEPAM_EPM_STICKLING2024
    light = entry.brain_exposure(body_weight_kg=0.25)
    heavy = entry.brain_exposure(body_weight_kg=0.50)
    assert heavy.brain_nM == pytest.approx(light.brain_nM, rel=1e-9)


def test_brain_exposure_refuses_to_invent_a_dose():
    """An entry with no recorded dose must not silently pick one."""
    entry = be.FENTANYL_NAC_DOPAMINE_YOSHIDA1999
    assert entry.doses_mg_per_kg is None
    with pytest.raises(ValueError, match="records no dose"):
        entry.brain_exposure(body_weight_kg=0.2375)


def test_brain_exposure_refuses_to_invent_a_timepoint():
    """An entry that has a dose but no recorded measurement time must not
    silently pick one either: "when was this measured" is a property of the
    experiment, and defaulting it would attach a real effect size to a
    concentration nobody reported."""
    dosed_but_untimed = be.BehavioralEvidenceEntry(
        key="test/dosed-but-untimed",
        drug="diazepam",
        species="rat",
        strain="Sprague Dawley",
        sex="male",
        route="IP",
        doses_mg_per_kg=(1.5,),
        timepoint_min=None,
        endpoint="test fixture",
        effect=be.UnavailableEffect("test fixture", "test fixture only"),
        citation="test fixture, doi:10.0000/none",
        source_access="test",
        body_weight_kg=0.30,
        provenance_gaps={
            "timepoint_min": "test fixture: deliberately absent, see test name"
        },
    )
    assert dosed_but_untimed.is_bridgeable
    with pytest.raises(ValueError, match="records no timepoint"):
        dosed_but_untimed.brain_exposure(body_weight_kg=0.30)
    # ...but it works the moment the caller supplies one explicitly.
    exposure = dosed_but_untimed.brain_exposure(
        body_weight_kg=0.30, timepoint_min=30.0
    )
    assert exposure.brain_nM > 0.0


def test_the_bridge_would_work_for_fentanyl_iv_the_moment_a_dose_arrives():
    """The fentanyl entry's blocker is the paper, not the model.

    Built as a hypothetical with an explicitly made-up dose, so this test
    demonstrates the machinery without smuggling an invented dose into the
    evidence module itself.
    """
    hypothetical = be.BehavioralEvidenceEntry(
        key="test/fentanyl-iv-hypothetical",
        drug="fentanyl",
        species="rat",
        strain="Sprague-Dawley",
        sex="male",
        route="IV",
        doses_mg_per_kg=(0.010,),  # NOT a literature value; a test fixture
        timepoint_min=20.0,
        endpoint="hypothetical",
        effect=be.UnavailableEffect("hypothetical", "test fixture only"),
        citation="test fixture, doi:10.0000/none",
        source_access="test",
        body_weight_kg=0.2375,
    )
    assert hypothetical.is_bridgeable
    exposure = hypothetical.brain_exposure(body_weight_kg=0.2375)
    assert exposure.plasma_nM > 0.0
    # fentanyl/rat has a Kp but no measured rat fu_brain, so P1.5 allows a
    # TOTAL brain concentration and forbids using it to drive a receptor.
    assert exposure.brain_basis == "total brain"
    assert not exposure.drives_receptor
    assert exposure.partition_status == "total-brain-only"
    assert exposure.receptor_compartment == (
        "total plasma (no brain partition attached)"
    )
    assert exposure.brain_nM > exposure.plasma_nM  # Kp ~ 16


# ---------------------------------------------------------------------------
# 3. Unusable entries say why, and the reason is computed
# ---------------------------------------------------------------------------


def test_yoshida_is_blocked_by_the_dataset_not_by_the_bridge():
    entry = be.FENTANYL_NAC_DOPAMINE_YOSHIDA1999
    status, explanation = entry.pk_route_status()
    assert status == "no-dosing-recorded"
    assert "dose" in explanation
    # The route/species pair itself IS covered - that is the whole point.
    drug_pk.fentanyl_rat_pk("IV", 0.2375)
    assert isinstance(entry.effect, be.UnavailableEffect)


def test_gao_is_blocked_because_no_mouse_haloperidol_pk_exists():
    entry = be.HALOPERIDOL_HYPERLOCOMOTION_GAO2021
    status, explanation = entry.pk_route_status()
    assert status == "no-species"
    assert "haloperidol" in explanation and "mouse" in explanation
    # And the route would be unsourced even in the species that has kinetics.
    with pytest.raises(drug_pk.UnsourcedParameterError):
        drug_pk.haloperidol_rat_pk("PO", 0.25)
    # Track E's second objection, unchanged by Phase 1: an ED50 is not a
    # group-means target.
    assert entry.effect.kind == "ed50"
    assert entry.effect.values == {"haloperidol": 0.11}


def test_pellow_is_blocked_by_a_missing_dose_and_route():
    entry = be.DIAZEPAM_EPM_PELLOW1985
    assert entry.route is None and entry.doses_mg_per_kg is None
    status, _ = entry.pk_route_status()
    assert status == "no-dosing-recorded"
    with pytest.raises(drug_pk.UnsourcedParameterError):
        entry.pk_model(0.25)


def test_every_entry_has_a_bridge_verdict_and_a_matching_status():
    """P2.2's version of this test read `pk_route_status()` alone and required
    every `available` entry to say "Bridge available and exercised". That was
    right while both available entries were diazepam, for which the PK step
    working meant the whole chain worked. P7.3 added three fentanyl/rat/IV
    entries, whose PK step `drug_pk` DOES build and whose plasma-to-brain step
    it reports as `total-brain-only`, so "available" no longer implies
    "exercised end to end" and requiring that phrase of them would force three
    verdicts to overstate their own coverage. The check is therefore read off
    BOTH computed statuses rather than relaxed: an entry whose receptor step
    fails has to name that failure, and no available entry may call the bridge
    unavailable.
    """
    assert set(be.BRIDGE_VERDICT) == set(be.BEHAVIORAL_EVIDENCE_BY_KEY)
    for key, verdict in be.BRIDGE_VERDICT.items():
        entry = be.BEHAVIORAL_EVIDENCE_BY_KEY[key]
        status, _ = entry.pk_route_status()
        if status == "available":
            drive, _ = entry.receptor_drive_status()
            # One required phrase per receptor-step status, so a verdict
            # cannot claim more of the chain than `drug_pk` supports.
            required = {
                "receptor-drivable": ("Bridge available and exercised",),
                "total-brain-only": ("total-brain-only",),
                "none": ("no sourced brain partition",),
            }[drive]
            # ... and none of them may claim less of it, either.
            assert "Bridge unavailable" not in verdict, key
        else:
            # One required phrase per PK status, so a verdict cannot drift
            # away from the coverage `drug_pk` actually reports.
            required = {
                # The pair may still be covered; what is missing is the
                # paper's own dosing, so the verdict has to name the dataset
                # as the gap.
                "no-dosing-recorded": ("dataset not obtainable",
                                       "Bridge inapplicable"),
                "unsourced": ("Bridge unavailable",),
                "no-species": ("Bridge unavailable",),
            }[status]
        assert any(phrase in verdict for phrase in required), (key, status)


def test_exactly_one_entry_is_calibration_ready_and_it_is_the_graziani_one():
    """P2.1's version of this test asserted `calibration_ready_entries() ==
    ()` and said that a later pass making an entry usable "must also update
    this test, deliberately". P2.2 is that pass, and this is that update.

    The expectation P2.1 encoded was right when it was written and is wrong
    now for a reason outside the code: [Graziani2022] was found, and it
    satisfies all three readiness conditions (the bridge reaches it, it
    carries a published effect with error bars, and the measure's
    operational definition is the assay's own primary variable rather than
    an undefined composite). The test is tightened rather than merely
    relaxed - it now names the entry, so an accidental fourth entry
    slipping into the fit still fails here.
    """
    ready = be.calibration_ready_entries()
    # P7.3 update, in the deliberate spirit this docstring asks for: three
    # more entries now satisfy all three readiness conditions, and all three
    # are diazepam/rat/IP because that remains the only triple with a
    # receptor-drivable brain partition. Two of them are what P7.3 was
    # looking for and P2.2 did not have - GRADED series rather than
    # two-point contrasts - and one ([Sagara2008]) is the only one in the
    # whole module. The list is named rather than counted so a fifth entry
    # cannot drift into a fit unnoticed.
    assert [e.key for e in ready] == [
        "diazepam/elevated-plus-maze/Graziani2022",
        "diazepam/open-field/Graziani2022",
        "diazepam/locomotor-activity/Sagara2008",
        "diazepam/hot-plate/Lustyk2023",
    ]
    # The three entries that are NOT ready are still not ready, each for its
    # own recorded reason; P2.2 loosened no criterion to get here.
    assert be.DIAZEPAM_EPM_STICKLING2024 not in ready  # measure undefined in source
    assert be.FENTANYL_NAC_DOPAMINE_YOSHIDA1999 not in ready  # no effect size
    assert be.HALOPERIDOL_HYPERLOCOMOTION_GAO2021 not in ready  # no bridge, ED50


def test_graziani_table_is_internally_consistent_so_its_units_are_not_assumed():
    """The source's Table 2 prints no unit on its EPM columns. Rather than
    assume seconds and a 300 s trial, check the table against itself: open +
    closed + centre time must be the stated 5 min trial, and open + closed
    entries must be the printed total. Both rows pass, in both respects,
    which is what licenses `open_arm_avoidance_fraction` to divide by
    `GRAZIANI2022_EPM_TRIAL_DURATION_S`."""
    values = be.DIAZEPAM_EPM_GRAZIANI2022.effect.values
    duration = be.GRAZIANI2022_EPM_TRIAL_DURATION_S
    assert duration == 300.0
    # Time rows: the module stores only the open-arm column, so the closed
    # and centre columns are quoted here from the same table row.
    for prefix, closed_s, centre_s in (
        ("vehicle", 197.0, 75.0),
        ("diazepam 2 mg/kg", 147.4, 63.5),
    ):
        total_s = values["%s open_arm_time_s" % prefix] + closed_s + centre_s
        assert abs(total_s - duration) < 0.05 * duration, prefix
    # Entry rows, against the printed totals.
    for prefix, closed_entries in (("vehicle", 5.8), ("diazepam 2 mg/kg", 6.4)):
        summed = values["%s open_arm_entries" % prefix] + closed_entries
        assert abs(summed - values["%s total_entries" % prefix]) <= 0.15


def test_anxiolysis_ratio_is_computed_from_the_sources_own_numbers():
    """No number in the P2.2 likelihood is hand-entered: both the ratio and
    its width fall out of [Graziani2022]'s printed means and SEMs."""
    effect = be.DIAZEPAM_EPM_GRAZIANI2022.effect
    duration = be.GRAZIANI2022_EPM_TRIAL_DURATION_S

    vehicle_mean, vehicle_sd = be.open_arm_avoidance_fraction(effect, "vehicle", duration)
    drug_mean, drug_sd = be.open_arm_avoidance_fraction(
        effect, "diazepam 2 mg/kg", duration
    )
    assert vehicle_mean == pytest.approx(1.0 - 30.7 / 300.0)
    assert drug_mean == pytest.approx(1.0 - 89.0 / 300.0)
    assert vehicle_sd == pytest.approx(6.4 / 300.0)
    assert drug_sd == pytest.approx(13.0 / 300.0)

    ratio, sd = be.anxiolysis_ratio(effect, "vehicle", "diazepam 2 mg/kg", duration)
    assert ratio == pytest.approx(drug_mean / vehicle_mean)
    assert ratio < 1.0  # diazepam is anxiolytic in this dataset
    # Delta-method width, and it is a real width: the effect is about 4 sd
    # from "no effect", which is why the gain comes out identified.
    assert sd == pytest.approx(
        ratio * np.sqrt((drug_sd / drug_mean) ** 2 + (vehicle_sd / vehicle_mean) ** 2)
    )
    assert (1.0 - ratio) / sd > 3.0


def test_composite_anxiety_index_is_a_sensitivity_check_and_carries_no_width():
    """The composite index mixes a time term and an entry term that share
    the open-arm entry count, so its uncertainty needs a covariance the
    source does not print. It must therefore return a point estimate only,
    and must be refused as a likelihood target."""
    effect = be.DIAZEPAM_EPM_GRAZIANI2022.effect
    duration = be.GRAZIANI2022_EPM_TRIAL_DURATION_S
    vehicle = be.epm_composite_anxiety_index(effect, "vehicle", duration)
    drug = be.epm_composite_anxiety_index(effect, "diazepam 2 mg/kg", duration)
    assert not isinstance(vehicle, tuple)
    assert 0.0 < drug < vehicle < 1.0
    with pytest.raises(ValueError):
        be.anxiolysis_ratio(
            effect, "vehicle", "diazepam 2 mg/kg", duration,
            index=be.epm_composite_anxiety_index,
        )


def test_the_superseded_zero_of_600_claim_is_corrected_in_place():
    """P2.1 reported "Zero of 600" for the query [Graziani2022] answers. The
    record must say so where a reader will meet it, not only in
    docs/architecture.md."""
    record = be.SEARCHED_AND_EMPTY_BEHAVIORAL[
        ("diazepam", "rat", "IP", "elevated plus maze open-arm time")
    ]
    assert "SUPERSEDED" in record
    assert "Zero of 600" in record
    assert "Graziani2022" in record


def test_searched_and_empty_records_exist_for_all_three_systems():
    keys = set(be.SEARCHED_AND_EMPTY_BEHAVIORAL)
    drugs = {k[0] for k in keys}
    assert drugs == {"fentanyl", "haloperidol", "diazepam"}
    for key, record in be.SEARCHED_AND_EMPTY_BEHAVIORAL.items():
        assert "Searched" in record, key
        assert len(record) > 400, key


def test_no_cross_drug_substitution_is_present():
    """Rule 2, asserted structurally: every entry's stored effect is about
    the same drug the entry names. Track E found real numbers for morphine
    and for chlordiazepoxide and refused to use them for fentanyl and
    diazepam; this test makes that refusal impossible to undo quietly."""
    forbidden = {
        "fentanyl": ("morphine", "buprenorphine", "heroin"),
        "diazepam": ("chlordiazepoxide", "midazolam", "flumazenil"),
        "haloperidol": ("risperidone", "clozapine", "chlorpromazine"),
    }
    for entry in be.BEHAVIORAL_EVIDENCE:
        text = " ".join(
            [entry.endpoint, entry.effect.measure]
            + list(getattr(entry.effect, "values", {}))
        ).lower()
        for other in forbidden.get(entry.drug, ()):
            assert other not in text, (entry.key, other)


def test_pk_source_body_weight_band_is_recorded_for_the_absolute_model():
    """[Yassen2008] reports absolute mL and mL/min rather than per-kg values,
    so a fentanyl dose's concentration DOES depend on the weight chosen. The
    band the estimates came from is the defensible choice and is recorded."""
    band = be.PK_SOURCE_BODY_WEIGHT_BANDS_KG[("fentanyl", "rat", "IV")]
    assert band == drug_pk.FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG
    assert band[0] < band[1]
