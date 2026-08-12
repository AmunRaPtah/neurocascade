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

from neurocascade import drug_pk
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
    bridgeable = {e.key for e in be.bridgeable_entries()}
    assert bridgeable == {"diazepam/elevated-plus-maze/Stickling2024"}
    assert len(be.unbridgeable_entries()) == len(be.BEHAVIORAL_EVIDENCE) - 1


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
    assert set(be.BRIDGE_VERDICT) == set(be.BEHAVIORAL_EVIDENCE_BY_KEY)
    for key, verdict in be.BRIDGE_VERDICT.items():
        entry = be.BEHAVIORAL_EVIDENCE_BY_KEY[key]
        status, _ = entry.pk_route_status()
        # One required phrase per status, so a verdict cannot drift away from
        # the coverage `drug_pk` actually reports.
        required = {
            "available": ("Bridge available and exercised",),
            # The pair may still be covered; what is missing is the paper's
            # own dosing, so the verdict has to name the dataset as the gap.
            "no-dosing-recorded": ("dataset not obtainable", "Bridge inapplicable"),
            "unsourced": ("Bridge unavailable",),
            "no-species": ("Bridge unavailable",),
        }[status]
        assert any(phrase in verdict for phrase in required), (key, status)


def test_no_entry_is_calibration_ready_yet_and_that_is_recorded():
    """P2.1 delivers evidence, not a fit. If a later pass makes an entry
    usable it must also update this test, deliberately."""
    assert be.calibration_ready_entries() == ()


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
