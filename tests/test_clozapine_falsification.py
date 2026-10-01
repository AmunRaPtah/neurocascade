"""P22.4: clozapine anomaly as a falsification test of occupancy-as-sole-mediator.

Pins every reported number and the fixed outcome-to-reading mapping. The numbers
are the EC50s (Gruender 2006, Uchida 2011), the clinically-effective plasma range
(AGNP consensus and three response-threshold studies), and the derived 65%/78%
concentrations; the mapping is the pre-registration's (docs/p22-preregistration.md).
"""
import math

import pytest

from neurocascade.human import clozapine_falsification as cf
from neurocascade.human.d2_ec50 import D2PlasmaEC50, d2_plasma_ec50, registered_d2_drugs


def test_clozapine_is_not_in_the_live_registry():
    # C1: registering it would retroactively change the committed P22.2 (V8) and
    # P22.3 clozapine records, and clozapine is the suite's unknown-drug fixture.
    assert "clozapine" not in registered_d2_drugs()
    with pytest.raises(ValueError, match="no human D2 plasma EC50 is registered for 'clozapine'"):
        d2_plasma_ec50("clozapine")


def test_ec50_records_are_schema_validated_and_striatal_primary():
    assert isinstance(cf.CLOZAPINE_D2_EC50_PRIMARY, D2PlasmaEC50)
    assert cf.CLOZAPINE_D2_EC50_PRIMARY is cf.CLOZAPINE_D2_EC50_PUTAMEN
    assert cf.CLOZAPINE_D2_EC50_PRIMARY.region == "putamen"
    assert cf.CLOZAPINE_D2_EC50_PRIMARY.drug == "clozapine"
    assert cf.CLOZAPINE_D2_EC50_PRIMARY.tracer == "[18F]fallypride"


def test_molar_mass_is_the_cited_label_value():
    assert cf.CLOZAPINE_MW_G_PER_MOL == 326.82


@pytest.mark.parametrize(
    "record,ng_per_mL,emax",
    [
        ("CLOZAPINE_D2_EC50_PUTAMEN", 950.0, 1.0),
        ("CLOZAPINE_D2_EC50_CAUDATE", 582.0, 1.0),
        ("CLOZAPINE_D2_EC50_CORTEX", 333.0, 1.0),
        ("CLOZAPINE_D2_EC50_UCHIDA_FREE", 105.0, 0.60),
        ("CLOZAPINE_D2_EC50_UCHIDA_FIXED", 483.0, 1.0),
    ],
)
def test_registered_ec50_values(record, ng_per_mL, emax):
    row = getattr(cf, record)
    assert row.ec50_ng_per_mL == pytest.approx(ng_per_mL, abs=1e-6)
    assert row.emax == emax


def test_therapeutic_reference_range_is_the_agnp_consensus():
    assert cf.THERAPEUTIC_REFERENCE_RANGE == (350.0, 600.0)
    agnp = cf.CLOZAPINE_EFFECTIVE_PLASMA[0]
    assert "28910830" in agnp.source
    assert (agnp.low_ng_per_mL, agnp.high_ng_per_mL) == (350.0, 600.0)


def test_every_effective_concentration_is_sourced_and_below_c65():
    # All four clinically-effective concentrations sit below the primary row's
    # 65% concentration: that gap is the anomaly.
    c65 = cf.plasma_for_occupancy(cf.CLOZAPINE_D2_EC50_PRIMARY, cf.TARGET_LOW)
    for row in cf.CLOZAPINE_EFFECTIVE_PLASMA:
        assert row.source.strip()
        assert row.high_ng_per_mL < c65


def test_plasma_for_occupancy_inverts_the_hyperbola():
    row = cf.CLOZAPINE_D2_EC50_PUTAMEN
    assert cf.plasma_for_occupancy(row, 0.65) == pytest.approx(1764.2857, abs=1e-3)
    assert cf.plasma_for_occupancy(row, 0.78) == pytest.approx(3368.1818, abs=1e-3)
    # unreachable: target at or above Emax
    assert cf.plasma_for_occupancy(cf.CLOZAPINE_D2_EC50_UCHIDA_FREE, 0.65) is None


def test_primary_result_is_outcome_A():
    result = cf.clozapine_falsification()
    assert result.region == "putamen"
    assert result.c65_ng_per_mL == pytest.approx(1764.2857, abs=1e-3)
    assert result.c78_ng_per_mL == pytest.approx(3368.1818, abs=1e-3)
    assert result.effective_high_ng_per_mL == 600.0
    assert result.fold_c65_over_effective_high == pytest.approx(2.9405, abs=1e-3)
    assert result.outcome == "A"
    assert "falsifies occupancy-threshold-as-sole-mediator" in result.reading
    assert not result.is_extrastriatal


def test_occupancy_across_the_effective_range_is_well_below_the_threshold():
    result = cf.clozapine_falsification()
    # At the clinically-effective plasma window the model predicts only ~27-39%
    # striatal occupancy, consistent with Gruender's measured 36% putamen.
    assert result.occupancy_at_effective_low == pytest.approx(0.2692, abs=1e-3)
    assert result.occupancy_at_effective_high == pytest.approx(0.3871, abs=1e-3)
    assert result.occupancy_at_effective_high < cf.TARGET_LOW


def test_label_pk_corroborates_the_anomaly_without_a_pk_model():
    pk = cf.VERSACLOZ_STEADY_STATE_PK
    row = cf.CLOZAPINE_D2_EC50_PUTAMEN
    # Even the highest observed steady-state peak across the whole label dose
    # range sits below 65% occupancy.
    occ_at_max_peak = row.emax * pk.peak_high_ng_per_mL / (
        pk.peak_high_ng_per_mL + row.ec50_ng_per_mL
    )
    assert occ_at_max_peak == pytest.approx(0.4322, abs=1e-3)
    assert occ_at_max_peak < cf.TARGET_LOW


def test_every_striatal_candidate_gives_outcome_A():
    for result in cf.sensitivity_results():
        if result.is_extrastriatal:
            continue
        assert result.outcome == "A", result.region


def test_uchida_free_emax_makes_65pct_unreachable():
    result = cf.clozapine_falsification(cf.CLOZAPINE_D2_EC50_UCHIDA_FREE)
    assert result.c65_ng_per_mL is None
    assert result.outcome == "A"
    assert "unreachable" in result.reading


def test_cortex_is_flagged_extrastriatal_and_never_decides_the_headline():
    result = cf.clozapine_falsification(cf.CLOZAPINE_D2_EC50_CORTEX)
    assert result.is_extrastriatal
    assert "EXTRASTRIATAL" in result.reading
    # It is the one candidate near the threshold rather than far above it: C65
    # just exceeds the top of the range (Gruender's extrastriatal thesis).
    assert result.c65_ng_per_mL == pytest.approx(618.4286, abs=1e-3)


def test_dose_domain_is_reported_as_a_pk_layer_refusal():
    refusal = cf.dose_domain_refusal()
    assert refusal.startswith("pk:")
    assert "searched-and-empty" in refusal
    assert "300" in refusal and "900" in refusal


def test_nothing_is_tuned_the_outcome_holds_at_a_wider_effective_range():
    # Robustness, not a tuning knob: even if the effective range top is pushed to
    # the highest observed clinical plasma (723 ng/mL), the primary stays A.
    result = cf.clozapine_falsification(effective_range=(350.0, 723.0))
    assert result.outcome == "A"
