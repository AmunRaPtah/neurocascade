"""P23.2: off-target occupancy from the same plasma curve that drives D2.

Pins the Ki-derived occupancy arithmetic, the distinct labelling that keeps a
Ki-derived occupancy from being read as the in-vivo-fitted D2 occupancy, the
measured floor that quantifies how weak the Ki-derived quantity is, and the one
directional cross-check against measured PET ordering.
"""
import types

import numpy as np
import pytest

from neurocascade.human import offtarget_occupancy as oo
from neurocascade.human import offtarget_ki
from neurocascade.human import dose_to_occupancy as dto
from neurocascade.human.d2_ec50 import HALOPERIDOL_D2_PLASMA_EC50_nM


# --- the hyperbola itself --------------------------------------------------

def test_ki_occupancy_is_the_plain_hyperbola():
    assert oo.ki_occupancy(5.0, 8.85) == pytest.approx(5.0 / 13.85)
    assert oo.ki_occupancy(0.0, 10.0) == 0.0


def test_ki_occupancy_clips_negative_concentration():
    # The plasma solver can undershoot zero by a rounding amount on the tail.
    assert oo.ki_occupancy(-1e-9, 10.0) == 0.0


def test_ki_occupancy_rejects_nonpositive_ki():
    with pytest.raises(ValueError):
        oo.ki_occupancy(5.0, 0.0)


def test_ki_occupancy_vectorises():
    out = oo.ki_occupancy(np.array([0.0, 5.0]), 8.85)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(5.0 / 13.85)


# --- the off-target band ---------------------------------------------------

def test_band_uses_full_fingerprint_span():
    ki_low, ki_high = offtarget_ki.human_ki_range_nM("haloperidol", "5HT2A")
    assert (ki_low, ki_high) == (25.0, 435.0)
    occ_low, occ_high = oo.offtarget_occupancy_band("haloperidol", "5HT2A", 5.0)
    # Largest Ki -> lowest occupancy.
    assert occ_low == pytest.approx(5.0 / 440.0)
    assert occ_high == pytest.approx(5.0 / 30.0)
    assert occ_low < occ_high


def test_band_is_monotonic_in_concentration():
    lo1, hi1 = oo.offtarget_occupancy_band("haloperidol", "D3", 1.0)
    lo2, hi2 = oo.offtarget_occupancy_band("haloperidol", "D3", 10.0)
    assert lo2 > lo1 and hi2 > hi1


# --- off-target occupancy from the shared plasma curve ---------------------

def test_from_curve_reuses_the_exact_plasma_curve():
    pred = dto.predict_occupancy(5.0)
    curves = oo.offtarget_occupancy_from_curve(pred)
    assert set(curves) == set(offtarget_ki.TARGETS)
    for curve in curves.values():
        # The same curve, not a re-integration.
        assert np.array_equal(curve.concentration_nM, np.asarray(pred.plasma_nM))
        assert np.array_equal(curve.times_min, np.asarray(pred.times_min))
        assert curve.concentration_basis == pred.receptor_compartment


def test_from_curve_occupancy_in_unit_interval():
    pred = dto.predict_occupancy(5.0)
    for curve in oo.offtarget_occupancy_from_curve(pred).values():
        assert np.all(curve.occ_low >= 0.0) and np.all(curve.occ_low <= 1.0)
        assert np.all(curve.occ_high >= curve.occ_low)
        assert np.all(curve.occ_high <= 1.0)


def test_from_curve_refuses_a_drug_without_a_fingerprint():
    # Olanzapine has a plasma curve but no P23.1 off-target fingerprint.
    fake = types.SimpleNamespace(drug="olanzapine")
    with pytest.raises(ValueError):
        oo.offtarget_occupancy_from_curve(fake)


# --- the distinct labelling (the second acceptance criterion) --------------

def test_ki_derived_is_labelled_distinctly_from_in_vivo():
    assert oo.QUANTITY_KIND_KI != oo.QUANTITY_KIND_INVIVO
    assert oo.KI_DERIVED_LABEL != oo.IN_VIVO_LABEL
    assert "WEAKER" in oo.KI_DERIVED_LABEL
    assert "in-vivo" in oo.IN_VIVO_LABEL


def test_every_offtarget_curve_carries_the_ki_label():
    pred = dto.predict_occupancy(5.0)
    for curve in oo.offtarget_occupancy_from_curve(pred).values():
        assert curve.quantity_kind == oo.QUANTITY_KIND_KI
        assert curve.label == oo.KI_DERIVED_LABEL
        assert "Ki-derived" in curve.summary()


# --- the measured floor on how weak the Ki-derived quantity is -------------

def test_d2_discrepancy_scalar_matches_hand_computation():
    d = oo.d2_invivo_vs_ki_discrepancy(5.0)
    assert d.ec50_nM == pytest.approx(HALOPERIDOL_D2_PLASMA_EC50_nM)
    assert d.ki_nM == 8.85
    assert d.ki_label == "bartuzi2022"
    assert d.occ_invivo == pytest.approx(5.0 / (5.0 + HALOPERIDOL_D2_PLASMA_EC50_nM))
    assert d.occ_ki == pytest.approx(5.0 / 13.85)
    assert d.difference == pytest.approx(d.occ_invivo - d.occ_ki)
    # In-vivo EC50 (1.885) below in-vitro Ki (8.85): Ki under-states occupancy.
    assert d.difference > 0.0


def test_d2_discrepancy_floor_is_large_over_the_clinical_curve():
    pred = dto.predict_occupancy(5.0)
    d = oo.d2_invivo_vs_ki_discrepancy(pred.plasma_nM)
    # The one target where both constants exist mispredicts by ~0.37 occupancy
    # units when its Ki is used as an EC50. That is the floor every off-target
    # inherits; it is a measured gap, not a chosen tolerance.
    assert 0.30 < d.max_abs_difference < 0.40


def test_d2_discrepancy_moves_with_the_chosen_ki():
    # Across the seven P19.4 in-vitro sources the floor must change, because the
    # ratio straddles unity: at the smallest Ki (0.13) the in-vivo EC50 is the
    # larger of the two and the sign of the gap flips.
    high_ki = oo.d2_invivo_vs_ki_discrepancy(5.0, ki_nM=8.85)
    low_ki = oo.d2_invivo_vs_ki_discrepancy(5.0, ki_nM=0.13)
    assert high_ki.difference > 0.0  # Ki under-states
    assert low_ki.difference < 0.0   # Ki over-states
    assert low_ki.ki_label == "caller-supplied"


# --- directional cross-check against measured PET ordering ------------------

def test_haloperidol_5ht2a_occupancy_sits_below_d2():
    # Measured signature: haloperidol is high striatal D2, low cortical 5-HT2
    # (Kapur & Remington 1996; Farde/Nyberg 1995 crossover for risperidone). The
    # Ki fingerprint must reproduce the SIGN of the 5-HT2A-minus-D2 difference.
    pred = dto.predict_occupancy(5.0)
    curves = oo.offtarget_occupancy_from_curve(pred)
    assert curves["5HT2A"].peak_occ_high < pred.peak_occupancy
