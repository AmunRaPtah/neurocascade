"""P25.2: literature population-PK records in the occupancy chain.

Pins which P25.1 records the bridge accepts and why each other one is refused,
the converted records' values against the P25.1 transcription, the widened
chain gate (one compartment accepted, three refused), the regression that
haloperidol and olanzapine predictions are unchanged, the frozen P22.2 status,
and the V14 Tmax comparison.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from neurocascade.human import dose_to_occupancy, label_dose_prediction
from neurocascade.human.dose_to_occupancy import (
    WeightScaling,
    _build_pk_model,
    _integrate,
    _require_pk_record,
    _scaled_parameters,
)
from neurocascade.human.literature_pk_bridge import (
    INPUT_SLOTS,
    bridge_survey,
    bridged_records,
    dropped_parameters,
    label_tmax,
    refusals,
    tmax_comparisons,
    tmax_table,
    to_human_pk_record,
)
from neurocascade.human.literature_popk import LITERATURE_POPPK_RECORDS, record
from neurocascade.human.pk_parameters import human_pk_records
from neurocascade.human.prediction_interface import predict, refusal_layers


def _reasons(drug, author):
    return " | ".join(refusals(record(drug, author)))


# ---------------------------------------------------------------------------
# which records convert
# ---------------------------------------------------------------------------


def test_exactly_zheng_and_knights_convert():
    assert [(r.drug, r.citation.split(".")[0][:5]) for r in bridged_records()] == [
        ("quetiapine", "Zheng"),
        ("aripiprazole", "Knigh"),
    ]
    assert bridge_survey().count("BRIDGED") == 2
    assert len(bridge_survey().splitlines()) == len(LITERATURE_POPPK_RECORDS)


def test_every_record_has_an_input_mapping():
    keys = {(r.drug, r.short_citation.split()[0]) for r in LITERATURE_POPPK_RECORDS}
    assert keys == set(INPUT_SLOTS)


@pytest.mark.parametrize(
    "drug, author, phrases",
    [
        ("clozapine", "Li", ("analyte is 'parent-and-metabolite'", "no reference weight")),
        ("clozapine", "Reeves", ("parent-and-metabolite", "parallel first-order rate constant (Km)",
                                 "36.80 h, is 3.07-fold")),
        ("quetiapine", "Chen", ("14.45 h, is 2.41-fold", "failed V13")),
        ("aripiprazole", "Koue", ("'L/h/kg'", "'L/kg'", "no reference weight")),
        ("risperidone", "Størset", ("parent-and-metabolite", "no reference weight")),
        ("paliperidone", "Korell", ("zero-order release phase (D1 = 25 h)",
                                    "no reference subject", "26.02-32.85 h")),
    ],
)
def test_each_refused_record_states_its_reasons(drug, author, phrases):
    reasons = _reasons(drug, author)
    for phrase in phrases:
        assert phrase in reasons
    with pytest.raises(ValueError, match="cannot drive the dose-to-occupancy chain"):
        to_human_pk_record(record(drug, author))


def test_chen_is_refused_only_for_its_half_life():
    # Structurally identical to Zheng 2024; V13 is the only thing between them.
    assert len(refusals(record("quetiapine", "Chen"))) == 1


def test_a_slot_printed_in_another_unit_is_refused_not_converted():
    knights = record("aripiprazole", "Knights")
    per_min = replace(knights.parameter("KA"), units="1/min")
    bad = replace(knights, parameters=tuple(
        per_min if p.name == "KA" else p for p in knights.parameters))
    assert any("ka ('KA') is printed in '1/min'" in r for r in refusals(bad))


# ---------------------------------------------------------------------------
# the converted records
# ---------------------------------------------------------------------------


def test_zheng_converts_to_the_transcribed_one_compartment_set():
    pk = to_human_pk_record(record("quetiapine", "Zheng"))
    assert pk.n_compartments == 1
    assert (pk.V1_L, pk.CL_L_per_h, pk.V2_L, pk.Q12_L_per_h) == (530.0, 76.1, None, None)
    assert pk.ka_per_min == 1.46 / 60.0
    assert pk.reference_weight_kg == 70.0
    assert pk.units == "absolute" and pk.apparent_over_F and pk.F is None
    assert pk.cl_allometric_exponent is None
    assert pk.status == "literature-population-pk-bridged"
    assert pk.complete_for_model
    assert "PMID 39635430" not in pk.citation and "PMID 39698209" in pk.citation
    assert "ka is FIXED" in pk.notes and "PMID 25975812" in pk.notes


def test_knights_converts_to_the_transcribed_two_compartment_set():
    pk = to_human_pk_record(record("aripiprazole", "Knights"))
    assert pk.n_compartments == 2
    assert (pk.V1_L, pk.V2_L, pk.Q12_L_per_h, pk.CL_L_per_h) == (192.0, 151.0, 12.2, 3.88)
    assert pk.ka_per_min == 0.515 / 60.0
    assert pk.reference_weight_kg == 74.19
    assert "ka estimated in this fit" in pk.notes
    dropped = dict(dropped_parameters(record("aripiprazole", "Knights")))
    assert set(dropped) == {"WT_VP", "AGE_VP", "WT_CL", "WT_Q", "AGE_CL",
                            "2D6PM_CL", "SEXF_Q", "SEXF_VP"}
    assert all("reference subject only" in why for why in dropped.values())


def test_literature_records_come_last_and_can_be_excluded():
    combined = human_pk_records()
    assert combined[-2:] == bridged_records()
    frozen = human_pk_records(include_literature=False)
    assert combined[: len(frozen)] == frozen
    assert {r.drug for r in frozen} & {"quetiapine", "aripiprazole"} == set()


# ---------------------------------------------------------------------------
# the widened chain gate
# ---------------------------------------------------------------------------


def test_chain_resolves_the_literature_records_by_drug():
    assert _require_pk_record("quetiapine", "PO", None).V1_L == 530.0
    assert _require_pk_record("aripiprazole", "PO", None).V1_L == 192.0


def test_chain_refuses_a_three_compartment_record_with_its_reason():
    halo = human_pk_records("haloperidol", "PO")[0]
    three = replace(halo, n_compartments=3, V3_L=1.0, Q13_L_per_h=1.0)
    with pytest.raises(ValueError, match="one- and two-compartment oral models only"):
        _require_pk_record("haloperidol", "PO", three)


def test_frozen_p22_modules_still_refuse_the_literature_drugs_at_pk():
    with pytest.raises(ValueError, match="no human quetiapine PK record"):
        _require_pk_record("quetiapine", "PO", None, include_literature=False)
    assert "pk" in [layer for layer, _ in label_dose_prediction.refusal_layers("quetiapine")]
    assert "pk" in [layer for layer, _ in label_dose_prediction.refusal_layers("aripiprazole")]


def test_the_interface_names_the_real_missing_layers():
    assert [layer for layer, _ in refusal_layers("quetiapine")] == ["occupancy", "exposure"]
    assert [layer for layer, _ in refusal_layers("aripiprazole")] == [
        "active-metabolite", "occupancy", "exposure"]


def test_one_compartment_chain_matches_the_bateman_function():
    pk = _require_pk_record("quetiapine", "PO", None)
    times = np.linspace(0.0, 48.0 * 60.0, 577)
    params = _scaled_parameters(pk, 70.0, None)[:4]
    plasma = _integrate(_build_pk_model(pk, 70.0, None, params), 100.0 / 70.0, times)
    ka, k = 1.46 / 60.0, (76.1 / 530.0) / 60.0
    expected = 100.0 * ka / (530.0 * (ka - k)) * (np.exp(-k * times) - np.exp(-ka * times))
    np.testing.assert_allclose(plasma, expected, rtol=1e-7, atol=1e-12)


def test_a_literature_record_cannot_be_moved_off_its_reference_weight():
    pk = _require_pk_record("aripiprazole", "PO", None)
    with pytest.raises(ValueError, match="states no weight-scaling rule"):
        _scaled_parameters(pk, 70.0, WeightScaling(1.0, 0.75, "test"))
    with pytest.raises(ValueError, match="needs a stated"):
        _scaled_parameters(pk, 70.0, None)


# ---------------------------------------------------------------------------
# regression: haloperidol and olanzapine unchanged
# ---------------------------------------------------------------------------

#: predict() outputs computed at base commit 3dfab55, before P25.2:
#: (drug, dose, regimen, peak, mean, trough, sum of plasma nM, plasma_nM[60]).
_BASELINE = (
    ("haloperidol", 2.0, "single-dose", 0.5252531905774322, 0.24772778951558894,
     0.0, 597.7473931736163, 1.9501398848390905),
    ("haloperidol", 2.0, "steady-state-daily", 0.6591850795781576, 0.5634924668356194,
     0.47175264657672916, 725.5956295879719, 3.448019674393783),
    ("olanzapine", 10.0, "single-dose", 0.6273581008463255, 0.45511335088328353,
     0.0, 18020.364685993205, 42.419693306548304),
    ("olanzapine", 10.0, "steady-state-daily", 0.7531006985491451, 0.7291350582857389,
     0.6973048650334256, 24778.63140180017, 102.32602696511977),
)


@pytest.mark.parametrize("drug, dose, regimen, peak, mean, trough, total, at60", _BASELINE)
def test_haloperidol_and_olanzapine_predictions_are_unchanged(
    drug, dose, regimen, peak, mean, trough, total, at60
):
    p = predict(drug, dose, regimen)
    assert p.body_weight_kg == 70.0
    got = (p.peak_occupancy, p.mean_occupancy, p.trough_occupancy,
           float(np.sum(p.plasma_nM)), float(p.plasma_nM[60]))
    assert got == pytest.approx((peak, mean, trough, total, at60), rel=1e-12, abs=0.0)


def test_default_weight_is_the_sources_reference_weight():
    # 70 kg for both P22-era drugs, which is why the default change is inert.
    pred = dose_to_occupancy.predict_occupancy(2.0, drug="haloperidol")
    assert pred.body_weight_kg == 70.0


# ---------------------------------------------------------------------------
# V14: Tmax against the innovator labels
# ---------------------------------------------------------------------------


def test_tmax_comparisons_match_v14():
    zheng, knights = tmax_comparisons()
    assert (zheng.drug, zheng.source, zheng.n_compartments, zheng.ka_fixed) == (
        "quetiapine", "Zheng 2024", 1, True)
    assert (knights.drug, knights.source, knights.n_compartments, knights.ka_fixed) == (
        "aripiprazole", "Knights 2015", 2, False)
    assert round(zheng.predicted_tmax_h, 2) == 1.77
    assert round(zheng.fold_error, 2) == 1.18
    assert round(knights.predicted_tmax_h, 2) == 4.55
    assert knights.fold_error == 1.0
    assert zheng.agrees_within_twofold and knights.agrees_within_twofold
    assert tmax_table().count("PASS") == 2


def test_one_compartment_tmax_agrees_with_the_closed_form():
    ka, k = 1.46, 76.1 / 530.0
    closed = math.log(ka / k) / (ka - k)
    assert abs(tmax_comparisons()[0].predicted_tmax_h - closed) <= 1.0 / 60.0


def test_label_tmax_quotes_carry_their_numbers():
    assert "1.5 hours" in label_tmax("quetiapine").quote
    assert "NDA 020639" in label_tmax("quetiapine").source
    assert "3 hours to 5 hours" in label_tmax("aripiprazole").quote
    assert "NDA 021436" in label_tmax("aripiprazole").source
    with pytest.raises(ValueError):
        label_tmax("clozapine")
