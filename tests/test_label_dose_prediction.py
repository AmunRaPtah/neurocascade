"""P22.2a: the predicted label-dose windows, committed before any label is read.

Every number pinned here is a prediction of the dose-to-occupancy chain. None of
them was compared with an FDA label range when it was written: the label ranges
are transcribed in a later commit, which is the ordering the P22.1
pre-registration requires. See tasks/specs/P22.2a.md for the sources of every
constant.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from neurocascade.human.d2_ec50 import (
    D2_PLASMA_EC50,
    HALOPERIDOL_D2_PLASMA_EC50,
    OLANZAPINE_D2_PLASMA_EC50,
    OLANZAPINE_MW_G_PER_MOL,
    d2_plasma_ec50,
    registered_d2_drugs,
)
from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.kp_uu import kp_uu_records
from neurocascade.human.label_dose_prediction import (
    DENOMINATOR,
    NOT_BLIND,
    TARGET_HIGH,
    TARGET_LOW,
    PredictedWindow,
    predict_window,
    prediction_table,
    refusal_layers,
)
from neurocascade.human.label_dose_scoring import round_sig
from neurocascade.human.occupancy_validation import (
    DOSE_SEARCH_MAX_MG_PER_DAY,
    DOSE_SEARCH_MIN_MG_PER_DAY,
    OccupancyUnreachableError,
    dose_reaching_occupancy,
    steady_state_occupancy,
)
from neurocascade.human.receptor_parameters import d2_occupancy_parameter_from_ec50


@pytest.fixture(scope="module")
def table():
    return {window.drug: window for window in prediction_table()}


# ---------------------------------------------------------------------------
# The pre-registered frame
# ---------------------------------------------------------------------------


def test_denominator_is_the_pre_registered_eight_in_order():
    assert DENOMINATOR == (
        "haloperidol",
        "risperidone",
        "paliperidone",
        "olanzapine",
        "quetiapine",
        "ziprasidone",
        "aripiprazole",
        "clozapine",
    )
    assert TARGET_LOW == 0.65
    assert TARGET_HIGH == 0.78
    assert NOT_BLIND == ("haloperidol",)


def test_a_drug_outside_the_denominator_is_refused():
    with pytest.raises(ValueError, match="denominator"):
        predict_window("amisulpride")


# ---------------------------------------------------------------------------
# The olanzapine EC50 row
# ---------------------------------------------------------------------------


def test_the_registry_holds_haloperidol_and_olanzapine_and_nothing_else():
    assert registered_d2_drugs() == ("haloperidol", "olanzapine")
    assert set(D2_PLASMA_EC50) == {"haloperidol", "olanzapine"}
    assert d2_plasma_ec50("olanzapine") is OLANZAPINE_D2_PLASMA_EC50
    assert d2_plasma_ec50("haloperidol") is HALOPERIDOL_D2_PLASMA_EC50


def test_olanzapine_row_is_de_greef_2011_table_iii_with_its_own_emax():
    row = OLANZAPINE_D2_PLASMA_EC50
    assert OLANZAPINE_MW_G_PER_MOL == 312.44
    assert row.molar_mass_g_per_mol == 312.44
    assert row.ec50_nM == pytest.approx(5.29 / 312.44 * 1000.0, rel=1e-12)
    assert row.ec50_ng_per_mL == pytest.approx(5.29, rel=1e-12)
    assert row.emax == 0.875
    assert row.emax_convention == "fitted-free"
    assert row.plasma_basis == "total"
    assert row.analyte == "parent olanzapine"
    assert row.analyte_is_summed is False
    assert row.region is None
    assert row.tracer == "[11C]raclopride"
    assert row.provenance == "primary"
    assert row.read_in is None
    assert row.plasma_free_fraction is None
    assert row.interval_low_nM is None and row.interval_high_nM is None
    assert "de Greef 2011" in row.citation
    assert "PMC3032087" in row.citation
    assert "Table III" in row.citation
    assert "NDA020592" in row.notes
    assert "312.44" in row.notes


def test_olanzapine_has_a_not_required_kp_uu_record():
    records = kp_uu_records("olanzapine")
    assert len(records) == 1
    record = records[0]
    assert record.kp_uu is None
    assert record.measurement_kind == "absent"
    assert record.status == "not-required"
    assert "de Greef 2011" in record.citation


def test_occupancy_parameter_built_from_a_row_borrows_nothing():
    row = OLANZAPINE_D2_PLASMA_EC50
    record = d2_occupancy_parameter_from_ec50(row)
    assert record.target == "D2"
    assert record.species == "human"
    assert record.ki_nM is None
    assert record.efficacy is None
    assert record.status == "in-vivo-ec50-transcribed"
    assert record.ligand == "olanzapine"
    assert record.ec50_in_vivo_nM == row.ec50_nM
    assert record.ec50_plasma_basis == "total"
    assert record.ec50_origin == "primary"
    assert record.ec50_citation == row.citation
    assert record.ec50_record is row
    assert record.occupancy_constant_nM() == row.ec50_nM
    # usable also demands an efficacy, which no source gave for olanzapine.
    assert record.usable is False


# ---------------------------------------------------------------------------
# The chain, driven for olanzapine
# ---------------------------------------------------------------------------


def test_the_chain_drives_olanzapine_from_the_bridged_fda_set():
    times = np.arange(0.0, 24.0 * 60.0 + 5.0, 5.0)
    prediction = predict_occupancy(10.0, drug="olanzapine", times_min=times)
    assert prediction.drug == "olanzapine"
    assert prediction.ec50_nM == OLANZAPINE_D2_PLASMA_EC50.ec50_nM
    assert prediction.emax == 0.875
    assert prediction.provenance == "measured"
    assert prediction.layer("occupancy").provenance == "measured"
    assert prediction.layer("pk").status == "fda-population-pk-bridged"
    assert "NDA 213378" in prediction.layer("pk").citation
    assert "de Greef 2011" in prediction.layer("occupancy").citation
    expected = 0.875 * prediction.plasma_nM / (prediction.plasma_nM + prediction.ec50_nM)
    np.testing.assert_allclose(prediction.occupancy, expected, rtol=1e-14, atol=0.0)
    assert prediction.peak_occupancy < 0.875


def test_haloperidol_keeps_its_fitted_in_repo_occupancy_layer():
    prediction = predict_occupancy(2.0)
    assert prediction.emax == 1.0
    assert prediction.layer("occupancy").provenance == "fitted-in-repo"


def test_steady_state_superposition_carries_the_rows_emax():
    result = steady_state_occupancy(10.0, drug="olanzapine")
    assert result.emax == 0.875
    expected = 0.875 * result.plasma_nM / (result.plasma_nM + result.ec50_nM)
    np.testing.assert_allclose(result.occupancy, expected, rtol=1e-14, atol=0.0)
    haloperidol = steady_state_occupancy(2.0)
    assert haloperidol.emax == 1.0


def test_olanzapine_steady_state_exposure_matches_the_reviews_reference_subject():
    """LYBALVI review (NDA 213378), Figures 15 and 16: the reference subject on
    10 mg once daily reaches Cmax,ss 31.7 ng/mL and AUCtau 635 ng.h/mL. Same fit
    as the parameters, so this checks the bridge, not the model."""
    result = steady_state_occupancy(10.0, drug="olanzapine")
    ng_per_ml = result.plasma_nM * OLANZAPINE_MW_G_PER_MOL / 1000.0
    assert ng_per_ml.max() == pytest.approx(31.7, rel=0.05)
    assert ng_per_ml.mean() * 24.0 == pytest.approx(635.0, rel=0.05)
    # Exact identity for a linear model at steady state: AUCtau = dose / (CL/F).
    assert ng_per_ml.mean() * 24.0 == pytest.approx(10.0 / 15.5 * 1000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Unreachable targets are named, not brentq sign errors
# ---------------------------------------------------------------------------


def test_the_search_bracket_is_named():
    assert DOSE_SEARCH_MIN_MG_PER_DAY == 0.01
    assert DOSE_SEARCH_MAX_MG_PER_DAY == 500.0
    assert issubclass(OccupancyUnreachableError, ValueError)


def test_a_target_above_the_fits_emax_is_unreachable_at_any_dose():
    capped = d2_occupancy_parameter_from_ec50(
        replace(OLANZAPINE_D2_PLASMA_EC50, emax=0.7)
    )
    with pytest.raises(OccupancyUnreachableError, match="at any dose"):
        dose_reaching_occupancy(0.78, drug="olanzapine", receptor_record=capped)


def test_a_reachable_target_beyond_the_bracket_says_so():
    with pytest.raises(OccupancyUnreachableError, match="dose bracket"):
        dose_reaching_occupancy(0.9999, drug="haloperidol")


# ---------------------------------------------------------------------------
# The prediction table
# ---------------------------------------------------------------------------


PREDICTED = {
    # drug: (D65, D78) in mg/day, steady-state once-daily mean occupancy.
    "haloperidol": (2.8922543, 5.5643311),
    "olanzapine": (5.7688673, 16.4444018),
}

REFUSED = {
    "risperidone": ("active-metabolite", "pk", "occupancy"),
    "paliperidone": ("pk", "occupancy"),
    "quetiapine": ("pk", "occupancy"),
    "ziprasidone": ("pk", "occupancy"),
    "aripiprazole": ("active-metabolite", "pk", "occupancy"),
    "clozapine": ("pk", "occupancy"),
}


def test_the_table_covers_the_denominator_in_order():
    windows = prediction_table()
    assert tuple(window.drug for window in windows) == DENOMINATOR
    assert all(isinstance(window, PredictedWindow) for window in windows)
    assert set(PREDICTED) | set(REFUSED) == set(DENOMINATOR)


@pytest.mark.parametrize("drug", sorted(PREDICTED))
def test_predicted_windows(table, drug):
    window = table[drug]
    d65, d78 = PREDICTED[drug]
    assert window.scored is True
    assert window.refused_layers == ()
    assert window.reasons == ()
    assert window.d65_mg_per_day == pytest.approx(d65, rel=1e-4)
    assert window.d78_mg_per_day == pytest.approx(d78, rel=1e-4)
    assert window.blind is (drug != "haloperidol")


def test_printed_windows_at_three_significant_figures(table):
    printed = {
        drug: (round_sig(table[drug].d65_mg_per_day), round_sig(table[drug].d78_mg_per_day))
        for drug in PREDICTED
    }
    assert printed == {"haloperidol": (2.89, 5.56), "olanzapine": (5.77, 16.4)}


def test_haloperidol_window_is_the_published_reference_point(table):
    """P19.5/P19.6 printed 2.89 to 5.56 mg/day; the P22.2 brief restates it."""
    window = table["haloperidol"]
    assert window.d65_mg_per_day == dose_reaching_occupancy(0.65)
    assert window.d78_mg_per_day == dose_reaching_occupancy(0.78)
    assert window.blind is False


@pytest.mark.parametrize("drug", sorted(REFUSED))
def test_refused_drugs_name_every_refusing_layer(table, drug):
    window = table[drug]
    assert window.scored is False
    assert window.d65_mg_per_day is None
    assert window.d78_mg_per_day is None
    assert window.refused_layers == REFUSED[drug]
    assert len(window.reasons) == len(window.refused_layers)
    assert all(reason.strip() for reason in window.reasons)
    assert window.blind is True
    assert tuple(layer for layer, _ in refusal_layers(drug)) == REFUSED[drug]


def test_pk_refusals_name_the_fda_application_that_could_not_supply_them(table):
    def pk_reason(drug):
        window = table[drug]
        return window.reasons[window.refused_layers.index("pk")]

    assert "NDA 210655" in pk_reason("risperidone")
    assert "parent-and-metabolite" in pk_reason("risperidone")
    assert "NDA 021999" in pk_reason("paliperidone")
    assert "redacted" in pk_reason("paliperidone")
    assert "NDA 022047" in pk_reason("quetiapine")
    assert "not-fitted" in pk_reason("quetiapine")
    assert "no FDA population-PK row exists for oral ziprasidone" in pk_reason("ziprasidone")
    assert "NDA 021436" in pk_reason("aripiprazole")
    assert "NDA 216655" in pk_reason("aripiprazole")
    assert "NDA 203479" in pk_reason("clozapine")


def test_the_active_metabolite_refusals_name_the_metabolite(table):
    risperidone = table["risperidone"]
    aripiprazole = table["aripiprazole"]
    assert "9-hydroxyrisperidone" in risperidone.reasons[0]
    assert "dehydro-aripiprazole" in aripiprazole.reasons[0]


def test_occupancy_refusals_point_at_the_reachability_record(table):
    for drug in REFUSED:
        window = table[drug]
        reason = window.reasons[window.refused_layers.index("occupancy")]
        assert "p21-2" in reason


def test_drivable_drugs_have_no_refusal():
    assert refusal_layers("haloperidol") == ()
    assert refusal_layers("olanzapine") == ()


def test_the_olanzapine_window_is_below_the_fits_emax_and_finite(table):
    window = table["olanzapine"]
    assert 0.0 < window.d65_mg_per_day < window.d78_mg_per_day < DOSE_SEARCH_MAX_MG_PER_DAY
    at_d78 = steady_state_occupancy(window.d78_mg_per_day, drug="olanzapine")
    assert at_d78.mean_occupancy == pytest.approx(0.78, abs=1e-6)
    assert math.isfinite(window.d78_mg_per_day)
