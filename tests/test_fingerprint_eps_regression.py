"""P23.3: the pre-registered fingerprint-vs-EPS regression and its verdict.

These tests pin the sourced inputs, the occupancy predictions, the two-model
comparison and -- the point of the task -- that the outcome is reported as
INDETERMINATE with its achievable power, not as a confirmed finding.
"""
import math

import numpy as np
import pytest

from neurocascade.human import fingerprint_eps_regression as m
from neurocascade.human import offtarget_occupancy


# --- denominator -----------------------------------------------------------

def test_denominator_is_lako_eight_minus_clozapine():
    """Seven drugs: the Lako-2013 eight minus clozapine (no ED95, no EPS OR)."""
    assert set(m.DRUGS) == set(m.LAKO_2013) - {"clozapine"}
    assert len(m.DRUGS) == 7
    assert "clozapine" not in m.DRUGS
    # clozapine really is excluded on both grounds, not by hand:
    assert "clozapine" not in m.ED95_MG_LEUCHT_2020
    assert "clozapine" not in m.SIAFIS_2023_MAX_OR
    # ...but its Lako formula is still carried, so the exclusion is visible.
    assert m.LAKO_2013["clozapine"] == (61.7, 125.0)


# --- sourced constants -----------------------------------------------------

def test_lako_formulas_verbatim():
    assert m.LAKO_2013["haloperidol"] == (91.9, 0.65)
    assert m.LAKO_2013["amisulpride"] == (85.0, 137.0)
    assert m.LAKO_2013["quetiapine"] == (49.1, 352.0)
    assert m.LAKO_2013["risperidone"] == (92.4, 1.07)


def test_ed95_and_or_verbatim():
    assert m.ED95_MG_LEUCHT_2020["haloperidol"] == 6.3
    assert m.ED95_MG_LEUCHT_2020["amisulpride"] == 537.0
    assert m.SIAFIS_2023_MAX_OR["haloperidol"][0] == 7.56
    assert m.SIAFIS_2023_MAX_OR["quetiapine"][0] == 1.11


# --- occupancy predictions -------------------------------------------------

def test_d2_occupancy_matches_hand_computation():
    # haloperidol: 91.9 * 6.3 / (6.3 + 0.65) / 100
    assert m.d2_occupancy_at_ed95("haloperidol") == pytest.approx(0.8333, abs=1e-3)
    assert m.d2_occupancy_at_ed95("quetiapine") == pytest.approx(0.2838, abs=1e-3)
    # all seven D2 occupancies are plausible fractions
    for d in m.DRUGS:
        assert 0.0 < m.d2_occupancy_at_ed95(d) < 1.0


def test_5ht2a_occupancy_is_ki_derived_and_ordered():
    # atypicals saturate 5-HT2A; typicals / amisulpride do not
    assert m.fivehtt2a_occupancy_at_ed95("risperidone") > 0.9
    assert m.fivehtt2a_occupancy_at_ed95("ziprasidone") > 0.9
    assert m.fivehtt2a_occupancy_at_ed95("amisulpride") < 0.01
    assert m.fivehtt2a_occupancy_at_ed95("haloperidol") < 0.3


def test_5ht2a_quantity_is_labelled_weaker_everywhere():
    """P23.2 acceptance: the Ki-derived quantity carries its warning label."""
    assert m.KI_DERIVED_LABEL is offtarget_occupancy.KI_DERIVED_LABEL
    for row in m.build_dataset():
        assert row.quantity_kind_5ht2a == "ki-derived-occupancy"
        assert row.label_5ht2a == m.KI_DERIVED_LABEL


def test_occupancy_ratio_orders_typicals_below_atypicals():
    r = {row.drug: row.occ_ratio for row in m.build_dataset()}
    # haloperidol and amisulpride (typical / D2-only) have the smallest ratio
    assert r["amisulpride"] < r["haloperidol"] < r["risperidone"]
    assert r["haloperidol"] < 0.3
    assert r["risperidone"] > 1.0


# --- the regression and its pre-registered verdict -------------------------

def test_model_fit_values():
    r = m.fit_models()
    assert r.n == 7
    assert r.f_df == (1, 4)
    # adding the fingerprint ratio roughly doubles R^2
    assert r.r2_null == pytest.approx(0.3136, abs=2e-3)
    assert r.r2_full == pytest.approx(0.6357, abs=2e-3)
    # the ratio term carries the hypothesised (negative) sign
    assert r.beta_ratio < 0
    assert r.beta_ratio == pytest.approx(-0.3965, abs=5e-3)
    # ...but the improvement is NOT significant at alpha = 0.05
    assert r.p_value == pytest.approx(0.1331, abs=3e-3)
    assert r.p_value > 0.05


def test_verdict_is_indeterminate_not_supported():
    """The whole point: a trend consistent with H1 that the data cannot confirm."""
    r = m.fit_models()
    assert r.verdict == "indeterminate"
    assert r.verdict != "supported"
    assert r.verdict != "null-favored"


def test_power_is_too_low_to_find_even_a_large_effect():
    r = m.fit_models()
    # power to detect a conventionally large increment (f^2 = 0.35) is ~0.20
    assert r.power(0.35) < 0.80
    assert r.power(0.35) == pytest.approx(0.202, abs=0.03)
    # even at the (very large) observed effect, power is well under 0.8
    assert r.power(r.increment_f2) < 0.5
    # the n that would be needed exceeds the drugs that exist (only 8 Lako drugs)
    assert m.fit_models().n_for_power(r.increment_f2) == 13


def test_sign_stable_but_significance_fragile_under_leave_one_out():
    """No single-drug deletion rescues significance: still indeterminate."""
    loo = m.leave_one_drug_out()
    assert set(loo) == set(m.DRUGS)
    for d, res in loo.items():
        assert res.verdict == "indeterminate", d
        assert res.beta_ratio < 0, d
        assert res.p_value > 0.05, d


def test_ln_or_outcome_regenerated_from_source():
    rows = {row.drug: row for row in m.build_dataset()}
    assert rows["haloperidol"].ln_or == pytest.approx(math.log(7.56), abs=1e-9)
    assert rows["quetiapine"].ln_or == pytest.approx(math.log(1.11), abs=1e-9)
