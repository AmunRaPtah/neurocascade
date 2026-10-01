"""P22.3: leave-one-drug-out against published PET occupancy."""
import pytest

from neurocascade.human.d2_ec50 import D2_PLASMA_EC50
from neurocascade.human.leave_one_drug_out import (
    DENOMINATOR,
    HELD_OUT_EC50,
    OBSERVATIONS,
    SEARCHED_AND_EMPTY,
    HeldOutObservation,
    drug_result,
    leave_one_drug_out_table,
    predict,
    unrecorded_counts,
)


@pytest.fixture(scope="module")
def table():
    return {row.drug: row for row in leave_one_drug_out_table()}


def test_every_denominator_drug_appears_once_in_order():
    assert tuple(r.drug for r in leave_one_drug_out_table()) == DENOMINATOR
    assert len(DENOMINATOR) == 9


def test_each_drug_is_exactly_one_of_scored_or_searched_and_empty(table):
    # C2: no silent drops, and unscored is never a pass.
    for drug, row in table.items():
        assert (drug in SEARCHED_AND_EMPTY) != bool(row.scored)
        if row.status == "searched-and-empty":
            assert row.passed is None and row.mae is None
            assert row.empty.searched
    assert {d for d, r in table.items() if r.status == "scored"} == {
        "aripiprazole",
        "ziprasidone",
    }


def test_held_out_ec50_rows_do_not_enter_the_chain_registry():
    # Adding them would rewrite P22.2's pinned refusal attribution.
    assert set(D2_PLASMA_EC50) == {"haloperidol", "olanzapine"}
    assert not set(HELD_OUT_EC50) & set(D2_PLASMA_EC50)


def test_ec50_rows_carry_their_sourced_values():
    ari = HELD_OUT_EC50["aripiprazole"]
    assert ari.ec50_ng_per_mL == pytest.approx(11.1, abs=1e-12)
    assert ari.emax == 1.0 and ari.tracer == "[11C]raclopride"
    zip_ = HELD_OUT_EC50["ziprasidone"]
    assert zip_.ec50_nM == 204.0
    assert zip_.ec50_ng_per_mL == pytest.approx(84.24, abs=0.01)


def test_tracer_and_region_recorded_on_every_observation():
    assert unrecorded_counts() == {"tracer": 0, "region": 0, "total": 7}
    for obs in OBSERVATIONS:
        assert obs.tracer and obs.region


def test_aripiprazole_result(table):
    row = table["aripiprazole"]
    errors = [round(s.error, 4) for s in row.scored]
    assert errors == [-0.0505, -0.0305, -0.0068, 0.0332, 0.0028, 0.0228]
    assert row.n == 6 and row.n_independent_groups == 3
    assert row.cross_tracer is False
    assert row.mae == pytest.approx(0.024433, abs=1e-6)
    assert row.bias == pytest.approx(-0.004850, abs=1e-6)
    assert row.passed is True and row.anecdotal is False
    by_region = row.mae_by_region()
    assert by_region["caudate"] == pytest.approx(0.020035, abs=1e-6)
    assert by_region["putamen"] == pytest.approx(0.028830, abs=1e-6)
    assert [s.contained for s in row.scored] == [False, True, True, False, True, False]
    sd = [s.standardized_error for s in row.scored]
    assert sd[4] is None and sd[5] is None
    assert round(sd[0], 2) == -1.26


def test_ziprasidone_result(table):
    row = table["ziprasidone"]
    (s,) = row.scored
    assert s.predicted == pytest.approx(0.346166, abs=1e-6)
    assert s.error == pytest.approx(-0.423834, abs=1e-6)
    assert s.contained is None and s.standardized_error is None
    assert s.observation.regimen_matches_ec50_source is False
    assert row.passed is False and row.anecdotal is True
    assert row.whole_chain.startswith("refused")


def test_prediction_uses_the_drugs_own_constant_and_emax():
    obs = OBSERVATIONS[0]
    c = obs.plasma_ng_per_mL
    assert predict(obs) == pytest.approx(c / (c + 11.1), rel=1e-12)


def test_basis_mismatch_is_refused_not_converted():
    from dataclasses import replace

    with pytest.raises(ValueError, match="basis"):
        predict(replace(OBSERVATIONS[0], plasma_basis="free"))


def test_observation_without_disjointness_reason_is_refused():
    from dataclasses import replace

    with pytest.raises(ValueError, match="disjoint"):
        replace(OBSERVATIONS[0], disjoint_because="")


def test_unknown_drug_is_refused():
    with pytest.raises(ValueError):
        drug_result("asenapine")


def test_quetiapine_record_states_the_nord_2011_constant():
    assert "1369 nmol/l" in SEARCHED_AND_EMPTY["quetiapine"].ec50_source


def test_observation_type_rejects_non_fraction_occupancy():
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(OBSERVATIONS[0], observed_occupancy=57.0)
    assert isinstance(OBSERVATIONS[0], HeldOutObservation)
