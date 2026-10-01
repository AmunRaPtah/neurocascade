"""P22.5: the multi-drug failure analysis regenerates its numbers from code.

The document embeds the two rendered tables verbatim, so any change to an
upstream module that moves a number fails here until the document is
regenerated. The pinned values are the ones the ledger rows V5-V11 print.
"""
from pathlib import Path

import pytest

from neurocascade.human import multidrug_failure_analysis as mfa
from neurocascade.human.active_metabolites import ACTIVE_METABOLITES
from neurocascade.human.label_dose_result import LABEL_TRANSCRIPTIONS, label_dose_table
from neurocascade.human.label_dose_scoring import score

DOC = Path(__file__).resolve().parent.parent / "docs" / "p22-5-multidrug-failure-analysis.md"


@pytest.fixture(scope="module")
def numbers():
    return mfa.regenerated_numbers()


def test_document_embeds_the_regenerated_tables_verbatim():
    text = DOC.read_text()
    assert mfa.layer_grid_markdown() in text
    assert mfa.failure_matrix_markdown() in text


def test_grid_covers_the_nine_drug_census_once_each():
    grid = mfa.layer_grid()
    assert tuple(r.drug for r in grid) == (
        "haloperidol", "risperidone", "olanzapine", "clozapine", "ziprasidone",
        "amisulpride", "aripiprazole", "paliperidone", "quetiapine",
    )
    by_drug = {r.drug: r for r in grid}
    assert {d for d, r in by_drug.items() if r.pk == "accepted"} == {"haloperidol", "olanzapine"}
    assert by_drug["risperidone"].analyte == by_drug["aripiprazole"].analyte == "refused"
    assert by_drug["haloperidol"].analyte == "bounded (0.0030)"
    assert by_drug["paliperidone"].pk == "refused: oral row redacted"
    assert by_drug["clozapine"].outcome == "V11 Outcome A"
    assert by_drug["aripiprazole"].heldout_observations == "n=6, 3 groups"


def test_every_row_has_a_layer_and_a_data_requirement():
    rows = mfa.failure_matrix()
    assert len({r.layer for r in rows}) == len(rows)
    for row in rows:
        assert row.data_required and row.attribution and row.drugs
    with pytest.raises(ValueError):
        mfa.FailureRow("pk", ("x",), "e", "a", "")


def test_every_p22_2_refusal_layer_has_a_matrix_row(numbers):
    # P22.2 names its layers by chain gate; the matrix names them by what fails.
    matrix_name = {"active-metabolite": "analyte", "pk": "pk", "occupancy": "occupancy constant"}
    layers = {r.layer.split(":")[0] for r in mfa.failure_matrix(numbers)}
    for layer in set(numbers["label"]["first_refusal"].values()):
        assert matrix_name[layer] in layers


def test_pk_numbers(numbers):
    pk = numbers["pk"]
    assert pk["fda_rows"] == 23
    assert pk["fda_first_refusal"] == {
        "bridged": 1, "route": 14, "analyte": 1, "searched-and-empty": 6, "reference weight": 1,
    }
    assert pk["chain_pk_drugs"] == ("haloperidol", "olanzapine")
    assert pk["dose_occupancy_pairs"] == {"haloperidol": 0, "olanzapine": None}
    assert (pk["half_life_within_twofold"], pk["half_life_n"]) == (5, 13)
    assert pk["half_life_by_route"] == {"IM": (8, 1), "IV": (1, 0), "PO": (2, 2), "SC": (2, 2)}
    assert pk["half_life_worst_fold"] == pytest.approx(163.95, abs=0.01)


def test_kp_uu_gate_is_masked_by_the_pk_refusal(numbers):
    assert set(numbers["no_kp_uu_record"]) == set(numbers["census"]) - set(numbers["pk"]["chain_pk_drugs"])


def test_analyte_numbers(numbers):
    an = numbers["analyte"]
    assert an["risperidone_drive_ratio"] == pytest.approx(12.2056, abs=1e-4)
    assert an["risperidone_max_error"] == pytest.approx(0.5684, abs=1e-4)
    assert an["haloperidol_max_error"] == pytest.approx(0.0030, abs=1e-4)
    assert an["aripiprazole_ratio_fold"] == pytest.approx(2.5)
    assert an["aripiprazole_max_error_label_equipotent"] == pytest.approx(0.0839, abs=1e-4)
    assert an["aripiprazole_max_error_derived_equipotent"] == pytest.approx(0.1716, abs=1e-4)
    assert an["v6_predicted"] == pytest.approx(6.523, abs=1e-3)
    assert an["v6_fold"] == pytest.approx(1.695, abs=1e-3)


def test_closed_form_error_matches_the_record_method():
    for drug in ("risperidone", "haloperidol"):
        record = ACTIVE_METABOLITES[drug][0]
        assert mfa.max_occupancy_error(record.drive_ratio()) == pytest.approx(
            record.max_occupancy_error(), rel=1e-12
        )


def test_occupancy_numbers(numbers):
    oc = numbers["occupancy"]
    assert oc["split_b_mae"] == pytest.approx(0.0351, abs=1e-4)
    assert oc["split_d_mae"] == pytest.approx(0.0939, abs=1e-4)
    assert oc["aripiprazole_mae"] == pytest.approx(0.0244, abs=1e-4)
    assert oc["aripiprazole_groups"] == 3
    assert oc["ziprasidone_error"] == pytest.approx(-0.4238, abs=1e-4)
    assert oc["ziprasidone_implied_ec50_ng"] == pytest.approx(44.6 * 0.23 / 0.77)
    assert oc["ziprasidone_implied_fold"] == pytest.approx(6.32, abs=0.01)
    assert oc["constants_total"] == 9
    assert oc["constants_region_none"] == ("haloperidol", "olanzapine")
    assert oc["constants_tracer_none"] == ("haloperidol",)


def test_clozapine_headline_is_cross_tracer_and_same_tracer_rows_still_give_a(numbers):
    oc, cl = numbers["occupancy"], numbers["clinical"]
    assert oc["clozapine_primary_tracer"] == "[18F]fallypride"
    assert oc["threshold_tracer"] == "[11C]raclopride"
    (free, fixed) = oc["clozapine_same_tracer"]
    assert free == (105.0, 0.6, None, None, "A")
    assert fixed[0] == 483.0 and fixed[2] == pytest.approx(897.0)
    assert fixed[3] == pytest.approx(1.495) and fixed[4] == "A"
    assert oc["clozapine_emax_convention_spread"] == pytest.approx(4.6)
    assert cl["clozapine_c65"] == pytest.approx(1764.29, abs=0.01)
    assert cl["clozapine_fold"] == pytest.approx(2.9405, abs=1e-4)
    assert cl["clozapine_outcomes"] == ("A",) * 5


def test_label_numbers(numbers):
    lb = numbers["label"]
    assert (lb["n_pass"], lb["n_scored"], lb["denominator"]) == (2, 2, 8)
    assert lb["e"]["haloperidol"] == pytest.approx(0.0149, abs=1e-4)
    assert lb["e"]["olanzapine"] == pytest.approx(-0.1625, abs=1e-4)
    assert lb["ec50_pass_band"]["olanzapine"] == pytest.approx((0.726, 2.904), abs=1e-3)
    assert lb["ec50_pass_band"]["haloperidol"] == pytest.approx((0.483, 1.931), abs=1e-3)
    assert lb["olanzapine_d65_over_label_low"] == pytest.approx(0.577, abs=1e-3)
    assert lb["haloperidol_label_span"] == 15.0


@pytest.mark.parametrize("drug", sorted(LABEL_TRANSCRIPTIONS))
def test_pass_band_edges_are_where_the_scaled_window_stops_passing(drug, numbers):
    window = {row.drug: row.window for row in label_dose_table()}[drug]
    reading = LABEL_TRANSCRIPTIONS[drug].scoring_reading
    low, high = numbers["label"]["ec50_pass_band"][drug]

    def passes(k):
        return score(drug, k * window.d65_mg_per_day, k * window.d78_mg_per_day,
                     reading.l_low, reading.l_high).passed

    assert passes(low * 1.01) and passes(high * 0.99)
    assert not passes(low * 0.98) and not passes(high * 1.02)


def test_observation_and_structural_counts(numbers):
    assert numbers["observations"]["scored"] == ("ziprasidone", "aripiprazole")
    assert len(numbers["observations"]["searched_and_empty"]) == 7
    assert numbers["observations"]["sources_opened"] == 37
    assert numbers["ke0_illustrative"] == 4
    assert numbers["partial_agonists"] == ("aripiprazole",)
