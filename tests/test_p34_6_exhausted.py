"""P34.6: can the fresh set be widened with the frozen candidate?

Checks results/P34.6.json is internally consistent: the counted receptors are
the frozen candidate's, Ki came from the unchanged v2_binding build (all three
sources), the best case per liability (every P34.4 labelled positive plus every
newly scorable drug as a positive) follows from the listed drugs, the result
follows from the floor, and the new drugs are new (not v2, not the fresh pool).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurocascade.benchmark.v2.candidate import LIABILITY_RECEPTORS

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"


@pytest.fixture(scope="module")
def result():
    return json.loads((ROOT / "results" / "P34.6.json").read_text())


def test_counts_are_nested(result):
    assert result["n_remaining"] == (result["n_approved_parents"] - result["n_p33_1_candidates"]
                                     - result["n_p34_4_pool"])
    assert (result["n_remaining"] >= result["n_remaining_with_rx_label"]
            >= result["n_remaining_with_placebo_ar_table"])
    assert sum(result["placebo_table_by_atc2"].values()) == result["n_remaining_with_placebo_ar_table"]
    assert result["inputs_only"] is True
    assert "v2_binding.py unchanged" in result["ki_source"]


def test_receptors_are_the_frozen_candidates(result):
    assert {L: tuple(v["receptors"]) for L, v in result["per_liability"].items()} == LIABILITY_RECEPTORS


def test_best_case_follows_from_the_listed_drugs(result):
    p344 = json.loads((ROOT / "results" / "P34.4.json").read_text())
    floor = result["power_floor"]
    for L, v in result["per_liability"].items():
        assert v["new_with_ki"] == len(v["new_drugs"])
        assert v["new_with_ki_and_placebo_table"] == sum(d["placebo_table"] for d in v["new_drugs"])
        assert all(set(d["genes"]) <= set(v["receptors"]) and d["genes"] for d in v["new_drugs"])
        assert v["max_positives"] == (p344["labelled_counts_upper_bound_on_common_set"][L]["labelled_pos"]
                                      + v["new_with_ki"])
        assert v["max_scorable"] == v["new_with_ki"] + v["fresh_with_ki"]
        assert v["could_reach_floor"] == (v["max_positives"] >= floor["positives"]
                                          and v["max_scorable"] >= floor["positives"] + floor["negatives"])


def test_result_follows_from_the_floor(result):
    feasible = sorted(L for L, v in result["per_liability"].items() if v["could_reach_floor"])
    assert result["feasible_liabilities"] == feasible
    assert result["result"] == ("WIDENING_FEASIBLE" if feasible else "EMPTY")


def test_new_drugs_are_new(result):
    v2 = {d["chembl_id"] for d in json.loads((V2 / "drugs.json").read_text())["drugs"]}
    pool = {m["chembl_id"] for m in json.loads((V2 / "truth_fresh" / "pool.json").read_text())["members"]}
    ids = {d["chembl_id"] for v in result["per_liability"].values() for d in v["new_drugs"]}
    assert not ids & v2 and not ids & pool
