"""P34.6: the approved-drug universe is exhausted for the frozen candidate.

Checks results/P34.6.json is internally consistent: the best-case positive
count per liability (every P34.4 labelled positive plus every newly scorable
drug) is below the power floor, the scorable drugs are new (not v2, not the
fresh pool), and the candidate's receptors are the ones counted.
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
            >= result["n_remaining_with_placebo_ar_table"] >= len(result["scorable_drugs"]))
    assert sum(result["placebo_table_by_atc2"].values()) == result["n_remaining_with_placebo_ar_table"]


def test_receptors_are_the_frozen_candidates():
    assert set(LIABILITY_RECEPTORS) == {"eps", "sedation", "anticholinergic", "orthostatic_hypotension"}


def test_per_liability_follows_from_the_scorable_drugs(result):
    for L, genes in LIABILITY_RECEPTORS.items():
        n = sum(bool(set(d["anchors"]) & set(genes)) for d in result["scorable_drugs"])
        assert result["max_scorable_per_liability"][L] == n


def test_no_liability_can_reach_the_floor(result):
    p344 = json.loads((ROOT / "results" / "P34.4.json").read_text())
    floor = result["power_floor"]["positives"]
    for L, n in result["max_scorable_per_liability"].items():
        best = p344["labelled_counts_upper_bound_on_common_set"][L]["labelled_pos"] + n
        assert result["max_positives_if_widened"][L] == best < floor, L
    assert result["result"] == "EMPTY" and result["inputs_only"] is True


def test_scorable_drugs_are_new(result):
    v2 = {d["chembl_id"] for d in json.loads((V2 / "drugs.json").read_text())["drugs"]}
    pool = {m["chembl_id"] for m in json.loads((V2 / "truth_fresh" / "pool.json").read_text())["members"]}
    ids = {d["chembl_id"] for d in result["scorable_drugs"]}
    assert not ids & v2 and not ids & pool
