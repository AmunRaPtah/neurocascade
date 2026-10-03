"""P29.4 records the data result instead of building an empty predictor."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "p29-4-occupancy-time.md"
SPLIT = ROOT / "src" / "neurocascade" / "benchmark" / "split.json"


PET_DRUGS = {
    "amisulpride", "asenapine", "escitalopram", "lurasidone",
    "milnacipran", "paroxetine", "quetiapine", "venlafaxine",
}


def test_p29_4_names_all_and_only_the_heldout_pet_drugs():
    heldout = set(json.loads(SPLIT.read_text())["heldout"])
    rows = DOC.read_text().splitlines()
    table = rows[rows.index("| drug | PET target(s) | panel targets | first missing prediction layer |") + 2:]
    table = [line for line in table if line.startswith("|")]
    names = {line.strip("|").split("|")[0].strip() for line in table}
    assert names == PET_DRUGS
    assert names <= heldout
    assert len(table) == 8


def test_p29_4_does_not_claim_a_score_without_a_prediction_curve():
    text = DOC.read_text()
    assert "complete\nprediction denominator is zero" in text
    assert "No MAE or bias is reported per receptor" in text
    assert "P29.2 held-out CL error (n=5; GMFE 6.774746573244732)" in text
    assert "using its dose-response\nrelation" in text
