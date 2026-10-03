"""P29.5 evidence records: data coverage and split isolation."""

from pathlib import Path

from neurocascade.benchmark import load_split
from neurocascade.benchmark.p29_5 import (
    load_oral_exposure,
    load_pet_plasma_pairs,
    load_preclinical_oral_inputs,
)


ROOT = Path(__file__).parents[1]
PET_DRUGS = {
    "amisulpride", "asenapine", "escitalopram", "lurasidone",
    "milnacipran", "paroxetine", "quetiapine", "venlafaxine",
}


def test_oral_exposure_has_eight_same_entity_heldout_records():
    data = load_oral_exposure()
    split = set(load_split()["heldout"])
    rows = {row["inn"]: row for row in data["drugs"]
            if row["status"] == "data"}
    assert set(rows) == PET_DRUGS
    assert set(rows) <= split
    for row in rows.values():
        assert row["observations"]
        for obs in row["observations"]:
            assert obs["citation"].get("pmid") or obs["citation"].get("nda")
            assert obs["quote"]
            assert obs["route"] in ("oral", "swallowed oral", "sublingual")


def test_pet_pairs_keep_model_and_inequality_qualifiers_visible():
    data = load_pet_plasma_pairs()
    rows = {row["inn"]: row for row in data["drugs"]}
    assert {inn for inn, row in rows.items() if row["status"] == "data"} == {
        "amisulpride", "asenapine", "paroxetine", "quetiapine", "venlafaxine",
    }
    assert rows["asenapine"]["pairs"][0]["statistic"] == "target_concentration_model"
    assert rows["paroxetine"]["pairs"][0]["plasma_relation"] == ">="
    assert set(data["searched_and_empty"]) == {
        "escitalopram", "lurasidone", "milnacipran",
    }
    assert len(rows["quetiapine"]["pairs"]) == 4
    assert len(rows["venlafaxine"]["pairs"]) == 12


def test_preclinical_supplement_does_not_promote_heldout_truth_into_fitting():
    data = load_preclinical_oral_inputs()
    split = load_split()
    rows = {row["inn"]: row for row in data["drugs"]}
    assert set(rows) <= set(split["train"]) | set(split["heldout"])
    assert rows["lurasidone"]["status"] == "data"
    assert rows["amisulpride"]["status"] == "searched_and_empty"
    assert rows["quetiapine"]["status"] == "searched_and_empty"

    # The P29.5 loaders are evidence readers only. No fitting module may import
    # them or read the held-out truth directory.
    src = ROOT / "src" / "neurocascade"
    fitting_text = "\n".join(
        p.read_text() for p in src.rglob("*.py")
        if p.name != "p29_5.py" and "test" not in p.name
    )
    assert "p29_5" not in fitting_text
    assert "oral_exposure" not in fitting_text
    assert "load_oral_exposure(" not in fitting_text
    assert "load_pet_plasma_pairs(" not in fitting_text
    assert "load_preclinical_oral_inputs(" not in fitting_text


def test_p29_5_reports_the_unmet_acceptance_counts_instead_of_fabricating():
    data = load_preclinical_oral_inputs()
    train_data = [r for r in data["drugs"]
                  if r["split"] == "train" and r["status"] == "data"]
    pet = load_pet_plasma_pairs()
    measured_or_summary = [r for r in pet["drugs"] if r["status"] == "data"]
    assert data["coverage"]["train"] == len(train_data)
    assert data["coverage"]["train"] == 36
    assert data["coverage"]["heldout_pet"] == 5
    assert data["coverage"]["train"] < 0.60 * len(load_split()["train"])
    assert len(measured_or_summary) < 6
