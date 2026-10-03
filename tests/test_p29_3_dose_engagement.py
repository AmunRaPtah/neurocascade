"""P29.3's searched-and-empty result must remain complete and explicit."""

import json
from pathlib import Path


DOC = Path(__file__).parents[1] / "docs" / "p29-3-dose-engagement.md"
SPLIT = Path(__file__).parents[1] / "src" / "neurocascade" / "benchmark" / "split.json"


def _census_rows():
    lines = DOC.read_text().splitlines()
    start = lines.index(
        "| drug | stratum | D2 Ki | rat Kp,uu | P29.2 CL | human fu | PET D2 | missing layer / reason not run |"
    ) + 2
    rows = []
    for line in lines[start:]:
        if not line.startswith("| "):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def test_p29_3_census_covers_exactly_the_frozen_heldout_split():
    heldout = set(json.loads(SPLIT.read_text())["heldout"])
    rows = _census_rows()
    assert len(rows) == 55
    assert len({row[0] for row in rows}) == len(rows)
    assert {row[0] for row in rows} == heldout
    assert all(len(row) == 8 for row in rows)


def test_p29_3_records_both_scores_as_searches_with_no_fabricated_value():
    text = DOC.read_text()
    assert "predicted target-engagement dose vs clinical dose | 0 | not estimable | searched-and-empty" in text
    assert "predicted target-engagement dose vs PET-anchored dose | 0 | not estimable | searched-and-empty" in text
    assert "setting\n`F = 1`" in text
    assert "unreferenced 70 kg" in text
