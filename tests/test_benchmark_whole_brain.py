"""P9.1: benchmark coverage and output contracts."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_whole_brain.py"
RESULTS = ROOT / "results" / "benchmarks"


def test_benchmark_script_covers_all_resolutions_and_target_counts():
    namespace = {"__file__": str(SCRIPT), "__name__": "benchmark_test"}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    assert namespace["RESOLUTIONS"] == (
        "regions", "subdivisions", "major_divisions"
    )
    assert sorted(namespace["TARGET_SETS"]) == [1, 3, 5]
    assert [len(namespace["TARGET_SETS"][n]) for n in (1, 3, 5)] == [1, 3, 5]


def test_committed_benchmark_has_one_row_per_resolution_and_target_count():
    rows = list(csv.DictReader(
        (RESULTS / "p9_1_benchmark.csv").open(newline="")
    ))
    assert len(rows) == 9
    assert {(r["resolution"], int(r["requested_target_count"])) for r in rows} == {
        (resolution, count)
        for resolution in ("regions", "subdivisions", "major_divisions")
        for count in (1, 3, 5)
    }
    for row in rows:
        assert float(row["build_seconds"]) > 0.0
        assert int(row["nodes"]) > 0
        assert int(row["edges_nonzero"]) > 0


def test_json_metadata_and_profile_are_committed():
    payload = json.loads((RESULTS / "p9_1_benchmark.json").read_text())
    assert len(payload["rows"]) == 9
    assert payload["metadata"]["profile_resolution"] == "regions"
    profile = (RESULTS / "p9_1_profile.txt").read_text()
    assert "Top 30 functions by cumulative time" in profile
    assert "steady_state_rates" in profile
