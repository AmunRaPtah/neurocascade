"""P9.3: steady-state benchmark coverage and output contracts."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_steady_state.py"
RESULTS = ROOT / "results" / "benchmarks"


def test_benchmark_script_covers_p91_workloads_and_two_solvers():
    namespace = {"__file__": str(SCRIPT), "__name__": "benchmark_test"}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    assert namespace["RESOLUTIONS"] == (
        "regions", "subdivisions", "major_divisions"
    )
    assert sorted(namespace["TARGET_SETS"]) == [1, 3, 5]


def test_committed_steady_state_benchmark_has_accuracy_rows_for_every_case():
    rows = list(csv.DictReader(
        (RESULTS / "p9_3_steady_state.csv").open(newline="")
    ))
    assert len(rows) == 36
    assert {
        (r["resolution"], int(r["requested_target_count"]), r["workload"], r["solver"])
        for r in rows
    } == {
        (resolution, count, workload, solver)
        for resolution in ("regions", "subdivisions", "major_divisions")
        for count in (1, 3, 5)
        for workload in ("baseline", "active")
        for solver in ("integration-vectorized", "root-jacobian")
    }
    for row in rows:
        assert float(row["seconds"]) > 0.0
        assert float(row["speedup_vs_p9_1"]) > 0.0
        assert float(row["max_abs_error_vs_integration"]) < 1e-6


def test_steady_state_benchmark_metadata_records_reference_tolerances():
    payload = json.loads((RESULTS / "p9_3_steady_state.json").read_text())
    assert payload["metadata"]["rtol"] == 1e-8
    assert payload["metadata"]["atol"] == 1e-11
    assert "P9.1" in payload["metadata"]["notes"][0]
