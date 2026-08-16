"""P9.2: stiff-solver benchmark coverage and output contracts."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_stiff_solver.py"
RESULTS = ROOT / "results" / "benchmarks"


def test_benchmark_script_declares_all_solvers_and_resolutions():
    namespace = {"__file__": str(SCRIPT), "__name__": "benchmark_test"}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    assert namespace["METHODS"] == ("RK45", "BDF", "Radau")
    assert namespace["RESOLUTIONS"] == (
        "regions", "subdivisions", "major_divisions"
    )


def test_committed_stiff_benchmark_has_accuracy_rows_for_every_case():
    rows = list(csv.DictReader(
        (RESULTS / "p9_2_stiff_solver.csv").open(newline="")
    ))
    assert len(rows) == 18
    assert {
        (r["resolution"], r["workload"], r["method"])
        for r in rows
    } == {
        (resolution, workload, method)
        for resolution in ("regions", "subdivisions", "major_divisions")
        for workload in ("baseline", "active")
        for method in ("RK45", "BDF", "Radau")
    }
    for row in rows:
        assert float(row["seconds"]) > 0.0
        assert float(row["max_abs_error_vs_rk45"]) < 1e-6
        assert row["sparse_jacobian"] == "True"


def test_benchmark_metadata_records_tolerances_and_methods():
    payload = json.loads((RESULTS / "p9_2_stiff_solver.json").read_text())
    assert payload["metadata"]["methods"] == ["RK45", "BDF", "Radau"]
    assert payload["metadata"]["rtol"] == 1e-8
    assert payload["metadata"]["atol"] == 1e-11
