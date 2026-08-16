"""P9.2: compare whole-brain steady-state solvers.

Run from the repository root with::

    PYTHONPATH=src python scripts/benchmark_stiff_solver.py

The benchmark compares RK45 with BDF and Radau on the public multi-target
whole-brain path at all supported resolutions. Accuracy is measured against
the RK45 result at the same tolerances; timings are machine-specific and are
recorded separately for the drug-free and active workloads.
"""

import argparse
import csv
import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.calibration import circuit_forward as cfwd  # noqa: E402
from neurocascade.calibration import heldout_forward as hf  # noqa: E402
from neurocascade.connectome import build as connectome_build  # noqa: E402


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "benchmarks")
TARGETS = ("D2", "GABA_A_BZD", "MOR", "M1", "NMDA")
METHODS = ("RK45", "BDF", "Radau")
RESOLUTIONS = tuple(connectome_build.RESOLUTIONS)


def _time_call(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def benchmark_resolution(resolution):
    model, provenance = hf.build_multi_target_model(
        TARGETS, "p9.2-%s" % resolution, resolution=resolution
    )
    active_input = {target: 0.1 for target in TARGETS}
    reference_baseline = model.baseline_rates
    reference_active = model.rates(active_input, method="RK45")
    rows = []

    for method in METHODS:
        baseline, baseline_seconds = _time_call(
            lambda: cfwd.steady_state_rates(
                model.network, method=method, use_sparse_jacobian=True
            )
        )
        active, active_seconds = _time_call(
            lambda: model.rates(
                active_input, method=method, use_sparse_jacobian=True
            )
        )
        for workload, rates, reference, seconds in (
            ("baseline", baseline, reference_baseline, baseline_seconds),
            ("active", active, reference_active, active_seconds),
        ):
            rows.append({
                "resolution": resolution,
                "nodes": int(model.network.n),
                "edges_nonzero": int(np.count_nonzero(model.network.connectivity)),
                "targets": ",".join(TARGETS),
                "workload": workload,
                "method": method,
                "sparse_jacobian": True,
                "seconds": seconds,
                "max_abs_error_vs_rk45": float(np.max(np.abs(rates - reference))),
                "max_rate": float(np.max(rates)),
                "connectome_source": provenance["connectome_source"],
            })
    return rows


def write_csv(rows, path):
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir=RESULTS):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for resolution in RESOLUTIONS:
        print("benchmarking %s" % resolution, flush=True)
        rows.extend(benchmark_resolution(resolution))

    csv_path = os.path.join(output_dir, "p9_2_stiff_solver.csv")
    write_csv(rows, csv_path)
    metadata = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "resolutions": list(RESOLUTIONS),
        "methods": list(METHODS),
        "targets": list(TARGETS),
        "rtol": cfwd._STEADY_STATE_RTOL,
        "atol": cfwd._STEADY_STATE_ATOL,
        "notes": [
            "RK45 is the accuracy reference, not a scientific ground truth.",
            "BDF and Radau receive the analytic CSR Jacobian and its sparsity pattern.",
            "The active workload uses x = 0.1 for each of five requested targets.",
            "Timings are machine-specific and include one steady-state solve only.",
        ],
    }
    with open(os.path.join(output_dir, "p9_2_stiff_solver.json"), "w") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, indent=2)
        handle.write("\n")
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=RESULTS)
    args = parser.parse_args()
    rows, _ = run(args.output_dir)
    print("wrote %d benchmark rows" % len(rows))


if __name__ == "__main__":
    main()
