"""P9.3: benchmark the vectorized RHS and direct fixed-point solver.

Run from the repository root with::

    PYTHONPATH=src python scripts/benchmark_steady_state.py

The workload matches P9.1: one, three and five target layers at every
connectome resolution. The integration path now uses the vectorized RHS;
the root path solves the same fixed-point equations with the analytic
Jacobian. Accuracy is compared to the vectorized long-integration result and
timings are compared to the committed P9.1 baseline.
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


ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results", "benchmarks")
TARGET_SETS = {
    1: ("D2",),
    3: ("D2", "GABA_A_BZD", "MOR"),
    5: ("D2", "GABA_A_BZD", "MOR", "M1", "NMDA"),
}
RESOLUTIONS = tuple(connectome_build.RESOLUTIONS)


def _timed(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def _p9_1_timings():
    path = os.path.join(RESULTS, "p9_1_benchmark.csv")
    with open(path, newline="") as handle:
        return {
            (row["resolution"], int(row["requested_target_count"]), workload): float(row[field])
            for row in csv.DictReader(handle)
            for workload, field in (
                ("baseline", "baseline_seconds"),
                ("active", "active_steady_state_seconds"),
            )
        }


def benchmark_case(resolution, target_count, targets, p9_1):
    model, provenance = hf.build_multi_target_model(
        targets, "p9.3-%s-%d" % (resolution, target_count),
        resolution=resolution,
    )
    active_input = {target: 0.1 for target in targets}
    reference_baseline = model.baseline_rates
    reference_active = model.rates(active_input, solver="integration")
    rows = []

    for method_name, baseline_call, active_call in (
        (
            "integration-vectorized",
            lambda: cfwd.steady_state_rates(model.network),
            lambda: model.rates(active_input, solver="integration"),
        ),
        (
            "root-jacobian",
            lambda: cfwd.steady_state_rates(model.network, solver="root"),
            lambda: model.rates(active_input, solver="root"),
        ),
    ):
        baseline, baseline_seconds = _timed(baseline_call)
        active, active_seconds = _timed(active_call)
        for workload, rates, reference, seconds in (
            ("baseline", baseline, reference_baseline, baseline_seconds),
            ("active", active, reference_active, active_seconds),
        ):
            p9_1_seconds = p9_1[(resolution, target_count, workload)]
            rows.append({
                "resolution": resolution,
                "nodes": int(model.network.n),
                "edges_nonzero": int(np.count_nonzero(model.network.connectivity)),
                "requested_target_count": target_count,
                "requested_targets": ",".join(targets),
                "workload": workload,
                "solver": method_name,
                "seconds": seconds,
                "p9_1_seconds": p9_1_seconds,
                "speedup_vs_p9_1": p9_1_seconds / seconds,
                "max_abs_error_vs_integration": float(
                    np.max(np.abs(rates - reference))
                ),
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
    p9_1 = _p9_1_timings()
    rows = []
    for resolution in RESOLUTIONS:
        for target_count, targets in TARGET_SETS.items():
            print("benchmarking %s / %d targets" % (resolution, target_count),
                  flush=True)
            rows.extend(benchmark_case(
                resolution, target_count, targets, p9_1
            ))

    csv_path = os.path.join(output_dir, "p9_3_steady_state.csv")
    write_csv(rows, csv_path)
    metadata = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "resolutions": list(RESOLUTIONS),
        "target_sets": {str(k): list(v) for k, v in TARGET_SETS.items()},
        "rtol": cfwd._STEADY_STATE_RTOL,
        "atol": cfwd._STEADY_STATE_ATOL,
        "notes": [
            "P9.1 is the committed pre-vectorization integration baseline.",
            "Integration-vectorized uses the same solve_ivp horizon and tolerances.",
            "Root-jacobian uses scipy.optimize.root(hybr) and the analytic circuit Jacobian.",
            "The active input is x = 0.1 for every requested target.",
            "Timings are machine-specific and include one steady-state solve only.",
        ],
    }
    with open(os.path.join(output_dir, "p9_3_steady_state.json"), "w") as handle:
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
