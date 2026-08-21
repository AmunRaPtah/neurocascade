"""P9.1: benchmark whole-brain construction and steady-state evaluation.

Run from the repository root with::

    PYTHONPATH=src python scripts/benchmark_whole_brain.py

The benchmark deliberately measures the public whole-brain builder rather
than a synthetic array operation. It records construction, drug-free steady
state, and one active steady-state evaluation across all three supported
parcellation resolutions and three receptor-system counts. Timings are
machine-specific baselines for P9.2/P9.3, not scientific parameters.
"""

import argparse
import cProfile
import csv
import datetime as dt
import json
import os
import platform
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.calibration import heldout_forward as hf  # noqa: E402
from neurocascade.legacy.connectome import build as connectome_build  # noqa: E402


RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "results", "benchmarks"
)

# The sets are cumulative so the count comparison changes one thing at a time.
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


def _model_size(model):
    weights = np.asarray(model.network.connectivity)
    return {
        "nodes": int(model.network.n),
        "edges_nonzero": int(np.count_nonzero(weights)),
        "connectivity_bytes": int(weights.nbytes),
        "layers_built": int(len(model.layers)),
        "density_entries": int(sum(len(layer.densities) for layer in model.layers)),
    }


def benchmark_case(resolution, target_count, targets):
    (model, provenance), build_seconds = _timed(
        lambda: hf.build_multi_target_model(
            targets, "p9.1-%s-%d" % (resolution, target_count),
            resolution=resolution,
        )
    )
    size = _model_size(model)

    _, baseline_seconds = _timed(lambda: model.baseline_rates)
    x = {target: 0.1 for target in targets}
    _, active_seconds = _timed(lambda: model.rates(x))

    return {
        "resolution": resolution,
        "requested_target_count": target_count,
        "requested_targets": ",".join(targets),
        "source": provenance["connectome_source"],
        "nodes": size["nodes"],
        "edges_nonzero": size["edges_nonzero"],
        "connectivity_bytes": size["connectivity_bytes"],
        "layers_built": size["layers_built"],
        "density_entries": size["density_entries"],
        "weight_scale": provenance["weight_scale"],
        "build_seconds": build_seconds,
        "baseline_seconds": baseline_seconds,
        "active_steady_state_seconds": active_seconds,
        "total_seconds": build_seconds + baseline_seconds + active_seconds,
    }


def profile_representative(path, resolution="regions"):
    """Profile one construction plus active solve and write human-readable text."""

    def workload():
        model, _ = hf.build_multi_target_model(
            TARGET_SETS[5], "p9.1-profile", resolution=resolution
        )
        model.baseline_rates
        model.rates({target: 0.1 for target in TARGET_SETS[5]})

    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.runcall(workload)
    elapsed = time.perf_counter() - started
    with open(path, "w") as handle:
        handle.write("P9.1 whole-brain representative profile\n")
        handle.write("resolution: %s\n" % resolution)
        handle.write("elapsed_seconds: %.6f\n\n" % elapsed)
        handle.write("Top 30 functions by cumulative time\n")
        handle.write("=====================================\n")
        stats = pstats.Stats(profiler, stream=handle)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
    return elapsed


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
        for target_count, targets in TARGET_SETS.items():
            print("benchmarking %s / %d targets" % (resolution, target_count),
                  flush=True)
            rows.append(benchmark_case(resolution, target_count, targets))

    profile_path = os.path.join(output_dir, "p9_1_profile.txt")
    profile_seconds = profile_representative(profile_path)
    csv_path = os.path.join(output_dir, "p9_1_benchmark.csv")
    write_csv(rows, csv_path)

    metadata = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "resolutions": list(RESOLUTIONS),
        "target_sets": {str(k): list(v) for k, v in TARGET_SETS.items()},
        "profile_resolution": "regions",
        "profile_seconds": profile_seconds,
        "notes": [
            "Timings are machine-specific baselines, not model parameters.",
            "Each case includes public model construction, baseline steady state, "
            "and one active steady-state evaluation.",
            "The cached published Knox connectome and Allen expression atlas are "
            "used when available; source is recorded per row.",
        ],
    }
    with open(os.path.join(output_dir, "p9_1_benchmark.json"), "w") as handle:
        json.dump({"metadata": metadata, "rows": rows}, handle, indent=2)
        handle.write("\n")
    return rows, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=RESULTS)
    args = parser.parse_args()
    rows, metadata = run(args.output_dir)
    print("wrote %d benchmark rows" % len(rows))
    print("profile seconds: %.3f" % metadata["profile_seconds"])


if __name__ == "__main__":
    main()
