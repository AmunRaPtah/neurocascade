"""Print the P33.3 held-out table in the host_eval pytest tail.

Only active with NC_HOST_EVAL=1 (the host's frozen-truth checkout); in a
session checkout held-out truth is absent and nothing is printed.
"""

from __future__ import annotations

import os


def pytest_terminal_summary(terminalreporter):
    if os.environ.get("NC_HOST_EVAL") != "1":
        return
    from neurocascade.benchmark.v2.chain_benchmark import heldout_summary_lines, run_chain_benchmark

    terminalreporter.write_sep("=", "P33.3 held-out per-stage table")
    for line in heldout_summary_lines(run_chain_benchmark()):
        terminalreporter.write_line(line)
