"""Print the P33.3 held-out table, then P30.4's held-out band coverage when
its host_eval test ran, in the host_eval pytest tail.

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
    # P30.4's held-out band coverage goes last so it stays in the host's tail.
    lines = getattr(terminalreporter.config, "_nc_p30_4_lines", None)
    if lines:
        terminalreporter.write_sep("=", "P30.4 held-out band coverage (retrospective)")
        for line in lines:
            terminalreporter.write_line(line)
