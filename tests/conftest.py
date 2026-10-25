import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print P30.3's held-out band coverage in a host_eval run.

    Only under NC_HOST_EVAL=1 (the host's frozen-truth checkout) and only when
    tests/test_conformal.py actually ran; a session checkout has no held-out
    truth and prints nothing.
    """
    if os.environ.get("NC_HOST_EVAL") != "1" or not getattr(config, "_nc_p30_3_ran", False):
        return
    from neurocascade.benchmark.conformal import heldout_coverage, heldout_summary_lines

    terminalreporter.write_sep("=", "P30.3 held-out conformal coverage (frozen bands)")
    for line in heldout_summary_lines(heldout_coverage()):
        terminalreporter.write_line(line)
