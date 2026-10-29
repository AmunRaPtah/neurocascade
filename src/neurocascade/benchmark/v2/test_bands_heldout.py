"""P30.4 host_eval target: held-out coverage of the frozen candidate bands, once.

    python -m pytest -q -p no:cacheprovider -s src/neurocascade/benchmark/v2/test_bands_heldout.py

The coverage is retrospective: the v2 held-out split was already scored in
P33.4, so it cannot certify the bands.  In a session checkout the held-out
truth is absent and every held-out drug is named as missing truth; on the host
the same call scores the restored rows.  Under NC_HOST_EVAL=1 the result is
printed last in the terminal summary (v2/conftest.py).
"""

from __future__ import annotations

from neurocascade.benchmark.v2 import v2_partition
from neurocascade.benchmark.v2.candidate_bands import (
    PROVENANCE_LABEL,
    heldout_coverage,
    heldout_summary_lines,
    load_bands,
)


def test_heldout_coverage_once_on_frozen_bands(request):
    bands = load_bands()
    assert bands["heldout_read_at_freeze"] is False
    report = heldout_coverage(bands)
    request.config._nc_p30_4_lines = heldout_summary_lines(report)
    assert report["retrospective"] is True and report["certifies_bands"] is False
    assert report["provenance"] == PROVENANCE_LABEL
    heldout = set(v2_partition()[1])
    for name, r in report["quantities"].items():
        cov = r["train_mean"]["coverage"]["0.80"]
        bucketed = r["missing_truth"] + r["not_scoreable"]
        assert len(bucketed) + cov["n"] == len(heldout), name
        assert set(bucketed) <= heldout
        assert (r["status"] == "scored") == (cov["n"] > 0)
        if "candidate" in r:
            assert r["candidate"]["coverage"]["0.80"]["n"] == cov["n"]
    assert (report["status"] == "scored") == (report["max_n_scored"] > 0)
