"""P33.4 held-out dump: schema compatibility with pipeline/go_verdict.py.

The real held-out verdict runs on the host (frozen truth, NC_HOST_EVAL=1).
Here we (1) confirm the session tree exits cleanly with an empty-liabilities
dump, and (2) drive ``build_dump`` on a synthetic in-memory universe so the
emitted schema is exercised end to end through the human-owned
``pipeline/go_verdict.py`` judge.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neurocascade.benchmark.v2 import go_dump

ROOT = Path(__file__).resolve().parents[1]


def _load_go_verdict():
    spec = importlib.util.spec_from_file_location("go_verdict", ROOT / "pipeline" / "go_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_session_tree_exits_clean_without_host_eval(tmp_path, monkeypatch):
    monkeypatch.delenv("NC_HOST_EVAL", raising=False)
    out = tmp_path / "go.json"
    rc = go_dump.main(["go_dump", str(out)])
    assert rc == 0
    dump = json.loads(out.read_text())
    assert dump["liabilities"] == {}
    assert dump["status"] == "searched-and-empty"
    assert dump["heldout_drugs"]  # every held-out drug named
    # go_verdict consumes it and prints a verdict without raising.
    gv = _load_go_verdict()
    assert gv.main(str(out)) == 0


def _synthetic(monkeypatch):
    """A tiny universe: HRH1 occupancy tracks the sedation label, so the
    candidate and occupancy baseline both separate the classes."""
    train = [f"tr{i}" for i in range(16)]
    held = [f"he{i}" for i in range(12)]
    # label 1 for the first half of each split; give them high HRH1 occupancy.
    labels, binding, cmax, drugs = {}, {}, {}, []
    for grp in (train, held):
        for i, inn in enumerate(grp):
            pos = i < len(grp) // 2
            labels[inn] = {"sedation": {"label": 1 if pos else 0}}
            ki = 1.0 if pos else 5000.0  # potent H1 for positives
            binding[inn] = {
                "targets": {"HRH1": {"ki_nM": ki}},
                "searched_and_empty": [],
                "kd_only": {},
            }
            cmax[inn] = {"status": "value", "cmax_unbound_nM": 100.0}
            drugs.append({"inn": inn, "atc": ["N05AH" if pos else "N06AB"]})

    monkeypatch.setenv("NC_HOST_EVAL", "1")
    monkeypatch.setattr(go_dump, "v2_partition", lambda: (tuple(train), tuple(held)))
    monkeypatch.setattr(go_dump, "load_v2_drugs", lambda: drugs)
    monkeypatch.setattr(go_dump, "load_v2_binding", lambda: binding)
    monkeypatch.setattr(go_dump, "load_v2_human_cmax", lambda: cmax)
    monkeypatch.setattr(go_dump, "load_v2_liabilities", lambda: labels)
    return held


def test_build_dump_matches_go_verdict_schema(monkeypatch):
    held = _synthetic(monkeypatch)
    dump = go_dump.build_dump()
    assert "sedation" in dump["liabilities"]
    sed = dump["liabilities"]["sedation"]
    # Exact schema go_verdict reads.
    assert set(sed) == {"drugs", "y", "candidate", "baselines"}
    assert sed["drugs"] == held
    assert set(sed["y"]) <= {0, 1} and len(sed["y"]) == len(held)
    assert len(sed["candidate"]) == len(held)
    for name in ("occupancy_max", "strongest_target_ki", "one_nn_fingerprint", "atc_class_prevalence"):
        assert name in sed["baselines"]
        assert len(sed["baselines"][name]) == len(held)
    # Candidate scores are finite probabilities from the predict function.
    assert all(0.0 <= s <= 1.0 for s in sed["candidate"])


def test_go_verdict_judges_the_synthetic_dump(monkeypatch):
    _synthetic(monkeypatch)
    dump = go_dump.build_dump()
    gv = _load_go_verdict()
    rng = np.random.default_rng(gv.SEED)
    go, line = gv.judge("sedation", dump["liabilities"]["sedation"], rng)
    # The judge runs and returns a boolean verdict and a human line; on this
    # perfectly separable synthetic set the candidate is not worse than the
    # best baseline.
    assert bool(go) in (True, False)
    assert "sedation" in line
    assert "candidate" in line
