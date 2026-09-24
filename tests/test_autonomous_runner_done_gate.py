"""Tests for the runner's completion gate.

On 2026-09-11 P22.3 was marked done because the suite was green and a commit
existed; the only commit was a swept-up Anubis bot-check page (base.html) and
the agent never claimed COMPLETE. Done now needs the agent's own TASK_RESULT
and, for COMPLETE, a change that could be the work.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "autonomous_runner_done_gate", ROOT / "scripts" / "autonomous_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_result_claim_kinds():
    assert runner.result_claim("x\nTASK_RESULT: COMPLETE") == ("complete", "")
    assert runner.result_claim("TASK_RESULT: BLOCKED no tool") == ("blocked", "no tool")
    assert runner.result_claim("TASK_RESULT: BLOCKED_ON_DATA paywalled") == \
        ("blocked-on-data", "paywalled")
    assert runner.result_claim("") == ("", "")
    assert runner.result_claim(None) == ("", "")


def test_result_claim_last_line_wins():
    text = "TASK_RESULT: BLOCKED tests red\n...\nTASK_RESULT: COMPLETE"
    assert runner.result_claim(text)[0] == "complete"


def test_root_debris_is_not_substantive():
    assert runner.substantive_paths(["base.html"]) == []
    assert runner.substantive_paths(["base.html", "tasks/backlog.json"]) == []


def test_real_work_is_substantive():
    paths = ["src/neurocascade/x.py", "docs/validation-ledger.md", "README.md"]
    assert runner.substantive_paths(paths + ["base.html"]) == paths
