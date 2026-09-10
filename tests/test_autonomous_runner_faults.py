"""Tests for the build runner's infrastructure-fault ceiling.

A tick whose `claude -p` session never starts (dead auth, refused flags, a
missing CLI) returns its task to `pending`, which is right: no attempt at the
work happened, so it must not consume one of the task's three attempts. What
was missing until 2026-09-07 is a bound. On 2026-09-06 an expired token sent
P21.2 to `attempts: 10` against a MAX_ATTEMPTS of 3, each cycle leaving a
`chore(backlog)` start/rollback pair behind, and one cycle swept an unrelated
uncommitted edit into an "autonomous" commit. Retrying an environment fault
cannot fix it, so after MAX_INFRA_FAULTS consecutive faults on one task the
build pauses itself for a human.

The runner is a script rather than a package module, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "autonomous_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("autonomous_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fault_rolls_back_the_attempt_it_was_counted_as(runner):
    """run_tick increments attempts before launching, so a fault must undo it."""
    task = {"id": "P0.1", "attempts": 1}
    faults, halted = runner.record_infra_fault(task)
    assert task["attempts"] == 0
    assert faults == 1
    assert halted is False


def test_faults_accumulate_to_the_ceiling(runner):
    task = {"id": "P0.1", "attempts": 0}
    halted = False
    for _ in range(runner.MAX_INFRA_FAULTS):
        task["attempts"] += 1          # as run_tick does before each launch
        _, halted = runner.record_infra_fault(task)
    assert halted is True
    # The task's real attempts are untouched: a repaired environment still gets
    # the full MAX_ATTEMPTS at the work itself.
    assert task["attempts"] == 0
    assert task["infra_faults"] == runner.MAX_INFRA_FAULTS


def test_attempts_never_goes_negative(runner):
    task = {"id": "P0.1"}
    runner.record_infra_fault(task)
    assert task["attempts"] == 0


def test_resume_leaves_the_tree_clean(runner, tmp_path, monkeypatch):
    """
    --resume clears the fault counters, and those live in a *tracked* file.

    Seen 2026-09-10: the resume path called save_backlog, so it cleared the
    pause flag and then left tasks/backlog.json modified. Every following tick
    aborted with "working tree dirty at tick start", which is the fourth
    variant of the same silent wedge: --status reports a runnable task, the
    pause flag is gone, and nothing ever runs.
    """
    repo = tmp_path / "repo"
    (repo / "tasks").mkdir(parents=True)
    (repo / "logs").mkdir()
    backlog = repo / "tasks" / "backlog.json"
    backlog.write_text(json.dumps(
        {"tasks": [{"id": "P0.1", "status": "pending", "attempts": 0,
                    "infra_faults": runner.MAX_INFRA_FAULTS}]}, indent=2) + "\n")

    def run(*args):
        subprocess.run(["git", *args], cwd=repo, check=True,
                       capture_output=True, text=True)

    (repo / ".gitignore").write_text("logs/\n")   # as the real repo does

    run("init", "-q")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "test")
    run("add", "tasks/backlog.json", ".gitignore")
    run("commit", "-q", "-m", "baseline")

    paused = repo / "logs" / ".paused"
    paused.write_text("3 consecutive infrastructure faults on P0.1\n")

    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(runner, "BACKLOG", backlog)
    monkeypatch.setattr(runner, "LOGS", repo / "logs")
    monkeypatch.setattr(runner, "PAUSED", paused)
    monkeypatch.setattr(runner, "EVENTS", repo / "logs" / "events.jsonl")
    monkeypatch.setattr(sys, "argv", ["autonomous_runner.py", "--resume"])

    assert runner.main() == 0
    assert not paused.exists()

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True).stdout.strip()
    assert dirty == "", f"resume left the tree dirty: {dirty}"

    committed = json.loads(subprocess.run(
        ["git", "show", "HEAD:tasks/backlog.json"], cwd=repo,
        capture_output=True, text=True, check=True).stdout)
    assert "infra_faults" not in committed["tasks"][0]
