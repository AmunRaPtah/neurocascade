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
