"""Tests for the build runner's quota gate (`scripts/autonomous_runner.py`).

The gate exists so an autonomous tick cannot drain the interactive Claude
quota, which is the failure that paused this build on 2026-08-14. On
2026-09-06 it was found to be inert: `quota-axi`'s Claude source returns a
cached reading whose windows have all expired, `quota_ok()` correctly discards
expired windows, and what remained was `session None%, week None%` and an
unconditional go. A gate that cannot fail is not a gate, and nothing in the
tick log or the digest says so.

`_quota_from_statusline_cache` reads the server-truth `rate_limits` block the
CLI reports, which the statusline caches to /var/lib/claude-quota. These tests
pin the three behaviours that matter: it gates when the reading is fresh, it
refuses to gate on a stale one (a stale percentage only decays, so trusting it
would wedge the build permanently the same way an expired window would), and
it falls through rather than raising when the file is absent or malformed.

The runner is a script rather than a package module, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "autonomous_runner.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("autonomous_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cache(runner, tmp_path, monkeypatch, *, five_hour, seven_day, age_s=0.0):
    """Point the runner at a cache file with the given used-percentages."""
    payload = {"ts": time.time() - age_s, "rate_limits": {}}
    if five_hour is not None:
        payload["rate_limits"]["five_hour"] = {"used_percentage": five_hour}
    if seven_day is not None:
        payload["rate_limits"]["seven_day"] = {"used_percentage": seven_day}
    path = tmp_path / "rate-limits.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(runner, "QUOTA_CACHE", path)
    return path


def test_fresh_cache_with_headroom_allows_the_tick(runner, tmp_path, monkeypatch):
    write_cache(runner, tmp_path, monkeypatch, five_hour=12.2, seven_day=40.9)
    ok, why = runner._quota_from_statusline_cache()
    assert ok is True
    # Reported as remaining, not used: 40.9% of the week spent is 59% left.
    assert "week 59.1%" in why


def test_weekly_floor_blocks_the_tick(runner, tmp_path, monkeypatch):
    write_cache(runner, tmp_path, monkeypatch, five_hour=10.0, seven_day=90.0)
    ok, why = runner._quota_from_statusline_cache()
    assert ok is False
    assert f"below {runner.MIN_WEEK_QUOTA}%" in why


def test_session_floor_blocks_the_tick(runner, tmp_path, monkeypatch):
    write_cache(runner, tmp_path, monkeypatch, five_hour=80.0, seven_day=10.0)
    ok, why = runner._quota_from_statusline_cache()
    assert ok is False
    assert f"below {runner.MIN_SESSION_QUOTA}%" in why


def test_stale_cache_is_refused_rather_than_believed(runner, tmp_path, monkeypatch):
    """A reading older than the bound must not gate, in either direction.

    The percentages in the file only decay with use, so an old one is not a
    conservative estimate of the current window: it can be arbitrarily wrong
    both ways once the window has rolled over.
    """
    write_cache(
        runner, tmp_path, monkeypatch,
        five_hour=12.2, seven_day=40.9,
        age_s=runner.QUOTA_CACHE_MAX_AGE_S + 60,
    )
    assert runner._quota_from_statusline_cache() is None


def test_missing_and_malformed_cache_fall_through(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "QUOTA_CACHE", tmp_path / "absent.json")
    assert runner._quota_from_statusline_cache() is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    monkeypatch.setattr(runner, "QUOTA_CACHE", broken)
    assert runner._quota_from_statusline_cache() is None

    no_windows = tmp_path / "empty.json"
    no_windows.write_text(json.dumps({"ts": time.time(), "rate_limits": {}}))
    monkeypatch.setattr(runner, "QUOTA_CACHE", no_windows)
    assert runner._quota_from_statusline_cache() is None


def test_quota_ok_prefers_the_cache_over_quota_axi(runner, tmp_path, monkeypatch):
    """A fresh cache must decide, without consulting quota-axi at all."""
    write_cache(runner, tmp_path, monkeypatch, five_hour=10.0, seven_day=95.0)

    def fail(*args, **kwargs):
        raise AssertionError("quota-axi consulted despite a fresh cache")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    ok, why = runner.quota_ok()
    assert ok is False
    assert "weekly quota 5%" in why
