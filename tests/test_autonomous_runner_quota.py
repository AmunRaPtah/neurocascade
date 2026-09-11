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
pin the behaviours that matter: it gates when the reading is fresh, it treats a
reading as valid for exactly as long as the window it was taken in is open
(usage only rises inside a window, so an old reading from an open window is a
lower bound and safe to block on, while one from a rolled-over window describes
refunded spending and would wedge the build permanently), and it falls through
rather than raising when the file is absent or malformed.

Second finding, 2026-09-07: the cache had been frozen since 2026-08-29 because
the CLI moved `rate_limits` from `.current_usage` to the payload's top level and
the statusline read only the old path. The window-aware validity below is what
makes that class of freeze visible instead of silent.

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


def write_cache(runner, tmp_path, monkeypatch, *, five_hour, seven_day, age_s=0.0,
                resets_in_s=None):
    """Point the runner at a cache file with the given used-percentages.

    `resets_in_s` writes a `resets_at` on both windows, relative to now: the
    statusline has recorded it since 2026-09-07. Left as None the payload has
    no window end, which is the pre-2026-09-07 shape the age bound still covers.
    """
    payload = {"ts": time.time() - age_s, "rate_limits": {}}
    if five_hour is not None:
        payload["rate_limits"]["five_hour"] = {"used_percentage": five_hour}
    if seven_day is not None:
        payload["rate_limits"]["seven_day"] = {"used_percentage": seven_day}
    if resets_in_s is not None:
        for window in payload["rate_limits"].values():
            window["resets_at"] = time.time() + resets_in_s
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


def test_old_reading_inside_an_open_window_still_gates(runner, tmp_path, monkeypatch):
    """`resets_at`, not file age, decides whether a reading still applies.

    The statusline only writes while an interactive session renders it, so an
    unattended overnight build sees readings far older than the age bound. As
    long as the window they were taken in is still open they are a lower bound
    on usage now, which is exactly the direction a gate needs.
    """
    write_cache(
        runner, tmp_path, monkeypatch,
        five_hour=10.0, seven_day=92.0,
        age_s=runner.QUOTA_CACHE_MAX_AGE_S * 4,
        resets_in_s=3600,
    )
    ok, why = runner._quota_from_statusline_cache()
    assert ok is False
    assert f"below {runner.MIN_WEEK_QUOTA}%" in why


def test_reading_from_a_rolled_over_window_imposes_no_constraint(runner, tmp_path, monkeypatch):
    """Spending in a window that has already reset has been refunded.

    Believing it is how the gate wedges the build permanently: the number can
    only decay, never recover, so a single exhausted window would block every
    future tick forever.
    """
    write_cache(
        runner, tmp_path, monkeypatch,
        five_hour=99.0, seven_day=99.0,
        age_s=60,
        resets_in_s=-60,
    )
    assert runner._quota_from_statusline_cache() is None


def test_one_rolled_over_window_does_not_hide_the_other(runner, tmp_path, monkeypatch):
    """The five-hour window rolls over many times inside one weekly window."""
    payload = {
        "ts": time.time(),
        "rate_limits": {
            "five_hour": {"used_percentage": 99.0, "resets_at": time.time() - 60},
            "seven_day": {"used_percentage": 95.0, "resets_at": time.time() + 86400},
        },
    }
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(runner, "QUOTA_CACHE", path)

    ok, why = runner._quota_from_statusline_cache()
    assert ok is False
    assert f"below {runner.MIN_WEEK_QUOTA}%" in why


def test_rolled_over_window_is_named_rather_than_printed_as_none(runner, tmp_path, monkeypatch):
    payload = {
        "ts": time.time(),
        "rate_limits": {
            "five_hour": {"used_percentage": 99.0, "resets_at": time.time() - 60},
            "seven_day": {"used_percentage": 20.0, "resets_at": time.time() + 86400},
        },
    }
    path = tmp_path / "named.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(runner, "QUOTA_CACHE", path)

    ok, why = runner._quota_from_statusline_cache()
    assert ok is True
    assert "five_hour" in why and "rolled over" in why


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


def test_agent_rate_limit_event_refreshes_the_cache(runner, tmp_path, monkeypatch):
    """A tick writes the quota reading its own session reports.

    Until 2026-09-11 the statusline was the only writer, so between interactive
    sessions the five_hour window rolled over unobserved and the gate ran on
    the weekly window alone. This is what closes that blind spot: the reading
    below is the shape the CLI emits, and it must gate the very next tick.
    """
    cache = tmp_path / "rate-limits.json"
    monkeypatch.setattr(runner, "QUOTA_CACHE", cache)

    assert runner.record_quota_reading({
        "status": "allowed",
        "rateLimitType": "five_hour",
        "unifiedWindows": {
            "five_hour": {"utilization": 0.9, "resetsAt": time.time() + 1800},
            "seven_day": {"utilization": 0.35, "resetsAt": time.time() + 86400},
        },
    }) is True

    payload = json.loads(cache.read_text())
    assert payload["rate_limits"]["five_hour"]["used_percentage"] == 90.0
    assert payload["rate_limits"]["seven_day"]["used_percentage"] == 35.0

    ok, why = runner._quota_from_statusline_cache()
    assert ok is False
    assert f"below {runner.MIN_SESSION_QUOTA}%" in why


def test_rate_limit_event_without_windows_leaves_the_cache_alone(runner, tmp_path,
                                                                 monkeypatch):
    """An event with no `unifiedWindows` must not overwrite a good reading."""
    cache = write_cache(runner, tmp_path, monkeypatch, five_hour=10.0,
                        seven_day=20.0, resets_in_s=1800)
    before = cache.read_text()

    assert runner.record_quota_reading({"status": "allowed"}) is False
    assert runner.record_quota_reading({}) is False
    assert cache.read_text() == before


def test_unwritable_cache_is_not_fatal(runner, tmp_path, monkeypatch):
    """The gate degrades to the statusline's reading; the tick does not die."""
    monkeypatch.setattr(runner, "QUOTA_CACHE", tmp_path / "nodir" / "x.json")

    def deny(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(runner.Path, "mkdir", deny)
    assert runner.record_quota_reading({
        "unifiedWindows": {"five_hour": {"utilization": 0.1,
                                         "resetsAt": time.time() + 60}},
    }) is False
