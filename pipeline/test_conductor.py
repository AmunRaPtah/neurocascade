"""Unit tests for the conductor's pure decision logic (no GitHub, Docker or Claude)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conductor as c  # noqa: E402


def diff(*paths):
    return "".join(f"diff --git a/{p} b/{p}\n+x\n" for p in paths)


def test_claim_last_line_wins_and_markdown_tolerated():
    text = "TASK_RESULT: SPLIT P9.1a\nthinking...\n**TASK_RESULT: COMPLETE** all done"
    assert c.parse_claim(text) == ("complete", "all done")
    assert c.parse_claim("TASK_RESULT: BLOCKED_ON_DATA no PK for X")[0] == "blocked-on-data"
    assert c.parse_claim("REVIEW: REJECT wrong units") == ("reject", "wrong units")
    assert c.parse_claim("I think it's COMPLETE") == ("", "")


def test_split_ids_only_children_of_parent():
    assert c.split_ids("P22.2a, P22.2b and P22.2a again P23.1a P22.2", "P22.2") == ["P22.2a", "P22.2b"]


def test_arch_patch_paths():
    assert c.check_arch_patch(diff("src/a.py", "docs/x.md", "tasks/specs/P1.1a.md")) is None
    assert "host-owned" in c.check_arch_patch(diff("src/a.py", "tasks/backlog.json"))
    assert "host-owned" in c.check_arch_patch(diff("pipeline/orchestrate.py"))
    assert "outside" in c.check_arch_patch(diff("setup.py"))
    assert c.check_arch_patch("") == "empty diff"
    assert "host-owned" in c.check_arch_patch(diff("tests/t.py"), protected=["tests/t.py"])


def test_split_patch_only_specs_and_tests():
    subs = ["P9.1a", "P9.1b"]
    ok = diff("tasks/specs/P9.1a.md", "tasks/specs/P9.1b.md", "tests/test_a.py")
    assert c.check_split_patch(ok, subs) is None
    assert "missing" in c.check_split_patch(diff("tasks/specs/P9.1a.md"), subs)
    assert "only add" in c.check_split_patch(ok + diff("src/x.py"), subs)
    assert c.check_split_patch(ok, []) is not None


def test_week_pacing_rises_through_the_week():
    assert c.week_allowance(0) == c.WEEK_FLOOR
    assert c.week_allowance(0.5) < c.week_allowance(1.0) <= 90
    assert c.week_allowance(2) == c.week_allowance(1)


def write_quota(tmp_path, monkeypatch, fh_used, fh_reset, wk_used, wk_reset):
    f = tmp_path / "q.json"
    f.write_text(json.dumps({"ts": time.time(), "rate_limits": {
        "five_hour": {"used_percentage": fh_used, "resets_at": fh_reset},
        "seven_day": {"used_percentage": wk_used, "resets_at": wk_reset}}}))
    monkeypatch.setattr(c.runner, "QUOTA_CACHE", f)


def test_claude_gate_blocks_then_auto_resumes_after_reset(tmp_path, monkeypatch):
    t = time.time()
    write_quota(tmp_path, monkeypatch, 90, t + 3600, 10, t + 6 * 86400)
    ok, why = c.claude_ok()
    assert not ok and "5h" in why
    # Same reading, but the five-hour window has since rolled over.
    write_quota(tmp_path, monkeypatch, 90, t - 1, 10, t + 6 * 86400)
    assert c.claude_ok()[0]


def test_claude_gate_weekly_pace(tmp_path, monkeypatch):
    t = time.time()
    # One day into the week, 60% used: well over the pace line.
    write_quota(tmp_path, monkeypatch, 0, t + 3600, 60, t + 6 * 86400)
    assert not c.claude_ok()[0]
    # Last day of the week, 60% used: allowed.
    write_quota(tmp_path, monkeypatch, 0, t + 3600, 60, t + 0.5 * 86400)
    assert c.claude_ok()[0]
    # Weekly window rolled over.
    write_quota(tmp_path, monkeypatch, 0, t + 3600, 99, t - 1)
    assert c.claude_ok()[0]


def test_next_task_respects_deps_and_skips_blocked_and_given_up():
    tasks = [
        {"id": "A", "status": "done", "deps": []},
        {"id": "B", "status": "blocked-on-data", "deps": ["A"]},
        {"id": "C", "status": "pending", "deps": ["B"]},
        {"id": "D", "status": "failed", "deps": ["A"], "conductor": {"gave_up": True}},
        {"id": "E", "status": "failed", "deps": ["A"]},
    ]
    assert c.next_task(tasks)["id"] == "E"
    tasks[4]["status"] = "done"
    assert c.next_task(tasks) is None


def test_ci_state():
    assert c.ci_state({"statusCheckRollup": []}) == "pending"
    assert c.ci_state({"statusCheckRollup": [{"conclusion": "SUCCESS"}]}) == "success"
    assert c.ci_state({"statusCheckRollup": [{"conclusion": "SUCCESS"}, {"conclusion": ""}]}) == "pending"
    assert c.ci_state({"statusCheckRollup": [{"conclusion": "FAILURE"}]}) == "failure"


def test_gemini_backoff_and_infra_detection(tmp_path):
    assert c.gemini_ok({})[0]
    assert not c.gemini_ok({"gemini_waiting_until": time.time() + 60})[0]
    assert c.gemini_ok({"gemini_waiting_until": time.time() - 1})[0]
    ok, bad, broken = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (ok, bad, broken):
        d.mkdir()
    (ok / "batch.json").write_text(json.dumps({"results": {"P1a": "pr #3", "P1b": "worker-failed"}}))
    (bad / "batch.json").write_text(json.dumps({"results": {"P1a": "pr #3", "P1b": "infra"}}))
    assert not c.batches_had_infra([ok, broken])
    assert c.batches_had_infra([ok, bad])
