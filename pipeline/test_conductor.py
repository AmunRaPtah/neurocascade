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
    assert c.parse_claim("TASK_RESULT: EMPTY no oral F") == ("empty", "no oral F")
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
    # The live conductor's CLAUDE_SPRINT file swaps the pace gate for a flat cap.
    monkeypatch.setattr(c, "SPRINT_FILE", tmp_path / "no-sprint")


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


def test_split_rejects_oversized_spec():
    big = "diff --git a/tasks/specs/P9.1a.md b/tasks/specs/P9.1a.md\n" + "+x\n" * (c.MAX_SPEC_LINES + 1)
    assert "too big" in c.check_split_patch(big + diff("tests/t.py"), ["P9.1a"])


def test_empty_result_marker_round_trips():
    body = f"Backlog-Task: P1.1\n{c.EMPTY_MARK}no PET pairs\n\nlog"
    assert c.empty_result(body) == "no PET pairs"
    assert c.empty_result("Backlog-Task: P1.1\n\nResult: searched-and-empty") is None


def test_optional_tasks_are_skipped_and_count_as_finished():
    tasks = [{"id": "A", "status": "done", "deps": []},
             {"id": "B", "status": "optional", "deps": ["A"]},
             {"id": "C", "status": "pending", "deps": ["A"]}]
    assert c.next_task(tasks)["id"] == "C"
    tasks[2]["status"] = "done"
    assert c.next_task(tasks) is None
    assert all(t["status"] in c.FINISHED for t in tasks)


def test_advance_complete_and_stuck_ignore_optional(monkeypatch):
    sent = []
    monkeypatch.setattr(c, "alert", lambda st, k, text: sent.append((k, text)))
    tasks = [{"id": "A", "status": "done", "deps": []}, {"id": "B", "status": "optional", "deps": []}]
    monkeypatch.setattr(c, "backlog_at", lambda sha: {"tasks": tasks})
    st = {}
    assert c.step_advance(st, "base", dry=True) and st["complete"] and sent[0][0] == "complete"
    tasks.append({"id": "C", "status": "blocked", "deps": []})
    sent.clear()
    assert not c.step_advance({}, "base", dry=True)
    assert sent == [("stuck", "no runnable task; unfinished: C (blocked or given up)")]


def test_arch_patch_rejects_split_and_heldout_truth_but_not_new_data():
    frozen = [c.g.SPLIT, c.g.truth_file("haloperidol")]
    assert "host-owned" in c.check_arch_patch(diff(c.g.SPLIT))     # always forbidden
    assert "host-owned" in c.check_arch_patch(diff("src/x.py", c.g.truth_file("haloperidol")), frozen)
    assert c.check_arch_patch(diff(c.g.truth_file("diazepam"), c.g.BENCH + "adme/new.json"), frozen) is None


ACC = "\n".join(["ACCEPTANCE 1: PASS tests/test_x.py:12 asserts 0.42",
                 "- **ACCEPTANCE 2**: PASS ledger row at docs/validation-ledger.md:88",
                 "CHECK thresholds: PASS no bar changed", "CHECK heldout: PASS split untouched",
                 "CHECK tests: PASS values from PMID 123", "CHECK citations: PASS spot-checked 3"])


def test_review_verdict_needs_evidence_for_every_bullet():
    assert c.review_verdict(ACC + "\nREVIEW: APPROVE all met", 2) == ("approve", "all met")
    # An APPROVE with a FAIL line is a rejection, with the failing line as findings.
    v, f = c.review_verdict(ACC.replace("CHECK tests: PASS", "CHECK tests: FAIL pins output;")
                            + "\nREVIEW: APPROVE", 2)
    assert v == "reject" and "CHECK TESTS: FAIL pins output" in f
    v, f = c.review_verdict(ACC + "\nREVIEW: REJECT n=0 really", 2)
    assert v == "reject" and "n=0 really" in f
    # Missing bullet or no verdict: the review failed, not the author.
    assert c.review_verdict(ACC + "\nREVIEW: APPROVE", 3)[0] is None
    assert c.review_verdict(ACC, 2)[0] is None


def test_review_backend_never_the_authors_model(monkeypatch):
    monkeypatch.delenv("NC_ACCEPT_REVIEWER", raising=False)
    for arch in ("deepseek:deepseek-v4-pro", "codex:work"):
        monkeypatch.setenv("NC_ARCH_BACKEND", arch)
        assert c.review_backend() == ("claude", "", "opus")
    monkeypatch.setenv("NC_ARCH_BACKEND", "claude")
    monkeypatch.setattr(c, "ARCH_MODEL", "opus")
    assert c.review_backend() == ("claude", "", "sonnet")
    monkeypatch.setattr(c, "ARCH_MODEL", "sonnet")
    assert c.review_backend() == ("claude", "", "opus")
    monkeypatch.setenv("NC_ACCEPT_REVIEWER", "codex:work")
    assert c.review_backend() == ("codex", "work", c.CODEX_MODEL)


def accept_env(monkeypatch, tmp_path, final, ci="SUCCESS"):
    """step_accept with GitHub, git, the backlog and the session mocked."""
    calls = {"gh": [], "run": [], "alerts": []}
    tasks = {"P9.9": {"id": "P9.9", "status": "pending", "title": "T", "brief": "B",
                      "acceptance": ["a", "b"]}}
    monkeypatch.setattr(c, "HOME", tmp_path)
    monkeypatch.setattr(c, "gh", lambda *a: calls["gh"].append(a) or "")
    monkeypatch.setattr(c, "git", lambda *a, **k: "sha" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(c, "backlog_at", lambda sha: {"tasks": list(tasks.values())})
    monkeypatch.setattr(c, "update_backlog", lambda msg, fn: fn(tasks))
    monkeypatch.setattr(c, "alert", lambda st, k, t: calls["alerts"].append(k))
    monkeypatch.setattr(c, "backend_ok", lambda k, p: (True, "ok"))
    monkeypatch.setattr(c, "review_backend", lambda: ("claude", "", "sonnet"))

    def run(name, base, prompt, model, adir, wip="", backend=None, readonly=False, timeout_s=None):
        calls["run"].append(dict(name=name, model=model, backend=backend, readonly=readonly, prompt=prompt))
        return {"final": final, "claim": c.parse_claim(final)[0], "rest": c.parse_claim(final)[1]}
    monkeypatch.setattr(c, "run_architect", run)
    pr = {"number": 7, "title": "[P9.9] T", "headRefName": "architect/P9.9",
          "labels": [{"name": c.L_ARCH}], "body": "Backlog-Task: P9.9\n- Host-read n_scored = 4",
          "statusCheckRollup": [{"conclusion": ci}]}
    return calls, tasks, pr


def test_step_accept_approve_labels_pr(monkeypatch, tmp_path):
    calls, tasks, pr = accept_env(monkeypatch, tmp_path, ACC + "\nREVIEW: APPROVE ok")
    st = {"review": {"P9.9": {"rejects": 1, "findings": "x"}}}
    assert c.step_accept(st, [pr], dry=False)
    r = calls["run"][0]
    assert r["readonly"] and r["backend"] == ("claude", "") and r["model"] == "sonnet"
    assert "1. a\n2. b" in r["prompt"] and "n_scored = 4" in r["prompt"]
    assert ("api", "-X", "POST", "repos/{owner}/{repo}/issues/7/labels", "-f", f"labels[]={c.L_APPROVED}") in calls["gh"]
    assert "P9.9" not in st["review"]
    # Approved PRs are left to step_merge.
    pr["labels"].append({"name": c.L_APPROVED})
    assert not c.step_accept(st, [pr], dry=False)


def test_step_accept_reject_feeds_back_and_blocks(monkeypatch, tmp_path):
    bad = ACC.replace("**ACCEPTANCE 2**: PASS", "**ACCEPTANCE 2**: FAIL no ledger row;") + "\nREVIEW: REJECT"
    calls, tasks, pr = accept_env(monkeypatch, tmp_path, bad)
    st = {}
    for k in range(1, c.MAX_REVIEW_REJECTS + 1):
        assert c.step_accept(st, [pr], dry=False)
        assert st["review"]["P9.9"]["rejects"] == k and "no ledger row" in st["review"]["P9.9"]["findings"]
    assert calls["gh"][0][:3] == ("pr", "close", "7")
    assert c.L_APPROVED not in str(calls["gh"])
    assert tasks["P9.9"]["status"] == "blocked" and calls["alerts"] == ["accept-P9.9"]
    assert tasks["P9.9"]["conductor"]["review_rejections"] == c.MAX_REVIEW_REJECTS
    # The next architect prompt carries the findings.
    assert "no ledger row" in c.REVIEW_NOTE.format(n=3, findings=st["review"]["P9.9"]["findings"])


def test_step_accept_no_verdict_is_not_a_rejection(monkeypatch, tmp_path):
    calls, tasks, pr = accept_env(monkeypatch, tmp_path, "I ran out of turns")
    st = {}
    for _ in range(3):
        c.step_accept(st, [pr], dry=False)
    assert "review" not in st and tasks["P9.9"]["status"] == "pending"
    assert ("api", "-X", "POST", "repos/{owner}/{repo}/issues/7/labels", "-f", f"labels[]={c.L_HUMAN}") in calls["gh"]
    assert not any(a[:2] == ("pr", "close") for a in calls["gh"])


def test_step_accept_ci_failure_rejects_without_a_session(monkeypatch, tmp_path):
    calls, tasks, pr = accept_env(monkeypatch, tmp_path, "", ci="FAILURE")
    st = {}
    assert c.step_accept(st, [pr], dry=False)
    assert not calls["run"] and st["review"]["P9.9"]["rejects"] == 1


def test_heldout_runs_counted_per_task(monkeypatch, tmp_path):
    log = tmp_path / "heldout-evals.jsonl"
    monkeypatch.setattr(c, "HELDOUT_LOG", log)
    assert c.heldout_runs() == {}
    log.write_text('{"task": "P1"}\n{"task": "P1"}\n{"task": "P2"}\nbroken\n')
    assert c.heldout_runs() == {"P1": 2, "P2": 1}


def test_next_task_skips_task_past_review_reject_limit():
    tasks = [{"id": "A", "status": "pending", "conductor": {"review_rejections": c.MAX_REVIEW_REJECTS}},
             {"id": "B", "status": "pending"}]
    assert c.next_task(tasks)["id"] == "B"


def test_requires_go_holds_a_task_until_a_host_go_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "GO_DIR", tmp_path)
    tasks = [{"id": "P32.1", "status": "pending", "requires_go": ["P33.4"]},
             {"id": "P29.6", "status": "pending"}]
    assert c.next_task(tasks)["id"] == "P29.6"
    (tmp_path / "P33.4.json").write_text(json.dumps({"verdict": "NO-GO"}))
    assert c.next_task(tasks)["id"] == "P29.6"
    (tmp_path / "P33.4.json").write_text(json.dumps({"verdict": "GO"}))
    assert c.next_task(tasks)["id"] == "P32.1"


def _fake_eval(monkeypatch, tmp_path, text, passed=True):
    def sandbox_pytest(tree, script, name, out):
        out.write_text(text)
        return passed, text
    monkeypatch.setattr(c.o, "sandbox_pytest", sandbox_pytest)
    monkeypatch.setattr(c, "GO_DIR", tmp_path / "go")
    monkeypatch.setattr(c, "HELDOUT_LOG", tmp_path / "log.jsonl")
    adir = tmp_path / "P33.4" / "run1"
    adir.mkdir(parents=True, exist_ok=True)
    return adir


def test_host_eval_writes_the_first_verdict_and_never_flips_it(tmp_path, monkeypatch):
    adir = _fake_eval(monkeypatch, tmp_path, "table\nNC_GO_VERDICT: NO-GO\n")
    c.host_eval(tmp_path, adir, "x")
    assert not c.go_granted("P33.4")
    adir = _fake_eval(monkeypatch, tmp_path, "NC_GO_VERDICT: GO\n")
    c.host_eval(tmp_path, adir, "x")
    assert not c.go_granted("P33.4")


def test_host_eval_go_needs_a_passing_eval(tmp_path, monkeypatch):
    adir = _fake_eval(monkeypatch, tmp_path, "NC_GO_VERDICT: GO\n", passed=False)
    c.host_eval(tmp_path, adir, "x")
    assert not c.go_granted("P33.4")


def test_fail_excerpt_reports_failures_not_the_warnings_summary(tmp_path):
    log = tmp_path / "pytest.txt"
    log.write_text("....F\n" + "tests/test_x.py::test_slow\n  FutureWarning: as_op\n" * 50
                   + "E       assert 'a' not in 'abc'\n"
                   + "FAILED tests/test_y.py::test_guard - AssertionError\n"
                   + "1 failed, 10 passed in 3s\n")
    out = c.fail_excerpt(log, log.read_text())
    assert "FAILED tests/test_y.py::test_guard" in out and "E       assert" in out
    assert "FutureWarning" not in out


def test_only_the_owners_own_prs_issues_and_comments_are_seen(monkeypatch):
    me, stranger = {"login": "AmunRaPtah"}, {"login": "someone"}
    prs = [{"number": 1, "author": me, "isCrossRepository": False, "headRefName": "architect/P1.1"},
           {"number": 2, "author": stranger, "isCrossRepository": True, "headRefName": "architect/P1.1"},
           {"number": 3, "author": me, "isCrossRepository": True, "headRefName": "architect/P1.2"},
           {"number": 4, "author": stranger, "isCrossRepository": False, "headRefName": "architect/P1.3"}]
    iss = [{"number": 5, "title": "[P1.1] mine", "author": me, "labels": []},
           {"number": 6, "title": "[P1.2] theirs", "author": stranger, "labels": []}]
    calls = []

    def fake(*a):
        calls.append(a)
        rows = prs if a[0] == "pr" else iss
        who = a[a.index("--author") + 1]
        # GitHub filters by author server-side; the fake does too, but keeps one
        # stranger fork PR in to show trusted() still re-checks.
        return json.dumps([r for r in rows if r["author"]["login"] == who or r["number"] == 2])
    monkeypatch.setattr(c, "gh", fake)
    assert [p["number"] for p in c.open_prs()] == [1]
    assert [i["number"] for i in c.issues("x")] == [5]
    assert c.open_issue_ids() == {"P1.1"}
    assert c.find_issue("P1.1") == 5 and c.find_issue("P1.2") is None
    assert not c.trusted({"author": None}) and not c.trusted({})
    assert calls and all("--author" in a for a in calls)


def test_stranger_flood_cannot_hide_the_owners_items(monkeypatch):
    me = {"login": "AmunRaPtah"}
    flood = [{"number": n, "author": {"login": "x"}, "isCrossRepository": True,
              "headRefName": "architect/P1.1", "title": "[P1.1] spam", "labels": []} for n in range(1, 300)]
    mine = {"number": 999, "author": me, "isCrossRepository": False, "headRefName": "architect/P1.1",
            "title": "[P1.1] real", "labels": []}

    def fake(*a):
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else 30
        who = a[a.index("--author") + 1] if "--author" in a else None
        return json.dumps([r for r in flood + [mine] if who in (None, r["author"]["login"])][:lim])
    monkeypatch.setattr(c, "gh", fake)
    assert [p["number"] for p in c.open_prs()] == [999]
    assert c.find_issue("P1.1") == 999 and c.open_issue_ids() == {"P1.1"}
