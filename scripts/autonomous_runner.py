#!/usr/bin/env python3
"""
Autonomous build runner for neurocascade.

One tick picks the first pending task whose dependencies are all done, hands it
to a headless `claude -p` session inside this repo, gates the result on the test
suite, and commits and pushes only if the suite is green. Work that fails the
gate is parked on a `failed/<task-id>` branch and main is reset to where it
started, so main is always green and no work is ever destroyed.

Designed to be called repeatedly from cron under flock. A tick that finds
nothing to do, or that hits a resource guard, exits 0 without side effects.

Usage:
  autonomous_runner.py               run one tick
  autonomous_runner.py --status      print backlog progress and exit
  autonomous_runner.py --dry-run     show what would run, change nothing
  autonomous_runner.py --verify-full run the whole suite, slow included
  autonomous_runner.py --pause "why" stop new ticks; --resume undoes it

Before editing this repo by hand, check --status for a tick in flight. A tick
commits whatever it finds uncommitted at the end, so an edit made while one is
running is attributed to the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKLOG = REPO / "tasks" / "backlog.json"
LOGS = REPO / "logs"
# Both live under logs/, which is gitignored, so neither can be swept into a
# commit by the very sweep ACTIVE exists to make safe.
ACTIVE = LOGS / ".tick-active"   # a tick is holding the tree right now
PAUSED = LOGS / ".paused"        # --pause, so humans need not edit crontab
EVENTS = LOGS / "events.jsonl"
VENV_PY = Path("/root/projects/pardalos/.venv/bin/python")

AGENT_TIMEOUT_S = int(os.environ.get("NC_AGENT_TIMEOUT_S", 150 * 60))
TEST_TIMEOUT_S = int(os.environ.get("NC_TEST_TIMEOUT_S", 40 * 60))
MAX_ATTEMPTS = int(os.environ.get("NC_MAX_ATTEMPTS", 3))
MIN_SESSION_QUOTA = int(os.environ.get("NC_MIN_SESSION_QUOTA", 25))
MIN_WEEK_QUOTA = int(os.environ.get("NC_MIN_WEEK_QUOTA", 15))
MIN_MEM_MB = int(os.environ.get("NC_MIN_MEM_MB", 1500))

# Server-truth quota, cached by the statusline (see /root/.claude/statusline-quota.sh).
# quota-axi's Claude source authenticates separately and has been returning a
# stale cached reading on this host, which quota_ok() then discards as expired,
# leaving the build ungated. This file is written from the `rate_limits` block
# the CLI itself reports, so when it is fresh it is the better source.
QUOTA_CACHE = Path("/var/lib/claude-quota/rate-limits.json")
QUOTA_CACHE_MAX_AGE_S = int(os.environ.get("NC_QUOTA_CACHE_MAX_AGE_S", 6 * 3600))

ALLOWED_TOOLS = "Read,Write,Edit,Glob,Grep,Bash,WebSearch,WebFetch,TodoWrite,NotebookEdit"

PROJECT_RULES = """
You are working autonomously inside the neurocascade repository at {repo}.
neurocascade models how a drug binding a receptor propagates through brain
circuits to a predicted behavioral effect, as one coupled ODE system
(pharmacokinetics, competitive receptor binding, Wilson-Cowan circuit dynamics,
behavior readout) integrated with scipy.solve_ivp.

Read docs/roadmap.md and docs/architecture.md before you start. architecture.md
is the running record of every design decision and calibration attempt in this
project, and it is long; read the sections relevant to your task.

Non-negotiable rules for this codebase:

1. Never invent a number. Every parameter is either literature-sourced with a
   citation (author, year, species, route where applicable), or explicitly
   labeled an illustrative placeholder. If you search for a value and cannot
   find a usable one, record that as a searched-and-empty result with what you
   searched and why each candidate was unusable. That is a legitimate outcome
   here; docs/architecture.md's Track E section is the template to follow.
2. Never substitute a different drug's measured effect, or a different species,
   for the one actually being modeled, even when that would let a fit succeed.
3. Remove an "illustrative" label only in the same commit that adds the real
   calibration for that specific parameter, and never for any other parameter.
4. Tests must pass. Do not delete, skip, weaken or xfail an existing test to get
   to green. If an existing test genuinely encodes a wrong expectation, fix the
   test and explain why in the commit message.
4b. The gate you are judged by runs `pytest -m "not slow"`. Iterate against that
   for speed, but run the full suite once before you commit, because the daily
   --verify-full sweep will run it anyway. `slow` classifies a test that takes
   minutes for real computational reasons (MCMC and the like). Marking a test
   `slow` to get it out of the gate is the same offence as skipping it under
   rule 4. Never apply the marker to a test that is failing.
5. Do not widen a numeric tolerance purely to make a validation pass. Report the
   real error instead.
6. Write tests for what you add, and update docs/architecture.md with the
   reasoning behind your design decisions, not just the outcome.
7. Use the project venv at {venv} for running anything Python.
8. Commit your work with a clear message. Do not push; the runner pushes after
   the test gate passes.

The next four rules were added on 2026-09-05 after an audit of how this backlog
reached 77/77 while the model could predict exactly one measured human quantity.
They are about what counts as finishing, and they override any instinct to leave
a task looking complete.

9. Scaffolding is not progress. A contract, a registry, a dataclass, a gate or a
   schema is not a result no matter how well tested. A task closes on either a
   prediction compared against a number nobody in this repository chose, or an
   explicit searched-and-empty record. Phases 14 to 18 produced fifteen modules
   and one sourced number; do not add to that pattern. Add your row to
   docs/validation-ledger.md, which is the real measure of progress here.
10. Never build a container for data you do not have. If the source cannot be
   obtained, stop and report BLOCKED_ON_DATA with exactly what was unreachable
   and where you looked. An empty registry with a `require_*` gate that always
   fails is worse than nothing: it reads as capability and delivers none.
11. The brief may be wrong about structure. It was written before the source was
   read. Read the source first and follow what it actually prints; if that
   contradicts the brief, follow the source and say so in your final message.
   P19.5's brief assumed a three-compartment model where the source published a
   two-compartment apparent-oral one.
12. Check the load-bearing assumption before you build on it, not after. Thirty-
   nine tasks of mouse-scale model were built before anyone tested whether the
   mRNA-to-receptor-density proxy was defensible; it was not, and the whole layer
   was retired. If your task rests on one assumption that could void it, test
   that assumption first and report if it fails.

The environment has network access. Use WebSearch and WebFetch to find real
literature values. Prefer primary sources over review summaries when you can
reach them.
""".strip()

TASK_PROMPT = """
{rules}

# Your task: {task_id} ({phase})

## {title}

{brief}

## Acceptance criteria

{acceptance}

Work until the task is complete and the full test suite passes. Commit your
work when done. If you conclude the task cannot be completed honestly with
available data, say so explicitly, commit whatever documentation of that
conclusion is worth keeping (a searched-and-empty record), and explain in your
final message.

Before you finish, add or update this task's row in docs/validation-ledger.md.
If the task produced no comparison against an externally measured number, say so
there in the untested table rather than leaving the file untouched.

End your final message with exactly one of these lines:
TASK_RESULT: COMPLETE
TASK_RESULT: BLOCKED <one-line reason>
TASK_RESULT: BLOCKED_ON_DATA <what source was unreachable, and where you looked>

BLOCKED_ON_DATA is the honest outcome when the work is gated on a number that
does not exist or cannot be reached, and it is not a failure. Use it instead of
building a placeholder, and instead of quietly narrowing the task to whatever
happened to be reachable.
""".strip()

FIX_PROMPT = """
{rules}

You were working on task {task_id} ({title}) in this repository. Your changes
are committed, but the test suite is failing. Here is the tail of the failure
output:

```
{failure}
```

Fix the failures. Do not delete, skip or weaken tests to get to green, and do
not widen numeric tolerances to hide a real error. If a test encodes a genuinely
wrong expectation, correct the test and explain why in the commit message.
Commit your fix when the suite passes.

End your final message with exactly one of these lines:
TASK_RESULT: COMPLETE
TASK_RESULT: BLOCKED <one-line reason>
""".strip()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(kind: str, **fields) -> None:
    LOGS.mkdir(exist_ok=True)
    rec = {"ts": now(), "kind": kind, **fields}
    with EVENTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{kind}] " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def load_backlog() -> dict:
    return json.loads(BACKLOG.read_text())


def save_backlog(data: dict) -> None:
    tmp = BACKLOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(BACKLOG)


def git(*args, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def commit_backlog(data: dict, message: str) -> None:
    """
    Persist backlog state and commit it on its own.

    backlog.json is tracked, so a bare save_backlog leaves the tree dirty and
    the next tick aborts on the cleanliness check. Committing it here keeps the
    tree clean between ticks, and keeping it in a separate commit stops
    bookkeeping from masquerading as agent work in the `made_commits` gate.
    """
    save_backlog(data)
    if git("status", "--porcelain", "--", str(BACKLOG)):
        git("commit", "-q", "-m", message, "--", str(BACKLOG))


def recover_stale(data: dict) -> bool:
    """
    Reclaim tasks left `running` by a tick that died.

    flock means only one tick is ever live, so reaching this point while holding
    the lock proves any `running` task belongs to a dead tick. Its partial work
    is parked on an `interrupted/` branch and main is reset to the baseline that
    tick recorded, so the build can move again without losing anything.
    """
    changed = False
    for t in data["tasks"]:
        if t.get("status") != "running":
            continue
        n = t.get("attempts", 1)
        baseline = t.get("baseline") or git("rev-parse", "origin/main")
        head = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
        branch = "none"
        if dirty or head != baseline:
            branch = f"interrupted/{t['id']}-attempt{n}"
            if dirty:
                git("add", "-A")
                git("commit", "-m",
                    f"{t['id']}: partial work from an interrupted tick (attempt {n})")
            git("branch", "-f", branch, "HEAD")
            git("reset", "--hard", baseline)
        t["status"] = "pending"
        t["result"] = f"interrupted on attempt {n}, partial work parked on {branch}"
        changed = True
        log_event("recovered", task=t["id"], branch=branch,
                  baseline=baseline[:8], attempts=n)
    return changed


def next_task(data: dict):
    done = {t["id"] for t in data["tasks"] if t["status"] == "done"}
    for t in data["tasks"]:
        if t["status"] != "pending":
            continue
        if all(d in done for d in t.get("deps", [])):
            return t
    return None


def _window_expired(resets_at) -> bool:
    """True when a quota window's reset time has already passed."""
    if not resets_at:
        return False
    try:
        ts = datetime.fromisoformat(str(resets_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts <= datetime.now(timezone.utc)


def _quota_from_statusline_cache() -> tuple[bool, str] | None:
    """Gate on the statusline's cached rate_limits, or None if unusable.

    Returns None when the file is missing, malformed or older than
    QUOTA_CACHE_MAX_AGE_S, so the caller falls back to quota-axi. A stale
    reading is worse than no reading: the percentages only decay, so trusting
    one wedges the build the same way an expired quota-axi window would.
    """
    try:
        payload = json.loads(QUOTA_CACHE.read_text())
        age = time.time() - float(payload["ts"])
        limits = payload["rate_limits"]
    except Exception:
        return None
    if age > QUOTA_CACHE_MAX_AGE_S:
        return None

    def remaining(window):
        used = limits.get(window, {}).get("used_percentage")
        return None if used is None else 100 - float(used)

    session, week = remaining("five_hour"), remaining("seven_day")
    if session is None and week is None:
        return None
    age_note = f"statusline cache {int(age // 60)}m old"
    if session is not None and session < MIN_SESSION_QUOTA:
        return False, f"session quota {session:.0f}% below {MIN_SESSION_QUOTA}% ({age_note})"
    if week is not None and week < MIN_WEEK_QUOTA:
        return False, f"weekly quota {week:.0f}% below {MIN_WEEK_QUOTA}% ({age_note})"
    return True, f"session {session}%, week {week}% ({age_note})"


def quota_ok() -> tuple[bool, str]:
    cached = _quota_from_statusline_cache()
    if cached is not None:
        return cached
    try:
        r = subprocess.run(
            ["quota-axi", "--provider", "claude", "--json"],
            capture_output=True, text=True, timeout=90,
        )
        payload = json.loads(r.stdout)
    except Exception as e:
        return True, f"quota check unavailable ({e}), proceeding"

    # quota-axi falls back to a cached reading when it cannot reach the live
    # source (this host authenticates with a pinned setup token, so "Claude
    # sign-in required" is the steady state, not an outage). A cached window
    # whose resetsAt has already passed describes a window that has since
    # rolled over, so its percentage says nothing about quota now. Treating it
    # as current wedges the build permanently: the number can only go stale,
    # never recover. Drop expired windows and let the remaining guards decide.
    windows = {}
    expired = []
    for prov in payload.get("providers", []):
        if prov.get("provider") == "claude":
            for w in prov.get("windows", []):
                if _window_expired(w.get("resetsAt")):
                    expired.append(w.get("id"))
                    continue
                windows[w.get("id")] = w.get("percentRemaining")

    session = windows.get("five_hour")
    week = windows.get("seven_day")
    if session is not None and session < MIN_SESSION_QUOTA:
        return False, f"session quota {session}% below {MIN_SESSION_QUOTA}%"
    if week is not None and week < MIN_WEEK_QUOTA:
        return False, f"weekly quota {week}% below {MIN_WEEK_QUOTA}%"
    note = f"session {session}%, week {week}%"
    if expired:
        note += f" (expired, ignored: {','.join(str(e) for e in expired)})"
    return True, note


def memory_ok() -> tuple[bool, str]:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mb = int(line.split()[1]) // 1024
            if mb < MIN_MEM_MB:
                return False, f"{mb}MB available, need {MIN_MEM_MB}MB"
            return True, f"{mb}MB available"
    return True, "unknown"


def run_agent(prompt: str, log_path: Path) -> tuple[bool, str]:
    """Run one headless claude session. Returns (ok, final_text)."""
    cmd = [
        "claude", "-p", prompt,
        "--allowedTools", ALLOWED_TOOLS,
        "--permission-mode", "bypassPermissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    final_text = ""
    plain = []  # non-JSON output: where the CLI puts startup refusals
    started = time.time()
    with log_path.open("a") as logf:
        logf.write(f"\n===== agent start {now()} =====\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, bufsize=1,
        )
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                line = line.strip()
                if not line.startswith("{"):
                    if line:
                        plain.append(line)
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "result":
                    final_text = obj.get("result", "") or ""
                if time.time() - started > AGENT_TIMEOUT_S:
                    proc.kill()
                    return False, "agent exceeded timeout"
            proc.wait(timeout=60)
        except Exception as e:
            proc.kill()
            return False, f"agent crashed: {e}"

    if proc.returncode != 0 and not final_text:
        # A session that never produced a result line failed to start. Surface
        # the CLI's own words; they are the whole diagnosis.
        return False, " | ".join(plain[-3:])
    return proc.returncode == 0, final_text


def run_tests(log_path: Path, full: bool = False) -> tuple[bool, str]:
    """
    Per-task gate, or the whole suite when `full`.

    The gate deselects `slow` (MCMC sampling: 409s for 7 tests against ~20s for
    the other ~130). Skipping it per task would be a real loss of coverage, so
    it is not skipped, only moved: --verify-full runs everything daily and the
    digest reports the result. --durations keeps suite growth visible in the
    log instead of arriving later as an unexplained gate timeout.
    """
    cmd = [str(VENV_PY), "-m", "pytest", "-q", "--durations=10"]
    if not full:
        cmd += ["-m", "not slow"]
    label = "pytest (full suite)" if full else "pytest (gate, slow deselected)"
    with log_path.open("a") as logf:
        logf.write(f"\n===== {label} {now()} =====\n")
        logf.flush()
        try:
            r = subprocess.run(
                cmd, cwd=REPO, capture_output=True, text=True, timeout=TEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logf.write("pytest timed out\n")
            return False, "pytest timed out"
        out = r.stdout + r.stderr
        logf.write(out)
    return r.returncode == 0, out[-4000:]


def verify_full() -> int:
    """
    Daily sweep over everything the per-task gate deselects.

    Reports rather than reverts: main is whatever the gate already accepted, so
    the useful action here is to say loudly that a slow test is red, not to
    unwind work that passed the rules it was given.
    """
    if git("status", "--porcelain"):
        log_event("full_suite_skipped", reason="working tree dirty")
        return 1
    LOGS.mkdir(exist_ok=True)
    ok, tail = run_tests(LOGS / "verify-full.log", full=True)
    summary = ""
    for line in reversed(tail.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip()
            break
    log_event("full_suite", ok=ok, head=git("rev-parse", "HEAD")[:8],
              summary=summary[:200])
    return 0 if ok else 1


def format_acceptance(items) -> str:
    return "\n".join(f"- {a}" for a in items)


def run_tick(dry_run: bool = False) -> int:
    """Wrapper so the active-tick marker is cleared on every exit path."""
    try:
        return _run_tick(dry_run=dry_run)
    finally:
        ACTIVE.unlink(missing_ok=True)


def _run_tick(dry_run: bool = False) -> int:
    if PAUSED.exists() and not dry_run:
        log_event("paused", reason=PAUSED.read_text().strip()[:160] or "no reason given")
        return 0

    data = load_backlog()
    if not dry_run and recover_stale(data):
        commit_backlog(data, "chore(backlog): reclaim tasks from an interrupted tick")
    task = next_task(data)
    if task is None:
        remaining = [t for t in data["tasks"] if t["status"] == "pending"]
        if not remaining:
            log_event("all_done", message="backlog complete")
        else:
            blocked = [t["id"] for t in remaining]
            log_event("nothing_runnable", pending=len(remaining), ids=",".join(blocked[:5]))
        return 0

    if dry_run:
        print(f"would run {task['id']}: {task['title']}")
        return 0

    ok, why = memory_ok()
    if not ok:
        log_event("skipped", task=task["id"], reason=f"memory: {why}")
        return 0

    ok, why = quota_ok()
    if not ok:
        log_event("skipped", task=task["id"], reason=f"quota: {why}")
        return 0

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        log_event("aborted", reason=f"not on main (on {branch})")
        return 1
    if git("status", "--porcelain"):
        log_event("aborted", reason="working tree dirty at tick start")
        return 1

    baseline = git("rev-parse", "HEAD")
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"task-{task['id']}.log"

    task["status"] = "running"
    task["attempts"] = task.get("attempts", 0) + 1
    task["started_at"] = now()
    task["baseline"] = baseline
    commit_backlog(data, f"chore(backlog): start {task['id']} (autonomous)")

    # Re-anchor after the bookkeeping commit. `baseline` recorded in the backlog
    # stays pre-marker so recovery can discard the marker too, but the gate below
    # must compare against HEAD-as-the-agent-found-it, or a run that produces
    # nothing still looks productive.
    gate_baseline = git("rev-parse", "HEAD")
    ACTIVE.write_text(f"{task['id']} pid={os.getpid()} since={now()}\n")
    log_event("started", task=task["id"], title=task["title"],
              attempt=task["attempts"], quota=why, baseline=gate_baseline[:8])

    rules = PROJECT_RULES.format(repo=REPO, venv=VENV_PY)
    prompt = TASK_PROMPT.format(
        rules=rules,
        task_id=task["id"],
        phase=task["phase"],
        title=task["title"],
        brief=task["brief"],
        acceptance=format_acceptance(task.get("acceptance", [])),
    )

    agent_ok, final_text = run_agent(prompt, log_path)

    if (not agent_ok and not git("status", "--porcelain")
            and git("rev-parse", "HEAD") == gate_baseline):
        # The session never ran: no output, no edits, no commits. That is an
        # infrastructure fault (refused flags, missing CLI, dead auth), not a
        # task the agent tried and lost. Put the task back to pending so a
        # repaired environment retries it, and name the cause in the event log
        # instead of recording a silent empty completion.
        task["status"] = "pending"
        task["result"] = f"agent did not start: {final_text.strip()[:200]}"
        task.pop("started_at", None)
        commit_backlog(data, f"chore(backlog): {task['id']} agent did not start (autonomous)")
        log_event("agent_launch_failed", task=task["id"], attempts=task["attempts"],
                  detail=final_text.strip()[:160] or "no output")
        return 1

    # BLOCKED_ON_DATA is kept distinct from BLOCKED. Both stop the task, but the
    # first says the science is waiting on a number that could not be reached and
    # the second says the work itself failed. Collapsing them is how a data
    # shortage gets re-read as an engineering backlog, which is the mistake that
    # produced fifteen modules holding one sourced value.
    blocked_reason = ""
    blocked_kind = ""
    for line in (final_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("TASK_RESULT: BLOCKED_ON_DATA"):
            blocked_reason = stripped.split("BLOCKED_ON_DATA", 1)[1].strip()
            blocked_kind = "blocked-on-data"
        elif stripped.startswith("TASK_RESULT: BLOCKED"):
            blocked_reason = stripped.split("BLOCKED", 1)[1].strip()
            blocked_kind = "blocked"

    # Commit anything the agent left uncommitted so the gate sees all of it.
    #
    # The tree was verified clean before the agent started, so in principle
    # everything here is the agent's. In practice this cannot tell the agent's
    # work from an edit someone made by hand while the tick was running, and it
    # will silently attribute the latter to the agent. Naming the paths in the
    # event log is what makes that recoverable after the fact; logs/.tick-active
    # is what lets a human avoid causing it in the first place.
    swept = git("status", "--porcelain")
    if swept:
        paths = [ln[3:] for ln in swept.splitlines()]
        git("add", "-A")
        git("commit", "-m",
            f"{task['id']}: {task['title']} (autonomous, uncommitted remainder)")
        log_event("swept_uncommitted", task=task["id"], count=len(paths),
                  paths=",".join(paths[:12])[:400])

    tests_ok, failure = run_tests(log_path)

    fix_attempts = 0
    while not tests_ok and fix_attempts < MAX_ATTEMPTS - 1:
        fix_attempts += 1
        log_event("test_gate_failed", task=task["id"], fix_attempt=fix_attempts)
        fix_prompt = FIX_PROMPT.format(
            rules=rules, task_id=task["id"], title=task["title"], failure=failure)
        run_agent(fix_prompt, log_path)
        if git("status", "--porcelain"):
            git("add", "-A")
            git("commit", "-m", f"{task['id']}: fix test failures (autonomous)")
        tests_ok, failure = run_tests(log_path)

    head = git("rev-parse", "HEAD")
    made_commits = head != gate_baseline

    if tests_ok and made_commits:
        task["status"] = blocked_kind if blocked_reason else "done"
        task["finished_at"] = now()
        task["result"] = blocked_reason or "complete"
        task["commits"] = git("rev-list", "--count", f"{gate_baseline}..HEAD")
        commit_backlog(data, f"chore(backlog): {task['id']} {task['status']} (autonomous)")
        try:
            git("push", "origin", "main")
            pushed = True
        except RuntimeError as e:
            pushed = False
            log_event("push_failed", task=task["id"], error=str(e)[:200])
        log_event("finished", task=task["id"], status=task["status"],
                  commits=task["commits"], pushed=pushed, note=blocked_reason[:120])
        return 0

    if tests_ok and not made_commits:
        task["status"] = blocked_kind if blocked_reason else "failed"
        task["finished_at"] = now()
        task["result"] = blocked_reason or "agent produced no commits"
        commit_backlog(data, f"chore(backlog): {task['id']} {task['status']} (autonomous)")
        log_event("finished", task=task["id"], status=task["status"],
                  note=task["result"][:160])
        return 0

    # Tests still red: park the work, restore main to its green baseline.
    quarantine = f"failed/{task['id']}-attempt{task['attempts']}"
    git("branch", "-f", quarantine, "HEAD")
    git("reset", "--hard", gate_baseline)
    task["status"] = "failed"
    task["finished_at"] = now()
    task["result"] = f"test gate failed, work parked on {quarantine}"
    commit_backlog(data, f"chore(backlog): {task['id']} failed the test gate (autonomous)")
    log_event("quarantined", task=task["id"], branch=quarantine,
              attempts=task["attempts"], agent_ok=agent_ok)
    return 0


def print_status() -> None:
    data = load_backlog()
    counts = {}
    for t in data["tasks"]:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    total = len(data["tasks"])
    done = counts.get("done", 0)
    print(f"neurocascade autonomous build: {done}/{total} done")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if PAUSED.exists():
        print(f"  PAUSED: {PAUSED.read_text().strip()}")
    if ACTIVE.exists():
        print(f"  tick in flight: {ACTIVE.read_text().strip()} "
              f"(do not edit the repo until it lands)")
    print()
    for t in data["tasks"]:
        mark = {"done": "x", "running": ">", "failed": "!", "blocked": "?",
                "blocked-on-data": "D", "pending": " "}.get(t["status"], " ")
        print(f"  [{mark}] {t['id']:6} {t['title']}")
    nxt = next_task(data)
    print(f"\nnext up: {nxt['id']} {nxt['title']}" if nxt else "\nnext up: nothing runnable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-full", action="store_true",
                    help="run the whole suite including slow, and report")
    ap.add_argument("--pause", metavar="REASON", nargs="?", const="paused by hand",
                    help="stop new ticks starting, without touching crontab")
    ap.add_argument("--resume", action="store_true", help="undo --pause")
    args = ap.parse_args()
    if args.pause is not None:
        LOGS.mkdir(exist_ok=True)
        PAUSED.write_text(f"{args.pause} (since {now()})\n")
        active = ACTIVE.read_text().strip() if ACTIVE.exists() else ""
        print(f"paused: {args.pause}")
        # Pausing stops the *next* tick; it does not stop the one already in
        # flight, and that distinction is the whole point of waiting for a
        # clean tree before editing anything.
        print(f"a tick is STILL RUNNING: {active}" if active
              else "no tick in flight; the tree is yours")
        return 0
    if args.resume:
        PAUSED.unlink(missing_ok=True)
        print("resumed")
        return 0
    if args.status:
        print_status()
        return 0
    if args.verify_full:
        return verify_full()
    return run_tick(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
