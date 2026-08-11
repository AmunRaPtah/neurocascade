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
  autonomous_runner.py            run one tick
  autonomous_runner.py --status   print backlog progress and exit
  autonomous_runner.py --dry-run  show what would run, change nothing
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
EVENTS = LOGS / "events.jsonl"
VENV_PY = Path("/root/projects/pardalos/.venv/bin/python")

AGENT_TIMEOUT_S = int(os.environ.get("NC_AGENT_TIMEOUT_S", 150 * 60))
TEST_TIMEOUT_S = int(os.environ.get("NC_TEST_TIMEOUT_S", 40 * 60))
MAX_ATTEMPTS = int(os.environ.get("NC_MAX_ATTEMPTS", 3))
MIN_SESSION_QUOTA = int(os.environ.get("NC_MIN_SESSION_QUOTA", 25))
MIN_WEEK_QUOTA = int(os.environ.get("NC_MIN_WEEK_QUOTA", 15))
MIN_MEM_MB = int(os.environ.get("NC_MIN_MEM_MB", 1500))

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
5. Do not widen a numeric tolerance purely to make a validation pass. Report the
   real error instead.
6. Write tests for what you add, and update docs/architecture.md with the
   reasoning behind your design decisions, not just the outcome.
7. Use the project venv at {venv} for running anything Python.
8. Commit your work with a clear message. Do not push; the runner pushes after
   the test gate passes.

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

End your final message with exactly one of these lines:
TASK_RESULT: COMPLETE
TASK_RESULT: BLOCKED <one-line reason>
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


def next_task(data: dict):
    done = {t["id"] for t in data["tasks"] if t["status"] == "done"}
    for t in data["tasks"]:
        if t["status"] != "pending":
            continue
        if all(d in done for d in t.get("deps", [])):
            return t
    return None


def quota_ok() -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["quota-axi", "--provider", "claude", "--json"],
            capture_output=True, text=True, timeout=90,
        )
        payload = json.loads(r.stdout)
    except Exception as e:
        return True, f"quota check unavailable ({e}), proceeding"

    windows = {}
    for prov in payload.get("providers", []):
        if prov.get("provider") == "claude":
            for w in prov.get("windows", []):
                windows[w.get("id")] = w.get("percentRemaining")

    session = windows.get("five_hour")
    week = windows.get("seven_day")
    if session is not None and session < MIN_SESSION_QUOTA:
        return False, f"session quota {session}% below {MIN_SESSION_QUOTA}%"
    if week is not None and week < MIN_WEEK_QUOTA:
        return False, f"weekly quota {week}% below {MIN_WEEK_QUOTA}%"
    return True, f"session {session}%, week {week}%"


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

    return proc.returncode == 0, final_text


def run_tests(log_path: Path) -> tuple[bool, str]:
    with log_path.open("a") as logf:
        logf.write(f"\n===== pytest {now()} =====\n")
        logf.flush()
        try:
            r = subprocess.run(
                [str(VENV_PY), "-m", "pytest", "-q"],
                cwd=REPO, capture_output=True, text=True, timeout=TEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logf.write("pytest timed out\n")
            return False, "pytest timed out"
        out = r.stdout + r.stderr
        logf.write(out)
    return r.returncode == 0, out[-4000:]


def format_acceptance(items) -> str:
    return "\n".join(f"- {a}" for a in items)


def run_tick(dry_run: bool = False) -> int:
    data = load_backlog()
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
    save_backlog(data)
    log_event("started", task=task["id"], title=task["title"],
              attempt=task["attempts"], quota=why, baseline=baseline[:8])

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
    blocked_reason = ""
    for line in (final_text or "").splitlines():
        if line.strip().startswith("TASK_RESULT: BLOCKED"):
            blocked_reason = line.split("BLOCKED", 1)[1].strip()

    # Commit anything the agent left uncommitted so the gate sees all of it.
    if git("status", "--porcelain"):
        git("add", "-A")
        git("commit", "-m",
            f"{task['id']}: {task['title']} (autonomous, uncommitted remainder)")

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
    made_commits = head != baseline

    if tests_ok and made_commits:
        try:
            git("push", "origin", "main")
            pushed = True
        except RuntimeError as e:
            pushed = False
            log_event("push_failed", task=task["id"], error=str(e)[:200])
        task["status"] = "blocked" if blocked_reason else "done"
        task["finished_at"] = now()
        task["result"] = blocked_reason or "complete"
        task["commits"] = git("rev-list", "--count", f"{baseline}..HEAD")
        save_backlog(data)
        log_event("finished", task=task["id"], status=task["status"],
                  commits=task["commits"], pushed=pushed, note=blocked_reason[:120])
        return 0

    if tests_ok and not made_commits:
        task["status"] = "blocked" if blocked_reason else "failed"
        task["finished_at"] = now()
        task["result"] = blocked_reason or "agent produced no commits"
        save_backlog(data)
        log_event("finished", task=task["id"], status=task["status"],
                  note=task["result"][:160])
        return 0

    # Tests still red: park the work, restore main to its green baseline.
    quarantine = f"failed/{task['id']}-attempt{task['attempts']}"
    git("branch", "-f", quarantine, "HEAD")
    git("reset", "--hard", baseline)
    task["status"] = "failed"
    task["finished_at"] = now()
    task["result"] = f"test gate failed, work parked on {quarantine}"
    save_backlog(data)
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
    print()
    for t in data["tasks"]:
        mark = {"done": "x", "running": ">", "failed": "!", "blocked": "?",
                "pending": " "}.get(t["status"], " ")
        print(f"  [{mark}] {t['id']:6} {t['title']}")
    nxt = next_task(data)
    print(f"\nnext up: {nxt['id']} {nxt['title']}" if nxt else "\nnext up: nothing runnable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.status:
        print_status()
        return 0
    return run_tick(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
