#!/usr/bin/env python3
"""Autonomous conductor: drives the backlog to the end with no human in the loop.

Cron runs `tick` every 20 minutes. Each tick takes at most one step, in this
order, and the first step that has something to do ends the tick:

  1. merge      PRs labelled `approved` whose CI is green; a merged architect
                PR for a backlog task marks that task done
  2. review     an unreviewed worker PR gets an architect review session, which
                may push a cleanup and approves, or rejects to `worker-failed`
  3. work       `worker-ready` issues go to the Modal worker (orchestrate.py run)
  4. takeover   `worker-failed` / `needs-human` issues, or ready issues when the
                Modal budget is spent, are implemented by the architect itself
  5. advance    the next backlog task whose deps are done gets an architect
                session that either does it (COMPLETE -> PR) or writes specs
                with failing tests for the worker (SPLIT -> spec branches and
                issues), and finishes it once those subtasks have merged

Limits. Claude sessions run only while the five-hour window has headroom and
the weekly window is under a pace line that rises through the week, so the
conductor never eats the quota the user needs interactively. Modal work runs
only under the month cap. Both are read fresh every tick from the live
readings, so a window reset or the new billing month is picked up with no
intervention: that is the whole "resume" mechanism.

Trust. Architect sessions are headless Claude Code inside the same nc-worker
image as the worker, on its own throwaway clone, as uid 20001, with internet
(it needs literature) but no GitHub credential and no path to /root. Its output
is a patch; the host applies it as data to its own clone, gates it in a
network-none container, and does every git push and GitHub write itself.

  pipeline/conductor.py tick [--dry-run]
  pipeline/conductor.py status
  touch /root/nc-pipeline-runs/conductor/PAUSE     # stop; rm to resume
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestrate as o  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "nc_runner", o.REPO / "scripts" / "autonomous_runner.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

HOME = o.RUNS / "conductor"
STATE = HOME / "state.json"
PAUSE = HOME / "PAUSE"
CLOCK = HOME / ".lock"
TOKEN_FILE = Path("/root/.claude-anthropic/oauth-token.pinned")
#: Architect backend: "claude" (default) or "codex[:profile]". Written to this
#: file (or NC_ARCH_BACKEND) to hand the architect role to Codex while the
#: Claude weekly pace gate holds (2026-10-02). Delete the file to go back.
ARCH_BACKEND_FILE = HOME / "ARCH_BACKEND"
CODEX_VENDOR = Path("/root/.nvm/versions/node/v24.18.0/lib/node_modules/@openai/codex/"
                    "node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl")
CODEX_PROFILES = Path("/root/.codex-profiles")
CODEX_MODEL = os.environ.get("NC_CODEX_MODEL", "gpt-5.6-luna")
CODEX_BACKOFF_S = 2 * 3600
CODEX_BACKOFF = HOME / "codex_backoff_until"

ARCH_MODEL = os.environ.get("NC_ARCH_MODEL", "opus")
REVIEW_MODEL = os.environ.get("NC_REVIEW_MODEL", "sonnet")
ARCH_TIMEOUT_S = int(os.environ.get("NC_ARCH_TIMEOUT_S", 2 * 3600))
ARCH_MAX_TURNS = int(os.environ.get("NC_ARCH_MAX_TURNS", 250))
MAX_TASK_SESSIONS = int(os.environ.get("NC_MAX_TASK_SESSIONS", 6))
MAX_TAKEOVERS = 2
#: Five-hour window: keep this much for the user.
MIN_SESSION_LEFT = float(os.environ.get("NC_MIN_SESSION_LEFT", 35))
#: Weekly window: used% may not exceed FLOOR + SPAN * (fraction of week elapsed).
WEEK_FLOOR, WEEK_SPAN = 20.0, 55.0
# P22.2a's 1108-line spec overflowed the worker's 64k context on 2026-09-25.
MAX_SPEC_LINES = 200
MODAL_HEADROOM = 1.5            # $ a batch may still cost
# Worker backends in order (2026-10-01): Antigravity (agy, Google-account quota,
# per model), then Gemini by API key. Modal is no longer used by the conductor;
# `orchestrate.py run` without --llm still drives it by hand.
AGY_TASKS = 4
AGY_BACKOFF_S = 2 * 3600
# Second worker backend, used while agy is out of quota: free-tier Gemini
# through the LiteLLM proxy (model entry in pardalos deploy/litellm-free.yaml).
# The free tier allows a few hundred requests a day, so batches are small and a
# quota or backend failure parks it for GEMINI_BACKOFF_S.
GEMINI_LLM = os.environ.get("NC_GEMINI_LLM", "worker-gemini")
GEMINI_TASKS = 2
GEMINI_BACKOFF_S = 3 * 3600
MODAL_BACKOFF_S = 2 * 3600
STALL_H = 36

L_ARCH, L_APPROVED, L_HUMAN = "architect-pr", "approved", "needs-human"

#: What an architect patch may touch. The pipeline, CI, the backlog (host-owned
#: state), the image's dependency pins and data/ stay out of its reach.
ARCH_FORBIDDEN = (".github/", "pipeline/", "tasks/backlog.json", "data/",
                  "requirements.txt", "scripts/autonomous_runner.py")
ARCH_ALLOWED = ("src/", "tests/", "docs/", "tasks/specs/", "scripts/", "examples/",
                "results/", "README.md")


def log(msg: str) -> None:
    o.log(msg)


def now() -> float:
    return time.time()


# -------------------------------------------------------------------- pure parts

def parse_claim(text: str) -> tuple[str, str]:
    """Last TASK_RESULT / REVIEW line: (kind, rest). kind in complete, split,
    blocked, blocked-on-data, done, approve, reject, or ""."""
    kind, rest = "", ""
    for line in (text or "").splitlines():
        s = line.strip().lstrip("*`").rstrip("*`")
        m = re.match(r"^(TASK_RESULT|REVIEW):\s*(BLOCKED_ON_DATA|BLOCKED|COMPLETE|SPLIT|DONE|"
                     r"APPROVE|REJECT)\b[\s:-]*(.*)$", s)
        if m:
            kind = m.group(2).lower().replace("_", "-")
            rest = m.group(3).strip().lstrip("*` ").rstrip("*` ")
    return kind, rest


def split_ids(rest: str, parent: str) -> list[str]:
    ids = re.findall(r"[A-Z]\d+\.\d+[a-z]\w*", rest)
    return [i for i in dict.fromkeys(ids) if i.startswith(parent) and i != parent]


def check_arch_patch(patch: str, protected: list[str] = ()) -> str | None:
    paths = o.patch_paths(patch)
    if not paths:
        return "empty diff"
    bad = [p for p in paths if p.startswith(ARCH_FORBIDDEN) or p in protected]
    if bad:
        return f"patch modifies host-owned paths: {bad}"
    stray = [p for p in paths if not p.startswith(ARCH_ALLOWED)]
    if stray:
        return f"patch touches files outside {', '.join(ARCH_ALLOWED)}: {stray}"
    return None


def check_split_patch(patch: str, subs: list[str]) -> str | None:
    """A SPLIT may add only the subtask specs and their tests."""
    paths = o.patch_paths(patch)
    specs = {f"tasks/specs/{s}.md" for s in subs}
    if not subs:
        return "SPLIT named no subtask ids"
    missing = specs - set(paths)
    if missing:
        return f"SPLIT is missing spec files {sorted(missing)}"
    stray = [p for p in paths if p not in specs and not p.startswith("tests/")]
    if stray:
        return f"a SPLIT may only add specs and tests, not {stray}"
    big = {f: n for f, n in added_lines(patch).items() if f in specs and n > MAX_SPEC_LINES}
    if big:
        return (f"spec(s) too big for the worker's context {big} (max {MAX_SPEC_LINES} lines): "
                "split further or do it yourself")
    return None


def added_lines(patch: str) -> dict[str, int]:
    out, cur = {}, None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur = line.split(" b/", 1)[-1]
            out[cur] = 0
        elif cur and line.startswith("+") and not line.startswith("+++"):
            out[cur] += 1
    return out


def week_allowance(elapsed_frac: float) -> float:
    return min(90.0, WEEK_FLOOR + WEEK_SPAN * max(0.0, min(1.0, elapsed_frac)))


def claude_ok() -> tuple[bool, str]:
    """Five-hour headroom plus a weekly pace line, from the live readings every
    Claude session writes (runner.record_quota_reading). A rolled-over window
    has no constraint, which is how a reset is noticed."""
    try:
        payload = json.loads(runner.QUOTA_CACHE.read_text())
        lim = payload["rate_limits"]
    except Exception:
        return True, "no quota reading yet"
    t = now()
    notes = []
    fh = lim.get("five_hour") or {}
    if fh.get("resets_at") and t < float(fh["resets_at"]):
        left = 100 - float(fh["used_percentage"])
        if left < MIN_SESSION_LEFT:
            return False, f"5h window {left:.0f}% left < {MIN_SESSION_LEFT:.0f}%, resets {time.strftime('%a %H:%M', time.localtime(float(fh['resets_at'])))}"
        notes.append(f"5h {left:.0f}% left")
    wk = lim.get("seven_day") or {}
    if wk.get("resets_at") and t < float(wk["resets_at"]):
        frac = 1 - (float(wk["resets_at"]) - t) / (7 * 86400)
        used, cap = float(wk["used_percentage"]), week_allowance(frac)
        if used > cap and os.environ.get("NC_IGNORE_PACE") != "1":   # one-off manual kick
            return False, f"week {used:.0f}% used > pace {cap:.0f}%"
        notes.append(f"week {used:.0f}%/{cap:.0f}%")
    return True, ", ".join(notes) or "windows rolled over"


def modal_ok() -> tuple[bool, str]:
    try:
        spent = o.modal_month_spend()
    except Exception as e:
        return False, f"modal billing unreadable: {str(e)[:120]}"
    ok = spent + MODAL_HEADROOM <= o.MONTH_CAP
    return ok, f"modal ${spent:.2f} of ${o.MONTH_CAP:.0f} this month"


def gemini_ok(st: dict) -> tuple[bool, str]:
    if not GEMINI_LLM:
        return False, "gemini worker disabled"
    until = st.get("gemini_waiting_until", 0)
    if time.time() < until:
        return False, f"{GEMINI_LLM} backing off until {time.strftime('%H:%M', time.localtime(until))}"
    return True, GEMINI_LLM


def batches_had_infra(batch_dirs: list[Path]) -> bool:
    """True if any of these orchestrate batches stopped on a backend failure."""
    for d in batch_dirs:
        try:
            res = json.loads((d / "batch.json").read_text()).get("results", {})
        except (OSError, ValueError):
            continue
        if "infra" in res.values():
            return True
    return False


def next_task(tasks: list[dict]) -> dict | None:
    done = {t["id"] for t in tasks if t["status"] == "done"}
    for t in tasks:
        if t["status"] in ("done", "blocked", "blocked-on-data"):
            continue
        if (t.get("conductor") or {}).get("gave_up"):
            continue
        if all(d in done for d in t.get("deps", [])):
            return t
    return None


# ------------------------------------------------------------------- state, git

def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"last_progress": now(), "alerts": {}}


def save_state(st: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def progress(st: dict, what: str) -> None:
    st["last_progress"], st["last_progress_what"] = now(), what


def alert(st: dict, key: str, text: str) -> None:
    """Telegram, once per distinct text per key (batched cadence, no spam)."""
    if st["alerts"].get(key) == text:
        return
    st["alerts"][key] = text
    try:
        sys.path.insert(0, str(o.REPO / "scripts"))
        import digest  # noqa: E402
        digest.send(f"neurocascade conductor: {text}")
    except Exception as e:
        log(f"alert not sent ({e}): {text}")


def git(*args, cwd: Path = o.REPO, check=True) -> str:
    return o.sh(["git", "-C", str(cwd), *args], check=check).stdout.strip()


def sync() -> str:
    """Fetch, fast-forward the working repo when it is clean, return origin/main."""
    git("fetch", "-q", "--prune", "origin")
    if not git("status", "--porcelain") and git("rev-parse", "--abbrev-ref", "HEAD") == "main":
        git("merge", "-q", "--ff-only", "origin/main", check=False)
    return git("rev-parse", "origin/main")


def backlog_at(sha: str) -> dict:
    return json.loads(git("show", f"{sha}:tasks/backlog.json"))


def host_clone(name: str, sha: str) -> Path:
    host = o.WORK / "arch" / name
    o.fresh_clone(host, sha)
    return host


def commit_push(host: Path, msg: str, ref: str, force: bool = False) -> None:
    git("-c", "core.hooksPath=/dev/null", "commit", "-q", "-m",
        msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", cwd=host)
    git("push", "-q", *(["-f"] if force else []), "origin", f"HEAD:refs/heads/{ref}", cwd=host)


def update_backlog(msg: str, fn) -> None:
    """Apply fn(tasks) to backlog.json on origin/main and push, retrying races."""
    for _ in range(3):
        base = sync()
        host = host_clone("backlog", base)
        path = host / "tasks" / "backlog.json"
        data = json.loads(path.read_text())
        fn({t["id"]: t for t in data["tasks"]})
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        git("add", "tasks/backlog.json", cwd=host)
        try:
            commit_push(host, f"chore(backlog): {msg} (conductor)", "main")
            sync()
            return
        except subprocess.CalledProcessError:
            time.sleep(5)
    raise RuntimeError(f"could not push backlog update: {msg}")


def cond(t: dict) -> dict:
    return t.setdefault("conductor", {})


# ------------------------------------------------------------------------ github

def gh(*args) -> str:
    p = o.sh(["gh", *args], cwd=o.REPO)
    if p.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])}: {p.stderr[-300:]}")
    return p.stdout.strip()


def open_prs() -> list[dict]:
    return json.loads(gh("pr", "list", "--state", "open", "--limit", "50", "--json",
                         "number,title,headRefName,labels,statusCheckRollup,body") or "[]")


def labels(x: dict) -> set[str]:
    return {lab["name"] for lab in x.get("labels", [])}


def ci_state(pr: dict) -> str:
    checks = pr.get("statusCheckRollup") or []
    if not checks:
        return "pending"
    states = [(c.get("conclusion") or c.get("state") or "").upper() for c in checks]
    if any(s in ("FAILURE", "ERROR", "CANCELLED", "TIMED_OUT") for s in states):
        return "failure"
    if all(s in ("SUCCESS", "NEUTRAL", "SKIPPED") for s in states):
        return "success"
    return "pending"


def issues(label: str) -> list[dict]:
    out = json.loads(gh("issue", "list", "--label", label, "--state", "open",
                        "--json", "number,title,labels", "--limit", "100") or "[]")
    for i in out:
        m = re.match(r"^\[([A-Z]\w*\.\w+)\]", i["title"])
        i["id"] = m.group(1) if m else None
    return sorted([i for i in out if i["id"]], key=lambda i: i["number"])


def open_issue_ids() -> set[str]:
    out = json.loads(gh("issue", "list", "--state", "open", "--json", "title",
                        "--limit", "200") or "[]")
    return {m.group(1) for i in out if (m := re.match(r"^\[([A-Z]\w*\.\w+)\]", i["title"]))}


def ensure_labels() -> None:
    for name, color in ((L_ARCH, "5319e7"), (L_APPROVED, "0e8a16"), (L_HUMAN, "b60205"),
                        (o.LABEL_READY, "1d76db"), (o.LABEL_PR, "fbca04"),
                        (o.LABEL_FAILED, "d93f0b")):
        o.sh(["gh", "label", "create", name, "--color", color, "--force"], cwd=o.REPO)


def arch_backend() -> tuple[str, str]:
    """("claude", "") or ("codex", profile)."""
    raw = os.environ.get("NC_ARCH_BACKEND") or (
        ARCH_BACKEND_FILE.read_text().strip() if ARCH_BACKEND_FILE.exists() else "claude")
    kind, _, prof = raw.partition(":")
    if kind != "codex":
        return "claude", ""
    if not prof:
        r = subprocess.run(["cx", "next"], capture_output=True, text=True, timeout=60)
        prof = (r.stdout.strip().split() or ["main"])[-1]
    return "codex", prof


def arch_ok() -> tuple[bool, str]:
    kind, prof = arch_backend()
    if kind == "claude":
        return claude_ok()
    until = float(CODEX_BACKOFF.read_text()) if CODEX_BACKOFF.exists() else 0
    if now() < until:
        return False, f"codex/{prof} backing off until {time.strftime('%H:%M', time.localtime(until))}"
    if not (CODEX_PROFILES / prof / "auth.json").exists():
        return False, f"codex profile {prof} has no auth.json"
    return True, f"codex architect ({prof}, {CODEX_MODEL})"


# ------------------------------------------------------------- architect session

RULES_OVERRIDE = """
Environment for this session, overriding rules 7 and 8 above:
- The repository is /work, a throwaway clone. data/ is read-only reference data.
- Use `python` (the image has every dependency pinned); there is no venv.
- You have internet for literature (WebSearch/WebFetch) but no GitHub access.
  Do not push, and do not create branches. Committing is optional: the host
  takes the diff of /work against the base commit and applies it itself.
- Never modify .github/, pipeline/, tasks/backlog.json, data/, requirements.txt
  or scripts/autonomous_runner.py; the patch is rejected if you do. The host
  owns task state: do not mark anything done yourself.
- The host gates your patch with `python -m pytest -q -m "not slow"` in a
  network-less container. Run it yourself before finishing.
- Delete scratch files before finishing. Leave no debug output behind.
""".strip()

SPEC_GUIDE = """
HOW TO WORK THIS TASK
You are the architect of a two-tier pipeline. A cheap open-weights worker model
(GLM-4.7-flash on Modal) implements code against specs; you own the science.
Choose per task:

A. Do it yourself, then end with `TASK_RESULT: COMPLETE`. Do this for anything
   that needs judgement: reading and choosing literature values, interpreting a
   result, writing the ledger row or architecture.md reasoning, or when the
   code is small enough that writing a spec would cost as much as the code.

B. Split off mechanical code for the worker, then end with
   `TASK_RESULT: SPLIT <ID>a <ID>b ...` (subtask ids are this task's id plus a
   letter). For each subtask write `tasks/specs/<SUBID>.md` and its failing
   tests, and change NOTHING else (no src, no docs: a SPLIT patch may only add
   specs and tests). Spec format:

       ---
       id: <SUBID>
       title: one line
       tests: [tests/test_<name>.py]
       ---
       Brief: exact function signatures, module paths, and every numeric value
       the code needs WITH its citation (PMID / NDA + review date). The worker
       is told never to invent a number, so give it all of them.

   The tests must fail now, must pass once the brief is implemented, and must
   pin every output field and value: the worker satisfies the tests, not the
   prose. Every other test in the suite must still pass on your spec branch.
   Size: the worker has a 64k-token context that must hold the spec, the tests
   and the code it reads. Keep each spec under 200 lines (the host rejects
   longer ones) and each subtask to one or two src modules; a task that needs
   more is several subtasks, or yours to do.
   Only split subtasks that are independent of each other against current
   main. When they have merged, you get this task back with their ids listed
   and can do the rest, or split again.

Prefer A for literature and judgement, B for self-contained code whose every
input you can pin in the brief. Doing everything yourself is fine when the
split would not save work.

If the honest outcome is that the data do not exist, commit the
searched-and-empty record and ledger row and end with COMPLETE: a documented
negative closes a task here. Use `TASK_RESULT: BLOCKED_ON_DATA <what>` only when
you cannot even establish that, and `TASK_RESULT: BLOCKED <why>` for an
engineering dead end. The last line of your reply must be the TASK_RESULT line.
""".strip()

TASK_PROMPT = """{rules}

{override}

# Task {id} ({phase})

## {title}

{brief}

## Acceptance criteria

{acceptance}
{note}{history}
{guide}
"""

TAKEOVER_PROMPT = """{rules}

{override}

# Take over subtask {id}: {title}

The spec below is yours (you, the architect, wrote it). The worker model could
not get it through the gate, or the spec itself was found unsound. Your tree is
the spec branch. Implement it so the spec tests and the not-slow suite pass.
If the spec itself is wrong (tests that encode a wrong expectation, a missing
value), fix the spec and tests and say why in your final message.

## Spec

{spec}

## What happened so far

{history}

End with exactly one line: `TASK_RESULT: DONE <one sentence>` or
`TASK_RESULT: BLOCKED <reason>`.
"""

REVIEW_PROMPT = """{rules}

{override}

# Review worker PR #{number}: {title}

You wrote the spec below; an open-weights worker implemented it. Your tree is
the PR branch. Review the change against the spec and the project rules:
correctness of the science, values matching the cited sources in the spec, no
invented numbers, no stray debug code or scratch files, tests meaningful.

Small problems: fix them yourself in /work (the host pushes your fix to the
PR). Fundamental problems: reject. CI on this PR: {ci}.

## Spec

{spec}

## Diff against main

```diff
{diff}
```

End with exactly one line: `REVIEW: APPROVE <one sentence>` or
`REVIEW: REJECT <reason the worker's approach cannot be salvaged>`.
"""


def rules() -> str:
    return runner.PROJECT_RULES.format(repo="/work", venv="(none, see below)")


NUDGE = ("Your session ended without the required final line. Background commands "
         "do not survive a headless session: run everything in the foreground "
         "(pass a Bash timeout of up to 45 min for the full suite). Finish the work "
         "that is in /work and end with the TASK_RESULT (or REVIEW) line.")

WIP_NOTE = """
## Work in progress from an earlier session

A previous session on this task stopped before reporting a result. Its
uncommitted changes are already in /work (`git status`, `git diff`). Review
them, keep what is right, and finish from there rather than starting over.
"""

# Headless sessions end when the model yields its turn, so a backgrounded test
# run is simply lost (first live tick, 2026-09-24). No background tasks, and
# a Bash timeout long enough for the full not-slow suite.
SESSION_ENV = {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
               "BASH_DEFAULT_TIMEOUT_MS": str(45 * 60_000),
               "BASH_MAX_TIMEOUT_MS": str(45 * 60_000)}


def _session(cname: str, tree: Path, home: Path, claude_args: list[str],
             log_path: Path, deadline: float) -> int:
    args = ["docker", "run", "--rm", "--name", cname, "--network", "bridge",
            "--memory", "3g", "--memory-swap", "3g", "--cpus", "4", "--pids-limit", "1024",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", str(o.WORKER_UID), "-e", "CLAUDE_CODE_OAUTH_TOKEN",
            *[x for k in SESSION_ENV for x in ("-e", k)],
            "-v", f"{tree}:/work", "-v", f"{o.DATA}:/work/data:ro",
            "-v", f"{home}:/home/worker/.claude", o.IMAGE,
            "claude", *claude_args, "--permission-mode", "bypassPermissions",
            "--max-turns", str(ARCH_MAX_TURNS), "--output-format", "stream-json", "--verbose"]
    # The token goes by environment inheritance, never on a command line.
    env = {**os.environ, **SESSION_ENV,
           "CLAUDE_CODE_OAUTH_TOKEN": TOKEN_FILE.read_text().strip()}
    o.sh(["docker", "rm", "-f", cname])
    # Wall-clock kill even when the session is silent inside a long command.
    killer = threading.Timer(max(60.0, deadline - now()), lambda: o.sh(["docker", "kill", cname]))
    killer.start()
    try:
        with log_path.open("w") as f:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True, env=env, bufsize=1)
            for line in proc.stdout:
                f.write(line)
                if '"rate_limit_event"' in line:
                    try:
                        runner.record_quota_reading(json.loads(line).get("rate_limit_info") or {})
                    except Exception:
                        pass
            return proc.wait()
    finally:
        killer.cancel()


def parse_codex_log(text: str) -> dict:
    """`codex exec --json` events: last agent message, item count, errors."""
    res = {"final": "", "turns": None, "subtype": None, "api_error": None, "thread": None}
    items = 0
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = obj.get("type")
        if typ == "thread.started":
            res["thread"] = obj.get("thread_id")
        elif typ == "item.completed":
            items += 1
            it = obj.get("item") or {}
            if it.get("type") == "agent_message":
                res["final"] = it.get("text") or ""
        elif typ in ("turn.failed", "error"):
            res["api_error"] = str((obj.get("error") or {}).get("message") or obj.get("message") or typ)[:300]
        elif typ == "turn.completed":
            res["api_error"] = None
            res["subtype"] = "success"
    res["turns"] = items if items else None
    return res


def _codex_session(cname: str, tree: Path, home: Path, prof: str, prompt: str,
                   resume: str | None, effort: str, log_path: Path, deadline: float) -> int:
    """One `codex exec` run in the sandbox image. The profile's auth.json is
    copied into a throwaway CODEX_HOME and copied back if Codex refreshed it,
    since a refresh rotates the token and would otherwise log the profile out."""
    auth = CODEX_PROFILES / prof / "auth.json"
    shutil.copy2(auth, home / "auth.json")
    before = (home / "auth.json").read_bytes()
    (home / "config.toml").write_text(
        f'model = "{CODEX_MODEL}"\nmodel_reasoning_effort = "{effort}"\n[tools]\nweb_search = true\n')
    (home / "prompt.txt").write_text(prompt)
    o.sh(["chown", "-R", f"{o.WORKER_UID}:{o.WORKER_UID}", str(home)])
    cmd = ["codex", "exec", *(["resume", resume] if resume else []), "--json",
           "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-"]
    args = ["docker", "run", "--rm", "-i", "--name", cname, "--network", "bridge",
            "--memory", "3g", "--memory-swap", "3g", "--cpus", "4", "--pids-limit", "1024",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", str(o.WORKER_UID), "-w", "/work",
            "-e", "HOME=/tmp", "-e", "CODEX_HOME=/cx",
            "-e", "PATH=/opt/codex/bin:/opt/codex/codex-path:/usr/local/bin:/usr/bin:/bin",
            "-v", f"{tree}:/work", "-v", f"{o.DATA}:/work/data:ro",
            "-v", f"{CODEX_VENDOR}:/opt/codex:ro", "-v", f"{home}:/cx", o.IMAGE, *cmd]
    o.sh(["docker", "rm", "-f", cname])
    killer = threading.Timer(max(60.0, deadline - now()), lambda: o.sh(["docker", "kill", cname]))
    killer.start()
    try:
        with log_path.open("w") as f, (home / "prompt.txt").open() as stdin:
            rc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, stdin=stdin).returncode
    finally:
        killer.cancel()
        after = (home / "auth.json").read_bytes()
        if after != before:
            try:
                json.loads(after)
                tmp = auth.with_suffix(".json.nc-tmp")
                tmp.write_bytes(after)
                tmp.chmod(0o600)
                tmp.replace(auth)
                log(f"codex/{prof}: refreshed auth.json copied back")
            except Exception as e:
                log(f"codex/{prof}: auth.json changed but not copied back: {e}")
    return rc


def run_codex_architect(name: str, base: str, prompt: str, model: str, adir: Path,
                        prof: str, wip: str = "") -> dict:
    """run_architect with Codex in place of Claude Code; same contract."""
    adir.mkdir(parents=True, exist_ok=True)
    host = host_clone(name, base)
    if wip and o.sh(["git", "-C", str(host), "apply", "-"], input=wip).returncode == 0:
        prompt += WIP_NOTE
    (adir / "prompt.txt").write_text(prompt)
    tree = o.WORK / "arch" / f"{name}-tree"
    o.disposable_copy(host, tree)
    o.sh(["git", "-C", str(host), "reset", "-q", "--hard"])
    o.sh(["git", "-C", str(host), "clean", "-qfd"])
    home = adir / "codex-home"
    home.mkdir(exist_ok=True)
    effort = "medium" if model == REVIEW_MODEL else "high"
    cname = f"nca-{re.sub(r'[^a-zA-Z0-9]', '-', name)}"
    t0 = now()
    deadline = t0 + ARCH_TIMEOUT_S
    _codex_session(cname, tree, home, prof, prompt, None, effort, adir / "agent.jsonl", deadline)
    res = parse_codex_log((adir / "agent.jsonl").read_text(errors="replace"))
    rc = 0 if res["subtype"] else 1
    if (res["thread"] and not res["api_error"] and not parse_claim(res["final"])[0]
            and deadline - now() > 600):
        log(f"{name}: no verdict line, resuming codex session once")
        _codex_session(cname, tree, home, prof, NUDGE, res["thread"], effort,
                       adir / "agent-resume.jsonl", deadline)
        more = parse_codex_log((adir / "agent-resume.jsonl").read_text(errors="replace"))
        res.update(api_error=more["api_error"],
                   final=(res["final"] + "\n\n" + more["final"]).strip(),
                   turns=(res["turns"] or 0) + (more["turns"] or 0))
    res.update(rc=rc, model=f"codex/{CODEX_MODEL}", secs=round(now() - t0))
    (adir / "final.txt").write_text(res["final"])
    res["patch"] = o.extract_patch(tree, base, adir)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)
    res["host"] = host
    res["claim"], res["rest"] = parse_claim(res["final"])
    log(f"{name}: codex architect ({prof}) {res['secs']}s items={res['turns']} "
        f"claim={res['claim']} {res['rest'][:80]} files={len(o.patch_paths(res['patch']))}")
    if res["api_error"] or (not res["final"] and res["turns"] is None):
        res["infra"] = True
        # Own file: tick() saves its state dict over anything written here.
        CODEX_BACKOFF.write_text(str(now() + CODEX_BACKOFF_S))
        log(f"codex/{prof}: infra failure ({res['api_error']}), backing off {CODEX_BACKOFF_S // 3600}h")
    return res


def run_architect(name: str, base: str, prompt: str, model: str, adir: Path,
                  wip: str = "") -> dict:
    """One headless Claude session in the sandbox image on a clone at `base`,
    optionally with an earlier session's unfinished patch already applied.
    A session that stops without its verdict line is resumed once."""
    kind, prof = arch_backend()
    if kind == "codex":
        return run_codex_architect(name, base, prompt, model, adir, prof, wip)
    adir.mkdir(parents=True, exist_ok=True)
    host = host_clone(name, base)
    if wip and o.sh(["git", "-C", str(host), "apply", "-"], input=wip).returncode == 0:
        prompt += WIP_NOTE
    (adir / "prompt.txt").write_text(prompt)
    tree = o.WORK / "arch" / f"{name}-tree"
    o.disposable_copy(host, tree)
    o.sh(["git", "-C", str(host), "reset", "-q", "--hard"])
    o.sh(["git", "-C", str(host), "clean", "-qfd"])
    home = adir / "claude-home"
    home.mkdir(exist_ok=True)
    o.sh(["chown", "-R", f"{o.WORKER_UID}:{o.WORKER_UID}", str(home)])
    cname = f"nca-{re.sub(r'[^a-zA-Z0-9]', '-', name)}"
    t0 = now()
    deadline = t0 + ARCH_TIMEOUT_S
    rc = _session(cname, tree, home, ["-p", prompt, "--model", model],
                  adir / "agent.jsonl", deadline)
    text = (adir / "agent.jsonl").read_text(errors="replace")
    res = {"rc": rc, "model": model, **o.parse_agent_log(text)}
    sid = re.search(r'"session_id":\s*"([0-9a-f-]{36})"', text)
    if (sid and not res["api_error"] and not parse_claim(res["final"])[0]
            and deadline - now() > 600):
        log(f"{name}: no verdict line, resuming session once")
        rc = _session(cname, tree, home, ["-p", NUDGE, "--resume", sid.group(1), "--model", model],
                      adir / "agent-resume.jsonl", deadline)
        more = o.parse_agent_log((adir / "agent-resume.jsonl").read_text(errors="replace"))
        res.update(rc=rc, api_error=more["api_error"],
                   final=(res["final"] + "\n\n" + more["final"]).strip(),
                   turns=(res["turns"] or 0) + (more["turns"] or 0))
    res["secs"] = round(now() - t0)
    (adir / "final.txt").write_text(res["final"])
    res["patch"] = o.extract_patch(tree, base, adir)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)
    res["host"] = host
    res["claim"], res["rest"] = parse_claim(res["final"])
    log(f"{name}: architect rc={rc} {res['secs']}s turns={res['turns']} "
        f"claim={res['claim']} {res['rest'][:80]} files={len(o.patch_paths(res['patch']))}")
    if res["api_error"] or (not res["final"] and res["turns"] is None):
        # Quota hit, auth, or the session never started: not the task's fault.
        res["infra"] = True
    return res


def gate(host: Path, adir: Path, tests: list[str] = ()) -> tuple[bool, str]:
    script = ""
    if tests:
        script = f"python -m pytest -q -p no:cacheprovider {' '.join(tests)} && "
    script += "python -m pytest -q -p no:cacheprovider -m 'not slow'"
    return o.sandbox_pytest(host, script, f"nca-gate-{adir.name}", adir / "pytest.txt")


def rundir(tid: str, mode: str) -> Path:
    return HOME / tid / f"{time.strftime('%Y%m%d-%H%M%S')}-{mode}"


def history_text(c: dict) -> str:
    h = c.get("history") or []
    if not h:
        return ""
    return "\n## Previous conductor sessions on this task\n\n" + "\n".join(
        f"- {x}" for x in h[-6:]) + "\n"


# ------------------------------------------------------------------------ steps

def step_merge(st: dict, prs: list[dict], dry: bool) -> bool:
    for pr in prs:
        if L_APPROVED not in labels(pr):
            continue
        state = ci_state(pr)
        if state == "failure":
            log(f"PR #{pr['number']}: approved but CI failed; back to review")
            if not dry:
                gh("pr", "edit", str(pr["number"]), "--remove-label", L_APPROVED)
                gh("pr", "comment", str(pr["number"]), "--body",
                   "CI failed after approval; the conductor will review again.")
            return True
        if state != "success":
            continue
        log(f"PR #{pr['number']}: CI green, merging")
        if dry:
            return True
        gh("pr", "merge", str(pr["number"]), "--merge", "--delete-branch")
        tid = pr["headRefName"].split("/", 1)[-1]
        if pr["headRefName"].startswith("architect/") and "Backlog-Task: " + tid in (pr.get("body") or ""):
            def done(ts, tid=tid, n=pr["number"]):
                t = ts[tid]
                t.update(status="done", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                         result="complete")
                cond(t)["pr"] = n
            update_backlog(f"{tid} done via PR #{pr['number']}", done)
            alert(st, f"done-{tid}", f"{tid} done (PR #{pr['number']} merged)")
        progress(st, f"merged PR #{pr['number']}")
        return True
    return False


def step_review(st: dict, prs: list[dict], dry: bool) -> bool:
    for pr in prs:
        lab = labels(pr)
        if o.LABEL_PR not in lab or lab & {L_APPROVED, L_HUMAN}:
            continue
        ci = ci_state(pr)
        if ci == "pending":
            continue
        sid = pr["headRefName"].split("/", 1)[-1]
        ok, why = arch_ok()
        if not ok:
            st["waiting"] = f"review PR #{pr['number']}: {why}"
            return True
        log(f"PR #{pr['number']}: architect review (CI {ci})")
        if dry:
            return True
        git("fetch", "-q", "origin", pr["headRefName"], f"spec/{sid}")
        head = git("rev-parse", f"origin/{pr['headRefName']}")
        spec_text = git("show", f"origin/spec/{sid}:tasks/specs/{sid}.md", check=False)
        spec = o.parse_spec(spec_text) if spec_text else {"tests": [], "protected": []}
        diff = git("diff", f"origin/main...{head}")[:60000]
        adir = rundir(sid, f"review-{pr['number']}")
        res = run_architect(f"review-{sid}", head, REVIEW_PROMPT.format(
            rules=rules(), override=RULES_OVERRIDE, number=pr["number"], title=pr["title"],
            ci=ci, spec=spec_text or "(no spec branch)", diff=diff), REVIEW_MODEL, adir)
        if res.get("infra"):
            st["waiting"] = f"review session failed to run: {res['final'][:120]}"
            return True
        if res["claim"] == "approve":
            host = res["host"]
            if res["patch"].strip():
                reason = (check_arch_patch(res["patch"], spec.get("protected", []))
                          or o.apply_patch(host, head, res["patch"]))
                if not reason:
                    passed, tail = gate(host, adir, spec.get("tests", []))
                    reason = None if passed else f"review cleanup failed the gate:\n{tail[-1500:]}"
                if reason:
                    gh("pr", "comment", str(pr["number"]), "--body",
                       f"Architect cleanup rejected by the host: {reason[:1500]}")
                    gh("pr", "edit", str(pr["number"]), "--add-label", L_HUMAN)
                    alert(st, f"review-{pr['number']}", f"PR #{pr['number']} cleanup failed; labelled needs-human")
                    return True
                commit_push(host, f"{sid}: architect review cleanup", pr["headRefName"])
            gh("pr", "edit", str(pr["number"]), "--add-label", L_APPROVED)
            gh("pr", "comment", str(pr["number"]), "--body",
               f"Architect review: APPROVE. {res['rest'][:500]}")
            progress(st, f"approved PR #{pr['number']}")
        else:
            reason = res["rest"] or f"no verdict ({res['claim'] or 'none'})"
            gh("pr", "close", str(pr["number"]), "--comment", f"Architect review: REJECT. {reason[:1500]}")
            num = find_issue(sid)
            if num:
                gh("issue", "edit", str(num), "--add-label", o.LABEL_FAILED,
                   "--remove-label", o.LABEL_PR)
                gh("issue", "comment", str(num), "--body", f"Worker PR rejected in review: {reason[:1500]}")
            progress(st, f"rejected PR #{pr['number']}")
        return True
    return False


def find_issue(sid: str) -> int | None:
    out = json.loads(gh("issue", "list", "--state", "open", "--search", f"[{sid}] in:title",
                        "--json", "number,title") or "[]")
    for i in out:
        if i["title"].startswith(f"[{sid}]"):
            return i["number"]
    return None


def step_work(st: dict, dry: bool) -> bool:
    ready = issues(o.LABEL_READY)
    if not ready:
        return False
    until = st.get("agy_waiting_until", 0)
    if time.time() >= until:
        llm, why = o.AGY, "antigravity"
    else:
        why = f"agy backing off until {time.strftime('%H:%M', time.localtime(until))}"
        gok, gwhy = gemini_ok(st)
        if not gok:
            st["worker_waiting"] = f"{why}; {gwhy}"
            log(f"work: {why}; {gwhy}")
            return False      # the architect may take these over instead
        llm = GEMINI_LLM
    st.pop("worker_waiting", None)
    st.pop("modal_waiting", None)
    if o.mem_available_mb() < o.MIN_FREE_MB:
        st["waiting"] = "host memory low"
        return True
    log(f"work: {len(ready)} ready issue(s) to {llm} ({why})")
    if dry:
        return True
    cmd = [sys.executable, str(Path(__file__).parent / "orchestrate.py"), "run",
           "--max-tasks", str(AGY_TASKS if llm == o.AGY else GEMINI_TASKS), "--llm", llm]
    started = time.time()
    p = o.sh(cmd, capture_output=False, text=True, timeout=6 * 3600)
    new = [d for d in (o.RUNS / "_batches").glob("*") if d.stat().st_mtime >= started - 5]
    if not new:
        # Nothing was runnable (spec failed to load or baseline). Not progress:
        # counting it would hide a stuck queue from the stall alert.
        log(f"worker batch ({llm}) ran nothing, rc={p.returncode}")
        stuck = st.setdefault("work_stuck", 0) + 1
        st["work_stuck"] = stuck
        if stuck == 3:
            alert(st, "work", f"worker queue stuck: {len(ready)} ready issue(s) but no runnable task "
                      "for 3 ticks; see conductor/cron.log")
        return False
    st.pop("work_stuck", None)
    progress(st, f"worker batch ({llm}) rc={p.returncode}")
    if llm == o.AGY:
        if batches_had_infra(new):
            # agy out of quota or broken: Gemini covers until the backoff ends.
            st["agy_waiting_until"] = time.time() + AGY_BACKOFF_S
            log(f"work: agy backend failure, backing off {AGY_BACKOFF_S // 3600}h")
        else:
            st.pop("agy_waiting_until", None)
    else:
        if batches_had_infra(new):
            st["gemini_waiting_until"] = time.time() + GEMINI_BACKOFF_S
            log(f"work: {llm} backend failure, backing off {GEMINI_BACKOFF_S // 3600}h")
        else:
            st.pop("gemini_waiting_until", None)
    return True


def step_takeover(st: dict, dry: bool) -> bool:
    cands = issues(o.LABEL_FAILED) + issues(L_HUMAN)
    if st.get("worker_waiting"):
        cands += issues(o.LABEL_READY)
    cands = [i for i in cands if st.setdefault("takeovers", {}).get(i["id"], 0) < MAX_TAKEOVERS]
    if not cands:
        return False
    issue = cands[0]
    sid = issue["id"]
    ok, why = arch_ok()
    if not ok:
        st["waiting"] = f"takeover {sid}: {why}"
        return True
    log(f"takeover {sid} (#{issue['number']})")
    if dry:
        return True
    git("fetch", "-q", "origin", f"spec/{sid}")
    base = git("rev-parse", f"origin/spec/{sid}")
    spec_text = git("show", f"{base}:tasks/specs/{sid}.md")
    spec = o.parse_spec(spec_text)
    comments = json.loads(gh("issue", "view", str(issue["number"]), "--json", "comments"))["comments"]
    hist = "\n\n".join(c["body"][:1500] for c in comments[-3:]) or "(no comments)"
    adir = rundir(sid, "takeover")
    res = run_architect(f"takeover-{sid}", base, TAKEOVER_PROMPT.format(
        rules=rules(), override=RULES_OVERRIDE, id=sid, title=spec["title"],
        spec=spec_text, history=hist), ARCH_MODEL, adir)
    if res.get("infra"):
        st["waiting"] = f"takeover session failed to run: {res['final'][:120]}"
        return True
    st["takeovers"][sid] = st["takeovers"].get(sid, 0) + 1
    host = res["host"]
    reason = None if res["claim"] == "done" else f"architect ended {res['claim'] or 'without a verdict'}: {res['rest'][:300]}"
    reason = reason or check_arch_patch(res["patch"]) or o.apply_patch(host, base, res["patch"])
    if not reason:
        passed, tail = gate(host, adir, spec["tests"])
        reason = None if passed else f"gate failed:\n{tail[-1500:]}"
    if reason:
        gh("issue", "comment", str(issue["number"]), "--body", f"Architect takeover failed: {reason[:1800]}")
        if st["takeovers"][sid] >= MAX_TAKEOVERS:
            alert(st, f"takeover-{sid}", f"{sid}: architect takeover failed {MAX_TAKEOVERS}x, needs a look")
        progress(st, f"takeover {sid} failed")
        return True
    branch = f"architect/{sid}"
    commit_push(host, f"{sid}: {spec['title']}\n\nArchitect takeover of the worker spec.\n\nCloses #{issue['number']}", branch, force=True)
    gh("pr", "create", "--base", "main", "--head", branch, "--title", f"[{sid}] {spec['title']}",
       "--body", f"Closes #{issue['number']}\n\nArchitect takeover ({res['model']}).\n\n{res['final'][-2500:]}\n\n"
       "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
       "--label", L_ARCH, "--label", L_APPROVED)
    for lab in (o.LABEL_FAILED, L_HUMAN, o.LABEL_READY):
        o.sh(["gh", "issue", "edit", str(issue["number"]), "--remove-label", lab], cwd=o.REPO)
    progress(st, f"takeover PR for {sid}")
    return True


def step_advance(st: dict, base: str, dry: bool) -> bool:
    tasks = backlog_at(base)["tasks"]
    if all(t["status"] == "done" for t in tasks):
        alert(st, "complete", "every backlog task is done; the product roadmap is complete")
        st["complete"] = True
        return True
    t = next_task(tasks)
    if t is None:
        stuck = [x["id"] for x in tasks if x["status"] != "done"][:5]
        alert(st, "stuck", f"no runnable task; unfinished: {', '.join(stuck)} (blocked or given up)")
        return False
    c = t.get("conductor") or {}
    open_ids = open_issue_ids()
    subs_open = [s for s in c.get("subtasks", []) if s in open_ids]
    if subs_open:
        st["waiting_on"] = f"{t['id']} subtasks in flight: {', '.join(subs_open)}"
        return False
    if any(p["headRefName"] == f"architect/{t['id']}" for p in open_prs()):
        st["waiting_on"] = f"{t['id']} PR awaiting CI/merge"
        return False
    ok, why = arch_ok()
    if not ok:
        st["waiting"] = f"advance {t['id']}: {why}"
        return True
    tid = t["id"]
    log(f"advance {tid}: architect session ({why})")
    if dry:
        return True
    note = f"\n## Notes carried on this task\n\n{t['note']}\n" if t.get("note") else ""
    if c.get("subtasks"):
        note += ("\n## Subtasks already merged into main\n\n" + ", ".join(c["subtasks"])
                 + "\nTheir specs are in tasks/specs/. Build on them.\n")
    adir = rundir(tid, "task")
    wip_path = Path(st.setdefault("wip", {}).get(tid, "/nonexistent"))
    wip = wip_path.read_text(errors="replace") if wip_path.is_file() else ""
    res = run_architect(f"task-{tid}", base, TASK_PROMPT.format(
        rules=rules(), override=RULES_OVERRIDE, id=tid, phase=t.get("phase", ""),
        title=t["title"], brief=t["brief"],
        acceptance="\n".join(f"- {a}" for a in t.get("acceptance", [])),
        note=note, history=history_text(c), guide=SPEC_GUIDE), ARCH_MODEL, adir, wip)
    # Unfinished work is kept for the next session; anything else starts clean.
    if res["patch"].strip() and not res["claim"]:
        (adir / "wip.patch").write_text(res["patch"])
        st["wip"][tid] = str(adir / "wip.patch")
    elif res["claim"]:
        st["wip"].pop(tid, None)
    if res.get("infra"):
        st["waiting"] = f"{tid} session failed to run: {res['final'][:120] or res['rc']}"
        return True

    host, claim = res["host"], res["claim"]
    outcome, reason = claim or "no verdict", None
    if claim == "complete":
        reason = (check_arch_patch(res["patch"])
                  or ("COMPLETE but nothing substantive changed"
                      if not runner.substantive_paths(o.patch_paths(res["patch"])) else None)
                  or o.apply_patch(host, base, res["patch"]))
        if not reason:
            passed, tail = gate(host, adir)
            reason = None if passed else f"gate failed:\n{tail[-1200:]}"
        if not reason:
            branch = f"architect/{tid}"
            commit_push(host, f"{tid}: {t['title']}\n\nArchitect session ({res['model']}), "
                        f"{res['turns']} turns.", branch, force=True)
            gh("pr", "create", "--base", "main", "--head", branch, "--title", f"[{tid}] {t['title']}",
               "--body", f"Backlog-Task: {tid}\n\n{res['final'][-3500:]}\n\n"
               "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
               "--label", L_ARCH, "--label", L_APPROVED)
            outcome = "COMPLETE, PR opened"
    elif claim == "split":
        subs = split_ids(res["rest"], tid)
        reason = check_split_patch(res["patch"], subs) or o.apply_patch(host, base, res["patch"])
        if not reason:
            reason = publish_specs(host, base, subs, adir)
        if not reason:
            outcome = f"SPLIT into {', '.join(subs)}"
            c.setdefault("subtasks", [])
            c["subtasks"] = list(dict.fromkeys(c["subtasks"] + subs))
    elif claim in ("blocked", "blocked-on-data"):
        outcome = f"{claim}: {res['rest'][:200]}"
    else:
        reason = "session ended without a TASK_RESULT line"

    entry = time.strftime("%Y-%m-%d %H:%M") + f" {outcome}" + (f"; rejected: {reason[:600]}" if reason else "")
    new_subs = c.get("subtasks", [])

    def rec(ts):
        x = cond(ts[tid])
        x["sessions"] = x.get("sessions", 0) + 1
        x.setdefault("history", []).append(entry)
        if new_subs:
            x["subtasks"] = new_subs
        if ts[tid]["status"] == "failed":
            ts[tid]["status"] = "pending"
        if claim in ("blocked", "blocked-on-data") and not reason:
            ts[tid]["status"] = claim
            ts[tid]["result"] = res["rest"][:400]
        if x["sessions"] >= MAX_TASK_SESSIONS and ts[tid]["status"] != "done":
            x["gave_up"] = True
    update_backlog(f"{tid}: {outcome[:60]}", rec)
    if claim in ("blocked", "blocked-on-data") or reason:
        alert(st, f"task-{tid}", f"{tid}: {entry[:300]}")
    progress(st, f"{tid}: {outcome[:80]}")
    return True


def publish_specs(host: Path, base: str, subs: list[str], adir: Path) -> str | None:
    """One spec/<SUB> branch per subtask holding only its spec and tests, then
    a worker-ready issue each. The baseline check runs in orchestrate."""
    git("-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "split", cwd=host)
    full = git("rev-parse", "HEAD", cwd=host)
    for sub in subs:
        spec = o.parse_spec((host / "tasks" / "specs" / f"{sub}.md").read_text())
        if spec["id"] != sub:
            return f"tasks/specs/{sub}.md declares id {spec['id']}"
        files = [f"tasks/specs/{sub}.md"] + sorted({t.split("::")[0] for t in spec["tests"]})
        git("checkout", "-q", "--detach", base, cwd=host)
        git("checkout", full, "--", *files, cwd=host)
        commit_push(host, f"spec: {sub} {spec['title']}", f"spec/{sub}", force=True)
        body = (f"Spec: `tasks/specs/{sub}.md` on `spec/{sub}`.\nTests: "
                + ", ".join(f"`{x}`" for x in spec["tests"]) + "\n\nWritten by the conductor's architect session.")
        if not find_issue(sub):
            gh("issue", "create", "--title", f"[{sub}] {spec['title']}", "--body", body,
               "--label", o.LABEL_READY)
    return None


# -------------------------------------------------------------------------- tick

def tick(dry: bool) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    lock = CLOCK.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    st = load_state()
    st["last_tick"] = now()
    for k in ("waiting", "waiting_on"):
        st.pop(k, None)
    try:
        if PAUSE.exists():
            st["waiting"] = "paused by PAUSE file"
            return
        if o.lock_held():
            st["waiting"] = "orchestrate.py batch running"
            return
        base = sync()
        prs = open_prs()
        (step_merge(st, prs, dry) or step_review(st, prs, dry) or step_work(st, dry)
         or step_takeover(st, dry) or step_advance(st, base, dry))
        limit_wait = any(w in (st.get("waiting") or "") for w in ("window", "week", "modal"))
        if not limit_wait and now() - st.get("last_progress", now()) > STALL_H * 3600:
            alert(st, "stall", f"no progress for {STALL_H}h; last: {st.get('last_progress_what')}; "
                  f"now: {st.get('waiting') or st.get('waiting_on') or 'idle'}")
    except Exception as e:
        log(f"tick error: {e!r}")
        st["last_error"] = f"{time.strftime('%F %T')} {e!r}"[:500]
        alert(st, "error", f"tick error: {str(e)[:300]}")
    finally:
        log("state: " + (st.get("waiting") or st.get("waiting_on") or st.get("last_progress_what") or "-"))
        save_state(st)


def status() -> None:
    st = load_state()
    base = sync()
    tasks = backlog_at(base)["tasks"]
    done = sum(t["status"] == "done" for t in tasks)
    print(f"backlog {done}/{len(tasks)} done; next: {(next_task(tasks) or {}).get('id')}")
    until = st.get("agy_waiting_until", 0)
    print(f"architect: {arch_ok()[1]}\nclaude: {claude_ok()[1]}\nagy: " + (f"backing off until {time.strftime('%H:%M', time.localtime(until))}"
          if time.time() < until else "ready (worker 1)"))
    print(f"gemini: {gemini_ok(load_state())[1]} (worker 2)")
    for k in ("waiting", "waiting_on", "last_progress_what", "last_error"):
        if st.get(k):
            print(f"{k}: {st[k]}")
    if st.get("last_progress"):
        print(f"last progress {(now() - st['last_progress']) / 3600:.1f}h ago; "
              f"last tick {(now() - st.get('last_tick', 0)) / 60:.0f}m ago")
    print("paused" if PAUSE.exists() else "running")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("labels")
    a = ap.parse_args()
    if a.cmd == "tick":
        tick(a.dry_run)
    elif a.cmd == "labels":
        ensure_labels()
    else:
        status()


if __name__ == "__main__":
    main()
