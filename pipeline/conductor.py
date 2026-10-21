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
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guards as g  # noqa: E402
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
#: "deepseek[:model]" runs the same Claude Code session against DeepSeek's
#: Anthropic-compatible API (2026-10-12, both Codex accounts down).
ARCH_BACKEND_FILE = HOME / "ARCH_BACKEND"
DEEPSEEK_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_MIN_BALANCE = 0.10
DEEPSEEK_BACKOFF = HOME / "deepseek_backoff_until"
CODEX_VENDOR = Path("/root/.nvm/versions/node/v24.18.0/lib/node_modules/@openai/codex/"
                    "node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl")
CODEX_PROFILES = Path("/root/.codex-profiles")
CODEX_MODEL = os.environ.get("NC_CODEX_MODEL", "gpt-5.6-luna")
CODEX_BACKOFF_S = 2 * 3600
CODEX_BACKOFF = HOME / "codex_backoff_until"

ARCH_MODEL = os.environ.get("NC_ARCH_MODEL", "opus")
REVIEW_MODEL = os.environ.get("NC_REVIEW_MODEL", "sonnet")
ARCH_TIMEOUT_S = int(os.environ.get("NC_ARCH_TIMEOUT_S", 2 * 3600))
#: The acceptance reviewer reads a diff and renders a verdict; healthy runs
#: finish well under an hour, so a tighter cap stops a stuck reviewer eating
#: the full task budget (the rc=137 7200s loops).
REVIEW_TIMEOUT_S = int(os.environ.get("NC_REVIEW_TIMEOUT_S", 3600))
ARCH_MAX_TURNS = int(os.environ.get("NC_ARCH_MAX_TURNS", 250))
MAX_TASK_SESSIONS = int(os.environ.get("NC_MAX_TASK_SESSIONS", 6))
MAX_TAKEOVERS = 2
#: Independent acceptance review: a task whose PR is rejected this often is
#: marked blocked and alerted rather than retried.
MAX_REVIEW_REJECTS = int(os.environ.get("NC_MAX_REVIEW_REJECTS", 3))
#: One line per host gate run on an architect patch: the gate is the only
#: place held-out truth is read, so this counts held-out evaluations per task.
HELDOUT_LOG = HOME / "heldout-evals.jsonl"
PUBMED_CACHE = HOME / "pubmed-cache.json"
#: Five-hour window: keep this much for the user.
MIN_SESSION_LEFT = float(os.environ.get("NC_MIN_SESSION_LEFT", 35))
#: Weekly window: used% may not exceed FLOOR + SPAN * (fraction of week elapsed).
WEEK_FLOOR, WEEK_SPAN = 20.0, 55.0
# P22.2a's 1108-line spec overflowed the worker's 64k context on 2026-09-25.
MAX_SPEC_LINES = 200
MODAL_HEADROOM = 1.5            # $ a batch may still cost
# Worker backends in order (2026-10-18): Modal GLM (orchestrate's default burst
# model) while the month is under the cap, then Antigravity (agy, Google-account
# quota, per model), then Gemini by API key. NC_MODAL_WORKER=0 drops Modal.
MODAL_WORKER = os.environ.get("NC_MODAL_WORKER", "1") != "0"
MODAL_TASKS = 4
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
                  "requirements.txt", "scripts/autonomous_runner.py", g.SPLIT)
ARCH_ALLOWED = ("src/", "tests/", "docs/", "tasks/specs/", "scripts/", "examples/",
                "results/", "README.md")


def log(msg: str) -> None:
    o.log(msg)


def now() -> float:
    return time.time()


# -------------------------------------------------------------------- pure parts

def parse_claim(text: str) -> tuple[str, str]:
    """Last TASK_RESULT / REVIEW line: (kind, rest). kind in complete, split,
    empty, blocked, blocked-on-data, done, approve, reject, or ""."""
    kind, rest = "", ""
    for line in (text or "").splitlines():
        s = line.strip().lstrip("*`").rstrip("*`")
        m = re.match(r"^(TASK_RESULT|REVIEW):\s*(BLOCKED_ON_DATA|BLOCKED|COMPLETE|EMPTY|SPLIT|DONE|"
                     r"APPROVE|REJECT)\b[\s:-]*(.*)$", s)
        if m:
            kind = m.group(2).lower().replace("_", "-")
            rest = m.group(3).strip().lstrip("*` ").rstrip("*` ")
    return kind, rest


EMPTY_MARK = "Backlog-Result: searched-and-empty: "


def empty_result(body: str) -> str | None:
    """The missing-data note of an EMPTY PR body, or None for a real result."""
    for line in body.splitlines():
        if line.startswith(EMPTY_MARK):
            return line[len(EMPTY_MARK):].strip()
    return None


def review_verdict(text: str, n_bullets: int, checks=("thresholds", "heldout", "tests", "citations")
                   ) -> tuple[str | None, str]:
    """Acceptance review outcome: ("approve" | "reject" | None, findings).
    APPROVE counts only with a PASS line for every acceptance bullet and every
    check; any FAIL line makes it a rejection whatever the last line says.
    None (no verdict, or an APPROVE missing lines) is a failed review session,
    not a rejection of the author's work."""
    claim, rest = parse_claim(text)
    got: dict[str, tuple[str, str]] = {}
    for line in (text or "").splitlines():
        s = line.strip().lstrip("-*` ").replace("*", "").replace("`", "")
        m = re.match(r"^(ACCEPTANCE\s+\d+|CHECK\s+\w+)\s*:\s*(PASS|FAIL)\b[\s:-]*(.*)$", s, re.I)
        if m:
            got[re.sub(r"\s+", " ", m.group(1).lower())] = (m.group(2).upper(), m.group(3).strip())
    want = [f"acceptance {k}" for k in range(1, n_bullets + 1)] + [f"check {c}" for c in checks]
    fails = [f"{k.upper()}: FAIL {why}"[:700] for k, (v, why) in got.items() if v == "FAIL"]
    if claim == "reject" or fails:
        return "reject", "\n".join(fails + ([f"REVIEW: REJECT {rest}"[:700]] if claim == "reject" else []))
    if claim == "approve" and all(got.get(k, ("",))[0] == "PASS" for k in want):
        return "approve", rest
    return None, f"no complete verdict (claim={claim or 'none'}, missing {[k for k in want if k not in got]})"


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


#: Statuses that need no more work. "optional" tasks (2026-10-12, P31.1) are
#: never started and never hold up completion or the stuck alert.
FINISHED = ("done", "optional")


def next_task(tasks: list[dict]) -> dict | None:
    done = {t["id"] for t in tasks if t["status"] == "done"}
    for t in tasks:
        if t["status"] in ("done", "optional", "blocked", "blocked-on-data"):
            continue
        c = t.get("conductor") or {}
        if c.get("gave_up") or c.get("review_rejections", 0) >= MAX_REVIEW_REJECTS:
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


def pr_label(num, add: str = "", remove: str = "") -> None:
    """Label a PR through the REST issues API. `gh pr edit` on gh 2.45 also
    queries classic projectCards, which GitHub now rejects (2026-10-15)."""
    if add:
        gh("api", "-X", "POST", f"repos/{{owner}}/{{repo}}/issues/{num}/labels", "-f", f"labels[]={add}")
    if remove:
        gh("api", "-X", "DELETE", f"repos/{{owner}}/{{repo}}/issues/{num}/labels/{remove}")


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
    """("claude", ""), ("codex", profile) or ("deepseek", model)."""
    return parse_backend(os.environ.get("NC_ARCH_BACKEND") or (
        ARCH_BACKEND_FILE.read_text().strip() if ARCH_BACKEND_FILE.exists() else "claude"))


def review_backend() -> tuple[str, str, str]:
    """(kind, profile, model) of the acceptance reviewer, never the author's
    own model. A different vendor is the stronger check and is used whenever
    the architect is not Claude. A Claude architect gets the other Claude
    model by default; NC_ACCEPT_REVIEWER (claude:<model>, codex:<profile>,
    deepseek:<model>) overrides, e.g. codex for a cross-vendor check."""
    raw = os.environ.get("NC_ACCEPT_REVIEWER", "")
    if raw:
        kind, _, x = raw.partition(":")
        if kind == "claude":
            return "claude", "", x or "opus"
        kind, prof = parse_backend(raw)
        return kind, prof, prof if kind == "deepseek" else CODEX_MODEL
    if arch_backend()[0] != "claude":
        return "claude", "", "opus"
    return "claude", "", "sonnet" if ARCH_MODEL == "opus" else "opus"


def parse_backend(raw: str) -> tuple[str, str]:
    kind, _, prof = raw.partition(":")
    if kind == "deepseek":
        return "deepseek", prof or DEEPSEEK_MODEL
    if kind != "codex":
        return "claude", ""
    if not prof:
        r = subprocess.run(["cx", "next"], capture_output=True, text=True, timeout=60)
        prof = (r.stdout.strip().split() or ["main"])[-1]
    return "codex", prof


def arch_ok() -> tuple[bool, str]:
    return backend_ok(*arch_backend())


def backend_ok(kind: str, prof: str) -> tuple[bool, str]:
    if kind == "claude":
        return claude_ok()
    if kind == "deepseek":
        return deepseek_ok(prof)
    until = float(CODEX_BACKOFF.read_text()) if CODEX_BACKOFF.exists() else 0
    if now() < until:
        return False, f"codex/{prof} backing off until {time.strftime('%H:%M', time.localtime(until))}"
    if not (CODEX_PROFILES / prof / "auth.json").exists():
        return False, f"codex profile {prof} has no auth.json"
    return True, f"codex architect ({prof}, {CODEX_MODEL})"


def deepseek_key() -> str:
    r = subprocess.run(["bw-get", "DeepSeek"], capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def deepseek_ok(model: str) -> tuple[bool, str]:
    until = float(DEEPSEEK_BACKOFF.read_text()) if DEEPSEEK_BACKOFF.exists() else 0
    if now() < until:
        return False, f"deepseek backing off until {time.strftime('%H:%M', time.localtime(until))}"
    key = deepseek_key()
    if not key:
        return False, "deepseek: no key from vault (bw-get DeepSeek)"
    try:
        req = urllib.request.Request("https://api.deepseek.com/user/balance",
                                     headers={"Authorization": f"Bearer {key}"})
        info = json.loads(urllib.request.urlopen(req, timeout=20).read())
        bal = sum(float(b["total_balance"]) for b in info["balance_infos"] if b["currency"] == "USD")
    except Exception as e:
        return False, f"deepseek balance check failed: {e!r}"[:200]
    if bal < DEEPSEEK_MIN_BALANCE:
        return False, f"deepseek balance ${bal:.2f} < ${DEEPSEEK_MIN_BALANCE:.2f}, top up"
    return True, f"deepseek architect ({model}, ${bal:.2f} left)"


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
- The held-out split is frozen. Never modify src/neurocascade/benchmark/split.json,
  a held-out drug's truth file, truth/effect_labels.json or the p29_5 truth
  files; the patch is rejected if you do. Held-out drugs' truth is absent
  from /work (and git history here starts at one commit), so held-out tests
  fail in /work: that is expected. The host scores held-out drugs with the
  full data when it gates your patch, and logs every such run.
- **The absence of held-out truth in /work is never, by itself, a reason to
  return `TASK_RESULT: EMPTY`.** Implement the scoring path so that it reports
  `searched-and-empty` with every held-out drug named when truth is absent and
  scores normally when it is present, write the `results/<task id>.json` the
  host reads, and return COMPLETE; the host runs the same function with the
  full data. `EMPTY` is for data that does not exist anywhere — not for data
  this sandbox is deliberately built without. P30.1 (2026-10-18) and P30.2
  (2026-10-19) were both parked on exactly this misreading, and each cost a
  full session plus a gate run.
- **What the host actually does with `results/<task id>.json`:** it reads
  `n_scored` and nothing else; it does not run your scoring function. So when a
  task's headline needs held-out truth, also report the largest honest
  available-data number there (e.g. leave-one-drug-out over train+held-out
  drugs whose truth is present) with `n_scored > 0`, record the held-out split
  as host-pending inside the same file or the PR body, and return COMPLETE. A
  `{"n_scored": 0}` file downgrades a COMPLETE to EMPTY no matter how good the
  code is.
- Every PMID your patch adds to benchmark JSON is checked against PubMed by
  the host: it must exist and its title, abstract or MeSH terms must name
  the record's drug; otherwise add a "pmid_note" to the record saying why.
- A patch that adds a row to docs/validation-ledger.md must also add
  results/<task id>.json holding {"n_scored": <int>}, the number of drugs
  actually scored. The host reads n from it: n_scored 0 is EMPTY, not COMPLETE.
- An independent reviewer checks your PR against the acceptance criteria
  before it can merge.
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

COMPLETE means the acceptance criteria are met. If the honest outcome is that
the data do not exist (the scored deliverable has n=0, or too few drugs to meet
the acceptance count), commit the searched-and-empty record and ledger row and
end with `TASK_RESULT: EMPTY <exactly which data are missing>`. The host also
turns any ledger-row patch whose results/<task id>.json says n_scored 0 into
EMPTY, whatever the final line says. The record
merges, but the task stays blocked-on-data and its dependents wait, so never
call an n=0 result COMPLETE. Use `TASK_RESULT: BLOCKED_ON_DATA <what>` only when
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

ACCEPT_PROMPT = """{rules}

# Independent acceptance review of PR #{number}: [{id}] {title}

Another model wrote this PR. You did not, and you owe it nothing: your job is
to find the reasons it should not merge. Approve only what the evidence in
/work shows.

/work is the PR branch, READ-ONLY (write scratch files under /tmp). data/ is
read-only reference data. Held-out drugs' truth is absent from /work, so
held-out tests fail here; the host scored them with the full data.
You have internet for literature but no GitHub access.

## Task brief

{brief}

## Acceptance criteria

{acceptance}

## Host checks

{host}

## The patch (against main)

```diff
{diff}
```

Check every acceptance criterion against the code, data and test output in
/work, not against the PR's own description of itself. Also check:
  thresholds: no acceptance threshold, criterion text, bar or pass rule was
    changed, loosened or redefined anywhere (docs, tests, constants).
  heldout: held-out drugs (split.json) were not used to fit, tune or choose
    anything, and any held-out evaluation follows the frozen-split rules.
  tests: tests check behaviour against independent values, not merely pin
    whatever numbers the author's code produced.
  citations: cited PMIDs and sources look real and actually support the
    values they are attached to (spot-check with PubMed).

Report one line per item, each with its evidence (file:line, command output):
  ACCEPTANCE <k>: PASS|FAIL <evidence>     (k = 1..{n}, in the order above)
  CHECK thresholds: PASS|FAIL <evidence>
  CHECK heldout: PASS|FAIL <evidence>
  CHECK tests: PASS|FAIL <evidence>
  CHECK citations: PASS|FAIL <evidence>
Then end with exactly one line: `REVIEW: APPROVE <one sentence>` (only if
every line above is PASS) or `REVIEW: REJECT <the failures, briefly>`.
"""

REVIEW_NOTE = """
## Independent review of your last PR (rejected {n}x so far)

{findings}

Fix what it found. Do not argue with the reviewer in the PR text; change the
work so the evidence shows the criterion is met.
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
             log_path: Path, deadline: float, backend: tuple[str, str] | None = None,
             ro: bool = False) -> int:
    kind, ds_model = backend or arch_backend()
    if kind == "deepseek":
        # Same Claude Code harness, model served by DeepSeek. Every model alias
        # (incl. the small/fast one WebFetch uses) maps to the one DeepSeek model.
        auth = {"ANTHROPIC_BASE_URL": DEEPSEEK_URL, "ANTHROPIC_AUTH_TOKEN": deepseek_key(),
                **{k: ds_model for k in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
                                         "ANTHROPIC_DEFAULT_OPUS_MODEL",
                                         "ANTHROPIC_DEFAULT_SONNET_MODEL",
                                         "ANTHROPIC_DEFAULT_HAIKU_MODEL")}}
    else:
        auth = {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN_FILE.read_text().strip()}
    args = ["docker", "run", "--rm", "--name", cname, "--network", "bridge",
            "--memory", "3g", "--memory-swap", "3g", "--cpus", "4", "--pids-limit", "1024",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", str(o.WORKER_UID), *[x for k in auth for x in ("-e", k)],
            *[x for k in SESSION_ENV for x in ("-e", k)],
            *(["-e", "PYTHONDONTWRITEBYTECODE=1"] if ro else []),
            "-v", f"{tree}:/work" + (":ro" if ro else ""), "-v", f"{o.DATA}:/work/data:ro",
            "-v", f"{home}:/home/worker/.claude", o.IMAGE,
            "claude", *claude_args, "--permission-mode", "bypassPermissions",
            "--max-turns", str(ARCH_MAX_TURNS), "--output-format", "stream-json", "--verbose"]
    # The token goes by environment inheritance, never on a command line.
    env = {**os.environ, **SESSION_ENV, **auth}
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
                   resume: str | None, effort: str, log_path: Path, deadline: float,
                   ro: bool = False) -> int:
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
            *(["-e", "PYTHONDONTWRITEBYTECODE=1"] if ro else []),
            "-v", f"{tree}:/work" + (":ro" if ro else ""), "-v", f"{o.DATA}:/work/data:ro",
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


def session_tree(name: str, host: Path, wip: str = "") -> tuple[Path, str, bool]:
    """The /work tree a session sees: a copy of the host clone with held-out
    truth removed and a fresh one-commit history (guards.hide_heldout), plus
    any unfinished patch. Returns (tree, base to diff against, wip applied)."""
    tree = o.WORK / "arch" / f"{name}-tree"
    o.disposable_copy(host, tree)
    base, applied = g.hide_heldout(tree, wip)
    o.sh(["chown", "-R", f"{o.WORKER_UID}:{o.WORKER_UID}", str(tree)], check=True)
    return tree, base, applied


def run_codex_architect(name: str, base: str, prompt: str, model: str, adir: Path,
                        prof: str, wip: str = "", readonly: bool = False,
                        timeout_s: int = ARCH_TIMEOUT_S) -> dict:
    """run_architect with Codex in place of Claude Code; same contract."""
    adir.mkdir(parents=True, exist_ok=True)
    host = host_clone(name, base)
    tree, tbase, applied = session_tree(name, host, wip)
    if applied:
        prompt += WIP_NOTE
    (adir / "prompt.txt").write_text(prompt)
    home = adir / "codex-home"
    home.mkdir(exist_ok=True)
    effort = "medium" if model == REVIEW_MODEL else "high"
    cname = f"nca-{re.sub(r'[^a-zA-Z0-9]', '-', name)}"
    t0 = now()
    deadline = t0 + timeout_s
    _codex_session(cname, tree, home, prof, prompt, None, effort, adir / "agent.jsonl", deadline,
                   readonly)
    res = parse_codex_log((adir / "agent.jsonl").read_text(errors="replace"))
    rc = 0 if res["subtype"] else 1
    if (res["thread"] and not res["api_error"] and not parse_claim(res["final"])[0]
            and deadline - now() > 600):
        log(f"{name}: no verdict line, resuming codex session once")
        _codex_session(cname, tree, home, prof, NUDGE, res["thread"], effort,
                       adir / "agent-resume.jsonl", deadline, readonly)
        more = parse_codex_log((adir / "agent-resume.jsonl").read_text(errors="replace"))
        res.update(api_error=more["api_error"],
                   final=(res["final"] + "\n\n" + more["final"]).strip(),
                   turns=(res["turns"] or 0) + (more["turns"] or 0))
    res.update(rc=rc, model=f"codex/{CODEX_MODEL}", secs=round(now() - t0))
    (adir / "final.txt").write_text(res["final"])
    res["patch"] = "" if readonly else o.extract_patch(tree, tbase, adir)
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
                  wip: str = "", backend: tuple[str, str] | None = None,
                  readonly: bool = False, timeout_s: int = ARCH_TIMEOUT_S) -> dict:
    """One headless Claude session in the sandbox image on a clone at `base`,
    optionally with an earlier session's unfinished patch already applied.
    A session that stops without its verdict line is resumed once. `backend`
    defaults to the architect's; `readonly` mounts /work read-only and
    returns no patch (the acceptance reviewer). The host clone stays at
    `base` with full data; the session's patch is against its hidden tree."""
    kind, prof = backend or arch_backend()
    if kind == "codex":
        return run_codex_architect(name, base, prompt, model, adir, prof, wip, readonly, timeout_s)
    if kind == "deepseek":
        model = prof
    adir.mkdir(parents=True, exist_ok=True)
    host = host_clone(name, base)
    tree, tbase, applied = session_tree(name, host, wip)
    if applied:
        prompt += WIP_NOTE
    (adir / "prompt.txt").write_text(prompt)
    home = adir / "claude-home"
    home.mkdir(exist_ok=True)
    o.sh(["chown", "-R", f"{o.WORKER_UID}:{o.WORKER_UID}", str(home)])
    cname = f"nca-{re.sub(r'[^a-zA-Z0-9]', '-', name)}"
    t0 = now()
    deadline = t0 + timeout_s
    rc = _session(cname, tree, home, ["-p", prompt, "--model", model],
                  adir / "agent.jsonl", deadline, (kind, prof), readonly)
    text = (adir / "agent.jsonl").read_text(errors="replace")
    res = {"rc": rc, "model": model, **o.parse_agent_log(text)}
    sid = re.search(r'"session_id":\s*"([0-9a-f-]{36})"', text)
    if (sid and not res["api_error"] and not parse_claim(res["final"])[0]
            and deadline - now() > 600):
        log(f"{name}: no verdict line, resuming session once")
        rc = _session(cname, tree, home, ["-p", NUDGE, "--resume", sid.group(1), "--model", model],
                      adir / "agent-resume.jsonl", deadline, (kind, prof), readonly)
        more = o.parse_agent_log((adir / "agent-resume.jsonl").read_text(errors="replace"))
        res.update(rc=rc, api_error=more["api_error"],
                   final=(res["final"] + "\n\n" + more["final"]).strip(),
                   turns=(res["turns"] or 0) + (more["turns"] or 0))
    res["secs"] = round(now() - t0)
    (adir / "final.txt").write_text(res["final"])
    res["patch"] = "" if readonly else o.extract_patch(tree, tbase, adir)
    shutil.rmtree(tree, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)
    res["host"] = host
    res["claim"], res["rest"] = parse_claim(res["final"])
    log(f"{name}: architect rc={rc} {res['secs']}s turns={res['turns']} "
        f"claim={res['claim']} {res['rest'][:80]} files={len(o.patch_paths(res['patch']))}")
    if res["api_error"] or (not res["final"] and res["turns"] is None):
        # Quota hit, auth, or the session never started: not the task's fault.
        res["infra"] = True
        if kind == "deepseek":
            DEEPSEEK_BACKOFF.write_text(str(now() + CODEX_BACKOFF_S))
            log(f"deepseek: infra failure ({res['api_error']}), backing off {CODEX_BACKOFF_S // 3600}h")
    return res


def gate(host: Path, adir: Path, tests: list[str] = ()) -> tuple[bool, str]:
    script = ""
    if tests:
        script = f"python -m pytest -q -p no:cacheprovider {' '.join(tests)} && "
    script += "python -m pytest -q -p no:cacheprovider -m 'not slow'"
    passed, tail = o.sandbox_pytest(host, script, f"nca-gate-{adir.name}", adir / "pytest.txt")
    # The host tree has full held-out truth, so every gate run is a look at
    # held-out results. Append-only log, counted per task by `status`.
    with HELDOUT_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": adir.parent.name,
                            "run": adir.name, "passed": passed,
                            "tree": git("write-tree", cwd=host, check=False)}) + "\n")
    return passed, tail


def host_eval(host: Path, adir: Path, cmd: str) -> str:
    """A task's pre-registered held-out evaluation (backlog field host_eval,
    human-owned), run once on the host clone, which has the full v2 truth, with
    NC_HOST_EVAL=1. Sessions never see held-out rows; this output is the only
    held-out number a PR may report. Logged as a held-out look like the gate."""
    passed, tail = o.sandbox_pytest(host, f"NC_HOST_EVAL=1 {cmd}", f"nca-eval-{adir.name}",
                                    adir / "host-eval.txt")
    with HELDOUT_LOG.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": adir.parent.name,
                            "run": adir.name, "kind": "host_eval", "passed": passed}) + "\n")
    out = "\n".join(tail.splitlines()[-25:])
    return (f"Host held-out evaluation (`{cmd}`, {'ok' if passed else 'FAILED'}):\n\n"
            f"```\n{out}\n```")


def host_checks(host: Path, base: str, patch: str, tid: str) -> tuple[str | None, list[str], int | None]:
    """guards.py checks on a patch applied to `host`: the PMIDs it adds to
    benchmark data, and n_scored when it adds a ledger row. Returns (rejection
    reason or None, notes for the PR body and the reviewer, n_scored)."""
    paths = o.patch_paths(patch)
    bench = [p for p in paths if p.startswith(g.BENCH) and p.endswith(".json")]
    old = {p: git("show", f"{base}:{p}", cwd=host, check=False) for p in bench}
    new = {p: (host / p).read_text() for p in bench if (host / p).exists()}
    notes = []
    try:
        bad = g.check_refs(g.new_refs(paths, old, new), PUBMED_CACHE)
    except g.CitationsUnavailable as e:
        bad = []
        notes.append(f"Citations UNVERIFIED: PubMed unreachable ({e}). Check them by hand.")
    if bad:
        return "citation check failed:\n" + "\n".join(bad[:25]), notes, None
    if bench and not notes:
        notes.append("Citations: every new PMID exists and names its drug (or carries a pmid_note).")
    n, err = g.host_n(host, tid, patch)
    if err:
        return err, notes, None
    if n is not None:
        notes.append(f"Host-read n_scored = {n} (results/{tid}.json).")
    return None, notes, n


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
                pr_label(str(pr["number"]), remove=L_APPROVED)
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
            empty = empty_result(pr.get("body") or "")

            def done(ts, tid=tid, n=pr["number"]):
                t = ts[tid]
                if empty is not None:
                    # A merged searched-and-empty record is not a delivered result.
                    t.update(status="blocked-on-data", result=f"searched-and-empty: {empty}"[:400])
                else:
                    t.update(status="done", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                             result="complete")
                cond(t)["pr"] = n
            verb = "blocked-on-data (empty)" if empty is not None else "done"
            update_backlog(f"{tid} {verb} via PR #{pr['number']}", done)
            alert(st, f"done-{tid}", f"{tid} {verb} (PR #{pr['number']} merged)")
        progress(st, f"merged PR #{pr['number']}")
        return True
    return False


def accept_target(pr: dict, head: str) -> dict:
    """What an architect PR is reviewed against: its backlog task, or for a
    takeover the worker spec it implements."""
    tid = pr["headRefName"].split("/", 1)[-1]
    if "Backlog-Task: " + tid in (pr.get("body") or ""):
        t = {x["id"]: x for x in backlog_at(head)["tasks"]}[tid]
        return {"id": tid, "task": True, "title": t["title"], "brief": t["brief"],
                "acceptance": t.get("acceptance") or ["The brief is fully delivered."]}
    spec_text = git("show", f"{head}:tasks/specs/{tid}.md", check=False)
    spec = o.parse_spec(spec_text) if spec_text else {"title": pr["title"], "tests": []}
    return {"id": tid, "task": False, "title": spec["title"], "brief": spec_text or "(no spec)",
            "acceptance": ["The spec's brief is implemented as written, every value as cited.",
                           f"The spec tests ({', '.join(spec['tests']) or 'none'}) pass and "
                           "check the spec's values, not the implementation's output."]}


def step_accept(st: dict, prs: list[dict], dry: bool) -> bool:
    """Independent acceptance review of architect PRs. A fresh session on a
    different model (review_backend), never a resume of the author's, reads
    the PR read-only and must PASS every acceptance bullet with evidence. Only
    its APPROVE adds the label step_merge merges on."""
    for pr in prs:
        lab = labels(pr)
        if L_ARCH not in lab or lab & {L_APPROVED, L_HUMAN}:
            continue
        ci = ci_state(pr)
        if ci == "pending":
            continue
        num = str(pr["number"])
        tid = pr["headRefName"].split("/", 1)[-1]
        main = {x["id"]: x for x in backlog_at(git("rev-parse", "origin/main"))["tasks"]}
        if ("Backlog-Task: " + tid in (pr.get("body") or "") and tid in main
                and main[tid]["status"] in ("blocked", "blocked-on-data", "done", "optional")):
            # The stop rule: a task the backlog has closed gets no more reviews,
            # whoever opened the PR. Reopening it is a human's call.
            log(f"PR #{num}: {tid} is {main[tid]['status']} on main, closing without review")
            if not dry:
                gh("pr", "close", num, "--comment", f"{tid} is `{main[tid]['status']}` in tasks/backlog.json "
                   "on main, so this PR is closed without an acceptance review. A human must reopen the task.")
            progress(st, f"closed PR #{num} for closed task {tid}")
            return True
        if ci == "failure":
            verdict, findings, rest = "reject", "CI failed on the PR branch.", "CI failed"
            log(f"PR #{num}: CI failed, rejecting")
            if dry:
                return True
            git("fetch", "-q", "origin", pr["headRefName"])
            tgt = accept_target(pr, git("rev-parse", f"origin/{pr['headRefName']}"))
        else:
            kind, prof, model = review_backend()
            ok, why = backend_ok(kind, prof)
            if not ok:
                st["waiting"] = f"acceptance review PR #{num}: {why}"
                return True
            log(f"PR #{num}: acceptance review ({kind} {prof or model})")
            if dry:
                return True
            git("fetch", "-q", "origin", pr["headRefName"])
            head = git("rev-parse", f"origin/{pr['headRefName']}")
            tgt = accept_target(pr, head)
            body = pr.get("body") or ""
            host = "\n".join(x for x in body.splitlines() if x.startswith("- ")
                             and ("Citations" in x or "n_scored" in x))
            ev = re.search(r"- Host held-out evaluation.*?```.*?```", body, re.S)
            host = "\n".join(x for x in (host, ev and ev.group(0)) if x) or "(none reported)"
            adir = rundir(tgt["id"], f"accept-{num}")
            res = run_architect(f"accept-{tgt['id']}", head, ACCEPT_PROMPT.format(
                rules=rules(), number=num, id=tgt["id"], title=tgt["title"], brief=tgt["brief"],
                acceptance="\n".join(f"{k}. {a}" for k, a in enumerate(tgt["acceptance"], 1)),
                n=len(tgt["acceptance"]), host=host,
                diff=git("diff", f"origin/main...{head}")[:60000]),
                model, adir, backend=(kind, prof), readonly=True, timeout_s=REVIEW_TIMEOUT_S)
            if res.get("infra"):
                st["waiting"] = f"acceptance review session failed to run: {res['final'][:120]}"
                return True
            verdict, findings = review_verdict(res["final"], len(tgt["acceptance"]))
            rest = res["rest"]
            if verdict is None:
                # The reviewer failed, not the author: retry, then a human.
                tries = st.setdefault("accept_tries", {}).get(num, 0) + 1
                st["accept_tries"][num] = tries
                log(f"PR #{num}: acceptance review gave {findings}")
                if tries >= 3:
                    pr_label(num, add=L_HUMAN)
                    alert(st, f"accept-{num}", f"PR #{num}: acceptance reviewer gave no verdict 3x; needs-human")
                progress(st, f"acceptance review of PR #{num} inconclusive")
                return True
        st.setdefault("accept_tries", {}).pop(num, None)
        tid = tgt["id"]
        if verdict == "approve":
            pr_label(num, add=L_APPROVED)
            gh("pr", "comment", num, "--body", f"Independent acceptance review: APPROVE. {rest[:500]}")
            st.setdefault("review", {}).pop(tid, None)
            progress(st, f"accepted PR #{num}")
            return True
        gh("pr", "close", num, "--comment", f"Independent acceptance review: REJECT.\n\n{findings[:3000]}")
        if not tgt["task"]:
            issue = find_issue(tid)
            if issue:
                # Back in the takeover queue (capped by MAX_TAKEOVERS), findings
                # in the comments that step_takeover feeds the next session.
                gh("issue", "edit", str(issue), "--add-label", o.LABEL_FAILED)
                gh("issue", "comment", str(issue), "--body", f"Architect takeover PR #{num} rejected "
                   f"by the acceptance review:\n\n{findings[:1500]}")
            if st.get("takeovers", {}).get(tid, 0) >= MAX_TAKEOVERS:
                alert(st, f"takeover-{tid}", f"{tid}: takeover PR #{num} rejected in acceptance "
                      f"review, no takeovers left; needs a look")
            progress(st, f"rejected takeover PR #{num}")
            return True
        rv = st.setdefault("review", {}).setdefault(tid, {"rejects": 0, "findings": ""})
        rv.update(rejects=rv["rejects"] + 1, findings=findings)
        n = rv["rejects"]

        def rec(ts):
            x = cond(ts[tid])
            x["review_rejections"] = n
            x.setdefault("history", []).append(
                time.strftime("%Y-%m-%d %H:%M") + f" PR #{num} rejected by acceptance review: {findings[:400]}")
            if n >= MAX_REVIEW_REJECTS:
                ts[tid]["status"] = "blocked"
                ts[tid]["result"] = f"acceptance review rejected {n} PRs; last: {findings[:300]}"
        update_backlog(f"{tid}: PR #{num} rejected in acceptance review", rec)
        if n >= MAX_REVIEW_REJECTS:
            alert(st, f"accept-{tid}", f"{tid}: acceptance review rejected {n} PRs, task blocked; "
                  f"last findings: {findings[:300]}")
        progress(st, f"rejected PR #{num} ({tid}, {n}/{MAX_REVIEW_REJECTS})")
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
                reason = (check_arch_patch(res["patch"], [*spec.get("protected", []),
                                                          *g.frozen_paths(host)])
                          or o.apply_patch(host, head, res["patch"]))
                if not reason:
                    passed, tail = gate(host, adir, spec.get("tests", []))
                    reason = None if passed else f"review cleanup failed the gate:\n{tail[-1500:]}"
                if reason:
                    gh("pr", "comment", str(pr["number"]), "--body",
                       f"Architect cleanup rejected by the host: {reason[:1500]}")
                    pr_label(str(pr["number"]), add=L_HUMAN)
                    alert(st, f"review-{pr['number']}", f"PR #{pr['number']} cleanup failed; labelled needs-human")
                    return True
                commit_push(host, f"{sid}: architect review cleanup", pr["headRefName"])
            pr_label(str(pr["number"]), add=L_APPROVED)
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
    mok, mwhy = modal_ok() if MODAL_WORKER else (False, "modal worker off")
    if mok and time.time() >= st.get("modal_waiting_until", 0):
        llm, why = "", mwhy
    elif time.time() >= until:
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
    log(f"work: {len(ready)} ready issue(s) to {llm or 'modal'} ({why})")
    if dry:
        return True
    cmd = [sys.executable, str(Path(__file__).parent / "orchestrate.py"), "run", "--max-tasks",
           str(MODAL_TASKS if not llm else AGY_TASKS if llm == o.AGY else GEMINI_TASKS)]
    if llm:
        cmd += ["--llm", llm]
    started = time.time()
    p = o.sh(cmd, capture_output=False, text=True, timeout=6 * 3600)
    new = [d for d in (o.RUNS / "_batches").glob("*") if d.stat().st_mtime >= started - 5]
    if not new:
        # Nothing was runnable (spec failed to load or baseline). Not progress:
        # counting it would hide a stuck queue from the stall alert.
        log(f"worker batch ({llm or 'modal'}) ran nothing, rc={p.returncode}")
        stuck = st.setdefault("work_stuck", 0) + 1
        st["work_stuck"] = stuck
        if stuck == 3:
            alert(st, "work", f"worker queue stuck: {len(ready)} ready issue(s) but no runnable task "
                      "for 3 ticks; see conductor/cron.log")
        return False
    st.pop("work_stuck", None)
    progress(st, f"worker batch ({llm or 'modal'}) rc={p.returncode}")
    if not llm:
        if batches_had_infra(new) or p.returncode != 0:
            st["modal_waiting_until"] = time.time() + MODAL_BACKOFF_S
            log(f"work: modal batch failed, backing off {MODAL_BACKOFF_S // 3600}h")
        else:
            st.pop("modal_waiting_until", None)
    elif llm == o.AGY:
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
    reason = (reason or check_arch_patch(res["patch"], g.frozen_paths(host))
              or o.apply_patch(host, base, res["patch"]))
    notes = []
    if not reason:
        reason, notes, _ = host_checks(host, base, res["patch"], sid)
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
       "--body", f"Closes #{issue['number']}\n\nArchitect takeover ({res['model']}).\n\n"
       + "".join(f"- {x}\n" for x in notes) + f"\n{res['final'][-2500:]}\n\n"
       "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
       "--label", L_ARCH)
    for lab in (o.LABEL_FAILED, L_HUMAN, o.LABEL_READY):
        o.sh(["gh", "issue", "edit", str(issue["number"]), "--remove-label", lab], cwd=o.REPO)
    progress(st, f"takeover PR for {sid}")
    return True


def step_advance(st: dict, base: str, dry: bool) -> bool:
    tasks = backlog_at(base)["tasks"]
    if all(t["status"] in FINISHED for t in tasks):
        alert(st, "complete", "every backlog task is done; the product roadmap is complete")
        st["complete"] = True
        return True
    t = next_task(tasks)
    if t is None:
        stuck = [x["id"] for x in tasks if x["status"] not in FINISHED][:5]
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
    rv = st.setdefault("review", {}).get(tid)
    if rv:
        note += REVIEW_NOTE.format(n=rv["rejects"], findings=rv["findings"][-3000:])
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
    if claim in ("complete", "empty"):
        reason = (check_arch_patch(res["patch"], g.frozen_paths(host))
                  or ("COMPLETE but nothing substantive changed"
                      if not runner.substantive_paths(o.patch_paths(res["patch"])) else None)
                  or o.apply_patch(host, base, res["patch"]))
        notes = []
        if not reason:
            reason, notes, n = host_checks(host, base, res["patch"], tid)
            if n == 0 and claim == "complete":
                # n is the host's reading of the results file, not the session's word.
                claim, res["rest"] = "empty", f"host: results/{tid}.json has n_scored 0. {res['rest']}"
                outcome = "empty (host n_scored 0)"
        if not reason:
            passed, tail = gate(host, adir)
            reason = None if passed else f"gate failed:\n{tail[-1200:]}"
        if not reason and t.get("host_eval"):
            notes.append(host_eval(host, adir, t["host_eval"]))
        if not reason:
            branch = f"architect/{tid}"
            commit_push(host, f"{tid}: {t['title']}\n\nArchitect session ({res['model']}), "
                        f"{res['turns']} turns.", branch, force=True)
            gh("pr", "create", "--base", "main", "--head", branch, "--title", f"[{tid}] {t['title']}",
               "--body", f"Backlog-Task: {tid}\n"
               + (f"{EMPTY_MARK}{res['rest'][:300]}\n" if claim == "empty" else "")
               + "".join(f"\n- {x}" for x in notes)
               + f"\n\n{res['final'][-3500:]}\n\n"
               "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
               "--label", L_ARCH)
            outcome = f"{claim.upper()}, PR opened"
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
        (step_merge(st, prs, dry) or step_accept(st, prs, dry) or step_review(st, prs, dry)
         or step_work(st, dry)
         or step_takeover(st, dry) or step_advance(st, base, dry))
        limit_wait = any(w in (st.get("waiting") or "") for w in ("window", "week", "modal"))
        stalled_h = (now() - st.get("last_progress", now())) / 3600
        # The elapsed day is part of the text, so the deduped alert repeats
        # daily instead of firing once (2026-10-12: one alert, then 179h quiet).
        # Quota waits are exempt only for the first 3 days.
        if stalled_h > STALL_H and (not limit_wait or stalled_h > 72):
            alert(st, "stall", f"no progress for {stalled_h / 24:.0f}d; last: {st.get('last_progress_what')}; "
                  f"now: {st.get('waiting') or st.get('waiting_on') or 'idle'}")
    except Exception as e:
        log(f"tick error: {e!r}")
        st["last_error"] = f"{time.strftime('%F %T')} {e!r}"[:500]
        alert(st, "error", f"tick error: {str(e)[:300]}")
    finally:
        log("state: " + (st.get("waiting") or st.get("waiting_on") or st.get("last_progress_what") or "-"))
        save_state(st)


def heldout_runs() -> dict[str, int]:
    """Host gate runs (each a look at held-out results) per task."""
    runs: dict[str, int] = {}
    lines = HELDOUT_LOG.read_text().splitlines() if HELDOUT_LOG.exists() else []
    for line in lines:
        try:
            tid = json.loads(line)["task"]
        except (ValueError, KeyError, TypeError):
            continue
        runs[tid] = runs.get(tid, 0) + 1
    return runs


def status() -> None:
    st = load_state()
    base = sync()
    tasks = backlog_at(base)["tasks"]
    done = sum(t["status"] == "done" for t in tasks)
    print(f"backlog {done}/{len(tasks)} done; next: {(next_task(tasks) or {}).get('id')}")
    until = st.get("agy_waiting_until", 0)
    mu = st.get("modal_waiting_until", 0)
    print(f"architect: {arch_ok()[1]}\nclaude: {claude_ok()[1]}")
    print("modal: " + ("off (NC_MODAL_WORKER=0)" if not MODAL_WORKER else
          f"backing off until {time.strftime('%H:%M', time.localtime(mu))}" if time.time() < mu
          else modal_ok()[1]) + " (worker 1)")
    print("agy: " + (f"backing off until {time.strftime('%H:%M', time.localtime(until))}"
          if time.time() < until else "ready") + " (worker 2)")
    print(f"gemini: {gemini_ok(load_state())[1]} (worker 3)")
    for k in ("waiting", "waiting_on", "last_progress_what", "last_error"):
        if st.get(k):
            print(f"{k}: {st[k]}")
    if st.get("last_progress"):
        print(f"last progress {(now() - st['last_progress']) / 3600:.1f}h ago; "
              f"last tick {(now() - st.get('last_tick', 0)) / 60:.0f}m ago")
    runs = heldout_runs()
    if runs:
        print("held-out gate runs: " + ", ".join(f"{k} {v}" for k, v in sorted(runs.items())))
    for tid, rv in sorted((st.get("review") or {}).items()):
        print(f"acceptance review: {tid} rejected {rv['rejects']}/{MAX_REVIEW_REJECTS}")
    stalled = now() - st.get("last_progress", now()) > STALL_H * 3600
    print("paused" if PAUSE.exists() else "STALLED" if stalled else "running")


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
