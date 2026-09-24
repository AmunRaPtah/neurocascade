#!/usr/bin/env python3
"""Architect/worker pipeline: Claude writes specs + failing tests, an open-weights
model on Modal writes the code, the gate and GitHub decide what lands.

    orchestrate.py queue                       # open worker-ready issues
    orchestrate.py run [--ids W1.1 ...] [--model glm47-flash] [--max-tasks N]
    orchestrate.py budget                      # Modal month-to-date vs cap

Trust boundaries (why this is shaped the way it is, see pipeline/README.md):

* The worker runs in the nc-worker container on the internal nc-sandbox network.
  It sees its own throwaway clone and data/ read-only, can reach the LLM proxy
  and nothing else, and holds no GitHub credential. On 2026-09-24 a worker with
  host access ran `rm -rf` on the real repo; this layout makes that impossible.
* Its only output is a patch, produced inside a container. The host never runs
  git inside a tree the worker controlled (hooks and .git/config are code), it
  applies the patch as data to a clean clone of its own.
* Tests execute worker code, so they run in a network-less container on a
  disposable copy, never on the tree the host commits from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
GITHUB = "https://github.com/AmunRaPtah/neurocascade.git"
LLM_BURST = Path("/root/projects/llm-burst")
WORK = Path("/tmp/ncw")
RUNS = Path("/root/nc-pipeline-runs")
IMAGE = "nc-worker"
NETWORK = "nc-sandbox"
PROXY = "http://172.31.250.1:4000"
WORKER_UID = 20001

MONTH_CAP = float(os.environ.get("NC_MODAL_MONTH_CAP", 25.0))
AGENT_TIMEOUT_S = int(os.environ.get("NC_AGENT_TIMEOUT_S", 45 * 60))
TEST_TIMEOUT_S = 40 * 60
MAX_ATTEMPTS = 2
MIN_FREE_MB = 1500

# Deploy targets in llm-burst/modal_app.py. Apps are deployed for a batch and
# stopped after it, so an idle month costs nothing and a stray client cannot
# wake a GPU.
DEPLOY = {"glm47-flash": "glm_app", "qwen38-27b": "qwen_app",
          "qwen3-coder-next": "coder_next_app"}

# The worker may never change these, whatever the spec says.
FORBIDDEN = (".github/", "pipeline/", "tasks/specs/", "tasks/backlog.json", "data/")
#: Everything a worker patch may touch. Keeps scratch scripts out of the repo.
ALLOWED = ("src/", "tests/", "docs/")

class InfraError(RuntimeError):
    """The LLM backend, not the worker, failed. Stops the batch; burns no attempt."""


LABEL_READY, LABEL_PR, LABEL_FAILED = "worker-ready", "worker-pr", "worker-failed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


# --------------------------------------------------------------------------- spec

def parse_spec(text: str) -> dict:
    """tasks/specs/<ID>.md: a `---` frontmatter of `key: value` / `key: [a, b]`, then the brief."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError("spec has no frontmatter")
    meta: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        meta[k.strip()] = ([x.strip() for x in v[1:-1].split(",") if x.strip()]
                           if v.startswith("[") else v)
    meta["brief"] = m.group(2).strip()
    for req in ("id", "title", "tests"):
        if not meta.get(req):
            raise ValueError(f"spec missing {req!r}")
    meta.setdefault("protected", meta["tests"])
    return meta


def load_spec(task_id: str, local: bool = False) -> tuple[dict, str]:
    """Spec from the spec/<ID> branch (pushed, or local for --dry-run), plus its head SHA."""
    ref = f"spec/{task_id}" if local else f"origin/spec/{task_id}"
    if not local:
        sh(["git", "-C", str(REPO), "fetch", "-q", "origin", f"spec/{task_id}"], check=True)
    sha = sh(["git", "-C", str(REPO), "rev-parse", ref], check=True).stdout.strip()
    text = sh(["git", "-C", str(REPO), "show", f"{ref}:tasks/specs/{task_id}.md"],
              check=True).stdout
    return parse_spec(text), sha


# ------------------------------------------------------------------------- github

def gh_json(args: list[str]):
    return json.loads(sh(["gh", *args], check=True, cwd=REPO).stdout or "null")


def ready_issues() -> list[dict]:
    issues = gh_json(["issue", "list", "--label", LABEL_READY, "--state", "open",
                      "--json", "number,title", "--limit", "100"])
    out = []
    for i in issues:
        m = re.match(r"^\[([A-Z]\w*\.\w+)\]", i["title"])
        if m:
            out.append({"id": m.group(1), **i})
    return sorted(out, key=lambda i: i["number"])


def relabel(num: int, add: str, remove: str = LABEL_READY) -> None:
    sh(["gh", "issue", "edit", str(num), "--add-label", add, "--remove-label", remove], cwd=REPO)


# ------------------------------------------------------------------------- budget

def modal_month_spend() -> float:
    p = sh(["modal", "billing", "report", "--for", "this month", "--json"])
    if p.returncode != 0:
        raise RuntimeError(f"modal billing failed: {p.stderr[-300:]}")
    return sum(float(r["cost"]) for r in json.loads(p.stdout))


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def model_ctx(model: str) -> int:
    sys.path.insert(0, str(LLM_BURST))
    from models import get  # noqa: E402
    return get(model).max_model_len


def modal_app(model: str, up: bool) -> None:
    if up:
        log(f"deploying llm-burst-{model}")
        sh(["modal", "deploy", f"modal_app.py::{DEPLOY[model]}"], cwd=LLM_BURST, check=True)
    else:
        log(f"stopping llm-burst-{model}")
        sh(["modal", "app", "stop", "-y", f"llm-burst-{model}"])


# ----------------------------------------------------------------------- sandbox

def proxy_key() -> str:
    """LiteLLM master key, read from the proxy unit's own environment."""
    env = sh(["systemctl", "show", "pardalos-freeproxy", "-p", "Environment", "--value"]).stdout
    for tok in env.split():
        if tok.startswith("PARDALOS_FREE_PROXY_KEY="):
            return tok.split("=", 1)[1]
    raise RuntimeError("PARDALOS_FREE_PROXY_KEY not set on pardalos-freeproxy")


def docker_run(name: str, mounts: list[str], cmd: list[str], *, network: str,
               env: dict | None = None, timeout: int, stdout_path: Path) -> int:
    args = ["docker", "run", "--rm", "--name", name, "--network", network,
            "--memory", "3g", "--memory-swap", "3g", "--cpus", "4", "--pids-limit", "1024",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", str(WORKER_UID)]
    for m in mounts:
        args += ["-v", m]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += [IMAGE, *cmd]
    with stdout_path.open("w") as f:
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            sh(["docker", "kill", name])
            proc.wait()
            return 124


def fresh_clone(dest: Path, sha: str) -> None:
    """Host-owned clone at `sha`, never touched by worker code."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "-q", "--no-local", "--no-checkout", str(REPO), str(dest)], check=True)
    sh(["git", "-C", str(dest), "checkout", "-q", "--detach", sha], check=True)
    sh(["git", "-C", str(dest), "remote", "set-url", "origin", GITHUB], check=True)


def disposable_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, symlinks=True)
    (dest / "data").mkdir(exist_ok=True)
    sh(["chown", "-R", f"{WORKER_UID}:{WORKER_UID}", str(dest)], check=True)


PROMPT = """You are implementing one task in the neurocascade repository, which is your
current directory (/work). Work only inside /work. data/ is read-only reference data.

TASK {id}: {title}

{brief}

DEFINITION OF DONE
These tests were written for this task and currently fail. Make them pass:
  {tests}
Run them with:  python -m pytest -q {tests}
Do NOT run the whole suite: it takes over 10 minutes and the orchestrator
runs it after you finish. If you touched an existing module, run only the
test files for that module (tests/test_<module>*.py).

RULES
- Do not edit, delete or skip these files: {protected}. They are the spec.
- Only change files under src/, tests/ and docs/. Do not modify .github/,
  pipeline/, tasks/ or data/, and delete any scratch or debug scripts you
  create before finishing; the patch is rejected if they are left behind.
- Remove any debug logging or print statements you added while investigating.
- Every numeric parameter must come from the spec or from code/data already in
  the repo. Never invent a value; if one is missing, stop and say so.
- Keep the change focused on this task. No unrelated refactors.
- When finished, end your reply with exactly one line:
  TASK_RESULT: DONE - <one sentence>    or    TASK_RESULT: BLOCKED - <reason>
{retry}"""

RETRY = """
PREVIOUS ATTEMPT
Your previous attempt is already applied to the tree. It failed the gate:
{failure}
Fix the failure rather than starting over."""


def run_worker(spec: dict, llm: str, ctx: int, tree: Path, adir: Path, failure: str | None) -> dict:
    prompt = PROMPT.format(
        id=spec["id"], title=spec["title"], brief=spec["brief"],
        tests=" ".join(spec["tests"]), protected=", ".join(spec["protected"]),
        retry=RETRY.format(failure=failure) if failure else "")
    (adir / "prompt.txt").write_text(prompt)
    env = {"ANTHROPIC_BASE_URL": PROXY, "ANTHROPIC_AUTH_TOKEN": proxy_key(),
           "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192",
           "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(ctx),
           **{k: llm for k in (
               "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
               "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL")}}
    t0 = time.time()
    rc = docker_run(f"ncw-{spec['id'].replace('.', '-')}-agent",
                    [f"{tree}:/work", f"{DATA}:/work/data:ro"],
                    ["claude", "-p", prompt, "--permission-mode", "bypassPermissions",
                     "--output-format", "stream-json", "--verbose"],
                    network=NETWORK, env=env, timeout=AGENT_TIMEOUT_S,
                    stdout_path=adir / "agent.jsonl")
    res = {"rc": rc, "agent_s": round(time.time() - t0), "final": "", "turns": None}
    for line in (adir / "agent.jsonl").read_text(errors="replace").splitlines():
        if not line.startswith("{") or '"result"' not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Key order varies between Claude Code versions, so match on the value.
        if obj.get("type") == "result":
            res["final"], res["turns"] = obj.get("result") or "", obj.get("num_turns")
            if obj.get("terminal_reason") == "api_error" or obj.get("api_error_status"):
                raise InfraError(f"LLM API error {obj.get('api_error_status')}: {res['final'][:200]}")
    return res


def extract_patch(tree: Path, base: str, adir: Path) -> str:
    """Diff of the worker tree against `base`, computed inside a container."""
    script = ("git add -A -- . ':!data' >/dev/null 2>&1; "
              f"git diff --cached --binary {base} -- . ':!data'")
    out = adir / "diff.patch"
    rc = docker_run(f"ncw-diff-{adir.parent.name.replace('.', '-')}-{adir.name}",
                    [f"{tree}:/work"], ["sh", "-c", script],
                    network="none", timeout=120, stdout_path=out)
    return out.read_text(errors="replace") if rc == 0 else ""


def patch_paths(patch: str) -> list[str]:
    paths = set()
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M):
        paths.update(m.groups())
    return sorted(paths)


def check_patch(patch: str, spec: dict) -> str | None:
    """Structural review of the patch. None if acceptable, else the reason."""
    paths = patch_paths(patch)
    if not paths:
        return "empty diff: no work was done"
    bad = [p for p in paths if p.startswith(FORBIDDEN) or p in spec["protected"]]
    if bad:
        return f"patch modifies protected paths: {bad}"
    stray = [p for p in paths if not p.startswith(ALLOWED)]
    if stray:
        return f"patch touches files outside {'/ '.join(a.rstrip('/') for a in ALLOWED)}/: {stray}; delete them"
    if not any(p.startswith("src/") for p in paths):
        return f"patch changes no source under src/: {paths}"
    return None


def run_gate(tree: Path, spec: dict, adir: Path) -> tuple[bool, str]:
    """Spec tests first (fast fail), then the not-slow suite, on a disposable copy."""
    copy = tree.parent / (tree.name + "-test")
    disposable_copy(tree, copy)
    tests = " ".join(spec["tests"])
    script = f"python -m pytest -q -p no:cacheprovider {tests}"
    if os.environ.get("NC_GATE_FULL", "1") != "0":   # 0 only for plumbing smoke tests
        script += " && python -m pytest -q -p no:cacheprovider -m 'not slow'"
    rc = docker_run(f"ncw-gate-{spec['id'].replace('.', '-')}-{adir.name}",
                    [f"{copy}:/work", f"{DATA}:/work/data:ro"], ["sh", "-c", script],
                    network="none", timeout=TEST_TIMEOUT_S, stdout_path=adir / "pytest.txt")
    shutil.rmtree(copy, ignore_errors=True)
    out = (adir / "pytest.txt").read_text(errors="replace")
    return rc == 0, "\n".join(out.strip().splitlines()[-40:])


def result_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip().startswith("TASK_RESULT:"):
            return line.strip()
    return "TASK_RESULT: (missing)"


# -------------------------------------------------------------------------- task

def run_task(issue: dict, llm: str, ctx: int, dry: bool = False) -> str:
    spec, base = load_spec(issue["id"], local=dry)
    tid = spec["id"]
    tdir = RUNS / tid / time.strftime("%Y%m%d-%H%M%S")
    tdir.mkdir(parents=True)
    host = WORK / tid / "host"          # host-owned: patches applied here, committed from here
    fresh_clone(host, base)
    patch, failure, records = "", None, []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        adir = tdir / f"attempt-{attempt}"
        adir.mkdir()
        # Worker-owned, never trusted again. Copied from host, which holds base plus
        # the previous attempt's patch when that applied, so a retry continues it.
        tree = WORK / tid / f"worker-{attempt}"
        disposable_copy(host, tree)
        log(f"{tid}: attempt {attempt} worker start ({llm})")
        agent = run_worker(spec, llm, ctx, tree, adir, failure)
        patch = extract_patch(tree, base, adir)
        shutil.rmtree(tree, ignore_errors=True)
        rec = {"attempt": attempt, "model": llm, **{k: v for k, v in agent.items() if k != "final"},
               "result_line": result_line(agent["final"]), "files": patch_paths(patch)}
        log(f"{tid}: agent rc={agent['rc']} {agent['agent_s']}s turns={agent['turns']} "
            f"{rec['result_line'][:100]} files={len(rec['files'])}")

        reason = check_patch(patch, spec)
        if not reason:
            sh(["git", "-C", str(host), "checkout", "-q", "--detach", base], check=True)
            sh(["git", "-C", str(host), "reset", "-q", "--hard"], check=True)
            p = sh(["git", "-C", str(host), "apply", "--index", "-"], input=patch)
            reason = None if p.returncode == 0 else f"patch does not apply: {p.stderr[-300:]}"
        if reason:
            rec.update(gate=False, reason=reason)
            failure = reason
        else:
            ok, tail = run_gate(host, spec, adir)
            rec.update(gate=ok, reason=None if ok else "tests failed")
            failure = None if ok else tail
        records.append(rec)
        (adir / "record.json").write_text(json.dumps(rec, indent=2))
        log(f"{tid}: attempt {attempt} gate={'PASS' if rec['gate'] else 'FAIL'} {rec['reason'] or ''}")
        if rec["gate"]:
            if dry:
                log(f"{tid}: dry run PASS, nothing pushed; records in {tdir}")
                return "pass (dry run)"
            return open_pr(issue, spec, host, records, agent["final"])

    if dry:
        log(f"{tid}: dry run FAIL; records in {tdir}")
        return "failed (dry run)"
    relabel(issue["number"], LABEL_FAILED)
    sh(["gh", "issue", "comment", str(issue["number"]), "--body",
        f"Worker ({llm}) failed after {MAX_ATTEMPTS} attempts.\n\n"
        f"Last failure:\n```\n{(failure or '')[-2500:]}\n```\n"
        f"Records: `{tdir}`. Needs rescoping or architect takeover."], cwd=REPO)
    return "failed"


def open_pr(issue: dict, spec: dict, host: Path, records: list[dict], final: str) -> str:
    tid, branch = spec["id"], f"worker/{spec['id']}"
    msg = (f"{tid}: {spec['title']}\n\nImplemented by the pipeline worker "
           f"({records[-1]['model']}, attempt {records[-1]['attempt']}) against the spec "
           f"tests on spec/{tid}.\n\nCloses #{issue['number']}")
    sh(["git", "-C", str(host), "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", msg], check=True)
    sh(["git", "-C", str(host), "push", "-q", "-f", "origin", f"HEAD:refs/heads/{branch}"], check=True)
    body = (f"Closes #{issue['number']}\n\n**Spec:** `tasks/specs/{tid}.md` on `spec/{tid}`\n"
            f"**Worker:** {records[-1]['model']}, {len(records)} attempt(s)\n\n"
            f"```json\n{json.dumps(records, indent=1)}\n```\n\n**Worker's summary:**\n\n"
            f"{final[-3000:]}\n")
    p = sh(["gh", "pr", "create", "--base", "main", "--head", branch, "--title",
            f"[{tid}] {spec['title']}", "--body", body, "--label", LABEL_PR], cwd=REPO)
    log(f"{tid}: PR {p.stdout.strip() or p.stderr.strip()[-200:]}")
    sh(["gh", "issue", "edit", str(issue["number"]), "--remove-label", LABEL_READY,
        "--add-label", LABEL_PR], cwd=REPO)
    return "pr"


# -------------------------------------------------------------------------- main

def cmd_run(args) -> None:
    if args.dry_run:
        if not args.ids:
            sys.exit("--dry-run needs --ids (read from local spec/<ID> branches)")
        issues = [{"id": i, "number": 0, "title": ""} for i in args.ids]
    else:
        issues = ready_issues()
        if args.ids:
            issues = [i for i in issues if i["id"] in args.ids]
    issues = issues[: args.max_tasks]
    if not issues:
        log("nothing worker-ready")
        return
    spent = modal_month_spend()
    if spent >= MONTH_CAP:
        sys.exit(f"Modal month-to-date ${spent:.2f} >= cap ${MONTH_CAP:.2f}; not starting")
    # --llm routes to any LiteLLM model (e.g. a free one for plumbing tests) and
    # leaves Modal alone; otherwise the burst model is deployed for the batch.
    llm = args.llm or f"burst-{args.model}"
    ctx = int(os.environ.get("NC_LLM_CTX", 65536)) if args.llm else model_ctx(args.model)
    use_modal = not args.llm
    log(f"{len(issues)} task(s), llm {llm}, Modal month-to-date ${spent:.2f}/{MONTH_CAP:.2f}")
    if use_modal:
        modal_app(args.model, up=True)
    try:
        for issue in issues:
            if mem_available_mb() < MIN_FREE_MB:
                log(f"only {mem_available_mb()} MB free, stopping batch")
                break
            spent = modal_month_spend()
            if spent >= MONTH_CAP:
                log(f"Modal cap reached (${spent:.2f}), stopping batch")
                break
            try:
                log(f"{issue['id']}: {run_task(issue, llm, ctx, args.dry_run)}")
            except InfraError as e:
                log(f"{issue['id']}: backend failure, stopping batch: {e}")
                break
            except Exception as e:  # one broken task must not strand the GPU
                log(f"{issue['id']}: ERROR {e!r}")
    finally:
        if use_modal:
            modal_app(args.model, up=False)
        shutil.rmtree(WORK, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ids", nargs="+")
    r.add_argument("--model", default=os.environ.get("NC_WORKER_MODEL", "glm47-flash"),
                   choices=sorted(DEPLOY))
    r.add_argument("--max-tasks", type=int, default=5)
    r.add_argument("--llm", help="LiteLLM model name instead of a Modal burst model")
    r.add_argument("--dry-run", action="store_true",
                   help="local spec branches, no GitHub writes")
    sub.add_parser("queue")
    sub.add_parser("budget")
    args = ap.parse_args()
    if args.cmd == "queue":
        for i in ready_issues():
            print(f"#{i['number']} {i['title']}")
    elif args.cmd == "budget":
        print(f"Modal month-to-date ${modal_month_spend():.2f} / cap ${MONTH_CAP:.2f}")
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
