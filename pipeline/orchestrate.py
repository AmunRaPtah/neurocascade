#!/usr/bin/env python3
"""Architect/worker pipeline: Claude writes specs + failing tests, an open-weights
model on Modal writes the code, the gate and GitHub decide what lands.

    orchestrate.py queue                       # open worker-ready issues
    orchestrate.py run [--ids W1.1 ...] [--model glm47-flash] [--max-tasks N]
    orchestrate.py budget                      # Modal month-to-date vs cap
    orchestrate.py regate <run>/attempt-N [--issue N]   # re-gate a saved patch, no GPU
    orchestrate.py doctor [--reap] [--alert]   # health checks; cron stops stray apps

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
import fcntl
import hashlib
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
# `--llm agy` runs the Antigravity CLI instead of Claude Code against a LiteLLM
# model. agy needs Google's endpoints, so its container is on the default bridge
# rather than nc-sandbox; the filesystem confinement is unchanged (fresh clone at
# /work, data read-only, a throwaway HOME holding only a copy of the OAuth token).
# Quota is per model, so the probe takes the first model in AGY_MODELS that answers.
AGY = "agy"
AGY_BIN = Path("/root/.local/bin/agy")
AGY_TOKEN = Path("/root/.gemini/antigravity-cli/antigravity-oauth-token")
AGY_MODELS = os.environ.get("NC_AGY_MODELS",
                            "claude-sonnet-4-6,gemini-3.1-pro-high,gemini-3.8-flash-high").split(",")
_agy_model: str | None = None

MONTH_CAP = float(os.environ.get("NC_MODAL_MONTH_CAP", 25.0))
# The first real task flailed for 30 min / 160 tool calls on attempt 1 and was
# fixed in 135 s on attempt 2 with the failure log. A short leash plus the retry
# is cheaper than a long leash.
AGENT_TIMEOUT_S = int(os.environ.get("NC_AGENT_TIMEOUT_S", 35 * 60))
MAX_TURNS = int(os.environ.get("NC_MAX_TURNS", 100))
TEST_TIMEOUT_S = 40 * 60
# Router cold-start budget and LiteLLM timeout are both 1500 s; a cold start
# measured ~270 s, so anything near this is a broken backend, not a slow one.
READY_TIMEOUT_S = 1700
MAX_ATTEMPTS = 2
MIN_FREE_MB = 1500
LOCK = RUNS / ".lock"
BASELINES = RUNS / "_baseline"

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


def acquire_lock():
    """One pipeline at a time: two batches would share /tmp/ncw and the GPU app."""
    RUNS.mkdir(parents=True, exist_ok=True)
    f = LOCK.open("a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another orchestrate.py holds {LOCK}")
    f.truncate(0)
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f  # held until the process exits


def lock_held() -> bool:
    if not LOCK.exists():
        return False
    with LOCK.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


# --------------------------------------------------------------------- provenance

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_head(path: Path) -> str:
    head = sh(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
    dirty = sh(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    return head + ("-dirty" if dirty else "")


def image_id() -> str:
    return sh(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"]).stdout.strip()


def data_fingerprint() -> str:
    """data/ is gitignored, so the gate's inputs are pinned by listing, not by commit."""
    rows = sorted(f"{p.relative_to(DATA)}\t{p.stat().st_size}\t{int(p.stat().st_mtime)}"
                  for p in DATA.rglob("*") if p.is_file())
    return sha256("\n".join(rows))[:16]


def provenance(llm: str, model: str | None) -> dict:
    """Everything that decides what a run does, so a result can be traced and rerun."""
    prov = {
        "pipeline": git_head(REPO),
        "orchestrate_sha256": sha256(Path(__file__).read_text())[:16],
        "image": image_id(),
        "claude_code": sh(["docker", "run", "--rm", "--network", "none", IMAGE,
                           "claude", "--version"]).stdout.strip(),
        "data": data_fingerprint(),
        "llm": llm, "max_turns": MAX_TURNS, "agent_timeout_s": AGENT_TIMEOUT_S,
        "gate_full": os.environ.get("NC_GATE_FULL", "1") != "0",
    }
    if model:
        sys.path.insert(0, str(LLM_BURST))
        import models  # noqa: E402
        s = models.get(model)
        prov.update(llm_burst=git_head(LLM_BURST), model_repo=s.repo, model_revision=s.revision,
                    gpu=s.gpu, max_model_len=s.max_model_len, vllm=models.VLLM_VERSION)
    return prov


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
    return spec_at(sha, task_id), sha


def spec_at(sha: str, task_id: str) -> dict:
    text = sh(["git", "-C", str(REPO), "show", f"{sha}:tasks/specs/{task_id}.md"],
              check=True).stdout
    return parse_spec(text)


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


def live_burst_apps() -> list[str]:
    """Deployed llm-burst-<model> apps. Raises if Modal cannot be asked."""
    p = sh(["modal", "app", "list", "--json"], timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"modal app list failed: {p.stderr[-300:]}")
    names = {f"llm-burst-{m}" for m in DEPLOY}
    return sorted({a["description"] for a in json.loads(p.stdout)
                   if a["description"] in names and a["state"] != "stopped"})


ROUTER_UNIT = "llm-burst-router"   # :8100, the proxy's route to the Modal apps
_router_started = False


def router(up: bool) -> None:
    """The router is started for a Modal batch and stopped after it, unless it was
    already running: it is disabled between batches to save host RAM (2026-09-25)."""
    global _router_started
    if up:
        if sh(["systemctl", "is-active", "--quiet", ROUTER_UNIT]).returncode == 0:
            return
        log(f"starting {ROUTER_UNIT}")
        if sh(["systemctl", "start", ROUTER_UNIT]).returncode != 0:
            raise InfraError(f"could not start {ROUTER_UNIT}")
        _router_started = True
        for _ in range(30):
            if sh(["curl", "-s", "-o", "/dev/null", "http://127.0.0.1:8100/"]).returncode == 0:
                return
            time.sleep(1)
        raise InfraError(f"{ROUTER_UNIT} started but :8100 is not answering")
    if _router_started:
        log(f"stopping {ROUTER_UNIT}")
        sh(["systemctl", "stop", ROUTER_UNIT])
        _router_started = False


def modal_app(model: str, up: bool) -> None:
    app = f"llm-burst-{model}"
    if up:
        router(True)
        log(f"deploying {app}")
        p = sh(["modal", "deploy", f"modal_app.py::{DEPLOY[model]}"], cwd=LLM_BURST)
        if p.returncode != 0:
            raise InfraError(f"modal deploy failed: {(p.stderr or p.stdout)[-500:]}")
        return
    # A deployed app idles at zero cost, but anything that reaches the router
    # wakes a GPU, so stopping is verified rather than assumed.
    for _ in range(3):
        log(f"stopping {app}")
        sh(["modal", "app", "stop", "-y", app], timeout=120)
        try:
            if app not in live_burst_apps():
                router(False)
                return
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            log(f"could not verify stop: {e}")
        time.sleep(10)
    log(f"WARNING: {app} still deployed after 3 stops; `orchestrate.py doctor --reap` will retry")
    router(False)


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


PROBE = r"""
import json, os, time, urllib.request
t0 = time.time()
req = urllib.request.Request(
    os.environ["PROXY"] + "/v1/messages",
    data=json.dumps({"model": os.environ["LLM"], "max_tokens": 16,
                     "messages": [{"role": "user", "content": "Reply with OK."}]}).encode(),
    headers={"Authorization": "Bearer " + os.environ["KEY"], "content-type": "application/json",
             "anthropic-version": "2023-06-01"})
body = json.load(urllib.request.urlopen(req, timeout=float(os.environ["T"])))
print(json.dumps({"ready_s": round(time.time() - t0, 1), "model": body.get("model"),
                  "stop_reason": body.get("stop_reason")}))
"""


def agy_home(adir: Path) -> Path:
    home = adir / "agy-home"
    (home / ".gemini" / "antigravity-cli").mkdir(parents=True, exist_ok=True)
    shutil.copy(AGY_TOKEN, home / ".gemini" / "antigravity-cli" / AGY_TOKEN.name)
    sh(["chown", "-R", str(WORKER_UID), str(home)], check=True)
    home.chmod(0o700)
    return home


def agy_run(name: str, model: str, prompt: str, mounts: list[str], home: Path,
            timeout: int, out: Path) -> dict:
    """One agy print-mode turn in the worker image; returns its JSON result."""
    rc = docker_run(name, [*mounts, f"{AGY_BIN}:/usr/local/bin/agy:ro", f"{home}:/home/agy"],
                    ["agy", "--model", model, "--dangerously-skip-permissions",
                     "--print-timeout", f"{timeout}s", "--output-format", "json", "-p", prompt],
                    network="bridge", env={"HOME": "/home/agy"}, timeout=timeout + 120,
                    stdout_path=out)
    text = out.read_text(errors="replace")
    try:
        obj = json.loads(text[text.index("{"):])
    except ValueError:
        obj = {}
    return {"rc": rc, **obj}


def agy_ready(out_dir: Path) -> float:
    """Pick the first AGY_MODELS entry that answers; quota is tracked per model,
    and an exhausted one can exit 0 with an empty response."""
    t0 = time.time()
    home = agy_home(out_dir)
    tried = []
    try:
        return _agy_pick(home, out_dir, tried, t0)
    finally:
        shutil.rmtree(home, ignore_errors=True)   # no token copies left in run dirs


def _agy_pick(home: Path, out_dir: Path, tried: list, t0: float) -> float:
    global _agy_model
    for m in AGY_MODELS:
        r = agy_run("ncw-probe", m, "Reply with the single word PONG", [], home, 120,
                    out_dir / f"probe-{m}.txt")
        if r.get("status") == "SUCCESS" and "PONG" in (r.get("response") or ""):
            _agy_model = m
            log(f"agy model {m}")
            return round(time.time() - t0, 1)
        tried.append(f"{m}: rc={r['rc']} {str(r.get('error') or r.get('response') or '')[:80]}")
    raise InfraError("no agy model answered: " + "; ".join(tried))


def wait_ready(llm: str, out: Path) -> float:
    """One real Anthropic-format request from inside the sandbox, the exact path the
    worker uses (container -> socat -> LiteLLM -> router -> Modal). It absorbs the
    cold start before the worker's clock starts, and a broken link anywhere fails
    here as InfraError instead of as a worker attempt."""
    t0 = time.time()
    rc = docker_run("ncw-probe", [], ["python", "-c", PROBE], network=NETWORK,
                    env={"PROXY": PROXY, "LLM": llm, "KEY": proxy_key(), "T": str(READY_TIMEOUT_S)},
                    timeout=READY_TIMEOUT_S + 60, stdout_path=out)
    if rc != 0:
        raise InfraError(f"backend not ready (rc={rc}): {out.read_text(errors='replace')[-400:]}")
    return round(time.time() - t0, 1)


def fresh_clone(dest: Path, sha: str) -> None:
    """Host-owned clone at `sha`, never touched by worker code.

    Spec and PR heads are often reachable only from REPO's origin/* refs, which
    a clone does not copy, so a missing commit is fetched by id."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "-q", "--no-local", "--no-checkout", str(REPO), str(dest)], check=True)
    if sh(["git", "-C", str(dest), "cat-file", "-e", f"{sha}^{{commit}}"]).returncode:
        sh(["git", "-C", str(dest), "fetch", "-q", str(REPO), sha], check=True)
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
    if llm == AGY:
        return run_agy_worker(spec, prompt, tree, adir)
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
                     "--max-turns", str(MAX_TURNS),
                     "--output-format", "stream-json", "--verbose"],
                    network=NETWORK, env=env, timeout=AGENT_TIMEOUT_S,
                    stdout_path=adir / "agent.jsonl")
    res = {"rc": rc, "agent_s": round(time.time() - t0),
           **parse_agent_log((adir / "agent.jsonl").read_text(errors="replace"))}
    (adir / "final.txt").write_text(res["final"])
    if res["api_error"]:
        raise InfraError(f"LLM API error {res['api_error']}: {res['final'][:200]}")
    return res


def run_agy_worker(spec: dict, prompt: str, tree: Path, adir: Path) -> dict:
    # agy has no turn cap; the print timeout is the leash.
    t0 = time.time()
    home = agy_home(adir)
    try:
        r = agy_run(f"ncw-{spec['id'].replace('.', '-')}-agent", _agy_model, prompt,
                    [f"{tree}:/work", f"{DATA}:/work/data:ro"], home,
                    AGENT_TIMEOUT_S, adir / "agent.json")
    finally:
        shutil.rmtree(home, ignore_errors=True)
    final = r.get("response") or ""
    res = {"rc": r["rc"], "agent_s": round(time.time() - t0), "final": final,
           "turns": r.get("num_turns"), "subtype": r.get("status"), "api_error": None,
           "model": _agy_model}
    (adir / "final.txt").write_text(final)
    if r.get("status") != "SUCCESS" or not final.strip():
        if r["rc"] == 124:
            return res        # our timeout: a worker failure, not a backend one
        raise InfraError(f"agy/{_agy_model} failed rc={r['rc']}: "
                         f"{str(r.get('error') or 'empty response, likely quota')[:200]}")
    return res


def parse_agent_log(text: str) -> dict:
    """The stream-json `result` event: final text, turn count, how it ended."""
    res = {"final": "", "turns": None, "subtype": None, "api_error": None}
    for line in text.splitlines():
        if not line.startswith("{") or '"result"' not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Key order varies between Claude Code versions, so match on the value.
        if obj.get("type") == "result":
            res.update(final=obj.get("result") or "", turns=obj.get("num_turns"),
                       subtype=obj.get("subtype"))
            if obj.get("terminal_reason") == "api_error" or obj.get("api_error_status"):
                res["api_error"] = str(obj.get("api_error_status") or "api_error")
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


def gate_full() -> bool:
    return os.environ.get("NC_GATE_FULL", "1") != "0"   # 0 only for plumbing smoke tests


def sandbox_pytest(tree: Path, script: str, name: str, out: Path) -> tuple[bool, str]:
    """Run `script` on a disposable copy of `tree`, no network, data/ read-only."""
    copy = tree.parent / (tree.name + "-test")
    disposable_copy(tree, copy)
    rc = docker_run(name, [f"{copy}:/work", f"{DATA}:/work/data:ro"], ["sh", "-c", script],
                    network="none", timeout=TEST_TIMEOUT_S, stdout_path=out)
    shutil.rmtree(copy, ignore_errors=True)
    return rc == 0, "\n".join(out.read_text(errors="replace").strip().splitlines()[-40:])


def run_gate(tree: Path, spec: dict, adir: Path) -> tuple[bool, str]:
    """Spec tests first (fast fail), then the not-slow suite."""
    script = f"python -m pytest -q -p no:cacheprovider {' '.join(spec['tests'])}"
    if gate_full():
        script += " && python -m pytest -q -p no:cacheprovider -m 'not slow'"
    return sandbox_pytest(tree, script, f"ncw-gate-{spec['id'].replace('.', '-')}-{adir.name}",
                          adir / "pytest.txt")


def exclude_args(tests: list[str]) -> str:
    return " ".join(f"--deselect {t}" if "::" in t else f"--ignore {t}" for t in tests)


def baseline(spec: dict, sha: str) -> str | None:
    """Is the spec branch itself sound? None if yes, else why not.

    The spec tests must fail (else they test nothing) and the rest of the suite
    must pass, so that a gate failure can only be the worker's. The first real
    run lost an attempt to missing F-TRACT data that looked like a worker bug.
    Runs before any GPU is up. Passes are cached per (commit, image, data)."""
    key = f"{sha[:12]}-{image_id()[7:19]}-{data_fingerprint()}"
    cache = BASELINES / f"{key}.json"
    if cache.exists():
        return None
    BASELINES.mkdir(parents=True, exist_ok=True)
    tree = WORK / "baseline" / key
    fresh_clone(tree, sha)
    tests = " ".join(spec["tests"])
    script = (f"if python -m pytest -q -p no:cacheprovider {tests}; then "
              f"echo 'SPEC TESTS PASS BEFORE ANY WORK'; exit 3; fi")
    if gate_full():
        script += f"; python -m pytest -q -p no:cacheprovider -m 'not slow' {exclude_args(spec['tests'])}"
    else:
        script += "; true"
    log(f"{spec['id']}: baseline check at {sha[:8]}")
    ok, tail = sandbox_pytest(tree, script, f"ncw-base-{spec['id'].replace('.', '-')}",
                              BASELINES / f"{key}.txt")
    shutil.rmtree(tree, ignore_errors=True)
    if ok:
        cache.write_text(json.dumps({"sha": sha, "spec": spec["id"], "at": time.time()}))
        return None
    return tail


def result_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip().startswith("TASK_RESULT:"):
            return line.strip()
    return "TASK_RESULT: (missing)"


# -------------------------------------------------------------------------- task

def run_task(issue: dict, spec: dict, base: str, llm: str, ctx: int, prov: dict,
             dry: bool = False) -> str:
    tid = spec["id"]
    tdir = RUNS / tid / time.strftime("%Y%m%d-%H%M%S")
    tdir.mkdir(parents=True)
    (tdir / "provenance.json").write_text(json.dumps(
        {**prov, "task": tid, "issue": issue["number"], "spec_sha": base}, indent=2))
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
        rec = {"task": tid, "attempt": attempt, "model": llm, "spec_sha": base,
               "prompt_sha256": sha256((adir / "prompt.txt").read_text())[:16],
               **{k: v for k, v in agent.items() if k not in ("final", "api_error")},
               "result_line": result_line(agent["final"]), "files": patch_paths(patch)}
        log(f"{tid}: agent rc={agent['rc']} {agent['agent_s']}s turns={agent['turns']} "
            f"{rec['result_line'][:100]} files={len(rec['files'])}")

        reason = check_patch(patch, spec) or apply_patch(host, base, patch)
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
            return open_pr(issue, spec, host, records, agent["final"], prov)

    if dry:
        log(f"{tid}: dry run FAIL; records in {tdir}")
        return "failed (dry run)"
    relabel(issue["number"], LABEL_FAILED)
    sh(["gh", "issue", "comment", str(issue["number"]), "--body",
        f"Worker ({llm}) failed after {MAX_ATTEMPTS} attempts.\n\n"
        f"Last failure:\n```\n{(failure or '')[-2500:]}\n```\n"
        f"Records: `{tdir}`. Needs rescoping or architect takeover."], cwd=REPO)
    return "failed"


def apply_patch(host: Path, base: str, patch: str) -> str | None:
    """Reset the host clone to `base` and apply the patch as data. None on success."""
    sh(["git", "-C", str(host), "checkout", "-q", "--detach", base], check=True)
    sh(["git", "-C", str(host), "reset", "-q", "--hard"], check=True)
    p = sh(["git", "-C", str(host), "apply", "--index", "-"], input=patch)
    return None if p.returncode == 0 else f"patch does not apply: {p.stderr[-300:]}"


def open_pr(issue: dict, spec: dict, host: Path, records: list[dict], final: str,
            prov: dict) -> str:
    tid, branch = spec["id"], f"worker/{spec['id']}"
    msg = (f"{tid}: {spec['title']}\n\nImplemented by the pipeline worker "
           f"({records[-1]['model']}, attempt {records[-1]['attempt']}) against the spec "
           f"tests on spec/{tid}.\n\nCloses #{issue['number']}")
    sh(["git", "-C", str(host), "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", msg], check=True)
    sh(["git", "-C", str(host), "push", "-q", "-f", "origin", f"HEAD:refs/heads/{branch}"], check=True)
    body = (f"Closes #{issue['number']}\n\n**Spec:** `tasks/specs/{tid}.md` on `spec/{tid}`\n"
            f"**Worker:** {records[-1]['model']}, {len(records)} attempt(s)\n\n"
            f"```json\n{json.dumps(records, indent=1)}\n```\n\n"
            f"<details><summary>Provenance</summary>\n\n```json\n{json.dumps(prov, indent=1)}\n```\n</details>\n\n"
            f"**Worker's summary:**\n\n"
            f"{final[-3000:]}\n")
    p = sh(["gh", "pr", "create", "--base", "main", "--head", branch, "--title",
            f"[{tid}] {spec['title']}", "--body", body, "--label", LABEL_PR], cwd=REPO)
    log(f"{tid}: PR {p.stdout.strip() or p.stderr.strip()[-200:]}")
    sh(["gh", "issue", "edit", str(issue["number"]), "--remove-label", LABEL_READY,
        "--add-label", LABEL_PR], cwd=REPO)
    return "pr"


# -------------------------------------------------------------------------- main

def cmd_run(args) -> None:
    lock = acquire_lock()  # noqa: F841 (held for the life of the process)
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
    # --llm routes to any LiteLLM model (e.g. a free one for plumbing tests) and
    # leaves Modal alone; otherwise the burst model is deployed for the batch.
    llm = args.llm or f"burst-{args.model}"
    ctx = int(os.environ.get("NC_LLM_CTX", 65536)) if args.llm else model_ctx(args.model)
    use_modal = not args.llm
    # The Modal cap only binds batches that use Modal; --llm is the fallback for
    # when the month is spent.
    spent = modal_month_spend() if use_modal else 0.0
    if spent >= MONTH_CAP:
        sys.exit(f"Modal month-to-date ${spent:.2f} >= cap ${MONTH_CAP:.2f}; not starting")
    prov = provenance(llm, None if args.llm else args.model)
    if use_modal and prov["llm_burst"].endswith("-dirty") and os.environ.get("NC_ALLOW_DIRTY") != "1":
        sys.exit(f"{LLM_BURST} has uncommitted changes; commit them so the deployed server "
                 "is reproducible (or NC_ALLOW_DIRTY=1)")

    # Everything that can fail without a GPU fails here, before one is up.
    runnable = []
    try:
        for issue in issues:
            try:
                spec, base = load_spec(issue["id"], local=args.dry_run)
                why = baseline(spec, base)
            except Exception as e:
                log(f"{issue['id']}: cannot load or baseline the spec, skipped: {e!r}")
                continue
            if why is None:
                runnable.append((issue, spec, base))
                continue
            log(f"{issue['id']}: spec branch unsound, skipped:\n{why}")
            if not args.dry_run:
                relabel(issue["number"], "needs-human")
                sh(["gh", "issue", "comment", str(issue["number"]), "--body",
                    "Baseline check failed on the spec branch before any worker ran "
                    "(spec tests already pass, or the rest of the suite is red):\n"
                    f"```\n{why[-2500:]}\n```"], cwd=REPO)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    if not runnable:
        log("no runnable tasks; GPU not started")
        return

    log(f"{len(runnable)} task(s), llm {llm}, Modal month-to-date ${spent:.2f}/{MONTH_CAP:.2f}")
    bdir = RUNS / "_batches" / time.strftime("%Y%m%d-%H%M%S")
    bdir.mkdir(parents=True)
    results = {}
    try:
        if use_modal:
            modal_app(args.model, up=True)
        prov["ready_s"] = agy_ready(bdir) if llm == AGY else wait_ready(llm, bdir / "probe.txt")
        if llm == AGY:
            prov["agy_model"] = _agy_model
        log(f"backend ready after {prov['ready_s']}s")
        for issue, spec, base in runnable:
            if mem_available_mb() < MIN_FREE_MB:
                log(f"only {mem_available_mb()} MB free, stopping batch")
                break
            if use_modal and modal_month_spend() >= MONTH_CAP:
                log("Modal cap reached, stopping batch")
                break
            try:
                results[spec["id"]] = run_task(issue, spec, base, llm, ctx, prov, args.dry_run)
            except InfraError:
                results[spec["id"]] = "infra"
                raise
            except Exception as e:  # one broken task must not strand the GPU
                results[spec["id"]] = f"error {e!r}"
            log(f"{spec['id']}: {results[spec['id']]}")
    except InfraError as e:
        log(f"backend failure, batch stopped: {e}")
    finally:
        if use_modal:
            modal_app(args.model, up=False)
        shutil.rmtree(WORK, ignore_errors=True)
        (bdir / "batch.json").write_text(json.dumps(
            {**prov, "spend_at_start": spent, "results": results}, indent=2))


def cmd_regate(args) -> None:
    """Re-run the gate on a saved attempt's patch: no GPU, no worker.

    For when the environment, not the worker, failed the gate (fix it, then
    regate). With --issue N a pass opens the PR as a normal run would."""
    lock = acquire_lock()  # noqa: F841
    adir = Path(args.attempt).resolve()
    rec = json.loads((adir / "record.json").read_text())
    tid, base = rec["task"], rec["spec_sha"]
    spec = spec_at(base, tid)
    patch = (adir / "diff.patch").read_text()
    host = WORK / tid / "host"
    try:
        fresh_clone(host, base)
        reason = check_patch(patch, spec) or apply_patch(host, base, patch)
        if reason:
            sys.exit(f"{tid}: {reason}")
        rdir = adir / f"regate-{time.strftime('%Y%m%d-%H%M%S')}"
        rdir.mkdir()
        ok, tail = run_gate(host, spec, rdir)
        log(f"{tid}: regate {'PASS' if ok else 'FAIL'} (log {rdir / 'pytest.txt'})")
        if not ok:
            print(tail)
            sys.exit(1)
        if args.issue:
            prov = json.loads((adir.parent / "provenance.json").read_text())
            final = (adir / "final.txt").read_text() if (adir / "final.txt").exists() else ""
            log(f"{tid}: " + open_pr({"number": args.issue, "id": tid}, spec, host,
                                     [{**rec, "gate": True, "regated": True}], final, prov))
    finally:
        shutil.rmtree(WORK, ignore_errors=True)


def cmd_doctor(args) -> None:
    """Health of every link the pipeline needs; `--reap` stops stray burst apps.

    Meant for cron: a burst app left deployed after a crash costs nothing idle,
    but any stray request through the router would wake a GPU."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    check("image", bool(image_id()), IMAGE)
    check("network", sh(["docker", "network", "inspect", NETWORK]).returncode == 0, NETWORK)
    # llm-burst-router is started per Modal batch (router()), so it is not required here.
    for unit in ("nc-sandbox-proxy", "pardalos-freeproxy"):
        state = sh(["systemctl", "is-active", unit]).stdout.strip()
        check(unit, state == "active", state)
    # Confinement both ways: the proxy is reachable, the internet is not.
    probe = ("import urllib.request as u,sys\n"
             "def get(x):\n"
             " try: u.urlopen(x, timeout=8); return True\n"
             " except Exception as e: return getattr(e, 'code', None) is not None\n"
             f"print(get('{PROXY}/health/liveliness'), get('https://api.github.com'))")
    out = RUNS / "doctor-net.txt"
    RUNS.mkdir(parents=True, exist_ok=True)
    docker_run("ncw-doctor", [], ["python", "-c", probe], network=NETWORK, timeout=60, stdout_path=out)
    net = out.read_text().split()
    check("sandbox->proxy", net[:1] == ["True"], " ".join(net))
    check("sandbox-/->internet", net[1:2] == ["False"], " ".join(net))
    try:
        spent = modal_month_spend()
        check("modal budget", spent < MONTH_CAP, f"${spent:.2f}/{MONTH_CAP:.2f}")
    except Exception as e:
        check("modal budget", False, str(e)[:200])
    busy = lock_held()
    try:
        live = live_burst_apps()
        stray = [] if busy else live
        if stray and args.reap:
            for app in stray:
                sh(["modal", "app", "stop", "-y", app], timeout=120)
            stray = live_burst_apps()
        check("no stray burst app", not stray,
              ("run in progress: " if busy else "") + (", ".join(live) or "none"))
    except Exception as e:
        check("no stray burst app", False, str(e)[:200])
    # The conductor (pipeline/conductor.py) ticks every 20 min; one architect
    # session can hold a tick for ~2 h, so a held lock counts as alive.
    cdir = RUNS / "conductor"
    try:
        age = (time.time() - json.loads((cdir / "state.json").read_text())["last_tick"]) / 60
    except Exception:
        age = None
    lock = cdir / ".lock"
    held = False
    if lock.exists():
        with lock.open("a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                held = True
    paused = (cdir / "PAUSE").exists()
    check("conductor heartbeat", paused or held or (age is not None and age < 90),
          "paused" if paused else "session running" if held else
          f"last tick {age:.0f} min ago" if age is not None else "never ticked")
    free_gb = shutil.disk_usage("/tmp").free / 2**30
    check("disk /tmp", free_gb > 5, f"{free_gb:.0f} GB free")

    bad = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        if not (args.quiet and ok):
            print(f"{'ok  ' if ok else 'FAIL'} {name:22} {detail}")
    if args.alert:
        alert_on_change(bad)
    if bad:
        sys.exit(1)


def alert_on_change(bad: list) -> None:
    """Telegram only when the set of failing checks changes, not every cron tick."""
    state_f = RUNS / "doctor-state.json"
    now = sorted(name for name, _, _ in bad)
    before = json.loads(state_f.read_text()) if state_f.exists() else []
    if now == before:
        return
    state_f.write_text(json.dumps(now))
    sys.path.insert(0, str(REPO / "scripts"))
    from digest import send  # noqa: E402
    send("neurocascade pipeline doctor: " +
         ("all checks pass again" if not now else
          "FAILING " + "; ".join(f"{n} ({d})" for n, _, d in bad)))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ids", nargs="+")
    r.add_argument("--model", default=os.environ.get("NC_WORKER_MODEL", "glm47-flash"),
                   choices=sorted(DEPLOY))
    r.add_argument("--max-tasks", type=int, default=5)
    r.add_argument("--llm", help="LiteLLM model name instead of a Modal burst model, "
                   "or 'agy' for the Antigravity CLI")
    r.add_argument("--dry-run", action="store_true",
                   help="local spec branches, no GitHub writes")
    g = sub.add_parser("regate", help="re-gate a saved attempt's patch without a GPU")
    g.add_argument("attempt", help="run dir .../<ID>/<ts>/attempt-N")
    g.add_argument("--issue", type=int, help="open the PR for this issue on a pass")
    d = sub.add_parser("doctor")
    d.add_argument("--reap", action="store_true", help="stop burst apps no run is using")
    d.add_argument("--quiet", action="store_true", help="print failures only")
    d.add_argument("--alert", action="store_true", help="Telegram when the failure set changes")
    sub.add_parser("queue")
    sub.add_parser("budget")
    args = ap.parse_args()
    if args.cmd == "queue":
        for i in ready_issues():
            print(f"#{i['number']} {i['title']}")
    elif args.cmd == "budget":
        print(f"Modal month-to-date ${modal_month_spend():.2f} / cap ${MONTH_CAP:.2f}")
    elif args.cmd == "regate":
        cmd_regate(args)
    elif args.cmd == "doctor":
        cmd_doctor(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
