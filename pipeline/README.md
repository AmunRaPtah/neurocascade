# Architect / worker pipeline

Claude is the architect: it turns the roadmap into small specs with failing
tests and supplies every scientific number with its citation. An open-weights
model served from Modal (llm-burst) is the worker: it writes code until those
tests pass. GitHub holds the queue (issues), the record (PRs) and the second
gate (CI). The user steps in only on `needs-human` issues and the daily digest.

## Flow

1. **Spec.** The architect branches `spec/<ID>` from main, commits
   `tasks/specs/<ID>.md` plus the failing tests, pushes it, and opens an issue
   titled `[<ID>] <title>` with label `worker-ready`.
2. **Run.** `pipeline/orchestrate.py run` first checks each spec branch with
   no GPU up (the baseline, below). Then it deploys the burst model and waits
   for one real request to succeed from inside the sandbox. It works the ready
   issues back to back while the container is warm, then stops the app and
   verifies that it stopped.
3. **Gate.** Per task: the patch must be non-empty, touch `src/`, and leave the
   spec's protected tests and `.github/ pipeline/ tasks/ data/` alone; then the
   spec tests and the `not slow` suite must pass. One retry, with the failure
   fed back and the previous patch kept.
4. **PR.** On a pass the host commits the patch to `worker/<ID>` and opens a PR
   (`worker-pr`) that closes the issue; CI runs on it. On a second failure the
   issue gets `worker-failed` and a comment with the failure, for the architect
   to rescope or take over.
5. **Review.** The architect reviews the diff against the spec and merges.

## Spec format

```
---
id: P22.2a
title: One line
tests: [tests/test_x.py]          # must pass; also protected by default
protected: [tests/test_x.py]      # optional, worker may not touch these
---
Brief: what to build, function signatures, and every numeric input with its
source (PMID / NDA + review date). The worker is told never to invent values.
```

## Sandbox (why it is built like this)

On 2026-09-24 a bake-off worker with host access ran `rm -rf` on the real
repo. Now:

* The worker runs in the `nc-worker` image (`pipeline/worker.Dockerfile`) as
  uid 20001, all capabilities dropped, 3 GB RAM, on the internal docker network
  `nc-sandbox`. It sees only its own throwaway clone at `/work` and `data/`
  read-only. It holds no GitHub credential.
* The only reachable host port is the LiteLLM proxy, forwarded onto the
  network gateway `172.31.250.1:4000` by `nc-sandbox-proxy.service`, which also
  installs the iptables rules that drop everything else from that bridge. The
  rules name the bridge interface, so recreating the network means updating
  the unit.
* The worker's output is a patch computed inside a container. The host never
  runs git in a tree the worker controlled (hooks and `.git/config` are code);
  it applies the patch as data to its own clean clone.
* Tests execute worker code, so they run in a `--network none` container on a
  disposable copy, never on the tree the host commits from.

## Reliability and reproducibility

* **Baseline.** Before any GPU starts, each spec branch is gated with no worker
  change. The spec tests must fail, or they test nothing. The rest of the
  `not slow` suite must pass, so a later gate failure can only be the worker's.
  (The first real run lost an attempt to missing F-TRACT data.) An unsound
  spec gets `needs-human` and a comment. Passes are cached in
  `/root/nc-pipeline-runs/_baseline/` per (commit, image id, data/ listing).
* **Readiness probe.** One Anthropic-format request goes from a container on
  `nc-sandbox`, the exact path the worker uses. It absorbs the ~5 min cold
  start before the worker's clock starts. A broken proxy, router or Modal app
  stops the batch as an infra failure instead of burning an attempt.
* **Short leash.** Each attempt is capped at 100 turns and 35 min
  (`NC_MAX_TURNS`, `NC_AGENT_TIMEOUT_S`); the retry gets the failure log.
  The first task flailed for 30 min on attempt 1 and was fixed in 135 s on
  attempt 2.
* **One run at a time.** `run` and `regate` hold the flock
  `/root/nc-pipeline-runs/.lock`.
* **Provenance.** Every run records the following in `provenance.json` next
  to the attempts, and in the PR body:
  - pipeline commit
  - orchestrator hash
  - image id
  - Claude Code version
  - data/ fingerprint
  - llm-burst commit
  - model repo and revision
  - vLLM version
  - GPU
  - turn and time limits

  Each attempt's `record.json` adds the spec commit and a prompt hash.
  `run` refuses to deploy an llm-burst tree with uncommitted changes.
* **Pinned environment.** `worker.Dockerfile` pins the Python, Node and
  Claude Code versions. The Python packages are installed with
  `-c pipeline/worker-constraints.txt`, the freeze of the validated image, and
  CI installs the same constraints, so the local gate and CI test the same
  versions. The LLM is the one nondeterministic part: rerunning a task can
  produce a different patch, but the gate that judges it is fixed.
* **Doctor.** `doctor` checks:
  - the image and network
  - the three services
  - sandbox confinement: proxy reachable, internet not
  - Modal budget
  - disk
  - burst apps left deployed with no run holding the lock

  Cron runs `doctor --reap --quiet --alert` every 30 min. It stops stray apps
  and sends a Telegram message only when the set of failing checks changes.

## Budget

`NC_MODAL_MONTH_CAP` (default $25 of the $30 free tier) is checked against
`modal billing report` before the batch and before each task. The burst app is
deployed only for the batch and stopped after, including on errors. LLM API
errors stop the batch without spending an attempt. The existing free-memory
floor (1500 MB) applies too.

## Commands

```
pipeline/orchestrate.py queue
pipeline/orchestrate.py budget
pipeline/orchestrate.py run [--ids ID ...] [--model glm47-flash|qwen38-27b|qwen3-coder-next] [--max-tasks N]
pipeline/orchestrate.py run --dry-run --ids ID      # local spec/<ID>, no GitHub writes
pipeline/orchestrate.py regate <run>/attempt-N [--issue N]  # re-gate a saved patch, no GPU
pipeline/orchestrate.py doctor [--reap] [--quiet] [--alert]
NC_GATE_FULL=0 ...                                  # spec tests only (plumbing tests)
python -m pytest -q pipeline/test_orchestrate.py    # orchestrator unit tests
```

After fixing an environment problem that failed a gate, use `regate` on the
saved attempt instead of paying for a new GPU run.

Run records: `/root/nc-pipeline-runs/<ID>/<timestamp>/attempt-N/`
(prompt, agent transcript, final text, patch, pytest output, record.json;
`provenance.json` one level up). Batch summaries: `_batches/<timestamp>/`.
