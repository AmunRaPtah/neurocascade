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
2. **Run.** `pipeline/orchestrate.py run` deploys the burst model, works the
   ready issues back to back while the container is warm, then stops the app.
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
NC_GATE_FULL=0 ...                                  # spec tests only (plumbing tests)
```

Run records: `/root/nc-pipeline-runs/<ID>/<timestamp>/attempt-N/`
(prompt, agent transcript, patch, pytest output, record.json).
