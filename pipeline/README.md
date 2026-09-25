# Architect / worker pipeline

Claude is the architect: it turns the roadmap into small specs with failing
tests and supplies every scientific number with its citation. An open-weights
model served from Modal (llm-burst) is the worker: it writes code until those
tests pass. GitHub holds the queue (issues), the record (PRs) and the second
gate (CI). The conductor (below) runs the whole loop unattended; the user gets
Telegram only on state changes and the daily digest.

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

`NC_MODAL_MONTH_CAP` (default $25 of the $30 free tier) binds Modal batches
only (`run --llm` skips it) and is checked against
`modal billing report` before the batch and before each task. The burst app, and
the `llm-burst-router` service in front of it (disabled between batches to save
host RAM), are started only for the batch and stopped after, including on
errors. A Modal batch that stops on a backend failure parks Modal for 2 h, and
Gemini takes the queue meanwhile. LLM API
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

## Conductor (unattended loop)

`pipeline/conductor.py tick` runs from cron every 20 min
(`pipeline/conductor-cron.sh`) and takes one step per tick, first match wins:

1. **Merge** PRs labelled `approved` once CI is green. A merged `architect/<T>`
   PR whose body says `Backlog-Task: <T>` marks backlog task T done. An
   approved PR whose CI fails loses `approved` and is reviewed again.
2. **Review** each `worker-pr` PR once CI has finished: a Sonnet architect
   session checks it against the spec. It either approves, after the host
   pushes any small cleanup through the gate, or rejects: the PR is closed and
   the issue goes to `worker-failed`.
3. **Work**: runs `orchestrate.py run` on `worker-ready` issues while the Modal
   month has headroom, and on free-tier Gemini once it has not (below).
4. **Takeover**: `worker-failed` and `needs-human` issues are handled by an Opus
   architect session, and so are ready issues while Modal is out of budget.
   The session works on the spec branch, and the result is gated and PR'd as
   `architect-pr` + `approved`. At most 2 tries per issue.
5. **Advance**: the next backlog task whose deps are done gets an Opus
   architect session. It ends with one of:
   - `TASK_RESULT: COMPLETE`: patch, full `not slow` gate, PR.
   - `TASK_RESULT: SPLIT <ids>`: specs + failing tests only. The host
     publishes one `spec/<sub>` branch and one `worker-ready` issue per
     subtask, and the task comes back once they have merged.
   - `BLOCKED_ON_DATA` / `BLOCKED`.

   Each session is recorded in the task's `conductor.history`. After 6
   sessions the task is given up and an alert is sent.

**Architect sessions** run headless Claude Code in the same `nc-worker` image
as uid 20001, on a throwaway clone. They have internet for literature. They
have no GitHub credential and no host paths. The OAuth token reaches the
container through environment inheritance, never on a command line. The patch
is checked against a path policy: no `.github/ pipeline/ tasks/backlog.json
data/ requirements.txt scripts/autonomous_runner.py`. The host then applies it
to its own clone, gates it in a `--network none` container, and does every
push and GitHub write itself.

**Limits, and resuming after them.** Every tick re-reads the live readings, so
nothing has to wake it:

- **Claude.** A session starts only if the 5-hour window has at least 35%
  left, and the weekly window's usage is under a pace line that rises from 20%
  to 75% across the week. The rest is left for interactive use. Every session
  writes its own `rate_limit_event` readings back, and a window whose
  `resets_at` has passed stops constraining.
- **Modal.** Work runs while month spend plus $1.50 stays under
  `NC_MODAL_MONTH_CAP`. The new billing month re-opens it.
- **Gemini (second worker).** While Modal is spent, ready issues go to
  `orchestrate.py run --llm worker-gemini`, 2 per batch: free-tier
  gemini-2.5-flash through the LiteLLM proxy (entry in pardalos
  `deploy/litellm-free.yaml`, rpm 8, 3.5-flash as fallback). A batch that stops
  on a backend or quota failure parks Gemini for 3 h. Only while both workers
  are unavailable do ready issues fall to the architect. `NC_GEMINI_LLM=`
  (empty) disables it. Free-tier prompts may be used by Google for training;
  the repo is private but the code sent is not.

A session that dies from an API or quota error does not count against the
task.

**Alerts**, each sent once per distinct message:

- a task done
- a task blocked or given up
- a failed takeover
- tick errors
- no progress for 36 h while not waiting on a limit
- the backlog complete

`doctor` also checks the conductor heartbeat.

```
pipeline/conductor.py status          # backlog, limits, what it is waiting on
pipeline/conductor.py tick --dry-run  # what the next tick would do
pipeline/conductor.py labels          # create the GitHub labels
touch /root/nc-pipeline-runs/conductor/PAUSE   # pause; rm to resume
```
Session records: `/root/nc-pipeline-runs/conductor/<ID>/<timestamp>-<mode>/`.
