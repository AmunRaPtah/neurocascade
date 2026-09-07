# The autonomous build

The roadmap in `docs/roadmap.md` is decomposed into 39 tasks in
`tasks/backlog.json`. A cron tick runs them one after another without manual
intervention, and stops on its own when the backlog is complete.

## How a tick works

`scripts/neurocascade-tick.sh` runs every 10 minutes under `flock`, so only one
tick is ever live. A task can take over two hours; cron firing far more often than
that is deliberate, so the next task starts promptly once one finishes rather
than waiting for the next slot.

Each tick:

1. Picks the first task whose status is `pending` and whose dependencies are all
   `done`. No runnable task means the tick exits without side effects.
2. Checks the guards. It skips the tick if available memory is under 1500MB, if
   the Claude session quota is under 25%, or if the weekly quota is under 15%.
   The quota floor is what keeps the build from starving interactive work.
3. Refuses to run unless the repo is on `main` with a clean tree, and records
   the current commit as the baseline.
4. Hands the task to a headless `claude -p` session running in this repo, with
   the project rules, the task brief, and its acceptance criteria. Output
   streams to `logs/task-<id>.log`.
5. Commits anything the agent left uncommitted, then runs the full test suite.
6. On a red suite, feeds the failure back to a fresh agent session to fix, up to
   3 attempts total.
7. On green with new commits, pushes to `origin/main` and marks the task `done`.
8. On still-red, force-creates a `failed/<task-id>-attemptN` branch at the
   current commit and resets `main` back to the baseline. Main stays green and
   the work is recoverable from that branch.

A task whose agent reports it cannot be completed honestly ends as `blocked`
with the reason recorded, rather than as a silent success.

## The quota gate, and how it reads a stale number

The gate at step 2 reads the server-truth `rate_limits` block the CLI reports,
which `/root/.claude/statusline-quota.sh` caches to
`/var/lib/claude-quota/rate-limits.json`. `quota-axi` is the fallback only; its
Claude source on this host returns a cached reading whose windows have all
expired.

Two failures shaped the current design, both found while resuming the build:

- **A silent freeze (found 2026-09-07).** The CLI moved `rate_limits` from
  `.current_usage` to the top level of the statusline payload; the statusline
  read only the old path, so the cache stopped updating on 2026-08-29 while
  every tick kept reading a plausible-looking file. A gate that cannot fail is
  not a gate. The statusline now accepts either location.
- **Age is the wrong validity test.** The statusline only writes while an
  interactive session is rendering it, so an unattended overnight build sees
  readings many hours old. Validity is decided per window by `resets_at`
  instead: usage only rises inside a window, so a reading from a window that is
  still open is a lower bound on usage now, which is safe to block on. Once the
  window has rolled over the same number describes spending that has already
  been refunded, and that window imposes no constraint. File age remains the
  fallback for payloads written before `resets_at` was recorded.

Covered by `tests/test_autonomous_runner_quota.py`.

## When the session never starts

Distinct from a killed tick: the agent process launches and exits having done
nothing, leaving no output, no edits and no commits. Dead auth, a refused flag
or a missing CLI all look like this. The task returns to `pending` and the
attempt is rolled back, because no attempt at the work happened and an
environment fault must not burn one of the task's three real attempts.

Retrying cannot fix the cause, so consecutive faults on one task are counted and
`MAX_INFRA_FAULTS` (3) of them writes `logs/.paused` and stops the build for a
human. Without that ceiling, an expired token on 2026-09-06 sent P21.2 to
`attempts: 10`, each cycle leaving a `chore(backlog)` start/rollback pair: 20
churn commits in two hours, one of which swept an unrelated uncommitted edit
into an "autonomous" commit. Covered by
`tests/test_autonomous_runner_faults.py`.

## When a tick is killed

A tick that dies mid-task (killed process tree, reboot, OOM) leaves its task
marked `running` with a dirty tree. Left alone that wedges the build for good:
the runner only picks `pending` tasks, and every later tick would abort on the
dirty tree anyway.

The next tick recovers automatically. Holding the flock proves any `running`
task belongs to a dead tick, so its partial work is committed to an
`interrupted/<task-id>-attemptN` branch, main is reset to the baseline recorded
on the task when it started, and the task returns to `pending` to be redone from
clean. Nothing is destroyed, and the build resumes on its own.

Launch a manual tick with `setsid` so it does not die with the shell that
started it:

```bash
setsid scripts/neurocascade-tick.sh </dev/null >/dev/null 2>&1 &
```

## Notifications

`scripts/digest.py` sends one batched Telegram summary daily at 08:00, covering
everything since the last digest: tasks completed, tasks that failed the gate,
skipped ticks and their reasons, and what is next. It is silent when nothing
happened. There is no per-task alert.

## Operating it

```bash
# progress and what runs next
python scripts/autonomous_runner.py --status

# what the next tick would pick, changing nothing
python scripts/autonomous_runner.py --dry-run

# see the digest without sending it
python scripts/digest.py --print

# live log of the current task
tail -f logs/task-P1.1.log

# pause the build (leaves everything else intact)
touch /tmp/neurocascade-autobuild.lock && \
  flock -n /tmp/neurocascade-autobuild.lock sleep infinity &

# stop it properly
crontab -l | grep -v neurocascade-tick | crontab -
```

To reorder, add, remove or rewrite work, edit `tasks/backlog.json` directly. The
runner reads it fresh every tick. Setting a task's status back to `pending`
requeues it; a `failed` task is not retried automatically, because the same
failure would usually repeat.

## Why the test gate is strict

The agent sessions are told never to delete, skip or weaken a test to reach
green, and never to widen a numeric tolerance to hide a real error. Those two
shortcuts are the ways an autonomous build quietly turns into a build that
passes while being wrong. The quarantine branch exists so that enforcing this
never costs any work: failing work is parked, not discarded.

The same reasoning drives the parameter rules in `docs/roadmap.md`. Every task
brief repeats them, because an agent session that cannot find a literature
value is under real pressure to invent one, and a fabricated parameter is far
more expensive to discover later than a task that honestly reports it found
nothing.

## One driver, from Phase 21 onward

Added 2026-09-05. Three autonomous drivers have written to this repository, all
pushing under the same GitHub account: the Claude cron described above, Codex,
and Codebuff. They applied different test discipline and left different amounts
of evidence, and every audit since mid-August has needed git archaeology to
answer "which of them wrote this".

**From Phase 21 onward the Claude cron is the only authorised driver.** The
reason is not preference. It is the only one of the three that runs the test
gate, quarantines failing work on a branch instead of discarding or keeping it,
records `started` and `finished` events, and, since P20.3, reports
`BLOCKED_ON_DATA` and updates the validation ledger. The other two bypass all of
it.

The evidence for that choice is in this repository's own history. The stretch
the validation ledger records as scaffolding, P13.2 through P18.3, is 23 tasks
produced in five commits inside a single run on 2026-08-29, between 10:57 and
16:31 UTC. No test gate ran over any of it. That is what an ungated driver
pointed at a data-starved layer produces, and it is the largest single reason
the backlog could read 77/77 while the model could predict one measured human
quantity.

If a second driver is ever used again, it runs through
`scripts/autonomous_runner.py` or it does not run.

### Who wrote what

| Commits | Dates | Driver | How it is identified |
|---|---|---|---|
| Initial commit and Tracks A to E | 2026-07-26 | Claude, interactive | `Claude Sonnet 5` trailer; merges of `worktree-agent-*` branches |
| P1.1 to P8.1 | 2026-08-11 to 08-14 | Claude cron | `started` and `finished` in `logs/events.jsonl`, a `logs/task-<id>.log` per task, and the runner's own `chore(backlog)` commits bracketing each one |
| P8.2 to P10.3 | 2026-08-16 | Codex | Not provable from this repository. No trailer, no log coverage, and the backlog status flipped inline in the same commit, which the cron never does. Named only from `~/.codex/sessions/2026/08/16/` outside the repo |
| Human connectome and expression scaffolding | 2026-08-17 | Codebuff | `Co-Authored-By: Codebuff` trailer |
| v2 roadmap, Phase 11 to 18 backlog, P11.1 | 2026-08-21 to 08-22 | Claude, interactive | `Claude Opus 5` trailer with no tick events |
| P11.2 to P12.4 | 2026-08-23 to 08-25 | Claude cron | Full event chains and per-task logs |
| P13.1 write-up | 2026-08-25 | Claude, interactive | written after the cron logged three `agent_launch_failed` for the same task |
| P13.1 F-TRACT ingest, P13.2 to P19.1 | 2026-08-25 to 08-30 | Codebuff | Codebuff trailer on most; the rest by identity and timezone |
| P19.2 to P20.5 | 2026-09-02 to 09-05 | Claude, interactive | `Claude Opus 5` trailer; the cron was paused on 2026-08-25 and has run no task since |
| `chore: automated backup snapshot <ISO>` | throughout | not an agent | `/usr/local/bin/git-backup-sweep.sh`, daily |

Four commits cannot be attributed from the repository alone: the eight of
2026-08-16, `e719812`, and `f399c63` and `672537f`, which have empty bodies. They
are recorded here as inferred rather than established.

### Grepping for a driver

All three pushed under the same GitHub account, so the git identity is nearly
useless on its own:

```
AmunRaPtah <work@supercriticalbooks.com>                        # cron, interactive Claude, Codex, and the backup sweep alike
Eniola Olutogun <94907067+AmunRaPtah@users.noreply.github.com>  # Codebuff exclusively, and always at +0000
```

What does distinguish them is the commit-body trailer, plus the subject patterns
the runner emits by itself:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>       # Claude, cron and interactive
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>     # Claude, 2026-07-26
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>      # Claude cron, 2026-08-14
Co-Authored-By: Codebuff <noreply@codebuff.com>             # Codebuff
                                                            # Codex left no marker anywhere
^chore\(backlog\): (start .*|.* (done|blocked|failed the test gate)) \(autonomous\)$
```

Going forward every commit carries a trailer naming its driver, so this table
does not need to grow. The convention exists because Codex left none and its
eight commits are consequently unattributable from the repository forever.

### The tick log is not a complete record

It covers 2026-08-11 to 08-14 and 2026-08-23 to 08-25, and nothing else. The
first gap is a tick that died mid-P8.2 on 2026-08-14 and was not restarted for
nine days; everything in that hole is invisible to the log. The permanent end is
`agent_launch_failed task=P13.1 attempts=3` on 2026-08-25T08:39Z, after which the
crontab tick line was commented out. Later entries in `logs/events.jsonl` are
full-suite heartbeats from a separate cron, not build activity.

So the log proves cron authorship inside those two windows and proves nothing
outside them. That is exactly the blindness `repo_health()` in `scripts/digest.py`
now compensates for, by reading repository state rather than the event history.
