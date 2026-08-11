# The autonomous build

The roadmap in `docs/roadmap.md` is decomposed into 39 tasks in
`tasks/backlog.json`. A cron tick runs them one after another without manual
intervention, and stops on its own when the backlog is complete.

## How a tick works

`scripts/neurocascade-tick.sh` runs every 45 minutes under `flock`, so only one
tick is ever live. A task can take over two hours; cron firing more often than
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
