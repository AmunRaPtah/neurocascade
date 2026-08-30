# Codebuff briefing: Phase 19, the haloperidol dose-to-occupancy chain

Hand this file to Codebuff as its instructions. It is self-contained: everything
needed to work the phase, run the gate, and know when to stop is here.

---

## 1. What you are building and why it matters

Phases 14 through 18 produced **contracts, registries and validators, but almost
no pharmacology**. An audit on 2026-08-29 found that of fifteen modules under
`src/neurocascade/human/`, exactly one carries a sourced human number: diazepam
`kp_uu = 0.79` in `kp_uu.py:27`. Every human PK parameter and every receptor
Ki/efficacy is `None`, sitting behind a gate that always fails. No test anywhere
compares a prediction to a measured human observation, because no measured human
observation exists in the repository.

The backlog says 70/70 done. That is true at the level of tasks and false at the
level of science. Phase 19 closes exactly one gap, properly:

> **dose -> plasma concentration -> striatal D2 receptor occupancy**, for
> haloperidol, in humans, checked against measured PET occupancy points that
> were never used to fit it.

Haloperidol was chosen because human D2 occupancy versus plasma concentration is
the best-measured receptor-occupancy relationship in pharmacology. That makes
this the first prediction in the repository that can be **wrong in public**,
which is the entire point. A chain that cannot fail a test has not been
validated, it has only been written.

Do not broaden scope. Fentanyl, diazepam, behavioural endpoints and the mouse
tree are all out of scope for this phase.

---

## 2. Repository and environment

| | |
|---|---|
| Repo | `/root/projects/neurocascade` |
| Branch | `main` (work directly on it; the phase is sequential) |
| Python | `/root/projects/pardalos/.venv/bin/python` (**use this, there is no repo-local venv**) |
| Task list | `tasks/backlog.json` |
| Roadmap | `docs/roadmap-human.md` |
| Design notes | `docs/architecture.md` |

Network access is available. Use it to find real literature values. Prefer
primary sources over review summaries where you can reach the primary.

There is **no CI on this repository**. Nothing will catch a regression except
the gate you run yourself. Treat that as a reason for more care, not less.

---

## 3. The rules. These are not style preferences

1. **Never invent a number.** A parameter is either literature-sourced with a
   resolvable citation, or explicitly labelled illustrative and gated so it
   cannot be consumed by accident. There is no third option.
2. **Never substitute another species' or another drug's measured value** for
   the one being modelled. A rat number under a human label is the single worst
   failure available in this repository. If you use a cross-species transfer,
   it must go through `kp_uu.transfer_kp_uu` with an explicit factor and come
   back marked `inferred-species-transfer`.
3. **"Searched and found nothing" is a real, publishable result.** Record what
   you searched and why it came up empty. That outcome is worth more than a
   plausible fabrication and it is explicitly what several existing records do.
4. **Never weaken a test to reach green.** Do not skip, do not `xfail`, do not
   mark a failing test `slow` to move it out of the gate. `slow` is a speed
   classification only; a test that is slow and red is still red and the daily
   sweep will say so.
5. **Never widen a numeric tolerance to make a validation pass.** Report the
   real error instead. This applies with full force to P19.6.
6. **Write tests for what you add**, and record the reasoning behind design
   decisions in `docs/architecture.md`, not just the outcome.
7. When you calibrate a parameter, **remove its illustrative label in the same
   commit** that adds the calibration, and cite the source.

---

## 4. The tasks

The seven tasks are already in `tasks/backlog.json` under
`"phase": "Phase 19: haloperidol dose-to-occupancy chain"`, each with a full
brief and its own acceptance criteria. **Read them from that file** rather than
from this summary; the briefs contain specific traps to avoid and specific
literature leads to verify.

Work them in dependency order. `deps` is authoritative:

```
P19.1  Gate the illustrative constants                     deps: none
P19.2  Human haloperidol PK parameter set                   deps: none
P19.3  Human D2 PET occupancy observation set               deps: none
P19.4  In-vivo D2 EC50 and the in-vitro Ki correction       deps: P19.3
P19.5  Dose-to-occupancy chain for haloperidol              deps: P19.2, P19.4
P19.6  Held-out validation against measured occupancy       deps: P19.5
P19.7  Scope statement and failure analysis                 deps: P19.6
```

To see the current state at any time:

```bash
cd /root/projects/neurocascade
/root/projects/pardalos/.venv/bin/python scripts/autonomous_runner.py --status
```

Three things worth knowing before you start, drawn from the audit:

- **P19.2's trap:** human haloperidol disposition is usually reported as
  *two*-compartment, but `HumanPKRecord` has `V3` and `Q13` slots. Do not invent
  a third compartment to fill them. Record the compartment count the source
  supports and leave the rest `None`.
- **P19.3's trap:** raclopride, IBZM and epidepride give systematically
  different occupancy estimates. Record the tracer per observation and never
  pool across tracers silently.
- **P19.5's design:** haloperidol's `Kp,uu` record is deliberately marked
  `not-required`, with the note that occupancy data should identify effective
  brain exposure. Identify exposure from the occupancy relation. Do not
  introduce an unmeasured brain-partition step to fill the gap.
- **P19.6's known hazard**, already recorded under Phase 18 in
  `docs/roadmap-human.md`: central occupancy decays more slowly than plasma
  (Kurose 2020, PMID 32726002). A single-dose occupancy curve driven purely by
  plasma kinetics will therefore mispredict the multi-day offset unless an
  effect compartment or an explicit slow k_off is introduced. If you hit this,
  **report the mismatch**; do not tune around it. The same section records that
  occupancy alone does not determine EPS, so do not claim a side-effect
  prediction from an occupancy number.

---

## 5. The test gate

Run from the repo root. This is the gate you are judged by:

```bash
cd /root/projects/neurocascade
/root/projects/pardalos/.venv/bin/python -m pytest -q --durations=10 -m "not slow"
```

Iterate against that. It deselects the `slow` MCMC tests so the gate stays
workable. Note that the suite has grown well past the point where that docstring
was written: **the gate now takes more than ten minutes**, and
`autonomous_runner.py` allows it 40 minutes (`NC_TEST_TIMEOUT_S`, default
`40 * 60`). Give it that budget rather than assuming it has hung. While
iterating on one module, run that module's test file alone and save the full
gate for before each commit.

Before the **final** commit of the phase, run the whole suite including `slow`:

```bash
/root/projects/pardalos/.venv/bin/python -m pytest -q --durations=10
```

To run just what you touched while iterating:

```bash
/root/projects/pardalos/.venv/bin/python -m pytest -q tests/test_human_pk_parameters.py
```

**Verified baseline, 2026-08-29 at commit `c13f2fd`:** the gate is green at
**2030 passed, 4 skipped, 15 deselected, in 559s**. The 4 skips are data
availability guards (`full_data_only`, `real_data`, "no cache"), not parked
failures. That is your starting point. If you find the suite red before you have
changed anything, **stop and report it** rather than working on top of it, and
if the passed count drops below 2030 without you having deliberately removed a
test, you have broken something.

---

## 6. The loop, per task

1. Read the task's `brief` and `acceptance` from `tasks/backlog.json`.
2. Set its `status` to `"running"` and record `started_at`.
3. Do the work. Source real values. Write tests for what you add.
4. Run the gate. Iterate until green.
5. Update `docs/architecture.md` with the reasoning, not just the result.
6. Set `status` to `"done"`, and write a `notes` field that says **what is still
   missing**, not just what landed. The existing Phase 14-18 notes are the model
   for this: "values remain illustrative pending fitting" was the honest and
   correct thing to write.
7. Commit with a message naming the task id, e.g.
   `P19.2: transcribe human haloperidol PK parameters (Kudo 1999)`.
8. Push only after the gate is green.

Commit per task, not once at the end of the phase.

---

## 7. When a task cannot be completed honestly

This will probably happen at least once in this phase, most likely in P19.2 or
P19.3, and handling it correctly is worth more than forcing a result.

If the literature does not support the value the task asks for:

1. Set the task `status` to `"blocked"`.
2. Record in `notes` exactly what you searched, where, and what you found or
   failed to find. Name the databases and the queries.
3. Commit the searched-and-empty record as a real deliverable, with a citation
   trail if there is one.
4. Move on to the next task whose deps are satisfied. Do not fabricate a value
   to unblock the chain, and do not mark the task done.

`docs/roadmap-human.md` already ends with a list of open questions recorded
rather than resolved. Add to it. That section is the template for how this
repository handles a gap.

---

## 8. What finishing looks like

The phase is complete when a single documented call maps a haloperidol dose,
route, body weight and time to a predicted striatal D2 occupancy curve, every
parameter it consumes is sourced and cited, and P19.6 reports a real prediction
error against PET points that were never used in the fit.

If that error turns out to be large, **the phase still succeeded**. Report the
number. A validated-and-wrong chain is a scientific result; a chain that was
quietly tuned until it agreed with its own training points is not.

Report at the end of the phase: which tasks completed, which are blocked and
why, the held-out error you measured, and every layer still running on an
illustrative constant.
