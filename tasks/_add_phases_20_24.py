#!/usr/bin/env python3
"""One-shot: append Phases 20-24 to tasks/backlog.json.

Written 2026-09-05 alongside the process audit. Kept in the tree so the change
to the backlog is reviewable as code rather than as a large opaque JSON diff.
"""
import json
from pathlib import Path

BACKLOG = Path(__file__).resolve().parent / "backlog.json"

P20 = "Phase 20: process repair, so the build can hold a validated result"
P21 = "Phase 21: multi-drug D2 occupancy, FDA-sourced"
P22 = "Phase 22: external validation against ground truth never used in fitting"
P23 = "Phase 23: does the binding fingerprint explain what occupancy cannot"
P24 = "Phase 24: the deliverable"

FDA = (
    "Primary data route for this phase is the US FDA, because it is free, "
    "unpaywalled, public domain, and holds exactly the analyses the paywalled "
    "literature withholds. Three surfaces: (1) Drugs@FDA approval packages at "
    "accessdata.fda.gov/scripts/cder/daf/, whose 'Clinical Pharmacology and "
    "Biopharmaceutics Review(s)' PDF carries the full population-PK analysis "
    "including structural model, parameter estimates and inter-individual "
    "variability, often with per-subject concentration tables; (2) the openFDA "
    "label API at api.fda.gov/drug/label.json for approved dose ranges; "
    "(3) DailyMed for current labelling. Record the application number (NDA/ANDA) "
    "and the review date with every parameter, the same way a PMID is recorded."
)

TASKS = [
    # ---------------------------------------------------------------- Phase 20
    dict(
        id="P20.1", phase=P20,
        title="Continuous integration on every push",
        brief=(
            "The repository ran from 2026-07-26 to 2026-09-05 with no CI. The "
            "only gate was a local cron a dirty tree silently disabled, so the "
            "P7.3-P10.3 and P14-P18 stretches were never gated at all. Add a "
            "GitHub Actions workflow: fast suite on every push, full suite "
            "nightly, results uploaded even on failure because the local log "
            "twice recorded a failure with an empty summary."
        ),
        acceptance=[
            ".github/workflows/ci.yml runs pytest -m 'not slow' on push",
            "Full suite runs on schedule and workflow_dispatch",
            "Results uploaded with if: always()",
        ],
        validation=dict(kind="process", evidence="CI green on main"),
        deps=[], status="done",
    ),
    dict(
        id="P20.2", phase=P20,
        title="The validation ledger replaces task count as the progress metric",
        brief=(
            "The backlog reached 70/70 and then 77/77 while the model could "
            "predict one measured human quantity. Create docs/validation-ledger.md "
            "recording every claim compared against a number nobody in this "
            "repository chose, its held-out error and its source, plus the "
            "untested layers and the searched-and-empty results. Populate it "
            "with the true current state, which is four rows, two of which are "
            "failures."
        ),
        acceptance=[
            "docs/validation-ledger.md exists and states the rules for a row",
            "Current state recorded honestly, including V2 and V3 as failures",
            "Every task prompt instructs the agent to update it",
        ],
        validation=dict(kind="process", evidence="ledger seeded with V1-V4"),
        deps=[], status="done",
    ),
    dict(
        id="P20.3", phase=P20,
        title="Detect the wedge modes that produce no events",
        brief=(
            "Every stall this build has suffered was invisible to the digest, "
            "because the digest keys off the event log and a wedged build emits "
            "no events. Three occurrences with the same signature: a killed tick "
            "leaving a task running with a dirty tree (2026-08-11), the same with "
            "the cron paused (2026-09-03), and three consecutive full-suite skips "
            "nobody was told about (2026-09-05). Add a repo_health() that reads "
            "repository state rather than history and forces a send."
        ),
        acceptance=[
            "digest.py reports dirty tree, stuck running task, trailing full-suite skips, and an exhausted backlog",
            "These findings send a digest even when the event log is empty",
        ],
        validation=dict(kind="process", evidence="detector reproduced all three historical faults"),
        deps=[], status="done",
    ),
    dict(
        id="P20.4", phase=P20,
        title="Freeze the layers that are blocked on data, not effort",
        brief=(
            "The autonomous drivers generate scaffolding wherever the backlog "
            "points, so pointing it at a data-starved layer produces contracts. "
            "Formally freeze, in docs/roadmap-human.md and in a test, the layers "
            "out of scope for v1: the circuit layer and every Wilson-Cowan "
            "parameter, whole-brain simulation, the mu-opioid and GABA-A chains, "
            "and all behavioural endpoints beyond threshold classification. "
            "Freezing is not deleting: state what would unfreeze each one."
        ),
        acceptance=[
            "Frozen layers named in the roadmap with the unfreezing condition for each",
            "A test asserts no v1 prediction path imports a frozen layer",
            "No task in Phases 21-24 touches a frozen layer",
        ],
        validation=dict(kind="process", evidence=""),
        deps=["P20.2"], status="pending",
    ),
    dict(
        id="P20.5", phase=P20,
        title="One driver, and a record of who did what",
        brief=(
            "Three autonomous drivers have pushed to this repository under the "
            "same GitHub identity: the Claude cron, Codex, and Codebuff. They "
            "applied different test discipline, the tick log stopped being a "
            "complete record in mid-August, and every audit since has needed git "
            "archaeology. Choose one driver for Phases 21-24, document it in "
            "docs/autonomous-build.md, and record the historical attribution so "
            "the August-September commits stay interpretable."
        ),
        acceptance=[
            "One driver named as authoritative for Phases 21-24",
            "docs/autonomous-build.md records which driver produced which phase historically",
            "Commit trailer convention stated so future attribution is readable from git log alone",
        ],
        validation=dict(kind="process", evidence=""),
        deps=["P20.4"], status="pending",
    ),

    # ---------------------------------------------------------------- Phase 21
    dict(
        id="P21.1", phase=P21,
        title="Turn the haloperidol EC50 constant into a drug registry",
        brief=(
            "src/neurocascade/human/d2_ec50.py hardcodes "
            "HALOPERIDOL_D2_PLASMA_EC50_nM = 1.8849 as a module constant, and "
            "dose_to_occupancy is built around it. Refactor to a drug-keyed "
            "registry carrying per-drug provenance, molecular weight, protein "
            "binding fraction, the basis (total or free) and the fitted interval. "
            "This is a pure refactor: add no drug in this task."
        ),
        acceptance=[
            "D2_PLASMA_EC50 registry keyed by drug, with per-drug provenance and basis",
            "Haloperidol predictions bit-identical before and after: a test asserts this",
            "The old constant still resolves, or every caller is updated",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P20.5"], status="pending",
    ),
    dict(
        id="P21.2", phase=P21,
        title="Can the multi-drug occupancy table actually be obtained",
        brief=(
            "Rule 12 applies: this phase rests on one assumption, that per-drug "
            "plasma EC50 values for D2 occupancy can be sourced. Test that before "
            "building on it. Uchida 2011 (PMID 21694629) is a pooled systematic "
            "review publishing exactly this table and is currently cited in this "
            "repository only in prose. Attempt it first, then the primary PET "
            "papers behind it. Report what is reachable per drug before any "
            "ingestion code is written. If fewer than four drugs are reachable, "
            "report BLOCKED_ON_DATA and stop rather than building a registry to "
            "hold values you do not have."
        ),
        acceptance=[
            "Per-drug reachability recorded: source, value or unreachable, and where you looked",
            "No ingestion code written in this task",
            "Fewer than four drugs reachable means BLOCKED_ON_DATA, not a partial build",
        ],
        validation=dict(kind="searched-and-empty", evidence=""),
        deps=["P21.1"], status="pending",
    ),
    dict(
        id="P21.3", phase=P21,
        title="Human population PK per antipsychotic from FDA review documents",
        brief=(
            "Source human population PK for the drugs P21.2 found reachable. "
            "Candidates, all with published D2 PET data: risperidone, "
            "paliperidone, olanzapine, quetiapine, amisulpride, ziprasidone, "
            "clozapine, aripiprazole. " + FDA + " Record apparent versus absolute "
            "parameters exactly as P19.2 did: if bioavailability was never "
            "identified, the values are apparent (CL/F, V/F) and passing a "
            "separate F must be forbidden. Do not assume a compartment count from "
            "this brief; read what each review actually fitted."
        ),
        acceptance=[
            "Per-drug PK parameter sets with NDA number, review date and structural model as printed",
            "apparent_parameters flag set correctly per drug",
            "Drugs whose PK could not be sourced recorded as searched-and-empty, not defaulted",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P21.2"], status="pending",
    ),
    dict(
        id="P21.4", phase=P21,
        title="Active metabolites that also occupy D2",
        brief=(
            "Risperidone is cleared to 9-hydroxy-risperidone, which is "
            "paliperidone and is itself a potent D2 antagonist marketed as its "
            "own drug. Aripiprazole is cleared to dehydro-aripiprazole, also "
            "D2-active. A chain that maps only parent concentration to occupancy "
            "will understate both, silently. Either model the summed occupancy "
            "contribution of parent and active metabolite with sourced values for "
            "each, or refuse those drugs explicitly. Silently ignoring the "
            "metabolite is not an option."
        ),
        acceptance=[
            "Active-metabolite handling implemented with sourced parameters, or the drug refused with a recorded reason",
            "A test asserts that a drug with a known active metabolite cannot be predicted through the parent-only path",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P21.3"], status="pending",
    ),
    dict(
        id="P21.5", phase=P21,
        title="Partial agonism breaks the occupancy-limited transducer",
        brief=(
            "D2Effect currently has emax=1.0 and baseline=0.0, which makes "
            "effect identically equal to occupancy. That is a re-expression, not "
            "a prediction, and it is simply wrong for aripiprazole, a D2 partial "
            "agonist that sustains occupancy above 80 percent without the "
            "extrapyramidal consequence a full antagonist would carry at that "
            "occupancy. Either represent intrinsic activity with a sourced value "
            "or exclude aripiprazole from v1 with the reason recorded. Aripiprazole "
            "is the strongest single counterexample to reading occupancy as effect, "
            "so whichever way this goes it belongs in the ledger."
        ),
        acceptance=[
            "Intrinsic activity represented with a sourced value, or aripiprazole excluded with a recorded reason",
            "The occupancy-equals-effect identity no longer holds silently for partial agonists",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P21.4"], status="pending",
    ),
    dict(
        id="P21.6", phase=P21,
        title="Generalize the chain to take a drug key",
        brief=(
            "Wire dose_to_occupancy, steady_state_occupancy, "
            "dose_reaching_occupancy and chain_testable_observations to accept a "
            "drug key and resolve PK and EC50 from the registries. Preserve every "
            "refusal the haloperidol chain already makes: the weight-scaling "
            "refusal off the 70 kg reference, the Kp,uu double-correction guard, "
            "and the illustrative-constant gate."
        ),
        acceptance=[
            "The chain runs for every drug that has both PK and EC50, and refuses clearly for those that do not",
            "Every existing haloperidol test still passes unchanged",
            "Refusals are per-drug and state which layer is missing",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P21.5"], status="pending",
    ),

    # ---------------------------------------------------------------- Phase 22
    dict(
        id="P22.1", phase=P22,
        title="Pre-register the validation metrics before computing any of them",
        brief=(
            "P19.6 recorded that its Kapur 1996 comparison was not blind: the "
            "prediction had already been published in the P19.5 document before "
            "the comparison ran, and the steady-state figure was computed after "
            "the single-dose miss was seen. That caveat is the reason this task "
            "exists and comes first. Write down, and commit, the exact metrics "
            "and pass criteria for P22.2, P22.3 and P22.4 before running any of "
            "them. Commit this document in its own commit, so git history proves "
            "the ordering."
        ),
        acceptance=[
            "docs/p22-preregistration.md committed before any P22 result exists",
            "Metric, pass criterion and handling of ties or missing drugs fixed in advance for each test",
            "It states explicitly that a negative result will be reported and is publishable",
        ],
        validation=dict(kind="process", evidence=""),
        deps=["P21.6"], status="pending",
    ),
    dict(
        id="P22.2", phase=P22,
        title="Label-dose test against FDA approved dose ranges",
        brief=(
            "For each supported drug, predict the daily dose reaching 65 percent "
            "and 78 percent steady-state striatal D2 occupancy, and compare the "
            "predicted window against the approved dose range for schizophrenia "
            "from the FDA label. " + FDA + " The label range is genuine held-out "
            "ground truth: it was set by clinical trials and has never entered any "
            "fit in this repository. For haloperidol the chain currently predicts "
            "2.89 to 5.56 mg/day, which is the reference point. Report per-drug "
            "overlap and the log-scale error on the window midpoint, using the "
            "criteria fixed in P22.1."
        ),
        acceptance=[
            "Per-drug predicted window against FDA label range, with the application number cited",
            "Metrics exactly as pre-registered, no post-hoc alternatives added",
            "Drugs where the prediction misses are reported with the miss attributed to a layer",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P22.1"], status="pending",
    ),
    dict(
        id="P22.3", phase=P22,
        title="Leave-one-drug-out against published PET occupancy",
        brief=(
            "Fit nothing on drug X, then predict X's occupancy-concentration "
            "relationship from its own PK and its own sourced EC50 and compare "
            "against published PET observations for X. Carry forward the tracer "
            "caveat from P19.6: the spread across raclopride, IBZM and FLB 457 is "
            "2 to 2.5 fold, larger than every held-out error reported so far, so "
            "record the tracer per observation and never pool across tracers "
            "without saying so."
        ),
        acceptance=[
            "Per-drug held-out error against measured PET, tracer and region recorded per observation",
            "Observations with region or tracer unrecorded are counted and reported, not silently pooled",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P22.1"], status="pending",
    ),
    dict(
        id="P22.4", phase=P22,
        title="The clozapine anomaly, as a falsification test",
        brief=(
            "Clozapine is the most effective antipsychotic available and is "
            "clinically effective at D2 occupancy well below the 65 percent "
            "threshold this repository sources from Kapur 2000. If the model "
            "predicts clozapine needs a dose far above its label range to reach "
            "65 percent, that is not a bug: it is the model correctly reproducing "
            "a real anomaly that falsifies occupancy-as-sole-mediator. Run it, "
            "report it, and state plainly which reading the result supports. Do "
            "not tune anything to make clozapine fit."
        ),
        acceptance=[
            "Clozapine predicted window reported against its label range without adjustment",
            "The result interpreted explicitly as evidence for or against occupancy as the sole mediator of response",
            "No parameter changed in response to the outcome",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P22.2"], status="pending",
    ),
    dict(
        id="P22.5", phase=P22,
        title="Failure analysis to the P19.7 standard",
        brief=(
            "Write the scope and failure analysis for the multi-drug chain to the "
            "standard set by P8.3 and P19.7: state what was tested, attribute "
            "every failure to a layer rather than to the model as a whole, and say "
            "what data would repair each one. Regenerate every number from the "
            "code rather than copying it from a brief or an earlier document."
        ),
        acceptance=[
            "Per-layer failure matrix with a concrete data requirement for each row",
            "Every number regenerated from code in this repository",
            "The validation ledger updated to match, including the failures",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P22.2", "P22.3", "P22.4"], status="pending",
    ),

    # ---------------------------------------------------------------- Phase 23
    dict(
        id="P23.1", phase=P23,
        title="Human off-target binding fingerprints",
        brief=(
            "This phase tests the project's founding thesis on the one dataset "
            "large enough to settle it. The repository already holds the "
            "falsification: Siafis 2023 (PMC10618092, N=37,193, open access) "
            "reports haloperidol at 11 mg/day with an extrapyramidal odds ratio "
            "of 4.7 against risperidone at 16 mg/day with 3.3, at approximately "
            "the same 87 percent D2 occupancy. Occupancy alone does not determine "
            "the outcome. Build per-drug human Ki fingerprints at 5-HT2A, M1, H1, "
            "alpha-1 and D3 from the PDSP Ki database, which is free and public, "
            "plus published human Ki values. Human tissue or human recombinant "
            "receptor only, and the species gate applies."
        ),
        acceptance=[
            "Per-drug human Ki at the named off-targets, with assay species and source per value",
            "Non-human values excluded by the existing species gate, not silently accepted",
            "Targets that could not be sourced per drug recorded as searched-and-empty",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P22.5"], status="pending",
    ),
    dict(
        id="P23.2", phase=P23,
        title="Off-target occupancy from the same plasma curve",
        brief=(
            "Extend the chain to predict fractional occupancy at each off-target "
            "from the same plasma concentration-time curve that drives D2. Be "
            "explicit that an in-vitro Ki is not an in-vivo EC50: P19.4 already "
            "separated these for D2 and found the in-vivo value differs, so "
            "off-target occupancies computed from Ki are a weaker quantity and "
            "must be labelled as such wherever they are reported."
        ),
        acceptance=[
            "Off-target occupancy computed per drug from the shared plasma curve",
            "Ki-derived occupancy labelled distinctly from the in-vivo-fitted D2 occupancy everywhere it appears",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P23.1"], status="pending",
    ),
    dict(
        id="P23.3", phase=P23,
        title="Does the fingerprint explain the residual that occupancy cannot",
        brief=(
            "The pre-registered hypothesis: the 5-HT2A to D2 occupancy ratio "
            "explains variance in extrapyramidal risk across drugs that D2 "
            "occupancy alone leaves unexplained. The null is that it does not. "
            "Ingest the Siafis 2023 per-drug dose-response curves, compute "
            "predicted D2 and 5-HT2A occupancy at each drug's ED95 dose, and "
            "regress the reported extrapyramidal odds ratios on D2 occupancy "
            "alone versus D2 plus the ratio. Pre-register hypothesis, model and "
            "criterion in the P22.1 document before running it. With roughly "
            "eight drugs the power is low and the result may be indeterminate: "
            "say so if it is, rather than reading a trend as a finding."
        ),
        acceptance=[
            "Hypothesis, model and criterion committed before the regression is run",
            "Both models fitted and compared on the pre-registered criterion",
            "An indeterminate result reported as indeterminate, with the achievable power stated",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P23.2"], status="pending",
    ),
    dict(
        id="P23.4", phase=P23,
        title="Report the result either way",
        brief=(
            "Write up P23.3 whichever way it came out. If the fingerprint "
            "explains the residual, that is the project's central claim supported "
            "on human outcome data at N=37,193 and it should be stated as such. "
            "If it does not, that is a real negative result about a widely held "
            "hypothesis and it is equally worth publishing. If the analysis was "
            "underpowered, the honest finding is that eight drugs cannot settle "
            "it and the write-up says what number could."
        ),
        acceptance=[
            "Result stated plainly, including if it is negative or indeterminate",
            "The validation ledger row matches the write-up",
            "No claim that the model predicts behaviour beyond what was tested",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P23.3"], status="pending",
    ),

    # ---------------------------------------------------------------- Phase 24
    dict(
        id="P24.1", phase=P24,
        title="The public prediction interface",
        brief=(
            "Expose predict(drug, dose_mg, regimen) returning the occupancy-time "
            "course and where the regimen sits relative to the sourced thresholds. "
            "It must refuse clearly for any drug lacking a layer, naming the "
            "missing layer, and must never silently fall back to another drug's "
            "parameters. Refusal is a feature here and the tests should treat it "
            "as one."
        ),
        acceptance=[
            "predict() implemented with per-drug refusal naming the missing layer",
            "Tests cover refusal paths as thoroughly as success paths",
            "No cross-drug parameter fallback exists anywhere in the path",
        ],
        validation=dict(kind="ledger-row", evidence=""),
        deps=["P23.4"], status="pending",
    ),
    dict(
        id="P24.2", phase=P24,
        title="Model card",
        brief=(
            "Write the model card carrying P19.7's scope limits verbatim where "
            "they still apply: striatal only, the depot-fitted versus "
            "orally-driven regimen mismatch, no chronic pharmacology, no effect "
            "compartment, and the populations the observations came from. State "
            "what the model must not be used for, first, before what it can do."
        ),
        acceptance=[
            "Model card states intended use, out-of-scope uses and every unrepaired mismatch",
            "Limitations appear before capabilities",
            "Every claim traceable to a validation-ledger row",
        ],
        validation=dict(kind="process", evidence=""),
        deps=["P24.1"], status="pending",
    ),
    dict(
        id="P24.3", phase=P24,
        title="Manuscript",
        brief=(
            "Draft the manuscript from the validation ledger rather than from the "
            "roadmap. The contribution is a mechanistic dose-to-occupancy chain "
            "for antipsychotics validated against FDA label dose ranges and "
            "held-out PET, plus whatever P23 concluded. Journal-direct rather than "
            "preprint. Do not claim behavioural prediction, whole-brain simulation "
            "or any frozen layer."
        ),
        acceptance=[
            "Every quantitative claim traceable to a ledger row",
            "Negative and indeterminate results included",
            "No claim about frozen layers",
        ],
        validation=dict(kind="process", evidence=""),
        deps=["P24.2"], status="pending",
    ),
]


def main() -> int:
    data = json.loads(BACKLOG.read_text())
    existing = {t["id"] for t in data["tasks"]}
    added = 0
    for t in TASKS:
        if t["id"] in existing:
            continue
        data["tasks"].append({
            "id": t["id"], "phase": t["phase"], "title": t["title"],
            "brief": t["brief"], "acceptance": t["acceptance"],
            "validation": t["validation"], "deps": t["deps"],
            "status": t["status"], "attempts": 0,
        })
        added += 1
    data["note"] = (
        "Progress is measured by docs/validation-ledger.md, not by the count of "
        "tasks marked done. A task closes on a prediction compared against an "
        "externally measured number, or on an explicit searched-and-empty record. "
        "Contracts, registries and gates do not close a task. Work gated on an "
        "unreachable number takes status blocked-on-data, never done."
    )
    BACKLOG.write_text(json.dumps(data, indent=2) + "\n")
    print(f"added {added} tasks, {len(data['tasks'])} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
