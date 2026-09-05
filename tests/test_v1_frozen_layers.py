"""P20.4: the v1 prediction path must not depend on a frozen layer.

The three autonomous drivers generate scaffolding wherever the backlog points,
so pointing them at a data-starved layer produced contracts rather than
predictions: `docs/validation-ledger.md` records every circuit parameter as
illustrative, every mu-opioid and GABA-A PK parameter and receptor Ki as `None`
behind a gate, and no test anywhere comparing a predicted behaviour to a
measured one. Those layers are frozen for v1, with the unfreezing condition for
each recorded in `docs/roadmap-human.md`.

Freezing is a claim about what the shipped chain depends on, so it is checked
against the source rather than against prose. A frozen module reachable from the
v1 chain would put an illustrative constant one attribute access away from a
reported number.

Dependency here means a declared import, resolved transitively through the
modules' own source. It deliberately does not mean "loaded at runtime": both
`neurocascade/__init__.py` and `neurocascade/human/__init__.py` eagerly
re-export their whole subtree, so importing any one module loads all of them.
That is a packaging property, not a dependency of the chain, and
`test_package_init_is_still_eager` below pins it so the distinction stays
visible rather than being mistaken for isolation.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

# The v1 product surface: oral dose to striatal D2 occupancy, and the validation
# that reports its error. Phase 24 adds `neurocascade.predict` here.
V1_ENTRY_POINTS = [
    "neurocascade.human.dose_to_occupancy",
    "neurocascade.human.occupancy_validation",
]

# Frozen for v1. Each entry is (why, what would unfreeze it) and must stay in
# step with the frozen-layer table in docs/roadmap-human.md.
FROZEN = {
    "neurocascade.circuit": (
        "every Wilson-Cowan parameter is illustrative; the one fitted "
        "connection gain was rat-fitted and retired in the human pivot",
        "human-measured E/I time constants and connection weights",
    ),
    "neurocascade.network": (
        "whole-brain simulation rests on a connectome whose edge directions "
        "are inferred, not measured",
        "a directed human connectome with measured directions",
    ),
    "neurocascade.behavior": (
        "no test compares a predicted behaviour to a measured one",
        "a human dataset linking exposure to a graded behavioural measure",
    ),
    "neurocascade.endpoints": (
        "behavioural endpoints beyond threshold classification are out of "
        "scope for v1",
        "the same, plus an identified Emax and baseline",
    ),
    "neurocascade.opioid": (
        "the mu-opioid chain is scaffolding: PK parameters and receptor Ki "
        "are None behind a gate",
        "sourced human mu-opioid PK and Ki",
    ),
    "neurocascade.human.behavioral_endpoints": (
        "human behavioural endpoints were retired in P17.4",
        "a human dose-response set identifying an Emax and a baseline",
    ),
    "neurocascade.human.mor_effect": (
        "reserve-dominated mu-opioid transducer, no human parameters",
        "sourced human mu-opioid PK and Ki",
    ),
    "neurocascade.human.gabaa_effect": (
        "operational model of agonism, no human parameters; P14.4 found no "
        "adult diazepam population PK exists",
        "an adult, not pediatric, benzodiazepine population PK model",
    ),
    "neurocascade.human.matrix": (
        "connectome matrix, whole-brain only",
        "a directed human connectome with measured directions",
    ),
    "neurocascade.human.ei_time_constants": (
        "circuit-layer parameter, illustrative",
        "human-measured E/I time constants",
    ),
    "neurocascade.human.directed_prior": (
        "connectome edge direction prior, inferred",
        "measured edge directions",
    ),
}


def _source_of(module: str) -> Path | None:
    stem = SRC / Path(module.replace(".", "/"))
    for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _direct_imports(module: str, path: Path) -> set[str]:
    """Package-internal modules `module` imports by name, absolute or relative.

    A dotted target may be either a submodule or an attribute of a package
    (`from .human import d2_ec50` versus `from .d2_ec50 import EC50`), so both
    readings are emitted and the caller drops whichever does not resolve to a
    file.
    """
    package = module.rsplit(".", 1)[0] if path.name != "__init__.py" else module
    found: set[str] = set()

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names
                         if a.name.split(".")[0] == "neurocascade")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                target = f"{base}.{node.module}" if node.module else base
            elif node.module and node.module.split(".")[0] == "neurocascade":
                target = node.module
            else:
                continue
            found.add(target)
            found.update(f"{target}.{a.name}" for a in node.names)

    return found


def _dependency_closure(entry_points: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(entry_points)
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        path = _source_of(module)
        if path is None:
            continue  # an imported attribute, not a module
        seen.add(module)
        stack.extend(_direct_imports(module, path) - seen)
    return seen


def test_v1_entry_points_resolve():
    """Guards the rest of this file: a renamed entry point would otherwise make
    every check below pass on an empty set."""
    for module in V1_ENTRY_POINTS:
        assert _source_of(module) is not None, f"{module} has no source file"
    closure = _dependency_closure(V1_ENTRY_POINTS)
    assert len(closure) > len(V1_ENTRY_POINTS), (
        "the v1 chain resolved to no dependencies at all, which means the "
        "import analysis is broken, not that the chain is self-contained")


def test_v1_prediction_path_depends_on_no_frozen_layer():
    reached = sorted(_dependency_closure(V1_ENTRY_POINTS) & set(FROZEN))
    assert not reached, (
        "the v1 prediction path depends on frozen layer(s): "
        + "; ".join(f"{m} ({FROZEN[m][0]})" for m in reached)
        + ". Either the dependency is a mistake, or the layer has acquired the "
        "data that unfreezes it and both this list and the frozen-layer table "
        "in docs/roadmap-human.md need updating. Do not delete the entry to "
        "make this pass."
    )


def test_package_init_is_still_eager():
    """Records, rather than asserts away, that import isolation does not exist.

    `neurocascade/__init__.py` and `neurocascade/human/__init__.py` re-export
    their whole subtree, so `import neurocascade.human.dose_to_occupancy` loads
    the circuit layer, the connectome and the opioid scaffolding too. Nothing in
    the v1 chain calls into them, which is what the test above checks, but a
    reader should not be able to infer from that test that the frozen code is
    unloaded. If this ever fails because the packages were made lazy, that is an
    improvement: delete this test and say so in the commit.
    """
    script = (
        "import sys, json\n"
        "import neurocascade.human.dose_to_occupancy\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('neurocascade'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO, capture_output=True, text=True, timeout=180,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"v1 entry point will not import:\n{result.stderr}"
    loaded = set(json.loads(result.stdout.strip().splitlines()[-1]))
    assert loaded & set(FROZEN), (
        "frozen modules are no longer loaded by a v1 import, so the packages "
        "appear to have been made lazy. Drop this test.")


@pytest.mark.parametrize("module", sorted(FROZEN))
def test_frozen_layer_is_documented(module):
    """A layer is frozen only if the roadmap says what would unfreeze it.

    Freezing is not deleting. A frozen layer with no recorded unfreezing
    condition is abandonment wearing the word.
    """
    roadmap = (REPO / "docs" / "roadmap-human.md").read_text()
    assert module in roadmap, (
        f"{module} is frozen in code but absent from docs/roadmap-human.md")
    assert FROZEN[module][1], f"{module} is frozen with no unfreezing condition"


def test_no_backlog_task_after_phase_20_touches_a_frozen_layer():
    """Phases 21-24 are the path to a validated product. None of them may
    resume a frozen layer, because that is how 39 tasks went into a mouse
    in-situ-hybridisation layer whose load-bearing assumption had not been
    checked."""
    tasks = json.loads((REPO / "tasks" / "backlog.json").read_text())["tasks"]
    forward = [t for t in tasks
               if t["id"].startswith(("P21.", "P22.", "P23.", "P24."))]
    assert forward, "Phases 21-24 are missing from the backlog"

    # Frozen subject matter, named as it would appear in a brief rather than as
    # a module path, since a brief is prose.
    banned = ["wilson-cowan", "whole-brain", "connectome", "mu-opioid",
              "behavioural endpoint", "behavioral endpoint"]
    offenders = []
    for task in forward:
        text = (task["brief"] + " " + " ".join(task["acceptance"])).lower()
        for phrase in banned:
            # Naming a frozen layer in order to exclude it is the point of
            # several briefs, so only an unqualified mention counts.
            if phrase in text and "frozen" not in text:
                offenders.append(f"{task['id']}: {phrase}")
    assert not offenders, (
        "task(s) in Phases 21-24 reach into a frozen layer: "
        + ", ".join(offenders))
