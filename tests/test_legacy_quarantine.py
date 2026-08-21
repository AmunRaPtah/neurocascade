"""P11.1: the rodent quarantine boundary, enforced and enumerated.

`docs/roadmap-human.md` moved the rodent-sourced material to
`neurocascade.legacy`. Quarantine is only worth anything if the boundary is
enforced, so these tests pin three things:

1. What is inside `legacy/`, exactly.
2. That the package root no longer re-exports it, so `neurocascade.drug_pk`
   fails rather than silently working.
3. Which non-legacy modules still reach into `legacy/`, as a pinned allowlist.

The third is the load-bearing one. The allowlist is deliberately not empty:
P11.1 moves the code, and P11.2 adds the species gate that severs the
remaining reaches. Pinning the list now means any *new* dependency on rodent
material has to be an explicit edit to this test rather than something that
drifts in unnoticed.

The human layer is held to a stricter rule: it may not reach into `legacy/` at
all. That is already true, and P11.1 is what made it true. Before the hoist,
`human/neurotransmitter.py` took the transmitter-sign constants and
`human/build.py` took `DEFAULT_NODE` from the mouse connectome package. Both
are species-neutral, so they moved to `transmitter_sign.py` and `circuit.py`
respectively, and the mouse package now re-exports them. Neither species
depends on the other.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import neurocascade
from neurocascade import legacy

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neurocascade"

#: Exactly what was quarantined in P11.1.
QUARANTINED = {
    "connectome",
    "connectome_scenarios",
    "drug_pk",
    "endogenous_tone",
    "expression",
    "pk_validation",
    "projection_density",
    "projection_scenarios",
    "projection_systems",
}

#: The one intentional, permanent reach: the package root exposes `legacy` so
#: the quarantined material stays importable, which is the whole point of
#: quarantining rather than deleting it.
EXPOSURE = {"__init__.py"}

#: Non-legacy modules permitted to import from `legacy`, and why.
#:
#: Every entry here is a rodent dependency that survived P11.1 and is scheduled
#: for removal. The task that removes each one is named, so this table doubles
#: as the work list.
PERMITTED_REACHES = {
    # Consumes rodent PK parameters for dose-driven prediction. P11.2 puts the
    # species gate in front of this; P14.2 replaces the parameters with human
    # ones and the reach goes away.
    "fingerprint.py": "P11.2 gate, then P14.2 replaces the parameters",
    # The mu-opioid coupling map still resolves through the Allen ISH scenario
    # maps. P12.1 replaces it with the Hansen PET atlas; P12.4 retires the path.
    "scenarios.py": "P12.1 replaces the density source, P12.4 retires the path",
    # The rodent dose-to-behaviour bridge. P11.3 retires all of it.
    "calibration/behavioral_evidence.py": "P11.3 retires the rodent bridge",
    "calibration/heldout_validation.py": "P11.3 retires the rodent bridge",
    # Whole-brain forward models built on the mouse connectome and ISH atlas.
    # P13 rebuilds on F-TRACT; P12 rebuilds the density side.
    "calibration/endpoint_forward.py": "P12 and P13 rebuild both inputs",
    "calibration/heldout_forward.py": "P12 and P13 rebuild both inputs",
}


def _module_files():
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel.parts[0] == "legacy":
            continue
        yield rel, path


def _imports_legacy(path: pathlib.Path) -> bool:
    """True if this module imports anything from `neurocascade.legacy`."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "legacy" in node.module.split("."):
                return True
            # `from .legacy import x` / `from ..legacy import x`
            if node.module and node.module.split(".")[0] == "legacy":
                return True
            if node.level and node.module is None:
                if any(a.name == "legacy" for a in node.names):
                    return True
            if node.module is None and any(
                a.name.startswith("legacy") for a in node.names
            ):
                return True
        elif isinstance(node, ast.Import):
            if any("legacy" in a.name.split(".") for a in node.names):
                return True
    return False


def test_legacy_contains_exactly_what_was_quarantined():
    assert set(legacy.__all__) == QUARANTINED
    for name in QUARANTINED:
        assert hasattr(legacy, name), f"legacy.{name} did not import"


@pytest.mark.parametrize("name", sorted(QUARANTINED))
def test_package_root_no_longer_re_exports_rodent_modules(name):
    """`neurocascade.drug_pk` must fail, not silently work.

    A quarantine that leaves the old import path alive is decorative.
    """
    assert not hasattr(neurocascade, name), (
        f"neurocascade.{name} still resolves; the quarantine is not enforced. "
        f"Use neurocascade.legacy.{name}."
    )


def test_only_permitted_modules_reach_into_legacy():
    """Pin the rodent dependencies that survived P11.1.

    Failing because a module was REMOVED from this list is good news: delete
    its entry. Failing because a NEW module appeared means rodent material is
    spreading, which is what this test exists to catch.
    """
    found = {
        str(rel) for rel, path in _module_files() if _imports_legacy(path)
    }
    permitted = set(PERMITTED_REACHES) | EXPOSURE

    unexpected = found - permitted
    assert not unexpected, (
        "these modules newly reach into legacy/ rodent material: "
        f"{sorted(unexpected)}. Either remove the dependency or add it to "
        "PERMITTED_REACHES with the task that will retire it."
    )

    stale = permitted - found
    assert not stale, (
        "these entries in PERMITTED_REACHES no longer reach into legacy/: "
        f"{sorted(stale)}. The dependency is gone, so delete the entry."
    )


def test_the_human_layer_never_reaches_into_legacy():
    """The human layer is held to the stricter rule: no reach at all.

    This is what the P11.1 hoist of `transmitter_sign` and `DEFAULT_NODE`
    bought. If it regresses, the species quarantine stops being separable.
    """
    offenders = [
        str(rel)
        for rel, path in _module_files()
        if rel.parts[0] == "human" and _imports_legacy(path)
    ]
    assert not offenders, (
        f"human modules importing rodent material: {offenders}. Anything they "
        "need that is genuinely species-neutral should be hoisted to the "
        "package root, as transmitter_sign.py and circuit.DEFAULT_NODE were."
    )


def test_legacy_never_reaches_into_the_human_layer():
    """The dependency may not run the other way either."""
    offenders = []
    for path in sorted((SRC / "legacy").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "human" in node.module.split("."):
                    offenders.append(str(path.relative_to(SRC)))
                    break
    assert not offenders, f"legacy modules importing the human layer: {offenders}"


def test_hoisted_constants_are_shared_not_duplicated():
    """The species-neutral constants have one definition, not two.

    A copy-paste hoist would leave the mouse and human layers free to drift
    apart silently, which is worse than the import it replaced.
    """
    from neurocascade import circuit, transmitter_sign
    from neurocascade.human import neurotransmitter as human_nt
    from neurocascade.legacy.connectome import build as mouse_build
    from neurocascade.legacy.connectome import neurotransmitter as mouse_nt

    assert mouse_nt.TRANSMITTER_SIGN is transmitter_sign.TRANSMITTER_SIGN
    assert human_nt.TRANSMITTER_SIGN is transmitter_sign.TRANSMITTER_SIGN
    assert mouse_build.DEFAULT_NODE is circuit.DEFAULT_NODE

    # The sign rule itself, which both species apply.
    assert transmitter_sign.TRANSMITTER_SIGN[transmitter_sign.GLUTAMATERGIC] == 1.0
    assert transmitter_sign.TRANSMITTER_SIGN[transmitter_sign.GABAERGIC] == -1.0
    for neuromodulator in (
        transmitter_sign.DOPAMINERGIC,
        transmitter_sign.SEROTONERGIC,
        transmitter_sign.NORADRENERGIC,
        transmitter_sign.CHOLINERGIC,
        transmitter_sign.MIXED,
    ):
        assert transmitter_sign.TRANSMITTER_SIGN[neuromodulator] is None


def test_legacy_docstring_records_the_retirement_policy():
    """The policy travels with the code, not only in the roadmap."""
    doc = legacy.__doc__ or ""
    assert "human-primary" in doc
    assert "species gate" in doc
    assert "docs/roadmap-human.md" in doc
    # The two reasons deletion was rejected must both be stated.
    assert "Cross-species calibration anchors" in doc
    assert "searched-and-empty" in doc
