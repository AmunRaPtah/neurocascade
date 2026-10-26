"""Read-only API for the v2 benchmark data universe.

The benchmark truth is deliberately accessed through this module.  In
particular, callers must not open a truth JSON file and infer membership from
whatever rows happen to be present: the conductor checkout removes held-out
rows, while the host checkout restores them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


V2_DIR = Path(__file__).resolve().parent
_DRUGS = V2_DIR / "drugs.json"
_BINDING = V2_DIR / "inputs" / "binding.json"
_PLASMA_FU = V2_DIR / "inputs" / "plasma_fu.json"
_HUMAN_PK = V2_DIR / "truth" / "human_pk.json"
_HUMAN_CMAX = V2_DIR / "truth" / "human_cmax.json"
_LIABILITIES = V2_DIR / "truth" / "liabilities.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_v2_drugs() -> list[dict[str, Any]]:
    """Return the complete v2 universe, as published by P33.1."""
    return list(_read(_DRUGS)["drugs"])


def load_v2_binding() -> dict[str, dict[str, Any]]:
    """Return the per-drug human binding panels keyed by INN."""
    return dict(_read(_BINDING)["drugs"])


def load_v2_plasma_fu() -> dict[str, dict[str, Any]]:
    """Return the v2 plasma-fraction input rows keyed by INN."""
    return {row["inn"]: row for row in _read(_PLASMA_FU)["drugs"]}


def load_v2_human_pk() -> dict[str, dict[str, Any]]:
    """Return human PK truth rows currently available, keyed by INN.

    Missing rows are meaningful: the host may add held-out rows without
    changing this API or the scoring code.
    """
    return {row["inn"]: row for row in _read(_HUMAN_PK)["drugs"]}


def load_v2_human_cmax() -> dict[str, dict[str, Any]]:
    """Return human therapeutic plasma Cmax rows currently available, keyed by INN.

    Rows have status "value" (cmax with source snippet, ng/mL and nM, plus
    unbound Cmax from inputs/plasma_fu.json) or "searched_and_empty". Like
    human_pk, held-out rows are stripped from the agent's tree.
    """
    return {row["inn"]: row for row in _read(_HUMAN_CMAX)["drugs"]}


def load_v2_liabilities() -> dict[str, dict[str, Any]]:
    """Return liability truth rows currently available, keyed by INN."""
    return {row["inn"]: row for row in _read(_LIABILITIES)["drugs"]}


def v2_heldout(inn: str) -> bool:
    """Return the frozen v2 hash partition for one INN.

    This is the benchmark partition rule, not the P28.1 ``split.json``.  It
    is kept here so benchmark consumers do not need to import pipeline code.
    """
    key = "nc-v2:" + inn.strip().lower()
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0


def v2_partition() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(train, heldout)`` INNs from the v2 hash rule."""
    inns = sorted(row["inn"] for row in load_v2_drugs())
    return (
        tuple(inn for inn in inns if not v2_heldout(inn)),
        tuple(inn for inn in inns if v2_heldout(inn)),
    )


__all__ = [
    "load_v2_binding",
    "load_v2_drugs",
    "load_v2_human_cmax",
    "load_v2_human_pk",
    "load_v2_liabilities",
    "load_v2_plasma_fu",
    "v2_heldout",
    "v2_partition",
]
