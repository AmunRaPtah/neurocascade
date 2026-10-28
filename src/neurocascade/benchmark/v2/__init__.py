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
_HUMAN_PK_CORRECTIONS = V2_DIR / "truth" / "human_pk_corrections.json"
_HUMAN_CMAX = V2_DIR / "truth" / "human_cmax.json"
_HUMAN_CMAX_CORRECTIONS = V2_DIR / "truth" / "human_cmax_corrections.json"
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


def load_v2_human_pk_corrections() -> dict[str, Any] | None:
    """Return the provenance corrections to the frozen human_pk.json, or None
    when the corrections file is absent from this checkout."""
    if not _HUMAN_PK_CORRECTIONS.exists():
        return None
    return _read(_HUMAN_PK_CORRECTIONS)


def apply_human_pk_corrections(
    rows: dict[str, dict[str, Any]], corrections: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Apply ``drop`` (field -> None) and ``replace`` (field -> recomputed
    entry) to human PK rows keyed by INN, without mutating the input. INNs not
    present in ``rows`` are skipped, never created."""
    if not corrections:
        return rows
    out = {inn: dict(row) for inn, row in rows.items()}
    for inn, entry in corrections.get("drop", {}).items():
        if inn in out:
            for field in entry["fields"]:
                out[inn][field] = None
    for inn, fields in corrections.get("replace", {}).items():
        if inn in out:
            for field, value in fields.items():
                out[inn][field] = dict(value)
    return out


def load_v2_human_pk(*, corrected: bool = True) -> dict[str, dict[str, Any]]:
    """Return human PK truth rows currently available, keyed by INN.

    Missing rows are meaningful: the host may add held-out rows without
    changing this API or the scoring code.

    human_pk.json is frozen; by default the provenance corrections in
    human_pk_corrections.json (2026-10-26 audit: in-vitro, partial-clearance,
    non-IV and wrong-entity sources removed) are applied on top of it.
    ``corrected=False`` returns the frozen file as published.
    """
    rows = {row["inn"]: row for row in _read(_HUMAN_PK)["drugs"]}
    if not corrected:
        return rows
    return apply_human_pk_corrections(rows, load_v2_human_pk_corrections())


def load_v2_human_cmax_corrections() -> dict[str, Any] | None:
    """Return the unbound-exposure corrections to the frozen human_cmax.json,
    or None when the corrections file is absent from this checkout."""
    if not _HUMAN_CMAX_CORRECTIONS.exists():
        return None
    return _read(_HUMAN_CMAX_CORRECTIONS)


def apply_human_cmax_corrections(
    rows: dict[str, dict[str, Any]], corrections: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Apply ``drop`` (field -> None) and ``replace`` (field -> new value) to
    human Cmax rows keyed by INN, without mutating the input. INNs not
    present in ``rows`` are skipped, never created. Unlike
    ``apply_human_pk_corrections``, replacement values are stored as-is
    (not wrapped in ``dict(...)``), since human_cmax.json's fu-derived
    fields include plain scalars (cmax_unbound_nM, cmax_unbound_ng_per_ml)
    and lists (fu_flags), not only nested dicts."""
    if not corrections:
        return rows
    out = {inn: dict(row) for inn, row in rows.items()}
    for inn, entry in corrections.get("drop", {}).items():
        if inn in out:
            for field in entry["fields"]:
                out[inn][field] = None
    for inn, fields in corrections.get("replace", {}).items():
        if inn in out:
            for field, value in fields.items():
                out[inn][field] = value
    return out


def load_v2_human_cmax(*, corrected: bool = True) -> dict[str, dict[str, Any]]:
    """Return human therapeutic plasma Cmax rows currently available, keyed by INN.

    Rows have status "value" (cmax with source snippet, ng/mL and nM, plus
    unbound Cmax from inputs/plasma_fu.json) or "searched_and_empty". Like
    human_pk, held-out rows are stripped from the agent's tree.

    human_cmax.json is frozen; by default the unbound-exposure corrections in
    human_cmax_corrections.json (fu, fu_flags, cmax_unbound_nM,
    cmax_unbound_ng_per_ml recomputed from the corrected inputs/plasma_fu.json)
    are applied on top of it. ``corrected=False`` returns the frozen file as
    published, with the original (pre-correction) fu-derived fields.
    """
    rows = {row["inn"]: row for row in _read(_HUMAN_CMAX)["drugs"]}
    if not corrected:
        return rows
    return apply_human_cmax_corrections(rows, load_v2_human_cmax_corrections())


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
    "apply_human_cmax_corrections",
    "apply_human_pk_corrections",
    "load_v2_binding",
    "load_v2_drugs",
    "load_v2_human_cmax",
    "load_v2_human_cmax_corrections",
    "load_v2_human_pk",
    "load_v2_human_pk_corrections",
    "load_v2_liabilities",
    "load_v2_plasma_fu",
    "v2_heldout",
    "v2_partition",
]
