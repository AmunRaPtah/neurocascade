"""Phase 28 retrospective benchmark: the drug list and its frozen held-out split.

``drugs.json`` and ``split.json`` were built by ``scripts/build_benchmark.py``
(P28.1). The split is frozen: its sha256 is pinned here, in
``tests/test_benchmark_split.py`` and in docs/validation-ledger.md, and
``load_split`` refuses a file whose bytes have changed. Every headline number
from Phase 29 on is scored on ``heldout``; nothing may be fitted on it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
DRUGS_PATH = BENCHMARK_DIR / "drugs.json"
SPLIT_PATH = BENCHMARK_DIR / "split.json"

SPLIT_SHA256 = "54a08acfe76b21237bd3152653184796c875f064c66408c9449980f9df6628d9"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_drugs() -> list[dict]:
    """The benchmark drug records, one per drug, sorted by INN."""
    return json.loads(DRUGS_PATH.read_text())["drugs"]


def load_split() -> dict:
    """The frozen split as ``{"train": [...], "heldout": [...]}`` of INNs.

    Raises ``RuntimeError`` if split.json no longer hashes to SPLIT_SHA256.
    """
    digest = sha256_of(SPLIT_PATH)
    if digest != SPLIT_SHA256:
        raise RuntimeError(
            f"benchmark split.json sha256 {digest} != frozen {SPLIT_SHA256}; "
            "the P28.1 split may not be redrawn or edited"
        )
    split = json.loads(SPLIT_PATH.read_text())
    return {"train": list(split["train"]), "heldout": list(split["heldout"])}
