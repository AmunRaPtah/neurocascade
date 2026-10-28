#!/usr/bin/env python3
"""Freeze the P34 data-construction build (P34.2) into results/P34.2.json.

docs/prereg-p34-go.md requires the fresh held-out set (P34.4) to be labelled,
given Cmax and given Ki by exactly the procedures that existed before it was
built. This script records the sha256 of every file those procedures consist
of, and runs the SmPC parser's train-drug validation (scripts/v2_smpc.py) so
the parser's only evidence is committed beside its hash. tests/test_p34_2_frozen_build.py
fails if any frozen file changes afterwards.

    NC_V2_CACHE=... python scripts/p34_2_freeze.py   # network or a warm emc cache
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "P34.2.json"

FROZEN = [
    ("src/neurocascade/benchmark/v2/liability_rules.json", "label",
     "pre-specified liability term map, table rules and 5-point / 2x placebo-subtracted margin (P33.2, amendments A1-A7)"),
    ("scripts/v2_liabilities.py", "label", "openFDA adverse-reaction table parser applying liability_rules.json"),
    ("src/neurocascade/benchmark/v2/smpc_rules.json", "label",
     "SmPC section 4.8 source, product selection and table normalisation rules (P34.2, amendment SA1)"),
    ("scripts/v2_smpc.py", "label", "SmPC parser: normalises section 4.8 tables and calls v2_liabilities.build unchanged"),
    ("scripts/v2_label_index.py", "label+cmax", "openFDA bulk label index read by v2_liabilities and v2_cmax"),
    ("scripts/v2_cmax.py", "cmax", "therapeutic total Cmax: label/PubMed candidate snippets, verbatim-snippet build, Cmax,u via plasma fu"),
    ("scripts/v2_pk.py", "cmax", "builds inputs/plasma_fu.json, the fu that v2_cmax.py joins for Cmax,u"),
    ("scripts/v2_binding.py", "binding", "ChEMBL human Ki panel: filters, per-document mean then median"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> list[dict]:
    return [{"path": p, "role": role, "what": what, "sha256": sha256(ROOT / p)} for p, role, what in FROZEN]


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import v2_smpc
    val = v2_smpc.validate_train()
    out = {
        "task": "P34.2",
        "description": ("Frozen P34 data-construction build. P34.4 must label, Cmax and Ki the fresh held-out set "
                        "with exactly these files; tests/test_p34_2_frozen_build.py pins their sha256. "
                        "No held-out or fresh drug was read: the SmPC validation ran on v2_partition() train INNs only."),
        "frozen_files": manifest(),
        "smpc_parser": {"frozen": True, "used_for": "fresh-set drugs with no openFDA placebo-subtracted label (smpc_rules.json precedence)",
                        "train_validation": val},
        "n_scored": val["n_compared_drugs"],
        "n_scored_definition": ("train drugs whose SmPC-derived placebo-subtracted labels were compared with their frozen "
                                "openFDA labels (truth/liabilities.json); no held-out drug is scored by this task"),
        "heldout": "not touched; v2 held-out stays burned, the fresh set does not exist yet (P34.4)",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in val.items() if k not in ("comparisons", "drugs")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
