#!/usr/bin/env python3
"""Freeze the P34 data-construction build (P34.2) into results/P34.2.json.

docs/prereg-p34-go.md requires the fresh held-out set (P34.4) to be labelled,
given Cmax and given Ki by exactly the procedures that existed before it was
built. This script records the sha256 of every file those procedures consist
of. tests/test_p34_2_frozen_build.py pins the same hashes as literals and fails
if any frozen file changes afterwards.

No EU SmPC parser is frozen, so none may be used for the fresh set: the train
labels the P34.1 candidate was selected on are openFDA-only, and an SmPC route
validated only on all-negative labels (3 train drugs) has no positive control.

    python scripts/p34_2_freeze.py
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
     "pre-specified liability term map, table rules and placebo-subtracted margin (P33.2, amendments A1-A7)"),
    ("scripts/v2_liabilities.py", "label", "openFDA adverse-reaction table parser applying liability_rules.json"),
    ("scripts/v2_label_index.py", "label+cmax", "openFDA bulk label index read by v2_liabilities and v2_cmax"),
    ("scripts/v2_cmax.py", "cmax", "therapeutic total Cmax: label/PubMed candidate snippets, verbatim-snippet build, Cmax,u via plasma fu"),
    ("scripts/v2_pk.py", "fu", "ChEMBL plasma fu build (inputs/plasma_fu.json), priority-1 fu source"),
    ("scripts/v2_fu_merge.py", "fu", "merge rule for additional measured plasma fu sources on top of v2_pk (PR #52)"),
    ("scripts/v2_fu_drugcentral_fetch.py", "fu", "DrugCentral (Lombardo 2018) plasma fu fetch with exact-title match (PR #52)"),
    ("scripts/v2_fu_structure_verify.py", "fu", "InChI cross-check that drops any DrugCentral fu match whose structure differs (PR #52)"),
    ("scripts/v2_fu_label_extract.py", "fu", "openFDA label %-bound extraction feeding v2_fu_merge (PR #52)"),
    ("scripts/v2_binding.py", "binding", "human Ki panel: ChEMBL filters, censored bounds, GtoPdb and PDSP merge, per-document mean then median (PR #51)"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> list[dict]:
    return [{"path": p, "role": role, "what": what, "sha256": sha256(ROOT / p)} for p, role, what in FROZEN]


def main() -> int:
    truth = json.loads((ROOT / "src/neurocascade/benchmark/v2/truth/liabilities.json").read_text())
    out = {
        "task": "P34.2",
        "description": ("Frozen P34 data-construction build. P34.4 must label, Cmax and Ki the fresh held-out set "
                        "with exactly these files; tests/test_p34_2_frozen_build.py pins their sha256. "
                        "No held-out or fresh drug was read."),
        "frozen_files": manifest(),
        "train_labels_built_with_frozen_rules": truth["rules_sha256"] == sha256(ROOT / FROZEN[0][0]),
        "label_sources_for_fresh_set": ["openFDA (scripts/v2_liabilities.py), identical to the train labels"],
        "smpc_parser": {
            "frozen": False,
            "decision": ("not used for the fresh set. The prereg allows SmPC tables only if frozen here; PR #49's "
                         "parser was validated on 3 train drugs whose 12 labels were all negative (no positive "
                         "control), and only 3 of 180 train drugs with an emc SmPC carried a placebo table. "
                         "Fresh labels therefore come from the same openFDA-only procedure as the train labels."),
            "superseded_pr": 49,
        },
        "not_freezable_by_hash": [
            "Cmax snippet curation is a judgement made per drug; P34.4 must follow the rules written in v2_cmax.py",
            "live sources (openFDA, ChEMBL, GtoPdb, PDSP, DrugCentral) can change; P34.4 records retrieval dates",
        ],
        "heldout": "not touched; the v2 held-out stays burned and the fresh set does not exist yet (P34.4)",
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({f["path"]: f["sha256"][:12] for f in out["frozen_files"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
