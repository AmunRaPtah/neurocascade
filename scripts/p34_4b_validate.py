#!/usr/bin/env python3
"""P34.4b: apply the train validation bar of docs/prereg-p34-amendment-smpc.md.

Reads the output of ``scripts/v2_smpc.py --validate-train`` (train drugs
only) and writes results/P34.4b.json. The bar, fixed in the amendment before
the run:

1. at least 3 compared labels are openFDA-positive;
2. the SmPC label is positive on at least 2/3 of those;
3. the SmPC label agrees on at least 90% of the openFDA-negative compared labels.

weight_gain is withheld under A7 and is not compared. If the bar fails the
parser is not frozen or applied, and the result is EMPTY.

    python scripts/p34_4b_validate.py VALIDATION.json

Touches no fresh drug.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
SCORED = ("eps", "sedation", "anticholinergic", "orthostatic_hypotension")
MIN_POS, SENS, NEG_AGREE = 3, Fraction(2, 3), Fraction(9, 10)
PARSER = ("scripts/v2_smpc.py", "src/neurocascade/benchmark/v2/smpc_rules.json")


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def bar(comparisons: list[dict]) -> dict:
    rows = [c for c in comparisons if c["liability"] in SCORED]
    pos = [c for c in rows if c["fda_label"] == 1]
    neg = [c for c in rows if c["fda_label"] == 0]
    tp = sum(c["smpc_label"] == 1 for c in pos)
    tn = sum(c["smpc_label"] == 0 for c in neg)
    checks = {
        "fda_positive_controls": {"n": len(pos), "need": MIN_POS, "pass": len(pos) >= MIN_POS},
        "sensitivity": {"smpc_pos": tp, "of": len(pos), "need": "2/3",
                        "pass": bool(pos) and Fraction(tp, len(pos)) >= SENS},
        "negative_agreement": {"smpc_neg": tn, "of": len(neg), "need": "9/10",
                               "pass": bool(neg) and Fraction(tn, len(neg)) >= NEG_AGREE},
    }
    return {"n_compared_labels": len(rows), "checks": checks,
            "pass": all(c["pass"] for c in checks.values())}


def main(path: str) -> int:
    val = json.loads(Path(path).read_text())
    b = bar(val["comparisons"])
    out = {
        "task": "P34.4b", "prereg": "docs/prereg-p34-amendment-smpc.md",
        "deviation": "SmPC parser added after P34.4 fresh openFDA counts were seen (disclosed)",
        "parser": [{"path": p, "sha256": sha(p)} for p in PARSER],
        "train_validation": {k: val[k] for k in ("n_train", "n_with_emc_smpc",
                                                 "n_smpc_placebo_subtracted", "n_fetch_failed",
                                                 "compared_drugs") if k in val},
        "bar": b,
        "parser_frozen": b["pass"],
        "applied_to_fresh": False,
        "result": None if b["pass"] else "EMPTY",
        "gap": None if b["pass"] else
               "SmPC parser failed the pre-registered train validation bar; not frozen, not "
               "applied, no fresh drug touched",
        "comparisons": val["comparisons"],
    }
    (ROOT / "results" / "P34.4b.json").write_text(json.dumps(out, indent=1) + "\n")
    print("TASK_RESULT:", "PARSER_PASSED" if b["pass"] else "EMPTY", json.dumps(b["checks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
