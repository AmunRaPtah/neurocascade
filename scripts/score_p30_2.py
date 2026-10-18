#!/usr/bin/env python3
"""Write results/P30.2.json from the scoring function itself.

The results file is generated, never hand-edited: the acceptance review of
PR #31 found that a hand-written summary had drifted from what
`score_p30_2()` returns. Run this on a checkout that holds the frozen held-out
truth (the conductor's worker sandbox deliberately strips it):

    python scripts/score_p30_2.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurocascade.benchmark.effect_scoring import score_p30_2  # noqa: E402


def main() -> int:
    result = score_p30_2()
    out = ROOT / "results" / "P30.2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {out} (n_scored={result['n_scored']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
