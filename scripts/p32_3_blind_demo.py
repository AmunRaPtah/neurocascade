"""P32.3: run predict_compound blind on held-out benchmark drugs and tabulate.

Writes three files. ``results/P32.3.json`` holds every row, the summaries, and
``n_scored`` for the host. ``results/P32.3_blind_demo.md`` is the committed
five-drug table. ``results/P32.3_blind_demo.html`` is the same table in the
report's audited conventions. On a checkout without P28.4 held-out truth, the
dose and occupancy actuals are host-pending, and CL, Vss and the liabilities
fall back to the v2 truth for the same drug. The host runs the same code with
full truth.

    python scripts/p32_3_blind_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurocascade.blind_demo import render_html, render_markdown, run_blind_demo  # noqa: E402
from neurocascade.cli import json_safe  # noqa: E402
from neurocascade.report import audit_report  # noqa: E402


def main() -> None:
    result = run_blind_demo()
    page = render_html(result)
    audit = audit_report(page)
    if any(audit.values()):
        raise SystemExit(f"demo page fails the report audit: {audit}")
    out = ROOT / "results"
    (out / "P32.3.json").write_text(json.dumps(json_safe(result), indent=1, allow_nan=False) + "\n")
    (out / "P32.3_blind_demo.md").write_text(render_markdown(result))
    (out / "P32.3_blind_demo.html").write_text(page)
    s = result["all_runnable"]["summary"]
    print(f"demo drugs {result['demo_drugs']}; n_scored {s['n_scored']} "
          f"(compound-specific {s['n_scored_compound_specific']}); "
          f"host-pending {len(s['host_pending'])}")


if __name__ == "__main__":
    main()
