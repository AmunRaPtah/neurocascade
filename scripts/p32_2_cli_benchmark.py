"""P32.2: run the ``neurocascade`` CLI end to end on every benchmark drug.

For each P28.1 drug, ``export-benchmark`` writes the drug's preclinical
package (binding panel + ADME record, no truth) as a CSV file. ``predict``
then reads that file and writes the report and the strict-JSON result. The CL
point and bands are read back **from the CLI's JSON file** and scored against
P28.4 human IV CL truth. The script also checks that every report it wrote
passes ``report.audit_report``: every number linked, the not-validated
label beside every predicted number, no broken in-page link. This tests the file
path, CSV round trip, report and exit codes. It is not a new model. The
numbers must equal P32.1's ``score_clearance``, and the script checks that.

    python scripts/p32_2_cli_benchmark.py        # writes results/P32.2.json
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurocascade import cli  # noqa: E402
from neurocascade.benchmark import load_split, load_truth_record  # noqa: E402
from neurocascade.benchmark import human_pk_translation as _pk  # noqa: E402
from neurocascade.benchmark.v2 import v2_partition  # noqa: E402
from neurocascade.compound import PROVENANCE_LABEL, score_clearance  # noqa: E402
from neurocascade.report import audit_report  # noqa: E402

RESULT_PATH = ROOT / "results" / "P32.2.json"


def score_cli(drugs: list[str], workdir: Path) -> dict:
    """Score human CL as read from the CLI's JSON output, per band."""
    by_band: dict[str, dict] = {}
    refused, missing_truth, no_cl_truth, audit_failures = [], [], [], []
    n_reports = 0
    for inn in drugs:
        pkg, report, out = (workdir / f"{inn}.csv", workdir / f"{inn}.html",
                            workdir / f"{inn}.json")
        assert cli.main(["export-benchmark", inn, "-o", str(pkg)]) == cli.EXIT_OK
        rc = cli.main(["predict", str(pkg), "-o", str(report), "--json", str(out)])
        if rc == cli.EXIT_REFUSED:
            refused.append(inn)
            continue
        assert rc == cli.EXIT_OK, (inn, rc)
        n_reports += 1
        if any(audit_report(report.read_text()).values()):
            audit_failures.append(inn)
        try:
            load_truth_record(inn)
        except KeyError:
            missing_truth.append(inn)
            continue
        truth = _pk._human_truth(inn, "cl")
        if truth is None:
            no_cl_truth.append(inn)
            continue
        rec = json.loads(out.read_text())["human_pk"]["clearance"]
        band = by_band.setdefault(rec["provenance"]["band"],
                                  {"drugs": [], "fold_errors": [], "covered": {}})
        band["drugs"].append(inn)
        band["fold_errors"].append(max(rec["point"] / truth[0], truth[0] / rec["point"]))
        for lv, b in rec["bands"].items():
            upper = math.inf if b["upper"] is None else b["upper"]
            band["covered"][lv] = band["covered"].get(lv, 0) + (b["lower"] <= truth[0] <= upper)
    v2_train = set(v2_partition()[0])
    for name, band in by_band.items():
        fe = band["fold_errors"]
        band["n"] = len(fe)
        band["gmfe"] = float(math.exp(sum(math.log(x) for x in fe) / len(fe)))
        band["in_sample"] = (sorted(d for d in band["drugs"] if d in v2_train)
                             if name == "human_cl_v2_train_mean" else [])
    n = sum(b["n"] for b in by_band.values())
    n_oos = n - sum(len(b["in_sample"]) for b in by_band.values())
    return {"status": "scored" if n else "searched-and-empty", "n_scored": n,
            "n_out_of_sample": n_oos, "n_reports_written": n_reports, "by_band": by_band,
            "refused": sorted(refused), "missing_truth": sorted(missing_truth),
            "no_cl_truth": sorted(no_cl_truth),
            "reports_failing_audit": sorted(audit_failures)}


def matches_p32_1(cli_score: dict, direct: dict) -> bool:
    """The CLI path must reproduce predict_compound's own score exactly."""
    if set(cli_score["by_band"]) != set(direct["by_band"]):
        return False
    for name, band in cli_score["by_band"].items():
        ref = direct["by_band"][name]
        if band["drugs"] != ref["drugs"] or band["covered"] != ref["covered"]:
            return False
        if not math.isclose(band["gmfe"], ref["gmfe"], rel_tol=1e-12):
            return False
    return cli_score["refused"] == direct["refused"]


def main() -> None:
    split = load_split()
    out = {"task": "P32.2", "provenance": PROVENANCE_LABEL}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("train", "heldout"):
            d = Path(tmp) / name
            d.mkdir()
            score = score_cli(split[name], d)
            score["matches_p32_1_score_clearance"] = matches_p32_1(
                score, score_clearance(split[name]))
            out[name] = score
    out["n_scored"] = out["train"]["n_out_of_sample"]
    out["n_scored_definition"] = (
        "P28.1 train drugs run through the installed-CLI path (export-benchmark to CSV, "
        "predict from the CSV file, CL read back from the --json output) and scored against "
        "P28.4 human IV CL truth, excluding train-mean-fallback drugs whose own CL is inside "
        "the v2 train mean")
    out["heldout"]["note"] = (
        "host-pending: the same score_cli(load_split()['heldout']) on a checkout with P28.4 "
        "held-out truth; in this session every non-refused held-out drug is missing_truth")
    RESULT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_scored",)}),
          {n: (out[n]["n_scored"], out[n]["matches_p32_1_score_clearance"])
           for n in ("train", "heldout")})


if __name__ == "__main__":
    main()
