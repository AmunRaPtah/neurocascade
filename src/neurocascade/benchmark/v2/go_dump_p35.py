"""Held-out dump for the P35.4 GO / NO-GO verdict (``docs/prereg-p35-go.md``).

    NC_HOST_EVAL=1 python -m neurocascade.benchmark.v2.go_dump_p35 OUT.json
    python pipeline/go_verdict.py OUT.json

With ``NC_HOST_EVAL=1`` (the host clone: full v2 truth and the frozen SIDER rows
of the P35 held-out drugs) this builds the P35 held-out SIDER labels, with the
P35.2 rule (``sider_labels.label_drug``, unchanged), and writes the schema
``pipeline/go_verdict.py`` reads:

    {"liabilities": {name: {"drugs": [inn...], "y": [0/1...],
                            "candidate": [score...],
                            "baselines": {name: [score...]},
                            "secondary": {"p34_3_frozen_candidate": [score...]}}}}

* Held-out drugs: P35 held-out (``sha256(inn) % 10 in {0, 1}``) universe members
  (anchor Ki, SIDER record, Cmax value) with a 0/1 label, restricted to the
  go_verdict common set (every baseline defined), the P35.3 rule.
* ``candidate``: ``p35_candidate.predict_cohort`` with the frozen params, ranked
  within that held-out cohort.  Its sha256 and the params' are checked against
  ``results/P35.3.json`` before anything is scored.
* ``baselines``: occupancy_max, strongest-target Ki, 1-NN fingerprint and ATC
  class prevalence, computed by ``p35_selection._views`` -- the P35.3 code --
  with the fitted ones (1-NN, ATC) fitted on the P35-train pool only.
* ``secondary``: the P34.3 frozen candidate (``frozen_candidate.predict_cohort``)
  on the same drugs.  go_verdict ignores it; this module prints its AUROC.

The prereg admits a liability to the decision only with >= 8 positives and >= 8
negatives on the held-out common set.  ``go_verdict.MIN_CLASS`` is 5 (P33.4's
floor), so a liability below 8/8 is NOT passed to go_verdict: it goes to
``below_prereg_floor`` with its counts, and is printed here.

Without ``NC_HOST_EVAL`` nothing held-out is read: the dump has empty
liabilities and names every P35 held-out drug as searched-and-empty, and
go_verdict prints NO-GO on it.  ``load_heldout_rows`` refuses to run without
the flag.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from . import V2_DIR, load_v2_binding, load_v2_drugs, load_v2_human_cmax, v2_heldout
from . import sider_labels as sl
from .candidate_benchmark import _cu, _ki_by_gene
from .chain_benchmark import auroc
from .frozen_candidate import predict_cohort as p34_predict_cohort
from .p35_candidate import fingerprint, load_params, predict_cohort
from .p35_selection import BASELINES, _views, load_inputs

HELDOUT_ROWS = V2_DIR / "truth" / "p35_sider_heldout_rows.json"
ROWS_KEY = "p35_heldout_rows"
REPO = V2_DIR.parents[3]
P35_3_RESULT = REPO / "results" / "P35.3.json"
# docs/prereg-p35-go.md, "Rule": GO only on a liability whose held-out common set
# has >= 8 positives and >= 8 negatives.
PREREG_MIN_CLASS = 8


def host_eval() -> bool:
    return os.environ.get("NC_HOST_EVAL") == "1"


def heldout_inns() -> list[str]:
    return sorted(d["inn"] for d in load_v2_drugs() if sl.p35_heldout(d["inn"]))


def check_freeze() -> dict[str, str]:
    """The P35.3 frozen code and params, unchanged since the freeze, or raise."""
    res = json.loads(P35_3_RESULT.read_text())
    out = {}
    for part in ("code", "params"):
        got = hashlib.sha256((REPO / res[part]["path"]).read_bytes()).hexdigest()
        if got != res[part]["sha256"]:
            raise RuntimeError(f"{res[part]['path']} changed after the P35.3 freeze")
        out[res[part]["path"]] = got
    return out


def load_heldout_rows() -> dict[str, Any] | None:
    """The frozen SIDER rows of the P35 held-out drugs; host-only.

    None when the file is absent (session trees: guards.hide_heldout deletes it)."""
    if not host_eval():
        raise RuntimeError("P35 held-out SIDER rows are read only under NC_HOST_EVAL=1")
    if not HELDOUT_ROWS.exists():
        return None
    rows = json.loads(HELDOUT_ROWS.read_text())[ROWS_KEY]
    bad = sorted(inn for inn in rows if not sl.p35_heldout(inn))
    if bad:
        raise ValueError(f"P35-train drugs in the held-out rows file: {bad}")
    return rows


def build_heldout_labels(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """P35 held-out labels, the P35.2 rule unchanged (``sider_labels.label_drug``).
    A train INN is refused: train labels come only from labels_train.json."""
    bad = sorted(inn for inn in rows if not sl.p35_heldout(inn))
    if bad:
        raise ValueError(f"P35-train drugs passed to the held-out label builder: {bad}")
    return {inn: sl.label_drug(d["mapped_rows"], d["has_record"], d.get("listed_pts", ()),
                               d.get("n_freq_rows"))
            for inn, d in sorted(rows.items())}


def _auroc_line(y: list[int], s: list[float | None]) -> str:
    keep = [(a, b) for a, b in zip(y, s) if b is not None]
    a = auroc([p[0] for p in keep], [p[1] for p in keep]) if keep else None
    return "n/a" if a is None else f"{a:.3f} (n {len(keep)})"


def build_dump(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Score the frozen candidate, every baseline and the P34.3 secondary on the
    P35 held-out common set of each receptor liability."""
    frozen = check_freeze()
    params = load_params()
    inputs = load_inputs()  # train labels, binding, Cmax, drugs; asserts train-only
    binding, cmax = inputs["binding"], inputs["cmax"]
    labels = build_heldout_labels(rows)
    anchors = list(json.loads((V2_DIR / "drugs.json").read_text())["anchor_targets"])
    universe = {inn: sl.universe_row(inn, anchors, binding.get(inn), rows[inn]["has_record"],
                                     cmax.get(inn), v2_heldout(inn)) for inn in sorted(rows)}
    members = [inn for inn, u in universe.items() if u["in_universe"] is True]
    fp = {inn: fingerprint(_ki_by_gene(binding, inn), params["panel"],
                           float(params["fingerprint_floor"])) for inn in members}

    liabilities: dict[str, Any] = {}
    below: dict[str, Any] = {}
    for effect in params["liability_receptors"]:
        train_lab = inputs["labels"]
        record = [i for i in sorted(train_lab) if train_lab[i][effect]["label"] in (0, 1)]
        scored = [i for i in members if labels[i][effect]["label"] in (0, 1)]
        views = {i: _views(i, effect, inputs, params, record, fp) for i in scored}
        common = [i for i in scored if all(views[i][b] is not None for b in BASELINES)]
        cu = [_cu(cmax, i) for i in common]
        ki = [_ki_by_gene(binding, i) for i in common]
        entry = {
            "drugs": common,
            "y": [int(labels[i][effect]["label"]) for i in common],
            "candidate": predict_cohort(effect, common, cu, ki, params) if common else [],
            "baselines": {b: [views[i][b] for i in common] for b in BASELINES},
            "secondary": {"p34_3_frozen_candidate": p34_predict_cohort(effect, cu, ki) if common else []},
        }
        y_ok = [y for y, c in zip(entry["y"], entry["candidate"]) if c is not None]
        npos, nneg = sum(y_ok), len(y_ok) - sum(y_ok)
        if npos >= PREREG_MIN_CLASS and nneg >= PREREG_MIN_CLASS:
            liabilities[effect] = entry
        else:
            below[effect] = {**entry, "n_positive": npos, "n_negative": nneg}
    return {
        "task": "P35.4",
        "prereg": "docs/prereg-p35-go.md",
        "candidate": "P35.3 frozen rank_occupancy_knn10 (p35_candidate.predict_cohort)",
        "frozen_sha256": frozen,
        "host_eval": True,
        "status": "scored",
        "n_heldout": len(rows),
        "n_heldout_universe": len(members),
        "universe": universe,
        "liabilities": liabilities,
        "below_prereg_floor": below,
        "not_scoreable": {e: "no pre-specified receptor: the frozen candidate cannot score it"
                          for e in sl.LIABILITY_PTS if e not in params["liability_receptors"]},
    }


def empty_dump(reason: str) -> dict[str, Any]:
    return {"task": "P35.4", "host_eval": host_eval(), "status": "searched-and-empty",
            "reason": reason, "liabilities": {}, "heldout_drugs": heldout_inns()}


def report(dump: Mapping[str, Any]) -> str:
    """Human lines for the host_eval tail (never an NC_GO_VERDICT line)."""
    if dump["status"] != "scored":
        return (f"P35.4 dump: {dump['reason']}; {len(dump['heldout_drugs'])} P35 held-out drugs "
                "searched-and-empty")
    lines = [f"P35.4 dump: {dump['n_heldout']} P35 held-out drugs, "
             f"{dump['n_heldout_universe']} in the universe"]
    for effect, d in sorted({**dump["liabilities"], **dump["below_prereg_floor"]}.items()):
        tag = "judged" if effect in dump["liabilities"] else (
            f"below prereg 8/8 floor (pos {d['n_positive']}, neg {d['n_negative']}), not judged")
        lines.append(f"{effect}: common n {len(d['drugs'])} (pos {sum(d['y'])}), {tag}; "
                     f"candidate AUROC {_auroc_line(d['y'], d['candidate'])}; "
                     f"P34.3 secondary AUROC {_auroc_line(d['y'], d['secondary']['p34_3_frozen_candidate'])}")
    for effect, why in dump["not_scoreable"].items():
        lines.append(f"{effect}: not scored ({why})")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: python -m neurocascade.benchmark.v2.go_dump_p35 OUT.json\n")
        return 2
    if not host_eval():
        dump = empty_dump("NC_HOST_EVAL not set: P35 held-out SIDER labels are not built in a session tree")
    else:
        rows = load_heldout_rows()
        dump = (empty_dump("held-out SIDER rows file absent from this checkout") if rows is None
                else build_dump(rows))
    Path(argv[1]).write_text(json.dumps(dump, indent=1) + "\n")
    sys.stdout.write(report(dump) + f"\nwritten to {argv[1]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
