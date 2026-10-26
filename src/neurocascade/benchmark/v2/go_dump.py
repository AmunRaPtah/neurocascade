"""Held-out dump for the P33.4 GO / NO-GO verdict.

    python -m neurocascade.benchmark.v2.go_dump OUT.json

With ``NC_HOST_EVAL=1`` (the host's frozen-truth checkout) this writes, for
each candidate liability, the held-out drugs, their 0/1 labels, the candidate
scores and every baseline score, in the exact schema ``pipeline/go_verdict.py``
reads:

    {"liabilities": {name: {"drugs": [inn...], "y": [0/1...],
                            "candidate": [score...],
                            "baselines": {name: [score or null...]}}}}

The baselines are the P33.3 set (strongest-target Ki, 1-NN fingerprint,
ATC-class prevalence) plus the Cmax,u/Ki occupancy baseline the pre-registration
adds.  The candidate scores come from ``candidate.liability_predict`` -- the
same function used for train leave-one-drug-out -- fitted on train drugs only
and applied to held-out drugs (one model, no held-out refit).  That function
imports no truth module or file; the (allowed) truth reads needed to assemble
features happen here, in the harness.

Without ``NC_HOST_EVAL`` the session tree has no held-out truth, so this writes
an empty-liabilities dump that records every held-out drug as searched-and-empty
and exits cleanly; ``go_verdict.py`` then prints NO-GO on it.  The binding
verdict is the host run with the flag set and the frozen truth present.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping

import numpy as np

from . import (
    load_v2_binding,
    load_v2_drugs,
    load_v2_human_cmax,
    load_v2_liabilities,
    v2_partition,
)
from .candidate import (
    L2_LAMBDA,
    LIABILITY_RECEPTORS,
    feature_vector,
    liability_predict,
    occupancy_baseline,
)
from .candidate_benchmark import CANDIDATE_LIABILITIES, _cu, _ki_by_gene
from .chain_benchmark import _all_targets, _atc_class_prevalence, _binding_score, _fingerprint


def _candidate_heldout_scores(
    effect: str,
    train: list[str],
    held: list[str],
    binding: Mapping[str, Any],
    cmax: Mapping[str, Any],
    liabilities: Mapping[str, Any],
) -> dict[str, float | None]:
    """Fit the candidate on train (label + Cmax) and score held-out drugs.

    One model, fitted on train only, applied to the held-out rows -- exactly the
    estimator used in train LODO, never refitted on held-out data."""
    receptors = LIABILITY_RECEPTORS[effect]
    tr = [
        inn for inn in sorted(train)
        if liabilities.get(inn, {}).get(effect, {}).get("label") is not None
    ]
    tr_feat = []
    tr_y = []
    for inn in tr:
        fv = feature_vector(_cu(cmax, inn), _ki_by_gene(binding, inn), receptors)
        if fv is not None:
            tr_feat.append(fv)
            tr_y.append(int(liabilities[inn][effect]["label"]))
    out: dict[str, float | None] = {}
    for inn in held:
        fv = feature_vector(_cu(cmax, inn), _ki_by_gene(binding, inn), receptors)
        if fv is None or not tr_feat:
            out[inn] = None
        else:
            out[inn] = liability_predict(tr_feat, tr_y, [fv], l2=L2_LAMBDA)[0]
    return out


def build_dump() -> dict[str, Any]:
    """Assemble the held-out dump from whatever truth this checkout holds.

    A held-out drug appears in a liability only when its label is present in the
    checkout.  In a session checkout held-out labels are stripped, so the
    liability lists come out empty and every held-out drug is recorded as
    searched-and-empty; on the host they are populated."""
    drugs = {row["inn"]: row for row in load_v2_drugs()}
    train, heldout = v2_partition()
    binding = load_v2_binding()
    cmax = load_v2_human_cmax()
    liabilities = load_v2_liabilities()
    fp_targets = _all_targets(binding)
    train_list = list(train)
    held_list = list(heldout)

    liabilities_out: dict[str, Any] = {}
    searched_empty: dict[str, list[str]] = {}
    for effect in CANDIDATE_LIABILITIES:
        receptors = LIABILITY_RECEPTORS[effect]
        labelled = [
            inn for inn in held_list
            if liabilities.get(inn, {}).get(effect, {}).get("label") is not None
        ]
        if not labelled:
            searched_empty[effect] = sorted(
                inn for inn in held_list if inn not in liabilities
                or liabilities.get(inn, {}).get(effect, {}).get("label") is None
            )
            continue
        cand = _candidate_heldout_scores(effect, train_list, labelled, binding, cmax, liabilities)
        # Fit-dependent baselines use the train set only.
        train_labelled = {
            inn: int(liabilities[inn][effect]["label"])
            for inn in sorted(train_list)
            if liabilities.get(inn, {}).get(effect, {}).get("label") is not None
        }
        train_vectors = {inn: _fingerprint(binding.get(inn, {}), fp_targets) for inn in train_labelled}
        y = []
        base_strong, base_occ, base_nn, base_atc = [], [], [], []
        for inn in labelled:
            y.append(int(liabilities[inn][effect]["label"]))
            base_strong.append(_binding_score(binding.get(inn, {}), receptors))
            base_occ.append(occupancy_baseline(_cu(cmax, inn), _ki_by_gene(binding, inn), receptors))
            vec = _fingerprint(binding.get(inn, {}), fp_targets)
            nbrs = [
                (float(np.linalg.norm(vec - train_vectors[o])), o)
                for o in sorted(train_labelled)
                if vec is not None and train_vectors[o] is not None
            ]
            base_nn.append(float(train_labelled[min(nbrs, key=lambda p: (p[0], p[1]))[1]]) if nbrs else None)
            base_atc.append(_atc_class_prevalence(inn, train_labelled, drugs, set(train_labelled)))
        liabilities_out[effect] = {
            "drugs": labelled,
            "y": y,
            "candidate": [cand[inn] for inn in labelled],
            "baselines": {
                "occupancy_max": base_occ,
                "strongest_target_ki": base_strong,
                "one_nn_fingerprint": base_nn,
                "atc_class_prevalence": base_atc,
            },
        }

    return {
        "task": "P33.4",
        "candidate": "multi-target occupancy -> L2 logistic (candidate.liability_predict)",
        "l2_lambda": L2_LAMBDA,
        "host_eval": os.environ.get("NC_HOST_EVAL") == "1",
        "liabilities": liabilities_out,
        "searched_and_empty": searched_empty,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: python -m neurocascade.benchmark.v2.go_dump OUT.json\n")
        return 2
    out_path = argv[1]
    if os.environ.get("NC_HOST_EVAL") != "1":
        # Session tree: held-out truth is absent. Emit a valid empty-liabilities
        # dump naming every held-out drug and exit cleanly; the binding verdict
        # is the host run.
        _, heldout = v2_partition()
        dump = {
            "task": "P33.4",
            "host_eval": False,
            "status": "searched-and-empty",
            "reason": "held-out truth absent in session checkout; NC_HOST_EVAL not set",
            "liabilities": {},
            "heldout_drugs": sorted(heldout),
        }
        with open(out_path, "w") as fh:
            json.dump(dump, fh, indent=1)
        sys.stdout.write(
            "NC_HOST_EVAL not set: wrote empty session dump "
            f"({len(dump['heldout_drugs'])} held-out drugs searched-and-empty) to {out_path}\n"
        )
        return 0
    dump = build_dump()
    with open(out_path, "w") as fh:
        json.dump(dump, fh, indent=1)
    scored = {k: len(v["drugs"]) for k, v in dump["liabilities"].items()}
    sys.stdout.write(f"host_eval dump written to {out_path}; held-out n per liability: {scored}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
