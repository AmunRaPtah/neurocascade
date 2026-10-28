"""P34.3: the frozen P34.1 candidate.

Pins the sha256 of the candidate's code, params and the modules it runs
(results/P34.3.json), checks the frozen predict path imports no truth, and
reproduces fixed train-drug predictions from the frozen params.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import (
    load_v2_binding,
    load_v2_drugs,
    load_v2_human_cmax,
    load_v2_liabilities,
    v2_heldout,
    v2_partition,
)
from neurocascade.benchmark.v2.candidate import LIABILITY_RECEPTORS
from neurocascade.benchmark.v2.candidate_benchmark import _cu, _ki_by_gene
from neurocascade.benchmark.v2.candidate_v2_benchmark import score_liability
from neurocascade.benchmark.v2.chain_benchmark import (
    _all_targets,
    _atc_class_prevalence,
    _binding_score,
    auroc,
)
from neurocascade.benchmark.v2.frozen_candidate import affinity, load_params, predict_cohort

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "P34.3.json"
V2 = "src/neurocascade/benchmark/v2/"

# Frozen 2026-10-28 by scripts/p34_3_freeze.py, before truth_fresh/ existed.
# Changing one breaks the P34 pre-registration: do not update a sha to pass.
PINNED = {
    V2 + "frozen_candidate.py": "d94aaecab870",
    V2 + "frozen/p34_candidate.json": "0d17a0717a1f",
}

# Frozen-path sedation scores for train drugs, read from the full labelled
# train cohort (19 of 168 labelled train drugs have both views).
TRAIN_FIXTURE = {
    "aripiprazole": 17.0,
    "asenapine": 27.0,
    "cetirizine": 29.0,
    "citalopram": 10.0,
    "clomipramine": 26.0,
    "clozapine": 33.0,
}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def v2():
    return load_v2_binding(), load_v2_human_cmax(), load_v2_liabilities()


def _labelled_train(liab, effect):
    train, _ = v2_partition()
    return [i for i in sorted(train) if liab.get(i, {}).get(effect, {}).get("label") is not None]


# ------------------------------------------------------------------ freeze

@pytest.mark.parametrize("rel", sorted(PINNED))
def test_frozen_file_sha_is_pinned(rel, result):
    assert sha(rel).startswith(PINNED[rel]), f"{rel} changed after the P34.3 freeze"
    recorded = {result["code"]["path"]: result["code"]["sha256"],
                result["params"]["path"]: result["params"]["sha256"]}
    assert recorded[rel] == sha(rel)


def test_dependency_shas_match_the_freeze(result):
    for dep in result["depends"]:
        assert sha(dep["path"]) == dep["sha256"], f"{dep['path']} changed after the P34.3 freeze"


def test_fresh_truth_did_not_exist_at_freeze(result):
    assert result["fresh_truth_existed_at_freeze"] is False


def test_result_is_generated_by_the_script(result):
    sys.path.insert(0, str(ROOT / "scripts"))
    import p34_3_freeze
    assert result["code"]["path"] == p34_3_freeze.CODE
    assert [d["path"] for d in result["depends"]] == p34_3_freeze.DEPENDS


# ------------------------------------------------------------------ same family as P34.1

def test_frozen_family_is_the_p34_1_selection(result):
    p341 = json.loads((ROOT / "results" / "P34.1.json").read_text())
    params = load_params()
    assert params["family"] == result["family"] == "candidate_rank_combined"
    assert "candidate_rank_combined" in p341["liabilities"]["sedation"]["lodo"]
    assert result["p34_1_candidate"] == p341["candidate"]
    assert params["n_fitted_parameters"] == 0 and params["fitted_parameters"] == {}
    assert params["l2"] is None and result["l2"] is None


def test_frozen_receptor_map_is_the_candidate_map():
    params = load_params()
    assert {k: tuple(v) for k, v in params["liability_receptors"].items()} == LIABILITY_RECEPTORS


# ------------------------------------------------------------------ reproduces train

def test_reproduces_fixed_train_predictions(v2):
    binding, cmax, liab = v2
    inns = _labelled_train(liab, "sedation")
    assert not any(v2_heldout(i) for i in inns)
    scores = dict(zip(inns, predict_cohort("sedation", [_cu(cmax, i) for i in inns],
                                           [_ki_by_gene(binding, i) for i in inns])))
    assert sum(v is not None for v in scores.values()) == 19
    for inn, want in TRAIN_FIXTURE.items():
        assert scores[inn] == want


@pytest.mark.parametrize("effect", ["sedation", "anticholinergic"])
def test_frozen_path_reproduces_the_p34_1_train_auroc(effect, v2):
    binding, cmax, liab = v2
    drugs = {r["inn"]: r for r in load_v2_drugs()}
    train, _ = v2_partition()
    harness = score_liability(effect, list(train), binding, cmax, liab, drugs, _all_targets(binding))
    labelled = _labelled_train(liab, effect)
    y_all = {i: int(liab[i][effect]["label"]) for i in labelled}
    # The harness common set: both views defined and a train-LODO ATC prevalence
    # (the 1-NN baseline is defined wherever affinity is).
    receptors = LIABILITY_RECEPTORS[effect]
    inns = [i for i in labelled
            if _cu(cmax, i) is not None and affinity(_ki_by_gene(binding, i), receptors) is not None
            and _atc_class_prevalence(i, {o: y_all[o] for o in labelled if o != i}, drugs,
                                      set(labelled) - {i}) is not None]
    s = predict_cohort(effect, [_cu(cmax, i) for i in inns], [_ki_by_gene(binding, i) for i in inns])
    kept = [(y_all[i], v) for i, v in zip(inns, s) if v is not None]
    assert len(kept) == harness["common_n"]
    got = auroc([int(y) for y, _ in kept], [v for _, v in kept])
    assert got == pytest.approx(harness["lodo"]["candidate_rank_combined"]["auroc"])


def test_affinity_matches_the_strongest_target_ki_view(v2):
    binding, _, _ = v2
    for inn in list(binding)[:80]:
        for receptors in LIABILITY_RECEPTORS.values():
            assert affinity(_ki_by_gene(binding, inn), receptors) == _binding_score(binding[inn], receptors)


# ------------------------------------------------------------------ no truth on the predict path

@pytest.mark.parametrize("rel", ["frozen_candidate.py", "candidate_v2.py", "candidate.py"])
def test_predict_path_imports_no_truth(rel):
    tree = ast.parse((ROOT / V2 / rel).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    joined = " ".join(imported)
    for forbidden in ("load_v2_", "truth", "chain_benchmark", "candidate_benchmark", "go_dump"):
        assert forbidden not in joined, f"{rel} imports {forbidden!r}"
    src = (ROOT / V2 / rel).read_text()
    assert "truth_fresh" not in src and "V2_FRESH" not in src
