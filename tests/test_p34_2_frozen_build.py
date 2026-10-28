"""P34.2: the frozen P34 data-construction build.

The sha256 of every rule/procedure file the fresh held-out set (P34.4) must be
built with is pinned here and in results/P34.2.json; editing any of them fails
this file. No SmPC parser is frozen, so the fresh set is labelled openFDA-only,
by the same procedure as the train labels.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import v2_partition

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "P34.2.json"
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"

# Frozen 2026-10-28 by scripts/p34_2_freeze.py. Changing a frozen file breaks
# the P34 pre-registration (docs/prereg-p34-go.md): do not update a sha here
# to make this pass.
PINNED = {
    "src/neurocascade/benchmark/v2/liability_rules.json":
        "4f00d880761b03cae3ace37d9057fe8e3259da4521da008b27386fe8eb836028",
    "scripts/v2_liabilities.py":
        "37e273df0c5e7cdfe5c6b4eecc598e15d2951e46dae674a5ed3d2d555e2c92e2",
    "scripts/v2_label_index.py":
        "f1c8ab1bd54ca1448bc0ff0af2310fe876f2440443aaca905f5d198b85b399cc",
    "scripts/v2_cmax.py":
        "9dbdf79403edd6dd248906de63b0b0656473cfc04d201451bf029ac6d5771d9e",
    "scripts/v2_pk.py":
        "1a7821c68226c36a1a726e9300bb312913d3908b3df84797ed8bbf5d2191193a",
    "scripts/v2_fu_merge.py":
        "f2e0d85d1cbcf1efbb89963c0386417231da480c81431c37adbff3682b617442",
    "scripts/v2_fu_drugcentral_fetch.py":
        "6be9d8a560b62a38f641ce6ec9b410bf7c3bf769472a3784912353c0a4877e4d",
    "scripts/v2_fu_structure_verify.py":
        "c3017115d198c2f9cf65cee96b5bf344858d2ed46d68704fcfdcbd1b0f0b4515",
    "scripts/v2_fu_label_extract.py":
        "60cf00813dc8501e80f00451f164d44814bacae99d3d58112016566b987e7c34",
    "scripts/v2_binding.py":
        "885fe6f2dde030a5b864e3d3e772f870426c31613428af034a5c61dfbbdb4fef",
}

# The v2 truth files and universe as of the freeze: P34.2 may not change them.
TRUTH_AT_FREEZE = {
    "truth/human_cmax.json": "5240ad5f3337e6b639c8412a94622c09f8d636c088e90beaba0e347971e260dd",
    "truth/human_cmax_corrections.json": "2c4148ed6a4f0c40063089169ff370a359d6c20086b99a779283ab9cdf0703db",
    "truth/human_pk.json": "ee833ca588e6b3ca53600b31386d11c0a53488f5ba0d67d2dc7e74546a8905cc",
    "truth/human_pk_corrections.json": "ce9806d2c2a30d3bb60ac8bda3353008f7defda71e6ecb06e0e5341b9b862db7",
    "truth/liabilities.json": "9b9332062161976b8213307c89538cf6c36a52896d0916b5bf042fb0f6a8c4e2",
    "drugs.json": "c73a3cc8c534baa0cad0428239f924ace9cae6a511d17d6c83ee6add4a5a80ab",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text())


@pytest.mark.parametrize("rel", sorted(PINNED))
def test_frozen_file_sha_is_pinned(rel):
    assert sha(ROOT / rel) == PINNED[rel], f"{rel} changed after the P34.2 freeze"


def test_result_lists_exactly_the_pinned_files(result):
    assert {f["path"]: f["sha256"] for f in result["frozen_files"]} == PINNED


def test_brief_files_are_frozen():
    for rel in ("src/neurocascade/benchmark/v2/liability_rules.json", "scripts/v2_liabilities.py",
                "scripts/v2_cmax.py", "scripts/v2_binding.py", "scripts/v2_fu_merge.py",
                "scripts/v2_fu_drugcentral_fetch.py", "scripts/v2_fu_structure_verify.py",
                "scripts/v2_fu_label_extract.py"):
        assert rel in PINNED


def test_freeze_manifest_is_generated_by_the_script(result):
    sys.path.insert(0, str(ROOT / "scripts"))
    import p34_2_freeze
    assert p34_2_freeze.manifest() == result["frozen_files"]


def test_liability_truth_was_built_with_the_frozen_rules(result):
    truth = json.loads((V2 / "truth" / "liabilities.json").read_text())
    assert truth["rules_sha256"] == PINNED["src/neurocascade/benchmark/v2/liability_rules.json"]
    assert result["train_labels_built_with_frozen_rules"] is True


def test_no_smpc_parser_is_frozen_so_none_may_be_used(result):
    assert result["smpc_parser"]["frozen"] is False
    assert not any("smpc" in p.lower() for p in PINNED)
    # P34.2 froze no SmPC parser. The PR #49 parser may exist only under the
    # disclosed P34.4b amendment, byte-for-byte as PR #49 left it (commit 163fdc0).
    # That it is never used on the fresh set unless it passed validation is
    # checked in test_p34_4b_smpc_amendment.py::test_a_failed_parser_is_not_applied.
    if (ROOT / "scripts" / "v2_smpc.py").exists():
        assert (ROOT / "docs" / "prereg-p34-amendment-smpc.md").exists()
        assert sha(ROOT / "scripts" / "v2_smpc.py").startswith("85a86afe4a77bd38")
        assert sha(V2 / "smpc_rules.json").startswith("a1725ea59bfafa86")


@pytest.mark.parametrize("rel", sorted(TRUTH_AT_FREEZE))
def test_v2_truth_and_universe_unchanged(rel):
    assert sha(V2 / rel) == TRUTH_AT_FREEZE[rel]


def test_burned_split_unchanged():
    train, heldout = v2_partition()
    assert (len(train), len(heldout)) == (267, 67)
