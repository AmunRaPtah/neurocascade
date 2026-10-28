"""P34.4b: the disclosed SmPC amendment (docs/prereg-p34-amendment-smpc.md).

Checks the train validation bar in results/P34.4b.json follows from the
recorded comparisons, every compared drug is a train drug (never v2 held-out,
never fresh), the parser is the PR #49 bytes, and a failed parser was not
applied to the fresh set.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
RESULT = ROOT / "results" / "P34.4b.json"
FRESH = V2 / "truth_fresh"

# PR #49, commit 163fdc0; byte-identical import under the amendment.
PARSER = {
    "scripts/v2_smpc.py": "85a86afe4a77bd38",
    "src/neurocascade/benchmark/v2/smpc_rules.json": "a1725ea59bfafa86",
}


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text())


def test_amendment_exists_and_is_cited(result):
    assert (ROOT / result["prereg"]).exists()
    assert "deviation" in result and "after" in result["deviation"]


def test_parser_is_the_pr49_bytes(result):
    recorded = {p["path"]: p["sha256"] for p in result["parser"]}
    for rel, prefix in PARSER.items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got.startswith(prefix) and recorded[rel] == got


def test_bar_follows_from_the_comparisons(result):
    sys.path.insert(0, str(ROOT / "scripts"))
    import p34_4b_validate
    assert p34_4b_validate.bar(result["comparisons"]) == result["bar"]
    assert (p34_4b_validate.MIN_POS, str(p34_4b_validate.SENS), str(p34_4b_validate.NEG_AGREE)) \
        == (3, "2/3", "9/10")
    assert result["parser_frozen"] is result["bar"]["pass"]


def test_every_compared_drug_is_a_train_drug(result):
    sys.path.insert(0, str(ROOT / "pipeline"))
    import guards
    v2 = {d["inn"] for d in json.loads((V2 / "drugs.json").read_text())["drugs"]}
    inns = ({c["inn"] for c in result["comparisons"]} | set(result["train_validation"]["compared_drugs"])
            | {r["inn"] for r in result["fetch_failed_retried_singly"]})
    for inn in inns:
        assert inn in v2 and not guards.v2_heldout(inn), inn
    if (FRESH / "pool.json").exists():
        fresh = {m["inn"] for m in json.loads((FRESH / "pool.json").read_text())["members"]}
        assert not inns & fresh


def test_a_failed_parser_is_not_applied(result):
    if not result["bar"]["pass"]:
        assert result["result"] == "EMPTY"
        assert result["applied_to_fresh"] is False and result["parser_frozen"] is False
        if (FRESH / "liabilities.json").exists():
            liab = json.loads((FRESH / "liabilities.json").read_text())
            for d in liab["drugs"]:
                for L in liab["liabilities"]:
                    assert "smpc" not in d[L].get("flags", []), (d["inn"], L)
