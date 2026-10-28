"""P34.2: the frozen P34 data-construction build and the SmPC parser.

The sha256 of every rule/procedure file the fresh held-out set (P34.4) must be
built with is pinned here and in results/P34.2.json; editing any of them fails
this file. The SmPC parser (scripts/v2_smpc.py) is tested offline on trimmed
emc SmPC pages of train drugs (tests/fixtures/smpc/), with the numbers read by
hand from those pages.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lxml.html
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v2_liabilities as V  # noqa: E402
import v2_smpc as S  # noqa: E402

from neurocascade.benchmark.v2 import v2_heldout, v2_partition  # noqa: E402

RESULT = ROOT / "results" / "P34.2.json"
FIX = ROOT / "tests" / "fixtures" / "smpc"

# Frozen 2026-10-28 by scripts/p34_2_freeze.py. Changing a frozen file breaks
# the P34 pre-registration (docs/prereg-p34-go.md): do not update a sha here
# to make this pass.
PINNED = {
    "src/neurocascade/benchmark/v2/liability_rules.json":
        "4f00d880761b03cae3ace37d9057fe8e3259da4521da008b27386fe8eb836028",
    "scripts/v2_liabilities.py":
        "37e273df0c5e7cdfe5c6b4eecc598e15d2951e46dae674a5ed3d2d555e2c92e2",
    "src/neurocascade/benchmark/v2/smpc_rules.json":
        "a1725ea59bfafa86d2c60bb84497b026c4c4810f8ce92d4109ac3cc6ccd759ce",
    "scripts/v2_smpc.py":
        "85a86afe4a77bd38950db512b6928d55d964a364f56805d434a37ef4ceca0c28",
    "scripts/v2_label_index.py":
        "f1c8ab1bd54ca1448bc0ff0af2310fe876f2440443aaca905f5d198b85b399cc",
    "scripts/v2_cmax.py":
        "9dbdf79403edd6dd248906de63b0b0656473cfc04d201451bf029ac6d5771d9e",
    "scripts/v2_pk.py":
        "1a7821c68226c36a1a726e9300bb312913d3908b3df84797ed8bbf5d2191193a",
    "scripts/v2_binding.py":
        "836fa9e84fafc7508fbfc458b0fa76480e11d112087b3386d026ee3a89cf0102",
}


def sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def rules():
    return V.load_rules()[0]


@pytest.fixture(scope="module")
def smpc_rules():
    return S.load_smpc_rules()[0]


def record(pid: str, inn: str, rules, smpc_rules):
    meta = S.product_meta(pid, (FIX / f"{pid}.html").read_text())
    return S.label_from_product(inn, meta, rules, smpc_rules)


# ------------------------------------------------------------------ freeze

@pytest.mark.parametrize("rel", sorted(PINNED))
def test_frozen_file_sha_is_pinned(rel):
    assert sha(rel) == PINNED[rel], f"{rel} changed after the P34.2 freeze"


def test_result_lists_exactly_the_pinned_files(result):
    assert {f["path"]: f["sha256"] for f in result["frozen_files"]} == PINNED


def test_brief_files_are_frozen():
    for rel in ("src/neurocascade/benchmark/v2/liability_rules.json", "scripts/v2_liabilities.py",
                "scripts/v2_cmax.py", "scripts/v2_binding.py"):
        assert rel in PINNED


def test_liability_truth_was_built_with_the_frozen_rules():
    truth = json.loads((ROOT / "src/neurocascade/benchmark/v2/truth/liabilities.json").read_text())
    assert truth["rules_sha256"] == PINNED["src/neurocascade/benchmark/v2/liability_rules.json"]


def test_freeze_manifest_is_generated_by_the_script(result):
    sys.path.insert(0, str(ROOT / "scripts"))
    import p34_2_freeze
    assert p34_2_freeze.manifest() == result["frozen_files"]


# ------------------------------------------------------------------ validation record

def test_validation_scored_train_drugs_only(result):
    val = result["smpc_parser"]["train_validation"]
    train, _ = v2_partition()
    assert val["n_train"] == len(train)
    assert set(val["compared_drugs"]) <= set(train)
    assert not any(v2_heldout(d["inn"]) for d in val["drugs"])
    assert result["n_scored"] == val["n_compared_drugs"] == len(val["compared_drugs"]) > 0


def test_validation_numbers(result):
    val = result["smpc_parser"]["train_validation"]
    assert val["compared_drugs"] == ["cetirizine", "levocetirizine", "rivastigmine"]
    assert (val["n_compared_labels"], val["n_agree"], val["n_positive_fda"]) == (12, 12, 0)
    assert val["n_fetch_failed"] == 0


def test_compare_skips_null_and_non_placebo_labels():
    fda = {L: {"basis": "placebo_subtracted", "label": 0} for L in V.LIABS}
    fda["weight_gain"]["label"] = None
    fda["eps"] = {"basis": "absolute_fallback", "label": 1}
    sm = {L: {"label": 0} for L in V.LIABS}
    sm["sedation"]["label"] = 1
    rows = S.compare(fda, sm)
    assert [r["liability"] for r in rows] == ["sedation", "anticholinergic", "orthostatic_hypotension"]
    assert [r["agree"] for r in rows] == [False, True, True]


# ------------------------------------------------------------------ parser on train SmPCs

def test_cetirizine_stacked_rows(rules, smpc_rules):
    # Zirtek (emc 175) stacks 'Psychiatric disorders' / 'Somnolence' in one cell
    # with a malformed '<p />'; the adult table reads somnolence 9.63 % vs 5.00 %.
    r = record("175", "cetirizine", rules, smpc_rules)
    s = r["sedation"]
    assert (s["drug_pct"], s["placebo_pct"], s["label"], s["terms"]) == (9.63, 5.0, 0, ["somnolence"])
    assert s["arms"][0]["n_drug"] == 3260 and s["arms"][0]["n_placebo"] == 3061
    a = r["anticholinergic"]
    assert (a["drug_pct"], a["placebo_pct"], a["label"]) == (2.09, 0.82, 0)
    assert s["source"]["emc_product_id"] == "175" and s["source"]["source"] == S.SOURCE
    assert "smpc" in s["flags"] and s["basis"] == "placebo_subtracted"
    # the paediatric table is not primary; its '1. 4 %' placebo cell is not guessed
    (other,) = s["other_tables"]
    assert other["population"].startswith("Paediatric population")
    assert other["flags"] == ["extraction_failed"]


def test_levocetirizine_counts_and_percent(rules, smpc_rules):
    r = record("348", "levocetirizine", rules, smpc_rules)
    s = r["sedation"]
    assert (s["drug_pct"], s["placebo_pct"], s["label"]) == (5.2, 1.4, 0)
    assert s["arms"][0]["n_drug"] == 935 and s["arms"][0]["n_placebo"] == 771
    a = r["anticholinergic"]
    assert (a["drug_pct"], a["placebo_pct"], a["terms"]) == (2.6, 1.6, ["mouth dry"])
    assert len(s["other_tables"]) == 2
    assert r["weight_gain"]["label"] is None  # openFDA amendment A7 applies unchanged


def test_bilastine_frequency_column_and_pooled_arm(rules, smpc_rules):
    # Ilaxten (emc 4551): 'Frequency | Adverse reaction | 20 mg | All doses | Placebo'
    r = record("4551", "bilastine", rules, smpc_rules)
    s = r["sedation"]
    assert [a["dose"] for a in s["arms"]] == ["Bilastine 20 mg N = 1697", "All Bilastine Doses N = 2525"]
    assert [a["drug_pct"] for a in s["arms"]] == [3.06, 3.25]
    assert (s["drug_pct"], s["placebo_pct"], s["pool_method"], s["label"]) == (3.25, 2.86, "pooled_column", 0)
    assert r["eps"]["label"] == 0 and "absent_below_threshold" in r["eps"]["flags"]


def test_rivastigmine_grouped_eps_terms(rules, smpc_rules):
    r = record("2059", "rivastigmine", rules, smpc_rules)
    e = r["eps"]
    assert (e["drug_pct"], e["placebo_pct"], e["label"]) == (2.2, 0.6, 0)


def test_laboratory_table_is_not_an_adverse_reaction_table(rules, smpc_rules):
    # amendment SA1: Nexavar's only placebo-controlled table is laboratory values
    assert record("226", "sorafenib", rules, smpc_rules) is None


def test_frequency_category_table_mentioning_placebo_is_not_placebo_controlled(rules, smpc_rules):
    assert record("2835", "sertraline", rules, smpc_rules) is None


# ------------------------------------------------------------------ normalisation units

def _table(html):
    return lxml.html.fromstring(html)


def test_split_stacked_rows_with_heading_line():
    t = _table("<table><tr><td><p>Nervous system disorders</p><p>Dizziness</p><p>Headache</p></td>"
               "<td><p>1.1 %</p><p>7.4 %</p></td><td><p>0.9 %</p><p>8.0 %</p></td></tr></table>")
    S.split_stacked_rows(t)
    rows = [[V.clean(c.text_content()) for c in tr] for tr in t.iter("tr")]
    assert rows == [["Nervous system disorders", "", ""], ["Dizziness", "1.1 %", "0.9 %"],
                    ["Headache", "7.4 %", "8.0 %"]]


def test_split_leaves_mismatched_and_header_rows():
    t = _table("<table><tr><th><p>Term</p><p>(WHO-ART)</p></th><th><p>Drug</p><p>(n=10)</p></th></tr>"
               "<tr><td><p>A</p><p>B</p><p>C</p><p>D</p></td><td><p>1 %</p><p>2 %</p></td></tr></table>")
    before = lxml.html.tostring(t)
    S.split_stacked_rows(t)
    assert lxml.html.tostring(t) == before


def test_no_absolute_fallback_for_smpc(rules, smpc_rules):
    html = ('<html><head><script type="application/ld+json">{"@type":"Drug","name":"X 1 mg tablets",'
            '"activeIngredient":["xdrug"]}</script></head><body>'
            '<details><summary id="FORM">3. Pharmaceutical form</summary><p>Tablet</p></details>'
            '<details><summary id="UNDESIRABLE_EFFECTS">4.8</summary><table>'
            '<tr><th>Term</th><th>Xdrug (N=100)</th></tr><tr><td>Somnolence</td><td>40 %</td></tr>'
            '</table></details></body></html>')
    assert S.label_from_product("xdrug", S.product_meta("1", html), rules, smpc_rules) is None


def test_product_selection(rules, smpc_rules):
    def page(name, ai, form):
        return ('<html><head><script type="application/ld+json">' + json.dumps(
            {"@type": "Drug", "name": name, "activeIngredient": ai}) + '</script></head><body>'
            f'<details><summary id="FORM">3.</summary><p>{form}</p></details></body></html>')
    pages = {
        f"{S.EMC}/search?q=xdrug": ''.join(f'<a href="/emc/product/{p}/smpc">' for p in (9, 7, 5, 3, 7)),
        f"{S.EMC}/product/9/smpc": page("X tabs", ["xdrug hydrochloride"], "Film-coated tablet"),
        f"{S.EMC}/product/7/smpc": page("X inj", ["xdrug"], "Solution for injection"),
        f"{S.EMC}/product/5/smpc": page("X+Y", ["xdrug", "ydrug"], "Tablet"),
        f"{S.EMC}/product/3/smpc": page("X eye", ["xdrug"], "Eye drops, solution"),
    }
    out = S.smpc_record("xdrug", rules, smpc_rules, get=pages.__getitem__)
    assert out["products_seen"] == 4
    assert out["products_eligible"] == ["9", "7"]  # oral first, then by id; combo and eye drops out
    assert out["basis"] == "absent" and out["record"] is None
    assert out["reason"] == "no placebo-controlled table in section 4.8 of any eligible product"
