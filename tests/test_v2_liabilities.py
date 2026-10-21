"""P33.2: placebo-subtracted CNS liability labels (scripts/v2_liabilities.py)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v2_liabilities as V  # noqa: E402

V2 = ROOT / "src/neurocascade/benchmark/v2"
OUT = V2 / "truth/liabilities.json"
RULES = V2 / "liability_rules.json"
LIABS = ["eps", "sedation", "weight_gain", "anticholinergic", "orthostatic_hypotension"]
BASES = {"placebo_subtracted", "absolute_fallback", "absent"}


@pytest.fixture(scope="module")
def data():
    return json.loads(OUT.read_text())


def _walk_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_keys(v)


def test_rules_hash_matches_committed_rules(data):
    assert data["rules_sha256"] == hashlib.sha256(RULES.read_bytes()).hexdigest()
    assert data["rules_file"].endswith("liability_rules.json")


def test_schema(data):
    assert data["liabilities"] == LIABS
    assert len(data["drugs"]) > 100
    inns = [r["inn"] for r in data["drugs"]]
    assert len(inns) == len(set(inns))
    for r in data["drugs"]:
        for L in LIABS:
            e = r[L]
            assert e["basis"] in BASES, (r["inn"], L)
            assert e.get("label") in (0, 1, None), (r["inn"], L)
            if e["basis"] != "absent":
                assert e["source"]["set_id"], (r["inn"], L)
            if e["basis"] == "absolute_fallback":
                assert e.get("flagged") is True, (r["inn"], L)
            if e.get("label") is not None and e["basis"] == "placebo_subtracted" and e.get("drug_pct") is not None:
                assert e["arms"], (r["inn"], L)


def test_percentages_in_range(data):
    for r in data["drugs"]:
        for L in LIABS:
            e = r[L]
            for k in ("drug_pct", "placebo_pct"):
                if e.get(k) is not None:
                    assert 0 <= e[k] <= 100, (r["inn"], L, k, e[k])
            for a in e.get("arms", []):
                for k in ("drug_pct", "placebo_pct"):
                    if a.get(k) is not None:
                        assert 0 <= a[k] <= 100, (r["inn"], L, a)


def test_placebo_arm_on_every_placebo_subtracted_row(data):
    for r in data["drugs"]:
        for L in LIABS:
            e = r[L]
            if e["basis"] == "placebo_subtracted":
                assert e.get("placebo_arm") and e["placebo_arm"].get("header"), (r["inn"], L)
                assert e["placebo_arm"]["header"][0].lower().count("placebo"), (r["inn"], L)
                for a in e.get("arms", []):
                    if a.get("drug_pct") is not None:  # absent_below_threshold arms carry no values
                        assert a.get("placebo_pct") is not None, (r["inn"], L, a)


def test_binary_rule_consistent(data):
    for r in data["drugs"]:
        for L in LIABS:
            e = r[L]
            if "label_withheld_A7" in (e.get("flags") or []):
                continue
            if e["basis"] == "placebo_subtracted" and e.get("risk_difference") is not None:
                want = int(e["risk_difference"] >= 5 and e["drug_pct"] >= 2 * e["placebo_pct"])
                assert e["label"] == want, (r["inn"], L)


def test_absent_terms_record_threshold(data):
    for r in data["drugs"]:
        for L in LIABS:
            e = r[L]
            if "absent_below_threshold" in (e.get("flags") or []):
                assert e.get("label") == (None if L == "weight_gain" else 0)
                assert "threshold_text" in e, (r["inn"], L)


def test_weight_gain_withheld(data):
    # amendment A7: labels under-call weight gain (olanzapine came out
    # negative), so the label is withheld while the numbers are kept.
    for r in data["drugs"]:
        e = r["weight_gain"]
        assert e["label"] is None, r["inn"]
        if e.get("risk_difference") is not None:
            assert "label_withheld_A7" in e["flags"], r["inn"]


def test_no_local_or_combo_source(data):
    """A7: aripiprazole is scored from its oral label, not the depot
    injection, and no row is sourced from a rejected fallback tier."""
    assert data_drug(data, "aripiprazole")["eps"]["label"] == 0
    for r in data["drugs"]:
        for L in LIABS:
            fl = r[L].get("flags") or []
            assert "local_route_label" not in fl and "multi_ingredient_label" not in fl, (r["inn"], L)


def data_drug(data, inn):
    return next(r for r in data["drugs"] if r["inn"] == inn)


def test_no_heldout_field(data):
    keys = set(_walk_keys(data))
    for bad in ("heldout", "held_out", "is_heldout", "pmid", "pmids", "split"):
        assert bad not in keys, bad
    assert "held" not in json.dumps([sorted(set(r)) for r in data["drugs"]]).lower()


# ------------------------------------------------------------------ parser units

def _cells(st, row_label, j):
    for r in st["data"]:
        if r[0]["text"].lower().startswith(row_label):
            return r[j]["text"]
    raise KeyError(row_label)


def test_parse_value_tokens():
    assert V.parse_value("12", False)["pct"] == 12.0
    lt = V.parse_value("<1", False)
    assert lt["censored"] and lt["pct"] == 0.5
    assert V.parse_value("-", False)["pct"] == 0.0
    assert V.parse_value("4*", False)["pct"] == 4.0
    assert V.parse_value("5 (2)", True)["pct"] == 2.0
    assert V.parse_value("Somnolence", False) is None


def test_rowspan_colspan_grid():
    html = ("<table><caption>Table 1 Adverse reactions (%)</caption>"
            "<thead><tr><th rowspan='2'>Reaction</th><th colspan='2'>Drug X</th><th rowspan='2'>Placebo (N=100)</th></tr>"
            "<tr><th>10 mg (N=100)</th><th>20 mg (N=100)</th></tr></thead>"
            "<tbody><tr><td>Somnolence</td><td>12</td><td>20</td><td>3</td></tr>"
            "<tr><td>Dry mouth</td><td>&lt;1</td><td>-</td><td>2*</td></tr></tbody></table>")
    tab = V.parse_table(html)
    assert tab["width"] == 4
    assert len(tab["grid"][0]) == len(tab["grid"][1]) == 4
    sts = V.structure(tab, ["drug x"])
    assert len(sts) == 1
    st = sts[0]
    assert [c["n"] for c in st["placebo"]] == [100]
    assert len(st["drug"]) == 2
    assert _cells(st, "somnolence", 2) == "20"


def test_column_groups_split():
    html = ("<table><thead><tr><th></th><th colspan='2'>Adults</th><th colspan='2'>Children</th></tr>"
            "<tr><th>Reaction</th><th>Drug X (N=50)</th><th>Placebo (N=50)</th><th>Drug X (N=40)</th><th>Placebo (N=40)</th></tr></thead>"
            "<tbody><tr><td>Somnolence</td><td>20</td><td>5</td><td>10</td><td>8</td></tr>"
            "<tr><td>Nausea</td><td>8</td><td>4</td><td>6</td><td>3</td></tr></tbody></table>")
    sts = V.structure(V.parse_table(html), ["drug x"])
    assert len(sts) == 2
    assert all(len(s["placebo"]) == 1 and len(s["drug"]) == 1 for s in sts)
    assert "Adults" in sts[0]["caption"] and "Children" in sts[1]["caption"]


def test_count_table_detected_and_percent_not():
    cnt = ("<table><caption>Adverse reactions in a placebo-controlled trial</caption>"
           "<thead><tr><th>Body system</th><th>Drug X (N=20)</th><th>Placebo (N=20)</th></tr></thead>"
           "<tbody><tr><td>Sedation</td><td>14</td><td>5</td></tr><tr><td>Dry mouth</td><td>5</td><td>1</td></tr>"
           "<tr><td>Headache</td><td>3</td><td>2</td></tr></tbody></table>")
    st = V.structure(V.parse_table(cnt), ["drug x"])[0]
    assert st["counts"]
    pct = cnt.replace("(N=20)", "(N=200)").replace("<td>14</td>", "<td>13</td>")
    st2 = V.structure(V.parse_table(pct), ["drug x"])[0]
    assert not st2["counts"]
