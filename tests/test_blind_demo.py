"""P32.3: blind demo of predict_compound on P28.1 held-out drugs.

Every test here holds in a session checkout without held-out truth and on the
host with it. Nothing compares a recomputed result with the committed
results/P32.3.json, because the host's held-out truth changes ``n_scored`` and
``host_pending``. That comparison failed the first P32.3 gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from neurocascade import blind_demo
from neurocascade.benchmark import load_split
from neurocascade.compound import PROVENANCE_LABEL
from neurocascade.report import audit_report

REPO = Path(__file__).resolve().parent.parent
CLEAN = {"unlinked_numbers": [], "rows_without_label": [], "broken_links": []}
QUANTITIES = ["target_engagement_dose", "occupancy", "clearance", "vss",
              "liability:sedation", "liability:anticholinergic", "liability:eps",
              "liability:orthostatic_hypotension"]


@pytest.fixture(scope="module")
def demo():
    return blind_demo.run_blind_demo()


def test_selection_is_five_runnable_heldout_drugs_by_a_truth_blind_rule(demo, monkeypatch):
    heldout = set(load_split()["heldout"])
    assert len(demo["demo_drugs"]) == blind_demo.N_DEMO == 5
    assert set(demo["runnable_heldout"]) <= heldout
    assert demo["demo_drugs"] == sorted(demo["runnable_heldout"])[:5]
    assert set(demo["runnable_heldout"]) | set(demo["refused_heldout"]) == heldout
    # The rule reads no truth: hiding every truth record changes nothing.
    def no_truth(inn):
        raise KeyError(inn)
    for target in ("neurocascade.benchmark.load_truth_record",
                   "neurocascade.compound.load_truth_record",
                   "neurocascade.blind_demo.load_truth_record"):
        monkeypatch.setattr(target, no_truth)
    assert blind_demo.demo_drugs() == demo["demo_drugs"]


def test_every_row_and_quantity_carries_the_label(demo):
    assert demo["label"] == PROVENANCE_LABEL
    for row in demo["demo"]["rows"]:
        assert row["label"] == PROVENANCE_LABEL
        assert list(row["quantities"]) == QUANTITIES
        for q in row["quantities"].values():
            assert q["label"] == PROVENANCE_LABEL
            assert q["predicted"]["provenance"]["label"] == PROVENANCE_LABEL


def test_dose_and_occupancy_are_shown_but_never_scored(demo):
    """predict_compound has no dose or occupancy predictor (P29.3/P29.4 empty)."""
    for row in demo["demo"]["rows"]:
        for name in ("target_engagement_dose", "occupancy"):
            q = row["quantities"][name]
            assert q["predicted"]["status"] == "pending"
            assert "point" not in q["predicted"]
            assert q["comparison"]["scored"] is False
            assert q["actual"]["status"] in {"observed", "host-pending", "searched-and-empty"}


def test_blind_inputs_only(demo):
    for row in demo["demo"]["rows"]:
        assert row["p28_1_split"] == "heldout"
        assert "no human exposure" in row["inputs"]
        for effect in blind_demo.ALL_LIABILITIES:
            prov = row["quantities"][f"liability:{effect}"]["predicted"]["provenance"]
            assert prov.get("exposure_source") is None


def test_truth_absent_path_uses_v2_truth_and_names_host_pending(monkeypatch):
    monkeypatch.setattr(blind_demo, "_p28_4", lambda inn: None)
    row = blind_demo.demo_row("alprazolam")
    assert row["p28_4_truth_present"] is False
    cl = row["quantities"]["clearance"]
    assert cl["predicted"]["provenance"]["band"] == "human_cl_ivive"
    assert cl["actual"]["value"] == pytest.approx(0.74)
    assert cl["actual"]["truth_source"].startswith("v2 human PK truth")
    assert cl["comparison"]["fold_error"] == pytest.approx(2.187 / 0.74, rel=1e-3)
    assert cl["comparison"]["covered"] == {"0.80": True, "0.95": True}
    assert cl["comparison"]["in_sample"] is False
    # The train-mean Vss averages alprazolam's own v2 Vss: in-sample.
    assert row["quantities"]["vss"]["comparison"]["in_sample"] is True
    sed = row["quantities"]["liability:sedation"]
    assert sed["actual"]["v2_label"] == 0
    assert sed["comparison"]["set_size"] == {"0.80": 2, "0.95": 2}
    assert sed["comparison"]["informative"] == {"0.80": False, "0.95": False}
    assert row["quantities"]["occupancy"]["actual"]["status"] == "host-pending"
    assert row["quantities"]["target_engagement_dose"]["actual"]["label_text"].startswith(
        "Recommended starting oral dosage is 0.5 mg")
    assert blind_demo.summarise([row])["host_pending"] == ["alprazolam"]


def test_truth_present_path_scores_against_p28_4(monkeypatch):
    record = {
        "dose": {"label": [{"indication": "panic disorder", "low": 1, "high": 10,
                            "units": "mg/day", "route": "oral"}]},
        "pet": [{"target": "TSPO", "region": "x", "dose_mg_per_day": 2, "occupancy_percent": 50,
                 "statistic": "mean", "first_author": "Test", "year": 2000, "pmid": "1"}],
        "effects": {"label": {"sedation": 1, "orthostasis": 0}},
    }
    monkeypatch.setattr(blind_demo, "_p28_4", lambda inn: record)
    monkeypatch.setattr(blind_demo._pk, "_human_truth",
                        lambda inn, p: (2.0, ("CHEMBL1",)) if p == "cl" else None)
    row = blind_demo.demo_row("alprazolam")
    assert row["p28_4_truth_present"] is True
    cl = row["quantities"]["clearance"]
    assert cl["actual"] == {"value": 2.0, "unit": "mL/min/kg",
                            "truth_source": "P28.4 IV compilation", "refs": ["CHEMBL1"]}
    assert cl["comparison"]["fold_error"] == pytest.approx(2.187 / 2.0, rel=1e-3)
    assert row["quantities"]["target_engagement_dose"]["actual"]["label_doses"][0]["high"] == 10
    assert row["quantities"]["occupancy"]["actual"]["pet"][0]["occupancy_percent"] == 50
    assert row["quantities"]["liability:sedation"]["actual"]["p28_4_label"] == 1
    assert row["quantities"]["liability:orthostatic_hypotension"]["actual"]["p28_4_label"] == 0
    assert blind_demo.summarise([row])["host_pending"] == []


def _row(inn, comparisons):
    return {"inn": inn, "p28_4_truth_present": False, "quantities": {
        name: {"predicted": {"provenance": {"compound_specific": specific}}, "comparison": c}
        for name, (c, specific) in comparisons.items()}}


def test_summary_excludes_in_sample_comparisons():
    hit = {"scored": True, "in_sample": False, "fold_error": 2.0,
           "covered": {"0.80": True, "0.95": True}}
    rows = [
        _row("a", {"clearance": (hit, True)}),
        _row("b", {"clearance": ({**hit, "in_sample": True}, False)}),
        _row("c", {"clearance": ({"scored": False, "reason": "no truth"}, True)}),
        _row("d", {"clearance": ({**hit, "fold_error": 8.0,
                                  "covered": {"0.80": False, "0.95": True}}, False)}),
    ]
    s = blind_demo.summarise(rows)
    assert s["n_scored"] == 2 and s["scored_drugs"] == ["a", "d"]
    assert s["n_scored_compound_specific"] == 1
    cl = s["per_quantity"]["clearance"]
    assert cl["n"] == 2 and cl["n_in_sample"] == 1
    assert cl["gmfe"] == pytest.approx(4.0)
    assert cl["covered"] == {"0.80": 1, "0.95": 2}
    assert cl["not_scored"] == {"c": "no truth"}
    assert s["host_pending"] == ["a", "b", "c", "d"]


def test_demo_html_passes_the_report_audit_and_labels_every_row(demo):
    text = blind_demo.render_html(demo)
    assert audit_report(text) == CLEAN
    rows = re.findall(r"<tr>(.*?)</tr>", text, flags=re.S)
    data_rows = [r for r in rows if "<th>" not in r]
    assert len(data_rows) == 5 * len(QUANTITIES)
    label = f'<span class="nv">{PROVENANCE_LABEL}</span>'
    assert all(label in r for r in data_rows)


def test_markdown_table_labels_every_row(demo):
    lines = [ln for ln in blind_demo.render_markdown(demo).splitlines()
             if ln.startswith("| ") and not ln.startswith("| drug")]
    assert len(lines) == 5 * len(QUANTITIES)
    assert all(ln.endswith(f"| {PROVENANCE_LABEL} |") for ln in lines)


def test_committed_artifacts_are_consistent_with_the_rule():
    """Checks what does not depend on held-out truth, never the scores."""
    result = json.loads((REPO / "results" / "P32.3.json").read_text())
    assert result["task"] == "P32.3"
    assert result["label"] == PROVENANCE_LABEL
    assert result["n_scored"] >= 1
    assert result["demo_drugs"] == blind_demo.demo_drugs()
    table = (REPO / "results" / "P32.3_blind_demo.md").read_text()
    rows = [ln for ln in table.splitlines() if ln.startswith("| ") and not ln.startswith("| drug")]
    assert len(rows) == 5 * len(QUANTITIES)
    assert all(ln.endswith(f"| {PROVENANCE_LABEL} |") for ln in rows)
    for inn in result["demo_drugs"]:
        assert any(ln.startswith(f"| {inn}") for ln in rows)
