"""P35.2: SIDER train labels, the PubChem CID map and the train universe."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import load_v2_drugs
from neurocascade.benchmark.v2 import sider_labels as sl

ROOT = Path(__file__).resolve().parents[1]
PREREG = (ROOT / "docs" / "prereg-p35-go.md").read_text()
TRAIN_FILES = (sl.CID_MAP, sl.TRAIN_ROWS, sl.TRAIN_LABELS, sl.PT_LISTING, sl.UNIVERSE)


def _row(name, upper, placebo=False, desc="x"):
    return {"side_effect_name": name, "frequency_upper": upper, "placebo": placebo,
            "frequency_desc": desc}


# ------------------------------------------------------------- prereg fidelity

def test_split_rule_is_sha256_mod_10_in_0_1():
    assert "`sha256(inn) mod 10`" in PREREG and "{0, 1}" in PREREG
    for d in load_v2_drugs():
        inn = d["inn"]
        want = int(hashlib.sha256(inn.encode()).hexdigest(), 16) % 10 in {0, 1}
        assert sl.p35_heldout(inn) is want
    assert sl.p35_split_digit("haloperidol") == int(hashlib.sha256(b"haloperidol").hexdigest(), 16) % 10


def test_pt_liability_map_equals_prereg():
    block = PREREG.split("**MedDRA PT -> liability map (fixed now):**")[1].split("Matching is on")[0]
    parsed = {}
    for liab, pts in re.findall(r"- (\w+): ([^\n]+(?:\n {4}[^\n-][^\n]*)*)", block):
        parsed[liab] = tuple(p.strip() for p in " ".join(pts.split()).split(","))
    assert parsed == sl.LIABILITY_PTS
    assert sl.THRESHOLD == Decimal("0.01")


# --------------------------------------------------------- held-out isolation

def test_no_heldout_inn_has_a_label_row_or_universe_row():
    labels = sl.load_train_labels()
    rows = sl.load_train_rows()["drugs"]
    universe = [u["inn"] for u in sl.load_universe()]
    for inns in (labels, rows, universe):
        assert inns and not [i for i in inns if sl.p35_heldout(i)]
    train = {d["inn"] for d in load_v2_drugs() if not sl.p35_heldout(d["inn"])}
    assert set(labels) == set(rows) == set(universe) == train


def test_cid_map_covers_every_v2_drug_but_matches_sider_for_train_only():
    cmap = sl.load_cid_map()
    assert set(cmap) == {d["inn"] for d in load_v2_drugs()}
    for inn, row in cmap.items():
        assert row["pubchem_cids"], inn
        assert ("sider" in row) is (not sl.p35_heldout(inn)), inn


def test_label_builder_refuses_a_heldout_drug():
    heldout = next(d["inn"] for d in load_v2_drugs() if sl.p35_heldout(d["inn"]))
    with pytest.raises(ValueError, match="held-out"):
        sl.build_train_labels({heldout: {"has_record": True, "mapped_rows": []}})


def test_selection_path_reads_only_the_train_label_file():
    # The only label file in the tree is the train one; nothing else under
    # sider/ could carry a held-out label.
    assert sorted(p.name for p in sl.SIDER_DIR.iterdir()) == sorted(p.name for p in TRAIN_FILES)
    src = (ROOT / "src" / "neurocascade" / "benchmark" / "v2" / "sider_labels.py").read_text()
    assert "heldout.json" not in src and "labels_heldout" not in src


# ------------------------------------------------------ committed build checks

def test_committed_labels_rebuild_from_the_frozen_extract():
    rows = sl.load_train_rows()
    assert rows["sider"]["sha256"] == sl.SIDER_FILES
    assert sl.build_train_labels(rows["drugs"]) == sl.load_train_labels()


def test_committed_counts_match_labels_and_universe():
    import json
    doc = json.loads(sl.TRAIN_LABELS.read_text())
    labels = doc["drugs"]
    members = [u["inn"] for u in sl.load_universe() if u["in_universe"] is True]
    assert doc["counts"]["universe"] == sl.label_counts(labels, members)
    uni = json.loads(sl.UNIVERSE.read_text())
    assert uni["n_in_universe"] == len(members) == 53
    assert uni["n_pending_host"] == 19
    assert {k: (v["positive"], v["negative"]) for k, v in doc["counts"]["universe"].items()} == {
        "sedation": (23, 30), "anticholinergic": (32, 21), "eps": (18, 35),
        "orthostatic_hypotension": (14, 39), "seizure": (6, 47)}


def test_universe_rows_follow_the_three_criteria():
    for u in sl.load_universe():
        ok = bool(u["anchor_ki"]) and u["sider_record"]
        if u["in_universe"] is True:
            assert ok and u["cmax"] == "value"
        elif u["in_universe"] == "pending_host":
            assert ok and u["cmax"] == "hidden_in_session_tree"
        else:
            assert not ok or u["cmax"] in ("searched_and_empty", "absent")


def test_pt_listing_reports_map_terms_missing_from_sider():
    import json
    listing = json.loads(sl.PT_LISTING.read_text())
    missing = sorted(e["pt"] for pts in listing["map"].values() for e in pts if not e["in_sider_vocabulary"])
    assert missing == ["Blurred vision", "Postural dizziness", "Seizure"]
    assert "Vision blurred" in listing["unmatched"] and "Dizziness postural" in listing["unmatched"]


def test_enantiomer_partners_do_not_share_sider_rows():
    cmap = sl.load_cid_map()
    esc = cmap["escitalopram"]["sider"]
    assert esc["stereo"] == [146570] and 2771 in [r["cid"] for r in esc["stereo_rejected"]]
    assert cmap["esketamine"]["sider"]["stereo"] == []
    assert sl.load_train_labels()["esketamine"]["sedation"] == {"label": "excluded"}


# --------------------------------------------------------------- rule units

def test_placebo_subtraction_uses_exact_decimals():
    lab = sl.label_drug([_row("Somnolence", "0.03"), _row("Somnolence", "0.02", placebo=True)], True)
    assert lab["sedation"]["label"] == 1
    assert lab["sedation"]["pts"][0]["adjusted"] == "0.01"
    assert lab["sedation"]["positive_only_at_boundary"] is True


def test_placebo_subtraction_can_make_a_negative():
    lab = sl.label_drug([_row("Dry mouth", "0.05"), _row("Dry mouth", "0.045", placebo=True)], True)
    assert lab["anticholinergic"]["label"] == 0
    assert lab["anticholinergic"]["pts"][0]["placebo_uncontrolled"] is False


def test_max_minus_max_aggregation_and_uncontrolled_flag():
    lab = sl.label_drug([_row("Tremor", "0.004"), _row("Tremor", "0.2"), _row("Akathisia", "0.02"),
                         _row("Akathisia", "0.15", placebo=True)], True, listed_pts=["Dystonia"],
                        n_freq_rows=4)
    eps = lab["eps"]
    tremor = next(p for p in eps["pts"] if p["pt"] == "Tremor")
    assert tremor["adjusted"] == "0.2" and tremor["placebo_uncontrolled"]
    akat = next(p for p in eps["pts"] if p["pt"] == "Akathisia")
    assert Decimal(akat["adjusted"]) == Decimal("-0.13") and not akat["meets_threshold"]
    assert eps["label"] == 1 and eps["positive_only_placebo_uncontrolled"] is True
    assert eps["listed_without_frequency"] == ["Dystonia"] and eps["no_frequency_rows"] is False


def test_record_without_mapped_pt_is_negative_and_no_record_is_excluded():
    neg = sl.label_drug([], True, n_freq_rows=0)
    assert all(v["label"] == 0 and v["no_frequency_rows"] for v in neg.values())
    assert sl.label_drug([], False) == {liab: {"label": "excluded"} for liab in sl.LIABILITY_PTS}


def test_matching_is_case_folded_exact():
    lab = sl.label_drug([_row("GRAND MAL CONVULSION", "0.02")], True)
    assert lab["seizure"]["label"] == 1
    with pytest.raises(ValueError):
        sl.label_drug([_row("Vision blurred", "0.5")], True)


def test_stitch_ids_and_stereo_matching():
    assert sl.stitch_cid("CID100002771") == 2771 and sl.stitch_cid("CID000146570") == 146570
    with pytest.raises(ValueError):
        sl.stitch_cid("CID2771")
    flat = {2771: [2771, 146570]}
    esc = sl.match_stereo("escitalopram", [146570], [2771], flat, {})
    assert esc["flat"] == [2771] and esc["stereo"] == [146570] and esc["stereo_rejected"] == [2771]
    cit = sl.match_stereo("citalopram", [2771], [], flat, {})
    assert cit["stereo"] == [2771]
    # Salt record linked by parent CID.
    salt = sl.match_stereo("sertraline", [68617], [5203], {5203: [63009]}, {63009: [68617]})
    assert salt["stereo"] == [63009]
    # By name: INN + salt words accepted, unless a synonym names another v2 drug.
    syn = {441277: ["Talwin 50", "Pentazocine HCl"]}
    assert sl.match_stereo("pentazocine", [4736], [], {4736: [441277]}, {}, syn)["stereo_by_name"] == {
        "441277": "Pentazocine HCl"}
    veto = {154100: ["Dexmethylphenidate hydrochloride", "methylphenidate hydrochloride"]}
    m = sl.match_stereo("methylphenidate", [4158], [], {4158: [4158, 154100]}, {}, veto,
                        ["methylphenidate", "dexmethylphenidate"])
    assert m["stereo"] == [4158] and m["stereo_rejected"] == [154100]


def test_universe_row_rule():
    anchors = ["DRD2", "HRH1"]
    b = {"targets": {"HRH1": {}, "KCNH2": {}}}
    assert sl.universe_row("x", anchors, b, True, {"status": "value"}, False)["in_universe"] is True
    assert sl.universe_row("x", anchors, b, True, None, True)["in_universe"] == "pending_host"
    assert sl.universe_row("x", anchors, b, False, {"status": "value"}, False)["in_universe"] is False
    assert sl.universe_row("x", anchors, {"targets": {"KCNH2": {}}}, True, {"status": "value"},
                           False)["in_universe"] is False
    assert sl.universe_row("x", anchors, b, True, {"status": "searched_and_empty"}, False)["in_universe"] is False


def test_concordance_table_and_kappa():
    sider = {"a": {"sedation": {"label": 1}}, "b": {"sedation": {"label": 0}},
             "c": {"sedation": {"label": 1}}, "d": {"sedation": {"label": "excluded"}}}
    ref = {k: {"sedation": {"basis": "placebo_subtracted", "label": v}}
           for k, v in {"a": 1, "b": 0, "c": 0, "d": 1}.items()}
    ref["e"] = {"sedation": {"basis": "placebo_subtracted", "label": 1}}  # no SIDER row
    out = sl.concordance(sider, ref, ["sedation"])["sedation"]
    assert out["n"] == 3 and out["fda_has_label_sider_excluded"] == 1
    assert out["confusion"] == {"fda1_sider1": 1, "fda1_sider0": 0, "fda0_sider1": 1, "fda0_sider0": 1}
    assert out["sider_sensitivity_to_fda_positive"] == 1.0
    assert out["sider_positive_rate_on_fda_negative"] == 0.5
    # po = 2/3, pe = (1*2 + 2*1)/9 = 4/9, kappa = (2/3 - 4/9) / (5/9) = 0.4
    assert out["cohen_kappa"] == 0.4


def test_results_file_matches_a_fresh_concordance():
    import json
    res = json.loads((ROOT / "results" / "P35.2.json").read_text())
    from neurocascade.benchmark.v2 import v2_heldout
    # v2 held-out rows exist only in the host tree; the committed result excludes them.
    ref = {r["inn"]: r for r in json.loads((sl.SIDER_DIR.parent / "truth" / "liabilities.json")
                                         .read_text())["drugs"]
           if not sl.p35_heldout(r["inn"]) and not v2_heldout(r["inn"])}
    shared = ["sedation", "anticholinergic", "eps", "orthostatic_hypotension"]
    assert res["concordance_vs_v2_fda_placebo_subtracted"] == sl.concordance(
        sl.load_train_labels(), ref, shared)
    assert res["n_scored"] == 88
    assert res["train_counts"] == json.loads(sl.TRAIN_LABELS.read_text())["counts"]
