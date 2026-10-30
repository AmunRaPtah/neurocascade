"""P35.4: the held-out SIDER dump for the GO / NO-GO verdict.

The binding run is the host's (NC_HOST_EVAL=1, full v2 truth). Here: the dump
reads no held-out row without the flag and its empty dump makes go_verdict print
NO-GO once; the held-out rows file holds only P35 held-out drugs, no label, and
is stripped from session trees by guards.hide_heldout; the real build path runs
end to end on P35-train drugs standing in for held-out ones (their labels are
train data) and go_verdict judges its output; the candidate predict path
imports no held-out truth.
"""

from __future__ import annotations

import ast
import contextlib
import gzip
import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import go_dump_p35 as gd
from neurocascade.benchmark.v2 import load_v2_drugs
from neurocascade.benchmark.v2 import sider_labels as sl

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src/neurocascade/benchmark/v2"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verdict_lines(path: Path) -> list[str]:
    gv = _load("go_verdict", "pipeline/go_verdict.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gv.main(str(path)) == 0
    return [l for l in buf.getvalue().splitlines() if l.startswith("NC_GO_VERDICT:")]


# ------------------------------------------------------------ no flag, no read

def test_without_host_eval_the_dump_is_empty_and_go_verdict_says_no_go(tmp_path, monkeypatch):
    monkeypatch.delenv("NC_HOST_EVAL", raising=False)
    monkeypatch.setattr(gd, "build_dump", lambda rows: pytest.fail("held-out path reached"))
    out = tmp_path / "go.json"
    assert gd.main(["go_dump_p35", str(out)]) == 0
    dump = json.loads(out.read_text())
    assert dump["liabilities"] == {}
    assert dump["status"] == "searched-and-empty" and dump["host_eval"] is False
    want = sorted(d["inn"] for d in load_v2_drugs() if sl.p35_heldout(d["inn"]))
    assert dump["heldout_drugs"] == want and len(want) == 70
    assert _verdict_lines(out) == ["NC_GO_VERDICT: NO-GO"]


def test_heldout_rows_refuse_to_load_without_host_eval(monkeypatch):
    monkeypatch.delenv("NC_HOST_EVAL", raising=False)
    with pytest.raises(RuntimeError, match="NC_HOST_EVAL"):
        gd.load_heldout_rows()
    monkeypatch.setenv("NC_HOST_EVAL", "0")
    with pytest.raises(RuntimeError, match="NC_HOST_EVAL"):
        gd.load_heldout_rows()


def test_host_eval_without_rows_file_is_empty_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("NC_HOST_EVAL", "1")
    monkeypatch.setattr(gd, "HELDOUT_ROWS", tmp_path / "absent.json")
    out = tmp_path / "go.json"
    assert gd.main(["go_dump_p35", str(out)]) == 0
    dump = json.loads(out.read_text())
    assert dump["liabilities"] == {} and dump["status"] == "searched-and-empty"
    assert _verdict_lines(out) == ["NC_GO_VERDICT: NO-GO"]


def test_heldout_label_builder_refuses_a_train_drug():
    train = next(d["inn"] for d in load_v2_drugs() if not sl.p35_heldout(d["inn"]))
    with pytest.raises(ValueError, match="P35-train"):
        gd.build_heldout_labels({train: {"has_record": True, "mapped_rows": []}})


# ------------------------------------------------------------ the rows file

def _rows_file() -> dict:
    if not gd.HELDOUT_ROWS.exists():
        pytest.skip("held-out rows file is stripped from session trees by guards.hide_heldout")
    return json.loads(gd.HELDOUT_ROWS.read_text())


def test_rows_file_covers_exactly_the_p35_heldout_drugs_and_stores_no_label():
    d = _rows_file()
    assert "drugs" not in d  # unknown shape: hide_heldout deletes it from session trees
    rows = d[gd.ROWS_KEY]
    assert sorted(rows) == sorted(i["inn"] for i in load_v2_drugs() if sl.p35_heldout(i["inn"]))
    assert d["sider"]["sha256"] == sl.SIDER_FILES
    train_rows = sl.load_train_rows()["drugs"]
    train_keys = set(next(iter(train_rows.values())))
    for inn, r in rows.items():
        assert set(r) == train_keys | {"sider_match"}, inn
        assert "label" not in json.dumps(r), inn
        for m in r["mapped_rows"]:
            assert sl.liability_of(m["side_effect_name"]), (inn, m["side_effect_name"])
            assert set(m) == {"flat", "stereo", "placebo", "frequency_desc", "frequency_lower",
                              "frequency_upper", "meddra_type", "side_effect_name"}
            assert m["flat"] in r["sider_match"]["flat"] and m["stereo"] in r["sider_match"]["stereo"]
        if r["mapped_rows"] or r["n_all_se_rows"]:
            assert r["has_record"]


def test_heldout_and_train_never_share_a_sider_stereo_id():
    rows = _rows_file()[gd.ROWS_KEY]
    train = {s for row in sl.load_cid_map().values() if "sider" in row for s in row["sider"]["stereo"]}
    held = [s for r in rows.values() for s in r["sider_match"]["stereo"]]
    assert len(held) == len(set(held)) and not train & set(held)


def test_guards_hide_heldout_deletes_the_rows_file_from_session_trees(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    guards = _load("guards", "pipeline/guards.py")
    tree = tmp_path / "x-tree"
    truth = tree / guards.V2_TRUTH
    truth.mkdir(parents=True)
    (truth / gd.HELDOUT_ROWS.name).write_text(json.dumps({"description": "x", gd.ROWS_KEY: {"x": {}}}))
    (truth / "kept.json").write_text(json.dumps({"drugs": [{"inn": "x"}]}))
    (tree / ".git").mkdir()
    guards.hide_heldout(tree)
    assert not (truth / gd.HELDOUT_ROWS.name).exists()
    assert (truth / "kept.json").exists()  # a drugs table is filtered, not deleted


# ------------------------------------------------- build path, on train stand-ins

@pytest.fixture(scope="module")
def standin_dump():
    """The real host path on P35-train drugs posing as held-out: every train
    drug whose split digit is 2 (labels from the train rows file, i.e. train
    data). They sit in the frozen kNN pool, but a drug is never its own
    neighbour, so the code path is exactly the host one."""
    mp = pytest.MonkeyPatch()
    train_rows = sl.load_train_rows()["drugs"]
    fake = {inn for inn in train_rows if sl.p35_split_digit(inn) == 2}
    mp.setattr(sl, "p35_heldout", lambda inn: inn in fake)
    try:
        rows = {inn: train_rows[inn] for inn in sorted(fake)}
        return gd.build_dump(rows)
    finally:
        mp.undo()


def test_dump_schema_is_what_go_verdict_reads(standin_dump, tmp_path):
    d = standin_dump
    assert d["status"] == "scored" and d["n_heldout"] > 20
    assert set(d["frozen_sha256"]) == {"src/neurocascade/benchmark/v2/p35_candidate.py",
                                       "src/neurocascade/benchmark/v2/frozen/p35_candidate.json"}
    assert set(d["not_scoreable"]) == {"seizure"}
    every = {**d["liabilities"], **d["below_prereg_floor"]}
    assert set(every) == {"sedation", "anticholinergic", "eps", "orthostatic_hypotension"}
    for name, e in every.items():
        n = len(e["drugs"])
        assert n > 0, name
        assert set(e["y"]) <= {0, 1} and len(e["y"]) == len(e["candidate"]) == n
        assert set(e["baselines"]) == {"occupancy_max", "strongest_target_ki",
                                       "one_nn_fingerprint", "atc_class_prevalence"}
        for b in e["baselines"].values():
            assert len(b) == n and None not in b  # the common set
        assert len(e["secondary"]["p34_3_frozen_candidate"]) == n
    for name, e in d["liabilities"].items():
        npos = sum(e["y"])
        assert npos >= gd.PREREG_MIN_CLASS and len(e["y"]) - npos >= gd.PREREG_MIN_CLASS
    for e in d["below_prereg_floor"].values():
        assert min(e["n_positive"], e["n_negative"]) < gd.PREREG_MIN_CLASS
    out = tmp_path / "go.json"
    out.write_text(json.dumps(d))
    assert len(_verdict_lines(out)) == 1
    assert "NC_GO_VERDICT" not in gd.report(d)


def test_standin_labels_equal_the_committed_train_labels(standin_dump):
    train = sl.load_train_labels()
    for e, entry in {**standin_dump["liabilities"], **standin_dump["below_prereg_floor"]}.items():
        assert entry["y"] == [train[i][e]["label"] for i in entry["drugs"]]


def test_check_freeze_refuses_an_edited_candidate(tmp_path, monkeypatch):
    res = json.loads(gd.P35_3_RESULT.read_text())
    res["code"]["sha256"] = "0" * 64
    fake = tmp_path / "P35.3.json"
    fake.write_text(json.dumps(res))
    monkeypatch.setattr(gd, "P35_3_RESULT", fake)
    with pytest.raises(RuntimeError, match="changed after the P35.3 freeze"):
        gd.check_freeze()


# ------------------------------------------------- predict path imports no truth

def test_candidate_predict_path_imports_no_heldout_truth():
    for rel in ("p35_candidate.py", "candidate.py", "frozen_candidate.py", "candidate_v2.py"):
        src = (V2 / rel).read_text()
        for token in ("p35_sider_heldout_rows", "HELDOUT_ROWS", "truth/", "load_v2_", "open("):
            assert token not in src, (rel, token)
        mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
        assert not any(m and ("go_dump" in m or "sider_labels" in m or m.lstrip(".") == "")
                       for m in mods), rel


def test_train_rows_extract_is_unchanged_by_p35_4():
    # P35.4 adds the held-out rows beside, never into, the P35.2 train extract.
    raw = json.loads(gzip.decompress(sl.TRAIN_ROWS.read_bytes()))
    assert not [i for i in raw["drugs"] if sl.p35_heldout(i)]


def test_results_file_reports_train_lodo_n_and_a_host_pending_heldout_split():
    res = json.loads((ROOT / "results" / "P35.4.json").read_text())
    p353 = json.loads((ROOT / "results" / "P35.3.json").read_text())
    assert res["n_scored"] == p353["n_scored"] == 41
    assert res["heldout"]["status"] == "host-pending"
    assert len(res["heldout"]["searched_and_empty_in_session"]) == 70
    assert res["session_chain_output_verbatim"].splitlines()[-1] == "NC_GO_VERDICT: NO-GO"
    for e, row in res["train_lodo_frozen_candidate"].items():
        assert row["candidate"] == p353["train_lodo_auroc"][e]["candidate"]
    ledger = (ROOT / "docs" / "validation-ledger.md").read_text()
    for line in res["session_chain_output_verbatim"].splitlines():
        if not line.startswith("written to"):
            assert f"`{line}`" in ledger  # quoted verbatim
