"""Unit tests for guards.py: held-out hiding, host-read n, citation checks
(PubMed mocked; no network, Docker or GitHub)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guards as g  # noqa: E402


def write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    tmp_path = tmp_path / "t-tree"
    write(tmp_path, g.SPLIT, {"heldout": ["haloperidol", "valproic acid"], "train": ["diazepam"]})
    for inn in ("haloperidol", "valproic acid", "diazepam"):
        write(tmp_path, g.truth_file(inn), {"inn": inn, "cmax": 1})
    write(tmp_path, g.TRUTH + "effect_labels.json",
          {"drugs": {"haloperidol": 1, "diazepam": 2}})
    write(tmp_path, g.BENCH + "p29_5/oral_exposure.json",
          {"drugs": [{"inn": "haloperidol"}, {"inn": "diazepam", "split": "heldout"}, {"inn": "x"}]})
    write(tmp_path, "src/m.py", "a = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_frozen_paths_cover_split_aggregates_and_heldout_truth(repo):
    fp = g.frozen_paths(repo)
    assert g.SPLIT in fp and g.TRUTH + "effect_labels.json" in fp
    assert g.truth_file("valproic acid") == g.TRUTH + "valproic_acid.json" and g.truth_file("valproic acid") in fp
    assert g.truth_file("diazepam") not in fp           # train truth stays editable
    assert g.frozen_paths(repo / "nowhere") == [g.SPLIT, *g.TRUTH_AGGREGATES]


def test_hide_heldout_strips_truth_and_history(repo):
    base, applied = g.hide_heldout(repo)
    assert not applied
    assert not (repo / g.truth_file("haloperidol")).exists()
    assert not (repo / g.truth_file("valproic acid")).exists()
    assert (repo / g.truth_file("diazepam")).exists()
    assert json.loads((repo / g.TRUTH / "effect_labels.json").read_text())["drugs"] == {"diazepam": 2}
    rows = json.loads((repo / g.BENCH / "p29_5/oral_exposure.json").read_text())["drugs"]
    assert rows == [{"inn": "x"}]
    # One commit, no trace of the held-out file anywhere in history.
    assert git(repo, "rev-list", "--count", "HEAD") == "1" and git(repo, "rev-parse", "HEAD") == base
    assert "haloperidol.json" not in git(repo, "log", "--all", "--name-only", "--format=")
    assert git(repo, "status", "--porcelain") == ""


def test_fresh_heldout_set_is_frozen_and_stripped_wholesale(repo):
    # P34.4 fresh held-out set (new drugs, not caught by the v2_heldout hash).
    write(repo, g.V2_FRESH + "fresh_liabilities.json",
          {"drugs": [{"inn": "brexpiprazole", "sedation": {"label": 1}}]})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fresh set")
    assert g.V2_FRESH + "fresh_liabilities.json" in g.frozen_paths(repo)
    g.hide_heldout(repo)
    assert not (repo / g.V2_FRESH / "fresh_liabilities.json").exists()
    assert "fresh_liabilities.json" not in git(repo, "log", "--all", "--name-only", "--format=")


def test_hide_heldout_refuses_anything_but_a_session_tree(tmp_path):
    (tmp_path / "neurocascade" / ".git").mkdir(parents=True)
    with pytest.raises(ValueError):
        g.hide_heldout(tmp_path / "neurocascade")
    assert (tmp_path / "neurocascade" / ".git").exists()


def test_hide_heldout_applies_wip_against_hidden_base(repo):
    wip = "diff --git a/src/m.py b/src/m.py\n--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    base, applied = g.hide_heldout(repo, wip)
    assert applied and (repo / "src/m.py").read_text() == "a = 2\n"
    assert "src/m.py" in git(repo, "diff", base, "--name-only")


def test_ledger_rows_and_host_n(tmp_path):
    row = f"diff --git a/{g.LEDGER} b/{g.LEDGER}\n+| P9.9 | thing | n=3 |\n"
    sep = f"diff --git a/{g.LEDGER} b/{g.LEDGER}\n+|---|---|\n"
    other = "diff --git a/docs/x.md b/docs/x.md\n+| P9.9 | thing |\n"
    assert g.ledger_rows_added(row) and not g.ledger_rows_added(sep) and not g.ledger_rows_added(other)
    assert g.host_n(tmp_path, "P9.9", other) == (None, None)
    n, err = g.host_n(tmp_path, "P9.9", row)
    assert n is None and "results/P9.9.json" in err
    for bad in ({"n_scored": "3"}, {"n_scored": True}, {"n_scored": -1}, {"n": 3}):
        write(tmp_path, "results/P9.9.json", bad)
        assert g.host_n(tmp_path, "P9.9", row)[1]
    write(tmp_path, "results/P9.9.json", {"n_scored": 0})
    assert g.host_n(tmp_path, "P9.9", row) == (0, None)


def test_pmid_refs_scope_inn_and_note():
    obj = {"inn": "diazepam", "rows": [{"pmid": "123"}, {"pmid": 456, "pmid_note": "class review"},
                                       {"pmid": None}, {"inn": "lorazepam", "pmid": "789"}]}
    assert g.pmid_refs(obj) == [("123", "diazepam", False), ("456", "diazepam", True),
                                ("789", "lorazepam", False)]
    # Per-drug files take the INN from the file name; aggregates do not.
    assert g.file_refs(g.BENCH + "binding/valproic_acid.json", '{"k": [{"pmid": "1"}]}') == [
        ("1", "valproic acid", False)]
    assert g.file_refs(g.TRUTH + "effect_labels.json", '{"pmid": "1"}') == [("1", None, False)]
    assert g.file_refs(g.BENCH + "binding/x.json", "not json") == []


def fake_pubmed(records):
    """fetch() stand-in serving esummary JSON and efetch XML for `records`
    ({pmid: (title, abstract)}); records absent from it do not exist."""
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        ids = url.split("id=")[1].split("&")[0].split("%2C")
        if "esummary" in url:
            res = {p: ({"uid": p, "title": records[p][0], "fulljournalname": "J"} if p in records
                       else {"uid": p, "error": "cannot get document summary"}) for p in ids}
            return json.dumps({"result": res}).encode()
        arts = "".join(f"<PubmedArticle><MedlineCitation><PMID>{p}</PMID><Article><ArticleTitle>"
                       f"{records[p][0]}</ArticleTitle><Abstract><AbstractText>{records[p][1]}"
                       f"</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>"
                       for p in ids if p in records)
        return f"<PubmedArticleSet>{arts}</PubmedArticleSet>".encode()
    return fetch, calls


def test_check_refs_missing_unnamed_note_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    fetch, calls = fake_pubmed({"111": ("Diazepam kinetics", ""), "222": ("Renal clearance", "of 600 drugs"),
                                "333": ("Benzodiazepines", "Valproic-acid interaction")})
    cache = tmp_path / "cache.json"
    refs = [("111", "diazepam", False), ("222", "diazepam", False), ("222", "lorazepam", True),
            ("0333", "valproic acid", False), ("999", "diazepam", False), ("abc", None, False)]
    bad = g.check_refs(refs, cache, fetch)
    assert len(bad) == 3
    assert "PMID 222 (diazepam): drug not named" in bad[0]
    assert bad[1].startswith("PMID 999 (diazepam): not in PubMed")
    assert bad[2].startswith("PMID abc (no drug): not in PubMed")
    n = len(calls)
    assert g.check_refs(refs, cache, fetch) == bad and len(calls) == n    # served from cache


def test_check_refs_network_failure_is_unavailable(tmp_path):
    def down(url, timeout):
        raise g.CitationsUnavailable("timeout")
    with pytest.raises(g.CitationsUnavailable):
        g.check_refs([("1", "x", False)], tmp_path / "c.json", down)


def test_new_refs_only_counts_added_citations():
    p = g.BENCH + "binding/diazepam.json"
    old = {p: json.dumps({"rows": [{"pmid": "1"}]})}
    new = {p: json.dumps({"rows": [{"pmid": "1"}, {"pmid": "2"}]}),
           "docs/x.json": json.dumps({"pmid": "3"})}
    assert g.new_refs([p, "docs/x.json"], old, new) == [("2", "diazepam", False)]
    assert g.new_refs([g.BENCH + "adme/new.json"], {}, {g.BENCH + "adme/new.json": '{"pmid": "4"}'}) == [
        ("4", "new", False)]


def test_v2_heldout_rule_is_deterministic_and_about_a_fifth():
    names = [f"drug{i}" for i in range(2000)]
    share = sum(map(g.v2_heldout, names)) / len(names)
    assert 0.17 < share < 0.23
    assert g.v2_heldout("Haloperidol ") == g.v2_heldout("haloperidol")


def test_v2_truth_frozen_and_heldout_rows_hidden(repo):
    inns = [f"drug{i}" for i in range(40)]
    write(repo, g.V2_TRUTH + "ki.json", {"drugs": [{"inn": i, "ki": 1} for i in inns]})
    write(repo, g.V2_TRUTH + "odd.json", [1, 2])
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v2")
    assert g.V2_TRUTH + "ki.json" in g.frozen_paths(repo)
    tree = repo.parent / "x-tree"
    subprocess.run(["cp", "-a", str(repo), str(tree)], check=True)
    g.hide_heldout(tree)
    kept = [r["inn"] for r in json.loads((tree / g.V2_TRUTH / "ki.json").read_text())["drugs"]]
    assert kept == [i for i in inns if not g.v2_heldout(i)] and len(kept) < len(inns)
    assert not (tree / g.V2_TRUTH / "odd.json").exists()


def test_v2_correction_maps_keep_train_and_hide_heldout(repo):
    inns = [f"drug{i}" for i in range(40)]
    write(repo, g.V2_TRUTH + "pk_corrections.json",
          {"version": 1, "rule": "r", "drop": {i: {"fields": ["cl"]} for i in inns},
           "replace": {i: {"vss": {"value": 1}} for i in inns}})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v2 corrections")
    tree = repo.parent / "c-tree"
    subprocess.run(["cp", "-a", str(repo), str(tree)], check=True)
    g.hide_heldout(tree)
    d = json.loads((tree / g.V2_TRUTH / "pk_corrections.json").read_text())
    train = [i for i in inns if not g.v2_heldout(i)]
    assert sorted(d["drop"]) == sorted(train) and sorted(d["replace"]) == sorted(train)
    assert d["rule"] == "r" and len(train) < len(inns)


def test_v2_human_cmax_corrections_heldout_inns_stripped(repo):
    """The real truth/human_cmax_corrections.json (PR #52) is applied by
    load_v2_human_cmax(); hide_heldout must strip every held-out INN from its
    per-INN maps and keep every train INN and the file metadata."""
    rel = g.V2_TRUTH + "human_cmax_corrections.json"
    real = json.loads((Path(__file__).resolve().parents[1] / rel).read_text())
    write(repo, rel, real)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "v2 cmax corrections")
    tree = repo.parent / "h-tree"
    subprocess.run(["cp", "-a", str(repo), str(tree)], check=True)
    g.hide_heldout(tree)
    d = json.loads((tree / rel).read_text())
    for k in g.V2_INN_MAPS:
        before = real.get(k) or {}
        assert sorted(d.get(k) or {}) == sorted(i for i in before if not g.v2_heldout(i))
        assert not any(g.v2_heldout(i) for i in d.get(k) or {})
    assert {k: v for k, v in d.items() if k not in g.V2_INN_MAPS} == \
        {k: v for k, v in real.items() if k not in g.V2_INN_MAPS}
    # non-vacuous: the committed file does carry held-out INNs to strip
    assert any(g.v2_heldout(i) for k in g.V2_INN_MAPS for i in real.get(k) or {})
