"""Unit tests for the pure parts of orchestrate.py (no docker, git or Modal).

    python -m pytest -q pipeline/test_orchestrate.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestrate as o  # noqa: E402

SPEC = """---
id: P1.2a
title: Do a thing
tests: [tests/test_a.py, tests/test_b.py::test_x]
---
Brief text.
"""


def test_parse_spec_defaults_protected_to_tests():
    s = o.parse_spec(SPEC)
    assert s["id"] == "P1.2a" and s["brief"] == "Brief text."
    assert s["tests"] == ["tests/test_a.py", "tests/test_b.py::test_x"]
    assert s["protected"] == s["tests"]


@pytest.mark.parametrize("text", ["no frontmatter", "---\nid: X.1\ntitle: t\n---\nbody"])
def test_parse_spec_rejects_incomplete(text):
    with pytest.raises(ValueError):
        o.parse_spec(text)


def patch(*paths):
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n" for p in paths)


SPEC_D = o.parse_spec(SPEC)


@pytest.mark.parametrize("paths,expect", [
    ((), "empty diff"),
    (("src/x.py", "tests/test_a.py"), "protected"),
    (("src/x.py", "pipeline/orchestrate.py"), "protected"),
    (("src/x.py", "data/f.csv"), "protected"),
    (("src/x.py", "debug_run.py"), "outside"),
    (("tests/test_new.py",), "no source"),
])
def test_check_patch_rejects(paths, expect):
    assert expect in o.check_patch(patch(*paths), SPEC_D)


def test_check_patch_accepts_src_tests_docs():
    assert o.check_patch(patch("src/x.py", "tests/test_new.py", "docs/x.md"), SPEC_D) is None


def test_exclude_args_ignores_files_and_deselects_nodes():
    assert o.exclude_args(SPEC_D["tests"]) == \
        "--ignore tests/test_a.py --deselect tests/test_b.py::test_x"


def result_event(**kw):
    return json.dumps({"subtype": "success", "num_turns": 7, "result": "x\nTASK_RESULT: DONE - ok",
                       **kw, "type": "result"})


def test_parse_agent_log_is_key_order_independent():
    # "type" last: the case the original prefix match missed.
    log = '{"type":"system"}\nnot json\n' + result_event()
    r = o.parse_agent_log(log)
    assert r["turns"] == 7 and r["subtype"] == "success" and r["api_error"] is None
    assert o.result_line(r["final"]) == "TASK_RESULT: DONE - ok"


def test_parse_agent_log_flags_api_errors():
    assert o.parse_agent_log(result_event(api_error_status=502))["api_error"] == "502"
    assert o.parse_agent_log(result_event(terminal_reason="api_error"))["api_error"] == "api_error"


def test_parse_agent_log_max_turns_is_worker_failure_not_infra():
    r = o.parse_agent_log(result_event(subtype="error_max_turns"))
    assert r["subtype"] == "error_max_turns" and r["api_error"] is None


def test_parse_agent_log_without_result():
    assert o.parse_agent_log("")["final"] == "" and o.result_line("") == "TASK_RESULT: (missing)"


def test_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "RUNS", tmp_path)
    monkeypatch.setattr(o, "LOCK", tmp_path / ".lock")
    assert not o.lock_held()
    held = o.acquire_lock()
    assert o.lock_held()
    held.close()
    assert not o.lock_held()
