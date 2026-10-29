"""P32.2: the ``neurocascade`` CLI, the CSV package format and the pip package."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import venv
from pathlib import Path

import pytest

from neurocascade import cli
from neurocascade.compound import (
    PROVENANCE_LABEL,
    package_from_benchmark,
    predict_compound,
    score_clearance,
)
from neurocascade.package_csv import dumps_csv, loads_csv
from neurocascade.report import audit_report

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _strict(result):
    return json.dumps(cli.json_safe(result), sort_keys=True, allow_nan=False)


# ---- CSV package ---------------------------------------------------------------
@pytest.mark.parametrize("inn", ["venlafaxine", "haloperidol", "clozapine", "citalopram"])
def test_csv_round_trip_gives_the_identical_prediction(inn):
    package = package_from_benchmark(inn)
    assert _strict(predict_compound(loads_csv(dumps_csv(package)))) == _strict(
        predict_compound(package))


def test_csv_round_trip_keeps_projected_exposure_and_its_source():
    package = package_from_benchmark("haloperidol")
    package["human_exposure"] = {"cmax_unbound_nM": 0.5, "source": "PMID:12345678 projection"}
    back = loads_csv(dumps_csv(package))
    assert back["human_exposure"] == package["human_exposure"]
    assert back["binding"] == package["binding"]
    assert _strict(predict_compound(back)) == _strict(predict_compound(package))


@pytest.mark.parametrize("bad, message", [
    ("section,key,value\ncompound,name,x\nwhatever,k,1\n", "unknown row"),
    ("section,key,value,units\ncompound,name,x\nbinding,DRD2,3,uM\n", "nM"),
    ("section,key,row_type,value\ncompound,name,x\nadme,clint,row,3\n", "row_type"),
    ("key,value\nname,x\n", "lacks column"),
])
def test_csv_rejects_malformed_rows(bad, message):
    with pytest.raises(ValueError, match=message):
        loads_csv(bad)


# ---- CLI -----------------------------------------------------------------------
def test_cli_runs_csv_and_json_package_files_to_the_same_report(tmp_path):
    for fmt in ("csv", "json"):
        pkg = tmp_path / f"venlafaxine.{fmt}"
        assert cli.main(["export-benchmark", "venlafaxine", "-o", str(pkg)]) == 0
        assert cli.main(["predict", str(pkg), "-o", str(tmp_path / f"{fmt}.html"),
                         "--json", str(tmp_path / f"{fmt}.out.json")]) == 0
    assert (tmp_path / "csv.html").read_text() == (tmp_path / "json.html").read_text()
    out = (tmp_path / "csv.out.json").read_text()
    assert out == (tmp_path / "json.out.json").read_text()
    assert "Infinity" not in out and "NaN" not in out
    result = json.loads(out)
    cl = result["human_pk"]["clearance"]
    # P32.1 pins venlafaxine's IVIVE path; its 95% band is unbounded (P30.3, n=15).
    assert cl["provenance"]["band"] == "human_cl_ivive"
    assert cl["bands"]["0.95"] == {"lower": 0.0, "upper": None, "bounded": False}
    assert math.isclose(cl["point"], predict_compound(
        package_from_benchmark("venlafaxine"))["human_pk"]["clearance"]["point"], rel_tol=1e-15)
    assert result["provenance"]["label"] == PROVENANCE_LABEL
    report = (tmp_path / "csv.html").read_text()
    assert audit_report(report) == {"unlinked_numbers": [], "rows_without_label": [],
                                    "broken_links": []}


def test_cli_default_report_path_is_next_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["export-benchmark", "haloperidol", "-o", "halo.json"]) == 0
    assert cli.main(["predict", "halo.json"]) == 0
    assert (tmp_path / "halo.report.html").exists()


def test_cli_refusal_names_the_missing_core_input_in_plain_english(tmp_path, capsys):
    pkg = tmp_path / "lorazepam.json"
    assert cli.main(["export-benchmark", "lorazepam", "-o", str(pkg)]) == 0
    capsys.readouterr()
    assert cli.main(["predict", str(pkg), "-o", str(tmp_path / "r.html")]) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "refused: no prediction for lorazepam" in err
    assert "binding panel: no measured Ki at any target" in err
    assert not (tmp_path / "r.html").exists()


def test_cli_rejects_an_invalid_package_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "something-else", "name": "x"}))
    assert cli.main(["predict", str(bad)]) == cli.EXIT_INVALID
    assert "not a valid compound package" in capsys.readouterr().err
    assert cli.main(["export-benchmark", "not-a-drug", "-o", str(tmp_path / "x.json")]) == 1


# ---- the CLI path scored, against P32.1 -------------------------------------
def _load_script():
    spec = importlib.util.spec_from_file_location(
        "p32_2_cli_benchmark", ROOT / "scripts" / "p32_2_cli_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_path_reproduces_p32_1_clearance_score(tmp_path):
    script = _load_script()
    drugs = ["haloperidol", "clozapine", "citalopram", "aripiprazole", "lorazepam",
             "venlafaxine"]
    score = script.score_cli(drugs, tmp_path)
    assert script.matches_p32_1(score, score_clearance(drugs))
    assert score["refused"] == ["lorazepam"]
    assert score["reports_failing_audit"] == []
    assert score["by_band"]["human_cl_v2_train_mean"]["in_sample"] == ["aripiprazole"]


def test_committed_result_file():
    data = json.loads((ROOT / "results" / "P32.2.json").read_text())
    assert data["n_scored"] == 15 == data["train"]["n_out_of_sample"]
    assert data["train"]["n_scored"] == 17
    assert data["train"]["matches_p32_1_score_clearance"] is True
    assert data["train"]["reports_failing_audit"] == []
    assert data["heldout"]["reports_failing_audit"] == []
    assert data["provenance"] == PROVENANCE_LABEL
    train = data["train"]["by_band"]
    assert train["human_cl_ivive"]["n"] == 13
    assert train["human_cl_ivive"]["covered"] == {"0.80": 11, "0.95": 13}
    assert round(train["human_cl_ivive"]["gmfe"], 2) == 3.09
    assert train["human_cl_v2_train_mean"]["covered"] == {"0.80": 2, "0.95": 4}
    assert len(data["train"]["refused"]) == 101


# ---- pip package -------------------------------------------------------------
def test_pip_install_then_the_console_script_runs_a_package_file_to_a_report(tmp_path):
    """Build and install from a copy of src/ into a fresh venv, then run the CLI.

    The venv sees the host's numpy/scipy (system site packages) because the gate
    container has no network; neurocascade itself must come from the install.
    """
    shutil.copytree(SRC, tmp_path / "src", ignore=shutil.ignore_patterns(
        "__pycache__", "build", "*.egg-info"))
    venv.create(tmp_path / "venv", system_site_packages=True, with_pip=True)
    bindir = tmp_path / "venv" / ("Scripts" if os.name == "nt" else "bin")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([str(bindir / "python"), "-m", "pip", "install", "-q", "--no-deps",
                    "--no-build-isolation", "--no-index", str(tmp_path / "src")],
                   check=True, env=env, cwd=tmp_path)
    work = tmp_path / "run"
    work.mkdir()
    where = subprocess.run([str(bindir / "python"), "-c",
                            "import neurocascade; print(neurocascade.__file__)"],
                           check=True, env=env, cwd=work, capture_output=True, text=True)
    assert Path(where.stdout.strip()).is_relative_to(tmp_path / "venv")
    exe = str(bindir / "neurocascade")
    subprocess.run([exe, "export-benchmark", "citalopram", "-o", "citalopram.csv"],
                   check=True, env=env, cwd=work)
    subprocess.run([exe, "predict", "citalopram.csv", "-o", "report.html", "--json", "out.json"],
                   check=True, env=env, cwd=work)
    report = (work / "report.html").read_text()
    assert PROVENANCE_LABEL in report
    assert audit_report(report)["unlinked_numbers"] == []
    installed = json.loads((work / "out.json").read_text())
    assert installed["human_pk"]["clearance"]["point"] == json.loads(json.dumps(
        cli.json_safe(predict_compound(package_from_benchmark("citalopram")))))[
        "human_pk"]["clearance"]["point"]
