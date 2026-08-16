"""P10.1: figure-generation script and artifact contracts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_figures.py"
FIGURES = ROOT / "results" / "figures"


def test_figure_script_declares_the_complete_figure_set():
    namespace = {"__file__": str(SCRIPT), "__name__": "figure_test"}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
    assert namespace["FIGURE_NAMES"] == (
        "region_activity_map.png",
        "occupancy_time_course.png",
        "dose_response_curves.png",
        "heldout_validation_scatter.png",
    )
    for name in namespace["FIGURE_NAMES"]:
        assert (FIGURES / name).is_file()
        assert (FIGURES / name).stat().st_size > 1000


def test_figure_manifest_records_the_explicit_limitations():
    manifest = json.loads((FIGURES / "README.json").read_text())
    assert len(manifest["figures"]) == 4
    assert any("section coordinates" in note for note in manifest["notes"])
    assert any("not a fitted prediction" in note for note in manifest["notes"])
