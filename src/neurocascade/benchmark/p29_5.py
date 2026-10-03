"""P29.5 oral-exposure and PET plasma-pair records.

These records are deliberately separate from the frozen P28.4 truth fields.
They are evidence additions for Phase 29: fitting code must not import them,
and held-out rows are comparison data only.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import BENCHMARK_DIR, load_adme_record, load_split

P29_5_DIR = BENCHMARK_DIR / "p29_5"
PET_DRUGS = {
    "amisulpride", "asenapine", "escitalopram", "lurasidone",
    "milnacipran", "paroxetine", "quetiapine", "venlafaxine",
}


def _load(name: str) -> dict:
    return json.loads((P29_5_DIR / name).read_text())


def load_oral_exposure() -> dict:
    """Return same-entity human oral exposure observations and coverage."""
    return _load("oral_exposure.json")


def load_pet_plasma_pairs() -> dict:
    """Return PET observations that also print a plasma concentration."""
    return _load("pet_plasma_pairs.json")


def load_preclinical_oral_inputs() -> dict:
    """Return supplemental preclinical oral/clearance evidence.

    Existing P28.3 records are counted only when their cited record contains a
    usable clearance or oral-absorption field. Fraction-unbound and
    brain-exposure fields alone are not promoted to oral inputs. The
    supplemental rows contain the newly sourced drug rows; searched-and-empty
    rows remain visible and are never treated as inputs.
    """
    supplement = _load("preclinical_oral_inputs.json")
    split = load_split()
    rows = {row["inn"]: dict(row) for row in supplement["drugs"]}
    for side in ("train", "heldout"):
        for inn in split[side]:
            if inn in rows:
                continue
            adme = load_adme_record(inn)
            fields = set(adme.get("fields", {}))
            usable = fields & {
                "oral_bioavailability", "oral_bioavailability_rat",
                "oral_bioavailability_dog", "permeability", "fa", "clint",
                "invivo_cl_preclinical", "invivo_vss_preclinical",
            }
            if usable:
                rows[inn] = {
                    "inn": inn,
                    "split": side,
                    "status": "data",
                    "source": "P28.3 ADME record",
                    "fields": sorted(usable),
                    "references": ["P28.3 record; per-row citations retained there"],
                }
    counts = {
        "train": sum(r["split"] == "train" and r["status"] == "data"
                      for r in rows.values()),
        "heldout": sum(r["split"] == "heldout" and r["status"] == "data"
                        for r in rows.values()),
        "heldout_pet": sum(
            r["split"] == "heldout" and r["status"] == "data"
            and r["inn"] in PET_DRUGS for r in rows.values()
        ),
    }
    return {"drugs": sorted(rows.values(), key=lambda r: r["inn"]),
            "coverage": counts, "source": supplement["source"]}
