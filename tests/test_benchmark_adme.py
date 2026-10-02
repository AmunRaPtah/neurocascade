"""P28.3: per-drug preclinical ADME/PK records (plasma fu, brain fu / Kp,uu,
in-vitro intrinsic clearance, rodent in-vivo CL/Vss) for the benchmark."""

import importlib.util
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

from neurocascade.benchmark import (
    ADME_DIR,
    load_adme_coverage,
    load_adme_record,
    load_drugs,
)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("plasma_fu", "brain_fu", "kp_uu", "kp_total", "clint",
          "invivo_cl_preclinical", "invivo_vss_preclinical")


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_adme", ROOT / "scripts" / "build_benchmark_adme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def records():
    return {d["inn"]: load_adme_record(d["inn"]) for d in load_drugs()}


def test_every_benchmark_drug_has_exactly_one_record(records):
    files = {p.name for p in ADME_DIR.glob("*.json")} - {"coverage.json"}
    assert len(files) == 185 == len(records)
    for inn, r in records.items():
        assert r["inn"] == inn
        assert r["status"] in ("data", "searched_and_empty")
        assert (r["status"] == "data") == bool(r["fields"])
        assert "chembl_adme_rows_returned" in r["searched"]
        assert set(r["fields"]) <= set(FIELDS)
    with pytest.raises(KeyError):
        load_adme_record("aspirin")


def test_every_measurement_carries_a_resolvable_citation(records):
    for inn, r in records.items():
        refs = r["references"]
        for field, blob in r["fields"].items():
            assert blob["measurements"], (inn, field)
            for m in blob["measurements"]:
                assert m["value"] is not None
                assert m["relation"] in ("=", "<", ">")
                assert m["ref"] in refs, (inn, field, m["ref"])
                ref = refs[m["ref"]]
                # a citation is a PMID or at least the ChEMBL document id
                assert ref["pmid"] or ref["chembl_document_id"]


def test_in_vivo_cl_vss_fields_are_non_human_only(records):
    """Human in-vivo CL/Vss is clinical truth (P28.4), not a preclinical input;
    it must be excluded here, and the exclusion counted."""
    for inn, r in records.items():
        for field in ("invivo_cl_preclinical", "invivo_vss_preclinical"):
            blob = r["fields"].get(field)
            if not blob:
                continue
            for m in blob["measurements"]:
                assert m["species"] != "human", (inn, field)
                assert m["species"] != "unspecified", (inn, field)


def test_summaries_never_pool_incompatible_units_or_matrices(records):
    """A single summary value may only be computed over one (species, units,
    matrix) group, so mL/min/g is never averaged with uL/min/10^6cells, nor
    microsomes with S9."""
    for inn, r in records.items():
        for field, blob in r["fields"].items():
            seen = set()
            for row in blob["summary"]:
                key = (row["species"], row.get("units"), row.get("matrix"))
                assert key not in seen, (inn, field, key)
                seen.add(key)
                members = [m for m in blob["measurements"]
                           if m["species"] == row["species"]
                           and m.get("units") == row.get("units")
                           and m.get("matrix") == row.get("matrix")]
                assert len(members) == row["n_measurements"]


def test_summary_value_is_median_over_documents_of_geomean(records):
    for inn, r in records.items():
        for field, blob in r["fields"].items():
            for row in blob["summary"]:
                members = [m for m in blob["measurements"]
                           if m["species"] == row["species"]
                           and m.get("units") == row.get("units")
                           and m.get("matrix") == row.get("matrix")
                           and m["relation"] == "="]
                if not members:
                    assert row["value"] is None
                    continue
                by_ref = defaultdict(list)
                for m in members:
                    by_ref[m["ref"]].append(m["value"])
                per_ref = [math.exp(statistics.fmean(math.log(v) for v in vs))
                           for vs in by_ref.values()]
                assert row["value"] == pytest.approx(
                    round(statistics.median(per_ref), 6), abs=1e-6)


def test_ppb_rows_are_converted_to_fraction_unbound(records):
    """A plasma_fu measurement sourced from a PPB% row stores fu = 1 - PPB/100,
    is unitless, and keeps the original percent."""
    for inn, r in records.items():
        blob = r["fields"].get("plasma_fu")
        if not blob:
            continue
        for m in blob["measurements"]:
            if "ppb_percent" in m:
                assert m["units"] is None
                assert m["value"] == pytest.approx(1.0 - m["ppb_percent"] / 100.0, abs=1e-6)
                assert 0.0 <= m["value"] <= 1.0


def test_fraction_unbound_values_are_fractions(records):
    for inn, r in records.items():
        for field in ("plasma_fu", "brain_fu", "kp_uu"):
            blob = r["fields"].get(field)
            if not blob:
                continue
            for m in blob["measurements"]:
                assert 0.0 < m["value"] <= (1.0 if field != "kp_uu" else 1e6)


def test_coverage_matches_records(records):
    cov = load_adme_coverage()
    s = cov["summary"]
    assert s["n_drugs"] == 185
    by_inn = {r["inn"]: r for r in cov["drugs"]}
    assert set(by_inn) == set(records)
    n_any = 0
    per_field = defaultdict(int)
    for inn, r in records.items():
        row = by_inn[inn]
        has = r["status"] == "data"
        n_any += has
        assert row["status"] == r["status"]
        assert row["plasma_fu"] == ("plasma_fu" in r["fields"])
        assert row["brain_fu_or_kp_uu"] == (
            "brain_fu" in r["fields"] or "kp_uu" in r["fields"])
        assert row["clint"] == ("clint" in r["fields"])
        for f in ("plasma_fu", "brain_fu_or_kp_uu", "clint",
                  "rodent_in_vivo_cl", "rodent_in_vivo_vss"):
            per_field[f] += bool(row[f])
    assert s["n_with_any_field"] == n_any
    for f, n in per_field.items():
        assert s["per_field"][f] == n


def test_known_plasma_fu_values_match_literature(records):
    """Spot-check human plasma fraction unbound against well-established protein
    binding. These are summary medians the build computes, not hand-entered."""
    def human_fu(inn):
        for row in records[inn]["fields"]["plasma_fu"]["summary"]:
            if row["species"] == "human" and row["units"] is None:
                return row["value"]
        return None
    # diazepam ~98-99% bound; haloperidol ~90%; fluoxetine ~94-95%
    assert human_fu("diazepam") == pytest.approx(0.0215, abs=0.01)
    assert human_fu("haloperidol") == pytest.approx(0.08, abs=0.03)
    assert human_fu("fluoxetine") == pytest.approx(0.0514, abs=0.02)


def test_classifier_boundaries():
    mod = _script()
    # PAMPA permeability (cm/s) is not a partition coefficient
    assert mod._classify("Kp", "10^-6 cm/s", "Permeability by PAMPA", None) is None
    # brain homogenate Fu is brain_fu, not plasma_fu
    assert mod._classify("Fu", None, "Fraction unbound in rat brain homogenate", "Brain") \
        == ("brain_fu", None)
    # liver microsome CL is clint, not in-vivo CL
    assert mod._classify("CL", "mL.min-1.g-1",
                         "Intrinsic clearance in human liver microsomes", None) \
        == ("clint", "microsomes")
    # whole-animal CL in mL/min/kg is in-vivo
    assert mod._classify("CL", "mL.min-1.kg-1", "Clearance in rat after iv", None) \
        == ("invivo_cl", None)
    # species falls back to the description when organism is null
    assert mod._species(None, "Volume of distribution in man (IV dose)") == "human"
    assert mod._species(None, "Clearance in rat after iv administration") == "rat"
