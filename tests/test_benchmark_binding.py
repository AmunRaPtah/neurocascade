"""P28.2: per-drug human receptor binding panels (Ki + action type) for the benchmark."""

import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

from neurocascade.benchmark import (
    BINDING_DIR,
    load_binding_coverage,
    load_binding_panel,
    load_drugs,
    load_split,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"ChEMBL", "PDSP", "GtoPdb"}


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_binding", ROOT / "scripts" / "build_benchmark_binding.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def panels():
    return {d["inn"]: load_binding_panel(d["inn"]) for d in load_drugs()}


def test_every_benchmark_drug_has_exactly_one_record(panels):
    files = {p.name for p in BINDING_DIR.glob("*.json")} - {"coverage.json"}
    assert len(files) == 185 == len(panels)
    for inn, p in panels.items():
        assert p["inn"] == inn
        assert p["status"] in ("panel", "searched_and_empty")
        assert (p["status"] == "panel") == bool(p["targets"])
        # a searched-and-empty record says what was searched
        assert p["searched"]["pdsp_ligand_names_tried"]
        assert "chembl_ki_rows_returned" in p["searched"]
    with pytest.raises(KeyError):
        load_binding_panel("aspirin")


def test_every_value_carries_a_resolvable_citation(panels):
    for inn, p in panels.items():
        refs = p["references"]
        for key, t in p["targets"].items():
            assert t["measurements"], (inn, key)
            for m in t["measurements"]:
                assert m["source"] in SOURCES
                assert m["ki_nM"] > 0
                assert m["relation"] in ("=", ">")
                assert m["record"]
                ref = refs[m["ref"]]
                assert ref["pmid"] or ref["citation"], (inn, key, m)
                if m["ref"].startswith("PMID:"):
                    assert ref["pmid"] == m["ref"][5:]
            for a in t["actions"]:
                assert a["ref"] in refs
        for v in p["actions_without_ki"].values():
            for a in v:
                assert a["ref"] in refs


def test_summary_ki_is_the_median_over_references_of_mean_pki(panels):
    for inn, p in panels.items():
        for key, t in p["targets"].items():
            exact = [m for m in t["measurements"] if m["relation"] == "="]
            by_ref = defaultdict(list)
            for m in exact:
                by_ref[m["ref"]].append(9 - math.log10(m["ki_nM"]))
            pki = statistics.median(statistics.fmean(v) for v in by_ref.values())
            assert t["pki_median"] == pytest.approx(pki, abs=1e-4)
            assert t["ki_nM"] == pytest.approx(10 ** (9 - pki), rel=1e-5)
            assert t["n_measurements"] == len(exact)
            assert t["n_references"] == len(by_ref)
            assert t["n_censored"] == len(t["measurements"]) - len(exact)
            assert t["ki_min_nM"] <= t["ki_nM"] * (1 + 1e-6)
            assert t["ki_nM"] <= t["ki_max_nM"] * (1 + 1e-6)


def test_coverage_file_matches_the_panels(panels):
    cov = load_binding_coverage()
    s = cov["summary"]
    covered = [p for p in panels.values() if p["status"] == "panel"]
    assert s["n_drugs"] == 185
    assert s["n_with_panel"] == len(covered) == 118
    assert s["n_searched_and_empty"] == 67
    assert s["searched_and_empty"] == sorted(
        i for i, p in panels.items() if p["status"] == "searched_and_empty")
    mean_t = statistics.fmean(len(p["targets"]) for p in covered)
    assert s["mean_targets_per_covered_drug"] == round(mean_t, 2) == 12.57
    assert sum(v["n"] for v in s["by_stratum"].values()) == 185
    assert {k: v["with_panel"] for k, v in s["by_stratum"].items()} == {
        "N05A": 57, "N05B": 5, "N06A": 47, "FAIL": 9}


def test_heldout_drugs_are_not_treated_differently(panels):
    # Panels are preclinical inputs; the frozen split must still load, and
    # held-out drugs get inputs exactly like training drugs.
    split = load_split()
    assert set(split["train"]) | set(split["heldout"]) == set(panels)


def test_anchor_values_from_the_committed_panels(panels):
    hal = panels["haloperidol"]["targets"]["DRD2"]
    assert hal["ki_nM"] == pytest.approx(2.0)
    assert hal["action_types"] == ["antagonist"]
    assert panels["aripiprazole"]["targets"]["DRD2"]["action_types"] == ["partial agonist"]
    assert panels["risperidone"]["targets"]["HTR2A"]["ki_nM"] < panels["risperidone"]["targets"]["DRD2"]["ki_nM"]
    # flupentixol is resolved to the active cis(Z) isomer, and says why
    flu = panels["flupentixol"]
    assert flu["chembl_id_queried"] == "CHEMBL54661"
    assert "1:1" in flu["entity_note"]
    assert "FLUPENTHIXOL, Alpha" in flu["searched"]["pdsp_ligand_names_matched"]


# ------------------------------------------------- inclusion rules, offline


def _drug(inn="testdrug"):
    return {"inn": inn, "stratum": "N05A", "chembl_id": "CHEMBL1", "standard_inchikey": "X"}


def _act(aid, assay, value, rel="=", target="CHEMBL_T1"):
    return {"activity_id": aid, "assay_chembl_id": assay, "target_chembl_id": target,
            "document_chembl_id": "DOC1", "standard_value": value, "standard_units": "nM",
            "standard_relation": rel, "data_validity_comment": None, "potential_duplicate": 0,
            "activity_comment": None, "standard_text_value": None}


TARGET = {"CHEMBL_T1": {"target_type": "SINGLE PROTEIN", "pref_name": "Dopamine D2 receptor",
                        "organism": "Homo sapiens",
                        "target_components": [{"accession": "P14416", "target_component_synonyms": [
                            {"syn_type": "GENE_SYMBOL", "component_synonym": "DRD2"}]}]}}
DOCS = {"DOC1": {"pubmed_id": 123456, "doi": None, "authors": "A", "year": 2000,
                 "journal": "J", "volume": "1", "first_page": "1"}}


def _pdsp(name, species, value, note=""):
    return {" Ligand Name": name, "species": species, "Name": "DOPAMINE D2", "ki Note": note,
            "ki Val": value, "Link": "https://www.ncbi.nlm.nih.gov/pubmed/7654321",
            "Reference": "B 1990", "Number": "1", "source": "CLONED", "Hotligand": "x"}


def test_chembl_rows_on_non_human_assays_are_excluded():
    mod = _script()
    assays = {
        # human recombinant, direct single-protein assignment: kept
        "A_HUMAN": {"assay_tax_id": 9606, "confidence_score": 9},
        # guinea-pig brain mapped to the human target as a homologue
        # (the real case: haloperidol sigma-1, CHEMBL802065): excluded
        "A_GPIG": {"assay_tax_id": 10141, "confidence_score": 8},
        # human assay but only homologous assignment: excluded
        "A_HOMOL": {"assay_tax_id": 9606, "confidence_score": 8},
    }
    acts = [_act(1, "A_HUMAN", "2.0"), _act(2, "A_GPIG", "0.5"), _act(3, "A_HOMOL", "0.5"),
            _act(4, "A_HUMAN", "10000", rel=">")]
    out = mod._build_one(_drug(), acts, assays, DOCS, TARGET, [], {}, {}, {}, defaultdict(list))
    t = out["targets"]["DRD2"]
    assert [m["ki_nM"] for m in t["measurements"] if m["relation"] == "="] == [2.0]
    assert t["n_censored"] == 1 and t["ki_nM"] == pytest.approx(2.0)
    assert out["searched"]["excluded_counts"]["chembl_assay_organism_not_human"] == 1
    assert out["searched"]["excluded_counts"][
        "chembl_target_not_direct_single_protein_or_complex"] == 1
    assert out["references"]["PMID:123456"]["pmid"] == "123456"


def test_pdsp_rows_need_human_species_and_a_same_entity_name():
    mod = _script()
    by_name = defaultdict(list)
    for r in (_pdsp("MIANSERIN", "HUMAN", "3"), _pdsp("mianserine", "HUMAN", "5"),
              _pdsp("MIANSERIN (+)", "HUMAN", "0.1"), _pdsp("MIANSERIN", "RAT", "0.2"),
              _pdsp("MIANSERIN", "HUMAN", "10000", note=">")):
        by_name[r[" Ligand Name"].strip().lower()].append(r)
    out = mod._build_one(_drug("mianserin"), [], {}, {}, {}, [], by_name, {}, {},
                         defaultdict(list))
    t = out["targets"]["DRD2"]
    exact = sorted(m["ki_nM"] for m in t["measurements"] if m["relation"] == "=")
    # the (+) enantiomer is a different entity and the rat row a different species
    assert exact == [3.0, 5.0]
    assert t["n_censored"] == 1
    assert out["searched"]["excluded_counts"] == {"pdsp_species_not_human": 1}
    assert "mianserin (+)" not in out["searched"]["pdsp_ligand_names_tried"]


def test_pdsp_aliases_never_cross_stereoisomers():
    mod = _script()
    racemic_vs_single = {
        "citalopram": "escitalopram", "sulpiride": "levosulpiride",
        "milnacipran": "levomilnacipran", "clopenthixol": "zuclopenthixol",
    }
    for racemate, single in racemic_vs_single.items():
        a = {n.lower() for n in mod.PDSP_ALIASES.get(racemate, [])} | {racemate}
        b = {n.lower() for n in mod.PDSP_ALIASES.get(single, [])} | {single}
        assert not (a & b), (racemate, single)


def test_gtopdb_action_labels():
    mod = _script()
    assert mod._gtopdb_action("Allosteric modulator", "Positive") == "positive allosteric modulator"
    assert mod._gtopdb_action("Allosteric modulator", "Negative") == "negative allosteric modulator"
    assert mod._gtopdb_action("Agonist", "Partial agonist") == "partial agonist"
    assert mod._gtopdb_action("Antagonist", "Inverse agonist") == "inverse agonist"
    assert mod._gtopdb_action("Inhibitor", "Inhibition") == "inhibitor"
    assert mod._gtopdb_action("Channel blocker", "None") == "blocker"
    assert mod._gtopdb_action("None", "Binding") is None


def test_panel_files_are_valid_json_with_provenance():
    for path in BINDING_DIR.glob("*.json"):
        rec = json.loads(path.read_text())
        prov = rec["provenance"]
        assert prov["chembl"]["release"] == "ChEMBL_37 (2026-05-01)"
        assert prov["gtopdb"]["version"].startswith("GtoPdb Version: 2026.3")
        assert len(prov["pdsp"]["sha256"]) == 64
