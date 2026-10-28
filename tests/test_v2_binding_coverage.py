"""v2 binding coverage extension (2026-10-28): censored lower bounds and the
GtoPdb and PDSP added sources. Checks the built table and the merge/mapping rules of
scripts/v2_binding.py on synthetic rows."""

import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDING = json.loads(
    (ROOT / "src" / "neurocascade" / "benchmark" / "v2" / "inputs" / "binding.json").read_text())
PANEL = set(BINDING["provenance"]["panel"])

_spec = importlib.util.spec_from_file_location("v2_binding", ROOT / "scripts" / "v2_binding.py")
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

MEASURED_KEYS = {"ki_nM", "pki_median", "ki_min_nM", "ki_max_nM", "status"}


# ------------------------------------------------------------ built table

def test_every_drug_has_lower_bound_field():
    for inn, rec in BINDING["drugs"].items():
        assert isinstance(rec["ki_lower_bounds"], dict), inn


def test_lower_bounds_separate_and_well_formed():
    n = 0
    for inn, rec in BINDING["drugs"].items():
        for gene, lb in rec["ki_lower_bounds"].items():
            n += 1
            assert gene in PANEL
            assert not (MEASURED_KEYS & set(lb)), f"{inn}/{gene}: measured field in a bound"
            assert 0 < lb["ki_lower_bound_nM"] <= lb["ki_lower_bound_max_nM"]
            assert lb["n_measurements"] >= lb["n_documents"] >= 1
            assert len(lb["documents"]) <= 10
            assert len(lb["documents"]) + len(lb.get("pdsp_references", [])) >= 1
            for r in lb.get("pdsp_references", []):
                assert r["source"] == "pdsp" and r["reference"]
            for d in lb["documents"]:
                assert d["document_chembl_id"].startswith("CHEMBL")
            # A bound never takes a gene out of searched_and_empty or into targets.
            assert (gene in rec["targets"]) != (gene in rec["searched_and_empty"])
    assert n > 0


def test_multi_source_merged_entries_consistent():
    n = 0
    for inn, rec in BINDING["drugs"].items():
        for gene, t in rec["targets"].items():
            if "sources" not in t:
                assert "gtopdb_references" not in t and "pdsp_references" not in t
                continue
            n += 1
            src = t["sources"]
            assert src["gtopdb"] + src["pdsp"] >= 1
            assert t["n_documents"] == src["chembl"] + src["gtopdb"] + src["pdsp"]
            assert len(t["documents"]) == min(src["chembl"], 10)
            assert len(t.get("gtopdb_references", [])) == min(src["gtopdb"], 10)
            assert len(t.get("pdsp_references", [])) == min(src["pdsp"], 10)
            earlier = {str(d["pubmed_id"]) for d in t["documents"] if d["pubmed_id"]}
            for r in t.get("gtopdb_references", []):
                assert r["source"] == "gtopdb"
                assert 3.0 <= r["pki"] <= 12.0
                assert r["pubmed_id"] is None or r["pubmed_id"] not in earlier
            earlier |= {r["pubmed_id"] for r in t.get("gtopdb_references", []) if r["pubmed_id"]}
            for r in t.get("pdsp_references", []):
                assert r["source"] == "pdsp" and r["reference"]
                assert 3.0 <= r["pki"] <= 12.0
                assert r["pubmed_id"] is None or r["pubmed_id"] not in earlier
            assert math.isclose(t["ki_nM"], 10 ** (9 - t["pki_median"]), rel_tol=1e-3)
    assert n > 0


def test_provenance_records_sources():
    prov = BINDING["provenance"]
    g = prov["gtopdb"]
    assert g["version"].startswith("GtoPdb Version")
    assert g["retrieved"]
    assert set(g["urls"]) == {"interactions.csv", "ligands.csv"}
    assert all(len(h) == 64 for h in g["sha256"].values())
    for key in ("merge_rule", "censored_handling"):
        assert prov[key]
    p = prov["pdsp"]
    assert p["url"].startswith("https://pdsp.unc.edu/") and p["retrieved"]
    assert len(p["sha256"]) == 64 and p["n_smiles_mapped"] > 0


# ------------------------------------------------------------ rules (synthetic)

def _act(doc, pki, nM=None):
    return {"document_chembl_id": doc, "pchembl_value": str(pki),
            "standard_value": str(nM if nM is not None else 10 ** (9 - pki))}


def test_merge_rule_median_of_publication_means_and_pmid_dedup():
    docs = {"CHEMBL_D1": {"pubmed_id": 111, "year": 2000},
            "CHEMBL_D2": {"pubmed_id": None, "year": 2001}}
    acts = [_act("CHEMBL_D1", 8.0), _act("CHEMBL_D1", 9.0), _act("CHEMBL_D2", 7.0)]
    gtop = [
        {"pki": 5.0, "pubmed_id": "111", "gtopdb_ligand_id": "1", "gtopdb_target_id": "9", "row": 0},
        {"pki": 6.0, "pubmed_id": "222", "gtopdb_ligand_id": "1", "gtopdb_target_id": "9", "row": 1},
        {"pki": 6.4, "pubmed_id": "222", "gtopdb_ligand_id": "1", "gtopdb_target_id": "9", "row": 2},
    ]
    t = vb.summarise_target(acts, docs, gtop)
    # groups: D1 mean 8.5, D2 7.0, PMID 222 mean 6.2 (PMID 111 dropped as dup)
    assert t["pki_median"] == 7.0
    assert t["n_documents"] == 3 and t["n_measurements"] == 5
    assert t["sources"] == {"chembl": 2, "gtopdb": 1, "pdsp": 0}
    assert "pdsp_references" not in t
    assert [r["pubmed_id"] for r in t["gtopdb_references"]] == ["222"]


def test_chembl_only_summary_unchanged_shape():
    t = vb.summarise_target([_act("CHEMBL_D1", 8.0)], {}, None)
    assert "sources" not in t and "gtopdb_references" not in t
    assert t["pki_median"] == 8.0 and t["n_documents"] == 1


def test_gtopdb_only_and_empty():
    g = [{"pki": 7.0, "pubmed_id": None, "gtopdb_ligand_id": "5", "gtopdb_target_id": "1", "row": 3}]
    t = vb.summarise_target([], {}, g)
    assert t["pki_median"] == 7.0 and t["documents"] == [] and t["sources"]["gtopdb"] == 1
    assert vb.summarise_target([], {}, []) == {}


def test_lower_bound_is_smallest_threshold_and_not_measured():
    rows = [{"document_chembl_id": "CHEMBL_A", "standard_value": "10000"},
            {"document_chembl_id": "CHEMBL_B", "standard_value": "1000"}]
    lb = vb.summarise_lower_bound(rows, {})
    assert lb["ki_lower_bound_nM"] == 1000.0 and lb["ki_lower_bound_max_nM"] == 10000.0
    assert lb["n_documents"] == 2
    assert not (MEASURED_KEYS & set(lb))


def test_censored_filter():
    base = {"standard_type": "Ki", "standard_units": "nM", "standard_value": "1000"}
    assert vb._censored({**base, "standard_relation": ">"})
    assert vb._censored({**base, "standard_relation": ">="})
    assert not vb._censored({**base, "standard_relation": "<"})
    assert not vb._censored({**base, "standard_relation": "="})
    assert not vb._censored({**base, "standard_relation": ">", "data_validity_comment": "x"})
    assert not vb._censored({**base, "standard_relation": ">", "standard_units": "ug/mL"})
    # censored rows never pass the measured filter
    assert not vb._usable({**base, "standard_relation": ">", "pchembl_value": "6"})


def test_build_drug_keeps_bound_only_gene_in_searched_and_empty():
    drug = {"inn": "x", "chembl_id": "CHEMBL_X"}
    tids = {"HRH1": "T1", "DRD2": "T2"}
    acts = [{**_act("CHEMBL_D", 8.0), "target_chembl_id": "T2", "standard_type": "Ki",
             "standard_relation": "=", "standard_units": "nM"}]
    cens = [{"document_chembl_id": "CHEMBL_C", "target_chembl_id": "T1", "standard_type": "Ki",
             "standard_relation": ">", "standard_units": "nM", "standard_value": "10000"}]
    row = vb.build_drug(drug, tids, acts, {}, cens, {})
    assert set(row["targets"]) == {"DRD2"}
    assert row["searched_and_empty"] == ["HRH1"]
    assert row["ki_lower_bounds"]["HRH1"]["ki_lower_bound_nM"] == 10000.0


def test_ligand_map_identifier_only_and_ambiguity_dropped():
    drugs = [{"inn": "a", "chembl_id": "CHEMBL1"}, {"inn": "b", "chembl_id": "CHEMBL2"}]
    keys = {"CHEMBL1": "KEY-A", "CHEMBL2": "KEY-B"}
    ligands = [
        {"Ligand ID": "10", "Name": "a", "ChEMBL ID": "", "InChIKey": ""},        # name only
        {"Ligand ID": "11", "Name": "zzz", "ChEMBL ID": "CHEMBL1", "InChIKey": ""},
        {"Ligand ID": "12", "Name": "zzz", "ChEMBL ID": "", "InChIKey": "KEY-B"},
        {"Ligand ID": "13", "Name": "zzz", "ChEMBL ID": "CHEMBL1", "InChIKey": "KEY-B"},
    ]
    m, amb = vb.gtopdb_ligand_map(ligands, drugs, keys)
    assert m == {"11": "a", "12": "b"}
    assert amb == ["13"]


def test_gtopdb_row_filter():
    base = {"Target Species": "Human", "Affinity Units": "pKi", "Ligand ID": "11",
            "Target UniProt ID": "P35367", "Affinity Median": "8.0",
            "Original Affinity Relation": "=", "PubMed ID": "123|456", "Target ID": "262"}
    rows = [base,
            {**base, "Original Affinity Relation": ""},
            {**base, "Original Affinity Relation": ">"},
            {**base, "Target Species": "Rat"},
            {**base, "Affinity Units": "pIC50"},
            {**base, "Target UniProt ID": "P35367|P14416"},
            {**base, "Ligand ID": "99"},
            {**base, "Affinity Median": ""}]
    out = vb.gtopdb_measurements(rows, {"11": "a"}, {"HRH1": "P35367"})
    assert list(out) == [("a", "HRH1")]
    assert [r["row"] for r in out[("a", "HRH1")]] == [0, 1]
    assert out[("a", "HRH1")][0]["pubmed_id"] == "123"


# ------------------------------------------------------------ PDSP (synthetic)

def _p(pki, pmid=None, ref="R"):
    return {"pki": pki, "ki_nM": 10 ** (9 - pki), "pubmed_id": pmid, "reference": ref,
            "row": 0, "key": f"PMID:{pmid}" if pmid else f"PDSP:{ref}"}


def test_pdsp_groups_dedup_against_chembl_and_gtopdb():
    docs = {"CHEMBL_D1": {"pubmed_id": 111, "year": 2000}}
    gtop = [{"pki": 6.0, "pubmed_id": "222", "gtopdb_ligand_id": "1",
             "gtopdb_target_id": "9", "row": 0}]
    pdsp = [_p(5.0, "111"), _p(5.0, "222"),           # same papers as ChEMBL / GtoPdb
            _p(7.0, None, "Smith 1999"), _p(7.4, None, "Smith 1999"), _p(9.0, "333")]
    t = vb.summarise_target([_act("CHEMBL_D1", 8.0)], docs, gtop, pdsp)
    # groups: 8.0 (ChEMBL), 6.0 (GtoPdb), 7.2 (Smith), 9.0 (PMID 333) -> median 7.6
    assert t["sources"] == {"chembl": 1, "gtopdb": 1, "pdsp": 2}
    assert t["pki_median"] == 7.6 and t["n_measurements"] == 5
    assert sorted(r["pubmed_id"] or r["reference"] for r in t["pdsp_references"]) == [
        "333", "Smith 1999"]


def test_pdsp_only_entry():
    t = vb.summarise_target([], {}, None, [_p(7.0)])
    assert t["pki_median"] == 7.0 and t["documents"] == []
    assert t["sources"] == {"chembl": 0, "gtopdb": 0, "pdsp": 1}
    assert "gtopdb_references" not in t


def test_pdsp_lower_bound_dedup_and_pool():
    docs = {"CHEMBL_A": {"pubmed_id": 111, "year": 2000}}
    rows = [{"document_chembl_id": "CHEMBL_A", "standard_value": "10000"}]
    lb = vb.summarise_lower_bound(rows, docs, [_p(5.0, "111"), _p(6.0, None, "X")])
    assert lb["ki_lower_bound_nM"] == 1000.0 and lb["ki_lower_bound_max_nM"] == 10000.0
    assert lb["n_documents"] == 2 and lb["n_measurements"] == 2
    assert [r["reference"] for r in lb["pdsp_references"]] == ["X"]
    assert not (MEASURED_KEYS & set(lb))
    assert vb.summarise_lower_bound([], {}, []) == {}


def test_pdsp_row_filter_and_structure_mapping():
    base = {"species": "HUMAN", "Unigene": "HRH1", "ki Note": "", "ki Val": "10",
            "SMILES": "S1", "Reference": "Ref A",
            "Link": "https://www.ncbi.nlm.nih.gov/pubmed/12345678"}
    rows = [base,
            {**base, "ki Note": ">", "ki Val": "10000"},
            {**base, "ki Note": "<"},
            {**base, "species": "? HUMAN"},
            {**base, "species": "RAT"},
            {**base, "Unigene": "KCNH2"},
            {**base, "ki Val": "2e18"},
            {**base, "ki Val": "abc"},
            {**base, "SMILES": ""},
            {**base, "SMILES": "S_OTHER"},
            {**base, "Link": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/"}]
    keys = {"S1": "KEY-A", "S_OTHER": "KEY-Z"}
    meas, cens, n = vb.pdsp_measurements(rows, {"KEY-A": "a"}, {"HRH1", "DRD2"},
                                         smiles_key=keys.get)
    assert set(meas) == {("a", "HRH1")} and set(cens) == {("a", "HRH1")}
    assert [r["row"] for r in meas[("a", "HRH1")]] == [0, 10]
    assert meas[("a", "HRH1")][0]["pubmed_id"] == "12345678"
    assert meas[("a", "HRH1")][1]["pubmed_id"] is None
    assert meas[("a", "HRH1")][1]["key"] == "PDSP:Ref A"
    assert math.isclose(meas[("a", "HRH1")][0]["pki"], 8.0)
    assert cens[("a", "HRH1")][0]["ki_nM"] == 10000.0 and n == 1


def test_smiles_inchikey_strips_salt_and_charge():
    base = vb.smiles_inchikey("CN(C)CCCN1c2ccccc2CCc2ccccc21")
    assert base and len(base) == 27
    assert vb.smiles_inchikey("CN(C)CCCN1c2ccccc2CCc2ccccc21.Cl") == base
    assert vb.smiles_inchikey("C[NH+](C)CCCN1c2ccccc2CCc2ccccc21") == base
    assert vb.smiles_inchikey("not a smiles") is None
