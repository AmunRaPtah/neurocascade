"""P29.7: PDSP Link digits are not PMIDs; repaired binding citations."""

import copy
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest

from neurocascade.benchmark import BINDING_DIR, load_binding_panel

ROOT = Path(__file__).resolve().parents[1]

#: The digit runs the pre-P29.7 builder stored as PMIDs, read off PDSP Links.
FORMER_IDS = {"001429990", "022352341", "0960894", "096808960", "13543784", "4119213",
              "430248", "5313063", "6500342", "908643", "93617739"}

#: PDSP Reference -> PMID, each matched on DOI/pii and checked against PubMed
#: esummary title and authors on 2026-10-15.
RESOLVED = {
    "Schmidt, AW. et, al. 2001": "11513838",
    "Zajdel, Pawel et. al, 2018": "29407591",
    "Ablordeppey S, et al., 2006": "16621538",
    "Panczyk K, et al., 2018": "29730027",
    "Yarravarapu N, et. al., 2018": "30266542",
    "Van der Mey. et, al. 2006": "16517171",
    "Keck, Paul e. et. al, 2003": "12665420",
    "Gassaway, MM.  et. al. 2014": "25026323",
    "Clayton, Terry et. al. 2015": "26682068",
    "Song, Gang et., al.  2017": "28239660",
    "Davies M, et al. 2005": "16314884",
    "Silvestre, JS. et, al. 2005": "16082416",
    "Seeman, Philip  2001": None,  # Clin Neurosci Res 1:53, not in PubMed
}


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_binding", ROOT / "scripts" / "build_benchmark_binding.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panels():
    return [json.loads(p.read_text()) for p in sorted(BINDING_DIR.glob("*.json"))
            if p.name != "coverage.json"]


@pytest.mark.parametrize("link", [
    "https://www.sciencedirect.com/science/article/pii/S0014299901011888?via%3Dihub",
    "http://ac.els-cdn.com/S0960894X06003465/1-s2.0-S0960894X06003465-main.pdf",
    "http://www.hindawi.com/journals/ijmc/2015/430248/",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5313063/pdf/jciinsight-2-91456.pdf",
    "https://journals.prous.com/journals/servlet/xmlxsl/pk_journals.xml_summaryn_pr?p_JournalId=6&p_RefId=908643",
    "http://www.tandfonline.com/doi/abs/10.1517/13543784.12.4.655",
    "", "http://pdsp.med.unc.edu/pdsp.php",
])
def test_non_pubmed_links_yield_no_pmid(link):
    assert _script()._pmid_from_link(link) is None


@pytest.mark.parametrize("link,pmid", [
    ("http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=9577836&dopt=Abstract", "9577836"),
    ("http://www.ncbi.nlm.nih.gov:80/entrez/query.fcgi?cmd=Retrieve&db=PubMed&list_uids=9454790", "9454790"),
    ("https://www.ncbi.nlm.nih.gov/pubmed/6086881", "6086881"),
    ("https://www.ncbi.nlm.nih.gov/pubmed/?term=30768942", "30768942"),
    ("https://pubmed.ncbi.nlm.nih.gov/31586817/", "31586817"),
])
def test_pubmed_links_still_yield_their_pmid(link, pmid):
    assert _script()._pmid_from_link(link) == pmid


def test_curated_papers_resolve_and_keep_the_former_id():
    mod = _script()
    assert {k: v["pmid"] for k, v in mod.PDSP_PAPERS.items()} == RESOLVED
    assert {v["pdsp_id"] for v in mod.PDSP_PAPERS.values()} == FORMER_IDS
    key, rec = mod._pdsp_ref("Schmidt, AW. et, al. 2001 ", "https://x.org/S0014299901011888", "clozapine")
    assert key == "PMID:11513838" and rec["pmid"] == "11513838"
    assert rec["pdsp_id"] == "001429990" and rec["doi"] == "10.1016/s0014-2999(01)01188-8"
    assert "clozapine is not named" in rec["pmid_note"]
    # the drug the abstract names needs no note
    assert "pmid_note" not in mod._pdsp_ref("Schmidt, AW. et, al. 2001", "", "ziprasidone")[1]
    key, rec = mod._pdsp_ref("Seeman, Philip  2001", "http://www.sciencedirect.com/x&_alid=93617739", "haloperidol")
    assert key == "PDSP_REF:Seeman, Philip  2001" and rec["pmid"] is None
    assert rec["doi"] == "10.1016/S1566-2772(00)00007-4" and rec["pdsp_id"] == "93617739"
    # an unknown non-PubMed link stays a PDSP reference string, never a PMID
    assert mod._pdsp_ref("X 1999", "http://www.hindawi.com/1234567/", "x") == (
        "PDSP_REF:X 1999", {"pmid": None, "doi": None, "citation": "X 1999"})


def test_committed_panels_carry_no_link_digits_as_pmids():
    n_pdsp_id = 0
    for p in _panels():
        for key, ref in p["references"].items():
            assert ref["pmid"] not in FORMER_IDS, (p["inn"], key)
            assert key[5:] not in FORMER_IDS, (p["inn"], key)
            if ref["pmid"]:
                assert ref["pmid"].isdigit() and not ref["pmid"].startswith("0"), (p["inn"], key)
            if "pdsp_id" in ref:
                n_pdsp_id += 1
                assert RESOLVED[ref["pdsp_reference"]] == ref["pmid"]
                assert ref["pdsp_id"] in FORMER_IDS
    # one record per (drug, paper): 71 drug-paper pairs across 32 drug files
    assert n_pdsp_id == 71


def test_repaired_citations_note_drugs_their_abstract_does_not_name():
    mod = _script()
    for p in _panels():
        for key, ref in p["references"].items():
            if "pdsp_reference" not in ref:
                continue
            paper = mod.PDSP_PAPERS[ref["pdsp_reference"]]
            if p["inn"] not in paper["named"] or ref["pmid"] is None:
                assert ref.get("pmid_note"), (p["inn"], key)



def test_every_citation_of_a_repaired_pmid_names_or_notes_the_drug():
    # Not only PDSP rows: ChEMBL cites Panczyk 2018 (PMID 29730027) for
    # hydroxyzine, which its title, abstract and MeSH do not name.
    mod = _script()
    by_pmid = {v["pmid"]: v for v in mod.PDSP_PAPERS.values() if v["pmid"]}
    cited = defaultdict(set)
    for p in _panels():
        for key, ref in p["references"].items():
            paper = by_pmid.get(ref.get("pmid"))
            if paper is None:
                continue
            cited[ref["pmid"]].add(p["inn"])
            if p["inn"] not in paper["named"]:
                assert ref.get("pmid_note"), (p["inn"], key)
    assert set(cited) == set(by_pmid)
    assert "hydroxyzine" in cited["29730027"]
    assert {"clozapine", "haloperidol", "olanzapine", "quetiapine",
            "risperidone"} <= cited["11513838"]
    assert "clozapine" in cited["16621538"]


def test_note_pass_covers_a_chembl_citation_of_a_repaired_paper():
    refs = {"PMID:29730027": {"pmid": "29730027", "chembl_document": "CHEMBL4190389"},
            "PMID:11513838": {"pmid": "11513838"}}
    _script()._note_unnamed_papers(refs, "hydroxyzine")
    assert "hydroxyzine is not named" in refs["PMID:29730027"]["pmid_note"]
    assert "N-(phenoxyalkyl)piperazines" in refs["PMID:29730027"]["pmid_note"]
    named = {"PMID:11513838": {"pmid": "11513838"}}
    _script()._note_unnamed_papers(named, "ziprasidone")
    assert "pmid_note" not in named["PMID:11513838"]

def test_seeman_2001_and_silvestre_2005_are_attributed_to_their_papers():
    hal = load_binding_panel("haloperidol")
    seeman = [m for m in hal["targets"]["DRD2"]["measurements"]
              if m["ref"] == "PDSP_REF:Seeman, Philip  2001"]
    assert seeman and all(m["source"] == "PDSP" for m in seeman)
    assert "Compilation" in hal["references"]["PMID:16082416"]["pmid_note"]
    n = sum(1 for p in _panels() if "PMID:16082416" in p["references"])
    assert n == 25


def test_relink_merges_a_paper_cited_by_two_sources_and_keeps_values():
    mod = _script()
    link = "https://www.sciencedirect.com/science/article/pii/S0223523418300023?via%3Dihub#!"
    rows = {"7": {"Reference": "Zajdel, Pawel et. al, 2018", "Link": link},
            "8": {"Reference": "B 1990", "Link": "https://www.ncbi.nlm.nih.gov/pubmed/7654321"}}
    ms = [{"ki_nM": 10.0, "relation": "=", "source": "PDSP", "ref": "PMID:022352341",
           "record": "PDSP Ki ID 7 (CLONED, hot ligand x)"},
          {"ki_nM": 10.0, "relation": "=", "source": "ChEMBL", "ref": "PMID:29407591",
           "record": "activity 1 / assay A"},
          {"ki_nM": 1000.0, "relation": "=", "source": "PDSP", "ref": "PMID:7654321",
           "record": "PDSP Ki ID 8 (CLONED, hot ligand x)"}]
    panel = {"inn": "haloperidol", "actions_without_ki": {},
             "targets": {"DRD2": {"actions": [], **mod._summarise(ms), "measurements": ms}},
             "references": {"PMID:022352341": {"pmid": "022352341", "doi": None, "citation": "Z"},
                            "PMID:29407591": {"pmid": "29407591", "doi": "d", "citation": "Zajdel P",
                                              "chembl_document": "CHEMBL4229332"},
                            "PMID:7654321": {"pmid": "7654321", "doi": None, "citation": "B 1990"}}}
    assert panel["targets"]["DRD2"]["n_references"] == 3
    before = copy.deepcopy(panel)
    out = mod.relink_pdsp(panel, rows)
    t = out["targets"]["DRD2"]
    assert sorted(m["ki_nM"] for m in t["measurements"]) == [10.0, 10.0, 1000.0]
    assert t["n_references"] == 2 and t["ki_nM"] == pytest.approx(100.0)
    assert set(out["references"]) == {"PMID:29407591", "PMID:7654321"}
    z = out["references"]["PMID:29407591"]
    # the ChEMBL record keeps its fields; PDSP adds what it lacked
    assert z["citation"] == "Zajdel P" and z["chembl_document"] == "CHEMBL4229332"
    assert z["pdsp_id"] == "022352341" and "haloperidol is not named" in z["pmid_note"]
    assert out["references"]["PMID:7654321"] == before["references"]["PMID:7654321"]
    # idempotent
    assert mod.relink_pdsp(copy.deepcopy(out), rows) == out


def test_build_merges_pdsp_fields_into_an_existing_chembl_reference():
    mod = _script()
    target = {"CHEMBL_T1": {"target_type": "SINGLE PROTEIN", "pref_name": "D2",
                            "organism": "Homo sapiens", "target_components": [
                                {"accession": "P14416", "target_component_synonyms": [
                                    {"syn_type": "GENE_SYMBOL", "component_synonym": "DRD2"}]}]}}
    docs = {"DOC1": {"pubmed_id": 29407591, "doi": "10.1016/j.ejmech.2018.01.002",
                     "authors": "Zajdel P", "year": 2018, "journal": "J", "volume": "1",
                     "first_page": "1"}}
    act = {"activity_id": 1, "assay_chembl_id": "A", "target_chembl_id": "CHEMBL_T1",
           "document_chembl_id": "DOC1", "standard_value": "4", "standard_units": "nM",
           "standard_relation": "=", "data_validity_comment": None, "potential_duplicate": 0,
           "activity_comment": None, "standard_text_value": None}
    pdsp = {" Ligand Name": "testdrug", "species": "HUMAN", "Name": "DOPAMINE D2",
            "ki Note": "", "ki Val": "4", "Number": "1", "source": "CLONED", "Hotligand": "x",
            "Reference": "Zajdel, Pawel et. al, 2018",
            "Link": "https://www.sciencedirect.com/science/article/pii/S0223523418300023"}
    by_name = defaultdict(list, {"testdrug": [pdsp]})
    out = mod._build_one({"inn": "testdrug", "stratum": "N05A", "chembl_id": "CHEMBL1",
                          "standard_inchikey": "X"}, [act],
                         {"A": {"assay_tax_id": 9606, "confidence_score": 9}}, docs, target,
                         [], by_name, {}, {}, defaultdict(list))
    t = out["targets"]["DRD2"]
    assert t["n_measurements"] == 2 and t["n_references"] == 1
    ref = out["references"]["PMID:29407591"]
    assert ref["citation"].startswith("Zajdel P") and ref["pdsp_id"] == "022352341"
