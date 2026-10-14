"""P29.7: PDSP link identifiers that were stored as PMIDs, repaired.

The P28.2 builder read any 6-9 digit run out of PDSP's ``Link`` URL as a PMID.
Eleven such numbers were not PMIDs (a ScienceDirect alert id, PII fragments,
PMC ids, publisher article numbers): two do not exist in PubMed, three exist
only with leading zeros stripped (as unrelated papers), and six resolve to
unrelated papers. These tests pin
the repair offline; the PubMed check itself is ``scripts/verify_pmids.py``.
"""

import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from neurocascade.benchmark import BINDING_DIR, load_binding_panel, load_drugs

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "src" / "neurocascade" / "benchmark"

# the numbers P28.2 stored as PMIDs, and the PMID each link really points at
# (None: Seeman 2001, Clin Neurosci Res, not indexed in PubMed)
REPAIRED = {
    "93617739": {None},
    "096808960": {"16517171"},
    "001429990": {"11513838"},
    "022352341": {"29407591"},
    "0960894": {"16621538", "30266542", "29730027"},
    "13543784": {"12665420"},
    "4119213": {"25026323"},
    "430248": {"26682068"},
    "5313063": {"28239660"},
    "6500342": {"16314884"},
    "908643": {"16082416"},
}


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_binding", ROOT / "scripts" / "build_benchmark_binding.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def panels():
    return {d["inn"]: load_binding_panel(d["inn"]) for d in load_drugs()}


def test_only_pubmed_links_yield_a_pmid():
    mod = _script()
    assert mod._pmid_from_link(
        "http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&db=pubmed"
        "&dopt=Abstract&list_uids=9537821") == "9537821"
    assert mod._pmid_from_link(
        "http://www.ncbi.nlm.nih.gov:80/entrez/query.fcgi?cmd=Retrieve&db=PubMed"
        "&list_uids=8935801&dopt=Abstract") == "8935801"
    assert mod._pmid_from_link("https://www.ncbi.nlm.nih.gov/pubmed/7654321") == "7654321"
    assert mod._pmid_from_link("https://www.ncbi.nlm.nih.gov/pubmed/?term=30768942") == "30768942"
    assert mod._pmid_from_link("https://pubmed.ncbi.nlm.nih.gov/33831560/") == "33831560"
    for link in (
        "http://www.sciencedirect.com/science?_ob=ArticleURL&_udi=B6W84-42KD8D9-7"
        "&_coverDate=01%2F31%2F2001&_alid=93617739&_rdoc=1",
        "https://www.sciencedirect.com/science/article/pii/S0014299901011888?via%3Dihub",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5313063/pdf/jciinsight-2-91456.pdf",
        "http://www.hindawi.com/journals/ijmc/2015/430248/",
        "http://www.tandfonline.com/doi/abs/10.1517/13543784.12.4.655",
        "http://www.nature.com/tpj/journal/v6/n1/pdf/6500342a.pdf",
        "https://journals.prous.com/journals/servlet/xmlxsl/pk_journals.xml_summaryn_pr"
        "?p_JournalId=6&p_RefId=908643",
        "CMD=search&DB=pubmed",
        "",
    ):
        assert mod._pmid_from_link(link) is None, link


def test_link_table_maps_each_bad_id_to_its_verified_paper():
    mod = _script()
    got = defaultdict(set)
    for rec in mod.PDSP_LINK_SOURCES.values():
        got[rec["pdsp_id"]].add(rec["pmid"])
        assert rec["doi"] and rec["citation"]
    assert dict(got) == REPAIRED


def _pdsp_row(link, reference, number="9"):
    return {" Ligand Name": "TESTDRUG", "species": "HUMAN", "Name": "DOPAMINE D2",
            "ki Note": "", "ki Val": "4.0", "Link": link, "Reference": reference,
            "Number": number, "source": "CLONED", "Hotligand": "x"}


def test_builder_resolves_a_non_pubmed_link_and_keeps_the_pdsp_id():
    mod = _script()
    rows = [
        _pdsp_row("https://journals.prous.com/journals/servlet/xmlxsl/pk_journals."
                  "xml_summaryn_pr?p_JournalId=6&p_RefId=908643",
                  "Silvestre, JS. et, al. 2005", "1"),
        _pdsp_row("http://www.sciencedirect.com/science?_ob=ArticleURL&_udi=B6W84-42KD8D9-7"
                  "&_coverDate=01%2F31%2F2001&_alid=93617739", "Seeman, Philip  2001", "2"),
    ]
    by_name = {"testdrug": rows}
    out = mod._build_one({"inn": "testdrug", "stratum": "N05A", "chembl_id": "CHEMBL1",
                          "standard_inchikey": "X"},
                         [], {}, {}, {}, [], by_name, {}, {}, defaultdict(list))
    ms = {m["record"].split()[3]: m for m in out["targets"]["DRD2"]["measurements"]}
    assert ms["1"]["ref"] == "PMID:16082416" and ms["1"]["pdsp_id"] == "908643"
    assert out["references"]["PMID:16082416"]["pmid"] == "16082416"
    assert out["references"]["PMID:16082416"]["doi"] == "10.1358/mf.2005.27.5.908643"
    seeman = out["references"][ms["2"]["ref"]]
    assert ms["2"]["ref"] == "PDSP_REF:Seeman, Philip  2001"
    assert seeman["pmid"] is None and seeman["pdsp_id"] == "93617739"
    assert seeman["source"] == "PDSP Ki database"
    assert seeman["doi"] == "10.1016/S1566-2772(00)00007-4"
    assert out["targets"]["DRD2"]["n_references"] == 2


def test_no_benchmark_json_carries_a_repaired_id_as_a_pmid():
    for path in BENCH.rglob("*.json"):
        text = path.read_text()
        for bad in REPAIRED:
            assert f"PMID:{bad}\"" not in text, (path.name, bad)
            assert f"\"pmid\": \"{bad}\"" not in text, (path.name, bad)
        # a PMID never starts with 0: zero-padding was the mark of a PII fragment
        assert not re.search(r"\"pmid\": \"0", text), path.name


def test_repaired_measurements_point_at_the_verified_paper(panels):
    n = 0
    seen = defaultdict(set)
    for inn, p in panels.items():
        for t in p["targets"].values():
            for m in t["measurements"]:
                if "pdsp_id" not in m:
                    continue
                n += 1
                assert m["source"] == "PDSP"
                ref = p["references"][m["ref"]]
                assert ref["pmid"] in REPAIRED[m["pdsp_id"]], (inn, m)
                seen[m["pdsp_id"]].add(inn)
    # 684 PDSP rows across 32 drugs carried one of the eleven bad ids
    assert n == 684
    assert set(seen) == set(REPAIRED)
    assert len(set().union(*seen.values())) == 32


def test_seeman_2001_is_cited_through_the_pdsp_not_a_pmid(panels):
    hal = panels["haloperidol"]
    ref = hal["references"]["PDSP_REF:Seeman, Philip  2001"]
    assert ref["pmid"] is None and ref["pdsp_id"] == "93617739"
    assert "Clin Neurosci Res 2001" in ref["citation"]


def test_values_checked_against_the_real_paper(panels):
    # Gassaway 2014 abstract: tianeptine human MOR Ki 383 +/- 183 nM
    mor = [m for m in panels["tianeptine"]["targets"]["OPRM1"]["measurements"]
           if m.get("pdsp_id") == "4119213"]
    assert [m["ki_nM"] for m in mor] == [383.0]
    # Yarravarapu 2018 abstract: (S)-citalopram hSERT Ki 1.77 nM
    sert = [m for m in panels["escitalopram"]["targets"]["SLC6A4"]["measurements"]
            if m.get("pdsp_id") == "0960894"]
    assert [(m["ki_nM"], m["ref"]) for m in sert] == [(1.77, "PMID:30266542")]


def test_double_counted_papers_now_count_once(panels):
    # Zajdel 2018 reached ziprasidone through ChEMBL (PMID:29407591) and
    # through the PDSP under the fake id 022352341, as two references.
    t = panels["ziprasidone"]["targets"]["HTR1A"]
    assert t["n_references"] == 12
    assert t["ki_nM"] == pytest.approx(3.162278)


def test_a_reference_not_naming_its_drug_says_why(panels):
    txt = json.loads((BINDING_DIR / "olanzapine.json").read_text())
    note = txt["references"]["PMID:16082416"]["pmid_note"]
    assert "25 typical and atypical antipsychotics" in note
    assert "PDSP" in note
