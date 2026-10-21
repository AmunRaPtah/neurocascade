"""v2 binding input table: schema, coverage, and PMID provenance."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "src" / "neurocascade" / "benchmark"
DRUGS = json.loads((BENCH / "v2" / "drugs.json").read_text())
BINDING = json.loads((BENCH / "v2" / "inputs" / "binding.json").read_text())

PANEL = sorted(BINDING["provenance"]["panel"])
DRUG_INNS = {d["inn"] for d in DRUGS["drugs"]}


def test_panel_includes_all_anchors():
    for gene in DRUGS["anchor_targets"]:
        assert gene in PANEL


def test_panel_resolved_every_target():
    panel = BINDING["provenance"]["panel"]
    failures = BINDING["provenance"]["panel_resolution_failures"]
    for gene, rec in panel.items():
        assert rec["uniprot"]
        if gene not in failures:
            assert rec["chembl_target_id"], f"{gene} has no resolved ChEMBL target id"


def test_every_drug_present():
    assert set(BINDING["drugs"]) == DRUG_INNS
    assert len(BINDING["drugs"]) == 331


def test_no_heldout_membership_recorded():
    """No per-drug held-out/split flag anywhere in the drugs table itself
    (provenance prose may discuss the policy; that is not membership data)."""
    blob = json.dumps(BINDING["drugs"])
    assert "heldout" not in blob.lower()
    assert '"split"' not in blob


def test_every_drug_full_panel_coverage():
    """Every panel target is either in targets (data) or searched_and_empty
    (explicitly absent); never missing from both, never in both."""
    for inn, rec in BINDING["drugs"].items():
        have = set(rec["targets"])
        empty = set(rec["searched_and_empty"])
        assert not (have & empty), f"{inn}: target(s) in both targets and searched_and_empty"
        assert have | empty == set(PANEL), f"{inn}: panel coverage incomplete"


def test_target_entries_well_formed_and_plausible():
    n_with_data = 0
    for inn, rec in BINDING["drugs"].items():
        for gene, t in rec["targets"].items():
            n_with_data += 1
            assert t["status"] == "data"
            assert 3.0 <= t["pki_median"] <= 12.0, f"{inn}/{gene}: implausible pKi {t['pki_median']}"
            assert t["ki_nM"] > 0
            assert t["ki_min_nM"] <= t["ki_nM"] * 1.01 or t["ki_min_nM"] <= t["ki_max_nM"]
            assert t["ki_min_nM"] <= t["ki_max_nM"]
            assert t["n_documents"] >= 1
            assert t["n_measurements"] >= t["n_documents"]
            assert len(t["documents"]) <= 10
            assert len(t["documents"]) <= t["n_documents"]
            for doc in t["documents"]:
                assert doc["document_chembl_id"].startswith("CHEMBL")
    assert n_with_data > 0


def test_kd_only_entries_well_formed():
    for inn, rec in BINDING["drugs"].items():
        for gene, kd in rec.get("kd_only", {}).items():
            assert gene in rec["searched_and_empty"]
            assert kd["n_measurements"] >= 1
            assert len(kd["documents"]) <= 10


def test_coverage_at_each_anchor_nonzero():
    """Every anchor receptor has at least one drug with a Ki (sanity the
    panel resolution and activity query actually pulled real data)."""
    for gene in DRUGS["anchor_targets"]:
        n = sum(1 for rec in BINDING["drugs"].values() if gene in rec["targets"])
        assert n >= 1, f"anchor {gene} has zero drugs with Ki data"


def test_every_pubmed_id_came_from_a_document_record():
    """Every pubmed_id in the file sits inside a {document_chembl_id,
    pubmed_id, year} document record, as produced by v2_binding.fetch_documents
    ; never a bare typed PMID with no backing ChEMBL document (P29.7)."""
    def walk(obj):
        if isinstance(obj, dict):
            if "pubmed_id" in obj:
                assert "document_chembl_id" in obj, f"pubmed_id without document_chembl_id: {obj}"
                assert obj["document_chembl_id"].startswith("CHEMBL")
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(BINDING["drugs"])


def test_provenance_fields_present():
    prov = BINDING["provenance"]
    for key in ("retrieved", "chembl", "panel", "aggregation_rule", "kd_handling",
                "no_data_handling", "held_out_note"):
        assert key in prov and prov[key]
    assert prov["chembl"]["release"]
