"""2026-10-26 provenance corrections to the frozen v2 human PK truth.

human_pk.json stays frozen; truth/human_pk_corrections.json removes values
whose source is not human in-vivo IV PK (microsomal, organ-partial, unbound
volume, non-IV route, analogue/other-compound papers), and the v2 loader
applies it."""

import importlib.util
import json
from pathlib import Path

import pytest

from neurocascade.benchmark.v2 import (
    apply_human_pk_corrections,
    load_v2_human_pk,
    load_v2_human_pk_corrections,
)

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "src" / "neurocascade" / "benchmark" / "v2"
CORRECTIONS_FILE = V2_DIR / "truth" / "human_pk_corrections.json"
FIELDS = ("clearance_iv", "vss")


@pytest.fixture(scope="module")
def corrections():
    return json.loads(CORRECTIONS_FILE.read_text())


@pytest.fixture(scope="module")
def v2_pk():
    spec = importlib.util.spec_from_file_location("v2_pk", ROOT / "scripts" / "v2_pk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows_in_this_checkout():
    """Session trees strip some rows (and may remove the corrections file)."""
    return load_v2_human_pk(corrected=False)


def test_corrections_schema(corrections):
    assert corrections["version"] == 1
    assert corrections["date"] == "2026-10-26"
    assert isinstance(corrections["rule"], str) and len(corrections["rule"]) > 200
    for inn, entry in corrections["drop"].items():
        assert entry["fields"] and set(entry["fields"]) <= set(FIELDS)
        assert entry["reason"] and entry["evidence"]
    for inn, fields in corrections["replace"].items():
        for field, e in fields.items():
            assert field in FIELDS
            assert e["min"] <= e["value"] <= e["max"]
            assert e["n_measurements"] >= 1 and e["n_documents"] >= 1
            assert e["sources"] and all(s.get("pmid") for s in e["sources"])
            assert e["excluded"], (inn, field, "replace without an excluded source")
            assert e["unit"] == {"clearance_iv": "mL/min/kg", "vss": "L/kg"}[field]


def test_corrections_do_not_name_a_partition(corrections):
    text = json.dumps(corrections).lower()
    for word in ("heldout", "held_out", "held-out"):
        assert word not in text


def test_microsomal_clearance_is_gone(corrections):
    raw = _rows_in_this_checkout()
    pk = load_v2_human_pk()
    for inn, assay in (("milnacipran", "CHEMBL948481"), ("melatonin", "CHEMBL2327999")):
        assert assay in corrections["drop"][inn]["evidence"]
        assert "in_vitro" in corrections["drop"][inn]["reason"]
        if inn in raw:
            assert raw[inn]["clearance_iv"] is not None  # frozen file untouched
            assert pk[inn]["clearance_iv"] is None


def test_analogue_paper_sources_are_removed(corrections):
    pk = load_v2_human_pk()
    excluded = {e["pmid"] for e in corrections["replace"]["propafenone"]["clearance_iv"]["excluded"]}
    assert {"21955244", "22708838"} <= excluded
    if "propafenone" in pk:
        kept = {s["pmid"] for s in pk["propafenone"]["clearance_iv"]["sources"]}
        assert not kept & {"21955244", "22708838"}
    assert "PMID:34329999" in corrections["drop"]["tetrabenazine"]["evidence"]


def test_loader_applies_every_correction(corrections):
    raw = _rows_in_this_checkout()
    pk = load_v2_human_pk()
    assert set(pk) == set(raw)
    for inn, entry in corrections["drop"].items():
        if inn in pk:
            for field in entry["fields"]:
                assert pk[inn][field] is None
    for inn, fields in corrections["replace"].items():
        if inn in pk:
            for field, e in fields.items():
                assert pk[inn][field]["value"] == e["value"]
    touched = set(corrections["drop"]) | set(corrections["replace"])
    for inn in set(raw) - touched:
        assert pk[inn] == raw[inn]


def test_apply_skips_absent_rows_and_does_not_mutate():
    rows = {"a": {"inn": "a", "clearance_iv": {"value": 1.0}, "vss": None}}
    corr = {"drop": {"a": {"fields": ["clearance_iv"]}, "b": {"fields": ["vss"]}},
            "replace": {"c": {"vss": {"value": 2.0}}}}
    out = apply_human_pk_corrections(rows, corr)
    assert set(out) == {"a"} and out["a"]["clearance_iv"] is None
    assert rows["a"]["clearance_iv"] == {"value": 1.0}
    assert apply_human_pk_corrections(rows, None) is rows
    assert load_v2_human_pk_corrections() is not None


@pytest.mark.parametrize("title,inn", [
    ("Characterization of thien-2-yl 1S,2R-milnacipran analogues as potent "
     "norepinephrine/serotonin transporter inhibitors for the treatment of "
     "neuropathic pain.", "milnacipran"),
    ("Optimization of propafenone analogues as antimalarial leads.", "propafenone"),
    ("Synthesis and analysis of dihydrotetrabenazine derivatives as novel "
     "vesicular monoamine transporter 2 inhibitors.", "tetrabenazine"),
    ("Interaction study of the combined use of paroxetine and "
     "fosamprenavir-ritonavir in healthy subjects.", "paroxetine"),
    ("Pharmacokinetic and pharmacodynamic properties of cholinesterase "
     "inhibitors donepezil, tacrine, and galantamine in aged and young Lister "
     "hooded rats.", "donepezil"),
])
def test_title_fallback_rejects_wrong_entity_papers(v2_pk, title, inn):
    assert not v2_pk._title_names_drug(title, inn)


@pytest.mark.parametrize("title,inn", [
    ("Pharmacokinetics of haloperidol in psychotic patients.", "haloperidol"),
    ("Disposition of valproic acid in man.", "valproic acid"),
    ("Pharmacokinetics of single and multiple doses of phenytoin in man.", "phenytoin"),
])
def test_title_fallback_keeps_human_pk_papers(v2_pk, title, inn):
    assert v2_pk._title_names_drug(title, inn)


def test_assay_rule(v2_pk):
    micro = {"bao_format": "BAO_0000251", "assay_organism": "Homo sapiens",
             "assay_tissue": "Liver", "assay_subcellular_fraction": "Microsome"}
    assert v2_pk._assay_exclusion(
        "invivo_cl", micro, "Intrinsic clearance in human liver microsomes") == "in_vitro"
    organism = {"bao_format": "BAO_0000218", "assay_organism": "Homo sapiens"}
    assert v2_pk._assay_exclusion(
        "invivo_cl", organism, "Clearance in human after iv administration") is None
    assert v2_pk._assay_exclusion(
        "invivo_cl", dict(organism, assay_tissue="Liver"),
        "Hepatic clearance in human") == "partial_clearance"
    assert v2_pk._assay_exclusion(
        "invivo_vss", organism,
        "Unbound volume of distribution at steady state in human") == "unbound_volume"
    assert v2_pk._assay_exclusion(
        "invivo_cl", organism, "Clearance in human at 2 mg, po single dose") == "non_iv_route"
    assert v2_pk._assay_exclusion(
        "invivo_vss", dict(organism, assay_organism="Rattus norvegicus"),
        "Volume of distribution after iv") == "non_human"
