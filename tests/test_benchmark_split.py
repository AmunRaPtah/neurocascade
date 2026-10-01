"""P28.1: benchmark drug list and the frozen drug-level held-out split."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from neurocascade import benchmark
from neurocascade.benchmark import (
    DRUGS_PATH,
    SPLIT_PATH,
    SPLIT_SHA256,
    load_drugs,
    load_split,
    sha256_of,
)

ROOT = Path(__file__).resolve().parents[1]

# Pinned independently of the module constant: changing the split requires
# editing this line, the module and the ledger, which is the point.
FROZEN_SPLIT_SHA256 = "54a08acfe76b21237bd3152653184796c875f064c66408c9449980f9df6628d9"

STRATA = {"N05A", "N05B", "N06A", "FAIL"}
FIELDS = (
    "inn", "stratum", "chembl_id", "chembl_pref_name", "standard_inchikey",
    "cas", "atc_code", "atc_codes_all", "max_phase", "withdrawn_flag",
)


def _cas_check_digit_ok(cas: str) -> bool:
    m = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", cas)
    if not m:
        return False
    digits = (m.group(1) + m.group(2))[::-1]
    return sum((i + 1) * int(d) for i, d in enumerate(digits)) % 10 == int(m.group(3))


# --- the split is frozen ---------------------------------------------------

def test_split_hash_never_changes():
    assert sha256_of(SPLIT_PATH) == FROZEN_SPLIT_SHA256
    assert SPLIT_SHA256 == FROZEN_SPLIT_SHA256


def test_split_hash_recorded_in_ledger():
    ledger = (ROOT / "docs" / "validation-ledger.md").read_text()
    assert FROZEN_SPLIT_SHA256 in ledger


def test_load_split_refuses_edited_file(tmp_path, monkeypatch):
    edited = tmp_path / "split.json"
    edited.write_bytes(SPLIT_PATH.read_bytes().replace(b'"drawn"', b'"drawn" '))
    monkeypatch.setattr(benchmark, "SPLIT_PATH", edited)
    with pytest.raises(RuntimeError, match="may not be redrawn"):
        load_split()


# --- the split is a stratified drug-level partition ------------------------

def test_split_partitions_the_drug_list():
    split = load_split()
    train, held = set(split["train"]), set(split["heldout"])
    assert len(train) == len(split["train"]) and len(held) == len(split["heldout"])
    assert not train & held
    assert train | held == {d["inn"] for d in load_drugs()}
    assert (len(train), len(held)) == (130, 55)


def test_split_is_stratified_70_30_by_class():
    split = load_split()
    stratum = {d["inn"]: d["stratum"] for d in load_drugs()}
    held = Counter(stratum[i] for i in split["heldout"])
    total = Counter(stratum.values())
    assert dict(total) == {"N05A": 67, "N05B": 35, "N06A": 66, "FAIL": 17}
    assert dict(held) == {"N05A": 20, "N05B": 10, "N06A": 20, "FAIL": 5}
    for s, n in total.items():
        assert held[s] == round(0.3 * n)
    recorded = json.loads(SPLIT_PATH.read_text())["strata_counts"]
    assert {s: c["heldout"] for s, c in recorded.items()} == dict(held)


# --- the drug list ----------------------------------------------------------

def test_at_least_120_drugs_with_unique_keys():
    drugs = load_drugs()
    assert len(drugs) == 185
    assert len({d["inn"] for d in drugs}) == len(drugs)
    assert len({d["chembl_id"] for d in drugs}) == len(drugs)
    assert {d["stratum"] for d in drugs} == STRATA


def test_every_field_has_a_source():
    for d in load_drugs():
        for f in FIELDS:
            assert f in d, (d["inn"], f)
            assert d["sources"].get(f), (d["inn"], f)


def test_identifier_formats():
    for d in load_drugs():
        assert re.fullmatch(r"CHEMBL\d+", d["chembl_id"]), d["inn"]
        assert d["cas"] is not None and _cas_check_digit_ok(d["cas"]), d["inn"]
        if d["stratum"] == "FAIL":
            assert d["atc_code"] is None
        else:
            assert d["atc_code"].startswith(d["stratum"]), d["inn"]
            assert re.fullmatch(r"N0[56][AB][A-Z]\d{2}", d["atc_code"])


def test_cas_match_quality_is_declared():
    matches = Counter(d["sources"]["cas"]["match"] for d in load_drugs())
    assert matches == {"inchikey": 180, "connectivity": 2, "name_only": 3}
    for d in load_drugs():
        if d["sources"]["cas"]["match"] != "inchikey":
            assert d["sources"]["cas"]["note"], d["inn"]


def test_known_identifiers():
    # Spot checks against ChEMBL/PubChem records this repository already uses.
    by = {d["inn"]: d for d in load_drugs()}
    assert by["haloperidol"]["chembl_id"] == "CHEMBL54"
    assert by["haloperidol"]["cas"] == "52-86-8"
    assert by["haloperidol"]["atc_code"] == "N05AD01"
    assert by["clozapine"]["chembl_id"] == "CHEMBL42"
    assert by["diazepam"]["atc_code"] == "N05BA01"
    assert by["pipotiazine"]["chembl_id"] == "CHEMBL398880"  # not the palmitate


def test_failures_cite_a_missed_primary_endpoint():
    fails = [d for d in load_drugs() if d["stratum"] == "FAIL"]
    assert len(fails) == 17
    for d in fails:
        f = d["failure"]
        assert isinstance(f["pmid"], int) and f["pmid"] > 0
        assert f["intended_atc_level3"] in {"N05A", "N05B", "N06A"}
        assert f["citation"] and f["evidence"] and f["indication"]


def test_excluded_atc_entries_are_recorded():
    excluded = json.loads(DRUGS_PATH.read_text())["excluded_atc_entries"]
    assert {e["atc_code"] for e in excluded} == {
        "N05AH53", "N05BA56", "N05BB51", "N05BC51", "N06AX62",
        "N05BX05", "N06AX25", "N05AN01",
    }
    assert all(e["reason"] for e in excluded)
