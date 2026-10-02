"""Build the Phase 28 benchmark drug list and its frozen held-out split (P28.1).

Two stages, run separately on purpose:

``python scripts/build_benchmark.py drugs``
    Network. Assembles ``src/neurocascade/benchmark/drugs.json`` from
    (a) every WHO ATC level-5 entry under N05A (antipsychotics), N05B
    (anxiolytics) and N06A (antidepressants), read through the ChEMBL REST
    API's ``atc_class`` endpoint, and (b) the hand-curated clinical-failure
    list ``FAILURES`` below, each backed by the PMID of a published trial
    whose primary endpoint was not met. ChEMBL ids come from ChEMBL's own
    ATC-to-molecule mapping (parent form); CAS numbers come from PubChem
    PUG-View, resolved by the parent's standard InChIKey, with the
    depositing source named per drug.

``python scripts/build_benchmark.py split``
    Offline. Draws the stratified ~70/30 drug-level split from drugs.json
    and writes ``split.json``. **Do not re-run once split.json is
    committed**: the split is frozen and its sha256 is pinned in
    ``neurocascade.benchmark.SPLIT_SHA256``, in a test, and in
    docs/validation-ledger.md. This stage is kept for the record of how it
    was drawn, not for redrawing.

The reasoning behind every rule here is in docs/architecture.md, section
"P28.1: benchmark drug list and frozen split".
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src" / "neurocascade" / "benchmark"

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest"

ATC_LEVEL3 = ("N05A", "N05B", "N06A")

# ATC level-5 entries excluded, with the reason. Mechanical rule: the
# benchmark wants one small-molecule entity per record, so fixed
# combinations, herbal preparations and an inorganic ion are out.
ATC_EXCLUDED = {
    "N05AH53": "fixed combination (olanzapine and samidorphan); olanzapine is in via N05AH03",
    "N05BA56": "fixed combination (lorazepam, combinations); lorazepam is in via N05BA06",
    "N05BB51": "fixed combination (hydroxyzine, combinations); hydroxyzine is in via N05BB01",
    "N05BC51": "fixed combination (meprobamate, combinations); meprobamate is in via N05BC01",
    "N06AX62": "fixed combination (bupropion and dextromethorphan); bupropion is in via N06AX12",
    "N05BX05": "herbal preparation (lavender oil), not a single molecular entity",
    "N06AX25": "herbal preparation (St John's wort), not a single molecular entity",
    "N05AN01": "lithium: inorganic ion with no receptor binding panel, outside the "
    "receptor-occupancy premise the benchmark scores",
}

# ChEMBL's ATC mapping points at a form that is not the active parent.
CHEMBL_OVERRIDE = {
    "N05AC04": (
        "CHEMBL398880",
        "ChEMBL maps N05AC04 to CHEMBL2106439 PIPOTIAZINE PALMITATE, a depot "
        "ester registered as its own parent; the active moiety pipotiazine is "
        "CHEMBL398880 (ChEMBL molecule search 'pipotiazine')",
    ),
}

# Why a CAS match is weaker than "inchikey" for these drugs, read off the two
# records' structures (ChEMBL molecule_structures vs the PubChem name hit).
CAS_NOTES = {
    "chlordiazepoxide": "ChEMBL CHEMBL451 draws the 2-methylamino tautomer with an "
    "N-oxide, PubChem CID 2712 the other tautomer; InChIKeys differ in the first "
    "block for the same substance, so the match is by name",
    "tranylcypromine": "ChEMBL CHEMBL3989843 is the trans racemate written as two "
    "enantiomer components, PubChem CID 5530 the stereo-unspecified single "
    "component; match is by name",
    "flupentixol": "ChEMBL CHEMBL6066864 (the ATC N05AF01 record) carries no "
    "structure; PubChem's 'flupentixol' CID 5281881 is the Z (cis) isomer, so this "
    "CAS identifies cis-flupentixol and NOT the E/Z mixture. Isomer identity is "
    "unresolved for this record",
    "cariprazine": "ChEMBL InChIKey carries the trans-cyclohexyl stereo, PubChem "
    "CID 11154555 is stereo-unspecified; same connectivity",
    "reboxetine": "ChEMBL CHEMBL14370 is stereo-unspecified, PubChem CID 127151 "
    "carries (R,R)/(S,S) stereo; same connectivity",
}

# Clinical failures: no marketing approval for the psychiatric indication
# tested, and a published placebo-controlled trial whose primary endpoint was
# not met. The PMID, title and quoted conclusion were read from Europe PMC on
# 2026-10-01. Candidates considered and rejected are listed in architecture.md.
FAILURES = [
    # (inn, chembl_parent_id, intended_atc_level3, indication, pmid, citation, evidence)
    ("bitopertin", "CHEMBL1171829", "N05A", "negative symptoms of schizophrenia (adjunct)",
     28117049, "Bugarski-Kirola D et al. 2017, Biol Psychiatry",
     "Phase III FlashLyte/DayLyte: no statistically significant separation from placebo on the primary end point"),
    ("pomaglumetad methionil", "CHEMBL2110730", "N05A", "schizophrenia",
     25539791, "Downing AM et al. 2014, BMC Psychiatry",
     "LY2140023: neither dose showed significant improvement vs placebo on PANSS total; risperidone separated"),
    ("sonepiprazole", "CHEMBL69759", "N05A", "schizophrenia",
     15023570, "Corrigan MH et al. 2004, Biol Psychiatry",
     "no significant difference vs placebo on primary or any secondary end point; olanzapine separated"),
    ("bradanicline", "CHEMBL1258006", "N05A", "negative and cognitive symptoms of schizophrenia",
     26071208, "Walling D et al. 2016, Schizophr Bull",
     "TC-5619: SANS showed no statistical benefit vs placebo at week 24"),
    ("iclepertin", "CHEMBL5314576", "N05A", "cognitive impairment associated with schizophrenia",
     41233083, "Keefe RSE et al. 2025, Lancet Psychiatry",
     "CONNEX phase 3: no significant difference in the primary endpoint (MCCB composite, p=0.63)"),
    ("roluperidone", "CHEMBL4297605", "N05A", "negative symptoms of schizophrenia",
     35211743, "Davidson M et al. 2022, Schizophr Bull",
     "phase 3 primary endpoint (NSFS) missed significance in the ITT analysis (P<=.064)"),
    ("balipodect", "CHEMBL3989972", "N05A", "acute exacerbation of schizophrenia",
     30190165, "Macek TA et al. 2019, Schizophr Res",
     "TAK-063: primary endpoint (PANSS total, week 6) not achieved (p=0.115)"),
    ("mardepodect", "CHEMBL562318", "N05A", "schizophrenia (adjunct)",
     31205187, "DeMartinis N et al. 2019, J Clin Psychopharmacol",
     "PF-02545920: no significant differences vs placebo on prespecified efficacy end points"),
    ("aprepitant", "CHEMBL1471", "N06A", "major depressive disorder",
     16248986, "Keller M et al. 2006, Biol Psychiatry",
     "no significant difference from placebo on HAM-D17 at week 8 in any trial; paroxetine separated"),
    ("basimglurant", "CHEMBL3301626", "N06A", "major depressive disorder (adjunct)",
     27304433, "Quiroz JA et al. 2016, JAMA Psychiatry",
     "no difference on the primary outcome (clinician-rated MADRS) vs placebo"),
    ("decoglurant", "CHEMBL3301593", "N06A", "major depressive disorder (adjunct)",
     32663909, "Umbricht D et al. 2020, J Clin Psychiatry",
     "no significant differences vs placebo on MADRS total at 6 weeks"),
    ("edivoxetine", "CHEMBL2103857", "N06A", "major depressive disorder (adjunct)",
     27035159, "Ball SG et al. 2016, J Clin Psychiatry",
     "each of 3 trials failed to meet the primary objective"),
    ("esmethadone", "CHEMBL350719", "N06A", "major depressive disorder (adjunct)",
     38917366, "Fava M et al. 2024, J Clin Psychiatry",
     "phase 3 (NCT04688164): the primary end point was not met (P=.154)"),
    ("navacaprant", "CHEMBL4592045", "N06A", "major depressive disorder",
     40199329, "Mathew SJ et al. 2025, J Clin Psychopharmacol",
     "phase 2: primary endpoint not met in the efficacy population"),
    ("lanicemine", "CHEMBL467084", "N06A", "major depressive disorder (adjunct)",
     27681442, "Sanacora G et al. 2017, Neuropsychopharmacology",
     "AZD6765: neither dose superior to placebo on the primary end point or any secondary measure"),
    ("pexacerfont", "CHEMBL482950", "N05B", "generalized anxiety disorder",
     20455246, "Coric V et al. 2010, Depress Anxiety",
     "did not separate from placebo on HAM-A (primary); escitalopram separated"),
    ("verucerfont", "CHEMBL1287935", "N05B", "posttraumatic stress disorder",
     28793974, "Dunlop BW et al. 2017, Biol Psychiatry",
     "GSK561679 failed to show superiority over placebo on CAPS total (primary)"),
]

# CAS source preference, most authoritative first. PubChem source names.
CAS_SOURCES = (
    "CAS Common Chemistry",
    "FDA Global Substance Registration System (GSRS)",
    "ChemIDplus",
    "EPA DSSTox",
)

SPLIT_SEED = 20261001  # the date the split was drawn
HELDOUT_FRACTION = 0.30


def _get(url: str):
    err = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # noqa: BLE001 - transient network errors
            err = e
        time.sleep(3)
    raise err


def _chembl_release() -> str:
    s = _get(f"{CHEMBL}/status.json")
    return f"{s['chembl_db_version']} ({s['chembl_release_date']})"


def _molecule(chembl_id: str) -> dict:
    return _get(f"{CHEMBL}/molecule/{chembl_id}.json")


def _cas_for_cid(cid: int) -> tuple[str | None, dict]:
    """CAS for one PubChem CID from the most-preferred source listing exactly one."""
    view = _get(f"{PUBCHEM}/pug_view/data/compound/{cid}/JSON?heading=CAS")
    time.sleep(0.3)
    if not view:
        return None, {}
    refs = {r["ReferenceNumber"]: r["SourceName"] for r in view["Record"]["Reference"]}
    by_source: dict[str, set[str]] = {}

    def walk(sec):
        for s in sec.get("Section", []):
            for inf in s.get("Information", []):
                src = refs.get(inf.get("ReferenceNumber"))
                for v in inf["Value"].get("StringWithMarkup", []):
                    by_source.setdefault(src, set()).add(v["String"])
            walk(s)

    walk(view["Record"])
    found = {k: sorted(v) for k, v in by_source.items() if k in CAS_SOURCES}
    for src in CAS_SOURCES:
        vals = by_source.get(src, set())
        if len(vals) == 1:
            cas = next(iter(vals))
            agree = sum(1 for v in by_source.values() if cas in v)
            return cas, {"pubchem_cid": cid, "source": src,
                         "n_pubchem_sources_listing_it": agree}
    return None, found


def _cas(inchikey: str | None, inn: str) -> tuple[str | None, dict]:
    """Return (cas, source record).

    1. ChEMBL standard InChIKey -> PubChem CIDs (first five), first CID with a
       preferred-source CAS; ``match`` = "inchikey".
    2. Otherwise PubChem name lookup on the INN, and the name hit's InChIKey is
       compared with ChEMBL's: "inchikey" (identical), "connectivity" (first
       block equal, stereo/protonation differs) or "name_only" (no structural
       agreement could be checked; both keys recorded).
    """
    tried = []
    if inchikey:
        cids = _get(f"{PUBCHEM}/pug/compound/inchikey/{inchikey}/cids/JSON")
        time.sleep(0.3)
        for cid in (cids or {}).get("IdentifierList", {}).get("CID", [])[:5]:
            tried.append(cid)
            cas, rec = _cas_for_cid(cid)
            if cas:
                return cas, {**rec, "match": "inchikey",
                             "lookup": f"PubChem InChIKey {inchikey}"}
    hit = _get(f"{PUBCHEM}/pug/compound/name/{urllib.parse.quote(inn)}/property/InChIKey/JSON")
    time.sleep(0.3)
    if hit:
        prop = hit["PropertyTable"]["Properties"][0]
        cas, rec = _cas_for_cid(prop["CID"])
        if cas:
            pk = prop["InChIKey"]
            if inchikey == pk:
                match = "inchikey"
            elif inchikey and inchikey.split("-")[0] == pk.split("-")[0]:
                match = "connectivity"
            else:
                match = "name_only"
            return cas, {**rec, "match": match,
                         "lookup": f"PubChem name '{inn}' -> CID {prop['CID']} "
                                   f"(InChIKey {pk}; ChEMBL InChIKey {inchikey})"}
    return None, {
        "searched": f"PubChem CIDs {tried} by InChIKey {inchikey} and name '{inn}', "
                    f"PUG-View heading CAS, sources {list(CAS_SOURCES)}",
        "result": "no preferred source gives exactly one CAS number",
    }


def build_drugs() -> dict:
    release = _chembl_release()
    chembl_src = f"ChEMBL REST API, {release}"
    records = []
    excluded = []
    seen: set[str] = set()
    for l3 in ATC_LEVEL3:
        atc = _get(f"{CHEMBL}/atc_class.json?level3={l3}&limit=1000")["atc"]
        for a in sorted(atc, key=lambda x: x["level5"]):
            code = a["level5"]
            if code in ATC_EXCLUDED:
                excluded.append({"atc_code": code, "who_name": a["who_name"],
                                 "reason": ATC_EXCLUDED[code]})
                continue
            if code in CHEMBL_OVERRIDE:
                parent_id, note = CHEMBL_OVERRIDE[code]
                id_src = f"{chembl_src}; {note}"
            else:
                mols = _get(f"{CHEMBL}/molecule.json?atc_classifications={code}&limit=20")["molecules"]
                parents = sorted({m["molecule_hierarchy"]["parent_chembl_id"] for m in mols})
                if len(parents) != 1:
                    raise RuntimeError(f"{code}: expected one parent, got {parents}")
                parent_id = parents[0]
                id_src = (f"{chembl_src}, molecule?atc_classifications={code}, "
                          f"parent_chembl_id of the mapped molecule")
            if parent_id in seen:
                raise RuntimeError(f"{code}: {parent_id} already in the list")
            seen.add(parent_id)
            records.append(_record(
                inn=a["who_name"], chembl_id=parent_id, id_source=id_src,
                stratum=l3, atc_code=code,
                atc_source=f"WHO ATC via {chembl_src}, atc_class level3={l3}",
                inn_source=f"WHO ATC/DDD name for {code} via {chembl_src}, atc_class",
                failure=None, chembl_src=chembl_src))
    for inn, cid, l3, indication, pmid, cite, evidence in FAILURES:
        if cid in seen:
            raise RuntimeError(f"failure {inn}: {cid} already in the list")
        seen.add(cid)
        records.append(_record(
            inn=inn, chembl_id=cid,
            id_source=f"{chembl_src}, molecule search; synonyms include the INN and the trial code name",
            stratum="FAIL", atc_code=None, atc_source=None,
            inn_source=f"{chembl_src}, molecule_synonyms of {cid}",
            failure={"intended_atc_level3": l3, "indication": indication,
                     "pmid": pmid, "citation": cite, "evidence": evidence,
                     "source": f"PMID {pmid}, abstract read via Europe PMC 2026-10-01"},
            chembl_src=chembl_src))
    records.sort(key=lambda r: r["inn"])
    return {
        "description": "P28.1 CNS benchmark drug list. One record per drug, keyed by name "
                       "(WHO ATC name, which is the INN where one exists). See "
                       "docs/architecture.md 'P28.1' for inclusion rules.",
        "built": "2026-10-01",
        "chembl_release": release,
        "strata": {"N05A": "antipsychotics (ATC N05A)", "N05B": "anxiolytics (ATC N05B)",
                   "N06A": "antidepressants (ATC N06A)",
                   "FAIL": "clinical failures: no approval for the psychiatric indication, "
                           "published trial missed its primary endpoint"},
        "excluded_atc_entries": excluded,
        "drugs": records,
    }


def _record(*, inn, chembl_id, id_source, stratum, atc_code, atc_source,
            inn_source, failure, chembl_src) -> dict:
    mol = _molecule(chembl_id)
    ik = (mol.get("molecule_structures") or {}).get("standard_inchi_key")
    all_atc = sorted(mol.get("atc_classifications") or [])
    cas, cas_src = _cas(ik, inn)
    if inn in CAS_NOTES:
        cas_src = {**cas_src, "note": CAS_NOTES[inn]}
    rec = {
        "inn": inn,
        "stratum": stratum,
        "chembl_id": chembl_id,
        "chembl_pref_name": mol.get("pref_name"),
        "standard_inchikey": ik,
        "cas": cas,
        "atc_code": atc_code,
        "atc_codes_all": all_atc,
        "max_phase": mol.get("max_phase"),
        "withdrawn_flag": mol.get("withdrawn_flag"),
        "sources": {
            "inn": inn_source,
            "stratum": atc_source or "FAILURES list in scripts/build_benchmark.py; see failure.source",
            "chembl_id": id_source,
            "chembl_pref_name": f"{chembl_src}, molecule/{chembl_id}",
            "standard_inchikey": f"{chembl_src}, molecule/{chembl_id} molecule_structures",
            "cas": cas_src,
            "atc_code": atc_source or "none: not in ATC N05A/N05B/N06A (failure stratum)",
            "atc_codes_all": f"{chembl_src}, molecule/{chembl_id} atc_classifications",
            "max_phase": f"{chembl_src}, molecule/{chembl_id}",
            "withdrawn_flag": f"{chembl_src}, molecule/{chembl_id}",
        },
    }
    if failure is not None:
        rec["failure"] = failure
    print(stratum, inn, chembl_id, cas, file=sys.stderr)
    return rec


def draw_split(drugs: dict) -> dict:
    """Stratified drug-level split; held-out count per stratum = round(0.3 n)."""
    rng = random.Random(SPLIT_SEED)
    strata: dict[str, list[str]] = {}
    for d in drugs["drugs"]:
        strata.setdefault(d["stratum"], []).append(d["inn"])
    train, heldout, counts = [], [], {}
    for s in sorted(strata):
        names = sorted(strata[s])
        rng.shuffle(names)
        k = round(HELDOUT_FRACTION * len(names))
        heldout += names[:k]
        train += names[k:]
        counts[s] = {"n": len(names), "heldout": k, "train": len(names) - k}
    return {
        "description": "P28.1 frozen drug-level train/held-out split. Never redraw; "
                       "the sha256 of this file is pinned in a test and in "
                       "docs/validation-ledger.md.",
        "drawn": "2026-10-01",
        "method": f"per stratum: sort INNs, random.Random({SPLIT_SEED}).shuffle (one "
                  f"generator across strata in sorted stratum order), first "
                  f"round({HELDOUT_FRACTION}*n) held out",
        "strata_counts": counts,
        "train": sorted(train),
        "heldout": sorted(heldout),
    }


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n")


def main(argv: list[str]) -> None:
    stage = argv[1] if len(argv) > 1 else ""
    if stage == "drugs":
        _dump(build_drugs(), OUT_DIR / "drugs.json")
    elif stage == "split":
        if (OUT_DIR / "split.json").exists():
            raise SystemExit("split.json exists and is frozen; refusing to redraw")
        _dump(draw_split(json.loads((OUT_DIR / "drugs.json").read_text())),
              OUT_DIR / "split.json")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
