"""P28.4: per-drug human clinical truth (PK, dose, PET occupancy, clinical-effect
labels) for the benchmark, and the effect-label matrix built from it."""

import importlib.util
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

from neurocascade.benchmark import (
    TRUTH_DIR,
    load_drugs,
    load_effect_labels,
    load_split,
    load_truth_coverage,
    load_truth_record,
)

ROOT = Path(__file__).resolve().parents[1]
EFFECTS = ("eps", "sedation", "weight_gain", "anticholinergic", "orthostasis",
           "sexual_dysfunction", "seizure")


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_benchmark_truth", ROOT / "scripts" / "build_benchmark_truth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _script()


@pytest.fixture(scope="module")
def records():
    return {d["inn"]: load_truth_record(d["inn"]) for d in load_drugs()}


def _numbers(text):
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


# ----------------------------------------------------------------- records


def test_every_benchmark_drug_has_exactly_one_record_on_the_frozen_split(records):
    files = {p.name for p in TRUTH_DIR.glob("*.json")} - {"coverage.json",
                                                           "effect_labels.json"}
    assert len(files) == 185 == len(records)
    split = load_split()
    side = {i: "train" for i in split["train"]} | {i: "heldout" for i in split["heldout"]}
    for inn, r in records.items():
        assert r["inn"] == inn
        assert r["split"] == side[inn]
        assert set(r) >= {"pk", "dose", "pet", "effects", "label", "searched", "references"}
    with pytest.raises(KeyError):
        load_truth_record("aspirin")


def test_iv_pk_is_human_in_vivo_total_and_cited(records, script):
    """Only total systemic CL, Vss and terminal t1/2 measured in people; every
    row resolves to a ChEMBL document, and no in-vitro or partial clearance."""
    units = {"cl_mL_min_kg": "mL.min-1.kg-1", "vss_L_kg": "L.kg-1", "t_half_h": "hr"}
    for inn, r in records.items():
        for field, blob in r["pk"]["iv_compilations"].items():
            assert field in units
            for m in blob["measurements"]:
                assert m["species"] == "human"
                assert m["units"] == units[field]
                assert m["value"] > 0
                assert m["ref"] in r["references"]
                d = m["description"].lower()
                assert not re.search(r"microsom|hepatocyte|intrinsic|hepatic clearance|"
                                     r"renal clearance|unbound", d), (inn, d)
                assert script.classify_human_pk(m["standard_type"], m["units"],
                                                m["description"]) == (field, m["route"])
                if field == "cl_mL_min_kg":
                    assert m["route"] != "oral"


def test_iv_pk_summary_is_recomputable_per_route(records):
    """Geometric mean within a document, median across documents, exact rows
    only, never pooling an explicit-IV value with an unspecified-route one."""
    for inn, r in records.items():
        for field, blob in r["pk"]["iv_compilations"].items():
            groups = defaultdict(list)
            for m in blob["measurements"]:
                groups[m["route"]].append(m)
            assert {s["route"] for s in blob["summary"]} == set(groups)
            for s in blob["summary"]:
                exact = [m for m in groups[s["route"]] if m["relation"] == "="]
                by_doc = defaultdict(list)
                for m in exact:
                    by_doc[m["ref"]].append(math.log(m["value"]))
                want = statistics.median(math.exp(statistics.fmean(v))
                                         for v in by_doc.values())
                assert s["value"] == pytest.approx(want, rel=1e-5), (inn, field)


def test_known_human_pk_spot_values(records):
    """Spot checks against the compilations as printed (Obach 2008 IV set via
    ChEMBL): haloperidol Vss 17.5 L/kg, diazepam CL 0.38 mL/min/kg,
    risperidone t1/2 3.2 h."""
    def iv(inn, field):
        return {s["route"]: s["value"]
                for s in records[inn]["pk"]["iv_compilations"][field]["summary"]}["iv"]
    assert iv("haloperidol", "vss_L_kg") == pytest.approx(17.5)
    assert iv("diazepam", "cl_mL_min_kg") == pytest.approx(0.38)
    assert iv("risperidone", "t_half_h") == pytest.approx(3.2)


def test_label_transcriptions_report_only_printed_numbers(records):
    """Every label PK/dose value is printed in its own quotes, or says how it
    was derived; and it comes from the record's own selected label."""
    n = 0
    for inn, r in records.items():
        items = r["pk"]["label"] + r["dose"]["label"]
        if items:
            assert r["label"] is not None
            assert r["label_transcription"]["set_id"] == r["label"]["set_id"]
        for it in items:
            n += 1
            printed = set().union(*(_numbers(q) for q in it["quotes"]))
            for k in ("value", "low", "high"):
                v = it.get(k)
                if v is not None and not it.get("derivation"):
                    assert float(v) in printed, (inn, it["quantity"], k, v)
            if it["kind"] == "dose":
                assert it["units"] == "mg/day"
                assert it["high"] is None or it["high"] >= it["low"]
            else:
                assert it["quantity"] in ("t_half_h", "cl", "cl_over_f", "vss", "vd",
                                          "v_over_f")
    assert n == 196


def test_innovator_label_dose_spot_values(records):
    """The selected label is the innovator: Zoloft 50-200 mg/day for MDD, not a
    150/200 mg capsule reformulation; Cymbalta 40-120 mg/day."""
    def dose(inn, indication):
        return [(i["low"], i["high"]) for i in records[inn]["dose"]["label"]
                if i["indication"] == indication]
    assert records["sertraline"]["label"]["brand_name"] == ["Zoloft"]
    assert dose("sertraline", "MDD") == [(50, 200)]
    assert dose("duloxetine", "MDD") == [(40, 120)]
    assert dose("aripiprazole", "schizophrenia") == [(10, 30)]


def test_ddd_rows_are_as_printed_for_every_atc_code(records):
    drugs = {d["inn"]: d for d in load_drugs()}
    for inn, r in records.items():
        assert [x["atc_code"] for x in r["dose"]["ddd"]] == drugs[inn]["atc_codes_all"]
        for x in r["dose"]["ddd"]:
            for row in x["ddd"]:
                assert row["ddd"] > 0
                assert row["unit"] in ("mg", "g", "mcg")
                assert row["route"]  # WHO's code as printed: O, P, R, N, SL, Inhal.powder
    hal = {(y["ddd"], y["route"], y["note"]) for y in records["haloperidol"]["dose"]["ddd"][0]["ddd"]}
    assert hal == {(8.0, "O", None), (8.0, "P", None), (3.3, "P", "depot")}
    assert records["sertraline"]["dose"]["ddd"][0]["ddd"] == [
        {"ddd": 50.0, "unit": "mg", "route": "O", "note": None}]


def test_parse_ddd_keeps_a_code_without_ddd_empty(script):
    page = ("<table><tr><td>ATC code</td><td>Name</td><td>DDD</td><td>U</td>"
            "<td>Adm.R</td><td>Note</td></tr><tr><td>N05AX99&nbsp;</td>"
            "<td><a>examplazine</a>&nbsp;</td><td></td><td></td><td></td><td></td></tr>"
            "</table><p>Last updated: 2026-01-20</p>")
    out = script.parse_ddd(page, "N05AX99")
    assert out == {"atc_code": "N05AX99", "who_name": "examplazine", "ddd": [],
                   "index_last_updated": "2026-01-20"}


# ----------------------------------------------------------------- PET


def test_pet_observations_are_printed_in_their_abstract_quotes(records):
    n, drugs = 0, set()
    for inn, r in records.items():
        for o in r["pet"]:
            n += 1
            drugs.add(inn)
            assert o["drug"] == inn
            assert o["target"] in ("D2", "5-HT2A", "SERT")
            assert o["pmid"].isdigit()
            printed = _numbers(o["quote"])
            occ = o["occupancy_percent"]
            for v in occ if isinstance(occ, list) else [occ]:
                assert float(v) in printed, (inn, o["pmid"], v)
            dose = o["dose_mg_per_day"]
            for v in dose if isinstance(dose, list) else [dose]:
                assert float(v) in printed or o.get("dose_derivation"), (inn, o["pmid"])
    assert n == 82
    assert len(drugs) == 27


def test_pet_bounds_are_not_stored_as_means(records):
    """'<5%', '>80%' and 'up to 39%' are bounds or peaks, not group means."""
    stats = {(o["pmid"], o["target"], str(o["dose_mg_per_day"])): o["statistic"]
             for r in records.values() for o in r["pet"]}
    assert stats[("31847007", "SERT", "4")] == "upper_bound"
    assert stats[("25843749", "5-HT2A", "10")] == "lower_bound"
    assert stats[("25843749", "D2", "40")] == "peak"


# ----------------------------------------------------------------- effects


def test_effect_label_follows_the_stated_rule(records):
    """Recompute every label from its stored evidence under EFFECT_RULE."""
    for inn, r in records.items():
        ef = r["effects"]
        assert set(ef["label"]) == set(EFFECTS)
        for e in EFFECTS:
            pcts = [ev["frequency_lower"] * 100 if "frequency_lower" in ev
                    else ev["drug_arm_percent"] for ev in ef["evidence"][e]]
            mx = max(pcts, default=None)
            assert ef["max_percent"][e] == (None if mx is None else pytest.approx(mx, abs=1e-4))
            sider_freq = ef["sources"].get("sider", {}).get("n_frequency_rows_drug", 0) > 0
            if mx is not None and mx >= 5.0:
                want = 1
            elif ef["has_label_table"]:
                want = 0
            elif sider_freq and not ef["named_without_frequency"][e]:
                want = 0
            else:
                want = None
            assert ef["label"][e] == want, (inn, e)
            for ev in ef["evidence"][e]:
                if ev["source"] == "SIDER 4.1":
                    assert ev["description"] != "postmarketing"


def test_a_named_effect_without_a_frequency_is_unknown_not_negative(records):
    """Haloperidol's old label names EPS with no incidence: unknown, never 0."""
    hal = records["haloperidol"]["effects"]
    assert hal["label"]["eps"] is None
    assert hal["listed"]["eps"] == 1
    assert records["olanzapine"]["effects"]["label"]["weight_gain"] == 1
    assert records["clozapine"]["effects"]["label"]["sedation"] == 1
    assert records["sertraline"]["effects"]["label"]["sexual_dysfunction"] == 1
    assert records["sertraline"]["effects"]["label"]["eps"] == 0


def test_sider_never_crosses_entities(records):
    """SIDER's flat id merges stereoisomers: citalopram and escitalopram rows are
    split by stereo id, and the ATC-only links (clopenthixol -> zuclopenthixol,
    esketamine -> racemic ketamine) are refused."""
    cit = records["citalopram"]["effects"]["sources"]["sider"]
    esc = records["escitalopram"]["effects"]["sources"]["sider"]
    assert cit["flat_ids"] == esc["flat_ids"] == ["CID100002771"]
    assert set(cit["stereo_ids"]).isdisjoint(esc["stereo_ids"])
    for inn in ("clopenthixol", "esketamine"):
        assert "sider" not in records[inn]["effects"]["sources"]
    assert records["fluphenazine"]["effects"]["sources"]["sider"]["flat_ids"] == ["CID100003372"]


def test_effect_matrix_matches_records_and_prevalence_is_recomputable(records):
    m = load_effect_labels()
    assert tuple(m["effects"]) == EFFECTS
    assert m["rule"]["threshold_percent"] == 5.0
    assert set(m["drugs"]) == set(records)
    for inn, v in m["drugs"].items():
        assert v["label"] == records[inn]["effects"]["label"]
        assert v["listed"] == records[inn]["effects"]["listed"]
    for scope, sel in (("all", lambda v: True), ("train", lambda v: v["split"] == "train"),
                       ("heldout", lambda v: v["split"] == "heldout")):
        for e in EFFECTS:
            known = [v["label"][e] for v in m["drugs"].values()
                     if sel(v) and v["label"][e] is not None]
            p = m["prevalence"][scope]["label"][e]
            assert p["n_known"] == len(known)
            assert p["n_positive"] == sum(known)
    allp = m["prevalence"]["all"]["label"]
    assert {e: (allp[e]["n_positive"], allp[e]["n_known"]) for e in EFFECTS} == {
        "eps": (16, 45), "sedation": (38, 49), "weight_gain": (16, 44),
        "anticholinergic": (42, 46), "orthostasis": (6, 48),
        "sexual_dysfunction": (14, 42), "seizure": (0, 46)}


def test_term_matching_and_table_parsing(script):
    assert script.effect_term("Weight gain") == ("weight_gain", "weight increased")
    assert script.effect_term("Somnolence (including sedation)") == ("sedation", "somnolence")
    assert script.effect_term("Eye Disorders Vision blurred") == ("anticholinergic",
                                                                   "vision blurred")
    assert script.effect_term("Tremor") is None
    # rowspan header (ABILIFY) and a Body System column (nefazodone)
    table = ('<table><caption>Adverse Reactions in Adults</caption>'
             '<thead><tr><th rowspan="2">Body System</th><th rowspan="2">Preferred Term</th>'
             '<th colspan="2">Percent</th></tr><tr><th>Drugzine (n=10)</th>'
             '<th>Placebo (n=10)</th></tr></thead><tbody>'
             '<tr><td>Nervous</td><td>Somnolence<sup>a</sup></td><td>25</td><td>9</td></tr>'
             '<tr><td></td><td>Akathisia</td><td>&lt;1</td><td>0</td></tr>'
             '</tbody></table>')
    out = script.parse_ar_tables({"adverse_reactions_table": [table]}, ["drugzine"])
    assert out["n_tables_used"] == 1
    assert out["rows"] == [{"term": "Somnolence", "max_drug_arm_percent": 25.0,
                            "table": 0, "caption": "Adverse Reactions in Adults"}]


def test_comparator_and_pediatric_columns_are_not_the_drug(script):
    table = ('<table><caption>{}</caption><tr><th>Term</th><th>Drugzine</th>'
             '<th>Haloperidol</th><th>Placebo</th></tr>'
             '<tr><td>Akathisia</td><td>3</td><td>20</td><td>2</td></tr></table>')
    adult = script.parse_ar_tables(
        {"adverse_reactions_table": [table.format("Adult trials")]}, ["drugzine"])
    assert adult["rows"][0]["max_drug_arm_percent"] == 3.0
    ped = script.parse_ar_tables(
        {"adverse_reactions_table": [table.format("Pediatric Patients (13-17)")]}, ["drugzine"])
    assert ped["rows"] == []


def test_label_selection_prefers_the_oral_psychiatric_innovator(script):
    def lab(app, route, ind, eff, sid):
        return {"openfda": {"generic_name": ["DRUGZINE HYDROCHLORIDE"],
                            "application_number": [app], "route": [route]},
                "indications_and_usage": [ind], "effective_time": eff, "set_id": sid}
    d = {"inn": "drugzine"}
    labels = [
        lab("NDA019000", "ORAL", "treatment of major depressive disorder", "2020", "innov"),
        lab("NDA215000", "ORAL", "treatment of major depressive disorder", "2026", "reform"),
        lab("NDA204000", "ORAL", "vasomotor symptoms. Not indicated for the treatment of "
            "any psychiatric condition", "2026", "vms"),
        lab("NDA209000", "INTRAVENOUS", "schizophrenia", "2026", "iv"),
        lab("ANDA070000", "ORAL", "schizophrenia", "2026", "generic"),
        {**lab("NDA018000", "ORAL", "depression", "2026", "combo"),
         "openfda": {"generic_name": ["DRUGZINE AND OTHERAZINE"],
                     "application_number": ["NDA018000"], "route": ["ORAL"]}},
    ]
    assert script.select_label(d, labels)["set_id"] == "innov"
    assert script.select_label(d, labels[2:4]) is labels[2]  # non-psychiatric oral only


def test_coverage_counts_match_records(records):
    cov = load_truth_coverage()
    rows = {r["inn"]: r for r in cov["drugs"]}
    assert set(rows) == set(records)
    s = cov["summary"]
    assert s["n_drugs"] == 185
    for f, n in s["per_field"].items():
        assert n == sum(1 for r in rows.values() if r[f]), f
    for inn, r in records.items():
        c = rows[inn]
        assert c["openfda_label"] == (r["label"] is not None)
        assert c["pet_d2"] == any(o["target"] == "D2" for o in r["pet"])
        assert c["iv_cl"] == ("cl_mL_min_kg" in r["pk"]["iv_compilations"])
        assert c["effect_vector_known"] == r["effects"]["has_frequency_source"]
    assert s["per_field"]["openfda_label"] == 67
    assert s["per_field"]["iv_cl"] == 46
    assert s["per_field"]["ddd_oral"] == 128
