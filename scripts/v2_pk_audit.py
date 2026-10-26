#!/usr/bin/env python3
"""Provenance audit of the frozen v2 human PK truth (2026-10-26).

``python scripts/v2_pk_audit.py [--cache DIR]``

truth/human_pk.json is frozen, so this script never edits it. It:

1. Re-derives, from the same ChEMBL cache scripts/v2_pk.py built from, every
   measurement behind every clearance_iv / vss value in human_pk.json, with the
   original (pre-audit) trust rule, and asserts the re-derivation reproduces
   the frozen file exactly (value and document count for every field).
2. Re-fetches live, for every assay and document those measurements cite, the
   ChEMBL assay record (bao_format, assay_organism, assay_tissue, cell type,
   subcellular fraction, description) and the PubMed esummary title, and
   stores them under <cache>/audit-2026-10-26/ as evidence.
3. Applies the exclusion rule in v2_pk._assay_exclusion (assay level) and, for
   documents outside the curated compilations, v2_pk._title_names_drug on the
   live PubMed title (source level).
4. Writes truth/human_pk_corrections.json: a field whose every measurement is
   excluded goes to "drop"; a field that keeps some valid measurements is
   recomputed with v2_pk._summarise (geomean within document, median across
   documents, censored rows ignored), the original aggregation, and goes to
   "replace" if the value or its source list changed.

The same rule is applied to every row. The script prints counts only, never a
value.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_pk as S  # noqa: E402

CORRECTIONS_FILE = S.V2_DIR / "truth" / "human_pk_corrections.json"
DATE = "2026-10-26"
FIELD_KEY = {"invivo_cl": "clearance_iv", "invivo_vss": "vss"}
UNIT = {"invivo_cl": "mL/min/kg", "invivo_vss": "L/kg"}

RULE = (
    "A clearance_iv or vss value is human in-vivo intravenous PK only if every "
    "ChEMBL activity behind it comes from an organism-based assay "
    "(bao_format BAO_0000218) in Homo sapiens with no cell type or subcellular "
    "fraction and no in-vitro wording (microsomes, hepatocytes, S9, intrinsic "
    "clearance, metabolic stability); for clearance, the assay must be total "
    "body clearance, not an organ-partial clearance (assay_tissue set, or "
    "hepatic/renal/biliary clearance); for vss, the assay must not be an unbound "
    "volume; and the assay must not state a non-intravenous route (po, oral, im, "
    "sc and similar) without also stating iv. A source document outside the "
    "curated human PK compilations listed in scripts/v2_pk.py is accepted only "
    "if its PubMed title names the drug as a whole word, carries a human or "
    "in-vivo PK term, and does not describe analogues, derivatives, a compound "
    "series, synthesis or lead optimisation, a drug-drug interaction, a "
    "non-human species or an in-vitro system. Activities failing any test are "
    "removed. A field with no valid activity left is dropped (set to null); a "
    "field with some valid activities left is recomputed from those alone with "
    "the original aggregation (geometric mean within a document, median across "
    "documents, censored values ignored). The rule concerns source type only "
    "and is applied identically to every drug."
)


def _legacy_trusted(field, doc_id, doc, inn):
    allow = {"invivo_cl": S.CL_COMPILATIONS, "invivo_vss": S.VSS_COMPILATIONS}[field]
    return doc_id in allow or S._title_mentions_drug(doc.get("title"), inn)


def legacy_measurements(cache):
    drugs = S._drugs()
    acts = {d["inn"]: S._chembl_all(
        cache, "activity",
        {"parent_molecule_chembl_id": d["chembl_id"], "assay_type": "A",
         "standard_type__in": ",".join(S.STANDARD_TYPES)}, "activities") for d in drugs}
    flat = [a for v in acts.values() for a in v]
    assays = S._chembl_by_ids(cache, "assay", "assay_chembl_id",
                              [a["assay_chembl_id"] for a in flat], "assays")
    docs = S._chembl_by_ids(cache, "document", "document_chembl_id",
                            [a["document_chembl_id"] for a in flat], "documents")
    verified = S._verified_docs(cache, docs)
    out = {}
    for d in drugs:
        inn = d["inn"]
        fields = {"invivo_cl": [], "invivo_vss": []}
        for a in acts[inn]:
            doc_id = a.get("document_chembl_id")
            if doc_id not in verified or a.get("standard_value") is None:
                continue
            value = float(a["standard_value"])
            if value <= 0:
                continue
            assay = assays.get(a["assay_chembl_id"], {})
            descr = a.get("assay_description") or assay.get("description") or ""
            cls = S._classify(a["standard_type"], a.get("standard_units"), descr)
            if cls is None or cls[0] == "plasma_fu":
                continue
            field = cls[0]
            if not _legacy_trusted(field, doc_id, docs[doc_id], inn):
                continue
            if S._species(assay.get("assay_organism"), descr) != "human":
                continue
            rel = a.get("standard_relation") or "="
            fields[field].append({
                "value": value, "relation": rel if rel in ("=", "<", ">") else "=",
                "ref": doc_id, "assay": a["assay_chembl_id"], "descr": descr,
                "activity_id": a.get("activity_id"),
            })
        out[inn] = fields
    return out, docs


def live_metadata(cache, assay_ids, docs, doc_ids):
    """Live ChEMBL assay records and PubMed esummaries, stored as evidence."""
    store = cache / f"audit-{DATE}"
    store.mkdir(parents=True, exist_ok=True)
    assays = {}
    ids = sorted(assay_ids)
    for i in range(0, len(ids), 50):
        url = (f"{S.CHEMBL}assay.json?assay_chembl_id__in={','.join(ids[i:i + 50])}"
               f"&limit=1000")
        for rec in json.loads(S._download(url))["assays"]:
            assays[rec["assay_chembl_id"]] = rec
    pmids = sorted({str(docs[d]["pubmed_id"]) for d in doc_ids if docs[d].get("pubmed_id")})
    url = f"{S.EUTILS}esummary.fcgi?db=pubmed&id={','.join(pmids)}&retmode=json"
    summ = json.loads(S._download(url))["result"]
    titles = {p: summ[p]["title"] for p in pmids if p in summ and "title" in summ[p]}
    (store / "assays.json").write_text(json.dumps(assays, indent=1, sort_keys=True))
    (store / "pubmed_esummary.json").write_text(json.dumps(summ, indent=1, sort_keys=True))
    missing = sorted(set(assay_ids) - set(assays))
    if missing:
        raise SystemExit(f"live ChEMBL did not return assays {missing}")
    return assays, titles


def main(argv):
    cache = S.DEFAULT_CACHE
    if "--cache" in argv:
        cache = Path(argv[argv.index("--cache") + 1])
    meas, docs = legacy_measurements(cache)

    truth = {r["inn"]: r for r in json.loads(S.TRUTH_FILE.read_text())["drugs"]}
    for inn, f in meas.items():
        for field, key in FIELD_KEY.items():
            s, t = S._summarise(f[field]), truth[inn][key]
            same = (s is None and t is None) or (
                s is not None and t is not None
                and (s["value"], s["n_documents"], s["n_measurements"])
                == (t["value"], t["n_documents"], t["n_measurements"]))
            if not same:
                raise SystemExit(f"re-derivation does not reproduce {inn}.{key}")
    print(f"reproduced {len(meas)} rows of human_pk.json exactly", file=sys.stderr)

    used = [m for f in meas.values() for lst in f.values() for m in lst]
    assays, titles = live_metadata(cache, {m["assay"] for m in used}, docs,
                                   {m["ref"] for m in used})
    print(f"live metadata: {len(assays)} assays, {len(titles)} PubMed titles",
          file=sys.stderr)

    def reason(field, m, inn):
        r = S._assay_exclusion(field, assays[m["assay"]], m["descr"])
        if r:
            return r
        allow = S.CL_COMPILATIONS if field == "invivo_cl" else S.VSS_COMPILATIONS
        if m["ref"] not in allow:
            pmid = str(docs[m["ref"]].get("pubmed_id") or "")
            if not S._title_names_drug(titles.get(pmid), inn):
                return "wrong_entity_source" if pmid in titles else "unverifiable_source"
        return None

    drop, replace = {}, {}
    counts = Counter()
    for inn in sorted(meas):
        for field, key in FIELD_KEY.items():
            ms = meas[inn][field]
            if S._summarise(ms) is None:
                continue
            counts["fields_audited"] += 1
            bad = [(m, reason(field, m, inn)) for m in ms]
            excluded = [(m, r) for m, r in bad if r]
            if not excluded:
                continue
            evidence = []
            for m, r in excluded:
                pmid = docs[m["ref"]].get("pubmed_id")
                evidence.append({"assay": m["assay"], "document": m["ref"],
                                 "pmid": str(pmid) if pmid else None, "reason": r})
            keep = [m for m, r in bad if not r]
            new = S._summarise(keep)
            reasons = sorted({r for _, r in excluded})
            if new is None:
                entry = drop.setdefault(inn, {"fields": [], "reason": "", "evidence": [],
                                              "detail": {}})
                entry["fields"].append(key)
                entry["detail"][key] = {"reasons": reasons, "excluded": evidence}
                for e in evidence:
                    for ident in (e["assay"], e["pmid"] and f"PMID:{e['pmid']}"):
                        if ident and ident not in entry["evidence"]:
                            entry["evidence"].append(ident)
                for r in reasons:
                    counts[f"drop:{r}"] += 1
                counts["drop_fields"] += 1
                continue
            refs = sorted({m["ref"] for m in keep if m["relation"] == "="})
            old = truth[inn][key]
            old_refs = sorted(s["chembl_document_id"] for s in old["sources"])
            if (new["value"], new["n_measurements"], refs) == (
                    old["value"], old["n_measurements"], old_refs):
                counts["excluded_censored_only"] += 1
                continue
            new.update({"sources": [S._doc_citation(docs[r]) for r in refs],
                        "matched_on": "chembl_id", "unit": UNIT[field],
                        "excluded": evidence})
            replace.setdefault(inn, {})[key] = new
            for r in reasons:
                counts[f"replace:{r}"] += 1
            counts["replace_fields"] += 1
    for inn, entry in drop.items():
        entry["reason"] = "; ".join(
            f"{k}: {', '.join(v['reasons'])}" for k, v in entry["detail"].items())

    CORRECTIONS_FILE.write_text(json.dumps({
        "version": 1, "date": DATE, "rule": RULE,
        "applies_to": "truth/human_pk.json",
        "drop": drop, "replace": replace,
    }, indent=1, ensure_ascii=False) + "\n")

    from neurocascade.benchmark.v2 import v2_heldout  # noqa: E402
    split = Counter()
    for inn, entry in drop.items():
        split["drop_rows_" + ("h" if v2_heldout(inn) else "t")] += 1
    for inn in replace:
        split["replace_rows_" + ("h" if v2_heldout(inn) else "t")] += 1
    print(json.dumps({"counts": dict(sorted(counts.items())),
                      "rows": dict(sorted(split.items()))}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
