#!/usr/bin/env python3
"""v2 placebo-subtracted CNS liability labels from FDA adverse-reaction tables (P33.2).

What it does
------------
For each drug in ``benchmark/v2/drugs.json`` it picks one FDA label by the
pre-specified rule in ``benchmark/v2/liability_rules.json``, parses that
label's ADVERSE REACTIONS tables (the structured ``adverse_reactions_table``
HTML of the openFDA SPL export) with lxml, finds the placebo-controlled ones,
and reads the drug-arm and placebo-arm percentages for five receptor-anchored
liabilities: EPS (D2), sedation (H1), weight gain (H1/5-HT2C), anticholinergic
burden (M1-M5) and orthostatic hypotension (alpha1). A binary label follows
from the pre-specified margin rule.

What it does not do
-------------------
It never reads a number out of prose (the project rule stated in
``scripts/fetch_fda_labels.py``). Numbers come only from table cells and arm
sizes only from column-header cells. The inclusion-threshold sentence is copied
verbatim, not parsed. No LLM is involved; everything is deterministic.

It stores no held-out membership: consumers apply ``guards.v2_heldout(inn)``.

Inputs
------
``$NC_V2_CACHE/liab/labels.jsonl.gz``: the labels named in drugs.json, cut from
the full bulk index (``scripts/v2_label_index.py``) in one pass by
``--build-cache``.

    scripts/v2_liabilities.py --build-cache     # one pass over iter_labels (~4 min)
    scripts/v2_liabilities.py                   # write truth/liabilities.json
    scripts/v2_liabilities.py --show olanzapine # print one drug's record
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import lxml.html

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "src/neurocascade/benchmark/v2"
RULES = V2 / "liability_rules.json"
DRUGS = V2 / "drugs.json"
OUT = V2 / "truth/liabilities.json"
CACHE = Path(os.environ.get("NC_V2_CACHE", "/root/nc-v2-cache")) / "liab/labels.jsonl.gz"

LIABS = ("eps", "sedation", "weight_gain", "anticholinergic", "orthostatic_hypotension")
LOCAL_ROUTES = {"OPHTHALMIC", "TOPICAL", "OTIC", "CUTANEOUS", "VAGINAL",
                "DENTAL", "INTRAVITREAL", "AURICULAR (OTIC)", "TRANSMUCOSAL",
                # amendment A7: local anaesthesia and skin-test routes
                "INFILTRATION", "EPIDURAL", "PERINEURAL", "SUBMUCOSAL", "PERCUTANEOUS",
                "INTRADERMAL", "INTRACAVERNOUS", "INTRA-ARTICULAR"}
# amendment A7: nasal, inhaled and rectal labels count as local unless the
# drug is a nervous-system drug (ATC N) given that way for systemic effect
# (diazepam nasal spray and rectal gel, inhaled levodopa, volatile
# anaesthetics), so inhaled bronchodilators, antihistamine nasal sprays and
# haemorrhoid products are excluded.
AIRWAY_ROUTES = {"NASAL", "RESPIRATORY (INHALATION)", "INHALATION", "RECTAL"}
FOOTNOTE_CHARS = "*†‡§¶#‖"
POOLED_WORDS = ("all doses", "total", "pooled", "combined", "all ")
PEDIATRIC = re.compile(r"pediatric|paediatric|children|adolescent|\bchild\b", re.I)
DISCONTINUATION = re.compile(r"discontinu", re.I)
ADJUNCT = re.compile(r"adjunct|add-on|combination", re.I)
N_RE = re.compile(r"\b[nN]\s*=\s*(\d[\d,]*)")
NUM = r"(\d+(?:\.\d+)?)"


# ------------------------------------------------------------------ cache

def build_cache() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from v2_label_index import iter_labels
    want = {lab["set_id"] for r in json.loads(DRUGS.read_text())["drugs"] for lab in r["labels"]}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    n = 0
    with gzip.open(tmp, "wt") as f:
        for lab in iter_labels():
            if lab.get("set_id") in want:
                f.write(json.dumps({k: lab.get(k) for k in (
                    "set_id", "id", "version", "effective_time", "openfda",
                    "adverse_reactions", "adverse_reactions_table")}) + "\n")
                n += 1
    os.replace(tmp, CACHE)
    print(f"cached {n} of {len(want)} labels -> {CACHE}")


def load_cache() -> dict[str, dict]:
    out = {}
    with gzip.open(CACHE, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["set_id"]] = r
    return out


# ------------------------------------------------------------------ text

def clean(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def cell_text(el) -> tuple[str, str]:
    """(text without footnotes/superscripts, footnote text)."""
    notes = []
    el = lxml.html.fromstring(lxml.html.tostring(el))  # private copy
    for tag in ("footnote", "sup"):
        for fn in el.iter(tag):
            notes.append(fn.text_content())
    for tag in ("footnote", "sup", "footnoteref"):
        for fn in list(el.iter(tag)):
            tail = fn.tail
            parent = fn.getparent()
            if parent is None:
                continue
            prev = fn.getprevious()
            if tail:
                if prev is not None:
                    prev.tail = (prev.tail or "") + tail
                else:
                    parent.text = (parent.text or "") + tail
            parent.remove(fn)
    # paragraphs inside one cell are separate lines
    parts = [clean(p.text_content()) for p in el.iter("paragraph")] or [clean(el.text_content())]
    txt = clean(" ".join(p for p in parts if p))
    return txt, clean(" ".join(notes))


def strip_marks(s: str) -> str:
    return clean(s.strip().strip(FOOTNOTE_CHARS).rstrip(":").strip(FOOTNOTE_CHARS))


# ------------------------------------------------------------------ table grid

def parse_table(html: str) -> dict | None:
    try:
        tab = lxml.html.fragment_fromstring(html, create_parent=False)
    except Exception:
        try:
            tab = lxml.html.fromstring(html)
        except Exception:
            return None
    if tab.tag != "table":
        found = tab.find(".//table")
        if found is None:
            return None
        tab = found
    cap = tab.find(".//caption")
    caption = clean(cap.text_content()) if cap is not None else ""
    rows = []  # (tr element, in_thead)
    for tr in tab.iter("tr"):
        anc = [a.tag for a in tr.iterancestors()]
        if "tfoot" in anc:
            continue
        rows.append((tr, "thead" in anc))
    grid: list[list[dict | None]] = []
    uid = 0
    pending: dict[int, tuple[dict, int]] = {}  # col -> (cell, rows remaining)
    for ri, (tr, in_head) in enumerate(rows):
        line: list[dict | None] = []
        col = 0

        def fill_pending():
            nonlocal col
            while col in pending:
                cell, left = pending[col]
                line.append({**cell, "copy": True})
                if left <= 1:
                    del pending[col]
                else:
                    pending[col] = (cell, left - 1)
                col += 1
        fill_pending()
        for td in tr:
            if td.tag not in ("td", "th"):
                continue
            fill_pending()
            txt, note = cell_text(td)
            cs = int(td.get("colspan", "1") or 1) if str(td.get("colspan", "1")).isdigit() else 1
            rs = int(td.get("rowspan", "1") or 1) if str(td.get("rowspan", "1")).isdigit() else 1
            uid += 1
            cell = {"text": txt, "note": note, "head": in_head, "copy": False, "uid": uid}
            for k in range(max(cs, 1)):
                line.append(cell if k == 0 else {**cell, "copy": True, "colcopy": True})
                if rs > 1:
                    pending[col] = (cell, rs - 1)
                col += 1
        fill_pending()
        grid.append(line)
    if not grid:
        return None
    width = max(len(r) for r in grid)
    for r in grid:
        r.extend([{"text": "", "note": "", "head": False, "copy": False, "uid": -1}] * (width - len(r)))
    # Title rows: one cell spanning the whole width (the caption written as a
    # table row). They are moved into the caption so that a title such as
    # "Table 12: ... Placebo-Controlled Trials" is not read as a column header.
    titles = []
    while grid and len({c["uid"] for c in grid[0] if c["text"]}) == 1 and width > 1 and \
            all(c["uid"] == grid[0][0]["uid"] or not c["text"] for c in grid[0]) and \
            not any(is_number_cell(c["text"]) for c in grid[0]):
        titles.append(grid[0][0]["text"] or next(c["text"] for c in grid[0] if c["text"]))
        grid = grid[1:]
    if titles:
        caption = clean(" ".join([caption] + titles))
    return {"caption": caption, "grid": grid, "width": width}


def parse_value(raw: str, npct: bool, n: int | None = None) -> dict | None:
    """One table cell -> {'pct', 'censored', 'flag', 'raw'} or None if not a value.

    npct: the table or column header declares 'n (%)' / '(%)'. n: the arm size
    from the column header, used only to confirm that 'a (b)' is count and
    percent (100*a/n within 1 point of b) when nothing declares it."""
    s = strip_marks(raw)
    s = s.replace("≤", "<=").replace("≥", ">=").replace(" %", "%")
    out = {"raw": raw, "censored": False, "flag": None}
    if s in ("", "-", "–", "—", "--", "−"):
        return {**out, "pct": 0.0, "flag": "dash_as_zero"}
    m = re.fullmatch(NUM + r"%?", s)
    if m:
        return {**out, "pct": float(m.group(1))}
    m = re.fullmatch(r"<\s*" + NUM + r"%?", s)
    if m:
        return {**out, "pct": float(m.group(1)) / 2, "censored": True}
    m = re.fullmatch(r"\(\s*(<\s*)?" + NUM + r"\s*%?\s*\)", s)  # '(3%)' in a % sub-column
    if m:
        pct = float(m.group(2))
        return {**out, "pct": pct / 2 if m.group(1) else pct, "censored": bool(m.group(1))}
    m = re.fullmatch(r"(\d+)\s*\(\s*(<\s*)?" + NUM + r"\s*(%?)\s*\)", s)
    if m:
        cnt, lt, pct = int(m.group(1)), bool(m.group(2)), float(m.group(3))
        if m.group(4) or npct or (n and not lt and abs(100.0 * cnt / n - pct) <= 1.0):
            return {**out, "pct": pct / 2 if lt else pct, "censored": lt}
        return {**out, "pct": None, "flag": "unparsed"}
    m = re.fullmatch(NUM + r"%\s*\(\s*\d+\s*\)", s)
    if m:
        return {**out, "pct": float(m.group(1))}
    return None


def is_number_cell(raw: str) -> bool:
    v = parse_value(raw, True)
    return v is not None and v["flag"] is None


def structure(tab: dict, names: list[str]) -> list[dict] | None:
    """Header rows, column roles and data rows of one parsed table.

    Returns one entry per placebo-controlled column group: a table that puts
    several trials side by side (MDD | OCD | bulimia, adults | children) under
    a spanning header row is split so each drug arm is compared with the
    placebo arm of its own group, never a placebo pooled across populations."""
    grid, width = tab["grid"], tab["width"]
    head_n = 0
    for r in grid:
        first = r[0]
        nums = sum(is_number_cell(c["text"]) for c in r[1:] if not c.get("colcopy"))
        is_data = (nums >= 1 and first["text"] and not first["copy"] and not first["head"]
                   and not is_number_cell(first["text"]))
        if is_data:
            break
        # a body-system row (text in the first cell only) after the header ends it
        if head_n and first["text"] and not first["copy"] and not any(c["text"] for c in r[1:] if not c.get("colcopy")):
            break
        head_n += 1
    if head_n == len(grid):
        return None
    headers = []
    for j in range(width):
        parts = []
        for r in grid[:head_n]:
            t = r[j]["text"]
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        headers.append(clean(" ".join(parts)))
    # logical columns: a header cell spanning two grid columns with nothing to
    # tell them apart is one arm. Some SPL tables span header and body cells
    # differently (lurasidone Table 20); when a body row has exactly one
    # original cell per logical column, its cells are re-seated onto them.
    logical = [j for j in range(1, width)
               if not (j > 1 and headers[j] == headers[j - 1] and head_n and grid[head_n - 1][j].get("colcopy"))]
    data = []
    for r in grid[head_n:]:
        orig = [j for j in range(1, width) if not r[j].get("colcopy")]
        if len(orig) == len(logical) and orig != logical and r[0]["text"]:
            r = list(r)
            cells = [r[j] for j in orig]
            for j, c in zip(logical, cells):
                r[j] = {**c, "reseated": True}
        data.append(r)
    npct_table = bool(re.search(r"\(\s*%\s*\)", tab["caption"] + " " + " ".join(headers)))
    cols = []
    for j in range(1, width):
        h = headers[j]
        low = h.lower()
        last = clean(grid[head_n - 1][j]["text"]).lower() if head_n else ""
        if last in ("n", "no.", "number", "n (no.)"):
            continue  # count sub-column next to a % sub-column
        if j not in logical:
            continue  # one arm header spanning two grid columns
        body = [r for r in data if r[0]["text"]]
        if body and all(r[j].get("colcopy") for r in body if r[j]["text"]):
            continue  # a colspan duplicate of the column to its left
        m = N_RE.search(h)
        n = int(m.group(1).replace(",", "")) if m else None
        vals = [parse_value(r[j]["text"], npct_table, n) for r in body]
        numeric = sum(1 for v in vals if v is not None and v["flag"] is None)
        if numeric == 0 or numeric < 0.3 * max(1, sum(1 for v in vals if v is not None and v["flag"] != "dash_as_zero")):
            continue
        role = "placebo" if "placebo" in low else "drug"
        names_drug = any(re.search(r"\b" + re.escape(nm) + r"\b", low) for nm in names)
        pooled = any(re.search(r"\b" + w, low) for w in POOLED_WORDS)
        cols.append({"j": j, "header": h, "role": role, "n": n, "names_drug": names_drug, "pooled": pooled})
    groups = [("", cols)]
    for level in range(head_n):
        row = grid[level]
        by: dict[int, list[dict]] = {}
        for c in cols:
            by.setdefault(row[c["j"]]["uid"], []).append(c)
        valid = [g for g in by.values() if any(c["role"] == "placebo" for c in g)
                 and any(c["role"] == "drug" for c in g)]
        if len(valid) >= 2:
            groups = [(row[g[0]["j"]]["text"], g) for g in valid]
            break
    # 'Body System | Preferred Term | arms...': the term is the text column just
    # before the first arm, not column 0
    if cols and cols[0]["j"] > 1:
        lj = cols[0]["j"] - 1
        texty = [r for r in data if r[lj]["text"] and not r[lj].get("colcopy") and not is_number_cell(r[lj]["text"])]
        if len(texty) >= 0.5 * max(1, sum(1 for r in data if r[0]["text"])):
            data = [[r[lj]] + r[1:] if r[lj]["text"] and not r[lj].get("colcopy") else r for r in data]
    counts = count_table(tab["caption"], headers, data, cols)
    out = []
    for label, gcols in groups:
        placebo = [c for c in gcols if c["role"] == "placebo"]
        drug = [c for c in gcols if c["role"] == "drug"]
        flags = []
        if any(c["names_drug"] for c in drug):
            drug = [c for c in drug if c["names_drug"]]
        elif drug:
            flags.append("drug_column_unnamed")
        if len(groups) > 1:
            flags.append("column_group")
        cap = clean(tab["caption"] + (f" [column group: {label}]" if len(groups) > 1 else ""))
        if counts:
            flags.append("counts_converted")
        out.append({"caption": cap, "headers": headers, "head_n": head_n, "data": data,
                    "placebo": placebo, "drug": drug, "npct": npct_table, "flags": flags,
                    "counts": counts})
    return out


def count_table(caption: str, headers: list[str], data: list, cols: list[dict]) -> bool:
    """Amendment A6: a table of patient counts rather than percentages.

    True only when nothing marks the values as percentages (no '%', 'percent'
    or 'incidence' in caption or headers, no '%' or decimal in any value cell),
    every arm column states its N, every value is an integer no larger than
    its column's N, and at least one value cannot be a rounded percentage of its
    column's N (no k <= N with round(100k/N) equal to it). Such cells are converted to 100*count/N (pimozide: 'Sedation
    14' with N=20 is 70%, not 14%)."""
    text = (caption + " " + " ".join(headers)).lower()
    if not cols or re.search(r"%|percent|incidence|rate", text):
        return False
    if not all(c["n"] for c in cols):
        return False
    impossible = 0
    for r in data:
        if not r[0]["text"]:
            continue
        for c in cols:
            t = strip_marks(r[c["j"]]["text"])
            if t in ("", "-", "–", "—", "--", "−"):
                continue
            if not re.fullmatch(r"\d+", t) or int(t) > c["n"]:
                return False
            # an integer percentage is only possible if some k <= N rounds to it
            if not any(round(100.0 * k / c["n"]) == int(t) for k in range(c["n"] + 1)):
                impossible += 1
    return impossible > 0


# ------------------------------------------------------------------ term matching

def norm_label(text: str, note: str) -> tuple[str, str]:
    """(normalised label without parenthetical, includes-note)."""
    s = strip_marks(text).lower()
    paren = " ".join(re.findall(r"\(([^)]*)\)", s))
    s = clean(re.sub(r"\([^)]*\)", " ", s))
    s = strip_marks(s)
    return s, clean(paren + " " + note).lower()


def components(label: str) -> list[str]:
    return [clean(p) for p in re.split(r"/|,| or ", label) if clean(p)]


# amendment A3: inflectional variants of a mapped term, applied to row labels only
VARIANTS = [(r"\bevent$", "events"), (r"\bincrease$", "increased"), (r"\bdecrease$", "decreased"),
            (r"\babnormalit(y|ies)$", "abnormal"), (r"^dystonic events?$", "dystonia"),
            (r"^(akathisia|parkinsonism|dystonia) events?$", r"\1"), (r"^drowsy$", "drowsiness"),
            (r"^daytime drowsiness$", "drowsiness")]


def variants(label: str) -> list[str]:
    out = [label]
    for pat, rep in VARIANTS:
        v = re.sub(pat, rep, out[-1])
        if v != out[-1]:
            out.append(v)
    return out


def match_row(label: str, note: str, spec: dict) -> tuple[bool, bool]:
    """(matches, grouped) for one liability spec."""
    terms = set(spec["terms"])
    partial = bool(re.match(r"other\b", label))  # 'other EPS' is not the whole group
    base = re.sub(r"^(any|other|all)\s+", "", label)
    labs = variants(base)
    comps = [v for c in components(base) for v in variants(c)]
    hits = [c for c in comps if c in terms]
    if not any(lb in terms for lb in labs) and not hits:
        return False, False
    grouped = not partial and (any(lb in set(spec["grouped_terms"]) for lb in labs) or len(set(hits)) >= 2
                               or bool(re.search(r"includ|combin", note)))
    return True, grouped


# ------------------------------------------------------------------ extraction

def arm_values(st: dict, row: list[dict]) -> tuple[list[dict], dict | None, list[str]]:
    flags = []

    def val(c):
        v = parse_value(row[c["j"]]["text"], st["npct"], c["n"])
        if v is None or v["pct"] is None:
            flags.append("unparsed_cell")
            return None
        if v["flag"]:
            flags.append(v["flag"])
        if v["censored"]:
            flags.append("censored")
        if st.get("counts") and c["n"]:
            v = {**v, "pct": round(100.0 * v["pct"] / c["n"], 3)}
        return v
    pv = [(c, val(c)) for c in st["placebo"]]
    pv = [(c, v) for c, v in pv if v is not None]
    if not pv:
        return [], None, flags + ["no_placebo_value"]
    pl = pool([(v["pct"], c["n"]) for c, v in pv])
    n_pl = sum(c["n"] for c, _ in pv) if all(c["n"] for c, _ in pv) else None
    placebo = {"pct": pl, "n": n_pl, "raw": [v["raw"] for _, v in pv]}
    arms = []
    for c in st["drug"]:
        v = val(c)
        if v is None:
            continue
        arms.append({"dose": c["header"], "drug_pct": v["pct"], "placebo_pct": pl,
                     "n_drug": c["n"], "n_placebo": n_pl, "pooled_column": c["pooled"],
                     "raw": v["raw"], "censored": v["censored"],
                     "rd": round(v["pct"] - pl, 3)})
    return arms, placebo, flags


def pool(pairs: list[tuple[float, int | None]]) -> float:
    if all(n for _, n in pairs):
        tot = sum(n for _, n in pairs)
        return round(sum(p * n for p, n in pairs) / tot, 3)
    return round(sum(p for p, _ in pairs) / len(pairs), 3)


def liability_from_table(st: dict, spec: dict) -> dict:
    """Per-arm values for one liability in one placebo-controlled table."""
    matched = []
    for row in st["data"]:
        if not row[0]["text"] or row[0]["copy"]:
            continue
        label, note = norm_label(row[0]["text"], row[0]["note"])
        ok, grouped = match_row(label, note, spec)
        if ok:
            matched.append((row, label, grouped))
    if not matched:
        return {"present": False}
    use = [m for m in matched if m[2]] or matched
    per_row = []
    flags = []
    for row, label, grouped in use:
        arms, placebo, fl = arm_values(st, row)
        flags += fl
        if arms:
            per_row.append((label, grouped, arms, placebo))
    if not per_row:
        return {"present": True, "unparsed": True, "terms": [m[1] for m in use], "flags": sorted(set(flags))}
    # Max over terms. The rules say "max, never sum"; the max is taken on the
    # pooled risk difference (ties: higher drug %), so the reported arms all
    # come from one row and drug % and placebo % stay paired.
    summaries = [summarise(label, grouped, arms) for label, grouped, arms, _ in per_row]
    best = max(summaries, key=lambda x: (x["rd"], x["drug_pct"]))
    best.update({"present": True, "terms": sorted({m[1] for m in use}),
                 "grouped": any(m[2] for m in use),
                 "term_values": [{"term": x["term"], "drug_pct": x["drug_pct"], "placebo_pct": x["placebo_pct"],
                                  "rd": x["rd"]} for x in summaries],
                 "censored": best["censored"],
                 "flags": sorted(set(flags))})
    return best


def summarise(label: str, grouped: bool, arms: list[dict]) -> dict:
    pooled_cols = [a for a in arms if a["pooled_column"]]
    dose_arms = [a for a in arms if not a["pooled_column"]] or pooled_cols
    if pooled_cols:
        src = pooled_cols[0]
        drug_pct, method, placebo_pct = src["drug_pct"], "pooled_column", src["placebo_pct"]
    else:
        drug_pct = pool([(a["drug_pct"], a["n_drug"]) for a in dose_arms])
        method = "n_weighted_mean" if all(a["n_drug"] for a in dose_arms) else "unweighted_mean"
        if len(dose_arms) == 1:
            method = "single_arm"
        placebo_pct = dose_arms[0]["placebo_pct"]
    return {"term": label, "arms": [{**a, "term": label} for a in arms], "drug_pct": drug_pct,
            "placebo_pct": placebo_pct, "rd": round(drug_pct - placebo_pct, 3), "pool_method": method,
            "max_arm_rd": max(a["rd"] for a in arms), "censored": any(a["censored"] for a in arms)}


def binary(rule: dict, drug_pct: float, placebo_pct: float) -> int:
    return int(drug_pct - placebo_pct >= rule["rd_pp"] and drug_pct >= rule["ratio"] * placebo_pct)


def table_n(st: dict) -> int | None:
    ns = [c["n"] for c in st["drug"] + st["placebo"] if not c["pooled"]]
    return sum(ns) if ns and all(ns) else None


def primary_index(tables: list[tuple[int, dict]]) -> int:
    """Pre-specified primary-table rule over (table_index, structure) pairs."""
    cand = tables
    adult = [t for t in cand if not PEDIATRIC.search(t[1]["caption"])]
    cand = adult or cand
    mono = [t for t in cand if not ADJUNCT.search(t[1]["caption"])]
    cand = mono or cand
    # amendment A2: an n tie goes to the table with more data rows (the complete
    # table rather than a ">=5% and twice placebo" subset of it), then label order
    return sorted(cand, key=lambda t: (table_n(t[1]) is None, -(table_n(t[1]) or 0),
                                       -len(t[1]["data"]), tables.index(t)))[0][0]


def threshold_text(caption: str, prose: str) -> str | None:
    if "%" in caption or "≥" in caption:
        return caption
    m = re.match(r"\s*(table\s*\d+[a-z]?)", caption, re.I)
    sents = re.split(r"(?<=[.;])\s+", prose or "")
    if m:
        key = m.group(1).lower().replace(" ", "")
        for s in sents:
            if key in s.lower().replace(" ", "") and ("%" in s or "≥" in s):
                return clean(s)
    return None


def drug_names(inn: str, label: dict) -> list[str]:
    of = label.get("openfda") or {}
    names = {inn.lower()}
    names.update(n.lower() for n in of.get("brand_name", []))
    names.update(n.lower() for n in of.get("generic_name", []))
    names.update(w for n in of.get("generic_name", []) for w in [n.lower().split()[0]])
    return sorted(n for n in names if len(n) >= 4)


def single_ingredient(label: dict) -> bool:
    of = label.get("openfda") or {}
    subs = of.get("substance_name") or []
    gen = " ".join(of.get("generic_name") or []).lower()
    return len(subs) <= 1 and " and " not in gen and "/" not in gen


def systemic(meta: dict, atc: list[str]) -> bool:
    routes = {r.upper() for r in meta.get("route") or []}
    local = LOCAL_ROUTES | (set() if any(c.startswith("N") for c in atc) else AIRWAY_ROUTES)
    return not routes or bool(routes - local)


def analyse_label(inn: str, label: dict) -> dict:
    names = drug_names(inn, label)
    out = {"placebo": [], "other": [], "n_tables": 0, "unparseable": 0}
    for i, html in enumerate(label.get("adverse_reactions_table") or []):
        out["n_tables"] += 1
        tab = parse_table(html)
        sts = structure(tab, names) if tab else None
        if sts is None:
            out["unparseable"] += 1
            continue
        for g, st in enumerate(sts):
            key = i if len(sts) == 1 else f"{i}.{g}"
            if DISCONTINUATION.search(st["caption"]):
                out["discontinuation"] = out.get("discontinuation", 0) + 1
                continue  # amendment A1: discontinuation tables are not incidence tables
            if st["placebo"] and st["drug"]:
                out["placebo"].append((key, st))
            elif st["drug"]:
                out["other"].append((key, st))
    return out


def select_label(drug: dict, labels: dict[str, dict]) -> tuple[dict | None, dict | None, dict | None, list[str]]:
    # amendment A7: multi-ingredient and local-route labels are never used
    # (they were fallback tiers), and oral labels come first so a drug is
    # scored from its tablet label rather than a depot injection.
    metas = [m for m in drug["labels"] if m["set_id"] in labels
             and single_ingredient(labels[m["set_id"]]) and systemic(m, drug.get("atc") or [])]
    metas.sort(key=lambda m: "ORAL" not in (m.get("route") or []))
    tiers = [(metas, [])]
    for want in ("placebo", "table", "prose"):
        for tier, fl in tiers:
            for m in tier:
                lab = labels[m["set_id"]]
                if want == "prose":
                    if lab.get("adverse_reactions"):
                        return m, lab, None, fl
                    continue
                an = analyse_label(drug["inn"], lab)
                if want == "placebo" and an["placebo"]:
                    return m, lab, an, fl
                if want == "table" and an["other"]:
                    return m, lab, an, fl
    return None, None, None, []


def split_key(idx):
    """Table key -> (table_index, column_group). A key 'i.g' is column group g
    of table i (a table with side-by-side placebo-controlled populations)."""
    if isinstance(idx, str):
        i, g = idx.split(".")
        return int(i), int(g)
    return idx, None


def provenance(m: dict, lab: dict, idx, caption: str | None) -> dict:
    ti, grp = split_key(idx)
    return {"source": "FDA SPL (openFDA bulk drug label export)", "set_id": lab["set_id"],
            "version": lab.get("version"), "effective_time": lab.get("effective_time"),
            "application_number": m.get("application_number"),
            "table_index": ti, "column_group": grp, "population": caption}


def build(rules: dict, drugs: list[dict], labels: dict[str, dict]) -> list[dict]:
    rule = rules["_binary"]
    rows = []
    for d in drugs:
        inn = d["inn"]
        m, lab, an, sel_flags = select_label(d, labels)
        rec = {"inn": inn}
        if lab is None:
            for L in LIABS:
                rec[L] = {"basis": "absent", "label": None, "flags": ["no_eligible_label"],
                          "arms": [], "source": None}
            rows.append(rec)
            continue
        prose = " ".join(lab.get("adverse_reactions") or [])
        if an and an["placebo"]:
            pidx = primary_index(an["placebo"])
            sts = dict(an["placebo"])
            for L in LIABS:
                spec = rules["liabilities"][L]
                per = {}
                for i, st in an["placebo"]:
                    per[i] = liability_from_table(st, spec)
                p = per[pidx]
                st = sts[pidx]
                flags = list(sel_flags) + list(st["flags"])
                base = {"basis": "placebo_subtracted", "source": provenance(m, lab, pidx, st["caption"]),
                        "threshold_text": threshold_text(st["caption"], prose),
                        "rule": rules["_rule_text"]}
                if not p["present"]:
                    entry = {**base, "label": 0, "drug_pct": None, "placebo_pct": None,
                             "risk_difference": None, "max_arm_rd": None, "terms": [],
                             "arms": [{"dose": c["header"], "drug_pct": None, "placebo_pct": None,
                                       "n_drug": c["n"],
                                       "n_placebo": sum(c["n"] for c in st["placebo"]) if all(c["n"] for c in st["placebo"]) else None}
                                      for c in st["drug"]],
                             "placebo_arm": {"header": [c["header"] for c in st["placebo"]],
                                             "n": sum(c["n"] for c in st["placebo"]) if all(c["n"] for c in st["placebo"]) else None},
                             "flags": flags + ["absent_below_threshold"]}
                elif p.get("unparsed"):
                    entry = {**base, "label": None, "drug_pct": None, "placebo_pct": None,
                             "risk_difference": None, "max_arm_rd": None, "terms": p["terms"], "arms": [],
                             "placebo_arm": {"header": [c["header"] for c in st["placebo"]], "n": None},
                             "flags": flags + p["flags"] + ["extraction_failed"]}
                else:
                    lab01 = binary(rule, p["drug_pct"], p["placebo_pct"])
                    fl = flags + p["flags"] + (["label_depends_on_censored"] if p["censored"] else [])
                    if p["grouped"]:
                        fl.append("grouped_term")
                    entry = {**base, "label": lab01, "drug_pct": p["drug_pct"], "placebo_pct": p["placebo_pct"],
                             "risk_difference": p["rd"], "max_arm_rd": p["max_arm_rd"],
                             "pool_method": p["pool_method"], "terms": p["terms"],
                             "arms": [{k: a[k] for k in ("dose", "drug_pct", "placebo_pct", "n_drug", "n_placebo", "rd", "raw", "term")}
                                      for a in p["arms"]],
                             "placebo_arm": {"header": [c["header"] for c in st["placebo"]],
                                             "n": p["arms"][0]["n_placebo"]},
                             "flags": sorted(set(fl))}
                others = []
                for i, q in per.items():
                    if i == pidx:
                        continue
                    o = {"table_index": split_key(i)[0], "column_group": split_key(i)[1], "population": sts[i]["caption"], "present": q["present"]}
                    if q["present"] and not q.get("unparsed"):
                        o.update({"drug_pct": q["drug_pct"], "placebo_pct": q["placebo_pct"],
                                  "risk_difference": q["rd"], "label": binary(rule, q["drug_pct"], q["placebo_pct"]),
                                  "terms": q["terms"],
                                  "arms": [{k: a[k] for k in ("dose", "drug_pct", "placebo_pct", "n_drug", "n_placebo", "rd")}
                                           for a in q["arms"]]})
                    elif q["present"]:
                        o["flags"] = ["extraction_failed"]
                    else:
                        o["flags"] = ["absent_below_threshold"]
                    others.append(o)
                entry["other_tables"] = others
                rec[L] = entry
        else:
            # absolute fallback: no placebo-controlled table in any qualifying label
            for L in LIABS:
                spec = rules["liabilities"][L]
                terms_re = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in spec["terms"] if len(t) > 3]
                listed = any(r.search(prose) for r in terms_re)
                entry = {"basis": "absolute_fallback", "flagged": True, "drug_pct": None, "placebo_pct": None,
                         "risk_difference": None, "arms": [], "listed_in_prose": listed,
                         "rule": rules["_fallback_text"], "flags": list(sel_flags) + ["no_placebo_table"]}
                if an and an["other"]:
                    best, where = None, None
                    for i, st in an["other"]:
                        for row in st["data"]:
                            if not row[0]["text"] or row[0]["copy"]:
                                continue
                            label, note = norm_label(row[0]["text"], row[0]["note"])
                            if not match_row(label, note, spec)[0]:
                                continue
                            for c in st["drug"]:
                                v = parse_value(row[c["j"]]["text"], st["npct"], c["n"])
                                if v and v["pct"] is not None and st.get("counts") and c["n"]:
                                    v = {**v, "pct": round(100.0 * v["pct"] / c["n"], 3)}
                                if v and v["pct"] is not None and (best is None or v["pct"] > best["drug_pct"]):
                                    best = {"dose": c["header"], "drug_pct": v["pct"], "placebo_pct": None,
                                            "n_drug": c["n"], "n_placebo": None, "term": label, "raw": v["raw"]}
                                    where = (i, st["caption"])
                    if best:
                        entry.update({"label": int(best["drug_pct"] >= rules["_fallback_pct"]),
                                      "drug_pct": best["drug_pct"], "arms": [best],
                                      "source": provenance(m, lab, where[0], where[1])})
                    else:
                        entry.update({"label": 0, "source": provenance(m, lab, an["other"][0][0], an["other"][0][1]["caption"])})
                        entry["flags"].append("absent_from_table")
                else:
                    entry.update({"label": None, "source": provenance(m, lab, None, None)})
                    entry["flags"].append("prose_only")
                rec[L] = entry
        # amendment A7: weight gain is withheld. Labels report it mostly
        # outside the adverse-reaction tables (a "Weight Gain" warning with % of
        # patients gaining >=7%), so a table-only read calls olanzapine
        # negative. The extracted numbers stay for reference; the label waits
        # for a source that measures weight directly.
        w = rec["weight_gain"]
        if w.get("label") is not None:
            w["label"] = None
            w["flags"] = sorted(set(w.get("flags", [])) | {"label_withheld_A7"})
        rows.append(rec)
    return rows


def load_rules() -> tuple[dict, str]:
    raw = RULES.read_bytes()
    rules = json.loads(raw)
    # the machine-readable parameters of the pre-specified binary rule
    par = rules["binary_rule"]["parameters"]
    rules["_binary"] = {"rd_pp": par["rd_pp"], "ratio": par["ratio"]}
    rules["_rule_text"] = rules["binary_rule"]["placebo_subtracted"]
    rules["_fallback_pct"] = par["fallback_drug_pct"]
    rules["_fallback_text"] = rules["binary_rule"]["absolute_fallback"]
    for L in LIABS:
        spec = rules["liabilities"][L]
        spec["terms"] = [norm_label(t, "")[0] for t in spec["terms"]]
        spec["grouped_terms"] = [norm_label(t, "")[0] for t in spec["grouped_terms"]]
    return rules, hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--show")
    a = ap.parse_args()
    if a.build_cache:
        build_cache()
        return 0
    rules, sha = load_rules()
    meta = json.loads(DRUGS.read_text())
    labels = load_cache()
    drugs = meta["drugs"]
    if a.show:
        drugs = [d for d in drugs if d["inn"] == a.show]
        print(json.dumps(build(rules, drugs, labels), indent=1))
        return 0
    rows = build(rules, drugs, labels)
    out = {
        "description": ("v2 receptor-anchored CNS liability labels (P33.2) from FDA adverse-reaction tables. "
                        "basis 'placebo_subtracted' = drug vs placebo arm of a placebo-controlled table; "
                        "'absolute_fallback' = no placebo table in any qualifying label, FLAGGED, absolute-incidence rule; "
                        "'absent' = no usable label. Built by scripts/v2_liabilities.py; no held-out field is stored, "
                        "consumers apply pipeline.guards.v2_heldout(inn)."),
        "rules_file": str(RULES.relative_to(ROOT)),
        "rules_sha256": sha,
        "binary_rule": rules["_rule_text"],
        "fallback_rule": rules["_fallback_text"],
        "liabilities": list(LIABS),
        "label_export": meta.get("openfda_label_export"),
        "drugs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    summary(rows)
    return 0


def summary(rows: list[dict]) -> None:
    for L in LIABS:
        c: dict[str, int] = {}
        for r in rows:
            e = r[L]
            k = e["basis"]
            c[k] = c.get(k, 0) + 1
            if e.get("label") == 1:
                c[k + "+"] = c.get(k + "+", 0) + 1
            if "absent_below_threshold" in e.get("flags", []):
                c["below_thr"] = c.get("below_thr", 0) + 1
            if "extraction_failed" in e.get("flags", []):
                c["failed"] = c.get("failed", 0) + 1
        print(L, c)


if __name__ == "__main__":
    sys.exit(main())
