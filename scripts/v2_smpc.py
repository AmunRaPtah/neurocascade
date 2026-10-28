#!/usr/bin/env python3
"""Placebo-subtracted liability labels from SmPC section 4.8 tables (P34.2).

What it is for
--------------
docs/prereg-p34-go.md lets the P34.4 fresh held-out set take liability labels
from an SmPC only through the same placebo-subtracted rule as the openFDA
labels, and only through a parser frozen in P34.2 on train drugs. This is that
parser. Its rules are ``benchmark/v2/smpc_rules.json``; the term map, cell
parsing, primary-table rule, pooling and 5-point / 2x margin are NOT
re-implemented: each SmPC table is normalised into the shape of an SPL table
and handed to ``scripts/v2_liabilities.py`` (``build``), unchanged.

What it does not do
-------------------
It never applies the openFDA absolute-incidence fallback (SmPC frequency
categories are not incidences in any arm), never reads a number from prose,
and never replaces an openFDA placebo-subtracted label (``precedence``).

Source: the electronic medicines compendium (medicines.org.uk/emc), the only
HTML SmPCs found; see ``smpc_rules.json`` ``source.why``.

    scripts/v2_smpc.py --show bilastine          # one drug's record (network/cache)
    scripts/v2_smpc.py --validate-train OUT.json # train-drug agreement with openFDA labels
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

import lxml.html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v2_liabilities as V  # noqa: E402

SMPC_RULES = V.V2 / "smpc_rules.json"
EMC = "https://www.medicines.org.uk/emc"
CACHE = Path(os.environ.get("NC_V2_CACHE", "/root/nc-v2-cache")) / "smpc"
SOURCE = "UK SmPC section 4.8 (emc)"


def load_smpc_rules() -> tuple[dict, str]:
    raw = SMPC_RULES.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------ fetch

def fetch(url: str, cache: Path = CACHE) -> str:
    """GET with an on-disk cache keyed by URL hash; three tries with backoff."""
    p = cache / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if p.exists():
        return p.read_text()
    err = None
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (neurocascade research)"})
            txt = urllib.request.urlopen(req, timeout=40).read().decode("utf8", "replace")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
            time.sleep(0.5)
            return txt
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            err = e
            time.sleep(2 + 3 * k)
    raise err


def search_products(inn: str, get=fetch) -> list[str]:
    html = get(f"{EMC}/search?q={urllib.parse.quote(inn)}")
    out: list[str] = []
    for pid in re.findall(r'href="/emc/product/(\d+)/smpc', html):
        if pid not in out:
            out.append(pid)
    return out


# ------------------------------------------------------------------ page

def _section(doc, sid: str):
    s = doc.find(f".//summary[@id='{sid}']")
    return None if s is None else s.getparent()


def product_meta(pid: str, html: str) -> dict:
    """Product id, name, active ingredients, section 3 text and section 4.8 element."""
    doc = lxml.html.fromstring(html)
    name, ingredients = "", []
    for sc in doc.iter("script"):
        if sc.get("type") == "application/ld+json":
            try:
                ld = json.loads(sc.text or "")
            except ValueError:
                continue
            if ld.get("@type") == "Drug":
                name, ingredients = ld.get("name", ""), list(ld.get("activeIngredient") or [])
    form = _section(doc, "FORM")
    return {"pid": pid, "name": name, "active_ingredients": ingredients,
            "form": V.clean(form.text_content()) if form is not None else "",
            "section_48": _section(doc, "UNDESIRABLE_EFFECTS"),
            "url": f"{EMC}/product/{pid}/smpc"}


def single_ingredient(meta: dict, inn: str) -> bool:
    ai = meta["active_ingredients"]
    if len(ai) != 1:
        return False
    a, i = ai[0].lower().strip(), inn.lower()
    return a == i or a.startswith(i + " ")


def systemic(meta: dict, sel: dict) -> bool:
    form = meta["form"].lower()
    return not any(w in form for w in sel["local_forms"])


def oral(meta: dict, sel: dict) -> bool:
    form = meta["form"].lower()
    return any(w in form for w in sel["oral_forms"])


# ------------------------------------------------------------------ tables

LINE_BREAK = re.compile(r"<\s*/?\s*(?:p|br)\b[^>]*>", re.I)


def cell_lines(cell) -> list[str]:
    """Non-empty text lines of a cell, split at <p> and <br> boundaries
    (including the malformed self-closing '<p />' some emc tables use)."""
    inner = (cell.text or "") + "".join(lxml.html.tostring(c, encoding="unicode") for c in cell)
    out = []
    for piece in LINE_BREAK.split(inner):
        t = V.clean(lxml.html.fromstring(f"<div>{piece}</div>").text_content()) if piece.strip() else ""
        if t:
            out.append(t)
    return out


def split_stacked_rows(table) -> None:
    """S1: split a row whose cells stack several terms and values as lines.

    With k = the line count shared by every value cell (cells after the first;
    empty cells allowed), a label cell of k lines gives k rows, and one of k+1
    lines gives a heading row (its first line, no values) plus k rows. Any
    other shape, k < 2 with no heading line, or a row with no numeric value
    (header rows) is left alone."""
    for tr in list(table.iter("tr")):
        cells = [c for c in tr if c.tag in ("td", "th")]
        if len(cells) < 2:
            continue
        lines = [cell_lines(c) for c in cells]
        if not any(V.is_number_cell(t) for x in lines[1:] for t in x):
            continue  # header and heading rows are never split
        ks = {len(x) for x in lines[1:] if x}
        if len(ks) != 1:
            continue
        k = ks.pop()
        lab = lines[0]
        if len(lab) == k + 1:
            heading, lab = lab[0], lab[1:]
        elif len(lab) == k and k >= 2:
            heading = None
        else:
            continue
        rows = []
        if heading is not None:
            rows.append([heading] + [""] * (len(cells) - 1))
        for i in range(k):
            rows.append([lab[i]] + [x[i] if x else "" for x in lines[1:]])
        parent, at = tr.getparent(), tr.getparent().index(tr)
        parent.remove(tr)
        for n, texts in enumerate(rows):
            row = lxml.html.Element("tr")
            for c, t in zip(cells, texts):
                td = lxml.html.Element(c.tag)
                if c.get("colspan"):
                    td.set("colspan", c.get("colspan"))
                td.text = t
                row.append(td)
            parent.insert(at + n, row)


def preceding_caption(section, table, n: int = 2) -> str:
    """S0: the last n non-empty paragraphs before the table that are not inside a table."""
    paras = []
    for el in section.iter():
        if el is table:
            break
        if el.tag == "p" and not any(a.tag == "table" for a in el.iterancestors()):
            t = V.clean(el.text_content())
            if t:
                paras.append(t)
    return " ".join(paras[-n:])


def _is_freq(text: str, words: set[str]) -> bool:
    return V.clean(text).lower().rstrip(":") in words


def normalise(tab: dict, words: set[str]) -> str:
    """S2: rewrite a parse_table grid as one label column + the value columns, as HTML."""
    grid, width = tab["grid"], tab["width"]
    def is_title(r):
        return len({c["uid"] for c in r if c["text"]}) == 1 and not any(V.is_number_cell(c["text"]) for c in r)

    # header rows end at the first numeric row or, after at least one header
    # row, at the first full-width SOC title row
    head_n = 0
    for r in grid:
        if any(V.is_number_cell(c["text"]) for c in r) or (head_n and is_title(r)):
            break
        head_n += 1
    value_cols = [j for j in range(width)
                  if any(V.is_number_cell(r[j]["text"]) and not r[j].get("colcopy") for r in grid[head_n:])]
    first = min(value_cols) if value_cols else width
    value_cols = list(range(first, width))

    def label(r, head):
        cells = r[:first]
        if head:
            parts = []
            for c in cells:
                if c["text"] and (not parts or parts[-1] != c["text"]):
                    parts.append(c["text"])
            return " ".join(parts)
        for c in reversed(cells):
            if c["text"] and not _is_freq(c["text"], words):
                return c["text"]
        return ""

    def tr(r, head):
        tag = "th" if head else "td"
        out = [f"<{tag}>{escape(label(r, head))}</{tag}>"]
        # a title/SOC row: one cell spanning the whole width carries no values
        title = not head and is_title(r)
        j = first
        while j < width:
            c, span = r[j], 1
            while j + span < width and r[j + span]["uid"] == c["uid"] and r[j + span].get("colcopy"):
                span += 1
            txt = "" if title else c["text"]
            cs = f' colspan="{span}"' if span > 1 else ""
            out.append(f"<{tag}{cs}>{escape(txt)}</{tag}>")
            j += span
        return "<tr>" + "".join(out) + "</tr>"

    cap = f"<caption>{escape(tab['caption'])}</caption>" if tab["caption"] else ""
    head = "".join(tr(r, True) for r in grid[:head_n])
    body = "".join(tr(r, False) for r in grid[head_n:])
    return f"<table>{cap}<thead>{head}</thead><tbody>{body}</tbody></table>"


LAB = re.compile(r"laborator", re.I)


def head_text(html: str) -> str:
    """Caption plus header-row text of a normalised table."""
    t = lxml.html.fromstring(html)
    cap = t.find(".//caption")
    head = t.find(".//thead")
    return " ".join(V.clean(x.text_content()) for x in (cap, head) if x is not None)


def section_tables(section, rules: dict) -> list[str]:
    """Normalised HTML for every table in section 4.8, in document order."""
    words = set(rules["table_normalisation"]["frequency_words"])
    out = []
    for table in list(section.iter("table")):
        cap = preceding_caption(section, table)
        t = lxml.html.fromstring(lxml.html.tostring(table))
        split_stacked_rows(t)
        tab = V.parse_table(lxml.html.tostring(t, encoding="unicode"))
        if tab is None:
            continue
        if not tab["caption"]:
            tab["caption"] = cap
        html = normalise(tab, words)
        if LAB.search(head_text(html)):
            continue  # amendment SA1: laboratory-parameter tables are not adverse-reaction tables
        out.append(html)
    return out


# ------------------------------------------------------------------ labels

def label_from_product(inn: str, meta: dict, rules: dict, smpc_rules: dict) -> dict | None:
    """The v2 liability record for one SmPC, or None without a placebo-controlled table."""
    if meta["section_48"] is None:
        return None
    sid = f"emc:{meta['pid']}"
    lab = {"set_id": sid, "adverse_reactions_table": section_tables(meta["section_48"], smpc_rules),
           "adverse_reactions": [V.clean(meta["section_48"].text_content())],
           "openfda": {"generic_name": [inn], "substance_name": meta["active_ingredients"],
                       "brand_name": [meta["name"].split()[0]] if meta["name"] else []}}
    an = V.analyse_label(inn, lab)
    if not an["placebo"]:
        return None
    drug = {"inn": inn, "labels": [{"set_id": sid, "route": None}], "atc": []}
    rec = V.build(rules, [drug], {sid: lab})[0]
    for L in V.LIABS:
        e = rec[L]
        if e.get("basis") != "placebo_subtracted":
            return None
        src = e["source"]
        e["source"] = {"source": SOURCE, "emc_product_id": meta["pid"], "product_name": meta["name"],
                       "url": meta["url"], "table_index": src["table_index"],
                       "column_group": src["column_group"], "population": src["population"]}
        e["flags"] = sorted(set(e.get("flags", [])) | {"smpc"})
        for o in e.get("other_tables", []):
            o["source"] = SOURCE
    return rec


def smpc_record(inn: str, rules: dict, smpc_rules: dict, get=fetch) -> dict:
    """Product selection (smpc_rules.product_selection) and the chosen product's record."""
    sel = smpc_rules["product_selection"]
    metas = [product_meta(pid, get(f"{EMC}/product/{pid}/smpc")) for pid in search_products(inn, get)]
    eligible = [m for m in metas if single_ingredient(m, inn) and systemic(m, sel)]
    eligible.sort(key=lambda m: (not oral(m, sel), int(m["pid"])))
    tried = []
    for m in eligible:
        tried.append(m["pid"])
        rec = label_from_product(inn, m, rules, smpc_rules)
        if rec is not None:
            return {"inn": inn, "basis": "placebo_subtracted", "products_seen": len(metas),
                    "products_eligible": [x["pid"] for x in eligible], "record": rec}
    return {"inn": inn, "basis": "absent", "products_seen": len(metas),
            "products_eligible": [x["pid"] for x in eligible], "record": None,
            "reason": ("no emc SmPC" if not metas else "no single-ingredient systemic product" if not eligible
                       else "no placebo-controlled table in section 4.8 of any eligible product")}


# ------------------------------------------------------------------ train validation

def compare(fda: dict, smpc: dict) -> list[dict]:
    """Per-liability agreement where both sources give a non-null placebo-subtracted label."""
    out = []
    for L in V.LIABS:
        f, s = fda[L], smpc[L]
        if f.get("basis") != "placebo_subtracted" or f.get("label") is None or s.get("label") is None:
            continue
        out.append({"liability": L, "fda_label": f["label"], "smpc_label": s["label"],
                    "fda_drug_pct": f.get("drug_pct"), "fda_placebo_pct": f.get("placebo_pct"),
                    "smpc_drug_pct": s.get("drug_pct"), "smpc_placebo_pct": s.get("placebo_pct"),
                    "agree": f["label"] == s["label"]})
    return out


def validate_train(get=fetch) -> dict:
    """Run the parser on every train INN and compare with the frozen openFDA labels."""
    sys.path.insert(0, str(ROOT / "src"))
    from neurocascade.benchmark.v2 import v2_partition
    train, _ = v2_partition()
    rules, _ = V.load_rules()
    smpc_rules, _ = load_smpc_rules()
    fda = {r["inn"]: r for r in json.loads(V.OUT.read_text())["drugs"]}
    def one(inn):
        try:
            return smpc_record(inn, rules, smpc_rules, get)
        except Exception as e:  # noqa: BLE001 - recorded, not hidden
            return {"inn": inn, "basis": "fetch_failed", "reason": str(e), "record": None}

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(one, train))
    rows, per_drug = [], []
    for r in results:
        inn = r["inn"]
        if r["basis"] == "fetch_failed":
            per_drug.append({k: r[k] for k in ("inn", "basis", "reason")})
            continue
        entry = {k: r[k] for k in ("inn", "basis", "products_seen") if k in r}
        if r["record"] is None:
            entry["reason"] = r["reason"]
        else:
            entry["emc_product_id"] = r["record"]["sedation"]["source"]["emc_product_id"]
            entry["population"] = r["record"]["sedation"]["source"]["population"]
            entry["labels"] = {L: r["record"][L]["label"] for L in V.LIABS}
            if inn in fda:
                entry["comparison"] = compare(fda[inn], r["record"])
                rows += [{"inn": inn, **c} for c in entry["comparison"]]
        per_drug.append(entry)
    compared = sorted({r["inn"] for r in rows})
    return {"n_train": len(train),
            "n_with_emc_smpc": sum(1 for d in per_drug if d.get("products_seen")),
            "n_smpc_placebo_subtracted": sum(1 for d in per_drug if d["basis"] == "placebo_subtracted"),
            "n_fetch_failed": sum(1 for d in per_drug if d["basis"] == "fetch_failed"),
            "compared_drugs": compared, "n_compared_drugs": len(compared),
            "n_compared_labels": len(rows), "n_agree": sum(r["agree"] for r in rows),
            "n_positive_fda": sum(r["fda_label"] for r in rows),
            "comparisons": rows,
            "drugs": [d for d in per_drug if d["basis"] != "absent"],
            "absent_reasons": {k: sum(1 for d in per_drug if d.get("reason") == k)
                               for k in sorted({d.get("reason") for d in per_drug if d["basis"] == "absent"})}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    ap.add_argument("--validate-train", metavar="OUT")
    a = ap.parse_args()
    if a.show:
        rules, _ = V.load_rules()
        print(json.dumps(smpc_record(a.show, rules, load_smpc_rules()[0]), indent=1, ensure_ascii=False))
    if a.validate_train:
        Path(a.validate_train).write_text(json.dumps(validate_train(), indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
