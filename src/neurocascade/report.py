"""P32.2: the one-page HTML report for one ``predict_compound`` result.

The report has one rule, and ``tests/test_report.py`` checks it on the parsed
HTML. Every number is a link: a predicted value or band links to its
provenance entry (predictor, translation method, calibration, input
references), and an input Ki links to its measurement references. Every row
that holds a prediction, a band, a dose or an effect value carries
``PROVENANCE_LABEL`` ("not validated: P34 returned no verdict") in the same
row, pending rows included.

The report renders only what ``predict_compound`` returns. Target-engagement
dose and panel occupancy-time have no scored predictor (P29.3 and P29.4 are
searched-and-empty), so ``predict_compound`` returns them as ``pending``, and
the report shows the reason, not a curve.
"""

from __future__ import annotations

import html
import math
import re
from html.parser import HTMLParser
from typing import Any, Mapping

import numpy as np

from .compound import PROVENANCE_LABEL

LEVELS = ("0.80", "0.95")
_LABEL_WORDS = {0: "absent", 1: "present"}
_REF_PATTERNS = (
    (re.compile(r"^PMID:?\s*(\d+)$"), "https://pubmed.ncbi.nlm.nih.gov/{}/"),
    (re.compile(r"^(CHEMBL\d+)$"), "https://www.ebi.ac.uk/chembl/document_report_card/{}/"),
)
_PMID_IN_TEXT = re.compile(r"PMID:?\s*(\d+)")

_CSS = """
@page { size: A4; margin: 12mm; }
body { font: 10px/1.35 system-ui, sans-serif; color: #222; max-width: 190mm; margin: auto; }
h1 { font-size: 15px; margin: 0 0 2px; } h2 { font-size: 11.5px; margin: 10px 0 3px; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #ddd; padding: 2px 4px; text-align: left; vertical-align: top; }
a.num { font-variant-numeric: tabular-nums; color: #0645ad; text-decoration: none; }
.nv { color: #a40000; font-weight: 600; white-space: nowrap; }
.banner { border: 2px solid #a40000; padding: 4px 6px; margin: 4px 0 8px; }
.reason { color: #555; } #provenance { font-size: 8.5px; color: #444; }
#provenance li { margin-bottom: 2px; }
"""


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _anchor(name: str) -> str:
    return "prov-" + re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _fmt(x: float, sig: int = 3) -> str:
    """``sig`` significant figures, positional (2230, not 2.23e+03)."""
    return np.format_float_positional(float(x), precision=sig, unique=False,
                                      fractional=False, trim="-")


def _num(text: str, target: str) -> str:
    return f'<a class="num" href="#{target}">{_esc(text)}</a>'


def _ident(text: str) -> str:
    return f'<span class="ident">{_esc(text)}</span>'


def _nv() -> str:
    return f'<td><span class="nv">{_esc(PROVENANCE_LABEL)}</span></td>'


def _ref_link(ref: str) -> str:
    for pattern, url in _REF_PATTERNS:
        m = pattern.match(ref.strip())
        if m:
            return f'<a class="ref" href="{_esc(url.format(m.group(1)))}">{_esc(ref)}</a>'
    return f'<a class="ref" href="#prov-unlinked-refs">{_esc(ref)}</a>'


def _linkify(text: str) -> str:
    """Escape free text and turn every 'PMID n' in it into a PubMed link."""
    out, last = [], 0
    for m in _PMID_IN_TEXT.finditer(text):
        out.append(_esc(text[last:m.start()]))
        url = f"https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/"
        out.append(f'<a class="ref" href="{_esc(url)}">{_esc(m.group(0))}</a>')
        last = m.end()
    out.append(_esc(text[last:]))
    return "".join(out)


def _fold_band(band: Mapping[str, Any], target: str) -> str:
    if not band.get("bounded") or not math.isfinite(band["upper"]):
        return _num("unbounded", target)
    return _num(f"{_fmt(band['lower'])} – {_fmt(band['upper'])}", target)


def _label_band(band: Mapping[str, Any], target: str) -> str:
    words = ", ".join(_LABEL_WORDS[int(v)] for v in band["labels"]) or "empty set"
    return _num("{" + words + "}", target)


def _pending_row(title: str, record: Mapping[str, Any], target: str, ncols: int) -> str:
    reason = f'<a class="reason" href="#{target}">{_esc(record.get("reason", ""))}</a>'
    return (f'<tr class="quantity pending"><td>{_esc(title)}</td>'
            f'<td colspan="{ncols}">{record["status"]}: {reason}</td>{_nv()}</tr>')


def _fold_row(title: str, record: Mapping[str, Any], target: str) -> str:
    if record.get("status") != "predicted":
        return _pending_row(title, record, target, 4)
    cells = [_num(_fmt(record["point"]), target), _esc(record["unit"])]
    cells += [_fold_band(record["bands"][lv], target) for lv in LEVELS]
    return (f'<tr class="quantity"><td>{_esc(title)}</td>'
            + "".join(f"<td>{c}</td>" for c in cells) + f"{_nv()}</tr>")


def _liability_row(name: str, record: Mapping[str, Any], target: str) -> str:
    if record.get("status") != "predicted":
        return _pending_row(name, record, target, 4)
    kind = "probability" if record.get("is_probability") else "not a probability"
    cells = [_num(_fmt(record["point"]), target), _esc(f"{record['point_kind']} ({kind})")]
    cells += [_label_band(record["bands"][lv], target) for lv in LEVELS]
    return (f'<tr class="quantity"><td>{_esc(name)}</td>'
            + "".join(f"<td>{c}</td>" for c in cells) + f"{_nv()}</tr>")


def _band_head(first: str) -> str:
    return (f"<tr><th>quantity</th><th>{first}</th><th>meaning / unit</th>"
            '<th><a href="#prov-bands">80% band</a></th>'
            '<th><a href="#prov-bands">95% band</a></th><th>status</th></tr>')


def _prov_item(target: str, title: str, provenance: Mapping[str, Any],
               extra: Mapping[str, Any] | None = None) -> str:
    parts = []
    for key, value in {**provenance, **(extra or {})}.items():
        if value is None or value == [] or key == "label":
            continue
        if isinstance(value, (list, tuple)):
            text = ", ".join(_ref_link(str(v)) for v in value)
        else:
            text = _linkify(str(value))
        parts.append(f"<b>{_esc(key)}</b>: {text}")
    parts.append(f'<span class="nv">{_esc(PROVENANCE_LABEL)}</span>')
    return f'<li id="{target}"><b>{_esc(title)}</b>. ' + "; ".join(parts) + "</li>"


def render_report(result: Mapping[str, Any], package: Mapping[str, Any]) -> str:
    """One self-contained HTML page for ``predict_compound(package)``."""
    prov: list[str] = []
    pk = result["human_pk"]

    dose_t = _anchor("target_engagement_dose")
    dose_rows = _fold_row("target-engagement dose", result["target_engagement_dose"], dose_t)
    prov.append(_prov_item(dose_t, "target-engagement dose",
                           result["target_engagement_dose"].get("provenance", {}),
                           {"reason": result["target_engagement_dose"].get("reason")}))

    pk_rows = []
    for key, title in (("clearance", "human CL"), ("vss", "human Vss"),
                       ("half_life", "half-life"), ("oral_exposure", "oral exposure")):
        rec, target = pk[key], _anchor(f"pk-{key}")
        pk_rows.append(_fold_row(title, rec, target))
        prov.append(_prov_item(target, title, rec.get("provenance", {}),
                               {"reason": rec.get("reason")}))

    occ = result["panel_occupancy_time"]
    occ_t = _anchor("panel_occupancy_time")
    occ_row = _fold_row("panel occupancy-time", occ, occ_t)
    prov.append(_prov_item(occ_t, "panel occupancy-time", occ.get("provenance", {}),
                           {"reason": occ.get("reason")}))

    liab_rows = []
    for name, rec in result["liabilities"].items():
        target = _anchor(f"liability-{name}")
        liab_rows.append(_liability_row(name, rec, target))
        prov.append(_prov_item(target, f"liability: {name}", rec.get("provenance", {}),
                               {"reason": rec.get("reason")}))

    near = result["nearest_benchmark_drugs"]
    near_t = _anchor("nearest")
    near_rows = [
        f'<tr><td>{_ident(d["inn"])}</td><td>'
        + ", ".join(_ident(g) for g in d["shared_measured_targets"])
        + f'</td><td><a href="#{near_t}">method</a></td>{_nv()}</tr>'
        for d in near.get("drugs", [])
    ] or [f'<tr><td colspan="3"><a href="#{near_t}">{_esc(near.get("reason", ""))}</a></td>{_nv()}</tr>']
    prov.append(_prov_item(near_t, "nearest benchmark drugs", near.get("provenance", {}),
                           {"reason": near.get("reason")}))

    ki_rows = []
    for gene, entry in sorted((package.get("binding") or {}).items()):
        if entry.get("ki_nM") is None:
            continue
        target = _anchor(f"binding-{gene}")
        ki_rows.append(f"<tr><td>{_ident(gene)}</td><td>{_num(_fmt(entry['ki_nM'], 4), target)}</td>"
                       f"<td>{_esc(', '.join(entry.get('efficacy') or []) or 'not given')}</td></tr>")
        prov.append(f'<li id="{target}"><b>Ki at {_ident(gene)}</b> (package input): '
                    + (", ".join(_ref_link(r) for r in entry.get("refs") or []) or "no reference given")
                    + "</li>")
    exposure = package.get("human_exposure")
    exposure_html = ""
    if exposure is not None:
        target = _anchor("human_exposure")
        exposure_html = ("<p>Projected human unbound Cmax (package input): "
                         f"{_num(_fmt(exposure['cmax_unbound_nM']), target)} nM</p>")
        prov.append(f'<li id="{target}"><b>human unbound Cmax</b> (package input): '
                    f"source {_linkify(exposure['source'])}</li>")

    prov.append(
        '<li id="prov-bands"><b>80% / 95% bands</b>: split-conformal bands frozen in P30.3 '
        "(human PK) and P30.4 (liabilities), Lei et al. 2018, J Am Stat Assoc; a fold band is "
        "lower – upper in the quantity's unit, a label set lists the effect labels the band "
        "cannot exclude. See each quantity's entry for its calibration set. "
        f'<span class="nv">{_esc(PROVENANCE_LABEL)}</span></li>')
    prov.append('<li id="prov-unlinked-refs"><b>Reference without a resolver</b>: '
                "the reference string is reported exactly as the package gave it.</li>")

    missing = ", ".join(_esc(m) for m in result.get("optional_inputs_missing", [])) or "none"
    unused = "".join(f"<li>{_esc(k)}: {_linkify(v)}</li>"
                     for k, v in result.get("inputs_not_used", {}).items())
    name = _ident(result["compound"])
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>neurocascade report: {_esc(result['compound'])}</title><style>{_CSS}</style></head>
<body>
<h1>neurocascade prediction: {name}</h1>
<div class="banner"><span class="nv">{_esc(PROVENANCE_LABEL)}</span>. Every value below comes
from a predictor and band frozen before this compound was seen. The go/no-go test closed
without a verdict, so none of them is validated for decision-making. Click any number for its
source or translation method.</div>
<section id="results">
<h2>Predicted dose</h2>
<table class="prediction">{_band_head("value")}{dose_rows}</table>
<h2>Human pharmacokinetics</h2>
<table class="prediction">{_band_head("point")}{''.join(pk_rows)}</table>
<h2>Panel occupancy over time</h2>
<table class="prediction">{_band_head("curve")}{occ_row}</table>
<h2>Effect and liability table</h2>
<table class="prediction">{_band_head("point")}{''.join(liab_rows)}</table>
<h2>Nearest benchmark drugs</h2>
<table class="prediction"><tr><th>drug</th><th>shared measured targets</th><th>how chosen</th>
<th>status</th></tr>{''.join(near_rows)}</table>
<h2>Package inputs used</h2>
<table class="inputs"><tr><th>target</th><th>Ki (nM)</th><th>efficacy</th></tr>{''.join(ki_rows)}</table>
{exposure_html}
<p>Optional inputs missing (each widens a band or leaves a quantity pending): {missing}</p>
</section>
<section id="provenance"><h2>Provenance</h2><ol>{''.join(prov)}</ol>
<p>Inputs accepted but not used:</p><ul>{unused}</ul></section>
</body></html>
"""


class _Auditor(HTMLParser):
    """Walks a rendered report and records every breach of the two rules."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict[str, str]]] = []
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.unlinked: list[str] = []
        self.in_results = False
        self.table_class: str | None = None
        self.row: dict[str, Any] | None = None
        self.rows_without_label: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and "href" in a:
            self.hrefs.append(a["href"])
        if tag == "section":
            self.in_results = a.get("id") == "results"
        if tag == "table":
            self.table_class = a.get("class")
        if tag == "tr":
            self.row = {"nums": 0, "label": False, "text": ""}
        if tag == "a" and "num" in a.get("class", "").split() and self.row is not None:
            self.row["nums"] += 1
        self.stack.append((tag, a))

    def handle_endtag(self, tag: str) -> None:
        while self.stack:
            open_tag, _ = self.stack.pop()
            if open_tag == tag:
                break
        if tag == "tr" and self.row is not None:
            if self.table_class == "prediction" and self.row["nums"] and not self.row["label"]:
                self.rows_without_label.append(self.row["text"].strip())
            self.row = None

    def handle_data(self, data: str) -> None:
        if self.row is not None:
            self.row["text"] += data
        inside = {(t, c) for t, attr in self.stack for c in attr.get("class", "").split() or [""]}
        if any(t == "span" and c == "nv" for t, c in inside) and data == PROVENANCE_LABEL:
            if self.row is not None:
                self.row["label"] = True
            return
        if not self.in_results or not any(ch.isdigit() for ch in data):
            return
        linked = any(t == "a" and attr.get("href") for t, attr in self.stack)
        ident = any(t == "span" and "ident" in attr.get("class", "").split()
                    for t, attr in self.stack)
        if not (linked or ident):
            self.unlinked.append(data.strip())


def audit_report(text: str) -> dict[str, list[str]]:
    """Breaches of the report's rules; every list is empty for a good report.

    * ``unlinked_numbers``: text in the results section holding a digit that
      is neither inside a link nor an identifier (gene symbol, drug name).
    * ``rows_without_label``: a row of a prediction table that holds a number
      but not ``PROVENANCE_LABEL``.
    * ``broken_links``: an in-page link whose target id does not exist.
    """
    auditor = _Auditor()
    auditor.feed(text)
    broken = [h for h in auditor.hrefs if h.startswith("#") and h[1:] not in auditor.ids]
    return {"unlinked_numbers": auditor.unlinked,
            "rows_without_label": auditor.rows_without_label,
            "broken_links": broken}


__all__ = ["LEVELS", "audit_report", "render_report"]
