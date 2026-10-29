"""P32.2: the one-page report links every number and labels every prediction."""

from __future__ import annotations

import re

import pytest

from neurocascade.compound import PROVENANCE_LABEL, package_from_benchmark, predict_compound
from neurocascade.report import audit_report, render_report

CLEAN = {"unlinked_numbers": [], "rows_without_label": [], "broken_links": []}
# 1 dose + 4 PK + 1 occupancy + 4 liabilities (P32.1's predict_compound records).
N_QUANTITY_ROWS = 10


def _report(inn, exposure=None):
    package = package_from_benchmark(inn)
    if exposure is not None:
        package["human_exposure"] = exposure
    return render_report(predict_compound(package), package), package


@pytest.fixture(scope="module")
def venlafaxine():
    return _report("venlafaxine")[0]


@pytest.mark.parametrize("inn", ["venlafaxine", "haloperidol", "aripiprazole", "clozapine"])
def test_report_passes_its_audit(inn):
    assert audit_report(_report(inn)[0]) == CLEAN


def test_every_quantity_row_carries_the_label_beside_it(venlafaxine):
    """Independent of audit_report: a regex over the raw HTML rows."""
    rows = re.findall(r'<tr class="quantity[^"]*">(.*?)</tr>', venlafaxine, flags=re.S)
    assert len(rows) == N_QUANTITY_ROWS
    label = f'<span class="nv">{PROVENANCE_LABEL}</span>'
    assert all(label in row for row in rows)
    # Every number in a prediction table (point, band or label set) is in such a row.
    for table in re.findall(r'<table class="prediction">(.*?)</table>', venlafaxine, flags=re.S):
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S):
            if 'class="num"' in row:
                assert label in row


def test_every_number_link_resolves_to_a_provenance_entry(venlafaxine):
    targets = re.findall(r'<a class="num" href="#([^"]+)"', venlafaxine)
    assert targets
    for t in set(targets):
        assert f'id="{t}"' in venlafaxine
    # CL point, two CL bands, Vss point + 2 bands, 2 liabilities x (point + 2 sets), 6 Ki.
    assert len(targets) == 3 + 3 + 6 + 6


def test_values_shown_are_the_predicted_ones(venlafaxine):
    result = predict_compound(package_from_benchmark("venlafaxine"))
    cl = result["human_pk"]["clearance"]
    assert f'href="#prov-pk-clearance">{float(f"{cl["point"]:.3g}"):g}</a>' in venlafaxine
    assert '<a class="num" href="#prov-pk-clearance">unbounded</a>' in venlafaxine
    # The SLC6A4 Ki input (39 nM) links to its references, which link to PubMed.
    assert '<a class="num" href="#prov-binding-SLC6A4">39</a>' in venlafaxine
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/9537821/"' in venlafaxine
    # The IVIVE method's citations and the CL input documents are links.
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/4764426/"' in venlafaxine
    assert 'href="https://www.ebi.ac.uk/chembl/document_report_card/CHEMBL1139373/"' in venlafaxine


def test_pending_quantities_show_the_reason_and_no_number(venlafaxine):
    for anchor in ("prov-target_engagement_dose", "prov-panel_occupancy_time",
                   "prov-pk-half_life", "prov-pk-oral_exposure"):
        row = re.search(rf'<tr class="quantity pending">((?:(?!</tr>).)*href="#{anchor}"'
                        r"(?:(?!</tr>).)*)</tr>", venlafaxine, flags=re.S)
        assert row is not None, anchor
        assert 'class="num"' not in row.group(1)
        assert "pending:" in row.group(1)
    assert "P29.3 is searched-and-empty" in venlafaxine


def test_candidate_path_is_marked_not_a_probability():
    report, _ = _report("haloperidol", {"cmax_unbound_nM": 0.5, "source": "PMID:12345678"})
    assert audit_report(report) == CLEAN
    assert "engagement percentile among the frozen train cohort (not a probability)" in report
    assert '<a class="num" href="#prov-human_exposure">0.5</a>' in report


@pytest.mark.parametrize("breach, field", [
    ('<section id="results"><table class="prediction"><tr><td>12.5</td></tr></table>',
     "unlinked_numbers"),
    ('<section id="results"><table class="prediction"><tr><td>'
     '<a class="num" href="#prov-bands">3</a></td></tr></table>', "rows_without_label"),
    ('<section id="results"><a href="#prov-missing">x</a>', "broken_links"),
])
def test_audit_catches_each_breach(venlafaxine, breach, field):
    doctored = venlafaxine.replace('<section id="results">', breach, 1)
    assert audit_report(doctored)[field]
