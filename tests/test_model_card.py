"""P24.2: the model card is traceable, ordered, verbatim and live.

The model card (``docs/model-card.md``) is prose, and prose drifts. These tests
guard the three acceptance criteria in terms a reviewer would otherwise have to
check by eye:

* every claim carries a citation, and every citation resolves to a row of
  ``docs/validation-ledger.md`` (``[V4]`` to a validated row, ``[ledger: X]`` to
  an untested or searched-and-empty row whose first cell starts with ``X``);
* the out-of-scope uses and mismatches come before the intended use and the
  capabilities, and every unrepaired mismatch is named;
* every block quote is verbatim from the P19.7 scope document, and the headline
  numbers are the ones the code computes now, not a frozen copy.
"""

import re
from pathlib import Path

import pytest

from neurocascade.human.d2_ec50 import d2_plasma_ec50
from neurocascade.human.label_dose_prediction import predict_window
from neurocascade.human.prediction_interface import (
    predict,
    refusal_layers,
    supported_drugs,
)

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "docs" / "model-card.md"
LEDGER = ROOT / "docs" / "validation-ledger.md"
P19_7 = ROOT / "docs" / "p19-7-scope-and-failure-analysis.md"

CITATION = re.compile(r"\[(V\d+)\]|\[ledger: ([^\]]+)\]")

SECTIONS_IN_ORDER = (
    "## 1. Do not use this model for",
    "## 2. Unrepaired mismatches",
    "## 3. The populations the observations came from",
    "## 4. Intended use",
    "## 5. What the model can do, and how well",
)


@pytest.fixture(scope="module")
def card():
    assert CARD.is_file(), "the P24.2 model card is missing"
    return CARD.read_text()


@pytest.fixture(scope="module")
def body(card):
    """Everything after the preamble: the part made of claims."""
    return card[card.index("\n## ") :]


@pytest.fixture(scope="module")
def ledger_rows():
    """First cells of every ledger table row, keyed by validated id or text."""
    validated, other = set(), []
    for line in LEDGER.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("|---"):
            continue
        first = line.split(" | ")[0][2:].replace("**", "").strip()
        if re.fullmatch(r"V\d+", first):
            validated.add(first)
        else:
            other.append(first)
    return validated, other


def _claim_units(body):
    """Paragraphs, bullets and table rows; block quotes are checked apart."""
    units = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines or lines[0].startswith("#"):
            continue
        if all(line.lstrip().startswith(">") for line in lines):
            continue
        if lines[0].startswith("|"):
            units.extend(
                line
                for line in lines
                if not line.startswith("|---") and not line.startswith("| Mismatch")
                and not line.startswith("| Claim") and not line.startswith("| Drug")
            )
            continue
        current = []
        for line in lines:
            if line.startswith("- ") and current:
                units.append(" ".join(current))
                current = []
            current.append(line.strip())
        units.append(" ".join(current))
    return units


def _quotes(body):
    quotes = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line.strip() for line in block.strip().splitlines()]
        if lines and all(line.startswith(">") for line in lines):
            quotes.append(" ".join(line.lstrip("> ").strip() for line in lines))
    return quotes


def _squash(text):
    return re.sub(r"\s+", " ", text).strip()


# --- traceability ----------------------------------------------------------

def test_every_claim_cites_the_ledger(body):
    units = _claim_units(body)
    assert len(units) > 40
    uncited = [unit[:80] for unit in units if not CITATION.search(unit)]
    assert uncited == []


def test_every_citation_resolves_to_a_ledger_row(card, ledger_rows):
    validated, other = ledger_rows
    citations = CITATION.findall(card)
    assert citations
    for row_id, prefix in citations:
        if row_id:
            assert row_id in validated, row_id
        else:
            matches = [first for first in other if first.startswith(prefix)]
            assert len(matches) == 1, (prefix, matches)


def test_every_validated_row_on_the_d2_chain_is_reported(body):
    """Failures included: the card cannot quietly drop a negative row."""
    capabilities = body[body.index(SECTIONS_IN_ORDER[4]) :]
    for row in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11"):
        assert "[%s]" % row in capabilities, row
    for status in ("fails", "mostly fails", "unresolved disagreement", "falsified"):
        assert status in capabilities


# --- ordering and coverage -------------------------------------------------

def test_limitations_come_before_capabilities(card):
    positions = [card.index(heading) for heading in SECTIONS_IN_ORDER]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    "mismatch",
    [
        "depot intramuscular steady state",
        "dose-to-plasma arm has never been tested",
        "No held-out occupancy series for olanzapine",
        "total versus free plasma",
        "disagrees with the published constants",
        "not printed on the fitting observations",
        "not independent",
        "Thresholds measured in one drug",
        "Single-dose readings against steady-state thresholds",
        "little discriminating power",
        "Population exchangeability",
        "Typical values only",
        "Absorption lag dropped",
        "Active metabolite ignored",
        "No effect compartment",
        "No chronic pharmacology",
    ],
)
def test_every_unrepaired_mismatch_is_named(body, mismatch):
    table = body[body.index(SECTIONS_IN_ORDER[1]) : body.index(SECTIONS_IN_ORDER[2])]
    assert mismatch in table


@pytest.mark.parametrize(
    "scope",
    [
        "Any region outside the striatum",
        "Chronic dosing",
        "no effect compartment",
        "Predicting behaviour",
        "Any route but oral",
        "Any drug other than haloperidol and olanzapine",
        "Any body weight other than the 70 kg reference",
    ],
)
def test_out_of_scope_uses_are_in_the_first_section(body, scope):
    first = body[: body.index(SECTIONS_IN_ORDER[1])]
    assert scope in first


# --- P19.7 carried verbatim -------------------------------------------------

def test_quotes_are_verbatim_from_p19_7(body):
    source = _squash(P19_7.read_text())
    quotes = _quotes(body)
    assert len(quotes) >= 13
    for quote in quotes:
        assert _squash(quote) in source, quote[:80]


@pytest.mark.parametrize(
    "fragment",
    [
        "Extrastriatal D2 is out of scope",
        "the chain cannot simulate the route its own EC50 was measured on",
        "The chain has no time-varying pharmacology.",
        "Kurose 2020 (PMID 32726002)",
        "**PK set.** Pilla Reddy 2013",
        "**Occupancy set.**",
        "Whether the sets are exchangeable is **not determined in the repo**",
    ],
)
def test_p19_7_scope_limits_are_quoted(body, fragment):
    assert any(fragment in quote for quote in _quotes(body))


# --- numbers are live -------------------------------------------------------

def test_supported_drugs_and_refusals_match_the_interface(body):
    assert supported_drugs() == ("haloperidol", "olanzapine")
    table = body[body.index("## 7.") :]
    for line in table.splitlines():
        if not line.startswith("| ") or line.startswith("| Drug"):
            continue
        drug, layers = [cell.strip() for cell in line.strip("|").split("|")[:2]]
        assert tuple(layer for layer, _ in refusal_layers(drug)) == tuple(
            layers.split(", ")
        ), drug
    for drug in ("risperidone", "aripiprazole", "paliperidone", "quetiapine",
                 "ziprasidone", "clozapine", "amisulpride", "fentanyl", "diazepam"):
        assert "| %s |" % drug in table


def test_v4_numbers_are_the_interface_output(body):
    single = predict("haloperidol", 2.0, "single-dose")
    steady = predict("haloperidol", 2.0, "steady-state-daily")
    assert "%.4f" % single.peak_occupancy == "0.5253" and "0.5253" in body
    assert "%.4f" % steady.mean_occupancy == "0.5635" and "0.5635" in body


def test_v8_window_is_the_chain_output(body):
    haloperidol = predict_window("haloperidol")
    olanzapine = predict_window("olanzapine")
    window = "haloperidol %.2f–%.2f mg/day" % (
        haloperidol.d65_mg_per_day, haloperidol.d78_mg_per_day)
    assert window == "haloperidol 2.89–5.56 mg/day" and window in body
    window = "olanzapine %.2f–%.1f mg/day" % (
        olanzapine.d65_mg_per_day, olanzapine.d78_mg_per_day)
    assert window == "olanzapine 5.77–16.4 mg/day" and window in body


def test_constants_are_the_registry_values(body):
    assert "%.4f" % d2_plasma_ec50("haloperidol").ec50_nM == "1.8849"
    assert "EC50 1.8849 nM" in body
    olanzapine = d2_plasma_ec50("olanzapine")
    assert olanzapine.emax == 0.875 and "fitted Emax 0.875" in body
    assert "5.29 ng/mL" in olanzapine.citation and "5.29 ng/mL" in body


def test_provenance_flag_caveat_is_still_true(body):
    """The card warns that 'measured' is a gate flag. Keep the warning honest."""
    prediction = predict("haloperidol", 2.0, "steady-state-daily")
    occupancy = [layer for layer in prediction.layers if layer.name == "occupancy"]
    assert prediction.provenance == "measured"
    assert occupancy[0].provenance == "fitted-in-repo"
    assert "`fitted-in-repo`" in body and '`"measured"`' in body


def test_ledger_rows_agree_with_the_card(body):
    ledger = LEDGER.read_text()
    for figure in ("0.0351", "0.0620", "+0.0601", "0.0939", "0.5253", "0.5635",
                   "2.89–5.56", "5.77–16.4", "0.0244", "−0.424", "1764", "1.696",
                   "0.368", "0.568", "0.0030", "0.782 h"):
        assert figure in body, figure
        assert figure in ledger, figure
