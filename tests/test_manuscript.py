"""P24.3: the manuscript is traceable, honest about negatives, and in scope.

The manuscript (``docs/manuscript.md``) is prose, and prose drifts away from the
evidence it was drawn from. These tests guard the three acceptance criteria a
reviewer would otherwise have to check by eye:

* every quantitative claim carries a citation, and every citation resolves to a
  row of ``docs/validation-ledger.md`` (``[V4]`` to a validated row,
  ``[ledger: X]`` to an untested or searched-and-empty row whose first cell
  begins with ``X``);
* the negative and the indeterminate results are reported, not quietly dropped;
* no frozen layer is claimed as a result -- behaviour, whole-brain/circuit
  simulation and the scaffold chains are named as out of scope;

and, so the prose cannot outrun the code, the headline numbers are the ones the
chain computes now.
"""

import re
from pathlib import Path

from neurocascade.human.d2_ec50 import d2_plasma_ec50
from neurocascade.human.label_dose_prediction import predict_window
from neurocascade.human.prediction_interface import predict, supported_drugs

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "docs" / "manuscript.md"
LEDGER = ROOT / "docs" / "validation-ledger.md"

CITATION = re.compile(r"\[(V\d+)\]|\[ledger: ([^\]]+)\]")
DECIMAL = re.compile(r"\d+\.\d")


@pytest.fixture(scope="module")
def manuscript():
    assert MANUSCRIPT.is_file(), "the P24.3 manuscript is missing"
    return MANUSCRIPT.read_text()


@pytest.fixture(scope="module")
def body(manuscript):
    """Everything from section 1 up to the References: the part made of claims.

    The preamble (which explains the citation syntax) and the reference list
    (which is full of DOIs and volume numbers) are excluded on purpose.
    """
    start = manuscript.index("## 1. Introduction")
    end = manuscript.index("## 7. References")
    return manuscript[start:end]


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
    """Paragraphs, bullets and table rows; headings and block quotes dropped."""
    units = []
    for block in re.split(r"\n\s*\n", body):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines or lines[0].startswith("#"):
            continue
        if all(line.lstrip().startswith(">") for line in lines):
            continue
        if lines[0].startswith("|"):
            units.extend(
                line for line in lines if not line.startswith("|---")
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


# --- traceability ----------------------------------------------------------

def test_every_quantitative_claim_cites_the_ledger(body):
    """Any claim unit carrying a decimal figure must cite a ledger row."""
    units = _claim_units(body)
    assert len(units) > 20
    uncited = [
        unit[:90]
        for unit in units
        if DECIMAL.search(unit) and not CITATION.search(unit)
    ]
    assert uncited == []


def test_every_citation_resolves_to_a_single_ledger_row(body, ledger_rows):
    validated, other = ledger_rows
    citations = CITATION.findall(body)
    assert citations
    for row_id, prefix in citations:
        if row_id:
            assert row_id in validated, row_id
        else:
            matches = [first for first in other if first.startswith(prefix)]
            assert len(matches) == 1, (prefix, matches)


def test_every_validated_d2_row_is_reported(body):
    """V1-V12 all appear: the manuscript cannot quietly drop a negative row."""
    for row in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                "V11", "V12"):
        assert "[%s]" % row in body, row


# --- negatives and indeterminates ------------------------------------------

def test_failures_are_named_as_failures(manuscript):
    results = manuscript[manuscript.index("## 3. Results"):]
    for word in ("over-predicts", "disagree", "fail", "miss", "falsif",
                 "anecdotal"):
        assert word in results, word


def test_the_fingerprint_result_is_reported_indeterminate(body):
    assert "indeterminate" in body.lower()
    assert "p = 0.133" in body and "0.314 to 0.636" in body
    assert "[V12]" in body


def test_clozapine_falsification_is_stated(body):
    assert "falsif" in body.lower()
    assert "2.94" in body and "1764" in body and "[V11]" in body


# --- no frozen-layer claim -------------------------------------------------

@pytest.mark.parametrize(
    "disclaimer",
    [
        "no behavioural or clinical prediction",
        "no whole-brain, circuit or network simulation",
        "every circuit-layer parameter is illustrative",
        "scaffolding with no sourced parameters",
    ],
)
def test_frozen_layers_are_named_as_out_of_scope(body, disclaimer):
    section = body[: body.index("## 2.")]
    assert disclaimer in section, disclaimer


def test_the_frozen_layers_carry_ledger_citations(body):
    section = body[: body.index("## 2.")]
    for cue, cite in (
        ("no behavioural", "[ledger: Any behavioural endpoint]"),
        ("circuit-layer parameter is illustrative", "[ledger: Any circuit-layer parameter]"),
        ("scaffolding", "[ledger: Mu-opioid and GABA-A chains]"),
    ):
        assert cue in section and cite in section, cue


# --- numbers are live ------------------------------------------------------

def test_supported_drugs_match_the_interface(body):
    assert supported_drugs() == ("haloperidol", "olanzapine")
    assert "haloperidol and olanzapine" in body


def test_v4_numbers_are_the_interface_output(body):
    single = predict("haloperidol", 2.0, "single-dose")
    steady = predict("haloperidol", 2.0, "steady-state-daily")
    assert "%.4f" % single.peak_occupancy == "0.5253" and "0.5253" in body
    assert "%.4f" % steady.mean_occupancy == "0.5635" and "0.5635" in body


def test_v8_windows_are_the_chain_output(body):
    haloperidol = predict_window("haloperidol")
    olanzapine = predict_window("olanzapine")
    hw = "haloperidol %.2f–%.2f mg/day" % (
        haloperidol.d65_mg_per_day, haloperidol.d78_mg_per_day)
    ow = "olanzapine %.2f–%.1f mg/day" % (
        olanzapine.d65_mg_per_day, olanzapine.d78_mg_per_day)
    assert hw == "haloperidol 2.89–5.56 mg/day" and hw in body
    assert ow == "olanzapine 5.77–16.4 mg/day" and ow in body


def test_constants_are_the_registry_values(body):
    assert "%.4f" % d2_plasma_ec50("haloperidol").ec50_nM == "1.8849"
    assert "EC50 1.8849 nM" in body or "1.8849 nM" in body
    olanzapine = d2_plasma_ec50("olanzapine")
    assert olanzapine.emax == 0.875 and "0.875" in body
    assert "5.29 ng/mL" in olanzapine.citation and "5.29 ng/mL" in body


def test_numbers_agree_with_the_ledger(body):
    ledger = LEDGER.read_text()
    for figure in ("0.0351", "0.0620", "+0.0601", "0.0939", "0.5253", "0.5635",
                   "2.89–5.56", "5.77–16.4", "0.0244", "−0.424",
                   "1764", "1.696", "2.94"):
        assert figure in body, figure
        assert figure in ledger, figure
