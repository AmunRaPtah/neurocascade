"""P23.4: the write-up reports the P23.3 result, and agrees with it and V12.

P23.4 is a reporting task. Its deliverable is the plain-language write-up in
``docs/p23-4-fingerprint-result.md`` plus the already-committed ledger row V12.
These tests do not re-derive the science (``test_fingerprint_eps_regression.py``
pins the regression); they guard the two acceptance criteria that a write-up can
drift on:

* the headline numbers in the write-up are the ones the module computes live, so
  the prose cannot silently outrun the code;
* the write-up and ledger row V12 state the same verdict and the same numbers, so
  "the validation ledger row matches the write-up" stays true if either is edited.

They also assert the no-overclaim criterion in terms: the write-up must deny that
the model predicts behaviour, and must say what n would settle the question.
"""

from pathlib import Path

import numpy as np
import pytest

from neurocascade.human import fingerprint_eps_regression as m

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "p23-4-fingerprint-result.md"
LEDGER = ROOT / "docs" / "validation-ledger.md"


@pytest.fixture(scope="module")
def report():
    assert REPORT.is_file(), "the P23.4 write-up is missing"
    return REPORT.read_text()


@pytest.fixture(scope="module")
def v12(report):
    """The V12 row of the validation ledger (the row the write-up must match)."""
    for line in LEDGER.read_text().splitlines():
        if line.startswith("| V12 "):
            return line
    raise AssertionError("ledger row V12 not found")


@pytest.fixture(scope="module")
def fit():
    return m.fit_models()


# --- the write-up reports the live result, not a frozen copy ---------------

def test_verdict_is_indeterminate_everywhere(report, fit, v12):
    """The one-sentence result is INDETERMINATE, in the module, prose and row."""
    assert fit.verdict == "indeterminate"
    assert "indeterminate" in report.lower()
    assert "Indeterminate" in v12


def test_headline_numbers_match_the_module(report, fit):
    """Every headline figure in the prose is the module's live computation."""
    assert round(fit.r2_null, 3) == 0.314 and "0.314" in report
    assert round(fit.r2_full, 3) == 0.636 and "0.636" in report
    assert round(fit.beta_ratio, 3) == -0.396 and "−0.396" in report
    assert round(fit.f_stat, 2) == 3.54 and "3.54" in report
    assert round(fit.p_value, 3) == 0.133 and "0.133" in report
    assert round(fit.increment_f2, 2) == 0.88 and "0.88" in report
    assert round(fit.power(0.35), 2) == 0.20 and "0.20" in report
    # the sign that carries the hypothesis
    assert fit.beta_ratio < 0


def test_power_and_n_claims_match_the_module(report, fit):
    """The 'what number could settle it' figures are the module's, not invented."""
    assert fit.n_for_power(fit.increment_f2) == 13 and "13" in report
    assert fit.n_for_power(0.35) == 26 and "26" in report
    # the design has at most eight drugs with a Lako formula; 13 and 26 exceed it
    assert len(m.LAKO_2013) == 8


def test_n_is_seven_in_prose_and_module(report, fit):
    assert fit.n == 7
    assert "n = 7" in report


# --- the write-up matches the ledger row V12 -------------------------------

def test_v12_carries_the_same_numbers_as_the_report(v12):
    """V12 and the write-up pin the same statistics (acceptance: they match)."""
    for token in ("0.314", "0.636", "−0.396", "3.54", "p=0.133", "0.88", "0.20", "13"):
        assert token in v12, "ledger V12 lost %r" % token


def test_siafis_anchor_is_the_measured_source(report, v12):
    """The outcome nobody here chose: Siafis 2023, N=37,193, its PMID."""
    assert "37,193" in report
    assert "Siafis" in report and "Siafis" in v12
    assert "37537284" in v12  # Siafis PMID, named in the ledger source column


# --- the no-overclaim acceptance criterion, in terms ----------------------

def test_report_denies_behavioural_prediction(report):
    """Acceptance: no claim the model predicts behaviour beyond what was tested."""
    low = " ".join(report.lower().replace("*", "").split())  # strip emphasis + wrap
    # the specific disclaimer sentence must be present
    assert "not claimed to predict behaviour" in low
    assert "neither confirmed nor" in low  # not a validated claim, not a refutation


def test_report_says_what_number_would_settle_it(report):
    """Acceptance: an underpowered result must say what n could settle it."""
    assert "What number could settle it" in report
    assert "80% power" in report
