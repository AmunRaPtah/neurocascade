"""Validation contracts for human PK evidence (P14.4).

This module records what can and cannot currently be validated; it does not
turn missing literature into fitted numbers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationOutcome:
    drug: str
    status: str
    metrics: tuple
    citation: str
    limitations: str


HUMAN_PK_VALIDATION = (
    ValidationOutcome("fentanyl", "blocked-missing-validated-parameter-set", (), "Shafer 1990; Scott & Stanski 1987; McClain & Hug 1980; Bae 2020", "Three-compartment human sources exist, but no common transcribed parameter set is validated here."),
    ValidationOutcome("haloperidol", "blocked-missing-readable-human-two-compartment-set", (), "Pilla Reddy 2013, PMID 24113674", "The most valuable population PK source is not available for verified transcription; no depot population model is asserted."),
    ValidationOutcome("diazepam", "blocked-no-adult-population-model", (), "Ku 2018, PMID 30267478", "Available population model is pediatric/methodological; adult evidence is intensive-sampling NCA and is not silently converted."),
)

KNOWN_FAILURE_MODES = {
    "haloperidol_rat_hybrid_pbpk_pd": {
        "citation": "Johnson 2016, PMID 26718955",
        "outcome": "under-predicted human D2 occupancy",
        "cause": "unmodelled brain reduced-haloperidol back-oxidation and active BBB influx",
    }
}


def validation_outcomes(drug=None):
    if drug is None:
        return HUMAN_PK_VALIDATION
    return tuple(item for item in HUMAN_PK_VALIDATION if item.drug == drug)


def known_failure_mode(name):
    try:
        return KNOWN_FAILURE_MODES[name].copy()
    except KeyError:
        raise KeyError("unknown human PK failure mode: %s" % name)
