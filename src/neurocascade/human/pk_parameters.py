"""Human PK parameter records for P14.2.

These are metadata records, not silently executable presets. Missing parameters
remain ``None`` so P14.4 can distinguish a sourced model from a gap.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HumanPKRecord:
    drug: str
    route: str
    species: str
    V1_L_per_kg: float | None
    V2_L_per_kg: float | None
    V3_L_per_kg: float | None
    Q12_mL_per_min_per_kg: float | None
    Q13_mL_per_min_per_kg: float | None
    CL_mL_per_min_per_kg: float | None
    F: float | None
    ka_per_min: float | None
    citation: str
    status: str
    notes: str

    @property
    def complete_for_model(self):
        return all(value is not None for value in (
            self.V1_L_per_kg, self.V2_L_per_kg, self.V3_L_per_kg,
            self.Q12_mL_per_min_per_kg, self.Q13_mL_per_min_per_kg,
            self.CL_mL_per_min_per_kg, self.F,
        )) and (self.route == "IV" or self.ka_per_min is not None)


HUMAN_PK_RECORDS = (
    HumanPKRecord(
        "fentanyl", "IV", "human", None, None, None, None, None, None, 1.0, None,
        "Shafer 1990, PMID 2248388; Scott & Stanski 1987, PMID 3100765; McClain & Hug 1980, PMID 7389247; Bae 2020, PMID 32861508",
        "parameter-set-required", "Canonical human studies establish three-compartment disposition, but this repository has not transcribed a single validated common parameter set yet.",
    ),
    HumanPKRecord(
        "haloperidol", "IV", "human", None, None, None, None, None, None, 1.0, None,
        "Human haloperidol PK source set pending verified transcription",
        "searched-and-empty", "No parameter values are substituted from rodent records.",
    ),
    HumanPKRecord(
        "diazepam", "IV", "human", None, None, None, None, None, None, 1.0, None,
        "Human diazepam PK source set pending verified transcription",
        "searched-and-empty", "No adult population parameter set is asserted here; pediatric/methodological studies are not substituted.",
    ),
)


def human_pk_records(drug=None):
    records = HUMAN_PK_RECORDS if drug is None else tuple(r for r in HUMAN_PK_RECORDS if r.drug == drug)
    return records
