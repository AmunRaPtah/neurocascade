"""Human Kp,uu records and explicit species-transfer handling (P14.3)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class KpUURecord:
    drug: str
    species: str
    kp_uu: float | None
    measurement_kind: str
    citation: str
    status: str
    notes: str

    def __post_init__(self):
        if self.measurement_kind not in ("Kp,uu", "Kp", "absent"):
            raise ValueError("measurement_kind must be Kp,uu, Kp, or absent")
        if self.kp_uu is not None and self.kp_uu <= 0:
            raise ValueError("kp_uu must be positive when present")
        if self.measurement_kind == "Kp,uu" and self.kp_uu is None:
            raise ValueError("Kp,uu records require a value")


HUMAN_KP_UU_RECORDS = (
    KpUURecord("fentanyl", "human", None, "absent", "Searched human Kp,uu source set; no defensible direct value", "least-supported", "Illustrative transfer may use f_species near 1 only when explicitly requested."),
    KpUURecord("haloperidol", "human", None, "absent", "Human occupancy inversion reserved for P16.1", "not-required", "No Kp,uu is consumed; occupancy data should identify the effective brain exposure."),
    KpUURecord("diazepam", "human", 0.79, "Kp,uu", "Friden 2009, PMID 19764786; DiVA thesis diva2:349401", "measured-human", "CSF surrogate bias is expected to be small for this non-efflux substrate."),
)


@dataclass(frozen=True)
class SpeciesTransferPrior:
    source_species: str
    f_species: float
    prior: str
    citation: str


TRANSFER_PRIORS = {
    "non-efflux": SpeciesTransferPrior("rat", 1.0, "approximately 1", "Sato 2021, AAPS J 23(4):81, PMC8175309"),
    "p-gp-substrate": SpeciesTransferPrior("rat", 2.0, "approximately 2-3", "Sato 2021; MDR1 relative activity 0.395 rat vs 0.096 human"),
}


def kp_uu_records(drug=None, species="human"):
    records = tuple(r for r in HUMAN_KP_UU_RECORDS if r.species == species)
    return records if drug is None else tuple(r for r in records if r.drug == drug)


def transfer_kp_uu(*, human_kp_uu=None, source_species=None, source_kp_uu=None, transfer_factor=None):
    if human_kp_uu is not None:
        if human_kp_uu <= 0:
            raise ValueError("human_kp_uu must be positive")
        return {"kp_uu": float(human_kp_uu), "provenance": "measured-human", "source_species": "human"}
    if source_kp_uu is None or transfer_factor is None:
        raise ValueError("human Kp,uu is unavailable; provide an explicit transfer_factor")
    if source_species in (None, "human"):
        raise ValueError("species transfer requires a non-human source species")
    if source_kp_uu <= 0 or transfer_factor <= 0:
        raise ValueError("source_kp_uu and transfer_factor must be positive")
    return {"kp_uu": float(source_kp_uu) * float(transfer_factor), "provenance": "inferred-species-transfer", "source_species": str(source_species)}
