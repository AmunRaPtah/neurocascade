"""Human Kp,uu records and explicit species-transfer handling (P14.3).

Since P29.1 a species transfer goes through the species gate's leaf primitive
(`neurocascade.translation`): the non-human Kp,uu must carry a `TranslationTag`
naming the registered
``kp-uu-transfer`` method, and the returned record carries the gate's
`TranslatedInput` so the prediction that consumes it can say so.
"""
import math
from dataclasses import dataclass

from ..translation import TranslationTag, check_translation


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
    KpUURecord("olanzapine", "human", None, "absent", "de Greef 2011 plasma EC50 (PMC3032087), fitted against plasma in living people", "not-required", "No Kp,uu is consumed; the in-vivo plasma EC50 already contains brain penetration."),
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


def transfer_kp_uu(*, human_kp_uu=None, source_species=None, source_kp_uu=None, transfer_factor=None, translation=None, label="Kp,uu"):
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
    if translation is not None and translation.method.key != "kp-uu-transfer":
        raise ValueError(
            "a Kp,uu transfer must be tagged with the kp-uu-transfer method; got %r"
            % translation.method.key
        )
    # No tag: the gate refuses the non-human number (UnsourcedParameterError).
    admitted = check_translation(
        source_species,
        layer="exposure layer",
        label=label,
        source="" if translation is None else translation.source,
        translation=translation,
    )
    return {
        "kp_uu": float(source_kp_uu) * float(transfer_factor),
        "provenance": "inferred-species-transfer",
        "source_species": admitted.species,
        "transfer_factor": float(transfer_factor),
        "translation": admitted,
    }


# ---------------------------------------------------------------------------
# P29.1: the Kp,uu translation scored against human numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanKpUUCSF:
    """A human unbound CSF:plasma ratio, the only human brain-exposure surrogate
    measured for these drugs. CSF is not brain interstitial fluid: Friden 2009's
    own rat data put Kp,uu,CSF within 3-fold of Kp,uu,brain for 33 of 39 drugs,
    and CSF over-predicts for strongly effluxed drugs."""

    drug: str
    kp_uu_csf: float
    citation: str


_FRIDEN_2009 = "Friden 2009, J Med Chem 52:6233-6243, PMID 19764786"

#: Every benchmark drug for which ChEMBL_37 carries a human Kp,uu,CSF in
#: Friden 2009 (document CHEMBL1795238, assay CHEMBL1798468, 32 compounds;
#: queried 2026-10-02). The other 28 compounds are not benchmark drugs.
HUMAN_KP_UU_CSF = (
    HumanKpUUCSF("amitriptyline", 0.18, _FRIDEN_2009 + "; ChEMBL activity 6262423"),
    HumanKpUUCSF("bupropion", 2.0, _FRIDEN_2009 + "; ChEMBL activity 6262426"),
    HumanKpUUCSF("diazepam", 0.79, _FRIDEN_2009 + "; ChEMBL activity 6262430"),
    HumanKpUUCSF("thioridazine", 1.4, _FRIDEN_2009 + "; ChEMBL activity 6263053"),
)

#: The identity transfer is the method, not a fitted coefficient: Friden 2009
#: carries rat Kp,uu to human unchanged. Nothing is fitted, so no split is used.
IDENTITY_TRANSFER_FACTOR = 1.0


def score_rat_kp_uu_transfer():
    """Translate each drug's rat Kp,uu,brain (P28.3 ADME record) to human by the
    tagged identity transfer and compare with the human Kp,uu,CSF.

    Returns ``(rows, gmfe)``; each row carries the gate's `TranslatedInput`.
    Fold error is predicted / observed, so above 1 is an over-prediction.
    """
    from ..benchmark import load_adme_record

    rows = []
    for observed in HUMAN_KP_UU_CSF:
        record = load_adme_record(observed.drug)
        rat = [
            m for m in record["fields"]["kp_uu"]["measurements"]
            if m["species"] == "rat" and m["relation"] == "="
        ]
        if len(rat) != 1:
            raise ValueError(
                "%s: expected one rat Kp,uu,brain in the P28.3 record, got %d"
                % (observed.drug, len(rat))
            )
        measurement = rat[0]
        tag = TranslationTag(
            "kp-uu-transfer",
            "rat",
            "ChEMBL %s assay %s (%s)" % (
                measurement["ref"], measurement["assay_chembl_id"], _FRIDEN_2009
            ),
            detail="identity transfer, rat Kp,uu,brain to human",
        )
        result = transfer_kp_uu(
            source_species="rat",
            source_kp_uu=measurement["value"],
            transfer_factor=IDENTITY_TRANSFER_FACTOR,
            translation=tag,
            label="%s rat Kp,uu,brain" % observed.drug,
        )
        rows.append({
            "drug": observed.drug,
            "rat_kp_uu_brain": float(measurement["value"]),
            "predicted_human_kp_uu": result["kp_uu"],
            "observed_human_kp_uu_csf": observed.kp_uu_csf,
            "fold_error": result["kp_uu"] / observed.kp_uu_csf,
            "translation": result["translation"],
        })
    gmfe = math.exp(
        sum(abs(math.log(row["fold_error"])) for row in rows) / len(rows)
    )
    return rows, gmfe
