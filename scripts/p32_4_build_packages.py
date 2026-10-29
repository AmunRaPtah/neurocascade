"""P32.4: build the prospective blind-test preclinical packages.

Each package is the ``neurocascade.compound_package/1`` schema that
``predict_compound`` consumes, built from PRECLINICAL sources only (in-vitro
binding, in-vitro ADME with human material, animal in-vivo PK).  No human
in-vivo quantity is read here; the observed human truth lives in a separate
file added in a later commit so that the prediction is demonstrably blind
(acceptance: prediction committed before any human-truth record exists).

Every number below carries its citation in the ``refs`` / ``description`` /
``source`` field, as the project's no-invented-number rule requires.  Where a
value is a ChEMBL curation of a paywalled table rather than a verbatim read,
the description says so (``chembl_curated``); where a binding Ki was read from
indexed excerpts because the publisher PDF was inaccessible, the ref says so.

Gene symbols are HGNC.  Receptors outside the v2 panel (orexin HCRTR1/2,
opioid OPRK1/OPRM1/OPRD1, GABA-A GABRA*, tachykinin TACR3) are kept in the
package because they are the compound's real pharmacology, even though no
scored predictor consumes them; the panel genes drive nearest-drug and
liability views.

Run: ``python scripts/p32_4_build_packages.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "neurocascade.compound_package/1"
OUT = Path(__file__).resolve().parents[1] / "src/neurocascade/benchmark/prospective/packages"

MICRO = "mL.min-1.g-1"          # == uL/min/mg protein (MICROSOME_CLINT_UNITS)
RAT_CL_UNITS = "mL.min-1.kg-1"
RAT_VSS_UNITS = "L.kg-1"


def ki(gene, value, efficacy, refs):
    return gene, {"ki_nM": value, "efficacy": efficacy, "refs": refs}


def binding(*entries):
    return {g: e for g, e in entries}


def clint_human_microsome(value, ref, description):
    return {"measurements": [{"assay_chembl_id": None, "ref": ref, "relation": "=",
                              "species": "human", "standard_type": "CL",
                              "matrix": "microsomes", "units": MICRO,
                              "description": description, "value": value}],
            "summary": [{"species": "human", "matrix": "microsomes", "units": MICRO,
                         "value": value, "min": value, "max": value,
                         "n_measurements": 1, "n_documents": 1, "n_censored": 0}]}


def plasma_fu(rows):
    """rows: list of (species, value, ref, description)."""
    meas = [{"assay_chembl_id": None, "ref": r, "relation": "=", "species": s,
             "standard_type": "Fu", "units": None, "description": d, "value": v}
            for s, v, r, d in rows]
    summ = [{"species": s, "units": None, "value": v, "min": v, "max": v,
             "n_measurements": 1, "n_documents": 1, "n_censored": 0}
            for s, v, r, d in rows]
    return {"measurements": meas, "summary": summ}


def rat_cl(value, ref, description):
    return {"measurements": [{"assay_chembl_id": None, "ref": ref, "relation": "=",
                              "species": "rat", "standard_type": "CL",
                              "units": RAT_CL_UNITS, "description": description,
                              "value": value}],
            "summary": [{"species": "rat", "units": RAT_CL_UNITS, "value": value,
                         "min": value, "max": value, "n_measurements": 1,
                         "n_documents": 1, "n_censored": 0}]}


def rat_vss(value, ref, description):
    return {"measurements": [{"assay_chembl_id": None, "ref": ref, "relation": "=",
                              "species": "rat", "standard_type": "Vdss",
                              "units": RAT_VSS_UNITS, "description": description,
                              "value": value}],
            "summary": [{"species": "rat", "units": RAT_VSS_UNITS, "value": value,
                         "min": value, "max": value, "n_measurements": 1,
                         "n_documents": 1, "n_censored": 0}]}


# References used repeatedly (short tags expanded in disclosure/ledger).
R = {
    "dedic2019": "Dedic 2019, JPET 371:1-14, PMID 31371483 (pmid_note: this paper's compound "
                 "'SEP-363856' is ulotaront, the INN assigned later; binding Ki read from "
                 "indexed excerpts, JPET PDF returned 403, not table-verified)",
    "xiao2022": "Xiao 2022, Pharm Res 39:837-850, PMID 35484370 (ulotaront named in title; "
                "SD rat, 10 mg/kg IV, Table V)",
    "letavic2015": "Letavic 2015, J Med Chem 58:5620-5636, PMID 26087021 (pmid_note: 'compound 34' "
                   "in this SAR paper is seltorexant / JNJ-42847922, the INN assigned later; Tables 4-6)",
    "hikichi2025": "Hikichi 2025, JPET 392:103624, PMID 40570549 (vornorexant named in title; "
                   "Table 1 via open-access Chino 2025 review, Int J Neuropsychopharmacol 28:pyaf077, PMC13223731)",
    "konno2024": "Konno 2024, Pharmacol Res Perspect 12:e1183, PMID 38491717 (vornorexant/TS-142 "
                 "named in title; SD rat, 1 mg/kg IV, Tables 4-5)",
    "fda216578": "FDA NDA 216578 Pharmacology Review (fezolinetant), studies ESN364-RBL-* and 2693-ME-0012/0020",
    "rorick2014": "Rorick-Kehn 2014, Neuropharmacology 77:131-144, PMID 24071566 (pmid_note: "
                  "'LY2456302' in this paper is aticaprant, the INN assigned later; Tables 1-2, SD rat, 1 mg/kg IV)",
    "nickolls2018": "Nickolls 2018, Br J Pharmacol 175:708-725, PMID 29214652 (pmid_note: this "
                    "paper's 'PF-06372865' is darigabat, the INN assigned later; Table 1)",
    "owen2019": "Owen 2019, J Med Chem 62:5773-5796, PMID 30964988 (pmid_note: this paper's "
                "identified compound 'PF-06372865' is darigabat; DMPK via ChEMBL doc CHEMBL4350959, chembl_curated)",
    "hipp2021": "Hipp 2021, Sci Rep 11:7700, PMID 33833333 (basmisanil named in title; binding "
                "Table S4; fu Methods; IV PK Suppl Table S3, Wistar rat 4.3 mg/kg)",
    "szabo2022": "Szabo 2022, J Med Chem 65:7876-7895, PMID 35584373 (pmid_note: basmisanil / "
                 "RO5186582 is a reference compound characterised in this paper, not named in the "
                 "abstract; HLM CLint via ChEMBL doc CHEMBL5154916, chembl_curated)",
}


PACKAGES = {
    # ---- allometry compounds (rat IV CL, no human CLint) ------------------
    "ulotaront": dict(
        binding=binding(
            ki("HTR7", 31.0, ["agonist"], [R["dedic2019"]]),
            ki("HTR1A", 284.0, ["agonist"], [R["dedic2019"]]),
            ki("ADRA2A", 590.0, ["agonist"], [R["dedic2019"]]),
            ki("HTR1D", 1130.0, ["agonist"], [R["dedic2019"]]),
            ki("HTR2B", 1100.0, [], [R["dedic2019"]]),
            ki("HTR1B", 1900.0, ["agonist"], [R["dedic2019"]]),
            ki("HTR2C", 2450.0, [], [R["dedic2019"]]),
            ki("HTR2A", 17250.0, [], [R["dedic2019"]]),
            ki("DRD2", 21300.0, ["partial agonist"], [R["dedic2019"]]),
        ),
        fields={"invivo_cl_preclinical": rat_cl(42.7, R["xiao2022"],
                    "Clearance in Sprague-Dawley rat after iv administration (10 mg/kg)"),
                "invivo_vss_preclinical": rat_vss(3.5, R["xiao2022"],
                    "Volume of distribution at steady state in rat after iv administration")},
    ),
    "seltorexant": dict(
        binding=binding(
            ki("HCRTR2", 10.0, ["antagonist"], [R["letavic2015"]]),
            ki("HCRTR1", 800.0, ["antagonist"], [R["letavic2015"]]),
        ),
        fields={"invivo_cl_preclinical": rat_cl(40.0, R["letavic2015"],
                    "Clearance in rat after iv administration (cmpd 34)")},
    ),
    "vornorexant": dict(
        binding=binding(
            ki("HCRTR2", 0.374, ["antagonist"], [R["hikichi2025"]]),
            ki("HCRTR1", 0.460, ["antagonist"], [R["hikichi2025"]]),
        ),
        fields={"invivo_cl_preclinical": rat_cl(35.8, R["konno2024"],
                    "Total clearance in male Sprague-Dawley rat after iv administration (1 mg/kg)"),
                "plasma_fu": plasma_fu([("rat", 0.088, R["konno2024"],
                    "Fraction unbound in rat plasma by equilibrium dialysis (Table 5)")])},
    ),
    "fezolinetant": dict(
        binding=binding(
            ki("TACR3", 21.0, ["antagonist"], [R["fda216578"]]),
        ),
        fields={"invivo_cl_preclinical": rat_cl(2.20, R["fda216578"],
                    "Clearance in rat after iv administration (1 mg/kg, study 2693-ME-0020)"),
                "plasma_fu": plasma_fu([
                    ("human", 0.49, R["fda216578"],
                     "Fraction unbound in human plasma by equilibrium dialysis (study 2693-ME-0012)"),
                    ("rat", 0.67, R["fda216578"],
                     "Fraction unbound in rat plasma by equilibrium dialysis (study 2693-ME-0012)")])},
    ),
    "aticaprant": dict(
        binding=binding(
            ki("OPRK1", 0.807, ["antagonist"], [R["rorick2014"]]),
            ki("OPRM1", 24.0, ["antagonist"], [R["rorick2014"]]),
            ki("OPRD1", 155.0, ["antagonist"], [R["rorick2014"]]),
        ),
        fields={"invivo_cl_preclinical": rat_cl(89.3, R["rorick2014"],
                    "Total body clearance in male Sprague-Dawley rat after iv administration (1 mg/kg)"),
                "plasma_fu": plasma_fu([("rat", 0.006, R["rorick2014"],
                    "Fraction unbound in rat plasma by equilibrium dialysis")]),
                "brain_fu": {"measurements": [{"assay_chembl_id": None, "ref": R["rorick2014"],
                    "relation": "=", "species": "rat", "standard_type": "Fu", "units": None,
                    "description": "Fraction unbound in rat brain by equilibrium dialysis",
                    "value": 0.003}],
                    "summary": [{"species": "rat", "units": None, "value": 0.003, "min": 0.003,
                                 "max": 0.003, "n_measurements": 1, "n_documents": 1, "n_censored": 0}]}},
    ),
    # ---- IVIVE compounds (human liver CLint + human plasma fu) ------------
    "darigabat": dict(
        binding=binding(
            ki("GABRA1", 0.18, ["positive allosteric modulator"], [R["nickolls2018"]]),
            ki("GABRA2", 2.92, ["positive allosteric modulator"], [R["nickolls2018"]]),
            ki("GABRA3", 1.06, ["positive allosteric modulator"], [R["nickolls2018"]]),
            ki("GABRA5", 18.04, ["positive allosteric modulator"], [R["nickolls2018"]]),
        ),
        fields={"clint": clint_human_microsome(9.0, R["owen2019"],
                    "Intrinsic clearance in human liver microsomes (chembl_curated)"),
                "plasma_fu": plasma_fu([
                    ("human", 0.128, R["owen2019"],
                     "Fraction unbound in human plasma (chembl_curated)"),
                    ("rat", 0.073, R["owen2019"],
                     "Fraction unbound in rat plasma (chembl_curated)")]),
                "invivo_cl_preclinical": rat_cl(22.0, R["owen2019"],
                    "Clearance in rat after iv administration (chembl_curated)"),
                "invivo_vss_preclinical": rat_vss(2.1, R["owen2019"],
                    "Volume of distribution at steady state in rat (chembl_curated)")},
    ),
    "basmisanil": dict(
        binding=binding(
            ki("GABRA5", 5.0, ["negative allosteric modulator"], [R["hipp2021"]]),
            ki("GABRA2", 458.0, [], [R["hipp2021"]]),
            ki("GABRA3", 510.0, [], [R["hipp2021"]]),
            ki("GABRA1", 1031.0, [], [R["hipp2021"]]),
        ),
        fields={"clint": clint_human_microsome(7.5, R["szabo2022"],
                    "Intrinsic clearance in human liver microsomes at 2.5 uM (chembl_curated)"),
                "plasma_fu": plasma_fu([
                    ("human", 0.055, R["hipp2021"],
                     "Fraction unbound in human/primate plasma (free fraction 5.5%)"),
                    ("rat", 0.14, R["hipp2021"],
                     "Fraction unbound in rat plasma (free fraction 14%)")]),
                "invivo_cl_preclinical": rat_cl(54.0, R["hipp2021"],
                    "Total clearance in male Wistar rat after iv administration (4.3 mg/kg, Suppl Table S3)"),
                "invivo_vss_preclinical": rat_vss(2.9, R["hipp2021"],
                    "Volume of distribution at steady state in Wistar rat (Suppl Table S3)")},
    ),
}


def build() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for name, spec in PACKAGES.items():
        pkg = {"schema": SCHEMA, "name": name, "binding": spec["binding"],
               "adme": {"fields": spec["fields"]}}
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(pkg, indent=1, sort_keys=True) + "\n")
        written.append(str(path.relative_to(OUT.parents[3])))
    return written


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
