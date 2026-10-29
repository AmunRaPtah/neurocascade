"""P32.4: observed human truth for the prospective blind-test compounds.

Added in a SEPARATE commit, AFTER the predictions and their hash are committed,
so git log shows the prediction preceded any human-truth record (acceptance
criterion).  Every value is a published human quantity with its citation.

The load-bearing distinction, recorded per compound in ``cl_basis``:

* ``iv_systemic``  - a true human intravenous systemic clearance.  The product
  predicts systemic CL, so this is the only apples-to-apples comparison.
* ``oral_clf_F_bounded`` - apparent oral clearance CL/F with an absolute oral
  bioavailability F that was estimated (not from an IV arm).  Systemic
  CL = F x (CL/F) lies within a bounded factor of CL/F.
* ``oral_clf`` - apparent oral clearance CL/F with F unmeasured in humans.
  Equals systemic CL only if F = 1; since F <= 1, CL/F >= true CL, so a correct
  systemic prediction should sit at or below the observed CL/F.

CL values are mL/min/kg; L/h figures were divided by 70 kg (shown in the note)
unless a per-kg value was published.  No number is derived from an unpublished
quantity: seltorexant has no author-stated human CL (only a value derivable
from AUC), so it carries no observed CL and is reported unscored.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src/neurocascade/benchmark/prospective/observed"


def clearance(cl_mL_min_kg, basis, source, note, extra=None):
    rec = {"cl_mL_min_kg": cl_mL_min_kg, "cl_basis": basis, "source": source, "note": note}
    if extra:
        rec.update(extra)
    return rec


OBSERVED = {
    "basmisanil": {
        "clearance": clearance(
            0.97, "iv_systemic",
            "Belubbi 2023, Pharmaceutics 15:191, PMID 36678820 (IV microdose 0.1 mg 13C, Table 1)",
            "CL/kg = 0.058 L/h/kg published; 0.058*1000/60 = 0.97 mL/min/kg. True systemic "
            "clearance (IV). An alternative NCA estimate of 7.26 L/h (Yau 2023, PMID 36647756) "
            "= 1.73 mL/min/kg at 70 kg; both indicate a low-clearance drug (~1-2 mL/min/kg).",
            {"cl_alt_mL_min_kg": 1.73, "absolute_F": 0.75,
             "absolute_F_source": "Hipp 2021, Sci Rep 11:7700, PMID 33833333 (F ~75%)"}),
        "dose_mg_per_day": {"value": "240-480", "indication": "Down syndrome / schizophrenia cognition",
                            "source": "Goeldner 2022, PMID 35123401 (120/240 mg BID); not approved"},
        "occupancy": {"status": "searched-and-empty",
                      "note": "no human GABRA5 PET occupancy with a matched dose located"},
        "disclosed": "phase 1 PK 2021-2023 (Hipp 2021; Belubbi 2023)",
    },
    "fezolinetant": {
        "clearance": clearance(
            2.57, "oral_clf_F_bounded",
            "FDA VEOZAH label NDA 216578 section 12.3 (apparent steady-state CL 10.8 L/h)",
            "CL/F = 10.8 L/h; 10800/60/70 = 2.57 mL/min/kg. Absolute F estimated >= 77% from "
            "urinary radiolabel recovery (EMA EPAR EMA/528775/2023 p.43), so systemic "
            "CL = F x CL/F lies in 1.98-2.57 mL/min/kg; single-dose observed CL/F 14.5-33.9 L/h.",
            {"cl_systemic_lower_mL_min_kg": 1.98, "absolute_F": 0.77,
             "absolute_F_source": "EMA EPAR EMA/528775/2023 p.43 (>= 77%)"}),
        "dose_mg_per_day": {"value": "45", "indication": "menopausal vasomotor symptoms (FDA approved 2023)",
                            "source": "FDA VEOZAH label NDA 216578"},
        "occupancy": {"status": "searched-and-empty", "note": "no human TACR3 PET occupancy located"},
        "disclosed": "approved 2023 (FDA); label PK NDA 216578",
    },
    "ulotaront": {
        "clearance": clearance(
            7.74, "oral_clf",
            "Galluppi 2021, CPT Pharmacometrics Syst Pharmacol 10:1245, PMID 34292664 (population PK)",
            "CL/F = 32.5 L/h (95% CI 28.9-36.5); 32500/60/70 = 7.74 mL/min/kg. Oral only; "
            "absolute F never measured in humans (animal >70%).",),
        "dose_mg_per_day": {"value": "50-75", "indication": "schizophrenia (phase 2/3; phase 3 failed)",
                            "source": "Galluppi 2021, PMID 34292664"},
        "occupancy": {"status": "not_collected",
                      "note": "ulotaront's primary target TAAR1 has no validated human PET tracer"},
        "disclosed": "phase 1/2 PK 2021 (Galluppi 2021)",
    },
    "vornorexant": {
        "clearance": clearance(
            2.50, "oral_clf",
            "Kambe 2023, Basic Clin Pharmacol Toxicol 133:576, PMID 37563858 (pmid_note: "
            "vornorexant/TS-142; 10 mg oral; "
            "value quoted in Konno 2024, PMID 38491717)",
            "CL/F = 10.5 L/h at 10 mg; 10500/60/70 = 2.50 mL/min/kg. Oral only; human F "
            "unmeasured (animal rat 7.6%, dog 58%).",),
        "dose_mg_per_day": {"value": "5-10", "indication": "insomnia (approved in Japan)",
                            "source": "Uchiyama 2026, Sleep, PMID 41001841"},
        "occupancy": {"status": "not_collected", "note": "no human orexin-receptor PET occupancy collected"},
        "disclosed": "phase 1 PK 2023 (Kambe 2023); Japan approval 2025",
    },
    "aticaprant": {
        "clearance": clearance(
            13.1, "oral_clf",
            "Naganawa 2016, JPET 356:260, PMID 26628406 (pmid_note: this paper's compound LY2456302 is aticaprant, the INN assigned later; population PK, Table 5)",
            "CL/F = 55.2 L/h; 55200/60/70 = 13.1 mL/min/kg. Oral only; human F unmeasured "
            "(animal 25%).",),
        "dose_mg_per_day": {"value": "10", "indication": "major depressive disorder (phase 2/3)",
                            "source": "Naganawa 2016, PMID 26628406 (pmid_note: LY2456302 = aticaprant; dose set for ~93% KOR saturation)"},
        "occupancy": {"target": "OPRK1", "occupancy_percent": 93, "dose_mg_per_day": 10,
                      "region": "whole brain (11C-LY2795050 PET)", "statistic": "near-saturation at trough",
                      "source": "Naganawa 2016, JPET 356:260, PMID 26628406 (pmid_note: LY2456302 = aticaprant)",
                      "status": "observed"},
        "disclosed": "phase 1 PK/PET 2016 (Naganawa 2016)",
    },
    "darigabat": {
        "clearance": clearance(
            5.14, "oral_clf",
            "Nickolls 2018, Br J Pharmacol 175:708, PMID 29214652 (pmid_note: this paper's "
            "PF-06372865 is darigabat, the INN assigned later; single oral 0.04-100 mg)",
            "CL/F range 17.35-26.86 L/h; geometric mean 21.6 L/h; 21600/60/70 = 5.14 mL/min/kg. "
            "Oral only; human F unmeasured (only relative BA tablet vs suspension measured).",
            {"cl_range_mL_min_kg": [4.13, 6.40]}),
        "dose_mg_per_day": {"value": "15-50", "indication": "focal epilepsy / anxiety (phase 2)",
                            "source": "Nickolls 2018, PMID 29214652; REALIZE NCT04244175"},
        "occupancy": {"status": "searched-and-empty",
                      "note": "human GABA-A benzodiazepine-site PET occupancy not collected here"},
        "disclosed": "phase 1 PK 2018 (Nickolls 2018)",
    },
    "seltorexant": {
        "clearance": {"cl_mL_min_kg": None, "cl_basis": "searched-and-empty",
                      "source": "Bonaventure 2015, JPET 354:471, PMID 26177655; van der Ark 2018, PMID 30182786 (pmid_note: both report JNJ-42847922/MIN-202 = seltorexant, the INN assigned later)",
                      "note": "no author-stated human CL or CL/F found; a value is only derivable from "
                              "reported AUC, which the project's no-invented-number rule forbids using "
                              "as truth. Reported unscored."},
        "dose_mg_per_day": {"value": "10-20", "indication": "insomnia / adjunctive MDD (phase 2)",
                            "source": "Mesens 2025, JAMA Psychiatry, PMID 40802194"},
        "occupancy": {"status": "not_collected", "note": "human OX2R PET occupancy not collected here"},
        "disclosed": "phase 1 PK 2015-2018",
    },
}


def build() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rec in OBSERVED.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        written.append(str(path))
    return written


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
