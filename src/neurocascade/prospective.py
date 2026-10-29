"""P32.4: prospective blind test on recently disclosed, repository-absent CNS compounds.

The P28.1 held-out set was in the repository during P19-P25, so it cannot
certify the product (P32.3 is retrospective).  This task runs ``predict_compound``
on CNS compounds that are **absent from every benchmark input and truth file**
in this repository (no ``binding/``, ``adme/``, ``truth/``, ``v2/`` record, and
not in ``split.json``) and whose human data were disclosed around or after the
model freeze (the 2026-10-01 benchmark draw and the frozen P34.3 candidate /
P30.4 bands).  The prediction is built from **preclinical sources only** and
committed (``results/P32.4_predictions.json``, SHA-256 in the ledger) before any
human-truth record for these compounds exists in this repository.

### What the source says, against the brief (rule 11)

The brief names xanomeline-trospium, zuranolone and gepirone.  Two of those are
*not* absent: ``zuranolone`` and ``gepirone`` already have full benchmark
``binding``/``adme``/``truth`` records and v2 PK/liability truth in this repo, so
they are held-out, not prospective, and are excluded.  Xanomeline (absent) is
surveyed but cannot be built (below).

### The load-bearing check, run first (rule 12)

The product's only compound-specific scored quantity is **human systemic
clearance** (P32.1/P32.3: dose, occupancy and half-life come back ``pending``;
effect sets are the uninformative ``{0,1}`` train-prevalence fallback without a
projected human exposure).  A clean prospective CL test therefore needs, per
compound: (1) a measured orthosteric binding Ki, (2) a readable preclinical
clearance input (human liver CLint + human fu, or rat in-vivo CL) that
``predict_compound`` can consume, and (3) a published human **systemic** CL
(IV or absolute-bioavailability study) to score against, because the P30.3
``human_cl_ivive`` band was calibrated on IV clearance (P28.4), not on apparent
oral CL/F.

This intersection was checked against the literature before building, and it is
nearly empty for repository-absent post-freeze CNS compounds (``SURVEY`` below,
full record in ``docs/architecture.md`` and the ledger):

* The muscarinic PAMs (emraclidine, mevidalen) have no orthosteric Ki at all,
  only a functional EC50 at an allosteric site, so ``predict_compound`` refuses
  them for a missing core input.
* For aticaprant, seltorexant, xanomeline, ecopipam and brilaroxazine a measured
  Ki is readable, but the preclinical clearance input the model consumes (rat
  in-vivo CL in mL/min/kg, or human liver CLint + fu) sits in paywalled DMPK
  tables; no rule-1 value is readable, so the package cannot be built.
* Only **ulotaront** has a fully readable preclinical package (5-HT1A Ki
  0.28 uM, Dedic 2019 PMID 31371483; rat IV CL 27.4 mL/min/kg and human fu
  >0.78, PMID 35484370), so it is the one compound ``predict_compound`` runs on.
* **No** surveyed compound has a published human **systemic** CL.  Every one
  reports only apparent oral CL/F (ulotaront 32.5 L/h, Galluppi 2021; aticaprant
  28.7 L/h, Lowe 2014; xanomeline 1950 L/h at F~1%, Cobenfy label; ...).  So the
  systemic-CL comparison the product's band is calibrated for is
  searched-and-empty for every compound, and ``n_scored = 0``.

The ulotaront systemic-CL prediction (6.70 mL/min/kg) is committed here as a
standing prospective prediction; it is shown beside ulotaront's apparent oral
CL/F as context only, explicitly uncounted (apparent CL/F >= systemic CL, and
with F>0.7 the two differ by <=1.43x, negligible against the model's x/7.55 80%
band, but it is still not the IV quantity the band scores).

Every record carries ``PROVENANCE_LABEL`` (``not validated: P34 returned no
verdict``), as P32.1-P32.3 do.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .compound import (
    CoreInputMissing,
    PROVENANCE_LABEL,
    load_package,
    predict_compound,
)

_HERE = Path(__file__).parent
PACKAGES_DIR = _HERE / "benchmark" / "prospective" / "packages"

#: v2 train arithmetic-mean human CL (P33.3 frozen point, ledger V22), the
#: compound-agnostic mean baseline shown beside every per-compound prediction.
MEAN_BASELINE_CL_ML_MIN_KG = 11.112250247706422

#: The surveyed compounds.  Each is absent from every benchmark input/truth file
#: in this repository (asserted by ``tests/test_prospective.py``).  ``package``
#: names the committed preclinical package file when the compound is buildable;
#: ``missing_field`` names the single blocking gap when it is not.  Observed
#: human values are recorded as context; ``systemic_cl_ml_min_kg`` is ``None``
#: for every compound because none has a published human IV/absolute-F clearance
#: (``apparent_oral_cl`` records the CL/F that does exist, with its source).
SURVEY: tuple[dict[str, Any], ...] = (
    {
        "name": "ulotaront",
        "aliases": ["SEP-363856"],
        "mechanism": "TAAR1 agonist with 5-HT1A agonist activity",
        "indication": "schizophrenia",
        "disclosure": "phase 2 positive 2020 (Koblan NEJM); phase 3 readouts 2023-2024",
        "buildable": True,
        "package": "ulotaront.json",
        "binding_source": "5-HT1A radioligand Ki 0.28 uM (Dedic 2019, PMID 31371483)",
        "preclinical_cl_source": "rat IV CL 27.4 mL/min/kg; human fu >0.78 (Fonseca, Pharm Res 2022, PMID 35484370)",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "searched-and-empty: no human IV or absolute-bioavailability "
                                  "clearance published; only apparent oral CL/F exists",
            "apparent_oral_cl": {
                "cl_f_L_per_h": 32.5,
                "cl_f_ml_min_kg_at_70kg": 7.738,
                "oral_F": ">0.70 (BCS1, high solubility/permeability)",
                "source": "Galluppi 2021 population PK, CPT:PSP, DOI 10.1002/psp4.12692 "
                          "(healthy tablet CL/F 30.0 L/h)",
            },
            "dose": "25-100 mg once daily (phase 2/3)",
            "occupancy": "non-D2/non-5-HT2A mechanism; no striatal D2 occupancy expected",
            "effects": "low EPS, low weight gain, low prolactin (phase 2/3)",
        },
    },
    {
        "name": "xanomeline",
        "aliases": ["LY246708", "NNC11-0232", "component of KarXT/Cobenfy"],
        "mechanism": "M1/M4 muscarinic agonist",
        "indication": "schizophrenia (Cobenfy, approved 2024-09)",
        "disclosure": "FDA approval 2024-09 (NDA 216158)",
        "buildable": False,
        "missing_field": "preclinical clearance input: no readable rat in-vivo CL (mL/min/kg) "
                         "or human liver CLint; rat IV t1/2 <0.5 h is reported only "
                         "qualitatively (npp 2025). Human clearance is dominated by gut FMO "
                         "and first-pass CYP, which a liver-microsome CLint would not capture",
        "binding_source": "M4 Ki 20.0 nM, M1 Ki 79.4 nM (Watson/Shekhar); oxotremorine-M Ki ~2 nM (Bymaster)",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "searched-and-empty: only apparent oral CL/F published",
            "apparent_oral_cl": {
                "cl_f_L_per_h": 1950.0,
                "oral_F": "~1% (extensive first-pass)",
                "source": "Cobenfy FDA integrated review / label, NDA 216158 (2024)",
            },
            "dose": "100/20 to 125/30 mg (xanomeline/trospium) twice daily",
        },
    },
    {
        "name": "aticaprant",
        "aliases": ["JNJ-67953964", "LY2456302", "CERC-501"],
        "mechanism": "kappa-opioid receptor antagonist",
        "indication": "major depressive disorder",
        "disclosure": "phase 3 MDD program (read out 2024-2025)",
        "buildable": False,
        "missing_field": "preclinical clearance input: no readable rat in-vivo CL or human "
                         "liver CLint (DMPK table paywalled)",
        "binding_source": "KOR Ki 0.807 nM, MOR 24 nM, DOR 155 nM (Rorick-Kehn 2014)",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "searched-and-empty: only apparent oral CL/F published; "
                                  "human absolute F not published",
            "apparent_oral_cl": {
                "cl_f_L_per_h": 28.7,
                "oral_F": "not published for humans",
                "source": "Lowe 2014, J Clin Pharmacol 54:968, PMID 24619932 (10 mg SAD)",
            },
            "dose": "10 mg once daily (MDD)",
            "occupancy": "full KOR occupancy at 10 mg (human PET)",
        },
    },
    {
        "name": "seltorexant",
        "aliases": ["JNJ-42847922", "MIN-202"],
        "mechanism": "selective orexin-2 receptor antagonist",
        "indication": "major depressive disorder / insomnia",
        "disclosure": "phase 3 MDD/insomnia (ongoing/recent)",
        "buildable": False,
        "missing_field": "preclinical clearance input: no readable rat in-vivo CL or human "
                         "liver CLint (Bonaventure 2015 DMPK table paywalled)",
        "binding_source": "OX2R pKi 8.0 (~10 nM), OX1R pKi 6.1 (Bonaventure 2015, PMID 26177655)",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "searched-and-empty: absolute-bioavailability/IV microdose "
                                  "study (NCT05236868) results not published",
            "apparent_oral_cl": {
                "cl_f_L_per_h": None,
                "oral_F": "absolute-F study registered, results unpublished",
                "source": "t1/2 2-3 h, CYP3A4 (first-in-human, NCT02230878)",
            },
            "dose": "20 mg once daily",
        },
    },
    {
        "name": "ecopipam",
        "aliases": ["SCH-39166", "PSYRX-101"],
        "mechanism": "dopamine D1/D5 receptor antagonist",
        "indication": "Tourette syndrome / stuttering",
        "disclosure": "phase 2b Tourette (2023); human mass-balance 2026",
        "buildable": False,
        "missing_field": "preclinical clearance input: no readable rat in-vivo CL or human "
                         "liver CLint; human clearance is UGT-glucuronidation dominated "
                         "(non-microsomal), which the IVIVE microsome path would not capture",
        "binding_source": "D1 Ki 1.2 nM, D5 Ki 2.0 nM, D2 980 nM (Chipkin/McQuade)",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "searched-and-empty: only oral exposure / mass balance published",
            "apparent_oral_cl": {
                "cl_f_L_per_h": None,
                "oral_F": "not published",
                "source": "t1/2 17.3 h; mass balance 83% urinary metabolites (Schmith 2026, Clin Transl Sci, PMC13180694)",
            },
            "dose": "~100-200 mg/day (phase 2)",
        },
    },
    {
        "name": "brilaroxazine",
        "aliases": ["RP5063", "oceperidone"],
        "mechanism": "dopamine-serotonin partial agonist (D2/D3/5-HT1A/5-HT2A)",
        "indication": "schizophrenia",
        "disclosure": "phase 3 RECOVER (2024)",
        "buildable": False,
        "missing_field": "preclinical clearance input: no readable rat in-vivo CL or human "
                         "liver CLint (DMPK not published numerically)",
        "binding_source": "D2S Ki 0.28 nM, D2L 0.45 nM, D3 3.7 nM, 5-HT1A 1.5 nM, 5-HT2A 2.5 nM",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "searched-and-empty: human CL not published numerically",
            "apparent_oral_cl": {
                "cl_f_L_per_h": None,
                "oral_F": "not published",
                "source": "once-daily PK (phase 1, pop-PK 175 patients)",
            },
            "dose": "15-50 mg once daily",
        },
    },
    {
        "name": "emraclidine",
        "aliases": ["CVL-231", "PF-06852231"],
        "mechanism": "M4 muscarinic positive allosteric modulator",
        "indication": "schizophrenia",
        "disclosure": "phase 1b (2022); phase 2 EMPOWER terminated 2024",
        "buildable": False,
        "missing_field": "no measured orthosteric binding Ki: a positive allosteric modulator "
                         "is characterised by functional EC50 (12.2 nM) at an allosteric site, "
                         "not an orthosteric Ki, so predict_compound refuses it (no core Ki)",
        "binding_source": "M4 functional EC50 12.2 nM (Butler 2024, J Med Chem) - not an orthosteric Ki",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "not applicable (compound refused upstream for a missing Ki)",
        },
    },
    {
        "name": "mevidalen",
        "aliases": ["LY3154207"],
        "mechanism": "dopamine D1 positive allosteric modulator",
        "indication": "Lewy body dementia / Parkinson's disease",
        "disclosure": "phase 2 PRESENCE (2021-2022)",
        "buildable": False,
        "missing_field": "no measured orthosteric binding Ki: a D1 positive allosteric "
                         "modulator has a functional EC50 at an allosteric site, not an "
                         "orthosteric Ki, so predict_compound refuses it (no core Ki)",
        "binding_source": "D1 functional EC50 (allosteric) - not an orthosteric Ki",
        "observed_human": {
            "systemic_cl_ml_min_kg": None,
            "systemic_cl_status": "not applicable (compound refused upstream for a missing Ki)",
        },
    },
)

SELECTION_RULE = (
    "CNS compounds absent from every benchmark input and truth file in this "
    "repository (no binding/, adme/, truth/ or v2/ record, and not in "
    "split.json), whose human data were disclosed around or after the "
    "2026-10-01 model freeze. navacaprant, zuranolone and gepirone are NOT "
    "absent (they carry full benchmark/v2 records) and are excluded."
)


def _compound_prediction(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Run ``predict_compound`` from the committed preclinical package, or record
    why the compound could not be built.  Buildable compounds carry the full
    ``predict_compound`` result; unbuildable ones carry ``missing_field``."""
    out: dict[str, Any] = {
        "name": entry["name"],
        "aliases": list(entry.get("aliases", [])),
        "mechanism": entry["mechanism"],
        "indication": entry["indication"],
        "disclosure": entry["disclosure"],
        "binding_source": entry.get("binding_source"),
        "observed_human": entry["observed_human"],
        "provenance": {"label": PROVENANCE_LABEL},
    }
    if not entry.get("buildable"):
        out["buildable"] = False
        out["missing_field"] = entry["missing_field"]
        out["prediction"] = None
        return out
    package = load_package(PACKAGES_DIR / entry["package"])
    try:
        result = predict_compound(package)
    except CoreInputMissing as exc:  # pragma: no cover - buildable packages do not refuse
        out["buildable"] = False
        out["missing_field"] = f"predict_compound refused: {exc.missing}"
        out["prediction"] = None
        return out
    out["buildable"] = True
    out["preclinical_cl_source"] = entry.get("preclinical_cl_source")
    cl = result["human_pk"]["clearance"]
    out["prediction"] = {
        "human_systemic_cl": {
            "point_ml_min_kg": cl.get("point"),
            "bands": cl.get("bands"),
            "predictor": cl["provenance"].get("predictor"),
            "compound_specific": cl["provenance"].get("compound_specific"),
        },
        "mean_baseline_cl_ml_min_kg": MEAN_BASELINE_CL_ML_MIN_KG,
        "full_result": result,
    }
    return out


def build_predictions() -> dict[str, Any]:
    """The committed prediction record: deterministic from the preclinical
    packages and the frozen P34.3 candidate / P30.3-P30.4 bands only.  Reads no
    human-truth file (``observed_human`` values are literals carried in
    ``SURVEY`` as context, never consumed by ``predict_compound``)."""
    compounds = [_compound_prediction(e) for e in SURVEY]
    return {
        "task": "P32.4",
        "title": "Prospective blind test on recently disclosed, repository-absent CNS compounds",
        "selection_rule": SELECTION_RULE,
        "provenance": {"label": PROVENANCE_LABEL},
        "n_surveyed": len(compounds),
        "n_buildable": sum(1 for c in compounds if c["buildable"]),
        "mean_baseline_cl_ml_min_kg": MEAN_BASELINE_CL_ML_MIN_KG,
        "compounds": compounds,
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def predictions_sha256(predictions: Mapping[str, Any] | None = None) -> str:
    """SHA-256 of the canonical (sorted-key) prediction record, the value
    recorded in the ledger before any human truth is added."""
    if predictions is None:
        predictions = build_predictions()
    return hashlib.sha256(_canonical(predictions).encode()).hexdigest()


def _fold_error(a: float, b: float) -> float:
    return max(a / b, b / a)


def score() -> dict[str, Any]:
    """Score each buildable compound's predicted human **systemic** CL against a
    published human systemic CL.

    No surveyed compound has one (all publish apparent oral CL/F only), so the
    clean comparison the P30.3 band is calibrated for is searched-and-empty and
    ``n_scored = 0``.  The ulotaront prediction is reported with its apparent
    oral CL/F as uncounted context (``apparent_cl_context``)."""
    predictions = build_predictions()
    scored: list[dict[str, Any]] = []
    searched_empty: list[dict[str, Any]] = []
    apparent_context: list[dict[str, Any]] = []
    for c in predictions["compounds"]:
        if not c["buildable"]:
            continue
        obs = c["observed_human"]
        systemic = obs.get("systemic_cl_ml_min_kg")
        pred = c["prediction"]["human_systemic_cl"]["point_ml_min_kg"]
        if systemic is not None:
            scored.append({
                "name": c["name"],
                "predicted_ml_min_kg": pred,
                "observed_ml_min_kg": systemic,
                "fold_error": _fold_error(pred, systemic),
                "mean_baseline_fold_error": _fold_error(MEAN_BASELINE_CL_ML_MIN_KG, systemic),
            })
        else:
            searched_empty.append({"name": c["name"], "reason": obs.get("systemic_cl_status")})
            app = obs.get("apparent_oral_cl") or {}
            cl_f_kg = app.get("cl_f_ml_min_kg_at_70kg")
            if cl_f_kg is not None:
                apparent_context.append({
                    "name": c["name"],
                    "predicted_systemic_cl_ml_min_kg": pred,
                    "observed_apparent_cl_f_ml_min_kg_at_70kg": cl_f_kg,
                    "apparent_fold_error_upper_bound": _fold_error(pred, cl_f_kg),
                    "mean_baseline_apparent_fold_error": _fold_error(MEAN_BASELINE_CL_ML_MIN_KG, cl_f_kg),
                    "oral_F": app.get("oral_F"),
                    "caveat": "apparent oral CL/F, not the IV quantity the P30.3 band scores; "
                              "CL/F >= systemic CL, so this fold-error is an upper bound. Uncounted.",
                    "source": app.get("source"),
                })
    return {
        "task": "P32.4",
        "status": "searched-and-empty" if not scored else "scored",
        "n_scored": len(scored),
        "n_surveyed": predictions["n_surveyed"],
        "n_buildable": predictions["n_buildable"],
        "scored": scored,
        "systemic_cl_searched_and_empty": searched_empty,
        "apparent_cl_context": apparent_context,
        "unbuildable": [
            {"name": c["name"], "missing_field": c["missing_field"]}
            for c in predictions["compounds"] if not c["buildable"]
        ],
        "provenance": {"label": PROVENANCE_LABEL},
        "note": "The product predicts human systemic CL; no repository-absent post-freeze CNS "
                "compound surveyed has a published human systemic (IV/absolute-F) clearance, so "
                "the systemic-CL comparison is searched-and-empty for all and n_scored=0. "
                "predictions SHA-256 recorded in docs/validation-ledger.md.",
    }


__all__ = [
    "MEAN_BASELINE_CL_ML_MIN_KG",
    "SELECTION_RULE",
    "SURVEY",
    "build_predictions",
    "predictions_sha256",
    "score",
]
