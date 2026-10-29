"""P32.4: a prospective blind test of ``predict_compound`` on recently disclosed
CNS compounds whose human data are absent from this frozen repository.

P28.1's held-out set (P32.3's blind demo) was in the repository during Phases
19-25, so it cannot certify the product: the drugs were seen.  This module runs
the product on compounds that are **absent from the repository entirely** -- no
binding panel, no ADME record, no human truth -- so the frozen P34.3 candidate
and the P30.3/P30.4 bands could not have been tuned to them.

The anti-gaming discipline is the point (see the task's acceptance criterion 1).
The prediction file ``results/P32.4_predictions.json`` and its SHA-256 are
committed **before** any human-truth record for these compounds exists in git
history.  The commit that adds the observed truth and the scoring comes after,
so ``git log --diff-filter=A`` shows the prediction landing first.  The
prediction is a pure function of the preclinical package and the frozen model,
so the file the scorer reads is byte-identical to the one that was hashed.

**Operational definition of "prospective" (rule 11).**  The brief speaks of
compounds "disclosed after the model freeze".  The only part of that this
repository can *check* is the git ordering above, so that is what is enforced:
every compound here has no truth record in the frozen repo, and its prediction
is committed before its truth is.  The real-world disclosure date is narrative,
not verifiable from the checkout, and is not relied upon.

**What the product actually predicts (rule 11).**  ``predict_compound`` returns
one compound-specific scored quantity, human clearance (P29.2 IVIVE or rat
allometry, P30.3 band).  Target-engagement dose and panel occupancy come back
``pending`` (P29.3/P29.4 are searched-and-empty), and without a projected human
Cmax every liability set is the P30.4 train-prevalence ``{0,1}`` -- it covers
the truth but carries no information.  So the scored headline of this test is
clearance; dose, occupancy and effects are tabulated beside the observed value
as the acceptance criterion asks, but only clearance is a real comparison.
Every record carries ``not validated: P34 returned no verdict`` (P32.1).

The compounds were chosen by a rule that reads no human truth: CNS new chemical
entities absent from the v2 binding panel, the v2 human-PK truth and the P28
benchmark, for which a preclinical package could be built **from preclinical
literature alone** -- at least one measured receptor affinity and a clearance
input the P29.2 arm accepts (rat i.v. plasma CL, or human liver-microsome /
hepatocyte CLint with a human fu).  Candidates surveyed but not buildable are
recorded in :data:`COULD_NOT_BUILD` with the exact missing field (acceptance
criterion 3); they are not silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..human_pk_translation import MICROSOME_CLINT_UNITS, RAT_CL_UNITS
from ..conformal import load_bands
from ...compound import SCHEMA, CoreInputMissing, predict_compound

PROSPECTIVE_SCHEMA = "neurocascade.prospective/1"
LEVELS = ("0.80", "0.95")

#: Frozen P33.3 v2-train arithmetic-mean clearance, the "mean baseline" the
#: acceptance criterion asks to be shown beside each prediction.  Read from the
#: frozen band file, not recomputed.
MEAN_BASELINE_CL = load_bands()["quantities"]["human_cl_v2_train_mean"]["frozen_point"]


# ---- package builders (preclinical inputs only) -----------------------------
def _rat_cl_adme(value: float, ref: str) -> dict[str, Any]:
    """An ADME record carrying one rat i.v. plasma clearance (allometry input)."""
    m = {"species": "rat", "relation": "=", "value": value, "units": RAT_CL_UNITS,
         "standard_type": "CL", "description": "clearance in rat after iv administration",
         "ref": ref}
    return {"fields": {"invivo_cl_preclinical": {
        "measurements": [m],
        "summary": [{"species": "rat", "units": RAT_CL_UNITS, "value": value,
                     "min": value, "max": value, "n_measurements": 1,
                     "n_documents": 1, "n_censored": 0}]}}}


def _clint_adme(clint: float, fu: float, ref: str) -> dict[str, Any]:
    """An ADME record carrying human liver-microsome CLint and a human fu (IVIVE input).

    ``clint`` is in uL/min/mg microsomal protein, which is numerically the
    benchmark's ``mL.min-1.g-1`` (uL/mg == mL/g), the unit P29.2's extractor
    reads and the P30.3 band was calibrated on.
    """
    return {"fields": {
        "clint": {
            "measurements": [{"species": "human", "relation": "=", "value": clint,
                              "units": MICROSOME_CLINT_UNITS, "matrix": "microsomes",
                              "standard_type": "CLint",
                              "description": "intrinsic clearance in human liver microsomes",
                              "ref": ref}],
            "summary": [{"species": "human", "matrix": "microsomes",
                         "units": MICROSOME_CLINT_UNITS, "value": clint}]},
        "plasma_fu": {
            "measurements": [{"species": "human", "relation": "=", "value": fu, "ref": ref}],
            "summary": [{"species": "human", "value": fu}]}}}


#: Per-compound sourcing note, kept out of the package dict so the package
#: stays a valid ``neurocascade.compound_package/1`` (validate_package rejects
#: unknown fields).  Carried alongside in the prediction file.
SOURCING: dict[str, str] = {}


def _pkg(name: str, binding: Mapping[str, Any], adme: Mapping[str, Any],
         *, provenance: str) -> dict[str, Any]:
    SOURCING[name] = provenance
    return {"schema": SCHEMA, "name": name, "binding": dict(binding), "adme": dict(adme)}


# Every number below is literature-sourced; the citation travels in each binding
# entry's ``refs`` and the ADME measurement's ``ref``, and is spelled out in the
# package's ``_provenance``.  No human clearance, dose or label value appears in
# any package -- those are the truth, added in a later commit.
PACKAGES: dict[str, dict[str, Any]] = {
    "aticaprant": _pkg(
        "aticaprant",
        {"OPRK1": {"ki_nM": 0.807, "efficacy": ["antagonist"], "refs": ["PMID24071566"]},
         "OPRM1": {"ki_nM": 24.0, "efficacy": ["antagonist"], "refs": ["PMID24071566"]},
         "OPRD1": {"ki_nM": 155.0, "efficacy": ["antagonist"], "refs": ["PMID24071566"]}},
        _rat_cl_adme(89.3, "PMID24071566"),
        provenance="aticaprant (JNJ-67953964 / LY2456302 / CERC-501), kappa-opioid "
        "antagonist. Human KOR/MOR/DOR Ki and rat i.v. CL 89.3 mL/min/kg (1 mg/kg, "
        "cannulated SD rats) from Rorick-Kehn et al. 2014, Neuropharmacology 77:131-144, "
        "PMID 24071566."),
    "ulotaront": _pkg(
        "ulotaront",
        {"HTR1A": {"ki_nM": 284.0, "efficacy": ["agonist"],
                   "refs": ["DOI:10.1021/acsmedchemlett.1c00527"]}},
        _rat_cl_adme(42.7, "PMID35484370"),
        provenance="ulotaront (SEP-363856), TAAR1/5-HT1A agonist. Radioligand 5-HT1A "
        "binding Ki 284 nM from Koblan et al. 2021, ACS Med Chem Lett (DOI "
        "10.1021/acsmedchemlett.1c00527); TAAR1 agonism is functional (EC50 ~38 nM, "
        "Dedic et al. 2019, JPET 371:1-14, PMID 31371483) and is not a binding Ki, so it "
        "is not used as the package affinity. Rat i.v. CL 42.7 mL/min/kg and rat F ~100% "
        "from Xiao et al. 2022, Pharm Res 39:837-850, PMID 35484370."),
    "samidorphan": _pkg(
        "samidorphan",
        {"OPRM1": {"ki_nM": 0.052, "efficacy": ["antagonist"], "refs": ["NDA213378"]},
         "OPRK1": {"ki_nM": 0.23, "efficacy": ["partial agonist"], "refs": ["NDA213378"]},
         "OPRD1": {"ki_nM": 2.7, "efficacy": ["partial agonist"], "refs": ["NDA213378"]}},
        _clint_adme(11.15, 0.72, "NDA213378"),
        provenance="samidorphan (ALKS 33), opioid antagonist. Human opioid-receptor Ki, "
        "human liver-microsome CLint 11.15 uL/min/mg (Study AM-0313-02) and human plasma "
        "protein binding 23-33% (fu taken as 0.72) from the FDA Lybalvi Multidiscipline "
        "Review, NDA 213378 (2021)."),
}

#: Surveyed CNS NMEs absent from the repository that could NOT be packaged from
#: open preclinical sources, each with the field that was missing.  Listed, not
#: dropped (acceptance criterion 3).
COULD_NOT_BUILD: list[dict[str, str]] = [
    {"name": "xanomeline", "missing": "preclinical clearance input",
     "detail": "no rat i.v. plasma CL and no quantitative human microsome/hepatocyte "
     "CLint in the public record; human CL/F (1950 L/h, FDA NDA 216158) is unusable as a "
     "systemic-CL comparator because oral F is <1%."},
    {"name": "centanafadine", "missing": "P29.2-compatible clearance input",
     "detail": "clearance is MAO-A mediated (CLint 5546 uL/min/mg on MAO-A; ~80% unchanged "
     "after 4 h in human hepatocytes, FDA NDA 218145), which the liver-microsome assay "
     "misses, and no rat i.v. plasma CL is reported; binding is IC50, not Ki."},
    {"name": "seltorexant", "missing": "preclinical clearance input",
     "detail": "OX2R Ki 0.13 nM is published, but no numeric rat i.v. plasma CL or human "
     "microsome CLint is in the open literature (Bonaventure et al. 2015, JPET, is paywalled)."},
    {"name": "emraclidine", "missing": "receptor Ki and clearance input",
     "detail": "M4 potency is a functional EC50 (12.3 nM, Butler et al. 2024 J Med Chem), "
     "not a binding Ki, and no rat i.v. CL / human microsome CLint is openly available."},
    {"name": "navacaprant", "missing": "P29.2-compatible clearance input",
     "detail": "KOR Ki 1.5 nM is published (Guerrero et al. 2019, PMID 30707578) but the "
     "only rat i.v. CL is 105 mL/min/kg (fed rats), above the plasma-CL plausibility cap, "
     "and the hepatocyte/microsome CLint is reported pre-scaled to mL/min/kg, not the "
     "per-protein unit the extractor reads."},
    {"name": "pridopidine", "missing": "preclinical clearance input",
     "detail": "S1R Ki 57 nM and clean human CL (541 mL/min, extensive metabolisers) are "
     "published, but no rat i.v. plasma CL or human microsome CLint is openly available."},
    {"name": "difelikefalin", "missing": "P29.2-compatible clearance input",
     "detail": "KOR Ki 0.32 nM is published (FDA NDA 214916), but difelikefalin is a "
     "renally-cleared peptide with no hepatic CLint and no clean rat i.v. plasma CL; its "
     "only human CL is a hemodialysis-population popPK estimate."},
    {"name": "zelquistinel", "missing": "measured receptor Ki",
     "detail": "rat i.v. CL 11.2 mL/min/kg is published (PMID 35882204) but zelquistinel "
     "is a spermine-site NMDA allosteric modulator with no measurable binding Ki at any "
     "site, so the binding panel has no measured affinity and the product refuses."},
    {"name": "deutetrabenazine", "missing": "a single dosed entity",
     "detail": "deutetrabenazine is a prodrug; the human PK (CL/F 47 and 70 L/h, FDA NDA "
     "208082) is for the active metabolites alpha/beta-HTBZ, not the dosed molecule the "
     "binding panel describes, and the product has no metabolite model."},
]


# ---- running the blind prediction -------------------------------------------
def _jsonsafe(obj: Any) -> Any:
    """Infinity -> null (JSON has no Infinity); recurse through dicts/lists."""
    if isinstance(obj, float):
        return None if math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonsafe(v) for v in obj]
    return obj


def _clearance_view(result: Mapping[str, Any]) -> dict[str, Any]:
    cl = result["human_pk"]["clearance"]
    bands = {lv: {"lower": b["lower"], "upper": b["upper"], "bounded": b["upper"] != math.inf}
             for lv, b in cl["bands"].items()}
    return {
        "predicted_cl_mL_min_kg": cl["point"],
        "band": cl["provenance"]["band"],
        "predictor": cl["provenance"]["predictor"],
        "compound_specific": cl["provenance"]["compound_specific"],
        "bands": bands,
        "mean_baseline_cl_mL_min_kg": MEAN_BASELINE_CL,
        "input_references": cl["provenance"].get("input_references", []),
    }


def _liability_view(result: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for effect, rec in result["liabilities"].items():
        if rec.get("status") == "predicted":
            out[effect] = {"status": "predicted", "point_kind": rec.get("point_kind"),
                           "is_probability": rec.get("is_probability"),
                           "label_sets": {lv: list(b["labels"]) for lv, b in rec["bands"].items()},
                           "compound_specific": rec["provenance"].get("compound_specific")}
        else:
            out[effect] = {"status": rec.get("status"), "reason": rec.get("reason")}
    return out


def predict_one(name: str) -> dict[str, Any]:
    """Run ``predict_compound`` on one prospective package; blind view of the result."""
    result = predict_compound(PACKAGES[name])
    return {
        "compound": name,
        "clearance": _clearance_view(result),
        "target_engagement_dose": {"status": result["target_engagement_dose"]["status"],
                                   "reason": result["target_engagement_dose"]["reason"]},
        "panel_occupancy": {"status": result["panel_occupancy_time"]["status"],
                            "reason": result["panel_occupancy_time"]["reason"]},
        "liabilities": _liability_view(result),
        "nearest_benchmark_drugs": result["nearest_benchmark_drugs"]["status"],
        "provenance": result["provenance"],
    }


def predictions_blob() -> dict[str, Any]:
    """The full, JSON-safe, truth-free prediction record for all buildable compounds."""
    preds = {name: predict_one(name) for name in sorted(PACKAGES)}
    return _jsonsafe({
        "schema": PROSPECTIVE_SCHEMA,
        "task": "P32.4",
        "kind": "prospective blind predictions (no human truth)",
        "mean_baseline_cl_mL_min_kg": MEAN_BASELINE_CL,
        "compounds_predicted": sorted(PACKAGES),
        "packages": {name: {**PACKAGES[name]} for name in sorted(PACKAGES)},
        "sourcing": {name: SOURCING[name] for name in sorted(PACKAGES)},
        "predictions": preds,
        "could_not_build": COULD_NOT_BUILD,
        "provenance": "not validated: P34 returned no verdict",
    })


def predictions_bytes() -> bytes:
    """Canonical serialisation hashed in the ledger and read back by the scorer."""
    return (json.dumps(predictions_blob(), indent=1, sort_keys=True) + "\n").encode()


def sha256_predictions() -> str:
    return hashlib.sha256(predictions_bytes()).hexdigest()


def write_predictions(path: str | Path) -> str:
    Path(path).write_bytes(predictions_bytes())
    return sha256_predictions()


# ---- scoring (added in the truth commit) ------------------------------------
_OBSERVED_DIR = Path(__file__).resolve().parent / "observed"


def load_observed(name: str) -> dict[str, Any] | None:
    """Observed human truth for one compound, or None if its file is absent.

    The truth lives in files added in a commit *after* the prediction commit.
    A missing file is the blind state and a legitimate searched-and-empty
    outcome, exactly as a held-out drug is in this sandbox (P32.1/P32.3).
    """
    path = _OBSERVED_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fold(a: float, b: float) -> float:
    return max(a / b, b / a)


def score_prospective() -> dict[str, Any]:
    """Compare each prospective prediction to the observed human value.

    Clearance is the only compound-specific scored quantity. For each compound
    the predicted CL, the P33.3 train-mean baseline and the observed CL are put
    side by side with fold-errors and band coverage. Dose and panel occupancy
    are shown as observed-not-scored (predict_compound returns them pending).
    Liabilities are scored for coverage only, and ``n_informative`` counts how
    many label sets are not the uninformative ``{0,1}`` -- which, with no
    projected human Cmax, is zero.
    """
    preds = predictions_blob()["predictions"]
    per_compound: dict[str, Any] = {}
    cl_fold, baseline_fold = [], []
    cov = {lv: 0 for lv in LEVELS}
    missing_truth = []
    liab_covered = {e: 0 for e in ("sedation", "anticholinergic")}
    liab_n = {e: 0 for e in ("sedation", "anticholinergic")}
    n_informative = 0

    for name in sorted(PACKAGES):
        obs = load_observed(name)
        pred = preds[name]
        cl = pred["clearance"]
        if obs is None:
            missing_truth.append(name)
            per_compound[name] = {"status": "missing_truth",
                                  "predicted_cl_mL_min_kg": cl["predicted_cl_mL_min_kg"],
                                  "mean_baseline_cl_mL_min_kg": cl["mean_baseline_cl_mL_min_kg"]}
            continue
        ocl = obs["clearance"]["value_mL_min_kg"]
        fe = _fold(cl["predicted_cl_mL_min_kg"], ocl)
        feb = _fold(cl["mean_baseline_cl_mL_min_kg"], ocl)
        cl_fold.append(fe)
        baseline_fold.append(feb)
        band_cov = {}
        for lv, b in cl["bands"].items():
            lo, hi = b["lower"], (math.inf if not b["bounded"] else b["upper"])
            inside = lo <= ocl <= hi
            band_cov[lv] = inside
            cov[lv] += int(inside)
        # liabilities: coverage of the observed 0/1 label by the predicted set
        liab = {}
        for effect in ("sedation", "anticholinergic"):
            eff_obs = obs["effects"].get(effect, {}).get("observed")
            rec = pred["liabilities"].get(effect, {})
            sets = rec.get("label_sets")
            if eff_obs is None or not sets:
                liab[effect] = {"status": "not_scored"}
                continue
            labels80 = sets.get("0.80", [])
            covered = eff_obs in labels80
            informative = set(labels80) != {0, 1}
            liab[effect] = {"observed": eff_obs, "predicted_set_0.80": labels80,
                            "covered": covered, "informative": informative,
                            "train_prevalence_baseline": rec.get("point_kind")}
            liab_covered[effect] += int(covered)
            liab_n[effect] += 1
            n_informative += int(informative)
        per_compound[name] = {
            "status": "scored",
            "predicted_cl_mL_min_kg": cl["predicted_cl_mL_min_kg"],
            "observed_cl_mL_min_kg": ocl,
            "observed_route": obs["clearance"]["route"],
            "mean_baseline_cl_mL_min_kg": cl["mean_baseline_cl_mL_min_kg"],
            "fold_error_prediction": fe,
            "fold_error_mean_baseline": feb,
            "band_covered": band_cov,
            "predictor": cl["predictor"],
            "dose": {"observed": obs["dose"]["value"], "predicted": "pending"},
            "occupancy": {"observed": obs["occupancy"]["status"], "predicted": "pending"},
            "liabilities": liab,
        }

    n = len(cl_fold)

    def _gmfe(xs):
        return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else None

    return {
        "task": "P32.4",
        "kind": "prospective blind test, scored",
        "status": "scored" if n else "searched-and-empty",
        "n_scored": n,
        "missing_truth": missing_truth,
        "clearance": {
            "n": n,
            "gmfe_prediction": _gmfe(cl_fold),
            "gmfe_mean_baseline": _gmfe(baseline_fold),
            "band_coverage": {lv: f"{cov[lv]}/{n}" for lv in LEVELS} if n else {},
            "finding": "compound-specific clearance vs the train-mean baseline on the "
                       "prospective compounds; a prediction worse than the baseline is a "
                       "negative result and is reported as such",
        },
        "liabilities": {
            "coverage": {e: f"{liab_covered[e]}/{liab_n[e]}" for e in liab_covered if liab_n[e]},
            "n_informative": n_informative,
            "note": "all label sets are the P30.4 train-prevalence {0,1} (no projected human "
                    "Cmax), so coverage is trivial and carries no information",
        },
        "per_compound": per_compound,
        "could_not_build": COULD_NOT_BUILD,
        "provenance": "not validated: P34 returned no verdict",
    }


def write_score(path: str | Path) -> dict[str, Any]:
    score = score_prospective()
    out = {"task": "P32.4", "n_scored": score["n_scored"],
           "n_scored_definition": "recently-disclosed CNS compounds, absent from the repo, "
           "run through predict_compound from a preclinical-only package and scored on "
           "compound-specific human clearance against truth committed after the prediction",
           "clearance_gmfe_prediction": score["clearance"]["gmfe_prediction"],
           "clearance_gmfe_mean_baseline": score["clearance"]["gmfe_mean_baseline"],
           "missing_truth": score["missing_truth"],
           "could_not_build": [r["name"] for r in COULD_NOT_BUILD],
           "provenance": score["provenance"]}
    Path(path).write_text(json.dumps(out, indent=1) + "\n")
    return out


__all__ = [
    "COULD_NOT_BUILD",
    "MEAN_BASELINE_CL",
    "PACKAGES",
    "SOURCING",
    "load_observed",
    "score_prospective",
    "write_score",
    "predict_one",
    "predictions_blob",
    "predictions_bytes",
    "sha256_predictions",
    "write_predictions",
]
