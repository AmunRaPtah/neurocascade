"""P29.2: human PK from preclinical inputs by standard textbook PK methods.

This module predicts human intravenous clearance (CL) and steady-state volume
of distribution (Vss) for the Phase 28 benchmark drugs from their preclinical
ADME records, using *mechanistic* methods with no fitted log-log regressions.
Every physiological constant is a literature value, cited below with a PMID
that was checked against PubMed to be the right paper.

It supersedes the previous P29.2 implementation, which fitted log-log
regressions (rat-CL exponent ~0.11, i.e. nearly constant; an IVIVE arm that was
a fitted regression on CLint x fu rather than the well-stirred liver model; and
a Vss model fitted on two drugs).  That approach scored held-out CL at
GMFE 6.77 (n=5), worse than simply predicting the train geometric mean
(GMFE ~3.08).  Its Vss result (GMFE 1.28, n=3) regressed human plasma Vss on
rat *brain-slice* unbound volumes, an incompatible matrix, so it was spurious.

Methods and constants
---------------------
1. IVIVE clearance -- well-stirred liver model (Rowland, Benet & Graham 1973,
   PMID 4764426; applied to microsomal CLint by Obach 1999, PMID 10534321):

       CL_H = Q_H * fu_b * CLint_u / (Q_H + fu_b * CLint_u)

   with, in mL/min/kg:
   * Q_H  = 20.7 mL/min/kg hepatic blood flow (Davies & Morris 1993,
     PMID 8378254).  The well-stirred form caps CL_H strictly below Q_H.
   * In-vitro CLint scaled to whole body with MPPGL ~= 40 mg microsomal
     protein per g liver and liver weight ~= 21.4 g/kg body weight
     (Barter et al. 2007, PMID 17266522); hepatocyte CLint scaled with
     hepatocellularity ~= 99e6 cells/g (Barter 2007; Sohlenius-Sternbeck 2006,
     PMID 16930941).
   * fu_b = fu_p / Rb, blood:plasma ratio Rb read per drug from
     ``drug_physchem.json`` (measured where public, bases/neutral default
     Rb = 1 otherwise; all recorded CNS drugs are bases or neutral, so no acid
     0.55 case is used).
   * fu_mic (microsomal binding) is now APPLIED using the Austin 2002 equation
     (PMID 12433825): ``fu_mic = 1 / (1 + C * 10**(0.56 * logP/D - 1.41))`` with
     C the microsomal protein concentration in mg/mL and logP/D = logP for bases
     and logD7.4 for neutral/acidic compounds.  P29.6 adds a sourced logP per
     drug (ChEMBL_37 ALogP, molecule id in ``drug_physchem.json``) and a
     declared default incubation protein concentration C = 1 mg/mL -- the
     concentration at which the Austin coefficients are expressed and squarely
     inside Austin's own 0.25-4 mg/mL incubation range (Austin 2002), standard
     for human-liver-microsome depletion assays (Obach 1999, PMID 10534321).
     Because every drug here is a base or a neutral, ALogP is the correct
     logP/D input (bases use logP directly; the neutral benzodiazepines carry no
     group ionized at pH 7.4, so logD7.4 = logP).  CLint_u = CLint / fu_mic, so
     a low-fu_mic lipophilic base gets a large upward correction and a
     near-unity hydrophile almost none -- the differential the omitted
     correction used to miss.  When a drug has no logP record the correction
     falls back to fu_mic = 1 (uncorrected) rather than being dropped.
   * CLint groups are keyed on (organ, in-vitro matrix, bound/unbound type,
     units) read from the measurement descriptions, so human LIVER microsomal
     CLint is never pooled with intestinal microsomes or with unbound vs total
     CLint (the ziprasidone pooling bug).  Only human liver microsome or
     hepatocyte CLint in per-protein / per-cell units is used; per-kg "CLint"
     rows (already a CL output) and S9/cytosol/brain/reactive-metabolite rows
     are rejected.
   * Aldehyde-oxidase- or otherwise non-CYP/UGT-dominant drugs are EXCLUDED
     because microsomes miss that clearance: ziprasidone is predominantly
     cleared by aldehyde oxidase (Obach 1999 context; AO is cytosolic and
     absent from the microsomal assay), so its microsomal CLint cannot predict
     its human CL.

2. Rat single-species scaling for CL -- simple allometry with the FIXED
   exponent b = 0.75 (Boxenbaum 1982, PMID 7120049), standard rat body weight
   0.25 kg and human 70 kg (Davies & Morris 1993, PMID 8378254):

       CL_human = CL_rat * (70 / 0.25) ** 0.75   (absolute), then per kg.

   No exponent is fitted.  Only rat *plasma/systemic* CL (<= 100 mL/min/kg) is
   used; rat brain-ECF/microdialysis, hepatic, distributional and intrinsic
   clearances are rejected at the measurement level.  This arm is a documented
   fallback used only when the IVIVE inputs are missing.

3. Vss -- Oie-Tozer fu-ratio method (Oie & Tozer 1979, PMID 501558):
   Vss_human = Vss_rat * fu_h / fu_rat where both fu are available, else rat
   Vss per kg directly (simple allometry, volume exponent ~1).  Only rat
   *plasma* Vss (steady-state / iv) is used; rat brain-slice unbound volume,
   central/peripheral microdialysis volumes and apparent (V/F) volumes are
   rejected.

Baselines and evaluation (see docs/validation-ledger.md V17 for the numbers):
train geometric-mean predictor and the superseded fitted method are reported
next to each metric.  Development used TRAIN only; the held-out split was
scored exactly once after the method was frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from statistics import geometric_mean, median
from typing import Iterable

from . import load_adme_record, load_split, load_truth_record

_PHYSCHEM_PATH = Path(__file__).with_name("drug_physchem.json")


# ---- truth / input field names -------------------------------------------
CL_TRUTH_FIELD = "cl_mL_min_kg"
VSS_TRUTH_FIELD = "vss_L_kg"
RAT_CL_UNITS = "mL.min-1.kg-1"
RAT_VSS_UNITS = "L.kg-1"
MICROSOME_CLINT_UNITS = "mL.min-1.g-1"
HEPATOCYTE_CLINT_UNITS = "uL.min-1.(10^6cells)-1"


# ---- cited physiological constants ----------------------------------------
@dataclass(frozen=True)
class PKConstants:
    q_h_ml_min_kg: float = 20.7          # Davies & Morris 1993, PMID 8378254
    mppgl_mg_g: float = 40.0             # Barter 2007, PMID 17266522
    liver_g_per_kg: float = 21.4         # Barter 2007, PMID 17266522
    hpgl_cells_g: float = 99.0e6         # Barter 2007 / Sohlenius-Sternbeck 2006, PMID 16930941
    blood_plasma_ratio: float = 1.0      # bases/neutral default when no measured Rb
    fu_mic: float = 1.0                  # fallback only, used when a drug has no logP
    microsomal_protein_mg_ml: float = 1.0  # Austin 2002 incubation protein C; Obach 1999
    rat_bw_kg: float = 0.25              # Davies & Morris 1993, PMID 8378254
    human_bw_kg: float = 70.0            # Davies & Morris 1993, PMID 8378254
    allometric_exponent: float = 0.75    # Boxenbaum 1982, PMID 7120049


CONSTANTS = PKConstants()

# Austin 2002 (PMID 12433825): microsomal binding from lipophilicity.
AUSTIN_SLOPE = 0.56      # coefficient on logP/D
AUSTIN_INTERCEPT = -1.41  # intercept
AUSTIN_CITATION = "Austin et al. 2002, PMID 12433825 (fu_mic from logP/D); Obach 1999, PMID 10534321 (incubation protein C)"


def load_physchem() -> dict:
    """Per-drug logP (ChEMBL_37 ALogP), ionization class and Rb, with sources.

    Keys starting with ``_`` are metadata (sources, search records), not drugs.
    """
    return json.loads(_PHYSCHEM_PATH.read_text())


_PHYSCHEM = load_physchem()


def austin_fu_mic(logp_d: float, protein_mg_ml: float) -> float:
    """Fraction unbound in the microsomal incubation, Austin 2002 (PMID 12433825).

    ``fu_mic = 1 / (1 + C * 10**(0.56 * logP/D - 1.41))`` with C the microsomal
    protein concentration in mg/mL.  ``logp_d`` is logP for bases and logD7.4 for
    neutral/acidic compounds; the caller supplies the correct term.  Returns a
    value in (0, 1]; higher binding (more lipophilic, more protein) gives lower
    fu_mic and hence a larger CLint_u = CLint / fu_mic.
    """
    return 1.0 / (1.0 + protein_mg_ml * 10.0 ** (AUSTIN_SLOPE * logp_d + AUSTIN_INTERCEPT))


def _drug_fu_mic(inn: str, k: PKConstants) -> tuple[float, tuple[str, ...]]:
    """fu_mic and its citation refs for ``inn``; (fallback 1.0, ()) if no logP."""
    entry = _PHYSCHEM.get(inn)
    if not entry or entry.get("logp") is None:
        return k.fu_mic, ()
    fu_mic = austin_fu_mic(float(entry["logp"]), k.microsomal_protein_mg_ml)
    return fu_mic, (f"ChEMBL_37 ALogP {entry['chembl_id']}", AUSTIN_CITATION)


def _drug_rb(inn: str, k: PKConstants) -> float:
    """Blood:plasma ratio for ``inn``; measured where the table applies one, else default."""
    entry = _PHYSCHEM.get(inn)
    if entry and entry.get("rb") is not None:
        return float(entry["rb"])
    return k.blood_plasma_ratio

WELL_STIRRED_CITATION = (
    "Rowland, Benet & Graham 1973, PMID 4764426; Obach 1999, PMID 10534321; "
    "Barter 2007, PMID 17266522; Davies & Morris 1993, PMID 8378254"
)
ALLOMETRY_CITATION = "Boxenbaum 1982, PMID 7120049; Davies & Morris 1993, PMID 8378254"
VSS_CITATION = "Oie & Tozer 1979, PMID 501558"

# Drugs whose human clearance is dominated by an enzyme the in-vitro matrix
# does not contain, so their microsomal CLint cannot predict human CL.
NON_MICROSOMAL_CLEARANCE = {
    "ziprasidone": "predominantly aldehyde-oxidase cleared; AO is cytosolic and "
    "absent from the liver-microsome assay (Obach 1999, PMID 10534321 context)",
}

# measurement-description tokens that mark an incompatible in-vitro CLint row
# a CLint reference this many fold from the other references' median is unit-suspect
CLINT_REF_OUTLIER_FOLD = 30.0
_CLINT_REJECT = (
    "reactive metabolite", "reactive metabolites", "gsh", "inactivation",
    "well stirred", "well-stirred", "tdi", "cytochrome p450 (unknown",
    "intestin", "brain", "cytosol", "efflux", "uptake",
)
# tokens that mark a non-plasma rat in-vivo CL row
_RAT_CL_REJECT = (
    "extracellular", "brain", "cns penetration", "distributional", "hepatic",
    "intrinsic", "csf", " ecf", "portal", "renal", "biliary",
)
# tokens that mark a non-plasma / non-Vss rat in-vivo volume row
_RAT_VSS_REJECT = (
    "unbound", "brain", "extracellular", "central volume", "peripheral",
    "apparent", "intrinsic", "slices", "csf", " ecf",
)
_RAT_CL_MAX = 100.0  # mL/min/kg; above this it is not a plasma systemic CL


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Feature:
    value: float
    references: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class Prediction:
    drug: str
    parameter: str
    method: str
    predicted: float
    observed: float
    fold_error: float
    input_references: tuple[str, ...]


@dataclass(frozen=True)
class Metric:
    parameter: str
    method: str
    n: int
    geometric_mean_fold_error: float
    within_2_fold: float
    within_3_fold: float
    predictions: tuple[Prediction, ...]


@dataclass(frozen=True)
class TranslationModel:
    """The frozen, mechanistic P29.2 model.

    Nothing per-drug is fitted.  The single optional empirical quantity is
    ``cl_bias_train`` -- the geometric mean of (predicted/observed) IVIVE CL on
    the TRAIN drugs -- recorded so a one-parameter bias correction can be
    reported separately.  ``fit_drugs`` makes the train set used for that
    estimate observable, which keeps the train/held-out leak test meaningful.
    """

    constants: PKConstants
    cl_bias_train: float
    fit_drugs: tuple[str, ...]
    citation: str = WELL_STIRRED_CITATION

    @property
    def cl_bias_correction(self) -> float:
        return 1.0 / self.cl_bias_train


# ---- input extraction -----------------------------------------------------
def _plasma_fu(record: dict, species: str) -> _Feature | None:
    blob = record.get("fields", {}).get("plasma_fu")
    if not blob:
        return None
    rows = [r for r in blob.get("summary", []) if r.get("species") == species and r.get("value")]
    if len(rows) != 1:
        return None
    refs = sorted({
        m["ref"] for m in blob.get("measurements", [])
        if m.get("relation") == "=" and m.get("species") == species and m.get("value") is not None
    })
    return _Feature(float(rows[0]["value"]), tuple(refs))


def _liver_clint(record: dict) -> _Feature | None:
    """Human LIVER CLint, grouped by (matrix, bound/unbound, units).

    Groups are built from the measurement level, not the lossy per-matrix
    summary, so liver and intestinal microsomes and total vs unbound CLint are
    never pooled.  Within the preferred group (microsomes before hepatocytes,
    total before unbound) each reference is reduced to its geometric mean, a
    reference more than CLINT_REF_OUTLIER_FOLD from the median of the others
    is dropped as unit-suspect (ChEMBL stores some mL/min/mg rows under the
    uL/min/mg-equivalent unit, 1000-fold low), and the median across the
    remaining references is returned.
    """
    blob = record.get("fields", {}).get("clint")
    if not blob:
        return None
    groups: dict[tuple, dict[str, list[float]]] = {}
    for m in blob.get("measurements", []):
        if m.get("species") != "human" or m.get("relation") != "=" or m.get("value") is None:
            continue
        desc = (m.get("description") or "").lower()
        if any(tok in desc for tok in _CLINT_REJECT):
            continue
        units, matrix = m.get("units"), m.get("matrix")
        if units == MICROSOME_CLINT_UNITS and matrix == "microsomes":
            if "liver" not in desc and "hepat" not in desc:
                continue  # organ not confirmed liver
        elif units == HEPATOCYTE_CLINT_UNITS and matrix == "hepatocytes":
            pass  # hepatocytes are liver by definition
        else:
            continue  # per-kg CL rows, S9, nL/pmol/min etc. are not CLint inputs
        bound = "unbound" if "unbound" in desc else "total"
        key = ("liver", matrix, bound, units)
        groups.setdefault(key, {}).setdefault(m.get("ref"), []).append(float(m["value"]))
    if not groups:
        return None
    key = sorted(groups, key=lambda k: (0 if k[1] == "microsomes" else 1,
                                        0 if k[2] == "total" else 1))[0]
    per_ref = {ref: geometric_mean(v) for ref, v in groups[key].items()}
    kept = {
        ref: v for ref, v in per_ref.items()
        if len(per_ref) < 3 or max(v, mid := median(x for r, x in per_ref.items() if r != ref))
        / min(v, mid) <= CLINT_REF_OUTLIER_FOLD
    } or per_ref  # references disagree with each other too much to single one out
    return _Feature(median(kept.values()), tuple(sorted(kept)),
                    detail=f"{key[1]}/{key[2]}")


def _rat_cl(record: dict) -> _Feature | None:
    blob = record.get("fields", {}).get("invivo_cl_preclinical")
    if not blob:
        return None
    vals: list[float] = []
    refs: set = set()
    for m in blob.get("measurements", []):
        if (m.get("species") != "rat" or m.get("relation") != "=" or m.get("value") is None
                or m.get("units") != RAT_CL_UNITS):
            continue
        desc = (m.get("description") or "").lower()
        if any(tok in desc for tok in _RAT_CL_REJECT):
            continue
        v = float(m["value"])
        if v > _RAT_CL_MAX:
            continue
        vals.append(v)
        refs.add(m.get("ref"))
    return _Feature(geometric_mean(vals), tuple(sorted(refs))) if vals else None


def _rat_vss(record: dict) -> _Feature | None:
    blob = record.get("fields", {}).get("invivo_vss_preclinical")
    if not blob:
        return None
    vals: list[float] = []
    refs: set = set()
    for m in blob.get("measurements", []):
        if (m.get("species") != "rat" or m.get("relation") != "=" or m.get("value") is None
                or m.get("units") != RAT_VSS_UNITS):
            continue
        desc = (m.get("description") or "").lower()
        if "volume of distribution" not in desc or any(tok in desc for tok in _RAT_VSS_REJECT):
            continue
        vals.append(float(m["value"]))
        refs.add(m.get("ref"))
    return _Feature(geometric_mean(vals), tuple(sorted(refs))) if vals else None


def _human_truth(inn: str, parameter: str) -> tuple[float, tuple[str, ...]] | None:
    field = CL_TRUTH_FIELD if parameter == "cl" else VSS_TRUTH_FIELD
    blob = load_truth_record(inn).get("pk", {}).get("iv_compilations", {}).get(field)
    if not blob:
        return None
    for route in ("iv", "unspecified"):
        rows = [r for r in blob.get("summary", []) if r.get("route") == route and r.get("value") is not None]
        if rows:
            refs = sorted({
                m["ref"] for m in blob.get("measurements", [])
                if m.get("relation") == "=" and m.get("route") == route
            })
            return float(rows[0]["value"]), tuple(refs)
    return None


# ---- mechanistic predictions ----------------------------------------------
def _ivive_cl(record: dict, inn: str, k: PKConstants) -> _Feature | None:
    """Well-stirred human CL_H in mL/min/kg from human liver CLint and fu_p."""
    if inn in NON_MICROSOMAL_CLEARANCE:
        return None
    clint = _liver_clint(record)
    fu = _plasma_fu(record, "human")
    if clint is None or fu is None:
        return None
    if clint.detail.startswith("microsomes"):
        clint_vivo = clint.value * k.mppgl_mg_g / 1000.0 * k.liver_g_per_kg
    else:  # hepatocytes: uL/min/1e6 cells -> mL/min/kg
        clint_vivo = clint.value * 1e-3 * k.hpgl_cells_g / 1e6 * k.liver_g_per_kg
    fu_mic, fu_mic_refs = _drug_fu_mic(inn, k)
    clint_u = clint_vivo / fu_mic
    fu_b = fu.value / _drug_rb(inn, k)
    cl_h = k.q_h_ml_min_kg * fu_b * clint_u / (k.q_h_ml_min_kg + fu_b * clint_u)
    refs = tuple(sorted(set(clint.references + fu.references + fu_mic_refs)))
    return _Feature(cl_h, refs)


def _allometric_cl(record: dict, k: PKConstants) -> _Feature | None:
    rat = _rat_cl(record)
    if rat is None:
        return None
    cl = rat.value * k.rat_bw_kg * (k.human_bw_kg / k.rat_bw_kg) ** k.allometric_exponent / k.human_bw_kg
    return _Feature(cl, rat.references)


def _vss(record: dict, k: PKConstants) -> _Feature | None:
    rat = _rat_vss(record)
    if rat is None:
        return None
    fh = _plasma_fu(record, "human")
    fr = _plasma_fu(record, "rat")
    if fh is not None and fr is not None:
        refs = tuple(sorted(set(rat.references + fh.references + fr.references)))
        return _Feature(rat.value * fh.value / fr.value, refs)
    return _Feature(rat.value, rat.references)  # simple allometry, volume exponent 1


def _cl_prediction(inn: str, model: TranslationModel, corrected: bool) -> Prediction | None:
    record = load_adme_record(inn)
    k = model.constants
    feat = _ivive_cl(record, inn, k)
    method = "well-stirred IVIVE (human liver CLint, human fu)"
    if feat is None:
        feat = _allometric_cl(record, k)
        method = "rat single-species allometry (fixed exponent 0.75), IVIVE-missing fallback"
    if feat is None:
        return None
    truth = _human_truth(inn, "cl")
    if truth is None:
        return None
    predicted = feat.value * model.cl_bias_correction if corrected else feat.value
    if corrected:
        method += f" x TRAIN bias {model.cl_bias_correction:.3f}"
    observed, truth_refs = truth
    return Prediction(
        drug=inn, parameter="cl", method=method,
        predicted=predicted, observed=observed,
        fold_error=max(predicted / observed, observed / predicted),
        input_references=tuple(sorted(set(feat.references + truth_refs))),
    )


def _vss_prediction(inn: str, k: PKConstants) -> Prediction | None:
    record = load_adme_record(inn)
    feat = _vss(record, k)
    if feat is None:
        return None
    truth = _human_truth(inn, "vss")
    if truth is None:
        return None
    observed, truth_refs = truth
    return Prediction(
        drug=inn, parameter="vss", method="Oie-Tozer fu-ratio (rat Vss, fu_h/fu_rat)",
        predicted=feat.value, observed=observed,
        fold_error=max(feat.value / observed, observed / feat.value),
        input_references=tuple(sorted(set(feat.references + truth_refs))),
    )


# ---- fitting (only the optional single bias factor, TRAIN only) ------------
def fit_translation_models(train_drugs: Iterable[str] | None = None) -> TranslationModel:
    """Build the frozen P29.2 model, estimating the TRAIN-only CL bias factor.

    The default is the frozen P28.1 training split.  Only training records are
    read; the leak test monkeypatches the loaders to prove this.
    """
    drugs = tuple(load_split()["train"] if train_drugs is None else train_drugs)
    ratios: list[float] = []
    used: list[str] = []
    k = CONSTANTS
    for inn in drugs:
        record = load_adme_record(inn)
        feat = _ivive_cl(record, inn, k)
        truth = _human_truth(inn, "cl")
        if feat is not None and truth is not None:
            ratios.append(feat.value / truth[0])
            used.append(inn)
    if len(ratios) < 2:
        raise ValueError(f"only {len(ratios)} train IVIVE CL pairs; need >= 2 for a bias factor")
    return TranslationModel(
        constants=k,
        cl_bias_train=geometric_mean(ratios),
        fit_drugs=tuple(used),
    )


# ---- scoring ---------------------------------------------------------------
def _metric(parameter: str, method: str, predictions: list[Prediction]) -> Metric:
    folds = [p.fold_error for p in predictions]
    n = len(folds)
    return Metric(
        parameter=parameter, method=method, n=n,
        geometric_mean_fold_error=geometric_mean(folds),
        within_2_fold=sum(f <= 2.0 for f in folds) / n,
        within_3_fold=sum(f <= 3.0 for f in folds) / n,
        predictions=tuple(predictions),
    )


def score(drugs: Iterable[str], model: TranslationModel | None = None,
          corrected: bool = False) -> dict[str, Metric]:
    """Score the mechanistic predictions over ``drugs``.

    No per-drug coefficient is fitted, so a no-fit pass over the train drugs is
    its own leave-one-drug-out result.  Only parameters with >= 1 compatible
    prediction appear in the result; Vss is absent when no drug in ``drugs``
    has a compatible rat *plasma* Vss paired with human Vss truth.
    """
    model = model or fit_translation_models()
    drugs = tuple(drugs)
    cl = [p for p in (_cl_prediction(d, model, corrected) for d in drugs) if p is not None]
    vss = [p for p in (_vss_prediction(d, model.constants) for d in drugs) if p is not None]
    out: dict[str, Metric] = {}
    if cl:
        out["cl"] = _metric("cl", cl[0].method.split(" x TRAIN")[0], cl)
    if vss:
        out["vss"] = _metric("vss", vss[0].method, vss)
    return out


def score_train() -> dict[str, Metric]:
    """Primary evaluation: no-fit (== leave-one-drug-out) over the TRAIN drugs."""
    return score(load_split()["train"])


def score_heldout(corrected: bool = False) -> dict[str, Metric]:
    """Secondary evaluation: the frozen held-out split, scored once."""
    model = fit_translation_models()
    return score(load_split()["heldout"], model=model, corrected=corrected)


def train_geomean_baseline(parameter: str) -> tuple[float, int]:
    """Leave-one-out geometric-mean-of-TRAIN predictor, the honest floor."""
    train = load_split()["train"]
    obs = [t[0] for t in (_human_truth(inn, parameter) for inn in train) if t is not None]
    folds = []
    for i, o in enumerate(obs):
        rest = obs[:i] + obs[i + 1:]
        gm = geometric_mean(rest)
        folds.append(max(gm / o, o / gm))
    return geometric_mean(folds), len(obs)


__all__ = [
    "PKConstants",
    "CONSTANTS",
    "AUSTIN_CITATION",
    "austin_fu_mic",
    "load_physchem",
    "TranslationModel",
    "Metric",
    "Prediction",
    "fit_translation_models",
    "score",
    "score_train",
    "score_heldout",
    "train_geomean_baseline",
    "NON_MICROSOMAL_CLEARANCE",
]
