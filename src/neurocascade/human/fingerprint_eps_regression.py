"""P23.3: does the off-target fingerprint explain the EPS residual D2 cannot?

Phase 23's thesis test, pre-registered in ``docs/p22-preregistration.md`` (the
P23.3 section) before this module was written. The question, from Siafis 2023
(PMC10618092): haloperidol and risperidone sit at ~equal striatal D2 occupancy
yet carry unequal extrapyramidal (EPS) risk. D2 occupancy alone is already
falsified as the sole mediator (ledger V11). The pre-registered hypothesis is
that the **5-HT2A:D2 occupancy ratio** — the Meltzer serotonin-dopamine
fingerprint — explains EPS variance that D2 occupancy leaves unexplained.

This module assembles the eight-drug dataset, computes predicted D2 and 5-HT2A
occupancy at each drug's ED95 dose, fits the two pre-registered regressions, and
returns the model comparison and its achievable power. It decides nothing that
the pre-registration did not fix in advance.

The honest headline, stated up front
------------------------------------
The result is **INDETERMINATE**, exactly as the pre-registration warned it would
likely be. Adding the fingerprint ratio moves R^2 from 0.31 to 0.64 and the ratio
term carries the hypothesised negative sign, but with n = 7 the nested F-test is
not significant (p = 0.13) and the test's power to detect even a large effect is
~0.20. The trend is consistent with H1; the data cannot confirm it and cannot
reject it. It is reported as a trend, not a finding (the task's instruction).

Every number is sourced
-----------------------
* **D2 dose->occupancy**: Lako et al. 2013 (J Clin Psychopharmacol 33:675-681,
  PMID 23948784), the OccMax and OccD50 of ``D2R(%) = OccMax * dose / (dose +
  OccD50)``, transcribed verbatim from the table reproduced in Siafis 2023's
  eAppendix 2 (the same meta-analysis Siafis used to build its own
  occupancy-response curves). See :data:`LAKO_2013`.
* **ED95 dose**: Leucht et al. 2020 (Am J Psychiatry 177:342-353, PMID 31838873),
  the near-maximal (95%) effective dose per drug. See :data:`ED95_MG_LEUCHT_2020`.
* **EPS odds ratio**: Siafis et al. 2023 (Mol Psychiatry 28:3267-3277, PMID
  37537284), per-drug maximum EPS OR vs placebo. See :data:`SIAFIS_2023_MAX_OR`.
* **Off-target affinity**: PDSP Ki database full download (2026-10-01, the same
  source and date as P23.1), human-only rows (species gate), per-drug median D2
  and 5-HT2A Ki in nM, with the min/max span and row count. See :data:`PDSP_KI`.

The 5-HT2A occupancy is Ki-derived, i.e. weaker (P23.2)
-------------------------------------------------------
There is no in-vivo 5-HT2A occupancy model for these drugs (no Lako-equivalent;
searched-and-empty, recorded in the pre-registration). So 5-HT2A occupancy is
computed on P23.2's principle that one plasma concentration drives every
receptor: the in-vitro 5-HT2A:D2 affinity ratio is transferred onto the D2
dose-response, ``OccD50_5HT2A = OccD50_D2 * (Ki_5HT2A / Ki_D2)``, with
``OccMax_5HT2A = 100%`` assumed. That makes every 5-HT2A occupancy here the
*weaker Ki-derived quantity* P23.2 measured at a 0.368-occupancy-unit floor,
plus the extra assumption that the in-vitro affinity ratio transfers to the
in-vivo dose-ED50 ratio. :data:`KI_DERIVED_LABEL` is re-exported and carried so
the quantity cannot be read without its warning.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .offtarget_occupancy import KI_DERIVED_LABEL

# ---------------------------------------------------------------------------
# Sourced inputs. Never invent a number: each dict below has a citation.
# ---------------------------------------------------------------------------

#: Lako et al. 2013 dose->D2-occupancy model, ``D2R(%) = OccMax * dose /
#: (dose + OccD50)``, dose in mg/day. (OccMax %, OccD50 mg). Transcribed from the
#: table in Siafis 2023 eAppendix 2 (Springer MediaObject MOESM1), which
#: reproduces Lako 2013 (PMID 23948784) verbatim. Clozapine (61.7, 125) is
#: carried for completeness but is NOT in the regression (no ED95, no EPS OR).
LAKO_2013 = {
    "amisulpride": (85.0, 137.0),
    "aripiprazole": (86.9, 0.25),
    "clozapine": (61.7, 125.0),
    "haloperidol": (91.9, 0.65),
    "olanzapine": (96.5, 6.46),
    "quetiapine": (49.1, 352.0),
    "risperidone": (92.4, 1.07),
    "ziprasidone": (82.9, 41.7),
}

#: Leucht et al. 2020 (PMID 31838873) near-maximal (95%) effective dose, mg/day.
#: Clozapine is absent from the source's ED95 list (too few fixed-dose studies).
ED95_MG_LEUCHT_2020 = {
    "amisulpride": 537.0,   # for patients with positive symptoms
    "aripiprazole": 11.5,
    "haloperidol": 6.3,
    "olanzapine": 15.2,
    "quetiapine": 482.0,
    "risperidone": 6.3,
    "ziprasidone": 186.0,
}

#: Siafis et al. 2023 (PMID 37537284) per-drug MAXIMUM EPS odds ratio vs placebo,
#: with the dose at which it occurs and the 95% CI (CI carried for provenance,
#: not consumed). Clozapine: "no usable data" -> absent. Quetiapine's maximum is
#: a negligible 1.11 at 1200 mg/d.
SIAFIS_2023_MAX_OR = {
    "haloperidol": (7.56, 30.0, (3.16, 18.08)),
    "amisulpride": (5.79, 1200.0, (2.06, 16.25)),
    "risperidone": (3.30, 16.0, (2.37, 4.61)),
    "ziprasidone": (3.06, 320.0, (1.47, 6.38)),
    "olanzapine": (2.73, 40.0, (0.78, 9.54)),
    "aripiprazole": (1.57, 30.0, (0.97, 2.56)),
    "quetiapine": (1.11, 1200.0, (0.48, 2.55)),
}

#: PDSP Ki database full download (2026-10-01), human-only (species gate, as
#: P23.1). Per cell: (median_nM, min_nM, max_nM, n_rows). Median is the primary
#: value; the span documents the within-cell spread (amisulpride is sparse and
#: anecdotal: n=3 at D2, n=2 at 5-HT2A, consistent with its known near-absence of
#: 5-HT2A affinity). haloperidol and risperidone agree with the P23.1 registry.
PDSP_KI = {
    "haloperidol": {"D2": (1.85, 0.12, 6400.0, 62), "5HT2A": (89.0, 25.0, 435.0, 27)},
    "risperidone": {"D2": (2.8, 0.3, 27.0, 41), "5HT2A": (0.43, 0.14, 7.0, 29)},
    "olanzapine": {"D2": (21.0, 3.0, 106.0, 33), "5HT2A": (3.3, 1.3183, 24.2, 25)},
    "quetiapine": {"D2": (500.0, 16.0, 5037.0, 25), "5HT2A": (184.9122, 31.0, 2500.0, 18)},
    "aripiprazole": {"D2": (1.6, 0.34, 30.3, 19), "5HT2A": (8.85, 3.4, 151.3561, 14)},
    "ziprasidone": {"D2": (4.1, 0.8, 9.7, 16), "5HT2A": (0.39, 0.08, 1.7, 12)},
    "amisulpride": {"D2": (3.0, 1.3, 2407.0, 3), "5HT2A": (4525.0, 746.0, 8304.0, 2)},
}

#: The pre-registered denominator (P22.1, P23.3 section): the Lako-2013 eight
#: minus clozapine. Order is fixed so every array below is aligned.
DRUGS = (
    "haloperidol",
    "amisulpride",
    "risperidone",
    "ziprasidone",
    "olanzapine",
    "aripiprazole",
    "quetiapine",
)

#: Assumed maximum 5-HT2A occupancy, stated in the pre-registration. There is no
#: in-vivo 5-HT2A OccMax for these drugs (searched-and-empty), so the hyperbola
#: is left unconstrained-free at 100%; most atypicals saturate 5-HT2A by PET,
#: which makes this defensible for them and conservative for the typicals.
OCCMAX_5HT2A_ASSUMED = 100.0


# ---------------------------------------------------------------------------
# Occupancy predictions.
# ---------------------------------------------------------------------------

def d2_occupancy_at_ed95(drug):
    """D2 occupancy FRACTION [0,1] at the drug's ED95, from Lako 2013.

    ``OccMax * dose / (dose + OccD50)`` with dose = Leucht 2020 ED95, divided by
    100 to a fraction. This is an in-vivo-fitted quantity (PET-calibrated), the
    strong half of the ratio.
    """
    occmax, occd50 = LAKO_2013[drug]
    dose = ED95_MG_LEUCHT_2020[drug]
    return occmax * dose / (dose + occd50) / 100.0


def fivehtt2a_occupancy_at_ed95(drug):
    """5-HT2A occupancy FRACTION [0,1] at ED95 -- Ki-DERIVED, see label.

    Computed on P23.2's shared-plasma-concentration principle: the in-vitro
    5-HT2A:D2 affinity ratio is transferred onto the D2 dose-response so that one
    dose (hence one plasma concentration) drives both receptors --
    ``OccD50_5HT2A = OccD50_D2 * (Ki_5HT2A / Ki_D2)`` -- with
    ``OccMax_5HT2A = OCCMAX_5HT2A_ASSUMED``. This is the WEAKER quantity of
    P23.2 (see :data:`KI_DERIVED_LABEL`): an in-vitro Ki is not an in-vivo EC50,
    and this additionally assumes the affinity ratio transfers to the dose-ED50
    ratio.
    """
    _, occd50_d2 = LAKO_2013[drug]
    dose = ED95_MG_LEUCHT_2020[drug]
    ki_d2 = PDSP_KI[drug]["D2"][0]
    ki_5ht2a = PDSP_KI[drug]["5HT2A"][0]
    occd50_5ht2a = occd50_d2 * (ki_5ht2a / ki_d2)
    return OCCMAX_5HT2A_ASSUMED * dose / (dose + occd50_5ht2a) / 100.0


def occupancy_ratio_at_ed95(drug):
    """5-HT2A:D2 occupancy ratio at ED95 (the fingerprint predictor). Ki-derived."""
    return fivehtt2a_occupancy_at_ed95(drug) / d2_occupancy_at_ed95(drug)


@dataclass(frozen=True)
class DrugRow:
    drug: str
    ed95_mg: float
    occ_d2: float
    occ_5ht2a: float
    occ_ratio: float
    eps_or: float
    ln_or: float
    ki_d2_nM: float
    ki_5ht2a_nM: float
    quantity_kind_5ht2a: str = "ki-derived-occupancy"
    label_5ht2a: str = KI_DERIVED_LABEL


def build_dataset(drugs=DRUGS):
    """The aligned per-drug regression rows, every field regenerated from source."""
    rows = []
    for d in drugs:
        occ_d2 = d2_occupancy_at_ed95(d)
        occ_5 = fivehtt2a_occupancy_at_ed95(d)
        or_val = SIAFIS_2023_MAX_OR[d][0]
        rows.append(
            DrugRow(
                drug=d,
                ed95_mg=ED95_MG_LEUCHT_2020[d],
                occ_d2=occ_d2,
                occ_5ht2a=occ_5,
                occ_ratio=occ_5 / occ_d2,
                eps_or=or_val,
                ln_or=float(np.log(or_val)),
                ki_d2_nM=PDSP_KI[d]["D2"][0],
                ki_5ht2a_nM=PDSP_KI[d]["5HT2A"][0],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Regression and the pre-registered comparison.
# ---------------------------------------------------------------------------

def _ols(y, *predictors):
    """Plain OLS with an intercept. Returns (beta, sse, r2)."""
    n = len(y)
    X = np.column_stack([np.ones(n)] + [np.asarray(p, float) for p in predictors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    return beta, sse, 1.0 - sse / sst


@dataclass(frozen=True)
class RegressionResult:
    n: int
    r2_null: float          # Model 0: ln OR ~ D2 occupancy
    r2_full: float          # Model 1: + log10(5HT2A:D2 occupancy ratio)
    beta_null: np.ndarray
    beta_full: np.ndarray
    beta_ratio: float       # the fingerprint coefficient (hypothesised < 0)
    f_stat: float           # nested partial F, Model 1 vs Model 0
    f_df: tuple
    p_value: float
    increment_f2: float     # Cohen f^2 of the added predictor
    verdict: str            # "supported" | "indeterminate" | "null-favored"

    def power(self, f2, alpha=0.05):
        """Achievable power of the nested F-test for a true increment f^2.

        Noncentral-F with lambda = f2 * (df1 + df2 + 1) (Cohen 1988).
        """
        df1, df2 = self.f_df
        crit = stats.f.ppf(1 - alpha, df1, df2)
        lam = f2 * (df1 + df2 + 1)
        return float(1 - stats.ncf.cdf(crit, df1, df2, lam))

    def n_for_power(self, f2, target=0.80, alpha=0.05, cap=200):
        """Smallest n giving >= target power for this increment f^2 (or None)."""
        for n in range(4, cap + 1):
            df2 = n - 3
            if df2 < 1:
                continue
            crit = stats.f.ppf(1 - alpha, 1, df2)
            lam = f2 * (1 + df2 + 1)
            if 1 - stats.ncf.cdf(crit, 1, df2, lam) >= target:
                return n
        return None


def fit_models(drugs=DRUGS, alpha=0.05, large_f2=0.35, power_threshold=0.80):
    """Fit both pre-registered models and apply the pre-registered decision rule.

    Decision rule (fixed in docs/p22-preregistration.md, P23.3 section):
      * SUPPORTED iff p < alpha AND beta_ratio < 0.
      * else INDETERMINATE iff power to detect a large increment (f^2 >= 0.35) <
        power_threshold.
      * else NULL-FAVORED.
    """
    rows = build_dataset(drugs)
    n = len(rows)
    y = np.array([r.ln_or for r in rows])
    occ_d2 = np.array([r.occ_d2 for r in rows])
    log_ratio = np.log10(np.array([r.occ_ratio for r in rows]))

    beta0, sse0, r20 = _ols(y, occ_d2)
    beta1, sse1, r21 = _ols(y, occ_d2, log_ratio)

    df1, df2 = 1, n - 2 - 1
    f_stat = ((sse0 - sse1) / df1) / (sse1 / df2)
    p_value = float(1 - stats.f.cdf(f_stat, df1, df2))
    increment_f2 = (r21 - r20) / (1 - r21) if r21 < 1 else float("inf")
    beta_ratio = float(beta1[-1])

    res = RegressionResult(
        n=n, r2_null=r20, r2_full=r21,
        beta_null=beta0, beta_full=beta1, beta_ratio=beta_ratio,
        f_stat=float(f_stat), f_df=(df1, df2), p_value=p_value,
        increment_f2=float(increment_f2), verdict="",
    )
    if p_value < alpha and beta_ratio < 0:
        verdict = "supported"
    elif res.power(large_f2, alpha) < power_threshold:
        verdict = "indeterminate"
    else:
        verdict = "null-favored"
    return RegressionResult(**{**res.__dict__, "verdict": verdict})


def leave_one_drug_out(alpha=0.05):
    """Pre-declared sensitivity: refit dropping each drug in turn.

    Returns ``{dropped_drug: RegressionResult}``. Not scored; it exists to show
    whether the verdict depends on a single (often confounded) drug -- it does,
    which is itself part of the indeterminate reading.
    """
    out = {}
    for d in DRUGS:
        out[d] = fit_models(drugs=tuple(x for x in DRUGS if x != d), alpha=alpha)
    return out


__all__ = [
    "LAKO_2013",
    "ED95_MG_LEUCHT_2020",
    "SIAFIS_2023_MAX_OR",
    "PDSP_KI",
    "DRUGS",
    "OCCMAX_5HT2A_ASSUMED",
    "KI_DERIVED_LABEL",
    "d2_occupancy_at_ed95",
    "fivehtt2a_occupancy_at_ed95",
    "occupancy_ratio_at_ed95",
    "DrugRow",
    "build_dataset",
    "RegressionResult",
    "fit_models",
    "leave_one_drug_out",
]
