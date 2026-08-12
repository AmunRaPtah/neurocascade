"""P1.6: validation of simulated concentration-time curves against published
PK data, with the circularity made visible.

Why this module exists, and what it may and may not claim
---------------------------------------------------------

`drug_pk.py` builds five PK models from published *parameter sets*
(clearances, volumes, absorption constants). The obvious "validation" - run
the model and check it reproduces the clearance it was built from - is a
round trip, not a validation: it tests the ODE integrator and the unit
arithmetic, and it cannot possibly fail for a reason that has anything to do
with the drug. Reporting that as agreement with the literature would be the
quantitative equivalent of citing yourself.

So every comparison in this module is tagged with a `tier`:

    ROUND_TRIP         the target is the model's own closed-form value.
                       Tolerance is numerical, the test MUST pass, and
                       passing says nothing about the drug. 16 comparisons.
    SOURCE_CONSISTENCY the target is a number the parameterising source
                       printed but did not contribute to the model (a
                       half-life where volumes were the inputs, a bootstrap
                       replicate set). Weak: a miss here is usually the
                       source disagreeing with itself at the precision it
                       printed. 3 comparisons.
    INDEPENDENT        the target number comes from a study that contributed
                       nothing to the model's parameters. This is the only
                       tier that can tell us the model is wrong about the
                       rat, and 9 of its 15 comparisons do.

The independent tier is built out of four rat studies that `drug_pk.py` does
not use for any parameter:

    haloperidol, IV   [Lei2018]        0.7 mg/kg IV, plasma, n = 7
    diazepam,    IV   [Rautiola2019]   1.0 mg/kg IV, plasma, pooled NCA
    fentanyl,    SC   [CanfieldSprague2025] 0.3 mg/kg SC, plasma, n = 6
    fentanyl,    IP   [Almousa2011]    0.1 mg/kg IP, plasma, n = 3

The fentanyl pair are already recorded in `drug_pk.py` as data-without-a-model
(no ka, so no SC or IP model is built); that is exactly what makes them usable
here as independent targets for the IV model's *disposition*.

Pass/fail criteria, and why they are not adjustable
---------------------------------------------------

Each comparison declares one of five `criterion_kind`s, chosen by what the
target number is rather than by how close the model got:

  * `NUMERICAL`   - round-trip only; |rel err| <= `NUMERICAL_TOLERANCE`.
  * `DISPERSION`  - the prediction must fall within the observed mean
    +/- 2 * (the dispersion the source itself prints). Where the source
    prints SEM and states n, SEM is converted to SD by sqrt(n); where it
    prints SEM and n is ambiguous (pooled sparse sampling), the SEM is used
    *as is*, which makes the criterion stricter, not looser, and is flagged
    by `dispersion_is_understated`.
  * `TWO_FOLD`    - 0.5 <= predicted/observed <= 2.0, the conventional
    predictive-performance criterion of the PBPK literature. Used where the
    source prints no dispersion at all.
  * `INTERVAL`    - the prediction must lie inside the interval the source's
    own printed rounding leaves open (e.g. "1.5 h" -> 87-93 min).
  * `UPPER_BOUND` - a structural impossibility test, not a fit: an implied
    bioavailability must be <= 1.

`meets_two_fold_criterion` and `meets_dispersion_criterion` are computed for
every record regardless of which one is primary, so a comparison that fails
one and passes the other says so. And `tests/test_pk_validation.py` pins the
*relative error itself* for every comparison, so a tolerance cannot be
widened to rescue a failure: the number that would have to change is the
error, not the threshold.

What comes out of it, in one paragraph
--------------------------------------

Of the four quantities P1.6 was asked to validate: **Cmax** has an
independent target for one of three drugs, and misses it 3.4-fold; **Tmax**
has one for none of them, structurally (see `TMAX_IS_NOT_VALIDATABLE`);
**terminal half-life** has one for all three and is the worst-predicted
quantity in the module (-47%, +101%, -19%/-73%); **AUC** has one for all
three and misses by -25% (haloperidol), +38% over 90 min (diazepam) and -38%
(fentanyl, as an upper bound that should not have been an under-prediction
at all). Underneath the AUC numbers sits a harder result: the fentanyl IV
clearance implies a subcutaneous bioavailability of 1.53, which is not a
disagreement but an impossibility. The detail, per comparison, is in
`validation_report()` and in docs/architecture.md's P1.6 section.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

from . import drug_pk
from .pk import LN2, DosedTwoCompartmentPK

# ---------------------------------------------------------------------------
# Tiers and criteria
# ---------------------------------------------------------------------------

#: The target is the model's own analytic value. Cannot fail for a drug
#: reason; it tests the integrator, the NCA and the unit arithmetic.
ROUND_TRIP = "round-trip"
#: The target is a number the parameterising source PRINTED but that was not
#: itself used as a model input (e.g. [ChengPaalzow1992] prints both V_beta,
#: which the model uses, and a terminal half-life, which it does not). A miss
#: here is a statement about the source's internal consistency at the
#: precision it printed, not about the rat and not about the model.
SOURCE_CONSISTENCY = "source-consistency"
#: The target number comes from a study that contributed no model parameter.
INDEPENDENT = "independent"

#: Criterion kinds. `passed` dispatches on these; nothing else does.
NUMERICAL = "numerical"
DISPERSION = "dispersion"
TWO_FOLD_CRITERION = "two-fold"
UPPER_BOUND = "upper-bound"
INTERVAL = "interval"

#: Tolerance for the round-trip tier. This is a *numerical* tolerance: it
#: covers trapezoidal integration of the simulated grid, the log-linear
#: regression used to recover lambda_z, and the tail extrapolation, nothing
#: else. 0.5% is ~25x the largest error the grid in `simulate_curve` actually
#: produces (2e-4, on the diazepam IP Tmax, which is limited by the grid
#: spacing rather than by the integrator; every other round-trip comparison
#: comes in under 1e-5). `test_pk_validation.py` pins those real errors, so
#: this constant is a guard rail, not a place to hide behind.
NUMERICAL_TOLERANCE = 0.005

#: The predictive-performance criterion used when a source prints no
#: dispersion: predicted within a factor of two of observed. Conventional in
#: the PBPK literature; stated here so it is a named convention rather than a
#: number chosen after seeing the answer.
TWO_FOLD = 2.0


class ValidationResult(object):
    """One (predicted, observed) comparison with its provenance and verdict.

    Deliberately not a namedtuple: the point of this object is that you
    cannot read `predicted` and `observed` off it without also being handed
    `tier`, `source` and `criterion`.
    """

    def __init__(
        self,
        key,
        drug,
        species,
        route,
        quantity,
        units,
        predicted,
        observed,
        source,
        tier,
        observed_dispersion=None,
        dispersion_kind=None,
        observed_n=None,
        criterion_kind=None,
        interval=None,
        note="",
    ):
        self.key = key
        self.drug = drug
        self.species = species
        self.route = route
        self.quantity = quantity
        self.units = units
        self.predicted = float(predicted)
        self.observed = float(observed)
        self.source = source
        self.tier = tier
        self.observed_dispersion = (
            None if observed_dispersion is None else float(observed_dispersion)
        )
        self.dispersion_kind = dispersion_kind
        self.observed_n = observed_n
        self.interval = None if interval is None else (float(interval[0]),
                                                       float(interval[1]))
        if criterion_kind is None:
            if tier == ROUND_TRIP:
                criterion_kind = NUMERICAL
            elif self.interval is not None:
                criterion_kind = INTERVAL
            elif observed_dispersion is not None:
                criterion_kind = DISPERSION
            else:
                criterion_kind = TWO_FOLD_CRITERION
        self.criterion_kind = criterion_kind
        self.note = note

    # --- error measures ----------------------------------------------------

    @property
    def relative_error(self):
        """(predicted - observed) / observed. Signed, so the direction of the
        miss is part of the record."""
        return (self.predicted - self.observed) / self.observed

    @property
    def fold_error(self):
        """predicted / observed."""
        return self.predicted / self.observed

    @property
    def sd_equivalent(self):
        """The dispersion converted to an SD, or None.

        SEM -> SD needs n. When the source prints SEM and n is unavailable or
        ambiguous, the SEM is returned unconverted, which yields a *narrower*
        interval and therefore a stricter test. `dispersion_is_understated`
        says which of the two happened.
        """
        if self.observed_dispersion is None:
            return None
        if self.dispersion_kind == "SEM" and self.observed_n:
            return self.observed_dispersion * math.sqrt(self.observed_n)
        return self.observed_dispersion

    @property
    def dispersion_is_understated(self):
        """True when an SEM is being used as if it were an SD because n was
        not usable - i.e. when the dispersion criterion is stricter than the
        source's own animal-to-animal spread would justify."""
        return self.dispersion_kind == "SEM" and not self.observed_n

    @property
    def z_score(self):
        """(predicted - observed) / SD, or None when no dispersion exists."""
        sd = self.sd_equivalent
        if sd is None or sd == 0.0:
            return None
        return (self.predicted - self.observed) / sd

    # --- verdicts ----------------------------------------------------------

    @property
    def meets_dispersion_criterion(self):
        z = self.z_score
        if z is None:
            return None
        return abs(z) <= 2.0

    @property
    def meets_two_fold_criterion(self):
        return (1.0 / TWO_FOLD) <= self.fold_error <= TWO_FOLD

    @property
    def criterion(self):
        if self.criterion_kind == NUMERICAL:
            return "numerical (|rel err| <= %.3g)" % NUMERICAL_TOLERANCE
        if self.criterion_kind == INTERVAL:
            return "inside [%.4g, %.4g], the interval the source's own " \
                   "printed precision leaves open" % self.interval
        if self.criterion_kind == UPPER_BOUND:
            return "structurally bounded: must be <= %.4g" % self.observed
        if self.criterion_kind == DISPERSION:
            return "within 2 %s of the reported mean" % (self.dispersion_kind or "SD")
        return "two-fold (no dispersion published)"

    @property
    def passed(self):
        if self.criterion_kind == NUMERICAL:
            return abs(self.relative_error) <= NUMERICAL_TOLERANCE
        if self.criterion_kind == INTERVAL:
            return self.interval[0] <= self.predicted <= self.interval[1]
        if self.criterion_kind == UPPER_BOUND:
            return self.predicted <= self.observed
        if self.criterion_kind == DISPERSION:
            return bool(self.meets_dispersion_criterion)
        return self.meets_two_fold_criterion

    def __repr__(self):
        return "ValidationResult(%s: pred=%.4g obs=%.4g, %+.1f%%, %s)" % (
            self.key,
            self.predicted,
            self.observed,
            100.0 * self.relative_error,
            "PASS" if self.passed else "FAIL",
        )


# ---------------------------------------------------------------------------
# Simulating the curve and doing non-compartmental analysis on it
# ---------------------------------------------------------------------------


def simulate_curve(model, dose_mg_per_kg, t_end_min=None, dt_min=None):
    """Integrate `model` through the same ODE protocol `engine.Simulation`
    uses and return `(times_min, concentrations)`.

    This is deliberately not the analytic solution. The quantity P1.6 is
    asked to validate is the *simulated* concentration-time curve, i.e. what
    a `Simulation` actually produces, so the metrics below are computed on an
    integrated trajectory with the integrator's error in them. (That the two
    agree is itself asserted, in `test_pk.py`'s
    `test_iv_bolus_matches_analytic_exponential` and friends.)

    Grid choice: `dt` defaults to 1/400 of the *fastest* half-life in the
    model, so a two-compartment distribution phase is resolved rather than
    trapezoid-chorded across, and `t_end` defaults to 12 terminal half-lives,
    which leaves under 0.03% of AUC(0-inf) in the extrapolated tail.
    """
    fast_half_life = getattr(model, "distribution_half_life_min", None)
    ka = getattr(model, "ka_per_min", None)
    candidates = [model.terminal_half_life_min]
    if fast_half_life:
        candidates.append(fast_half_life)
    if ka:
        candidates.append(LN2 / ka)
    if dt_min is None:
        dt_min = min(candidates) / 400.0
    if t_end_min is None:
        t_end_min = 12.0 * model.terminal_half_life_min
    n = int(round(t_end_min / dt_min)) + 1
    times = np.linspace(0.0, t_end_min, n)
    y0 = model.initial_state(dose_mg_per_kg)
    sol = solve_ivp(
        lambda t, s: model.concentration_rhs(s),
        (0.0, t_end_min),
        y0,
        t_eval=times,
        rtol=1e-11,
        atol=1e-16,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError("PK integration failed: %s" % sol.message)
    return times, np.asarray([model.concentration(col) for col in sol.y.T])


def linear_trapezoid_auc(times, concentrations, t_upper=None):
    """AUC by the linear trapezoidal rule, optionally truncated at `t_upper`.

    Linear (not log-linear) trapezoid on purpose: it is what WinNonlin's
    default and every source quoted here used, so the model's AUC and the
    published AUC are computed the same way.
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    if t_upper is not None:
        if t_upper > t[-1]:
            raise ValueError(
                "t_upper=%g is beyond the simulated window (%g)" % (t_upper, t[-1])
            )
        keep = t <= t_upper
        t, c = t[keep], c[keep]
        if t[-1] < t_upper:  # land exactly on the boundary by interpolation
            c_end = np.interp(t_upper, times, concentrations)
            t = np.append(t, t_upper)
            c = np.append(c, c_end)
    return float(np.trapezoid(c, t)) if hasattr(np, "trapezoid") else float(
        np.trapz(c, t)
    )


def terminal_rate_constant(times, concentrations, from_fraction=0.5):
    """lambda_z by ordinary least squares on ln(C) over the last
    `from_fraction` of the simulated window, plus the fit's R^2.

    Returns `(lambda_z_per_min, r_squared, n_points)`. The window is taken
    late deliberately: a two-compartment curve is only log-linear once the
    distribution phase is over, and taking the tail of a 12-half-life window
    guarantees that.
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(concentrations, dtype=float)
    start = t[-1] * (1.0 - from_fraction)
    keep = (t >= start) & (c > 0.0)
    t, c = t[keep], c[keep]
    if len(t) < 3:
        raise ValueError("not enough positive points to fit a terminal phase")
    slope, intercept = np.polyfit(t, np.log(c), 1)
    fitted = slope * t + intercept
    ss_res = float(np.sum((np.log(c) - fitted) ** 2))
    ss_tot = float(np.sum((np.log(c) - np.mean(np.log(c))) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return -float(slope), r_squared, len(t)


class CurveMetrics(object):
    """Non-compartmental summary of one simulated curve."""

    def __init__(self, times, concentrations):
        self.times = np.asarray(times, dtype=float)
        self.concentrations = np.asarray(concentrations, dtype=float)
        peak = int(np.argmax(self.concentrations))
        self.cmax = float(self.concentrations[peak])
        self.tmax_min = float(self.times[peak])
        self.auc_0_t = linear_trapezoid_auc(self.times, self.concentrations)
        lam, r2, n = terminal_rate_constant(self.times, self.concentrations)
        self.lambda_z_per_min = lam
        self.terminal_fit_r_squared = r2
        self.terminal_fit_n_points = n
        self.terminal_half_life_min = LN2 / lam
        #: AUC(0-inf) = AUC(0-t_last) + C(t_last)/lambda_z, the standard NCA
        #: extrapolation.
        self.auc_0_inf = self.auc_0_t + float(self.concentrations[-1]) / lam

    def auc_to(self, t_upper):
        return linear_trapezoid_auc(self.times, self.concentrations, t_upper)

    def __repr__(self):
        return (
            "CurveMetrics(Cmax=%.4g, Tmax=%.4g min, t1/2=%.4g min, "
            "AUC(0-inf)=%.5g)"
            % (self.cmax, self.tmax_min, self.terminal_half_life_min, self.auc_0_inf)
        )


def metrics_for(model, dose_mg_per_kg, **kwargs):
    times, concentrations = simulate_curve(model, dose_mg_per_kg, **kwargs)
    return CurveMetrics(times, concentrations)


# ---------------------------------------------------------------------------
# Sources used ONLY here, i.e. that contributed no parameter to any model
# ---------------------------------------------------------------------------
#
#   [Lei2018]    Lei K, He GF, Zhang CL, et al. "Investigation of the
#                synergistic effects of haloperidol combined with Calculus
#                Bovis Sativus in treating MK-801-induced schizophrenia in
#                rats." Exp Anim. 2018;67(2):163-73.
#                doi:10.1538/expanim.17-0091, PMID 29225304, open access
#                (PMC5955748, read in full). Male Sprague-Dawley rats
#                180-200 g, n = 7 per group, HPLC-MS/MS on PLASMA. The arm
#                used here is the haloperidol-alone control of the
#                intravenous limb: 0.7 mg/kg i.v., Table 2 -
#                AUC(0-inf) 188.4 +/- 20.5 ng*h/mL, t1/2 2.9 +/- 0.7 h,
#                CL 3.8 +/- 0.4 L/h/kg (mean +/- SD). No Cmax and no Tmax
#                are reported for the intravenous arm (the table prints
#                dashes), which is why haloperidol has no independent Cmax
#                comparison below.
#                Internal consistency, checked rather than assumed: Dose/AUC
#                = 0.7/0.1884 = 3.715 L/h/kg against the printed CL of 3.8,
#                a 2.2% gap of the usual mean-of-per-animal-ratios kind, so
#                the intravenous column hangs together. Its ORAL column does
#                NOT: the printed CL/F of 17.1 L/h/kg implies an oral AUC of
#                82 ng*h/mL against the 43.3 it prints, a 1.9-fold internal
#                contradiction. Nothing from the oral column is used as a
#                target here; only the paper's own IV-vs-oral bioavailability
#                (23.0 +/- 5.7%, which IS consistent with its clearances) is
#                quoted, and only as context.
#   [Rautiola2019] Rautiola D, Maglalang PD, Cheryala N, et al. "Intranasal
#                coadministration of a diazepam prodrug with a converting
#                enzyme results in rapid absorption of diazepam in rats."
#                J Pharmacol Exp Ther. 2019;370(3):796-805.
#                doi:10.1124/jpet.118.255943, PMID 30837282, open access
#                (PMC6806351, read in full). Male Sprague-Dawley rats,
#                8-9 weeks, 0.271 +/- 0.023 kg; the arm used here is the
#                intravenous reference: 1.00 mg/kg IV bolus, plasma by
#                LC-MS/MS, naive-pooled non-compartmental analysis in
#                WinNonlin, n = 6 at 2/5/10 min and n = 3 at 15-90 min.
#                Table 2, IV column - C0 (extrapolated to t = 0) 441 ng/mL,
#                t1/2,z 32.9 min, AUC(0-15 min) 1910 +/- 70 min*ng/mL,
#                AUC(0-90 min) 5430 +/- 190 min*ng/mL (+/- S.E.). Sampling
#                stops at 90 min and no AUC(0-inf) is reported, so the AUC
#                comparisons below are made over the source's own windows
#                rather than extrapolated - the model is asked the question
#                the study actually answered.
#                Why the SEMs are used unconverted: the sparse/leapfrog
#                design pools n = 6 and n = 3 animals across time points and
#                the SEs come from Nedelman-Jia sparse-sampling formulae, so
#                there is no single n to multiply by. Using SE as if it were
#                SD makes the interval ~2x narrower than it should be, i.e.
#                it makes the model's job harder, which is the safe
#                direction.
#
# `drug_pk.py` already carries [CanfieldSprague2025] (fentanyl rat SC) and
# [Almousa2011] (fentanyl rat IP) as recorded-but-unmodelled data; they are
# used here as independent targets and their citations are not repeated.


# ---------------------------------------------------------------------------
# Round-trip tier
# ---------------------------------------------------------------------------
#
# Body weights: every model here is parameterised per kg, so the weight
# cancels out of every concentration and every ratio below (proved by
# `test_pk.py::test_body_weight_cancels_for_per_kg_parameters...`). Where a
# comparison is against a specific study, that study's own mean weight is
# used anyway, so the absolute quantities the model reports (`dose_mg`,
# `CL_L_per_min`) describe the animals the target came from.

#: Weight of [Lei2018]'s rats (midpoint of the stated 180-200 g).
LEI2018_BODY_WEIGHT_KG = 0.19
#: Dose of [Lei2018]'s intravenous haloperidol arm, mg/kg.
LEI2018_IV_DOSE_MG_PER_KG = 0.7
#: [Lei2018] Table 2, intravenous haloperidol-alone column.
LEI2018_IV_AUC_NG_H_PER_ML = 188.4
LEI2018_IV_AUC_SD_NG_H_PER_ML = 20.5
LEI2018_IV_TERMINAL_HALF_LIFE_H = 2.9
LEI2018_IV_TERMINAL_HALF_LIFE_SD_H = 0.7
LEI2018_IV_CL_L_PER_H_PER_KG = 3.8
LEI2018_IV_CL_SD_L_PER_H_PER_KG = 0.4
LEI2018_N = 7
#: [Lei2018]'s own oral bioavailability, used as context only.
LEI2018_ORAL_F = 0.230
LEI2018_ORAL_CL_OVER_F_L_PER_H_PER_KG = 17.1

#: Weight of [Rautiola2019]'s rats, kg.
RAUTIOLA2019_BODY_WEIGHT_KG = 0.271
#: Dose of its intravenous reference arm, mg/kg.
RAUTIOLA2019_IV_DOSE_MG_PER_KG = 1.00
#: [Rautiola2019] Table 2, intravenous column.
RAUTIOLA2019_IV_C0_NG_PER_ML = 441.0
RAUTIOLA2019_IV_TERMINAL_HALF_LIFE_MIN = 32.9
RAUTIOLA2019_IV_AUC_0_15_NG_MIN_PER_ML = 1910.0
RAUTIOLA2019_IV_AUC_0_15_SE_NG_MIN_PER_ML = 70.0
RAUTIOLA2019_IV_AUC_0_90_NG_MIN_PER_ML = 5430.0
RAUTIOLA2019_IV_AUC_0_90_SE_NG_MIN_PER_ML = 190.0

#: n of [CanfieldSprague2025], needed to turn its SEMs into SDs.
CANFIELD2025_N = 6
#: n of [Almousa2011].
ALMOUSA2011_N = 3
#: SEMs printed by those two sources for the quantities used below.
CANFIELD2025_SC_AUC_SEM_NG_H_PER_ML = 10.82
CANFIELD2025_SC_HALF_LIFE_SEM_MIN = 0.01 * 60.0
ALMOUSA2011_IP_HALF_LIFE_SD_MIN = 44.7


def _mg_per_L_min_to_ng_h_per_mL(value):
    """mg/L * min -> ng/mL * h. 1 mg/L = 1000 ng/mL; 1 min = 1/60 h."""
    return value * 1000.0 / 60.0


def round_trip_results():
    """Every simulated-curve-versus-the-model's-own-analytic-value
    comparison, for all five models `drug_pk.py` builds.

    What this tier can catch: a unit slip, a broken tail extrapolation, an
    integration grid too coarse to resolve a distribution phase, an
    absorption model whose Tmax does not match the Bateman stationary point.
    What it cannot catch: any way in which the published parameters are
    themselves wrong about the rat. That is what the independent tier is for.
    """
    out = []

    def add(key, drug, route, quantity, units, predicted, observed, note=""):
        out.append(
            ValidationResult(
                key=key,
                drug=drug,
                species="rat",
                route=route,
                quantity=quantity,
                units=units,
                predicted=predicted,
                observed=observed,
                source="model's own closed-form value",
                tier=ROUND_TRIP,
                note=note,
            )
        )

    # --- haloperidol, rat, IV, two-compartment [ChengPaalzow1992] ----------
    model = drug_pk.haloperidol_rat_iv_pk(
        LEI2018_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, LEI2018_IV_DOSE_MG_PER_KG)
    add(
        "haloperidol/rat/IV/2C:t_half",
        "haloperidol",
        "IV",
        "terminal half-life",
        "min",
        m.terminal_half_life_min,
        model.terminal_half_life_min,
        "ln2/beta from the model's own hybrid rate constants; the tail fit "
        "has to recover it from a biexponential curve, which is the part "
        "worth testing",
    )
    add(
        "haloperidol/rat/IV/2C:auc",
        "haloperidol",
        "IV",
        "AUC(0-inf)",
        "mg/L*min",
        m.auc_0_inf,
        model.auc_0_inf(LEI2018_IV_DOSE_MG_PER_KG),
        "Dose/CL, exact for any linear model with central elimination",
    )
    add(
        "haloperidol/rat/IV/2C:cmax",
        "haloperidol",
        "IV",
        "Cmax",
        "mg/L",
        m.cmax,
        model.dose_equivalent_concentration(LEI2018_IV_DOSE_MG_PER_KG),
        "IV bolus peak is Dose/Vc by construction",
    )

    # --- haloperidol, rat, IV, one-compartment collapse --------------------
    model = drug_pk.haloperidol_rat_iv_one_compartment_pk(
        LEI2018_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, LEI2018_IV_DOSE_MG_PER_KG)
    add(
        "haloperidol/rat/IV/1C:t_half",
        "haloperidol",
        "IV",
        "terminal half-life",
        "min",
        m.terminal_half_life_min,
        model.terminal_half_life_min,
        "the collapse uses V_beta precisely so the terminal phase survives",
    )
    add(
        "haloperidol/rat/IV/1C:auc",
        "haloperidol",
        "IV",
        "AUC(0-inf)",
        "mg/L*min",
        m.auc_0_inf,
        model.auc_0_inf(LEI2018_IV_DOSE_MG_PER_KG),
    )

    # --- fentanyl, rat, IV, two-compartment [Yassen2008] -------------------
    weight = drug_pk._FENTANYL_RAT_IV_PROBE_BODY_WEIGHT_KG
    dose = 0.05  # mid of the source's own 0.015-0.085 mg/kg range
    model = drug_pk.fentanyl_rat_iv_pk(
        weight, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, dose)
    add(
        "fentanyl/rat/IV/2C:t_half",
        "fentanyl",
        "IV",
        "terminal half-life",
        "min",
        m.terminal_half_life_min,
        model.terminal_half_life_min,
        "the source publishes no half-life at all, so the model's own value "
        "is the only thing this can be checked against",
    )
    add(
        "fentanyl/rat/IV/2C:auc",
        "fentanyl",
        "IV",
        "AUC(0-inf)",
        "mg/L*min",
        m.auc_0_inf,
        model.auc_0_inf(dose),
    )
    add(
        "fentanyl/rat/IV/2C:cmax",
        "fentanyl",
        "IV",
        "Cmax",
        "mg/L",
        m.cmax,
        model.dose_equivalent_concentration(dose),
    )

    # --- fentanyl, rat, IV, one-compartment collapse -----------------------
    model = drug_pk.fentanyl_rat_iv_one_compartment_pk(
        weight, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, dose)
    add(
        "fentanyl/rat/IV/1C:t_half",
        "fentanyl",
        "IV",
        "terminal half-life",
        "min",
        m.terminal_half_life_min,
        model.terminal_half_life_min,
    )
    add(
        "fentanyl/rat/IV/1C:auc",
        "fentanyl",
        "IV",
        "AUC(0-inf)",
        "mg/L*min",
        m.auc_0_inf,
        model.auc_0_inf(dose),
    )

    # --- diazepam, rat, IV, one-compartment [Klotz1976] --------------------
    model = drug_pk.diazepam_rat_iv_pk(
        RAUTIOLA2019_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, RAUTIOLA2019_IV_DOSE_MG_PER_KG)
    add(
        "diazepam/rat/IV/1C:t_half",
        "diazepam",
        "IV",
        "terminal half-life",
        "min",
        m.terminal_half_life_min,
        model.terminal_half_life_min,
    )
    add(
        "diazepam/rat/IV/1C:auc",
        "diazepam",
        "IV",
        "AUC(0-inf)",
        "mg/L*min",
        m.auc_0_inf,
        model.auc_0_inf(RAUTIOLA2019_IV_DOSE_MG_PER_KG),
    )
    add(
        "diazepam/rat/IV/1C:cmax",
        "diazepam",
        "IV",
        "Cmax",
        "mg/L",
        m.cmax,
        model.dose_equivalent_concentration(RAUTIOLA2019_IV_DOSE_MG_PER_KG),
    )

    # --- diazepam, rat, IP, composite (the only absorption model) ----------
    dose_ip = drug_pk.DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG
    model = drug_pk.diazepam_rat_ip_pk(
        0.25, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, dose_ip)
    add(
        "diazepam/rat/IP:t_half",
        "diazepam",
        "IP",
        "terminal half-life",
        "min",
        m.terminal_half_life_min,
        LN2 / model.ke_per_min,
        "(CL/F)/(V/F) from [Friedman1986]",
    )
    add(
        "diazepam/rat/IP:tmax",
        "diazepam",
        "IP",
        "Tmax",
        "min",
        m.tmax_min,
        model.tmax_min(),
        "Bateman stationary point ln(ka/ke)/(ka-ke); the ONLY Tmax in this "
        "whole module that has anything to compare against, and what it is "
        "compared against is the model's own algebra",
    )
    add(
        "diazepam/rat/IP:auc",
        "diazepam",
        "IP",
        "AUC(0-inf)",
        "mg/L*min",
        m.auc_0_inf,
        model.auc_0_inf(dose_ip),
    )
    return out


def source_consistency_results():
    """Simulated curve versus a number the *parameterising* source printed but
    that was not fed into the model.

    This is a genuinely weaker tier than the independent one and is kept
    separate for that reason. Its comparisons cannot distinguish "the model is
    wrong about the rat" from "the paper's own printed numbers do not agree
    with each other at the precision they were printed to", and in both cases
    below it is demonstrably the second. Criteria here are therefore the
    intervals the sources' own rounding leaves open, not chosen tolerances.
    """
    out = []

    # --- haloperidol: V_beta went in, the printed 1.5 h did not ------------
    model = drug_pk.haloperidol_rat_iv_pk(
        LEI2018_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, LEI2018_IV_DOSE_MG_PER_KG)
    out.append(
        ValidationResult(
            key="haloperidol/rat/IV/2C vs ChengPaalzow1992:printed_t_half",
            drug="haloperidol",
            species="rat",
            route="IV",
            quantity="terminal half-life",
            units="min",
            predicted=m.terminal_half_life_min,
            observed=drug_pk.HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_MIN,
            interval=drug_pk.HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN,
            source="[ChengPaalzow1992]",
            tier=SOURCE_CONSISTENCY,
            note="the model anchors on the printed V_beta (11.1 L/kg, three "
            "significant figures) rather than on the printed '1.5 h' (two), "
            "for the reason drug_pk.py's comment gives. The simulated curve "
            "therefore has a 92.7 min terminal phase, which is +3.0% off the "
            "midpoint of '1.5 h' and inside the 1.45-1.55 h that rounding "
            "leaves open. Anchoring the other way would have moved the "
            "derived Q by 24%.",
        )
    )

    # --- diazepam IP: V/F and CL/F went in, the printed 0.88 h did not -----
    model = drug_pk.diazepam_rat_ip_pk(
        0.25, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, drug_pk.DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG)
    out.append(
        ValidationResult(
            key="diazepam/rat/IP vs Friedman1986:printed_t_half",
            drug="diazepam",
            species="rat",
            route="IP",
            quantity="terminal half-life",
            units="min",
            predicted=m.terminal_half_life_min,
            observed=drug_pk.DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN,
            # "0.88 h" printed to two decimal places rounds from 0.875-0.885 h.
            interval=(0.875 * 60.0, 0.885 * 60.0),
            source="[Friedman1986]",
            tier=SOURCE_CONSISTENCY,
            note="misses the printed-rounding interval by 0.07% at the low "
            "edge, i.e. -0.64% against the printed 52.8 min. That number is "
            "already pinned by drug_pk.diazepam_rat_ip_reported_half_life_"
            "consistency and its cause is known and is not the model: "
            "ln2/((CL/F)/(V/F)) from the printed 255 and 19.3 gives "
            "52.23-52.71 min across those two numbers' own rounding, and the "
            "printed 52.8 min lies just outside THAT. The source's three "
            "printed values are mutually inconsistent at the fourth "
            "significant figure; the model reproduces two of them exactly and "
            "therefore cannot reproduce the third.",
        )
    )

    # --- fentanyl: the source's own bootstrap replicate parameter set ------
    weight = drug_pk._FENTANYL_RAT_IV_PROBE_BODY_WEIGHT_KG
    cl, v1, v2, q = drug_pk.FENTANYL_RAT_IV_BOOTSTRAP
    bootstrap_model = DosedTwoCompartmentPK(
        route="IV",
        V1_L_per_kg=v1 / 1000.0 / weight,
        V2_L_per_kg=v2 / 1000.0 / weight,
        Q_mL_per_min_per_kg=q / weight,
        CL_mL_per_min_per_kg=cl / weight,
        body_weight_kg=weight,
    )
    m = metrics_for(bootstrap_model, 0.05)
    out.append(
        ValidationResult(
            key="fentanyl/rat/IV vs Yassen2008:bootstrap_t_half",
            drug="fentanyl",
            species="rat",
            route="IV",
            quantity="terminal half-life",
            units="min",
            predicted=m.terminal_half_life_min,
            observed=drug_pk.FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN,
            interval=drug_pk.fentanyl_rat_iv_terminal_half_life_interval_min(),
            source="[Yassen2008] 1,000-replicate bootstrap",
            tier=SOURCE_CONSISTENCY,
            note="[Yassen2008] publishes no half-life, so the only check "
            "available inside that paper is whether its bootstrap parameter "
            "set gives the same terminal phase as its point estimates. It "
            "does, to 2.4%, and both sit inside the 44.3-63.6 min the "
            "source's own confidence intervals leave open. What this does "
            "NOT show is that either is right for a rat.",
        )
    )
    return out


# ---------------------------------------------------------------------------
# Independent tier
# ---------------------------------------------------------------------------


def haloperidol_independent_results():
    """Simulated haloperidol curve versus [Lei2018], which contributed no
    parameter to it.

    Three comparisons, all from that paper's intravenous haloperidol-alone
    column at 0.7 mg/kg. Note that its AUC and its CL are not two independent
    facts (CL = Dose/AUC up to that paper's own 2.2% rounding), so they are
    reported as one finding measured twice, not as two corroborating hits.
    """
    model = drug_pk.haloperidol_rat_iv_pk(
        LEI2018_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, LEI2018_IV_DOSE_MG_PER_KG)
    matrix_note = (
        "[ChengPaalzow1992] reports BLOOD clearance and blood-based volumes "
        "while [Lei2018] measured PLASMA, and no rat blood:plasma ratio for "
        "haloperidol was found (drug_pk.SEARCHED_AND_EMPTY['haloperidol','rat',"
        "'blood:plasma']). An Rb of 0.76 would reconcile the two clearances "
        "exactly; that is a plausible value and an unmeasured one, so the gap "
        "is reported at full size rather than explained away."
    )
    return [
        ValidationResult(
            key="haloperidol/rat/IV vs Lei2018:auc",
            drug="haloperidol",
            species="rat",
            route="IV",
            quantity="AUC(0-inf)",
            units="ng*h/mL",
            predicted=_mg_per_L_min_to_ng_h_per_mL(m.auc_0_inf),
            observed=LEI2018_IV_AUC_NG_H_PER_ML,
            observed_dispersion=LEI2018_IV_AUC_SD_NG_H_PER_ML,
            dispersion_kind="SD",
            observed_n=LEI2018_N,
            source="[Lei2018]",
            tier=INDEPENDENT,
            note=matrix_note,
        ),
        ValidationResult(
            key="haloperidol/rat/IV vs Lei2018:t_half",
            drug="haloperidol",
            species="rat",
            route="IV",
            quantity="terminal half-life",
            units="min",
            predicted=m.terminal_half_life_min,
            observed=LEI2018_IV_TERMINAL_HALF_LIFE_H * 60.0,
            observed_dispersion=LEI2018_IV_TERMINAL_HALF_LIFE_SD_H * 60.0,
            dispersion_kind="SD",
            observed_n=LEI2018_N,
            source="[Lei2018]",
            tier=INDEPENDENT,
            note="a half-life is matrix-invariant if the blood:plasma ratio "
            "is time-independent, so unlike the AUC gap this one is NOT "
            "explainable by the blood-versus-plasma mismatch. [Lei2018] "
            "sampled to 24 h with HPLC-MS/MS; the 1992 source's sampling "
            "window and assay sensitivity are unknown (closed access), and a "
            "shorter or less sensitive design finding a shorter terminal "
            "phase is the standard explanation for a gap of this shape.",
        ),
        ValidationResult(
            key="haloperidol/rat/IV vs Lei2018:clearance",
            drug="haloperidol",
            species="rat",
            route="IV",
            quantity="clearance",
            units="L/h/kg",
            predicted=drug_pk.HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG * 60.0 / 1000.0,
            observed=LEI2018_IV_CL_L_PER_H_PER_KG,
            observed_dispersion=LEI2018_IV_CL_SD_L_PER_H_PER_KG,
            dispersion_kind="SD",
            observed_n=LEI2018_N,
            source="[Lei2018]",
            tier=INDEPENDENT,
            note="same finding as the AUC row, measured the other way up. "
            + matrix_note,
        ),
    ]


def diazepam_independent_results():
    """Simulated diazepam curve versus [Rautiola2019], which contributed no
    parameter to it.

    The two AUC comparisons are over the source's own windows (0-15 and 0-90
    min) rather than extrapolated to infinity, because that study stopped
    sampling at 90 min and reported no AUC(0-inf). Asking the model for
    AUC(0-inf) and comparing it to a 90-minute truncated observation would
    manufacture a discrepancy out of the study design.
    """
    model = drug_pk.diazepam_rat_iv_pk(
        RAUTIOLA2019_BODY_WEIGHT_KG, molar_mass_g_per_mol=None, brain_partition=None
    )
    m = metrics_for(model, RAUTIOLA2019_IV_DOSE_MG_PER_KG)
    return [
        ValidationResult(
            key="diazepam/rat/IV vs Rautiola2019:c0",
            drug="diazepam",
            species="rat",
            route="IV",
            quantity="C0 (Cmax of an IV bolus)",
            units="ng/mL",
            predicted=m.cmax * 1000.0,
            observed=RAUTIOLA2019_IV_C0_NG_PER_ML,
            source="[Rautiola2019]",
            tier=INDEPENDENT,
            note="this is the failure `diazepam_rat_iv_pk`'s docstring "
            "predicted and could not size: the model starts at Dose/V_beta "
            "because [Klotz1976] publishes no central volume, and V_beta > Vc "
            "always. The observed C0 now sizes it - Dose/C0 = 2.27 L/kg is an "
            "empirical Vc against the model's V_beta of 7.77 L/kg. That is a "
            "cross-study estimate, so it does not become a parameter here; it "
            "bounds the error.",
        ),
        ValidationResult(
            key="diazepam/rat/IV vs Rautiola2019:auc_0_15",
            drug="diazepam",
            species="rat",
            route="IV",
            quantity="AUC(0-15 min)",
            units="ng*min/mL",
            predicted=m.auc_to(15.0) * 1000.0,
            observed=RAUTIOLA2019_IV_AUC_0_15_NG_MIN_PER_ML,
            observed_dispersion=RAUTIOLA2019_IV_AUC_0_15_SE_NG_MIN_PER_ML,
            dispersion_kind="SEM",
            observed_n=None,
            source="[Rautiola2019]",
            tier=INDEPENDENT,
            note="the one independent comparison in this module that passes "
            "cleanly, and it passes for a reason worth stating: the missing "
            "distribution phase makes the model start 3.4-fold too low, and "
            "the too-long half-life makes it fall too slowly, and over the "
            "first 15 minutes those two errors very nearly cancel.",
        ),
        ValidationResult(
            key="diazepam/rat/IV vs Rautiola2019:auc_0_90",
            drug="diazepam",
            species="rat",
            route="IV",
            quantity="AUC(0-90 min)",
            units="ng*min/mL",
            predicted=m.auc_to(90.0) * 1000.0,
            observed=RAUTIOLA2019_IV_AUC_0_90_NG_MIN_PER_ML,
            observed_dispersion=RAUTIOLA2019_IV_AUC_0_90_SE_NG_MIN_PER_ML,
            dispersion_kind="SEM",
            observed_n=None,
            source="[Rautiola2019]",
            tier=INDEPENDENT,
            note="by 90 min the cancellation in the 0-15 min row has gone the "
            "other way and the too-slow decline dominates.",
        ),
        ValidationResult(
            key="diazepam/rat/IV vs Rautiola2019:t_half",
            drug="diazepam",
            species="rat",
            route="IV",
            quantity="terminal half-life",
            units="min",
            predicted=m.terminal_half_life_min,
            observed=RAUTIOLA2019_IV_TERMINAL_HALF_LIFE_MIN,
            source="[Rautiola2019]",
            tier=INDEPENDENT,
            note="[Rautiola2019] sampled only to 90 min, under three of its "
            "own half-lives, so its lambda_z may still contain distribution; "
            "[Klotz1976]'s 1.1 h is the longer-window number. Neither can be "
            "declared right from here, and the model is pinned to the second, "
            "so the disagreement is reported at full size.",
        ),
    ]


def fentanyl_independent_results():
    """Fentanyl has no independent rat INTRAVENOUS dataset in reach (see
    `SEARCHED_AND_EMPTY_VALIDATION`), so what is possible is cross-route: does
    the [Yassen2008] disposition survive contact with the two extravascular
    rat datasets `drug_pk.py` records but does not model?

    Two kinds of comparison:

      * **structural.** For any route, CL/F >= CL, because F <= 1. So
        `CL(Yassen, IV) / (CL/F)(extravascular study)` is an implied
        bioavailability, and an implied F above 1 is not a small
        disagreement, it is an impossibility: the studies cannot all be
        right. Both fentanyl extravascular studies produce one.
      * **half-life.** Terminal half-life is route-independent when
        absorption is faster than elimination, so the IV model's derived
        53.6 min can be held against the two extravascular half-lives, with
        the flip-flop caveat that if absorption were the slower process the
        observed values would be absorption-limited and would legitimately
        exceed it.
    """
    weight = drug_pk._FENTANYL_RAT_IV_PROBE_BODY_WEIGHT_KG
    model = drug_pk.fentanyl_rat_iv_pk(
        weight, molar_mass_g_per_mol=None, brain_partition=None
    )
    cl_iv_ml_per_min_per_kg = drug_pk.FENTANYL_RAT_IV_CL_ML_PER_MIN / weight

    cl_over_f_sc = (
        drug_pk.FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H
        / drug_pk.FENTANYL_RAT_SC_SOURCE_BODY_WEIGHT_KG
        / 60.0
    )
    ip_weight = sum(drug_pk.FENTANYL_RAT_IP_SOURCE_BODY_WEIGHT_RANGE_KG) / 2.0
    cl_over_f_ip = drug_pk.FENTANYL_RAT_IP_CL_OVER_F_ML_PER_MIN / ip_weight

    m = metrics_for(model, drug_pk.FENTANYL_RAT_SC_SOURCE_DOSE_MG_PER_KG)
    predicted_sc_auc = _mg_per_L_min_to_ng_h_per_mL(m.auc_0_inf)

    return [
        ValidationResult(
            key="fentanyl/rat/IV vs CanfieldSprague2025:implied_F",
            drug="fentanyl",
            species="rat",
            route="IV vs SC",
            quantity="implied subcutaneous bioavailability",
            units="fraction",
            predicted=cl_iv_ml_per_min_per_kg / cl_over_f_sc,
            observed=1.0,
            criterion_kind=UPPER_BOUND,
            source="[Yassen2008] / [CanfieldSprague2025]",
            tier=INDEPENDENT,
            note="'observed' here is the structural ceiling F = 1, not a "
            "measurement: any value above it is impossible. The two studies "
            "differ in strain (Wistar vs Sprague-Dawley), weight (237 vs "
            "327 g), dose (0.015-0.085 vs 0.300 mg/kg) and administration "
            "(20-40 min infusion vs bolus injection), and the dose gap is "
            "4-20x, so saturable elimination at the higher dose is the "
            "leading candidate. Whatever the cause, at least one of the two "
            "clearances is wrong for the other's animals.",
        ),
        ValidationResult(
            key="fentanyl/rat/IV vs Almousa2011:implied_F",
            drug="fentanyl",
            species="rat",
            route="IV vs IP",
            quantity="implied intraperitoneal bioavailability",
            units="fraction",
            predicted=cl_iv_ml_per_min_per_kg / cl_over_f_ip,
            observed=1.0,
            criterion_kind=UPPER_BOUND,
            source="[Yassen2008] / [Almousa2011]",
            tier=INDEPENDENT,
            note="and this is a LOWER bound on the implied F, because "
            "[Almousa2011]'s clearance is Dose/AUC(0-180), a truncated AUC, "
            "so the true CL/F is smaller and the implied F larger still. That "
            "source is independently known to contradict itself two-fold "
            "(drug_pk.fentanyl_rat_ip_auc_consistency), so it is the weaker "
            "of the two structural failures, not corroboration of the first.",
        ),
        ValidationResult(
            key="fentanyl/rat/IV vs CanfieldSprague2025:t_half",
            drug="fentanyl",
            species="rat",
            route="IV vs SC",
            quantity="terminal half-life",
            units="min",
            predicted=model.terminal_half_life_min,
            observed=drug_pk.FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN,
            observed_dispersion=CANFIELD2025_SC_HALF_LIFE_SEM_MIN,
            dispersion_kind="SEM",
            observed_n=CANFIELD2025_N,
            source="[CanfieldSprague2025]",
            tier=INDEPENDENT,
            note="the source's SEM of 0.01 h is so tight that the dispersion "
            "criterion is effectively 'agree to 1.5 min', which no "
            "cross-study comparison would survive; the two-fold criterion is "
            "the informative one here and it is met (0.81x).",
        ),
        ValidationResult(
            key="fentanyl/rat/IV vs Almousa2011:t_half",
            drug="fentanyl",
            species="rat",
            route="IV vs IP",
            quantity="terminal half-life",
            units="min",
            predicted=model.terminal_half_life_min,
            observed=drug_pk.FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN,
            observed_dispersion=ALMOUSA2011_IP_HALF_LIFE_SD_MIN,
            dispersion_kind="SD",
            observed_n=ALMOUSA2011_N,
            source="[Almousa2011]",
            tier=INDEPENDENT,
            note="n = 3, and that half-life was fitted on 90-180 min points "
            "at or below the assay's lowest calibrator. Recorded because it "
            "exists, weighted accordingly.",
        ),
        ValidationResult(
            key="fentanyl/rat/IV vs CanfieldSprague2025:auc",
            drug="fentanyl",
            species="rat",
            route="IV vs SC",
            quantity="AUC(0-inf) at 0.3 mg/kg",
            units="ng*h/mL",
            predicted=predicted_sc_auc,
            observed=drug_pk.FENTANYL_RAT_SC_AUC_NG_H_PER_ML,
            observed_dispersion=CANFIELD2025_SC_AUC_SEM_NG_H_PER_ML,
            dispersion_kind="SEM",
            observed_n=CANFIELD2025_N,
            source="[CanfieldSprague2025]",
            tier=INDEPENDENT,
            note="the model has no SC route, so this is its IV disposition "
            "asked what exposure 0.3 mg/kg would give if all of it reached "
            "the circulation - an UPPER bound on any SC prediction it could "
            "make, since F <= 1. The upper bound still lands 38% below the "
            "observed mean, which is the implied-F>1 finding restated as an "
            "exposure. It nevertheless passes both criteria, because the "
            "study's between-animal spread is wide; that the same disagreement "
            "reads as a pass one way round and an impossibility the other way "
            "round is why both rows are kept.",
        ),
    ]


def bioavailability_structural_results():
    """The same CL_iv/(CL/F) >= 1 structural test applied to all three drugs,
    so that fentanyl's failure has a control group.

    Diazepam and haloperidol both pass it, from data drawn across studies in
    exactly the same way, which is what makes it evidence about fentanyl's
    parameter set rather than about the test.
    """
    results = []
    results.append(
        ValidationResult(
            key="diazepam/rat: implied F(IP) <= 1",
            drug="diazepam",
            species="rat",
            route="IV vs IP",
            quantity="implied intraperitoneal bioavailability",
            units="fraction",
            predicted=drug_pk.DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG
            / drug_pk.DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG,
            observed=1.0,
            criterion_kind=UPPER_BOUND,
            source="[Klotz1976] / [Friedman1986]",
            tier=INDEPENDENT,
        )
    )
    results.append(
        ValidationResult(
            key="diazepam/rat: implied F(PO) <= 1",
            drug="diazepam",
            species="rat",
            route="IV vs PO",
            quantity="implied oral bioavailability",
            units="fraction",
            predicted=drug_pk.DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG
            / (drug_pk.DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG * 1000.0 / 60.0),
            observed=1.0,
            criterion_kind=UPPER_BOUND,
            source="[Klotz1976] / [Hua2025]",
            tier=INDEPENDENT,
        )
    )
    results.append(
        ValidationResult(
            key="haloperidol/rat: implied F(PO) <= 1",
            drug="haloperidol",
            species="rat",
            route="IV vs PO",
            quantity="implied oral bioavailability",
            units="fraction",
            predicted=drug_pk.HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG
            / (LEI2018_ORAL_CL_OVER_F_L_PER_H_PER_KG * 1000.0 / 60.0),
            observed=1.0,
            criterion_kind=UPPER_BOUND,
            source="[ChengPaalzow1992] / [Lei2018]",
            tier=INDEPENDENT,
            note="[Lei2018]'s own within-study answer is 23.0 +/- 5.7%, "
            "computed from its own IV and oral clearances. Substituting the "
            "1992 blood clearance for its plasma one moves that to 29%, i.e. "
            "the cross-study substitution costs about a quarter of the "
            "answer and does not change its sign or order of magnitude.",
        )
    )
    return results


def independent_results():
    return (
        haloperidol_independent_results()
        + diazepam_independent_results()
        + fentanyl_independent_results()
        + bioavailability_structural_results()
    )


def all_results():
    return round_trip_results() + source_consistency_results() + independent_results()


def failing_results(results=None):
    """Only the comparisons that miss their stated criterion. This is the
    list docs/architecture.md's P1.6 section is written from, and the list
    `tests/test_pk_validation.py` pins by key."""
    return [r for r in (all_results() if results is None else results) if not r.passed]


def validation_report(results=None):
    """A plain-text table of every comparison, both tiers, for the docs and
    for `python -m` style inspection."""
    results = all_results() if results is None else results
    lines = []
    for tier in (ROUND_TRIP, SOURCE_CONSISTENCY, INDEPENDENT):
        rows = [r for r in results if r.tier == tier]
        if not rows:
            continue
        lines.append("== %s tier (%d comparisons) ==" % (tier, len(rows)))
        lines.append(
            "%-58s %14s %14s %9s %8s %6s"
            % ("comparison", "predicted", "observed", "rel err", "fold", "")
        )
        for r in rows:
            lines.append(
                "%-58s %14.5g %14.5g %+8.2f%% %7.3fx %6s"
                % (
                    r.key,
                    r.predicted,
                    r.observed,
                    100.0 * r.relative_error,
                    r.fold_error,
                    "PASS" if r.passed else "FAIL",
                )
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# What could not be validated, and why
# ---------------------------------------------------------------------------

#: Tmax is one of the four quantities P1.6 was asked to validate and it is the
#: one that cannot be validated at all here. Stated as data so a test can
#: assert it is still on the record rather than quietly dropped.
TMAX_IS_NOT_VALIDATABLE = (
    "Tmax is not validated for any drug in this codebase, and the reason is "
    "structural rather than a gap in searching.\n"
    "Four of the five models are IV bolus models, whose Tmax is 0 by "
    "construction - there is no absorption phase to get right, so an observed "
    "Tmax would have nothing to compare against.\n"
    "The fifth, `diazepam_rat_ip_pk`, is the only absorption model in the "
    "project, and it predicts Tmax = 17.1 min. No rat intraperitoneal "
    "diazepam study found reports an observed Tmax: [Friedman1986], whose "
    "disposition the composite uses, is closed access and its abstract "
    "reports none, and its first post-dose sample is at 5 min, which brackets "
    "nothing. Searched (August 2026) beyond the sweep already recorded in "
    "drug_pk.SEARCHED_AND_EMPTY: Europe PMC for rat/mouse intraperitoneal "
    "diazepam studies reporting Tmax or a peak plasma time. The nearest hit, "
    "Nelson et al., 'Development of a novel dosing paradigm to model diazepam "
    "rescue therapy in preclinical seizure and epilepsy models' "
    "(PMC11296087, 2024, read in full), does dose male Sprague-Dawley rats "
    "diazepam intraperitoneally at 0.75-6 mg/kg and measure plasma by LC-MS, "
    "but it samples at 10 min after the LAST of three hourly injections and "
    "then at 1, 3 and 6 h, reports no non-compartmental parameters at all, "
    "and its design is a repeat-dose accumulation study - there is no single "
    "-dose ascending limb in it from which a Tmax could be read.\n"
    "Observed rat Tmax values DO exist in this project's records, but every "
    "one of them is for a route with no model, which is the same gap seen "
    "from the other side: fentanyl IP 28.3 min [Almousa2011], fentanyl SC "
    "0.5 h [CanfieldSprague2025] (and that one is the first sampling time, "
    "not a resolved peak), diazepam PO 1.33 h [Hua2025]. Each of those routes "
    "is unmodelled precisely because its ka was not measured, and ka is the "
    "parameter Tmax would test. Validating Tmax and building the absorption "
    "models are therefore the same missing piece, not two."
)

SEARCHED_AND_EMPTY_VALIDATION = {
    ("fentanyl", "rat", "IV"): (
        "No independent rat INTRAVENOUS fentanyl concentration-time dataset "
        "or non-compartmental parameter set could be obtained, so the "
        "[Yassen2008] model is validated only cross-route (against SC and IP "
        "studies), which confounds disposition with the unmeasured "
        "bioavailability of those routes.\n"
        "Searched (August 2026): PubMed and Europe PMC for rat intravenous "
        "fentanyl pharmacokinetics with clearance/half-life/AUC, and "
        "Unpaywall on every closed candidate.\n"
        "Candidates found and why each is unusable: Ohtsuka H, Fujita K, "
        "Kobayashi H, 'Pharmacokinetics of fentanyl in male and female rats "
        "after intravenous administration', Arzneimittelforschung "
        "2007;57(5):260-3 (PMID 17598696, doi 10.1055/s-0031-1296615) is "
        "exactly the right study - Wistar rats, 0.03 mg/kg IV, inside "
        "[Yassen2008]'s own 0.015-0.085 mg/kg range - but it is closed access "
        "(Unpaywall is_oa false, no repository copy) and its abstract prints "
        "no numerical parameter, only that the female volume of distribution "
        "was 1.34x the male one. THIS IS THE FIRST PAPER A FUTURE PASS WITH "
        "JOURNAL ACCESS SHOULD READ FOR THIS GAP. Yu C, Yuan M, Yang H, "
        "Zhuang X, Li H, 'P-glycoprotein on blood-brain barrier plays a vital "
        "role in fentanyl brain exposure and respiratory toxicity in rats', "
        "Toxicol Sci 2018;164(1):353-62 (PMID 29669042) dosed rats 40 and "
        "80 ug/kg intravenously with full plasma and brain profiles, which "
        "would serve, but the publisher returns 403 to both the article page "
        "and the PDF and there is no PMC deposit. Zhang et al., "
        "'QSAR-based physiologically based pharmacokinetic modeling for 34 "
        "fentanyl analogs', Front Pharmacol 2025 (PMC12575352, read in full) "
        "prints an 'observed' rat fentanyl parameter set in its Table 3 "
        "(1.75 ug IV to a 0.25 kg rat: CLsys 0.493 L/h, Vss 1.496 L, "
        "T1/2 1.60 h, AUC(0-inf) 3552 pg*h/mL, and those four are mutually "
        "consistent) that would disagree strikingly with [Yassen2008] - "
        "0.493 L/h is 8.2 mL/min against 17.0 - but the paper attributes it "
        "only to 'experimentally measured data reported in the literature' "
        "and names no source for the rat row. Quoting it would be citing a "
        "number whose origin nobody in this project has seen, which is the "
        "same objection that kept the Groningen haloperidol tables out of "
        "drug_pk.SEARCHED_AND_EMPTY['haloperidol','rat','SC']. Shao et al., "
        "Drug Metab Dispos 2026;54(7):100319 (PMID 42424960) has rat and "
        "mouse intravenous fentanyl PK as the anchor of a cross-species "
        "model and is closed access.\n"
        "Consequence, stated rather than hidden: fentanyl is the one drug of "
        "the three whose disposition model has never been checked against a "
        "second measurement of the same route in the same species, and its "
        "two cross-route checks both fail."
    ),
    ("haloperidol", "rat", "Cmax"): (
        "[Lei2018], the independent rat haloperidol dataset used here, prints "
        "a dash rather than a Cmax and a Tmax for its intravenous arm (both "
        "are meaningless for a bolus under naive NCA), so the haloperidol "
        "model's initial concentration - Dose/Vc, the quantity most sensitive "
        "to the central volume - has nothing to be validated against. It is "
        "also the quantity the one-compartment collapse gets most wrong "
        "(2-fold low by construction, per `haloperidol_rat_iv_one_"
        "compartment_pk`), so this is a real gap and not a formality."
    ),
    ("*", "mouse", "*"): (
        "No mouse PK model exists for any of the three drugs "
        "(drug_pk.SEARCHED_AND_EMPTY), so there is nothing to validate in the "
        "species the roadmap targets. Validation of the mouse layer is "
        "blocked behind the same two closed-access papers P1.2/P1.4 named: "
        "Kalvass et al. 2007 (JPET, mouse fentanyl) and Greenblatt & Sethy "
        "1990 (Psychopharmacology, mouse diazepam)."
    ),
}
