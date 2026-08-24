"""P7.2: a nonlinear decoder, adopted only where held-out data says so.

    from neurocascade import decoder

    for verdict in decoder.assess_all():
        print(verdict.report())

    readout = decoder.resolve_nonlinear("analgesia", network, ontology)

P7.1 built seven behavioural endpoints, each a *linear* readout
`s = sum_i w_i r_i` over regional Wilson-Cowan rates, and named three places
where that form is provably wrong. P7.2 is the task that decides which of
those three (and any others) to actually change, using held-out data and a
model-comparison criterion, and which to leave alone.

The whole module rests on one argument about what data can decide this, so
that argument is stated first. The result, so it is not buried: **the answer
is no everywhere.** When P7.2 ran, five graded single-region series existed
in this literature with per-level numbers outside a figure - three for
locomotion, one for sedation, one for anxiety - and all five were rejected,
each for a different reason; the other four endpoints had no such series at
all. P11.3 then retired the five rodent series to `legacy/rodent_series.py`
under the human-primary pivot, so the library this module assesses is now
empty (`SERIES == ()`) and every endpoint reports `no-data`. Nothing was
adopted before the retirement and nothing is adopted now; the decision rule
below is what will evaluate the first HUMAN graded series, and the synthetic
tests are what keep it executable until then.


What data can identify a decoder nonlinearity, and what cannot
--------------------------------------------------------------

The decoder is the map `D` from the rate vector to a behavioural score. To
compare a linear `D` against a nonlinear one you need observations of `D`,
which means observations that pair regional ACTIVITY with BEHAVIOUR. Drug
dose-response curves do not do that: a drug moves every region at once
through pharmacokinetics, receptor binding and the circuit, so a curved
dose-response curve is curved for reasons that live in four layers, only the
last of which is the decoder. Fitting a nonlinear decoder to a drug curve
would attribute all of that curvature to the readout. That is why P7.2 does
NOT use the dose-response data P7.3 will use, and it is not a scheduling
accident that P7.2 comes first.

The experiments that DO observe `D` are graded single-region manipulations:
set one region's drive to several levels - stimulation intensity, optogenetic
frequency, intracerebral dose - and measure the behaviour at each level. Call
the drive `u`, the region's rate `r(u)`, and the behaviour `b`. Under a
linear decoder, and holding the other regions fixed,

    b(u) = const + w * r(u)

so `b` is an affine image of `r`. The transduction `u -> r` is not known, but
one thing about it is: it is MONOTONE and SATURATING. More current, more
light, more agonist does not reverse the direction in which it pushes the
region, and a Wilson-Cowan unit's output is a bounded sigmoid of its input.
Monotone DECREASING counts: driving an inhibitory input to a region harder
lowers that region's rate monotonically, which is the case for
[DeLuca2022]'s orexin terminals in the VLPO below, and an affine map of a
monotone-decreasing function is still monotone. Two consequences follow, and
they are the two halves of this module's decision rule.

**A saturating `b(u)` is NOT evidence for a nonlinear decoder.** If `b(u)`
bends over as `u` grows, `r(u)` bending over explains it completely with a
linear `D`. A model comparison that adopted a saturating decoder here would
be attributing the sigmoid the circuit layer ALREADY HAS to the readout, and
would double-count it. This is the single most common shape in the graded
literature and it has its own rejection category,
`saturating-wins-not-identifiable`, rather than being quietly counted as a
win: the module records that the saturating family DID win out of sample and
that winning is not sufficient.

**A NON-MONOTONE `b(u)` IS evidence for a nonlinear decoder.** A monotone
`r(u)` composed with an affine `b = const + w r` is monotone, whatever `w`
is, and whichever way `r(u)` runs. So if behaviour rises and then falls (or falls and then
rises) as one region's drive is raised, no linear decoder over that region's
rate can produce it, and no choice of `w` repairs it. Non-monotonicity is the
identifiable signature, and it is the only one this module will adopt on.

That asymmetry is the reason the model families below are what they are: the
comparison is not "linear versus curved", which the transduction confounds,
it is "monotone versus non-monotone", which it does not.


What is compared, and on what criterion
---------------------------------------

Four families, fitted to one graded series each, over a drive coordinate `z`
rescaled to `[0, 1]` across the series' own levels:

    constant     a                          1 parameter
    linear       a + b z                    2
    satexp       a + c (1 - exp(-lam z))    3   monotone, saturating
    quadratic    a + b z + c z^2            3   the only non-monotone family

(`satexp`'s basis is normalised to span `[0, 1]` for every `lam`; see
`_satexp_basis` for why the textbook form is unusable inside leave-one-out.)

`satexp` and `quadratic` have the SAME parameter count over the SAME
predictor, so the head-to-head that decides the verdict is not confounded by
complexity. `linear` is nested in both and `constant` is the null that says
the series carries no signal at all.

The criterion is **leave-one-out cross-validation**: each drive level in turn
is held out, the family is refitted on the rest, and the held-out level is
predicted. `loo_rmse` is the root mean square of those held-out errors, in
the assay's own units. AICc is computed as a second, in-sample criterion and
reported wherever `n` allows it, but it does not gate the verdict, because at
four or five drive levels AICc for a 3-parameter family is either undefined
or resting on a correction term larger than the quantity it corrects.

Nothing here has a tuned threshold. The adoption rule is:

    ADOPT a nonlinearity for a series iff ALL of
      1. it has at least 4 drive levels (a 3-parameter family needs 3 points
         to be determined, so LOO needs a 4th to have anything to predict);
      2. `quadratic` has the strictly lowest `loo_rmse` of the four;
      3. `quadratic` beats `constant` on `loo_rmse` (the series has signal);
      4. the fitted turning point `z* = -b / (2c)` lies strictly inside
         `(0, 1)`, i.e. inside the range of drive levels actually tested;
      5. the MEASURED MEANS are themselves ordinally non-monotone - they go
         up and then down, or down and then up, with no fit involved;
      6. rules 2-5 hold under BOTH weightings of the levels (uniform, and
         `1/SEM^2` from the published dispersion).

Rules 4 and 5 are both there because rule 2 alone is not enough, and the
second of them was added after the first failed a test. A parabola's rising
limb approximates a saturating curve well, so `quadratic` beats `satexp` out
of sample on some strictly monotone series. Rule 4 catches the version of
that where the vertex lands past the end of the range. It does NOT catch the
version where the vertex lands just inside it: the six-level series
`0, 3, 5.2, 6.6, 7.4, 7.7` - strictly increasing and decelerating, the
canonical saturating shape - puts the vertex at `z* = 0.945` and passes rules
2 and 4. It was adopted, which is precisely the false positive this module
exists to prevent, and it is kept as `tests/test_decoder.py`'s
`test_a_saturating_series_whose_vertex_falls_just_inside_is_still_rejected`.

Rule 5 is the fix, and it is deliberately model-free: if the measured means
never actually turn round, then no reversal was measured, whatever a fitted
parabola's stationary point does. It reads the claim a second, independent
way rather than trusting one reading of it - the same discipline that
replaced P7.1's tautological sign test. It can only remove adoptions, never
add one, so it cannot manufacture a nonlinearity; the cost is that a real
reversal whose final level happens to land a hair the wrong side of its
predecessor is rejected, and rejection is the safe direction here.

Rule 6 exists because the weighting is a modelling choice with no right
answer, and on real data it changes which family wins. `1/SEM^2` is the
textbook choice and it is actively misleading on a series whose lowest
levels sit on a floor: in [Herent2023]'s CnF frequency series the 5 and 10 Hz
levels evoke no locomotion in any animal, so their SDs are 0.35 and 0.12 cm/s
against 20.9 cm/s at 15 Hz, and `1/SEM^2` gives those two floor levels about
four orders of magnitude more leverage than the level where the behaviour
appears. Uniform weighting discards genuine precision information instead.
Requiring both to agree means no adoption in this module rests on that
choice, and it adds no constant to tune. A series that adopts under one and
not the other is recorded as `weighting-dependent`, which is a rejection.

There is still no minimum effect size, no p-value and no margin by which
quadratic must beat the monotone families, because any such number would be
invented. Strict inequality on a held-out criterion, a structural constraint
on where the turning point sits, a model-free look at the means, and
agreement across the two weightings is the whole rule.


What adoption changes, and what stays illustrative
--------------------------------------------------

An adopted verdict says: this region's contribution to this endpoint is
NON-MONOTONE IN ITS OWN RATE, with the turning point somewhere inside the
range of drive the experiment covered. It does NOT say where the turning
point is IN RATE UNITS, and cannot, because the drive-to-rate transduction
that would convert microamps into a Wilson-Cowan rate was never measured in
these experiments and is not in this codebase. So `NonlinearTerm.peak_rate`
carries the same `ILLUSTRATIVE` label every P7.1 weight carries, set to the
network's resting rate 0.5 for want of anything measured, and P7.3 fits it.
The FORM is what the data buys; the LOCATION is not.

`NonlinearReadout` therefore scores

    s = sum_{i linear} w_i r_i + sum_{j nonlinear} w_j f_j(r_j)

with `f_j` a U or inverted-U basis mapping `[0, 1] -> [0, 1]` exactly as
`r` does, so the two sums are on one scale and P7.1's normalisation
(`sum |w| == 1`) is untouched. An endpoint with no adopted term resolves to
a plain `behavior.BehaviorReadout` and is bit-for-bit what P7.1 produced; a
test asserts that by `type(...) is`.
"""

import math

import numpy as np

from . import behavior
from . import endpoints as endpoints_module


ILLUSTRATIVE = "ILLUSTRATIVE (not calibrated; P7.3 fits it)"

#: How a series' drive levels are mapped onto the fitting coordinate before
#: rescaling to [0, 1]. Recorded per series with a reason, because it is a
#: modelling choice and not a property of the experiment.
DRIVE_TRANSFORMS = ("linear", "log10")

#: What was varied. Only needed so the report can say it; nothing branches
#: on it, because the argument above is about monotonicity of drive -> rate
#: and holds for all three.
MANIPULATIONS = (
    "electrical_stimulation",
    "optogenetic_stimulation",
    "intracerebral_microinjection",
)

VERDICTS = (
    "adopt-nonlinear",
    "keep-linear",
)

REJECTION_REASONS = {
    "too-few-levels":
        "fewer than 4 graded levels, so a 3-parameter family cannot be "
        "cross-validated against a 2-parameter one at all",
    "no-signal":
        "no family beat the constant model out of sample; the series does "
        "not constrain the decoder in either direction",
    "linear-wins":
        "the straight line predicted the held-out levels better than "
        "anything curved. The decoder P7.1 already has is the best model "
        "of this series",
    "saturating-wins-not-identifiable":
        "the monotone SATURATING family predicted the held-out levels best. "
        "The series really does bend over - but a linear decoder reading a "
        "region whose rate saturates in its own drive produces exactly that "
        "curve, so the bend is attributable to the Wilson-Cowan sigmoid the "
        "circuit layer already has. Adopting a saturating decoder here would "
        "count that sigmoid twice",
    "turning-point-outside-range":
        "the quadratic won out of sample but its turning point lies outside "
        "the range of drive tested, so it is a curved monotone segment "
        "rather than a reversal",
    "no-measured-reversal":
        "the quadratic won out of sample with its turning point nominally "
        "inside the range, but the measured means never turn round. A "
        "stationary point of a fitted parabola is not a measured reversal, "
        "and a saturating series puts one near the top of its range as a "
        "matter of course",
    "weighting-dependent":
        "the quadratic won under one weighting of the levels and not under "
        "the other. A reversal that depends on whether the published SEMs "
        "are used is not a reversal this module will adopt",
    "no-data":
        "no graded single-region series with per-level numbers outside a "
        "figure was found for this endpoint",
}


# --------------------------------------------------------------------------
# The observations
# --------------------------------------------------------------------------

class DriveResponseSeries(object):
    """One graded single-region manipulation, with behaviour at each level.

    key: stable identifier.
    endpoint: the `endpoints.ENDPOINTS` key this series speaks to.
    structures: CCF acronyms the manipulation targeted, in the same
        vocabulary P7.1 uses, so a verdict can be attached to a term.
    manipulation: one of `MANIPULATIONS`.
    drive_units: the units of `levels` ("uA", "Hz", "nmol", ...).
    levels: the drive levels, strictly increasing. A vehicle / sham level is
        included as 0.0 when the paper reports one, because for a reversal
        the control point is often the only thing that makes the series
        non-monotone at all.
    values: the behavioural mean at each level, in `value_units`, in the
        direction the PAPER reports it.
    score_sign: +1 if a higher `value` means a higher score on this
        endpoint's own `direction`, -1 if it means a lower one. The polarity
        conversion happens exactly once, here, in the open - the same
        discipline `endpoints.EndpointTerm` uses for `activity`/`measure`.
    dispersion / dispersion_kind: per-level SEM or SD, or None.
    n_per_level: group sizes, or None.
    drive_transform: one of `DRIVE_TRANSFORMS`, with `transform_reason`
        saying why. "log10" is illegal when a level is 0.
    provenance: where the per-level numbers came from - "table", "text" or
        "source data". A series whose numbers exist only in a figure is not
        constructible, by design: there is no honest way to read one here.
    """

    def __init__(self, key, endpoint, structures, manipulation, drive_units,
                 levels, values, value_units, score_sign, citation, species,
                 finding, provenance, drive_transform="linear",
                 transform_reason=None, dispersion=None, dispersion_kind=None,
                 n_per_level=None, note=None):
        if isinstance(structures, str):
            structures = (structures,)
        levels = tuple(float(x) for x in levels)
        values = tuple(float(x) for x in values)
        if len(levels) != len(values):
            raise ValueError(
                "%r has %d levels and %d values" % (key, len(levels),
                                                    len(values))
            )
        if len(levels) < 3:
            raise ValueError(
                "%r has %d levels; a series with fewer than 3 is a "
                "comparison, not a graded manipulation" % (key, len(levels))
            )
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("%r levels must be strictly increasing" % (key,))
        if manipulation not in MANIPULATIONS:
            raise ValueError(
                "manipulation must be one of %s, got %r"
                % (list(MANIPULATIONS), manipulation)
            )
        if score_sign not in (1, -1):
            raise ValueError(
                "%r: score_sign must be +1 or -1, spelled out, so the "
                "polarity conversion is written down once" % (key,)
            )
        if drive_transform not in DRIVE_TRANSFORMS:
            raise ValueError(
                "drive_transform must be one of %s, got %r"
                % (list(DRIVE_TRANSFORMS), drive_transform)
            )
        if drive_transform == "log10" and min(levels) <= 0.0:
            raise ValueError(
                "%r: a log10 drive transform cannot represent a zero "
                "(vehicle / sham) level" % (key,)
            )
        if drive_transform != "linear" and not transform_reason:
            raise ValueError(
                "%r: a non-default drive transform is a modelling choice and "
                "needs its reason recorded" % (key,)
            )
        if provenance not in ("table", "text", "source data"):
            raise ValueError(
                "%r: provenance must say where the per-level numbers came "
                "from; a figure is not a source this module can read"
                % (key,)
            )
        if dispersion is not None:
            dispersion = tuple(float(x) for x in dispersion)
            if len(dispersion) != len(levels):
                raise ValueError("%r: dispersion length mismatch" % (key,))
            if dispersion_kind not in ("SEM", "SD"):
                raise ValueError(
                    "%r: dispersion needs its kind named" % (key,)
                )
        if n_per_level is not None:
            n_per_level = tuple(int(x) for x in n_per_level)
            if len(n_per_level) != len(levels):
                raise ValueError("%r: n_per_level length mismatch" % (key,))
        if not finding:
            raise ValueError(
                "%r: the finding has to be written out so the shape can be "
                "checked against the paper rather than trusted" % (key,)
            )

        self.key = key
        self.endpoint = endpoint
        self.structures = tuple(structures)
        self.manipulation = manipulation
        self.drive_units = drive_units
        self.levels = levels
        self.values = values
        self.value_units = value_units
        self.score_sign = score_sign
        self.citation = citation
        self.species = species
        self.finding = finding
        self.provenance = provenance
        self.drive_transform = drive_transform
        self.transform_reason = transform_reason
        self.dispersion = dispersion
        self.dispersion_kind = dispersion_kind
        self.n_per_level = n_per_level
        self.note = note

    @property
    def n_levels(self):
        return len(self.levels)

    def scores(self):
        """`values` converted into this endpoint's score direction."""
        return np.array(self.values, dtype=float) * self.score_sign

    def drive_coordinate(self):
        """The levels as the fitting coordinate `z`, rescaled to [0, 1]."""
        u = np.array(self.levels, dtype=float)
        if self.drive_transform == "log10":
            u = np.log10(u)
        span = u[-1] - u[0]
        if span <= 0:
            raise ValueError("%r: degenerate drive range" % (self.key,))
        return (u - u[0]) / span

    def z_to_level(self, z):
        """Invert `drive_coordinate`, so a turning point can be reported."""
        u = np.array(self.levels, dtype=float)
        if self.drive_transform == "log10":
            u = np.log10(u)
        raw = u[0] + z * (u[-1] - u[0])
        if self.drive_transform == "log10":
            return 10.0 ** raw
        return raw

    def weights(self, kind="uniform"):
        """Fitting weights under one of the two schemes, and their label.

        `kind="uniform"` weights every level equally. `kind="dispersion"`
        weights by `1/SEM^2`, converting a published SD with the reported
        group size and falling back to uniform when there is no usable
        dispersion rather than guessing one.

        NEITHER is the right answer on its own, which is why `assess_series`
        runs both and adopts only on agreement. `1/SEM^2` is the textbook
        choice and it is actively misleading on some of these series: an
        optogenetic frequency below the threshold for locomotion produces
        NO locomotion in every animal, so its SD is near zero because of a
        floor, not because the level was measured precisely. On
        [Herent2023]'s CnF series the 5 and 10 Hz levels carry SDs of 0.35
        and 0.12 cm/s against 20.9 at 15 Hz, so `1/SEM^2` hands the two
        floor levels about four orders of magnitude more leverage than the
        level where the behaviour actually appears, and every family is then
        fitted to predict zero. Uniform weighting throws away real precision
        information in exchange for not doing that.
        """
        if kind == "uniform":
            return np.ones(self.n_levels), "uniform"
        if kind != "dispersion":
            raise ValueError("weighting must be 'uniform' or 'dispersion'")
        if self.dispersion is None:
            return np.ones(self.n_levels), "uniform (no dispersion published)"
        d = np.array(self.dispersion, dtype=float)
        if self.dispersion_kind == "SD":
            if self.n_per_level is None:
                return (np.ones(self.n_levels),
                        "uniform (SD published but no group size to convert "
                        "it to a SEM)")
            d = d / np.sqrt(np.array(self.n_per_level, dtype=float))
        if np.any(d <= 0):
            return np.ones(self.n_levels), "uniform (a published SEM is zero)"
        return 1.0 / d ** 2, "1/SEM^2"

    def observed_shape(self):
        """The model-free ordinal shape of the score sequence.

        Returns "increasing", "decreasing", "flat" or "non-monotone". This
        is rule 5 of the adoption rule and it is deliberately model-free:
        no fit, no dispersion, no allowance for noise, just whether the
        published means turn round. It is a NECESSARY condition and never a
        sufficient one - noise alone flips the ordinal shape easily, but a
        noise wiggle does not also win leave-one-out against two monotone
        families - so it can only remove adoptions.
        """
        s = self.scores()
        ups = sum(1 for a, b in zip(s, s[1:]) if b > a)
        downs = sum(1 for a, b in zip(s, s[1:]) if b < a)
        if ups and downs:
            return "non-monotone"
        if ups:
            return "increasing"
        if downs:
            return "decreasing"
        return "flat"

    def __repr__(self):
        return "DriveResponseSeries(%r, %s, %d levels)" % (
            self.key, self.endpoint, self.n_levels
        )


# --------------------------------------------------------------------------
# The model families
# --------------------------------------------------------------------------

#: Singular values below this fraction of the largest are treated as zero
#: when solving a weighted least squares problem. `rcond=None` (machine
#: epsilon) is wrong here for a specific reason worth recording: inside
#: leave-one-out, `satexp`'s two columns become numerically collinear
#: whenever the held-out level is the only one in the basis's transition
#: region, and a machine-epsilon cutoff then returns two enormous
#: coefficients that cancel on the training points and do not cancel on the
#: held-out one. That produced a leave-one-out error of ~2e6 on a flat
#: six-level series - a number with no meaning, decided by how the LAPACK
#: driver broke a tie. Truncating at 1e-10 gives the minimum-norm solution
#: in the numerically supported subspace instead, which predicts the mean
#: and is punished normally. This is a numerical-linear-algebra choice, not
#: a modelling one: it changes no full-rank fit anywhere in the module.
_RCOND = 1e-10


def _wls(design, y, w):
    """Weighted least squares. Returns (coefficients, fitted values)."""
    sw = np.sqrt(w)
    a = design * sw[:, None]
    b = y * sw
    coef, _, _, _ = np.linalg.lstsq(a, b, rcond=_RCOND)
    return coef, design.dot(coef)


class _Family(object):
    """A model family: a name, a parameter count, and a weighted fit."""

    def __init__(self, name, n_params, monotone, fit):
        self.name = name
        self.n_params = n_params
        self.monotone = monotone
        self._fit = fit

    def fit(self, z, y, w):
        """Returns (predict(z_new) -> array, params dict)."""
        return self._fit(z, y, w)


def _fit_constant(z, y, w):
    coef, _ = _wls(np.ones((len(z), 1)), y, w)
    a = float(coef[0])
    return (lambda zz: np.full(np.shape(zz), a)), {"a": a}


def _fit_linear(z, y, w):
    design = np.column_stack([np.ones_like(z), z])
    coef, _ = _wls(design, y, w)
    a, b = float(coef[0]), float(coef[1])
    return (lambda zz: a + b * np.asarray(zz, dtype=float)), {"a": a, "b": b}


def _fit_quadratic(z, y, w):
    design = np.column_stack([np.ones_like(z), z, z ** 2])
    coef, _ = _wls(design, y, w)
    a, b, c = (float(coef[0]), float(coef[1]), float(coef[2]))

    def predict(zz):
        zz = np.asarray(zz, dtype=float)
        return a + b * zz + c * zz ** 2

    return predict, {"a": a, "b": b, "c": c}


#: The grid `satexp`'s rate constant is profiled over. Log-spaced and wide:
#: `lam -> 0` degenerates to the linear family and `lam` large to a step, so
#: the grid brackets everything between. Refined by golden section, so the
#: grid's coarseness does not enter the answer.
_LAM_GRID = np.logspace(-2, 2, 41)


def _satexp_basis(lam, z):
    """`(1 - exp(-lam z)) / (1 - exp(-lam))`: 0 at z=0, 1 at z=1, any lam>0.

    The unnormalised `1 - exp(-lam z)` is what the family is usually
    written as, and it is badly conditioned here: as `lam -> 0` the column
    shrinks like `lam z`, so the coefficient it multiplies grows like
    `1/lam` and a leave-one-out refit can return a numerically meaningless
    prediction. Dividing by the value at the top of the range fixes the
    basis to span `[0, 1]` for every `lam`, which makes the `lam -> 0`
    limit exactly the linear family with the same coefficient rather than a
    ratio of two vanishing quantities. It changes no fit, only the
    parameterisation.
    """
    return np.expm1(-lam * z) / np.expm1(-lam)


def _satexp_rss(lam, z, y, w):
    design = np.column_stack([np.ones_like(z), _satexp_basis(lam, z)])
    coef, fitted = _wls(design, y, w)
    return float(np.sum(w * (y - fitted) ** 2)), coef


def _resolvable_lam_max(z):
    """The largest `lam` these drive levels can tell apart from a step.

    A saturating exponential that reaches 99% of its range inside ONE
    sampling interval is a step function as far as this design is
    concerned: its two coefficients are collinear on the fit points and
    only their sum is determined, so the fit is decided by how the solver
    breaks a tie rather than by the data. That is not a hypothetical - it
    is what produced a leave-one-out error of ~2e6 on a flat six-level test
    series before this bound existed, and an inflated `satexp` error is
    dangerous in exactly one direction: it can hand the comparison to
    `quadratic` and cause a false adoption.

    The bound is read off the design rather than chosen: `lam_max =
    ln(100) / dz_min`, where `dz_min` is the smallest gap between
    consecutive drive coordinates actually being fitted. Below it the
    family is identifiable; above it it is not a member of the family this
    comparison is about.
    """
    gaps = np.diff(np.sort(np.asarray(z, dtype=float)))
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return float(_LAM_GRID[-1])
    return float(math.log(100.0) / gaps.min())


def _fit_satexp(z, y, w):
    lam_max = _resolvable_lam_max(z)
    grid = _LAM_GRID[_LAM_GRID <= lam_max]
    if len(grid) == 0:
        grid = np.array([min(float(_LAM_GRID[0]), lam_max)])
    best = None
    for lam in grid:
        rss, coef = _satexp_rss(lam, z, y, w)
        if best is None or rss < best[0]:
            best = (rss, lam, coef)
    lam0 = best[1]
    # Golden-section refine inside the bracketing grid cell.
    lo = lam0 / 10.0 ** 0.1
    hi = min(lam0 * 10.0 ** 0.1, lam_max)
    if hi <= lo:
        lo, hi = lam0, lam0
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        m1 = hi - phi * (hi - lo)
        m2 = lo + phi * (hi - lo)
        if _satexp_rss(m1, z, y, w)[0] < _satexp_rss(m2, z, y, w)[0]:
            hi = m2
        else:
            lo = m1
    lam = 0.5 * (lo + hi)
    rss, coef = _satexp_rss(lam, z, y, w)
    if rss > best[0]:
        lam, coef = best[1], best[2]
    a, c = float(coef[0]), float(coef[1])

    def predict(zz):
        zz = np.asarray(zz, dtype=float)
        return a + c * _satexp_basis(lam, zz)

    return predict, {"a": a, "c": c, "lam": float(lam)}


FAMILIES = (
    _Family("constant", 1, True, _fit_constant),
    _Family("linear", 2, True, _fit_linear),
    _Family("satexp", 3, True, _fit_satexp),
    _Family("quadratic", 3, False, _fit_quadratic),
)

FAMILIES_BY_NAME = {f.name: f for f in FAMILIES}

#: The head-to-head the verdict turns on: same parameter count, same
#: predictor, one monotone by construction and one not.
DISCRIMINATING_PAIR = ("satexp", "quadratic")


class FamilyFit(object):
    """One family's fit and its cross-validated predictive error."""

    def __init__(self, family, params, in_sample_rmse, loo_rmse,
                 loo_weighted, aicc, loo_predictions, weighting="uniform"):
        self.weighting = weighting
        self.family = family
        self.params = params
        self.in_sample_rmse = in_sample_rmse
        self.loo_rmse = loo_rmse
        self.loo_weighted = loo_weighted
        self.aicc = aicc
        self.loo_predictions = loo_predictions

    @property
    def name(self):
        return self.family.name

    @property
    def eligible(self):
        return self.loo_rmse is not None

    def __repr__(self):
        return "FamilyFit(%r, loo_rmse=%s)" % (
            self.name,
            "n/a" if self.loo_rmse is None else "%.4g" % self.loo_rmse,
        )


def fit_family(series, family, weighting="uniform"):
    """Fit one family to one series and cross-validate it leave-one-out."""
    z = series.drive_coordinate()
    y = series.scores()
    w, _ = series.weights(weighting)
    n = len(z)

    predict, params = family.fit(z, y, w)
    resid = y - predict(z)
    in_sample_rmse = float(np.sqrt(np.mean(resid ** 2)))

    # AICc, in-sample, with sigma^2 counted as a parameter. Undefined when
    # the correction term's denominator is not positive, which at these
    # sample sizes happens often and is reported rather than papered over.
    k = family.n_params + 1
    rss = float(np.sum(resid ** 2))
    if n - k - 1 > 0 and rss > 0:
        aic = n * math.log(rss / n) + 2 * k
        aicc = aic + 2.0 * k * (k + 1) / (n - k - 1)
    else:
        aicc = None

    # LOO needs the training fit to be determined: n - 1 points must be at
    # least the parameter count.
    if n - 1 < family.n_params:
        return FamilyFit(family, params, in_sample_rmse, None, None, aicc,
                         (), weighting)

    preds = []
    for i in range(n):
        keep = np.array([j for j in range(n) if j != i])
        p, _ = family.fit(z[keep], y[keep], w[keep])
        preds.append(float(np.asarray(p(z[i:i + 1]))[0]))
    preds = np.array(preds)
    errors = y - preds
    loo_rmse = float(np.sqrt(np.mean(errors ** 2)))
    loo_weighted = float(np.sqrt(np.sum(w * errors ** 2) / np.sum(w)))
    return FamilyFit(family, params, in_sample_rmse, loo_rmse, loo_weighted,
                     aicc, tuple(preds), weighting)


#: The two weightings every series is assessed under. A verdict of
#: `adopt-nonlinear` requires BOTH to adopt; see `assess_series`.
WEIGHTINGS = ("uniform", "dispersion")


class WeightedAssessment(object):
    """One series' four-family comparison under ONE weighting of the levels."""

    def __init__(self, weighting, label, fits, verdict, reason,
                 turning_point_z, turning_point_level, shape):
        self.weighting = weighting
        self.label = label
        self.fits = tuple(fits)
        self.fits_by_name = {f.name: f for f in fits}
        self.verdict = verdict
        self.reason = reason
        self.turning_point_z = turning_point_z
        self.turning_point_level = turning_point_level
        self.shape = shape

    @property
    def adopted(self):
        return self.verdict == "adopt-nonlinear"

    @property
    def best(self):
        eligible = [f for f in self.fits if f.eligible]
        if not eligible:
            return None
        return min(eligible, key=lambda f: f.loo_rmse)

    def table(self):
        lines = [
            "  weighting %-11s (%s)" % (self.weighting, self.label),
            "    %-10s %7s %9s %9s %9s"
            % ("family", "params", "in-RMSE", "LOO-RMSE", "AICc"),
        ]
        for f in self.fits:
            lines.append(
                "    %-10s %7d %9.4g %9s %9s"
                % (f.name, f.family.n_params, f.in_sample_rmse,
                   "n/a" if f.loo_rmse is None else "%.4g" % f.loo_rmse,
                   "n/a" if f.aicc is None else "%.4g" % f.aicc)
            )
        best = self.best
        lines.append(
            "    best out of sample: %s   ->  %s"
            % ("none" if best is None else best.name,
               self.verdict if self.adopted else self.reason)
        )
        if self.turning_point_z is not None:
            lines.append(
                "    quadratic turning point: z=%.3f%s"
                % (self.turning_point_z,
                   "" if 0.0 < self.turning_point_z < 1.0
                   else "  (OUTSIDE the tested range)")
            )
        return lines

    def __repr__(self):
        return "WeightedAssessment(%r, %s)" % (self.weighting, self.verdict)


class SeriesVerdict(object):
    """The comparison for one series under both weightings, and the outcome."""

    def __init__(self, series, assessments, verdict, reason, shape,
                 turning_point_z, turning_point_level):
        self.series = series
        self.assessments = tuple(assessments)
        self.by_weighting = {a.weighting: a for a in assessments}
        self.verdict = verdict
        self.reason = reason
        self.shape = shape
        self.turning_point_z = turning_point_z
        self.turning_point_level = turning_point_level

    @property
    def adopted(self):
        return self.verdict == "adopt-nonlinear"

    @property
    def primary(self):
        """The uniform-weighted assessment, which is the one reported first."""
        return self.by_weighting["uniform"]

    @property
    def fits(self):
        return self.primary.fits

    @property
    def fits_by_name(self):
        return self.primary.fits_by_name

    @property
    def best(self):
        return self.primary.best

    def report(self):
        s = self.series
        lines = [
            "%s  [%s]  %s" % (s.key, s.endpoint, "+".join(s.structures)),
            "  %s, %s, %d levels of %s (%s), readout %s"
            % (s.species, s.manipulation, s.n_levels, s.drive_units,
               s.provenance, s.value_units),
            "  citation:      %s" % s.citation,
            "  drive coord:   %s%s"
            % (s.drive_transform,
               "" if not s.transform_reason else " - " + s.transform_reason),
            "  ordinal shape: %s (model-free, means only)"
            % s.observed_shape(),
        ]
        for assessment in self.assessments:
            lines.extend(assessment.table())
        if self.turning_point_level is not None:
            lines.append(
                "  quadratic turning point (uniform): %.4g %s"
                % (self.turning_point_level, s.drive_units)
            )
        lines.append("  VERDICT: %s" % self.verdict)
        lines.append("    %s" % self.reason)
        if self.shape:
            lines.append("    shape: %s in the region's own rate" % self.shape)
        if s.note:
            lines.append("  note: %s" % s.note)
        return "\n".join(lines)

    def __repr__(self):
        return "SeriesVerdict(%r, %s)" % (self.series.key, self.verdict)


def _assess_under(series, weighting):
    """The four-family comparison and rules 1-5, under one weighting.

    Rule 3 (`quadratic` must beat `constant`) is not a separate branch: it
    is implied by `quadratic` being the argmin over all four families, which
    include `constant`. It is listed separately in the module docstring
    because it is a separate claim, and it is checked here by construction.
    Rule 6 is the combination of two of these and lives in `assess_series`.
    """
    fits = [fit_family(series, f, weighting) for f in FAMILIES]
    label = series.weights(weighting)[1]
    by_name = {f.name: f for f in fits}
    quad = by_name["quadratic"]

    turning_z = None
    turning_level = None
    shape = None
    if "b" in quad.params and quad.params.get("c"):
        c = quad.params["c"]
        turning_z = -quad.params["b"] / (2.0 * c)
        turning_level = float(series.z_to_level(turning_z))
        shape = "inverted_U" if c < 0 else "U"

    def out(verdict, reason, keep_shape=False):
        return WeightedAssessment(
            weighting, label, fits, verdict, reason, turning_z, turning_level,
            shape if keep_shape else None,
        )

    if series.n_levels < 4:
        return out("keep-linear", "too-few-levels")

    eligible = [f for f in fits if f.eligible]
    best = min(eligible, key=lambda f: f.loo_rmse)

    if best.name == "constant":
        return out("keep-linear", "no-signal")
    if best.name == "linear":
        return out("keep-linear", "linear-wins")
    if best.name == "satexp":
        return out("keep-linear", "saturating-wins-not-identifiable")
    if turning_z is None or not (0.0 < turning_z < 1.0):
        return out("keep-linear", "turning-point-outside-range")
    if series.observed_shape() != "non-monotone":
        return out("keep-linear", "no-measured-reversal")
    return out("adopt-nonlinear", "adopted", keep_shape=True)


ADOPTED_REASON = (
    "under BOTH weightings the quadratic predicted the held-out levels "
    "better than either monotone family, its turning point fell inside the "
    "tested range, and the measured means themselves reverse. No monotone "
    "drive-to-rate map composed with a linear weight can produce that"
)


def assess_series(series):
    """Run the comparison under both weightings and combine the two verdicts.

    A nonlinearity is adopted only if BOTH weightings adopt it. The two are
    not a formality: `1/SEM^2` is the textbook choice and it is actively
    wrong on a series whose sub-threshold levels have a floor-effect SD near
    zero, while uniform weighting discards real precision. Requiring
    agreement means no adoption rests on that choice, and it introduces no
    tuned constant.
    """
    assessments = [_assess_under(series, w) for w in WEIGHTINGS]
    primary = assessments[0]
    adopting = [a for a in assessments if a.adopted]
    if len(adopting) == len(assessments):
        return SeriesVerdict(series, assessments, "adopt-nonlinear",
                             ADOPTED_REASON, primary.shape,
                             primary.turning_point_z,
                             primary.turning_point_level)
    if adopting:
        return SeriesVerdict(series, assessments, "keep-linear",
                             "weighting-dependent", None,
                             primary.turning_point_z,
                             primary.turning_point_level)
    return SeriesVerdict(series, assessments, "keep-linear", primary.reason,
                         None, primary.turning_point_z,
                         primary.turning_point_level)


# --------------------------------------------------------------------------
# The nonlinear readout
# --------------------------------------------------------------------------

class NonlinearTerm(object):
    """A U or inverted-U basis for one node's contribution to a score.

    shape: "U" (the contribution has a minimum in the node's rate) or
        "inverted_U" (a maximum). Taken from the fitted curvature of the
        series that licensed it.
    peak_rate: where in `[0, 1]` the turning point sits IN RATE UNITS. The
        graded experiments that license the SHAPE measured drive, not rate,
        and no drive-to-rate transduction was measured with them, so this is
        `ILLUSTRATIVE` and defaults to the network's own resting rate. P7.3
        fits it.
    """

    provenance = ILLUSTRATIVE

    def __init__(self, shape, peak_rate=0.5, licensed_by=None):
        if shape not in ("U", "inverted_U"):
            raise ValueError("shape must be 'U' or 'inverted_U', got %r"
                             % (shape,))
        if not 0.0 < peak_rate < 1.0:
            raise ValueError("peak_rate must lie strictly in (0, 1)")
        self.shape = shape
        self.peak_rate = float(peak_rate)
        self.licensed_by = licensed_by

    def basis(self, rate):
        """Map a rate in [0, 1] onto a basis value in [0, 1].

        Normalised so the basis spans the full unit interval over the rate
        range that is actually reachable, which keeps a nonlinear term on
        the same scale as the linear terms it is summed with.
        """
        m = max(self.peak_rate, 1.0 - self.peak_rate)
        d = ((rate - self.peak_rate) / m) ** 2
        if self.shape == "inverted_U":
            return 1.0 - d
        return d

    def __repr__(self):
        return "NonlinearTerm(%r, peak_rate=%.3f)" % (self.shape,
                                                      self.peak_rate)


class NonlinearReadout(object):
    """`BehaviorReadout`'s interface, with a basis function on some nodes.

    Nodes not in `nonlinear` contribute `w_i * r_i` exactly as before, so an
    endpoint with no adopted term is not merely equivalent to P7.1's readout
    but is constructed as one (`resolve_nonlinear` returns a plain
    `BehaviorReadout` in that case).
    """

    def __init__(self, name, weights, nonlinear=None):
        self.name = name
        self.weights = dict(weights)
        self.nonlinear = dict(nonlinear or {})
        unknown = set(self.nonlinear) - set(self.weights)
        if unknown:
            raise ValueError(
                "nonlinear terms on nodes with no weight: %s"
                % ", ".join(sorted(unknown))
            )

    def score(self, network, rates):
        total = 0.0
        for node_name, weight in self.weights.items():
            idx = network.index[node_name]
            r = rates[idx]
            term = self.nonlinear.get(node_name)
            total += weight * (term.basis(r) if term is not None else r)
        return total

    def __repr__(self):
        return "NonlinearReadout(%r, %d nodes, %d nonlinear)" % (
            self.name, len(self.weights), len(self.nonlinear)
        )


# --------------------------------------------------------------------------
# The data
# --------------------------------------------------------------------------

CITATIONS = {
    "Herent2023": (
        "Hérent C, Diem S, Usseglio G, Fortin G, Bouvier J. Upregulation of "
        "breathing rate during running exercise by central locomotor "
        "circuits in mice. Nat Commun 2023;14(1):2939, PMID:37217517. "
        "MOUSE, Vglut2-Cre, AAV-ChR2 in the cuneiform nucleus, unilateral, "
        "473 nm, 1 s trains at 5/10/15/20/30/40 Hz, displacement speed in "
        "an 80 x 10 cm corridor tracked with DeepLabCut. n = 5 mice x 3 "
        "trials per frequency. Per-frequency per-animal values, and the "
        "mean and SD rows used here, are in the Source Data file "
        "'FigS6-b,c.xlsx' sheet 'FigS6-panel-b'; the article body states no "
        "numeric speed."
    ),
    "Caggiano2018": (
        "Caggiano V, Leiras R, Goñi-Erro H, Masini D, Bellardita C, Bouvier "
        "J, Caldeira V, Fisone G, Kiehn O. Midbrain circuits that set "
        "locomotor speed and gait selection. Nature 2018;553(7689):455-460, "
        "PMID:29342142. MOUSE, Vglut2-Cre, AAVdj-DIO-hChR2 in the cuneiform "
        "nucleus or the pedunculopontine nucleus, unilateral, 473 nm, "
        "linear corridor. Trial-level maximum speeds at 5/10/20/50 Hz are "
        "in the published version's Source Data to Fig. 1, sheet "
        "'Panel H'. Already cited by P7.1 for the CUN locomotion term."
    ),
    "DeLuca2022": (
        "De Luca R, Nardone S, Grace KP, Venner A, Cristofolini M, Bandaru "
        "SS, Sohn LT, Kong D, Mochizuki T, Viberti B, Zhu L, Zito A, "
        "Scammell TE, Saper CB, Lowell BB, Fuller PM, Arrigoni E. Orexin "
        "neurons inhibit sleep to promote arousal. Nat Commun "
        "2022;13(1):4163, PMID:35851580. MOUSE, Ox-IRES-Cre, ChR2 in "
        "lateral hypothalamic orexin neurons with bilateral fibres over the "
        "VLPO, 12-15 mW, 5 ms pulses, 10 s at sham/1/5/10/20 Hz delivered "
        "in random order. Readout is the probability of arousal from NREM "
        "sleep. Per-frequency means, SEMs, n and per-animal values are in "
        "the Source Data file '41467_2022_31591_MOESM5_ESM.xlsx' sheet "
        "'Fig.1i-j'."
    ),
    "Diaz2011": (
        "Diaz MR, Chappell AM, Christian DT, Anderson NJ, McCool BA. "
        "Dopamine D3-like receptors modulate anxiety-like behavior and "
        "regulate GABAergic transmission in the rat lateral/basolateral "
        "amygdala. Neuropsychopharmacology 2011;36(5):1090-103, "
        "PMID:21270771. RAT, male Sprague-Dawley, bilateral intra-BLA "
        "microinjection of the D3 antagonist U99194, 0.5 uL over 30 s, "
        "0 / 1.5 / 5.0 pmol, elevated plus maze, 5 min. Open-arm time is "
        "printed with SEM in Table 1 - one of very few EPM dose series in "
        "the literature whose per-dose means are in a table rather than a "
        "figure."
    ),
    "ZhuoGebhart1997": (
        "Zhuo M, Gebhart GF. Biphasic modulation of spinal nociceptive "
        "transmission from the medullary raphe nuclei in the rat. J "
        "Neurophysiol 1997;78(2):746-58, PMID:9307109. RAT, focal "
        "electrical stimulation of NRM/NRP/NRO/NGC, tail flick. The paper "
        "P7.1 named as P7.2's flagship case. Its 5/10/15/25 uA series "
        "exists but the Results give only the minimum and maximum of the "
        "four group means and do not say which intensity produced which; "
        "the per-intensity values are in Fig. 1A alone. See "
        "SEARCHED_AND_EMPTY['analgesia']."
    ),
    "Kroeger2018": (
        "Kroeger D, Absi G, Gagliardi C, Bandaru SS, Madara JC, Ferrari LL, "
        "Arrigoni E, Münzberg H, Scammell TE, Saper CB, Vetrivelan R. "
        "Galanin neurons in the ventrolateral preoptic area promote sleep "
        "and heat loss in mice. Nat Commun 2018;9(1):4129, PMID:30297727. "
        "MOUSE, VLPO galanin, six stimulation frequencies (0.5, 1, 2, 4, 8, "
        "16 Hz) against percent NREM sleep - the cleanest graded VLPO "
        "design found, and figure-only. Already cited by P7.1 for the VLPO "
        "sedation term."
    ),
    "Liang2023": (
        "Liang M, Jian T, Tao J, Wang X, Wang R, Jin W, Chen Q, Yao J, Zhao "
        "Z, Yang X, Xiao J, Yang Z, Liao X, Chen X, Wang L, Qin H. "
        "Hypothalamic supramammillary neurons that project to the medial "
        "septum modulate wakefulness in mice. Commun Biol 2023;6(1):1255, "
        "PMID:38087004. MOUSE, 1/5/8/10/20 Hz, per-animal latencies in "
        "Supplementary Data 2. Excluded because the light is delivered onto "
        "SuM AXON TERMINALS in the medial septum, so the region whose rate "
        "is graded is not identified by the experiment."
    ),
    "Guo2023": (
        "Guo H, Jiang JB, Xu W, Zhang MT, Chen H, Shi HY, Wang L, He M, "
        "Lazarus M, Li SQ, Huang ZL, Qu WM. Parasubthalamic calretinin "
        "neurons modulate wakefulness associated with exploration in male "
        "mice. Nat Commun 2023;14(1):2346, PMID:37095092. MOUSE, "
        "parasubthalamic nucleus somata at 1/5/10/20/30 Hz, plus PSTh axon "
        "terminals over the parabrachial nucleus at 1/5/10/20 Hz and over "
        "the VTA at 5/10/20 Hz, all with per-animal NREM-to-wake latencies "
        "in the Source Data. Excluded twice over: the soma arm grades the "
        "PSTh, which carries no term in P7.1's sedation endpoint, and the "
        "two terminal arms grade a projection whose effect on the target "
        "region's own rate the paper does not establish."
    ),
    "Yamagata2021": (
        "Yamagata T, Kahn MC, Prius-Mengual J, Meijer E, Šabanović M, "
        "Guillaumin MCC, van der Vinne V, Huang YG, McKillop LE, Jagannath "
        "A, Peirson SN, Mann EO, Foster RG, Vyazovskiy VV. The hypothalamic "
        "link between arousal and sleep homeostasis in mice. Proc Natl Acad "
        "Sci U S A 2021;118(51), PMID:34903646. MOUSE, lateral preoptic "
        "GAD2, 1/2/5/10/20 Hz. Three of the five awakening probabilities "
        "are in the text and two are figure-only, so the series cannot be "
        "completed."
    ),
    "Zahrt1997": (
        "Zahrt J, Taylor JR, Mathew RG, Arnsten AF. Supranormal stimulation "
        "of D1 dopamine receptors in the rodent prefrontal cortex impairs "
        "spatial working memory performance. J Neurosci 1997;17(21):8528-35, "
        "PMID:9334425. RAT, bilateral intra-PFC SKF 81297 at 0 / 0.01 / "
        "0.1 ug in 0.5 uL, T-maze delayed alternation, n = 5. The canonical "
        "prefrontal inverted-U experiment. Per-dose percent correct is in "
        "Fig. 2 only; the paper's single table reports perseveration and "
        "run time, not accuracy."
    ),
    "BakshiGeyer1998": (
        "Bakshi VP, Geyer MA. Multiple limbic regions mediate the "
        "disruption of prepulse inhibition produced in rats by the "
        "noncompetitive NMDA antagonist dizocilpine. J Neurosci "
        "1998;18(20):8394-401, PMID:9763482. RAT, dizocilpine 0 / 0.25 / "
        "1.25 / 6.25 ug per 0.5 uL per side into six limbic regions, Latin "
        "square, n = 7-10 per site. Four levels in six regions and the "
        "paper contains no tables at all."
    ),
    "Alsene2011": (
        "Alsene KM, Rajbhandari AK, Ramaker MJ, Bakshi VP. Discrete "
        "forebrain neuronal networks supporting noradrenergic regulation of "
        "sensorimotor gating. Neuropsychopharmacology 2011;36(5):1003-14, "
        "PMID:21248721. RAT, phenylephrine + isoproterenol 0 / 3 / 10 / "
        "30 ug per 0.5 uL into nine forebrain regions. Its two tables give "
        "per-dose STARTLE MAGNITUDE with SEM, which is not %PPI and is not "
        "the endpoint."
    ),
    "Swerdlow2007": (
        "Swerdlow NR, Shoemaker JM, Bongiovanni MJ, Neary AC, Tochen LS, "
        "Saint Marie RL. Strain differences in the disruption of prepulse "
        "inhibition of startle after systemic and intra-accumbens "
        "amphetamine administration. Pharmacol Biochem Behav 2007;87(1):1-10, "
        "PMID:17475315. RAT, intra-striatal amphetamine 0 / 10 / 20 / 40 ug "
        "per 0.5 uL per side across five subregions and two strains. Four "
        "levels, and the paper has no tables."
    ),
    "MendoncaDosSantos2024": (
        "Mendonça-Dos-Santos M, Gonçalves TCT, Falconi-Sobrinho LL, Dos "
        "Anjos-Garcia T, Matias I Jr, de Oliveira RC, Dos Santos Sampaio "
        "MF, Dos Santos Cardoso F, Dos Santos WF, Machado HR, Coimbra NC. "
        "GABAergic and glutamatergic inputs to the medulla oblongata and "
        "locus coeruleus noradrenergic pathways are critical for seizures "
        "and postictal antinociception neuromodulation. Sci Rep "
        "2024;14(1):4069, PMID:38374419. RAT, intra-NTS NMDA and AP-7 dose "
        "series with tail-flick latencies printed with SEM in Table 1 - the "
        "only tabulated graded thermal-latency series found anywhere. "
        "Rejected; see SEARCHED_AND_EMPTY['analgesia']."
    ),
    "Zhang2025": (
        "Zhang Z, Cherasse Y, Louis C, Saito YC, Soya S, Hirano A, Sakurai "
        "T. The Bed Nucleus of the Stria Terminalis-Deep Mesencephalic "
        "Nucleus Circuit Linking Emotion and Wakefulness. J Neurosci "
        "2025;45(50), PMID:41193254. MOUSE, GABAergic BNST and "
        "glutamatergic deep mesencephalic nucleus at 5/10/20 Hz, with "
        "latencies and wake percentages as mean +/- SEM in running text - "
        "three levels, and neither structure carries a term in P7.1's "
        "sedation endpoint."
    ),
}

#: Citation keys that appear in this module's prose rather than on a series.
#: The orphan test checks both directions, which is the check P7.1's own
#: orphan test was found to be doing only half of.
#:
#: P11.3 moved the five rodent series to `legacy/rodent_series.py`, which
#: turned the four series-only citations below into prose-only ones: the
#: docstring still argues from [Herent2023]'s SD floor and [DeLuca2022]'s
#: terminal-field design because they are the clearest illustrations of two
#: of this module's rules. Their bibliographic entries stay here so there is
#: exactly one copy of each reference; the data they describe does not.
CITED_IN_PROSE = (
    "Herent2023", "Caggiano2018", "DeLuca2022", "Diaz2011",
    "ZhuoGebhart1997", "Kroeger2018", "Liang2023", "Guo2023", "Yamagata2021",
    "Zahrt1997", "BakshiGeyer1998", "Alsene2011", "Swerdlow2007",
    "MendoncaDosSantos2024", "Zhang2025",
)


#: P11.3 retired this library. `SERIES` held five graded single-region
#: series from four papers - four mouse optogenetic series and one rat
#: microinjection series - covering locomotion, sedation and anxiety; every
#: one describes a rodent experiment, so under the human-primary roadmap none
#: may inform a human readout and all five moved to
#: `legacy/rodent_series.py`, where they stay importable and tested.
#:
#: The empty library is a real state of the record, not a stub: no endpoint
#: has graded series data this module may use, so every verdict is
#: `no-data`, nothing is adopted (which was also true when the five series
#: were present), and the machinery below keeps running against synthetic
#: series in its tests until a HUMAN graded series exists. The searched-and-
#: empty records below describe what was found for each endpoint and why it
#: was unusable; they are the standing instruction for what to re-search.
SERIES = ()

CONTEST_RESOLUTIONS = {}


P71_TARGETS = """
P7.1 named three places where a linear readout is provably wrong and handed
them to P7.2. None of the three could be decided on held-out data, and the
reason differs in each case.

1. RM / rostral ventromedial medulla, in analgesia. [ZhuoGebhart1997] is a
   within-paper, within-assay, within-site reversal: at 43 of 86 RVM sites,
   low-intensity stimulation facilitated the tail flick and higher intensity
   inhibited it. The 5/10/15/25 uA series exists. Its per-intensity means do
   not: the Results print only "from a minimal mean of 18.4 +/- 4.5% to a
   maximum mean of 27.3 +/- 7.3%" without saying which intensity gave which,
   and the four values are in Fig. 1A. The reversal itself is quantified only
   as two summary points in incommensurable units - facilitation as percent
   change from control at a mean 11.8 +/- 1.4 uA, inhibition as percent of
   maximum possible inhibition at a mean 66.6 +/- 9.0 uA - so even the
   two-point version cannot be assembled into one series. The glutamate
   replication has two doses. P7.1's flagship case is unusable, and that is
   the single most consequential finding of this task.

2. ACB and ventral CA1/SUB, in prepulse inhibition. The inverted U P7.1
   recorded is assembled from separate hyperactivation and inactivation
   papers, which is two points from two experiments and not a graded series.
   A systematic scan of 170 open-access papers matching prepulse inhibition
   with intracerebral infusion found no per-dose %PPI in any table.

3. Working memory, structurally. P7.1's own note is that all six terms are
   positive, so the readout is up to scale the mean rate over six regions and
   cannot hold the prefrontal inverted U. [Zahrt1997] is exactly the
   experiment that would decide it - three intra-PFC D1-agonist doses against
   delayed-alternation accuracy - and its per-dose accuracy is in Fig. 2
   only. The paper's one table reports perseveration and run time.

The honest summary is that P7.1's three targets are three DIFFERENT kinds of
unavailability: a paper that measured the reversal and printed a summary of
it, a literature that measured the two limbs in different papers, and a
paper that measured the whole curve and plotted it.
"""


SEARCHED_AND_EMPTY = {
    "locomotion": """
P11.3 QUARANTINED THE THREE USABLE SERIES - [Herent2023]'s CnF frequency
series and [Caggiano2018]'s CnF and PPN series, all mouse optogenetics -
into `legacy/rodent_series.py` under the species pivot, so this endpoint has
NO usable series again and the search below is the standing instruction for
what a human one would have to beat. Rejected additionally:
  - intracerebral dose series vs locomotion. The rat microinjection
    literature (Reavill 1990 nicotine/cytisine into VTA, NAc, striatum,
    dorsal hippocampus and motor thalamus; Delfs 1990 intra-accumbens
    cocaine 0/10/30/100 ug; Zangen 2002 intra-VTA and intra-NAS EM-1
    0/0.1/0.3/1 nmol; Bristow 1988 intra-NAc histamine at six levels, an
    explicit inverted U) has the right designs and prints F-statistics.
    Reavill 1990 states vehicle and top-dose cage crosses in text and leaves
    the intermediate doses in the figure; the rest are figure-only
    throughout. Bristow 1988 is the one that would most likely have been
    adopted and its six group means are a bar chart.
  - Roseberry 2016, Josset 2018, Fougere 2021, van der Zouwen 2021, Dautan
    2021, Capelli 2017, Cregg 2024, Friedman 2023: graded designs, no
    per-level speed outside a figure, and in three cases the discrete level
    list is itself only on a figure axis.
""",
    "anxiety": """
P11.3 quarantined this endpoint's one series - [Diaz2011], rat intra-BLA
microinjection, three levels, rejected for want of a fourth even before the
species pivot - into `legacy/rodent_series.py` with the rest of the rodent
bridge. Beyond it:
  - No optogenetic frequency or power series anywhere against elevated-plus-
    maze open-arm time or open-field centre time. The four canonical papers
    (Tye 2011, Kim 2013, Jennings 2013, Felix-Ortiz 2013) were each read in
    full text AND supplement: every anxiety measurement in all four is at a
    SINGLE stimulation condition. Where those papers do contain a graded
    series, the readout is something else - BLA spiking, slice spike
    fidelity, or intracranial self-stimulation nose-pokes.
  - A screen of ~530 open-access full texts for three or more distinct
    Hz/mW/uA values in a paragraph mentioning an anxiety readout returned 19
    hits and all 19 were false positives of the same kind: the multi-level
    series belonged to a different experiment in the same paper.
  - Intra-dorsal-PAG electrical intensity series exist in quantity and every
    one is a threshold titration reporting a single scalar escape threshold
    per animal, not a behavioural value per intensity.
  - A second, independent screen downloaded 1,629 open-access full texts and
    tested every <table-wrap> and every 1,100-character window around an
    open-arm or centre-time mention for three or more mean +/- SEM values.
    [Diaz2011] Table 1 was the only clean hit in all 1,629. Two partials:
    Zangrossi & Graeff 1994 (intra-amygdala midazolam 20/40 nmol, ketanserin
    1/10 nmol, 8-OH-DPAT 2/4/8 nmol) prints control and top-dose percentages
    in its ABSTRACT and its 1994 volume is not digitised anywhere; Askari-
    Zahabi 2022 (intra-BLA SB-334867 at 10/20/40 nM) states one dose per
    measure in text and leaves the rest in figures, under a nitroglycerin
    migraine model. Rodriguez-Landa 2022 is the best-designed graded EPM
    series found - CA1 chrysin and allopregnanolone, four levels each, with
    supplementary TABLES - and those tables give closed-arm entries and
    locomotion, never open-arm time. Carvalho 2012 (intra-BLA and intra-CeA
    midazolam, saline/20/30 nmol) is the best-designed unreachable one.
""",
    "sedation": """
P11.3 quarantined this endpoint's one series - [DeLuca2022], mouse, orexin
terminal-field stimulation of the VLPO - into `legacy/rodent_series.py`
with the rest of the rodent library. The arousal literature is the richest
source of graded designs in this whole search and most of it still cannot
be used:
  - [Kroeger2018] is the cleanest VLPO design found - six frequencies
    against percent NREM sleep, and the region P7.1 already weights most
    strongly - and every value is in Fig. 2c. Its peer-review file was
    checked too and contains no numbers.
  - [Liang2023] (1/5/8/10/20 Hz) and [Guo2023] (five levels at the soma, four
    and three at two terminal fields) both deposit per-animal values. Both
    are excluded on identifiability rather than availability. Guo's soma arm
    grades the parasubthalamic nucleus, which carries no term in this
    endpoint, and a verdict about a region the readout does not contain
    cannot change the readout. Guo's parabrachial terminal arm and all of
    Liang illuminate AXON TERMINALS, and there the region whose rate is
    graded is not identified by the experiment: the drive is to a
    projection, and which of its source and target the behaviour follows is
    the question rather than the answer. [DeLuca2022] is a terminal-field
    manipulation too and IS used, and the difference is the point - its own
    result is about what happens to VLPO sleep-active neurons, so the
    graded region is named by the paper. Guo's and Liang's are not.
  - [Yamagata2021] states three of five awakening probabilities in text and
    leaves two in a figure, so the series cannot be completed.
  - [Zhang2025] prints every value as mean +/- SEM in running text, which is
    rare and welcome, and has three levels in two structures neither of
    which this endpoint contains.
  - Carter 2010 and Adamantidis 2007 - the two papers P7.1 cites for LC and
    LHA and the obvious first place to look - are both figure-only. Carter
    reports no latency at any frequency in any form; Adamantidis's discrete
    frequency set is itself only on a figure axis.
  - Li 2022 (Science) ran a full 5 x 5 frequency-by-power matrix in LH
    hypocretin neurons. Its per-condition statistics are in a non-open-access
    SI appendix and could not be reached.
  - No TMN/histaminergic, median raphe or thalamic-reticular in vivo graded
    series with a sleep readout exists at all.
""",
    "analgesia": """
NO SERIES. This is the endpoint P7.1 pointed P7.2 at, and it is empty. See
`P71_TARGETS` for [ZhuoGebhart1997] itself.

  - A regex scan of 251 open-access full texts containing "tail flick" or
    "hot plate" for three or more distinct uA/mA or Hz values in a brain-
    stimulation context returned 24 candidates and every one was a threshold
    titration, a two-level design, or a non-thermal readout. The Gebhart-lab
    intensity paradigm - ramp the current stepwise until the reflex reaches
    cutoff - dominates this literature and by construction reports one
    scalar threshold per animal instead of a latency per level.
  - Graded intracerebral series with thermal latency: de Mello Rosa 2022
    (vlPAG NMDA, three doses), Mascarenhas 2017 (dPAG anandamide, capsaicin
    and WIN, four doses each), Buhler 2008 (RVM beta-lactotensin, four
    doses), and the Morgan-lab vlPAG cumulative opioid series (five doses,
    nine papers) are all figure-only. The Morgan papers tabulate D50 with a
    confidence interval, which is a derived summary and not a per-level
    value.
  - [MendoncaDosSantos2024] is the ONLY paper found anywhere that prints a
    graded thermal-latency series in a table, and it is rejected on the
    numbers rather than on availability. Its Methods and Results say the
    middle NMDA dose was 3 nmol and its Table 1 says 6 nmol. A drive level
    the paper contradicts itself about cannot be the x-axis of a fit, and
    choosing one of the two would be inventing the number this codebase's
    first rule forbids. Two further problems would have to be answered even
    if that were resolved: every dose group is co-treated with i.p.
    pentylenetetrazole so the readout is postictal antinociception rather
    than a clean single-region manipulation, and the table gives one value
    per group without stating which of the fourteen measured timepoints it
    is.
  - Thomas & Hammond 1995 (NRM and NGCpalpha R(+)-baclofen over 0.1-150 ng,
    tail flick AND hot plate, explicitly biphasic - latencies rise at
    0.1-1.0 ng and fall back to control by 30-50 ng) is the single
    highest-value unreachable target in this entire search. It is a
    published, graded, within-region, both-assay reversal in the right
    species. JPET returns 403 to every automated route and the paper has no
    PMC deposit. Urban & Smith 1993 (NRM neurotensin, hyperreflexia at
    0.03-0.3 nmol and inhibition at 3-30 nmol) is the same story.
""",
    "reward": """
NO SERIES.
  - Ilango 2014 is the closest structural match found anywhere: mouse VTA
    dopamine ChR2 with FOUR separate graded series in one paper (pulse
    duration 1/3/10 ms, intensity 0.1/1/10 mW, frequency 8/20/50 Hz, pulse
    number 0/2/4/8/16) against lever-press self-stimulation. Every rate is
    in Fig. 4 and the text gives only F-values. Its one table is a
    schedule-feasibility table.
  - Rate-frequency intracranial self-stimulation is a large literature with
    genuinely graded designs (Witten 2011, nine pulse-number levels; Yoo
    2016 and 2017; Galaj 2020/2022 and Humburg 2021, six frequencies each).
    None tabulates a rate per frequency; the classic Negus- and
    Carlezon-lab papers tabulate only derived curve parameters (Emax, T50,
    EF50), and Witten's rates are normalised per rat so they are not
    absolute even in the figure.
  - Two Shizgal-lab deposits (Trujillo-Pisanty 2020, TH::Cre rat VTA
    optogenetic; Solomon 2017, rat lateral hypothalamic electrical, 9 to 25
    pulse frequencies) put raw trial-level records in the Concordia Spectrum
    repository as paired .DAT/.PRM files. Those are real data at the right
    resolution and they are not ingested here: reconstructing a
    frequency-response curve from them means writing a parser for an
    undocumented binary-ish format on the authority of a README, which is a
    different kind of task from transcribing a published table and carries a
    different kind of error. The URLs are in [the reward record] so P7.3 can
    decide whether to spend that effort:
    https://spectrum.library.concordia.ca/id/eprint/986807/1/OptoGBR.zip and
    https://spectrum.library.concordia.ca/id/eprint/982609/1/OppCost_PLOSONE_Dataset.zip
  - Intra-VTA and intra-NAc conditioned-place-preference dose series: none
    open access. Bals-Kubik 1993 is the canonical one and has no PMC record.
""",
    "working_memory": """
NO SERIES. See `P71_TARGETS` for [Zahrt1997], which is the experiment this
endpoint needs and is figure-only.

The structural point P7.1 recorded still stands and P7.2 could not move it:
all six working-memory terms are positive, so the readout is the mean rate
over its regions up to scale, and the prefrontal inverted U is the one shape
an all-positive linear readout cannot hold. Vijayraghavan 2007 is the other
famous inverted-U experiment and is excluded twice over - it is MONKEY, and
its graded readout is neuronal firing rate rather than a behavioural score.
""",
    "sensorimotor_gating": """
NO SERIES, and this is the most systematically established negative in the
task.

A Europe PMC query for prepulse inhibition with intracerebral infusion,
restricted to open access with full text, returned 170 papers. Every one was
downloaded and every <table-wrap> was regex-tested for a PPI token
co-occurring with a mean +/- SEM numeric pattern. ZERO hits. No open-access
prepulse-inhibition paper tabulates %PPI.

Ten studies were verified as qualifying on design (>= 3 intracerebral levels
with %PPI at each) and all ten are figure-only:
[BakshiGeyer1998] (4 doses x 6 regions, and the paper has no tables at all),
[Alsene2011] (4 doses x 9 regions), [Swerdlow2007] (4 doses x 5 subregions x
2 strains, no tables), Alsene & Bakshi 2011 (peri-LC bethanechol), Chang
2012 (pramipexole into 3 striatal sites), Cadeddu 2022 (mouse mPFC and NAc
allopregnanolone), Pinnock 2015 (PnC nicotinic antagonists), Larrauri 2015
(lateral habenula nicotine - the only one with any %PPI number in text, and
only three of its conditions), Sotoyama 2011 (globus pallidus SCH23390 and
raclopride), Arime 2012 (mouse mPFC and NAc nisoxetine).

Seven further classic Swerdlow/Wan/Caine designs from 1990-2005 qualify from
their abstracts and are paywalled with no PMC deposit.

[Alsene2011] deserves its own line because it is the one place a table of a
graded intracerebral series with SEM does exist - 4 levels x 9 regions - and
its readout is STARTLE MAGNITUDE, not %PPI. Prepulse inhibition is a ratio
of two startle responses and startle magnitude is one of them; substituting
it would be substituting a different measured effect for the one being
modelled, which rule 2 of this codebase forbids by name. It is recorded here
rather than used.
""",
}


# --------------------------------------------------------------------------
# From verdicts to readouts
# --------------------------------------------------------------------------

class EndpointVerdict(object):
    """Every graded series that speaks to one endpoint, and the outcome."""

    def __init__(self, endpoint_key, series_verdicts, reason):
        self.endpoint_key = endpoint_key
        self.series_verdicts = tuple(series_verdicts)
        self.reason = reason

    @property
    def adopted(self):
        return tuple(v for v in self.series_verdicts if v.adopted)

    @property
    def rejected(self):
        return tuple(v for v in self.series_verdicts if not v.adopted)

    @property
    def nonlinear(self):
        return bool(self.adopted)

    def report(self):
        lines = [
            "=== %s: %s"
            % (self.endpoint_key,
               "NONLINEAR (%d series adopted)" % len(self.adopted)
               if self.nonlinear else "LINEAR"),
            "    %s" % self.reason,
        ]
        for v in self.series_verdicts:
            lines.append("")
            lines.append("\n".join("  " + ln for ln in v.report().split("\n")))
        return "\n".join(lines)

    def __repr__(self):
        return "EndpointVerdict(%r, %d series, nonlinear=%s)" % (
            self.endpoint_key, len(self.series_verdicts), self.nonlinear
        )


def series_for(endpoint_key):
    return tuple(s for s in SERIES if s.endpoint == endpoint_key)


_ASSESSMENT_CACHE = {}


def assess_all():
    """Every series' verdict, in `SERIES` order. Cached; fits are cheap."""
    if "all" not in _ASSESSMENT_CACHE:
        _ASSESSMENT_CACHE["all"] = tuple(assess_series(s) for s in SERIES)
    return _ASSESSMENT_CACHE["all"]


def endpoint_verdict(endpoint_key):
    """The decoder verdict for one endpoint of `endpoints.ENDPOINTS`."""
    if endpoint_key not in endpoints_module.ENDPOINTS:
        raise KeyError(endpoint_key)
    verdicts = [v for v in assess_all() if v.series.endpoint == endpoint_key]
    if not verdicts:
        return EndpointVerdict(endpoint_key, (),
                               REJECTION_REASONS["no-data"])
    adopted = [v for v in verdicts if v.adopted]
    if adopted:
        reason = (
            "%d of %d graded series showed a reversal that survived "
            "leave-one-out; the decoder takes a nonlinear basis on %s"
            % (len(adopted), len(verdicts),
               ", ".join(sorted({s for v in adopted
                                 for s in v.series.structures})))
        )
    else:
        reason = (
            "%d graded series, none of which licensed a nonlinearity: %s"
            % (len(verdicts),
               "; ".join("%s -> %s" % (v.series.key, v.reason)
                         for v in verdicts))
        )
    return EndpointVerdict(endpoint_key, verdicts, reason)


def all_endpoint_verdicts():
    return tuple(endpoint_verdict(k)
                 for k in endpoints_module.ENDPOINT_ORDER)


def adopted_nonlinearities():
    """{(endpoint_key, acronym): NonlinearTerm}, DERIVED from `assess_all`.

    Never hand-written. If a series' numbers change, or a family's fit
    changes, the adoption list changes with it, and there is no second copy
    of the answer to drift out of step with the comparison.
    """
    out = {}
    for v in assess_all():
        if not v.adopted:
            continue
        for acronym in v.series.structures:
            out[(v.series.endpoint, acronym)] = NonlinearTerm(
                shape=v.shape, licensed_by=v.series.key
            )
    return out


class NonlinearResolution(object):
    """A resolved endpoint whose readout carries the adopted nonlinearities.

    readout: a `NonlinearReadout` when at least one adopted term reached a
        node of this network, and a plain `behavior.BehaviorReadout`
        otherwise - constructed as one, not merely equivalent to one, so an
        endpoint P7.2 leaves alone is exactly the object P7.1 produced.
    """

    def __init__(self, endpoint_key, resolved, readout, applied, blocked,
                 contests_resolved):
        self.endpoint_key = endpoint_key
        self.resolved = resolved
        self.readout = readout
        self.applied = dict(applied)
        self.blocked = tuple(blocked)
        self.contests_resolved = tuple(contests_resolved)

    @property
    def weights(self):
        return dict(self.resolved.weights)

    def score(self, network, rates):
        return self.readout.score(network, rates)

    def report(self):
        lines = [self.resolved.report()]
        if self.applied:
            for node in sorted(self.applied):
                term = self.applied[node]
                lines.append(
                    "  NONLINEAR at %-8s %s, peak_rate=%.2f (%s), "
                    "licensed by %s"
                    % (node, term.shape, term.peak_rate, term.provenance,
                       term.licensed_by)
                )
        else:
            lines.append("  decoder: linear (P7.1 unchanged)")
        for node, why in self.blocked:
            lines.append("  NONLINEARITY BLOCKED at %s: %s" % (node, why))
        for note in self.contests_resolved:
            lines.append("  CONTEST RESOLVED: %s" % note)
        return "\n".join(lines)

    def __repr__(self):
        return "NonlinearResolution(%r, %d nonlinear node(s))" % (
            self.endpoint_key, len(self.applied)
        )


def resolve_nonlinear(endpoint_key, network, ontology, region_sizes=None,
                      collapse="mean", normalize=True,
                      include_contested=False):
    """P7.1's `Endpoint.resolve`, plus whatever P7.2's data licensed.

    A contested term whose contest IS the reversal the data licensed is
    re-included, because the two studies that contest it are the two limbs
    of one non-monotone curve rather than a disagreement. That re-inclusion
    is not automatic for every adopted term: it needs an entry in
    `CONTEST_RESOLUTIONS` saying which two reports the nonlinearity
    reconciles, and a term contested for any other reason stays out.
    """
    endpoint = endpoints_module.ENDPOINTS[endpoint_key]
    adopted = {a: t for (ek, a), t in adopted_nonlinearities().items()
               if ek == endpoint_key}

    contests_resolved = []
    if not include_contested and adopted:
        readmit = []
        for term in endpoint.terms:
            if not term.contested:
                continue
            hit = [a for a in term.structures if a in adopted]
            if not hit:
                continue
            note = CONTEST_RESOLUTIONS.get((endpoint_key, hit[0]))
            if note is None:
                continue
            readmit.append(term)
            contests_resolved.append("%s: %s" % (hit[0], note))
        if readmit:
            endpoint = endpoints_module.Endpoint(
                key=endpoint.key, name=endpoint.name, assay=endpoint.assay,
                direction=endpoint.direction,
                terms=list(endpoint.active_terms(False)) + readmit,
                notes=endpoint.notes,
            )

    resolved = endpoint.resolve(
        network, ontology, region_sizes=region_sizes, collapse=collapse,
        normalize=normalize, include_contested=include_contested,
    )

    # Which node did each adopted structure land on, and does that node
    # carry anything else? A node fed by an adopted term AND by a linear
    # term cannot take the basis function: one scalar rate would have to
    # pass through two different response shapes at once. That is P7.1's
    # sign-cancellation story in a new place, and it is reported rather
    # than resolved by picking one.
    terms_on_node = {}
    for p in resolved.placements:
        if p.rule == "unmapped":
            continue
        for node in p.nodes:
            terms_on_node.setdefault(node, set()).add(id(p.term))

    applied = {}
    blocked = []
    for acronym, term in adopted.items():
        nodes = tuple(
            node
            for p in resolved.placements
            if p.structure == acronym and p.rule != "unmapped"
            for node in p.nodes
        )
        if not nodes:
            blocked.append((acronym, "the structure did not reach any node "
                                     "of this network"))
            continue
        for node in nodes:
            if len(terms_on_node.get(node, ())) > 1:
                blocked.append((
                    node,
                    "%s shares this node with %d other term(s); one rate "
                    "cannot carry a nonlinear and a linear response at "
                    "once, so this parcellation is too coarse to express "
                    "the adopted nonlinearity"
                    % (acronym, len(terms_on_node[node]) - 1),
                ))
                continue
            if node in resolved.weights:
                applied[node] = term

    if applied:
        readout = NonlinearReadout(endpoint_key, resolved.weights, applied)
    else:
        readout = behavior.BehaviorReadout(endpoint_key, resolved.weights)

    return NonlinearResolution(endpoint_key, resolved, readout, applied,
                               blocked, contests_resolved)
