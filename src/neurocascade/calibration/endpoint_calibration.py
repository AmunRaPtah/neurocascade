"""P7.3: fitting the P7.1 endpoints against published dose-response data.

The brief, and the three things that had to be true
---------------------------------------------------

    Fit each behavioral endpoint against real published dose-response
    curves through the Phase 1 PK bridge ... remove the illustrative label
    only from parameters actually fit.

For one endpoint to be fitted, three things have to hold at once, and this
module's whole structure is the bookkeeping of which of them fail where:

1. **A dataset.** Real, published, in a species and by a route the PK layer
   models, with per-group numbers and a dispersion. Between P7.3 and P11.3
   that was `behavioral_evidence.py`'s job; P11.3 moved every dataset the
   module ever had - all rodent - to `legacy/behavioral_evidence.py`, and no
   human dataset has replaced them. So `TARGETS` below is EMPTY, and the
   honest state of this module is machinery awaiting data.
2. **A forward model that reaches the endpoint.** Dose -> brain
   concentration -> occupancy -> regional rates -> the P7.1 readout. That is
   `endpoint_forward.py`, built for this task because nothing composed those
   pieces before.
3. **Something identifiable at the end of it.** A published behavioural
   number is in seconds or counts; the model's readout is in rates. What
   connects them is an unknown scale, and whether a dataset can separate
   that scale from a circuit-layer gain is a question about the dataset's
   SHAPE, not about the code.

Point 3 is the one this task turns on, so it is stated exactly rather than
discovered later:

    a two-point contrast (vehicle, one dose) CANNOT separate a gain from a
    scale.

Write the behavioural measure as `b = k * A`, `A` the model's endpoint
index, `k` the unknown assay scale. One drug group gives one number. Two
unknowns, one equation. Every dataset P7.3 had except one was a two-point
contrast, and the one exception was a curve whose own two published
parameters faced three model parameters. So there was no dataset in reach
from which a gain and a scale could BOTH be identified.

P2.2 met the same wall and got past it by ASSUMING `k` away: if `b = k * A`
with the same `k` in both groups, the drug/vehicle RATIO is `A_drug /
A_vehicle` and `k` cancels. That assumption is what this module inherits,
and it is why every fit here is a fit to a ratio. It is not free - see
`circuit_forward.py` and P2.2's section - and at whole-brain scale it needs
one more thing P2.2 did not: an index whose zero means something. That is
`endpoint_forward.directed_activity_index`, and the fact that it equals
exactly 0.5 at rest for all seven endpoints is what keeps these ratios off
a near-zero denominator.

What the P7.3 fits found, and where that record now lives
---------------------------------------------------------

Every fit P7.3 ran CONVERGED and none licensed a number. The three failure
modes were different and all were informative:

* **anxiety**: the model moved the right way and not far enough. At the
  largest gain the mechanism allowed, a whole-brain benzodiazepine moved the
  anxiety index by ~4% where [Graziani2022]'s rats moved ~22%. The posterior
  piled against its upper bound, exactly the situation P2.2 refused to
  report as a fit.
* **locomotion and sedation**: the model moved the WRONG WAY. Potentiating
  inhibition brain-wide lowers the locomotion index, and [Graziani2022]'s
  rats crossed MORE lines under diazepam.
* **the two graded series**: the model moved too little for the data's own
  precision to see, and both posteriors came back as their own prior. The
  binding constraint was P3.4's illustrative operating point, not the
  datasets.

All three were reported with their diagnostics and none removed a label;
`docs/architecture.md`'s P7.3 section works through why. Those runs were on
rodent targets, so P11.3 retired the targets with their dataset - the
committed diagnostics in `results/calibration/endpoint_fits.txt` and
`endpoint_fit_diagnostics.csv` survive as the historical record, reproducible
from `legacy.behavioral_evidence`. What stays here is everything species-
independent: the target container, the verdict machinery, the fit, and
`synthetic_target()`, which exists so the machinery demonstrably still runs
end to end against a clearly-labelled synthetic observation instead of rotting
until a human dataset arrives.
"""

import os

import numpy as np

from .. import endpoints as _endpoints
from . import endpoint_forward as efwd
from . import forward as fwd

SEED = 20260814

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "results", "calibration"
)

#: The whole-brain gains P7.3 attempted to fit. NOT in `priors.PRIOR_SPECS`,
#: and that is deliberate: `priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` is the
#: register of circuit-layer parameters this project HAS calibrated - empty
#: since P11.3 - and nothing here survives its own fit as a calibration
#: (see the module docstring). They live here, with their bounds and the
#: reason for each bound, and `tests/test_endpoint_calibration.py` asserts
#: they stay out of `priors`.
#:
#: `gaba_a_whole_brain_modulation_gain` is a DIFFERENT parameter from
#: `scenarios.GABA_A_CONNECTION_MODULATION_GAIN`, and the difference matters:
#: the scenario gain potentiates ONE synapse in a three-node amygdala circuit,
#: this one potentiates every inhibitory connection in a 291-region brain,
#: weighted by each region's Gabra1 density. They are not comparable and
#: neither is evidence about the other.
#:
#: Bounds, from the mechanism rather than from a convention:
#:   lower 0.0 - the null, "diazepam does not potentiate GABAergic
#:     transmission", which predicts a ratio of exactly 1 for every
#:     endpoint and every circuit. A meaningful, falsifiable value.
#:   upper 5.0 - at near-saturating benzodiazepine-site occupancy, a gain of
#:     5 multiplies every inhibitory weight by about 5.5.
#:     Benzodiazepines potentiate GABA-A currents by roughly two-fold;
#:     five and a half is a wide box, not a claim.
WHOLE_BRAIN_PRIOR_SPECS = {
    "gaba_a_whole_brain_modulation_gain": ("uniform", 0.0, 5.0),
}


class EndpointTarget(object):
    """One published contrast, reduced to the two numbers a fit compares.

    `observed_ratio` is drug-group over vehicle-group on a measure that is
    bounded below by a real zero, so that the P2.2 cancellation argument
    applies to it; `observed_sd` is the delta-method propagation of the
    source's own dispersions and is never an allowance.

    `endpoint_key` names the P7.1 endpoint whose index the model side
    reads, and `measure_direction` says whether the published measure rises
    or falls with that endpoint's own direction. It is +1 when they agree
    (immobility rises with sedation) and -1 when they oppose, and it is
    written down rather than inferred for the same reason `EndpointTerm`
    refuses a `sign=` argument: a polarity nobody wrote down is a sign error
    waiting to happen.
    """

    def __init__(self, key, endpoint_key, evidence_key, measure, gene,
                 mechanism, levels, exposure_timepoint_min, Ki_nM,
                 efficacy, measure_direction, note=None):
        if endpoint_key not in _endpoints.ENDPOINTS:
            raise ValueError("unknown endpoint %r" % (endpoint_key,))
        if measure_direction not in (1, -1):
            raise ValueError(
                "measure_direction must be +1 or -1, never 0 or omitted"
            )
        if not levels:
            raise ValueError("a target needs at least one drug level")
        self.key = str(key)
        self.endpoint_key = endpoint_key
        self.evidence_key = evidence_key
        self.measure = measure
        self.gene = gene
        self.mechanism = mechanism
        #: (dose mg/kg, unbound brain nM, observed drug/vehicle ratio, sd of
        #: that ratio), one per non-vehicle dose. A single-dose contrast is
        #: the length-1 case rather than a different shape, exactly as
        #: `BehavioralEvidenceEntry.doses_mg_per_kg` is a tuple.
        self.levels = tuple(
            (float(d), float(c), float(r), float(sd)) for d, c, r, sd in levels
        )
        self.exposure_timepoint_min = float(exposure_timepoint_min)
        self.Ki_nM = float(Ki_nM)
        self.efficacy = float(efficacy)
        self.measure_direction = int(measure_direction)
        self.note = note

    @property
    def n_levels(self):
        return len(self.levels)

    @property
    def doses_mg_per_kg(self):
        return tuple(level[0] for level in self.levels)

    def activation(self, Ki_nM=None):
        """Receptor activation at each level's exposure, one per dose.

        Integrated through `forward.equilibrium_occupancy_from_kd`, i.e.
        the real `ReceptorSite.occupancy_rhs`, not a closed form.
        """
        Ki = self.Ki_nM if Ki_nM is None else float(Ki_nM)
        return np.array([
            fwd.equilibrium_occupancy_from_kd(
                Ki, exposure, efficacy=self.efficacy
            )
            for _, exposure, _, _ in self.levels
        ])

    def endpoint_ratio(self):
        """`(ratios, sds)`, oriented so that they can be compared with the
        model's ENDPOINT index ratio directly.

        For `measure_direction == -1` the published measure falls as the
        endpoint rises, so the comparable quantity is the reciprocal, whose
        first-order sd is `sd / ratio**2`. Doing this here, once, is the
        whole reason `measure_direction` exists.
        """
        ratios = np.array([level[2] for level in self.levels])
        sds = np.array([level[3] for level in self.levels])
        if self.measure_direction == 1:
            return ratios, sds
        return 1.0 / ratios, sds / ratios ** 2

    def __repr__(self):
        return "EndpointTarget(%r, %s)" % (self.key, self.endpoint_key)


#: Diazepam's Ki at the benzodiazepine site ([MohlerOkada1977] via
#: `priors.PRIOR_SPECS`), kept as the DEFAULT affinity a future diazepam
#: target holds fixed rather than resamples: at near-saturating benzodiazepine-
#: site occupancy the ratio depends on the Ki far too weakly for a whole-brain
#: fit to say anything new about it, which is why P7.3 held it constant and
#: why P2.2 reported its own update as weak "rather than overstated". It is a
#: receptor-binding constant from a binding assay, not a circuit parameter,
#: and survives the species pivot under P11.2's binding-record gate.
DIAZEPAM_KI_NM = 3.6

#: Diazepam's efficacy at the benzodiazepine site in `scenarios.py`, where
#: efficacy carries potentiation strength for a connection_modulation-
#: coupled site (see that module).
DIAZEPAM_EFFICACY = 1.0


#: The gene and mechanism every GABA-A target of this module declares: the
#: whole-brain gain acts through Gabra1-weighted inhibitory connection
#: modulation (`endpoint_forward.py`). When the five rodent targets lived
#: here this pair was also how `drug_pk`+`brain_partition` were reached -
#: diazepam/rat/IP was then the only receptor-driving triple - but the
#: reachability check itself always lived in the evidence entry, which moved
#: to `legacy/behavioral_evidence.py` with its datasets.
GABA_A_GENE = "Gabra1"
GABA_A_MECHANISM = "connection_modulation"


# ---------------------------------------------------------------------------
# Targets: EMPTY since P11.3
# ---------------------------------------------------------------------------

#: Every target P7.3 fitted was rodent - five contrasts across three
#: datasets ([Graziani2022] anxiety/locomotion/sedation, [Sagara2008]
#: locomotion, [Lustyk2023] analgesia) - so P11.3 retired them all together
#: with their dataset. The five remain reproducible from
#: `legacy.behavioral_evidence`, and the committed diagnostics under
#: `results/calibration/` are their historical record. Nothing here claims
#: any of those fits as a calibration: none came back identified (see the
#: module docstring), which is a large part of WHY the species pivot could
#: retire them without losing a single labelled parameter.
#:
#: EMPTY BY DESIGN, not a stub: everything below still works, exercised on
#: explicitly-labelled synthetic targets by `tests/test_endpoint_calibration.py`.
#: The next real entry must be built from HUMAN behavioural evidence
#: (docs/roadmap-human.md); a rodent entry would fail the quarantine test.
TARGETS = ()
TARGETS_BY_KEY = {t.key: t for t in TARGETS}


def synthetic_target(key, endpoint_key, ratios, sds, exposures_nM,
                     doses_mg_per_kg=None, measure_direction=1,
                     Ki_nM=DIAZEPAM_KI_NM, efficacy=DIAZEPAM_EFFICACY,
                     exposure_timepoint_min=float("nan"),
                     gene=GABA_A_GENE, mechanism=GABA_A_MECHANISM,
                     evidence_key="SYNTHETIC", note=None):
    """Build an `EndpointTarget` whose observation NO dataset supports.

    P11.3 emptied `TARGETS` - every dataset this module ever had was rodent
    (see the module docstring) - and an empty register cannot exercise the
    verdict and fit machinery below. This builder exists so tests, scripts
    and any future sanity check can construct an explicitly-labelled
    synthetic observation instead of letting the machinery rot until a human
    dataset arrives. Exposures are stated directly in nM, bypassing the
    Phase 1 bridge; nothing downstream can mistake the result for literature
    evidence, because `evidence_key` says SYNTHETIC and the caller is
    expected to say so in `key` too.

    `doses_mg_per_kg` defaults to NaN per level: a synthetic exposure has no
    honest dose behind it, and inventing one would be exactly what rule 1
    forbids. Pass real doses only if you have them.
    """
    if not len(ratios) == len(sds) == len(exposures_nM):
        raise ValueError("ratios, sds and exposures_nM must be equal length")
    if doses_mg_per_kg is None:
        doses_mg_per_kg = [float("nan")] * len(ratios)
    levels = tuple(
        (dose, exposure, ratio, sd)
        for dose, exposure, ratio, sd in zip(
            doses_mg_per_kg, exposures_nM, ratios, sds)
    )
    return EndpointTarget(
        key=key, endpoint_key=endpoint_key, evidence_key=evidence_key,
        measure="synthetic drug/vehicle index ratio", gene=gene,
        mechanism=mechanism, levels=levels,
        exposure_timepoint_min=exposure_timepoint_min, Ki_nM=Ki_nM,
        efficacy=efficacy, measure_direction=measure_direction, note=note,
    )


def synthetic_target_from_curve(curve, endpoint_key, gain, exposures_nM, sds,
                                Ki_nM=DIAZEPAM_KI_NM,
                                efficacy=DIAZEPAM_EFFICACY,
                                measure_direction=1, key=None, note=None):
    """A synthetic target SELF-CONSISTENT with a response curve.

    Each level's observed index ratio is exactly what `predicted_ratio`
    returns at the requested gain for that level's exposure, so fitting the
    returned target against the same curve recovers the gain. That is the
    demonstration that this module's machinery still fits an endpoint target
    end to end on synthetic data, used by `tests/test_endpoint_calibration.py`
    exactly as `bayesian_calibration.build_single_kd_model` is used there.

    `measure_direction=-1` is honoured both ways: the stored observation is
    the reciprocal the polarity rule demands, so the target's own
    `endpoint_ratio()` returns the same numbers a real oppositely-signed
    measure would produce.
    """
    activations = [
        fwd.equilibrium_occupancy_from_kd(Ki_nM, float(exposure), efficacy=efficacy)
        for exposure in exposures_nM
    ]
    x = float(gain) * np.array(activations)
    index_ratios = np.asarray(curve(endpoint_key, x)) / efwd.BASELINE_INDEX
    if measure_direction == -1:
        observed = 1.0 / index_ratios
        observed_sds = np.asarray(sds, dtype=float) / index_ratios ** 2
    else:
        observed = index_ratios
        observed_sds = sds
    return synthetic_target(
        key=key if key is not None else "%s/SYNTHETIC@gain=%g" % (endpoint_key, gain),
        endpoint_key=endpoint_key, ratios=observed, sds=observed_sds,
        exposures_nM=exposures_nM, measure_direction=measure_direction,
        Ki_nM=Ki_nM, efficacy=efficacy,
        note=note if note is not None else (
            "generated by synthetic_target_from_curve at gain=%g" % gain
        ),
    )


# ---------------------------------------------------------------------------
# The forward map a fit needs
# ---------------------------------------------------------------------------


def predicted_ratio(curve, target, gain, Ki_nM=None):
    """Model drug/vehicle ratios of the endpoint index at `gain`, one per
    dose level.

    `curve` is an `endpoint_forward.ResponseCurve` over
    `x = gain * activation`; the denominator is
    `endpoint_forward.BASELINE_INDEX`, which is exactly 0.5 at the drug-free
    state for every endpoint and is not a fitted or measured quantity.
    """
    x = float(gain) * target.activation(Ki_nM)
    return np.asarray(curve(target.endpoint_key, x)) / efwd.BASELINE_INDEX


def reachable_ratio_range(curve, target):
    """`(ratios at the prior's lower bound, ratios at its upper bound)`.

    The numbers that decide whether a posterior is a fit or a description
    of its box, computed rather than eyeballed off a trace. P2.2 made the
    same check before sampling and it is the reason its own prior bounds
    were taken from the mechanism.
    """
    _, lo, hi = WHOLE_BRAIN_PRIOR_SPECS["gaba_a_whole_brain_modulation_gain"]
    return (
        predicted_ratio(curve, target, lo),
        predicted_ratio(curve, target, hi),
    )


def best_residual_sd(curve, target, points=201):
    """`(gain, worst standardised residual)` at the best gain in the box.

    The decisive number of this whole task, and the reason it is computed
    rather than read off a posterior: a posterior can look perfectly
    healthy - converged, unimodal, interior - while the model's BEST
    achievable prediction misses the observation by several of its own
    standard deviations. Scanned on a grid rather than optimised because
    the box is one-dimensional and 201 points over it is cheaper than an
    optimiser call.
    """
    _, lo, hi = WHOLE_BRAIN_PRIOR_SPECS["gaba_a_whole_brain_modulation_gain"]
    observed, sds = target.endpoint_ratio()
    best, best_gain = np.inf, None
    for gain in np.linspace(lo, hi, points):
        residual = np.max(np.abs(predicted_ratio(curve, target, gain) - observed) / sds)
        if residual < best:
            best, best_gain = residual, gain
    return float(best_gain), float(best)


def bound_verdict(curve, target):
    """Whether the published contrast is even inside the model's reach.

    Returns one of:

    ``"reachable"``
        Every level's observation lies between the predictions at the two
        prior bounds, so some gain in the box predicts them and the
        posterior can be interior.
    ``"beyond-upper-bound"`` / ``"beyond-lower-bound"``
        Some level does not, and the posterior will pile against that
        bound. The fit still runs and still converges; what it reports is
        the bound.
    ``"wrong-direction"``
        Some level is on the opposite side of 1.0 from everything the model
        can produce. No gain inside a mechanistically bounded prior moves
        the endpoint the way the animals moved.

    The distinction between the last two is the difference between "this
    circuit is not sensitive enough" and "this circuit has the sign wrong",
    which are different problems with different fixes. Where levels
    disagree, the most severe verdict wins, in the order
    wrong-direction > beyond-* > reachable, because a target is only as
    good as its worst level.
    """
    observed, _ = target.endpoint_ratio()
    lo_ratio, hi_ratio = reachable_ratio_range(curve, target)
    verdicts = []
    for value, low_r, high_r in zip(observed, lo_ratio, hi_ratio):
        low, high = min(low_r, high_r), max(low_r, high_r)
        if low <= value <= high:
            verdicts.append("reachable")
        elif (value - 1.0) * (high_r - 1.0) < 0.0:
            # `lo_ratio` is the null gain, which always predicts exactly 1.
            verdicts.append("wrong-direction")
        else:
            verdicts.append(
                "beyond-upper-bound" if value > high else "beyond-lower-bound"
            )
    for severe in ("wrong-direction", "beyond-lower-bound",
                   "beyond-upper-bound"):
        if severe in verdicts:
            return severe
    return "reachable"


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def build_model(target, curve):
    """The one-parameter `pm.Model` for one target.

    One parameter, because the assay scale that would be the second is
    exactly degenerate with it for a single contrast and only weakly
    separable for a graded series - `information_matrix` measures which,
    per target, instead of this docstring asserting either.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op

    activation = target.activation()
    endpoint_key = target.endpoint_key
    observed, observed_sd = target.endpoint_ratio()

    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def simulate(vec):
        x = float(vec[0]) * activation
        return np.asarray(curve(endpoint_key, x), dtype=float) / efwd.BASELINE_INDEX

    spec = WHOLE_BRAIN_PRIOR_SPECS["gaba_a_whole_brain_modulation_gain"]
    with pm.Model() as model:
        gain = pm.Uniform(
            "gaba_a_whole_brain_modulation_gain", lower=spec[1], upper=spec[2]
        )
        predicted = pm.Deterministic("predicted_ratio", simulate(pt.stack([gain])))
        pm.Normal(
            "%s_ratio_obs" % target.endpoint_key,
            mu=predicted,
            sigma=observed_sd,
            observed=observed,
        )
    return model


def run_fit(target, curve, draws=1000, tune=1000, chains=4, cores=1,
            progressbar=False, seed=SEED):
    """Sample one target and return `(idata, summary)`.

    Same sampler as P2.2 (`DEMetropolisZ`, gradient-free, because the
    forward model is a table lookup over an ODE solve rather than anything
    PyTensor can differentiate) and the same diagnostics.
    """
    import arviz as az
    import pymc as pm

    model = build_model(target, curve)
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            step=pm.DEMetropolisZ(), random_seed=seed, progressbar=progressbar,
        )
    summary = az.summary(
        idata, var_names=["gaba_a_whole_brain_modulation_gain"]
    )
    return idata, summary


def information_matrix(curve, target, gain, scale=1.0, step=1e-3):
    """Fisher information over `(gain, assay scale)` for one target.

    The model with the scale left free is `b = scale * A(gain)`, and this
    returns the 2x2 information matrix of the target's Gaussian
    observations of `b`. For a SINGLE observation the matrix is
    `s s^T / sd^2`, which is RANK ONE for any values whatsoever - the same
    structure P2.3 measured for the amygdala circuit, here for a different
    reason (one datum, not one direction). For a graded series it can have
    rank two, and whether it does in practice is exactly what "a
    dose-response curve buys something a contrast does not" means.

    Returned rather than described so that the eigenvalues can go in the
    results file, where "the scale and the gain are degenerate" is a
    measured spectrum instead of an argument.
    """
    _, lo, hi = WHOLE_BRAIN_PRIOR_SPECS["gaba_a_whole_brain_modulation_gain"]
    up = min(gain + step, hi)
    down = max(gain - step, lo)
    ratio = predicted_ratio(curve, target, gain)
    d_gain = scale * (predicted_ratio(curve, target, up)
                      - predicted_ratio(curve, target, down)) / (up - down)
    d_scale = ratio
    _, sds = target.endpoint_ratio()
    fisher = np.zeros((2, 2))
    for dg, ds, sd in zip(d_gain, d_scale, sds):
        s = np.array([dg, ds], dtype=float)
        fisher += np.outer(s, s) / sd ** 2
    return fisher


__all__ = [
    "SEED",
    "RESULTS_DIR",
    "WHOLE_BRAIN_PRIOR_SPECS",
    "DIAZEPAM_KI_NM",
    "DIAZEPAM_EFFICACY",
    "GABA_A_GENE",
    "GABA_A_MECHANISM",
    "EndpointTarget",
    "TARGETS",
    "TARGETS_BY_KEY",
    "synthetic_target",
    "synthetic_target_from_curve",
    "predicted_ratio",
    "reachable_ratio_range",
    "bound_verdict",
    "build_model",
    "run_fit",
    "information_matrix",
]
