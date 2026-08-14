"""P7.3: fitting the P7.1 endpoints against published dose-response data.

The brief, and the three things that had to be true
---------------------------------------------------

    Fit each behavioral endpoint against real published dose-response
    curves through the Phase 1 PK bridge ... remove the illustrative label
    only from parameters actually fit.

For one endpoint to be fitted, three things have to hold at once, and this
module's whole structure is the bookkeeping of which of them fail where:

1. **A dataset.** Real, published, in a species and by a route
   `drug_pk` models, with per-group numbers and a dispersion. That is
   `behavioral_evidence.py`'s job and `ENDPOINT_COVERAGE` is its answer.
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
unknowns, one equation. Every dataset in this module except one is a
two-point contrast, and the one exception is a curve whose own two
published parameters face three model parameters. So there is no dataset in
reach from which a gain and a scale can BOTH be identified.

P2.2 met the same wall and got past it by ASSUMING `k` away: if `b = k * A`
with the same `k` in both groups, the drug/vehicle RATIO is `A_drug /
A_vehicle` and `k` cancels. That assumption is what this module inherits,
and it is why every fit here is a fit to a ratio. It is not free - see
`circuit_forward.py` and P2.2's section - and at whole-brain scale it needs
one more thing P2.2 did not: an index whose zero means something. That is
`endpoint_forward.directed_activity_index`, and the fact that it equals
exactly 0.5 at rest for all seven endpoints is what keeps these ratios off
a near-zero denominator.

What the fits found
-------------------

Every fit below CONVERGES and none of them licenses a number. The two
failure modes are different and both are informative:

* **anxiety**: the model moves the right way and not far enough. At the
  largest gain the mechanism allows, a whole-brain benzodiazepine moves the
  anxiety index by ~4% where [Graziani2022] measured ~22%. The posterior
  therefore piles against its upper bound, which is exactly the situation
  P2.2 refused to report as a fit ("the reported fit would have been partly
  a description of the box").
* **locomotion and sedation**: the model moves the WRONG WAY. Potentiating
  inhibition brain-wide lowers the locomotion index, and [Graziani2022]'s
  rats crossed MORE lines under diazepam. The posterior piles against the
  mechanistic null at zero.

Both are reported with their diagnostics, and neither removes a label.
`docs/architecture.md`'s P7.3 section works through why, and what each
would need.
"""

import os

import numpy as np

from .. import endpoints as _endpoints
from . import behavioral_evidence as bev
from . import endpoint_forward as efwd
from . import evidence as ev
from . import forward as fwd

SEED = 20260814

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "results", "calibration"
)

#: The whole-brain gains P7.3 fits. NOT in `priors.PRIOR_SPECS`, and that is
#: deliberate: `priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` is the register of
#: circuit-layer parameters this project HAS calibrated, guarded by a test
#: that requires each of them to name a calibration-ready dataset. None of
#: the gains below survives its own fit (see the module docstring), so
#: putting them there would be claiming a calibration that did not happen.
#: They live here, with their bounds and the reason for each bound, and
#: `tests/test_endpoint_calibration.py` asserts they stay out of `priors`.
#:
#: `gaba_a_whole_brain_modulation_gain` is a DIFFERENT parameter from
#: `priors.gaba_a_connection_modulation_gain`, and the difference matters:
#: P2.2's gain potentiates ONE synapse in a three-node amygdala circuit,
#: this one potentiates every inhibitory connection in a 291-region brain,
#: weighted by each region's Gabra1 density. They are not comparable and
#: neither is evidence about the other.
#:
#: Bounds, from the mechanism rather than from a convention, exactly as
#: P2.2 argued for its own:
#:   lower 0.0 - the null, "diazepam does not potentiate GABAergic
#:     transmission", which predicts a ratio of exactly 1 for every
#:     endpoint and every circuit. A meaningful, falsifiable value.
#:   upper 5.0 - at the near-saturating occupancy [Graziani2022]'s dose
#:     produces, a gain of 5 multiplies every inhibitory weight by about
#:     5.5. Benzodiazepines potentiate GABA-A currents by roughly two-fold;
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

    @property
    def entry(self):
        return bev.BEHAVIORAL_EVIDENCE_BY_KEY[self.evidence_key]

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


def _diazepam_exposure_nM():
    """[Graziani2022]'s 2 mg/kg IP dose as unbound brain diazepam, nM.

    Computed by P2.2's own bridge call so the two tasks cannot disagree
    about the number, and at P2.2's own stated 60 min timepoint.
    """
    return ev.gaba_a_epm_unbound_brain_nM()


#: Diazepam's Ki at the benzodiazepine site, the value P2.2's posterior sat
#: on (3.60 nM, [MohlerOkada1977] via `priors.PRIOR_SPECS`), used here as a
#: fixed constant rather than resampled: the site is at 90% occupancy at
#: this exposure, so the ratio depends on the Ki far too weakly for a
#: whole-brain fit to say anything new about it, and P2.2 already reported
#: that its own update was weak "rather than overstated".
DIAZEPAM_KI_NM = 3.6

#: Diazepam's efficacy at the benzodiazepine site in `scenarios.py`, where
#: efficacy carries potentiation strength for a connection_modulation-
#: coupled site (see that module).
DIAZEPAM_EFFICACY = 1.0


def _graziani_targets():
    """The three single-dose targets [Graziani2022] supports, built from
    the evidence module rather than restated here."""
    exposure = _diazepam_exposure_nM()
    dose = max(bev.DIAZEPAM_EPM_GRAZIANI2022.doses_mg_per_kg)
    epm = bev.DIAZEPAM_EPM_GRAZIANI2022.effect
    field = bev.DIAZEPAM_OPEN_FIELD_GRAZIANI2022.effect

    anxiety_ratio, anxiety_sd = bev.anxiolysis_ratio(
        epm,
        vehicle_prefix="vehicle",
        drug_prefix="diazepam 2 mg/kg",
        trial_duration_s=bev.GRAZIANI2022_EPM_TRIAL_DURATION_S,
    )
    crossings = bev.measure_ratio(
        field, "vehicle crossings", "diazepam 2 mg/kg crossings"
    )
    immobility = bev.measure_ratio(
        field, "vehicle immobility_s", "diazepam 2 mg/kg immobility_s"
    )
    common = dict(
        gene=GABA_A_GENE,
        mechanism=GABA_A_MECHANISM,
        exposure_timepoint_min=ev.GABA_A_EPM_EXPOSURE_TIMEPOINT_MIN,
        Ki_nM=DIAZEPAM_KI_NM,
        efficacy=DIAZEPAM_EFFICACY,
    )
    return (
        EndpointTarget(
            key="anxiety/diazepam/Graziani2022",
            endpoint_key="anxiety",
            evidence_key=bev.DIAZEPAM_EPM_GRAZIANI2022.key,
            measure="open-arm avoidance, 1 - open-arm time / trial duration",
            levels=((dose, exposure, anxiety_ratio, anxiety_sd),),
            # P7.1's anxiety endpoint is "higher = MORE anxious", and
            # open-arm avoidance rises with anxiety too.
            measure_direction=1,
            note=(
                "The same contrast P2.2 fitted the amygdala scenario's "
                "single-synapse gain against, read here through the "
                "whole-brain endpoint instead. The two gains are different "
                "parameters and are not compared as numbers."
            ),
            **common
        ),
        EndpointTarget(
            key="locomotion/diazepam/Graziani2022",
            endpoint_key="locomotion",
            evidence_key=bev.DIAZEPAM_OPEN_FIELD_GRAZIANI2022.key,
            measure="open-field crossing episodes",
            levels=((dose, exposure) + crossings,),
            # P7.1 locomotion is "higher = more locomotion (greater
            # distance travelled)"; crossings rise with distance.
            measure_direction=1,
            note="n = 4 in the drug group; see the entry.",
            **common
        ),
        EndpointTarget(
            key="sedation/diazepam/Graziani2022",
            endpoint_key="sedation",
            evidence_key=bev.DIAZEPAM_OPEN_FIELD_GRAZIANI2022.key,
            measure="open-field immobility time",
            levels=((dose, exposure) + immobility,),
            # P7.1 sedation is "higher = MORE sedated", and its declared
            # assay list includes "reduced spontaneous locomotion".
            # Immobility rises with sedation, so the two agree.
            measure_direction=1,
            note=(
                "The same four animals as the locomotion target, in the "
                "same five minutes, read in the opposite direction. Not an "
                "independent dataset, and reported as one contrast counted "
                "twice rather than two."
            ),
            **common
        ),
    )


def _graded_levels(entry, vehicle_key, drug_keys, body_weight_kg,
                   sd_scale=1.0):
    """`((dose, unbound brain nM, ratio, sd), ...)` for a graded series.

    Each dose is pushed through the Phase 1 bridge separately - the whole
    point of a graded series is that the exposures differ - at the entry's
    own stated timepoint, which for both graded entries the source DOES
    state for the drug by name.

    `sd_scale` divides the propagated dispersion, and exists for exactly
    one case: [Lustyk2023] prints SDs, and a ratio of group MEANS needs the
    sd of the mean, i.e. SD/sqrt(n). Passing it explicitly keeps that
    division at the call site, where the n it used can be named, rather
    than hidden in `measure_ratio`.
    """
    levels = []
    for dose, drug_key in zip(entry.doses_mg_per_kg, drug_keys):
        exposure = entry.brain_exposure(
            body_weight_kg, dose_mg_per_kg=dose,
            timepoint_min=entry.timepoint_min,
        )
        if not exposure.drives_receptor:
            raise RuntimeError(
                "%s at %g mg/kg yielded a %s concentration, which may not "
                "drive a receptor" % (entry.key, dose, exposure.brain_basis)
            )
        ratio, sd = bev.measure_ratio(entry.effect, vehicle_key, drug_key)
        levels.append((dose, exposure.brain_nM, ratio, sd / sd_scale))
    return tuple(levels)


def _graded_targets():
    """The two graded targets, which are what P7.3 was actually asking for:
    a dose-RESPONSE rather than a contrast."""
    sagara = bev.DIAZEPAM_LOCOMOTION_SAGARA2008
    lustyk = bev.DIAZEPAM_HOT_PLATE_LUSTYK2023
    sagara_weight = sum(bev.SAGARA2008_BODY_WEIGHT_BAND_KG) / 2.0
    lustyk_weight = sum(bev.LUSTYK2023_BODY_WEIGHT_BAND_KG) / 2.0
    common = dict(
        gene=GABA_A_GENE,
        mechanism=GABA_A_MECHANISM,
        Ki_nM=DIAZEPAM_KI_NM,
        efficacy=DIAZEPAM_EFFICACY,
    )
    return (
        EndpointTarget(
            key="locomotion/diazepam/Sagara2008",
            endpoint_key="locomotion",
            evidence_key=sagara.key,
            measure="ANIMEX locomotor counts in 10 min",
            levels=_graded_levels(
                sagara, "vehicle",
                ["diazepam 0.2 mg/kg", "diazepam 0.5 mg/kg", "diazepam 1.0 mg/kg"],
                sagara_weight,
            ),
            exposure_timepoint_min=sagara.timepoint_min,
            measure_direction=1,
            note=(
                "Three doses and a vehicle, SEMs, n = 10 per group, and a "
                "null result. The only graded, tabulated, bridgeable "
                "behavioural dose-response found for any of the seven "
                "endpoints."
            ),
            **common
        ),
        EndpointTarget(
            key="analgesia/diazepam/Lustyk2023",
            endpoint_key="analgesia",
            evidence_key=lustyk.key,
            measure="hot-plate time of reaction",
            levels=_graded_levels(
                lustyk, "vehicle",
                ["diazepam 5 mg/kg", "diazepam 10 mg/kg"],
                lustyk_weight,
                sd_scale=np.sqrt(bev.LUSTYK2023_MIN_GROUP_SIZE),
            ),
            exposure_timepoint_min=lustyk.timepoint_min,
            # P7.1 analgesia is "higher = MORE analgesia = LONGER
            # withdrawal latency"; hot-plate latency rises with analgesia.
            measure_direction=1,
            note=(
                "Two doses and a vehicle, SDs converted to sds of the mean "
                "with the source's SMALLEST stated group size, which is the "
                "widest and therefore the weakest of the widths its band "
                "allows. Also a null."
            ),
            **common
        ),
    )


#: The gene and mechanism every target in this module shares, because every
#: target in it is diazepam: `drug_pk` plus `brain_partition` reach a
#: receptor-driving concentration for exactly one drug/species/route triple
#: (diazepam/rat/IP), which `BehavioralEvidenceEntry.receptor_drive_status`
#: computes rather than this module asserting.
GABA_A_GENE = "Gabra1"
GABA_A_MECHANISM = "connection_modulation"

TARGETS = _graziani_targets() + _graded_targets()
TARGETS_BY_KEY = {t.key: t for t in TARGETS}


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
    "EndpointTarget",
    "TARGETS",
    "TARGETS_BY_KEY",
    "predicted_ratio",
    "reachable_ratio_range",
    "bound_verdict",
    "build_model",
    "run_fit",
    "information_matrix",
]
