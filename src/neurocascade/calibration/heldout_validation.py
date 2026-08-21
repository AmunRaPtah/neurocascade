"""P8.2: what the model predicts for the held-out compounds, and by how much
each prediction is wrong.

The three tiers, and why they are separate
------------------------------------------

`heldout_forward` explains the split; this module runs it and scores it.

* **Tier 0, coverage.** Before any prediction, which links of the chain
  actually exist for each compound. This turned out to be the headline: see
  `chain_status`.
* **Tier 2, direction.** A parameter-free sign prediction per compound per
  endpoint, scored against the published direction.
* **Tier 3, magnitude.** The reachable range, the implied gain, and a
  monotonicity test that no gain can pass.

There is no tier 1 in this module even though `heldout_forward`'s docstring
names one. Tier 1 is the occupancy comparison, and it needs a dose-to-unbound-
brain-concentration model; `chain_status` reports that no held-out compound
has one, so the tier cannot be run and `occupancy_gap` says which link is
missing for each of the compounds that has an occupancy observation. Leaving
the tier numbered but empty is deliberate: renumbering the tiers to hide the
hole would make the report read as though occupancy had been checked.

The monotonicity test, which is the sharpest thing here
-------------------------------------------------------

Every stage between dose and endpoint index is monotone. Concentration rises
with dose; occupancy `C/(C+Kd)` rises with concentration; activation rises
with occupancy; `x = gain * activation` is monotone in activation for a fixed
gain; and the steady-state index is monotone in `x` over the range these
compounds reach. Composing monotone maps gives a monotone map, so

    the model cannot produce a dose-response that rises and then falls,
    for any gain, at any target, with any density map.

That is a structural prediction with no free parameter in it at all. It is
stronger than a reachable range, because it does not even depend on where the
index saturates. `monotonicity_verdict` tests it against each compound's published
series, and a compound whose published dose-response turns over refutes the
model outright rather than merely calibrating it.

Two published series in this task turn over: alprazolam's elevated-plus-maze
curve and caffeine's biphasic locomotor curve. Whether each turnover survives
its own error bars is computed rather than asserted, because a non-monotone
point estimate inside the noise is not a refutation of anything.
"""

import numpy as np

from .. import fingerprint as _fp
from .. import heldout as _heldout
from . import endpoint_forward as _efwd
from . import heldout_forward as _hf


#: The links a dose-to-behaviour prediction needs, in order. `chain_status`
#: reports one verdict per link per compound, and the point of the list is
#: that the report can state how many compounds have ALL of them.
CHAIN_LINKS = ("binding", "pharmacokinetics", "brain_partition",
               "behavioural_path", "observation")


class ChainStatus(object):
    """Which links of the dose-to-behaviour chain a compound actually has."""

    __slots__ = ("compound", "links", "detail")

    def __init__(self, compound, links, detail):
        self.compound = compound
        self.links = dict(links)
        self.detail = dict(detail)

    @property
    def complete(self):
        return all(self.links[link] for link in CHAIN_LINKS)

    @property
    def missing(self):
        return tuple(link for link in CHAIN_LINKS if not self.links[link])

    def __repr__(self):
        return "ChainStatus(%s, missing=%s)" % (
            self.compound, ",".join(self.missing) or "nothing"
        )


def chain_status(compound, expression_genes=None):
    """One `ChainStatus` for one `heldout.HeldOutCompound`.

    `pharmacokinetics` asks `drug_pk` whether it can build a model for any
    (species, route) this compound has an observation for, rather than
    whether some PK number was found in the literature: a half-life without a
    volume of distribution cannot turn a mg/kg dose into a concentration, and
    counting it as "PK available" would make the coverage table say the
    opposite of what the code can do.
    """
    from ..legacy import drug_pk as _drug_pk

    if expression_genes is None:
        expression_genes = {
            spec.gene for spec in _fp.TARGETS.values() if spec.gene
        }

    detail = {}

    targets = compound.modelled_targets
    detail["binding"] = (
        "%d modelled target(s): %s" % (len(targets), ", ".join(targets))
        if targets else "no modelled target with a measured Ki"
    )

    pairs = sorted({
        (obs.species, obs.route)
        for obs in compound.behavioural_observations + compound.occupancy_observations
    })
    buildable, reasons = [], []
    for species, route in pairs:
        try:
            _drug_pk.pk_model_for(
                compound.name, species, route, body_weight_kg=0.25
            )
        except Exception as exc:
            reasons.append("%s/%s: %s" % (species, route, type(exc).__name__))
        else:
            buildable.append("%s/%s" % (species, route))
    detail["pharmacokinetics"] = (
        "buildable for " + ", ".join(buildable) if buildable
        else "drug_pk builds no model for any observed (species, route): "
             + "; ".join(reasons or ["no observation to build one for"])
    )

    partition = _drug_pk.brain_partition_status(compound.name, "rat")
    detail["brain_partition"] = "%s - %s" % (partition[0], partition[1])

    paths = {}
    for target in targets:
        spec = _fp.TARGETS[target]
        if spec.is_transporter:
            paths[target] = "transporter-no-whole-brain-field"
            continue
        kind = compound.efficacy[target].kind if target in compound.efficacy \
            else compound.efficacy[_raw_key(compound, target)].kind
        # A sign is all that is needed here, and it follows from the kind
        # rather than from a concentration: an agonist adds activation, an
        # antagonist removes tone, and with no tone it removes nothing.
        delta_sign = _delta_activation_sign(target, kind)
        paths[target] = _hf.behavioural_path_status(
            target, 0.0 if delta_sign == 0 else float(delta_sign),
            expression_genes,
        )
    detail["behavioural_path"] = ", ".join(
        "%s: %s" % (t, s) for t, s in sorted(paths.items())
    ) or "no target"

    quantitative = [
        obs for obs in compound.behavioural_observations if obs.is_quantitative
    ]
    detail["observation"] = (
        "%d quantitative behavioural observation(s)" % len(quantitative)
        if quantitative else
        "%d behavioural observation(s), none with both a ratio and a "
        "dispersion" % len(compound.behavioural_observations)
    )

    links = {
        "binding": bool(targets),
        "pharmacokinetics": bool(buildable),
        "brain_partition": partition[0] == "receptor-drivable",
        "behavioural_path": any(v == "available" for v in paths.values()),
        "observation": bool(quantitative),
    }
    return ChainStatus(compound.name, links, detail)


def _raw_key(compound, canonical):
    for record in compound.ki:
        if not record.is_censored and _fp.is_modelled(record.target) \
                and _fp.resolve_target(record.target) == canonical:
            return record.target
    raise KeyError(canonical)


#: How each functional class changes a site's activation, as a SIGN only.
#: `+1` adds activation, `-1` removes whatever the endogenous tone was
#: providing, `0` reaches the circuit not at all.
_DELTA_SIGN_BY_KIND = {
    "agonist": +1,
    "positive_allosteric_modulator": +1,
    "antagonist": -1,
    "inverse_agonist": -1,
    "channel_blocker": -1,
    "uptake_blocker": 0,
    "substrate_releaser": 0,
}


def _delta_activation_sign(target, kind):
    """`sign(delta_activation)` from the functional class and the tone.

    A negative-going class at a site with NO measured endogenous tone gives
    zero, not a negative: there is nothing there to displace. That is the
    single fact that decides whether an antagonist is predictable at all in
    this model, and it is computed here rather than assumed per compound.
    """
    sign = _DELTA_SIGN_BY_KIND[kind]
    if sign < 0:
        has_tone = any(
            entry.name == target for entry in _tone_targets()
        )
        return -1 if has_tone else 0
    return sign


def _tone_targets():
    from ..legacy import endogenous_tone as _tone

    return _tone.TONE_TARGETS


# ---------------------------------------------------------------------------
# Tier 2: direction
# ---------------------------------------------------------------------------


class DirectionPrediction(object):
    """A parameter-free sign prediction for one compound at one endpoint."""

    __slots__ = ("compound", "endpoint", "target", "predicted", "observed",
                 "reason", "x_sign", "slope")

    def __init__(self, compound, endpoint, target, predicted, observed,
                 reason, x_sign=0, slope=0.0):
        self.compound = compound
        self.endpoint = endpoint
        self.target = target
        self.predicted = predicted
        self.observed = observed
        self.reason = reason
        self.x_sign = x_sign
        self.slope = slope

    @property
    def scorable(self):
        """True when both a prediction and an observation exist.

        A prediction of `no-change` IS scorable: "this model says caffeine
        does nothing" is a claim that can be wrong, and counting it as
        unscorable would let the model's least defensible prediction escape
        being marked.
        """
        return self.predicted is not None and self.observed is not None

    @property
    def correct(self):
        return self.scorable and self.predicted == self.observed

    def __repr__(self):
        return "DirectionPrediction(%s/%s, predicted=%s, observed=%s)" % (
            self.compound, self.endpoint, self.predicted, self.observed
        )


def delta_activation_sign_for(compound, target):
    """`sign(delta_activation)` for one compound at one of its targets."""
    return _delta_activation_sign(
        target, compound.efficacy[_raw_key(compound, target)].kind
    )


def behavioural_path_status_for(compound, target):
    """`heldout_forward.behavioural_path_status` for one compound-target pair.

    Wraps the sign lookup so callers do not have to reach into this module's
    private helpers to ask the one question they actually want answered.
    """
    kind = compound.efficacy[_raw_key(compound, target)].kind
    sign = _delta_activation_sign(target, kind)
    genes = {spec.gene for spec in _fp.TARGETS.values() if spec.gene}
    return _hf.behavioural_path_status(
        target, 0.0 if sign == 0 else float(sign), genes
    )


def endpoint_direction(response, endpoint_key, x_sign, tolerance=1e-9):
    """`("increase"|"decrease"|"no-change", slope)` for one endpoint.

    The slope is taken at the smallest tabulated `|x|` rather than at a large
    one, because the sign of the response near zero is the sign of the effect
    at any dose a real compound reaches, while the index is not guaranteed
    monotone all the way out to saturation.
    """
    if x_sign == 0:
        return "no-change", 0.0
    grid = response.x_grid
    small = float(np.min(np.abs(grid[grid != 0.0])))
    x = small * float(np.sign(x_sign))
    delta = float(response(endpoint_key, x)) - response.baseline[endpoint_key]
    if abs(delta) <= tolerance:
        return "no-change", delta / x
    return ("increase" if delta > 0 else "decrease"), delta / x


# ---------------------------------------------------------------------------
# Tier 3: magnitude, and the monotonicity test
# ---------------------------------------------------------------------------


class MonotonicityVerdict(object):
    """Whether a published dose-response turns over, and whether that matters.

    `turns_over` is about the point estimates; `significant` asks whether the
    turnover survives the published dispersions, by requiring the descending
    step to exceed the combined standard deviation of its two endpoints. A
    turnover inside the noise refutes nothing and is reported as such.
    """

    __slots__ = ("compound", "endpoint", "doses", "ratios", "sds",
                 "turns_over", "significant", "note")

    def __init__(self, compound, endpoint, doses, ratios, sds, turns_over,
                 significant, note):
        self.compound = compound
        self.endpoint = endpoint
        self.doses = tuple(doses)
        self.ratios = tuple(ratios)
        self.sds = tuple(sds)
        self.turns_over = bool(turns_over)
        self.significant = bool(significant)
        self.note = note

    def __repr__(self):
        return "MonotonicityVerdict(%s/%s, turns_over=%s, significant=%s)" % (
            self.compound, self.endpoint, self.turns_over, self.significant
        )


def monotonicity_verdict(compound, endpoint, observations, n_sigma=1.0):
    """Test one compound's published series against the model's monotonicity.

    Needs three or more dose levels: with two you cannot tell a turnover from
    a slope. Returns None when there are fewer, which is the usual case and
    is itself worth counting in the report.
    """
    usable = sorted(
        (o for o in observations
         if o.endpoint == endpoint and o.is_quantitative),
        key=lambda o: o.dose_mg_per_kg,
    )
    if len(usable) < 3:
        return None

    doses = [o.dose_mg_per_kg for o in usable]
    ratios = [o.ratio for o in usable]
    sds = [o.ratio_sd for o in usable]

    deltas = np.diff(ratios)
    turns_over = bool(np.any(deltas > 0) and np.any(deltas < 0))

    significant = False
    if turns_over:
        # The turnover counts only if the rise and the fall each exceed the
        # combined uncertainty of their own endpoints. Testing only the larger
        # step would call a monotone series with one noisy point a turnover.
        rises, falls = [], []
        for i, delta in enumerate(deltas):
            combined = np.hypot(sds[i], sds[i + 1])
            resolved = abs(delta) > n_sigma * combined
            (rises if delta > 0 else falls).append(resolved)
        significant = any(rises) and any(falls)

    return MonotonicityVerdict(
        compound=compound, endpoint=endpoint, doses=doses, ratios=ratios,
        sds=sds, turns_over=turns_over, significant=significant,
        note=(
            "monotone in dose; consistent with the model's structure"
            if not turns_over else
            "turns over, and the turnover exceeds %g sd on both the rising "
            "and the falling step: NO gain reproduces this" % n_sigma
            if significant else
            "turns over in the point estimates, and the turnover does NOT "
            "survive the published dispersions, so it refutes nothing"
        ),
    )


class MagnitudeResult(object):
    """One compound-endpoint-dose magnitude comparison, with its error."""

    __slots__ = ("compound", "endpoint", "target", "dose_mg_per_kg",
                 "observed_ratio", "observed_sd", "reachable", "implied_x",
                 "verdict", "residual_sd")

    def __init__(self, compound, endpoint, target, dose_mg_per_kg,
                 observed_ratio, observed_sd, reachable, implied_x, verdict,
                 residual_sd):
        self.compound = compound
        self.endpoint = endpoint
        self.target = target
        self.dose_mg_per_kg = dose_mg_per_kg
        self.observed_ratio = observed_ratio
        self.observed_sd = observed_sd
        self.reachable = reachable
        self.implied_x = implied_x
        self.verdict = verdict
        self.residual_sd = residual_sd

    def __repr__(self):
        return "MagnitudeResult(%s/%s@%g, %s)" % (
            self.compound, self.endpoint, self.dose_mg_per_kg, self.verdict
        )


#: What a magnitude comparison can conclude.
#:
#: ``out-of-reach``
#:     The observation lies outside the index's range over ALL gains. A
#:     structural refutation: no parameter choice reaches it.
#: ``reachable-gain-unidentified``
#:     Some gain reproduces it. This is NOT a successful prediction - it is
#:     the model being fitted to the observation after the fact, and
#:     `implied_x` is the value that fit would need.
MAGNITUDE_VERDICTS = ("out-of-reach", "reachable-gain-unidentified")


def magnitude_result(compound, target, response, observation):
    """Score one quantitative observation against one target's response curve.

    `residual_sd` is the error this task reports per compound per endpoint:
    how far the NEAREST prediction the model can make sits from the
    observation, in units of the observation's own standard deviation. It is
    zero whenever the observation is reachable, and that zero means "the model
    can be fitted to this", not "the model predicted this".
    """
    ranges = _hf.reachable_index_range(response)
    reachable = ranges[observation.endpoint]
    shortfall = reachable.shortfall(observation.ratio)
    if shortfall > 0.0:
        return MagnitudeResult(
            compound=compound, endpoint=observation.endpoint, target=target,
            dose_mg_per_kg=observation.dose_mg_per_kg,
            observed_ratio=observation.ratio, observed_sd=observation.ratio_sd,
            reachable=reachable, implied_x=None, verdict="out-of-reach",
            residual_sd=shortfall / observation.ratio_sd,
        )
    return MagnitudeResult(
        compound=compound, endpoint=observation.endpoint, target=target,
        dose_mg_per_kg=observation.dose_mg_per_kg,
        observed_ratio=observation.ratio, observed_sd=observation.ratio_sd,
        reachable=reachable,
        implied_x=_hf.implied_x(response, observation.endpoint, observation.ratio),
        verdict="reachable-gain-unidentified", residual_sd=0.0,
    )


# ---------------------------------------------------------------------------
# The tone ceiling, which bounds several compounds before any dose is chosen
# ---------------------------------------------------------------------------


def tone_ceiling():
    """`{target: max |delta_activation| an antagonist can ever produce}`.

    A competitive antagonist can at most remove the activation the endogenous
    transmitter was supplying, so its `|delta_activation|` is bounded by the
    tonic activation - by the MEASURED tone, not by the drug's own affinity.
    For most targets that bound is tiny: P6.2's histamine tone sits ~300-fold
    below H1's Kd, so H1 blockade can move activation by at most 0.3%,
    however potent the blocker.

    This is a ceiling on the compound's reach that is fixed before any dose,
    any gain or any density is chosen, and it is the reason several
    antagonists in this set are predicted to do almost nothing even where a
    tone exists.
    """
    from ..legacy import endogenous_tone as _tone

    return {
        entry.name: float(_tone.tonic_activation(entry))
        for entry in _tone.TONE_TARGETS
    }


def occupancy_gap(compound):
    """Why each occupancy observation cannot be scored, per compound.

    Tier 1 is the one tier whose prediction has no unfitted parameter in it,
    so naming exactly what blocks it matters more than for the other two.
    """
    status = chain_status(compound)
    blocked = [
        link for link in ("pharmacokinetics", "brain_partition")
        if not status.links[link]
    ]
    return {
        "observations": len(compound.occupancy_observations),
        "blocked_by": tuple(blocked),
        "detail": {link: status.detail[link] for link in blocked},
    }


__all__ = [
    "CHAIN_LINKS",
    "MAGNITUDE_VERDICTS",
    "ChainStatus",
    "DirectionPrediction",
    "MagnitudeResult",
    "MonotonicityVerdict",
    "behavioural_path_status_for",
    "chain_status",
    "delta_activation_sign_for",
    "endpoint_direction",
    "magnitude_result",
    "monotonicity_verdict",
    "occupancy_gap",
    "tone_ceiling",
]
