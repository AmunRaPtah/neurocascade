"""P3.4: baseline stability and saturation, at whole-brain scale.

    from neurocascade import stability

    assessment = stability.assess(network)     # every node, every check
    assert assessment.ok, assessment.summary()

This module generalizes one lesson that this codebase learned twice by
hand. The first time was the D2 scenario's first parameter draft, which
put GPi's net input far outside `[-1, 1]` before the gain multiplier, so
GPi sat against its sigmoid's ceiling at rest and stopped responding to
anything upstream of it; the drug then moved the readout by almost
nothing. The second time was P3.3's port of that same scenario onto
published weights, where GPe's hand-set `baseline_input = 1.0` met a
striatopallidal density of -12.56 instead of the hand-drawn -2.0 and
GPe was pinned at zero.

At four or five nodes you can find that by printing the rates. At 291 it
has to be a check, and the baseline drive that avoids it has to be a
procedure rather than five hand-chosen numbers. That is what this module
is: the check (`assess`), the procedure (`balanced_baselines`), and the
certificate that says the procedure's answer is not merely *a* fixed
point but the *only* one and that every trajectory reaches it
(`contraction_number`).

Nothing here fits anything. Every quantity below is either derived in
closed form from the model's own equations or is an explicitly labelled
reporting convention, and the one free modelling number
(`DEFAULT_OPERATING_GAIN`) is labelled ILLUSTRATIVE in the same way
`connectome/build.py`'s node parameters are.

The measure of saturation, and why it is this one and not a rate band
---------------------------------------------------------------------

A node is "saturated" if its output has stopped depending on its input.
That is a statement about a derivative, so the honest measure is the
derivative that actually appears when the network is linearized. For

    tau_i dr_i/dt = -r_i + S_i(net_i),   net_i = sum_j W[i,j] r_j + b_i
    S_i(x) = 1 / (1 + exp(-g_i (x - theta_i)))

the Jacobian is

    J = diag(1/tau) (-I + diag(S_i'(net_i)) W)

and at a fixed point, where `r_i = S_i(net_i)`, the slope is the rate's own
quadratic: `S_i'(net_i) = g_i r_i (1 - r_i)`. Every use in this module is
at a fixed point, which is what lets the check be written in terms of rates
rather than of net inputs. (`jacobian()` itself does not assume it, and
takes the slope at the net input, so it is the exact derivative of `rhs`
anywhere.) So `S_i'` is exactly the factor through which everything upstream of node
`i` reaches everything downstream of it. If `S_i'` is zero the node's row
of the Jacobian is dead and the node is a constant, whatever the rest of
the network does. Normalizing by that factor's own maximum, `g_i / 4` at
`r_i = 1/2`, removes the gain and leaves a dimensionless index:

    sensitivity_i = S_i'(net_i) / (g_i / 4) = 4 r_i (1 - r_i)   in (0, 1]

1.0 means the node is at its most responsive point; 0 means it is a rail.
This is preferred over "is the rate inside [0.05, 0.95]" because it is not
a band anyone chose - it is the linearization's own coefficient, written
in units where 1 is the best a node can do.

`SATURATION_FLOOR` is a REPORTING CONVENTION, not a fitted or physical
number, and nothing in this codebase is tuned against it: reports carry
the raw sensitivities so a reader can apply a different floor. It is set
to 0.01 because two independent readings of "saturated" land there:

* a node at `sensitivity = 0.01` transmits 1% of the input change an
  optimally-placed node would, i.e. it is within 0.0025 of a rail;
* solving `4 r (1 - r) = 0.01` gives `|g (net - theta)| = 5.99`, so at
  the illustrative `gain = 4.0` the flag fires at `|net - theta| = 1.50`
  - just outside the `[-1, 1]` band that `docs/architecture.md`'s
  original GPi note already used to describe the failure it saw.

Procedure: assigning baseline drive at scale
--------------------------------------------

Four steps, no fitting, no per-node hand-tuning, and cost that is one
matrix-vector product plus one integration per initial condition.

**1. Choose the operating point `r*`.** The default is `r*_i = 1/2` for
every node, and that is a derivation rather than a preference: `S_i'` is
maximized at `S_i = 1/2`, so `r* = 1/2` is the unique point at which
every node is as far from saturation as it can be. Per-node targets are
supported (step 2 is exact for them) and are the place to spend
region-wise baseline firing-rate data. There is none to spend:
`docs/architecture.md`'s P3.4 section records the search, and the reason
the one whole-brain candidate (the IBL Brain-Wide Map, 279 CCF-registered
mouse areas) cannot be used is structural rather than a coverage gap -
`r_i` is the fraction of a population that is active, not spikes/s, and
converting between them needs a per-region maximum firing rate that does
not exist for 291 regions in any species.

**2. Solve for the drive, in closed form.** `r*` is an exact fixed point
iff `S_i(net_i) = r*_i`, i.e. `net_i = theta_i + logit(r*_i) / g_i`, so

    b_i = theta_i + logit(r*_i) / g_i - sum_j W[i,j] r*_j

which is `balanced_baselines()`. At `r* = 1/2` the logit term vanishes and
this is P3.3's `rest_balanced_baselines()`, whose credentials are that
applied to the hand-built basal-ganglia network's own weights it returns
that network's own `baseline_input` values `(0, 1, 1, -1, 1)` exactly. It
is O(n^2) dense / O(nnz) sparse, one pass, and it is what makes the
procedure scale: nothing is searched for, iterated on, or tuned per node.

**3. Certify the fixed point.** Being an exact fixed point is not being a
stable one. Two conditions, in increasing strength:

    LOCAL   max Re eig(J) < 0,  J = diag(1/tau)(-I + diag(S') W)
            <=> r* is locally asymptotically stable

    GLOBAL  kappa = sigma_max(diag(g/4) W) < 1
            ==> r* is the UNIQUE fixed point and every trajectory from
                every initial condition converges to it

The global condition is the contraction argument of Kelly DG, "Stability
in contractive nonlinear neural networks", IEEE Trans Biomed Eng.
1990;37(3):231-242, doi:10.1109/10.52325 ("if the map WF(x) is a
contraction, the system has a unique equilibrium which is globally
asymptotically stable"), applied to this codebase's rate-form equation.
The derivation in this form is three lines and is worth having in the
file it is used in. Write `u = r - r*`; by the mean value theorem
`S(net(r)) - S(net(r*)) = diag(S'(xi)) W u` for some `xi`, and
`0 < S_i'(xi_i) <= g_i/4` always, so with `A = diag(g/4) W` and
`Lambda = diag(S_i'(xi_i)/(g_i/4))` (entries in `(0, 1]`, hence a
contraction as a diagonal operator),

    V = (1/2) sum_i tau_i u_i^2
    dV/dt = -||u||^2 + u . (Lambda A u) <= -(1 - sigma_max(A)) ||u||^2

so `kappa < 1` gives global exponential convergence, at a rate that does
not depend on the `tau`s at all. The LOCAL criterion is not tau-free in
the same way: `J`'s eigenvalues are those of `-I + diag(S')W` rescaled row
by row, and for a non-symmetric matrix a positive diagonal scaling can
move them, so `spectral_abscissa` computes `eig(J)` itself rather than
reading `max Re eig(diag(S')W) < 1` off (which is the same statement only
when every `tau` is equal, as it happens to be in every network built
here). `kappa` is a sufficient condition and is
genuinely conservative - on the 291-region mouse network it is 2.8x the
local threshold - so `assess()` reports both and never uses one where the
other is meant.

**4. Verify by integration.** Integrate from several spread initial
conditions (`default_initial_conditions`), check that all of them reach
the same point, that `max |dr/dt|` there is at solver noise, and that
every node's sensitivity clears the floor. A certificate that is never
checked against the integrator is a certificate about a different model
than the one this package actually solves.

Where the one free number lives
-------------------------------

Steps 1-4 say whether a given `(W, gain)` has a usable rest state. They do
not say what to do when it does not, and the whole-brain connectome does
not: at `weight_scale = 1.0` and the illustrative `gain = 4.0`, the
291-region network has `kappa = 44.7`. Step 2 still makes `r* = 1/2` an
exact fixed point of it, and step 3 then says that point is a repeller -
integrating away from it lands with 194 of 291 nodes past the saturation
floor, and on a different answer for different initial conditions. (Built
with `build.py`'s own `baseline_input = 0.0` instead, 218 of 291 are past
the floor.) Steps 1-2 without step 3 produce a rest state the model never
visits, which is the whole reason step 3 is in the procedure.

P2.3 established that a coupling gain and a connection weight are
degenerate against behavioural data: only their product is identified. So
the honest reparameterization is to name the product. Define

    operating gain  rho = kappa(weight_scale * W) = weight_scale * kappa(W)

a single dimensionless number that is the network's loop gain at its most
sensitive operating point, and set `weight_scale = rho / kappa(W)`. That
is `stabilize()`. `rho` is ONE ILLUSTRATIVE number replacing the
two-number pair (`weight_scale`, `gain`) that P3.2 deliberately refused to
choose, it is dimensionless and interpretable (`rho < 1` is the
certificate; `rho -> 1` is the edge of instability), and it leaves the
*relative* weights the connectome fixed completely untouched. It is not a
calibration and is not labelled as one: `DEFAULT_OPERATING_GAIN = 0.9` is
ILLUSTRATIVE until behavioural data identifies it.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .circuit import CircuitNode, sigmoid
from .network import CircuitNetwork

#: The rate at which a logistic node's slope is maximal, hence the default
#: operating point. DERIVED, not chosen: S' = g S (1 - S) is maximized at
#: S = 1/2.
MAX_SENSITIVITY_RATE = 0.5

#: REPORTING CONVENTION (see the module docstring for the two readings that
#: put it here). Nothing is fitted to it, and every report carries the raw
#: sensitivities so a different floor can be applied after the fact.
SATURATION_FLOOR = 0.01

#: ILLUSTRATIVE. The dimensionless loop gain `stabilize()` targets. Any
#: value < 1 carries the global-uniqueness certificate; which value is
#: right is a calibration this project has no behavioural data for (P2.3:
#: only the product of coupling gain and connection weight is identified).
DEFAULT_OPERATING_GAIN = 0.9

#: Integration settings for the fixed-point search. `t_max` is generous
#: because convergence slows like 1/(1 - rho) near the certificate's edge:
#: at rho = 0.9 and tau = 1 the slowest mode has a time constant of 10.
DEFAULT_T_MAX = 200.0
DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10

#: `max |dr/dt|` below which an endpoint is called a fixed point, and the
#: distance below which two endpoints are called the same fixed point.
#: Both are numerical, sit ~2 orders above the integrator's own error, and
#: are arguments everywhere they are used.
RESIDUAL_TOLERANCE = 1e-6
FIXED_POINT_TOLERANCE = 1e-5


# --- pointwise quantities --------------------------------------------------


def sensitivity(rates):
    """`4 r (1 - r)`: the sigmoid's slope at `r`, over its maximum.

    Dimensionless, in `(0, 1]`, gain-free. 1 at `r = 1/2`, 0 at a rail.
    """
    r = np.asarray(rates, dtype=float)
    return 4.0 * r * (1.0 - r)


def sigmoid_slopes(network, rates):
    """`g_i r_i (1 - r_i)`, the Jacobian's own coefficient.

    Written in terms of the node's OUTPUT, so it is `S_i'(net_i)` exactly
    when `rates` is a fixed point (`r_i = S_i(net_i)`) and is the slope at
    the rate `r_i` otherwise. `jacobian()` is the version that does not
    assume a fixed point.
    """
    r = np.asarray(rates, dtype=float)
    gains = np.array([node.gain for node in network.nodes], dtype=float)
    return gains * r * (1.0 - r)


def net_inputs(network, rates, drive=None):
    """`sum_j W[i,j] r_j + drive_i`, i.e. the sigmoid's argument."""
    r = np.asarray(rates, dtype=float)
    d = network.baseline_drive() if drive is None else np.asarray(drive, dtype=float)
    return network.connectivity @ r + d


def operating_matrix(network, rates=None):
    """`A = diag(S_i'(r_i)) W`, the network's linearization coefficient.

    `rates=None` means the worst case for stability, `r_i = 1/2` for every
    node, where `S_i' = g_i/4` is maximal. That is the matrix the global
    certificate is about, because `S_i' <= g_i/4` everywhere makes the
    `r = 1/2` evaluation an upper bound over the whole state space, not a
    guess about where the network sits.
    """
    if rates is None:
        rates = np.full(network.n, MAX_SENSITIVITY_RATE)
    return sigmoid_slopes(network, rates)[:, None] * network.connectivity


def contraction_number(network, rates=None):
    """`kappa = sigma_max(diag(g/4) W)`. `kappa < 1` is the certificate.

    See the module docstring for the Lyapunov argument and for Kelly 1990.
    `kappa < 1` implies the network has exactly one fixed point and that
    every trajectory converges to it exponentially, whatever the `tau`s.
    """
    A = operating_matrix(network, rates)
    if A.size == 0:
        return 0.0
    return float(np.linalg.svd(A, compute_uv=False)[0])


def jacobian(network, rates, drive=None):
    """`J = diag(1/tau) (-I + diag(S'(net)) W)`, the exact Jacobian of `rhs`.

    The slope is taken at the net input, not at the rate: away from a fixed
    point `S(net) != r` and using `r` would give the derivative of a
    different function. AT a fixed point the two coincide, so there
    `jacobian` equals `diag(1/tau)(-I + operating_matrix(r))` - which is the
    only place this module ever evaluates it, and is why the docstring's
    `S' = g r (1-r)` is written that way.
    """
    taus = np.array([node.tau for node in network.nodes], dtype=float)
    outputs = np.array(
        [
            sigmoid(x, gain=node.gain, threshold=node.threshold)
            for x, node in zip(net_inputs(network, rates, drive), network.nodes)
        ]
    )
    gains = np.array([node.gain for node in network.nodes], dtype=float)
    A = (gains * outputs * (1.0 - outputs))[:, None] * network.connectivity
    return (A - np.eye(network.n)) / taus[:, None]


def spectral_abscissa(network, rates, drive=None):
    """`max Re eig(J)`: negative means locally asymptotically stable."""
    if network.n == 0:
        return 0.0
    return float(np.linalg.eigvals(jacobian(network, rates, drive)).real.max())


def critical_weight_scale(network):
    """The `weight_scale` at which the global certificate is exactly lost.

    `1 / kappa`, or infinity for a network with no edges (which is already
    a contraction at any scale).
    """
    kappa = contraction_number(network)
    return np.inf if kappa == 0.0 else 1.0 / kappa


def weight_scale_for_operating_gain(network, operating_gain=DEFAULT_OPERATING_GAIN):
    """The `weight_scale` that gives the network loop gain `rho`.

    `rho / kappa(W)`, or 1.0 for an edgeless network. `rho` is
    ILLUSTRATIVE; see the module docstring's last section for why it, and
    not `weight_scale` and `gain` separately, is the number to set.
    """
    if operating_gain <= 0.0:
        raise ValueError("operating_gain must be positive, got %r" % (operating_gain,))
    kappa = contraction_number(network)
    return 1.0 if kappa == 0.0 else float(operating_gain) / kappa


# --- the procedure ---------------------------------------------------------


def balanced_baselines(network, target_rates=MAX_SENSITIVITY_RATE):
    """Step 2 of the procedure: `baseline_input` making `r*` exact.

        b_i = theta_i + logit(r*_i) / g_i - sum_j W[i,j] r*_j

    `target_rates` is a scalar or a per-node array. Returns a new
    `CircuitNetwork` over new `CircuitNode`s; the input network, its nodes
    and its connectivity array are not modified.

    At the default `r* = 1/2` the logit term is zero and this is P3.3's
    `connectome_scenarios.rest_balanced_baselines`, which still exists and
    now delegates here. Its credential is worth repeating: applied to the
    hand-built basal-ganglia network's own weights it reproduces that
    network's own `baseline_input` values `(0, 1, 1, -1, 1)` exactly, so
    this rule is a restatement of a choice this codebase already made by
    hand, not a new modelling decision introduced at scale.
    """
    targets = np.broadcast_to(
        np.asarray(target_rates, dtype=float), (network.n,)
    ).astype(float)
    if np.any(targets <= 0.0) or np.any(targets >= 1.0):
        raise ValueError(
            "target rates must lie strictly inside (0, 1) - a logistic node "
            "reaches 0 or 1 only at infinite drive; got min=%g max=%g"
            % (targets.min(), targets.max())
        )
    W = network.connectivity
    nodes = []
    for i, node in enumerate(network.nodes):
        offset = np.log(targets[i] / (1.0 - targets[i])) / node.gain
        nodes.append(
            CircuitNode(
                name=node.name,
                tau=node.tau,
                baseline_input=node.threshold + offset - float(W[i] @ targets),
                gain=node.gain,
                threshold=node.threshold,
            )
        )
    return CircuitNetwork(nodes=nodes, connectivity=W)


def rescale_weights(network, scale):
    """A copy of `network` with every connection weight multiplied by `scale`.

    Node parameters come over unchanged, which means the baselines do NOT
    follow the weights: rescaling then rebalancing is the correct order and
    is what `stabilize()` does.
    """
    nodes = [
        CircuitNode(
            name=node.name,
            tau=node.tau,
            baseline_input=node.baseline_input,
            gain=node.gain,
            threshold=node.threshold,
        )
        for node in network.nodes
    ]
    return CircuitNetwork(nodes=nodes, connectivity=float(scale) * network.connectivity)


class StabilizedNetwork(object):
    """What `stabilize()` did, as values rather than as prose."""

    def __init__(
        self,
        network,
        source_network,
        weight_scale,
        operating_gain,
        target_rates,
        contraction_before,
        contraction_after,
    ):
        self.network = network
        self.source_network = source_network
        self.weight_scale = weight_scale
        self.operating_gain = operating_gain
        self.target_rates = target_rates
        self.contraction_before = contraction_before
        self.contraction_after = contraction_after

    @property
    def certified(self):
        return self.contraction_after < 1.0

    def summary(self):
        return "\n".join(
            [
                "stabilized network: %d nodes" % self.network.n,
                "  operating gain rho = %.4g (ILLUSTRATIVE)" % self.operating_gain,
                "  weight_scale = %.6g   (= rho / kappa, kappa = %.6g)"
                % (self.weight_scale, self.contraction_before),
                "  contraction number after scaling: %.6g %s"
                % (
                    self.contraction_after,
                    "< 1, unique globally attracting fixed point"
                    if self.certified
                    else ">= 1, NO global certificate",
                ),
                "  baseline_input: balanced to r* = %s"
                % np.array2string(
                    np.unique(
                        np.round(
                            np.broadcast_to(
                                np.asarray(self.target_rates, dtype=float),
                                (self.network.n,),
                            ),
                            6,
                        )
                    ),
                    separator=", ",
                    threshold=8,
                ),
            ]
        )


def stabilize(
    network,
    operating_gain=DEFAULT_OPERATING_GAIN,
    target_rates=MAX_SENSITIVITY_RATE,
):
    """Rescale the weights to loop gain `rho`, then balance the baselines.

    The order matters: `balanced_baselines` solves against the weights it
    is given, so rescaling afterwards would break the fixed point it just
    constructed. Relative weights are untouched - only one global scalar
    and the `n` baselines change.
    """
    scale = weight_scale_for_operating_gain(network, operating_gain)
    scaled = rescale_weights(network, scale)
    balanced = balanced_baselines(scaled, target_rates)
    return StabilizedNetwork(
        network=balanced,
        source_network=network,
        weight_scale=scale,
        operating_gain=operating_gain,
        target_rates=target_rates,
        contraction_before=contraction_number(network),
        contraction_after=contraction_number(balanced),
    )


# --- the check -------------------------------------------------------------


class NodeState(object):
    """One node at one operating point."""

    def __init__(self, name, rate, net_input, slope, sensitivity, floor):
        self.name = name
        self.rate = rate
        self.net_input = net_input
        self.slope = slope
        self.sensitivity = sensitivity
        self.floor = floor

    @property
    def saturated(self):
        return self.sensitivity < self.floor

    @property
    def rail(self):
        """Which rail a saturated node is against; None if it is not."""
        if not self.saturated:
            return None
        return "floor" if self.rate < MAX_SENSITIVITY_RATE else "ceiling"

    def __repr__(self):
        return "NodeState(%r, rate=%.6g, sensitivity=%.4g%s)" % (
            self.name,
            self.rate,
            self.sensitivity,
            ", SATURATED at the %s" % self.rail if self.saturated else "",
        )


class SaturationReport(object):
    """The automated check of the task's own sentence, over every node."""

    def __init__(self, states, floor):
        self.states = list(states)
        self.floor = floor

    @property
    def ok(self):
        return not self.saturated

    @property
    def saturated(self):
        return [state for state in self.states if state.saturated]

    @property
    def rates(self):
        return np.array([state.rate for state in self.states])

    @property
    def sensitivities(self):
        return np.array([state.sensitivity for state in self.states])

    @property
    def min_sensitivity(self):
        return float(self.sensitivities.min()) if self.states else 1.0

    @property
    def worst(self):
        if not self.states:
            return None
        return min(self.states, key=lambda state: state.sensitivity)

    def summary(self, limit=12):
        """`limit` truncates the printed list only; `self.saturated` is whole."""
        lines = [
            "saturation check over %d node(s), floor=%g" % (len(self.states), self.floor),
            "  min sensitivity %.4g at %s"
            % (self.min_sensitivity, "-" if self.worst is None else self.worst.name),
        ]
        if self.saturated:
            ranked = sorted(self.saturated, key=lambda s: s.sensitivity)
            lines.append("  SATURATED (%d):" % len(ranked))
            for state in ranked[:limit]:
                lines.append(
                    "    %-20s rate=%.6g  sensitivity=%.3g  at the %s"
                    % (state.name, state.rate, state.sensitivity, state.rail)
                )
            if len(ranked) > limit:
                lines.append("    ... and %d more" % (len(ranked) - limit))
        else:
            lines.append("  no node below the floor")
        return "\n".join(lines)


def check_saturation(network, rates, drive=None, floor=SATURATION_FLOOR):
    """The check, at rates the caller already has."""
    r = np.asarray(rates, dtype=float)
    nets = net_inputs(network, r, drive)
    slopes = sigmoid_slopes(network, r)
    sens = sensitivity(r)
    states = [
        NodeState(
            name=node.name,
            rate=float(r[i]),
            net_input=float(nets[i]),
            slope=float(slopes[i]),
            sensitivity=float(sens[i]),
            floor=floor,
        )
        for i, node in enumerate(network.nodes)
    ]
    return SaturationReport(states, floor)


def default_initial_conditions(n, count=5, seed=0):
    """Spread, deterministic starting points for the multi-IC check.

    The three corners worth naming (all-silent, all-saturated, the target
    itself) plus seeded uniform draws. Deterministic so a failure is
    reproducible; spread so "it converges" is not a statement about one
    lucky basin.
    """
    if count < 3:
        raise ValueError("count must be at least 3 to include the named corners")
    conditions = [
        np.zeros(n),
        np.ones(n),
        np.full(n, MAX_SENSITIVITY_RATE),
    ]
    rng = np.random.default_rng(seed)
    for _ in range(count - 3):
        conditions.append(rng.random(n))
    return conditions


def steady_state(
    network,
    drive=None,
    initial_rates=None,
    t_max=DEFAULT_T_MAX,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
):
    """Integrate to `t_max` and return `(rates, residual)`.

    `residual` is `max |dr/dt|` at the endpoint - the number that says
    whether this is a fixed point at all, rather than a slow transient or
    a point on a limit cycle.
    """
    d = network.baseline_drive() if drive is None else np.asarray(drive, dtype=float)
    r0 = (
        np.full(network.n, MAX_SENSITIVITY_RATE)
        if initial_rates is None
        else np.asarray(initial_rates, dtype=float)
    )
    solution = solve_ivp(
        lambda t, r: network.rhs(r, d),
        (0.0, t_max),
        r0,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError("integration failed: %s" % solution.message)
    rates = solution.y[:, -1]
    residual = float(np.abs(network.rhs(rates, d)).max())
    return rates, residual


class StabilityAssessment(object):
    """Everything P3.4's three acceptance criteria ask about one network."""

    def __init__(
        self,
        network,
        drive,
        fixed_points,
        residuals,
        saturation,
        contraction,
        abscissa,
        floor,
        residual_tolerance,
        fixed_point_tolerance,
    ):
        self.network = network
        self.drive = drive
        self.fixed_points = np.asarray(fixed_points, dtype=float)
        self.residuals = np.asarray(residuals, dtype=float)
        self.saturation = saturation
        self.contraction = contraction
        self.abscissa = abscissa
        self.floor = floor
        self.residual_tolerance = residual_tolerance
        self.fixed_point_tolerance = fixed_point_tolerance

    @property
    def n_initial_conditions(self):
        return len(self.fixed_points)

    @property
    def rates(self):
        """The steady state reached from the first initial condition."""
        return self.fixed_points[0]

    @property
    def spread(self):
        """Largest disagreement between the endpoints of different runs."""
        if len(self.fixed_points) < 2:
            return 0.0
        return float(np.abs(self.fixed_points - self.fixed_points[0]).max())

    @property
    def converged(self):
        return bool(self.residuals.max() < self.residual_tolerance)

    @property
    def unique(self):
        return bool(self.spread < self.fixed_point_tolerance)

    @property
    def locally_stable(self):
        return bool(self.abscissa < 0.0)

    @property
    def globally_certified(self):
        return bool(self.contraction < 1.0)

    @property
    def ok(self):
        """Stable, single-valued and non-saturating. The gate."""
        return (
            self.converged
            and self.unique
            and self.locally_stable
            and self.saturation.ok
        )

    def summary(self):
        lines = [
            "stability assessment: %d nodes, %d initial condition(s)"
            % (self.network.n, self.n_initial_conditions),
            "  fixed point: residual max |dr/dt| = %.3g (tol %g) -> %s"
            % (
                self.residuals.max(),
                self.residual_tolerance,
                "converged" if self.converged else "NOT CONVERGED",
            ),
            "  same point from every start: spread = %.3g (tol %g) -> %s"
            % (
                self.spread,
                self.fixed_point_tolerance,
                "unique" if self.unique else "MULTIPLE ATTRACTORS",
            ),
            "  local: max Re eig(J) = %+.4g -> %s"
            % (
                self.abscissa,
                "stable" if self.locally_stable else "UNSTABLE",
            ),
            "  global: kappa = sigma_max(diag(g/4) W) = %.4g -> %s"
            % (
                self.contraction,
                "< 1, unique and globally attracting (Kelly 1990)"
                if self.globally_certified
                else ">= 1, no global certificate (sufficient, not necessary)",
            ),
            "  critical weight_scale (kappa = 1): %.4g"
            % critical_weight_scale(self.network),
        ]
        lines.extend("  " + line for line in self.saturation.summary().split("\n"))
        lines.append("  VERDICT: %s" % ("ok" if self.ok else "NOT ok"))
        return "\n".join(lines)


def assess(
    network,
    drive=None,
    initial_conditions=None,
    n_initial_conditions=5,
    seed=0,
    floor=SATURATION_FLOOR,
    t_max=DEFAULT_T_MAX,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    residual_tolerance=RESIDUAL_TOLERANCE,
    fixed_point_tolerance=FIXED_POINT_TOLERANCE,
):
    """Run the whole check on one network at one drive.

    `drive=None` means the drug-free baseline, `network.baseline_drive()`,
    which is the state the task's sentence ("every node must sit in the
    sensitive part of its sigmoid at its own steady state") is about. Pass
    a drive vector to ask the same question of a drugged state.

    Integrates from several initial conditions, then evaluates the
    saturation check and both stability criteria at the steady state the
    first one reached.
    """
    if initial_conditions is None:
        initial_conditions = default_initial_conditions(
            network.n, count=n_initial_conditions, seed=seed
        )
    fixed_points = []
    residuals = []
    for r0 in initial_conditions:
        rates, residual = steady_state(
            network, drive=drive, initial_rates=r0, t_max=t_max, rtol=rtol, atol=atol
        )
        fixed_points.append(rates)
        residuals.append(residual)
    rates = fixed_points[0]
    return StabilityAssessment(
        network=network,
        drive=network.baseline_drive() if drive is None else np.asarray(drive, float),
        fixed_points=fixed_points,
        residuals=residuals,
        saturation=check_saturation(network, rates, drive=drive, floor=floor),
        contraction=contraction_number(network),
        abscissa=spectral_abscissa(network, rates, drive=drive),
        floor=floor,
        residual_tolerance=residual_tolerance,
        fixed_point_tolerance=fixed_point_tolerance,
    )


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_OPERATING_GAIN",
    "DEFAULT_RTOL",
    "DEFAULT_T_MAX",
    "FIXED_POINT_TOLERANCE",
    "MAX_SENSITIVITY_RATE",
    "RESIDUAL_TOLERANCE",
    "SATURATION_FLOOR",
    "NodeState",
    "SaturationReport",
    "StabilityAssessment",
    "StabilizedNetwork",
    "assess",
    "balanced_baselines",
    "check_saturation",
    "contraction_number",
    "critical_weight_scale",
    "default_initial_conditions",
    "jacobian",
    "net_inputs",
    "operating_matrix",
    "rescale_weights",
    "sensitivity",
    "sigmoid_slopes",
    "spectral_abscissa",
    "stabilize",
    "steady_state",
    "weight_scale_for_operating_gain",
]
