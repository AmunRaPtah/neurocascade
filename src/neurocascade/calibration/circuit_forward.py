"""Forward model for the CIRCUIT layer (P2.2), the counterpart to
`forward.py`'s receptor layer.

`forward.py` answers "what fraction of receptor is bound at concentration
C?". This module answers the next question down the chain: "given that
occupancy, and given one circuit-layer gain, what does the behavioural
readout do?" - and, specifically, by what FACTOR does it move relative to
the same circuit with no drug on board.

Why a ratio, and not a readout value
------------------------------------

The three receptor-layer targets in `evidence.py` are directly comparable
with the model because occupancy is dimensionless in both the paper and the
code. No behavioural target is: `anxiety_index_readout()` returns a CeA
population firing rate in (0, 1), and the elevated plus maze returns seconds
on the open arms. There is no published constant converting one into the
other, and inventing one would be inventing a number.

What is comparable without any such constant is the drug's RELATIVE effect.
Write the (unknown, possibly nonlinear) behavioural scale as b = f(a), where
a is the model's anxiety index. If f is approximately linear through the
range the drug moves the animal over, then

    b_drug / b_vehicle  ~=  a_drug / a_vehicle

and every scale factor in f cancels. That proportionality is the one
assumption in the P2.2 fit with no citation behind it; it is declared here,
in `docs/architecture.md`, and in the `gaba_a_connection_modulation`
docstring, and P2.2 reports the fit under a second, independently defined
behavioural index as a sensitivity check on it (see
`behavioral_evidence.epm_composite_anxiety_index`).

Note what this does NOT assume: it does not assume the model's baseline
anxiety index equals the vehicle animals' anxiety, and it therefore does not
fit, use or bless any of the circuit's illustrative `baseline_input`, `tau`,
`gain` or connectivity values. Those set the ratio's SENSITIVITY to the
fitted gain, which is a real dependence and is reported as such, not hidden.

Steady state, and why the circuit is integrated rather than solved
-----------------------------------------------------------------

`CircuitNetwork.rhs` is the same function `engine.Simulation.rhs` calls, and
it is integrated here with `scipy.integrate.solve_ivp`, the integrator
`engine.py` uses, exactly as `forward.py` integrates the real
`ReceptorSite.occupancy_rhs` rather than substituting the closed-form
occupancy. Nothing about the circuit is reimplemented for calibration.

Receptor activation is held CONSTANT at its equilibrium value during that
integration rather than being co-integrated with a decaying plasma
concentration. That is a deliberate approximation and it is defensible here
for a measured reason rather than a convenient one: at the exposure the
[Graziani2022] dose produces, the benzodiazepine site is near saturation for
the whole plausible pre-test window, so the activation the circuit sees is
flat on the timescale over which the circuit relaxes (its slowest tau is
2 min against a >=30 min window). `tests/test_circuit_forward.py` checks
that against a full `engine.Simulation` run rather than asserting it.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse

#: Integration horizon for the circuit's approach to steady state, in units
#: of the network's slowest time constant. A Wilson-Cowan node relaxes
#: exponentially with rate 1/tau toward its sigmoid target, so 30 tau leaves
#: a residual of order exp(-30) ~ 1e-13, below the integrator's own
#: tolerances; the same reasoning `forward.equilibrium_occupancy` uses to
#: pick its horizon. Horizon and tolerances both matter here because this
#: runs once per MCMC step. `tests/test_circuit_forward.py` checks the
#: result against a 10x longer, 100x tighter integration instead of taking
#: this comment's word for it.
_STEADY_STATE_TAU_MULTIPLES = 30.0
_STEADY_STATE_RTOL = 1e-8
_STEADY_STATE_ATOL = 1e-11


def circuit_jacobian(network, rates, drive=None, connectivity=None):
    """Return the exact circuit RHS Jacobian as a CSR sparse matrix.

    For ``tau_i * dr_i/dt = -r_i + S_i(W_i r + drive_i)``, the Jacobian is

    ``diag(1 / tau) @ (-I + diag(S'(net)) @ W)``.

    The derivative follows :func:`neurocascade.circuit.sigmoid` exactly,
    including its finite clipping range.  This is the Jacobian supplied to
    stiff ``solve_ivp`` methods; keeping it here beside the forward path
    makes the differentiated equations and the integrated equations share
    the same connectivity override.
    """
    rates = np.asarray(rates, dtype=float)
    d = network.baseline_drive() if drive is None else np.asarray(drive, dtype=float)
    W = network.connectivity if connectivity is None else connectivity
    W = W.tocsr() if sparse.issparse(W) else sparse.csr_matrix(W)

    gains = np.array([node.gain for node in network.nodes], dtype=float)
    thresholds = np.array([node.threshold for node in network.nodes], dtype=float)
    taus = np.array([node.tau for node in network.nodes], dtype=float)
    net_input = np.asarray(W @ rates + d, dtype=float)
    z = gains * (net_input - thresholds)
    clipped = np.clip(z, -500.0, 500.0)
    outputs = 1.0 / (1.0 + np.exp(-clipped))
    slopes = gains * outputs * (1.0 - outputs)
    # sigmoid() is constant beyond the clipping rails, so its derivative is
    # zero there rather than the tiny nonzero value of the unclipped formula.
    slopes[np.abs(z) > 500.0] = 0.0

    return sparse.diags(1.0 / taus) @ (
        -sparse.eye(network.n, format="csr") + sparse.diags(slopes) @ W
    )


def _jacobian_sparsity(connectivity):
    """Return the implicit solver's structural Jacobian sparsity pattern."""
    W = connectivity.tocsr() if sparse.issparse(connectivity) else sparse.csr_matrix(connectivity)
    pattern = (W != 0).astype(float) + sparse.eye(W.shape[0], format="csr")
    pattern.data[:] = 1.0
    return pattern.tocsr()


def modulated_connectivity(network, connection_modulation, activation):
    """The network's connectivity with each modulated connection scaled by
    `(1 + gain * activation)`.

    This reproduces, for a single fixed activation, what
    `engine.Simulation.rhs` does per timestep with its own
    `connection_modulation` dict, including the sign convention: a positive
    gain on an inhibitory (negative) weight makes it MORE negative, i.e.
    strengthens the inhibition, which is what a benzodiazepine does to a
    GABAergic synapse.
    """
    W = np.array(network.connectivity, dtype=float, copy=True)
    for (post_name, pre_name), gain in connection_modulation.items():
        i = network.index[post_name]
        j = network.index[pre_name]
        W[i, j] = W[i, j] * (1.0 + gain * activation)
    return W


def steady_state_rates(network, connection_modulation=None, activation=0.0,
                       coupling=None, initial_rates=None, rtol=_STEADY_STATE_RTOL,
                       atol=_STEADY_STATE_ATOL, tau_multiples=_STEADY_STATE_TAU_MULTIPLES,
                       method="RK45", use_sparse_jacobian=True):
    """Integrate `network.rhs` to steady state and return the final rates.

    `coupling` is the node_name -> gain dict of a direct drive coupling
    (`engine.Simulation`'s `coupling`), added to the baseline drive scaled by
    `activation`, so this function covers both of the two ways a receptor
    system can act on a circuit in this codebase, not just the one the
    GABA-A scenario uses.
    """
    drive = network.baseline_drive().astype(float).copy()
    for node_name, gain in (coupling or {}).items():
        drive[network.index[node_name]] += gain * activation
    W = modulated_connectivity(network, connection_modulation or {}, activation)

    y0 = (
        np.full(network.n, 0.5, dtype=float)
        if initial_rates is None
        else np.asarray(initial_rates, dtype=float)
    )
    t_end = tau_multiples * max(node.tau for node in network.nodes)

    def rhs(_t, rates):
        return network.rhs(rates, drive, connectivity=W)

    solver_kwargs = {
        "method": method,
        "rtol": rtol,
        "atol": atol,
    }
    if use_sparse_jacobian and method in ("BDF", "Radau"):
        solver_kwargs["jac"] = lambda _t, rates: circuit_jacobian(
            network, rates, drive=drive, connectivity=W
        )
        solver_kwargs["jac_sparsity"] = _jacobian_sparsity(W)

    sol = solve_ivp(rhs, (0.0, t_end), y0, **solver_kwargs)
    if not sol.success:
        raise RuntimeError("circuit steady-state integration failed: %s" % sol.message)
    return sol.y[:, -1]


def readout_ratio(network, readout, connection_modulation=None, activation=0.0,
                  coupling=None, baseline_rates=None):
    """`readout(drug) / readout(no drug)` at steady state.

    The quantity P2.2's likelihood compares against a published
    drug/vehicle ratio. Both legs are integrated with the same network,
    baseline drive and readout, so everything common to them cancels
    exactly, which is the whole point of using a ratio.

    `baseline_rates` may be passed in when the caller knows the drug-free
    leg does not depend on the parameter being sampled - which it never
    does, since every modulation factor is `1 + gain * 0` and every
    coupling contribution is `gain * 0` at zero activation. That makes it a
    pure caching argument, and
    `tests/test_circuit_forward.py::test_passing_the_baseline_in_changes_
    nothing` pins that it is only that.
    """
    if baseline_rates is None:
        baseline_rates = steady_state_rates(network, connection_modulation, 0.0, coupling)
    drug_rates = steady_state_rates(
        network, connection_modulation, activation, coupling,
        initial_rates=baseline_rates,
    )
    baseline_score = readout.score(network, baseline_rates)
    if baseline_score == 0.0:
        raise ValueError(
            "%s scores zero with no drug on board, so no drug/vehicle ratio "
            "is defined for it" % (readout.name,)
        )
    return readout.score(network, drug_rates) / baseline_score
